"""
Quick checks of the parts most likely to break. Run from the project folder:

    python -m unittest discover tests

They use the built database (data/atrisk.db); run build_database.py first.
Nothing is written to it except the summary cache.
"""

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import shutil  # noqa: E402
import tempfile  # noqa: E402

# Work on a copy of the database, so the tests never change the real one.
_SRC = os.path.join(ROOT, "data", "atrisk.db")
HAVE_DB = os.path.exists(_SRC)
if HAVE_DB:
    _TMP = os.path.join(tempfile.mkdtemp(), "test.db")
    shutil.copyfile(_SRC, _TMP)
    os.environ["ATRISK_DB_PATH"] = _TMP
    os.environ["ATRISK_STAGING_DIR"] = os.path.dirname(_TMP)

import config  # noqa: E402

COURSE = {}


def setUpModule():
    """Adds a course the way an admin does: create it with thresholds, then
    upload this semester's marks through the normal pipeline."""
    if not HAVE_DB:
        return
    import database as db
    import ingest
    db.init_db()
    cid = db.create_course("TST101", "Test course", "Semester 2", "Laucala", risk_threshold=0.4, borderline_threshold=0.2)
    path = os.path.join(ROOT, "static", "examples", "IS413_semester_marks.csv")
    with open(path, "rb") as f:
        token, err = ingest.stage("marks.csv", f.read(), cid, "current", "Semester 2", "admin")
    assert not err, err
    result, err = ingest.commit(token, ingest.load_staged(token), "admin")
    assert not err, err
    COURSE["id"] = cid


@unittest.skipUnless(HAVE_DB, "Run build_database.py first")
class TestStoredModel(unittest.TestCase):
    def test_model_loads_from_database(self):
        import database as db
        import ml_engine
        v = db.get_active_model()
        self.assertIsNotNone(v, "no active model")
        self.assertIsNone(v.get("model_path"), "model should be stored in the database, not a file")
        bundle = ml_engine.load_bundle(v)
        self.assertIsNotNone(bundle, ml_engine.load_errors.get(v["id"]))
        self.assertEqual(bundle["algorithm"], "forest")

    def test_new_course_uses_random_forest(self):
        import analytics
        s = analytics.cached_summary(COURSE["id"])
        self.assertTrue(s["trained"], "a new course without history should still be scored by the random forest")
        self.assertEqual(s["n"], 120)

    def test_summary_is_cached(self):
        import analytics
        first = analytics.cached_summary(COURSE["id"])
        analytics._summaries.clear()          # force a read from the database
        second = analytics.cached_summary(COURSE["id"])
        self.assertEqual(first["counts"], second["counts"])

    def test_two_checks(self):
        import analytics
        for r in analytics.cached_summary(COURSE["id"])["rows"]:
            a = r["checks"]["arithmetic"]
            if not r["feasible"]:
                self.assertEqual(a["verdict"], "Cannot pass")
                self.assertEqual(r["status"], "At Risk")
                self.assertEqual(r["p_fail"], 1.0)
            if a["verdict"] == "Still open":
                self.assertTrue(0 < r["p_fail"] < 1)

    def test_threshold_change_updates_results(self):
        import analytics
        import database as db
        before = analytics.cached_summary(COURSE["id"])["counts"]["At Risk"]
        db.update_course(COURSE["id"], risk_threshold=0.8, borderline_threshold=0.5)
        after = analytics.cached_summary(COURSE["id"])["counts"]["At Risk"]
        db.update_course(COURSE["id"], risk_threshold=0.4, borderline_threshold=0.2)
        self.assertLess(after, before)


@unittest.skipUnless(HAVE_DB, "Run build_database.py first")
class TestAssistant(unittest.TestCase):
    def setUp(self):
        import analytics
        import database as db
        self.course = db.get_course(COURSE["id"])
        self.summary = analytics.cached_summary(COURSE["id"])
        self.version = db.get_active_model()

    def ask(self, q, last=None):
        import assistant
        return assistant.answer(q, self.course, self.summary, self.version, last)

    def test_class_questions(self):
        for q in ("who should I contact first", "give me a class summary", "how many are at risk",
                  "which assessment is hardest", "who missed submissions", "how accurate is it"):
            r = self.ask(q)
            self.assertNotIn("did not understand", r["text"], q)

    def test_student_follow_up_uses_memory(self):
        sid = next(r["student_id"] for r in self.summary["rows"] if r["status"] == "At Risk")
        first = self.ask(f"why is {sid} at risk")
        self.assertEqual(first["student"], sid)
        follow = self.ask("what does she need to pass", last=sid)
        self.assertIn(sid, follow["text"])

    def test_what_if(self):
        s = next(r for r in self.summary["rows"] if not r["complete"] and r["status"] != "No data")
        r = self.ask(f"what if {s['student_id']} gets 100%")
        self.assertIn(f"{s['best_possible']:.0f}%", r["text"])


class TestOutlines(unittest.TestCase):
    """Course outlines in different formats and layouts."""
    EXPECT = {
        "IS413_Course_Outline.pdf": ("IS413", 6, "table"),
        "CS111_outline.docx": ("CS111", 5, "table"),
        "MA102_outline.xlsx": ("MA102", 5, "table"),
        "BIO150_outline.pdf": ("BIO150", 4, "text"),
    }

    def test_outlines(self):
        import outline
        for name, (code, n, method) in self.EXPECT.items():
            with open(os.path.join(ROOT, "static", "examples", name), "rb") as f:
                r = outline.parse(name, f.read())
            self.assertEqual(r["code"], code, name)
            self.assertEqual(len(r["assessments"]), n, name)
            self.assertAlmostEqual(sum(a["weight"] for a in r["assessments"]), 100, places=1, msg=name)
            self.assertEqual(r["method"], method, name)
            self.assertTrue(r["name"] and not r["name"].startswith(("-", "\u2013")), name)

    def test_bad_file(self):
        import outline
        with self.assertRaises(ValueError):
            outline.parse("notes.txt", b"hello")


class TestMapping(unittest.TestCase):
    def test_complete_file_ignores_old_structure(self):
        import schema_mapping as sm
        staged = {"columns": ["student_id", "Test 1 (40%)", "Final Exam (60%)"],
                  "data": [["S1", "50", ""], ["S2", "70", ""]], "kind": "current", "row_numbers": [2, 3]}
        profile = {"layout": "wide", "assessments": {"oldassignment": {"name": "Old assignment", "weight": 30}}}
        mapping = sm.propose(staged, profile)
        names = [a["name"] for a in mapping["assessments"]]
        self.assertEqual(names, ["Test 1", "Final Exam"])


if __name__ == "__main__":
    unittest.main()
