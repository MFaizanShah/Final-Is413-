"""
build_database.py
Builds data/atrisk.db from the real Open University data in data/oulad/,
trains the model ONCE and stores it inside the database, and precomputes
every course dashboard. Run it once; after that the app just loads the
database and never trains on start.

    python build_database.py            # build (keeps an existing database)
    python build_database.py --fresh    # delete data/atrisk.db and rebuild
    python build_database.py --retrain  # keep the data, train a new model version

Every file goes through the same upload pipeline an administrator uses
(mapping, validation, pseudonymisation), so the result is exactly what an
admin would get by uploading these files by hand.
"""

import glob
import os
import sys
import time

import numpy as np

import config

if "--fresh" in sys.argv and os.path.exists(config.DB_PATH):
    os.remove(config.DB_PATH)

import analytics  # noqa: E402
import database as db  # noqa: E402
import ingest  # noqa: E402
import ml_engine  # noqa: E402
import pseudo  # noqa: E402
import schema_mapping as sm  # noqa: E402

ADMIN = "admin"
SRC = config.REAL_DATA_DIR
COURSES = {
    "AAA": "Social Sciences module AAA",
    "BBB": "Social Sciences module BBB",
    "CCC": "STEM module CCC",
    "DDD": "STEM module DDD",
    "EEE": "STEM module EEE",
    "FFF": "STEM module FFF",
}
SEMESTER = "2014J (October 2014)"
CAMPUS = "Open University (UK), real anonymised data"
DEMO_LECTURER = ("lecturer", "Demo Lecturer", "lecturer123", [])


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def upload(course_id, kind, path, label):
    with open(path, "rb") as f:
        token, err = ingest.stage(os.path.basename(path), f.read(), course_id, kind, label, ADMIN)
    if err:
        raise SystemExit(f"{path}: {err}")
    staged = ingest.load_staged(token)
    result, err = ingest.commit(token, staged, ADMIN, retrain=False)
    if err:
        problems = sm.build(staged, staged["mapping"])["errors"]
        ingest.discard(token)
        raise SystemExit(f"{path}: {err} {problems[:3]}")
    return result


def load_courses():
    """The Open University modules become the TRAINING LIBRARY: past cohorts
    with known results that the random forest learns from. They are hidden
    from the course lists, so the dashboard only shows courses you add."""
    for code, name in COURSES.items():
        course = next((c for c in db.list_courses(training=True) if c["code"] == code), None)
        if course and db.course_history_stats(code)["n"]:
            log(f"{code} already in the training library")
            continue
        cid = course["id"] if course else db.create_course(code, f"Open University {name}", "2013 to 2014", CAMPUS,
                                                           is_training=True)
        if not course:
            db.audit(ADMIN, "course_created", f"{code} added to the training library (Open University data)", course_id=cid)
        for path in sorted(glob.glob(os.path.join(SRC, "history", f"{code}_*.csv"))):
            pres = os.path.basename(path)[4:-4]
            r = upload(cid, "historical", path, pres)
            log(f"{code} history {pres}: {r['students']} students, {r['assessments']} assessments")


def add_lecturer():
    """A demo lecturer account. It sees no courses until an admin adds a
    course and ticks this lecturer."""
    username, full, pw, _ = DEMO_LECTURER
    if not any(u["username"] == username for u in db.list_users()):
        db.create_user(username, full, pw, "lecturer")


def train():
    if db.get_active_model() and "--retrain" not in sys.argv:
        log(f"Model version {db.get_active_model()['id']} is already active; skipping training "
            "(delete data/atrisk.db or use --fresh to retrain).")
        return
    ok, msg = ml_engine.readiness()
    if not ok:
        raise SystemExit(msg)
    model_id = db.create_model_version(ADMIN, "Initial build from Open University data")
    log(f"Training model version {model_id}: {msg}")
    ml_engine._train(model_id, ADMIN, progress=log)


def check_against_reality():
    """The 2014J presentation really happened, so the predictions made from
    its first half can be compared with how each student actually finished."""
    from sklearn.metrics import roc_auc_score
    per_course, all_y, all_p, all_flag = [], [], [], []
    withdrawn_flagged = withdrawn_total = 0
    for code in COURSES:
        actual_path = os.path.join(SRC, "actual_outcomes", f"{code}_2014J.csv")
        current_path = os.path.join(SRC, "current", f"{code}_2014J.csv")
        if not os.path.exists(actual_path):
            continue
        # A temporary, hidden course holds the 2014J marks while they are
        # scored exactly as an uploaded course would be. It is then removed.
        course = {"id": db.create_course(code, "Reality check", SEMESTER, CAMPUS, is_training=True)}
        upload(course["id"], "current", current_path, SEMESTER)
        actual = {}
        with open(actual_path) as f:
            next(f)
            for line in f:
                sid, res = line.strip().split(",")
                actual[pseudo.pseudonymise(code, sid)] = res
        summary = analytics.cached_summary(course["id"])
        y, p, flag = [], [], []
        for r in summary["rows"]:
            res = actual.get(r["student_id"])
            if res is None or r["p_fail"] is None:
                continue
            flagged = r["status"] == "At Risk"
            if res == "Withdrawn":
                withdrawn_total += 1
                withdrawn_flagged += flagged
                continue
            y.append(1 if res == "Fail" else 0)
            p.append(r["p_fail"])
            flag.append(flagged)
        db.delete_course(course["id"])
        y, p, flag = np.array(y), np.array(p), np.array(flag)
        caught = int((flag & (y == 1)).sum())
        per_course.append({"course": code, "students": int(len(y)), "failed": int(y.sum()),
                           "flagged": int(flag.sum()), "caught": caught,
                           "recall": round(caught / max(y.sum(), 1), 3),
                           "precision": round(caught / max(flag.sum(), 1), 3),
                           "auc": round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None})
        all_y += list(y)
        all_p += list(p)
        all_flag += list(flag)
    y, p, flag = np.array(all_y), np.array(all_p), np.array(all_flag)
    caught = int((flag & (y == 1)).sum())
    overall = {"students": int(len(y)), "failed": int(y.sum()), "flagged": int(flag.sum()), "caught": caught,
               "recall": round(caught / max(y.sum(), 1), 3), "precision": round(caught / max(flag.sum(), 1), 3),
               "auc": round(float(roc_auc_score(y, p)), 3),
               "brier": round(float(np.mean((p - y) ** 2)), 4),
               "accuracy": round(float(np.mean(flag == (y == 1))), 3),
               "withdrawn": withdrawn_total, "withdrawn_flagged": int(withdrawn_flagged)}
    result = {"cohort": "2014J", "overall": overall, "courses": per_course,
              "note": "Predictions made with about half of the coursework marked, compared with each student's real "
                      "final result. Students who later withdrew are reported separately."}
    db.set_meta("reality_check", result)
    log(f"Reality check on 2014J: AUC {overall['auc']}, caught {caught} of {int(y.sum())} students who failed "
        f"({overall['recall']:.0%}); {overall['precision']:.0%} of flagged students failed.")


def main():
    if not os.path.isdir(SRC):
        raise SystemExit(f"{SRC} not found. Run tools/prepare_oulad.py first.")
    db.init_db()
    load_courses()
    add_lecturer()
    train()
    check_against_reality()
    log("Precomputing dashboards")
    for c in db.list_courses(training=False):
        analytics.cached_summary(c["id"])
    c = db.conn()
    c.execute("VACUUM")
    c.close()
    log(f"Done: {config.DB_PATH} ({os.path.getsize(config.DB_PATH) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
