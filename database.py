"""
database.py
SQLite storage. SQLite ships with Python, so there is nothing to install.

Data flow
---------
Admins upload datasets. Each upload becomes a versioned row in `datasets`
with its rows in `dataset_rows` (pseudonymised IDs and marks only). One
version per slot (course, kind, cohort label) is active at a time, so a
dataset can be replaced or rolled back.

The active CURRENT dataset of a course is materialised into `assessments`,
`students` and `marks`, which the dashboards read. Active HISTORICAL
datasets from every course are pooled to train the global model, whose
versions live in `model_versions`. Every change is written to `audit_log`.
"""

import json
import os
import re
import shutil
import sqlite3
import zlib
from datetime import datetime, timedelta
from werkzeug.security import generate_password_hash, check_password_hash

from config import DB_PATH, BUNDLED_DB_PATH, LEGACY_DB_PATHS, IS_VERCEL, detect_type


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def conn():
    c = sqlite3.connect(DB_PATH, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys = ON")
    return c


def _prepare_file():
    """On Vercel the bundled database is read only, so a working copy is
    made in /tmp. Locally the database lives in data/atrisk.db."""
    folder = os.path.dirname(DB_PATH)
    if folder:
        os.makedirs(folder, exist_ok=True)
    if IS_VERCEL and not os.path.exists(DB_PATH) and os.path.exists(BUNDLED_DB_PATH):
        shutil.copyfile(BUNDLED_DB_PATH, DB_PATH)


def _add_column(c, table, column, decl):
    cols = {r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db():
    _prepare_file()
    c = conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        full_name TEXT,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'lecturer'
    );
    CREATE TABLE IF NOT EXISTS courses (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        code TEXT NOT NULL,
        name TEXT NOT NULL,
        semester TEXT,
        campus TEXT,
        risk_threshold REAL DEFAULT 0.40,
        borderline_threshold REAL DEFAULT 0.20,
        created_at TEXT
    );
    CREATE TABLE IF NOT EXISTS course_lecturers (
        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
        PRIMARY KEY (course_id, user_id)
    );
    CREATE TABLE IF NOT EXISTS assessments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
        name TEXT NOT NULL,
        weight REAL NOT NULL,
        atype TEXT,
        max_mark REAL DEFAULT 100,
        due_date TEXT,
        reported_class_avg REAL,
        sort_order INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS students (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
        student_id TEXT NOT NULL,
        UNIQUE(course_id, student_id)
    );
    CREATE TABLE IF NOT EXISTS marks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
        student_id TEXT NOT NULL,
        assessment_id INTEGER NOT NULL REFERENCES assessments(id) ON DELETE CASCADE,
        mark REAL NOT NULL,
        UNIQUE(course_id, student_id, assessment_id)
    );
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        course_id INTEGER NOT NULL REFERENCES courses(id) ON DELETE CASCADE,
        kind TEXT NOT NULL,              -- 'current' or 'historical'
        label TEXT NOT NULL,             -- semester / cohort label (the slot)
        version INTEGER NOT NULL,
        filename TEXT,
        uploaded_by TEXT,
        uploaded_at TEXT,
        is_active INTEGER DEFAULT 0,
        n_rows INTEGER,
        n_with_outcome INTEGER,
        structure_json TEXT,             -- assessments: name, weight, atype, max_mark, due_date, reported_class_avg
        mapping_json TEXT,               -- how file columns were mapped
        validation_json TEXT             -- warnings accepted at upload
    );
    CREATE TABLE IF NOT EXISTS dataset_rows (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
        pseudo_id TEXT NOT NULL,
        cohort TEXT,
        marks_json TEXT NOT NULL,        -- list aligned with structure, percent or null
        final_grade TEXT,
        final_mark REAL
    );
    CREATE INDEX IF NOT EXISTS ix_rows_dataset ON dataset_rows(dataset_id);
    CREATE TABLE IF NOT EXISTS mapping_profiles (
        course_code TEXT NOT NULL,
        kind TEXT NOT NULL,
        profile_json TEXT NOT NULL,
        updated_at TEXT,
        updated_by TEXT,
        PRIMARY KEY (course_code, kind)
    );
    CREATE TABLE IF NOT EXISTS model_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        created_by TEXT,
        trigger TEXT,
        status TEXT,                     -- training, ready, failed
        error TEXT,
        algorithm TEXT,
        n_students INTEGER,
        n_rows INTEGER,
        n_courses INTEGER,
        n_cohorts INTEGER,
        metrics_json TEXT,
        summary_json TEXT,
        model_path TEXT,
        is_active INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        at TEXT,
        username TEXT,
        action TEXT,
        course_id INTEGER,
        dataset_id INTEGER,
        model_id INTEGER,
        detail TEXT
    );
    CREATE TABLE IF NOT EXISTS app_meta (
        key TEXT PRIMARY KEY,
        value_json TEXT
    );
    -- Computed dashboards, so a page does not recompute every prediction on
    -- each visit. Rebuilt automatically when the data, model or settings change.
    CREATE TABLE IF NOT EXISTS summary_cache (
        course_id INTEGER PRIMARY KEY REFERENCES courses(id) ON DELETE CASCADE,
        cache_key TEXT NOT NULL,
        created_at TEXT,
        payload BLOB NOT NULL
    );
    """)
    # The trained model itself is stored in the database (model_blob), so the
    # database file is all that is needed to run the app on another computer.
    _add_column(c, "model_versions", "model_blob", "BLOB")
    # Training library courses hold past results only. They teach the model
    # but are not shown as teaching courses to lecturers.
    _add_column(c, "courses", "is_training", "INTEGER DEFAULT 0")
    # The assessment structure read from the course outline when the course
    # was created: shown before marks arrive and used for the marks template.
    _add_column(c, "courses", "outline_json", "TEXT")
    _add_column(c, "model_versions", "blob_bytes", "INTEGER")
    # The student ID as uploaded, shown to the course's lecturers. Kept for
    # this semester's marks only; past results stay pseudonymous.
    _add_column(c, "dataset_rows", "display_id", "TEXT")
    _add_column(c, "students", "display_id", "TEXT")
    if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
        copied = _copy_legacy_users(c)
        if not copied:
            c.execute("INSERT INTO users (username, full_name, password_hash, role) VALUES (?,?,?,?)",
                      ("admin", "System Administrator", generate_password_hash("admin123"), "admin"))
    # On a hosted copy, set ATRISK_ADMIN_PASSWORD so the default password
    # in the bundled database is never usable.
    if os.environ.get("ATRISK_ADMIN_PASSWORD"):
        c.execute("UPDATE users SET password_hash=? WHERE username='admin'",
                  (generate_password_hash(os.environ["ATRISK_ADMIN_PASSWORD"]),))
    c.commit()
    c.close()


def _copy_legacy_users(c):
    """Carry user accounts over from an older database, if present."""
    for path in LEGACY_DB_PATHS:
        if not os.path.exists(path) or os.path.abspath(path) == os.path.abspath(DB_PATH):
            continue
        try:
            old = sqlite3.connect(path)
            rows = old.execute("SELECT username, full_name, password_hash, role FROM users").fetchall()
            old.close()
        except sqlite3.Error:
            continue
        for r in rows:
            c.execute("INSERT OR IGNORE INTO users (username, full_name, password_hash, role) VALUES (?,?,?,?)", r)
        if rows:
            return True
    return False


def _rows(sql, args=()):
    c = conn()
    rows = c.execute(sql, args).fetchall()
    c.close()
    return [dict(r) for r in rows]


def _row(sql, args=()):
    rows = _rows(sql, args)
    return rows[0] if rows else None


# ---------- audit ----------

def audit(username, action, detail="", course_id=None, dataset_id=None, model_id=None):
    c = conn()
    c.execute("INSERT INTO audit_log (at, username, action, course_id, dataset_id, model_id, detail) VALUES (?,?,?,?,?,?,?)",
              (now(), username, action, course_id, dataset_id, model_id, detail))
    c.commit()
    c.close()


def list_audit(limit=200, course_id=None):
    if course_id:
        return _rows("""SELECT a.*, c.code AS course_code FROM audit_log a LEFT JOIN courses c ON c.id=a.course_id
                        WHERE a.course_id=? ORDER BY a.id DESC LIMIT ?""", (course_id, limit))
    return _rows("""SELECT a.*, c.code AS course_code FROM audit_log a LEFT JOIN courses c ON c.id=a.course_id
                    ORDER BY a.id DESC LIMIT ?""", (limit,))


# ---------- users ----------

def get_user(user_id):
    return _row("SELECT * FROM users WHERE id=?", (user_id,))


def verify_user(username, password):
    r = _row("SELECT * FROM users WHERE username=?", (username.strip(),))
    if r and check_password_hash(r["password_hash"], password):
        return r
    return None


def list_users():
    return _rows("SELECT id, username, full_name, role FROM users ORDER BY username")


def create_user(username, full_name, password, role):
    c = conn()
    try:
        cur = c.execute("INSERT INTO users (username, full_name, password_hash, role) VALUES (?,?,?,?)",
                        (username.strip(), full_name.strip(), generate_password_hash(password), role))
        c.commit()
        return cur.lastrowid
    except sqlite3.IntegrityError:
        return None
    finally:
        c.close()


def set_password(user_id, password):
    c = conn()
    c.execute("UPDATE users SET password_hash=? WHERE id=?", (generate_password_hash(password), user_id))
    c.commit()
    c.close()


def delete_user(user_id):
    c = conn()
    c.execute("DELETE FROM users WHERE id=? AND role != 'admin'", (user_id,))
    c.commit()
    c.close()


# ---------- courses ----------

def create_course(code, name, semester, campus, risk_threshold=0.40, borderline_threshold=0.20, is_training=False):
    c = conn()
    cur = c.execute("""INSERT INTO courses (code, name, semester, campus, created_at, risk_threshold, borderline_threshold,
                       is_training) VALUES (?,?,?,?,?,?,?,?)""",
                    (code.strip().upper(), name.strip(), (semester or "").strip(), (campus or "").strip(), now(),
                     risk_threshold, borderline_threshold, 1 if is_training else 0))
    c.commit()
    course_id = cur.lastrowid
    c.close()
    return course_id


def list_courses(training=None):
    """training=None: every course; False: teaching courses; True: the training library."""
    if training is None:
        return _rows("SELECT * FROM courses ORDER BY code, id")
    return _rows("SELECT * FROM courses WHERE COALESCE(is_training, 0)=? ORDER BY code, id", (1 if training else 0,))


def courses_for_user(user):
    """Teaching courses: admins see every one; lecturers only those assigned to them."""
    if user["role"] == "admin":
        return list_courses(training=False)
    return _rows("""SELECT c.* FROM courses c JOIN course_lecturers l ON l.course_id=c.id
                    WHERE l.user_id=? AND COALESCE(c.is_training, 0)=0 ORDER BY c.code, c.id""", (user["id"],))


def user_can_view(user, course_id):
    if user["role"] == "admin":
        return True
    return _row("SELECT 1 AS ok FROM course_lecturers WHERE course_id=? AND user_id=?", (course_id, user["id"])) is not None


def get_course(course_id):
    return _row("SELECT * FROM courses WHERE id=?", (course_id,))


def find_course_by_code(code):
    return _row("SELECT * FROM courses WHERE code=? ORDER BY id DESC LIMIT 1", (code.strip().upper(),))


# ---------- named training sets ----------
# A training set is a file (or several cohorts) of past results an admin adds
# under a name. It is stored as a training library course whose code starts
# with TRAINING_PREFIX, which no real course code uses, so it can never be
# mistaken for a teaching course.

TRAINING_PREFIX = "TRAIN-"


def find_training_set(name):
    return _row("""SELECT * FROM courses WHERE COALESCE(is_training, 0)=1 AND code LIKE ? AND LOWER(name)=LOWER(?)
                   ORDER BY id LIMIT 1""", (TRAINING_PREFIX + "%", name.strip()))


def create_training_set(name):
    slug = re.sub(r"[^A-Z0-9]+", "-", name.upper()).strip("-")[:30].strip("-") or "SET"
    code, n = TRAINING_PREFIX + slug, 1
    while _row("SELECT 1 AS x FROM courses WHERE code=?", (code,)):
        n += 1
        code = f"{TRAINING_PREFIX}{slug}-{n}"
    return get_course(create_course(code, name.strip(), "Training set", "", is_training=True))


def course_has_datasets(course_id):
    return _row("SELECT 1 AS x FROM datasets WHERE course_id=? LIMIT 1", (course_id,)) is not None


def purge_empty_training_sets(max_age_seconds=24 * 3600):
    """Removes training sets that never had a file saved, once they are older
    than max_age_seconds. Returns how many were removed."""
    cutoff = (datetime.now() - timedelta(seconds=max_age_seconds)).strftime("%Y-%m-%d %H:%M:%S")
    c = conn()
    try:
        cur = c.execute("""DELETE FROM courses WHERE COALESCE(is_training, 0)=1 AND code LIKE ? AND created_at < ?
                           AND NOT EXISTS (SELECT 1 FROM datasets d WHERE d.course_id=courses.id)""",
                        (TRAINING_PREFIX + "%", cutoff))
        c.commit()
        return cur.rowcount
    except sqlite3.OperationalError:
        return 0  # read only database
    finally:
        c.close()


def update_course(course_id, **fields):
    allowed = {"code", "name", "semester", "campus", "risk_threshold", "borderline_threshold"}
    sets = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not sets:
        return
    c = conn()
    c.execute(f"UPDATE courses SET {', '.join(k + '=?' for k in sets)} WHERE id=?", (*sets.values(), course_id))
    c.commit()
    c.close()


def delete_course(course_id):
    c = conn()
    c.execute("DELETE FROM courses WHERE id=?", (course_id,))
    c.commit()
    c.close()


def get_lecturers(course_id):
    return _rows("""SELECT u.id, u.username, u.full_name FROM users u JOIN course_lecturers l ON l.user_id=u.id
                    WHERE l.course_id=? ORDER BY u.username""", (course_id,))


def set_lecturers(course_id, user_ids):
    c = conn()
    c.execute("DELETE FROM course_lecturers WHERE course_id=?", (course_id,))
    for uid in user_ids:
        c.execute("INSERT OR IGNORE INTO course_lecturers (course_id, user_id) VALUES (?,?)", (course_id, uid))
    c.commit()
    c.close()


# ---------- current structure, students and marks ----------

def get_assessments(course_id):
    return _rows("SELECT * FROM assessments WHERE course_id=? ORDER BY sort_order, id", (course_id,))


def update_assessment_types(course_id, types_by_id):
    c = conn()
    for aid, t in types_by_id.items():
        c.execute("UPDATE assessments SET atype=? WHERE id=? AND course_id=?", (t, aid, course_id))
    c.commit()
    c.close()


def get_students(course_id):
    return _rows("SELECT * FROM students WHERE course_id=? ORDER BY COALESCE(display_id, student_id)", (course_id,))


def get_marks(course_id):
    """Returns {student_id: {assessment_id: mark percent}}"""
    out = {}
    for r in _rows("SELECT student_id, assessment_id, mark FROM marks WHERE course_id=?", (course_id,)):
        out.setdefault(r["student_id"], {})[r["assessment_id"]] = r["mark"]
    return out


def _materialise_current(c, course_id, dataset_id):
    """Rebuild assessments, students and marks from a current dataset."""
    c.execute("DELETE FROM marks WHERE course_id=?", (course_id,))
    c.execute("DELETE FROM students WHERE course_id=?", (course_id,))
    c.execute("DELETE FROM assessments WHERE course_id=?", (course_id,))
    if dataset_id is None:
        return
    ds = c.execute("SELECT structure_json FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    structure = json.loads(ds["structure_json"])
    ids = []
    for i, a in enumerate(structure):
        cur = c.execute("""INSERT INTO assessments (course_id, name, weight, atype, max_mark, due_date, reported_class_avg, sort_order)
                           VALUES (?,?,?,?,?,?,?,?)""",
                        (course_id, a["name"], float(a["weight"]), a.get("atype") or detect_type(a["name"]),
                         a.get("max_mark") or 100, a.get("due_date"), a.get("reported_class_avg"), i))
        ids.append(cur.lastrowid)
    for r in c.execute("SELECT pseudo_id, display_id, marks_json FROM dataset_rows WHERE dataset_id=?", (dataset_id,)).fetchall():
        c.execute("INSERT OR IGNORE INTO students (course_id, student_id, display_id) VALUES (?,?,?)",
                  (course_id, r["pseudo_id"], r["display_id"]))
        for aid, m in zip(ids, json.loads(r["marks_json"])):
            if m is not None:
                c.execute("INSERT OR REPLACE INTO marks (course_id, student_id, assessment_id, mark) VALUES (?,?,?,?)",
                          (course_id, r["pseudo_id"], aid, float(m)))


# ---------- datasets ----------

def save_dataset(course_id, kind, label, filename, username, structure, mapping, validation, rows):
    """rows: list of dicts pseudo_id, display_id, cohort, marks (list), final_grade, final_mark.
    Saves a new version of the (course, kind, label) slot and activates it."""
    c = conn()
    try:
        version = c.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM datasets WHERE course_id=? AND kind=? AND label=?",
                            (course_id, kind, label)).fetchone()[0]
        n_outcome = sum(1 for r in rows if r.get("final_grade"))
        cur = c.execute("""INSERT INTO datasets (course_id, kind, label, version, filename, uploaded_by, uploaded_at, is_active,
                           n_rows, n_with_outcome, structure_json, mapping_json, validation_json)
                           VALUES (?,?,?,?,?,?,?,0,?,?,?,?,?)""",
                        (course_id, kind, label, version, filename, username, now(), len(rows), n_outcome,
                         json.dumps(structure), json.dumps(mapping), json.dumps(validation)))
        dataset_id = cur.lastrowid
        c.executemany("""INSERT INTO dataset_rows (dataset_id, pseudo_id, display_id, cohort, marks_json, final_grade,
                         final_mark) VALUES (?,?,?,?,?,?,?)""",
                      [(dataset_id, r["pseudo_id"], r.get("display_id"), r.get("cohort") or label, json.dumps(r["marks"]),
                        r.get("final_grade"), r.get("final_mark")) for r in rows])
        _activate(c, dataset_id)
        c.commit()
        return dataset_id, version
    finally:
        c.close()


def _activate(c, dataset_id):
    ds = c.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    if ds["kind"] == "current":
        # A course has one current cohort, so activating any current dataset
        # deactivates every other current dataset of the course.
        c.execute("UPDATE datasets SET is_active=0 WHERE course_id=? AND kind='current'", (ds["course_id"],))
    else:
        c.execute("UPDATE datasets SET is_active=0 WHERE course_id=? AND kind=? AND label=?",
                  (ds["course_id"], ds["kind"], ds["label"]))
    c.execute("UPDATE datasets SET is_active=1 WHERE id=?", (dataset_id,))
    if ds["kind"] == "current":
        _materialise_current(c, ds["course_id"], dataset_id)


def activate_dataset(dataset_id):
    c = conn()
    _activate(c, dataset_id)
    c.commit()
    c.close()


def deactivate_dataset(dataset_id):
    c = conn()
    ds = c.execute("SELECT * FROM datasets WHERE id=?", (dataset_id,)).fetchone()
    c.execute("UPDATE datasets SET is_active=0 WHERE id=?", (dataset_id,))
    if ds["kind"] == "current":
        _materialise_current(c, ds["course_id"], None)
    c.commit()
    c.close()


def get_dataset(dataset_id):
    d = _row("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if d:
        d["structure"] = json.loads(d["structure_json"] or "[]")
        d["mapping"] = json.loads(d["mapping_json"] or "{}")
        d["validation"] = json.loads(d["validation_json"] or "{}")
    return d


def list_datasets(course_id=None):
    sql = """SELECT d.id, d.course_id, d.kind, d.label, d.version, d.filename, d.uploaded_by, d.uploaded_at,
                    d.is_active, d.n_rows, d.n_with_outcome, c.code AS course_code
             FROM datasets d JOIN courses c ON c.id=d.course_id"""
    if course_id:
        return _rows(sql + " WHERE d.course_id=? ORDER BY d.kind, d.label, d.version DESC", (course_id,))
    return _rows(sql + " ORDER BY c.code, d.kind, d.label, d.version DESC")


def active_current_dataset(course_id):
    return _row("SELECT * FROM datasets WHERE course_id=? AND kind='current' AND is_active=1", (course_id,))


def historical_training_data():
    """Every active historical dataset with outcome rows, for pooled training."""
    out = []
    for d in _rows("""SELECT d.*, c.code AS course_code FROM datasets d JOIN courses c ON c.id=d.course_id
                      WHERE d.kind='historical' AND d.is_active=1 AND d.n_with_outcome > 0 ORDER BY d.id"""):
        rows = _rows("SELECT pseudo_id, cohort, marks_json, final_grade, final_mark FROM dataset_rows WHERE dataset_id=?",
                     (d["id"],))
        for r in rows:
            r["marks"] = json.loads(r.pop("marks_json"))
        out.append({"dataset_id": d["id"], "course_code": d["course_code"], "course_id": d["course_id"],
                    "label": d["label"], "structure": json.loads(d["structure_json"]), "rows": rows})
    return out


def course_history_stats(course_code):
    """Historical outcomes for a course code, across all its active cohorts."""
    r = _row("""SELECT COUNT(*) AS n, SUM(CASE WHEN dr.final_grade='D' THEN 1 ELSE 0 END) AS fails,
                       COUNT(DISTINCT dr.cohort) AS cohorts
                FROM dataset_rows dr JOIN datasets d ON d.id=dr.dataset_id JOIN courses c ON c.id=d.course_id
                WHERE c.code=? AND d.kind='historical' AND d.is_active=1 AND dr.final_grade IS NOT NULL""",
             (course_code.upper(),))
    n = r["n"] or 0
    return {"n": n, "cohorts": r["cohorts"] or 0, "pass_rate": (1 - (r["fails"] or 0) / n) if n else None}


# ---------- mapping profiles (reusable course structure configuration) ----------

def get_profile(course_code, kind):
    r = _row("SELECT profile_json FROM mapping_profiles WHERE course_code=? AND kind=?", (course_code.upper(), kind))
    if not r:
        r = _row("SELECT profile_json FROM mapping_profiles WHERE course_code=? ORDER BY updated_at DESC LIMIT 1",
                 (course_code.upper(),))
    return json.loads(r["profile_json"]) if r else None


def save_profile(course_code, kind, profile, username):
    c = conn()
    c.execute("""INSERT INTO mapping_profiles (course_code, kind, profile_json, updated_at, updated_by) VALUES (?,?,?,?,?)
                 ON CONFLICT(course_code, kind) DO UPDATE SET profile_json=excluded.profile_json,
                 updated_at=excluded.updated_at, updated_by=excluded.updated_by""",
              (course_code.upper(), kind, json.dumps(profile), now(), username))
    c.commit()
    c.close()


# ---------- model versions ----------

def create_model_version(username, trigger):
    c = conn()
    cur = c.execute("INSERT INTO model_versions (created_at, created_by, trigger, status) VALUES (?,?,?,?)",
                    (now(), username, trigger, "training"))
    c.commit()
    mid = cur.lastrowid
    c.close()
    return mid


def finish_model_version(model_id, summary, metrics, blob, activate=True):
    """Marks a training run ready and stores the trained model in the database."""
    c = conn()
    if activate:
        c.execute("UPDATE model_versions SET is_active=0")
    c.execute("""UPDATE model_versions SET status='ready', algorithm=?, n_students=?, n_rows=?, n_courses=?, n_cohorts=?,
                 metrics_json=?, summary_json=?, model_path=NULL, model_blob=?, blob_bytes=?, is_active=? WHERE id=?""",
              (summary["algorithm"], summary["n_students"], summary["n_rows"], summary["n_courses"], summary["n_cohorts"],
               json.dumps(metrics), json.dumps(summary), sqlite3.Binary(blob), len(blob), 1 if activate else 0, model_id))
    c.commit()
    c.close()


def get_model_blob(model_id):
    c = conn()
    r = c.execute("SELECT model_blob FROM model_versions WHERE id=?", (model_id,)).fetchone()
    c.close()
    return bytes(r["model_blob"]) if r and r["model_blob"] is not None else None


def store_model_blob(model_id, blob):
    c = conn()
    c.execute("UPDATE model_versions SET model_blob=?, blob_bytes=? WHERE id=?", (sqlite3.Binary(blob), len(blob), model_id))
    c.commit()
    c.close()


def fail_model_version(model_id, error):
    c = conn()
    c.execute("UPDATE model_versions SET status='failed', error=? WHERE id=?", (error, model_id))
    c.commit()
    c.close()


def _model(r):
    if r:
        r["metrics"] = json.loads(r["metrics_json"] or "{}")
        r["summary"] = json.loads(r["summary_json"] or "{}")
    return r


_MODEL_COLS = """id, created_at, created_by, trigger, status, error, algorithm, n_students, n_rows, n_courses,
                 n_cohorts, metrics_json, summary_json, model_path, blob_bytes, is_active"""


def get_active_model():
    return _model(_row(f"SELECT {_MODEL_COLS} FROM model_versions WHERE is_active=1 AND status='ready' ORDER BY id DESC LIMIT 1"))


def get_model_version(model_id):
    return _model(_row(f"SELECT {_MODEL_COLS} FROM model_versions WHERE id=?", (model_id,)))


def list_model_versions():
    return [_model(r) for r in _rows(f"SELECT {_MODEL_COLS} FROM model_versions ORDER BY id DESC")]


def activate_model_version(model_id):
    c = conn()
    c.execute("UPDATE model_versions SET is_active=0")
    c.execute("UPDATE model_versions SET is_active=1 WHERE id=? AND status='ready'", (model_id,))
    c.commit()
    c.close()


def model_training_in_progress():
    return _row("SELECT id FROM model_versions WHERE status='training' ORDER BY id DESC LIMIT 1")


# ---------- summary cache ----------

def get_summary_cache(course_id, cache_key):
    c = conn()
    r = c.execute("SELECT payload FROM summary_cache WHERE course_id=? AND cache_key=?", (course_id, cache_key)).fetchone()
    c.close()
    return bytes(r["payload"]) if r else None


def put_summary_cache(course_id, cache_key, payload):
    c = conn()
    try:
        c.execute("""INSERT INTO summary_cache (course_id, cache_key, created_at, payload) VALUES (?,?,?,?)
                     ON CONFLICT(course_id) DO UPDATE SET cache_key=excluded.cache_key, created_at=excluded.created_at,
                     payload=excluded.payload""", (course_id, cache_key, now(), sqlite3.Binary(payload)))
        c.commit()
    except sqlite3.OperationalError:
        pass  # read only database: the in-memory cache still works
    finally:
        c.close()


def course_cache_key(course_id):
    """Changes whenever anything that affects a course's results changes."""
    course = get_course(course_id)
    ds = active_current_dataset(course_id)
    model = _row("SELECT id FROM model_versions WHERE is_active=1 AND status='ready' ORDER BY id DESC LIMIT 1")
    types = ",".join(f"{a['id']}:{a['atype']}" for a in get_assessments(course_id))
    hist = course_history_stats(course["code"])
    return (f"d{ds['id'] if ds else 0}|m{model['id'] if model else 0}|r{course['risk_threshold']}|"
            f"b{course['borderline_threshold']}|h{hist['n']}|t{zlib.crc32(types.encode())}")


# ---------- small settings and results store ----------

def set_meta(key, value):
    c = conn()
    c.execute("INSERT INTO app_meta (key, value_json) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json",
              (key, json.dumps(value)))
    c.commit()
    c.close()


def get_meta(key, default=None):
    r = _row("SELECT value_json FROM app_meta WHERE key=?", (key,))
    return json.loads(r["value_json"]) if r else default


def set_course_outline(course_id, outline):
    c = conn()
    c.execute("UPDATE courses SET outline_json=? WHERE id=?", (json.dumps(outline), course_id))
    c.commit()
    c.close()


def get_course_outline(course):
    try:
        return json.loads(course.get("outline_json") or "null")
    except (TypeError, ValueError):
        return None
