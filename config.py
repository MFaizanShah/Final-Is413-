"""
config.py
Central settings for the USP At Risk Tracker. Edit the grade scale here if
your faculty uses different boundaries.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Running on Vercel (serverless). The project folder is read only there, so
# the bundled database is copied to /tmp on start. Changes made on Vercel
# last only until that copy is recycled, and training is switched off
# because it takes longer than a serverless request may run.
IS_VERCEL = bool(os.environ.get("VERCEL"))
BUNDLED_DB_PATH = os.path.join(DATA_DIR, "atrisk.db")
if IS_VERCEL:
    DB_PATH = os.environ.get("ATRISK_DB_PATH", "/tmp/atrisk.db")
    STAGING_DIR = "/tmp/atrisk_staging"
else:
    DB_PATH = os.environ.get("ATRISK_DB_PATH", BUNDLED_DB_PATH)
    STAGING_DIR = os.environ.get("ATRISK_STAGING_DIR", os.path.join(BASE_DIR, "staging"))
TRAINING_ENABLED = os.environ.get("ATRISK_TRAINING", "0" if IS_VERCEL else "1") == "1"

# Older databases whose user accounts are copied across on first start.
LEGACY_DB_PATHS = [os.path.join(BASE_DIR, "usp_atrisk_v2.db"), os.path.join(BASE_DIR, "usp_atrisk.db")]
# Folder that older versions saved model files in (models now live in the database).
LEGACY_MODEL_DIR = os.path.join(BASE_DIR, "models")
SAMPLE_DIR = os.path.join(BASE_DIR, "sample_data")
REAL_DATA_DIR = os.path.join(DATA_DIR, "oulad")

SECRET_KEY = os.environ.get("ATRISK_SECRET_KEY", "change-this-secret-key")

# Key for pseudonymising student identifiers. Anyone holding this key can
# re-identify a student by hashing a known ID, so keep it out of source
# control in production and never change it once data has been loaded.
# The bundled database was built with the default below; if you change it,
# rebuild the database (python build_database.py --fresh) so stored codes match.
PSEUDONYM_KEY = os.environ.get("ATRISK_PSEUDONYM_KEY", "change-this-secret-key")

# Optional LLM suggestion for assessment types. Off unless explicitly enabled.
# Only assessment names are sent, never marks or student identifiers.
LLM_SUGGESTIONS = os.environ.get("ATRISK_LLM_SUGGESTIONS", "0") == "1"
LLM_MODEL = os.environ.get("ATRISK_LLM_MODEL", "claude-sonnet-5-5")

# Optional AI wording for the assistant. When ANTHROPIC_API_KEY is set and
# ATRISK_LLM_ASSISTANT=1, the assistant's answers (always computed from the
# course data first) are rephrased conversationally. Only the computed
# facts for the question are sent, with pseudonymous codes, never real IDs.
LLM_ASSISTANT = os.environ.get("ATRISK_LLM_ASSISTANT", "0") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))

# Grade scale, best to worst. D is the fail grade.
GRADES = ["A+", "A", "B+", "B", "C+", "C", "D"]
GRADE_BOUNDS = [("A+", 85), ("A", 78), ("B+", 71), ("B", 64), ("C+", 57), ("C", 50), ("D", 0)]
FAIL_GRADE = "D"
PASS_MARK = 50.0

# Representative mark for each grade, used to turn grade probabilities into
# a single projected mark.
GRADE_MIDPOINTS = {"A+": 90, "A": 81, "B+": 74, "B": 67, "C+": 60, "C": 53, "D": 40}

# The global model needs at least this many historical students with
# outcomes, and both passes and fails, before it is trained.
MIN_TRAINING_STUDENTS = 30

# Training snapshots per past student. Students in courses with many
# assessments contribute at most this many evenly spaced stages, which keeps
# training fast without losing the early, middle and late picture.
MAX_STAGES_PER_STUDENT = 5

# Monte Carlo cold start settings.
MC_SIMULATIONS = 2000

# Broad assessment types. Every course is mapped onto these so features and
# advice work for any structure.
TYPES = ["Coursework", "Quiz", "Test", "Practical", "Exam", "Other"]

# Rule based classification from the assessment name. Order matters: the
# first matching rule wins.
TYPE_RULES = [
    ("Exam", ["final exam", "exam", "examination"]),
    ("Practical", ["lab", "practical", "prac", "workshop", "studio", "fieldwork", "field work", "clinical"]),
    ("Test", ["mid sem", "midsem", "mid-sem", "midterm", "mid term", "test", "mst", "in-class test"]),
    ("Quiz", ["quiz", "mcq", "check"]),
    ("Coursework", ["assign", "project", "essay", "report", "case", "portfolio", "tut", "presentation",
                    "participation", "attendance", "discussion", "journal", "reflection", "homework",
                    "hw", "group", "poster", "proposal", "review", "coursework", "cw", "seminar"]),
    ("Exam", ["final"]),
]

# Suggested support actions per weak type. These are prompts for the
# lecturer, not automatic decisions.
TYPE_ADVICE = {
    "Exam": ("Exam performance",
             "Recommend past paper practice, timed revision and an exam technique workshop through Student Learning Support."),
    "Test": ("Test performance",
             "Suggest revisiting the tested topics with the tutor and practising under timed conditions before the next test."),
    "Quiz": ("Regular topic checks",
             "Encourage weekly revision and attendance at tutorials, since quiz results usually reflect week to week engagement."),
    "Practical": ("Practical skills",
                  "Recommend extra lab or practical sessions and a check in with the demonstrator on the techniques missed."),
    "Coursework": ("Written and applied work",
                   "Suggest reviewing marker feedback, planning submissions earlier, and using writing and referencing support."),
    "Other": ("Other assessed work",
              "Review this assessment with the student to identify the specific gap."),
}


def detect_type(name):
    lower = " " + (name or "").lower().replace("_", " ") + " "
    for atype, keywords in TYPE_RULES:
        if any(k in lower for k in keywords):
            return atype
    return "Other"


def normalise_type(value):
    text = str(value or "").strip().lower()
    for t in TYPES:
        if text == t.lower():
            return t
    aliases = {"lab": "Practical", "labs": "Practical", "assignment": "Coursework", "assignments": "Coursework",
               "participation": "Coursework", "project": "Coursework", "midterm": "Test", "final": "Exam"}
    if text in aliases:
        return aliases[text]
    return detect_type(text) if text else None


def grade_for_mark(mark):
    if mark is None:
        return None
    for grade, lower in GRADE_BOUNDS:
        if mark >= lower:
            return grade
    return FAIL_GRADE


def grade_lower_bound(grade):
    return dict(GRADE_BOUNDS)[grade]


def normalise_grade(value):
    if value is None:
        return None
    text = str(value).strip().upper().replace(" ", "")
    # D* is a fail (D) recorded with a note, as some faculties write it.
    aliases = {"F": "D", "FAIL": "D", "E": "D", "D*": "D"}
    text = aliases.get(text, text)
    return text if text in GRADES else None
