"""
sample_data.py
Creates realistic SAMPLE files so the system can be demonstrated before
real, de-identified USP data is available. All students are fictional.

The courses deliberately use DIFFERENT assessment structures and file
layouts, to exercise the flexible mapping:

  IS413   wide, weights and maximum marks written in the column headers
  CS111   wide, with "Weight" and "Out of" rows under the header, raw marks
  MA102   long format: one row per student per assessment
  BIO150  practical heavy course, history uses a final mark not a grade
  ED200   no exam, unusual assessment names, and NO history (cold start)

Run on its own to regenerate the files in sample_data/:
    python sample_data.py
"""

import csv
import os
import random

from config import SAMPLE_DIR, grade_for_mark

# name, weight, type, max mark, due date (month-day)
COURSES = {
    "IS413": {"name": "Advances in ICT", "layout": "header",
              "assessments": [("Quiz 1", 5, "Quiz", 20, "08-01"), ("Assignment 1", 15, "Coursework", 40, "08-15"),
                              ("Lab Test", 10, "Practical", 100, "08-29"), ("Mid Semester Test", 20, "Test", 50, "09-12"),
                              ("Assignment 2", 15, "Coursework", 40, "10-03"), ("Final Exam", 35, "Exam", 100, "11-10")],
              "history": [("2024 S2", 130), ("2025 S1", 150), ("2025 S2", 140)], "graded_now": 4},
    "CS111": {"name": "Introduction to Computing Science", "layout": "metarows",
              "assessments": [("Lab 1", 5, "Practical", 10, "07-25"), ("Quiz 1", 5, "Quiz", 15, "08-01"),
                              ("Assignment 1", 10, "Coursework", 50, "08-15"), ("Mid Semester Test", 20, "Test", 60, "09-05"),
                              ("Lab 2", 5, "Practical", 10, "09-19"), ("Assignment 2", 15, "Coursework", 50, "10-03"),
                              ("Final Exam", 40, "Exam", 100, "11-12")],
              "history": [("2024 S1", 160), ("2024 S2", 110), ("2025 S1", 170)], "graded_now": 4},
    "MA102": {"name": "Mathematics for Social Sciences", "layout": "long",
              "assessments": [("Quiz A", 10, "Quiz", 25, "08-08"), ("Tutorial Participation", 10, "Coursework", 10, "08-30"),
                              ("In-class Test", 25, "Test", 40, "09-05"), ("Quiz B", 10, "Quiz", 25, "09-19"),
                              ("Final Examination", 45, "Exam", 100, "11-08")],
              "history": [("2025 S1", 125)], "graded_now": 3},
    "BIO150": {"name": "Foundations of Biology", "layout": "header_finalmark",
               "assessments": [("Practical 1", 8, "Practical", 100, "07-30"), ("Practical 2", 8, "Practical", 100, "08-13"),
                               ("Field Report", 14, "Coursework", 100, "09-01"), ("Practical 3", 8, "Practical", 100, "09-17"),
                               ("Practical 4", 7, "Practical", 100, "10-01"), ("Final Exam", 55, "Exam", 100, "11-14")],
               "history": [("2024 S2", 105), ("2025 S2", 115)], "graded_now": 3},
    "ED200": {"name": "Curriculum and Teaching Practice", "layout": "header",
              "assessments": [("Reflective Journal", 20, "Coursework", 100, "08-08"), ("Online Check-ins", 10, "Quiz", 100, "08-22"),
                              ("Microteaching Presentation", 30, "Coursework", 100, "09-12"),
                              ("Lesson Plan Portfolio", 40, "Coursework", 100, "10-24")],
              "history": [], "graded_now": 2},
}

CURRENT_SEMESTER = "Semester 2, 2026"


def _student(rng, assessments, harshness):
    ability = rng.gauss(62, 14)
    skills = {"Exam": rng.gauss(-2, 7), "Test": rng.gauss(0, 6), "Coursework": rng.gauss(3, 6),
              "Quiz": rng.gauss(0, 6), "Practical": rng.gauss(3, 6), "Other": 0}
    drift = rng.choice([0, 0, 0, -3.5, -2, 1.5])
    marks = []
    for i, (name, _, atype, _, _) in enumerate(assessments):
        m = ability + skills[atype] + harshness[i] + drift * i + rng.gauss(0, 7)
        if rng.random() < 0.02:
            m = 0
        marks.append(max(0.0, min(100.0, m)))
    return marks


def _raw(pct, max_mark):
    return round(pct * max_mark / 100.0, 1)


def _header(a):
    name, weight, _, mx, _ = a
    return f"{name} ({weight}%)" + (f" /{mx}" if mx != 100 else "")


def _write(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def _cohort(rng, spec, n):
    """Marks for one cohort. Each cohort has its own marking harshness, so
    raw marks differ between semesters but relative standing does not."""
    shift = rng.gauss(0, 4)
    harsh = [shift + rng.gauss(0, 4) for _ in spec["assessments"]]
    return [_student(rng, spec["assessments"], harsh) for _ in range(n)]


def generate(code, seed=7):
    rng = random.Random(seed * 31 + sum(map(ord, code)))
    spec = COURSES[code]
    A = spec["assessments"]
    weights = [a[1] for a in A]
    year = "2026"
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    files = {}

    # ---- historical
    if spec["history"]:
        hist = []
        for label, n in spec["history"]:
            for marks in _cohort(rng, spec, n):
                final = sum(m * w for m, w in zip(marks, weights)) / sum(weights)
                hist.append((label, marks, final))
        path = os.path.join(SAMPLE_DIR, f"{code}_history.csv")
        if spec["layout"] == "long":
            rows = [["student_id", "cohort", "assessment", "mark", "max_mark", "weight", "date", "type", "final_grade"]]
            for i, (label, marks, final) in enumerate(hist):
                sid = f"H{code[:2]}{i + 1:05d}"
                for a, m in zip(A, marks):
                    rows.append([sid, label, a[0], _raw(m, a[3]), a[3], a[1], f"{label[:4]}-{a[4]}", a[2], grade_for_mark(final)])
        elif spec["layout"] == "metarows":
            rows = [["Student ID", "Cohort"] + [a[0] for a in A] + ["Final Grade"],
                    ["Weight", ""] + [a[1] for a in A] + [""],
                    ["Out of", ""] + [a[3] for a in A] + [""]]
            for i, (label, marks, final) in enumerate(hist):
                rows.append([f"H{code[:2]}{i + 1:05d}", label] + [_raw(m, a[3]) for m, a in zip(marks, A)] + [grade_for_mark(final)])
        elif spec["layout"] == "header_finalmark":
            rows = [["student_id", "semester"] + [_header(a) for a in A] + ["Final Mark"]]
            for i, (label, marks, final) in enumerate(hist):
                rows.append([f"H{code[:2]}{i + 1:05d}", label] + [_raw(m, a[3]) for m, a in zip(marks, A)] + [round(final, 1)])
        else:
            rows = [["student_id", "cohort"] + [_header(a) for a in A] + ["final_grade"]]
            for i, (label, marks, final) in enumerate(hist):
                rows.append([f"H{code[:2]}{i + 1:05d}", label] + [_raw(m, a[3]) for m, a in zip(marks, A)] + [grade_for_mark(final)])
        _write(path, rows)
        files["history"] = path

    # ---- current semester, only the first assessments marked so far
    k = spec["graded_now"]
    current = _cohort(rng, spec, 120)
    ids = [f"S11{rng.randint(100000, 999999)}" for _ in current]
    path = os.path.join(SAMPLE_DIR, f"{code}_current.csv")
    if spec["layout"] == "long":
        rows = [["student_id", "name", "assessment", "mark", "max_mark", "weight", "date", "type"]]
        for n, (sid, marks) in enumerate(zip(ids, current)):
            for j, (a, m) in enumerate(zip(A, marks)):
                if j < k:
                    rows.append([sid, f"Student {n + 1:03d}", a[0], _raw(m, a[3]), a[3], a[1], f"{year}-{a[4]}", a[2]])
    elif spec["layout"] == "metarows":
        rows = [["Student ID", "Name"] + [a[0] for a in A],
                ["Weight", ""] + [a[1] for a in A],
                ["Out of", ""] + [a[3] for a in A],
                ["Due date", ""] + [f"{year}-{a[4]}" for a in A]]
        for n, (sid, marks) in enumerate(zip(ids, current)):
            rows.append([sid, f"Student {n + 1:03d}"] + [_raw(m, a[3]) if j < k else "" for j, (m, a) in enumerate(zip(marks, A))])
    else:
        rows = [["student_id", "name", "programme"] + [_header(a) for a in A]]
        for n, (sid, marks) in enumerate(zip(ids, current)):
            rows.append([sid, f"Student {n + 1:03d}", rng.choice(["BSE", "BNS", "BCom IS", "BSc CS", "BEd"])]
                        + [_raw(m, a[3]) if j < k else "" for j, (m, a) in enumerate(zip(marks, A))])
    _write(path, rows)
    files["current"] = path
    return files


def generate_all():
    return {code: generate(code) for code in COURSES}


if __name__ == "__main__":
    generate_all()
    print(f"Sample files written to {SAMPLE_DIR}")
