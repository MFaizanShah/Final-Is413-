"""
tools/prepare_oulad.py
Converts the Open University Learning Analytics Dataset (OULAD) into the
upload format this system reads, one file per module presentation.

OULAD is real, anonymised data from the Open University (UK), released under
the Creative Commons CC BY 4.0 licence:
    Kuzilek J., Hlosta M., Zdrahal Z. Open University Learning Analytics
    dataset. Sci. Data 4:170171 (2017). https://doi.org/10.1038/sdata.2017.171
    https://analyse.kmi.open.ac.uk/open-dataset

Usage:
    python tools/prepare_oulad.py <folder with assessments.csv, studentAssessment.csv,
                                   studentInfo.csv, studentRegistration.csv>

How OULAD is mapped onto this system
------------------------------------
* Course = module (AAA to FFF). Cohort = presentation (2013B, 2013J, 2014B, 2014J).
  Module GGG is left out because none of its coursework carries any weight.
* Assessments: Tutor Marked Assignments (TMA, typed Coursework) and Computer
  Marked Assessments (CMA, typed Quiz) with a weight above zero, in date
  order, plus the final exam. OU modules usually combine continuous
  assessment and exam equally, so coursework weights are scaled to 50% of the
  course and the exam is 50%.
* A coursework item with no submission is recorded as 0 (not submitted).
  Exam scores are only published for some modules; elsewhere the exam is
  left blank (not assessed).
* Outcome: Pass and Distinction are Pass, Fail is Fail. Students who
  withdrew have no final result and are left out of the historical files.
* History = 2013B, 2013J and 2014B. The 2014J presentation is written as the
  "current semester", cut at the point where about half of the coursework
  weight has been marked, and its real outcomes are written separately so
  predictions can be checked against what actually happened.
"""

import csv
import os
import sys

import pandas as pd

MODULES = {
    "AAA": "Social Sciences module AAA",
    "BBB": "Social Sciences module BBB",
    "CCC": "STEM module CCC",
    "DDD": "STEM module DDD",
    "EEE": "STEM module EEE",
    "FFF": "STEM module FFF",
}
HISTORY = ["2013B", "2013J", "2014B"]
CURRENT = "2014J"
CURRENT_SHARE = 0.5   # share of coursework weight marked in the "current" file

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "oulad")


def structure(assess, module, pres):
    a = assess[(assess.code_module == module) & (assess.code_presentation == pres)]
    cw = a[(a.assessment_type != "Exam") & (a.weight > 0)].sort_values(["date", "id_assessment"])
    exams = a[a.assessment_type == "Exam"]
    total = cw.weight.sum()
    items, counts = [], {"TMA": 0, "CMA": 0}
    for _, r in cw.iterrows():
        counts[r.assessment_type] += 1
        name = (f"Assignment {counts['TMA']}" if r.assessment_type == "TMA" else f"Online Quiz {counts['CMA']}")
        items.append({"ids": [int(r.id_assessment)], "name": name, "weight": 50.0 * r.weight / total,
                      "date": r.date, "kind": r.assessment_type})
    # Rounded weights that still add up to exactly 50.
    rounded = [round(i["weight"], 2) for i in items]
    rounded[-1] = round(50.0 - sum(rounded[:-1]), 2)
    for i, w in zip(items, rounded):
        i["weight"] = w
    items.append({"ids": [int(x) for x in exams.id_assessment], "name": "Final Exam", "weight": 50.0,
                  "date": None, "kind": "Exam"})
    return items


def header(item):
    w = item["weight"]
    return f"{item['name']} ({w:g}%)"


def write(path, columns, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(columns)
        w.writerows(rows)


def main(src):
    assess = pd.read_csv(os.path.join(src, "assessments.csv"))
    sa = pd.read_csv(os.path.join(src, "studentAssessment.csv"), na_values=["?"])
    info = pd.read_csv(os.path.join(src, "studentInfo.csv"))
    reg = pd.read_csv(os.path.join(src, "studentRegistration.csv"), na_values=["?"])
    score = {(int(r.id_assessment), int(r.id_student)): r.score for r in sa.itertuples()}

    summary = []
    for module in MODULES:
        for pres in HISTORY + [CURRENT]:
            students = info[(info.code_module == module) & (info.code_presentation == pres)]
            if students.empty:
                continue
            items = structure(assess, module, pres)
            sid = lambda s: f"OU{int(s):07d}"

            def mark(item, s):
                vals = [score.get((i, int(s))) for i in item["ids"]]
                vals = [v for v in vals if v is not None and not pd.isna(v)]
                if vals:
                    return round(float(max(vals)), 1)
                return 0.0 if item["kind"] != "Exam" else ""

            if pres in HISTORY:
                done = students[students.final_result != "Withdrawn"]
                rows = [[sid(s.id_student), pres] + [mark(i, s.id_student) for i in items] +
                        ["Fail" if s.final_result == "Fail" else "Pass"] for s in done.itertuples()]
                write(os.path.join(OUT, "history", f"{module}_{pres}.csv"),
                      ["student_id", "cohort"] + [header(i) for i in items] + ["final_result"], rows)
                summary.append((module, pres, "history", len(rows), sum(r[-1] == "Fail" for r in rows)))
            else:
                cw = [i for i in items if i["kind"] != "Exam"]
                cum, cut = 0.0, 0
                for k, i in enumerate(cw):
                    cum += i["weight"]
                    cut = k + 1
                    if cum >= CURRENT_SHARE * 50 - 0.01:
                        break
                cutoff_day = cw[cut - 1]["date"]
                r = reg[(reg.code_module == module) & (reg.code_presentation == pres)].set_index("id_student")
                # Students still registered when the last included assessment was due.
                active = [s for s in students.itertuples()
                          if pd.isna(r.loc[s.id_student, "date_unregistration"])
                          or r.loc[s.id_student, "date_unregistration"] > cutoff_day]
                rows = [[sid(s.id_student)] + [mark(i, s.id_student) if k < cut else "" for k, i in enumerate(items)]
                        for s in active]
                write(os.path.join(OUT, "current", f"{module}_{pres}.csv"),
                      ["student_id"] + [header(i) for i in items], rows)
                write(os.path.join(OUT, "actual_outcomes", f"{module}_{pres}.csv"),
                      ["student_id", "actual_result"], [[sid(s.id_student), s.final_result] for s in active])
                summary.append((module, pres, f"current, {cut} of {len(items)} assessments marked", len(rows), None))

    for s in summary:
        print(*s)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
