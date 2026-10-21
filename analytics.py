"""
analytics.py
Turns marks into decisions a lecturer can act on, for any course structure:

  * feasibility: what is still mathematically possible, from the weights
  * prediction: the calibrated global model when the course has history,
    otherwise a clearly labelled Monte Carlo simulation
  * risk flag: At Risk / Borderline / On Track, always with reasons
  * weakness analysis by broad assessment type (coursework, quiz, test,
    practical, exam), compared with the class
  * recommendations: suggested support actions for the lecturer
"""

import pickle
import zlib

import numpy as np

import database as db
import features as fx
import ml_engine
from config import GRADES, FAIL_GRADE, PASS_MARK, TYPE_ADVICE, grade_for_mark, grade_lower_bound

STATUS_ORDER = {"At Risk": 0, "Borderline": 1, "On Track": 2, "No data": 3}
TONE = {"At Risk": "danger", "Borderline": "warning", "On Track": "success", "No data": "neutral"}


def _worse(current, candidate):
    return candidate if STATUS_ORDER[candidate] < STATUS_ORDER[current] else current


class CourseModel:
    """Everything needed to assess students of one course: class matrix,
    features and which prediction path applies."""

    def __init__(self, course, assessments, students, marks, version, bundle):
        self.course, self.assessments, self.students = course, assessments, students
        self.version, self.bundle = version, bundle
        self.history = db.course_history_stats(course["code"])
        # The global random forest scores ANY course, because it learns from
        # structure independent features. A course whose own history is in
        # the training data is simply a closer match.
        self.trained = bool(bundle and assessments)
        self.own_history = bool(bundle and course["code"].upper() in bundle["courses"])
        self.X = fx.matrix_for_course(assessments, students, marks) if assessments else np.zeros((0, 0))
        mean, sd = fx.class_stats(self.X) if len(students) and assessments else (np.array([]), np.array([]))
        self.class_avg = {a["id"]: (None if np.isnan(mean[j]) else round(float(mean[j]), 1))
                          for j, a in enumerate(assessments)} if len(mean) else {a["id"]: None for a in assessments}
        self.class_sd = {a["id"]: (None if np.isnan(sd[j]) else float(sd[j]))
                         for j, a in enumerate(assessments)} if len(sd) else {a["id"]: None for a in assessments}
        self.F = (fx.current_features(assessments, self.X, self.history["pass_rate"])
                  if len(students) and assessments else None)
        self.source_label = self._source_label()

    def _source_label(self):
        if self.trained:
            n = self.bundle["n_students"]
            if self.own_history:
                extra = f", including {self.history['n']:,} past {self.course['code']} students"
            else:
                extra = (f". {self.course['code']} has no past results of its own yet, so it is compared with "
                         "students from other courses at the same stage")
            return {"kind": "model", "own_history": self.own_history,
                    "text": f"Random forest trained on {n:,} past students from {len(self.bundle['courses'])} courses{extra}."}
        why = "no model has been trained yet" if not self.bundle else "the course has no assessments yet"
        return {"kind": "simulation", "own_history": False,
                "text": f"Simulated estimates, because {why}. Upload past results and train the model for trained predictions."}

    def predict_rows(self, indices):
        """Predictions for student rows in self.X."""
        out = {}
        if not len(indices):
            return out
        if self.trained:
            preds = ml_engine.predict_trained(self.bundle, self.F[indices])
            for i, p in zip(indices, preds):
                out[i] = p
        else:
            for i in indices:
                out[i] = ml_engine.simulate(self.assessments, self.X, i, self.students[i]["student_id"])
        return out

    def drift(self):
        if not self.bundle or self.F is None:
            return None
        has = ~np.isnan(self.F[:, 0])
        return ml_engine.drift_report(self.bundle, self.F[has], self.assessments)


def evaluate(course, assessments, marks, cm, pred):
    """marks: {assessment_id: percent} for one student."""
    total_weight = sum(a["weight"] for a in assessments) or 100.0
    graded = [a for a in assessments if a["id"] in marks]
    graded_weight = sum(a["weight"] for a in graded)
    earned = sum(a["weight"] * marks[a["id"]] / 100.0 for a in graded) * 100.0 / total_weight
    remaining = (total_weight - graded_weight) * 100.0 / total_weight
    current_avg = (sum(a["weight"] * marks[a["id"]] for a in graded) / graded_weight) if graded_weight else None
    best_possible = earned + remaining
    complete = remaining <= 0.001

    def required_for(target):
        if complete:
            return None
        return round((target - earned) / remaining * 100.0, 1)

    grade_paths = []
    for g in reversed(GRADES[:-1]):  # C, C+, ... A+
        req = required_for(grade_lower_bound(g))
        grade_paths.append({"grade": g, "required": req,
                            "status": ("secured" if req is not None and req <= 0 else
                                       "possible" if req is not None and req <= 100 else "out of reach")})

    result = {
        "graded_count": len(graded), "total_count": len(assessments),
        "graded_weight": round(graded_weight * 100.0 / total_weight, 1),
        "earned": round(earned, 1), "remaining": round(remaining, 1),
        "current_avg": round(current_avg, 1) if current_avg is not None else None,
        "best_possible": round(best_possible, 1), "best_grade": grade_for_mark(best_possible),
        "required_for_pass": required_for(PASS_MARK), "feasible": best_possible >= PASS_MARK,
        "complete": complete, "grade_paths": grade_paths, "prediction": None, "reasons": [],
    }

    if not graded:
        result.update(status="No data", tone="neutral", predicted_grade=None, p_fail=None,
                      reasons=["No marks uploaded for this student yet."], weaknesses=[],
                      types=[], recommendations=[], trend=0.0, factors=[])
        return result

    graded_series = [marks[a["id"]] for a in graded]
    trend = float(np.polyfit(np.arange(len(graded_series)), graded_series, 1)[0]) if len(graded_series) >= 2 else 0.0
    result["trend"] = round(trend, 1)

    if complete:
        final_grade = grade_for_mark(earned)
        pred = {"source": "actual", "probs": {g: (1.0 if g == final_grade else 0.0) for g in GRADES},
                "predicted_grade": final_grade, "confidence": 1.0,
                "p_fail": 1.0 if final_grade == FAIL_GRADE else 0.0, "projected_mark": round(earned, 1),
                "factors": [], "confidence_text": "Final result, all assessments graded"}
    # ---- Check 1: arithmetic. What is mathematically possible from the
    # weights alone. When it settles the outcome, the model is not asked.
    secured = (not complete) and result["required_for_pass"] is not None and result["required_for_pass"] <= 0
    model_name = ("Random forest" if pred and pred.get("source") == "model" else
                  "Simulation" if pred and pred.get("source") == "simulation" else "Prediction")
    if complete:
        arith = {"verdict": "Finished", "tone": "neutral", "decides": True,
                 "text": f"All assessments are marked. Final mark {earned:.1f}%."}
    elif not result["feasible"]:
        arith = {"verdict": "Cannot pass", "tone": "danger", "decides": True,
                 "text": f"Even 100% on the remaining {remaining:.0f}% gives at most {best_possible:.1f}%, "
                         f"below the {PASS_MARK:.0f}% pass mark."}
    elif secured:
        arith = {"verdict": "Pass secured", "tone": "success", "decides": True,
                 "text": f"{earned:.1f}% is already banked, so the pass mark is reached even with 0 on the rest."}
    else:
        req = result["required_for_pass"]
        arith = {"verdict": "Still open", "tone": "warning" if req > 70 else "info", "decides": False,
                 "text": f"A pass is still possible: an average of {req:.0f}% is needed on the remaining "
                         f"{remaining:.0f}%. Best possible final mark {best_possible:.0f}%."}
    if pred and not complete and arith["decides"]:
        # The arithmetic is certain, so it overrides the model's estimate.
        pred = dict(pred)
        if not result["feasible"]:
            pred["probs"] = {g: (1.0 if g == FAIL_GRADE else 0.0) for g in GRADES}
            pred.update(p_fail=1.0, predicted_grade=FAIL_GRADE, confidence=1.0)
        else:
            passing = {g: v for g, v in pred["probs"].items() if g != FAIL_GRADE}
            total = sum(passing.values()) or 1.0
            pred["probs"] = {**{g: v / total for g, v in passing.items()}, FAIL_GRADE: 0.0}
            best = max(passing, key=passing.get)
            pred.update(p_fail=0.0, predicted_grade=best, confidence=round(pred["probs"][best], 3))
        pred["guardrail"] = arith["verdict"]
    if complete or arith["decides"]:
        model_check = {"verdict": "Not needed", "tone": "neutral",
                       "text": "The arithmetic already settles whether this student can pass."}
    elif pred:
        pf = pred["p_fail"] * 100
        mv = ("At risk" if pred["p_fail"] >= course["risk_threshold"] else
              "Borderline" if pred["p_fail"] >= course["borderline_threshold"] else "Likely to pass")
        model_check = {"verdict": mv, "tone": TONE[{"At risk": "At Risk", "Borderline": "Borderline"}.get(mv, "On Track")],
                       "text": (f"{model_name}: {pf:.0f}% chance of failing. " +
                                (f"Of 100 similar past students at this point, about {pf:.0f} failed."
                                 if pred["source"] == "model" else
                                 "Based on simulating the remaining marks from the student's standing in the class."))}
    else:
        model_check = {"verdict": "Unavailable", "tone": "neutral", "text": "No prediction could be made."}
    result["checks"] = {"arithmetic": arith, "model": model_check, "model_name": model_name}

    result["prediction"] = pred
    result["predicted_grade"] = pred["predicted_grade"] if pred else grade_for_mark(current_avg)
    result["p_fail"] = pred["p_fail"] if pred else None
    result["factors"] = (pred or {}).get("factors", [])

    risk_t, border_t = course["risk_threshold"], course["borderline_threshold"]
    reasons, status = [], "On Track"
    who = "The random forest" if pred and pred["source"] == "model" else "The simulation"

    if complete:
        status = "At Risk" if result["predicted_grade"] == FAIL_GRADE else "On Track"
        reasons.append(f"All assessments graded. Final mark {earned:.1f}% (grade {result['predicted_grade']}).")
    elif secured:
        reasons.append(f"Pass already secured: {earned:.1f}% of the final mark is banked.")
    else:
        if not result["feasible"]:
            status = "At Risk"
            reasons.append(f"Even 100% on the remaining {remaining:.0f}% of the course gives at most "
                           f"{best_possible:.0f}%, below the {PASS_MARK:.0f}% pass mark.")
        if pred and not arith["decides"]:
            pf = pred["p_fail"] * 100
            if pred["p_fail"] >= risk_t:
                status = "At Risk"
                reasons.append(f"{who} estimates a {pf:.0f}% chance of a D (fail) with "
                               f"{result['graded_weight']:.0f}% of the course assessed.")
            elif pred["predicted_grade"] == FAIL_GRADE:
                status = "At Risk"
                reasons.append(f"Most likely final grade is D, with a {pf:.0f}% chance of failing.")
            elif pred["p_fail"] >= border_t:
                status = _worse(status, "Borderline")
                reasons.append(f"A {pf:.0f}% chance of failing is above the borderline threshold of {border_t * 100:.0f}%.")
            elif pred["predicted_grade"] == "C":
                status = _worse(status, "Borderline")
                reasons.append("Predicted to pass with the lowest passing grade (C), leaving little margin.")
        req = result["required_for_pass"]
        if result["feasible"] and req is not None and req > 70:
            status = _worse(status, "Borderline")
            reasons.append(f"Needs an average of {req:.0f}% on the remaining assessments to pass.")
        if len(graded_series) >= 3 and trend <= -6:
            status = _worse(status, "Borderline")
            reasons.append(f"Marks are falling by about {abs(trend):.0f} points per assessment.")

    zeros = [a["name"] for a in graded if marks[a["id"]] == 0]
    if zeros and not secured and not complete:
        status = _worse(status, "Borderline")
        reasons.append("No submission recorded for: " + ", ".join(zeros) + ".")
    if status == "On Track" and not reasons:
        reasons.append(f"Performing well. Most likely final grade {result['predicted_grade']}.")

    result["status"], result["tone"], result["reasons"] = status, TONE[status], reasons
    _weakness(result, assessments, marks, cm, zeros)
    result["concern"] = _concern(result, zeros)
    return result


def _concern(r, zeros):
    """The single most useful, concrete thing to know about a student, for
    the class list. Specific facts come before the model's overall view."""
    if r["complete"]:
        return r["reasons"][0] if r["reasons"] else ""
    if not r["feasible"]:
        return f"Cannot reach 50% even with full marks on the rest (best possible {r['best_possible']:.0f}%)."
    if zeros:
        return "Missed " + ", ".join(zeros[:2]) + (f" and {len(zeros) - 2} more" if len(zeros) > 2 else "") + "."
    if r.get("trend", 0) <= -6 and r["graded_count"] >= 3:
        return f"Marks falling by about {abs(r['trend']):.0f} points per assessment."
    weak = [t for t in r.get("types", []) if t["level"] == "Weak"]
    req = r.get("required_for_pass")
    if r["status"] != "On Track" and req is not None and req > 65:
        return f"Needs {req:.0f}% on the rest to pass" + (f"; weakest in {weak[0]['atype'].lower()}." if weak else ".")
    if weak:
        t = weak[0]
        return f"Weak in {t['atype'].lower()}: {t['avg_mark']:.0f}% average, {t['gap']:+.0f} against the class."
    if r["status"] == "On Track":
        strong = [t["atype"].lower() for t in r.get("types", []) if t["level"] == "Strong"]
        return "Doing well" + (f", especially in {' and '.join(strong)}." if strong else ".")
    return r["reasons"][0] if r["reasons"] else ""


def _weakness(result, assessments, marks, cm, zeros):
    rows = []
    for a in assessments:
        if a["id"] not in marks:
            continue
        mark = marks[a["id"]]
        cavg, csd = cm.class_avg.get(a["id"]), cm.class_sd.get(a["id"])
        z = (mark - cavg) / max(csd, 3.0) if cavg is not None and csd is not None else None
        rows.append({"name": a["name"], "atype": a.get("atype") or "Other", "weight": a["weight"], "mark": round(mark, 1),
                     "class_avg": cavg, "gap_class": round(mark - cavg, 1) if cavg is not None else None,
                     "z": round(z, 2) if z is not None else None})
    result["weaknesses"] = rows

    groups = {}
    for r in rows:
        t = groups.setdefault(r["atype"], {"atype": r["atype"], "marks": [], "gaps": [], "zs": [], "weight": 0.0})
        t["marks"].append(r["mark"])
        t["weight"] += r["weight"]
        t["gaps"].append(r["gap_class"] if r["gap_class"] is not None else r["mark"] - PASS_MARK)
        if r["z"] is not None:
            t["zs"].append(r["z"])
    types = []
    for t in groups.values():
        avg_mark, avg_gap = float(np.mean(t["marks"])), float(np.mean(t["gaps"]))
        avg_z = float(np.mean(t["zs"])) if t["zs"] else None
        weak = avg_mark < PASS_MARK or avg_gap <= -8 or (avg_z is not None and avg_z <= -0.75)
        strong = avg_mark >= 65 and (avg_gap >= 8 or (avg_z is not None and avg_z >= 0.75))
        types.append({"atype": t["atype"], "avg_mark": round(avg_mark, 1), "gap": round(avg_gap, 1),
                      "z": round(avg_z, 2) if avg_z is not None else None,
                      "level": "Weak" if weak else "Strong" if strong else "Adequate", "weight": t["weight"],
                      "count": len(t["marks"])})
    types.sort(key=lambda c: c["gap"])
    result["types"] = types

    recs = []
    if result["status"] == "At Risk":
        recs.append({"priority": "High", "title": "Early intervention",
                     "detail": "Contact the student this week and consider a referral to Student Learning Support or an academic advisor."})
    if result["status"] == "Borderline":
        recs.append({"priority": "Medium", "title": "Light check in",
                     "detail": "A short conversation or encouraging note now can stop this student slipping. Point them to "
                               "tutorials or consultation hours before the next assessment."})
    if not result["feasible"] and not result["complete"]:
        recs.append({"priority": "High", "title": "Discuss options",
                     "detail": "A pass is no longer possible on current marks. Discuss the student's options with them and their academic advisor as early as possible."})
    if zeros:
        recs.append({"priority": "High", "title": "Missing submissions",
                     "detail": "Follow up on the missing work (" + ", ".join(zeros) + ") and check for circumstances such as illness or access issues."})
    remaining_types = {a.get("atype") or "Other" for a in assessments if a["id"] not in marks}
    for t in types:
        if t["level"] == "Weak":
            title, advice = TYPE_ADVICE.get(t["atype"], TYPE_ADVICE["Other"])
            upcoming_name = {"Coursework": "coursework", "Quiz": "quizzes", "Test": "tests", "Practical": "practical work",
                             "Exam": "the exam", "Other": "similar work"}.get(t["atype"], "similar work")
            still = (f" With {upcoming_name} still to come, there is time to act on this."
                     if t["atype"] in remaining_types else "")
            recs.append({"priority": "Medium" if result["status"] != "At Risk" else "High",
                         "title": f"{title} ({t['atype']}: {t['avg_mark']:.0f}% average, {t['gap']:+.0f} vs class)",
                         "detail": advice + still})
    if result.get("trend", 0) <= -6 and result["graded_count"] >= 3:
        recs.append({"priority": "Medium", "title": "Declining results",
                     "detail": "Results are dropping across recent assessments. A short check in about workload and wellbeing is recommended."})
    req = result["required_for_pass"]
    if req is not None and result["feasible"] and req > 0:
        upcoming = sorted({a.get("atype") or "Other" for a in assessments if a["id"] not in marks})
        recs.append({"priority": "Info", "title": "Target for remaining work",
                     "detail": f"An average of {max(req, 0):.0f}% on the remaining {result['remaining']:.0f}% of the course "
                               f"({', '.join(upcoming).lower()}) secures a pass."})
    strong = [t["atype"] for t in types if t["level"] == "Strong"]
    if strong:
        recs.append({"priority": "Info", "title": "Strengths to build on",
                     "detail": "Performing above the class in: " + ", ".join(strong) + "."})
    result["recommendations"] = recs


def course_summary(course, assessments, students, all_marks, version, bundle):
    cm = CourseModel(course, assessments, students, all_marks, version, bundle)
    rows = []
    to_predict = []
    for i, s in enumerate(students):
        m = all_marks.get(s["student_id"], {})
        if m and len(m) < len(assessments):
            to_predict.append(i)
    preds = cm.predict_rows(np.array(to_predict, dtype=int))
    for i, s in enumerate(students):
        marks = all_marks.get(s["student_id"], {})
        ev = evaluate(course, assessments, marks, cm, preds.get(i))
        weakest = next((t["atype"] for t in ev.get("types", []) if t["level"] == "Weak"), "")
        # label: what lecturers see, the real student ID when the marksheet
        # had one. student_id stays the pseudonymous key used in links.
        rows.append({**s, **ev, "weakest": weakest, "label": s.get("display_id") or s["student_id"]})
    rows.sort(key=lambda r: (STATUS_ORDER[r["status"]], -(r["p_fail"] or 0), r["current_avg"] or 0))

    counts = {k: sum(1 for r in rows if r["status"] == k) for k in ["At Risk", "Borderline", "On Track", "No data"]}
    # How the two checks split the class: arithmetic settles some students
    # outright, the random forest assesses the rest.
    arith_counts = {"Cannot pass": 0, "Pass secured": 0, "Finished": 0, "Still open": 0}
    model_counts = {"At Risk": 0, "Borderline": 0, "On Track": 0}
    for r in rows:
        ch = r.get("checks")
        if not ch:
            continue
        arith_counts[ch["arithmetic"]["verdict"]] += 1
        if not ch["arithmetic"]["decides"]:
            model_counts[r["status"]] = model_counts.get(r["status"], 0) + 1
    evaluated = [r for r in rows if r["status"] != "No data"]
    grade_dist = {g: sum(1 for r in evaluated if r["predicted_grade"] == g) for g in GRADES}
    avgs = [r["current_avg"] for r in evaluated if r["current_avg"] is not None]
    hist = [0] * 10
    for v in avgs:
        hist[min(int(v // 10), 9)] += 1

    per_assessment = []
    for a in assessments:
        vals = [m[a["id"]] for m in all_marks.values() if a["id"] in m]
        per_assessment.append({
            "id": a["id"], "name": a["name"], "atype": a.get("atype"), "weight": a["weight"], "graded": len(vals),
            "class_avg": round(float(np.mean(vals)), 1) if vals else None,
            "below_pass": sum(1 for v in vals if v < PASS_MARK), "zeros": sum(1 for v in vals if v == 0),
        })
    graded_assessments = [p for p in per_assessment if p["graded"]]
    hardest = min(graded_assessments, key=lambda p: p["class_avg"]) if graded_assessments else None

    return {
        "rows": rows, "counts": counts, "n": len(rows), "evaluated": len(evaluated), "grade_dist": grade_dist,
        "pass_rate": round(100 * sum(1 for r in evaluated if r["predicted_grade"] != FAIL_GRADE) / len(evaluated), 1) if evaluated else None,
        "class_avg": round(float(np.mean(avgs)), 1) if avgs else None,
        "hist": {"labels": [f"{b}-{b + 9}" if b < 90 else "90-100" for b in range(0, 100, 10)], "values": hist},
        "per_assessment": per_assessment, "hardest": hardest,
        "scatter": [{"x": r["current_avg"], "y": round((r["p_fail"] or 0) * 100, 1), "label": r["label"], "id": r["student_id"],
                     "status": r["status"]} for r in evaluated if r["current_avg"] is not None and r["p_fail"] is not None],
        "class_avg_by_id": cm.class_avg, "source": cm.source_label, "trained": cm.trained,
        "history": cm.history, "drift": cm.drift(), "arith_counts": arith_counts, "model_counts": model_counts,
    }


def evaluate_single(course, assessments, students, all_marks, version, bundle, marks, label):
    """What-if analysis for one set of marks, compared with the current class.
    Nothing is stored."""
    cm = CourseModel(course, assessments, students, all_marks, version, bundle)
    row = np.array([[marks.get(a["id"], np.nan) for a in assessments]], dtype=float)
    cm.X = np.vstack([cm.X, row]) if len(cm.X) else row
    cm.students = list(students) + [{"student_id": label}]
    cm.F = fx.current_features(assessments, cm.X, cm.history["pass_rate"])
    i = len(cm.X) - 1
    pred = None
    if marks and len(marks) < len(assessments):
        pred = cm.predict_rows(np.array([i]))[i]
    return evaluate(course, assessments, marks, cm, pred), cm


_summaries = {}
# Bump when the summary's contents change shape, so stored copies are rebuilt.
SUMMARY_FORMAT = 9


def cached_summary(course_id):
    """Predictions are expensive to compute, so a course's summary is kept
    in memory and in the database (summary_cache) under a key that changes
    whenever its current data, the active model, its thresholds or its
    assessment types change. Opening a page then never reruns the model."""
    key = f"f{SUMMARY_FORMAT}|" + db.course_cache_key(course_id)
    hit = _summaries.get(course_id)
    if hit and hit[0] == key:
        return hit[1]
    summary = None
    blob = db.get_summary_cache(course_id, key)
    if blob:
        try:
            summary = pickle.loads(zlib.decompress(blob))
        except Exception:
            summary = None
    if summary is None:
        course = db.get_course(course_id)
        version = db.get_active_model()
        bundle = ml_engine.load_bundle(version)
        summary = course_summary(course, db.get_assessments(course_id), db.get_students(course_id),
                                 db.get_marks(course_id), version, bundle)
        db.put_summary_cache(course_id, key, zlib.compress(pickle.dumps(summary), 6))
    _summaries[course_id] = (key, summary)
    return summary
