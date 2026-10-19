"""
features.py
Structure independent features. Every course, whatever its assessments,
is described by the same set of numbers, so one model can learn from all
courses at once and score a course it has never seen.

Each student is described at a point in time by which assessments have
been marked so far (the "assessed mask"). For past cohorts, training rows
are snapshots after the first k assessments in chronological order, for
every k, so the model learns to predict from any stage of a course.
"""

import warnings

import numpy as np

from config import PASS_MARK, MAX_STAGES_PER_STUDENT

TYPE_FEATURES = [("Coursework", "type_avg_coursework"), ("Quiz", "type_avg_quiz"), ("Test", "type_avg_test"),
                 ("Practical", "type_avg_practical"), ("Exam", "type_avg_exam")]

FEATURES = ["weighted_avg", "prop_assessed", "earned", "best_possible", "req_pass",
            "mean_gap", "mean_z", "min_mark", "max_mark", "sd_mark", "share_below_50",
            "trend", "trend_z", "n_zeros"] + [f for _, f in TYPE_FEATURES] + \
           ["class_avg", "hist_pass_rate", "n_assessments", "cohort_size"]

LABELS = {
    "weighted_avg": "Weighted average so far", "prop_assessed": "Share of course assessed",
    "earned": "Final mark banked", "best_possible": "Best possible final mark",
    "req_pass": "Average needed to pass", "mean_gap": "Gap to class mean", "mean_z": "Standing in class (z score)",
    "min_mark": "Lowest mark", "max_mark": "Highest mark", "sd_mark": "Spread of marks",
    "share_below_50": "Weight scored below 50%", "trend": "Trend in marks", "trend_z": "Trend relative to class",
    "n_zeros": "Missed submissions", "type_avg_coursework": "Coursework average", "type_avg_quiz": "Quiz average",
    "type_avg_test": "Test average", "type_avg_practical": "Practical average", "type_avg_exam": "Exam average",
    "class_avg": "Class average", "hist_pass_rate": "Course pass rate in past years",
    "n_assessments": "Number of assessments", "cohort_size": "Class size",
}

# Features compared for drift, and course level context checked for range.
DRIFT_FEATURES = ["weighted_avg", "sd_mark", "share_below_50", "min_mark"]
CONTEXT_FEATURES = ["class_avg", "hist_pass_rate", "n_assessments", "cohort_size"]


def compute(X, weights, types, mask, class_mean, class_sd, context):
    """X: students x assessments (percent, NaN = no mark), columns in
    chronological order. mask: which cells count as assessed. Returns a
    float matrix with columns in FEATURES order."""
    X = np.asarray(X, dtype=float)
    W = np.asarray(weights, dtype=float)
    A = np.asarray(mask, dtype=bool) & ~np.isnan(X)
    n, m = X.shape
    total = W.sum() or 100.0
    pos = np.broadcast_to(np.arange(m, dtype=float), (n, m))

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        Xm = np.where(A, X, np.nan)
        wsum = (A * W).sum(axis=1)
        wx = np.nansum(Xm * W, axis=1)
        weighted_avg = np.where(wsum > 0, wx / np.where(wsum > 0, wsum, 1), np.nan)
        prop = wsum / total
        earned = wx / total
        remaining = (total - wsum) / total * 100.0
        best = earned + remaining
        req = np.where(remaining > 0.001, (PASS_MARK - earned) / np.where(remaining > 0.001, remaining, 1) * 100.0, np.nan)
        req = np.clip(req, -100, 200)

        cm = np.asarray(class_mean, dtype=float)
        cs = np.maximum(np.nan_to_num(np.asarray(class_sd, dtype=float), nan=10.0), 3.0)
        gap = Xm - cm
        z = gap / cs
        mean_gap = np.nanmean(gap, axis=1)
        mean_z = np.nanmean(z, axis=1)
        count = A.sum(axis=1)
        mn = np.nanmin(np.where(A, X, np.inf), axis=1)
        mx = np.nanmax(np.where(A, X, -np.inf), axis=1)
        mn = np.where(count > 0, mn, np.nan)
        mx = np.where(count > 0, mx, np.nan)
        sd = np.where(count >= 2, np.nanstd(Xm, axis=1), np.nan)
        below = np.where(wsum > 0, ((Xm < PASS_MARK) & A).astype(float) @ W / np.where(wsum > 0, wsum, 1), np.nan)
        trend = _slope(pos, Xm, A)
        trend_z = _slope(pos, z, A)
        zeros = ((Xm == 0) & A).sum(axis=1).astype(float)

        type_cols = []
        types = np.asarray(types)
        for t, _ in TYPE_FEATURES:
            sel = (types == t)
            At = A & sel
            wt = (At * W).sum(axis=1)
            type_cols.append(np.where(wt > 0, np.nansum(np.where(At, X, np.nan) * W, axis=1) / np.where(wt > 0, wt, 1), np.nan))

        valid_avg = weighted_avg[~np.isnan(weighted_avg)]
        class_avg = float(valid_avg.mean()) if len(valid_avg) else np.nan

    cols = [weighted_avg, prop, earned, best, req, mean_gap, mean_z, mn, mx, sd, below, trend, trend_z, zeros] + type_cols
    cols += [np.full(n, class_avg),
             np.full(n, np.nan if context.get("hist_pass_rate") is None else context["hist_pass_rate"]),
             np.full(n, float(m)),
             np.full(n, float(context.get("cohort_size") or n))]
    return np.column_stack(cols).astype(float)


def _slope(P, Y, A):
    c = A.sum(axis=1)
    Pm = np.where(A, P, np.nan)
    Ym = np.where(A, Y, np.nan)
    pbar = np.nanmean(Pm, axis=1, keepdims=True)
    ybar = np.nanmean(Ym, axis=1, keepdims=True)
    num = np.nansum((Pm - pbar) * (Ym - ybar), axis=1)
    den = np.nansum((Pm - pbar) ** 2, axis=1)
    return np.where((c >= 2) & (den > 0), num / np.where(den > 0, den, 1), np.nan)


def class_stats(X):
    X = np.asarray(X, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(X, axis=0), np.nanstd(X, axis=0)


def matrix_for_course(assessments, students, marks):
    """Current cohort as a matrix, in the course's chronological order."""
    X = np.full((len(students), len(assessments)), np.nan)
    for i, s in enumerate(students):
        sm = marks.get(s["student_id"], {})
        for j, a in enumerate(assessments):
            if a["id"] in sm:
                X[i, j] = sm[a["id"]]
    return X


def current_features(assessments, X, hist_pass_rate):
    """Features for a current cohort: each student at their own stage."""
    weights = [a["weight"] for a in assessments]
    types = [a.get("atype") or "Other" for a in assessments]
    mean, sd = class_stats(X)
    for j, a in enumerate(assessments):
        if np.isnan(mean[j]) and a.get("reported_class_avg") is not None:
            mean[j] = a["reported_class_avg"] / (a.get("max_mark") or 100) * 100
    ctx = {"hist_pass_rate": hist_pass_rate, "cohort_size": len(X)}
    return compute(X, weights, types, ~np.isnan(X), mean, sd, ctx)


def training_set(history):
    """history: output of database.historical_training_data(). Returns a dict
    with the feature matrix, labels, groups and per row stage information."""
    rows_X, fail, grade, course, cohort, student, weight, prop = [], [], [], [], [], [], [], []

    # Pass rate per (course, cohort), so each cohort's context uses only the
    # OTHER cohorts of its course (no label leakage).
    outcomes = {}
    for d in history:
        for r in d["rows"]:
            if r["final_grade"]:
                key = (d["course_code"], r["cohort"] or d["label"])
                o = outcomes.setdefault(key, [0, 0])
                o[0] += 1
                o[1] += r["final_grade"] == "D"

    for d in history:
        structure = d["structure"]
        weights = [a["weight"] for a in structure]
        types = [a.get("atype") or "Other" for a in structure]
        n_a = len(structure)
        by_cohort = {}
        for r in d["rows"]:
            if r["final_grade"]:
                by_cohort.setdefault(r["cohort"] or d["label"], []).append(r)
        for coh, rows in by_cohort.items():
            others = [v for (c, k), v in outcomes.items() if c == d["course_code"] and k != coh]
            n_other = sum(v[0] for v in others)
            hist_pass = (1 - sum(v[1] for v in others) / n_other) if n_other else None
            X = np.array([[np.nan if m is None else m for m in r["marks"]] for r in rows], dtype=float)
            mean, sd = class_stats(X)
            ctx = {"hist_pass_rate": hist_pass, "cohort_size": len(rows)}
            stages = list(range(1, n_a)) if n_a > 1 else [1]
            if len(stages) > MAX_STAGES_PER_STUDENT:
                stages = sorted({int(round(v)) for v in np.linspace(1, n_a - 1, MAX_STAGES_PER_STUDENT)})
            n_stages = len(stages)
            for k in stages:
                mask = np.zeros_like(X, dtype=bool)
                mask[:, :k] = True
                F = compute(X, weights, types, mask, mean, sd, ctx)
                ok = ~np.isnan(F[:, 0])
                for i in np.where(ok)[0]:
                    rows_X.append(F[i])
                    fail.append(1 if rows[i]["final_grade"] == "D" else 0)
                    grade.append(rows[i]["final_grade"])
                    course.append(d["course_code"])
                    cohort.append(f"{d['course_code']}|{coh}")
                    student.append(f"{d['course_code']}|{coh}|{rows[i]['pseudo_id']}")
                    weight.append(1.0 / n_stages)
                    prop.append(F[i, FEATURES.index("prop_assessed")])

    if not rows_X:
        return None
    return {"X": np.vstack(rows_X), "y_fail": np.array(fail), "y_grade": np.array(grade),
            "course": np.array(course), "cohort": np.array(cohort), "student": np.array(student),
            "weight": np.array(weight), "prop": np.array(prop)}


# ---------------- plain language ----------------

def describe(feature, value, contribution):
    """One sentence for a SHAP contribution. contribution > 0 raises risk."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    up = contribution > 0
    effect = "raises the risk" if up else "lowers the risk"
    v = float(value)
    texts = {
        "weighted_avg": f"Weighted average so far is {v:.0f}%",
        "prop_assessed": f"{v * 100:.0f}% of the course has been assessed so far",
        "earned": f"{v:.0f}% of the final mark is already banked",
        "best_possible": f"The best possible final mark is {v:.0f}%",
        "req_pass": (f"Needs an average of {v:.0f}% on the remaining work to pass" if v > 0
                     else "A pass is already secured on marks banked"),
        "mean_gap": f"Marks are on average {abs(v):.0f} points {'below' if v < 0 else 'above'} the class mean",
        "mean_z": f"Standing in class is {abs(v):.1f} standard deviations {'below' if v < 0 else 'above'} average",
        "min_mark": f"Lowest mark so far is {v:.0f}%",
        "max_mark": f"Highest mark so far is {v:.0f}%",
        "sd_mark": f"Marks vary by about {v:.0f} points between assessments",
        "share_below_50": f"{v * 100:.0f}% of the assessed weight was scored below 50%",
        "trend": (f"Marks are {'falling' if v < 0 else 'rising'} by about {abs(v):.0f} points per assessment"
                  if abs(v) >= 1 else "Marks are steady across assessments"),
        "trend_z": f"Position relative to the class is {'slipping' if v < 0 else 'improving'} over time",
        "n_zeros": f"{v:.0f} assessment(s) not submitted" if v else "No missed submissions",
        "class_avg": f"The class as a whole is averaging {v:.0f}%",
        "hist_pass_rate": f"This course has a past pass rate of {v * 100:.0f}%",
        "n_assessments": f"The course has {v:.0f} assessments",
        "cohort_size": f"Class size is {v:.0f}",
    }
    for t, f in TYPE_FEATURES:
        texts[f] = f"{t} average is {v:.0f}%"
    text = texts.get(feature)
    return f"{text}, which {effect}." if text else None
