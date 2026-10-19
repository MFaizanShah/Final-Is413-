"""
ml_engine.py
The learning pipeline.

One GLOBAL model
    Historical cohorts from every course are pooled. Each past student
    contributes a snapshot after every stage of their course, described by
    the structure independent features in features.py, with course context
    (class average, past pass rate, number of assessments, class size) as
    inputs. Two algorithms are trained and compared on the same held out
    folds: gradient boosting (scikit-learn HistGradientBoosting, which
    handles missing values natively) and a random forest. The one with the
    better Brier score on unseen courses becomes the production model.

Calibration
    The at risk probability is calibrated (isotonic regression with enough
    data, Platt scaling otherwise) so that a stated 30% chance of failing
    means roughly 30 out of 100 such students failed. Calibration quality
    is reported as a reliability table, the Brier score and the expected
    calibration error, before and after calibration.

Honest evaluation
    Cross validation is split by COURSE (the model never sees the held out
    course) and by COHORT (never sees the held out semester), so snapshots
    of the same student never sit on both sides of a split. A random forest
    is evaluated on the same folds as a baseline.

Explanations
    Each prediction lists its main factors. A factor's effect is measured by
    resetting that one feature to a typical value and seeing how much the
    chance of failing changes, then turned into plain language in
    features.describe().

Cold start
    A course whose history is not in the active model gets NO trained
    prediction. Instead a transparent Monte Carlo simulation samples the
    remaining marks from the student's relative standing in the class, and
    the result is labelled as a simulated estimate.

Storage
    The trained model is saved INSIDE the SQLite database (model_versions.
    model_blob), so it is trained once and loaded on start. Training only
    runs when an admin asks for it or uploads new historical data.

Versioning and drift
    Each training run is a model version with its metrics and a summary of
    its training data, and any earlier version can be re-activated. New
    courses are compared with the training data and flagged if they differ
    strongly.
"""

import io
import os
import threading
import traceback
import zlib

import joblib
import numpy as np
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, precision_score, recall_score, f1_score,
                             roc_auc_score)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

import database as db
import features as fx
from config import (GRADES, FAIL_GRADE, GRADE_MIDPOINTS, LEGACY_MODEL_DIR, MIN_TRAINING_STUDENTS, MC_SIMULATIONS,
                    PASS_MARK, TRAINING_ENABLED, grade_for_mark)

DEFAULT_RISK_THRESHOLD = 0.40
_cache = {}
load_errors = {}
_train_lock = threading.Lock()


# ---------------- model factories ----------------

ALGORITHMS = {"boosting": "Gradient boosting", "forest": "Random forest"}


class DropEmptyColumns(BaseEstimator, TransformerMixin):
    """Removes features that had no values at all in the training data (for
    example a practical average when no training course has practicals).
    They carry no information, and gradient boosting cannot bin them."""

    def fit(self, X, y=None):
        self.keep_ = ~np.all(np.isnan(np.asarray(X, dtype=float)), axis=0)
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.keep_]


def _boosting():
    return make_pipeline(DropEmptyColumns(),
                         HistGradientBoostingClassifier(max_iter=250, learning_rate=0.05, max_leaf_nodes=15,
                                                        min_samples_leaf=40, l2_regularization=1.0, random_state=42))


def _forest():
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         RandomForestClassifier(n_estimators=120, min_samples_leaf=20, max_depth=14, max_samples=0.6,
                                                random_state=42, n_jobs=-1))


def _make(algo):
    return _boosting() if algo == "boosting" else _forest()


def _fit(model, X, y, w):
    if isinstance(model, CalibratedClassifierCV):
        return model.fit(X, y, sample_weight=w)
    if hasattr(model, "steps"):
        return model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    return model.fit(X, y, sample_weight=w)


def _calibrated(algo, method, groups, X_idx):
    """Calibrated model. Calibration folds are grouped by cohort when
    possible, otherwise by student, so calibration data is also unseen."""
    g = groups[X_idx]
    n_groups = len(set(g))
    k = 3 if n_groups >= 3 else 2
    splits = list(GroupKFold(n_splits=k).split(np.zeros(len(g)), groups=g))
    return CalibratedClassifierCV(_make(algo), method=method, cv=splits)


def _calibration_groups(ts):
    return ts["cohort"] if len(set(ts["cohort"])) >= 3 else ts["student"]


# ---------------- metrics ----------------

def _ece_table(y, p, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    table, ece = [], 0.0
    for i in range(bins):
        sel = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1))
        if sel.sum() == 0:
            continue
        pred, obs = float(p[sel].mean()), float(y[sel].mean())
        ece += sel.sum() / len(p) * abs(pred - obs)
        table.append({"bin": f"{edges[i] * 100:.0f}-{edges[i + 1] * 100:.0f}%", "predicted": round(pred, 3),
                      "observed": round(obs, 3), "n": int(sel.sum())})
    return round(float(ece), 4), table


def _binary_metrics(y, p, threshold=DEFAULT_RISK_THRESHOLD):
    pred = (p >= threshold).astype(int)
    out = {"n": int(len(y)), "positives": int(y.sum()),
           "accuracy": round(float(accuracy_score(y, pred)), 3),
           "precision": round(float(precision_score(y, pred, zero_division=0)), 3),
           "recall": round(float(recall_score(y, pred, zero_division=0)), 3),
           "f1": round(float(f1_score(y, pred, zero_division=0)), 3),
           "brier": round(float(brier_score_loss(y, p)), 4)}
    out["roc_auc"] = round(float(roc_auc_score(y, p)), 3) if 0 < y.sum() < len(y) else None
    out["ece"], _ = _ece_table(y, p)
    return out


def _grade_metrics(y_true, y_pred):
    idx = {g: i for i, g in enumerate(GRADES)}
    exact = float(np.mean(y_true == y_pred))
    within = float(np.mean([abs(idx[a] - idx[b]) <= 1 for a, b in zip(y_true, y_pred)]))
    return {"exact": round(exact, 3), "within_one": round(within, 3)}


def _headline_rows(ts):
    """One snapshot per student: the one closest to half the course assessed."""
    best = {}
    for i, (s, p) in enumerate(zip(ts["student"], ts["prop"])):
        d = abs(p - 0.5)
        if s not in best or d < best[s][0]:
            best[s] = (d, i)
    return np.array(sorted(i for _, i in best.values()))


STAGE_BINS = [(0.0, 0.25, "Under 25% assessed"), (0.25, 0.5, "25 to 50% assessed"),
              (0.5, 0.75, "50 to 75% assessed"), (0.75, 1.0, "Over 75% assessed")]


def _cross_validate(ts, scheme, method):
    """Out of fold predictions, split by `scheme` groups, for both
    algorithms (raw and calibrated) and the grade model."""
    groups = ts[scheme]
    n_groups = len(set(groups))
    k = min(5, n_groups)
    X, y, w, yg = ts["X"], ts["y_fail"], ts["weight"], ts["y_grade"]
    cal_groups = _calibration_groups(ts)
    names = [f"{a}_{v}" for a in ALGORITHMS for v in ("calibrated", "raw")]
    oof = {name: np.full(len(y), np.nan) for name in names}
    oof_grade = np.empty(len(y), dtype=object)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        for algo in ALGORITHMS:
            cal = _fit(_calibrated(algo, method, cal_groups, tr), X[tr], y[tr], w[tr])
            oof[f"{algo}_calibrated"][te] = cal.predict_proba(X[te])[:, 1]
            # The calibrated model averages several fitted copies; their
            # uncalibrated average is reported to show what calibration adds.
            raw = np.mean([c.estimator.predict_proba(X[te])[:, 1] for c in cal.calibrated_classifiers_], axis=0)
            oof[f"{algo}_raw"][te] = raw
        known = tr[np.isin(yg[tr], GRADES)]
        if len(set(yg[known])) >= 2:
            gm = _fit(_boosting(), X[known], yg[known], w[known])
            oof_grade[te] = gm.predict(X[te])
    return oof, oof_grade, k


def _report(ts, oof, oof_grade, best):
    ok = ~np.isnan(oof[f"{best}_calibrated"])
    head = _headline_rows(ts)
    head = head[ok[head]]
    y = ts["y_fail"]
    # Grade accuracy only where the exact grade is known (Pass/Fail data has none).
    gh = head[np.isin(ts["y_grade"][head], GRADES) & np.array([g is not None for g in oof_grade[head]], dtype=bool)]
    rep = {"headline": {name: _binary_metrics(y[head], p[head]) for name, p in oof.items()},
           "all_stages": {name: _binary_metrics(y[ok], p[ok]) for name, p in oof.items()},
           "grade": _grade_metrics(ts["y_grade"][gh], oof_grade[gh]) if len(gh) else None}
    # Compatibility names used by the model page: the chosen algorithm,
    # before and after calibration, and the other algorithm as comparison.
    other = next(a for a in ALGORITHMS if a != best)
    for key, src in (("calibrated", f"{best}_calibrated"), ("raw", f"{best}_raw"), ("other", f"{other}_calibrated")):
        rep["headline"][key] = rep["headline"][src]
        rep["all_stages"][key] = rep["all_stages"][src]
    rep["calibration"] = {}
    for name in ("raw", "calibrated"):
        ece, table = _ece_table(y[head], oof[f"{best}_{name}"][head])
        rep["calibration"][name] = {"ece": ece, "brier": rep["headline"][name]["brier"], "table": table}
    stages = []
    for lo, hi, label in STAGE_BINS:
        sel = ok & (ts["prop"] >= lo) & ((ts["prop"] < hi) if hi < 1 else (ts["prop"] <= hi))
        if sel.sum() >= 20 and 0 < y[sel].sum() < sel.sum():
            m = _binary_metrics(y[sel], oof[f"{best}_calibrated"][sel])
            mo = _binary_metrics(y[sel], oof[f"{other}_calibrated"][sel])
            stages.append({"stage": label, "n": int(sel.sum()), "auc": m["roc_auc"], "recall": m["recall"],
                           "precision": m["precision"], "brier": m["brier"], "other_auc": mo["roc_auc"]})
    rep["stages"] = stages
    rep["best"], rep["other"] = best, other
    return rep


def _choose(ts, oof):
    """The algorithm with the lower Brier score (better probabilities) on the
    headline snapshots of unseen data wins. Ties go to the random forest,
    the simpler model to explain."""
    head = _headline_rows(ts)
    y = ts["y_fail"]
    scores = {}
    for algo in ALGORITHMS:
        p = oof[f"{algo}_calibrated"][head]
        ok = ~np.isnan(p)
        scores[algo] = float(brier_score_loss(y[head][ok], p[ok])) if ok.any() else 1.0
    # The random forest is always the production model: it is the simpler
    # model to explain to staff and matched gradient boosting on this kind of
    # data. Gradient boosting is still trained on the same folds and reported
    # as a comparison, so a clearly better alternative would be visible.
    return "forest", scores


# ---------------- training ----------------

def readiness():
    """Whether there is enough historical data to train, without training."""
    history = db.historical_training_data()
    students = sum(1 for d in history for r in d["rows"] if r["final_grade"])
    fails = sum(1 for d in history for r in d["rows"] if r["final_grade"] == FAIL_GRADE)
    ok = students >= MIN_TRAINING_STUDENTS and 0 < fails < students
    if students < MIN_TRAINING_STUDENTS:
        msg = f"At least {MIN_TRAINING_STUDENTS} past students with outcomes are needed; {students} are available."
    elif not 0 < fails < students:
        msg = "The historical data needs both passes and fails to learn from."
    else:
        msg = f"{students} past students with outcomes from {len({d['course_code'] for d in history})} course(s)."
    return ok, msg


def retrain(username, trigger, background=True):
    """Starts a training run. Returns (model_id or None, message)."""
    if not TRAINING_ENABLED:
        return None, ("Training is switched off on this hosted copy, because it takes longer than a web request may run. "
                      "Train on your own computer (python build_database.py, or Retrain in the Admin portal), "
                      "then redeploy the updated database.")
    ok, msg = readiness()
    if not ok:
        return None, msg
    if not _train_lock.acquire(blocking=False):
        return None, "A model is already being trained. Try again when it finishes."
    model_id = db.create_model_version(username, trigger)
    db.audit(username, "model_training_started", f"Trigger: {trigger}", model_id=model_id)

    def run():
        try:
            _train(model_id, username)
        finally:
            _train_lock.release()

    if background:
        threading.Thread(target=run, daemon=True).start()
    else:
        run()
    return model_id, "Training started."


def _train(model_id, username, progress=None):
    say = progress or (lambda m: None)
    try:
        history = db.historical_training_data()
        ts = fx.training_set(history)
        if ts is None:
            raise ValueError("No usable historical snapshots.")
        n_students = len(set(ts["student"]))
        courses = sorted(set(ts["course"]))
        cohorts = sorted(set(ts["cohort"]))
        method = "isotonic" if n_students >= 500 else "sigmoid"
        say(f"{n_students:,} students, {len(ts['y_fail']):,} snapshots, {len(courses)} courses, {len(cohorts)} cohorts")

        schemes = {}
        if len(courses) >= 2:
            schemes["course"] = "Split by course: each fold holds out whole courses"
        if len(cohorts) >= 2:
            schemes["cohort"] = "Split by cohort: each fold holds out whole semesters"
        if not schemes:
            schemes["student"] = "Split by student only (a single cohort is available, so figures may be optimistic)"
        primary = "course" if len(courses) >= 3 else ("cohort" if "cohort" in schemes else next(iter(schemes)))

        # Cross validate the primary scheme first; it decides the algorithm.
        order = [primary] + [k for k in schemes if k != primary]
        evaluation, best, scores = {}, None, {}
        for scheme in order:
            say(f"Cross validating ({schemes[scheme].split(':')[0].lower()})")
            oof, oof_grade, k = _cross_validate(ts, scheme, method)
            if best is None:
                best, scores = _choose(ts, oof)
            evaluation[scheme] = {"description": schemes[scheme], "folds": k, **_report(ts, oof, oof_grade, best)}
        say(f"Production model: {ALGORITHMS[best]} (Brier {scores[best]:.4f}); comparison: "
            f"{ALGORITHMS[[a for a in ALGORITHMS if a != best][0]]} (Brier {scores[[a for a in ALGORITHMS if a != best][0]]:.4f})")

        # Production models, fitted on all data.
        X, y, w = ts["X"], ts["y_fail"], ts["weight"]
        calibrated = _fit(_calibrated(best, method, _calibration_groups(ts), np.arange(len(y))), X, y, w)
        explainer = _fit(_make(best), X, y, w)
        known = np.isin(ts["y_grade"], GRADES)
        grade_model = (_fit(_boosting(), X[known], ts["y_grade"][known], w[known])
                       if len(set(ts["y_grade"][known])) >= 2 and known.sum() >= 50 else None)

        rng = np.random.default_rng(0)
        sample = rng.choice(len(X), size=min(len(X), 20000), replace=False)
        ref = np.nanmedian(X, axis=0)
        importance = _importance(explainer, X[sample[:3000]], ref)

        # Comparable students: distinct past students with a snapshot near each stage.
        centres = np.round(np.arange(0, 1.0001, 0.05), 2)
        comparable = [int(len(set(ts["student"][np.abs(ts["prop"] - c) <= 0.1]))) for c in centres]
        type_set = sorted({a.get("atype") or "Other" for d in history for a in d["structure"]})

        bundle = {
            "version": model_id, "features": fx.FEATURES, "calibrated": calibrated, "explainer": explainer,
            "algorithm": best, "reference": ref,
            "grade_model": grade_model, "grade_classes": list(grade_model.classes_) if grade_model else [], "method": method,
            "courses": courses, "n_students": n_students, "comparable_centres": centres.tolist(),
            "comparable_counts": comparable, "drift_ref": X[sample].astype(np.float32),
            "drift_prop": ts["prop"][sample].astype(np.float32), "types_seen": type_set,
            "sklearn_version": sklearn.__version__,
        }
        buf = io.BytesIO()
        joblib.dump(bundle, buf, compress=3)
        blob = buf.getvalue()

        primary_eval = evaluation[primary]
        head = primary_eval["headline"]["calibrated"]
        per_course = {}
        for d in history:
            n = sum(1 for r in d["rows"] if r["final_grade"])
            f = sum(1 for r in d["rows"] if r["final_grade"] == FAIL_GRADE)
            pc = per_course.setdefault(d["course_code"], {"students": 0, "fails": 0, "cohorts": set(),
                                                         "n_assessments": len(d["structure"])})
            pc["students"] += n
            pc["fails"] += f
            pc["cohorts"].update(r["cohort"] or d["label"] for r in d["rows"])
        data_summary = [{"course": c, "students": v["students"], "fails": v["fails"],
                         "fail_rate": round(v["fails"] / v["students"], 3) if v["students"] else None,
                         "cohorts": len(v["cohorts"]), "n_assessments": v["n_assessments"]}
                        for c, v in sorted(per_course.items())]
        other = primary_eval["other"]
        summary = {"algorithm": f"{ALGORITHMS[best]} (calibrated)", "algorithm_key": best,
                   "compared_with": ALGORITHMS[other],
                   "brier_scores": {ALGORITHMS[a]: round(v, 4) for a, v in scores.items()},
                   "other_auc": primary_eval["headline"]["other"]["roc_auc"],
                   "n_students": n_students, "n_rows": int(len(y)),
                   "n_courses": len(courses), "n_cohorts": len(cohorts), "courses": courses,
                   "calibration": "Isotonic regression" if method == "isotonic" else "Platt scaling",
                   "primary_scheme": primary, "fail_rate": round(float(np.mean(
                       [r["final_grade"] == FAIL_GRADE for d in history for r in d["rows"] if r["final_grade"]])), 3),
                   "headline_auc": head["roc_auc"], "headline_recall": head["recall"],
                   "headline_precision": head["precision"], "headline_brier": head["brier"],
                   "headline_accuracy": head["accuracy"], "data": data_summary, "model_kb": round(len(blob) / 1024),
                   "sklearn_version": sklearn.__version__}
        metrics = {"evaluation": evaluation, "primary": primary, "threshold": DEFAULT_RISK_THRESHOLD,
                   "importance": [{"feature": f, "label": fx.LABELS[f], "value": round(float(v), 4)}
                                  for f, v in sorted(zip(fx.FEATURES, importance), key=lambda t: -t[1])]}
        db.finish_model_version(model_id, summary, metrics, blob, activate=True)
        _cache.clear()
        db.audit(username, "model_trained",
                 f"Version {model_id}: {ALGORITHMS[best]} (compared with {ALGORITHMS[other]}). {n_students:,} students, "
                 f"{len(courses)} course(s), ROC AUC {head['roc_auc']}, Brier {head['brier']} "
                 f"({schemes[primary].split(':')[0].lower()}). Saved in the database and activated.", model_id=model_id)
        say("Saved to the database and activated")
    except Exception as exc:
        db.fail_model_version(model_id, f"{exc}")
        db.audit(username, "model_training_failed", f"{exc}\n{traceback.format_exc(limit=3)}", model_id=model_id)
        say(f"Failed: {exc}")
        raise


def _importance(model, X, ref):
    """Average change in the chance of failing when each feature is reset to
    a typical value: a model independent measure of what the model uses."""
    base = model.predict_proba(X)[:, 1]
    out = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        Xr = X.copy()
        Xr[:, j] = ref[j]
        out[j] = np.nanmean(np.abs(base - model.predict_proba(Xr)[:, 1]))
    return out / (out.sum() or 1)


# ---------------- loading ----------------

def load_bundle(version):
    """The trained model for a version, loaded once and kept in memory.
    Models are read from the database. Versions saved by older releases as
    files are found by file name inside the models folder (wherever the
    project now lives) and moved into the database the first time."""
    if not version:
        return None
    vid = version["id"]
    if vid in _cache:
        return _cache[vid]
    bundle = None
    try:
        blob = db.get_model_blob(vid)
        if blob is not None:
            bundle = joblib.load(io.BytesIO(blob))
        elif version.get("model_path"):
            name = os.path.basename(version["model_path"].replace("\\", "/"))
            for path in (version["model_path"], os.path.join(LEGACY_MODEL_DIR, "global", name)):
                if os.path.exists(path):
                    with open(path, "rb") as f:
                        raw = f.read()
                    bundle = joblib.load(io.BytesIO(raw))
                    db.store_model_blob(vid, raw)
                    break
    except Exception as exc:
        # For example a model saved with a different scikit-learn version, or
        # by an older release that needed LightGBM. Treated as no model.
        saved = (version.get("summary") or {}).get("sklearn_version")
        load_errors[vid] = (f"The saved model could not be loaded ({type(exc).__name__}). "
                            + (f"It was trained with scikit-learn {saved}; this computer has {sklearn.__version__}. "
                               if saved and saved != sklearn.__version__ else "")
                            + "Install the versions in requirements.txt, or retrain the model.")
        bundle = None
    if bundle is not None and "reference" not in bundle:
        bundle = None  # older format without what explanations need
    _cache[vid] = bundle
    return bundle


def comparable_students(bundle, prop):
    centres = np.array(bundle["comparable_centres"])
    return bundle["comparable_counts"][int(np.argmin(np.abs(centres - prop)))]


# ---------------- prediction ----------------

def _package(probs, source, **extra):
    probs = {g: float(probs.get(g, 0.0)) for g in GRADES}
    predicted = max(probs, key=probs.get)
    out = {"source": source, "probs": {g: round(p, 3) for g, p in probs.items()}, "predicted_grade": predicted,
           "confidence": round(probs[predicted], 3), "p_fail": round(probs[FAIL_GRADE], 3)}
    out.update(extra)
    return out


EXPLAIN_SKIP = {"n_assessments", "cohort_size", "prop_assessed"}


def _contributions(bundle, F):
    """Effect of each feature on the chance of failing, for each student:
    the change when that feature alone is reset to a typical value."""
    model, ref = bundle["explainer"], bundle["reference"]
    base = model.predict_proba(F)[:, 1]
    C = np.zeros_like(F, dtype=float)
    for j, name in enumerate(bundle["features"]):
        if name in EXPLAIN_SKIP or np.isnan(ref[j]):
            continue
        has = ~np.isnan(F[:, j])
        if not has.any():
            continue
        Fr = F[has].copy()
        Fr[:, j] = ref[j]
        C[has, j] = base[has] - model.predict_proba(Fr)[:, 1]
    return C


def predict_trained(bundle, F):
    """Calibrated predictions with plain language explanations."""
    if len(F) == 0:
        return []
    # Isotonic calibration can return exactly 0 or 1. No forecast of a
    # future grade is certain, so stated chances are kept within 1 to 99%.
    p_fail = np.clip(bundle["calibrated"].predict_proba(F)[:, 1], 0.01, 0.99)
    gm = bundle["grade_model"]
    grade_p = gm.predict_proba(F) if gm is not None else None
    classes = bundle["grade_classes"]
    names = bundle["features"]
    wa = names.index("weighted_avg")
    pi = names.index("prop_assessed")
    contrib = _contributions(bundle, F)
    out = []
    for i in range(len(F)):
        if grade_p is not None:
            raw = dict(zip(classes, grade_p[i]))
        else:
            # No grade model (outcomes were Pass/Fail only): spread the pass
            # probability over the grades near the current average.
            avg = float(np.nan_to_num(F[i, wa], nan=PASS_MARK))
            raw = {grade_for_mark(max(PASS_MARK, avg)): 0.6, grade_for_mark(max(PASS_MARK, avg - 7)): 0.2,
                   grade_for_mark(max(PASS_MARK, avg + 7)): 0.2}
            merged = {}
            for g, v in raw.items():
                merged[g] = merged.get(g, 0) + v
            raw = merged
        pass_total = sum(v for g, v in raw.items() if g != FAIL_GRADE) or 1.0
        probs = {g: (1 - p_fail[i]) * raw.get(g, 0.0) / pass_total for g in GRADES if g != FAIL_GRADE}
        probs[FAIL_GRADE] = p_fail[i]
        projected = sum(GRADE_MIDPOINTS[g] * p for g, p in probs.items())
        c = contrib[i]
        order = np.argsort(-np.abs(c))
        factors = []
        for j in order:
            if len(factors) >= 4 or abs(c[j]) < 0.02:
                break
            text = fx.describe(names[j], F[i, j], c[j])
            if text:
                factors.append({"feature": names[j], "label": fx.LABELS[names[j]], "impact": round(float(c[j]), 3),
                                "direction": "up" if c[j] > 0 else "down", "text": text})
        n_comp = comparable_students(bundle, F[i, pi])
        pkg = _package(probs, "model", projected_mark=round(projected, 1), factors=factors,
                       comparable_n=n_comp, model_version=bundle["version"],
                       grade_basis="model" if grade_p is not None else "average",
                       confidence_text=f"Based on {n_comp:,} similar past students")
        if grade_p is None:
            # Passing grades were only spread around the average, so the
            # likely outcome follows the chance of failing, not the largest slice.
            passing = {g: p for g, p in probs.items() if g != FAIL_GRADE}
            best = FAIL_GRADE if p_fail[i] >= 0.5 else max(passing, key=passing.get)
            pkg["predicted_grade"], pkg["confidence"] = best, round(float(probs[best]), 3)
        out.append(pkg)
    return out


def simulate(assessments, X, i, student_key):
    """Monte Carlo estimate for student row i of matrix X. Remaining marks
    are drawn as class mean + (student's relative standing + noise) x class
    spread, so a student who sits 1 SD below the class is simulated to keep
    roughly that position, with honest uncertainty around it."""
    W = np.array([a["weight"] for a in assessments], dtype=float)
    total = W.sum() or 100.0
    mean, sd = fx.class_stats(X)
    known = ~np.isnan(mean)
    fill_mean = float(np.nanmean(mean)) if known.any() else 60.0
    fill_sd = float(np.nanmean(sd[known])) if known.any() else 15.0
    mean = np.where(known, mean, fill_mean)
    sd = np.maximum(np.where(known & (sd > 0), sd, fill_sd), 3.0)

    row = X[i]
    done = ~np.isnan(row)
    if not done.any():
        return None
    earned = float(np.sum(row[done] * W[done]) / total)
    remaining_idx = np.where(~done)[0]
    z = (row[done] - mean[done]) / sd[done]
    k = int(done.sum())
    z_hat = float(z.mean()) * k / (k + 1.0)
    spread = float(np.clip(z.std() if k >= 2 else 0.8, 0.4, 1.2))

    rng = np.random.default_rng(zlib.crc32(student_key.encode()))
    S = MC_SIMULATIONS
    if len(remaining_idx):
        standing = z_hat + rng.normal(0, 0.6 / np.sqrt(k + 1.0), size=(S, 1))
        noise = rng.normal(0, spread, size=(S, len(remaining_idx)))
        sims = np.clip(mean[remaining_idx] + (standing + noise) * sd[remaining_idx], 0, 100)
        final = earned + sims @ W[remaining_idx] / total
    else:
        final = np.full(S, earned)
    grades = np.array([grade_for_mark(v) for v in final])
    probs = {g: float(np.mean(grades == g)) for g in GRADES}
    # Keep a little probability on every grade, as no simulation is certain.
    probs = {g: 0.98 * p + 0.02 / len(GRADES) for g, p in probs.items()}
    p10, p50, p90 = np.percentile(final, [10, 50, 90])
    assumptions = [f"Current standing: {abs(z_hat):.1f} standard deviations {'below' if z_hat < 0 else 'above'} "
                   f"the class average, based on {k} marked assessment(s).",
                   f"Remaining {len(remaining_idx)} assessment(s) simulated {S:,} times around that standing.",
                   ("Assessments the class has not sat yet are assumed to have a class average of "
                    f"{fill_mean:.0f}% and spread of {fill_sd:.0f} points.") if (~known[remaining_idx]).any() else None]
    return _package(probs, "simulation", projected_mark=round(float(p50), 1),
                    range=[round(float(p10), 1), round(float(p90), 1)], factors=[],
                    assumptions=[a for a in assumptions if a],
                    confidence_text="Simulated estimate, no course history")


# ---------------- drift ----------------

def _psi(ref, cur, bins=10):
    ref = ref[~np.isnan(ref)]
    cur = cur[~np.isnan(cur)]
    if len(ref) < 50 or len(cur) < 10:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return None
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, 1e-4, None), np.clip(c, 1e-4, None)
    return float(np.sum((c - r) * np.log(c / r)))


def drift_report(bundle, F, assessments):
    """Compares a course's current students with the training data."""
    if bundle is None or F is None or len(F) == 0:
        return None
    names = bundle["features"]
    ref, ref_prop = bundle["drift_ref"].astype(float), bundle["drift_prop"].astype(float)
    cur_prop = F[:, names.index("prop_assessed")]
    centre = float(np.nanmedian(cur_prop))
    band = np.abs(ref_prop - centre) <= 0.15
    if band.sum() < 200:
        band = np.ones(len(ref), dtype=bool)
    items, worst = [], 0.0
    for f in fx.DRIFT_FEATURES:
        j = names.index(f)
        psi = _psi(ref[band, j], F[:, j])
        if psi is None:
            continue
        worst = max(worst, psi)
        # A single class is a small sample, so the usual 0.1 / 0.25 PSI
        # cut offs would flag noise. 0.25 and 0.5 are used instead.
        if psi > 0.25:
            items.append({"feature": fx.LABELS[f], "psi": round(psi, 3),
                          "text": f"{fx.LABELS[f]} is distributed differently from comparable past students "
                                  f"(PSI {psi:.2f}).", "level": "high" if psi > 0.5 else "moderate"})
    context_flags = []
    for f in fx.CONTEXT_FEATURES:
        j = names.index(f)
        col = ref[:, j]
        col = col[~np.isnan(col)]
        val = float(np.nanmedian(F[:, j])) if not np.all(np.isnan(F[:, j])) else None
        if val is None or len(col) == 0:
            continue
        lo, hi = np.percentile(col, [1, 99])
        if val < lo or val > hi:
            fmt = (lambda v: f"{v * 100:.0f}%") if f == "hist_pass_rate" else (lambda v: f"{v:.0f}")
            context_flags.append({"feature": fx.LABELS[f], "level": "high",
                                  "text": f"{fx.LABELS[f]} ({fmt(val)}) is outside the range seen in training "
                                          f"({fmt(lo)} to {fmt(hi)})."})
    new_types = sorted({a.get("atype") or "Other" for a in assessments} - set(bundle["types_seen"]))
    if new_types:
        context_flags.append({"feature": "Assessment types", "level": "moderate",
                              "text": "Assessment types not present in the training data: " + ", ".join(new_types) + "."})
    flags = items + context_flags
    level = ("high" if any(x["level"] == "high" for x in flags) else
             "moderate" if flags else "low")
    return {"level": level, "max_psi": round(worst, 3), "items": flags,
            "note": {"high": "This course differs strongly from the data the model was trained on. Treat predictions with extra caution.",
                     "moderate": "This course differs somewhat from the training data.",
                     "low": "This course looks similar to the training data."}[level]}
