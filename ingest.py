"""
ingest.py
The admin upload workflow: stage -> map -> validate -> confirm -> save.

A staged upload is held in staging/ only while the admin reviews the
mapping, because it still contains real student IDs. It is deleted as soon
as the dataset is saved or the upload is cancelled, and anything older than
a day is purged automatically. Saved data holds pseudonymised IDs only.
"""

import os
import pickle
import secrets
import time

import database as db
import ml_engine
import pseudo
import schema_mapping as sm
from config import STAGING_DIR

STAGING_MAX_AGE = 24 * 3600


def _path(token):
    if not token or not token.isalnum():
        raise KeyError(token)
    return os.path.join(STAGING_DIR, f"{token}.pkl")


def purge_old():
    os.makedirs(STAGING_DIR, exist_ok=True)
    cutoff = time.time() - STAGING_MAX_AGE
    for f in os.listdir(STAGING_DIR):
        p = os.path.join(STAGING_DIR, f)
        if f.endswith(".pkl") and os.path.getmtime(p) < cutoff:
            os.remove(p)
    # A training set is created before its file is saved; one whose upload
    # was abandoned is left empty and removed here.
    db.purge_empty_training_sets(STAGING_MAX_AGE)


def stage(filename, raw, course_id, kind, label, username):
    """Reads the file and proposes a mapping. Returns (token, error)."""
    purge_old()
    columns, data, numbers, err = sm.read_upload(filename, raw)
    if err:
        return None, err
    course = db.get_course(course_id)
    staged = {"filename": filename, "course_id": course_id, "course_code": course["code"], "kind": kind,
              "label": label.strip() or (course["semester"] if kind == "current" else "History"),
              "uploaded_by": username, "columns": columns, "data": data, "row_numbers": numbers,
              "created": time.time()}
    profile = db.get_profile(course["code"], kind)
    staged["mapping"] = sm.propose(staged, profile)
    staged["used_profile"] = bool(profile)
    token = secrets.token_hex(12)
    save_staged(token, staged)
    return token, None


def load_staged(token):
    with open(_path(token), "rb") as f:
        return pickle.load(f)


def save_staged(token, staged):
    os.makedirs(STAGING_DIR, exist_ok=True)
    with open(_path(token), "wb") as f:
        pickle.dump(staged, f)


def discard(token):
    try:
        os.remove(_path(token))
    except (OSError, KeyError):
        pass


def apply_form(staged, form):
    """Reads the admin's edits from the confirmation screen."""
    mapping = staged["mapping"]
    old_roles = [c["role"] for c in mapping["columns"]]
    for i, c in enumerate(mapping["columns"]):
        role = form.get(f"role_{i}")
        allowed = sm.WIDE_ROLES if mapping["layout"] == "wide" else sm.LONG_ROLES
        if role in allowed:
            c["role"] = role
    layout = form.get("layout")
    if layout in ("wide", "long") and layout != mapping["layout"]:
        # Re-detect the column roles for the chosen layout rather than
        # clearing them, so switching back and forth is harmless.
        staged["mapping"] = sm.propose(staged, db.get_profile(staged["course_code"], staged["kind"]), layout=layout)
        staged["label"] = form.get("label", staged["label"]).strip() or staged["label"]
        return
    edited = []
    mapping.setdefault("removed", [])
    for i, a in enumerate(mapping["assessments"]):
        a = dict(a)
        if form.get(f"a_remove_{i}") == "yes":
            mapping["removed"].append(a["key"])
            continue
        if form.get(f"a_name_{i}") is not None:
            a["name"] = form.get(f"a_name_{i}", "").strip()
            changed = set()
            for f in ("weight", "max_mark", "order"):
                v = form.get(f"a_{f}_{i}", "").strip()
                try:
                    new = float(v) if v else None
                except ValueError:
                    new = None
                if new != a.get(f):
                    changed.add(f)
                    if f == "weight":
                        for flag in ("weight_estimated", "weight_equal_share", "weight_from_summary"):
                            a.pop(flag, None)
                    if f == "max_mark":
                        a.pop("max_assumed", None)
                        a.pop("max_inferred", None)
                a[f] = new
            t = form.get(f"a_type_{i}")
            if t and t != a.get("atype"):
                a["atype"], a["type_source"] = t, "admin"
                changed.add("atype")
            a["due_date"] = form.get(f"a_date_{i}", "").strip() or None
            # A conflicting value in the file stays an error until the admin
            # explicitly sets that field here.
            a["conflicts"] = [f for f in a.get("conflicts", []) if f not in changed]
            if form.get(f"a_confirm_max_{i}") == "yes":
                a.pop("max_assumed", None)
        edited.append(a)
    mapping["assessments"] = edited
    if [c["role"] for c in mapping["columns"]] != old_roles:
        mapping["assessments"] = sm.derive_assessments(staged, mapping, db.get_profile(staged["course_code"], staged["kind"]),
                                                       previous=edited)
    mapping["exclude_error_rows"] = form.get("exclude_error_rows") == "yes"
    staged["label"] = form.get("label", staged["label"]).strip() or staged["label"]


def commit(token, staged, username, retrain=True):
    """Validates and saves. Returns (result dict, error)."""
    result = sm.build(staged, staged["mapping"])
    if result["errors"]:
        return None, "The data has problems that must be fixed first."
    code = staged["course_code"]
    rows, seen = [], set()
    for r in result["rows"]:
        pid = pseudo.pseudonymise(code, r["raw_id"])
        if (r["cohort"], pid) in seen:
            return None, "Two different student IDs produced the same pseudonym. Contact the system administrator."
        seen.add((r["cohort"], pid))
        # This semester's students are shown to their lecturers by their real
        # ID; past results only train the model, so they keep the code alone.
        rows.append({"pseudo_id": pid, "display_id": r["raw_id"].strip() if staged["kind"] == "current" else None,
                     "cohort": r["cohort"], "marks": r["marks"],
                     "final_grade": r.get("final_grade"), "final_mark": r.get("final_mark")})
    accepted = {"warnings": result["warnings"], "info": result["info"],
                "excluded_rows": sorted(result["error_rows"]) if staged["mapping"].get("exclude_error_rows") else []}
    mapping_record = {"layout": staged["mapping"]["layout"],
                      "columns": [{"col": c["col"], "role": c["role"]} for c in staged["mapping"]["columns"]],
                      "assessments": [{k: a.get(k) for k in ("key", "name", "atype", "type_source", "weight", "max_mark",
                                                             "due_date", "order")} for a in staged["mapping"]["assessments"]]}
    dataset_id, version = db.save_dataset(staged["course_id"], staged["kind"], staged["label"], staged["filename"],
                                          username, result["structure"], mapping_record, accepted, rows)
    db.save_profile(code, staged["kind"], sm.profile_from_mapping(staged["mapping"]), username)
    excluded = len(accepted["excluded_rows"])
    db.audit(username, "dataset_uploaded",
             f"{staged['kind'].capitalize()} data '{staged['label']}' version {version} from {staged['filename']}: "
             f"{len(rows)} students, {len(result['structure'])} assessments"
             + (f", {excluded} row(s) excluded" if excluded else "")
             + (f", {len(result['warnings'])} warning(s) accepted" if result["warnings"] else "") + ".",
             course_id=staged["course_id"], dataset_id=dataset_id)
    discard(token)
    out = {"dataset_id": dataset_id, "version": version, "students": len(rows),
           "assessments": len(result["structure"]), "retrain": None}
    if staged["kind"] == "historical" and retrain:
        model_id, msg = ml_engine.retrain(username, f"Historical upload for {code}")
        out["retrain"] = {"model_id": model_id, "message": msg}
    return out, None


def load_demo(username):
    """Loads the fictional sample courses through the same pipeline an admin
    upload uses, with the detected mapping accepted as is. Returns messages."""
    import sample_data
    files = sample_data.generate_all()
    messages = []
    for code, spec in sample_data.COURSES.items():
        course = db.find_course_by_code(code)
        if course and db.active_current_dataset(course["id"]):
            continue
        cid = course["id"] if course else db.create_course(code, spec["name"], sample_data.CURRENT_SEMESTER, "Laucala")
        if not course:
            db.audit(username, "course_created", f"{code} {spec['name']} (demo)", course_id=cid)
        for kind in ("historical", "current"):
            path = files[code].get("history" if kind == "historical" else "current")
            if not path:
                continue
            with open(path, "rb") as f:
                token, err = stage(os.path.basename(path), f.read(), cid, kind,
                                   "History" if kind == "historical" else sample_data.CURRENT_SEMESTER, username)
            if err:
                messages.append(f"{code}: {err}")
                continue
            staged = load_staged(token)
            _, err = commit(token, staged, username, retrain=False)
            if err:
                result = sm.build(staged, staged["mapping"])
                messages.append(f"{code} {kind}: {err} " + " ".join(result["errors"][:2]))
                discard(token)
    model_id, msg = ml_engine.retrain(username, "Demo data loaded")
    messages.append(msg)
    return messages
