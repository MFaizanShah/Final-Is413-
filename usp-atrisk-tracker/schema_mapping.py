"""
schema_mapping.py
Reads an uploaded CSV or XLSX of any reasonable layout, proposes a mapping
of its columns onto a course structure, and validates the result.

Supported layouts
-----------------
Wide   one row per student, one column per assessment. Weights and maximum
       marks may be written in the header ("Quiz 1 (10%) /20") or in extra
       rows labelled Weight, Out of, Class average, Due date or Type.
Long   one row per student per assessment, with columns such as student_id,
       assessment, mark, max_mark, weight, date, class_average, type.

Detection only PROPOSES a mapping. The admin always sees and can edit it on
the confirmation screen before anything is saved. Validation reports
problems and never fixes them silently: errors block saving until the admin
corrects the mapping or explicitly chooses to exclude the listed rows.
"""

import io
import re
from collections import Counter, OrderedDict

import pandas as pd

from config import TYPES, GRADES, PASS_MARK, detect_type, normalise_grade, normalise_type, grade_for_mark
import type_suggest


def key(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


ID_ALIASES = {"studentid", "id", "studentnumber", "studentno", "sid", "idnumber", "studentidnumber", "matric",
              "registrationnumber", "regno", "username", "candidate", "candidateno"}
PERSONAL_ALIASES = {"name", "studentname", "fullname", "firstname", "lastname", "surname", "givenname", "email",
                    "emailaddress", "phone", "mobile", "gender", "sex", "dob", "dateofbirth", "programme", "program",
                    "major", "address", "nationality", "campus"}
GRADE_ALIASES = {"finalgrade", "finalresult", "grade", "result", "outcome", "lettergrade", "coursegrade", "final"}
FINAL_MARK_ALIASES = {"finalmark", "finaltotal", "total", "overall", "finalscore", "coursemark", "totalmark",
                      "overallmark", "finalpercent", "finalpercentage", "coursetotal"}
COHORT_ALIASES = {"cohort", "semester", "sem", "year", "term", "intake", "offering", "session", "period"}

LONG_ASSESSMENT = {"assessment", "assessmentname", "component", "item", "task", "assessmentitem", "assignment"}
LONG_MARK = {"mark", "marks", "score", "rawmark", "points", "rawscore", "markawarded"}
LONG_MAX = {"max", "maxmark", "maxmarks", "outof", "maximum", "possible", "totalpossible", "maxscore"}
LONG_WEIGHT = {"weight", "weighting", "percent", "percentage", "worth", "contribution"}
LONG_DATE = {"date", "duedate", "due", "submitted", "submissiondate", "week"}
LONG_CLASSAVG = {"classavg", "classaverage", "average", "mean", "cohortaverage", "classmean"}
LONG_TYPE = {"type", "category", "assessmenttype", "kind"}

META_LABELS = OrderedDict([
    ("weight", {"weight", "weighting", "weights", "percent", "percentage", "worth", "contribution"}),
    ("max_mark", {"outof", "max", "maximum", "maxmark", "maxmarks", "totalmarks", "possible", "marksoutof", "outofmarks"}),
    ("class_avg", {"classaverage", "average", "mean", "classavg", "classmean", "cohortaverage"}),
    ("due_date", {"due", "date", "duedate", "dates", "week"}),
    ("atype", {"type", "category", "assessmenttype"}),
])

WIDE_ROLES = ["student_id", "assessment", "final_grade", "final_mark", "cohort", "ignore"]
LONG_ROLES = ["student_id", "assessment_name", "mark", "max_mark", "weight", "date", "class_avg", "atype",
              "final_grade", "final_mark", "cohort", "ignore"]
ROLE_LABELS = {"student_id": "Student ID", "assessment": "Assessment marks", "final_grade": "Final grade (outcome)",
               "final_mark": "Final mark (outcome)", "cohort": "Cohort / semester", "ignore": "Ignore",
               "assessment_name": "Assessment name", "mark": "Mark", "max_mark": "Maximum mark", "weight": "Weight",
               "date": "Date", "class_avg": "Class average", "atype": "Assessment type"}

NOT_SUBMITTED = {"ns", "nsub", "dns", "abs", "absent", "nosubmission", "notsubmitted", "fail", "nil", "zero"}
EXCUSED = {"ex", "exc", "excused", "aeg", "aegrotat", "exempt", "med", "medical", "special", "sc"}
BLANK = {"", "nan", "none", "-", "na", "n/a", "null", "--", "notyet", "pending"}
PASS_WORDS = {"pass", "p", "passed", "pa", "y", "yes", "satisfactory", "s", "complete", "completed", "credit"}
FAIL_WORDS = {"fail", "f", "failed", "fl", "n", "no", "unsatisfactory", "u", "e", "nc", "notcomplete"}
PASS_UNKNOWN = "PASS"  # passed, exact grade unknown


# ---------------- reading ----------------

def read_upload(filename, raw):
    """Returns (columns, data_rows, row_numbers, error). Finds the header row
    automatically, so title rows above the table are skipped."""
    name = (filename or "").lower()
    try:
        if name.endswith(".csv"):
            text = raw.decode("utf-8-sig", errors="replace")
            grid = pd.read_csv(io.StringIO(text), header=None, dtype=str, keep_default_na=False,
                               skip_blank_lines=False, sep=None, engine="python")
        elif name.endswith((".xlsx", ".xls")):
            grid = pd.read_excel(io.BytesIO(raw), header=None, dtype=str).fillna("")
        else:
            return None, None, None, "Please upload a .csv or .xlsx file."
    except Exception as exc:
        return None, None, None, f"The file could not be read: {exc}"
    grid = grid.astype(str).apply(lambda col: col.str.strip())
    grid = grid.replace({"nan": "", "NaT": ""})
    values = grid.values.tolist()
    if not values:
        return None, None, None, "The file is empty."

    header_idx = _find_header(values)
    header = [str(h).strip() for h in values[header_idx]]
    keep = [i for i, h in enumerate(header) if h and not h.lower().startswith("unnamed")]
    if len(keep) < 2:
        return None, None, None, "Could not find a header row with column names."
    columns = _dedupe([header[i] for i in keep])
    rows, numbers = [], []
    for r_i in range(header_idx + 1, len(values)):
        row = [str(values[r_i][i]).strip() if i < len(values[r_i]) else "" for i in keep]
        if any(row):
            rows.append(row)
            numbers.append(r_i + 1)  # 1 based, as in a spreadsheet
    if not rows:
        return None, None, None, "The file has a header but no data rows."
    return columns, rows, numbers, None


def _find_header(values):
    best, best_score = 0, -1
    for i, row in enumerate(values[:15]):
        cells = [str(c).strip() for c in row]
        filled = [c for c in cells if c]
        if len(filled) < 2:
            continue
        texty = sum(1 for c in filled if _to_number(c) is None)
        keys = {key(c) for c in filled}
        bonus = 3 if keys & (ID_ALIASES | LONG_ASSESSMENT | GRADE_ALIASES) else 0
        score = texty + bonus + len(filled) * 0.1
        if score > best_score:
            best, best_score = i, score
        if bonus:
            break
    return best


def _dedupe(names):
    seen, out = Counter(), []
    for n in names:
        seen[n] += 1
        out.append(n if seen[n] == 1 else f"{n} ({seen[n]})")
    return out


def _to_number(text):
    t = str(text).strip().replace("%", "").replace(",", "")
    if not t:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _parse_date(text):
    t = str(text or "").strip()
    if not t:
        return None
    iso = re.fullmatch(r"\d{4}-\d{1,2}-\d{1,2}", t.split(" ")[0])
    try:
        if iso:
            d = pd.to_datetime(t.split(" ")[0], format="%Y-%m-%d", errors="coerce")
        else:
            d = pd.to_datetime(t, dayfirst=True, errors="coerce", format="mixed")
    except (ValueError, TypeError):
        return None
    return None if pd.isna(d) else d.strftime("%Y-%m-%d")


def parse_header(header):
    """'Mid Sem Test (20%) /50' -> ('Mid Sem Test', 20.0, 50.0)."""
    text = str(header)
    weight = max_mark = None
    m = re.search(r"(\d+(?:\.\d+)?)\s*%", text)
    if m:
        weight = float(m.group(1))
        text = text[:m.start()] + text[m.end():]
    m = re.search(r"(?:/|\bout\s*of\b|\bmax\b\.?:?|\[)\s*(\d+(?:\.\d+)?)\s*\]?", text, re.I)
    if m:
        max_mark = float(m.group(1))
        text = text[:m.start()] + text[m.end():]
    text = re.sub(r"[\(\)\[\]]", " ", text)
    text = re.sub(r"\b(weight|wt|marks?)\b\s*:?", " ", text, flags=re.I) if (weight or max_mark) else text
    name = re.sub(r"\s+", " ", text).strip(" -_:,;")
    return name or str(header).strip(), weight, max_mark


# ---------------- detection ----------------

def _numeric_share(values):
    filled = [v for v in values if str(v).strip() and key(v) not in BLANK]
    if not filled:
        return 0.0, 0
    ok = sum(1 for v in filled if _to_number(v) is not None or key(v) in NOT_SUBMITTED | EXCUSED or "/" in str(v))
    return ok / len(filled), len(filled)


def _id_like_share(values):
    filled = [str(v).strip() for v in values if str(v).strip()]
    if not filled:
        return 0
    return sum(1 for v in filled if re.fullmatch(r"[A-Za-z]{0,3}\d{5,}[A-Za-z]?", v)) / len(filled)


def _meta_label(row, id_index):
    """Which meta row type (weight, max_mark, ...) a row is, or None. Only the
    ID cell and the first cell are checked, so student names cannot match."""
    cells = [row[id_index]] if id_index is not None else []
    cells.append(row[0])
    for c in cells:
        k = key(c)
        if not k or re.search(r"\d{4,}", k):
            continue
        for label, aliases in META_LABELS.items():
            if k in aliases or any(len(a) >= 6 and k.startswith(a) for a in aliases):
                return label
    return None


def detect_layout(columns):
    keys = [key(c) for c in columns]
    has_name = any(k in LONG_ASSESSMENT for k in keys)
    has_mark = any(k in LONG_MARK for k in keys)
    return "long" if has_name and has_mark else "wide"


def propose(staged, profile=None, layout=None):
    """Builds the initial mapping shown on the confirmation screen. If the
    admin switches layout, it is re-run with that layout forced."""
    columns, data, kind = staged["columns"], staged["data"], staged["kind"]
    layout = layout or detect_layout(columns)
    saved_roles = (profile or {}).get("column_roles", {}) if profile and profile.get("layout") == layout else {}
    roles = []
    id_taken = False
    for ci, col in enumerate(columns):
        k = key(col)
        values = [r[ci] for r in data]
        if k in saved_roles:
            role, source = saved_roles[k], "saved"
        else:
            role, source = _guess_role(layout, kind, k, values, id_taken, col), "detected"
        if role == "student_id":
            if id_taken:
                role = "ignore"
            id_taken = True
        roles.append({"col": col, "role": role, "source": source,
                      "personal": k in PERSONAL_ALIASES or "email" in k or "name" in k})
    if not id_taken:
        best = max(range(len(columns)), key=lambda i: _id_like_share([r[i] for r in data]))
        if _id_like_share([r[best] for r in data]) > 0.6:
            roles[best]["role"] = "student_id"
    mapping = {"layout": layout, "columns": roles, "exclude_error_rows": False}
    if layout == "wide":
        # Columns with a value in a Weight row are assessments, even if no
        # marks have been entered yet.
        meta, _ = _split_meta(staged, mapping)
        if "weight" in meta:
            for ci, c in enumerate(roles):
                if (c["role"] == "ignore" and c["source"] == "detected" and not c["personal"]
                        and _to_number(meta["weight"][ci]) is not None):
                    c["role"] = "assessment"
    mapping["assessments"] = derive_assessments(staged, mapping, profile)
    return mapping


def _guess_role(layout, kind, k, values, id_taken, header=""):
    if (k in ID_ALIASES or "studentid" in k or "studentnumber" in k) and not id_taken:
        return "student_id"
    if k in PERSONAL_ALIASES or "email" in k or k.endswith("name") and k not in LONG_ASSESSMENT | {"assessmentname"}:
        return "ignore"
    if k in GRADE_ALIASES:
        return "final_grade" if kind == "historical" else "ignore"
    if k in FINAL_MARK_ALIASES:
        return "final_mark" if kind == "historical" else "ignore"
    if k in COHORT_ALIASES:
        return "cohort"
    if layout == "long":
        for role, aliases in [("assessment_name", LONG_ASSESSMENT | {"assessmentname"}), ("mark", LONG_MARK),
                              ("max_mark", LONG_MAX), ("weight", LONG_WEIGHT), ("date", LONG_DATE),
                              ("class_avg", LONG_CLASSAVG), ("atype", LONG_TYPE)]:
            if k in aliases:
                return role
        return "ignore"
    if not id_taken and _id_like_share(values) > 0.8:
        return "student_id"
    share, n = _numeric_share(values)
    if n == 0:
        # An empty column is an assessment not marked yet if its header
        # looks like one.
        name, weight, _ = parse_header(header or k)
        return "assessment" if weight is not None or detect_type(name) != "Other" else "ignore"
    return "assessment" if share >= 0.6 else "ignore"


def _split_meta(staged, mapping):
    """Separates meta rows (Weight, Out of, ...) from student rows. Wide only."""
    cols = mapping["columns"]
    id_index = next((i for i, c in enumerate(cols) if c["role"] == "student_id"), None)
    meta, students = {}, []
    for row, num in zip(staged["data"], staged["row_numbers"]):
        label = _meta_label(row, id_index) if mapping["layout"] == "wide" else None
        if label and label not in meta:
            meta[label] = row
        elif label:
            continue
        else:
            students.append((row, num))
    return meta, students


def derive_assessments(staged, mapping, profile=None, previous=None):
    """Assessment list from the column roles. Values the admin already edited
    (previous) win, then values in the file, then the saved course profile."""
    previous = {a["key"]: a for a in (previous or [])}
    saved = (profile or {}).get("assessments", {})
    cols = mapping["columns"]
    found = []

    if mapping["layout"] == "wide":
        meta, _ = _split_meta(staged, mapping)
        for ci, c in enumerate(cols):
            if c["role"] != "assessment":
                continue
            name, w, mx = parse_header(c["col"])
            item = {"key": c["col"], "name": name, "weight": w, "max_mark": mx, "due_date": None, "class_avg": None,
                    "atype": None, "type_source": None}
            if "weight" in meta and item["weight"] is None:
                item["weight"] = _to_number(meta["weight"][ci])
            if "max_mark" in meta and item["max_mark"] is None:
                item["max_mark"] = _to_number(meta["max_mark"][ci])
            if "class_avg" in meta:
                item["class_avg"] = _to_number(meta["class_avg"][ci])
            if "due_date" in meta:
                item["due_date"] = _parse_date(meta["due_date"][ci])
            if "atype" in meta and normalise_type(meta["atype"][ci]):
                item["atype"], item["type_source"] = normalise_type(meta["atype"][ci]), "file"
            found.append(item)
    else:
        idx = {c["role"]: i for i, c in enumerate(cols) if c["role"] != "ignore"}
        if "assessment_name" in idx:
            per = OrderedDict()
            for row in staged["data"]:
                nm = row[idx["assessment_name"]].strip()
                if not nm:
                    continue
                d = per.setdefault(nm, {"weight": set(), "max_mark": set(), "date": set(), "class_avg": set(), "atype": set()})
                for role, field in [("weight", "weight"), ("max_mark", "max_mark"), ("class_avg", "class_avg")]:
                    if role in idx and _to_number(row[idx[role]]) is not None:
                        d[field].add(_to_number(row[idx[role]]))
                if "date" in idx and _parse_date(row[idx["date"]]):
                    d["date"].add(_parse_date(row[idx["date"]]))
                if "atype" in idx and normalise_type(row[idx["atype"]]):
                    d["atype"].add(normalise_type(row[idx["atype"]]))
            for nm, d in per.items():
                name, w, mx = parse_header(nm)
                item = {"key": nm, "name": name,
                        "weight": w if w is not None else (min(d["weight"]) if len(d["weight"]) == 1 else None),
                        "max_mark": mx if mx is not None else (min(d["max_mark"]) if len(d["max_mark"]) == 1 else None),
                        "due_date": min(d["date"]) if d["date"] else None,
                        "class_avg": sum(d["class_avg"]) / len(d["class_avg"]) if d["class_avg"] else None,
                        "atype": next(iter(d["atype"])) if len(d["atype"]) == 1 else None,
                        "type_source": "file" if len(d["atype"]) == 1 else None,
                        "conflicts": [f for f in ("weight", "max_mark", "atype") if len(d[f]) > 1]}
                found.append(item)

    if mapping["layout"] == "wide" and found and all(i.get("weight") is None for i in found) \
            and not any((saved.get(key(i["name"])) or {}).get("weight") for i in found):
        _estimate_weights(staged, mapping, found)

    # Assessments in the saved course structure that this file has no column
    # or rows for yet (typically not marked yet). The admin can remove them.
    matched = {key(i["name"]) for i in found} | {key(i["key"]) for i in found}
    removed = set(mapping.get("removed", []))
    # When the file's own weights already cover the whole course, the file
    # describes a complete (possibly changed) structure, so assessments from
    # an earlier semester's saved structure are not added.
    file_total = sum(i["weight"] for i in found if i.get("weight") is not None)
    self_describing = found and all(i.get("weight") is not None for i in found) and abs(file_total - 100) <= 0.5
    for pk, p in ({} if self_describing else saved).items():
        k = "profile:" + p["name"]
        if pk in matched or k in removed:
            continue
        found.append({"key": k, "name": p["name"], "weight": p.get("weight"), "max_mark": p.get("max_mark"),
                      "due_date": p.get("due_date"), "class_avg": None, "atype": None, "type_source": None,
                      "no_data": True})

    out = []
    unknown = []
    for i, item in enumerate(found):
        prev = previous.get(item["key"])
        prof = saved.get(key(item["name"])) or saved.get(key(item["key"]))
        if prev:
            for f in ("name", "weight", "max_mark", "due_date", "atype", "type_source", "order"):
                if prev.get(f) not in (None, ""):
                    item[f] = prev[f]
            item["conflicts"] = prev.get("conflicts", item.get("conflicts", []))
            if prev.get("weight_estimated"):
                item["weight_estimated"] = prev["weight_estimated"]
            else:
                item.pop("weight_estimated", None)
            if prev.get("max_mark") is not None and not prev.get("max_assumed"):
                item["max_assumed"] = False
        if prof:
            for f in ("weight", "max_mark", "due_date"):
                if item.get(f) is None and prof.get(f) is not None:
                    item[f] = prof[f]
                    item.setdefault("from_profile", []).append(f)
            if not item.get("atype") and prof.get("atype"):
                item["atype"], item["type_source"] = prof["atype"], "saved"
        if not item.get("atype"):
            item["atype"] = detect_type(item["name"])
            item["type_source"] = "rule" if item["atype"] != "Other" else "none"
            if item["atype"] == "Other":
                unknown.append(item)
        if item.get("max_mark") is None:
            item["max_mark"] = 100.0
            item["max_assumed"] = item.get("max_assumed", True)
        item.setdefault("order", i + 1)
        out.append(item)

    if unknown:
        suggestions = type_suggest.suggest([u["name"] for u in unknown])
        for u in unknown:
            s = suggestions.get(u["name"])
            if s and s in TYPES:
                u["atype"], u["type_source"] = s, "llm"

    if all(a.get("due_date") for a in out) and out and not previous:
        # Dates carried over from a saved structure belong to an earlier
        # year, so compare month and day only when years are mixed.
        mixed = len({a["due_date"][:4] for a in out}) > 1
        for rank, a in enumerate(sorted(out, key=lambda a: a["due_date"][5:] if mixed else a["due_date"]), start=1):
            a["order"] = rank
    return out


def _estimate_weights(staged, mapping, found):
    """Suggests weights when the file has none but has a final mark: the
    final mark is (almost) a weighted sum of the assessment marks, so
    non-negative least squares recovers the weights. The result is only a
    pre-filled suggestion on the confirmation screen, flagged as estimated."""
    cols = mapping["columns"]
    fm_i = next((i for i, c in enumerate(cols) if c["role"] == "final_mark"), None)
    if fm_i is None:
        return
    import numpy as np
    from scipy.optimize import nnls
    col_i = [next(i for i, c in enumerate(cols) if c["col"] == item["key"]) for item in found]
    _, students = _split_meta(staged, mapping)
    X, y = [], []
    for row, _ in students:
        fm = _to_number(row[fm_i])
        vals = [_to_number(row[i]) for i in col_i]
        if fm is None or any(v is None for v in vals):
            continue
        X.append([v / (item.get("max_mark") or 100.0) * 100.0 for v, item in zip(vals, found)])
        y.append(fm)
    if len(y) < max(10, 2 * len(found)):
        return
    X, y = np.array(X), np.array(y)
    w, _ = nnls(X, y)
    pred = X @ w
    r2 = 1 - ((y - pred) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    if r2 < 0.9:
        return
    for item, wi in zip(found, w * 100):
        half = round(wi * 2) / 2
        item["weight"] = half if abs(wi - half) < 0.15 else round(float(wi), 1)
        item["weight_estimated"] = round(float(r2), 4)


# ---------------- building and validation ----------------

def _cell_to_percent(raw, max_mark):
    """Returns (percent or None, flag) where flag is None, 'ns', 'excused',
    'fraction' or 'invalid'."""
    text = str(raw or "").strip()
    k = key(text)
    if k in BLANK or not text:
        return None, None
    if k in NOT_SUBMITTED:
        return 0.0, "ns"
    if k in EXCUSED:
        return None, "excused"
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*", text)
    if m and float(m.group(2)) > 0:
        return float(m.group(1)) / float(m.group(2)) * 100.0, "fraction"
    pct = text.endswith("%")
    v = _to_number(text)
    if v is None:
        return None, "invalid"
    if pct:
        return v, None
    return v / max_mark * 100.0, None


def build(staged, mapping):
    """Applies the mapping and validates. Returns a dict with structure,
    rows (raw IDs, not yet pseudonymised), errors, warnings, info and
    error_rows. Nothing is changed or dropped unless the admin has ticked
    exclude_error_rows, and then only the listed rows."""
    kind = staged["kind"]
    cols = mapping["columns"]
    errors, warnings, info = [], [], []
    error_rows = {}

    def row_error(num, msg):
        error_rows.setdefault(num, []).append(msg)

    idx = {}
    for i, c in enumerate(cols):
        if c["role"] in ("ignore", "assessment"):
            continue
        if c["role"] in idx:
            errors.append(f"More than one column is mapped to '{ROLE_LABELS[c['role']]}': "
                          f"{cols[idx[c['role']]]['col']} and {c['col']}.")
        else:
            idx[c["role"]] = i
    if "student_id" not in idx:
        errors.append("No column is mapped to Student ID.")
    if kind == "historical" and "final_grade" not in idx and "final_mark" not in idx:
        errors.append("Historical data needs an outcome column: a final grade or a final mark.")
    if mapping["layout"] == "long":
        for need in ("assessment_name", "mark"):
            if need not in idx:
                errors.append(f"No column is mapped to {ROLE_LABELS[need]}.")
    personal = [c["col"] for c in cols if c.get("personal")]
    if personal:
        info.append("Personal data columns will not be stored: " + ", ".join(personal) + ".")

    # ---- structure checks
    assessments = sorted(mapping["assessments"], key=lambda a: (float(a.get("order") or 0)))
    if not assessments:
        errors.append("No assessments are mapped. Mark at least one column as assessment marks.")
    names = [a["name"].strip() for a in assessments]
    dupes = [n for n, cnt in Counter(key(n) for n in names).items() if cnt > 1]
    if dupes:
        errors.append("Two or more assessments have the same name: "
                      + ", ".join(n for n in names if key(n) in dupes) + ".")
    for a in assessments:
        if not a["name"].strip():
            errors.append(f"An assessment from column '{a['key']}' has no name.")
        if a.get("weight") is None:
            errors.append(f"No weight given for {a['name']}. Enter it on this screen.")
        elif a["weight"] <= 0 or a["weight"] > 100:
            errors.append(f"Weight for {a['name']} is {a['weight']:g}%, which must be above 0 and at most 100.")
        if not a.get("max_mark") or a["max_mark"] <= 0:
            errors.append(f"Maximum mark for {a['name']} must be above 0.")
        if a.get("atype") not in TYPES:
            errors.append(f"Choose a type for {a['name']}.")
        for f in a.get("conflicts", []):
            errors.append(f"The file gives more than one {f.replace('_', ' ')} for {a['name']}. Fix the file or set it here.")
        if a.get("weight_estimated") and a.get("weight") is not None:
            warnings.append(f"Weight for {a['name']} ({a['weight']:g}%) was estimated from the final marks "
                            f"(fit R\u00b2 {a['weight_estimated']:.3f}). Check it against the course outline.")
        if a.get("max_assumed"):
            warnings.append(f"No maximum mark found for {a['name']}; assumed 100. Change it here if marks are out of something else.")
    weights = [a["weight"] for a in assessments if a.get("weight") is not None]
    if weights and len(weights) == len(assessments):
        total = sum(weights)
        if abs(total - 100) > 0.5:
            errors.append(f"Assessment weights add up to {total:g}%, not 100%. Correct the weights or the column mapping.")

    if errors and not assessments:
        return {"errors": errors, "warnings": warnings, "info": info, "error_rows": {}, "structure": [], "rows": []}

    # ---- rows
    id_i = idx.get("student_id")
    records = OrderedDict()  # (cohort, id) -> record
    flags = Counter()
    if mapping["layout"] == "wide" and id_i is not None:
        _, students = _split_meta(staged, mapping)
        col_of = {a["key"]: next(i for i, c in enumerate(cols) if c["col"] == a["key"]) for a in assessments
                  if any(c["col"] == a["key"] for c in cols)}
        for row, num in students:
            sid = row[id_i].strip() if id_i is not None else ""
            cohort = row[idx["cohort"]].strip() if "cohort" in idx else ""
            if not sid:
                row_error(num, "no student ID")
                continue
            rk = (cohort, sid.upper())
            if rk in records:
                row_error(num, f"duplicate student ID {sid} (also on row {records[rk]['row']})")
                continue
            marks = []
            for a in assessments:
                ci = col_of.get(a["key"])
                raw = row[ci] if ci is not None else ""
                pct, flag = _cell_to_percent(raw, a["max_mark"])
                pct = _check_mark(pct, flag, raw, a, num, row_error, flags)
                marks.append(pct)
            records[rk] = {"row": num, "raw_id": sid, "cohort": cohort, "marks": marks,
                           "grade_raw": row[idx["final_grade"]] if "final_grade" in idx else "",
                           "mark_raw": row[idx["final_mark"]] if "final_mark" in idx else ""}
    elif "assessment_name" in idx and "mark" in idx and id_i is not None:
        pos = {a["key"]: j for j, a in enumerate(assessments)}
        by_a = {a["key"]: a for a in assessments}
        seen_pairs = {}
        for row, num in zip(staged["data"], staged["row_numbers"]):
            sid = row[id_i].strip()
            nm = row[idx["assessment_name"]].strip()
            cohort = row[idx["cohort"]].strip() if "cohort" in idx else ""
            if not sid:
                row_error(num, "no student ID")
                continue
            if nm not in pos:
                continue
            rk = (cohort, sid.upper())
            rec = records.setdefault(rk, {"row": num, "raw_id": sid, "cohort": cohort, "marks": [None] * len(assessments),
                                          "grade_raw": "", "mark_raw": ""})
            pair = (rk, nm)
            if pair in seen_pairs:
                row_error(num, f"duplicate mark for {sid} on {nm} (also on row {seen_pairs[pair]})")
                continue
            seen_pairs[pair] = num
            a = by_a[nm]
            raw = row[idx["mark"]]
            mx = _to_number(row[idx["max_mark"]]) if "max_mark" in idx and _to_number(row[idx["max_mark"]]) else a["max_mark"]
            pct, flag = _cell_to_percent(raw, mx)
            rec["marks"][pos[nm]] = _check_mark(pct, flag, raw, a, num, row_error, flags)
            if "final_grade" in idx and row[idx["final_grade"]].strip():
                rec["grade_raw"] = row[idx["final_grade"]]
            if "final_mark" in idx and row[idx["final_mark"]].strip():
                rec["mark_raw"] = row[idx["final_mark"]]

    # ---- outcomes
    if kind == "historical":
        mismatch = 0
        for rk, rec in records.items():
            raw_g = str(rec["grade_raw"]).strip()
            g = normalise_grade(raw_g) if raw_g else None
            passed_only = False
            if raw_g and g is None:
                if key(raw_g) in PASS_WORDS:
                    passed_only = True
                elif key(raw_g) in FAIL_WORDS:
                    g = "D"
            word_outcome = bool(raw_g) and key(raw_g) in PASS_WORDS | FAIL_WORDS | {"fail"}
            fm = _to_number(rec["mark_raw"]) if str(rec["mark_raw"]).strip() else None
            if raw_g and g is None and not passed_only:
                row_error(rec["row"], f"unrecognised final outcome '{raw_g}' (expected {', '.join(GRADES)}, Pass or Fail)")
            if str(rec["mark_raw"]).strip() and fm is None:
                row_error(rec["row"], f"final mark '{rec['mark_raw']}' is not a number")
            if fm is not None and not 0 <= fm <= 100:
                row_error(rec["row"], f"final mark {fm:g} is outside 0 to 100")
                fm = None
            if word_outcome:
                flags["pass_fail"] += 1
            if passed_only:
                if fm is None:
                    g = PASS_UNKNOWN
                    flags["pass_no_grade"] += 1
                elif fm >= PASS_MARK:
                    g = grade_for_mark(fm)
                else:
                    row_error(rec["row"], f"outcome is '{raw_g}' but the final mark {fm:g} is below the pass mark")
            if g is None and fm is not None:
                g = grade_for_mark(fm)
                flags["grade_from_mark"] += 1
            if g in GRADES and fm is not None:
                # For Pass/Fail outcomes only the pass/fail side can disagree.
                disagree = ((g == "D") != (fm < PASS_MARK)) if word_outcome else grade_for_mark(fm) != g
                mismatch += disagree
            if g is None and not raw_g and not str(rec["mark_raw"]).strip():
                row_error(rec["row"], "no final outcome")
            rec["final_grade"], rec["final_mark"] = g, fm
        if mismatch:
            warnings.append(f"{mismatch} student(s) have a final outcome that does not match their final mark on this "
                            "grade scale. The outcome column is used.")
        if flags["pass_fail"]:
            info.append(f"Outcome given as Pass or Fail for {flags['pass_fail']} student(s). Fail is recorded as grade D; "
                        "passing grades are taken from the final mark where there is one.")
        if flags["pass_no_grade"]:
            warnings.append(f"{flags['pass_no_grade']} passing student(s) have no final mark, so their exact grade is unknown. "
                            "They are used for pass or fail risk, not for grade predictions.")
        if flags["grade_from_mark"]:
            info.append(f"Final grade worked out from the final mark for {flags['grade_from_mark']} student(s).")

    # ---- missing values, reported per assessment
    rec_list = list(records.values())
    if rec_list and assessments:
        for j, a in enumerate(assessments):
            n_missing = sum(1 for r in rec_list if r["marks"][j] is None)
            if 0 < n_missing < len(rec_list):
                if kind == "current":
                    warnings.append(f"{a['name']}: {n_missing} of {len(rec_list)} students have no mark while others do. "
                                    "Blank is treated as not yet assessed for those students; enter 0 for a non submission.")
                else:
                    warnings.append(f"{a['name']}: {n_missing} of {len(rec_list)} past students have no mark. "
                                    "The model treats these as missing.")
            elif n_missing == len(rec_list) and kind == "historical":
                warnings.append(f"{a['name']}: no marks at all in this historical file.")
            computed = [r["marks"][j] for r in rec_list if r["marks"][j] is not None]
            if a.get("class_avg") is not None and len(computed) >= 5:
                cavg = sum(computed) / len(computed)
                reported = a["class_avg"] / a["max_mark"] * 100
                if abs(cavg - reported) > 5:
                    warnings.append(f"{a['name']}: the reported class average ({a['class_avg']:g}) differs from the "
                                    f"average of the uploaded marks ({cavg:.1f}%).")
    if flags["ns"]:
        info.append(f"{flags['ns']} cell(s) marked as not submitted (NS, ABS and similar) are recorded as 0.")
    if flags["excused"]:
        warnings.append(f"{flags['excused']} cell(s) marked as excused or aegrotat are treated as not assessed.")
    if flags["fraction"]:
        info.append(f"{flags['fraction']} cell(s) written as a fraction (for example 18/20) were converted to percent.")
    if flags["pct"]:
        info.append(f"{flags['pct']} cell(s) already given as a percentage were used as written.")

    # ---- row errors
    if error_rows:
        lines = [f"Row {n}: " + "; ".join(msgs) for n, msgs in sorted(error_rows.items())]
        if mapping.get("exclude_error_rows"):
            info.append(f"{len(error_rows)} row(s) with problems are excluded at your request.")
        else:
            errors.append(f"{len(error_rows)} row(s) have problems. Fix the file and upload again, or tick "
                          "'Exclude these rows' to save without them.")
        bad = set(error_rows)
        records = OrderedDict((k, r) for k, r in records.items() if r["row"] not in bad)
    else:
        lines = []

    structure = [{"name": a["name"].strip(), "weight": a.get("weight"), "atype": a.get("atype"),
                  "max_mark": a.get("max_mark"), "due_date": a.get("due_date"),
                  "reported_class_avg": a.get("class_avg")} for a in assessments]
    rows = [{"raw_id": r["raw_id"], "cohort": r["cohort"], "marks": [None if m is None else round(m, 2) for m in r["marks"]],
             "final_grade": r.get("final_grade"), "final_mark": r.get("final_mark")} for r in records.values()]
    if not rows and not errors:
        errors.append("No usable student rows were found.")
    stats = {"students": len(rows), "assessments": len(structure),
             "with_outcome": sum(1 for r in rows if r.get("final_grade")),
             "cohorts": sorted({r["cohort"] for r in rows if r["cohort"]})}
    return {"errors": errors, "warnings": warnings, "info": info, "error_rows": error_rows, "error_lines": lines,
            "structure": structure, "rows": rows, "stats": stats}


def _check_mark(pct, flag, raw, a, num, row_error, flags):
    if flag == "invalid":
        row_error(num, f"{a['name']}: '{raw}' is not a mark")
        return None
    if flag:
        flags[flag] += 1
    if str(raw).strip().endswith("%"):
        flags["pct"] += 1
    if pct is not None and (pct < 0 or pct > 100.0001):
        row_error(num, f"{a['name']}: {str(raw).strip()} is outside 0 to {a['max_mark']:g}")
        return None
    return pct


def profile_from_mapping(mapping):
    """Reusable configuration saved per course code for later semesters."""
    return {"layout": mapping["layout"],
            "column_roles": {key(c["col"]): c["role"] for c in mapping["columns"]},
            "assessments": {key(a["name"]): {"name": a["name"], "weight": a.get("weight"), "atype": a.get("atype"),
                                             "max_mark": a.get("max_mark"), "due_date": a.get("due_date")}
                            for a in mapping["assessments"]}}


def template_csv(kind):
    if kind == "historical":
        return ("student_id,cohort,Quiz 1 (10%),Assignment 1 (20%) /40,Mid Semester Test (30%),Final Exam (40%),final_grade\n"
                "S11000001,2025 S1,7,31,62,58,B\n")
    if kind == "long":
        return ("student_id,assessment,mark,max_mark,weight,date,type\n"
                "S11000001,Quiz 1,7,10,10,2026-08-01,Quiz\n"
                "S11000001,Assignment 1,31,40,20,2026-08-20,Coursework\n")
    return ("student_id,Quiz 1,Assignment 1,Mid Semester Test,Final Exam\n"
            "Weight,10,20,30,40\n"
            "Out of,10,40,100,100\n"
            "S11000001,7,31,62,\n")
