"""
outline.py
Reads a course outline (PDF, Word, Excel or CSV) and finds what is needed to
set up the course: code, name, semester, campus and the assessment table
(name, weight, marked out of, due, type).

How it works
------------
1. Every table in the file is collected (PDF tables via pdfplumber, Word
   tables, spreadsheet sheets), along with the plain text.
2. The assessment table is the one whose header has an assessment column
   (Assessment, Task, Component, Item...) and a weight column (Weight, %,
   Worth, Value...). Each row becomes an assessment; total rows are skipped.
   A table that continues on the next PDF page is joined on.
3. If no such table is found, lines of text that contain a percentage, such
   as "Assignment 1 ........ 15%", are used instead.
4. Course code, name, semester and campus come from labelled rows ("Course
   code", "Course title") or from the text ("IS413 Advances in ICT").
5. Optional: with ATRISK_LLM_OUTLINE=1 and ANTHROPIC_API_KEY set, outlines the
   rules cannot read completely are sent to Claude for a second reading. The
   outline text only is sent; it contains no student data.

The result is always shown to a person to confirm and correct before
anything is saved.
"""

import io
import json
import os
import re
import zipfile

from config import detect_type, normalise_type

NAME_WORDS = ("assessment", "task", "component", "item", "activity", "description", "assessment item",
              "assessment task", "assessment type", "work")
WEIGHT_WORDS = ("weight", "weighting", "worth", "value", "percentage", "contribution", "%", "marks (%)")
MAX_WORDS = ("out of", "marked out of", "max", "maximum", "total marks", "marks", "score out of")
DUE_WORDS = ("due", "date", "week", "deadline", "when", "submission")
TYPE_WORDS = ("type", "kind", "category", "format", "mode")
SKIP_ROW = re.compile(r"^\s*(total|overall|sum|grand total|coursework total|course total)\b", re.I)
CODE_RE = re.compile(r"\b([A-Z]{2,4})\s?(\d{3})\b")
PCT_RE = re.compile(r"(\d{1,3}(?:\.\d+)?)\s*%")
NUM_RE = re.compile(r"^\s*(\d{1,3}(?:\.\d+)?)\s*$")
SEM_RE = re.compile(r"\b(Semester|Trimester|Term)\s*([12I]{1,2}|One|Two|Three)\b[,\s]*((?:19|20)\d\d)?", re.I)
YEAR_RE = re.compile(r"\b(20\d\d)\b")


# ---------------- reading the file ----------------

def _clean(v):
    return re.sub(r"\s+", " ", str(v or "")).strip()


def _from_pdf(raw):
    import pdfplumber
    tables, text = [], []
    with pdfplumber.open(io.BytesIO(raw)) as pdf:
        for page_no, page in enumerate(pdf.pages[:30]):
            text.append(page.extract_text() or "")
            page_tables = page.extract_tables() or []
            for i, t in enumerate(page_tables):
                rows = [[_clean(c) for c in r] for r in t if r and any(_clean(c) for c in r)]
                if rows:
                    tables.append({"rows": rows, "page": page_no, "last_on_page": i == len(page_tables) - 1,
                                   "first_on_page": i == 0})
    return tables, "\n".join(text)


def _from_docx(raw):
    import docx
    d = docx.Document(io.BytesIO(raw))
    tables = []
    for t in d.tables:
        rows = []
        for r in t.rows:
            cells, seen = [], set()
            for c in r.cells:
                # Merged cells repeat; keep each one once.
                if id(c._tc) in seen:
                    continue
                seen.add(id(c._tc))
                cells.append(_clean(c.text))
            if any(cells):
                rows.append(cells)
        if rows:
            tables.append({"rows": rows})
    text = "\n".join(p.text for p in d.paragraphs)
    # Header text often holds the course code and title.
    for s in d.sections:
        text = "\n".join(p.text for p in s.header.paragraphs) + "\n" + text
    for t in tables:
        text += "\n" + "\n".join(" | ".join(r) for r in t["rows"])
    return tables, text


def _from_sheet(raw, name):
    import pandas as pd
    if name.endswith(".csv"):
        frames = {"Sheet1": pd.read_csv(io.BytesIO(raw), header=None, dtype=str, keep_default_na=False,
                                        sep=None, engine="python")}
    else:
        frames = pd.read_excel(io.BytesIO(raw), header=None, dtype=str, sheet_name=None)
    tables, text = [], []
    for _, df in frames.items():
        rows = [[_clean(c) if _clean(c).lower() != "nan" else "" for c in r] for r in df.fillna("").values.tolist()]
        rows = [r for r in rows if any(r)]
        if rows:
            # A sheet may hold details above the table, so every start row is a candidate.
            tables.append({"rows": rows, "sheet": True})
            text.append("\n".join(" | ".join(c for c in r if c) for r in rows))
    return tables, "\n".join(text)


def read_file(filename, raw):
    name = (filename or "").lower()
    if name.endswith(".pdf"):
        return _from_pdf(raw)
    if name.endswith(".docx"):
        return _from_docx(raw)
    if name.endswith((".xlsx", ".xls", ".csv")):
        return _from_sheet(raw, name)
    if name.endswith(".doc"):
        raise ValueError("Old Word .doc files cannot be read. Save it as .docx or PDF and upload again.")
    raise ValueError("Please upload the outline as a PDF, Word (.docx) or Excel (.xlsx) file.")


# ---------------- finding the assessment table ----------------

def _col(header, words, exclude=()):
    for i, h in enumerate(header):
        hl = h.lower()
        if any(w in hl for w in words) and not any(x in hl for x in exclude) and hl:
            return i
    return None


def _header_roles(header):
    h = [c.lower() for c in header]
    w = _col(h, WEIGHT_WORDS, exclude=("final mark", "grade"))
    n = _col(h, NAME_WORDS, exclude=("type", "weight", "%", "due", "date", "outcome"))
    if n is None and w is not None:
        # First text column that is not the weight.
        n = next((i for i, c in enumerate(h) if i != w and c and not any(x in c for x in DUE_WORDS + TYPE_WORDS)), None)
    m = _col(h, MAX_WORDS, exclude=("final", "%", "weight"))
    if m == w:
        m = None
    d = _col(h, DUE_WORDS, exclude=("weight",))
    t = _col(h, TYPE_WORDS)
    if t == n:
        t = None
    return {"name": n, "weight": w, "max": m, "due": d, "type": t}


def _weight(v):
    v = _clean(v)
    m = PCT_RE.search(v)
    if m:
        return float(m.group(1))
    m = NUM_RE.match(v)
    if m:
        return float(m.group(1))
    return None


def _max_mark(v):
    v = _clean(v)
    m = re.search(r"(?:out of|/)\s*(\d{1,4}(?:\.\d+)?)", v, re.I) or NUM_RE.match(v)
    return float(m.group(1)) if m else None


def _rows_to_items(rows, roles):
    items = []
    for r in rows:
        def get(k):
            i = roles.get(k)
            return r[i] if i is not None and i < len(r) else ""
        name = get("name")
        if not name or SKIP_ROW.match(name) or name.lower().strip("* ") in ("total", "totals"):
            continue
        w = _weight(get("weight"))
        if w is None:
            # The weight sometimes sits in the name, e.g. "Quiz 1 (5%)".
            m = PCT_RE.search(name)
            w = float(m.group(1)) if m else None
        if w is None:
            continue
        name = re.sub(r"\(?\s*\d{1,3}(?:\.\d+)?\s*%\s*\)?", "", name).strip(" -:,")
        mx = _max_mark(get("max")) if roles.get("max") is not None else None
        m = re.search(r"/\s*(\d{1,4})", get("name"))
        if mx is None and m:
            mx = float(m.group(1))
        name = re.sub(r"/\s*\d{1,4}", "", name).strip()
        t = normalise_type(get("type")) if roles.get("type") is not None and get("type") else None
        items.append({"name": name, "weight": w, "max_mark": mx, "due": get("due") or None,
                      "atype": t or detect_type(name), "type_source": "outline" if t else "rule"})
    return items


def _find_table(tables):
    """Best assessment table: most rows with weights, preferring totals of 100."""
    best, best_score = None, 0
    for ti, t in enumerate(tables):
        rows = t["rows"]
        starts = range(min(len(rows), 25)) if t.get("sheet") else [0]
        for s in starts:
            roles = _header_roles(rows[s])
            if roles["weight"] is None or roles["name"] is None:
                continue
            body = rows[s + 1:]
            # Join a continuation on the next PDF page (same width, no header).
            if t.get("last_on_page") and ti + 1 < len(tables):
                nxt = tables[ti + 1]
                if nxt.get("first_on_page") and nxt.get("page") == t.get("page", -2) + 1 \
                        and len(nxt["rows"][0]) == len(rows[s]) and _header_roles(nxt["rows"][0])["weight"] is None:
                    body = body + nxt["rows"]
            items = _rows_to_items(body, roles)
            if len(items) < 2:
                continue
            total = sum(i["weight"] for i in items)
            score = len(items) + (10 if abs(total - 100) <= 0.5 else 0)
            if score > best_score:
                best, best_score = items, score
    return best


def _from_text(text):
    """Fallback: lines that name an assessment and give a percentage."""
    items = []
    for line in text.splitlines():
        line = _clean(line)
        m = PCT_RE.search(line)
        if not m or len(line) > 140 or SKIP_ROW.match(line):
            continue
        name = _clean(line[:m.start()].rstrip(" .:-|(")).strip("(")
        name = re.sub(r"\.{2,}", " ", name).strip()
        if not name or len(name) > 60 or not re.search(r"[A-Za-z]", name):
            continue
        if re.search(r"\b(pass|worth|grade|attendance requirement|at least|minimum|above|below|and|is|are)\b", name, re.I) \
                and not re.search(r"\b(quiz|test|exam|assignment|lab|project|essay|report|presentation)\b", name, re.I):
            continue
        rest = line[m.end():]
        mx = re.search(r"(?:out of|/)\s*(\d{1,4})", rest, re.I)
        due = re.search(r"\b(week\s*\d{1,2}|exam period|\d{1,2}\s+\w{3,9}\s+20\d\d)\b", rest, re.I)
        items.append({"name": name, "weight": float(m.group(1)), "max_mark": float(mx.group(1)) if mx else None,
                      "due": due.group(1) if due else None, "atype": detect_type(name), "type_source": "rule"})
    # Keep the run that adds up to 100 if there is one.
    seen, out = set(), []
    for i in items:
        k = i["name"].lower()
        if k not in seen:
            seen.add(k)
            out.append(i)
    return out if len(out) >= 2 else None


# ---------------- course details ----------------

def _labelled(tables, text, labels):
    for t in tables:
        for r in t["rows"]:
            for j, c in enumerate(r[:-1]):
                if c and any(c.lower().rstrip(": ") == l for l in labels):
                    v = next((x for x in r[j + 1:] if x), "")
                    if v:
                        return v
    for line in text.splitlines():
        for l in labels:
            m = re.match(rf"\s*{re.escape(l)}\s*[:\-|]\s*(.+)$", line, re.I)
            if m:
                return _clean(m.group(1))
    return None


def _details(tables, text):
    code = _labelled(tables, text, ["course code", "unit code", "code", "subject code"])
    m = CODE_RE.search(code or "") or None
    if not m:
        counts = {}
        for mm in CODE_RE.finditer(text[:4000]):
            k = mm.group(1) + mm.group(2)
            counts[k] = counts.get(k, 0) + 1
        top = max(counts, key=lambda k: (counts[k], -text.find(k))) if counts else None
        code = top
    else:
        code = m.group(1) + m.group(2)
    name = _labelled(tables, text, ["course title", "course name", "title", "unit title", "subject title", "course"])
    if name and CODE_RE.fullmatch(name.replace(" ", "")):
        name = None
    if not name and code:
        for line in text.splitlines()[:15]:
            mm = re.search(rf"{code[:-3]}\s?{code[-3:]}\s*[:\-–—|]?\s*(.+)", line)
            if mm:
                name = mm.group(1)
                break
    if name:
        name = name.strip(" -–—:|·")
        name = re.split(r":|\bcourse outline\b|\bcourse syllabus\b|\bunit outline\b|\(", name, flags=re.I)[0].strip(" -–—|")
    sem = _labelled(tables, text, ["semester", "trimester", "term", "teaching period", "offered"])
    if not sem or not SEM_RE.search(sem):
        mm = SEM_RE.search(text)
        sem = mm.group(0).strip(" ,") if mm else sem
    if sem and not YEAR_RE.search(sem):
        y = YEAR_RE.search(text[:3000])
        if y:
            sem = f"{sem}, {y.group(1)}"
    campus = _labelled(tables, text, ["campus", "campus and mode", "location", "venue", "mode"])
    if campus:
        campus = re.split(r",|;| - ", campus)[0].strip()
    return {"code": (code or "").upper(), "name": name or "", "semester": sem or "", "campus": campus or ""}


# ---------------- optional AI reading ----------------

def _llm(text):
    if not (os.environ.get("ATRISK_LLM_OUTLINE") == "1" and os.environ.get("ANTHROPIC_API_KEY")):
        return None
    try:
        import anthropic
        from config import LLM_MODEL
        msg = anthropic.Anthropic().messages.create(
            model=LLM_MODEL, max_tokens=1500,
            system=("Extract the course details and the assessment table from a university course outline. Reply with "
                    "JSON only: {\"code\":str,\"name\":str,\"semester\":str,\"campus\":str,\"assessments\":[{\"name\":str,"
                    "\"weight\":number,\"max_mark\":number|null,\"due\":str|null,\"type\":\"Coursework|Quiz|Test|"
                    "Practical|Exam|Other\"}]}. Weights are percentages of the final mark. Use only what the text says."),
            messages=[{"role": "user", "content": text[:30000]}])
        raw = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        data = json.loads(raw[raw.find("{"): raw.rfind("}") + 1])
        items = [{"name": _clean(a["name"]), "weight": float(a["weight"]), "max_mark": a.get("max_mark"),
                  "due": a.get("due"), "atype": normalise_type(a.get("type")) or detect_type(a["name"]),
                  "type_source": "ai"} for a in data.get("assessments", []) if a.get("name") and a.get("weight")]
        return {"details": {k: _clean(data.get(k)) for k in ("code", "name", "semester", "campus")}, "items": items}
    except Exception:
        return None


# ---------------- main entry ----------------

def parse(filename, raw):
    """Returns a dict: code, name, semester, campus, assessments, notes, method.
    Raises ValueError with a readable message when the file cannot be read."""
    try:
        tables, text = read_file(filename, raw)
    except ValueError:
        raise
    except zipfile.BadZipFile:
        raise ValueError("The file looks damaged or is not really a Word or Excel file.")
    except Exception as exc:
        raise ValueError(f"The outline could not be read ({type(exc).__name__}). Try saving it as a PDF.")
    if not text.strip() and not tables:
        raise ValueError("No text was found. If the PDF is a scanned image, upload the Word version or type the "
                         "assessments in instead.")

    details = _details(tables, text)
    items, method = _find_table(tables), "table"
    if not items:
        items, method = _from_text(text), "text"
    total = sum(i["weight"] for i in items) if items else 0
    if not items or abs(total - 100) > 0.5 or not details["code"]:
        ai = _llm(text)
        if ai and ai["items"]:
            items, method = ai["items"], "ai"
            for k, v in ai["details"].items():
                if v and not details.get(k):
                    details[k] = v
    items = items or []
    notes = []
    if not items:
        notes.append("No assessment table was found. Add the assessments below; the weights should add up to 100%.")
    else:
        total = sum(i["weight"] for i in items)
        where = {"table": "the assessment table", "text": "lines of text with percentages", "ai": "an AI reading"}[method]
        notes.append(f"Found {len(items)} assessments from {where}, adding up to {total:g}%.")
        if abs(total - 100) > 0.5:
            notes.append("The weights do not add up to 100%. Check for a missing or extra row.")
        missing_max = [i["name"] for i in items if not i.get("max_mark")]
        if missing_max and len(missing_max) < len(items):
            notes.append("No 'marked out of' value for: " + ", ".join(missing_max) + ". Marks are taken as percentages.")
    for k, label in (("code", "course code"), ("name", "course name")):
        if not details[k]:
            notes.append(f"The {label} was not found; please type it in.")
    for i, it in enumerate(items):
        it["order"] = i + 1
    return {**details, "assessments": items, "notes": notes, "method": method}
