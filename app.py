"""
app.py
USP At Risk Tracker: web application entry point.
Run:  python app.py   then open http://127.0.0.1:5000
Default logins: admin / admin123 and lecturer / lecturer123 (change them on the Users page).
The model and data live in data/atrisk.db; build it once with python build_database.py.

Roles
-----
admin     uploads, maps, validates and manages datasets, courses, lecturer
          assignments and model versions (the Admin portal).
lecturer  read only access to results for the courses assigned to them.
"""

import csv
import io
import os
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for, flash, session,
                   jsonify, Response, abort, g)

import analytics
import assistant
import config
import database as db
import ingest
import ml_engine
import outline
import pseudo
import schema_mapping as sm

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

db.init_db()
for v in db.list_model_versions():
    if v["status"] == "training":
        db.fail_model_version(v["id"], "Interrupted: the application stopped during training.")


# ---------------- helpers ----------------

def login_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        if not g.user:
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def admin_required(view):
    @wraps(view)
    @login_required
    def wrapper(*args, **kwargs):
        if g.user["role"] != "admin":
            abort(403)
        return view(*args, **kwargs)
    return wrapper


@app.before_request
def load_user():
    g.user = db.get_user(session["user_id"]) if session.get("user_id") else None


@app.context_processor
def inject():
    user = g.get("user")
    courses = db.courses_for_user(user) if user else []
    counts = {}
    for c in courses:
        try:
            counts[c["id"]] = analytics.cached_summary(c["id"])["counts"]["At Risk"]
        except Exception:
            counts[c["id"]] = 0
    is_admin = bool(user and user["role"] == "admin")
    return {"all_courses": courses, "nav_counts": counts, "GRADES": config.GRADES, "TYPES": config.TYPES,
            "is_admin": is_admin, "hosted_notice": config.IS_VERCEL and is_admin}


def course_or_404(course_id):
    course = db.get_course(course_id)
    if not course:
        abort(404)
    if not db.user_can_view(g.user, course_id):
        abort(403)
    return course


def active_model():
    version = db.get_active_model()
    return version, ml_engine.load_bundle(version)


def build_summary(course_id):
    """A course's computed results, worked out once and kept (in memory and
    in the database) until its data, the model or its settings change."""
    course = course_or_404(course_id)
    summary = analytics.cached_summary(course_id)
    return course, db.get_assessments(course_id), db.get_active_model(), summary


def _float(value, default=None):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------- auth ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = db.verify_user(request.form.get("username", ""), request.form.get("password", ""))
        if user:
            session["user_id"] = user["id"]
            nxt = request.args.get("next") or ""
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("overview"))
        flash("Incorrect username or password.", "error")
    # Show the demo accounts only while their default passwords still work.
    demo = [(u, pw, note) for u, pw, note in (("admin", "admin123", "adds courses and uploads marks"),
                                              ("lecturer", "lecturer123", "sees the courses given to them"))
            if db.verify_user(u, pw)]
    return render_template("login.html", demo_accounts=demo)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


# ---------------- results (lecturers and admins) ----------------

@app.route("/")
@login_required
def overview():
    cards = []
    totals = {"students": 0, "At Risk": 0, "Borderline": 0, "On Track": 0, "No data": 0}
    priority = []
    for c in db.courses_for_user(g.user):
        course, assessments, version, summary = build_summary(c["id"])
        cards.append({"course": course, "summary": summary, "n_assessments": len(assessments)})
        totals["students"] += summary["n"]
        for k in totals:
            if k != "students":
                totals[k] += summary["counts"].get(k, 0)
        for r in summary["rows"]:
            if r["status"] == "At Risk" and r["feasible"] and (r["required_for_pass"] or 0) <= 70:
                priority.append({"course": course, "row": r})
    # Students to contact first: at risk, but a pass is still realistic
    # (70% or less needed on the rest), so support is most likely to change
    # the outcome. Highest chance of failing first.
    priority.sort(key=lambda x: -(x["row"]["p_fail"] or 0))
    return render_template("overview.html", cards=cards, totals=totals, version=db.get_active_model(),
                           priority=priority[:8], n_priority=len(priority))


PAGE_SIZE = 25


@app.route("/course/<int:course_id>")
@login_required
def course_dashboard(course_id):
    course, assessments, version, summary = build_summary(course_id)
    view = request.args.get("view", "students")
    status_filter = request.args.get("status", "")
    q = request.args.get("q", "").strip()
    rows = summary["rows"]
    if status_filter:
        rows = [r for r in rows if r["status"] == status_filter]
    grade_filter = request.args.get("grade", "")
    if grade_filter:
        rows = [r for r in rows if r.get("predicted_grade") == grade_filter]
    if q:
        # Search matches the student ID shown, the pseudonymous code, or a
        # real ID hashed with the same key (for marks uploaded before IDs were kept).
        code = pseudo.pseudonymise(course["code"], q)
        rows = [r for r in rows if r["student_id"] == code or q.upper() in r["student_id"]
                or q.upper() in (r.get("display_id") or "").upper()]
    pages = max(1, -(-len(rows) // PAGE_SIZE))
    page = min(max(request.args.get("page", 1, type=int), 1), pages)
    shown = rows[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]
    return render_template("course_dashboard.html", course=course, assessments=assessments, version=version,
                           summary=summary, rows=shown, n_rows=len(rows), page=page, pages=pages,
                           status_filter=status_filter, grade_filter=grade_filter, q=q, view=view,
                           uploaded=bool(request.args.get("uploaded")), planned=db.get_course_outline(course))


@app.route("/course/<int:course_id>/student/<student_id>")
@login_required
def student_view(course_id, student_id):
    course, assessments, version, summary = build_summary(course_id)
    row = next((r for r in summary["rows"] if r["student_id"] == student_id), None)
    if not row:
        abort(404)
    return render_template("student.html", course=course, assessments=assessments, version=version,
                           s=row, class_avg=summary["class_avg_by_id"], source=summary["source"],
                           drift=summary["drift"], saved=True)


@app.route("/course/<int:course_id>/individual", methods=["GET", "POST"])
@login_required
def individual(course_id):
    """What-if analysis: type in marks to see the outlook. Nothing is stored."""
    course = course_or_404(course_id)
    assessments = db.get_assessments(course_id)
    version, bundle = active_model()
    if not assessments:
        flash("This course has no data yet. An administrator needs to upload it.", "error")
        return redirect(url_for("course_dashboard", course_id=course_id))
    if request.method == "POST":
        marks = {}
        for a in assessments:
            v = request.form.get(f"mark_{a['id']}", "").strip()
            if v:
                mark = _float(v)
                if mark is None or not 0 <= mark <= 100:
                    flash(f"'{v}' is not a valid mark out of 100 for {a['name']}.", "error")
                    return redirect(url_for("individual", course_id=course_id))
                marks[a["id"]] = mark
        if not marks:
            flash("Enter at least one mark.", "error")
            return redirect(url_for("individual", course_id=course_id))
        label = "What-if analysis"
        ev, cm = analytics.evaluate_single(course, assessments, db.get_students(course_id), db.get_marks(course_id),
                                           version, bundle, marks, label)
        s = {"student_id": label, **ev}
        return render_template("student.html", course=course, assessments=assessments, version=version, s=s,
                               class_avg=cm.class_avg, source=cm.source_label, drift=None, saved=False)
    return render_template("individual.html", course=course, assessments=assessments,
                           source=analytics.CourseModel(course, assessments, [], {}, version, bundle).source_label)


@app.route("/course/<int:course_id>/export")
@login_required
def export(course_id):
    course, assessments, version, summary = build_summary(course_id)
    status = request.args.get("status", "")
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["student_id", "pseudonymous_id", "status", "current_average", "predicted_grade", "chance_of_fail_pct",
                "prediction_source", "best_possible_grade", "required_avg_to_pass", "weakest_area", "reasons"])
    for r in summary["rows"]:
        if status and r["status"] != status:
            continue
        p = r.get("prediction") or {}
        w.writerow([r.get("display_id") or "", r["student_id"], r["status"], r["current_avg"], r.get("predicted_grade") or "",
                    round((r["p_fail"] or 0) * 100, 1) if r["p_fail"] is not None else "",
                    p.get("confidence_text", ""), r["best_grade"],
                    r["required_for_pass"] if r["required_for_pass"] is not None else "",
                    r["weakest"], " ".join(r["reasons"])])
    fname = f"{course['code']}_{status.replace(' ', '_') or 'all'}_students.csv"
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={fname}"})


@app.route("/model")
@login_required
def model_view():
    versions = db.list_model_versions()
    selected = request.args.get("version", type=int)
    version = db.get_model_version(selected) if selected else db.get_active_model()
    if version and version["status"] != "ready":
        version = None
    ok, readiness = ml_engine.readiness()
    return render_template("model.html", version=version, versions=versions, readiness=readiness, ready=ok,
                           training=db.model_training_in_progress(), reality=db.get_meta("reality_check"),
                           training_enabled=config.TRAINING_ENABLED,
                           load_error=(ml_engine.load_errors.get(version["id"]) if version and g.user["role"] == "admin"
                                       and ml_engine.load_bundle(version) is None else None))


# ---------------- assistant ----------------

@app.route("/assistant")
@login_required
def assistant_view():
    courses = db.courses_for_user(g.user)
    course_id = request.args.get("course_id", type=int) or (courses[0]["id"] if courses else None)
    if course_id and not db.user_can_view(g.user, course_id):
        abort(403)
    return render_template("assistant.html", courses=courses, course_id=course_id)


def _ask(course_id, question):
    course, _, version, summary = build_summary(course_id)
    ctx = session.get("chat_ctx", {})
    reply = assistant.answer(question, course, summary, version, ctx.get(str(course_id)))
    if reply.get("student"):
        ctx[str(course_id)] = reply["student"]
        session["chat_ctx"] = ctx
    # The reply names students by code (that is all the optional AI wording
    # sees); the browser shows each code as the student's ID.
    labels = {r["student_id"]: r.get("label") or r["student_id"] for r in summary["rows"]}
    for link in reply.get("links", []):
        link["href"] = url_for("student_view", course_id=course_id, student_id=link["student_id"])
        link["label"] = labels.get(link["student_id"], link["student_id"])
    reply["course"] = course["code"]
    return reply


@app.route("/api/ask", methods=["POST"])
@login_required
def api_ask_any():
    data = request.get_json(silent=True) or {}
    course_id = data.get("course_id")
    try:
        course_id = int(course_id)
    except (TypeError, ValueError):
        courses = db.courses_for_user(g.user)
        if not courses:
            return jsonify({"text": "You do not have access to any courses yet.", "suggestions": []})
        course_id = courses[0]["id"]
    if not db.user_can_view(g.user, course_id):
        abort(403)
    if data.get("intro"):
        course, _, _, summary = build_summary(course_id)
        return jsonify(assistant.intro(course, summary))
    return jsonify(_ask(course_id, data.get("question", "")))


@app.route("/api/course/<int:course_id>/ask", methods=["POST"])
@login_required
def api_ask(course_id):
    return jsonify(_ask(course_id, (request.get_json(silent=True) or {}).get("question", "")))


# ---------------- account ----------------

@app.route("/users", methods=["GET", "POST"])
@login_required
def users():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "password":
            pw = request.form.get("password", "")
            if len(pw) < 6:
                flash("Password must be at least 6 characters.", "error")
            else:
                db.set_password(g.user["id"], pw)
                flash("Password changed.", "success")
        elif g.user["role"] == "admin" and action == "create":
            pw = request.form.get("password", "")
            role = request.form.get("role", "lecturer")
            if len(pw) < 6 or not request.form.get("username"):
                flash("Username and a password of at least 6 characters are required.", "error")
            elif role not in ("lecturer", "admin"):
                flash("Unknown role.", "error")
            elif db.create_user(request.form["username"], request.form.get("full_name", ""), pw, role):
                db.audit(g.user["username"], "user_created", f"{request.form['username']} ({role})")
                flash("User created. Assign courses to lecturers in the Admin portal.", "success")
            else:
                flash("That username already exists.", "error")
        elif g.user["role"] == "admin" and action == "delete":
            target = db.get_user(int(request.form["user_id"]))
            db.delete_user(int(request.form["user_id"]))
            if target:
                db.audit(g.user["username"], "user_removed", target["username"])
            flash("User removed.", "success")
        return redirect(url_for("users"))
    return render_template("users.html", users=db.list_users() if g.user["role"] == "admin" else [])


# ---------------- admin portal ----------------

@app.route("/admin")
@admin_required
def admin_home():
    rows = []
    version = db.get_active_model()
    trained_courses = set((version or {}).get("summary", {}).get("courses", []))
    datasets = db.list_datasets()
    for c in db.list_courses(training=False):
        current = next((d for d in datasets if d["course_id"] == c["id"] and d["kind"] == "current" and d["is_active"]), None)
        hist = db.course_history_stats(c["code"])
        rows.append({"course": c, "current": current, "history": hist, "lecturers": db.get_lecturers(c["id"]),
                     "in_model": c["code"] in trained_courses})
    ok, readiness = ml_engine.readiness()
    library = [{"course": c, "history": db.course_history_stats(c["code"]), "in_model": c["code"] in trained_courses,
                "is_set": c["code"].startswith(db.TRAINING_PREFIX)} for c in db.list_courses(training=True)]
    return render_template("admin/home.html", rows=rows, library=library, version=version, audit=db.list_audit(15),
                           readiness=readiness, ready=ok, training=db.model_training_in_progress())


@app.route("/admin/demo", methods=["POST"])
@admin_required
def admin_demo():
    for msg in ingest.load_demo(g.user["username"]):
        flash(msg, "info")
    flash("Demo courses loaded with fictional data. ED200 has no history, to show the simulated estimate path. "
          "The model is training in the background.", "success")
    return redirect(url_for("admin_home"))


@app.route("/admin/course/new")
@admin_required
def admin_course_new():
    return redirect(url_for("course_new"))


def _blank_course():
    return {"code": "", "name": "", "semester": "", "campus": "", "assessments": [], "notes": [],
            "risk_threshold": 40, "borderline_threshold": 20, "filename": None}


def _course_from_form(form):
    """Reads the confirmation screen back into a course and its assessments."""
    data = {k: form.get(k, "").strip() for k in ("code", "name", "semester", "campus", "filename")}
    data["code"] = data["code"].upper().replace(" ", "")
    data["risk_threshold"] = _float(form.get("risk_threshold"), 40)
    data["borderline_threshold"] = _float(form.get("borderline_threshold"), 20)
    items = []
    for i in range(int(_float(form.get("rows"), 0) or 0)):
        name = form.get(f"a_name_{i}", "").strip()
        if not name:
            continue
        items.append({"name": name, "atype": form.get(f"a_type_{i}") if form.get(f"a_type_{i}") in config.TYPES else config.detect_type(name),
                      "weight": _float(form.get(f"a_weight_{i}")), "max_mark": _float(form.get(f"a_max_{i}")),
                      "due": form.get(f"a_due_{i}", "").strip() or None})
    data["assessments"] = items
    data["notes"] = []
    return data


def _course_errors(data):
    errors = []
    if not data["code"] or not data["name"]:
        errors.append("Enter the course code and name.")
    if data["code"].startswith(db.TRAINING_PREFIX):
        errors.append(f"Course codes cannot start with {db.TRAINING_PREFIX}; that prefix is kept for training sets.")
    if any(c["code"] == data["code"] for c in db.list_courses(training=False)):
        errors.append(f"A course with the code {data['code']} already exists. Open it from the course list instead.")
    if not (5 <= data["risk_threshold"] <= 95 and 1 <= data["borderline_threshold"] <= 90) \
            or data["borderline_threshold"] >= data["risk_threshold"]:
        errors.append("The borderline threshold must be lower than the at risk threshold.")
    items = data["assessments"]
    if not items:
        errors.append("Add at least one assessment.")
    names = [a["name"].lower() for a in items]
    if len(set(names)) != len(names):
        errors.append("Two assessments have the same name. Give each a different name.")
    if any(a["weight"] is None or not 0 < a["weight"] <= 100 for a in items):
        errors.append("Every assessment needs a weight above 0%.")
    elif items and abs(sum(a["weight"] for a in items) - 100) > 0.5:
        errors.append(f"The weights add up to {sum(a['weight'] for a in items):g}%. They must add up to 100%.")
    if any(a["max_mark"] is not None and a["max_mark"] <= 0 for a in items):
        errors.append("A 'marked out of' value must be above 0.")
    return errors


@app.route("/courses/new", methods=["GET", "POST"])
@login_required
def course_new():
    """Step 1 for admins and lecturers: upload the course outline (or type the
    assessments in), confirm what was read, set the thresholds, create."""
    lecturers = [u for u in db.list_users() if u["role"] == "lecturer"] if g.user["role"] == "admin" else []
    if request.method == "GET":
        return render_template("course_new.html", stage="upload", data=_blank_course(), lecturers=lecturers)
    action = request.form.get("action")
    if action == "read":
        f = request.files.get("outline")
        if not f or not f.filename:
            flash("Choose the course outline file first, or type the assessments in.", "error")
            return render_template("course_new.html", stage="upload", data=_blank_course(), lecturers=lecturers)
        try:
            data = {**_blank_course(), **outline.parse(f.filename, f.read()), "filename": f.filename}
        except ValueError as exc:
            flash(str(exc), "error")
            return render_template("course_new.html", stage="upload", data=_blank_course(), lecturers=lecturers)
        return render_template("course_new.html", stage="confirm", data=data, lecturers=lecturers)
    if action == "manual":
        data = _blank_course()
        data["assessments"] = [{"name": "", "atype": "Coursework", "weight": None, "max_mark": None, "due": None}
                               for _ in range(4)]
        return render_template("course_new.html", stage="confirm", data=data, lecturers=lecturers)

    # action == "create"
    data = _course_from_form(request.form)
    errors = _course_errors(data)
    if errors:
        for e in errors:
            flash(e, "error")
        return render_template("course_new.html", stage="confirm", data=data, lecturers=lecturers,
                               chosen=[int(x) for x in request.form.getlist("lecturer")])
    cid = db.create_course(data["code"], data["name"], data["semester"], data["campus"],
                           risk_threshold=data["risk_threshold"] / 100,
                           borderline_threshold=data["borderline_threshold"] / 100)
    structure = [{"name": a["name"], "atype": a["atype"], "weight": a["weight"], "max_mark": a["max_mark"] or 100.0,
                  "due": a["due"], "order": i + 1} for i, a in enumerate(data["assessments"])]
    db.set_course_outline(cid, {"assessments": structure, "filename": data.get("filename") or None,
                                "added_by": g.user["username"]})
    # Saved as the course's structure, so a marks file that only names the
    # assessments (no weights in the headings) is still understood.
    db.save_profile(data["code"], "current", {
        "layout": "wide", "column_roles": {},
        "assessments": {sm.key(a["name"]): {"name": a["name"], "weight": a["weight"], "atype": a["atype"],
                                            "max_mark": a["max_mark"], "due_date": None} for a in structure}},
        g.user["username"])
    ids = [int(x) for x in request.form.getlist("lecturer")] if g.user["role"] == "admin" else [g.user["id"]]
    if ids:
        db.set_lecturers(cid, ids)
    src = f"from {data['filename']}" if data.get("filename") else "entered by hand"
    db.audit(g.user["username"], "course_created",
             f"{data['code']} {data['name']}; {len(structure)} assessments {src}; at risk from "
             f"{data['risk_threshold']:.0f}%, borderline from {data['borderline_threshold']:.0f}%", course_id=cid)
    flash(f"{data['code']} created with {len(structure)} assessments.", "success")
    return redirect(url_for("admin_upload", course_id=cid, kind="current"))


@app.route("/course/<int:course_id>/marks-template.csv")
@login_required
def marks_template(course_id):
    """A marksheet ready to fill in, with this course's assessments and weights."""
    course = course_or_404(course_id)
    items = (db.get_course_outline(course) or {}).get("assessments") or [
        {"name": a["name"], "weight": a["weight"], "max_mark": a["max_mark"]} for a in db.get_assessments(course_id)]
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["student_id"] + [f"{a['name']} ({a['weight']:g}%) /{(a.get('max_mark') or 100):g}" for a in items])
    w.writerow(["S11000001"] + [""] * len(items))
    return Response(out.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename={course['code']}_marksheet_template.csv"})


@app.route("/admin/course/<int:course_id>", methods=["GET", "POST"])
@admin_required
def admin_course(course_id):
    course = course_or_404(course_id)
    if request.method == "POST":
        action = request.form.get("action")
        user = g.user["username"]
        if action == "details":
            risk = _float(request.form.get("risk_threshold"), 40) / 100
            border = _float(request.form.get("borderline_threshold"), 20) / 100
            if border >= risk:
                flash("The borderline threshold must be lower than the at risk threshold.", "error")
                return redirect(url_for("admin_course", course_id=course_id))
            db.update_course(course_id, code=request.form.get("code", "").upper() or None, name=request.form.get("name"),
                             semester=request.form.get("semester"), campus=request.form.get("campus"),
                             risk_threshold=risk, borderline_threshold=border)
            db.audit(user, "course_updated", f"Details and thresholds (at risk {risk:.0%}, borderline {border:.0%})",
                     course_id=course_id)
            flash("Course settings saved.", "success")
        elif action == "lecturers":
            ids = [int(x) for x in request.form.getlist("lecturer")]
            db.set_lecturers(course_id, ids)
            names = [u["username"] for u in db.list_users() if u["id"] in ids]
            db.audit(user, "lecturers_assigned", ", ".join(names) or "none", course_id=course_id)
            flash("Lecturer access updated.", "success")
        elif action == "types":
            types = {int(k[5:]): v for k, v in request.form.items() if k.startswith("type_") and v in config.TYPES}
            db.update_assessment_types(course_id, types)
            db.audit(user, "assessment_types_updated", "Types changed on the course page", course_id=course_id)
            flash("Assessment types updated. They take effect immediately for weakness analysis and predictions.", "success")
        elif action == "delete":
            db.audit(user, "course_deleted", f"{course['code']} {course['name']}", course_id=None)
            db.delete_course(course_id)
            flash("Course deleted with all its datasets.", "success")
            return redirect(url_for("admin_home"))
        return redirect(url_for("admin_course", course_id=course_id))
    lecturers = {u["id"] for u in db.get_lecturers(course_id)}
    return render_template("admin/course.html", course=course, assessments=db.get_assessments(course_id),
                           datasets=db.list_datasets(course_id), users=db.list_users(), lecturers=lecturers,
                           history=db.course_history_stats(course["code"]), audit=db.list_audit(30, course_id),
                           profile=db.get_profile(course["code"], "current"))


@app.route("/admin/dataset/<int:dataset_id>/<action>", methods=["POST"])
@admin_required
def admin_dataset(dataset_id, action):
    ds = db.get_dataset(dataset_id)
    if not ds or action not in ("activate", "deactivate"):
        abort(404)
    if action == "activate":
        db.activate_dataset(dataset_id)
        verb = "activated (rolled back or forward to this version)"
    else:
        db.deactivate_dataset(dataset_id)
        verb = "deactivated"
    db.audit(g.user["username"], f"dataset_{action}d", f"{ds['kind'].capitalize()} '{ds['label']}' version {ds['version']} {verb}",
             course_id=ds["course_id"], dataset_id=dataset_id)
    flash(f"Dataset version {ds['version']} {verb}.", "success")
    if ds["kind"] == "historical":
        model_id, msg = ml_engine.retrain(g.user["username"], f"Historical dataset {action}d")
        flash(("Retraining the model on the updated history. " if model_id else "") + msg, "info")
    return redirect(url_for("admin_course", course_id=ds["course_id"]))


@app.route("/admin/upload", methods=["GET", "POST"])
@login_required
def admin_upload():
    """This semester's marks for a teaching course. Admins upload for any
    course, lecturers for the courses they added or were given. Past results
    are added as a named training set instead (admin_training_new)."""
    courses = db.courses_for_user(g.user)
    if request.method == "POST":
        f = request.files.get("file")
        course_id = request.form.get("course_id", type=int)
        if not f or not f.filename or not course_id:
            flash("Choose a course and a file.", "error")
            return redirect(url_for("admin_upload", course_id=course_id))
        course = course_or_404(course_id)
        if course.get("is_training"):
            return redirect(url_for("admin_training_new", name=course["name"]))
        token, err = ingest.stage(f.filename, f.read(), course_id, "current", request.form.get("label", ""),
                                  g.user["username"])
        if err:
            flash(err, "error")
            return redirect(url_for("admin_upload", course_id=course_id))
        return redirect(url_for("admin_mapping", token=token))
    if request.args.get("kind") == "historical" and g.user["role"] == "admin":
        return redirect(url_for("admin_training_new"))
    return render_template("admin/upload.html", courses=courses, course_id=request.args.get("course_id", type=int))


TRAINING_NAME_MAX = 60


@app.route("/admin/training")
@admin_required
def admin_training():
    return redirect(url_for("admin_training_new"))


@app.route("/admin/training/new", methods=["GET", "POST"])
@admin_required
def admin_training_new():
    """Adds a file of past results under a name. The set joins the training
    history, the model retrains, and every course's predictions use it.
    Reusing a set's name adds the file to it as another cohort."""
    form = {"name": request.values.get("name", "").strip(), "cohort": request.values.get("cohort", "").strip()}
    if request.method == "POST":
        f = request.files.get("file")
        problems = []
        if not form["name"]:
            problems.append("Give the training set a name.")
        elif len(form["name"]) > TRAINING_NAME_MAX:
            problems.append(f"The name must be {TRAINING_NAME_MAX} characters or fewer.")
        if not f or not f.filename:
            problems.append("Choose a file of past results (.csv or .xlsx).")
        elif not f.filename.lower().endswith((".csv", ".xlsx", ".xls")):
            problems.append("The file must be a .csv or .xlsx file.")
        if not problems:
            existing = db.find_training_set(form["name"])
            course = existing or db.create_training_set(form["name"])
            labels = {d["label"].lower() for d in db.list_datasets(course["id"])}
            # Every file is another cohort of the set, so a repeated or empty
            # cohort name is made unique rather than replacing earlier results.
            label = form["cohort"] or f"Cohort {len(labels) + 1}"
            n, base = 1, label
            while label.lower() in labels:
                n += 1
                label = f"{base} ({n})"
            token, err = ingest.stage(f.filename, f.read(), course["id"], "historical", label, g.user["username"])
            if not err:
                return redirect(url_for("admin_mapping", token=token))
            if not existing:
                db.delete_course(course["id"])
            problems.append(err)
        for p in problems:
            flash(p, "error")
    ingest.purge_old()
    return render_template("admin/training_new.html", form=form, name_max=TRAINING_NAME_MAX,
                           sets=[c for c in db.list_courses(training=True) if c["code"].startswith(db.TRAINING_PREFIX)],
                           training_enabled=config.TRAINING_ENABLED)


def _after_upload(staged):
    if staged["kind"] == "historical":
        return url_for("admin_home")
    if g.user["role"] == "admin":
        return url_for("admin_course", course_id=staged["course_id"])
    return url_for("course_dashboard", course_id=staged["course_id"])


@app.route("/admin/upload/<token>", methods=["GET", "POST"])
@login_required
def admin_mapping(token):
    try:
        staged = ingest.load_staged(token)
    except (OSError, KeyError):
        flash("That upload has expired or was already saved. Please upload the file again.", "error")
        return redirect(url_for("admin_upload"))
    if not db.user_can_view(g.user, staged["course_id"]) or (g.user["role"] != "admin" and staged["kind"] != "current"):
        abort(403)
    if request.method == "POST":
        action = request.form.get("action")
        if action == "cancel":
            ingest.discard(token)
            course = db.get_course(staged["course_id"])
            if course and course.get("is_training") and course["code"].startswith(db.TRAINING_PREFIX)                     and not db.course_has_datasets(course["id"]):
                db.delete_course(course["id"])
            flash("Upload cancelled. Nothing was saved.", "info")
            return redirect(_after_upload(staged))
        ingest.apply_form(staged, request.form)
        ingest.save_staged(token, staged)
        if action == "save":
            result, err = ingest.commit(token, staged, g.user["username"])
            if err:
                flash(err, "error")
            else:
                flash(f"Saved version {result['version']}: {result['students']} students and "
                      f"{result['assessments']} assessments. Student IDs were pseudonymised.", "success")
                if result["retrain"]:
                    flash(result["retrain"]["message"] + (" The model page shows progress." if result["retrain"]["model_id"] else ""),
                          "info")
                if staged["kind"] == "current":
                    # Straight to the results for what was just uploaded.
                    return redirect(url_for("course_dashboard", course_id=staged["course_id"], uploaded=1))
                course = db.get_course(staged["course_id"])
                flash(f"Training set '{course['name']}' now includes {result['students']} past students from "
                      f"'{staged['label']}'. When the model finishes retraining, predictions for every course use it.",
                      "success")
                return redirect(_after_upload(staged))
        return redirect(url_for("admin_mapping", token=token))
    result = sm.build(staged, staged["mapping"])
    preview = staged["data"][:8]
    return render_template("admin/mapping.html", token=token, staged=staged, mapping=staged["mapping"],
                           result=result, preview=preview, course=db.get_course(staged["course_id"]),
                           roles=sm.WIDE_ROLES if staged["mapping"]["layout"] == "wide" else sm.LONG_ROLES,
                           role_labels=sm.ROLE_LABELS, llm=config.LLM_SUGGESTIONS)


@app.route("/admin/models", methods=["GET", "POST"])
@admin_required
def admin_models():
    if request.method == "POST":
        action = request.form.get("action")
        if action == "retrain":
            model_id, msg = ml_engine.retrain(g.user["username"], "Manual retrain")
            flash(msg, "success" if model_id else "error")
        elif action == "activate":
            mid = int(request.form["model_id"])
            db.activate_model_version(mid)
            db.audit(g.user["username"], "model_activated", f"Version {mid} made active (rollback)", model_id=mid)
            flash(f"Model version {mid} is now active.", "success")
        return redirect(url_for("model_view"))
    return redirect(url_for("model_view"))


@app.route("/admin/audit")
@admin_required
def admin_audit():
    return render_template("admin/audit.html", audit=db.list_audit(500))


@app.route("/admin/reidentify", methods=["GET", "POST"])
@admin_required
def admin_reidentify():
    """Adds pseudonymous codes to a class list the admin already holds, so
    results can be matched back to students. Nothing is stored."""
    courses = db.list_courses()
    if request.method == "POST":
        f = request.files.get("file")
        course = course_or_404(request.form.get("course_id", type=int))
        if not f or not f.filename:
            flash("Choose a class list file.", "error")
            return redirect(url_for("admin_reidentify"))
        cols, data, _, err = sm.read_upload(f.filename, f.read())
        if err:
            flash(err, "error")
            return redirect(url_for("admin_reidentify"))
        id_i = next((i for i, c in enumerate(cols) if sm.key(c) in sm.ID_ALIASES), None)
        if id_i is None:
            flash("The file needs a student ID column.", "error")
            return redirect(url_for("admin_reidentify"))
        out = io.StringIO()
        w = csv.writer(out)
        w.writerow(["pseudonymous_id"] + cols)
        for row in data:
            w.writerow([pseudo.pseudonymise(course["code"], row[id_i])] + row)
        db.audit(g.user["username"], "reidentification_key_generated",
                 f"{len(data)} IDs from {f.filename}; file returned to the admin, nothing stored", course_id=course["id"])
        return Response(out.getvalue(), mimetype="text/csv",
                        headers={"Content-Disposition": f"attachment; filename={course['code']}_pseudonym_key.csv"})
    return render_template("admin/reidentify.html", courses=courses)


@app.route("/admin/template/<kind>")
@login_required
def admin_template(kind):
    if kind not in ("current", "historical", "long"):
        abort(404)
    return Response(sm.template_csv(kind), mimetype="text/csv",
                    headers={"Content-Disposition": f"attachment; filename=template_{kind}.csv"})


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", message="That page or record could not be found."), 404


@app.errorhandler(403)
def forbidden(e):
    return render_template("error.html", message="You do not have access to that page."), 403


if __name__ == "__main__":
    app.run(debug=os.environ.get("ATRISK_DEBUG") == "1", port=int(os.environ.get("PORT", 5000)))
