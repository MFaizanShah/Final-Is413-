"""
assistant.py
Question answering for lecturers. Every answer is computed from the live
course data, so it cannot invent numbers. It runs offline and free.

Answers come back as a small dict:
    text         short reply; **bold**, "- " bullet lines and [[CODE]] student
                 links are rendered by static/assistant.js
    links        students mentioned, so the page can link to their reports
    suggestions  follow-up questions shown as buttons
    student      the student the answer was about, remembered so a follow-up
                 such as "what is she weak in?" works

Optional: with ATRISK_LLM_ASSISTANT=1 and ANTHROPIC_API_KEY set, the computed
answer is reworded conversationally by Claude. Only the question and the
computed answer (pseudonymous codes, no real IDs) are sent, and the numbers
must be kept unchanged; if the call fails the plain answer is used.
"""

import re

import config
import pseudo

PRONOUNS = re.compile(r"\b(he|she|they|them|him|her|his|their|this student|that student|the student)\b")


def _has(q, *words):
    return any(w in q for w in words)


def _pct(v):
    return f"{v * 100:.0f}%"


def _find_student(text, rows, course_code):
    upper = text.upper()
    for r in rows:
        if r["student_id"].upper() in upper:
            return r
    # The student ID shown on the dashboard, typed as it appears.
    for r in sorted((r for r in rows if r.get("display_id")), key=lambda r: -len(r["display_id"])):
        if re.search(r"(?<![A-Za-z0-9])" + re.escape(r["display_id"].upper()) + r"(?![A-Za-z0-9])", upper):
            return r
    # A real student ID is hashed with the pseudonym key to find the record.
    codes = {pseudo.pseudonymise(course_code, tok) for tok in re.findall(r"[A-Za-z]{0,3}\d{4,}", text)}
    return next((r for r in rows if r["student_id"] in codes), None)


def _reply(text, links=(), suggestions=(), student=None):
    return {"text": text, "links": [{"student_id": s} for s in dict.fromkeys(links)],
            "suggestions": list(suggestions)[:4], "student": student}


def _line(r):
    """One line about a student for lists."""
    bits = [f"[[{r['student_id']}]]"]
    if r["p_fail"] is not None:
        bits.append(f"{_pct(r['p_fail'])} chance of failing")
    if r["current_avg"] is not None:
        bits.append(f"average {r['current_avg']:.0f}%")
    reason = r.get("concern") or (r["reasons"][0] if r.get("reasons") else "")
    return "- " + ", ".join(bits) + (f". {reason}" if reason else "")


def _list(rows, title, limit=10, empty="None."):
    if not rows:
        return title + "\n" + empty, []
    shown = rows[:limit]
    more = f"\nShowing {limit} of {len(rows)}. Ask for \"all\" to see more." if len(rows) > limit else ""
    return title + "\n" + "\n".join(_line(r) for r in shown) + more, [r["student_id"] for r in shown]


def intro(course, summary):
    c = summary["counts"]
    if not summary["rows"]:
        text = f"No current marks have been uploaded for **{course['code']}** yet, so there is nothing to ask about."
        return _reply(text)
    text = (f"Hello. I can answer questions about **{course['code']}** using its current marks. "
            f"Right now {c['At Risk']} of {summary['n']} students are at risk and {c['Borderline']} are borderline.")
    return _reply(text, suggestions=["Who should I contact first?", "Give me a class summary",
                                     "Which assessment is hardest?", "Who missed submissions?"])


def answer(question, course, summary, version, last_student=None):
    out = _answer(question, course, summary, version, last_student)
    if config.LLM_ASSISTANT and out.get("text") and question.strip():
        out["text"] = _reword(question, out["text"]) or out["text"]
    return out


def _answer(question, course, summary, version, last_student):
    raw = (question or "").strip()
    q = raw.lower()
    rows = summary["rows"]
    code = course["code"]
    c = summary["counts"]

    if not q or q in ("hi", "hello", "hey", "help", "?") or _has(q, "what can you", "how do i use", "help me"):
        return _reply(
            "You can ask me things like:\n"
            "- Who should I contact first?\n- How many students are at risk?\n- Give me a class summary\n"
            "- Why is [student code or ID] at risk?\n- What does [student] need to pass?\n"
            "- What if [student] gets 70% on the rest?\n- Write an email to [student]\n"
            "- Who missed submissions? Whose marks are falling?\n- Which assessment is hardest?\n- How accurate are the predictions?\n"
            "Use the student ID shown on the dashboard.",
            suggestions=["Who should I contact first?", "Give me a class summary"])
    if not rows:
        return _reply(f"No current semester data has been uploaded for {code} yet.")

    # ---- questions about one student
    student = _find_student(raw, rows, code)
    if not student and last_student and PRONOUNS.search(q):
        student = next((r for r in rows if r["student_id"] == last_student), None)
    if student:
        return _about_student(q, student, course)
    if re.search(r"\b[a-z]{0,3}\d{5,}\b|\bp[0-9a-f]{10}\b", q):
        return _reply("I could not find that student in " + code + ". Check the ID, or search the class list.")

    # ---- questions about the class
    limit = 50 if _has(q, " all", "every", "full list") else 10
    if _has(q, "contact first", "priorit", "urgent", "first", "start with", "focus on", "who needs help", "who should i"):
        cand = [r for r in rows if r["status"] == "At Risk" and r["feasible"] and (r["required_for_pass"] or 0) <= 70]
        cand.sort(key=lambda r: -(r["p_fail"] or 0))
        text, links = _list(cand, "These at risk students can still pass with a realistic effort (70% or less needed on "
                                  "the rest), so support now is most likely to change the outcome:", limit=min(limit, 8))
        lost = sum(1 for r in rows if r["status"] == "At Risk" and not r["feasible"])
        if lost:
            text += f"\n{lost} other student(s) can no longer reach the pass mark; ask \"who cannot pass\" to see them."
        return _reply(text, links, ["Write an email to " + links[0] if links else "Give me a class summary",
                                    "Who missed submissions?"], student=links[0] if links else None)
    if _has(q, "summary", "overview", "how is the class", "how's the class", "how are they doing", "class doing", "snapshot"):
        return _summary(course, summary)
    if _has(q, "cannot pass", "can't pass", "cant pass", "impossible", "no longer", "can not pass", "not possible"):
        lost = sorted([r for r in rows if not r["feasible"] and not r["complete"]], key=lambda r: r["best_possible"])
        text, links = _list(lost, "These students cannot reach the pass mark even with full marks on the remaining work:",
                            limit, "Every student can still pass on the marks so far.")
        return _reply(text, links, ["Who should I contact first?"])
    if _has(q, "missed", "missing", "not submit", "didn't submit", "did not submit", "non submission", "non-submission", "zero"):
        miss = [r for r in rows if any((w.get("mark") == 0) for w in r.get("weaknesses", []))]
        miss.sort(key=lambda r: -(r["p_fail"] or 0))
        text, links = _list(miss, f"{len(miss)} student(s) have at least one assessment with no submission:", limit,
                            "No missed submissions so far.")
        return _reply(text, links, ["Who should I contact first?"])
    if _has(q, "falling", "declin", "dropping", "getting worse", "trend", "slipping"):
        fall = sorted([r for r in rows if r.get("trend", 0) <= -6 and r["graded_count"] >= 3], key=lambda r: r["trend"])
        text, links = _list(fall, "Students whose marks are falling across recent assessments:", limit,
                            "Nobody's marks are falling sharply (this needs at least three marked assessments).")
        return _reply(text, links)
    if _has(q, "borderline", "watch", "monitor"):
        if _has(q, "how many", "number", "count"):
            return _reply(f"{c['Borderline']} student(s) are borderline in {code}.", suggestions=["List borderline students"])
        text, links = _list([r for r in rows if r["status"] == "Borderline"], "Borderline students:", limit)
        return _reply(text, links)
    if _has(q, "on track", "doing well", "top student", "best student", "strong"):
        good = sorted([r for r in rows if r["status"] == "On Track" and r["current_avg"] is not None],
                      key=lambda r: -r["current_avg"])
        if _has(q, "how many", "number", "count"):
            return _reply(f"{c['On Track']} student(s) are on track in {code}.")
        text, links = _list(good, "Students on track, highest average first:", limit)
        return _reply(text, links)
    if _has(q, "how many", "number of", "count", "percentage", "proportion"):
        return _reply(f"**{c['At Risk']}** of {summary['n']} students ({c['At Risk'] / summary['n']:.0%}) are at risk in {code}, "
                      f"and {c['Borderline']} more are borderline.", suggestions=["List at risk students", "Who should I contact first?"])
    if "risk" in q or _has(q, "fail", "struggling"):
        risk = [r for r in rows if r["status"] == "At Risk"]
        text, links = _list(risk, f"{len(risk)} student(s) are at risk, highest chance of failing first:", limit)
        return _reply(text, links, ["Who should I contact first?", "Who cannot pass?"])
    if _has(q, "hardest", "difficult", "lowest", "worst assessment", "toughest"):
        h = summary["hardest"]
        if not h:
            return _reply("No assessments have been marked yet.")
        return _reply(f"The lowest class average so far is **{h['name']}** at {h['class_avg']}%. "
                      f"{h['below_pass']} of {h['graded']} students scored below 50%"
                      + (f" and {h['zeros']} did not submit." if h["zeros"] else "."),
                      suggestions=["How did each assessment go?"])
    if _has(q, "each assessment", "assessments", "assessment results", "how did the"):
        lines = [f"- **{a['name']}** ({a['weight']:g}% of the course): " +
                 (f"class average {a['class_avg']}%, {a['below_pass']} below 50%" if a["graded"] else "not marked yet")
                 for a in summary["per_assessment"]]
        return _reply("Assessment results so far:\n" + "\n".join(lines))
    if _has(q, "pass rate", "how many will pass", "expected to pass"):
        hist = summary["history"]["pass_rate"]
        return _reply(f"The predicted pass rate for {code} is **{summary['pass_rate']}%**"
                      + (f", compared with {hist * 100:.0f}% in past cohorts." if hist is not None else "."))
    if _has(q, "average", "mean"):
        return _reply(f"The class average so far in {code} is **{summary['class_avg']}%**, weighted by assessment.")
    if _has(q, "distribution", "grades"):
        d = summary["grade_dist"]
        return _reply("Likely final grades: " + ", ".join(f"{g} {n}" for g, n in d.items() if n) + ".")
    if _has(q, "accura", "trust", "reliable", "model", "how does it work", "how do you know", "predict", "simulat"):
        return _accuracy(course, summary, version)

    return _reply("I did not understand that. Try one of these, or type \"help\".",
                  suggestions=["Who should I contact first?", "Give me a class summary", "How many students are at risk?"])


def _summary(course, summary):
    c, n = summary["counts"], summary["n"]
    lines = [f"**{course['code']}** has {n} students with {summary['evaluated']} marked so far.",
             f"- At risk: **{c['At Risk']}** ({c['At Risk'] / n:.0%}); borderline: {c['Borderline']}; on track: {c['On Track']}.",
             f"- Class average so far: {summary['class_avg']}%. Predicted pass rate: {summary['pass_rate']}%"
             + (f" (past cohorts {summary['history']['pass_rate'] * 100:.0f}%)." if summary["history"]["pass_rate"] is not None else ".")]
    if summary["hardest"]:
        h = summary["hardest"]
        lines.append(f"- Hardest so far: {h['name']} (class average {h['class_avg']}%).")
    miss = sum(1 for r in summary["rows"] if any(w.get("mark") == 0 for w in r.get("weaknesses", [])))
    if miss:
        lines.append(f"- {miss} student(s) have missed at least one submission.")
    return _reply("\n".join(lines), suggestions=["Who should I contact first?", "Who missed submissions?"])


def _accuracy(course, summary, version):
    if not summary["trained"]:
        return _reply(summary["source"]["text"] + " These estimates come from a simulation, not a trained model, "
                      "so treat them as a rough guide.")
    s = version["summary"]
    return _reply(
        f"Predictions for {course['code']} come from a model that learned from **{s['n_students']:,} past students** "
        f"in {s['n_courses']} courses ({s['algorithm'].lower()}).\n"
        f"- Tested on courses it had never seen, at about half way through, it picked out "
        f"**{s['headline_recall'] * 100:.0f}%** of the students who went on to fail.\n"
        f"- Of the students it flagged, {s['headline_precision'] * 100:.0f}% did fail.\n"
        "A stated chance is calibrated: \"60% chance of failing\" means about 60 in 100 similar past students failed. "
        "It is a prompt to check in, not a verdict.",
        suggestions=["Who should I contact first?"])


def _about_student(q, s, course):
    sid = s["student_id"]
    tag = f"[[{sid}]]"
    follow = ["What does this student need to pass?",
              "What is this student weak in?", "Write an email to this student"]
    if s["status"] == "No data":
        return _reply(f"{tag} has no marks uploaded yet.", [sid], student=sid)

    what_if = re.search(r"(?:gets?|scores?|averages?|achieves?|makes?)\s+(\d{1,3}(?:\.\d+)?)\s*%?", q)
    if what_if and _has(q, "if", "what"):
        target = float(what_if.group(1))
        if s["complete"]:
            return _reply(f"{tag} has already completed every assessment, with a final mark of {s['earned']}%.", [sid], student=sid)
        final = s["earned"] + s["remaining"] * min(target, 100) / 100
        g = config.grade_for_mark(final)
        verdict = "a pass" if final >= config.PASS_MARK else "still a fail"
        return _reply(f"If {tag} averages {target:g}% on the remaining {s['remaining']:.0f}% of the course, "
                      f"the final mark would be about **{final:.0f}%** (grade {g}), {verdict}.", [sid],
                      [f"What if this student gets {min(100, int(target) + 10)}%?", "What does this student need to pass?"], student=sid)

    if _has(q, "email", "message", "write to", "draft", "contact"):
        return _email(s, course)

    if _has(q, "weak", "lack", "struggl", "improve", "falling behind", "area"):
        weak = [t for t in s.get("types", []) if t["level"] == "Weak"]
        if not weak:
            return _reply(f"{tag} has no clearly weak area compared with the class.", [sid], follow[:1], student=sid)
        parts = "\n".join(f"- **{t['atype']}**: {t['avg_mark']:.0f}% average, {t['gap']:+.0f} points against the class"
                          for t in weak)
        return _reply(f"{tag} is weakest in:\n{parts}", [sid], ["What should I do for this student?"], student=sid)

    if _has(q, "what should i do", "support", "recommend", "action", "help them", "help her", "help him", "intervention"):
        recs = s.get("recommendations", [])[:4]
        if not recs:
            return _reply(f"Nothing specific is suggested for {tag}.", [sid], student=sid)
        return _reply(f"Suggested support for {tag}:\n" + "\n".join(f"- **{r['title']}.** {r['detail']}" for r in recs),
                      [sid], ["Write an email to this student"], student=sid)

    if _has(q, "need", "require", "to pass", "target"):
        if s["complete"]:
            return _reply(f"{tag} has completed every assessment with a final mark of {s['earned']}%.", [sid], student=sid)
        if not s["feasible"]:
            return _reply(f"{tag} cannot reach the pass mark: even 100% on everything left gives at most "
                          f"{s['best_possible']:.0f}%.", [sid], ["What should I do for this student?"], student=sid)
        return _reply(f"{tag} needs an average of **{max(s['required_for_pass'], 0):.0f}%** on the remaining "
                      f"{s['remaining']:.0f}% of the course to pass. The best grade still possible is {s['best_grade']}.",
                      [sid], ["What if this student gets 60%?", "What is this student weak in?"], student=sid)

    if _has(q, "mark", "score", "result"):
        lines = [f"- {w['name']}: **{w['mark']:g}%**" + (f" (class {w['class_avg']}%)" if w["class_avg"] is not None else "")
                 for w in s.get("weaknesses", [])]
        return _reply(f"Marks so far for {tag}:\n" + "\n".join(lines), [sid], follow[:2], student=sid)

    if _has(q, "grade", "chance", "predict", "likely", "pass?"):
        p = s.get("prediction")
        if not p:
            return _reply(f"{tag}: no prediction yet. Current average {s['current_avg']}%.", [sid], student=sid)
        return _reply(f"{tag} has a **{_pct(p['p_fail'])} chance of failing**. Most likely final grade "
                      f"{p['predicted_grade']}, projected mark about {p['projected_mark']:.0f}%. ({p.get('confidence_text', '')}.)",
                      [sid], follow, student=sid)

    # Default: why this status.
    lines = [f"{tag} is **{s['status'].lower()}**" + (f", with a {_pct(s['p_fail'])} chance of failing." if s["p_fail"] is not None else ".")]
    lines += [f"- {r}" for r in s["reasons"][:3]]
    if s.get("factors"):
        lines.append("Main factors: " + " ".join(f["text"] for f in s["factors"][:2]))
    return _reply("\n".join(lines), [sid], follow, student=sid)


def _email(s, course):
    weak = [t["atype"].lower() for t in s.get("types", []) if t["level"] == "Weak"]
    missing = [w["name"] for w in s.get("weaknesses", []) if w.get("mark") == 0]
    body = [f"Draft email (replace the greeting with the student's name):\n",
            f"Subject: Checking in about {course['code']}\n",
            "Dear [student name],\n",
            f"I am getting in touch because your results in {course['code']} so far suggest you may need some extra support, "
            "and I would like to help you finish the course well."]
    if missing:
        body.append(f"I noticed I have not received {', '.join(missing)}. If something has made it hard to submit, please let me know.")
    if weak:
        body.append(f"Your {' and '.join(weak)} work is where the most marks are being lost, and that is something we can work on together.")
    if s["feasible"] and not s["complete"] and s["required_for_pass"] is not None:
        body.append(f"The good news is that a pass is still within reach: an average of about "
                    f"{max(s['required_for_pass'], 0):.0f}% on the remaining work would get you there.")
    body.append("Could you come to my consultation hours this week, or reply to suggest a time? "
                "Student Learning Support also offers free sessions that many students find useful.")
    body.append("\nKind regards,\n[your name]")
    return _reply("\n".join(body), [s["student_id"]], ["What is this student weak in?"], student=s["student_id"])


def _reword(question, text):
    """Optional conversational wording by Claude. Never adds facts."""
    try:
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=config.LLM_MODEL, max_tokens=600,
            system=("You reword answers for a university lecturer using an early warning tool. Rewrite the ANSWER so it "
                    "reads naturally and briefly. Keep every number, percentage and [[CODE]] token exactly as written, "
                    "keep '- ' bullet lines and **bold**, add no new facts, no advice beyond what is given, no greeting."),
            messages=[{"role": "user", "content": f"QUESTION: {question}\n\nANSWER:\n{text}"}])
        out = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
        # Guard: reject a rewording that dropped any number or student code.
        for token in re.findall(r"\[\[[^\]]+\]\]|\d+(?:\.\d+)?%?", text):
            if token not in out:
                return None
        return out
    except Exception:
        return None
