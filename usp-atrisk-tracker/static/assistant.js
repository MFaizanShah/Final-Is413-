/* assistant.js: chat component used by the Assistant page and the
   "Ask a question" panel on every page. */
(function () {
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; });
  }
  // **bold**, "- " bullets and [[CODE]] student links.
  function render(text, links) {
    var hrefs = {};
    (links || []).forEach(function (l) { hrefs[l.student_id] = l.href; });
    var html = "", inList = false;
    String(text).split("\n").forEach(function (line) {
      var t = esc(line)
        .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")
        .replace(/\[\[([^\]]+)\]\]/g, function (_, code) {
          return hrefs[code] ? '<a href="' + hrefs[code] + '">' + code + "</a>" : "<b>" + code + "</b>";
        });
      if (/^- /.test(line)) {
        if (!inList) { html += "<ul>"; inList = true; }
        html += "<li>" + t.slice(2) + "</li>";
      } else {
        if (inList) { html += "</ul>"; inList = false; }
        if (t.trim()) html += "<p>" + t + "</p>";
      }
    });
    if (inList) html += "</ul>";
    return html;
  }

  function mount(o) {
    var log = document.getElementById(o.log), form = document.getElementById(o.form),
        input = document.getElementById(o.input), sugg = document.getElementById(o.suggest),
        courseSel = o.course ? document.getElementById(o.course) : null;
    if (!log || !form) return;

    function courseId() { return courseSel ? courseSel.value : o.courseId; }
    function add(cls, html) {
      var d = document.createElement("div");
      d.className = "msg " + cls;
      d.innerHTML = html;
      log.appendChild(d);
      log.scrollTop = log.scrollHeight;
      return d;
    }
    function suggestions(list) {
      sugg.innerHTML = "";
      (list || []).forEach(function (s) {
        var b = document.createElement("button");
        b.type = "button"; b.className = "chip"; b.textContent = s;
        b.onclick = function () { ask(s); };
        sugg.appendChild(b);
      });
    }
    function post(body) {
      return fetch(o.endpoint, { method: "POST", headers: { "Content-Type": "application/json" }, credentials: "same-origin",
                                 body: JSON.stringify(body) }).then(function (r) {
        if (!r.ok) throw new Error(r.status);
        return r.json();
      });
    }
    function ask(q) {
      q = (q || "").trim();
      if (!q) return;
      add("me", esc(q));
      input.value = "";
      var wait = add("bot typing", "Working it out");
      post({ question: q, course_id: courseId() }).then(function (res) {
        wait.className = "msg bot";
        wait.innerHTML = render(res.text, res.links);
        suggestions(res.suggestions);
        log.scrollTop = log.scrollHeight;
      }).catch(function () {
        wait.className = "msg bot";
        wait.textContent = "That question could not be answered. Check your connection and sign in again if needed.";
      });
    }
    function intro() {
      log.innerHTML = "";
      post({ intro: true, course_id: courseId() }).then(function (res) {
        add("bot", render(res.text, res.links));
        suggestions(res.suggestions);
      }).catch(function () {});
    }
    form.addEventListener("submit", function (e) { e.preventDefault(); ask(input.value); });
    if (courseSel) courseSel.addEventListener("change", intro);
    intro();
    return { ask: ask };
  }

  window.AtRiskChat = { mount: mount, render: render };
})();
