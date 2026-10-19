/*
 * charts.js
 * A small, dependency free SVG chart library for the USP At Risk Tracker.
 * Works fully offline. Supports bar (vertical, horizontal, grouped),
 * line and scatter charts, each with hover tooltips.
 */
(function () {
  var NS = "http://www.w3.org/2000/svg";
  var INK2 = "#44566b", GRID = "#e6ebf0", AXIS = "#c3cdd7";

  function el(tag, attrs, parent) {
    var e = document.createElementNS(NS, tag);
    for (var k in attrs) e.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(e);
    return e;
  }
  function fmt(v, f) { if (v === null || v === undefined || isNaN(v)) return "-"; return f ? f(v) : (Math.round(v * 10) / 10).toString(); }
  // Maximum that divides into four whole, readable ticks.
  function niceMax(v) {
    if (v <= 0) return 4;
    var raw = v / 4, p = Math.pow(10, Math.floor(Math.log10(raw))), n = raw / p;
    var m = n <= 1 ? 1 : n <= 2 ? 2 : n <= 2.5 ? 2.5 : n <= 5 ? 5 : 10;
    var step = m * p;
    if (step < 1) step = 1;
    if (step % 1 !== 0 && step > 1) step = Math.ceil(step);
    return step * 4;
  }
  // Bar with rounded data end (4px) and a square baseline.
  function barPath(x, y, w, h, horizontal) {
    var r = Math.min(4, w / 2, h);
    if (h <= 0 || w <= 0) return "";
    if (!horizontal) {
      return "M" + x + "," + (y + h) + "V" + (y + r) + "Q" + x + "," + y + " " + (x + r) + "," + y +
        "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) + "V" + (y + h) + "Z";
    }
    r = Math.min(4, h / 2, w);
    return "M" + x + "," + y + "H" + (x + w - r) + "Q" + (x + w) + "," + y + " " + (x + w) + "," + (y + r) +
      "V" + (y + h - r) + "Q" + (x + w) + "," + (y + h) + " " + (x + w - r) + "," + (y + h) + "H" + x + "Z";
  }
  function setup(container, opts) {
    container.innerHTML = "";
    container.classList.add("chart");
    if (opts.series && opts.series.length > 1 && opts.legend !== false) {
      var lg = document.createElement("div"); lg.className = "legend";
      opts.series.forEach(function (s) { lg.innerHTML += '<span><i style="background:' + s.color + '"></i>' + s.name + "</span>"; });
      container.appendChild(lg);
    }
    if (opts.groups) {
      var lg2 = document.createElement("div"); lg2.className = "legend";
      opts.groups.forEach(function (s) { lg2.innerHTML += '<span><i style="background:' + s.color + ';border-radius:50%"></i>' + s.name + "</span>"; });
      container.appendChild(lg2);
    }
    var tip = document.createElement("div"); tip.className = "chart-tip"; container.appendChild(tip);
    var W = Math.max(container.clientWidth, 260), H = opts.height || 260;
    var svg = el("svg", { width: W, height: H, viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.title || "chart" });
    container.appendChild(svg);
    function show(evt, html) {
      tip.innerHTML = html; tip.style.display = "block";
      var rect = container.getBoundingClientRect();
      var x = evt.clientX - rect.left + 14, y = evt.clientY - rect.top - 10;
      if (x + tip.offsetWidth > rect.width) x = evt.clientX - rect.left - tip.offsetWidth - 14;
      tip.style.left = x + "px"; tip.style.top = Math.max(0, y) + "px";
    }
    function hide() { tip.style.display = "none"; }
    return { svg: svg, W: W, H: H, show: show, hide: hide };
  }
  function text(svg, x, y, str, attrs) {
    var t = el("text", Object.assign({ x: x, y: y, "font-size": 11, fill: INK2, "font-family": "Source Sans 3, Segoe UI, Arial, sans-serif" }, attrs || {}), svg);
    t.textContent = str; return t;
  }

  function bar(container, opts) {
    var c = setup(container, opts), svg = c.svg;
    var labels = opts.labels, series = opts.series, horiz = !!opts.horizontal;
    var all = []; series.forEach(function (s) { s.values.forEach(function (v) { if (v !== null) all.push(v); }); });
    var max = opts.max || niceMax(Math.max.apply(null, all.concat([0])) * 1.08);
    var padL = horiz ? (opts.labelWidth || 130) : 40, padR = horiz ? 40 : 12, padT = 16, padB = horiz ? 24 : 42;
    var longest = Math.max.apply(null, labels.map(function (l) { return String(l).length; }));
    var rotate = !horiz && longest * 6.2 > (c.W - padL - padR) / labels.length - 6;
    if (rotate) padB = Math.min(96, 24 + longest * 4.2);
    var pw = c.W - padL - padR, ph = c.H - padT - padB;
    var ticks = 4;
    for (var i = 0; i <= ticks; i++) {
      var val = max * i / ticks;
      if (!horiz) {
        var y = padT + ph - ph * i / ticks;
        el("line", { x1: padL, x2: padL + pw, y1: y, y2: y, stroke: i === 0 ? AXIS : GRID, "stroke-width": 1 }, svg);
        text(svg, padL - 6, y + 4, fmt(val, opts.tickFormat), { "text-anchor": "end" });
      } else {
        var x = padL + pw * i / ticks;
        el("line", { x1: x, x2: x, y1: padT, y2: padT + ph, stroke: i === 0 ? AXIS : GRID, "stroke-width": 1 }, svg);
        text(svg, x, padT + ph + 16, fmt(val, opts.tickFormat), { "text-anchor": "middle" });
      }
    }
    var band = (horiz ? ph : pw) / labels.length;
    var inner = band * 0.72, gap = 2, bw = (inner - gap * (series.length - 1)) / series.length;
    labels.forEach(function (lab, li) {
      var start = (horiz ? padT : padL) + li * band + (band - inner) / 2;
      series.forEach(function (s, si) {
        var v = s.values[li]; if (v === null || v === undefined) return;
        var len = (horiz ? pw : ph) * Math.max(0, v) / max;
        var color = (s.colors && s.colors[li]) || s.color;
        var p = horiz ? barPath(padL, start + si * (bw + gap), len, bw, true)
                      : barPath(start + si * (bw + gap), padT + ph - len, bw, len, false);
        if (p) el("path", { d: p, fill: color }, svg);
      });
      if (!horiz) {
        var lx = padL + li * band + band / 2;
        var t = text(svg, lx, padT + ph + 16, lab, { "text-anchor": rotate ? "end" : "middle" });
        if (rotate) t.setAttribute("transform", "rotate(-30 " + lx + " " + (padT + ph + 16) + ")");
      } else {
        var shown = lab.length > 22 ? lab.slice(0, 21) + "…" : lab;
        text(svg, padL - 8, padT + li * band + band / 2 + 4, shown, { "text-anchor": "end" });
        if (opts.directLabels) {
          var v0 = series[0].values[li];
          text(svg, padL + pw * Math.max(0, v0) / max + 6, padT + li * band + band / 2 + 4, fmt(v0, opts.valueFormat), { fill: "#13253A" });
        }
      }
      if (!horiz && opts.directLabels && series.length === 1) {
        var v1 = series[0].values[li];
        if (v1) text(svg, padL + li * band + band / 2, padT + ph - ph * v1 / max - 5, fmt(v1, opts.valueFormat), { "text-anchor": "middle", fill: "#13253A" });
      }
      var hit = horiz ? el("rect", { x: padL, y: padT + li * band, width: pw, height: band, fill: "transparent" }, svg)
                      : el("rect", { x: padL + li * band, y: padT, width: band, height: ph, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", function (e) {
        var h = "<b>" + lab + "</b>";
        series.forEach(function (s) { h += (series.length > 1 ? s.name + ": " : "") + fmt(s.values[li], opts.valueFormat) + "<br>"; });
        if (opts.extra) h += opts.extra[li] || "";
        c.show(e, h);
        hit.setAttribute("fill", "rgba(11,46,89,0.05)");
      });
      hit.addEventListener("mouseleave", function () { c.hide(); hit.setAttribute("fill", "transparent"); });
      if (opts.onClick) { hit.style.cursor = "pointer"; hit.addEventListener("click", function () { opts.onClick(li); }); }
    });
    if (opts.refLine !== undefined) {
      var rv = opts.refLine.value;
      if (!horiz) {
        var ry = padT + ph - ph * rv / max;
        el("line", { x1: padL, x2: padL + pw, y1: ry, y2: ry, stroke: "#13253A", "stroke-dasharray": "4 4", "stroke-width": 1 }, svg);
        text(svg, padL + pw, ry - 4, opts.refLine.label, { "text-anchor": "end", fill: "#13253A" });
      }
    }
    if (opts.yLabel && !horiz) text(svg, 4, 10, opts.yLabel, { "font-size": 10 });
  }

  function line(container, opts) {
    var c = setup(container, opts), svg = c.svg;
    var labels = opts.labels, series = opts.series;
    var max = opts.max !== undefined ? opts.max : niceMax(Math.max.apply(null, series.flatMap(function (s) { return s.values.filter(function (v) { return v !== null; }); })));
    var min = opts.min || 0;
    var padL = 40, padR = 24, padT = 12, padB = 42;
    var longest = Math.max.apply(null, labels.map(function (l) { return String(l).length; }));
    var rotate = longest * 6.2 > (c.W - padL - padR) / Math.max(labels.length - 1, 1) - 6;
    if (rotate) padB = Math.min(96, 24 + longest * 4.2);
    var pw = c.W - padL - padR, ph = c.H - padT - padB;
    function X(i) { return padL + (labels.length === 1 ? pw / 2 : pw * i / (labels.length - 1)); }
    function Y(v) { return padT + ph - ph * (v - min) / (max - min); }
    for (var i = 0; i <= 4; i++) {
      var val = min + (max - min) * i / 4, y = Y(val);
      el("line", { x1: padL, x2: padL + pw, y1: y, y2: y, stroke: i === 0 ? AXIS : GRID }, svg);
      text(svg, padL - 6, y + 4, fmt(val, opts.tickFormat), { "text-anchor": "end" });
    }
    labels.forEach(function (lab, li) {
      var t = text(svg, X(li), padT + ph + 16, lab, { "text-anchor": rotate ? "end" : "middle" });
      if (rotate) t.setAttribute("transform", "rotate(-30 " + X(li) + " " + (padT + ph + 16) + ")");
    });
    if (opts.refLine) {
      var ry = Y(opts.refLine.value);
      el("line", { x1: padL, x2: padL + pw, y1: ry, y2: ry, stroke: "#13253A", "stroke-dasharray": "4 4" }, svg);
      text(svg, padL + pw, ry - 4, opts.refLine.label, { "text-anchor": "end", fill: "#13253A" });
    }
    series.forEach(function (s) {
      var d = "", started = false;
      s.values.forEach(function (v, i) { if (v === null) { started = false; return; } d += (started ? "L" : "M") + X(i) + "," + Y(v); started = true; });
      el("path", { d: d, fill: "none", stroke: s.color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-dasharray": s.dashed ? "5 4" : "" }, svg);
      s.values.forEach(function (v, i) { if (v !== null) el("circle", { cx: X(i), cy: Y(v), r: 4, fill: s.color, stroke: "#fff", "stroke-width": 2 }, svg); });
    });
    var cross = el("line", { y1: padT, y2: padT + ph, stroke: AXIS, "stroke-width": 1, visibility: "hidden" }, svg);
    var step = labels.length > 1 ? pw / (labels.length - 1) : pw;
    labels.forEach(function (lab, li) {
      var hit = el("rect", { x: X(li) - step / 2, y: padT, width: step, height: ph, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", function (e) {
        cross.setAttribute("x1", X(li)); cross.setAttribute("x2", X(li)); cross.setAttribute("visibility", "visible");
        var h = "<b>" + lab + "</b>";
        series.forEach(function (s) { h += '<span style="color:' + s.color + '">&#9679;</span> ' + s.name + ": " + fmt(s.values[li], opts.valueFormat) + "<br>"; });
        c.show(e, h);
      });
      hit.addEventListener("mouseleave", function () { cross.setAttribute("visibility", "hidden"); c.hide(); });
    });
  }

  function scatter(container, opts) {
    var c = setup(container, opts), svg = c.svg;
    var padL = 44, padR = 16, padT = 22, padB = 40, pw = c.W - padL - padR, ph = c.H - padT - padB;
    var xMax = opts.xMax || 100, yMax = opts.yMax || 100;
    function X(v) { return padL + pw * v / xMax; }
    function Y(v) { return padT + ph - ph * v / yMax; }
    for (var i = 0; i <= 4; i++) {
      el("line", { x1: padL, x2: padL + pw, y1: Y(yMax * i / 4), y2: Y(yMax * i / 4), stroke: i === 0 ? AXIS : GRID }, svg);
      text(svg, padL - 6, Y(yMax * i / 4) + 4, fmt(yMax * i / 4), { "text-anchor": "end" });
      text(svg, X(xMax * i / 4), padT + ph + 16, fmt(xMax * i / 4), { "text-anchor": "middle" });
    }
    if (opts.xLabel) text(svg, padL + pw / 2, c.H - 4, opts.xLabel, { "text-anchor": "middle" });
    if (opts.yLabel) text(svg, 4, 10, opts.yLabel, { "font-size": 10 });
    (opts.refX || []).forEach(function (r) {
      el("line", { x1: X(r.value), x2: X(r.value), y1: padT, y2: padT + ph, stroke: "#13253A", "stroke-dasharray": "4 4" }, svg);
      text(svg, X(r.value) + 4, padT + 12, r.label, { fill: "#13253A" });
    });
    (opts.refY || []).forEach(function (r) {
      el("line", { x1: padL, x2: padL + pw, y1: Y(r.value), y2: Y(r.value), stroke: "#13253A", "stroke-dasharray": "4 4" }, svg);
      text(svg, padL + pw, Y(r.value) - 4, r.label, { "text-anchor": "end", fill: "#13253A" });
    });
    opts.points.forEach(function (p) {
      var dot = el("circle", { cx: X(p.x), cy: Y(p.y), r: 5, fill: p.color, stroke: "#fff", "stroke-width": 2 }, svg);
      var hit = el("circle", { cx: X(p.x), cy: Y(p.y), r: 10, fill: "transparent" }, svg);
      hit.addEventListener("mousemove", function (e) { dot.setAttribute("r", 7); c.show(e, "<b>" + p.label + "</b>" + (p.html || "")); });
      hit.addEventListener("mouseleave", function () { dot.setAttribute("r", 5); c.hide(); });
      if (p.href) { hit.style.cursor = "pointer"; hit.addEventListener("click", function () { window.location = p.href; }); }
    });
  }

  var registry = [];
  function register(fn, container, opts) { registry.push([fn, container, opts]); fn(container, opts); }
  var timer;
  window.addEventListener("resize", function () {
    clearTimeout(timer);
    timer = setTimeout(function () { registry.forEach(function (r) { r[0](r[1], r[2]); }); }, 150);
  });

  window.USPCharts = {
    bar: function (c, o) { register(bar, c, o); },
    line: function (c, o) { register(line, c, o); },
    scatter: function (c, o) { register(scatter, c, o); },
    colors: { s1: "#2c62a8", s2: "#9aa8b6", s3: "#2F7D4F", good: "#2F7D4F", warn: "#E0A43A", crit: "#C43D2F",
              seq: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"] }
  };
})();
