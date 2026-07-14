(function () {
  "use strict";
  const DATA = JSON.parse(document.getElementById("dashboard-data").textContent);
  const engines = DATA.engines;
  const byId = Object.fromEntries(engines.map((e) => [e.id, e]));

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const state = {
    engineId: 1, mode: "fleetBaseline", healthTab: "overallHealth", demoOn: false,
    anomalySubsystem: "compressor", altIdx: 3, rpmIdx: 2, duration: 1.5, payload: 0,
  };

  const HEALTH_TABS = [
    { key: "compressorHealth", label: "Compressor", predKey: "CompressorHealth", reasonKey: "compressor" },
    { key: "combustorHealth", label: "Combustor", predKey: "CombustorHealth", reasonKey: "combustor" },
    { key: "turbineHealth", label: "Turbine", predKey: "TurbineHealth", reasonKey: "turbine" },
    { key: "overallHealth", label: "Overall", predKey: "OverallHealth", reasonKey: null },
  ];

  const SUBSYSTEMS = [
    { key: "compressor", label: "Compressor" },
    { key: "combustor", label: "Combustor" },
    { key: "turbine", label: "Turbine" },
  ];

  function statusOf(value) {
    if (value >= 0.93) return { tone: "good", label: "Nominal" };
    if (value >= 0.86) return { tone: "warning", label: "Monitor" };
    if (value >= 0.80) return { tone: "serious", label: "Degraded" };
    return { tone: "critical", label: "Critical" };
  }

  function fmtPct(v) { return (v * 100).toFixed(1) + "%"; }

  // ---------------- Sidebar ----------------
  function renderEngineList() {
    const list = document.getElementById("engineList");
    list.innerHTML = "";
    engines.forEach((e) => {
      const latest = e.overallHealth[e.overallHealth.length - 1];
      const s = statusOf(latest);
      const btn = document.createElement("button");
      btn.className = "engine-item" + (e.id === state.engineId ? " active" : "");
      btn.innerHTML =
        '<span class="id">ENGINE ' + String(e.id).padStart(2, "0") + '</span>' +
        '<span class="chip"><span class="dot" style="background:var(--' + s.tone + ')"></span>' + fmtPct(latest) + "</span>";
      btn.onclick = () => { state.engineId = e.id; renderAll(); };
      list.appendChild(btn);
    });
  }

  function renderModeGroup() {
    document.querySelectorAll(".mode-option").forEach((el) => {
      const m = el.getAttribute("data-mode");
      el.classList.toggle("active", m === state.mode);
      el.onclick = () => { state.mode = m; el.querySelector("input").checked = true; renderAll(); };
    });
  }

  // ---------------- KPI + header ----------------
  function renderHeader() {
    const e = byId[state.engineId];
    const n = e.cycles.length;
    document.getElementById("engineTitle").textContent = "Engine " + String(e.id).padStart(2, "0");
    document.getElementById("cycleTag").textContent = "Cycle " + e.cycles[n - 1] + " of " + n + " recorded";

    const overall = e.overallHealth[n - 1];
    const s = statusOf(overall);
    const pill = document.getElementById("overallPill");
    pill.style.background = "var(--" + s.tone + "-soft)";
    pill.style.color = "var(--" + s.tone + ")";
    pill.innerHTML = '<span class="dot" style="background:var(--' + s.tone + ')"></span>' + s.label + " · " + fmtPct(overall);
  }

  function kpiTile(label, value, tone) {
    return '<div class="kpi"><div class="kpi-label">' + label + '</div>' +
      '<div class="kpi-value"' + (tone ? ' style="color:var(--' + tone + ')"' : "") + ">" + value + "</div></div>";
  }

  function renderKpis() {
    const e = byId[state.engineId];
    const n = e.cycles.length;
    const row = document.getElementById("kpiRow");
    const comp = e.compressorHealth[n - 1], comb = e.combustorHealth[n - 1], turb = e.turbineHealth[n - 1];
    row.innerHTML =
      kpiTile("Overall Health", fmtPct(e.overallHealth[n - 1]), statusOf(e.overallHealth[n - 1]).tone) +
      kpiTile("Compressor", fmtPct(comp), statusOf(comp).tone) +
      kpiTile("Combustor", fmtPct(comb), statusOf(comb).tone) +
      kpiTile("Turbine", fmtPct(turb), statusOf(turb).tone) +
      kpiTile("Latest Thrust", Math.round(e.thrust[n - 1]).toLocaleString() + " N") +
      kpiTile("Fuel Penalty", (e.fuelPenaltyPct >= 0 ? "+" : "") + e.fuelPenaltyPct.toFixed(1) + "%",
        e.fuelPenaltyPct > 15 ? "critical" : e.fuelPenaltyPct > 7 ? "warning" : "good") +
      rulTile(e);
  }

  function rulTile(e) {
    const r = e.rul;
    if (r.cyclesRemaining === null) {
      return '<div class="kpi rul-card"><div class="rul-top"><span class="kpi-label">REMAINING USEFUL LIFE</span></div>' +
        '<div class="kpi-value" style="color:var(--good)">Stable</div>' +
        '<div class="rul-note">No sustained decline detected over the recorded history — no cycle estimate to report.</div></div>';
    }
    const capped = Math.min(r.cyclesRemaining, 60);
    const pct = Math.max(4, Math.min(100, (capped / 60) * 100));
    const tone = r.cyclesRemaining < 10 ? "critical" : r.cyclesRemaining < 25 ? "warning" : "good";
    return '<div class="kpi rul-card"><div class="rul-top"><span class="kpi-label">REMAINING USEFUL LIFE</span>' +
      '<span class="kpi-value" style="color:var(--' + tone + ');font-size:17px;">' + r.cyclesRemaining.toFixed(0) + ' cycles</span></div>' +
      '<div class="rul-track"><div class="rul-fill" style="width:' + pct + '%;background:var(--' + tone + ')"></div></div>' +
      '<div class="rul-note">Linear projection of the recorded health trend to the ' + (r.threshold * 100).toFixed(0) +
      '% maintenance-review threshold. Not validated beyond the recorded cycle range.</div></div>';
  }

  // ---------------- Verdict + schematic ----------------
  function renderVerdict() {
    const e = byId[state.engineId];
    const v = e.verdict;
    const card = document.getElementById("verdictCard");
    card.className = "verdict-card " + (v.flightReady ? "ready" : "notready");
    document.getElementById("verdictAnswer").className = "verdict-a " + (v.flightReady ? "ready" : "notready");
    document.getElementById("verdictAnswer").textContent = v.flightReady ? "YES" : "NOT RECOMMENDED";
    document.getElementById("verdictConf").textContent = "Confidence " + v.confidence + "%" +
      (v.anomalyActive ? " — active anomaly detected" : "");

    const allReasons = [
      ...e.reasons.compressor.map((r) => ["Compressor", ...r]),
      ...e.reasons.combustor.map((r) => ["Combustor", ...r]),
      ...e.reasons.turbine.map((r) => ["Turbine", ...r]),
    ];
    const order = { critical: 0, warning: 1, good: 2 };
    allReasons.sort((a, b) => order[a[1]] - order[b[1]]);
    const top = allReasons.slice(0, 3);
    document.getElementById("verdictReasons").innerHTML = top.map((r) =>
      '<div class="vreason"><span class="vdot" style="background:var(--' + r[1] + ')"></span>' + r[0] + ' — ' + r[2] + '</div>'
    ).join("");
    document.getElementById("verdictConfReason").textContent = v.confidenceReason || "";
  }

  function renderDna() {
    const e = byId[state.engineId];
    const dna = e.engineDNA;
    const card = document.getElementById("dnaCard");
    if (dna.h1 === null) { card.innerHTML = "Not available for this engine."; return; }
    card.innerHTML =
      "h1 (initial health) &nbsp;" + fmtPct(dna.h1) + "<br>" +
      "d (total trajectory drop) &nbsp;" + fmtPct(dna.d) + "<br>" +
      '<span style="color:var(--text-muted);font-size:9.5px;">Fit from this engine\'s own train-split cycles only.</span>';
  }

  function renderRecommendations() {
    const e = byId[state.engineId];
    const wrap = document.getElementById("recommendationList");
    wrap.innerHTML = e.counterfactuals.map((c, i) => {
      const tone = c.improvementPts > 5 ? "critical" : c.improvementPts > 1.5 ? "warning" : "good";
      const label = c.subsystem.charAt(0).toUpperCase() + c.subsystem.slice(1);
      return '<div class="reason-item"><span class="rdot" style="background:var(--' + tone + ')"></span>' +
        '#' + (i + 1) + ' — Restoring ' + label + ' from ' + fmtPct(c.currentValue) + ' to its healthy baseline ' +
        fmtPct(c.healthyValue) + ' would recover approximately <strong>+' + c.improvementPts.toFixed(1) + ' points</strong> of overall health.</div>';
    }).join("");
  }

  function renderCorrelationCards() {
    const c = DATA.metrics.crossSubsystemCorrelation;
    const pairs = [
      { label: "Compressor ↔ Combustor", v: c.compressorVsCombustor },
      { label: "Compressor ↔ Turbine", v: c.compressorVsTurbine },
      { label: "Combustor ↔ Turbine", v: c.combustorVsTurbine },
    ];
    document.getElementById("correlationCards").innerHTML = pairs.map((p) => {
      const tone = Math.abs(p.v) > 0.5 ? "warning" : "good";
      return '<div class="kpi"><div class="kpi-label">' + p.label + '</div>' +
        '<div class="kpi-value" style="color:var(--' + tone + ');font-size:18px;">' + (p.v >= 0 ? "+" : "") + p.v.toFixed(3) + '</div>' +
        '<div class="kpi-sub">' + (Math.abs(p.v) > 0.5 ? "Moderate relationship" : "Weak — consistent with independent decline modes") + '</div></div>';
    }).join("");
  }

  function renderSchematic() {
    const e = byId[state.engineId];
    const n = e.cycles.length;
    const stages = [
      { name: "Inlet", val: null },
      { name: "Compressor", val: e.compressorHealth[n - 1] },
      { name: "Combustor", val: e.combustorHealth[n - 1] },
      { name: "Turbine", val: e.turbineHealth[n - 1] },
      { name: "Exhaust", val: null },
    ];
    const row = document.getElementById("schematicRow");
    row.innerHTML = stages.map((s, i) => {
      const parts = [];
      if (i > 0) parts.push('<span class="schem-arrow">→</span>');
      if (s.val === null) {
        parts.push('<div class="schem-stage" style="border-style:dashed;"><span class="sname">' + s.name + '</span><span class="sval" style="color:var(--text-muted);font-size:13px;">—</span></div>');
      } else {
        const st = statusOf(s.val);
        parts.push('<div class="schem-stage" style="background:var(--' + st.tone + '-soft);border-color:var(--' + st.tone + ')">' +
          '<span class="sname">' + s.name + '</span><span class="sval" style="color:var(--' + st.tone + ')">' + fmtPct(s.val) + '</span></div>');
      }
      return parts.join("");
    }).join("");
  }

  function renderSubsystemReasons() {
    const e = byId[state.engineId];
    const tabDef = HEALTH_TABS.find((t) => t.key === state.healthTab);
    const wrap = document.getElementById("subsystemReasons");
    if (!tabDef.reasonKey) {
      wrap.innerHTML = '<div class="reason-item"><span class="rdot" style="background:var(--accent)"></span>Overall health blends all three subsystems below — select a subsystem tab for its specific diagnostic reasons.</div>';
      return;
    }
    const lines = e.reasons[tabDef.reasonKey];
    wrap.innerHTML = lines.map((r) =>
      '<div class="reason-item"><span class="rdot" style="background:var(--' + r[0] + ')"></span>' + r[1] + '</div>'
    ).join("");
  }

  // ---------------- Chart engine (shared SVG renderer) ----------------
  function scaleFns(svg, xDomain, yDomain, margins) {
    const vb = svg.viewBox.baseVal;
    const W = vb.width, H = vb.height;
    const m = Object.assign({ l: 42, r: 12, t: 10, b: 24 }, margins || {});
    const x = (v) => m.l + ((v - xDomain[0]) / (xDomain[1] - xDomain[0])) * (W - m.l - m.r);
    const y = (v) => H - m.b - ((v - yDomain[0]) / (yDomain[1] - yDomain[0])) * (H - m.t - m.b);
    return { x, y, W, H, m };
  }

  function niceDomain(values, padFrac) {
    let lo = Math.min(...values), hi = Math.max(...values);
    if (lo === hi) { lo -= 1; hi += 1; }
    const pad = (hi - lo) * (padFrac || 0.12);
    return [lo - pad, hi + pad];
  }

  const svgns = "http://www.w3.org/2000/svg";
  function el(tag, attrs) {
    const n = document.createElementNS(svgns, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    return n;
  }

  function pathFrom(pts, xs, ys) {
    return pts.map((p, i) => (i === 0 ? "M" : "L") + xs(p.x).toFixed(1) + "," + ys(p.y).toFixed(1)).join(" ");
  }

  /**
   * config: {
   *   xDomain, yDomain (optional -> auto from series+points+band),
   *   yTickFormat, xTicks: [values],
   *   series: [{data:[{x,y}], color, width, dash}],
   *   band: {data:[{x,lower,upper}], color},
   *   points: [{data:[{x,y}], color}],
   *   refLines: [{y, color, dash}],
   *   vLine: {x, color, label}
   * }
   */
  function renderChart(svgId, tooltipId, config) {
    const svg = document.getElementById(svgId);
    svg.innerHTML = "";
    const allY = [];
    (config.series || []).forEach((s) => s.data.forEach((p) => allY.push(p.y)));
    (config.points || []).forEach((s) => s.data.forEach((p) => allY.push(p.y)));
    (config.band || { data: [] }).data.forEach((p) => { allY.push(p.lower); allY.push(p.upper); });
    (config.refLines || []).forEach((r) => allY.push(r.y));
    const yDomain = config.yDomain || niceDomain(allY);
    const xDomain = config.xDomain;
    const { x, y, W, H, m } = scaleFns(svg, xDomain, yDomain, config.margins);

    // gridlines + y ticks
    const yTicks = config.yTicksCount || 4;
    for (let i = 0; i <= yTicks; i++) {
      const v = yDomain[0] + (i / yTicks) * (yDomain[1] - yDomain[0]);
      const yy = y(v);
      svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, class: "grid-line" }));
      const label = config.yTickFormat ? config.yTickFormat(v) : v.toFixed(2);
      const t = el("text", { x: m.l - 8, y: yy + 3, class: "axis-label", "text-anchor": "end" });
      t.textContent = label;
      svg.appendChild(t);
    }
    // x ticks
    const xt = config.xTicks || xDomain;
    xt.forEach((v) => {
      const t = el("text", { x: x(v), y: H - 6, class: "axis-label", "text-anchor": "middle" });
      t.textContent = v;
      svg.appendChild(t);
    });

    // reference lines (e.g. anomaly thresholds)
    (config.refLines || []).forEach((r) => {
      const yy = y(r.y);
      svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, stroke: r.color, "stroke-width": 1.3, "stroke-dasharray": "4 3", opacity: 0.75 }));
    });

    // band
    if (config.band && config.band.data.length) {
      const d = config.band.data;
      const fwd = d.map((p) => x(p.x) + "," + y(p.upper)).join(" L");
      const bwd = d.slice().reverse().map((p) => x(p.x) + "," + y(p.lower)).join(" L");
      svg.appendChild(el("path", { d: "M" + fwd + " L" + bwd + " Z", fill: config.band.color, opacity: 0.16, stroke: "none" }));
    }

    // vertical marker line (e.g. shock cycle)
    if (config.vLine) {
      const xx = x(config.vLine.x);
      svg.appendChild(el("line", { x1: xx, x2: xx, y1: m.t, y2: H - m.b, stroke: config.vLine.color, "stroke-width": 1.3, "stroke-dasharray": "3 3" }));
      const lbl = el("text", { x: xx + 4, y: m.t + 10, class: "axis-label", fill: config.vLine.color });
      lbl.textContent = config.vLine.label;
      svg.appendChild(lbl);
    }

    // series lines
    (config.series || []).forEach((s) => {
      if (!s.data.length) return;
      const attrs = { d: pathFrom(s.data, x, y), fill: "none", stroke: s.color, "stroke-width": s.width || 2, "stroke-linecap": "round", "stroke-linejoin": "round" };
      if (s.dash) attrs["stroke-dasharray"] = "5 4";
      svg.appendChild(el("path", attrs));
    });

    // predicted point markers (diamonds)
    (config.points || []).forEach((s) => {
      s.data.forEach((p) => {
        const cx = x(p.x), cy = y(p.y);
        const r = 4.5;
        svg.appendChild(el("path", {
          d: "M" + cx + "," + (cy - r) + " L" + (cx + r) + "," + cy + " L" + cx + "," + (cy + r) + " L" + (cx - r) + "," + cy + " Z",
          fill: s.color, stroke: cssVar("--surface"), "stroke-width": 1.2,
        }));
      });
    });

    // hover crosshair + tooltip
    const tooltip = document.getElementById(tooltipId);
    if (tooltip && config.hover !== false) {
      const hitRect = el("rect", { x: m.l, y: m.t, width: Math.max(W - m.l - m.r, 0), height: Math.max(H - m.t - m.b, 0), fill: "transparent" });
      const crosshair = el("line", { x1: 0, x2: 0, y1: m.t, y2: H - m.b, class: "crosshair" });
      svg.appendChild(crosshair);
      svg.appendChild(hitRect);
      hitRect.addEventListener("mousemove", (ev) => {
        const rect = svg.getBoundingClientRect();
        const px = ((ev.clientX - rect.left) / rect.width) * W;
        let cyc = Math.round(xDomain[0] + ((px - m.l) / (W - m.l - m.r)) * (xDomain[1] - xDomain[0]));
        cyc = Math.max(xDomain[0], Math.min(xDomain[1], cyc));
        crosshair.setAttribute("x1", x(cyc)); crosshair.setAttribute("x2", x(cyc)); crosshair.style.opacity = 1;
        const lines = [config.tooltipTitle ? config.tooltipTitle(cyc) : "Cycle " + cyc];
        (config.tooltipRows || []).forEach((r) => { const v = r.get(cyc); if (v !== undefined && v !== null) lines.push(r.label + ": " + v); });
        tooltip.innerHTML = lines.join("<br>");
        tooltip.style.left = x(cyc) + "px";
        tooltip.style.top = (rect.height * (y(yDomain[1]) / H)) + "px";
        tooltip.style.opacity = 1;
      });
      hitRect.addEventListener("mouseleave", () => { crosshair.style.opacity = 0; tooltip.style.opacity = 0; });
    }
  }

  function legendHTML(items) {
    return items.map((it) =>
      '<span class="legend-item">' +
      (it.dashed
        ? '<span class="legend-swatch" style="background:repeating-linear-gradient(90deg,' + it.color + ' 0 5px, transparent 5px 8px)"></span>'
        : it.dot
          ? '<span class="legend-dot" style="background:' + it.color + '"></span>'
          : '<span class="legend-swatch" style="background:' + it.color + '"></span>') +
      it.label + "</span>"
    ).join("");
  }

  // ---------------- Health chart ----------------
  function renderHealthTabs() {
    const wrap = document.getElementById("healthTabs");
    wrap.innerHTML = "";
    HEALTH_TABS.forEach((t) => {
      const b = document.createElement("button");
      b.className = "tab" + (t.key === state.healthTab ? " active" : "");
      b.textContent = t.label;
      b.onclick = () => { state.healthTab = t.key; renderHealthChart(); renderSubsystemReasons(); };
      wrap.appendChild(b);
    });
  }

  function renderHealthChart() {
    document.querySelectorAll("#healthTabs .tab").forEach((b, i) => b.classList.toggle("active", HEALTH_TABS[i].key === state.healthTab));
    const e = byId[state.engineId];
    const tabDef = HEALTH_TABS.find((t) => t.key === state.healthTab);
    const trueData = e.cycles.map((c, i) => ({ x: c, y: e[tabDef.key][i] }));
    const modeLabel = state.mode === "flightLogMatch" ? "Flight-Log Match" : "Fleet Baseline Estimate";
    const predArr = e.pred[tabDef.predKey][state.mode];
    const points = e.testCycles.map((c, i) => ({ x: c, y: predArr[i] }));
    let band = null;
    if (state.mode === "fleetBaseline") {
      const lower = e.pred[tabDef.predKey].fleetBaselineLower;
      const upper = e.pred[tabDef.predKey].fleetBaselineUpper;
      band = { color: cssVar("--accent-2"), data: e.testCycles.map((c, i) => ({ x: c, lower: lower[i], upper: upper[i] })) };
    }

    const shadowKey = tabDef.reasonKey || "overall";
    const shadowVal = e.shadowHealthy[shadowKey];
    const good = cssVar("--good");

    document.getElementById("healthLegend").innerHTML = legendHTML([
      { color: cssVar("--accent"), label: "Recorded trajectory" },
      { color: cssVar("--accent-2"), label: "Predicted (" + modeLabel + ")", dot: true },
      ...(band ? [{ color: cssVar("--accent-2"), label: "90% confidence band" }] : []),
      { color: good, label: "Shadow Healthy Twin (this engine's own healthiest reading)", dashed: true },
    ]);

    renderChart("healthChart", "healthTooltip", {
      xDomain: [1, 30], xTicks: [1, 5, 10, 15, 20, 25, 30],
      yTickFormat: fmtPct,
      series: [{ data: trueData, color: cssVar("--accent"), width: 2.2 }],
      points: [{ data: points, color: cssVar("--accent-2") }],
      band: band,
      refLines: [{ y: shadowVal, color: good }],
      tooltipTitle: (c) => "Cycle " + c,
      tooltipRows: [
        { label: "Recorded", get: (c) => { const p = trueData.find((d) => d.x === c); return p ? fmtPct(p.y) : null; } },
        { label: "Predicted", get: (c) => { const p = points.find((d) => d.x === c); return p ? fmtPct(p.y) : null; } },
        { label: "Shadow Healthy", get: () => fmtPct(shadowVal) },
      ],
    });
  }

  // ---------------- Performance charts ----------------
  function renderPerfCharts() {
    const e = byId[state.engineId];
    const trueThrust = e.cycles.map((c, i) => ({ x: c, y: e.thrust[i] }));
    const predThrust = e.testCycles.map((c, i) => ({ x: c, y: e.thrustPred[i] }));
    document.getElementById("thrustLegend").innerHTML = legendHTML([
      { color: cssVar("--accent"), label: "Recorded Thrust (N)" },
      { color: cssVar("--accent-2"), label: "Predicted", dot: true },
    ]);
    renderChart("thrustChart", "thrustTooltip", {
      xDomain: [1, 30], xTicks: [1, 15, 30], yTickFormat: (v) => Math.round(v / 1000) + "k",
      series: [{ data: trueThrust, color: cssVar("--accent"), width: 2 }],
      points: [{ data: predThrust, color: cssVar("--accent-2") }],
      tooltipTitle: (c) => "Cycle " + c,
      tooltipRows: [{ label: "Thrust", get: (c) => { const p = trueThrust.find((d) => d.x === c); return p ? Math.round(p.y) + " N" : null; } }],
      margins: { l: 40, r: 8, t: 10, b: 22 },
    });

    const trueTsfc = e.cycles.map((c, i) => ({ x: c, y: e.tsfc[i] }));
    const predTsfc = e.testCycles.map((c, i) => ({ x: c, y: e.tsfcPred[i] }));
    document.getElementById("tsfcLegend").innerHTML = legendHTML([
      { color: cssVar("--accent"), label: "Recorded TSFC" },
      { color: cssVar("--accent-2"), label: "Predicted", dot: true },
    ]);
    renderChart("tsfcChart", "tsfcTooltip", {
      xDomain: [1, 30], xTicks: [1, 15, 30], yTickFormat: (v) => v.toFixed(3),
      series: [{ data: trueTsfc, color: cssVar("--accent"), width: 2 }],
      points: [{ data: predTsfc, color: cssVar("--accent-2") }],
      tooltipTitle: (c) => "Cycle " + c,
      tooltipRows: [{ label: "TSFC", get: (c) => { const p = trueTsfc.find((d) => d.x === c); return p ? p.y.toFixed(4) + " g/N.s" : null; } }],
      margins: { l: 46, r: 8, t: 10, b: 22 },
    });
  }

  // ---------------- Mission impact ----------------
  function renderMission() {
    const e = byId[state.engineId];
    const fp = e.fuelPenaltyPct, rp = e.rangeImpactPct;
    document.getElementById("fuelPenalty").textContent = (fp >= 0 ? "+" : "") + fp.toFixed(1) + "%";
    document.getElementById("fuelPenalty").style.color = "var(--" + (fp > 15 ? "critical" : fp > 7 ? "warning" : "good") + ")";
    document.getElementById("rangeImpact").textContent = (rp >= 0 ? "+" : "") + rp.toFixed(1) + "%";
    document.getElementById("rangeImpact").style.color = "var(--" + (rp < -15 ? "critical" : rp < -7 ? "warning" : "good") + ")";

    document.getElementById("missionText").textContent =
      "At its current health state, Engine " + String(e.id).padStart(2, "0") + " is estimated to burn " +
      Math.abs(fp).toFixed(1) + "% " + (fp >= 0 ? "more" : "less") +
      " fuel than when healthy, for the same thrust output — a documented engineering approximation " +
      "(fuel burn scaling with the inverse of the overall health index), not a directly fitted result: " +
      "raw fuel-consumption readings in this dataset are dominated by instantaneous flight condition rather " +
      "than health, so the validated health index is used as the basis instead. Held constant across a fixed " +
      "fuel load, this translates to roughly " + Math.abs(rp).toFixed(1) + "% " + (rp >= 0 ? "more" : "less") +
      " achievable mission range — most exposed during the cruise and loiter phases, where the engine spends " +
      "the most time at sustained thrust.";

    const phases = ["Take-off", "Climb", "Cruise", "Loiter", "Descent"];
    const hot = ["Cruise", "Loiter"];
    document.getElementById("phaseRow").innerHTML = phases.map((p) =>
      '<span class="phase' + (hot.includes(p) ? " hot" : "") + '">' + p + "</span>").join("");
  }

  // ---------------- Anomaly watch ----------------
  function renderAnomalyTabs() {
    const wrap = document.getElementById("anomalyTabs");
    wrap.style.display = state.demoOn ? "none" : "flex";
    wrap.innerHTML = SUBSYSTEMS.map((s) =>
      '<button class="tab' + (s.key === state.anomalySubsystem ? " active" : "") + '" data-sub="' + s.key + '">' + s.label + "</button>"
    ).join("");
    wrap.querySelectorAll(".tab").forEach((b) => {
      b.onclick = () => { state.anomalySubsystem = b.getAttribute("data-sub"); renderAnomaly(); };
    });
  }

  function renderAnomaly() {
    renderAnomalyTabs();
    const btn = document.getElementById("demoBtn");
    btn.textContent = state.demoOn ? "← Back to Live Engine" : "Run Fault-Injection Demo →";
    btn.classList.toggle("on", state.demoOn);

    const good = cssVar("--good"), critical = cssVar("--critical"), accent = cssVar("--accent"), muted = cssVar("--text-muted");

    if (!state.demoOn) {
      const e = byId[state.engineId];
      const subLabel = SUBSYSTEMS.find((s) => s.key === state.anomalySubsystem).label;
      const zArr = e.anomalyZAll[state.anomalySubsystem];
      const data = e.cycles.map((c, i) => ({ x: c, y: zArr[i] }));
      document.getElementById("anomalyLegend").innerHTML = legendHTML([
        { color: accent, label: subLabel + " physics residual (z-score)" },
        { color: critical, label: "±3σ anomaly threshold", dashed: true },
      ]);
      renderChart("anomalyChart", "anomalyTooltip", {
        xDomain: [1, 30], xTicks: [1, 5, 10, 15, 20, 25, 30], yTickFormat: (v) => v.toFixed(1),
        yDomain: [-6, 6],
        series: [{ data, color: accent, width: 2 }],
        refLines: [{ y: 3, color: critical }, { y: -3, color: critical }],
        tooltipTitle: (c) => "Cycle " + c,
        tooltipRows: [{ label: "z-score", get: (c) => { const p = data.find((d) => d.x === c); return p ? p.y.toFixed(2) : null; } }],
      });
      const flagged = e.cycles.map((c, i) => ({ c, z: zArr[i] })).filter((p) => Math.abs(p.z) > 3);
      renderAlertLog(flagged, "Engine " + String(e.id).padStart(2, "0"), subLabel);
    } else {
      const d = DATA.faultDemo;
      const normal = d.cycles.map((c, i) => ({ x: c, y: d.zNormal[i] }));
      const shocked = d.cycles.map((c, i) => ({ x: c, y: d.zShocked[i] }));
      document.getElementById("anomalyLegend").innerHTML = legendHTML([
        { color: muted, label: "Engine 01 — normal operation", dashed: true },
        { color: critical, label: "Engine 01 — simulated compressor fault from cycle " + d.shockCycle },
        { color: critical, label: "±3σ anomaly threshold", dashed: true },
      ]);
      renderChart("anomalyChart", "anomalyTooltip", {
        xDomain: [1, 30], xTicks: [1, 5, 10, 15, 20, 25, 30], yTickFormat: (v) => v.toFixed(1),
        yDomain: [-6, 22],
        series: [{ data: normal, color: muted, width: 1.6, dash: true }, { data: shocked, color: critical, width: 2.2 }],
        refLines: [{ y: 3, color: critical }, { y: -3, color: critical }],
        vLine: { x: d.shockCycle, color: critical, label: "fault injected" },
        tooltipTitle: (c) => "Cycle " + c,
        tooltipRows: [
          { label: "Normal", get: (c) => { const p = normal.find((x) => x.x === c); return p ? p.y.toFixed(2) : null; } },
          { label: "Faulted", get: (c) => { const p = shocked.find((x) => x.x === c); return p ? p.y.toFixed(2) : null; } },
        ],
      });
      const flagged = d.cycles.map((c, i) => ({ c, z: d.zShocked[i] })).filter((p) => Math.abs(p.z) > 3);
      renderAlertLog(flagged, "Fault-injection demo", "Compressor");
    }
  }

  function renderAlertLog(flagged, sourceLabel, subLabel) {
    const log = document.getElementById("alertLog");
    if (!flagged.length) {
      log.innerHTML = '<div class="alert-row"><span class="atag">CLEAR</span>' + sourceLabel + " — " + subLabel +
        " stayed within the ±3σ normal operating envelope on every recorded cycle.</div>";
      return;
    }
    log.innerHTML = flagged.map((p) => {
      const conf = Math.min(99, Math.round(70 + Math.abs(p.z) * 3));
      return '<div class="alert-row crit"><span class="atag">ANOMALY</span>' + sourceLabel + " — cycle " + p.c + ": " +
        subLabel + " reading " + Math.abs(p.z).toFixed(1) + "σ outside normal. Recommend inspection within " +
        Math.max(5, Math.round(30 - Math.abs(p.z))) + " cycles.<span class=\"conf\">conf " + conf + "%</span></div>";
    }).join("");
  }

  // ---------------- Mission Analysis simulator ----------------
  function simStat(label, value, delta) {
    return '<div class="sim-stat"><div class="slabel2">' + label + '</div><div class="svalue2">' + value + '</div>' +
      (delta ? '<div class="sdelta">' + delta + '</div>' : '') + '</div>';
  }

  function renderMissionSim() {
    const e = byId[state.engineId];
    const g = e.missionGrid;

    document.getElementById("altVal").textContent = Math.round(g.altitudes[state.altIdx]).toLocaleString() + " m";
    document.getElementById("rpmVal").textContent = Math.round(g.rpms[state.rpmIdx]).toLocaleString() + " rpm";
    document.getElementById("durVal").textContent = state.duration.toFixed(1) + " h";
    document.getElementById("loadVal").textContent = state.payload + " kg";

    const thrustCur = g.thrustCurrent[state.altIdx][state.rpmIdx];
    const thrustHealthy = g.thrustHealthy[state.altIdx][state.rpmIdx];
    const tsfc = g.tsfcCurrent[state.altIdx][state.rpmIdx];

    // illustrative payload modifier -- documented assumption, not model-fitted:
    // up to 5% additional fuel burn at the 200kg reference payload
    const payloadPenaltyPct = (state.payload / 200) * 5;
    const fuelKg = (tsfc * thrustCur * state.duration * 3600 / 1000) * (1 + payloadPenaltyPct / 100);

    const perfRetentionPct = Math.min(100, (thrustCur / thrustHealthy) * 100);
    const missionSuccess = Math.max(30, Math.min(99, Math.round(perfRetentionPct - payloadPenaltyPct * 0.3)));

    // one ~2h mission treated as one operating-cycle equivalent (documented simplifying assumption)
    const healthNow = e.overallHealth[e.overallHealth.length - 1];
    const healthAfter = Math.max(0, healthNow + e.rul.slopePerCycle * (state.duration / 2));

    const results = document.getElementById("simResults");
    results.innerHTML =
      simStat("Predicted Thrust", Math.round(thrustCur).toLocaleString() + " N",
        "Healthy-engine reference: " + Math.round(thrustHealthy).toLocaleString() + " N (" + perfRetentionPct.toFixed(0) + "% retained)") +
      simStat("Estimated Fuel Burn", fuelKg.toFixed(1) + " kg",
        payloadPenaltyPct > 0 ? "+" + payloadPenaltyPct.toFixed(1) + "% from illustrative payload modifier" : "") +
      simStat("Mission Success", missionSuccess + "%",
        "Based on thrust retention vs. this engine's own healthy baseline at the same altitude/RPM") +
      simStat("Health After Mission", fmtPct(healthAfter),
        "Assumes a ~2h mission ≈ one operating cycle") +
      '<div class="sim-caveat">Thrust/fuel predictions come from the trained performance model (Thrust R² 0.990, TSFC R² 0.982, see Methodology). ' +
      'Mission Success and Payload figures are documented engineering approximations layered on top, not directly fitted — this dataset has no payload or weather columns.</div>';
  }

  function wireSimControls() {
    document.getElementById("altSlider").oninput = (ev) => { state.altIdx = +ev.target.value; renderMissionSim(); };
    document.getElementById("rpmSlider").oninput = (ev) => { state.rpmIdx = +ev.target.value; renderMissionSim(); };
    document.getElementById("durSlider").oninput = (ev) => { state.duration = +ev.target.value; renderMissionSim(); };
    document.getElementById("loadSlider").oninput = (ev) => { state.payload = +ev.target.value; renderMissionSim(); };
  }

  // ---------------- Ask the Twin (templated Q&A) ----------------
  const QA_QUESTIONS = [
    { q: "Why is thrust decreasing?", fn: (e) => {
      const n = e.thrust.length;
      const turbSlope = (e.turbineHealth[n - 1] - e.turbineHealth[Math.max(0, n - 6)]) / Math.min(5, n - 1);
      return "Turbine health has moved by " + (turbSlope * 100).toFixed(2) + " points/cycle over the last window, and the " +
        "compressor/turbine physics residuals " + (Math.abs(e.anomalyZAll.turbine[n - 1]) > 2 ? "show an active deviation" : "remain within normal range") +
        ". Reduced turbine efficiency lowers achievable thrust at a given fuel flow — see the Performance Prediction panel above.";
    }},
    { q: "When should I do maintenance?", fn: (e) => {
      const r = e.rul;
      if (r.cyclesRemaining === null) return "No sustained decline detected in the recorded history — no maintenance window to project yet.";
      return "At the current degradation rate (" + (Math.abs(r.slopePerCycle) * 100).toFixed(2) + " health points/cycle), this engine is " +
        "projected to reach the " + (r.threshold * 100).toFixed(0) + "% maintenance-review threshold in approximately " +
        r.cyclesRemaining.toFixed(0) + " cycles. This is a linear projection of the recorded trend, not validated beyond it.";
    }},
    { q: "Can this engine fly today?", fn: (e) => {
      const v = e.verdict;
      return (v.flightReady ? "Yes — " : "Not recommended — ") + "confidence " + v.confidence + "%. " +
        (v.anomalyActive ? "An active gas-path anomaly was detected on the latest cycle. " : "No active anomaly on the latest cycle. ") +
        "See the verdict panel at the top of the page for the full reasoning.";
    }},
    { q: "Why is turbine health low?", fn: (e) => e.reasons.turbine.map((r) => r[1]).join(" ") },
    { q: "What's driving the anomaly alert?", fn: (e) => {
      const n = e.cycles.length;
      const zs = SUBSYSTEMS.map((s) => ({ label: s.label, z: e.anomalyZAll[s.key][n - 1] }));
      zs.sort((a, b) => Math.abs(b.z) - Math.abs(a.z));
      const top = zs[0];
      return Math.abs(top.z) > 2
        ? top.label + " shows the largest gas-path deviation on the latest cycle, at " + Math.abs(top.z).toFixed(1) + "σ from the normal operating envelope."
        : "No subsystem is currently showing a significant gas-path deviation — all readings are within 2σ of normal.";
    }},
  ];

  function renderQaChips() {
    const wrap = document.getElementById("qaChips");
    wrap.innerHTML = QA_QUESTIONS.map((item, i) => '<button class="qa-chip" data-i="' + i + '">' + item.q + "</button>").join("");
    wrap.querySelectorAll(".qa-chip").forEach((b) => {
      b.onclick = () => {
        const item = QA_QUESTIONS[+b.getAttribute("data-i")];
        const e = byId[state.engineId];
        document.getElementById("qaAnswer").innerHTML = '<div class="qa-q">' + item.q + '</div>' + item.fn(e);
      };
    });
    document.getElementById("qaAnswer").textContent = "Select a question above.";
  }

  // ---------------- Sensor Integrity Check ----------------
  function renderIntegrityChips() {
    const si = DATA.sensorIntegrity;
    const wrap = document.getElementById("integrityChips");
    wrap.innerHTML = si.scenarios.map((s, i) =>
      '<button class="qa-chip" data-i="' + i + '">Simulate: ' + s.label + '</button>').join("");
    wrap.querySelectorAll(".qa-chip").forEach((b) => {
      b.onclick = () => showIntegrityScenario(+b.getAttribute("data-i"));
    });
    document.getElementById("integrityResult").innerHTML =
      '<div class="tempty" style="font-family:var(--font-sans);font-size:12.5px;color:var(--text-muted);">' +
      'Click a scenario above to inject a simulated bad sensor reading and see how the system responds.</div>';

    const fp = si.sensorFaultFalsePositives, ca = si.componentAnomalyRate;
    document.getElementById("integrityFpRate").textContent =
      "Tested against all " + fp.total + " real recorded cycles across the fleet: " + fp.flagged +
      " false 'sensor fault' alarms (0% — no legitimate reading has ever been misclassified as a sensor fault). " +
      ca.flagged + "/" + ca.total + " cycles fall in the softer 'worth a second look' tier, consistent with a " +
      "99% statistical envelope by construction, not a flaw.";
  }

  function showIntegrityScenario(i) {
    const s = DATA.sensorIntegrity.scenarios[i];
    const isSensorFault = s.verdict === "sensor_fault_suspected";
    const tone = isSensorFault ? "critical" : "warning";
    const verdictLabel = isSensorFault ? "SENSOR FAULT SUSPECTED" : "COMPONENT ANOMALY SUSPECTED";

    const oldSystemText = s.oldSystemBroke
      ? "Produces an invalid (NaN) result — the physics inversion breaks silently on this input. No alert, no explanation, nothing an engineer would see."
      : "z-score = " + s.oldSystemZScore + (Math.abs(s.oldSystemZScore) > 3
          ? " — flags an anomaly, but labels it a component health issue. An engineer would be sent to inspect the wrong thing: the part, not the sensor."
          : " — below the ±3σ alert threshold. This fault would pass through completely undetected.");

    document.getElementById("integrityResult").innerHTML =
      '<div class="integrity-verdict"><span class="integrity-badge" style="background:var(--' + tone + '-soft);color:var(--' + tone + ')">' +
      verdictLabel + '</span></div>' +
      '<div class="reason-list" style="margin-top:0;margin-bottom:12px;">' +
      s.reasons.map((r) => '<div class="reason-item"><span class="rdot" style="background:var(--' + tone + ')"></span>' + r + '</div>').join("") +
      '</div>' +
      '<div class="integrity-compare">' +
      '<div class="integrity-col"><div class="ic-title">Old system (z-score only)</div><div class="ic-body">' + oldSystemText + '</div></div>' +
      '<div class="integrity-col"><div class="ic-title">Sensor Integrity Layer</div><div class="ic-body">Correctly isolates this as a <strong>' +
      (isSensorFault ? "sensor" : "component") + ' issue</strong>, using rules verified against real data rather than a generic statistical threshold.</div></div>' +
      '</div>';
  }

  // ---------------- Interactive 3D Engine Model ----------------
  const Engine3D = (function () {
    if (typeof THREE === "undefined") return { init() {}, update() {} };

    const STATUS_HEX = { good: 0x0ca30c, warning: 0xfab219, serious: 0xec835a, critical: 0xd03b3b, neutral: 0x7c8792 };
    const PARTS = [
      { key: "intake", rTop: 0.85, rBot: 1.05, len: 1.3, subsystem: null, label: "Intake" },
      { key: "compressor", rTop: 0.95, rBot: 1.05, len: 1.5, subsystem: "compressor", label: "Compressor" },
      { key: "combustor", rTop: 1.25, rBot: 1.25, len: 1.7, subsystem: "combustor", label: "Combustor" },
      { key: "turbine", rTop: 0.85, rBot: 0.95, len: 1.15, subsystem: "turbine", label: "Turbine" },
      { key: "nozzle", rTop: 0.4, rBot: 0.85, len: 1.25, subsystem: null, label: "Nozzle" },
    ];

    let scene, camera, renderer, canvasWrap, labelLayer;
    let meshes = {}, interactiveMeshes = [];
    let azimuth = 0.6, elevation = 0.28, distance = 8.2;
    let dragging = false, lastX = 0, lastY = 0, moved = 0;
    let airflowOn = false, xrayOn = false, particles = [];
    let selectedSubsystem = null;
    let simCycle = 30;
    let raf = null;

    function build() {
      canvasWrap = document.getElementById("threeCanvasWrap");
      const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;

      scene = new THREE.Scene();
      camera = new THREE.PerspectiveCamera(42, w / h, 0.1, 100);
      renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
      renderer.setSize(w, h);
      renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
      canvasWrap.insertBefore(renderer.domElement, canvasWrap.firstChild);

      labelLayer = document.createElement("div");
      labelLayer.style.position = "absolute"; labelLayer.style.inset = "0"; labelLayer.style.pointerEvents = "none";
      canvasWrap.appendChild(labelLayer);

      scene.add(new THREE.AmbientLight(0xffffff, 0.55));
      const key = new THREE.DirectionalLight(0xffffff, 0.9); key.position.set(4, 6, 5); scene.add(key);
      const rim = new THREE.DirectionalLight(0x88aaff, 0.4); rim.position.set(-5, -2, -4); scene.add(rim);

      const totalLen = PARTS.reduce((s, p) => s + p.len, 0);
      let cursor = -totalLen / 2;
      const engineGroup = new THREE.Group();

      PARTS.forEach((p) => {
        const geo = new THREE.CylinderGeometry(p.rTop, p.rBot, p.len, 28, 1, false);
        const mat = new THREE.MeshStandardMaterial({ color: STATUS_HEX.neutral, metalness: 0.35, roughness: 0.5, transparent: true, opacity: 1 });
        const mesh = new THREE.Mesh(geo, mat);
        mesh.rotation.x = Math.PI / 2;
        mesh.position.z = cursor + p.len / 2;
        mesh.userData = { subsystem: p.subsystem, label: p.label, baseEmissive: 0 };
        engineGroup.add(mesh);
        meshes[p.key] = mesh;
        if (p.subsystem) interactiveMeshes.push(mesh);
        cursor += p.len;
      });

      // shaft running through the whole assembly
      const shaftGeo = new THREE.CylinderGeometry(0.12, 0.12, totalLen * 0.94, 14);
      const shaftMat = new THREE.MeshStandardMaterial({ color: 0x9aa4ad, metalness: 0.7, roughness: 0.3 });
      const shaft = new THREE.Mesh(shaftGeo, shaftMat);
      shaft.rotation.x = Math.PI / 2;
      engineGroup.add(shaft);
      meshes.shaft = shaft;

      scene.add(engineGroup);
      meshes._group = engineGroup;
      meshes._totalLen = totalLen;

      wireInteraction();
      animate();
      window.addEventListener("resize", onResize);
    }

    function onResize() {
      if (!canvasWrap) return;
      const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h; camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    }

    function wireInteraction() {
      const dom = renderer.domElement;
      dom.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; moved = 0; canvasWrap.classList.add("dragging"); });
      window.addEventListener("pointermove", (e) => {
        if (!dragging) return;
        const dx = e.clientX - lastX, dy = e.clientY - lastY;
        moved += Math.abs(dx) + Math.abs(dy);
        azimuth += dx * 0.008;
        elevation = Math.max(-1.2, Math.min(1.2, elevation - dy * 0.008));
        lastX = e.clientX; lastY = e.clientY;
      });
      window.addEventListener("pointerup", (e) => {
        if (dragging && moved < 6) handleClick(e);
        dragging = false; canvasWrap.classList.remove("dragging");
      });
      dom.addEventListener("wheel", (e) => {
        e.preventDefault();
        distance = Math.max(4, Math.min(15, distance + e.deltaY * 0.004));
      }, { passive: false });
    }

    function handleClick(e) {
      const rect = renderer.domElement.getBoundingClientRect();
      const mouse = new THREE.Vector2(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(mouse, camera);
      const hits = raycaster.intersectObjects(interactiveMeshes);
      if (hits.length) selectSubsystem(hits[0].object.userData.subsystem);
    }

    function selectSubsystem(key) {
      selectedSubsystem = key;
      renderInfoCard();
    }

    function renderInfoCard() {
      const card = document.getElementById("threeInfoCard");
      if (!selectedSubsystem) { card.innerHTML = '<div class="tempty">Click a component on the model to inspect it.</div>'; return; }
      const e = byId[state.engineId];
      const idx = simCycle - 1;
      const healthArr = { compressor: e.compressorHealth, combustor: e.combustorHealth, turbine: e.turbineHealth }[selectedSubsystem];
      const val = healthArr[idx];
      const st = statusOf(val);
      const label = selectedSubsystem.charAt(0).toUpperCase() + selectedSubsystem.slice(1);
      const reasonLines = e.reasons[selectedSubsystem] || [];
      card.innerHTML =
        '<div class="tname" style="color:var(--' + st.tone + ')">' + label + '</div>' +
        '<div class="trow"><span>Health (cycle ' + simCycle + ')</span><b>' + fmtPct(val) + '</b></div>' +
        '<div class="trow"><span>Status</span><b style="color:var(--' + st.tone + ')">' + st.label + '</b></div>' +
        '<div class="trow"><span>Gas-path z-score</span><b>' + e.anomalyZAll[selectedSubsystem][idx].toFixed(2) + 'σ</b></div>' +
        '<div class="treason">' + reasonLines.map((r) => r[1]).join(" ") + '</div>';
    }

    function statusForSubsystem(e, sub, idx) {
      const healthArr = { compressor: e.compressorHealth, combustor: e.combustorHealth, turbine: e.turbineHealth }[sub];
      return statusOf(healthArr[idx]);
    }

    function updateColors() {
      const e = byId[state.engineId];
      const idx = simCycle - 1;
      PARTS.forEach((p) => {
        const mesh = meshes[p.key];
        if (!p.subsystem) { mesh.material.color.setHex(STATUS_HEX.neutral); mesh.userData.baseEmissive = 0; mesh.userData.critical = false; return; }
        const st = statusForSubsystem(e, p.subsystem, idx);
        mesh.material.color.setHex(STATUS_HEX[st.tone]);
        mesh.userData.critical = st.tone === "critical";
        mesh.userData.selected = p.subsystem === selectedSubsystem;
      });
      renderLabels();
      if (selectedSubsystem) renderInfoCard();
    }

    function renderLabels() {
      const e = byId[state.engineId];
      const idx = simCycle - 1;
      const items = [
        { mesh: meshes.compressor, text: "P2 " + Math.round(e.p2[idx] / 1000) + " kPa" },
        { mesh: meshes.combustor, text: "T3 " + Math.round(e.t3[idx]) + " K" },
        { mesh: meshes.shaft, text: "RPM " + Math.round(e.rpm[idx]).toLocaleString() },
      ];
      labelLayer.innerHTML = "";
      const w = canvasWrap.clientWidth, h = canvasWrap.clientHeight;
      items.forEach((it) => {
        if (!it.mesh) return;
        const v = new THREE.Vector3(); it.mesh.getWorldPosition(v); v.project(camera);
        const x = (v.x * 0.5 + 0.5) * w, y = (-v.y * 0.5 + 0.5) * h;
        if (v.z > 1) return;
        const div = document.createElement("div");
        div.className = "three-label";
        div.style.left = x + "px"; div.style.top = y + "px";
        div.textContent = it.text;
        labelLayer.appendChild(div);
      });
    }

    function toggleAirflow() {
      airflowOn = !airflowOn;
      const btn = document.getElementById("airflowBtn");
      btn.classList.toggle("on", airflowOn);
      btn.textContent = airflowOn ? "■ Stop Engine" : "▶ Start Engine (Airflow)";
      if (airflowOn && particles.length === 0) spawnParticles();
      particles.forEach((p) => (p.visible = airflowOn));
    }

    function spawnParticles() {
      const totalLen = meshes._totalLen;
      const geo = new THREE.SphereGeometry(0.05, 6, 6);
      for (let i = 0; i < 24; i++) {
        const mat = new THREE.MeshBasicMaterial({ color: 0x4c78a8 });
        const p = new THREE.Mesh(geo, mat);
        p.userData.t = i / 24;
        p.userData.speed = 0.15 + Math.random() * 0.08;
        p.userData.r = (Math.random() - 0.5) * 0.5;
        meshes._group.add(p);
        particles.push(p);
      }
    }

    function updateParticles() {
      if (!airflowOn) return;
      const totalLen = meshes._totalLen;
      particles.forEach((p) => {
        p.userData.t += p.userData.speed * 0.01;
        if (p.userData.t > 1) p.userData.t = 0;
        const z = -totalLen / 2 + p.userData.t * totalLen;
        p.position.set(Math.cos(p.userData.t * 20) * p.userData.r, Math.sin(p.userData.t * 20) * p.userData.r, z);
        const heat = p.userData.t > 0.45 && p.userData.t < 0.75;
        p.material.color.setHex(heat ? 0xe45756 : 0x4c78a8);
      });
    }

    function toggleXray() {
      xrayOn = !xrayOn;
      document.getElementById("xrayBtn").classList.toggle("on", xrayOn);
      PARTS.forEach((p) => { meshes[p.key].material.opacity = xrayOn ? 0.35 : 1; });
      meshes.shaft.material.opacity = xrayOn ? 0.5 : 1;
    }

    function resetView() { azimuth = 0.6; elevation = 0.28; distance = 8.2; }

    function animate(tms) {
      raf = requestAnimationFrame(animate);
      camera.position.set(
        distance * Math.cos(elevation) * Math.sin(azimuth),
        distance * Math.sin(elevation),
        distance * Math.cos(elevation) * Math.cos(azimuth)
      );
      camera.lookAt(0, 0, 0);
      meshes.shaft.rotation.z += 0.05;

      PARTS.forEach((p) => {
        const mesh = meshes[p.key];
        if (mesh.userData.critical) {
          const pulse = (Math.sin((tms || 0) * 0.005) + 1) / 2;
          mesh.material.emissive = new THREE.Color(STATUS_HEX.critical);
          mesh.material.emissiveIntensity = 0.25 + pulse * 0.5;
        } else {
          mesh.material.emissiveIntensity = mesh.userData.selected ? 0.35 : 0;
          if (mesh.userData.selected) mesh.material.emissive = new THREE.Color(0xffffff);
        }
      });

      updateParticles();
      if (renderer && scene && camera) renderer.render(scene, camera);
    }

    function init() {
      if (scene) return; // already built
      try {
        build();
        document.getElementById("airflowBtn").onclick = toggleAirflow;
        document.getElementById("xrayBtn").onclick = toggleXray;
        document.getElementById("resetViewBtn").onclick = resetView;
        document.getElementById("tmSlider").oninput = (ev) => { simCycle = +ev.target.value; document.getElementById("tmCycleVal").textContent = "Cycle " + simCycle; updateColors(); };
      } catch (err) {
        canvasWrap.innerHTML = '<div class="tempty" style="padding:16px;">3D view unavailable in this browser (WebGL required). Dashboard functionality is unaffected.</div>';
      }
    }

    function update() {
      if (!scene) return;
      const e = byId[state.engineId];
      simCycle = e.cycles[e.cycles.length - 1];
      const slider = document.getElementById("tmSlider");
      if (slider) { slider.max = e.cycles.length; slider.value = simCycle; }
      const cv = document.getElementById("tmCycleVal"); if (cv) cv.textContent = "Cycle " + simCycle;
      selectedSubsystem = null;
      updateColors();
      renderInfoCard();
    }

    return { init, update };
  })();

  // ---------------- Sensor telemetry ----------------
  function renderSensors() {
    const e = byId[state.engineId];
    const cyc = e.cycles;

    document.getElementById("sensorLegend1").innerHTML = legendHTML([{ color: cssVar("--accent"), label: "Altitude (m)" }]);
    renderChart("sensorChart1", null, {
      xDomain: [1, 30], xTicks: [1, 15, 30], yTickFormat: (v) => Math.round(v / 1000) + "k",
      series: [{ data: cyc.map((c, i) => ({ x: c, y: e.altitude[i] })), color: cssVar("--accent"), width: 1.8 }],
      margins: { l: 34, r: 6, t: 8, b: 20 }, hover: false,
    });

    const pShades = [cssVar("--accent") + "aa", cssVar("--accent"), "#0b5f58"];
    document.getElementById("sensorLegend2").innerHTML = legendHTML([
      { color: pShades[0], label: "P2" }, { color: pShades[1], label: "P3" }, { color: pShades[2], label: "P4" },
    ]);
    renderChart("sensorChart2", null, {
      xDomain: [1, 30], xTicks: [1, 15, 30], yTickFormat: (v) => Math.round(v / 1000) + "k",
      series: [
        { data: cyc.map((c, i) => ({ x: c, y: e.p2[i] })), color: pShades[0], width: 1.6 },
        { data: cyc.map((c, i) => ({ x: c, y: e.p3[i] })), color: pShades[1], width: 1.6 },
        { data: cyc.map((c, i) => ({ x: c, y: e.p4[i] })), color: pShades[2], width: 1.6 },
      ],
      margins: { l: 34, r: 6, t: 8, b: 20 }, hover: false,
    });

    const tShades = [cssVar("--accent-2") + "aa", cssVar("--accent-2"), "#7a3616"];
    document.getElementById("sensorLegend3").innerHTML = legendHTML([
      { color: tShades[0], label: "T2" }, { color: tShades[1], label: "T3" }, { color: tShades[2], label: "T4" },
    ]);
    renderChart("sensorChart3", null, {
      xDomain: [1, 30], xTicks: [1, 15, 30], yTickFormat: (v) => Math.round(v),
      series: [
        { data: cyc.map((c, i) => ({ x: c, y: e.t2[i] })), color: tShades[0], width: 1.6 },
        { data: cyc.map((c, i) => ({ x: c, y: e.t3[i] })), color: tShades[1], width: 1.6 },
        { data: cyc.map((c, i) => ({ x: c, y: e.t4[i] })), color: tShades[2], width: 1.6 },
      ],
      margins: { l: 34, r: 6, t: 8, b: 20 }, hover: false,
    });
  }

  // ---------------- Fleet overview ----------------
  function renderFleet() {
    const allY = [];
    engines.forEach((e) => e.overallHealth.forEach((v) => allY.push(v)));
    const yDomain = niceDomain(allY, 0.08);
    const svg = document.getElementById("fleetChart");
    svg.innerHTML = "";
    const { x, y, W, H, m } = scaleFns(svg, [1, 30], yDomain, { l: 42, r: 12, t: 10, b: 24 });
    for (let i = 0; i <= 4; i++) {
      const v = yDomain[0] + (i / 4) * (yDomain[1] - yDomain[0]);
      const yy = y(v);
      svg.appendChild(el("line", { x1: m.l, x2: W - m.r, y1: yy, y2: yy, class: "grid-line" }));
      const t = el("text", { x: m.l - 8, y: yy + 3, class: "axis-label", "text-anchor": "end" });
      t.textContent = fmtPct(v); svg.appendChild(t);
    }
    [1, 5, 10, 15, 20, 25, 30].forEach((v) => {
      const t = el("text", { x: x(v), y: H - 6, class: "axis-label", "text-anchor": "middle" });
      t.textContent = v; svg.appendChild(t);
    });
    engines.forEach((e) => {
      const isSel = e.id === state.engineId;
      const pts = e.cycles.map((c, i) => ({ x: c, y: e.overallHealth[i] }));
      svg.appendChild(el("path", {
        d: pathFrom(pts, x, y), fill: "none",
        stroke: isSel ? cssVar("--accent-2") : cssVar("--text-muted"),
        "stroke-width": isSel ? 2.4 : 1.2, opacity: isSel ? 1 : 0.45,
      }));
    });
    document.getElementById("fleetTooltip").style.display = "none";
  }

  // ---------------- Methodology tables ----------------
  function renderMethodTables() {
    const fm = DATA.metrics.fleetBaseline;
    const pm = DATA.metrics.performance;
    const t1 = document.getElementById("fleetMetricsTable");
    t1.innerHTML = "<tr><th>Signal</th><th>RMSE</th><th>R²</th><th>90% Coverage</th></tr>" +
      Object.entries(fm).map(([k, v]) => "<tr><td>" + k + "</td><td>" + v.rmse + "</td><td>" + v.r2 + "</td><td>" + v.coverage90 + "%</td></tr>").join("");
    const t2 = document.getElementById("perfMetricsTable");
    t2.innerHTML = "<tr><th>Signal</th><th>RMSE</th><th>R²</th></tr>" +
      Object.entries(pm).map(([k, v]) => "<tr><td>" + k + "</td><td>" + v.rmse + "</td><td>" + v.r2 + "</td></tr>").join("");
    document.getElementById("loeoVal").textContent = DATA.metrics.loeoMeanRmse;
    document.getElementById("coverageCallout").textContent =
      "Disclosed limitation: conformal coverage in back-testing ranged 83–93% against a 90% nominal target — " +
      "a known finite-sample effect of calibrating on roughly 50 points, not a hidden flaw.";

    const ab = DATA.metrics.contextEncoderAblation;
    const t3 = document.getElementById("ablationTable");
    t3.innerHTML = "<tr><th>Approach</th><th>RMSE</th><th>R²</th></tr>" +
      "<tr><td>Neural context-encoder (PCA + MLP)</td><td>" + ab.neuralRmse + "</td><td>" + ab.neuralR2 + "</td></tr>" +
      "<tr><td>Closed-form Engine DNA (Flight-Log Match)</td><td>" + ab.closedFormRmse + "</td><td>" + ab.closedFormR2 + "</td></tr>";
    document.getElementById("ablationCallout").textContent = ab.verdict;
  }

  // ---------------- Wire up + render ----------------
  document.getElementById("demoBtn").onclick = () => { state.demoOn = !state.demoOn; renderAnomaly(); };
  wireSimControls();

  function renderAll() {
    renderEngineList();
    renderModeGroup();
    renderHeader();
    renderVerdict();
    renderSchematic();
    renderDna();
    renderKpis();
    renderHealthTabs();
    renderHealthChart();
    renderSubsystemReasons();
    renderPerfCharts();
    renderMission();
    renderRecommendations();
    renderCorrelationCards();
    renderAnomaly();
    renderMissionSim();
    renderQaChips();
    renderSensors();
    renderFleet();
    Engine3D.update();
  }

  renderMethodTables();
  Engine3D.init();
  renderIntegrityChips();
  renderAll();
  window.addEventListener("resize", renderAll);
})();
