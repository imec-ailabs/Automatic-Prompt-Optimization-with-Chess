import { iterationInfo, paretoFront, pointsFor } from "./data.js";
import { bindNavigation, costTransform, costInverse, fullView, numericTicks } from "./chart-view.js";

const $ = (id) => document.getElementById(id);
const NS = "http://www.w3.org/2000/svg";
const colors = ["#7b6fa6", "#967230", "#3b87a3", "#be6672", "#566abb", "#168366", "#bd6534", "#925d91"];
const shapes = {
  Baseline: "M -5,-5 L 5,5 M -5,5 L 5,-5",
  BFS: "M -5,-5 H 5 V 5 H -5 Z",
  BRS: "M 0,-6 L 6,0 0,6 -6,0 Z",
  COPRO: "M 0,-6 L 6,5 -6,5 Z",
  GEPA: "M -6,-4 L 0,-7 6,-4 6,4 0,7 -6,4 Z",
  MIPROv2: "M -6,-5 L 6,-5 0,6 Z",
  SIMBA: "M -2,-6 H 2 V -2 H 6 V 2 H 2 V 6 H -2 V 2 H -6 V -2 H -2 Z",
};
const costLabel = "Inference cost · full test set (USD)";
const statusNames = { baseline: "Baseline (unoptimized)", updated: "Updated prompt" };
let data;
let activeModels;
let activeMethods;
let currentPoints = [];
let inspected = null;
let pinned = false;
let displayedRun = null;
let resizeTimer;
let view = fullView();
const money = (value) => value == null ? "Not recorded" : `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 4 })}`;

function svgElement(tag, attrs = {}, text = "") {
  const node = document.createElementNS(NS, tag);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (text) node.textContent = text;
  return node;
}

function filterButton(name, index, kind) {
  const button = document.createElement("button");
  button.className = "chip";
  button.type = "button";
  button.setAttribute("aria-pressed", "true");
  if (kind === "model") {
    const dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = colors[index];
    button.append(dot);
  } else {
    const icon = svgElement("svg", { viewBox: "-8 -8 16 16", "aria-hidden": "true" });
    icon.append(svgElement("path", { d: shapes[name], fill: "currentColor", stroke: "currentColor", "stroke-width": 1.5 }));
    button.append(icon);
  }
  button.append(document.createTextNode(name));
  button.addEventListener("click", () => {
    const active = kind === "model" ? activeModels : activeMethods;
    if (active.has(name)) active.delete(name); else active.add(name);
    button.setAttribute("aria-pressed", String(active.has(name)));
    render();
  });
  return button;
}

function drawChart(points, frontier) {
  const width = Math.max(320, $("chart").clientWidth);
  const height = Math.max(300, Math.min(460, width * .64));
  const margin = { left: 51, right: 23, top: 24, bottom: 61 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const known = points.filter((point) => point.cost !== null);
  const maximum = Math.max(.01, ...known.map((point) => point.cost)) * 1.08;
  const transform = costTransform;
  const normalizedX = (value) => transform(value) / transform(maximum);
  const x = (value) => margin.left + (normalizedX(value) - view.x[0]) / (view.x[1] - view.x[0]) * plotWidth;
  const maxAccuracy = Math.max(10, Math.ceil(Math.max(0, ...known.map((point) => point.accuracy + point.sd)) / 5) * 5 + 5);
  const y = (value) => margin.top + plotHeight * (1 - (value / maxAccuracy - view.y[0]) / (view.y[1] - view.y[0]));
  const lowerCost = costInverse(view.x[0] * transform(maximum));
  const upperCost = costInverse(view.x[1] * transform(maximum));
  const bottom = margin.top + plotHeight;
  const svg = svgElement("svg", { viewBox: `0 0 ${width} ${height}`, role: "group", "aria-label": `${costLabel} versus puzzle accuracy. Focus a point to inspect it; Enter pins it.` });
  svg.append(svgElement("title", {}, "Cost–performance Pareto explorer"));
  const defs = svgElement("defs");
  const clip = svgElement("clipPath", { id: "plot-clip" });
  clip.append(svgElement("rect", { x: margin.left, y: margin.top, width: plotWidth, height: plotHeight }));
  defs.append(clip);
  svg.append(defs);
  svg.append(svgElement("rect", { x: margin.left, y: margin.top, width: plotWidth, height: plotHeight, fill: "white", class: "plot-surface" }));
  for (const tick of numericTicks(view.y[0] * maxAccuracy, view.y[1] * maxAccuracy, 7)) {
    svg.append(svgElement("line", { x1: margin.left, x2: width - margin.right, y1: y(tick), y2: y(tick), stroke: "#e7e9e0", "stroke-width": 1 }));
    svg.append(svgElement("text", { x: margin.left - 12, y: y(tick) + 4, "text-anchor": "end" }, `${tick}%`));
  }
  const candidates = numericTicks(lowerCost, upperCost, 8);
  for (let power = -5; power <= Math.ceil(Math.log10(maximum)); power++) {
    for (const multiple of [1, 2, 5]) candidates.push(multiple * 10 ** power);
  }
  const ticks = [...new Set(candidates)].filter((tick) => tick >= lowerCost && tick <= upperCost).sort((a, b) => a - b);
  let previousX = -Infinity;
  for (const tick of ticks) {
    if (x(tick) - previousX < 65) continue;
    previousX = x(tick);
    svg.append(svgElement("line", { x1: x(tick), x2: x(tick), y1: margin.top, y2: bottom, stroke: "#eff0e9" }));
    svg.append(svgElement("text", { x: x(tick), y: bottom + 22, "text-anchor": "middle" }, money(tick)));
  }
  svg.append(svgElement("text", { x: margin.left, y: 12, class: "axis-title" }, "Puzzle accuracy ↑"));
  svg.append(svgElement("text", { x: margin.left + plotWidth / 2, y: height - 7, "text-anchor": "middle", class: "axis-title" }, costLabel));
  const clipped = svgElement("g", { "clip-path": "url(#plot-clip)" });
  const layer = svgElement("g", { class: "data-layer" });
  clipped.append(layer);
  svg.append(clipped);
  if (frontier.length > 1) {
    layer.append(svgElement("polyline", { points: frontier.map((point) => `${x(point.cost)},${y(point.accuracy)}`).join(" "), fill: "none", stroke: "#196b53", "stroke-width": 1.7, "stroke-dasharray": "5 5", opacity: .7 }));
  }
  const frontierIds = new Set(frontier.map((point) => point.id));
  for (const point of known) {
    const color = colors[data.models.indexOf(point.model)];
    if (point.runs.length > 1 && point.sd > 0) {
      const upper = y(point.accuracy + point.sd);
      const lower = y(Math.max(0, point.accuracy - point.sd));
      layer.append(svgElement("path", {
        d: `M ${x(point.cost)},${upper} V ${lower} M ${x(point.cost) - 3},${upper} H ${x(point.cost) + 3} M ${x(point.cost) - 3},${lower} H ${x(point.cost) + 3}`,
        fill: "none", stroke: color, "stroke-width": 1.2, "pointer-events": "none",
      }));
    }
    if (normalizedX(point.cost) < view.x[0] || normalizedX(point.cost) > view.x[1] || point.accuracy / maxAccuracy < view.y[0] || point.accuracy / maxAccuracy > view.y[1]) continue;
    const group = svgElement("g", {
      transform: `translate(${x(point.cost)},${y(point.accuracy)})`,
      class: `point status-${point.status}${inspected?.id === point.id ? " selected" : ""}`,
      tabindex: "0", role: "button", "data-id": point.id,
      "aria-label": `${point.model}, ${point.method}, ${statusNames[point.status]}, ${point.runs.length === 1 ? `run ${point.runs[0].seed}` : `${point.runs.length}-run mean`}, ${point.accuracy.toFixed(2)} percent, ${money(point.cost)}${frontierIds.has(point.id) ? ", Pareto optimal" : ""}`,
    });
    group.append(svgElement("circle", { r: 10, class: "halo" }));
    group.append(svgElement("path", { d: shapes[point.method], fill: point.status === "updated" ? color : "#ffffff", stroke: color, "stroke-width": 1.8 }));
    group.addEventListener("pointerenter", () => { if (!pinned) inspect(point); });
    group.addEventListener("focus", () => { if (!pinned) inspect(point); });
    group.addEventListener("click", () => inspect(point, true));
    group.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inspect(point, true); }
    });
    layer.append(group);
  }
  if (!known.length) svg.append(svgElement("text", { x: width / 2, y: height / 2, "text-anchor": "middle" }, "No recorded budgets for these filters."));
  $("chart").replaceChildren(svg);
  bindNavigation(svg, { left: margin.left, top: margin.top, width: plotWidth, height: plotHeight }, view, updateView);
  $("view-status").textContent = `View: ${money(lowerCost)}–${money(upperCost)}; ${(view.y[0] * maxAccuracy).toFixed(1)}–${(view.y[1] * maxAccuracy).toFixed(1)}% accuracy. ${layer.querySelectorAll(".point").length} points in view. Frontier uses all filtered points.`;
}

function updateView(next) {
  view = next;
  drawChart(currentPoints, paretoFront(currentPoints));
}

function metric(label, value) {
  const group = document.createElement("div");
  const term = document.createElement("dt");
  const description = document.createElement("dd");
  term.textContent = label;
  description.textContent = value;
  group.append(term, description);
  return group;
}

function inspect(point, pin = false) {
  const same = inspected?.id === point.id;
  inspected = point;
  if (pin) pinned = true;
  $("unpin").hidden = !pinned;
  $("artifact-title").textContent = point.model;
  $("artifact-subtitle").textContent = `${point.method} · ${point.runs.length > 1 ? `mean of ${point.runs.length} retained runs; select an artifact below` : `run ${point.runs[0].seed} · n = 1 (SD unavailable)`} ${pinned ? "· pinned" : ""}`;
  $("prompt-status").hidden = false;
  $("prompt-status").textContent = `${statusNames[point.status]} · ${point.runs.length} retained ${point.runs.length === 1 ? "run" : "runs"}`;
  $("prompt-status").className = `prompt-status status-${point.status}`;
  const seedSelect = $("artifact-seed");
  const previousSeed = same ? seedSelect.value : null;
  seedSelect.replaceChildren(...point.runs.map((run) => new Option(`Run ${run.seed} · ${run.accuracy.toFixed(2)}% solved`, run.id)));
  if (previousSeed && point.runs.some((run) => run.id === previousSeed)) seedSelect.value = previousSeed;
  $("seed-label").hidden = point.runs.length < 2;
  showRun();
  for (const node of document.querySelectorAll(".point")) node.classList.toggle("selected", node.dataset.id === point.id);
}

function showRun() {
  if (!inspected) return;
  const point = inspected;
  const run = point.runs.find((item) => item.id === $("artifact-seed").value) || point.runs[0];
  displayedRun = run;
  const prompt = data.prompts[run.prompt_id];
  const info = iterationInfo(run, prompt);
  const metrics = [
    metric(point.runs.length > 1 ? "Mean accuracy ± sample SD" : "Puzzle accuracy", `${point.accuracy.toFixed(2)}%${point.runs.length > 1 ? ` ± ${point.sd.toFixed(2)}` : ""}`),
    metric("Mean inference cost", money(point.cost)),
    metric(`Run ${run.seed} · puzzles solved`, `${run.correct} / ${run.puzzles}`),
    metric(`Run ${run.seed} · inference cost`, money(run.evaluation)),
    metric("Best trace index", info.best),
    metric("Total trace entries", info.total),
  ];
  $("artifact-metrics").replaceChildren(...metrics);
  $("trace-note").textContent = `${info.note}${prompt?.unchanged ? " This run retained the original prompt." : ""}${prompt?.condition_no_update ? " The paper marks this model/optimizer cell N.U. (no seed updated the prompt)." : ""}`;
  $("prompt-text").textContent = prompt?.text ?? data.baseline_prompt;
  $("prompt-text").scrollTop = 0;
  $("copy").disabled = false;
  $("copy").textContent = "Copy";
  $("artifact-provenance").textContent = `Evaluation: ${run.id}. ${prompt ? `Prompt: ${run.prompt_id}. Compiled program SHA-256: ${prompt.sha256}.` : "Baseline instruction: src/chess_self_improvement/dspy_program.py, ChessMoveSignature."} Inference cost source: ${run.evaluation_cost_source ?? "not recorded"}. Artifact text contains the released static prompt, not the runtime adapter request.`;
}

function renderTable(points) {
  $("cost-heading").textContent = "Inference cost (USD)";
  $("results-table").replaceChildren(...points.slice().sort((a, b) => b.accuracy - a.accuracy).map((point) => {
    const row = document.createElement("tr");
    for (const value of [point.model, point.method, point.runs.length > 1 ? `Mean of ${point.runs.length}` : `Seed ${point.runs[0].seed} (n=1)`, statusNames[point.status], `${point.accuracy.toFixed(2)}%${point.runs.length > 1 ? ` ± ${point.sd.toFixed(2)}` : " · SD unavailable"}`, money(point.cost)]) {
      const cell = document.createElement("td");
      cell.textContent = value;
      row.append(cell);
    }
    const cell = document.createElement("td");
    const button = document.createElement("button");
    button.textContent = "Prompt ↗";
    button.className = "quiet";
    button.setAttribute("aria-label", `Inspect ${point.model} ${point.method} ${point.runs.length === 1 ? point.runs[0].seed : "prompts"}`);
    button.addEventListener("click", () => {
      inspect(point, true);
      $("artifact-title").scrollIntoView({ block: "center" });
      $("artifact-title").focus({ preventScroll: true });
    });
    cell.append(button);
    row.append(cell);
    return row;
  }));
}

function clearInspector() {
  pinned = false;
  inspected = null;
  displayedRun = null;
  $("unpin").hidden = true;
  $("seed-label").hidden = true;
  $("prompt-status").hidden = true;
  $("artifact-title").textContent = "Inspect a point";
  $("artifact-subtitle").textContent = "Hover, focus, or tap a plotted point to see its artifact.";
  $("artifact-metrics").replaceChildren();
  $("trace-note").textContent = "";
  $("prompt-text").textContent = "The instruction and demonstrations will appear here.";
  $("artifact-provenance").textContent = "";
  $("copy").disabled = true;
}

function render() {
  view = fullView();
  const filtered = data.runs.filter((run) => activeModels.has(run.model) && activeMethods.has(run.method));
  currentPoints = pointsFor(filtered);
  const frontier = paretoFront(currentPoints);
  const replacement = inspected && currentPoints.find((point) => point.id === inspected.id);
  if (replacement) inspect(replacement); else clearInspector();
  drawChart(currentPoints, frontier);
  renderTable(currentPoints);
  renderMissingCosts(currentPoints);
  const missing = currentPoints.filter((point) => point.cost === null).length;
  const plotted = currentPoints.length - missing;
  const retained = currentPoints.reduce((count, point) => count + point.runs.length, 0);
  $("coverage").textContent = `${plotted} conditions · ${retained} retained runs · ${filtered.length - retained} unchanged runs excluded.${missing ? ` ${missing} conditions with unavailable inference cost listed below.` : ""}`;
}

function renderMissingCosts(points) {
  const missing = points.filter((point) => point.cost === null);
  $("missing-costs").hidden = !missing.length;
  $("missing-cost-list").replaceChildren(...missing.map((point) => {
    const button = document.createElement("button");
    button.className = "missing-cost-result";
    button.textContent = `${point.model} · ${point.method} · ${point.accuracy.toFixed(2)}% · n=${point.runs.length}`;
    button.dataset.model = point.model;
    button.addEventListener("pointerenter", () => { if (!pinned) inspect(point); });
    button.addEventListener("focus", () => { if (!pinned) inspect(point); });
    button.addEventListener("click", () => {
      inspect(point, true);
      $("artifact-title").focus({ preventScroll: true });
      $("artifact-title").scrollIntoView({ block: "center" });
    });
    return button;
  }));
}

async function initialize() {
  try {
    const response = await fetch(new URL("results.json", import.meta.url));
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    data = await response.json();
    activeModels = new Set(data.models);
    activeMethods = new Set(data.methods);
    $("model-filters").replaceChildren(...data.models.map((name, index) => filterButton(name, index, "model")));
    $("method-filters").replaceChildren(...data.methods.map((name, index) => filterButton(name, index, "method")));
    $("reset-view").addEventListener("click", () => updateView(fullView()));
    $("artifact-seed").addEventListener("change", showRun);
    $("unpin").addEventListener("click", () => { pinned = false; if (inspected) inspect(inspected); });
    document.addEventListener("keydown", (event) => { if (event.key === "Escape") { pinned = false; if (inspected) inspect(inspected); } });
    $("reset").addEventListener("click", () => {
      activeModels = new Set(data.models);
      activeMethods = new Set(data.methods);
      for (const chip of document.querySelectorAll(".chip")) chip.setAttribute("aria-pressed", "true");
      clearInspector();
      render();
    });
    $("copy").addEventListener("click", async () => {
      const runId = displayedRun?.id;
      try {
        await navigator.clipboard.writeText($("prompt-text").textContent);
        if (displayedRun?.id === runId) $("copy").textContent = "Copied ✓";
      } catch {
        $("copy").textContent = "Select text to copy";
        const range = document.createRange();
        range.selectNodeContents($("prompt-text"));
        window.getSelection()?.removeAllRanges();
        window.getSelection()?.addRange(range);
      }
    });
    window.addEventListener("resize", () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => drawChart(currentPoints, paretoFront(currentPoints)), 100);
    });
    render();
  } catch (error) {
    $("coverage").textContent = `Could not load the results (${error.message}). Reload the page or download the data using the link above. Local previews must be served over HTTP; see docs/results-site.md.`;
  }
}

initialize();
