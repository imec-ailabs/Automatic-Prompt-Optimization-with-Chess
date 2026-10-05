import assert from "node:assert/strict";
import { test } from "node:test";
import { iterationInfo, paretoFront, pointsFor } from "../site/data.js";
import { readFileSync } from "node:fs";
import { boxView, costInverse, costTransform, fullView, numericTicks } from "../site/chart-view.js";

test("dominance requires a strict improvement; ties survive, unknown costs do not", () => {
  const points = [
    { id: "baseline", cost: 0, accuracy: 10 },
    { id: "tie", cost: 0, accuracy: 10 },
    { id: "worse", cost: 0, accuracy: 9 },
    { id: "dominated", cost: 2, accuracy: 10 },
    { id: "better", cost: 3, accuracy: 12 },
    { id: "unknown", cost: null, accuracy: 100 },
  ];
  assert.deepEqual(paretoFront(points).map((point) => point.id), ["baseline", "tie", "better"]);
  assert.deepEqual(paretoFront([]), []);
});

test("aggregate costs require every run; SD uses the sample denominator", () => {
  const runs = [10, 12, 14].map((accuracy, index) => ({
    id: String(index), model: "model", method: "GEPA", accuracy, evaluation: index === 1 ? null : index,
  }));
  const [point] = pointsFor(runs);
  assert.equal(point.accuracy, 12);
  assert.equal(point.sd, 2);
  assert.equal(point.cost, null);
  assert.equal(point.runs.length, 3);
  assert.equal(pointsFor(runs.slice(0, 1))[0].cost, 0);
});

test("recorded iteration zero is retained; unavailable trace is not zero iterations", () => {
  const run = { method: "COPRO" };
  assert.equal(iterationInfo(run, { trace: { source: "copro_depth", best_iteration: 0, num_iterations: 3 } }).best, "0");
  assert.equal(iterationInfo(run, { trace: { source: "unavailable", num_iterations: 0 } }).total, "Not recorded");
  assert.equal(iterationInfo(run, null).best, "Not recorded");
});

test("unchanged runs are excluded before aggregation, including wholly unchanged conditions", () => {
  const makeRun = (prompt_status, index) => ({ id: String(index), model: "model", method: "COPRO", accuracy: 10, evaluation: 2, prompt_status });
  const runs = ["updated", "unchanged", "unchanged"].map(makeRun);
  runs[0].accuracy = 20;
  const [retained] = pointsFor(runs);
  assert.equal(retained.status, "updated");
  assert.equal(retained.runs.length, 1);
  assert.equal(retained.accuracy, 20);
  assert.equal(retained.sd, null);
  assert.equal(pointsFor(runs.slice(1)).length, 0);
  assert.equal(pointsFor(runs.slice(0, 1))[0].status, "updated");
});

test("inference costs plot all six Jev conditions without compilation or baseline additions", () => {
  const data = JSON.parse(readFileSync(new URL("../site/results.json", import.meta.url)));
  const points = pointsFor(data.runs.filter((run) => run.model === "Jev 1.13"));
  assert.equal(points.length, 6);
  assert.ok(points.every((point) => point.cost > 0));
  assert.equal(new Set(points.map((point) => point.accuracy)).size, 6);
  for (const point of points) {
    assert.equal(point.cost, point.runs.reduce((sum, run) => sum + run.evaluation, 0) / point.runs.length);
  }
  const synthetic = pointsFor([{ model: "m", method: "GEPA", accuracy: 20, evaluation: .2, compile: 100, baseline_reference: 50, cumulative: 150 }]);
  assert.equal(synthetic[0].cost, .2);
});

test("cost transforms round-trip zero and positive USD values and preserve order", () => {
    const values = [0, .001, .1, 1, 15, 100];
    const mapped = values.map(costTransform);
    for (let i = 0; i < values.length; i++) {
      assert.ok(Math.abs(costInverse(mapped[i]) - values[i]) < 1e-9);
      if (i) assert.ok(mapped[i] > mapped[i - 1]);
    }
});

test("box zoom is direction-independent and ticks adapt to narrow viewports", () => {
  const start = { x: .8, y: .2 };
  const end = { x: .2, y: .8 };
  assert.deepEqual(boxView(fullView(), start, end), { x: [.2, .8], y: [.2, .8] });
  assert.deepEqual(boxView(fullView(), start, end), boxView(fullView(), end, start));
  const ticks = numericTicks(1.03, 1.08);
  assert.ok(ticks.length >= 3);
  assert.ok(ticks.every((tick) => tick >= 1.03 && tick <= 1.08));
});
