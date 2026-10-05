/** Pure chart calculations, shared with the offline Node tests. */
export const mean = (values) => values.reduce((sum, value) => sum + value, 0) / values.length;

export function sampleSD(values) {
  if (values.length < 2) return 0;
  const average = mean(values);
  return Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / (values.length - 1));
}

export function pointsFor(runs) {
  const groups = new Map();
  for (const run of runs) {
    if (run.method !== "Baseline" && run.prompt_status === "unchanged") continue;
    const id = `${run.model}/${run.method}`;
    if (!groups.has(id)) groups.set(id, []);
    groups.get(id).push(run);
  }
  return [...groups].map(([id, members]) => ({
    id,
    model: members[0].model,
    method: members[0].method,
    runs: members,
    accuracy: mean(members.map((run) => run.accuracy)),
    sd: members.length > 1 ? sampleSD(members.map((run) => run.accuracy)) : null,
    status: members[0].prompt_status,
    cost: members.every((run) => Number.isFinite(run.evaluation))
      ? mean(members.map((run) => run.evaluation)) : null,
  }));
}

export function paretoFront(points) {
  const known = points.filter((point) => Number.isFinite(point.cost));
  return known.filter((point) => !known.some((other) =>
    other.cost <= point.cost && other.accuracy >= point.accuracy
    && (other.cost < point.cost || other.accuracy > point.accuracy),
  )).sort((a, b) => a.cost - b.cost || a.accuracy - b.accuracy);
}

export function iterationInfo(run, prompt) {
  if (run.method === "Baseline") return { best: "Not applicable", total: "No optimization", note: "Original benchmark instruction; no compiled artifact." };
  if (run.method === "BFS") return { best: "Not applicable", total: "Demonstration selection", note: "BFS selects demonstrations. No iterative search trace is reported." };
  const trace = prompt?.trace;
  if (!trace || trace.source === "unavailable") return { best: "Not recorded", total: "Not recorded", note: "The released artifact has no iteration history. Model-call counts are not iteration counts." };
  return {
    best: trace.best_iteration == null ? "Not recorded" : String(trace.best_iteration),
    total: trace.num_iterations == null ? "Not recorded" : String(trace.num_iterations),
    note: `Recorded best trace index, in the optimizer’s original indexing. Trace source: ${trace.source}.`,
  };
}
