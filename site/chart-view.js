/** Scale and viewport operations. View coordinates are normalized transformed axes. */
export const fullView = () => ({ x: [0, 1], y: [0, 1] });
export const costTransform = (value) => Math.sqrt(value);
export const costInverse = (value) => value ** 2;

export function boundRange(low, high) {
  const span = Math.min(1, Math.max(.005, high - low));
  const start = Math.max(0, Math.min(1 - span, low));
  return [start, start + span];
}

export function boxView(view, start, end) {
  return Object.fromEntries(["x", "y"].map((axis) => {
    const [low, high] = view[axis];
    return [axis, boundRange(low + Math.min(start[axis], end[axis]) * (high - low), low + Math.max(start[axis], end[axis]) * (high - low))];
  }));
}

export function numericTicks(low, high, count = 5) {
  const rough = (high - low) / count;
  if (!(rough > 0)) return [];
  const power = 10 ** Math.floor(Math.log10(rough));
  const fraction = rough / power;
  const step = (fraction >= 5 ? 5 : fraction >= 2 ? 2 : 1) * power;
  const ticks = [];
  for (let index = Math.ceil(low / step); index * step <= high + step * 1e-8; index++) {
    ticks.push(Number((index * step).toPrecision(12)));
  }
  return ticks;
}

/** Bind a single rendered SVG. Redraw occurs on completion, preserving pointer capture. */
export function bindNavigation(svg, geometry, view, update) {
  const { left, top, width, height } = geometry;
  let drag = null;
  let selection = null;
  const position = (event) => {
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    return { x: Math.max(0, Math.min(1, (point.x - left) / width)), y: Math.max(0, Math.min(1, 1 - (point.y - top) / height)) };
  };
  const inside = (event) => {
    const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
    return point.x >= left && point.x <= left + width && point.y >= top && point.y <= top + height;
  };
  svg.addEventListener("pointerdown", (event) => {
    if (drag || event.button !== 0 || !inside(event) || event.target.closest(".point")) return;
    drag = { start: position(event), pointer: event.pointerId };
    svg.setPointerCapture(event.pointerId);
    selection = document.createElementNS(svg.namespaceURI, "rect");
    selection.setAttribute("class", "zoom-selection");
    selection.setAttribute("pointer-events", "none");
    svg.append(selection);
  });
  svg.addEventListener("pointermove", (event) => {
    if (!drag || event.pointerId !== drag.pointer) return;
    const end = position(event);
    const attrs = { x: left + Math.min(end.x, drag.start.x) * width, y: top + (1 - Math.max(end.y, drag.start.y)) * height, width: Math.abs(end.x - drag.start.x) * width, height: Math.abs(end.y - drag.start.y) * height };
    for (const [key, value] of Object.entries(attrs)) selection.setAttribute(key, value);
  });
  const finish = (event) => {
    if (!drag || event.pointerId !== drag.pointer) return;
    const end = position(event);
    const { start } = drag;
    drag = null;
    selection.remove();
    if (event.type === "pointercancel") return;
    if (Math.abs(end.x - start.x) * width > 8 && Math.abs(end.y - start.y) * height > 8) update(boxView(view, start, end));
  };
  svg.addEventListener("pointerup", finish);
  svg.addEventListener("pointercancel", finish);
}
