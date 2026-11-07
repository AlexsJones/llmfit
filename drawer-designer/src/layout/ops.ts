/**
 * Pure 2D layout operations on compartment rectangles.
 *
 * All functions are deterministic and side-effect free. They operate in the
 * usable-area coordinate system (0,0 = back-left of the usable area).
 */
import { MIN_COMPARTMENT, newId } from '../model/defaults';
import type { Compartment, Divider, DividerOrientation, Rect } from '../model/types';

export const EPS = 1e-6;

export interface Bounds {
  width: number;
  depth: number;
}

const r2 = (v: number) => Math.round(v * 100) / 100;

export function rectOf(c: Rect): Rect {
  return { x: c.x, y: c.y, w: c.w, d: c.d };
}

export function overlaps(a: Rect, b: Rect, eps = 0.01): boolean {
  return a.x < b.x + b.w - eps && a.x + a.w > b.x + eps && a.y < b.y + b.d - eps && a.y + a.d > b.y + eps;
}

function overlapsY(a: Rect, b: Rect, eps = 0.01) {
  return a.y < b.y + b.d - eps && a.y + a.d > b.y + eps;
}
function overlapsX(a: Rect, b: Rect, eps = 0.01) {
  return a.x < b.x + b.w - eps && a.x + a.w > b.x + eps;
}

export function inBounds(r: Rect, b: Bounds, eps = 0.01): boolean {
  return r.x >= -eps && r.y >= -eps && r.x + r.w <= b.width + eps && r.y + r.d <= b.depth + eps;
}

/**
 * Move a rectangle by (dx, dy), sliding along obstacles instead of passing
 * through them. X is resolved first, then Y, which gives natural sliding along
 * neighbouring compartments.
 */
export function sweepMove(orig: Rect, targetX: number, targetY: number, others: Rect[], bounds: Bounds): Rect {
  let x = Math.min(Math.max(targetX, 0), bounds.width - orig.w);
  // resolve x
  if (x > orig.x) {
    for (const o of others) {
      if (overlapsY(orig, o) && o.x >= orig.x + orig.w - 0.01) x = Math.min(x, o.x - orig.w);
    }
  } else if (x < orig.x) {
    for (const o of others) {
      if (overlapsY(orig, o) && o.x + o.w <= orig.x + 0.01) x = Math.max(x, o.x + o.w);
    }
  }
  const afterX = { ...orig, x };
  let y = Math.min(Math.max(targetY, 0), bounds.depth - orig.d);
  if (y > orig.y) {
    for (const o of others) {
      if (overlapsX(afterX, o) && o.y >= orig.y + orig.d - 0.01) y = Math.min(y, o.y - orig.d);
    }
  } else if (y < orig.y) {
    for (const o of others) {
      if (overlapsX(afterX, o) && o.y + o.d <= orig.y + 0.01) y = Math.max(y, o.y + o.d);
    }
  }
  const result = { ...orig, x, y };
  // Last resort: never return an overlapping placement.
  if (others.some((o) => overlaps(result, o))) return orig;
  return result;
}

export type Edge = 'left' | 'right' | 'top' | 'bottom';

/** Limits how far an edge can move before hitting a neighbour or the drawer bounds. */
export function edgeLimit(r: Rect, edge: Edge, others: Rect[], bounds: Bounds): number {
  switch (edge) {
    case 'right': {
      let lim = bounds.width;
      for (const o of others) if (overlapsY(r, o) && o.x >= r.x + r.w - 0.01) lim = Math.min(lim, o.x);
      return lim;
    }
    case 'left': {
      let lim = 0;
      for (const o of others) if (overlapsY(r, o) && o.x + o.w <= r.x + 0.01) lim = Math.max(lim, o.x + o.w);
      return lim;
    }
    case 'bottom': {
      let lim = bounds.depth;
      for (const o of others) if (overlapsX(r, o) && o.y >= r.y + r.d - 0.01) lim = Math.min(lim, o.y);
      return lim;
    }
    case 'top': {
      let lim = 0;
      for (const o of others) if (overlapsX(r, o) && o.y + o.d <= r.y + 0.01) lim = Math.max(lim, o.y + o.d);
      return lim;
    }
  }
}

/** Move one edge of a rectangle to `value`, respecting neighbours, bounds and minimum size. */
export function resizeEdge(
  r: Rect,
  edge: Edge,
  value: number,
  others: Rect[],
  bounds: Bounds,
  min = MIN_COMPARTMENT,
): Rect {
  const lim = edgeLimit(r, edge, others, bounds);
  switch (edge) {
    case 'right': {
      const right = Math.min(Math.max(value, r.x + min), lim);
      return { ...r, w: right - r.x };
    }
    case 'left': {
      const left = Math.max(Math.min(value, r.x + r.w - min), lim);
      return { ...r, x: left, w: r.x + r.w - left };
    }
    case 'bottom': {
      const bottom = Math.min(Math.max(value, r.y + min), lim);
      return { ...r, d: bottom - r.y };
    }
    case 'top': {
      const top = Math.max(Math.min(value, r.y + r.d - min), lim);
      return { ...r, y: top, d: r.y + r.d - top };
    }
  }
}

/** Grow a rectangle in all four directions until it touches neighbours or the drawer walls. */
export function expandToFill(r: Rect, others: Rect[], bounds: Bounds): Rect {
  let out = { ...r };
  out = resizeEdge(out, 'right', bounds.width, others, bounds, 0);
  out = resizeEdge(out, 'bottom', bounds.depth, others, bounds, 0);
  out = resizeEdge(out, 'left', 0, others, bounds, 0);
  out = resizeEdge(out, 'top', 0, others, bounds, 0);
  return out;
}

/** Find a free spot for a new compartment, preferring the requested size. */
export function findFreeSpot(w: number, d: number, others: Rect[], bounds: Bounds, min = MIN_COMPARTMENT): Rect | null {
  const xs = [0, ...others.map((o) => o.x + o.w)].filter((v) => v <= bounds.width - min + EPS);
  const ys = [0, ...others.map((o) => o.y + o.d)].filter((v) => v <= bounds.depth - min + EPS);
  const candidates: { x: number; y: number }[] = [];
  for (const y of ys) for (const x of xs) candidates.push({ x, y });
  candidates.sort((a, b) => a.y - b.y || a.x - b.x);
  let best: Rect | null = null;
  for (const p of candidates) {
    const seed = { x: p.x, y: p.y, w: min, d: min };
    if (!inBounds(seed, bounds) || others.some((o) => overlaps(seed, o))) continue;
    let r = resizeEdge(seed, 'right', p.x + w, others, bounds, min);
    r = resizeEdge(r, 'bottom', p.y + d, others, bounds, min);
    if (r.w >= w - EPS && r.d >= d - EPS) return r;
    if (!best || r.w * r.d > best.w * best.d) best = r;
  }
  return best;
}

/** Generate an evenly divided grid of rectangles filling the bounds. */
export function gridRects(rows: number, cols: number, bounds: Bounds): Rect[] {
  const out: Rect[] = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x0 = r2((bounds.width * c) / cols);
      const x1 = r2((bounds.width * (c + 1)) / cols);
      const y0 = r2((bounds.depth * r) / rows);
      const y1 = r2((bounds.depth * (r + 1)) / rows);
      out.push({ x: x0, y: y0, w: r2(x1 - x0), d: r2(y1 - y0) });
    }
  }
  return out;
}

/**
 * Split a compartment into `n` equal compartments.
 * 'vertical' cuts produce side-by-side pieces along x; 'horizontal' cuts stack pieces along y.
 * Existing dividers are carried over to whichever piece contains them.
 */
export function splitCompartment(c: Compartment, cut: DividerOrientation, n: number): Compartment[] {
  const pieces: Compartment[] = [];
  const along = cut === 'vertical' ? c.w : c.d;
  for (let i = 0; i < n; i++) {
    const a0 = r2((along * i) / n);
    const a1 = r2((along * (i + 1)) / n);
    const size = r2(a1 - a0);
    const dividers: Divider[] = [];
    for (const div of c.dividers) {
      if (div.orientation === cut) {
        // Divider parallel to the cut: keep it only in the piece it falls into.
        if (div.position > a0 + 3 && div.position < a1 - 3) {
          dividers.push({ ...div, id: newId('dv'), position: r2(div.position - a0) });
        }
      } else {
        dividers.push({ ...div, id: newId('dv') });
      }
    }
    pieces.push({
      ...c,
      id: newId('cp'),
      name: n > 1 ? `${c.name} ${String.fromCharCode(65 + i)}` : c.name,
      x: cut === 'vertical' ? r2(c.x + a0) : c.x,
      y: cut === 'horizontal' ? r2(c.y + a0) : c.y,
      w: cut === 'vertical' ? size : c.w,
      d: cut === 'horizontal' ? size : c.d,
      dividers,
      params: { ...c.params },
    });
  }
  return pieces;
}

/**
 * Evenly redistribute the run of compartments that share the target's row
 * (axis 'x') or column (axis 'y') and touch each other edge-to-edge.
 */
export function distributeRun(all: Compartment[], targetId: string, axis: 'x' | 'y'): Compartment[] {
  const target = all.find((c) => c.id === targetId);
  if (!target) return all;
  const tol = 0.5;
  const sameLane = all.filter((c) =>
    axis === 'x'
      ? Math.abs(c.y - target.y) < tol && Math.abs(c.d - target.d) < tol
      : Math.abs(c.x - target.x) < tol && Math.abs(c.w - target.w) < tol,
  );
  const pos = (c: Compartment) => (axis === 'x' ? c.x : c.y);
  const len = (c: Compartment) => (axis === 'x' ? c.w : c.d);
  sameLane.sort((a, b) => pos(a) - pos(b));
  // Keep only the contiguous chain that contains the target.
  const idx = sameLane.findIndex((c) => c.id === targetId);
  let lo = idx;
  let hi = idx;
  while (lo > 0 && Math.abs(pos(sameLane[lo - 1]) + len(sameLane[lo - 1]) - pos(sameLane[lo])) < tol) lo--;
  while (hi < sameLane.length - 1 && Math.abs(pos(sameLane[hi]) + len(sameLane[hi]) - pos(sameLane[hi + 1])) < tol) hi++;
  const chain = sameLane.slice(lo, hi + 1);
  if (chain.length < 2) return all;
  const start = pos(chain[0]);
  const end = pos(chain[chain.length - 1]) + len(chain[chain.length - 1]);
  const each = (end - start) / chain.length;
  const updates = new Map<string, Partial<Compartment>>();
  chain.forEach((c, i) => {
    const p0 = r2(start + each * i);
    const p1 = r2(start + each * (i + 1));
    const scale = (p1 - p0) / len(c);
    const dividers = c.dividers.map((dv) =>
      (axis === 'x' && dv.orientation === 'vertical') || (axis === 'y' && dv.orientation === 'horizontal')
        ? { ...dv, position: r2(dv.position * scale) }
        : dv,
    );
    updates.set(c.id, axis === 'x' ? { x: p0, w: r2(p1 - p0), dividers } : { y: p0, d: r2(p1 - p0), dividers });
  });
  return all.map((c) => (updates.has(c.id) ? { ...c, ...updates.get(c.id) } : c));
}

/** Evenly space `count` dividers of an orientation across a compartment. */
export function evenDividers(c: Compartment, orientation: DividerOrientation, count: number): Divider[] {
  const span = orientation === 'vertical' ? c.w : c.d;
  const keep = c.dividers.filter((d) => d.orientation !== orientation);
  const added: Divider[] = [];
  for (let i = 1; i <= count; i++) {
    added.push({ id: newId('dv'), orientation, position: r2((span * i) / (count + 1)) });
  }
  return [...keep, ...added];
}

export interface SnapResult {
  value: number;
  guide: number | null;
}

/**
 * Snap a pair of edges (start, start+size) to candidate lines; falls back to grid snapping.
 * Returns the new start position and the guide line that was snapped to (if any).
 */
export function snapSpan(start: number, size: number, lines: number[], threshold: number, grid: number): SnapResult {
  let best: SnapResult | null = null;
  let bestDist = threshold;
  for (const line of lines) {
    const d0 = Math.abs(start - line);
    if (d0 < bestDist) {
      bestDist = d0;
      best = { value: line, guide: line };
    }
    const d1 = Math.abs(start + size - line);
    if (d1 < bestDist) {
      bestDist = d1;
      best = { value: line - size, guide: line };
    }
  }
  if (best) return best;
  return { value: grid > 0 ? Math.round(start / grid) * grid : start, guide: null };
}

export function snapPoint(v: number, lines: number[], threshold: number, grid: number): SnapResult {
  let best: SnapResult | null = null;
  let bestDist = threshold;
  for (const line of lines) {
    const d = Math.abs(v - line);
    if (d < bestDist) {
      bestDist = d;
      best = { value: line, guide: line };
    }
  }
  if (best) return best;
  return { value: grid > 0 ? Math.round(v / grid) * grid : v, guide: null };
}

export function snapLines(others: Rect[], bounds: Bounds): { xs: number[]; ys: number[] } {
  const xs = [0, bounds.width];
  const ys = [0, bounds.depth];
  for (const o of others) {
    xs.push(o.x, o.x + o.w);
    ys.push(o.y, o.y + o.d);
  }
  return { xs, ys };
}

/**
 * Compartments can be merged when their union is itself a rectangle (they tile
 * their bounding box with no gaps) and no other compartment sits inside it.
 */
export function mergeBox(selected: Rect[], others: Rect[]): Rect | null {
  if (selected.length < 2) return null;
  const x = Math.min(...selected.map((r) => r.x));
  const y = Math.min(...selected.map((r) => r.y));
  const w = Math.max(...selected.map((r) => r.x + r.w)) - x;
  const d = Math.max(...selected.map((r) => r.y + r.d)) - y;
  const box = { x, y, w, d };
  const area = selected.reduce((s, r) => s + r.w * r.d, 0);
  if (Math.abs(area - w * d) > 0.5) return null;
  // Overlapping selections could also hit the area check, so reject them explicitly.
  for (let i = 0; i < selected.length; i++)
    for (let j = i + 1; j < selected.length; j++) if (overlaps(selected[i], selected[j])) return null;
  if (others.some((o) => overlaps(box, o))) return null;
  return box;
}

/** Rectangle spanned by two points, normalised. */
export function spanRect(ax: number, ay: number, bx: number, by: number): Rect {
  return { x: Math.min(ax, bx), y: Math.min(ay, by), w: Math.abs(bx - ax), d: Math.abs(by - ay) };
}

/** Move several rectangles together; the whole group is clamped and must not hit anything else. */
export function moveGroup(group: Rect[], dx: number, dy: number, others: Rect[], b: Bounds): { dx: number; dy: number } {
  const x = Math.min(...group.map((r) => r.x));
  const y = Math.min(...group.map((r) => r.y));
  const w = Math.max(...group.map((r) => r.x + r.w)) - x;
  const d = Math.max(...group.map((r) => r.y + r.d)) - y;
  let ddx = Math.min(Math.max(dx, -x), b.width - w - x);
  let ddy = Math.min(Math.max(dy, -y), b.depth - d - y);
  const hits = (ex: number, ey: number) => group.some((r) => others.some((o) => overlaps({ ...r, x: r.x + ex, y: r.y + ey }, o)));
  if (hits(ddx, ddy)) {
    if (!hits(ddx, 0)) ddy = 0;
    else if (!hits(0, ddy)) ddx = 0;
    else return { dx: 0, dy: 0 };
  }
  return { dx: ddx, dy: ddy };
}

export interface LayoutIssue {
  compartmentId: string;
  message: string;
  severity: 'error' | 'warning';
}

export function validateLayout(all: Compartment[], bounds: Bounds): LayoutIssue[] {
  const issues: LayoutIssue[] = [];
  for (const c of all) {
    if (!inBounds(c, bounds)) issues.push({ compartmentId: c.id, message: 'Extends outside the drawer', severity: 'error' });
    for (const o of all) {
      if (o.id < c.id && overlaps(c, o)) {
        issues.push({ compartmentId: c.id, message: `Overlaps ${o.name}`, severity: 'error' });
      }
    }
  }
  return issues;
}

/** Fraction of the usable area covered by compartments (0..1). */
export function coverage(all: Rect[], bounds: Bounds): number {
  const area = bounds.width * bounds.depth;
  if (area <= 0) return 0;
  return Math.min(1, all.reduce((s, r) => s + r.w * r.d, 0) / area);
}

/** Shrink every compartment that no longer fits after the drawer was resized. */
export function fitToBounds(all: Compartment[], bounds: Bounds): Compartment[] {
  return all
    .map((c) => {
      const x = Math.min(c.x, Math.max(0, bounds.width - MIN_COMPARTMENT));
      const y = Math.min(c.y, Math.max(0, bounds.depth - MIN_COMPARTMENT));
      const w = Math.min(c.w, bounds.width - x);
      const d = Math.min(c.d, bounds.depth - y);
      return { ...c, x: r2(x), y: r2(y), w: r2(w), d: r2(d) };
    })
    .filter((c) => c.w >= MIN_COMPARTMENT - EPS && c.d >= MIN_COMPARTMENT - EPS);
}

/** Split a compartment into the fewest equal pieces whose printed footprint fits the build plate. */
export function splitToFit(c: Compartment, maxX: number, maxY: number, gap: number): Compartment[] {
  // Allow the part to be rotated 90° on the bed: try both orientations, keep the one with fewer pieces.
  const plan = (mx: number, my: number) => ({
    nx: Math.max(1, Math.ceil((c.w - gap) / mx - 1e-9)),
    ny: Math.max(1, Math.ceil((c.d - gap) / my - 1e-9)),
  });
  const a = plan(maxX, maxY);
  const b = plan(maxY, maxX);
  const { nx, ny } = a.nx * a.ny <= b.nx * b.ny ? a : b;
  let pieces = [c];
  if (nx > 1) pieces = pieces.flatMap((p) => splitCompartment(p, 'vertical', nx));
  if (ny > 1) pieces = pieces.flatMap((p) => splitCompartment(p, 'horizontal', ny));
  return pieces;
}
