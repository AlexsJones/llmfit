/**
 * Custom outlines: the shape that gets cut into an insert.
 *
 * An outline is a closed polygon in normalised coordinates (0..1 in both axes,
 * y pointing down, i.e. the 2D editor's frame). It is stored on a compartment
 * as a compact string so it travels with the design file.
 *
 * Outlines can be drawn by hand or traced from a photo. The tracing pipeline is
 * pure and deterministic: luminance → threshold → largest blob → contour walk →
 * Douglas–Peucker simplification.
 */

export interface Pt {
  x: number;
  y: number;
}

export const MAX_OUTLINE_POINTS = 400;

// ---- serialisation --------------------------------------------------------------

/** Encode as "x,y x,y …" with 4 decimals – compact and diff-friendly inside a design file. */
export function encodeOutline(points: Pt[]): string {
  return points.map((p) => `${round4(p.x)},${round4(p.y)}`).join(' ');
}

export function decodeOutline(raw: unknown): Pt[] {
  if (typeof raw !== 'string' || !raw) return [];
  const pts: Pt[] = [];
  for (const chunk of raw.split(/\s+/)) {
    const [a, b] = chunk.split(',');
    const x = Number(a);
    const y = Number(b);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    pts.push({ x: clamp01(x), y: clamp01(y) });
    if (pts.length >= MAX_OUTLINE_POINTS) break;
  }
  return pts;
}

const round4 = (v: number) => Math.round(v * 1e4) / 1e4;
const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

// ---- geometry helpers -----------------------------------------------------------

export function polygonArea(pts: Pt[]): number {
  let a = 0;
  for (let i = 0; i < pts.length; i++) {
    const p = pts[i];
    const q = pts[(i + 1) % pts.length];
    a += p.x * q.y - q.x * p.y;
  }
  return a / 2;
}

/** Scale an outline into a w × d box, keeping its aspect ratio and centring it. */
export function fitOutline(pts: Pt[], w: number, d: number, margin = 0): Pt[] {
  if (pts.length < 3) return [];
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const sx = Math.max(1e-6, Math.max(...xs) - minX);
  const sy = Math.max(1e-6, Math.max(...ys) - minY);
  const aw = Math.max(0, w - 2 * margin);
  const ad = Math.max(0, d - 2 * margin);
  const k = Math.min(aw / sx, ad / sy);
  const ox = margin + (aw - sx * k) / 2;
  const oy = margin + (ad - sy * k) / 2;
  return pts.map((p) => ({ x: ox + (p.x - minX) * k, y: oy + (p.y - minY) * k }));
}

/** Rotate a normalised outline about its own centre, then re-normalise to 0..1. */
export function rotateOutline(pts: Pt[], degrees: number): Pt[] {
  const d = ((degrees % 360) + 360) % 360;
  if (pts.length < 3 || d === 0) return pts;
  const a = (d * Math.PI) / 180;
  const cos = Math.cos(a);
  const sin = Math.sin(a);
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  const spun = pts.map((p) => ({ x: cx + (p.x - cx) * cos - (p.y - cy) * sin, y: cy + (p.x - cx) * sin + (p.y - cy) * cos }));
  const minX = Math.min(...spun.map((p) => p.x));
  const minY = Math.min(...spun.map((p) => p.y));
  const k = 1 / Math.max(1e-6, Math.max(...spun.map((p) => p.x)) - minX, Math.max(...spun.map((p) => p.y)) - minY);
  return spun.map((p) => ({ x: round4((p.x - minX) * k), y: round4((p.y - minY) * k) }));
}

/**
 * Place an outline at a real-world size: its longest axis becomes `lengthMm`, and the
 * shape is centred in a w × d cell. Unlike `fitOutline` this does not stretch the shape
 * to the cell, so a traced tool comes out the size the tool actually is.
 */
export function sizeOutline(pts: Pt[], lengthMm: number, w: number, d: number): Pt[] {
  if (pts.length < 3 || lengthMm <= 0) return [];
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const sx = Math.max(...xs) - minX;
  const sy = Math.max(...ys) - minY;
  const k = lengthMm / Math.max(1e-6, Math.max(sx, sy));
  const ow = sx * k;
  const od = sy * k;
  return pts.map((p) => ({ x: (w - ow) / 2 + (p.x - minX) * k, y: (d - od) / 2 + (p.y - minY) * k }));
}

/** Footprint of an outline placed at a real size (mm). */
export function outlineExtent(pts: Pt[], lengthMm: number): { w: number; d: number } {
  if (pts.length < 3) return { w: 0, d: 0 };
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const sx = Math.max(...xs) - Math.min(...xs);
  const sy = Math.max(...ys) - Math.min(...ys);
  const k = lengthMm / Math.max(1e-6, Math.max(sx, sy));
  return { w: sx * k, d: sy * k };
}

/** Douglas–Peucker simplification of a closed polygon. */
export function simplify(pts: Pt[], tolerance: number): Pt[] {
  if (pts.length < 4) return pts;
  const keep = new Uint8Array(pts.length);
  keep[0] = 1;
  keep[pts.length - 1] = 1;
  const stack: [number, number][] = [[0, pts.length - 1]];
  while (stack.length) {
    const [i0, i1] = stack.pop()!;
    let best = -1;
    let bestD = tolerance;
    for (let i = i0 + 1; i < i1; i++) {
      const d = pointLineDistance(pts[i], pts[i0], pts[i1]);
      if (d > bestD) {
        bestD = d;
        best = i;
      }
    }
    if (best > 0) {
      keep[best] = 1;
      stack.push([i0, best], [best, i1]);
    }
  }
  return pts.filter((_, i) => keep[i]);
}

function pointLineDistance(p: Pt, a: Pt, b: Pt): number {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const len2 = dx * dx + dy * dy;
  if (len2 < 1e-12) return Math.hypot(p.x - a.x, p.y - a.y);
  const t = Math.min(1, Math.max(0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2));
  return Math.hypot(p.x - (a.x + t * dx), p.y - (a.y + t * dy));
}

/** Offset a closed polygon outwards by `d` (used as a tolerance around a traced shape). */
export function growPolygon(pts: Pt[], d: number): Pt[] {
  if (pts.length < 3 || d === 0) return pts;
  const cx = pts.reduce((s, p) => s + p.x, 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p.y, 0) / pts.length;
  return pts.map((p) => {
    const len = Math.hypot(p.x - cx, p.y - cy) || 1;
    return { x: p.x + ((p.x - cx) / len) * d, y: p.y + ((p.y - cy) / len) * d };
  });
}

// ---- image tracing ---------------------------------------------------------------

export interface TraceOptions {
  /** 0..1 luminance cut-off; pixels darker than this count as "object". */
  threshold: number;
  /** Simplification tolerance in pixels. */
  tolerance?: number;
  /** Invert, for light objects on a dark background. */
  invert?: boolean;
}

export interface GrayImage {
  width: number;
  height: number;
  /** Luminance 0..1, row-major. */
  data: Float32Array;
}

export function toGray(rgba: Uint8ClampedArray, width: number, height: number): GrayImage {
  const data = new Float32Array(width * height);
  for (let i = 0; i < width * height; i++) {
    const a = rgba[i * 4 + 3] / 255;
    const lum = (0.2126 * rgba[i * 4] + 0.7152 * rgba[i * 4 + 1] + 0.0722 * rgba[i * 4 + 2]) / 255;
    // Transparent pixels count as background (white).
    data[i] = a * lum + (1 - a);
  }
  return { width, height, data };
}

export interface TraceResult {
  /** Outline normalised to its own bounds (longest side = 1), ready to store. */
  points: Pt[];
  /** The same outline in image space (0..1 of the image), for drawing over the photo. */
  inImage: Pt[];
}

/** Trace the largest dark region and return its outline, normalised to 0..1. */
export function traceOutline(img: GrayImage, opts: TraceOptions): Pt[] {
  return traceOutlineDetailed(img, opts).points;
}

/**
 * Trace the largest dark region in an image.
 * Returns empty arrays when nothing usable is found.
 */
export function traceOutlineDetailed(img: GrayImage, opts: TraceOptions): TraceResult {
  const { width: w, height: h } = img;
  const inv = opts.invert ?? false;
  const mask = new Uint8Array(w * h);
  for (let i = 0; i < mask.length; i++) {
    const dark = img.data[i] < opts.threshold;
    mask[i] = (inv ? !dark : dark) ? 1 : 0;
  }
  const blob = largestBlob(mask, w, h);
  if (!blob) return EMPTY_TRACE;
  const contour = traceContour(blob, w, h);
  if (contour.length < 8) return EMPTY_TRACE;
  const simplified = simplify(contour, opts.tolerance ?? Math.max(1.2, Math.min(w, h) / 160));
  const pts = (simplified.length > MAX_OUTLINE_POINTS ? simplify(contour, (opts.tolerance ?? 1.5) * 3) : simplified).slice(0, MAX_OUTLINE_POINTS);
  if (pts.length < 3) return EMPTY_TRACE;
  // Normalise against the shape's own bounds for storage, and against the image for display.
  const xs = pts.map((p) => p.x);
  const ys = pts.map((p) => p.y);
  const minX = Math.min(...xs);
  const minY = Math.min(...ys);
  const sx = Math.max(1, Math.max(...xs) - minX);
  const sy = Math.max(1, Math.max(...ys) - minY);
  const k = 1 / Math.max(sx, sy);
  let out = pts.map((p) => ({ x: round4((p.x - minX) * k), y: round4((p.y - minY) * k) }));
  let inImage = pts.map((p) => ({ x: round4(p.x / w), y: round4(p.y / h) }));
  if (polygonArea(out) < 0) {
    out = out.reverse();
    inImage = inImage.reverse();
  }
  return { points: out, inImage };
}

const EMPTY_TRACE: TraceResult = { points: [], inImage: [] };

/** Keep only the biggest 4-connected region of the mask, with holes filled. */
function largestBlob(mask: Uint8Array, w: number, h: number): Uint8Array | null {
  const label = new Int32Array(w * h).fill(-1);
  const stack: number[] = [];
  let best = -1;
  let bestSize = 0;
  let current = 0;
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start] || label[start] >= 0) continue;
    let size = 0;
    stack.push(start);
    label[start] = current;
    while (stack.length) {
      const i = stack.pop()!;
      size++;
      const x = i % w;
      const y = (i / w) | 0;
      if (x > 0 && mask[i - 1] && label[i - 1] < 0) (label[i - 1] = current), stack.push(i - 1);
      if (x < w - 1 && mask[i + 1] && label[i + 1] < 0) (label[i + 1] = current), stack.push(i + 1);
      if (y > 0 && mask[i - w] && label[i - w] < 0) (label[i - w] = current), stack.push(i - w);
      if (y < h - 1 && mask[i + w] && label[i + w] < 0) (label[i + w] = current), stack.push(i + w);
    }
    if (size > bestSize) {
      bestSize = size;
      best = current;
    }
    current++;
  }
  if (best < 0 || bestSize < 16) return null;
  const out = new Uint8Array(w * h);
  for (let i = 0; i < out.length; i++) out[i] = label[i] === best ? 1 : 0;
  return out;
}

/**
 * Moore-neighbour contour trace of a filled mask; returns pixel coordinates.
 * The search always restarts one step clockwise from where we came, which keeps
 * the walk on the outer boundary instead of cutting across the shape.
 */
function traceContour(mask: Uint8Array, w: number, h: number): Pt[] {
  let start = -1;
  for (let i = 0; i < mask.length; i++)
    if (mask[i]) {
      start = i;
      break;
    }
  if (start < 0) return [];
  const at = (x: number, y: number) => (x < 0 || y < 0 || x >= w || y >= h ? 0 : mask[y * w + x]);
  // 8-neighbourhood in clockwise order (screen axes: y points down).
  const nb = [
    [-1, 0], // W
    [-1, -1], // NW
    [0, -1], // N
    [1, -1], // NE
    [1, 0], // E
    [1, 1], // SE
    [0, 1], // S
    [-1, 1], // SW
  ];
  const sx = start % w;
  const sy = (start / w) | 0;
  const out: Pt[] = [{ x: sx, y: sy }];
  let cx = sx;
  let cy = sy;
  // The start pixel is the first set pixel in row order, so its west neighbour is background.
  let back = 0;
  const limit = 8 * mask.length;
  for (let step = 0; step < limit; step++) {
    let found = false;
    for (let k = 0; k < 8; k++) {
      const d = (back + k) % 8;
      const nx = cx + nb[d][0];
      const ny = cy + nb[d][1];
      if (at(nx, ny)) {
        cx = nx;
        cy = ny;
        // Where we came from, plus one step clockwise.
        back = (d + 5) % 8;
        found = true;
        break;
      }
    }
    if (!found) break;
    if (cx === sx && cy === sy) break;
    out.push({ x: cx, y: cy });
  }
  return out;
}
