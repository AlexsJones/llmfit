/**
 * Build-plate planning: pack parts onto plates and estimate print time.
 *
 * Packing uses MaxRects with the Best-Area-Fit heuristic and optional 90°
 * rotation. It is deterministic: parts are sorted by a stable key before
 * packing, so the same design always produces the same plates.
 */
import type { BuildVolume } from '../model/types';
import type { PartMesh } from './types';

export interface PlacedPart {
  id: string;
  name: string;
  /** Position of the part's lower-left corner on the plate (mm). */
  x: number;
  y: number;
  w: number;
  d: number;
  h: number;
  rotated: boolean;
  volume: number;
  area: number;
}

export interface Plate {
  index: number;
  parts: PlacedPart[];
  /** Fraction of the plate area covered (0..1). */
  usage: number;
  /** Tallest part on the plate (mm). */
  height: number;
  volume: number;
  area: number;
}

export interface NestingResult {
  plates: Plate[];
  /** Parts that do not fit on an empty plate at all. */
  rejected: { id: string; name: string; w: number; d: number; h: number }[];
}

export interface NestingOptions {
  /** Gap kept between neighbouring parts (mm). Parts may still touch the plate edge. */
  spacing: number;
  allowRotation: boolean;
}

interface FreeRect {
  x: number;
  y: number;
  w: number;
  d: number;
}

const EPS = 1e-6;

export function nestParts(parts: PartMesh[], bv: BuildVolume, opts: NestingOptions): NestingResult {
  const spacing = Math.max(0, opts.spacing);
  // Each part is grown by one spacing and the plate by the same amount, so neighbours
  // always keep a full gap while a part may still sit flush against the plate edge.
  const plateW = bv.x + spacing;
  const plateD = bv.y + spacing;

  const items = parts
    .map((p) => ({ id: p.id, name: p.name, w: p.size[0] + spacing, d: p.size[1] + spacing, h: p.size[2], volume: p.volume, area: p.area }))
    .sort((a, b) => Math.max(b.w, b.d) - Math.max(a.w, a.d) || b.w * b.d - a.w * a.d || a.id.localeCompare(b.id));

  const rejected: NestingResult['rejected'] = [];
  const queue = items.filter((it) => {
    const fitsFlat = it.w <= plateW + EPS && it.d <= plateD + EPS;
    const fitsTurned = opts.allowRotation && it.d <= plateW + EPS && it.w <= plateD + EPS;
    if ((fitsFlat || fitsTurned) && it.h <= bv.z + EPS) return true;
    rejected.push({ id: it.id, name: it.name, w: it.w - spacing, d: it.d - spacing, h: it.h });
    return false;
  });

  const plates: Plate[] = [];
  let remaining = queue;
  while (remaining.length) {
    const free: FreeRect[] = [{ x: 0, y: 0, w: plateW, d: plateD }];
    const placed: PlacedPart[] = [];
    const left: typeof remaining = [];
    for (const it of remaining) {
      const spot = bestSpot(free, it, opts.allowRotation);
      if (!spot) {
        left.push(it);
        continue;
      }
      const w = spot.rotated ? it.d : it.w;
      const d = spot.rotated ? it.w : it.d;
      placed.push({
        id: it.id,
        name: it.name,
        x: spot.x,
        y: spot.y,
        w: w - spacing,
        d: d - spacing,
        h: it.h,
        rotated: spot.rotated,
        volume: it.volume,
        area: it.area,
      });
      splitFree(free, { x: spot.x, y: spot.y, w, d });
      prune(free);
    }
    // Nothing fitted on an empty plate: stop instead of looping forever.
    if (!placed.length) {
      for (const it of left) rejected.push({ id: it.id, name: it.name, w: it.w - spacing, d: it.d - spacing, h: it.h });
      break;
    }
    const area = placed.reduce((s, p) => s + p.w * p.d, 0);
    plates.push({
      index: plates.length,
      parts: placed,
      usage: Math.min(1, area / (bv.x * bv.y)),
      height: Math.max(...placed.map((p) => p.h)),
      volume: placed.reduce((s, p) => s + p.volume, 0),
      area: placed.reduce((s, p) => s + p.area, 0),
    });
    remaining = left;
  }
  return { plates, rejected };
}

function bestSpot(free: FreeRect[], it: { w: number; d: number }, allowRotation: boolean) {
  let best: { x: number; y: number; rotated: boolean; score: number; tie: number } | null = null;
  for (const r of free) {
    for (const rotated of allowRotation ? [false, true] : [false]) {
      const w = rotated ? it.d : it.w;
      const d = rotated ? it.w : it.d;
      if (w > r.w + EPS || d > r.d + EPS) continue;
      // Best area fit, breaking ties on the shorter leftover side.
      const score = r.w * r.d - w * d;
      const tie = Math.min(r.w - w, r.d - d);
      if (!best || score < best.score - EPS || (Math.abs(score - best.score) < EPS && tie < best.tie)) {
        best = { x: r.x, y: r.y, rotated, score, tie };
      }
    }
  }
  return best;
}

function splitFree(free: FreeRect[], used: FreeRect) {
  for (let i = free.length - 1; i >= 0; i--) {
    const r = free[i];
    if (used.x >= r.x + r.w - EPS || used.x + used.w <= r.x + EPS || used.y >= r.y + r.d - EPS || used.y + used.d <= r.y + EPS) continue;
    free.splice(i, 1);
    if (used.x > r.x + EPS) free.push({ x: r.x, y: r.y, w: used.x - r.x, d: r.d });
    if (used.x + used.w < r.x + r.w - EPS) free.push({ x: used.x + used.w, y: r.y, w: r.x + r.w - (used.x + used.w), d: r.d });
    if (used.y > r.y + EPS) free.push({ x: r.x, y: r.y, w: r.w, d: used.y - r.y });
    if (used.y + used.d < r.y + r.d - EPS) free.push({ x: r.x, y: used.y + used.d, w: r.w, d: r.y + r.d - (used.y + used.d) });
  }
}

/** Drop free rectangles fully contained in another one. */
function prune(free: FreeRect[]) {
  for (let i = free.length - 1; i >= 0; i--) {
    for (let j = 0; j < free.length; j++) {
      if (i === j) continue;
      const a = free[i];
      const b = free[j];
      if (a.x >= b.x - EPS && a.y >= b.y - EPS && a.x + a.w <= b.x + b.w + EPS && a.y + a.d <= b.y + b.d + EPS) {
        free.splice(i, 1);
        break;
      }
    }
  }
}

// ---- print time ------------------------------------------------------------------

export interface PrinterProfile {
  layerHeight: number;
  lineWidth: number;
  /** Nominal print speed (mm/s). */
  speed: number;
  /** Fraction of nominal speed actually reached, covering acceleration and travel. */
  efficiency: number;
  /** Fixed overhead per print job (minutes): heat-up, bed levelling, purge. */
  startupMinutes: number;
  /** Infill density 0..1 used for the solid core of chunky parts. */
  infill: number;
}

export const DEFAULT_PROFILE: PrinterProfile = {
  layerHeight: 0.2,
  lineWidth: 0.42,
  speed: 120,
  efficiency: 0.55,
  startupMinutes: 5,
  infill: 0.15,
};

/**
 * Rough print-time estimate from extruded volume and layer count.
 *
 * This is a flow-rate model, not a slicer: time ≈ volume / (lineWidth ×
 * layerHeight × effective speed), plus a small per-layer overhead. Organiser
 * parts are thin-walled, so this lands in the right ballpark, but a slicer
 * remains the authority.
 */
export function estimateMinutes(volumeMm3: number, heightMm: number, profile: PrinterProfile): number {
  const flow = profile.lineWidth * profile.layerHeight * profile.speed * profile.efficiency; // mm³/s
  if (flow <= 0) return 0;
  const layers = Math.max(1, Math.ceil(heightMm / profile.layerHeight));
  const extrudeSeconds = volumeMm3 / flow;
  // ~0.35 s per layer for the Z move, seam and corner slow-downs.
  return profile.startupMinutes + (extrudeSeconds + layers * 0.35) / 60;
}

export function formatDuration(minutes: number): string {
  if (!Number.isFinite(minutes) || minutes <= 0) return '–';
  const total = Math.round(minutes);
  const h = Math.floor(total / 60);
  const m = total % 60;
  return h ? `${h} h ${String(m).padStart(2, '0')} min` : `${m} min`;
}
