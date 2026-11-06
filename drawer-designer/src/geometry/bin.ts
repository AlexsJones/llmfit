/**
 * Printable bin generation using exact CSG (manifold-3d).
 *
 * A bin is built as:
 *   solid   = extrude(roundedRect(outer), height)
 *   pocket  = extrude(roundedRect(inner = outer offset by -wall), height) raised by the floor thickness
 *   body    = solid − pocket
 *   walls   = ∪ divider slabs, each intersected with the solid so it is clipped to the footprint
 *   bin     = body ∪ walls
 *
 * Every boolean result from manifold-3d is guaranteed to be a closed, oriented
 * 2-manifold, so the exported STL is watertight by construction.
 * Local frame: X = width, Y = depth (towards the back), Z = up; min corner at the origin.
 */
import type { CrossSection, Manifold } from 'manifold-3d';
import type { Kernel } from './kernel';

export interface BinDivider {
  orientation: 'vertical' | 'horizontal';
  /** Local coordinate of the divider centre: X for vertical, Y for horizontal (mm). */
  at: number;
}

export interface BinSpec {
  width: number;
  depth: number;
  height: number;
  wall: number;
  bottom: number;
  /** Outer corner radius; 0 = square corners. */
  cornerRadius: number;
  dividers: BinDivider[];
  dividerThickness: number;
  dividerHeight: number;
}

/** Number of segments for a full circle; fixed so geometry is deterministic. */
export const CIRCLE_SEGMENTS = 64;

type Disposable = { delete(): void };

/** Tracks intermediate WASM objects so they can be freed in one go. */
export class Scope {
  private items: Disposable[] = [];
  keep<T extends Disposable>(o: T): T {
    this.items.push(o);
    return o;
  }
  /** Free everything except the given survivors. */
  dispose(...survivors: Disposable[]) {
    for (const o of this.items) if (!survivors.includes(o)) o.delete();
    this.items = [];
  }
}

export function roundedRect(k: Kernel, s: Scope, w: number, d: number, r: number): CrossSection {
  const rr = Math.max(0, Math.min(r, w / 2 - 0.01, d / 2 - 0.01));
  if (rr <= 0.05) return s.keep(k.CrossSection.square([w, d]));
  const core = s.keep(k.CrossSection.square([w - 2 * rr, d - 2 * rr]));
  const moved = s.keep(core.translate([rr, rr]));
  return s.keep(moved.offset(rr, 'Round', 2, CIRCLE_SEGMENTS));
}

export function validateBinSpec(spec: BinSpec): string | null {
  if (spec.width < 2 * spec.wall + 1 || spec.depth < 2 * spec.wall + 1) return 'Too small for the wall thickness';
  if (spec.height <= spec.bottom + 0.2) return 'Height must exceed the floor thickness';
  return null;
}

/** Build a bin. The returned Manifold is owned by the caller (call `.delete()`). */
export function buildBin(k: Kernel, spec: BinSpec, s: Scope = new Scope()): Manifold {
  const { width: w, depth: d, height: h, wall, bottom } = spec;
  const r = Math.max(0, spec.cornerRadius);

  const outline = roundedRect(k, s, w, d, r);
  const solid = s.keep(outline.extrude(h));

  // Inner corner radius follows the outer one; keep a small fillet when the outer radius is tight
  // so the wall never becomes razor thin at the corners.
  const iw = w - 2 * wall;
  const id = d - 2 * wall;
  const ri = r > 0 ? Math.max(r - wall, Math.min(0.5, iw / 4, id / 4)) : 0;
  const innerOutline = s.keep(roundedRect(k, s, iw, id, ri).translate([wall, wall]));
  // The pocket overshoots the top by 1 mm so the boolean has no coplanar faces.
  const pocket = s.keep(s.keep(innerOutline.extrude(h - bottom + 1)).translate([0, 0, bottom]));
  let body = s.keep(solid.subtract(pocket));

  const t = spec.dividerThickness;
  const dh = Math.min(Math.max(spec.dividerHeight, bottom + 0.4), h);
  const slabs: Manifold[] = [];
  for (const div of spec.dividers) {
    const slab =
      div.orientation === 'vertical'
        ? s.keep(s.keep(k.Manifold.cube([t, d + 2, dh])).translate([div.at - t / 2, -1, 0]))
        : s.keep(s.keep(k.Manifold.cube([w + 2, t, dh])).translate([-1, div.at - t / 2, 0]));
    slabs.push(s.keep(slab.intersect(solid)));
  }
  if (slabs.length) {
    const walls = s.keep(k.Manifold.union(slabs));
    body = s.keep(body.add(walls));
  }
  s.dispose(body);
  return body;
}
