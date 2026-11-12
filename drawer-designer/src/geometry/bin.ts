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
  /** Radius of the finger scoop along the front inner wall (0 = none). */
  scoopRadius?: number;
  /** Depth of the label ledge along the back wall at the rim (0 = none). */
  labelTab?: number;
  /** Additional scoops: `y0` is the local Y of a front-facing wall face (e.g. a divider's back face). */
  extraScoops?: { y0: number; r: number }[];
}

/** Inner (pocket) corner radius for a bin; shared by modules that fill or cut the pocket. */
export function innerRadius(spec: BinSpec): number {
  const iw = spec.width - 2 * spec.wall;
  const id = spec.depth - 2 * spec.wall;
  const r = Math.max(0, spec.cornerRadius);
  return r > 0 ? Math.max(r - spec.wall, Math.min(0.5, iw / 4, id / 4)) : 0;
}

/** Thickness of the flat top of a label ledge (mm). */
export const LABEL_TAB_TOP = 1.2;

/** How far a fillet profile reaches into the surrounding walls so the union fuses cleanly (mm). */
const FILLET_OVERLAP = 0.5;
/** How far a fillet's circle cuts past the faces it meets, avoiding a tangential contact (mm). */
const FILLET_BITE = 0.1;

/**
 * Extrude a profile drawn in the (Y, Z) plane along X from x0 to x1.
 * Maps profile (a, b) + extrusion c to world (x = x0 + c, y = a, z = b); a proper rotation, so
 * the winding (and therefore manifold orientation) is preserved.
 */
export function extrudeAlongX(s: Scope, profile: CrossSection, x0: number, x1: number): Manifold {
  const m = s.keep(profile.extrude(x1 - x0));
  return s.keep(m.transform([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, x0, 0, 0, 1]));
}

/** Number of segments for a full circle; fixed so geometry is deterministic. */
export const CIRCLE_SEGMENTS = 64;

type Disposable = { delete(): void };

/** Tracks intermediate WASM objects so they can be freed in one go. */
export class Scope {
  private items: Disposable[] = [];
  keep<T extends Disposable>(o: T): T {
    // Guard against registering the same handle twice: the second delete would throw.
    if (!this.items.includes(o)) this.items.push(o);
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
  const ri = innerRadius(spec);
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

  // Finger scoops: quarter-round fillets between the floor and a front-facing wall
  // (local y = 0 is the front).
  const scoops: { y0: number; r: number }[] = [];
  if (spec.scoopRadius) scoops.push({ y0: wall, r: spec.scoopRadius });
  for (const sc of spec.extraScoops ?? []) scoops.push(sc);
  for (const sc of scoops) {
    const r2 = Math.min(sc.r, h - bottom - 0.5, d - wall - sc.y0 - 1);
    if (r2 < 2) continue;
    // The profile is a corner block overlapping floor and wall, minus a circle whose
    // radius is a touch larger, so the fillet CROSSES both faces instead of meeting
    // them tangentially. A tangential meeting would leave coincident vertices and a
    // zero-thickness sliver, which some slicers and mesh checkers flag.
    const rect = s.keep(s.keep(k.CrossSection.square([r2 + FILLET_OVERLAP, r2 + FILLET_OVERLAP])).translate([sc.y0 - FILLET_OVERLAP, bottom - FILLET_OVERLAP]));
    const circle = s.keep(s.keep(k.CrossSection.circle(r2 + FILLET_BITE, CIRCLE_SEGMENTS)).translate([sc.y0 + r2, bottom + r2]));
    const profile = s.keep(rect.subtract(circle));
    const fillet = extrudeAlongX(s, profile, -1, w + 1);
    body = s.keep(body.add(s.keep(fillet.intersect(solid))));
  }

  // Label ledge along the back wall: flat top at the rim, 45° underside so it prints
  // without supports. The profile runs past the rim and through the surrounding walls and
  // is then trimmed by `solid`, so every face of the result comes from one body – ending
  // a feature inside wall material would leave a T-junction with coincident vertices.
  const tab = Math.min(spec.labelTab ?? 0, id / 2, h - bottom - LABEL_TAB_TOP - 1);
  if (tab >= 3) {
    const yEdge = d - wall - tab; // front edge of the ledge
    // The 45° underside runs on through the back wall rather than stopping on its inner
    // face: crossing a face is clean, ending exactly on one leaves a coincident vertex.
    const profile = s.keep(
      k.CrossSection.ofPolygons([
        [
          [yEdge, h + 1],
          [yEdge, h - LABEL_TAB_TOP],
          [d + 1, h - LABEL_TAB_TOP - (d + 1 - yEdge)],
          [d + 1, h + 1],
        ],
      ]),
    );
    const ledge = extrudeAlongX(s, profile, -1, w + 1);
    body = s.keep(body.add(s.keep(ledge.intersect(solid))));
  }
  s.dispose(body);
  return body;
}
