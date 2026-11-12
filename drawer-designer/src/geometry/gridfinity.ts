/**
 * Gridfinity geometry.
 *
 * Gridfinity is Zack Freedman's open storage standard: a 42 mm grid, bins that
 * drop into baseplates, and a stepped foot that locates them. Building to the
 * standard means parts from here mix with anything else in the ecosystem.
 *
 * Figures taken from the unofficial specification's block-profile diagram
 * (41.5 mm square per unit, 4.75 mm foot, 3.75 mm top corner radius) and the
 * widely used 0.8 / 1.8 / 2.15 mm foot breakdown. Anything the specification
 * leaves open is marked below.
 */
import type { CrossSection, Manifold } from 'manifold-3d';
import { CIRCLE_SEGMENTS, Scope, roundedRect } from './bin';
import type { Kernel } from './kernel';

/** Grid pitch (mm). */
export const PITCH = 42;
/** Height unit (mm). A bin's total height, foot included, is a multiple of this. */
export const UNIT_H = 7;
/** Total clearance between a bin and its baseplate socket, split across both sides (mm). */
export const CLEARANCE = 0.5;
/** Footprint of an n-unit bin (mm). */
export const binFootprint = (units: number) => PITCH * units - CLEARANCE;

/** Outer corner radius at the widest point of the foot (mm). */
export const R_TOP = 3.75;

/**
 * Foot profile from the bottom up: each step is a height and the horizontal
 * inset it adds. A 45° chamfer insets by the same amount as its height.
 */
export const FOOT = [
  { h: 0.8, inset: 0.8 }, // bottom chamfer
  { h: 1.8, inset: 0 }, // vertical
  { h: 2.15, inset: 2.15 }, // upper chamfer
] as const;

/** Total foot height (mm). */
export const FOOT_H = FOOT.reduce((s, f) => s + f.h, 0);
/** Total inset at the very bottom of the foot (mm). */
export const FOOT_INSET = FOOT.reduce((s, f) => s + f.inset, 0);

/**
 * Stacking lip height (mm). The specification's text does not dimension the lip,
 * so this mirrors the foot profile, which is what the common implementations do.
 */
export const LIP_H = FOOT_H;

/** Magnet pocket, as given in the specification text. */
export const MAGNET = { diameter: 6, depth: 2, inset: 8 / Math.SQRT2 } as const;

export interface GridfinitySpec {
  /** Grid units along X. */
  unitsX: number;
  /** Grid units along Y. */
  unitsY: number;
  /** Height in 7 mm units, foot included. */
  unitsZ: number;
  wall: number;
  floor: number;
  /** Add the stacking lip so other bins sit on top. */
  lip: boolean;
  /** Cut 6 × 2 mm magnet pockets into the foot. */
  magnets: boolean;
  /** Dividers inside the bin, as fractions of the inner width/depth. */
  dividersX: number;
  dividersY: number;
  /** Scoop radius along the front wall (0 = none). */
  scoop: number;
}

export function gridfinityHeight(unitsZ: number): number {
  return unitsZ * UNIT_H;
}

/** How many whole grid units fit in a span of `mm`. */
export function unitsIn(mm: number): number {
  return Math.max(0, Math.floor((mm + CLEARANCE) / PITCH));
}

/**
 * Cross-section of the footprint at a given height within the foot, as the
 * profile steps outward. Above the foot the full footprint is used.
 */
function profileAt(k: Kernel, s: Scope, w: number, d: number, inset: number): CrossSection {
  const r = Math.max(0.1, R_TOP - inset);
  return roundedRect(k, s, w - 2 * inset, d - 2 * inset, r).translate([inset, inset]);
}

/**
 * The stepped foot as a stack of extruded sections. Each section overlaps the one
 * below by a hair so the union fuses into one solid rather than touching faces.
 */
export function buildFoot(k: Kernel, s: Scope, w: number, d: number): Manifold {
  const parts: Manifold[] = [];
  let z = 0;
  let inset = FOOT_INSET;
  for (const step of FOOT) {
    const bottom = profileAt(k, s, w, d, inset);
    const top = profileAt(k, s, w, d, inset - step.inset);
    // A chamfer is a loft between two outlines; a straight step is a plain extrusion.
    const section = step.inset > 0 ? s.keep(loftOutlines(s, bottom, top, step.h)) : s.keep(bottom.extrude(step.h));
    parts.push(s.keep(section.translate([0, 0, z])));
    z += step.h;
    inset -= step.inset;
  }
  return s.keep(k.Manifold.union(parts));
}

/**
 * Loft between two concentric rounded rectangles by extruding the lower one and
 * scaling the top face. `scaleTop` is uniform per axis, which is exact here because
 * both outlines are concentric rounded rectangles with matching radius offsets.
 */
function loftOutlines(s: Scope, bottom: CrossSection, top: CrossSection, height: number): Manifold {
  const bb = bottom.bounds();
  const tb = top.bounds();
  const bw = bb.max[0] - bb.min[0];
  const bd = bb.max[1] - bb.min[1];
  const tw = tb.max[0] - tb.min[0];
  const td = tb.max[1] - tb.min[1];
  const centred = s.keep(bottom.translate([-(bb.min[0] + bw / 2), -(bb.min[1] + bd / 2)]));
  const solid = s.keep(centred.extrude(height, 1, 0, [tw / bw, td / bd]));
  return s.keep(solid.translate([bb.min[0] + bw / 2, bb.min[1] + bd / 2, 0]));
}

/**
 * A complete Gridfinity bin, standing on the origin with its foot at z = 0.
 * Ownership follows `buildBin`: the scope is freed except for the returned solid,
 * which the caller deletes.
 */
export function buildGridfinityBin(k: Kernel, spec: GridfinitySpec, s: Scope = new Scope()): Manifold {
  const w = binFootprint(spec.unitsX);
  const d = binFootprint(spec.unitsY);
  const h = gridfinityHeight(spec.unitsZ);
  const bodyTop = h;

  // Foot, then the straight body above it.
  const foot = buildFoot(k, s, w, d);
  const outline = s.keep(roundedRect(k, s, w, d, R_TOP));
  const body = s.keep(s.keep(outline.extrude(bodyTop - FOOT_H)).translate([0, 0, FOOT_H]));
  let solid = s.keep(foot.add(body));

  // Pocket.
  const wall = spec.wall;
  const floorZ = Math.max(FOOT_H + 0.4, spec.floor + FOOT_H);
  const iw = w - 2 * wall;
  const id = d - 2 * wall;
  if (iw > 2 && id > 2 && bodyTop - floorZ > 0.4) {
    const inner = s.keep(roundedRect(k, s, iw, id, Math.max(R_TOP - wall, 0.5)).translate([wall, wall]));
    const pocket = s.keep(s.keep(inner.extrude(bodyTop - floorZ + 1)).translate([0, 0, floorZ]));
    solid = s.keep(solid.subtract(pocket));
  }

  // Internal dividers.
  const slabs: Manifold[] = [];
  const t = Math.max(0.8, wall);
  for (let i = 1; i <= spec.dividersX; i++) {
    const x = wall + (iw * i) / (spec.dividersX + 1);
    slabs.push(s.keep(s.keep(k.Manifold.cube([t, d + 2, bodyTop - floorZ])).translate([x - t / 2, -1, floorZ])));
  }
  for (let i = 1; i <= spec.dividersY; i++) {
    const y = wall + (id * i) / (spec.dividersY + 1);
    slabs.push(s.keep(s.keep(k.Manifold.cube([w + 2, t, bodyTop - floorZ])).translate([-1, y - t / 2, floorZ])));
  }
  if (slabs.length) {
    const walls = s.keep(k.Manifold.union(slabs));
    solid = s.keep(solid.add(s.keep(walls.intersect(solid))));
  }

  // Stacking lip: the foot profile again, cut into the rim so the bin above drops in.
  if (spec.lip && bodyTop - floorZ > LIP_H + 1) {
    const socket = buildSocket(k, s, w, d, LIP_H + 0.2);
    solid = s.keep(solid.subtract(s.keep(socket.translate([0, 0, bodyTop - LIP_H]))));
  }

  // Magnet pockets, one per grid unit corner.
  if (spec.magnets) {
    const holes: Manifold[] = [];
    for (let ux = 0; ux < spec.unitsX; ux++) {
      for (let uy = 0; uy < spec.unitsY; uy++) {
        const cx = ux * PITCH + PITCH / 2;
        const cy = uy * PITCH + PITCH / 2;
        for (const [sx, sy] of [
          [-1, -1],
          [1, -1],
          [-1, 1],
          [1, 1],
        ]) {
          const hole = s.keep(k.Manifold.cylinder(MAGNET.depth + 0.1, MAGNET.diameter / 2, MAGNET.diameter / 2, 48));
          holes.push(s.keep(hole.translate([cx + sx * MAGNET.inset, cy + sy * MAGNET.inset, -0.05])));
        }
      }
    }
    if (holes.length) solid = s.keep(solid.subtract(s.keep(k.Manifold.union(holes))));
  }

  // Finger scoop along the front wall (local y = 0).
  if (spec.scoop >= 2) {
    const r = Math.min(spec.scoop, bodyTop - floorZ - 0.5, id - 1);
    if (r >= 2) {
      const rect = s.keep(s.keep(k.CrossSection.square([r + 0.5, r + 0.5])).translate([wall - 0.5, floorZ - 0.5]));
      const circle = s.keep(s.keep(k.CrossSection.circle(r + 0.1, CIRCLE_SEGMENTS)).translate([wall + r, floorZ + r]));
      const profile = s.keep(rect.subtract(circle));
      const fillet = s.keep(
        s.keep(profile.extrude(w + 2)).transform([0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, -1, 0, 0, 1]),
      );
      solid = s.keep(solid.add(s.keep(fillet.intersect(solid))));
    }
  }
  s.dispose(solid);
  return solid;
}

/** The socket a bin's foot drops into: the foot profile grown by the fit clearance. */
export function buildSocket(k: Kernel, s: Scope, w: number, d: number, height: number): Manifold {
  const parts: Manifold[] = [];
  let z = 0;
  let inset = FOOT_INSET;
  for (const step of FOOT) {
    const bottom = profileAt(k, s, w, d, inset);
    const top = profileAt(k, s, w, d, inset - step.inset);
    const section = step.inset > 0 ? s.keep(loftOutlines(s, bottom, top, step.h)) : s.keep(bottom.extrude(step.h));
    parts.push(s.keep(section.translate([0, 0, z])));
    z += step.h;
    inset -= step.inset;
  }
  // Straight continuation above the profile, so a lip cut reaches the rim.
  if (height > FOOT_H) {
    const topOutline = profileAt(k, s, w, d, 0);
    parts.push(s.keep(s.keep(topOutline.extrude(height - FOOT_H)).translate([0, 0, FOOT_H])));
  }
  return s.keep(k.Manifold.union(parts));
}

export interface BaseplateSpec {
  unitsX: number;
  unitsY: number;
  /** Solid material under the sockets (mm); 0 gives a frame with open bottoms. */
  floor: number;
}

/** A Gridfinity baseplate: a slab with one socket per grid cell. Caller owns the result. */
export function buildBaseplate(k: Kernel, spec: BaseplateSpec, s: Scope = new Scope()): Manifold {
  const w = spec.unitsX * PITCH;
  const d = spec.unitsY * PITCH;
  const h = FOOT_H + spec.floor;
  const slab = s.keep(s.keep(roundedRect(k, s, w, d, R_TOP + 0.5)).extrude(h));
  const sockets: Manifold[] = [];
  for (let ux = 0; ux < spec.unitsX; ux++) {
    for (let uy = 0; uy < spec.unitsY; uy++) {
      // Sockets are the bin footprint plus clearance, centred in their cell.
      const sw = PITCH - CLEARANCE;
      const socket = buildSocket(k, s, sw, sw, FOOT_H + 1);
      sockets.push(s.keep(socket.translate([ux * PITCH + CLEARANCE / 2, uy * PITCH + CLEARANCE / 2, spec.floor])));
    }
  }
  const plate = s.keep(slab.subtract(s.keep(k.Manifold.union(sockets))));
  s.dispose(plate);
  return plate;
}
