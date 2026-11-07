import { beforeAll, describe, expect, it } from 'vitest';
import { Scope } from './bin';
import {
  CLEARANCE,
  FOOT_H,
  PITCH,
  R_TOP,
  UNIT_H,
  binFootprint,
  buildBaseplate,
  buildGridfinityBin,
  gridfinityHeight,
  unitsIn,
} from './gridfinity';
import { type Kernel, loadKernel } from './kernel';
import { checkMesh } from './meshCheck';
import type { MeshData } from './types';

let k: Kernel;
beforeAll(async () => {
  k = await loadKernel();
});

function meshOf(m: ReturnType<typeof buildGridfinityBin>): MeshData {
  const raw = m.getMesh();
  const n = raw.vertProperties.length / raw.numProp;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = raw.vertProperties[i * raw.numProp];
    positions[i * 3 + 1] = raw.vertProperties[i * raw.numProp + 1];
    positions[i * 3 + 2] = raw.vertProperties[i * raw.numProp + 2];
  }
  return { positions, indices: new Uint32Array(raw.triVerts) };
}

const spec = (over: Partial<Parameters<typeof buildGridfinityBin>[1]> = {}) => ({
  unitsX: 1,
  unitsY: 1,
  unitsZ: 6,
  wall: 1.2,
  floor: 1.2,
  lip: false,
  magnets: false,
  dividersX: 0,
  dividersY: 0,
  scoop: 0,
  ...over,
});

describe('gridfinity constants', () => {
  it('matches the published standard', () => {
    expect(PITCH).toBe(42);
    expect(UNIT_H).toBe(7);
    expect(CLEARANCE).toBe(0.5);
    expect(R_TOP).toBe(3.75);
    expect(FOOT_H).toBeCloseTo(4.75, 5);
    expect(binFootprint(1)).toBe(41.5);
    expect(binFootprint(2)).toBe(83.5);
    expect(gridfinityHeight(6)).toBe(42);
  });

  it('counts whole units that fit a span', () => {
    expect(unitsIn(41.5)).toBe(1);
    expect(unitsIn(84)).toBe(2);
    expect(unitsIn(41)).toBe(0);
    expect(unitsIn(250)).toBe(5);
  });
});

describe('gridfinity bin', () => {
  it('builds a watertight 1x1 bin with standard outer dimensions', () => {
    const s = new Scope();
    const m = buildGridfinityBin(k, spec(), s);
    const bb = m.boundingBox();
    expect(m.status()).toBe('NoError');
    expect(bb.max[0] - bb.min[0]).toBeCloseTo(41.5, 2);
    expect(bb.max[1] - bb.min[1]).toBeCloseTo(41.5, 2);
    expect(bb.max[2] - bb.min[2]).toBeCloseTo(42, 2);
    expect(bb.min[2]).toBeCloseTo(0, 5);
    const rep = checkMesh(meshOf(m));
    expect(rep.watertight).toBe(true);
    expect(rep.coincident).toBe(0);
    m.delete();
  });

  it('tapers towards the bottom following the foot profile', () => {
    const s = new Scope();
    const m = buildGridfinityBin(k, spec(), s);
    const mesh = meshOf(m);
    // Width of the solid at the very bottom must be the fully inset footprint.
    let minX = Infinity;
    let maxX = -Infinity;
    for (let i = 0; i < mesh.positions.length; i += 3) {
      if (Math.abs(mesh.positions[i + 2]) < 1e-4) {
        minX = Math.min(minX, mesh.positions[i]);
        maxX = Math.max(maxX, mesh.positions[i]);
      }
    }
    // 41.5 - 2 x (0.8 + 2.15) = 35.6
    expect(maxX - minX).toBeCloseTo(35.6, 1);
    m.delete();
  });

  it('scales to multi-unit footprints', () => {
    const s = new Scope();
    const m = buildGridfinityBin(k, spec({ unitsX: 3, unitsY: 2, unitsZ: 3 }), s);
    const bb = m.boundingBox();
    expect(bb.max[0] - bb.min[0]).toBeCloseTo(125.5, 2); // 3 x 42 - 0.5
    expect(bb.max[1] - bb.min[1]).toBeCloseTo(83.5, 2);
    expect(bb.max[2] - bb.min[2]).toBeCloseTo(21, 2);
    expect(checkMesh(meshOf(m)).watertight).toBe(true);
    m.delete();
  });

  it('stays watertight with lip, magnets, dividers and a scoop', () => {
    const s = new Scope();
    const m = buildGridfinityBin(k, spec({ unitsX: 2, lip: true, magnets: true, dividersX: 1, dividersY: 1, scoop: 8 }), s);
    const rep = checkMesh(meshOf(m));
    expect(m.status()).toBe('NoError');
    expect(rep.watertight).toBe(true);
    expect(rep.coincident).toBe(0);
    // magnets and the lip remove material compared with a plain bin
    const plain = buildGridfinityBin(k, spec({ unitsX: 2 }), s);
    expect(m.volume()).toBeLessThan(plain.volume());
    m.delete();
    plain.delete();
  });
});

describe('gridfinity baseplate', () => {
  it('builds a watertight plate whose sockets accept a bin foot', () => {
    const s = new Scope();
    const plate = buildBaseplate(k, { unitsX: 2, unitsY: 2, floor: 1 }, s);
    const bb = plate.boundingBox();
    expect(bb.max[0] - bb.min[0]).toBeCloseTo(84, 2);
    expect(bb.max[2] - bb.min[2]).toBeCloseTo(FOOT_H + 1, 2);
    expect(plate.status()).toBe('NoError');
    expect(checkMesh(meshOf(plate)).watertight).toBe(true);
    // The sockets are voids, so the plate weighs less than a solid slab of the same size.
    expect(plate.volume()).toBeLessThan(84 * 84 * (FOOT_H + 1));
    plate.delete();
  });

  it('leaves a bin foot clear of the socket walls', () => {
    const s = new Scope();
    const plate = buildBaseplate(k, { unitsX: 1, unitsY: 1, floor: 1 }, s);
    const bin = buildGridfinityBin(k, spec(), s);
    // Drop the bin into the plate: at the correct offset they must not intersect.
    const placed = bin.translate([CLEARANCE / 2, CLEARANCE / 2, 1]);
    const clash = plate.intersect(placed);
    expect(clash.volume()).toBeLessThan(1); // mm³ – numerical noise only
    clash.delete();
    placed.delete();
    plate.delete();
    bin.delete();
  });
});
