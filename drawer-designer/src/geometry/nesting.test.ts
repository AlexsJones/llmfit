import { describe, expect, it } from 'vitest';
import type { PartMesh } from './types';
import { estimateMinutes, estimateRange, formatRange, nestParts } from './nesting';
import { printerFromPreset } from '../model/defaults';
import { SHELL_THICKNESS, extrudedVolume, filamentGrams } from './parts';

const BV = { x: 256, y: 256, z: 256 };
const part = (id: string, w: number, d: number, h = 40): PartMesh => ({
  id,
  compartmentId: id,
  name: id,
  role: 'bin',
  mesh: { positions: new Float32Array(), indices: new Uint32Array() },
  offset: [0, 0, 0],
  size: [w, d, h],
  volume: w * d * h * 0.1,
  area: 2 * (w * d + w * h + d * h),
  triangles: 0,
  manifold: true,
  warnings: [],
});

const opts = { spacing: 2, allowRotation: true };

/** No two placed parts may overlap, and all must stay on the plate. */
function assertValid(plates: ReturnType<typeof nestParts>['plates'], spacing = 2) {
  for (const pl of plates) {
    for (const p of pl.parts) {
      expect(p.x).toBeGreaterThanOrEqual(-1e-6);
      expect(p.y).toBeGreaterThanOrEqual(-1e-6);
      expect(p.x + p.w).toBeLessThanOrEqual(BV.x + 1e-6);
      expect(p.y + p.d).toBeLessThanOrEqual(BV.y + 1e-6);
    }
    for (let i = 0; i < pl.parts.length; i++) {
      for (let j = i + 1; j < pl.parts.length; j++) {
        const a = pl.parts[i];
        const b = pl.parts[j];
        const gapX = Math.max(a.x - (b.x + b.w), b.x - (a.x + a.w));
        const gapY = Math.max(a.y - (b.y + b.d), b.y - (a.y + a.d));
        // separated on at least one axis, by at least the spacing
        expect(Math.max(gapX, gapY)).toBeGreaterThanOrEqual(spacing - 1e-6);
      }
    }
  }
}

describe('nesting', () => {
  it('packs four quarter-plate parts onto one plate', () => {
    const r = nestParts([part('a', 120, 120), part('b', 120, 120), part('c', 120, 120), part('d', 120, 120)], BV, opts);
    expect(r.plates).toHaveLength(1);
    expect(r.plates[0].parts).toHaveLength(4);
    expect(r.rejected).toEqual([]);
    assertValid(r.plates);
  });

  it('opens more plates when parts do not fit', () => {
    const parts = Array.from({ length: 9 }, (_, i) => part(`p${i}`, 120, 120));
    const r = nestParts(parts, BV, opts);
    expect(r.plates.length).toBe(3);
    expect(r.plates.flatMap((p) => p.parts)).toHaveLength(9);
    assertValid(r.plates);
  });

  it('rotates a part that only fits turned', () => {
    const r = nestParts([part('wide', 250, 60), part('tall', 60, 250)], BV, opts);
    assertValid(r.plates);
    expect(r.rejected).toEqual([]);
    expect(r.plates[0].parts.some((p) => p.rotated)).toBe(true);
  });

  it('rejects parts larger than the plate, keeping the rest', () => {
    const r = nestParts([part('huge', 400, 400), part('tall', 50, 50, 900), part('ok', 50, 50)], BV, opts);
    expect(r.rejected.map((p) => p.id).sort()).toEqual(['huge', 'tall']);
    expect(r.plates).toHaveLength(1);
    expect(r.plates[0].parts.map((p) => p.id)).toEqual(['ok']);
  });

  it('fits a part that exactly fills the plate', () => {
    const r = nestParts([part('exact', 256, 256)], BV, opts);
    expect(r.rejected).toEqual([]);
    expect(r.plates[0].parts).toHaveLength(1);
    expect(r.plates[0].usage).toBeCloseTo(1, 2);
  });

  it('reports usage and height per plate', () => {
    const r = nestParts([part('a', 128, 250, 30), part('b', 60, 60, 90)], BV, opts);
    expect(r.plates[0].height).toBe(90);
    expect(r.plates[0].usage).toBeGreaterThan(0.5);
    expect(r.plates[0].usage).toBeLessThanOrEqual(1);
  });

  it('is deterministic regardless of input order', () => {
    const parts = [part('a', 100, 80), part('b', 150, 120), part('c', 60, 200), part('d', 90, 90)];
    const key = (r: ReturnType<typeof nestParts>) => JSON.stringify(r.plates.map((p) => p.parts.map((q) => [q.id, q.x, q.y, q.rotated])));
    expect(key(nestParts(parts, BV, opts))).toBe(key(nestParts([...parts].reverse(), BV, opts)));
  });

  it('honours the spacing setting', () => {
    const r = nestParts([part('a', 100, 100), part('b', 100, 100)], BV, { spacing: 10, allowRotation: true });
    assertValid(r.plates, 10);
  });
});

describe('material estimate', () => {
  it('splits a part into shell and infilled core', () => {
    // a 100 mm cube: 60000 mm² surface, 1e6 mm³ volume
    const cube = { volume: 1e6, area: 60000 };
    const shell = 60000 * SHELL_THICKNESS;
    expect(extrudedVolume(cube, 0)).toBeCloseTo(shell, 0);
    expect(extrudedVolume(cube, 1)).toBeCloseTo(1e6, 0);
    expect(extrudedVolume(cube, 0.15)).toBeCloseTo(shell + (1e6 - shell) * 0.15, 0);
    // a thin-walled part is all shell, so infill barely matters
    const thin = { volume: 20000, area: 40000 };
    expect(extrudedVolume(thin, 0)).toBeCloseTo(20000, 0);
    expect(extrudedVolume(thin, 1)).toBeCloseTo(20000, 0);
  });

  it('weighs a chunky insert far less than its solid volume', () => {
    const block = { volume: 300_000, area: 50_000 };
    expect(filamentGrams(block, 0.15)).toBeLessThan(filamentGrams(block, 1) * 0.6);
  });
});

describe('print time', () => {
  const prusa = printerFromPreset('prusa-mk4s');

  it('scales with volume and adds startup', () => {
    const small = estimateMinutes(10_000, 20, prusa);
    const big = estimateMinutes(100_000, 20, prusa);
    expect(small).toBeGreaterThan(prusa.startupMinutes);
    expect(big).toBeGreaterThan(small * 5);
    expect(estimateMinutes(0, 0, prusa)).toBeCloseTo(prusa.startupMinutes, 1);
  });

  it('lands in a plausible range for a typical bin', () => {
    // 130 x 150 x 48 bin with 1.6 mm walls ≈ 60 cm³ of plastic
    const minutes = estimateMinutes(60_000, 48, prusa);
    expect(minutes).toBeGreaterThan(60);
    expect(minutes).toBeLessThan(900);
  });

  it('a faster machine with the same job prints quicker', () => {
    const fast = printerFromPreset('bambu-x2d');
    expect(estimateMinutes(60_000, 48, fast)).toBeLessThan(estimateMinutes(60_000, 48, prusa));
  });

  it('a wider nozzle lays down plastic faster', () => {
    const wide = { ...prusa, nozzle: 0.6 };
    expect(estimateMinutes(60_000, 48, wide)).toBeLessThan(estimateMinutes(60_000, 48, prusa));
  });

  it('reports a range rather than a falsely precise figure', () => {
    const r = estimateRange(60_000, 48, prusa);
    expect(r.low).toBeLessThan(r.mid);
    expect(r.high).toBeGreaterThan(r.mid);
    expect(formatRange(r)).toMatch(/\d+(\.\d)?–\d+(\.\d)? h/);
    expect(formatRange(estimateRange(1000, 5, prusa))).toMatch(/min$/);
    expect(formatRange({ low: 0, high: 0, mid: 0 })).toBe('–');
  });
});
