import { describe, expect, it } from 'vitest';
import type { Compartment } from '../model/types';
import {
  distributeRun,
  expandToFill,
  findFreeSpot,
  gridRects,
  overlaps,
  resizeEdge,
  snapSpan,
  splitCompartment,
  splitToFit,
  sweepMove,
  validateLayout,
} from './ops';

const B = { width: 300, depth: 200 };
const comp = (over: Partial<Compartment>): Compartment => ({
  id: 'c', kind: 'bin', name: 'C', x: 0, y: 0, w: 50, d: 50, height: null, dividers: [], params: {}, ...over,
});

describe('layout ops', () => {
  it('slides a moved rectangle up against neighbours instead of overlapping', () => {
    const other = { x: 100, y: 0, w: 50, d: 50 };
    const r = sweepMove({ x: 0, y: 0, w: 50, d: 50 }, 120, 0, [other], B);
    expect(r.x).toBe(50);
    expect(overlaps(r, other)).toBe(false);
  });

  it('keeps moves inside the drawer bounds', () => {
    const r = sweepMove({ x: 10, y: 10, w: 50, d: 50 }, 1000, -40, [], B);
    expect(r).toEqual({ x: 250, y: 0, w: 50, d: 50 });
  });

  it('limits resizing at neighbours and minimum size', () => {
    const r = resizeEdge({ x: 0, y: 0, w: 50, d: 50 }, 'right', 500, [{ x: 120, y: 20, w: 10, d: 10 }], B);
    expect(r.w).toBe(120);
    const tiny = resizeEdge({ x: 0, y: 0, w: 50, d: 50 }, 'right', 1, [], B, 12);
    expect(tiny.w).toBe(12);
  });

  it('expands to fill the free region', () => {
    const r = expandToFill({ x: 100, y: 50, w: 12, d: 12 }, [{ x: 0, y: 0, w: 60, d: 200 }], B);
    expect(r).toEqual({ x: 60, y: 0, w: 240, d: 200 });
  });

  it('generates grids that tile the area exactly', () => {
    const rects = gridRects(3, 4, B);
    expect(rects).toHaveLength(12);
    const area = rects.reduce((s, r) => s + r.w * r.d, 0);
    expect(area).toBeCloseTo(300 * 200, 0);
    expect(validateLayout(rects.map((r, i) => comp({ ...r, id: `c${i}` })), B)).toEqual([]);
  });

  it('finds free space next to existing compartments', () => {
    const spot = findFreeSpot(80, 80, [{ x: 0, y: 0, w: 100, d: 200 }], B);
    expect(spot).toEqual({ x: 100, y: 0, w: 80, d: 80 });
    expect(findFreeSpot(20, 20, [{ x: 0, y: 0, w: 300, d: 200 }], B)).toBeNull();
  });

  it('splits compartments and carries dividers into the right piece', () => {
    const c = comp({
      w: 120,
      dividers: [
        { id: 'v', orientation: 'vertical', position: 90 },
        { id: 'h', orientation: 'horizontal', position: 25 },
      ],
    });
    const [a, b] = splitCompartment(c, 'vertical', 2);
    expect(a.w + b.w).toBeCloseTo(120);
    expect(b.x).toBe(60);
    expect(a.dividers.map((d) => d.orientation)).toEqual(['horizontal']);
    expect(b.dividers.find((d) => d.orientation === 'vertical')?.position).toBe(30);
  });

  it('splits oversized compartments into printable pieces', () => {
    const pieces = splitToFit(comp({ w: 600, d: 200 }), 256, 256, 0.5);
    expect(pieces).toHaveLength(3);
    for (const p of pieces) expect(p.w - 0.5).toBeLessThanOrEqual(256);
  });

  it('distributes a row evenly', () => {
    const cs = [comp({ id: 'a', w: 30 }), comp({ id: 'b', x: 30, w: 90 }), comp({ id: 'c', x: 120, w: 60 })];
    const out = distributeRun(cs, 'b', 'x');
    expect(out.map((c) => c.w)).toEqual([60, 60, 60]);
    expect(out.map((c) => c.x)).toEqual([0, 60, 120]);
  });

  it('snaps edges to nearby lines before the grid', () => {
    expect(snapSpan(47, 50, [100], 5, 10)).toEqual({ value: 50, guide: 100 });
    expect(snapSpan(23, 50, [], 5, 10)).toEqual({ value: 20, guide: null });
  });

  it('detects overlaps and out-of-bounds compartments', () => {
    const issues = validateLayout([comp({ id: 'a' }), comp({ id: 'b', x: 20 }), comp({ id: 'c', x: 280 })], B);
    expect(issues.map((i) => i.compartmentId).sort()).toEqual(['b', 'c']);
  });
});
