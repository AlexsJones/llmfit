import { describe, expect, it } from 'vitest';
import type { Compartment } from '../model/types';
import {
  boundsOf,
  corners,
  distributeRun,
  expandToFill,
  findFreeSpot,
  gridRects,
  mergeBox,
  moveGroup,
  overlaps,
  resizeEdge,
  inBounds,
  snapSpan,
  spanRect,
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

  it('merges only selections that form a full rectangle', () => {
    const a = { x: 0, y: 0, w: 50, d: 50 };
    const b = { x: 50, y: 0, w: 50, d: 50 };
    const c = { x: 0, y: 50, w: 50, d: 50 };
    expect(mergeBox([a, b], [])).toEqual({ x: 0, y: 0, w: 100, d: 50 });
    // L-shape leaves a hole in the bounding box
    expect(mergeBox([a, b, c], [])).toBeNull();
    // something else sits inside the box
    expect(mergeBox([a, { x: 100, y: 0, w: 50, d: 50 }], [{ x: 50, y: 0, w: 50, d: 50 }])).toBeNull();
    expect(mergeBox([a], [])).toBeNull();
  });

  it('moves a group together and stops at obstacles', () => {
    const group = [
      { x: 0, y: 0, w: 50, d: 50 },
      { x: 50, y: 0, w: 50, d: 50 },
    ];
    expect(moveGroup(group, 20, 10, [], B)).toEqual({ dx: 20, dy: 10 });
    // clamped at the right wall (group is 100 wide, bounds 300)
    expect(moveGroup(group, 500, 0, [], B).dx).toBe(200);
    // blocked in x by a neighbour, still free in y
    expect(moveGroup(group, 40, 30, [{ x: 120, y: 0, w: 10, d: 50 }], B)).toEqual({ dx: 0, dy: 30 });
  });

  it('spans a normalised rectangle between two points', () => {
    expect(spanRect(80, 60, 20, 10)).toEqual({ x: 20, y: 10, w: 60, d: 50 });
  });

  it('detects overlaps and out-of-bounds compartments', () => {
    const issues = validateLayout([comp({ id: 'a' }), comp({ id: 'b', x: 20 }), comp({ id: 'c', x: 280 })], B);
    expect(issues.map((i) => i.compartmentId).sort()).toEqual(['b', 'c']);
  });
});

describe('turned compartments', () => {
  it('rotates the corners about the centre', () => {
    const square = corners({ x: 10, y: 10, w: 20, d: 20, rot: 90 });
    // A square turned a quarter turn covers exactly the same ground.
    expect(square.map((p) => [Math.round(p.x), Math.round(p.y)]).sort()).toEqual([
      [10, 10],
      [10, 30],
      [30, 10],
      [30, 30],
    ]);
    const box = boundsOf({ x: 0, y: 0, w: 40, d: 10, rot: 90 });
    expect(box.w).toBeCloseTo(10, 6);
    expect(box.d).toBeCloseTo(40, 6);
    expect(box.x).toBeCloseTo(15, 6);
    expect(box.y).toBeCloseTo(-15, 6);
  });

  it('sees overlaps that only the turned footprint has', () => {
    const flat = { x: 0, y: 0, w: 60, d: 10 };
    const above = { x: 20, y: 20, w: 20, d: 20 };
    expect(overlaps(flat, above)).toBe(false);
    // Stood on end about its centre, the same bar reaches up into the other one.
    expect(overlaps({ ...flat, rot: 90 }, above)).toBe(true);
    // Two bars crossing at a right angle overlap even though neither contains a corner of the other.
    expect(overlaps({ x: 0, y: 45, w: 100, d: 10 }, { x: 45, y: 0, w: 10, d: 100 })).toBe(true);
    // Turning by a whole turn changes nothing.
    expect(overlaps({ ...flat, rot: 360 }, above)).toBe(false);
  });

  it('keeps a turned rectangle inside the drawer', () => {
    const bar = { x: 0, y: 95, w: 300, d: 10 };
    expect(inBounds(bar, B)).toBe(true);
    // Turned 45° about its centre the ends swing far outside.
    expect(inBounds({ ...bar, rot: 45 }, B)).toBe(false);
    expect(inBounds({ x: 100, y: 80, w: 40, d: 40, rot: 30 }, B)).toBe(true);
  });

  it('moves a turned rectangle without sliding it through a neighbour', () => {
    const other = { x: 120, y: 60, w: 40, d: 40 };
    const bar = { x: 20, y: 60, w: 60, d: 20, rot: 30 };
    // A clear move is taken as asked.
    const ok = sweepMove(bar, 40, 60, [other], B);
    expect(ok.x).toBeCloseTo(40, 6);
    expect(ok.rot).toBe(30);
    // A move onto the neighbour is refused rather than slid along it.
    const blocked = sweepMove(bar, 118, 60, [other], B);
    expect(overlaps(blocked, other)).toBe(false);
    expect(inBounds(blocked, B)).toBe(true);
  });

  it('merging still refuses anything turned', () => {
    // mergeBox works on footprints; two halves of a rectangle only tile when both are square to the grid.
    expect(mergeBox([{ x: 0, y: 0, w: 50, d: 50 }, { x: 50, y: 0, w: 50, d: 50 }], [])).not.toBeNull();
    expect(mergeBox([{ x: 0, y: 0, w: 50, d: 50, rot: 20 }, { x: 50, y: 0, w: 50, d: 50 }], [])).toBeNull();
  });

  it('stops a growing edge at a turned neighbour, not at its unturned box', () => {
    // An 80 mm square turned 30° about its centre spans -14.64 to 94.64 mm, past its own x + w.
    const turned = { x: 0, y: 0, w: 80, d: 80, rot: 30 };
    const spot = findFreeSpot(80, 80, [turned], B)!;
    expect(spot).not.toBeNull();
    expect(overlaps(spot, turned)).toBe(false);
    expect(spot.x).toBeGreaterThanOrEqual(boundsOf(turned).x + boundsOf(turned).w - 0.01);
    expect(expandToFill({ x: 150, y: 0, w: 20, d: 20 }, [turned], B).x).toBeCloseTo(94.64, 2);
  });

  it('never places or grows a compartment into a turned one', () => {
    const turned = { x: 14.64, y: 14.64, w: 80, d: 80, rot: 30 };
    const spot = findFreeSpot(80, 80, [turned], B)!;
    expect(overlaps(spot, turned)).toBe(false);
    // The corner at (40, 0) means the top-left seed must not simply grow over it.
    const grown = resizeEdge({ x: 0, y: 0, w: 20, d: 20 }, 'right', 300, [turned], B);
    expect(overlaps(grown, turned)).toBe(false);
  });
});
