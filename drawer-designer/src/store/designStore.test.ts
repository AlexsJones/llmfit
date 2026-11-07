import { beforeEach, describe, expect, it } from 'vitest';
import { useDesign } from './designStore';

const s = () => useDesign.getState();

describe('design store', () => {
  beforeEach(() => {
    s().newDesign({ width: 300, depth: 200, height: 40, clearance: 1 }, {}, { kind: 'grid', rows: 2, cols: 2 });
  });

  it('creates a design with the usable area tiled', () => {
    const cs = s().design.compartments;
    expect(cs).toHaveLength(4);
    expect(cs[3].x + cs[3].w).toBeCloseTo(298);
    expect(cs[3].y + cs[3].d).toBeCloseTo(198);
    expect(s().design.print.binHeight).toBe(38);
  });

  it('rejects overlapping rectangles and supports undo/redo', () => {
    const [a] = s().design.compartments;
    expect(s().setCompartmentRect(a.id, { x: 0, y: 0, w: 200, d: 99 })).toBe(false);
    s().deleteCompartment(s().design.compartments[1].id);
    expect(s().setCompartmentRect(a.id, { x: 0, y: 0, w: 298, d: 99 })).toBe(true);
    expect(s().design.compartments).toHaveLength(3);
    s().undo();
    s().undo();
    expect(s().design.compartments).toHaveLength(4);
    s().redo();
    expect(s().design.compartments).toHaveLength(3);
  });

  it('groups transient drag updates into one undo step', () => {
    const [a] = s().design.compartments;
    s().deleteCompartment(s().design.compartments[1].id);
    const before = s().past.length;
    s().beginGesture();
    for (let w = 160; w <= 290; w += 10) s().setCompartmentRect(a.id, { x: 0, y: 0, w, d: 99 }, true);
    s().endGesture();
    expect(s().past.length).toBe(before + 1);
    s().undo();
    expect(s().design.compartments[0].w).toBeCloseTo(149);
  });

  it('shrinks the layout when the drawer gets smaller', () => {
    s().updateDrawer({ width: 200 });
    for (const c of s().design.compartments) expect(c.x + c.w).toBeLessThanOrEqual(198 + 1e-6);
  });

  it('merges a row into one compartment and keeps the rest', () => {
    const cs = s().design.compartments;
    const id = s().mergeCompartments([cs[0].id, cs[1].id]);
    expect(id).toBeTruthy();
    const after = s().design.compartments;
    expect(after).toHaveLength(3);
    const merged = after.find((c) => c.id === id)!;
    expect(merged.w).toBeCloseTo(298);
    expect(merged.d).toBeCloseTo(99);
    // the merged row plus one cell below is an L-shape, so it is refused
    const row = after.find((c) => c.id === id)!;
    const below = after.filter((c) => c.id !== id);
    expect(s().mergeCompartments([row.id, below[0].id])).toBeNull();
    // all three together tile the drawer again, so that merge is allowed
    expect(s().mergeCompartments(after.map((c) => c.id))).toBeTruthy();
  });

  it('moves a group without letting it overlap or leave the drawer', () => {
    const cs = s().design.compartments;
    const ids = [cs[0].id, cs[2].id]; // left column
    s().moveMany(ids, 5, 0);
    const after = s().design.compartments;
    // blocked by the right column, so nothing moved
    expect(after[0].x).toBe(cs[0].x);
    s().deleteMany([cs[1].id, cs[3].id]);
    s().moveMany(ids, 500, 0);
    const moved = s().design.compartments;
    expect(moved[0].x + moved[0].w).toBeCloseTo(298);
    expect(moved[1].x).toBe(moved[0].x);
  });

  it('carries the chosen printer into the design and keeps walls clean', () => {
    s().newDesign({ width: 300, depth: 200, height: 40, clearance: 1 }, {}, { kind: 'empty' }, 'Mini', 'bambu-a1-mini');
    expect(s().design.printer.presetId).toBe('bambu-a1-mini');
    expect(s().design.printer.build).toEqual({ x: 180, y: 180, z: 180 });
    expect(s().design.print.wallThickness).toBeCloseTo(1.68, 2);
    // a wider nozzle retunes the default wall, but a hand-set wall is left alone
    s().selectPrinter('bambu-x2d');
    expect(s().design.printer.presetId).toBe('bambu-x2d');
    s().updatePrinter({ nozzle: 0.6 });
    expect(s().design.printer.presetId).toBe('custom');
    s().updatePrint({ wallThickness: 2.5 });
    s().selectPrinter('bambu-a1-mini');
    expect(s().design.print.wallThickness).toBe(2.5);
  });

  it('adds dividers into the largest gap and clamps their position', () => {
    const [a] = s().design.compartments;
    const id1 = s().addDivider(a.id, 'vertical');
    const id2 = s().addDivider(a.id, 'vertical');
    const ds = s().design.compartments[0].dividers;
    expect(ds.map((d) => d.position)).toEqual([74.5, 37.25]);
    s().updateDivider(a.id, id2!, -50);
    expect(s().design.compartments[0].dividers.find((d) => d.id === id2)!.position).toBeGreaterThan(2);
    expect(id1).toBeTruthy();
  });
});

describe('clipboard and lanes', () => {
  beforeEach(() => {
    s().newDesign({ width: 300, depth: 200, height: 40, clearance: 1 }, {}, { kind: 'grid', rows: 2, cols: 2 });
  });

  it('pastes a copy of a group, keeping its shape', () => {
    const cs = s().design.compartments;
    // free the bottom row so the pasted pair has somewhere to go
    s().deleteMany([cs[2].id, cs[3].id]);
    const items = s().design.compartments.map((c) => ({ ...c }));
    const ids = s().pasteCompartments(items);
    expect(ids).toHaveLength(2);
    const pasted = s().design.compartments.filter((c) => ids.includes(c.id));
    // same relative offset between the two as in the original
    expect(pasted[1].x - pasted[0].x).toBeCloseTo(items[1].x - items[0].x, 5);
    expect(s().design.compartments).toHaveLength(4);
  });

  it('refuses to paste when the drawer is full', () => {
    const items = s().design.compartments.map((c) => ({ ...c }));
    expect(s().pasteCompartments(items)).toEqual([]);
  });

  it('inserts a column and shifts everything to the right of it', () => {
    s().deleteMany([s().design.compartments[1].id, s().design.compartments[3].id]);
    const [left] = s().design.compartments;
    const before = s().design.compartments.map((c) => c.x);
    const ids = s().insertLane(left.id, 'x', false);
    expect(ids).toHaveLength(1);
    const after = s().design.compartments;
    expect(after).toHaveLength(3);
    const fresh = after.find((c) => c.id === ids[0])!;
    expect(fresh.x).toBeCloseTo(left.x + left.w, 5);
    expect(fresh.w).toBeCloseTo(left.w, 5);
    expect(before[0]).toBe(after[0].x); // the source stayed put
  });

  it('refuses a lane that would push the layout out of the drawer', () => {
    const [a] = s().design.compartments;
    expect(s().insertLane(a.id, 'x', false)).toEqual([]);
  });
});
