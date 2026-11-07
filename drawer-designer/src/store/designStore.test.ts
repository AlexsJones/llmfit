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
