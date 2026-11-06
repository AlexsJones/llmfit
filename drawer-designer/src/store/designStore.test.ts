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
