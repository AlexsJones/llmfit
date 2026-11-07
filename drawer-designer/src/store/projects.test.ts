import { describe, expect, it } from 'vitest';
import { createEmptyDesign } from '../model/defaults';
import { parseDesign, serialiseDesign } from './projects';

const sample = () => {
  const d = createEmptyDesign({ width: 300, depth: 200, height: 40, clearance: 1 });
  d.name = 'Workshop';
  d.compartments = [
    { id: 'a', kind: 'bin', name: 'One', x: 0, y: 0, w: 100, d: 90, height: null, dividers: [{ id: 'd', orientation: 'vertical', position: 50 }], params: { scoop: true } },
  ];
  return d;
};

describe('design files', () => {
  it('round-trips a design', () => {
    const out = parseDesign(JSON.parse(serialiseDesign(sample())));
    expect(out).not.toBeNull();
    expect(out!.name).toBe('Workshop');
    expect(out!.drawer).toEqual({ width: 300, depth: 200, height: 40, clearance: 1 });
    expect(out!.compartments).toHaveLength(1);
    expect(out!.compartments[0].w).toBe(100);
    expect(out!.compartments[0].params.scoop).toBe(true);
    expect(out!.compartments[0].dividers[0].position).toBe(50);
  });

  it('rejects files that are not designs', () => {
    for (const bad of [null, 42, 'hello', {}, { compartments: [] }, { drawer: {} }]) {
      expect(parseDesign(bad)).toBeNull();
    }
  });

  it('clamps hostile or broken values instead of trusting the file', () => {
    const out = parseDesign({
      name: 'x'.repeat(500),
      drawer: { width: 1e9, depth: 200, height: NaN, clearance: 'bad' },
      print: { wallThickness: 999, cornerStyle: '<script>', dividerHeight: null },
      buildVolume: { x: 0, y: null, z: 1e6 },
      compartments: [
        { kind: 'bin', w: 50, d: 50, x: 0, y: 0, dividers: [{ orientation: 'vertical', position: 1e6 }], params: { a: 1, b: () => 1 } },
        { kind: 'bin', w: 0, d: 10 },
        { kind: 'bin', w: 50, d: 50, x: 5000, y: 0 },
      ],
    })!;
    expect(out.name.length).toBeLessThanOrEqual(80);
    expect(out.drawer.width).toBeLessThanOrEqual(2000);
    expect(out.drawer.depth).toBe(200);
    expect(parseDesign({ drawer: { depth: -5 }, compartments: [] })!.drawer.depth).toBeGreaterThanOrEqual(30);
    expect(Number.isFinite(out.drawer.height)).toBe(true);
    expect(out.print.wallThickness).toBeLessThanOrEqual(6);
    expect(out.print.cornerStyle).toBe('rounded');
    expect(out.printer.build.x).toBeGreaterThanOrEqual(50);
    // zero-size and out-of-bounds compartments are dropped; the divider is clamped to the footprint
    expect(out.compartments).toHaveLength(1);
    expect(out.compartments[0].dividers[0].position).toBe(50);
    expect(out.compartments[0].params.b).toBeUndefined();
  });
});
