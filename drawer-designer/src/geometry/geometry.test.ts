import { beforeAll, describe, expect, it } from 'vitest';
import { createEmptyDesign } from '../model/defaults';
import type { Compartment, Design } from '../model/types';
import { loadKernel, type Kernel } from './kernel';
import { checkMesh } from './meshCheck';
import { buildCompartmentGeometry, buildDesignGeometry, fitsBuildVolume } from './parts';
import { writeBinaryStl } from './stl';
import { MODULES } from '../modules/registry';

let k: Kernel;
beforeAll(async () => {
  k = await loadKernel();
});

const comp = (over: Partial<Compartment> = {}): Compartment => ({
  id: 'c1', kind: 'bin', name: 'Bin', x: 0, y: 0, w: 80, d: 60, height: null, dividers: [], params: {}, ...over,
});

function design(compartments: Compartment[], over: Partial<Design['print']> = {}): Design {
  const d = createEmptyDesign({ width: 300, depth: 200, height: 50, clearance: 1 });
  return { ...d, print: { ...d.print, ...over }, compartments };
}

describe('bin geometry', () => {
  it('produces a watertight bin with the expected dimensions', () => {
    const d = design([comp()]);
    const g = buildCompartmentGeometry(k, d, d.compartments[0]);
    expect(g.error).toBeNull();
    const part = g.parts[0];
    expect(part.manifold).toBe(true);
    expect(part.size[0]).toBeCloseTo(80 - d.print.fitGap, 3);
    expect(part.size[1]).toBeCloseTo(60 - d.print.fitGap, 3);
    expect(part.size[2]).toBeCloseTo(d.print.binHeight, 3);
    const report = checkMesh(part.mesh);
    expect(report.watertight).toBe(true);
    expect(report.signedVolume).toBeGreaterThan(0);
    expect(report.signedVolume).toBeCloseTo(part.volume, 0);
  });

  it('stays watertight with crossing dividers, square and rounded corners', () => {
    for (const cornerStyle of ['rounded', 'square'] as const) {
      const c = comp({
        dividers: [
          { id: 'a', orientation: 'vertical', position: 30 },
          { id: 'b', orientation: 'vertical', position: 55 },
          { id: 'c', orientation: 'horizontal', position: 20 },
        ],
      });
      const d = design([c], { cornerStyle, dividerHeight: 30 });
      const part = buildCompartmentGeometry(k, d, c).parts[0];
      expect(part.manifold).toBe(true);
      expect(checkMesh(part.mesh).watertight).toBe(true);
      expect(part.volume).toBeGreaterThan(buildCompartmentGeometry(k, design([comp({ id: 'x' })], { cornerStyle }), comp({ id: 'x' })).parts[0].volume);
    }
  });

  it('adds a watertight finger scoop and label ledge that stay inside the bin', () => {
    const plain = comp();
    const featured = comp({ params: { scoop: true, labelTab: true }, dividers: [{ id: 'a', orientation: 'vertical', position: 40 }] });
    for (const cornerStyle of ['rounded', 'square'] as const) {
      const d = design([featured], { cornerStyle });
      const base = buildCompartmentGeometry(k, design([plain], { cornerStyle }), plain).parts[0];
      const part = buildCompartmentGeometry(k, d, featured).parts[0];
      expect(part.manifold).toBe(true);
      expect(checkMesh(part.mesh).watertight).toBe(true);
      // Features add material but never change the outer envelope.
      expect(part.volume).toBeGreaterThan(base.volume + 500);
      part.size.forEach((v, i) => expect(v).toBeCloseTo(base.size[i], 4));
    }
  });

  it('is deterministic', () => {
    const d = design([comp({ dividers: [{ id: 'a', orientation: 'vertical', position: 40 }] })]);
    const a = writeBinaryStl([{ mesh: buildDesignGeometry(k, d)[0].parts[0].mesh }]);
    const d2 = design([comp({ id: 'other', dividers: [{ id: 'z', orientation: 'vertical', position: 40 }] })], { binHeight: d.print.binHeight + 0 });
    const b = writeBinaryStl([{ mesh: buildDesignGeometry(k, d2)[0].parts[0].mesh }]);
    expect(Buffer.from(a).equals(Buffer.from(b))).toBe(true);
  });

  it('builds removable cups as separate watertight parts that fit inside the tray', () => {
    const c = comp({ kind: 'cup-tray', w: 120, d: 90, params: { cols: 3, rows: 2 } });
    const d = design([c]);
    const g = buildCompartmentGeometry(k, d, c);
    expect(g.parts).toHaveLength(7);
    const tray = g.parts[0];
    for (const p of g.parts) {
      expect(p.manifold).toBe(true);
      expect(checkMesh(p.mesh).watertight).toBe(true);
    }
    for (const cup of g.parts.slice(1)) {
      expect(cup.offset[0]).toBeGreaterThan(tray.offset[0] + d.print.wallThickness);
      expect(cup.offset[0] + cup.size[0]).toBeLessThan(tray.offset[0] + tray.size[0] - d.print.wallThickness);
      expect(cup.offset[1]).toBeGreaterThan(tray.offset[1] + d.print.wallThickness);
      expect(cup.offset[2]).toBeCloseTo(d.print.bottomThickness, 5);
    }
  });

  it('reports invalid specs instead of producing broken geometry', () => {
    const c = comp({ w: 3, d: 3 });
    expect(buildCompartmentGeometry(k, design([c]), c).error).toMatch(/small/);
  });

  it('places bins in the drawer world frame', () => {
    const c = comp({ x: 10, y: 20 });
    const d = design([c]);
    const part = buildCompartmentGeometry(k, d, c).parts[0];
    const gap = d.print.fitGap;
    expect(part.offset[0]).toBeCloseTo(1 + 10 + gap / 2, 5);
    // front edge in world Y = drawer depth - clearance - (y + d) + gap/2
    expect(part.offset[1]).toBeCloseTo(200 - 1 - 80 + gap / 2, 5);
  });
});

/** Area of triangles lying in z=0 and facing down. */
function baseArea(m: { positions: Float32Array; indices: Uint32Array }) {
  const p = m.positions;
  let area = 0;
  for (let t = 0; t < m.indices.length; t += 3) {
    const [a, b, c] = [m.indices[t] * 3, m.indices[t + 1] * 3, m.indices[t + 2] * 3];
    if (Math.abs(p[a + 2]) > 1e-4 || Math.abs(p[b + 2]) > 1e-4 || Math.abs(p[c + 2]) > 1e-4) continue;
    const cz = (p[b] - p[a]) * (p[c + 1] - p[a + 1]) - (p[b + 1] - p[a + 1]) * (p[c] - p[a]);
    if (cz < 0) area += -cz / 2;
  }
  return area;
}

describe('module library', () => {
  for (const mod of MODULES) {
    it(`${mod.label}: every part is watertight and inside the bin envelope`, () => {
      for (const cornerStyle of ['rounded', 'square'] as const) {
        const c = comp({ kind: mod.kind, w: 140, d: 100, params: { ...mod.defaults } });
        const d = design([c], { cornerStyle });
        const g = buildCompartmentGeometry(k, d, c);
        expect(g.error).toBeNull();
        expect(g.parts.length).toBeGreaterThan(0);
        const bin = g.parts[0];
        for (const p of g.parts) {
          expect(p.manifold, p.name).toBe(true);
          const rep = checkMesh(p.mesh);
          expect(rep.watertight, p.name).toBe(true);
          expect(rep.signedVolume).toBeGreaterThan(0);
          // nothing pokes outside the bin's footprint or height
          for (let i = 0; i < 3; i++) {
            expect(p.offset[i]).toBeGreaterThanOrEqual(bin.offset[i] - 1e-3);
            expect(p.offset[i] + p.size[i]).toBeLessThanOrEqual(bin.offset[i] + bin.size[i] + 1e-3);
          }
        }
        expect(bin.size[0]).toBeCloseTo(140 - d.print.fitGap, 3);
        expect(bin.size[2]).toBeCloseTo(d.print.binHeight, 3);
      }
    });
  }

  it('module features actually change the geometry', () => {
    const plain = buildCompartmentGeometry(k, design([comp({ w: 140, d: 100 })]), comp({ w: 140, d: 100 })).parts[0];
    for (const mod of MODULES.filter((m) => m.kind !== 'bin' && m.kind !== 'cup-tray')) {
      const c = comp({ kind: mod.kind, w: 140, d: 100, params: { ...mod.defaults } });
      const part = buildCompartmentGeometry(k, design([c]), c).parts[0];
      expect(Math.abs(part.volume - plain.volume), mod.kind).toBeGreaterThan(100);
    }
  });

  it('pen cradles, holes and slots stay above the floor', () => {
    for (const kind of ['pen-holder', 'custom-insert', 'tool-holder']) {
      const mod = MODULES.find((m) => m.kind === kind)!;
      const c = comp({ kind, w: 140, d: 100, params: { ...mod.defaults, depth: 200, cutDepth: 200 } });
      const d = design([c]);
      const m = buildCompartmentGeometry(k, d, c).parts[0].mesh;
      const plain = buildCompartmentGeometry(k, d, comp({ w: 140, d: 100 })).parts[0].mesh;
      // Cuts must never break through the floor: the downward-facing base area is unchanged.
      expect(baseArea(m)).toBeCloseTo(baseArea(plain), 1);
      expect(checkMesh(m).watertight).toBe(true);
    }
  });
});

describe('build volume', () => {
  it('allows 90° rotation on the bed', () => {
    expect(fitsBuildVolume([300, 100, 50], { x: 320, y: 150, z: 100 })).toBe(true);
    expect(fitsBuildVolume([100, 300, 50], { x: 320, y: 150, z: 100 })).toBe(true);
    expect(fitsBuildVolume([300, 300, 50], { x: 256, y: 256, z: 256 })).toBe(false);
  });
});

describe('stl', () => {
  it('writes a binary STL with the right size and triangle count', () => {
    const d = design([comp()]);
    const part = buildDesignGeometry(k, d)[0].parts[0];
    const stl = writeBinaryStl([{ mesh: part.mesh }]);
    const view = new DataView(stl.buffer);
    expect(view.getUint32(80, true)).toBe(part.triangles);
    expect(stl.byteLength).toBe(84 + part.triangles * 50);
  });
});
