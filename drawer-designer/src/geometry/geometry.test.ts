import { beforeAll, describe, expect, it } from 'vitest';
import { createEmptyDesign } from '../model/defaults';
import type { Compartment, Design } from '../model/types';
import { loadKernel, type Kernel } from './kernel';
import { checkMesh } from './meshCheck';
import { buildCompartmentGeometry, buildDesignGeometry, fitsBuildVolume } from './parts';
import { writeBinaryStl } from './stl';
import { MODULES } from '../modules/registry';
import { encodeOutline } from './outline';

let k: Kernel;
beforeAll(async () => {
  k = await loadKernel();
});

/** A star-ish outline with a concave part, encoded as the module stores it. */
const STAR = encodeOutline(
  Array.from({ length: 10 }, (_, i) => {
    const a = (Math.PI * 2 * i) / 10 - Math.PI / 2;
    const r = i % 2 ? 0.22 : 0.5;
    return { x: 0.5 + Math.cos(a) * r, y: 0.5 + Math.sin(a) * r };
  }),
);

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
        const c = comp({ kind: mod.kind, w: 140, d: 100, params: { ...mod.defaults, ...(mod.kind === 'outline-insert' ? { outline: STAR } : {}) } });
        const d = design([c], { cornerStyle });
        const g = buildCompartmentGeometry(k, d, c);
        expect(g.error).toBeNull();
        expect(g.parts.length).toBeGreaterThan(0);
        const bin = g.parts[0];
        for (const p of g.parts) {
          expect(p.manifold, p.name).toBe(true);
          const rep = checkMesh(p.mesh);
          expect(rep.watertight, p.name).toBe(true);
          expect(rep.coincident, `${p.name} has coincident vertices`).toBe(0);
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
      const c = comp({ kind: mod.kind, w: 140, d: 100, params: { ...mod.defaults, ...(mod.kind === 'outline-insert' ? { outline: STAR } : {}) } });
      const part = buildCompartmentGeometry(k, design([c]), c).parts[0];
      expect(Math.abs(part.volume - plain.volume), mod.kind).toBeGreaterThan(100);
    }
  });

  it('cuts a concave custom outline, and an empty outline leaves a plain bin', () => {
    const d = design([]);
    const base = { outline: STAR, sizeMode: 'fit', count: 2, depth: 15, margin: 5, tolerance: 0.6 };
    const withShape = comp({ kind: 'outline-insert', w: 140, d: 100, params: base });
    const one = comp({ kind: 'outline-insert', w: 140, d: 100, params: { ...base, count: 1 } });
    const empty = comp({ kind: 'outline-insert', w: 140, d: 100, params: { ...base, outline: '', tolerance: 0 } });
    const plain = comp({ w: 140, d: 100 });
    const a = buildCompartmentGeometry(k, { ...d, compartments: [withShape] }, withShape).parts[0];
    const single = buildCompartmentGeometry(k, { ...d, compartments: [one] }, one).parts[0];
    const b = buildCompartmentGeometry(k, { ...d, compartments: [empty] }, empty).parts[0];
    const openBin = buildCompartmentGeometry(k, { ...d, compartments: [plain] }, plain).parts[0];
    expect(checkMesh(a.mesh).watertight).toBe(true);
    expect(a.volume).toBeGreaterThan(openBin.volume);
    expect(a.volume).toBeGreaterThan(single.volume);
    expect(checkMesh(single.mesh).watertight).toBe(true);
    expect(b.volume).toBeCloseTo(openBin.volume, 5);
    expect(baseArea(a.mesh)).toBeCloseTo(baseArea(openBin.mesh), 1);
  });

  it('cuts an outline at its real measured size, not stretched to the compartment', () => {
    const d = design([]);
    const at = (length: number, w = 140) =>
      comp({ kind: 'outline-insert', w, d: 100, params: { outline: STAR, sizeMode: 'actual', length, count: 1, depth: 15, margin: 5, tolerance: 0 } });
    const small = at(40);
    const large = at(80);
    const gs = buildCompartmentGeometry(k, { ...d, compartments: [small] }, small);
    const gl = buildCompartmentGeometry(k, { ...d, compartments: [large] }, large);
    expect(gs.error).toBeNull();
    expect(gl.error).toBeNull();
    // The star is square-ish, so doubling its length removes roughly four times the material.
    const blockOnly = buildCompartmentGeometry(k, { ...d, compartments: [at(3.1)] }, at(3.1)).parts[0].volume;
    const removedSmall = blockOnly - gs.parts[0].volume;
    const removedLarge = blockOnly - gl.parts[0].volume;
    expect(removedSmall).toBeGreaterThan(0);
    expect(removedLarge).toBeGreaterThan(removedSmall * 3);
    expect(removedLarge).toBeLessThan(removedSmall * 5);
    // The same shape in a wider compartment keeps its size instead of stretching.
    const wide = at(40, 200);
    const inWide = buildCompartmentGeometry(k, { ...d, compartments: [wide] }, wide).parts[0];
    const blockWide = buildCompartmentGeometry(k, { ...d, compartments: [at(3.1, 200)] }, at(3.1, 200)).parts[0].volume;
    expect(blockWide - inWide.volume).toBeCloseTo(removedSmall, -1);
    for (const g of [gs, gl]) expect(checkMesh(g.parts[0].mesh).watertight).toBe(true);
  });

  it('refuses a real-size outline that does not fit the compartment', () => {
    const d = design([]);
    const big = comp({ kind: 'outline-insert', w: 60, d: 60, params: { outline: STAR, sizeMode: 'actual', length: 200, count: 1, depth: 15, margin: 5, tolerance: 0 } });
    const g = buildCompartmentGeometry(k, { ...d, compartments: [big] }, big);
    expect(g.error).toMatch(/larger than/);
    expect(g.parts).toHaveLength(0);
  });

  it('rotates an outline by 90 degrees', () => {
    const d = design([]);
    // A bar twice as long as it is wide: rotating swaps which axis is longer.
    const BAR = encodeOutline([{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 0.5 }, { x: 0, y: 0.5 }]);
    const mk = (rotation: string) =>
      comp({ kind: 'outline-insert', w: 120, d: 120, params: { outline: BAR, sizeMode: 'actual', length: 60, rotation, count: 1, depth: 15, margin: 5, tolerance: 0 } });
    const flat = buildCompartmentGeometry(k, { ...d, compartments: [mk('0')] }, mk('0')).parts[0];
    const turned = buildCompartmentGeometry(k, { ...d, compartments: [mk('90')] }, mk('90')).parts[0];
    // Same volume removed, but the pocket now runs the other way, so the meshes differ.
    expect(turned.volume).toBeCloseTo(flat.volume, 0);
    expect(Buffer.from(writeBinaryStl([{ mesh: turned.mesh }])).equals(Buffer.from(writeBinaryStl([{ mesh: flat.mesh }])))).toBe(false);
    expect(checkMesh(turned.mesh).watertight).toBe(true);
  });

  it('pen cradles, holes and slots stay above the floor, even at absurd depths', () => {
    for (const kind of ['pen-holder', 'custom-insert', 'tool-holder', 'card-holder']) {
      const mod = MODULES.find((m) => m.kind === kind)!;
      const c = comp({ kind, w: 140, d: 100, params: { ...mod.defaults, depth: 200, cutDepth: 200 } });
      const d = design([c]);
      const m = buildCompartmentGeometry(k, d, c).parts[0].mesh;
      const plain = buildCompartmentGeometry(k, d, comp({ w: 140, d: 100 })).parts[0].mesh;
      // Cuts must never break through the floor: the downward-facing base area is unchanged.
      expect(baseArea(m), kind).toBeCloseTo(baseArea(plain), 1);
      const rep = checkMesh(m);
      expect(rep.watertight, kind).toBe(true);
      expect(rep.coincident, kind).toBe(0);
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
