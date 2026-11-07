import { describe, expect, it } from 'vitest';
import { createEmptyDesign } from '../model/defaults';
import {
  deleteProject,
  isBackupBundle,
  listProjects,
  loadProject,
  parseDesign,
  restoreBundle,
  saveProject,
  serialiseDesign,
  storageReport,
} from './projects';

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

/** Minimal localStorage so the project list can be exercised under the node runner. */
function stubStorage() {
  const mem = new Map<string, string>();
  const storage = {
    get length() {
      return mem.size;
    },
    clear: () => mem.clear(),
    getItem: (k: string) => mem.get(k) ?? null,
    key: (i: number) => [...mem.keys()][i] ?? null,
    removeItem: (k: string) => void mem.delete(k),
    setItem: (k: string, v: string) => void mem.set(k, v),
  } as Storage;
  (globalThis as { window?: unknown }).window = { localStorage: storage };
  return mem;
}

describe('saved projects and backups', () => {
  it('saves, lists, reloads and deletes', () => {
    stubStorage();
    const id = saveProject(sample());
    expect(id).not.toBeNull();
    const list = listProjects();
    expect(list).toHaveLength(1);
    expect(list[0].name).toBe('Workshop');
    expect(list[0].compartments).toBe(1);
    expect(loadProject(id!)!.drawer.width).toBe(300);
    // saving under the same id updates in place instead of adding a second entry
    expect(saveProject(sample(), id!)).toBe(id);
    expect(listProjects()).toHaveLength(1);
    deleteProject(id!);
    expect(listProjects()).toHaveLength(0);
  });

  it('reports storage use and asks for a backup when there has never been one', () => {
    stubStorage();
    expect(storageReport().stale).toBe(false); // nothing saved yet, nothing to lose
    saveProject(sample());
    const r = storageReport();
    expect(r.projects).toBe(1);
    expect(r.bytes).toBeGreaterThan(0);
    expect(r.lastBackupAt).toBeNull();
    expect(r.unbacked).toBe(1);
    expect(r.stale).toBe(true);
  });

  it('restores a backup bundle under fresh ids, keeping what is already there', () => {
    stubStorage();
    const mine = saveProject(sample())!;
    const bundle = {
      format: 'organa-backup',
      version: 1,
      projects: [{ name: 'From file', savedAt: 1, design: sample() }, { name: 'Broken', savedAt: 2, design: { nope: true } }],
    };
    expect(isBackupBundle(bundle)).toBe(true);
    expect(isBackupBundle(JSON.parse(serialiseDesign(sample())))).toBe(false);
    const result = restoreBundle(bundle)!;
    expect(result).toEqual({ added: 1, skipped: 1 });
    const list = listProjects();
    expect(list).toHaveLength(2);
    // the project already in this browser is untouched
    expect(list.some((p) => p.id === mine)).toBe(true);
    expect(restoreBundle({ nope: true })).toBeNull();
  });
});
