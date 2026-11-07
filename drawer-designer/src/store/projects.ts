/**
 * Saved projects and design files.
 *
 * Projects live in localStorage under one key; design files are plain JSON
 * (`.organa.json`). Imported files are untrusted input, so `parseDesign`
 * validates and clamps every field instead of trusting the file's shape.
 */
import { DEFAULT_BUILD_VOLUME, DEFAULT_DRAWER, DEFAULT_PRINT, LIMITS, clamp, createEmptyDesign, newId } from '../model/defaults';
import type { Compartment, Design, Divider, ParamValue } from '../model/types';

export const FILE_VERSION = 1;
const KEY = 'organa-projects-v1';
const MAX_COMPARTMENTS = 2000;

export interface ProjectMeta {
  id: string;
  name: string;
  savedAt: number;
  compartments: number;
}

interface StoredProject extends ProjectMeta {
  design: Design;
}

function readAll(): StoredProject[] {
  try {
    const raw = window.localStorage.getItem(KEY);
    if (!raw) return [];
    const data: unknown = JSON.parse(raw);
    if (!Array.isArray(data)) return [];
    return data.flatMap((p) => {
      const rec = p as Partial<StoredProject>;
      const design = parseDesign(rec.design);
      if (!design || typeof rec.id !== 'string') return [];
      return [{ id: rec.id, name: design.name, savedAt: Number(rec.savedAt) || 0, compartments: design.compartments.length, design }];
    });
  } catch {
    return [];
  }
}

function writeAll(list: StoredProject[]) {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(list));
    return true;
  } catch {
    return false;
  }
}

export function listProjects(): ProjectMeta[] {
  return readAll()
    .map(({ id, name, savedAt, compartments }) => ({ id, name, savedAt, compartments }))
    .sort((a, b) => b.savedAt - a.savedAt);
}

/** Save under an existing id, or create a new project. Returns the id, or null if storage is full. */
export function saveProject(design: Design, id?: string): string | null {
  const list = readAll();
  const pid = id ?? newId('pr');
  const entry: StoredProject = { id: pid, name: design.name, savedAt: Date.now(), compartments: design.compartments.length, design };
  const at = list.findIndex((p) => p.id === pid);
  if (at >= 0) list[at] = entry;
  else list.push(entry);
  return writeAll(list) ? pid : null;
}

export function loadProject(id: string): Design | null {
  return readAll().find((p) => p.id === id)?.design ?? null;
}

export function deleteProject(id: string) {
  writeAll(readAll().filter((p) => p.id !== id));
}

// ---- file format ----------------------------------------------------------------

export function serialiseDesign(design: Design): string {
  return JSON.stringify({ format: 'organa-design', version: FILE_VERSION, design }, null, 2);
}

const n = (v: unknown, min: number, max: number, fallback: number) => {
  const x = typeof v === 'number' && Number.isFinite(v) ? v : fallback;
  return Math.round(clamp(x, min, max) * 100) / 100;
};
const str = (v: unknown, fallback: string, max = 80) => (typeof v === 'string' && v.trim() ? v.slice(0, max) : fallback);

function parseParams(v: unknown): Record<string, ParamValue> {
  if (!v || typeof v !== 'object' || Array.isArray(v)) return {};
  const out: Record<string, ParamValue> = {};
  for (const [k, val] of Object.entries(v as Record<string, unknown>)) {
    if (out && Object.keys(out).length >= 40) break;
    if (typeof val === 'number' && Number.isFinite(val)) out[k.slice(0, 40)] = val;
    else if (typeof val === 'boolean') out[k.slice(0, 40)] = val;
    else if (typeof val === 'string') out[k.slice(0, 40)] = val.slice(0, 2000);
  }
  return out;
}

function parseCompartment(v: unknown, i: number): Compartment | null {
  if (!v || typeof v !== 'object') return null;
  const c = v as Record<string, unknown>;
  const w = n(c.w, 0, LIMITS.drawer.max, 0);
  const d = n(c.d, 0, LIMITS.drawer.max, 0);
  if (w <= 0 || d <= 0) return null;
  const dividers: Divider[] = Array.isArray(c.dividers)
    ? c.dividers.slice(0, 64).flatMap((x) => {
        const dv = x as Record<string, unknown>;
        const orientation = dv?.orientation === 'horizontal' ? 'horizontal' : 'vertical';
        const span = orientation === 'vertical' ? w : d;
        const position = n(dv?.position, 0, span, span / 2);
        return [{ id: newId('dv'), orientation, position } as Divider];
      })
    : [];
  return {
    id: newId('cp'),
    kind: str(c.kind, 'bin', 40),
    name: str(c.name, `Compartment ${i + 1}`),
    x: n(c.x, 0, LIMITS.drawer.max, 0),
    y: n(c.y, 0, LIMITS.drawer.max, 0),
    w,
    d,
    height: typeof c.height === 'number' && Number.isFinite(c.height) ? n(c.height, LIMITS.binHeight.min, LIMITS.binHeight.max, 40) : null,
    dividers,
    params: parseParams(c.params),
  };
}

/** Validate and clamp an untrusted design object. Returns null when it is not a design at all. */
export function parseDesign(input: unknown): Design | null {
  if (!input || typeof input !== 'object') return null;
  const raw = input as Record<string, unknown>;
  const src = (raw.design && typeof raw.design === 'object' ? raw.design : raw) as Record<string, unknown>;
  if (!Array.isArray(src.compartments) || typeof src.drawer !== 'object' || !src.drawer) return null;
  const dr = src.drawer as Record<string, unknown>;
  const pr = (typeof src.print === 'object' && src.print ? src.print : {}) as Record<string, unknown>;
  const bv = (typeof src.buildVolume === 'object' && src.buildVolume ? src.buildVolume : {}) as Record<string, unknown>;
  const base = createEmptyDesign();
  const design: Design = {
    version: 1,
    name: str(src.name, 'Imported organiser'),
    drawer: {
      width: n(dr.width, LIMITS.drawer.min, LIMITS.drawer.max, DEFAULT_DRAWER.width),
      depth: n(dr.depth, LIMITS.drawer.min, LIMITS.drawer.max, DEFAULT_DRAWER.depth),
      height: n(dr.height, LIMITS.drawerHeight.min, LIMITS.drawerHeight.max, DEFAULT_DRAWER.height),
      clearance: n(dr.clearance, LIMITS.clearance.min, LIMITS.clearance.max, DEFAULT_DRAWER.clearance),
    },
    print: {
      wallThickness: n(pr.wallThickness, LIMITS.wall.min, LIMITS.wall.max, DEFAULT_PRINT.wallThickness),
      bottomThickness: n(pr.bottomThickness, LIMITS.bottom.min, LIMITS.bottom.max, DEFAULT_PRINT.bottomThickness),
      cornerStyle: pr.cornerStyle === 'square' ? 'square' : 'rounded',
      cornerRadius: n(pr.cornerRadius, LIMITS.radius.min, LIMITS.radius.max, DEFAULT_PRINT.cornerRadius),
      binHeight: n(pr.binHeight, LIMITS.binHeight.min, LIMITS.binHeight.max, base.print.binHeight),
      dividerThickness: n(pr.dividerThickness, LIMITS.dividerThickness.min, LIMITS.dividerThickness.max, DEFAULT_PRINT.dividerThickness),
      dividerHeight:
        pr.dividerHeight === null || pr.dividerHeight === undefined ? null : n(pr.dividerHeight, 1, LIMITS.binHeight.max, 30),
      fitGap: n(pr.fitGap, LIMITS.fitGap.min, LIMITS.fitGap.max, DEFAULT_PRINT.fitGap),
    },
    buildVolume: {
      x: n(bv.x, LIMITS.build.min, LIMITS.build.max, DEFAULT_BUILD_VOLUME.x),
      y: n(bv.y, LIMITS.build.min, LIMITS.build.max, DEFAULT_BUILD_VOLUME.y),
      z: n(bv.z, LIMITS.build.min, LIMITS.build.max, DEFAULT_BUILD_VOLUME.z),
    },
    compartments: src.compartments.slice(0, MAX_COMPARTMENTS).flatMap((c, i) => {
      const parsed = parseCompartment(c, i);
      return parsed ? [parsed] : [];
    }),
  };
  // Keep the layout inside the drawer even if the file disagreed.
  const uw = design.drawer.width - 2 * design.drawer.clearance;
  const ud = design.drawer.depth - 2 * design.drawer.clearance;
  design.compartments = design.compartments.filter((c) => c.x + c.w <= uw + 0.01 && c.y + c.d <= ud + 0.01);
  return design;
}

export function downloadDesign(design: Design, filename: string) {
  const blob = new Blob([serialiseDesign(design)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename.endsWith('.json') ? filename : `${filename}.organa.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
