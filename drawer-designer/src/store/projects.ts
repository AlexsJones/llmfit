/**
 * Saved projects and design files.
 *
 * Projects live in localStorage under one key; design files are plain JSON
 * (`.organa.json`). Imported files are untrusted input, so `parseDesign`
 * validates and clamps every field instead of trusting the file's shape.
 */
import { DEFAULT_DRAWER, DEFAULT_PRINT, LIMITS, clamp, createEmptyDesign, newId, printerFromPreset } from '../model/defaults';
import { inBounds } from '../layout/ops';
import { findPrinter } from '../model/printers';
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
    rot: typeof c.rot === 'number' && Number.isFinite(c.rot) ? (((Math.round(c.rot * 10) / 10) % 360) + 360) % 360 || undefined : undefined,
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

/**
 * Printer settings from a file. Older files carried only `buildVolume`, so that is
 * accepted as a fallback and the rest filled from the default machine.
 */
function parsePrinter(input: unknown, legacyBuild: unknown): Design['printer'] {
  const base = printerFromPreset('bambu-x2d');
  const raw = (typeof input === 'object' && input ? input : {}) as Record<string, unknown>;
  const legacy = (typeof legacyBuild === 'object' && legacyBuild ? legacyBuild : {}) as Record<string, unknown>;
  const b = (typeof raw.build === 'object' && raw.build ? raw.build : legacy) as Record<string, unknown>;
  const presetId = typeof raw.presetId === 'string' ? raw.presetId.slice(0, 40) : base.presetId;
  const preset = findPrinter(presetId);
  return {
    presetId: preset ? preset.id : 'custom',
    label: str(raw.label, preset ? `${preset.brand} ${preset.model}` : 'Custom printer', 60),
    build: {
      x: n(b.x, LIMITS.build.min, LIMITS.build.max, base.build.x),
      y: n(b.y, LIMITS.build.min, LIMITS.build.max, base.build.y),
      z: n(b.z, LIMITS.build.min, LIMITS.build.max, base.build.z),
    },
    nozzle: n(raw.nozzle, LIMITS.nozzle.min, LIMITS.nozzle.max, base.nozzle),
    layerHeight: n(raw.layerHeight, LIMITS.layer.min, LIMITS.layer.max, base.layerHeight),
    speed: n(raw.speed, LIMITS.speed.min, LIMITS.speed.max, base.speed),
    efficiency: n(raw.efficiency, 0.05, 1, base.efficiency),
    infill: n(raw.infill, 0, 1, base.infill),
    startupMinutes: n(raw.startupMinutes, 0, 240, base.startupMinutes),
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
    printer: parsePrinter(src.printer, src.buildVolume),
    compartments: src.compartments.slice(0, MAX_COMPARTMENTS).flatMap((c, i) => {
      const parsed = parseCompartment(c, i);
      return parsed ? [parsed] : [];
    }),
  };
  // Keep the layout inside the drawer even if the file disagreed.
  const uw = design.drawer.width - 2 * design.drawer.clearance;
  const ud = design.drawer.depth - 2 * design.drawer.clearance;
  design.compartments = design.compartments.filter((c) => inBounds(c, { width: uw, depth: ud }));
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

// ---- backups ---------------------------------------------------------------------

const BACKUP_KEY = 'organa-last-backup';
/** How long a backup counts as recent before the dialog asks for a new one. */
export const BACKUP_STALE_DAYS = 14;

export interface StorageReport {
  projects: number;
  /** Bytes the project list occupies, measured as UTF-16 code units of the stored string. */
  bytes: number;
  lastBackupAt: number | null;
  /** Projects saved since the last backup. */
  unbacked: number;
  stale: boolean;
}

function lastBackup(): number | null {
  try {
    const v = Number(window.localStorage.getItem(BACKUP_KEY));
    return Number.isFinite(v) && v > 0 ? v : null;
  } catch {
    return null;
  }
}

export function storageReport(): StorageReport {
  const list = readAll();
  const at = lastBackup();
  let bytes = 0;
  try {
    bytes = (window.localStorage.getItem(KEY)?.length ?? 0) * 2;
  } catch {
    bytes = 0;
  }
  const unbacked = at === null ? list.length : list.filter((p) => p.savedAt > at).length;
  return {
    projects: list.length,
    bytes,
    lastBackupAt: at,
    unbacked,
    stale: list.length > 0 && (at === null || Date.now() - at > BACKUP_STALE_DAYS * 86400_000),
  };
}

/**
 * Write every saved project into one file. This is the only copy that survives
 * clearing the browser's site data, so the dialog pushes the user towards it.
 */
export function backupAll(): number {
  const list = readAll();
  if (!list.length) return 0;
  const body = JSON.stringify(
    {
      format: 'organa-backup',
      version: FILE_VERSION,
      savedAt: new Date().toISOString(),
      projects: list.map((p) => ({ name: p.name, savedAt: p.savedAt, design: p.design })),
    },
    null,
    2,
  );
  const stamp = new Date().toISOString().slice(0, 10);
  const blob = new Blob([body], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `organa-backup-${stamp}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  try {
    window.localStorage.setItem(BACKUP_KEY, String(Date.now()));
  } catch {
    /* a missing stamp only means the dialog keeps asking */
  }
  return list.length;
}

/** True when the object looks like a backup bundle rather than a single design. */
export function isBackupBundle(input: unknown): boolean {
  if (!input || typeof input !== 'object') return false;
  const raw = input as Record<string, unknown>;
  return raw.format === 'organa-backup' || Array.isArray(raw.projects);
}

/**
 * Add every design in a bundle back to this browser under fresh ids, so a
 * restore never overwrites a project that is already here.
 */
export function restoreBundle(input: unknown): { added: number; skipped: number } | null {
  if (!isBackupBundle(input)) return null;
  const items = (input as Record<string, unknown>).projects;
  if (!Array.isArray(items)) return null;
  const list = readAll();
  let added = 0;
  let skipped = 0;
  for (const item of items.slice(0, 500)) {
    const rec = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>;
    const design = parseDesign(rec.design ?? rec);
    if (!design) {
      skipped++;
      continue;
    }
    list.push({
      id: newId('pr'),
      name: design.name,
      savedAt: Number(rec.savedAt) || Date.now(),
      compartments: design.compartments.length,
      design,
    });
    added++;
  }
  if (!writeAll(list)) return null;
  return { added, skipped };
}
