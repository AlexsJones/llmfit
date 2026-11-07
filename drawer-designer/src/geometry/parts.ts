/**
 * Design → printable parts pipeline.
 *
 * This is the single source of truth for geometry: both the 3D preview and the
 * STL export consume the meshes produced here, so what you see is exactly
 * what you print. Results are cached by a deterministic key derived from the
 * inputs that affect a part's shape (not its placement), so dragging a
 * compartment around never re-runs CSG.
 */
import type { Manifold } from 'manifold-3d';
import { compartmentHeight } from '../model/defaults';
import type { BuildVolume, Compartment, Design, ParamValue } from '../model/types';
import { getModule, getMounting } from '../modules/registry';
import { type BinSpec, validateBinSpec } from './bin';
import type { Kernel } from './kernel';
import type { MeshData, PartMesh, PartRole, Vec3 } from './types';

export function resolveBinSpec(design: Design, c: Compartment): BinSpec {
  const p = design.print;
  const gap = p.fitGap;
  const width = c.w - gap;
  const depth = c.d - gap;
  const height = compartmentHeight(c, p);
  const mod = getModule(c.kind);
  return {
    width,
    depth,
    height,
    wall: p.wallThickness,
    bottom: p.bottomThickness,
    cornerRadius: p.cornerStyle === 'rounded' ? p.cornerRadius : 0,
    dividerThickness: p.dividerThickness,
    dividerHeight: Math.min(p.dividerHeight ?? height, height),
    dividers: mod.supportsDividers
      ? c.dividers.map((dv) => ({
          orientation: dv.orientation,
          // Layout positions are measured from the compartment's left / back edge.
          // Locally Y grows towards the back, so horizontal positions are mirrored.
          at: dv.orientation === 'vertical' ? dv.position - gap / 2 : depth - (dv.position - gap / 2),
        }))
      : [],
  };
}

export function resolveParams(c: Compartment): Record<string, ParamValue> {
  return { ...getModule(c.kind).defaults, ...c.params };
}

/** World position of a compartment's bin origin. */
export function binWorldOrigin(design: Design, c: Compartment): Vec3 {
  // World frame: X to the right, Y from the drawer FRONT towards the back, Z up.
  // Origin = front-left-bottom inner corner of the drawer.
  const { clearance, depth } = design.drawer;
  const gap = design.print.fitGap;
  return [clearance + c.x + gap / 2, depth - clearance - c.y - c.d + gap / 2, 0];
}

interface LocalPart {
  key: string;
  name: string;
  role: PartRole;
  mesh: MeshData;
  localOffset: Vec3;
  size: Vec3;
  volume: number;
  area: number;
  triangles: number;
  manifold: boolean;
}

/** Total surface area of a mesh (mm²). */
function surfaceArea(mesh: MeshData): number {
  const { positions: p, indices: idx } = mesh;
  let total = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const a = idx[t] * 3;
    const b = idx[t + 1] * 3;
    const c = idx[t + 2] * 3;
    const ux = p[b] - p[a];
    const uy = p[b + 1] - p[a + 1];
    const uz = p[b + 2] - p[a + 2];
    const vx = p[c] - p[a];
    const vy = p[c + 1] - p[a + 1];
    const vz = p[c + 2] - p[a + 2];
    total += Math.hypot(uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx) / 2;
  }
  return total;
}

function extractMesh(m: Manifold): { mesh: MeshData; min: Vec3; size: Vec3 } {
  const raw = m.getMesh();
  const bb = m.boundingBox();
  const n = raw.vertProperties.length / raw.numProp;
  const positions = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = raw.vertProperties[i * raw.numProp] - bb.min[0];
    positions[i * 3 + 1] = raw.vertProperties[i * raw.numProp + 1] - bb.min[1];
    positions[i * 3 + 2] = raw.vertProperties[i * raw.numProp + 2] - bb.min[2];
  }
  return {
    mesh: { positions, indices: new Uint32Array(raw.triVerts) },
    min: [bb.min[0], bb.min[1], bb.min[2]],
    size: [bb.max[0] - bb.min[0], bb.max[1] - bb.min[1], bb.max[2] - bb.min[2]],
  };
}

const cache = new Map<string, LocalPart[]>();
const CACHE_LIMIT = 400;

function buildLocal(kernel: Kernel, design: Design, c: Compartment): { parts: LocalPart[]; error: string | null } {
  const spec = resolveBinSpec(design, c);
  const params = resolveParams(c);
  const error = validateBinSpec(spec);
  if (error) return { parts: [], error };
  const key = JSON.stringify({ k: c.kind, spec, params, m: 'none' });
  const hit = cache.get(key);
  if (hit) {
    // refresh LRU position
    cache.delete(key);
    cache.set(key, hit);
    return { parts: hit, error: null };
  }
  const mod = getModule(c.kind);
  if (!mod.build) return { parts: [], error: `${mod.label} is not available yet` };
  const mounting = getMounting('none');
  const built = mod.build({ kernel, design, compartment: c, bin: spec, params });
  const parts: LocalPart[] = built.map((b) => {
    let solid = b.manifold;
    if (b.role === 'bin' && mounting.applyToBin) {
      const next = mounting.applyToBin(kernel, solid, spec);
      solid.delete();
      solid = next;
    }
    const { mesh, min, size } = extractMesh(solid);
    const part: LocalPart = {
      key: b.key,
      name: b.name,
      role: b.role,
      mesh,
      localOffset: [b.offset[0] + min[0], b.offset[1] + min[1], b.offset[2] + min[2]],
      size,
      volume: solid.volume(),
      area: surfaceArea(mesh),
      triangles: solid.numTri(),
      manifold: solid.status() === 'NoError' && !solid.isEmpty(),
    };
    solid.delete();
    return part;
  });
  cache.set(key, parts);
  if (cache.size > CACHE_LIMIT) {
    const oldest = cache.keys().next().value;
    if (oldest !== undefined) cache.delete(oldest);
  }
  return { parts, error: null };
}

export interface CompartmentGeometry {
  compartmentId: string;
  parts: PartMesh[];
  error: string | null;
}

export function buildCompartmentGeometry(kernel: Kernel, design: Design, c: Compartment): CompartmentGeometry {
  const { parts, error } = buildLocal(kernel, design, c);
  const origin = binWorldOrigin(design, c);
  const height = compartmentHeight(c, design.print);
  const warnings: string[] = [];
  if (height > design.drawer.height) warnings.push(`Taller than the drawer (${height} > ${design.drawer.height} mm)`);
  return {
    compartmentId: c.id,
    error,
    parts: parts.map((p) => ({
      id: `${c.id}:${p.key}`,
      compartmentId: c.id,
      name: parts.length > 1 ? `${c.name} – ${p.name}` : c.name,
      role: p.role,
      mesh: p.mesh,
      offset: [origin[0] + p.localOffset[0], origin[1] + p.localOffset[1], origin[2] + p.localOffset[2]],
      size: p.size,
      volume: p.volume,
      area: p.area,
      triangles: p.triangles,
      manifold: p.manifold,
      warnings: [...warnings, ...buildVolumeWarnings(p.size, design.buildVolume)],
    })),
  };
}

export function buildDesignGeometry(kernel: Kernel, design: Design): CompartmentGeometry[] {
  return design.compartments.map((c) => buildCompartmentGeometry(kernel, design, c));
}

/** Can a part of this size be printed? Allows a 90° rotation on the bed. */
export function fitsBuildVolume(size: Vec3, bv: BuildVolume): boolean {
  const [x, y, z] = size;
  const flat = (x <= bv.x && y <= bv.y) || (x <= bv.y && y <= bv.x);
  return flat && z <= bv.z;
}

export function buildVolumeWarnings(size: Vec3, bv: BuildVolume): string[] {
  if (fitsBuildVolume(size, bv)) return [];
  return [`Exceeds build volume (${bv.x} × ${bv.y} × ${bv.z} mm)`];
}

/** PLA density, g/mm³. */
export const PLA_DENSITY = 1.24e-3;

/**
 * Thickness of the printed shell (perimeters plus solid top and bottom layers) used to
 * split a part into shell and infilled core (mm).
 */
export const SHELL_THICKNESS = 0.9;

/**
 * Plastic actually extruded for a part (mm³).
 *
 * A slicer prints a solid shell and fills the rest at the infill density, so a chunky
 * insert uses far less material than its solid volume suggests. The shell is estimated as
 * surface area × shell thickness, capped at the part's own volume.
 */
export function extrudedVolume(part: { volume: number; area: number }, infill: number): number {
  const shell = Math.min(part.volume, part.area * SHELL_THICKNESS);
  return shell + (part.volume - shell) * Math.min(1, Math.max(0, infill));
}

export function filamentGrams(part: { volume: number; area: number }, infill: number): number {
  return extrudedVolume(part, infill) * PLA_DENSITY;
}

export function clearGeometryCache() {
  cache.clear();
}
