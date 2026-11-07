/**
 * Module system: the extension point for compartments and accessories.
 *
 * Each compartment in the layout has a `kind`. A ModuleDefinition for that kind
 * turns the compartment + global print settings into one or more printable
 * parts. New accessories (pen holders, screw organisers, tool holders,
 * stackable trays, custom inserts …) plug in by registering a definition –
 * the editor, 3D preview, properties panel and STL export pick them up
 * automatically.
 */
import type { Manifold } from 'manifold-3d';
import type { Kernel } from '../geometry/kernel';
import type { BinSpec } from '../geometry/bin';
import type { PartRole, Vec3 } from '../geometry/types';
import type { Compartment, Design, ParamValue } from '../model/types';

export interface ParamDef {
  key: string;
  label: string;
  type: 'number' | 'boolean' | 'select';
  min?: number;
  max?: number;
  step?: number;
  unit?: string;
  options?: { value: string; label: string }[];
  help?: string;
}

export interface ModuleBuildContext {
  kernel: Kernel;
  design: Design;
  compartment: Compartment;
  /** Resolved geometry of the compartment's base bin, in its local frame. */
  bin: BinSpec;
  /** Parameters merged with module defaults. */
  params: Record<string, ParamValue>;
}

export interface BuiltPart {
  /** Unique within the compartment. */
  key: string;
  name: string;
  role: PartRole;
  /** Owned by the pipeline after return – it will be meshed then deleted. */
  manifold: Manifold;
  /** Placement relative to the bin's local origin (mm). */
  offset: Vec3;
}

/** Context for drawing a module's top-down preview in the 2D editor. */
export interface ModulePreviewContext {
  compartment: Compartment;
  bin: BinSpec;
  params: Record<string, ParamValue>;
  /** Fit gap; the bin starts at gap/2 inside the compartment rectangle. */
  gap: number;
}

export type PreviewTone = 'wall' | 'block' | 'hole' | 'scoop' | 'outline';

/**
 * 2D preview shapes in compartment-local layout coordinates:
 * x from the compartment's left edge, y from its back edge (down on screen), in mm.
 */
export type PreviewShape =
  | { kind: 'rect'; x: number; y: number; w: number; d: number; r?: number; tone: PreviewTone }
  | { kind: 'circle'; cx: number; cy: number; r: number; tone: PreviewTone };

export type ModuleCategory = 'storage' | 'organisers' | 'inserts';

export interface ModuleDefinition {
  kind: string;
  label: string;
  description: string;
  category: ModuleCategory;
  /** 'planned' modules are shown in the library but cannot be placed yet. */
  status: 'available' | 'planned';
  /** Small inline SVG path data (24×24 viewbox) used as the library icon. */
  icon: string;
  params: ParamDef[];
  defaults: Record<string, ParamValue>;
  supportsDividers: boolean;
  build?: (ctx: ModuleBuildContext) => BuiltPart[];
  /** Optional top-down sketch of internal features for the 2D editor. */
  preview?: (ctx: ModulePreviewContext) => PreviewShape[];
}

/**
 * Mounting systems are applied to every part that opts in. The MVP ships only
 * the "none" system; a future shared rail / peg / magnet interface implements
 * this contract to add features (e.g. sockets in the bin floor) without
 * touching individual modules.
 */
export interface MountingSystem {
  id: string;
  label: string;
  /** Optionally modify a base bin (e.g. subtract sockets). Must return a new Manifold. */
  applyToBin?: (kernel: Kernel, bin: Manifold, spec: BinSpec) => Manifold;
}
