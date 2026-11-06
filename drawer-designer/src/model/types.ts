/**
 * Core design data model.
 *
 * Everything here is plain, serialisable data measured in millimetres.
 * It contains no rendering or geometry-kernel state, so it can be persisted,
 * diffed, undone and shared. Visualisation (2D/3D) and printable geometry are
 * both *derived* from this model.
 *
 * Coordinate convention for the layout (top-down, as seen in the 2D editor):
 *   x: 0 at the drawer's left usable edge, grows to the right.
 *   y: 0 at the drawer's back usable edge, grows towards the front (down on screen).
 */

export type Id = string;

export interface DrawerConfig {
  /** Inner drawer width (mm). */
  width: number;
  /** Inner drawer depth, back to front (mm). */
  depth: number;
  /** Available vertical space inside the drawer (mm). */
  height: number;
  /** Gap kept between the organiser and each drawer wall (mm). */
  clearance: number;
}

export type CornerStyle = 'rounded' | 'square';

export interface PrintSettings {
  /** Outer wall thickness of each bin (mm). */
  wallThickness: number;
  /** Floor thickness of each bin (mm). */
  bottomThickness: number;
  cornerStyle: CornerStyle;
  /** Outer corner radius (mm), used when cornerStyle is 'rounded'. */
  cornerRadius: number;
  /** Default printed height of bins (mm). */
  binHeight: number;
  dividerThickness: number;
  /** Divider height measured from the bin's base (mm). `null` = full bin height. */
  dividerHeight: number | null;
  /** Total gap between neighbouring bins so they drop in easily (mm). */
  fitGap: number;
}

export interface BuildVolume {
  x: number;
  y: number;
  z: number;
}

export type DividerOrientation = 'vertical' | 'horizontal';

export interface Divider {
  id: Id;
  /**
   * 'vertical'   – a wall parallel to the depth axis, positioned along x.
   * 'horizontal' – a wall parallel to the width axis, positioned along y.
   */
  orientation: DividerOrientation;
  /** Distance from the compartment's left (vertical) or back (horizontal) edge (mm). */
  position: number;
}

export type ParamValue = number | boolean | string;

export interface Compartment {
  id: Id;
  /** Module kind, resolved through the module registry (see `src/modules`). */
  kind: string;
  name: string;
  x: number;
  y: number;
  /** Footprint width along x (mm). */
  w: number;
  /** Footprint depth along y (mm). */
  d: number;
  /** Per-compartment height override (mm). `null` uses PrintSettings.binHeight. */
  height: number | null;
  dividers: Divider[];
  /** Module-specific parameters. */
  params: Record<string, ParamValue>;
}

export interface Design {
  version: 1;
  name: string;
  drawer: DrawerConfig;
  print: PrintSettings;
  buildVolume: BuildVolume;
  compartments: Compartment[];
}

export interface Rect {
  x: number;
  y: number;
  w: number;
  d: number;
}

export interface UsableArea {
  width: number;
  depth: number;
  height: number;
}
