export type Vec3 = [number, number, number];

/** Triangle mesh extracted from the geometry kernel. Positions are in mm, Z up. */
export interface MeshData {
  positions: Float32Array;
  indices: Uint32Array;
}

export type PartRole = 'bin' | 'insert' | 'accessory';

/** A single printable object. */
export interface PartMesh {
  /** Stable key: `${compartmentId}:${localKey}`. */
  id: string;
  compartmentId: string;
  name: string;
  role: PartRole;
  /** Mesh in local coordinates; its bounding box starts at (0,0,0). */
  mesh: MeshData;
  /** World placement of the local origin inside the drawer (Z up, origin = drawer inner front-left-bottom corner, Y towards the back). */
  offset: Vec3;
  /** Bounding-box size (mm). */
  size: Vec3;
  /** Solid volume (mm³). */
  volume: number;
  triangles: number;
  /** True when the kernel reports a valid closed manifold. */
  manifold: boolean;
  /** Free-form geometry warnings. */
  warnings: string[];
}
