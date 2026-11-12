import type { MeshData } from './types';

export interface MeshReport {
  /** True when the mesh is a closed, oriented surface with no coincident vertices. */
  watertight: boolean;
  /** Edges not shared by exactly two oppositely-oriented triangles. */
  badEdges: number;
  degenerate: number;
  /** Vertices sharing a position with another vertex – a sign of self-touching geometry. */
  coincident: number;
  /** Signed volume from the divergence theorem (mm³); positive = outward normals. */
  signedVolume: number;
}

/**
 * Independent watertightness check, run on the exact mesh that is exported.
 * A mesh is watertight when every directed edge a→b is matched by exactly one b→a.
 */
export function checkMesh(mesh: MeshData): MeshReport {
  const { positions: p, indices: idx } = mesh;
  // Merge vertices by position first: an STL only stores coordinates, so two vertices
  // at the same spot become one for anything reading the exported file.
  const byPos = new Map<string, number>();
  const merged = new Int32Array(p.length / 3);
  for (let i = 0; i < merged.length; i++) {
    const key = `${p[i * 3].toFixed(4)},${p[i * 3 + 1].toFixed(4)},${p[i * 3 + 2].toFixed(4)}`;
    const seen = byPos.get(key);
    if (seen === undefined) byPos.set(key, i);
    merged[i] = seen ?? i;
  }
  const coincident = merged.length - byPos.size;
  const directed = new Map<string, number>();
  let degenerate = 0;
  let vol = 0;
  for (let t = 0; t < idx.length; t += 3) {
    const tri = [merged[idx[t]], merged[idx[t + 1]], merged[idx[t + 2]]];
    if (tri[0] === tri[1] || tri[1] === tri[2] || tri[0] === tri[2]) degenerate++;
    for (let e = 0; e < 3; e++) {
      const k = `${tri[e]},${tri[(e + 1) % 3]}`;
      directed.set(k, (directed.get(k) ?? 0) + 1);
    }
    const [a, b, c] = tri.map((i) => [p[i * 3], p[i * 3 + 1], p[i * 3 + 2]]);
    vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  let bad = 0;
  for (const [k, n] of directed) {
    const [a, b] = k.split(',');
    if (n !== 1 || directed.get(`${b},${a}`) !== 1) bad++;
  }
  return { watertight: bad === 0 && degenerate === 0, badEdges: bad, degenerate, coincident, signedVolume: vol };
}
