/**
 * Binary STL writer. Takes meshes in mm (Z up), applies an optional offset and
 * writes facet normals computed from the (outward, CCW) triangle winding.
 */
import { strToU8, zipSync } from 'fflate';
import type { MeshData, Vec3 } from './types';

export interface StlItem {
  mesh: MeshData;
  offset?: Vec3;
  /** Rotation about world Z (degrees, counter-clockwise) around the pivot [cx, cy]. */
  spin?: { deg: number; cx: number; cy: number };
}

export function writeBinaryStl(items: StlItem[], header = 'Organa drawer organiser'): Uint8Array {
  const triCount = items.reduce((s, it) => s + it.mesh.indices.length / 3, 0);
  const buf = new ArrayBuffer(84 + triCount * 50);
  const view = new DataView(buf);
  const bytes = new Uint8Array(buf);
  bytes.set(strToU8(header.slice(0, 80)), 0);
  view.setUint32(80, triCount, true);
  let o = 84;
  for (const { mesh, offset = [0, 0, 0], spin } of items) {
    const p = mesh.positions;
    const idx = mesh.indices;
    const a0 = spin ? (spin.deg * Math.PI) / 180 : 0;
    const cosA = Math.cos(a0);
    const sinA = Math.sin(a0);
    const px = (x: number, y: number) => (spin ? spin.cx + (x - spin.cx) * cosA - (y - spin.cy) * sinA : x);
    const py = (x: number, y: number) => (spin ? spin.cy + (x - spin.cx) * sinA + (y - spin.cy) * cosA : y);
    for (let t = 0; t < idx.length; t += 3) {
      const a = idx[t] * 3;
      const b = idx[t + 1] * 3;
      const c = idx[t + 2] * 3;
      const ax0 = p[a] + offset[0], ay0 = p[a + 1] + offset[1], az = p[a + 2] + offset[2];
      const bx0 = p[b] + offset[0], by0 = p[b + 1] + offset[1], bz = p[b + 2] + offset[2];
      const cx0 = p[c] + offset[0], cy0 = p[c + 1] + offset[1], cz = p[c + 2] + offset[2];
      const ax = px(ax0, ay0), ay = py(ax0, ay0);
      const bx = px(bx0, by0), by = py(bx0, by0);
      const cx = px(cx0, cy0), cy = py(cx0, cy0);
      const ux = bx - ax, uy = by - ay, uz = bz - az;
      const vx = cx - ax, vy = cy - ay, vz = cz - az;
      let nx = uy * vz - uz * vy;
      let ny = uz * vx - ux * vz;
      let nz = ux * vy - uy * vx;
      const len = Math.hypot(nx, ny, nz) || 1;
      nx /= len; ny /= len; nz /= len;
      view.setFloat32(o, nx, true); view.setFloat32(o + 4, ny, true); view.setFloat32(o + 8, nz, true);
      view.setFloat32(o + 12, ax, true); view.setFloat32(o + 16, ay, true); view.setFloat32(o + 20, az, true);
      view.setFloat32(o + 24, bx, true); view.setFloat32(o + 28, by, true); view.setFloat32(o + 32, bz, true);
      view.setFloat32(o + 36, cx, true); view.setFloat32(o + 40, cy, true); view.setFloat32(o + 44, cz, true);
      view.setUint16(o + 48, 0, true);
      o += 50;
    }
  }
  return bytes;
}

export function zipFiles(files: Record<string, Uint8Array | string>): Uint8Array {
  const entries: Record<string, Uint8Array> = {};
  for (const [name, data] of Object.entries(files)) entries[name] = typeof data === 'string' ? strToU8(data) : data;
  return zipSync(entries, { level: 6 });
}

export function safeFileName(name: string): string {
  return (
    name
      .normalize('NFKD')
      .replace(/[^\w\s.-]+/g, '')
      .trim()
      .replace(/\s+/g, '-')
      .toLowerCase() || 'part'
  );
}

export function downloadBytes(bytes: Uint8Array, filename: string, mime: string) {
  const blob = new Blob([bytes.slice().buffer], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
