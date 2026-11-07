/** High-level export actions shared by the properties panel and the export dialog. */
import type { Design } from '../model/types';
import { DEFAULT_PROFILE } from './nesting';
import { filamentGrams, fitsBuildVolume } from './parts';
import { downloadBytes, safeFileName, writeBinaryStl, zipFiles } from './stl';
import type { PartMesh } from './types';

const f1 = (v: number) => (Math.round(v * 10) / 10).toFixed(1);

/** Unique, readable file names for a set of parts. */
function fileNames(parts: PartMesh[]): string[] {
  const used = new Map<string, number>();
  return parts.map((p) => {
    const base = safeFileName(p.name);
    const n = (used.get(base) ?? 0) + 1;
    used.set(base, n);
    return `${n > 1 ? `${base}-${n}` : base}.stl`;
  });
}

export function exportPart(part: PartMesh) {
  downloadBytes(writeBinaryStl([{ mesh: part.mesh }], part.name), `${safeFileName(part.name)}.stl`, 'model/stl');
}

export function billOfMaterials(design: Design, parts: PartMesh[]): string {
  const names = fileNames(parts);
  const lines = [
    `${design.name}`,
    `Drawer: ${design.drawer.width} x ${design.drawer.depth} x ${design.drawer.height} mm (clearance ${design.drawer.clearance} mm)`,
    `Walls ${design.print.wallThickness} mm · floor ${design.print.bottomThickness} mm · fit gap ${design.print.fitGap} mm`,
    `Build volume: ${design.buildVolume.x} x ${design.buildVolume.y} x ${design.buildVolume.z} mm`,
    '',
    `file, size X x Y x Z (mm), est. PLA at ${Math.round(DEFAULT_PROFILE.infill * 100)}% infill (g), fits build volume`,
    ...parts.map(
      (p, i) =>
        `${names[i]}, ${f1(p.size[0])} x ${f1(p.size[1])} x ${f1(p.size[2])}, ${f1(filamentGrams(p, DEFAULT_PROFILE.infill))}, ${fitsBuildVolume(p.size, design.buildVolume) ? 'yes' : 'NO'}`,
    ),
    '',
    'All parts print open side up, no supports needed. Units: millimetres.',
  ];
  return lines.join('\n');
}

/** Each part as its own STL inside a ZIP, plus a parts list. */
export function exportZip(design: Design, parts: PartMesh[], filename = safeFileName(design.name)) {
  const names = fileNames(parts);
  const files: Record<string, Uint8Array | string> = { 'parts.txt': billOfMaterials(design, parts) };
  parts.forEach((p, i) => (files[names[i]] = writeBinaryStl([{ mesh: p.mesh }], p.name)));
  downloadBytes(zipFiles(files), `${filename}.zip`, 'application/zip');
}

/** All parts in their assembled positions as one STL (separate shells). */
export function exportAssembled(design: Design, parts: PartMesh[]) {
  const stl = writeBinaryStl(
    parts.map((p) => ({ mesh: p.mesh, offset: p.offset })),
    design.name,
  );
  downloadBytes(stl, `${safeFileName(design.name)}-assembled.stl`, 'model/stl');
}

export function exportCompartment(design: Design, parts: PartMesh[]) {
  if (parts.length === 1) exportPart(parts[0]);
  else if (parts.length > 1) exportZip(design, parts, safeFileName(parts[0].name.split(' – ')[0]));
}
