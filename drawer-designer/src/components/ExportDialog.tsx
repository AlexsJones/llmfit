import { AlertTriangle, CheckCircle2, Download, FileArchive, Layers, Scissors, X } from 'lucide-react';
import { useMemo } from 'react';
import { useGeometry } from '../hooks/geometry';
import { exportAssembled, exportPart, exportZip } from '../geometry/exporter';
import { checkMesh } from '../geometry/meshCheck';
import { PLA_DENSITY, fitsBuildVolume } from '../geometry/parts';
import { LIMITS } from '../model/defaults';
import { useDesign } from '../store/designStore';
import { useUi } from '../store/uiStore';
import { Badge, Button, IconButton, Modal, NumberField } from './ui/controls';

const f1 = (v: number) => String(Math.round(v * 10) / 10);

export function ExportDialog() {
  const open = useUi((s) => s.exportOpen);
  return (
    <Modal open={open} onClose={() => useUi.getState().setExportOpen(false)} width="max-w-[860px]">
      {open && <ExportContent />}
    </Modal>
  );
}

function ExportContent() {
  const design = useDesign((s) => s.design);
  const { parts } = useGeometry();
  const close = () => useUi.getState().setExportOpen(false);
  // Independent watertightness check of the exact meshes being exported.
  const reports = useMemo(() => new Map(parts.map((p) => [p.id, checkMesh(p.mesh)])), [parts]);
  const oversized = parts.filter((p) => !fitsBuildVolume(p.size, design.buildVolume));
  const allWatertight = parts.every((p) => p.manifold && reports.get(p.id)?.watertight);
  const grams = parts.reduce((s, p) => s + p.volume * PLA_DENSITY, 0);
  const bv = design.buildVolume;

  return (
    <div className="flex max-h-[92vh] flex-col">
      <div className="flex items-start gap-3 border-b border-line px-6 py-5">
        <div className="flex-1">
          <h2 className="text-[17px] font-semibold tracking-tight text-ink">Export for printing</h2>
          <p className="mt-0.5 text-[12.5px] text-muted">
            {parts.length} parts · ≈ {Math.round(grams)} g PLA · binary STL in millimetres, Z up
          </p>
        </div>
        <IconButton label="Close" onClick={close}>
          <X size={16} />
        </IconButton>
      </div>

      <div className="grid gap-3 border-b border-line bg-canvas/60 px-6 py-4 sm:grid-cols-[1fr_auto] sm:items-end">
        <div className="flex flex-wrap items-center gap-2">
          {allWatertight ? (
            <Badge tone="ok">
              <CheckCircle2 size={12} /> All parts watertight
            </Badge>
          ) : (
            <Badge tone="warn">
              <AlertTriangle size={12} /> Some parts failed validation
            </Badge>
          )}
          {oversized.length ? (
            <Badge tone="warn">
              <AlertTriangle size={12} /> {oversized.length} part{oversized.length > 1 ? 's exceed' : ' exceeds'} {bv.x}×{bv.y}×{bv.z}
            </Badge>
          ) : (
            <Badge tone="ok">
              <CheckCircle2 size={12} /> Every part fits the build plate
            </Badge>
          )}
        </div>
        <div className="flex items-end gap-1.5">
          <span className="pb-1.5 text-[11px] font-medium text-muted">Build volume</span>
          <div className="w-[76px]">
            <NumberField compact unit="X" value={bv.x} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(x) => useDesign.getState().updateBuildVolume({ x })} />
          </div>
          <div className="w-[76px]">
            <NumberField compact unit="Y" value={bv.y} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(y) => useDesign.getState().updateBuildVolume({ y })} />
          </div>
          <div className="w-[76px]">
            <NumberField compact unit="Z" value={bv.z} min={LIMITS.build.min} max={LIMITS.build.max} onChange={(z) => useDesign.getState().updateBuildVolume({ z })} />
          </div>
        </div>
      </div>

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-3 py-2 sm:px-4">
        {parts.length === 0 && <div className="py-12 text-center text-[13px] text-muted">Add compartments to generate printable parts.</div>}
        <table className="w-full text-left text-[12.5px]">
          {parts.length > 0 && (
            <thead className="sticky top-0 bg-surface text-[11px] text-muted">
              <tr>
                <th className="px-2 py-2 font-medium">Part</th>
                <th className="px-2 py-2 font-medium">Size (mm)</th>
                <th className="hidden px-2 py-2 font-medium sm:table-cell">PLA</th>
                <th className="px-2 py-2 font-medium">Status</th>
                <th className="px-2 py-2" />
              </tr>
            </thead>
          )}
          <tbody>
            {parts.map((p) => {
              const fits = fitsBuildVolume(p.size, bv);
              const ok = p.manifold && reports.get(p.id)?.watertight;
              return (
                <tr key={p.id} className="border-t border-line/70 hover:bg-subtle/50">
                  <td className="px-2 py-2">
                    <div className="font-medium text-ink">{p.name}</div>
                    <div className="text-[11px] text-faint">{p.role === 'insert' ? 'Removable insert' : 'Bin'} · {p.triangles.toLocaleString()} triangles</div>
                  </td>
                  <td className="num px-2 py-2 font-mono text-[12px] text-ink-2">
                    {f1(p.size[0])} × {f1(p.size[1])} × {f1(p.size[2])}
                  </td>
                  <td className="num hidden px-2 py-2 text-muted sm:table-cell">{f1(p.volume * PLA_DENSITY)} g</td>
                  <td className="px-2 py-2">
                    <div className="flex flex-wrap gap-1">
                      {ok ? <Badge tone="ok">Watertight</Badge> : <Badge tone="warn">Invalid mesh</Badge>}
                      {!fits && <Badge tone="warn">Too large</Badge>}
                    </div>
                  </td>
                  <td className="px-2 py-2 text-right">
                    <div className="flex justify-end gap-1">
                      {!fits && p.role === 'bin' && (
                        <IconButton
                          label="Split to fit build plate"
                          onClick={() => {
                            const ids = useDesign.getState().splitToFit(p.compartmentId);
                            if (ids.length) useUi.getState().notify(`Split into ${ids.length} printable pieces`);
                          }}
                        >
                          <Scissors size={14} />
                        </IconButton>
                      )}
                      <IconButton label="Download STL" onClick={() => exportPart(p)}>
                        <Download size={14} />
                      </IconButton>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div className="flex flex-col gap-2 border-t border-line px-6 py-4 sm:flex-row sm:items-center">
        <p className="flex-1 text-[11.5px] leading-relaxed text-muted">
          Print open side up without supports. Each bin is a single closed solid generated with exact CSG; the ZIP includes a parts list with sizes.
        </p>
        <Button disabled={!parts.length} onClick={() => exportAssembled(design, parts)} title="One STL with every part in its position in the drawer">
          <Layers size={14} /> Assembled STL
        </Button>
        <Button variant="primary" disabled={!parts.length} onClick={() => exportZip(design, parts)}>
          <FileArchive size={14} /> Download all parts (.zip)
        </Button>
      </div>
    </div>
  );
}
