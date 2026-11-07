/**
 * Build-plate view inside the export dialog: packs every part onto plates,
 * draws them to scale and estimates print time.
 */
import { AlertTriangle, Clock, Download, Layers, RotateCw } from 'lucide-react';
import { useMemo } from 'react';
import { exportZip } from '../geometry/exporter';
import { TIME_UNCERTAINTY, estimateRange, formatRange, nestParts } from '../geometry/nesting';
import { extrudedVolume, filamentGrams } from '../geometry/parts';
import { safeFileName } from '../geometry/stl';
import type { PartMesh } from '../geometry/types';
import type { Design } from '../model/types';
import { useDesign } from '../store/designStore';
import { useT, useUi, usePalette } from '../store/uiStore';
import { Badge, Button, NumberField, Toggle, cx } from './ui/controls';

const PLATE_PX = 196;
const f1 = (v: number) => String(Math.round(v * 10) / 10);

/** "Compartment 12 – Cup 2·1" → "12·Cup 2·1"; enough to find the part again. */
function shortName(name: string): string {
  const [head, ...rest] = name.split(' – ');
  const n = head.match(/(\d+)\s*$/)?.[1];
  const tail = rest.join(' ');
  if (n && tail) return `${n}·${tail}`;
  if (n) return n;
  return tail || head;
}

export function PlatePlanner({ design, parts }: { design: Design; parts: PartMesh[] }) {
  const spacing = useUi((s) => s.plateSpacing);
  const rotate = useUi((s) => s.plateRotate);
  const printer = design.printer;
  const ui = useUi.getState();
  const actions = useDesign.getState();
  const bv = printer.build;
  const t = useT();
  const c = usePalette();

  const { plates, rejected } = useMemo(
    () => nestParts(parts, bv, { spacing, allowRotation: rotate }),
    [parts, bv, spacing, rotate],
  );
  const times = useMemo(
    () => plates.map((p) => estimateRange(extrudedVolume(p, printer.infill), p.height, printer)),
    [plates, printer],
  );
  const total = times.reduce((s, t) => ({ low: s.low + t.low, high: s.high + t.high, mid: s.mid + t.mid }), { low: 0, high: 0, mid: 0 });
  const grams = parts.reduce((s, p) => s + filamentGrams(p, printer.infill), 0);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-end gap-3 border-b border-line bg-canvas/60 px-6 py-3">
        <div className="flex items-center gap-2">
          <Badge tone="accent">
            <Layers size={12} /> {plates.length === 1 ? t('1 plate') : t('{n} plates', { n: plates.length })}
          </Badge>
          <Badge>
            <Clock size={12} /> {formatRange(total)}
          </Badge>
          <Badge>≈ {Math.round(grams)} g PLA</Badge>
          {rejected.length > 0 && (
            <Badge tone="warn">
              <AlertTriangle size={12} /> {t('{n} too large', { n: rejected.length })}
            </Badge>
          )}
        </div>
        <div className="flex-1" />
        <div className="w-[96px]">
          <NumberField compact label={t('Part spacing')} value={spacing} min={0} max={30} step={1} onChange={ui.setPlateSpacing} />
        </div>
        <div className="pb-0.5">
          <Toggle label={t('Allow rotation')} checked={rotate} onChange={ui.setPlateRotate} />
        </div>
      </div>

      <div className="scroll-thin min-h-0 flex-1 overflow-y-auto px-6 py-4">
        {plates.length === 0 && <p className="py-10 text-center text-[13px] text-muted">{t('Nothing to place yet.')}</p>}
        <div className="flex flex-wrap gap-5">
          {plates.map((plate, i) => (
            <figure key={plate.index} className="m-0">
              <div
                className="relative rounded-lg border border-line-strong shadow-panel"
                style={{
                  width: PLATE_PX,
                  height: (PLATE_PX * bv.y) / bv.x,
                  // Hatched plate, drawn from the palette so it follows the theme.
                  backgroundImage: `repeating-linear-gradient(45deg, ${c.drawerFloor} 0 6px, ${c.drawerBody} 6px 12px)`,
                }}
              >
                {plate.parts.map((p) => {
                  const k = PLATE_PX / bv.x;
                  return (
                    <div
                      key={p.id}
                      title={`${p.name} · ${f1(p.w)} × ${f1(p.d)} × ${f1(p.h)} mm${p.rotated ? ` (${t('rotated')})` : ''}`}
                      className="absolute flex items-center justify-center gap-0.5 overflow-hidden rounded-[3px] border border-accent/45 bg-accent-soft px-0.5"
                      style={{ left: p.x * k, bottom: p.y * k, width: p.w * k, height: p.d * k }}
                    >
                      {p.rotated && <RotateCw size={9} className="shrink-0 text-accent-ink/60" />}
                      {p.w * k > 34 && p.d * k > 13 && (
                        <span className="truncate text-[8.5px] leading-none text-accent-ink/80">{shortName(p.name)}</span>
                      )}
                    </div>
                  );
                })}
              </div>
              <figcaption className="mt-1.5 text-[11px] text-muted">
                <span className="font-medium text-ink">{t('Plate {n}', { n: i + 1 })}</span> ·{' '}
                {plate.parts.length === 1 ? t('1 part') : t('{n} parts', { n: plate.parts.length })} · {Math.round(plate.usage * 100)}%
                · {formatRange(times[i])}
              </figcaption>
            </figure>
          ))}
        </div>

        {rejected.length > 0 && (
          <div className="mt-5 rounded-lg border border-warn/30 bg-warn-soft px-3 py-2.5">
            <div className="flex items-center gap-1.5 text-[12px] font-medium text-warn">
              <AlertTriangle size={13} /> {t('Too large for the {size} mm plate', { size: `${bv.x} × ${bv.y} × ${bv.z}` })}
            </div>
            <ul className="mt-1 space-y-0.5 text-[11.5px] text-warn/90">
              {rejected.map((r) => (
                <li key={r.id}>
                  {r.name} — {f1(r.w)} × {f1(r.d)} × {f1(r.h)} mm
                </li>
              ))}
            </ul>
            <p className="mt-1.5 text-[11px] text-warn/80">{t('Use “Split to fit” on the Parts tab, or raise the build volume.')}</p>
          </div>
        )}

        <details className="mt-5 rounded-lg border border-line px-3 py-2">
          <summary className="cursor-pointer text-[12px] font-medium text-ink-2">{t('Print-time model')}</summary>
          <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <NumberField label={t('Layer height')} value={printer.layerHeight} min={0.04} max={1.2} step={0.02} onChange={(layerHeight) => actions.updatePrinter({ layerHeight })} />
            <NumberField label={t('Nozzle')} value={printer.nozzle} min={0.1} max={2} step={0.1} onChange={(nozzle) => actions.updatePrinter({ nozzle })} />
            <NumberField label={t('Speed')} unit="mm/s" value={printer.speed} min={10} max={1000} step={5} onChange={(speed) => actions.updatePrinter({ speed })} />
            <NumberField label={t('Efficiency')} unit="" value={printer.efficiency} min={0.1} max={1} step={0.05} onChange={(efficiency) => actions.updatePrinter({ efficiency })} hint={t('Share of the nominal speed actually reached')} />
            <NumberField label={t('Infill')} unit="" value={printer.infill} min={0} max={1} step={0.05} onChange={(infill) => actions.updatePrinter({ infill })} hint={t('Density used for the solid core of chunky parts (0.15 = 15%)')} />
          </div>
          <p className="mt-2 text-[11px] leading-relaxed text-muted">
            {t(
              'A rough flow-rate estimate, not a slicer: each part is split into a printed shell and an infilled core, then divided by the extrusion rate with a per-layer allowance. The range is ±{pct}%; slice the files for a real figure.',
              { pct: Math.round(TIME_UNCERTAINTY * 100) },
            )}
          </p>
        </details>
      </div>

      <div className="flex flex-col gap-2 border-t border-line px-6 py-4 sm:flex-row sm:items-center">
        <p className={cx('flex-1 text-[11.5px] leading-relaxed text-muted')}>
          {t('Plates are a packing suggestion — the ZIP contains one STL per part, which you arrange in your slicer.')}
        </p>
        <Button variant="primary" disabled={!parts.length} onClick={() => exportZip(design, parts, safeFileName(design.name))}>
          <Download size={14} /> {t('Download all parts (.zip)')}
        </Button>
      </div>
    </div>
  );
}
