import { ArrowRight, Printer, Ruler } from 'lucide-react';
import { useState } from 'react';
import { gridRects } from '../layout/ops';
import { DEFAULT_DRAWER, DEFAULT_PRINT, DEFAULT_PRINTER_ID, LIMITS, TOP_CLEARANCE, usableArea } from '../model/defaults';
import { findPrinter, printersByBrand } from '../model/printers';
import { type StartLayout, useDesign } from '../store/designStore';
import { useUi } from '../store/uiStore';
import { Logo } from './TopBar';
import { Button, Modal, NumberField, cx } from './ui/controls';

const LAYOUTS: { id: string; label: string; layout: StartLayout }[] = [
  { id: 'empty', label: 'Blank', layout: { kind: 'empty' } },
  { id: '2x2', label: '2 × 2', layout: { kind: 'grid', rows: 2, cols: 2 } },
  { id: '2x3', label: '2 × 3', layout: { kind: 'grid', rows: 2, cols: 3 } },
  { id: '3x4', label: '3 × 4', layout: { kind: 'grid', rows: 3, cols: 4 } },
];

export function SetupDialog() {
  const hasProject = useDesign((s) => s.hasProject);
  const open = useUi((s) => s.setupOpen) || !hasProject;
  const current = useDesign((s) => s.design);
  const [name, setName] = useState(hasProject ? 'Untitled organiser' : 'My drawer organiser');
  const [drawer, setDrawer] = useState(hasProject ? current.drawer : DEFAULT_DRAWER);
  const [bottom, setBottom] = useState(hasProject ? current.print.bottomThickness : DEFAULT_PRINT.bottomThickness);
  const [layout, setLayout] = useState('2x3');
  const [printerId, setPrinterId] = useState(hasProject ? current.printer.presetId : DEFAULT_PRINTER_ID);
  const printer = findPrinter(printerId);
  const usable = usableArea(drawer);
  const close = () => useUi.getState().setSetupOpen(false);

  const create = () => {
    const l = LAYOUTS.find((x) => x.id === layout)!.layout;
    useDesign.getState().newDesign(drawer, { bottomThickness: bottom }, l, name.trim() || 'Untitled organiser', printerId);
    useUi.getState().select(null);
    close();
  };

  const preview = LAYOUTS.find((x) => x.id === layout)!.layout;
  const rects = preview.kind === 'grid' ? gridRects(preview.rows, preview.cols, { width: usable.width, depth: usable.depth }) : [];
  const vw = 320;
  const scale = Math.min(vw / drawer.width, 200 / drawer.depth);

  return (
    <Modal open={open} onClose={hasProject ? close : undefined} width="max-w-[880px]">
      <div className="grid max-h-[92vh] overflow-y-auto md:grid-cols-[1fr_1.05fr]">
        <div className="flex flex-col p-6 sm:p-8">
          <Logo />
          <h1 className="mt-6 text-[22px] font-semibold tracking-tight text-ink">{hasProject ? 'New organiser' : 'Design an organiser that fits perfectly.'}</h1>
          <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted">
            Measure the <em>inside</em> of your drawer. We’ll work out the usable area and generate print-ready bins.
          </p>

          <div className="mt-6 space-y-3">
            <label className="block">
              <span className="mb-1 block text-[11px] font-medium text-muted">Design name</span>
              <input
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="h-8 w-full rounded-lg border border-line px-2.5 text-[13px] outline-none focus:border-accent focus:ring-3 focus:ring-accent/12"
              />
            </label>
            <div className="grid grid-cols-3 gap-2">
              <NumberField label="Width" value={drawer.width} min={LIMITS.drawer.min} max={LIMITS.drawer.max} onChange={(width) => setDrawer({ ...drawer, width })} />
              <NumberField label="Depth" value={drawer.depth} min={LIMITS.drawer.min} max={LIMITS.drawer.max} onChange={(depth) => setDrawer({ ...drawer, depth })} />
              <NumberField label="Height" value={drawer.height} min={LIMITS.drawerHeight.min} max={LIMITS.drawerHeight.max} onChange={(height) => setDrawer({ ...drawer, height })} />
            </div>
            <div className="grid grid-cols-2 gap-2">
              <NumberField label="Wall clearance" value={drawer.clearance} step={0.5} min={LIMITS.clearance.min} max={LIMITS.clearance.max} onChange={(clearance) => setDrawer({ ...drawer, clearance })} hint="Gap between organiser and drawer walls" />
              <NumberField label="Bottom thickness" value={bottom} step={0.2} min={LIMITS.bottom.min} max={LIMITS.bottom.max} onChange={setBottom} hint="Floor thickness of every bin" />
            </div>
          </div>

          <div className="mt-3">
            <label className="block">
              <span className="mb-1 block text-[11px] font-medium text-muted">Your printer</span>
              <select
                value={printerId}
                onChange={(e) => setPrinterId(e.target.value)}
                className="h-8 w-full rounded-lg border border-line bg-surface px-2 text-[13px] outline-none focus:border-accent focus:ring-3 focus:ring-accent/12"
              >
                {printersByBrand().map((g) => (
                  <optgroup key={g.brand} label={g.brand}>
                    {g.items.map((m) => (
                      <option key={m.id} value={m.id}>
                        {m.model} · {m.build.x}×{m.build.y}×{m.build.z} mm
                      </option>
                    ))}
                  </optgroup>
                ))}
                <option value="custom">Other / custom</option>
              </select>
            </label>
          </div>

          <div className="mt-5">
            <div className="mb-1.5 text-[11px] font-medium text-muted">Start from</div>
            <div className="grid grid-cols-4 gap-1.5">
              {LAYOUTS.map((l) => (
                <button
                  key={l.id}
                  type="button"
                  onClick={() => setLayout(l.id)}
                  className={cx(
                    'h-9 rounded-lg border text-[12px] font-medium transition-all',
                    layout === l.id ? 'border-ink bg-ink text-white' : 'border-line text-ink-2 hover:border-line-strong',
                  )}
                >
                  {l.label}
                </button>
              ))}
            </div>
          </div>

          <div className="mt-6 flex items-center gap-2 pt-2 md:mt-auto">
            {hasProject && (
              <Button variant="ghost" onClick={close}>
                Cancel
              </Button>
            )}
            <Button variant="primary" className="ml-auto h-9 px-4" onClick={create} disabled={usable.width < 20 || usable.depth < 20}>
              {hasProject ? 'Create design' : 'Start designing'} <ArrowRight size={14} />
            </Button>
          </div>
        </div>

        <div className="dot-grid relative hidden flex-col items-center justify-center border-l border-line bg-canvas p-8 md:flex">
          <svg width={vw + 40} height={drawer.depth * scale + 70} className="overflow-visible">
            <g transform="translate(20 30)">
              <rect x={-6} y={-6} width={drawer.width * scale + 12} height={drawer.depth * scale + 12} rx={8} fill="#e9e8e4" />
              <rect width={drawer.width * scale} height={drawer.depth * scale} rx={3} fill="#fbfbfa" stroke="#cfcec9" />
              <rect
                x={drawer.clearance * scale}
                y={drawer.clearance * scale}
                width={usable.width * scale}
                height={usable.depth * scale}
                fill="none"
                stroke="#a1a1aa"
                strokeDasharray="3 3"
              />
              {rects.map((r, i) => (
                <rect
                  key={i}
                  x={(drawer.clearance + r.x + 0.8) * scale}
                  y={(drawer.clearance + r.y + 0.8) * scale}
                  width={Math.max(0, (r.w - 1.6) * scale)}
                  height={Math.max(0, (r.d - 1.6) * scale)}
                  rx={3}
                  fill="#fff"
                  stroke="#c9c8c3"
                  className="animate-fade-in"
                />
              ))}
              <text x={(drawer.width * scale) / 2} y={-14} textAnchor="middle" fontSize={11} fill="#71717a" className="num font-mono">
                {drawer.width} mm
              </text>
              <text x={-12} y={(drawer.depth * scale) / 2} textAnchor="middle" fontSize={11} fill="#71717a" className="num font-mono" transform={`rotate(-90 -12 ${(drawer.depth * scale) / 2})`}>
                {drawer.depth} mm
              </text>
            </g>
          </svg>
          <div className="mt-5 w-full max-w-[320px] rounded-xl border border-line bg-surface p-3 shadow-panel">
            <div className="flex items-center gap-2 text-[12px] font-medium text-ink">
              <Ruler size={14} className="text-accent" /> Usable design area
            </div>
            <div className="num mt-1.5 font-mono text-[15px] text-ink">
              {usable.width} × {usable.depth} mm
            </div>
            <div className="num mt-0.5 text-[11.5px] text-muted">Bins {Math.max(LIMITS.binHeight.min, drawer.height - TOP_CLEARANCE)} mm tall · {TOP_CLEARANCE} mm headroom</div>
            {printer && (
              <div className="mt-2 flex items-center gap-1.5 border-t border-line pt-2 text-[11.5px] text-muted">
                <Printer size={12} className="text-accent" />
                <span className="truncate">
                  {printer.brand} {printer.model}
                </span>
                <span className="num ml-auto whitespace-nowrap text-faint">
                  {printer.build.x}×{printer.build.y}×{printer.build.z}
                </span>
              </div>
            )}
          </div>
        </div>
      </div>
    </Modal>
  );
}
