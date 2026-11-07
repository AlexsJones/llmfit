/**
 * Dialog for creating the outline used by the "Custom shape insert" module:
 * either click points to draw one, or trace the silhouette of a photo.
 *
 * Images are processed entirely in the browser (canvas → luminance → trace);
 * nothing is uploaded.
 */
import { Check, Image as ImageIcon, PenLine, RotateCcw, Trash2, Undo2, Upload, X } from 'lucide-react';
import { useCallback, useEffect, useRef, useState } from 'react';
import { MAX_OUTLINE_POINTS, type Pt, decodeOutline, encodeOutline, polygonArea, toGray, traceOutlineDetailed } from '../geometry/outline';
import { useT, usePalette } from '../store/uiStore';
import { Button, IconButton, Modal, NumberField, Segmented, Toggle, cx } from './ui/controls';

const SIZE = 420;
/** Images are downscaled before tracing; this keeps tracing fast and the outline smooth. */
const TRACE_MAX = 420;

const PRESETS: { label: string; points: Pt[] }[] = [
  { label: 'Circle', points: ring(48, () => 0.5) },
  { label: 'Rounded bar', points: bar() },
  { label: 'Hex', points: ring(6, () => 0.5) },
  { label: 'Teardrop', points: ring(40, (a) => 0.28 + 0.22 * Math.cos(a / 2) ** 2) },
];

function ring(n: number, radius: (a: number) => number): Pt[] {
  return Array.from({ length: n }, (_, i) => {
    const a = (Math.PI * 2 * i) / n;
    const r = radius(a);
    return { x: 0.5 + Math.cos(a) * r, y: 0.5 + Math.sin(a) * r };
  });
}
function bar(): Pt[] {
  const pts: Pt[] = [];
  for (let i = 0; i <= 16; i++) {
    const a = Math.PI / 2 - (Math.PI * i) / 16;
    pts.push({ x: 0.8 + Math.cos(a) * 0.2, y: 0.5 - Math.sin(a) * 0.2 });
  }
  for (let i = 0; i <= 16; i++) {
    const a = -Math.PI / 2 - (Math.PI * i) / 16;
    pts.push({ x: 0.2 + Math.cos(a) * 0.2, y: 0.5 - Math.sin(a) * 0.2 });
  }
  return pts;
}

export function OutlineEditor({
  value,
  onCancel,
  onApply,
}: {
  value: string;
  onCancel: () => void;
  onApply: (encoded: string) => void;
}) {
  const t = useT();
  const pal = usePalette();
  const [mode, setMode] = useState<'draw' | 'photo'>('draw');
  const [points, setPoints] = useState<Pt[]>(() => decodeOutline(value));
  const [history, setHistory] = useState<Pt[][]>([]);
  const [threshold, setThreshold] = useState(0.55);
  const [smooth, setSmooth] = useState(2.5);
  const [invert, setInvert] = useState(false);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  /** Trace result in image space, drawn on top of the photo so the two line up. */
  const [overlay, setOverlay] = useState<Pt[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const push = (next: Pt[]) => {
    setHistory((h) => [...h.slice(-40), points]);
    setPoints(next);
  };

  // ---- image tracing ----------------------------------------------------------
  const retrace = useCallback(
    (img: HTMLImageElement, cut: number, tol: number, inv: boolean) => {
      const scale = Math.min(1, TRACE_MAX / Math.max(img.naturalWidth, img.naturalHeight));
      const w = Math.max(8, Math.round(img.naturalWidth * scale));
      const h = Math.max(8, Math.round(img.naturalHeight * scale));
      const cv = document.createElement('canvas');
      cv.width = w;
      cv.height = h;
      const ctx = cv.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;
      ctx.drawImage(img, 0, 0, w, h);
      const gray = toGray(ctx.getImageData(0, 0, w, h).data, w, h);
      const traced = traceOutlineDetailed(gray, { threshold: cut, tolerance: tol, invert: inv });
      if (!traced.points.length) {
        setError(t('No clear shape found – try the threshold slider or invert.'));
        setPoints([]);
        setOverlay([]);
        return;
      }
      setError(null);
      setPoints(traced.points);
      setOverlay(traced.inImage);
    },
    [],
  );

  useEffect(() => {
    if (image) retrace(image, threshold, smooth, invert);
  }, [image, threshold, smooth, invert, retrace]);

  const loadFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      setError(t('Please choose an image file.'));
      return;
    }
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      setImage(img);
      setMode('photo');
      URL.revokeObjectURL(url);
    };
    img.onerror = () => {
      setError(t('That image could not be read.'));
      URL.revokeObjectURL(url);
    };
    img.src = url;
  };

  // ---- canvas drawing ----------------------------------------------------------
  useEffect(() => {
    const cv = canvasRef.current;
    const ctx = cv?.getContext('2d');
    if (!cv || !ctx) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    cv.width = SIZE * dpr;
    cv.height = SIZE * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, SIZE, SIZE);

    ctx.fillStyle = pal.drawerFloor;
    ctx.fillRect(0, 0, SIZE, SIZE);
    ctx.strokeStyle = pal.grid === '#ffffff' ? 'rgba(255,255,255,0.07)' : 'rgba(24,24,27,0.06)';
    ctx.lineWidth = 1;
    for (let i = 1; i < 8; i++) {
      const v = (SIZE * i) / 8;
      ctx.beginPath();
      ctx.moveTo(v, 0);
      ctx.lineTo(v, SIZE);
      ctx.moveTo(0, v);
      ctx.lineTo(SIZE, v);
      ctx.stroke();
    }
    const photo = image && mode === 'photo';
    let place: ((p: Pt) => Pt) | null = null;
    if (photo) {
      const k = Math.min(SIZE / image.naturalWidth, SIZE / image.naturalHeight);
      const w = image.naturalWidth * k;
      const h = image.naturalHeight * k;
      const ox = (SIZE - w) / 2;
      const oy = (SIZE - h) / 2;
      ctx.globalAlpha = 0.4;
      ctx.drawImage(image, ox, oy, w, h);
      ctx.globalAlpha = 1;
      place = (p) => ({ x: ox + p.x * w, y: oy + p.y * h });
    }
    const shown = photo ? overlay : points;
    if (shown.length >= 2) {
      // Outlines are normalised to their own bounds, so draw them inset with a margin.
      const m = 30;
      const s = SIZE - 2 * m;
      const at = place ?? ((p: Pt) => ({ x: m + p.x * s, y: m + p.y * s }));
      ctx.beginPath();
      shown.forEach((p, i) => {
        const q = at(p);
        return i ? ctx.lineTo(q.x, q.y) : ctx.moveTo(q.x, q.y);
      });
      ctx.closePath();
      ctx.fillStyle = 'rgba(47,169,124,0.12)';
      ctx.fill();
      ctx.strokeStyle = pal.accent;
      ctx.lineWidth = 1.75;
      ctx.stroke();
      if (mode === 'draw') {
        points.forEach((p, i) => {
          ctx.beginPath();
          ctx.arc(m + p.x * s, m + p.y * s, i === drag ? 6 : 4, 0, Math.PI * 2);
          ctx.fillStyle = i === 0 ? pal.accent : pal.labelPlate;
          ctx.fill();
          ctx.strokeStyle = pal.accent;
          ctx.lineWidth = 1.5;
          ctx.stroke();
        });
      }
    }
  }, [points, overlay, image, mode, drag]);

  const toNorm = (e: React.PointerEvent) => {
    const r = canvasRef.current!.getBoundingClientRect();
    const m = 30;
    const s = SIZE - 2 * m;
    return { x: ((e.clientX - r.left) * (SIZE / r.width) - m) / s, y: ((e.clientY - r.top) * (SIZE / r.height) - m) / s };
  };

  const onDown = (e: React.PointerEvent) => {
    if (mode !== 'draw') return;
    const p = toNorm(e);
    const m = 30;
    const s = SIZE - 2 * m;
    const hit = points.findIndex((q) => Math.hypot((q.x - p.x) * s, (q.y - p.y) * s) < 9);
    if (hit >= 0) {
      setDrag(hit);
      canvasRef.current?.setPointerCapture(e.pointerId);
      return;
    }
    if (points.length >= MAX_OUTLINE_POINTS) return;
    push([...points, p]);
  };
  const onMove = (e: React.PointerEvent) => {
    if (drag === null) return;
    const p = toNorm(e);
    setPoints((cur) => cur.map((q, i) => (i === drag ? p : q)));
  };
  const onUp = () => setDrag(null);

  const normalised = () => {
    if (points.length < 3) return '';
    const xs = points.map((p) => p.x);
    const ys = points.map((p) => p.y);
    const minX = Math.min(...xs);
    const minY = Math.min(...ys);
    const k = 1 / Math.max(1e-6, Math.max(Math.max(...xs) - minX, Math.max(...ys) - minY));
    const out = points.map((p) => ({ x: (p.x - minX) * k, y: (p.y - minY) * k }));
    return encodeOutline(polygonArea(out) < 0 ? out.reverse() : out);
  };

  const valid = points.length >= 3 && Math.abs(polygonArea(points)) > 1e-4;

  return (
    <Modal open onClose={onCancel} width="max-w-[860px]">
      <div className="flex max-h-[92vh] flex-col">
        <div className="flex items-start gap-3 border-b border-line px-6 py-4">
          <div className="flex-1">
            <h2 className="text-[16px] font-semibold tracking-tight text-ink">{t('Shape outline')}</h2>
            <p className="mt-0.5 text-[12.5px] text-muted">{t('Draw the shape, or trace it from a photo of the object.')}</p>
          </div>
          <IconButton label={t('Close')} onClick={onCancel}>
            <X size={16} />
          </IconButton>
        </div>

        <div className="grid min-h-0 flex-1 gap-4 overflow-y-auto p-5 sm:grid-cols-[auto_1fr]">
          <div className="mx-auto">
            <canvas
              ref={canvasRef}
              style={{ width: SIZE, height: SIZE }}
              className={cx('touch-none rounded-xl border border-line shadow-panel', mode === 'draw' && 'cursor-crosshair')}
              onPointerDown={onDown}
              onPointerMove={onMove}
              onPointerUp={onUp}
              onPointerCancel={onUp}
            />
            <p className="mt-2 text-center text-[11px] text-faint">
              {mode === 'draw' ? t('Click to add points · drag a point to move it') : t('Adjust the threshold until the traced outline matches')}
            </p>
          </div>

          <div className="min-w-0">
            <Segmented
              className="w-full"
              value={mode}
              onChange={(m) => setMode(m)}
              options={[
                { value: 'draw', label: <><PenLine size={13} /> {t('Draw')}</> },
                { value: 'photo', label: <><ImageIcon size={13} /> {t('From photo')}</> },
              ]}
            />

            {mode === 'draw' ? (
              <div className="mt-4 space-y-3">
                <div className="flex flex-wrap gap-1.5">
                  <Button size="sm" disabled={!history.length} onClick={() => { setPoints(history[history.length - 1]); setHistory((h) => h.slice(0, -1)); }}>
                    <Undo2 size={12} /> {t('Undo')}
                  </Button>
                  <Button size="sm" disabled={!points.length} onClick={() => push([])}>
                    <Trash2 size={12} /> {t('Clear')}
                  </Button>
                  <Button size="sm" disabled={points.length < 3} onClick={() => push([...points].reverse())} title={t('Reverse the point order')}>
                    <RotateCcw size={12} /> {t('Reverse')}
                  </Button>
                </div>
                <div>
                  <div className="mb-1.5 text-[11px] font-medium text-muted">{t('Start from a shape')}</div>
                  <div className="grid grid-cols-2 gap-1.5">
                    {PRESETS.map((p) => (
                      <Button key={p.label} size="sm" onClick={() => push(p.points)}>
                        {t(p.label)}
                      </Button>
                    ))}
                  </div>
                </div>
                <p className="text-[11.5px] leading-relaxed text-muted">
                  {t('The outline is placed at its measured size, so only its proportions matter for the drawing. Use the module’s clearance setting to add room around the object.')}
                </p>
              </div>
            ) : (
              <div className="mt-4 space-y-3">
                <Button className="w-full" onClick={() => fileRef.current?.click()}>
                  <Upload size={13} /> {image ? t('Choose another photo') : t('Choose a photo')}
                </Button>
                <input
                  ref={fileRef}
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    e.target.value = '';
                    if (f) loadFile(f);
                  }}
                />
                <p className="text-[11.5px] leading-relaxed text-muted">
                  {t('Photograph the object on a plain, contrasting background, straight from above. The largest shape is traced.')}
                </p>
                <div className={cx(!image && 'pointer-events-none opacity-40')}>
                  <NumberField label={t('Threshold')} unit="" value={threshold} min={0.05} max={0.95} step={0.05} onChange={setThreshold} hint={t('Lower keeps only very dark pixels')} />
                  <div className="mt-2">
                    <NumberField label={t('Smoothing')} unit="px" value={smooth} min={0.5} max={12} step={0.5} onChange={setSmooth} />
                  </div>
                  <div className="mt-2">
                    <Toggle label={t('Light object on dark background')} checked={invert} onChange={setInvert} />
                  </div>
                </div>
              </div>
            )}

            <div className="mt-4 rounded-lg bg-subtle px-3 py-2 text-[11.5px]">
              {error ? (
                <span className="text-warn">{error}</span>
              ) : valid ? (
                <span className="text-muted">
                  <span className="num font-medium text-ink">{points.length}</span> {t('points · ready')}
                </span>
              ) : (
                <span className="text-muted">{t('Add at least three points.')}</span>
              )}
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end gap-2 border-t border-line px-6 py-4">
          <Button variant="ghost" onClick={onCancel}>
            {t('Cancel')}
          </Button>
          <Button variant="primary" disabled={!valid} onClick={() => onApply(normalised())}>
            <Check size={14} /> {t('Use this shape')}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
