/**
 * Top-down 2D layout editor.
 *
 * Renders the drawer, usable area and compartments as SVG in millimetre
 * coordinates and translates pointer gestures into layout operations
 * (move, resize, divider drag, pan, zoom). All geometric rules (snapping,
 * collisions, bounds) live in `layout/ops.ts`; this component only maps
 * screen space to millimetres.
 */
import { AlertTriangle, Maximize, Minus, Plus } from 'lucide-react';
import { type PointerEvent as ReactPointerEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useGeometry } from '../hooks/geometry';
import {
  type Edge,
  edgeLimit,
  expandToFill,
  overlaps,
  rectOf,
  resizeEdge,
  snapLines,
  snapPoint,
  snapSpan,
  spanRect,
  sweepMove,
  validateLayout,
} from '../layout/ops';
import { MIN_COMPARTMENT, usableArea } from '../model/defaults';
import type { Compartment, Rect } from '../model/types';
import { PITCH } from '../geometry/gridfinity';
import { resolveBinSpec, resolveParams } from '../geometry/parts';
import { getModule } from '../modules/registry';
import type { PreviewShape, PreviewTone } from '../modules/types';
import { useDesign } from '../store/designStore';
import { useUi } from '../store/uiStore';
import { IconButton, cx } from './ui/controls';

type Gesture =
  | { type: 'pan'; startX: number; startY: number; panX: number; panY: number; moved: boolean }
  | { type: 'move'; id: string; start: { x: number; y: number }; orig: Rect; began: boolean }
  | { type: 'group'; ids: string[]; start: { x: number; y: number }; began: boolean; last: { dx: number; dy: number } }
  | { type: 'marquee'; start: { x: number; y: number }; add: boolean; base: string[] }
  | { type: 'resize'; id: string; edges: Edge[]; start: { x: number; y: number }; orig: Rect; began: boolean }
  | { type: 'divider'; id: string; dividerId: string; start: { x: number; y: number }; orig: number; began: boolean };

const SNAP_PX = 7;
const DRAG_THRESHOLD_PX = 3;
const r1 = (v: number) => Math.round(v * 10) / 10;
const fmt = (v: number) => String(r1(v));

export function Editor2D({ compact = false }: { compact?: boolean }) {
  const design = useDesign((s) => s.design);
  const { drawer, print, compartments } = design;
  const usable = usableArea(drawer);
  const bounds = useMemo(() => ({ width: usable.width, depth: usable.depth }), [usable.width, usable.depth]);
  const c0 = drawer.clearance;

  const selectedId = useUi((s) => s.selectedId);
  const selectedIds = useUi((s) => s.selectedIds);
  const selectedDividerId = useUi((s) => s.selectedDividerId);
  const hoverId = useUi((s) => s.hoverId);
  const gridSnap = useUi((s) => s.gridSnap);
  const gridfinity = useUi((s) => s.gridfinity);
  const edgeSnap = useUi((s) => s.edgeSnap);
  const { byCompartment } = useGeometry();

  const wrapRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [size, setSize] = useState({ w: 800, h: 600 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [guides, setGuides] = useState<{ x: number | null; y: number | null }>({ x: null, y: null });
  const gesture = useRef<Gesture | null>(null);
  const [activeGesture, setActiveGesture] = useState<Gesture['type'] | null>(null);
  const [marquee, setMarquee] = useState<Rect | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setSize({ w: e.contentRect.width, h: e.contentRect.height }));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ---- view transform -------------------------------------------------------
  const pad = compact ? 44 : 64;
  const base = Math.max(0.05, Math.min((size.w - 2 * pad) / drawer.width, (size.h - 2 * pad) / drawer.depth));
  const s = base * zoom;
  const tx = (size.w - drawer.width * s) / 2 + pan.x;
  const ty = (size.h - drawer.depth * s) / 2 + pan.y;
  const px = (n: number) => n / s; // screen px → mm

  const toUsable = useCallback(
    (clientX: number, clientY: number) => {
      const rect = svgRef.current!.getBoundingClientRect();
      return { x: (clientX - rect.left - tx) / s - c0, y: (clientY - rect.top - ty) / s - c0 };
    },
    [tx, ty, s, c0],
  );

  const fit = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const zoomAt = useCallback(
    (factor: number, clientX?: number, clientY?: number) => {
      const rect = svgRef.current?.getBoundingClientRect();
      const cxp = clientX !== undefined && rect ? clientX - rect.left : size.w / 2;
      const cyp = clientY !== undefined && rect ? clientY - rect.top : size.h / 2;
      const nextZoom = Math.min(12, Math.max(0.4, zoom * factor));
      const s2 = base * nextZoom;
      const mmX = (cxp - tx) / s;
      const mmY = (cyp - ty) / s;
      setZoom(nextZoom);
      setPan({ x: cxp - mmX * s2 - (size.w - drawer.width * s2) / 2, y: cyp - mmY * s2 - (size.h - drawer.depth * s2) / 2 });
    },
    [zoom, base, tx, ty, s, size, drawer.width, drawer.depth],
  );

  useEffect(() => {
    const el = svgRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey || Math.abs(e.deltaY) > Math.abs(e.deltaX)) {
        zoomAt(Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0015)), e.clientX, e.clientY);
      } else {
        setPan((p) => ({ x: p.x - e.deltaX, y: p.y - e.deltaY }));
      }
    };
    el.addEventListener('wheel', onWheel, { passive: false });
    return () => el.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  // ---- gestures --------------------------------------------------------------
  const store = useDesign.getState;
  const ui = useUi.getState;

  const othersOf = (id: string) => store().design.compartments.filter((c) => c.id !== id).map(rectOf);

  const beginOnce = (g: Gesture & { began: boolean }) => {
    if (!g.began) {
      g.began = true;
      store().beginGesture();
    }
  };

  const onPointerDownBackground = (e: ReactPointerEvent) => {
    if (e.button !== 0 && e.button !== 1 && e.button !== 2) return;
    svgRef.current?.setPointerCapture(e.pointerId);
    // Left button on empty space draws a selection rectangle; middle button, right button
    // and Alt-drag pan. Shift keeps the existing selection.
    if (e.button === 0 && !e.altKey) {
      gesture.current = { type: 'marquee', start: toUsable(e.clientX, e.clientY), add: e.shiftKey, base: e.shiftKey ? ui().selectedIds : [] };
      setActiveGesture('marquee');
      return;
    }
    gesture.current = { type: 'pan', startX: e.clientX, startY: e.clientY, panX: pan.x, panY: pan.y, moved: false };
    setActiveGesture('pan');
  };

  const onPointerDownCompartment = (e: ReactPointerEvent, c: Compartment) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    if (e.altKey) {
      onPointerDownBackground(e);
      return;
    }
    svgRef.current?.setPointerCapture(e.pointerId);
    if (e.shiftKey) {
      ui().toggleSelect(c.id);
      gesture.current = null;
      return;
    }
    const current = ui().selectedIds;
    if (current.length > 1 && current.includes(c.id)) {
      gesture.current = { type: 'group', ids: current, start: toUsable(e.clientX, e.clientY), began: false, last: { dx: 0, dy: 0 } };
      setActiveGesture('group');
      return;
    }
    ui().select(c.id);
    gesture.current = { type: 'move', id: c.id, start: toUsable(e.clientX, e.clientY), orig: rectOf(c), began: false };
    setActiveGesture('move');
  };

  const onPointerDownHandle = (e: ReactPointerEvent, c: Compartment, edges: Edge[]) => {
    if (e.button !== 0 || e.altKey) return;
    e.stopPropagation();
    svgRef.current?.setPointerCapture(e.pointerId);
    ui().select(c.id);
    gesture.current = { type: 'resize', id: c.id, edges, start: toUsable(e.clientX, e.clientY), orig: rectOf(c), began: false };
    setActiveGesture('resize');
  };

  const onPointerDownDivider = (e: ReactPointerEvent, c: Compartment, dividerId: string, position: number) => {
    if (e.button !== 0) return;
    e.stopPropagation();
    svgRef.current?.setPointerCapture(e.pointerId);
    ui().select(c.id, dividerId);
    gesture.current = { type: 'divider', id: c.id, dividerId, start: toUsable(e.clientX, e.clientY), orig: position, began: false };
    setActiveGesture('divider');
  };

  const onPointerMove = (e: ReactPointerEvent) => {
    const g = gesture.current;
    if (!g) return;
    if (g.type === 'pan') {
      const dx = e.clientX - g.startX;
      const dy = e.clientY - g.startY;
      if (Math.hypot(dx, dy) > DRAG_THRESHOLD_PX) g.moved = true;
      if (g.moved) setPan({ x: g.panX + dx, y: g.panY + dy });
      return;
    }
    const p = toUsable(e.clientX, e.clientY);
    const dx = p.x - g.start.x;
    const dy = p.y - g.start.y;
    if (g.type === 'marquee') {
      const box = spanRect(g.start.x, g.start.y, p.x, p.y);
      setMarquee(box);
      const hit = store()
        .design.compartments.filter((c) => overlaps(box, c, 0.001))
        .map((c) => c.id);
      ui().selectMany(g.add ? [...new Set([...g.base, ...hit])] : hit);
      return;
    }
    if (!g.began && Math.hypot(dx, dy) * s < DRAG_THRESHOLD_PX) return;
    const threshold = px(SNAP_PX);
    const grid = gridfinity ? PITCH : gridSnap;

    if (g.type === 'group') {
      beginOnce(g);
      const grid = gridSnap;
      const snapped = grid > 0 ? { x: Math.round(dx / grid) * grid, y: Math.round(dy / grid) * grid } : { x: dx, y: dy };
      // moveMany works in deltas, so only apply what has not been applied yet.
      store().moveMany(g.ids, snapped.x - g.last.dx, snapped.y - g.last.dy, true);
      g.last = { dx: snapped.x, dy: snapped.y };
      return;
    }
    if (g.type === 'move') {
      beginOnce(g);
      const others = othersOf(g.id);
      const lines = edgeSnap ? snapLines(others, bounds) : { xs: [0, bounds.width], ys: [0, bounds.depth] };
      const sx = snapSpan(g.orig.x + dx, g.orig.w, lines.xs, threshold, grid);
      const sy = snapSpan(g.orig.y + dy, g.orig.d, lines.ys, threshold, grid);
      const cur = rectOf(store().design.compartments.find((c) => c.id === g.id)!);
      const next = sweepMove(cur, sx.value, sy.value, others, bounds);
      store().setCompartmentRect(g.id, next, true);
      setGuides({ x: sx.guide !== null && Math.abs(next.x - sx.value) < 0.01 ? sx.guide : null, y: sy.guide !== null && Math.abs(next.y - sy.value) < 0.01 ? sy.guide : null });
    } else if (g.type === 'resize') {
      beginOnce(g);
      const others = othersOf(g.id);
      const lines = edgeSnap ? snapLines(others, bounds) : { xs: [0, bounds.width], ys: [0, bounds.depth] };
      let r = { ...g.orig };
      const gs: { x: number | null; y: number | null } = { x: null, y: null };
      for (const edge of g.edges) {
        const horizontal = edge === 'left' || edge === 'right';
        const raw =
          edge === 'left' ? g.orig.x + dx : edge === 'right' ? g.orig.x + g.orig.w + dx : edge === 'top' ? g.orig.y + dy : g.orig.y + g.orig.d + dy;
        const sn = snapPoint(raw, horizontal ? lines.xs : lines.ys, threshold, grid);
        r = resizeEdge(r, edge, sn.value, others, bounds);
        if (sn.guide !== null) {
          const actual = edge === 'left' ? r.x : edge === 'right' ? r.x + r.w : edge === 'top' ? r.y : r.y + r.d;
          if (Math.abs(actual - sn.guide) < 0.01) gs[horizontal ? 'x' : 'y'] = sn.guide;
        }
      }
      store().setCompartmentRect(g.id, r, true);
      setGuides(gs);
    } else if (g.type === 'divider') {
      beginOnce(g);
      const c = store().design.compartments.find((x) => x.id === g.id);
      const dv = c?.dividers.find((x) => x.id === g.dividerId);
      if (!c || !dv) return;
      const vertical = dv.orientation === 'vertical';
      const span = vertical ? c.w : c.d;
      const raw = g.orig + (vertical ? dx : dy);
      const sn = snapPoint(raw, [span / 2, span / 3, (2 * span) / 3, span / 4, (3 * span) / 4], threshold, grid);
      store().updateDivider(g.id, g.dividerId, sn.value, true);
      setGuides(sn.guide !== null ? (vertical ? { x: c.x + sn.value, y: null } : { x: null, y: c.y + sn.value }) : { x: null, y: null });
    }
  };

  const onPointerUp = (e: ReactPointerEvent) => {
    const g = gesture.current;
    gesture.current = null;
    setActiveGesture(null);
    setGuides({ x: null, y: null });
    setMarquee(null);
    svgRef.current?.releasePointerCapture?.(e.pointerId);
    if (!g) return;
    if (g.type === 'pan') {
      if (!g.moved) ui().select(null);
      return;
    }
    if (g.type === 'marquee') {
      // A click with no drag clears the selection.
      if (!marquee && !g.add) ui().select(null);
      return;
    }
    if (g.began) store().endGesture();
  };

  const onDoubleClick = (e: React.MouseEvent) => {
    // Double-click on empty space: create a compartment filling the free region.
    const p = toUsable(e.clientX, e.clientY);
    if (p.x < 0 || p.y < 0 || p.x > bounds.width || p.y > bounds.depth) return;
    const half = MIN_COMPARTMENT / 2;
    const seed = {
      x: Math.min(Math.max(p.x - half, 0), bounds.width - MIN_COMPARTMENT),
      y: Math.min(Math.max(p.y - half, 0), bounds.depth - MIN_COMPARTMENT),
      w: MIN_COMPARTMENT,
      d: MIN_COMPARTMENT,
    };
    const others = store().design.compartments.map(rectOf);
    if (others.some((o) => overlaps(seed, o))) return;
    const r = expandToFill(seed, others, bounds);
    const id = store().addCompartmentRect(r);
    if (id) ui().select(id);
  };

  // ---- derived display data ---------------------------------------------------
  const issues = useMemo(() => validateLayout(compartments, bounds), [compartments, bounds]);
  const issueIds = new Set(issues.map((i) => i.compartmentId));
  const selected = compartments.find((c) => c.id === selectedId) ?? null;
  const gap = print.fitGap;
  const radius = print.cornerStyle === 'rounded' ? print.cornerRadius : 0;
  const fontPx = (n: number) => px(n);

  const gridMinor = 10;
  const gridMajor = 50;
  const showMinor = gridMinor * s >= 6;

  return (
    <div ref={wrapRef} className="relative h-full w-full overflow-hidden bg-canvas select-none">
      <svg
        ref={svgRef}
        width={size.w}
        height={size.h}
        className={cx('block touch-none', activeGesture === 'pan' ? 'cursor-grabbing' : activeGesture === 'marquee' ? 'cursor-crosshair' : 'cursor-default')}
        onPointerDown={onPointerDownBackground}
        onContextMenu={(e) => e.preventDefault()}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onDoubleClick={onDoubleClick}
      >
        <defs>
          <pattern id="grid-minor" width={gridMinor} height={gridMinor} patternUnits="userSpaceOnUse" x={c0} y={c0}>
            <path d={`M ${gridMinor} 0 L 0 0 0 ${gridMinor}`} fill="none" stroke="#18181b" strokeOpacity={0.05} strokeWidth={px(1)} />
          </pattern>
          <pattern id="grid-major" width={gridMajor} height={gridMajor} patternUnits="userSpaceOnUse" x={c0} y={c0}>
            {showMinor && <rect width={gridMajor} height={gridMajor} fill="url(#grid-minor)" />}
            <path d={`M ${gridMajor} 0 L 0 0 0 ${gridMajor}`} fill="none" stroke="#18181b" strokeOpacity={0.09} strokeWidth={px(1)} />
          </pattern>
          <filter id="bin-shadow" x="-10%" y="-10%" width="120%" height="130%">
            <feDropShadow dx="0" dy={px(1)} stdDeviation={px(1.5)} floodColor="#18181b" floodOpacity="0.08" />
          </filter>
          <linearGradient id="scoop" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#18181b" stopOpacity="0" />
            <stop offset="1" stopColor="#18181b" stopOpacity="0.09" />
          </linearGradient>
          <pattern id="hatch" width={px(6)} height={px(6)} patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2={px(6)} stroke="#dc2626" strokeOpacity={0.35} strokeWidth={px(2)} />
          </pattern>
        </defs>

        <g transform={`translate(${tx} ${ty}) scale(${s})`}>
          {/* Drawer body */}
          <rect
            x={-px(10)}
            y={-px(10)}
            width={drawer.width + px(20)}
            height={drawer.depth + px(20)}
            rx={px(10)}
            fill="#e9e8e4"
            stroke="#d6d5d1"
            strokeWidth={px(1)}
          />
          <rect x={0} y={0} width={drawer.width} height={drawer.depth} rx={px(3)} fill="#fbfbfa" stroke="#cfcec9" strokeWidth={px(1)} />
          <rect x={c0} y={c0} width={bounds.width} height={bounds.depth} fill="url(#grid-major)" />
          {c0 > 0 && (
            <rect
              x={c0}
              y={c0}
              width={bounds.width}
              height={bounds.depth}
              fill="none"
              stroke="#a1a1aa"
              strokeWidth={px(1)}
              strokeDasharray={`${px(4)} ${px(4)}`}
            />
          )}

          {gridfinity && <GridfinityLattice width={bounds.width} depth={bounds.depth} offset={c0} s={s} />}

          {/* Rulers */}
          <Dimension x1={0} y1={-px(26)} x2={drawer.width} y2={-px(26)} label={`${fmt(drawer.width)} mm`} s={s} muted />
          <Dimension x1={-px(26)} y1={0} x2={-px(26)} y2={drawer.depth} label={`${fmt(drawer.depth)} mm`} s={s} muted vertical />
          <text x={drawer.width / 2} y={drawer.depth + px(26)} textAnchor="middle" fontSize={fontPx(10.5)} fill="#a1a1aa" className="font-medium tracking-wide uppercase">
            Front
          </text>

          <g transform={`translate(${c0} ${c0})`}>
            {/* Compartments */}
            {compartments.map((c) => {
              const isSel = selectedIds.includes(c.id);
              const isPrimary = c.id === selectedId;
              const isHover = c.id === hoverId && !isSel;
              const bad = issueIds.has(c.id);
              const geo = byCompartment.get(c.id);
              const warn = (geo?.parts.some((p) => p.warnings.length) ?? false) || !!geo?.error;
              const bw = c.w - gap;
              const bd = c.d - gap;
              const rr = Math.min(radius, bw / 2, bd / 2);
              const wall = print.wallThickness;
              const ri = rr > 0 ? Math.max(rr - wall, 0.5) : 0;
              const mod = getModule(c.kind);
              const wPx = c.w * s;
              const dPx = c.d * s;
              const shownName = c.name.length > 22 ? `${c.name.slice(0, 21)}…` : c.name;
              const showName = wPx > shownName.length * 6.9 + 20 && dPx > 44;
              const showDims = wPx > 58 && dPx > 26;
              const labelW = px(Math.max(showName ? shownName.length * 6.9 : 0, showDims ? 70 : 0) + 12);
              return (
                <g key={c.id} onPointerEnter={() => ui().setHover(c.id)} onPointerLeave={() => ui().setHover(null)}>
                  <g
                    onPointerDown={(e) => onPointerDownCompartment(e, c)}
                    onDoubleClick={(e) => e.stopPropagation()}
                    className={activeGesture === 'move' && isSel ? 'cursor-grabbing' : 'cursor-grab'}
                  >
                    <rect
                      x={c.x + gap / 2}
                      y={c.y + gap / 2}
                      width={bw}
                      height={bd}
                      rx={rr}
                      fill={bad ? '#fef2f2' : '#ffffff'}
                      stroke={bad ? '#dc2626' : isSel ? '#0f7a57' : isHover ? '#71717a' : '#c9c8c3'}
                      strokeWidth={px(isPrimary ? 1.75 : isSel ? 1.4 : 1)}
                      filter="url(#bin-shadow)"
                    />
                    {/* inner pocket – shows the wall thickness */}
                    {bw > 2 * wall && bd > 2 * wall && (
                      <rect
                        x={c.x + gap / 2 + wall}
                        y={c.y + gap / 2 + wall}
                        width={bw - 2 * wall}
                        height={bd - 2 * wall}
                        rx={ri}
                        fill={isSel ? '#f1f9f5' : '#f7f7f5'}
                        stroke={isSel ? '#b5ddcb' : '#e4e3df'}
                        strokeWidth={px(1)}
                      />
                    )}
                    {bad && <rect x={c.x + gap / 2} y={c.y + gap / 2} width={bw} height={bd} rx={rr} fill="url(#hatch)" />}
                    {mod.preview && (
                      <ModulePreview
                        shapes={mod.preview({ compartment: c, bin: resolveBinSpec(design, c), params: resolveParams(c), gap })}
                        ox={c.x}
                        oy={c.y}
                        s={s}
                        selected={isSel}
                      />
                    )}
                    {(showName || showDims) && (
                      <g pointerEvents="none">
                        {/* A soft plate behind the label keeps it readable over module previews. */}
                        <rect
                          x={c.x + c.w / 2 - labelW / 2}
                          y={c.y + c.d / 2 - px(showName ? 14 : 8)}
                          width={labelW}
                          height={px(showName && showDims ? 30 : 16)}
                          rx={px(4)}
                          fill="#ffffff"
                          fillOpacity={0.82}
                        />
                        {showName && (
                          <text
                            x={c.x + c.w / 2}
                            y={c.y + c.d / 2 - px(3)}
                            textAnchor="middle"
                            fontSize={fontPx(12)}
                            fontWeight={550}
                            fill={isSel ? '#0b5c42' : '#3f3f46'}
                          >
                            {shownName}
                          </text>
                        )}
                        {showDims && (
                          <text
                            x={c.x + c.w / 2}
                            y={c.y + c.d / 2 + (showName ? px(12) : px(4))}
                            textAnchor="middle"
                            fontSize={fontPx(10.5)}
                            fill="#8b8b93"
                            className="num font-mono"
                          >
                            {fmt(c.w)} × {fmt(c.d)}
                            {c.height !== null ? ` · h${fmt(c.height)}` : ''}
                          </text>
                        )}
                      </g>
                    )}
                    {warn && (
                      <g transform={`translate(${c.x + c.w - px(18)} ${c.y + px(6)}) scale(${px(1) * 0.5})`} pointerEvents="none">
                        <circle cx="12" cy="12" r="12" fill="#fff4ec" />
                        <AlertTriangle x={4} y={4} width={16} height={16} color="#c2410c" />
                      </g>
                    )}
                  </g>

                  {/* Dividers */}
                  {mod.supportsDividers &&
                    c.dividers.map((dv) => {
                      const vertical = dv.orientation === 'vertical';
                      const t = print.dividerThickness;
                      const hit = Math.max(t, px(10));
                      const dvSel = isPrimary && dv.id === selectedDividerId;
                      const x0 = c.x + gap / 2 + wall;
                      const y0 = c.y + gap / 2 + wall;
                      const len = vertical ? bd - 2 * wall : bw - 2 * wall;
                      return (
                        <g
                          key={dv.id}
                          className={vertical ? 'cursor-col-resize' : 'cursor-row-resize'}
                          onPointerDown={(e) => onPointerDownDivider(e, c, dv.id, dv.position)}
                          onDoubleClick={(e) => {
                            e.stopPropagation();
                            store().removeDivider(c.id, dv.id);
                          }}
                        >
                          <rect
                            x={vertical ? c.x + dv.position - t / 2 : x0}
                            y={vertical ? y0 : c.y + dv.position - t / 2}
                            width={vertical ? t : len}
                            height={vertical ? len : t}
                            fill={dvSel ? '#0f7a57' : isSel ? '#6fb49a' : '#b9b8b2'}
                            rx={Math.min(t / 2, px(1))}
                          />
                          <rect
                            x={vertical ? c.x + dv.position - hit / 2 : x0}
                            y={vertical ? y0 : c.y + dv.position - hit / 2}
                            width={vertical ? hit : len}
                            height={vertical ? len : hit}
                            fill="transparent"
                          />
                          {dvSel && (
                            <text
                              x={vertical ? c.x + dv.position + px(5) : c.x + gap / 2 + print.wallThickness + px(4)}
                              y={vertical ? c.y + px(16) : c.y + dv.position - px(5)}
                              fontSize={fontPx(10.5)}
                              fill="#0b5c42"
                              className="num font-mono"
                              textAnchor="start"
                              pointerEvents="none"
                            >
                              {fmt(dv.position)}
                            </text>
                          )}
                        </g>
                      );
                    })}
                </g>
              );
            })}

            {/* Selection chrome */}
            {selected && selectedIds.length <= 1 && <SelectionChrome c={selected} s={s} bounds={bounds} onHandle={onPointerDownHandle} others={compartments.filter((o) => o.id !== selected.id).map(rectOf)} />}

            {selectedIds.length > 1 && <GroupOutline rects={compartments.filter((c) => selectedIds.includes(c.id))} s={s} />}
            {marquee && (
              <rect x={marquee.x} y={marquee.y} width={marquee.w} height={marquee.d} fill="#0f7a57" fillOpacity={0.07} stroke="#0f7a57" strokeWidth={px(1)} strokeDasharray={`${px(3)} ${px(3)}`} />
            )}

            {/* Snap guides */}
            {guides.x !== null && (
              <line x1={guides.x} x2={guides.x} y1={-px(12)} y2={bounds.depth + px(12)} stroke="#0f7a57" strokeWidth={px(1)} strokeDasharray={`${px(3)} ${px(3)}`} />
            )}
            {guides.y !== null && (
              <line y1={guides.y} y2={guides.y} x1={-px(12)} x2={bounds.width + px(12)} stroke="#0f7a57" strokeWidth={px(1)} strokeDasharray={`${px(3)} ${px(3)}`} />
            )}
          </g>
        </g>
      </svg>

      {compartments.length === 0 && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="animate-fade-in rounded-xl border border-line bg-surface/90 px-4 py-3 text-center shadow-panel backdrop-blur">
            <div className="text-[13px] font-medium text-ink">Empty drawer</div>
            <div className="mt-0.5 text-[12px] text-muted">Double-click to add a compartment, or pick a quick layout.</div>
          </div>
        </div>
      )}

      <div className="absolute right-3 bottom-3 flex items-center gap-0.5 rounded-xl border border-line bg-surface/90 p-1 shadow-panel backdrop-blur">
        <IconButton label="Zoom out" onClick={() => zoomAt(1 / 1.25)}>
          <Minus size={15} />
        </IconButton>
        <span className="num w-11 text-center text-[11px] text-muted">{Math.round(zoom * 100)}%</span>
        <IconButton label="Zoom in" onClick={() => zoomAt(1.25)}>
          <Plus size={15} />
        </IconButton>
        <IconButton label="Fit to view" onClick={fit}>
          <Maximize size={14} />
        </IconButton>
      </div>
    </div>
  );
}

const TONE_STYLE: Record<PreviewTone, { fill: string; fillSel?: string; stroke?: string; dash?: boolean }> = {
  wall: { fill: '#b9b8b2', fillSel: '#6fb49a' },
  block: { fill: '#efeeea', fillSel: '#ddf0e7' },
  hole: { fill: '#ffffff', stroke: '#cfcec9' },
  scoop: { fill: 'url(#scoop)' },
  outline: { fill: 'none', stroke: '#a1a1aa', dash: true },
};

/** Renders a module's preview shapes (compartment-local mm) on top of the bin. */
function ModulePreview({ shapes, ox, oy, s, selected }: { shapes: PreviewShape[]; ox: number; oy: number; s: number; selected: boolean }) {
  return (
    <g pointerEvents="none" transform={`translate(${ox} ${oy})`}>
      {shapes.map((sh, i) => {
        const st = TONE_STYLE[sh.tone];
        const common = {
          fill: selected && st.fillSel ? st.fillSel : st.fill,
          stroke: st.stroke ?? 'none',
          strokeWidth: 1 / s,
          strokeDasharray: st.dash ? `${3 / s} ${3 / s}` : undefined,
        };
        if (sh.kind === 'rect')
          return <rect key={i} x={sh.x} y={sh.y} width={Math.max(0, sh.w)} height={Math.max(0, sh.d)} rx={sh.r ?? 0} {...common} />;
        if (sh.kind === 'circle') return <circle key={i} cx={sh.cx} cy={sh.cy} r={Math.max(0, sh.r)} {...common} />;
        return <polygon key={i} points={sh.points.map(([x, y]) => `${x},${y}`).join(' ')} {...common} />;
      })}
    </g>
  );
}

/** The 42 mm Gridfinity lattice drawn over the usable area. */
function GridfinityLattice({ width, depth, offset, s }: { width: number; depth: number; offset: number; s: number }) {
  const px = (n: number) => n / s;
  const cols = Math.floor(width / PITCH);
  const rows = Math.floor(depth / PITCH);
  if (cols < 1 || rows < 1) return null;
  const cells = [];
  for (let c = 0; c < cols; c++)
    for (let r = 0; r < rows; r++)
      cells.push(
        <rect
          key={`${c}-${r}`}
          x={offset + c * PITCH}
          y={offset + r * PITCH}
          width={PITCH}
          height={PITCH}
          fill="none"
          stroke="#0f7a57"
          strokeOpacity={0.22}
          strokeWidth={px(1)}
        />,
      );
  return (
    <g pointerEvents="none">
      {cells}
      <text x={offset + 2} y={offset + depth - px(4)} fontSize={px(10)} fill="#0f7a57" fillOpacity={0.5} className="num font-mono">
        {cols} × {rows} grid
      </text>
    </g>
  );
}

/** Dashed box around a multi-compartment selection, with a count badge. */
function GroupOutline({ rects, s }: { rects: Rect[]; s: number }) {
  if (!rects.length) return null;
  const px = (n: number) => n / s;
  const x = Math.min(...rects.map((r) => r.x));
  const y = Math.min(...rects.map((r) => r.y));
  const w = Math.max(...rects.map((r) => r.x + r.w)) - x;
  const d = Math.max(...rects.map((r) => r.y + r.d)) - y;
  const label = `${rects.length} selected`;
  const lw = px(label.length * 6.2 + 12);
  return (
    <g pointerEvents="none">
      <rect x={x} y={y} width={w} height={d} fill="none" stroke="#0f7a57" strokeWidth={px(1.25)} strokeDasharray={`${px(4)} ${px(3)}`} />
      <rect x={x} y={y - px(19)} width={lw} height={px(16)} rx={px(4)} fill="#0f7a57" />
      <text x={x + lw / 2} y={y - px(7.5)} textAnchor="middle" fontSize={px(10.5)} fill="#ffffff" className="font-medium">
        {label}
      </text>
    </g>
  );
}

function SelectionChrome({
  c,
  s,
  bounds,
  others,
  onHandle,
}: {
  c: Compartment;
  s: number;
  bounds: { width: number; depth: number };
  others: Rect[];
  onHandle: (e: ReactPointerEvent, c: Compartment, edges: Edge[]) => void;
}) {
  const px = (n: number) => n / s;
  const hit = px(10);
  const knob = px(8);
  const edges: { edges: Edge[]; x: number; y: number; w: number; h: number; cursor: string }[] = [
    { edges: ['top'], x: c.x + hit / 2, y: c.y - hit / 2, w: c.w - hit, h: hit, cursor: 'cursor-ns-resize' },
    { edges: ['bottom'], x: c.x + hit / 2, y: c.y + c.d - hit / 2, w: c.w - hit, h: hit, cursor: 'cursor-ns-resize' },
    { edges: ['left'], x: c.x - hit / 2, y: c.y + hit / 2, w: hit, h: c.d - hit, cursor: 'cursor-ew-resize' },
    { edges: ['right'], x: c.x + c.w - hit / 2, y: c.y + hit / 2, w: hit, h: c.d - hit, cursor: 'cursor-ew-resize' },
  ];
  const corners: { edges: Edge[]; x: number; y: number; cursor: string }[] = [
    { edges: ['top', 'left'], x: c.x, y: c.y, cursor: 'cursor-nwse-resize' },
    { edges: ['top', 'right'], x: c.x + c.w, y: c.y, cursor: 'cursor-nesw-resize' },
    { edges: ['bottom', 'left'], x: c.x, y: c.y + c.d, cursor: 'cursor-nesw-resize' },
    { edges: ['bottom', 'right'], x: c.x + c.w, y: c.y + c.d, cursor: 'cursor-nwse-resize' },
  ];
  const r = rectOf(c);
  // Distance from each edge to the nearest neighbour / drawer wall.
  const gaps = [
    { edge: 'left' as const, d: c.x - edgeLimit(r, 'left', others, bounds) },
    { edge: 'right' as const, d: edgeLimit(r, 'right', others, bounds) - (c.x + c.w) },
    { edge: 'top' as const, d: c.y - edgeLimit(r, 'top', others, bounds) },
    { edge: 'bottom' as const, d: edgeLimit(r, 'bottom', others, bounds) - (c.y + c.d) },
  ].filter((g) => g.d > 0.5);

  return (
    <g>
      <rect x={c.x} y={c.y} width={c.w} height={c.d} fill="none" stroke="#0f7a57" strokeOpacity={0.35} strokeWidth={px(1)} strokeDasharray={`${px(2)} ${px(2)}`} pointerEvents="none" />
      {/* footprint dimensions */}
      <Dimension x1={c.x} y1={c.y - px(14)} x2={c.x + c.w} y2={c.y - px(14)} label={fmt(c.w)} s={s} />
      <Dimension x1={c.x - px(14)} y1={c.y} x2={c.x - px(14)} y2={c.y + c.d} label={fmt(c.d)} s={s} vertical />
      {/* clearances to neighbours */}
      {gaps.map((g) => {
        const midY = c.y + c.d / 2;
        const midX = c.x + c.w / 2;
        if (g.edge === 'left') return <GapMark key={g.edge} x1={c.x - g.d} y1={midY} x2={c.x} y2={midY} d={g.d} s={s} />;
        if (g.edge === 'right') return <GapMark key={g.edge} x1={c.x + c.w} y1={midY} x2={c.x + c.w + g.d} y2={midY} d={g.d} s={s} />;
        if (g.edge === 'top') return <GapMark key={g.edge} x1={midX} y1={c.y - g.d} x2={midX} y2={c.y} d={g.d} s={s} />;
        return <GapMark key={g.edge} x1={midX} y1={c.y + c.d} x2={midX} y2={c.y + c.d + g.d} d={g.d} s={s} />;
      })}
      {edges.map((h) => (
        <rect key={h.edges.join()} x={h.x} y={h.y} width={Math.max(h.w, 0)} height={Math.max(h.h, 0)} fill="transparent" className={h.cursor} onPointerDown={(e) => onHandle(e, c, h.edges)} />
      ))}
      {/* edge knobs */}
      {[
        [c.x + c.w / 2, c.y, true],
        [c.x + c.w / 2, c.y + c.d, true],
        [c.x, c.y + c.d / 2, false],
        [c.x + c.w, c.y + c.d / 2, false],
      ].map(([x, y, horiz], i) => (
        <rect
          key={i}
          x={(x as number) - (horiz ? knob : px(2.5))}
          y={(y as number) - (horiz ? px(2.5) : knob)}
          width={horiz ? 2 * knob : px(5)}
          height={horiz ? px(5) : 2 * knob}
          rx={px(2.5)}
          fill="#ffffff"
          stroke="#0f7a57"
          strokeWidth={px(1.25)}
          pointerEvents="none"
        />
      ))}
      {corners.map((h) => (
        <rect
          key={h.edges.join()}
          x={h.x - px(5)}
          y={h.y - px(5)}
          width={px(10)}
          height={px(10)}
          rx={px(2)}
          fill="#ffffff"
          stroke="#0f7a57"
          strokeWidth={px(1.25)}
          className={h.cursor}
          onPointerDown={(e) => onHandle(e, c, h.edges)}
        />
      ))}
    </g>
  );
}

function Dimension({
  x1,
  y1,
  x2,
  y2,
  label,
  s,
  vertical,
  muted,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label: string;
  s: number;
  vertical?: boolean;
  muted?: boolean;
}) {
  const px = (n: number) => n / s;
  const color = muted ? '#a1a1aa' : '#0f7a57';
  const tick = px(4);
  const mx = (x1 + x2) / 2;
  const my = (y1 + y2) / 2;
  const fs = px(10.5);
  const labelW = px(label.length * 6.4 + 10);
  return (
    <g pointerEvents="none">
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={px(1)} strokeOpacity={muted ? 0.7 : 0.8} />
      {vertical ? (
        <>
          <line x1={x1 - tick} x2={x1 + tick} y1={y1} y2={y1} stroke={color} strokeWidth={px(1)} />
          <line x1={x2 - tick} x2={x2 + tick} y1={y2} y2={y2} stroke={color} strokeWidth={px(1)} />
        </>
      ) : (
        <>
          <line x1={x1} x2={x1} y1={y1 - tick} y2={y1 + tick} stroke={color} strokeWidth={px(1)} />
          <line x1={x2} x2={x2} y1={y2 - tick} y2={y2 + tick} stroke={color} strokeWidth={px(1)} />
        </>
      )}
      <g transform={vertical ? `rotate(-90 ${mx} ${my})` : undefined}>
        <rect x={mx - labelW / 2} y={my - px(8)} width={labelW} height={px(16)} rx={px(4)} fill={muted ? '#f6f6f4' : '#ffffff'} stroke={muted ? 'none' : '#b5ddcb'} strokeWidth={px(1)} />
        <text x={mx} y={my + px(3.6)} textAnchor="middle" fontSize={fs} fill={muted ? '#71717a' : '#0b5c42'} className="num font-mono">
          {label}
        </text>
      </g>
    </g>
  );
}

function GapMark({ x1, y1, x2, y2, d, s }: { x1: number; y1: number; x2: number; y2: number; d: number; s: number }) {
  const px = (n: number) => n / s;
  const len = Math.hypot(x2 - x1, y2 - y1) * s;
  return (
    <g pointerEvents="none">
      <line x1={x1} y1={y1} x2={x2} y2={y2} stroke="#f97316" strokeWidth={px(1)} strokeDasharray={`${px(2)} ${px(2)}`} />
      {len > 26 && (
        <text x={(x1 + x2) / 2} y={(y1 + y2) / 2 - px(4)} textAnchor="middle" fontSize={px(10)} fill="#c2410c" className="num font-mono">
          {fmt(d)}
        </text>
      )}
    </g>
  );
}
