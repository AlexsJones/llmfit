/**
 * Interactive 3D preview. Renders the exact meshes that are exported to STL
 * (no separate "display" geometry), so the preview is always faithful.
 */
import { ContactShadows, OrbitControls } from '@react-three/drei';
import { Canvas, type ThreeEvent, useThree } from '@react-three/fiber';
import { Box, Eye, EyeOff, Layers, Loader2 } from 'lucide-react';
import { type ComponentRef, useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { toCreasedNormals } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { useGeometry } from '../hooks/geometry';
import type { MeshData, PartMesh } from '../geometry/types';
import type { Design } from '../model/types';
import { useDesign } from '../store/designStore';
import { useUi } from '../store/uiStore';
import { IconButton, Segmented, cx } from './ui/controls';

type Preset = 'iso' | 'top' | 'front';

const geometryCache = new WeakMap<MeshData, { surface: THREE.BufferGeometry; edges: THREE.EdgesGeometry }>();

function threeGeometry(mesh: MeshData) {
  let g = geometryCache.get(mesh);
  if (!g) {
    const indexed = new THREE.BufferGeometry();
    indexed.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    indexed.setIndex(new THREE.BufferAttribute(mesh.indices, 1));
    const edges = new THREE.EdgesGeometry(indexed, 28);
    const surface = toCreasedNormals(indexed, Math.PI / 7);
    indexed.dispose();
    g = { surface, edges };
    geometryCache.set(mesh, g);
  }
  return g;
}

function PartView({
  part,
  selected,
  hovered,
  position,
}: {
  part: PartMesh;
  selected: boolean;
  hovered: boolean;
  position: [number, number, number];
}) {
  const { surface, edges } = useMemo(() => threeGeometry(part.mesh), [part.mesh]);
  const insert = part.role === 'insert';
  const color = selected ? '#d9e2ff' : insert ? '#e6e3da' : '#f6f6f3';
  const warn = part.warnings.length > 0;
  return (
    <group position={position}>
      <mesh
        geometry={surface}
        castShadow
        receiveShadow
        onClick={(e: ThreeEvent<MouseEvent>) => {
          if (e.delta > 4) return;
          e.stopPropagation();
          useUi.getState().select(part.compartmentId);
        }}
        onPointerOver={(e) => {
          e.stopPropagation();
          useUi.getState().setHover(part.compartmentId);
        }}
        onPointerOut={() => useUi.getState().setHover(null)}
      >
        <meshStandardMaterial
          color={color}
          roughness={0.62}
          metalness={0}
          emissive={selected ? '#3a63f3' : hovered ? '#71717a' : '#000000'}
          emissiveIntensity={selected ? 0.12 : hovered ? 0.06 : 0}
        />
      </mesh>
      <lineSegments geometry={edges}>
        <lineBasicMaterial color={selected ? '#3a63f3' : warn ? '#ea580c' : '#86867f'} transparent opacity={selected ? 0.9 : 0.6} />
      </lineSegments>
    </group>
  );
}

function DrawerGhost({ design }: { design: Design }) {
  const { width: W, depth: D, height: H } = design.drawer;
  const t = 3; // visual wall thickness
  const mat = useMemo(
    () => new THREE.MeshStandardMaterial({ color: '#c8d0e0', transparent: true, opacity: 0.16, depthWrite: false, side: THREE.DoubleSide }),
    [],
  );
  const edges = useMemo(() => {
    const g = new THREE.BoxGeometry(W, D, H);
    const e = new THREE.EdgesGeometry(g);
    g.dispose();
    return e;
  }, [W, D, H]);
  useEffect(() => () => edges.dispose(), [edges]);
  useEffect(() => () => mat.dispose(), [mat]);
  return (
    <group>
      {/* floor */}
      <mesh position={[W / 2, D / 2, -t / 2]} receiveShadow>
        <boxGeometry args={[W + 2 * t, D + 2 * t, t]} />
        <meshStandardMaterial color="#e7e6e2" roughness={0.9} />
      </mesh>
      {/* walls: back, left, right, front */}
      <mesh position={[W / 2, D + t / 2, H / 2]} material={mat}>
        <boxGeometry args={[W + 2 * t, t, H]} />
      </mesh>
      <mesh position={[-t / 2, D / 2, H / 2]} material={mat}>
        <boxGeometry args={[t, D, H]} />
      </mesh>
      <mesh position={[W + t / 2, D / 2, H / 2]} material={mat}>
        <boxGeometry args={[t, D, H]} />
      </mesh>
      <mesh position={[W / 2, -t / 2, H / 2]} material={mat}>
        <boxGeometry args={[W + 2 * t, t, H]} />
      </mesh>
      <lineSegments geometry={edges} position={[W / 2, D / 2, H / 2]}>
        <lineBasicMaterial color="#8a94a8" transparent opacity={0.5} />
      </lineSegments>
    </group>
  );
}

/**
 * Exploded placement: parts drift outwards from the drawer's centre and
 * inserts lift clear of their bin, so the assembly can be read apart.
 */
function explodedPosition(p: PartMesh, t: number, W: number, D: number, H: number): [number, number, number] {
  if (t <= 0) return p.offset;
  const cx = p.offset[0] + p.size[0] / 2 - W / 2;
  const cy = p.offset[1] + p.size[1] / 2 - D / 2;
  const len = Math.hypot(cx, cy) || 1;
  const spread = t * Math.max(W, D) * 0.35;
  const lift = p.role === 'insert' ? t * (H + 35) : 0;
  return [p.offset[0] + (cx / len) * spread, p.offset[1] + (cy / len) * spread, p.offset[2] + lift];
}

function Scene({ preset, nonce }: { preset: Preset; nonce: number }) {
  const design = useDesign((s) => s.design);
  const { parts } = useGeometry();
  const selectedId = useUi((s) => s.selectedId);
  const hoverId = useUi((s) => s.hoverId);
  const showDrawer = useUi((s) => s.showDrawer);
  const explode = useUi((s) => s.explode);
  const controls = useRef<ComponentRef<typeof OrbitControls>>(null);
  const { width: W, depth: D, height: H } = design.drawer;
  const span = Math.max(W, D);

  const size = useThree((st) => st.size);
  const camera = useThree((st) => st.camera) as THREE.PerspectiveCamera;

  // Frame the drawer: fit its bounding sphere inside both the vertical and horizontal field of view.
  useEffect(() => {
    const c = controls.current;
    if (!c || !size.width || !size.height) return;
    const radius = Math.hypot(W, D, H) / 2;
    const vfov = THREE.MathUtils.degToRad(camera.fov);
    const hfov = 2 * Math.atan(Math.tan(vfov / 2) * (size.width / size.height));
    const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.04;
    const dir =
      preset === 'top' ? new THREE.Vector3(0, 1, 0.0001) : preset === 'front' ? new THREE.Vector3(0, 0.55, 1) : new THREE.Vector3(0.55, 0.85, 1);
    dir.normalize();
    const target = new THREE.Vector3(0, H * 0.25, 0);
    c.object.position.copy(target).addScaledVector(dir, dist);
    c.target.copy(target);
    c.update();
  }, [preset, nonce, W, D, H, size.width, size.height, camera]);

  return (
    <>
      <color attach="background" args={['#efefec']} />
      <fog attach="fog" args={['#efefec', span * 4, span * 10]} />
      <hemisphereLight args={['#ffffff', '#cfccc4', 1.15]} />
      <directionalLight position={[span * 0.5, span * 1.2, span * 0.8]} intensity={2.3} castShadow shadow-mapSize={[2048, 2048]}>
        <orthographicCamera attach="shadow-camera" args={[-span, span, span, -span, 1, span * 4]} />
      </directionalLight>
      <directionalLight position={[-span, span * 0.6, -span * 0.4]} intensity={0.45} />

      {/* World is Z-up (as in the STL); convert to three's Y-up and centre on the drawer. */}
      <group rotation={[-Math.PI / 2, 0, 0]} position={[-W / 2, 0, D / 2]}>
        {showDrawer && explode < 0.02 && <DrawerGhost design={design} />}
        {parts.map((p) => (
          <PartView
            key={p.id}
            part={p}
            position={explodedPosition(p, explode, W, D, H)}
            selected={p.compartmentId === selectedId}
            hovered={p.compartmentId === hoverId}
          />
        ))}
      </group>
      <ContactShadows position={[0, -3.1, 0]} scale={span * 1.8} blur={2.6} opacity={0.28} far={span} />
      <OrbitControls ref={controls} makeDefault enableDamping dampingFactor={0.12} maxPolarAngle={Math.PI * 0.495} minDistance={span * 0.15} maxDistance={span * 8} />
    </>
  );
}


export function Viewport3D() {
  const [preset, setPreset] = useState<Preset>('iso');
  const [nonce, setNonce] = useState(0);
  const { stale, parts } = useGeometry();
  const showDrawer = useUi((s) => s.showDrawer);
  const explode = useUi((s) => s.explode);
  const span = useDesign((s) => Math.max(s.design.drawer.width, s.design.drawer.depth));

  return (
    <div className="relative h-full w-full overflow-hidden bg-[#efefec]">
      <Canvas
        shadows
        dpr={[1, 2]}
        camera={{ position: [span * 0.62, span * 0.8, span * 0.95], fov: 35, near: 1, far: span * 20 }}
        gl={{ antialias: true, preserveDrawingBuffer: true }}
        onCreated={({ gl }) => {
          gl.toneMapping = THREE.ACESFilmicToneMapping;
          gl.toneMappingExposure = 1.05;
        }}
      >
        <Scene preset={preset} nonce={nonce} />
      </Canvas>

      <div className="absolute top-3 left-3 flex items-center gap-2">
        <Segmented
          size="sm"
          value={preset}
          onChange={(v) => {
            setPreset(v);
            setNonce((n) => n + 1);
          }}
          options={[
            { value: 'iso', label: 'Iso' },
            { value: 'top', label: 'Top' },
            { value: 'front', label: 'Front' },
          ]}
          className="border border-line bg-surface/90 shadow-panel backdrop-blur"
        />
        <div className="rounded-lg border border-line bg-surface/90 shadow-panel backdrop-blur">
          <IconButton label={showDrawer ? 'Hide drawer' : 'Show drawer'} onClick={() => useUi.getState().setShowDrawer(!showDrawer)} className="h-7 w-7">
            {showDrawer ? <Eye size={14} /> : <EyeOff size={14} />}
          </IconButton>
        </div>
      </div>

      <div className="absolute top-3 right-3 flex w-[min(210px,46%)] items-center gap-2 rounded-lg border border-line bg-surface/90 px-2.5 py-1.5 shadow-panel backdrop-blur">
        <Layers size={13} className="shrink-0 text-muted" />
        <input
          type="range"
          min={0}
          max={1}
          step={0.02}
          value={explode}
          onChange={(e) => useUi.getState().setExplode(Number(e.target.value))}
          aria-label="Exploded view"
          title="Exploded view"
          className="h-1 w-full cursor-pointer appearance-none rounded-full bg-line-strong accent-accent"
        />
        {explode > 0 && (
          <button type="button" onClick={() => useUi.getState().setExplode(0)} className="shrink-0 text-[11px] text-muted hover:text-ink">
            reset
          </button>
        )}
      </div>

      <div className="pointer-events-none absolute right-3 bottom-3 flex items-center gap-2">
        <span
          className={cx(
            'flex items-center gap-1.5 rounded-lg border border-line bg-surface/90 px-2.5 py-1 text-[11px] text-muted shadow-panel backdrop-blur transition-opacity',
          )}
        >
          {stale ? <Loader2 size={12} className="animate-spin" /> : <Box size={12} />}
          {parts.length} printable part{parts.length === 1 ? '' : 's'}
        </span>
      </div>
      <div className="pointer-events-none absolute bottom-3 left-3 hidden text-[11px] text-faint lg:block">Drag to orbit · Right-drag to pan · Scroll to zoom</div>
    </div>
  );
}
