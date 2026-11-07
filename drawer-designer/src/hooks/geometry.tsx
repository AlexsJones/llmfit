import wasmUrl from 'manifold-3d/manifold.wasm?url';
import { type ReactNode, createContext, useContext, useDeferredValue, useEffect, useMemo, useState } from 'react';
import { type Kernel, loadKernel } from '../geometry/kernel';
import { type CompartmentGeometry, buildDesignGeometry } from '../geometry/parts';
import type { PartMesh } from '../geometry/types';
import { useDesign } from '../store/designStore';

export function useKernel() {
  const [kernel, setKernel] = useState<Kernel | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    loadKernel(wasmUrl)
      .then((k) => alive && setKernel(k))
      .catch((e: unknown) => alive && setError(e instanceof Error ? e.message : String(e)));
    return () => {
      alive = false;
    };
  }, []);
  return { kernel, error };
}

export interface GeometryState {
  byCompartment: Map<string, CompartmentGeometry>;
  parts: PartMesh[];
  /** True while the deferred geometry lags behind the latest layout. */
  stale: boolean;
}

const GeometryContext = createContext<GeometryState>({ byCompartment: new Map(), parts: [], stale: false });

export function GeometryProvider({ kernel, children }: { kernel: Kernel; children: ReactNode }) {
  const design = useDesign((s) => s.design);
  // Defer heavy CSG so the 2D editor stays responsive while dragging.
  const deferred = useDeferredValue(design);
  const value = useMemo<GeometryState>(() => {
    const list = buildDesignGeometry(kernel, deferred);
    return {
      byCompartment: new Map(list.map((g) => [g.compartmentId, g])),
      parts: list.flatMap((g) => g.parts),
      stale: false,
    };
  }, [kernel, deferred]);
  const state = useMemo(() => ({ ...value, stale: deferred !== design }), [value, deferred, design]);
  return <GeometryContext.Provider value={state}>{children}</GeometryContext.Provider>;
}

export const useGeometry = () => useContext(GeometryContext);
