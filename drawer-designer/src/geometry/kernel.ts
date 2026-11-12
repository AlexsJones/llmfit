/**
 * Loads the manifold-3d WASM geometry kernel once and shares it.
 *
 * manifold-3d performs exact, robust CSG and guarantees that every output is a
 * closed 2-manifold (watertight), which is exactly what slicers need.
 */
import Module from 'manifold-3d';
import type { ManifoldToplevel } from 'manifold-3d';

export type Kernel = ManifoldToplevel;

let kernel: Kernel | null = null;
let loading: Promise<Kernel> | null = null;

export function loadKernel(wasmUrl?: string): Promise<Kernel> {
  if (kernel) return Promise.resolve(kernel);
  if (!loading) {
    loading = Module(wasmUrl ? { locateFile: () => wasmUrl } : undefined).then((k) => {
      k.setup();
      kernel = k;
      return k;
    });
  }
  return loading;
}

export function getKernel(): Kernel {
  if (!kernel) throw new Error('Geometry kernel not loaded yet');
  return kernel;
}
