import { defineConfig } from 'vitest/config';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // manifold-3d ships an Emscripten build that loads its own .wasm file;
  // pre-bundling it would break the relative wasm lookup.
  optimizeDeps: { exclude: ['manifold-3d'] },
  build: { target: 'es2022', chunkSizeWarningLimit: 2000 },
  test: { environment: 'node', include: ['src/**/*.test.ts'] },
});
