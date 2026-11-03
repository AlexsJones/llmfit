import '@testing-library/jest-dom/vitest';

// Node 25+ ships a built-in Web Storage implementation and defines
// `localStorage` / `sessionStorage` getters on globalThis. Without
// `--localstorage-file` those getters return `undefined`. Vitest's jsdom
// environment only copies window keys that are not already present on the
// global, so Node's (empty) getters shadow jsdom's working Storage objects and
// any `window.localStorage.*` call throws. Re-expose jsdom's storage so the
// tests run the same on every Node version (no-op on Node <= 24).
const dom = globalThis.jsdom;
if (dom?.window) {
  for (const key of ['localStorage', 'sessionStorage']) {
    const storage = dom.window[key];
    // Don't read globalThis[key] here: touching Node's getter emits an
    // ExperimentalWarning. Redefining is harmless where jsdom already won.
    if (storage) {
      Object.defineProperty(globalThis, key, {
        configurable: true,
        enumerable: true,
        get: () => storage
      });
    }
  }
}
