import { describe, expect, it } from 'vitest';
import { printerFromPreset } from './defaults';
import { PRINTERS, findPrinter, lineWidthFor, matchesPreset, maxLayerHeight, printersByBrand, wallAdvice } from './printers';

describe('printer presets', () => {
  it('has unique ids and plausible build volumes', () => {
    const ids = PRINTERS.map((p) => p.id);
    expect(new Set(ids).size).toBe(ids.length);
    for (const p of PRINTERS) {
      for (const axis of [p.build.x, p.build.y, p.build.z]) {
        expect(axis).toBeGreaterThanOrEqual(100);
        expect(axis).toBeLessThanOrEqual(600);
      }
      expect(p.nozzle).toBeGreaterThan(0);
      expect(p.efficiency).toBeGreaterThan(0);
      expect(p.efficiency).toBeLessThanOrEqual(1);
    }
  });

  it('groups by brand in list order', () => {
    const groups = printersByBrand();
    expect(groups[0].brand).toBe('Bambu Lab');
    expect(groups[0].items[0].model).toBe('X2D');
    expect(groups.flatMap((g) => g.items)).toHaveLength(PRINTERS.length);
  });

  it('builds a config from a preset and falls back to custom', () => {
    const x2d = printerFromPreset('bambu-x2d');
    expect(x2d.label).toBe('Bambu Lab X2D');
    expect(x2d.build).toEqual({ x: 256, y: 256, z: 260 });
    expect(printerFromPreset('nope').presetId).toBe('custom');
  });

  it('knows when settings no longer match the preset', () => {
    const p = findPrinter('bambu-a1-mini')!;
    expect(matchesPreset(p.id, p.build, p.nozzle)).toBe(true);
    expect(matchesPreset(p.id, { ...p.build, x: 300 }, p.nozzle)).toBe(false);
    expect(matchesPreset(p.id, p.build, 0.6)).toBe(false);
    expect(matchesPreset('custom', p.build, p.nozzle)).toBe(false);
  });
});

describe('nozzle advice', () => {
  it('derives an extrusion width from the nozzle', () => {
    expect(lineWidthFor(0.4)).toBeCloseTo(0.42, 2);
    expect(lineWidthFor(0.6)).toBeCloseTo(0.63, 2);
  });

  it('accepts walls that are whole multiples of the line width', () => {
    const a = wallAdvice(1.68, 0.4); // 4 × 0.42
    expect(a.perimeters).toBe(4);
    expect(a.awkward).toBe(false);
    expect(a.tooThin).toBe(false);
  });

  it('flags awkward walls and suggests the nearest clean value', () => {
    // 1.5 mm is 3.57 lines of 0.42, so the nearest clean wall is 4 lines
    const up = wallAdvice(1.5, 0.4);
    expect(up.awkward).toBe(true);
    expect(up.perimeters).toBe(4);
    expect(up.suggested).toBeCloseTo(1.68, 2);
    expect(wallAdvice(up.suggested, 0.4).awkward).toBe(false);
    // 1.15 mm rounds down to 3 lines
    const down = wallAdvice(1.15, 0.4);
    expect(down.perimeters).toBe(3);
    expect(down.suggested).toBeCloseTo(1.26, 2);
  });

  it('flags walls thinner than a single line', () => {
    expect(wallAdvice(0.8, 0.8).tooThin).toBe(true);
    expect(wallAdvice(1.2, 0.4).tooThin).toBe(false);
  });

  it('caps layer height at three quarters of the nozzle', () => {
    expect(maxLayerHeight(0.4)).toBeCloseTo(0.3, 2);
    expect(maxLayerHeight(0.8)).toBeCloseTo(0.6, 2);
  });
});
