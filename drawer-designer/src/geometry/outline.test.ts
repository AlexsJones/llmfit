import { describe, expect, it } from 'vitest';
import {
  decodeOutline,
  encodeOutline,
  fitOutline,
  growPolygon,
  polygonArea,
  simplify,
  toGray,
  traceOutline,
  traceOutlineDetailed,
  type GrayImage,
} from './outline';

/** Render a filled shape into a greyscale test image. */
function render(w: number, h: number, inside: (x: number, y: number) => boolean): GrayImage {
  const data = new Float32Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) data[y * w + x] = inside(x, y) ? 0.1 : 0.95;
  return { width: w, height: h, data };
}

describe('outline encoding', () => {
  it('round-trips points and clamps junk', () => {
    const pts = [{ x: 0, y: 0 }, { x: 0.5, y: 0.25 }, { x: 1, y: 1 }];
    expect(decodeOutline(encodeOutline(pts))).toEqual(pts);
    expect(decodeOutline('bad')).toEqual([]);
    expect(decodeOutline(42)).toEqual([]);
    expect(decodeOutline('5,-3 0.5,0.5')).toEqual([{ x: 1, y: 0 }, { x: 0.5, y: 0.5 }]);
  });
});

describe('outline geometry', () => {
  const square = [{ x: 0, y: 0 }, { x: 1, y: 0 }, { x: 1, y: 1 }, { x: 0, y: 1 }];

  it('computes signed area', () => {
    expect(polygonArea(square)).toBeCloseTo(1);
    expect(polygonArea([...square].reverse())).toBeCloseTo(-1);
  });

  it('fits into a box keeping the aspect ratio and centring', () => {
    const out = fitOutline(square, 100, 40, 5);
    // 30 mm available in depth, so the square becomes 30 x 30, centred in x
    const w = Math.max(...out.map((p) => p.x)) - Math.min(...out.map((p) => p.x));
    const d = Math.max(...out.map((p) => p.y)) - Math.min(...out.map((p) => p.y));
    expect(w).toBeCloseTo(30);
    expect(d).toBeCloseTo(30);
    expect(Math.min(...out.map((p) => p.x))).toBeCloseTo(35);
    expect(Math.min(...out.map((p) => p.y))).toBeCloseTo(5);
  });

  it('simplifies a dense straight edge back to its corners', () => {
    const dense = Array.from({ length: 50 }, (_, i) => ({ x: i / 49, y: 0 })).concat([{ x: 1, y: 1 }, { x: 0, y: 1 }]);
    expect(simplify(dense, 0.01).length).toBeLessThan(6);
  });

  it('grows a polygon outwards', () => {
    const big = growPolygon(square, 0.1);
    expect(Math.abs(polygonArea(big))).toBeGreaterThan(1);
  });
});

describe('image tracing', () => {
  it('traces a rectangle to a four-corner outline', () => {
    const img = render(200, 200, (x, y) => x > 40 && x < 160 && y > 60 && y < 140);
    const pts = traceOutline(img, { threshold: 0.5 });
    expect(pts.length).toBeGreaterThanOrEqual(4);
    expect(pts.length).toBeLessThan(12);
    // normalised: long side spans the full 0..1, short side the aspect ratio (80/120)
    const w = Math.max(...pts.map((p) => p.x));
    const h = Math.max(...pts.map((p) => p.y));
    expect(w).toBeCloseTo(1, 1);
    expect(h).toBeCloseTo(80 / 120, 1);
    expect(polygonArea(pts)).toBeGreaterThan(0);
  });

  it('traces a disc and keeps its area close to a circle', () => {
    const img = render(200, 200, (x, y) => Math.hypot(x - 100, y - 100) < 70);
    const pts = traceOutline(img, { threshold: 0.5 });
    expect(pts.length).toBeGreaterThan(8);
    // area of a unit-diameter circle = pi/4
    expect(Math.abs(polygonArea(pts))).toBeCloseTo(Math.PI / 4, 1);
  });

  it('picks the largest shape and ignores specks', () => {
    const img = render(200, 200, (x, y) => (x > 20 && x < 120 && y > 20 && y < 120) || (x > 170 && x < 176 && y > 170 && y < 176));
    const pts = traceOutline(img, { threshold: 0.5 });
    expect(Math.max(...pts.map((p) => p.x))).toBeCloseTo(1, 1);
    expect(Math.max(...pts.map((p) => p.y))).toBeCloseTo(1, 1);
  });

  it('returns nothing for a blank image and supports inversion', () => {
    expect(traceOutline(render(60, 60, () => false), { threshold: 0.5 })).toEqual([]);
    const light = render(120, 120, (x, y) => !(x > 30 && x < 90 && y > 30 && y < 90));
    expect(traceOutline(light, { threshold: 0.5, invert: true }).length).toBeGreaterThanOrEqual(4);
  });

  it('also reports the outline in image space so it can be drawn over the photo', () => {
    const img = render(300, 150, (x, y) => x > 150 && x < 270 && y > 30 && y < 120);
    const { points, inImage } = traceOutlineDetailed(img, { threshold: 0.5 });
    expect(points.length).toBe(inImage.length);
    // the shape sits in the right half, lower part of the image
    expect(Math.min(...inImage.map((p) => p.x))).toBeCloseTo(150 / 300, 1);
    expect(Math.max(...inImage.map((p) => p.x))).toBeCloseTo(270 / 300, 1);
    expect(Math.min(...inImage.map((p) => p.y))).toBeCloseTo(30 / 150, 1);
  });

  it('converts RGBA to luminance and treats transparency as background', () => {
    const rgba = new Uint8ClampedArray([0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 0]);
    const g = toGray(rgba, 3, 1);
    expect(g.data[0]).toBeCloseTo(0);
    expect(g.data[1]).toBeCloseTo(1);
    expect(g.data[2]).toBeCloseTo(1);
  });
});
