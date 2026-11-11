// Heat-map rendering of a 2-D function over a box (used by the creative-problem landscape and decision boundaries).
// Colour ramp: deep indigo (low) -> magenta -> lime (high).
const STOPS = [[8, 10, 30], [60, 30, 120], [255, 79, 216], [255, 180, 60], [182, 255, 61]];

export function ramp(t) {
  t = Math.min(1, Math.max(0, t)) * (STOPS.length - 1);
  const i = Math.min(STOPS.length - 2, Math.floor(t)), f = t - i, a = STOPS[i], b = STOPS[i + 1];
  return [0, 1, 2].map(k => Math.round(a[k] + (b[k] - a[k]) * f));
}

// Evaluate f on an n x n grid over [x0,x1]x[y0,y1] (row 0 = top = y1). Returns {vals, lo, hi, n}.
export function grid(f, [x0, x1], [y0, y1], n = 120) {
  const vals = new Float64Array(n * n);
  let lo = Infinity, hi = -Infinity;
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    const v = f([x0 + (x1 - x0) * (c + 0.5) / n, y1 - (y1 - y0) * (r + 0.5) / n]);
    vals[r * n + c] = v; if (v < lo) lo = v; if (v > hi) hi = v;
  }
  return { vals, lo, hi, n };
}

// Paint the grid into an offscreen canvas (cached by the caller) and return it.
export function paint(g, norm = v => (v - g.lo) / (g.hi - g.lo || 1)) {
  const off = document.createElement('canvas'); off.width = off.height = g.n;
  const ctx = off.getContext('2d'), img = ctx.createImageData(g.n, g.n);
  g.vals.forEach((v, i) => { const [r, gg, b] = ramp(norm(v)); img.data.set([r, gg, b, 255], i * 4); });
  ctx.putImageData(img, 0, 0);
  return off;
}
