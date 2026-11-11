// Small canvas charts shared by the problem pages: a multi-series line chart (fitness over generations) and helpers
// for mapping data coordinates to a canvas.

export const COLORS = ['#b6ff3d', '#ff4fd8', '#3dd6ff', '#ffb43d', '#9d7bff', '#ff6b6b', '#5dffb0'];

// Fit a canvas's backing store to its displayed width (keeps the aspect ratio from its width/height attributes).
export function fitCanvas(cv) {
  const ratio = cv.height / cv.width, w = Math.max(200, Math.round(cv.clientWidth || cv.width));
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(w * ratio * dpr); }
  return cv.getContext('2d');
}

// series: [{ data: number[], label, color }]; opts: { xLabel, yLabel, x0 }
export function lineChart(cv, series, { xLabel = 'Generation', yLabel = '', x0 = 0 } = {}) {
  const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
  ctx.clearRect(0, 0, W, H); ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
  const pad = { l: 58 * s, r: 12 * s, t: 14 * s, b: 34 * s };
  const vals = series.flatMap(d => d.data.filter(Number.isFinite));
  const n = Math.max(2, ...series.map(d => d.data.length));
  let lo = vals.length ? Math.min(...vals) : 0, hi = vals.length ? Math.max(...vals) : 1;
  if (hi - lo < 1e-12) { hi += Math.abs(hi) * 0.05 + 1e-6; lo -= Math.abs(lo) * 0.05 + 1e-6; }
  const X = i => pad.l + (W - pad.l - pad.r) * i / (n - 1), Y = v => H - pad.b - (H - pad.t - pad.b) * (v - lo) / (hi - lo);
  ctx.strokeStyle = '#212c4c'; ctx.lineWidth = 1; ctx.font = `${11 * s}px ui-monospace,Menlo,monospace`; ctx.fillStyle = '#94a3c6';
  for (let k = 0; k <= 4; k++) {
    const v = lo + (hi - lo) * k / 4, y = Y(v);
    ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(W - pad.r, y); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(fmt(v), pad.l - 6 * s, y + 4 * s);
  }
  ctx.textAlign = 'center';
  for (let k = 0; k <= 4; k++) { const i = Math.round((n - 1) * k / 4); ctx.fillText(String(i + x0), X(i), H - pad.b + 15 * s); }
  ctx.fillText(xLabel, (pad.l + W - pad.r) / 2, H - 4 * s);
  ctx.save(); ctx.translate(12 * s, (pad.t + H - pad.b) / 2); ctx.rotate(-Math.PI / 2); ctx.fillText(yLabel, 0, 0); ctx.restore();
  series.forEach((d, j) => {
    ctx.strokeStyle = d.color || COLORS[j % COLORS.length]; ctx.lineWidth = 2 * s; ctx.beginPath();
    let pen = false;
    d.data.forEach((v, i) => { if (!Number.isFinite(v)) { pen = false; return; } pen ? ctx.lineTo(X(i), Y(v)) : ctx.moveTo(X(i), Y(v)); pen = true; });
    ctx.stroke();
  });
  if (series.length > 1) {
    ctx.textAlign = 'left'; let x = pad.l + 6 * s;
    series.forEach((d, j) => { ctx.fillStyle = d.color || COLORS[j % COLORS.length]; ctx.fillText('■ ' + d.label, x, pad.t + 10 * s); x += ctx.measureText('■ ' + d.label).width + 14 * s; });
  }
}

export function fmt(v) {
  if (!Number.isFinite(v)) return String(v);
  const a = Math.abs(v);
  return a !== 0 && (a < 0.01 || a >= 1e5) ? v.toExponential(2) : (+v.toFixed(a < 1 ? 3 : a < 100 ? 2 : 1)).toString();
}

// Map a data box [x0,x1]x[y0,y1] onto the canvas with a margin; y grows upwards. Returns {X, Y, scale}.
export function frame(cv, x0, x1, y0, y1, margin = 0.06) {
  const W = cv.width, H = cv.height, m = margin * Math.min(W, H);
  const k = Math.min((W - 2 * m) / (x1 - x0), (H - 2 * m) / (y1 - y0));
  const ox = (W - k * (x1 - x0)) / 2, oy = (H - k * (y1 - y0)) / 2;
  return { X: x => ox + k * (x - x0), Y: y => H - oy - k * (y - y0), scale: k };
}
