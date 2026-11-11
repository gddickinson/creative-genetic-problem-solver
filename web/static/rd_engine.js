// JS port of reaction-diffusion.py (ReactionDiffusionSimulator): Gray-Scott model on a periodic grid, explicit Euler
// steps with values clipped to [0, 1], and the script's pattern metrics. Grids are Float64Array, row-major.

export const PATTERNS = { spots: [0.035, 0.065], stripes: [0.030, 0.060], waves: [0.025, 0.055], chaos: [0.020, 0.050] };
export const INITS = [['random', 'Random spots'], ['center', 'Single central spot'], ['stripes', 'Vertical stripes'], ['checkerboard', 'Checkerboard']];

export class GrayScott {
  constructor(width = 100, height = 100, R) {
    Object.assign(this, { W: width, H: height, R, Du: 0.16, Dv: 0.08, F: 0.035, k: 0.065, dt: 1.0, stepsPerFrame: 10 });
    this.U = new Float64Array(width * height).fill(1); this.V = new Float64Array(width * height);
    this.metrics = []; this.steps = 0;
  }
  setParameters(name) { if (PATTERNS[name]) [this.F, this.k] = PATTERNS[name]; }
  fillV(r0, r1, c0, c1) {                         // V[r0:r1, c0:c1] = 1 (NumPy slice bounds)
    for (let r = Math.max(0, r0); r < Math.min(this.H, r1); r++) for (let c = Math.max(0, c0); c < Math.min(this.W, c1); c++) this.V[r * this.W + c] = 1;
  }
  initializePattern(type = 'random') {
    const cx = Math.floor(this.W / 2), cy = Math.floor(this.H / 2);
    if (type === 'random') {
      for (let n = 0; n < 10; n++) { const x = this.R.randint(cx - 5, cx + 5), y = this.R.randint(cy - 5, cy + 5); this.fillV(y - 3, y + 3, x - 3, x + 3); }
    } else if (type === 'center') this.fillV(cy - 3, cy + 3, cx - 3, cx + 3);
    else if (type === 'stripes') for (let x = 0; x < this.W; x += 10) this.fillV(0, this.H, x, x + 3);
    else if (type === 'checkerboard') for (let i = 0; i < this.H; i += 10) for (let j = 0; j < this.W; j += 10) this.fillV(i, i + 5, j, j + 5);
  }
  lap(X, i, r, c) {
    const W = this.W, H = this.H;
    const top = X[((r - 1 + H) % H) * W + c], bottom = X[((r + 1) % H) * W + c];
    const left = X[r * W + (c - 1 + W) % W], right = X[r * W + (c + 1) % W];
    return top + bottom + left + right - 4 * X[i];
  }
  step() {
    const { U, V, W, H, Du, Dv, F, k, dt } = this, nU = new Float64Array(U.length), nV = new Float64Array(V.length);
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const i = r * W + c, uvv = U[i] * V[i] * V[i];
      const dU = Du * this.lap(U, i, r, c) - uvv + F * (1.0 - U[i]), dV = Dv * this.lap(V, i, r, c) + uvv - (F + k) * V[i];
      nU[i] = Math.min(1, Math.max(0, U[i] + dU * dt)); nV[i] = Math.min(1, Math.max(0, V[i] + dV * dt));
    }
    this.U = nU; this.V = nV; this.steps++;
  }
  // calculate_metrics: means, mean |np.gradient(V)|, and the count of cells above mean + std.
  calculateMetrics() {
    const { U, V, W, H } = this, n = W * H, mean = a => a.reduce((s, v) => s + v, 0) / n;
    const avgU = mean(U), avgV = mean(V);
    const g = (a, i, j, len, at) => (len < 2 ? 0 : i === 0 ? at(1) - at(0) : i === len - 1 ? at(len - 1) - at(len - 2) : (at(i + 1) - at(i - 1)) / 2);
    let comp = 0;
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const gy = g(V, r, c, H, k => V[k * W + c]), gx = g(V, c, r, W, k => V[r * W + k]);
      comp += Math.sqrt(gx * gx + gy * gy);
    }
    const sd = Math.sqrt(V.reduce((s, v) => s + (v - avgV) ** 2, 0) / n), thr = avgV + sd;
    const m = { avg_u: avgU, avg_v: avgV, complexity: comp / n, feature_count: V.reduce((s, v) => s + (v > thr ? 1 : 0), 0) };
    this.metrics.push(m);
    return m;
  }
  frame() { for (let s = 0; s < this.stepsPerFrame; s++) this.step(); return this.calculateMetrics(); }   // update_plot
}
