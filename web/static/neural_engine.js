// JS port of neural-vis.py (NeuralNetVisualizer): datasets, forward pass (ReLU hidden layers, sigmoid output) and
// genetic_train (weights evolved by uniform crossover and sparse gaussian mutation; fitness = accuracy).
// A network is [{ W: rows x cols array, b: [cols] }], same random draw order as NumPy's row-major randn.

export const PATTERNS = [['spiral', 'Two interleaving spirals (challenging)'], ['circles', 'Concentric circles (tests radial separation)'],
  ['moons', 'Two half-moons (tests curved boundaries)'], ['xor', 'XOR pattern (tests logical operations)'],
  ['squares', 'Nested squares (tests straight boundaries)'], ['gaussian', 'Gaussian clusters (tests basic clustering)']];

const linspace = (a, b, n, end = true) => Array.from({ length: n }, (_, i) =>
  (end && i === n - 1 && n > 1) ? b : a + i * (b - a) / (end ? Math.max(1, n - 1) : n));

export function randomNetwork(sizes, R) {
  const net = [];
  for (let i = 0; i < sizes.length - 1; i++) {
    const w = R.normalN(0, 1, sizes[i] * sizes[i + 1]).map(v => v * 0.1), b = R.normalN(0, 1, sizes[i + 1]).map(v => v * 0.1);
    net.push({ W: Array.from({ length: sizes[i] }, (_, r) => w.slice(r * sizes[i + 1], (r + 1) * sizes[i + 1])), b });
  }
  return net;
}

export function forward(x, net) {          // one input point -> output in (0, 1)
  let cur = x;
  net.forEach((L, li) => {
    const out = L.b.slice();
    for (let r = 0; r < cur.length; r++) { const v = cur[r], row = L.W[r]; for (let c = 0; c < out.length; c++) out[c] += v * row[c]; }
    cur = li === net.length - 1 ? out.map(z => 1 / (1 + Math.exp(-z))) : out.map(z => Math.max(0, z));
  });
  return cur[0];
}

export function generateData(n, pattern, R) {
  const ppc = Math.floor(n / 2), X = Array.from({ length: n }, () => [0, 0]), y = new Array(n).fill(0);
  const randn2 = k => { const v = R.normalN(0, 1, k * 2); return Array.from({ length: k }, (_, i) => [v[2 * i], v[2 * i + 1]]); };
  const put = (start, pts) => pts.forEach((p, i) => { X[start + i] = p; });
  if (pattern === 'spiral') {
    for (let i = 0; i < 2; i++) {
      const t = linspace(0, 4 * Math.PI, ppc), noise = R.normalN(0, 1, ppc);
      put(i * ppc, t.map((tt, k) => { const r = tt + noise[k] * 0.2, a = tt + i * Math.PI; return [r * Math.cos(a), r * Math.sin(a)]; }));
      for (let k = 0; k < ppc; k++) y[i * ppc + k] = i;
    }
  } else if (pattern === 'circles') {
    for (let i = 0; i < 2; i++) {
      const rn = R.normalN(0, 1, ppc), base = linspace(0, 2 * Math.PI, ppc), tn = R.normalN(0, 1, ppc);
      put(i * ppc, base.map((b, k) => { const r = i + 1 + rn[k] * 0.1, t = b + tn[k] * 0.2; return [r * Math.cos(t), r * Math.sin(t)]; }));
      for (let k = 0; k < ppc; k++) y[i * ppc + k] = i;
    }
  } else if (pattern === 'moons') {
    const t = linspace(0, Math.PI, ppc), a = randn2(ppc);
    put(0, t.map((tt, k) => [Math.cos(tt) + a[k][0] * 0.1, Math.sin(tt) + a[k][1] * 0.1]));
    const b = randn2(ppc);
    put(ppc, t.map((tt, k) => [1 - Math.cos(tt) + b[k][0] * 0.1, 0.5 - Math.sin(tt) + b[k][1] * 0.1]));
    for (let k = ppc; k < n; k++) y[k] = 1;
  } else if (pattern === 'xor') {
    const h = Math.floor(ppc / 2), centres = [[-1, -1], [1, 1], [-1, 1], [1, -1]];
    centres.forEach((c, q) => put(q * h, randn2(h).map(([u, v]) => [u * 0.2 + c[0], v * 0.2 + c[1]])));
    for (let k = ppc; k < n; k++) y[k] = 1;
  } else if (pattern === 'squares') {
    const square = (size, off, m) => {
      const pts = [];
      for (let e = 0; e < 4; e++) {
        const k = Math.floor(m / 4) + (e < m % 4 ? 1 : 0), t = linspace(0, 1, k, false);
        t.forEach(tt => pts.push(e === 0 ? [tt * size, 0] : e === 1 ? [size, tt * size] : e === 2 ? [(1 - tt) * size, size] : [0, (1 - tt) * size]));
      }
      return pts.map(([u, v]) => [u + off, v + off]);
    };
    const inner = square(1, -0.5, ppc), ni = randn2(ppc);
    put(0, inner.map((p, k) => [p[0] + ni[k][0] * 0.05, p[1] + ni[k][1] * 0.05]));
    const outer = square(2, -1, n - ppc), no = randn2(n - ppc);
    put(ppc, outer.map((p, k) => [p[0] + no[k][0] * 0.05, p[1] + no[k][1] * 0.05]));
    for (let k = ppc; k < n; k++) y[k] = 1;
  } else if (pattern === 'gaussian') {
    const centres = [[-1, -1], [1, 1]];
    for (let i = 0; i < 2; i++) {
      put(i * ppc, randn2(ppc).map(([u, v]) => [u * 0.2 + centres[i][0], v * 0.2 + centres[i][1]]));
      for (let k = 0; k < ppc; k++) y[i * ppc + k] = i;
    }
  }
  for (const d of [0, 1]) {                                      // normalise each column (population std)
    const m = X.reduce((s, p) => s + p[d], 0) / n, sd = Math.sqrt(X.reduce((s, p) => s + (p[d] - m) ** 2, 0) / n);
    X.forEach(p => { p[d] = (p[d] - m) / sd; });
  }
  return { X, y };
}

export const accuracy = (net, X, y) => X.reduce((s, x, i) => s + ((forward(x, net) > 0.5 ? 1 : 0) === y[i] ? 1 : 0), 0) / X.length;

export class NeuroGA {
  constructor(sizes, X, y, { populationSize = 50 } = {}, R) {
    Object.assign(this, { sizes, X, y, populationSize, R });
    this.history = []; this.best = null; this.bestFitness = -Infinity; this.generation = 0;
  }
  crossover(a, b) {
    return a.map((L, i) => {
      const mw = this.R.randomN(L.W.length * L.W[0].length), mb = this.R.randomN(L.b.length), cols = L.W[0].length;
      return { W: L.W.map((row, r) => row.map((v, c) => (mw[r * cols + c] < 0.5 ? v : b[i].W[r][c]))),
        b: L.b.map((v, c) => (mb[c] < 0.5 ? v : b[i].b[c])) };
    });
  }
  mutate(net, rate = 0.1) {
    return net.map(L => {
      const cols = L.W[0].length, n = L.W.length * cols;
      const mask = this.R.randomN(n), noise = this.R.normalN(0, 1, n);
      const W = L.W.map((row, r) => row.map((v, c) => (mask[r * cols + c] < 0.1 ? v + noise[r * cols + c] * rate : v)));
      const mb = this.R.randomN(L.b.length), nb = this.R.normalN(0, 1, L.b.length);
      return { W, b: L.b.map((v, c) => (mb[c] < 0.1 ? v + nb[c] * rate : v)) };
    });
  }
  // One generation of genetic_train.
  step() {
    if (!this.population) this.population = Array.from({ length: this.populationSize }, () => randomNetwork(this.sizes, this.R));
    const scored = this.population.map(net => ({ net, f: accuracy(net, this.X, this.y) })).sort((a, b) => b.f - a.f);
    if (scored[0].f > this.bestFitness) { this.bestFitness = scored[0].f; this.best = scored[0].net; }
    this.history.push(this.bestFitness);
    const next = scored.slice(0, 2).map(s => s.net), top = scored.slice(0, 10).map(s => s.net);
    while (next.length < this.populationSize) { const [a, b] = this.R.sample(top, 2); next.push(this.mutate(this.crossover(a, b))); }
    this.population = next; this.generation++;
  }
}
