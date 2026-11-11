// JS port of portfolio-optimizer.py (PortfolioOptimizer): Sharpe-ratio GA over long-only weights, plus the random
// search for the efficient frontier. Market data: generateSampleData(R) ports generate_sample_data (used by the tests);
// the page uses the exact NumPy-generated data from the API instead.

export function generateSampleData(n, R) {
  R.seed(42);                                    // the script calls np.random.seed(42) here
  const u = R.uniformN(0.1, 0.7, n * n), corr = [];
  for (let i = 0; i < n; i++) corr.push(u.slice(i * n, i * n + n));
  const sym = corr.map((row, i) => row.map((v, j) => (i === j ? 1 : (v + corr[j][i]) / 2)));
  const vols = R.uniformN(0.1, 0.3, n);
  return { names: Array.from({ length: n }, (_, i) => `Asset ${i + 1}`), expected_returns: R.uniformN(0.05, 0.15, n),
    cov_matrix: sym.map((row, i) => row.map((c, j) => vols[i] * vols[j] * c)), risk_free_rate: 0.02 };
}

export class PortfolioGA {
  constructor(market, { populationSize = 100 } = {}, R) {
    Object.assign(this, { populationSize, R });
    this.er = market.expected_returns; this.cov = market.cov_matrix; this.rf = market.risk_free_rate; this.n = this.er.length;
    this.history = []; this.best = null; this.bestSharpe = -Infinity; this.generation = 0; this.frontier = null;
  }
  metrics(w) {
    const ret = w.reduce((s, v, i) => s + this.er[i] * v, 0);
    let q = 0;
    for (let i = 0; i < this.n; i++) { let row = 0; for (let j = 0; j < this.n; j++) row += this.cov[i][j] * w[j]; q += w[i] * row; }
    const vol = Math.sqrt(q);
    return { weights: w, ret, vol, sharpe: (ret - this.rf) / vol };
  }
  randomPortfolio() { const w = this.R.randomN(this.n), s = w.reduce((a, b) => a + b, 0); return w.map(v => v / s); }
  crossover(a, b) { const c = a.map((v, i) => (v + b[i]) / 2), s = c.reduce((x, y) => x + y, 0); return c.map(v => v / s); }
  mutate(w, rate = 0.1) {
    const noise = this.R.normalN(0, 0.1, this.n), mask = this.R.randomN(this.n);
    const m = w.map((v, i) => Math.max(v + noise[i] * (mask[i] < rate ? 1 : 0), 0)), s = m.reduce((x, y) => x + y, 0);
    return m.map(v => v / s);
  }
  // calculate_efficient_frontier: for each target return, the lowest volatility among 1000 random portfolios within 0.001.
  efficientFrontier(nPoints = 100, tries = 1000) {
    const lo = Math.min(...this.er), hi = Math.max(...this.er), out = [];
    for (let k = 0; k < nPoints; k++) {
      const target = nPoints === 1 ? lo : lo + (hi - lo) * k / (nPoints - 1);
      let best = Infinity, found = false;
      for (let t = 0; t < tries; t++) {
        const p = this.metrics(this.randomPortfolio());
        if (Math.abs(p.ret - target) < 0.001 && p.vol < best) { best = p.vol; found = true; }
      }
      if (found) out.push([target, best]);
    }
    return (this.frontier = out);
  }
  // One generation of optimize_portfolio.
  step() {
    if (!this.population) this.population = Array.from({ length: this.populationSize }, () => this.randomPortfolio());
    const ps = this.population.map(w => this.metrics(w)).sort((a, b) => b.sharpe - a.sharpe);
    if (ps[0].sharpe > this.bestSharpe) { this.bestSharpe = ps[0].sharpe; this.best = ps[0]; }
    this.current = ps; this.history.push(ps[0]);
    const next = ps.slice(0, Math.floor(this.populationSize / 10)).map(p => p.weights);
    const top = ps.slice(0, 20), pick = () => { const [a, b] = this.R.sample(top, 2); return (b.sharpe > a.sharpe ? b : a).weights; };
    while (next.length < this.populationSize) {
      const p1 = pick(), p2 = pick();
      next.push(this.mutate(this.crossover(p1, p2)));
    }
    this.population = next; this.generation++;
  }
}
