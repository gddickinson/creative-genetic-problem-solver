// Portfolio page: PortfolioOptimizer's Sharpe-ratio GA (portfolio_engine.js) on the script's synthetic market,
// fetched from the API exactly as NumPy generates it (np.random.seed(42) in generate_sample_data).
import { route, api } from './core.js';
import { Rng } from './ga_rng.js';
import { PortfolioGA } from './portfolio_engine.js';
import { lab } from './lab.js';
import { fitCanvas, frame, fmt, COLORS } from './chart.js';

route('portfolio', async () => {
  const sizes = Array.from({ length: 11 }, (_, i) => i + 2);
  const markets = Object.fromEntries(await Promise.all(sizes.map(async n => [n, await api(`/portfolio/market/${n}`)])));
  let ga, market;
  const L = lab({
    kicker: 'Weights GA · Sharpe ratio · efficient frontier', title: 'Portfolio Optimizer',
    lead: `Split money across synthetic assets to maximise the Sharpe ratio, (expected return − 2% risk-free rate) /
      volatility. Each individual is a vector of non-negative weights summing to 1. The market (returns 5-15%,
      volatilities 10-30%, correlations 0.1-0.7) is the script's own generator with NumPy seed 42, computed by the Python
      server; the seed slider only changes the GA's random numbers.`,
    controls: [
      { id: 'n', label: 'Number of assets', type: 'select', value: 5, options: sizes.map(n => [n, `${n} assets`]) },
      { id: 'pop', label: 'Population size', min: 20, max: 300, step: 10, value: 100 },
      { id: 'gens', label: 'Generations', min: 10, max: 300, step: 10, value: 100 },
    ],
    stages: [{ id: 'rr', w: 600, h: 380, label: 'Risk-return plane: current population (pink), best portfolio (cyan), efficient frontier from random search (dashed), single assets (lime)' },
      { id: 'alloc', w: 600, h: 200, label: 'Current best allocation' }],
    chartLabel: 'Best Sharpe ratio per generation', source: 'portfolio-optimizer.py',
    onReset(p) {
      market = markets[+p.n];
      ga = new PortfolioGA(market, { populationSize: p.pop }, new Rng(p.seed));
      ga.efficientFrontier(); ga.maxGen = p.gens;     // the desktop app computes the frontier first, then evolves
    },
    step: () => ga.step(), done: () => ga.generation >= ga.maxGen, render,
    notes: `<h2>How this solver works</h2><div class="ops">
      <div class="card"><b>Population</b><span>Random weights (uniform, then normalised to sum to 1).</span></div>
      <div class="card"><b>Fitness</b><span>Sharpe ratio: (wᵀμ − 0.02) / √(wᵀΣw).</span></div>
      <div class="card"><b>Selection</b><span>Two-way tournaments among the 20 best portfolios.</span></div>
      <div class="card"><b>Crossover</b><span>Average of the two parents' weights, renormalised.</span></div>
      <div class="card"><b>Mutation</b><span>Each weight, with probability 0.1, gets gaussian noise (σ 0.1); negatives are set to 0 and the vector renormalised.</span></div>
      <div class="card"><b>Elitism</b><span>The best 10% of portfolios survive unchanged.</span></div></div>
      <p class="small mute">The frontier is not solved exactly: for 100 target returns the script keeps the lowest
      volatility among 1,000 random portfolios whose return is within 0.1 percentage point of the target.</p>`,
  });

  function render() {
    const cv = L.canvas('rr'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H);
    const vols = market.cov_matrix.map((r, i) => Math.sqrt(r[i])), rets = market.expected_returns;
    const pts = (ga.current || []).concat(ga.best ? [ga.best] : []);
    const vx = [...vols, ...pts.map(p => p.vol)], ry = [...rets, ...pts.map(p => p.ret)];
    const f = frame(cv, 0, Math.max(...vx) * 1.08, Math.min(...ry) * 0.9, Math.max(...ry) * 1.05, 0.1);
    ctx.strokeStyle = '#212c4c'; ctx.fillStyle = '#94a3c6'; ctx.font = `${10 * s}px ui-monospace,Menlo,monospace`;
    ctx.fillText('volatility →', W - 90 * s, H - 8 * s); ctx.fillText('↑ expected return', 8 * s, 14 * s);
    if (ga.frontier.length) {
      ctx.setLineDash([5 * s, 4 * s]); ctx.strokeStyle = '#e8f1ff'; ctx.lineWidth = 1.2 * s; ctx.beginPath();
      ga.frontier.forEach(([r, v], i) => (i ? ctx.lineTo(f.X(v), f.Y(r)) : ctx.moveTo(f.X(v), f.Y(r)))); ctx.stroke(); ctx.setLineDash([]);
    }
    vols.forEach((v, i) => { ctx.fillStyle = '#b6ff3d'; ctx.fillRect(f.X(v) - 4 * s, f.Y(rets[i]) - 4 * s, 8 * s, 8 * s);
      ctx.fillText(String(i + 1), f.X(v) + 6 * s, f.Y(rets[i]) + 4 * s); });
    for (const p of ga.current || []) { ctx.fillStyle = 'rgba(255,79,216,.55)'; ctx.beginPath(); ctx.arc(f.X(p.vol), f.Y(p.ret), 3 * s, 0, 7); ctx.fill(); }
    if (ga.best) { ctx.fillStyle = '#3dd6ff'; ctx.beginPath(); ctx.arc(f.X(ga.best.vol), f.Y(ga.best.ret), 7 * s, 0, 7); ctx.fill(); }
    bars(L.canvas('alloc'), ga.best ? ga.best.weights : [], market.names);
    L.chart([{ data: ga.history.map(p => p.sharpe), label: 'best Sharpe' }], { yLabel: 'Sharpe ratio', x0: 1 });
    const b = ga.best;
    L.setReadout([['Generation', `${ga.generation} / ${ga.maxGen}`], ['Best Sharpe ratio', b ? fmt(b.sharpe) : '-'],
      ['Expected return', b ? (b.ret * 100).toFixed(1) + '%' : '-'], ['Volatility', b ? (b.vol * 100).toFixed(1) + '%' : '-'],
      ['Diversification (1 − Σw²)', b ? ((1 - b.weights.reduce((a, w) => a + w * w, 0)) * 100).toFixed(1) + '%' : '-'],
      ['Largest position', b ? (Math.max(...b.weights) * 100).toFixed(1) + '%' : '-']]);
    L.setStatus(ga.generation >= ga.maxGen ? 'Finished.' : ga.generation ? 'Evolving...' : 'Frontier ready. Press Run or Step.');
  }
});

function bars(cv, w, names) {
  const ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600, n = names.length, bw = (W - 20 * s) / n;
  ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H); ctx.font = `${10 * s}px ui-monospace,Menlo,monospace`; ctx.textAlign = 'center';
  names.forEach((name, i) => {
    const v = w[i] || 0, h = (H - 40 * s) * v;
    ctx.fillStyle = COLORS[i % COLORS.length]; ctx.fillRect(10 * s + i * bw + 3 * s, H - 22 * s - h, bw - 6 * s, h);
    ctx.fillStyle = '#94a3c6'; ctx.fillText(name.replace('Asset ', 'A'), 10 * s + (i + 0.5) * bw, H - 8 * s);
    if (v) ctx.fillText((v * 100).toFixed(0) + '%', 10 * s + (i + 0.5) * bw, H - 26 * s - h);
  });
}
