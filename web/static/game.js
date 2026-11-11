// Game theory page: GameEvolutionVisualizer from game-evolution.py (game_engine.js).
import { route, esc, list } from './core.js';
import { Rng } from './ga_rng.js';
import { GAMES, STRATEGIES, GameEvolution } from './game_engine.js';
import { lab } from './lab.js';
import { fitCanvas, lineChart, fmt, COLORS } from './chart.js';

const PAY = g => `<table class="small"><tr><th></th><th>other cooperates</th><th>other defects</th></tr>
  <tr><th>cooperate</th><td>${g.pay.CC[0]}</td><td>${g.pay.CD[0]}</td></tr><tr><th>defect</th><td>${g.pay.DC[0]}</td><td>${g.pay.DD[0]}</td></tr></table>`;

route('game', () => {
  let g;
  const L = lab({
    kicker: 'Evolutionary game theory · fitness-proportional reproduction', title: 'Game Theory Evolution',
    lead: `A population of strategies plays a repeated two-player game: every individual meets every other for 10
      rounds. Each generation, individuals reproduce with probability proportional to their total payoff (shifted so the
      worst gets ~0), and a child's cooperation probability mutates with probability 0.1.`,
    controls: [
      { id: 'game', label: 'Game', type: 'select', value: 'prisoners_dilemma', options: Object.entries(GAMES).map(([k, v]) => [k, `${v.name}: ${v.desc}`]) },
      { id: 'pop', label: 'Population size', min: 10, max: 150, step: 10, value: 100 },
      { id: 'gens', label: 'Generations', min: 10, max: 300, step: 10, value: 100 },
    ],
    stages: [{ id: 'bars', w: 600, h: 230, label: 'Current strategy shares (bars) and mean shifted payoff per strategy (dots, right scale)' },
      { id: 'avg', w: 600, h: 180, label: 'Average population fitness (mean of the strategies’ payoffs)' }],
    chartLabel: 'Population share of each strategy', source: 'game-evolution.py', maxSpeed: 5,
    onReset(p) { g = new GameEvolution(p.game, { populationSize: p.pop }, new Rng(p.seed)); g.maxGen = p.gens; g.key = p.game; },
    step: () => g.step(), done: () => g.generation >= g.maxGen, render,
    notes: `<h2>Strategies as coded</h2>
      <p>All five strategies use one rule (<code>Strategy.decide</code>). Until the opponent has made
      <i>memory</i> moves, cooperate with probability <i>p</i>; after that, cooperate with probability
      <i>p</i> × (1 + c) / 2, where <i>c</i> is the opponent's cooperation rate over its last <i>memory</i> moves.
      The names come from the script; the desktop prompt describes Tit-for-Tat as reciprocating and Grudger as defecting
      for good after a defection, but in the code they differ only in <i>p</i> and <i>memory</i>. With memory 0
      (Random) the rate is taken over the opponent's whole history.</p>
      <table class="small"><tr><th>Strategy</th><th>p (start)</th><th>memory</th><th>mutation rate</th></tr>
      ${list(STRATEGIES, s => `<tr><td>${esc(s.name)}</td><td>${s.p}</td><td>${s.mem}</td><td>${s.mut}</td></tr>`)}</table>
      <h2>Payoffs (row player)</h2><div class="three">${list(Object.values(GAMES), v => `<div><h3>${esc(v.name)}</h3>${PAY(v)}</div>`)}</div>`,
  });

  function render() {
    const h = g.history.at(-1), names = STRATEGIES.map(s => s.name);
    const cv = L.canvas('bars'), ctx = fitCanvas(cv), W = cv.width, H = cv.height, s = W / 600, bw = (W - 40 * s) / names.length;
    ctx.fillStyle = '#060814'; ctx.fillRect(0, 0, W, H); ctx.font = `${10 * s}px ui-monospace,Menlo,monospace`; ctx.textAlign = 'center';
    const counts = Object.fromEntries(names.map(n => [n, g.population.filter(x => x.name === n).length / g.populationSize]));
    const perf = h ? h.perf : {}, pmax = Math.max(1e-9, ...Object.values(perf));
    names.forEach((n, i) => {
      const x = 20 * s + i * bw, bh = (H - 50 * s) * counts[n];
      ctx.fillStyle = COLORS[i]; ctx.fillRect(x + 8 * s, H - 26 * s - bh, bw - 16 * s, bh);
      ctx.fillStyle = '#94a3c6'; ctx.fillText(n, x + bw / 2, H - 10 * s); ctx.fillText((counts[n] * 100).toFixed(0) + '%', x + bw / 2, H - 30 * s - bh);
      if (h && h.dist[n] > 0) { ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x + bw / 2, H - 26 * s - (H - 50 * s) * perf[n] / pmax, 4 * s, 0, 7); ctx.fill(); }
    });
    lineChart(L.canvas('avg'), [{ data: g.history.map(x => Object.values(x.perf).reduce((a, b) => a + b, 0) / names.length), label: 'average fitness', color: '#e8f1ff' }], { yLabel: 'fitness', x0: 1 });
    L.chart(names.map((n, i) => ({ data: g.history.map(x => x.dist[n]), label: n, color: COLORS[i] })), { yLabel: 'share', x0: 1 });
    const dom = names.reduce((a, b) => (counts[b] > counts[a] ? b : a));
    L.setReadout([['Game', GAMES[g.key].name], ['Generation', `${g.generation} / ${g.maxGen}`], ['Dominant strategy', dom],
      ['Surviving strategies (≥1%)', names.filter(n => counts[n] >= 0.01).length],
      ['Mean cooperation probability', fmt(g.population.reduce((a, x) => a + x.p, 0) / g.populationSize)]]);
    L.setStatus(g.generation >= g.maxGen ? 'Finished.' : g.generation ? 'Evolving...' : 'Press Run or Step.');
  }
});
