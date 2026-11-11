// Node test: neural_engine.js reproduces neural-vis.py (tests/fixtures/neural.json).
// Run: node web/static/neural_engine.test.mjs
import { Rng } from './ga_rng.js';
import { randomNetwork, generateData, forward, NeuroGA } from './neural_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('neural').runs) {
  const R = new Rng(r.seed), init = randomNetwork(r.sizes, R);      // __init__ builds self.network first
  near(init, r.init, 'initial network');
  const { X, y } = generateData(r.n, r.pattern, R);
  near(X, r.X, `${r.pattern} data`); near(y, r.y, 'labels');
  const ga = new NeuroGA(r.sizes, X, y, { populationSize: r.pop }, R);
  for (let g = 0; g < r.gens; g++) ga.step();
  near(ga.history, r.history, `${r.pattern} history`);
  near(ga.best, r.best, 'best network');
  near(X.slice(0, 5).map(x => forward(x, ga.best)), r.probe, 'forward');
}
console.log('neural_engine: ok');
