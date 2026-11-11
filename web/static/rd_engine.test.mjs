// Node test: rd_engine.js reproduces reaction-diffusion.py (tests/fixtures/rd.json).
// Run: node web/static/rd_engine.test.mjs
import { Rng } from './ga_rng.js';
import { GrayScott } from './rd_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('rd').runs) {
  const g = new GrayScott(r.size, r.size, new Rng(r.seed));
  g.setParameters(r.pattern); g.initializePattern(r.init);
  near(g.V.reduce((a, b) => a + b, 0), r.v0, 'initial V');
  for (let f = 0; f < r.frames; f++) g.frame();
  near(g.metrics, r.metrics, `${r.pattern}/${r.init} metrics`, 1e-8);
  near(Array.from(g.V.slice(r.size * Math.floor(r.size / 2), r.size * (Math.floor(r.size / 2) + 1))), r.V_row, 'V row');
  const u = Math.floor(r.size / 3);
  near(Array.from(g.U.slice(r.size * u, r.size * (u + 1))), r.U_row, 'U row');
}
console.log('rd_engine: ok');
