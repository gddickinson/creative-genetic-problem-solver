// Node test: packing_engine.js reproduces packing-optimizer.py (tests/fixtures/packing.json).
// Run: node web/static/packing_engine.test.mjs
import { Rng } from './ga_rng.js';
import { Packing, overlaps, overlapDepth } from './packing_engine.js';
import { fixture, near } from './util.test.mjs';

const F = fixture('packing'), toItem = ([x, y, w, h, rot]) => ({ x, y, w, h, rot }), fromItem = i => [i.x, i.y, i.w, i.h, i.rot];
const shapes = F.shapes.map(toItem);
for (const [i, j, o, d] of F.overlaps) {
  near(overlaps(shapes[i], shapes[j]), o, `overlap ${i},${j}`); near(overlapDepth(shapes[i], shapes[j]), d, `depth ${i},${j}`);
}
{ const p = new Packing({ width: 10, height: 10 }, new Rng(1)); near(p.score(shapes), F.score, 'ranking score'); near(p.violations(shapes), F.violations, 'violations'); }
for (const r of F.runs) {
  const p = new Packing({ width: r.w, height: r.h, populationSize: r.pop, mutationRate: r.rate }, new Rng(r.seed));
  p.generateItems(r.kind, r.n);
  near(p.items.map(fromItem), r.items, 'items');
  for (let g = 0; g < r.gens; g++) p.step();
  near(p.history, r.history, `seed ${r.seed} history`);
  near(p.best ? p.best.map(fromItem) : null, r.best, 'best layout');
}
console.log('packing_engine: ok');
