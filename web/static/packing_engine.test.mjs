// Node test: packing_engine.js reproduces packing-optimizer.py (tests/fixtures/packing.json).
// Run: node web/static/packing_engine.test.mjs
import { Rng } from './ga_rng.js';
import { Packing, overlaps } from './packing_engine.js';
import { fixture, near } from './util.test.mjs';

const F = fixture('packing'), toItem = ([x, y, w, h, rot]) => ({ x, y, w, h, rot }), fromItem = i => [i.x, i.y, i.w, i.h, i.rot];
const shapes = F.shapes.map(toItem);
for (const [i, j, o] of F.overlaps) near(overlaps(shapes[i], shapes[j]), o, `overlap ${i},${j}`);
for (const r of F.runs) {
  const p = new Packing({ width: r.w, height: r.h, populationSize: r.pop, mutationRate: r.rate }, new Rng(r.seed));
  p.generateItems(r.kind, r.n);
  near(p.items.map(fromItem), r.items, 'items');
  for (let g = 0; g < r.gens; g++) p.step();
  near(p.history, r.history, `seed ${r.seed} history`);
  near(p.best ? p.best.map(fromItem) : null, r.best, 'best layout');
}
console.log('packing_engine: ok');
