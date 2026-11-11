// Node test: maze_engine.js reproduces maze-pathfinder.py (tests/fixtures/maze.json).
// Run: node web/static/maze_engine.test.mjs
import { Rng } from './ga_rng.js';
import { Maze } from './maze_engine.js';
import { fixture, near } from './util.test.mjs';

for (const r of fixture('maze').runs) {
  const m = new Maze(r.w, r.h, r.pop, new Rng(r.seed));
  near(m.generate(), r.maze, `seed ${r.seed} maze`);
  const a = m.astar(); near(a.path, r.astar, 'A* path'); near(a.steps.map(s => s.cur), r.astar_steps, 'A* steps');
  const w = m.wave(); near(w.path, r.wave, 'BFS path'); near(w.steps.map(s => s.cur), r.wave_steps, 'BFS steps');
  const g = m.genetic();
  near(g.path, r.ga, 'GA path');
  near(g.gens.reduce((n, x) => n + x.best.length, 0), r.ga_steps, 'GA steps');
}
console.log('maze_engine: ok');
