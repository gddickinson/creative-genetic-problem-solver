// Maze page: MazeVisualizer from maze-pathfinder.py (maze_engine.js): one maze, three solvers, replayed step by step.
import { route } from './core.js';
import { Rng } from './ga_rng.js';
import { Maze } from './maze_engine.js';
import { lab } from './lab.js';
import { fitCanvas, fmt } from './chart.js';

const ALGS = [['genetic', 'Genetic algorithm'], ['astar', 'A* (Manhattan heuristic)'], ['wave', 'Wave propagation (BFS)']];

route('maze', () => {
  let m, run, alg, i;
  const L = lab({
    kicker: 'Path GA · A* · breadth-first search', title: 'Maze Pathfinder',
    lead: `A maze is carved by recursive backtracking from the top-left cell, a few extra walls are knocked out, then it
      is solved from the start (top left) to the goal (bottom right) by one of three methods. Use the same seed to compare
      them on the same maze.`,
    controls: [
      { id: 'alg', label: 'Algorithm', type: 'select', value: 'genetic', options: ALGS },
      { id: 'size', label: 'Maze size (width = height)', min: 11, max: 61, step: 1, value: 20 },
      { id: 'pop', label: 'GA population size', min: 10, max: 150, step: 10, value: 50 },
    ],
    stages: [{ id: 'grid', w: 600, h: 600, label: 'Walls dark; explored cells blue; current cell yellow; start green, goal red; path cyan' }],
    chartLabel: 'GA: best fitness per generation · A*/BFS: cells discovered per step', source: 'maze-pathfinder.py', maxSpeed: 40,
    onReset(p) {
      m = new Maze(p.size, p.size, p.pop, new Rng(p.seed)); m.generate(); alg = p.alg; i = 0;
      run = alg === 'astar' ? m.astar() : alg === 'wave' ? m.wave() : m.genetic();
      run.frames = alg === 'genetic' ? run.gens.length : run.steps.length;
    },
    step: () => { i++; }, done: () => i >= run.frames, render,
    notes: `<h2>The three solvers</h2><div class="ops">
      <div class="card"><b>A*</b><span>Priority queue on steps taken + Manhattan distance to the goal; a cell is re-queued when a shorter route to it is found.</span></div>
      <div class="card"><b>Wave propagation</b><span>Breadth-first search: expands cells in order of distance from the start, so its path is a shortest one.</span></div>
      <div class="card"><b>GA population</b><span>Greedy random walks: always step to the unvisited open neighbour closest to the goal; if none, to a random open neighbour (up to width × height steps).</span></div>
      <div class="card"><b>GA fitness</b><span>+1000 if the walk ends on the goal, minus the distance left, minus 0.5 per step, minus 2 per revisited cell.</span></div>
      <div class="card"><b>GA operators</b><span>Two-way tournaments among the 20 best; crossover splices the parents at a shared cell; with probability 0.1 the child is cut at a random point and continued by a new greedy walk; the 2 best are kept. Up to 100 generations, stopping once the best walk reaches the goal.</span></div></div>
      <p class="small mute">The greedy walk usually reaches the goal in the first generation on small mazes, so the GA often
      finishes at once; larger mazes give it work to do. The GA's path is valid but rarely the shortest.</p>`,
  });

  function render() {
    const cv = L.canvas('grid'), ctx = fitCanvas(cv), W = cv.width, n = m.width, k = W / n;
    const fill = (r, c, color) => { ctx.fillStyle = color; ctx.fillRect(c * k, r * k, Math.ceil(k), Math.ceil(k)); };
    ctx.fillStyle = '#c6d4ee'; ctx.fillRect(0, 0, W, cv.height);
    m.maze.forEach((row, r) => row.forEach((v, c) => { if (v) fill(r, c, '#0b0f22'); }));
    const f = Math.min(i, run.frames - 1), finished = i >= run.frames;
    let path = [], cur = null, chart;
    if (alg === 'genetic') {
      const g = run.gens[f];
      g.explored.forEach(p => p.forEach(([r, c]) => fill(r, c, 'rgba(61,214,255,.45)')));
      path = g.best; chart = run.gens.slice(0, f + 1).map(x => x.bestF);
    } else {
      const s = run.steps[f];
      run.order.slice(0, s.seen).forEach(([r, c]) => fill(r, c, 'rgba(61,214,255,.45)'));
      cur = s.cur; if (finished) path = run.path; chart = run.steps.slice(0, f + 1).map(x => x.seen);
    }
    if (path.length) {
      ctx.strokeStyle = alg === 'genetic' && !finished ? '#ff4fd8' : '#00e5ff'; ctx.lineWidth = Math.max(2, k * 0.35); ctx.lineJoin = 'round'; ctx.beginPath();
      path.forEach(([r, c], j) => (j ? ctx.lineTo((c + 0.5) * k, (r + 0.5) * k) : ctx.moveTo((c + 0.5) * k, (r + 0.5) * k))); ctx.stroke();
    }
    if (cur) fill(cur[0], cur[1], '#ffd23f');
    fill(...m.start, '#3dff8f'); fill(...m.end, '#ff5c7a');
    L.chart([{ data: chart, label: alg }], { yLabel: alg === 'genetic' ? 'best fitness' : 'cells discovered', xLabel: alg === 'genetic' ? 'Generation' : 'Step', x0: 1 });
    const found = alg === 'genetic' ? run.path : run.path.length ? run.path : null;
    L.setReadout([['Algorithm', ALGS.find(a => a[0] === alg)[1]], [alg === 'genetic' ? 'Generation' : 'Step', `${f + 1} / ${run.frames}`],
      ['Path found', finished ? (found ? `yes, ${found.length} cells` : 'no') : '…'],
      alg === 'genetic' ? ['Best fitness', fmt(run.gens[f].bestF)] : ['Cells discovered', run.steps[f].seen]]);
    L.setStatus(finished ? 'Finished.' : i ? 'Solving...' : 'Press Run or Step.');
  }
});
