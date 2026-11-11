// JS port of maze-pathfinder.py (MazeVisualizer): recursive-backtracking maze, A*, BFS wave propagation and the GA.
// Cells are [row, col]; maze[r][c] = 1 wall, 0 open. Same random calls as the Python, in the same order.
const DIRS = [[0, 1], [1, 0], [0, -1], [-1, 0]];              // (dx, dy) as in the script; next = (r + dy, c + dx)
const key = (r, c) => r * 10000 + c;

// CPython heapq (heappush / heappop with _siftdown / _siftup) on [priority, r, c] tuples.
const lt = (a, b) => a[0] !== b[0] ? a[0] < b[0] : a[1] !== b[1] ? a[1] < b[1] : a[2] < b[2];
function siftdown(h, start, pos) {
  const item = h[pos];
  while (pos > start) { const pp = (pos - 1) >> 1, parent = h[pp]; if (lt(item, parent)) { h[pos] = parent; pos = pp; continue; } break; }
  h[pos] = item;
}
function siftup(h, pos) {
  const end = h.length, start = pos, item = h[pos];
  let child = 2 * pos + 1;
  while (child < end) { const right = child + 1; if (right < end && !lt(h[child], h[right])) child = right; h[pos] = h[child]; pos = child; child = 2 * pos + 1; }
  h[pos] = item; siftdown(h, start, pos);
}
export function heappush(h, item) { h.push(item); siftdown(h, 0, h.length - 1); }
export function heappop(h) { const last = h.pop(); if (h.length) { const top = h[0]; h[0] = last; siftup(h, 0); return top; } return last; }

export class Maze {
  constructor(width = 20, height = 20, populationSize = 50, R) {
    Object.assign(this, { width, height, populationSize, R });
    this.start = [1, 1]; this.end = [height - 2, width - 2];
  }
  open(r, c) { return r >= 0 && r < this.height && c >= 0 && c < this.width && this.maze[r][c] === 0; }
  generate() {
    const { width: W, height: H, R } = this;
    this.maze = Array.from({ length: H }, () => new Array(W).fill(1));
    const enter = (x, y) => { this.maze[y][x] = 0; const d = [[0, 2], [2, 0], [0, -2], [-2, 0]]; R.shuffle(d); return { x, y, d, i: 0 }; };
    const stack = [enter(1, 1)];                                  // iterative carve_path, same visiting order
    while (stack.length) {
      const f = stack[stack.length - 1];
      if (f.i >= 4) { stack.pop(); continue; }
      const [dx, dy] = f.d[f.i++], nx = f.x + dx, ny = f.y + dy;
      if (nx >= 0 && nx < W && ny >= 0 && ny < H && this.maze[ny][nx] === 1) {
        this.maze[f.y + Math.floor(dy / 2)][f.x + Math.floor(dx / 2)] = 0; stack.push(enter(nx, ny));
      }
    }
    this.maze[this.start[0]][this.start[1]] = 0; this.maze[this.end[0]][this.end[1]] = 0;
    for (let k = 0; k < Math.floor(W / 2); k++) {
      const x = R.randrange(1, W - 1), y = R.randrange(1, H - 1);
      if (this.maze[y][x] === 1) this.maze[y][x] = 0;
    }
    return this.maze;
  }
  // steps: [{ cur: [r, c], seen: number of cells discovered so far }], order: cells in discovery order.
  astar() {
    const [er, ec] = this.end, frontier = [[0, ...this.start]], came = new Map([[key(...this.start), null]]), cost = new Map([[key(...this.start), 0]]);
    const order = [this.start], steps = [];
    while (frontier.length) {
      const [, r, c] = heappop(frontier);
      steps.push({ cur: [r, c], seen: came.size });
      if (r === er && c === ec) break;
      for (const [dx, dy] of DIRS) {
        const nr = r + dy, nc = c + dx;
        if (!this.open(nr, nc)) continue;
        const nk = key(nr, nc), nCost = cost.get(key(r, c)) + 1;
        if (!cost.has(nk) || nCost < cost.get(nk)) {
          cost.set(nk, nCost); heappush(frontier, [nCost + Math.abs(er - nr) + Math.abs(ec - nc), nr, nc]);
          if (!came.has(nk)) order.push([nr, nc]);
          came.set(nk, [r, c]);
        }
      }
    }
    let path = [];
    if (came.has(key(er, ec))) for (let cur = [er, ec]; cur; cur = came.get(key(...cur))) path.push(cur);
    return { path: path.reverse(), steps, order };
  }
  wave() {
    const [er, ec] = this.end, queue = [[this.start, [this.start]]], visited = new Set([key(...this.start)]), order = [this.start], steps = [];
    for (let qi = 0; qi < queue.length; qi++) {
      const [cur, path] = queue[qi];
      steps.push({ cur, seen: visited.size });
      if (cur[0] === er && cur[1] === ec) return { path, steps, order };
      for (const [dx, dy] of DIRS) {
        const nr = cur[0] + dy, nc = cur[1] + dx;
        if (this.open(nr, nc) && !visited.has(key(nr, nc))) { queue.push([[nr, nc], [...path, [nr, nc]]]); visited.add(key(nr, nc)); order.push([nr, nc]); }
      }
    }
    return { path: [], steps, order };
  }
  // --- genetic_pathfind ---
  randomPath(origin = this.start) {
    const [er, ec] = this.end, path = [origin], visited = new Set([key(...origin)]), maxSteps = this.width * this.height;
    let cur = origin;
    while (path.length < maxSteps && !(cur[0] === er && cur[1] === ec)) {
      const moves = DIRS.map(([dx, dy]) => [cur[0] + dy, cur[1] + dx]).filter(([r, c]) => this.open(r, c));
      if (!moves.length) break;
      const unvisited = moves.filter(([r, c]) => !visited.has(key(r, c)));
      if (unvisited.length) {
        const d = ([r, c]) => Math.abs(r - er) + Math.abs(c - ec);
        cur = unvisited.reduce((a, b) => (d(b) < d(a) ? b : a));
      } else cur = this.R.choice(moves);
      path.push(cur); visited.add(key(...cur));
    }
    return path;
  }
  fitness(path) {
    if (!path.length) return -Infinity;
    const [r, c] = path[path.length - 1], [er, ec] = this.end;
    const reached = r === er && c === ec ? 1000 : 0, unique = new Set(path.map(p => key(...p))).size;
    return reached - (Math.abs(r - er) + Math.abs(c - ec)) - path.length * 0.5 - (path.length - unique) * 2;
  }
  // Crossover point: a random cell common to both paths (excluding their ends). Python draws it from a set, whose
  // iteration order is an implementation detail; here the common cells are listed in path-1 order.
  crossover(p1, p2) {
    if (p1.length < 2 || p2.length < 2) return p1;
    const in2 = new Set(p2.slice(1, -1).map(p => key(...p))), seen = new Set(), common = [];
    for (const p of p1.slice(1, -1)) { const k = key(...p); if (in2.has(k) && !seen.has(k)) { seen.add(k); common.push(p); } }
    if (!common.length) return p1;
    const cp = this.R.choice(common), k = key(...cp);
    const i1 = p1.findIndex(p => key(...p) === k), i2 = p2.findIndex(p => key(...p) === k);
    return [this.start, ...p1.slice(1, i1), ...p2.slice(i2)];
  }
  genetic(maxGenerations = 100) {
    const R = this.R, [er, ec] = this.end, isEnd = p => p && p[p.length - 1][0] === er && p[p.length - 1][1] === ec;
    let pop = Array.from({ length: this.populationSize }, () => this.randomPath()), best = null, bestF = -Infinity;
    const gens = [];
    for (let g = 0; g < maxGenerations; g++) {
      const scored = pop.map(p => ({ p, f: this.fitness(p) })).sort((a, b) => (b.f > a.f) - (b.f < a.f));
      if (scored[0].f > bestF) { bestF = scored[0].f; best = scored[0].p; }
      gens.push({ best: scored[0].p, bestF, explored: scored.slice(0, 5).map(x => x.p) });
      if (isEnd(best)) break;
      const next = scored.slice(0, 2).map(x => x.p), top = scored.slice(0, 20);
      const pick = () => { const [a, b] = R.sample(top, 2); return (b.f > a.f ? b : a).p; };
      while (next.length < this.populationSize) {
        let child = this.crossover(pick(), pick());
        if (R.random() < 0.1) {
          const mp = R.randint(1, child.length - 1), tail = this.randomPath(child[mp - 1]);
          child = [...child.slice(0, mp), ...tail.slice(1)];
        }
        next.push(child);
      }
      pop = next;
    }
    return { path: isEnd(best) ? best : null, gens };
  }
}
