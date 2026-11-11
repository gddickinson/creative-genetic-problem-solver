// JS port of packing-optimizer.py (PackingItem.overlaps + PackingOptimizer). Items are {x, y, w, h, rot}: a circle
// has h === null and w is its radius; a rectangle's (x, y) is its lower-left corner. Same random calls as the Python.
const MARGIN = 0.2;                               // SAFETY_MARGIN in PackingItem.overlaps
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const desc = (a, b) => (b.f > a.f) - (b.f < a.f); // stable descending sort that copes with -Infinity

export function overlaps(a, b) {
  if (a.h === null && b.h === null) return Math.hypot(a.x - b.x, a.y - b.y) < a.w + b.w + MARGIN;
  if (a.h !== null && b.h !== null) {
    const m = MARGIN / 2;
    const xo = !(a.x + a.w + m < b.x - m || a.x - m > b.x + b.w + m);
    const yo = !(a.y + a.h + m < b.y - m || a.y - m > b.y + b.h + m);
    return xo && yo;
  }
  const [c, r] = a.h === null ? [a, b] : [b, a];
  const cx = Math.max(r.x, Math.min(c.x, r.x + r.w)), cy = Math.max(r.y, Math.min(c.y, r.y + r.h));
  return Math.hypot(c.x - cx, c.y - cy) < c.w + MARGIN;
}

export class Packing {
  constructor({ width = 10, height = 10, populationSize = 50, mutationRate = 0.1 } = {}, R) {
    Object.assign(this, { W: width, H: height, populationSize, mutationRate, R });
    this.items = []; this.best = null; this.bestFitness = -Infinity; this.history = []; this.generation = 0;
  }
  generateItems(type = 'circles', n = 10) {
    const lo = Math.min(this.W, this.H) / 15, hi = Math.min(this.W, this.H) / 6;
    this.type = type;
    this.items = Array.from({ length: n }, () => (type === 'circles'
      ? { x: 0, y: 0, w: this.R.uniform(lo, hi), h: null, rot: 0 }
      : { x: 0, y: 0, w: this.R.uniform(lo, hi), h: this.R.uniform(lo, hi), rot: 0 }));
  }
  randomSolution() {
    const sol = [], R = this.R;
    const sorted = this.items.slice().sort((a, b) => b.w * (b.h ?? b.w) - a.w * (a.h ?? a.w));
    for (const it of sorted) {
      let placed = false, best = null, minCount = Infinity;
      for (let k = 0; k < 200; k++) {
        const n = it.h === null
          ? { x: R.uniform(it.w + 0.5, this.W - it.w - 0.5), y: R.uniform(it.w + 0.5, this.H - it.w - 0.5), w: it.w, h: null, rot: 0 }
          : { x: R.uniform(0.5, this.W - it.w - 0.5), y: R.uniform(0.5, this.H - it.h - 0.5), w: it.w, h: it.h, rot: 0 };
        const count = sol.filter(e => overlaps(n, e)).length;
        if (count === 0) { sol.push(n); placed = true; break; }
        if (count < minCount) { minCount = count; best = n; }
      }
      if (!placed && best) sol.push(best);
    }
    return sol;
  }
  // -Infinity if any item leaves the container or two items overlap; else minus the summed distance to the centre.
  fitness(sol) {
    for (const it of sol) {
      if (it.h === null ? (it.x - it.w < 0 || it.x + it.w > this.W || it.y - it.w < 0 || it.y + it.w > this.H)
        : (it.x < 0 || it.x + it.w > this.W || it.y < 0 || it.y + it.h > this.H)) return -Infinity;
    }
    for (let i = 0; i < sol.length; i++) for (let j = i + 1; j < sol.length; j++) if (overlaps(sol[i], sol[j])) return -Infinity;
    const cx = this.W / 2, cy = this.H / 2;
    return -sol.reduce((s, it) => s + Math.sqrt((it.x - cx) ** 2 + (it.y - cy) ** 2), 0);
  }
  crossover(p1, p2) { return this.items.map((_, i) => ({ ...(this.R.random() < 0.5 ? p1[i] : p2[i]) })); }
  mutate(sol) {
    const R = this.R, scale = this.W / Math.sqrt(sol.length) * 0.2, out = [];
    for (const it of sol) {
      if (R.random() < this.mutationRate) {
        let bestItem = null, bestF = -Infinity;
        for (let k = 0; k < 5; k++) {
          let n;
          if (it.h === null) {
            const x = it.x + R.gauss(0, scale), y = it.y + R.gauss(0, scale);
            n = { x: clip(x, it.w, this.W - it.w), y: clip(y, it.w, this.H - it.w), w: it.w, h: null, rot: 0 };
          } else {
            const x = it.x + R.gauss(0, scale), y = it.y + R.gauss(0, scale);
            let rot = it.rot; if (R.random() < 0.1) rot = (rot + 90) % 360;
            n = { x: clip(x, 0, this.W - it.w), y: clip(y, 0, this.H - it.h), w: it.w, h: it.h, rot };
          }
          const f = this.fitness([...out, n, ...sol.slice(out.length + 1)]);
          if (f > bestF) { bestItem = n; bestF = f; }
        }
        out.push(bestItem || it);
      } else out.push(it);
    }
    return out;
  }
  // One generation of evolve().
  step() {
    if (!this.population) this.population = Array.from({ length: this.populationSize }, () => this.randomSolution());
    const scored = this.population.map(s => ({ s, f: this.fitness(s) })).sort(desc);
    if (scored[0].f > this.bestFitness) { this.bestFitness = scored[0].f; this.best = scored[0].s; }
    this.history.push(this.bestFitness); this.current = scored[0].s;
    this.valid = scored.filter(x => x.f > -Infinity).length;
    const next = scored.slice(0, 2).map(x => x.s), top = scored.slice(0, 10);
    while (next.length < this.populationSize) {
      const [a, b] = this.R.sample(top, 2);
      next.push(this.mutate(this.crossover(a.s, b.s)));
    }
    this.population = next; this.generation++;
  }
}
