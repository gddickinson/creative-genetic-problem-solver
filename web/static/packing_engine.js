// JS port of packing-optimizer.py (PackingItem.overlaps + PackingOptimizer). Items are {x, y, w, h, rot}: a circle
// has h === null and w is its radius; a rectangle's (x, y) is its lower-left corner. Same random calls as the Python.
const MARGIN = 0.2;                               // SAFETY_MARGIN in PackingItem.overlaps
const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const desc = (a, b) => (b.f > a.f) - (b.f < a.f); // stable descending sort that copes with -Infinity

// [width, height] as placed: a rectangle turned by 90 or 270 degrees swaps its sides.
export const footprint = it => (it.h !== null && it.rot % 180 === 90 ? [it.h, it.w] : [it.w, it.h]);

export function overlaps(a, b) {
  if (a.h === null && b.h === null) return Math.hypot(a.x - b.x, a.y - b.y) < a.w + b.w + MARGIN;
  if (a.h !== null && b.h !== null) {
    const m = MARGIN / 2, [aw, ah] = footprint(a), [bw, bh] = footprint(b);
    const xo = !(a.x + aw + m < b.x - m || a.x - m > b.x + bw + m);
    const yo = !(a.y + ah + m < b.y - m || a.y - m > b.y + bh + m);
    return xo && yo;
  }
  const [c, r] = a.h === null ? [a, b] : [b, a], [rw, rh] = footprint(r);
  const cx = Math.max(r.x, Math.min(c.x, r.x + rw)), cy = Math.max(r.y, Math.min(c.y, r.y + rh));
  return Math.hypot(c.x - cx, c.y - cy) < c.w + MARGIN;
}

// PackingItem.overlap_depth: how far two items (with the margin) interpenetrate.
export function overlapDepth(a, b) {
  if (a.h === null && b.h === null) return Math.max(0, a.w + b.w + MARGIN - Math.hypot(a.x - b.x, a.y - b.y));
  if (a.h !== null && b.h !== null) {
    const [aw, ah] = footprint(a), [bw, bh] = footprint(b);
    const xo = Math.min(a.x + aw, b.x + bw) - Math.max(a.x, b.x) + MARGIN, yo = Math.min(a.y + ah, b.y + bh) - Math.max(a.y, b.y) + MARGIN;
    return Math.max(0, Math.min(xo, yo));
  }
  const [c, r] = a.h === null ? [a, b] : [b, a], [rw, rh] = footprint(r);
  const cx = Math.max(r.x, Math.min(c.x, r.x + rw)), cy = Math.max(r.y, Math.min(c.y, r.y + rh));
  return Math.max(0, c.w + MARGIN - Math.hypot(c.x - cx, c.y - cy));
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
  bounds(it) {                                   // [left, right, bottom, top]
    if (it.h === null) return [it.x - it.w, it.x + it.w, it.y - it.w, it.y + it.w];
    const [w, h] = footprint(it); return [it.x, it.x + w, it.y, it.y + h];
  }
  outside(it) { const [l, r, b, t] = this.bounds(it); return l < 0 || r > this.W || b < 0 || t > this.H; }
  outsideDepth(it) { const [l, r, b, t] = this.bounds(it); return Math.max(0, -l) + Math.max(0, r - this.W) + Math.max(0, -b) + Math.max(0, t - this.H); }
  violations(sol) {
    let n = sol.filter(it => this.outside(it)).length;
    for (let i = 0; i < sol.length; i++) for (let j = i + 1; j < sol.length; j++) n += overlaps(sol[i], sol[j]) ? 1 : 0;
    return n;
  }
  // ranking_score: the fitness for valid layouts; invalid ones rank below, by total overlap depth and violations.
  score(sol) {
    const f = this.fitness(sol);
    if (f > -Infinity) return f;
    let depth = sol.reduce((a, it) => a + this.outsideDepth(it), 0);
    for (let i = 0; i < sol.length; i++) for (let j = i + 1; j < sol.length; j++) depth += overlapDepth(sol[i], sol[j]);
    return -1e6 - 1000 * depth - 10 * this.violations(sol);
  }
  fitness(sol) {
    if (sol.some(it => this.outside(it))) return -Infinity;
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
            const [w, h] = footprint({ w: it.w, h: it.h, rot });
            n = { x: clip(x, 0, this.W - w), y: clip(y, 0, this.H - h), w: it.w, h: it.h, rot };
          }
          const f = this.score([...out, n, ...sol.slice(out.length + 1)]);
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
    const scored = this.population.map(s => ({ s, f: this.score(s) })).sort(desc);
    const topF = this.fitness(scored[0].s);                    // true fitness (-Infinity if invalid)
    if (topF > this.bestFitness) { this.bestFitness = topF; this.best = scored[0].s; }
    this.history.push(this.bestFitness); this.current = scored[0].s;
    this.valid = scored.filter(x => x.f > -1e6).length;
    const next = scored.slice(0, 2).map(x => x.s), top = scored.slice(0, 10);
    while (next.length < this.populationSize) {
      const [a, b] = this.R.sample(top, 2);
      next.push(this.mutate(this.crossover(a.s, b.s)));
    }
    this.population = next; this.generation++;
  }
}
