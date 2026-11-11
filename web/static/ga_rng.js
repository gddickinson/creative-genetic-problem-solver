// Seeded random numbers for every engine: mulberry32 plus small, fully specified versions of the Python `random` /
// `numpy.random` calls the solver scripts use. scripts/fixture_rng.py is the same code in Python, swapped into the
// repo's real solver modules when the fixtures are made, so JS and Python consume identical random numbers.

export class Rng {
  constructor(seed = 1) { this.seed(seed); }
  seed(s) { this.a = (s >>> 0); }
  random() {
    this.a = (this.a + 0x6D2B79F5) >>> 0;
    const a = this.a;
    let t = Math.imul(a ^ (a >>> 15), 1 | a) >>> 0;
    t = ((t + Math.imul(t ^ (t >>> 7), 61 | t)) >>> 0) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  // --- `random` module ---
  uniform(a, b) { return a + (b - a) * this.random(); }
  randint(a, b) { return a + Math.floor(this.random() * (b - a + 1)); }
  randrange(a, b) { if (b === undefined) { b = a; a = 0; } return a + Math.floor(this.random() * (b - a)); }
  choice(seq) { return seq[Math.floor(this.random() * seq.length)]; }
  choices(seq, k) { return Array.from({ length: k }, () => this.choice(seq)); }
  sample(seq, k) {
    const pool = Array.from(seq), n = pool.length;
    if (k > n) throw new Error('Sample larger than population');
    for (let i = 0; i < k; i++) {
      const j = i + Math.floor(this.random() * (n - i));
      [pool[i], pool[j]] = [pool[j], pool[i]];
    }
    return pool.slice(0, k);
  }
  shuffle(x) {
    for (let i = x.length - 1; i > 0; i--) { const j = Math.floor(this.random() * (i + 1)); [x[i], x[j]] = [x[j], x[i]]; }
  }
  gauss(mu = 0, sigma = 1) {
    const u1 = this.random(), u2 = this.random();
    return mu + sigma * Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  }
  // --- `numpy.random` (flat arrays, row-major) ---
  uniformN(lo, hi, n) { return Array.from({ length: n }, () => this.uniform(lo, hi)); }
  randomN(n) { return Array.from({ length: n }, () => this.random()); }
  normalN(loc, scale, n) { return Array.from({ length: n }, () => this.gauss(loc, scale)); }
  // np.random.choice(n, p=p): index i of the first cumulative weight above u * total.
  choiceP(p) {
    let total = 0; const cdf = p.map(v => (total += v));
    const u = this.random() * cdf[cdf.length - 1];
    for (let i = 0; i < cdf.length; i++) if (u < cdf[i]) return i;
    return cdf.length - 1;
  }
  // np.random.choice(n, k, replace=False)
  choiceNoReplace(n, k) { return this.sample(Array.from({ length: n }, (_, i) => i), k); }
}

export const sequence = (seed, n) => { const r = new Rng(seed); return Array.from({ length: n }, () => r.random()); };
