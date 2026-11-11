"""Shared, portable random numbers for the Python-vs-JavaScript fixtures.

The browser engines (web/static/*_engine.js) draw every random number from ``web/static/ga_rng.js``: a mulberry32
generator plus small, fully specified versions of the ``random`` / ``numpy.random`` calls the solver scripts use
(randint, sample, shuffle, gauss, choice with p, ...). This module is the same generator in Python. ``patched(mod, seed)``
swaps it into a loaded solver module (its ``random`` and ``np.random``), so the repo's real Python code and the JS port
consume identical random numbers and their results can be compared number for number.
"""
import contextlib
import math

import numpy as np

MASK = 0xFFFFFFFF


def _imul(a: int, b: int) -> int:
    return (a * b) & MASK


class Mulberry32:
    """Same sequence as ``rng(seed)`` in core.js / ``Mulberry`` in ga_rng.js."""

    def __init__(self, seed: int = 1):
        self.seed(seed)

    def seed(self, seed: int) -> None:
        self.a = int(seed) & MASK

    def next(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & MASK
        a = self.a
        t = _imul(a ^ (a >> 15), 1 | a)
        t = ((t + _imul(t ^ (t >> 7), 61 | t)) & MASK) ^ t
        return ((t ^ (t >> 14)) & MASK) / 4294967296


class PyRandom:
    """Stand-in for the ``random`` module (only the functions the solver scripts call)."""

    def __init__(self, gen: Mulberry32):
        self.g = gen

    def seed(self, s=None):
        self.g.seed(0 if s is None else s)

    def random(self):
        return self.g.next()

    def uniform(self, a, b):
        return a + (b - a) * self.g.next()

    def randint(self, a, b):
        return a + int(math.floor(self.g.next() * (b - a + 1)))

    def randrange(self, a, b=None):
        if b is None:
            a, b = 0, a
        return a + int(math.floor(self.g.next() * (b - a)))

    def choice(self, seq):
        seq = list(seq) if not hasattr(seq, "__getitem__") else seq
        return seq[int(math.floor(self.g.next() * len(seq)))]

    def choices(self, seq, k=1):
        return [self.choice(seq) for _ in range(k)]

    def sample(self, seq, k):
        pool = list(seq)
        n = len(pool)
        if k > n:
            raise ValueError("Sample larger than population")
        for i in range(k):
            j = i + int(math.floor(self.g.next() * (n - i)))
            pool[i], pool[j] = pool[j], pool[i]
        return pool[:k]

    def shuffle(self, x):
        for i in range(len(x) - 1, 0, -1):
            j = int(math.floor(self.g.next() * (i + 1)))
            x[i], x[j] = x[j], x[i]

    def gauss(self, mu=0.0, sigma=1.0):
        u1, u2 = self.g.next(), self.g.next()
        return mu + sigma * math.sqrt(-2.0 * math.log(1.0 - u1)) * math.cos(2.0 * math.pi * u2)


class NpRandom:
    """Stand-in for ``numpy.random`` (row-major fill, one generator draw per element)."""

    def __init__(self, gen: Mulberry32):
        self.py = PyRandom(gen)

    def _fill(self, size, f):
        if size is None:
            return f()
        shape = (size,) if isinstance(size, (int, np.integer)) else tuple(size)
        n = int(np.prod(shape)) if shape else 1
        return np.array([f() for _ in range(n)], dtype=float).reshape(shape)

    def seed(self, s=None):
        self.py.seed(s)

    def random(self, size=None):
        return self._fill(size, self.py.random)

    def uniform(self, low=0.0, high=1.0, size=None):
        return self._fill(size, lambda: self.py.uniform(low, high))

    def normal(self, loc=0.0, scale=1.0, size=None):
        return self._fill(size, lambda: self.py.gauss(loc, scale))

    def randn(self, *shape):
        return self._fill(shape if shape else None, lambda: self.py.gauss(0.0, 1.0))

    def choice(self, a, size=None, replace=True, p=None):
        items = list(range(a)) if isinstance(a, (int, np.integer)) else list(a)
        if p is not None:
            cdf = np.cumsum(p)

            def one():
                u = self.py.random() * cdf[-1]
                for i, c in enumerate(cdf):
                    if u < c:
                        return items[i]
                return items[-1]
            return one() if size is None else np.array([one() for _ in range(size)])
        if not replace:
            return np.array(self.py.sample(items, size if size is not None else 1)) if size is not None \
                else self.py.sample(items, 1)[0]
        return self.py.choice(items) if size is None else np.array([self.py.choice(items) for _ in range(size)])


class _NumpyProxy:
    """``numpy`` with ``.random`` replaced (everything else forwards to the real numpy)."""

    def __init__(self, gen):
        self.random = NpRandom(gen)

    def __getattr__(self, name):
        return getattr(np, name)


@contextlib.contextmanager
def patched(mod, seed: int = 1):
    """Within the block, ``mod.random`` and ``mod.np.random`` draw from one seeded Mulberry32 generator."""
    gen = Mulberry32(seed)
    saved = {k: getattr(mod, k) for k in ("random", "np") if hasattr(mod, k)}
    if "random" in saved:
        mod.random = PyRandom(gen)
    if "np" in saved:
        mod.np = _NumpyProxy(gen)
    try:
        yield gen
    finally:
        for k, v in saved.items():
            setattr(mod, k, v)


def sequence(seed: int, n: int):
    """First n numbers of the generator (checks the JS port of Mulberry32 itself)."""
    g = Mulberry32(seed)
    return [g.next() for _ in range(n)]
