# Session log

## 2026-11-11 -- web interface and static GitHub Pages site (Claude Code)
Branch `web-interface` (not pushed).

**Bugs fixed** (each in its own commit with a regression test in `tests/test_regressions.py` / `tests/test_wrappers.py`):
- Underscore modules: five were stale hand copies (neural_vis lacked 'squares', game_evolution the final analysis);
  the other five did nothing when run. All ten now wrap the hyphenated originals via `_original_loader.py`.
- flow-network.py: Ford-Fulkerson ignored residual reverse edges, so it could report less than the maximum flow
  (7 of 100 seeded default networks).
- vehicle-routing-optimizer.py: the insertion heuristic "restored" a probed route by dropping its last element,
  losing/duplicating locations; the final summary divided by zero when nothing was delivered.
- vehicle-routing-optimizer.py: new `speed` parameter (default 1.0 = old behaviour). At speed 1 almost no location
  fits its time window (needs George's decision on the default).
- maze-pathfinder.py: GA mutation spliced a new walk that restarted at the maze start (teleporting paths); A*
  returned `[goal]` when the goal was unreachable.
- neural-vis.py: 'squares' drew crossing strokes (edge parameter 0-4) and crashed for counts not divisible by 8;
  'spiral' put both classes on the same arm.
- reaction-diffusion.py: 'random' seeds were fixed at cells 45-55 (blank on small grids).
- packing-optimizer.py: removed ~85 lines of unreachable code (file now < 500 lines).

**Web**: `web/` (FastAPI + ES modules), one page per solver with a JS engine node-tested against the Python using a
shared seeded generator (`ga_rng.js` / `scripts/fixture_rng.py`), fixtures in `tests/fixtures/`. Static site via
`scripts/build_static.py` and `.github/workflows/pages.yml`.

**Open questions for George**: VRP default speed / time units, and that its fitness ignores unserved locations;
game-evolution strategy names vs `Strategy.decide` (Tit-for-Tat and Grudger do not reciprocate or hold grudges;
memory 0 uses the whole history); packing rotation flag has no effect; with the packing defaults about half of the
seeds never find a valid layout.

Nothing ran on the Mac mini.
