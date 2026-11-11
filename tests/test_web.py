"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific -------------------------------------------------------------------------------------------
FIXTURE_SCRIPTS = sorted(p.stem for p in (ROOT / "scripts").glob("make_*_fixture.py"))


@pytest.mark.parametrize("script", FIXTURE_SCRIPTS)
def test_js_fixture_is_current(script):
    """tests/fixtures/<name>.json still matches what the repo's Python solver computes now."""
    import importlib
    import json
    sys.path.insert(0, str(ROOT / "scripts"))
    from fixture_common import FIXTURES, close, to_json
    mod = importlib.import_module(script)
    committed = json.loads((FIXTURES / f"{mod.NAME}.json").read_text())
    assert close(committed, to_json(mod.fixture())) == []


def test_every_engine_has_a_node_test_and_fixture():
    engines = {p.name[:-len("_engine.js")] for p in STATIC.glob("*_engine.js")}
    assert engines == {p.name[:-len("_engine.test.mjs")] for p in STATIC.glob("*_engine.test.mjs")}
    assert engines == {s[len("make_"):-len("_fixture")] for s in FIXTURE_SCRIPTS}


def test_mulberry32_matches_core_js_rng():
    sys.path.insert(0, str(ROOT / "scripts"))
    import fixture_rng
    if not shutil.which("node"):
        pytest.skip("node not installed")
    js = "import('%s').then(m => console.log(JSON.stringify(m.sequence(12345, 6))))" % (STATIC / "ga_rng.js").as_uri()
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout
    import json
    assert json.loads(out) == fixture_rng.sequence(12345, 6)


def test_catalogue_lists_all_ten_solvers_with_desktop_defaults():
    cat = client.get("/api/problems").json()
    assert [c["route"] for c in cat] == ["tsp", "creative", "vrp", "packing", "portfolio", "neural", "game", "maze", "flow", "rd"]
    assert sum(c["is_ga"] for c in cat) == 8
    tsp = cat[0]["defaults"]
    assert tsp["population_size"] == 100 and tsp["generations"] == 100
    for c in cat:
        assert (STATIC / f"{c['route']}.js").exists() and (ROOT / c["script"]).exists()


def test_creative_problems_and_portfolio_market():
    probs = client.get("/api/creative/problems").json()
    assert len(probs) == 12 and probs[0]["name"] == "Solar Panel Optimization" and probs[0]["bounds"] == [[0, 90], [0, 10]]
    m = client.get("/api/portfolio/market/5").json()
    assert len(m["expected_returns"]) == 5 and all(0.05 <= r <= 0.15 for r in m["expected_returns"])
    assert m == client.get("/api/portfolio/market/5").json()          # NumPy seed 42: same data every time
    assert client.get("/api/portfolio/market/40").status_code == 404
