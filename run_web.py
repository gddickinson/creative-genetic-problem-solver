"""Start the Creative Genetic Algorithm Visualizers web app on http://127.0.0.1:8154 (local only).

Run: python run_web.py [--port 8154]
Needs: pip install -r requirements-web.txt
"""
import argparse

import uvicorn

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8154)
    port = ap.parse_args().port
    print(f"Creative GA web app: http://127.0.0.1:{port}")
    uvicorn.run("web.app:app", host="127.0.0.1", port=port)
