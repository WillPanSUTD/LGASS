import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent


def _run(args):
    return subprocess.run(
        [sys.executable, "visualize.py", *args],
        capture_output=True, text=True, cwd=str(REPO),
    )


def test_visualize_help():
    r = _run(["--help"])
    assert r.returncode == 0
    assert "--input" in r.stdout
    assert "--save" in r.stdout


def test_visualize_loads_npz_and_renders_offscreen(tmp_path):
    # Tiny synthetic .npz: 100 points, all class 0.
    pts = np.zeros((100, 7), dtype=np.float32)
    pts[:, 0:3] = np.random.rand(100, 3)
    pts[:, 6] = 0
    npz_path = tmp_path / "sample.npz"
    np.savez(npz_path, points=pts)
    out_png = tmp_path / "out.png"
    r = _run(["--input", str(npz_path), "--save", str(out_png), "--no-window"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert out_png.is_file()
    assert out_png.stat().st_size > 0
