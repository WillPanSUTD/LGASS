import subprocess
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent


def _make_fake_results(dirpath):
    """Two synthetic samples, each with gt + pred .npy."""
    for i in range(2):
        n = 200
        xyz = np.random.rand(n, 3).astype(np.float32)
        gt = np.random.randint(0, 8, size=n).astype(np.int64)
        pred = gt.copy()
        pred[: n // 4] = (pred[: n // 4] + 1) % 8  # 25% error
        np.savez(dirpath / f"sample_{i:03d}.npz", points=xyz, gt=gt, pred=pred)


def test_export_report_generates_index_html(tmp_path):
    results = tmp_path / "results"; results.mkdir()
    out = tmp_path / "reports"
    _make_fake_results(results)
    r = subprocess.run(
        [sys.executable, "export_report.py",
         "--input", str(results), "--output", str(out)],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "index.html").is_file()
    text = (out / "index.html").read_text(encoding="utf-8")
    assert "sample_000" in text and "sample_001" in text
    assert "OA" in text and "mIoU" in text
