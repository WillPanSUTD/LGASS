import subprocess, sys
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parent.parent


def test_evaluate_help():
    r = subprocess.run(
        [sys.executable, "evaluate.py", "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0
    for flag in ["--checkpoint", "--data_root", "--split", "--output", "--raw-schema"]:
        assert flag in r.stdout, f"missing flag: {flag}"


@pytest.mark.gpu
def test_evaluate_emits_markdown_with_paper_columns(tmp_path):
    pytest.skip("requires real model + dataset; run manually after retrain")
