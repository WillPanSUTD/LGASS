from pathlib import Path
import os
import stat

REPO = Path(__file__).resolve().parent.parent


def test_reproduce_paper_sh_exists_and_is_executable():
    p = REPO / "scripts" / "reproduce_paper.sh"
    assert p.is_file()
    if os.name == "posix":
        assert p.stat().st_mode & stat.S_IXUSR, "script not chmod +x"


def test_reproduce_paper_sh_invokes_train_with_paper_config():
    text = (REPO / "scripts" / "reproduce_paper.sh").read_text(encoding="utf-8")
    assert "train.py" in text
    assert "configs/paper.yaml" in text


import subprocess
import sys


def test_upload_to_hf_help():
    r = subprocess.run(
        [sys.executable, "scripts/upload_to_hf.py", "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0
    for flag in ["--checkpoint", "--repo", "--dry-run"]:
        assert flag in r.stdout


def test_upload_to_hf_dry_run_does_not_upload(tmp_path):
    fake_ckpt = tmp_path / "fake.pth"
    fake_ckpt.write_bytes(b"\x00" * 16)
    r = subprocess.run(
        [sys.executable, "scripts/upload_to_hf.py",
         "--checkpoint", str(fake_ckpt),
         "--repo", "vpan1226/LGASS",
         "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert r.returncode == 0, r.stdout + r.stderr
    assert "DRY RUN" in r.stdout
