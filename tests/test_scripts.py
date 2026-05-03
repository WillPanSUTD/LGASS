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
