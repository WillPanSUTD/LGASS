from pathlib import Path
import yaml
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

REQUIRED_KEYS = {
    "seed", "num_classes", "num_points", "epoch",
    "optimizer", "learning_rate", "batch_size", "weight_decay",
    "lr_decay", "lr_decay_milestones", "lr_decay_step_size", "lr_decay_gamma",
    "k_neighbors", "input_type", "data_root",
}

def _load(name):
    return yaml.safe_load((CONFIGS / name).read_text(encoding="utf-8"))

def test_paper_config_has_required_keys():
    cfg = _load("paper.yaml")
    assert REQUIRED_KEYS.issubset(cfg.keys()), REQUIRED_KEYS - cfg.keys()

def test_paper_config_matches_paper_table_2():
    cfg = _load("paper.yaml")
    assert cfg["optimizer"] == "adamw"
    assert cfg["learning_rate"] == 0.01
    assert cfg["batch_size"] == 8
    assert cfg["epoch"] == 300
    assert cfg["num_points"] == 16384

def test_original_config_matches_actual_original_train_py_behavior():
    cfg = _load("original.yaml")
    # Original train.py: string label was 'Adam' but dispatch built AdamW.
    assert cfg["optimizer"] == "adamw"
    assert cfg["learning_rate"] == 0.001
    assert cfg["batch_size"] == 4
    # Original LR schedule was StepLR-equivalent every 20 epochs by 0.5.
    assert cfg["lr_decay"] == "step"
    assert cfg["lr_decay_step_size"] == 20
    assert cfg["lr_decay_gamma"] == 0.5

def test_both_configs_use_8_classes():
    for name in ("paper.yaml", "original.yaml"):
        assert _load(name)["num_classes"] == 8

def test_train_py_accepts_config_flag():
    out = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True, text=True, cwd=str(REPO),
    )
    assert "--config" in out.stdout, out.stdout + out.stderr
