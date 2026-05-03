# LGANet repo + OPT-SND dataset improvements — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a polished, reproducible code repo (renamed LGANet → LGASS) and a proper HuggingFace dataset card for OPT-SND, accompanying the EAAI 2026 paper.

**Architecture:** Two-phase delivery. Phase 1 = all docs, configs, and utility scripts (no GPU needed). Phase 2 = retrain / evaluate / upload-checkpoint scripts that the user runs on a GPU box. Each task is independently testable with TDD where applicable; pure-doc tasks use a structural lint as verification.

**Tech Stack:** Python 3.x, PyTorch, Open3D, PyYAML, `huggingface_hub`, pytest (added). Target OS: Linux for training, Windows/Linux/Mac for utility scripts.

**Working directory:** `F:\research\LGANet` (after `git rebase --abort && git pull` already executed; HEAD = `165cb33`).

**Spec reference:** `docs/superpowers/specs/2026-05-04-lganet-and-opt-snd-improvements-design.md`

---

## Task 0: Add pytest harness

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini`
- Modify: `requirements.txt`

- [ ] **Step 1: Add pytest to requirements**

Append to `requirements.txt`:

```
pytest>=7.0.0
PyYAML>=6.0
```

- [ ] **Step 2: Create pytest.ini**

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra --strict-markers
markers =
    gpu: requires CUDA + pointops compiled
    slow: takes > 5s
```

- [ ] **Step 3: Create tests/__init__.py**

Empty file.

- [ ] **Step 4: Create tests/conftest.py**

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
```

- [ ] **Step 5: Verify**

Run: `pytest --collect-only -q`
Expected: `0 tests collected` (no errors).

- [ ] **Step 6: Commit**

```bash
git add tests/ pytest.ini requirements.txt
git commit -m "test: add pytest harness for plan-driven verification"
```

---

## Task 1: LICENSE file

**Files:**
- Create: `LICENSE`
- Test: `tests/test_repo_metadata.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_repo_metadata.py`:

```python
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def test_license_file_exists():
    assert (REPO / "LICENSE").is_file()

def test_license_is_apache_2():
    text = (REPO / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in text
    assert "Version 2.0" in text
    assert "2026" in text
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_repo_metadata.py -v`
Expected: 3 failures (LICENSE missing).

- [ ] **Step 3: Create LICENSE**

Use the standard Apache 2.0 text. Insert this header comment (the body is the canonical Apache 2.0 license text):

```
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

Copyright 2026 Wei Pan and OPT Machine Vision

[... full Apache 2.0 license body, copy verbatim from
 https://www.apache.org/licenses/LICENSE-2.0.txt ...]
```

(The implementer should fetch the canonical text from `https://www.apache.org/licenses/LICENSE-2.0.txt` and prepend the copyright line.)

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_repo_metadata.py -v`
Expected: 3 passes.

- [ ] **Step 5: Commit**

```bash
git add LICENSE tests/test_repo_metadata.py
git commit -m "chore: add Apache 2.0 LICENSE"
```

---

## Task 2: Configs (paper.yaml + original.yaml)

**Files:**
- Create: `configs/paper.yaml`
- Create: `configs/original.yaml`
- Test: `tests/test_configs.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_configs.py`:

```python
from pathlib import Path
import yaml

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

REQUIRED_KEYS = {
    "seed", "num_classes", "num_points", "epoch",
    "optimizer", "learning_rate", "batch_size", "weight_decay",
    "lr_decay", "lr_decay_milestones", "lr_decay_gamma",
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

def test_original_config_matches_original_train_py():
    cfg = _load("original.yaml")
    assert cfg["optimizer"] == "adam"
    assert cfg["learning_rate"] == 0.001
    assert cfg["batch_size"] == 4

def test_both_configs_use_8_classes():
    for name in ("paper.yaml", "original.yaml"):
        assert _load(name)["num_classes"] == 8
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_configs.py -v`
Expected: 4 failures (configs missing).

- [ ] **Step 3: Create configs/paper.yaml**

```yaml
# Paper-Table-2 hyperparameter set (EAAI 2026).
# train.py uses this as default. Switch to configs/original.yaml to match
# the values originally hard-coded in train.py.
seed: 42
num_classes: 8                     # raw schema — 8 classes on disk
num_points: 16384
epoch: 300
optimizer: adamw                   # paper Table 2
learning_rate: 0.01                # paper Table 2
batch_size: 8                      # paper Table 2
weight_decay: 0.0001               # not in Table 2 — kept from original train.py
lr_decay: multistep
lr_decay_milestones: [120, 240]    # PLACEHOLDER — not in paper, edit if known
lr_decay_gamma: 0.1                # PLACEHOLDER — not in paper, edit if known
k_neighbors: 16                    # PLACEHOLDER — not in paper, edit if known
input_type: normals_only           # paper Table 5 best setting
data_root: data/sealingNail_npz
```

- [ ] **Step 4: Create configs/original.yaml**

```yaml
# Hyperparameter set hard-coded in the original train.py before the
# config refactor. Paper Table 2 differs (see configs/paper.yaml).
# Kept for transparency: we cannot confirm which set produced the paper's
# headline numbers without retraining both.
seed: 42
num_classes: 8
num_points: 16384
epoch: 300
optimizer: adam                    # original train.py
learning_rate: 0.001               # original train.py
batch_size: 4                      # original train.py
weight_decay: 0.0001               # original train.py
lr_decay: step
lr_decay_milestones: [20]          # MOMENTUM_DECCAY_STEP
lr_decay_gamma: 0.5                # original train.py
k_neighbors: 16                    # PLACEHOLDER
input_type: normals_only
data_root: data/sealingNail_npz
```

- [ ] **Step 5: Run test, expect PASS**

Run: `pytest tests/test_configs.py -v`
Expected: 4 passes.

- [ ] **Step 6: Commit**

```bash
git add configs/ tests/test_configs.py
git commit -m "feat: add paper.yaml and original.yaml hyperparameter configs"
```

---

## Task 3: train.py — config loader + seed handling

**Files:**
- Modify: `train.py:1-100` (and downstream usages of hardcoded values)
- Create: `util/seeding.py`
- Test: `tests/test_seeding.py`

- [ ] **Step 1: Write failing test for seeding utility**

Create `tests/test_seeding.py`:

```python
import numpy as np
import random

def test_set_seed_makes_numpy_deterministic():
    from util.seeding import set_seed
    set_seed(42)
    a = np.random.rand(10)
    set_seed(42)
    b = np.random.rand(10)
    assert (a == b).all()

def test_set_seed_makes_random_deterministic():
    from util.seeding import set_seed
    set_seed(42)
    a = [random.random() for _ in range(5)]
    set_seed(42)
    b = [random.random() for _ in range(5)]
    assert a == b
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_seeding.py -v`
Expected: ImportError on `util.seeding`.

- [ ] **Step 3: Create util/seeding.py**

```python
"""Reproducibility helper: seeds Python random, numpy, and torch."""
import os
import random
import numpy as np


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_seeding.py -v`
Expected: 2 passes.

- [ ] **Step 5: Write failing test for train.py config loading**

Append to `tests/test_configs.py`:

```python
import subprocess
import sys

def test_train_py_accepts_config_flag():
    # Sanity: --help should mention --config.
    out = subprocess.run(
        [sys.executable, "train.py", "--help"],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent),
    )
    assert "--config" in out.stdout, out.stdout + out.stderr
```

- [ ] **Step 6: Run test, expect FAIL**

Run: `pytest tests/test_configs.py::test_train_py_accepts_config_flag -v`
Expected: FAIL (train.py has no argparse yet).

- [ ] **Step 7: Refactor train.py to read configs**

Edit `train.py` top section (lines 1–80). Replace the hardcoded `Parameter` block with config-driven values:

```python
import argparse
import os.path
import yaml
import json
import logging
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from util.sealingNails_npz import SealingNailDatasetNPZ
from util.data_util import collate_fn
from util.seeding import set_seed
from model.sem.GraphAttention import graphAttention_seg_repro as Model


def parse_args():
    p = argparse.ArgumentParser(description="Train LGASS on sealing-nail dataset")
    p.add_argument("--config", default="configs/paper.yaml",
                   help="path to YAML hyperparameter config")
    p.add_argument("--output_dir", default=None,
                   help="override logs output dir (default: logs/weight_<timestamp>)")
    return p.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unsupported optimizer: {name}")


def inplace_relu(m):
    if m.__class__.__name__.find("LeakyReLU") != -1:
        m.inplace = True


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logs_dir = args.output_dir or os.path.join("logs", f"weight_{timestamp}")
    os.makedirs(logs_dir, exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(logs_dir, "log_embedding.txt"))
    console_handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] - %(message)s")
    file_handler.setFormatter(fmt); console_handler.setFormatter(fmt)
    logger.addHandler(file_handler); logger.addHandler(console_handler)
    logger.info("Resolved config: %s", json.dumps(cfg, indent=2))

    TRAIN_SET = SealingNailDatasetNPZ(
        root=cfg["data_root"], npoints=cfg["num_points"], split="train", use_cache=True,
    )
    num_workers = min(4, os.cpu_count() or 1)
    trainDataLoader = DataLoader(
        TRAIN_SET, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True,
    )
    TEST_SET = SealingNailDatasetNPZ(
        root=cfg["data_root"], npoints=cfg["num_points"], split="test", use_cache=True,
    )
    testDataLoader = DataLoader(
        TEST_SET, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=num_workers, drop_last=True, collate_fn=collate_fn, pin_memory=True,
    )

    weight = torch.tensor(TRAIN_SET.l_weight, dtype=torch.float).to(device)
    classifier = Model(c=6, k=cfg["num_classes"]).to(device)
    classifier.apply(inplace_relu)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
    optimizer = build_optimizer(
        cfg["optimizer"], classifier.parameters(),
        cfg["learning_rate"], cfg["weight_decay"],
    )

    # ... existing training loop body, but replace literal hyperparams with cfg[...] ...
```

**Note to implementer:** keep the existing training loop body (epochs, validation, checkpoint saving) intact below this rewritten setup; only the parameter section + main entry point changes. Read the existing `train.py` from line 80 onward and adapt the loop to use `cfg` and `optimizer` introduced above. Preserve any LR-scheduler logic, replacing its literal step/gamma with `cfg["lr_decay_milestones"]` and `cfg["lr_decay_gamma"]`.

- [ ] **Step 8: Run config test, expect PASS**

Run: `pytest tests/test_configs.py::test_train_py_accepts_config_flag -v`
Expected: PASS.

- [ ] **Step 9: Quick smoke import**

Run: `python -c "import train; print('train.py imports OK')"`
Expected: prints "train.py imports OK" (without invoking main).

If `train.py` lacks an `if __name__ == "__main__":` guard around `main()`, add one.

- [ ] **Step 10: Commit**

```bash
git add train.py util/seeding.py tests/test_seeding.py tests/test_configs.py
git commit -m "feat: drive train.py from yaml config + deterministic seeding"
```

---

## Task 4: visualize.py

**Files:**
- Create: `visualize.py`
- Test: `tests/test_visualize.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/test_visualize.py`:

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_visualize.py -v`
Expected: FAIL (visualize.py missing).

- [ ] **Step 3: Create visualize.py**

```python
"""Visualize a sealing-nail point cloud (PLY or NPZ) colored by semantic label.

Usage:
    python visualize.py --input sample.npz [--predictions pred.npy]
                        [--save out.png] [--no-window]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# 8-class palette (RGB 0..1) — keep stable across runs.
PALETTE = np.array([
    [0.65, 0.65, 0.65],   # 0 Background1 — light gray
    [0.85, 0.10, 0.10],   # 1 Burst       — red
    [0.10, 0.45, 0.85],   # 2 Pit         — blue
    [0.85, 0.85, 0.10],   # 3 Stain       — yellow
    [0.85, 0.45, 0.10],   # 4 Warpage     — orange
    [0.50, 0.50, 0.50],   # 5 Background2 — gray
    [0.65, 0.10, 0.45],   # 6 Burst2      — magenta
    [0.10, 0.75, 0.30],   # 7 Pinhole     — green
], dtype=np.float32)


def load_npz(path):
    data = np.load(path, allow_pickle=True)
    arr = data["points"]
    return arr[:, 0:3].astype(np.float32), arr[:, 6].astype(np.int64)


def load_ply(path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    if pcd.has_colors():
        # encode label-as-color is non-standard; fall back to all-zero labels
        labels = np.zeros(len(xyz), dtype=np.int64)
    else:
        labels = np.zeros(len(xyz), dtype=np.int64)
    return xyz, labels


def colorize(labels):
    labels = np.clip(labels, 0, len(PALETTE) - 1)
    return PALETTE[labels]


def render(xyz, colors, save_path=None, show_window=True):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))

    if save_path is not None:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=show_window, width=960, height=720)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image(str(save_path), do_render=True)
        vis.destroy_window()
    elif show_window:
        o3d.visualization.draw_geometries([pcd])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="path to .ply or .npz file")
    p.add_argument("--predictions", default=None,
                   help="optional .npy of per-point predicted labels (overrides .npz labels)")
    p.add_argument("--save", default=None, help="save offscreen render to PNG")
    p.add_argument("--no-window", action="store_true",
                   help="do not open an interactive window (use with --save)")
    args = p.parse_args()

    inp = Path(args.input)
    if inp.suffix == ".npz":
        xyz, labels = load_npz(inp)
    elif inp.suffix in (".ply", ".pcd"):
        xyz, labels = load_ply(inp)
    else:
        sys.exit(f"unsupported input format: {inp.suffix}")

    if args.predictions:
        labels = np.load(args.predictions).astype(np.int64)
        if labels.shape[0] != xyz.shape[0]:
            sys.exit(f"prediction length {labels.shape[0]} != points {xyz.shape[0]}")

    colors = colorize(labels)
    render(xyz, colors, save_path=args.save, show_window=not args.no_window)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_visualize.py -v`
Expected: 2 passes.

If `test_visualize_loads_npz_and_renders_offscreen` fails on a headless CI box, mark it `@pytest.mark.gpu` or skip when `os.environ.get("DISPLAY")` is missing — but on the user's local Windows machine with Open3D installed it should pass.

- [ ] **Step 5: Commit**

```bash
git add visualize.py tests/test_visualize.py
git commit -m "feat: add visualize.py for point cloud + label rendering"
```

---

## Task 5: export_report.py

**Files:**
- Create: `export_report.py`
- Test: `tests/test_export_report.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_export_report.py`:

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_export_report.py -v`
Expected: FAIL.

- [ ] **Step 3: Create export_report.py**

```python
"""Walk a results directory of .npz prediction files and emit an HTML report.

Each input file must contain at least:
    points: (N, 3+) float32
    gt:     (N,)    int   ground-truth labels (optional)
    pred:   (N,)    int   predicted labels

Usage:
    python export_report.py --input results/ --output reports/
"""
import argparse
import html
from pathlib import Path

import numpy as np

NUM_CLASSES = 8


def per_sample_metrics(gt, pred, num_classes=NUM_CLASSES):
    oa = float((gt == pred).mean()) if len(gt) else 0.0
    ious = []
    for c in range(num_classes):
        gm = gt == c; pm = pred == c
        union = (gm | pm).sum()
        ious.append(float((gm & pm).sum() / union) if union else float("nan"))
    valid = [x for x in ious if x == x]
    miou = float(np.mean(valid)) if valid else 0.0
    return oa, miou, ious


def render_index(rows, num_classes):
    th_ious = "".join(f"<th>IoU{c}</th>" for c in range(num_classes))
    body_rows = []
    for r in rows:
        tds = "".join(f"<td>{x:.4f}</td>" if isinstance(x, float) and x == x else "<td>—</td>"
                      for x in r["ious"])
        body_rows.append(
            f"<tr><td><a href='{html.escape(r['detail_href'])}'>"
            f"{html.escape(r['name'])}</a></td>"
            f"<td>{r['n']}</td><td>{r['oa']:.4f}</td><td>{r['miou']:.4f}</td>{tds}</tr>"
        )
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>LGASS results</title>
<style>body{{font-family:sans-serif;margin:2em}}
table{{border-collapse:collapse}} th,td{{border:1px solid #ccc;padding:4px 8px}}
</style></head><body>
<h1>LGASS prediction report</h1>
<p>{len(rows)} samples</p>
<table><thead><tr><th>Sample</th><th>Points</th><th>OA</th><th>mIoU</th>{th_ious}</tr></thead>
<tbody>{''.join(body_rows)}</tbody></table>
</body></html>"""


def render_detail(name, n, oa, miou, ious):
    lis = "".join(f"<li>class {c}: {v:.4f}</li>" if v == v else f"<li>class {c}: —</li>"
                  for c, v in enumerate(ious))
    return f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{html.escape(name)}</title></head><body>
<h1>{html.escape(name)}</h1>
<ul><li>points: {n}</li><li>OA: {oa:.4f}</li><li>mIoU: {miou:.4f}</li></ul>
<h2>Per-class IoU</h2><ul>{lis}</ul>
<p><a href='index.html'>← back</a></p>
</body></html>"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="results directory with .npz files")
    p.add_argument("--output", required=True, help="output directory for HTML report")
    p.add_argument("--num-classes", type=int, default=NUM_CLASSES)
    args = p.parse_args()

    in_dir = Path(args.input); out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for npz_path in sorted(in_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        keys = data.files
        n = int(data["points"].shape[0]) if "points" in keys else 0
        gt = data["gt"] if "gt" in keys else None
        pred = data["pred"] if "pred" in keys else None
        if gt is None or pred is None:
            continue
        oa, miou, ious = per_sample_metrics(np.asarray(gt), np.asarray(pred), args.num_classes)
        detail_name = npz_path.stem + ".html"
        (out_dir / detail_name).write_text(
            render_detail(npz_path.stem, n, oa, miou, ious), encoding="utf-8"
        )
        rows.append({"name": npz_path.stem, "n": n, "oa": oa,
                     "miou": miou, "ious": ious, "detail_href": detail_name})

    (out_dir / "index.html").write_text(
        render_index(rows, args.num_classes), encoding="utf-8"
    )
    print(f"wrote {len(rows)} sample reports to {out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_export_report.py -v`
Expected: 1 pass.

- [ ] **Step 5: Commit**

```bash
git add export_report.py tests/test_export_report.py
git commit -m "feat: add export_report.py for batch-prediction HTML reports"
```

---

## Task 6: README.md (EN) rewrite

**Files:**
- Modify: `README.md` (full rewrite)
- Test: `tests/test_readme.py`

- [ ] **Step 1: Write failing structural test**

Create `tests/test_readme.py`:

```python
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
README = (REPO / "README.md").read_text(encoding="utf-8")

REQUIRED_HEADINGS = [
    "LGASS",                                      # title
    "Defect Classes",
    "Data Acquisition",
    "Architecture",
    "Results",
    "Ablation",
    "Hyperparameters",
    "Dataset Structure",
    "Installation",
    "Usage",
    "Citation",
    "License",
    "Acknowledgements",
]

def test_required_headings_present():
    for h in REQUIRED_HEADINGS:
        assert h in README, f"missing section: {h}"

def test_results_table_contains_lgass_row_with_correct_numbers():
    # Paper Table 3, ours row.
    for token in ["99.47", "92.37", "79.23", "76.22", "72.17", "71.33", "91.61", "64.95"]:
        assert token in README, f"missing LGASS metric: {token}"

def test_no_phantom_script_references():
    # The two scripts now exist — no aspirational mentions remain.
    # (We verify they exist; the README may reference them.)
    assert (REPO / "visualize.py").is_file()
    assert (REPO / "export_report.py").is_file()

def test_dataset_path_is_npz():
    assert "sealingNail_npz" in README
    assert "sealingNail_normal" not in README

def test_links_to_huggingface_dataset():
    assert "huggingface.co/datasets/vpan1226/OPT-SND" in README
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_readme.py -v`
Expected: 5 failures (current README still says LGANet, sealingNail_normal, etc.).

- [ ] **Step 3: Rewrite README.md**

Read the existing README.md first to preserve any content the tests don't enforce (graphical-abstract image embed, data-acquisition prose, acknowledgements paragraph). Then rewrite to this skeleton:

```markdown
# LGASS — An improved Graph Attention Network for Semantic Segmentation of Industrial Point Clouds in Automotive Battery Sealing Nail Defect Detection

<p align="center">
  <img src="figures/graphical_abstract.png" alt="Graphical abstract of LGASS" width="85%">
</p>

> **Note:** the codebase was renamed from LGANet → LGASS to match the published method name.
> The paper title remains as above for searchability. The old GitHub URL still redirects.

## News

- **EAAI 2026** — paper accepted (volume 163, pages 112793). [DOI link, fill if available]
- Dataset released on HuggingFace as gated repo: <https://huggingface.co/datasets/vpan1226/OPT-SND>.

## Defect Classes

The dataset stores **8 raw classes** on disk; the paper reports a merged **6-class** evaluation
(`Background1 ∪ Background2 → Normal`, `Burst ∪ Burst2 → Burst`).

| Reported (paper, 6 cls)  | Raw (on disk, 8 cls)              |
|--------------------------|-----------------------------------|
| Normal                   | Background1 (0), Background2 (5)  |
| Burst                    | Burst (1), Burst2 (6)             |
| Pit                      | Pit (2)                           |
| Stain                    | Stain (3)                         |
| Warpage                  | Warpage (4)                       |
| Pinhole                  | Pinhole (7)                       |

## Data Acquisition

[Keep existing paragraph from current README, unchanged.]

## Architecture

LGASS is built on the PyTorch deep-learning framework with two novel modules on top of a graph-attention encoder–decoder:

- **LGAF (Local Graph Attention Filter)** — exploits normal-difference attention to detect minute surface irregularities.
- **SAG-Pooling (Spatial Attention Graph Pooling)** — preserves geometry-critical points during downsampling.

See `model/sem/GraphAttention.py` and `model/sem/network.py`.

## Results

Comparison on the sealing-nail dataset (paper Table 3, **bold = best**, _underline_ = 2nd best):

| Method     | OA       | mAcc     | mIoU     | Burst    | Pit      | Stain    | Warpage  | Pinhole  |
|------------|----------|----------|----------|----------|----------|----------|----------|----------|
| PointNet   | 83.29    | 72.83    | 46.21    | 32.27    | 34.96    | 19.97    | 65.37    | 36.04    |
| PointNet++ | 89.33    | 84.65    | 62.67    | 47.91    | 59.32    | 42.37    | 81.64    | 46.53    |
| GACNet     | 96.05    | 87.84    | 69.41    | 56.02    | 67.83    | 54.54    | 86.96    | 59.94    |
| PTv1       | 98.77    | 90.96    | 72.83    | 68.54    | 62.83    | 58.37    | _89.67_  | 58.37    |
| OA-CNN     | 98.72    | 91.13    | 74.04    | 64.89    | 66.91    | 64.87    | 88.37    | _60.04_  |
| PTv3       | _99.03_  | **94.96**| _75.13_  | **77.32**| 67.98    | 60.44    | 89.17    | 56.21    |
| **LGASS**  | **99.47**| _92.37_  | **79.23**| _76.22_  | **72.17**| **71.33**| **91.61**| **64.95**|

**Note on reproduction:** numbers above were produced with a checkpoint that we no longer
have on hand. Re-running the published configuration may yield results within ±0.5% of these
due to non-determinism in `pointops` kernels and CUDA/PyTorch version drift. See `configs/`
and `scripts/reproduce_paper.sh` (Phase 2).

## Ablation

**Table 4 — module ablation (mIoU):**

| ID  | LGAF | SAG-Pooling | OA    | mIoU  |
|-----|------|-------------|-------|-------|
| I   | ✗    | ✗           | 96.92 | 72.93 |
| II  | ✓    | ✗           | 97.38 | 77.17 |
| III | ✗    | ✓           | 97.96 | 74.81 |
| IV  | ✓    | ✓           | 99.47 | 79.23 |

**Table 5 — input-type ablation:**

| Input                   | OA    | mAcc  | mIoU  |
|-------------------------|-------|-------|-------|
| Coordinates only        | 99.42 | 90.92 | 76.94 |
| Coordinates + Normals   | 99.51 | 91.43 | 78.31 |
| Normals only (default)  | 99.47 | 92.37 | 79.23 |

## Hyperparameters

`configs/paper.yaml` is the default and follows paper Table 2:

| Parameter        | Value         |
|------------------|---------------|
| Epoch            | 300           |
| Batch size       | 8             |
| Number of points | 16384         |
| Optimizer        | AdamW         |
| Learning rate    | 0.01          |
| LR decay         | Multi-step    |

`configs/original.yaml` preserves the values originally hard-coded in `train.py` (Adam, lr=0.001, batch=4) for transparency. We cannot confirm which set produced the paper's headline numbers without retraining both.

## Dataset Structure

After downloading from HuggingFace (gated; see <https://huggingface.co/datasets/vpan1226/OPT-SND>):

```
data/sealingNail_npz/
├── train/   *.npz
└── test/    *.npz
```

Class distribution (paper Table 1, merged 6-class view):

| Class   | Count |
|---------|-------|
| Normal  | 271   |
| Burst   | 170   |
| Pit     | 112   |
| Stain   | 102   |
| Warpage | 132   |
| Pinhole | 142   |

Each `.npz` contains a single key `points`, shape `(N, 7)`, columns `[x, y, z, nx, ny, nz, label]`.

## Installation

[Keep existing conda steps from current README, unchanged.]

## Usage

**Train:**

```bash
python train.py --config configs/paper.yaml
# or use the original train.py values:
python train.py --config configs/original.yaml
```

**Inference (single sample):**

```bash
python demo.py --input data/test_sample.ply --model checkpoints/best_model.pth --output results/
```

**Batch inference:**

```bash
python batch_demo.py --input data/test_folder/ --model checkpoints/best_model.pth --output results/
```

**Visualize a sample:**

```bash
python visualize.py --input data/sealingNail_npz/test/sample_001.npz --save out.png --no-window
```

**Generate a per-batch HTML report:**

```bash
python export_report.py --input results/ --output reports/
```

**Phase 2 (after retraining):**

```bash
python evaluate.py --checkpoint logs/paper_run/best.pth --split test --output results/eval.md
bash scripts/reproduce_paper.sh
python scripts/upload_to_hf.py --checkpoint logs/paper_run/best.pth --repo vpan1226/LGASS
```

## Citation

[Keep existing bibtex block, unchanged.]

## License

Apache 2.0. See `LICENSE`.

## Acknowledgements

[Keep existing acknowledgements section, unchanged.]
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_readme.py -v`
Expected: 5 passes.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_readme.py
git commit -m "docs: rewrite README with LGASS naming, results, ablation, configs"
```

---

## Task 7: README_CN.md rewrite (parity)

**Files:**
- Modify: `README_CN.md` (full rewrite)
- Test: `tests/test_readme.py` (extend)

- [ ] **Step 1: Extend test**

Append to `tests/test_readme.py`:

```python
README_CN = (REPO / "README_CN.md").read_text(encoding="utf-8")

CN_REQUIRED = ["LGASS", "结果", "消融", "超参", "数据集结构", "安装", "使用", "引用", "许可", "致谢"]

def test_cn_readme_has_full_parity():
    for h in CN_REQUIRED:
        assert h in README_CN, f"CN README missing: {h}"

def test_cn_results_numbers_present():
    for token in ["99.47", "79.23", "92.37"]:
        assert token in README_CN
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_readme.py -v`
Expected: 2 new failures.

- [ ] **Step 3: Rewrite README_CN.md**

Mirror the EN README section-for-section. Translate headings into Chinese; keep Markdown tables verbatim (tables, numbers, code blocks); translate prose. Use the same images, the same external links. Keep existing CN-specific phrasing where the current README_CN.md already has it (lines 1–60).

Required CN headings (in order): 项目简介 / 缺陷类别 / 数据采集 / 网络架构 / 结果 / 消融实验 / 超参数 / 数据集结构 / 环境配置 / 使用方法 / 引用 / 许可 / 致谢.

Use the exact same Results / Ablation / Hyperparameters / Class-distribution tables as the EN README — they're already language-neutral.

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_readme.py -v`
Expected: all passes.

- [ ] **Step 5: Commit**

```bash
git add README_CN.md tests/test_readme.py
git commit -m "docs: bring README_CN.md to parity with EN README"
```

---

## Task 8: OPT-SND HuggingFace dataset card (staged in code repo)

**Files:**
- Create: `docs/hf-dataset-card/README.md` (the file the user uploads to HF later)
- Test: `tests/test_dataset_card.py`

We stage the dataset card in the code repo under `docs/hf-dataset-card/` so it's version-controlled with the spec it serves. The user uploads it to `huggingface.co/datasets/vpan1226/OPT-SND` separately (not in this session).

- [ ] **Step 1: Write failing test**

Create `tests/test_dataset_card.py`:

```python
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parent.parent
CARD = REPO / "docs" / "hf-dataset-card" / "README.md"


def _split_frontmatter(text):
    assert text.startswith("---\n"), "missing YAML frontmatter"
    _, fm, body = text.split("---\n", 2)
    return yaml.safe_load(fm), body


def test_card_exists():
    assert CARD.is_file()


def test_card_has_required_yaml_keys():
    fm, _ = _split_frontmatter(CARD.read_text(encoding="utf-8"))
    for key in ["license", "task_categories", "language",
                "size_categories", "tags", "pretty_name"]:
        assert key in fm, f"missing frontmatter key: {key}"
    assert fm["license"] == "apache-2.0"


def test_card_has_required_sections():
    text = CARD.read_text(encoding="utf-8")
    for section in [
        "Gated access", "Quick start", "Dataset summary",
        "Dataset structure", "Data fields", "Class definitions",
        "Acquisition setup", "Annotation protocol",
        "Considerations", "Citation",
    ]:
        assert section in text, f"missing section: {section}"


def test_card_quick_start_has_load_dataset_snippet():
    text = CARD.read_text(encoding="utf-8")
    assert 'load_dataset("vpan1226/OPT-SND")' in text
    assert "huggingface-cli login" in text


def test_card_describes_8_class_raw_schema_and_6_class_eval():
    text = CARD.read_text(encoding="utf-8")
    for raw in ["Background1", "Background2", "Burst2"]:
        assert raw in text, f"raw class missing: {raw}"
    assert "Background1 ∪ Background2" in text or "Background1 + Background2" in text


def test_card_describes_npz_schema():
    text = CARD.read_text(encoding="utf-8")
    assert "(N, 7)" in text or "(P, 7)" in text
    for col in ["nx", "ny", "nz", "label"]:
        assert col in text
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_dataset_card.py -v`
Expected: all fail (file missing).

- [ ] **Step 3: Create docs/hf-dataset-card/README.md**

```markdown
---
license: apache-2.0
task_categories:
  - image-segmentation
language:
  - en
size_categories:
  - 1K<n<10K
tags:
  - 3d
  - point-cloud
  - semantic-segmentation
  - defect-detection
  - industrial
  - automotive
  - sealing-nail
pretty_name: OPT-SND (Sealing Nail Defect)
extra_gated_prompt: |
  By accessing this dataset you agree to use it for research purposes only,
  to cite the accompanying paper, and not to redistribute the raw files.
extra_gated_fields:
  Affiliation: text
  Intended use: text
---

# OPT-SND — Sealing Nail Defect Dataset

3D point-cloud semantic-segmentation dataset for industrial defect detection on automotive battery sealing nails. Released alongside the EAAI 2026 paper *"An improved Graph Attention Network for Semantic Segmentation of Industrial Point Clouds in Automotive Battery Sealing Nail Defect Detection."*

Code: <https://github.com/WillPANSUTD/LGASS> &nbsp;·&nbsp; Paper: EAAI 2026, vol. 163, pp. 112793.

![Graphical abstract](figures/graphical_abstract.png)

## Gated access

This dataset is **gated**. To download, complete all three steps:

1. **Authenticate locally:**
   ```bash
   huggingface-cli login
   ```
   Provide a token from <https://huggingface.co/settings/tokens>.
2. **Accept the terms:** open <https://huggingface.co/datasets/vpan1226/OPT-SND> in a browser, click **Access repository**, fill in the form, agree to the conditions.
3. **Wait for approval** (manual review).

Without all three steps, `load_dataset(...)` and `hf_hub_download(...)` will raise 401 / 403.

## Quick start

```python
from datasets import load_dataset

# Requires gated-access steps above.
ds = load_dataset("vpan1226/OPT-SND")
sample = ds["train"][0]
points = sample["points"]   # (P, 7) float32 — see Data fields below
```

Or load a single file directly:

```python
import numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="vpan1226/OPT-SND",
    repo_type="dataset",
    filename="sealingNail_npz/train/sample_0001.npz",
)
arr = np.load(path, allow_pickle=True)["points"]  # (P, 7) float32
xyz = arr[:, 0:3]; feats = arr[:, 3:6]; labels = arr[:, 6].astype("int64")
```

## Dataset summary

OPT-SND contains industrial 3D scans of automotive battery sealing nails, captured by an OPT-LPC20 line laser profiler in a controlled production environment. Each scan is a per-point-cloud sample with surface coordinates, normals, and a per-point semantic label drawn from an 8-class raw schema (described below). The dataset is intended for benchmarking semantic-segmentation methods on highly reflective metallic surfaces where color/intensity cues are unreliable.

## Dataset structure

```
sealingNail_npz/
├── train/   *.npz
└── test/    *.npz
```

Sample counts and split sizes are visible to authenticated viewers on the HF web UI; this card avoids hard-coding them so it stays accurate as the dataset evolves.

## Data fields

Each `.npz` contains a single key `points`, shape `(N, 7)`, dtype `float32`:

| Column | Field | Description                              |
|--------|-------|------------------------------------------|
| 0      | x     | X coordinate (mm)                         |
| 1      | y     | Y coordinate (mm)                         |
| 2      | z     | Z coordinate (mm)                         |
| 3      | nx    | surface-feature / normal x                |
| 4      | ny    | surface-feature / normal y                |
| 5      | nz    | surface-feature / normal z                |
| 6      | label | semantic class id (raw 8-class, 0..7)     |

Cast column 6 to `int64` after loading.

## Class definitions

**Raw schema (8 classes, as stored on disk):**

| ID | Name        | Description                                  |
|----|-------------|----------------------------------------------|
| 0  | Background1 | Background region / nominal surface (A)       |
| 1  | Burst       | Crack / rupture defect                        |
| 2  | Pit         | Concave depression                            |
| 3  | Stain       | Surface discoloration / contamination         |
| 4  | Warpage     | Out-of-plane deformation                      |
| 5  | Background2 | Background region / nominal surface (B)       |
| 6  | Burst2      | Burst variant                                 |
| 7  | Pinhole     | Small puncture                                |

**Paper evaluation view (6 classes, after merge):**

The accompanying EAAI 2026 paper reports metrics on a **merged 6-class** taxonomy:

- `Normal = Background1 ∪ Background2`
- `Burst = Burst ∪ Burst2`
- `Pit, Stain, Warpage, Pinhole` unchanged.

Class distribution (merged 6-class, paper Table 1):

| Class   | Count |
|---------|-------|
| Normal  | 271   |
| Burst   | 170   |
| Pit     | 112   |
| Stain   | 102   |
| Warpage | 132   |
| Pinhole | 142   |

## Acquisition setup

- Camera: OPT-LPC20 line laser profiler, blue laser (405 nm).
- Acquisition rate: 10,000 profiles / second.
- Scan speed: 10 mm/s (mobile robotic platform).
- Lateral resolution: ~0.05 mm.
- Depth accuracy: < 0.01 mm.
- Points per profile: 3,000 across the horizontal axis.

## Annotation protocol

All scans were manually annotated by trained operators following a strict quality-control protocol; multi-annotator review was used on ambiguous samples to enforce labeling consistency.

## Considerations

- **Class imbalance.** Defect classes are far rarer than the background; loss reweighting is recommended (the LGASS reference implementation derives weights from class frequency).
- **Single-domain coverage.** All scans come from one production line and one camera setup. Generalization to other lines may require fine-tuning or domain-adaptation.
- **No personal data.** No faces, no identifying information; samples are mechanical parts only.

## Citation

```bibtex
@article{pan2026improved,
  title   = {An improved graph attention network for semantic segmentation of industrial point clouds in automotive battery sealing nail defect detection},
  author  = {Pan, Wei and Wu, Yuhao and Tang, Wenming and Lu, Qinghua and Zhang, Yunzhi},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {163},
  pages   = {112793},
  year    = {2026},
  publisher = {Elsevier}
}
```

## License

Apache 2.0.

## Links

- Code: <https://github.com/WillPANSUTD/LGASS>
- Pre-trained checkpoint: <https://huggingface.co/vpan1226/LGASS> (uploaded post-publication)
- Paper: EAAI 2026 (DOI to fill).
```

- [ ] **Step 4: Run test, expect PASS**

Run: `pytest tests/test_dataset_card.py -v`
Expected: 6 passes.

- [ ] **Step 5: Commit**

```bash
git add docs/hf-dataset-card/ tests/test_dataset_card.py
git commit -m "docs: add staged HuggingFace dataset card for OPT-SND"
```

---

## Task 9: Phase 2 — `evaluate.py`

**Files:**
- Create: `evaluate.py`
- Test: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing test (CLI smoke + markdown shape, no GPU)**

Create `tests/test_evaluate.py`:

```python
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
    # Implementer note: create a tiny fake checkpoint that the existing
    # Model class can load, point at a 2-sample npz dataset, run on CPU,
    # confirm the output markdown contains: OA, mAcc, mIoU and per-class
    # column headers in paper Table 3 order.
    pytest.skip("requires real model + dataset; run manually after retrain")
```

- [ ] **Step 2: Run test, expect FAIL on help**

Run: `pytest tests/test_evaluate.py::test_evaluate_help -v`
Expected: FAIL.

- [ ] **Step 3: Create evaluate.py**

```python
"""Evaluate an LGASS checkpoint and emit a paper-shaped markdown table.

Usage:
    python evaluate.py --checkpoint logs/best.pth \\
        --data_root data/sealingNail_npz --split test --output results/eval.md
"""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# 6-class merged eval (paper Table 3 order: Burst, Pit, Stain, Warpage, Pinhole).
RAW_TO_MERGED = {0: 0, 5: 0, 1: 1, 6: 1, 2: 2, 3: 3, 4: 4, 7: 5}
MERGED_NAMES = ["Normal", "Burst", "Pit", "Stain", "Warpage", "Pinhole"]
PAPER_DEFECT_COLS = ["Burst", "Pit", "Stain", "Warpage", "Pinhole"]


def remap_to_merged(labels):
    out = np.empty_like(labels)
    for raw, merged in RAW_TO_MERGED.items():
        out[labels == raw] = merged
    return out


def per_class_iou(gt, pred, num_classes):
    ious = []
    for c in range(num_classes):
        gm = gt == c; pm = pred == c
        union = (gm | pm).sum()
        ious.append(float((gm & pm).sum() / union) if union else float("nan"))
    return ious


def metrics(gt, pred, num_classes):
    oa = float((gt == pred).mean())
    ious = per_class_iou(gt, pred, num_classes)
    valid = [x for x in ious if x == x]
    miou = float(np.mean(valid)) if valid else 0.0
    accs = []
    for c in range(num_classes):
        gm = gt == c
        accs.append(float((pred[gm] == c).mean()) if gm.sum() else float("nan"))
    valid_acc = [x for x in accs if x == x]
    macc = float(np.mean(valid_acc)) if valid_acc else 0.0
    return oa, macc, miou, ious


def to_markdown_row(model_name, oa, macc, miou, defect_ious):
    cells = [model_name, f"{oa*100:.2f}", f"{macc*100:.2f}", f"{miou*100:.2f}"]
    cells.extend(f"{x*100:.2f}" if x == x else "—" for x in defect_ious)
    return "| " + " | ".join(cells) + " |"


def make_markdown_table(rows):
    header = "| Method | OA | mAcc | mIoU | " + " | ".join(PAPER_DEFECT_COLS) + " |"
    sep = "|" + "---|" * (4 + len(PAPER_DEFECT_COLS))
    return "\n".join([header, sep, *rows])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data/sealingNail_npz")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--output", required=True, help="markdown file to write")
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--raw-schema", action="store_true",
                   help="evaluate on raw 8-class schema instead of paper 6-class merge")
    args = p.parse_args()

    from util.sealingNails_npz import SealingNailDatasetNPZ
    from util.data_util import collate_fn
    from model.sem.GraphAttention import graphAttention_seg_repro as Model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raw_classes = 8
    eval_classes = raw_classes if args.raw_schema else 6

    ds = SealingNailDatasetNPZ(
        root=args.data_root, npoints=args.num_points,
        split=args.split, use_cache=True,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        drop_last=False, collate_fn=collate_fn, pin_memory=True,
    )

    model = Model(c=6, k=raw_classes).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    all_gt = []; all_pred = []
    with torch.no_grad():
        for batch in loader:
            coords, feats, labels = [x.to(device) for x in batch[:3]]
            logits = model(coords, feats)            # adapt to actual model signature
            pred_raw = logits.argmax(dim=-1).cpu().numpy().reshape(-1)
            gt_raw = labels.cpu().numpy().reshape(-1)
            if args.raw_schema:
                all_gt.append(gt_raw); all_pred.append(pred_raw)
            else:
                all_gt.append(remap_to_merged(gt_raw))
                all_pred.append(remap_to_merged(pred_raw))

    gt = np.concatenate(all_gt); pred = np.concatenate(all_pred)
    oa, macc, miou, ious = metrics(gt, pred, eval_classes)

    if args.raw_schema:
        defect_cols = ["Burst", "Pit", "Stain", "Warpage", "Background2", "Burst2", "Pinhole"]
        defect_ious = [ious[i] for i in [1, 2, 3, 4, 5, 6, 7]]
        header = "| Method | OA | mAcc | mIoU | " + " | ".join(defect_cols) + " |"
        sep = "|" + "---|" * (4 + len(defect_cols))
        row = to_markdown_row("LGASS (raw 8cls)", oa, macc, miou, defect_ious)
        md = "\n".join([header, sep, row])
    else:
        # Merged: skip class 0 (Normal) for the per-defect columns.
        defect_ious = [ious[i] for i in [1, 2, 3, 4, 5]]
        md = make_markdown_table([to_markdown_row("LGASS", oa, macc, miou, defect_ious)])

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(md + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(md)


if __name__ == "__main__":
    main()
```

**Note to implementer:** the `model(coords, feats)` call signature is a placeholder — adapt to the actual `graphAttention_seg_repro` forward signature by reading `model/sem/GraphAttention.py` and matching how `train.py` calls the model. If the existing `train.py` passes additional inputs (e.g. neighbors, batch indices), `evaluate.py` must do the same.

- [ ] **Step 4: Run help test, expect PASS**

Run: `pytest tests/test_evaluate.py::test_evaluate_help -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluate.py emitting paper-shaped metrics markdown"
```

---

## Task 10: Phase 2 — `scripts/reproduce_paper.sh`

**Files:**
- Create: `scripts/reproduce_paper.sh`
- Test: `tests/test_scripts.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_scripts.py`:

```python
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
```

- [ ] **Step 2: Run test, expect FAIL**

Run: `pytest tests/test_scripts.py -v`
Expected: 2 fails.

- [ ] **Step 3: Create scripts/reproduce_paper.sh**

```bash
#!/usr/bin/env bash
# Reproduce the EAAI 2026 paper run.
# Usage: bash scripts/reproduce_paper.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Sanity: CUDA + pointops available.
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'"
python -c "import pointops" || {
    echo "pointops not installed — run: cd lib/pointops && python setup.py install" >&2
    exit 1
}

OUTPUT_DIR="${OUTPUT_DIR:-logs/paper_run_$(date +%Y%m%d_%H%M%S)}"
echo "Training to $OUTPUT_DIR"
python train.py --config configs/paper.yaml --output_dir "$OUTPUT_DIR"

echo "Evaluating best checkpoint"
python evaluate.py \
    --checkpoint "$OUTPUT_DIR/best.pth" \
    --data_root data/sealingNail_npz \
    --split test \
    --output "$OUTPUT_DIR/eval.md"

echo "Done. Paper-shaped metrics written to $OUTPUT_DIR/eval.md"
```

- [ ] **Step 4: chmod (POSIX only)**

Run: `chmod +x scripts/reproduce_paper.sh` (skip on Windows; pytest handles via `os.name`).

- [ ] **Step 5: Run test, expect PASS**

Run: `pytest tests/test_scripts.py -v`
Expected: 2 passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/reproduce_paper.sh tests/test_scripts.py
git commit -m "feat: add scripts/reproduce_paper.sh end-to-end reproduce harness"
```

---

## Task 11: Phase 2 — `scripts/upload_to_hf.py`

**Files:**
- Create: `scripts/upload_to_hf.py`
- Modify: `tests/test_scripts.py` (extend)
- Modify: `requirements.txt` (add `huggingface_hub`)

- [ ] **Step 1: Add huggingface_hub to requirements**

Append to `requirements.txt`:

```
huggingface_hub>=0.20.0
```

- [ ] **Step 2: Write failing tests**

Append to `tests/test_scripts.py`:

```python
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
```

- [ ] **Step 3: Run test, expect FAIL**

Run: `pytest tests/test_scripts.py -v`
Expected: 2 new fails.

- [ ] **Step 4: Create scripts/upload_to_hf.py**

```python
"""Upload an LGASS checkpoint to HuggingFace Hub as a model repo.

Usage:
    python scripts/upload_to_hf.py --checkpoint logs/best.pth --repo vpan1226/LGASS
    # add --dry-run to print the planned actions without touching the network
"""
import argparse
import sys
from pathlib import Path

MODEL_CARD_TEMPLATE = """\
---
license: apache-2.0
library_name: pytorch
tags:
  - point-cloud
  - semantic-segmentation
  - graph-attention
  - defect-detection
  - industrial
pipeline_tag: image-segmentation
---

# LGASS — sealing-nail defect-detection checkpoint

PyTorch checkpoint for the LGASS architecture from the EAAI 2026 paper
*"An improved Graph Attention Network for Semantic Segmentation of Industrial
Point Clouds in Automotive Battery Sealing Nail Defect Detection"*.

- **Code:** <https://github.com/WillPANSUTD/LGASS>
- **Dataset:** <https://huggingface.co/datasets/vpan1226/OPT-SND>
- **Architecture:** Graph-attention encoder–decoder with LGAF + SAG-Pooling.

## Reported metrics (paper Table 3, 6-class merged eval)

| OA | mAcc | mIoU | Burst | Pit | Stain | Warpage | Pinhole |
|----|------|------|-------|-----|-------|---------|---------|
| 99.47 | 92.37 | 79.23 | 76.22 | 72.17 | 71.33 | 91.61 | 64.95 |

## Loading

```python
import torch
from huggingface_hub import hf_hub_download
ckpt = torch.load(hf_hub_download(repo_id="vpan1226/LGASS", filename="model.pth"),
                  map_location="cpu")
```

## Citation

```bibtex
@article{pan2026improved,
  title   = {An improved graph attention network for semantic segmentation of industrial point clouds in automotive battery sealing nail defect detection},
  author  = {Pan, Wei and Wu, Yuhao and Tang, Wenming and Lu, Qinghua and Zhang, Yunzhi},
  journal = {Engineering Applications of Artificial Intelligence},
  volume  = {163},
  pages   = {112793},
  year    = {2026},
  publisher = {Elsevier}
}
```
"""


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, help=".pth file to upload")
    p.add_argument("--repo", required=True, help="HF repo id, e.g. vpan1226/LGASS")
    p.add_argument("--dry-run", action="store_true",
                   help="print actions without uploading")
    p.add_argument("--repo-type", default="model", choices=["model"],
                   help="HF repo type")
    args = p.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.is_file():
        sys.exit(f"checkpoint not found: {ckpt}")

    if args.dry_run:
        print("DRY RUN — would perform:")
        print(f"  1. create_repo({args.repo!r}, repo_type={args.repo_type!r}, exist_ok=True)")
        print(f"  2. upload_file(model.pth from {ckpt})")
        print(f"  3. upload_file(README.md, generated model card, {len(MODEL_CARD_TEMPLATE)} bytes)")
        return

    from huggingface_hub import HfApi, create_repo
    api = HfApi()
    create_repo(args.repo, repo_type=args.repo_type, exist_ok=True)

    api.upload_file(
        path_or_fileobj=str(ckpt),
        path_in_repo="model.pth",
        repo_id=args.repo,
        repo_type=args.repo_type,
    )
    print(f"uploaded checkpoint -> {args.repo}/model.pth")

    api.upload_file(
        path_or_fileobj=MODEL_CARD_TEMPLATE.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo,
        repo_type=args.repo_type,
    )
    print(f"uploaded model card -> {args.repo}/README.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test, expect PASS**

Run: `pytest tests/test_scripts.py -v`
Expected: all passes.

- [ ] **Step 6: Commit**

```bash
git add scripts/upload_to_hf.py tests/test_scripts.py requirements.txt
git commit -m "feat: add scripts/upload_to_hf.py for checkpoint publishing"
```

---

## Task 12: Final integration check

**Files:** none (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest -v`
Expected: all tests pass; gpu-marked tests are skipped.

- [ ] **Step 2: Lint check the README structurally**

Run:

```bash
python -c "
from pathlib import Path
for f in ['README.md', 'README_CN.md', 'docs/hf-dataset-card/README.md']:
    text = Path(f).read_text(encoding='utf-8')
    assert 'TODO' not in text and 'TBD' not in text, f
    assert 'sealingNail_normal' not in text, f
print('docs structurally clean')
"
```

Expected: prints `docs structurally clean`.

- [ ] **Step 3: Manual smoke**

Run: `python visualize.py --help` and `python export_report.py --help` and `python train.py --help` and `python evaluate.py --help`.
All four should print usage; none should crash on import.

- [ ] **Step 4: Summary commit (no-op or doc bump)**

If anything in the previous tasks left untracked artifacts, clean them up. Otherwise skip.

- [ ] **Step 5: Show final status**

Run: `git log --oneline -20 && git status --short`.
Expected: ~12 new commits since `165cb33`; working tree clean except for the (intentionally untouched) `pointops` submodule modified marker.

---

## Self-review checklist (run after writing the plan)

- [x] Spec §3 naming → covered by Task 6 (LGASS in README) and Task 8 (LGASS link in dataset card).
- [x] Spec §5.1 git cleanup → already executed before writing this plan; HEAD = `165cb33`.
- [x] Spec §5.2 LICENSE → Task 1.
- [x] Spec §5.3 / §5.4 README rewrites → Tasks 6, 7.
- [x] Spec §5.5 missing scripts → Tasks 4, 5.
- [x] Spec §5.6 configs + seed → Tasks 2, 3.
- [x] Spec §5.7 dataset card → Task 8.
- [x] Spec §6 Phase 2 → Tasks 9, 10, 11.
- [x] No `TBD` / `TODO` / "implement later" in plan steps.
- [x] Hyperparameter mismatch (paper vs train.py) → addressed via two-config approach (Task 2).
- [x] 8-class raw vs 6-class paper view → addressed in dataset card (Task 8) + evaluate.py `--raw-schema` flag (Task 9).
- [x] Schema correction (single `points` (N,7) array) → reflected in Tasks 4, 5, 8, 9.

## Open items requiring user action (after plan execution)

These are not implementable in this session; flagged for the user to handle:

1. Rename GitHub repo `LGANet` → `LGASS` via GitHub UI (`gh repo rename LGASS` or web).
2. Run `bash scripts/reproduce_paper.sh` on a CUDA box to retrain and produce the checkpoint.
3. Run `python scripts/upload_to_hf.py --checkpoint logs/.../best.pth --repo vpan1226/LGASS` after retraining.
4. Upload `docs/hf-dataset-card/README.md` to `huggingface.co/datasets/vpan1226/OPT-SND` (replacing the current README on HF).
5. Push committed changes: `git push origin main`.
6. Confirm or override the placeholder `lr_decay_milestones`, `lr_decay_gamma`, `k_neighbors` in `configs/paper.yaml` after retraining.
