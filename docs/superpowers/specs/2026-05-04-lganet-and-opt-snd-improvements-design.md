# LGANet repo + OPT-SND dataset improvements — Design

**Date:** 2026-05-04
**Author:** Wei Pan (with Claude assistance)
**Status:** Draft for user review

## 1. Goals

Improve two coupled artifacts so that a third-party reader of the published paper can:

1. Find a clear, well-documented code repository.
2. Locate, download (after gating), and load the dataset with a single `load_dataset()` call.
3. Reproduce the paper's reported numbers given GPU time, using a fixed config and a published checkpoint.
4. Run inference and visualize predictions with the included scripts.

Both artifacts target the **published** state — they accompany the EAAI paper (volume 163, pages 112793, 2026).

## 2. Scope

In scope:

- Code repo `WillPANSUTD/LGANet` (to be renamed to `LGASS`): README rewrite (EN + CN), LICENSE, missing scripts, configs, seed handling, evaluation script, checkpoint upload script.
- HuggingFace dataset `vpan1226/OPT-SND`: full dataset card rewrite with proper YAML frontmatter, `load_dataset()` snippet, gated access workflow, schema, splits, acquisition details.
- Two naming alignments: method `LGANet` → `LGASS` (matches paper), class `Stain` confirmed (typo `Moltenbead` in paper Table 3).

Out of scope:

- Retraining the network and producing the final checkpoint. This is a Phase-2 GPU-time activity; we deliver scripts and configs that make the user's retrain a one-command operation.
- Any modification of the paper PDF itself.
- Changes to `pointops` library internals.

## 3. Naming decisions

**Method name:** `LGASS` (Local Graph Attention for Semantic Segmentation), matching the paper abstract, Table 3, and conclusion. Both READMEs and Phase-2 HF model repo will use `LGASS`. The GitHub repo will be renamed `LGANet` → `LGASS` by the user via GitHub UI; GitHub auto-redirects the old URL so the link printed in the paper stays valid.

**Class name:** `Stain` (污渍). The paper's Table 1 (dataset description) and the existing README both use `Stain`. The string `Moltenbead` appearing in paper Table 3 is a translation slip and will not be propagated into the code/dataset documentation. The dataset card and code use `Stain` consistently.

**Class count — 8 raw vs 6 reported:** The on-disk dataset (and `train.py`) defines **8 classes**: `{Background1: 0, Burst: 1, Pit: 2, Stain: 3, Warpage: 4, Background2: 5, Burst2: 6, Pinhole: 7}`. The paper reports results on a **merged 6-class** view: `Background1 ∪ Background2 → Normal`, `Burst ∪ Burst2 → Burst`. The dataset card describes both the raw 8-class schema and the merged 6-class evaluation view used in the paper. The code-repo README's results table keeps the 6-column format (matching paper Table 3).

**Hyperparameter mismatch — paper Table 2 vs original `train.py`:** Paper Table 2 lists `optimizer=AdamW, lr=0.01, batch_size=8`. The original `train.py` uses `optimizer=Adam, lr=0.001, batch_size=4`. We do not know which run produced the paper's headline numbers. We ship two configs side-by-side with explicit comments: `configs/paper.yaml` (Table 2 values) and `configs/original.yaml` (original `train.py` values). `train.py` defaults to `paper.yaml`. The README documents both files and the discrepancy.

## 4. Phase split

**Phase 1 — this session:** All documentation, all utility scripts that don't require a checkpoint, configs, license, git cleanup. Deliverable: a coherent repo + dataset card the user can publish immediately.

**Phase 2 — user runs on GPU:** Retrain via `scripts/reproduce_paper.sh`, evaluate via `evaluate.py`, upload `.pth` to `vpan1226/LGASS` via `scripts/upload_to_hf.py`. Deliverable from us: ready-to-run scripts.

## 5. Phase 1 deliverables

### 5.1 Git state cleanup

Steps (with user approval):

1. `git rebase --abort` to clear the in-progress rebase.
2. `git fetch origin` to refresh `origin/main` (currently 2 commits behind: `a60368c`, `8def69e`).
3. `git pull` (fast-forward) to align local with `8def69e`.
4. `git submodule foreach git reset --hard` if the `pointops` submodule's "modified content" is just build artifacts — confirm with user first.

### 5.2 LGASS code repo: LICENSE

New file `LICENSE` at repo root: standard Apache 2.0 text, copyright `2026 Wei Pan and OPT Machine Vision`.

### 5.3 LGASS code repo: README.md (EN) rewrite

Sections (in order):

1. **Title** — "LGASS — An improved Graph Attention Network for Semantic Segmentation of Industrial Point Clouds in Automotive Battery Sealing Nail Defect Detection" (verbatim paper title prefixed with the method name; preserves searchability for readers landing here from the EAAI paper)
2. **Graphical abstract** image (existing).
3. **News / status** — paper accepted at EAAI 2026.
4. **Quick links** — paper DOI · dataset (HF) · checkpoint (HF, when published).
5. **Overview** — one paragraph, current content kept.
6. **Defect classes** — Burst, Pit, Stain, Warpage, Pinhole, Normal (background).
7. **Data acquisition** — current paragraph kept.
8. **Architecture** — text + the two key innovations (LGAF, SAG-Pooling) named.
9. **Results** — full Table 3 reproduction (markdown table, our row bolded):

   | Method     | OA    | mAcc  | mIoU  | Burst | Pit   | Stain | Warpage | Pinhole |
   |------------|-------|-------|-------|-------|-------|-------|---------|---------|
   | PointNet   | 83.29 | 72.83 | 46.21 | 32.27 | 34.96 | 19.97 | 65.37   | 36.04   |
   | PointNet++ | 89.33 | 84.65 | 62.67 | 47.91 | 59.32 | 42.37 | 81.64   | 46.53   |
   | GACNet     | 96.05 | 87.84 | 69.41 | 56.02 | 67.83 | 54.54 | 86.96   | 59.94   |
   | PTv1       | 98.77 | 90.96 | 72.83 | 68.54 | 62.83 | 58.37 | 89.67   | 58.37   |
   | OA-CNN     | 98.72 | 91.13 | 74.04 | 64.89 | 66.91 | 64.87 | 88.37   | 60.04   |
   | PTv3       | 99.03 | 94.96 | 75.13 | 77.32 | 67.98 | 60.44 | 89.17   | 56.21   |
   | **LGASS**  | **99.47** | 92.37 | **79.23** | 76.22 | **72.17** | **71.33** | **91.61** | **64.95** |

10. **Ablation** — Table 4 (modules) and Table 5 (input types) reproduced.
11. **Hyperparameters** — Table 2 reproduced as a "reproduction config" reference.
12. **Dataset structure** — directory tree pointing at `sealingNail_npz/{train,test}/...`, class distribution table:

    | Class   | Count |
    |---------|-------|
    | Normal  | 271   |
    | Burst   | 170   |
    | Pit     | 112   |
    | Stain   | 102   |
    | Warpage | 132   |
    | Pinhole | 142   |

13. **Installation** — current conda steps; verify `pointops` source.
14. **Usage** — train/eval/visualize/report-export commands (all of these scripts exist after Phase 1).
15. **Citation** — current bibtex retained.
16. **License** — Apache 2.0.
17. **Acknowledgements** — current content retained.

Path inconsistency fix: every reference to `data/sealingNail_normal/` becomes `data/sealingNail_npz/` to match what's actually on HuggingFace.

Phantom scripts: `visualize.py` and `export_report.py` references will resolve to actual files (see 5.5).

### 5.4 LGASS code repo: README_CN.md rewrite

Mirror of the English README, section-for-section. Currently CN README ends mid-Installation (line 60); we rewrite it to full parity.

### 5.5 LGASS code repo: missing scripts

#### 5.5.1 `visualize.py`

CLI: `python visualize.py --input <path-to-.ply-or-.npz> [--predictions <pred-path>] [--save <output.png>]`

- Loads point cloud (PLY via Open3D, NPZ via numpy).
- Colorizes points by semantic label (predefined class palette: 6 colors).
- Opens an interactive Open3D viewer; with `--save`, renders an offscreen PNG to the output path.

#### 5.5.2 `export_report.py`

CLI: `python export_report.py --input <results-dir> --output <reports-dir>`

- Walks `results-dir` for prediction `.npz` / `.npy` outputs from `batch_demo.py`.
- For each sample: renders ground truth + prediction screenshots (reuses visualize.py's render code), computes per-sample OA / mIoU / per-class point counts.
- Aggregates into `reports-dir/index.html` with a sortable table + thumbnail grid; per-sample `reports-dir/<sample>.html` for detail pages.
- Pure-stdlib HTML generation, no external templating dep.

### 5.6 LGASS code repo: configs and seed handling

#### 5.6.1 `configs/paper.yaml` (default)

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

Three values flagged as **placeholders** because the paper does not specify them: `lr_decay_milestones`, `lr_decay_gamma`, `k_neighbors`. Comments tell the reader to edit if the true values are recovered.

#### 5.6.2 `configs/original.yaml`

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

#### 5.6.3 `train.py` modifications

- Top of file: add `set_seed(seed)` helper (seeds `random`, `numpy`, `torch`, `torch.cuda`, sets `torch.backends.cudnn.deterministic=True` and `benchmark=False`).
- Add `argparse` `--config` flag defaulting to `configs/default.yaml`.
- Replace hardcoded hyperparameters (lr / batch / epoch / etc.) with reads from the loaded config dict.
- Log the resolved config as JSON at run start so future debugging knows what was used.

### 5.7 OPT-SND HuggingFace dataset card

Replace the current `README.md` (which appears to be a copy of the code-repo README — not a dataset card) with a proper card. Sections:

#### 5.7.1 YAML frontmatter

```yaml
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
```

#### 5.7.2 One-line description + graphical abstract image

References `figures/graphical_abstract.png` (already on the HF repo) at the top.

#### 5.7.3 Gated access workflow (prominent, before any code)

A dedicated section with three explicit steps:

1. Log in: `huggingface-cli login` (provide token from huggingface.co/settings/tokens).
2. Visit the dataset page in a browser, click **"Access repository"**, fill the gated form, accept the terms.
3. Wait for approval (manual review).

Only after these three steps does any code in the next section work. We make this its own section because users hit confusing 401/403 errors otherwise.

#### 5.7.4 Quick start — `load_dataset` snippet

```python
from datasets import load_dataset

# Requires: huggingface-cli login + acceptance of gated terms (see above).
ds = load_dataset("vpan1226/OPT-SND")  # streaming=True optional

sample = ds["train"][0]
points = sample["points"]      # (N, 3) float32 — xyz coords
normals = sample["normals"]    # (N, 3) float32 — surface normals
labels = sample["labels"]      # (N,)   int64   — semantic class id
```

Plus a fallback raw-NPZ snippet for users who want to bypass the `datasets` library:

```python
import numpy as np
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="vpan1226/OPT-SND",
    repo_type="dataset",
    filename="sealingNail_npz/train/sample_0001.npz",
)
data = np.load(path)
print(data.files)  # ['points', 'normals', 'labels']
```

#### 5.7.5 Dataset summary

Two paragraphs: what / why / who / where collected, lifted from the paper's data-acquisition section.

#### 5.7.6 Dataset structure

```
sealingNail_npz/
├── train/
│   └── *.npz         (N samples)
└── test/
    └── *.npz         (M samples)
```

Note: exact N/M counts will be inserted after I check the actual HF repo file listing during implementation. If listing requires authenticated access, we'll ask the user to run `huggingface-cli ls` and paste counts.

#### 5.7.7 Data fields

Each `.npz` contains a single key `points` with shape `(P, 7)`, dtype `float32` (label column may be cast at load time):

| Column index | Field    | Description                          |
|--------------|----------|--------------------------------------|
| 0            | x        | X coordinate (mm)                     |
| 1            | y        | Y coordinate (mm)                     |
| 2            | z        | Z coordinate (mm)                     |
| 3            | nx       | surface-feature / normal x            |
| 4            | ny       | surface-feature / normal y            |
| 5            | nz       | surface-feature / normal z            |
| 6            | label    | semantic class id (raw 8-class, 0..7) |

Loading example:

```python
import numpy as np
data = np.load("sealingNail_npz/train/sample_0001.npz", allow_pickle=True)
arr = data["points"]            # (P, 7) float32
xyz = arr[:, 0:3]
feats = arr[:, 3:6]
labels = arr[:, 6].astype(np.int64)
```

`P` varies per sample; the training loader subsamples to 16384 points.

#### 5.7.8 Class definitions

**Raw schema (8 classes, as stored on disk):**

| ID | Name        | Description                                |
|----|-------------|--------------------------------------------|
| 0  | Background1 | Background region / nominal surface (A)     |
| 1  | Burst       | Crack / rupture defect                      |
| 2  | Pit         | Concave depression                          |
| 3  | Stain       | Surface discoloration / contamination       |
| 4  | Warpage     | Out-of-plane deformation                    |
| 5  | Background2 | Background region / nominal surface (B)     |
| 6  | Burst2      | Burst variant                               |
| 7  | Pinhole     | Small puncture                              |

**Paper evaluation view (6 classes, after merge):**

The EAAI 2026 paper evaluates on a merged 6-class taxonomy:

- `Normal = Background1 ∪ Background2`  (raw 0 ∪ 5)
- `Burst = Burst ∪ Burst2`               (raw 1 ∪ 6)
- `Pit, Stain, Warpage, Pinhole` unchanged

Class distribution from paper Table 1 (in the merged 6-class view) is reproduced in 5.3 above. Users intending to reproduce paper numbers should apply the same merge at evaluation time; the included `evaluate.py` (Phase 2) does this by default and exposes `--raw-schema` to evaluate on all 8 classes.

#### 5.7.9 Acquisition setup

Hardware/optical specs (currently in HF README; keep): OPT-LPC20, 405nm blue laser, 10k profiles/s, 10 mm/s scan, 0.05mm lateral res, <0.01mm depth accuracy, 3000 points/profile.

#### 5.7.10 Annotation protocol

One paragraph: manual labeling, multi-annotator quality control.

#### 5.7.11 Considerations

- Class imbalance (Normal:Burst:Pit:Stain:Warpage:Pinhole ≈ 271:170:112:102:132:142).
- Single source domain (one production line, one camera) — generalization to other lines may require fine-tuning.
- No personal data, no faces, no identifying information.

#### 5.7.12 Citation + links

Bibtex (current paper) + links to GitHub `WillPANSUTD/LGASS` and HF model `vpan1226/LGASS` (placeholder until Phase 2 publishes weights).

## 6. Phase 2 deliverables (scripts only, user runs)

### 6.1 `scripts/reproduce_paper.sh`

Single-line bash script: `python train.py --config configs/paper.yaml --output_dir logs/paper_run`. Adds environment sanity checks (CUDA available, pointops importable) up front.

### 6.2 `evaluate.py`

CLI: `python evaluate.py --checkpoint <path.pth> --data_root data/sealingNail_npz --split test --output results/eval.md`

- Loads checkpoint, runs forward over the test split.
- Computes OA, mAcc, mIoU, per-class IoU.
- Outputs a markdown table directly pasteable into the README — same column order as paper Table 3 so "ours" row is drop-in.

### 6.3 `scripts/upload_to_hf.py`

CLI: `python scripts/upload_to_hf.py --checkpoint <path.pth> --repo vpan1226/LGASS`

- Uses `huggingface_hub` SDK (no git-lfs dependency).
- Creates the model repo if it doesn't exist.
- Uploads checkpoint plus a generated `model_card.md` (template included) with: bibtex, dataset link, expected metrics, expected hardware, license.

## 7. File change summary

```
LGANet repo (to be renamed LGASS):
├── LICENSE                          (NEW, Apache 2.0)
├── README.md                        (REWRITE: Phase 1)
├── README_CN.md                     (REWRITE: Phase 1, full parity)
├── configs/
│   ├── paper.yaml                   (NEW, Phase 1, default config)
│   └── original.yaml                (NEW, Phase 1, original train.py values)
├── scripts/
│   ├── reproduce_paper.sh           (NEW, Phase 2)
│   └── upload_to_hf.py              (NEW, Phase 2)
├── visualize.py                     (NEW, Phase 1)
├── export_report.py                 (NEW, Phase 1)
├── evaluate.py                      (NEW, Phase 2)
└── train.py                         (MODIFY: Phase 1, add seed + config)

OPT-SND HF dataset:
└── README.md                        (REWRITE: Phase 1, proper dataset card)
```

## 8. Risks and known unknowns

- **Phase 2 reproduced numbers may drift from paper** by ±0.5% due to non-determinism in pointops kernels, CUDA / cudnn version drift, and PyTorch version differences. README will state this explicitly.
- **`k_neighbors` and `lr_decay_milestones` are not in paper Table 2.** Configs use sensible defaults marked as placeholders; user must verify or adjust on first retrain.
- **Gated dataset means the `load_dataset()` example fails for unauthenticated readers.** Mitigated by the prominent gated-access section in the dataset card.
- **GitHub repo rename is a user-only action** (requires GitHub UI or `gh repo rename`). Until done, dataset card and Phase-2 model card link to the eventual new URL; old URL keeps working via GitHub's auto-redirect.
- **`pointops` submodule modified state** may represent intentional user edits. Will not reset without explicit user confirmation.
- **HF dataset file listing requires authenticated access**, so exact train/test sample counts may need user input during implementation.

## 9. Approval gates

- §3 naming decisions confirmed by user: D1=(b) LGASS, D2=(a) Stain.
- §5.1 git cleanup actions require user "go" before execution.
- §5.6.1 placeholder hyperparameters require user to flag any known correct values before we lock the config.
- After Phase 1 implementation, the user reviews the rewritten READMEs and dataset card before any push to `origin` or `huggingface.co`.
