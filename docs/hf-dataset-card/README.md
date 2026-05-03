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
