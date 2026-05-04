# LGASS — An improved Graph Attention Network for Semantic Segmentation of Industrial Point Clouds in Automotive Battery Sealing Nail Defect Detection

<p align="center">
  <img src="figures/graphical_abstract.png" alt="Graphical abstract of LGASS" width="85%">
</p>

> **Note:** the codebase was renamed from LGANet → LGASS to match the published method name.
> The paper title remains as above for searchability. The old GitHub URL still redirects.

## News

- **EAAI 2026** — paper accepted (volume 163, pages 112793).
- Dataset released on HuggingFace as a gated repo: <https://huggingface.co/datasets/vpan1226/OPT-SND>.

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

<p align="center">
  <img src="figures/data_acqusition.png" alt="data source" width="85%">
</p>

The dataset used in this project is collected from real-world industrial production lines for automotive battery sealing nail inspection.

All point cloud data are acquired using industrial 3D vision systems deployed in controlled manufacturing environments, covering various defect types such as surface deformation, misalignment, and structural anomalies of sealing nails. The image acquisition and imaging module consists of
a Laser Profiler as a 3D scanning system (OPT-LPC20 Laser Profiler) with a mobile robotic platform and a test specimen. 
The laser profiler is configured in linear scanning mode. The system uses a blue laser stripe (wavelength 405nm) with a profile acquisition rate of 10,000 profiles per second, ensuring high fidelity on metallic reflective surfaces.
During scanning, the sealing nails were placed on a precision rotary stage to ensure uniform coverage. The scanning speed was set to 10mm/s, and each profile captured 3000 points across the horizontal axis, resulting in a lateral resolution of ~0.05mm and vertical depth accuracy of <0.01mm. All scans were manually verified for completeness and noise artifacts.The battery is mounted on the mobile robot platform, and the calibrated 3D line laser scanning camera projects laser stripes onto the sealing nail area of the battery. Through synchronized image data acquisition during planar scanning trajectories, surface data of thesealing nail specimen are obtained.The data are manually annotated following a strict quality control process to ensure labeling accuracy and consistency.

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

**Note on reproduction:** the numbers above are the published values from paper Table 3. A locally re-trained checkpoint matching these numbers is **not yet bundled** with this code release — the original checkpoint is no longer on hand and a 300-epoch retrain is in progress. To reproduce yourself, run `bash scripts/reproduce_paper.sh` (uses `configs/paper.yaml`); expect drift of ±0.5% on the headline metrics due to non-determinism in `pointops` kernels and CUDA / PyTorch version differences. Plan for several days of GPU time per the schedule in Table 2.

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

`configs/original.yaml` preserves the values originally hard-coded in `train.py` (AdamW, lr=0.001, batch=4, StepLR every 20 epochs by 0.5x) for transparency. We cannot confirm which set produced the paper's headline numbers without retraining both.

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

### Recommended: Use Conda

1. Create a new Conda environment:

   ```bash
   conda env create -f environment.yml
   ```

2. Activate the environment:

   ```bash
   conda activate sealingnail
   ```

3. Install PyTorch with CUDA:

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. Install other dependencies:

   ```bash
   conda install numpy scipy scikit-learn
   conda install -c conda-forge open3d
   conda install tqdm tensorboard h5py
   ```

5. Ensure C++/build tools (Linux):

   ```bash
   sudo apt install g++
   sudo apt install build-essential
   sudo apt install ninja-build
   ninja --version
   ```

6. Install custom pointops library:

   ```bash
   cd lib/pointops
   python setup.py install
   ```

7. Verify installation:

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

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

If you consider our work useful, please cite:

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

Apache 2.0. See `LICENSE`.

## Acknowledgements

This work was supported in part by **OPT Machine Vision** through internal research and development projects on industrial 3D vision and point cloud inspection systems.

The authors would like to thank all collaborators involved in data collection, annotation, and system deployment for automotive battery sealing nail inspection. We also appreciate the constructive feedback from anonymous reviewers, which helped improve the quality and clarity of this work.

In addition, we acknowledge the open-source contributions from the community. This project benefits from several excellent open-source works in point cloud processing and graph-based learning, including but not limited to:

- [PointNet++](https://github.com/charlesq34/pointnet2)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
- [PointNeXt](https://github.com/guochengqian/PointNeXt)
- [PointOps](https://github.com/POSTECH-CVLab/pointops)

We sincerely thank the authors of these works for making their code publicly available, which greatly facilitated reproducibility and comparative evaluation. 
Parts of the implementation are adapted or inspired by prior works, with appropriate modifications for industrial point cloud segmentation scenarios.
