# LGASS — 用于汽车动力电池密封钉缺陷检测的工业点云语义分割改进图注意力网络

<p align="center">
  <img src="figures/graphical_abstract.png" alt="Graphical abstract of LGASS" width="85%">
</p>

> **说明：** 代码仓库已从 LGANet 重命名为 LGASS，与论文中的方法名保持一致。论文标题保持不变以便检索。原 GitHub URL 仍可重定向。

## 最新动态

- **EAAI 2026** — 论文已录用（第 163 卷，第 112793 页）。
- 数据集已发布于 HuggingFace（需申请访问权限）：<https://huggingface.co/datasets/vpan1226/OPT-SND>。

## 缺陷类别

数据集磁盘上存储 **8 个原始类别**；论文评估采用合并后的 **6 类**
（`Background1 ∪ Background2 → Normal`，`Burst ∪ Burst2 → Burst`）。

| 报告类别（论文，6 类）  | 原始类别（磁盘，8 类）            |
|-------------------------|-----------------------------------|
| Normal                  | Background1 (0), Background2 (5)  |
| Burst                   | Burst (1), Burst2 (6)             |
| Pit                     | Pit (2)                           |
| Stain                   | Stain (3)                         |
| Warpage                 | Warpage (4)                       |
| Pinhole                 | Pinhole (7)                       |

## 数据采集

<p align="center">
  <img src="figures/data_acqusition.png" alt="data source" width="85%">
</p>

本项目使用的数据集采集自真实工业生产线上的汽车动力电池密封钉检测场景。

所有点云数据均通过部署于受控制造环境中的工业三维视觉系统采集，涵盖密封钉表面的多种缺陷类型，包括表面形变、对准偏差及结构异常等。图像采集与成像模块由激光轮廓仪（OPT-LPC20 激光轮廓仪）与移动机器人平台及测试样件共同构成。
激光轮廓仪采用线扫描模式，使用蓝色激光条纹（波长 405nm），轮廓采集速率为每秒 10,000 条，确保对金属反光表面具有高保真性。
扫描时，密封钉被置于精密旋转台上以保证均匀覆盖。扫描速度设置为 10mm/s，每条轮廓沿水平轴采集 3,000 个点，横向分辨率约为 0.05mm，纵向深度精度优于 0.01mm。所有扫描均经过人工核验以确保完整性并排除噪声伪影。电池安装于移动机器人平台上，经标定的三维线激光扫描摄像头将激光条纹投射至电池密封钉区域，通过平面扫描轨迹中的同步图像数据采集，获取密封钉样件的表面数据。数据标注遵循严格的质量控制流程，以确保标注精度与一致性。

## 网络架构

LGASS 基于 PyTorch 深度学习框架，在图注意力编码器-解码器骨干网络之上引入两个创新模块：

- **LGAF（局部图注意力滤波器）** — 利用法向量差异注意力检测细微表面缺陷。
- **SAG-Pooling（空间注意力图池化）** — 在下采样过程中保留几何关键点。

详见 `model/sem/GraphAttention.py` 与 `model/sem/network.py`。

## 实验结果

在密封钉数据集上的对比结果（论文表 3，**加粗 = 最优**，_下划线_ = 次优）：

| Method     | OA       | mAcc     | mIoU     | Burst    | Pit      | Stain    | Warpage  | Pinhole  |
|------------|----------|----------|----------|----------|----------|----------|----------|----------|
| PointNet   | 83.29    | 72.83    | 46.21    | 32.27    | 34.96    | 19.97    | 65.37    | 36.04    |
| PointNet++ | 89.33    | 84.65    | 62.67    | 47.91    | 59.32    | 42.37    | 81.64    | 46.53    |
| GACNet     | 96.05    | 87.84    | 69.41    | 56.02    | 67.83    | 54.54    | 86.96    | 59.94    |
| PTv1       | 98.77    | 90.96    | 72.83    | 68.54    | 62.83    | 58.37    | _89.67_  | 58.37    |
| OA-CNN     | 98.72    | 91.13    | 74.04    | 64.89    | 66.91    | 64.87    | 88.37    | _60.04_  |
| PTv3       | _99.03_  | **94.96**| _75.13_  | **77.32**| 67.98    | 60.44    | 89.17    | 56.21    |
| **LGASS**  | **99.47**| _92.37_  | **79.23**| _76.22_  | **72.17**| **71.33**| **91.61**| **64.95**|

**关于复现：** 上述数字使用的最优 checkpoint 已无法获取。重新运行公开配置可能因 `pointops` 内核不确定性以及 CUDA / PyTorch 版本差异产生 ±0.5% 范围内的偏差，详见 `configs/` 与 `scripts/reproduce_paper.sh`。

## 消融实验

**表 4 — 模块消融（mIoU）：**

| ID  | LGAF | SAG-Pooling | OA    | mIoU  |
|-----|------|-------------|-------|-------|
| I   | ✗    | ✗           | 96.92 | 72.93 |
| II  | ✓    | ✗           | 97.38 | 77.17 |
| III | ✗    | ✓           | 97.96 | 74.81 |
| IV  | ✓    | ✓           | 99.47 | 79.23 |

**表 5 — 输入类型消融：**

| Input                   | OA    | mAcc  | mIoU  |
|-------------------------|-------|-------|-------|
| Coordinates only        | 99.42 | 90.92 | 76.94 |
| Coordinates + Normals   | 99.51 | 91.43 | 78.31 |
| Normals only (default)  | 99.47 | 92.37 | 79.23 |

## 超参数

`configs/paper.yaml` 为默认配置，对应论文表 2：

| 参数             | 值            |
|------------------|---------------|
| Epoch            | 300           |
| Batch size       | 8             |
| Number of points | 16384         |
| Optimizer        | AdamW         |
| Learning rate    | 0.01          |
| LR decay         | Multi-step    |

`configs/original.yaml` 保留了最初硬编码于 `train.py` 中的参数值（AdamW，lr=0.001，batch=4，每 20 个 epoch 以 0.5x StepLR 衰减），以保持透明度。在未对两组配置分别重新训练的情况下，我们无法确认哪组参数产生了论文中的核心指标数字。

## 数据集结构

从 HuggingFace 下载后（需申请访问；见 <https://huggingface.co/datasets/vpan1226/OPT-SND>）：

```
data/sealingNail_npz/
├── train/   *.npz
└── test/    *.npz
```

类别分布（论文表 1，合并后 6 类视角）：

| Class   | Count |
|---------|-------|
| Normal  | 271   |
| Burst   | 170   |
| Pit     | 112   |
| Stain   | 102   |
| Warpage | 132   |
| Pinhole | 142   |

每个 `.npz` 文件包含单一键 `points`，形状为 `(N, 7)`，各列依次为 `[x, y, z, nx, ny, nz, label]`。

## 安装

### 推荐方式：使用 Conda

1. 创建新的 Conda 环境：

   ```bash
   conda env create -f environment.yml
   ```

2. 激活环境：

   ```bash
   conda activate sealingnail
   ```

3. 安装带 CUDA 支持的 PyTorch：

   ```bash
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
   ```

4. 安装其他依赖：

   ```bash
   conda install numpy scipy scikit-learn
   conda install -c conda-forge open3d
   conda install tqdm tensorboard h5py
   ```

5. 确保已安装 C++/编译工具（Linux）：

   ```bash
   sudo apt install g++
   sudo apt install build-essential
   sudo apt install ninja-build
   ninja --version
   ```

6. 安装自定义 pointops 库：

   ```bash
   cd lib/pointops
   python setup.py install
   ```

7. 验证安装：

   ```bash
   python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
   ```

## 使用方法

**训练：**

```bash
python train.py --config configs/paper.yaml
# or use the original train.py values:
python train.py --config configs/original.yaml
```

**单样本推理：**

```bash
python demo.py --input data/test_sample.ply --model checkpoints/best_model.pth --output results/
```

**批量推理：**

```bash
python batch_demo.py --input data/test_folder/ --model checkpoints/best_model.pth --output results/
```

**可视化样本：**

```bash
python visualize.py --input data/sealingNail_npz/test/sample_001.npz --save out.png --no-window
```

**生成逐批次 HTML 报告：**

```bash
python export_report.py --input results/ --output reports/
```

**阶段二（重新训练后）：**

```bash
python evaluate.py --checkpoint logs/paper_run/best.pth --split test --output results/eval.md
bash scripts/reproduce_paper.sh
python scripts/upload_to_hf.py --checkpoint logs/paper_run/best.pth --repo vpan1226/LGASS
```

## 引用

如果您认为我们的工作对您有帮助，请引用：

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

## 许可证

Apache 2.0 协议。详见 `LICENSE`。

## 致谢

本工作由 **OPT Machine Vision** 通过工业三维视觉与点云检测系统内部研发项目提供支持。

作者感谢所有参与汽车动力电池密封钉检测数据采集、标注与系统部署工作的合作者，以及匿名审稿人提出的建设性意见，这些意见有效提升了本工作的质量与表达清晰度。

此外，本项目受益于社区中若干优秀开源工作在点云处理与基于图学习领域的贡献，包括但不限于：

- [PointNet++](https://github.com/charlesq34/pointnet2)
- [DGCNN](https://github.com/WangYueFt/dgcnn)
- [DeepGCN](https://github.com/lightaime/deep_gcns_torch)
- [PointNeXt](https://github.com/guochengqian/PointNeXt)
- [PointOps](https://github.com/POSTECH-CVLab/pointops)

衷心感谢上述工作的作者将代码公开发布，这极大地促进了可复现性与对比评估工作。本项目部分实现参考或借鉴了先前工作，并针对工业点云分割场景进行了相应修改。
