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
