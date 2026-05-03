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


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data_root", default="data/sealingNail_npz")
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--output", required=True, help="markdown file to write")
    p.add_argument("--num-points", type=int, default=16384)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--raw-schema", action="store_true",
                   help="evaluate on raw 8-class schema instead of paper 6-class merge")
    return p.parse_args()


def main():
    args = parse_args()

    # Heavy imports deferred (pointops_cuda only available on training boxes)
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
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"])
    else:
        model.load_state_dict(state)
    model.eval()

    all_gt = []; all_pred = []
    with torch.no_grad():
        for batch in loader:
            # Match train.py model call signature: classifier([coords, feats, offset])
            coords, feats, labels, offset = (
                batch[0].float().to(device),
                batch[1].float().to(device),
                batch[2].long().to(device),
                batch[3].to(device),
            )
            seg_pred = model([coords, feats, offset])
            seg_pred = seg_pred.contiguous().view(-1, raw_classes)
            preds_raw = seg_pred.argmax(dim=-1).cpu().numpy().reshape(-1)
            gt_raw = labels.view(-1).cpu().numpy()
            if args.raw_schema:
                all_gt.append(gt_raw); all_pred.append(preds_raw)
            else:
                all_gt.append(remap_to_merged(gt_raw))
                all_pred.append(remap_to_merged(preds_raw))

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
