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
