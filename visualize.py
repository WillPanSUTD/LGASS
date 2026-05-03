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
