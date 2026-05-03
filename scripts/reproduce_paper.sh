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
    --checkpoint "$OUTPUT_DIR/best_model.pth" \
    --data_root data/sealingNail_npz \
    --split test \
    --output "$OUTPUT_DIR/eval.md"

echo "Done. Paper-shaped metrics written to $OUTPUT_DIR/eval.md"
