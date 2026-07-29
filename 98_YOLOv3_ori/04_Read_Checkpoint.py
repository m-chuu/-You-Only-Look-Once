"""
Print the validation metrics stored inside a YOLO .pt checkpoint.

Ultralytics embeds the best epoch's validation scores into every best.pt it
writes, so precision / recall / mAP can be recovered from the weights alone -
no re-run, no dataset, and no surviving runs/ folder needed. Useful when the
training output directory (results.csv, results.png) was deleted or never kept.

Usage
-----
    python 04_Read_Checkpoint.py CHECKPOINT [CHECKPOINT ...] [--history]

Example
-------
    python 04_Read_Checkpoint.py \
        BackUp-BestPtFile/Ori-best-0724.pt \
        BackUp-BestPtFile/Ori-best-0727.pt
"""

import argparse
import os

import torch

# -------------------------------------------------
# Keys worth showing, in the order Ultralytics reports them
# -------------------------------------------------
METRIC_ORDER = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "fitness",
]


def load_checkpoint(path):
    """Load a .pt as a plain dict.

    weights_only=False is required: the checkpoint pickles Ultralytics model
    objects, and torch 2.x defaults to weights_only=True, which refuses them.
    Only load checkpoints you trust.
    """
    return torch.load(path, map_location="cpu", weights_only=False)


def print_metrics(path, ckpt):
    """One block per checkpoint, matching the layout of Ultralytics' own summary."""
    date = (ckpt.get("date") or "unknown")[:10]
    version = ckpt.get("version", "unknown")
    label = os.path.basename(path)

    print(f"--- {label} --- (trained {date}, ultralytics {version})")

    metrics = ckpt.get("train_metrics")
    if not metrics:
        # Base weights and mid-training last.pt files carry no validation block.
        print("   no train_metrics in this checkpoint")
        return

    # Known keys first in canonical order, then anything unexpected.
    keys = [k for k in METRIC_ORDER if k in metrics]
    keys += [k for k in metrics if k not in METRIC_ORDER]
    for k in keys:
        print(f"   {k:24s} {metrics[k]}")


def print_history(ckpt, key="metrics/mAP50-95(B)"):
    """Per-epoch curve - the same series Ultralytics writes to results.csv."""
    results = ckpt.get("train_results") or {}
    series = results.get(key)
    if not series:
        print(f"   no per-epoch history for {key}")
        return

    best = max(series)
    print(f"   {key} over {len(series)} epochs")
    print(f"   best {best:.5f} at epoch {series.index(best) + 1}, final {series[-1]:.5f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+", help="paths to .pt files")
    ap.add_argument("--history", action="store_true",
                    help="also summarise the per-epoch mAP50-95 curve")
    args = ap.parse_args()

    for path in args.checkpoints:
        ckpt = load_checkpoint(path)
        print_metrics(path, ckpt)
        if args.history:
            print_history(ckpt)


if __name__ == "__main__":
    main()
