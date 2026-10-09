"""
Train the bird model on the GPU.

    python 02_Train.py                      # full run (~60 epochs)
    python 02_Train.py --epochs 1 --epoch-size 64 --name smoke   # quick check

Starts from Darknet's COCO tiny-YOLOv2 and learns "bird" vs everything else on
board-shaped samples (see bird_data.py). Before training it scores two
baselines on the same validation set - the stock HWAC weights now on the board,
and the untouched COCO model - so every later number has a reference.

Writes runs/<name>/: log.csv, last.pt, best.pt (best = ap50 + top1_hit - fp30),
and train_preview.jpg (training samples with their boxes, to eyeball).
"""

import argparse
import csv
import math
import time
from pathlib import Path

import torch

import bird_data as bd
import bird_model as bm
import hwac_format as hw

HERE = Path(__file__).resolve().parent


def worker_init(_):
    import cv2
    cv2.setNumThreads(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--epoch-size", type=int, default=2400)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=3)
    ap.add_argument("--name", default="bird_v1")
    args = ap.parse_args()

    out = HERE / "runs" / args.name
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.cudnn.benchmark = True
    dev = "cuda"

    neg_train, neg_val = bd.humans()
    sources = {"birds_rf": (bd.rf_split("train"), 0.35),
               "birds_bf": (bd.birdsflying(), 0.35),
               "negatives": (neg_train, 0.30)}
    for name, (items, w) in sources.items():
        print(f"  {name:10s} {len(items):5d} images, sampled {w:.0%}")
    train_set = bd.TrainSet(sources, epoch_size=args.epoch_size)
    bd.preview_sheet(train_set, out / "train_preview.jpg")

    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, num_workers=args.workers, collate_fn=bd.collate,
        persistent_workers=True, pin_memory=True, drop_last=True, worker_init_fn=worker_init)
    val_birds = torch.utils.data.DataLoader(
        bd.EvalSet(bd.rf_split("valid") + bd.rf_split("test")), batch_size=32,
        num_workers=4, collate_fn=bd.collate)
    val_neg = torch.utils.data.DataLoader(bd.EvalSet(neg_val), batch_size=32,
                                          num_workers=4, collate_fn=bd.collate)
    print(f"  validation: {len(val_birds.dataset)} bird images, {len(val_neg.dataset)} people photos")

    stock = hw.BoardNet.from_files(HERE.parent / "sw1" / "weight_file.bin",
                                   HERE.parent / "sw1" / "padding_file.bin").to(dev).eval()
    base = bm.evaluate(lambda x: stock(x)[:, bm.HWAC_TO_25], val_birds, val_neg)
    print(f"stock HWAC on the board : {fmt(base)}")

    model = bm.load_coco(bm.TinyYoloV2()).to(dev)
    model.eval()
    coco = bm.evaluate(model, val_birds, val_neg)
    print(f"COCO start (untrained)  : {fmt(coco)}")

    head = list(model.head.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith("head")]
    opt = torch.optim.AdamW([{"params": backbone, "lr": args.lr * 0.5},
                             {"params": head, "lr": args.lr}], weight_decay=5e-4)
    base_lrs = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda")
    steps, warm = args.epochs * len(loader), 2 * len(loader)

    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "loss", "ap50", "top1_hit", "bird_top_score",
                                 "fp30", "fp50", "neg_top_score", "minutes"])
    best, step, t0 = -1e9, 0, time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for pics, boxes in loader:
            f = step / warm if step < warm else 0.02 + 0.98 * 0.5 * (
                1 + math.cos(math.pi * (step - warm) / max(1, steps - warm)))
            for g, lr in zip(opt.param_groups, base_lrs):
                g["lr"] = lr * f
            with torch.autocast("cuda", dtype=torch.float16):
                pred = model(pics.to(dev, non_blocking=True))
            loss, _ = bm.yolo_loss(pred.float(), boxes)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss)
            step += 1
        loss = total / len(loader)
        minutes = (time.time() - t0) / 60
        row = {}
        if epoch % args.eval_every == 0 or epoch == args.epochs:
            model.eval()
            row = bm.evaluate(model, val_birds, val_neg)
            score = row["ap50"] + row["top1_hit"] - row["fp30"]
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, out / "best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, out / "last.pt")
        with open(log_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, f"{loss:.3f}"] + [f"{row[k]:.3f}" if row else "" for k in
                                    ("ap50", "top1_hit", "bird_top_score", "fp30", "fp50", "neg_top_score")]
                                    + [f"{minutes:.1f}"])
        print(f"epoch {epoch:3d}  loss {loss:7.3f}  {fmt(row) if row else ''}  [{minutes:.1f} min]", flush=True)

    print(f"done. best checkpoint: {out / 'best.pt'}")


def fmt(m):
    return (f"AP50 {m['ap50']:.3f}  top-box-on-bird {m['top1_hit']:.0%}  "
            f"(score {m['bird_top_score']:.2f})  people-with-box>0.3 {m['fp30']:.0%}  "
            f">0.5 {m['fp50']:.0%}  (score {m['neg_top_score']:.2f})")


if __name__ == "__main__":
    main()
