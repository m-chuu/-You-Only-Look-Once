"""
Turn a trained checkpoint into the two files the board loads, and prove them.

    python 03_Export.py                          # runs/bird_v1/best.pt
    python 03_Export.py --ckpt runs/bird_v1/last.pt

Writes output/:
  weight_file.bin     22,542,352 bytes - upload as weight_file.bin
  padding_file.bin       346,112 bytes - upload as padding_file.bin
  preview.jpg         validation pictures: stock box (blue) vs new box (red)
  test_pictures/*.bin board-format pictures for UPLOAD IMAGE TO RAM, with the
                      box the board should report listed in report.txt
  report.txt          the checks below, checksums and upload steps

Checks - the files are only written if 1 and 2 pass:
  1. the packed files, read back through BoardNet (the software model of the
     board), pick the same top box as the trained network on every picture
  2. the same holds at half precision, which is how the board computes
  3. headline numbers vs the stock weights on the same validation pictures
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch

import bird_data as bd
import bird_model as bm
import hwac_format as hw

HERE = Path(__file__).resolve().parent


def predictions(net, loader, dtype=torch.float32):
    """Top box per picture from a 500-channel board network."""
    out = []
    with torch.no_grad():
        for pics, _ in loader:
            head = net(pics.cuda().to(dtype)).float().cpu()
            out += [hw.top_box(h) for h in head]
    return out


def agree(a, b, px=6.0):
    same_cell = a["cell"] == b["cell"] and a["anchor"] == b["anchor"]
    close = max(abs(x - y) for x, y in zip(a["picture"], b["picture"])) <= px
    return (same_cell or close) and abs(a["objectness"] - b["objectness"]) < 0.05


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(HERE / "runs" / "bird_v1" / "best.pt"))
    ap.add_argument("--out", default=str(HERE / "output"))
    args = ap.parse_args()
    out = Path(args.out)
    staging = out / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model = bm.TinyYoloV2()
    model.load_state_dict(ckpt["model"])
    model.cuda().eval()
    print(f"checkpoint {args.ckpt}  (epoch {ckpt['epoch']})")

    # ---- pack
    layers = model.hwac_layers()
    (staging / "weight_file.bin").write_bytes(hw.weight_file_bytes(layers))
    shaper = hw.BoardNet(layers, np.zeros((64, 52, 52), np.float32)).cuda().eval()
    with torch.no_grad():
        g = torch.full((1, 3, hw.FRAME, hw.FRAME), hw.GREY, device="cuda")
        for i in range(3):
            g = shaper.layer(g, i)
    (staging / "padding_file.bin").write_bytes(hw.padding_file_bytes(g[0].cpu().numpy()))

    # ---- check against the trained network
    _, neg_val = bd.humans()
    birds = torch.utils.data.DataLoader(bd.EvalSet(bd.rf_split("valid") + bd.rf_split("test")),
                                        batch_size=32, collate_fn=bd.collate)
    people = torch.utils.data.DataLoader(bd.EvalSet(neg_val), batch_size=32, collate_fn=bd.collate)
    board = hw.BoardNet.from_files(staging / "weight_file.bin", staging / "padding_file.bin").cuda().eval()

    class Reference(torch.nn.Module):          # trained net, widened to the board's 500 channels
        def forward(self, x):
            out25 = model(x)
            head = torch.zeros(x.shape[0], 500, hw.GRID, hw.GRID, device=x.device)
            head[:, bm.HWAC_TO_25] = out25
            return head

    ref = Reference()
    lines = []
    ok = True
    for name, loader in (("bird pictures", birds), ("people photos", people)):
        want = predictions(ref, loader)
        for label, dtype in (("fp32", torch.float32), ("fp16", torch.float16)):
            got = predictions(board, loader, dtype)
            rate = np.mean([agree(a, b) for a, b in zip(want, got)])
            lines.append(f"files vs trained network, {name}, {label}: same top box on {rate:.1%}")
            ok &= rate >= 0.97
    for line in lines:
        print(line)
    if not ok:
        print("FAIL - the packed files do not reproduce the trained network; nothing written")
        raise SystemExit(1)

    # ---- headline numbers, stock vs new (both as the board computes them)
    stock = hw.BoardNet.from_files(HERE.parent / "sw1" / "weight_file.bin",
                                   HERE.parent / "sw1" / "padding_file.bin").cuda().eval()
    table = []
    for label, net, dtype in (("stock HWAC", stock, torch.float32), ("new bird model", board, torch.float16)):
        m = bm.evaluate(lambda x: net(x.to(dtype))[:, bm.HWAC_TO_25].float(), birds, people)
        table.append(f"{label:15s} top box on a bird {m['top1_hit']:.0%} (median score {m['bird_top_score']:.2f})"
                     f" | people photos with a box >0.3: {m['fp30']:.0%}, >0.5: {m['fp50']:.0%}"
                     f" | AP50 {m['ap50']:.3f}")
    for line in table:
        print(line)

    # ---- publish
    out.mkdir(exist_ok=True)
    for f in ("weight_file.bin", "padding_file.bin"):
        shutil.move(str(staging / f), str(out / f))
    shutil.rmtree(staging, ignore_errors=True)
    tests = test_pictures(birds.dataset, board, out / "test_pictures")
    preview(birds.dataset, people.dataset, stock, board, out / "preview.jpg")
    write_report(out, args.ckpt, ckpt["epoch"], lines, table, tests)
    print(f"\nwrote {out / 'weight_file.bin'}\n      {out / 'padding_file.bin'}\n      {out / 'report.txt'}")


def test_pictures(dataset, board, folder, picks=(0, 7, 15, 30)):
    folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for k, i in enumerate(picks, 1):
        pic, _ = dataset[i]
        buf = np.zeros((hw.IMG_H, hw.IMG_W, 4), np.float16)
        buf[:, :, :3] = pic.numpy().transpose(1, 2, 0)
        name = f"B{k}.bin"
        buf.tofile(folder / name)
        with torch.no_grad():
            box = hw.top_box(board(pic[None].cuda().half()).float()[0].cpu())
        l, r, t, b = box["display"]
        rows.append(f"  {name}  ({Path(dataset.items[i][0]).name[:40]})  expect L{l} R{r} T{t} B{b}"
                    f"  score {box['objectness']:.2f}")
        cv2.imwrite(str(folder / f"B{k}_check.png"),
                    (pic.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8))
    return rows


def preview(birds, people, stock, board, path, n_birds=8, n_people=4):
    tiles = []
    idx = [(birds, i) for i in np.linspace(0, len(birds) - 1, n_birds).astype(int)]
    idx += [(people, i) for i in range(n_people)]
    for ds, i in idx:
        pic, _ = ds[i]
        img = (pic.numpy().transpose(1, 2, 0)[:, :, ::-1] * 255).astype(np.uint8).copy()
        with torch.no_grad():
            for net, colour, dtype in ((stock, (255, 128, 0), torch.float32),
                                       (board, (0, 0, 255), torch.float16)):
                box = hw.top_box(net(pic[None].cuda().to(dtype)).float()[0].cpu())
                l, r, t, b = (int(v) for v in box["picture"])
                cv2.rectangle(img, (l, t), (r, b), colour, 2)
                cv2.putText(img, f"{box['objectness']:.2f}", (max(0, l), max(10, t - 3)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, colour, 1, cv2.LINE_AA)
        tiles.append(cv2.copyMakeBorder(img, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=(40, 40, 40)))
    rows = [np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(len(tiles) // 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def write_report(out, ckpt, epoch, checks, table, tests):
    w, p = out / "weight_file.bin", out / "padding_file.bin"
    text = [
        "New bird weights for the LogikHaus SoC (HWAC tiny-YOLOv2 core)",
        f"checkpoint: {ckpt} (epoch {epoch})", "",
        "Checks", *[f"  {c}" for c in checks], "",
        "Validation (122 bird pictures, 300 people photos never used in training)",
        *[f"  {t}" for t in table], "",
        "Files",
        f"  weight_file.bin   {w.stat().st_size:,} bytes  sha256 {hw.sha256(w)}",
        f"  padding_file.bin  {p.stat().st_size:,} bytes  sha256 {hw.sha256(p)}", "",
        "Upload (lh-dashboard-P4.5, AI Control tab)",
        "  1. Press Clear (AI off).",
        "  2. File: weight_file.bin -> Browse to output\\weight_file.bin -> UPLOAD TO SD  (~33 min at 115200)",
        "  3. File: padding_file.bin -> Browse to output\\padding_file.bin -> UPLOAD TO SD (~30 s)",
        "  4. LOAD FROM SD TO RAM, wait for SUCCESS, then RUN AI DETECTION.",
        "  To go back to the stock model, upload sw1\\weight_file.bin and sw1\\padding_file.bin the same way.", "",
        "Test pictures (UPLOAD IMAGE TO RAM; boxes the board should report if it reads uploads)",
        *tests,
    ]
    (out / "report.txt").write_text("\n".join(text) + "\n")


if __name__ == "__main__":
    main()
