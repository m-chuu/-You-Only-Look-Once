"""
Lock-on bird tracking on the laptop: BoardNet finds the birds, bird_tracker
keeps the box on one of them.

    python 06_Track_Video.py checks/geese_test.mp4
    python 06_Track_Video.py 0                          (webcam 0)
    python 06_Track_Video.py video.mp4 --no-window      (no window, only the saved files)

Each frame is squeezed to 416x256 the way the board does and run through
BoardNet (the software model of the board) with the new files in output/.
Every box scoring above --min-score goes to the tracker, which works in the
same 416x256 picture pixels as the board.

On screen:
    red, thick    the locked bird (LOCK #id score)
    red, thin     the locked bird is not seen right now; its box keeps moving
                  at the bird's last speed for --lost-seconds (SEARCH)
    yellow        other birds, with their IDs

    mouse click   lock the bird under the mouse
    A             back to auto: lock the highest-scoring bird
    Space         pause / play
    Q or Esc      quit

For a video file it writes, next to the video:
    <name>_tracked.mp4   what the window shows
    <name>_track.csv     the locked box per frame, L R T B on the 1920x1080
                         display (the numbers the board's UART line uses)
"""

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

import bird_model as bm
import hwac_format as hw
from bird_tracker import LockOnTracker, CONFIRM_HITS

HERE = Path(__file__).resolve().parent
NMS_IOU = 0.45
MAX_VIEW_W = 1280
RED, YELLOW, WHITE, BLACK = (0, 0, 255), (0, 220, 255), (255, 255, 255), (0, 0, 0)


def detect(net, frame, dev, dtype, min_score):
    """All birds in a frame: list of (l, t, r, b, score) in 416x256 picture pixels."""
    pic = cv2.resize(frame, (hw.IMG_W, hw.IMG_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(np.ascontiguousarray(pic[:, :, ::-1].transpose(2, 0, 1))).float().div(255)[None]
    with torch.no_grad():
        head = net(x.to(dev, dtype)).float()
    boxes, scores = bm.decode(head[:, bm.HWAC_TO_25])
    b, s = boxes[0], scores[0]
    keep = s > min_score
    b, s = b[keep], s[keep]
    keep = torchvision.ops.nms(b, s, NMS_IOU)
    return [(*b[k].tolist(), float(s[k])) for k in keep.cpu()]


def label(img, text, x, y, colour, scale):
    y = max(int(18 * scale), y)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, BLACK, int(4 * scale), cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * scale, colour, max(1, int(1.5 * scale)),
                cv2.LINE_AA)


def draw(frame, tracker, frame_no, fps):
    """The frame with every bird's box; the locked bird in red."""
    img = frame.copy()
    sx, sy = img.shape[1] / hw.IMG_W, img.shape[0] / hw.IMG_H
    scale = img.shape[1] / 1280
    locked = tracker.locked()
    for bird in tracker.birds:
        if bird is locked or bird.misses or bird.hits < CONFIRM_HITS:
            continue
        l, t, r, b = (int(v * s) for v, s in zip(bird.box(), (sx, sy, sx, sy)))
        cv2.rectangle(img, (l, t), (r, b), YELLOW, max(1, int(scale)))
        label(img, f"#{bird.id} {bird.score:.2f}", l, t - 6, YELLOW, scale)
    if locked:
        l, t, r, b = (int(v * s) for v, s in zip(locked.box(), (sx, sy, sx, sy)))
        if locked.misses == 0:
            cv2.rectangle(img, (l, t), (r, b), RED, max(2, int(3 * scale)))
            label(img, f"LOCK #{locked.id} {locked.score:.2f}", l, t - 6, RED, scale)
        else:
            cv2.rectangle(img, (l, t), (r, b), RED, max(1, int(scale)))
            label(img, f"SEARCH #{locked.id} {locked.misses / fps:.1f}s", l, t - 6, RED, scale)
    mode = "AUTO" if tracker.mode == "auto" else "MANUAL (press A for auto)"
    seen = sum(1 for b in tracker.birds if b.misses == 0 and b.hits >= CONFIRM_HITS)
    label(img, f"{mode}   birds {seen}   frame {frame_no}", int(10 * scale), int(28 * scale), WHITE, scale)
    return img


def locked_row(frame_no, tracker):
    bird = tracker.locked()
    seen = sum(1 for b in tracker.birds if b.misses == 0 and b.hits >= CONFIRM_HITS)
    if bird is None:
        return [frame_no, tracker.mode, "", "none", "", "", "", "", "", seen]
    sx, sy = hw.DISPLAY_W / hw.IMG_W, hw.DISPLAY_H / hw.IMG_H
    l, t, r, b = bird.box()
    state = "locked" if bird.misses == 0 else "searching"
    return [frame_no, tracker.mode, bird.id, state, f"{bird.score:.2f}",
            round(l * sx), round(r * sx), round(t * sy), round(b * sy), seen]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="video file, or a webcam number such as 0")
    ap.add_argument("--weights", type=Path, default=HERE / "output" / "weight_file.bin")
    ap.add_argument("--padding", type=Path, default=HERE / "output" / "padding_file.bin")
    ap.add_argument("--min-score", type=float, default=0.3, help="boxes below this are ignored")
    ap.add_argument("--lost-seconds", type=float, default=1.0,
                    help="how long to search for a lost locked bird before locking another")
    ap.add_argument("--no-window", action="store_true", help="do not show a window, only save the files")
    args = ap.parse_args()

    webcam = args.video.isdigit()
    cap = cv2.VideoCapture(int(args.video) if webcam else args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    net = hw.BoardNet.from_files(args.weights, args.padding).to(dev).eval()
    tracker = LockOnTracker(lost_frames=round(args.lost_seconds * fps))

    writer = rows = None
    if not webcam:
        video = Path(args.video)
        out_video = video.with_name(video.stem + "_tracked.mp4")
        out_csv = video.with_name(video.stem + "_track.csv")
        writer = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        rows = []

    view = min(1.0, MAX_VIEW_W / W)
    clicks = []
    if not args.no_window:
        win = "Lock-on bird tracking"
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(win, lambda e, x, y, *_: clicks.append((x, y)) if e == cv2.EVENT_LBUTTONDOWN else None)

    frame_no, paused, shown, last_target, busy = 0, False, None, None, 0.0
    while True:
        tick = time.perf_counter()
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            t0 = time.perf_counter()
            tracker.step(detect(net, frame, dev, dtype, args.min_score))
            busy += time.perf_counter() - t0
            if tracker.target != last_target:
                print(f"frame {frame_no}: lock {'#' + str(last_target) if last_target else 'nobody'} -> "
                      f"{'#' + str(tracker.target) if tracker.target else 'nobody'}")
                last_target = tracker.target
            shown = draw(frame, tracker, frame_no, fps)
            if writer:
                writer.write(shown)
                rows.append(locked_row(frame_no, tracker))
        if args.no_window:
            continue

        while clicks:
            x, y = clicks.pop(0)
            px, py = x / view * hw.IMG_W / W, y / view * hw.IMG_H / H
            bird = tracker.pick(px, py)
            print(f"frame {frame_no}: click -> " + (f"lock #{bird.id}" if bird else "no bird there"))
            last_target = tracker.target
            shown = draw(frame, tracker, frame_no, fps)
        cv2.imshow(win, cv2.resize(shown, None, fx=view, fy=view) if view < 1 else shown)
        wait = 30 if paused or webcam else max(1, int(1000 / fps - (time.perf_counter() - tick) * 1000))
        key = cv2.waitKey(wait) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(" "):
            paused = not paused
        if key in (ord("a"), ord("A")):
            tracker.auto()
            print(f"frame {frame_no}: auto -> lock #{tracker.target}")
            last_target = tracker.target
            shown = draw(frame, tracker, frame_no, fps)

    cap.release()
    cv2.destroyAllWindows()
    print(f"{frame_no} frames, detect + track {frame_no / max(busy, 1e-9):.0f} fps on {dev}")
    if writer:
        writer.release()
        with open(out_csv, "w", newline="") as fh:
            out = csv.writer(fh)
            out.writerow(["frame", "mode", "bird", "state", "score", "L", "R", "T", "B", "birds_seen"])
            out.writerows(rows)
        print(f"saved {out_video}\n      {out_csv}")


if __name__ == "__main__":
    main()
