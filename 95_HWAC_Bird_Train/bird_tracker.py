"""
Lock-on tracking: keep the box on ONE bird, even when other birds appear.

Every frame the detector gives a list of boxes (L, T, R, B, score). This
module gives each bird an ID that stays the same from frame to frame, and
picks one of them as the locked bird:

    auto    the highest-scoring bird is locked once, then kept - a bird that
            appears later does not take the lock, even if it scores higher.
    manual  the user clicked a bird; that bird is kept.

When the locked bird is not detected (hidden, flown off, or missed by the
detector) its box keeps moving at the bird's last speed ("searching"). If it
is not found again within `lost_frames`, the lock goes to the
highest-scoring bird still in view, and manual mode falls back to auto.

Plain Python, no weights, no pictures needed: only boxes in, boxes out, so
it can be ported to C to run on the board right after the AI core.

How a box is matched to a bird:
    1. parts    a box mostly inside a higher-scoring box (a wing, a head)
                is part of that bird and is dropped
    2. predict  every bird moves by its speed (pixels per frame)
    3. match    pairs (bird, box) are taken best overlap (IoU) first;
                a pair needs IoU >= MIN_IOU
    4. cross    if the box also covers another bird that got no box of its
                own, two birds are crossing and the detector sees them as
                one. Neither bird follows that box: both keep moving at
                their own speed until they come apart again.
    5. update   the bird moves part of the way to its box (ALPHA) and its
                speed is corrected (BETA) - an alpha-beta filter, a simple
                cousin of the Kalman filter
    6. new      a box left over with score >= NEW_SCORE, away from every
                bird, starts a new bird (weaker boxes only keep birds alive)
    7. delete   a bird not matched for more than lost_frames is dropped
"""

MIN_IOU = 0.2        # overlap needed to say "this box is that bird"
INSIDE = 0.6         # share of a box inside another box that makes it a part of it
NEW_SCORE = 0.5      # a box this sure may start a new bird
CONFIRM_HITS = 2     # frames a bird must be seen before auto can lock it
ALPHA = 0.7          # how far the box moves to the new detection (0..1)
BETA = 0.15          # how fast the speed adapts (0..1)
SIZE_ALPHA = 0.5     # how fast width/height adapt (0..1)
COAST_DAMP = 0.97    # speed kept per frame while a bird is not seen


def area(a):
    return max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])


def overlap(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a, b):
    """a, b: (l, t, r, b). Overlap area / union area, 0..1."""
    inter = overlap(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def inside(a, b):
    """Share of box a that lies inside box b, 0..1."""
    return overlap(a, b) / area(a) if area(a) > 0 else 0.0


def drop_parts(detections):
    """Best first; a box mostly inside a box already kept is dropped."""
    kept = []
    for d in sorted(detections, key=lambda d: -d[4]):
        if all(inside(d[:4], k[:4]) < INSIDE for k in kept):
            kept.append(d)
    return kept


class Bird:
    """One tracked bird: centre, size and speed in picture pixels."""

    def __init__(self, bird_id, box, score):
        l, t, r, b = box
        self.id = bird_id
        self.cx, self.cy = (l + r) / 2, (t + b) / 2
        self.w, self.h = r - l, b - t
        self.vx = self.vy = 0.0
        self.score = score
        self.hits = 1        # frames this bird was detected
        self.misses = 0      # frames since it was last detected

    def box(self):
        return (self.cx - self.w / 2, self.cy - self.h / 2,
                self.cx + self.w / 2, self.cy + self.h / 2)

    def area(self):
        return self.w * self.h

    def predict(self):
        if self.misses:
            self.vx *= COAST_DAMP
            self.vy *= COAST_DAMP
        self.cx += self.vx
        self.cy += self.vy

    def update(self, box, score):
        l, t, r, b = box
        ex, ey = (l + r) / 2 - self.cx, (t + b) / 2 - self.cy
        self.cx += ALPHA * ex
        self.cy += ALPHA * ey
        self.vx += BETA * ex
        self.vy += BETA * ey
        self.w += SIZE_ALPHA * ((r - l) - self.w)
        self.h += SIZE_ALPHA * ((b - t) - self.h)
        self.score = score
        self.hits += 1
        self.misses = 0


class LockOnTracker:
    def __init__(self, lost_frames=24):
        self.lost_frames = lost_frames
        self.birds = []
        self.next_id = 1
        self.mode = "auto"
        self.target = None   # ID of the locked bird, or None

    # ---------------------------------------------------------- per frame

    def step(self, detections):
        """detections: list of (l, t, r, b, score). Returns the locked Bird or None."""
        dets = drop_parts(detections)
        for bird in self.birds:
            bird.predict()
        preds = [bird.box() for bird in self.birds]

        pairs = []
        for i, pred in enumerate(preds):
            for j, d in enumerate(dets):
                o = iou(pred, d[:4])
                if o >= MIN_IOU:
                    pairs.append((o, i, j))
        pairs.sort(reverse=True)
        match, used_det = {}, set()
        for _, i, j in pairs:
            if i not in match and j not in used_det:
                match[i] = j
                used_det.add(j)

        for i, bird in enumerate(self.birds):
            j = match.get(i)
            crossing = j is not None and any(
                k != i and k not in match and self.birds[k].hits >= CONFIRM_HITS
                and iou(preds[k], dets[j][:4]) >= MIN_IOU for k in range(len(self.birds)))
            if j is None or crossing:
                bird.misses += 1
            else:
                bird.update(dets[j][:4], dets[j][4])

        for j, d in enumerate(dets):
            if j in used_det or d[4] < NEW_SCORE:
                continue
            if all(iou(b.box(), d[:4]) < MIN_IOU and inside(d[:4], b.box()) < INSIDE for b in self.birds):
                self.birds.append(Bird(self.next_id, d[:4], d[4]))
                self.next_id += 1
        self.birds = [b for b in self.birds if b.misses <= self.lost_frames]

        if self.locked() is None:
            if self.mode == "manual":
                self.mode = "auto"
            self.target = self.best_id()
        return self.locked()

    # ------------------------------------------------------------- lock

    def locked(self):
        for bird in self.birds:
            if bird.id == self.target:
                return bird
        return None

    def best_id(self):
        """Highest-scoring bird seen now and confirmed; ties go to the bigger box."""
        ready = [b for b in self.birds if b.misses == 0 and b.hits >= CONFIRM_HITS]
        if not ready:
            return None
        return max(ready, key=lambda b: (round(b.score, 2), b.area())).id

    def pick(self, x, y):
        """User clicked at (x, y) in picture pixels: lock the bird under it."""
        under = [b for b in self.birds
                 if b.box()[0] <= x <= b.box()[2] and b.box()[1] <= y <= b.box()[3]]
        if not under:
            return None
        bird = min(under, key=lambda b: b.area())   # the smaller box when boxes overlap
        self.mode, self.target = "manual", bird.id
        return bird

    def auto(self):
        """Back to auto: lock the highest-scoring bird now."""
        self.mode = "auto"
        self.target = self.best_id()
