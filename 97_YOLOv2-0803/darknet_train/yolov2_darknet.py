"""
Tiny-YOLOv2 read straight from Darknet's .cfg and .weights, in pure NumPy.

Shared by the numbered scripts here, which cannot import one another because
their names begin with a digit.

Requires numpy and nothing else. The forward pass exists only so the exported
weights can be checked against darknet's own output; it is not meant to be
fast.

Batch-norm is folded into the preceding convolution at load time. Darknet
normalises with sqrt(var) + 1e-6, not the sqrt(var + eps) that most frameworks
use, so the fold reproduces the C code exactly rather than approximately.
"""

import configparser
import io
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import as_strided

DARKNET_BN_EPS = 1e-6

# Darknet's stride-1 maxpool reads one row and column past the edge and treats
# the out-of-range values as -infinity. A large finite sentinel is equivalent
# for real activations and avoids inf arithmetic.
NEG_SENTINEL = np.float32(-1e30)


# ---------------------------------------------------------------- cfg parsing

def parse_cfg(path):
    """Darknet .cfg -> list of (section_name, {key: value}).

    configparser does the value parsing but cannot read the file as-is:
    Darknet repeats section names, which configparser rejects, so each one is
    given a unique suffix first.
    """
    text = Path(path).read_text()

    counts = {}
    numbered = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1]
            counts[name] = counts.get(name, 0) + 1
            numbered.append(f"[{name}_{counts[name]}]")
        else:
            numbered.append(line)

    parser = configparser.ConfigParser(inline_comment_prefixes=("#",))
    parser.read_file(io.StringIO("\n".join(numbered)))

    return [(section.rsplit("_", 1)[0], dict(parser.items(section)))
            for section in parser.sections()]


def describe_network(blocks):
    """Pull the pieces of the cfg the rest of this module needs."""
    net = blocks[0][1]
    region = next(cfg for name, cfg in blocks if name == "region")

    return {
        "width": int(net["width"]),
        "height": int(net["height"]),
        "channels": int(net["channels"]),
        "num_anchors": int(region["num"]),
        "num_classes": int(region["classes"]),
        "coords": int(region["coords"]),
        "softmax": bool(int(region.get("softmax", 0))),
        "anchors": [float(v) for v in region["anchors"].split(",")],
    }


def layer_plan(blocks):
    """Ordered layer descriptions, with convolution shapes resolved."""
    spec = describe_network(blocks)
    in_ch = spec["channels"]
    plan = []

    for name, cfg in blocks:
        if name == "convolutional":
            out_ch = int(cfg["filters"])
            size = int(cfg["size"])
            plan.append({
                "type": "conv",
                "in_ch": in_ch,
                "out_ch": out_ch,
                "size": size,
                "stride": int(cfg["stride"]),
                "pad": (size - 1) // 2 if int(cfg.get("pad", 0)) else 0,
                "batch_normalize": bool(int(cfg.get("batch_normalize", 0))),
                "activation": cfg["activation"],
            })
            in_ch = out_ch
        elif name == "maxpool":
            plan.append({
                "type": "maxpool",
                "size": int(cfg["size"]),
                "stride": int(cfg["stride"]),
            })
        elif name in ("net", "region"):
            continue
        else:
            raise ValueError(f"unsupported layer type: {name}")

    return plan


# ------------------------------------------------------------ weight loading

def read_weights_file(path):
    """Return (header dict, float32 payload) for a Darknet .weights file."""
    path = Path(path)
    with open(path, "rb") as fh:
        major, minor, revision = (int(v) for v in np.fromfile(fh, dtype=np.int32, count=3))
        # Darknet switched the "images seen" counter to 64-bit at version 0.2.
        if (major * 10 + minor) >= 2:
            seen = int(np.fromfile(fh, dtype=np.int64, count=1)[0])
            header_len = 20
        else:
            seen = int(np.fromfile(fh, dtype=np.int32, count=1)[0])
            header_len = 16
        payload = np.fromfile(fh, dtype=np.float32)

    expected = (path.stat().st_size - header_len) // 4
    if payload.size != expected:
        raise ValueError(f"read {payload.size} floats but file holds {expected}")

    header = {"major": major, "minor": minor, "revision": revision,
              "seen": seen, "header_len": header_len}
    return header, payload


def load_folded_weights(plan, weights_path, verbose=True):
    """Load every convolution, folding batch-norm into weight and bias.

    Darknet stores, per convolution: bias, then (when batch-normalised) scale,
    running mean and running variance, then the kernel as [out, in, kh, kw]
    row-major - the same layout PyTorch and the sibling YOLOv3 .bin use, so
    nothing is transposed anywhere in this pipeline.
    """
    header, payload = read_weights_file(weights_path)

    if verbose:
        print(f"{Path(weights_path).name}: v{header['major']}.{header['minor']}."
              f"{header['revision']}, header {header['header_len']} B, "
              f"seen {header['seen']:,}, {payload.size:,} float32 parameters")

    cursor = 0

    def take(count):
        nonlocal cursor
        chunk = payload[cursor:cursor + count]
        if chunk.size != count:
            raise ValueError(f"weights exhausted: wanted {count}, got {chunk.size}")
        cursor += count
        return chunk

    conv_index = 0
    for layer in plan:
        if layer["type"] != "conv":
            continue
        conv_index += 1

        out_ch, in_ch, size = layer["out_ch"], layer["in_ch"], layer["size"]
        bias = take(out_ch).astype(np.float64)

        if layer["batch_normalize"]:
            scale = take(out_ch).astype(np.float64)
            running_mean = take(out_ch).astype(np.float64)
            running_var = take(out_ch).astype(np.float64)

        weight = take(out_ch * in_ch * size * size).astype(np.float64)
        weight = weight.reshape(out_ch, in_ch, size, size)

        if layer["batch_normalize"]:
            factor = scale / (np.sqrt(running_var) + DARKNET_BN_EPS)
            weight = weight * factor[:, None, None, None]
            bias = bias - running_mean * factor

        layer["name"] = f"conv_{conv_index}"
        layer["weight"] = np.ascontiguousarray(weight, dtype=np.float32)
        layer["bias"] = np.ascontiguousarray(bias, dtype=np.float32)

    if cursor != payload.size:
        raise ValueError(f"{payload.size - cursor:,} unread floats remain - "
                         f"the cfg does not describe this weights file")

    if verbose:
        print(f"consumed all {cursor:,} parameters across {conv_index} convolutions")

    return plan


def conv_tensors(plan):
    """(role, name, array) per tensor, in execution order - weight then bias.

    Mirrors what 98_YOLOv3_ori/06_Export_Weights_Bin.py collects by walking
    ONNX Conv nodes, so both projects emit the same kind of manifest.
    """
    tensors = []
    for layer in plan:
        if layer["type"] != "conv":
            continue
        tensors.append(("weight", f"{layer['name']}.weight", layer["weight"]))
        tensors.append(("bias", f"{layer['name']}.bias", layer["bias"]))
    return tensors


# ------------------------------------------------------------- forward pass

def conv2d(x, weight, bias, stride=1, pad=0):
    """Convolution over (C, H, W) via im2col, so the matmul goes through BLAS."""
    if pad:
        x = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    x = np.ascontiguousarray(x, dtype=np.float32)

    channels, height, width = x.shape
    filters, _, kh, kw = weight.shape
    out_h = (height - kh) // stride + 1
    out_w = (width - kw) // stride + 1

    sc, sh, sw = x.strides
    patches = as_strided(x,
                         shape=(channels, kh, kw, out_h, out_w),
                         strides=(sc, sh, sw, sh * stride, sw * stride))
    columns = patches.reshape(channels * kh * kw, out_h * out_w)

    out = weight.reshape(filters, -1) @ columns + bias[:, None]
    return out.reshape(filters, out_h, out_w)


def maxpool2d(x, size, stride):
    """Maxpool over (C, H, W). Stride 1 pads right/bottom, as Darknet does."""
    if stride == 1:
        x = np.pad(x, ((0, 0), (0, size - 1), (0, size - 1)),
                   constant_values=NEG_SENTINEL)
    x = np.ascontiguousarray(x, dtype=np.float32)

    channels, height, width = x.shape
    out_h = (height - size) // stride + 1
    out_w = (width - size) // stride + 1

    sc, sh, sw = x.strides
    windows = as_strided(x,
                         shape=(channels, out_h, out_w, size, size),
                         strides=(sc, sh * stride, sw * stride, sh, sw))
    return windows.max(axis=(3, 4))


def forward(plan, x):
    """Run the stack over a single (3, H, W) image, returning the raw head."""
    for layer in plan:
        if layer["type"] == "conv":
            x = conv2d(x, layer["weight"], layer["bias"],
                       layer["stride"], layer["pad"])
            if layer["activation"] == "leaky":
                x = np.where(x > 0, x, np.float32(0.1) * x)
            elif layer["activation"] != "linear":
                raise ValueError(f"unsupported activation: {layer['activation']}")
        else:
            x = maxpool2d(x, layer["size"], layer["stride"])
    return x


# ------------------------------------------------------------ pre-processing

def resize_darknet(image, new_w, new_h):
    """Bilinear resize matching Darknet's resize_image (align-corners).

    Takes and returns (H, W, C) float32.
    """
    height, width = image.shape[:2]
    if (width, height) == (new_w, new_h):
        return image.astype(np.float32, copy=True)

    def axis_weights(src_len, dst_len):
        if src_len == 1 or dst_len == 1:
            return np.zeros(dst_len, dtype=np.intp), np.zeros(dst_len, dtype=np.float32)
        scale = (src_len - 1) / (dst_len - 1)
        pos = np.arange(dst_len, dtype=np.float64) * scale
        idx = np.floor(pos).astype(np.intp)
        frac = (pos - idx).astype(np.float32)
        # Darknet copies the final column/row verbatim rather than interpolating.
        idx[-1] = src_len - 1
        frac[-1] = 0.0
        idx = np.minimum(idx, src_len - 1)
        return idx, frac

    col_idx, col_frac = axis_weights(width, new_w)
    col_next = np.minimum(col_idx + 1, width - 1)
    part = (image[:, col_idx] * (1 - col_frac)[None, :, None]
            + image[:, col_next] * col_frac[None, :, None])

    row_idx, row_frac = axis_weights(height, new_h)
    row_next = np.minimum(row_idx + 1, height - 1)
    out = (part[row_idx] * (1 - row_frac)[:, None, None]
           + part[row_next] * row_frac[:, None, None])

    return out.astype(np.float32)


def letterbox(image, net_w, net_h):
    """Fit the image inside net_w x net_h on a 0.5 canvas, preserving aspect."""
    height, width = image.shape[:2]
    if net_w / width < net_h / height:
        new_w, new_h = net_w, (height * net_w) // width
    else:
        new_h, new_w = net_h, (width * net_h) // height

    resized = resize_darknet(image, new_w, new_h)
    canvas = np.full((net_h, net_w, image.shape[2]), 0.5, dtype=np.float32)
    top, left = (net_h - new_h) // 2, (net_w - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = resized
    return canvas


# ------------------------------------------------------------ post-processing

def decode_region(head, spec):
    """Raw (425, 13, 13) head -> (boxes cxcywh in 0..1, objectness, class probs).

    Channels are anchor-major: anchor n occupies n*85 .. n*85+84, ordered
    tx, ty, tw, th, objectness, then the class scores.
    """
    anchors = spec["anchors"]
    num_classes = spec["num_classes"]
    n_anchors = len(anchors) // 2
    entry = spec["coords"] + 1 + num_classes

    _, grid_h, grid_w = head.shape
    pred = head.reshape(n_anchors, entry, grid_h, grid_w).astype(np.float64)

    def sigmoid(v):
        return 1.0 / (1.0 + np.exp(-v))

    tx, ty = sigmoid(pred[:, 0]), sigmoid(pred[:, 1])
    objectness = sigmoid(pred[:, 4])

    scores = pred[:, 5:]
    if spec["softmax"]:
        shifted = scores - scores.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        class_probs = exp / exp.sum(axis=1, keepdims=True)
    else:
        class_probs = sigmoid(scores)

    cols = np.arange(grid_w, dtype=np.float64).reshape(1, 1, grid_w)
    rows = np.arange(grid_h, dtype=np.float64).reshape(1, grid_h, 1)
    anchor_w = np.array(anchors[0::2], dtype=np.float64).reshape(n_anchors, 1, 1)
    anchor_h = np.array(anchors[1::2], dtype=np.float64).reshape(n_anchors, 1, 1)

    boxes = np.stack([
        (cols + tx) / grid_w,
        (rows + ty) / grid_h,
        np.exp(pred[:, 2]) * anchor_w / grid_w,
        np.exp(pred[:, 3]) * anchor_h / grid_h,
    ], axis=-1).reshape(-1, 4)

    class_probs = class_probs.transpose(0, 2, 3, 1).reshape(-1, num_classes)
    return boxes, objectness.reshape(-1), class_probs


def correct_boxes(boxes, orig_w, orig_h, net_w, net_h):
    """Undo the letterbox padding so boxes refer to the original image."""
    if net_w / orig_w < net_h / orig_h:
        new_w, new_h = net_w, (orig_h * net_w) // orig_w
    else:
        new_h, new_w = net_h, (orig_w * net_h) // orig_h

    boxes = boxes.copy()
    boxes[:, 0] = (boxes[:, 0] - (net_w - new_w) / 2.0 / net_w) / (new_w / net_w)
    boxes[:, 1] = (boxes[:, 1] - (net_h - new_h) / 2.0 / net_h) / (new_h / net_h)
    boxes[:, 2] *= net_w / new_w
    boxes[:, 3] *= net_h / new_h
    return boxes


def _iou(box, others):
    """IoU between one centre-form box and an array of them."""
    ax0, ax1 = box[0] - box[2] / 2, box[0] + box[2] / 2
    ay0, ay1 = box[1] - box[3] / 2, box[1] + box[3] / 2
    bx0, bx1 = others[:, 0] - others[:, 2] / 2, others[:, 0] + others[:, 2] / 2
    by0, by1 = others[:, 1] - others[:, 3] / 2, others[:, 1] + others[:, 3] / 2

    inter = (np.clip(np.minimum(ax1, bx1) - np.maximum(ax0, bx0), 0, None)
             * np.clip(np.minimum(ay1, by1) - np.maximum(ay0, by0), 0, None))
    union = box[2] * box[3] + others[:, 2] * others[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def detections(boxes, objectness, class_probs, thresh=0.5, nms_thresh=0.45):
    """Threshold then per-class NMS, following Darknet's do_nms_sort.

    Returns dicts of class_id, prob and box, sorted by descending probability.
    """
    scores = class_probs * objectness[:, None]
    scores[objectness <= thresh] = 0.0
    scores[scores <= thresh] = 0.0

    kept = []
    for class_id in range(scores.shape[1]):
        candidates = np.flatnonzero(scores[:, class_id] > 0)
        if candidates.size == 0:
            continue
        candidates = candidates[np.argsort(-scores[candidates, class_id])]

        suppressed = np.zeros(candidates.size, dtype=bool)
        for i, index in enumerate(candidates):
            if suppressed[i]:
                continue
            kept.append((class_id, float(scores[index, class_id]), index))
            if i + 1 < candidates.size:
                rest = candidates[i + 1:]
                suppressed[i + 1:] |= _iou(boxes[index], boxes[rest]) > nms_thresh

    kept.sort(key=lambda item: -item[1])
    return [{"class_id": c, "prob": p, "box": boxes[i].tolist()} for c, p, i in kept]


def load_names(path):
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
