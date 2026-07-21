"""End-to-end deep-learning part detector for rasterized CAD sheets.

A small U-Net segments part-outline pixels from clutter (dimension lines,
text, frames) on rasterized sheet tiles; connected components on the
predicted mask yield part bounding boxes. Pixel-wise evidence fits thin
wireframe geometry far better than center-keypoint detection, whose target
falls on empty space inside large outlines.

Training data is fully procedural: reference part outlines are stamped onto
synthetic tiles together with clutter, so no manual annotation is needed.
A short fine-tune on real sheet tiles with pseudo-labels from the heuristic
pipeline closes the synthetic-to-real domain gap.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from cad_cv_demo.shape_match_diagnostics import Point, Primitive, components  # bootstraps vendor path
from cad_cv_demo.dl_raster import draw_polylines, rasterize_scene, transform_polylines  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402


class ProgressPrinter:
    """Dependency-free step-progress logger for training loops."""

    def __init__(self, prefix: str, total: int, every: int | None = None, min_interval: float = 10.0) -> None:
        self.prefix = prefix
        self.total = total
        self.every = every or max(1, total // 100)
        self.min_interval = min_interval
        self.started = time.time()
        self._last = 0.0

    def update(self, step: int, history: list[float]) -> None:
        done = step + 1
        now = time.time()
        if done < self.total and done % self.every != 0 and now - self._last < self.min_interval:
            return
        self._last = now
        elapsed = now - self.started
        rate = done / max(elapsed, 1e-9)
        eta = (self.total - done) / max(rate, 1e-9)
        window = history[-50:]
        avg = sum(window) / max(len(window), 1)
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        print(f"{self.prefix} step {done}/{self.total} loss={avg:.4f} (avg{len(window)}) {rate:.1f} it/s eta {eta_str}", flush=True)


TILE = 1024
OVERLAP = 192
OUT_STRIDE = 2
OUT_SIZE = TILE // OUT_STRIDE
PIX_PER_UNIT = 3.0


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _conv_block(channels_in: int, channels_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class UNetSegmenter(nn.Module):
    """1024² tile -> stride-2 foreground mask for part-outline pixels."""

    def __init__(self) -> None:
        super().__init__()
        self.enc1 = nn.Sequential(_conv_block(1, 16), _conv_block(16, 16))      # 1024
        self.enc2 = nn.Sequential(_conv_block(16, 32), _conv_block(32, 32))     # 512
        self.enc3 = nn.Sequential(_conv_block(32, 64), _conv_block(64, 64))     # 256
        self.enc4 = nn.Sequential(_conv_block(64, 128), _conv_block(128, 128))  # 128
        self.mid = nn.Sequential(_conv_block(128, 256), _conv_block(256, 256))  # 64
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = nn.Sequential(_conv_block(256, 128), _conv_block(128, 128))
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(_conv_block(128, 64), _conv_block(64, 64))
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(_conv_block(64, 32), _conv_block(32, 32))
        self.head = nn.Conv2d(32, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        mid = self.mid(self.pool(e4))
        d3 = self.dec3(torch.cat([self.up3(mid), e4], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        return self.head(d1)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def segmentation_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float = 8.0) -> torch.Tensor:
    weight = torch.full_like(target, pos_weight)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=weight, reduction="none")
    return bce.mean()


# ---------------------------------------------------------------------------
# Synthetic training tiles
# ---------------------------------------------------------------------------

def _clutter_dimension(img: np.ndarray, rng: np.random.Generator) -> None:
    x1, y1 = rng.integers(0, TILE, 2)
    angle = rng.uniform(0, math.pi)
    length = rng.uniform(40, 300)
    x2, y2 = x1 + length * math.cos(angle), y1 + length * math.sin(angle)
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, cv2.LINE_AA)
    for px, py in ((x1, y1), (x2, y2)):
        tx, ty = 8 * math.sin(angle), 8 * math.cos(angle)
        cv2.line(img, (int(px - tx), int(py - ty)), (int(px + tx), int(py + ty)), 255, 1, cv2.LINE_AA)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    for _ in range(rng.integers(2, 5)):
        ox, oy = rng.uniform(-14, 14, 2)
        cv2.line(img, (int(mx + ox), int(my + oy - 8)), (int(mx + ox + rng.uniform(3, 9)), int(my + oy - 8)), 255, 1, cv2.LINE_AA)


def _clutter_scribble(img: np.ndarray, rng: np.random.Generator) -> None:
    kind = rng.integers(0, 4)
    x, y = rng.integers(0, TILE, 2)
    if kind == 0:
        cv2.line(img, (int(x), int(y)), (int(x + rng.uniform(-60, 60)), int(y + rng.uniform(-60, 60))), 255, 1, cv2.LINE_AA)
    elif kind == 1:
        cv2.circle(img, (int(x), int(y)), int(rng.uniform(2, 9)), 255, 1, cv2.LINE_AA)
    elif kind == 2:
        cv2.ellipse(img, (int(x), int(y)), (int(rng.uniform(6, 20)), int(rng.uniform(4, 14))), float(rng.uniform(0, 180)), 0, int(rng.uniform(90, 300)), 255, 1)
    else:
        for i in range(rng.integers(2, 6)):
            cv2.rectangle(img, (int(x + i * 7), int(y)), (int(x + i * 7 + rng.uniform(3, 5)), int(y + rng.uniform(4, 7))), 255, 1)


def _clutter_frame(img: np.ndarray, rng: np.random.Generator) -> None:
    if rng.random() < 0.5:
        y = int(rng.uniform(0, TILE))
        cv2.line(img, (0, y), (TILE, int(y + rng.uniform(-8, 8))), 255, 1, cv2.LINE_AA)
    else:
        x = int(rng.uniform(0, TILE))
        cv2.line(img, (x, 0), (int(x + rng.uniform(-8, 8)), TILE), 255, 1, cv2.LINE_AA)


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def _downsample_mask(mask: np.ndarray) -> np.ndarray:
    """Max-pool a binary TILE mask to OUT_SIZE so thin lines survive downsampling."""
    return mask.reshape(OUT_SIZE, OUT_STRIDE, OUT_SIZE, OUT_STRIDE).max(axis=(1, 3)).astype(np.float32)


class SyntheticTileGenerator:
    """Procedural tiles: stamped reference outlines + drawing-style clutter."""

    def __init__(self, reference_outlines: list[list[list[Point]]], seed: int = 0) -> None:
        self.references = [lines for lines in reference_outlines if lines]
        self.rng = np.random.default_rng(seed)

    def _stamp_part(self, img: np.ndarray, part_img: np.ndarray, boxes: list[tuple[float, float, float, float]]) -> None:
        rng = self.rng
        lines = self.references[rng.integers(0, len(self.references))]
        ref_max = max(max(p[0] for ln in lines for p in ln) - min(p[0] for ln in lines for p in ln),
                      max(p[1] for ln in lines for p in ln) - min(p[1] for ln in lines for p in ln), 1e-9)
        target = float(np.clip(rng.lognormal(math.log(90), 0.7), 24, 420))
        scale = target / ref_max
        angle = float(rng.uniform(0, 360))
        aspect = float(rng.uniform(0.95, 1.05))
        placed = transform_polylines(lines, angle_deg=angle, scale=scale, scale_y=scale * aspect)
        xs = [p[0] for ln in placed for p in ln]
        ys = [p[1] for ln in placed for p in ln]
        w, h = max(xs) - min(xs), max(ys) - min(ys)
        for _ in range(8):
            cx = float(rng.uniform(w / 2 + 6, TILE - w / 2 - 6))
            cy = float(rng.uniform(h / 2 + 6, TILE - h / 2 - 6))
            candidate = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            if all(_iou(candidate, (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)) < 0.15 for b in boxes):
                break
        else:
            return
        moved = [[(p[0] + cx - (min(xs) + max(xs)) / 2, p[1] + cy - (min(ys) + max(ys)) / 2) for p in ln] for ln in placed]
        width = int(rng.integers(2, 4))
        draw_polylines(img, moved, lambda p: (int(round(p[0])), int(round(p[1]))), width)
        draw_polylines(part_img, moved, lambda p: (int(round(p[0])), int(round(p[1]))), width)
        boxes.append((cx, cy, w, h))
        if rng.random() < 0.7:
            self._attached_dimensions(img, cx, cy, w, h)

    def _attached_dimensions(self, img: np.ndarray, cx: float, cy: float, w: float, h: float) -> None:
        """Dimension lines anchored on the part bbox (negative-only ink).

        Real sheets have extension/dimension lines touching the outlines; the
        model must learn to exclude them even though they touch the part.
        """
        rng = self.rng
        for _ in range(rng.integers(1, 4)):
            horizontal = rng.random() < 0.5
            side = float(rng.choice([-1, 1]))
            if horizontal:
                ex = cx + float(rng.uniform(-0.5, 0.5)) * w
                ey = cy + side * h / 2
                length = float(rng.uniform(30, 140))
                cv2.line(img, (int(ex), int(ey)), (int(ex), int(ey + side * length)), 255, 1, cv2.LINE_AA)
                tx = ex + float(rng.uniform(-6, 6))
                cv2.line(img, (int(tx - 6), int(ey + side * length)), (int(tx + 6), int(ey + side * length)), 255, 1, cv2.LINE_AA)
                for _ in range(rng.integers(1, 4)):
                    ox = float(rng.uniform(-12, 12))
                    cv2.line(img, (int(tx + ox), int(ey + side * (length + 6))), (int(tx + ox + rng.uniform(3, 8)), int(ey + side * (length + 6))), 255, 1, cv2.LINE_AA)
            else:
                ex = cx + side * w / 2
                ey = cy + float(rng.uniform(-0.5, 0.5)) * h
                length = float(rng.uniform(30, 140))
                cv2.line(img, (int(ex), int(ey)), (int(ex + side * length), int(ey)), 255, 1, cv2.LINE_AA)
                ty = ey + float(rng.uniform(-6, 6))
                cv2.line(img, (int(ex + side * length), int(ty - 6)), (int(ex + side * length), int(ty + 6)), 255, 1, cv2.LINE_AA)
                for _ in range(rng.integers(1, 4)):
                    oy = float(rng.uniform(-12, 12))
                    cv2.line(img, (int(ex + side * (length + 6)), int(ty + oy)), (int(ex + side * (length + 6)), int(ty + oy + rng.uniform(3, 8))), 255, 1, cv2.LINE_AA)

    def compose(self) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float, float]]]:
        """Returns (tile image, part-only image, boxes)."""
        rng = self.rng
        img = np.zeros((TILE, TILE), dtype=np.uint8)
        part_img = np.zeros((TILE, TILE), dtype=np.uint8)
        boxes: list[tuple[float, float, float, float]] = []
        count = int(rng.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.28, 0.16, 0.06]))
        for _ in range(count):
            self._stamp_part(img, part_img, boxes)
        for _ in range(rng.integers(2, 11)):
            roll = rng.random()
            if roll < 0.35:
                _clutter_dimension(img, rng)
            elif roll < 0.85:
                _clutter_scribble(img, rng)
            else:
                _clutter_frame(img, rng)
        return img.astype(np.float32) / 255.0, part_img.astype(np.float32) / 255.0, boxes

    def sample(self) -> tuple[np.ndarray, np.ndarray]:
        img, part_img, _ = self.compose()
        mask = _downsample_mask((part_img > 0).astype(np.float32))
        return img[None], mask[None]


# ---------------------------------------------------------------------------
# Training loop (with batch prefetch)
# ---------------------------------------------------------------------------

def _batch_to_tensors(batch: list[tuple[np.ndarray, np.ndarray]], device: torch.device):
    images = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
    masks = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
    return images, masks


def train_steps(model: UNetSegmenter, sampler, steps: int, device: torch.device, lr: float, batch_size: int = 8, prefix: str = "detector") -> list[float]:
    """Generic loop; sampler() must return one (image, mask) tuple."""
    import threading
    from concurrent.futures import ThreadPoolExecutor

    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=lr * 0.05)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    lock = threading.Lock()
    progress = ProgressPrinter(prefix, steps)

    def make_batch():
        with lock:
            return [sampler() for _ in range(batch_size)]

    history: list[float] = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [pool.submit(make_batch) for _ in range(2)]
        for step in range(steps):
            images, masks = _batch_to_tensors(pending.pop(0).result(), device)
            pending.append(pool.submit(make_batch))
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(images)
                loss = segmentation_loss(logits, masks)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            history.append(float(loss.item()))
            progress.update(step, history)
    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def tile_origins(length: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    if length <= tile:
        return [0]
    stride = tile - overlap
    origins = list(range(0, length - tile + 1, stride))
    if origins[-1] != length - tile:
        origins.append(length - tile)
    return origins


@torch.no_grad()
def detect_tiles(model: UNetSegmenter, tiles: list[np.ndarray], device: torch.device, conf_thresh: float = 0.5, batch_size: int = 8) -> list[list[tuple[float, float, float, float, float]]]:
    """Per-tile detections as (cx, cy, w, h, conf) in tile pixels."""
    model.to(device).eval()
    output: list[list[tuple[float, float, float, float, float]]] = []
    for start in range(0, len(tiles), batch_size):
        chunk = tiles[start : start + batch_size]
        batch = torch.from_numpy(np.stack([t if t.ndim == 3 else t[None] for t in chunk])).to(device)
        probs = torch.sigmoid(model(batch)).cpu().numpy()
        for prob in probs:
            binary = (prob[0] >= conf_thresh).astype(np.uint8)
            # close 1-2px anti-aliasing gaps so fragmented outlines become whole
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
            count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            boxes: list[tuple[float, float, float, float, float]] = []
            for label in range(1, count):
                x, y, w, h, area = stats[label]
                if area < 12 or w < 4 or h < 4:
                    continue
                conf = float(prob[0][labels == label].mean())
                boxes.append(((x + w / 2) * OUT_STRIDE, (y + h / 2) * OUT_STRIDE, w * OUT_STRIDE, h * OUT_STRIDE, conf))
            output.append(boxes)
    return output


def nms(boxes: list[tuple[float, float, float, float, float]], iou_thresh: float = 0.3) -> list[tuple[float, float, float, float, float]]:
    remaining = sorted(boxes, key=lambda b: -b[4])
    kept: list[tuple[float, float, float, float, float]] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        bb = (best[0] - best[2] / 2, best[1] - best[3] / 2, best[0] + best[2] / 2, best[1] + best[3] / 2)
        remaining = [
            b for b in remaining
            if _iou(bb, (b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2)) < iou_thresh
        ]
    return kept


@torch.no_grad()
def detect_parts(
    model: UNetSegmenter,
    primitives: list[Primitive],
    device: torch.device,
    conf_thresh: float = 0.5,
    pix_per_unit: float | tuple[float, ...] = (3.0, 5.0),
) -> list[dict]:
    """Full-sheet detection; returns boxes in model coordinates with confidence.

    Multi-scale: small/thin parts only segment cleanly at higher resolution,
    large parts at lower; boxes from all scales are merged with one NMS pass.
    """
    ppus = (float(pix_per_unit),) if isinstance(pix_per_unit, (int, float)) else tuple(pix_per_unit)
    model_boxes: list[tuple[float, float, float, float, float]] = []
    for ppu in ppus:
        scene, transform = rasterize_scene(primitives, pix_per_unit=ppu)
        height, width = scene.shape
        tiles: list[np.ndarray] = []
        origins: list[tuple[int, int]] = []
        for oy in tile_origins(height):
            for ox in tile_origins(width):
                tile = np.zeros((1, TILE, TILE), dtype=np.float32)
                view = scene[oy : oy + TILE, ox : ox + TILE]
                tile[0, : view.shape[0], : view.shape[1]] = view
                tiles.append(tile)
                origins.append((ox, oy))
        per_tile = detect_tiles(model, tiles, device, conf_thresh)
        for (ox, oy), dets in zip(origins, per_tile):
            for cx, cy, w, h, conf in dets:
                mx1, my1 = transform.to_model(cx + ox - w / 2, cy + oy - h / 2)
                mx2, my2 = transform.to_model(cx + ox + w / 2, cy + oy + h / 2)
                model_boxes.append(((mx1 + mx2) / 2, (my1 + my2) / 2, abs(mx2 - mx1), abs(my2 - my1), conf))
    results = []
    for cx, cy, w, h, conf in nms(model_boxes):
        results.append({"bbox": (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2), "conf": conf})
    return results


def candidates_from_detections(
    primitives: list[Primitive],
    detections: list[dict],
    scale: float,
    tolerance: float,
    pad_frac: float = 0.15,
) -> list[dict]:
    """Turn detection boxes into candidate dicts identical to _candidate_pool output.

    Detections act as localization proposals only: connectivity components are
    computed once over the full vector primitive set, so a contour is recovered
    whole even when the mask box is tight or fragmented. A component is kept
    when its center falls inside a (padded) detection box.
    """
    comps = components(primitives, tolerance=tolerance)
    comps = [c for c in comps if c["perimeter"] * scale >= 40 and c["width"] * scale >= 8 and c["height"] * scale >= 4]
    padded = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        pad_x, pad_y = (x2 - x1) * pad_frac + 1.0, (y2 - y1) * pad_frac + 1.0
        padded.append((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, det["conf"]))
    candidates: list[dict] = []
    for comp in comps:
        cx = (comp["bbox"][0] + comp["bbox"][2]) / 2
        cy = (comp["bbox"][1] + comp["bbox"][3]) / 2
        conf = max((conf for x1, y1, x2, y2, conf in padded if x1 <= cx <= x2 and y1 <= cy <= y2), default=0.0)
        if conf > 0:
            comp["det_conf"] = conf
            candidates.append(comp)
    return candidates


def save_detector(model: UNetSegmenter, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_detector(path: Path, device: torch.device) -> UNetSegmenter:
    model = UNetSegmenter()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
