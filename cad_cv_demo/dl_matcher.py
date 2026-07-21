"""Deep-learning shape matcher: CNN silhouette embeddings with metric learning.

The encoder is trained self-supervised with InfoNCE: two augmented views of
the same part outline form a positive pair, all other parts in the batch are
negatives. Augmentations (rotation, aspect jitter, primitive dropout for open
contours, clutter injection) make the embedding robust to the distortions
found between reference DXFs and the source DWG views. Extra procedural
random shapes act as additional negative classes.
"""

from __future__ import annotations

import math
from pathlib import Path

from cad_cv_demo.shape_match_diagnostics import Point  # bootstraps vendor path
from cad_cv_demo.dl_raster import (  # noqa: E402
    draw_polylines,
    drop_primitives,
    rasterize_primitives,
    transform_polylines,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cad_cv_demo.dl_detector import ProgressPrinter, pick_device  # noqa: E402


CANVAS = 128
EMBED_DIM = 128


def _conv_block(channels_in: int, channels_out: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(channels_in, channels_out, 3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(channels_out),
        nn.ReLU(inplace=True),
    )


class ShapeEncoder(nn.Module):
    """128x128 binary silhouette -> L2-normalized 128-d embedding."""

    def __init__(self, embed_dim: int = EMBED_DIM) -> None:
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(1, 32, stride=2),
            _conv_block(32, 32),
            _conv_block(32, 64, stride=2),
            _conv_block(64, 64),
            _conv_block(64, 128, stride=2),
            _conv_block(128, 128),
            _conv_block(128, 256, stride=2),
        )
        self.projection = nn.Linear(256, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.features(x).mean(dim=(2, 3))
        return F.normalize(self.projection(pooled), dim=1)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_silhouette(
    polylines: list[list[Point]],
    rng: np.random.Generator,
    canvas: int = CANVAS,
    max_dropout: float = 0.10,
    clutter_prob: float = 0.5,
) -> np.ndarray:
    lines = transform_polylines(
        polylines,
        angle_deg=float(rng.uniform(0, 360)),
        scale=1.0,
        scale_y=float(rng.uniform(1.0 - 0.06, 1.0 + 0.06)),
    )
    lines = drop_primitives(lines, max_dropout * float(rng.random()), rng)
    image = rasterize_primitives([], canvas=canvas, polylines=lines, line_width=int(rng.integers(1, 4)))
    if rng.random() < clutter_prob:
        noisy = (image * 255.0).astype(np.uint8)
        for _ in range(rng.integers(1, 4)):
            x1, y1 = rng.integers(0, canvas, 2)
            x2, y2 = np.clip([x1 + rng.uniform(-30, 30), y1 + rng.uniform(-30, 30)], 0, canvas - 1)
            cv2.line(noisy, (int(x1), int(y1)), (int(x2), int(y2)), 255, 1, cv2.LINE_AA)
        image = noisy.astype(np.float32) / 255.0
    return image


# ---------------------------------------------------------------------------
# Procedural negative classes
# ---------------------------------------------------------------------------

def random_shape_polylines(rng: np.random.Generator) -> list[list[Point]]:
    kind = int(rng.integers(0, 5))
    if kind == 0:  # random radial polygon
        n = int(rng.integers(4, 13))
        angles = np.sort(rng.uniform(0, 2 * math.pi, n))
        radii = rng.uniform(0.55, 1.0, n)
        pts = [(float(r * math.cos(a)), float(r * math.sin(a))) for r, a in zip(radii, angles)]
        return [pts + [pts[0]]]
    if kind == 1:  # stadium / slot
        length, radius = float(rng.uniform(0.8, 1.8)), float(rng.uniform(0.2, 0.5))
        pts = [(length / 2 + radius * math.cos(t), radius * math.sin(t)) for t in np.linspace(-math.pi / 2, math.pi / 2, 25)]
        pts += [(-length / 2 + radius * math.cos(t), radius * math.sin(t)) for t in np.linspace(math.pi / 2, 3 * math.pi / 2, 25)]
        return [[(float(x), float(y)) for x, y in pts] + [pts[0]]]
    if kind == 2:  # ring
        r_out, r_in = float(rng.uniform(0.8, 1.1)), float(rng.uniform(0.3, 0.6))
        circle = lambda r: [(float(r * math.cos(t)), float(r * math.sin(t))) for t in np.linspace(0, 2 * math.pi, 49)]
        return [circle(r_out), circle(r_in)]
    if kind == 3:  # rounded rectangle
        w, h, r = float(rng.uniform(1.0, 2.0)), float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.1, 0.4))
        pts: list[Point] = []
        for cx, cy, start in ((w / 2 - r, h / 2 - r, 0), (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180), (w / 2 - r, -h / 2 + r, 270)):
            pts.extend((cx + r * math.cos(math.radians(start + t * 90 / 8)), cy + r * math.sin(math.radians(start + t * 90 / 8))) for t in range(9))
        return [[(float(x), float(y)) for x, y in pts] + [pts[0]]]
    # orthogonal notched polygon
    n = int(rng.integers(6, 14))
    xs = np.cumsum(rng.uniform(0.1, 0.4, n)) * rng.choice([-1, 1])
    ys = np.cumsum(rng.uniform(0.1, 0.4, n)) * rng.choice([-1, 1])
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)]
    return [pts + [pts[0]]]


# ---------------------------------------------------------------------------
# Training (InfoNCE)
# ---------------------------------------------------------------------------

def train_matcher(
    model: ShapeEncoder,
    class_outlines: list[list[list[Point]]],
    device: torch.device,
    steps: int = 800,
    batch_classes: int = 32,
    lr: float = 1e-3,
    temperature: float = 0.1,
    extra_negative_classes: int = 40,
    seed: int = 0,
) -> list[float]:
    rng = np.random.default_rng(seed)
    classes = [lines for lines in class_outlines if lines]
    for _ in range(extra_negative_classes):
        classes.append(random_shape_polylines(rng))
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    progress = ProgressPrinter("matcher", steps)
    history: list[float] = []
    for step in range(steps):
        picks = rng.integers(0, len(classes), size=batch_classes)
        views = []
        for pick in picks:
            lines = classes[int(pick)]
            views.append(augment_silhouette(lines, rng))
            views.append(augment_silhouette(lines, rng))
        batch = torch.from_numpy(np.stack(views)).unsqueeze(1).to(device)
        embeddings = model(batch)
        sim = (embeddings @ embeddings.T) / temperature
        sim.fill_diagonal_(float("-inf"))  # exclude self-similarity from the softmax
        # views are interleaved (view1, view2) per class: index i pairs with i^1
        targets = torch.arange(2 * batch_classes, device=device) ^ 1
        loss = F.cross_entropy(sim, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
        progress.update(step, history)
    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def embed_images(model: ShapeEncoder, images: list[np.ndarray], device: torch.device, batch_size: int = 32) -> np.ndarray:
    model.to(device).eval()
    output: list[np.ndarray] = []
    for start in range(0, len(images), batch_size):
        chunk = np.stack(images[start : start + batch_size])
        batch = torch.from_numpy(chunk).unsqueeze(1).to(device)
        output.append(model(batch).cpu().numpy())
    return np.concatenate(output) if output else np.zeros((0, EMBED_DIM), dtype=np.float32)


@torch.no_grad()
def embed_outline(model: ShapeEncoder, polylines: list[list[Point]], device: torch.device, views: int = 8, seed: int = 0) -> np.ndarray:
    """Stable embedding: mean over lightly augmented views, renormalized."""
    rng = np.random.default_rng(seed)
    images = [augment_silhouette(polylines, rng, max_dropout=0.02, clutter_prob=0.0) for _ in range(views)]
    embedding = embed_images(model, images, device).mean(axis=0)
    return embedding / max(float(np.linalg.norm(embedding)), 1e-9)


def save_matcher(model: ShapeEncoder, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_matcher(path: Path, device: torch.device) -> ShapeEncoder:
    model = ShapeEncoder()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return model.to(device).eval()
