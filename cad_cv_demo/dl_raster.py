"""Rasterization helpers shared by the deep-learning detector and matcher.

All rasterization starts from vector primitives (never from rendered images),
so silhouettes keep full fidelity and augmentation happens in point space.
"""

from __future__ import annotations

import math

from cad_cv_demo.shape_match_diagnostics import Point, Primitive  # bootstraps vendor path

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def sample_primitive_points(primitive: Primitive, arc_steps: int = 48) -> list[Point]:
    """Approximate a primitive as a polyline (same sampling rules as the CV pipeline)."""
    data = primitive.data
    if primitive.kind == "LINE" and primitive.endpoints:
        return [primitive.endpoints[0], primitive.endpoints[1]]
    if primitive.kind == "ARC":
        center = data["center"]
        radius = float(data["radius"])
        start = float(data["start_angle"])
        sweep = (float(data["end_angle"]) - start) % 360.0
        return [
            (
                float(center[0]) + radius * math.cos(math.radians(start + sweep * i / arc_steps)),
                float(center[1]) + radius * math.sin(math.radians(start + sweep * i / arc_steps)),
            )
            for i in range(arc_steps + 1)
        ]
    if primitive.kind == "CIRCLE":
        center = data.get("center", ((primitive.bbox[0] + primitive.bbox[2]) / 2, (primitive.bbox[1] + primitive.bbox[3]) / 2))
        radius = float(data.get("radius", (primitive.bbox[2] - primitive.bbox[0]) / 2))
        return [
            (float(center[0]) + radius * math.cos(2 * math.pi * i / arc_steps), float(center[1]) + radius * math.sin(2 * math.pi * i / arc_steps))
            for i in range(arc_steps + 1)
        ]
    if primitive.kind == "LWPOLYLINE":
        points = [tuple(map(float, p[:2])) for p in data.get("points", [])]
        if data.get("closed") and points:
            points.append(points[0])
        return points
    return []


def primitive_polylines(primitives: list[Primitive], arc_steps: int = 48) -> list[list[Point]]:
    return [pts for pts in (sample_primitive_points(p, arc_steps) for p in primitives) if len(pts) >= 2]


def draw_polylines(image: np.ndarray, polylines: list[list[Point]], to_pixel, line_width: int = 1) -> None:
    for points in polylines:
        pixels = np.asarray([to_pixel(p) for p in points], dtype=np.int32)
        cv2.polylines(image, [pixels], False, 255, line_width, cv2.LINE_AA)


class SceneTransform:
    """Uniform model->pixel mapping (y flipped) for a full-sheet raster."""

    def __init__(self, x1: float, y2: float, pix_per_unit: float) -> None:
        self.x1 = float(x1)
        self.y2 = float(y2)
        self.pix_per_unit = float(pix_per_unit)

    def to_pixel(self, point: Point) -> tuple[int, int]:
        return (
            int(round((point[0] - self.x1) * self.pix_per_unit)),
            int(round((self.y2 - point[1]) * self.pix_per_unit)),
        )

    def to_model(self, px: float, py: float) -> Point:
        return (px / self.pix_per_unit + self.x1, self.y2 - py / self.pix_per_unit)


def scene_extent(primitives: list[Primitive]) -> tuple[float, float, float, float]:
    return (
        min(p.bbox[0] for p in primitives),
        min(p.bbox[1] for p in primitives),
        max(p.bbox[2] for p in primitives),
        max(p.bbox[3] for p in primitives),
    )


def rasterize_scene(
    primitives: list[Primitive],
    pix_per_unit: float = 3.0,
    extent: tuple[float, float, float, float] | None = None,
    pad_units: float = 6.0,
    line_width: int = 2,
) -> tuple[np.ndarray, SceneTransform]:
    """Rasterize the whole model space; returns (float32 image in [0,1], transform)."""
    x1, y1, x2, y2 = extent if extent is not None else scene_extent(primitives)
    x1, y1, x2, y2 = x1 - pad_units, y1 - pad_units, x2 + pad_units, y2 + pad_units
    width_px = int(math.ceil((x2 - x1) * pix_per_unit))
    height_px = int(math.ceil((y2 - y1) * pix_per_unit))
    transform = SceneTransform(x1, y2, pix_per_unit)
    image = np.zeros((height_px, width_px), dtype=np.uint8)
    draw_polylines(image, primitive_polylines(primitives), transform.to_pixel, line_width)
    return image.astype(np.float32) / 255.0, transform


def rasterize_primitives(
    primitives: list[Primitive],
    canvas: int = 128,
    margin_frac: float = 0.10,
    line_width: int = 2,
    polylines: list[list[Point]] | None = None,
) -> np.ndarray:
    """Scale/translation-normalized binary silhouette, float32 in [0,1], shape (canvas, canvas)."""
    lines = polylines if polylines is not None else primitive_polylines(primitives)
    image = np.zeros((canvas, canvas), dtype=np.uint8)
    if not lines:
        return image.astype(np.float32)
    xs = [p[0] for line in lines for p in line]
    ys = [p[1] for line in lines for p in line]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    span = max(x2 - x1, y2 - y1, 1e-9)
    usable = canvas * (1.0 - 2.0 * margin_frac)
    scale = usable / span
    ox = (canvas - (x2 - x1) * scale) / 2.0
    oy = (canvas - (y2 - y1) * scale) / 2.0

    def to_pixel(point: Point) -> tuple[int, int]:
        return (
            int(round((point[0] - x1) * scale + ox)),
            int(round(canvas - ((point[1] - y1) * scale + oy))),
        )

    draw_polylines(image, lines, to_pixel, line_width)
    return image.astype(np.float32) / 255.0


def transform_polylines(
    polylines: list[list[Point]],
    angle_deg: float = 0.0,
    scale: float = 1.0,
    scale_y: float | None = None,
    translate: Point = (0.0, 0.0),
) -> list[list[Point]]:
    """Rotate/scale/translate polylines around their collective centroid."""
    sy = scale if scale_y is None else scale_y
    cx = sum(p[0] for line in polylines for p in line) / max(sum(len(line) for line in polylines), 1)
    cy = sum(p[1] for line in polylines for p in line) / max(sum(len(line) for line in polylines), 1)
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    output: list[list[Point]] = []
    for line in polylines:
        transformed = []
        for x, y in line:
            dx, dy = (x - cx) * scale, (y - cy) * sy
            transformed.append((cx + dx * ca - dy * sa + translate[0], cy + dx * sa + dy * ca + translate[1]))
        output.append(transformed)
    return output


def drop_primitives(polylines: list[list[Point]], drop_prob: float, rng: np.random.Generator) -> list[list[Point]]:
    """Randomly drop whole polylines to simulate open / incomplete contours."""
    kept = [line for line in polylines if rng.random() > drop_prob]
    return kept if kept else polylines[:1]
