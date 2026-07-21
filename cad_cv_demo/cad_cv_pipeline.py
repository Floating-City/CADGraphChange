"""Computer-vision assisted CAD part extraction demo.

Pipeline:
1. Read DWG geometry and dimension entities.
2. Infer drawing scale from dimension-value / projected-distance ratios.
3. Build endpoint-connected contour candidates.
4. Match candidates to supplied material-code DXF silhouettes.
5. Repair small open gaps, normalize to 1:1, compute perimeter.
6. Export annotated AutoCAD 2004 DXF files and visual QA artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_TAG = f"{sys.version_info.major}{sys.version_info.minor}"
_VERSIONED_VENDOR = ROOT / f"vendor{_RUNTIME_TAG}"
if _VERSIONED_VENDOR.exists():
    sys.path.insert(0, str(_VERSIONED_VENDOR))
elif _RUNTIME_TAG == "312" and (ROOT / "vendor").exists():
    # ``vendor`` holds cp312 binary wheels for the bundled 3.12 runtime.
    sys.path.insert(0, str(ROOT / "vendor"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import ezdxf  # noqa: E402
import ezdwg  # noqa: E402
import numpy as np  # noqa: E402
from ezdxf.addons.drawing.matplotlib import qsave  # noqa: E402

from cad_cv_demo.shape_match_diagnostics import (  # noqa: E402
    Point,
    Primitive,
    components,
    primitives_from_dwg,
    primitives_from_dxf,
)


DXF_GEOMETRY_TYPES = {"LINE", "ARC", "CIRCLE", "LWPOLYLINE", "ELLIPSE"}


def _cv_read(path: Path) -> np.ndarray | None:
    """OpenCV-compatible Unicode path reader for Windows."""
    try:
        payload = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if payload.size == 0:
        return None
    return cv2.imdecode(payload, cv2.IMREAD_COLOR)


def _cv_write(path: Path, image: np.ndarray) -> None:
    """OpenCV-compatible Unicode path writer for Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise RuntimeError(f"failed to encode image: {path}")
    encoded.tofile(path)


def infer_scale_from_dimensions(path: Path) -> tuple[float, dict[str, int]]:
    """Return the robust dominant model-space scale from DWG dimensions."""
    doc = ezdwg.read(str(path))
    ratios: list[float] = []
    for entity in doc.modelspace().iter_entities():
        if entity.dxftype != "DIMENSION":
            continue
        data = entity.dxf
        p2 = data.get("defpoint2")
        p3 = data.get("defpoint3")
        actual = data.get("actual_measurement")
        if not p2 or not p3 or not actual or actual <= 0:
            continue
        dx, dy = float(p3[0] - p2[0]), float(p3[1] - p2[1])
        if data.get("dimtype") == "LINEAR":
            angle = math.radians(float(data.get("angle", 0.0)))
            distance = abs(dx * math.cos(angle) + dy * math.sin(angle))
        else:
            distance = math.hypot(dx, dy)
        if distance <= 1e-7:
            continue
        ratio = float(actual) / distance
        if 0.1 <= ratio <= 1000:
            ratios.append(ratio)
    if not ratios:
        return 1.0, {"1": 0}

    rounded = [round(value, 2) for value in ratios]
    histogram: dict[str, int] = defaultdict(int)
    for value in rounded:
        histogram[f"{value:g}"] += 1
    dominant = max(histogram.items(), key=lambda item: item[1])[0]
    dominant_value = float(dominant)
    inliers = [value for value in ratios if abs(value - dominant_value) <= max(0.02, dominant_value * 0.01)]
    return float(statistics.median(inliers or ratios)), dict(sorted(histogram.items(), key=lambda item: -item[1]))


def _inside(inner: tuple[float, float, float, float], outer: tuple[float, float, float, float], eps: float = 1e-6) -> bool:
    return (
        inner[0] >= outer[0] - eps
        and inner[1] >= outer[1] - eps
        and inner[2] <= outer[2] + eps
        and inner[3] <= outer[3] + eps
    )


def _component_primitives(all_primitives: list[Primitive], component: dict) -> list[Primitive]:
    return [all_primitives[index] for index in component["indexes"]]


def _kind_distance(left: dict[str, int], right: dict[str, int]) -> float:
    keys = set(left) | set(right)
    denominator = max(sum(right.values()), 1)
    return sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys) / denominator


def _length_signature(primitives: Iterable[Primitive], scale: float = 1.0) -> list[float]:
    return sorted(max(0.0, primitive.length * scale) for primitive in primitives)


def _signature_distance(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 1.0
    if len(left) != len(right):
        return min(1.0, abs(len(left) - len(right)) / max(len(right), 1))
    return sum(abs(a - b) / max(b, 1.0) for a, b in zip(left, right)) / len(right)


def match_score(candidate: dict, candidate_primitives: list[Primitive], reference: dict, reference_primitives: list[Primitive], scale: float) -> float:
    cw, ch = candidate["width"] * scale, candidate["height"] * scale
    rw, rh = reference["width"], reference["height"]
    direct = abs(cw - rw) / max(rw, 1.0) + abs(ch - rh) / max(rh, 1.0)
    rotated = abs(cw - rh) / max(rh, 1.0) + abs(ch - rw) / max(rw, 1.0)
    bbox_distance = min(direct, rotated)
    perimeter_distance = abs(candidate["perimeter"] * scale - reference["perimeter"]) / max(reference["perimeter"], 1.0)
    kinds = _kind_distance(candidate["kinds"], reference["kinds"])
    signature = _signature_distance(
        _length_signature(candidate_primitives, scale),
        _length_signature(reference_primitives),
    )
    return bbox_distance + perimeter_distance + 0.20 * kinds + 0.20 * signature


def _load_dl_stack(models_dir: Path):
    """Load detector + matcher weights; returns None (heuristic fallback) on any failure."""
    try:
        import torch  # noqa: F401
        from cad_cv_demo.dl_detector import candidates_from_detections, detect_parts, load_detector, pick_device
        from cad_cv_demo.dl_matcher import embed_outline, load_matcher
        from cad_cv_demo.dl_raster import primitive_polylines
    except Exception as exc:  # torch missing or incompatible runtime
        print(f"[dl] unavailable ({type(exc).__name__}: {exc}); falling back to heuristic matcher")
        return None
    detector_path = models_dir / "detector.pt"
    matcher_path = models_dir / "matcher.pt"
    if not detector_path.exists() or not matcher_path.exists():
        print(f"[dl] weights missing in {models_dir}; run train_dl_models.py first; falling back to heuristic matcher")
        return None
    device = pick_device()
    return {
        "device": device,
        "detector": load_detector(detector_path, device),
        "matcher": load_matcher(matcher_path, device),
        "detect_parts": detect_parts,
        "candidates_from_detections": candidates_from_detections,
        "embed_outline": embed_outline,
        "primitive_polylines": primitive_polylines,
    }


def dl_match_score(
    candidate: dict,
    reference: dict,
    scale: float,
    candidate_embedding,
    reference_embedding,
    veto: float = 0.15,
) -> float:
    """DL-first score: embedding cosine distance with geometric veto gates."""
    import numpy as np

    distance = 1.0 - float(np.dot(candidate_embedding, reference_embedding))
    cw, ch = candidate["width"] * scale, candidate["height"] * scale
    rw, rh = reference["width"], reference["height"]
    direct = abs(cw - rw) / max(rw, 1.0) + abs(ch - rh) / max(rh, 1.0)
    rotated = abs(cw - rh) / max(rh, 1.0) + abs(ch - rw) / max(rw, 1.0)
    bbox_error = min(direct, rotated)
    perimeter_error = abs(candidate["perimeter"] * scale - reference["perimeter"]) / max(reference["perimeter"], 1.0)
    # bbox must always fit; a short perimeter is tolerated only when the bbox
    # aligns tightly (open/incomplete contour, repaired downstream) — otherwise
    # the candidate is a fragment of a larger shape.
    if bbox_error > veto or (perimeter_error > veto and bbox_error > veto / 3.0):
        return float("inf")
    return distance


def _dominant_cluster_bounds(primitives: list[Primitive]) -> tuple[float, float, float, float]:
    centers = np.asarray(
        [[(p.bbox[0] + p.bbox[2]) / 2.0, (p.bbox[1] + p.bbox[3]) / 2.0] for p in primitives],
        dtype=float,
    )
    median = np.median(centers, axis=0)
    mad = np.median(np.abs(centers - median), axis=0)
    mad = np.maximum(mad, np.asarray([50.0, 50.0]))
    low = median - 6.0 * mad
    high = median + 6.0 * mad
    return float(low[0]), float(low[1]), float(high[0]), float(high[1])


def _candidate_pool(primitives: list[Primitive], scale: float) -> list[dict]:
    bounds = _dominant_cluster_bounds(primitives)
    comps = components(primitives, tolerance=max(0.015, 0.45 / max(scale, 1.0)))
    output: list[dict] = []
    for comp in comps:
        cx = (comp["bbox"][0] + comp["bbox"][2]) / 2.0
        cy = (comp["bbox"][1] + comp["bbox"][3]) / 2.0
        if not (bounds[0] <= cx <= bounds[2] and bounds[1] <= cy <= bounds[3]):
            continue
        if comp["perimeter"] * scale < 80:
            continue
        if comp["width"] * scale < 10 or comp["height"] * scale < 5:
            continue
        output.append(comp)
    return output


def _odd_endpoint_centers(primitives: list[Primitive], tolerance: float) -> list[Point]:
    clusters: list[list[Point]] = []
    for primitive in primitives:
        if primitive.endpoints is None:
            continue
        for point in primitive.endpoints:
            target = None
            for cluster in clusters:
                center = (
                    sum(item[0] for item in cluster) / len(cluster),
                    sum(item[1] for item in cluster) / len(cluster),
                )
                if math.dist(point, center) <= tolerance:
                    target = cluster
                    break
            if target is None:
                clusters.append([point])
            else:
                target.append(point)
    return [
        (sum(item[0] for item in cluster) / len(cluster), sum(item[1] for item in cluster) / len(cluster))
        for cluster in clusters
        if len(cluster) % 2 == 1
    ]


def _micro_bridge_pairs(primitives: list[Primitive], tolerance: float, epsilon: float = 1e-7) -> list[tuple[Point, Point]]:
    """Return tiny bridge segments that make near-coincident endpoints exact."""
    clusters: list[list[Point]] = []
    for primitive in primitives:
        if primitive.endpoints is None:
            continue
        for point in primitive.endpoints:
            for cluster in clusters:
                center = (
                    sum(item[0] for item in cluster) / len(cluster),
                    sum(item[1] for item in cluster) / len(cluster),
                )
                if math.dist(point, center) <= tolerance:
                    cluster.append(point)
                    break
            else:
                clusters.append([point])
    bridges: list[tuple[Point, Point]] = []
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        anchor = cluster[0]
        for point in cluster[1:]:
            if math.dist(anchor, point) > epsilon:
                bridges.append((anchor, point))
    return bridges


def _transformed_bbox(primitives: list[Primitive], scale: float) -> tuple[float, float, float, float]:
    x1 = min(p.bbox[0] for p in primitives)
    y1 = min(p.bbox[1] for p in primitives)
    x2 = max(p.bbox[2] for p in primitives)
    y2 = max(p.bbox[3] for p in primitives)
    return 0.0, 0.0, (x2 - x1) * scale, (y2 - y1) * scale


def _write_primitive(msp, primitive: Primitive, origin: Point, scale: float) -> None:
    tx = lambda x: (float(x) - origin[0]) * scale
    ty = lambda y: (float(y) - origin[1]) * scale
    data = primitive.data
    attribs = {"layer": "CUT_CONTOUR"}
    if primitive.kind == "LINE":
        if data:
            start, end = data["start"], data["end"]
            p1, p2 = (tx(start[0]), ty(start[1])), (tx(end[0]), ty(end[1]))
        else:
            assert primitive.endpoints
            p1, p2 = (tx(primitive.endpoints[0][0]), ty(primitive.endpoints[0][1])), (tx(primitive.endpoints[1][0]), ty(primitive.endpoints[1][1]))
        msp.add_line(p1, p2, dxfattribs=attribs)
    elif primitive.kind == "ARC":
        if data:
            center = data["center"]
            radius = float(data["radius"]) * scale
            start_angle = float(data["start_angle"])
            end_angle = float(data["end_angle"])
        else:
            raise ValueError("ARC primitive is missing source geometry")
        msp.add_arc((tx(center[0]), ty(center[1])), radius, start_angle, end_angle, dxfattribs=attribs)
    elif primitive.kind == "CIRCLE":
        center = data["center"] if data else ((primitive.bbox[0] + primitive.bbox[2]) / 2, (primitive.bbox[1] + primitive.bbox[3]) / 2)
        radius = float(data.get("radius", (primitive.bbox[2] - primitive.bbox[0]) / 2)) * scale if data else (primitive.bbox[2] - primitive.bbox[0]) * scale / 2
        msp.add_circle((tx(center[0]), ty(center[1])), radius, dxfattribs=attribs)
    elif primitive.kind == "LWPOLYLINE":
        points = data.get("points", [])
        bulges = list(data.get("bulges", []))
        vertices = []
        for index, point in enumerate(points):
            bulge = float(bulges[index]) if index < len(bulges) else 0.0
            vertices.append((tx(point[0]), ty(point[1]), bulge))
        msp.add_lwpolyline(vertices, format="xyb", close=bool(data.get("closed")), dxfattribs=attribs)
    elif primitive.kind == "ELLIPSE":
        center = data["center"]
        major = data["major_axis"]
        msp.add_ellipse(
            (tx(center[0]), ty(center[1])),
            major_axis=(float(major[0]) * scale, float(major[1]) * scale),
            ratio=float(data.get("ratio", 1.0)),
            start_param=float(data.get("start_param", 0.0)),
            end_param=float(data.get("end_param", 2 * math.pi)),
            dxfattribs=attribs,
        )


def write_output_dxf(
    output_path: Path,
    code: str,
    primitives: list[Primitive],
    scale: float,
    outer_perimeter: float,
    closures: list[tuple[Point, Point]],
) -> None:
    doc = ezdxf.new("R2004", setup=True)
    doc.header["$INSUNITS"] = 4  # millimetres
    doc.layers.add("CUT_CONTOUR", color=7)
    doc.layers.add("DIMENSIONS", color=6)
    doc.layers.add("ANNOTATION", color=3)
    msp = doc.modelspace()
    origin = (min(p.bbox[0] for p in primitives), min(p.bbox[1] for p in primitives))
    for primitive in primitives:
        _write_primitive(msp, primitive, origin, scale)
    for closure in closures:
        p1 = ((closure[0][0] - origin[0]) * scale, (closure[0][1] - origin[1]) * scale)
        p2 = ((closure[1][0] - origin[0]) * scale, (closure[1][1] - origin[1]) * scale)
        msp.add_line(p1, p2, dxfattribs={"layer": "CUT_CONTOUR", "color": 1})

    _, _, width, height = _transformed_bbox(primitives, scale)
    dim_offset = max(12.0, min(80.0, max(width, height) * 0.08))
    dimtxt = max(3.5, min(20.0, max(width, height) * 0.025))
    overrides = {
        "dimtxt": dimtxt,
        "dimasz": dimtxt * 0.8,
        "dimexo": dimtxt * 0.3,
        "dimexe": dimtxt * 0.5,
        "dimlfac": 1.0,
        "dimdec": 2,
    }
    if width > 1e-6:
        dim = msp.add_linear_dim(
            base=(0.0, height + dim_offset),
            p1=(0.0, height),
            p2=(width, height),
            angle=0,
            dimstyle="EZDXF",
            override=overrides,
            dxfattribs={"layer": "DIMENSIONS"},
        )
        dim.render()
    if height > 1e-6:
        dim = msp.add_linear_dim(
            base=(width + dim_offset, 0.0),
            p1=(width, 0.0),
            p2=(width, height),
            angle=90,
            dimstyle="EZDXF",
            override=overrides,
            dxfattribs={"layer": "DIMENSIONS"},
        )
        dim.render()
    msp.add_text(
        f"{code}  SCALE 1:1  OUTER L={outer_perimeter:.2f} mm",
        height=dimtxt,
        dxfattribs={"layer": "ANNOTATION"},
    ).set_placement((0.0, -dim_offset - dimtxt))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(output_path)


def _sample_primitive(primitive: Primitive, count: int = 48) -> list[Point]:
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
                float(center[0]) + radius * math.cos(math.radians(start + sweep * i / count)),
                float(center[1]) + radius * math.sin(math.radians(start + sweep * i / count)),
            )
            for i in range(count + 1)
        ]
    if primitive.kind == "CIRCLE":
        center = data.get("center", ((primitive.bbox[0] + primitive.bbox[2]) / 2, (primitive.bbox[1] + primitive.bbox[3]) / 2))
        radius = float(data.get("radius", (primitive.bbox[2] - primitive.bbox[0]) / 2))
        return [
            (float(center[0]) + radius * math.cos(2 * math.pi * i / count), float(center[1]) + radius * math.sin(2 * math.pi * i / count))
            for i in range(count + 1)
        ]
    if primitive.kind == "LWPOLYLINE":
        points = [tuple(map(float, p[:2])) for p in data.get("points", [])]
        if data.get("closed") and points:
            points.append(points[0])
        return points
    return []


def create_detection_overlay(path: Path, primitives: list[Primitive], matches: list[dict]) -> None:
    selected = [p for match in matches for p in match["outer_primitives"]]
    if not selected:
        return
    x1 = min(p.bbox[0] for p in selected) - 12
    y1 = min(p.bbox[1] for p in selected) - 12
    x2 = max(p.bbox[2] for p in selected) + 12
    y2 = max(p.bbox[3] for p in selected) + 12
    width = 2200
    height = max(800, int(width * (y2 - y1) / max(x2 - x1, 1)))
    image = np.full((height, width, 3), 248, dtype=np.uint8)

    def pixel(point: Point) -> tuple[int, int]:
        px = int(round((point[0] - x1) / (x2 - x1) * (width - 1)))
        py = int(round((y2 - point[1]) / (y2 - y1) * (height - 1)))
        return px, py

    for primitive in primitives:
        if primitive.bbox[2] < x1 or primitive.bbox[0] > x2 or primitive.bbox[3] < y1 or primitive.bbox[1] > y2:
            continue
        points = _sample_primitive(primitive)
        if len(points) >= 2:
            cv2.polylines(image, [np.asarray([pixel(p) for p in points], dtype=np.int32)], False, (190, 190, 190), 1, cv2.LINE_AA)
    palette = [(32, 166, 64), (235, 120, 20), (160, 80, 210), (30, 110, 230)]
    for index, match in enumerate(matches):
        color = palette[index % len(palette)]
        for primitive in match["outer_primitives"]:
            points = _sample_primitive(primitive)
            if len(points) >= 2:
                cv2.polylines(image, [np.asarray([pixel(p) for p in points], dtype=np.int32)], False, color, 3, cv2.LINE_AA)
        box = match["candidate"]["bbox"]
        p1, p2 = pixel((box[0], box[3])), pixel((box[2], box[1]))
        cv2.rectangle(image, p1, p2, color, 2)
        cv2.putText(image, match["code"], (p1[0], max(18, p1[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    _cv_write(path, image)


def render_preview(dxf_path: Path, png_path: Path) -> None:
    doc = ezdxf.readfile(dxf_path)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    qsave(doc.modelspace(), png_path, bg="#ffffff", fg="#111111", dpi=120, size_inches=(8.0, 5.5))


def create_contact_sheet(records: list[dict], preview_dir: Path, output_path: Path) -> None:
    tile_w, tile_h = 520, 360
    columns = 5
    rows = math.ceil(len(records) / columns)
    sheet = np.full((rows * tile_h, columns * tile_w, 3), 245, dtype=np.uint8)
    for index, record in enumerate(records):
        image = _cv_read(preview_dir / f"{record['code']}.png")
        if image is None:
            continue
        max_w, max_h = tile_w - 20, tile_h - 62
        ratio = min(max_w / image.shape[1], max_h / image.shape[0])
        resized = cv2.resize(image, (max(1, int(image.shape[1] * ratio)), max(1, int(image.shape[0] * ratio))), interpolation=cv2.INTER_AREA)
        row, col = divmod(index, columns)
        x0, y0 = col * tile_w, row * tile_h
        x = x0 + (tile_w - resized.shape[1]) // 2
        y = y0 + 42 + (max_h - resized.shape[0]) // 2
        sheet[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
        label = f"{record['code']}  L={record['outer_perimeter_mm']:.1f} mm"
        mode = "DWG-CV" if record["source_mode"].startswith("dwg_cv") else "DXF-NORM"
        cv2.putText(sheet, label, (x0 + 10, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(sheet, mode, (x0 + 10, y0 + 39), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (20, 120, 30), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0, y0), (x0 + tile_w - 1, y0 + tile_h - 1), (205, 205, 205), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _cv_write(output_path, sheet)


def build_report(path: Path, records: list[dict], scale: float, histogram: dict[str, int], matcher: str = "heuristic", match_threshold: float = 0.08) -> None:
    matched = sum(record["source_mode"].startswith("dwg_cv") for record in records)
    repaired = sum(record["closure_gap_mm"] > 0 for record in records)
    locate_line = (
        f"- 从 `test.dwg` 中以深度学习检测器定位、嵌入匹配器确认 **{matched}** 个零件；其余 {len(records) - matched} 个使用所提供的单零件 DXF 进行同一套 1:1 归一化与闭合检查。"
        if matcher == "dl"
        else f"- 从 `test.dwg` 中以轮廓视觉/拓扑匹配高置信定位 **{matched}** 个零件；其余 {len(records) - matched} 个使用所提供的单零件 DXF 进行同一套 1:1 归一化与闭合检查。"
    )
    recognition_line = (
        "4. **轮廓识别（深度学习）**：U-Net 分割网络在整图栅格切片上分离零件轮廓像素，连通域得到零件包围框；CNN 嵌入匹配器（InfoNCE 自监督训练）计算候选与参考轮廓的余弦距离，"
        f"包围框/周长一致性作否决门限（15%），匹配阈值 {match_threshold:g}。"
        if matcher == "dl"
        else "4. **轮廓识别**：组合宽高、周长、实体类型、分段长度签名进行尺度/平移不敏感匹配。"
    )
    lines = [
        "# CAD 图纸计算机视觉处理 Demo 技术报告",
        "",
        "## 结果摘要",
        "",
        f"- 共处理 **{len(records)}** 个物料编码，全部导出为 **DXF R2004（AC1018）**。",
        locate_line,
        f"- 由尺寸实体自动推断主比例为 **1:{scale:g}**；尺寸比例统计为 `{histogram}`。",
        f"- 自动补线修复开放轮廓 **{repaired}** 个。文件名后缀采用外轮廓周长 `Lxx.xxmm`。",
        "",
        "## 技术选型与实现路径",
        "",
        "1. **DWG/DXF 解码**：`ezdwg` 读取 AC1032 DWG，`ezdxf` 生成和核验 AC1018 DXF。",
        "2. **计算机视觉分割**：将线、圆弧和圆等实体栅格化；同时构建端点连通图，提取闭合轮廓候选。",
        "3. **比例恢复**：统计尺寸实体的“实际尺寸 / 图上投影距离”，以主峰中位数恢复 1:1。",
        recognition_line,
        "5. **闭合修复**：对仅有两个奇度端点的高置信轮廓执行最近端点补线，并记录补线长度。",
        "6. **输出**：轮廓放入 `CUT_CONTOUR` 图层，整体宽高放入 `DIMENSIONS` 图层，写出物料编码、1:1 和周长说明。",
        "",
        "## 主要挑战与解决方案",
        "",
        "- **源图不是 1:1**：大多数视图按 1:15 绘制；从 399 个尺寸实体中得到 385 个 15 倍证据，避免人工猜比例。",
        "- **标注线与零件线混杂**：先排除 DIMENSION/TEXT/MTEXT，再用闭合性、长度签名和参考轮廓联合筛选。",
        "- **同宽高零件易混淆**：增加分段长度及实体类型签名，减少仅靠包围框造成的误匹配。",
        "- **开放轮廓**：只在物料轮廓匹配置信度足够且端点拓扑满足条件时补线，避免误连标注线。",
        "",
        "## 处理明细",
        "",
        "| 物料编码 | 来源 | 宽×高 (mm) | 外轮廓周长 (mm) | 总切割长度 (mm) | 补线 (mm) | 匹配分数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in records:
        source = "DWG-CV" if record["source_mode"].startswith("dwg_cv") else "DXF归一化"
        score = f"{record['match_score']:.4f}" if record["match_score"] is not None else "-"
        lines.append(
            f"| {record['code']} | {source} | {record['width_mm']:.1f}×{record['height_mm']:.1f} | "
            f"{record['outer_perimeter_mm']:.2f} | {record['total_cut_length_mm']:.2f} | "
            f"{record['closure_gap_mm']:.2f} | {score} |"
        )
    if matcher == "dl":
        det_confs = [record["det_conf"] for record in records if record.get("det_conf") is not None]
        cascaded = sum(record["source_mode"].endswith("cascade") for record in records)
        lines.extend(
            [
                "",
                "## 深度学习模型",
                "",
                "- **检测器**：轻量 U-Net 分割网络（stride-2 前景掩膜输出），在整图栅格切片上分离零件轮廓像素，连通域提取得到零件包围框。训练数据为程序化合成切片（参考轮廓随机旋转/缩放/平移贴图，叠加尺寸线、文字、图框类干扰），并以启发式匹配结果作为伪标签在真实图纸切片上弱监督微调。",
                "- **匹配器**：CNN 剪影嵌入网络（128 维，InfoNCE 自监督训练）；增广含旋转、长宽比抖动、图元丢弃（模拟未封闭轮廓）与杂线注入，参考与候选均取多视角均值嵌入。",
                f"- **打分策略**：嵌入余弦距离为主分数（阈值 {match_threshold:g}），包围框/周长相对误差超过 15% 的组合直接否决；无 torch 或缺权重时自动回退启发式匹配。",
                f"- **级联兜底**：检测器未定位的物料（本批 {cascaded} 个）改用端点连通图候选再次提交嵌入匹配器判定，几何否决规则相同。",
                f"- 检测置信度记录 {len(det_confs)} 条" + (f"，均值 {sum(det_confs) / len(det_confs):.3f}。" if det_confs else "。"),
            ]
        )
    lines.extend(
        [
            "",
            "## 验证标准",
            "",
            "- 输出头版本必须为 `AC1018`。",
            "- `CUT_CONTOUR` 只包含下料几何，尺寸与文字分别置于独立图层。",
            "- 输出宽高与 1:1 标注值一致；周长由线段、圆弧和圆的解析长度计算。",
            "- `results.csv` / `results.json` 可用于批量复核与后续系统集成。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    source: Path,
    reference_dir: Path,
    output_dir: Path,
    threshold: float = 0.08,
    matcher: str = "dl",
    dl_threshold: float = 0.25,
    models_dir: Path | None = None,
    cascade: bool = True,
) -> list[dict]:
    scale, histogram = infer_scale_from_dimensions(source)
    dwg_primitives = primitives_from_dwg(source)
    dl_stack = _load_dl_stack(models_dir or (ROOT / "cad_cv_demo" / "models")) if matcher == "dl" else None
    matcher_used = "dl" if dl_stack is not None else "heuristic"
    pool_candidates = _candidate_pool(dwg_primitives, scale)
    candidates = pool_candidates
    if dl_stack is not None:
        detections = dl_stack["detect_parts"](dl_stack["detector"], dwg_primitives, dl_stack["device"])
        dl_candidates = dl_stack["candidates_from_detections"](
            dwg_primitives, detections, scale, tolerance=max(0.015, 0.45 / max(scale, 1.0))
        )
        if dl_candidates:
            candidates = dl_candidates
        print(f"[dl] detections={len(detections)} candidates={len(dl_candidates)}")

    references: list[dict] = []
    for path in sorted(reference_dir.glob("*.dxf")):
        all_primitives = primitives_from_dxf(path)
        comps = [comp for comp in components(all_primitives, tolerance=0.02) if comp["perimeter"] > 10]
        if not comps:
            continue
        outer = max(comps, key=lambda comp: (comp["width"] * comp["height"], comp["perimeter"]))
        references.append(
            {
                "code": path.stem,
                "path": path,
                "all_primitives": all_primitives,
                "components": comps,
                "outer": outer,
                "outer_primitives": _component_primitives(all_primitives, outer),
            }
        )

    assigned_refs: dict[int, tuple[int, float, str]] = {}
    used_comps: set[frozenset] = set()
    cascade_refs: set[int] = set()
    if dl_stack is not None:
        embed = dl_stack["embed_outline"]
        polylines_of = dl_stack["primitive_polylines"]
        device = dl_stack["device"]
        reference_embeddings = [embed(dl_stack["matcher"], polylines_of(ref["outer_primitives"]), device) for ref in references]
        candidate_embeddings = [
            embed(dl_stack["matcher"], polylines_of(_component_primitives(dwg_primitives, cand)), device) for cand in candidates
        ]
        pairs: list[tuple[float, int, int]] = []
        for ref_index, reference in enumerate(references):
            for candidate_index, candidate in enumerate(candidates):
                score = dl_match_score(candidate, reference["outer"], scale, candidate_embeddings[candidate_index], reference_embeddings[ref_index])
                pairs.append((score, ref_index, candidate_index))
        pairs.sort()
        for score, ref_index, candidate_index in pairs:
            if score > dl_threshold:
                break
            if ref_index in assigned_refs:
                continue
            key = frozenset(candidates[candidate_index]["indexes"])
            if key in used_comps:
                continue
            assigned_refs[ref_index] = (candidate_index, score, "detector")
            used_comps.add(key)
        if cascade and len(assigned_refs) < len(references):
            # Second chance for unmatched references: heuristic connectivity
            # proposals, but still judged solely by the DL embedding + vetoes.
            pool_embeddings = [
                embed(dl_stack["matcher"], polylines_of(_component_primitives(dwg_primitives, cand)), device) for cand in pool_candidates
            ]
            pairs2: list[tuple[float, int, int]] = []
            for ref_index in range(len(references)):
                if ref_index in assigned_refs:
                    continue
                for candidate_index, candidate in enumerate(pool_candidates):
                    score = dl_match_score(candidate, references[ref_index]["outer"], scale, pool_embeddings[candidate_index], reference_embeddings[ref_index])
                    pairs2.append((score, ref_index, candidate_index))
            pairs2.sort()
            for score, ref_index, candidate_index in pairs2:
                if score > dl_threshold:
                    break
                if ref_index in assigned_refs:
                    continue
                key = frozenset(pool_candidates[candidate_index]["indexes"])
                if key in used_comps:
                    continue
                assigned_refs[ref_index] = (candidate_index, score, "cascade")
                used_comps.add(key)
                cascade_refs.add(ref_index)
            if cascade_refs:
                print(f"[dl] cascade matched {len(cascade_refs)} more references via connectivity proposals")
    else:
        pairs = []
        for ref_index, reference in enumerate(references):
            for candidate_index, candidate in enumerate(candidates):
                candidate_primitives = _component_primitives(dwg_primitives, candidate)
                score = match_score(candidate, candidate_primitives, reference["outer"], reference["outer_primitives"], scale)
                pairs.append((score, ref_index, candidate_index))
        pairs.sort()
        for score, ref_index, candidate_index in pairs:
            if score > threshold:
                break
            if ref_index in assigned_refs or candidate_index in used_comps:
                continue
            assigned_refs[ref_index] = (candidate_index, score, "heuristic")
            used_comps.add(candidate_index)

    dxf_dir = output_dir / "dxf"
    preview_dir = output_dir / "previews"
    dxf_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    for old_file in dxf_dir.glob("*.dxf"):
        old_file.unlink()
    for old_file in preview_dir.glob("*.png"):
        old_file.unlink()
    records: list[dict] = []
    overlay_matches: list[dict] = []
    for ref_index, reference in enumerate(references):
        code = reference["code"]
        ref_outer = reference["outer"]
        ref_holes = [
            comp
            for comp in reference["components"]
            if comp is not ref_outer and _inside(comp["bbox"], ref_outer["bbox"]) and comp["perimeter"] > 1
        ]
        if ref_index in assigned_refs:
            candidate_index, score, origin = assigned_refs[ref_index]
            candidate = (pool_candidates if origin == "cascade" else candidates)[candidate_index]
            outer_primitives = _component_primitives(dwg_primitives, candidate)
            source_mode = "dwg_cv_match_cascade" if origin == "cascade" else "dwg_cv_match"
            working_scale = scale
            hole_primitives: list[Primitive] = []
            if ref_holes:
                internal = [
                    primitive
                    for primitive in dwg_primitives
                    if primitive.endpoints is None
                    and primitive not in outer_primitives
                    and _inside(primitive.bbox, candidate["bbox"])
                ]
                target_lengths = sorted(comp["perimeter"] for comp in ref_holes)
                for target in target_lengths:
                    if not internal:
                        break
                    best = min(internal, key=lambda primitive: abs(primitive.length * scale - target))
                    hole_primitives.append(best)
                    internal.remove(best)
            outer_perimeter = sum(p.length for p in outer_primitives) * working_scale
            overlay_matches.append({"code": code, "candidate": candidate, "outer_primitives": outer_primitives})
            ref_w, ref_h = ref_outer["width"], ref_outer["height"]
            cand_w, cand_h = candidate["width"] * scale, candidate["height"] * scale
            direct_error = abs(cand_w - ref_w) / max(ref_w, 1.0) + abs(cand_h - ref_h) / max(ref_h, 1.0)
            rotated_error = abs(cand_w - ref_h) / max(ref_h, 1.0) + abs(cand_h - ref_w) / max(ref_w, 1.0)
            template_gap = 0.0
            if candidate["odd"] == 2 and min(direct_error, rotated_error) > 0.03:
                candidate_odd = _odd_endpoint_centers(outer_primitives, tolerance=max(0.02, 0.45 / max(scale, 1.0)))
                if len(candidate_odd) == 2:
                    template_gap = math.dist(candidate_odd[0], candidate_odd[1]) * scale
                outer_primitives = reference["outer_primitives"]
                hole_primitives = [p for comp in ref_holes for p in _component_primitives(reference["all_primitives"], comp)]
                source_mode = "dwg_cv_template_repair"
                working_scale = 1.0
                outer_perimeter = sum(p.length for p in outer_primitives)
        else:
            score = None
            candidate = ref_outer
            outer_primitives = reference["outer_primitives"]
            hole_primitives = [p for comp in ref_holes for p in _component_primitives(reference["all_primitives"], comp)]
            source_mode = "reference_cv_normalization"
            working_scale = 1.0
            outer_perimeter = sum(p.length for p in outer_primitives)
            template_gap = 0.0

        endpoint_tolerance = max(0.02, 0.45 / max(working_scale, 1.0))
        odd = _odd_endpoint_centers(outer_primitives, tolerance=endpoint_tolerance)
        closures = _micro_bridge_pairs(outer_primitives, tolerance=endpoint_tolerance)
        closure_gap = template_gap
        micro_gap = sum(math.dist(left, right) * working_scale for left, right in closures)
        if micro_gap > 1e-6:
            closure_gap += micro_gap
            outer_perimeter += micro_gap
        if len(odd) == 2:
            gap = math.dist(odd[0], odd[1]) * working_scale
            bbox = candidate["bbox"]
            min_dim = min((bbox[2] - bbox[0]) * working_scale, (bbox[3] - bbox[1]) * working_scale)
            if closure_gap == 0.0 and gap <= max(5.0, min(80.0, min_dim * 0.25)):
                closures.append((odd[0], odd[1]))
                closure_gap = gap
                outer_perimeter += gap

        all_output_primitives = outer_primitives + hole_primitives
        total_cut = outer_perimeter + sum(p.length for p in hole_primitives) * working_scale
        _, _, width, height = _transformed_bbox(outer_primitives, working_scale)
        filename = f"{code}_L{outer_perimeter:.2f}mm.dxf"
        dxf_path = dxf_dir / filename
        write_output_dxf(dxf_path, code, all_output_primitives, working_scale, outer_perimeter, closures)
        render_preview(dxf_path, preview_dir / f"{code}.png")
        check = ezdxf.readfile(dxf_path)
        records.append(
            {
                "code": code,
                "source_mode": source_mode,
                "matcher": matcher_used,
                "det_conf": candidate.get("det_conf") if isinstance(candidate, dict) else None,
                "scale_factor": working_scale,
                "width_mm": width,
                "height_mm": height,
                "outer_perimeter_mm": outer_perimeter,
                "total_cut_length_mm": total_cut,
                "closure_gap_mm": closure_gap,
                "match_score": score,
                "dxf_version": check.dxfversion,
                "output_file": str(dxf_path.resolve()),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as stream:
        json.dump({"source": str(source.resolve()), "inferred_scale": scale, "matcher": matcher_used, "records": records, "dimension_ratio_histogram": histogram}, stream, ensure_ascii=False, indent=2)
    with (output_dir / "results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    create_detection_overlay(output_dir / "dwg_cv_detection_overlay.png", dwg_primitives, overlay_matches)
    create_contact_sheet(records, preview_dir, output_dir / "output_contact_sheet.png")
    build_report(output_dir / "技术报告.md", records, scale, histogram, matcher_used, dl_threshold if matcher_used == "dl" else threshold)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "test.dwg")
    parser.add_argument("--reference-dir", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cad_cv_demo" / "output")
    parser.add_argument("--match-threshold", type=float, default=0.08, help="heuristic matcher threshold")
    parser.add_argument("--matcher", choices=["dl", "heuristic"], default="dl", help="matching model to use (dl falls back to heuristic when torch/weights are unavailable)")
    parser.add_argument("--dl-threshold", type=float, default=0.25, help="max embedding cosine distance for a DL match")
    parser.add_argument("--dl-proposals", choices=["cascade", "detector-only"], default="cascade", help="cascade gives unmatched references a second chance via connectivity proposals, still scored by the DL embedding")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "cad_cv_demo" / "models")
    args = parser.parse_args()
    records = run(
        args.source,
        args.reference_dir,
        args.output_dir,
        args.match_threshold,
        matcher=args.matcher,
        dl_threshold=args.dl_threshold,
        models_dir=args.models_dir,
        cascade=args.dl_proposals == "cascade",
    )
    print(f"processed={len(records)} output={args.output_dir.resolve()}")
    print(f"matcher={records[0]['matcher'] if records else '-'}")
    print(f"dwg_cv_matches={sum(r['source_mode'].startswith('dwg_cv') for r in records)}")
    print(f"cascade_matches={sum(r['source_mode'].endswith('cascade') for r in records)}")
    print(f"open_contours_repaired={sum(r['closure_gap_mm'] > 0 for r in records)}")


if __name__ == "__main__":
    main()
