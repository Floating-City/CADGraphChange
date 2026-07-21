"""Find closed shape candidates in the source DWG and reference DXFs.

This diagnostic is intentionally vector-assisted CV: entity endpoints define the
connectivity graph, while normalized raster silhouettes provide the final shape
descriptor used by the production pipeline.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _bootstrap_vendor() -> None:
    """Insert the vendored dependencies matching the running interpreter.

    ``vendor/`` carries cp312 binary wheels for the bundled 3.12 runtime;
    ``vendor313/`` carries the cp313 wheels used with the system 3.13
    interpreter (which also provides torch). Other interpreters rely on
    their own site-packages.
    """
    root = Path(__file__).resolve().parents[1]
    tag = f"{sys.version_info.major}{sys.version_info.minor}"
    versioned = root / f"vendor{tag}"
    if versioned.exists():
        sys.path.insert(0, str(versioned))
    elif tag == "312" and (root / "vendor").exists():
        sys.path.insert(0, str(root / "vendor"))


_bootstrap_vendor()

import ezdxf  # noqa: E402
import ezdwg  # noqa: E402


Point = tuple[float, float]


@dataclass
class Primitive:
    kind: str
    data: dict
    handle: int | str
    endpoints: tuple[Point, Point] | None
    bbox: tuple[float, float, float, float]
    length: float


def _distance(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _arc_points(center: Point, radius: float, start: float, end: float) -> tuple[Point, Point]:
    return (
        (center[0] + radius * math.cos(math.radians(start)), center[1] + radius * math.sin(math.radians(start))),
        (center[0] + radius * math.cos(math.radians(end)), center[1] + radius * math.sin(math.radians(end))),
    )


def _arc_sweep(start: float, end: float) -> float:
    return (end - start) % 360.0


def _bbox_points(points: Iterable[Point]) -> tuple[float, float, float, float]:
    pts = list(points)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _arc_bbox(center: Point, radius: float, start: float, end: float) -> tuple[float, float, float, float]:
    sweep = _arc_sweep(start, end)
    angles = [start, end]
    for angle in (0.0, 90.0, 180.0, 270.0):
        if (angle - start) % 360.0 <= sweep + 1e-9:
            angles.append(angle)
    return _bbox_points(
        (center[0] + radius * math.cos(math.radians(a)), center[1] + radius * math.sin(math.radians(a)))
        for a in angles
    )


def primitives_from_dwg(path: Path) -> list[Primitive]:
    doc = ezdwg.read(str(path))
    result: list[Primitive] = []
    for entity in doc.modelspace().iter_entities():
        d = entity.dxf
        kind = entity.dxftype
        if kind == "LINE":
            p1 = tuple(map(float, d["start"][:2]))
            p2 = tuple(map(float, d["end"][:2]))
            result.append(Primitive(kind, d, entity.handle, (p1, p2), _bbox_points((p1, p2)), _distance(p1, p2)))
        elif kind == "ARC":
            center = tuple(map(float, d["center"][:2]))
            radius = float(d["radius"])
            start = float(d["start_angle"])
            end = float(d["end_angle"])
            endpoints = _arc_points(center, radius, start, end)
            result.append(Primitive(kind, d, entity.handle, endpoints, _arc_bbox(center, radius, start, end), radius * math.radians(_arc_sweep(start, end))))
        elif kind == "CIRCLE":
            center = tuple(map(float, d["center"][:2]))
            radius = float(d["radius"])
            result.append(Primitive(kind, d, entity.handle, None, (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), 2 * math.pi * radius))
        elif kind == "LWPOLYLINE":
            points = [tuple(map(float, p[:2])) for p in d.get("points", [])]
            if len(points) < 2:
                continue
            closed = bool(d.get("closed"))
            endpoints = None if closed else (points[0], points[-1])
            seq = points + ([points[0]] if closed else [])
            length = sum(_distance(a, b) for a, b in zip(seq, seq[1:]))
            result.append(Primitive(kind, d, entity.handle, endpoints, _bbox_points(points), length))
        elif kind == "ELLIPSE":
            center = tuple(map(float, d["center"][:2]))
            major = tuple(map(float, d["major_axis"][:2]))
            a = math.hypot(*major)
            b = a * float(d.get("ratio", 1.0))
            result.append(Primitive(kind, d, entity.handle, None, (center[0] - a, center[1] - b, center[0] + a, center[1] + b), math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))))
    return result


def primitives_from_dxf(path: Path) -> list[Primitive]:
    doc = ezdxf.readfile(path)
    result: list[Primitive] = []
    for entity in doc.modelspace():
        kind = entity.dxftype()
        if kind == "LINE":
            p1 = (float(entity.dxf.start.x), float(entity.dxf.start.y))
            p2 = (float(entity.dxf.end.x), float(entity.dxf.end.y))
            data = {"start": (p1[0], p1[1], 0.0), "end": (p2[0], p2[1], 0.0)}
            result.append(Primitive(kind, data, entity.dxf.handle, (p1, p2), _bbox_points((p1, p2)), _distance(p1, p2)))
        elif kind == "ARC":
            center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
            radius = float(entity.dxf.radius)
            start = float(entity.dxf.start_angle)
            end = float(entity.dxf.end_angle)
            endpoints = _arc_points(center, radius, start, end)
            data = {"center": (center[0], center[1], 0.0), "radius": radius, "start_angle": start, "end_angle": end}
            result.append(Primitive(kind, data, entity.dxf.handle, endpoints, _arc_bbox(center, radius, start, end), radius * math.radians(_arc_sweep(start, end))))
        elif kind == "CIRCLE":
            center = (float(entity.dxf.center.x), float(entity.dxf.center.y))
            radius = float(entity.dxf.radius)
            data = {"center": (center[0], center[1], 0.0), "radius": radius}
            result.append(Primitive(kind, data, entity.dxf.handle, None, (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius), 2 * math.pi * radius))
    return result


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        a, b = self.find(left), self.find(right)
        if a != b:
            self.parent[b] = a


def components(primitives: list[Primitive], tolerance: float) -> list[dict]:
    endpoints: list[tuple[Point, int]] = []
    closed_singletons: list[int] = []
    for index, primitive in enumerate(primitives):
        if primitive.endpoints is None:
            closed_singletons.append(index)
        else:
            endpoints.extend((point, index) for point in primitive.endpoints)

    uf = UnionFind(len(primitives))
    grid: dict[tuple[int, int], list[tuple[Point, int]]] = defaultdict(list)
    cell = max(tolerance, 1e-9)
    for point, index in endpoints:
        key = (math.floor(point[0] / cell), math.floor(point[1] / cell))
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for other, other_index in grid.get((key[0] + dx, key[1] + dy), []):
                    if _distance(point, other) <= tolerance:
                        uf.union(index, other_index)
        grid[key].append((point, index))

    groups: dict[int, list[int]] = defaultdict(list)
    for index in range(len(primitives)):
        groups[uf.find(index)].append(index)

    output = []
    for indexes in groups.values():
        items = [primitives[i] for i in indexes]
        x1 = min(p.bbox[0] for p in items)
        y1 = min(p.bbox[1] for p in items)
        x2 = max(p.bbox[2] for p in items)
        y2 = max(p.bbox[3] for p in items)
        endpoint_counts: dict[tuple[int, int], int] = defaultdict(int)
        for p in items:
            if p.endpoints:
                for pt in p.endpoints:
                    endpoint_counts[(round(pt[0] / cell), round(pt[1] / cell))] += 1
        odd = sum(value % 2 for value in endpoint_counts.values())
        output.append(
            {
                "indexes": indexes,
                "bbox": (x1, y1, x2, y2),
                "width": x2 - x1,
                "height": y2 - y1,
                "perimeter": sum(p.length for p in items),
                "kinds": dict(__import__("collections").Counter(p.kind for p in items)),
                "odd": odd,
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dwg", type=Path)
    parser.add_argument("--references", type=Path, default=Path("."))
    parser.add_argument("--scale", type=float, default=15.0)
    args = parser.parse_args()

    refs = {}
    for path in sorted(args.references.glob("*.dxf")):
        comps = components(primitives_from_dxf(path), tolerance=0.02)
        comps = [c for c in comps if c["perimeter"] > 10]
        if not comps:
            continue
        refs[path.stem] = max(comps, key=lambda c: c["perimeter"])

    dwg_primitives = primitives_from_dwg(args.dwg)
    candidates = components(dwg_primitives, tolerance=0.03)
    candidates = [
        c
        for c in candidates
        if c["bbox"][0] > 2000
        and c["bbox"][1] < 20
        and c["width"] * args.scale >= 40
        and c["height"] * args.scale >= 20
        and c["width"] * args.scale <= 4000
        and c["height"] * args.scale <= 2500
        and c["perimeter"] * args.scale >= 100
    ]
    print(f"references={len(refs)} candidates={len(candidates)}")
    for index, cand in enumerate(sorted(candidates, key=lambda c: (c["bbox"][1], c["bbox"][0]), reverse=True)):
        sw, sh = cand["width"] * args.scale, cand["height"] * args.scale
        score_rows = []
        for code, ref in refs.items():
            rw, rh = ref["width"], ref["height"]
            direct = abs(sw - rw) / max(rw, 1) + abs(sh - rh) / max(rh, 1)
            rotated = abs(sw - rh) / max(rh, 1) + abs(sh - rw) / max(rw, 1)
            perim = abs(cand["perimeter"] * args.scale - ref["perimeter"]) / max(ref["perimeter"], 1)
            count = abs(sum(cand["kinds"].values()) - sum(ref["kinds"].values())) / max(sum(ref["kinds"].values()), 1)
            score_rows.append((min(direct, rotated) + perim + 0.2 * count, code))
        score_rows.sort()
        print(
            f"{index:02d} bbox={tuple(round(v, 2) for v in cand['bbox'])} "
            f"actual={sw:.1f}x{sh:.1f} P={cand['perimeter'] * args.scale:.1f} "
            f"odd={cand['odd']} kinds={cand['kinds']} best={score_rows[:3]}"
        )


if __name__ == "__main__":
    main()
