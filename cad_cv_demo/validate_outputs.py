"""Structural validation for generated R2004 cutting DXFs."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_TAG = f"{sys.version_info.major}{sys.version_info.minor}"
_VERSIONED_VENDOR = ROOT / f"vendor{_RUNTIME_TAG}"
if _VERSIONED_VENDOR.exists():
    sys.path.insert(0, str(_VERSIONED_VENDOR))
elif _RUNTIME_TAG == "312" and (ROOT / "vendor").exists():
    # ``vendor`` holds cp312 binary wheels for the bundled 3.12 runtime.
    sys.path.insert(0, str(ROOT / "vendor"))

import ezdxf  # noqa: E402


def _cluster_odd_endpoints(points: list[tuple[float, float]], tolerance: float = 1e-5) -> int:
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for x, y in points:
        counts[(round(x / tolerance), round(y / tolerance))] += 1
    return sum(value % 2 for value in counts.values())


def validate(path: Path) -> dict:
    doc = ezdxf.readfile(path)
    endpoints: list[tuple[float, float]] = []
    perimeter = 0.0
    outline_count = 0
    segment_records: list[tuple[float, list[tuple[float, float]]]] = []
    for entity in doc.modelspace():
        if entity.dxf.get("layer", "") != "CUT_CONTOUR":
            continue
        outline_count += 1
        kind = entity.dxftype()
        if kind == "LINE":
            p1, p2 = entity.dxf.start, entity.dxf.end
            points = [(float(p1.x), float(p1.y)), (float(p2.x), float(p2.y))]
            length = math.hypot(p2.x - p1.x, p2.y - p1.y)
            endpoints.extend(points)
            perimeter += length
            segment_records.append((length, points))
        elif kind == "ARC":
            radius = float(entity.dxf.radius)
            sweep = (float(entity.dxf.end_angle) - float(entity.dxf.start_angle)) % 360.0
            length = radius * math.radians(sweep)
            perimeter += length
            center = entity.dxf.center
            for angle in (float(entity.dxf.start_angle), float(entity.dxf.end_angle)):
                endpoints.append(
                    (
                        float(center.x) + radius * math.cos(math.radians(angle)),
                        float(center.y) + radius * math.sin(math.radians(angle)),
                    )
                )
            segment_records.append((length, endpoints[-2:]))
        elif kind == "CIRCLE":
            length = 2 * math.pi * float(entity.dxf.radius)
            perimeter += length
            segment_records.append((length, []))
        elif kind == "LWPOLYLINE":
            length = entity.length()
            perimeter += length
            local_endpoints: list[tuple[float, float]] = []
            if not entity.closed:
                points = list(entity.get_points("xy"))
                if points:
                    local_endpoints = [tuple(map(float, points[0])), tuple(map(float, points[-1]))]
                    endpoints.extend(local_endpoints)
            segment_records.append((length, local_endpoints))

    parent = list(range(len(segment_records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[b] = a

    endpoint_owners: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (_, local_points) in enumerate(segment_records):
        for x, y in local_points:
            endpoint_owners[(round(x / 1e-5), round(y / 1e-5))].append(index)
    for owners in endpoint_owners.values():
        for index in owners[1:]:
            union(owners[0], index)
    component_lengths: dict[int, float] = defaultdict(float)
    for index, (length, _) in enumerate(segment_records):
        component_lengths[find(index)] += length
    outer_perimeter = max(component_lengths.values(), default=0.0)

    match = re.search(r"_L([0-9]+(?:\.[0-9]+)?)mm$", path.stem)
    filename_perimeter = float(match.group(1)) if match else None
    dimensions = len(doc.modelspace().query("DIMENSION"))
    odd = _cluster_odd_endpoints(endpoints)
    filename_error = abs(outer_perimeter - filename_perimeter) if filename_perimeter is not None else None
    passed = (
        doc.dxfversion == "AC1018"
        and outline_count > 0
        and dimensions >= 2
        and odd == 0
        and filename_error is not None
        and filename_error <= 0.02
    )
    return {
        "file": path.name,
        "dxf_version": doc.dxfversion,
        "outline_entities": outline_count,
        "dimension_entities": dimensions,
        "odd_endpoints": odd,
        "recomputed_cut_length_mm": perimeter,
        "recomputed_outer_perimeter_mm": outer_perimeter,
        "filename_outer_perimeter_mm": filename_perimeter,
        "perimeter_difference_mm": filename_error,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, nargs="?", default=ROOT / "cad_cv_demo" / "output" / "dxf")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    records = [validate(path) for path in sorted(args.directory.glob("*.dxf"))]
    payload = {"count": len(records), "passed": sum(item["passed"] for item in records), "records": records}
    output = args.output or args.directory.parent / "validation_report.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validated={len(records)} passed={payload['passed']} report={output.resolve()}")
    if payload["passed"] != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
