"""Inspect DXF structure before running the CV processing pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _bootstrap_vendor() -> None:
    vendor = Path(__file__).resolve().parents[1] / "vendor"
    if vendor.exists():
        sys.path.insert(0, str(vendor))


_bootstrap_vendor()

import ezdxf  # noqa: E402
from ezdxf import bbox  # noqa: E402


TEXT_TYPES = {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}


def inspect_file(path: Path) -> dict:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    entities = Counter(entity.dxftype() for entity in msp)
    layers: dict[str, Counter] = defaultdict(Counter)
    texts: list[str] = []
    inserts: list[str] = []

    for entity in msp:
        kind = entity.dxftype()
        layer = getattr(entity.dxf, "layer", "")
        layers[layer][kind] += 1
        if kind in TEXT_TYPES:
            if kind == "MTEXT":
                value = entity.plain_text()
            else:
                value = getattr(entity.dxf, "text", "")
            if value.strip():
                texts.append(value.strip())
        elif kind == "INSERT":
            inserts.append(entity.dxf.name)
            for attrib in entity.attribs:
                if attrib.dxf.text.strip():
                    texts.append(attrib.dxf.text.strip())

    ext = bbox.extents(msp, fast=True)
    return {
        "file": path.name,
        "version": doc.dxfversion,
        "entities": dict(entities),
        "layers": {name: dict(counts) for name, counts in layers.items()},
        "insert_blocks": inserts,
        "extents": {
            "min": list(ext.extmin) if ext.has_data else None,
            "max": list(ext.extmax) if ext.has_data else None,
            "size": list(ext.size) if ext.has_data else None,
        },
        "texts": texts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default=".")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(args.directory)
    records = [inspect_file(p) for p in sorted(root.glob("*.dxf"))]
    payload = json.dumps(records, ensure_ascii=False, indent=2)
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
