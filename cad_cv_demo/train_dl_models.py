"""Train the deep-learning detector and matcher used by the CV pipeline.

Outputs (default ``cad_cv_demo/models/``):
- ``matcher.pt`` / ``detector.pt``: model weights.
- ``training_loss.png``: loss curves for both models.
- ``detector_samples.png``: synthetic detector training tiles with boxes.
- ``matcher_samples.png``: augmented silhouette view pairs.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cad_cv_demo.shape_match_diagnostics import (  # noqa: E402
    Point,
    components,
    primitives_from_dwg,
    primitives_from_dxf,
)
from cad_cv_demo.cad_cv_pipeline import (  # noqa: E402
    _candidate_pool,
    _component_primitives,
    infer_scale_from_dimensions,
    match_score,
)
from cad_cv_demo.dl_raster import draw_polylines, primitive_polylines, rasterize_scene  # noqa: E402
from cad_cv_demo.dl_detector import (  # noqa: E402
    PIX_PER_UNIT,
    TILE,
    SyntheticTileGenerator,
    UNetSegmenter,
    _downsample_mask,
    pick_device,
    save_detector,
    tile_origins,
    train_steps,
)
from cad_cv_demo.dl_matcher import (  # noqa: E402
    ShapeEncoder,
    augment_silhouette,
    save_matcher,
    train_matcher,
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def load_reference_outlines(reference_dir: Path) -> list[dict]:
    """Per reference: outer-contour polylines plus inner hole polylines."""
    references: list[dict] = []
    for path in sorted(reference_dir.glob("*.dxf")):
        all_primitives = primitives_from_dxf(path)
        comps = [comp for comp in components(all_primitives, tolerance=0.02) if comp["perimeter"] > 10]
        if not comps:
            continue
        outer = max(comps, key=lambda comp: (comp["width"] * comp["height"], comp["perimeter"]))
        outer_primitives = _component_primitives(all_primitives, outer)
        holes = [
            prim
            for comp in comps
            if comp is not outer and comp["perimeter"] > 1
            for prim in _component_primitives(all_primitives, comp)
        ]
        references.append(
            {
                "code": path.stem,
                "outer": primitive_polylines(outer_primitives),
                "full": primitive_polylines(outer_primitives + holes),
                "record": (outer, outer_primitives),
            }
        )
    return references


def heuristic_matched_bboxes(source: Path, reference_dir: Path, scale: float, threshold: float) -> tuple[list[tuple[float, float, float, float]], list[list[list[Point]]], list]:
    """Re-run the heuristic matching to harvest pseudo-label boxes on the DWG.

    Returns (matched candidate bboxes, matched outline polylines in model
    coordinates, all DWG primitives). The polylines give pixel-precise
    positive supervision for detector fine-tuning.
    """
    dwg_primitives = primitives_from_dwg(source)
    candidates = _candidate_pool(dwg_primitives, scale)
    references = load_reference_outlines(reference_dir)
    pairs: list[tuple[float, int, int]] = []
    for ref_index, reference in enumerate(references):
        outer, outer_primitives = reference["record"]
        for candidate_index, candidate in enumerate(candidates):
            score = match_score(candidate, _component_primitives(dwg_primitives, candidate), outer, outer_primitives, scale)
            pairs.append((score, ref_index, candidate_index))
    pairs.sort()
    assigned_refs: set[int] = set()
    assigned_candidates: set[int] = set()
    boxes: list[tuple[float, float, float, float]] = []
    outlines: list[list[list[Point]]] = []
    for score, ref_index, candidate_index in pairs:
        if score > threshold:
            break
        if ref_index in assigned_refs or candidate_index in assigned_candidates:
            continue
        assigned_refs.add(ref_index)
        assigned_candidates.add(candidate_index)
        candidate = candidates[candidate_index]
        boxes.append(candidate["bbox"])
        outlines.append(primitive_polylines(_component_primitives(dwg_primitives, candidate)))
    return boxes, outlines, dwg_primitives


class RealTileSampler:
    """Real sheet tiles with pixel-precise labels from matched outline polylines."""

    def __init__(self, primitives: list, outlines_model: list[list[list[Point]]], seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        scene, transform = rasterize_scene(primitives, pix_per_unit=PIX_PER_UNIT)
        height, width = scene.shape
        pos_full = np.zeros((height, width), dtype=np.uint8)
        for lines in outlines_model:
            draw_polylines(pos_full, lines, transform.to_pixel, 2)
        self.tiles: list[tuple[np.ndarray, np.ndarray]] = []
        for oy in tile_origins(height):
            for ox in tile_origins(width):
                view = np.zeros((1, TILE, TILE), dtype=np.float32)
                chunk = scene[oy : oy + TILE, ox : ox + TILE]
                view[0, : chunk.shape[0], : chunk.shape[1]] = chunk
                mask_full = np.zeros((TILE, TILE), dtype=np.float32)
                mchunk = pos_full[oy : oy + TILE, ox : ox + TILE]
                mask_full[: mchunk.shape[0], : mchunk.shape[1]] = (mchunk > 0)
                mask = _downsample_mask(mask_full)
                self.tiles.append((view, mask[None]))

    def sample(self):
        return self.tiles[int(self.rng.integers(0, len(self.tiles)))]


# ---------------------------------------------------------------------------
# Visualization artifacts
# ---------------------------------------------------------------------------

def _cv_write(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(path.suffix or ".png", image)[1].tofile(path)


def save_detector_samples(generator: SyntheticTileGenerator, path: Path, count: int = 4) -> None:
    tiles = []
    for _ in range(count):
        img, part_img, boxes = generator.compose()
        vis = (img * 255).astype(np.uint8)
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        vis[(part_img > 0)] = (0, 0, 255)
        for cx, cy, w, h in boxes:
            cv2.rectangle(vis, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (40, 200, 40), 2)
        tiles.append(cv2.resize(vis, (512, 512), interpolation=cv2.INTER_AREA))
    sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    _cv_write(path, sheet)


def save_matcher_samples(class_outlines: list[list[list[Point]]], path: Path, count: int = 6) -> None:
    rng = np.random.default_rng(7)
    rows = []
    for lines in class_outlines[:count]:
        pair = [augment_silhouette(lines, rng) for _ in range(2)]
        rows.append(np.hstack([(p * 255).astype(np.uint8) for p in pair]))
    _cv_write(path, np.vstack(rows))


def save_loss_curves(histories: dict[str, list[float]], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(histories), figsize=(5 * len(histories), 3.4))
    for ax, (name, history) in zip(np.atleast_1d(axes), histories.items()):
        ax.plot(history, linewidth=0.7)
        if len(history) > 20:
            kernel = np.ones(20) / 20
            ax.plot(np.convolve(history, kernel, mode="valid"), linewidth=1.4)
        ax.set_title(name)
        ax.set_xlabel("step")
        ax.set_ylabel("loss")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _stage(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "test.dwg")
    parser.add_argument("--reference-dir", type=Path, default=ROOT)
    parser.add_argument("--models-dir", type=Path, default=ROOT / "cad_cv_demo" / "models")
    parser.add_argument("--matcher-steps", type=int, default=2400)
    parser.add_argument("--detector-steps", type=int, default=6000)
    parser.add_argument("--finetune-steps", type=int, default=800)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--match-threshold", type=float, default=0.08)
    args = parser.parse_args()

    device = pick_device()
    _stage(f"device={device}; loading references from {args.reference_dir}")
    references = load_reference_outlines(args.reference_dir)
    _stage(f"references={len(references)}; start matcher training ({args.matcher_steps} steps)")

    # 1. Matcher (embedding network)
    started = time.time()
    matcher = ShapeEncoder()
    matcher_history = train_matcher(
        matcher,
        [ref["outer"] for ref in references],
        device,
        steps=args.matcher_steps,
        batch_classes=64,
    )
    save_matcher(matcher, args.models_dir / "matcher.pt")
    _stage(f"matcher done loss={np.mean(matcher_history[-50:]):.4f} elapsed={time.time() - started:.0f}s; start detector pretraining ({args.detector_steps} steps)")

    # 2. Detector pre-training on synthetic tiles
    started = time.time()
    generator = SyntheticTileGenerator([ref["full"] for ref in references], seed=1)
    detector = UNetSegmenter()
    detector_history = train_steps(detector, generator.sample, args.detector_steps, device, lr=1e-3, batch_size=args.batch_size, prefix="detector")
    _stage(f"detector pretrain done loss={np.mean(detector_history[-50:]):.4f} elapsed={time.time() - started:.0f}s; mining pseudo-labels")

    # 3. Fine-tune on real tiles with pixel-precise matched-outline labels
    started = time.time()
    scale, _ = infer_scale_from_dimensions(args.source)
    boxes, outlines, dwg_primitives = heuristic_matched_bboxes(args.source, args.reference_dir, scale, args.match_threshold)
    _stage(f"pseudo_labels={len(boxes)} scale={scale:g}; start finetune ({args.finetune_steps} steps)")
    finetune_history: list[float] = []
    if outlines:
        real_sampler = RealTileSampler(dwg_primitives, outlines)
        rng = np.random.default_rng(3)
        finetune_history = train_steps(
            detector,
            lambda: real_sampler.sample() if rng.random() < 0.6 else generator.sample(),
            args.finetune_steps,
            device,
            lr=1e-4,
            batch_size=args.batch_size,
            prefix="finetune",
        )
        _stage(f"finetune done loss={np.mean(finetune_history[-50:]):.4f} elapsed={time.time() - started:.0f}s")
    save_detector(detector, args.models_dir / "detector.pt")

    # 4. Visual artifacts
    _stage("saving loss curves and sample sheets")
    save_loss_curves(
        {"matcher": matcher_history, "detector": detector_history + finetune_history},
        args.models_dir / "training_loss.png",
    )
    save_detector_samples(generator, args.models_dir / "detector_samples.png")
    save_matcher_samples([ref["outer"] for ref in references], args.models_dir / "matcher_samples.png")
    _stage(f"models saved to {args.models_dir.resolve()}")


if __name__ == "__main__":
    main()
