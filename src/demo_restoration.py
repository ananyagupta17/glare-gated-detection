"""One-command Flare7K++ glare-removal demonstration.

Examples:
    python src/demo_restoration.py
    python src/demo_restoration.py --input path/to/glared.png --clean path/to/clean.png
    python src/demo_restoration.py --input data/glared_full --clean datasets/kitti/images/val
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np

from restore import DEFAULT_CHECKPOINT, DEFAULT_FLARE7K_REPO, Flare7KRestorer


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and visualize Flare7K++ glare removal")
    parser.add_argument("--input", type=Path, help="Glared image or directory")
    parser.add_argument("--clean", type=Path, help="Optional paired clean image or directory")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "restoration_demo")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--flare7k-repo", type=Path, default=DEFAULT_FLARE7K_REPO)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0")
    parser.add_argument("--max-images", type=int, default=1)
    parser.add_argument("--overlap", type=int, default=128)
    return parser.parse_args()


def read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {path}")
    return image


def extract_embedded_example(figure: Path, output_dir: Path) -> tuple[Path, Path]:
    """Recover clean/glared panels from glare_synth.py's two-row example figure."""
    image = read_image(figure)
    height, width = image.shape[:2]
    # Matplotlib save_comparison has two image panels separated by white title bands.
    nonwhite = np.any(image < 248, axis=2)
    row_coverage = nonwhite.mean(axis=1)
    active = row_coverage > 0.45
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(active):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if index - start > 40:
                runs.append((start, index))
            start = None
    if start is not None and height - start > 40:
        runs.append((start, height))
    if len(runs) < 2:
        raise RuntimeError(f"Could not identify two image panels in {figure}")
    runs = sorted(runs, key=lambda pair: pair[1] - pair[0], reverse=True)[:2]
    runs.sort()
    panels = []
    for top, bottom in runs:
        rows = image[top:bottom]
        col_coverage = np.any(rows < 248, axis=2).mean(axis=0)
        columns = np.flatnonzero(col_coverage > 0.45)
        left, right = int(columns[0]), int(columns[-1] + 1)
        panels.append(image[top:bottom, left:right])
    target_h = min(panel.shape[0] for panel in panels)
    target_w = min(panel.shape[1] for panel in panels)
    panels = [panel[:target_h, :target_w] for panel in panels]
    clean_path = output_dir / "embedded_clean.png"
    glared_path = output_dir / "embedded_glared.png"
    cv2.imwrite(str(clean_path), panels[0])
    cv2.imwrite(str(glared_path), panels[1])
    return glared_path, clean_path


def collect_inputs(path: Path, limit: int) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input does not exist: {path}")
    files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        raise RuntimeError(f"No images found in {path}")
    return files[:limit]


def paired_clean(glared: Path, clean_root: Path | None) -> Path | None:
    if clean_root is None:
        return None
    if clean_root.is_file():
        return clean_root
    exact = clean_root / glared.name
    if exact.is_file():
        return exact
    for suffix in IMAGE_SUFFIXES:
        candidate = clean_root / f"{glared.stem}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    error = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if error == 0 else float(20 * np.log10(255.0 / np.sqrt(error)))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Standard channel-averaged SSIM with an 11x11 Gaussian window."""
    x, y = reference.astype(np.float64), test.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_x = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_y = cv2.GaussianBlur(y, (11, 11), 1.5)
    sigma_x = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu_x * mu_x
    sigma_y = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu_y * mu_y
    sigma_xy = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x + sigma_y + c2)
    )
    return float(np.mean(score))


def fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def make_comparison(glared: np.ndarray, restored: np.ndarray, clean: np.ndarray | None,
                    runtime_ms: float, metrics: dict) -> np.ndarray:
    items = [("GLARED INPUT", glared), ("RESTORED (UFORMER)", restored)]
    if clean is not None:
        items.append(("CLEAN REFERENCE", clean))
    panel_w, panel_h, header, footer = 560, 360, 48, 62
    canvas = np.full((header + panel_h + footer, panel_w * len(items), 3), 255, np.uint8)
    for index, (label, image) in enumerate(items):
        x = index * panel_w
        canvas[header:header + panel_h, x:x + panel_w] = fit_panel(image, panel_w, panel_h)
        cv2.putText(canvas, label, (x + 18, 32), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (32, 32, 32), 2, cv2.LINE_AA)
    message = f"Inference: {runtime_ms:.1f} ms | device: {metrics['device']}"
    if "restored_psnr_db" in metrics:
        message += (f" | PSNR {metrics['input_psnr_db']:.2f} -> "
                    f"{metrics['restored_psnr_db']:.2f} dB | SSIM "
                    f"{metrics['input_ssim']:.4f} -> {metrics['restored_ssim']:.4f}")
    cv2.putText(canvas, message, (18, header + panel_h + 39), cv2.FONT_HERSHEY_SIMPLEX,
                0.57, (32, 32, 32), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    args = parse_args()
    if args.max_images < 1:
        raise SystemExit("--max-images must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_path, clean_path = args.input, args.clean
    if input_path is None:
        figure = ROOT / "results" / "glare_full_example_0.png"
        if not figure.is_file():
            raise SystemExit("Give --input because no generated example figure was found")
        input_path, clean_path = extract_embedded_example(figure, args.output_dir)
        print(f"Using embedded project example: {figure.relative_to(ROOT)}")

    inputs = collect_inputs(input_path, args.max_images)
    print(f"Loading Flare7K++ Uformer from: {args.checkpoint}")
    restorer = Flare7KRestorer(
        checkpoint=args.checkpoint,
        flare7k_repo=args.flare7k_repo,
        device=args.device,
        overlap=args.overlap,
    )
    print(f"Model ready on {restorer.device} in {restorer.load_seconds:.2f} s")

    records = []
    for index, path in enumerate(inputs, 1):
        glared = read_image(path)
        reference_path = paired_clean(path, clean_path)
        clean = read_image(reference_path) if reference_path else None
        if clean is not None and clean.shape != glared.shape:
            clean = cv2.resize(clean, (glared.shape[1], glared.shape[0]), interpolation=cv2.INTER_AREA)

        started = perf_counter()
        restored = restorer.restore(glared)
        runtime_ms = (perf_counter() - started) * 1000
        record = {
            "image": path.name,
            "restored_file": f"{path.stem}_restored.png",
            "comparison_file": f"{path.stem}_comparison.png",
            "device": str(restorer.device),
            "runtime_ms": round(runtime_ms, 2),
        }
        if clean is not None:
            record.update({
                "input_psnr_db": round(psnr(clean, glared), 4),
                "restored_psnr_db": round(psnr(clean, restored), 4),
                "input_ssim": round(ssim(clean, glared), 6),
                "restored_ssim": round(ssim(clean, restored), 6),
            })
            record["psnr_change_db"] = round(record["restored_psnr_db"] - record["input_psnr_db"], 4)
            record["ssim_change"] = round(record["restored_ssim"] - record["input_ssim"], 6)

        restored_path = args.output_dir / record["restored_file"]
        comparison_path = args.output_dir / record["comparison_file"]
        cv2.imwrite(str(restored_path), restored)
        cv2.imwrite(str(comparison_path), make_comparison(glared, restored, clean, runtime_ms, record))
        records.append(record)
        print(f"[{index}/{len(inputs)}] {path.name}: {runtime_ms:.1f} ms -> {restored_path}")

    summary = {
        "model": "Flare7K++ Uformer",
        "checkpoint": display_path(args.checkpoint),
        "device": str(restorer.device),
        "model_load_seconds": round(restorer.load_seconds, 3),
        "images": records,
    }
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if records:
        with (args.output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    print(f"Metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
