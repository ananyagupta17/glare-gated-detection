"""Synthesise daytime sun glare on half of the fixed KITTI subset, using a camera model.

The glare is built from how a real lens and sensor actually respond to the sun, because
the shortcuts that look reasonable in isolation are exactly what make synthetic glare
read as a decal pasted onto the frame.

WHY NOT A GAUSSIAN

A Gaussian falls off as exp(-r^2). Real veiling glare -- light scattered by the lens
elements, the sensor cover glass and the windscreen -- falls off as a POWER LAW, roughly
1/r^2 to 1/r^3. The difference is not cosmetic. At four radii out a Gaussian has decayed
to ~1e-4 and is gone; a power law is still at ~1e-2 and still visibly lifting the frame.
That long tail is the entire reason a real sun bleeds into the whole image instead of
stopping at a circular boundary. Using a Gaussian and then raising the peak to compensate
produces precisely the artifact this generator used to produce: a hard-edged white ball
with nothing around it.

So every lobe here is a Moffat profile, the standard model for an optical point-spread
function:

    I(r) = amp * (1 + (r/alpha)^2) ^ -beta

beta controls tail weight (lower = heavier). It is the same family astronomers fit to
stars, for the same reason: it is what a point source through real glass looks like.

WHAT MAKES IT LOOK PHOTOGRAPHED

  * Multi-scale. One profile cannot be both a tight disc and a frame-wide wash, because a
    real lens PSF is not one scale either. Each type stacks a narrow core, a mid glow and
    a very wide veil, which is what produces a continuous falloff with no visible shoulder.

  * The disc is never drawn. The core amplitude is 60-200x full scale, because the sun
    genuinely is ~1e5 times brighter than the road. The saturated disc is then just the
    contour where that profile crosses 1.0 after clipping -- an emergent size, surrounded
    by its own bloom, rather than a flat-topped stamp with a drawn edge.

  * Chromatic halo. Each channel gets its own alpha (red widest, blue narrowest), since
    scattering is wavelength-dependent. The core saturates in all three channels and stays
    white, while the surround reddens with radius on its own. Real sun glare goes
    white -> yellow -> orange outward, and this is the mechanism that does it.

  * Black lift. Scattered light reaches the whole sensor, not just near the sun, so a
    small floor is added frame-wide. This crushes the blacks and washes out contrast in
    shadows far from the source -- the give-away of a real glared frame, and the thing a
    localised blob can never reproduce.

  * The sun goes where the sky is. Position is sampled weighted by local brightness of
    the clean frame (luminance^3), so the source lands in open sky instead of halfway
    down a dark treeline or on top of a car. Occlusion is the cue that most cheaply
    destroys realism, and with no depth map this is the closest honest approximation.

  * CCD smear. KITTI was shot on Point Grey Flea2 CCDs, which bleed charge vertically out
    of a saturated well into the readout column. A faint vertical band through the sun is
    a genuine artifact of that sensor class, so direct_sun gets a subtle one.

THE FOUR TYPES

  diffuse_haze    Sun high behind haze. No disc: a broad, soft, near-white glow.
  direct_sun      Sun low and in frame. Saturated core, chromatic halo, smear.
  veiling_wash    Scatter with the sun near or off frame. Very heavy tails, low peak,
                  strongest black lift. Drains contrast rather than blowing out a spot.
  horizon_streak  Low sun smeared horizontally by the windscreen. Anisotropic and warm.

Compositing is additive in LINEAR light, and clipped only once, at the very end. Both
matter: light adds linearly, so adding in gamma space gives a flat grey decal; and
clipping early throws away the blown-out core that makes glare look like glare.

This is deliberately NOT a nighttime lens-flare model -- no ghosts, no starbursts. The
phenomenon is daytime large-area overexposure.

Exactly half the subset is glared, which is the point: if every evaluation frame were
glared the gate would always fire and the C3 arm would be byte-identical to C2. The
50/50 split is what makes trigger rate a measurable quantity.

The evaluation set is assembled per image ID using glare_labels.csv:
    is_glared == 1  ->  data/glared/<id>.png
    is_glared == 0  ->  data/clean/<id>.png
Labels in data/labels/ apply unchanged to both, since glare does not move objects.

Determinism: every random choice derives from --seed (default 42). Each image's
parameters derive from a per-image sub-seed, so an image's glare never depends on
processing order. The type assignment is balanced across the glared set, so it is a
function of (seed, n_glared) rather than of the image alone.

Usage:
    python src/glare_synth.py                  # 200 of 400 glared, seed 42
    python src/glare_synth.py --dry-run        # report the plan, write nothing
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent

GAMMA = 2.2  # sRGB <-> linear approximation


@dataclass(frozen=True)
class Lobe:
    """One Moffat light lobe: I(r) = amp * (1 + (r/alpha)^2) ^ -beta, in linear RGB."""
    cx: float
    cy: float
    alpha: float                       # radius scale in px
    beta: float                        # tail weight; lower = reaches further
    amp: float                         # linear-light peak; >1 saturates on purpose
    tint: tuple[float, float, float]   # per-channel amplitude
    chroma: tuple[float, float, float] = (1.0, 1.0, 1.0)   # per-channel alpha scale
    aspect: float = 1.0                # sy/sx; <1 flattens into a streak
    theta: float = 0.0                 # rotation, radians
    vertical: bool = False             # ignore dy -> full-height band (CCD smear)


@dataclass(frozen=True)
class GlareSpec:
    type_name: str
    lobes: list[Lobe] = field(default_factory=list)
    floor: float = 0.0                 # frame-wide scattered light (black lift)

    def describe(self) -> str:
        core = self.lobes[0]
        return (f"{self.type_name}   centre=({core.cx:.0f},{core.cy:.0f})  "
                f"alpha={core.alpha:.0f}px beta={core.beta:.2f}  "
                f"amp={core.amp:.1f} ({len(self.lobes)} lobes)  "
                f"floor={self.floor:.4f}")


def _warm(rng: random.Random, g: tuple[float, float], b: tuple[float, float]):
    return (1.0, rng.uniform(*g), rng.uniform(*b))


# --- per-type samplers ---------------------------------------------------------------
# Each stacks lobes at different scales. Amplitudes are in linear light, where 1.0 is
# full scale, so core amps in the tens or hundreds are not a mistake -- that is the
# actual dynamic range between the sun and a road surface.

def _diffuse_haze(rng: random.Random) -> tuple[list[Lobe], float]:
    """Sun high behind haze: no disc at all, just a broad soft near-white glow."""
    a = rng.uniform(150.0, 320.0)
    return [
        Lobe(0, 0, alpha=a, beta=rng.uniform(1.05, 1.45),
             amp=rng.uniform(0.45, 0.95),
             tint=_warm(rng, (0.94, 0.99), (0.88, 0.97)),
             chroma=(1.0, 0.95, 0.90)),
        Lobe(0, 0, alpha=a * rng.uniform(2.5, 4.0), beta=rng.uniform(0.85, 1.10),
             amp=rng.uniform(0.10, 0.22),
             tint=_warm(rng, (0.90, 0.97), (0.82, 0.93)),
             chroma=(1.0, 0.92, 0.84)),
    ], rng.uniform(0.002, 0.007)


def _direct_sun(rng: random.Random) -> tuple[list[Lobe], float]:
    """Sun low and in frame: saturated core, chromatic halo, wide veil, CCD smear."""
    core_a = rng.uniform(4.0, 9.5)
    return [
        # Core: amp far past full scale. The visible disc is where this crosses 1.0.
        Lobe(0, 0, alpha=core_a, beta=rng.uniform(1.7, 2.4),
             amp=rng.uniform(60.0, 200.0),
             tint=_warm(rng, (0.96, 1.00), (0.90, 0.98)),
             chroma=(1.0, 0.96, 0.92)),
        # Glow: bridges core to veil so there is no shoulder between them.
        Lobe(0, 0, alpha=rng.uniform(26.0, 62.0), beta=rng.uniform(1.30, 1.75),
             amp=rng.uniform(1.5, 5.0),
             tint=_warm(rng, (0.82, 0.92), (0.60, 0.76)),
             chroma=(1.0, 0.90, 0.79)),
        # Veil: heavy tail, carries the glare across the frame.
        Lobe(0, 0, alpha=rng.uniform(130.0, 300.0), beta=rng.uniform(0.85, 1.20),
             amp=rng.uniform(0.16, 0.40),
             tint=_warm(rng, (0.68, 0.82), (0.42, 0.60)),
             chroma=(1.0, 0.87, 0.74)),
        # CCD charge bleed up and down the readout column. Subtle by design.
        Lobe(0, 0, alpha=core_a * rng.uniform(1.1, 2.0), beta=rng.uniform(1.0, 1.5),
             amp=rng.uniform(0.05, 0.16),
             tint=_warm(rng, (0.92, 0.99), (0.86, 0.96)),
             vertical=True),
    ], rng.uniform(0.004, 0.013)


def _veiling_wash(rng: random.Random) -> tuple[list[Lobe], float]:
    """Scatter with the sun near or off frame: contrast drains out of everything."""
    a = rng.uniform(300.0, 620.0)
    return [
        Lobe(0, 0, alpha=a, beta=rng.uniform(0.70, 0.95),
             amp=rng.uniform(0.22, 0.50),
             tint=_warm(rng, (0.82, 0.94), (0.68, 0.86)),
             chroma=(1.0, 0.91, 0.82),
             aspect=rng.uniform(0.7, 1.3), theta=rng.uniform(-0.4, 0.4)),
    ], rng.uniform(0.010, 0.026)          # the strongest black lift of the four


def _horizon_streak(rng: random.Random) -> tuple[list[Lobe], float]:
    """Low sun smeared horizontally by the windscreen: anisotropic and strongly warm."""
    a = rng.uniform(200.0, 430.0)
    th = math.radians(rng.uniform(-9.0, 9.0))
    asp = rng.uniform(0.10, 0.26)
    return [
        Lobe(0, 0, alpha=a, beta=rng.uniform(1.15, 1.70),
             amp=rng.uniform(2.0, 9.0),                    # saturates along the band
             tint=_warm(rng, (0.78, 0.90), (0.54, 0.72)),
             chroma=(1.0, 0.90, 0.80), aspect=asp, theta=th),
        Lobe(0, 0, alpha=a * rng.uniform(1.6, 2.6), beta=rng.uniform(0.85, 1.15),
             amp=rng.uniform(0.14, 0.34),
             tint=_warm(rng, (0.70, 0.84), (0.46, 0.64)),
             chroma=(1.0, 0.88, 0.76), aspect=asp * rng.uniform(1.5, 2.5), theta=th),
    ], rng.uniform(0.006, 0.016)


GLARE_TYPES = {
    "diffuse_haze": _diffuse_haze,
    "direct_sun": _direct_sun,
    "veiling_wash": _veiling_wash,
    "horizon_streak": _horizon_streak,
}

# Where each type's source may sit, as a fraction of image height, and whether the
# position is drawn toward bright pixels. A sun behind haze or in open sky belongs in
# sky; scatter from an off-frame sun does not, so it is placed freely.
TYPE_PLACEMENT = {
    "diffuse_haze": dict(y_range=(0.00, 0.34), sky_weighted=True, off_frame=False),
    "direct_sun": dict(y_range=(0.08, 0.46), sky_weighted=True, off_frame=False),
    "veiling_wash": dict(y_range=(-0.10, 0.42), sky_weighted=False, off_frame=True),
    "horizon_streak": dict(y_range=(0.18, 0.50), sky_weighted=True, off_frame=False),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clean-dir", type=Path, default=REPO_ROOT / "data" / "clean")
    p.add_argument("--labels-dir", type=Path, default=REPO_ROOT / "data" / "labels")
    p.add_argument("--glared-dir", type=Path, default=REPO_ROOT / "data" / "glared")
    p.add_argument("--ids-file", type=Path, default=REPO_ROOT / "data" / "subset_ids.txt")
    p.add_argument("--csv-out", type=Path, default=REPO_ROOT / "data" / "glare_labels.csv")
    p.add_argument("--results-dir", type=Path, default=REPO_ROOT / "results")
    p.add_argument("--n-glared", type=int, default=200,
                   help="how many images to glare (default: %(default)s)")
    p.add_argument("--examples", type=int, default=len(GLARE_TYPES),
                   help="comparison figures; at least one per type (default: %(default)s)")
    p.add_argument("--only", type=str, default=None,
                   help="restrict to one glare type, for visual tuning")
    p.add_argument("--seed", type=int, default=42, help="master seed (default: %(default)s)")
    p.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    return p.parse_args(argv)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_subset(clean_dir: Path, labels_dir: Path, ids_file: Path) -> list[tuple[str, Path]]:
    """Collect (image_id, image path) pairs, checking them against the committed contract."""
    if not clean_dir.is_dir():
        sys.exit(f"error: {rel(clean_dir)} not found. Run src/prepare_subset.py first.")

    images = sorted(p for p in clean_dir.iterdir() if p.suffix.lower() == ".png")
    if not images:
        sys.exit(f"error: no PNGs in {rel(clean_dir)}. Run src/prepare_subset.py first.")

    missing_labels = [p.stem for p in images if not (labels_dir / f"{p.stem}.txt").is_file()]
    if missing_labels:
        sys.exit(f"error: {len(missing_labels)} image(s) have no label in {rel(labels_dir)} "
                 f"(e.g. {', '.join(missing_labels[:5])})")

    # All three components must run on the same IDs; a silent drift here would break the
    # C1/C2/C3 comparison rather than raise an error, so check it loudly.
    if ids_file.is_file():
        expected = {ln.strip() for ln in ids_file.read_text().splitlines() if ln.strip()}
        found = {p.stem for p in images}
        if expected != found:
            sys.exit(f"error: {rel(clean_dir)} does not match {rel(ids_file)} "
                     f"({len(found - expected)} extra, {len(expected - found)} missing). "
                     f"Re-run src/prepare_subset.py.")

    return [(p.stem, p) for p in images]


def assign_types(glared_ids: set[str], seed: int, only: str | None) -> dict[str, str]:
    """Balanced type assignment, so every type gets ~n/4 images for the gate to learn."""
    ids = sorted(glared_ids)
    if only:
        return {i: only for i in ids}
    types = list(itertools.islice(itertools.cycle(GLARE_TYPES), len(ids)))
    random.Random(seed + 7).shuffle(types)
    return dict(zip(ids, types))


def place_source(rng: random.Random, clean: np.ndarray, type_name: str) -> tuple[float, float]:
    """Pick the source position; sky-weighted types are drawn toward bright pixels.

    Sampling proportional to luminance^3 within the type's vertical band puts the sun in
    open sky rather than against a dark treeline, which is the cheapest large win in
    realism available without a depth map.
    """
    h, w = clean.shape[:2]
    place = TYPE_PLACEMENT[type_name]
    y_lo, y_hi = (f * h for f in place["y_range"])

    if place["off_frame"]:
        return rng.uniform(-0.25 * w, 1.25 * w), rng.uniform(y_lo, y_hi)

    band_lo, band_hi = max(0, int(y_lo)), max(1, min(h, int(y_hi)))
    step = 4                                   # subsample; we only need a coarse position
    band = clean[band_lo:band_hi:step, ::step].astype(np.float32).mean(axis=2)
    weights = np.power(band / 255.0, 3.0).ravel()   # ^3 strongly favours sky over road

    total = float(weights.sum())
    if not np.isfinite(total) or total <= 1e-6:     # tunnel/night frame: nothing bright
        return rng.uniform(0, w - 1), rng.uniform(y_lo, y_hi)

    idx = int(np.searchsorted(np.cumsum(weights), rng.random() * total))
    idx = min(idx, weights.size - 1)
    row, col = divmod(idx, band.shape[1])
    # jitter within the subsample cell so positions are not quantised to the grid
    return (col * step + rng.uniform(0, step),
            band_lo + row * step + rng.uniform(0, step))


def build_spec(image_id: str, type_name: str, clean: np.ndarray, seed: int) -> GlareSpec:
    """Per-image RNG so an image's parameters never depend on processing order."""
    rng = random.Random(f"{seed}:{image_id}")
    lobes, floor = GLARE_TYPES[type_name](rng)
    cx, cy = place_source(rng, clean, type_name)
    return GlareSpec(type_name,
                     [Lobe(**{**lobe.__dict__, "cx": cx, "cy": cy}) for lobe in lobes],
                     floor)


def apply_glare(rgb: np.ndarray, spec: GlareSpec) -> np.ndarray:
    """uint8 sRGB in, uint8 sRGB out. Sums lobes in linear light and clips once, last."""
    height, width = rgb.shape[:2]

    linear = (rgb.astype(np.float32) / 255.0) ** GAMMA

    ys = np.arange(height, dtype=np.float32)[:, None]
    xs = np.arange(width, dtype=np.float32)[None, :]

    glare = np.full_like(linear, 0.0)
    for lobe in spec.lobes:
        dx = xs - lobe.cx
        dy = ys - lobe.cy
        if lobe.theta:
            cos_t, sin_t = math.cos(lobe.theta), math.sin(lobe.theta)
            dx, dy = dx * cos_t + dy * sin_t, -dx * sin_t + dy * cos_t
        # Per-channel alpha: red scatters widest, so the halo reddens outward by itself
        # while the saturated core stays white.
        for ch in range(3):
            a = lobe.alpha * lobe.chroma[ch]
            r2 = (dx / a) ** 2
            if not lobe.vertical:
                r2 = r2 + (dy / (a * lobe.aspect)) ** 2
            glare[:, :, ch] += lobe.amp * lobe.tint[ch] * np.power(1.0 + r2, -lobe.beta)

    # Scattered light reaches the whole sensor, not just the neighbourhood of the sun.
    # This lifts the blacks everywhere and is what washes out shadow contrast.
    glare += spec.floor

    # Clip only now: the flat region this creates is the blown-out core, and its size is
    # emergent from the profile rather than drawn.
    combined = np.clip(linear + glare, 0.0, 1.0)

    return np.rint((combined ** (1.0 / GAMMA)) * 255.0).astype(np.uint8)


def save_comparison(out_path: Path, image_id: str, clean: np.ndarray,
                    glared: np.ndarray, spec: GlareSpec) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7))
    for ax, img, title in ((axes[0], clean, f"{image_id} — clean"),
                           (axes[1], glared, f"{image_id} — {spec.type_name}")):
        ax.imshow(img)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    fig.suptitle(spec.describe(), fontsize=8, y=0.03)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def choose_examples(types_by_id: dict[str, str], count: int, seed: int) -> dict[str, int]:
    """Cover every type first, then fill remaining slots, so the figures show the range."""
    rng = random.Random(seed + 1)
    by_type: dict[str, list[str]] = {}
    for image_id, type_name in sorted(types_by_id.items()):
        by_type.setdefault(type_name, []).append(image_id)

    picked = [rng.choice(members) for _, members in sorted(by_type.items())][:count]
    if len(picked) < count:
        rest = sorted(set(types_by_id) - set(picked))
        picked += rng.sample(rest, min(count - len(picked), len(rest)))

    # Number the figures by type so glare_example_N is stable across runs.
    picked.sort(key=lambda i: (types_by_id[i], i))
    return {image_id: slot for slot, image_id in enumerate(picked)}


def clear_stale(directory: Path, keep: set[str]) -> int:
    stale = [f for f in directory.iterdir()
             if f.is_file() and f.name != ".gitkeep" and f.stem not in keep]
    for f in stale:
        f.unlink()
    return len(stale)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.only and args.only not in GLARE_TYPES:
        sys.exit(f"error: --only must be one of {', '.join(GLARE_TYPES)}")

    subset = load_subset(args.clean_dir, args.labels_dir, args.ids_file)
    ids = [image_id for image_id, _ in subset]
    paths = dict(subset)
    print(f"subset: {len(ids)} images in {rel(args.clean_dir)} "
          f"(labels verified in {rel(args.labels_dir)})")

    if args.n_glared > len(ids):
        sys.exit(f"error: asked to glare {args.n_glared} of only {len(ids)} images")

    # Selection derives from the master seed alone, so the split is identical for everyone.
    glared_ids = set(random.Random(args.seed).sample(ids, args.n_glared))
    types_by_id = assign_types(glared_ids, args.seed, args.only)
    print(f"  glaring {len(glared_ids)}, leaving {len(ids) - len(glared_ids)} clean "
          f"(seed {args.seed})" + (f"  [--only {args.only}]" if args.only else ""))
    for type_name in GLARE_TYPES:
        n = sum(1 for t in types_by_id.values() if t == type_name)
        if n:
            print(f"    {type_name:<15} {n:>4}")

    if args.dry_run:
        print("\n  --dry-run, nothing written. One sample of each type in play:")
        for type_name in GLARE_TYPES:
            match = [i for i, t in sorted(types_by_id.items()) if t == type_name]
            if not match:
                continue
            with Image.open(paths[match[0]]) as im:
                clean = np.asarray(im.convert("RGB"))
            print(f"    {match[0]}  "
                  f"{build_spec(match[0], type_name, clean, args.seed).describe()}")
        return 0

    args.glared_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    removed = clear_stale(args.glared_dir, glared_ids)
    if removed:
        print(f"  removed {removed} stale file(s) from a previous run")

    example_slots = choose_examples(types_by_id, args.examples, args.seed)
    for old in args.results_dir.glob("glare_example_*.png"):
        # A shorter run must not leave figures from a longer one lying around.
        if int(old.stem.rsplit("_", 1)[1]) >= len(example_slots):
            old.unlink()

    rows: list[tuple[str, int, str]] = []
    written = 0
    for image_id, path in subset:
        if image_id not in glared_ids:
            rows.append((image_id, 0, ""))
            continue

        with Image.open(path) as im:
            clean = np.asarray(im.convert("RGB"))

        spec = build_spec(image_id, types_by_id[image_id], clean, args.seed)
        glared = apply_glare(clean, spec)

        Image.fromarray(glared).save(args.glared_dir / f"{image_id}.png")
        written += 1
        rows.append((image_id, 1, spec.type_name))

        if image_id in example_slots:
            out = args.results_dir / f"glare_example_{example_slots[image_id]}.png"
            save_comparison(out, image_id, clean, glared, spec)

    print(f"\n  wrote {written} glared images -> {rel(args.glared_dir)}/")

    with args.csv_out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_id", "is_glared", "glare_type"])
        writer.writerows(rows)
    print(f"  wrote {rel(args.csv_out)}  ({len(rows)} rows)  <-- commit this")
    print(f"  wrote {len(example_slots)} comparison figure(s) -> {rel(args.results_dir)}/")

    print("\nNext (build order): src/detect.py -> YOLOv8 inference + latency, then the C1 arm.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
