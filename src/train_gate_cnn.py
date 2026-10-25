"""CNN gate: MobileNetV3-Small fine-tuned on synthetic KITTI glare + REAL glare photos.

WHY THIS EXISTS
The hand-engineered gates (9 and 12 features, src/train_gate_classifier.py) score ~100% on
synthetic glare and miss most real glare: 17% recall on 204 GLARE photos (9 features),
35% on the 200 Esfahani photos (12 features). The cause is the data, not the classifier.
Synthetic glare washes the whole frame out (brighter, greyer, many clipped pixels); real
glare is localised, warm and saturated, with frame-wide brightness close to a clean frame.
Any model fit on synthetic-only data learns the synthesis, so this script mixes in real
glare and is scored on real photos it never saw.

REAL DATA AND THE CAMERA SHORTCUT
Real positives: the Esfahani set (arXiv:2110.06006, 200 photos with glare masks), cloned to
datasets/real_glare_esfahani/ (gitignored):
    git clone --depth 1 https://github.com/maesfahani/glaredetection datasets/real_glare_esfahani
If every real photo were a positive and every negative were KITTI, a CNN would learn
"not a KITTI camera = glare". So the masks are used to cut NEGATIVE crops from the
glare-free parts of the same photos: both classes then share cameras and scenes.
Positive crops are taken around the masked glare region.

NO PEEKING
Real photos are split by photo id (a photo's crops never straddle train/test); KITTI is
split by frame id (a frame's clean and glared versions stay together). Epoch count and
threshold (0.5) are fixed up front, never tuned on the test split.

Data:
    datasets/real_glare_esfahani/RGB/<id>/{images/<id>.jpg, masks/<id>_GT.jpg}
    datasets/kitti/images/val/<id>.png    clean KITTI
    data/glared_full/<id>.png             synthetic glare (v1, data/glare_labels_full.csv)

Integration contract (same as the other gates):
    model = torch.load("models/gate_cnn.pt", weights_only=False)
    is_glared, confidence = model.predict(image)   # image: BGR uint8, cv2.imread output

Usage:
    python src/train_gate_cnn.py --data mixed        # synthetic + real (proposed)
    python src/train_gate_cnn.py --data synthetic    # synthetic only (control)
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

REPO_ROOT = Path(__file__).resolve().parent.parent
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def to_input(image: np.ndarray) -> np.ndarray:
    """BGR uint8 of any shape -> SIZE x SIZE RGB uint8 (squashed, as at runtime)."""
    return cv2.cvtColor(cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA),
                        cv2.COLOR_BGR2RGB)


class GateCNN(nn.Module):
    """Module scope so it unpickles from any script (see GateClassifier)."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.from_numpy(MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.from_numpy(STD).view(1, 3, 1, 1))

    def forward(self, x_uint8: torch.Tensor) -> torch.Tensor:
        x = (x_uint8.float() / 255.0 - self.mean) / self.std
        return self.net(x).squeeze(1)

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> tuple[bool, float]:
        x = torch.from_numpy(to_input(image)).permute(2, 0, 1).unsqueeze(0)
        p = float(torch.sigmoid(self(x))[0])
        return p > 0.5, p


# --- data -----------------------------------------------------------------------------

def square_crops(h: int, w: int, mask: np.ndarray, want_glare: bool, n: int,
                 rng: random.Random) -> list[tuple[int, int, int]]:
    """Up to n square crops (y, x, side) that contain glare (>= 5% of the crop masked)
    or contain none (< 0.1% masked)."""
    out = []
    for _ in range(n * 40):
        if len(out) == n:
            break
        side = int(min(h, w) * rng.uniform(0.35, 0.7))
        y, x = rng.randint(0, h - side), rng.randint(0, w - side)
        frac = float(mask[y:y + side, x:x + side].mean())
        if (want_glare and frac >= 0.05) or (not want_glare and frac < 0.001):
            out.append((y, x, side))
    return out


def real_samples(root: Path, ids: list[str], crops: int, seed: int) -> list[tuple]:
    """(rgb224, label, source) from Esfahani photos: the full photo (positive), glare
    crops (positive) and glare-free crops of the same photo (negative)."""
    rng = random.Random(seed)
    out = []
    for i in ids:
        img = cv2.imread(str(root / i / "images" / f"{i}.jpg"))
        m = cv2.imread(str(root / i / "masks" / f"{i}_GT.jpg"), cv2.IMREAD_GRAYSCALE)
        if img is None or m is None:
            continue
        m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST) > 127
        out.append((to_input(img), 1, "real_full"))
        h, w = m.shape
        for want, label, src in ((True, 1, "real_crop_pos"), (False, 0, "real_crop_neg")):
            for y, x, s in square_crops(h, w, m, want, crops, rng):
                out.append((to_input(img[y:y + s, x:x + s]), label, src))
    return out


def kitti_samples(ids: list[str]) -> list[tuple]:
    out = []
    for i in ids:
        out.append((to_input(cv2.imread(str(REPO_ROOT / "datasets/kitti/images/val" / f"{i}.png"))),
                    0, "kitti_clean"))
        out.append((to_input(cv2.imread(str(REPO_ROOT / "data/glared_full" / f"{i}.png"))),
                    1, "kitti_synthetic"))
    return out


def split(ids: list[str], test_frac: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(ids)
    random.Random(seed).shuffle(ids)
    k = round(len(ids) * test_frac)
    return sorted(ids[k:]), sorted(ids[:k])


def augment(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Flip + per-image gain/gamma jitter, so absolute exposure is not a free cue."""
    if torch.rand(1, generator=gen) < 0.5:
        x = x.flip(-1)
    f = x.float() / 255.0
    gain = 0.75 + 0.5 * torch.rand(1, generator=gen)
    gamma = 0.8 + 0.4 * torch.rand(1, generator=gen)
    return ((f * gain).clamp(0, 1) ** gamma * 255).to(torch.uint8)


# --- metrics --------------------------------------------------------------------------

def report(prob: np.ndarray, y: np.ndarray, src: np.ndarray) -> dict:
    pred = prob > 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    out = {"precision": round(100 * p, 2), "recall": round(100 * r, 2),
           "f1": round(100 * 2 * p * r / (p + r), 2) if p + r else 0.0, "per_source": {}}
    for s in sorted(set(src)):
        k = src == s
        out["per_source"][s] = {"n": int(k.sum()), "fired": int(pred[k].sum()),
                                "fire_rate": round(100 * float(pred[k].mean()), 2)}
    return out


# --- main -----------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=("mixed", "synthetic"), default="mixed")
    ap.add_argument("--real-root", type=Path,
                    default=REPO_ROOT / "datasets" / "real_glare_esfahani" / "RGB")
    ap.add_argument("--crops", type=int, default=3, help="crops per class per real photo")
    ap.add_argument("--test-frac", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model-out", type=Path, default=None)
    ap.add_argument("--metrics-out", type=Path, default=None)
    args = ap.parse_args(argv)
    args.model_out = args.model_out or REPO_ROOT / "models" / f"gate_cnn_{args.data}.pt"
    args.metrics_out = args.metrics_out or REPO_ROOT / "results" / f"gate_cnn_{args.data}_metrics.json"
    torch.manual_seed(args.seed)
    torch.set_num_threads(max(1, torch.get_num_threads()))

    if not args.real_root.is_dir():
        sys.exit(f"error: {rel(args.real_root)} not found; clone the Esfahani set (see docstring)")
    real_tr, real_te = split([p.name for p in args.real_root.iterdir() if p.is_dir()],
                             args.test_frac, args.seed)
    kitti_ids = [r["image_id"] for r in csv.DictReader((REPO_ROOT / "data/glare_labels_full.csv").open())
                 if r["is_glared"] == "1"]
    kit_tr, kit_te = split(kitti_ids, args.test_frac, args.seed)

    print(f"gate CNN  |  data {args.data}  |  real photos {len(real_tr)} train / {len(real_te)} "
          f"test  |  KITTI frames {len(kit_tr)} train / {len(kit_te)} test")
    t0 = time.perf_counter()
    train = kitti_samples(kit_tr)
    if args.data == "mixed":
        train += real_samples(args.real_root, real_tr, args.crops, args.seed)
    test = kitti_samples(kit_te) + real_samples(args.real_root, real_te, args.crops, args.seed + 1)
    print(f"  loaded {len(train)} train / {len(test)} test samples in {time.perf_counter() - t0:.0f} s")

    def stack(samples):
        return (torch.from_numpy(np.stack([s[0] for s in samples])).permute(0, 3, 1, 2).contiguous(),
                np.array([s[1] for s in samples]), np.array([s[2] for s in samples]))
    x_tr, y_tr, s_tr = stack(train)
    x_te, y_te, s_te = stack(test)

    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.classifier[3] = nn.Linear(net.classifier[3].in_features, 1)
    model = GateCNN(net)
    # Real samples are outnumbered by KITTI ones; weight them up so the two domains count
    # equally in the loss.
    is_real = np.char.startswith(s_tr.astype(str), "real")
    real_w = (~is_real).sum() / is_real.sum() if is_real.any() else 1.0
    w_tr = torch.from_numpy(np.where(is_real, max(1.0, real_w), 1.0)).float()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    gen = torch.Generator().manual_seed(args.seed)
    y_tr_t = torch.from_numpy(y_tr).float()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(y_tr), generator=gen)
        total, t0 = 0.0, time.perf_counter()
        for start in range(0, len(perm), args.batch):
            idx = perm[start:start + args.batch]
            xb = torch.stack([augment(x_tr[i], gen) for i in idx])
            loss = (nn.functional.binary_cross_entropy_with_logits(
                model(xb), y_tr_t[idx], reduction="none") * w_tr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        print(f"  epoch {epoch}  loss {total / len(perm):.4f}  ({time.perf_counter() - t0:.0f} s)")

    model.eval()
    with torch.no_grad():
        prob = torch.cat([torch.sigmoid(model(x_te[i:i + 64])) for i in range(0, len(x_te), 64)]).numpy()

    real = np.char.startswith(s_te.astype(str), "real")
    crop = np.isin(s_te, ["real_crop_pos", "real_crop_neg"])
    metrics = {"data": args.data, "model": "MobileNetV3-Small, ImageNet init, fine-tuned",
               "input": f"{SIZE}x{SIZE}", "threshold": 0.5, "epochs": args.epochs, "seed": args.seed,
               "n_train": len(y_tr), "test_real_photos": len(real_te),
               "test_all": report(prob, y_te, s_te),
               "test_real_crops": report(prob[crop], y_te[crop], s_te[crop]),
               "test_kitti": report(prob[~real], y_te[~real], s_te[~real])}

    sample = cv2.imread(str(REPO_ROOT / "datasets/kitti/images/val" / f"{kit_te[0]}.png"))
    for _ in range(5):
        model.predict(sample)
    t0 = time.perf_counter()
    for _ in range(50):
        model.predict(sample)
    metrics["latency_ms_per_frame"] = round((time.perf_counter() - t0) / 50 * 1000, 2)
    metrics["device"] = "cpu"

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics, indent=2) + "\n")

    print("\n" + "=" * 68)
    for name in ("test_real_crops", "test_kitti"):
        m = metrics[name]
        print(f"{name:<16} P {m['precision']:6.2f}  R {m['recall']:6.2f}  F1 {m['f1']:6.2f}")
    for s, r in metrics["test_all"]["per_source"].items():
        print(f"  {s:<16} fired {r['fired']:4d}/{r['n']:<4d} ({r['fire_rate']:.1f}%)")
    print(f"  latency {metrics['latency_ms_per_frame']} ms/frame (cpu)")
    print(f"  wrote {rel(args.model_out)}\n  wrote {rel(args.metrics_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
