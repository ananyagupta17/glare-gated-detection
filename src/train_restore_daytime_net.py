"""Train the learned daytime restorer (src/restore_daytime_net.py) on v2 TRAIN pairs only.

Input: the glared frame; target: its paired clean frame (same scene, same base exposure,
src/glare_synth.py). Both are resized to 624 x 192. The network predicts a correction
that is added to the input; loss is L1 on the corrected image.

The last 5% of train IDs (sorted, seeded shuffle) are held out to pick the best epoch.
No val frame is opened. Seed 42.

    python src/train_restore_daytime_net.py               # 15 epochs, MPS if available
Output: models/restore_daytime_net.pt, results/restore_daytime_net_training.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from restore_daytime_net import NET_H, NET_W, WEIGHTS, DayGlareUNet  # noqa: E402

V2 = REPO_ROOT / "data" / "v2"
BDD_PAIRS = REPO_ROOT / "data" / "bdd_pairs"


class Pairs(Dataset):
    def __init__(self, ids, flip: bool, v2_root: Path = None, bdd_root: Path = None):
        # Roots are stored on the dataset: DataLoader workers re-import this module, so a
        # module-level path changed in main() would not reach them.
        self.ids, self.flip = ids, flip
        self.v2_root, self.bdd_root = v2_root or V2, bdd_root or BDD_PAIRS

    def __len__(self):
        return len(self.ids)

    def load(self, version, image_id):
        # ids are "kitti:<id>" or "bdd:<name>" (bare ids are KITTI, for compatibility)
        src, _, name = image_id.rpartition(":")
        path = (self.bdd_root / version / f"{name}.jpg") if src == "bdd" else \
            (self.v2_root / version / "images" / "train" / f"{name}.jpg")
        img = cv2.imread(str(path))
        img = cv2.resize(img, (NET_W, NET_H), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    def __getitem__(self, k):
        i = self.ids[k]
        g, c = self.load("glared", i), self.load("clean", i)
        if self.flip and random.random() < 0.5:
            g, c = g[:, ::-1], c[:, ::-1]
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1)))  # noqa: E731
        return to(g), to(c)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--base", type=int, default=16)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--with-bdd", action="store_true",
                   help="add the BDD train pairs (data/bdd_pairs, src/make_bdd_pairs.py)")
    p.add_argument("--out", type=Path, default=WEIGHTS)
    p.add_argument("--synth", choices=["v2", "v3"], default="v2",
                   help="which synthetic glare to train on (data/v2 + data/bdd_pairs, or the v3 folders)")
    args = p.parse_args(argv)
    args.out = args.out.resolve()
    global V2, BDD_PAIRS
    if args.synth == "v3":
        V2, BDD_PAIRS = REPO_ROOT / "data" / "v3", REPO_ROOT / "data" / "bdd_pairs_v3"

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ids = sorted(f"kitti:{q.stem}" for q in (V2 / "glared" / "images" / "train").glob("*.jpg"))
    n_kitti = len(ids)
    if args.with_bdd:
        ids += sorted(f"bdd:{q.stem}" for q in (BDD_PAIRS / "glared").glob("*.jpg"))
    print(f"pairs: {n_kitti} KITTI + {len(ids) - n_kitti} BDD")
    random.Random(f"{args.seed}:daynet-holdout").shuffle(ids)
    n_hold = len(ids) // 20
    hold, train = sorted(ids[:n_hold]), sorted(ids[n_hold:])
    print(f"train {len(train)} pairs, holdout {len(hold)} (train split only), device {device}")

    g = torch.Generator().manual_seed(args.seed)
    tl = DataLoader(Pairs(train, True, V2, BDD_PAIRS), batch_size=args.batch, shuffle=True, generator=g,
                    num_workers=args.workers, persistent_workers=True, drop_last=True)
    hl = DataLoader(Pairs(hold, False, V2, BDD_PAIRS), batch_size=16, num_workers=args.workers)

    model = DayGlareUNet(args.base).to(device)
    n_params = sum(q.numel() for q in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                                                steps_per_epoch=len(tl))
    l1 = torch.nn.L1Loss()

    def holdout_l1():
        model.eval()
        tot = n = 0
        base = 0.0
        with torch.no_grad():
            for x, y in hl:
                x, y = x.to(device), y.to(device)
                tot += l1((x + model(x)).clamp(0, 1), y).item() * len(x)
                base += l1(x, y).item() * len(x)
                n += len(x)
        model.train()
        return tot / n, base / n

    history, best, t0 = [], None, time.time()
    for epoch in range(1, args.epochs + 1):
        run = 0.0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            loss = l1(x + model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item()
        h, base = holdout_l1()
        history.append({"epoch": epoch, "train_l1": round(run / len(tl), 5), "holdout_l1": round(h, 5)})
        print(f"epoch {epoch:>2}  train L1 {run / len(tl):.4f}  holdout L1 {h:.4f} "
              f"(no restoration {base:.4f})  {time.time() - t0:.0f}s", flush=True)
        if best is None or h < best[1]:
            best = (epoch, h)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "base": args.base, "epoch": epoch}, args.out)

    out = REPO_ROOT / "results" / (f"{args.out.stem}_training.json" if args.out != WEIGHTS
                                   else "restore_daytime_net_training.json")
    out.write_text(json.dumps({
        "synth": args.synth,
        "data": (f"{args.synth} KITTI train pairs" + (f" + BDD train pairs ({BDD_PAIRS.relative_to(REPO_ROOT)})" if args.with_bdd else "")
                 + "; holdout = 5% of pair IDs (seeded); no val frame of either dataset"),
        "n_kitti_pairs": n_kitti, "n_bdd_pairs": len(ids) - n_kitti,
        "n_train": len(train), "n_holdout": len(hold), "params": n_params,
        "input": f"{NET_W}x{NET_H} RGB [0,1]; predicts a correction added to the input",
        "loss": "L1", "optimizer": f"AdamW lr {args.lr} OneCycle, wd 1e-4",
        "epochs": args.epochs, "batch": args.batch, "seed": args.seed, "device": str(device),
        "best_epoch": best[0], "best_holdout_l1": round(best[1], 5),
        "holdout_l1_no_restoration": round(base, 5), "history": history,
        "weights": str(args.out.relative_to(REPO_ROOT)), "train_seconds": round(time.time() - t0, 1),
    }, indent=2) + "\n")
    print(f"best epoch {best[0]} (holdout L1 {best[1]:.4f}); wrote {args.out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
