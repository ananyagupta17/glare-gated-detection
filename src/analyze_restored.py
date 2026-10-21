"""Two checks on the whole-frame restorer, on the 200-frame trial subset (subset: 200).

The subset is the same 200 val pairs as the --limit 200 trial
(results/restorer_map_whole_limit200.json): the first 200 eligible IDs, listed in
results/restored_analysis_subset200_ids.txt. The script refuses to run if the list and
the trial's limit disagree.

1. Clean-frame cost. Restore the 200 CLEAN v2 val frames too, and score detection on
   clean vs restored-clean (all / severe / non-severe). A restorer that is safe to run on
   frames without glare should leave their mAP unchanged.

2. Separation after restoration. A logistic regression on the gate classifier's 9
   features (src/train_gate_classifier.py), trained on v2 train clean vs glared exactly
   as in shortcut_test.py's mild_vs_severe check, scored on the 200 val pairs as:
       clean vs glared                     (reference)
       clean vs restored-glared            (what a gate sees after restoration)
       restored-clean vs restored-glared   (both sides restored)
   Report only: no threshold from shortcut_test.py is used or changed.

Restored clean frames are cached in data/restored/v2_whole_clean/val/.
Output: results/restored_analysis.json

    python src/analyze_restored.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
import eval_restorer_map as erm  # noqa: E402
import shortcut_test as st  # noqa: E402

SUBSET = 200
IDS_FILE = REPO_ROOT / "results" / "restored_analysis_subset200_ids.txt"
TRIAL = REPO_ROOT / "results" / "restorer_map_whole_limit200.json"
GLARED_CACHE = REPO_ROOT / "data" / "restored" / "v2_whole" / "val"
CLEAN_CACHE = REPO_ROOT / "data" / "restored" / "v2_whole_clean" / "val"
OUT = REPO_ROOT / "results" / "restored_analysis.json"
WORKERS = 7


def main() -> int:
    trial = json.loads(TRIAL.read_text())
    ids = [ln.strip() for ln in IDS_FILE.read_text().splitlines() if ln.strip()]
    if trial["limit"] != SUBSET or len(ids) != SUBSET:
        sys.exit(f"error: subset mismatch (trial limit {trial['limit']}, ids {len(ids)})")
    dirs = erm.v2_val_dirs(erm.V2_ROOT)
    missing = [i for i in ids if not (GLARED_CACHE / f"{i}.png").is_file()]
    if missing:
        sys.exit(f"error: {len(missing)} restored glared frames missing, e.g. {missing[:3]}")
    print(f"subset: {SUBSET} val pairs ({ids[0]}..{ids[-1]})")

    # ---- 1. restore the 200 clean frames, score clean vs restored-clean ----------------
    info = erm.restore_all("whole", ids, dirs["clean"], CLEAN_CACHE, threads=4)
    print(f"restored clean frames: {info}")
    from train_baseline import evaluate, pick_device
    from ultralytics import YOLO
    device = pick_device(None)
    model = YOLO(str(erm.DEFAULT_WEIGHTS))
    ev = SimpleNamespace(device=device, imgsz=640, batch=16)
    severe = erm.load_severe(erm.PARAMS)
    detection = {}
    for version, (image_dir, suffix) in {"clean": (dirs["clean"], ".jpg"),
                                         "clean_restored": (CLEAN_CACHE, ".png")}.items():
        detection[version] = {}
        for group, group_ids in erm.severity_groups(ids, severe).items():
            y = erm.subset(REPO_ROOT / "datasets" / "eval_restorer" / "clean_restore_200",
                           f"{version}_{group}", group_ids, image_dir, suffix)
            m = evaluate(model, y, ev)
            detection[version][group] = {"mAP50": m["mAP50"], "mAP50_95": m["mAP50_95"],
                                         "count": len(group_ids)}
            print(f"  [subset {SUBSET}] {version:<15}{group:<11} n={len(group_ids):>4}  "
                  f"mAP50 {m['mAP50']:.4f}")
    change = {g: round(detection["clean_restored"][g]["mAP50"] - detection["clean"][g]["mAP50"], 4)
              for g in detection["clean"]}

    # ---- 2. gate-feature logistic regression, before and after restoration -----------
    train_ids = sorted(p.stem for p in (erm.V2_ROOT / "glared" / "images" / "train").glob("*.jpg"))
    print(f"extracting gate features: {len(train_ids)} train pairs, {SUBSET} val pairs x 4")
    tr_clean = st.extract_all([erm.V2_ROOT / "clean" / "images" / "train" / f"{i}.jpg"
                               for i in train_ids], WORKERS)
    tr_glared = st.extract_all([erm.V2_ROOT / "glared" / "images" / "train" / f"{i}.jpg"
                                for i in train_ids], WORKERS)
    val = {
        "clean": st.extract_all([dirs["clean"] / f"{i}.jpg" for i in ids], WORKERS),
        "glared": st.extract_all([dirs["glared"] / f"{i}.jpg" for i in ids], WORKERS),
        "clean_restored": st.extract_all([CLEAN_CACHE / f"{i}.png" for i in ids], WORKERS),
        "glared_restored": st.extract_all([GLARED_CACHE / f"{i}.png" for i in ids], WORKERS),
    }
    lr = st.LogReg().fit(np.vstack([st.matrix(tr_clean, ["gate"]), st.matrix(tr_glared, ["gate"])]),
                         np.r_[np.zeros(len(tr_clean)), np.ones(len(tr_glared))])
    separation = {}
    for name, (neg, pos) in {"clean_vs_glared": ("clean", "glared"),
                             "clean_vs_glared_restored": ("clean", "glared_restored"),
                             "clean_restored_vs_glared_restored": ("clean_restored", "glared_restored"),
                             }.items():
        sn, sp = lr.score(st.matrix(val[neg], ["gate"])), lr.score(st.matrix(val[pos], ["gate"]))
        acc = float(np.r_[sn < 0.5, sp >= 0.5].mean())
        separation[name] = {"accuracy": round(acc, 4),
                            "auc": round(st.two_sided(st.auc(sp, sn)), 4)}
        print(f"  [subset {SUBSET}] {name:<36} accuracy {acc:.4f}  AUC {separation[name]['auc']:.4f}")

    OUT.write_text(json.dumps({
        "subset": SUBSET,
        "subset_ids": str(IDS_FILE.relative_to(REPO_ROOT)),
        "subset_source": "first 200 eligible val IDs, identical to the --limit 200 trial",
        "restorer": "whole", "detector_weights": "models/yolo_c1_train.pt", "device": device,
        "clean_restore": {"subset": SUBSET, "restore": info, "detection": detection,
                          "mAP50_change": change},
        "separation": {
            "subset": SUBSET,
            "classifier": "logistic regression on the gate classifier's 9 features, trained "
                          "on v2 train clean vs glared (5,985 pairs); as in shortcut_test.py",
            "auc": "two-sided, max(AUC, 1-AUC)", "results": separation,
            "note": "report only; no shortcut_test.py threshold is applied"},
    }, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
