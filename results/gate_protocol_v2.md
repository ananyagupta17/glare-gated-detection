# Gate protocol for v2 data

**Pre-registered 2026-10-20**, before any v2 frame was generated and before any gate was
trained or scored on v2. It follows from `results/severity_scale_decision.json` (commit
`ddf2673`).

## Labels
- **Positive (glare):** the glared version of a frame, unless excluded below.
- **Negative (no glare):** the clean version of every frame, including frames whose glared
  version is excluded.
- **Excluded:** glared veiling_wash frames with severity s < 0.1753
  (`gate_exclude_below_s` in `results/severity_calibration.json`, once written).

## Why veiling_wash below 0.1753 is excluded
At its weakest multiplier, veiling_wash costs the detector an F1 drop of 0.006, 95% CI
[−0.019, 0.027], which includes zero (`results/severity_s0_bootstrap.json`). Up to
s = 0.1753 its drop stays below 0.05. Those frames are glared in name only, and calling them
positives would penalise a gate for correctly ignoring glare that doesn't matter. No other
type is excluded: at s = 0, each of the other three already costs at least 0.0497.

## Where the rule applies
- **Gate training:** excluded frames are left out of the positives. Their clean versions stay
  in as negatives.
- **Gate scoring** (precision, recall, F1, per-type recall): the same rule applies.
- **Not detection.** The C1/C2/C3 detection arms run on every val frame. At runtime the gate
  decides on each frame, and an excluded frame that triggers restoration is simply counted in
  the trigger rate.
- **Data files:** excluded frames stay `is_glared = 1` in `data/glare_manifest_v2.csv`.
  They're identified by `gate_excluded = 1` in `data/glare_params_v2.csv` (added by the
  proposed sampler change, `results/sampler_change_proposal.md`).

## How many frames it drops
This assumes the proposed stratified sampler, where s is uniform on [0, 1] per type. The
exact count is fixed by the seed, and is one of two adjacent values:

| Split | veiling_wash frames | Excluded (s < 0.1753) | Gate positives left (all types) | Gate negatives |
|---|---|---|---|---|
| train | 1,496 | **262–263** | 5,722–5,723 of 5,985 | 5,985 |
| val | 374 | **65–66** | 1,430–1,431 of 1,496 | 1,496 |

Gate training and scoring are therefore slightly imbalanced, at about 49/51 positive to
negative. Report precision and recall, not only accuracy. Once the data is generated, the
exact counts will be added under this line, without changing anything above it.

## Owner
Gate training and scoring: Harshitha. Her pipeline must read `gate_excluded` from
`data/glare_params_v2.csv` and apply the rule above.
