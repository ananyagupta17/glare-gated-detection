# Severity sampler change: proposal

Status: **proposed, not implemented.** `src/glare_synth.py` is unchanged. Written 2026-10-20,
after `results/severity_scale_decision.json` dropped the mild band.

## Current
`assign_bands()` gives each glare type a third of its frames in each of mild, moderate and
severe, using a seeded shuffle. `draw_severity()` then draws s uniformly inside the frame's
band, between the calibrated cutpoints `mild_max_s` and `severe_min_s`. This needs both
cutpoints. A severe-only calibration has no `mild_max_s`, so `load_cutpoints()` would fail
on it.

## Proposed
s is drawn **uniformly on [0, 1] per glare type, stratified**:

1. For the n frames of one type in one split (sorted by ID), take the strata 0 … n−1.
2. Shuffle the strata with a seeded RNG (`"{seed}:{split}:strata:{type}"`), assigning stratum
   k to frame i.
3. s = (k + u) / n, with u ~ U(0, 1) from the frame's own `"severity"` stream (the existing
   per-frame seed).

Every type then covers [0, 1] evenly, with no clustering. Whether a frame is severe is
**derived, not assigned**: severe = 1 when s ≥ `severe_min_s` for its type.
`load_cutpoints()` reads `severe_min_s` (and `gate_exclude_below_s` for veiling_wash), and
no longer reads `mild_max_s`.

Determinism is unchanged. Every draw still comes from (seed, split, type or frame) hashes,
so the output still does not depend on processing order.

## Expected severe fraction
Severe fraction = 1 − `severe_min_s`. Cutoffs from `results/severity_scale_decision.json`.
With stratification, each count is one of two adjacent values: n − ⌈n·c⌉ or n − ⌊n·c⌋, depending on the jitter u of the one stratum that straddles the cutoff.

| Type | severe_min_s | Severe fraction | Train n | Expected severe (train) | Val n | Expected severe (val) |
|---|---|---|---|---|---|---|
| diffuse_haze | 0.6042 | 39.6% | 1,497 | 592–593 | 374 | 148–149 |
| direct_sun | 0.5903 | 41.0% | 1,496 | 612–613 | 374 | 153–154 |
| veiling_wash | 0.6911 | 30.9% | 1,496 | 462–463 | 374 | 115–116 |
| horizon_streak | 0.4007 | 59.9% | 1,496 | 896–897 | 374 | 224–225 |
| **All** | | **42.8–43.0%** | 5,985 | 2,562–2,566 | 1,496 | 640–644 |

horizon_streak skews severe because its drop curve rises fastest. This is a property of
that glare type, not of the sampler.

## What changes in the files

**`data/glare_manifest_v2.csv`: no column change.**
`image_id, split, is_glared, glare_type, severity, sat_frac` stay as they are. `severity`
becomes uniform on [0, 1] per type instead of band-clustered. Clean rows keep `severity = 0`.

**`data/glare_params_v2.csv`:**
- `band` (mild / moderate / severe) is **removed**.
- `stratum` (integer k) is **added**, so every draw can be traced.
- `severe` (0/1, s ≥ `severe_min_s`) is **added**.
- `gate_excluded` (0/1; 1 only for veiling_wash with s < `gate_exclude_below_s`) is
  **added**. See `results/gate_protocol_v2.md`.

**Code that reads `band` and must follow:**
- `src/shortcut_test.py` reads `band` for the two checks now marked `not_evaluable`. It
  must stop reading it.
- `src/eval_restorer_map.py` reports per band. It changes to all / severe / non-severe.

**Tests:** `test_types_and_bands_balanced` and `test_severity_inside_band` in
`tests/test_glare_synth.py` are replaced by:
- a stratification test: each type's s values occupy distinct strata, one per frame;
- a severe-fraction test against the cutoffs, to within one frame.
