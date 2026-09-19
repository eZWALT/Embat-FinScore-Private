# Wave 3 slot 2 — Y3 per-group LightGBM

**Decision: PARK.** Mixture group-fold CV AUROC **0.694** does not beat the
published global **0.710**, and is below the keep bar **0.720**.

## Brief question 2 — who is improving?

Y3 `y3_recover_cash_6m` is the 45→65 direction: among companies already
stressed (liq<0 or runway<1), who returns to runway≥3 for 3 consecutive
months inside t+1..t+6. This slot asked whether **pooling siblings inside
`group_id`** reads that recovery better than the one global engine.

It does not, on the number we are allowed to quote (train group-fold CV).
The global panel already uses family H (sibling flows, group size). A
separate LightGBM per legal group does not add a more honest “this
subsidiary is healing” signal. Hidden-test companies are whole unseen
groups, so a group-specific model would not apply there anyway.

Not a 0–100 product score. Engine evidence only.

## Files

- `analysis/models/gbm_y3_group.py` (this slot; does not edit `gbm_y3y6.py`)
- `overnight/waves/wave3_slot2_y3_group.md` (this note)
- `analysis/experiments/registry.csv` (4 rows appended, agent `823061e3`)

## Setup (same X/Y as the published 0.710)

- Y: `y3_recover_cash_6m`, stressed/labeled rows only (NaN otherwise).
- X: families **A,C,D,E,F,G,H** — **never B**. Lags 1,3. Meta: `group_size`,
  `n_banking`. 278 numeric columns after lags.
- Train labeled: **5648** company-months / **402** positives / rate **0.0712**
  (coverage 5648/21157 = **0.267** of train company-months).
- Eligibility for a group model: `group_id` has **≥5 train companies**.
  Fit also needs ≥20 stressed rows, both classes, ≥2 labeled companies in
  the sibling slice. Else **global model fit on the train fold only**.
- Holdout (72 companies / 15 groups) never in any fit, percentile, or
  early-stop. Zero group overlap with train.

### Why sibling company-fold inside large groups

`group_folds` keeps every sibling on one side. A held-out group has **zero**
in-group train companies, so leave-group-out cannot score a per-group
model (the mixture would equal the global engine by construction). The
test that can actually use pooling: leave-company-out on siblings, then
evaluate those OOS scores with the same five group-folds as the bake-off
(mean of fold AUROCs = the published 0.710 aggregation). Small / thin
groups keep the global group-fold OOF.

## Results

| Engine | Split | AUROC | Notes |
|--------|-------|------:|-------|
| Global (same run, same protocol as `gbm_y3y6`) | cv5_group mean | **0.710** | folds 0.653 / 0.749 / 0.682 / 0.709 / 0.754 |
| Published global | cv5_group | **0.710** | registry `lightgbm_y3y6` |
| Per-group mixture | cv5_group mean | **0.694** | folds 0.527 / 0.791 / 0.671 / 0.755 / 0.728 |
| Per-group mixture | pooled OOS | 0.728 | still below global pooled 0.731 |
| Global | pooled OOS | 0.731 | |
| Mixture on rows that used a group model | sibling OOS | 0.763 | vs global OOF 0.718 on the same 2528 rows |
| Holdout (unseen groups → global only) | holdout | 0.899 | **n_pos=14 LOW_POWER — do not keep/kill** |

- Train groups: 235. Size-eligible (≥5 companies): 90.
- Group models actually used: **39** groups / **2528** stressed rows (45% of
  train labeled). Final train-only group fits: 40 (none apply to holdout).
- Skips: 92 small groups, 41 thin or one-class (several large groups have
  0 recoveries, e.g. GROUP_0158: 21 companies / 339 stressed rows / 0 pos),
  1 fold-slice too thin.
- Holdout: 235 labeled, 14 positives, base rate 0.060, coverage 1.0.
  Dummy prior AUROC 0.50. Same global holdout number as `gbm_y3y6`
  (0.899) because holdout groups are unseen.

**KEEP rule:** mixture CV AUROC > 0.720. **0.694 ≤ 0.720 → PARK.**
Holdout was not used for the decision.

## What failed

- Fold-mean mixture **lost** to the global engine (0.694 vs 0.710). Fold 0
  collapsed (0.527 vs global 0.653): sibling GBMs on ~2–15 positives per
  group are high-variance and can wreck a fold.
- Pooled AUROC also does not beat global (0.728 vs 0.731).
- Most legal groups cannot host a binary GBM: 145/235 are small or have
  no / almost no recoveries. Recovery is not a within-group event we can
  pool reliably.
- Even a win on the 39 usable groups would not transfer to the contest
  hidden test (new groups, same structure as this holdout).

The 0.763 vs 0.718 slice on rows that *did* get a group score is not a
keep: it is not the group-fold mean, it is selected on groups that
already have both classes, and it did not lift the overall CV.

## Next idea

Leave Y3 on the **global** LightGBM (CV 0.710). Spend the next slot on
explainability / lead time (questions 5–6) or on a **cluster** pool from
`feature_report` (similar cash-shape companies, not `group_id`), only if
someone still wants a pooling bet. Do not fit another per-`group_id` GBM
on this Y.
