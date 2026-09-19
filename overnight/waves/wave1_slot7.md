# Wave 1 · slot 7 — Y1 forecast + Y2 liquidity stress

- **Files:** `analysis/targets/y1_forecast.py`, `analysis/targets/y2_stress.py`
- **Grid:** monthly `company_id × period` (22,230 rows, 1,286 companies)
- **Train only** (holdout `analysis/splits/holdout_companies.csv` excluded from rates/AUROC): 21,157 company-months, 1,214 companies
- **Thresholds:** fixed from the catalogue. Not searched on train or holdout.
- **Liq path:** same unwind as `score_pipeline._liquidity` (checking/saving/tpv); max |diff| vs that function = 0
- **Y2 META.forbidden_x_families:** `["b"]`
- **Re-run:** `python -m analysis.targets.y2_stress`

## Y1 (continuous, values at t+h, not known at t)

| column | train n | train coverage | accepted |
|--------|--------:|---------------:|----------|
| y1_net_h1 | 19,943 | 94.26% | n/a (continuous) |
| y1_net_h3 | 17,515 | 82.79% | n/a (continuous) |
| y1_in_h1 | 19,943 | 94.26% | n/a (continuous) |
| y1_in_h3 | 17,515 | 82.79% | n/a (continuous) |
| y1_liq_h1 | 19,746 | 93.33% | n/a (continuous) |
| y1_liq_h3 | 17,356 | 82.03% | n/a (continuous) |

Nulls are the last 1 / 3 months (no future) and months without a reconstructed balance. Check: `y1_net_h1` equals `net` at t+1 (max abs diff 0).

## Y2 (binary, sustained; size = AUROC of log1p(\|monthly op_in\|))

| column | train n | pos | base rate | accepted | size AUROC | flag |
|--------|--------:|----:|----------:|----------|-----------:|------|
| y2_neg_2of3 | 17,356 | 1,271 | **7.32%** | **yes** | 0.531 | — |
| y2_runway_lt1_sust | 16,161 | 6,161 | 38.12% | **no** | 0.655 | SIZE_PROXY |
| y2_onset_neg | 17,515 | 191 | 1.09% | **no** | 0.541 | — |

Definitions shipped as specified (still present when rejected):

- `y2_neg_2of3` = 1 if liq < 0 in ≥ 2 of the next 3 months
- `y2_runway_lt1_sust` = 1 if runway < 1 in t+1, t+2 and t+3
- `y2_onset_neg` = 1 if a first negative month after ≥ 6 clean (liq ≥ 0) months falls in t+1..t+3

## What failed

- **y2_runway_lt1_sust:** 48% of reconstructed months already have runway < 1, so a 3-month future stretch is 38% (above 30%) and tracks size (AUROC 0.655).
- **y2_onset_neg:** true first-onset after six clean months is rare (1.1%).

## Next idea (later wave; do not retune here)

Use `y2_neg_2of3` as the R1 binary. If a second stress label is needed, try runway < 0 (or liq below 0.5× monthly outflow) for 3 months, or onset with h = 6 — only as a new column, not by fitting a cut on train.
