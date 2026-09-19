# Wave 3 slot 1 — LightGBM Y7 + Y8

- **When:** 2026-09-19 ~00:13 CEST
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3` (lightgbm)
- **Holdout:** `analysis/splits/holdout_companies.csv` (72 companies). Never fitted.
- **X:** `data/feature_store/monthly.parquet` (22230 × 118)
- **Y:** `data/feature_store/targets.parquet` (frozen Y7/Y8 modules not edited)
- **Re-run:** `python -m analysis.models.gbm_y7y8`

## Files written

- `analysis/models/gbm_y7y8.py`
- `overnight/waves/wave3_slot1_gbm_y78.md` (this file)
- 16 rows appended to `analysis/experiments/registry.csv`

Did not edit `y7_concentration.py` or `y8_cross.py`. Did not use Y8 `META.forbidden_x_families = [a,b,e]` as a union.

## Setup

5 group-fold CV on train groups (`FOLD_SEED=20260918`), then one fit on all train and one holdout pass. Lags 1 and 3 on family X (not meta). Dummy = constant train prevalence. Best single = train-only sign, ≥25% labeled-train coverage (min 200), first train-best that is defined on holdout.

Per-column forbidden X (this is what the model used):

| Y | forbidden | allowed X | intended story |
|---|-----------|-----------|----------------|
| y7_top1_lost | **D** | A B C E F G H | looser; no inflow clause |
| y7_top1_lost_inflow | **D** | A B C E F G H | **interesting** concentration shock |
| y8_inv_worse_6 | **E** | A B C D F G H | **interesting**; cash → invoice |
| y8_cash_worse_6 | **A, B** | C D E F G H | **interesting**; invoice → cash |

Early stopping often collapsed to 1 tree on noisy val AUC (same pattern as the y2 panel). Final `n_trees` floored at 50.

## Holdout AUROC (power = n_pos)

| Y | train n / pos / rate | hold n / **n_pos** / rate | CV AUROC | **hold AUROC** | dummy | best single (train-pick) | beats dummy | beats single |
|---|---------------------:|--------------------------:|---------:|---------------:|------:|--------------------------|:-----------:|:------------:|
| y7_top1_lost | 7464 / 2149 / 28.8% | 391 / **122** / 31.2% | 0.663 | **0.680** | 0.500 | e_ar_issued 0.598 (−) | yes | yes |
| y7_top1_lost_inflow | 6739 / 420 / 6.2% | 318 / **15** / 4.7% | 0.645 | **0.763** | 0.500 | e_ar_issued 0.588 (−) | yes | yes |
| y8_inv_worse_6 | 3883 / 761 / 19.6% | 132 / **17** / 12.9% | 0.549 | **0.520** | 0.500 | a_in12 0.586 (+) | yes* | **no** |
| y8_cash_worse_6 | 6986 / 1615 / 23.1% | 233 / **61** / 26.2% | 0.565 | **0.490** | 0.500 | h_share_group_in 0.517 (−) | **no** | **no** |

\*y8_inv holdout 0.520 vs dummy 0.500 is a rounding-level edge, not a model.

## Interesting labels

### y7_top1_lost_inflow — GBM 0.763, but **n_pos = 15**

Holdout power is too thin to treat 0.76 as a confirmed win. CV 0.645 is the more honest number. Top gain: `e_dso_proxy`, `a_growth_3`, `e_ar_issued` (invoice book size / DSO + recent inflow growth — not family D). Train-best single is also `e_ar_issued` (smaller book → more likely to “lose” top-1). GBM does beat that single on this sample; do not ship on 15 events.

### y8_inv_worse_6 — GBM 0.520 loses to single

Cash/non-E X does not predict 6-month invoice-side deterioration. CV 0.549, hold 0.520, n_pos=17. Train-best single `a_in12` (0.586 hold) itself has almost no holdout coverage (2 pos on the defined rows). Gain is dominated by **D** (`d_n_cust`, `d_tx_cp_share`, `d_cust_hhi`) — allowed, but that is concentration persistence, not the cash→invoice story. `d_cust_lost` would have been 0.718 on holdout if we had peeked; we did not pick it (train 0.604 < `a_in12`).

### y8_cash_worse_6 — GBM 0.490 loses to dummy

Best-powered of the interesting three (61 holdout pos). Invoice/non-AB X does not predict 6-month cash-side deterioration. CV 0.565, hold 0.490. Top gain: `c_recency_days`, `group_size`, `h_share_group_in` — not family E. Train-best single `h_share_group_in` 0.517. Cross-source claim is not supported by this GBM.

## Secondary

`y7_top1_lost` (no inflow clause) is the only well-powered holdout: **122 pos**, GBM **0.680** vs dummy 0.500 vs `e_ar_issued` 0.598. Top gain `e_dso_proxy` / `e_ar_issued`. Useful as a concentration-event detector, not the catalogue shock.

## What failed

- **Y8 both columns:** LightGBM does not beat the best allowed single feature. y8_cash is the clean fail (61 pos, 0.490 < 0.500). y8_inv is underpowered and still loses to the train-picked single.
- **y7_top1_lost_inflow holdout n_pos=15:** AUROC 0.76 is not usable as a go/no-go.
- Early-stopping val AUC was noisy (several folds stopped at 1 tree); floor of 50 trees used for the final fit.
- Did not write a 0–100 score. Did not touch `product/`.

## Next idea

Park Y8 GBM until there is either a denser holdout or a different X cut (cash-only A+B for `y8_inv`, E-only for `y8_cash`) — the current “all non-forbidden families” mix lets D/H soak gain and still loses. For Y7, keep `y7_top1_lost` as the powered baseline (0.680 / 122 pos) and treat `y7_top1_lost_inflow` as a low-power diagnostic until more events exist. Do not raise Y8 on this bake-off.
