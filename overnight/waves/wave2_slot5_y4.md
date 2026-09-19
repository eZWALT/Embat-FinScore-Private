# Wave 2 · slot 5 — Y4 debt pressure

- **Files:** `analysis/targets/y4_debt.py` (this note)
- **API:** `META.forbidden_x_families = ["f"]`, `horizon = 6`, `build(con, grid)` → `company_id`, `period`, `y4_*`
- **Grid:** monthly `company_id × period` (22,230 rows, 1,286 companies). Weekly rows align to the containing month.
- **Train only** (holdout `analysis/splits/holdout_companies.csv` excluded from rates/AUROC): 21,157 company-months, 1,214 companies
- **Thresholds:** fixed from the catalogue. Not searched on train or holdout.
- **Flows:** computed here from `transactions` via `CAT_MAP` (`debt_repayment` → debt_service; `fee` / `interest_charge` → fin_cost). Does **not** import `analysis.features.debt`.
- **Dip:** operational `net < 0` in ≥ 2 of the last 3 months at t (did not copy family B liquidity reconstruction).
- **Re-run:** `python -m analysis.targets.y4_debt`
- **Smoke:** `/tmp/smoke_wave2_slot5_y4.py` → `/tmp/wave2_slot5_y4_summary.json`. `overnight/verify_wave.py`: `OK analysis.targets.y4_debt rows=22230 companies=1286 cols=4`.

## Y4 (binary; size = AUROC of log1p(\|monthly op_in\|))

ACCEPTED if train base rate ∈ [5%, 30%] **and** size AUROC < 0.60. Rejected columns are still shipped.

| column | train n | pos | base rate | accepted | size AUROC | flag |
|--------|--------:|----:|----------:|----------|-----------:|------|
| y4_ds_r_gt05_sust | 16,301 | 421 | 2.58% | **no** | 0.353 | — |
| y4_ds_r_double | 2,370 | 329 | **13.88%** | **yes** | 0.443 | — |
| y4_new_facility_after_dip | 12,674 | 578 | 4.56% | **no** | 0.663 | SIZE_PROXY |
| y4_ogtg_appear | 1,214 | 32 | 2.64% | **no** | 0.664 | SIZE_PROXY; snapshot |

Definitions:

- `y4_ds_r_gt05_sust` = 1 if `ds_r = debt_service_3m / max(in3, 1) > 0.5` in t+1, t+2 and t+3 (`ds_r` unclipped; 3-month window must be full).
- `y4_ds_r_double` = 1 if `ds_r` at t+3 ≥ 2 × `ds_r` at t, both defined, and `ds_r_t > 0.05`. Rows with `ds_r_t ≤ 0.05` are NaN (not 0).
- `y4_new_facility_after_dip` = 1 if a `debt_products.created_at` is in `(t, t+6m]` **and** the net-dip holds at t. Label is NaN when t+6m is after the 2026-09-01 extract (window not fully observed) or the 3-month dip window is incomplete.
- `y4_ogtg_appear` = snapshot flag only. 1/0 on **2026-08** if the company has any `debt_products.outstanding_gt_granted`; **NaN on every other month**. 35 companies have the flag in the extract (32 in train). This is not an onset that “appears” in 2024.

## What failed

- **y4_ds_r_gt05_sust:** trailing repayment / inflow > 0.5 is already rare (~3.8% of months). Requiring three consecutive *future* months drops the train rate to 2.58% (below 5%). Size AUROC is fine (0.35). Including `fin_cost` in the numerator would have cleared 5% but would no longer be `debt_service_3m` under `CAT_MAP`; not done here.
- **y4_new_facility_after_dip:** 4.56% (just under 5%) and size AUROC 0.66 — larger firms open more facilities. Using reconstructed `liq < 0` instead of net made the event rarer, not cleaner.
- **y4_ogtg_appear:** only the last month is valid (n = 1,214). Rate 2.6% and size-correlated. `debt_schedule_config` has 6 `outstanding_gt_granted` rows; they were not used as a time path.

## Next idea (later wave; do not retune here)

Keep `y4_ds_r_double` as the R2 accepted debt Y. If a second label is needed, add a *new* column that uses 2-of-3 future months with `ds_r` (or repayment + interest only), not a train-fitted cut. Do not turn `outstanding_gt_granted` into a 24-month series.
