# Wave 3 · slot 6 — assemble Y panel + feature dictionary

- **Files written:** `analysis/targets/build_targets.py`, `data/feature_store/targets.parquet`, `data/feature_store/y_acceptance.csv`, `data/feature_store/feature_dictionary.md`
- **Not written:** `monthly.parquet` (another process owns it; left untouched). Family D not edited.
- **Grid:** monthly `company_id × period` (22,230 rows, 1,286 companies)
- **Train only** for rates: 21,157 company-months (holdout excluded). No threshold search / no refit.
- **Re-run:** `python -m analysis.targets.build_targets`

Assembler mirrors `build_feature_store.py`: load `monthly_grid`, import every `analysis.targets.y*.py` that exposes `build()`, merge on `company_id, period`, skip modules that are not written yet.

## Targets.parquet

Shape **(22230, 27)** = keys + 25 Y columns from y1–y8.

| module | columns |
|--------|---------|
| y1_forecast | y1_net_h1, y1_net_h3, y1_in_h1, y1_in_h3, y1_liq_h1, y1_liq_h3 |
| y2_stress | y2_neg_2of3, y2_runway_lt1_sust, y2_onset_neg |
| y3_recovery | y3_recover_6m, y3_recover_cash_6m |
| y4_debt | y4_ds_r_gt05_sust, y4_ds_r_double, y4_new_facility_after_dip, y4_ogtg_appear |
| y5_payment | y5_ap_od30_ownp80, y5_ar_od30_sust, y5_ap_delay_up15 |
| y6_activity | y6_zero_in_3, y6_missed_payroll, y6_silent_60 |
| y7_concentration | y7_top1_lost, y7_top1_lost_inflow |
| y8_cross | y8_inv_worse_6, y8_cash_worse_6 |

## Accepted (do not refit)

Frozen from earlier waves: `y2_neg_2of3`, `y5_ap_od30_ownp80`, `y5_ar_od30_sust`, `y4_ds_r_double`. All six Y1 continuous columns kept.

Later wave notes raised: `y3_recover_cash_6m` (wave2_slot7), `y6_zero_in_3` and `y6_missed_payroll` (wave2_slot7). Same-line scan only (a previous-row ACCEPTED must not mark the next REJECTED column).

| column | kind | train labeled | base rate | accepted | source |
|--------|------|--------------:|----------:|---------:|--------|
| y1_* (6 cols) | continuous | 82–94% | — | 1 | y1_continuous |
| y2_neg_2of3 | binary | 17,356 | 7.32% | 1 | frozen_wave |
| y3_recover_cash_6m | binary | 5,648 | 7.12% | 1 | wave_note |
| y4_ds_r_double | binary | 2,370 | 13.88% | 1 | frozen_wave |
| y5_ap_od30_ownp80 | binary | 4,905 | 8.52% | 1 | frozen_wave |
| y5_ar_od30_sust | binary | 3,315 | 7.15% | 1 | frozen_wave |
| y6_zero_in_3 | binary | 17,083 | 8.72% | 1 | wave_note |
| y6_missed_payroll | binary | 6,004 | 6.11% | 1 | wave_note |

Shipped `accepted=0`: y2_runway_lt1_sust (38%), y2_onset_neg (1.1%), y3_recover_6m (4.3%), y4_ds_r_gt05_sust (2.6%), y4_new_facility_after_dip (4.6%), y4_ogtg_appear (2.6% snapshot), y5_ap_delay_up15 (4.5%), y6_silent_60 (3.0%), y7_top1_lost, y7_top1_lost_inflow, y8_inv_worse_6, y8_cash_worse_6. y7/y8 sit in the 5–30% band but have no ACCEPTED wave line yet.

## Feature dictionary

One section per meta / family / Y column from module docstrings and `META`. `d_interco_share` documented as all-NaN (COMP_* vs COUNTERPARTY_* do not overlap). Forbidden-X filled for every Y (per-column where META has it: y3_recover_6m also forbids E; y8_inv_worse_6 forbids E; y8_cash_worse_6 forbids A+B).

## What failed

- First pass dropped Y columns (`_keys` kept only join keys). Fixed before the parquet write.
- Cross-line ACCEPTED/REJECTED regex marked `y6_silent_60` accepted and missed `y6_missed_payroll`. Replaced with same-line scan.
- Did not assemble X; `monthly.parquet` already present from another slot.

## Next idea

Bake-off should use the accepted list as-is. Do not promote y7/y8 from base rate alone — wait for their wave note. If a later note flips a shipped column, re-run `build_targets` (scanner + frozen set, no threshold search).
