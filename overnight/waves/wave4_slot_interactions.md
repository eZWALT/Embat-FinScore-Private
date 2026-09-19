# Wave 4 slot — Family I (cross-family shapes)

## Files written

- `analysis/features/interactions.py` — `SOURCE_TABLES=[]`, `FAMILY="i"`, `build(con, grid) -> company_id, period, i_*`
- `analysis/outputs/interactions_report.md` — train-only coverage / size ρ / acf1
- This note

Did **not** rewrite `data/feature_store/monthly.parquet`. Did **not** add I to `FAMILIES` (parent merges after review). Did not edit other family files, `product/`, or A–H columns.

`build` loads A–H from the assembled parquet (or uses those columns if they are already on `grid`). `con` is unused. No look-ahead: every `i_*` is a same-period product of already causal features. No holdout fit (formulas only; DSO clip at 24 months is a domain bound, not a train percentile).

Smoke: `python -m analysis.features.interactions` (22,230 rows, 1,286 companies, 12 `i_*`, no key dups). Overlay path (pass the parquet as `grid`) matches parquet-load path at max abs 0.

## Columns

| column | formula | brief |
|--------|---------|-------|
| `i_runway_x_hhi` | `b_runway * (1 - d_cust_hhi)` | Q1/Q4 cushion × diversified customers |
| `i_io_x_zeroin` | `a_io_ratio * c_zero_in_share_6` | Q4 coverage while activity fades |
| `i_transfer_x_ss` | `a_transfer * c_ss_month` | Q5 Y3 SHAP pair |
| `i_dso_x_dsr` | `clip(e_dso_proxy, 0, 24) * f_ds_r` | Q5 collections × debt service |
| `i_runway_x_ar30` | `b_runway * e_ar_overdue_30` | Q4 Hirshleifer late AR × cushion |
| `i_runway_x_zeroin` | `b_runway * c_zero_in_share_6` | Q4 dying inflow with cash left |
| `i_gap_x_supphhi` | `c_gap_sd * d_supp_hhi` | Q5 Pérez-Salazar ops vol × supplier HHI |
| `i_io_x_dsr` | `a_io_ratio * f_ds_r` | Q5 coverage × debt-service ratio |
| `i_below0_x_payroll` | `b_below_0 * c_missed_salary` | Q4 fall not dip |
| `i_transfer_x_salary` | `a_transfer * c_salary_month` | Q5 Y3 SHAP pair |
| `i_miss_e` | 1 if `e_ar_open` and `e_ap_open` both null | invoice coverage |
| `i_miss_d` | 1 if `d_cust_hhi` and `d_supp_hhi` both null | counterparty coverage |

## Train coverage / size ρ / acf1

Holdout 72 excluded. **21,157** company-months, **1,214** companies. Size = Spearman vs `log1p(max(a_in3, 0))`. SIZE if `|ρ| > 0.85`.

| column | cov_cm | size_ρ | acf1 | flags |
|--------|--------|--------|------|-------|
| `i_runway_x_hhi` | 40.2% | -0.072 | 0.425 | — |
| `i_io_x_zeroin` | 88.5% | -0.193 | 0.536 | — |
| `i_transfer_x_ss` | 100.0% | 0.071 | -0.003 | — |
| `i_dso_x_dsr` | 36.9% | 0.248 | 0.477 | — |
| `i_runway_x_ar30` | 42.5% | -0.235 | 0.441 | — |
| `i_runway_x_zeroin` | 87.7% | -0.441 | 0.752 | — |
| `i_gap_x_supphhi` | 48.6% | -0.507 | 0.706 | — |
| `i_io_x_dsr` | 88.5% | 0.312 | 0.641 | — |
| `i_below0_x_payroll` | 99.0% | 0.020 | 0.209 | NZV (rare keep) |
| `i_transfer_x_salary` | 100.0% | 0.071 | -0.040 | — |
| `i_miss_e` | 100.0% | 0.035 | — | — |
| `i_miss_d` | 100.0% | -0.028 | 0.871 | — |

**SIZE: none. CONSTANT: none.**

## What failed

- `i_miss_e` acf1 is undefined: no train company flips (has-ERP vs never-ERP). Still a usable missingness flag (36% of train company-months).
- `i_transfer_x_ss` / `i_transfer_x_salary` keep raw euro `a_transfer` (parked level). Not SIZE vs `log1p(a_in3)` (ρ ≈ 0.07) but acf1 ≈ 0 — Y3 lead time is `a_transfer_lag1`, not this t-only product.
- `i_below0_x_payroll` fires on 0.25% of train company-months (NZV). Keep as the fall-not-dip rare event.
- Intersection coverage is the cap: `i_dso_x_dsr` 36.9%, `i_runway_x_hhi` 40.2% (need both parents non-null).
- Weekly grid is unsupported until a weekly store exists (parquet is monthly).

## Next idea

When parent adds I to `FAMILIES`, prefer passing the in-memory A–H panel into `build` so I does not depend on a stale parquet. Optional shape sibling for transfer: `clip(a_transfer / max(a_in3, 1), -3, 3) * c_ss_month` if a GBM still treats the euro product as a level.
