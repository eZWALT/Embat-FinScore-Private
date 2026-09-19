# Interactions report (train only)

Holdout `analysis/splits/holdout_companies.csv` (72 companies) is **excluded** from every number below. No percentiles or bins were fit.

- Panel: **21157** company-months, **1214** train companies (monthly grid 2024-09 … 2026-08).
- Family: **I** (`i_*`). Source: `data/feature_store/monthly.parquet` A–H columns. Parquet was **not** rewritten.
- Size proxy: Spearman vs `log1p(max(a_in3, 0))`. Flag `|ρ| > 0.85` as SIZE.
- Constant: modal-value share `≥ 0.999` or a single value. NZV: modal share `≥ 0.95`.
- Persistence: median company-wise Pearson acf at lag 1 (≥ 4 finite pairs, non-zero s.d.).
- DSO products clip `e_dso_proxy` at 24 months (domain bound; not a train fit). Raw A–H columns are untouched.

## Brief map

These are *shapes* that answer dip vs fall / why / lead time. They are not a 0–100 score.

| feature | formula | six-question map |
| --- | --- | --- |
| `i_runway_x_hhi` | `b_runway * (1 - d_cust_hhi)` | Q1/Q4 healthy vs concentrated+thin — cushion × diversified customers |
| `i_io_x_zeroin` | `a_io_ratio * c_zero_in_share_6` | Q4 dip vs fall — coverage while trailing months show zero inflow |
| `i_transfer_x_ss` | `a_transfer * c_ss_month` | Q5 why / turning — Y3 SHAP pair (a_transfer × c_ss_month) |
| `i_dso_x_dsr` | `clip(e_dso_proxy, 0, 24) * f_ds_r` | Q5 why — collections months outstanding × debt-service / inflow |
| `i_runway_x_ar30` | `b_runway * e_ar_overdue_30` | Q4 dip vs fall — Hirshleifer: late AR with / without liquidity cushion |
| `i_runway_x_zeroin` | `b_runway * c_zero_in_share_6` | Q4 dip vs fall — dying activity with cash still on the book |
| `i_gap_x_supphhi` | `c_gap_sd * d_supp_hhi` | Q5 why — Pérez-Salazar: inter-tx gap sd × supplier HHI |
| `i_io_x_dsr` | `a_io_ratio * f_ds_r` | Q5 why — cash coverage × debt-service ratio (keep + Y3 SHAP) |
| `i_below0_x_payroll` | `b_below_0 * c_missed_salary` | Q4 fall not dip — negative reconstructed cash × missed payroll |
| `i_transfer_x_salary` | `a_transfer * c_salary_month` | Q5 why / turning — Y3 SHAP pair (a_transfer × c_salary_month) |
| `i_miss_e` | `1 if e_ar_open and e_ap_open are both null` | coverage — no invoice book this company-month (~36% train) |
| `i_miss_d` | `1 if d_cust_hhi and d_supp_hhi are both null` | coverage — no customer/supplier HHI (no ERP or incomplete 6m window) |

## Battery

| feature | cov_cm | cov_co | size_ρ vs log1p(a_in3) | acf1 | modal% | n_unique | flags |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `i_runway_x_hhi` | 40.2% | 54.9% | -0.072 | 0.425 | 15.5% | 7045 | — |
| `i_io_x_zeroin` | 88.5% | 100.0% | -0.193 | 0.536 | 85.0% | 2022 | — |
| `i_transfer_x_ss` | 100.0% | 100.0% | 0.071 | -0.003 | 74.1% | 4373 | — |
| `i_dso_x_dsr` | 36.9% | 55.0% | 0.248 | 0.477 | 70.7% | 2272 | — |
| `i_runway_x_ar30` | 42.5% | 54.2% | -0.235 | 0.441 | 13.6% | 7287 | — |
| `i_runway_x_zeroin` | 87.7% | 98.4% | -0.441 | 0.752 | 77.5% | 2387 | — |
| `i_gap_x_supphhi` | 48.6% | 61.0% | -0.507 | 0.706 | 3.1% | 9875 | — |
| `i_io_x_dsr` | 88.5% | 100.0% | 0.312 | 0.641 | 71.7% | 5253 | — |
| `i_below0_x_payroll` | 99.0% | 98.4% | 0.020 | 0.209 | 99.8% | 2 | NZV |
| `i_transfer_x_salary` | 100.0% | 100.0% | 0.071 | -0.040 | 76.6% | 4090 | — |
| `i_miss_e` | 100.0% | 100.0% | 0.035 | — | 64.1% | 2 | — |
| `i_miss_d` | 100.0% | 100.0% | -0.028 | 0.871 | 50.7% | 2 | — |

## SIZE / CONSTANT

- SIZE (`|ρ| > 0.85` vs `log1p(a_in3)`): none
- CONSTANT: none

`i_transfer_x_ss` and `i_transfer_x_salary` keep the raw euro `a_transfer` (parked as a level in the feature report, size ρ ≈ 0.02 vs log inflow). They inherit that scale; check the SIZE flag above before putting them in a GBM next to `log1p(a_in3)`.

`i_below0_x_payroll` is a rare joint event (keep even if NZV): NSF-like cash and a missed payroll month together is the fall-not-dip flag.

`i_miss_e` / `i_miss_d` are coverage diagnostics so a model can see 'no invoice book' / 'no counterparty HHI' instead of silently imputing. `i_miss_e` never flips inside a train company (acf1 undefined): it is has-ERP vs never-ERP. `i_miss_d` does flip (early incomplete 6m window then HHI appears) and is persistent (acf1 0.87).

`i_transfer_x_ss` / `i_transfer_x_salary` have acf1 ≈ 0 (same as raw `a_transfer`). Y3 lead time sits in `a_transfer_lag1`, not in these t-only products.

Do not treat holdout as confirmation. Do not drop A–H columns.

