# Wave 1 slot 5 — Family C operational regularity

- **Owner files:** `analysis/features/ops.py` (this note)
- **API:** `SOURCE_TABLES=["transactions"]`, `FAMILY="c"`, `build(con, grid)` → `company_id`, `period`, `c_*`
- **Grid:** monthly (22,230 company-months, 1,286 companies). No look-ahead past month-end. No holdout fit (no `REF`).

## Files written

- `analysis/features/ops.py`
- `overnight/waves/wave1_slot5.md`
- Scratch only: `/tmp/smoke_wave1_slot5.py`, `/tmp/wave1_slot5_coverage.csv`, `/tmp/wave1_slot5_summary.json`

## Columns (12)

| column | definition |
|--------|------------|
| `c_n_tx` | tx count in the month (0 if none) |
| `c_n_days_with_tx` | distinct booking dates in the month |
| `c_gap_sd` | sample stdev of day-gaps between consecutive *unique* booking dates in the last 90 days (Pérez-Salazar 2026 σ_Δt). Null if &lt;3 distinct days |
| `c_zero_in_month` | 1 if no `amount > 0` this month (any category, not CAT_MAP `op_in`) |
| `c_zero_in_share_6` | mean of `c_zero_in_month` over last ≤6 months (`min_periods=1`) |
| `c_salary_month` | 1 if category `salary` present |
| `c_tax_month` | 1 if category `tax` present (`tax_refund` excluded) |
| `c_ss_month` | 1 if category `social_security` present |
| `c_missed_salary` | 1 if ≥3 of last ≤6 months have salary and this month does not |
| `c_missed_tax` | same for tax |
| `c_recency_days` | days from last booking date ≤ month-end to month-end |
| `c_last_tx_before_2026_06` | 1 if `period ≥ 2026-06-01` and last tx as-of month-end &lt; 2026-06-01 |

`c_gap_sd` uses unique calendar days: 94% of timestamps are midnight and same-day multi-tx is common (~8 txs/day), so all-event gaps are almost all 0 and become a size proxy.

## Train coverage (1,214 companies / 21,157 company-months; holdout 72 left out of the %)

| name | pct company-months | pct companies | train mean | train p50 |
|------|-------------------:|--------------:|-----------:|----------:|
| `c_n_tx` | 100% | 100% | 114.20 | 43 |
| `c_n_days_with_tx` | 100% | 100% | 14.12 | 14 |
| `c_gap_sd` | 95.1% | 100% | 2.66 | 1.36 |
| `c_zero_in_month` | 100% | 100% | 0.118 | 0 |
| `c_zero_in_share_6` | 100% | 100% | 0.116 | 0 |
| `c_salary_month` | 100% | 100% | 0.401 | 0 |
| `c_tax_month` | 100% | 100% | 0.518 | 1 |
| `c_ss_month` | 100% | 100% | 0.457 | 0 |
| `c_missed_salary` | 100% | 100% | 0.029 | 0 |
| `c_missed_tax` | 100% | 100% | 0.083 | 0 |
| `c_recency_days` | 100% | 100% | 6.79 | 0 |
| `c_last_tx_before_2026_06` | 100% | 100% | 0.010 | 0 |

`overnight/verify_wave.py`: `OK analysis.features.ops rows=22230 companies=1286 cols=12`.

## What failed / caveats

- Nothing failed in this slot. Smoke test at `/tmp` passed prefix, keys, no duplicate `company_id,period`, Dec-2025 `c_n_tx` match, independent `c_gap_sd` check, non-negative recency.
- Javier's extract-level count is **61** companies with last tx &lt; 2026-06-01. As of 2026-08-31 (no look-ahead) the flag is **62**. The extra company is `COMP_0981` (last tx 2025-04-07, then one tx on 2026-09-01). Using Sept-1 would leak into the August row.
- `c_gap_sd` is null on 4.9% of train company-months (fewer than 3 active days in the 90-day window), e.g. long-silent firms.
- `c_zero_in_*` is any positive amount, not Family A `op_in`. Intentional: Y6 forbids A/C same window; the two zero-inflow definitions are not identical.
- Weekday/seasonal profile from the plan was not requested and is not in this module.

## Next idea

- `c_missed_ss` with the same ≥3/6 rule (686 companies have social_security).
- Weekday mix + month-of-year seasonality (plan Family C leftover).
- Burstiness `c_n_tx / c_n_days_with_tx` as a complement to `c_gap_sd`.
