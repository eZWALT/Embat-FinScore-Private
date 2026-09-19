# Wave 1 slot 4 — Family F (debt and financing)

- **Owner files:** `analysis/features/debt.py`, this note
- **API:** `SOURCE_TABLES=["debt_products","debt_schedule_config","transactions"]`, `FAMILY="f"`, `build(con, grid) -> company_id, period, f_*`
- **Smoke:** `/tmp/wave1_slot4_smoke.py` (log `/tmp/wave1_slot4_smoke.txt`). `overnight/verify_wave.py` → `OK analysis.features.debt rows=22230 companies=1286 cols=15`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`

## Files written

- `analysis/features/debt.py`
- `overnight/waves/wave1_slot4.md`

## Columns (15)

Time-varying from transactions (in3 / ds3 / fc3 computed here; cashflow not imported):

| column | definition |
|--------|------------|
| `f_debt_service` | −sum(debt_repayment) in the period |
| `f_fin_cost` | −sum(fee + interest_charge) in the period |
| `f_ds_r` | ds3 / max(in3, 1), clip [0, 2] — 3 months / 13 weeks |
| `f_fc_r` | fc3 / max(in3, 1), clip [0, 1] |

Snapshot / as-of `created_at` (outstanding/granted are the 2026-09-01 extract; **not** a 2024 book):

| column | definition |
|--------|------------|
| `f_n_facilities` | debt products with created_at ≤ period_end, not `created_after_snapshot` |
| `f_n_types` | distinct `type` on that as-of set |
| `f_util_snapshot` | sum\|outstanding\| / sum\|granted\| where \|granted\| > 1; clip [0, 5]; **null except last period** (2026-08 / last week) |
| `f_has_loc` / `f_has_factoring` / `f_has_confirming` | 0/1 on the as-of set |
| `f_w_rate` | granted_balance-weighted `annual_interest_rate_or_spread` (schedule) |
| `f_months_to_next_pay` | (earliest snapshot `next_payment_date` − period_end) / 30.4375 |
| `f_sched_vs_obs` | (granted_balance / total_periods, frequency-scaled) / max(observed debt_service, 1), clip [0, 20] |
| `f_outstanding_gt_granted` | clean flag; **null except last period** |
| `f_new_facility` | count of products whose created_at falls in the period |

Sign note: `debt_products.granted` / `outstanding` are typically **negative** (5 rows with granted>0). Util uses abs. `is_extreme` txs excluded from flows; `is_dup` kept.

## Train coverage (1,214 companies, 21,157 company-months; holdout excluded)

| column | % company-months | % companies | notes |
|--------|----------------:|------------:|-------|
| `f_debt_service`, `f_fin_cost` | 100.0 | 100.0 | 0 if no flow that month |
| `f_ds_r`, `f_fc_r` | 88.5 | 100.0 | first 2 months NA (need a 3-month window) |
| `f_n_facilities`, `f_n_types`, `f_has_*`, `f_new_facility` | 100.0 | 100.0 | mostly zeros |
| `f_util_snapshot` | 1.58 | 27.5 | last month only; 334/1,214 train cos with \|granted\|>1 |
| `f_outstanding_gt_granted` | 5.74 | 100.0 | last month only; 32/1,214 = 2.6% flagged |
| `f_w_rate`, `f_months_to_next_pay`, `f_sched_vs_obs` | 1.74 | 3.13 | **38 train / 40 schedule companies**; 87 schedule rows |

`f_ds_r` / `f_fc_r` match `score_pipeline.debt_serv_r` / `fin_cost_r` at corr 1.0 (MAE 0 on `f_debt_service`).

## Share of companies with any debt

- **debt_products:** 378 / 1,286 = **29.4%** (train: ~29.5%)
- **any `f_debt_service` > 0:** 523 / 1,286 = **40.7%** (241 companies repay with no debt_products row)
- **union:** 48.0%
- **schedule:** 40 / 1,286 = **3.1%** (87 product rows) — coverage is genuinely that thin

## What failed / honest limits

- Nothing failed in the Family F contract check.
- `next_payment_date` is stale: 82/87 dates are ≤ 2026-09-01, so `f_months_to_next_pay` at 2026-08 is mostly **negative** (train last-month median −5.5 months). It is not a reliable “months until next installment” clock.
- `f_sched_vs_obs` hits the clip of 20 when observed service is ~0 (last-month 75th percentile already 20).
- Utilisation cannot be a 2024–2025 panel feature; emitting it only on the last period is intentional.
- Weekly grid works (96,165 rows); 13-week roll for the ratios.

## Next idea

Rebuild a payment calendar from `last_payment_date` + `amortising_frequency` (and/or match `debt_repayment` to `settlement_product_id`) so `f_months_to_next_pay` and `f_sched_vs_obs` are time-varying instead of extract-dated. Until then Y4 should treat schedule columns as rare, last-month-only snapshot flags.
