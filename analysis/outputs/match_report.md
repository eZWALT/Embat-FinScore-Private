# Family J — amount-match rates

Train-only. Holdout 72 out. Company-month *rates*, not invoice-tx pairs.
Greedy 1-1 inside `(company, period, sign, cents)`. `|Δ| ≤ 0.01` from QA;
not retuned. `build()` does not read or write the parquet.

## Contract

- `SOURCE_TABLES = ["transactions", "invoices"]`, `FAMILY = "j"`.
- Period P uses payment/issuance/tx dates ≤ period end. Last raw date in
  store is 2026-09-01; those rows stay off August (`date < 2026-09-01`).
- `j_pay_match` = share of *paid this month* book invoices with a
  same-company same-sign tx in the payment month, greedy 1-1.
  **NaN if no paid invoices** (470 stay NaN, not 0).
- `j_iss_match` = same on issuance-month (QA 21.2%). Diagnostic.
- `j_has_book` = 1 if the company had a book invoice with
  `issuance_date ≤ period end` (causal miss flag).
- `j_pay_match_t3` = `sum(matched) / sum(paid)` over t-2..t,
  `min_periods=3`. Not the mean of monthly rates.
- Not emitted: row FK, COMP↔CP map, `j_interco_*`, D-family HHI.

## 470 NaN check (hard rule)

- Never-ERP train companies (max `j_has_book` = 0): **470**.
- `j_pay_match` non-null on those cm: **0.0%** (must be 0).
- Mean `j_has_book` on those cm: **0.000** (must be 0).
- Train cm with ≥1 paid invoice: **9,648** / 21,157.

## Coverage / size / acf

| feature | cov_all_cm | cov_ever-ERP_cm | cov_co | size_ρ vs log1p(a_in3) | acf1 | acf3 | modal% | n_unique | mean | flags | decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `j_pay_match` | 45.6% | 71.2% | 56.9% | 2.16e-04 | 0.226 | 0.077 | 15.0% | 2366 | 0.392 | — | **KEEP-Q5** |
| `j_iss_match` | 52.8% | 82.5% | 61.2% | 0.018 | 0.262 | 0.096 | 19.6% | 2623 | 0.236 | — | **KEEP-Q5** |
| `j_has_book` | 100.0% | 100.0% | 100.0% | -0.015 | 0.835 | 0.627 | 60.3% | 2 | 0.603 | — | **KEEP-Q5** |
| `j_pay_match_t3` | 45.2% | 70.5% | 56.9% | 0.022 | 0.689 | -0.003 | 10.2% | 4112 | 0.379 | — | **CLOSE** |
| `j_pay_unmatched` | 45.6% | 71.2% | 56.9% | -2.16e-04 | 0.226 | 0.077 | 15.0% | 2366 | 0.608 | — | **PARK** |
| `j_n_paid` | 45.6% | 71.2% | 56.9% | 0.450 | 0.283 | 0.078 | 7.8% | 378 | 49.802 | — | **PARK** |
| `j_n_matched` | 45.6% | 71.2% | 56.9% | 0.415 | 0.331 | 0.110 | 15.0% | 183 | 16.730 | — | **PARK** |
| `j_pay_set` | 45.6% | 71.2% | 56.9% | -0.002 | 0.228 | 0.071 | 15.0% | 2425 | 0.411 | — | **PARK** |
| `j_pay_wrong` | 45.6% | 71.2% | 56.9% | 0.203 | -0.056 | -0.070 | 76.4% | 753 | 0.015 | — | **PARK** |

## Pair / complement ρ

- `j_pay_match` vs `j_iss_match`: **0.678** (drop iss if |ρ| ≥ 0.9).
- `j_pay_match` vs `j_pay_unmatched`: **-1.000** (expect −1; emit only one).
- `j_pay_match` vs `j_pay_match_t3`: **0.848**.

## 1-1 vs QA set-overlap (train cm with paid invoices)

- Cm-mean `j_pay_match` (greedy 1-1): **39.2%**.
- Cm-mean set-overlap (QA-style any-tx-exists): **41.1%**.
- Invoice-weighted 1-1 (`sum n_matched / sum n_paid`): **33.6%**.
- Invoice-weighted set-overlap: **36.7%** (QA KEEP headline 35.7%).
- Mean wrong-sign control: **1.5%** (report only).

## Rewrite ρ vs `e_*` (PARK if |ρ| ≥ 0.95)

| feature | e_* | Spearman |
| --- | --- | ---: |
| `j_pay_match` | `e_ar_issued` | -0.079 |
| `j_pay_match` | `e_delay_coll` | 0.057 |
| `j_pay_match` | `e_pending_amt_share` | -0.093 |
| `j_pay_match` | `e_ar_open` | -0.051 |
| `j_pay_match` | `e_dso_proxy` | -0.017 |
| `j_iss_match` | `e_ar_issued` | -0.011 |
| `j_iss_match` | `e_delay_coll` | 0.045 |
| `j_iss_match` | `e_pending_amt_share` | -0.344 |
| `j_iss_match` | `e_ar_open` | -0.077 |
| `j_iss_match` | `e_dso_proxy` | -0.194 |
| `j_has_book` | `e_ar_issued` | 0.280 |
| `j_has_book` | `e_delay_coll` | — |
| `j_has_book` | `e_pending_amt_share` | — |
| `j_has_book` | `e_ar_open` | 0.316 |
| `j_has_book` | `e_dso_proxy` | — |
| `j_pay_match_t3` | `e_ar_issued` | -0.078 |
| `j_pay_match_t3` | `e_delay_coll` | 0.034 |
| `j_pay_match_t3` | `e_pending_amt_share` | -0.091 |
| `j_pay_match_t3` | `e_ar_open` | -0.051 |
| `j_pay_match_t3` | `e_dso_proxy` | -0.040 |
| `j_pay_unmatched` | `e_ar_issued` | 0.079 |
| `j_pay_unmatched` | `e_delay_coll` | -0.057 |
| `j_pay_unmatched` | `e_pending_amt_share` | 0.093 |
| `j_pay_unmatched` | `e_ar_open` | 0.051 |
| `j_pay_unmatched` | `e_dso_proxy` | 0.017 |
| `j_n_paid` | `e_ar_issued` | 0.551 |
| `j_n_paid` | `e_delay_coll` | 0.020 |
| `j_n_paid` | `e_pending_amt_share` | -0.119 |
| `j_n_paid` | `e_ar_open` | 0.402 |
| `j_n_paid` | `e_dso_proxy` | -0.031 |
| `j_n_matched` | `e_ar_issued` | 0.429 |
| `j_n_matched` | `e_delay_coll` | 0.042 |
| `j_n_matched` | `e_pending_amt_share` | -0.151 |
| `j_n_matched` | `e_ar_open` | 0.310 |
| `j_n_matched` | `e_dso_proxy` | -0.046 |
| `j_pay_set` | `e_ar_issued` | -0.075 |
| `j_pay_set` | `e_delay_coll` | 0.065 |
| `j_pay_set` | `e_pending_amt_share` | -0.098 |
| `j_pay_set` | `e_ar_open` | -0.049 |
| `j_pay_set` | `e_dso_proxy` | -0.018 |
| `j_pay_wrong` | `e_ar_issued` | 0.273 |
| `j_pay_wrong` | `e_delay_coll` | 0.051 |
| `j_pay_wrong` | `e_pending_amt_share` | -0.037 |
| `j_pay_wrong` | `e_ar_open` | 0.219 |
| `j_pay_wrong` | `e_dso_proxy` | 0.021 |

## Spearman vs Y (train labeled)

| feature | y | n_labeled | Spearman |
| --- | --- | ---: | ---: |
| `j_pay_match` | `y3_recover_cash_6m` | 5,648 | -0.060 |
| `j_pay_match` | `y7_top1_lost` | 7,464 | 0.051 |
| `j_pay_match` | `y8_inv_worse_6` | 3,883 | 0.015 |
| `j_pay_match` | `y8_cash_worse_6` | 6,986 | -0.042 |
| `j_iss_match` | `y3_recover_cash_6m` | 5,648 | -0.027 |
| `j_iss_match` | `y7_top1_lost` | 7,464 | 0.047 |
| `j_iss_match` | `y8_inv_worse_6` | 3,883 | -0.037 |
| `j_iss_match` | `y8_cash_worse_6` | 6,986 | -0.049 |
| `j_has_book` | `y3_recover_cash_6m` | 5,648 | -0.009 |
| `j_has_book` | `y7_top1_lost` | 7,464 | — |
| `j_has_book` | `y8_inv_worse_6` | 3,883 | — |
| `j_has_book` | `y8_cash_worse_6` | 6,986 | -0.036 |
| `j_pay_match_t3` | `y3_recover_cash_6m` | 5,648 | -0.063 |
| `j_pay_match_t3` | `y7_top1_lost` | 7,464 | 0.052 |
| `j_pay_match_t3` | `y8_inv_worse_6` | 3,883 | 0.031 |
| `j_pay_match_t3` | `y8_cash_worse_6` | 6,986 | -0.048 |
| `j_pay_unmatched` | `y3_recover_cash_6m` | 5,648 | 0.060 |
| `j_pay_unmatched` | `y7_top1_lost` | 7,464 | -0.051 |
| `j_pay_unmatched` | `y8_inv_worse_6` | 3,883 | -0.015 |
| `j_pay_unmatched` | `y8_cash_worse_6` | 6,986 | 0.042 |
| `j_n_paid` | `y3_recover_cash_6m` | 5,648 | -0.142 |
| `j_n_paid` | `y7_top1_lost` | 7,464 | -0.061 |
| `j_n_paid` | `y8_inv_worse_6` | 3,883 | 0.069 |
| `j_n_paid` | `y8_cash_worse_6` | 6,986 | 0.052 |
| `j_n_matched` | `y3_recover_cash_6m` | 5,648 | -0.155 |
| `j_n_matched` | `y7_top1_lost` | 7,464 | -0.018 |
| `j_n_matched` | `y8_inv_worse_6` | 3,883 | 0.063 |
| `j_n_matched` | `y8_cash_worse_6` | 6,986 | 0.013 |
| `j_pay_set` | `y3_recover_cash_6m` | 5,648 | -0.050 |
| `j_pay_set` | `y7_top1_lost` | 7,464 | 0.061 |
| `j_pay_set` | `y8_inv_worse_6` | 3,883 | 0.014 |
| `j_pay_set` | `y8_cash_worse_6` | 6,986 | -0.048 |
| `j_pay_wrong` | `y3_recover_cash_6m` | 5,648 | -0.065 |
| `j_pay_wrong` | `y7_top1_lost` | 7,464 | -0.013 |
| `j_pay_wrong` | `y8_inv_worse_6` | 3,883 | 0.051 |
| `j_pay_wrong` | `y8_cash_worse_6` | 6,986 | 0.059 |

## Y3 X vs Q5 diagnostic

- `j_pay_match` vs `y3_recover_cash_6m`: **-0.060**.
- `j_pay_unmatched` vs `y3_recover_cash_6m`: **0.060**.
- **J is not a Y3 X** (|ρ| < 0.08). **KEEP as a Q5 diagnostic / miss flag** (`j_has_book` + `j_pay_match` on ever-ERP months).

## Pass-2 residuals (paid + Y3-labeled months)

- `j_n_paid` vs Y3 raw: **-0.142** (looks like a Y3 X).
- same, rank-residual after `log1p(a_in3)`: **-3.04e-04** → size, PARK.
- same, rank-residual after `e_ar_issued`: **0.015** → PARK.
- `j_n_matched` vs Y3 | `n_paid`: **0.033** (no extra match-count signal).
- `j_iss_match` | `j_pay_match` vs Y3: **-0.012** (iss adds nothing for Y3).
- `j_has_book` is a causal step (156 train cos flip 0→1; 470 stay 0; 588 start 1).
  High acf1 (0.84) is the step, not a cycle. KEEP as a miss flag.

## Decisions (this pass)

- `j_pay_match`: **KEEP-Q5**
- `j_iss_match`: **KEEP-Q5**
- `j_has_book`: **KEEP-Q5**
- `j_pay_match_t3`: **CLOSE** — acf1=0.689 acf3=-0.003 (lag-3 = non-overlap)
- `j_pay_unmatched`: **PARK** — exact complement
- `j_n_paid`: **PARK**
- `j_n_matched`: **PARK**
- `j_pay_set`: **PARK**
- `j_pay_wrong`: **PARK**

## Look-ahead / construction notes

- Book filter = Family E: `document_type=invoice`, `status<>cancel`,
  `amount<>0`, `issuance_date` not null.
- Paid = `payment_date` not null and not `payment_date_invalid`.
- Monthly: `date_trunc('month', date) = period` and `date < 2026-09-01`.
- Weekly: `date_trunc('week', date)` (Monday, matches `W-MON`).
- Greedy 1-1 is amount-bucket `min(n_inv, n_tx)`, not a global optimizer.
- `j_has_book` uses `MIN(issuance_date) ≤ period_end` only — no future ERP.
- Did not merge into parquet. Did not edit FAMILIES. Did not rebuild Y8.

