# Wave 4 — trail length / Q6 honesty (one note)

Agent `6bf54618`. Lane: `analysis/evaluate/trail_length.py`. No commit. No parquet/duckdb rewrite. No `product/`. No 0–100. Holdout 72 is coverage only. Fixed cuts 6 / 12 / 18 / 24 — no train quantiles as features. Seed 20260918 never used.

## Files written

- `analysis/evaluate/trail_length.py` — create; same-module write→run cuts (bank trail, 73.6% replica, histogram, size/subsidiary, Y rates, Q6 lags, group wave, `g_*` constants, lag-missing why, honest lead × trail, zero-account cash, ERP×trail, post-snapshot, invoice clipped to grid, Y7 pre-grid, Sep-2026 extract, holdout groups + Y4 HHI coverage, Y3 long=2026-02 sliver, 24m company-level Q6)
- `analysis/outputs/trail_length.md` — tables + PARK
- `analysis/outputs/trail_length_months_on_book.png` — stacked months-on-book (first tx 2024-09 vs late)
- `analysis/experiments/registry.csv` — append-only coverage rows (`6bf54618`; train rates/cuts; holdout rows tagged `split=holdout` after a first mis-tag as train)
- `overnight/waves/wave4_trail_length.md` — this note

Did not edit `y11_dark.py`, `gbm_i_lift.py`, `y4_why.py`, family modules, parquet, `product/`, or invent a Y.

## Columns (read-only + derived in this module)

Read: `clean` banking_products / debt_products / invoices / transactions / companies; `monthly.parquet` (`a_in3`, `d_cust_hhi`, `e_ar_issued`, `g_n_accounts`, `g_created_*`); `targets.parquet` (`y3_recover_cash_6m`, `y2_neg_2of3`, `y7_top1_lost`, `y4_ds_r_double`).

Derived (not written to parquet): `n_grid_months`, `n_tx_months`, `months_so_far`, `d_cust_hhi_lag3` = company-groupby shift(3), `e_ar_issued_lag1` = shift(1), trail buckets, honest = so-far − k.

## Coverage (train; quote these)

- **First banking `created_at` after 2024-09-01:** **891 / 1,211 = 73.6%**. Join QA **CONFIRMED** (count + share). No banking row: 3. Stricter first-created *month* ≥ Oct-2024: 859/1,211 = 70.9%. Sep-2024 first connection: 32 (inside the 73.6% because QA used `>` not month>). First debt created after 2024-09-01: 285/358 = 79.6%.
- **Bank trail (first tx after 2024-09):** **779 / 1,214 = 64.2%**. Connection clock ≠ trail. `created_at` after first tx: 851 (70.3% of banking books).
- **Months-on-book (official grid, first tx → 2026-08):** <6 = **0.2%** (3), <12 = **28.8%** (350), ≥18 = **58.8%** (714), =24 = **35.8%** (435). Median 20; mean 17.4. Any gap 15.6%.
- **Holdout 72 (coverage):** late first-tx 69/72 = 95.8%. <12 = 26.4% (19) ≈ train 28.8%. ≥18 = 33.3% vs train 58.8%. =24 = 4.2% (3) vs 35.8%. Median 15 vs 20. The hidden 72 is missing the **24-month pile**, not a <12 pile. 12/15 groups first appear after 2024-09; 65/72 sit in those new groups; 0 train siblings.
- **Ever-ERP:** 744 (61.3%), 470 dark. First iss after 2024-09: 367/744 (49.3%). **43.5%** of ever-ERP have ≥1 issuance month *before* first tx (pass 15 clip).
- **Left-trunc size:** late median log1p(a_in3) 12.669 vs full 12.493, Δ +0.176, MW p=0.1045 — **not smaller**. Of 779 late: new subsidiaries 113 (14.5%), new-group members 666 (85.5%).
- **Group wave:** 235 train groups: all_full 45, all_late_same_month 85 (40 solo / 45 multi, spread 2024-10→2026-05, fattest 2026-01/02), all_late_staggered 58, mixed 47. Late companies: 28.5% same-month wave, **57.0% staggered**, 14.5% mixed. Short trail = new group arriving, mostly staggered.
- **`g_created_*` CONSTANTS ≠ 73.6%.** Those flags are null/post-snapshot quality (0 null `created_at`; post-snap dropped). The 73.6% surface is `g_n_accounts=0` (12.4% of train cm; 63.7% of first grid months). 96.6% of zero-account months still have txs. PARK `created_at` as a health Y.

## Y base rates short (<12) vs long (≥18), train labeled

None only-defined-on-long. Bucket 24 labeled = 0 (horizon). Y3 long so-far = **213 rows, all 2026-02, all first-tx 2024-09** — a calendar sliver; prefer company-total.

| Y | so-far short | so-far long | company short | company long |
| --- | ---: | ---: | ---: | ---: |
| y3_recover | 6.8% (3,723) | 7.5% (213) | 8.9% | 6.9% |
| y2_neg_2of3 | 7.9% (11,186) | 6.2% (1,908) | 8.0% | 7.3% |
| y7_top1_lost | 30.2% (4,421) | 22.8% (916) | 22.2% | 29.7% |
| y4_ds_r_double | 13.9% (1,334) | 10.1% (297) | 13.4% | 13.9% |

ERP×trail: Y2 never-ERP is the reverse (8.4% short vs 10.4% long). Short trail is **not** a miss indicator. **PARK `y_short_trail`.**

## Q6 lags — missing on short trails?

- **Y4 `d_cust_hhi_lag3`:** nn **35.8%** all train labeled, **21.7% short**, 53.2% long. Short miss 78.3% = shift 15.5% + calendar full6 38.8% + source NaN 45.7% (residual 0). Among short rows *with* the lag: honest≥6 = 66.2% (p50=6) vs 100% on long.
- **Y7 `e_ar_issued_lag1`:** nn **97.2%** / **95.2% short**. Hole is only so-far<2 (shift). Y7's short Q6 limit is **length** (honest≥6 = 47%, p50=5), not missingness. so-far=1 labels still have `e_ar_issued` (pre-grid invoices).
- **24-month train books are not a Y4 HHI bench:** 73/435 (**16.8%**) ever have a labeled Y4 row with HHI_lag3. Y7: 259/435 (59.5%), and every labeled 24m Y7 has the lag.
- **Holdout Y4 HHI:** 21/135 labeled (short 18/88). Only **2** of those 21 sit on the three 24-month books (COMP_0975 / 1177 / 1236 — two of those have **0** Y4 labels: debt-service floor, not missing trail). Late-book Y4+HHI from 7 companies. Quote **train CV** for any lead-time claim.

## What failed / next idea

Join-QA 73.6% is a *connection* clock, not first-tx (64.2%) and not the `g_created_*` constants. First invoice-month draft counted off the bank grid; clipped in pass 15. August quiet is not a one-month gap (2/121 reappear in Sep-2026) — still PARK as a death Y.

**Legal next (not this lane):** do not invent `y_short_trail` or a `created_at` health Y. If someone wants the 121 quiet tail, it must beat Y2 as an activity drop. Family G should keep `g_n_accounts` / `g_new_this_month` as the 73.6% surface. Q6 sentences for Y4 need an ERP HHI *and* 3 months on the panel *and* a ds_r pair — a 24-month book is not enough (16.8%).
