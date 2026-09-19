# Family G — banking products QA (non-debt)

Generated `2026-09-19T00:13` UTC by `python -m analysis.evaluate.banking_g_qa`.
Holdout 72 (seed 20260918) is **coverage only**. Rates, tertiles, AUROC,
and PARK/CLOSE/KEEP are train. No parquet rewrite. No new GBM. No 0–100.
Complementary to `debt_schedule_qa.md` — this is the *non-debt* book.
`created_at` is a connection clock (trail + debt QA). Do not invent a health Y from onboarding.

## Headline

- Raw `banking_products`: **5,987 rows / 1,283 companies**. Null `created_at`: 0 (0.0%). Post-snapshot: 44 (dq_log 44). Train companies with a row: 1211; holdout coverage 72.
- `g_n_accounts` is a **created_at inventory panel**: YES, rise-only (1,561 rises, 0 drops, 18,382 flats). Same monotone as `f_n_facilities`. First-month p50=0 (63.7% still 0); last-month p50=3 (0.4% still 0).
- `g_new_this_month>0`: 1,680 CM (7.9%) / 1,016 companies. acf1=-0.077 (flag -0.083). Size ρ vs log1p(a_in3)=0.084. 52.8% of new months are a first-ever as-of account. **PARK as a health Y** — connection wave, not Q3 turning.
- Best `g_has_*` vs Y3 (stressed, group-fold, **oriented** max(auc,1−auc)): **g_has_card 0.551** (raw 0.449). `c_n_days_with_tx` oriented CV=0.711 (raw 0.289 = published 0.711 flipped). Size dummy oriented CV=0.617 (raw 0.383). No access flag beats oriented-size by ≥0.02 — **CLOSE as Y3 X**. Card × TPV 2×2 after size terciles: does not survive (TPV n=10 last-month companies). Do not KEEP as an operating type; do not revive clusters (sil 0.234).
- `g_created_after_snapshot` is **0 on every train CM** (True). Feature-report drop of the three `g_created_*` constants: **CONFIRM**. 44-col starter is 44 cols; `g_has_*` all in starter=True; `g_created_*` in starter=none. `g_n_accounts` / `g_new_this_month` are KEEP-list but **not** in the 44-col starter.
- 470 dark vs 744 invoiced: CONFIRM. Access ≠ ERP — dark companies still have checking; TPV is rare on both. `g_has_checking=0` is 99.1% the connection hole (`g_n_accounts=0`), not a mix flag.

## Brief questions

1. **Who is healthy?** — access flags (card / TPV / checking) are operating-type tags, not a health reading. They lose to size on Y3.
2. **Who is improving?** — `g_n_accounts` only rises. A higher count is more connections, not 45→65.
3. **Who is turning?** — `g_new_this_month` is a connection birth (acf ≈ 0, often first as-of account). PARK as Q3.
4. **Dip vs fall?** — not this table. Complementary to debt inventory, not a cash-path.
5. **Why did it change?** — card / TPV as a type tag: does not survive size terciles. Do not revive k-means (silhouette 0.234).
6. **Months earlier?** — `created_at` is not lead time. The 73.6% late first-created is `g_n_accounts=0` on early months, not the CONSTANT `g_created_*` columns.

## 1. Raw `banking_products`

| item | n |
| --- | ---: |
| rows | 5,987 |
| companies | 1,283 (train 1211 / holdout 72) |
| product_id | 5,987 |
| null created_at | 0 (0.0%) |
| created_after_snapshot | 44 |

Dictionary names checking / card / investment / tpv / saving / expensesPlatform.
`wallet`, `risk`, `lineofcomex` sit in **other** — Family G does not emit `g_has_*` for them (they still count in `g_n_accounts`).

| bucket | type | rows | companies | null created_at |
| --- | --- | ---: | ---: | ---: |
| checking | checking | 4854 | 1282 | 0 |
| card | card | 796 | 205 | 0 |
| investment | investment | 201 | 94 | 0 |
| other | wallet | 34 | 21 | 0 |
| other | risk | 25 | 5 | 0 |
| tpv | tpv | 25 | 10 | 0 |
| other | expensesPlatform | 23 | 7 | 0 |
| other | lineofcomex | 19 | 8 | 0 |
| saving | saving | 10 | 9 | 0 |

Named-type buckets (distinct companies; other is union of leftover types):

| bucket | rows | companies | null created_at |
| --- | ---: | ---: | ---: |
| checking | 4854 | 1282 | 0 |
| card | 796 | 205 | 0 |
| investment | 201 | 94 | 0 |
| other | 101 | 39 | 0 |
| tpv | 25 | 10 | 0 |
| saving | 10 | 9 | 0 |

## 2. `g_n_accounts` is a rise-only panel

Train panel: **21,157** company-months / **1,214** companies. Month-to-month: 1,561 rises, 0 drops, 18,382 flats. `g_n_banks` drops: 0. Inventory is monotone within company — same as-of `created_at` rule as debt facilities.

acf1=0.804 acf3=0.700 acf6=0.505. Spearman vs a_in3=0.421; vs log1p(a_in3)=0.421 (not SIZE; threshold |ρ|≥0.5 vs a_in3 / log1p).

| slice | mean | p50 | share 0 |
| --- | ---: | ---: | ---: |
| first month | 1.15 | 0 | 63.7% |
| last month | 4.60 | 3 | 0.4% |

First-month vs last-month count distribution (train companies):

| slice | g_n_accounts | n companies | share |
| --- | --- | ---: | --- |
| first_month | 0 | 773 | 63.7% |
| first_month | 1 | 171 | 14.1% |
| first_month | 2 | 105 | 8.6% |
| first_month | 3 | 49 | 4.0% |
| first_month | 4 | 39 | 3.2% |
| first_month | 5 | 16 | 1.3% |
| first_month | 6 | 17 | 1.4% |
| first_month | 7 | 10 | 0.8% |
| first_month | 8 | 3 | 0.2% |
| first_month | 9 | 8 | 0.7% |
| first_month | 10 | 3 | 0.2% |
| first_month | 11 | 3 | 0.2% |
| first_month | 12 | 4 | 0.3% |
| first_month | 13 | 4 | 0.3% |
| first_month | 14 | 2 | 0.2% |
| first_month | 15 | 1 | 0.1% |
| first_month | 16 | 1 | 0.1% |
| first_month | 19 | 1 | 0.1% |
| first_month | 21 | 1 | 0.1% |
| first_month | 27 | 1 | 0.1% |
| first_month | 34 | 1 | 0.1% |
| first_month | 37 | 1 | 0.1% |
| last_month | 0 | 5 | 0.4% |
| last_month | 1 | 292 | 24.1% |
| last_month | 2 | 250 | 20.6% |
| last_month | 3 | 153 | 12.6% |
| last_month | 4 | 129 | 10.6% |
| last_month | 5 | 97 | 8.0% |
| last_month | 6 | 58 | 4.8% |
| last_month | 7 | 38 | 3.1% |
| last_month | 8 | 27 | 2.2% |
| last_month | 9 | 29 | 2.4% |
| last_month | 10 | 21 | 1.7% |
| last_month | 11 | 22 | 1.8% |
| last_month | 12 | 13 | 1.1% |
| last_month | 13 | 11 | 0.9% |
| last_month | 14 | 10 | 0.8% |
| last_month | 15 | 11 | 0.9% |
| last_month | 16 | 9 | 0.7% |
| last_month | 17 | 5 | 0.4% |
| last_month | 18 | 2 | 0.2% |
| last_month | 19 | 3 | 0.2% |
| last_month | 20 | 1 | 0.1% |
| last_month | 21 | 5 | 0.4% |
| last_month | 22 | 1 | 0.1% |
| last_month | 23 | 4 | 0.3% |
| last_month | 24 | 2 | 0.2% |
| last_month | 26 | 1 | 0.1% |
| last_month | 27 | 2 | 0.2% |
| last_month | 28 | 3 | 0.2% |
| last_month | 30 | 1 | 0.1% |
| last_month | 34 | 1 | 0.1% |
| last_month | 36 | 1 | 0.1% |
| last_month | 37 | 1 | 0.1% |
| last_month | 39 | 2 | 0.2% |
| last_month | 43 | 1 | 0.1% |
| last_month | 45 | 1 | 0.1% |
| last_month | 49 | 1 | 0.1% |
| last_month | 65 | 1 | 0.1% |

First-month zeros are the 73.6% connection clock (`created_at` after first tx), not a missing product. Last-month zeros are the handful with no on-panel banking row (trail QA: 5 never-`g_n_accounts>0`).

## 3. `g_new_this_month` — connection, not Q3

Prevalence: 1,680 / 21,157 CM = 7.9% (1,016 companies). Mean 0.210, p50 0.
acf1=-0.077 (flag -0.083), acf3=-0.082, acf6=-0.083 — low persistence, same shape as `f_new_facility`.
Size ρ vs log1p(a_in3)=0.084 (flag 0.081) — not SIZE.
First grid month has `g_new>0` on 9.8%; `g_n_accounts=0` on 63.7% (cash can start before the product is connected).
Of 1,680 new months, 887 (52.8%) are a first-ever as-of account (n_birth=1,209). That is onboarding / connection, not a turn.

**PARK** `g_new_this_month` as a health Y and as a Q3 turning flag. Keep it as an inventory clock (already on the feature-report KEEP list, not in the 44-col starter).

New-connection calendar (train CM with `g_new>0`):

| month | n CM | n companies |
| --- | ---: | ---: |
| 2024-09 | 43 | 43 |
| 2024-10 | 41 | 41 |
| 2024-11 | 72 | 72 |
| 2024-12 | 46 | 46 |
| 2025-01 | 55 | 55 |
| 2025-02 | 45 | 45 |
| 2025-03 | 66 | 66 |
| 2025-04 | 59 | 59 |
| 2025-05 | 77 | 77 |
| 2025-06 | 42 | 42 |
| 2025-07 | 50 | 50 |
| 2025-08 | 18 | 18 |
| 2025-09 | 98 | 98 |
| 2025-10 | 89 | 89 |
| 2025-11 | 73 | 73 |
| 2025-12 | 66 | 66 |
| 2026-01 | 83 | 83 |
| 2026-02 | 109 | 109 |
| 2026-03 | 172 | 172 |
| 2026-04 | 142 | 142 |
| 2026-05 | 78 | 78 |
| 2026-06 | 58 | 58 |
| 2026-07 | 69 | 69 |
| 2026-08 | 29 | 29 |

## 4. Access flags vs Q1 (Y2 / Y3 stressed / Y1 last-value)

Train labeled: Y2 n=17,356 base 7.3%; Y3 n=5,648 base 7.1% (stressed-only by construction).
Single-feature AUROC. Raw = score as-is. Oriented = max(auc, 1−auc) — the published `c_n_days_with_tx` 0.711 is the flipped raw ~0.289. Group-fold CV (5, seed 20260918) is the quote. Holdout not used.

| Y | feature | pooled raw | pooled orient | CV raw | CV orient | sd | n labeled | ρ vs a_in3 | ρ vs log1p(a_in3) |
| --- | --- | --- | --- | --- | --- | --- | ---: | --- | --- |
| y2_neg_2of3 | g_has_card | 0.497 | 0.503 | 0.508 | 0.508 | 0.050 | 17356 | 0.1506 | 0.1505 |
| y2_neg_2of3 | g_has_tpv | 0.498 | 0.502 | 0.498 | 0.502 | 0.003 | 17356 | 0.03927 | 0.03926 |
| y2_neg_2of3 | g_has_checking | 0.478 | 0.522 | 0.454 | 0.546 | 0.063 | 17356 | 0.03135 | 0.03137 |
| y2_neg_2of3 | g_n_banks | 0.519 | 0.519 | 0.481 | 0.519 | 0.103 | 17356 | 0.3654 | 0.3655 |
| y2_neg_2of3 | g_n_accounts | 0.501 | 0.501 | 0.471 | 0.529 | 0.098 | 17356 | 0.4213 | 0.4214 |
| y2_neg_2of3 | g_new_gt0 | 0.495 | 0.505 | 0.500 | 0.500 | 0.011 | 17356 | 0.08116 | 0.08114 |
| y2_neg_2of3 | c_n_days_with_tx | 0.577 | 0.577 | 0.571 | 0.571 | 0.046 | 17356 | 0.5948 | 0.5953 |
| y2_neg_2of3 | log1p_a_in3 | 0.540 | 0.540 | 0.552 | 0.552 | 0.046 | 17356 | 0.9999 | 1 |
| y3_recover_cash_6m | g_has_card | 0.451 | 0.549 | 0.449 | 0.551 | 0.034 | 5648 | 0.1506 | 0.1505 |
| y3_recover_cash_6m | g_has_tpv | 0.498 | 0.502 | 0.498 | 0.502 | 0.004 | 5648 | 0.03927 | 0.03926 |
| y3_recover_cash_6m | g_has_checking | 0.494 | 0.506 | 0.494 | 0.506 | 0.036 | 5648 | 0.03135 | 0.03137 |
| y3_recover_cash_6m | g_n_banks | 0.385 | 0.615 | 0.388 | 0.612 | 0.068 | 5648 | 0.3654 | 0.3655 |
| y3_recover_cash_6m | g_n_accounts | 0.415 | 0.585 | 0.419 | 0.581 | 0.092 | 5648 | 0.4213 | 0.4214 |
| y3_recover_cash_6m | g_new_gt0 | 0.501 | 0.501 | 0.499 | 0.501 | 0.010 | 5648 | 0.08116 | 0.08114 |
| y3_recover_cash_6m | c_n_days_with_tx | 0.277 | 0.723 | 0.289 | 0.711 | 0.031 | 5648 | 0.5948 | 0.5953 |
| y3_recover_cash_6m | log1p_a_in3 | 0.380 | 0.620 | 0.383 | 0.617 | 0.061 | 5648 | 0.9999 | 1 |

KEEP-as-Y3-X rule: **oriented** CV beats oriented size dummy by ≥0.02 **and** not SIZE (|ρ| vs a_in3 or log1p ≥ 0.5). First-cut as-is comparison was wrong — size raw 0.38 is inverse-size skill 0.62; chance flags at 0.50 are not a win.

| feature | CV raw | CV orient | size orient | days orient | Δ size | Δ days | SIZE? | KEEP as Y3 X? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_card | 0.449 | 0.551 | 0.617 | 0.711 | -0.065 | -0.160 | False | False |
| g_has_tpv | 0.498 | 0.502 | 0.617 | 0.711 | -0.115 | -0.210 | False | False |
| g_has_checking | 0.494 | 0.506 | 0.617 | 0.711 | -0.111 | -0.206 | False | False |
| g_n_banks | 0.388 | 0.612 | 0.617 | 0.711 | -0.005 | -0.099 | False | False |
| g_n_accounts | 0.419 | 0.581 | 0.617 | 0.711 | -0.036 | -0.130 | False | False |
| g_new_gt0 | 0.499 | 0.501 | 0.617 | 0.711 | -0.116 | -0.210 | False | False |

Base rates by access (train company-months):

| flag | group | Y | n labeled | n pos | base rate | n CM | n companies |
| --- | --- | --- | ---: | ---: | --- | ---: | ---: |
| g_has_card | has=1 | y2_neg_2of3 | 2196 | 154 | 0.07013 | 2788 | 197 |
| g_has_card | has=0 | y2_neg_2of3 | 15160 | 1117 | 0.07368 | 18369 | 1168 |
| g_has_card | has=1 | y3_recover_cash_6m | 981 | 33 | 0.03364 | 2788 | 197 |
| g_has_card | has=0 | y3_recover_cash_6m | 4667 | 369 | 0.07907 | 18369 | 1168 |
| g_has_tpv | has=1 | y2_neg_2of3 | 67 | 1 | 0.01493 | 95 | 10 |
| g_has_tpv | has=0 | y2_neg_2of3 | 17289 | 1270 | 0.07346 | 21062 | 1213 |
| g_has_tpv | has=1 | y3_recover_cash_6m | 25 | 0 | 0 | 95 | 10 |
| g_has_tpv | has=0 | y3_recover_cash_6m | 5623 | 402 | 0.07149 | 21062 | 1213 |
| g_has_checking | has=1 | y2_neg_2of3 | 14821 | 1034 | 0.06977 | 18516 | 1208 |
| g_has_checking | has=0 | y2_neg_2of3 | 2535 | 237 | 0.09349 | 2641 | 779 |
| g_has_checking | has=1 | y3_recover_cash_6m | 5178 | 364 | 0.0703 | 18516 | 1208 |
| g_has_checking | has=0 | y3_recover_cash_6m | 470 | 38 | 0.08085 | 2641 | 779 |
| g_has_saving | has=1 | y2_neg_2of3 | 41 | 0 | 0 | 68 | 9 |
| g_has_saving | has=0 | y2_neg_2of3 | 17315 | 1271 | 0.0734 | 21089 | 1214 |
| g_has_saving | has=1 | y3_recover_cash_6m | 9 | 0 | 0 | 68 | 9 |
| g_has_saving | has=0 | y3_recover_cash_6m | 5639 | 402 | 0.07129 | 21089 | 1214 |
| g_has_investment | has=1 | y2_neg_2of3 | 788 | 57 | 0.07234 | 1040 | 88 |
| g_has_investment | has=0 | y2_neg_2of3 | 16568 | 1214 | 0.07327 | 20117 | 1206 |
| g_has_investment | has=1 | y3_recover_cash_6m | 328 | 13 | 0.03963 | 1040 | 88 |
| g_has_investment | has=0 | y3_recover_cash_6m | 5320 | 389 | 0.07312 | 20117 | 1206 |
| g_n_banks | one_bank | y2_neg_2of3 | 6393 | 341 | 0.05334 | 7945 | 666 |
| g_n_banks | many_banks | y2_neg_2of3 | 8443 | 697 | 0.08255 | 10594 | 724 |
| g_n_banks | zero_banks | y2_neg_2of3 | 2520 | 233 | 0.09246 | 2618 | 773 |
| g_n_banks | one_bank | y3_recover_cash_6m | 1715 | 192 | 0.112 | 7945 | 666 |
| g_n_banks | many_banks | y3_recover_cash_6m | 3467 | 172 | 0.04961 | 10594 | 724 |
| g_n_banks | zero_banks | y3_recover_cash_6m | 466 | 38 | 0.08155 | 2618 | 773 |

Y1 last-value story (train labeled rows). Inflow last-value = `a_op_in` vs `y1_in_h1`; liquidity = `b_liq` vs `y1_liq_h1`. Access should not rewrite 'last-value wins liq / hist wins inflow'.

| series | slice | n | Spearman | median |err| |
| --- | --- | ---: | --- | --- |
| op_in | all | 19943 | 0.765 | 40008.660 |
| op_in | g_has_card has=1 | 2591 | 0.831 | 80178.370 |
| op_in | g_has_card has=0 | 17352 | 0.749 | 35000.000 |
| op_in | g_has_tpv has=1 | 85 | 0.933 | 141625.510 |
| op_in | g_has_tpv has=0 | 19858 | 0.764 | 39955.900 |
| op_in | g_has_checking has=1 | 17308 | 0.772 | 39669.805 |
| op_in | g_has_checking has=0 | 2635 | 0.723 | 44783.590 |
| liq | all | 19746 | 0.909 | 21103.855 |
| liq | g_has_card has=1 | 2581 | 0.901 | 26838.280 |
| liq | g_has_card has=0 | 17165 | 0.910 | 20362.440 |
| liq | g_has_tpv has=1 | 85 | 0.606 | 45898.260 |
| liq | g_has_tpv has=0 | 19661 | 0.910 | 21000.100 |
| liq | g_has_checking has=1 | 17184 | 0.912 | 21120.950 |
| liq | g_has_checking has=0 | 2562 | 0.892 | 21053.645 |

## 5. `g_created_*` constants + 44-col starter

`g_created_after_snapshot` all-zero on the monthly train panel: **True**. Last as-of cut is `created_at < 2026-09-01`, so the 44 post-extract rows never enter. This is **not** the 73.6% late-first-created fact (that is `g_n_accounts=0`).

| split | column | cov CM | n unique | modal | share 0 | max |
| --- | --- | --- | ---: | --- | --- | --- |
| train | g_created_unknown_share | 87.6% | 1 | 0.0% | 1 | 0 |
| train | g_created_after_snapshot | 100.0% | 1 | 0.0% | 1 | 0 |
| train | g_created_after_snapshot_share | 87.6% | 1 | 0.0% | 1 | 0 |
| holdout | g_created_unknown_share | 90.8% | 1 | 0.0% | 1 | 0 |
| holdout | g_created_after_snapshot | 100.0% | 1 | 0.0% | 1 | 0 |
| holdout | g_created_after_snapshot_share | 90.8% | 1 | 0.0% | 1 | 0 |

Starter set length **44** (is 44: True). G columns in starter: `g_custom_share, g_has_card, g_has_checking, g_has_investment, g_has_saving, g_has_tpv`. Drop `g_created_*`: **CONFIRM**. Keep `g_has_*` in starter: **CONFIRM** (g_has_card, g_has_checking, g_has_investment, g_has_saving, g_has_tpv). `g_n_accounts` in starter: False. `g_new_this_month` in starter: False. Those two stay on the broader KEEP list as inventory clocks, not in the 44-col GBM starter. **CAUTION:** `g_has_checking` is in the 44 but is 99% the connection hole, not a mix flag — consider dropping it from the starter later; do not edit products.py for that.

## 6. 470 dark vs 744 invoiced (access ≠ ERP)

Train last-month companies: 1,214. Ever-ERP 744 / never-ERP 470 (CONFIRM 744 / 470). Holdout ever-ERP coverage only: 40 / 72.

Last-month as-of inventory:

| group | n companies | accounts mean | accounts p50 | share 0 accounts | has_card | has_tpv | has_checking | has_saving | has_invest | n_banks p50 | custom share p50 |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ever_erp | 744 | 4.488 | 3.000 | 0.004032 | 0.1734 | 0.0121 | 0.996 | 0.009409 | 0.06183 | 2 | 0 |
| never_erp | 470 | 4.787 | 3.000 | 0.004255 | 0.1447 | 0.002128 | 0.9936 | 0.004255 | 0.08936 | 2 | 0 |

Company-ever (max on the panel):

| group | n | accounts mean | accounts p50 | ever card | ever TPV | ever checking |
| --- | ---: | --- | --- | --- | --- | --- |
| ever_erp | 744 | 4.488 | 3.000 | 0.1734 | 0.0121 | 0.996 |
| never_erp | 470 | 4.787 | 3.000 | 0.1447 | 0.002128 | 0.9936 |

## 7. One bank vs many vs Y2

Train last-month: one bank 485 (40.0%), many 724, zero 5.

Company-ever Y2 (max of labeled months):

| group | n companies | n with a Y2 label | n ever Y2 | ever-Y2 rate |
| --- | ---: | ---: | ---: | --- |
| one_bank | 485 | 471 | 58 | 12.3% |
| many_banks | 724 | 722 | 112 | 15.5% |
| zero_banks | 5 | 2 | 0 | 0.0% |

Company-month Y2:

| group | n CM | n labeled | n pos | Y2 rate |
| --- | ---: | ---: | ---: | --- |
| one_bank | 7945 | 6393 | 341 | 0.05334 |
| many_banks | 10594 | 8443 | 697 | 0.08255 |
| zero_banks | 2618 | 2520 | 233 | 0.09246 |

Top raw `bank_name` (all companies, extract stock):

| bank | rows | companies |
| --- | ---: | ---: |
| Banco Santander Empresas | 824 | 424 |
| Caixabank Empresas | 758 | 324 |
| BBVA Net Cash Empresas | 413 | 264 |
| Other (customer-defined) | 347 | 141 |
| Banco Sabadell Empresas | 346 | 195 |
| Caixabank | 293 | 145 |
| Bankinter Empresas | 263 | 187 |
| Banco Santander | 250 | 151 |
| Banca March | 186 | 88 |
| Banco Sabadell T. sec - CAL Empresas | 170 | 96 |
| Bankinter | 123 | 57 |
| BBVA | 121 | 74 |
| Abanca Empresas | 113 | 56 |
| Paypal | 88 | 29 |
| Banco Sabadell | 87 | 54 |

## 8. `company_meta.n_banking` vs last-month `g_n_accounts`

Train companies: 1,214. Exact match: 1,188 (97.9%). Delta explained by post-snapshot products: 1,214 / 1,214. Delta p50=0 mean=0.03 max=4.

`n_banking` is extract stock (includes the 44 post-snapshot rows). Last-month G drops those rows. A mismatch of +1…k is the connection-after-extract pile, not a products.py bug.

Largest mismatches (train):

| company | n_banking | last g_n_accounts | delta | n post-snapshot |
| --- | --- | --- | --- | ---: |
| COMP_0302 | 6 | 2 | 4 | 4 |
| COMP_0265 | 8 | 5 | 3 | 3 |
| COMP_0406 | 14 | 11 | 3 | 3 |
| COMP_1097 | 3 | 0 | 3 | 3 |
| COMP_0236 | 9 | 6 | 3 | 3 |
| COMP_0849 | 4 | 2 | 2 | 2 |
| COMP_0978 | 12 | 10 | 2 | 2 |
| COMP_0855 | 12 | 10 | 2 | 2 |

## 9. Custom / Other banks vs the 470

All-custom extract books: 42 train companies, 18 of them never-ERP (42.9% of all-custom). Never-ERP with a banking row: 469 — custom is **not** the 470.

| group | n companies | any custom | all custom | mean custom share | n all-custom | n with a TPV row |
| --- | ---: | --- | --- | --- | --- | --- |
| ever_erp | 742 | 0.1199 | 0.03235 | 0.05943 | 24 | 9 |
| never_erp | 469 | 0.07036 | 0.03838 | 0.05017 | 18 | 1 |

## 10. Card × TPV after size terciles (no cluster revival)

Last-month train: card 197, TPV 10. A 2×2 cannot beat silhouette 0.234 when TPV is 10 companies. Card Y3 gap same sign in every tercile with ≥5 card positives: **no**.

Last-month 2×2 counts (and size mix):

| cell | n companies | T1 | T2 | T3 |
| --- | ---: | --- | --- | --- |
| card+tpv | 3 | 2 | 1 | 0 |
| card_only | 194 | 30 | 76 | 88 |
| tpv_only | 7 | 0 | 1 | 6 |
| neither | 1010 | 373 | 326 | 311 |

Last-month access share by size tercile (the PNG):

| tercile | has_card | has_tpv |
| --- | --- | --- |
| T1_small | 7.9% | 0.5% |
| T2_mid | 19.1% | 0.5% |
| T3_large | 21.7% | 1.5% |

Company-month rates in the 2×2 (train):

| Y | cell | n labeled | n pos | base rate | n companies |
| --- | --- | ---: | ---: | --- | ---: |
| y2_neg_2of3 | card+tpv | 4 | 0 | 0 | 3 |
| y2_neg_2of3 | card_only | 2192 | 154 | 0.07026 | 196 |
| y2_neg_2of3 | tpv_only | 63 | 1 | 0.01587 | 7 |
| y2_neg_2of3 | neither | 15097 | 1116 | 0.07392 | 1168 |
| y3_recover_cash_6m | card+tpv | 0 | 0 |  | 3 |
| y3_recover_cash_6m | card_only | 981 | 33 | 0.03364 | 196 |
| y3_recover_cash_6m | tpv_only | 25 | 0 | 0 | 7 |
| y3_recover_cash_6m | neither | 4642 | 369 | 0.07949 | 1168 |

Card vs no-card inside size terciles — the only cell large enough to test 'operating type after size':

| Y | tercile | group | n labeled | n pos | base rate | n companies |
| --- | --- | --- | ---: | ---: | --- | ---: |
| y2_neg_2of3 | all | card | 2196 | 154 | 0.07013 | 197 |
| y2_neg_2of3 | all | no_card | 15160 | 1117 | 0.07368 | 1168 |
| y2_neg_2of3 | T1_small | card | 412 | 8 | 0.01942 | 32 |
| y2_neg_2of3 | T1_small | no_card | 5679 | 312 | 0.05494 | 392 |
| y2_neg_2of3 | T2_mid | card | 872 | 76 | 0.08716 | 77 |
| y2_neg_2of3 | T2_mid | no_card | 4919 | 425 | 0.0864 | 387 |
| y2_neg_2of3 | T3_large | card | 912 | 70 | 0.07675 | 88 |
| y2_neg_2of3 | T3_large | no_card | 4562 | 380 | 0.0833 | 389 |
| y3_recover_cash_6m | all | card | 981 | 33 | 0.03364 | 197 |
| y3_recover_cash_6m | all | no_card | 4667 | 369 | 0.07907 | 1168 |
| y3_recover_cash_6m | T1_small | card | 146 | 18 | 0.1233 | 32 |
| y3_recover_cash_6m | T1_small | no_card | 1426 | 251 | 0.176 | 392 |
| y3_recover_cash_6m | T2_mid | card | 391 | 13 | 0.03325 | 77 |
| y3_recover_cash_6m | T2_mid | no_card | 1653 | 60 | 0.0363 | 387 |
| y3_recover_cash_6m | T3_large | card | 444 | 2 | 0.004505 | 88 |
| y3_recover_cash_6m | T3_large | no_card | 1588 | 58 | 0.03652 | 389 |

Y3 card−no_card gap by tercile:

| tercile | rate gap | card labeled | card pos |
| --- | --- | --- | --- |
| T1_small | -0.053 | 146 | 18 |
| T2_mid | -0.003 | 391 | 13 |
| T3_large | -0.032 | 444 | 2 |

The card split does **not** survive size terciles as a stable type. CLOSE as Q1 operating-type. Do not revive clusters (silhouette 0.234).

## 11. One bank vs many, size-controlled

Raw Y2 is higher on many-bank months (8.3% vs 5.3%). If that is size (more banks ↔ larger), it dies inside terciles.

| Y | tercile | group | n CM | n labeled | n pos | base rate | n companies |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| y2_neg_2of3 | all | one_bank | 7945 | 6393 | 341 | 0.05334 | 666 |
| y2_neg_2of3 | all | many_banks | 10594 | 8443 | 697 | 0.08255 | 724 |
| y2_neg_2of3 | all | zero_banks | 2618 | 2520 | 233 | 0.09246 | 773 |
| y2_neg_2of3 | T1_small | one_bank | 4029 | 3239 | 138 | 0.04261 | 292 |
| y2_neg_2of3 | T1_small | many_banks | 2522 | 2039 | 117 | 0.05738 | 161 |
| y2_neg_2of3 | T1_small | zero_banks | 868 | 813 | 65 | 0.07995 | 242 |
| y2_neg_2of3 | T2_mid | one_bank | 2370 | 1908 | 98 | 0.05136 | 216 |
| y2_neg_2of3 | T2_mid | many_banks | 3890 | 3127 | 339 | 0.1084 | 255 |
| y2_neg_2of3 | T2_mid | zero_banks | 778 | 756 | 64 | 0.08466 | 248 |
| y2_neg_2of3 | T3_large | one_bank | 1546 | 1246 | 105 | 0.08427 | 158 |
| y2_neg_2of3 | T3_large | many_banks | 4182 | 3277 | 241 | 0.07354 | 308 |
| y2_neg_2of3 | T3_large | zero_banks | 972 | 951 | 104 | 0.1094 | 283 |
| y3_recover_cash_6m | all | one_bank | 7945 | 1715 | 192 | 0.112 | 666 |
| y3_recover_cash_6m | all | many_banks | 10594 | 3467 | 172 | 0.04961 | 724 |
| y3_recover_cash_6m | all | zero_banks | 2618 | 466 | 38 | 0.08155 | 773 |
| y3_recover_cash_6m | T1_small | one_bank | 4029 | 794 | 149 | 0.1877 | 292 |
| y3_recover_cash_6m | T1_small | many_banks | 2522 | 652 | 98 | 0.1503 | 161 |
| y3_recover_cash_6m | T1_small | zero_banks | 868 | 126 | 22 | 0.1746 | 242 |
| y3_recover_cash_6m | T2_mid | one_bank | 2370 | 579 | 29 | 0.05009 | 216 |
| y3_recover_cash_6m | T2_mid | many_banks | 3890 | 1325 | 41 | 0.03094 | 255 |
| y3_recover_cash_6m | T2_mid | zero_banks | 778 | 140 | 3 | 0.02143 | 248 |
| y3_recover_cash_6m | T3_large | one_bank | 1546 | 342 | 14 | 0.04094 | 158 |
| y3_recover_cash_6m | T3_large | many_banks | 4182 | 1490 | 33 | 0.02215 | 308 |
| y3_recover_cash_6m | T3_large | zero_banks | 972 | 200 | 13 | 0.065 | 283 |

## 12. First-tx vs first-created (73.6% replica)

Train companies on the grid: 1,214. With a banking row: 1,211 (no banking 3). First `created_at` after 2024-09-01: 891 / 1,211 = 73.6%. CONFIRM 73.6%.
`created_at` after first tx: 873. Months from first tx to first `g_n_accounts>0`: p50=2.0 mean=2.1; share already connected on first tx month=36.5%; never as-of account on panel=5.

This is the connection clock. `g_created_*` constants are **not** this fact.

## 13. Leftover types (no `g_has_*`)

Train companies with wallet / risk / expensesPlatform / lineofcomex: 37 (ever-ERP 19). They still increment `g_n_accounts`. Family G does not flag them.

| type | n companies | ever-ERP | never-ERP | last accounts p50 | Y2 rate | Y3 rate | n Y3 labeled |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| expensesPlatform | 7 | 0 | 7 | 7.000 | 0.0% | 23.1% | 26 |
| lineofcomex | 7 | 1 | 6 | 7.000 | 0.0% | 0.0% | 35 |
| risk | 5 | 2 | 3 | 11.000 | 0.0% | 0.0% | 17 |
| wallet | 20 | 16 | 4 | 5.000 | 10.8% | 3.6% | 111 |

Small-n leftover types. Do not invent `g_has_wallet`. Not a health Y. expensesPlatform Y3 23% sits on 26 labeled months / 7 companies — do not promote.

## 14. `g_new` same-month Y — first birth vs add-on

If new-this-month were Q3 turning, same-month Y2/Y3 would jump. They should not.

| Y | kind | n CM | n labeled | n pos | base rate | n companies |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| y2_neg_2of3 | add_on | 793 | 675 | 39 | 0.05778 | 503 |
| y2_neg_2of3 | first_birth | 887 | 834 | 60 | 0.07194 | 887 |
| y2_neg_2of3 | none | 19477 | 15847 | 1172 | 0.07396 | 1214 |
| y3_recover_cash_6m | add_on | 793 | 242 | 13 | 0.05372 | 503 |
| y3_recover_cash_6m | first_birth | 887 | 213 | 20 | 0.0939 | 887 |
| y3_recover_cash_6m | none | 19477 | 5193 | 369 | 0.07106 | 1214 |

## 15. Checking = connected; card AUROC on the connected book

`g_has_checking=0` CM: 2,641. Of those, 2,618 (99.1%) also have `g_n_accounts=0`. Checking-off but accounts>0: 23 CM / 8 companies (card 47.8%, TPV 4.3%, saving 0.0%, invest 8.7%). `g_has_checking` is almost the connection clock, not a mix flag — raw checking covers 1,282 / 1,283 companies.

Connected months only (`g_n_accounts>0`, n=18,539). Y3 oriented CV: card 0.556 vs size 0.622 (Δ -0.066) vs days 0.716. Still CLOSE as Y3 X.

| Y | feature | pooled raw | CV raw | CV orient | sd | n labeled |
| --- | --- | --- | --- | --- | --- | ---: |
| y2_neg_2of3 | g_has_card | 0.500 | 0.528 | 0.528 | 0.083 | 14836 |
| y2_neg_2of3 | g_has_tpv | 0.498 | 0.498 | 0.502 | 0.004 | 14836 |
| y2_neg_2of3 | g_n_banks | 0.556 | 0.532 | 0.532 | 0.078 | 14836 |
| y2_neg_2of3 | c_n_days_with_tx | 0.594 | 0.593 | 0.593 | 0.031 | 14836 |
| y2_neg_2of3 | log1p_a_in3 | 0.546 | 0.562 | 0.562 | 0.048 | 14836 |
| y3_recover_cash_6m | g_has_card | 0.447 | 0.444 | 0.556 | 0.039 | 5182 |
| y3_recover_cash_6m | g_has_tpv | 0.497 | 0.498 | 0.502 | 0.005 | 5182 |
| y3_recover_cash_6m | g_n_banks | 0.370 | 0.367 | 0.633 | 0.062 | 5182 |
| y3_recover_cash_6m | c_n_days_with_tx | 0.273 | 0.284 | 0.716 | 0.031 | 5182 |
| y3_recover_cash_6m | log1p_a_in3 | 0.380 | 0.378 | 0.622 | 0.060 | 5182 |

Residual size of one-bank vs many-bank companies (last-month log1p(a_in3) inside the same tercile). T1 still has a size gap (p50 6.4 vs 8.1). T2 is matched (12.42 vs 12.54) — the T2 Y2 5.1% vs 10.8% is not leftover size, but T3 **flips**, so one-vs-many stays CLOSE as Y2 X. Connected-only `g_n_banks` Y3 oriented 0.633 vs size 0.622 (Δ 0.011 < 0.02).

| tercile | group | n companies | log1p(a_in3) p50 | mean | accounts p50 |
| --- | --- | ---: | --- | --- | --- |
| T1_small | one_bank | 242 | 6.407 | 5.116 | 1.000 |
| T1_small | many_banks | 161 | 8.120 | 5.848 | 3.000 |
| T2_mid | one_bank | 147 | 12.422 | 12.420 | 1.000 |
| T2_mid | many_banks | 255 | 12.537 | 12.538 | 4.000 |
| T3_large | one_bank | 96 | 14.913 | 15.332 | 2.000 |
| T3_large | many_banks | 308 | 14.809 | 15.162 | 6.000 |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `g_new_this_month` as health Y / Q3 | **PARK** | connection birth; acf≈0; often first as-of account |
| `g_created_*` constants as X | **DROP** (CONFIRM) | all-zero / modal 100%; not the 73.6% fact |
| `g_has_*` as Y3 X | **CLOSE** | oriented CV loses to size 0.617 and to days 0.711 |
| `g_has_*` as Q1 descriptive | **CLOSE** as operating type | 2×2 / tercile test below; TPV n too small; no cluster revival |
| `g_n_accounts` inventory | **KEEP** (clock, not Y) | rise-only as-of panel; complementary to debt facilities |
| `g_n_accounts` / `g_new` in 44-col starter | **out** (CONFIRM) | KEEP-list inventory, not in the 44 GBM columns |
| `g_has_checking` in 44-col starter | **CAUTION** | present (CONFIRM) but it is the connection hole, not mix |
| invent a dark-access Y | **PARK** | access ≠ ERP; the 470 still have checking |
| `g_has_checking` as mix / type | **CLOSE** | 99% of checking=0 is `g_n_accounts=0` (connection hole) |
| one-vs-many banks as Y2/Y3 X | **CLOSE** | Y2 sign flips in T3; residual size remains inside tercile (pass 15) |
| card AUROC on connected months | **CLOSE** as Y3 X | still loses to size / days |

## Plot

- `analysis/outputs/banking_g_has_vs_size.png` — last-month `g_has_card` / `g_has_tpv` vs company size tercile (train).

## Closed in this module

- Rise-only inventory: **yes** (0 drops), same as-of `created_at` rule as debt.
- `g_new` = connection: **yes** (52.8% first birth; calendar spread; PARK as Y).
- Feature-report drop `g_created_*` + keep `g_has_*` in 44-col: **CONFIRM**. `g_n_accounts` / `g_new` stay off the 44.
- Oriented AUROC: first as-is KEEP table was a false win vs inverse-size 0.38. Corrected.
- 73.6% replica + n_banking vs last G: measured.
- Custom/Other is not the 470. Dark do not have fewer accounts.
- Card × TPV after size: does not survive. Leftover types: measured. `g_new` same-month Y: measured.
- `g_has_checking` ≈ connection hole. Card AUROC on connected months: still CLOSE vs size/days.

