# Clean flags QA — DQ, not health

Generated `2026-09-19T02:56:05+02:00` by agent `d7235c84`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Family modules not edited.

Flags are the README keep-and-flag set: `is_dup`, `is_extreme` (|amount|≥1e9), `product_known`, `payment_date_invalid`, `created_after_snapshot`, `outstanding_gt_granted`, `balance_sentinel`.

## Verdicts

| probe | verdict | why |
| --- | --- | --- |
| flags as health Y | **PARK** | They are DQ. Row bases are tiny (tx extreme 24, inv extreme 10) or not a trajectory (snapshot sentinels / post-extract created_at). |
| flags as X | **CLOSE** | Do not put DQ bits on the score path. Size/overlap below; none is a six-question signal. |
| never-drop extremes | **KEEP** | KEEP never-drop. Quoted singles days/issued/HHI move ≥0.02 if extremes dropped: no. |
| product_known vs G 63.7% hole | **different hole** | First-month `g_n_accounts=0` train=63.7%. Unknown-product share of first-month txs on those companies=0.1% (g>0=0.0%). Unknown products in banking/debt books=0/29. |
| Family B sentinel risk | **snapshot OK; flow walk sees extremes** | Sentinels (3 checking) are NULLed — garbage does not seed the snapshot. All 24 extreme txs are checking. COMP_0306: 11/18 Y2 flips if the walk dropped the −1.61e9 Aug-2026 outflow. |
| debt exclude-extreme vs cashflow keep | **report only** | Train `a_op_in` rel move if debt rule used=2.7763%. `f_ds_r` |Δ| mean=0.0000 max=0.0000; CM changed=0. Do not edit family modules. |

### Numbers to quote (train unless noted)

- **tx is_extreme:** 24 / 2,556,068 rows (train 8 / 2,424,704); amount mass train 2.3938%; companies train 4 / holdout 2.
- **inv is_extreme:** 10 / 896,711 (join QA said 10 / 896,711 — **CONFIRM**); **gross** amount mass train 52.0140% — 3 wash pairs; unpaired leftover is the real mass (pass 12c). All 10 are train (holdout 0 companies). Train companies 4.
- **tx is_dup:** train 98,175 / 2,424,704 (4.0%), amount mass 2.317%. Full-content extract clones, not a second booking with a new description.
- **product_known=false txs:** train 1,040 (0.043%), mass 0.0608% — not the G connection hole.
- **Quoted singles if drop extremes:** days keep 0.7114 → drop 0.7114; issued_lag1 0.6157 → 0.6157; HHI_lag3 0.6047 → 0.6047. Threshold ≥0.02: not hit.
- **B-walk / Y2 footnote:** train COMP_0306 checking −1.613e9 (2026-08-14) same-magnitude as overdue invoice −1.616e9 (2025-12-31, 226d). Drop-from-walk would flip 11/18 Y2 labels (0→11). Sibling COMP_1192 is GROUP_0094. Y2 already PARK. Do not rewrite `liquidity.py`. Holdout extremes are GROUP_0199 (COMP_0900 + COMP_0276) on 2026-02-10 / 02-13.

## Brief questions

1. **Who is healthy?** — a DQ flag is not a health reading.
2. **Who is improving?** — clones / sentinels / unknown products do not say 45→65.
3. **Who is turning?** — `created_after_snapshot` is post-extract (Family G already 0 on the panel).
4. **Dip vs fall?** — not these bits.
5. **Why did it change?** — do not explain a score move with `is_dup`.
6. **Months earlier?** — flags are not lead time.

## Pass 1 — prevalence (n, companies, amount mass)

Clean-table denominators (amount=0 already dropped). `dq_log` counts on raw; tx `is_extreme` 24 vs raw 24; inv 10 vs raw 10; `product_known=false` txs 1,313 vs raw 1,314 (the extra raw row was amount=0).

### Train

| table | flag | n_flag | n_rows | row % | cos | |amt| % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| transactions | is_extreme | 8 | 2,424,704 | 0.000% | 4/1214 | 2.3938% |
| transactions | is_dup | 98,175 | 2,424,704 | 4.049% | 957/1214 | 2.3172% |
| transactions | product_known=false | 1,040 | 2,424,704 | 0.043% | 11/1214 | 0.0608% |
| invoices | is_extreme | 10 | 840,634 | 0.001% | 4/745 | 52.0140% |
| invoices | payment_date_invalid | 36,802 | 840,634 | 4.378% | 597/745 | 23.5626% |
| balances | balance_sentinel | 3 | 7,556 | 0.040% | 2/1201 | 0.0000% |
| balances | product_known=false | 22 | 7,556 | 0.291% | 16/1201 | 0.5692% |
| banking_products | created_after_snapshot | 41 | 5,630 | 0.728% | 26/1211 | — |
| debt_products | created_after_snapshot | 18 | 2,128 | 0.846% | 15/358 | 0.8459% |
| debt_products | outstanding_gt_granted | 47 | 2,128 | 2.209% | 32/358 | 5.3034% |
| debt_schedule_config | outstanding_gt_granted | 5 | 86 | 5.814% | 4/39 | 5.1593% |

### Holdout (coverage only)

| table | flag | n_flag | n_rows | row % | cos | |amt| % |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| transactions | is_extreme | 16 | 131,364 | 0.012% | 2/72 | 33.0894% |
| transactions | is_dup | 7,467 | 131,364 | 5.684% | 53/72 | 3.1600% |
| transactions | product_known=false | 273 | 131,364 | 0.208% | 3/72 | 0.0001% |
| invoices | is_extreme | 0 | 56,077 | 0.000% | 0/40 | 0.0000% |
| invoices | payment_date_invalid | 2,431 | 56,077 | 4.335% | 38/40 | 2.4366% |
| balances | balance_sentinel | 0 | 440 | 0.000% | 0/72 | 0.0000% |
| balances | product_known=false | 7 | 440 | 1.591% | 3/72 | 0.1322% |
| banking_products | created_after_snapshot | 3 | 357 | 0.840% | 2/72 | — |
| debt_products | created_after_snapshot | 1 | 111 | 0.901% | 1/20 | 0.9009% |
| debt_products | outstanding_gt_granted | 3 | 111 | 2.703% | 3/20 | 0.2063% |
| debt_schedule_config | outstanding_gt_granted | 1 | 1 | 100.000% | 1/1 | 100.0000% |

Balances sentinels have `balance` NULLed in `clean`. Raw values:

| company | split | product | raw balance |
| --- | --- | --- | ---: |
| COMP_0420 | train | PRODUCT_00410 | -1.000e+09 |
| COMP_0420 | train | PRODUCT_03809 | -1.000e+09 |
| COMP_1068 | train | PRODUCT_00001 | 1.000e+11 |

Plot: `clean_flags_prevalence.png`.

## Pass 2 — calendar concentration

Train. One-month pile if max-month share is high; spread if many months and max share is low.

| table | flag | n months | max month (rows) | max row % | max month (mass) | max mass % |
| --- | --- | ---: | --- | ---: | --- | ---: |
| transactions | is_extreme | 7 | 2025-02-01 | 25.0% | 2025-02-01 | 30.5% |
| transactions | is_dup | 25 | 2026-03-01 | 16.1% | 2026-02-01 | 15.4% |
| transactions | product_known=false | 23 | 2026-01-01 | 15.1% | 2025-12-01 | 18.8% |
| invoices | is_extreme | 7 | 2025-07-01 | 20.0% | 2026-07-01 | 86.6% |
| invoices | payment_date_invalid | 25 | 2026-08-01 | 25.7% | 2026-07-01 | 96.0% |

## Pass 3 — extreme row cards

Invoices: **10** (train 10 / holdout 0). Join QA 10 / 896,711 — confirm 10 / 896,711.

| company | split | iss | type | status | amount |
| --- | --- | --- | --- | --- | ---: |
| COMP_0629 | train | 2026-07-27 00:00:00 | invoice | paid | -6.244e+10 |
| COMP_0629 | train | 2026-07-29 00:00:00 | note | paid | 6.244e+10 |
| COMP_0629 | train | 2025-07-22 00:00:00 | note | paid | -4.568e+09 |
| COMP_0629 | train | 2025-07-22 00:00:00 | invoice | paid | 4.568e+09 |
| COMP_0629 | train | 2024-12-26 00:00:00 | invoice | paid | 3.715e+09 |
| COMP_0306 | train | 2025-12-31 00:00:00 | invoice | overdue | -1.616e+09 |
| COMP_0163 | train | 2024-10-08 00:00:00 | invoice | paid | 1.355e+09 |
| COMP_0042 | train | 2026-06-29 00:00:00 | invoice | cancel | 1.272e+09 |
| COMP_0042 | train | 2026-06-30 00:00:00 | invoice | cancel | -1.272e+09 |
| COMP_0629 | train | 2025-11-11 00:00:00 | invoice | paid | 1.026e+09 |

Transactions: **24** (train 8 / holdout 16). Category groups: {'other': 21, 'op_in': 2, 'op_out': 1}.

| company | split | date | cat | grp | amount |
| --- | --- | --- | --- | --- | ---: |
| COMP_0900 | holdout | 2026-02-10 00:00:00 | uncategorized | other | 3.100e+09 |
| COMP_0900 | holdout | 2026-02-13 00:00:00 | uncategorized | other | -3.070e+09 |
| COMP_0900 | holdout | 2026-06-12 00:00:00 | uncategorized | other | 2.828e+09 |
| COMP_0900 | holdout | 2026-04-30 00:00:00 | uncategorized | other | 2.209e+09 |
| COMP_0900 | holdout | 2026-04-17 00:00:00 | uncategorized | other | 2.162e+09 |
| COMP_0276 | holdout | 2026-02-13 00:00:00 | uncategorized | other | -2.061e+09 |
| COMP_0900 | holdout | 2026-04-16 00:00:00 | uncategorized | other | 2.000e+09 |
| COMP_0487 | train | 2026-03-31 00:00:00 | uncategorized | other | 2.000e+09 |
| COMP_0276 | holdout | 2026-02-10 00:00:00 | uncategorized | other | 1.800e+09 |
| COMP_0629 | train | 2025-02-18 00:00:00 | cash_withdrawal | op_out | -1.694e+09 |
| COMP_0629 | train | 2025-02-18 00:00:00 | cash_settlement | op_in | 1.694e+09 |
| COMP_0306 | train | 2026-08-14 00:00:00 | uncategorized | other | -1.613e+09 |
| COMP_0900 | holdout | 2026-04-21 00:00:00 | uncategorized | other | -1.376e+09 |
| COMP_0900 | holdout | 2026-08-11 00:00:00 | uncategorized | other | 1.305e+09 |
| COMP_0900 | holdout | 2026-05-04 00:00:00 | uncategorized | other | 1.203e+09 |
| COMP_1192 | train | 2026-06-24 00:00:00 | uncategorized | other | 1.125e+09 |
| COMP_0900 | holdout | 2026-03-26 00:00:00 | uncategorized | other | 1.082e+09 |
| COMP_0900 | holdout | 2026-07-07 00:00:00 | uncategorized | other | 1.078e+09 |
| COMP_0629 | train | 2026-01-20 00:00:00 | uncategorized | other | 1.001e+09 |
| COMP_0629 | train | 2025-10-14 00:00:00 | collection | op_in | 1.001e+09 |
| COMP_0487 | train | 2026-04-08 00:00:00 | uncategorized | other | -1.000e+09 |
| COMP_0900 | holdout | 2026-08-13 00:00:00 | uncategorized | other | 1.000e+09 |
| COMP_0900 | holdout | 2026-04-20 00:00:00 | uncategorized | other | 1.000e+09 |
| COMP_0900 | holdout | 2026-04-16 00:00:00 | uncategorized | other | 1.000e+09 |

## Pass 4 — accepted-Y rates on flagged vs not (train company-months)

Company-month flags from booking / issuance month. Snapshot flags (`ever_sentinel`, `ever_ogtg`, `ever_*_after`) mark every panel month of that company. Size AUROC is `log1p(a_in3)` vs the flag (Mann–Whitney). ≥0.60 = SIZE.

| flag | train CM | CM % | y2 on/off | y3 on/off | y4 on/off | y7 on/off | y9 on/off | size AUROC |
| --- | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| has_tx_extreme | 7 | 0.0% | 0.0% / 7.3% | — / 7.1% | — / 13.9% | 33.3% / 28.8% | 0.0% / 14.1% | 0.7335 |
| has_tx_dup | 7,635 | 36.1% | 9.1% / 6.3% | 4.3% / 9.6% | 12.8% / 15.1% | 27.6% / 29.6% | 16.4% / 12.7% | 0.7505 |
| has_tx_unkprod | 72 | 0.3% | 12.9% / 7.3% | 4.5% / 7.1% | 5.6% / 13.9% | 52.6% / 28.7% | 11.1% / 14.1% | 0.6025 |
| has_inv_extreme | 6 | 0.0% | 0.0% / 7.3% | — / 7.1% | — / 13.9% | 75.0% / 28.8% | 0.0% / 14.1% | 0.9964 |
| has_pdi | 3,626 | 17.1% | 4.4% / 7.8% | 2.8% / 7.9% | 13.2% / 14.0% | 21.8% / 31.8% | 14.1% / 14.1% | 0.5902 |
| ever_sentinel | 15 | 0.1% | 0.0% / 7.3% | — / 7.1% | 0.0% / 13.9% | 60.0% / 28.8% | — / 14.1% | 0.3041 |
| ever_ogtg | 597 | 2.8% | 7.8% / 7.3% | 1.4% / 7.3% | 11.9% / 14.0% | 38.3% / 28.4% | 11.2% / 14.2% | 0.6895 |
| ever_bank_after | 419 | 2.0% | 11.7% / 7.2% | 0.0% / 7.3% | 13.8% / 13.9% | 26.4% / 28.8% | 8.8% / 14.2% | 0.6680 |
| ever_debt_after | 255 | 1.2% | 7.1% / 7.3% | 0.0% / 7.2% | 16.9% / 13.8% | 20.2% / 28.9% | 16.8% / 14.0% | 0.7097 |

## Pass 5 — `is_dup`: extract clones vs same-day same amount

Full-content clone groups (the clean key: company, product, date, value_date, amount, category, description, counterparty, status): **51,231**. Train groups 48,570. Extra copies (the `is_dup` flag) 105,642. Copies per group p50=2.0 max=4997.

Same-day same-amount (looser): **106,927** groups. Of those, **59,381** have zero `is_dup` (same payment booked twice with a different description/status/product — not flagged). Multi-status among soft groups: 2,409.

Train CM `has_tx_dup` base=36.1%. Size AUROC vs `log1p(a_in3)` = 0.7505 (SIZE).

Train clone extras by category:

| category | groups | extra copies | |amt| |
| --- | ---: | ---: | ---: |
| uncategorized | 14,833 | 27,695 | 2.010e+09 |
| collection | 7,593 | 21,598 | 7.838e+08 |
| utility | 5,950 | 11,912 | 1.467e+08 |
| payment | 6,615 | 11,603 | 3.239e+08 |
| fee | 5,814 | 10,410 | 3.652e+06 |
| tax | 1,991 | 5,023 | 1.321e+07 |
| transfer | 2,797 | 3,659 | 9.824e+07 |
| debt_repayment | 457 | 2,258 | 5.852e+06 |
| salary | 638 | 1,091 | 3.626e+07 |
| pos_withdrawal | 331 | 871 | 248,072.36 |
| bulk_payment | 345 | 402 | 1.114e+06 |
| social_security | 190 | 347 | 771,402.60 |

## Pass 6 — `product_known` vs Family G first-month hole

G quote: first-month `g_n_accounts` p50=0, **63.7% still 0**. This module: train first-month g=0 share **63.7%**.

Unknown-product txs are a **different hole**. First-month unk-product row share on g=0 companies=0.1%; on g>0 companies=0.0%. Distinct unknown `product_id`=29; of those in banking∪debt books=0 (must be 0 by construction).

Train panel CM: `g_n_accounts=0` 2,618; any unknown-product tx 72; both 54 (unk among g0 months 2.1%). Trail already said g=0 months still have txs — those txs are almost all `product_known=true` (product exists in the extract, `created_at` is later). Unknown product is a missing-ID hole, not the connection clock.

## Pass 7 — `balance_sentinel` vs Family B walk

`liquidity.py` `_cash_balances` keeps checking/saving/tpv with `balance IS NOT NULL`. Clean nulls sentinel balances, so a cash-type sentinel is **silent-omitted** from the snapshot.

Sentinel rows=3. On cash types=3. B cash products=4682; used (not-null)=4679; companies with no usable cash snapshot=0; of which sentinel-only=0.

| company | split | bank type | debt type | cash? | raw balance |
| --- | --- | --- | --- | --- | ---: |
| COMP_0420 | train | checking |  | yes | -1.000e+09 |
| COMP_0420 | train | checking |  | yes | -1.000e+09 |
| COMP_1068 | train | checking |  | yes | 1.000e+11 |

## Pass 8 — debt exclude-extreme vs cashflow keep-extreme

Cashflow / ops / groupctx: flags are **not** drop filters. Debt `_flows`: `AND NOT coalesce(t.is_extreme, false)` — so `f_ds_r` / `f_fc_r` omit extremes from op_in, debt_service, and fin_cost. Family E also drops `payment_date_invalid` from open/delay (timing unknown). Family D HHI does **not** drop `is_extreme` invoices. Inconsistency stays in the report; modules not edited.

- Train sum `a_op_in` keep-extreme (cashflow) = 9.707e+10.
- Train sum `a_op_in` drop-extreme (debt rule) = 9.438e+10.
- Relative move = 2.7763%. Company-months with any op_in change: 2.
- Train sum debt_service keep=6.163e+08 drop=6.163e+08 rel=0.0000%.
- `f_ds_r` if cashflow rule vs debt rule: mean |Δ|=0.0000, p99=0.0000, max=0.0000. CM changed=0; |Δ|≥0.01 → 0.
- Store `f_ds_r` vs recomputed drop-extreme max|Δ|=0.0000 (sanity).

## Pass 9 — quoted singles if extremes dropped

Night quotes: `c_n_days_with_tx` 0.711 (Y3), `e_ar_issued_lag1` 0.630 (Y7), `d_cust_hhi_lag3` 0.605 (Y4). Recompute the single with extremes removed; train group-fold signed AUROC. Move = |Δ|≥0.02.

| single | quote | keep CV | drop-ext CV | Δ | changed CM | ≥0.02? |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| days | 0.7110 | 0.7114 | 0.7114 | 0.0000 | 0 | no |
| issued_lag1 | 0.6300 | 0.6157 | 0.6157 | 0.0000 | 4 | no |
| hhi_lag3 | 0.6050 | 0.6047 | 0.6047 | 0.0000 | 0 | no |

Store `c_n_days_with_tx` CV=0.7114 (night 0.711). Store issued_lag1 CV=0.6295 (night 0.630). In-module issued_lag1 keep=0.6157 is a 0-fill / cancel-filter gap vs the store, **not** a drop-extreme effect (keep and drop match). Days that vanish if extremes dropped: 1 train CM. Extreme AR invoices with a CP (can touch HHI): 4.

**Never-drop:** KEEP never-drop.

## Pass 10 — company-level ever-flagged vs never

| split | flag | ever | share |
| --- | --- | ---: | ---: |
| train | has_tx_extreme | 4/1214 | 0.3% |
| train | has_tx_dup | 955/1214 | 78.7% |
| train | has_tx_unkprod | 11/1214 | 0.9% |
| train | has_inv_extreme | 3/1214 | 0.2% |
| train | has_pdi | 587/1214 | 48.4% |
| train | ever_sentinel | 2/1214 | 0.2% |
| train | ever_ogtg | 32/1214 | 2.6% |
| train | ever_bank_after | 26/1214 | 2.1% |
| train | ever_debt_after | 15/1214 | 1.2% |
| holdout | has_tx_extreme | 2/72 | 2.8% |
| holdout | has_tx_dup | 53/72 | 73.6% |
| holdout | has_tx_unkprod | 3/72 | 4.2% |
| holdout | has_inv_extreme | 0/72 | 0.0% |
| holdout | has_pdi | 38/72 | 52.8% |
| holdout | ever_sentinel | 0/72 | 0.0% |
| holdout | ever_ogtg | 3/72 | 4.2% |
| holdout | ever_bank_after | 2/72 | 2.8% |
| holdout | ever_debt_after | 1/72 | 1.4% |

## Family-module inconsistency (not patched)

- `cashflow.py` / `ops.py` / `groupctx.py` / `catmix.py`: *Clean flags are not drop filters.*
- `debt.py` `_flows`: excludes `is_extreme`. `created_after_snapshot` is excluded from as-of inventory.
- `invoices.py` / `match.py`: `payment_date_invalid` dropped from timing / paid-match.
- `liquidity.py`: sentinels already NULL, then `balance IS NOT NULL` — cash-type sentinels never enter the walk.
- `counterparties.py` HHI: no `is_extreme` filter on invoices.

No family file was edited. If a later wave unifies the extreme rule, pick **keep** (never-drop) unless pass 9 is revisited and a quote moves.

## PARK / CLOSE / KEEP

| item | call |
| --- | --- |
| any flag as a health Y | **PARK** (DQ; do not invent a Y from a flag) |
| any flag as X | **CLOSE** |
| never-drop doubtful rows | **KEEP** unless pass 9 says a quote moved ≥0.02 |
| debt vs cashflow extreme rule | **report** — do not patch tonight |
| product_known as the G hole | **CLOSE** that story — different hole |
| B snapshot sentinels | **protects** — 3 checking NULLed; other cash remains |
| B flow walk vs extreme txs | **footnote** — all 24 extremes are checking. COMP_0306: 11/18 Y2 flips (0→11) if the walk dropped the −1.61e9 checking flow. Do not rewrite `liquidity.py` here |

## Pass 11 — canceling pairs, PDI pile, B gap, HHI windows

### 11a — extreme invoices are mostly washes

Opposite-sign same-|amount| pairs: **3**. Leftover unpaired extremes: 4. HHI-eligible (book invoice, not cancel, amount>0, has CP): **4**. The 52% invoice amount-mass is a **gross** figure — COMP_0629's ±6.24e10 invoice/note pair nets toward 0. That is why issued_lag1 / HHI AUROC did not move.

| company | n | gross | net | types | statuses |
| --- | ---: | ---: | ---: | --- | --- |
| COMP_0042 | 2 | 2.544e+09 | 0.00 | invoice | cancel |
| COMP_0163 | 1 | 1.355e+09 | 1.355e+09 | invoice | paid |
| COMP_0306 | 1 | 1.616e+09 | -1.616e+09 | invoice | overdue |
| COMP_0629 | 6 | 1.388e+11 | 4.742e+09 | invoice,note | paid |

### 11b — `payment_date_invalid` July pile

July 2026 is 96.0% of PDI |amount| mass (6.275e+10 / 6.537e+10). Of that July mass, extremes are 6.244e+10 (99.5%). Invoice ext∩PDI rows=1, mass=6.244e+10.

Top PDI companies by |amount|:

| company | n | |amt| | n_ext |
| --- | ---: | ---: | ---: |
| COMP_0629 | 55 | 6.324e+10 | 1 |
| COMP_0469 | 93 | 6.611e+08 | 0 |
| COMP_0306 | 41 | 3.082e+08 | 0 |
| COMP_0110 | 9 | 2.000e+08 | 0 |
| COMP_0521 | 47 | 1.604e+08 | 0 |
| COMP_1192 | 31 | 1.281e+08 | 0 |
| COMP_1105 | 69 | 1.075e+08 | 0 |
| COMP_0149 | 18 | 1.015e+08 | 0 |

### 11c — Family B snapshot gap on the two sentinel companies

| company | cash products | used | sentinel | snapshot used | raw sentinel |
| --- | ---: | ---: | ---: | ---: | ---: |
| COMP_0420 | 9 | 7 | 2 | 33,265.10 | -2.000e+09 |
| COMP_1068 | 5 | 4 | 1 | 40,250.00 | 1.000e+11 |

Walk **exists** (other used cash products). The omit is silent: COMP_1068's +1e11 raw checking never enters `_cash_balances`. Do not rewrite `liquidity.py` here — report only.

### 11d — do extreme AR invoices sit in a defined HHI window?

| company | iss | amount | defined HHI CM | Y4 labeled CM |
| --- | --- | ---: | ---: | ---: |
| COMP_0629 | 2025-07-22 | 4.568e+09 | 6 | 0 |
| COMP_0629 | 2024-12-26 | 3.715e+09 | 4 | 0 |
| COMP_0163 | 2024-10-08 | 1.355e+09 | 2 | 0 |
| COMP_0629 | 2025-11-11 | 1.026e+09 | 6 | 0 |

Defined HHI windows **exist** (2–6 CM). Y4-labeled CM on those windows = **0**, and pass 13 shows these companies are never Y4-labeled at all. Dropping extremes cannot move `d_cust_hhi_lag3` 0.605. That matches pass 9 n_changed=0.

### 11e — clone whale (max copies)

| company | split | date | cat | n copies | amount |
| --- | --- | --- | --- | ---: | ---: |
| COMP_0611 | train | 2026-03-31 00:00:00 | collection | 4997 | 20.00 |
| COMP_0611 | train | 2026-03-31 00:00:00 | collection | 4913 | 100.00 |
| COMP_1195 | train | 2025-12-04 00:00:00 | utility | 247 | -397.84 |
| COMP_1195 | train | 2026-06-02 00:00:00 | utility | 196 | -397.84 |
| COMP_1195 | train | 2026-03-16 00:00:00 | payment | 135 | -39.19 |

### 11f — soft same-day-same-amount that are **not** `is_dup`

Groups=59,381 (n=2: 48,951). Multi-status=2,310; multi-product=15,252; multi-description=48,590. These are a different booking (status/product/description differs) — not extract clones. The clean key is right to leave them unflagged as `is_dup`.

| status | n txs |
| --- | ---: |
| booked | 139,225 |
| unknown | 5,533 |
| pending | 1,955 |

### 11g — why `f_ds_r` does not move

Train op_in extremes (cashflow keep, debt drop). Debt-service totals were identical (no extreme in `debt_repayment`). If `f_ds_r` is 0 on those months, the denominator change does not move the ratio.

| company | date | extreme | a_op_in | f_ds_r |
| --- | --- | ---: | ---: | ---: |
| COMP_0629 | 2025-02-18 | 1.694e+09 | 2.581e+09 | -0.0000 |
| COMP_0629 | 2025-10-14 | 1.001e+09 | 1.157e+09 | -0.0000 |

### 11h — PDI as a health Y? Design note only

Train CM `has_pdi` base=17.1% (in 5–30%? yes). Size AUROC=`log1p(a_in3)` 0.5902 (<0.60). Numeric gate PASS. **PARK anyway** — invalid payment dates are DQ, July-mass piled, Family E already drops them from timing. No assembler. No new Y file.

### 11i — cross-flag overlap

tx ext∩dup=0; ext∩unk=0; dup∩unk=7. inv ext∩PDI=1 mass=6.244e+10.

### 12a — HHI *levels* on defined windows (not the Y4 quote)

| company | defined CM | HHI changed | max |Δ| | mean |Δ| |
| --- | ---: | ---: | ---: | ---: |
| COMP_0163 | 19 | 2 | 0.8712 | 0.0910 |
| COMP_0629 | 19 | 14 | 0.2983 | 0.0843 |

HHI **levels** can move on unlabeled months. The quoted single is AUROC vs `y4_ds_r_double`; those labeled rows do not overlap the extreme-AR windows (11d). Quote stays.

### 12b — COMP_0611 clone whale

| company | date | amount | product | cp | n | n_id | n_desc |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| COMP_0611 | 2026-03-31 00:00:00 | 20.00 | PRODUCT_02418 | None | 4997 | 4997 | 1 |
| COMP_0611 | 2026-03-31 00:00:00 | 100.00 | PRODUCT_02418 | None | 4913 | 4913 | 1 |
| COMP_0611 | 2026-03-31 00:00:00 | 100.00 | PRODUCT_02418 | COUNTERPARTY_00644 | 84 | 84 | 1 |

Same product, same CP, same description, distinct `transaction_id` — extract clones, not a second booking.

### 12c — leftover unpaired extremes (the real amount-mass after washes)

Extreme |amount| gross=1.443e+11 (52.0% of all invoices). After 3 wash pairs, leftover unpaired |amount|=7.713e+09 (2.8% of all invoices). Quote the 52% as **gross**; net leftover is the 4 unpaired rows.

### 12d — PDI without the one extreme row

PDI rows excluding `is_extreme`: 39,232 ; |amount|=2.924e+09. The July 96% pile is almost entirely COMP_0629's −6.24e10 paid invoice (also `is_extreme`). Residual PDI is ordinary invalid dates, not a second giant.

### 12e — `is_dup` is row-SIZE

Train CM `has_tx_dup` vs `a_n_tx` AUROC=0.8664 (vs `log1p(a_in3)` 0.7505). More txs → more clones. **SIZE**. CLOSE as X. Base 36.1% is outside 5–30% anyway. PARK as Y.

### 13 — holdout extremes + Y4 ever-label on leftover companies

Holdout extreme txs=16 (COMP_0900 + COMP_0276, uncategorized). Opposite-sign pairs=0; unpaired=16. 33% of holdout tx |amount| is this DQ pile — coverage only; do not fit on it.

| company | CM | Y4 labeled | Y4 pos | Y3 labeled |
| --- | ---: | ---: | ---: | ---: |
| COMP_0629 | 24 | 0 | 0 | 0 |
| COMP_0163 | 24 | 0 | 0 | 5 |
| COMP_0306 | 21 | 0 | 0 | 0 |
| COMP_0042 | 24 | 0 | 0 | 0 |
| COMP_0487 | 13 | 0 | 0 | 0 |
| COMP_1192 | 21 | 0 | 0 | 7 |

Unknown-product IDs span 2024-09-09 → 2026-07-10 (23 months, 29 products) — a persistent missing-ID hole, not a first-month-only G artifact.

### 14 — residual PDI is spread; `has_pdi` as X is CLOSE

PDI excluding the one extreme: 25 months. Max-month rows 2026-08-01 = 25.6%. Max-month mass 2026-08-01 = 30.6%. Without COMP_0629's giant, invalid payment dates are a **spread** DQ, not a one-month event.

| Y | CV AUROC | n | n_pos |
| --- | ---: | ---: | ---: |
| y2_neg_2of3 | 0.5353 | 17,356 | 1271 |
| y3_recover_cash_6m | 0.5475 | 5,648 | 402 |
| y4_ds_r_double | 0.5046 | 2,370 | 329 |
| y7_top1_lost | 0.5457 | 7,464 | 2149 |
| y9_fee_r_ownp80 | 0.4851 | 9,591 | 1350 |

`has_pdi` as X vs accepted Ys: do not KEEP. Even if a CV is above 0.55 it is a DQ bit (Family E already drops invalid dates from timing). **CLOSE as X. PARK as Y.**

### 15 — dropping `is_dup` vs the Y3 15-col (`a_n_tx` / days)

First copy stays (`is_dup` is rn>1), so `c_n_days_with_tx` is unchanged (days vanish=0). `a_n_tx` keep CV=0.7033 drop-dup CV=0.7068 Δ=0.0034 (changed CM=2621). Does not move ≥0.02 — never-drop still **KEEP** for clones as well as extremes.

### 16 — Family E PDI on issued (open/delay already drop it)

`e_ar_issued_lag1` if PDI rows are also dropped from issued: keep=0.6157 drop-PDI=0.6105 Δ=-0.0052 changed CM=885. Does not move ≥0.02 — do not patch invoices.py.

### 17 — unknown-product IDs vs unknown balances

Unknown tx `product_id`=29 ; unknown balance `product_id`=29 ; intersection=1. Train companies with unk txs whose last-month `g_n_accounts`>0: 10/11 — they get a real book later; the unknown IDs never join it.

| category | n | |amt| | cos |
| --- | ---: | ---: | ---: |
| uncategorized | 416 | 8.847e+07 | 10 |
| payment | 215 | 7.209e+07 | 10 |
| utility | 183 | 7.969e+06 | 9 |
| cash_withdrawal | 110 | 1.447e+06 | 5 |
| fee | 92 | 106,836.36 | 6 |
| salary | 87 | 1.245e+07 | 2 |
| transfer | 66 | 4.918e+07 | 8 |
| collection | 64 | 4.466e+07 | 10 |

### 18 — `created_after_snapshot` products still have a pre-extract book?

| book | products | pre-extract txs | cos |
| --- | ---: | ---: | ---: |
| banking | 44 | 8186 | 15 |
| debt | 19 | 4 | 1 |

Last-month train `f_outstanding_gt_granted` nn=1214 mean=0.0264 (Family F already NaNs this before last month). `g_created_after_snapshot` max on last-month train=0.0000 (G said 0 on every train CM — confirm).

### 19 — those post-snapshot products still flow on the train panel

Train `a_op_in` sitting on `created_after_snapshot` banking products: 5.507e+06 / 9.707e+10 = 0.0057% (18 CM). G excludes the product from inventory; cashflow keeps the txs (`product_known` is true). Report only — do not patch G or A.

### 20 — extreme companies: share of *their own* tx |amount|

| company | split | n ext / n tx | own |amt| share |
| --- | --- | ---: | ---: |
| COMP_0900 | holdout | 14/1739 | 37.1% |
| COMP_0629 | train | 4/2173 | 14.3% |
| COMP_0276 | holdout | 2/932 | 32.5% |
| COMP_0487 | train | 2/2056 | 13.6% |
| COMP_0306 | train | 1/2354 | 4.9% |
| COMP_1192 | train | 1/1217 | 4.1% |

### 21 — extreme txs on Family B cash products?

24 / 24 extreme txs sit on checking/saving/tpv. Bank types: {'checking': 24}. `liquidity.py` `_product_flows` has **no** `is_extreme` filter. If cash-type count > 0, those amounts walk the snapshot backward. Holdout COMP_0900 is coverage-only; train cash-type extremes would bias `b_liq`.

### 22 — estimated `b_liq` shift from those checking extremes

Identity: `end_bal_t = snapshot − sum(flows after t)`. Drop extremes → `alt_liq_t = b_liq_t + sum(extreme checking flow after t)`. Table `max |shift|` is `|sum(extreme after t)|`. Report only.

| company | split | ext months | sum ext flow | max |shift| | b_liq p50 |
| --- | --- | ---: | ---: | ---: | ---: |
| COMP_0276 | holdout | 1 | -2.609e+08 | 2.609e+08 | 7.561e+08 |
| COMP_0306 | train | 1 | -1.613e+09 | 1.613e+09 | 1.390e+09 |
| COMP_0487 | train | 2 | 1.000e+09 | 1.000e+09 | 4.916e+08 |
| COMP_0629 | train | 3 | 2.001e+09 | 2.001e+09 | 8.854e+08 |
| COMP_0900 | holdout | 7 | 1.552e+10 | 1.549e+10 | 3.841e+08 |
| COMP_1192 | train | 1 | 1.125e+09 | 1.125e+09 | 2.223e+08 |

### 23 — Y2 on those four train companies

Y2 labeled CM on COMP_0629 / 0306 / 0487 / 1192: 67 (pos=0, rate=0.0%) vs rest of train 7.4%. Y2 already PARK as a model. This is a B-walk footnote, not a new Y.

### 24 — Y2 flips if the B walk dropped those checking extremes

Identity (corrected): drop extremes → `alt_liq_t = b_liq_t + sum(extreme cash-type flow after t)`. Y2 rebuilt from `alt_liq` with the same 3-month horizon. Report only.

| company | split | labeled | Y2 flips | store pos | alt pos | liq sign flips |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| COMP_0276 | holdout | 5 | 0 | 0 | 0 | 0 |
| COMP_0306 | train | 18 | 11 | 0 | 11 | 12 |
| COMP_0487 | train | 10 | 0 | 0 | 0 | 0 |
| COMP_0629 | train | 21 | 0 | 0 | 0 | 0 |
| COMP_0900 | holdout | 4 | 1 | 1 | 0 | 3 |
| COMP_1192 | train | 18 | 0 | 0 | 0 | 0 |

Train: 11 Y2 flips / 67 labeled (store pos=0, alt pos=11; 12 months change `b_liq` sign). Y2 is already PARK. Do not rewrite `liquidity.py` in this lane.

### 25 — COMP_0306 is the train flip (Y2 0→11)

| date | amount | category | status | bank type |
| --- | ---: | --- | --- | --- |
| 2026-08-14 | -1.613e+09 | uncategorized | booked | checking |

12 months change `b_liq` sign (12 newly negative). Store Y3 labeled=0 (pos=0). Months with alt_liq<0 (would enter Y3 stressed via liq): 12. Y3 already forbids family B as X. Footnote for Family B QA — no patch here.

### 26 — COMP_0306 extreme invoice vs the checking outflow

| inv iss | inv pay | inv amt | type | status | tx date | tx amt | |Δamt|/|inv| | days vs iss |
| --- | --- | ---: | --- | --- | --- | ---: | ---: | ---: |
| 2025-12-31 |  | -1.616e+09 | invoice | overdue | 2026-08-14 | -1.613e+09 | 0.0020 | 226 |

Same-magnitude (|Δamt|/|inv|<2%): yes. If yes, one economic event is double-counted across invoice HHI and the B walk; still KEEP never-drop (quoted singles unmoved). Do not redo join QA.

### 27 — same-magnitude extreme inv↔tx on companies that have both

Companies with at least one extreme invoice **and** one extreme tx: ['COMP_0306', 'COMP_0629'] (n=2). Same-magnitude pairs (|Δamt|/|inv|<2%): 1.

| company | split | inv iss | inv amt | status | tx date | tx amt | cat | |Δamt|/|inv| | days |
| --- | --- | --- | ---: | --- | --- | ---: | --- | ---: | ---: |
| COMP_0306 | train | 2025-12-31 | -1.616e+09 | overdue | 2026-08-14 | -1.613e+09 | uncategorized | 0.0020 | 226 |

### 28 — COMP_0629 extremes are two different books

Extreme invoices=6, extreme txs=4. No same-magnitude pair (pass 27). Washes live on invoices; checking `op_in` giants are a separate book. Y2 does not flip here (inflows raise `alt_liq` further above zero).

### 29 — same-day opposite-sign extreme txs (net-zero on the B walk)

Pairs: 1. COMP_0629 2025-02-18 ±1.694e9 is one; it nets out of `sum_extreme_flow` and does not move `alt_liq`. Unpaired leftovers are what pass 22/24 see.

| company | date | n | pos | neg | net | cats |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| COMP_0629 | 2025-02-18 | 2 | 1.694e+09 | -1.694e+09 | 0.00 | cash_settlement,cash_withdrawal |

### 30 — remaining train extreme-checking txs (0487 / 1192)

| company | date | amount | category | grp |
| --- | --- | ---: | --- | --- |
| COMP_0487 | 2026-03-31 | 2.000e+09 | uncategorized | other |
| COMP_1192 | 2026-06-24 | 1.125e+09 | uncategorized | other |
| COMP_0487 | 2026-04-08 | -1.000e+09 | uncategorized | other |

Neither flips Y2 (pass 24). Inflows / leftover `other` do not push `alt_liq` below 0.

### 31 — holdout extreme txs (coverage only)

| company | date | amount | category | grp |
| --- | --- | ---: | --- | --- |
| COMP_0900 | 2026-02-10 | 3.100e+09 | uncategorized | other |
| COMP_0900 | 2026-02-13 | -3.070e+09 | uncategorized | other |
| COMP_0900 | 2026-06-12 | 2.828e+09 | uncategorized | other |
| COMP_0900 | 2026-04-30 | 2.209e+09 | uncategorized | other |
| COMP_0900 | 2026-04-17 | 2.162e+09 | uncategorized | other |
| COMP_0276 | 2026-02-13 | -2.061e+09 | uncategorized | other |
| COMP_0900 | 2026-04-16 | 2.000e+09 | uncategorized | other |
| COMP_0276 | 2026-02-10 | 1.800e+09 | uncategorized | other |
| COMP_0900 | 2026-04-21 | -1.376e+09 | uncategorized | other |
| COMP_0900 | 2026-08-11 | 1.305e+09 | uncategorized | other |
| COMP_0900 | 2026-05-04 | 1.203e+09 | uncategorized | other |
| COMP_0900 | 2026-03-26 | 1.082e+09 | uncategorized | other |
| COMP_0900 | 2026-07-07 | 1.078e+09 | uncategorized | other |
| COMP_0900 | 2026-08-13 | 1.000e+09 | uncategorized | other |
| COMP_0900 | 2026-04-20 | 1.000e+09 | uncategorized | other |
| COMP_0900 | 2026-04-16 | 1.000e+09 | uncategorized | other |

COMP_0900 / COMP_0276 are 33–37% of their own tx |amount|. Do not tune on them.

### 32 — extreme-tx dates shared by more than one company

Dates with extremes on >1 company: 2. Holdout COMP_0900 and COMP_0276 share 2026-02-10 / 2026-02-13 (uncategorized giants). Coverage only — do not redo sibling H.

### 33 — extreme-tx companies and groups

| company | group | split |
| --- | --- | --- |
| COMP_0276 | GROUP_0199 | holdout |
| COMP_0306 | GROUP_0094 | train |
| COMP_0487 | GROUP_0170 | train |
| COMP_0629 | GROUP_0132 | train |
| COMP_0900 | GROUP_0199 | holdout |
| COMP_1192 | GROUP_0094 | train |

Groups that contain more than one extreme-tx company: 2. Coverage fact. Do not redo sibling H.

### 34 — size of those two groups

| group | n companies |
| --- | ---: |
| GROUP_0094 | 13 |
| GROUP_0199 | 11 |

Members listed: 24; of which extreme-tx: 4. GROUP_0094 has 13 train companies including extreme-invoice COMP_0163 plus extreme-tx COMP_0306 / COMP_1192. GROUP_0199 has 11 holdout companies; only 0900 / 0276 carry extreme txs. Coverage — do not redo sibling H.

## Still unknown / next cut in this module

- Family B QA owns whether to footnote `b_liq` / Y2 on COMP_0306 (11/18 flips). This lane does not rewrite `liquidity.py`.
- Hidden-test GROUP_0199 (COMP_0900 + COMP_0276) books uncategorized giants on the same two days (2026-02-10 / 02-13). Coverage only; do not redo sibling H.
- Train GROUP_0094 pairs COMP_0306 (Y2-flip outflow) with sibling COMP_1192 (+1.13e9). Coverage fact.

