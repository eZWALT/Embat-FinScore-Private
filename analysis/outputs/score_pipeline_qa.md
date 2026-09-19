# Javier 14 ↔ feature store (raw signals, no 0–100)

Generated `2026-09-19T01:14` UTC by `python -m analysis.evaluate.score_pipeline_qa`.
Holdout 72 (seed 20260918) is **coverage only**. Spearman / SAME / CLOSE / DRIFT are train.
Does not call `fit_ref` / `run_score` / `make_traj`. Does not write a scorecard or pillar weights.
Does not redo Family B balances. Does not touch `product/`.

## Headline

- Verdicts on the **primary** store twin: **11 SAME / 2 CLOSE / 1 DRIFT** (of 14).
- Worst primary drift: `volatility` vs `b_bal_vol` ρ=0.354 n=14968 (DRIFT).
- CAT_MAP parity: **IDENTICAL** (22 pipeline keys, 22 store keys).
- The 14 do not include days-with-tx / issued_lag1 — those are *beyond* Javier, not a drift.
- CLOSE: `ar_overdue` / `ap_overdue` vs all-open store (3m-window reconstruction is identity). DRIFT: `volatility` vs `b_bal_vol` (cashflow sd, not balance sd). Reconstruct from A is identity — no named `a_vol`.
- Concentration is **top1** (ρ=0.995), not HHI (ρ=0.990; store top1↔HHI 0.994). HHI stays PARK as a gradient X (Y4 monopoly tail).
- Train panel: 1214 companies / 21,157 company-months. Invoice-dark: **470** (38.7%; `COMP_0962` refund ghost). Schedule as-of rate: 38 train companies — not a hole in the 14.
- Group-fold ρ (seed 20260918) does not flip a verdict. Vol fold-0 is 0.125, still DRIFT; overdue stays CLOSE.
- Wall 9s. PNG: `score_pipeline_rho.png`.

## Brief questions (the 14, not a score)

| signal | store twin | Q | hole | PARK / note |
| --- | --- | --- | --- | --- |
| `runway` | `b_runway` | Q1 | b_only | KEEP last-value Q1 description; never B as Y2/Y3 X |
| `d_runway` | `b_d_runway` | Q2 | b_only | 3-month Δ; store is unclipped, Javier clips [-12, 12] |
| `neg_liq` | `b_neg_liq_3` | Q1/Q4 | b_only | rolling 3-month share; b_below_0 is the 1-month still |
| `coverage` | `a_io_ratio` | Q1 | none | in3/out3 clip 3 — cashflow, not B |
| `net_margin` | `a_net_margin` | Q1 | none | clip (in3-out3)/in3 |
| `volatility` | `b_bal_vol` | Q3 | none | Javier is sd6(net)/mean6(op_in); b_bal_vol is sd6(liq)/out — different object |
| `growth` | `a_growth_3` | Q2 | none | trailing-3 vs t-3; a_growth_12 is YoY of the same window |
| `concentration` | `d_cust_top1` | Q5 | invoice | pipeline is top1 (max/sum); HHI is a monopoly tail (Y4 ρ 0.991) — PARK as gradient X |
| `fin_cost_r` | `f_fc_r` | Q3 | flow | KEEP flow; schedule rate / util are 1.7% / last-month — not in the 14 |
| `debt_serv_r` | `f_ds_r` | Q3 | flow | KEEP flow; store drops is_extreme txs |
| `ar_overdue` | `e_ar_overdue` | Q5 | invoice | Javier open stock is 3m issuance; store is all unpaid. DSO is Y7 SHAP#1 and fails short-DSO — not in the 14 |
| `ap_overdue` | `e_ap_overdue` | Q5 | invoice | same 3m-vs-all-open window as AR |
| `delay_coll` | `e_delay_coll` | Q5/Q6 | invoice | null first 6 calendar months (left truncation) |
| `delay_paid` | `e_delay_paid` | Q5/Q6 | invoice | null first 6 calendar months (left truncation) |

## 1. CAT_MAP parity

Pipeline `score_pipeline.CAT_MAP` vs `analysis.features.common.CAT_MAP`.

- keys equal: True
- mapping equal: **True**
- only in pipeline: ∅
- only in store: ∅
- value diffs: ∅
- groups: ['debt_service', 'fin_cost', 'invest', 'op_in', 'op_out', 'transfer']

## 2. Train Spearman (raw, pre-percentile)

ρ ≥ 0.95 SAME · 0.80–0.95 CLOSE · < 0.80 DRIFT. Pairwise finite n. Train only.

| signal | store | ρ | n | verdict | Javier cov | store cov | best twin | best ρ |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |
| `runway` | `b_runway` | 1.000 | 18,551 | **SAME** | 87.7% | 87.7% | `b_runway` | 1.000 |
| `d_runway` | `b_d_runway` | 1.000 | 14,968 | **SAME** | 70.7% | 70.7% | `b_d_runway_clip` | 1.000 |
| `neg_liq` | `b_neg_liq_3` | 1.000 | 18,551 | **SAME** | 87.7% | 87.7% | `b_neg_liq_3` | 1.000 |
| `coverage` | `a_io_ratio` | 1.000 | 18,729 | **SAME** | 88.5% | 88.5% | `a_io_ratio` | 1.000 |
| `net_margin` | `a_net_margin` | 1.000 | 18,729 | **SAME** | 88.5% | 88.5% | `a_net_margin` | 1.000 |
| `volatility` | `b_bal_vol` | 0.354 | 14,968 | **DRIFT** | 71.3% | 70.7% | `vol_from_a` | 1.000 |
| `growth` | `a_growth_3` | 1.000 | 13,532 | **SAME** | 64.0% | 64.0% | `a_growth_3` | 1.000 |
| `concentration` | `d_cust_top1` | 0.995 | 8,896 | **SAME** | 42.0% | 42.2% | `d_cust_top1` | 0.995 |
| `fin_cost_r` | `f_fc_r` | 1.000 | 18,729 | **SAME** | 88.5% | 88.5% | `f_fc_r` | 1.000 |
| `debt_serv_r` | `f_ds_r` | 1.000 | 18,729 | **SAME** | 88.5% | 88.5% | `f_ds_r` | 1.000 |
| `ar_overdue` | `e_ar_overdue` | 0.918 | 8,178 | **CLOSE** | 38.7% | 46.4% | `ar_od_3m` | 1.000 |
| `ap_overdue` | `e_ap_overdue` | 0.883 | 10,615 | **CLOSE** | 50.2% | 57.5% | `ap_od_3m` | 1.000 |
| `delay_coll` | `e_delay_coll` | 1.000 | 5,766 | **SAME** | 27.3% | 31.9% | `e_delay_coll` | 1.000 |
| `delay_paid` | `e_delay_paid` | 1.000 | 8,056 | **SAME** | 38.1% | 40.7% | `e_delay_paid` | 1.000 |

### Alternates

| signal | alt | ρ | n | verdict |
| --- | --- | ---: | ---: | --- |
| `d_runway` | `b_d_runway_clip` | 1.000 | 14,968 | SAME |
| `neg_liq` | `b_below_0` | 0.844 | 18,551 | CLOSE |
| `volatility` | `vol_from_a` | 1.000 | 15,089 | SAME |
| `growth` | `a_growth_12` | 0.476 | 5,206 | DRIFT |
| `concentration` | `d_cust_hhi` | 0.990 | 8,896 | SAME |
| `ar_overdue` | `ar_od_3m` | 1.000 | 8,178 | SAME |
| `ar_overdue` | `e_dso_proxy` | 0.147 | 7,510 | DRIFT |
| `ap_overdue` | `ap_od_3m` | 1.000 | 10,615 | SAME |

## 3. Coverage holes

- Invoice-gated signals: `concentration`, `ar_overdue`, `ap_overdue`, `delay_coll`, `delay_paid`.
- Train companies with **no** book invoices (the 470-dark class): **470** / 1214 (38.7%). Those five signals are undefined there — not a formula drift.
- B-only signals: `runway`, `d_runway`, `neg_liq`. Companies with all-null `b_liq`: 19. Last-value Q1 KEEP as description; never B as Y2/Y3 X.
- Schedule-thin is **not** a hole in the 14. Flow coverage `f_fc_r`=88.5%, `f_ds_r`=88.5%. Schedule / util live outside the 14:

  - `f_w_rate`: 1.7% CM / 38 train companies
  - `f_months_to_next_pay`: 1.7% CM / 38 train companies
  - `f_util_snapshot`: 1.6% CM / 334 train companies
  - `f_sched_vs_obs`: 1.7% CM / 38 train companies

- Delay signals are null on the first 6 calendar months (2024-09..2025-02) by construction (left truncation).

## 4. Concentration: top1 or HHI?

- Javier `concentration` vs `d_cust_top1`: ρ=0.995 n=8,896 **SAME**.
- Javier `concentration` vs `d_cust_hhi`: ρ=0.990 n=8,896 **SAME**.
- Store `d_cust_top1` ↔ `d_cust_hhi`: ρ=0.994 n=8,928 (Y4 quote 0.991).
- Y4 quoted top1↔HHI ρ=0.991; HHI PARK as gradient (monopoly tail)

## 5. Volatility twin

- vs `b_bal_vol`: ρ=0.354 n=14,968 **DRIFT**.
- vs reconstructed `vol_from_a` = min(3, sd6(`a_net`) / max(mean6(`a_op_in`), 1)): ρ=1.000 n=15,089 **SAME**.
- No named a_vol in the store. Mapping volatility→b_bal_vol is a different object.

## 6. Overdue window (3m issuance vs all-open)

- AR: vs store `e_ar_overdue` ρ=0.918 **CLOSE**; vs reconstructed 3m-window ρ=1.000 **SAME**.
- AP: vs store `e_ap_overdue` ρ=0.883 **CLOSE**; vs reconstructed 3m-window ρ=1.000 **SAME**.

## 7. Beyond Javier (night kept — not drift)

The 14 do not include days-with-tx / issued_lag1.

| column | in store? | train cov | why the night kept it |
| --- | --- | ---: | --- |
| `c_ss_month` | yes | 100.0% | Y3 shallow-A stem (Q3/Q5 payroll regularity). Not one of the 14. |
| `c_n_days_with_tx` | yes | 100.0% | Y3 night quote 0.711; Q6 KEEP days_lag1. Not one of the 14. |
| `e_ar_issued_lag1` | yes | 60.6% | Y7 TURNOVER / Q6 KEEP issued_lag1 0.626. Model-time lag of e_ar_issued; not a store column and not one of the 14. |

## 8. Identity tightness (Spearman SAME ≠ value copy)

| pair | Spearman | Pearson | max\|Δ\| | p50\|Δ\| | exact | n |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `runway` vs `b_runway` | 1.000 | 1.000 | 0 | 0 | 100.0% | 18,551 |
| `d_runway` vs `b_d_runway` | 1.000 | 0.960 | 18 | 0 | 89.9% | 14,968 |
| `neg_liq` vs `b_neg_liq_3` | 1.000 | 1.000 | 0 | 0 | 100.0% | 18,551 |
| `coverage` vs `a_io_ratio` | 1.000 | 1.000 | 0 | 0 | 100.0% | 18,729 |
| `net_margin` vs `a_net_margin` | 1.000 | 1.000 | 0 | 0 | 100.0% | 18,729 |
| `volatility` vs `b_bal_vol` | 0.354 | 0.012 | 1.77e+08 | 0.534 | 1.2% | 14,968 |
| `growth` vs `a_growth_3` | 1.000 | 1.000 | 0 | 0 | 100.0% | 13,532 |
| `concentration` vs `d_cust_top1` | 0.995 | 0.995 | 0.555 | 0 | 70.4% | 8,896 |
| `fin_cost_r` vs `f_fc_r` | 1.000 | 1.000 | 0.0099 | 0 | 100.0% | 18,729 |
| `debt_serv_r` vs `f_ds_r` | 1.000 | 1.000 | 0 | 0 | 100.0% | 18,729 |
| `ar_overdue` vs `e_ar_overdue` | 0.918 | 0.856 | 1 | 0.00984 | 42.0% | 8,178 |
| `ap_overdue` vs `e_ap_overdue` | 0.883 | 0.853 | 1 | 0.0207 | 33.0% | 10,615 |
| `delay_coll` vs `e_delay_coll` | 1.000 | 1.000 | 1.56e-13 | 0 | 100.0% | 5,766 |
| `delay_paid` vs `e_delay_paid` | 1.000 | 1.000 | 1.71e-13 | 2.78e-17 | 100.0% | 8,056 |
| `volatility` vs `vol_from_a` | 1.000 | 1.000 | 0 | 0 | 100.0% | 15,089 |
| `d_runway` vs `b_d_runway_clip` | 1.000 | 1.000 | 0 | 0 | 100.0% | 14,968 |
| `ar_overdue` vs `ar_od_3m` | 1.000 | 1.000 | 0 | 0 | 100.0% | 8,178 |
| `ap_overdue` vs `ap_od_3m` | 1.000 | 1.000 | 0 | 0 | 100.0% | 10,615 |
| `concentration` vs `d_cust_hhi` | 0.990 | 0.979 | 0.644 | 0.0923 | 15.6% | 8,896 |

## 9. Why volatility is the only primary DRIFT

Javier `volatility` = min(3, sd6(op_in−op_out) / max(mean6(op_in), 1)). `b_bal_vol` = sd6(liq) / max(mean |out|, 1). Different object.

- `b_bal_vol`: ρ=0.354 n=14,968 DRIFT
- `vol_from_a`: ρ=1.000 n=15,089 SAME
- `b_liq`: ρ=-0.010 n=14,968 DRIFT
- `a_net`: ρ=-0.237 n=15,089 DRIFT
- `a_in3`: ρ=-0.395 n=15,089 DRIFT
- `b_runway`: ρ=0.195 n=14,968 DRIFT
- `j_vs_log1p_a_in3`: ρ=-0.391 n=15,089 DRIFT
- `b_bal_vol_vs_log1p_a_in3`: ρ=-0.280 n=14,968 DRIFT
- Javier vol hits the clip=3 on 11.8% of train CM (defined rows use pairwise n).
- Do not invent an `a_vol` column tonight (cashflow.py is not owned). Reconstruct from A is identity.

## 10. Overdue ρ by month (CLOSE is aging stock, not CAT_MAP)

Store `e_*_overdue` uses **all unpaid** invoices. Javier uses issuance in the last 3 months. The 3m reconstruction is identity (ρ=1). Monthly ρ vs the store twin should fall as the unpaid tail grows.

| month | AR ρ | AR n | AP ρ | AP n |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024-09-01 | — | 0 | — | 0 |
| 2024-10-01 | — | 0 | — | 0 |
| 2024-11-01 | 1.000 | 199 | 1.000 | 263 |
| 2024-12-01 | 0.989 | 224 | 0.983 | 294 |
| 2025-01-01 | 0.947 | 259 | 0.953 | 347 |
| 2025-02-01 | 0.970 | 273 | 0.956 | 370 |
| 2025-03-01 | 0.894 | 267 | 0.899 | 381 |
| 2025-04-01 | 0.899 | 286 | 0.924 | 401 |
| 2025-05-01 | 0.923 | 314 | 0.905 | 420 |
| 2025-06-01 | 0.917 | 315 | 0.864 | 416 |
| 2025-07-01 | 0.896 | 326 | 0.850 | 416 |
| 2025-08-01 | 0.917 | 316 | 0.869 | 415 |
| 2025-09-01 | 0.869 | 329 | 0.819 | 442 |
| 2025-10-01 | 0.913 | 347 | 0.865 | 467 |
| 2025-11-01 | 0.926 | 361 | 0.892 | 483 |
| 2025-12-01 | 0.898 | 377 | 0.837 | 524 |
| 2026-01-01 | 0.925 | 475 | 0.899 | 592 |
| 2026-02-01 | 0.921 | 476 | 0.889 | 619 |
| 2026-03-01 | 0.897 | 480 | 0.855 | 627 |
| 2026-04-01 | 0.913 | 507 | 0.872 | 632 |
| 2026-05-01 | 0.928 | 501 | 0.890 | 640 |
| 2026-06-01 | 0.902 | 528 | 0.876 | 642 |
| 2026-07-01 | 0.908 | 526 | 0.844 | 633 |
| 2026-08-01 | 0.935 | 492 | 0.871 | 591 |

## 11. `is_extreme` vs flow twins

- Extreme txs: 24 / 2,556,068 (0.001%). Family F drops them; Javier `_monthly_flows` does not.
- `fin_cost_r` vs `f_fc_r`: Spearman 1.000, Pearson 1.000, max|Δ|=0.0099, exact=100.0%.
- `debt_serv_r` vs `f_ds_r`: Spearman 1.000, Pearson 1.000, max|Δ|=0, exact=100.0%.
- Rank SAME can hide a few extreme-driven level shifts. Not enough to leave SAME.

| extreme category | n |
| --- | ---: |
| uncategorized | 21 |
| collection | 1 |
| cash_withdrawal | 1 |
| cash_settlement | 1 |

## 12. Clip + invoice-dark fill

- Javier `d_runway` already clipped: rows outside [-12,12] = 0.
- Store `b_d_runway` unclipped rows |x|>12: **1519**. Spearman still 1.000 — clip is a tail, not a rank change.
- Dark (no `e_ar_issued` ever) train companies: **469** / 1214 (7,590 CM).

| column on dark CM | finite share |
| --- | ---: |
| `j_concentration` | 0.0% |
| `j_ar_overdue` | 0.0% |
| `j_ap_overdue` | 0.0% |
| `j_delay_coll` | 0.0% |
| `j_delay_paid` | 0.0% |
| `e_ar_overdue` | 0.0% |
| `e_ap_overdue` | 0.0% |
| `e_delay_coll` | 0.0% |
| `e_delay_paid` | 0.0% |
| `d_cust_top1` | 0.0% |

- `delay_*` finite share before 2025-03: coll 0.0% / paid 0.0% (should be ~0).
- after: coll 32.2% / paid 44.9%.

## 13. DSO is not in the 14

- Javier `ar_overdue` vs `e_dso_proxy`: ρ=0.147 n=7,510 **DRIFT**.
- Store `e_ar_overdue` vs `e_dso_proxy`: ρ=0.229 n=8,029.
- DSO is beyond the 14. Night PARK as Y7 X on short-DSO fold. issued_lag1 is the Q6 KEEP.

## 13b. Dark 470 vs store issued-null

- Live invoice ever (train): dark **470**.
- Store `e_ar_issued` all-null: dark **469**.
- only live-dark: ['COMP_0962']
- only store-dark: ∅
- `COMP_0962` is the join-QA refund-only ghost (one `Abono` / `refund`). Live ever-invoice excludes it (470 dark). Store writes `e_ar_issued=0` / `e_credit_note_ratio=1` on the refund month, so issued-null dark is 469. The five invoice-gated signals stay all-null. Do not count the ghost as ERP.

| company | n invoices | null iss | n invoice-type | n cancel |
| --- | ---: | ---: | ---: | ---: |
| `COMP_0962` | 1 | 0 | 0 | 0 |

## 13c. Concentration residual (top1 is the pipeline object)

- Both finite: 8,896. Differ at 1e-9: 2,634 (29.6%).
- max|Δ|=0.555; p50=0; p90=0.0128.
- Javier-only finite: 0. Store-only finite: 32.
- Median `d_n_cust` on differ vs exact: 31.0 vs 5.0.
- Residual is the due_date / paid-validity filter Javier applies and Family D does not — not HHI vs top1.

## 13d. `fin_cost_r` level shifts

- Differing train CM: 6 / 18,729 (0.032%). max|Δ|=0.0099. Rank stays SAME.

## 13e. Old unpaid stock at last month (CLOSE mechanism)

- AP: 72.1% of open |amount| was issued before the 3m window (694 train companies).
- AR: 70.8% of open |amount| was issued before the 3m window (605 train companies).

## 13f. Concentration residual toward extract

- |Δ| > 0.01 on 11.5% of both-finite rows (1,023). Median Δ is ~0.
- Share of rows that differ at all: 20.8% (first full6 month) → 54.9% (2026-08). Rank stays SAME (ρ=0.995).
- Book filters Javier applies and D does not: null due 352 (AR 217); payment_date_invalid 32,869; paid with null payment_date 32,869.
- Javier drops null-due (and paid-null-pay) from the whole book; Family D does not. Rank stays SAME.

## 13g. The 14 are not one number

- Largest |ρ| among distinct signals: `coverage` ↔ `net_margin` ρ=0.989.
- B cluster (runway / d_runway / neg_liq) vs invoice cluster: median |ρ|=0.038, max=0.099.
- Do not average the 14. Invoice-gated signals are a different object from last-value liquidity.
- `coverage` ↔ `net_margin` ρ=0.989 is the same 3m in/out window (caja twins), not two independent readings.

## 13h. Schedule-thin is outside the 14

- Live train schedule companies: 39. Live rate: 39. Store `f_w_rate`: 38.
- Extra vs store: ['COMP_1027']. `COMP_1027` is created_after_snapshot (2026-09-15 loan). Night 38 is the as-of panel. The 14 already chose flow.

## 13i. `payment_date_invalid` (why D residual grows)

- PDI invoices: 32,869 / 747,349 (4.4%).
- Share of book |amount|: 33.8% (mostly AP). AR |amount| share: 1.5%.
- Javier drops paid-null-pay from the whole invoice book. Family D concentration does not. Family E open stock does drop PDI — delays stay SAME.

## 13j. Holdout coverage (descriptive — no ρ)

- Holdout 72 companies / 1,073 CM. Seed 20260918. Not used for Spearman.

| signal | holdout finite share |
| --- | ---: |
| `runway` | 86.6% |
| `d_runway` | 66.5% |
| `neg_liq` | 86.6% |
| `coverage` | 86.6% |
| `net_margin` | 86.6% |
| `volatility` | 66.5% |
| `growth` | 61.3% |
| `concentration` | 46.8% |
| `fin_cost_r` | 86.6% |
| `debt_serv_r` | 86.6% |
| `ar_overdue` | 40.4% |
| `ap_overdue` | 46.3% |
| `delay_coll` | 35.4% |
| `delay_paid` | 40.5% |

## 13k. SIG types (not applied)

- Javier marks only `neg_liq` as type=share (the score would do 100×(1−value)). The other 13 are percentile-against-reference. **This module applies neither.**
- Pillar labels exist on SIG. They are not computed here.

## 13l. Best-twin footnote (primary map unchanged)

- If `volatility`→`vol_from_a` and overdue→3m-window: **14 SAME / 0 CLOSE / 0 DRIFT**.
- That is a reconstruction, not a store rename. Primary verdicts stay on the named columns.

## 13m. Volatility DRIFT is typical, not a few names

- Per-company Spearman (n≥8 months): 802 train companies. p10=-0.568 p50=0.232 p90=0.785.
- Company verdicts: SAME 0.6% / CLOSE 11.6% / DRIFT 87.8%.
- `b_bal_vol` is not a noisy twin of Javier vol. It is a different series for most books.

## 13n. Overdue CLOSE per company

- AR: p50 ρ=0.900 on 446 companies with ≥8 overlapping months. SAME 34.5% / CLOSE 31.2% / DRIFT 34.3%.
- AP: p50 ρ=0.855 on 589 companies with ≥8 overlapping months. SAME 28.9% / CLOSE 30.2% / DRIFT 40.9%.
- Panel CLOSE is a mixture: many books are SAME, a minority with a long unpaid tail are DRIFT.

## 13o. due < issuance + grid

- Invoices with due < iss: 0 / 746,997 (paid 0). Javier uses GREATEST(due, iss); store delay uses due as-is. Delay twins are still exact — those rows do not move the 3m paid window enough to break identity.
- Grid: Javier 22,230 rows / 1286 companies; store 22,230 / 1286. Inner merge is the store panel.

## 13p. SIZE screen (log1p |a_in3|, train — not a score)

| signal | ρ vs size | n | |ρ|≥0.50 |
| --- | ---: | ---: | --- |
| `runway` | -0.337 | 18,551 | ok |
| `d_runway` | -0.046 | 14,968 | ok |
| `neg_liq` | 0.059 | 18,551 | ok |
| `coverage` | 0.351 | 18,729 | ok |
| `net_margin` | 0.303 | 18,729 | ok |
| `volatility` | -0.391 | 15,089 | ok |
| `growth` | 0.345 | 13,532 | ok |
| `concentration` | -0.165 | 8,507 | ok |
| `fin_cost_r` | 0.039 | 18,729 | ok |
| `debt_serv_r` | 0.274 | 18,729 | ok |
| `ar_overdue` | -0.153 | 7,754 | ok |
| `ap_overdue` | -0.119 | 10,010 | ok |
| `delay_coll` | -0.105 | 5,619 | ok |
| `delay_paid` | -0.054 | 7,810 | ok |

## 13q. First two calendar months (Javier skips invoice features)

- Early CM (2024-09..10): 908. `concentration` finite 0.0%; `ar_overdue` 0.0%.
- Later: conc 43.9%; ar_od 40.4%.
- Delay stays null until month 7 (left truncation), separate from the i<2 skip.

## 13r. Group-fold ρ (seed 20260918 — stability, not a fit)

| signal | fold ρ | min | max | spread |
| --- | --- | ---: | ---: | ---: |
| `runway` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `d_runway` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `neg_liq` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `coverage` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `net_margin` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `volatility` | 0.125, 0.378, 0.436, 0.411, 0.415 | 0.125 | 0.436 | 0.312 |
| `growth` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `concentration` | 0.997, 0.999, 0.997, 0.991, 0.994 | 0.991 | 0.999 | 0.009 |
| `fin_cost_r` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `debt_serv_r` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `ar_overdue` | 0.933, 0.967, 0.909, 0.894, 0.911 | 0.894 | 0.967 | 0.073 |
| `ap_overdue` | 0.916, 0.846, 0.867, 0.902, 0.848 | 0.846 | 0.916 | 0.069 |
| `delay_coll` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
| `delay_paid` | 1.000, 1.000, 1.000, 1.000, 1.000 | 1.000 | 1.000 | 0.000 |
- Verdicts do not flip across folds. Holdout never entered a fold.

## 13s. Dark books still have bank signals

- Live never-invoice train companies: 470 (7,603 CM).
- On those rows: `b_runway` 86.0%, `a_io_ratio` 87.6%, `f_fc_r` 87.6%. Invoice-gated ≠ missing company.
- All-null Javier `ar_overdue` is a wider set (549 companies) — AP-only ERP books sit there too.

## 14. What this is not

- Not a 0–100. Not pillar weights. Not trajectory states.
- Not a percentile fit against a reference (that *is* the frozen score).
- Not an average of the 14. Not a scorecard.
- Family B walk identity already closed — this module does not redo balances.

Elapsed 9s.
