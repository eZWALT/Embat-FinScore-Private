# Wave 4 — zero-in QA (87c77f5b)

Long-lived data lane. Same module ≥30 min: write → run → next cut (must-do 1–12 plus extras 13–28).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
`ops.py` not edited (store vs raw amount>0 agree 100%). Holdout 72 coverage only; rates / AUROC on train. Seed 20260918.
Did not touch recency / gap_sd / transfer QA, `y6_activity.py`, the 15-col card, or the night Y3 quote **0.762 / 0.752**. Days bar stays **0.711**.

## Files

- `analysis/evaluate/zero_in_qa.py` (create)
- `analysis/outputs/zero_in_qa.md`
- `analysis/outputs/zero_in_piles.png`
- append-only `analysis/experiments/registry.csv` (skip key includes `x_families`)
- this note

## What we measured (train)

- Prevalence: `c_zero_in_month` **11.8%** (2,486 / 21,157). Modal 0 share **88.2% CONFIRM**. `c_zero_in_share_6` mean 0.116, modal-0 share 77.4%. Holdout coverage only: month 8.0% / share 0.081 on 1,073 CM / 72 cos. No holdout AUROC.
- Flag pile: **all-out** (n_tx>0, no amount>0) **1,597** = **64.2%** of the flag. Empty grid (n_tx=0) **889** = **35.8%**. Empty ∩ has-in = 0. `a_n_tx` ≡ `c_n_tx`. days==0 ≡ n_tx==0. All 1,597 all-out months are only-neg amounts (not zero-amt tokens).
- Formula: store vs raw (any amount>0) **100%**. share_6 vs rolling-6 max|Δ|=0. Not an ops.py bug. 10 CM have zero-in ∩ `a_op_in`≠0 — all **negative** CAT_MAP op_in (refunds / dust, p50 −803). Do not patch.
- Spearman: month vs log1p(a_in3) **−0.429** (not SIZE on the KEEP clock). Month vs log1p(|a_op_in|) **−0.508 CONFIRM** (feature-report SIZE clock). share_6 vs a_in3 **−0.500 SIZE**; vs |a_op_in| **−0.519 CONFIRM**. vs days −0.509 / n_tx −0.514 / recency 0.443 — **not** a |ρ|≥0.80 twin of days, n_tx, or recency.
- Y3 group-fold: month **0.580**, share_6 **0.617**, empty 0.530, all-out 0.550 vs size **0.617** (quote 0.617) vs days **0.711** (night replica OK) vs n_tx 0.703 vs `a_op_in` 0.676. Month does **not** beat size by ≥0.02 (Δ −0.037). Y2 month 0.522 / share 0.530.
- Leftover after days: month **0.553** / n_tx 0.544 / card 0.556 (Δsize −0.063). share_6 leftover **0.537**. Busy-only (n_tx>0) all-out **0.554** vs size 0.612 vs days 0.699. All-out OLS leftover 0.614 is a **fake** days leak (ρ(resid,days)=0.664). Leftover **dies**.
- SIZE terciles: T1 month 0.573 “beats” an inverted T1 size 0.427; leftover-days 0.538 **loses to T1 days 0.603**. T3 0.534. Inverse-size / Y6 mode, not leftover Q3.
- vs `y6_zero_in_3`: Jaccard **0.286**, ρ 0.391. Different window (now vs t+1..t+3). Y6 vs size **0.854** / vs log1p(|a_op_in|) **0.878** — CONFIRM inverse-size failure. Do not revive.
- Dark 470 vs 744: zero-in CM **11.0% vs 10.9%** — same bank-book rate. CONFIRM 744/470.
- Chronic 12 (0158/0172): Y2 0.522 → 0.520 (no flip). Those 12 are **busy** (zero-in 1.6% vs rest 11.9%).
- ICC: share_6 **0.973** acf1 **0.866 CONFIRM 0.87** TRAIT. Month ICC 0.933 acf1 −0.027. Company-demean of share_6 is a *transform* (Y3 0.648 / after days 0.686, ρ vs days 0.332) — not the stored column, not on the 44.
- Q6: now 0.580 / lag1 0.552 / lag3 0.551. Short-book 6.1% (181/2,970). **CLOSE** — leftover died, nothing to lead.
- All-out mix: uncategorized 35.3% / payment 21.3% / utility 14.6% / fee 7.5%. Not an uncat twin (ρ −0.122). Y3 rate empty 26.8% ≈ all-out 27.4% vs has-in 5.9% — both piles are the quiet-recover story already on the card.

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `c_zero_in_month` on the 44 | **DROP from the 44** / **CLOSE as X** (SIZE on the report clock; leftover after days dies) |
| `c_zero_in_share_6` on the 44 | **DROP from the 44** / **CLOSE as X** (SIZE on both clocks; leftover dies; TRAIT) |
| either as a health Y | **PARK** — do not invent `y_zero_in`; do not revive Y6 |
| either on tonight's 15-col card | **no** |
| empty vs all-out | flag is **all-out** (64.2%), not the empty grid |
| Q6 lag1 / lag3 | **CLOSE** |
| Y6 `y6_zero_in_3` | **PARK / do not revive** |

## Brief map

1. Who is healthy? — not this flag (PARK as Y).
3. Who is turning? — **CLOSE**. Inverse activity / SIZE, not leftover quiet after days.
5. Why? — same quiet-recover story as days / n_tx (already on the card).
6. Months earlier? — **CLOSE**.

## What we did not do

- Did not edit `ops.py`, `y6_activity.py`, recency / gap_sd / transfer QA, `product/`, parquet / duckdb.
- Did not run `build_targets`. Did not write the parent journal.
- Did not invent a zero-in Y or put zero-in on the 15-col card.
- Did not change the night Y3 quote 0.762 / 0.752 or days 0.711.

## What failed / next

- First SIZE screen used log1p(a_in3) and missed the feature-report clock log1p(|a_op_in|). Tightened; −0.508 / −0.519 CONFIRM.
- First Q6 KEEP was too loose (now 0.580 ≥ 0.55). Closed once leftover died.
- T1 leftover looked like KEEP vs an inverted size clock (0.427). Honest compare is days 0.603 — dies.
- Demean leftover 0.686 is a transform, not a reason to keep the stored columns.
- Next (not this owner): if someone wants a going-quiet X, it is already `c_n_days_with_tx` 0.711.
