# Unused leftover of `c_last_tx_before_2026_06` after days as Y3 X

Generated `2026-09-19T06:31:23+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_june` / `y_silent`. Do not revive Y6. Do not put the June flag on the 15-col card. Do not overwrite `recency_qa.*`. Do not grow TURNOVER.

`c_last_tx_before_2026_06` = 1 if period ≥ 2026-06-01 and last booking as of month_end is < 2026-06-01. Javier extract-level **61**; as-of 2026-08-31 **62** (COMP_0981 silent 2025-04-07 → 2026-09-01). `c_recency_days` DROP leftover 0.607.

## Headline

`c_last_tx_before_2026_06` as Y3 X: **PARK as extract** (extract hole: flag only 2026-06..08; last Y3 2026-02-01; mean(flag|Y3)=0.0%. Contemporaneous leftover after days rank 0.711 const=False dies. Single 0.500 (peek 0.500). Javier 61 vs 62 / COMP_0981 +1. DROP from the 44 as Y3 X. Off the 15-col card. Do not revive Y6.). Honest leftover on Y3-labeled rows rank 0.500 const. Panel leftover OLS 0.711 rank 0.711 is a fake days leak (flag=0 on Y3 rows). Inverse days after flag 0.711. Single 0.500 vs size 0.617 vs days 0.711 vs recency 0.659. extract_hole=True only 2026-06..08=True. Javier 61 vs as-of 62; COMP_0981 +1=True. asof62 leftover after days 0.619. leftover after recency 0.659. Q6 lag1 leftover 0.684. 15-col card: no — do not put the June flag on the 15-col card. PARK as extract — do not invent y_june / y_silent. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as extract — do not invent y_june / y_silent. Extract hole, not health. |
| 2 | Who is improving? | Q6 lag1 leftover 0.684 — extract cannot lead. |
| 3 | Who is turning? | **PARK as extract** leftover after days 0.711 vs days 0.711. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | Javier 61 vs 62; COMP_0981 is the +1. Flag is a cutoff. |
| 6 | Months earlier? | lag1 leftover 0.684; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `c_last_tx_before_2026_06` as Y3 X / 15-col card | **PARK as extract** | extract hole: flag only 2026-06..08; last Y3 2026-02-01; mean(flag|Y3)=0.0%. Contemporaneous leftover after days rank 0.711 const=False dies. Single 0.500 (peek 0.500). Javier 61 vs 62 / COMP_0981 +1. DROP from the 44 as Y3 X. Off the 15-col card. Do not revive Y6. |
| `c_last_tx_before_2026_06` as engine X on the 44 | **DROP** | leftover lives=False hole=True twin=False |
| `y_june` / `y_silent` | **PARK** | do not invent; do not revive Y6 |
| `c_recency_days` leftover | 0.567 | peek 0.607 — not overwritten |
| COMP_0981 +1 | **CONFIRM** | as-of 62 vs extract 61 |
| Q6 lag1 | **CLOSE** | leftover 0.684 const=False |

## 1 — Coverage; twin / SIZE; only 2026-06..08

Train 1,214 co / 21,157 CM. Flag =1 on 203 CM (203 in 2026-06..08, 0 outside) months=['2026-06', '2026-07', '2026-08'] ONLY 2026-06..08 CONFIRM. Last Y3 2026-02-01 last Y2 2026-05-01; mean(flag|Y3 labeled)=0.0% extract_hole=True. ρ vs days -0.163 vs a_n_tx -0.163 vs recency 0.186 vs size -0.149. SIZE=False twins=none.

| col | n_nn | share=1 | n=1 |
| --- | --- | --- | --- |
| c_last_tx_before_2026_06 | 21,157 | 1.0% | 203 |
| c_recency_days | 21,157 | — | — |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | -0.163 | no |
| a_n_tx | -0.163 | no |
| c_recency_days | 0.186 | no |
| log1p(a_in3) | -0.149 | no |


## 2 — Single-feature group-fold Y3

Y3 c_last_tx_before_2026_06 0.500 n=5,648 pos=402 const=True (peek 0.500 / 5,648 / 402 CONFIRM). vs size 0.617 vs days 0.711 vs recency 0.659 (quote 0.659 CONFIRM). Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.117 FAIL. asof62 roster leftover-as-X 0.557 extract61 0.555.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_last_tx_before_2026_06 | 5,648 | 402 | 0.500 const | 0.500 | — | — | const |
| y3_recover_cash_6m | c_recency_days | 5,648 | 402 | 0.659 | 0.665 | 0.051 | 1 | 0.590 0.666 0.666 0.730 0.642 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | june_asof62 | 5,648 | 402 | 0.557 | 0.557 | 0.033 | 1 | 0.577 0.536 0.548 0.602 0.521 |
| y3_recover_cash_6m | june_extract61 | 5,648 | 402 | 0.555 | 0.554 | 0.035 | 1 | 0.577 0.526 0.548 0.602 0.521 |


## 3 — Honest leftover after days

June-flag leftover after days OLS 0.711 rank 0.711 const=False fake=True honest_dies=True n=5,648 pos=402. Inverse: days leftover after flag OLS 0.711 rank 0.711 (should stay ~0.711 — flag is 0 on Y3 rows). Recency leftover after days OLS 0.607 rank 0.567 (peek 0.607 was OLS CONFIRM).

OLS folds: 0.665 0.738 0.700 0.715 0.740. Rank folds: 0.665 0.738 0.700 0.715 0.740.

## 4 — Javier 61 vs 62 / COMP_0981

Store vs raw June-flag agree 100.0%. Extract-level last tx < 2026-06-01: **61** CONFIRM Javier 61. As-of 2026-08-31: **62** CONFIRM 62. +1=['COMP_0981'] COMP_0981 is the +1. COMP_0981 last-ever=2026-09-01 00:00:00 last-asof-Aug=2025-04-07 00:00:00 next-after-2025-04-07=2026-09-01 00:00:00. Store Aug-2026 flag n=62 match as-of 62. Train as-of 61 / holdout 1.

## 5 — Leftover after recency

Contemporaneous flag leftover after recency OLS 0.659 rank 0.659 const=False dies=True R²=0.322. after days+recency 0.705. ρ(flag, recency)=0.186 rewrite_twin=False. asof62 leftover after recency 0.560 after days 0.619.

## 6 — Holdout coverage only

Holdout 72 co / 1073 CM flag=1 n=7 (7 in 2026-06..08) n_co=3 share=0.7% (no fit, no AUROC). Extract hole vs health: coverage only.

## 7 — Q6 lag1 leftover after days_lag1

Y3 flag_lag1 0.500 const=True mean on labeled Y3=0.0%. leftover after days_lag1 rank 0.684 dies=True const=False. Days lag1 0.684 (quote 0.684 CONFIRM). Extract cannot lead — lag1 still 0 on Y3-labeled months.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_last_tx_before_2026_06 | 5,648 | 402 | 0.500 const | 0.500 | — | — | const |
| y3_recover_cash_6m | c_last_tx_before_2026_06_lag1 | 5,648 | 402 | 0.500 const | 0.500 | — | — | const |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 8 — Drop COMP_0981

COMP_0981 in train=True. asof62 Y3 0.557 leftover-days 0.619 dies=False. extract61 0.555 leftover 0.622 dies=False. drop-0981 0.555 leftover 0.622 dies=False. Roster leftover is extract, not a 44 stem.

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.500 p50=0.500 p95=0.500 share<0.55=100.0% n_const=40/40.

### Y2 leftover (report-only)

Y2 last labeled 2026-05-01 mean(flag|Y2)=0.0%. Y2 flag 0.500 leftover-days 0.571 const=False. asof62 Y2 0.519 leftover 0.544 dies=True.

### ICC / month share

ICC=0.728 k=1214 (rare dummy; ICC is not a health trait).

| period | n | n=1 | share |
| --- | --- | --- | --- |
| 2024-09 | 435 | 0 | 0.0% |
| 2024-10 | 473 | 0 | 0.0% |
| 2024-11 | 483 | 0 | 0.0% |
| 2024-12 | 525 | 0 | 0.0% |
| 2025-01 | 636 | 0 | 0.0% |
| 2025-02 | 677 | 0 | 0.0% |
| 2025-03 | 714 | 0 | 0.0% |
| 2025-04 | 733 | 0 | 0.0% |
| 2025-05 | 752 | 0 | 0.0% |
| 2025-06 | 762 | 0 | 0.0% |
| 2025-07 | 796 | 0 | 0.0% |
| 2025-08 | 833 | 0 | 0.0% |
| 2025-09 | 864 | 0 | 0.0% |
| 2025-10 | 908 | 0 | 0.0% |
| 2025-11 | 945 | 0 | 0.0% |
| 2025-12 | 1006 | 0 | 0.0% |
| 2026-01 | 1132 | 0 | 0.0% |
| 2026-02 | 1204 | 0 | 0.0% |
| 2026-03 | 1211 | 0 | 0.0% |
| 2026-04 | 1212 | 0 | 0.0% |
| 2026-05 | 1214 | 0 | 0.0% |
| 2026-06 | 1214 | 78 | 6.4% |
| 2026-07 | 1214 | 64 | 5.3% |
| 2026-08 | 1214 | 61 | 5.0% |


### asof62 leftover after size

asof62 leftover after size 0.558 dies=False ρ vs size -0.163. after days+size 0.628 dies=False. after recency 0.560 dies=False.

### Y3 rate on extract roster

Y3 rate asof62 23.2% n=263 vs rest 6.3% n=5385.

| slice | n | n_pos | Y3 |
| --- | --- | --- | --- |
| asof62 | 263 | 61 | 23.2% |
| not asof62 | 5385 | 341 | 6.3% |
| extract61 | 261 | 59 | 22.6% |


### leftover fitted on Y3-labeled rows only

On Y3-labeled rows only: flag leftover after days OLS 0.500 rank 0.500 const=True fake=False (honest extract-hole leftover is chance). Recency leftover OLS 0.540 rank 0.581 (peek 0.607 was OLS). asof62 leftover 0.619 dies=True.

### asof62 leftover folds

asof62 single 0.557 folds=0.577 0.536 0.548 0.602 0.521 beat-size=False. leftover after days OLS 0.619 rank 0.619 fake=False folds=0.569 0.657 0.627 0.532 0.711. after days+recency 0.610 dies=False. KEEP roster as 44 stem=False — no, beat-size FAIL / extract.

### flagged roster counts

Train as-of roster 61 (COMP_0981 in train=1). Y3-labeled asof CM n=263 pos=61. Flag is still extract — roster leftover is not a contemporaneous X.

### asof last-tx vs last Y3=1

asof train roster 61; have any Y3=1: 24; last booking after last Y3=1 month-end: 15. High Y3 rate on the roster is extract-end silence after (or without) recover — not a leading X.

### asof62 leftover bootstrap

asof62 leftover-after-days bootstrap p05=0.558 p50=0.627 p95=0.676 share<0.55=4.2%.

### asof vs recency>60

Y3-lab recency>60 n=45 asof62 n=263 both=25 Jaccard=0.088. Roster is not the recency>60 twin (do not revive Y6).

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| d_supp_top1 leftover | 0.429 DROP |
| c_recency_days leftover | 0.607 DROP |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put the June flag on the 15-col card.

## Files written

- `analysis/evaluate/june_tx_qa.py`
- `analysis/outputs/june_tx_qa.md`
- `analysis/outputs/june_tx_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_june_tx.md` (end, if WRITE_WAVE)

Elapsed 39s. Failed: none.

