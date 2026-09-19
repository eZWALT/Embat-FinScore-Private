# Wave 4 — issued leftover after days

Agent `e8b2c0d4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/issued_qa.py`
- `analysis/outputs/issued_qa.md`
- `analysis/outputs/issued_leftover.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `dso_qa.*`, `n_tx_qa.*`, `credit_note_qa.*`, `invoices.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617.

## Locked verdict

| object | decision |
| --- | --- |
| issued leftover after days (Y3) | **CLOSE** |
| 15-col Y3 card | **KEEP off the card** |
| TURNOVER add-on (now after lag1) | **CLOSE** |
| issued_lag1 Q6 / TURNOVER | **KEEP (locked)** |
| y_issued | **PARK** |

CLOSE leftover-after-days rank 0.608 (OLS 0.694, fake=True). Y3 issued 0.687 vs days 0.711 vs size 0.617. SIZE=False twin=True. Inverse days-after-issued 0.676. Y7 leftover after lag1 0.581 — CLOSE. Card: CLOSE as unused leftover / KEEP off the 15-col card. Q6 CLOSE. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## What failed / next

- Y3 leftover after days rank 0.608 OLS 0.694 fake=True
- inverse days leftover 0.676
- leftover after size 0.645 after lag1 0.653
- terciles T1 0.530 T2+T3 0.569
- Q6 short leftover days_lag1 0.585 issued_lag1 0.640
- Y7 leftover after lag1 0.581 addon=CLOSE
- fold 4 issued 0.696 leftover 0.612
- log1p leftover 0.608 intensity 0.605
- dark stub n=1
- log1p leftover days+size 0.613
- fold leftover 0.723 0.560 0.553 0.605 0.600
- demean leftover 0.477 mean 0.572
- same-n leftover-days 0.608 fake=True
- T2+T3 leftover 0.569 fake=True
- issued>0 leftover 0.412 fake=True
- log1p vs lag1 ρ=0.814 leftover 0.653
- short leftover days_lag1 0.585 fake=True
- leftover after n_tx 0.603 n_tx+days 0.605
- Y7 leftover after days 0.665 fake=True
- n_tx+days leftover 0.605 fake=False
- early6 cov 68.8% after7 leftover 0.614
- zero-flag leftover 0.582 fake=False
- log1p>0 leftover 0.412 fake=False
- leftover days+lag1 0.587 dies=False
- CN leftover after now 0.614
- leftover after pending 0.694 pending+days 0.613
- fold4 leftover-days Y3 0.400 Y7 0.321
- lag1 leftover after now Y3 0.596 Y7 0.518
- leftover after CN 0.654
- issued/in3 leftover 0.615 fake=True
- trail leftover T1 0.559 T3 0.638
- log1p leftover days+lag1 0.587
- co-median ρ vs log1p(a_in3) 0.511 SIZE=True
- leftover after days_lag3 0.632 after issued_lag3 0.662
- leftover after lag1 folds 0.748 0.622 0.583 0.672 0.640
- Y7 leftover days+lag1 0.590
- days leftover after log1p 0.676
- SIZE×trail leftover T1×T1 — T3×T3 —
- leftover days+size+lag1 0.601
- Q6 leftover days_lag1+issued_lag1 0.584
- leftover after DSO 0.586
- fold 1 issued 0.675 days 0.738
- rank-resid ICC 0.958
- issued vs months_so_far ρ=0.077 leftover 0.698
- n_tx leftover after issued 0.668
- leftover after intensity 0.654 fake=False
- Y7 leftover after CN 0.662
- SIZE T2 leftover 0.452 fake=True
- month leftover median —
- now vs lag1 ρ T1 0.761 T3 0.770
- days leftover after n_tx 0.564
- co-mean leftover 0.586 fake=True
- leftover days+lag1+CN 0.591
- quarter leftover Q1 0.594 Q3 0.653
- lag1 leftover after days Y3 0.591 fake=True
- Y7 leftover after size 0.658
- issued>0 vs lag1 ρ=0.728
- T1 zero leftover 0.557 fake=False
- both>0 leftover-days 0.370 leftover-lag1 0.596
- leftover days+n_tx+lag1 0.587
- size leftover after issued 0.561
- leftover after days_lag1 all 0.621 fake=True
- Y7 both>0 leftover-lag1 0.540
- leftover after pending+CN 0.655
- days leftover both>0 0.679
- group leftover n=0 median —
- Y7 leftover after n_tx 0.665 fake=False
- rank-resid acf1 0.293
- both>0 intensity leftover 0.409 fake=True
- 2025-cohort leftover 0.685 fake=True
- kitchen-sink leftover 0.596
- 2024-cohort leftover 0.596 fake=True
- Y7 leftover after pending 0.661
- leftover after days+DSO 0.446 fake=True
- log1p leftover days+n_tx+lag1 0.587
- ERP-zero leftover 0.697 fake=True
- mid-trail leftover 0.610 fake=True
- Y7 leftover days+size 0.663
- long-trail leftover — fake=True
- Y7 leftover days+lag1+size 0.593
- leftover high-acf1 0.575 low-acf1 0.586
- Y7 leftover pending+CN 0.663
- n_grid>=18 leftover 0.612 fake=True
- Y7 leftover after DSO 0.583
- leftover days+pending+lag1 0.598
- n_grid<12 leftover — fake=True
- leftover days+CN+lag1 0.591
- Y7 leftover days+pending 0.664 fake=False
- leftover days+n_tx+size 0.610
- days leftover after issued+n_tx 0.570
- SIZE T3 leftover — fake=True
- leftover issued>p50 0.543 <=p50 0.601
- Y7 leftover days+CN 0.658 fake=False
- leftover p50-p90 0.532 p90+ —
- days leftover after issued+size 0.648
- leftover <=p25 0.697 >p75 —
- Y7 leftover days+n_tx 0.665 fake=False
- leftover p25-p75 0.575 fake=True
- days leftover after issued+lag1 0.673
- n_tx>0 leftover 0.605 fake=True
- p10-p90 leftover 0.605 fake=True
- days>0 leftover 0.605 fake=True
- lag1>0 leftover 0.559 fake=True
- onset leftover — fake=True
- stop leftover — fake=True
- issued>0 days>0 leftover 0.413 fake=True
- issued>0 T1 leftover — fake=True
- issued>0 T2 leftover — fake=True
- issued>0 mid leftover — fake=True
- issued>0 T2+T3 leftover 0.462 fake=True
- either-pos leftover 0.536 fake=True
- complete-case leftover 0.611 fake=True
- days>p50 leftover 0.642 fake=True
- days<=p50 leftover 0.578 fake=True
- n_tx>p50 leftover 0.654 fake=True
- n_tx<=p50 leftover 0.583 fake=True
- a_in3>p50 leftover 0.634 fake=True
- CN-defined leftover 0.594 fake=True
- pending-defined leftover 0.616 fake=True
- a_in3<=p50 leftover 0.583 fake=True
- days leftover after issued+size+lag1 0.648
- pending+CN leftover 0.594 fake=True
- n_tx-defined leftover 0.608 fake=True
- issued+days leftover 0.608 fake=True
- DSO-defined leftover 0.412 fake=True
- pending days>0 leftover 0.611 fake=True
- log1p complete leftover 0.611 fake=False
- DSO issued>0 leftover 0.412 fake=True
- pending issued>0 leftover 0.412 fake=True
- log1p DSO leftover 0.412 fake=False
- card: CLOSE as unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put issued on the 15-col card
