# Wave 4 — customer-lost leftover after days

Agent `e8b2c0d4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/cust_lost_qa.py`
- `analysis/outputs/cust_lost_qa.md`
- `cust_lost_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `top1_qa.*`, `cust_hhi_qa.*`, `n_cust_qa.*`, `y4_why.*`, `delay_qa.*`, `ap_overdue_qa.*`, `ap_open_qa.*`, `dso_qa.*`, `gbm_core.py`, `counterparties.py`, `gbm_y7_core.py`, `lit_*.md`, `product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617.

## Locked verdict

| object | decision |
| --- | --- |
| lost leftover after days (Y3) | **DROP** |
| 15-col Y3 card | **KEEP off the card** |
| TURNOVER add-on | **CLOSE** |
| d_n_cust leftover | **DROP (locked)** |
| d_cust_top1 leftover | **DROP (locked)** |
| y_cust_lost | **PARK** |

DROP leftover-after-days rank 0.522 (OLS 0.663, fake=False). Y3 lost 0.581 vs days 0.711 vs size 0.617. SIZE=False twin=True. Inverse days-after-lost 0.730. Leftover after n_cust 0.574 rewrite=False. Leftover after top1 0.547. Boot leftover-days p50=0.511 p05=0.434. Card: DROP from the 44 as Y3 X / CLOSE unused leftover. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## What failed / next

- Y3 lost 0.581 vs days 0.711 vs size 0.617 beat=-0.036
- leftover-after-days rank 0.522 OLS 0.663 fake=False dies=True
- inverse days-after-lost 0.730
- leftover after n_cust 0.574 rewrite=False +days 0.553
- leftover after top1 0.547 +days 0.586 HHI 0.551 new 0.463
- Q6 lag1 leftover 0.464 days_lag1 0.684
- Y7 issued leftover-days 0.665 leak_ok=False
- boot leftover-days p05=0.434 p50=0.511 p95=0.562
- terciles T1 0.575 T2+T3 0.587
- so_far short 0.474 mid 0.542
- per-fold — — — 0.541 0.582
- BETWEEN cmean 0.516 lag1 0.564 within 0.551
- same-n lost 0.581 days 0.737 beat=-0.037 lost>0 0.577
- Y4 leftover-top1 0.420 leftover-days 0.446
- leftover after days+size 0.523 card 0.484 issued 0.476 n_cust+top1 0.595
- company-median ρ vs n_cust 0.896 twin=True
- Q6 mid leftover 0.545 long raw —
- lost/n_cust leftover-days 0.623 raw 0.587 leftover-ntx 0.519 new-after-lost 0.613
- card: DROP from the 44 as Y3 X / CLOSE unused leftover
- do not grow TURNOVER 0.720; do not put d_cust_lost on the 15-col card; PARK y_cust_lost
