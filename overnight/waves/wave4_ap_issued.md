# Wave 4 — AP issued leftover after days

Agent `e8b2c0d4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ap_issued_qa.py`
- `analysis/outputs/ap_issued_qa.md`
- `ap_issued_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `issued_qa.*`, `dso_qa.*`, `dpo_qa.*`, `credit_note_qa.*`, `n_cust_qa.*`, `n_types_qa.*`, `invoices.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617.

## Locked verdict

| object | decision |
| --- | --- |
| ap leftover after days (Y3) | **CLOSE** |
| 15-col Y3 card | **KEEP off the card** |
| TURNOVER add-on | **CLOSE** |
| AR issued leftover | **CLOSE (locked)** |
| y_ap_issued | **PARK** |

CLOSE leftover-after-days rank 0.591 (OLS 0.714, fake=True). Y3 ap 0.675 vs days 0.711 vs size 0.617. SIZE=True twin=False. Inverse days-after-ap 0.683. Leftover after AR 0.578. After d_n_supp 0.523. both>0 leftover-days 0.471. ap>0 leftover 0.532. Boot leftover-days p50=0.584 p05=0.544. Card: CLOSE unused leftover / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## What failed / next

- Y3 leftover after days rank 0.591 OLS 0.714 fake=True
- inverse days leftover 0.683
- leftover after AR 0.578 rewrite=False
- Y5 leftover after size 0.563 leak_ok=False
- Q6 ap_lag1 leftover 0.547 now 0.568
- leftover after ap_lag1 0.614 Y7 0.542
- leftover after size 0.620 DPO 0.635
- boot leftover p05=0.544 p50=0.584 p95=0.630
- ap>0 leftover 0.532 fake=True
- log1p leftover 0.591 fake=False
- intensity leftover 0.582 fake=False
- terciles T1 0.532 T2+T3 0.540
- co-median SIZE=True ρ=0.594
- after d_n_supp 0.523 dies=True after n_tx 0.580
- both>0 leftover-days 0.471 leftover-AR 0.469
- so_far short 0.562 mid 0.605 long —
- per-fold leftover 0.745 0.522 0.445 0.619 0.624
- log1p after size 0.620 SIZE=True twin_lag=True
- Q6 mid Y7 0.557 long Y7 0.494
- ICC(1) 0.009 share 0.063 vs quote 0.86
- AR leftover after AP 0.640 Y7 leftover after AR 0.529
- ever-ERP leftover 0.591 early — late 0.595
- ap/(ap+ar) leftover 0.588 fake=False raw 0.615
- quintiles Q1 0.579 Q5 — hi 0.564
- fold0 leftover — fake=True rest 0.553 fake=True
- ≥6mo leftover 0.591 fake=True ρ vs supp 0.789
- boot leftover-AR p05=0.541 p50=0.580 p95=0.626
- w/b 0.862 ICC 0.537 (0.86 is w/b; ICC 0.54 CONFIRM)
- winsor leftover 0.591 fake=False
- lag3 leftover 0.550 after lag1 0.545
- Y7 leftover-days 0.565 mid 0.605 fake=True days-after-AP+AR 0.670
- ap>0 after supp 0.498 triple 0.588 log1p days+size 0.586
- DPO-defined leftover 0.532 after DPO+days 0.545 ρ(rank-resid,days)=0.081
- country ES leftover — fake=True missing 0.599 fake=True
- zero-heavy leftover 0.586 zero-light 0.370
- leftover after open 0.677 overdue 0.652 days+AR+size 0.533 open+days 0.611 zero-light-AR 0.477
- never-zero leftover-days 0.281 leftover-AR 0.594 days-after-ap 0.629
- has_ap leftover 0.533 has-any-zero 0.461
- issued/open leftover 0.625 after issued 0.601
- leftover after AR_lag1 Y3 0.593 Y7 0.447 +days 0.533
- leftover after both lag1s Y3 0.592 Y7 0.517
- leftover after group-mean days 0.648 group-AP 0.668 +days 0.602
- leftover after ss 0.641 salary 0.644 ss+salary+days 0.601
- card: CLOSE unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put e_ap_issued on the 15-col card
