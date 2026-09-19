# Wave 4 — AP open leftover after days

Agent `e8b2c0d4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ap_open_qa.py`
- `analysis/outputs/ap_open_qa.md`
- `ap_open_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `dpo_qa.*`, `ap_issued_qa.*`, `issued_qa.*`, `pending_qa.*`, `supp_top1_qa.*`, `ogtg_qa.*`, `invoices.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617.

## Locked verdict

| object | decision |
| --- | --- |
| open leftover after days (Y3) | **CLOSE** |
| 15-col Y3 card | **KEEP off the card** |
| TURNOVER add-on | **CLOSE** |
| AP issued leftover | **CLOSE (locked)** |
| DPO | **DROP (locked)** |
| y_ap_open | **PARK** |

CLOSE leftover-after-days rank 0.418 (OLS 0.695, fake=False). Y3 open 0.569 vs days 0.711 vs size 0.617. SIZE=False twin=False. Inverse days-after-open 0.720. Leftover after issued 0.459 rewrite=True. Leftover after DPO 0.605. Boot leftover-days p50=0.455 p05=0.399. Card: CLOSE unused leftover / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## What failed / next

- Y3 leftover after days rank 0.418 OLS 0.695 fake=False
- inverse days leftover 0.720
- leftover after issued 0.459 rewrite=True
- leftover after DPO 0.605 numer=False
- Y5 leftover after size 0.541 leak_ok=False
- Q6 open_lag1 leftover 0.442 now 0.517
- leftover after AR open 0.423 same=True
- boot leftover p05=0.399 p50=0.455 p95=0.571
- open>0 leftover 0.553 fake=False
- log1p leftover 0.418 fake=False
- leftover after pending 0.562 +days 0.469 fake=True
- terciles T1 0.408 T2+T3 0.560
- co-median SIZE=True ρ=0.517
- so_far short 0.512 mid 0.463 long —
- per-fold leftover 0.412 0.455 0.412 0.361 0.452
- after issued+DPO 0.531 +days 0.540 issued-after-open 0.677
- Q6 mid Y7 0.433 leftover 0.376
- days+size leftover 0.417 intensity 0.422 Y7 0.462 never-zero 0.593
- leftover after overdue 0.482
- never-zero leftover-days 0.593 leftover-issued 0.637 +days 0.648
- leftover after ss 0.553 salary 0.545 ss+salary+days 0.414
- DPO-defined leftover 0.552 after DPO+days 0.429
- ever-ERP leftover 0.418 late 0.406
- winsor leftover 0.418 after AR issued 0.459 AR-open+AP-issued 0.573
- leftover after issued_lag1 Y3 0.447 Y7 0.430 +days 0.458
- BETWEEN leftover after cmean 0.530 +days 0.563 after lag1 0.589 within 0.615
- leftover after n_tx 0.422 +days 0.421 days+issued+size 0.550 DPO-after-open 0.658
- leftover after trail 0.580 +days 0.419 after AR_open_lag1 0.423 +days 0.478
- leftover after log1p(issued) 0.459 +days 0.549 issued+AR_open+days 0.569
- card: CLOSE unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put e_ap_open on the 15-col card
