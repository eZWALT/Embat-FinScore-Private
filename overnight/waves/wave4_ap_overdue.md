# Wave 4 — AP overdue leftover after days

Agent `e8b2c0d4`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ap_overdue_qa.py`
- `analysis/outputs/ap_overdue_qa.md`
- `ap_overdue_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `delay_qa.*`, `dpo_qa.*`, `ap_open_qa.*`, `n_accounts_qa.*`, `util_snap_qa.*`, `invoices.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, LIVE, canvas, TURNOVER, or the 15-col card. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617.

## Locked verdict

| object | decision |
| --- | --- |
| overdue leftover after days (Y3) | **CLOSE** |
| 15-col Y3 card | **KEEP off the card** |
| TURNOVER add-on | **CLOSE** |
| AP open leftover | **CLOSE (locked)** |
| delay / overdue as Y3 X | **DROP (locked delay_qa)** |
| DPO | **DROP (locked)** |
| y_ap_overdue | **PARK** |

CLOSE leftover-after-days rank 0.584 (OLS 0.564, fake=False). Y3 overdue 0.625 vs days 0.711 vs size 0.617. SIZE=False twin=False. Inverse days-after-overdue 0.711. Leftover after open 0.623 rewrite=False. Leftover after DPO 0.544 / delay 0.579. Boot leftover-days p50=0.579 p05=0.420. Card: CLOSE unused leftover / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## What failed / next

- Y3 overdue 0.625 vs days 0.711 vs size 0.617 beat=0.008
- leftover-after-days rank 0.584 OLS 0.564 fake=False dies=False
- inverse days-after-overdue 0.711
- leftover after open 0.623 rewrite=False +days 0.581
- leftover after DPO 0.544 +days 0.525 after delay 0.579 +days 0.549
- vs overdue_30 leftover 0.567 leftover-days-30 0.590
- Q6 Y7 lag1 leftover 0.433 Y3 now 0.557 delay_early_nn=0
- Y5 leftover-size 0.427 leak_ok=False
- boot leftover-days p05=0.420 p50=0.579 p95=0.632
- terciles T1 0.476 T2+T3 0.559
- so_far short 0.526 mid 0.598
- per-fold — — — 0.693 0.469
- Q6 mid leftover 0.379 long raw 0.505
- ever-ERP leftover 0.584 late 0.583
- BETWEEN cmean 0.579 lag1 0.600 within 0.483
- Y7 leftover-days 0.406 leftover-open 0.404
- leftover after days+size 0.589 ss+salary+days 0.568 AR 0.603 triple 0.452
- overdue>0 leftover 0.624 winsor 0.584 open-defined 0.584
- same-n overdue 0.625 days 0.725 size 0.599 beat=0.026 leftover-ntx 0.582 trail+days 0.583
- leftover after delay_lag1 0.573 +days 0.543 DPO-defined 0.547 DPO+days 0.525 sign=1
- leftover after open_lag1 0.624 +days 0.574 days+n_tx+size 0.586
- card: CLOSE unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put e_ap_overdue on the 15-col card
