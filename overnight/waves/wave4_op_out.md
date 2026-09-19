# Wave 4 — a_op_out leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/op_out_qa.py`
- `analysis/outputs/op_out_qa.md`
- `analysis/outputs/op_out_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `a_vol_qa.*`, `transfer_qa.*`, `growth_qa.*`, `in3_qa.*`, `zero_in_qa.*`, `n_tx_qa.*`, `n_accounts_qa.*`, `gbm_core.py`, `cashflow.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. Do not quote a_out_vol 0.722 as the engine.

## Locked verdict

| object | decision |
| --- | --- |
| `a_op_out` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `a_op_out` as engine X on the 44 | **DROP** |
| `y_op_out` | **PARK** |
| TURNOVER | **CLOSE** — do not grow 0.720 |
| size bar | **KEEP quote 0.617** |

## Locked extras

- Honest leftover after days rank 0.586 (lives=True, OLS fake=True); inverse 0.638.
- Single 0.678 vs days 0.711 vs size 0.617 vs a_out3 0.601.
- SIZE=True twins=['a_out3', 'a_out6'].
- leftover after a_out3 0.617 rewrite=False. after days+size 0.576.
- Q6 lag1 leftover 0.561. Dark leftover 0.632 ERP 0.581.
- ICC=0.955 median company Pearson acf1=0.008 n_co=1199. BETWEEN trait.
- a_out3 Y3 0.601 leftover after days 0.517 dies=True fake=True. a_op_out leftover after days+a_n_tx 0.577 fake=False. after days+size+a_out3 0.572 dies=False.
- demeaned a_op_out Y3 0.485 leftover after days 0.525 dies=True fake=False. ICC 0.955 BETWEEN + acf1 0.008 — demean kills the trait.
- out>0 leftover 0.541 T1 0.575 T2 0.553 T3 0.438.
- a_op_out/a_out3 Y3 0.608 leftover after days 0.583 dies=False fake=False. after days+a_out3 0.583.
- leftover after size rank 0.623 OLS 0.598 fake=True ρ=-0.866. leftover after a_n_tx rank 0.576 fake=True ρ=-0.818. leftover after days+a_out6 0.563 fake=False.
- month-shock (a_op_out-a_out3/3) Y3 0.561 leftover after days 0.548 dies=True fake=False. after days+size 0.545.
- last-month Y3 — leftover — n=0. first-half leftover 0.585 last-half 0.566. trail≥12 leftover 0.577 short —.
- MoM Δ a_op_out Y3 0.536 leftover after days 0.524 dies=True fake=False. a_op_out/a_op_in Y3 0.550 leftover 0.533 dies=True.
- 2025 leftover 0.561 n=4208. 2026 leftover 0.635 n=955. a_out3-defined leftover 0.582. after days+size+out3+n_tx 0.575 fake=False. after a_out3+size 0.614.
- sibling n_co≥2 leftover 0.588 n=5408. solo leftover —. never-zero leftover 0.535 ever-zero leftover 0.538.
- last-labeled Y3 0.729 leftover 0.660 n=725 pos=154. first-labeled leftover 0.580 n=725. days>0 leftover 0.569.
- winsor 1/99 Y3 0.678 leftover 0.586. mid-quintile leftover 0.469 tail leftover 0.611. log_opout leftover after size 0.623 fake=False. leftover after days+io 0.585.
- last-labeled leftover after size 0.704 fake=True. after days+size 0.672. outflow/day Y3 0.630 leftover after days 0.569 after days+size 0.563. Q5 leftover 0.695 Q1 leftover 0.456.
- leftover after days+size+out3+n_tx+io 0.575 fake=False. after days+a_out12 0.559. log leftover after days+size 0.575. high-median-out leftover 0.622 low-median-out leftover 0.454.
- last-labeled Dark leftover 0.683 n=262 ERP leftover 0.659 n=463. Dark after size 0.693 ERP after size 0.718. last-labeled leftover after a_out3 0.672.
- early-onboard leftover 0.601 late-onboard leftover 0.598. high ρ(out,size) leftover 0.482 low ρ leftover 0.695 median company ρ=0.336 n_co=347.
- low-ρ Y3 0.767 leftover after days+size 0.603 after days+out3 0.581 a_out3 leftover 0.678 ρ vs a_out3 0.830 n=2155. high-ρ Y3 0.596 leftover after days+size 0.581 after days+out3 0.630 ρ vs a_out3 0.910 n=2205.
- a_op_out/a_in3 Y3 0.612 leftover after days 0.583 after days+size 0.579. net-out leftover 0.525 net-in leftover 0.662. high-n_tx leftover 0.590 low-n_tx leftover 0.574.
- ever-Y3 leftover 0.566 n=919 never-Y3 leftover —. io≥1 leftover 0.609 io<1 leftover 0.569. last-3-labeled leftover 0.604 n=1901. log_opout-log_in3 Y3 0.548 leftover 0.551.
- last-3 leftover after days+size 0.601 after days+out3 0.580. last-3 2026 leftover 0.635 n=955. last-labeled leftover after days+size+out3 0.654. high a_n_tx leftover 0.562.
- company ρ(out,out3)<0.80 leftover 0.614 Y3 0.703 n=3896 n_co=304 panel ρ vs a_out3 0.854. after days+size 0.608. twin-co leftover — n_co=47 median company ρ=0.555.
- a_out3-missing leftover — Y3 — n=120. age≤3 leftover — after days+size —. age>12 leftover 0.569.
- age 4–12 leftover 0.573 n=3504. age>12 leftover after days+size 0.580 after days+out3 0.573. Dark 2026 leftover — n=356 ERP 2026 leftover 0.629 n=599.
- stable-outflow leftover 0.437 noisy leftover 0.565. size-IQR leftover 0.562. age>12 leftover after days+size+out3 0.573.
- noisy leftover after days+size 0.557 after days+out3 0.582. outflow CV Y3 0.795 leftover after days 0.734 dies=False.
- outflow CV Y3 0.795 leftover after size 0.763 fake=False ρ vs size -0.367. after days+size 0.732. ρ vs days -0.469. PARK — do not put CV on the 15-col card; this is not a_op_out KEEP.
- CV leftover after a_n_tx 0.735 ρ=-0.478. after a_out3 0.760 ρ=-0.448. after days+n_tx+size 0.732. PARK — not a_op_out KEEP; off the card.
- a_op_out leftover after days+CV 0.518 fake=False. after days+CV+size 0.469. after days+CV+out3 0.534.
- CV leftover after a_op_out 0.730 fake=False. after a_op_out+days 0.717. a_op_out leftover after days+CV died 0.518 — vol ate the leftover.
- a_op_out leftover after CV-only 0.548 fake=False. last-labeled leftover after days+CV 0.586 n=719.
- leftover after days+CV Dark 0.562 ERP 0.440 T3 0.469 2026 0.582.
- leftover after days+CV Q5 0.766 mid 0.638 net-out 0.549.
- Bootstrap leftover-after-days rank p05=0.568 p50=0.590 p95=0.644 share<0.55=0.0% n=40.

honest leftover after days rank 0.586 lives (OLS 0.675 fake=True ρ(resid,days)=-0.841). TWIN of ['a_out3', 'a_out6']. SIZE (|ρ| vs log1p(a_in3) ≥0.50). DROP from the 44. Off the 15-col card.

## What failed / next

- none

Elapsed 18s.
