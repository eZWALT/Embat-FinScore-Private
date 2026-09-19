# Wave 4 — d_cust_top1 leftover after days as Y3 X

Agent `572fb928`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/top1_qa.py`
- `analysis/outputs/top1_qa.md`
- `analysis/outputs/top1_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `cust_hhi_qa.*`, `n_cust_qa.*`, `n_supp_qa.*`, `n_types_qa.*`, `ap_issued_qa.*`, `counterparties.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, `brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. Y4 >0.975 footnote **KEEP**.

## Locked verdict

| object | decision |
| --- | --- |
| `d_cust_top1` as Y3 X / 15-col card | **CLOSE unused leftover** |
| `d_cust_top1` as engine X on the 44 | **DROP** |
| Y4 >0.975 monopoly footnote | **KEEP** |
| `y_top1` | **PARK** |

Y3 leftover after days rank 0.525 (dies=True, fake=False); inverse days after top1 0.715. Single 0.590 vs days 0.711 vs size 0.617. ρ vs HHI 0.994 vs n_cust -0.806 vs days -0.287 vs size -0.169. after HHI 0.434 after n_cust 0.532. Q6 lag1 leftover 0.523. Y4 top1_lag3 0.604. vs supp_top1 same=False. unused leftover after days: honest rank 0.525 dies (OLS 0.419 fake=False). Also TWIN of ['d_cust_hhi', 'd_n_cust']. DROP from the 44 as Y3 X. Y4 >0.975 footnote KEEP. Do not invent y_top1. Off the 15-col card.

Dark 470 NaN CONFIRM. Calendar incomplete (period<2025-02) top1 nn=0 CONFIRM. HHI leftover after top1_lag3 OLS 0.549 / rank 0.461 CONFIRM. Y4 HHI_lag3 >0.975 22.1% / 38 pos vs rest 11.5% CONFIRM; body CV 0.445 CONFIRM. after days+HHI 0.572 is rewrite residual R²=0.966 same n, not leftover. Bootstrap leftover p05/p50/p95 0.394/0.538/0.599 (65% die). Permute-within-days p50=0.521 — observed 0.525 is the null. Q6 days_lag1 0.684 CONFIRM; HHI Q6 short present 21.7% / 42 pos CONFIRM. Do not put top1 on the 15-col card. Do not invent y_top1.

## What failed / next

- none

Elapsed 31s.
