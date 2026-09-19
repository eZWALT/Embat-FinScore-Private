# Wave 4 — a_n_tx leftover after days

Agent `c81e4b07`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/n_tx_qa.py`
- `analysis/outputs/n_tx_qa.md`
- `analysis/outputs/n_tx_qa.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `cashflow.py`, `ops.py`, `gap_sd_qa.*`, `cust_hhi_qa.*`, `dso_qa.*`, `fc_r_qa.*`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **CLOSE** 0.538 |
| 15-col card | **CLOSE as a days rewrite / DROP from the card** |
| identity vs c_n_tx | **YES** |
| days leftover after n_tx | **0.564** thin rank / OLS-high |
| Q6 | **CLOSE** |

Y3 a_n_tx 0.703 vs days 0.711 vs size 0.617. ρ vs days 0.938 / c_n_tx 1.000 / gap -0.866. Leftover after days rank 0.538 OLS 0.623 fake=False. Inverse rank 0.564 OLS 0.710. ICC 0.969. Identity dust leftover 0.702. lag1 after days_lag1 0.534. Leftover after gap_sd 0.615. so-far<6 leftover 0.580 (n_tx 0.656 vs days 0.622).

## What failed / next

- leftover after c_n_tx rank 0.702 is identity dust (max|resid|=2.8e-14)
- OLS leftover after days 0.623 is high vs honest rank 0.538 (ρ(resid,days)=-0.434 < 0.80 — not a fake-days clone, still rank-dies)

Elapsed 9s.
