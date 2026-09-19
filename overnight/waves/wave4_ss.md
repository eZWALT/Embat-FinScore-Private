# Wave 4 — c_ss_month leftover after days

Agent `d4c8e201`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/ss_qa.py`
- `analysis/outputs/ss_qa.md`
- `analysis/outputs/ss_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not touch `n_tx_qa.*`, `salary_qa.*`, `tax_qa.*`, `ops.py`, `gbm_core.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, canvas, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. Salary 0.671. TURNOVER **0.720 / 0.712**.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **KEEP** 0.635 |
| 15-col card | **KEEP** |
| leftover after salary | **0.668** payroll_twin=False |
| leftover after days+salary | **0.633** |
| leftover after tax | **0.668** |
| days leftover after SS | **0.641** |
| leftover after full stack | **0.635** |
| BETWEEN leftover after days-mean | **0.651** |
| mixed-cadence leftover after days | **0.419** (dies) |
| Q6 | **KEEP** |
| calendar | **monthly** |

Y3 SS 0.693 vs days 0.711 vs size 0.617 vs salary 0.671. ρ vs days 0.424 / a_n_tx 0.409 / salary 0.619 / tax 0.330 / log1p(a_in3) 0.342. Jaccard(SS, salary) 0.639. Jaccard(SS, tax) 0.486. Leftover after days rank 0.635 OLS 0.635 ρ(resid,days)=-0.086. lag1 leftover after days_lag1 0.631. ICC 0.990. Dark leftover 0.574 invoiced 0.677. Bootstrap leftover-after-days p05/p50/p95 0.593 / 0.634 / 0.675. Perm null p50 0.609 p(obs) 0.025. Leave-one-group leftover min/med 0.628 / 0.634. Sibling leftover after group-mean 0.680; BETWEEN boot p50 0.649.

KEEP is BETWEEN payer identity (always Y3 1.9% vs never 14.3%), not a days twin and not a month flip. Parent absorbs the card.

## What failed / next

- no replica miss; leftover after days lives (KEEP 0.635). Mixed leftover after days 0.419 dies — KEEP is BETWEEN payer identity, not a month flip.

Elapsed 80s.
