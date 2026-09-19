# Wave 4 — c_tax_month leftover after days

Agent `a91c4e02`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/tax_month_qa.py`
- `analysis/outputs/tax_month_qa.md`
- `analysis/outputs/tax_month_qa.png`
- append-only `analysis/experiments/registry.csv`
- this note

Did not overwrite `tax_qa.*` (missed-tax), `ss_qa.*`, `salary_qa.*`, `salary_month_qa.*`, `in3_qa.*`, `issued_qa.*`. Did not touch `ops.py`, `gbm_core.py`, the 15-col card, TURNOVER, product/, or the parent journal. Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. TURNOVER **0.720 / 0.712**.

## Locked verdict

| object | decision |
| --- | --- |
| leftover after days | **CLOSE** 0.520 |
| as Y3 X | **CLOSE unused leftover** |
| leftover after ss | **0.528** |
| leftover after salary | **0.538** |
| leftover after days+salary+ss | **0.489** |
| days leftover after tax | **0.682** |
| calendar | **q_peaked** Q-gap 25.7% |
| Q6 | **CLOSE** |

Y3 tax 0.611 vs days 0.711 vs size 0.617 vs salary 0.671 vs ss 0.693 vs missed 0.511. Leftover after days rank 0.520 OLS 0.520 ρ(resid,days)=-0.075. lag1 leftover after days_lag1 0.519. Jaccard vs missed 0.000 complement=False. Bootstrap leftover-after-days p05/p50/p95 0.458 / 0.520 / 0.554 (share die 93.2%). Calendar leftover after Q+days 0.537. BETWEEN leftover after days-mean 0.645.

CLOSE: leftover after days dies and Y3 tax does not beat size. Do not put tax on the 15-col card. Presence is Q-peaked like missed-tax, not a monthly salary/SS cousin.

## What failed / next

- no replica miss; leftover after days is the unused-leftover decision

Elapsed 42s.
