# Wave 4 — missed-salary QA (slot 2)

Agent `0fd41cbf`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/salary_qa.py`
- `analysis/outputs/salary_qa.md`
- `analysis/outputs/salary_calendar.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+y+model+split+metric+x_families)
- this note

Did not touch `ops.py`, `tax_qa.*`, `uncat_qa.py`, `companies_qa.py`, `a_vol_qa.py`, `factoring_qa.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, or the 15-col card. Night Y3 quote stays **0.762 / 0.752**.

## Columns / coverage (train)

- `c_salary_month` 40.1% (8,484 / 21,157 CM, 1,214 cos). Store vs raw `category=salary` 100%.
- `c_missed_salary` 2.9% (623). Modal 0 share **97.1%** (CONFIRM feature-report). Holdout mean 1.4% / 15 CM — no AUROC.
- Formula OK. Include-current quirk is stricter: exclude-current would add 144 CM (Y3 0.516). Do not rewrite ops.py.
- Payroll is real mass: |amt| p50 1,977 (≤1€ share 0.3%). Last salary before a miss p50 4,512.

## Locked verdict

| object | decision |
| --- | --- |
| `c_missed_salary` as Y3 / Q3-Q5 X | **CLOSE** |
| `c_missed_salary` as a health Y | **PARK** (do not invent `y_missed_salary`) |
| calendar dummy | **no** — monthly (Q-sal 40.5% vs other 39.9%), not tax-style Q-peak |
| Q6 lag1/lag3 | **CLOSE** |

Y3 CV **0.513** vs size **0.617** (Δ −0.104) vs days **0.711**. `c_salary_month` already on the 15-col card at **0.671**. No |ρ|≥0.80 twin (vs `c_missed_tax` ρ=0.075). ICC 0.736 BETWEEN (salary_month ICC 0.978 is the trait). Drop 12 chronic Y2 names: 0.494 → 0.492 (no flip).

## What failed / extras that stayed CLOSE

- Usual-only (sal6≥3) Y3 0.591 vs size 0.641 / days 0.719. Rate gap 11.6% vs 2.7% is a **size pile** (missers log1p(a_in3) p50 12.95 vs paid 13.45).
- Monthly-cadence skip 0.511 vs size 0.757. Irregular 0.591. Halt (no SS) 0.517 / token-skip 0.504 / first-miss 0.498.
- Jaccard vs `not_salary` 0.049 — most non-salary months have no usual history. The 15-col lever is salary presence, not the skip.
- vs rejected `y6_missed_payroll`: Jaccard 0.365, ρ 0.505, P(Y6\|missed)=47.7% vs base 6.1%. Distinct window (now vs t+1..t+3). Do not merge.
- Q6: contemporaneous is chance; short books are **not** empty (72 miss / 2,970 CM) but LOW_POWER.
- Dark 470 vs 744: same miss rate (3.1% vs 2.7%).

## Next (parent)

Park `c_missed_salary` next to `c_missed_tax`. Keep `c_salary_month` / `c_ss_month` on the 15-col card. Do not invent a salary Y. Do not put missed-salary on tonight's card.
