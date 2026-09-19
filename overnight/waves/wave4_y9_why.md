# Wave 4 — Y9 why (end note)

Agent `0511f2af`. Lane: `analysis/evaluate/y9_why.py`. No commit. No parquet
rewrite. No `product/`. No 0–100. No GBM/XGB. Did not run `build_targets`.
Family M built **in memory** only. Never F / `a_fin_cost` / D/E as X.

## Files written

- `analysis/evaluate/y9_why.py` — create; write→run cuts 1–28 in this module
- `analysis/outputs/y9_why.md` — tables + CLOSE / PARK letters
- `analysis/outputs/y9_fee_share_quintiles.png` — fee-share bins vs both Y9s
- `overnight/waves/wave4_y9_why.md` — this note
- `analysis/experiments/registry.csv` — append-only once (143 train-only rows)

Did not edit q6_quoted, sibling_h, debt_schedule_qa, catmix, gbm_y9,
y9_fees, build_targets, parquet, product/.

## Verdict

Y9 is **not an outflow tail** (31% of own-p80 positives are high `a_out6`;
2×2 neither 42%). It is **not a usable mix shift**. Columns that beat
`a_out6` **0.565** (`m_fin_share` **0.618**, `m_fee_share` **0.613**,
`m_fee_n_share` **0.611**) equal any-`a_fin_cost>0` (**0.610**, gap +0.001)
and fail |ρ|≥0.80 vs `a_fin_cost` (0.875 / 0.830). That is the Y. PARK.

Best legal leftover: `m_int_share` **0.525** (CLOSE; loses to outflow).
Fee vs int ρ=+0.11 — not substitutes. `m_uncat_share` 0.511 CLOSE.

**Merge Family M: no.** KEEP gate (beat `a_out6` by ≥0.02 and not an F-copy)
is not met. Write a recommendation only if it had cleared — it did not.

## Why the GBM parked

After dropping F / `a_fin_cost`, leftover `a_out6` 0.565 is a **no-fee
shield** (Q1 no-fee 6.7% vs any-fee 22.9%). It collapses to 0.51 on
later-fin / always-fin and to **0.530** on large∩any-fee. The Q1 shield
survives inside inflow terciles (T1 −8.1 pp / T2 −6.5 / T3 −3.7). Spike
GBM's `a_op_in` (−) **0.563** matches this module and is size (AUROC 0.94).

## Q6

Lag-1 `m_fee_share` **0.571** does not clear 0.565+0.02. Company-median
acf1 **−0.055** (pooled +0.315 is zeros lining up). The 64 before-first-fin
own-p80 positives are **exactly** the month before first fee (rate 1.000;
2–3m before is 0) — algebraic forward label (Y looks at t+1..t+3). t3 mix
already CLOSE. Do not claim a fee lead.

## Other

- later-fin intensity is tail-only like Y4 HHI (15.8% → 22.5% Q5; single 0.550).
- Dark 470 vs invoiced 744: own-p80 **14.1% = 14.1%**. Never D/E.
- Spike is turning (company ρ −0.03). Own-p80 is a mild trait (ρ +0.32).
- Label stays. Holdout is coverage / LOW_POWER only.

## Next idea (parent)

Do not merge M. Do not retune Y9 trees. Optional footnote: the 110
mixed-dark spike residual after no-fin (21.4% vs 12.9%) — not a merge
reason and not a D/E X.
