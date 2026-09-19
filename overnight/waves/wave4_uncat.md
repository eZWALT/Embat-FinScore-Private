# Wave 4 — uncat Q5 readability (slot 29)

Agent `ec17da1b`. Train only (holdout 72 coverage). Seed 20260918. No parquet rewrite. No CAT_MAP edit. No merged Y.

## Files written

- `analysis/evaluate/uncat_qa.py` (create; 46 cuts)
- `analysis/outputs/uncat_qa.md`
- `analysis/outputs/uncat_quintiles.png`
- append-only `analysis/experiments/registry.csv`
- this note

## Columns

- Store keep: `a_uncat_share` (count share, uncategorized OR not in CAT_MAP).
- In-memory only (not merged): `amt_uncat`, leftover tokens, inflow/outflow uncat. Family M stays CLOSED.

## Train coverage

- 21,157 CM / 1,214 companies. `a_uncat_share` cov 95.8% (CONFIRM). Holdout 72 / 1,073 CM defined 97.3% (no AUROC).

## Decision

| object | call |
| --- | --- |
| Q5 why (`a_uncat_share` / amount-uncat) | **CLOSE** |
| uncat as health Y | **PARK** (do not invent) |
| uncat as Y3 X | **PARK** (size 0.617 ≥ 0.60) |
| Y2 amount-uncat 0.602 | **PARK as X** (style, not a month why) |
| Q6 lag1 | **CLOSE** |
| CAT_MAP leftover add | **CLOSE** (only token `uncategorized`) |
| Family M merge | **CLOSED** |

## What failed

- Amount-uncat does **not** beat size by ≥0.02 on Y3 (0.530 vs 0.617, Δ −0.075) and loses to `c_n_days_with_tx` 0.711.
- Count acf1 0.270 / ICC 0.985 and amount acf1 0.050 / ICC 0.978 are style dummies (always-messy 206 / always-clean 309 / shock 111). Company-mean Y2 0.586 vs demean 0.527.
- Leftover besides `uncategorized`: 0 txs. Do not add that token. `cash_settlements` is a never-seen Javier alias.
- High-uncat is not mix-missingness, pending, card, dark-only, weekday, or a blank memo.

## Next idea

Drop `a_uncat_share` from Q5 why. Do not merge amount-uncat. Do not invent a Y. `y2_why` (in flight) should not treat uncat as a month change — invoiced-only style 0.557 vs shock 0.525 still carries.
