# Wave 4 — credit-note leftover after issued_lag1

Long-lived child `0c3bf32d`. Same module ≥30 min: write → run → leftover artifact → who-uses-notes (passes 1–23). No parquet rewrite. No new GBM. No `build_targets`. No 0–100. Did not edit `invoices.py`, `gbm_y7_core.py`, `zero_in_qa.py`, `growth_qa.py`, `recency_qa.py`, or `product/`. Night Y7 **TURNOVER 0.720 / B_shallow 0.712** and Y3 **0.762 / 0.752** unchanged. Y7 never D. Y3 never B.

## Files

- `analysis/evaluate/credit_note_qa.py` (create / owner)
- `analysis/outputs/credit_note_qa.md`
- `analysis/outputs/credit_note_quintiles.png`
- append-only `analysis/experiments/registry.csv` (`agent=0c3bf32d`, skip key includes `x_families`)
- `overnight/waves/wave4_credit_note.md` — this note

## Columns

`e_credit_note_ratio` — |credit notes| / (|invoices| + |credit notes|) issued this period. No canonical `credit_note` type in the book; feature uses ERP stand-ins `note` + `refund`. In-memory lags 1/3 and issued-lag CV (not written to the store).

## Coverage (train; holdout count only)

| item | n |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| CN defined | 11,207 (53.0%); ever-defined 744 |
| among defined: zero / >0 | 57.7% / 42.3% |
| ever-ERP / never-ERP | 744 / 470 |
| dark CN nn / 0-filled | 1 / 0 — COMP_0962 refund-only; **not a 0-fill** |
| Y7 labeled / pos / CN-nn | 7,464 / 2,149 / 7,308 |
| Y3 labeled / pos / CN-nn | 5,648 / 402 / 3,079 |
| holdout Y7 pos | **122** (coverage only) |
| holdout dark 32 CN nn | 0 |

Formula vs raw: **MATCH** (max\|Δ\|=7.8e-16). Token mass: note **98.6%**, refund **1.4%**, canonical 0.

## Verdict (map to Q1–Q6)

| object | decision |
| --- | --- |
| CN as Y7 leftover after issued_lag1 | **KEEP** — residual 0.597; ρ vs issued_lag1 0.237 / DSO −0.050 (not a twin); leftover after issued 0.600; leftover after DSO 0.479 (not DSO). Same-n lift +0.049. Demean leftover 0.509 dies — who-uses-notes, not this month’s correction. |
| CN as Y7 add-on / grow TURNOVER | **CLOSE** — do not grow the card; night 0.720 / 0.712 stays |
| CN as Y3 X / the 44 | **CLOSE / DROP from the 44** — 0.579 vs days 0.711; leftover after days 0.560 |
| CN as a health Y | **PARK** — do not invent `y_credit_note` |
| Q5 footnote without “thin issuance” | **KEEP-Q5 footnote** — who-uses-notes leftover after issued |
| Q6 CN lag1 | **CLOSE** — short so-far 0.542 CONFIRM q6_quoted |
| Fold 4 Y7 | TURNOVER 0.680 is issued (0.647), not CN (0.608 / residual 0.631) |

## NORTH_STAR

Q4 dip-vs-fall is the Y7 label. Q5 why: companies that use the `note` stand-in book have higher P(top-1 lost) after issued volume — not stretched DSO, not thin issuance, not a within-month correction. Q6 lag1 is weak on short books. Hidden 72: coverage only (122 Y7 pos).

## What failed / next

- First dark check said FAIL on 1 non-null. Tightened: 0-fill is the crime; COMP_0962 is refund-only (amount 1.87), the other 469 stay NaN.
- Raw CN 0.545 looked like chance; leftover 0.597 with R²=0.002 looked like a sample trick. Same-n raw is 0.548 — the +0.049 lift is real. Demean then dies: leftover is between-company style.
- Fold 4 leftover 0.631 does not reach TURNOVER 0.680. Do not put CN back as the fold-4 save.
- Do not grow TURNOVER. Do not put CN on the Y3 15-col card.

## Re-run

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.credit_note_qa
```
