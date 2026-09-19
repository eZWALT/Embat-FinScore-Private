# Wave 4 — unused `e_pending_amt_share` leftover after DSO / issued

Long-lived child `86399952`. Same module ≥30 min: write → run → leftover / twins / Q6 → U-shape → fold 4 → new-book → all-paid dummy → AR/AP mix → same-n → inverse delay. No parquet rewrite. No new GBM. No `build_targets`. No 0–100. Did not edit `invoices.py`, `gbm_y7_core.py`, `delay_qa.py`, `credit_note_qa.py`, `zero_in_qa.py`, or `product/`. Night Y7 **TURNOVER 0.720 / B_shallow 0.712** and Y3 **0.762 / 0.752** unchanged. Y7 never D. Y5 never E. TURNPEND 0.7184 already lost — do not grow TURNOVER.

## Files

- `analysis/evaluate/pending_qa.py` (create / owner)
- `analysis/outputs/pending_qa.md`
- `analysis/outputs/pending_qa.png`
- append-only `analysis/experiments/registry.csv` (`agent=86399952`)
- `overnight/waves/wave4_pending.md` — this note

## Columns

`e_pending_amt_share` — among issued non-cancel invoices with `issuance_date ≤ period_end`, abs-amount share still unpaid as of period_end (`payment_date_invalid` = 0 in num and den). Snapshot `pending_amount` not used (would leak). In-memory only: `j_pay_match`, AR-only / AP-only pending shares, lags 1/3. Not written to the store.

## Coverage (train; holdout count only)

| item | n |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| pending defined | 12,762 (**60.3%**; quote 60.3%) |
| train NaN CM | 8,395 (dark + 792 ERP months before first issued) |
| ever-ERP / never-ERP | 744 / **470** |
| dark pending nn / 0-filled | **0 / 0 — CONFIRM NaN not 0** |
| acf1 / size ρ vs log1p(a_in3) / vs \|op_in\| | 0.779 / 0.063 / **0.044** (quote 0.044) |
| formula vs raw SQL | **MATCH** max\|Δ\|=0 on 12,762 |
| holdout coverage | 557/1,073 = 51.9% on 72 (no AUROC) |
| holdout dark nn | 0 |
| Y7 labeled on dark | 0 |

## Verdict (map to Q1–Q6)

| object | decision |
| --- | --- |
| pending as Y7 leftover after DSO / issued_lag1 | **DROP** — leftover 0.421 / 0.418 / both 0.419. Rank-ortho 0.461 / 0.424. Same-n residual 0.418 vs issued 0.630 (lift −0.211). |
| pending as TURNOVER add-on (TURNPEND) | **CLOSE** — 0.7184 vs TURNOVER 0.720 already lost. Do not grow the card. |
| pending as Y3 X / 15-col card | **DROP** — leftover after days 0.443 vs days 0.711. |
| `e_pending_amt_share` on the 44 | **DROP from the 44** — cluster rep, but leftover dies and TURNPEND lost. |
| pending as a health Y | **PARK** — do not invent `y_pending`. |
| Q5 why (unpaid stock) | **PARK** — ICC 0.971 η² 0.672 (trait like a_out_vol 0.74); demean Y7 0.541. Company-mean vs Y7 rate ρ=0.022. |
| Q6 lead | **CLOSE** — now 0.420 / lag1 0.430 / lag3 0.443. First 6m pending cov 58.3% vs delay **0.0%** (populated earlier, not a lead). |
| Family J merge | **CLOSE** — leftover after match 0.441. Do not merge J. |
| AP pending parquet column | **CLOSE** — none in store; in-memory AP ρ=0.867. Do not invent. |
| U-shape / all-paid dummy | **CLOSE** — \|p−0.30\| leftover 0.558; all-paid leftover after issued 0.572. Thin dummy. Not KEEP-as-X. |

Twin/SIZE: not SIZE (ρ 0.063 / 0.044). Not a \|ρ\|≥0.80 twin of DSO (0.492), overdue (0.017), delay (0.016), issued_lag1 (−0.056), `j_pay_match` (−0.093).

Siblings (read-only): CN leftover after issued_lag1 **0.597 KEEP**; delay leftover after DSO **0.581** / issued_lag1 **0.583 KEEP**. Inverse: delay after pending **0.582** — delay keeps its leftover; pending does not.

## NORTH_STAR

Q4 dip-vs-fall is the Y7 label — pending is unused stock (linear AUROC 0.420, leftover dies). Q5: BETWEEN who-has-unpaid-stock, not this-month shock; the hotter Y7 tail is the **emptied book** (Q1 35.6%), not unpaid stock (high-pending & low-overdue Y7 23.0% vs both-high 36.3%). Q6: populated in the first 6 months unlike delay, but lag ≈ now. Hidden 72: coverage only.

## What failed / next

- Leftover after DSO / issued_lag1 / days all die (<0.55). KEEP-as-X fails even though it is not a twin and not SIZE.
- TURNPEND 0.7184 already lost to TURNOVER 0.720. Fold 4 pending 0.363 vs issued 0.647 — hard groups GROUP_0222 / GROUP_0108 have pending medians 0.062 / 0.013 (already-paid short-DSO churn).
- U-shape Q1/Q5 35.6% / 33.3% vs mid ~24%. \|p−0.30\| leftover 0.558 is a thin dummy — CLOSE, do not invent `y_pending`.
- Mixed share blends AR+AP (closer to AR). No supplier-pending column to add.
- Delay leftover survives pending (0.582). CN leftover is the unused-E KEEP, not pending.

## Re-run

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.pending_qa
```
