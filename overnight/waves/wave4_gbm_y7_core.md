# Wave 4 — shrink global Y7 to a SHAP card (`gbm_y7_core`)

Agent `410a183a`. Global LightGBM on `y7_top1_lost` only (Q4 dip vs fall:
top AR customer issues 0 next quarter — not bankruptcy). No product. No 0–100.
Holdout 72 never in fit, early-stop, pick, or SHAP. Quote **train group-fold CV
0.663**, not holdout 0.680. Never family D.

Re-run:

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_y7_core
```

## Files written

- `analysis/models/gbm_y7_core.py` (owned; copies protocol from `gbm_y7y8`, does not edit it)
- `analysis/outputs/y7_core.md`
- this note
- registry rows appended for A / A_shallow / A_n50 / B / B_shallow / C_shallow /
  MIN / Q5 / Q6 / VOL / TURNOVER / TURNDELAY and the turnover ablations

Did not edit `gbm_y7y8.py`, `explain_y7.py`, `gbm_y1.py`, `catmix.py`,
`data_join_qa.py`, parquet, `product/`, `LIVE.json`. No commit.

## Setup

| item | choice |
|------|--------|
| Y | `y7_top1_lost` only (not the inflow sibling) |
| Train | 7464 / 2149 / 28.79% |
| Split | 5 `group_folds`, seed `20260918`. Holdout 72 out of every fit and val fold. |
| A | 7 cols: clipped DSO, `e_ar_issued_lag1`, credit-note ±lag1, `e_delay_coll`, DSO lag1, `f_fc_r_lag3` |
| B | A minus `e_dso_proxy_lag1` (sign flip) |
| C | A plus `log1p(e_ar_issued)` |
| Trees | 400+ES = published 0.663 spec. Shallow = 50 / depth-3 (Y3 diagnostic). |
| DSO clip | 24 months, constant, not a fit |

## Quote (train CV)

| spec | n_x | CV | sd | vs 0.663 | vs dummy 0.50 | vs single on this X | collapsed |
|------|----:|---:|---:|----------|---------------|---------------------|-----------|
| A 400+ES | 7 | 0.7077 | 0.095 | +0.045 | yes | yes (`e_ar_issued_lag1` 0.630) | **yes** 5/5/42/5/6 |
| **A_shallow** | 7 | **0.7111** | 0.084 | +0.048 | yes | yes | no |
| A_n50 | 7 | 0.7124 | 0.090 | +0.049 | yes | yes | no |
| B 400+ES | 6 | 0.7052 | 0.081 | +0.042 | yes | yes | **yes** 1/11/20/4/6 |
| **B_shallow** | 6 | **0.7121** | 0.086 | +0.049 | yes | yes | no |
| C_shallow | 8 | 0.7114 | 0.089 | +0.048 | yes | `log1p(issued)` 0.663 | no |
| D_shallow (− fc lag3) | 6 | 0.6989 | 0.080 | +0.036 | yes | yes | no |
| MIN (DSO + issued lag1) | 2 | 0.6357 | 0.084 | −0.027 | yes | ~tie | no |
| Q5 only | 3 | 0.6737 | 0.080 | +0.011 | yes | yes | no |
| Q6 only | 4 | 0.6440 | 0.074 | −0.019 | yes | yes | no |
| **TURNOVER** (no DSO) | 5 | **0.7200** | **0.034** | +0.057 | yes | yes 0.622 | no |
| **TURNDELAY** | 6 | **0.7293** | 0.041 | +0.066 | yes | yes | no |
| published 278-col | 278 | 0.663 | — | — | yes | `e_ar_issued` 0.598 hold | — |

A folds: 0.739 / 0.796 / 0.742 / 0.715 / **0.546** (same weak fold as 278-col 0.556).
B_shallow folds: 0.741 / 0.802 / 0.739 / 0.707 / **0.571**.
TURNOVER folds: 0.745 / 0.765 / 0.713 / 0.698 / **0.680**.

Leak/size PASS. Worst |ρ| vs D: `e_credit_note_ratio_lag1` −0.299 (fail ≥ 0.80).
Worst |ρ| vs `a_in3`: `e_ar_issued_lag1` 0.464 (fail > 0.85). Y vs `log1p(a_in3)` = 0.465.

Holdout is a check only (122 events): B_shallow 0.706, TURNOVER 0.724. Not the claim.

## KEEP / PARK / CLOSE

Rule: KEEP if claimable 400+ES CV ≥ 0.663 (eps 0.002) with fewer columns **and**
trees do not collapse to 1–5. CLOSE if within 0.02. PARK if lose by > 0.02.

- A/B/C early-stop print 0.70–0.71 but collapse. **Do not KEEP those numbers** (Y3 lesson).
- Honest 50 / depth-3 **holds**: B_shallow **0.712 / n_x=6** vs 0.663 / 278.
- Sign-flip lag is unused (B = A). Euro issued level is unused (C does not lift).
- Two-col MIN PARKs (0.636). Q5-only 0.674 still beats 0.663; Q6-only 0.644 does not.

**Decision: CLOSE — do not PARK. Do not KEEP A/B/C on the 400+ES spec.**
Prefer **B_shallow** if a SHAP card must be named. The number that holds on that
card is **0.712 / n_x=6**.

## What failed

- 400+ES on a 6–8 col X collapses (1–6 trees on most folds). Same Y3 failure.
- `e_dso_proxy` is #1 global SHAP and **worse than chance** on the short-DSO
  quintile (OOF AUROC **0.410**). Fold 4 is GROUP_0222 + GROUP_0108: rate 0.69,
  DSO median **0.18 vs 1.97**. The card under-scores them. Permuting DSO on
  fold 4 drops only 0.007.
- Optional 8ths (overdue, delay lag1) do not help the SHAP card.
- Adding issued-lag CV *on top of DSO* (VOL) only lifts fold 4 to 0.597.

## Next idea

**Drop DSO. Add issued-lag CV** (`std(issued_{t-1,t-2,t-3}) / mean`).
TURNOVER 0.720 / n_x=5 / sd 0.034 / fold 4 **0.680**. Short-DSO quintile flips
from 0.410 to **0.706**. OOF score vs DSO ρ goes from +0.26 to −0.05.
Fold-4 perm: issued lag1 Δ+0.063, issued-lag CV Δ+0.052. |ρ| vs D and size ≈ 0.
TURNDELAY 0.729 is a hair higher; TURNOVER is the more even card.
Do not put family D in X. Do not KEEP the collapsed 400+ES SHAP-card numbers.

Later TURNOVER add-ons (clip CV at 2, lag1/lag3 slope, zero-issue count,
f_fc_r_lag1 / now, 3-month issued mean, log1p lag1, CN×CV, issued lag3,
pending share, FX share, DPO, f_ds_r_lag3, overdue-30, AP delay) either
tie TURNOVER or **hurt fold 4**. TURN3 (drop last-month issued) is 0.693 /
fold 4 0.653 — lag1 is required. TURNOVER 5-col stays the even card.
Gated DSO (≥1 month, BHI) lifts B fold 4 0.571 → 0.641 but mean falls to 0.695.
TURNHI (TURNOVER + gated DSO) fold 4 0.664 — still worse than TURNOVER.

## Story (questions 4–6)

Question 4 is the label. Question 5 (why): stretched AR and credit notes raise
P(top-1 gone) *in the long-DSO majority*. Question 6 (lead): thin last-month
issuance and jumpy issued volume are visible at t−1; finance-cost / inflow at
t−3. Q5-only already beats 0.663; Q6-only does not. The 278-col tree’s fold-4
0.556 is the same short-DSO churn wound — more columns do not fix it. D stayed
out of X.
