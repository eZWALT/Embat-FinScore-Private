# Y7 core card — `y7_top1_lost`

Generated 2026-09-19 by `analysis/models/gbm_y7_core.py`.
Global LightGBM on the accepted dip-vs-fall label only (not the inflow sibling).
Holdout 72 never in fit, early-stop, pick, or SHAP. Quote **train group-fold CV**.
Published 278-col claim is **CV 0.663** (not holdout 0.680).

## Setup

| item | choice |
|------|--------|
| Y | `y7_top1_lost` (top AR customer of t−2..t issues 0 in t+1..t+3). Q4 dip vs fall, not bankruptcy. |
| Train | 7464 labeled / 2149 pos / rate **0.2879** (7464/21157 = 35.3% of train company-months) |
| Split | 5 `group_folds`, seed `20260918`. Holdout 72 excluded from every fit and val fold. |
| X | 6–8 E+F columns. **Never D.** Never `n_banking` / `group_size`. Never contemporaneous `e_ar_issued`. |
| DSO | fixed 24-month clip (constant, not a fit) |
| A/B/C trees | same as published 0.663 (`LGB_BASE` 400 + early stopping 40) |
| Shallow | 50 trees, `max_depth=3`, `num_leaves=8`, no early stop (Y3 diagnostic) |
| A_n50 | 50 trees, no depth cap (explain_y7 final-fit trees) |

**A (7 cols):** `e_dso_proxy`, `e_ar_issued_lag1`, `e_credit_note_ratio`, `e_credit_note_ratio_lag1`, `e_delay_coll`, `e_dso_proxy_lag1`, `f_fc_r_lag3`.

**B (6 cols):** A minus `e_dso_proxy_lag1` (SHAP + vs univ − 0.506).

**C (8 cols):** A plus `log1p(e_ar_issued)` volume proxy. Raw euro level still out.

## Quote (train CV)

| spec | mode | n_x | CV | sd | vs 0.663 | vs dummy | vs single on this X | collapsed |
|------|------|----:|---:|---:|----------|----------|---------------------|-----------|
| **A** | 400+ES | 7 | 0.7077 | 0.095 | +0.045 | yes | yes (`e_ar_issued_lag1` 0.630) | **yes** (5/5/42/5/6) |
| **A_shallow** | 50/d3 | 7 | **0.7111** | 0.084 | +0.048 | yes | yes | no |
| **A_n50** | 50 | 7 | **0.7124** | 0.090 | +0.049 | yes | yes | no |
| **B** | 400+ES | 6 | 0.7052 | 0.081 | +0.042 | yes | yes (0.630) | **yes** (1/11/20/4/6) |
| **B_shallow** | 50/d3 | 6 | **0.7121** | 0.086 | +0.049 | yes | yes | no |
| B_n50 | 50 | 6 | 0.7106 | 0.079 | +0.048 | yes | yes | no |
| C | 400+ES | 8 | 0.7135 | 0.084 | +0.051 | yes | yes (`e_ar_issued_log1p` 0.663) | **yes** |
| C_shallow | 50/d3 | 8 | 0.7114 | 0.089 | +0.048 | yes | yes | no |
| D_shallow (− `f_fc_r_lag3`) | 50/d3 | 6 | 0.6989 | 0.080 | +0.036 | yes | yes | no |
| E_shallow (− now credit-note) | 50/d3 | 6 | 0.7041 | 0.085 | +0.041 | yes | yes | no |
| F_shallow (+ overdue) | 50/d3 | 8 | 0.7088 | 0.085 | +0.046 | yes | yes | no |
| G_shallow (+ delay lag1) | 50/d3 | 8 | 0.7088 | 0.087 | +0.046 | yes | yes | no |
| MIN (DSO + issued lag1) | 50/d3 | 2 | 0.6357 | 0.084 | −0.027 | yes | ~tie 0.630 | no |
| Q5 only | 50/d3 | 3 | 0.6737 | 0.080 | +0.011 | yes | yes | no |
| Q6 only | 50/d3 | 4 | 0.6440 | 0.074 | −0.019 | yes | yes 0.630 | no |
| B5 (− DSO lag1 − fc lag) | 50/d3 | 5 | 0.6971 | 0.082 | +0.034 | yes | yes | no |
| NOCLIP | 50/d3 | 7 | 0.7125 | 0.083 | +0.050 | yes | yes | no |
| C12 (DSO clip 12m) | 50/d3 | 6 | 0.7087 | 0.086 | +0.046 | yes | yes | no |
| IX (DSO×credit-note) | 50/d3 | 7 | 0.7118 | 0.083 | +0.049 | yes | yes | no |
| SLOPE (issued lag1/lag3) | 50/d3 | 7 | 0.7146 | 0.088 | +0.052 | yes | yes | no |
| VOL (B + issued-lag CV) | 50/d3 | 7 | 0.7289 | 0.081 | +0.066 | yes | yes 0.622 | no |
| VOL 400+ES | 400+ES | 7 | 0.7200 | 0.088 | +0.057 | yes | yes | borderline (6/9/26/7/11) |
| **TURNOVER** (no DSO) | 50/d3 | 5 | **0.7200** | **0.034** | +0.057 | yes | yes 0.622 | no |
| TURNOVER 400+ES | 400+ES | 5 | 0.7118 | 0.027 | +0.049 | yes | yes | **yes** (2/9/21/3/11) |
| TURNOVER_n50 | 50 | 5 | 0.7086 | 0.025 | +0.046 | yes | yes | no |
| **TURNDELAY** (no DSO + delay) | 50/d3 | 6 | **0.7293** | 0.041 | +0.066 | yes | yes | no |
| TURN2 (issued lag1 + CV) | 50/d3 | 2 | 0.6773 | 0.036 | +0.014 | yes | yes | no |
| TURN_NOFC | 50/d3 | 4 | 0.7141 | 0.030 | +0.051 | yes | yes | no |
| TURN_NOCN | 50/d3 | 4 | 0.7069 | 0.034 | +0.044 | yes | yes | no |
| TURNCV (issued-lag CV only) | 50/d3 | 1 | 0.6510 | 0.034 | −0.012 | yes | 0.627 | no |
| VOL13 (B + lag1/lag3 CV) | 50/d3 | 7 | 0.7162 | 0.090 | +0.053 | yes | yes | no |
| TURNLOG (log1p issued CV) | 50/d3 | 5 | 0.7230 | 0.036 | +0.060 | yes | yes | no |
| TURNOV (TURNOVER + overdue) | 50/d3 | 6 | 0.7254 | 0.042 | +0.062 | yes | yes | no |
| TURNFULL (no DSO + delay + overdue) | 50/d3 | 7 | 0.7301 | 0.047 | +0.067 | yes | yes | no |
| TURNCLIP (CV clipped at 2) | 50/d3 | 5 | 0.7200 | 0.034 | +0.057 | yes | yes | no |
| TURNSLOPE (+ lag1/lag3) | 50/d3 | 6 | 0.7225 | 0.037 | +0.059 | yes | yes | no |
| TURNZERO (+ zero-issue count) | 50/d3 | 6 | 0.7205 | 0.038 | +0.058 | yes | yes | no |
| TURNFC1 (+ f_fc_r_lag1) | 50/d3 | 6 | 0.7194 | 0.036 | +0.056 | yes | yes | no |
| TURNFC0 (+ f_fc_r now) | 50/d3 | 6 | 0.7210 | 0.040 | +0.058 | yes | yes | no |
| TURNMEAN (3m issued mean, no lag1) | 50/d3 | 5 | 0.7125 | 0.038 | +0.050 | yes | yes 0.627 | no |
| TURNL1LOG (log1p lag1) | 50/d3 | 5 | 0.7200 | 0.034 | +0.057 | yes | yes | no |
| TURNIX (CN × issued CV) | 50/d3 | 6 | 0.7197 | 0.036 | +0.057 | yes | yes | no |
| TURN3 (no last-month issued) | 50/d3 | 4 | 0.6929 | 0.032 | +0.030 | yes | yes | no |
| TURNL13 (+ issued lag3) | 50/d3 | 6 | 0.7198 | 0.038 | +0.057 | yes | yes | no |
| TURNPEND (+ pending share) | 50/d3 | 6 | 0.7184 | 0.040 | +0.055 | yes | yes | no |
| TURNFX (+ FX share) | 50/d3 | 6 | 0.7151 | 0.043 | +0.052 | yes | yes | no |
| TURNDPO (+ DPO) | 50/d3 | 6 | 0.7230 | 0.040 | +0.060 | yes | yes | no |
| TURNDS (+ f_ds_r_lag3) | 50/d3 | 6 | 0.7201 | 0.039 | +0.057 | yes | yes | no |
| TURNOV30 (+ overdue 30) | 50/d3 | 6 | 0.7164 | 0.046 | +0.053 | yes | yes | no |
| TURNPAID (+ AP delay) | 50/d3 | 6 | 0.7210 | 0.044 | +0.058 | yes | yes | no |
| TURNDSSWAP (ds_r instead of fc_r) | 50/d3 | 5 | 0.7120 | 0.032 | +0.049 | yes | yes | no |
| TURNCNCV (+ CN lag CV) | 50/d3 | 6 | 0.7205 | 0.036 | +0.057 | yes | yes | no |
| BHI (B, DSO gated ≥1m) | 50/d3 | 6 | 0.6954 | 0.039 | +0.032 | yes | yes | no |
| TURNHI (TURNOVER + gated DSO) | 50/d3 | 6 | 0.7180 | 0.041 | +0.055 | yes | yes | no |
| published 278-col | 400+ES → 50 | 278 | 0.663 | — | — | yes | `e_ar_issued` 0.598 hold / 0.656 train | — |

A folds (ES trees): 0.739 (5), 0.796 (5), 0.742 (42), 0.715 (5), 0.546 (6).
B_shallow folds: 0.741, 0.802, 0.739, 0.707, 0.571.
VOL_shallow folds: 0.766, 0.813, 0.745, 0.725, 0.597.
TURNOVER_shallow folds: 0.745, 0.765, 0.713, 0.698, **0.680**.
TURNDELAY_shallow folds: 0.752, 0.785, 0.721, 0.711, **0.679**.
Published 278-col folds: 0.556 / 0.781 / 0.690 / 0.659 / 0.556.

Holdout is a check only (LOW_POWER, 122 events): A_shallow 0.701, B_shallow 0.706, VOL 0.725, TURNOVER 0.724. Do not quote these.

## Leak / size (train labeled)

Y vs `log1p(a_in3)` AUROC = **0.465** (published 0.465; gate < 0.60).
Worst |ρ| vs `d_cust_top1` / `d_cust_hhi`: `e_credit_note_ratio_lag1` −0.299 / −0.306 (fail ≥ 0.80). **PASS.**
Worst |ρ| vs `a_in3`: `e_ar_issued_lag1` 0.464 (fail > 0.85). **PASS.**
`e_ar_issued_lag_cv` is cleaner still: ρ_D ≈ −0.02, ρ_size ≈ 0.017.

## KEEP / PARK / CLOSE

Rule: KEEP if a claimable 400+ES spec CV ≥ 0.663 (eps 0.002) with fewer columns **and** early-stop does not collapse to 1–5 trees. CLOSE if within 0.02. PARK if lose by > 0.02.

- A/B/C early-stop all print 0.70–0.71, but trees collapse (1–9 on most folds). **Do not KEEP those numbers** (Y3 lesson).
- Honest 50-tree / depth-3 diagnostics **hold**: B_shallow **0.712 / n_x=6**, A_shallow **0.711 / n_x=7**, vs 0.663 / 278.
- Dropping the sign-flip lag (B) does not cost. Dropping `f_fc_r_lag3` costs ~0.013. Two-col MIN **PARKs** (0.636, lose 0.027).
- Q5-only 0.674 still beats 0.663. Q6-only 0.644 is inside the 0.02 band but below the claim.

**Decision: CLOSE — do not PARK. Do not KEEP A/B/C on the 400+ES spec.**
Prefer **B_shallow** if a SHAP card must be named (6 cols, ties A, drops the sign-flip). The number that actually holds on that card is **0.712 / n_x=6**.
The stable next card is **TURNOVER / TURNDELAY** (no DSO): 0.720–0.729, fold 4 **0.680**, sd 0.03–0.04. That is not the KEEP gate for the assigned SHAP stems. Later add-ons on TURNOVER (clip, slope, zero-count, FX, pending, DPO, overdue-30, AP delay, extra finance lags, issued mean instead of lag1) do not lift fold 4. Dropping last-month issued (TURN3) costs ~0.027.

## Train SHAP on B (6-col, sample 4000, holdout out)

| rank | feature | sign | mean\|SHAP\| | brief |
| --- | --- | :---: | ---: | --- |
| 1 | `e_dso_proxy` | + | 0.341 | Q5 why (stretched AR) |
| 2 | `e_ar_issued_lag1` | − | 0.238 | Q6 lead (thin issuance last month) |
| 3 | `f_fc_r_lag3` | + | 0.197 | Q6 lead (finance-cost / inflow at t−3) |
| 4 | `e_credit_note_ratio_lag1` | + | 0.158 | Q6 lead |
| 5 | `e_credit_note_ratio` | + | 0.137 | Q5 why |
| 6 | `e_delay_coll` | + | 0.135 | Q5 why (slow collections) |

DSO univariate is ~0.51 (near chance) but dominates gain/SHAP — it works in interaction, not alone. That is why MIN (DSO + issued lag1) dies.

### Train SHAP on TURNOVER (5-col, no DSO)

| rank | feature | sign | mean\|SHAP\| | brief |
| --- | --- | :---: | ---: | --- |
| 1 | `e_ar_issued_lag_cv` | + | 0.345 | Q6 lead (issuance volatility) |
| 2 | `e_ar_issued_lag1` | − | 0.295 | Q6 lead (thin issuance last month) |
| 3 | `f_fc_r_lag3` | + | 0.196 | Q6 lead |
| 4 | `e_credit_note_ratio` | + | 0.161 | Q5 why |
| 5 | `e_credit_note_ratio_lag1` | + | 0.136 | Q6 lead |

Signs are stable. This is the fold-4-safe story: jumpy / thinning issuance, not stretched DSO.

## Fold 4 (the 0.55 fold, same as the 278-col run)

Fold rates: 0.159 / 0.303 / 0.379 / 0.217 / **0.382**. Fold 4 DSO median is the lowest (1.00 vs 1.5–2.5).

Two groups dominate the miss:

| group | n | pos | rate | OOF AUROC | mean pred |
| --- | ---: | ---: | ---: | ---: | ---: |
| GROUP_0222 | 336 | 241 | 0.717 | 0.563 | 0.299 |
| GROUP_0108 | 204 | 130 | 0.637 | 0.558 | 0.443 |

Problem groups vs rest (train labeled): rate 0.687 vs 0.257; **DSO median 0.18 vs 1.97**; issued lag1 28k vs 67k; credit-note 0.10 vs 0.00. They are **short-DSO, high-turnover** names. The card’s main lever (high DSO → top-1 gone) is inverted, so the model under-scores them. The 278-col tree fails the same fold (0.556). More E columns do not fix it.

Issued-lag CV is higher in those groups (0.51 vs 0.40). Adding it on top of DSO (VOL) only lifts fold 4 to 0.597. **Dropping DSO** (TURNOVER) lifts fold 4 to **0.680** and GROUP_0222 from 0.563 / pred 0.30 to 0.650 / pred 0.47. DSO is the majority-group lever and the fold-4 poison. The 278-col tree has the same wound because SHAP ranked DSO #1.

B_shallow OOF AUROC by clipped-DSO quintile (NaN DSO dropped):

| DSO quintile | n | rate | AUROC |
| --- | ---: | ---: | ---: |
| ≤ 0.54 (med 0.06) | 1331 | 0.255 | **0.410** |
| 0.54–1.16 | 1330 | 0.250 | 0.691 |
| 1.16–2.45 | 1330 | 0.221 | 0.659 |
| 2.45–6.74 | 1330 | 0.198 | 0.628 |
| 6.74–24 | 1330 | 0.296 | 0.660 |

The SHAP card is **worse than chance** on the short-DSO fifth. That is not a small-sample wobble.

TURNOVER (no DSO) on the same quintiles: 0.706 / 0.698 / 0.661 / 0.655 / 0.679. The short-DSO fifth flips from 0.410 to **0.706**. DSO was the poison, not the missing column.
TURNDELAY (no DSO + collection delay): 0.666 / 0.699 / 0.688 / 0.666 / 0.675. Mean CV is a bit higher (0.729) but Q1 is weaker than TURNOVER. Prefer TURNOVER if fold 4 / short-DSO is the wound; TURNDELAY if you want the extra 0.009 mean.

OOF score vs clipped DSO: B_shallow ρ=**0.261** (the score tracks DSO); TURNOVER ρ=**−0.051** (uncorrelated). Fold-wise B vs TURNOVER: B wins fold 1 (0.802 vs 0.765) where DSO helps; TURNOVER wins fold 4 (0.680 vs 0.571) where DSO poisons. Mean and sd favor TURNOVER.

Fold-4 permutation (train on folds 0–3, shuffle one col on fold 4):

| card | base | biggest drop |
| --- | ---: | --- |
| B_shallow | 0.571 | `e_credit_note_ratio_lag1` +0.021; DSO only +0.007 |
| TURNOVER | 0.680 | `e_ar_issued_lag1` +0.063; `e_ar_issued_lag_cv` +0.052 |

On the hard fold, global SHAP’s #1 (`e_dso_proxy`) is almost unused. Issued lag + issued-lag CV are the fold-4 model.

## What this means (questions 4–6)

Question 4 is the label. Question 5: stretched AR (clipped DSO, slow collections, credit notes) plus thin last-month issuance raise P(top-1 gone next quarter). Question 6: the same book is visible at t−1 (issued, credit-note) and finance-cost / inflow at t−3. Q5-only already beats 0.663; Q6-only does not. Lead time is real but secondary. Family D stayed out.

## What failed

- 400+ES on a 6–8 col X collapses to 1–6 trees (same Y3 failure mode).
- `e_dso_proxy_lag1` sign flip is real and unused (B = A).
- Euro issued level is not needed; `log1p(issued)` becomes a 0.663 single and does not lift the tree.
- Optional 8ths (`e_ar_overdue`, `e_delay_coll_lag1`) do not help.
- Two-col floor PARKs. Fold 4 is a different mechanism (short-DSO churn), not missing SHAP stems.
- Gating DSO at 1 month (BHI) lifts fold 4 vs B (0.571 → 0.641) but loses mean (0.712 → 0.695). Putting gated DSO on TURNOVER (TURNHI) still hurts fold 4 (0.680 → 0.664). Do not put DSO back.

## Next idea

Drop `e_dso_proxy` and add issued-lag CV (`std(issued_{t-1,t-2,t-3}) / mean`). TURNOVER 0.720 / n_x=5 (sd 0.034) or TURNDELAY 0.729 / n_x=6; fold 4 0.680. |ρ| vs D and size on the CV feature ≈ 0. Do not put family D (`d_cust_lost` / HHI) in X. Do not KEEP the collapsed 400+ES SHAP-card numbers.

## Re-run

```bash
/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_y7_core
```
