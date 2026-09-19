# Wave 3 — SHAP y7_top1_lost

Train-only TreeExplainer on the accepted dip-vs-fall label. Holdout never
entered fit, early stopping, or the SHAP sample. Quote CV **0.663**, not
holdout 0.680.

## Files written

- `analysis/models/explain_y7.py`
- `analysis/outputs/shap_y7.md`
- `analysis/outputs/shap_y7_meanabs.csv`
- `analysis/outputs/shap_y7_summary_bar.png`
- `analysis/outputs/shap_y7_beeswarm.png`

## Columns

- Y: `y7_top1_lost` (top AR counterparty of t−2..t issues 0 in t+1..t+3).
- X: 278 numeric columns, families **A+B+C+E+F+G+H**, lags 1 and 3, meta
  `group_size` + `n_banking`. **Never D** (HHI / top-1 are rebuilt in Y7).
- Store: `data/feature_store/monthly.parquet` (22230×118) + `targets.parquet`.

## Coverage on train

- Train labeled: **7464** company-months, **2149** positives, rate **0.2879**.
- SHAP sample: **4000** (seed 20260918).
- Holdout labeled: 391 unused (122 events; LOW_POWER).
- This-run 5-fold CV AUROC **0.663** (folds 0.556 / 0.781 / 0.690 / 0.659 / 0.556),
  final trees **50**. Matches the bake-off claim.

## What failed

- Family B is allowed and in X, but not a top-10 name. Cash-shape residual
  is 2.9% of mean |SHAP| — liquidity does not carry the dip-vs-fall story.
- Static meta (`n_banking`, `group_size`) entered the top 10. Not a lever.
- One sign flip: `e_dso_proxy_lag1` SHAP + vs univariate − 0.506 (near chance).
- CV fold 4 is 0.556; the 0.663 mean is the claim, not every fold.

## Next idea

- Drop static meta and the euro-level `e_ar_issued` stem; retrain a 6–8
  feature E+lag card (DSO, credit-note ratio, collection delay, lag1 issued,
  `f_fc_r_lag3`) vs the 278-col tree.
- Do not chase holdout 0.680.

## Story (questions 5–6)

Question 4 is the label. Question 5: stretched AR (high DSO, slow collections,
credit notes) and thin current issuance raise P(top-1 gone next quarter).
Question 6: the same invoice book is already visible at t−1
(`e_ar_issued_lag1`, `e_dso_proxy_lag1`, `e_credit_note_ratio_lag1`) plus
finance-cost / inflow at t−3. Lead-time / turning is **52%** of mean |SHAP|.
Family E is **56%**. Worst |ρ| vs `d_cust_top1` is −0.30; D stayed out of X.
