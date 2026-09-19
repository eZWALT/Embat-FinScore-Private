# Wave 4 — Y3 Siddiqi / CFPB reasons (leftover-KEEP only)

Owner: cash-flow card reasons. Deliverable: `analysis/outputs/y3_reasons.md`.
Sibling invoice reasons stay on TURNOVER / `lit_invoice`. Hidden 72 never fit.

## What was scored

Named levers only:

| stem | leftover / bar | Q |
|------|----------------|---|
| `c_ss_month` | leftover after days **0.635** | Q3 / Q5 |
| `c_salary_month` | leftover after days **0.603** | Q3 / Q5 |
| `c_n_days_with_tx` | bar **0.711** | Q2 / Q3 |
| `c_n_days_with_tx_lag1` | **q6_keep 0.684** | Q6 |
| `c_ss_month_lag1` | leftover after days_lag1 **0.631** | Q6 |
| `c_salary_month_lag1` | leftover after days_lag1 **0.606** | Q6 footnote |

All signs − (quiet-stressed recover). Replica matched the night leftovers.
`issued_lag1` 0.626 is Y7, not a Y3 reason.

## Will not say

`a_n_tx` (perm 0.030 / leftover 0.538), `a_op_in`, `a_transfer`±lags,
`e_dso_proxy`, `f_ds_r`±lag1, `c_gap_sd`, `h_n_siblings_active_lag3`,
`a_out_vol` 0.722, any `b_*`, connection clocks, utilisation.
Do not pad the historical 15-col 0.752 quote with DROPped SHAP names.

## Extras the parent should absorb

- **Folds:** SS leftover 0.646 0.582 0.698 0.617 0.630 (min 0.582). Salary
  leftover 0.550 0.573 0.708 0.618 0.567 — fold 0 sits on the 0.55 line;
  KEEP the mean 0.603.
- **Dark vs ERP:** dark salary leftover **0.547 dies**; dark SS 0.574 lives
  thin; ERP SS 0.677 / salary 0.634. Dark sentence = days + SS.
- **Books:** 355/402 Y3 positives are company-long (≥18). Company-short
  recover is LOW_POWER (13 pos). so-far≥18 labels empty (need t+1..t+6).
- **Tercile:** salary leftover dies on size T1 (0.399). Smallest stressed
  firms: days + SS only.

## Do not do

- Rewrite `y3_importances.md`, `explain_y3.py`, `shap_y3.md`, `gbm_core.py`.
- Touch leftover QA files. Write a 0–100. Put B on Y3 X. Quote vol 0.722
  as the engine. Grow TURNOVER. Fit hidden 72. Merge parquet. `build_targets`.

## Night numbers this wave must not move

Y3 0.762 / 0.752. days 0.711. size 0.617. `c_ss_month` leftover 0.635.
`c_salary_month` leftover 0.603. q6_keep = issued_lag1 / days_lag1 / ss_lag1.
`a_in3` leftover 0.521 DROP. `a_out_vol` 0.722 trait.
