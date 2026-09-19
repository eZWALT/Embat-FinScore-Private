# 2026-09-19-1130 — which night-explored variables go into the FICO-like score

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-19 ~11:30 CEST, after pulling `5d89c51` (METHOD.md, signal cards, `night-closed` entry)
- **Sources read:** `overnight/dashboards/METHOD.md`, `MORNING_REPORT.md` §5–7, `data/feature_store/feature_dictionary.md` (families B, E). Docs only; no column was run on data by me.

## Two conclusions about how to use the night's work

1. **Do not follow the report's §9 recommendation** (score = social security, payroll, movement days + lags). Their signs are all *negative* (quiet stressed months "recover"), so a monotone card on them rewards going dark. It is also built on Y3, which is ~70% an outflow collapse (`2026-09-19-1015-y3-recovery-mechanical.md`). At most these become an "operating normally" guard with the sign a priori positive; not validated by the night.
2. **The night's "set aside" list is largely our candidate pool.** Its gates (ICC > 0.90 = trait, beat size, leftover after the activity clock) answer "what predicts a change in the next 6 months". A FICO-like level score is mostly "who is this company", so stable traits (volatility, concentration, lateness style) are legitimate there. The same traits are useless for the monitor, which should chart within-company change.

## Candidate map (FICO category → variable, direction a priori, night evidence)

| Category | Variable | Direction | Night evidence / caveat |
|---|---|---|---|
| Payment history | `e_delay_coll` (customers' lateness, 3-mo, amount-weighted) | lower better | footnote signal, ICC 0.92, leftover after DSO 0.581; null first 6 months (left truncation) |
| Payment history | `e_delay_paid` (own lateness to suppliers) | lower better | same construction; Y5 outcomes accepted (`y5_ap_od30_ownp80`, `y5_ar_od30_sust`), no model beat a single column |
| Payment history | `e_ar_overdue_30`, `e_ap_overdue_30` (share of open > 30 days past due) | lower better | Banque de France: only > 30 d matters; `e_ap_overdue` fails beat-size but that is a prediction gate |
| Amounts owed | `b_runway` (liquidity / mean monthly outflow, clip −6..24) | higher better | the "health photograph": median 1.08 months, ρ 0.966 vs snapshot; reconstruction is an identity but needs product rows (29 products with tx have none) |
| Amounts owed | `b_neg_liq_3`, `b_neg_episodes`, `b_min_liq_3` | lower / lower / higher | negative-cash outcome `y2_neg_2of3` accepted; use as validation target, not as X for it |
| Amounts owed | `f_ds_r` (debt service / inflows), `f_fc_r` (fees + interest / inflows) | lower better | nothing after the clock as *predictors*; fine as level items. `y4_ds_r_double`, `y9_*` accepted as validation targets |
| Length / stability | months of trail, share of months active | higher better | not tested overnight; compute from grid |
| Length / stability | `a_out_vol` (outflow volatility) | lower better | 74% between-company variance; Lundmark 2020 supports "trait" |
| Length / stability | `c_n_days_with_tx`, `c_ss_month`, `c_salary_month`, `c_recency_days` | activity higher better | guard against going dark; night evidence points the other way for Y3 (see above) |
| New credit | `f_ds_r` change (debt service doubling), new facility after a cash dip | lower better | thin: `f_n_types`/`f_n_facilities`/`g_n_accounts` are rise-only connection artefacts (drop); utilisation exists for 1.6% of rows (hole); NSF/overdraft is a data hole. Consider shrinking this category's weight |
| Mix | `d_cust_hhi` / `d_cust_top1`, tail only (> 0.975 one buyer) | lower better | 22% vs 12% debt-service shock; body of the distribution is noise. Use a capped/threshold term |
| Mix | `d_supp_hhi` | do not sign | night finds concentration *protective* (2.7% vs 8.6% late risk), contradicting Pérez-Salazar 2026; leave out or neutral |
| Mix | `e_credit_note_ratio` | lower better (a priori) | in the TURNOVER card; no paper measures it (literature gap); mark as experimental |

## Customer-loss reading (Y7 / TURNOVER) — belongs to the monitor and the alert reasons more than to the static score

- `e_issued_top1` (billing this month to last quarter's largest customer): leftover 0.789 after last month's issuance; event rate 58% when it is 0 vs 8% otherwise. It is nearly the label one month early, so it is an alert reason ("top customer went quiet"), not a predictor to score against `y7_top1_lost`.
- Card: `e_ar_issued_lag1` (size ρ 0.46, borderline), `e_ar_issued_lag_cv` (cleanest, ρ −0.02 vs concentration, 0.02 vs size), `e_credit_note_ratio`, `f_fc_r_lag3`; 0.720 ± 0.034 by group-fold CV. DSO is explicitly off the card (0.410 on the fastest-collecting fifth).
- Only for the ~744 of 1,214 train companies with invoices; the score needs the reweight + confidence flag from the plan.

*Update 13:15:* measured in `2026-09-19-1300-y7-alert-grade-eval.md`; decided to include as an alert with the rule as trigger. `e_issued_top1` figures above are the night's; use the 1300 numbers when quoting.

## Validation targets we can reuse (accepted in `data/feature_store/y_acceptance.csv`)

`y2_neg_2of3`, `y4_ds_r_double`, `y5_ap_od30_ownp80`, `y5_ar_od30_sust`, `y7_top1_lost`, `y7_top1_lost_inflow`, `y9_fee_r_ownp80`, `y9_fee_spike` (forbid the listed X families). **Not** `y3_recover_cash_6m` until fixed. `y8_*` labels accepted, models parked.

## Holes to state, not fill

Credit-line utilisation (1.6% coverage), NSF / bounced payments (no token), interco (`d_interco_share` all null). No 82 → 68 explanation exists without cash columns; here it comes from charting the score.

## Still unknown

- Whether the feature store parquet builds locally (not committed; needs `python -m analysis.features.build_feature_store`).
- Actual score distribution, correlation between chosen items and category balance; untested.
