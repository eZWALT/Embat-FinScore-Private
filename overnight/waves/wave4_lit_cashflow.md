# Wave 4 — cash-flow literature (as-of morning)

Owner: cash-flow underwriting + FICO-like SME / bank-statement scorecards.
Deliverable: `analysis/outputs/lit_cashflow.md` + `lit_cashflow_cluster.csv`.
Sibling invoice / concentration / DSO → `analysis/outputs/lit_invoice.md`. Hidden 72 never fit.

## Cluster

11 rows (10 ultra + Altman 1968 contrast). All URLs fetched. No invented cites.

Starters **verified**: FinRegLab 2025, Yao 2017, Ng 2025 **v4**, Pedregal 2024, Siddiqi 2017 (SAS 2nd-ed excerpt). FinRegLab 2019 Participant #6 ¶65 re-fetched: bureau **0.720** / CF **0.675–0.688** / combo **0.758**.

Beyond: Norden 2010, Mester 2007, Farrell–Wheat 2016, Farrell–Wheat–Grandet 2019, FinRegLab 2019 Empirical Findings.

Companions (not extra cluster rows): Hair NBER w33367, Jiménez 2009, Farrell–Wheat 2017 payroll, Sufi 2009, Ciampi 2018, Farrell–Wheat–Mac 2018/2020, Nemoto ADBI WP 857, Formisano/Modina 2016/2023, Ehling–Haushalter 2014, La Rocca 2019, Lundmark/Coad/Frankish/Storey 2020 ET&P (`a_out_vol` is a trait), Østergaard/Sasson/Sørensen 2020 (cash-poor amplify bank shocks).

**Ng correction:** live **v4** (*AI-BAAM*) headlines validation blended **0.806**. 5-fold CV: application **0.672** / statement **0.821** / blended **0.850**. Validation split: **0.647 / 0.763 / 0.806**. Top IV = log growth of average balance **0.484**. Do not collapse val and CV.

## Mapping headlines

- Q1 last-value `b_runway` **SAME** Farrell 2016 (p50 1.079m ≈ 27 days, not the 2020 metro 15). **Never B as Y2/Y3 X.**
- Quiet-stressed recover **NEW** vs default papers (they want bigger credits); **SAME** JPMC payroll (27→18 days). `a_in3` leftover 0.521 dies. `a_op_out` leftover 0.586 is SIZE+twin — CLOSE.
- Leftover-after-days is **NEW**. Literature quotes raw AUROC. IV≥0.5 is the published cousin. Residual is a referee, not a new X.
- Norden AMPLI = HIGH−LOW of **balances** (B-forbidden). Without a credit line, ΔCUMOVER is the only activity predictor → Wave D, not utilisation.
- `created_at` / `g_n_accounts` leftover 0.428 are connection clocks, not Hair’s <5-year firm age.
- Y10 utilisation **impossible** (1.6%). Pedregal is inventory stockouts — PARK stands.
- Siddiqi 8–15: do **not** refill the historical 15-col card with DROPped SHAP names. Reasons = leftover KEEP only (SS 0.635 / salary 0.603 / days 0.711).

## Do not do

- Grow the 15-col card. Rewrite `gbm_core.py`. Merge parquet. `build_targets`. Fit hidden 72.
- Touch leftover QA files (`op_out_qa`, `fin_cost_qa`, `ap_overdue_qa`, `n_accounts_qa`, `util_snap_qa`, …).
- Put B on Y2/Y3 X. Quote `a_out_vol` 0.722 as the engine. Revive `c_gap_sd`. Invent `y_nsf` / utilisation Y.
- Invoice papers except one-line `see lit_invoice`. Altman-family PD as the cluster.

## Three pitches the parent should absorb (in order)

1. **Wave C — Siddiqi reasons on leftover-KEEP stems** (`analysis/outputs/y3_reasons.md`). Read-only. Named levers = `c_ss_month` / `c_salary_month` / `c_n_days_with_tx` ± lags only. Do not quote `a_n_tx`, `a_op_in`, `a_transfer`, `e_dso_proxy`, `f_ds_r`. Do not pad to 15.
2. **Wave B — Norden lead without utilisation** (`days_delta_qa`). 3-/6-month **change** in days leftover after days-**level** on ≥18m books. KEEP only if leftover ≥0.60 AND beat-size ≥0.02 AND not a twin AND not SIZE. Not AMPLI (B). Hidden 72 stays 1-month.
3. **Wave A — last-value vs Hair’s 3-month mean runway** (`runway_window_qa`). Q1 photograph only. Do not edit `liquidity.py`. If leftover after last-value dies (<0.55), last-value KEEP stays. Never engine X.

Wave D (`nsf_count_qa`) is the fourth seat: NSF/overdraft **token count** leftover after days. Dictionary has no NSF token. Do not reconstruct neg-days from B (Y2 lock). Do not score vs Y9.

## Night numbers this wave must not move

Y3 0.762 / 0.752. days 0.711. size 0.617. `c_ss_month` leftover 0.635. `c_salary_month` 0.603. `a_in3` leftover 0.521 DROP. `a_out_vol` 0.722 trait (demean 0.549). `g_n_accounts` leftover 0.428. `f_util_snapshot` 1.6%. Javier 14: 11 SAME / 2 CLOSE / 1 DRIFT (vol).
