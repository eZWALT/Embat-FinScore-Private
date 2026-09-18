# 2026-09-18-2210 — Python port of the score pipeline + first validation findings

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-18

## What changed

- `analysis/score_pipeline.py`: Python port of the R notebook logic (features, percentile scoring, pillars, trajectory, alerts). Runs end to end in ~5 s on `data/embat.duckdb`.
- `analysis/validate_pipeline.py`: runs it and prints diagnostics. `cd analysis && python validate_pipeline.py`.
- `requirements.txt` now has pandas + numpy. R is still not installed on the authoring machine (installer `R-4.6.1-win.exe` sits in Downloads), so the `.Rmd` files are still unexecuted.

## Findings (facts from running the code)

- **Invoice sign confirmed:** of 445,775 paid invoices, 93,611 match a same-company transaction of the same date and *same* sign/amount vs 1,628 with the opposite sign. `amount > 0` = issued/collected. (A cross-company Spearman check is misleading: size dominates.)
- **Artifact fixed:** payment-delay signals looked like they worsened ~11 pts over time for everyone: left truncation (invoices issued before 2024-09 and paid late are missing early on). `delay_*` are now masked for the first months (both Python and R).
- **Pillars are almost uncorrelated** (company-month r between pillars: -0.10..0.15, except caja-estabilidad 0.36 which share a source). There is no strong single latent "health" across data sources; the composite is a multi-dimensional profile.
- **Persistence t -> t+6:** level signals persist (r 0.4-0.7: runway, neg_liq, concentration, fin_cost_r...), change signals do not (`d_runway` -0.04, `growth` 0.00, `coverage` -0.01). `coverage` (capped ratio) is essentially noise; `net_margin` is the cleaner version.
- **Naive proxy events fail:** "3-month inflow falls >=40% within 6 months" happens in 33% of company-months (boom 34%), driven by mean reversion in lumpy monthly cash. AUC of score/trend/growth vs this event is 0.35-0.48 (below 0.5), and alerts give no lift (crash rate after alert 0.199 vs 0.217 base). Company-level changes in inflow are not persistent (corr of H1 vs H2 change = 0.005).
- Group-fold stability (Spearman ~0.9999) is real but trivial: the percentile reference is stable, this says nothing about predictive validity.
- 174 of 1,286 companies have zero inflow in the last 3 months; 61 have their last transaction before 2026-06-01.

## Decisions / implications

- The current proxy events in the R notebook (section 7) and the backtest that depends on them are not evidence of predictive value; they need a better outcome definition before being shown to judges.
- Candidate fixes (not done): drop or replace `coverage`; use only persistent signals in the trajectory; define sustained-change outcomes (6-month windows, both sides) and test on pillar-level or cross-source targets (e.g. cash-based pillar at t predicting invoice-behavior pillar at t+6).

## Still unknown

- What the organizers' hidden "truth" is (latent health trajectory?). The brief's examples (45 -> 65, 82 -> 68) suggest a latent trajectory exists, but we have not found a signal that tracks it.
