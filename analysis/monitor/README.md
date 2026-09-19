# analysis/monitor/ — plan steps 3 and 4

Behaviour clusters, robust control charts, alerts (including "top customer went quiet") and the score forecast, all on top of the explainable score in `product/score/`. **Monitoring claim only**: the alerts say a company moved away from its own normal, with the reason and the amount behind it. They do not say it will fail, and on the accepted outcomes they do not (measured below). Nothing here is fitted on the holdout.

```bash
PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo>
python -m analysis.monitor.fit        # fit on train -> monitor_params.json, y7_rank_model.txt (10 s)
python -m analysis.monitor.evaluate   # evaluation.md, monitor_stats.json, evaluation_outcomes.csv (1 min)
python -m analysis.monitor.forecast   # forecast_params.json, forecast_evaluation.md (6 s)
```

The export (`python -m product.score.export`) applies the fitted parameters to whatever CSV folder it is given and writes the bundle; see `product/score/DATA_CONTRACT.md`. Re-fitting is a deliberate step, not part of a run.

| File | What |
|---|---|
| `control.py` | Robust EWMA + CUSUM on within-series change: baseline = median of the 12 months ending 3 months earlier, scale = 1.4826·MAD floored at the train median scale, `EWMA λ=0.3 L=3`, `CUSUM k=0.5 h=4`, persistence = 3 of the last 4 months signal the same way, `onsets()` keeps the first month of a run. `funnel_chart()` is the group-vs-groups limit. |
| `behaviour.py` | 17 behaviour features per company, size regressed out, winsorised and standardised, k-means, k by silhouette. `Clusterer.fit` (train) / `.assign` (any). |
| `topcustomer.py` | The trigger (last quarter's top customer got no invoice this month, as-of, onset only), open receivable in €, and the night's shallow-LightGBM TURNOVER card, used **only to rank** (`y7_rank_model.txt`); out-of-fold scores for evaluation. |
| `engine.py` | `run_monitor(...)`: charts for company, cluster gap, categories, groups; alerts of five kinds; `vs_cluster` snapshot. Materiality thresholds are constants fixed in advance (score 8, category 10, group 8 points). |
| `routing.py` | Owner (treasurer / CFO / collections) and action per alert from the item that moved most. |
| `fit.py` | Fits on train companies only: clusters, floors, group variance laws, rank-model cut. → `monitor_params.json`. |
| `evaluate.py` | False-alarm rate, lead time, volume, dips, cluster diagnostics, top-customer precision by severity, funnel calibration. → `evaluation.md`, `monitor_stats.json`. |
| `forecast.py` | Naive last value vs exponentially smoothed level, group-fold CV, fan from scaled out-of-fold errors. → `forecast_params.json`, `forecast_evaluation.md`. |
| `y7_alert_eval.py` | The night's alert-grade evaluation of the top-customer card (kept as evidence, not imported). |

## What the numbers say (train companies, holdout untouched; details in `evaluation.md`)

- **Score-fall alerts do not predict the accepted outcomes.** Material falls (≥ 8 points from own baseline, 3 of the last 4 months, plus going dark) are followed by an outcome label within 6 months 29% of the time on average over the eight outcomes, against 31% for an alert on a random month; lift 0.72–1.24 per outcome, several intervals below 1 (fees, payables). The score was built to explain and monitor, and this repeats what `product/score/validation.md` found. Median lead time to the outcome for the alerts that are followed: 2 months. Do not quote these alerts as early warnings of failure.
- **Volume:** 0.55 risk alerts and 0.20 improvement alerts per company-year at the fixed thresholds. 47% of one-month falls of 8 points or more became an alert within 3 months; the rest reverted, which is what the persistence rule is for.
- **Top customer went quiet is the alert with a measured lift.** Onset-only trigger: 56% of flagged company-months lose the customer against 29% base (lift 1.9, 95% CI 48–63%), 22% of labelled rows flagged, recall 42% (every quiet month: 58%, lift 2.0, 38% of rows flagged, recall 76%). Out-of-fold ranking: top decile of onsets 62%; a customer that billed in all 3 of the last 3 months is only 33%, near base, hence `info`. Top-1 share of billing is not a useful severity filter (precision 50–61% across share bins). Still "review exposure and collections", never "revenue at risk". The open receivable in € is missing for a share of events (customers with no unpaid invoice), and the alert says so by leaving it out.
- **Clusters are weak** (silhouette 0.19 at k=4, chosen from 4–8; adjusted Rand between fits on random halves of train 0.78–0.82, so the partition is stable even though the clusters are not sharp). The size signal is gone from them (η² with log inflow 0.007, was 0.23 before regressing size out). Use as peer groups for the "vs cluster" comparison. Suggested labels are generated from the strongest centroid features, not hand-picked stories.
- **Group funnel:** 2.4% of train group-months fall outside the 3-sigma funnel (heavy tails, above the 0.3% of a normal), 0.3% persistently. Groups of fewer than 3 scored companies get no limits (median group has 2).
- **Forecast is a tie.** The smoothed level is worse than the naive last value at 1–2 months (−5.5%, −1.5%), a tie at 3, and +0.4% to +1.0% at 4–6 months (intervals just above zero, small). The rule "ship the smoothed level only if it wins at every horizon with an interval above zero" is not met, so `naive_last` ships, with a fan whose 50% and 80% intervals cover 50% and 80% out-of-fold. This is the first step to cut.

## Guardrails kept

Holdout never used to fit or evaluate anything here (the export scores every company in a folder, holdout included, like any new company). No look-ahead: baselines are lagged, the top customer at month t comes from months t−3..t−1, the trigger uses only invoices issued up to t. The outcome labels are used only to date alerts against, with each outcome's own feature families removed from the score. The shipped alerts use the full score; only the evaluation drops an outcome's own feature families. The night's model is never shown as a probability. No claim on the hidden test.

## Known limits

- Score-fall alerts are descriptive: they follow the score, which is not predictive on the accepted outcomes.
- Cluster membership uses the whole trail, so it is a trait for a comparison, not an as-of signal.
- The evaluation uses the saved percentile reference (fitted on train companies, not on the outcomes), so it is in-sample for the marginal item distributions only. The ranking model is evaluated out-of-fold by group.
- Suppliers: the same alert on suppliers was not tested (counterparty IDs do not map to companies; the trigger would work on AP invoices but nobody can act on an entity that is not scored).
- Group composition changes (a company entering or leaving) can move a group mean; the chart does not adjust for it.
