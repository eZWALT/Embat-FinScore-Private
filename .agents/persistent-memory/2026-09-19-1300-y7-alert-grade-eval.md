# 2026-09-19-1300 — top-customer-lost (Y7): alert-grade numbers

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-19 ~13:00 CEST
- **Script:** `analysis/monitor/y7_alert_eval.py` (`PYTHONUTF8=1 PYTHONPATH=<repo>:<repo>/.venv/Lib/site-packages python -m analysis.monitor.y7_alert_eval`, ~15 s)
- **Why:** the night reported 0.720 AUROC and no precision/lift; a "75% predicted" reading was circulating. This adds what a buyer would ask.

## Setup notes

- Feature store builds locally in ~7 s: `monthly.parquet` 22,230 x 118 (gitignored). Settles the "does it build here" unknown in the plan.
- Installed lightgbm, scikit-learn, pyarrow, scipy into the system Python (`pip install`), needed by `analysis/models/`.
- Uses the night's own panel, TURNOVER card (5 columns), shallow LightGBM (50 trees, depth 3), 5-fold group CV seed 20260918. Train = 655 companies with a label (of 1,214), 147 groups, 7,464 company-months, base 28.8%. Group-bootstrap CIs.

## Results (train, out-of-fold; 95% group-bootstrap CI in brackets)

AUROC of the same scores against three labels: 3-month loss (night's Y7) **0.710 [0.679, 0.746]**; 3-month loss plus 25% inflow drop **0.702 [0.666, 0.738]**; strict 6-month loss (no billing t+1..t+6) **0.705 [0.668, 0.740]**. Night's 0.720 is the fold mean; pooled OOF is 0.710.

Flag the top X% of company-months by score:

| Label (base rate) | Flag 5% | Flag 10% | Flag 20% |
|---|---|---|---|
| 3-month loss (28.8%) | 60.5% prec [48, 72], recall 11% | **52.7%** [45, 59], lift 1.8x, recall 18% | 49.2%, recall 34% |
| loss + inflow drop (6.2%) | 20.2%, lift 3.2x | **17.0%** [13, 21], lift 2.7x, recall 27% | 13.5%, recall 43% |
| strict 6-month loss (24.9%) | 52.7% | 47.3%, lift 1.9x | 42.3% |

One-line rule, no model: last quarter's top customer got **no invoice this month** (defined on 6,889 rows): flags 40% of rows, precision **57.6%** [51, 65] (lift 2.1x), recall **82.7%**, AUROC 0.796 (0.719 on the inflow label). It beats the 5-column model on AUROC, but on a different row subset (needs a lag-1 top customer) and it is nearly the label starting a month early.

Other checks:
- **Customers that come back:** 17.9% of "lost" top customers bill again in t+4..t+6 (1,764 rows observable). The strict 6-month label performs the same.
- **Alert load at the 10% cut:** 293 of 655 companies flagged, 1.73 alert episodes per flagged company over the window, 47% of flagged months are false alarms.
- **Holdout, one look** (39 companies, 391 rows, 122 positives, base 31%; LOW_POWER): AUROC 0.724, precision 67.5% at the fixed train 10% cut, recall 22%.

## Reading

- "About 75% predicted" is wrong. About half of flagged months lose the customer (base 29%): a 1.8-2.1x lift, modest. It is a useful early flag, not a forecast.
- Material loss is much weaker: only ~1 in 6 flagged months is followed by a sustained 25% inflow drop (base 6%). Alert wording must be "top customer stopped billing, review exposure and collections", not "revenue at risk".
- Coverage: only companies with invoices and an identifiable top customer (655 of 1,214 train).
- The transparent rule is a better trigger than the model (explainable, high recall); the model score is only useful to rank alerts. Combining the two (rule AND high score, or top-customer share as severity) is untested.

## Decision input for the alert layer

Include as the alert "top customer went quiet" with the rule as trigger, model score as ranking, wording as above, and the numbers above quoted with their intervals. Not in the 0-100 score. Untested: severity filter to cut the 40% flag rate, and the same alert on suppliers.

## Still unknown

- Whether the organizers' hidden companies have invoice history long enough (night: 4% have 24 months).
- Alert precision on a severity-filtered rule.
