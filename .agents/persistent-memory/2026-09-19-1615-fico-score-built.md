# 2026-09-19-1615 — plan step 2: FICO-like score in `product/score/` (built, validated, not predictive)

- **Author:** Claude Code (Sonnet 5) for Javier Boix
- **When:** 2026-09-19 ~16:15 CEST. Step 1 is in `2026-09-19-1420-pipeline-entry-point.md`. Steps 3–5 not started.

## What exists

`product/score/` (README has the method, run commands, limits): `spec.py` (categories, items, weights, guard: all fixed a priori), `items.py` (trailing-window item values from the feature store), `fit.py` (percentile reference, train companies only, `reference.json` 93 KB committed), `score.py`, `explain.py` (trajectory, month-on-month attribution, reasons with €), `amounts.py` (€ from `clean.invoices`), `run.py` (`score_new(csv_folder)`), `validate.py`, `guard_test.py`, `score_new_check.py`, `validation.md` (+2 csv).

- Score 0–100: payment history 35, amounts owed 30, stability 15, new credit 10 → shrunk to 5, mix 10; 17 items, equal weights inside a category, percentile points vs the train reference (ties count against the reference), reweighting over categories that exist, `confidence` high/medium/low with a note.
- Going-dark guard: no booking for 60 days → capped at 30; inflow < 25% of own earlier 6-month mean → capped at 50. Real data shows why: the raw score of companies silent for 200–500 days is ~90.
- Trajectory: improving / stable / dip / deteriorating from 3- and 6-month OLS slopes with a 3-month persistence rule; dark → deteriorating; "insufficient history" before 4 scored months.
- Outputs: `scores.csv` (company_id, month, score, trajectory, confidence) and `scores_detail.parquet` (categories, per-item points and contributions, `d_*` attribution that sums exactly to the score change, top-4 `reasons` and `change_reasons` with €).
- `score_new` on the full CSVs: 24 s end to end; 19,658 scored company-months (score needs ≥ 3 months of trail).

## Numbers (train, 1,214 companies; all in validation.md)

- Score: mean 65.1, sd 13.6; confidence high 23% / medium 47% / low 30%; capped 11% of rows (fading 9%, dark 3%); month-to-month autocorrelation 0.84; Spearman with log inflow 0.145 (not a size proxy). Companies without invoices average 68.3 vs 63.4: reweighting, not health (payment history is the harshest category, mean 53).
- **Validation, level:** AUROC 0.48–0.55 on the eight outcomes, every 95% group-bootstrap interval contains 0.5, never significantly above the size baseline (also ~0.5). **Change** (3-month score fall): 0.455–0.537, no better than a naive inflow-drop monitor. Trajectory states: no consistent relative risk (`deteriorating` RR 0.31–2.11 across outcomes; y7_top1_lost_inflow 2.11 [1.07, 3.41] but y9_fee_r_ownp80 0.55 [0.38, 0.74]). Weights: random ±30% reweighting keeps Spearman ≥ 0.977, moves AUROC ≤ 0.013; dropping a whole category moves it 0.03–0.06.
- Machinery check: with the family left in, `amounts_owed` predicts `y2_neg_2of3` at 0.84 (leaky by construction). The outcomes are mostly defined against the company's own history, so a level score has no reason to predict them.
- **Holdout was looked at once** (final `validate --holdout` run): AUROC 0.39–0.63 on 8–96 positives, LOW_POWER, same picture. Do not look again.
- Guard test (60 top-40% companies whose data stops after 2026-02): all dark rows ≤ 30, none above the pre-cutoff score, all flagged by month 3; guard off → median post-cutoff score stays 71–87. Gap: month 1 of silence is not flagged (7% of months 1–2 above the pre-cutoff score, max +8.9). No look-ahead: cutting later data leaves earlier scores unchanged (bank only: identical; bank + invoices: max 0.09 point, not chased).

## Decisions / things I changed from the brief

- Runway uses a 6-month outflow denominator (3-month when the trail is short), not `b_runway`'s 3-month one, so a one-quarter outflow collapse does not push runway to its clip. Negative-cash items are recomputed from cash rounded to cents (store flags flip on 1e-10 noise, see step 1 entry).
- New credit has two items (rise of `f_ds_r` and of `f_fc_r` vs 6 months earlier); `f_new_facility` is dropped (monthly counts rise 32 → 186 with connection dates).
- Dark = no booking at all. "Three months without incoming money" was not used: 99% of such rows keep booking transactions.
- Trajectory thresholds (3 and 1.5 points/month) were chosen after seeing state shares (dip 11%, deteriorating 5%, improving 1%), from the score's scale, not from outcomes. A monitor that flags ~15% of company-months is noisy: step 3's persistence/EWMA layer has to cut it.
- Sensitivity variants, guard, thresholds were all set before the holdout was looked at; validation numbers reproduced exactly on the final run (fixed seeds).

## Still unknown / for Javier

- Whether to accept "explainable and monitorable, not predictive" as the pitch, or add an inverse-size/own-history-relative outcome that a level score can be tested on. I did not build a new outcome (rule: no new discovery).
- No-invoice bias (+5 points): alternatives are to fill a missing category with a neutral 50 or the population mean. Not done; the plan says reweight + flag.
- Submission format; `debt_schedule_config` as a debt balance path (87 loans, untested); counterparty ↔ company mapping (IDs `COUNTERPARTY_*` vs `COMP_*`, no overlap per the feature dictionary).
- The default `data/embat.duckdb` and `data/feature_store/monthly.parquet` were not regenerated; all runs used scratch folders. Running `python -m analysis.pipeline` without `--work-dir` would replace them (harmless: same data, `clean.dq_log` gains a `source` column, `b_*` float noise differs).
