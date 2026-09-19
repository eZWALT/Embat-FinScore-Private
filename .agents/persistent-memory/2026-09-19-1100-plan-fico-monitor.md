# 2026-09-19-1100 — plan: cleaning → FICO-like score → clusters + control charts → forecasts → product

- **Author:** Javier Boix (with Claude Code)
- **When:** 2026-09-19 ~11:00 CEST
- **Decision by:** Javier. Short version lives in `AGENTS.md` ("Plan"). This entry holds the reasoning.

## Revision (~11:45, after `2026-09-19-1130-night-variables-for-score.md`)

Scope shrinks: the night did feature/outcome discovery, so we select and combine; no model training. Changes by step:

- **1:** entry point also rebuilds the feature store (features read the `clean` schema); must handle short trails (~4% of hidden companies have 24 months).
- **2:** candidate pool is the 1130 map; level score, so stable traits are allowed. New credit is thin (rise-only connection artefacts, utilisation 1.6%, no NSF token): shrink its 10% and say so. We build the going-dark guard; the night's social-security/payroll/movement-days card is not used (all signs negative, built on the flawed Y3). Validate on eight accepted outcomes, not `y3`.
- **3:** chart within-company change (levels are traits). "Top customer went quiet" is an alert reason. Monitoring claim only; lead time = alarm vs later outcome, not the night's lag-1 "one month".
- **4:** expect a tie with naive last value (night: cash best forecast by last value; weekly forecasts lose to the historical mean). Persistence + interval; first to cut.
- **5:** unchanged scope; alerts get plain-language reasons from the night's list.
- **New risks:** feature store dependency; short hidden trails; invoice signals cover ~744 of 1,214 train companies.

The sections below are the original text and still hold except where this revision overrides them.

## Brief coverage check (~12:10)

Checked against the four capabilities and the submission table. Product idea 01 = "Health Score Sentinel" (`product/web/index.html`, static mockup, pushed in `4425998`): an agent inside Embat that watches each company, customer and supplier, explains the change and alerts the right person with a concrete recommendation.

| Brief item | Where the plan answers it | Gap closed here |
|---|---|---|
| Read the trail (bank, issued/received invoices, payment behaviour, funding cost, debt balances) | Step 1 + 1130 variable map covers bank (runway, volatility, activity), invoices both ways, delays, `f_fc_r`, `f_ds_r` | **Debt balances are the weakest:** only debt-service ratio over time; outstanding/utilisation is a snapshot (1.6% coverage). Check whether the debt schedule (`debt_schedule_config`) lets us rebuild a balance path; otherwise state the limit |
| Prediction on hidden test (mandatory) | Step 1 makes cleaning re-runnable | **No deliverable scored unseen companies.** Add: `score_new(csv_folder)` → `company_id, month, score, trajectory`. The submission format and what the leaderboard scores (level or change) are not in our notes (brief heading "Qué ponemos nosotros" was never captured): ask the organizers |
| Trajectory, not snapshot (mandatory) | Step 3 charts, step 4 forecast | **A level score alone is a snapshot.** Define the headline output as score (trailing 3–6 month windows, not one month) plus a trajectory state (3- and 6-month slope with persistence: improving / stable / dip / deteriorating) |
| Two-way signal (mandatory) | Step 3 | Charts must be two-sided, and improving companies get an alert too ("opportunity", idea 01 Northbrook). Evidence for the improving side is weak: the only night outcome for it (Y3) is invalid. Validate by "score rise precedes fewer negative-cash / late-payment events" and report as weak |
| Explanation: why this score, why it changed (mandatory) | Step 2 "explainable" | **Not concrete.** Add per-variable point contributions, month-on-month change attribution (which variable moved, when, how many points), top-4 reason codes in plain sentences, and the € amount behind each (idea 01 quotes €184k delayed collections, €92k debt instalment) |
| Product on top (mandatory) | Step 5 | Idea 01 already chosen as the base; needs routing to owner (tesorero / CFO / Cobros), and customers and suppliers as watched entities. Verify counterparty IDs map to `company_id`s before promising that |
| Identified buyer (mandatory) | Step 5 says Embat | **No rationale written.** Draft: Embat sells it as a premium module / TellMe skill on data it already holds; it moves Embat from visibility to decisions (see `2026-09-19-embat-business-context.md`), adds upsell and retention. State it in one paragraph in the demo |
| Interactive demo (mandatory) | Step 5 | **Dockerfile and runtime are empty; nothing is hosted.** Needs a live URL, not a laptop notebook: precompute scores and alerts, pick the stack when step 5 starts, deploy a first version early |
| Measured advance notice (bonus) | Step 3 lead time | Define the event: first alarm date vs the later accepted outcome (`y2_neg_2of3`, `y7_top1_lost`, `y9_*`); report the median in months and its base rate. Night's "one month" is not this |
| Proactive monitor (bonus) | Step 3 persistence rule + Step 5 alerts | Alert feed generated without a query; include a false-alarm rate |
| Stability: dip vs decline | Step 3 persistence rule | Report how many single-month dips do not become alerts on train |
| Generalization | Group-fold CV on train; holdout reported once | Say plainly it is a documented method, not proven on the hidden set |

## Why this order

- No official label and no time to fix the overnight recovery outcome (see `2026-09-19-1015-y3-recovery-mechanical.md`). So the score is a documented, explainable method; the monitor (control charts) covers "both directions" and "dip vs decline" without a label.
- The control charts are the brief's monitor bonus and the core of the Health Sentinel product.
- Overnight docs (`NORTH_STAR.md`, `CONTRACT.md`) say no 0–100 formula and no `product/` tonight. That was scoped to the night; this plan opens goals 2–4.

## Step 1 — cleaning pipeline

Exists: `analysis/build_db.py`, `analysis/clean_db.py`, `clean.dq_log` (rules in `2026-09-18-2245-clean-schema.md`). Gap: single re-runnable entry point over an input folder, so a new CSV drop (e.g. hidden companies) is cleaned by the same rules.

## Step 2 — score, FICO analogy

| FICO category (weight) | Analogue in the trail |
|---|---|
| Payment history (35) | Customers' lateness on our invoices, our lateness to suppliers |
| Amounts owed (30) | Liquidity runway, debt service vs inflows, financing cost |
| Length of history (15) | Trail length, stability of activity |
| New credit (10) | New facilities, debt appearing after a cash dip |
| Credit mix (10) | Customer/supplier concentration, product mix |

- Use 10–12 as-of columns from the feature store rather than rebuilding features. Store is built by `python -m analysis.features.build_feature_store`; need to confirm it runs on this machine before relying on it. Fallback: `analysis/score_pipeline.py`.
- Weights fixed a priori (mirror FICO), sensitivity analysis only, no fitting.
- 470 of 1,214 train companies have no invoices: reweight over available categories, show a confidence flag.
- Must not reward going dark (the Y3 trap: shrinking outflows look like recovery).
- Validity evidence: score at t vs accepted outcomes (negative cash, debt-service shock, late payment, top-customer loss), group-fold, vs size bar. Expect modest AUROC; report as is.

## Step 3 — clusters and control charts

- Cluster on behaviour features (not size), fit on train only.
- Robust EWMA/CUSUM on score and its categories, median/MAD baseline. Alert needs persistence (3 of last 4 months) so a dip is not a decline; measure false-alarm rate on train.
- Comparisons: company vs own history; company vs cluster; group vs own history; group vs other groups.
- 250 groups, median 2 companies: enforce a minimum group size and use funnel-style limits; small groups fall back to company charts.
- Lead time = months from first alarm to a later accepted outcome.

## Step 4 — forecasts

Level signals persist, change signals do not (`2026-09-18-2210-python-port-first-validation.md`). So forecast a smoothed score level with an interval and compare to naive last value; no fancy model.

## Step 5 — product

Health Sentinel as an Embat / TellMe skill (business context: `2026-09-19-embat-business-context.md`). Read the product-idea entry pushed in `4425998` before starting.

## Risks

- Deterioration side (82→68) had no explanation overnight; here it rests on control charts, a monitoring claim, not a predictive one.
- Nothing can be proven on the organizers' hidden test; describe the score as a documented method, not a validated predictor.

## Decision (~13:15, after `2026-09-19-1300-y7-alert-grade-eval.md`)

- "Top customer went quiet" goes into the alert layer (step 3 / product step 5), not the 0-100 score. Trigger: last quarter's top customer received no invoice this month. Model score (night's TURNOVER card) ranks alerts only.
- Measured on train (group-fold OOF): top 10% by score → 53% lose the customer vs 29% base (1.8x); the rule → 58% precision, 83% recall, flags 40% of rows; with a sustained 25% inflow drop the precision is 17% vs 6% base. About one month of notice; 18% of lost customers bill again within 3 more months; 47% of flagged months are false alarms.
- Wording is "top customer stopped billing, review exposure and collections", never "revenue at risk". The "75% predicted" reading is wrong: the night's 0.720 is an AUROC.
- Open: severity filter (customer share of billing) to cut the 40% flag rate; the same alert for suppliers.

## Unknowns resolved

- The feature store builds locally in ~7 s (22,230 x 118, `data/feature_store/monthly.parquet`, gitignored).
- "Product idea 1" = Health Score Sentinel (`product/web/index.html`, idea 01); read. Ideas 02 (Group Health Map) and 03 (Debt Opportunity Advisor) exist as static mockups too; not chosen.

## Still unknown

- Submission format and what the leaderboard scores (level or change); ask the organizers.
- Whether the debt schedule lets us rebuild a balance path over time.
- Whether counterparty IDs map to `company_id`s (needed to watch customers and suppliers).
- Hidden-set invoice history length.
