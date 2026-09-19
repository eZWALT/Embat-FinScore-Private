# product/score/ — v0 dummy FICO-like card

A first 0–100 number the product can show. Weights are fixed in advance.
The only fit is a train-only percentile table. Not a model, not a claim
about the hidden test.

```bash
PYTHONPATH=. python -m product.score
```

Writes `outputs/monthly_scores.parquet` and `outputs/ref_v0.json` (gitignored).

## Row contract (v0, for `poc/` and later `product/web/`)

One row per `company_id` × `period`. Bind these; treat the rest as extras.

| Column | Meaning |
|---|---|
| `score` | Monthly 0–100 after stretch and dark cap |
| `score_3m` | **Headline.** 3-month median of `score` |
| `state` | `improving` / `stable` / `deteriorating` from Δ `score_3m` over 3 months (long dark → deteriorating) |
| `confidence` | 0–1 from available category weight (thin file ×0.8; dark capped at 0.55) |
| `confidence_band` | `high` / `medium` / `low` |
| `no_invoices` | Payment + mix missing; do not show a 90 as “perfect payer” |
| `going_dark` / `going_dark_long` | Dark cap fired |
| `cat_payment_history` / `cat_amounts_owed` / `cat_length_stability` / `cat_mix` | Category 0–100 (NaN if dropped) |
| `reason_1_es` … `reason_3_es` (and `_en`, `_code`) | Weakest items; Spanish for the UI |
| `group_id` | From the store, when present |

Not in v0: fifth “new credit” subscore, € on each reason, owner/action alerts (those stay step 3 / Sentinel). Rebuild the parquet after card changes.

## What is on the card

New credit is empty here (utilisation 1.6%, no bounced-payment token,
facility counts are connection artefacts). Its 10 FICO points move to
amounts owed.

| Category | Weight | Variables | Direction |
|---|---:|---|---|
| Payment history | 35 | `e_ar_overdue_30`, `e_ap_overdue_30`, `e_delay_coll`, `e_delay_paid` | lower better |
| Amounts owed | 40 | `b_runway`, `f_ds_r`, `f_fc_r` | runway up, ratios down |
| Length / stability | 15 | `active_share_6` (share of last 6 months with a bank movement) | higher better |
| Mix | 10 | `d_cust_top1`, but only the tail: 0.70 → 100 pts, 0.975 → 0 pts | lower better |

Missing category → drop it and reweight the rest. No invoices (about 470
of 1,214 train companies) means payment and mix are empty, so the number
is runway + debt cost + “still operating”, with a medium/low confidence
flag.

Left out on purpose:

- Social security / payroll / movement-days as “quiet = healthier” (Y3 trap).
- `trail_months` as points (mostly when the company connected, not age).
- `d_supp_hhi` (night found concentration *protective*; unsigned).
- `e_credit_note_ratio` (experimental).
- `e_issued_top1` / customer-loss (alert layer, not this score).

## Guards

- **Going dark scores low, never high.** No movements this month → cap 50.
  Recency > 45 days → cap 35.
- **Thin file:** fewer than 6 months on the grid → confidence × 0.8.
  Not extra points for a long extract.
- **Holdout** (`analysis/splits/holdout_companies.csv`) never enters the
  percentile table.

Averaging several percentile items would leave almost everyone in the
40s–60s. After the weighted mean we stretch by 2 around 50 (a priori, not
fit): `clip(50 + 2*(raw − 50), 0, 100)`. Then the dark cap.

## Headline the product should show

`score_3m` = 3-month median of the monthly score, plus a coarse state
from the 3-month change (improving / stable / deteriorating). Long dark
is always deteriorating.

Reasons are the weakest items below 50 points, plus the dark cap when it
fires. English and Spanish sentences sit on the row.

## First run (train, latest month 2026-08)

1,214 companies. `score_3m` p10 / p50 / p90 = **17 / 47 / 80**.
39% have no invoices (cash + “still operating” only; flag the gap).
10% are going dark (median 35). Mix is 100 for almost everyone — the
body of concentration is treated as noise; only the >97.5% tail hurts.

## What this is not

It does not explain 82→68 without cash (runway *is* cash; that is the
photograph, not a cause). It is not validated yet against the eight
accepted outcomes. Do not quote an AUROC for this card until that check
runs.

---

## Explainable scorecard (modules `spec`, `items`, `fit`, `scorecard`, `explain`, `run`, `validate`)

Second, separate implementation of plan step 2 living beside the v0 dummy above: it does not write the v0 parquet and the POC does not read it yet. Modules are `spec.py`, `items.py`, `fit.py` (writes `reference.json`), `scorecard.py`, `explain.py`, `amounts.py`, `run.py`, `validate.py`, `guard_test.py`, `score_new_check.py`; run them as `python -m product.score.run` etc. (not `python -m product.score`, which is the v0 card).
A documented, explainable score computed monthly and as-of (no look-ahead) from trailing 3–6 month windows of the treasury trail, with a trajectory state, per-variable contributions, month-on-month attribution and plain-language reasons with the € behind each. **It is a method, not a proven predictor**: on the accepted outcomes it does not beat a company-size baseline (see [validation.md](validation.md)). The claim is *explainable and monitorable*. Nothing here says anything about the hidden test.

### Run

Set `PYTHONUTF8=1` and `PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo>` (system Python; duckdb, pandas, pyarrow, scipy).

| What | Command |
|---|---|
| Score unseen companies: CSV folder → clean DB → features → score | `python -m product.score.run --csv-folder <dir> --out <dir>` (or `run.score_new(csv_folder)`) |
| Refit the percentile reference (train companies only) | `python -m product.score.fit` |
| Validation, group-fold, size baseline, weight sensitivity, holdout once | `python -m product.score.validate [--holdout]` (~8 min) |
| Going-dark and no-look-ahead tests | `python -m product.score.guard_test` |
| `score_new` on a subset folder with short trails and no invoices | `python -m product.score.score_new_check` |
| **Static JSON bundle for the web app** (Next.js on Vercel), contract in [DATA_CONTRACT.md](DATA_CONTRACT.md), sample in [sample_bundle/](sample_bundle/) | `python -m product.score.export --csv-folder <dir> --out <bundle>` |

`score_new` writes `scores.csv` (`company_id, month, score, trajectory, confidence`) and `scores_detail.parquet` (categories, per-item points and contributions, month-on-month attribution `d_*`, `reasons`, `change_reasons`). The output format is ours; the organizers' submission format is still unknown.

### Method

Fixed a priori (`spec.py`): categories, weights, items, directions, caps, thresholds. Fitted: only the percentile reference (`reference.json`: 501 quantiles per item, train companies only, holdout excluded by assertion).

**Item points** = share of the train reference that is at least as bad as the company's trailing-window value, ×100 (100 = healthiest; ties count against the reference so "no overdue invoices" scores 100). Two items use a fixed mapping. **Category score** = mean of its available item points (a category is scored if at least half its items exist). **Score** = weighted mean of the category scores that exist, weights renormalised, then the going-dark cap.

| Category (weight) | Items (equal weight inside) |
|---|---|
| Payment history (35) | days suppliers are paid after due date; days customers pay after due date; payables > 30 days overdue; receivables > 30 days overdue |
| Amounts owed and liquidity (30) | cash / monthly outflows (runway); month-ends with negative cash; times cash turned negative; debt service / inflows; fees + interest / inflows |
| Length and stability (15) | months of history (fixed, full at 12); months with incoming money (fixed); outflow volatility |
| New credit (10 → 5) | debt service rising vs 6 months earlier; fees + interest rising vs 6 months earlier |
| Customer mix (10) | dependence on one customer (HHI tail above 0.975); credit notes / billing (experimental) |

**New credit is thin and shrunk.** The trail barely observes it (facility counts rise with connection dates; utilisation exists for 1.6% of rows; no NSF token), so its 10 is shrunk to 5 and the 5 points are redistributed by renormalising. It is scored only from month 9 of a trail, and it is the least informative category. Not used, on purpose: social-security / payroll / movement-days card (built on the flawed Y3, its signs reward going quiet), `f_n_facilities` / `f_n_types` / `g_n_accounts` (rise-only artefacts), `d_supp_hhi` (the night found concentration protective, against the literature).

**Missing data.** Categories that do not exist for a company-month (no invoices, no cash accounts, short trail; delay features are null for the first 6 calendar months) are dropped and the rest reweighted. `confidence` is `high` (≥ 85% of the weight present, ≥ 12 months of history, not dark, money coming in), `low` (< 50% of the weight, < 6 months, or dark), else `medium`; `confidence_note` says why. **Bias to know:** dropping payment history — the harshest category (mean 53 points) — lifts companies without invoices by about 5 points on average (68.3 vs 63.4); that is reweighting, not health, and the flag is there for it.

**Going dark scores low, never high** (`spec.py` guard). Outflow ratios improve and invoice delays vanish when activity dies: on the real data the raw score of a company silent for 200–500 days is ~90. So: no booking of any kind in the 60 days before month-end → **dark**, score capped at 30, confidence low, trajectory `deteriorating`; last-3-month inflow under 25% of the company's own earlier 6-month mean → **fading**, capped at 50. Tested in `guard_test.py` on 60 companies from the top 40% by score whose data stops after 2026-02 (bank only, and bank + invoices): every dark row is ≤ 30, none above the pre-cutoff score, from month 3 every company is flagged; with the guard off the median post-cutoff score stays 71–87. The gap: the first silent month is not yet flagged (7% of rows in months 1–2 score above the pre-cutoff score, largest rise +8.9 points). A bank connection that drops looks the same as a company that stops; the score cannot tell them apart.

**Trajectory** (`explain.py`): OLS slope of the last 3 and 6 scores. *improving / deteriorating*: the 6-month slope has pointed the same way (±1.5 points/month) for 3 months in a row and the 3-month slope confirms it (±3 points/month). *dip*: 3-month slope down but no persistent 6-month trend. *stable*: otherwise (including a step that has levelled off; the level is in the score). *insufficient history* before 4 scored months. Thresholds were set from the score's own scale (within-company sd ≈ 7 points), not from outcomes.

**Explanations.** `contrib_<item>` sums to the pre-cap score; `guard` adjustment = score − pre-cap score. `d_<item>` is the month-on-month change per item and sums exactly to the score change (max error 3e-14), including re-weighting when a category appears. `reasons`: top 4 items by points lost against a perfect item, each with a sentence and the € behind it (late-paid amounts from `clean.invoices` over the 5 months the delay averages, overdue stock now, cash and monthly outflows, debt service and fees over 5 months, top-customer billing, credit notes). `change_reasons`: top 4 movers since last month.

### Validation (numbers in [validation.md](validation.md), regenerated by `validate.py`)

Eight accepted outcomes (`y2_neg_2of3`, `y4_ds_r_double`, `y5_*`, `y7_*`, `y9_*`; never `y3_recover_cash_6m`), 5-fold CV by group on train, reference refit per fold, the items from the label's own feature family removed per outcome, AUROC with 95% group-bootstrap intervals, size baseline. Result: level AUROC 0.48–0.55 with every interval containing 0.5, never significantly above the size baseline (which is itself ~0.5); a 3-month score fall does no better (0.455–0.537), and a naive inflow-drop monitor is at least as good. Trajectory states have no consistent relative risk across outcomes. Weights barely matter: 60 random reweightings (±30%) keep Spearman ≥ 0.977 with the base score and move AUROC by at most 0.013 on any outcome (dropping the payment-history or amounts-owed category moves it by up to 0.03–0.06). **Holdout looked at once** (72 companies, 8–96 positives per outcome, LOW_POWER; reference fitted on all train): AUROC 0.39–0.63, same picture (table in validation.md); it was not used for any choice, do not look again. Sanity check that the machinery works: with the family left in, `amounts_owed` predicts `y2_neg_2of3` at AUROC 0.84 (leaky by construction, not a result). Most accepted outcomes are defined against the company's own history (`ownp80`, `double`, `top1_lost`), so a *level* score has no reason to predict them; the night's own finding that levels are traits points the same way.

### Limits and open items

- Not predictive on these outcomes; explainable and monitorable only. No claim on the hidden test.
- No debt balance path: `debt_schedule_config` covers 87 loans (40 companies) and was not tested for rebuilding balances; debt enters only as debt service and fees over inflows.
- Counterparty IDs are `COUNTERPARTY_*`, company IDs `COMP_*`: no link, so customers and suppliers cannot be scored as entities.
- Euro amounts are in each invoice's own currency, as in the feature store (no FX conversion).
- The store's negative-cash flags flip on 1e-10 noise; the score recomputes them from cash rounded to cents.
- The "top customer went quiet" alert is not in the score (decided, plan step 3).
