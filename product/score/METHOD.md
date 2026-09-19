# Method: what is calculated, how, and why

A plain-language walk through the whole pipeline, from raw CSVs to the alerts in the bundle. It is the document to read before explaining the work to someone else. Code is the source of truth; every section says where.

- **What it is:** a documented, explainable **company health score (0-100)**, computed monthly and as-of (no look-ahead) from the treasury trail, plus a **monitor** that watches how the score moves against the company's own normal and raises alerts with a reason, an owner and an action.
- **What it is not:** a bankruptcy predictor. On the eight accepted outcomes the score does not beat a company-size baseline, and score-fall alerts are followed by those outcomes about as often as an alert on a random month (numbers below). The claim is **explainable and monitorable**. Nothing here says anything about the hidden test.

```text
raw CSVs ─▶ clean (58 rules, logged) ─▶ monthly feature store ─▶ 17 items ─▶ 5 categories ─▶ score 0-100
                                                                                   │
                                            reasons + € amounts ◀──────────────────┤
                                            trajectory (improving / stable / dip / deteriorating)
                                                                                   ▼
                                    monitor: control charts on change ─▶ alerts (owner, action, evidence) ─▶ bundle JSON
                                                                     └─▶ clusters, groups, forecast fan
```

## 1. Cleaning (`analysis/clean_db.py`, `clean.dq_log`)

- **Policy:** the raw tables (`main`) are never touched. In `clean`: rows with no information are dropped (amount 0 or null, orphan rows, null dates), impossible values are set to NULL, doubtful ones get a flag and stay. Every rule, including those that find 0 rows, is counted in `clean.dq_log`, so new dirt in new data shows up in the log instead of silently changing a number. The same rules run on any CSV folder, with nothing re-fitted.
- **What matters most (full data):**
  - **Transactions (2.56 M rows):** empty category or "-" becomes `uncategorized` (25%); rows repeated except for the id are flagged `is_dup`, not dropped (4.1%); empty status becomes `unknown`; zero amounts dropped (369); `value_date` more than 60 days from the date set to NULL (650); exchange rate ≤ 0 set to NULL (77); a typo category (`cash_settlements`) fixed.
  - **Invoices (898 k rows):** `payment_date` on an unpaid invoice (a placeholder equal to the due date) set to NULL (26%); a paid invoice with an impossible payment date (before issue, in the future, year 2000 or 6913) gets NULL plus `payment_date_invalid` (4.4%); due date before issue set to the issue date (20,865); due date more than 730 days after issue set to NULL (376); zero amounts dropped (1,183); pending amount above amount capped at amount.
  - **Other tables:** country names to ISO-2 (36 companies), sentinel balances (-999999999, ±1e10) to NULL with a flag, products created after the snapshot flagged, outstanding above granted flagged.
  - **Detectors for dirt not seen yet** (`source = guard`): duplicate keys, orphan rows, NaN or infinite amounts, null dates, out-of-window dates, unknown categories or statuses. They find 0 rows today and stay logged.
- **Why:** a score built on payment delays is only as good as the payment dates. The two invoice-date rules touch about 30% of invoices, so they are the ones to mention.

## 2. Feature store (`analysis/features/`, `analysis/pipeline.py`)

One row per company-month: 22,230 rows × 118 columns in families a-h (volumes, balances, activity timing, concentration, invoice behaviour, debt and fees...), built deterministically (`SET threads = 1`) from `clean`. Short trails (about 4% of hidden companies have 24 months, some far fewer) are handled: nothing needs a full history.

## 3. The score (`spec.py`, `items.py`, `fit.py`, `scorecard.py`)

### 3.1 Seventeen items in five categories

Item values are computed looking back from each month. "3-mo" means the mean of the last three monthly values. Nothing here is fitted; the definitions and weights are fixed in `spec.py`.

| Category (weight) | Item | How it is calculated | Why it is in this category |
|---|---|---|---|
| **Payment history (35)** | Days paid after due date (suppliers) | Amount-weighted days between due date and payment on invoices paid, 3-mo | The closest analogue of FICO payment history: how the company itself pays |
| | Days customers pay after due date | Same, on issued invoices | Late collection is what strains liquidity |
| | Payables > 30 days overdue | Share of open payables more than 30 days past due, 3-mo | Only lateness beyond 30 days moves default risk (Banque de France) |
| | Receivables > 30 days overdue | Same on receivables | Same |
| **Amounts owed and liquidity (30)** | Months of outflows covered by cash | Month-end cash ÷ mean monthly operating outflow (6 months, 3 if the trail is short), clipped to -6..24, 3-mo | How long the company can pay without income; the analogue of amounts owed |
| | Month-ends with negative cash | Share of the last 3 month-ends with cash < 0 (cash rounded to cents) | Direct stress signal |
| | Times cash turned negative | Onsets of negative cash in the last 6 month-ends | Separates one bad month from a recurring problem |
| | Debt service / inflows | 3-month debt repayment ÷ 3-month operating inflow, 3-mo | Debt burden |
| | Bank fees and interest / inflows | Fees plus interest ÷ operating inflow, same window | Cost of funding |
| **Length and stability (15)** | Months of history | **Fixed:** 100 × min(months, 12) ÷ 12 | A thin file is less certain (the confidence flag says so too) |
| | Months with incoming money | **Fixed:** 100 × share of the last 6 months with any inflow | Guard item: a company that stops receiving money loses points |
| | Outflow volatility | Std ÷ mean of monthly operating outflows over 6 months, capped at 3 | Irregular outflows are hard to plan around; a company trait, fine in a level score |
| **New credit (10, effective 5)** | Debt service rising | Rise of debt service ÷ inflows vs the same window 6 months earlier (0 if it fell) | The observable stand-in for "taking on new credit" |
| | Fees and interest rising | Same rise for fees plus interest | Same |
| **Customer mix (10)** | Dependence on one customer | Customer concentration (HHI), counting only the part above 0.975, 3-mo | The body of the distribution is noise; only the "nearly one buyer" tail carries signal |
| | Credit notes / billing | Share of billing reversed by credit notes, 3-mo | Many credit notes point to disputes or errors. The most experimental item: no paper measures it |

Why FICO's structure: it is a known, defensible way to organise credit health, and each variable sits where its FICO analogue does. New credit is thin (the trail barely observes it), so its weight is halved and the 5 points are spread over the other categories.

### 3.2 From values to points to score

1. **Item points (0-100, 100 = healthiest).** For the 15 percentile items: the share of the **train reference** that is at least as bad as the company's value, times 100. Ties count against the reference, so the best value of a zero-heavy variable (no overdue invoices, no debt service) gets 100. The reference is 501 quantiles per item, fitted on train company-months only (`reference.json`, `fit.py`); holdout companies are excluded by an assertion. The two fixed items use their formulas above.
2. **Category score** = the equal-weight mean of its available items. A category is scored only if at least half of its items exist.
3. **Score** = the weighted mean of category scores over the categories that exist, weights renormalised. A company without invoices has no payment history or mix, so it is scored on the rest, which reads about 5 points higher on average: that is why it carries a **confidence flag**.
4. A company needs 3 months of trail to get a score.

Fixed weights and percentiles are deliberate: they are documented, explainable, and cannot overfit any outcome. Sensitivity check: 60 random reweightings (±30%) keep Spearman ≥ 0.976 with the base score and move AUROC by at most 0.014.

### 3.3 Guard, confidence, trajectory

- **Going-dark guard.** No bank booking of any kind for 60 days: cap of **30** (`guard = dark`). Last-3-month inflows under 25% of the company's own earlier 6-month mean: cap of **50** (`fading`). Both are shown with the score before the guard. Why: several ratios (runway, debt service over inflows) *improve* when activity dies, so without a cap a silent company could score high.
  - **The cap glides, it is not a cliff.** While a guard is on, the score cannot exceed a *ceiling* that starts from the company's previous score and comes down at most **10 points a month** toward the cap (`GUARD_STEP`, `spec.py`), and lifts at once when the guard ends. Why: a guard fires when data is missing or activity drops, and a hard cap turned that into a 20-40 point step (median 20, up to 67) in one month; a third of the dark spells and a quarter of the fading ones last a single month, so they were V-shaped artefacts. 10 points is the size of the largest ordinary one-month fall of the score (guard-free months: 5th percentile -7, 1st percentile -14), so the guard never moves a score more than normal movement does. The cost: a persistent silence takes up to 6 months to reach 30 from a score of 85 (the median company in the test reaches it in month 6), and a company whose inflows collapse reads 10 points less each month instead of at once. The `going_dark` alert still fires in the first month, so the urgency is carried by the alert, not by the score. Effect on the data: month-on-month falls of 20+ points went from 467 to 48, 30+ from 185 to 2, the worst from -67 to -34 (the rest are real moves of the unguarded score).
- **Confidence** (`high` / `medium` / `low`, with a note). Low: under half the category weight available, under 6 months of history, or dark. High: at least 85% weight, 12+ months, not dark, and inflows in at least half of the last 6 months. Medium: the rest.
- **Trajectory.** From the 3- and 6-month regression slopes of the score. `deteriorating` / `improving`: the 6-month slope (±1.5 points per month) has pointed the same way for 3 months in a row and the 3-month slope (±3) confirms it now, a sustained drift. `dip`: the 3-month slope is down now but the trend has not persisted. `stable`: everything else. Needs 4 scored months; a dark company is `deteriorating` by rule.

## 4. Explainability (`explain.py`, `amounts.py`)

- **Contributions:** each item's contribution is its points × (its category's renormalised weight ÷ the number of available items in the category), and the contributions plus the guard adjustment sum exactly to the score (checked to ±0.15, in the export validator and the inspector).
- **Attribution:** the month-on-month change is split by item and by guard; the split sums to the score change when the item set is unchanged.
- **Reasons:** up to 4 for "why the score is not higher" (points lost against a perfect item) and up to 4 for "what moved since last month", each as an English sentence with the euro amount behind it ("Customers paid 62 days after the due date on average… (€184k collected late)"). Structured fields (`item`, `value`, `unit`, `eur`) let the UI template other languages.
- Reasons and items are kept for the last 12 months of the bundle (`--detail-months`).

## 5. Validation of the score (`validate.py`, `validation.md`)

Eight accepted outcomes (`y2_neg_2of3`, `y4_ds_r_double`, `y5_*`, `y7_*`, `y9_*`; never `y3`), 5-fold CV by group, the reference refitted per fold, and each outcome's own feature families removed from the score. Result: AUROC 0.48-0.55, every interval contains 0.5, never significantly above a size baseline (itself about 0.5). Holdout looked at once (72 companies, low power): same picture. Reason: most outcomes are defined against the company's own history, so a *level* score has no reason to predict them. Hence the claim: explainable and monitorable, not predictive.

## 6. The monitor (`analysis/monitor/`)

It watches **change**, because levels are largely company traits (size, business model).

- **Control chart** (`control.py`). For a series (the score, a category score, a gap to the cluster, a group mean): baseline = median of the 12 months ending 3 months earlier; scale = 1.4826 × MAD of that window, floored at the train median so quiet companies do not alert on noise; standardised deviation smoothed with an **EWMA** (λ 0.3, 3σ limit) and a **CUSUM** (k 0.5, h 4). A month "signals" low or high when either crosses.
- **Persistence:** a signal counts only if it holds in **3 of the last 4 months**, so a one-month dip is not an alert.
- **Alert = the onset of a persistent signal that is also material** (`engine.py`): the smoothed level at least 8 points from the baseline for the score, 10 for a category, 8 for a group. Thresholds were fixed in advance, not tuned on outcomes. Only the first month of a run alerts, so a company that stays low does not alert every month.
- **Two-sided:** improvements alert too (`direction: opportunity`).
- **Kinds and severity:**
  - `score_deterioration` / `score_improvement`: `act` at 20+ points from baseline, `watch` at 12+, else `info`.
  - `category_drop`: only when no score alert covers the window.
  - `going_dark`: first month with no bank booking for 60 days; always `act`; it suppresses score alerts for the same run.
  - `top_customer_quiet` (decided in the plan): last quarter's top customer (largest invoiced receivables in months t-3..t-1) got no invoice this month. A transparent rule, first month of the quiet spell only. The night's shallow LightGBM ("TURNOVER card") only **ranks** these alerts (`act` = top decile of onsets) and is never shown as a probability. Wording is fixed: "top customer stopped billing, review exposure and collections", never "revenue at risk". The open receivable in € comes from unpaid invoices to that customer.
- **Reasons of a score alert:** the items whose points moved most since the baseline month, with the € of the current window (`routing.py` maps the biggest mover to an owner: payables and cash → treasurer, receivables → collections, debt and fees → CFO, plus a concrete action).
- **Four comparisons:** company vs its own history; company vs its behaviour **cluster** (the chart of the company's gap to the cluster median, a change of relative standing); group vs its own history; group vs other groups.
- **Clusters** (`behaviour.py`): 17 behaviour features per company (volatility, months without inflows, payroll and tax presence, debt service, collection delay, concentration…), with the linear size signal regressed out, k-means, k chosen by silhouette (k = 4, silhouette 0.19). The structure is weak but stable (adjusted Rand 0.78-0.82 between fits on random halves), so they are shown as **peer groups for a comparison, not segments**. Membership uses the whole trail (a trait) and never triggers an alert.
- **Groups:** the median group has 2 companies, so charts and alerts start at 3 scored members, and the limits widen as the group shrinks (variance = a + b/n fitted on train groups). "Group vs other groups" uses funnel limits on the 3-month change of the group mean.

## 7. Forecast (`forecast.py`)

A fan (median, 50% and 80% intervals) for the score 1-6 months ahead, not a point estimate. The score is not a trending series: monthly changes are negatively autocorrelated and the score is pulled toward the company's own average and the portfolio level. Damped trend (Holt) loses to the last value at 1-3 months and ties at 4-6, per-company ARIMA loses at short horizons, and no yearly seasonality shows in 24 months. What works is a **pooled quantile regression** on the company's own features (deviation from its average, level, last 1- and 3-month moves, volatility): a fan that is skewed by level and shifts with the deviation. Against the naive fan (last value, pooled error quantiles) it improves the pinball loss by 4% at 1 month up to 13% at 6, with 50% and 80% intervals that cover 50% and 80% on held-out companies (holding out the later months too: +2% to +9%, 80% interval covering 74-77%); the median alone gains little (up to 5% at 6 months). Each forecast comes with the pulls behind its 3-month median. Short-lag reversion (lag-1 autocorrelation of monthly changes) disappeared once the guard stopped being a hard cap, so the pull that remains is the slower one. After a material fall the typical company stays down (only 28% recover at least half in 3 months); the average one partly recovers because of a minority. Persistence and mean reversion, not a prediction of outcomes.

## 8. What was measured (`analysis/monitor/evaluation.md`, train companies only)

| Alert | Measured | Reading |
|---|---|---|
| Score falls (≥ 8 points, 3 of 4 months) and going dark | Followed by an accepted outcome within 6 months 29% of the time, vs 31% for an alert on a random month; lift 0.67-1.13 per outcome; median lead time 2 months where followed | Descriptive: says what moved, why and by how much. Does not predict |
| Top customer quiet (onset) | 56% of flagged company-months lose the customer vs 29% base (lift ≈ 1.9, CI 48-63%); 22% of rows flagged; recall 42% | The one alert with a measured lift. Top decile of the ranking 62%; a customer that billed in all 3 of the last 3 months only 33% (near base, so `info`) |
| Volume | 0.53 risk and 0.22 improvement alerts per company-year | 41% of one-month falls of 8+ points became an alert within 3 months; the rest reverted. (The count of such falls rose from 848 to 1,259 because the guard now comes down in several 10-point steps instead of one cliff) |
| Group funnel | 1.6% of train group-months outside 3σ, 0.06% persistently | Groups under 3 members get no limits |

## 9. Decisions and why (one line each)

| Decision | Why |
|---|---|
| Percentile scorecard with fixed weights | Explainable, documented, no outcome fitting; weights barely matter (sensitivity) |
| FICO categories | A defensible structure the audience knows; each item sits with its analogue |
| Reference fitted on train companies only; holdout excluded by assertion | No leakage; holdout looked at once at the end |
| Reweight over available categories + confidence flag | Companies without invoices must still be scored, but not ranked next to full-data ones unmarked |
| Going-dark cap | Some ratios improve when a company goes silent; silence must never score high |
| Monitor change, not level | Levels are traits (size, model); movement is what needs attention |
| Robust (median/MAD) charts with a floor | Outliers and very quiet series would otherwise trigger or hide alerts |
| Persistence 3 of 4 and onset-only alerts | A dip is not a trend; a persistent state should not alert every month |
| Materiality thresholds fixed a priori | Avoid tuning thresholds on the outcomes |
| Two-sided alerts | Improvement is also actionable (refinancing, cash use) |
| Top customer quiet is an alert, not a score input | It is the only signal with measured lift; a transparent rule, with a model only for ranking |
| Clusters with size removed, treated as peer groups | Compare behaviour, not size; structure is weak so no segment stories |
| Group minimum size 3 and funnel limits | The median group has 2 companies; small groups are noisy |
| Pooled quantile regression for the forecast fan, no trend, no seasonality | Trend models and per-company ARIMA do not beat the last value; the mean-reversion fan beats the naive fan at every horizon out of fold; no yearly pattern in 24 months |
| Static JSON bundle plus a clean DuckDB | The pipeline runs once; the bundle explains, the DuckDB answers questions about records |

## 10. Guardrails

Holdout companies are never fitted on. Y is never built from the X that predicts it (each outcome's own feature families are removed in validation). No look-ahead: windows, baselines and the top-customer window are all lagged. The night's social-security / payroll / movement-days card is not used. `y3_recover_cash_6m` is not reused. No claim on the hidden test.

## 11. Known limits

- The score is not predictive on the accepted outcomes; score-fall alerts are descriptive.
- Companies without invoices have no payment-history or mix category (about 5 points higher; flagged).
- Counterparty IDs (`COUNTERPARTY_*`) do not map to company IDs, so customers and suppliers cannot be scored or watched as entities. The suppliers version of the top-customer alert was not tested.
- Amounts are in each invoice's own currency (no FX conversion). No debt balance path (debt enters as debt service and fees over inflows).
- The runway item is slow to react to a one-quarter collapse of outflows (6-month denominator).
- Group means move when a company enters or leaves; the chart does not adjust.
- Two scorecards coexist in `product/score/` (the v0 dummy card and this v1); the bundle exports v1 only.

## 12. Where things live

| What | Where |
|---|---|
| Cleaning rules and log | `analysis/clean_db.py`, `clean.dq_log`, `analysis/README.md` |
| Pipeline entry point | `analysis/pipeline.py` |
| Scorecard definition, items, points, explain | `product/score/spec.py`, `items.py`, `fit.py`, `scorecard.py`, `explain.py`, `amounts.py`, `reference.json` |
| Score validation | `product/score/validate.py`, `validation.md` |
| Monitor, clusters, forecast, evaluation | `analysis/monitor/` (README, `evaluation.md`, `forecast_evaluation.md`) |
| Bundle, contract, inspector | `product/score/export.py`, `DATA_CONTRACT.md`, `inspector/` |
| What to ship to the web app | `AGENTS.md` ("What goes to the web app's storage") |
