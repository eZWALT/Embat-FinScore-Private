# Scores and alerts database (`api` schema, Postgres on Neon, read-only)

Chart data comes from here through `query_api(sql)`: one `SELECT`, only `api.*` views, always with `LIMIT` (at most 2,000 rows). Every view reads the **current score run**. Months are text `YYYY-MM`. Money in `eur` columns is in the company's own currency. Use SQL to filter, join and aggregate; do not do it in your chart code. Alias columns to what your chart reads (`company_id AS entity_id`, `score AS value`).

## Views that exist today

| View | One row is | Columns |
|---|---|---|
| `api.score_series` | a company-month | `company_id, month, score, trajectory, confidence, guard` |
| `api.current_index` | a company at its latest month | `company_id, group_id, latest_month, score, trajectory, confidence, guard, delta_1m, delta_3m, top_reason, cluster_id, n_alerts, max_alert_severity, sparkline` |
| `api.current_scores` | a company-month, full detail | `company_id, month, score, score_pre_cap, guard, guard_adjustment, trajectory, slope3, slope6, confidence, confidence_note, coverage, trail_months, change_guard` |
| `api.top_reasons` | a level reason (why the score is not higher) | `company_id, month, position, item, label, points, eur, sentence` |
| `api.alerts` | an alert | `alert_id, entity_type (company/group), entity_id, month, kind, direction, severity, title, summary, owner, action, persistence_rule, months_flagged, evidence (jsonb)` (never plot `rank_score`) |
| `api.dashboard_companies` | a company | profile + latest score + `score_history` (jsonb) + `categories` (jsonb): prefer the views above for charts |
| `api.manifest` / `api.current_run` | the run | `as_of_month, months, spec (jsonb: labels, weights, items), disclaimer` |

## Views for charts (added by migration 007; if a query says the relation does not exist, use the bundle tools instead)

| View | One row is | Columns |
|---|---|---|
| `api.score_categories_long` | a company-month-category | `company_id, month, category_id, score, contribution` |
| `api.score_items_long` | a company-month-item (last `detail_months` only) | `company_id, month, item_id, value, points, contribution, delta` |
| `api.score_reasons_long` | a reason | `company_id, month, kind ('level'/'change'), position, item, label, points, value, unit, eur, sentence` |
| `api.control_chart_rows` | a month of a control chart | `entity_type, entity_id, comparison, metric, month, value, center, lower, upper, ewma, signal, persistent` (comparisons: `own_history`, `cluster`, `group_own_history`, `group_vs_groups`; metrics `score, payment_history, amounts_owed, stability`) |
| `api.forecast_rows` | a future month of the fan | `company_id, method, origin_month, naive_last, skill_vs_naive, month, median, lo50, hi50, lo80, hi80` |
| `api.group_series` | a group-month | `group_id, n_companies, limits_available, month, mean_score` |
| `api.group_members_v` | a member | `group_id, company_id` |
| `api.company_profile` | a company | `company_id, group_id, country, currency, erp, first_month, latest_month, cluster_id, cluster_label, cluster_description` |
| `api.cluster_vs_rows` | a metric of a company | `company_id, cluster_id, metric, percentile, robust_z` (percentile 100 = healthiest in its behaviour cluster) |
| `api.alert_reasons_v` | an alert reason | `alert_id, position, item, label, points, value, unit, eur, sentence` |

Facts that shape queries: scores start in 2024-11 (a score needs 3 months); `score_items_long`/reasons exist only for the last 12 months; control charts need 7 scored months and group charts need 3 members; a company's group mates are `api.group_members_v` joined by `group_id`; the score is capped at 30 (`guard = 'dark'`) or 50 (`'fading'`), `score_pre_cap` in `api.current_scores` is what it would be.

## Cookbook: SQL for the recipes

```sql
-- A. a company against its own normal
select month, value, center, lower, upper, ewma, signal, persistent
from api.control_chart_rows
where entity_type = 'company' and entity_id = 'COMP_0016' and comparison = 'own_history' and metric = 'score'
order by month limit 200;

-- B. what moved the score (items) and the two totals
select item_id as item, delta as value from api.score_items_long
where company_id = 'COMP_0016' and month = '2026-08' and delta is not null limit 50;
select month as label, score as value from api.score_series
where company_id = 'COMP_0016' and month in ('2026-07', '2026-08') order by month;

-- C / F / H / J. one company against the other companies of its group (long rows: month, entity_id, value)
select s.month, s.company_id as entity_id, s.score as value
from api.score_series s
join api.group_members_v m on m.company_id = s.company_id
where m.group_id = (select group_id from api.company_profile where company_id = 'COMP_0016')
order by s.month limit 2000;

-- D. position among peers
select metric, percentile from api.cluster_vs_rows where company_id = 'COMP_0016' and percentile is not null;

-- E. portfolio map
select company_id, score, delta_3m, n_alerts, trajectory, cluster_id from api.current_index limit 2000;

-- I. forecast fan with the score history (same columns in both halves)
with hist as (
  select month, score, null::float as median, null::float as lo50, null::float as hi50, null::float as lo80, null::float as hi80, null::float as naive_last
  from api.score_series where company_id = 'COMP_0016' order by month desc limit 12
), fan as (
  select month, null::float as score, median, lo50, hi50, lo80, hi80, naive_last
  from api.forecast_rows where company_id = 'COMP_0016'
)
select * from hist union all select * from fan order by month;

-- K. alerts per month and kind
select month, kind, count(*)::int as n from api.alerts group by 1, 2 order by 1 limit 200;
```

Records (invoices, transactions, balances, debt) are not in `api`: use `query_clean_db` (schema `clean`), when the records are mounted. If it answers `records not mounted`, tell the user record-level questions are not available yet and offer the score-level view of the same question.
