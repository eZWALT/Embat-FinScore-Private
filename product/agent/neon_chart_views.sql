-- PROPOSED migration (not applied, not run against Neon): tidy "one row per point" views over the current run, for the chat that draws charts.
-- Would live in infra/neon/migrations/007_chart_views.sql. Syntax checked with pglast (Postgres 17 parser); semantics follow 003_analytics.sql
-- and 005_api_views.sql. Every view reads the latest run only (api.current_run), like the existing api.* views.
--
-- Why: api.score_series is already tidy, but items, categories, control charts, forecasts, group series and cluster comparisons are stored as
-- arrays or normalised child tables. A chart wants rows: (month, entity, value). These views give the agent's SQL tool exactly that, and keep
-- the agent away from analytics.* and core.* (the SQL guard can allow the api schema only for chart data).

-- score categories per company-month
CREATE OR REPLACE VIEW api.score_categories_long AS
SELECT s.company_id, s.month, c.category_id, c.score, c.contribution
FROM analytics.score_categories c
JOIN analytics.company_scores s ON s.score_id = c.score_id
JOIN api.current_run r ON r.run_id = s.run_id;

-- score items per company-month (present for the last detail_months only)
CREATE OR REPLACE VIEW api.score_items_long AS
SELECT s.company_id, s.month, i.item_id, i.value, i.points, i.contribution, i.delta
FROM analytics.score_items i
JOIN analytics.company_scores s ON s.score_id = i.score_id
JOIN api.current_run r ON r.run_id = s.run_id;

-- reasons (level = why the score is not higher, change = what moved since last month)
CREATE OR REPLACE VIEW api.score_reasons_long AS
SELECT s.company_id, s.month, x.kind, x.position, x.item, x.label, x.points, x.value, x.unit, x.eur, x.sentence
FROM analytics.score_reasons x
JOIN analytics.company_scores s ON s.score_id = x.score_id
JOIN api.current_run r ON r.run_id = s.run_id;

-- control charts, one row per month of each chart (company and group charts)
CREATE OR REPLACE VIEW api.control_chart_rows AS
SELECT c.entity_type, c.entity_id, c.comparison, c.metric,
       t.month, t.value, t.center, t.lower, t.upper, t.ewma, t.signal, t.persistent
FROM analytics.control_charts c
JOIN api.current_run r ON r.run_id = c.run_id
CROSS JOIN LATERAL unnest(c.months, c."values", c.center, c.lower, c.upper, c.ewma, c.signal, c.persistent)
     AS t(month, value, center, lower, upper, ewma, signal, persistent);

-- score forecast fan, one row per future month, with the naive last value and the origin
CREATE OR REPLACE VIEW api.forecast_rows AS
SELECT f.company_id, f.method, f.origin_month, f.naive_last, f.skill_vs_naive,
       p.month, p.median, p.lo50, p.hi50, p.lo80, p.hi80
FROM analytics.forecasts f
JOIN api.current_run r ON r.run_id = f.run_id
JOIN analytics.forecast_points p ON p.forecast_id = f.forecast_id;

-- group mean score by month (mean_scores is aligned with the run's months)
CREATE OR REPLACE VIEW api.group_series AS
SELECT g.group_id, g.n_companies, g.limits_available, t.month, t.mean_score
FROM analytics.groups_index g
JOIN api.current_run r ON r.run_id = g.run_id
CROSS JOIN LATERAL unnest(r.months, g.mean_scores) AS t(month, mean_score);

-- group membership (join to api.score_series for member scores over time)
CREATE OR REPLACE VIEW api.group_members_v AS
SELECT m.group_id, m.company_id
FROM analytics.group_members m
JOIN api.current_run r ON r.run_id = m.run_id;

-- company profile with its behaviour cluster
CREATE OR REPLACE VIEW api.company_profile AS
SELECT p.company_id, p.group_id, p.country, p.currency, p.erp, p.first_month, p.latest_month,
       p.cluster_id, cl.label AS cluster_label, cl.description AS cluster_description
FROM analytics.company_profiles p
JOIN api.current_run r ON r.run_id = p.run_id
LEFT JOIN analytics.clusters cl ON cl.run_id = p.run_id AND cl.cluster_id = p.cluster_id;

-- percentile of each company inside its behaviour cluster (latest month)
CREATE OR REPLACE VIEW api.cluster_vs_rows AS
SELECT v.company_id, cc.cluster_id, v.metric, v.percentile, v.robust_z
FROM analytics.cluster_vs v
JOIN api.current_run r ON r.run_id = v.run_id
JOIN analytics.company_cluster cc ON cc.run_id = v.run_id AND cc.company_id = v.company_id;

-- alert reasons (with the euro amount behind each)
CREATE OR REPLACE VIEW api.alert_reasons_v AS
SELECT a.alert_id, x.position, x.item, x.label, x.points, x.value, x.unit, x.eur, x.sentence
FROM analytics.alert_reasons x
JOIN analytics.alerts a ON a.run_id = x.run_id AND a.alert_id = x.alert_id
JOIN api.current_run r ON r.run_id = a.run_id;

GRANT SELECT ON ALL TABLES IN SCHEMA api TO CURRENT_USER;
