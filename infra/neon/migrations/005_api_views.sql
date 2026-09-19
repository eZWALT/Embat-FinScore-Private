-- Product-facing read models. The Next.js server queries these, never core.*
-- from the browser.

CREATE OR REPLACE VIEW api.current_run AS
SELECT *
FROM analytics.score_runs
ORDER BY loaded_at DESC
LIMIT 1;

CREATE OR REPLACE VIEW api.current_scores AS
SELECT s.*
FROM analytics.company_scores s
JOIN api.current_run r ON r.run_id = s.run_id;

CREATE OR REPLACE VIEW api.current_index AS
SELECT i.*
FROM analytics.company_index i
JOIN api.current_run r ON r.run_id = i.run_id;

CREATE OR REPLACE VIEW api.current_trajectory AS
SELECT
  i.company_id,
  i.group_id,
  i.latest_month,
  i.score,
  i.trajectory,
  i.delta_1m,
  i.delta_3m,
  i.guard
FROM api.current_index i;

CREATE OR REPLACE VIEW api.top_reasons AS
SELECT
  s.company_id,
  s.month,
  r.position,
  r.item,
  r.label,
  r.points,
  r.eur,
  r.sentence
FROM analytics.score_reasons r
JOIN analytics.company_scores s ON s.score_id = r.score_id
JOIN api.current_run run ON run.run_id = s.run_id
WHERE r.kind = 'level'
ORDER BY s.company_id, s.month, r.position;

CREATE OR REPLACE VIEW api.alerts AS
SELECT a.*
FROM analytics.alerts a
JOIN api.current_run r ON r.run_id = a.run_id;

CREATE OR REPLACE VIEW api.score_series AS
SELECT
  s.company_id,
  s.month,
  s.score,
  s.trajectory,
  s.confidence,
  s.guard
FROM api.current_scores s
ORDER BY s.company_id, s.month;

CREATE OR REPLACE VIEW api.dashboard_companies AS
WITH run AS (
  SELECT * FROM api.current_run
),
cats AS (
  SELECT
    s.company_id,
    jsonb_object_agg(c.category_id, jsonb_build_object('score', c.score, 'contribution', c.contribution)) AS categories
  FROM analytics.company_scores s
  JOIN run ON run.run_id = s.run_id
  JOIN analytics.score_categories c ON c.score_id = s.score_id
  JOIN analytics.company_index i
    ON i.run_id = s.run_id AND i.company_id = s.company_id AND i.latest_month = s.month
  GROUP BY s.company_id
),
hist AS (
  SELECT
    s.company_id,
    jsonb_agg(jsonb_build_object('month', s.month, 'score', s.score) ORDER BY s.month) AS score_history
  FROM analytics.company_scores s
  JOIN run ON run.run_id = s.run_id
  GROUP BY s.company_id
)
SELECT
  i.company_id,
  i.group_id,
  p.country,
  p.currency,
  p.erp,
  i.latest_month,
  i.score,
  i.delta_1m,
  i.delta_3m,
  i.trajectory,
  i.confidence,
  s.confidence_note,
  s.coverage,
  i.top_reason,
  i.guard,
  r.item AS reason_item,
  r.points AS reason_points,
  r.eur AS reason_eur,
  h.score_history,
  c.categories
FROM analytics.company_index i
JOIN run ON run.run_id = i.run_id
JOIN analytics.company_profiles p
  ON p.run_id = i.run_id AND p.company_id = i.company_id
JOIN analytics.company_scores s
  ON s.run_id = i.run_id AND s.company_id = i.company_id AND s.month = i.latest_month
JOIN cats c ON c.company_id = i.company_id
JOIN hist h ON h.company_id = i.company_id
LEFT JOIN analytics.score_reasons r
  ON r.score_id = s.score_id AND r.kind = 'level' AND r.position = 0;

CREATE OR REPLACE VIEW api.manifest AS
SELECT
  schema_version,
  scorecard_version,
  generated_at,
  as_of_month,
  months,
  detail_months,
  detail_from_month,
  is_sample,
  jsonb_build_object(
    'companies', n_companies,
    'groups', n_groups,
    'company_months', n_company_months
  ) AS counts,
  spec,
  disclaimer
FROM api.current_run;
