-- After COPY into analytics tables + staging.

INSERT INTO analytics.score_categories (score_id, category_id, score, contribution)
SELECT s.score_id, stg.category_id, stg.score, stg.contribution
FROM analytics.stg_score_categories stg
JOIN analytics.company_scores s
  ON s.run_id = stg.run_id AND s.company_id = stg.company_id AND s.month = stg.month;

INSERT INTO analytics.score_items (score_id, item_id, value, points, contribution, delta)
SELECT s.score_id, stg.item_id, stg.value, stg.points, stg.contribution, stg.delta
FROM analytics.stg_score_items stg
JOIN analytics.company_scores s
  ON s.run_id = stg.run_id AND s.company_id = stg.company_id AND s.month = stg.month;

INSERT INTO analytics.score_reasons (score_id, kind, position, item, label, points, value, unit, eur, sentence)
SELECT s.score_id, stg.kind, stg.position, stg.item, stg.label, stg.points, stg.value, stg.unit, stg.eur, stg.sentence
FROM analytics.stg_score_reasons stg
JOIN analytics.company_scores s
  ON s.run_id = stg.run_id AND s.company_id = stg.company_id AND s.month = stg.month;

INSERT INTO analytics.forecast_points (forecast_id, month, median, lo50, hi50, lo80, hi80)
SELECT f.forecast_id, stg.month, stg.median, stg.lo50, stg.hi50, stg.lo80, stg.hi80
FROM analytics.stg_forecast_points stg
JOIN analytics.forecasts f
  ON f.run_id = stg.run_id AND f.company_id = stg.company_id;

TRUNCATE analytics.stg_score_categories, analytics.stg_score_items, analytics.stg_score_reasons, analytics.stg_forecast_points;
