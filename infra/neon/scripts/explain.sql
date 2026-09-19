-- Main dashboard / repository queries. Run after ANALYZE.

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM api.dashboard_companies ORDER BY score DESC;

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM api.current_index;

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM api.score_series WHERE company_id = 'COMP_0001';

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM api.top_reasons WHERE company_id = 'COMP_0001' AND month = '2026-08';

EXPLAIN (ANALYZE, BUFFERS)
SELECT * FROM api.alerts WHERE entity_id = 'COMP_0001';
