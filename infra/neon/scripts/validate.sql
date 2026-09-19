-- Validation against the handoff snapshot. Run after load.sh.

\echo '=== row counts ==='
SELECT 'core.groups' AS rel, count(*) FROM core.groups
UNION ALL SELECT 'core.companies', count(*) FROM core.companies
UNION ALL SELECT 'core.products', count(*) FROM core.products
UNION ALL SELECT 'core.products_unknown', count(*) FROM core.products WHERE family = 'unknown'
UNION ALL SELECT 'core.transactions', count(*) FROM core.transactions
UNION ALL SELECT 'core.invoices', count(*) FROM core.invoices
UNION ALL SELECT 'core.balances', count(*) FROM core.balances
UNION ALL SELECT 'core.counterparties', count(*) FROM core.counterparties
UNION ALL SELECT 'analytics.company_index', count(*) FROM analytics.company_index
UNION ALL SELECT 'analytics.company_scores', count(*) FROM analytics.company_scores
UNION ALL SELECT 'analytics.groups_index', count(*) FROM analytics.groups_index
UNION ALL SELECT 'analytics.alerts', count(*) FROM analytics.alerts
UNION ALL SELECT 'analytics.clusters', count(*) FROM analytics.clusters
ORDER BY 1;

\echo '=== expected bundle 1286 / 250 / 19658 ==='
SELECT n_companies, n_groups, n_company_months FROM api.current_run;
SELECT count(*) AS index_companies FROM api.current_index;
SELECT count(*) AS scored_months FROM api.current_scores;
SELECT count(*) AS groups FROM analytics.groups_index;

\echo '=== key integrity ==='
SELECT count(*) AS companies_missing_group
FROM core.companies c LEFT JOIN core.groups g ON g.group_id = c.group_id
WHERE c.group_id IS NOT NULL AND g.group_id IS NULL;

SELECT count(*) AS tx_orphan_company
FROM core.transactions t LEFT JOIN core.companies c ON c.company_id = t.company_id
WHERE c.company_id IS NULL;

SELECT count(*) AS tx_orphan_product
FROM core.transactions t LEFT JOIN core.products p ON p.product_id = t.product_id
WHERE p.product_id IS NULL;

SELECT count(*) AS counterparty_equals_company
FROM core.counterparties cp
JOIN core.companies c ON c.company_id = cp.counterparty_id;

SELECT count(*) AS duplicate_tx_id
FROM (
  SELECT transaction_id FROM core.transactions
  WHERE transaction_id IS NOT NULL
  GROUP BY 1 HAVING count(*) > 1
) d;

SELECT min(score) AS min_score, max(score) AS max_score,
       count(*) FILTER (WHERE score < 0 OR score > 100) AS out_of_range
FROM analytics.company_scores;

SELECT count(*) AS index_without_latest_month
FROM analytics.company_index i
LEFT JOIN analytics.company_scores s
  ON s.run_id = i.run_id AND s.company_id = i.company_id AND s.month = i.latest_month
WHERE s.score_id IS NULL;

\echo '=== sizes ==='
SELECT pg_size_pretty(pg_database_size(current_database())) AS database_size;
SELECT n.nspname AS schema, c.relname AS relation,
       pg_size_pretty(pg_total_relation_size(c.oid)) AS total,
       pg_size_pretty(pg_relation_size(c.oid)) AS table_bytes,
       pg_size_pretty(pg_indexes_size(c.oid)) AS indexes
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname IN ('core', 'analytics', 'api') AND c.relkind IN ('r', 'm')
ORDER BY pg_total_relation_size(c.oid) DESC
LIMIT 30;
