#!/usr/bin/env bash
# Load CSVs into Neon. Idempotent full refresh of selected schemas.
# Usage: DATABASE_URL=... ./load.sh [--analytics-only|--core-only]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CSV="${CSV_DIR:-$ROOT/.staging/csv}"
MIG="$ROOT/migrations"

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is required (unpooled Neon connection string)." >&2
  exit 1
fi

MODE="${1:-all}"
PSQL=(psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q)

copy_table() {
  local table="$1"
  local file="$2"
  local cols="$3"
  if [[ ! -f "$file" ]]; then
    echo "missing $file" >&2
    exit 1
  fi
  echo "COPY $table <- $(basename "$file")"
  "${PSQL[@]}" -c "\\copy $table ($cols) FROM '$file' WITH (FORMAT csv, HEADER true, NULL '')"
}

echo "applying schema 001-003"
"${PSQL[@]}" -f "$MIG/001_schemas.sql"
"${PSQL[@]}" -f "$MIG/002_core.sql"
"${PSQL[@]}" -f "$MIG/003_analytics.sql"

if [[ "$MODE" != "--analytics-only" ]]; then
  echo "refresh core"
  "${PSQL[@]}" -c "TRUNCATE core.dq_log, core.transactions, core.invoices, core.balances, core.debt_schedule, core.counterparties, core.products, core.companies, core.groups, core.load_batches RESTART IDENTITY CASCADE;"
  copy_table core.groups "$CSV/groups.csv" "group_id,erp,n_companies_in_sample"
  copy_table core.companies "$CSV/companies.csv" "company_id,group_id,country,currency,erp,created_at"
  copy_table core.products "$CSV/products.csv" "product_id,company_id,family,label,type,bank_name,service,currency,created_at,granted,outstanding,liquidity,created_after_snapshot,outstanding_gt_granted"
  copy_table core.debt_schedule "$CSV/debt_schedule.csv" "product_id,company_id,settlement_product_id,currency,amortization_type,interest_calc_method,amortising_frequency,granted_balance,outstanding_balance,total_periods,next_payment_date,last_payment_date,annual_interest_rate_or_spread,interest_type,outstanding_gt_granted"
  copy_table core.counterparties "$CSV/counterparties.csv" "counterparty_id,seen_in_transactions,seen_in_invoices"
  copy_table core.balances "$CSV/balances.csv" "product_id,company_id,as_of_date,balance,granted,liquidity,countable,balance_sentinel,product_known"
  copy_table core.transactions "$CSV/transactions.csv" "transaction_id,company_id,product_id,booked_at,value_date,amount,exchange_rate,status,accounting_status,category,description,counterparty_id,is_dup,is_extreme,product_known,out_of_window"
  copy_table core.invoices "$CSV/invoices.csv" "operation_id,company_id,document_type,issuance_date,due_date,payment_date,amount,pending_amount,currency,accounting_currency,exchange_rate,status,concept,counterparty_id,payment_date_invalid,is_extreme,issued_after_snapshot"
  copy_table core.dq_log "$CSV/dq_log.csv" "table_name,issue,action,rows_affected,source,rows_in_raw_table,pct"
  "${PSQL[@]}" -c "INSERT INTO core.load_batches (source_duckdb, note) VALUES ('embat_clean.duckdb', '$MODE');"
fi

if [[ "$MODE" != "--core-only" ]]; then
  echo "refresh analytics"
  "${PSQL[@]}" -c "TRUNCATE analytics.score_runs RESTART IDENTITY CASCADE;"
  "${PSQL[@]}" -c "TRUNCATE analytics.stg_score_categories, analytics.stg_score_items, analytics.stg_score_reasons, analytics.stg_forecast_points;"
  copy_table analytics.score_runs "$CSV/score_runs.csv" "run_id,schema_version,scorecard_version,generated_at,as_of_month,months,detail_months,detail_from_month,is_sample,n_companies,n_groups,n_company_months,source,spec,monitor,sections,reference,files,disclaimer"
  copy_table analytics.clusters "$CSV/clusters.csv" "run_id,cluster_id,label,description,n_companies,n_companies_train_fit"
  copy_table analytics.cluster_quality "$CSV/cluster_quality.csv" "run_id,chosen_k,silhouette,note"
  copy_table analytics.company_profiles "$CSV/company_profiles.csv" "run_id,company_id,group_id,country,currency,erp,first_month,latest_month,cluster_id,alert_ids"
  copy_table analytics.company_index "$CSV/company_index.csv" "run_id,company_id,group_id,latest_month,score,trajectory,confidence,guard,delta_1m,delta_3m,top_reason,cluster_id,n_alerts,max_alert_severity,sparkline"
  copy_table analytics.company_scores "$CSV/company_scores.csv" "run_id,company_id,month,score,score_pre_cap,guard,guard_adjustment,trajectory,slope3,slope6,confidence,confidence_note,coverage,trail_months,change_guard"
  copy_table analytics.stg_score_categories "$CSV/score_categories.csv" "run_id,company_id,month,category_id,score,contribution"
  copy_table analytics.stg_score_items "$CSV/score_items.csv" "run_id,company_id,month,item_id,value,points,contribution,delta"
  copy_table analytics.stg_score_reasons "$CSV/score_reasons.csv" "run_id,company_id,month,kind,position,item,label,points,value,unit,eur,sentence"
  copy_table analytics.company_cluster "$CSV/company_cluster.csv" "run_id,company_id,cluster_id,month"
  copy_table analytics.cluster_vs "$CSV/cluster_vs.csv" "run_id,company_id,metric,percentile,robust_z"
  copy_table analytics.control_charts "$CSV/control_charts.csv" "run_id,entity_type,entity_id,comparison,metric,months,values,center,lower,upper,ewma,cusum_low,cusum_high,signal,persistent,method"
  copy_table analytics.forecasts "$CSV/forecasts.csv" "run_id,company_id,metric,method,origin_month,horizon_months,naive_last,skill_vs_naive,note"
  copy_table analytics.stg_forecast_points "$CSV/forecast_points.csv" "run_id,company_id,month,median,lo50,hi50,lo80,hi80"
  copy_table analytics.alerts "$CSV/alerts.csv" "run_id,alert_id,entity_type,entity_id,entity_name,month,kind,direction,severity,title,summary,owner,action,persistence_rule,months_flagged,rank_score,evidence"
  copy_table analytics.alert_reasons "$CSV/alert_reasons.csv" "run_id,alert_id,position,item,label,points,value,unit,eur,sentence"
  copy_table analytics.groups_index "$CSV/groups_index.csv" "run_id,group_id,n_companies,latest_mean_score,latest_min_score,latest_min_company_id,mean_scores,limits_available,alert_ids"
  copy_table analytics.group_members "$CSV/group_members.csv" "run_id,group_id,company_id"
  echo "map score children"
  "${PSQL[@]}" -f "$MIG/003b_map_score_children.sql"
fi

echo "indexes, views, security"
"${PSQL[@]}" -f "$MIG/004_indexes_fks.sql"
"${PSQL[@]}" -f "$MIG/005_api_views.sql"
"${PSQL[@]}" -f "$MIG/006_security.sql"
"${PSQL[@]}" -c "ANALYZE;"
echo "load complete"
