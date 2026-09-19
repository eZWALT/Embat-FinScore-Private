-- How big each company is, from its own bank records. One row per company per score run, so a reload of the run drops it.
-- Lets the app size commercial offers without loading core.transactions (2.5M rows). Filled by
-- infra/neon/scripts/export_company_size.py. Amounts are in the company's currency, not converted.

CREATE TABLE IF NOT EXISTS analytics.company_size (
  run_id uuid NOT NULL REFERENCES analytics.score_runs (run_id) ON DELETE CASCADE,
  company_id text NOT NULL,
  -- Last month of the window, YYYY-MM, and how many months it averages.
  as_of_month text NOT NULL,
  window_months smallint NOT NULL,
  -- Mean monthly operating inflow / outflow over the window (categories as in analysis/features/common.py CAT_MAP).
  monthly_inflow double precision NOT NULL,
  monthly_outflow double precision NOT NULL,
  -- Sum of the positive account balances on the latest snapshot. Null without a snapshot.
  cash double precision,
  currency text,
  PRIMARY KEY (run_id, company_id)
);

-- Same access rule as the other analytics tables (006): server-only.
ALTER TABLE analytics.company_size ENABLE ROW LEVEL SECURITY;
