-- Apply AFTER bulk COPY.

CREATE INDEX IF NOT EXISTS transactions_company_date_idx
  ON core.transactions (company_id, booked_at);
CREATE INDEX IF NOT EXISTS transactions_product_idx
  ON core.transactions (product_id);
CREATE INDEX IF NOT EXISTS transactions_counterparty_idx
  ON core.transactions (counterparty_id)
  WHERE counterparty_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS transactions_natural_id_uidx
  ON core.transactions (transaction_id)
  WHERE transaction_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS invoices_company_issued_idx
  ON core.invoices (company_id, issuance_date);
CREATE INDEX IF NOT EXISTS invoices_counterparty_idx
  ON core.invoices (counterparty_id)
  WHERE counterparty_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS invoices_company_operation_uidx
  ON core.invoices (company_id, operation_id)
  WHERE operation_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS balances_company_idx ON core.balances (company_id);
CREATE INDEX IF NOT EXISTS products_company_idx ON core.products (company_id);

CREATE INDEX IF NOT EXISTS company_scores_month_idx
  ON analytics.company_scores (run_id, month);
CREATE INDEX IF NOT EXISTS score_reasons_score_idx
  ON analytics.score_reasons (score_id, kind, position);
CREATE INDEX IF NOT EXISTS alerts_month_idx
  ON analytics.alerts (run_id, month DESC, severity);
CREATE INDEX IF NOT EXISTS alerts_entity_idx
  ON analytics.alerts (run_id, entity_type, entity_id);
CREATE INDEX IF NOT EXISTS control_charts_entity_idx
  ON analytics.control_charts (run_id, entity_type, entity_id);

-- Known-product FK. Unknown product_ids were inserted as family='unknown'.
ALTER TABLE core.transactions
  DROP CONSTRAINT IF EXISTS transactions_product_fk;
ALTER TABLE core.transactions
  ADD CONSTRAINT transactions_product_fk
  FOREIGN KEY (product_id) REFERENCES core.products (product_id);

ALTER TABLE core.balances
  DROP CONSTRAINT IF EXISTS balances_product_fk;
ALTER TABLE core.balances
  ADD CONSTRAINT balances_product_fk
  FOREIGN KEY (product_id) REFERENCES core.products (product_id);

ALTER TABLE core.debt_schedule
  DROP CONSTRAINT IF EXISTS debt_schedule_product_fk;
ALTER TABLE core.debt_schedule
  ADD CONSTRAINT debt_schedule_product_fk
  FOREIGN KEY (product_id) REFERENCES core.products (product_id);

-- Counterparties are not companies. Optional FK only onto the synthesized table.
ALTER TABLE core.transactions
  DROP CONSTRAINT IF EXISTS transactions_counterparty_fk;
ALTER TABLE core.transactions
  ADD CONSTRAINT transactions_counterparty_fk
  FOREIGN KEY (counterparty_id) REFERENCES core.counterparties (counterparty_id);

ALTER TABLE core.invoices
  DROP CONSTRAINT IF EXISTS invoices_counterparty_fk;
ALTER TABLE core.invoices
  ADD CONSTRAINT invoices_counterparty_fk
  FOREIGN KEY (counterparty_id) REFERENCES core.counterparties (counterparty_id);
