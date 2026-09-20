# Capa 7 — RECORDS (`clean.*` / Neon `core`; solo Ask)

Solo estas tablas, prefijo `clean.`. Solo `SELECT`. Siempre `WHERE company_id` (o una lista) y un `LIMIT`. El dinero va en `currency` de cada fila. Fechas = timestamps; `date_trunc('month', col)` para meses. Ventana: 2024-09-01 → 2026-09-01. Responde en español; los nombres de columna se quedan en inglés.

| Table | Rows | Columns | Notes |
|---|---|---|---|
| `clean.companies` | 1,286 | `company_id, group_id, country, currency, erp, created_at` | `created_at` is the Embat onboarding date, not the company's age. |
| `clean.groups` | 250 | `group_id, erp, n_companies_in_sample` | |
| `clean.transactions` | 2,556,068 | `transaction_id, company_id, product_id, date, value_date, amount, exchange_rate, status, accounting_status, category, description, counterparty_id, is_dup, is_extreme, product_known` | `amount` signed (negative = outflow). `category` values: `uncategorized, collection, payment, utility, fee, transfer, bulk_collection, tax, cash_settlement, pos_settlement, salary, bulk_payment, social_security, debt_repayment, cash_withdrawal, collection_refund, interest_charge, pos_withdrawal, investment_deployment, investment_return, payment_refund, tax_refund`. `transfer` = own-account moves; exclude for operating flows. `fee` + `interest_charge` = financing cost. `debt_repayment` = debt service. |
| `clean.invoices` | 896,711 | `operation_id, company_id, document_type, issuance_date, due_date, payment_date, amount, pending_amount, currency, accounting_currency, exchange_rate, status, concept, counterparty_id, payment_date_invalid, is_extreme` | **Direction is the sign**: `amount > 0` = issued to a customer (receivable, AR); `amount < 0` = received from a supplier (payable, AP). This is how the feature pipeline splits them (confirmed against matched bank transactions). Use `abs(amount)` for sizes. The core invoice book is `document_type = 'invoice' and status <> 'cancel' and amount <> 0`. `document_type`: `invoice, paymentDocument, note (credit note), deposit, invoiceGroup, deliveryNote, refund, purchaseOrder, other, cheque`. `status`: `paid, overdue, pending, cancel, payment_in_progress, paymentOrder, shipped`. Overdue = `status = 'overdue'` or (`due_date < today and pending_amount > 0`). |
| `clean.balances` | 7,996 | `product_id, company_id, date, balance, granted, liquidity, countable, balance_sentinel, product_known` | Snapshot as of 2026-09-01 only. `liquidity` = available cash on the product. |
| `clean.banking_products` | 5,987 | `product_id, company_id, label, type, bank_name, service, currency, created_at, created_after_snapshot` | `type`: `checking, card, investment, wallet, risk, tpv, expensesPlatform, lineofcomex, saving`. |
| `clean.debt_products` | 2,239 | `product_id, company_id, label, type, bank_name, service, currency, created_at, granted, outstanding, liquidity, created_after_snapshot, outstanding_gt_granted` | `type`: `loan, lineofcredit, confirming, leasing, guarantee, mortgage, renting, factoring`. `outstanding` is negative (a liability). |
| `clean.debt_schedule_config` | 87 | `product_id, company_id, settlement_product_id, currency, amortization_type, interest_calc_method, amortising_frequency, granted_balance, outstanding_balance, total_periods, next_payment_date, last_payment_date, annual_interest_rate_or_spread, interest_type, outstanding_gt_granted` | Only 87 loans across 40 companies carry a rate. |
| `clean.dq_log` | 26 | `table_name, issue, action, rows_affected, rows_in_raw_table, pct` | What cleaning did (dropped zero-amount rows, nulled impossible dates, flagged extremes). |

Useful patterns:

```sql
-- monthly operating inflows / outflows
select date_trunc('month', date) m,
       sum(case when amount > 0 then amount end) inflow,
       sum(case when amount < 0 then -amount end) outflow
from clean.transactions
where company_id = ? and category <> 'transfer'
group by 1 order by 1 limit 30;

-- open receivables by counterparty, overdue first
select counterparty_id, count(*) n, sum(pending_amount) open_amount,
       min(due_date) oldest_due
from clean.invoices
where company_id = ? and amount > 0 and pending_amount > 0
group by 1 order by open_amount desc limit 20;

-- what did last quarter's top customer bill by month
select date_trunc('month', issuance_date) m, sum(amount)
from clean.invoices
where company_id = ? and counterparty_id = ? and amount > 0
group by 1 order by 1 limit 24;
```
