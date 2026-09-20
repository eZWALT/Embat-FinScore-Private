# Capa 7 — RECORDS (`clean.*` / Neon `core`; solo Ask)

Solo estas tablas, prefijo `clean.`. Solo `SELECT`. Siempre `WHERE company_id` (o una lista) y un `LIMIT`. El dinero va en `currency` de cada fila. Fechas = timestamps; `date_trunc('month', col)` para meses. Ventana: 2024-09-01 → 2026-09-01. Responde en español; los nombres de columna se quedan en inglés.

| Tabla | Filas | Columnas | Notas |
|---|---|---|---|
| `clean.companies` | 1.286 | `company_id, group_id, country, currency, erp, created_at` | `created_at` es el alta en Embat, no la edad de la empresa. |
| `clean.groups` | 250 | `group_id, erp, n_companies_in_sample` | |
| `clean.transactions` | 2.556.068 | `transaction_id, company_id, product_id, date, value_date, amount, exchange_rate, status, accounting_status, category, description, counterparty_id, is_dup, is_extreme, product_known` | `amount` con signo (negativo = salida). Categorías: `uncategorized, collection, payment, utility, fee, transfer, bulk_collection, tax, cash_settlement, pos_settlement, salary, bulk_payment, social_security, debt_repayment, cash_withdrawal, collection_refund, interest_charge, pos_withdrawal, investment_deployment, investment_return, payment_refund, tax_refund`. `transfer` = movimientos entre cuentas propias: exclúyelos del flujo operativo. `fee` + `interest_charge` = coste financiero. `debt_repayment` = servicio de deuda. |
| `clean.invoices` | 896.711 | `operation_id, company_id, document_type, issuance_date, due_date, payment_date, amount, pending_amount, currency, accounting_currency, exchange_rate, status, concept, counterparty_id, payment_date_invalid, is_extreme` | El signo es la dirección: `amount > 0` = emitida a un cliente (cobro); `amount < 0` = recibida de un proveedor (pago). Usa `abs(amount)` para tamaños. Libro: `document_type = 'invoice' and status <> 'cancel' and amount <> 0`. `document_type`: `invoice, paymentDocument, note` (abono), `deposit, invoiceGroup, deliveryNote, refund, purchaseOrder, other, cheque`. `status`: `paid, overdue, pending, cancel, payment_in_progress, paymentOrder, shipped`. Vencida = `status = 'overdue'` o (`due_date < hoy` y `pending_amount > 0`). |
| `clean.balances` | 7.996 | `product_id, company_id, date, balance, granted, liquidity, countable, balance_sentinel, product_known` | Solo el snapshot del 2026-09-01. `liquidity` = caja disponible en el producto. |
| `clean.banking_products` | 5.987 | `product_id, company_id, label, type, bank_name, service, currency, created_at, created_after_snapshot` | `type`: `checking, card, investment, wallet, risk, tpv, expensesPlatform, lineofcomex, saving`. |
| `clean.debt_products` | 2.239 | `product_id, company_id, label, type, bank_name, service, currency, created_at, granted, outstanding, liquidity, created_after_snapshot, outstanding_gt_granted` | `type`: `loan, lineofcredit, confirming, leasing, guarantee, mortgage, renting, factoring`. `outstanding` es negativo (pasivo). |
| `clean.debt_schedule_config` | 87 | `product_id, company_id, settlement_product_id, currency, amortization_type, interest_calc_method, amortising_frequency, granted_balance, outstanding_balance, total_periods, next_payment_date, last_payment_date, annual_interest_rate_or_spread, interest_type, outstanding_gt_granted` | Solo 87 préstamos / 40 empresas tienen tipo. |
| `clean.dq_log` | 26 | `table_name, issue, action, rows_affected, rows_in_raw_table, pct` | Qué hizo la limpieza (ceros, fechas imposibles, extremos). |

Patrones útiles:

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
