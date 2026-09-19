#!/usr/bin/env python3
"""Export clean DuckDB tables to CSV for Postgres COPY. Does not export zip duplicates."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb

HANDOFF = Path(os.environ.get("EMBAT_HANDOFF", "/Users/ruben/Desktop/Embat-handoff"))


def copy_csv(con: duckdb.DuckDBPyConnection, sql: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY ({sql}) TO '{dest.as_posix()}' (HEADER, DELIMITER ',', QUOTE '\"', ESCAPE '\"', NULL '')"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duckdb", default=str(HANDOFF / "embat_clean.duckdb"))
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / ".staging" / "csv"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(args.duckdb, read_only=True)

    copy_csv(con, "SELECT group_id, erp, n_companies_in_sample FROM clean.groups", out / "groups.csv")
    copy_csv(
        con,
        "SELECT company_id, group_id, country, currency, erp, created_at FROM clean.companies",
        out / "companies.csv",
    )
    copy_csv(
        con,
        """
        SELECT product_id, company_id, 'banking' AS family, label, type, bank_name, service, currency,
               created_at, NULL::DOUBLE AS granted, NULL::DOUBLE AS outstanding, NULL::DOUBLE AS liquidity,
               created_after_snapshot, NULL::BOOLEAN AS outstanding_gt_granted
        FROM clean.banking_products
        UNION ALL
        SELECT product_id, company_id, 'debt', label, type, bank_name, service, currency,
               created_at, granted, outstanding, liquidity, created_after_snapshot, outstanding_gt_granted
        FROM clean.debt_products
        UNION ALL
        SELECT p.product_id, p.company_id, 'unknown', NULL, NULL, NULL, NULL, NULL,
               NULL, NULL, NULL, NULL, NULL, NULL
        FROM (
          SELECT product_id, min(company_id) AS company_id FROM clean.transactions GROUP BY 1
          UNION
          SELECT product_id, min(company_id) FROM clean.balances GROUP BY 1
        ) p
        WHERE p.product_id NOT IN (
          SELECT product_id FROM clean.banking_products
          UNION
          SELECT product_id FROM clean.debt_products
        )
        """,
        out / "products.csv",
    )
    copy_csv(
        con,
        """
        SELECT product_id, company_id, settlement_product_id, currency, amortization_type,
               interest_calc_method, amortising_frequency, granted_balance, outstanding_balance,
               total_periods, next_payment_date, last_payment_date, annual_interest_rate_or_spread,
               interest_type, outstanding_gt_granted
        FROM clean.debt_schedule_config
        """,
        out / "debt_schedule.csv",
    )
    copy_csv(
        con,
        """
        SELECT DISTINCT counterparty_id,
               bool_or(src = 'tx') AS seen_in_transactions,
               bool_or(src = 'inv') AS seen_in_invoices
        FROM (
          SELECT counterparty_id, 'tx' AS src FROM clean.transactions WHERE counterparty_id IS NOT NULL
          UNION ALL
          SELECT counterparty_id, 'inv' FROM clean.invoices WHERE counterparty_id IS NOT NULL
        )
        GROUP BY 1
        """,
        out / "counterparties.csv",
    )
    copy_csv(
        con,
        """
        SELECT product_id, company_id, CAST("date" AS DATE) AS as_of_date, balance, granted, liquidity,
               countable, balance_sentinel, product_known
        FROM clean.balances
        """,
        out / "balances.csv",
    )
    copy_csv(
        con,
        """
        SELECT transaction_id, company_id, product_id, "date" AS booked_at, value_date, amount,
               exchange_rate, status, accounting_status, category, description, counterparty_id,
               is_dup, is_extreme, product_known, out_of_window
        FROM clean.transactions
        """,
        out / "transactions.csv",
    )
    copy_csv(
        con,
        """
        SELECT operation_id, company_id, document_type, issuance_date, due_date, payment_date, amount,
               pending_amount, currency, accounting_currency, exchange_rate, status, concept,
               counterparty_id, payment_date_invalid, is_extreme, issued_after_snapshot
        FROM clean.invoices
        """,
        out / "invoices.csv",
    )
    copy_csv(
        con,
        """
        SELECT table_name, issue, action, rows_affected, source, rows_in_raw_table, pct
        FROM clean.dq_log
        """,
        out / "dq_log.csv",
    )
    print(f"exported core csv -> {out}")
    con.close()


if __name__ == "__main__":
    main()
