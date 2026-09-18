"""Crea el esquema `clean` en data/embat.duckdb a partir de las tablas crudas (`main`).

Politica: `main` no se toca. En `clean` se eliminan solo filas sin informacion (importe 0),
se anulan (NULL) los campos imposibles y se anaden flags para lo dudoso. Cada regla queda
contada en `clean.dq_log`.

Uso:  python analysis/clean_db.py        (o se llama desde build_db.py)
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "embat.duckdb"
SNAPSHOT = "TIMESTAMP '2026-09-01'"

# (tabla, incidencia, accion, SQL que cuenta las filas afectadas en la tabla cruda)
RULES = [
    ("transactions", "importe = 0", "fila eliminada", "SELECT count(*) FROM main.transactions WHERE amount = 0"),
    ("transactions", "value_date a mas de 60 dias de la fecha (2022, 2099...)", "value_date -> NULL",
     """SELECT count(*) FROM main.transactions WHERE amount <> 0 AND value_date IS NOT NULL
        AND abs(date_diff('day', "date", value_date)) > 60"""),
    ("transactions", "exchange_rate <= 0", "exchange_rate -> NULL", "SELECT count(*) FROM main.transactions WHERE amount <> 0 AND exchange_rate <= 0"),
    ("transactions", "categoria vacia o '-'", "-> 'uncategorized'", "SELECT count(*) FROM main.transactions WHERE amount <> 0 AND (category IS NULL OR category = '-')"),
    ("transactions", "categoria 'cash_settlements' (typo)", "-> 'cash_settlement'", "SELECT count(*) FROM main.transactions WHERE category = 'cash_settlements'"),
    ("transactions", "status vacio", "-> 'unknown'", "SELECT count(*) FROM main.transactions WHERE amount <> 0 AND status IS NULL"),
    ("transactions", "fila repetida (mismo contenido salvo id)", "flag is_dup (no se elimina)",
     """SELECT count(*) FROM (SELECT row_number() OVER (PARTITION BY company_id, product_id, "date", value_date, amount, category,
        description, counterparty_id, status ORDER BY transaction_id) rn FROM main.transactions WHERE amount <> 0) WHERE rn > 1"""),
    ("transactions", "|importe| >= 1e9", "flag is_extreme", "SELECT count(*) FROM main.transactions WHERE abs(amount) >= 1e9"),
    ("transactions", "producto ausente de banking/debt_products", "flag product_known = false",
     """SELECT count(*) FROM main.transactions WHERE product_id NOT IN
        (SELECT product_id FROM main.banking_products UNION SELECT product_id FROM main.debt_products)"""),

    ("invoices", "importe = 0", "fila eliminada", "SELECT count(*) FROM main.invoices WHERE amount = 0"),
    ("invoices", "payment_date en factura no pagada (placeholder = vencimiento)", "payment_date -> NULL",
     "SELECT count(*) FROM main.invoices WHERE amount <> 0 AND status <> 'paid' AND payment_date IS NOT NULL"),
    ("invoices", "factura pagada con payment_date imposible (antes de emision, futura, ano 2000/6913...)", "payment_date -> NULL + flag payment_date_invalid",
     f"""SELECT count(*) FROM main.invoices WHERE amount <> 0 AND status = 'paid'
         AND NOT coalesce(payment_date >= issuance_date AND payment_date <= {SNAPSHOT}, FALSE)"""),
    ("invoices", "vencimiento anterior a la emision", "due_date -> issuance_date", "SELECT count(*) FROM main.invoices WHERE amount <> 0 AND due_date < issuance_date"),
    ("invoices", "vencimiento a mas de 730 dias de la emision (ano 7025...)", "due_date -> NULL",
     "SELECT count(*) FROM main.invoices WHERE amount <> 0 AND due_date > issuance_date + INTERVAL 730 DAY"),
    ("invoices", "|pending_amount| > |amount|", "pending_amount -> amount", "SELECT count(*) FROM main.invoices WHERE amount <> 0 AND abs(pending_amount) > abs(amount)"),
    ("invoices", "exchange_rate <= 0", "exchange_rate -> NULL", "SELECT count(*) FROM main.invoices WHERE amount <> 0 AND currency <> accounting_currency AND exchange_rate <= 0"),
    ("invoices", "misma moneda con exchange_rate <> 1", "exchange_rate -> 1", "SELECT count(*) FROM main.invoices WHERE amount <> 0 AND currency = accounting_currency AND exchange_rate <> 1"),
    ("invoices", "|importe| >= 1e9", "flag is_extreme", "SELECT count(*) FROM main.invoices WHERE abs(amount) >= 1e9"),

    ("companies", "country no es codigo ISO-2 (ESPANA, Spain, Portugal...)", "-> ISO-2 o NULL", "SELECT count(*) FROM main.companies WHERE length(trim(country)) <> 2"),

    ("banking_products", "created_at posterior al snapshot (2026-09-01)", "flag created_after_snapshot", f"SELECT count(*) FROM main.banking_products WHERE created_at > {SNAPSHOT}"),
    ("debt_products", "created_at posterior al snapshot (2026-09-01)", "flag created_after_snapshot", f"SELECT count(*) FROM main.debt_products WHERE created_at > {SNAPSHOT}"),
    ("debt_products", "|outstanding| > |granted|", "flag outstanding_gt_granted", "SELECT count(*) FROM main.debt_products WHERE abs(outstanding) > abs(granted) * 1.0001"),
    ("debt_schedule_config", "outstanding_balance > granted_balance", "flag outstanding_gt_granted", "SELECT count(*) FROM main.debt_schedule_config WHERE outstanding_balance > granted_balance * 1.0001"),

    ("balances", "saldo centinela (-999999999, -1e9, >= 1e10)", "balance -> NULL + flag balance_sentinel",
     "SELECT count(*) FROM main.balances WHERE balance IN (-999999999, -1000000000) OR abs(balance) >= 1e10"),
    ("balances", "producto ausente de banking/debt_products", "flag product_known = false",
     """SELECT count(*) FROM main.balances WHERE product_id NOT IN
        (SELECT product_id FROM main.banking_products UNION SELECT product_id FROM main.debt_products)"""),
    ("balances", "columna 'available' 100% nula", "columna eliminada", "SELECT count(*) FROM main.balances WHERE available IS NOT NULL"),
]

KNOWN_PRODUCTS = "(SELECT product_id FROM main.banking_products UNION SELECT product_id FROM main.debt_products)"

BUILD = {
    "groups": "SELECT * FROM main.groups ORDER BY group_id",

    "companies": """
        SELECT company_id, group_id,
          CASE WHEN country IS NULL THEN NULL
               WHEN length(trim(country)) = 2 THEN upper(trim(country))
               WHEN regexp_matches(lower(trim(country)), '^(espa.{1,3}a|espanya|spain)$') THEN 'ES'
               WHEN lower(trim(country)) = 'portugal' THEN 'PT'
               WHEN lower(trim(country)) = 'italia' THEN 'IT'
               WHEN lower(trim(country)) = 'alemania' THEN 'DE'
               WHEN lower(trim(country)) = 'malaysia' THEN 'MY'
          END AS country,
          currency, erp, created_at
        FROM main.companies ORDER BY company_id""",

    "banking_products": f"""
        SELECT *, created_at > {SNAPSHOT} AS created_after_snapshot
        FROM main.banking_products ORDER BY company_id, product_id""",

    "debt_products": f"""
        SELECT *, created_at > {SNAPSHOT} AS created_after_snapshot,
               abs(outstanding) > abs(granted) * 1.0001 AS outstanding_gt_granted
        FROM main.debt_products ORDER BY company_id, product_id""",

    "debt_schedule_config": """
        SELECT *, outstanding_balance > granted_balance * 1.0001 AS outstanding_gt_granted
        FROM main.debt_schedule_config ORDER BY company_id, product_id""",

    "balances": f"""
        SELECT product_id, company_id, "date",
          CASE WHEN balance IN (-999999999, -1000000000) OR abs(balance) >= 1e10 THEN NULL ELSE balance END AS balance,
          granted, liquidity, countable,
          (balance IN (-999999999, -1000000000) OR abs(balance) >= 1e10) AS balance_sentinel,
          product_id IN {KNOWN_PRODUCTS} AS product_known
        FROM main.balances ORDER BY company_id, product_id""",

    "transactions": f"""
        WITH base AS (
          SELECT *, row_number() OVER (PARTITION BY company_id, product_id, "date", value_date, amount, category,
                                       description, counterparty_id, status ORDER BY transaction_id) AS dup_seq
          FROM main.transactions WHERE amount <> 0)
        SELECT transaction_id, company_id, product_id, "date",
          CASE WHEN value_date IS NOT NULL AND abs(date_diff('day', "date", value_date)) <= 60 THEN value_date END AS value_date,
          amount,
          CASE WHEN exchange_rate > 0 THEN exchange_rate END AS exchange_rate,
          coalesce(status, 'unknown') AS status, accounting_status,
          CASE WHEN category IS NULL OR category = '-' THEN 'uncategorized'
               WHEN category = 'cash_settlements' THEN 'cash_settlement' ELSE category END AS category,
          description, counterparty_id,
          dup_seq > 1 AS is_dup,
          abs(amount) >= 1e9 AS is_extreme,
          product_id IN {KNOWN_PRODUCTS} AS product_known
        FROM base ORDER BY company_id, "date" """,

    "invoices": f"""
        SELECT operation_id, company_id, document_type, issuance_date,
          CASE WHEN due_date IS NULL THEN NULL
               WHEN due_date < issuance_date THEN issuance_date
               WHEN due_date > issuance_date + INTERVAL 730 DAY THEN NULL
               ELSE due_date END AS due_date,
          CASE WHEN status = 'paid' AND payment_date >= issuance_date AND payment_date <= {SNAPSHOT} THEN payment_date END AS payment_date,
          amount,
          CASE WHEN abs(pending_amount) > abs(amount) THEN amount ELSE pending_amount END AS pending_amount,
          currency, accounting_currency,
          CASE WHEN currency = accounting_currency THEN 1.0 WHEN exchange_rate > 0 THEN exchange_rate END AS exchange_rate,
          status, concept, counterparty_id,
          (status = 'paid' AND NOT coalesce(payment_date >= issuance_date AND payment_date <= {SNAPSHOT}, FALSE)) AS payment_date_invalid,
          abs(amount) >= 1e9 AS is_extreme
        FROM main.invoices WHERE amount <> 0 ORDER BY company_id, issuance_date""",
}


def run(con):
    con.execute("CREATE SCHEMA IF NOT EXISTS clean")
    log = []
    for table, issue, action, sql in RULES:
        log.append((table, issue, action, con.execute(sql).fetchone()[0]))
    for table, sql in BUILD.items():
        con.execute(f"CREATE OR REPLACE TABLE clean.{table} AS {sql}")
    df = pd.DataFrame(log, columns=["table_name", "issue", "action", "rows_affected"])
    tot = {t: con.execute(f"SELECT count(*) FROM main.{t}").fetchone()[0] for t in BUILD}
    df["rows_in_raw_table"] = df["table_name"].map(tot)
    df["pct"] = (100 * df["rows_affected"] / df["rows_in_raw_table"]).round(3)
    con.register("dq_df", df)
    con.execute("CREATE OR REPLACE TABLE clean.dq_log AS SELECT * FROM dq_df")
    return df


def main():
    con = duckdb.connect(str(DB_PATH))
    df = run(con)
    pd.set_option("display.width", 250, "display.max_colwidth", 95, "display.max_rows", 100)
    print(df[["table_name", "issue", "action", "rows_affected", "pct"]].to_string(index=False))
    print("\nFilas raw -> clean:")
    for t in BUILD:
        a = con.execute(f"SELECT count(*) FROM main.{t}").fetchone()[0]
        b = con.execute(f"SELECT count(*) FROM clean.{t}").fetchone()[0]
        print(f"  {t:22s} {a:>10,d} -> {b:>10,d}  ({b - a:+,d})")
    con.close()


if __name__ == "__main__":
    sys.exit(main())
