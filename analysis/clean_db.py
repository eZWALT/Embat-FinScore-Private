"""Crea el esquema `clean` en data/embat.duckdb a partir de las tablas crudas (`main`).

Politica: `main` no se toca. En `clean` se eliminan solo filas sin informacion (importe 0, nulo,
huerfanas), se anulan (NULL) los campos imposibles y se anaden flags para lo dudoso. Cada regla queda
contada en `clean.dq_log` (columna `source`: clean = reglas vistas en entrenamiento, guard = detectores
de suciedad nueva, load = lo que build_db no pudo cargar tal cual). Un detector con 0 filas tambien
se anota: asi se ve que corrio. Las mismas reglas se aplican a cualquier carpeta de CSV, sin reajustar.

Uso:  python analysis/clean_db.py        (o se llama desde build_db.py / analysis.pipeline)
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

KNOWN_COMPANIES = "(SELECT company_id FROM main.companies WHERE company_id IS NOT NULL)"
# mantener alineado con analysis/features/common.CAT_MAP (+ los valores que clean ya normaliza)
KNOWN_CATEGORIES = (
    "collection", "bulk_collection", "cash_settlement", "cash_settlements", "pos_settlement", "collection_refund",
    "payment", "bulk_payment", "utility", "payment_refund", "salary", "social_security", "tax", "tax_refund",
    "cash_withdrawal", "pos_withdrawal", "fee", "interest_charge", "debt_repayment", "transfer",
    "investment_deployment", "investment_return", "-",
)
KNOWN_INV_TYPES = ("invoice", "credit_note", "note", "refund", "paymentDocument", "deposit", "invoiceGroup",
                   "deliveryNote", "purchaseOrder", "cheque", "other")
KNOWN_INV_STATUS = ("paid", "pending", "overdue", "cancel", "payment_in_progress", "paymentOrder", "shipped")
WINDOW_START = "TIMESTAMP '2024-09-01'"
_BAD_AMOUNT = "(amount IS NULL OR isnan(amount) OR isinf(amount))"


def _in(vals):
    return "(" + ", ".join(f"'{v}'" for v in vals) + ")"


# detectores de suciedad que no vimos en entrenamiento: (tabla, incidencia, accion, SQL de conteo sobre la tabla cruda)
GUARDS = [
    ("transactions", "importe nulo / NaN / infinito", "fila eliminada", f"SELECT count(*) FROM main.transactions WHERE {_BAD_AMOUNT}"),
    ("transactions", "company_id nulo o ausente de companies (huerfana; crearia empresas fantasma en el grid)", "fila eliminada",
     f"SELECT count(*) FROM main.transactions WHERE coalesce(company_id NOT IN {KNOWN_COMPANIES}, TRUE)"),
    ("transactions", "fecha nula", "fila eliminada", 'SELECT count(*) FROM main.transactions WHERE "date" IS NULL'),
    ("transactions", "fecha fuera de la ventana 2024-09-01..2026-09-01", "flag out_of_window (no se elimina; las features recortan a su ventana)",
     f"""SELECT count(*) FROM main.transactions WHERE "date" < {WINDOW_START} OR "date" >= {SNAPSHOT} + INTERVAL 1 DAY"""),
    ("transactions", "transaction_id repetido", "se conserva la primera fila, el resto se elimina",
     """SELECT count(*) FROM (SELECT row_number() OVER (PARTITION BY transaction_id ORDER BY company_id, product_id, "date", amount, description) rn
        FROM main.transactions WHERE transaction_id IS NOT NULL) WHERE rn > 1"""),
    ("transactions", "transaction_id nulo", "se conserva (no se puede deduplicar)", "SELECT count(*) FROM main.transactions WHERE transaction_id IS NULL"),
    ("transactions", "categoria fuera del mapa conocido", "se conserva; las features la cuentan como no clasificada",
     f"SELECT count(*) FROM main.transactions WHERE category IS NOT NULL AND category NOT IN {_in(KNOWN_CATEGORIES)}"),
    ("transactions", "status distinto de booked/pending/vacio", "se conserva",
     "SELECT count(*) FROM main.transactions WHERE status IS NOT NULL AND status NOT IN ('booked', 'pending')"),

    ("invoices", "importe nulo / NaN / infinito", "fila eliminada", f"SELECT count(*) FROM main.invoices WHERE {_BAD_AMOUNT}"),
    ("invoices", "company_id nulo o ausente de companies (huerfana)", "fila eliminada",
     f"SELECT count(*) FROM main.invoices WHERE coalesce(company_id NOT IN {KNOWN_COMPANIES}, TRUE)"),
    ("invoices", "issuance_date nula", "fila eliminada", "SELECT count(*) FROM main.invoices WHERE issuance_date IS NULL"),
    ("invoices", "emision posterior al snapshot", "flag issued_after_snapshot (no se elimina)",
     f"SELECT count(*) FROM main.invoices WHERE issuance_date >= {SNAPSHOT} + INTERVAL 1 DAY"),
    ("invoices", "(company_id, operation_id) repetido", "se conserva la primera fila, el resto se elimina",
     """SELECT count(*) FROM (SELECT row_number() OVER (PARTITION BY company_id, operation_id
        ORDER BY issuance_date, amount, counterparty_id) rn FROM main.invoices WHERE operation_id IS NOT NULL) WHERE rn > 1"""),
    ("invoices", "document_type fuera de la lista conocida", "se conserva",
     f"SELECT count(*) FROM main.invoices WHERE document_type IS NULL OR document_type NOT IN {_in(KNOWN_INV_TYPES)}"),
    ("invoices", "status fuera de la lista conocida", "se conserva",
     f"SELECT count(*) FROM main.invoices WHERE status IS NULL OR status NOT IN {_in(KNOWN_INV_STATUS)}"),

    ("companies", "company_id nulo o repetido", "se conserva la primera fila",
     """SELECT count(*) FROM (SELECT company_id, row_number() OVER (PARTITION BY company_id ORDER BY created_at, group_id) rn
        FROM main.companies) WHERE company_id IS NULL OR rn > 1"""),
    ("companies", "group_id nulo", "se conserva (empresa sin grupo)", "SELECT count(*) FROM main.companies WHERE group_id IS NULL"),
    ("banking_products", "product_id repetido", "se conserva la primera fila",
     "SELECT count(*) - count(DISTINCT product_id) FROM main.banking_products WHERE product_id IS NOT NULL"),
    ("banking_products", "empresa ausente de companies", "se conserva (se ignora al unir por company_id)",
     f"SELECT count(*) FROM main.banking_products WHERE coalesce(company_id NOT IN {KNOWN_COMPANIES}, TRUE)"),
    ("debt_products", "product_id repetido", "se conserva la primera fila",
     "SELECT count(*) - count(DISTINCT product_id) FROM main.debt_products WHERE product_id IS NOT NULL"),
    ("debt_products", "empresa ausente de companies", "se conserva (se ignora al unir por company_id)",
     f"SELECT count(*) FROM main.debt_products WHERE coalesce(company_id NOT IN {KNOWN_COMPANIES}, TRUE)"),
    ("balances", "empresa ausente de companies", "se conserva (se ignora al unir por company_id)",
     f"SELECT count(*) FROM main.balances WHERE coalesce(company_id NOT IN {KNOWN_COMPANIES}, TRUE)"),
    ("balances", "balance nulo", "se conserva NULL (la caja de esa cuenta no se reconstruye)", "SELECT count(*) FROM main.balances WHERE balance IS NULL"),
    ("balances", "saldo con fecha anterior al 2026-08-25 (snapshot desfasado)", "se conserva; la caja se reconstruye hacia atras desde el snapshot",
     f"SELECT count(*) FROM main.balances WHERE \"date\" < {SNAPSHOT} - INTERVAL 7 DAY"),
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
        FROM main.companies WHERE company_id IS NOT NULL
        QUALIFY row_number() OVER (PARTITION BY company_id ORDER BY created_at, group_id) = 1
        ORDER BY company_id""",

    "banking_products": f"""
        SELECT *, created_at > {SNAPSHOT} AS created_after_snapshot
        FROM main.banking_products
        QUALIFY product_id IS NULL OR row_number() OVER (PARTITION BY product_id ORDER BY company_id, created_at) = 1
        ORDER BY company_id, product_id""",

    "debt_products": f"""
        SELECT *, created_at > {SNAPSHOT} AS created_after_snapshot,
               abs(outstanding) > abs(granted) * 1.0001 AS outstanding_gt_granted
        FROM main.debt_products
        QUALIFY product_id IS NULL OR row_number() OVER (PARTITION BY product_id ORDER BY company_id, created_at) = 1
        ORDER BY company_id, product_id""",

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
        WITH ids AS (
          SELECT *, row_number() OVER (PARTITION BY transaction_id ORDER BY company_id, product_id, "date", amount, description) AS id_seq
          FROM main.transactions
          WHERE NOT {_BAD_AMOUNT} AND amount <> 0 AND "date" IS NOT NULL AND company_id IN {KNOWN_COMPANIES}),
        base AS (
          SELECT *, row_number() OVER (PARTITION BY company_id, product_id, "date", value_date, amount, category,
                                       description, counterparty_id, status ORDER BY transaction_id) AS dup_seq
          FROM ids WHERE transaction_id IS NULL OR id_seq = 1)
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
          product_id IN {KNOWN_PRODUCTS} AS product_known,
          ("date" < {WINDOW_START} OR "date" >= {SNAPSHOT} + INTERVAL 1 DAY) AS out_of_window
        FROM base ORDER BY company_id, "date", transaction_id""",

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
          abs(amount) >= 1e9 AS is_extreme,
          issuance_date >= {SNAPSHOT} + INTERVAL 1 DAY AS issued_after_snapshot
        FROM main.invoices
        WHERE NOT {_BAD_AMOUNT} AND amount <> 0 AND issuance_date IS NOT NULL AND company_id IN {KNOWN_COMPANIES}
        QUALIFY operation_id IS NULL OR row_number() OVER (PARTITION BY company_id, operation_id
                                                           ORDER BY issuance_date, amount, counterparty_id) = 1
        ORDER BY company_id, issuance_date, operation_id""",
}


def run(con, load_log=()):
    """`load_log`: filas (tabla, incidencia, accion, n) que build_db anoto al cargar los CSV."""
    con.execute("CREATE SCHEMA IF NOT EXISTS clean")
    log = []
    for source, rules in (("clean", RULES), ("guard", GUARDS)):
        for table, issue, action, sql in rules:
            log.append((table, issue, action, con.execute(sql).fetchone()[0], source))
    for table, issue, action, n in load_log:
        log.append((table, issue, action, n, "load"))
    for table, sql in BUILD.items():
        con.execute(f"CREATE OR REPLACE TABLE clean.{table} AS {sql}")
    df = pd.DataFrame(log, columns=["table_name", "issue", "action", "rows_affected", "source"])
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
