"""Carga los CSV de una carpeta en una base DuckDB y crea el esquema `clean` (+ clean.dq_log).

Uso:  python analysis/build_db.py [--force] [--csv-dir DIR] [--db PATH]
Requiere:  pip install duckdb

Los tipos de cada columna estan fijados aqui (no se autodetectan), asi una carpeta nueva se
carga igual que la de entrenamiento. Lo que no encaja (columna ausente, fila que no parsea,
archivo ausente) se anota en clean.dq_log en vez de perderse.
"""
import argparse
import sys
import time
from pathlib import Path

import duckdb

try:
    from analysis import clean_db
except ImportError:  # ejecutado como script: python analysis/build_db.py
    import clean_db

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = DATA / "embat.duckdb"


def find_csv_dir():
    """data/ o, si se descargo el zip del reto, data/raw/output/ (ver data/README.md)."""
    return next((d for d in (DATA, DATA / "raw" / "output") if (d / "transactions.csv").exists()), None)


# tabla -> (csv, orden fisico para que DuckDB pode bloques al filtrar por empresa/fecha, columnas y tipos,
#           columnas clave: si faltan no hay pipeline posible)
# El orden incluye siempre el id para que la carga sea determinista (empates entre hilos).
TABLES = {
    "groups": ("groups.csv", "group_id", {
        "group_id": "VARCHAR", "erp": "VARCHAR", "n_companies_in_sample": "BIGINT"},
        ["group_id"]),
    "companies": ("companies.csv", "company_id", {
        "company_id": "VARCHAR", "group_id": "VARCHAR", "country": "VARCHAR", "currency": "VARCHAR",
        "erp": "VARCHAR", "created_at": "TIMESTAMP"},
        ["company_id"]),
    "banking_products": ("banking_products.csv", "company_id, product_id", {
        "product_id": "VARCHAR", "company_id": "VARCHAR", "label": "VARCHAR", "type": "VARCHAR",
        "bank_name": "VARCHAR", "service": "VARCHAR", "currency": "VARCHAR", "created_at": "TIMESTAMP"},
        ["product_id", "company_id", "type"]),
    "debt_products": ("debt_products.csv", "company_id, product_id", {
        "product_id": "VARCHAR", "company_id": "VARCHAR", "label": "VARCHAR", "type": "VARCHAR",
        "bank_name": "VARCHAR", "service": "VARCHAR", "currency": "VARCHAR", "created_at": "TIMESTAMP",
        "granted": "DOUBLE", "outstanding": "DOUBLE", "liquidity": "DOUBLE"},
        ["product_id", "company_id"]),
    "debt_schedule_config": ("debt_schedule_config.csv", "company_id, product_id", {
        "product_id": "VARCHAR", "company_id": "VARCHAR", "settlement_product_id": "VARCHAR",
        "currency": "VARCHAR", "amortization_type": "VARCHAR", "interest_calc_method": "VARCHAR",
        "amortising_frequency": "VARCHAR", "granted_balance": "DOUBLE", "outstanding_balance": "DOUBLE",
        "total_periods": "BIGINT", "next_payment_date": "TIMESTAMP", "last_payment_date": "TIMESTAMP",
        "annual_interest_rate_or_spread": "DOUBLE", "interest_type": "VARCHAR"},
        ["product_id", "company_id"]),
    "balances": ("balances.csv", "company_id, product_id", {
        "product_id": "VARCHAR", "company_id": "VARCHAR", "date": "TIMESTAMP", "balance": "DOUBLE",
        "available": "DOUBLE", "granted": "DOUBLE", "liquidity": "DOUBLE", "countable": "DOUBLE"},
        ["product_id", "company_id", "balance"]),
    "transactions": ("transactions.csv", "company_id, date, transaction_id", {
        "transaction_id": "VARCHAR", "company_id": "VARCHAR", "product_id": "VARCHAR", "date": "TIMESTAMP",
        "value_date": "TIMESTAMP", "amount": "DOUBLE", "exchange_rate": "DOUBLE", "status": "VARCHAR",
        "accounting_status": "VARCHAR", "category": "VARCHAR", "description": "VARCHAR",
        "counterparty_id": "VARCHAR"},
        ["company_id", "date", "amount"]),
    "invoices": ("invoices.csv", "company_id, issuance_date, operation_id", {
        "operation_id": "VARCHAR", "company_id": "VARCHAR", "document_type": "VARCHAR",
        "issuance_date": "TIMESTAMP", "due_date": "TIMESTAMP", "payment_date": "TIMESTAMP",
        "amount": "DOUBLE", "pending_amount": "DOUBLE", "currency": "VARCHAR",
        "accounting_currency": "VARCHAR", "exchange_rate": "DOUBLE", "status": "VARCHAR",
        "concept": "VARCHAR", "counterparty_id": "VARCHAR"},
        ["company_id", "issuance_date", "amount"]),
}
# sin estas tablas no hay grid de empresas ni caja; el resto puede faltar (empresas sin facturas, sin deuda...)
REQUIRED_TABLES = ("companies", "transactions", "banking_products", "balances")


class InputError(Exception):
    pass


def _q(name):
    return '"' + name.replace('"', '""') + '"'


def load_table(con, csv_dir, table, log):
    """Crea main.<table> con el esquema fijo. Anota en `log` lo que no encaja."""
    csv, order, cols, keys = TABLES[table]
    path = Path(csv_dir) / csv
    typed = ", ".join(f"{_q(c)} {t}" for c, t in cols.items())
    if not path.exists():
        if table in REQUIRED_TABLES:
            raise InputError(f"falta {csv} en {csv_dir}")
        con.execute(f"CREATE TABLE {table} ({typed})")
        log.append((table, f"archivo {csv} ausente", "tabla vacia", 0))
        return
    header = [r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_csv('{path.as_posix()}', header = true, all_varchar = true, sample_size = 100)"
    ).fetchall()]
    missing = [c for c in cols if c not in header]
    extra = [c for c in header if c not in cols]
    if set(keys) & set(missing):
        raise InputError(f"{csv}: faltan columnas clave {sorted(set(keys) & set(missing))}")
    for c in missing:
        log.append((table, f"columna '{c}' ausente en el CSV", "columna -> NULL", 0))
    for c in extra:
        log.append((table, f"columna '{c}' no esperada en el CSV", "columna ignorada", 0))
    present = {c: t for c, t in cols.items() if c in header}
    types = ", ".join(f"'{c}': '{t}'" for c, t in present.items())
    select = ", ".join(_q(c) if c in present else f"CAST(NULL AS {cols[c]}) AS {_q(c)}" for c in cols)
    rej = f"_rej_{table}"
    con.execute(
        f"""
        CREATE TABLE {table} AS
        SELECT {select} FROM read_csv('{path.as_posix()}', header = true, types = {{{types}}},
            ignore_errors = true, store_rejects = true, rejects_table = '{rej}_errors',
            rejects_scan = '{rej}_scans', rejects_limit = 0)
        ORDER BY {order}
        """
    )
    n_bad = con.execute(f"SELECT count(DISTINCT line) FROM {rej}_errors").fetchone()[0]
    con.execute(f"DROP TABLE IF EXISTS {rej}_errors")
    con.execute(f"DROP TABLE IF EXISTS {rej}_scans")
    log.append((table, "fila del CSV que no parsea con el esquema (tipo/comillas/columnas)", "fila eliminada", n_bad))


def build(csv_dir, db_path, force=False, verbose=True):
    csv_dir, db_path = Path(csv_dir), Path(db_path)
    if db_path.exists():
        if not force:
            raise FileExistsError(f"{db_path} ya existe (usa --force para recrearla)")
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    load_log = []
    try:
        for table in TABLES:
            t0 = time.time()
            load_table(con, csv_dir, table, load_log)
            if verbose:
                n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
                print(f"{table:22s} {n:>10,d} filas  ({time.time() - t0:5.1f}s)")
        if verbose:
            print("\nLimpieza -> esquema `clean` (ver clean.dq_log)")
        clean_db.run(con, load_log)
        con.execute("CHECKPOINT")
    except Exception:
        con.close()
        db_path.unlink(missing_ok=True)
        raise
    con.close()
    if verbose:
        print(f"\nBase creada: {db_path} ({db_path.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recrea la base si ya existe")
    ap.add_argument("--csv-dir", type=Path, default=None, help="carpeta con los CSV (por defecto data/ o data/raw/output/)")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    args = ap.parse_args()

    csv_dir = args.csv_dir or find_csv_dir()
    if csv_dir is None:
        print("No encuentro transactions.csv en data/ ni en data/raw/output/ (ver data/README.md)")
        return 1
    if args.db.exists() and not args.force:
        print(f"{args.db} ya existe (usa --force para recrearla)")
        return 0
    try:
        build(csv_dir, args.db, force=args.force)
    except InputError as e:
        print(f"ERROR de entrada: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
