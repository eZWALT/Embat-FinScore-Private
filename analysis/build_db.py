"""Carga los CSV de data/ en una base DuckDB (data/embat.duckdb).

Uso:  python analysis/build_db.py [--force]
Requiere:  pip install duckdb
"""
import argparse
import sys
import time
from pathlib import Path

import duckdb

import clean_db

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DB_PATH = DATA / "embat.duckdb"
# los CSV pueden estar en data/ o, si se descargo el zip del reto, en data/raw/output/
CSV_DIR = next((d for d in (DATA, DATA / "raw" / "output") if (d / "transactions.csv").exists()), None)

# tabla -> (csv, orden fisico para que DuckDB pode bloques al filtrar por empresa/fecha)
TABLES = {
    "groups": ("groups.csv", "group_id"),
    "companies": ("companies.csv", "company_id"),
    "banking_products": ("banking_products.csv", "company_id, product_id"),
    "debt_products": ("debt_products.csv", "company_id, product_id"),
    "debt_schedule_config": ("debt_schedule_config.csv", "company_id, product_id"),
    "balances": ("balances.csv", "company_id, product_id"),
    "transactions": ("transactions.csv", "company_id, date"),
    "invoices": ("invoices.csv", "company_id, issuance_date"),
}


# columnas que la autodeteccion infiere mal (casi todo nulo)
TYPE_OVERRIDES = {"balances": {"available": "DOUBLE"}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="recrea la base si ya existe")
    args = ap.parse_args()

    if CSV_DIR is None:
        print("No encuentro transactions.csv en data/ ni en data/raw/output/ (ver data/README.md)")
        return 1
    if DB_PATH.exists():
        if not args.force:
            print(f"{DB_PATH} ya existe (usa --force para recrearla)")
            return 0
        DB_PATH.unlink()

    con = duckdb.connect(str(DB_PATH))
    for table, (csv, order) in TABLES.items():
        t0 = time.time()
        path = (CSV_DIR / csv).as_posix()
        over = TYPE_OVERRIDES.get(table)
        types = ", types = {" + ", ".join(f"'{c}': '{t}'" for c, t in over.items()) + "}" if over else ""
        con.execute(
            f"""
            CREATE TABLE {table} AS
            SELECT * FROM read_csv('{path}', header = true, sample_size = -1{types})
            ORDER BY {order}
            """
        )
        n = con.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"{table:22s} {n:>10,d} filas  ({time.time() - t0:5.1f}s)")

    print("\nLimpieza -> esquema `clean` (ver clean.dq_log)")
    clean_db.run(con)
    con.execute("CHECKPOINT")
    con.close()
    print(f"\nBase creada: {DB_PATH} ({DB_PATH.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
