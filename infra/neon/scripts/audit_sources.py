#!/usr/bin/env python3
"""Audit the handoff DuckDB and score bundle. Prints a JSON summary to stdout."""
from __future__ import annotations

import json
import os
from pathlib import Path

import duckdb

HANDOFF = Path(os.environ.get("EMBAT_HANDOFF", "/Users/ruben/Desktop/Embat-handoff"))
DUCKDB = Path(os.environ.get("EMBAT_CLEAN_DUCKDB", HANDOFF / "embat_clean.duckdb"))
BUNDLE = Path(os.environ.get("SCORE_BUNDLE_DIR", HANDOFF / "bundle"))


def main() -> None:
    con = duckdb.connect(str(DUCKDB), read_only=True)
    tables = [r[0] for r in con.execute("SHOW TABLES FROM clean").fetchall()]
    schema = []
    for table in tables:
        cols = con.execute(f"DESCRIBE clean.{table}").fetchall()
        n = con.execute(f"SELECT count(*) FROM clean.{table}").fetchone()[0]
        schema.append(
            {
                "table": table,
                "rows": n,
                "columns": [
                    {"name": c[0], "type": c[1], "null": c[2], "key": c[3], "default": c[4]}
                    for c in cols
                ],
            }
        )

    product_overlap = con.execute(
        """
        SELECT count(*) FROM (
          SELECT product_id FROM clean.banking_products
          INTERSECT
          SELECT product_id FROM clean.debt_products
        )
        """
    ).fetchone()[0]
    tx_null_id = con.execute(
        "SELECT count(*) FROM clean.transactions WHERE transaction_id IS NULL"
    ).fetchone()[0]
    tx_dup_id = con.execute(
        """
        SELECT count(*) FROM (
          SELECT transaction_id FROM clean.transactions
          WHERE transaction_id IS NOT NULL
          GROUP BY 1 HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    cp_as_company = con.execute(
        """
        SELECT count(DISTINCT t.counterparty_id)
        FROM clean.transactions t
        JOIN clean.companies c ON c.company_id = t.counterparty_id
        WHERE t.counterparty_id IS NOT NULL
        """
    ).fetchone()[0]
    tx_unknown_product = con.execute(
        "SELECT count(*) FROM clean.transactions WHERE NOT product_known"
    ).fetchone()[0]
    inv_dup = con.execute(
        """
        SELECT count(*) FROM (
          SELECT company_id, operation_id FROM clean.invoices
          WHERE operation_id IS NOT NULL
          GROUP BY 1,2 HAVING count(*) > 1
        )
        """
    ).fetchone()[0]
    group_co = con.execute("SELECT count(DISTINCT group_id) FROM clean.companies").fetchone()[0]
    groups_n = con.execute("SELECT count(*) FROM clean.groups").fetchone()[0]
    companies_n = con.execute("SELECT count(*) FROM clean.companies").fetchone()[0]

    manifest = json.loads((BUNDLE / "manifest.json").read_text())
    company_files = list((BUNDLE / "companies").glob("COMP_*.json"))
    zip_files = [p.name for p in HANDOFF.glob("*.zip")]

    summary = {
        "duckdb_path": str(DUCKDB),
        "duckdb_bytes": DUCKDB.stat().st_size,
        "tables": schema,
        "keys": {
            "companies": companies_n,
            "groups_table": groups_n,
            "distinct_group_id_in_companies": group_co,
            "product_id_overlap_banking_debt": product_overlap,
            "transactions_null_id": tx_null_id,
            "transactions_duplicate_id": tx_dup_id,
            "counterparty_ids_that_match_company_id": cp_as_company,
            "transactions_unknown_product": tx_unknown_product,
            "invoice_duplicate_company_operation": inv_dup,
        },
        "bundle": {
            "schema_version": manifest.get("schema_version"),
            "counts": manifest.get("counts"),
            "company_files": len(company_files),
            "is_sample": manifest.get("is_sample"),
            "as_of_month": manifest.get("as_of_month"),
            "zip_duplicates_not_loaded": zip_files,
        },
        "classification": {
            "source_clean": tables,
            "computed_bundle": [
                "manifest",
                "companies.json",
                "companies/*.json",
                "groups.json",
                "alerts.json",
                "clusters.json",
            ],
            "derived_not_loaded": zip_files + ["types.ts"],
        },
    }
    print(json.dumps(summary, indent=2, default=str))
    con.close()


if __name__ == "__main__":
    main()
