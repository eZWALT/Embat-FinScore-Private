"""Euro amounts behind the reasons that the feature store does not carry, straight from clean.invoices.

Trailing 3 months (5 for late payments, the span the delay items average over) up to the month-end, same conventions as analysis/features/invoices.py (document_type =
'invoice', status <> 'cancel', AR = amount > 0, AP = amount < 0, payment dates that are impossible dropped).
Amounts are in the invoice's own currency, as in the feature store.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

COLS = ["ar_late5", "ap_late5", "cn3", "top1_eur"]


def compute_amounts(con, grid: pd.DataFrame) -> pd.DataFrame:
    """`grid`: company_id, period (month start). Returns the same keys plus COLS (0 where nothing happened)."""
    g = grid[["company_id", "period"]].drop_duplicates().copy()
    g["period"] = pd.to_datetime(g["period"])
    con.register("_amt_grid", g)
    try:
        cn_types = [r[0] for r in con.execute(
            "SELECT DISTINCT document_type FROM invoices WHERE lower(document_type) IN ('credit_note', 'creditnote')").fetchall()]
        if not cn_types:
            cn_types = [r[0] for r in con.execute(
                "SELECT DISTINCT document_type FROM invoices WHERE lower(document_type) IN ('note', 'refund')").fetchall()]
        cn_sql = ", ".join(f"'{t}'" for t in cn_types) or "NULL"
        late = con.execute(
            """
            SELECT g.company_id, g.period,
                   SUM(CASE WHEN i.amount > 0 THEN i.amount END) AS ar_late5,
                   SUM(CASE WHEN i.amount < 0 THEN -i.amount END) AS ap_late5
            FROM _amt_grid g
            JOIN invoices i ON i.company_id = g.company_id
             AND i.payment_date >= g.period - INTERVAL 4 MONTH AND i.payment_date < g.period + INTERVAL 1 MONTH
            WHERE i.document_type = 'invoice' AND i.status <> 'cancel' AND i.payment_date IS NOT NULL
              AND NOT i.payment_date_invalid AND i.due_date IS NOT NULL AND i.payment_date > i.due_date
            GROUP BY 1, 2
            """
        ).df()
        billed = con.execute(
            f"""
            SELECT g.company_id, g.period,
                   SUM(CASE WHEN i.document_type IN ({cn_sql}) THEN abs(i.amount) END) AS cn3
            FROM _amt_grid g
            JOIN invoices i ON i.company_id = g.company_id
             AND i.issuance_date >= g.period - INTERVAL 2 MONTH AND i.issuance_date < g.period + INTERVAL 1 MONTH
            GROUP BY 1, 2
            """
        ).df()
        top = con.execute(
            """
            WITH by_cp AS (
              SELECT g.company_id, g.period, i.counterparty_id, SUM(i.amount) AS amt
              FROM _amt_grid g
              JOIN invoices i ON i.company_id = g.company_id
               AND i.issuance_date >= g.period - INTERVAL 2 MONTH AND i.issuance_date < g.period + INTERVAL 1 MONTH
              WHERE i.document_type = 'invoice' AND i.status <> 'cancel' AND i.amount > 0 AND i.counterparty_id IS NOT NULL
              GROUP BY 1, 2, 3)
            SELECT company_id, period, MAX(amt) AS top1_eur FROM by_cp GROUP BY 1, 2
            """
        ).df()
    finally:
        con.unregister("_amt_grid")
    out = g.merge(late, on=["company_id", "period"], how="left").merge(billed, on=["company_id", "period"], how="left") \
           .merge(top, on=["company_id", "period"], how="left")
    out[COLS] = out[COLS].fillna(0.0)
    return out
