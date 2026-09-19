"""Write the SQL that fills analytics.company_size, from the clean DuckDB. Paste the files into the Neon SQL editor.

    PYTHONUTF8=1 python infra/neon/scripts/export_company_size.py --duckdb data/embat.duckdb \
        --companies data/feature_store/monthly.parquet --as-of 2026-08 --out <dir>

Needs migration 007 applied first. Writes size_1.sql, size_2.sql, ... in chunks small enough to paste; run them in any order.
Each file is an upsert into the current run, so re-running is safe. Nothing here touches Neon.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

# Mirrors CAT_MAP in analysis/features/common.py (op_in / op_out), which feeds a_op_in / a_op_out.
OP_IN = ("collection", "bulk_collection", "cash_settlement", "cash_settlements", "pos_settlement", "collection_refund")
OP_OUT = ("payment", "bulk_payment", "utility", "payment_refund", "salary", "social_security", "tax", "tax_refund", "cash_withdrawal", "pos_withdrawal")
WINDOW = 3
CHUNK = 450


def month_bounds(as_of: str) -> tuple[str, str]:
    year, month = map(int, as_of.split("-"))
    start_index = year * 12 + (month - 1) - (WINDOW - 1)
    end_index = year * 12 + month
    return f"{start_index // 12:04d}-{start_index % 12 + 1:02d}-01", f"{end_index // 12:04d}-{end_index % 12 + 1:02d}-01"


def lit(value: float | None) -> str:
    return "NULL" if value is None else repr(round(float(value), 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duckdb", type=Path, required=True)
    parser.add_argument("--companies", type=Path, required=True, help="a parquet/csv with a company_id column: the scored companies")
    parser.add_argument("--as-of", required=True, help="last month of the window, YYYY-MM")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    start, end = month_bounds(args.as_of)
    con = duckdb.connect(str(args.duckdb), read_only=True)
    con.execute("SET search_path = 'clean,main'")
    ins = ", ".join(f"'{c}'" for c in OP_IN)
    outs = ", ".join(f"'{c}'" for c in OP_OUT)
    rows = con.execute(
        f"""
        WITH scored AS (SELECT DISTINCT company_id FROM '{args.companies.as_posix()}'),
        flows AS (
          SELECT company_id,
                 SUM(CASE WHEN category IN ({ins}) THEN amount ELSE 0 END) / {WINDOW} AS inflow,
                 -SUM(CASE WHEN category IN ({outs}) THEN amount ELSE 0 END) / {WINDOW} AS outflow
          FROM transactions WHERE "date" >= DATE '{start}' AND "date" < DATE '{end}' GROUP BY 1),
        latest AS (
          SELECT * FROM balances b
          WHERE "date" = (SELECT MAX("date") FROM balances b2 WHERE b2.company_id = b.company_id)
            AND NOT COALESCE(balance_sentinel, false)),
        cash AS (SELECT company_id, SUM(GREATEST(balance, 0)) AS cash FROM latest GROUP BY 1)
        SELECT s.company_id, COALESCE(f.inflow, 0), COALESCE(f.outflow, 0), c.cash, co.currency
        FROM scored s
        LEFT JOIN flows f USING (company_id)
        LEFT JOIN cash c USING (company_id)
        LEFT JOIN companies co USING (company_id)
        ORDER BY s.company_id
        """
    ).fetchall()

    args.out.mkdir(parents=True, exist_ok=True)
    chunks = [rows[i : i + CHUNK] for i in range(0, len(rows), CHUNK)]
    for number, chunk in enumerate(chunks, start=1):
        values = ",\n".join(
            f"('{company}', {lit(inflow)}, {lit(outflow)}, {lit(cash)}, {'NULL' if currency is None else repr(str(currency))})"
            for company, inflow, outflow, cash, currency in chunk
        )
        sql = f"""INSERT INTO analytics.company_size (run_id, company_id, as_of_month, window_months, monthly_inflow, monthly_outflow, cash, currency)
SELECT r.run_id, v.company_id, '{args.as_of}', {WINDOW}, v.monthly_inflow::double precision, v.monthly_outflow::double precision, v.cash::double precision, v.currency::text
FROM api.current_run r
CROSS JOIN (VALUES
{values}
) AS v(company_id, monthly_inflow, monthly_outflow, cash, currency)
ON CONFLICT (run_id, company_id) DO UPDATE SET
  as_of_month = EXCLUDED.as_of_month, window_months = EXCLUDED.window_months, monthly_inflow = EXCLUDED.monthly_inflow,
  monthly_outflow = EXCLUDED.monthly_outflow, cash = EXCLUDED.cash, currency = EXCLUDED.currency;
"""
        (args.out / f"size_{number}.sql").write_text(sql, encoding="utf-8")
    print(f"{len(rows)} companies, window {start} .. {end} (exclusive), {len(chunks)} files in {args.out}")


if __name__ == "__main__":
    main()
