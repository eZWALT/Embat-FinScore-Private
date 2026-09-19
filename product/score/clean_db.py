"""Write a clean-only DuckDB: the `clean` schema (cleaned tables + dq_log) without the raw `main` copy.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> \
      python -m product.score.clean_db --work-dir <pipeline work dir> --out <file.duckdb>

With the export bundle this is all the web app's storage needs: the bundle for scores, alerts and explanations, this file for questions about the
records behind them. Tables keep their names inside schema `clean` (clean.transactions, clean.invoices, ..., clean.dq_log).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb


def write_clean_db(work_dir: Path, out: Path) -> Path:
    src = Path(work_dir) / "embat.duckdb"
    out = Path(out)
    if out.exists():
        out.unlink()
    con = duckdb.connect(str(out))
    try:
        con.execute("SET threads = 1")  # same order every run
        con.execute(f"ATTACH '{src.as_posix()}' AS s (READ_ONLY)")
        tables = [r[0] for r in con.execute("SELECT table_name FROM information_schema.tables WHERE table_catalog = 's' AND table_schema = 'clean' ORDER BY 1").fetchall()]
        con.execute("CREATE SCHEMA clean")
        for t in tables:
            con.execute(f"CREATE TABLE clean.{t} AS SELECT * FROM s.clean.{t}")
        con.execute("DETACH s")
    finally:
        con.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    p = write_clean_db(args.work_dir, args.out)
    print(f"clean-only DuckDB: {p} ({p.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
