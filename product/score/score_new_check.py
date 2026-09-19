"""score_new on a folder of unseen-style CSVs: a subset of companies, some without invoices, some with a 3-month trail.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m product.score.score_new_check [--csv-dir DIR]
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import pandas as pd

from analysis import build_db
from analysis.pipeline_check import _write_subset
from .run import score_new


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=None)
    args = ap.parse_args()
    src = args.csv_dir or build_db.find_csv_dir()
    tmp = Path(tempfile.mkdtemp(prefix="score_new_check_"))
    ok = True
    try:
        con = duckdb.connect()
        comp = con.execute(f"SELECT company_id FROM read_csv('{(src / 'companies.csv').as_posix()}', header = true) ORDER BY 1").fetchall()
        inv = {r[0] for r in con.execute(f"SELECT DISTINCT company_id FROM read_csv('{(src / 'invoices.csv').as_posix()}', header = true)").fetchall()}
        con.close()
        ids = [r[0] for r in comp]
        with_inv, without_inv = [c for c in ids if c in inv][100:120], [c for c in ids if c not in inv][100:120]
        short = [with_inv[0], without_inv[0]]
        folder = tmp / "csv"
        _write_subset(src, folder, sorted(with_inv + without_inv), short_trail=short)
        table = score_new(folder, tmp / "out", verbose=False)
        back = pd.read_csv(tmp / "out" / "scores.csv")
        ok &= _check("columns", list(back.columns) == ["company_id", "month", "score", "trajectory", "confidence"], str(list(back.columns)))
        ok &= _check("scores are 0-100 and finite", bool(back["score"].between(0, 100).all()))
        ok &= _check("every scored company is in the folder", set(back["company_id"]) <= set(with_inv + without_inv),
                     f"{back['company_id'].nunique()} companies, {len(back)} company-months")
        s = back[back["company_id"].isin(short)]
        ok &= _check("short-trail companies are scored, flagged low confidence and 'insufficient history'",
                     bool(len(s)) and (s["confidence"] == "low").all() and (s["trajectory"] == "insufficient history").all(),
                     f"{s['company_id'].nunique()} companies, months {sorted(s['month'].unique())}")
        n = back[back["company_id"].isin(without_inv)]
        ok &= _check("companies without invoices are scored with confidence below 'high'", bool(len(n)) and (n["confidence"] != "high").all(),
                     n["confidence"].value_counts().to_dict().__str__())
        detail = pd.read_parquet(tmp / "out" / "scores_detail.parquet")
        ok &= _check("reasons carry a euro amount", detail["reasons"].str.contains("€").mean() > 0.5, f"{detail['reasons'].str.contains('€').mean():.0%} of rows")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL OK" if ok else "\nFAILURES")
    return 0 if ok else 1


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


if __name__ == "__main__":
    sys.exit(main())
