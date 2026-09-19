"""One re-runnable command: input folder of CSVs -> clean DuckDB (+ clean.dq_log) -> feature store.

    python -m analysis.pipeline [--input DIR] [--work-dir DIR] [--no-features]

    input     folder with the eight track CSVs (default: data/ or data/raw/output/)
    work-dir  where the outputs go (default: data/). Writes
                <work-dir>/embat.duckdb                  raw in `main`, cleaned in `clean`, `clean.dq_log`
                <work-dir>/feature_store/monthly.parquet company x month features (families a-h)
                <work-dir>/feature_store/dq_log.csv      copy of clean.dq_log
                <work-dir>/feature_store/pipeline_run.json  input/output fingerprints

Same rules for any folder, nothing is fitted here. Companies without invoices, short trails and
subsets of companies are handled by the feature modules (NaN where a window is not observable).
Requires the system Python with duckdb, pandas, pyarrow; set PYTHONUTF8=1 on Windows.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import duckdb
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis import build_db
from analysis.features import common
from analysis.features.build_feature_store import FAMILIES, coverage_report
from analysis.features.grid import company_meta, monthly_grid


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_feature_store(db_path: Path, out_dir: Path) -> pd.DataFrame:
    """Same assembly as analysis.features.build_feature_store.main, on an arbitrary database."""
    out_dir.mkdir(parents=True, exist_ok=True)
    common.DB_PATH = Path(db_path)  # the family modules only see `con`; grid/meta helpers reopen via common.connect
    con = common.connect()
    # DuckDB sums doubles in a thread-dependent order; one thread makes the parquet byte-identical between runs
    con.execute("SET threads = 1")
    grid = monthly_grid(con)
    if grid.empty:
        con.close()
        raise SystemExit("no company has transactions in clean.transactions: nothing to build")
    panel = grid.merge(company_meta(con), on="company_id", how="left")
    used = ["transactions", "companies", "groups", "banking_products", "debt_products"]
    loaded = []
    for letter, modname in FAMILIES:
        mod = importlib.import_module(modname)
        part = mod.build(con, grid[["company_id", "period"]].copy())
        extra = [c for c in part.columns if c not in {"company_id", "period"}]
        clash = set(extra) & set(panel.columns)
        if clash:
            raise ValueError(f"{modname} column clash: {clash}")
        panel = panel.merge(part, on=["company_id", "period"], how="left")
        loaded.append(letter)
        used.extend(getattr(mod, "SOURCE_TABLES", []))
    con.close()
    panel = panel.sort_values(["company_id", "period"]).reset_index(drop=True)
    used = sorted(set(used))
    panel.to_parquet(out_dir / "monthly.parquet", index=False)
    coverage_report(panel, used).to_csv(out_dir / "coverage.csv", index=False)
    (out_dir / "build.json").write_text(
        json.dumps(
            {"rows": int(len(panel)), "companies": int(panel.company_id.nunique()), "families": loaded,
             "tables": used, "n_cols": int(panel.shape[1])},
            indent=2,
        )
        + "\n"
    )
    return panel


def run(input_dir: Path, work_dir: Path, features: bool = True, verbose: bool = True) -> dict:
    input_dir, work_dir = Path(input_dir), Path(work_dir)
    db_path = work_dir / "embat.duckdb"
    out_dir = work_dir / "feature_store"
    t0 = time.time()
    build_db.build(input_dir, db_path, force=True, verbose=verbose)
    con = duckdb.connect(str(db_path), read_only=True)
    dq = con.execute("SELECT * FROM clean.dq_log").df()
    counts = {t: con.execute(f"SELECT count(*) FROM clean.{t}").fetchone()[0] for t in build_db.TABLES}
    con.close()
    out_dir.mkdir(parents=True, exist_ok=True)
    dq.to_csv(out_dir / "dq_log.csv", index=False)
    info = {
        "input_dir": str(input_dir),
        "input_files": {p.name: {"bytes": p.stat().st_size, "sha256": _sha256(p)}
                        for p in sorted(input_dir.glob("*.csv")) if p.name in {v[0] for v in build_db.TABLES.values()}},
        "clean_rows": counts,
        "dq_log_rows_affected": {f"{r.source}:{r.table_name}:{r.issue}": int(r.rows_affected) for r in dq.itertuples()},
    }
    if features:
        panel = build_feature_store(db_path, out_dir)
        info["feature_store"] = {"rows": int(len(panel)), "companies": int(panel.company_id.nunique()),
                                 "cols": int(panel.shape[1]),
                                 "parquet_sha256": _sha256(out_dir / "monthly.parquet")}
    info["seconds"] = round(time.time() - t0, 1)
    (out_dir / "pipeline_run.json").write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")
    if verbose:
        new = dq[(dq.source != "clean") & (dq.rows_affected > 0)]
        print(f"\ndq_log: {len(dq)} rules, {int((dq.rows_affected > 0).sum())} with rows affected; "
              f"{len(new)} from guard/load detectors:")
        for r in new.itertuples():
            print(f"  [{r.source}] {r.table_name}: {r.issue} -> {r.action}  ({r.rows_affected:,})")
        print(f"\ndone in {info['seconds']} s -> {db_path}" + (f", {out_dir / 'monthly.parquet'}" if features else ""))
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=None, help="folder with the CSVs (default data/ or data/raw/output/)")
    ap.add_argument("--work-dir", type=Path, default=common.DATA, help="output folder (default data/)")
    ap.add_argument("--no-features", action="store_true", help="stop after the clean schema")
    args = ap.parse_args()
    input_dir = args.input or build_db.find_csv_dir()
    if input_dir is None:
        print("no input folder: pass --input DIR (needs transactions.csv)")
        return 1
    try:
        run(input_dir, args.work_dir, features=not args.no_features)
    except build_db.InputError as e:
        print(f"input error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
