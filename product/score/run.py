"""Score a feature store end to end, and `score_new(csv_folder)` for unseen companies.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m product.score.run --csv-folder <dir> --out <dir>

score_new: analysis.pipeline (CSV -> clean DB -> feature store) -> items -> score with the saved train reference.
Writes company_id, month, score, trajectory (+ confidence) to <out>/scores.csv, and the full detail to
<out>/scores_detail.parquet (categories, per-item points and contributions, month-on-month attribution,
reasons). Nothing is fitted: the reference comes from reference.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from . import spec
from .amounts import compute_amounts
from .explain import attribution, reasons, trajectory
from .fit import load_reference
from .items import compute_items
from .scorecard import score_frame


def score_store(store: pd.DataFrame, ref: dict, con=None, with_reasons: bool = True, drop_families=frozenset()) -> pd.DataFrame:
    """Full scoring of a feature-store panel. `con`: read-only DuckDB on the matching `clean` schema for the euro amounts."""
    items = compute_items(store)
    if con is not None:
        items = items.merge(compute_amounts(con, items[["company_id", "period"]]), on=["company_id", "period"], how="left")
    scored = score_frame(items, ref, drop_families=drop_families)
    res = scored["res"]
    base = pd.concat([items[["company_id", "period", "trail_months", "recency_days", "inflow_recent", "inflow_older"]], res], axis=1)
    traj = trajectory(base[["company_id", "period", "score", "guard"]])
    attr = attribution(items, scored["contrib"], res)
    out = pd.concat([items[["company_id", "period"]], res, traj[["slope3", "slope6", "trajectory"]],
                     scored["cats"].add_prefix("cat_"), scored["pts"].add_prefix("pts_"),
                     scored["contrib"].add_prefix("contrib_"), attr.drop(columns=["company_id", "period"])], axis=1)
    if with_reasons:
        out = pd.concat([out, reasons(items, scored, res, attr)], axis=1)
    return out


def month_str(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s).dt.strftime("%Y-%m")


def score_new(csv_folder, out_dir=None, work_dir=None, reference: dict | None = None, verbose: bool = True) -> pd.DataFrame:
    """Score every company in a folder of track CSVs. Returns and writes company_id, month, score, trajectory."""
    from analysis import pipeline  # imported here: the pipeline pulls in the whole feature stack

    csv_folder = Path(csv_folder)
    out_dir = Path(out_dir) if out_dir else csv_folder.parent / (csv_folder.name + "_scores")
    work_dir = Path(work_dir) if work_dir else out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    info = pipeline.run(csv_folder, work_dir, verbose=verbose)
    store = pd.read_parquet(work_dir / "feature_store" / "monthly.parquet")
    con = duckdb.connect(str(work_dir / "embat.duckdb"), read_only=True)
    con.execute("SET search_path = 'clean,main'")
    try:
        detail = score_store(store, reference or load_reference(), con=con)
    finally:
        con.close()
    detail = detail[detail["score"].notna()].reset_index(drop=True)
    table = pd.DataFrame({"company_id": detail["company_id"], "month": month_str(detail["period"]),
                          "score": detail["score"].round(1), "trajectory": detail["trajectory"],
                          "confidence": detail["confidence"]})
    table.to_csv(out_dir / "scores.csv", index=False)
    for c in ("reasons", "change_reasons"):
        detail[c] = detail[c].map(lambda x: json.dumps(x, ensure_ascii=False))
    detail.to_parquet(out_dir / "scores_detail.parquet", index=False)
    if verbose:
        print(f"scored {table.company_id.nunique()} companies, {len(table)} company-months -> {out_dir / 'scores.csv'}")
        print(f"dq_log: {sum(1 for v in info['dq_log_rows_affected'].values() if v)} rules with rows affected (see {work_dir / 'feature_store' / 'dq_log.csv'})")
    return table


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv-folder", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--work-dir", type=Path, default=None)
    args = ap.parse_args()
    score_new(args.csv_folder, args.out, args.work_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
