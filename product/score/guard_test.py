"""Going-dark test: a company that stops moving money must score low, never high.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m product.score.guard_test [--csv-dir DIR] [--keep]

Takes 60 train companies that score in the top 40% and are active at the cutoff (2026-02), writes a copy of their CSVs
where activity stops after the cutoff and runs the whole path (analysis.pipeline -> score):
  A  bank dark    no transactions after the cutoff; the balance snapshot is moved back so the earlier cash path is unchanged
  B  all dark     as A, and no invoices issued after the cutoff; invoices paid after the cutoff stay open
Checks: every row the guard calls dark is capped (score <= 30); how many post-cutoff scores exceed the pre-cutoff score;
and the same rows with the guard switched off, to show what it prevents. Also looks at the real silent companies.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from analysis import build_db, pipeline
from . import spec
from .fit import DEFAULT_STORE, fit_reference, load_holdout, load_reference
from .items import compute_items
from .run import score_store
from .score import score_frame

CUTOFF = pd.Timestamp("2026-03-01")   # first month without activity
N_COMPANIES = 60


def _rd(src: Path, csv: str) -> str:
    return f"read_csv('{(src / csv).as_posix()}', header = true, all_varchar = true, sample_size = -1)"


def write_variant(src: Path, dst: Path, companies: list[str], all_dark: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ids = ", ".join(f"'{c}'" for c in companies)
    cut = f"TIMESTAMP '{CUTOFF:%Y-%m-%d}'"
    con = duckdb.connect()
    con.execute(f"""COPY (SELECT * FROM {_rd(src, 'groups.csv')} WHERE group_id IN
                     (SELECT group_id FROM {_rd(src, 'companies.csv')} WHERE company_id IN ({ids})))
                    TO '{(dst / 'groups.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    for csv in ("companies.csv", "banking_products.csv", "debt_products.csv", "debt_schedule_config.csv"):
        con.execute(f"COPY (SELECT * FROM {_rd(src, csv)} WHERE company_id IN ({ids})) TO '{(dst / csv).as_posix()}' (HEADER, DELIMITER ',')")
    con.execute(f"""COPY (SELECT * FROM {_rd(src, 'transactions.csv')} WHERE company_id IN ({ids})
                     AND CAST("date" AS TIMESTAMP) < {cut}) TO '{(dst / 'transactions.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    # snapshot balance moved back by the removed flows, so cash before the cutoff is what it was
    con.execute(f"""COPY (
        SELECT b.* REPLACE (CAST(CAST(b.balance AS DOUBLE) - coalesce(r.removed, 0) AS VARCHAR) AS balance)
        FROM {_rd(src, 'balances.csv')} b
        LEFT JOIN (SELECT product_id, SUM(CAST(amount AS DOUBLE)) AS removed FROM {_rd(src, 'transactions.csv')}
                   WHERE company_id IN ({ids}) AND CAST("date" AS TIMESTAMP) >= {cut} GROUP BY 1) r USING (product_id)
        WHERE b.company_id IN ({ids})) TO '{(dst / 'balances.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    if all_dark:
        # paid after the cutoff -> stays open. Impossible payment dates (year 6913...) are left alone: they are dropped by clean
        late_paid = f"(CAST(payment_date AS TIMESTAMP) >= {cut} AND CAST(payment_date AS TIMESTAMP) <= TIMESTAMP '2026-09-01')"
        con.execute(f"""COPY (
            SELECT * REPLACE (CASE WHEN {late_paid} THEN NULL ELSE payment_date END AS payment_date,
                              CASE WHEN {late_paid} THEN 'pending' ELSE status END AS status)
            FROM {_rd(src, 'invoices.csv')} WHERE company_id IN ({ids}) AND CAST(issuance_date AS TIMESTAMP) < {cut})
            TO '{(dst / 'invoices.csv').as_posix()}' (HEADER, DELIMITER ',')""")
    else:
        con.execute(f"COPY (SELECT * FROM {_rd(src, 'invoices.csv')} WHERE company_id IN ({ids})) TO '{(dst / 'invoices.csv').as_posix()}' (HEADER, DELIMITER ',')")
    con.close()


def pick_companies(store: pd.DataFrame, ref: dict, hold: set[str]) -> list[str]:
    sc = score_store(store, ref, with_reasons=False)
    it = compute_items(store)
    d = pd.concat([sc[["company_id", "period", "score", "confidence"]], it[["trail_months", "dark_level", "recency_days"]]], axis=1)
    at = d[(d["period"] == CUTOFF - pd.DateOffset(months=1)) & ~d["company_id"].isin(hold) & (d["trail_months"] >= 14)
           & (d["dark_level"] == 0) & (d["recency_days"] <= 5) & d["score"].notna()]
    at = at[at["score"] >= at["score"].quantile(0.6)]
    # invoices before the cutoff: the feature layer zero-fills invoice columns for every month of a company that has
    # invoices anywhere in the file (and leaves them NaN if it has none), so a company whose only invoices come later
    # would score differently before the cutoff for a reason unrelated to look-ahead in the score
    early = store[store["period"] < CUTOFF]
    inv = early.groupby("company_id")["e_ar_issued"].apply(lambda s: (s > 0).any())
    with_inv = at[at["company_id"].map(inv)].sort_values("company_id")["company_id"].tolist()
    without = at[~at["company_id"].map(inv)].sort_values("company_id")["company_id"].tolist()
    half = N_COMPANIES // 2
    return with_inv[:half] + without[:half]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    src = args.csv_dir or build_db.find_csv_dir()
    store = pd.read_parquet(args.store)
    ref = load_reference()
    hold = load_holdout()
    companies = pick_companies(store, ref, hold)
    print(f"{len(companies)} train companies, top-40% score and active in {CUTOFF - pd.DateOffset(months=1):%Y-%m}")
    tmp = Path(tempfile.mkdtemp(prefix="guard_test_"))
    ok = True
    try:
        pre = score_store(store[store["company_id"].isin(companies)], ref, with_reasons=False)
        pre_score = pre[pre["period"] == CUTOFF - pd.DateOffset(months=1)].set_index("company_id")["score"]
        for name, all_dark in (("A bank dark", False), ("B all dark", True)):
            d = tmp / name[0]
            write_variant(src, d / "csv", companies, all_dark)
            pipeline.run(d / "csv", d / "work", verbose=False)
            st = pd.read_parquet(d / "work" / "feature_store" / "monthly.parquet")
            items = compute_items(st)
            with_g = score_frame(items, ref)["res"]
            no_g = score_frame(items, ref, guard=False)["res"]
            x = pd.concat([items[["company_id", "period", "trail_months", "dark_level", "recency_days"]],
                           with_g[["score", "guard", "confidence"]], no_g["score"].rename("score_no_guard")], axis=1)
            # no look-ahead: months before the cutoff must score as they did with the full data
            early = pd.concat([items[["company_id", "period"]], with_g["score"]], axis=1)
            early = early[early["period"] < CUTOFF].merge(pre[["company_id", "period", "score"]], on=["company_id", "period"], suffixes=("", "_full"))
            gap = (early["score"] - early["score_full"]).abs()
            ok &= _check(f"{name}: scores before the cutoff unchanged by removing later data (no look-ahead)", bool(gap.max() < 1e-6 or (gap > 0.5).mean() < 0.005),
                         f"{len(early)} company-months, max abs diff {gap.max():.2g}, {(gap > 0.5).sum()} differ by > 0.5 point")
            x = x[x["period"] >= CUTOFF]
            x["pre"] = x["company_id"].map(pre_score)
            x["k"] = ((x["period"].dt.year - CUTOFF.year) * 12 + x["period"].dt.month - CUTOFF.month + 1)  # months of silence
            dark = x[x["guard"] == "dark"]
            print(f"\n{name}: {x['company_id'].nunique()} companies, months after the cutoff 1..{int(x['k'].max())}")
            ok &= _check("every row flagged dark has score <= cap", bool((dark["score"] <= spec.CAP_DARK + 1e-9).all()), f"{len(dark)} rows")
            ok &= _check("dark rows never above the pre-cutoff score", bool((dark["score"] <= dark["pre"]).all()))
            ok &= _check("dark rows have low confidence", bool((dark["confidence"] == "low").all()))
            late = x[x["k"] >= 3]
            ok &= _check("from month 3 of silence every company is flagged dark or fading", bool((late["guard"] != "").all()),
                         f"{(late['guard'] == '').sum()} of {len(late)} rows not flagged")
            tab = x.groupby("k").agg(rows=("score", "size"), dark=("guard", lambda g: (g == "dark").mean()),
                                     score_med=("score", "median"), no_guard_med=("score_no_guard", "median"),
                                     above_pre=("score", lambda s: np.nan), pre_med=("pre", "median"))
            tab["above_pre"] = x.assign(a=x["score"] > x["pre"]).groupby("k")["a"].mean()
            tab["no_guard_above_pre"] = x.assign(a=x["score_no_guard"] > x["pre"]).groupby("k")["a"].mean()
            print(tab.round(2).to_string())
            first = x[x["k"] <= 2]
            print(f"  months 1-2 (guard not yet triggered): score above the pre-cutoff score in {(first['score'] > first['pre']).mean():.0%} of rows "
                  f"(largest rise {(first['score'] - first['pre']).max():+.1f} points)")
        # real silent companies
        real = store[store["company_id"].map(store.groupby("company_id")["c_last_tx_before_2026_06"].max()) > 0]
        r_items = compute_items(store)
        r_sc = score_frame(r_items, ref)["res"]
        z = pd.concat([r_items[["company_id", "period", "dark_level"]], r_sc[["score", "guard"]]], axis=1)
        z = z[z["company_id"].isin(real["company_id"].unique()) & (z["dark_level"] == 2)]
        print(f"\nreal silent companies (last booking before 2026-06): {z['company_id'].nunique()} companies, {len(z)} dark rows, "
              f"max score {z['score'].max():.1f}, median {z['score'].median():.1f}")
    finally:
        if args.keep:
            print("kept", tmp)
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    print("\nALL OK" if ok else "\nFAILURES")
    return 0 if ok else 1


def _check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    return ok


if __name__ == "__main__":
    sys.exit(main())
