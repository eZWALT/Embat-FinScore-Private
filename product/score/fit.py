"""Fit the percentile reference on train companies only and save it as reference.json.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m product.score.fit [--store PATH]

The reference is the only thing fitted in the score: for every "pct" item, 501 quantiles (0, 0.2%, ... 100%)
of the trailing-window values over train company-months. Weights, thresholds and caps are fixed a priori.
Holdout companies (analysis/splits/holdout_companies.csv) never enter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import spec
from .items import compute_items

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORE = ROOT / "data" / "feature_store" / "monthly.parquet"
REFERENCE_PATH = Path(__file__).resolve().parent / "reference.json"
HOLDOUT_PATH = ROOT / "analysis" / "splits" / "holdout_companies.csv"
N_Q = 501


def load_holdout() -> set[str]:
    return set(pd.read_csv(HOLDOUT_PATH)["company_id"].astype(str))


def quantile_table(values: pd.Series) -> list[float]:
    v = values.to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    return np.quantile(v, np.linspace(0.0, 1.0, N_Q)).tolist()


def fit_reference(items: pd.DataFrame, company_ids: set[str] | None = None) -> dict:
    """Quantile tables from `items` rows of `company_ids` (default: every non-holdout company)."""
    hold = load_holdout()
    ids = set(items["company_id"].astype(str)) - hold if company_ids is None else set(company_ids)
    leaked = ids & hold
    if leaked:
        raise AssertionError(f"holdout companies in the reference fit: {sorted(leaked)[:5]}")
    tr = items[items["company_id"].astype(str).isin(ids)]
    tr = tr[tr["trail_months"] >= spec.WINDOW]
    return {
        "n_companies": int(tr["company_id"].nunique()),
        "n_company_months": int(len(tr)),
        "items": {name: {"n": int(tr[name].notna().sum()), "q": quantile_table(tr[name])} for name in spec.PCT_ITEMS},
    }


def save_reference(ref: dict, path: Path = REFERENCE_PATH) -> None:
    ref = {"note": "Percentile reference fitted on train companies only (holdout excluded). See product/score/fit.py.",
           "quantiles": N_Q, **ref}
    path.write_text(json.dumps(ref) + "\n")


def load_reference(path: Path = REFERENCE_PATH) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = ap.parse_args()
    items = compute_items(pd.read_parquet(args.store))
    ref = fit_reference(items)
    save_reference(ref)
    print(f"reference from {ref['n_companies']} train companies, {ref['n_company_months']} company-months -> {REFERENCE_PATH}")
    for name, r in ref["items"].items():
        q = np.array(r["q"])
        print(f"  {name:14s} n={r['n']:6d}  p10={q[50]:.4g}  p50={q[250]:.4g}  p90={q[450]:.4g}")


if __name__ == "__main__":
    main()
