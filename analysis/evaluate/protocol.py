"""Leakage-safe evaluation protocol. No model fitting.

Holdout companies (whole groups) never enter percentiles, bins, or any fit.
Group-fold CV keeps every group_id on one side of a split.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[1]
HOLDOUT_PATH = ANALYSIS / "splits" / "holdout_companies.csv"
FOLD_SEED = 20260918

__all__ = [
    "HOLDOUT_PATH",
    "FOLD_SEED",
    "load_holdout",
    "train_companies",
    "group_folds",
    "assert_no_holdout",
    "leakage_check",
    "auroc",
    "pr_auc",
    "rolling_origins",
]


def load_holdout() -> set[str]:
    """Frozen group-aware holdout company_id set."""
    df = pd.read_csv(HOLDOUT_PATH)
    if "company_id" not in df.columns:
        raise ValueError(f"{HOLDOUT_PATH} missing company_id")
    return set(df["company_id"].astype(str))


def train_companies(con) -> pd.DataFrame:
    """Companies not in the holdout, with group_id (clean.companies)."""
    df = con.execute("SELECT company_id, group_id FROM companies").df()
    df["company_id"] = df["company_id"].astype(str)
    df["group_id"] = df["group_id"].astype(str)
    hold = load_holdout()
    out = df.loc[~df["company_id"].isin(hold)].reset_index(drop=True)
    return out


def group_folds(companies_df: pd.DataFrame, n: int = 5, seed: int = FOLD_SEED) -> pd.DataFrame:
    """Assign fold 0..n-1 by group_id so siblings stay together."""
    if n < 2:
        raise ValueError("n must be >= 2")
    if "group_id" not in companies_df.columns:
        raise ValueError("companies_df must have group_id")
    out = companies_df.copy()
    groups = out["group_id"].astype(str).drop_duplicates().sort_values().to_numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)
    fold_of = {gid: int(i % n) for i, gid in enumerate(groups)}
    out["fold"] = out["group_id"].astype(str).map(fold_of).astype(int)
    return out


def _as_id_set(index_or_ids) -> set[str]:
    if index_or_ids is None:
        return set()
    if isinstance(index_or_ids, pd.DataFrame):
        if "company_id" in index_or_ids.columns:
            return set(index_or_ids["company_id"].astype(str))
        return set(pd.Index(index_or_ids.index).astype(str))
    if isinstance(index_or_ids, pd.MultiIndex):
        names = list(index_or_ids.names)
        if "company_id" in names:
            return set(index_or_ids.get_level_values("company_id").astype(str))
        return set(index_or_ids.get_level_values(0).astype(str))
    if isinstance(index_or_ids, (pd.Index, pd.Series)):
        return set(index_or_ids.astype(str))
    if isinstance(index_or_ids, (set, frozenset, list, tuple, np.ndarray)):
        return {str(x) for x in index_or_ids}
    return {str(index_or_ids)}


def assert_no_holdout(index_or_ids) -> None:
    """Raise if any holdout company_id appears in the collection."""
    ids = _as_id_set(index_or_ids)
    leaked = ids & load_holdout()
    if leaked:
        sample = ", ".join(sorted(leaked)[:8])
        extra = f" (+{len(leaked) - 8} more)" if len(leaked) > 8 else ""
        raise AssertionError(f"holdout companies present: {sample}{extra}")


def leakage_check(
    X_cols: Iterable[str],
    y_col: str,
    forbidden_prefixes: Iterable[str] | None = None,
) -> dict:
    """Y must not sit in X; forbidden feature-family prefixes must be absent.

    ``forbidden_prefixes`` are family letters (``\"e\"``) or column prefixes
    (``\"e_\"``). Matching is on the start of each X column name.
    """
    x_list = [str(c) for c in X_cols]
    x_set = set(x_list)
    issues: list[str] = []
    if y_col in x_set:
        issues.append(f"y_col {y_col!r} is in X")
    for raw in forbidden_prefixes or []:
        pref = raw if str(raw).endswith("_") else f"{raw}_"
        bad = [c for c in x_list if c.startswith(pref)]
        if bad:
            shown = bad[:8]
            more = f" (+{len(bad) - 8} more)" if len(bad) > 8 else ""
            issues.append(f"forbidden prefix {pref!r}: {shown}{more}")
    return {
        "ok": len(issues) == 0,
        "issues": issues,
        "y_col": y_col,
        "n_x": len(x_list),
        "forbidden_prefixes": [str(p) for p in (forbidden_prefixes or [])],
    }


def auroc(y_true, y_score) -> float:
    """Mann–Whitney AUROC. Average ranks for ties. NaN if one class is empty."""
    d = pd.DataFrame({"y": y_true, "s": y_score}).dropna()
    if d.empty:
        return float("nan")
    y = d["y"].to_numpy(dtype=float)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = d["s"].rank(method="average").to_numpy(dtype=float)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def pr_auc(y_true, y_score) -> float:
    """Average precision (area under the precision–recall curve)."""
    d = pd.DataFrame({"y": y_true, "s": y_score}).dropna()
    if d.empty:
        return float("nan")
    y = d["y"].to_numpy(dtype=float)
    s = d["s"].to_numpy(dtype=float)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1.0 - y)
    precision = tp / (tp + fp)
    return float(precision[y == 1].sum() / n_pos)


def rolling_origins(
    periods: Iterable,
    horizon: int = 3,
    n_last: int = 9,
) -> list[dict]:
    """Last ``n_last`` origins: train on periods <= t, label at t+horizon."""
    p = pd.to_datetime(pd.Index(periods)).sort_values().unique()
    if len(p) == 0:
        return []
    origins = p[-(n_last + horizon) : -horizon] if horizon > 0 else p[-n_last:]
    out = []
    for t in origins:
        pred = t + pd.DateOffset(months=int(horizon))
        out.append(
            {
                "t": pd.Timestamp(t),
                "horizon": int(horizon),
                "predict_period": pd.Timestamp(pred),
                "train_periods": p[p <= t],
            }
        )
    return out
