"""Global LightGBM regressor for monthly Y1 forecast targets.

Primary: y1_net_h1, y1_net_h3. Cheap extra: y1_liq_h1.
X: families A–H from monthly.parquet (or family modules), lags 1 and 3.
Y1 has no forbidden families (labels are t+h, not the same column at t).

5 group-fold CV on TRAIN groups only (raw MAE), then one train fit and
one holdout pass. Compared to company historical mean and last-value
from the cash panel (same construction as baselines.py).

A win is declared only if holdout median-normalized MAE
(median over companies of MAE / mean|y|) beats historical mean.
Quote train group-fold CV MAE (and OOF median-norm) as the fit claim;
holdout is a check. Variants (--residual-last, --log-size) KEEP only if
holdout median-norm beats hist by CLEAR_MARGIN. Default CLI runs ONLY
the three series not posted by 0137801b. --all runs all six.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    group_folds,
    leakage_check,
    load_holdout,
    train_companies,
)
from analysis.features.common import ANALYSIS, DATA, connect, train_mask
from analysis.features.grid import monthly_grid
from analysis.targets.y1_forecast import build as build_y1, cash_month_panel

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
# 0137801b already posted y1_net_h1 / y1_net_h3 / y1_liq_h1. New rows use this id.
AGENT = "a41f9c72"
ROUND = "R4"
WAVE = 4
LAGS = (1, 3)
META_OK = ("group_size", "n_banking")
N_FOLDS = 5
ALLOWED = ("a", "b", "c", "d", "e", "f", "g", "h")
FORBIDDEN: tuple[str, ...] = ()
# Default CLI / run() gate: the three series not yet in TASKS when 0137801b posted.
ONLY: tuple[str, ...] = ("y1_in_h1", "y1_in_h3", "y1_liq_h3")
CLEAR_MARGIN = 0.02  # holdout median-norm vs hist for a variant KEEP

FAMILIES = (
    ("a", "analysis.features.cashflow"),
    ("b", "analysis.features.liquidity"),
    ("c", "analysis.features.ops"),
    ("d", "analysis.features.counterparties"),
    ("e", "analysis.features.invoices"),
    ("f", "analysis.features.debt"),
    ("g", "analysis.features.products"),
    ("h", "analysis.features.groupctx"),
)

KEY_COLS = {
    "company_id",
    "period",
    "month",
    "freq",
    "first_month",
    "first_week",
    "group_id",
    "fold",
    "country",
    "currency",
    "erp",
    "group_erp",
    "created_at",
}

LGB_BASE = dict(
    objective="mae",
    n_estimators=250,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=FOLD_SEED,
    n_jobs=4,
    verbosity=-1,
)

TASKS = (
    {"y": "y1_net_h1", "series": "net", "horizon": 1, "primary": True},
    {"y": "y1_net_h3", "series": "net", "horizon": 3, "primary": True},
    {"y": "y1_liq_h1", "series": "liq", "horizon": 1, "primary": False},
    {"y": "y1_in_h1", "series": "op_in", "horizon": 1, "primary": True},
    {"y": "y1_in_h3", "series": "op_in", "horizon": 3, "primary": True},
    {"y": "y1_liq_h3", "series": "liq", "horizon": 3, "primary": False},
)


def last_resid_task(y: str = "y1_in_h1") -> dict:
    """Predict y − last_value, then add last back. Same X as the level GBM."""
    base = next(t for t in TASKS if t["y"] == y)
    out = dict(base)
    out["residual"] = "last"
    out["model_tag"] = "lightgbm_y1_last_resid"
    return out


def hist_resid_task(y: str = "y1_in_h1") -> dict:
    """Predict y − hist_mean, then add hist_mean back. Inflow's honest naive."""
    base = next(t for t in TASKS if t["y"] == y)
    out = dict(base)
    out["residual"] = "hist"
    out["model_tag"] = "lightgbm_y1_hist_resid"
    return out


def log_size_task(y: str = "y1_in_h1") -> dict:
    """Drop raw euro inflow stems; keep in-memory log1p(a_op_in / a_in3)."""
    base = next(t for t in TASKS if t["y"] == y)
    out = dict(base)
    out["x_mode"] = "log_size"
    out["model_tag"] = "lightgbm_y1_log_size"
    return out


def select_tasks(argv: list[str] | None = None, only: tuple[str, ...] | None = None) -> list[dict]:
    """Pick TASKS. Default ONLY = the three missing series. --all runs all six."""
    args = list(sys.argv[1:] if argv is None else argv)
    if "--residual-last" in args:
        return [last_resid_task("y1_in_h1")]
    if "--residual-hist" in args:
        return [hist_resid_task("y1_in_h1")]
    if "--log-size" in args:
        return [log_size_task("y1_in_h1")]
    names: tuple[str, ...] | None = only
    if "--all" in args:
        names = tuple(t["y"] for t in TASKS)
    for i, a in enumerate(args):
        if a == "--only" and i + 1 < len(args):
            names = tuple(x.strip() for x in args[i + 1].split(",") if x.strip())
        elif a.startswith("--only="):
            names = tuple(x.strip() for x in a.split("=", 1)[1].split(",") if x.strip())
    if names is None:
        env = os.environ.get("ONLY") or os.environ.get("Y1_ONLY")
        if env:
            names = tuple(x.strip() for x in env.split(",") if x.strip())
    if names is None:
        names = ONLY
    if not names:
        return [dict(t) for t in TASKS]
    wanted = set(names)
    picked = [dict(t) for t in TASKS if t["y"] in wanted]
    missing = wanted - {t["y"] for t in picked}
    if missing:
        raise ValueError(f"unknown Y1 task(s): {sorted(missing)}")
    return picked


SIZE_LEVEL_STEMS = ("a_op_in", "a_in3", "a_in6", "a_in12")


def apply_x_mode(panel: pd.DataFrame, x_mode: str | None) -> pd.DataFrame:
    """In-memory X transform. Does not write parquet. log1p is not in the store."""
    if not x_mode or x_mode == "raw":
        return panel
    out = panel.copy()
    if x_mode == "log_size":
        for stem in ("a_op_in", "a_in3"):
            if stem not in out.columns:
                continue
            logname = f"{stem}_log1p"
            if logname not in out.columns:
                raw = pd.to_numeric(out[stem], errors="coerce")
                out[logname] = np.log1p(np.maximum(raw, 0.0))
        drop = [c for c in SIZE_LEVEL_STEMS if c in out.columns]
        if drop:
            out = out.drop(columns=drop)
            print(f"x_mode=log_size dropped {drop} added {[s + '_log1p' for s in ('a_op_in', 'a_in3')]}")
    return out


def _resid_base_col(residual: str | None, last_col: str, mean_col: str) -> str | None:
    if residual == "last":
        return last_col
    if residual == "hist":
        return mean_col
    return None


def _level_or_resid(y_level, base, residual: str | None) -> pd.Series:
    y = pd.to_numeric(y_level, errors="coerce")
    if residual in {"last", "hist"}:
        return y - pd.to_numeric(base, errors="coerce")
    return y


def _add_base(pred, base, residual: str | None) -> np.ndarray:
    out = np.asarray(pred, dtype=float)
    if residual in {"last", "hist"}:
        out = out + pd.to_numeric(base, errors="coerce").to_numpy(dtype=float)
    return out


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    if "month" in out.columns:
        out["month"] = pd.to_datetime(out["month"])
    return out


def _family_of(col: str) -> str:
    return str(col).split("_", 1)[0]


def allowed_x_cols(df: pd.DataFrame, y_col: str) -> list[str]:
    """Numeric A–H family columns plus META_OK. Never y_*."""
    allow = set(ALLOWED)
    cols: list[str] = []
    for c in df.columns:
        if c in KEY_COLS or c == y_col or str(c).startswith("y"):
            continue
        if c in META_OK:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
            continue
        if _family_of(c) not in allow:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    return cols


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Past-only lags within company. shift(k) uses t-k, no look-ahead."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra: dict[str, pd.Series] = {}
    for c in cols:
        if c in META_OK:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def load_store(con) -> tuple[pd.DataFrame, str]:
    if STORE.exists():
        panel = _keys(pd.read_parquet(STORE))
        print(f"loaded store {STORE} shape={panel.shape}")
        return panel, "parquet"
    print(f"{STORE} missing; building families from modules")
    grid = _keys(monthly_grid(con)[["company_id", "period"]])
    panel = grid.copy()
    for letter, modname in FAMILIES:
        try:
            mod = importlib.import_module(modname)
            part = _keys(mod.build(con, grid.copy()))
        except Exception as exc:
            print(f"skip family {letter}: {type(exc).__name__}: {exc}")
            continue
        extra = [c for c in part.columns if c not in {"company_id", "period"}]
        clash = set(extra) & set(panel.columns)
        if clash:
            print(f"skip family {letter}: column clash {sorted(clash)[:4]}")
            continue
        panel = panel.merge(part, on=["company_id", "period"], how="left")
        print(f"loaded family {letter}: {len(extra)} cols")
    return panel, "modules"


def mae(y, yhat) -> float:
    d = pd.DataFrame({"y": y, "p": yhat}).dropna()
    if d.empty:
        return float("nan")
    return float((d["y"] - d["p"]).abs().mean())


def mae_over_mean_abs(y, yhat) -> float:
    d = pd.DataFrame({"y": y, "p": yhat}).dropna()
    if d.empty:
        return float("nan")
    scale = float(d["y"].abs().mean())
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return float((d["y"] - d["p"]).abs().mean() / scale)


def company_beat_share(companies, y, pred, baseline) -> dict:
    """Share of companies where MAE(pred) < MAE(baseline). Skips mean|y|=0."""
    d = pd.DataFrame({"company_id": companies, "y": y, "p": pred, "b": baseline}).dropna()
    if d.empty:
        return {"beat_share": float("nan"), "n_companies": 0}
    wins = 0
    n = 0
    for _, g in d.groupby("company_id", sort=False):
        scale = float(g["y"].abs().mean())
        if not np.isfinite(scale) or scale <= 0:
            continue
        n += 1
        if float((g["y"] - g["p"]).abs().mean()) < float((g["y"] - g["b"]).abs().mean()):
            wins += 1
    return {"beat_share": float(wins / n) if n else float("nan"), "n_companies": int(n)}


def median_norm_mae(companies, y, yhat) -> dict:
    """Median over companies of (MAE / mean|y|). Companies with mean|y|=0 skipped."""
    d = pd.DataFrame({"company_id": companies, "y": y, "p": yhat}).dropna()
    if d.empty:
        return {
            "median_norm_mae": float("nan"),
            "n_companies": 0,
            "mean_norm_mae": float("nan"),
        }
    rows = []
    for _, g in d.groupby("company_id", sort=False):
        scale = float(g["y"].abs().mean())
        if not np.isfinite(scale) or scale <= 0:
            continue
        rows.append(float((g["y"] - g["p"]).abs().mean() / scale))
    if not rows:
        return {
            "median_norm_mae": float("nan"),
            "n_companies": 0,
            "mean_norm_mae": float("nan"),
        }
    arr = np.asarray(rows, dtype=float)
    return {
        "median_norm_mae": float(np.median(arr)),
        "n_companies": int(len(arr)),
        "mean_norm_mae": float(np.mean(arr)),
    }


def naive_last_mean(panel: pd.DataFrame) -> pd.DataFrame:
    """Origin-t last value and expanding mean. Uses only the series at or before t."""
    p = _keys(panel).sort_values(["company_id", "month"]).reset_index(drop=True)
    g = p.groupby("company_id", sort=False)
    for col in ("net", "op_in", "liq"):
        p[f"{col}_last"] = p[col]
        p[f"{col}_mean"] = g[col].transform(lambda s: s.expanding(min_periods=1).mean())
    return p


def _fit_lgb(Xtr, ytr, Xva=None, yva=None, n_estimators: int | None = None) -> lgb.LGBMRegressor:
    params = dict(LGB_BASE)
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    clf = lgb.LGBMRegressor(**params)
    fit_kw: dict = {}
    if Xva is not None and yva is not None and n_estimators is None:
        fit_kw["eval_set"] = [(Xva, yva)]
        fit_kw["eval_metric"] = "l1"
        fit_kw["callbacks"] = [
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    clf.fit(Xtr, ytr, **fit_kw)
    return clf


def _n_trees(clf: lgb.LGBMRegressor, default: int) -> int:
    v = getattr(clf, "best_iteration_", None)
    if v is not None and np.isfinite(v) and int(v) > 0:
        return int(v)
    return int(default)


def _gain_table(clf: lgb.LGBMRegressor, cols: list[str], n: int = 12) -> pd.DataFrame:
    gain = clf.booster_.feature_importance(importance_type="gain")
    names = list(clf.booster_.feature_name())
    if names != cols and set(names) == set(cols):
        order = [names.index(c) for c in cols]
        gain = np.asarray(gain)[order]
        names = cols
    elif len(names) != len(gain):
        names = cols[: len(gain)]
    tab = pd.DataFrame({"feature": names, "gain": gain}).sort_values("gain", ascending=False)
    return tab.head(n).reset_index(drop=True)


def _assert_allowed(cols: list[str], y_col: str) -> None:
    leak = leakage_check(cols, y_col, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage for {y_col}: {leak['issues']}")
    bad_y = [c for c in cols if c == y_col or str(c).startswith("y")]
    if bad_y:
        raise RuntimeError(f"y columns in X for {y_col}: {bad_y[:8]}")


def _metrics(companies, y, pred) -> dict:
    return {
        "mae": mae(y, pred),
        "mae_over_mean_abs": mae_over_mean_abs(y, pred),
        **median_norm_mae(companies, y, pred),
        "n": int(pd.DataFrame({"y": y, "p": pred}).dropna().shape[0]),
    }


def append_registry(rows: list[dict]) -> None:
    if not rows or not REGISTRY.exists():
        return
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _fmt(v) -> str:
    return f"{v:.6g}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else ""


def run_task(
    task: dict,
    store: pd.DataFrame,
    y_panel: pd.DataFrame,
    naive: pd.DataFrame,
    folds: pd.DataFrame,
    hold_ids: set[str],
) -> dict:
    y_col = task["y"]
    series = task["series"]
    residual = task.get("residual")
    x_mode = task.get("x_mode")
    last_col = f"{series}_last"
    mean_col = f"{series}_mean"
    resid_col = _resid_base_col(residual, last_col, mean_col)
    print("\n" + "=" * 72)
    print(
        f"TASK {y_col} series={series} h={task['horizon']} primary={task['primary']} "
        f"residual={residual or 'none'} x_mode={x_mode or 'raw'}"
    )
    print("=" * 72)

    panel = store.merge(y_panel[["company_id", "period", y_col]], on=["company_id", "period"], how="left")
    naive_keep = naive[["company_id", "month", last_col, mean_col]].rename(
        columns={"month": "period"}
    )
    panel = panel.merge(naive_keep, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    panel = apply_x_mode(panel, x_mode)

    base_cols = allowed_x_cols(panel, y_col)
    _assert_allowed(base_cols, y_col)
    print(
        f"base X cols={len(base_cols)} families="
        f"{sorted({_family_of(c) for c in base_cols if c not in META_OK})}"
    )

    lag_cols = [c for c in base_cols if c not in META_OK]
    print(f"adding lags {list(LAGS)} on {len(lag_cols)} family X cols (meta not lagged)")
    panel = add_lags(panel, lag_cols, LAGS)
    feat_cols = allowed_x_cols(panel, y_col)
    _assert_allowed(feat_cols, y_col)
    if y_col in feat_cols or any(c.startswith("y") for c in feat_cols):
        raise RuntimeError(f"Y leaked into X for {y_col}")

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[y_col].notna()
    if resid_col is not None:
        labeled = labeled & pd.to_numeric(panel[resid_col], errors="coerce").notna()
    print(
        f"X cols={len(feat_cols)} train_labeled={int((is_train & labeled).sum())} "
        f"hold_labeled={int((is_hold & labeled).sum())} residual={residual or 'none'}"
    )

    cv_rows = []
    best_iters = []
    oof_cid: list[pd.Series] = []
    oof_y: list[pd.Series] = []
    oof_p: list[np.ndarray] = []
    oof_last: list[pd.Series] = []
    oof_hist: list[pd.Series] = []
    write_reg = bool(task.get("write_registry", True))
    for k in range(N_FOLDS):
        tr = is_train & labeled & (panel["fold"] != k)
        va = is_train & labeled & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        if int(tr.sum()) < 50 or int(va.sum()) < 20:
            print(f"fold {k}: skip (n_tr={int(tr.sum())} n_va={int(va.sum())})")
            cv_rows.append({"fold": k, "mae": float("nan"), "n_val": int(va.sum())})
            continue
        base_tr = panel.loc[tr, resid_col] if resid_col else panel.loc[tr, last_col]
        base_va = panel.loc[va, resid_col] if resid_col else panel.loc[va, last_col]
        ytr = _level_or_resid(panel.loc[tr, y_col], base_tr, residual)
        yva_fit = _level_or_resid(panel.loc[va, y_col], base_va, residual)
        yva = panel.loc[va, y_col].astype(float)
        clf = _fit_lgb(panel.loc[tr, feat_cols], ytr, panel.loc[va, feat_cols], yva_fit)
        pred = _add_base(clf.predict(panel.loc[va, feat_cols]), base_va, residual)
        trees = _n_trees(clf, LGB_BASE["n_estimators"])
        row = {
            "fold": k,
            "mae": mae(yva, pred),
            "mae_over_mean_abs": mae_over_mean_abs(yva, pred),
            "n_val": int(va.sum()),
            "best_iteration": trees,
        }
        best_iters.append(trees)
        cv_rows.append(row)
        oof_cid.append(panel.loc[va, "company_id"])
        oof_y.append(yva)
        oof_p.append(np.asarray(pred, dtype=float))
        oof_last.append(pd.to_numeric(panel.loc[va, last_col], errors="coerce"))
        oof_hist.append(pd.to_numeric(panel.loc[va, mean_col], errors="coerce"))
        print(
            f"fold {k}: mae={row['mae']:.6g} mae/mean|y|={row['mae_over_mean_abs']:.4f} "
            f"n={row['n_val']} trees={trees}"
        )

    cv = pd.DataFrame(cv_rows)
    cv_mae = float(cv["mae"].mean()) if not cv.empty else float("nan")
    n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
    n_trees = max(50, n_trees)
    if oof_y:
        oof_c = pd.concat(oof_cid, ignore_index=True)
        oof_yy = pd.concat(oof_y, ignore_index=True)
        oof_pp = np.concatenate(oof_p)
        oof_gbm = median_norm_mae(oof_c, oof_yy, oof_pp)
        oof_last_m = median_norm_mae(oof_c, oof_yy, pd.concat(oof_last, ignore_index=True))
        oof_hist_m = median_norm_mae(oof_c, oof_yy, pd.concat(oof_hist, ignore_index=True))
    else:
        oof_gbm = {"median_norm_mae": float("nan"), "n_companies": 0, "mean_norm_mae": float("nan")}
        oof_last_m = dict(oof_gbm)
        oof_hist_m = dict(oof_gbm)
    print(f"CV mean MAE={cv_mae:.6g} n_trees_final={n_trees}")
    print(
        f"CV OOF median_norm gbm={oof_gbm['median_norm_mae']:.4f} "
        f"hist={oof_hist_m['median_norm_mae']:.4f} last={oof_last_m['median_norm_mae']:.4f} "
        f"cos={oof_gbm['n_companies']}"
    )

    tr_all = is_train & labeled
    ho_lab = is_hold & labeled
    assert_no_holdout(panel.loc[tr_all, "company_id"])
    ytr_all = _level_or_resid(
        panel.loc[tr_all, y_col],
        panel.loc[tr_all, resid_col] if resid_col else panel.loc[tr_all, last_col],
        residual,
    )
    final = _fit_lgb(panel.loc[tr_all, feat_cols], ytr_all, n_estimators=n_trees)

    if ho_lab.sum() == 0:
        gbm = {"mae": float("nan"), "mae_over_mean_abs": float("nan"), "median_norm_mae": float("nan"), "n": 0, "n_companies": 0, "mean_norm_mae": float("nan")}
        last = dict(gbm)
        hist = dict(gbm)
        beat_hist = {"beat_share": float("nan"), "n_companies": 0}
        beat_last = {"beat_share": float("nan"), "n_companies": 0}
        n_hold_pred = 0
        ho_y = np.array([])
        ho_pred = np.array([])
    else:
        ho_pred = _add_base(
            final.predict(panel.loc[ho_lab, feat_cols]),
            panel.loc[ho_lab, resid_col] if resid_col else panel.loc[ho_lab, last_col],
            residual,
        )
        ho_y = panel.loc[ho_lab, y_col].to_numpy(dtype=float)
        ho_cid = panel.loc[ho_lab, "company_id"]
        ho_last = pd.to_numeric(panel.loc[ho_lab, last_col], errors="coerce")
        ho_mean = pd.to_numeric(panel.loc[ho_lab, mean_col], errors="coerce")
        gbm = _metrics(ho_cid, ho_y, ho_pred)
        last = _metrics(ho_cid, ho_y, ho_last)
        hist = _metrics(ho_cid, ho_y, ho_mean)
        beat_hist = company_beat_share(ho_cid, ho_y, ho_pred, ho_mean)
        beat_last = company_beat_share(ho_cid, ho_y, ho_pred, ho_last)
        n_hold_pred = int(np.isfinite(ho_pred).sum())

    n_hold_lab = int(ho_lab.sum())
    coverage = (n_hold_pred / n_hold_lab) if n_hold_lab else float("nan")
    beats_hist = (
        np.isfinite(gbm["median_norm_mae"])
        and np.isfinite(hist["median_norm_mae"])
        and gbm["median_norm_mae"] < hist["median_norm_mae"]
    )
    beats_last = (
        np.isfinite(gbm["median_norm_mae"])
        and np.isfinite(last["median_norm_mae"])
        and gbm["median_norm_mae"] < last["median_norm_mae"]
    )
    print(
        f"HOLDOUT GBM mae={gbm['mae']:.6g} mae/mean|y|={gbm['mae_over_mean_abs']:.4f} "
        f"median_norm={gbm['median_norm_mae']:.4f} n={gbm['n']} cos={gbm['n_companies']} cov={coverage:.4f}"
    )
    print(
        f"         last mae={last['mae']:.6g} mae/mean|y|={last['mae_over_mean_abs']:.4f} "
        f"median_norm={last['median_norm_mae']:.4f}"
    )
    print(
        f"         hist mae={hist['mae']:.6g} mae/mean|y|={hist['mae_over_mean_abs']:.4f} "
        f"median_norm={hist['median_norm_mae']:.4f}"
    )
    hist_gap = (
        float(hist["median_norm_mae"] - gbm["median_norm_mae"])
        if np.isfinite(gbm["median_norm_mae"]) and np.isfinite(hist["median_norm_mae"])
        else float("nan")
    )
    clear_hist = bool(beats_hist and np.isfinite(hist_gap) and hist_gap >= CLEAR_MARGIN)
    print(f"WIN vs hist_mean (median_norm_mae)={beats_hist} vs last_value={beats_last}")
    print(
        f"holdout company beat-share vs hist={beat_hist['beat_share']:.3f} "
        f"vs last={beat_last['beat_share']:.3f} n={beat_hist['n_companies']}"
    )
    print(
        f"CV OOF vs holdout median_norm: oof={oof_gbm['median_norm_mae']:.4f} "
        f"hold={gbm['median_norm_mae']:.4f} hist_gap={hist_gap:.4f} "
        f"clear_margin({CLEAR_MARGIN})={clear_hist}"
    )

    imp = _gain_table(final, feat_cols, 12)
    print("top 12 gain importances")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    train_grid = int(is_train.sum())
    train_cov = (int(tr_all.sum()) / train_grid) if train_grid else float("nan")
    fam = "+".join(s.upper() for s in ALLOWED)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    win_note = "WIN" if beats_hist else "NO_WIN"
    notes = (
        f"{win_note} vs hist_mean median_norm; trees={n_trees}; n_x={len(feat_cols)}; "
        f"lags=1,3; meta={','.join(c for c in META_OK if c in feat_cols)}; "
        f"residual={residual or 'none'}; x_mode={x_mode or 'raw'}; "
        f"gbm_mae={gbm['mae']:.6g}; hist_mae={hist['mae']:.6g}; last_mae={last['mae']:.6g}; "
        f"gbm_medn={gbm['median_norm_mae']:.4f}; hist_medn={hist['median_norm_mae']:.4f}; "
        f"last_medn={last['median_norm_mae']:.4f}; beats_last={beats_last}; "
        f"cv_oof_medn={oof_gbm['median_norm_mae']:.4f}; cv_hist_medn={oof_hist_m['median_norm_mae']:.4f}"
    )
    tag = str(task.get("model_tag") or "lightgbm_y1")
    reg = []
    for split, metric, value, cov, model in (
        ("cv5_group", "mae", cv_mae, train_cov, tag),
        ("cv5_group", "median_norm_mae", oof_gbm["median_norm_mae"], train_cov, tag),
        ("holdout", "mae", gbm["mae"], coverage, tag),
        ("holdout", "mae_over_mean_abs", gbm["mae_over_mean_abs"], coverage, tag),
        ("holdout", "median_norm_mae", gbm["median_norm_mae"], coverage, tag),
        ("holdout", "mae", hist["mae"], coverage, "hist_mean"),
        ("holdout", "mae_over_mean_abs", hist["mae_over_mean_abs"], coverage, "hist_mean"),
        ("holdout", "median_norm_mae", hist["median_norm_mae"], coverage, "hist_mean"),
        ("holdout", "mae", last["mae"], coverage, "last_value"),
        ("holdout", "mae_over_mean_abs", last["mae_over_mean_abs"], coverage, "last_value"),
        ("holdout", "median_norm_mae", last["median_norm_mae"], coverage, "last_value"),
    ):
        reg.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": fam if model == tag else "-",
                "y": y_col,
                "model": model,
                "split": split,
                "metric": metric,
                "value": _fmt(value),
                "coverage": f"{cov:.4f}" if np.isfinite(cov) else "",
                "notes": notes if model == tag else f"same rows as {tag}; series={series}",
            }
        )
    if write_reg:
        append_registry(reg)
        print(f"appended {len(reg)} registry rows")
    else:
        print(f"skipped {len(reg)} registry rows (--no-registry)")

    return {
        "y": y_col,
        "series": series,
        "horizon": task["horizon"],
        "primary": task["primary"],
        "residual": residual or "none",
        "x_mode": x_mode or "raw",
        "model_tag": tag,
        "cv_mae": cv_mae,
        "cv_folds": cv_rows,
        "n_trees": n_trees,
        "gbm": gbm,
        "hist_mean": hist,
        "last_value": last,
        "coverage": coverage,
        "n_hold_labeled": n_hold_lab,
        "n_hold_pred": n_hold_pred,
        "beats_hist_mean": beats_hist,
        "beats_last_value": beats_last,
        "n_x": len(feat_cols),
        "n_x_base": len(base_cols),
        "train_labeled": int(tr_all.sum()),
        "train_coverage": train_cov,
        "top12": imp.to_dict(orient="records"),
        "declared_win": beats_hist,
        "clear_hist_margin": clear_hist,
        "hist_gap": hist_gap,
        "hold_beat_hist": beat_hist,
        "hold_beat_last": beat_last,
        "cv_oof_median_norm": oof_gbm,
        "cv_oof_hist_median_norm": oof_hist_m,
        "cv_oof_last_median_norm": oof_last_m,
    }


def series_persistence(cash: pd.DataFrame, hold_ids: set[str], col: str, h: int) -> dict:
    """Train-only corr(col_t, col_{t+h}). No model, no holdout rows."""
    p = _keys(cash).sort_values(["company_id", "month"]).reset_index(drop=True)
    tr = ~p["company_id"].isin(hold_ids)
    assert_no_holdout(p.loc[tr, "company_id"])
    g = p.loc[tr].groupby("company_id", sort=False)[col]
    fwd = g.shift(-int(h))
    d = pd.DataFrame({"x": p.loc[tr, col], "fwd": fwd}).dropna()
    if d.empty:
        return {"col": col, "h": int(h), "n": 0, "pearson": float("nan"), "spearman": float("nan")}
    return {
        "col": col,
        "h": int(h),
        "n": int(len(d)),
        "pearson": float(d["x"].corr(d["fwd"])),
        "spearman": float(d["x"].corr(d["fwd"], method="spearman")),
    }


def liq_persistence(cash: pd.DataFrame, hold_ids: set[str], h: int = 3) -> dict:
    """Train-only corr(liq_t, liq_{t+h}). No model, no holdout rows."""
    return series_persistence(cash, hold_ids, "liq", h)


def month_of_year_r2(cash: pd.DataFrame, hold_ids: set[str], col: str) -> dict:
    """Train-only: share of company-demeaned variance explained by calendar month."""
    p = _keys(cash)
    tr = ~p["company_id"].isin(hold_ids)
    assert_no_holdout(p.loc[tr, "company_id"])
    d = p.loc[tr, ["company_id", "month", col]].copy()
    d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna()
    if d.empty:
        return {"col": col, "n": 0, "r2": float("nan"), "amp_over_std": float("nan")}
    d["moy"] = pd.to_datetime(d["month"]).dt.month
    d["dev"] = d[col] - d.groupby("company_id")[col].transform("mean")
    tot = float(d["dev"].var(ddof=0))
    moy_mean = d.groupby("moy")["dev"].transform("mean")
    resid = d["dev"] - moy_mean
    r2 = float(1.0 - resid.var(ddof=0) / tot) if tot > 0 else float("nan")
    amp = float(d.groupby("moy")["dev"].mean().abs().max())
    std = float(np.sqrt(tot)) if tot > 0 else float("nan")
    return {
        "col": col,
        "n": int(len(d)),
        "r2": r2,
        "amp_over_std": float(amp / std) if std and np.isfinite(std) and std > 0 else float("nan"),
    }


def run(only: tuple[str, ...] | None = None) -> dict:
    hold_ids = load_holdout()
    tasks = select_tasks(only=only)
    if "--no-registry" in sys.argv[1:]:
        for t in tasks:
            t["write_registry"] = False
        print("registry writes disabled")
    print(f"agent={AGENT} tasks={[t['y'] for t in tasks]}")
    con = connect()
    store, x_source = load_store(con)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    grid = store[["company_id", "period"]].drop_duplicates()
    print(
        f"store={store.shape} source={x_source} train_cos={train_cos['company_id'].nunique()} "
        f"hold_cos={len(hold_ids)} folds={folds['fold'].nunique()}"
    )

    print("building Y1 labels and cash panel (existing modules; not edited)")
    y1 = _keys(build_y1(con, grid))
    cash = cash_month_panel(con)
    con.close()
    naive = naive_last_mean(cash)
    print(f"y1={y1.shape} cash={cash.shape} naive={naive.shape}")
    persist = liq_persistence(cash, hold_ids, h=3)
    persist_h1 = liq_persistence(cash, hold_ids, h=1)
    in_h1 = series_persistence(cash, hold_ids, "op_in", 1)
    in_h3 = series_persistence(cash, hold_ids, "op_in", 3)
    print(
        f"TRAIN liq_t vs liq_t+3 n={persist['n']} "
        f"pearson={persist['pearson']:.4f} spearman={persist['spearman']:.4f}"
    )
    print(
        f"TRAIN liq_t vs liq_t+1 n={persist_h1['n']} "
        f"pearson={persist_h1['pearson']:.4f} spearman={persist_h1['spearman']:.4f}"
    )
    print(
        f"TRAIN op_in_t vs t+1 n={in_h1['n']} "
        f"pearson={in_h1['pearson']:.4f} spearman={in_h1['spearman']:.4f}"
    )
    print(
        f"TRAIN op_in_t vs t+3 n={in_h3['n']} "
        f"pearson={in_h3['pearson']:.4f} spearman={in_h3['spearman']:.4f}"
    )
    seas_in = month_of_year_r2(cash, hold_ids, "op_in")
    seas_liq = month_of_year_r2(cash, hold_ids, "liq")
    print(
        f"TRAIN month-of-year R2 (company-demeaned) op_in={seas_in['r2']:.4f} "
        f"amp/std={seas_in['amp_over_std']:.3f} liq={seas_liq['r2']:.4f} "
        f"amp/std={seas_liq['amp_over_std']:.3f}"
    )
    if "--diag" in sys.argv[1:]:
        print("diag-only: skipping GBM fits")
        return {
            "_liq_persist_train_h3": persist,
            "_liq_persist_train_h1": persist_h1,
            "_op_in_persist_train_h1": in_h1,
            "_op_in_persist_train_h3": in_h3,
            "_op_in_month_r2": seas_in,
            "_liq_month_r2": seas_liq,
        }

    results = []
    for task in tasks:
        out = run_task(task, store, y1, naive, folds, hold_ids)
        out["x_source"] = x_source
        results.append(out)

    summary = {}
    for r in results:
        key = r["y"] if r.get("residual", "none") == "none" and r.get("x_mode", "raw") == "raw" else f"{r['y']}:{r.get('model_tag', 'x')}"
        summary[key] = r
    print("\nSUMMARY")
    slim = {}
    for y, r in summary.items():
        slim[y] = {
            "horizon": r["horizon"],
            "residual": r.get("residual", "none"),
            "x_mode": r.get("x_mode", "raw"),
            "cv_mae": r["cv_mae"],
            "holdout_mae": r["gbm"]["mae"],
            "holdout_mae_over_mean_abs": r["gbm"]["mae_over_mean_abs"],
            "holdout_median_norm_mae": r["gbm"]["median_norm_mae"],
            "hist_mae": r["hist_mean"]["mae"],
            "hist_mae_over_mean_abs": r["hist_mean"]["mae_over_mean_abs"],
            "hist_median_norm_mae": r["hist_mean"]["median_norm_mae"],
            "last_mae": r["last_value"]["mae"],
            "last_mae_over_mean_abs": r["last_value"]["mae_over_mean_abs"],
            "last_median_norm_mae": r["last_value"]["median_norm_mae"],
            "beats_hist_mean": r["beats_hist_mean"],
            "beats_last_value": r["beats_last_value"],
            "declared_win": r["declared_win"],
            "n_hold": r["n_hold_labeled"],
            "coverage": r["coverage"],
            "cv_oof_median_norm": r.get("cv_oof_median_norm", {}).get("median_norm_mae"),
            "cv_oof_hist_median_norm": r.get("cv_oof_hist_median_norm", {}).get("median_norm_mae"),
            "cv_oof_last_median_norm": r.get("cv_oof_last_median_norm", {}).get("median_norm_mae"),
            "hold_beat_hist": r.get("hold_beat_hist", {}).get("beat_share"),
            "hold_beat_last": r.get("hold_beat_last", {}).get("beat_share"),
        }
    slim["_liq_persist_train_h3"] = persist
    slim["_liq_persist_train_h1"] = persist_h1
    slim["_op_in_persist_train_h1"] = in_h1
    slim["_op_in_persist_train_h3"] = in_h3
    slim["_op_in_month_r2"] = seas_in
    slim["_liq_month_r2"] = seas_liq
    print(json.dumps(slim, indent=2, default=str))
    any_win = any(r["declared_win"] for r in results)
    print(f"DECLARED WIN (any horizon vs hist_mean median_norm_mae)={any_win}")
    summary["_liq_persist_train_h3"] = persist
    summary["_liq_persist_train_h1"] = persist_h1
    summary["_op_in_persist_train_h1"] = in_h1
    summary["_op_in_persist_train_h3"] = in_h3
    summary["_op_in_month_r2"] = seas_in
    summary["_liq_month_r2"] = seas_liq
    return summary


if __name__ == "__main__":
    run()
