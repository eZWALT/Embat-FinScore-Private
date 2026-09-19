"""XGBoost (and a LightGBM shallow twin) for accepted Y4 only.

Y4 is brief Q3 (turning) / Q5 (why): debt-service / inflow *doubles* over
three months. Not bankruptcy. Not a 0–100 score.

Label: y4_ds_r_double (ds_r[t+3] >= 2 * ds_r[t], both defined, ds_r[t] > 0.05).
Other Y4 columns stay rejected — this module does not revive them.

X: families A (minus debt-service clones), B, C, D, E, G, H. Never family F.
Also drop a_fin_cost*, a_debt* (numerator of ds_r), and any CAT_MAP
debt_repayment share (m_debt*) if it appears. Lags 1 and 3, past-only.

Protocol is a copy of analysis.models.xgb_panel (Y5 PARK lives there — do
not edit that file). 5 train group-fold CV. Holdout companies never enter
a fit, a sign pick, or a percentile. Holdout may be scored as a LOW_POWER
check; quote n_pos.

KEEP if CV >= best-single + 0.02, beats dummy 0.50, and trees did not
collapse to 1–10. CLOSE if within 0.02 of single. PARK if lose to single
or dummy, or if the model only rediscovers inflow size.

Night result (quote train group-fold CV): PARK. Best fixed OOF single is
d_cust_hhi_lag3 at 0.605. XGB 400+ES 0.585 collapsed; XGB 50/depth-3
0.561; LGBM twin 0.540. Holdout n_pos=16 LOW_POWER, XGB 0.467 inverts.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.xgb_y4 --variant xgb_d3_n50
    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.xgb_y4 --list
"""
from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    auroc,
    group_folds,
    leakage_check,
    load_holdout,
    pr_auc,
    train_companies,
)
from analysis.features.common import ANALYSIS, DATA, connect
from analysis.features.grid import monthly_grid
from analysis.targets.y4_debt import META as Y4_META
from analysis.targets.y4_debt import build as build_y4

try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover
    lgb = None

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
ACCEPT = DATA / "feature_store" / "y_acceptance.csv"
OUT_MD = ANALYSIS / "outputs" / "y4_xgb.md"
AGENT = "f248c993"
WAVE = 4
ROUND = "R4"
Y_COL = "y4_ds_r_double"
FORBIDDEN = ("f",)
ALLOWED = ("a", "b", "c", "d", "e", "g", "h")
N_FOLDS = 5
LAGS = (1, 3)
CLEAR_MARGIN = 0.02
DUMMY_AUROC = 0.50
LEAK_RHO = 0.80
SIZE_AUROC_MAX = 0.60
COLLAPSE_MAX_TREES = 10
MIN_SINGLE_COV = 0.25
MIN_SINGLE_N = 200
MIN_X_COV = 0.05
LOW_POWER_POS = 30
META_OK = ("group_size", "n_banking")

# Same-column clones of the Y numerator / family-F rewrite.
# protocol.leakage_check("a_fin_cost") becomes "a_fin_cost_" and misses the base.
DEBT_LEAK_PREFIXES = (
    "f_",
    "a_fin_cost",
    "a_debt",
    "m_debt",
)

# Raw euro SIZE stems (and their lags). Used by the nosize variant.
SIZE_STEMS = frozenset(
    {
        "a_op_in",
        "a_op_out",
        "a_net",
        "a_in3",
        "a_in6",
        "a_in12",
        "a_out3",
        "a_out6",
        "a_out12",
        "a_transfer",
        "a_invest",
        "a_fin_cost",
        "a_debt_service",
        "b_liq",
        "b_min_liq_3",
        "b_mean_liq_3",
        "h_sib_in",
        "h_sib_out",
        "h_sib_net",
    }
)

# Activity / inventory counts and group-scale. Ratios variant drops these too.
COUNT_STEMS = frozenset(
    {
        "a_n_tx",
        "c_n_tx",
        "c_n_days_with_tx",
        "d_n_cust",
        "d_n_supp",
        "e_ar_open",
        "e_ap_open",
        "e_ar_issued",
        "e_ap_issued",
        "g_n_accounts",
        "g_n_banks",
        "g_n_types",
        "h_group_size",
        "h_n_siblings_active",
        "group_size",
        "n_banking",
    }
)

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

# Copied from xgb_panel.py (Y5). Do not edit that file.
XGB_BASE = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    max_depth=4,
    learning_rate=0.05,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    n_jobs=4,
    random_state=FOLD_SEED,
    missing=np.nan,
)

XGB_SHALLOW = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    max_depth=3,
    learning_rate=0.05,
    min_child_weight=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.5,
    n_jobs=4,
    random_state=FOLD_SEED,
    missing=np.nan,
)

LGB_SHALLOW = dict(
    objective="binary",
    n_estimators=50,
    max_depth=3,
    num_leaves=8,
    learning_rate=0.05,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=FOLD_SEED,
    n_jobs=1,
    verbosity=-1,
)

VARIANTS = (
    {
        "id": "xgb_es",
        "model": "xgb_y4_es400",
        "kind": "xgb",
        "n_estimators": 400,
        "early_stopping_rounds": 40,
        "shallow": False,
        "drop_size": False,
        "drop_counts": False,
        "notes": "Y5 protocol copy: 400 trees, max_depth=4, early-stop 40",
    },
    {
        "id": "xgb_d3_n50",
        "model": "xgb_y4_d3_n50",
        "kind": "xgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": False,
        "drop_counts": False,
        "notes": "diagnostic 50 trees, max_depth=3 (use if 400+ES collapses)",
    },
    {
        "id": "lgb_d3_n50",
        "model": "lgb_y4_d3_n50",
        "kind": "lgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": False,
        "drop_counts": False,
        "notes": "LightGBM shallow twin of the 50 / depth-3 spec",
    },
    {
        "id": "xgb_d3_n50_nosize",
        "model": "xgb_y4_d3_n50_nosize",
        "kind": "xgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": True,
        "drop_counts": False,
        "notes": "50 / depth-3 after dropping raw euro SIZE stems",
    },
    {
        "id": "lgb_d3_n50_nosize",
        "model": "lgb_y4_d3_n50_nosize",
        "kind": "lgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": True,
        "drop_counts": False,
        "notes": "LightGBM shallow twin, no raw euro SIZE stems",
    },
    {
        "id": "xgb_d3_n50_ratios",
        "model": "xgb_y4_d3_n50_ratios",
        "kind": "xgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": True,
        "drop_counts": True,
        "notes": "50 / depth-3; drop euro SIZE and activity/group counts",
    },
    {
        "id": "xgb_d3_n50_d",
        "model": "xgb_y4_d3_n50_d",
        "kind": "xgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": False,
        "drop_counts": False,
        "families": ("d",),
        "notes": "50 / depth-3, family D only (best OOF singles live here)",
    },
    {
        "id": "xgb_d3_n50_d2",
        "model": "xgb_y4_d3_n50_d2",
        "kind": "xgb",
        "n_estimators": 50,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": False,
        "drop_counts": False,
        "families": ("d",),
        "stems": ("d_cust_hhi", "d_n_supp"),
        "notes": "50 / depth-3 on d_cust_hhi + d_n_supp and lags only",
    },
    {
        "id": "d_zavg",
        "model": "zavg_d_hhi_supp",
        "kind": "zavg",
        "n_estimators": 0,
        "early_stopping_rounds": None,
        "shallow": True,
        "drop_size": False,
        "drop_counts": False,
        "notes": (
            "not a tree: train-only signed z-average of "
            "d_cust_hhi_lag3 (+) and d_n_supp_lag3; next idea, not XGB KEEP"
        ),
    },
)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    return out


def _family_of(col: str) -> str:
    return str(col).split("_", 1)[0]


def _stem(col: str) -> str:
    s = str(col)
    for k in LAGS:
        tail = f"_lag{k}"
        if s.endswith(tail):
            return s[: -len(tail)]
    return s


def _is_debt_leak(col: str) -> bool:
    """True for F / a_fin_cost / a_debt / m_debt and their lags."""
    s = str(col)
    return s.startswith(DEBT_LEAK_PREFIXES) or _family_of(s) == "f"


def _is_size(col: str) -> bool:
    return _stem(col) in SIZE_STEMS


def _is_count(col: str) -> bool:
    return _stem(col) in COUNT_STEMS


def recheck_acceptance() -> pd.DataFrame:
    """Refuse to revive rejected Y4 columns. Only y4_ds_r_double is in play."""
    if not ACCEPT.exists():
        print(f"WARN {ACCEPT} missing; proceeding with {Y_COL} only")
        return pd.DataFrame()
    acc = pd.read_csv(ACCEPT)
    y4 = acc[acc["column"].astype(str).str.startswith("y4_")].copy()
    print("Y4 rows in y_acceptance.csv")
    print(y4.to_string(index=False))
    for _, row in y4.iterrows():
        col = str(row["column"])
        accepted = int(row["accepted"]) if pd.notna(row["accepted"]) else 0
        if col == Y_COL and accepted != 1:
            raise RuntimeError(f"{Y_COL} is no longer accepted — stop")
        if col != Y_COL and accepted == 1:
            print(f"NOTE {col} is now accepted in the CSV; this module still ignores it")
        if col != Y_COL and accepted != 1:
            print(f"still rejected (not revived): {col}")
    return y4


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


def load_y(con, grid: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Prefer live y4_debt.build so we do not depend on a parent rewrite."""
    print("building Y4 via analysis.targets.y4_debt.build (live, not parquet)")
    y4 = _keys(build_y4(con, grid))
    if Y_COL not in y4.columns:
        raise RuntimeError(f"{Y_COL} missing from y4_debt.build")
    return y4[["company_id", "period", Y_COL]].copy(), "y4_debt.build"


def allowed_x_cols(
    df: pd.DataFrame,
    y_col: str,
    drop_size: bool = False,
    drop_counts: bool = False,
    families: tuple[str, ...] | None = None,
) -> list[str]:
    allow = set(families) if families else set(ALLOWED)
    cols: list[str] = []
    dropped: list[str] = []
    for c in df.columns:
        if c in KEY_COLS or c == y_col or str(c).startswith("y"):
            continue
        if _is_debt_leak(c):
            dropped.append(c)
            continue
        if drop_size and _is_size(c):
            dropped.append(c)
            continue
        if drop_counts and _is_count(c):
            dropped.append(c)
            continue
        if c in META_OK:
            if families is not None:
                continue
            if drop_counts and c in COUNT_STEMS:
                dropped.append(c)
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
            continue
        if _family_of(c) not in allow:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        cols.append(c)
    if dropped:
        shown = dropped[:16]
        more = f" (+{len(dropped) - 16} more)" if len(dropped) > 16 else ""
        print(f"dropped leak/size/count X ({len(dropped)}): {shown}{more}")
    return cols


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Past-only lags within company. shift(k) uses t-k, no look-ahead."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def usable_x_cols(df: pd.DataFrame, cols: list[str], mask: pd.Series, min_cov: float = MIN_X_COV) -> list[str]:
    n = int(mask.sum())
    keep = []
    for c in cols:
        x = pd.to_numeric(df.loc[mask, c], errors="coerce")
        if n == 0 or int(x.notna().sum()) / n < min_cov:
            continue
        if x.nunique(dropna=True) < 2:
            continue
        keep.append(c)
    return keep


def choose_sign(y: pd.Series, x: pd.Series) -> int:
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1
    if not np.isfinite(auc_p):
        return -1
    if not np.isfinite(auc_n):
        return 1
    return -1 if auc_n > auc_p else 1


def dummy_prior_scores(y_train: pd.Series, n: int) -> np.ndarray:
    clf = DummyClassifier(strategy="prior", random_state=FOLD_SEED)
    y = y_train.to_numpy(dtype=int)
    clf.fit(np.zeros((len(y), 1)), y)
    prior = float(clf.class_prior_[1] if len(clf.class_prior_) > 1 else clf.class_prior_[0])
    return np.full(n, prior)


def best_single_on_mask(
    X: pd.DataFrame,
    y: pd.Series,
    fit_mask: pd.Series,
    eval_mask: pd.Series,
    cols: list[str],
) -> dict:
    """Pick best signed feature on fit_mask; score eval_mask. Sign from fit only."""
    y_fit = y[fit_mask]
    n_fit = int(fit_mask.sum())
    rows = []
    for col in cols:
        if _is_debt_leak(col):
            continue
        x = pd.to_numeric(X[col], errors="coerce")
        x_fit = x[fit_mask]
        n_ok = int(x_fit.notna().sum())
        if n_fit and (n_ok / n_fit) < MIN_SINGLE_COV:
            continue
        if n_ok < MIN_SINGLE_N:
            continue
        if x_fit.nunique(dropna=True) < 2:
            continue
        sign = choose_sign(y_fit, x_fit)
        fit_auc = auroc(y_fit, sign * x_fit)
        if not np.isfinite(fit_auc):
            continue
        rows.append((col, sign, fit_auc, n_ok / n_fit if n_fit else float("nan")))
    if not rows:
        return {
            "feature": "",
            "sign": 1,
            "fit_auc": float("nan"),
            "eval_auc": float("nan"),
            "eval_pr": float("nan"),
            "n_eval": 0,
            "fit_cov": float("nan"),
        }
    col, sign, fit_auc, fit_cov = max(rows, key=lambda t: t[2])
    x = pd.to_numeric(X[col], errors="coerce")
    pair = pd.DataFrame({"y": y[eval_mask], "s": sign * x[eval_mask]}).dropna()
    return {
        "feature": col,
        "sign": sign,
        "fit_auc": fit_auc,
        "eval_auc": auroc(pair["y"], pair["s"]),
        "eval_pr": pr_auc(pair["y"], pair["s"]),
        "n_eval": int(len(pair)),
        "fit_cov": float(fit_cov),
    }


def _scale_pos_weight(y: pd.Series) -> float:
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 <= 0:
        return 1.0
    return n0 / n1


def _n_trees_xgb(clf: XGBClassifier, default: int) -> int:
    for attr in ("best_iteration", "best_ntree_limit"):
        v = getattr(clf, attr, None)
        if v is not None and np.isfinite(v) and int(v) > 0:
            return int(v) + (1 if attr == "best_iteration" else 0)
    booster = getattr(clf, "get_booster", lambda: None)()
    if booster is not None:
        try:
            return int(booster.num_boosted_rounds())
        except Exception:
            pass
    return int(default)


def fit_xgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame | None = None,
    y_va: pd.Series | None = None,
    n_estimators: int = 400,
    early_stopping_rounds: int | None = 40,
    shallow: bool = False,
) -> XGBClassifier:
    params = dict(XGB_SHALLOW if shallow else XGB_BASE)
    params["n_estimators"] = int(n_estimators)
    params["scale_pos_weight"] = _scale_pos_weight(y_tr)
    if X_va is not None and early_stopping_rounds:
        params["early_stopping_rounds"] = int(early_stopping_rounds)
    clf = XGBClassifier(**params)
    fit_kw: dict = {}
    if X_va is not None and y_va is not None:
        fit_kw["eval_set"] = [(X_va, y_va.to_numpy(dtype=int))]
        fit_kw["verbose"] = False
    clf.fit(X_tr, y_tr.to_numpy(dtype=int), **fit_kw)
    return clf


def fit_lgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame | None = None,
    y_va: pd.Series | None = None,
    n_estimators: int = 50,
    early_stopping_rounds: int | None = None,
) -> "lgb.LGBMClassifier":
    if lgb is None:
        raise RuntimeError("lightgbm is not installed")
    params = dict(LGB_SHALLOW)
    params["n_estimators"] = int(n_estimators)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(y_tr))
    clf = lgb.LGBMClassifier(**params)
    fit_kw: dict = {}
    if X_va is not None and y_va is not None and early_stopping_rounds:
        fit_kw["eval_set"] = [(X_va, y_va)]
        fit_kw["eval_metric"] = "auc"
        fit_kw["callbacks"] = [
            lgb.early_stopping(int(early_stopping_rounds), verbose=False),
            lgb.log_evaluation(period=0),
        ]
    clf.fit(X_tr, y_tr, **fit_kw)
    return clf


def _assert_allowed(cols: list[str], y_col: str) -> None:
    leak = leakage_check(cols, y_col, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage {y_col}: {leak['issues']}")
    bad_y = [c for c in cols if c == y_col or str(c).startswith("y")]
    if bad_y:
        raise RuntimeError(f"y columns in X: {bad_y[:8]}")
    extra = [c for c in cols if _is_debt_leak(c)]
    if extra:
        raise RuntimeError(f"debt-service clone in X: {extra[:8]}")
    fam_f = [c for c in cols if _family_of(c) == "f"]
    if fam_f:
        raise RuntimeError(f"family F leaked into X: {fam_f[:8]}")


def leak_screen(
    panel: pd.DataFrame,
    x_cols: list[str],
    mask: pd.Series,
    probes: list[str],
) -> dict:
    """Max |Pearson| of each X vs f_ds_r / lags on train labeled. Fail if >= 0.80."""
    rows = []
    for col in x_cols:
        x = pd.to_numeric(panel.loc[mask, col], errors="coerce")
        for pr in probes:
            if pr not in panel.columns:
                continue
            p = pd.to_numeric(panel.loc[mask, pr], errors="coerce")
            d = pd.DataFrame({"x": x, "p": p}).dropna()
            if len(d) < 50 or d["x"].nunique() < 2 or d["p"].nunique() < 2:
                continue
            rho = float(d["x"].corr(d["p"]))
            if not np.isfinite(rho):
                continue
            rows.append({"x": col, "probe": pr, "rho": rho, "n": int(len(d))})
    if not rows:
        return {
            "ok": True,
            "max_abs_rho": float("nan"),
            "worst": None,
            "n_pairs": 0,
            "offenders": [],
        }
    tab = pd.DataFrame(rows)
    tab["abs_rho"] = tab["rho"].abs()
    tab = tab.sort_values("abs_rho", ascending=False)
    worst = tab.iloc[0]
    offenders = tab[tab["abs_rho"] >= LEAK_RHO]
    print(
        f"leak screen: max |ρ|={worst['abs_rho']:.4f} "
        f"{worst['x']} vs {worst['probe']} n={int(worst['n'])} "
        f"offenders={len(offenders)}"
    )
    if len(offenders):
        print(offenders.head(12).to_string(index=False))
    return {
        "ok": bool(len(offenders) == 0),
        "max_abs_rho": float(worst["abs_rho"]),
        "worst": {
            "x": str(worst["x"]),
            "probe": str(worst["probe"]),
            "rho": float(worst["rho"]),
            "n": int(worst["n"]),
        },
        "n_pairs": int(len(tab)),
        "offenders": offenders["x"].drop_duplicates().tolist(),
        "top": tab.head(8).to_dict(orient="records"),
    }


def oof_named_singles(
    panel: pd.DataFrame,
    y: pd.Series,
    train_lab: pd.Series,
    cols: list[str],
) -> list[dict]:
    """Same-fold signed AUROC for named columns (honesty vs gain leaders)."""
    rows = []
    for col in cols:
        if col not in panel.columns:
            continue
        aucs = []
        for k in range(N_FOLDS):
            tr = train_lab & (panel["fold"] != k)
            va = train_lab & (panel["fold"] == k)
            sf = best_single_on_mask(panel, y, tr, va, [col])
            if np.isfinite(sf["eval_auc"]):
                aucs.append(sf["eval_auc"])
        rows.append(
            {
                "feature": col,
                "cv": float(np.nanmean(aucs)) if aucs else float("nan"),
                "n_folds": len(aucs),
            }
        )
    rows.sort(key=lambda r: (-r["cv"] if np.isfinite(r["cv"]) else 0.0))
    return rows


def best_oof_single(
    panel: pd.DataFrame,
    y: pd.Series,
    train_lab: pd.Series,
    cols: list[str],
) -> dict:
    """Best allowed feature by mean OOF signed AUROC (sign from each train fold).

    The per-fold *train-best pick* can be a 0.50 noise baseline when the
    winning train feature fails on val. KEEP must beat the best *fixed*
    feature on the same folds (Y5 honesty).
    """
    ranked = oof_named_singles(panel, y, train_lab, cols)
    ranked = [r for r in ranked if np.isfinite(r["cv"])]
    if not ranked:
        return {"feature": "", "cv": float("nan"), "top5": []}
    print("best OOF singles (fixed feature, sign from train fold):")
    for row in ranked[:8]:
        print(f"  {row['feature']}: {row['cv']:.4f}")
    top = ranked[0]
    return {"feature": top["feature"], "cv": top["cv"], "top5": ranked[:5]}


def size_screen(panel: pd.DataFrame, y: pd.Series, mask: pd.Series) -> dict:
    """Acceptance-style size AUROC of log1p(a_in3) vs Y on train labeled."""
    if "a_in3" not in panel.columns:
        return {"ok": False, "auroc": float("nan"), "n": 0, "reason": "a_in3 missing"}
    score = np.log1p(pd.to_numeric(panel.loc[mask, "a_in3"], errors="coerce").abs())
    auc = auroc(y[mask], score)
    n = int(pd.DataFrame({"y": y[mask], "s": score}).dropna().shape[0])
    ok = bool(np.isfinite(auc) and auc < SIZE_AUROC_MAX)
    print(f"size screen: log1p(|a_in3|) AUROC={auc:.4f} n={n} ok={ok} (need < {SIZE_AUROC_MAX})")
    return {"ok": ok, "auroc": float(auc), "n": n}


def _verdict(cv: float, single: float, dummy: float, collapsed: bool, size_ok: bool, leak_ok: bool) -> dict:
    if not leak_ok:
        return {"decision": "PARK", "reason": "leak screen failed (|ρ|>=0.80 vs f_ds_r)"}
    if not size_ok:
        return {"decision": "PARK", "reason": "size screen failed (log1p(a_in3) AUROC >= 0.60)"}
    if not np.isfinite(cv):
        return {"decision": "PARK", "reason": "CV AUROC undefined"}
    if cv <= dummy:
        return {"decision": "PARK", "reason": f"CV {cv:.3f} does not beat dummy {dummy:.3f}"}
    if collapsed:
        return {
            "decision": "PARK",
            "reason": (
                f"trees collapsed to <= {COLLAPSE_MAX_TREES}; "
                "do not KEEP a 400+ES collapse (run 50/depth-3)"
            ),
        }
    if not np.isfinite(single):
        return {"decision": "PARK", "reason": "best-single CV undefined"}
    gap = cv - single
    if gap >= CLEAR_MARGIN:
        return {
            "decision": "KEEP",
            "reason": f"CV {cv:.3f} >= single {single:.3f} + {CLEAR_MARGIN:.2f} and beats dummy",
        }
    if abs(gap) < CLEAR_MARGIN:
        return {
            "decision": "CLOSE",
            "reason": f"CV {cv:.3f} within {CLEAR_MARGIN:.2f} of single {single:.3f} (honest tie)",
        }
    return {
        "decision": "PARK",
        "reason": f"CV {cv:.3f} loses to single {single:.3f} by {abs(gap):.3f}",
    }


def top_gain(clf, cols: list[str], n: int = 10) -> list[tuple[str, float]]:
    if hasattr(clf, "feature_importances_"):
        imp = pd.Series(clf.feature_importances_, index=cols).sort_values(ascending=False)
        return [(str(i), float(v)) for i, v in imp.head(n).items() if float(v) > 0]
    return []


def top_shap_names(clf, X: pd.DataFrame, n: int = 10) -> list[str]:
    """Train-only SHAP names. No plots. Skip quietly if shap is missing."""
    try:
        import shap
    except ImportError:
        print("shap not installed; gain names only")
        return []
    try:
        explainer = shap.TreeExplainer(clf)
        vals = explainer.shap_values(X)
        if isinstance(vals, list):
            vals = vals[1] if len(vals) > 1 else vals[0]
        mean_abs = np.mean(np.abs(np.asarray(vals)), axis=0)
        order = np.argsort(-mean_abs)[:n]
        names = [str(X.columns[i]) for i in order if mean_abs[i] > 0]
        print(f"SHAP top {len(names)} (train only): {names}")
        return names
    except Exception as exc:
        print(f"SHAP skipped: {type(exc).__name__}: {exc}")
        return []


def append_registry(rows: list[dict]) -> None:
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _fmt(v) -> str:
    return f"{v:.6g}" if isinstance(v, (int, float, np.floating)) and np.isfinite(v) else ""


def registry_rows(res: dict, ts: str) -> list[dict]:
    fam = "+".join(s.upper() for s in ALLOWED)
    y = res["y"]
    notes = (
        f"{res['decision']}; {res['notes']}; q3_turning/q5_why; never F; "
        f"drop a_fin_cost*,a_debt*,m_debt*; lags=1,3; n_x={res['n_x']}; "
        f"trees_median={res['trees_median']}; collapsed={res['collapsed']}; "
        f"cv={res['cv_auroc']:.4f}±{res['cv_auroc_sd']:.4f}; "
        f"dummy={res['cv_dummy']:.3f}; single_oof={res['best_single']} {res['cv_single']:.4f}; "
        f"single_pick={res.get('single_pick_most')} {res.get('cv_single_pick')}; "
        f"gap_single={res['gap_single']}; leak_max_abs_rho={res['leak_max_abs_rho']}; "
        f"size_auroc={res['size_auroc']}; hold_n_pos={res['n_hold_pos']} {res['hold_power']}; "
        f"{res['reason']}"
    )
    cov = res["train_coverage_grid"]
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": res["model"],
            "split": "cv5_group",
            "metric": "auroc",
            "value": _fmt(res["cv_auroc"]),
            "coverage": f"{cov:.4f}" if np.isfinite(cov) else "",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": res["model"],
            "split": "cv5_group",
            "metric": "auroc_sd",
            "value": _fmt(res["cv_auroc_sd"]),
            "coverage": f"{cov:.4f}" if np.isfinite(cov) else "",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": "dummy_prior",
            "split": "cv5_group",
            "metric": "auroc",
            "value": _fmt(res["cv_dummy"]),
            "coverage": f"{cov:.4f}" if np.isfinite(cov) else "",
            "notes": "constant 0.50; DummyClassifier prior on each train fold",
        },
    ]
    if res.get("best_single"):
        rows.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": _family_of(str(res["best_single"])).upper(),
                "y": y,
                "model": f"single_{res['best_single']}",
                "split": "cv5_group",
                "metric": "auroc",
                "value": _fmt(res["cv_single"]),
                "coverage": f"{cov:.4f}" if np.isfinite(cov) else "",
                "notes": (
                    f"best fixed OOF single among allowed X; feat={res['best_single']}; "
                    f"train_pick_most={res.get('single_pick_most')}; "
                    f"counts={res['single_counts']}"
                ),
            }
        )
    if np.isfinite(res.get("hold_auroc", float("nan"))):
        rows.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": fam,
                "y": y,
                "model": res["model"],
                "split": "holdout",
                "metric": "auroc",
                "value": _fmt(res["hold_auroc"]),
                "coverage": (
                    f"{res['hold_coverage']:.4f}" if np.isfinite(res.get("hold_coverage", float("nan"))) else ""
                ),
                "notes": (
                    f"LOW_POWER check only; n_pos={res['n_hold_pos']}; "
                    f"not used for KEEP/PARK; {res['decision']}"
                ),
            }
        )
    return rows


def run_variant(
    spec: dict,
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    hold_ids: set[str],
    write_registry: bool = True,
    do_shap: bool = False,
) -> dict:
    print("\n" + "=" * 72)
    print(f"VARIANT {spec['id']} model={spec['model']} {spec['notes']}")
    print("=" * 72)

    df = panel.merge(folds[["company_id", "fold", "group_id"]], on="company_id", how="left")

    drop_counts = bool(spec.get("drop_counts", False))
    fams = spec.get("families")
    base_cols = allowed_x_cols(
        df,
        Y_COL,
        drop_size=spec["drop_size"],
        drop_counts=drop_counts,
        families=fams,
    )
    _assert_allowed(base_cols, Y_COL)
    lag_src = [c for c in base_cols if c not in META_OK]
    probe_src = [c for c in ("f_ds_r",) if c in df.columns]
    print(f"base X={len(base_cols)}; adding lags {list(LAGS)} on {len(lag_src)} (meta not lagged)")
    # One sort/reset so later masks stay aligned. f_ds_r lags are leak probes, not X.
    df = add_lags(df, lag_src + probe_src, LAGS)
    probes = [c for c in ("f_ds_r", "f_ds_r_lag1", "f_ds_r_lag3") if c in df.columns]
    if not probes:
        print("WARN f_ds_r missing — leak screen cannot run against family F")

    # Masks AFTER add_lags (it sorts + reset_index). fold is NaN on holdout.
    is_hold = df["company_id"].astype(str).isin(hold_ids)
    labeled = df[Y_COL].notna()
    train_lab = (~is_hold) & labeled & df["fold"].notna()
    hold_lab = is_hold & labeled
    assert_no_holdout(df.loc[train_lab, "company_id"])

    feat_cols = allowed_x_cols(
        df,
        Y_COL,
        drop_size=spec["drop_size"],
        drop_counts=drop_counts,
        families=fams,
    )
    stems = spec.get("stems")
    if stems:
        allow_stems = set(stems)
        feat_cols = [c for c in feat_cols if _stem(c) in allow_stems]
        print(f"stem filter {stems} -> {feat_cols}")
    feat_cols = usable_x_cols(df, feat_cols, train_lab, MIN_X_COV)
    _assert_allowed(feat_cols, Y_COL)
    print(f"usable X after train coverage>={MIN_X_COV}: {len(feat_cols)}")

    leak = leak_screen(df, feat_cols, train_lab, probes)
    if leak["offenders"]:
        print(f"dropping {len(leak['offenders'])} leak offenders and re-screening")
        feat_cols = [c for c in feat_cols if c not in set(leak["offenders"])]
        _assert_allowed(feat_cols, Y_COL)
        leak = leak_screen(df, feat_cols, train_lab, probes)
        if not leak["ok"]:
            raise RuntimeError(f"leak screen still failing after drop: {leak['offenders'][:8]}")

    size = size_screen(df, df[Y_COL], train_lab)
    n_tr = int(train_lab.sum())
    n_tr_pos = int((df.loc[train_lab, Y_COL] == 1).sum())
    n_tr_grid = int((~is_hold).sum())
    rate = float(df.loc[train_lab, Y_COL].mean()) if n_tr else float("nan")
    print(
        f"train labeled={n_tr} pos={n_tr_pos} rate={rate:.4f} "
        f"coverage={n_tr / n_tr_grid if n_tr_grid else float('nan'):.4f} "
        f"hold_labeled={int(hold_lab.sum())}"
    )

    cv_model, cv_dummy, cv_single = [], [], []
    cv_single_feats: list[str] = []
    best_iters: list[int] = []
    fold_rows = []
    for k in range(N_FOLDS):
        tr_m = train_lab & (df["fold"] != k)
        va_m = train_lab & (df["fold"] == k)
        if int(tr_m.sum()) < 50 or int(va_m.sum()) < 20:
            print(f"  fold {k}: skip (n_tr={int(tr_m.sum())} n_va={int(va_m.sum())})")
            continue
        y_tr, y_va = df.loc[tr_m, Y_COL].astype(float), df.loc[va_m, Y_COL].astype(float)
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            print(f"  fold {k}: skip (one class)")
            continue
        X_tr = df.loc[tr_m, feat_cols]
        X_va = df.loc[va_m, feat_cols]
        assert_no_holdout(df.loc[tr_m, "company_id"])

        d_sc = dummy_prior_scores(y_tr, int(va_m.sum()))
        d_auc = auroc(y_va, d_sc)
        cv_dummy.append(d_auc)

        sf = best_single_on_mask(df, df[Y_COL], tr_m, va_m, feat_cols)
        cv_single.append(sf["eval_auc"])
        cv_single_feats.append(sf["feature"])

        if spec["kind"] == "xgb":
            clf = fit_xgb(
                X_tr,
                y_tr,
                X_va,
                y_va,
                n_estimators=spec["n_estimators"],
                early_stopping_rounds=spec["early_stopping_rounds"],
                shallow=spec["shallow"],
            )
            trees = _n_trees_xgb(clf, spec["n_estimators"])
        else:
            clf = fit_lgb(
                X_tr,
                y_tr,
                X_va if spec["early_stopping_rounds"] else None,
                y_va if spec["early_stopping_rounds"] else None,
                n_estimators=spec["n_estimators"],
                early_stopping_rounds=spec["early_stopping_rounds"],
            )
            trees = int(
                getattr(clf, "best_iteration_", spec["n_estimators"]) or spec["n_estimators"]
            )
        pred = clf.predict_proba(X_va)[:, 1]
        m_auc = auroc(y_va, pred)
        cv_model.append(m_auc)
        best_iters.append(trees)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(m_auc),
                "single": float(sf["eval_auc"]) if np.isfinite(sf["eval_auc"]) else float("nan"),
                "feat": sf["feature"],
                "dummy": float(d_auc),
                "trees": int(trees),
                "n_va": int(va_m.sum()),
                "n_pos": int((y_va == 1).sum()),
                "va_index": df.index[va_m].to_numpy(),
                "pred": np.asarray(pred, dtype=float),
                "y_va": y_va.to_numpy(dtype=float),
            }
        )
        print(
            f"  fold {k}: model={m_auc:.4f} single={sf['eval_auc']:.4f} "
            f"feat={sf['feature']} dummy={d_auc:.4f} trees={trees} "
            f"n_va={int(va_m.sum())} pos={int((y_va == 1).sum())}"
        )

    cv_auroc = float(np.nanmean(cv_model)) if cv_model else float("nan")
    cv_sd = float(np.nanstd(cv_model, ddof=1)) if len(cv_model) > 1 else float("nan")
    cv_s = float(np.nanmean(cv_single)) if cv_single else float("nan")
    cv_d = float(np.nanmean(cv_dummy)) if cv_dummy else DUMMY_AUROC
    trees_median = int(np.median(best_iters)) if best_iters else int(spec["n_estimators"])
    collapsed = bool(best_iters) and (
        int(np.median(best_iters)) <= COLLAPSE_MAX_TREES
        or min(best_iters) <= COLLAPSE_MAX_TREES
    )
    print(
        f"CV model={cv_auroc:.4f}±{cv_sd:.4f} single={cv_s:.4f} dummy={cv_d:.4f} "
        f"trees_median={trees_median} collapsed={collapsed} n_x={len(feat_cols)}"
    )

    # Holdout check: fit on all train. Never ES on holdout. Not used for KEEP.
    y_tr_all = df.loc[train_lab, Y_COL].astype(float)
    X_tr_all = df.loc[train_lab, feat_cols]
    assert_no_holdout(df.loc[train_lab, "company_id"])
    if spec["kind"] == "xgb":
        n_final = spec["n_estimators"] if spec["shallow"] else max(50, trees_median)
        if collapsed and not spec["shallow"]:
            n_final = 50
        final = fit_xgb(
            X_tr_all,
            y_tr_all,
            n_estimators=n_final,
            early_stopping_rounds=None,
            shallow=spec["shallow"] or collapsed,
        )
    else:
        final = fit_lgb(X_tr_all, y_tr_all, n_estimators=spec["n_estimators"])

    y_ho = df.loc[hold_lab, Y_COL].astype(float)
    n_hold_pos = int((y_ho == 1).sum()) if len(y_ho) else 0
    hold_power = "LOW_POWER" if n_hold_pos < LOW_POWER_POS else "USABLE"
    if hold_lab.any():
        pred_ho = final.predict_proba(df.loc[hold_lab, feat_cols])[:, 1]
        ho_auc = auroc(y_ho, pred_ho)
        ho_pr = pr_auc(y_ho, pred_ho)
        sf_all = best_single_on_mask(df, df[Y_COL], train_lab, hold_lab, feat_cols)
    else:
        pred_ho = np.array([])
        ho_auc = float("nan")
        ho_pr = float("nan")
        sf_all = {"feature": "", "eval_auc": float("nan")}
    print(
        f"HOLDOUT check AUROC={ho_auc:.4f} PR={ho_pr:.4f} "
        f"n={int(hold_lab.sum())} n_pos={n_hold_pos} {hold_power} "
        f"(not a KEEP claim)"
    )

    gain = top_gain(final, feat_cols, 10)
    print("top 10 gain (train fit, not a claim):", [n for n, _ in gain])
    shap_names: list[str] = []
    if do_shap:
        shap_names = top_shap_names(final, X_tr_all, 10)
    oof_single = best_oof_single(df, df[Y_COL], train_lab, feat_cols)
    named = oof_single.get("top5") or []
    cc_aucs = []
    cc_n = []
    win = oof_single.get("feature") or ""
    if win and win in df.columns:
        wx = pd.to_numeric(df[win], errors="coerce")
        for fr in fold_rows:
            idx = fr.get("va_index")
            if idx is None:
                continue
            ok = wx.loc[idx].notna().to_numpy()
            if ok.sum() < 20 or pd.Series(fr["y_va"][ok]).nunique() < 2:
                continue
            cc_aucs.append(auroc(fr["y_va"][ok], fr["pred"][ok]))
            cc_n.append(int(ok.sum()))
        if cc_aucs:
            print(
                f"complete-case model AUROC on {win} defined rows: "
                f"{float(np.mean(cc_aucs)):.4f} mean n_va={float(np.mean(cc_n)):.0f} "
                f"(vs single {oof_single['cv']:.4f})"
            )

    pick_counts: dict[str, int] = {}
    for f in cv_single_feats:
        if f:
            pick_counts[f] = pick_counts.get(f, 0) + 1
    best_single = max(pick_counts, key=pick_counts.get) if pick_counts else (sf_all.get("feature") or "")

    size_is_story = bool(
        gain
        and (
            _is_size(gain[0][0])
            or _stem(gain[0][0]) in {"a_op_in", "a_in3", "a_in6", "a_in12"}
        )
    )
    # KEEP bar = best *fixed* OOF single, not the noisy train-pick mean.
    single_bar = oof_single["cv"] if np.isfinite(oof_single.get("cv", float("nan"))) else cv_s
    verd = _verdict(cv_auroc, single_bar, DUMMY_AUROC, collapsed, size["ok"], leak["ok"])
    if verd["decision"] == "KEEP" and size_is_story:
        verd = {
            "decision": "PARK",
            "reason": (
                f"top gain is inflow SIZE ({gain[0][0]}); "
                "model only rediscovers size — PARK"
            ),
        }
    print(f"VERDICT {verd['decision']}: {verd['reason']}")
    cc_mean = float(np.mean(cc_aucs)) if cc_aucs else float("nan")
    slim_folds = []
    for fr in fold_rows:
        slim_folds.append({k: v for k, v in fr.items() if k not in {"va_index", "pred", "y_va"}})

    out = {
        "y": Y_COL,
        "variant": spec["id"],
        "model": spec["model"],
        "notes": spec["notes"],
        "kind": spec["kind"],
        "n_x": len(feat_cols),
        "cv_auroc": cv_auroc,
        "cv_auroc_sd": cv_sd,
        "cv_folds": slim_folds,
        "cc_model_auroc": cc_mean,
        "cc_n_mean": float(np.mean(cc_n)) if cc_n else float("nan"),
        "cv_dummy": float(cv_d) if np.isfinite(cv_d) else DUMMY_AUROC,
        "cv_single_pick": cv_s,
        "cv_single": float(single_bar) if np.isfinite(single_bar) else cv_s,
        "best_single": oof_single.get("feature") or best_single,
        "single_pick_most": best_single,
        "single_counts": pick_counts,
        "gap_single": (
            float(cv_auroc - single_bar)
            if np.isfinite(cv_auroc) and np.isfinite(single_bar)
            else float("nan")
        ),
        "gap_dummy": float(cv_auroc - DUMMY_AUROC) if np.isfinite(cv_auroc) else float("nan"),
        "trees_median": trees_median,
        "trees_folds": best_iters,
        "collapsed": collapsed,
        "leak_ok": leak["ok"],
        "leak_max_abs_rho": leak["max_abs_rho"],
        "leak_worst": leak["worst"],
        "size_ok": size["ok"],
        "size_auroc": size["auroc"],
        "train_labeled": n_tr,
        "train_pos": n_tr_pos,
        "train_base_rate": rate,
        "train_coverage_grid": n_tr / n_tr_grid if n_tr_grid else float("nan"),
        "n_hold_labeled": int(hold_lab.sum()),
        "n_hold_pos": n_hold_pos,
        "hold_auroc": float(ho_auc),
        "hold_pr_auc": float(ho_pr),
        "hold_single": sf_all.get("eval_auc", float("nan")),
        "hold_coverage": float(len(pred_ho) / hold_lab.sum()) if hold_lab.sum() else float("nan"),
        "hold_power": hold_power,
        "top_gain": gain,
        "shap_names": shap_names,
        "named_singles": named,
        "decision": verd["decision"],
        "reason": verd["reason"],
        "forbidden_x": list(Y4_META.get("forbidden_x_families", ["f"])),
        "dropped_clones": "a_fin_cost*, a_debt* (ds_r numerator), m_debt* (not in store)",
    }
    if write_registry:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
        rows = registry_rows(out, ts)
        append_registry(rows)
        print(f"appended {len(rows)} registry rows")
    return out


def write_output_md(results: list[dict], started: str) -> None:
    lines = [
        "# Y4 XGBoost — y4_ds_r_double",
        "",
        f"- **When:** {datetime.now().isoformat(timespec='minutes')}",
        f"- **Started:** {started}",
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        f"- **Holdout:** 72 companies, seed {FOLD_SEED}. Never fitted.",
        f"- **Y:** `{Y_COL}` only (accepted 13.9% train). Other Y4 columns not revived.",
        "- **X:** A (no a_debt* / a_fin_cost*) + B C D E G H. **Never F.** Lags 1, 3.",
        "- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER.",
        "- **Brief:** Q3 turning / Q5 why. Not bankruptcy. Not a 0–100.",
        "",
        "## Table",
        "",
        "| variant | CV AUROC ± sd | dummy | best single | gap vs single | n_x | trees | collapsed | hold n_pos / AUROC | decision |",
        "|---|---:|---:|---|---:|---:|---:|---|---|---|",
    ]
    for r in results:
        single = f"{r['best_single']} {r['cv_single']:.3f}" if r.get("best_single") else f"{r['cv_single']:.3f}"
        lines.append(
            f"| {r['variant']} | {r['cv_auroc']:.3f} ± {r['cv_auroc_sd']:.3f} | "
            f"{r['cv_dummy']:.2f} | {single} | {r['gap_single']:+.3f} | {r['n_x']} | "
            f"{r['trees_median']} | {'yes' if r['collapsed'] else 'no'} | "
            f"{r['n_hold_pos']} / {r['hold_auroc']:.3f} | **{r['decision']}** |"
        )
    lines += [
        "",
        "## Screens",
        "",
    ]
    if results:
        r0 = results[0]
        lines.append(
            f"- Leak vs `f_ds_r` / lags on train labeled: max |ρ| = "
            f"{r0['leak_max_abs_rho']:.4f} (fail ≥ {LEAK_RHO:.2f}). "
            f"Worst: {r0['leak_worst']}."
        )
        lines.append(
            f"- Size `log1p(|a_in3|)` AUROC = {r0['size_auroc']:.4f} "
            f"(need < {SIZE_AUROC_MAX:.2f})."
        )
        lines.append(f"- Dropped clones: {r0['dropped_clones']}.")
    lines += [
        "",
        "## Mapping",
        "",
        "Y4 doubling of debt-service / inflow is *who is turning* (Q3) and *why* (Q5).",
        "A model that only ranks on inflow size is not a debt-pressure reading.",
        "",
        "## Top gain / SHAP names (train only)",
        "",
    ]
    for r in results:
        names = [n for n, _ in r.get("top_gain") or []]
        shap_n = r.get("shap_names") or []
        lines.append(f"- **{r['variant']}** gain: {', '.join(names) if names else '—'}")
        if shap_n:
            lines.append(f"  SHAP: {', '.join(shap_n)}")
    lines += ["", "## Reasons", ""]
    for r in results:
        lines.append(f"- **{r['variant']} {r['decision']}:** {r['reason']}")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


ZAVG_STEMS = ("d_cust_hhi", "d_n_supp")


def run_zavg(
    spec: dict,
    panel: pd.DataFrame,
    folds: pd.DataFrame,
    hold_ids: set[str],
    write_registry: bool = True,
) -> dict:
    """Train-only signed z-average of d_cust_hhi_lag3 and d_n_supp_lag3.

    Not a tree. Moments and signs from the train fold only. Holdout is a check.
    """
    print("\n" + "=" * 72)
    print(f"VARIANT {spec['id']} model={spec['model']} {spec['notes']}")
    print("=" * 72)
    df = panel.merge(folds[["company_id", "fold", "group_id"]], on="company_id", how="left")
    stems = [c for c in ZAVG_STEMS if c in df.columns]
    if len(stems) != 2:
        raise RuntimeError(f"zavg missing stems {ZAVG_STEMS}; have {stems}")
    _assert_allowed(stems, Y_COL)
    df = add_lags(df, stems, LAGS)
    cols = [f"{s}_lag3" for s in stems]
    for c in cols:
        if c not in df.columns:
            raise RuntimeError(f"missing {c}")
        if _is_debt_leak(c):
            raise RuntimeError(f"zavg leaked debt clone {c}")

    is_hold = df["company_id"].astype(str).isin(hold_ids)
    labeled = df[Y_COL].notna()
    train_lab = (~is_hold) & labeled & df["fold"].notna()
    hold_lab = is_hold & labeled
    assert_no_holdout(df.loc[train_lab, "company_id"])
    leak = leak_screen(df, cols, train_lab, [c for c in ("f_ds_r", "f_ds_r_lag1", "f_ds_r_lag3") if c in df.columns])
    size = size_screen(df, df[Y_COL], train_lab)
    n_tr = int(train_lab.sum())
    n_tr_pos = int((df.loc[train_lab, Y_COL] == 1).sum())
    rate = float(df.loc[train_lab, Y_COL].mean()) if n_tr else float("nan")
    print(f"train labeled={n_tr} pos={n_tr_pos} rate={rate:.4f} cols={cols}")

    fold_rows = []
    cv_model, cv_dummy = [], []
    for k in range(N_FOLDS):
        tr_m = train_lab & (df["fold"] != k)
        va_m = train_lab & (df["fold"] == k)
        y_tr = df.loc[tr_m, Y_COL].astype(float)
        y_va = df.loc[va_m, Y_COL].astype(float)
        assert_no_holdout(df.loc[tr_m, "company_id"])
        parts = []
        for c in cols:
            x = pd.to_numeric(df[c], errors="coerce")
            sign = choose_sign(y_tr, x[tr_m])
            mu = float(x[tr_m].mean())
            sd = float(x[tr_m].std(ddof=0))
            z = (x - mu) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0
            parts.append(sign * z)
        score = sum(parts)
        m_auc = auroc(y_va, score[va_m])
        d_auc = auroc(y_va, dummy_prior_scores(y_tr, int(va_m.sum())))
        cv_model.append(m_auc)
        cv_dummy.append(d_auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(m_auc),
                "dummy": float(d_auc),
                "n_va": int(va_m.sum()),
                "n_pos": int((y_va == 1).sum()),
            }
        )
        print(
            f"  fold {k}: zavg={m_auc:.4f} dummy={d_auc:.4f} "
            f"n_va={int(va_m.sum())} pos={int((y_va == 1).sum())}"
        )

    cv_auroc = float(np.nanmean(cv_model)) if cv_model else float("nan")
    cv_sd = float(np.nanstd(cv_model, ddof=1)) if len(cv_model) > 1 else float("nan")
    oof_single = best_oof_single(df, df[Y_COL], train_lab, cols)
    single_bar = oof_single["cv"] if np.isfinite(oof_single.get("cv", float("nan"))) else float("nan")
    # Not an XGB KEEP. Record the number; trees stay PARK.
    verd = {
        "decision": "PARK",
        "reason": (
            f"zavg CV {cv_auroc:.3f} vs single {single_bar:.3f} — "
            "reproducible next idea, not an XGB/LGBM KEEP"
        ),
    }
    print(f"CV zavg={cv_auroc:.4f}±{cv_sd:.4f} single={single_bar:.4f}")
    print(f"VERDICT {verd['decision']}: {verd['reason']}")

    # Holdout check: moments from all train. Never a KEEP claim.
    y_tr_all = df.loc[train_lab, Y_COL].astype(float)
    parts = []
    for c in cols:
        x = pd.to_numeric(df[c], errors="coerce")
        sign = choose_sign(y_tr_all, x[train_lab])
        mu = float(x[train_lab].mean())
        sd = float(x[train_lab].std(ddof=0))
        z = (x - mu) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0
        parts.append(sign * z)
    score = sum(parts)
    y_ho = df.loc[hold_lab, Y_COL].astype(float)
    ho_auc = auroc(y_ho, score[hold_lab]) if hold_lab.any() else float("nan")
    n_hold_pos = int((y_ho == 1).sum()) if len(y_ho) else 0
    print(f"HOLDOUT check AUROC={ho_auc:.4f} n_pos={n_hold_pos} LOW_POWER")

    n_tr_grid = int((~is_hold).sum())
    out = {
        "y": Y_COL,
        "variant": spec["id"],
        "model": spec["model"],
        "notes": spec["notes"],
        "kind": "zavg",
        "n_x": 2,
        "cv_auroc": cv_auroc,
        "cv_auroc_sd": cv_sd,
        "cv_folds": fold_rows,
        "cv_dummy": DUMMY_AUROC,
        "cv_single_pick": single_bar,
        "cv_single": float(single_bar) if np.isfinite(single_bar) else float("nan"),
        "best_single": oof_single.get("feature") or cols[0],
        "single_pick_most": oof_single.get("feature") or "",
        "single_counts": {},
        "gap_single": (
            float(cv_auroc - single_bar)
            if np.isfinite(cv_auroc) and np.isfinite(single_bar)
            else float("nan")
        ),
        "gap_dummy": float(cv_auroc - DUMMY_AUROC) if np.isfinite(cv_auroc) else float("nan"),
        "trees_median": 0,
        "trees_folds": [],
        "collapsed": False,
        "leak_ok": leak["ok"],
        "leak_max_abs_rho": leak["max_abs_rho"],
        "leak_worst": leak["worst"],
        "size_ok": size["ok"],
        "size_auroc": size["auroc"],
        "train_labeled": n_tr,
        "train_pos": n_tr_pos,
        "train_base_rate": rate,
        "train_coverage_grid": n_tr / n_tr_grid if n_tr_grid else float("nan"),
        "n_hold_labeled": int(hold_lab.sum()),
        "n_hold_pos": n_hold_pos,
        "hold_auroc": float(ho_auc),
        "hold_pr_auc": float("nan"),
        "hold_single": float("nan"),
        "hold_coverage": float("nan"),
        "hold_power": "LOW_POWER",
        "top_gain": [(c, 1.0) for c in cols],
        "shap_names": [],
        "named_singles": oof_single.get("top5") or [],
        "decision": verd["decision"],
        "reason": verd["reason"],
        "forbidden_x": list(Y4_META.get("forbidden_x_families", ["f"])),
        "dropped_clones": "a_fin_cost*, a_debt*, m_debt*; zavg uses D lags only",
    }
    if write_registry:
        ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
        rows = registry_rows(out, ts)
        append_registry(rows)
        print(f"appended {len(rows)} registry rows")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Y4 XGB / LGBM on y4_ds_r_double")
    p.add_argument(
        "--variant",
        action="append",
        dest="variants",
        default=None,
        help="variant id (repeatable). default: xgb_es",
    )
    p.add_argument("--all", action="store_true", help="run every variant")
    p.add_argument("--list", action="store_true", help="print variant ids and exit")
    p.add_argument("--shap", action="store_true", help="train-only SHAP names on the last variant")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument(
        "--md",
        action="store_true",
        help="overwrite analysis/outputs/y4_xgb.md (default: leave the handwritten note)",
    )
    p.add_argument("--no-md", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def print_label_mechanism(store: pd.DataFrame, y_panel: pd.DataFrame) -> None:
    """Train-only: Y4 doubling is mostly a future inflow crash. Not used as X."""
    from analysis.targets.y4_debt import debt_month_panel

    is_train = ~store["company_id"].astype(str).isin(load_holdout())
    keys = store.loc[is_train, ["company_id", "period"]].copy()
    keys = keys.merge(y_panel[["company_id", "period", Y_COL]], on=["company_id", "period"], how="left")
    labeled = keys[Y_COL].notna()
    assert_no_holdout(keys.loc[labeled, "company_id"])
    if "d_cust_hhi" in store.columns:
        tmp = store[["company_id", "period", "d_cust_hhi"]].copy()
        tmp = tmp.sort_values(["company_id", "period"])
        tmp["hhi_l3"] = tmp.groupby("company_id", sort=False)["d_cust_hhi"].shift(3)
        s = keys.loc[labeled, ["company_id", "period", Y_COL]].merge(
            tmp[["company_id", "period", "hhi_l3"]], on=["company_id", "period"], how="left"
        )
        d = s.dropna(subset=["hhi_l3"])
        if len(d) >= 50:
            d = d.copy()
            d["q"] = pd.qcut(d["hhi_l3"], 5, duplicates="drop")
            rates = d.groupby("q", observed=True)[Y_COL].mean()
            print("Y rate by d_cust_hhi_lag3 quintile (train labeled, complete):")
            print(rates.to_string())
    try:
        con = connect()
        flows = debt_month_panel(con).rename(columns={"month": "period"})
        con.close()
    except Exception as exc:
        print(f"mechanism flows skipped: {type(exc).__name__}: {exc}")
        return
    flows["company_id"] = flows["company_id"].astype(str)
    flows["period"] = pd.to_datetime(flows["period"])
    flows = flows.sort_values(["company_id", "period"])
    g = flows.groupby("company_id", sort=False)
    flows["in3_f3"] = g["in3"].shift(-3)
    flows["ds3_f3"] = g["ds3"].shift(-3)
    lab = keys.loc[labeled, ["company_id", "period", Y_COL]].merge(
        flows[["company_id", "period", "in3", "ds3", "in3_f3", "ds3_f3"]],
        on=["company_id", "period"],
        how="left",
    )
    lab["in_ratio"] = lab["in3_f3"] / lab["in3"].replace(0, np.nan)
    lab["ds_ratio"] = lab["ds3_f3"] / lab["ds3"].replace(0, np.nan)
    pos = lab[lab[Y_COL] == 1]
    neg = lab[lab[Y_COL] == 0]
    print(
        f"label mechanism (train): pos med in3[t+3]/in3[t]={float(pos['in_ratio'].median()):.3f} "
        f"ds3 ratio={float(pos['ds_ratio'].median()):.3f} "
        f"in_drop>20%={float((pos['in_ratio'] < 0.8).mean()):.3f} "
        f"ds_up>20%={float((pos['ds_ratio'] > 1.2).mean()):.3f} "
        f"| neg med in_ratio={float(neg['in_ratio'].median()):.3f}"
    )


def run(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    if args.list:
        for v in VARIANTS:
            print(f"{v['id']:22s} {v['model']:24s} {v['notes']}")
        return {"results": [], "slim": [], "started": ""}
    started = datetime.now().isoformat(timespec="minutes")
    print(f"xgb_y4 start {started} seed={FOLD_SEED} y={Y_COL}")
    print("forbidden X = family F; also drop a_fin_cost* a_debt* m_debt*")
    print("META.forbidden_x_families", Y4_META.get("forbidden_x_families"))
    demo = ["a_op_in", "a_debt_service", "a_fin_cost", "f_ds_r", "d_cust_hhi"]
    print(
        "leakage_check demo (a_debt_service / a_fin_cost bases are NOT flagged):",
        leakage_check(demo, Y_COL, FORBIDDEN),
    )
    leftover = [c for c in demo if not _is_debt_leak(c)]
    print("after this module's debt-leak drop leftover:", leftover)
    recheck_acceptance()

    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    store = _keys(store)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    if set(train_cos["company_id"].astype(str)) & hold_ids:
        raise RuntimeError("holdout companies in train_companies()")
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    grid = store[["company_id", "period"]].drop_duplicates()
    y4, y_source = load_y(con, grid)
    con.close()

    panel = store.merge(y4, on=["company_id", "period"], how="left")
    assert_no_holdout(panel.loc[~panel["company_id"].isin(hold_ids), "company_id"])
    print(
        f"panel={panel.shape} x_source={x_source} y_source={y_source} "
        f"train_cos={train_cos['company_id'].nunique()} hold_cos={len(hold_ids)}"
    )
    print_label_mechanism(store, y4)

    by_id = {v["id"]: v for v in VARIANTS}
    if args.all:
        wanted = [v["id"] for v in VARIANTS]
    elif args.variants:
        wanted = args.variants
    else:
        wanted = ["xgb_es"]
    for vid in wanted:
        if vid not in by_id:
            raise SystemExit(f"unknown variant {vid}; choose from {list(by_id)}")

    results = []
    for i, vid in enumerate(wanted):
        do_shap = bool(args.shap and i == len(wanted) - 1)
        spec = by_id[vid]
        if spec["kind"] == "zavg":
            results.append(
                run_zavg(
                    spec,
                    panel,
                    folds,
                    hold_ids,
                    write_registry=not args.no_registry,
                )
            )
        else:
            results.append(
                run_variant(
                    spec,
                    panel,
                    folds,
                    hold_ids,
                    write_registry=not args.no_registry,
                    do_shap=do_shap,
                )
            )

    slim = []
    for r in results:
        slim.append(
            {
                "variant": r["variant"],
                "cv": r["cv_auroc"],
                "sd": r["cv_auroc_sd"],
                "dummy": r["cv_dummy"],
                "single": r["cv_single"],
                "single_feat": r["best_single"],
                "gap_single": r["gap_single"],
                "n_x": r["n_x"],
                "trees": r["trees_median"],
                "collapsed": r["collapsed"],
                "hold_n_pos": r["n_hold_pos"],
                "hold_auroc": r["hold_auroc"],
                "decision": r["decision"],
                "reason": r["reason"],
            }
        )
    print("\nQUOTE (train group-fold CV; holdout LOW_POWER)")
    print(json.dumps(slim, indent=2, default=str))
    if args.md:
        write_output_md(results, started)
    return {"results": results, "slim": slim, "started": started}


if __name__ == "__main__":
    run()
