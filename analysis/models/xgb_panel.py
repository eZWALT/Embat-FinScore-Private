"""XGBoost panel for accepted Y5 payment labels.

Primary: y5_ap_od30_ownp80. Secondary (same X, cheap): y5_ar_od30_sust.
Allowed X: families A, B, C, F, G, H, and D if importable. Never family E.

Group-fold CV on train companies, then one holdout eval. Dummy (prior) and
best single allowed feature are the mandatory baselines. Holdout companies
never enter a fit, sign choice, or percentile.
"""
from __future__ import annotations

import csv
import importlib
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
from analysis.features.common import ANALYSIS, connect
from analysis.features.grid import monthly_grid
from analysis.targets.y5_payment import Y_COLS, build as build_y5

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "0c6820b4"
WAVE = 3
ROUND = "R3"
PRIMARY = "y5_ap_od30_ownp80"
SECONDARY = "y5_ar_od30_sust"
FORBIDDEN = ("e",)
N_FOLDS = 5
LAGS = (1, 3)
# Schedule / rate snapshot cols are ~1.7% of company-months (family F note).
# Univariate AUROC on those n is not a baseline.
MIN_SINGLE_COV = 0.25
MIN_SINGLE_N = 200
MIN_X_COV = 0.05  # drop snapshot-thin F cols from the model too (train-only)

# D last and optional. E is never listed.
ALLOWED_FAMILIES = (
    ("a", "analysis.features.cashflow"),
    ("b", "analysis.features.liquidity"),
    ("c", "analysis.features.ops"),
    ("f", "analysis.features.debt"),
    ("g", "analysis.features.products"),
    ("h", "analysis.features.groupctx"),
    ("d", "analysis.features.counterparties"),
)

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


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    return out


def load_allowed_x(con, grid: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build allowed Y5 families. Skip missing / broken modules. Never E."""
    keys = _keys(grid[["company_id", "period"]])
    panel = keys.copy()
    loaded: list[str] = []
    skipped: list[str] = []
    for letter, modname in ALLOWED_FAMILIES:
        if letter == "e":
            skipped.append("e:forbidden")
            continue
        try:
            mod = importlib.import_module(modname)
            part = mod.build(con, keys.copy())
        except Exception as exc:
            skipped.append(f"{letter}:{type(exc).__name__}:{exc}")
            print(f"skip family {letter}: {type(exc).__name__}: {exc}")
            continue
        extra = [c for c in part.columns if c not in {"company_id", "period"}]
        if any(c.startswith("e_") for c in extra):
            skipped.append(f"{letter}:contains_e_")
            print(f"skip family {letter}: produced e_ columns")
            continue
        bad_pref = [c for c in extra if not c.startswith(f"{letter}_")]
        if bad_pref:
            skipped.append(f"{letter}:bad_prefix:{bad_pref[:4]}")
            print(f"skip family {letter}: columns not prefixed {letter}_ {bad_pref[:4]}")
            continue
        part = _keys(part)
        clash = set(extra) & set(panel.columns)
        if clash:
            skipped.append(f"{letter}:clash:{sorted(clash)[:4]}")
            print(f"skip family {letter}: column clash {clash}")
            continue
        panel = panel.merge(part, on=["company_id", "period"], how="left")
        loaded.append(letter)
        print(f"loaded family {letter}: {len(extra)} cols")
    return panel, loaded, skipped


def numeric_x_cols(X: pd.DataFrame) -> list[str]:
    skip = {"company_id", "period", "month", "freq", "first_month", "first_week", "fold", "group_id"}
    cols = []
    for c in X.columns:
        if c in skip or str(c).startswith("y") or str(c).startswith("e_"):
            continue
        if pd.api.types.is_numeric_dtype(X[c]):
            cols.append(c)
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
    """Keep columns with enough non-null values on the train mask. No holdout."""
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
    # Dummy needs a dummy X of matching length.
    x_tr = np.zeros((len(y), 1))
    clf.fit(x_tr, y)
    return np.full(n, float(clf.class_prior_[1] if len(clf.class_prior_) > 1 else clf.class_prior_[0]))


def best_single_on_mask(
    X: pd.DataFrame,
    y: pd.Series,
    fit_mask: pd.Series,
    eval_mask: pd.Series,
    cols: list[str],
) -> dict:
    """Pick best signed feature on fit_mask; score eval_mask. Sign from fit only.

    Features must cover at least MIN_SINGLE_COV of labeled fit rows and
    MIN_SINGLE_N finite pairs. This blocks snapshot-thin F columns
    (f_w_rate, f_sched_vs_obs) from winning on n≈20.
    """
    y_fit = y[fit_mask]
    n_fit = int(fit_mask.sum())
    rows = []
    for col in cols:
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


def _n_trees(clf: XGBClassifier, default: int) -> int:
    for attr in ("best_iteration", "best_ntree_limit"):
        v = getattr(clf, attr, None)
        if v is not None and np.isfinite(v) and int(v) > 0:
            # best_iteration is 0-indexed in recent xgboost
            return int(v) + (1 if attr == "best_iteration" else 0)
    return int(default)


def fit_xgb(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_va: pd.DataFrame | None = None,
    y_va: pd.Series | None = None,
    n_estimators: int = 400,
    early_stopping_rounds: int | None = 40,
) -> XGBClassifier:
    params = dict(XGB_BASE)
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


def _labeled(df: pd.DataFrame, y_col: str) -> pd.Series:
    return df[y_col].notna()


def eval_one_y(
    panel: pd.DataFrame,
    y_col: str,
    x_cols: list[str],
    hold_ids: set[str],
    folds: pd.DataFrame,
) -> dict:
    df = panel.loc[_labeled(panel, y_col)].copy()
    df["y"] = df[y_col].astype(float)
    df = df.merge(folds[["company_id", "fold", "group_id"]], on="company_id", how="left")
    is_hold = df["company_id"].isin(hold_ids)
    is_train = ~is_hold
    train_df = df.loc[is_train]
    hold_df = df.loc[is_hold]
    assert_no_holdout(train_df)
    if train_df["fold"].isna().any():
        missing = train_df.loc[train_df["fold"].isna(), "company_id"].unique()[:6]
        raise RuntimeError(f"{y_col}: train rows without fold (sample {list(missing)})")

    x_cols = usable_x_cols(df, x_cols, is_train, MIN_X_COV)
    print(f"  {y_col}: usable X after train coverage>={MIN_X_COV}: {len(x_cols)}")
    leak = leakage_check(x_cols, y_col, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage {y_col}: {leak['issues']}")
    if not x_cols:
        raise RuntimeError(f"{y_col}: no usable X after coverage filter")

    # --- group-fold CV (train companies only) ---
    cv_xgb, cv_dummy, cv_single = [], [], []
    cv_single_feats = []
    best_iters = []
    for k in range(N_FOLDS):
        tr_m = is_train & (df["fold"] != k)
        va_m = is_train & (df["fold"] == k)
        if int(tr_m.sum()) < 50 or int(va_m.sum()) < 20:
            print(f"  {y_col} fold {k}: skip (n_tr={int(tr_m.sum())} n_va={int(va_m.sum())})")
            continue
        y_tr, y_va = df.loc[tr_m, "y"], df.loc[va_m, "y"]
        if y_tr.nunique() < 2 or y_va.nunique() < 2:
            print(f"  {y_col} fold {k}: skip (one class)")
            continue
        X_tr = df.loc[tr_m, x_cols]
        X_va = df.loc[va_m, x_cols]
        assert_no_holdout(df.loc[tr_m])

        d_sc = dummy_prior_scores(y_tr, int(va_m.sum()))
        cv_dummy.append(auroc(y_va, d_sc))

        sf = best_single_on_mask(df, df["y"], tr_m, va_m, x_cols)
        cv_single.append(sf["eval_auc"])
        cv_single_feats.append(sf["feature"])

        clf = fit_xgb(X_tr, y_tr, X_va, y_va)
        pred = clf.predict_proba(X_va)[:, 1]
        cv_xgb.append(auroc(y_va, pred))
        best_iters.append(_n_trees(clf, 200))
        print(
            f"  {y_col} fold {k}: xgb={cv_xgb[-1]:.4f} single={sf['eval_auc']:.4f} "
            f"feat={sf['feature']} dummy={cv_dummy[-1]:.4f} trees={best_iters[-1]} "
            f"n_va={int(va_m.sum())}"
        )

    n_est = int(np.median(best_iters)) if best_iters else 200
    n_est = max(60, min(n_est, 400))

    # --- one holdout eval: fit on all train ---
    y_tr_all = train_df["y"]
    X_tr_all = train_df[x_cols]
    X_ho = hold_df[x_cols]
    y_ho = hold_df["y"]
    assert_no_holdout(train_df)

    dummy_ho = dummy_prior_scores(y_tr_all, len(hold_df))
    dummy_ho_auc = auroc(y_ho, dummy_ho)
    dummy_ho_pr = pr_auc(y_ho, dummy_ho)

    sf_all = best_single_on_mask(df, df["y"], is_train, is_hold, x_cols)

    clf = fit_xgb(
        X_tr_all,
        y_tr_all,
        n_estimators=n_est,
        early_stopping_rounds=None,
    )
    pred_ho = clf.predict_proba(X_ho)[:, 1]
    ho_auc = auroc(y_ho, pred_ho)
    ho_pr = pr_auc(y_ho, pred_ho)

    imp = pd.Series(clf.feature_importances_, index=x_cols).sort_values(ascending=False)
    top = [(str(i), float(v)) for i, v in imp.head(12).items() if v > 0]

    n_ho_grid = int((panel["company_id"].isin(hold_ids)).sum())
    n_ho_lab = int(len(hold_df))
    n_ho_pred = int(pd.Series(pred_ho).notna().sum())
    cov_lab = n_ho_pred / n_ho_lab if n_ho_lab else float("nan")
    cov_grid = n_ho_pred / n_ho_grid if n_ho_grid else float("nan")

    n_tr_lab = int(len(train_df))
    n_tr_grid = int((~panel["company_id"].isin(hold_ids)).sum())

    out = {
        "y": y_col,
        "n_train_labeled": n_tr_lab,
        "n_hold_labeled": n_ho_lab,
        "train_coverage_grid": n_tr_lab / n_tr_grid if n_tr_grid else float("nan"),
        "hold_coverage_labeled": cov_lab,
        "hold_coverage_grid": cov_grid,
        "train_base_rate": float(y_tr_all.mean()) if n_tr_lab else float("nan"),
        "hold_base_rate": float(y_ho.mean()) if n_ho_lab else float("nan"),
        "cv_xgb_auroc": float(np.nanmean(cv_xgb)) if cv_xgb else float("nan"),
        "cv_xgb_auroc_std": float(np.nanstd(cv_xgb, ddof=1)) if len(cv_xgb) > 1 else float("nan"),
        "cv_xgb_folds": [float(v) for v in cv_xgb],
        "cv_dummy_auroc": float(np.nanmean(cv_dummy)) if cv_dummy else float("nan"),
        "cv_single_auroc": float(np.nanmean(cv_single)) if cv_single else float("nan"),
        "cv_single_features": cv_single_feats,
        "n_estimators": n_est,
        "hold_xgb_auroc": float(ho_auc),
        "hold_xgb_pr_auc": float(ho_pr),
        "hold_dummy_auroc": float(dummy_ho_auc),
        "hold_dummy_pr_auc": float(dummy_ho_pr),
        "hold_single_auroc": float(sf_all["eval_auc"]),
        "hold_single_pr_auc": float(sf_all["eval_pr"]),
        "best_single_feature": sf_all["feature"],
        "best_single_sign": int(sf_all["sign"]),
        "best_single_train_auc": float(sf_all["fit_auc"]),
        "xgb_beats_single_hold": bool(
            np.isfinite(ho_auc)
            and np.isfinite(sf_all["eval_auc"])
            and ho_auc > sf_all["eval_auc"]
        ),
        "xgb_beats_single_cv": bool(
            np.isfinite(np.nanmean(cv_xgb) if cv_xgb else np.nan)
            and np.isfinite(np.nanmean(cv_single) if cv_single else np.nan)
            and (np.nanmean(cv_xgb) > np.nanmean(cv_single))
        ),
        "top_importances": top,
        "n_x": len(x_cols),
    }
    print(
        f"{y_col}: CV xgb={out['cv_xgb_auroc']:.4f} single={out['cv_single_auroc']:.4f} "
        f"| hold xgb={out['hold_xgb_auroc']:.4f} single={out['hold_single_auroc']:.4f} "
        f"({out['best_single_feature']}) dummy={out['hold_dummy_auroc']:.4f} "
        f"| beats_single={out['xgb_beats_single_hold']} cov_lab={cov_lab:.3f}"
    )
    return out


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


def registry_rows(res: dict, loaded: list[str], ts: str) -> list[dict]:
    fam = "+".join(s.upper() for s in loaded) if loaded else "-"
    y = res["y"]
    sign = "+" if res["best_single_sign"] == 1 else "-"
    top = ",".join(f"{n}:{v:.3f}" for n, v in res["top_importances"][:6])
    notes_xgb = (
        f"trees={res['n_estimators']}; cv_std={res['cv_xgb_auroc_std']:.4f}; "
        f"beats_single_hold={res['xgb_beats_single_hold']}; top={top}"
    )
    notes_sf = (
        f"sign={sign}; train_auc={res['best_single_train_auc']:.4f}; "
        f"feat={res['best_single_feature']}; min_cov={MIN_SINGLE_COV}"
    )
    cov_h = res["hold_coverage_labeled"]
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": "xgb_panel",
            "split": "cv5_group",
            "metric": "auroc",
            "value": _fmt(res["cv_xgb_auroc"]),
            "coverage": f"{res['train_coverage_grid']:.4f}" if np.isfinite(res["train_coverage_grid"]) else "",
            "notes": notes_xgb,
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": "xgb_panel",
            "split": "holdout",
            "metric": "auroc",
            "value": _fmt(res["hold_xgb_auroc"]),
            "coverage": f"{cov_h:.4f}" if np.isfinite(cov_h) else "",
            "notes": notes_xgb,
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": "dummy_prior",
            "split": "holdout",
            "metric": "auroc",
            "value": _fmt(res["hold_dummy_auroc"]),
            "coverage": f"{cov_h:.4f}" if np.isfinite(cov_h) else "",
            "notes": "sklearn DummyClassifier strategy=prior",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": f"single_{res['best_single_feature']}",
            "split": "holdout",
            "metric": "auroc",
            "value": _fmt(res["hold_single_auroc"]),
            "coverage": f"{cov_h:.4f}" if np.isfinite(cov_h) else "",
            "notes": notes_sf,
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": fam,
            "y": y,
            "model": "xgb_panel",
            "split": "holdout",
            "metric": "pr_auc",
            "value": _fmt(res["hold_xgb_pr_auc"]),
            "coverage": f"{cov_h:.4f}" if np.isfinite(cov_h) else "",
            "notes": notes_xgb,
        },
    ]
    return rows


def run(also_secondary: bool = True) -> dict:
    hold_ids = load_holdout()
    con = connect()
    grid = _keys(monthly_grid(con)[["company_id", "period"]])
    cos = train_companies(con)
    folds = group_folds(cos, n=N_FOLDS, seed=FOLD_SEED)
    print(
        f"grid={grid.shape} train_cos={len(cos)} hold_cos={len(hold_ids)} "
        f"groups={folds['group_id'].nunique()} folds={N_FOLDS}"
    )

    print("allowed X families for Y5 (never E)")
    X, loaded, skipped = load_allowed_x(con, grid)
    base_cols = numeric_x_cols(X)
    print(f"adding lags {LAGS} on {len(base_cols)} contemporaneous cols")
    X = add_lags(X, base_cols, LAGS)
    x_cols = numeric_x_cols(X)
    leak = leakage_check(x_cols, PRIMARY, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage: {leak['issues']}")
    if not x_cols:
        raise RuntimeError("no numeric X columns after loading families")
    print(f"X cols={len(x_cols)} families={loaded} skipped={skipped} leak_ok={leak['ok']}")

    print("building Y5")
    y5 = _keys(build_y5(con, grid))
    con.close()

    panel = X.merge(y5[["company_id", "period", *Y_COLS]], on=["company_id", "period"], how="left")
    assert_no_holdout(panel.loc[~panel["company_id"].isin(hold_ids)])

    targets = [PRIMARY]
    if also_secondary:
        targets.append(SECONDARY)

    results = []
    for y_col in targets:
        print(f"=== {y_col} ===")
        results.append(eval_one_y(panel, y_col, x_cols, hold_ids, folds))

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    reg: list[dict] = []
    for res in results:
        reg.extend(registry_rows(res, loaded, ts))
    append_registry(reg)
    print(f"appended {len(reg)} registry rows")
    return {
        "loaded": loaded,
        "skipped": skipped,
        "n_x": len(x_cols),
        "x_cols": x_cols,
        "results": results,
        "n_reg": len(reg),
    }


if __name__ == "__main__":
    out = run()
    for res in out["results"]:
        print(res)
