"""LightGBM for accepted Y7 concentration and Y8 cross-source labels.

Per-column forbidden X (do NOT use Y8 META union a+b+e blindly):

- y7_top1_lost / y7_top1_lost_inflow: never D (invoice HHI / top-1 rebuilt in Y7)
- y8_inv_worse_6: never E (label is invoice-side; cash A/B is the intended X)
- y8_cash_worse_6: never A or B (label is cash-side; invoice E is the intended X)

Interesting bake-off: y7_top1_lost_inflow and both Y8.
y7_top1_lost is the looser sibling (no inflow clause) — still fit, lower priority.

X from data/feature_store/monthly.parquet when present (22230x118).
Y from data/feature_store/targets.parquet when the four columns exist,
else rebuilt from the frozen Y7/Y8 modules (not edited here).

5 group-fold CV on TRAIN groups, then one train fit and one holdout pass.
Compared to a dummy prior and the best single allowed feature (sign from train).
Holdout n_pos is reported as power.
"""
from __future__ import annotations

import csv
import json
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
    auroc,
    assert_no_holdout,
    group_folds,
    leakage_check,
    load_holdout,
    pr_auc,
    train_companies,
)
from analysis.features.common import ANALYSIS, DATA, connect, train_mask
from analysis.features.grid import monthly_grid
from analysis.targets.y7_concentration import Y7_COLS, build as build_y7
from analysis.targets.y8_cross import Y8_COLS, build as build_y8

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
Y_STORE = DATA / "feature_store" / "targets.parquet"
AGENT = "170793d7"
LAGS = (1, 3)
META_OK = ("group_size", "n_banking")
N_FOLDS = 5

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

# Same family as gbm_y3y6. Rare y7_top1_lost_inflow uses scale_pos_weight.
LGB_BASE = dict(
    objective="binary",
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=FOLD_SEED,
    n_jobs=2,
    verbosity=-1,
)

# Per-column allowed / forbidden. Union META ["a","b","e"] is NOT used.
TASKS = (
    {
        "y": "y7_top1_lost",
        "source": "y7",
        "allowed": ("a", "b", "c", "e", "f", "g", "h"),
        "forbidden": ("d",),
        "interesting": False,
        "notes": "looser Y7 (no inflow clause); never D",
    },
    {
        "y": "y7_top1_lost_inflow",
        "source": "y7",
        "allowed": ("a", "b", "c", "e", "f", "g", "h"),
        "forbidden": ("d",),
        "interesting": True,
        "notes": "Y7 shock + sustained op_in drop; never D; thin positives",
    },
    {
        "y": "y8_inv_worse_6",
        "source": "y8",
        "allowed": ("a", "b", "c", "d", "f", "g", "h"),
        "forbidden": ("e",),
        "interesting": True,
        "notes": "invoice-side Y; never E (not a+b+e union); intended cash A/B",
    },
    {
        "y": "y8_cash_worse_6",
        "source": "y8",
        "allowed": ("c", "d", "e", "f", "g", "h"),
        "forbidden": ("a", "b"),
        "interesting": True,
        "notes": "cash-side Y; never A or B (not a+b+e union); intended invoice E",
    },
)


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


def allowed_x_cols(df: pd.DataFrame, allowed: tuple[str, ...], y_col: str) -> list[str]:
    """Numeric columns in allowed families, plus META_OK. Never y_*."""
    allow = {a.lower() for a in allowed}
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
    extra = {}
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def load_store(con) -> tuple[pd.DataFrame, str]:
    """Prefer the assembled monthly parquet; rebuild families only if missing."""
    if STORE.exists():
        panel = _keys(pd.read_parquet(STORE))
        print(f"loaded store {STORE} shape={panel.shape}")
        return panel, "parquet"
    print(f"{STORE} missing; building families from modules")
    import importlib

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


def load_y(con, grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Y7/Y8 from targets.parquet when present; else frozen modules."""
    need = list(Y7_COLS) + list(Y8_COLS)
    if Y_STORE.exists():
        yall = _keys(pd.read_parquet(Y_STORE))
        have = [c for c in need if c in yall.columns]
        if len(have) == len(need):
            print(f"loaded Y from {Y_STORE} cols={have}")
            y7 = yall[["company_id", "period", *Y7_COLS]].copy()
            y8 = yall[["company_id", "period", *Y8_COLS]].copy()
            return y7, y8, "targets_parquet"
        print(f"{Y_STORE} missing Y cols {sorted(set(need) - set(have))}; rebuilding")
    print("building Y7 / Y8 from modules (not edited)")
    y7 = _keys(build_y7(con, grid))
    y8 = _keys(build_y8(con, grid))
    return y7, y8, "modules"


def _scale_pos_weight(y: pd.Series) -> float:
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 <= 0:
        return 1.0
    return n0 / n1


def _fit_lgb(Xtr, ytr, Xva=None, yva=None, n_estimators: int | None = None) -> lgb.LGBMClassifier:
    params = dict(LGB_BASE)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    clf = lgb.LGBMClassifier(**params)
    fit_kw: dict = {}
    if Xva is not None and yva is not None and n_estimators is None:
        fit_kw["eval_set"] = [(Xva, yva)]
        fit_kw["eval_metric"] = "auc"
        fit_kw["callbacks"] = [
            lgb.early_stopping(40, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    clf.fit(Xtr, ytr, **fit_kw)
    return clf


def dummy_prior(y_train: pd.Series, y_eval: pd.Series) -> dict:
    """Constant train prevalence. AUROC of a constant is 0.5; PR-AUC = eval base rate."""
    prior = float(y_train.mean()) if len(y_train) else float("nan")
    y = y_eval.dropna()
    n = int(len(y))
    rate = float(y.mean()) if n else float("nan")
    score = pd.Series(prior, index=y.index)
    return {
        "prior": prior,
        "auroc": auroc(y, score),
        "pr_auc": rate,
        "pr_auc_ranked": pr_auc(y, score),
        "n": n,
        "n_pos": int((y == 1).sum()) if n else 0,
    }


def best_single_feature(
    X: pd.DataFrame,
    y: pd.Series,
    is_train: pd.Series,
    is_hold: pd.Series,
    cols: list[str],
) -> dict:
    """Train-only sign and pick among features with real train coverage.

    Snapshot-thin columns can post a fake 0.8+ train AUROC on a handful of
    rows. Require >=25% of labeled train rows (min 200). Among those, take
    the train-best that is still evaluable on holdout.
    """
    y_tr, y_ho = y[is_train], y[is_hold]
    n_tr = int(y_tr.notna().sum())
    n_ho = int(y_ho.notna().sum())
    min_n = max(200, int(0.25 * n_tr)) if n_tr else 200
    ranked: list[dict] = []
    for col in cols:
        x = pd.to_numeric(X[col], errors="coerce")
        tr = pd.DataFrame({"y": y_tr, "s": x[is_train]}).dropna()
        if len(tr) < min_n or tr["y"].nunique() < 2 or tr["s"].nunique() < 2:
            continue
        auc_p = auroc(tr["y"], tr["s"])
        auc_n = auroc(tr["y"], -tr["s"])
        if not np.isfinite(auc_p) and not np.isfinite(auc_n):
            continue
        sign = -1 if (np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p)) else 1
        train_auc = auc_n if sign < 0 else auc_p
        ho = pd.DataFrame({"y": y_ho, "s": sign * x[is_hold]}).dropna()
        ho_auc = auroc(ho["y"], ho["s"]) if len(ho) and ho["y"].nunique() == 2 else float("nan")
        ranked.append(
            {
                "feature": col,
                "sign": sign,
                "train_auroc": float(train_auc),
                "holdout_auroc": ho_auc,
                "holdout_pr_auc": pr_auc(ho["y"], ho["s"]) if len(ho) else float("nan"),
                "train_n": int(len(tr)),
                "holdout_n": int(len(ho)),
                "holdout_n_pos": int((ho["y"] == 1).sum()) if len(ho) else 0,
                "coverage": (len(ho) / n_ho) if n_ho else float("nan"),
            }
        )
    if not ranked:
        return {"feature": None, "holdout_auroc": float("nan"), "top5": []}
    ranked.sort(key=lambda r: r["train_auroc"], reverse=True)
    pick = ranked[0]
    pick["picked"] = "train_best"
    for r in ranked:
        if np.isfinite(r["holdout_auroc"]):
            pick = r
            pick["picked"] = "train_best_holdout_defined"
            break
    pick["top5"] = [
        {k: x[k] for k in ("feature", "sign", "train_auroc", "holdout_auroc")}
        for x in ranked[:5]
    ]
    return pick


def append_registry(rows: list[dict]) -> None:
    if not rows or not REGISTRY.exists():
        return
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def _gain_table(clf: lgb.LGBMClassifier, cols: list[str], n: int = 12) -> pd.DataFrame:
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


def _assert_allowed(cols: list[str], y_col: str, forbidden: tuple[str, ...]) -> None:
    leak = leakage_check(cols, y_col, forbidden)
    if not leak["ok"]:
        raise RuntimeError(f"leakage for {y_col}: {leak['issues']}")
    bad_y = [c for c in cols if c == y_col or str(c).startswith("y")]
    if bad_y:
        raise RuntimeError(f"y columns in X for {y_col}: {bad_y[:8]}")
    prefs = tuple(f"{p}_" if not str(p).endswith("_") else p for p in forbidden)
    bad_f = [c for c in cols if str(c).startswith(prefs)]
    if bad_f:
        raise RuntimeError(f"forbidden family leaked into X for {y_col}: {bad_f[:8]}")
    # Combined Y8 META is a+b+e. Per-column must not apply the union.
    if y_col == "y8_inv_worse_6":
        if any(str(c).startswith("e_") for c in cols):
            raise RuntimeError(f"E leaked into X for {y_col}")
        # A/B must be allowed (cash is the intended side).
    if y_col == "y8_cash_worse_6":
        if any(str(c).startswith(("a_", "b_")) for c in cols):
            raise RuntimeError(f"A/B leaked into X for {y_col}")
    if y_col.startswith("y7_") and any(str(c).startswith("d_") for c in cols):
        raise RuntimeError(f"D leaked into X for {y_col}")


def run_task(task: dict, store: pd.DataFrame, y_panel: pd.DataFrame, folds: pd.DataFrame, hold_ids: set[str]) -> dict:
    y_col = task["y"]
    allowed = task["allowed"]
    forbidden = task["forbidden"]
    print("\n" + "=" * 72)
    print(
        f"TASK {y_col} allowed={''.join(a.upper() for a in allowed)} "
        f"forbidden={forbidden} interesting={task['interesting']}"
    )
    print("=" * 72)

    panel = store.merge(y_panel[["company_id", "period", y_col]], on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")

    base_cols = allowed_x_cols(panel, allowed, y_col)
    _assert_allowed(base_cols, y_col, forbidden)
    print(
        f"base X cols={len(base_cols)} "
        f"families={sorted({_family_of(c) for c in base_cols if c not in META_OK})}"
    )

    lag_cols = [c for c in base_cols if c not in META_OK]
    print(f"adding lags {list(LAGS)} on {len(lag_cols)} family X cols (meta not lagged)")
    panel = add_lags(panel, lag_cols, LAGS)
    feat_cols = allowed_x_cols(panel, allowed, y_col)
    _assert_allowed(feat_cols, y_col, forbidden)

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[y_col].notna()

    print(
        f"X cols={len(feat_cols)} train_labeled={int((is_train & labeled).sum())} "
        f"hold_labeled={int((is_hold & labeled).sum())}"
    )
    y_tr = panel.loc[is_train & labeled, y_col]
    y_tr_rate = float(y_tr.mean()) if len(y_tr) else float("nan")
    n_tr_pos = int((y_tr == 1).sum()) if len(y_tr) else 0
    print(f"train labeled base_rate={y_tr_rate:.4f} n_pos={n_tr_pos}")

    cv_rows = []
    best_iters = []
    for k in range(N_FOLDS):
        tr = is_train & labeled & (panel["fold"] != k)
        va = is_train & labeled & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, y_col].astype(float)
        yva = panel.loc[va, y_col].astype(float)
        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"fold {k}: skip (one class empty)")
            cv_rows.append(
                {
                    "fold": k,
                    "auroc": float("nan"),
                    "pr_auc": float("nan"),
                    "n_val": int(va.sum()),
                    "n_pos": int((yva == 1).sum()) if va.any() else 0,
                }
            )
            continue
        clf = _fit_lgb(panel.loc[tr, feat_cols], ytr, panel.loc[va, feat_cols], yva)
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "pr_auc": pr_auc(yva, pred),
            "n_val": int(va.sum()),
            "n_pos": int((yva == 1).sum()),
            "best_iteration": int(
                getattr(clf, "best_iteration_", LGB_BASE["n_estimators"]) or LGB_BASE["n_estimators"]
            ),
        }
        best_iters.append(row["best_iteration"])
        cv_rows.append(row)
        print(
            f"fold {k}: auroc={row['auroc']:.4f} pr_auc={row['pr_auc']:.4f} "
            f"n={row['n_val']} pos={row['n_pos']} trees={row['best_iteration']}"
        )

    cv = pd.DataFrame(cv_rows)
    cv_auroc = float(cv["auroc"].mean())
    cv_pr = float(cv["pr_auc"].mean())
    n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
    n_trees = max(50, n_trees)
    print(f"CV mean AUROC={cv_auroc:.4f} PR-AUC={cv_pr:.4f} n_trees_final={n_trees}")

    tr_all = is_train & labeled
    assert_no_holdout(panel.loc[tr_all, "company_id"])
    ytr_all = panel.loc[tr_all, y_col].astype(float)
    final = _fit_lgb(panel.loc[tr_all, feat_cols], ytr_all, n_estimators=n_trees)

    ho_lab = is_hold & labeled
    X_ho = panel.loc[ho_lab, feat_cols]
    if X_ho.empty:
        ho_pred = np.array([])
        ho_y = np.array([])
        hold_auroc = float("nan")
        hold_pr = float("nan")
        n_hold_pred = 0
    else:
        ho_pred = final.predict_proba(X_ho)[:, 1]
        ho_y = panel.loc[ho_lab, y_col].to_numpy(dtype=float)
        ho_ok = np.isfinite(ho_y) & np.isfinite(ho_pred)
        hold_auroc = auroc(ho_y[ho_ok], ho_pred[ho_ok])
        hold_pr = pr_auc(ho_y[ho_ok], ho_pred[ho_ok])
        n_hold_pred = int(ho_ok.sum())
    n_hold_lab = int(ho_lab.sum())
    coverage = (n_hold_pred / n_hold_lab) if n_hold_lab else float("nan")
    hold_rate = float(np.nanmean(ho_y)) if len(ho_y) and np.isfinite(ho_y).any() else float("nan")
    n_hold_pos = int((np.asarray(ho_y) == 1).sum()) if len(ho_y) else 0
    print(
        f"HOLDOUT AUROC={hold_auroc:.4f} PR-AUC={hold_pr:.4f} "
        f"coverage={coverage:.4f} labeled={n_hold_lab} pred={n_hold_pred} "
        f"n_pos={n_hold_pos} base_rate={hold_rate:.4f}  << power"
    )

    dummy = dummy_prior(ytr_all, panel.loc[ho_lab, y_col])
    print(
        f"dummy prior={dummy['prior']:.4f} holdout AUROC={dummy['auroc']:.4f} "
        f"PR-AUC={dummy['pr_auc']:.4f}"
    )

    single = best_single_feature(panel, panel[y_col], tr_all, ho_lab, base_cols)
    print(
        f"best single {single.get('feature')} sign={single.get('sign')} "
        f"picked={single.get('picked')} train_n={single.get('train_n')} "
        f"train_auroc={single.get('train_auroc')} holdout_auroc={single.get('holdout_auroc')} "
        f"holdout_pr_auc={single.get('holdout_pr_auc')}"
    )
    for i, row in enumerate(single.get("top5") or [], 1):
        print(
            f"  single#{i} {row['feature']} sign={row['sign']} "
            f"train={row['train_auroc']:.4f} hold={row['holdout_auroc']}"
        )
    beats_single = (
        np.isfinite(hold_auroc)
        and np.isfinite(single.get("holdout_auroc", float("nan")))
        and hold_auroc > float(single["holdout_auroc"])
    )
    dummy_auc = dummy["auroc"] if np.isfinite(dummy["auroc"]) else 0.5
    beats_dummy = np.isfinite(hold_auroc) and hold_auroc > dummy_auc
    print(f"GBM beats dummy={beats_dummy} beats_single={beats_single}")

    imp = _gain_table(final, feat_cols, 12)
    print("top 12 gain importances")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    train_cov = float((is_train & labeled).mean()) if is_train.any() else float("nan")
    fam = "+".join(s.upper() for s in allowed)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    tag = "lightgbm_y7y8"
    star = "INTERESTING; " if task["interesting"] else "secondary; "
    notes = (
        f"{star}{task['notes']}; trees={n_trees}; n_x={len(feat_cols)}; lags=1,3; "
        f"meta={','.join(c for c in META_OK if c in feat_cols)}; "
        f"hold_n_pos={n_hold_pos}; hold_n={n_hold_lab}; "
        f"vs_dummy={hold_auroc:.3f}/{dummy['auroc']:.3f}; "
        f"vs_single={single.get('feature')} {single.get('holdout_auroc')}; "
        f"beats_single={beats_single}"
    )
    reg = [
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": y_col,
            "model": tag,
            "split": "cv5_group",
            "metric": "auroc",
            "value": f"{cv_auroc:.6g}" if np.isfinite(cv_auroc) else "",
            "coverage": f"{train_cov:.4f}" if np.isfinite(train_cov) else "",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": y_col,
            "model": tag,
            "split": "holdout",
            "metric": "auroc",
            "value": f"{hold_auroc:.6g}" if np.isfinite(hold_auroc) else "",
            "coverage": f"{coverage:.4f}" if np.isfinite(coverage) else "",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": "-",
            "y": y_col,
            "model": "dummy_prior",
            "split": "holdout",
            "metric": "auroc",
            "value": f"{dummy['auroc']:.6g}" if np.isfinite(dummy["auroc"]) else "",
            "coverage": f"{coverage:.4f}" if np.isfinite(coverage) else "",
            "notes": f"{star}constant train prevalence {dummy['prior']:.4f}; hold_n_pos={n_hold_pos}",
        },
    ]
    if single.get("feature"):
        reg.append(
            {
                "ts": ts,
                "round": "R3",
                "wave": 3,
                "agent": AGENT,
                "x_families": _family_of(str(single["feature"])).upper(),
                "y": y_col,
                "model": f"single_{single['feature']}",
                "split": "holdout",
                "metric": "auroc",
                "value": f"{single['holdout_auroc']:.6g}" if np.isfinite(single["holdout_auroc"]) else "",
                "coverage": f"{single['coverage']:.4f}" if np.isfinite(single.get("coverage", float("nan"))) else "",
                "notes": (
                    f"{star}sign={single.get('sign')}; train_auc={single.get('train_auroc'):.4f}; "
                    f"pick=train_best_allowed; hold_n_pos={single.get('holdout_n_pos')}"
                ),
            }
        )
    append_registry(reg)
    print(f"appended {len(reg)} registry rows")

    return {
        "y": y_col,
        "interesting": task["interesting"],
        "allowed": list(allowed),
        "forbidden": list(forbidden),
        "cv_auroc": cv_auroc,
        "cv_pr_auc": cv_pr,
        "cv_folds": cv_rows,
        "n_trees": n_trees,
        "holdout_auroc": hold_auroc,
        "holdout_pr_auc": hold_pr,
        "coverage": coverage,
        "n_hold_labeled": n_hold_lab,
        "n_hold_pred": n_hold_pred,
        "n_hold_pos": n_hold_pos,
        "holdout_base_rate": hold_rate,
        "dummy": dummy,
        "single": single,
        "beats_dummy": beats_dummy,
        "beats_single": beats_single,
        "n_x": len(feat_cols),
        "n_x_base": len(base_cols),
        "train_labeled": int(tr_all.sum()),
        "train_base_rate": float(ytr_all.mean()) if len(ytr_all) else float("nan"),
        "train_pos": int((ytr_all == 1).sum()) if len(ytr_all) else 0,
        "top12": imp.to_dict(orient="records"),
        "x_source": None,
    }


def run() -> dict:
    hold_ids = load_holdout()
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

    y7, y8, y_source = load_y(con, grid)
    con.close()
    print(f"y_source={y_source}")

    results = []
    for task in TASKS:
        y_panel = y7 if task["source"] == "y7" else y8
        out = run_task(task, store, y_panel, folds, hold_ids)
        out["x_source"] = x_source
        out["y_source"] = y_source
        results.append(out)

    summary = {r["y"]: r for r in results}
    print("\nSUMMARY")
    slim = {}
    for y, r in summary.items():
        slim[y] = {
            "interesting": r["interesting"],
            "forbidden": r["forbidden"],
            "cv_auroc": r["cv_auroc"],
            "holdout_auroc": r["holdout_auroc"],
            "dummy_auroc": r["dummy"]["auroc"],
            "single_feature": r["single"].get("feature"),
            "single_holdout_auroc": r["single"].get("holdout_auroc"),
            "beats_dummy": r["beats_dummy"],
            "beats_single": r["beats_single"],
            "train_labeled": r["train_labeled"],
            "train_base_rate": r["train_base_rate"],
            "train_pos": r["train_pos"],
            "n_hold_labeled": r["n_hold_labeled"],
            "n_hold_pos": r["n_hold_pos"],
            "holdout_base_rate": r["holdout_base_rate"],
        }
    print(json.dumps(slim, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
