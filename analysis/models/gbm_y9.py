"""LightGBM for accepted Y9 fee/interest-pressure labels.

Y9 is brief questions 3 (turning) and 5 (why): sustained fee+interest /
inflow pressure. Not a bankruptcy label. Not a 0–100 score.

- y9_fee_r_ownp80 (primary): own-history p80 for the next quarter
- y9_fee_spike: fee_r doubles vs trailing-6m mean in 2 of next 3 months

X families A,B,C,D,E,G,H — never F. Also drop every column that starts
with a_fin_cost or a_fc (including lags). protocol.leakage_check appends
"_" to prefixes, so it will not catch the exact name a_fin_cost; this
module drops those columns itself.

X from data/feature_store/monthly.parquet when present.
Y from data/feature_store/targets.parquet when both Y9 columns exist,
else analysis.targets.y9_fees.build (parent may be rewriting targets).

5 group-fold CV on TRAIN groups, then one train fit and one holdout pass.
Compared to a dummy prior and the best single allowed feature (sign from
train; single also scored OOF on the same folds). Holdout is LOW_POWER
unless n_pos is large — quote CV AUROC for KEEP/PARK.
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
from analysis.targets.y9_fees import Y9_COLS, build as build_y9

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
Y_STORE = DATA / "feature_store" / "targets.parquet"
AGENT = "634720f6"
LAGS = (1, 3)
META_OK = ("group_size", "n_banking")
N_FOLDS = 5
CLEAR_MARGIN = 0.02
LOW_POWER_POS = 30
# leakage_check("a_fin_cost") becomes "a_fin_cost_" and misses the base col.
FEE_LEAK_PREFIXES = ("a_fin_cost", "a_fc")

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

TASKS = (
    {
        "y": "y9_fee_r_ownp80",
        "primary": True,
        "allowed": ("a", "b", "c", "d", "e", "g", "h"),
        "forbidden": ("f",),
        "notes": (
            "primary Y9 own-p80; questions 3+5 turning/why; "
            "never F; drop a_fin_cost* and a_fc*"
        ),
    },
    {
        "y": "y9_fee_spike",
        "primary": False,
        "allowed": ("a", "b", "c", "d", "e", "g", "h"),
        "forbidden": ("f",),
        "notes": (
            "Y9 NSF/fee spike (2 of next 3); questions 3+5; "
            "never F; drop a_fin_cost* and a_fc*"
        ),
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


def _is_fee_leak(col: str) -> bool:
    """True for a_fin_cost / a_fc and their lags. Not caught by leakage_check."""
    s = str(col)
    return s.startswith(FEE_LEAK_PREFIXES)


def allowed_x_cols(df: pd.DataFrame, allowed: tuple[str, ...], y_col: str) -> list[str]:
    """Numeric columns in allowed families, plus META_OK. Never y_* / F / fee leak."""
    allow = {a.lower() for a in allowed}
    cols: list[str] = []
    dropped_fee: list[str] = []
    for c in df.columns:
        if c in KEY_COLS or c == y_col or str(c).startswith("y"):
            continue
        if _is_fee_leak(c):
            dropped_fee.append(c)
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
    if dropped_fee:
        print(f"dropped fee-leak X ({len(dropped_fee)}): {dropped_fee[:12]}")
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


def load_y(con, grid: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Y9 from targets.parquet when both columns exist; else y9_fees.build."""
    need = list(Y9_COLS)
    if Y_STORE.exists():
        try:
            yall = _keys(pd.read_parquet(Y_STORE))
            have = [c for c in need if c in yall.columns]
            if len(have) == len(need):
                print(f"loaded Y9 from {Y_STORE} cols={have}")
                return yall[["company_id", "period", *need]].copy(), "targets_parquet"
            print(f"{Y_STORE} missing Y9 cols {sorted(set(need) - set(have))}; rebuilding")
        except Exception as exc:
            print(f"{Y_STORE} unreadable ({type(exc).__name__}: {exc}); rebuilding Y9")
    print("building Y9 via analysis.targets.y9_fees.build")
    return _keys(build_y9(con, grid)), "y9_fees.build"


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


def _signed_auroc(y: pd.Series, x: pd.Series) -> tuple[int, float]:
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1, float("nan")
    if np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p):
        return -1, float(auc_n)
    return 1, float(auc_p)


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
        if _is_fee_leak(col) or _family_of(col) == "f":
            continue
        x = pd.to_numeric(X[col], errors="coerce")
        tr = pd.DataFrame({"y": y_tr, "s": x[is_train]}).dropna()
        if len(tr) < min_n or tr["y"].nunique() < 2 or tr["s"].nunique() < 2:
            continue
        sign, train_auc = _signed_auroc(tr["y"], tr["s"])
        if not np.isfinite(train_auc):
            continue
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
        return {"feature": None, "holdout_auroc": float("nan"), "cv_auroc": float("nan"), "top5": []}
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


def single_cv_auroc(
    panel: pd.DataFrame,
    y_col: str,
    feature: str,
    is_train: pd.Series,
    labeled: pd.Series,
) -> float:
    """OOF AUROC of one feature on the same train group folds (sign from train fold)."""
    if not feature or feature not in panel.columns:
        return float("nan")
    x = pd.to_numeric(panel[feature], errors="coerce")
    aucs: list[float] = []
    for k in range(N_FOLDS):
        tr = is_train & labeled & (panel["fold"] != k)
        va = is_train & labeled & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        trd = pd.DataFrame({"y": panel.loc[tr, y_col], "s": x[tr]}).dropna()
        vad = pd.DataFrame({"y": panel.loc[va, y_col], "s": x[va]}).dropna()
        if len(trd) < 2 or trd["y"].nunique() < 2 or vad["y"].nunique() < 2 or trd["s"].nunique() < 2:
            continue
        sign, _ = _signed_auroc(trd["y"], trd["s"])
        aucs.append(auroc(vad["y"], sign * vad["s"]))
    return float(np.nanmean(aucs)) if aucs else float("nan")


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
    # leakage_check("a_fin_cost") → "a_fin_cost_" and misses the base column.
    fee = [c for c in cols if _is_fee_leak(c)]
    if fee:
        raise RuntimeError(f"fee-cost leak in X for {y_col}: {fee[:8]}")
    fam_f = [c for c in cols if _family_of(c) == "f"]
    if fam_f:
        raise RuntimeError(f"family F leaked into X for {y_col}: {fam_f[:8]}")


def _demo_leakage_gap() -> None:
    """Show why a_fin_cost must be dropped outside leakage_check."""
    demo_cols = ["a_op_in", "a_fin_cost", "a_fin_cost_lag1", "a_fc_r", "f_fc_r", "b_runway"]
    leak = leakage_check(demo_cols, "y9_fee_r_ownp80", ["f", "a_fin_cost", "a_fc"])
    print("leakage_check demo (a_fin_cost itself is NOT flagged):", leak)
    leftover = [c for c in demo_cols if _family_of(c) != "f" and not str(c).startswith(("a_fc_", "a_fin_cost_"))]
    print("after leakage_check-style prefixes leftover includes a_fin_cost:", leftover)


def run_task(task: dict, store: pd.DataFrame, y_panel: pd.DataFrame, folds: pd.DataFrame, hold_ids: set[str]) -> dict:
    y_col = task["y"]
    allowed = task["allowed"]
    forbidden = task["forbidden"]
    print("\n" + "=" * 72)
    print(
        f"TASK {y_col} allowed={''.join(a.upper() for a in allowed)} "
        f"forbidden={forbidden} primary={task['primary']}"
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
    fee_in_x = [c for c in feat_cols if _is_fee_leak(c) or _family_of(c) == "f"]
    if fee_in_x:
        raise RuntimeError(f"post-lag fee/F leak: {fee_in_x[:8]}")

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
    cv_sd = float(cv["auroc"].std(ddof=1)) if cv["auroc"].notna().sum() > 1 else float("nan")
    n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
    n_trees = max(50, n_trees)
    print(f"CV mean AUROC={cv_auroc:.4f} sd={cv_sd:.4f} PR-AUC={cv_pr:.4f} n_trees_final={n_trees}")

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
    hold_power = "LOW_POWER" if n_hold_pos < LOW_POWER_POS else "USABLE"
    print(
        f"HOLDOUT AUROC={hold_auroc:.4f} PR-AUC={hold_pr:.4f} "
        f"coverage={coverage:.4f} labeled={n_hold_lab} pred={n_hold_pred} "
        f"n_pos={n_hold_pos} base_rate={hold_rate:.4f} power={hold_power}"
    )

    dummy = dummy_prior(ytr_all, panel.loc[ho_lab, y_col])
    dummy_cv = 0.5
    print(
        f"dummy prior={dummy['prior']:.4f} holdout AUROC={dummy['auroc']:.4f} "
        f"PR-AUC={dummy['pr_auc']:.4f} cv_auroc={dummy_cv:.4f}"
    )

    single = best_single_feature(panel, panel[y_col], tr_all, ho_lab, base_cols)
    single_cv = single_cv_auroc(panel, y_col, str(single.get("feature") or ""), is_train, labeled)
    single["cv_auroc"] = single_cv
    print(
        f"best single {single.get('feature')} sign={single.get('sign')} "
        f"picked={single.get('picked')} train_n={single.get('train_n')} "
        f"train_auroc={single.get('train_auroc')} cv_auroc={single_cv} "
        f"holdout_auroc={single.get('holdout_auroc')} "
        f"holdout_pr_auc={single.get('holdout_pr_auc')}"
    )
    for i, row in enumerate(single.get("top5") or [], 1):
        print(
            f"  single#{i} {row['feature']} sign={row['sign']} "
            f"train={row['train_auroc']:.4f} hold={row['holdout_auroc']}"
        )

    dummy_gap = cv_auroc - dummy_cv if np.isfinite(cv_auroc) else float("nan")
    single_gap = cv_auroc - single_cv if np.isfinite(cv_auroc) and np.isfinite(single_cv) else float("nan")
    beats_dummy_cv = bool(np.isfinite(cv_auroc) and dummy_gap > CLEAR_MARGIN)
    beats_single_cv = bool(np.isfinite(single_gap) and single_gap > CLEAR_MARGIN)
    verdict = "KEEP" if (beats_dummy_cv and beats_single_cv) else "PARK"
    print(
        f"CV vs dummy={cv_auroc:.4f}/{dummy_cv:.4f} gap={dummy_gap:.4f} "
        f"vs single_cv={single_cv:.4f} gap={single_gap:.4f} "
        f"clear_margin={CLEAR_MARGIN} verdict={verdict}"
    )

    imp = _gain_table(final, feat_cols, 12)
    print("top 12 gain importances")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    train_cov = float((is_train & labeled).mean()) if is_train.any() else float("nan")
    fam = "+".join(s.upper() for s in allowed)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    tag = "lightgbm_y9"
    star = "PRIMARY; " if task["primary"] else "secondary; "
    notes = (
        f"{verdict}; {star}{task['notes']}; trees={n_trees}; n_x={len(feat_cols)}; lags=1,3; "
        f"meta={','.join(c for c in META_OK if c in feat_cols)}; "
        f"cv={cv_auroc:.3f} dummy_cv=0.500 single_cv={single.get('feature')} {single_cv}; "
        f"cv_gap_dummy={dummy_gap:.3f} cv_gap_single={single_gap}; "
        f"clear_margin={CLEAR_MARGIN}; hold_n_pos={n_hold_pos} {hold_power}; "
        f"hold={hold_auroc:.3f}/{dummy['auroc']:.3f}; "
        f"vs_single_hold={single.get('feature')} {single.get('holdout_auroc')}"
    )
    reg = [
        {
            "ts": ts,
            "round": "R4",
            "wave": 4,
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
            "round": "R4",
            "wave": 4,
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
            "round": "R4",
            "wave": 4,
            "agent": AGENT,
            "x_families": "-",
            "y": y_col,
            "model": "dummy_prior",
            "split": "cv5_group",
            "metric": "auroc",
            "value": f"{dummy_cv:.6g}",
            "coverage": f"{train_cov:.4f}" if np.isfinite(train_cov) else "",
            "notes": f"{star}constant 0.5; train prevalence {dummy['prior']:.4f}",
        },
    ]
    if single.get("feature"):
        reg.append(
            {
                "ts": ts,
                "round": "R4",
                "wave": 4,
                "agent": AGENT,
                "x_families": _family_of(str(single["feature"])).upper(),
                "y": y_col,
                "model": f"single_{single['feature']}",
                "split": "cv5_group",
                "metric": "auroc",
                "value": f"{single_cv:.6g}" if np.isfinite(single_cv) else "",
                "coverage": f"{train_cov:.4f}" if np.isfinite(train_cov) else "",
                "notes": (
                    f"{star}sign={single.get('sign')}; train_auc={single.get('train_auroc'):.4f}; "
                    f"pick=train_best_allowed; WIN_COMPARE=cv5_group"
                ),
            }
        )
        reg.append(
            {
                "ts": ts,
                "round": "R4",
                "wave": 4,
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
                    f"pick=train_best_allowed; hold_n_pos={single.get('holdout_n_pos')} {hold_power}"
                ),
            }
        )
    append_registry(reg)
    print(f"appended {len(reg)} registry rows verdict={verdict}")

    return {
        "y": y_col,
        "primary": task["primary"],
        "allowed": list(allowed),
        "forbidden": list(forbidden),
        "verdict": verdict,
        "cv_auroc": cv_auroc,
        "cv_auroc_sd": cv_sd,
        "cv_pr_auc": cv_pr,
        "cv_folds": cv_rows,
        "n_trees": n_trees,
        "holdout_auroc": hold_auroc,
        "holdout_pr_auc": hold_pr,
        "holdout_power": hold_power,
        "coverage": coverage,
        "n_hold_labeled": n_hold_lab,
        "n_hold_pred": n_hold_pred,
        "n_hold_pos": n_hold_pos,
        "holdout_base_rate": hold_rate,
        "dummy": dummy,
        "dummy_cv_auroc": dummy_cv,
        "single": single,
        "single_cv_auroc": single_cv,
        "cv_gap_dummy": dummy_gap,
        "cv_gap_single": single_gap,
        "clear_margin": CLEAR_MARGIN,
        "beats_dummy_cv": beats_dummy_cv,
        "beats_single_cv": beats_single_cv,
        "n_x": len(feat_cols),
        "n_x_base": len(base_cols),
        "train_labeled": int(tr_all.sum()),
        "train_base_rate": float(ytr_all.mean()) if len(ytr_all) else float("nan"),
        "train_pos": int((ytr_all == 1).sum()) if len(ytr_all) else 0,
        "top12": imp.to_dict(orient="records"),
        "x_source": None,
    }


def run() -> dict:
    _demo_leakage_gap()
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

    y9, y_source = load_y(con, grid)
    con.close()
    print(f"y_source={y_source} y9_cols={[c for c in y9.columns if str(c).startswith('y9')]}")

    results = []
    for task in TASKS:
        out = run_task(task, store, y9, folds, hold_ids)
        out["x_source"] = x_source
        out["y_source"] = y_source
        results.append(out)

    summary = {r["y"]: r for r in results}
    print("\nSUMMARY")
    slim = {}
    for y, r in summary.items():
        slim[y] = {
            "primary": r["primary"],
            "verdict": r["verdict"],
            "cv_auroc": r["cv_auroc"],
            "cv_auroc_sd": r["cv_auroc_sd"],
            "dummy_cv_auroc": r["dummy_cv_auroc"],
            "single_feature": r["single"].get("feature"),
            "single_cv_auroc": r["single_cv_auroc"],
            "single_train_auroc": r["single"].get("train_auroc"),
            "cv_gap_dummy": r["cv_gap_dummy"],
            "cv_gap_single": r["cv_gap_single"],
            "clear_margin": r["clear_margin"],
            "beats_dummy_cv": r["beats_dummy_cv"],
            "beats_single_cv": r["beats_single_cv"],
            "holdout_auroc": r["holdout_auroc"],
            "holdout_power": r["holdout_power"],
            "n_hold_pos": r["n_hold_pos"],
            "n_hold_labeled": r["n_hold_labeled"],
            "train_labeled": r["train_labeled"],
            "train_base_rate": r["train_base_rate"],
            "train_pos": r["train_pos"],
            "n_x": r["n_x"],
        }
    print(json.dumps(slim, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()
