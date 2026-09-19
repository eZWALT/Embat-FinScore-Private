"""Global LightGBM panel classifier for y2_neg_2of3.

X: families A, C, E, F, G, H (D if importable). Never family B.
Lags 1 and 3 of numeric X. Company meta: group_size, n_banking (raw counts,
not holdout-fitted encodings of company_id).

5 group-fold CV on TRAIN groups only, then one fit on all train and one
holdout evaluation. Monotone constraints are skipped (signs are not sure).
"""
from __future__ import annotations

import csv
import importlib
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
from analysis.features.common import ANALYSIS, connect, train_mask
from analysis.features.grid import company_meta, monthly_grid
from analysis.targets.y2_stress import build as build_y2

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
Y_COL = "y2_neg_2of3"
AGENT = "6bfff878"
FORBIDDEN = ("b",)
LAGS = (1, 3)
META_OK = ("group_size", "n_banking")
N_FOLDS = 5

# Required + optional. Never B.
FAMILIES = (
    ("a", "analysis.features.cashflow", False),
    ("c", "analysis.features.ops", False),
    ("d", "analysis.features.counterparties", True),
    ("e", "analysis.features.invoices", False),
    ("f", "analysis.features.debt", False),
    ("g", "analysis.features.products", False),
    ("h", "analysis.features.groupctx", False),
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

# Fixed trees: fold-level early stopping collapsed to 1 tree (noisy val AUC
# with 89–374 positives). Same hyperparameters on every fold and on the
# final train fit — no holdout peek, no monotone constraints.
LGB_BASE = dict(
    objective="binary",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=80,
    subsample=0.8,
    colsample_bytree=0.7,
    reg_lambda=2.0,
    random_state=FOLD_SEED,
    n_jobs=1,
    verbosity=-1,
)
# Static within company — lagging them only duplicates the same column.
STATIC_NO_LAG = frozenset(META_OK) | {"h_group_size"}


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    return out


def load_allowed_x(con, grid: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Build allowed families. Skip optional/broken. Never keep b_ columns."""
    keys = _keys(grid[["company_id", "period"]])
    panel = keys.copy()
    loaded: list[str] = []
    skipped: list[str] = []
    for letter, modname, optional in FAMILIES:
        try:
            mod = importlib.import_module(modname)
            part = _keys(mod.build(con, keys.copy()))
        except Exception as exc:
            skipped.append(f"{letter}:{type(exc).__name__}:{exc}")
            print(f"skip family {letter}: {type(exc).__name__}: {exc}")
            if optional:
                continue
            raise
        extra = [c for c in part.columns if c not in {"company_id", "period"}]
        if any(str(c).startswith("b_") for c in extra):
            skipped.append(f"{letter}:contains_b_")
            print(f"skip family {letter}: produced b_ columns")
            continue
        clash = set(extra) & set(panel.columns)
        if clash:
            skipped.append(f"{letter}:clash:{sorted(clash)[:4]}")
            print(f"skip family {letter}: column clash {clash}")
            continue
        panel = panel.merge(part, on=["company_id", "period"], how="left")
        loaded.append(letter)
        print(f"loaded family {letter}: {len(extra)} cols")
    return panel, loaded, skipped


def numeric_x_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in KEY_COLS or c == Y_COL or str(c).startswith("y") or str(c).startswith("b_"):
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
        if c in STATIC_NO_LAG:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def _scale_pos_weight(y: pd.Series) -> float:
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 <= 0:
        return 1.0
    return n0 / n1


def _fit_lgb(Xtr, ytr, n_estimators: int | None = None) -> lgb.LGBMClassifier:
    params = dict(LGB_BASE)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    return clf


def _best_single_feature(X: pd.DataFrame, y: pd.Series, is_train: pd.Series, is_hold: pd.Series) -> dict:
    """Train-only sign; holdout metrics. Prefer baselines.py helpers if present."""
    try:
        from analysis.models.baselines import choose_sign, single_feature_auroc

        tbl = single_feature_auroc(X, y, is_train, is_hold)
        if tbl is None or tbl.empty:
            return {"feature": None, "holdout_auroc": float("nan"), "source": "baselines.empty"}
        # baselines sorts by holdout_auc; pick the train-best to avoid holdout peek for "the" baseline,
        # then report that feature's holdout numbers (sign already from train).
        pick = tbl.sort_values("train_auc", ascending=False).iloc[0]
        feat = str(pick["feature"])
        sign = int(pick["sign"])
        x = pd.to_numeric(X[feat], errors="coerce")
        ho = pd.DataFrame({"y": y[is_hold], "s": sign * x[is_hold]}).dropna()
        return {
            "feature": feat,
            "sign": sign,
            "train_auroc": float(pick["train_auc"]),
            "holdout_auroc": auroc(ho["y"], ho["s"]),
            "holdout_pr_auc": pr_auc(ho["y"], ho["s"]),
            "coverage": float(pick["coverage"]) if np.isfinite(pick["coverage"]) else float("nan"),
            "source": "baselines.single_feature_auroc",
        }
    except Exception as exc:
        print(f"baselines.py single-feature helper unavailable ({type(exc).__name__}: {exc}); computing locally")

    best = None
    y_tr, y_ho = y[is_train], y[is_hold]
    for col in numeric_x_cols(X):
        x = pd.to_numeric(X[col], errors="coerce")
        if x[is_train].nunique(dropna=True) < 2:
            continue
        auc_p = auroc(y_tr, x[is_train])
        auc_n = auroc(y_tr, -x[is_train])
        if not np.isfinite(auc_p) and not np.isfinite(auc_n):
            continue
        sign = -1 if (np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p)) else 1
        train_auc = auc_n if sign < 0 else auc_p
        if best is None or train_auc > best["train_auroc"]:
            ho = pd.DataFrame({"y": y_ho, "s": sign * x[is_hold]}).dropna()
            n_y = int(y_ho.notna().sum())
            best = {
                "feature": col,
                "sign": sign,
                "train_auroc": float(train_auc),
                "holdout_auroc": auroc(ho["y"], ho["s"]),
                "holdout_pr_auc": pr_auc(ho["y"], ho["s"]),
                "coverage": (len(ho) / n_y) if n_y else float("nan"),
                "source": "local_univariate",
            }
    return best or {"feature": None, "holdout_auroc": float("nan"), "source": "none"}


def _published_single_from_registry() -> dict:
    """Best y2 single-feature row already written by baselines.py (holdout-ranked).

    Registry is appended by several agents; some notes contain unquoted commas,
    so this uses csv.reader and the header width instead of pandas.
    """
    empty = {"feature": None, "holdout_auroc": float("nan")}
    if not REGISTRY.exists():
        return empty
    rows: list[tuple[float, dict]] = []
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return empty
        for raw in reader:
            if len(raw) < len(header):
                continue
            row = dict(zip(header, raw[: len(header)]))
            if (
                str(row.get("y", "")) != Y_COL
                or not str(row.get("model", "")).startswith("single_")
                or str(row.get("split", "")) != "holdout"
                or str(row.get("metric", "")) != "auroc"
                or str(row.get("agent", "")) == AGENT
            ):
                continue
            try:
                rows.append((float(row["value"]), row))
            except (TypeError, ValueError):
                continue
    if not rows:
        return empty
    rows.sort(key=lambda x: x[0], reverse=True)
    val, row = rows[0]
    feat = str(row["model"]).removeprefix("single_")
    train_auc = float("nan")
    notes = str(row.get("notes") or "")
    if "train_auc=" in notes:
        try:
            train_auc = float(notes.split("train_auc=", 1)[1].split(";")[0])
        except Exception:
            pass
    cov = float("nan")
    try:
        cov = float(row.get("coverage") or "nan")
    except ValueError:
        pass
    return {
        "feature": feat,
        "holdout_auroc": val,
        "train_auc": train_auc,
        "coverage": cov,
        "notes": notes,
        "source": "registry.baselines",
    }


def dummy_prior(y_train: pd.Series, y_eval: pd.Series) -> dict:
    """Constant train prevalence. AUROC of a constant is 0.5; PR-AUC = eval base rate."""
    prior = float(y_train.mean()) if len(y_train) else float("nan")
    y = y_eval.dropna()
    n = int(len(y))
    rate = float(y.mean()) if n else float("nan")
    # Constant score: protocol AUROC is 0.5 when both classes exist.
    score = pd.Series(prior, index=y.index)
    return {
        "prior": prior,
        "auroc": auroc(y, score),
        "pr_auc": rate,  # theoretical AP of a constant classifier
        "pr_auc_ranked": pr_auc(y, score),
        "n": n,
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


def _gain_table(clf: lgb.LGBMClassifier, cols: list[str], n: int = 20) -> pd.DataFrame:
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


def run() -> dict:
    hold_ids = load_holdout()
    con = connect()
    grid = _keys(monthly_grid(con))
    meta = company_meta(con)
    meta["company_id"] = meta["company_id"].astype(str)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    print(
        f"grid={grid.shape} train_cos={train_cos['company_id'].nunique()} "
        f"hold_cos={len(hold_ids)} folds={folds['fold'].nunique()}"
    )

    print("building allowed X (never B)")
    X, loaded, skipped = load_allowed_x(con, grid)
    meta_keep = [c for c in META_OK if c in meta.columns]
    X = X.merge(meta[["company_id"] + meta_keep], on="company_id", how="left")
    for c in meta_keep:
        X[c] = pd.to_numeric(X[c], errors="coerce")

    print("building y2_neg_2of3")
    y2 = _keys(build_y2(con, grid))[["company_id", "period", Y_COL]]
    con.close()

    panel = X.merge(y2, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")

    base_cols = numeric_x_cols(panel)
    leak = leakage_check(base_cols, Y_COL, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage before lags: {leak['issues']}")

    print(f"adding lags {list(LAGS)} on {len(base_cols)} numeric X cols")
    panel = add_lags(panel, base_cols, LAGS)
    feat_cols = numeric_x_cols(panel)
    leak2 = leakage_check(feat_cols, Y_COL, FORBIDDEN)
    if not leak2["ok"]:
        raise RuntimeError(f"leakage after lags: {leak2['issues']}")
    if any(c.startswith("b_") for c in feat_cols) or Y_COL in feat_cols:
        raise RuntimeError("forbidden column leaked into X")

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[Y_COL].notna()
    print(
        f"X cols={len(feat_cols)} families={loaded} meta={meta_keep} "
        f"train_labeled={int((is_train & labeled).sum())} "
        f"hold_labeled={int((is_hold & labeled).sum())}"
    )

    # --- 5 group-fold CV on train only ---
    cv_rows = []
    for k in range(N_FOLDS):
        tr = is_train & labeled & (panel["fold"] != k)
        va = is_train & labeled & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"fold {k}: skip (one class empty)")
            cv_rows.append({"fold": k, "auroc": float("nan"), "pr_auc": float("nan"), "n_val": int(va.sum())})
            continue
        clf = _fit_lgb(panel.loc[tr, feat_cols], ytr)
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "pr_auc": pr_auc(yva, pred),
            "n_val": int(va.sum()),
            "n_pos": int((yva == 1).sum()),
            "n_trees": int(LGB_BASE["n_estimators"]),
        }
        cv_rows.append(row)
        print(
            f"fold {k}: auroc={row['auroc']:.4f} pr_auc={row['pr_auc']:.4f} "
            f"n={row['n_val']} pos={row['n_pos']} trees={row['n_trees']}"
        )

    cv = pd.DataFrame(cv_rows)
    cv_auroc = float(cv["auroc"].mean())
    cv_pr = float(cv["pr_auc"].mean())
    n_trees = int(LGB_BASE["n_estimators"])
    print(f"CV mean AUROC={cv_auroc:.4f} PR-AUC={cv_pr:.4f} n_trees={n_trees}")

    # --- fit all train, one holdout pass ---
    tr_all = is_train & labeled
    assert_no_holdout(panel.loc[tr_all, "company_id"])
    ytr_all = panel.loc[tr_all, Y_COL].astype(float)
    final = _fit_lgb(panel.loc[tr_all, feat_cols], ytr_all, n_estimators=n_trees)

    # Predict every holdout company-month (LightGBM handles NaN).
    X_ho = panel.loc[is_hold, feat_cols]
    ho_pred = final.predict_proba(X_ho)[:, 1]
    ho_finite = np.isfinite(ho_pred)
    n_hold_rows = int(is_hold.sum())
    n_hold_pred = int(ho_finite.sum())
    coverage = (n_hold_pred / n_hold_rows) if n_hold_rows else float("nan")

    ho_y = panel.loc[is_hold, Y_COL].to_numpy(dtype=float)
    ho_ok = np.isfinite(ho_y) & ho_finite
    hold_auroc = auroc(ho_y[ho_ok], ho_pred[ho_ok])
    hold_pr = pr_auc(ho_y[ho_ok], ho_pred[ho_ok])
    hold_rate = float(np.nanmean(ho_y[np.isfinite(ho_y)])) if np.isfinite(ho_y).any() else float("nan")
    print(
        f"HOLDOUT AUROC={hold_auroc:.4f} PR-AUC={hold_pr:.4f} "
        f"coverage={coverage:.4f} ({n_hold_pred}/{n_hold_rows}) "
        f"labeled={int(ho_ok.sum())} base_rate={hold_rate:.4f}"
    )

    dummy = dummy_prior(ytr_all, panel.loc[is_hold & labeled, Y_COL])
    print(
        f"dummy prior={dummy['prior']:.4f} holdout AUROC={dummy['auroc']:.4f} "
        f"PR-AUC={dummy['pr_auc']:.4f}"
    )

    # Best single feature: train-selected (honest) + published baselines.py row.
    X_now = panel[["company_id", "period"] + base_cols].copy()
    single = _best_single_feature(X_now, panel[Y_COL], is_train, is_hold)
    published = _published_single_from_registry()
    print(
        f"best single (train-selected) {single.get('feature')} sign={single.get('sign')} "
        f"train_auroc={single.get('train_auroc')} holdout_auroc={single.get('holdout_auroc')} "
        f"holdout_pr_auc={single.get('holdout_pr_auc')} src={single.get('source')}"
    )
    if published.get("feature"):
        print(
            f"baselines.py published best {published['feature']} "
            f"holdout_auroc={published['holdout_auroc']} "
            f"train_auc={published.get('train_auc')} (ranked on holdout)"
        )
    beats_single_train = (
        np.isfinite(hold_auroc)
        and np.isfinite(single.get("holdout_auroc", float("nan")))
        and hold_auroc > float(single["holdout_auroc"])
    )
    pub_auc = published.get("holdout_auroc", float("nan"))
    beats_single_published = np.isfinite(hold_auroc) and np.isfinite(pub_auc) and hold_auroc > float(pub_auc)
    beats_dummy = np.isfinite(hold_auroc) and hold_auroc > (dummy["auroc"] if np.isfinite(dummy["auroc"]) else 0.5)
    # Headline comparison uses the published baselines.py number when present.
    beats_single = beats_single_published if published.get("feature") else beats_single_train
    print(
        f"GBM beats dummy={beats_dummy} beats_single_trainpick={beats_single_train} "
        f"beats_single_published={beats_single_published}"
    )

    imp = _gain_table(final, feat_cols, 20)
    print("top 20 gain importances")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    top8 = imp.head(8)

    train_cov = float((is_train & labeled).mean()) if is_train.any() else float("nan")
    fam = "+".join(s.upper() for s in loaded) if loaded else "-"
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    notes = (
        f"trees={n_trees}; n_x={len(feat_cols)}; lags=1,3; meta={','.join(meta_keep)}; "
        f"no_monotone; skipped={skipped or '-'}; "
        f"vs_dummy={hold_auroc:.3f}/{dummy['auroc']:.3f}; "
        f"vs_single_train={single.get('feature')} {single.get('holdout_auroc')}; "
        f"vs_single_pub={published.get('feature')} {published.get('holdout_auroc')}"
    )
    reg = [
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_panel",
            "split": "cv5_group",
            "metric": "auroc",
            "value": f"{cv_auroc:.6g}",
            "coverage": f"{train_cov:.4f}",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_panel",
            "split": "cv5_group",
            "metric": "pr_auc",
            "value": f"{cv_pr:.6g}",
            "coverage": f"{train_cov:.4f}",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_panel",
            "split": "holdout",
            "metric": "auroc",
            "value": f"{hold_auroc:.6g}",
            "coverage": f"{coverage:.4f}",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_panel",
            "split": "holdout",
            "metric": "pr_auc",
            "value": f"{hold_pr:.6g}",
            "coverage": f"{coverage:.4f}",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": "-",
            "y": Y_COL,
            "model": "dummy_prior",
            "split": "holdout",
            "metric": "auroc",
            "value": f"{dummy['auroc']:.6g}",
            "coverage": f"{coverage:.4f}",
            "notes": f"constant train prevalence {dummy['prior']:.4f}",
        },
    ]
    if single.get("feature"):
        reg.append(
            {
                "ts": ts,
                "round": "R3",
                "wave": 3,
                "agent": AGENT,
                "x_families": str(single["feature"]).split("_", 1)[0].upper(),
                "y": Y_COL,
                "model": f"single_{single['feature']}",
                "split": "holdout",
                "metric": "auroc",
                "value": f"{single['holdout_auroc']:.6g}",
                "coverage": f"{single['coverage']:.4f}" if np.isfinite(single.get("coverage", float("nan"))) else "",
                "notes": f"sign={single.get('sign')}; train_auc={single.get('train_auroc'):.4f}; src={single.get('source')}",
            }
        )
    append_registry(reg)
    print(f"appended {len(reg)} registry rows")

    summary = {
        "cv_auroc": cv_auroc,
        "cv_pr_auc": cv_pr,
        "cv_folds": cv_rows,
        "n_trees": n_trees,
        "holdout_auroc": hold_auroc,
        "holdout_pr_auc": hold_pr,
        "coverage": coverage,
        "n_hold_rows": n_hold_rows,
        "n_hold_pred": n_hold_pred,
        "n_hold_labeled": int(ho_ok.sum()),
        "holdout_base_rate": hold_rate,
        "dummy": dummy,
        "single": single,
        "published_single": published,
        "beats_dummy": beats_dummy,
        "beats_single": beats_single,
        "beats_single_trainpick": beats_single_train,
        "beats_single_published": beats_single_published,
        "families": loaded,
        "skipped": skipped,
        "n_x": len(feat_cols),
        "n_x_base": len(base_cols),
        "meta": meta_keep,
        "top8": top8.to_dict(orient="records"),
        "top20": imp.to_dict(orient="records"),
        "train_labeled": int(tr_all.sum()),
        "train_base_rate": float(ytr_all.mean()),
    }
    print("SUMMARY")
    print(
        json.dumps(
            {
                "cv_auroc": cv_auroc,
                "cv_pr_auc": cv_pr,
                "holdout_auroc": hold_auroc,
                "holdout_pr_auc": hold_pr,
                "coverage": coverage,
                "top8": summary["top8"],
                "beats_single": beats_single,
                "beats_single_trainpick": beats_single_train,
                "beats_single_published": beats_single_published,
                "single": {k: single.get(k) for k in ("feature", "holdout_auroc", "holdout_pr_auc", "source")},
                "published_single": {
                    k: published.get(k) for k in ("feature", "holdout_auroc", "train_auc", "source")
                },
            },
            indent=2,
            default=str,
        )
    )
    return summary


if __name__ == "__main__":
    run()
