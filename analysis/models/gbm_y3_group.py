"""Per-group LightGBM for Y3 cash-recovery (brief 45→65 / who is improving).

Same spec as analysis.models.gbm_y3y6 for y3_recover_cash_6m:
stressed/labeled rows only; X families A,C,D,E,F,G,H — never B; lags 1,3.

Question: does one LightGBM per group_id beat the published global
group-fold CV AUROC 0.710? KEEP only if mixture CV AUROC > 0.720.
Holdout is reported but n_pos~14 is LOW_POWER — not used to keep/kill.
Holdout companies never enter a fit.

Why sibling holdout (not pure group-out) for the group models
-------------------------------------------------------------
`group_folds` puts every sibling on the same side of a split. A held-out
group therefore has zero in-group train companies, so a per-group model
cannot be scored under leave-group-out. Hidden-test groups are the same
(frozen holdout is 15 whole groups). For those rows the mixture *is* the
global engine.

The test that can actually use a group model: companies in a group with
>=5 train companies are scored leave-company-out on their siblings
(fallback to the global group-fold score when the sibling slice is too
thin or one-class). Small groups always use the global train-fold model.

Primary keep/kill number = mean of the five *group-fold* AUROCs of that
mixture (same fold aggregation as the published 0.710).
"""
from __future__ import annotations

import hashlib
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
    load_holdout,
    pr_auc,
    train_companies,
)
from analysis.features.common import ANALYSIS, connect, train_mask
from analysis.models.gbm_y3y6 import (
    LAGS,
    LGB_BASE,
    META_OK,
    N_FOLDS,
    _assert_allowed,
    _family_of,
    _fit_lgb,
    _keys,
    _scale_pos_weight,
    add_lags,
    allowed_x_cols,
    append_registry,
    dummy_prior,
    load_store,
)
from analysis.targets.y3_recovery import build as build_y3

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "823061e3"
Y_COL = "y3_recover_cash_6m"
ALLOWED = ("a", "c", "d", "e", "f", "g", "h")
FORBIDDEN = ("b",)
MIN_TRAIN_COMPANIES = 5
MIN_FIT_ROWS = 20
MIN_FIT_POS = 2
MIN_FIT_NEG = 2
MIN_FIT_COMPANIES = 2
PUBLISHED_GLOBAL_CV = 0.710
KEEP_THRESHOLD = 0.720

# Smaller trees than the global engine: a typical eligible group has
# tens-to-low-hundreds of stressed rows, not 5k.
LGB_GROUP = dict(
    objective="binary",
    n_estimators=150,
    learning_rate=0.05,
    num_leaves=8,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    random_state=FOLD_SEED,
    n_jobs=1,
    verbosity=-1,
)


def _stable_seed(group_id: str) -> int:
    digest = hashlib.md5(f"{FOLD_SEED}:{group_id}".encode(), usedforsecurity=False).hexdigest()
    return int(digest[:8], 16)


def _company_folds(company_ids: list[str], group_id: str, n: int = N_FOLDS) -> dict[str, int]:
    """Fold companies inside one group. n=min(5, n_cos) → LOO when n_cos<=5."""
    ids = sorted({str(c) for c in company_ids})
    rng = np.random.default_rng(_stable_seed(str(group_id)))
    rng.shuffle(ids)
    k = min(n, len(ids))
    return {cid: int(i % k) for i, cid in enumerate(ids)}


def _y_counts(y: pd.Series) -> tuple[int, int, int]:
    y = pd.to_numeric(y, errors="coerce")
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    return int(y.notna().sum()), n1, n0


def _fit_slice_ok(df: pd.DataFrame) -> bool:
    """Enough stressed rows, both classes, and at least two companies."""
    n, n1, n0 = _y_counts(df[Y_COL])
    n_cos = int(df["company_id"].nunique())
    return (
        n_cos >= MIN_FIT_COMPANIES
        and n >= MIN_FIT_ROWS
        and n1 >= MIN_FIT_POS
        and n0 >= MIN_FIT_NEG
    )


def _fit_lgb_group(Xtr, ytr, Xva=None, yva=None) -> lgb.LGBMClassifier:
    n = int(len(ytr))
    params = dict(LGB_GROUP)
    params["min_child_samples"] = max(5, min(int(params["min_child_samples"]), max(5, n // 8)))
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    clf = lgb.LGBMClassifier(**params)
    fit_kw: dict = {}
    if Xva is not None and yva is not None and pd.Series(yva).nunique() == 2:
        fit_kw["eval_set"] = [(Xva, yva)]
        fit_kw["eval_metric"] = "auc"
        fit_kw["callbacks"] = [
            lgb.early_stopping(30, verbose=False),
            lgb.log_evaluation(period=0),
        ]
    clf.fit(Xtr, ytr, **fit_kw)
    return clf


def _mean_fold_auroc(rows: list[dict]) -> float:
    vals = [r["auroc"] for r in rows if np.isfinite(r.get("auroc", float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def _fold_table(panel: pd.DataFrame, scores: pd.Series, is_train: pd.Series, labeled: pd.Series) -> list[dict]:
    rows = []
    for k in range(N_FOLDS):
        va = is_train & labeled & (panel["fold"] == k)
        assert_no_holdout(panel.loc[va, "company_id"])
        yva = panel.loc[va, Y_COL].astype(float)
        s = scores.loc[va]
        ok = yva.notna() & s.notna() & np.isfinite(s)
        n_pos = int((yva[ok] == 1).sum())
        auc = auroc(yva[ok], s[ok]) if ok.sum() and yva[ok].nunique() == 2 else float("nan")
        pra = pr_auc(yva[ok], s[ok]) if ok.sum() and yva[ok].nunique() == 2 else float("nan")
        row = {
            "fold": k,
            "auroc": auc,
            "pr_auc": pra,
            "n_val": int(ok.sum()),
            "n_pos": n_pos,
        }
        rows.append(row)
        print(
            f"  fold {k}: auroc={auc:.4f} pr_auc={pra:.4f} n={row['n_val']} pos={n_pos}"
            if np.isfinite(auc)
            else f"  fold {k}: auroc=nan n={row['n_val']} pos={n_pos}"
        )
    return rows


def _score_holdout(clf: lgb.LGBMClassifier, panel: pd.DataFrame, ho_lab: pd.Series, feat_cols: list[str]) -> dict:
    X_ho = panel.loc[ho_lab, feat_cols]
    if X_ho.empty:
        return {
            "auroc": float("nan"),
            "pr_auc": float("nan"),
            "n_labeled": 0,
            "n_pred": 0,
            "n_pos": 0,
            "base_rate": float("nan"),
            "coverage": float("nan"),
        }
    pred = clf.predict_proba(X_ho)[:, 1]
    y = panel.loc[ho_lab, Y_COL].to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    return {
        "auroc": auroc(y[ok], pred[ok]),
        "pr_auc": pr_auc(y[ok], pred[ok]),
        "n_labeled": int(ho_lab.sum()),
        "n_pred": int(ok.sum()),
        "n_pos": int((np.asarray(y) == 1).sum()),
        "base_rate": float(np.nanmean(y)) if len(y) and np.isfinite(y).any() else float("nan"),
        "coverage": float(ok.sum() / ho_lab.sum()) if ho_lab.any() else float("nan"),
    }


def run() -> dict:
    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    n_train_cos_by_g = (
        train_cos.groupby("group_id")["company_id"].nunique().astype(int).to_dict()
    )
    grid = store[["company_id", "period"]].drop_duplicates()
    print("building Y3 labels (existing module; not edited)")
    y3 = _keys(build_y3(con, grid))
    con.close()

    panel = store.merge(y3[["company_id", "period", Y_COL]], on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    if "group_id" not in panel.columns:
        raise RuntimeError("feature store missing group_id")
    panel["group_id"] = panel["group_id"].astype(str)
    panel["company_id"] = panel["company_id"].astype(str)

    base_cols = allowed_x_cols(panel, ALLOWED, Y_COL)
    _assert_allowed(base_cols, Y_COL, FORBIDDEN)
    print(
        f"base X cols={len(base_cols)} families="
        f"{sorted({_family_of(c) for c in base_cols if c not in META_OK})}"
    )
    lag_cols = [c for c in base_cols if c not in META_OK]
    print(f"adding lags {list(LAGS)} on {len(lag_cols)} family X cols (meta not lagged)")
    panel = add_lags(panel, lag_cols, LAGS)
    feat_cols = allowed_x_cols(panel, ALLOWED, Y_COL)
    _assert_allowed(feat_cols, Y_COL, FORBIDDEN)
    if any(str(c).startswith("b_") for c in feat_cols):
        raise RuntimeError("family B leaked into X")

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[Y_COL].notna()
    train_lab = is_train & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])

    n_train_cm = int(is_train.sum())
    n_lab = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, Y_COL] == 1).sum())
    rate = float(panel.loc[train_lab, Y_COL].mean()) if n_lab else float("nan")
    coverage = (n_lab / n_train_cm) if n_train_cm else float("nan")
    print(
        f"store={store.shape} source={x_source} X={len(feat_cols)} "
        f"train_cos={train_cos['company_id'].nunique()} hold_cos={len(hold_ids)} "
        f"train_labeled={n_lab} pos={n_pos} rate={rate:.4f} coverage={coverage:.4f}"
    )

    n_groups_train = int(train_cos["group_id"].nunique())
    n_large = sum(1 for n in n_train_cos_by_g.values() if n >= MIN_TRAIN_COMPANIES)
    print(
        f"train groups={n_groups_train} with>={MIN_TRAIN_COMPANIES} companies={n_large} "
        f"(eligibility is company count, not labeled count)"
    )

    # --- Global group-fold OOF (same protocol as gbm_y3y6) ---
    print("\n" + "=" * 72)
    print("GLOBAL group-fold CV (train groups only; published claim 0.710)")
    print("=" * 72)
    global_oof = pd.Series(np.nan, index=panel.index, dtype=float)
    global_models: dict[int, lgb.LGBMClassifier] = {}
    best_iters: list[int] = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"global fold {k}: skip (one class empty)")
            continue
        clf = _fit_lgb(panel.loc[tr, feat_cols], ytr, panel.loc[va, feat_cols], yva)
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        global_oof.loc[va] = pred
        global_models[k] = clf
        best_iters.append(
            int(getattr(clf, "best_iteration_", LGB_BASE["n_estimators"]) or LGB_BASE["n_estimators"])
        )
    global_folds = _fold_table(panel, global_oof, is_train, labeled)
    cv_global = _mean_fold_auroc(global_folds)
    print(f"GLOBAL CV mean AUROC={cv_global:.4f} (published {PUBLISHED_GLOBAL_CV:.3f})")

    # --- Per-group sibling OOF ---
    print("\n" + "=" * 72)
    print(
        f"PER-GROUP sibling company-fold (min {MIN_TRAIN_COMPANIES} train cos; "
        f"fit needs >={MIN_FIT_ROWS} rows, both classes, >={MIN_FIT_COMPANIES} cos)"
    )
    print("=" * 72)
    group_oof = pd.Series(np.nan, index=panel.index, dtype=float)
    used_group = pd.Series(False, index=panel.index)
    group_stats: list[dict] = []
    n_group_fits = 0
    n_group_skips = 0

    train_lab_df = panel.loc[train_lab]
    for gid, gdf in train_lab_df.groupby("group_id", sort=True):
        n_cos_all = int(n_train_cos_by_g.get(str(gid), 0))
        n, n1, n0 = _y_counts(gdf[Y_COL])
        rec = {
            "group_id": str(gid),
            "n_train_companies": n_cos_all,
            "n_labeled_companies": int(gdf["company_id"].nunique()),
            "n_rows": n,
            "n_pos": n1,
            "n_neg": n0,
            "eligible_size": n_cos_all >= MIN_TRAIN_COMPANIES,
            "used": False,
            "n_scored": 0,
            "reason": "",
        }
        if n_cos_all < MIN_TRAIN_COMPANIES:
            rec["reason"] = "small_group"
            n_group_skips += 1
            group_stats.append(rec)
            continue
        if n1 < MIN_FIT_POS or n0 < MIN_FIT_NEG or n < MIN_FIT_ROWS:
            rec["reason"] = "thin_or_one_class"
            n_group_skips += 1
            group_stats.append(rec)
            continue

        cids = sorted(gdf["company_id"].astype(str).unique())
        cf = _company_folds(cids, str(gid))
        n_scored = 0
        for k in sorted(set(cf.values())):
            tr_ids = {c for c, f in cf.items() if f != k}
            va_ids = {c for c, f in cf.items() if f == k}
            tr_m = gdf["company_id"].isin(tr_ids)
            va_m = gdf["company_id"].isin(va_ids)
            tr_slice = gdf.loc[tr_m]
            va_slice = gdf.loc[va_m]
            if va_slice.empty:
                continue
            if not _fit_slice_ok(tr_slice):
                continue
            assert_no_holdout(tr_slice["company_id"])
            ytr = tr_slice[Y_COL].astype(float)
            yva = va_slice[Y_COL].astype(float)
            clf = _fit_lgb_group(
                tr_slice[feat_cols],
                ytr,
                va_slice[feat_cols] if yva.nunique() == 2 else None,
                yva if yva.nunique() == 2 else None,
            )
            pred = clf.predict_proba(va_slice[feat_cols])[:, 1]
            group_oof.loc[va_slice.index] = pred
            used_group.loc[va_slice.index] = True
            n_scored += int(len(va_slice))
            n_group_fits += 1
        rec["n_scored"] = n_scored
        rec["used"] = n_scored > 0
        rec["reason"] = "ok" if n_scored > 0 else "fold_slices_thin"
        if n_scored == 0:
            n_group_skips += 1
        group_stats.append(rec)

    n_used_groups = sum(1 for r in group_stats if r["used"])
    n_group_rows = int(used_group.sum())
    print(
        f"group models used on {n_used_groups} groups / {n_group_rows} stressed rows "
        f"({n_group_rows / n_lab:.3f} of train labeled); sibling fits={n_group_fits}; "
        f"skipped_groups={n_group_skips}"
    )
    reasons = pd.Series([r["reason"] for r in group_stats]).value_counts().to_dict()
    print(f"group reasons={reasons}")

    mixture = global_oof.copy()
    mixture.loc[used_group] = group_oof.loc[used_group]

    print("\nMIXTURE group-fold AUROC (sibling OOF where used, else global OOF)")
    mix_folds = _fold_table(panel, mixture, is_train, labeled)
    cv_mix = _mean_fold_auroc(mix_folds)
    pooled_mix = auroc(panel.loc[train_lab, Y_COL], mixture.loc[train_lab])
    pooled_glob = auroc(panel.loc[train_lab, Y_COL], global_oof.loc[train_lab])

    large_rows = train_lab & used_group
    small_or_fb = train_lab & ~used_group
    auc_mix_used = (
        auroc(panel.loc[large_rows, Y_COL], mixture.loc[large_rows])
        if large_rows.any()
        else float("nan")
    )
    auc_glob_used = (
        auroc(panel.loc[large_rows, Y_COL], global_oof.loc[large_rows])
        if large_rows.any()
        else float("nan")
    )
    auc_glob_fb = (
        auroc(panel.loc[small_or_fb, Y_COL], global_oof.loc[small_or_fb])
        if small_or_fb.any()
        else float("nan")
    )
    print(
        f"MIXTURE CV mean AUROC={cv_mix:.4f} pooled={pooled_mix:.4f} | "
        f"GLOBAL CV mean={cv_global:.4f} pooled={pooled_glob:.4f}"
    )
    print(
        f"on rows actually scored by a group model: mix={auc_mix_used:.4f} "
        f"global_oof={auc_glob_used:.4f} n={int(large_rows.sum())} "
        f"pos={int((panel.loc[large_rows, Y_COL] == 1).sum())}"
    )
    print(
        f"on fallback rows (small/thin groups): global={auc_glob_fb:.4f} "
        f"n={int(small_or_fb.sum())} pos={int((panel.loc[small_or_fb, Y_COL] == 1).sum())}"
    )

    keep = bool(np.isfinite(cv_mix) and cv_mix > KEEP_THRESHOLD)
    decision = "KEEP" if keep else "PARK"
    print(
        f"\nDECISION {decision}: mixture CV AUROC {cv_mix:.4f} "
        f"{'>' if keep else '<='} {KEEP_THRESHOLD:.3f} "
        f"(vs same-run global {cv_global:.4f}, published {PUBLISHED_GLOBAL_CV:.3f})"
    )
    print(
        "Unseen groups (hidden test / this holdout) always fall back to global — "
        "per-group pooling cannot score a group that was never in train."
    )

    # --- Final train-only fits + holdout (LOW_POWER, not keep/kill) ---
    print("\n" + "=" * 72)
    print("HOLDOUT pass (LOW_POWER; all 15 holdout groups unseen → global only)")
    print("=" * 72)
    n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
    n_trees = max(50, n_trees)
    ytr_all = panel.loc[train_lab, Y_COL].astype(float)
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    final_global = _fit_lgb(panel.loc[train_lab, feat_cols], ytr_all, n_estimators=n_trees)

    # Fit per-group models on full train groups that qualify — none apply to holdout
    # (zero group overlap). Count them so the wave note can say how many would
    # exist for *seen* groups.
    n_final_group = 0
    for gid, gdf in train_lab_df.groupby("group_id", sort=True):
        n_cos_all = int(n_train_cos_by_g.get(str(gid), 0))
        if n_cos_all < MIN_TRAIN_COMPANIES or not _fit_slice_ok(gdf):
            continue
        assert_no_holdout(gdf["company_id"])
        _fit_lgb_group(gdf[feat_cols], gdf[Y_COL].astype(float))
        n_final_group += 1
    print(f"final train-only group models fit={n_final_group} (0 apply to holdout groups)")

    ho_lab = is_hold & labeled
    hold = _score_holdout(final_global, panel, ho_lab, feat_cols)
    dummy = dummy_prior(ytr_all, panel.loc[ho_lab, Y_COL])
    print(
        f"HOLDOUT AUROC={hold['auroc']:.4f} PR-AUC={hold['pr_auc']:.4f} "
        f"labeled={hold['n_labeled']} pred={hold['n_pred']} pos={hold['n_pos']} "
        f"base_rate={hold['base_rate']:.4f} coverage={hold['coverage']:.4f} "
        f"LOW_POWER={hold['n_pos'] < 30}"
    )
    print(f"dummy prior={dummy['prior']:.4f} holdout AUROC={dummy['auroc']:.4f}")

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    fam = "+".join(s.upper() for s in ALLOWED)
    notes = (
        f"{decision}; mixture_cv={cv_mix:.4f} vs global_cv={cv_global:.4f} "
        f"published={PUBLISHED_GLOBAL_CV:.3f} keep_if>{KEEP_THRESHOLD:.3f}; "
        f"sibling OOF on groups with>={MIN_TRAIN_COMPANIES} train cos "
        f"(fit >={MIN_FIT_ROWS} rows/both classes); else global group-fold; "
        f"used_groups={n_used_groups}/{n_groups_train} used_rows={n_group_rows}/{n_lab}; "
        f"group_row_mix={auc_mix_used:.4f} group_row_global={auc_glob_used:.4f}; "
        f"n_x={len(feat_cols)}; lags=1,3; trees_final={n_trees}; "
        f"holdout_pos={hold['n_pos']} LOW_POWER; never B; stressed-only"
    )
    reg = [
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_y3_group",
            "split": "cv5_group",
            "metric": "auroc",
            "value": f"{cv_mix:.6g}" if np.isfinite(cv_mix) else "",
            "coverage": f"{coverage:.4f}" if np.isfinite(coverage) else "",
            "notes": notes,
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_y3_global_equiv",
            "split": "cv5_group",
            "metric": "auroc",
            "value": f"{cv_global:.6g}" if np.isfinite(cv_global) else "",
            "coverage": f"{coverage:.4f}" if np.isfinite(coverage) else "",
            "notes": f"same-run global group-fold; published {PUBLISHED_GLOBAL_CV:.3f}; {notes}",
        },
        {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": Y_COL,
            "model": "lightgbm_y3_group",
            "split": "holdout",
            "metric": "auroc",
            "value": f"{hold['auroc']:.6g}" if np.isfinite(hold["auroc"]) else "",
            "coverage": f"{hold['coverage']:.4f}" if np.isfinite(hold["coverage"]) else "",
            "notes": f"LOW_POWER n_pos={hold['n_pos']}; unseen groups → global only; {notes}",
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
            "value": f"{dummy['auroc']:.6g}" if np.isfinite(dummy["auroc"]) else "",
            "coverage": f"{hold['coverage']:.4f}" if np.isfinite(hold["coverage"]) else "",
            "notes": f"constant train prevalence {dummy['prior']:.4f}; LOW_POWER",
        },
    ]
    append_registry(reg)

    out = {
        "y": Y_COL,
        "decision": decision,
        "keep": keep,
        "cv_auroc_mixture": cv_mix,
        "cv_auroc_global": cv_global,
        "published_global_cv": PUBLISHED_GLOBAL_CV,
        "keep_threshold": KEEP_THRESHOLD,
        "pooled_auroc_mixture": pooled_mix,
        "pooled_auroc_global": pooled_glob,
        "auc_mixture_on_group_rows": auc_mix_used,
        "auc_global_on_group_rows": auc_glob_used,
        "cv_folds_mixture": mix_folds,
        "cv_folds_global": global_folds,
        "n_x": len(feat_cols),
        "train_labeled": n_lab,
        "train_pos": n_pos,
        "train_base_rate": rate,
        "coverage": coverage,
        "n_train_groups": n_groups_train,
        "n_large_groups": n_large,
        "n_used_groups": n_used_groups,
        "n_group_rows": n_group_rows,
        "n_final_group_models": n_final_group,
        "group_reasons": reasons,
        "holdout": hold,
        "dummy": dummy,
        "n_trees_final": n_trees,
        "x_source": x_source,
    }
    print("\nSUMMARY")
    print(json.dumps({k: v for k, v in out.items() if k != "cv_folds_mixture" and k != "cv_folds_global"}, indent=2, default=str))
    return out


if __name__ == "__main__":
    run()
