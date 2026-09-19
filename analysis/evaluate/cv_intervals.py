"""Train group-fold CV AUROC intervals. No holdout AUROC.

holdout_power.py: every accepted binary has <30 holdout positives. Tonight
we quote 5 group-fold CV on TRAIN companies only (mean ± fold sd).

Cheap LightGBM (50 trees, max_depth=3) on monthly.parquet for:

- y3_recover_cash_6m — families A,C,D,E,F,G,H; never B; stressed/labeled rows
- y2_neg_2of3 — same X families; never B

X construction is the light reuse of analysis.models.gbm_y3y6 (allowed
families, lags 1/3, leakage assert). This module never fits on holdout
and never scores holdout.

    python -m analysis.evaluate.cv_intervals
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
    train_companies,
)
from analysis.features.common import ANALYSIS, DATA, connect, train_mask
from analysis.models.gbm_y3y6 import (
    LAGS,
    META_OK,
    STORE,
    _assert_allowed,
    _family_of,
    _keys,
    _scale_pos_weight,
    add_lags,
    allowed_x_cols,
)

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
TARGETS = DATA / "feature_store" / "targets.parquet"
AGENT = "67e8ef01"
N_FOLDS = 5
N_TREES = 50
MAX_DEPTH = 3

# Same allowed set as gbm_y3y6 Y3 / gbm_panel Y2. Never family B.
ALLOWED = ("a", "c", "d", "e", "f", "g", "h")
FORBIDDEN = ("b",)

LGB_SMALL = dict(
    objective="binary",
    n_estimators=N_TREES,
    max_depth=MAX_DEPTH,
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

TASKS = (
    {
        "y": "y3_recover_cash_6m",
        "notes": (
            "stressed-only (label NaN otherwise); X=A,C,D,E,F,G,H; never B; "
            "train group-fold CV only; no holdout AUROC"
        ),
    },
    {
        "y": "y2_neg_2of3",
        "notes": (
            "X=A,C,D,E,F,G,H; never B; train group-fold CV only; no holdout AUROC"
        ),
    },
)


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing — need monthly.parquet")
    panel = _keys(pd.read_parquet(STORE))
    print(f"loaded store {STORE} shape={panel.shape}")
    return panel


def load_y_panel() -> pd.DataFrame:
    """Prefer assembled targets.parquet; otherwise call the two Y builds."""
    need = [t["y"] for t in TASKS]
    if TARGETS.exists():
        raw = pd.read_parquet(TARGETS)
        have = set(raw.columns)
        if set(need) <= have and {"company_id", "period"} <= have:
            panel = _keys(raw[["company_id", "period", *need]])
            print(f"Y panel from {TARGETS} shape={panel.shape}")
            return panel
        missing = sorted(set(need) - have)
        print(f"{TARGETS} missing {missing}; falling back to build()")

    from analysis.features.grid import monthly_grid
    from analysis.targets.y2_stress import build as build_y2
    from analysis.targets.y3_recovery import build as build_y3

    con = connect()
    grid = _keys(monthly_grid(con)[["company_id", "period"]])
    y2 = _keys(build_y2(con, grid.copy()))
    y3 = _keys(build_y3(con, grid.copy()))
    con.close()
    panel = grid.merge(y2[["company_id", "period", "y2_neg_2of3"]], on=["company_id", "period"], how="left")
    panel = panel.merge(
        y3[["company_id", "period", "y3_recover_cash_6m"]],
        on=["company_id", "period"],
        how="left",
    )
    print(f"Y panel from build() shape={panel.shape}")
    return panel


def _fit_small(Xtr, ytr) -> lgb.LGBMClassifier:
    params = dict(LGB_SMALL)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    return clf


def run_task(task: dict, store: pd.DataFrame, y_panel: pd.DataFrame, folds: pd.DataFrame) -> dict:
    y_col = task["y"]
    print("\n" + "=" * 72)
    print(f"TASK {y_col} allowed={''.join(a.upper() for a in ALLOWED)} forbidden={FORBIDDEN}")
    print("split=cv5_group TRAIN only — holdout companies excluded, never scored")
    print("=" * 72)

    panel = store.merge(y_panel[["company_id", "period", y_col]], on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")

    base_cols = allowed_x_cols(panel, ALLOWED, y_col)
    _assert_allowed(base_cols, y_col, FORBIDDEN)
    print(
        f"base X cols={len(base_cols)} families="
        f"{sorted({_family_of(c) for c in base_cols if c not in META_OK})}"
    )

    lag_cols = [c for c in base_cols if c not in META_OK]
    print(f"adding lags {list(LAGS)} on {len(lag_cols)} family X cols (meta not lagged)")
    panel = add_lags(panel, lag_cols, LAGS)
    feat_cols = allowed_x_cols(panel, ALLOWED, y_col)
    _assert_allowed(feat_cols, y_col, FORBIDDEN)
    if any(str(c).startswith("b_") for c in feat_cols):
        raise RuntimeError(f"family B leaked into X for {y_col}")

    is_train = train_mask(panel["company_id"])
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[y_col].notna()
    train_lab = is_train & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])

    n_train_cm = int(is_train.sum())
    n_lab = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, y_col] == 1).sum())
    rate = float(panel.loc[train_lab, y_col].mean()) if n_lab else float("nan")
    coverage = (n_lab / n_train_cm) if n_train_cm else float("nan")
    print(
        f"X cols={len(feat_cols)} train_cm={n_train_cm} train_labeled={n_lab} "
        f"train_pos={n_pos} base_rate={rate:.4f} coverage={coverage:.4f}"
    )

    cv_rows = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
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
                    "n_val": int(va.sum()),
                    "n_pos": int((yva == 1).sum()) if len(yva) else 0,
                    "n_train": int(tr.sum()),
                }
            )
            continue
        clf = _fit_small(panel.loc[tr, feat_cols], ytr)
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "n_val": int(va.sum()),
            "n_pos": int((yva == 1).sum()),
            "n_train": int(tr.sum()),
        }
        cv_rows.append(row)
        print(
            f"fold {k}: auroc={row['auroc']:.4f} n={row['n_val']} "
            f"pos={row['n_pos']} train_n={row['n_train']}"
        )

    aucs = np.asarray([r["auroc"] for r in cv_rows], dtype=float)
    ok = aucs[np.isfinite(aucs)]
    mean_auc = float(ok.mean()) if ok.size else float("nan")
    # Sample sd across folds (the number we quote with the mean).
    sd_auc = float(ok.std(ddof=1)) if ok.size >= 2 else float("nan")
    print(
        f"CV mean AUROC={mean_auc:.4f} fold_sd={sd_auc:.4f} "
        f"n_folds_ok={int(ok.size)} trees={N_TREES} max_depth={MAX_DEPTH}"
    )
    print("NO holdout AUROC computed (all accepted binaries LOW_POWER)")

    return {
        "y": y_col,
        "allowed": list(ALLOWED),
        "forbidden": list(FORBIDDEN),
        "cv_auroc_mean": mean_auc,
        "cv_auroc_sd": sd_auc,
        "cv_folds": cv_rows,
        "n_folds_ok": int(ok.size),
        "n_x": len(feat_cols),
        "n_x_base": len(base_cols),
        "train_cm": n_train_cm,
        "train_labeled": n_lab,
        "train_pos": n_pos,
        "train_base_rate": rate,
        "coverage": coverage,
        "notes": task["notes"],
    }


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
    print(f"appended {len(rows)} registry rows")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return "" if v is None else str(v)


def registry_rows(ts: str, results: list[dict]) -> list[dict]:
    fam = "+".join(s.upper() for s in ALLOWED)
    out = []
    for r in results:
        folds = ",".join(
            f"{x['fold']}:{x['auroc']:.4f}" if np.isfinite(x["auroc"]) else f"{x['fold']}:nan"
            for x in r["cv_folds"]
        )
        notes = (
            f"{r['notes']}; trees={N_TREES}; max_depth={MAX_DEPTH}; n_x={r['n_x']}; "
            f"lags=1,3; meta={','.join(META_OK)}; "
            f"mean={r['cv_auroc_mean']:.4f}; sd={r['cv_auroc_sd']:.4f}; "
            f"folds={folds}; train={r['train_labeled']}/{r['train_pos']}/"
            f"{r['train_base_rate']:.4f}; quote=cv5_group_not_holdout"
        )
        cov = r["coverage"]
        cov_s = f"{cov:.4f}" if np.isfinite(cov) else ""
        base = {
            "ts": ts,
            "round": "R3",
            "wave": 3,
            "agent": AGENT,
            "x_families": fam,
            "y": r["y"],
            "model": "lgbm_small_d3_n50",
            "coverage": cov_s,
            "notes": notes,
        }
        out.append(
            {
                **base,
                "split": "cv5_group",
                "metric": "auroc",
                "value": _fmt(r["cv_auroc_mean"]),
            }
        )
        out.append(
            {
                **base,
                "split": "cv5_group",
                "metric": "auroc_sd",
                "value": _fmt(r["cv_auroc_sd"]),
            }
        )
    return out


def run() -> dict:
    con = connect()
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    con.close()

    store = load_store()
    y_panel = load_y_panel()
    print(
        f"store={store.shape} train_cos={train_cos['company_id'].nunique()} "
        f"folds={folds['fold'].nunique()} lgb=trees{N_TREES}/depth{MAX_DEPTH}"
    )

    results = []
    for task in TASKS:
        results.append(run_task(task, store, y_panel, folds))

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    append_registry(registry_rows(ts, results))

    slim = {
        r["y"]: {
            "cv_auroc_mean": r["cv_auroc_mean"],
            "cv_auroc_sd": r["cv_auroc_sd"],
            "n_folds_ok": r["n_folds_ok"],
            "train_labeled": r["train_labeled"],
            "train_pos": r["train_pos"],
            "train_base_rate": r["train_base_rate"],
            "n_x": r["n_x"],
            "quote": f"{r['cv_auroc_mean']:.3f} ± {r['cv_auroc_sd']:.3f}",
        }
        for r in results
    }
    print("\nQUOTE (train group-fold CV only; not holdout)")
    print(json.dumps(slim, indent=2, default=str))
    return {r["y"]: r for r in results}


if __name__ == "__main__":
    run()
