"""Shrink the global Y3 engine to a small core (brief 45→65 / who is improving).

Two global LightGBMs on y3_recover_cash_6m, stressed/labeled rows only,
families never B, lags 1 and 3, 5 train group-fold CV. Holdout companies
never enter a fit, a pick, or a percentile.

A) SHAP-stable / perm-stable stems from y3_importances.md (plus their lags).
   c_ss_month, c_salary_month, a_n_tx, f_ds_r, c_n_days_with_tx.
   a_op_in is SIZE and is dropped. e_dso_proxy is not in this A set
   (optional 6th left out so A stays the smaller candidate).

B) feature_report.md recommended-keep ∩ ratios/flags. Still never B.
   Raw euro levels (a_in3, a_op_in, …) and count stems are dropped.
   Cluster reps replace redundant levels (c_gap_sd, not a_n_tx).

Both A and B reuse the published gbm_y3y6 LightGBM + early-stopping spec
so the number is comparable to CV 0.710 (n_x=278). A third pass is A
under the night 0.762 spec (50 trees, max_depth=3) — diagnostic only.

KEEP if either A or B has group-fold CV AUROC >= 0.710 with fewer columns
(prefer the smaller). PARK if both lose by more than 0.02.
Quote train group-fold CV only. Holdout is LOW_POWER and is not scored.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_core
"""
from __future__ import annotations

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
from analysis.models.gbm_y3y6 import (
    LAGS,
    LGB_BASE,
    N_FOLDS,
    STORE,
    _assert_allowed,
    _family_of,
    _fit_lgb,
    _gain_table,
    _keys,
    _scale_pos_weight,
    add_lags,
    append_registry,
    load_store,
)

Y_COL = "y3_recover_cash_6m"
ALLOWED = ("a", "c", "d", "e", "f", "g", "h")
FORBIDDEN = ("b",)
AGENT = "7f37f0fd"
WAVE = 4
ROUND = "R4"
PUBLISHED_CV = 0.710
PARK_DELTA = 0.02
PUBLISHED_N_X = 278
CLIP_DSO = 24.0
TARGETS = DATA / "feature_store" / "targets.parquet"

# A: signed SHAP + fold-perm core. Not a_op_in. Not a_transfer (perm≈0).
CORE_A = (
    "c_ss_month",
    "c_salary_month",
    "a_n_tx",
    "f_ds_r",
    "c_n_days_with_tx",
)

# B: keep-list ratios / flags only. Never B. No raw euro. No counts.
CORE_B = (
    "a_growth_12",
    "a_growth_3",
    "a_io_ratio",
    "a_uncat_share",
    "c_gap_sd",
    "c_last_tx_before_2026_06",
    "c_missed_salary",
    "c_missed_tax",
    "c_recency_days",
    "c_salary_month",
    "c_ss_month",
    "c_tax_month",
    "c_zero_in_month",
    "c_zero_in_share_6",
    "d_cust_hhi",
    "d_supp_hhi",
    "d_tx_cp_share",
    "e_ap_overdue_30",
    "e_ar_overdue",
    "e_ar_overdue_30",
    "e_credit_note_ratio",
    "e_delay_coll",
    "e_delay_paid",
    "e_dpo_proxy",
    "e_dso_proxy",
    "e_fx_share",
    "e_pending_amt_share",
    "f_ds_r",
    "f_fc_r",
    "f_has_confirming",
    "f_has_factoring",
    "f_has_loc",
    "f_new_facility",
    "f_outstanding_gt_granted",
    "g_custom_share",
    "g_has_card",
    "g_has_checking",
    "g_has_investment",
    "g_has_saving",
    "g_has_tpv",
    "g_new_this_month",
    "h_sib_neg_share",
)

BANNED_STEMS = {
    "a_op_in",
    "a_in3",
    "a_in6",
    "a_in12",
    "a_op_out",
    "a_out3",
    "a_out6",
    "a_out12",
    "a_net",
    "a_fin_cost",
    "a_invest",
    "a_transfer",
    "a_debt_service",
}

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

SPECS = (
    {
        "id": "A",
        "model": "lgbm_y3_core_shap",
        "stems": CORE_A,
        "shallow": False,
        "notes": (
            "SHAP-stable core only (plus lags 1,3); never B; drop a_op_in SIZE; "
            "no e_dso_proxy (optional 6th left out)"
        ),
    },
    {
        "id": "B",
        "model": "lgbm_y3_core_keep",
        "stems": CORE_B,
        "shallow": False,
        "notes": (
            "feature_report keep ∩ ratios/flags; never B; drop raw euro levels "
            "and a_op_in; no count stems"
        ),
    },
    {
        "id": "A_shallow",
        "model": "lgbm_y3_core_shap_d3_n50",
        "stems": CORE_A,
        "shallow": True,
        "notes": (
            "same X as A; 0.762 spec (50 trees, max_depth=3); diagnostic — "
            "not used for KEEP/PARK"
        ),
    },
)


def _clip_proxies(df: pd.DataFrame) -> pd.DataFrame:
    """Fixed 24-month clip. Constant, not a fit — holdout never used."""
    out = df.copy()
    for c in ("e_dso_proxy", "e_dpo_proxy"):
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        out[c] = s.clip(lower=0.0, upper=CLIP_DSO)
    return out


def _resolve_stems(panel: pd.DataFrame, stems: tuple[str, ...]) -> list[str]:
    have = [c for c in stems if c in panel.columns]
    missing = [c for c in stems if c not in panel.columns]
    if missing:
        print(f"WARN missing stems (skipped): {missing}")
    bad = [c for c in have if c in BANNED_STEMS or str(c).startswith("b_")]
    if bad:
        raise RuntimeError(f"banned / family-B stem in core: {bad}")
    if not have:
        raise RuntimeError("no requested stems present on the panel")
    return have


def _feat_from_stems(panel: pd.DataFrame, stems: list[str]) -> list[str]:
    cols: list[str] = []
    for c in stems:
        cols.append(c)
        for k in LAGS:
            lag = f"{c}_lag{k}"
            if lag not in panel.columns:
                raise RuntimeError(f"missing lag column {lag}")
            cols.append(lag)
    leak = leakage_check(cols, Y_COL, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage: {leak['issues']}")
    _assert_allowed(cols, Y_COL, FORBIDDEN)
    banned = [c for c in cols if c.split("_lag")[0] in BANNED_STEMS or str(c).startswith("b_")]
    if banned:
        raise RuntimeError(f"banned columns in X: {banned}")
    yish = [c for c in cols if c == Y_COL or str(c).startswith("y")]
    if yish:
        raise RuntimeError(f"y columns in X: {yish}")
    return cols


def _fit_shallow(Xtr, ytr) -> lgb.LGBMClassifier:
    params = dict(LGB_SHALLOW)
    params["scale_pos_weight"] = _scale_pos_weight(pd.Series(ytr))
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    return clf


def _best_single_cv(
    panel: pd.DataFrame,
    feat_cols: list[str],
    train_lab: pd.Series,
) -> dict:
    """Per-fold train pick, val score. Holdout rows are already out of train_lab."""
    rows = []
    pick_counts: dict[str, int] = {}
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        n_tr = int(ytr.notna().sum())
        min_n = max(200, int(0.25 * n_tr)) if n_tr else 200
        best = None
        for col in feat_cols:
            s = pd.to_numeric(panel[col], errors="coerce")
            trd = pd.DataFrame({"y": ytr, "s": s[tr]}).dropna()
            if len(trd) < min_n or trd["y"].nunique() < 2 or trd["s"].nunique() < 2:
                continue
            auc_p = auroc(trd["y"], trd["s"])
            auc_n = auroc(trd["y"], -trd["s"])
            if not np.isfinite(auc_p) and not np.isfinite(auc_n):
                continue
            sign = -1 if (np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p)) else 1
            train_auc = auc_n if sign < 0 else auc_p
            if best is None or train_auc > best["train_auroc"]:
                best = {"feature": col, "sign": sign, "train_auroc": float(train_auc)}
        if best is None or yva.nunique() < 2:
            rows.append({"fold": k, "auroc": float("nan"), "feature": None})
            continue
        sva = best["sign"] * pd.to_numeric(panel.loc[va, best["feature"]], errors="coerce")
        vad = pd.DataFrame({"y": yva, "s": sva}).dropna()
        auc = auroc(vad["y"], vad["s"]) if len(vad) and vad["y"].nunique() == 2 else float("nan")
        rows.append(
            {
                "fold": k,
                "auroc": auc,
                "feature": best["feature"],
                "sign": best["sign"],
                "train_auroc": best["train_auroc"],
            }
        )
        pick_counts[best["feature"]] = pick_counts.get(best["feature"], 0) + 1
    aucs = np.asarray([r["auroc"] for r in rows], dtype=float)
    ok = aucs[np.isfinite(aucs)]
    top = max(pick_counts, key=pick_counts.get) if pick_counts else None
    return {
        "cv_auroc": float(ok.mean()) if ok.size else float("nan"),
        "folds": rows,
        "picked_most": top,
        "pick_counts": pick_counts,
    }


def _verdict(results: dict[str, dict]) -> dict:
    a = results["A"]["cv_auroc"]
    b = results["B"]["cv_auroc"]
    n_a = results["A"]["n_x"]
    n_b = results["B"]["n_x"]
    cand = []
    if np.isfinite(a) and a >= PUBLISHED_CV and n_a < PUBLISHED_N_X:
        cand.append(("A", a, n_a))
    if np.isfinite(b) and b >= PUBLISHED_CV and n_b < PUBLISHED_N_X:
        cand.append(("B", b, n_b))
    if cand:
        cand.sort(key=lambda t: (t[2], -t[1]))
        winner = cand[0][0]
        return {
            "decision": "KEEP",
            "winner": winner,
            "reason": (
                f"{winner} CV {results[winner]['cv_auroc']:.3f} >= {PUBLISHED_CV:.3f} "
                f"with n_x={results[winner]['n_x']} < {PUBLISHED_N_X}"
            ),
        }
    lose_a = (not np.isfinite(a)) or a < PUBLISHED_CV - PARK_DELTA
    lose_b = (not np.isfinite(b)) or b < PUBLISHED_CV - PARK_DELTA
    if lose_a and lose_b:
        return {
            "decision": "PARK",
            "winner": None,
            "reason": (
                f"both lose by >{PARK_DELTA:.2f} vs {PUBLISHED_CV:.3f} "
                f"(A={a:.3f}, B={b:.3f})"
            ),
        }
    return {
        "decision": "CLOSE",
        "winner": None,
        "reason": (
            f"neither reaches {PUBLISHED_CV:.3f} but at least one is within "
            f"{PARK_DELTA:.2f} (A={a:.3f}, B={b:.3f})"
        ),
    }


def load_y3() -> pd.DataFrame:
    if TARGETS.exists():
        raw = pd.read_parquet(TARGETS)
        if {Y_COL, "company_id", "period"} <= set(raw.columns):
            panel = _keys(raw[["company_id", "period", Y_COL]])
            print(f"Y from {TARGETS} shape={panel.shape}")
            return panel
    from analysis.features.grid import monthly_grid
    from analysis.targets.y3_recovery import build as build_y3

    con = connect()
    grid = _keys(monthly_grid(con)[["company_id", "period"]])
    y3 = _keys(build_y3(con, grid))
    con.close()
    print(f"Y from build_y3 shape={y3.shape}")
    return y3[["company_id", "period", Y_COL]]


def run_spec(
    spec: dict,
    store: pd.DataFrame,
    y_panel: pd.DataFrame,
    folds: pd.DataFrame,
) -> dict:
    print("\n" + "=" * 72)
    print(
        f"SPEC {spec['id']} model={spec['model']} shallow={spec['shallow']} "
        f"stems={len(spec['stems'])}"
    )
    print("=" * 72)

    panel = store.merge(y_panel, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    panel = _clip_proxies(panel)

    stems = _resolve_stems(panel, spec["stems"])
    print(f"stems present={stems}")
    panel = add_lags(panel, stems, LAGS)
    feat_cols = _feat_from_stems(panel, stems)
    fams = sorted({_family_of(c) for c in stems})
    print(f"n_x={len(feat_cols)} families={fams} lags={list(LAGS)}")

    is_train = train_mask(panel["company_id"])
    labeled = panel[Y_COL].notna()
    train_lab = is_train & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    hold_in_train = set(panel.loc[train_lab, "company_id"].astype(str)) & load_holdout()
    if hold_in_train:
        raise RuntimeError("holdout leaked into train labeled rows")

    n_lab = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, Y_COL] == 1).sum())
    rate = float(panel.loc[train_lab, Y_COL].mean()) if n_lab else float("nan")
    n_train_cm = int(is_train.sum())
    coverage = (n_lab / n_train_cm) if n_train_cm else float("nan")
    print(
        f"train_cm={n_train_cm} stressed_labeled={n_lab} pos={n_pos} "
        f"rate={rate:.4f} coverage={coverage:.4f}"
    )

    cv_rows = []
    best_iters = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"fold {k}: skip (one class empty)")
            cv_rows.append(
                {"fold": k, "auroc": float("nan"), "pr_auc": float("nan"), "n_val": int(va.sum())}
            )
            continue
        if spec["shallow"]:
            clf = _fit_shallow(panel.loc[tr, feat_cols], ytr)
            trees = int(LGB_SHALLOW["n_estimators"])
        else:
            clf = _fit_lgb(panel.loc[tr, feat_cols], ytr, panel.loc[va, feat_cols], yva)
            trees = int(getattr(clf, "best_iteration_", LGB_BASE["n_estimators"]) or LGB_BASE["n_estimators"])
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "pr_auc": pr_auc(yva, pred),
            "n_val": int(va.sum()),
            "n_pos": int((yva == 1).sum()),
            "best_iteration": trees,
        }
        best_iters.append(trees)
        cv_rows.append(row)
        print(
            f"fold {k}: auroc={row['auroc']:.4f} pr_auc={row['pr_auc']:.4f} "
            f"n={row['n_val']} pos={row['n_pos']} trees={trees}"
        )

    aucs = np.asarray([r["auroc"] for r in cv_rows], dtype=float)
    ok = aucs[np.isfinite(aucs)]
    cv_auroc = float(ok.mean()) if ok.size else float("nan")
    cv_sd = float(ok.std(ddof=1)) if ok.size >= 2 else float("nan")
    prs = np.asarray([r.get("pr_auc", float("nan")) for r in cv_rows], dtype=float)
    pr_ok = prs[np.isfinite(prs)]
    cv_pr = float(pr_ok.mean()) if pr_ok.size else float("nan")
    print(
        f"CV mean AUROC={cv_auroc:.4f} sd={cv_sd:.4f} PR-AUC={cv_pr:.4f} "
        f"n_x={len(feat_cols)} vs_published={PUBLISHED_CV:.3f}"
    )

    single = _best_single_cv(panel, feat_cols, train_lab)
    print(
        f"best-single CV AUROC={single['cv_auroc']:.4f} "
        f"picked_most={single['picked_most']} counts={single['pick_counts']}"
    )
    dummy_cv = 0.5
    beats_dummy = np.isfinite(cv_auroc) and cv_auroc > dummy_cv
    beats_single = (
        np.isfinite(cv_auroc)
        and np.isfinite(single["cv_auroc"])
        and cv_auroc > single["cv_auroc"]
    )
    print(f"beats dummy={beats_dummy} beats_single={beats_single}")

    # Train-only fit for a gain peek. Holdout never in this slice.
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    ytr_all = panel.loc[train_lab, Y_COL].astype(float)
    if spec["shallow"]:
        final = _fit_shallow(panel.loc[train_lab, feat_cols], ytr_all)
        n_trees = int(LGB_SHALLOW["n_estimators"])
    else:
        n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
        n_trees = max(50, n_trees)
        final = _fit_lgb(panel.loc[train_lab, feat_cols], ytr_all, n_estimators=n_trees)
    imp = _gain_table(final, feat_cols, 12)
    print("top gain (train fit, not a claim)")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    return {
        "id": spec["id"],
        "model": spec["model"],
        "shallow": spec["shallow"],
        "stems": stems,
        "feat_cols": feat_cols,
        "n_x": len(feat_cols),
        "n_stems": len(stems),
        "families": fams,
        "cv_auroc": cv_auroc,
        "cv_auroc_sd": cv_sd,
        "cv_pr_auc": cv_pr,
        "cv_folds": cv_rows,
        "n_trees_median": n_trees,
        "train_labeled": n_lab,
        "train_pos": n_pos,
        "train_base_rate": rate,
        "coverage": coverage,
        "dummy_cv": dummy_cv,
        "single": single,
        "beats_dummy": beats_dummy,
        "beats_single": beats_single,
        "top12": imp.to_dict(orient="records"),
        "notes": spec["notes"],
    }


def registry_rows(ts: str, results: list[dict], verdict: dict) -> list[dict]:
    out = []
    for r in results:
        folds = ",".join(
            f"{x['fold']}:{x['auroc']:.4f}" if np.isfinite(x.get("auroc", float("nan"))) else f"{x['fold']}:nan"
            for x in r["cv_folds"]
        )
        fam = "+".join(s.upper() for s in r["families"])
        cov = r["coverage"]
        notes = (
            f"{verdict['decision']}; {r['notes']}; vs_published={PUBLISHED_CV:.3f}/n_x={PUBLISHED_N_X}; "
            f"cv={r['cv_auroc']:.4f}±{r['cv_auroc_sd']:.4f}; n_x={r['n_x']}; stems={r['n_stems']}; "
            f"lags=1,3; trees={r['n_trees_median']}; "
            f"vs_dummy={r['cv_auroc']:.3f}/{r['dummy_cv']:.3f}; "
            f"vs_single={r['single'].get('picked_most')} {r['single']['cv_auroc']}; "
            f"beats_single={r['beats_single']}; folds={folds}; "
            f"train={r['train_labeled']}/{r['train_pos']}/{r['train_base_rate']:.4f}; "
            f"quote=cv5_group_not_holdout; {verdict['reason']}"
        )
        cov_s = f"{cov:.4f}" if np.isfinite(cov) else ""
        val = f"{r['cv_auroc']:.6g}" if np.isfinite(r["cv_auroc"]) else ""
        out.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": fam,
                "y": Y_COL,
                "model": r["model"],
                "split": "cv5_group",
                "metric": "auroc",
                "value": val,
                "coverage": cov_s,
                "notes": notes,
            }
        )
        if np.isfinite(r["cv_auroc_sd"]):
            out.append(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": fam,
                    "y": Y_COL,
                    "model": r["model"],
                    "split": "cv5_group",
                    "metric": "auroc_sd",
                    "value": f"{r['cv_auroc_sd']:.6g}",
                    "coverage": cov_s,
                    "notes": notes,
                }
            )
        if r["single"].get("picked_most") and not r["shallow"]:
            out.append(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": _family_of(str(r["single"]["picked_most"])).upper(),
                    "y": Y_COL,
                    "model": f"single_{r['single']['picked_most']}",
                    "split": "cv5_group",
                    "metric": "auroc",
                    "value": (
                        f"{r['single']['cv_auroc']:.6g}"
                        if np.isfinite(r["single"]["cv_auroc"])
                        else ""
                    ),
                    "coverage": cov_s,
                    "notes": (
                        f"train-fold pick among {r['id']} cols; "
                        f"picked_most={r['single']['picked_most']}; "
                        f"counts={r['single']['pick_counts']}"
                    ),
                }
            )
    return out


def run() -> dict:
    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    store = _keys(store)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    if set(train_cos["company_id"].astype(str)) & hold_ids:
        raise RuntimeError("holdout companies in train_companies()")
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    con.close()
    y_panel = load_y3()
    print(
        f"store={store.shape} source={x_source} train_cos={train_cos['company_id'].nunique()} "
        f"hold_cos={len(hold_ids)} folds={folds['fold'].nunique()}"
    )
    print("Y3 stressed-only (label already NaN off the path). Never family B. Never a_op_in.")

    results = []
    for spec in SPECS:
        results.append(run_spec(spec, store, y_panel, folds))
    by_id = {r["id"]: r for r in results}
    verdict = _verdict(by_id)
    print("\nVERDICT")
    print(json.dumps(verdict, indent=2))

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    append_registry(registry_rows(ts, results, verdict))

    slim = {
        r["id"]: {
            "cv_auroc": r["cv_auroc"],
            "cv_auroc_sd": r["cv_auroc_sd"],
            "n_x": r["n_x"],
            "n_stems": r["n_stems"],
            "dummy_cv": r["dummy_cv"],
            "single_cv": r["single"]["cv_auroc"],
            "single": r["single"].get("picked_most"),
            "beats_dummy": r["beats_dummy"],
            "beats_single": r["beats_single"],
            "train_labeled": r["train_labeled"],
            "train_pos": r["train_pos"],
        }
        for r in results
    }
    slim["verdict"] = verdict
    slim["published"] = {"cv": PUBLISHED_CV, "n_x": PUBLISHED_N_X, "shallow_full": 0.762}
    print("\nQUOTE (train group-fold CV only; not holdout)")
    print(json.dumps(slim, indent=2, default=str))
    return {"results": by_id, "verdict": verdict, "slim": slim}


if __name__ == "__main__":
    run()
