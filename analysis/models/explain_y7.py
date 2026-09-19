"""Explain the accepted Y7 concentration-shock LightGBM (train only).

Refits the same `y7_top1_lost` spec as `analysis.models.gbm_y7y8`
(X families A,B,C,E,F,G,H — never D; lags 1,3). Holdout companies never
enter fit, early stopping, or the SHAP sample.

Y7 is brief question 4 (dip vs fall: top AR customer gone next quarter),
not bankruptcy. TreeSHAP answers questions 5–6 (why / how many months
earlier). Quote published group-fold CV AUROC 0.663; holdout 0.680 is
not the claim.

Primary importance: mean |TreeSHAP| on a train-only sample (≤4000 rows,
seed 20260918). Direction = sign of Spearman(feature, SHAP) so '+' means
higher feature → higher P(top-1 lost).

Writes `analysis/outputs/shap_y7.md`, `shap_y7_meanabs.csv`,
`shap_y7_summary_bar.png`, `shap_y7_beeswarm.png`.
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
    load_holdout,
    train_companies,
)
from analysis.features.common import ANALYSIS, connect, train_mask
from analysis.models.gbm_y7y8 import (
    LAGS,
    LGB_BASE,
    META_OK,
    N_FOLDS,
    _assert_allowed,
    _family_of,
    _fit_lgb,
    _keys,
    add_lags,
    allowed_x_cols,
    load_store,
    load_y,
)

OUT_DIR = ANALYSIS / "outputs"
OUT_MD = OUT_DIR / "shap_y7.md"
OUT_CSV = OUT_DIR / "shap_y7_meanabs.csv"
OUT_BAR = OUT_DIR / "shap_y7_summary_bar.png"
OUT_BEE = OUT_DIR / "shap_y7_beeswarm.png"

Y_COL = "y7_top1_lost"
ALLOWED = ("a", "b", "c", "e", "f", "g", "h")
FORBIDDEN = ("d",)
TOP_N = 15
SAMPLE_N = 4000
SAMPLE_SEED = 20260918
PUBLISHED_CV = 0.663
SIZE_CUT = 0.85
NEAR_SIZE_CUT = 0.70
CONC_COPY_CUT = 0.80

SIZE_STEMS = {
    "a_op_in",
    "a_in3",
    "a_in6",
    "a_in12",
    "a_op_out",
    "a_out3",
    "a_out6",
    "a_out12",
    "a_net",
    "a_n_tx",
    "c_n_tx",
    "c_n_days_with_tx",
    "h_share_group_in",
    "h_sib_in",
    "h_sib_out",
    "e_ar_issued",
    "e_ap_issued",
    "e_ar_open",
    "e_ap_open",
}

GLOSS = {
    "a_growth_3": "trailing-3m inflow growth vs t-3",
    "a_growth_12": "YoY of trailing-3m inflow",
    "a_in3": "trailing 3-month operational inflow (euro)",
    "a_in6": "trailing 6-month operational inflow (euro)",
    "a_in12": "trailing 12-month operational inflow (euro)",
    "a_io_ratio": "in3 / out3 coverage, capped at 3",
    "a_n_tx": "transaction count this month",
    "a_net": "op_in − op_out this month",
    "a_net_margin": "(in3 − out3) / in3",
    "a_op_in": "operational inflow this month",
    "a_op_out": "operational outflow this month",
    "a_out3": "trailing 3-month operational outflow",
    "a_out6": "trailing 6-month operational outflow",
    "a_out12": "trailing 12-month operational outflow",
    "a_transfer": "signed net transfers this month",
    "a_invest": "signed net investment flows this month",
    "a_fin_cost": "finance-cost outflow this month",
    "a_debt_service": "debt-service outflow this month",
    "a_uncat_share": "share of uncategorized transactions",
    "a_pending_share": "share of pending (vs booked) transactions",
    "b_liq": "reconstructed period-end cash",
    "b_runway": "cash / monthly operational outflow, clipped",
    "b_d_runway": "runway change vs t−3",
    "b_neg_liq_3": "share of last 3 months with liq < 0",
    "b_min_liq_3": "minimum reconstructed cash in last 3 months",
    "b_mean_liq_3": "mean reconstructed cash in last 3 months",
    "b_below_0": "1 if reconstructed cash is negative",
    "b_below_half_runway": "1 if cash < half a month of outflow",
    "b_bal_vol": "6-month cash volatility / mean outflow",
    "b_neg_episodes": "count of new negative-cash onsets in 6 months",
    "c_gap_sd": "stdev of inter-booking-day gaps (90d)",
    "c_n_days_with_tx": "distinct booking days this month",
    "c_n_tx": "transaction count this month",
    "c_recency_days": "days since last booking",
    "c_salary_month": "any salary booking this month",
    "c_ss_month": "any social-security booking this month",
    "c_tax_month": "any tax booking this month",
    "c_zero_in_month": "no inflow booking this month",
    "c_zero_in_share_6": "share of last 6 months with zero inflow",
    "c_missed_salary": "usual payroll, missed this month",
    "c_missed_tax": "usual tax booking, missed this month",
    "c_last_tx_before_2026_06": "silent since before 2026-06 as of this month",
    "e_ar_issued": "AR issuance volume this period",
    "e_ap_issued": "AP issuance volume this period",
    "e_ar_open": "unpaid AR stock at period end",
    "e_ap_open": "unpaid AP stock at period end",
    "e_ar_overdue": "overdue AR / open AR",
    "e_ap_overdue": "overdue AP / open AP",
    "e_ar_overdue_30": "open-AR share >30 days late",
    "e_ap_overdue_30": "open-AP share >30 days late",
    "e_delay_coll": "amount-weighted AR collection delay (days)",
    "e_delay_paid": "amount-weighted AP payment delay (days)",
    "e_dso_proxy": "AR open / this-period AR issued",
    "e_dpo_proxy": "AP open / this-period AP issued",
    "e_credit_note_ratio": "credit-notes / (invoices + credit-notes)",
    "e_pending_amt_share": "reconstructed pending share of issued invoices",
    "e_fx_share": "share of issued |amount| not in accounting currency",
    "f_ds_r": "trailing debt-service / inflow",
    "f_fc_r": "trailing finance-cost / inflow",
    "f_debt_service": "debt-repayment outflow this month",
    "f_fin_cost": "fee + interest outflow this month",
    "f_n_facilities": "debt facilities known as of period end",
    "f_n_types": "distinct debt product types as of period end",
    "f_util_snapshot": "outstanding / granted (extract snapshot)",
    "f_has_loc": "has a line of credit",
    "f_has_factoring": "has factoring",
    "f_has_confirming": "has confirming",
    "f_w_rate": "granted-weighted interest / spread",
    "f_months_to_next_pay": "months to next scheduled payment",
    "f_sched_vs_obs": "scheduled installment / observed debt service",
    "f_outstanding_gt_granted": "any facility with outstanding > granted",
    "f_new_facility": "new debt products opened this period",
    "g_n_accounts": "banking products as of period end",
    "g_n_banks": "distinct banks as of period end",
    "g_n_types": "distinct banking product types",
    "g_has_card": "has a card product",
    "g_has_tpv": "has a TPV / POS product",
    "g_has_checking": "has a checking account",
    "g_has_saving": "has a saving account",
    "g_has_investment": "has an investment product",
    "g_custom_share": "share of custom / customer-defined banking products",
    "g_created_unknown_share": "share of banking products with null created_at",
    "g_new_this_month": "banking products created this period",
    "group_size": "companies in the group (static)",
    "h_group_size": "companies in the group (static)",
    "h_n_siblings_active": "other group companies with txs this month",
    "h_share_group_in": "company share of group inflow",
    "h_sib_in": "sibling operational inflow",
    "h_sib_out": "sibling operational outflow",
    "h_sib_net": "sibling operational net",
    "h_sib_neg_share": "share of siblings with negative net",
    "n_banking": "count of banking products (snapshot)",
}

LEAD_STEMS = {
    "c_recency_days",
    "c_last_tx_before_2026_06",
    "b_d_runway",
    "a_growth_3",
    "a_growth_12",
}
CASH_SHAPE_STEMS = {
    "a_io_ratio",
    "a_net_margin",
    "a_growth_3",
    "a_growth_12",
    "a_net",
}
OPS_STEMS = {
    "a_n_tx",
    "a_uncat_share",
    "c_ss_month",
    "c_salary_month",
    "c_tax_month",
    "c_n_tx",
    "c_n_days_with_tx",
    "c_zero_in_month",
    "c_zero_in_share_6",
    "c_missed_salary",
    "c_missed_tax",
    "c_gap_sd",
}


def _stem(col: str) -> str:
    name = str(col)
    for suf in ("_lag1", "_lag3", "_lag6"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _is_lag(col: str) -> bool:
    return str(col).endswith(("_lag1", "_lag3", "_lag6"))


def _gloss(col: str) -> str:
    stem = _stem(col)
    base = GLOSS.get(stem, stem.replace("_", " "))
    if col.endswith("_lag1"):
        return f"{base} (t−1)"
    if col.endswith("_lag3"):
        return f"{base} (t−3)"
    return base


def _fam(col: str) -> str:
    if col in META_OK or _stem(col) in META_OK:
        return "meta"
    return _family_of(col)


def brief_bucket(col: str) -> str:
    """Map a column to brief questions 5–6 (and the Q4 cash-shape residual)."""
    if _is_lag(col) or _stem(col) in LEAD_STEMS:
        return "lead_time / turning"
    fam = _fam(col)
    stem = _stem(col)
    if fam == "e":
        return "why (counterparties / invoices)"
    if fam == "c" or stem in OPS_STEMS:
        return "why (ops regularity)"
    if fam == "f":
        return "why (financing)"
    if fam == "b" or stem in CASH_SHAPE_STEMS:
        return "healthy vs stressed cash shape"
    return "other"


def _spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def _univ_sign(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": y}).dropna()
    if len(d) < 30 or d["y"].nunique() < 2 or d["x"].nunique() < 2:
        return 0, float("nan")
    auc_p = auroc(d["y"], d["x"])
    auc_n = auroc(d["y"], -d["x"])
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 0, float("nan")
    if np.isfinite(auc_n) and (not np.isfinite(auc_p) or auc_n > auc_p):
        return -1, float(auc_n)
    return 1, float(auc_p)


def _try_shap(clf: lgb.LGBMClassifier, X: pd.DataFrame) -> tuple[np.ndarray | None, str]:
    try:
        import shap
    except Exception as exc:
        return None, f"import failed: {type(exc).__name__}: {exc}"
    try:
        explainer = shap.TreeExplainer(clf)
        raw = explainer.shap_values(X)
        if isinstance(raw, list):
            arr = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        else:
            arr = np.asarray(raw)
        if arr.ndim == 3:
            arr = arr[:, :, -1]
        if arr.shape != (len(X), X.shape[1]):
            explainer = shap.TreeExplainer(clf.booster_)
            raw = explainer.shap_values(X)
            arr = np.asarray(raw[1] if isinstance(raw, list) and len(raw) > 1 else raw)
            if arr.ndim == 3:
                arr = arr[:, :, -1]
        if arr.shape != (len(X), X.shape[1]):
            return None, f"unexpected shap shape {arr.shape} vs {X.shape}"
        return arr.astype(float), f"shap {getattr(shap, '__version__', '?')} TreeExplainer"
    except Exception as exc:
        return None, f"TreeExplainer failed: {type(exc).__name__}: {exc}"


def _sign_char(s: int) -> str:
    if s > 0:
        return "+"
    if s < 0:
        return "−"
    return "?"


def _prepare_panel() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series, str, str]:
    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    grid = store[["company_id", "period"]].drop_duplicates()
    y7, _y8, y_source = load_y(con, grid)
    con.close()
    if Y_COL not in y7.columns:
        raise RuntimeError(f"{Y_COL} missing from Y source={y_source}")

    panel = store.merge(y7[["company_id", "period", Y_COL]], on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    base_cols = allowed_x_cols(panel, ALLOWED, Y_COL)
    _assert_allowed(base_cols, Y_COL, FORBIDDEN)
    lag_cols = [c for c in base_cols if c not in META_OK]
    panel = add_lags(panel, lag_cols, LAGS)
    feat_cols = allowed_x_cols(panel, ALLOWED, Y_COL)
    _assert_allowed(feat_cols, Y_COL, FORBIDDEN)
    bad_d = [c for c in feat_cols if str(c).startswith("d_")]
    if bad_d:
        raise RuntimeError(f"D leaked into X: {bad_d[:8]}")

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[Y_COL].notna()
    tr = is_train & labeled
    assert_no_holdout(panel.loc[tr, "company_id"])
    print(
        f"store={store.shape} source={x_source} y_source={y_source} X={len(feat_cols)} "
        f"train_labeled={int(tr.sum())} hold_labeled={int((is_hold & labeled).sum())} "
        f"(holdout unused for fit/explain)"
    )
    return panel, feat_cols, tr, is_hold & labeled, x_source, y_source


def _fit_cv_and_final(panel: pd.DataFrame, feat_cols: list[str], tr: pd.Series):
    cv_rows = []
    best_iters = []
    for k in range(N_FOLDS):
        tr_k = tr & (panel["fold"] != k)
        va_k = tr & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr_k, "company_id"])
        assert_no_holdout(panel.loc[va_k, "company_id"])
        ytr = panel.loc[tr_k, Y_COL].astype(float)
        yva = panel.loc[va_k, Y_COL].astype(float)
        if ytr.nunique() < 2 or yva.nunique() < 2:
            print(f"fold {k}: skip")
            continue
        clf = _fit_lgb(panel.loc[tr_k, feat_cols], ytr, panel.loc[va_k, feat_cols], yva)
        pred = clf.predict_proba(panel.loc[va_k, feat_cols])[:, 1]
        trees = int(getattr(clf, "best_iteration_", LGB_BASE["n_estimators"]) or LGB_BASE["n_estimators"])
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "n_val": int(va_k.sum()),
            "n_pos": int((yva == 1).sum()),
            "trees": trees,
        }
        best_iters.append(trees)
        cv_rows.append(row)
        print(f"fold {k}: auroc={row['auroc']:.4f} n={row['n_val']} pos={row['n_pos']} trees={trees}")

    cv_auroc = float(np.nanmean([r["auroc"] for r in cv_rows])) if cv_rows else float("nan")
    n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
    n_trees = max(50, n_trees)
    print(f"CV mean AUROC={cv_auroc:.4f} n_trees_final={n_trees} published_cv={PUBLISHED_CV}")

    ytr_all = panel.loc[tr, Y_COL].astype(float)
    assert_no_holdout(panel.loc[tr, "company_id"])
    final = _fit_lgb(panel.loc[tr, feat_cols], ytr_all, n_estimators=n_trees)
    return final, cv_rows, cv_auroc, n_trees


def _sample_train(
    panel: pd.DataFrame, tr: pd.Series, feat_cols: list[str]
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    idx = panel.index[tr].to_numpy()
    assert_no_holdout(panel.loc[idx, "company_id"])
    n_full = int(len(idx))
    if n_full > SAMPLE_N:
        rng = np.random.default_rng(SAMPLE_SEED)
        idx = rng.choice(idx, size=SAMPLE_N, replace=False)
        idx = np.sort(idx)
    assert_no_holdout(panel.loc[idx, "company_id"])
    X = panel.loc[idx, feat_cols]
    y = panel.loc[idx, Y_COL].astype(float)
    print(f"SHAP sample={len(X)} of {n_full} train labeled seed={SAMPLE_SEED}")
    return X, y, idx


def _leak_row(panel: pd.DataFrame, tr: pd.Series, col: str) -> dict:
    x = panel.loc[tr, col]
    y = panel.loc[tr, Y_COL]
    size = np.log1p(pd.to_numeric(panel.loc[tr, "a_in3"], errors="coerce").abs())
    rho_size = _spearman(x, size)
    if "a_op_in" in panel.columns:
        size_m = np.log1p(pd.to_numeric(panel.loc[tr, "a_op_in"], errors="coerce").abs())
        rho_opin = _spearman(x, size_m)
        if np.isfinite(rho_opin) and (not np.isfinite(rho_size) or abs(rho_opin) > abs(rho_size)):
            rho_size = rho_opin
    rho_top1 = (
        _spearman(x, panel.loc[tr, "d_cust_top1"]) if "d_cust_top1" in panel.columns else float("nan")
    )
    rho_hhi = _spearman(x, panel.loc[tr, "d_cust_hhi"]) if "d_cust_hhi" in panel.columns else float("nan")
    sign_u, auc_u = _univ_sign(x, y)
    stem = _stem(col)
    return {
        "feature": col,
        "stem": stem,
        "family": _fam(col),
        "rho_size": rho_size,
        "rho_d_top1": rho_top1,
        "rho_d_hhi": rho_hhi,
        "sign_univ": sign_u,
        "univ_auroc": auc_u,
        "known_size_stem": stem in SIZE_STEMS,
        "gloss": _gloss(col),
        "brief_bucket": brief_bucket(col),
    }


def _flags(row: dict) -> str:
    tags = []
    if row.get("known_size_stem"):
        tags.append("SIZE_STEM")
    rho_s = row.get("rho_size")
    if np.isfinite(rho_s):
        if abs(rho_s) >= SIZE_CUT:
            tags.append("SIZE")
        elif abs(rho_s) >= NEAR_SIZE_CUT:
            tags.append("NEAR_SIZE")
    for key, tag in (("rho_d_top1", "D_TOP1_COPY"), ("rho_d_hhi", "D_HHI_COPY")):
        rho = row.get(key)
        if np.isfinite(rho) and abs(rho) >= CONC_COPY_CUT:
            tags.append(tag)
    return ",".join(tags) if tags else "—"


def _write_plots(shap_arr: np.ndarray, X: pd.DataFrame) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    names = list(X.columns)
    data = X.to_numpy(dtype=float)
    note = "summary_plot"
    try:
        exp = shap.Explanation(values=shap_arr, data=data, feature_names=names)
        plt.figure(figsize=(8.2, 7.2))
        shap.plots.bar(exp, max_display=20, show=False)
        plt.savefig(OUT_BAR, dpi=140, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(8.2, 7.2))
        shap.plots.beeswarm(exp, max_display=20, show=False)
        plt.savefig(OUT_BEE, dpi=140, bbox_inches="tight")
        plt.close()
        note = "shap.plots.bar + beeswarm"
        return note
    except Exception as exc:
        print(f"shap.plots API failed ({type(exc).__name__}: {exc}); falling back to summary_plot")
        plt.close("all")

    shap.summary_plot(shap_arr, X, feature_names=names, plot_type="bar", show=False, max_display=20)
    plt.savefig(OUT_BAR, dpi=140, bbox_inches="tight")
    plt.close()
    shap.summary_plot(shap_arr, X, feature_names=names, show=False, max_display=20)
    plt.savefig(OUT_BEE, dpi=140, bbox_inches="tight")
    plt.close()
    return note


def _fmt(v, nd=5):
    if v is None or not np.isfinite(v):
        return "—"
    return f"{v:.{nd}g}" if nd >= 5 else f"{v:.{nd}f}"


def _judge(top: pd.DataFrame, fam_share: pd.Series, bucket_share: pd.Series, leak: dict) -> str:
    recs = top.to_dict("records")
    top10 = recs[:10]
    names = ", ".join(f"`{r['feature']}` {_sign_char(int(r['sign']))}" for r in top10)
    lead = float(bucket_share.get("lead_time / turning", 0.0))
    why_e = float(bucket_share.get("why (counterparties / invoices)", 0.0))
    why_c = float(bucket_share.get("why (ops regularity)", 0.0))
    why_f = float(bucket_share.get("why (financing)", 0.0))
    cash = float(bucket_share.get("healthy vs stressed cash shape", 0.0))
    tot = float(bucket_share.sum()) if len(bucket_share) else 0.0
    lead_pct = 100.0 * lead / tot if tot else 0.0
    q5_pct = 100.0 * (why_e + why_c + why_f) / tot if tot else 0.0
    conc = [r["feature"] for r in recs if "D_TOP1_COPY" in str(r.get("flags")) or "D_HHI_COPY" in str(r.get("flags"))]
    size = [r["feature"] for r in recs if "SIZE" in set(str(r.get("flags", "")).split(","))]
    e_now = [r for r in top10 if r["family"] == "e" and not _is_lag(r["feature"])]
    lags = [r for r in top10 if _is_lag(r["feature"])]
    b_hits = [r for r in top10 if r["family"] == "b"]
    conc_txt = (
        f" Leak flag: {', '.join('`' + c + '`' for c in conc)} tracks family-D concentration "
        f"(|ρ|≥{CONC_COPY_CUT:.2f}) — a back-door to the label path."
        if conc
        else " No top feature copies `d_cust_top1` / `d_cust_hhi` at |ρ|≥0.80; D stayed out of X."
    )
    size_txt = (
        f" Size clones in the top 15: {', '.join('`' + c + '`' for c in size)}."
        if size
        else " No |ρ|≥0.85 size clone in the top 15."
    )
    e_txt = (
        " Invoice-book volume/timing (family E, allowed) is the contemporaneous why: "
        + ", ".join(f"`{r['feature']}`" for r in e_now)
        + "."
        if e_now
        else " Family E is not in the top-10 contemporaneous set."
    )
    lag_txt = (
        " Lead time (question 6) sits in "
        + ", ".join(f"`{r['feature']}`" for r in lags)
        + "."
        if lags
        else " No lag column in the top 10 — question 6 has to be read from the lag mass, not the head."
    )
    b_txt = (
        " Liquidity already in the top 10 ("
        + ", ".join(f"`{r['feature']}`" for r in b_hits)
        + ") is the cash-side dip-vs-fall shape, not a default flag."
        if b_hits
        else " Family B is allowed and present in X, but it is not a top-10 name."
    )
    return (
        f"`y7_top1_lost` is the brief's dip-vs-fall event (question 4): the current top AR "
        f"customer issues nothing in t+1..t+3. It is not bankruptcy. Family D never enters X. "
        f"Question 5 (why it changed) and question 6 (how many months earlier) are the SHAP "
        f"object. Top 10: {names}. {e_txt} {lag_txt} {b_txt} "
        f"Across all X, lead-time / turning holds {lead_pct:.0f}% of mean |SHAP|; "
        f"explicit why-buckets (invoices + ops + financing) hold {q5_pct:.0f}%; "
        f"cash-shape residual is {cash:.3g}. Family shares are led by "
        + ", ".join(f"{k}={v:.3g}" for k, v in fam_share.head(4).items())
        + f".{size_txt}{conc_txt} "
        f"Quote train group-fold CV AUROC **{PUBLISHED_CV:.3f}** (this run {leak.get('cv_auroc', float('nan')):.3f}). "
        "Holdout 0.680 / 122 events is LOW_POWER — not the story."
    )


def _write_md(payload: dict) -> None:
    top = pd.DataFrame(payload["top15"])
    fam = pd.Series(payload["family_share"])
    buck = pd.Series(payload["bucket_share"])
    lines = [
        "# SHAP — y7_top1_lost",
        "",
        f"Generated {payload['ts']}.",
        "Train labeled company-months only. Holdout excluded from fit and sample.",
        "Brief map: this Y is **dip vs fall** (question 4) — the current top AR customer",
        "issues 0 in the next quarter. It is **not** bankruptcy.",
        "Question 5 (why) = contemporaneous features. Question 6 (lead time) = lag1/lag3 mass.",
        "",
        f"- rows used: {payload['n_sample']} (of {payload['n_train']} train labeled)",
        f"- positives in full train labeled: {payload['n_pos']}",
        f"- train base rate: {payload['base_rate']:.4f}",
        f"- X columns: {payload['n_x']} families ABCEFGH (never D)",
        f"- trees: {payload['n_trees']}",
        f"- store: {payload['x_source']} / Y: {payload['y_source']}",
        f"- importance: {payload['importance_source']}",
        f"- this-run group-fold CV AUROC: {payload['cv_auroc']:.3f} "
        f"(published bake-off **{PUBLISHED_CV:.3f}**; holdout 0.680 is not the claim)",
        "- Sign `+` = higher feature → higher P(top-1 lost).",
        "",
        "## Top 15 by mean |SHAP|",
        "",
        "| rank | feature | sign | mean\\|SHAP\\| | family | brief bucket |",
        "| --- | --- | :---: | ---: | --- | --- |",
    ]
    for i, r in enumerate(top.to_dict("records"), 1):
        lines.append(
            f"| {i} | `{r['feature']}` | {_sign_char(int(r['sign']))} | "
            f"{_fmt(r.get('mean_abs_shap'))} | {r.get('family')} | {r.get('brief_bucket')} |"
        )
    lines += [
        "",
        "### What each of the top 10 is (questions 5–6)",
        "",
    ]
    for i, r in enumerate(top.to_dict("records")[:10], 1):
        q = "6" if r.get("brief_bucket") == "lead_time / turning" else "5"
        if r.get("brief_bucket") == "healthy vs stressed cash shape":
            q = "4–5"
        lines.append(
            f"{i}. `{r['feature']}` ({_sign_char(int(r['sign']))}) — {r.get('gloss', '')}. "
            f"Brief Q{q}: {r.get('brief_bucket')}."
        )
    flip = [
        r
        for r in top.to_dict("records")
        if int(r.get("sign") or 0) and int(r.get("sign_univ") or 0) and int(r["sign"]) != int(r["sign_univ"])
    ]
    lines += ["", "### Notes on signs", ""]
    lines.append(
        "- Sign is Spearman(feature, TreeSHAP) on the train sample. Univariate AUROC is a check."
    )
    if flip:
        lines.append(
            "- Sign disagreement (report SHAP): "
            + "; ".join(
                f"`{r['feature']}` SHAP {_sign_char(int(r['sign']))} vs univ "
                f"{_sign_char(int(r['sign_univ']))} {r['univ_auroc']:.3f}"
                for r in flip
            )
            + "."
        )
    else:
        lines.append("- No SHAP vs univariate sign flips in the top 15.")
    lines += [
        "",
        "## Family share of mean |SHAP|",
        "",
        "| family | sum mean\\|SHAP\\| |",
        "| --- | ---: |",
    ]
    for fam_name, val in fam.items():
        lines.append(f"| {fam_name} | {_fmt(val)} |")
    lines += [
        "",
        "## Brief-question share of mean |SHAP|",
        "",
        "| bucket | sum mean\\|SHAP\\| |",
        "| --- | ---: |",
    ]
    for bname, val in buck.items():
        lines.append(f"| {bname} | {_fmt(val)} |")
    lines += [
        "",
        "## Size proxy and D-concentration leak",
        "",
        payload["leak_block"],
        "",
        "## Judge paragraph — questions 5–6",
        "",
        payload["judge"],
        "",
        f"Plots: `{OUT_BAR.name}`, `{OUT_BEE.name}`. Table: `{OUT_CSV.name}`.",
        "",
        "Do not treat holdout AUROC 0.680 as confirmation. Quote train CV (0.663) plus this SHAP.",
        "",
        "## Re-run",
        "",
        "```bash",
        "/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.explain_y7",
        "```",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def run() -> dict:
    panel, feat_cols, tr, _ho, x_source, y_source = _prepare_panel()
    n_train = int(tr.sum())
    n_pos = int((panel.loc[tr, Y_COL] == 1).sum())
    base_rate = float(panel.loc[tr, Y_COL].mean())
    n_train_months = int(train_mask(panel["company_id"]).sum())
    ytr = panel.loc[tr, Y_COL].astype(float)
    size_auc = auroc(ytr, np.log1p(pd.to_numeric(panel.loc[tr, "a_in3"], errors="coerce").abs()))

    final, cv_rows, cv_auroc, n_trees = _fit_cv_and_final(panel, feat_cols, tr)
    X_s, _y_s, sample_idx = _sample_train(panel, tr, feat_cols)
    assert_no_holdout(panel.loc[sample_idx, "company_id"])

    gain = np.asarray(final.booster_.feature_importance(importance_type="gain"), dtype=float)
    names = list(final.booster_.feature_name())
    if names != feat_cols and set(names) == set(feat_cols):
        gain = np.array([gain[names.index(c)] for c in feat_cols], dtype=float)
    elif len(gain) != len(feat_cols):
        gain = np.resize(gain, len(feat_cols))

    shap_arr, shap_note = _try_shap(final, X_s)
    print(f"SHAP: {shap_note}")
    if shap_arr is None:
        raise RuntimeError(f"TreeExplainer required for this wave: {shap_note}")

    mean_abs = np.nanmean(np.abs(shap_arr), axis=0)
    mean_shap = np.nanmean(shap_arr, axis=0)
    shap_sign = []
    for j, c in enumerate(feat_cols):
        rho = _spearman(X_s[c], pd.Series(shap_arr[:, j], index=X_s.index))
        if np.isfinite(rho) and abs(rho) >= 1e-12:
            shap_sign.append(1 if rho > 0 else -1)
        else:
            shap_sign.append(0)
    importance_source = f"mean |TreeSHAP| ({shap_note})"
    plot_note = _write_plots(shap_arr, X_s)
    print(f"plots: {plot_note} -> {OUT_BAR.name}, {OUT_BEE.name}")

    rows = []
    for i, col in enumerate(feat_cols):
        leak = _leak_row(panel, tr, col)
        sign = int(shap_sign[i]) if shap_sign[i] else int(leak["sign_univ"])
        rec = {
            **leak,
            "gain": float(gain[i]) if i < len(gain) else float("nan"),
            "mean_abs_shap": float(mean_abs[i]),
            "mean_shap": float(mean_shap[i]),
            "sign": sign,
            "sign_shap": int(shap_sign[i]),
        }
        rec["flags"] = _flags(rec)
        rows.append(rec)
    tab = pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    tab["rank"] = np.arange(1, len(tab) + 1)
    csv_cols = ["feature", "mean_abs_shap", "mean_shap", "family", "brief_bucket", "rank"]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    tab[csv_cols].to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV}")

    fam_share = tab.groupby("family", sort=False)["mean_abs_shap"].sum().sort_values(ascending=False)
    bucket_share = tab.groupby("brief_bucket", sort=False)["mean_abs_shap"].sum().sort_values(ascending=False)

    top15 = tab.head(TOP_N)
    copies = tab.loc[
        tab["rho_d_top1"].abs().ge(CONC_COPY_CUT) | tab["rho_d_hhi"].abs().ge(CONC_COPY_CUT)
    ]
    leak_lines = [
        f"Y7 vs `log1p(a_in3)` AUROC on train labeled = {size_auc:.3f} "
        "(acceptance gate is < 0.60; the label is not a size clone).",
        f"X families **A+B+C+E+F+G+H**, never D (invoice HHI / top-1 are rebuilt inside Y7). "
        f"n_x={len(feat_cols)} including lags {list(LAGS)}.",
    ]
    if "d_cust_top1" in panel.columns:
        worst = tab.reindex(tab["rho_d_top1"].abs().sort_values(ascending=False).index).iloc[0]
        leak_lines.append(
            f"Worst |ρ| vs `d_cust_top1` on train labeled (diagnostic, D is not X): "
            f"`{worst['feature']}` ρ={worst['rho_d_top1']:.3f}."
        )
    if len(copies):
        leak_lines.append(
            "Near-copies of family-D concentration (|ρ|≥0.80): "
            + ", ".join(
                f"`{r.feature}` top1={r.rho_d_top1:.3f} hhi={r.rho_d_hhi:.3f}"
                for r in copies.head(8).itertuples()
            )
            + "."
        )
    else:
        leak_lines.append(
            f"No allowed X column reaches |ρ|≥{CONC_COPY_CUT:.2f} vs `d_cust_top1` or "
            "`d_cust_hhi`. Family E (volume, DSO, overdue) is allowed and is not a "
            "rebuilt HHI."
        )
    leak = {
        "size_auroc_y": float(size_auc),
        "n_d_copies": int(len(copies)),
        "cv_auroc": float(cv_auroc),
    }
    judge = _judge(top15, fam_share, bucket_share, leak)
    top10 = [
        {
            "feature": r["feature"],
            "sign": int(r["sign"]),
            "mean_abs_shap": float(r["mean_abs_shap"]),
            "family": r["family"],
            "brief_bucket": r["brief_bucket"],
            "gloss": r["gloss"],
        }
        for r in top15.to_dict("records")[:10]
    ]

    payload = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "y": Y_COL,
        "n_train": n_train,
        "n_pos": n_pos,
        "base_rate": base_rate,
        "n_train_months": n_train_months,
        "n_sample": int(len(X_s)),
        "cv_auroc": cv_auroc,
        "n_trees": n_trees,
        "n_x": len(feat_cols),
        "x_source": x_source,
        "y_source": y_source,
        "importance_source": importance_source,
        "shap_note": shap_note,
        "top15": top15.to_dict("records"),
        "top10": top10,
        "family_share": {k: float(v) for k, v in fam_share.items()},
        "bucket_share": {k: float(v) for k, v in bucket_share.items()},
        "judge": judge,
        "leak": leak,
        "leak_block": "\n".join(leak_lines),
        "cv_folds": cv_rows,
    }
    _write_md(payload)
    print("\nTOP 10 (sign + = higher feature -> higher P(top1 lost))")
    for i, r in enumerate(top10, 1):
        print(f"  {i:2d}. {_sign_char(r['sign'])} {r['feature']:28s}  |SHAP|={r['mean_abs_shap']:.5f}  {r['family']}  {r['brief_bucket']}")
    print("\nFAMILY SHARE")
    print(fam_share.to_string())
    print("\nBUCKET SHARE")
    print(bucket_share.to_string())
    print("\nJUDGE")
    print(judge)
    print(
        json.dumps(
            {
                "top10": top10,
                "family_share": payload["family_share"],
                "cv_auroc": cv_auroc,
                "published_cv": PUBLISHED_CV,
                "shap": shap_note,
                "paths": {
                    "script": "analysis/models/explain_y7.py",
                    "md": str(OUT_MD),
                    "csv": str(OUT_CSV),
                    "bar": str(OUT_BAR),
                    "beeswarm": str(OUT_BEE),
                },
            },
            default=str,
        )
    )
    return payload


if __name__ == "__main__":
    run()
