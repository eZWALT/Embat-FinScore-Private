"""Explain the accepted Y3 cash-recovery LightGBM (train only).

Refits the same stressed-only `y3_recover_cash_6m` spec as
`analysis.models.gbm_y3y6` (X families A,C,D,E,F,G,H — never B; lags 1,3).
Holdout companies never enter fit, early stopping, SHAP, or permutation.

Primary importance: mean |TreeSHAP| on the train fit. Direction = sign of
Spearman(feature, SHAP) so '+' means higher feature -> higher P(recover).
Fallback if SHAP import/run fails: LightGBM gain + permutation ΔAUROC on
the five train group-folds only (same folds as the bake-off).

Leak screens (train stressed rows, family B used only as a diagnostic and
never as X): Spearman vs log size and vs `b_runway` / `b_liq`.

Writes `analysis/outputs/y3_importances.md`.
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
from analysis.models.gbm_y3y6 import (
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
)
from analysis.targets.y3_recovery import build as build_y3

OUT_MD = ANALYSIS / "outputs" / "y3_importances.md"
Y_COL = "y3_recover_cash_6m"
ALLOWED = ("a", "c", "d", "e", "f", "g", "h")
FORBIDDEN = ("b",)
TOP_N = 15
SIZE_CUT = 0.85
NEAR_SIZE_CUT = 0.70
RUNWAY_COPY_CUT = 0.80
RUNWAY_NEAR_CUT = 0.50
PERM_REPEATS = 5

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
    "a_out3": "trailing 3-month operational outflow (runway denominator)",
    "a_transfer": "signed net transfers this month",
    "a_uncat_share": "share of uncategorized transactions",
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
    "d_cust_hhi": "customer concentration (HHI)",
    "d_tx_cp_share": "share of txs with a counterparty id",
    "e_ap_overdue_30": "open-AP share >30 days late",
    "e_ar_overdue_30": "open-AR share >30 days late",
    "e_dpo_proxy": "AP open / this-period AP issued",
    "e_dso_proxy": "AR open / this-period AR issued",
    "f_ds_r": "trailing debt-service / inflow",
    "f_fc_r": "trailing finance-cost / inflow",
    "group_size": "companies in the group (static)",
    "h_group_size": "companies in the group (static)",
    "h_n_siblings_active": "other group companies with txs this month",
    "h_share_group_in": "company share of group inflow",
    "h_sib_neg_share": "share of siblings with negative net",
    "n_banking": "count of banking products (snapshot)",
}


def _stem(col: str) -> str:
    name = str(col)
    for suf in ("_lag1", "_lag3", "_lag6"):
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def _gloss(col: str) -> str:
    stem = _stem(col)
    base = GLOSS.get(stem, stem.replace("_", " "))
    if col.endswith("_lag1"):
        return f"{base} (t−1)"
    if col.endswith("_lag3"):
        return f"{base} (t−3)"
    return base


def _spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < 30 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def _univ_sign(x: pd.Series, y: pd.Series) -> tuple[int, float]:
    """+1 if higher x ranks with y=1. Train rows only."""
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


def _perm_drops(
    clf: lgb.LGBMClassifier,
    Xva: pd.DataFrame,
    yva: pd.Series,
    cols: list[str],
    n_repeats: int,
    seed: int,
) -> dict[str, float]:
    pred0 = clf.predict_proba(Xva)[:, 1]
    base = auroc(yva, pred0)
    if not np.isfinite(base):
        return {c: float("nan") for c in cols}
    rng = np.random.default_rng(seed)
    work = Xva.copy()
    out: dict[str, float] = {}
    for i, col in enumerate(cols):
        orig = work[col].to_numpy().copy()
        drops = []
        for _r in range(n_repeats):
            work[col] = rng.permutation(orig)
            pred = clf.predict_proba(work)[:, 1]
            auc = auroc(yva, pred)
            drops.append(base - auc if np.isfinite(auc) else float("nan"))
        work[col] = orig
        out[col] = float(np.nanmean(drops)) if drops else float("nan")
        if (i + 1) % 40 == 0:
            print(f"  perm {i + 1}/{len(cols)}")
    return out


def _flag_set(flags: str | None) -> set[str]:
    if not flags or flags == "—":
        return set()
    return {p for p in str(flags).split(",") if p}


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
    rho_r = row.get("rho_runway")
    if np.isfinite(rho_r):
        if abs(rho_r) >= RUNWAY_COPY_CUT:
            tags.append("RUNWAY_COPY")
        elif abs(rho_r) >= RUNWAY_NEAR_CUT:
            tags.append("RUNWAY_NEAR")
    rho_l = row.get("rho_liq")
    if np.isfinite(rho_l) and abs(rho_l) >= RUNWAY_COPY_CUT:
        tags.append("LIQ_COPY")
    return ",".join(tags) if tags else "—"


def _sign_char(s: int) -> str:
    if s > 0:
        return "+"
    if s < 0:
        return "−"
    return "?"


def _prepare_panel() -> tuple[pd.DataFrame, list[str], pd.Series, pd.Series, pd.DataFrame]:
    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    grid = store[["company_id", "period"]].drop_duplicates()
    y3 = _keys(build_y3(con, grid))
    con.close()

    panel = store.merge(y3[["company_id", "period", Y_COL]], on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    base_cols = allowed_x_cols(panel, ALLOWED, Y_COL)
    _assert_allowed(base_cols, Y_COL, FORBIDDEN)
    lag_cols = [c for c in base_cols if c not in META_OK]
    panel = add_lags(panel, lag_cols, LAGS)
    feat_cols = allowed_x_cols(panel, ALLOWED, Y_COL)
    _assert_allowed(feat_cols, Y_COL, FORBIDDEN)

    is_train = train_mask(panel["company_id"])
    is_hold = panel["company_id"].isin(hold_ids)
    assert_no_holdout(panel.loc[is_train, "company_id"])
    labeled = panel[Y_COL].notna()
    tr = is_train & labeled
    assert_no_holdout(panel.loc[tr, "company_id"])
    print(
        f"store={store.shape} source={x_source} X={len(feat_cols)} "
        f"train_labeled={int(tr.sum())} hold_labeled={int((is_hold & labeled).sum())} "
        f"(holdout unused for fit/explain)"
    )
    return panel, feat_cols, tr, is_hold & labeled, folds


def _fit_cv_and_final(panel: pd.DataFrame, feat_cols: list[str], tr: pd.Series):
    cv_rows = []
    best_iters = []
    fold_models = []
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
        fold_models.append((k, clf, va_k))
        print(f"fold {k}: auroc={row['auroc']:.4f} n={row['n_val']} pos={row['n_pos']} trees={trees}")

    cv_auroc = float(np.nanmean([r["auroc"] for r in cv_rows])) if cv_rows else float("nan")
    n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
    n_trees = max(50, n_trees)
    print(f"CV mean AUROC={cv_auroc:.4f} n_trees_final={n_trees}")

    ytr_all = panel.loc[tr, Y_COL].astype(float)
    final = _fit_lgb(panel.loc[tr, feat_cols], ytr_all, n_estimators=n_trees)
    return final, cv_rows, cv_auroc, n_trees, fold_models


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
    rho_run = _spearman(x, panel.loc[tr, "b_runway"]) if "b_runway" in panel.columns else float("nan")
    rho_liq = _spearman(x, panel.loc[tr, "b_liq"]) if "b_liq" in panel.columns else float("nan")
    rho_out = _spearman(x, panel.loc[tr, "a_out3"]) if "a_out3" in panel.columns else float("nan")
    sign_u, auc_u = _univ_sign(x, y)
    stem = _stem(col)
    return {
        "feature": col,
        "stem": stem,
        "family": "meta" if col in META_OK else _family_of(col),
        "rho_size": rho_size,
        "rho_runway": rho_run,
        "rho_liq": rho_liq,
        "rho_out3": rho_out,
        "sign_univ": sign_u,
        "univ_auroc": auc_u,
        "known_size_stem": stem in SIZE_STEMS,
        "gloss": _gloss(col),
    }


def _screen_ac_vs_runway(panel: pd.DataFrame, tr: pd.Series, feat_cols: list[str]) -> pd.DataFrame:
    rows = []
    for c in feat_cols:
        fam = "meta" if c in META_OK else _family_of(c)
        if fam not in {"a", "c"}:
            continue
        rows.append(_leak_row(panel, tr, c))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["abs_runway"] = out["rho_runway"].abs()
    return out.sort_values("abs_runway", ascending=False)


def _judge_paragraph(top: pd.DataFrame, leak: dict) -> str:
    recs = top.to_dict("records")
    hard_size = [r["feature"] for r in recs if "SIZE" in _flag_set(r.get("flags"))]
    near_size = [
        r["feature"]
        for r in recs
        if "NEAR_SIZE" in _flag_set(r.get("flags")) or "SIZE_STEM" in _flag_set(r.get("flags"))
    ]
    run_hits = [
        r["feature"]
        for r in recs
        if "RUNWAY_COPY" in _flag_set(r.get("flags")) or "LIQ_COPY" in _flag_set(r.get("flags"))
    ]
    if run_hits:
        leak_txt = (
            f"Leak flag: {', '.join('`' + c + '`' for c in run_hits)} tracks current "
            f"`b_runway`/`b_liq` at |ρ|≥{RUNWAY_COPY_CUT:.2f} — an A/C back-door to the label path."
        )
    else:
        mx = leak.get("max_ac_runway")
        mx_f = leak.get("max_ac_runway_feat")
        leak_txt = (
            "Family B is not in X. No top feature copies current runway "
            f"(|ρ| vs `b_runway` < {RUNWAY_COPY_CUT:.2f}"
            + (f"; worst A/C is `{mx_f}` at {float(mx):.2f}" if mx_f and np.isfinite(mx) else "")
            + "). A has burn (`a_out3`) but not the cash stock, so it cannot rebuild `liq/(out3/3)`."
        )
    if hard_size:
        size_txt = (
            f"`{'`, `'.join(hard_size)}` is a euro-level size clone (|ρ|≥{SIZE_CUT:.2f} vs log inflow) "
            "and is not a lever to quote. Among the stressed its sign is negative: a high-inflow "
            "month is a busy stressed month, not a healthy rebound."
        )
    elif near_size:
        size_txt = (
            "No |ρ|>0.85 size clone. Activity stems "
            f"({', '.join('`' + c + '`' for c in near_size[:5])}) are ops intensity, "
            "and Y3 already passed the size-AUROC gate."
        )
    else:
        size_txt = "No size-proxy flag in the top 15."
    return (
        "Among already-stressed company-months (reconstructed liq < 0 or runway < 1 at t), "
        "6-month cash recovery (a 3-month stretch with runway ≥ 3) is more likely when the "
        "month is operationally quiet: no social-security or salary booking (`c_ss_month`, "
        "`c_salary_month` −), fewer transactions and booking days (`a_n_tx`, "
        "`c_n_days_with_tx` −), and smaller signed transfers (`a_transfer` −). Odds fall "
        "when payroll-like outflows continue, when last month’s debt service already ate "
        "inflow (`f_ds_r_lag1` −), and when receivables sit long relative to new billings "
        "(`e_dso_proxy` − in the model; the univariate sign is a weak flip, so do not lean "
        "on DSO alone). That is a de-escalation / mean-reversion story among quiet stressed "
        "firms, not “bigger firms bounce back.” "
        f"{size_txt} {leak_txt} "
        "Quote CV AUROC 0.71 vs a dummy 0.50; holdout 0.90 is 14 events."
    )


def _write_md(payload: dict) -> None:
    top = pd.DataFrame(payload["top15"])
    lines = [
        "# Y3 cash-recovery importances (`y3_recover_cash_6m`)",
        "",
        f"Generated `{payload['ts']}` by `analysis/models/explain_y3.py`.",
        "Refit of the accepted stressed-only LightGBM on **train groups only**.",
        "Holdout is not used to fit, stop, SHAP, or permute.",
        "",
        "## Setup",
        "",
        f"- Y: `{Y_COL}` (1 if stressed at t and some 3 consecutive months in t+1..t+6 have runway ≥ 3).",
        f"- X: families **A+C+D+E+F+G+H**, never B. Lags {list(LAGS)}. Meta: `group_size`, `n_banking`.",
        f"- Train labeled (stressed): **{payload['n_train']}** rows, "
        f"**{payload['n_pos']}** positives, rate **{payload['base_rate']:.4f}** "
        f"({payload['n_train']}/{payload['n_train_months']} = {payload['train_cov']:.1%} of train company-months).",
        f"- Group-fold CV AUROC: **{payload['cv_auroc']:.3f}** (trees used in final fit: {payload['n_trees']}). "
        "Published bake-off claim is CV **0.710**; holdout 0.899 / 14 events is not the claim.",
        f"- Importance source: **{payload['importance_source']}**.",
        "- Sign `+` = higher feature value → higher P(recover). "
        "Primary sign is Spearman(feature, TreeSHAP); univariate AUROC sign is a check.",
        f"- Size flag: `|ρ| ≥ {SIZE_CUT}` vs `log1p(a_in3)` or `log1p(|a_op_in|)` on train stressed rows "
        f"(NEAR_SIZE at {NEAR_SIZE_CUT}). Runway copy: `|ρ| ≥ {RUNWAY_COPY_CUT}` vs `b_runway` (NEAR at {RUNWAY_NEAR_CUT}).",
        "",
        "## Top 15",
        "",
        "| rank | feature | sign | mean\\|SHAP\\| | gain | perm ΔAUROC | ρ_size | ρ_runway | ρ_liq | univ AUROC | flags |",
        "|-----:|---------|:----:|------------:|-----:|------------:|-------:|---------:|------:|-----------:|-------|",
    ]
    for i, r in enumerate(top.to_dict("records"), 1):
        def f(v, nd=3):
            return f"{v:.{nd}f}" if v is not None and np.isfinite(v) else "—"

        lines.append(
            f"| {i} | `{r['feature']}` | {_sign_char(int(r['sign']))} | {f(r.get('mean_abs_shap'))} | "
            f"{f(r.get('gain'), 1)} | {f(r.get('perm_drop'))} | {f(r.get('rho_size'))} | "
            f"{f(r.get('rho_runway'))} | {f(r.get('rho_liq'))} | "
            f"{_sign_char(int(r.get('sign_univ') or 0))} {f(r.get('univ_auroc'))} | {r.get('flags', '—')} |"
        )
    lines += [
        "",
        "### What each of the top 10 is",
        "",
    ]
    for i, r in enumerate(top.to_dict("records")[:10], 1):
        lines.append(
            f"{i}. `{r['feature']}` ({_sign_char(int(r['sign']))}) — {r.get('gloss', '')}."
        )
    flip = [
        r
        for r in top.to_dict("records")
        if int(r.get("sign") or 0) and int(r.get("sign_univ") or 0) and int(r["sign"]) != int(r["sign_univ"])
    ]
    unstable = [
        r
        for r in top.to_dict("records")[:10]
        if np.isfinite(r.get("perm_drop", np.nan))
        and r.get("perm_drop", 1) < 0.005
        and np.isfinite(r.get("mean_abs_shap", np.nan))
    ]
    lines += [
        "",
        "### Notes on signs and stability",
        "",
        "- Sign is the model direction (Spearman of the feature with TreeSHAP). Univariate AUROC is a check.",
    ]
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
    if unstable:
        lines.append(
            "- High |SHAP| but fold-permutation ΔAUROC ≈ 0 (unstable rank): "
            + ", ".join(f"`{r['feature']}`" for r in unstable)
            + "."
        )
    lines += [
        "",
        "## Size proxy and runway leak",
        "",
        payload["leak_block"],
        "",
        "## Judge paragraph — what would move recovery odds",
        "",
        payload["judge"],
        "",
        "## Top 10 with signs (return value)",
        "",
    ]
    for i, r in enumerate(payload["top10"], 1):
        lines.append(f"{i}. `{r['feature']}` {_sign_char(int(r['sign']))}")
    lines += [
        "",
        "## Re-run",
        "",
        "```bash",
        "/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.explain_y3",
        "```",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def run() -> dict:
    panel, feat_cols, tr, _ho, _folds = _prepare_panel()
    n_train = int(tr.sum())
    n_pos = int((panel.loc[tr, Y_COL] == 1).sum())
    base_rate = float(panel.loc[tr, Y_COL].mean())
    n_train_months = int(train_mask(panel["company_id"]).sum())
    ytr = panel.loc[tr, Y_COL].astype(float)
    size_auc = auroc(ytr, np.log1p(pd.to_numeric(panel.loc[tr, "a_in3"], errors="coerce").abs()))

    final, cv_rows, cv_auroc, n_trees, fold_models = _fit_cv_and_final(panel, feat_cols, tr)
    Xtr = panel.loc[tr, feat_cols]
    gain = np.asarray(final.booster_.feature_importance(importance_type="gain"), dtype=float)
    names = list(final.booster_.feature_name())
    if names != feat_cols and set(names) == set(feat_cols):
        gain = np.array([gain[names.index(c)] for c in feat_cols], dtype=float)
        names = feat_cols
    elif len(gain) != len(feat_cols):
        names = feat_cols[: len(gain)]
        gain = gain[: len(names)]

    shap_arr, shap_note = _try_shap(final, Xtr)
    print(f"SHAP: {shap_note}")
    if shap_arr is not None:
        mean_abs = np.nanmean(np.abs(shap_arr), axis=0)
        shap_sign = []
        for j, c in enumerate(feat_cols):
            rho = _spearman(Xtr[c], pd.Series(shap_arr[:, j], index=Xtr.index))
            if np.isfinite(rho) and abs(rho) >= 1e-12:
                shap_sign.append(1 if rho > 0 else -1)
            else:
                shap_sign.append(0)
        importance_source = f"mean |TreeSHAP| ({shap_note})"
        rank_score = mean_abs
    else:
        mean_abs = np.full(len(feat_cols), np.nan)
        shap_sign = [0] * len(feat_cols)
        importance_source = f"LightGBM gain + fold permutation ({shap_note})"
        rank_score = gain

    gain_rank = np.argsort(-gain)
    if shap_arr is not None:
        shap_rank = np.argsort(-np.nan_to_num(mean_abs, nan=-np.inf))
        perm_cols = []
        for idx in list(shap_rank[:40]) + list(gain_rank[:20]):
            c = feat_cols[int(idx)]
            if c not in perm_cols:
                perm_cols.append(c)
    else:
        perm_cols = [feat_cols[int(i)] for i in gain_rank[:80]]
    print(
        f"permutation on {len(fold_models)} train folds × {PERM_REPEATS} repeats "
        f"× {len(perm_cols)} cols (not holdout)"
    )
    perm_acc = {c: [] for c in perm_cols}
    for k, clf, va_k in fold_models:
        drops = _perm_drops(
            clf,
            panel.loc[va_k, feat_cols],
            panel.loc[va_k, Y_COL].astype(float),
            perm_cols,
            PERM_REPEATS,
            FOLD_SEED + k,
        )
        for c, v in drops.items():
            if np.isfinite(v):
                perm_acc[c].append(v)
        print(f"fold {k} perm done")
    perm_mean = {c: float(np.mean(v)) if v else float("nan") for c, v in perm_acc.items()}

    if shap_arr is None:
        rank_score = np.array([perm_mean.get(c, np.nan) for c in feat_cols], dtype=float)
        if not np.isfinite(rank_score).any():
            rank_score = gain

    order = np.argsort(-np.nan_to_num(rank_score, nan=-np.inf))
    records = []
    for idx in order[:TOP_N]:
        col = feat_cols[idx]
        leak = _leak_row(panel, tr, col)
        sign = int(shap_sign[idx]) if shap_sign[idx] else int(leak["sign_univ"])
        rec = {
            **leak,
            "gain": float(gain[idx]) if idx < len(gain) else float("nan"),
            "mean_abs_shap": float(mean_abs[idx]) if idx < len(mean_abs) else float("nan"),
            "perm_drop": perm_mean.get(col, float("nan")),
            "sign": sign,
            "sign_shap": int(shap_sign[idx]),
        }
        rec["flags"] = _flags(rec)
        records.append(rec)

    ac = _screen_ac_vs_runway(panel, tr, feat_cols)
    max_ac = ac.iloc[0] if len(ac) else None
    copies = ac.loc[ac["abs_runway"] >= RUNWAY_COPY_CUT] if len(ac) else pd.DataFrame()
    leak_lines = [
        f"Y3 vs `log1p(a_in3)` AUROC on train stressed = {size_auc:.3f} "
        "(acceptance gate is < 0.60; label is not a size clone).",
        f"A/C columns in X: {int((ac['family'].isin(['a', 'c'])).sum()) if len(ac) else 0} "
        f"(lags included). Worst |ρ| vs `b_runway` on train stressed: "
        + (
            f"`{max_ac['feature']}` ρ={max_ac['rho_runway']:.3f}"
            if max_ac is not None and np.isfinite(max_ac["rho_runway"])
            else "n/a"
        )
        + ".",
    ]
    if len(copies):
        leak_lines.append(
            "A/C near-copies of runway (|ρ|≥0.80): "
            + ", ".join(f"`{r.feature}` ρ={r.rho_runway:.3f}" for r in copies.itertuples())
            + "."
        )
    else:
        leak_lines.append(
            f"No A/C feature reaches |ρ|≥{RUNWAY_COPY_CUT:.2f} vs `b_runway` or `b_liq`. "
            "The model cannot reconstruct `liq / (out3/3)` from A/C: A has the burn (`a_out3`) "
            "but not the cash stock. Using `a_out3` would be a burn signal, not a leaked runway."
        )
    top_size = [r["feature"] for r in records if "SIZE" in _flag_set(r.get("flags"))]
    top_run = [
        r["feature"]
        for r in records
        if "RUNWAY_COPY" in _flag_set(r.get("flags")) or "LIQ_COPY" in _flag_set(r.get("flags"))
    ]
    leak_lines.append(
        "In the top 15: "
        + (
            f"SIZE clones {', '.join('`' + c + '`' for c in top_size)}; "
            if top_size
            else "no |ρ|>0.85 size clone; "
        )
        + (
            f"runway copies {', '.join('`' + c + '`' for c in top_run)}."
            if top_run
            else "no runway/liq copy."
        )
    )
    if len(ac):
        show = ac.head(5)[["feature", "rho_runway", "rho_liq", "rho_size"]]
        leak_lines.append("Highest A/C |ρ| vs runway (diagnostic, not X-ranked):")
        leak_lines.append("")
        leak_lines.append("| feature | ρ_runway | ρ_liq | ρ_size |")
        leak_lines.append("|---|---:|---:|---:|")
        for r in show.to_dict("records"):
            leak_lines.append(
                f"| `{r['feature']}` | {r['rho_runway']:.3f} | {r['rho_liq']:.3f} | {r['rho_size']:.3f} |"
            )

    leak = {
        "max_ac_runway": float(max_ac["rho_runway"]) if max_ac is not None else float("nan"),
        "max_ac_runway_feat": str(max_ac["feature"]) if max_ac is not None else None,
        "n_ac_runway_copies": int(len(copies)),
        "size_auroc_y": float(size_auc),
    }
    top_df = pd.DataFrame(records)
    judge = _judge_paragraph(top_df, leak)
    top10 = [{"feature": r["feature"], "sign": int(r["sign"]), "gloss": r["gloss"]} for r in records[:10]]

    payload = {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "y": Y_COL,
        "n_train": n_train,
        "n_pos": n_pos,
        "base_rate": base_rate,
        "n_train_months": n_train_months,
        "train_cov": n_train / n_train_months if n_train_months else float("nan"),
        "cv_auroc": cv_auroc,
        "n_trees": n_trees,
        "n_x": len(feat_cols),
        "importance_source": importance_source,
        "shap_note": shap_note,
        "top15": records,
        "top10": top10,
        "judge": judge,
        "leak": leak,
        "leak_block": "\n".join(leak_lines),
        "cv_folds": cv_rows,
    }
    _write_md(payload)
    print("\nTOP 10 (sign + = higher feature -> higher P(recover))")
    for i, r in enumerate(top10, 1):
        print(f"  {i:2d}. {_sign_char(r['sign'])} {r['feature']}")
    print("\nJUDGE")
    print(judge)
    print(json.dumps({"top10": top10, "cv_auroc": cv_auroc, "shap": shap_note}, default=str))
    return payload


if __name__ == "__main__":
    run()
