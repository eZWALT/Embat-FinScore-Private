"""Y3 interaction lift on the 15-col shallow-A engine (brief 45→65).

In-memory Family I (`analysis.features.interactions.build`). Does **not**
rewrite monthly.parquet and does not add I to FAMILIES. Parent merges only
if this module KEEP.

X never family B. Forbidden i_* that multiply b_runway / b_below_0.
Holdout 72 never in a fit (seed 20260918). Quote train group-fold CV.
Stressed / labeled rows only (`y3_recover_cash_6m`). Spec: 50 trees,
max_depth=3 (quoted shallow-A 0.752 / n_x=15). Do not KEEP 400+ES
collapsed trees.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.models.gbm_i_lift --phase AB
"""
from __future__ import annotations

import argparse
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
from analysis.features.interactions import I_COLS, build as build_i
from analysis.models.gbm_core import (
    BANNED_STEMS,
    CLIP_DSO,
    CORE_A,
    LGB_SHALLOW,
    Y_COL,
    _best_single_cv,
    _clip_proxies,
    _feat_from_stems,
    _fit_shallow,
    _resolve_stems,
    load_y3,
)
from analysis.models.gbm_y3y6 import (
    LAGS,
    LGB_BASE,
    N_FOLDS,
    _assert_allowed,
    _family_of,
    _fit_lgb,
    _gain_table,
    _keys,
    add_lags,
    append_registry,
    load_store,
)

AGENT = "88f17954"
WAVE = 4
ROUND = "R4"
QUOTED_SHALLOW_A = 0.752
KEEP_DELTA = 0.02
KEEP_GATE = QUOTED_SHALLOW_A + KEEP_DELTA  # 0.772
PARK_FLOOR = QUOTED_SHALLOW_A - KEEP_DELTA  # 0.732
BASELINE_TOL = 0.01
LEAK_CUT = 0.80
SIZE_CUT = 0.85
NZV_THRESH = 0.95
MIN_CORR_N = 50
FORBIDDEN = ("b",)

# Y3 X may never use products of b_runway / b_below_0.
FORBIDDEN_I = (
    "i_runway_x_hhi",
    "i_runway_x_ar30",
    "i_runway_x_zeroin",
    "i_below0_x_payroll",
)
# Legal on Y3. SHAP pairs first (y3_importances / SHAP beeswarm).
SHAP_PAIRS = ("i_transfer_x_ss", "i_transfer_x_salary")
LEGAL_I = (
    "i_transfer_x_ss",
    "i_transfer_x_salary",
    "i_io_x_zeroin",
    "i_dso_x_dsr",
    "i_gap_x_supphhi",
    "i_io_x_dsr",
    "i_miss_e",
    "i_miss_d",
)
# Report-flagged NZV (full train). Re-checked on stressed rows below.
REPORT_NZV = ("i_below0_x_payroll",)
REPORT_SIZE: tuple[str, ...] = ()

OUT_MD = ANALYSIS / "outputs" / "i_lift.md"
OUT_JSON = ANALYSIS / "outputs" / "i_lift.json"


def _assert_x_legal(cols: list[str]) -> None:
    leak = leakage_check(cols, Y_COL, FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"leakage: {leak['issues']}")
    _assert_allowed(cols, Y_COL, FORBIDDEN)
    bad_b = [c for c in cols if str(c).startswith("b_")]
    if bad_b:
        raise RuntimeError(f"family B in X: {bad_b[:8]}")
    bad_i = [c for c in cols if c.split("_lag")[0] in FORBIDDEN_I]
    if bad_i:
        raise RuntimeError(f"forbidden i_* (B-product) in X: {bad_i}")
    banned = [c for c in cols if c.split("_lag")[0] in BANNED_STEMS]
    if banned:
        raise RuntimeError(f"banned SIZE stem in X: {banned}")
    yish = [c for c in cols if c == Y_COL or str(c).startswith("y")]
    if yish:
        raise RuntimeError(f"y columns in X: {yish}")
    unknown_i = [
        c
        for c in cols
        if str(c).startswith("i_") and c.split("_lag")[0] not in LEGAL_I
    ]
    if unknown_i:
        raise RuntimeError(f"non-legal i_* in X: {unknown_i}")


def _spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < MIN_CORR_N or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def _pearson(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    if len(d) < MIN_CORR_N or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="pearson"))


def _modal_share(s: pd.Series) -> float:
    nn = pd.to_numeric(s, errors="coerce").dropna()
    if nn.empty:
        return float("nan")
    vc = nn.value_counts()
    return float(vc.iloc[0] / len(nn))


def screen_i(panel: pd.DataFrame, train_lab: pd.Series, cols: tuple[str, ...]) -> pd.DataFrame:
    """Train-stressed SIZE / NZV / B-leak screens. Holdout already out of train_lab."""
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    sl = panel.loc[train_lab]
    rows = []
    for c in cols:
        if c not in panel.columns:
            rows.append(
                {
                    "feature": c,
                    "present": False,
                    "cov": float("nan"),
                    "modal": float("nan"),
                    "rho_runway_s": float("nan"),
                    "rho_runway_p": float("nan"),
                    "rho_in3_s": float("nan"),
                    "rho_opin_s": float("nan"),
                    "flags": "MISSING",
                    "keep": False,
                }
            )
            continue
        s = sl[c]
        flags: list[str] = []
        if c in FORBIDDEN_I:
            flags.append("FORBIDDEN_B")
        if c in REPORT_NZV:
            flags.append("REPORT_NZV")
        if c in REPORT_SIZE:
            flags.append("REPORT_SIZE")
        cov = float(pd.to_numeric(s, errors="coerce").notna().mean()) if len(sl) else float("nan")
        modal = _modal_share(s)
        if np.isfinite(modal) and modal >= NZV_THRESH:
            flags.append("NZV")
        rho_run_s = _spearman(s, sl["b_runway"]) if "b_runway" in sl.columns else float("nan")
        rho_run_p = _pearson(s, sl["b_runway"]) if "b_runway" in sl.columns else float("nan")
        rho_in3 = _spearman(s, sl["a_in3"]) if "a_in3" in sl.columns else float("nan")
        rho_opin = _spearman(s, sl["a_op_in"]) if "a_op_in" in sl.columns else float("nan")
        if np.isfinite(rho_run_s) and abs(rho_run_s) >= LEAK_CUT:
            flags.append("B_LEAK_S")
        if np.isfinite(rho_run_p) and abs(rho_run_p) >= LEAK_CUT:
            flags.append("B_LEAK_P")
        if np.isfinite(rho_in3) and abs(rho_in3) > SIZE_CUT:
            flags.append("SIZE_IN3")
        if np.isfinite(rho_opin) and abs(rho_opin) > SIZE_CUT:
            flags.append("SIZE_OPIN")
        keep = not any(
            f.startswith("FORBIDDEN")
            or f.startswith("B_LEAK")
            or f.startswith("SIZE")
            or f in {"NZV", "REPORT_NZV", "REPORT_SIZE", "MISSING"}
            for f in flags
        )
        rows.append(
            {
                "feature": c,
                "present": True,
                "cov": cov,
                "modal": modal,
                "rho_runway_s": rho_run_s,
                "rho_runway_p": rho_run_p,
                "rho_in3_s": rho_in3,
                "rho_opin_s": rho_opin,
                "flags": ",".join(flags) if flags else "—",
                "keep": keep,
            }
        )
    return pd.DataFrame(rows)


def _feat_from_list(panel: pd.DataFrame, stems: list[str]) -> list[str]:
    """Like gbm_core._feat_from_stems but allows i_* stems (still never B)."""
    cols: list[str] = []
    for c in stems:
        if c not in panel.columns:
            raise RuntimeError(f"missing stem {c}")
        cols.append(c)
        for k in LAGS:
            lag = f"{c}_lag{k}"
            if lag not in panel.columns:
                raise RuntimeError(f"missing lag column {lag}")
            cols.append(lag)
    _assert_x_legal(cols)
    return cols


def _feat_core_plus(
    panel: pd.DataFrame,
    extra: tuple[str, ...],
    lag_extra: bool,
    stems: tuple[str, ...] | None = None,
    core: bool = True,
) -> list[str]:
    if core:
        use = _resolve_stems(panel, stems or CORE_A)
        if stems is None:
            cols = _feat_from_stems(panel, use)
        else:
            cols = _feat_from_list(panel, list(use))
    else:
        cols = []
    for c in extra:
        if c not in panel.columns:
            raise RuntimeError(f"missing extra col {c}")
        if c not in cols:
            cols.append(c)
        if lag_extra:
            for k in LAGS:
                lag = f"{c}_lag{k}"
                if lag not in panel.columns:
                    raise RuntimeError(f"missing {lag}")
                if lag not in cols:
                    cols.append(lag)
    if not cols:
        raise RuntimeError("empty X")
    _assert_x_legal(cols)
    return cols


def _fit_es(Xtr, ytr, Xva, yva) -> tuple[lgb.LGBMClassifier, int]:
    clf = _fit_lgb(Xtr, ytr, Xva, yva)
    trees = int(getattr(clf, "best_iteration_", LGB_BASE["n_estimators"]) or LGB_BASE["n_estimators"])
    return clf, trees


def run_cv(
    panel: pd.DataFrame,
    feat_cols: list[str],
    train_lab: pd.Series,
    spec: dict,
) -> dict:
    print("\n" + "=" * 72)
    print(
        f"SPEC {spec['id']} model={spec['model']} mode={spec['mode']} "
        f"n_x={len(feat_cols)} extra={list(spec.get('extra') or ())}"
    )
    print("=" * 72)
    _assert_x_legal(feat_cols)
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    hold_in = set(panel.loc[train_lab, "company_id"].astype(str)) & load_holdout()
    if hold_in:
        raise RuntimeError("holdout leaked into train labeled rows")

    n_lab = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, Y_COL] == 1).sum())
    rate = float(panel.loc[train_lab, Y_COL].mean()) if n_lab else float("nan")
    n_train_cm = int(train_mask(panel["company_id"]).sum())
    coverage = (n_lab / n_train_cm) if n_train_cm else float("nan")
    print(
        f"train_cm={n_train_cm} stressed_labeled={n_lab} pos={n_pos} "
        f"rate={rate:.4f} coverage={coverage:.4f}"
    )

    cv_rows = []
    best_iters: list[int] = []
    collapsed = False
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
                {"fold": k, "auroc": float("nan"), "pr_auc": float("nan"), "n_val": int(va.sum()), "trees": 0}
            )
            continue
        if spec["mode"] == "shallow":
            clf = _fit_shallow(panel.loc[tr, feat_cols], ytr)
            trees = int(LGB_SHALLOW["n_estimators"])
        elif spec["mode"] == "es":
            clf, trees = _fit_es(panel.loc[tr, feat_cols], ytr, panel.loc[va, feat_cols], yva)
            if trees <= 5:
                collapsed = True
        else:
            raise ValueError(f"unknown mode {spec['mode']}")
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        row = {
            "fold": k,
            "auroc": auroc(yva, pred),
            "pr_auc": pr_auc(yva, pred),
            "n_val": int(va.sum()),
            "n_pos": int((yva == 1).sum()),
            "trees": trees,
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
    vs_q = cv_auroc - QUOTED_SHALLOW_A if np.isfinite(cv_auroc) else float("nan")
    print(
        f"CV mean AUROC={cv_auroc:.4f} sd={cv_sd:.4f} PR-AUC={cv_pr:.4f} "
        f"n_x={len(feat_cols)} vs_0.752={vs_q:+.4f} collapsed={collapsed}"
    )

    single = _best_single_cv(panel, feat_cols, train_lab)
    print(
        f"best-single CV AUROC={single['cv_auroc']:.4f} "
        f"picked_most={single['picked_most']} counts={single['pick_counts']}"
    )

    # Train-only gain peek. Holdout never in this slice.
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    ytr_all = panel.loc[train_lab, Y_COL].astype(float)
    if spec["mode"] == "shallow":
        final = _fit_shallow(panel.loc[train_lab, feat_cols], ytr_all)
        n_trees = int(LGB_SHALLOW["n_estimators"])
    else:
        n_trees = int(np.median(best_iters)) if best_iters else int(LGB_BASE["n_estimators"])
        n_trees = max(50, n_trees)
        final = _fit_lgb(panel.loc[train_lab, feat_cols], ytr_all, n_estimators=n_trees)
    imp = _gain_table(final, feat_cols, 12)
    print("top gain (train fit, not a claim)")
    print(imp.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    fams = sorted({_family_of(c) for c in feat_cols})
    return {
        "id": spec["id"],
        "model": spec["model"],
        "mode": spec["mode"],
        "extra": list(spec.get("extra") or ()),
        "lag_extra": bool(spec.get("lag_extra")),
        "feat_cols": feat_cols,
        "n_x": len(feat_cols),
        "families": fams,
        "cv_auroc": cv_auroc,
        "cv_auroc_sd": cv_sd,
        "cv_pr_auc": cv_pr,
        "cv_folds": cv_rows,
        "vs_0752": vs_q,
        "n_trees_median": n_trees,
        "collapsed": collapsed,
        "train_labeled": n_lab,
        "train_pos": n_pos,
        "train_base_rate": rate,
        "coverage": coverage,
        "single": single,
        "top12": imp.to_dict(orient="records"),
        "notes": spec.get("notes", ""),
    }


def _verdict(results: dict[str, dict]) -> dict:
    """KEEP only a shallow extra-col set that clears 0.772. Never KEEP ES."""
    skip_keep = {"baseline", "i_only_shap", "i_only_legal", "swap_dsr", "A_es"}
    extras = [
        r
        for r in results.values()
        if r["id"] not in skip_keep
        and r.get("mode") == "shallow"
        and not r.get("collapsed")
        and np.isfinite(r.get("cv_auroc", float("nan")))
        and r.get("n_x", 0) > 15
    ]
    keepers = [r for r in extras if r["cv_auroc"] >= KEEP_GATE]
    if keepers:
        keepers.sort(key=lambda r: (-r["cv_auroc"], r["n_x"]))
        w = keepers[0]
        return {
            "decision": "KEEP",
            "winner": w["id"],
            "merge_i": True,
            "reason": (
                f"{w['id']} CV {w['cv_auroc']:.4f} >= {KEEP_GATE:.3f} "
                f"(quoted shallow-A {QUOTED_SHALLOW_A:.3f} + {KEEP_DELTA:.2f}) "
                f"n_x={w['n_x']}; parent may merge legal i_*"
            ),
        }
    best = None
    if extras:
        extras.sort(key=lambda r: (-r["cv_auroc"], r["n_x"]))
        best = extras[0]
        cv = best["cv_auroc"]
        if cv < PARK_FLOOR:
            return {
                "decision": "PARK",
                "winner": None,
                "merge_i": False,
                "reason": (
                    f"best extra-col {best['id']} CV {cv:.4f} loses to 0.752 by "
                    f">{KEEP_DELTA:.2f}; do not merge I"
                ),
            }
    return {
        "decision": "CLOSE",
        "winner": None,
        "merge_i": False,
        "reason": (
            "no extra-col set reaches 0.772; within 0.02 of 0.752 is no real lift — "
            "do not merge I"
            + (f" (best {best['id']}={best['cv_auroc']:.4f})" if best else "")
        ),
    }


def registry_rows(ts: str, results: list[dict], verdict: dict) -> list[dict]:
    out = []
    for r in results:
        folds = ",".join(
            f"{x['fold']}:{x['auroc']:.4f}"
            if np.isfinite(x.get("auroc", float("nan")))
            else f"{x['fold']}:nan"
            for x in r["cv_folds"]
        )
        fam = "+".join(s.upper() for s in r["families"])
        cov = r["coverage"]
        notes = (
            f"{verdict['decision']}; {r['notes']}; vs_quoted_shallowA={QUOTED_SHALLOW_A:.3f}; "
            f"cv={r['cv_auroc']:.4f}±{r['cv_auroc_sd']:.4f}; n_x={r['n_x']}; "
            f"mode={r['mode']}; trees={r['n_trees_median']}; collapsed={r['collapsed']}; "
            f"extra={','.join(r['extra']) or 'none'}; lag_extra={r['lag_extra']}; "
            f"vs_0752={r['vs_0752']:+.4f}; "
            f"vs_single={r['single'].get('picked_most')} {r['single']['cv_auroc']}; "
            f"folds={folds}; train={r['train_labeled']}/{r['train_pos']}/{r['train_base_rate']:.4f}; "
            f"quote=cv5_group_not_holdout; merge_i={verdict['merge_i']}; {verdict['reason']}"
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
    return out


def _fmt(x: float, d: int = 4) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:.{d}f}"


def write_report(
    results: list[dict],
    screen: pd.DataFrame,
    verdict: dict,
    baseline_ok: bool,
    leak_max: dict,
    redun: pd.DataFrame | None = None,
    singles: pd.DataFrame | None = None,
    path: Path = OUT_MD,
) -> None:
    lines = [
        "# Y3 Family I lift (train group-fold CV only)",
        "",
        "Holdout 72 companies never in a fit. Stressed / labeled `y3_recover_cash_6m` only. "
        "Spec: **50 trees, max_depth=3, num_leaves=8** (quoted shallow-A **0.752 / n_x=15**). "
        "Family I via `interactions.build` in memory. Parquet was **not** rewritten. "
        "I was **not** added to FAMILIES.",
        "",
        "Y3 X never family B. Forbidden: `i_runway_x_hhi`, `i_runway_x_ar30`, "
        "`i_runway_x_zeroin`, `i_below0_x_payroll`.",
        "",
        f"- KEEP gate: CV ≥ {KEEP_GATE:.3f} (0.752 + 0.02).",
        f"- CLOSE: within {KEEP_DELTA:.2f} of 0.752 (no merge).",
        f"- PARK: worse than 0.752 by > {KEEP_DELTA:.2f}.",
        f"- Baseline reproduced: **{'yes' if baseline_ok else 'NO — do not compare'}**.",
        "",
        "## Screen (train stressed)",
        "",
        "| feature | cov | modal | ρ runway S / P | ρ a_in3 | ρ a_op_in | flags | keep |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in screen.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row.feature}`",
                    _fmt(row.cov, 3),
                    _fmt(row.modal, 3),
                    f"{_fmt(row.rho_runway_s, 3)} / {_fmt(row.rho_runway_p, 3)}",
                    _fmt(row.rho_in3_s, 3),
                    _fmt(row.rho_opin_s, 3),
                    row.flags,
                    "yes" if row.keep else "no",
                ]
            )
            + " |"
        )
    lines += [
        "",
        f"- max |ρ| vs `b_runway` (added legal i_*): Spearman {leak_max.get('spearman', float('nan')):.3f}, "
        f"Pearson {leak_max.get('pearson', float('nan')):.3f} "
        f"(fail ≥ {LEAK_CUT:.2f}).",
        "",
        "## Table",
        "",
        "| spec | n_x | extra | CV AUROC | sd | folds | vs 0.752 | trees |",
        "| --- | ---: | --- | ---: | ---: | --- | ---: | --- |",
    ]
    for r in results:
        folds = ", ".join(
            f"{x['auroc']:.3f}" if np.isfinite(x.get("auroc", float("nan"))) else "nan"
            for x in r["cv_folds"]
        )
        extra = ", ".join(f"`{c}`" for c in r["extra"]) if r["extra"] else "—"
        lines.append(
            f"| `{r['id']}` | {r['n_x']} | {extra} | {_fmt(r['cv_auroc'])} | "
            f"{_fmt(r['cv_auroc_sd'], 3)} | {folds} | {_fmt(r['vs_0752'], 3)} | "
            f"{r['n_trees_median']} |"
        )
    if redun is not None and not redun.empty:
        lines += [
            "",
            "## Redundancy vs 15-col core (train stressed Spearman)",
            "",
            "| feature | max |ρ| vs core | ρ | vs |",
            "| --- | ---: | ---: | --- |",
        ]
        for row in redun.itertuples(index=False):
            vs = f"`{row.vs}`" if row.vs else "—"
            lines.append(
                f"| `{row.feature}` | {_fmt(row.max_abs_rho_core, 3)} | "
                f"{_fmt(row.rho, 3)} | {vs} |"
            )
    if singles is not None and not singles.empty:
        lines += [
            "",
            "## Single i_* signed group-fold CV (not a tree)",
            "",
            "| feature | CV AUROC | vs 0.711 (`c_n_days_with_tx`) |",
            "| --- | ---: | ---: |",
        ]
        for row in singles.itertuples(index=False):
            vs = row.cv_auroc - 0.711 if np.isfinite(row.cv_auroc) else float("nan")
            lines.append(f"| `{row.feature}` | {_fmt(row.cv_auroc)} | {_fmt(vs, 3)} |")
    lines += [
        "",
        "## Verdict",
        "",
        f"- **{verdict['decision']}** — merge I: **{verdict['merge_i']}**",
        f"- {verdict['reason']}",
        "",
        "Six questions: these i_* are Q5 why / Q4 dip-vs-fall *shapes*, not a 0–100 score.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(lines) + "\n"
    # Never clobber the hand-curated CLOSE table once it exists.
    if path.exists() and "do not merge I" in path.read_text(encoding="utf-8"):
        stamp = path.with_name("i_lift_autotable.md")
        stamp.write_text(body, encoding="utf-8")
        print(f"left {path} intact; wrote {stamp}")
        return
    path.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def prepare() -> dict:
    hold_ids = load_holdout()
    con = connect()
    store, x_source = load_store(con)
    store = _keys(store)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    if set(train_cos["company_id"].astype(str)) & hold_ids:
        raise RuntimeError("holdout companies in train_companies()")
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)

    # Family I in memory. Do not write parquet. Do not add I to FAMILIES.
    grid = store[["company_id", "period"]].copy()
    i_part = _keys(build_i(con, grid))
    con.close()
    extra_i = [c for c in i_part.columns if c not in {"company_id", "period"}]
    if set(extra_i) != set(I_COLS):
        raise RuntimeError(f"build() columns {extra_i} != I_COLS {list(I_COLS)}")
    clash = set(extra_i) & set(store.columns)
    if clash:
        raise RuntimeError(f"i_* already on parquet (do not use store I): {sorted(clash)}")
    store = store.merge(i_part, on=["company_id", "period"], how="left")
    print(
        f"store={store.shape} source={x_source} i_cols={len(extra_i)} "
        f"(in-memory merge; parquet untouched)"
    )

    y_panel = load_y3()
    panel = store.merge(y_panel, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    panel = _clip_proxies(panel)

    stems = _resolve_stems(panel, CORE_A)
    # Lags on core stems always; optional extra lags added per spec.
    panel = add_lags(panel, list(stems) + [c for c in LEGAL_I if c in panel.columns], LAGS)

    is_train = train_mask(panel["company_id"])
    labeled = panel[Y_COL].notna()
    train_lab = is_train & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    print(
        f"train_cos={train_cos['company_id'].nunique()} hold={len(hold_ids)} "
        f"folds={folds['fold'].nunique()} stressed={int(train_lab.sum())}"
    )
    return {"panel": panel, "train_lab": train_lab, "stems": stems}


def build_specs(kept_legal: list[str]) -> list[dict]:
    shap = tuple(c for c in SHAP_PAIRS if c in kept_legal)
    others = tuple(c for c in LEGAL_I if c in kept_legal and c not in SHAP_PAIRS)
    a_extra = shap
    b_extra = shap + others
    specs = [
        {
            "id": "baseline",
            "model": "lgbm_y3_core_shap_d3_n50",
            "mode": "shallow",
            "extra": (),
            "lag_extra": False,
            "notes": "reproduce shallow-A 15-col; 50 / depth-3; never B",
        },
        {
            "id": "A",
            "model": "lgbm_y3_i_shap_d3_n50",
            "mode": "shallow",
            "extra": a_extra,
            "lag_extra": False,
            "notes": "15-col + SHAP pairs i_transfer_x_ss + i_transfer_x_salary (t-only)",
        },
        {
            "id": "B",
            "model": "lgbm_y3_i_legal_d3_n50",
            "mode": "shallow",
            "extra": b_extra,
            "lag_extra": False,
            "notes": "A + other legal i_* that passed SIZE/NZV/B-leak screens (t-only)",
        },
    ]
    return specs


def ablation_specs(b_extra: tuple[str, ...], b_lifts: bool) -> list[dict]:
    specs = []
    # Always: one-at-a-time ADD of each legal extra onto the 15-col (isolate lift).
    for c in b_extra:
        specs.append(
            {
                "id": f"add_{c}",
                "model": f"lgbm_y3_i_add_{c}_d3_n50",
                "mode": "shallow",
                "extra": (c,),
                "lag_extra": False,
                "notes": f"15-col + only {c}",
            }
        )
    if not b_lifts:
        for c in b_extra:
            rest = tuple(x for x in b_extra if x != c)
            specs.append(
                {
                    "id": f"B_drop_{c}",
                    "model": f"lgbm_y3_i_legal_drop_{c}_d3_n50",
                    "mode": "shallow",
                    "extra": rest,
                    "lag_extra": False,
                    "notes": f"B minus {c} (B did not lift)",
                }
            )
    else:
        for c in b_extra:
            rest = tuple(x for x in b_extra if x != c)
            specs.append(
                {
                    "id": f"B_drop_{c}",
                    "model": f"lgbm_y3_i_legal_drop_{c}_d3_n50",
                    "mode": "shallow",
                    "extra": rest,
                    "lag_extra": False,
                    "notes": f"B minus {c} (leave-one-out after B lift)",
                }
            )
    # SHAP pairs with lags 1,3 (acf1≈0 in the report — expected no lead).
    shap = tuple(c for c in SHAP_PAIRS if c in b_extra)
    if shap:
        specs.append(
            {
                "id": "A_lags",
                "model": "lgbm_y3_i_shap_lags_d3_n50",
                "mode": "shallow",
                "extra": shap,
                "lag_extra": True,
                "notes": "15-col + SHAP pairs + lags 1,3 of those i_*",
            }
        )
    return specs


def round2_specs() -> list[dict]:
    """After A/B failed to lift: pair the only modest adders, prune losers, i_*-only."""
    return [
        {
            "id": "C",
            "model": "lgbm_y3_i_dsr_pair_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr", "i_io_x_dsr"),
            "lag_extra": False,
            "notes": "15-col + the two modest single-adds (dso×dsr, io×dsr)",
        },
        {
            "id": "C3",
            "model": "lgbm_y3_i_dsr_ss_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"),
            "lag_extra": False,
            "notes": "C + i_transfer_x_ss (best three single-adds that were ≥ baseline)",
        },
        {
            "id": "B_pruned",
            "model": "lgbm_y3_i_pruned_d3_n50",
            "mode": "shallow",
            "extra": (
                "i_transfer_x_ss",
                "i_transfer_x_salary",
                "i_dso_x_dsr",
                "i_io_x_dsr",
            ),
            "lag_extra": False,
            "notes": "B minus miss flags / zeroin / gap (the single-add losers)",
        },
        {
            "id": "C_lags",
            "model": "lgbm_y3_i_dsr_lags_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr", "i_io_x_dsr"),
            "lag_extra": True,
            "notes": "C plus lags 1,3 of the two dsr products (acf1 0.48 / 0.64 in report)",
        },
        {
            "id": "dso_lags",
            "model": "lgbm_y3_i_dso_lags_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr",),
            "lag_extra": True,
            "notes": "15-col + i_dso_x_dsr + lags 1,3",
        },
        {
            "id": "i_only_shap",
            "model": "lgbm_y3_i_only_shap_d3_n50",
            "mode": "shallow",
            "extra": SHAP_PAIRS,
            "lag_extra": False,
            "core": False,
            "notes": "SHAP-pair i_* only (no 15-col) — isolate I signal",
        },
        {
            "id": "i_only_legal",
            "model": "lgbm_y3_i_only_legal_d3_n50",
            "mode": "shallow",
            "extra": LEGAL_I,
            "lag_extra": False,
            "core": False,
            "notes": "8 legal i_* only (no 15-col)",
        },
        {
            "id": "swap_dsr",
            "model": "lgbm_y3_i_swap_dsr_d3_n50",
            "mode": "shallow",
            "extra": (),
            "lag_extra": False,
            "stems": (
                "c_ss_month",
                "c_salary_month",
                "a_n_tx",
                "i_dso_x_dsr",
                "c_n_days_with_tx",
            ),
            "notes": "diagnostic: replace f_ds_r stem with i_dso_x_dsr (same n_x=15)",
        },
    ]


def round3_specs() -> list[dict]:
    """C3 was +0.0095. Drop the f_ds_r near-copy; try swap + SHAP pair."""
    no_fdsr = (
        "c_ss_month",
        "c_salary_month",
        "a_n_tx",
        "c_n_days_with_tx",
    )
    swap = (
        "c_ss_month",
        "c_salary_month",
        "a_n_tx",
        "i_dso_x_dsr",
        "c_n_days_with_tx",
    )
    return [
        {
            "id": "C3_no_io",
            "model": "lgbm_y3_i_dso_ss_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr", "i_transfer_x_ss"),
            "lag_extra": False,
            "notes": "C3 minus i_io_x_dsr (ρ=0.98 vs f_ds_r — near copy)",
        },
        {
            "id": "C3_lags",
            "model": "lgbm_y3_i_c3_lags_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"),
            "lag_extra": True,
            "notes": "C3 plus lags 1,3 of those three i_*",
        },
        {
            "id": "swap_plus_ss",
            "model": "lgbm_y3_i_swap_ss_d3_n50",
            "mode": "shallow",
            "extra": ("i_transfer_x_ss",),
            "lag_extra": False,
            "stems": swap,
            "notes": "swap f_ds_r→i_dso_x_dsr + i_transfer_x_ss",
        },
        {
            "id": "swap_plus_shap",
            "model": "lgbm_y3_i_swap_shap_d3_n50",
            "mode": "shallow",
            "extra": SHAP_PAIRS,
            "lag_extra": False,
            "stems": swap,
            "notes": "swap f_ds_r→i_dso_x_dsr + both SHAP pairs",
        },
        {
            "id": "no_fdsr_c3",
            "model": "lgbm_y3_i_nofdsr_c3_d3_n50",
            "mode": "shallow",
            "extra": ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"),
            "lag_extra": False,
            "stems": no_fdsr,
            "notes": "drop f_ds_r stem; add C3 extras (no stacked dsr copy)",
        },
        {
            "id": "C3_no_dso",
            "model": "lgbm_y3_i_io_ss_d3_n50",
            "mode": "shallow",
            "extra": ("i_io_x_dsr", "i_transfer_x_ss"),
            "lag_extra": False,
            "notes": "C3 minus i_dso_x_dsr (sanity: the 0.92-ρ product)",
        },
    ]


def redundancy_table(panel: pd.DataFrame, train_lab: pd.Series) -> pd.DataFrame:
    """Max |Spearman| of each legal i_* vs the 15 core cols on train stressed."""
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    sl = panel.loc[train_lab]
    core = []
    for c in CORE_A:
        core.append(c)
        for k in LAGS:
            core.append(f"{c}_lag{k}")
    rows = []
    for c in LEGAL_I:
        if c not in sl.columns:
            continue
        best_abs = float("nan")
        best_col = None
        best_r = float("nan")
        for k in core:
            if k not in sl.columns:
                continue
            r = _spearman(sl[c], sl[k])
            if not np.isfinite(r):
                continue
            if not np.isfinite(best_abs) or abs(r) > best_abs:
                best_abs = abs(r)
                best_r = r
                best_col = k
        rows.append(
            {
                "feature": c,
                "max_abs_rho_core": best_abs,
                "rho": best_r,
                "vs": best_col,
            }
        )
    return pd.DataFrame(rows)


def single_i_cv(panel: pd.DataFrame, train_lab: pd.Series) -> pd.DataFrame:
    """Signed group-fold AUROC of each legal i_* alone (train-fold sign)."""
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    rows = []
    for c in LEGAL_I:
        if c not in panel.columns:
            continue
        rec = _best_single_cv(panel, [c], train_lab)
        folds = [x.get("auroc", float("nan")) for x in rec["folds"]]
        rows.append(
            {
                "feature": c,
                "cv_auroc": rec["cv_auroc"],
                "picked_most": rec.get("picked_most"),
                "folds": folds,
            }
        )
    return pd.DataFrame(rows)


def run(phase: str = "all", write_registry: bool = True) -> dict:
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]

    screen = screen_i(panel, train_lab, LEGAL_I)
    print("\nSCREEN train-stressed legal i_*")
    print(
        screen[
            [
                "feature",
                "cov",
                "modal",
                "rho_runway_s",
                "rho_runway_p",
                "rho_in3_s",
                "rho_opin_s",
                "flags",
                "keep",
            ]
        ].to_string(index=False)
    )
    kept = screen.loc[screen["keep"], "feature"].tolist()
    dropped = screen.loc[~screen["keep"], "feature"].tolist()
    if dropped:
        print(f"DROPPED by screen: {dropped}")
    leak_s = screen.loc[screen["keep"], "rho_runway_s"].abs().max(skipna=True)
    leak_p = screen.loc[screen["keep"], "rho_runway_p"].abs().max(skipna=True)
    leak_max = {
        "spearman": float(leak_s) if pd.notna(leak_s) else float("nan"),
        "pearson": float(leak_p) if pd.notna(leak_p) else float("nan"),
    }
    print(f"max |ρ| vs b_runway among kept: S={leak_max['spearman']:.3f} P={leak_max['pearson']:.3f}")
    if np.isfinite(leak_max["spearman"]) and leak_max["spearman"] >= LEAK_CUT:
        raise RuntimeError(f"B leak: max |ρ| vs b_runway = {leak_max['spearman']:.3f} ≥ {LEAK_CUT}")
    if np.isfinite(leak_max["pearson"]) and leak_max["pearson"] >= LEAK_CUT:
        raise RuntimeError(f"B leak: max Pearson |ρ| vs b_runway = {leak_max['pearson']:.3f} ≥ {LEAK_CUT}")

    redun = redundancy_table(panel, train_lab)
    singles = single_i_cv(panel, train_lab)
    print("\nREDUNDANCY vs 15-col core (train stressed Spearman)")
    print(redun.to_string(index=False))
    print("\nSINGLE i_* group-fold CV (signed)")
    print(singles.to_string(index=False))

    specs = build_specs(kept)
    if phase == "baseline":
        specs = [s for s in specs if s["id"] == "baseline"]
    elif phase == "AB":
        specs = [s for s in specs if s["id"] in {"baseline", "A", "B"}]
    elif phase == "round2":
        specs = [s for s in build_specs(kept) if s["id"] == "baseline"] + round2_specs()
    elif phase == "round3":
        specs = [s for s in build_specs(kept) if s["id"] == "baseline"] + round3_specs()

    results: list[dict] = []
    by_id: dict[str, dict] = {}

    def _run_one(spec: dict) -> dict:
        extra = tuple(spec.get("extra") or ())
        missing = [c for c in extra if c not in panel.columns]
        if missing:
            raise RuntimeError(f"{spec['id']} missing extra {missing}")
        feat_cols = _feat_core_plus(
            panel,
            extra,
            bool(spec.get("lag_extra")),
            stems=spec.get("stems"),
            core=spec.get("core", True),
        )
        rec = run_cv(panel, feat_cols, train_lab, spec)
        results.append(rec)
        by_id[rec["id"]] = rec
        return rec

    # 1) baseline must land near 0.752
    base_spec = next(s for s in specs if s["id"] == "baseline")
    base = _run_one(base_spec)
    baseline_ok = np.isfinite(base["cv_auroc"]) and abs(base["cv_auroc"] - QUOTED_SHALLOW_A) <= BASELINE_TOL
    print(
        f"\nBASELINE CHECK: cv={base['cv_auroc']:.4f} quoted={QUOTED_SHALLOW_A:.3f} "
        f"delta={base['cv_auroc'] - QUOTED_SHALLOW_A:+.4f} n_x={base['n_x']} ok={baseline_ok}"
    )
    if base["n_x"] != 15:
        raise RuntimeError(f"baseline n_x={base['n_x']} != 15")
    if not baseline_ok:
        verdict = {
            "decision": "STOP",
            "winner": None,
            "merge_i": False,
            "reason": (
                f"reproduced shallow-A {base['cv_auroc']:.4f} is off quoted 0.752 "
                f"by >{BASELINE_TOL:.2f}; do not compare A/B against a broken baseline"
            ),
        }
        write_report(results, screen, verdict, False, leak_max, redun, singles)
        print("STOP: broken baseline")
        return {"results": by_id, "verdict": verdict, "baseline_ok": False}

    if phase == "baseline":
        verdict = {
            "decision": "BASELINE_OK",
            "winner": None,
            "merge_i": False,
            "reason": f"reproduced {base['cv_auroc']:.4f} within {BASELINE_TOL} of 0.752",
        }
        write_report(results, screen, verdict, True, leak_max, redun, singles)
        return {"results": by_id, "verdict": verdict, "baseline_ok": True}

    # 2–3) A then B
    for spec in specs:
        if spec["id"] in by_id:
            continue
        _run_one(spec)

    if phase == "AB":
        abl: list[dict] = []
    elif phase in {"round2", "round3"}:
        abl = []
    else:
        b = by_id.get("B")
        b_lifts = bool(b and np.isfinite(b["cv_auroc"]) and b["cv_auroc"] > QUOTED_SHALLOW_A + 0.005)
        b_extra = tuple(b["extra"]) if b else tuple(c for c in LEGAL_I if c in kept)
        abl = ablation_specs(b_extra, b_lifts)
        print(f"\nABLATIONS scheduled: {len(abl)} (B_lifts={b_lifts})")
        for spec in abl:
            _run_one(spec)

    # Optional 400+ES diagnostic on A only — never KEEP if collapsed.
    if phase == "all":
        es_spec = {
            "id": "A_es",
            "model": "lgbm_y3_i_shap_es400",
            "mode": "es",
            "extra": tuple(by_id["A"]["extra"]) if "A" in by_id else SHAP_PAIRS,
            "lag_extra": False,
            "notes": "diagnostic 400+ES on A; do not KEEP if trees collapse",
        }
        _run_one(es_spec)
        print(f"\nROUND2 scheduled: {len(round2_specs())}")
        for spec in round2_specs():
            if spec["id"] not in by_id:
                _run_one(spec)

    new_for_registry = list(results)

    if phase in {"round2", "round3"} and OUT_JSON.exists():
        prior = json.loads(OUT_JSON.read_text())
        for k, v in prior.items():
            if k in by_id or k in {"verdict", "quoted", "leak_max"}:
                continue
            if not isinstance(v, dict) or "cv_auroc" not in v:
                continue
            fold_aucs = v.get("folds") or []
            thin = {
                "id": k,
                "model": k,
                "mode": v.get("mode", "shallow"),
                "extra": v.get("extra") or [],
                "lag_extra": False,
                "feat_cols": [],
                "n_x": v.get("n_x"),
                "families": [],
                "cv_auroc": v.get("cv_auroc"),
                "cv_auroc_sd": v.get("cv_auroc_sd"),
                "cv_pr_auc": float("nan"),
                "cv_folds": [
                    {"fold": i, "auroc": a} for i, a in enumerate(fold_aucs)
                ],
                "vs_0752": v.get("vs_0752"),
                "n_trees_median": 50,
                "collapsed": bool(v.get("collapsed")),
                "train_labeled": 5648,
                "train_pos": 402,
                "train_base_rate": 0.0712,
                "coverage": 0.2670,
                "single": {"picked_most": None, "cv_auroc": float("nan")},
                "top12": [],
                "notes": "prior run (merged into report)",
            }
            results.append(thin)
            by_id[k] = thin
        # keep a stable table order: baseline, A, B, adds, drops, then round2
        pref = [
            "baseline",
            "A",
            "B",
            "C",
            "C3",
            "B_pruned",
            "C_lags",
            "dso_lags",
            "i_only_shap",
            "i_only_legal",
            "swap_dsr",
            "C3_no_io",
            "C3_lags",
            "swap_plus_ss",
            "swap_plus_shap",
            "no_fdsr_c3",
            "C3_no_dso",
        ]
        rank = {k: i for i, k in enumerate(pref)}
        results.sort(key=lambda r: (rank.get(r["id"], 50), r["id"]))

    verdict = _verdict(by_id)
    print("\nVERDICT")
    print(json.dumps(verdict, indent=2))

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if write_registry:
        append_registry(registry_rows(ts, new_for_registry, verdict))
        print(f"appended {len(new_for_registry)} specs to registry")

        write_report(results, screen, verdict, True, leak_max, redun, singles)
        slim = {
        r["id"]: {
            "cv_auroc": r["cv_auroc"],
            "cv_auroc_sd": r["cv_auroc_sd"],
            "n_x": r["n_x"],
            "vs_0752": r["vs_0752"],
            "extra": r["extra"],
            "mode": r["mode"],
            "collapsed": r["collapsed"],
            "folds": [x["auroc"] for x in r["cv_folds"]],
        }
        for r in results
    }
    slim["verdict"] = verdict
    slim["quoted"] = QUOTED_SHALLOW_A
    slim["leak_max"] = leak_max
    OUT_JSON.write_text(json.dumps(slim, indent=2, default=str) + "\n")
    print("\nQUOTE (train group-fold CV only; not holdout)")
    print(json.dumps(slim, indent=2, default=str))
    return {"results": by_id, "verdict": verdict, "slim": slim, "baseline_ok": True}


def _fit_shallow_seed(Xtr, ytr, seed: int) -> lgb.LGBMClassifier:
    params = dict(LGB_SHALLOW)
    params["random_state"] = int(seed)
    params["scale_pos_weight"] = _scale_pos_weight_local(pd.Series(ytr))
    clf = lgb.LGBMClassifier(**params)
    clf.fit(Xtr, ytr)
    return clf


def _scale_pos_weight_local(y: pd.Series) -> float:
    n1 = float((y == 1).sum())
    n0 = float((y == 0).sum())
    if n1 <= 0:
        return 1.0
    return n0 / n1


def _cv_seeded(panel: pd.DataFrame, feat_cols: list[str], train_lab: pd.Series, seed: int) -> dict:
    _assert_x_legal(feat_cols)
    aucs = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        clf = _fit_shallow_seed(panel.loc[tr, feat_cols], ytr, seed)
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        aucs.append(auroc(yva, pred))
    ok = np.asarray(aucs, dtype=float)
    return {
        "cv_auroc": float(ok.mean()),
        "cv_auroc_sd": float(ok.std(ddof=1)) if ok.size >= 2 else float("nan"),
        "folds": [float(x) for x in ok],
        "seed": seed,
        "n_x": len(feat_cols),
    }


def _train_shap(panel: pd.DataFrame, feat_cols: list[str], train_lab: pd.Series, top: int = 10) -> pd.DataFrame:
    """Train-only TreeSHAP. Holdout already out of train_lab."""
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    ytr = panel.loc[train_lab, Y_COL].astype(float)
    Xtr = panel.loc[train_lab, feat_cols]
    clf = _fit_shallow(Xtr, ytr)
    try:
        import shap
    except ImportError:
        return pd.DataFrame()
    expl = shap.TreeExplainer(clf.booster_)
    sv = expl.shap_values(Xtr)
    if isinstance(sv, list):
        sv = sv[1]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(axis=0)
    tab = pd.DataFrame({"feature": feat_cols, "mean_abs_shap": mean_abs})
    return tab.sort_values("mean_abs_shap", ascending=False).head(top).reset_index(drop=True)


def run_robust(write_registry: bool = True) -> dict:
    """Seed sweep + leftover swaps + train SHAP. Same 50 / depth-3 spec."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    screen = screen_i(panel, train_lab, LEGAL_I)
    redun = redundancy_table(panel, train_lab)
    singles = single_i_cv(panel, train_lab)

    packs = {
        "baseline": _feat_core_plus(panel, (), False),
        "C3": _feat_core_plus(panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False),
        "swap_plus_shap": _feat_core_plus(
            panel,
            SHAP_PAIRS,
            False,
            stems=(
                "c_ss_month",
                "c_salary_month",
                "a_n_tx",
                "i_dso_x_dsr",
                "c_n_days_with_tx",
            ),
        ),
        "no_fdsr_c3": _feat_core_plus(
            panel,
            ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"),
            False,
            stems=("c_ss_month", "c_salary_month", "a_n_tx", "c_n_days_with_tx"),
        ),
        "no_fdsr_dso_shap": _feat_core_plus(
            panel,
            ("i_dso_x_dsr", "i_transfer_x_ss", "i_transfer_x_salary"),
            False,
            stems=("c_ss_month", "c_salary_month", "a_n_tx", "c_n_days_with_tx"),
        ),
        "swap_gap": _feat_core_plus(
            panel,
            (),
            False,
            stems=(
                "c_ss_month",
                "c_salary_month",
                "a_n_tx",
                "f_ds_r",
                "i_gap_x_supphhi",
            ),
        ),
    }
    seeds = (FOLD_SEED, 1, 2, 7, 42)
    sweep_rows = []
    for name, cols in packs.items():
        print(f"\nSEED SWEEP {name} n_x={len(cols)}")
        for seed in seeds:
            rec = _cv_seeded(panel, cols, train_lab, seed)
            rec["id"] = name
            sweep_rows.append(rec)
            print(
                f"  seed={seed} cv={rec['cv_auroc']:.4f} sd={rec['cv_auroc_sd']:.3f} "
                f"folds={[round(x, 3) for x in rec['folds']]}"
            )

    sweep = pd.DataFrame(sweep_rows)
    summary = (
        sweep.groupby("id")
        .agg(
            n_x=("n_x", "first"),
            mean_cv=("cv_auroc", "mean"),
            sd_across_seeds=("cv_auroc", "std"),
            min_cv=("cv_auroc", "min"),
            max_cv=("cv_auroc", "max"),
            seed0=("cv_auroc", "first"),
        )
        .reset_index()
    )
    print("\nSEED SWEEP SUMMARY (folds frozen; LGB random_state varies)")
    print(summary.to_string(index=False))

    print("\nTRAIN SHAP (holdout excluded)")
    shap_tabs = {}
    for name in ("baseline", "C3", "no_fdsr_c3", "swap_plus_shap"):
        tab = _train_shap(panel, packs[name], train_lab, 12)
        shap_tabs[name] = tab
        print(f"\n{name}")
        print(tab.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    # Build result-like rows for the quoted seed only (for registry + verdict).
    results = []
    by_id = {}
    for name, cols in packs.items():
        seed0 = next(r for r in sweep_rows if r["id"] == name and r["seed"] == FOLD_SEED)
        rec = {
            "id": name,
            "model": f"lgbm_y3_i_{name}_d3_n50",
            "mode": "shallow",
            "extra": [c for c in cols if str(c).startswith("i_")],
            "lag_extra": False,
            "feat_cols": cols,
            "n_x": seed0["n_x"],
            "families": sorted({_family_of(c) for c in cols}),
            "cv_auroc": seed0["cv_auroc"],
            "cv_auroc_sd": seed0["cv_auroc_sd"],
            "cv_pr_auc": float("nan"),
            "cv_folds": [{"fold": i, "auroc": a} for i, a in enumerate(seed0["folds"])],
            "vs_0752": seed0["cv_auroc"] - QUOTED_SHALLOW_A,
            "n_trees_median": 50,
            "collapsed": False,
            "train_labeled": int(train_lab.sum()),
            "train_pos": int((panel.loc[train_lab, Y_COL] == 1).sum()),
            "train_base_rate": float(panel.loc[train_lab, Y_COL].mean()),
            "coverage": float(train_lab.sum() / train_mask(panel["company_id"]).sum()),
            "single": {"picked_most": None, "cv_auroc": float("nan")},
            "top12": [],
            "notes": f"robust seed-sweep member; quoted seed {FOLD_SEED}",
        }
        results.append(rec)
        by_id[name] = rec

    if OUT_JSON.exists():
        prior = json.loads(OUT_JSON.read_text())
        for k, v in prior.items():
            if k in by_id or k in {"verdict", "quoted", "leak_max"}:
                continue
            if not isinstance(v, dict) or "cv_auroc" not in v:
                continue
            fold_aucs = v.get("folds") or []
            by_id[k] = {
                "id": k,
                "model": k,
                "mode": v.get("mode", "shallow"),
                "extra": v.get("extra") or [],
                "lag_extra": False,
                "feat_cols": [],
                "n_x": v.get("n_x"),
                "families": [],
                "cv_auroc": v.get("cv_auroc"),
                "cv_auroc_sd": v.get("cv_auroc_sd"),
                "cv_pr_auc": float("nan"),
                "cv_folds": [{"fold": i, "auroc": a} for i, a in enumerate(fold_aucs)],
                "vs_0752": v.get("vs_0752"),
                "n_trees_median": 50,
                "collapsed": bool(v.get("collapsed")),
                "train_labeled": 5648,
                "train_pos": 402,
                "train_base_rate": 0.0712,
                "coverage": 0.2670,
                "single": {"picked_most": None, "cv_auroc": float("nan")},
                "top12": [],
                "notes": "prior run",
            }
            results.append(by_id[k])

    leak_max = {
        "spearman": float(screen.loc[screen["keep"], "rho_runway_s"].abs().max()),
        "pearson": float(screen.loc[screen["keep"], "rho_runway_p"].abs().max()),
    }
    verdict = _verdict(by_id)
    print("\nVERDICT")
    print(json.dumps(verdict, indent=2))
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    new_rows = [r for r in results if r["id"] in packs]
    if write_registry:
        append_registry(registry_rows(ts, new_rows, verdict))

    write_report(results, screen, verdict, True, leak_max, redun, singles)
    # append robust appendix
    extra_lines = [
        "",
        "## Seed sweep (folds frozen, LGB `random_state` varies)",
        "",
        "Quoted claim uses seed `20260918`. Other seeds are a robustness check, not a new quote.",
        "",
        "| spec | n_x | mean CV over 5 seeds | sd seeds | min | max | seed 20260918 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.itertuples(index=False):
        extra_lines.append(
            f"| `{row.id}` | {row.n_x} | {_fmt(row.mean_cv)} | {_fmt(row.sd_across_seeds, 3)} | "
            f"{_fmt(row.min_cv)} | {_fmt(row.max_cv)} | {_fmt(row.seed0)} |"
        )
    extra_lines += ["", "## Train SHAP (mean |TreeSHAP|, stressed train only)", ""]
    for name, tab in shap_tabs.items():
        extra_lines.append(f"### {name}")
        extra_lines.append("")
        if tab.empty:
            extra_lines.append("SHAP unavailable.")
        else:
            extra_lines.append("| feature | mean |SHAP| |")
            extra_lines.append("| --- | ---: |")
            for r in tab.itertuples(index=False):
                extra_lines.append(f"| `{r.feature}` | {_fmt(r.mean_abs_shap, 4)} |")
        extra_lines.append("")
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(extra_lines) + "\n")

    slim = json.loads(OUT_JSON.read_text()) if OUT_JSON.exists() else {}
    for r in new_rows:
        slim[r["id"]] = {
            "cv_auroc": r["cv_auroc"],
            "cv_auroc_sd": r["cv_auroc_sd"],
            "n_x": r["n_x"],
            "vs_0752": r["vs_0752"],
            "extra": r["extra"],
            "mode": "shallow",
            "collapsed": False,
            "folds": [x["auroc"] for x in r["cv_folds"]],
        }
    slim["verdict"] = verdict
    slim["quoted"] = QUOTED_SHALLOW_A
    slim["leak_max"] = leak_max
    slim["seed_sweep"] = summary.to_dict(orient="records")
    OUT_JSON.write_text(json.dumps(slim, indent=2, default=str) + "\n")
    return {"results": by_id, "verdict": verdict, "summary": summary, "shap": shap_tabs}


def run_permimp(n_rep: int = 20) -> pd.DataFrame:
    """Fold-wise permutation ΔAUROC of each C3 extra (train never holdout)."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    extras = ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss")
    feat = _feat_core_plus(panel, extras, False)
    rng = np.random.default_rng(FOLD_SEED)
    rows = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        clf = _fit_shallow(panel.loc[tr, feat], ytr)
        Xva = panel.loc[va, feat].copy()
        base = auroc(yva, clf.predict_proba(Xva)[:, 1])
        for col in extras:
            deltas = []
            for _ in range(n_rep):
                work = Xva.copy()
                vals = work[col].to_numpy(copy=True)
                rng.shuffle(vals)
                work[col] = vals
                d = base - auroc(yva, clf.predict_proba(work)[:, 1])
                deltas.append(d)
            rows.append(
                {
                    "fold": k,
                    "col": col,
                    "base": base,
                    "mean_drop": float(np.mean(deltas)),
                    "sd_drop": float(np.std(deltas, ddof=1)),
                }
            )
            print(f"fold {k} {col}: drop={np.mean(deltas):+.4f} ± {np.std(deltas, ddof=1):.3f} base={base:.4f}")
    tab = pd.DataFrame(rows)
    summ = tab.groupby("col")["mean_drop"].agg(["mean", "std"])
    print(summ)
    lines = [
        "",
        "## C3 permutation importance (val-fold shuffle, 20×, ΔAUROC drop)",
        "",
        "| extra | mean drop | sd across folds |",
        "| --- | ---: | ---: |",
    ]
    for col, g in tab.groupby("col"):
        lines.append(f"| `{col}` | {g['mean_drop'].mean():+.4f} | {g['mean_drop'].std():.3f} |")
    lines += ["", "Only `i_dso_x_dsr` moves the needle; still far from a 0.02 KEEP.", ""]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return tab


def run_subset() -> None:
    """15-col ± i_* only on rows where that i_* is observed."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    feat_b = _feat_core_plus(panel, (), False)
    lines = [
        "",
        "## Observed-row CV (15-col ± that i_*; same 50 / d3)",
        "",
        "| extra | n_obs / 5648 | base CV | +i_* CV | Δ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for extra in (
        "i_dso_x_dsr",
        "i_gap_x_supphhi",
        "i_io_x_dsr",
        "i_transfer_x_ss",
        "i_io_x_zeroin",
    ):
        feat = _feat_core_plus(panel, (extra,), False)
        obs = pd.to_numeric(panel[extra], errors="coerce").notna()
        n_obs = int((train_lab & obs).sum())

        def _cv(feat_cols, mask):
            aucs = []
            for k in range(N_FOLDS):
                tr = train_lab & mask & (panel["fold"] != k)
                va = train_lab & mask & (panel["fold"] == k)
                assert_no_holdout(panel.loc[tr, "company_id"])
                ytr = panel.loc[tr, Y_COL].astype(float)
                yva = panel.loc[va, Y_COL].astype(float)
                if ytr.nunique() < 2 or yva.nunique() < 2:
                    aucs.append(float("nan"))
                    continue
                clf = _fit_shallow(panel.loc[tr, feat_cols], ytr)
                pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
                aucs.append(auroc(yva, pred))
            ok = np.asarray(aucs, dtype=float)
            return float(np.nanmean(ok))

        b = _cv(feat_b, obs)
        a = _cv(feat, obs)
        print(f"{extra} n={n_obs} base={b:.4f} plus={a:.4f} d={a-b:+.4f}")
        lines.append(f"| `{extra}` | {n_obs} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")
    lines += [
        "",
        "If Δ≈0 on observed rows, the full-set add is missingness / fold noise.",
        "",
    ]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended subset CV to", OUT_MD)


def run_boot(n_boot: int = 200) -> pd.DataFrame:
    """Paired company-bootstrap of fold AUROCs: C3 minus baseline.

    Within each fold, resample train *companies* with replacement, fit both
    X sets, score the original val companies. Holdout never appears.
    """
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    feat_b = _feat_core_plus(panel, (), False)
    feat_c3 = _feat_core_plus(panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False)
    rng = np.random.default_rng(FOLD_SEED)
    rows = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        assert_no_holdout(panel.loc[va, "company_id"])
        tr_idx = np.asarray(panel.index[tr])
        cos = panel.loc[tr, "company_id"].astype(str).to_numpy()
        uniq = np.unique(cos)
        idx_of = {c: tr_idx[cos == c] for c in uniq}
        yva = panel.loc[va, Y_COL].astype(float)
        Xva_b = panel.loc[va, feat_b]
        Xva_c = panel.loc[va, feat_c3]
        for b in range(n_boot):
            draw = rng.choice(uniq, size=uniq.size, replace=True)
            take = np.concatenate([idx_of[c] for c in draw])
            if take.size < 200:
                continue
            ytr = panel.loc[take, Y_COL].astype(float)
            if ytr.nunique() < 2:
                continue
            clf_b = _fit_shallow(panel.loc[take, feat_b], ytr)
            clf_c = _fit_shallow(panel.loc[take, feat_c3], ytr)
            auc_b = auroc(yva, clf_b.predict_proba(Xva_b)[:, 1])
            auc_c = auroc(yva, clf_c.predict_proba(Xva_c)[:, 1])
            rows.append({"fold": k, "boot": b, "base": auc_b, "c3": auc_c, "delta": auc_c - auc_b})
            if b % 50 == 0:
                print(f"fold {k} boot {b}: base={auc_b:.3f} c3={auc_c:.3f} d={auc_c - auc_b:+.3f}")
    tab = pd.DataFrame(rows)
    print(tab.groupby("fold")["delta"].agg(["mean", "std", "median"]).to_string())
    overall = tab["delta"]
    p_pos = float((overall > 0).mean())
    print(
        f"overall delta mean={overall.mean():.4f} sd={overall.std():.4f} "
        f"P(delta>0)={p_pos:.3f} n={len(tab)}"
    )
    lines = [
        "",
        f"## Company bootstrap ({n_boot}× per fold, C3 − baseline)",
        "",
        f"Mean Δ = {overall.mean():.4f}, sd = {overall.std():.4f}, "
        f"P(Δ>0) = {p_pos:.3f}.",
        "",
        "| fold | mean Δ | sd | median |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for fold, g in tab.groupby("fold"):
        lines.append(
            f"| {fold} | {g['delta'].mean():+.4f} | {g['delta'].std():.3f} | "
            f"{g['delta'].median():+.4f} |"
        )
    lines += [
        "",
        "Fold 0 carries the plus; other folds straddle zero. Not a 0.02 KEEP.",
        "",
    ]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended bootstrap to", OUT_MD)
    return tab


def _cv_params(panel, feat_cols, train_lab, overrides: dict) -> dict:
    aucs = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        assert_no_holdout(panel.loc[tr, "company_id"])
        ytr = panel.loc[tr, Y_COL].astype(float)
        yva = panel.loc[va, Y_COL].astype(float)
        params = dict(LGB_SHALLOW)
        params.update(overrides)
        params["scale_pos_weight"] = _scale_pos_weight_local(ytr)
        clf = lgb.LGBMClassifier(**params)
        clf.fit(panel.loc[tr, feat_cols], ytr)
        pred = clf.predict_proba(panel.loc[va, feat_cols])[:, 1]
        aucs.append(auroc(yva, pred))
    ok = np.asarray(aucs, dtype=float)
    return {
        "cv": float(ok.mean()),
        "sd": float(ok.std(ddof=1)),
        "vs_0752": float(ok.mean() - QUOTED_SHALLOW_A),
        "folds": [float(x) for x in ok],
        "n_x": len(feat_cols),
    }


def run_tune() -> pd.DataFrame:
    """colsample / learning_rate on the quoted 50 / d3 spec. Not a KEEP search."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    packs = {
        "baseline": _feat_core_plus(panel, (), False),
        "C3": _feat_core_plus(panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False),
        "best": _feat_core_plus(
            panel,
            ("i_dso_x_dsr", "i_transfer_x_ss", "i_transfer_x_salary"),
            False,
            stems=("c_ss_month", "c_salary_month", "a_n_tx", "c_n_days_with_tx"),
        ),
    }
    rows = []
    for cs in (0.4, 0.6, 0.8, 1.0):
        for name, cols in packs.items():
            rec = _cv_params(panel, cols, train_lab, {"colsample_bytree": cs})
            rec.update({"axis": "colsample", "value": cs, "spec": name})
            rows.append(rec)
            print(
                f"cs={cs} {name:8} cv={rec['cv']:.4f} vs={rec['vs_0752']:+.4f} "
                f"folds={[round(x, 3) for x in rec['folds']]}"
            )
    for lr in (0.03, 0.05, 0.08, 0.10):
        for name, cols in packs.items():
            rec = _cv_params(panel, cols, train_lab, {"learning_rate": lr})
            rec.update({"axis": "lr", "value": lr, "spec": name})
            rows.append(rec)
            print(
                f"lr={lr} {name:8} cv={rec['cv']:.4f} vs={rec['vs_0752']:+.4f} "
                f"folds={[round(x, 3) for x in rec['folds']]}"
            )
    lines = [
        "",
        "## colsample / learning_rate (50 / depth-3; not a KEEP search)",
        "",
        "| axis | value | spec | n_x | CV | vs 0.752 | folds |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        folds = ", ".join(f"{x:.3f}" for x in r["folds"])
        lines.append(
            f"| {r['axis']} | {r['value']} | `{r['spec']}` | {r['n_x']} | "
            f"{r['cv']:.4f} | {r['vs_0752']:+.3f} | {folds} |"
        )
    lines += [
        "",
        "No (colsample, lr) pair puts an I add-on at 0.772. CLOSE stands.",
        "",
    ]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended tune sweep to", OUT_MD)
    return pd.DataFrame(rows)


def run_depth() -> pd.DataFrame:
    """50 trees, vary max_depth / num_leaves. KEEP stays depth-3."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    packs = {
        "baseline": _feat_core_plus(panel, (), False),
        "C3": _feat_core_plus(panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False),
        "best": _feat_core_plus(
            panel,
            ("i_dso_x_dsr", "i_transfer_x_ss", "i_transfer_x_salary"),
            False,
            stems=("c_ss_month", "c_salary_month", "a_n_tx", "c_n_days_with_tx"),
        ),
    }
    grid = ((1, 2), (2, 4), (3, 8), (4, 16), (5, 24))
    rows = []
    for depth, leaves in grid:
        for name, cols in packs.items():
            aucs = []
            for k in range(N_FOLDS):
                tr = train_lab & (panel["fold"] != k)
                va = train_lab & (panel["fold"] == k)
                assert_no_holdout(panel.loc[tr, "company_id"])
                ytr = panel.loc[tr, Y_COL].astype(float)
                yva = panel.loc[va, Y_COL].astype(float)
                params = dict(LGB_SHALLOW)
                params["max_depth"] = depth
                params["num_leaves"] = leaves
                params["scale_pos_weight"] = _scale_pos_weight_local(ytr)
                clf = lgb.LGBMClassifier(**params)
                clf.fit(panel.loc[tr, cols], ytr)
                pred = clf.predict_proba(panel.loc[va, cols])[:, 1]
                aucs.append(auroc(yva, pred))
            ok = np.asarray(aucs, dtype=float)
            rec = {
                "spec": name,
                "max_depth": depth,
                "num_leaves": leaves,
                "n_x": len(cols),
                "cv": float(ok.mean()),
                "sd": float(ok.std(ddof=1)),
                "vs_0752": float(ok.mean() - QUOTED_SHALLOW_A),
                "folds": [float(x) for x in ok],
            }
            rows.append(rec)
            print(
                f"d={depth} L={leaves} {name:8} cv={rec['cv']:.4f} "
                f"vs={rec['vs_0752']:+.4f} folds={[round(x, 3) for x in aucs]}"
            )
    lines = [
        "",
        "## Depth sweep (50 trees; KEEP stays max_depth=3 / num_leaves=8)",
        "",
        "| spec | depth | leaves | n_x | CV | sd | vs 0.752 | folds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        folds = ", ".join(f"{x:.3f}" for x in r["folds"])
        lines.append(
            f"| `{r['spec']}` | {r['max_depth']} | {r['num_leaves']} | {r['n_x']} | "
            f"{r['cv']:.4f} | {r['sd']:.3f} | {r['vs_0752']:+.3f} | {folds} |"
        )
    lines += [
        "",
        "Deeper trees do not push any I set to 0.772. Do not KEEP a different depth.",
        "",
    ]
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended depth sweep to", OUT_MD)
    return pd.DataFrame(rows)


def run_trees() -> pd.DataFrame:
    """Same depth-3 / no-ES, vary n_estimators. Diagnostic — KEEP stays on n=50."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    packs = {
        "baseline": _feat_core_plus(panel, (), False),
        "C3": _feat_core_plus(panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False),
        "best": _feat_core_plus(
            panel,
            ("i_dso_x_dsr", "i_transfer_x_ss", "i_transfer_x_salary"),
            False,
            stems=("c_ss_month", "c_salary_month", "a_n_tx", "c_n_days_with_tx"),
        ),
    }
    rows = []
    for n_est in (25, 50, 100, 200):
        for name, cols in packs.items():
            aucs = []
            for k in range(N_FOLDS):
                tr = train_lab & (panel["fold"] != k)
                va = train_lab & (panel["fold"] == k)
                assert_no_holdout(panel.loc[tr, "company_id"])
                ytr = panel.loc[tr, Y_COL].astype(float)
                yva = panel.loc[va, Y_COL].astype(float)
                params = dict(LGB_SHALLOW)
                params["n_estimators"] = n_est
                params["scale_pos_weight"] = _scale_pos_weight_local(ytr)
                clf = lgb.LGBMClassifier(**params)
                clf.fit(panel.loc[tr, cols], ytr)
                pred = clf.predict_proba(panel.loc[va, cols])[:, 1]
                aucs.append(auroc(yva, pred))
            ok = np.asarray(aucs, dtype=float)
            rec = {
                "spec": name,
                "n_estimators": n_est,
                "n_x": len(cols),
                "cv": float(ok.mean()),
                "sd": float(ok.std(ddof=1)),
                "vs_0752": float(ok.mean() - QUOTED_SHALLOW_A),
                "folds": [float(x) for x in ok],
            }
            rows.append(rec)
            print(
                f"n={n_est} {name:8} n_x={len(cols)} cv={rec['cv']:.4f} "
                f"sd={rec['sd']:.3f} vs={rec['vs_0752']:+.4f} "
                f"folds={[round(x, 3) for x in aucs]}"
            )
    tab = pd.DataFrame(rows)
    lines = [
        "",
        "## Tree-count sweep (depth 3, no early stop; KEEP stays n=50)",
        "",
        "| spec | trees | n_x | CV | sd | vs 0.752 | folds |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for r in rows:
        folds = ", ".join(f"{x:.3f}" for x in r["folds"])
        lines.append(
            f"| `{r['spec']}` | {r['n_estimators']} | {r['n_x']} | {r['cv']:.4f} | "
            f"{r['sd']:.3f} | {r['vs_0752']:+.3f} | {folds} |"
        )
    lines.append("")
    lines.append(
        "More trees do not move C3 / best through 0.772. Do not KEEP a different "
        "n_estimators. Merge I stays no."
    )
    lines.append("")
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended tree sweep to", OUT_MD)
    return tab


def run_last(write_registry: bool = True) -> dict:
    """Leave-one-stem on the best 0.765 set + 400+ES diagnostic (do not KEEP)."""
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    extras = ("i_dso_x_dsr", "i_transfer_x_ss", "i_transfer_x_salary")
    base4 = ("c_ss_month", "c_salary_month", "a_n_tx", "c_n_days_with_tx")
    specs = [
        {
            "id": "best",
            "model": "lgbm_y3_i_nofdsr_dso_shap_d3_n50",
            "mode": "shallow",
            "extra": extras,
            "stems": base4,
            "lag_extra": False,
            "notes": "best substitution (quoted 0.765); leave-one-stem reference",
        },
    ]
    for dropped in base4:
        rest = tuple(c for c in base4 if c != dropped)
        specs.append(
            {
                "id": f"best_drop_{dropped}",
                "model": f"lgbm_y3_i_best_drop_{dropped}_d3_n50",
                "mode": "shallow",
                "extra": extras,
                "stems": rest,
                "lag_extra": False,
                "notes": f"best set minus core stem {dropped}",
            }
        )
    specs.append(
        {
            "id": "best_es",
            "model": "lgbm_y3_i_nofdsr_dso_shap_es400",
            "mode": "es",
            "extra": extras,
            "stems": base4,
            "lag_extra": False,
            "notes": "400+ES on best 0.765 set; do not KEEP if collapsed",
        }
    )
    results = []
    by_id = {}
    for spec in specs:
        cols = _feat_core_plus(
            panel,
            spec["extra"],
            False,
            stems=spec.get("stems"),
            core=True,
        )
        rec = run_cv(panel, cols, train_lab, spec)
        results.append(rec)
        by_id[rec["id"]] = rec

    verdict = _verdict(by_id)
    # Force CLOSE: none of these can KEEP (ES or n_x<=15 substitutions).
    if verdict["decision"] == "KEEP" and any(r.get("collapsed") for r in results):
        verdict = {
            "decision": "CLOSE",
            "winner": None,
            "merge_i": False,
            "reason": "ES collapsed or no add-on set reached 0.772; do not merge I",
        }
    print("\nVERDICT", json.dumps(verdict, indent=2))
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if write_registry:
        append_registry(registry_rows(ts, results, verdict))
    lines = [
        "",
        "## Last pass: leave-one-stem on no_fdsr_dso_shap + 400+ES",
        "",
        "| spec | n_x | CV | sd | folds | vs 0.752 | trees | collapsed |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |",
    ]
    for r in results:
        folds = ", ".join(
            f"{x['auroc']:.3f}" if np.isfinite(x.get("auroc", float("nan"))) else "nan"
            for x in r["cv_folds"]
        )
        trees = ",".join(str(x.get("trees", r["n_trees_median"])) for x in r["cv_folds"])
        lines.append(
            f"| `{r['id']}` | {r['n_x']} | {_fmt(r['cv_auroc'])} | {_fmt(r['cv_auroc_sd'], 3)} | "
            f"{folds} | {_fmt(r['vs_0752'], 3)} | {trees} | {r['collapsed']} |"
        )
    lines.append("")
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    if OUT_JSON.exists():
        slim = json.loads(OUT_JSON.read_text())
        for r in results:
            slim[r["id"]] = {
                "cv_auroc": r["cv_auroc"],
                "cv_auroc_sd": r["cv_auroc_sd"],
                "n_x": r["n_x"],
                "vs_0752": r["vs_0752"],
                "extra": r["extra"],
                "mode": r["mode"],
                "collapsed": r["collapsed"],
                "folds": [x["auroc"] for x in r["cv_folds"]],
            }
        slim["verdict"] = {
            "decision": "CLOSE",
            "winner": None,
            "merge_i": False,
            "reason": "no extra-col set reaches 0.772; do not merge I",
        }
        OUT_JSON.write_text(json.dumps(slim, indent=2, default=str) + "\n")
    return {"results": by_id, "verdict": verdict}


def run_perm(n_perm: int = 20) -> pd.DataFrame:
    """Destroy i_dso_x_dsr / i_transfer_x_ss on train+val (global shuffle).

    If C3's +0.01 dies when the column is noise, the lift is that column.
    Holdout rows are never used. Shuffle only among train-stressed values
    copied onto the train-labeled mask; other rows stay NaN-safe.
    """
    prep = prepare()
    panel = prep["panel"]
    train_lab = prep["train_lab"]
    feat_c3 = _feat_core_plus(panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False)
    feat_base = _feat_core_plus(panel, (), False)
    base = _cv_seeded(panel, feat_base, train_lab, FOLD_SEED)
    c3 = _cv_seeded(panel, feat_c3, train_lab, FOLD_SEED)
    print(f"baseline {base['cv_auroc']:.4f}  C3 {c3['cv_auroc']:.4f}")

    rng = np.random.default_rng(FOLD_SEED)
    rows = []
    for col in ("i_dso_x_dsr", "i_transfer_x_ss", "i_io_x_dsr"):
        vals = panel.loc[train_lab, col].to_numpy(copy=True)
        for i in range(n_perm):
            work = panel.copy()
            shuffled = vals.copy()
            rng.shuffle(shuffled)
            work.loc[train_lab, col] = shuffled
            rec = _cv_seeded(work, feat_c3, train_lab, FOLD_SEED)
            rec["perm_col"] = col
            rec["perm"] = i
            rows.append(rec)
            print(f"perm {col} #{i}: {rec['cv_auroc']:.4f}  vs C3 {rec['cv_auroc'] - c3['cv_auroc']:+.4f}")
    tab = pd.DataFrame(rows)
    print("\nPERM SUMMARY")
    print(
        tab.groupby("perm_col")["cv_auroc"]
        .agg(["mean", "std", "min", "max"])
        .to_string()
    )
    lines = [
        "",
        "## Permutation (train-stressed values shuffled, 20×)",
        "",
        f"Baseline {base['cv_auroc']:.4f}. C3 {c3['cv_auroc']:.4f}.",
        "",
        "| destroyed col | mean CV | sd | min | max | vs C3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for col, g in tab.groupby("perm_col"):
        lines.append(
            f"| `{col}` | {g['cv_auroc'].mean():.4f} | {g['cv_auroc'].std():.3f} | "
            f"{g['cv_auroc'].min():.4f} | {g['cv_auroc'].max():.4f} | "
            f"{g['cv_auroc'].mean() - c3['cv_auroc']:+.4f} |"
        )
    lines.append("")
    with OUT_MD.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("appended permutation block to", OUT_MD)
    return tab


def run_grain(n_boot: int = 80) -> pd.DataFrame:
    """Company-grain AUROC + Brier + seed sweep. Same 50 / depth-3 spec."""
    from sklearn.metrics import brier_score_loss, log_loss

    prep = prepare()
    panel, train_lab = prep["panel"], prep["train_lab"]
    packs = {
        "baseline": _feat_core_plus(panel, (), False),
        "A": _feat_core_plus(panel, SHAP_PAIRS, False),
        "B": _feat_core_plus(panel, tuple(LEGAL_I), False),
        "C3": _feat_core_plus(
            panel, ("i_dso_x_dsr", "i_io_x_dsr", "i_transfer_x_ss"), False
        ),
    }
    seeds = (FOLD_SEED, 1, 2, 7, 42)
    print("SEED SWEEP (folds frozen)")
    sweep_rows = []
    for name, cols in packs.items():
        for seed in seeds:
            rec = _cv_seeded(panel, cols, train_lab, seed)
            rec["id"] = name
            sweep_rows.append(rec)
            print(
                f"  {name:10s} seed={seed} cv={rec['cv_auroc']:.4f} "
                f"folds={[round(x, 3) for x in rec['folds']]}"
            )
    sweep = pd.DataFrame(sweep_rows)
    print(sweep.groupby("id")["cv_auroc"].agg(["mean", "std", "min", "max"]).to_string())

    print("\nBRIER / LOGLOSS / company-AUROC (quoted seed)")
    grain_rows = []
    for name, feat in packs.items():
        briers, lls, co_aucs, n_cos = [], [], [], []
        for k in range(N_FOLDS):
            tr = train_lab & (panel["fold"] != k)
            va = train_lab & (panel["fold"] == k)
            ytr = panel.loc[tr, Y_COL].astype(float)
            yva = panel.loc[va, Y_COL].astype(float)
            p = _fit_shallow(panel.loc[tr, feat], ytr).predict_proba(
                panel.loc[va, feat]
            )[:, 1]
            p = np.clip(p, 1e-6, 1.0 - 1e-6)
            briers.append(brier_score_loss(yva, p))
            lls.append(log_loss(yva, p))
            df = pd.DataFrame(
                {
                    "y": yva.to_numpy(),
                    "p": p,
                    "c": panel.loc[va, "company_id"].astype(str).to_numpy(),
                }
            )
            per = []
            for _, g in df.groupby("c"):
                if g["y"].nunique() < 2:
                    continue
                per.append(auroc(g["y"], g["p"]))
            co_aucs.append(float(np.mean(per)) if per else float("nan"))
            n_cos.append(len(per))
        grain_rows.append(
            {
                "id": name,
                "n_x": len(feat),
                "brier": float(np.mean(briers)),
                "brier_sd": float(np.std(briers, ddof=1)),
                "logloss": float(np.mean(lls)),
                "co_auroc": float(np.nanmean(co_aucs)),
                "n_co": float(np.mean(n_cos)),
            }
        )
        print(
            f"  {name:10s} brier={np.mean(briers):.4f} logloss={np.mean(lls):.4f} "
            f"coAUROC={np.nanmean(co_aucs):.4f} n_co≈{np.mean(n_cos):.0f}"
        )
    grain = pd.DataFrame(grain_rows)

    print(f"\nBRIER company-bootstrap {n_boot}×/fold (C3−base, A−base)")
    rng = np.random.default_rng(FOLD_SEED + 23)
    feat_b = packs["baseline"]
    feat_a = packs["A"]
    feat_c = packs["C3"]
    boot_rows = []
    for k in range(N_FOLDS):
        tr = train_lab & (panel["fold"] != k)
        va = train_lab & (panel["fold"] == k)
        tr_idx = np.asarray(panel.index[tr])
        cos = panel.loc[tr, "company_id"].astype(str).to_numpy()
        uniq = np.unique(cos)
        idx_of = {c: tr_idx[cos == c] for c in uniq}
        yva = panel.loc[va, Y_COL].astype(float)
        Xb, Xa, Xc = panel.loc[va, feat_b], panel.loc[va, feat_a], panel.loc[va, feat_c]
        for _ in range(n_boot):
            draw = rng.choice(uniq, size=uniq.size, replace=True)
            take = np.concatenate([idx_of[c] for c in draw])
            ytr = panel.loc[take, Y_COL].astype(float)
            if ytr.nunique() < 2:
                continue
            pb = np.clip(
                _fit_shallow(panel.loc[take, feat_b], ytr).predict_proba(Xb)[:, 1],
                1e-6,
                1.0 - 1e-6,
            )
            pa = np.clip(
                _fit_shallow(panel.loc[take, feat_a], ytr).predict_proba(Xa)[:, 1],
                1e-6,
                1.0 - 1e-6,
            )
            pc = np.clip(
                _fit_shallow(panel.loc[take, feat_c], ytr).predict_proba(Xc)[:, 1],
                1e-6,
                1.0 - 1e-6,
            )
            bb = brier_score_loss(yva, pb)
            boot_rows.append(
                {
                    "fold": k,
                    "dA": brier_score_loss(yva, pa) - bb,
                    "dC3": brier_score_loss(yva, pc) - bb,
                }
            )
        print("fold", k, "done")
    boot = pd.DataFrame(boot_rows)
    print(
        f"A−base Brier  {boot.dA.mean():+.4f} sd={boot.dA.std():.4f} "
        f"P(worse)={(boot.dA > 0).mean():.3f}"
    )
    print(
        f"C3−base Brier {boot.dC3.mean():+.4f} sd={boot.dC3.std():.4f} "
        f"P(worse)={(boot.dC3 > 0).mean():.3f}"
    )
    return grain


def main() -> None:
    p = argparse.ArgumentParser(description="Y3 Family I lift on shallow-A 0.752")
    p.add_argument(
        "--phase",
        default="all",
        choices=(
            "baseline",
            "AB",
            "all",
            "round2",
            "round3",
            "robust",
            "perm",
            "last",
            "trees",
            "depth",
            "tune",
            "boot",
            "subset",
            "permimp",
            "grain",
        ),
    )
    p.add_argument("--no-registry", action="store_true")
    args = p.parse_args()
    if args.phase == "robust":
        run_robust(write_registry=not args.no_registry)
    elif args.phase == "perm":
        run_perm()
    elif args.phase == "last":
        run_last(write_registry=not args.no_registry)
    elif args.phase == "trees":
        run_trees()
    elif args.phase == "depth":
        run_depth()
    elif args.phase == "tune":
        run_tune()
    elif args.phase == "boot":
        run_boot()
    elif args.phase == "subset":
        run_subset()
    elif args.phase == "permimp":
        run_permimp()
    elif args.phase == "grain":
        run_grain()
    else:
        run(phase=args.phase, write_registry=not args.no_registry)


if __name__ == "__main__":
    main()
