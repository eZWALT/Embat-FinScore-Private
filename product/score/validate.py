"""Validation of the score against the accepted outcomes, group-fold, with a size baseline. Also weight sensitivity.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m product.score.validate [--holdout]

Design
- Outcomes: the eight accepted in data/feature_store/y_acceptance.csv that the plan keeps (never y3_recover_cash_6m),
  built by analysis.targets from the same feature store.
- For each outcome the score is recomputed without the items from the feature families the label is built from
  (spec.OUTCOME_FORBIDDEN_FAMILIES); the other categories are re-weighted as for a company missing them.
- 5-fold CV by group_id (analysis.evaluate.protocol.group_folds, train companies only). In each fold the percentile
  reference is fitted on the other folds' companies and used to score the held-out fold. Weights and caps are fixed.
- Risk = 100 - score against a bad event (y = 1). AUROC pooled over out-of-fold company-months, 95% interval by
  resampling groups (500 draws). Size baseline: log1p of the trailing 3-month operating inflow, best of the two
  directions (the most flattering fixed reading), same rows, same bootstrap draws (paired difference).
- The holdout is looked at once, only with --holdout, at the end.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analysis.evaluate.protocol import FOLD_SEED, group_folds
from analysis.features.common import connect
from . import spec
from .fit import DEFAULT_STORE, fit_reference, load_holdout
from .items import compute_items
from .explain import trajectory
from .score import score_frame

OUTCOMES = list(spec.OUTCOME_FORBIDDEN_FAMILIES)
OUTCOME_MODULES = ["y2_stress", "y4_debt", "y5_payment", "y7_concentration", "y9_fees"]
N_BOOT = 500
BOOT_SEED = 20260919
OUT_MD = Path(__file__).resolve().parent / "validation.md"


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    n1 = int((y == 1).sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def load_outcomes(store: pd.DataFrame) -> pd.DataFrame:
    con = connect()
    grid = store[["company_id", "period"]].copy()
    out = grid.copy()
    for m in OUTCOME_MODULES:
        y = importlib.import_module("analysis.targets." + m).build(con, grid.copy())
        keep = [c for c in y.columns if c in OUTCOMES]
        out = out.merge(y[["company_id", "period", *keep]], on=["company_id", "period"], how="left")
    con.close()
    out["period"] = pd.to_datetime(out["period"])
    return out


def cv_scores(items: pd.DataFrame, folds: pd.DataFrame, drop_families: frozenset, weights: dict | None = None) -> pd.DataFrame:
    """Out-of-fold score and category scores for train companies (NaN elsewhere)."""
    cols = ["score"] + [f"cat_{c}" for c in spec.CATEGORY_WEIGHTS]
    out = pd.DataFrame(np.nan, index=items.index, columns=cols)
    for k in sorted(folds["fold"].unique()):
        fit_ids = set(folds.loc[folds["fold"] != k, "company_id"])
        ref = fit_reference(items, fit_ids)
        test = items["company_id"].isin(set(folds.loc[folds["fold"] == k, "company_id"]))
        sc = score_frame(items[test], ref, drop_families=drop_families, weights=weights)
        out.loc[test, "score"] = sc["res"]["score"]
        for c in sc["cats"].columns:
            out.loc[test, f"cat_{c}"] = sc["cats"][c]
    return out


def _boot_indices(groups: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    order = np.argsort(groups, kind="stable")
    g_sorted = groups[order]
    uniq, start = np.unique(g_sorted, return_index=True)
    ends = np.append(start[1:], len(g_sorted))
    idx = [order[s:e] for s, e in zip(start, ends)]
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([idx[i] for i in pick])


def evaluate_outcome(d: pd.DataFrame, ycol: str, risk: np.ndarray, base_risk: np.ndarray, seed: int = BOOT_SEED) -> dict:
    """AUROC of `risk` vs the outcome with a group bootstrap, against a baseline (best of its two directions).

    d: rows with the outcome, group_id, period. Same rows and same bootstrap draws for the score and the baseline.
    """
    y = d[ycol].to_numpy(dtype=float)
    month_rank = pd.Series(risk, index=d.index).groupby(d["period"]).rank(pct=True).to_numpy()
    a_score, a_base = auroc(y, risk), auroc(y, base_risk)
    a_base2 = max(a_base, 1 - a_base)
    a_month = auroc(y, month_rank)
    boots = {"score": [], "base": [], "diff": [], "month": []}
    for ix in _boot_indices(d["group_id"].to_numpy(), N_BOOT, seed):
        yb = y[ix]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        s_, z = auroc(yb, risk[ix]), auroc(yb, base_risk[ix])
        z2 = max(z, 1 - z)
        boots["score"].append(s_); boots["base"].append(z2); boots["diff"].append(s_ - z2)
        boots["month"].append(auroc(yb, month_rank[ix]))
    ci = {k: np.percentile(v, [2.5, 97.5]) for k, v in boots.items()}
    return {
        "outcome": ycol, "rows": len(d), "companies": d["company_id"].nunique(), "groups": d["group_id"].nunique(),
        "positives": int(y.sum()), "pos_companies": d.loc[d[ycol] == 1, "company_id"].nunique(), "base_rate": y.mean(),
        "auroc": a_score, "auroc_lo": ci["score"][0], "auroc_hi": ci["score"][1],
        "size_auroc": a_base2, "size_lo": ci["base"][0], "size_hi": ci["base"][1], "size_direction_auroc": a_base,
        "diff": a_score - a_base2, "diff_lo": ci["diff"][0], "diff_hi": ci["diff"][1],
        "auroc_within_month": a_month, "month_lo": ci["month"][0], "month_hi": ci["month"][1],
    }


def run_cv(store: pd.DataFrame, items: pd.DataFrame, ys: pd.DataFrame, companies: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Level (score vs outcome), change (3-month fall of the score vs outcome) and per-category tables."""
    folds = group_folds(companies, n=5, seed=FOLD_SEED)
    base = items[["company_id", "period"]].merge(store[["company_id", "period", "a_in3"]], on=["company_id", "period"], how="left")
    base["size"] = np.log1p(base["a_in3"].clip(lower=0))
    base["size_d3"] = base["size"] - base.groupby("company_id", sort=False)["size"].shift(3)
    base = base.merge(folds[["company_id", "group_id"]], on="company_id", how="left")
    rows, chg_rows, cat_rows = [], [], []
    cache: dict[frozenset, pd.DataFrame] = {}
    for ycol, fam in spec.OUTCOME_FORBIDDEN_FAMILIES.items():
        key = frozenset(fam)
        if key not in cache:
            oof = cv_scores(items, folds, key)
            oof["score_d3"] = oof["score"] - oof.groupby(items["company_id"], sort=False)["score"].shift(3)
            cache[key] = oof
        d = pd.concat([base, cache[key]], axis=1).merge(ys[["company_id", "period", ycol]], on=["company_id", "period"], how="left")
        d = d[d["group_id"].notna() & d[ycol].notna() & d["score"].notna() & d["size"].notna()]
        r = evaluate_outcome(d, ycol, 100.0 - d["score"].to_numpy(), -d["size"].to_numpy())
        r["dropped_families"] = ",".join(sorted(fam))
        rows.append(r)
        dc = d[d["score_d3"].notna() & d["size_d3"].notna()]
        c = evaluate_outcome(dc, ycol, -dc["score_d3"].to_numpy(), -dc["size_d3"].to_numpy())
        c["dropped_families"] = ",".join(sorted(fam))
        c["level_auroc_same_rows"] = auroc(dc[ycol].to_numpy(dtype=float), 100.0 - dc["score"].to_numpy())
        chg_rows.append(c)
        for cat in spec.CATEGORY_WEIGHTS:
            dk = d[d[f"cat_{cat}"].notna()]
            if len(dk) and dk[ycol].sum() > 0:
                cat_rows.append({"outcome": ycol, "category": cat, "rows": len(dk),
                                 "auroc": auroc(dk[ycol].to_numpy(dtype=float), 100.0 - dk[f"cat_{cat}"].to_numpy())})
    return pd.DataFrame(rows), pd.DataFrame(chg_rows), pd.DataFrame(cat_rows)


def trajectory_table(items: pd.DataFrame, ys: pd.DataFrame, companies: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """Outcome rate by trajectory state (state computed on out-of-fold scores), relative to the 'stable' rate."""
    folds = group_folds(companies, n=5, seed=FOLD_SEED)
    grp = folds.set_index("company_id")["group_id"]
    rows = []
    cache: dict[frozenset, pd.DataFrame] = {}
    for ycol, fam in spec.OUTCOME_FORBIDDEN_FAMILIES.items():
        key = frozenset(fam)
        if key not in cache:
            oof = cv_scores(items, folds, key)
            t = pd.concat([items[["company_id", "period"]], oof[["score"]]], axis=1)
            t["guard"] = np.where(items["dark_level"] == 2, "dark", "")
            cache[key] = trajectory(t)
        d = cache[key].merge(ys[["company_id", "period", ycol]], on=["company_id", "period"], how="left")
        d = d[d["company_id"].isin(grp.index) & d[ycol].notna() & d["score"].notna()].copy()
        d["group_id"] = d["company_id"].map(grp)
        y = d[ycol].to_numpy(dtype=float)
        st = d["trajectory"].to_numpy()
        base = y[st == "stable"].mean()
        rec = {"outcome": ycol, "stable_rate": base, "stable_rows": int((st == "stable").sum())}
        for state in ("improving", "dip", "deteriorating"):
            m = st == state
            rec[f"{state}_rows"] = int(m.sum())
            rec[f"{state}_rr"] = y[m].mean() / base if m.sum() and base > 0 else np.nan
            rr = []
            for ix in _boot_indices(d["group_id"].to_numpy(), n_boot, BOOT_SEED):
                ms, mb = st[ix] == state, st[ix] == "stable"
                if ms.sum() and mb.sum() and y[ix][mb].mean() > 0:
                    rr.append(y[ix][ms].mean() / y[ix][mb].mean())
            rec[f"{state}_rr_lo"], rec[f"{state}_rr_hi"] = np.percentile(rr, [2.5, 97.5]) if rr else (np.nan, np.nan)
        rows.append(rec)
    return pd.DataFrame(rows)


def sensitivity(items: pd.DataFrame, ys: pd.DataFrame, companies: pd.DataFrame, n_draws: int = 60, seed: int = 7) -> pd.DataFrame:
    """Alternative fixed weightings. Reference fitted on all train companies (it never sees an outcome)."""
    train_ids = set(companies["company_id"])
    ref = fit_reference(items, train_ids)
    tr = items["company_id"].isin(train_ids).to_numpy()
    ys_ = ys.set_index(["company_id", "period"])
    keys = pd.MultiIndex.from_frame(items.loc[tr, ["company_id", "period"]])

    def per_outcome(weights: dict | None):
        aucs, scores = {}, None
        for fam in ({"b"}, {"f"}, {"e"}, {"d"}):
            sc = score_frame(items[tr], ref, drop_families=frozenset(fam), weights=weights)["res"]["score"]
            for ycol, f in spec.OUTCOME_FORBIDDEN_FAMILIES.items():
                if f == fam:
                    y = ys_[ycol].reindex(keys).to_numpy()
                    ok = np.isfinite(y) & sc.notna().to_numpy()
                    aucs[ycol] = auroc(y[ok], 100.0 - sc.to_numpy()[ok])
        full = score_frame(items[tr], ref, weights=weights)["res"]["score"]
        return aucs, full

    base_auc, base_full = per_outcome(None)
    variants = {
        "nominal FICO 35/30/15/10/10 (no shrink)": dict(spec.CATEGORY_WEIGHTS),
        "equal 20 each": {c: 20.0 for c in spec.CATEGORY_WEIGHTS},
    }
    for c in spec.CATEGORY_WEIGHTS:
        w = dict(spec.EFFECTIVE_WEIGHTS)
        w[c] = 0.0
        variants[f"drop {c}"] = w
    rng = np.random.default_rng(seed)
    rows = []
    for name, w in variants.items():
        a, full = per_outcome(w)
        ok = full.notna() & base_full.notna()
        rows.append({"variant": name, "mean_auroc": np.mean(list(a.values())), "mean_auroc_base": np.mean(list(base_auc.values())),
                     "max_abs_auroc_change": max(abs(a[k] - base_auc[k]) for k in a),
                     "spearman_vs_base": spearmanr(full[ok], base_full[ok])[0]})
    draws = []
    for _ in range(n_draws):
        w = {c: v * rng.uniform(0.7, 1.3) for c, v in spec.EFFECTIVE_WEIGHTS.items()}
        a, full = per_outcome(w)
        ok = full.notna() & base_full.notna()
        draws.append((np.mean(list(a.values())), max(abs(a[k] - base_auc[k]) for k in a), spearmanr(full[ok], base_full[ok])[0]))
    dr = np.array(draws)
    rows.append({"variant": f"{n_draws} random weightings, each category weight x U(0.7, 1.3)", "mean_auroc": dr[:, 0].mean(),
                 "mean_auroc_base": np.mean(list(base_auc.values())), "max_abs_auroc_change": dr[:, 1].max(),
                 "spearman_vs_base": dr[:, 2].min()})
    return pd.DataFrame(rows)


def holdout_once(items: pd.DataFrame, ys: pd.DataFrame, companies: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Reference fitted on all train companies; holdout scored once. Not used for any choice."""
    ref = fit_reference(items, set(companies["company_id"]))
    hold = load_holdout()
    h = items["company_id"].isin(hold).to_numpy()
    rows = []
    ys_ = ys.set_index(["company_id", "period"])
    keys = pd.MultiIndex.from_frame(items.loc[h, ["company_id", "period"]])
    size = np.log1p(store.set_index(["company_id", "period"])["a_in3"].clip(lower=0)).reindex(keys).to_numpy()
    for ycol, fam in spec.OUTCOME_FORBIDDEN_FAMILIES.items():
        sc = score_frame(items[h], ref, drop_families=frozenset(fam))["res"]["score"].to_numpy()
        y = ys_[ycol].reindex(keys).to_numpy()
        ok = np.isfinite(y) & np.isfinite(sc) & np.isfinite(size)
        s = auroc(y[ok], 100.0 - sc[ok]); z = auroc(y[ok], -size[ok])
        rows.append({"outcome": ycol, "rows": int(ok.sum()), "positives": int(y[ok].sum()), "auroc": s, "size_auroc": max(z, 1 - z)})
    return pd.DataFrame(rows)


def fmt_ci(v, lo, hi) -> str:
    return f"{v:.3f} [{lo:.3f}, {hi:.3f}]"


def _table(cv: pd.DataFrame, extra_col: str | None = None) -> list[str]:
    head = "| outcome | dropped | rows / positives / pos. companies | base | score AUROC | baseline AUROC | score - baseline | score, ranked within month |"
    if extra_col:
        head += f" {extra_col} |"
    L = [head, "|" + "---|" * (head.count("|") - 1)]
    for r in cv.itertuples():
        line = (f"| {r.outcome} | {r.dropped_families} | {r.rows:,} / {r.positives:,} / {r.pos_companies} | {r.base_rate:.3f} | "
                f"{fmt_ci(r.auroc, r.auroc_lo, r.auroc_hi)} | {fmt_ci(r.size_auroc, r.size_lo, r.size_hi)} | "
                f"{fmt_ci(r.diff, r.diff_lo, r.diff_hi)} | {fmt_ci(r.auroc_within_month, r.month_lo, r.month_hi)} |")
        if extra_col:
            line += f" {r.level_auroc_same_rows:.3f} |"
        L.append(line)
    return L


def to_markdown(cv: pd.DataFrame, chg: pd.DataFrame, cats: pd.DataFrame, sens: pd.DataFrame, hold: pd.DataFrame | None,
                traj: pd.DataFrame | None = None) -> str:
    L = ["# Score validation (auto-generated by `python -m product.score.validate`)", "",
         "Score vs the accepted outcomes, 5-fold group CV on train companies, percentile reference refit in every fold. "
         "Risk = 100 - score against a bad event. Intervals: 95% group bootstrap (500 draws). For each outcome the items from the "
         "families its label is built from are removed from the score. Weights, caps and thresholds are fixed a priori; only the "
         "percentile reference is fitted.", "",
         "## Level: is a low score followed by the bad event?", "",
         "Baseline: log1p trailing 3-month operating inflow (company size), best of the two directions.", ""] + _table(cv)
    L += ["", "## Change: is a 3-month fall of the score followed by the bad event?", "",
          "Risk = -(score_t - score_{t-3}). Baseline: -(change in log1p 3-month inflow) over the same 3 months, best of two directions "
          "(a naive revenue-decline monitor). Last column: the level score's AUROC on the same rows.", ""] + _table(chg, "level score, same rows")
    L += ["", "## Per-category AUROC (level, category score only, same rows as the level table)", "",
          "| outcome | " + " | ".join(spec.CATEGORY_WEIGHTS) + " |", "|---|" + "---|" * len(spec.CATEGORY_WEIGHTS)]
    for ycol, g in cats.groupby("outcome", sort=False):
        m = g.set_index("category")["auroc"]
        L.append(f"| {ycol} | " + " | ".join(f"{m[c]:.3f}" if c in m else "n/a" for c in spec.CATEGORY_WEIGHTS) + " |")
    if traj is not None:
        L += ["", "## Trajectory state vs outcome (states from out-of-fold scores)", "",
              "Outcome rate in the state divided by the rate in 'stable' (relative risk, 95% group bootstrap, 300 draws).", "",
              "| outcome | stable rate | improving RR (rows) | dip RR (rows) | deteriorating RR (rows) |", "|---|---|---|---|---|"]
        for r in traj.itertuples():
            cells = [f"{getattr(r, st + '_rr'):.2f} [{getattr(r, st + '_rr_lo'):.2f}, {getattr(r, st + '_rr_hi'):.2f}] ({getattr(r, st + '_rows'):,})"
                     for st in ("improving", "dip", "deteriorating")]
            L.append(f"| {r.outcome} | {r.stable_rate:.3f} | " + " | ".join(cells) + " |")
    if len(sens):
        L += ["", "## Weight sensitivity", "",
              "Mean AUROC over the eight outcomes (level, forbidden families dropped per outcome), reference fitted on all train companies.", "",
              "| variant | mean AUROC | base mean AUROC | max abs change on one outcome | Spearman vs base score (min for random draws) |", "|---|---|---|---|---|"]
        for r in sens.itertuples():
            L.append(f"| {r.variant} | {r.mean_auroc:.3f} | {r.mean_auroc_base:.3f} | {r.max_abs_auroc_change:.3f} | {r.spearman_vs_base:.3f} |")
    if hold is not None:
        L += ["", "## Holdout, one look", "", "Reference fitted on all train companies; holdout scored once, not used for any choice. LOW_POWER.", "",
              "| outcome | rows / positives | score AUROC | size AUROC |", "|---|---|---|---|"]
        for r in hold.itertuples():
            L.append(f"| {r.outcome} | {r.rows} / {r.positives} | {r.auroc:.3f} | {r.size_auroc:.3f} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, default=DEFAULT_STORE)
    ap.add_argument("--holdout", action="store_true", help="also score the holdout, once")
    ap.add_argument("--skip-sensitivity", action="store_true")
    args = ap.parse_args()
    store = pd.read_parquet(args.store)
    store["period"] = pd.to_datetime(store["period"])
    items = compute_items(store)
    ys = load_outcomes(store)
    con = connect()
    comp = con.execute("SELECT company_id, group_id FROM companies").df()
    con.close()
    hold = load_holdout()
    companies = comp[~comp["company_id"].isin(hold)].reset_index(drop=True)
    cv, chg, cats = run_cv(store, items, ys, companies)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(cv[["outcome", "dropped_families", "rows", "positives", "base_rate", "auroc", "auroc_lo", "auroc_hi", "size_auroc", "diff", "diff_lo", "diff_hi", "auroc_within_month"]].round(3).to_string(index=False))
    print(chg[["outcome", "rows", "positives", "auroc", "auroc_lo", "auroc_hi", "size_auroc", "diff", "diff_lo", "diff_hi", "level_auroc_same_rows"]].round(3).to_string(index=False))
    sens = pd.DataFrame(columns=["variant", "mean_auroc", "mean_auroc_base", "max_abs_auroc_change", "spearman_vs_base"]) \
        if args.skip_sensitivity else sensitivity(items, ys, companies)
    print(sens.round(3).to_string(index=False))
    traj = trajectory_table(items, ys, companies)
    print(traj.round(2).to_string(index=False))
    h = holdout_once(items, ys, companies, store) if args.holdout else None
    if h is not None:
        print(h.round(3).to_string(index=False))
    OUT_MD.write_text(to_markdown(cv, chg, cats, sens, h, traj), encoding="utf-8")
    cv.to_csv(OUT_MD.with_name("validation_cv.csv"), index=False)
    chg.to_csv(OUT_MD.with_name("validation_change.csv"), index=False)
    print("wrote", OUT_MD)
    return 0


if __name__ == "__main__":
    sys.exit(main())
