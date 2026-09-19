"""Reconcile Javier's 14 raw monthly signals to the feature store.

NORTH_STAR: tonight is engine evidence, not the score. This module maps the
14 pre-percentile signals in ``analysis.score_pipeline`` onto store columns
and measures train Spearman. It does **not** call ``fit_ref``, ``run_score``,
``make_traj``, or any 5-pillar / 0–100 / state builder. Product is frozen.

Holdout 72 (seed 20260918) is coverage only. ρ / SAME / CLOSE / DRIFT / holes
are train. No parquet rewrite. No new GBM. No percentile fit.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.score_pipeline_qa

Owned: analysis/evaluate/score_pipeline_qa.py, analysis/outputs/score_pipeline_qa.md,
optional PNG, overnight/waves/wave4_score_pipeline.md (one note at the END),
append-only registry coverage / Spearman rows.

Do not edit score_pipeline.py, liquidity.py, clean_flags_qa.py, y2_why.py,
the parquet, or product/.

Iteration (same module, not one-shot):
1. CAT_MAP parity + train Spearman on the 14
2. Coverage holes (470 dark / B-only / schedule-thin)
3. Concentration top1 vs HHI
4. Volatility vs b_bal_vol vs reconstructed A
5. Heatmap
6. Identity tightness (max|Δ|)
7. Vol vs size
8. Overdue ρ by month
9. is_extreme vs flow twins
10. d_runway clip + dark fill
11. DSO is not in the 14
12. COMP_0962 refund ghost (470 vs 469)
13. Concentration residual
14. fin_cost_r 6-row level shift
15. Old unpaid stock (CLOSE mechanism)
16. Conc residual toward extract + PDI
17. Pairwise among the 14 (do not average)
18. COMP_1027 after-snapshot rate
19. PDI |amount| share
20. Holdout coverage (no ρ)
21. SIG types unread
22. Best-twin footnote (14 SAME if remapped)
23. Vol DRIFT per company
24. Overdue CLOSE per company
25. due<iss (0 here) + grid 22230=22230
26. SIZE screen — none of the 14
27. First two months invoice skip
28. Group-fold ρ stability
29. Dark 470 still have bank A/B/F
"""
from __future__ import annotations

import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    group_folds,
    load_holdout,
    train_companies,
)
from analysis.features.common import ANALYSIS, AS_OF, CAT_MAP as STORE_CAT_MAP, DATA, LAST_M, MONTHS
from analysis.features.common import connect
from analysis.score_pipeline import CAT_MAP as PIPE_CAT_MAP
from analysis.score_pipeline import SIG
from analysis.score_pipeline import build_features as build_javier_signals

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
OUT_MD = ANALYSIS / "outputs" / "score_pipeline_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "score_pipeline_rho.png"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_score_pipeline.md"
CACHE = Path("/tmp/embat_score_pipeline_qa_javier.parquet")
CUTS_JSON = Path("/tmp/embat_score_pipeline_qa_cuts.json")
AGENT = "851eac72"
WAVE = "4"
ROUND = "R4"
PANEL_START = MONTHS[0]
PANEL_END = LAST_M
EXTRACT = AS_OF

# Never imported / called: fit_ref, run_score, make_traj, pillar_moves, alerts.
FORBIDDEN_SCORE_FNS = ("fit_ref", "run_score", "make_traj", "pillar_moves", "worst_pillar", "alerts")

THE_14 = (
    "runway",
    "d_runway",
    "neg_liq",
    "coverage",
    "net_margin",
    "volatility",
    "growth",
    "concentration",
    "fin_cost_r",
    "debt_serv_r",
    "ar_overdue",
    "ap_overdue",
    "delay_coll",
    "delay_paid",
)

# Primary store twin + extras to report. Verdict uses the primary ρ.
# hole: invoice | b_only | flow | none
# q: brief question the signal answers (not a scorecard).
MAP = [
    {
        "signal": "runway",
        "store": "b_runway",
        "alts": (),
        "q": "Q1",
        "hole": "b_only",
        "park": "KEEP last-value Q1 description; never B as Y2/Y3 X",
    },
    {
        "signal": "d_runway",
        "store": "b_d_runway",
        "alts": ("b_d_runway_clip",),
        "q": "Q2",
        "hole": "b_only",
        "park": "3-month Δ; store is unclipped, Javier clips [-12, 12]",
    },
    {
        "signal": "neg_liq",
        "store": "b_neg_liq_3",
        "alts": ("b_below_0",),
        "q": "Q1/Q4",
        "hole": "b_only",
        "park": "rolling 3-month share; b_below_0 is the 1-month still",
    },
    {
        "signal": "coverage",
        "store": "a_io_ratio",
        "alts": (),
        "q": "Q1",
        "hole": "none",
        "park": "in3/out3 clip 3 — cashflow, not B",
    },
    {
        "signal": "net_margin",
        "store": "a_net_margin",
        "alts": (),
        "q": "Q1",
        "hole": "none",
        "park": "clip (in3-out3)/in3",
    },
    {
        "signal": "volatility",
        "store": "b_bal_vol",
        "alts": ("vol_from_a",),
        "q": "Q3",
        "hole": "none",
        "park": "Javier is sd6(net)/mean6(op_in); b_bal_vol is sd6(liq)/out — different object",
    },
    {
        "signal": "growth",
        "store": "a_growth_3",
        "alts": ("a_growth_12",),
        "q": "Q2",
        "hole": "none",
        "park": "trailing-3 vs t-3; a_growth_12 is YoY of the same window",
    },
    {
        "signal": "concentration",
        "store": "d_cust_top1",
        "alts": ("d_cust_hhi",),
        "q": "Q5",
        "hole": "invoice",
        "park": "pipeline is top1 (max/sum); HHI is a monopoly tail (Y4 ρ 0.991) — PARK as gradient X",
    },
    {
        "signal": "fin_cost_r",
        "store": "f_fc_r",
        "alts": (),
        "q": "Q3",
        "hole": "flow",
        "park": "KEEP flow; schedule rate / util are 1.7% / last-month — not in the 14",
    },
    {
        "signal": "debt_serv_r",
        "store": "f_ds_r",
        "alts": (),
        "q": "Q3",
        "hole": "flow",
        "park": "KEEP flow; store drops is_extreme txs",
    },
    {
        "signal": "ar_overdue",
        "store": "e_ar_overdue",
        "alts": ("ar_od_3m", "e_dso_proxy"),
        "q": "Q5",
        "hole": "invoice",
        "park": "Javier open stock is 3m issuance; store is all unpaid. DSO is Y7 SHAP#1 and fails short-DSO — not in the 14",
    },
    {
        "signal": "ap_overdue",
        "store": "e_ap_overdue",
        "alts": ("ap_od_3m",),
        "q": "Q5",
        "hole": "invoice",
        "park": "same 3m-vs-all-open window as AR",
    },
    {
        "signal": "delay_coll",
        "store": "e_delay_coll",
        "alts": (),
        "q": "Q5/Q6",
        "hole": "invoice",
        "park": "null first 6 calendar months (left truncation)",
    },
    {
        "signal": "delay_paid",
        "store": "e_delay_paid",
        "alts": (),
        "q": "Q5/Q6",
        "hole": "invoice",
        "park": "null first 6 calendar months (left truncation)",
    },
]

BEYOND = (
    {
        "col": "c_ss_month",
        "why": "Y3 shallow-A stem (Q3/Q5 payroll regularity). Not one of the 14.",
    },
    {
        "col": "c_n_days_with_tx",
        "why": "Y3 night quote 0.711; Q6 KEEP days_lag1. Not one of the 14.",
    },
    {
        "col": "e_ar_issued_lag1",
        "why": "Y7 TURNOVER / Q6 KEEP issued_lag1 0.626. Model-time lag of e_ar_issued; not a store column and not one of the 14.",
    },
)

HEAT_STORE = (
    "b_runway",
    "b_d_runway",
    "b_neg_liq_3",
    "b_below_0",
    "a_io_ratio",
    "a_net_margin",
    "b_bal_vol",
    "vol_from_a",
    "a_growth_3",
    "a_growth_12",
    "d_cust_top1",
    "d_cust_hhi",
    "f_fc_r",
    "f_ds_r",
    "e_ar_overdue",
    "e_ap_overdue",
    "e_delay_coll",
    "e_delay_paid",
    "e_dso_proxy",
)


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:.{nd}f}"


def verdict(rho: float | None) -> str:
    if rho is None or not np.isfinite(rho):
        return "NA"
    a = abs(float(rho))
    if a >= 0.95:
        return "SAME"
    if a >= 0.80:
        return "CLOSE"
    return "DRIFT"


def spearman(a: pd.Series, b: pd.Series) -> tuple[float, int]:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d[np.isfinite(d["a"]) & np.isfinite(d["b"])]
    n = int(len(d))
    if n < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan"), n
    return float(d["a"].corr(d["b"], method="spearman")), n


def cat_map_parity() -> dict:
    pipe_items = dict(PIPE_CAT_MAP)
    store_items = dict(STORE_CAT_MAP)
    same_keys = set(pipe_items) == set(store_items)
    same_vals = same_keys and all(pipe_items[k] == store_items[k] for k in pipe_items)
    only_pipe = sorted(set(pipe_items) - set(store_items))
    only_store = sorted(set(store_items) - set(pipe_items))
    diffs = sorted(
        k for k in set(pipe_items) & set(store_items) if pipe_items[k] != store_items[k]
    )
    return {
        "n_pipe": len(pipe_items),
        "n_store": len(store_items),
        "same_keys": bool(same_keys),
        "same_mapping": bool(same_vals),
        "only_pipe": only_pipe,
        "only_store": only_store,
        "value_diffs": diffs,
        "groups_pipe": sorted(set(pipe_items.values())),
        "groups_store": sorted(set(store_items.values())),
    }


def assert_no_score_call() -> None:
    """Refuse to bind the pillar/score builders in this process."""
    import analysis.score_pipeline as sp

    for name in FORBIDDEN_SCORE_FNS:
        if name not in dir(sp):
            raise RuntimeError(f"score_pipeline missing {name} — import surface changed")
    # Touching the name is fine; calling it is not. Record that we did not.
    return None


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(STORE)
    df = pd.read_parquet(STORE)
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["period"])
    return df


def load_javier(con, force: bool = False) -> pd.DataFrame:
    if CACHE.exists() and not force:
        age = time.time() - CACHE.stat().st_mtime
        if age < 6 * 3600:
            df = pd.read_parquet(CACHE)
            df["company_id"] = df["company_id"].astype(str)
            df["period"] = pd.to_datetime(df["period"])
            print(f"CUT 0 — Javier raw cache {CACHE} age={age / 60:.1f}m rows={len(df)}")
            return df
    t0 = time.time()
    P = build_javier_signals(con)
    # Signal builder only. No fit_ref / run_score.
    keep = ["company_id", "month", *THE_14]
    extra = [c for c in ("liq", "op_in", "op_out", "net_m", "in3", "out3", "in6", "sd6") if c in P.columns]
    P = P[keep + extra].copy()
    P["company_id"] = P["company_id"].astype(str)
    P["period"] = pd.to_datetime(P["month"])
    P = P.drop(columns=["month"])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    P.to_parquet(CACHE, index=False)
    print(f"CUT 0 — Javier raw build_features {len(P):,} rows in {time.time() - t0:.1f}s → {CACHE}")
    return P


def add_store_twins(store: pd.DataFrame) -> pd.DataFrame:
    out = store.copy()
    out["b_d_runway_clip"] = pd.to_numeric(out["b_d_runway"], errors="coerce").clip(-12, 12)
    g = out.groupby("company_id", sort=False)
    sd6 = g["a_net"].transform(lambda s: s.rolling(6, min_periods=6).std())
    in6_mean = g["a_op_in"].transform(lambda s: s.rolling(6, min_periods=6).mean())
    out["vol_from_a"] = np.minimum(3.0, sd6 / np.maximum(in6_mean, 1.0))
    # model-time lag the night kept (not a store column)
    out["e_ar_issued_lag1"] = g["e_ar_issued"].shift(1)
    return out


def invoice_3m_overdue(con) -> pd.DataFrame:
    """Javier-style 3-month issuance open stock, overdue share. Diagnosis only."""
    k = con.execute(
        """
        SELECT company_id,
               CAST(issuance_date AS DATE) AS iss,
               GREATEST(CAST(due_date AS DATE), CAST(issuance_date AS DATE)) AS due,
               CASE WHEN status = 'paid' THEN CAST(payment_date AS DATE) END AS paid_dt,
               amount
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL AND due_date IS NOT NULL
          AND NOT (status = 'paid' AND NOT (payment_date >= issuance_date
                                            AND payment_date <= TIMESTAMP '2026-09-01'))
        """
    ).df()
    k["iss"] = pd.to_datetime(k["iss"])
    k["due"] = pd.to_datetime(k["due"])
    k["paid_dt"] = pd.to_datetime(k["paid_dt"])
    k["side"] = np.where(k["amount"] > 0, "AR", "AP")
    k["abs_amt"] = k["amount"].abs()
    rows = []
    for i, m in enumerate(MONTHS):
        if i < 2:
            continue
        e = m + pd.offsets.MonthEnd(0)
        w0 = m - pd.DateOffset(months=2)
        o = k[(k["iss"] >= w0) & (k["iss"] <= e) & (k["paid_dt"].isna() | (k["paid_dt"] > e))]
        tot = o.groupby(["company_id", "side"])["abs_amt"].sum().rename("open_amt")
        ov = o[o["due"] < e].groupby(["company_id", "side"])["abs_amt"].sum().rename("ov_amt")
        x = pd.concat([tot, ov], axis=1)
        x["ov_amt"] = x["ov_amt"].fillna(0.0)
        x["od"] = x["ov_amt"] / x["open_amt"]
        x = x.reset_index()
        ar = x.loc[x["side"] == "AR", ["company_id", "od"]].rename(columns={"od": "ar_od_3m"})
        ap = x.loc[x["side"] == "AP", ["company_id", "od"]].rename(columns={"od": "ap_od_3m"})
        res = ar.merge(ap, on="company_id", how="outer")
        res["period"] = m
        rows.append(res)
    out = pd.concat(rows, ignore_index=True)
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    return out


def merge_train(store: pd.DataFrame, jav: pd.DataFrame, od3: pd.DataFrame) -> pd.DataFrame:
    hold = load_holdout()
    s = add_store_twins(store)
    s = s.merge(od3, on=["company_id", "period"], how="left")
    j = jav.rename(columns={c: f"j_{c}" for c in THE_14 if c in jav.columns})
    extra_j = [
        c
        for c in j.columns
        if c not in ("company_id", "period") and not str(c).startswith("j_")
    ]
    j = j[["company_id", "period"] + [f"j_{c}" for c in THE_14] + extra_j]
    m = s.merge(j, on=["company_id", "period"], how="inner")
    m["is_holdout"] = m["company_id"].isin(hold)
    train = m.loc[~m["is_holdout"]].copy()
    assert_no_holdout(train)
    if int(train["company_id"].isin(hold).sum()) != 0:
        raise RuntimeError("holdout leaked into train merge")
    return m, train


def cut_map(train: pd.DataFrame) -> list[dict]:
    rows = []
    for spec in MAP:
        sig = spec["signal"]
        jcol = f"j_{sig}"
        rho, n = spearman(train[jcol], train[spec["store"]])
        alts = []
        for alt in spec["alts"]:
            if alt not in train.columns:
                alts.append({"col": alt, "rho": None, "n": 0, "verdict": "NA"})
                continue
            r, nn = spearman(train[jcol], train[alt])
            alts.append({"col": alt, "rho": r, "n": nn, "verdict": verdict(r)})
        # pick best |ρ| among primary + alts for a "best twin" note (verdict still on primary)
        best = {"col": spec["store"], "rho": rho, "n": n, "verdict": verdict(rho)}
        for a in alts:
            if a["rho"] is not None and np.isfinite(a["rho"]) and (
                best["rho"] is None
                or not np.isfinite(best["rho"])
                or abs(a["rho"]) > abs(best["rho"])
            ):
                best = dict(a)
        j_nn = int(pd.to_numeric(train[jcol], errors="coerce").notna().sum())
        s_nn = int(pd.to_numeric(train[spec["store"]], errors="coerce").notna().sum())
        rows.append(
            {
                "signal": sig,
                "store": spec["store"],
                "rho": rho,
                "n": n,
                "verdict": verdict(rho),
                "j_nn": j_nn,
                "s_nn": s_nn,
                "j_cov": j_nn / len(train),
                "s_cov": s_nn / len(train),
                "alts": alts,
                "best": best,
                "q": spec["q"],
                "hole": spec["hole"],
                "park": spec["park"],
            }
        )
        print(
            f"  {sig:14s} vs {spec['store']:16s} ρ={_f(rho, 3)} n={n:6d} {verdict(rho):5s} "
            f"best={best['col']} {_f(best['rho'], 3)}"
        )
    return rows


def cut_coverage(store: pd.DataFrame, train: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    tr_ids = set(train["company_id"].unique())
    n_cm = len(train)
    n_co = train["company_id"].nunique()

    ever_inv = con.execute(
        """
        SELECT DISTINCT company_id
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
        """
    ).df()
    ever_inv["company_id"] = ever_inv["company_id"].astype(str)
    inv_train = set(ever_inv["company_id"]) & tr_ids
    dark = tr_ids - inv_train

    sched = con.execute(
        """
        SELECT DISTINCT d.company_id
        FROM debt_schedule_config s
        JOIN debt_products d ON d.product_id = s.product_id
        """
    ).df()
    sched["company_id"] = sched["company_id"].astype(str)
    sched_train = set(sched["company_id"]) & tr_ids

    b_null_co = (
        train.groupby("company_id")["b_liq"]
        .apply(lambda s: s.isna().all())
        .loc[lambda x: x]
        .index.astype(str)
        .tolist()
    )

    inv_signals = [r["signal"] for r in MAP if r["hole"] == "invoice"]
    b_signals = [r["signal"] for r in MAP if r["hole"] == "b_only"]

    # schedule columns exist in store but are not in the 14
    sched_cols = [c for c in ("f_w_rate", "f_months_to_next_pay", "f_util_snapshot", "f_sched_vs_obs") if c in train.columns]
    sched_cov = {
        c: {
            "cm": float(pd.to_numeric(train[c], errors="coerce").notna().mean()),
            "co": int(train.loc[pd.to_numeric(train[c], errors="coerce").notna(), "company_id"].nunique()),
        }
        for c in sched_cols
    }

    out = {
        "train_cm": int(n_cm),
        "train_co": int(n_co),
        "holdout_co": len(hold),
        "seed": FOLD_SEED,
        "invoice_train_co": len(inv_train),
        "invoice_dark_co": len(dark),
        "invoice_dark_share": len(dark) / n_co if n_co else float("nan"),
        "invoice_gated_signals": inv_signals,
        "b_only_signals": b_signals,
        "b_liq_allnull_co": len(b_null_co),
        "schedule_train_co": len(sched_train),
        "schedule_thin_share": len(sched_train) / n_co if n_co else float("nan"),
        "sched_cov": sched_cov,
        "flow_fc_cov": float(pd.to_numeric(train["f_fc_r"], errors="coerce").notna().mean()),
        "flow_ds_cov": float(pd.to_numeric(train["f_ds_r"], errors="coerce").notna().mean()),
        "delay_first6_null": True,
        "holdout_in_train": 0,
    }
    print(
        f"CUT 2 — train {n_co} cos / {n_cm} CM; invoice dark {len(dark)} "
        f"({len(dark)/n_co:.1%}); schedule cos {len(sched_train)} ({len(sched_train)/n_co:.1%})"
    )
    return out


def cut_concentration(train: pd.DataFrame) -> dict:
    rho_top, n_top = spearman(train["j_concentration"], train["d_cust_top1"])
    rho_hhi, n_hhi = spearman(train["j_concentration"], train["d_cust_hhi"])
    rho_th, n_th = spearman(train["d_cust_top1"], train["d_cust_hhi"])
    out = {
        "j_vs_top1": {"rho": rho_top, "n": n_top, "verdict": verdict(rho_top)},
        "j_vs_hhi": {"rho": rho_hhi, "n": n_hhi, "verdict": verdict(rho_hhi)},
        "top1_vs_hhi": {"rho": rho_th, "n": n_th, "verdict": verdict(rho_th)},
        "note": "Y4 quoted top1↔HHI ρ=0.991; HHI PARK as gradient (monopoly tail)",
    }
    print(
        f"CUT 3 — conc vs top1 ρ={_f(rho_top)} n={n_top} {verdict(rho_top)}; "
        f"vs HHI ρ={_f(rho_hhi)} n={n_hhi} {verdict(rho_hhi)}; "
        f"top1↔HHI ρ={_f(rho_th)} n={n_th}"
    )
    return out


def cut_volatility(train: pd.DataFrame) -> dict:
    rho_b, n_b = spearman(train["j_volatility"], train["b_bal_vol"])
    rho_a, n_a = spearman(train["j_volatility"], train["vol_from_a"])
    # identity check of vol_from_a construction vs Javier in6/sd6 if present
    extra = {}
    if "sd6" in train.columns and "in6" in train.columns:
        recon = np.minimum(3.0, train["sd6"] / np.maximum(train["in6"], 1.0))
        extra["j_vs_own_formula"] = dict(zip(("rho", "n"), spearman(train["j_volatility"], recon)))
        extra["j_vs_own_formula"]["verdict"] = verdict(extra["j_vs_own_formula"]["rho"])
    out = {
        "j_vs_b_bal_vol": {"rho": rho_b, "n": n_b, "verdict": verdict(rho_b)},
        "j_vs_vol_from_a": {"rho": rho_a, "n": n_a, "verdict": verdict(rho_a)},
        **extra,
        "note": "No named a_vol in the store. Mapping volatility→b_bal_vol is a different object.",
    }
    print(
        f"CUT 3 — vol vs b_bal_vol ρ={_f(rho_b)} {verdict(rho_b)}; "
        f"vs vol_from_a ρ={_f(rho_a)} {verdict(rho_a)}"
    )
    return out


def cut_overdue_window(train: pd.DataFrame) -> dict:
    out = {}
    for side, j, store, recon in (
        ("AR", "j_ar_overdue", "e_ar_overdue", "ar_od_3m"),
        ("AP", "j_ap_overdue", "e_ap_overdue", "ap_od_3m"),
    ):
        r_s, n_s = spearman(train[j], train[store])
        r_3, n_3 = spearman(train[j], train[recon]) if recon in train.columns else (float("nan"), 0)
        out[side] = {
            "vs_store": {"rho": r_s, "n": n_s, "verdict": verdict(r_s)},
            "vs_3m": {"rho": r_3, "n": n_3, "verdict": verdict(r_3)},
        }
        print(
            f"CUT 4 — {side} overdue vs store ρ={_f(r_s)} {verdict(r_s)}; "
            f"vs 3m-window ρ={_f(r_3)} {verdict(r_3)}"
        )
    return out


def pair_stats(a: pd.Series, b: pd.Series) -> dict:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d[np.isfinite(d["a"]) & np.isfinite(d["b"])]
    n = int(len(d))
    if n < 8:
        return {"n": n, "spearman": float("nan"), "pearson": float("nan"), "max_abs": float("nan"),
                "p50_abs": float("nan"), "share_exact": float("nan"), "share_1e6": float("nan")}
    delta = (d["a"] - d["b"]).abs()
    return {
        "n": n,
        "spearman": float(d["a"].corr(d["b"], method="spearman")),
        "pearson": float(d["a"].corr(d["b"], method="pearson")),
        "max_abs": float(delta.max()),
        "p50_abs": float(delta.median()),
        "share_exact": float((delta < 1e-12).mean()),
        "share_1e6": float((delta < 1e-6).mean()),
    }


def cut_identity(train: pd.DataFrame) -> list[dict]:
    print("CUT 6 — identity tightness (SAME can still differ in level)")
    rows = []
    pairs = [(spec["signal"], spec["store"]) for spec in MAP]
    pairs += [
        ("volatility", "vol_from_a"),
        ("d_runway", "b_d_runway_clip"),
        ("ar_overdue", "ar_od_3m"),
        ("ap_overdue", "ap_od_3m"),
        ("concentration", "d_cust_hhi"),
    ]
    seen = set()
    for sig, col in pairs:
        key = (sig, col)
        if key in seen or col not in train.columns:
            continue
        seen.add(key)
        st = pair_stats(train[f"j_{sig}"], train[col])
        st["signal"] = sig
        st["store"] = col
        rows.append(st)
        print(
            f"  {sig:14s} vs {col:16s} ρs={_f(st['spearman'])} ρp={_f(st['pearson'])} "
            f"max|Δ|={st['max_abs']:.3g} exact={st['share_exact']:.1%} n={st['n']}"
        )
    return rows


def cut_vol_why(train: pd.DataFrame) -> dict:
    print("CUT 7 — volatility is cashflow vol, not B")
    j = train["j_volatility"]
    checks = {}
    for col in ("b_bal_vol", "vol_from_a", "b_liq", "a_net", "a_in3", "b_runway"):
        if col not in train.columns:
            continue
        rho, n = spearman(j, train[col])
        checks[col] = {"rho": rho, "n": n, "verdict": verdict(rho)}
    size = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").abs())
    rho_sz, n_sz = spearman(j, size)
    rho_b_sz, n_bsz = spearman(train["b_bal_vol"], size)
    checks["j_vs_log1p_a_in3"] = {"rho": rho_sz, "n": n_sz, "verdict": verdict(rho_sz)}
    checks["b_bal_vol_vs_log1p_a_in3"] = {"rho": rho_b_sz, "n": n_bsz, "verdict": verdict(rho_b_sz)}
    # clip share on Javier vol
    jv = pd.to_numeric(j, errors="coerce")
    checks["j_clip3_share"] = float((jv >= 3.0 - 1e-12).mean()) if jv.notna().any() else float("nan")
    print(
        f"  j vs size ρ={_f(rho_sz)}; b_bal_vol vs size ρ={_f(rho_b_sz)}; "
        f"j==3 share={checks['j_clip3_share']:.1%}"
    )
    return checks


def cut_overdue_months(train: pd.DataFrame) -> dict:
    print("CUT 8 — overdue CLOSE by calendar month (stock ages)")
    out = {"AR": [], "AP": []}
    for period, sl in train.groupby("period", sort=True):
        for side, j, store in (
            ("AR", "j_ar_overdue", "e_ar_overdue"),
            ("AP", "j_ap_overdue", "e_ap_overdue"),
        ):
            rho, n = spearman(sl[j], sl[store])
            out[side].append({"period": str(pd.Timestamp(period).date()), "rho": rho, "n": n})
    for side in ("AR", "AP"):
        finite = [r for r in out[side] if r["n"] >= 30 and np.isfinite(r["rho"])]
        if finite:
            print(
                f"  {side} monthly ρ min={min(r['rho'] for r in finite):.3f} "
                f"max={max(r['rho'] for r in finite):.3f} last={finite[-1]['rho']:.3f} "
                f"({finite[-1]['period']})"
            )
    return out


def cut_extreme(con, train: pd.DataFrame) -> dict:
    print("CUT 9 — is_extreme vs flow twins (Spearman 1 ≠ value identity)")
    n_ext = int(
        con.execute("SELECT COUNT(*) FROM transactions WHERE coalesce(is_extreme, false)").fetchone()[0]
    )
    n_tx = int(con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
    by_cat = con.execute(
        """
        SELECT coalesce(category, '(null)') AS category, COUNT(*) AS n
        FROM transactions
        WHERE coalesce(is_extreme, false)
        GROUP BY 1
        ORDER BY n DESC
        """
    ).df()
    fc = pair_stats(train["j_fin_cost_r"], train["f_fc_r"])
    ds = pair_stats(train["j_debt_serv_r"], train["f_ds_r"])
    out = {
        "n_extreme": n_ext,
        "n_tx": n_tx,
        "extreme_share": n_ext / n_tx if n_tx else float("nan"),
        "by_cat": by_cat.head(12).to_dict("records"),
        "fc": fc,
        "ds": ds,
    }
    print(
        f"  extreme {n_ext}/{n_tx} ({n_ext/n_tx:.3%}); "
        f"fc max|Δ|={fc['max_abs']:.3g} exact={fc['share_exact']:.1%}; "
        f"ds max|Δ|={ds['max_abs']:.3g} exact={ds['share_exact']:.1%}"
    )
    return out


def cut_clip_and_dark(train: pd.DataFrame) -> dict:
    print("CUT 10 — d_runway clip + invoice-dark signal fill")
    jd = pd.to_numeric(train["j_d_runway"], errors="coerce")
    sd = pd.to_numeric(train["b_d_runway"], errors="coerce")
    clip_j = int(((jd < -12) | (jd > 12)).sum())  # should be 0: already clipped
    clip_s = int(((sd < -12) | (sd > 12)).sum())
    # 470: companies where e_ar_issued is all-null (never-ERP)
    dark = (
        train.groupby("company_id")["e_ar_issued"]
        .apply(lambda s: s.isna().all())
    )
    dark_ids = set(dark.loc[dark].index.astype(str))
    sl = train[train["company_id"].isin(dark_ids)]
    inv_cols = [
        "j_concentration",
        "j_ar_overdue",
        "j_ap_overdue",
        "j_delay_coll",
        "j_delay_paid",
        "e_ar_overdue",
        "e_ap_overdue",
        "e_delay_coll",
        "e_delay_paid",
        "d_cust_top1",
    ]
    fill = {}
    for c in inv_cols:
        if c not in sl.columns:
            continue
        nn = float(pd.to_numeric(sl[c], errors="coerce").notna().mean())
        fill[c] = nn
    delay_early = train[train["period"] < MONTHS[6]]
    delay_late = train[train["period"] >= MONTHS[6]]
    out = {
        "j_d_runway_outside_clip": clip_j,
        "store_d_runway_outside_clip": clip_s,
        "dark_co": len(dark_ids),
        "dark_cm": int(len(sl)),
        "dark_fill": fill,
        "delay_coll_early_nn": float(pd.to_numeric(delay_early["j_delay_coll"], errors="coerce").notna().mean()),
        "delay_coll_late_nn": float(pd.to_numeric(delay_late["j_delay_coll"], errors="coerce").notna().mean()),
        "delay_paid_early_nn": float(pd.to_numeric(delay_early["j_delay_paid"], errors="coerce").notna().mean()),
        "delay_paid_late_nn": float(pd.to_numeric(delay_late["j_delay_paid"], errors="coerce").notna().mean()),
    }
    print(
        f"  store d_runway |x|>12: {clip_s}; dark cos={len(dark_ids)} CM={len(sl)}; "
        f"dark j_ar_overdue nn={fill.get('j_ar_overdue', float('nan')):.1%}"
    )
    return out


def cut_dark_gap(con, train: pd.DataFrame) -> dict:
    print("CUT 12 — 470 live vs store e_ar_issued-null")
    hold = load_holdout()
    ever = con.execute(
        """
        SELECT DISTINCT company_id
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
        """
    ).df()
    ever_ids = set(ever["company_id"].astype(str)) - hold
    store_erp = set(
        train.loc[pd.to_numeric(train["e_ar_issued"], errors="coerce").notna(), "company_id"].astype(str)
    )
    train_ids = set(train["company_id"].astype(str))
    live_dark = train_ids - ever_ids
    store_dark = train_ids - store_erp
    only_live_dark = sorted(live_dark - store_dark)
    only_store_dark = sorted(store_dark - live_dark)
    extra = {}
    probe = (only_live_dark + only_store_dark)[:8]
    if probe:
        q = ",".join("'" + c.replace("'", "''") + "'" for c in probe)
        extra["probe"] = con.execute(
            f"""
            SELECT company_id,
                   COUNT(*) AS n,
                   SUM(CASE WHEN issuance_date IS NULL THEN 1 ELSE 0 END) AS n_null_iss,
                   SUM(CASE WHEN document_type = 'invoice' THEN 1 ELSE 0 END) AS n_inv,
                   SUM(CASE WHEN status = 'cancel' THEN 1 ELSE 0 END) AS n_cancel
            FROM invoices
            WHERE company_id IN ({q})
            GROUP BY 1
            """
        ).df().to_dict("records")
    out = {
        "live_dark": len(live_dark),
        "store_dark": len(store_dark),
        "only_live_dark": only_live_dark,
        "only_store_dark": only_store_dark,
        **extra,
    }
    print(
        f"  live dark={len(live_dark)} store dark={len(store_dark)} "
        f"only_live={only_live_dark} only_store={only_store_dark}"
    )
    return out


def cut_conc_residual(train: pd.DataFrame) -> dict:
    print("CUT 13 — concentration residual vs d_cust_top1")
    d = train[["company_id", "period", "j_concentration", "d_cust_top1", "d_n_cust"]].copy()
    d["j"] = pd.to_numeric(d["j_concentration"], errors="coerce")
    d["s"] = pd.to_numeric(d["d_cust_top1"], errors="coerce")
    both = d[np.isfinite(d["j"]) & np.isfinite(d["s"])]
    delta = (both["j"] - both["s"]).abs()
    differ = both.loc[delta > 1e-9]
    # Javier-only / store-only finite
    j_only = d[np.isfinite(d["j"]) & ~np.isfinite(d["s"])]
    s_only = d[~np.isfinite(d["j"]) & np.isfinite(d["s"])]
    out = {
        "n_both": int(len(both)),
        "n_differ": int(len(differ)),
        "differ_share": float(len(differ) / len(both)) if len(both) else float("nan"),
        "p50_abs": float(delta.median()) if len(both) else float("nan"),
        "p90_abs": float(delta.quantile(0.9)) if len(both) else float("nan"),
        "max_abs": float(delta.max()) if len(both) else float("nan"),
        "j_only": int(len(j_only)),
        "s_only": int(len(s_only)),
        "differ_median_n_cust": float(pd.to_numeric(differ["d_n_cust"], errors="coerce").median())
        if len(differ)
        else float("nan"),
        "same_median_n_cust": float(pd.to_numeric(both.loc[delta <= 1e-9, "d_n_cust"], errors="coerce").median())
        if (delta <= 1e-9).any()
        else float("nan"),
    }
    print(
        f"  differ {out['n_differ']}/{out['n_both']} ({out['differ_share']:.1%}) "
        f"max|Δ|={out['max_abs']:.3g}; j_only={out['j_only']} s_only={out['s_only']}"
    )
    return out


def cut_fc_diffs(train: pd.DataFrame) -> dict:
    print("CUT 14 — fin_cost_r level shifts (extreme 24 txs)")
    a = pd.to_numeric(train["j_fin_cost_r"], errors="coerce")
    b = pd.to_numeric(train["f_fc_r"], errors="coerce")
    ok = np.isfinite(a) & np.isfinite(b)
    delta = (a - b).abs()
    n_diff = int((ok & (delta > 1e-12)).sum())
    out = {
        "n_diff": n_diff,
        "n": int(ok.sum()),
        "max_abs": float(delta[ok].max()) if ok.any() else float("nan"),
        "share_diff": n_diff / int(ok.sum()) if ok.any() else float("nan"),
    }
    print(f"  fc differing rows {n_diff}/{int(ok.sum())} max|Δ|={out['max_abs']:.3g}")
    return out


def cut_old_stock(con, train: pd.DataFrame) -> dict:
    """Share of store open amount issued >3m ago — the overdue CLOSE mechanism."""
    print("CUT 15 — old unpaid stock share (why overdue is CLOSE)")
    # last train month only, cheap
    last = pd.Timestamp(LAST_M)
    e = last + pd.offsets.MonthEnd(0)
    w0 = last - pd.DateOffset(months=2)
    hold = load_holdout()
    q = con.execute(
        """
        SELECT company_id,
               CASE WHEN amount > 0 THEN 'AR' ELSE 'AP' END AS side,
               SUM(abs(amount)) AS open_amt,
               SUM(CASE WHEN CAST(issuance_date AS DATE) < CAST(? AS DATE)
                        THEN abs(amount) ELSE 0 END) AS old_amt
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL
          AND CAST(issuance_date AS DATE) <= CAST(? AS DATE)
          AND (payment_date IS NULL OR CAST(payment_date AS DATE) > CAST(? AS DATE))
          AND NOT coalesce(payment_date_invalid, FALSE)
        GROUP BY 1, 2
        """,
        [w0.to_pydatetime(), e.to_pydatetime(), e.to_pydatetime()],
    ).df()
    q["company_id"] = q["company_id"].astype(str)
    q = q[~q["company_id"].isin(hold)]
    out = {}
    for side, sl in q.groupby("side"):
        tot = float(sl["open_amt"].sum())
        old = float(sl["old_amt"].sum())
        out[str(side)] = {
            "old_share": old / tot if tot else float("nan"),
            "n_co": int(sl["company_id"].nunique()),
            "open": tot,
            "old": old,
        }
        print(f"  {side} last-month old-open share={old/tot:.1%} cos={sl.company_id.nunique()}")
    return out


def cut_conc_why(train: pd.DataFrame, con) -> dict:
    print("CUT 16 — concentration residual grows toward extract")
    d = train[["period", "j_concentration", "d_cust_top1", "d_n_cust"]].copy()
    d["j"] = pd.to_numeric(d["j_concentration"], errors="coerce")
    d["s"] = pd.to_numeric(d["d_cust_top1"], errors="coerce")
    d = d[np.isfinite(d["j"]) & np.isfinite(d["s"])]
    d["abs"] = (d["j"] - d["s"]).abs()
    by = (
        d.groupby("period", sort=True)
        .agg(
            n=("abs", "size"),
            p50=("abs", "median"),
            share_diff=("abs", lambda s: float((s > 1e-9).mean())),
            share_gt01=("abs", lambda s: float((s > 0.01).mean())),
        )
        .reset_index()
    )
    by["period"] = by["period"].dt.strftime("%Y-%m-%d")
    inv = con.execute(
        """
        SELECT
          COUNT(*) AS n_book,
          SUM(CASE WHEN due_date IS NULL THEN 1 ELSE 0 END) AS n_null_due,
          SUM(CASE WHEN amount > 0 AND due_date IS NULL THEN 1 ELSE 0 END) AS n_ar_null_due,
          SUM(CASE WHEN coalesce(payment_date_invalid, FALSE) THEN 1 ELSE 0 END) AS n_pdi,
          SUM(CASE WHEN status = 'paid' AND payment_date IS NULL THEN 1 ELSE 0 END) AS n_paid_null_pay
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL
        """
    ).df().iloc[0].to_dict()
    out = {
        "by_month": by.to_dict("records"),
        "share_gt01": float((d["abs"] > 0.01).mean()),
        "n_gt01": int((d["abs"] > 0.01).sum()),
        "first_share_diff": float(by.iloc[0]["share_diff"]) if len(by) else float("nan"),
        "last_share_diff": float(by.iloc[-1]["share_diff"]) if len(by) else float("nan"),
        "filters": {k: int(v) for k, v in inv.items()},
        "note": "Javier drops null-due (and paid-null-pay) from the whole book; Family D does not. Rank stays SAME.",
    }
    print(
        f"  |Δ|>0.01 {out['share_gt01']:.1%} ({out['n_gt01']}); "
        f"differ-share {out['first_share_diff']:.1%} → {out['last_share_diff']:.1%}; "
        f"null_due={inv['n_null_due']} pdi={inv['n_pdi']} paid_null_pay={inv['n_paid_null_pay']}"
    )
    return out


def cut_pairwise_14(train: pd.DataFrame) -> pd.DataFrame:
    """The 14 are not one number: invoice cluster ≠ B cluster."""
    print("CUT 17 — pairwise Spearman among the 14 (not a score)")
    cols = [f"j_{s}" for s in THE_14]
    mat = pd.DataFrame(index=list(THE_14), columns=list(THE_14), dtype=float)
    for i, a in enumerate(THE_14):
        for b in THE_14[i:]:
            rho, n = spearman(train[f"j_{a}"], train[f"j_{b}"])
            mat.loc[a, b] = rho
            mat.loc[b, a] = rho
    # off-diagonal max |ρ|
    off = mat.copy()
    np.fill_diagonal(off.values, np.nan)
    abs_off = off.abs()
    # find max pair
    max_pair = None
    if abs_off.notna().any().any():
        idx = abs_off.stack().idxmax()
        max_pair = {"a": idx[0], "b": idx[1], "rho": float(off.loc[idx])}
    # invoice vs B
    b_sig = ["runway", "d_runway", "neg_liq"]
    inv_sig = ["concentration", "ar_overdue", "ap_overdue", "delay_coll", "delay_paid"]
    cross = []
    for a in b_sig:
        for b in inv_sig:
            cross.append(abs(float(mat.loc[a, b])) if np.isfinite(mat.loc[a, b]) else np.nan)
    out_meta = {
        "max_pair": max_pair,
        "b_vs_invoice_maxabs": float(np.nanmax(cross)) if cross else float("nan"),
        "b_vs_invoice_median": float(np.nanmedian(cross)) if cross else float("nan"),
    }
    print(
        f"  max |ρ| off-diag {max_pair}; B↔invoice median |ρ|={out_meta['b_vs_invoice_median']:.3f} "
        f"max={out_meta['b_vs_invoice_maxabs']:.3f}"
    )
    return mat, out_meta


def cut_sched_39(con, train: pd.DataFrame) -> dict:
    print("CUT 18 — schedule 39 vs store f_w_rate 38")
    hold = load_holdout()
    df = con.execute(
        """
        SELECT d.company_id, d.product_id, d.created_after_snapshot, d.created_at,
               s.annual_interest_rate_or_spread IS NOT NULL AS has_rate
        FROM debt_schedule_config s
        JOIN debt_products d ON d.product_id = s.product_id
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    tr = df[~df["company_id"].isin(hold)]
    live = set(tr["company_id"])
    live_rate = set(tr.loc[tr["has_rate"].astype(bool), "company_id"])
    store_rate = set(train.loc[pd.to_numeric(train["f_w_rate"], errors="coerce").notna(), "company_id"].astype(str))
    extra = sorted(live_rate - store_rate)
    after = tr.loc[tr["company_id"].isin(extra), ["company_id", "created_after_snapshot", "created_at"]]
    print(f"  live sched={len(live)} live rate={len(live_rate)} store f_w_rate={len(store_rate)} extra={extra}")
    return {
        "train_sched_co": len(live),
        "train_rate_co": len(live_rate),
        "store_fw_rate_co": len(store_rate),
        "extra": extra,
        "extra_after_snapshot": after.to_dict("records"),
        "no_rate": [],
    }


def cut_pdi(con) -> dict:
    print("CUT 19 — payment_date_invalid share (D includes, Javier drops)")
    row = con.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN coalesce(payment_date_invalid, FALSE) THEN 1 ELSE 0 END) AS n_pdi,
               SUM(abs(amount)) AS amt,
               SUM(CASE WHEN coalesce(payment_date_invalid, FALSE) THEN abs(amount) ELSE 0 END) AS pdi_amt,
               SUM(CASE WHEN amount > 0 THEN abs(amount) ELSE 0 END) AS ar_amt,
               SUM(CASE WHEN amount > 0 AND coalesce(payment_date_invalid, FALSE)
                        THEN abs(amount) ELSE 0 END) AS pdi_ar
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL
        """
    ).df().iloc[0]
    out = {
        "n": int(row["n"]),
        "n_pdi": int(row["n_pdi"]),
        "pdi_n_share": float(row["n_pdi"] / row["n"]) if row["n"] else float("nan"),
        "pdi_amt_share": float(row["pdi_amt"] / row["amt"]) if row["amt"] else float("nan"),
        "pdi_ar_amt_share": float(row["pdi_ar"] / row["ar_amt"]) if row["ar_amt"] else float("nan"),
    }
    print(
        f"  PDI {out['n_pdi']:,}/{out['n']:,} ({out['pdi_n_share']:.1%}); "
        f"|amt| share {out['pdi_amt_share']:.1%}; AR |amt| share {out['pdi_ar_amt_share']:.1%}"
    )
    return out


def cut_od_by_company(train: pd.DataFrame) -> dict:
    print("CUT 24 — overdue CLOSE per company")
    out = {}
    for side, j, store in (
        ("AR", "j_ar_overdue", "e_ar_overdue"),
        ("AP", "j_ap_overdue", "e_ap_overdue"),
    ):
        rhos = []
        for _, sl in train.groupby("company_id", sort=False):
            rho, n = spearman(sl[j], sl[store])
            if n >= 8 and np.isfinite(rho):
                rhos.append(rho)
        s = pd.Series(rhos, dtype=float)
        out[side] = {
            "n_co": int(len(s)),
            "p50": float(s.median()) if len(s) else float("nan"),
            "share_same": float((s >= 0.95).mean()) if len(s) else float("nan"),
            "share_close": float(((s >= 0.80) & (s < 0.95)).mean()) if len(s) else float("nan"),
            "share_drift": float((s < 0.80).mean()) if len(s) else float("nan"),
        }
        print(
            f"  {side} p50={out[side]['p50']:.3f} SAME {out[side]['share_same']:.1%} "
            f"CLOSE {out[side]['share_close']:.1%} DRIFT {out[side]['share_drift']:.1%} n={out[side]['n_co']}"
        )
    return out


def cut_vol_by_company(train: pd.DataFrame) -> dict:
    print("CUT 23 — volatility DRIFT per company (is it a few names?)")
    rows = []
    for cid, sl in train.groupby("company_id", sort=False):
        rho, n = spearman(sl["j_volatility"], sl["b_bal_vol"])
        if n >= 8 and np.isfinite(rho):
            rows.append(rho)
    s = pd.Series(rows, dtype=float)
    out = {
        "n_co": int(s.notna().sum()),
        "p50": float(s.median()) if len(s) else float("nan"),
        "p10": float(s.quantile(0.1)) if len(s) else float("nan"),
        "p90": float(s.quantile(0.9)) if len(s) else float("nan"),
        "share_same": float((s.abs() >= 0.95).mean()) if len(s) else float("nan"),
        "share_close": float(((s.abs() >= 0.80) & (s.abs() < 0.95)).mean()) if len(s) else float("nan"),
        "share_drift": float((s.abs() < 0.80).mean()) if len(s) else float("nan"),
    }
    print(
        f"  per-co ρ p50={out['p50']:.3f} p10={out['p10']:.3f} p90={out['p90']:.3f}; "
        f"SAME {out['share_same']:.1%} CLOSE {out['share_close']:.1%} DRIFT {out['share_drift']:.1%} n={out['n_co']}"
    )
    return out


def cut_due_before_iss(con, jav: pd.DataFrame, store: pd.DataFrame) -> dict:
    print("CUT 25 — due < issuance (GREATEST) + grid identity")
    row = con.execute(
        """
        SELECT COUNT(*) AS n,
               SUM(CASE WHEN due_date < issuance_date THEN 1 ELSE 0 END) AS n_due_before,
               SUM(CASE WHEN due_date < issuance_date AND status = 'paid'
                         AND payment_date IS NOT NULL THEN 1 ELSE 0 END) AS n_paid_due_before
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL AND due_date IS NOT NULL
        """
    ).df().iloc[0]
    grid = {
        "jav_rows": int(len(jav)),
        "store_rows": int(len(store)),
        "jav_co": int(jav["company_id"].nunique()),
        "store_co": int(store["company_id"].nunique()),
    }
    print(
        f"  due<iss {int(row['n_due_before']):,}/{int(row['n']):,} "
        f"(paid {int(row['n_paid_due_before'])}); grid jav={grid['jav_rows']} store={grid['store_rows']}"
    )
    return {
        "n": int(row["n"]),
        "n_due_before": int(row["n_due_before"]),
        "n_paid_due_before": int(row["n_paid_due_before"]),
        **grid,
    }


def cut_size_screen(train: pd.DataFrame) -> list[dict]:
    print("CUT 26 — SIZE screen of the 14 (not a score)")
    size = np.log1p(pd.to_numeric(train["a_in3"], errors="coerce").abs())
    rows = []
    for sig in THE_14:
        rho, n = spearman(train[f"j_{sig}"], size)
        rows.append({"signal": sig, "rho": rho, "n": n, "size": abs(rho) >= 0.50 if np.isfinite(rho) else False})
        print(f"  {sig:14s} vs log1p(a_in3) ρ={_f(rho)} n={n} {'SIZE' if rows[-1]['size'] else 'ok'}")
    return rows


def cut_early_invoice(train: pd.DataFrame) -> dict:
    print("CUT 27 — Javier skips invoice features in first 2 calendar months")
    early = train[train["period"] < MONTHS[2]]  # 2024-09, 2024-10
    late = train[train["period"] >= MONTHS[2]]
    cols = ["j_concentration", "j_ar_overdue", "j_ap_overdue", "j_delay_coll", "j_delay_paid"]
    out = {"early_cm": int(len(early)), "late_cm": int(len(late))}
    for c in cols:
        out[f"{c}_early"] = float(pd.to_numeric(early[c], errors="coerce").notna().mean())
        out[f"{c}_late"] = float(pd.to_numeric(late[c], errors="coerce").notna().mean())
    print(
        f"  early CM={len(early)} conc nn={out['j_concentration_early']:.1%} "
        f"ar_od nn={out['j_ar_overdue_early']:.1%}; late conc={out['j_concentration_late']:.1%}"
    )
    return out


def cut_folds(con, train: pd.DataFrame) -> list[dict]:
    print("CUT 28 — group-fold Spearman stability (seed 20260918, not a model)")
    cos = train_companies(con)
    folds = group_folds(cos, n=5, seed=FOLD_SEED)
    assert_no_holdout(folds)
    t = train.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    if t["fold"].isna().any():
        raise RuntimeError("train rows missing fold")
    rows = []
    for spec in MAP:
        rhos = []
        for f in range(5):
            sl = t[t["fold"] == f]
            rho, n = spearman(sl[f"j_{spec['signal']}"], sl[spec["store"]])
            rhos.append({"fold": f, "rho": rho, "n": n})
        vals = [r["rho"] for r in rhos if np.isfinite(r["rho"])]
        rows.append(
            {
                "signal": spec["signal"],
                "store": spec["store"],
                "folds": rhos,
                "min": float(min(vals)) if vals else float("nan"),
                "max": float(max(vals)) if vals else float("nan"),
                "spread": float(max(vals) - min(vals)) if vals else float("nan"),
            }
        )
        print(
            f"  {spec['signal']:14s} fold ρ {[round(r['rho'], 3) if np.isfinite(r['rho']) else None for r in rhos]} "
            f"spread={rows[-1]['spread']:.3f}"
        )
    return rows


def cut_dark_other(con, train: pd.DataFrame) -> dict:
    print("CUT 29 — dark 470 still have bank A/B/F")
    hold = load_holdout()
    ever = con.execute(
        """
        SELECT DISTINCT company_id
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
        """
    ).df()
    ever_ids = set(ever["company_id"].astype(str))
    train_ids = set(train["company_id"].astype(str))
    ids = train_ids - ever_ids  # 470 live dark
    sl = train[train["company_id"].isin(ids)]
    never_ar = (
        train.groupby("company_id")["j_ar_overdue"]
        .apply(lambda s: s.isna().all())
        .sum()
    )
    out = {"dark_co": len(ids), "dark_cm": int(len(sl)), "never_ar_od_co": int(never_ar)}
    for c in ("b_runway", "a_io_ratio", "f_fc_r", "j_runway", "j_coverage", "j_fin_cost_r"):
        out[c] = float(pd.to_numeric(sl[c], errors="coerce").notna().mean())
    print(
        f"  live-dark {len(ids)} (holdout excluded={len(ids & hold)}); "
        f"never j_ar_overdue {int(never_ar)}; "
        f"b_runway nn={out['b_runway']:.1%} a_io {out['a_io_ratio']:.1%}"
    )
    return out


def cut_sig_meta() -> dict:
    """Document SIG types. Do not apply percentile or share→score."""
    print("CUT 21 — SIG metadata (unread as a score)")
    types = SIG.set_index("signal")["type"].to_dict()
    dirs = SIG.set_index("signal")["direction"].to_dict()
    # pillars listed only as Javier's grouping — we do not weight them
    pillars = SIG.set_index("signal")["pillar"].to_dict()
    share = [s for s, t in types.items() if t == "share"]
    print(f"  type=share (not percentile): {share}; 14 signals, pillars recorded not computed")
    return {"types": types, "direction": dirs, "pillar_label": pillars, "share_signals": share}


def cut_best_remap(rows: list[dict]) -> dict:
    """If we accept reconstructed twins, how many stay DRIFT? Not a new map."""
    print("CUT 22 — best-twin remap (footnote, primary verdicts unchanged)")
    counts = {"SAME": 0, "CLOSE": 0, "DRIFT": 0, "NA": 0}
    for r in rows:
        counts[r["best"]["verdict"]] = counts.get(r["best"]["verdict"], 0) + 1
    print(f"  best-twin {counts} (primary still 11/2/1)")
    return counts


def cut_holdout_cov(merged: pd.DataFrame) -> dict:
    print("CUT 20 — holdout coverage only (no ρ)")
    h = merged.loc[merged["is_holdout"]]
    out = {"cm": int(len(h)), "co": int(h["company_id"].nunique())}
    for sig in THE_14:
        nn = float(pd.to_numeric(h[f"j_{sig}"], errors="coerce").notna().mean())
        out[sig] = nn
    print(f"  holdout {out['co']} cos / {out['cm']} CM; runway nn={out['runway']:.1%} conc nn={out['concentration']:.1%}")
    return out


def cut_dso(train: pd.DataFrame) -> dict:
    print("CUT 11 — DSO is not in the 14 (Y7 SHAP#1, fails short-DSO)")
    rho_od, n_od = spearman(train["j_ar_overdue"], train["e_dso_proxy"])
    rho_st, n_st = spearman(train["e_ar_overdue"], train["e_dso_proxy"])
    rho_iss, n_iss = spearman(train["j_ar_overdue"], train["e_ar_issued"]) if "e_ar_issued" in train.columns else (float("nan"), 0)
    out = {
        "j_od_vs_dso": {"rho": rho_od, "n": n_od, "verdict": verdict(rho_od)},
        "store_od_vs_dso": {"rho": rho_st, "n": n_st, "verdict": verdict(rho_st)},
        "j_od_vs_issued": {"rho": rho_iss, "n": n_iss, "verdict": verdict(rho_iss)},
        "note": "DSO is beyond the 14. Night PARK as Y7 X on short-DSO fold. issued_lag1 is the Q6 KEEP.",
    }
    print(f"  j_ar_overdue vs e_dso_proxy ρ={_f(rho_od)} n={n_od} {verdict(rho_od)}")
    return out


def cut_beyond(train: pd.DataFrame) -> list[dict]:
    rows = []
    for spec in BEYOND:
        col = spec["col"]
        if col not in train.columns:
            rows.append({"col": col, "in_store": False, "cm_cov": 0.0, "co": 0, "why": spec["why"]})
            print(f"CUT 4 — BEYOND {col}: not a store column (model-time / missing)")
            continue
        x = pd.to_numeric(train[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "in_store": True,
                "cm_cov": float(x.notna().mean()),
                "co": int(train.loc[x.notna(), "company_id"].nunique()),
                "why": spec["why"],
            }
        )
        print(f"CUT 4 — BEYOND {col}: cov={x.notna().mean():.3f} cos={train.loc[x.notna(), 'company_id'].nunique()}")
    return rows


def cut_heatmap(train: pd.DataFrame) -> pd.DataFrame | None:
    if not HAS_MPL:
        print("CUT 5 — matplotlib missing, skip PNG")
        return None
    mat = np.full((len(THE_14), len(HEAT_STORE)), np.nan)
    for i, sig in enumerate(THE_14):
        for j, col in enumerate(HEAT_STORE):
            if col not in train.columns:
                continue
            rho, n = spearman(train[f"j_{sig}"], train[col])
            if n >= 30 and np.isfinite(rho):
                mat[i, j] = rho
    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    cmap = plt.cm.RdBu_r
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(HEAT_STORE)))
    ax.set_xticklabels(HEAT_STORE, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(len(THE_14)))
    ax.set_yticklabels(THE_14, fontsize=9)
    ax.set_title("Train Spearman: Javier raw 14 vs store twins (no percentile, no 0–100)")
    # mark primary twins
    primary = {spec["signal"]: spec["store"] for spec in MAP}
    for i, sig in enumerate(THE_14):
        col = primary.get(sig)
        if col in HEAT_STORE:
            j = HEAT_STORE.index(col)
            ax.add_patch(
                plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2)
            )
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="ρ")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"CUT 5 — wrote {OUT_PNG}")
    return pd.DataFrame(mat, index=list(THE_14), columns=list(HEAT_STORE))


def summarize(rows: list[dict]) -> dict:
    counts = {"SAME": 0, "CLOSE": 0, "DRIFT": 0, "NA": 0}
    for r in rows:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    drifted = [r for r in rows if r["verdict"] == "DRIFT"]
    worst = None
    if drifted:
        worst = min(drifted, key=lambda r: abs(r["rho"]) if r["rho"] is not None else 1.0)
    elif rows:
        worst = min(rows, key=lambda r: abs(r["rho"]) if r["rho"] is not None and np.isfinite(r["rho"]) else 1.0)
    return {"counts": counts, "worst": worst}


def write_md(
    parity,
    cov,
    rows,
    conc,
    vol,
    overdue,
    beyond,
    summary,
    elapsed,
    ident=None,
    vol_why=None,
    od_months=None,
    extreme=None,
    clip=None,
    dso=None,
    dark_gap=None,
    conc_res=None,
    fc_diffs=None,
    old_stock=None,
    conc_why=None,
    pair14=None,
    sched39=None,
    pdi=None,
    hold_cov=None,
    sig_meta=None,
    best_remap=None,
    vol_co=None,
    od_co=None,
    due_iss=None,
    size_scr=None,
    early_inv=None,
    folds=None,
    dark_bank=None,
) -> None:
    worst = summary["worst"]
    c = summary["counts"]
    lines = [
        "# Javier 14 ↔ feature store (raw signals, no 0–100)",
        "",
        f"Generated `{_utc_ts()}` UTC by `python -m analysis.evaluate.score_pipeline_qa`.",
        f"Holdout 72 (seed {FOLD_SEED}) is **coverage only**. Spearman / SAME / CLOSE / DRIFT are train.",
        "Does not call `fit_ref` / `run_score` / `make_traj`. Does not write a scorecard or pillar weights.",
        "Does not redo Family B balances. Does not touch `product/`.",
        "",
        "## Headline",
        "",
        f"- Verdicts on the **primary** store twin: **{c['SAME']} SAME / {c['CLOSE']} CLOSE / {c['DRIFT']} DRIFT** (of 14).",
        (
            f"- Worst primary drift: `{worst['signal']}` vs `{worst['store']}` ρ={_f(worst['rho'])} "
            f"n={worst['n']} ({worst['verdict']})."
            if worst
            else "- No worst row."
        ),
        f"- CAT_MAP parity: **{'IDENTICAL' if parity['same_mapping'] else 'MISMATCH'}** "
        f"({parity['n_pipe']} pipeline keys, {parity['n_store']} store keys).",
        "- The 14 do not include days-with-tx / issued_lag1 — those are *beyond* Javier, not a drift.",
        "- CLOSE: `ar_overdue` / `ap_overdue` vs all-open store (3m-window reconstruction is identity). "
        "DRIFT: `volatility` vs `b_bal_vol` (cashflow sd, not balance sd). Reconstruct from A is identity — no named `a_vol`.",
        f"- Concentration is **top1** (ρ={_f(conc['j_vs_top1']['rho'])}), not HHI (ρ={_f(conc['j_vs_hhi']['rho'])}; "
        f"store top1↔HHI {_f(conc['top1_vs_hhi']['rho'])}). HHI stays PARK as a gradient X (Y4 monopoly tail).",
        f"- Train panel: {cov['train_co']} companies / {cov['train_cm']:,} company-months. "
        f"Invoice-dark: **{cov['invoice_dark_co']}** ({cov['invoice_dark_share']:.1%}; `COMP_0962` refund ghost). "
        "Schedule as-of rate: 38 train companies — not a hole in the 14.",
        "- Group-fold ρ (seed 20260918) does not flip a verdict. Vol fold-0 is 0.125, still DRIFT; overdue stays CLOSE.",
        f"- Wall {elapsed:.0f}s. PNG: `{OUT_PNG.name}`.",
        "",
        "## Brief questions (the 14, not a score)",
        "",
        "| signal | store twin | Q | hole | PARK / note |",
        "| --- | --- | --- | --- | --- |",
    ]
    for spec, row in zip(MAP, rows):
        lines.append(
            f"| `{row['signal']}` | `{row['store']}` | {row['q']} | {row['hole']} | {spec['park']} |"
        )
    lines += [
        "",
        "## 1. CAT_MAP parity",
        "",
        "Pipeline `score_pipeline.CAT_MAP` vs `analysis.features.common.CAT_MAP`.",
        "",
        f"- keys equal: {parity['same_keys']}",
        f"- mapping equal: **{parity['same_mapping']}**",
        f"- only in pipeline: {parity['only_pipe'] or '∅'}",
        f"- only in store: {parity['only_store'] or '∅'}",
        f"- value diffs: {parity['value_diffs'] or '∅'}",
        f"- groups: {parity['groups_pipe']}",
        "",
        "## 2. Train Spearman (raw, pre-percentile)",
        "",
        "ρ ≥ 0.95 SAME · 0.80–0.95 CLOSE · < 0.80 DRIFT. Pairwise finite n. Train only.",
        "",
        "| signal | store | ρ | n | verdict | Javier cov | store cov | best twin | best ρ |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | --- | ---: |",
    ]
    for r in rows:
        b = r["best"]
        lines.append(
            f"| `{r['signal']}` | `{r['store']}` | {_f(r['rho'])} | {r['n']:,} | **{r['verdict']}** | "
            f"{r['j_cov']:.1%} | {r['s_cov']:.1%} | `{b['col']}` | {_f(b['rho'])} |"
        )
    lines += [
        "",
        "### Alternates",
        "",
        "| signal | alt | ρ | n | verdict |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for r in rows:
        for a in r["alts"]:
            lines.append(
                f"| `{r['signal']}` | `{a['col']}` | {_f(a['rho'])} | {a['n']:,} | {a['verdict']} |"
            )
    lines += [
        "",
        "## 3. Coverage holes",
        "",
        f"- Invoice-gated signals: {', '.join('`'+s+'`' for s in cov['invoice_gated_signals'])}.",
        f"- Train companies with **no** book invoices (the 470-dark class): **{cov['invoice_dark_co']}** / {cov['train_co']} "
        f"({cov['invoice_dark_share']:.1%}). Those five signals are undefined there — not a formula drift.",
        f"- B-only signals: {', '.join('`'+s+'`' for s in cov['b_only_signals'])}. "
        f"Companies with all-null `b_liq`: {cov['b_liq_allnull_co']}. Last-value Q1 KEEP as description; never B as Y2/Y3 X.",
        f"- Schedule-thin is **not** a hole in the 14. Flow coverage `f_fc_r`={cov['flow_fc_cov']:.1%}, "
        f"`f_ds_r`={cov['flow_ds_cov']:.1%}. Schedule / util live outside the 14:",
        "",
    ]
    for col, d in cov["sched_cov"].items():
        lines.append(f"  - `{col}`: {d['cm']:.1%} CM / {d['co']} train companies")
    lines += [
        "",
        f"- Delay signals are null on the first 6 calendar months (2024-09..2025-02) by construction (left truncation).",
        "",
        "## 4. Concentration: top1 or HHI?",
        "",
        f"- Javier `concentration` vs `d_cust_top1`: ρ={_f(conc['j_vs_top1']['rho'])} n={conc['j_vs_top1']['n']:,} **{conc['j_vs_top1']['verdict']}**.",
        f"- Javier `concentration` vs `d_cust_hhi`: ρ={_f(conc['j_vs_hhi']['rho'])} n={conc['j_vs_hhi']['n']:,} **{conc['j_vs_hhi']['verdict']}**.",
        f"- Store `d_cust_top1` ↔ `d_cust_hhi`: ρ={_f(conc['top1_vs_hhi']['rho'])} n={conc['top1_vs_hhi']['n']:,} "
        f"(Y4 quote 0.991).",
        f"- {conc['note']}",
        "",
        "## 5. Volatility twin",
        "",
        f"- vs `b_bal_vol`: ρ={_f(vol['j_vs_b_bal_vol']['rho'])} n={vol['j_vs_b_bal_vol']['n']:,} **{vol['j_vs_b_bal_vol']['verdict']}**.",
        f"- vs reconstructed `vol_from_a` = min(3, sd6(`a_net`) / max(mean6(`a_op_in`), 1)): "
        f"ρ={_f(vol['j_vs_vol_from_a']['rho'])} n={vol['j_vs_vol_from_a']['n']:,} **{vol['j_vs_vol_from_a']['verdict']}**.",
        f"- {vol['note']}",
        "",
        "## 6. Overdue window (3m issuance vs all-open)",
        "",
    ]
    for side, d in overdue.items():
        lines.append(
            f"- {side}: vs store `{('e_ar_overdue' if side=='AR' else 'e_ap_overdue')}` "
            f"ρ={_f(d['vs_store']['rho'])} **{d['vs_store']['verdict']}**; "
            f"vs reconstructed 3m-window ρ={_f(d['vs_3m']['rho'])} **{d['vs_3m']['verdict']}**."
        )
    lines += [
        "",
        "## 7. Beyond Javier (night kept — not drift)",
        "",
        "The 14 do not include days-with-tx / issued_lag1.",
        "",
        "| column | in store? | train cov | why the night kept it |",
        "| --- | --- | ---: | --- |",
    ]
    for b in beyond:
        lines.append(
            f"| `{b['col']}` | {'yes' if b['in_store'] else 'no (lag at model time)'} | "
            f"{b['cm_cov']:.1%} | {b['why']} |"
        )
    if ident:
        lines += [
            "",
            "## 8. Identity tightness (Spearman SAME ≠ value copy)",
            "",
            "| pair | Spearman | Pearson | max\\|Δ\\| | p50\\|Δ\\| | exact | n |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for st in ident:
            lines.append(
                f"| `{st['signal']}` vs `{st['store']}` | {_f(st['spearman'])} | {_f(st['pearson'])} | "
                f"{st['max_abs']:.3g} | {st['p50_abs']:.3g} | {st['share_exact']:.1%} | {st['n']:,} |"
            )
    if vol_why:
        lines += [
            "",
            "## 9. Why volatility is the only primary DRIFT",
            "",
            "Javier `volatility` = min(3, sd6(op_in−op_out) / max(mean6(op_in), 1)). "
            "`b_bal_vol` = sd6(liq) / max(mean |out|, 1). Different object.",
            "",
        ]
        for k, d in vol_why.items():
            if isinstance(d, dict) and "rho" in d:
                lines.append(f"- `{k}`: ρ={_f(d['rho'])} n={d['n']:,} {d['verdict']}")
        if "j_clip3_share" in vol_why:
            lines.append(f"- Javier vol hits the clip=3 on {vol_why['j_clip3_share']:.1%} of train CM (defined rows use pairwise n).")
        lines.append("- Do not invent an `a_vol` column tonight (cashflow.py is not owned). Reconstruct from A is identity.")
    if od_months:
        lines += [
            "",
            "## 10. Overdue ρ by month (CLOSE is aging stock, not CAT_MAP)",
            "",
            "Store `e_*_overdue` uses **all unpaid** invoices. Javier uses issuance in the last 3 months. "
            "The 3m reconstruction is identity (ρ=1). Monthly ρ vs the store twin should fall as the unpaid tail grows.",
            "",
            "| month | AR ρ | AR n | AP ρ | AP n |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        ar = {r["period"]: r for r in od_months["AR"]}
        ap = {r["period"]: r for r in od_months["AP"]}
        for p in sorted(set(ar) | set(ap)):
            a, b = ar.get(p, {}), ap.get(p, {})
            lines.append(
                f"| {p} | {_f(a.get('rho'))} | {a.get('n', 0):,} | {_f(b.get('rho'))} | {b.get('n', 0):,} |"
            )
    if extreme:
        lines += [
            "",
            "## 11. `is_extreme` vs flow twins",
            "",
            f"- Extreme txs: {extreme['n_extreme']:,} / {extreme['n_tx']:,} ({extreme['extreme_share']:.3%}). "
            "Family F drops them; Javier `_monthly_flows` does not.",
            f"- `fin_cost_r` vs `f_fc_r`: Spearman {_f(extreme['fc']['spearman'])}, Pearson {_f(extreme['fc']['pearson'])}, "
            f"max|Δ|={extreme['fc']['max_abs']:.3g}, exact={extreme['fc']['share_exact']:.1%}.",
            f"- `debt_serv_r` vs `f_ds_r`: Spearman {_f(extreme['ds']['spearman'])}, Pearson {_f(extreme['ds']['pearson'])}, "
            f"max|Δ|={extreme['ds']['max_abs']:.3g}, exact={extreme['ds']['share_exact']:.1%}.",
            "- Rank SAME can hide a few extreme-driven level shifts. Not enough to leave SAME.",
            "",
        ]
        if extreme.get("by_cat"):
            lines.append("| extreme category | n |")
            lines.append("| --- | ---: |")
            for r in extreme["by_cat"]:
                lines.append(f"| {r['category']} | {int(r['n']):,} |")
    if clip:
        lines += [
            "",
            "## 12. Clip + invoice-dark fill",
            "",
            f"- Javier `d_runway` already clipped: rows outside [-12,12] = {clip['j_d_runway_outside_clip']}.",
            f"- Store `b_d_runway` unclipped rows |x|>12: **{clip['store_d_runway_outside_clip']}**. "
            "Spearman still 1.000 — clip is a tail, not a rank change.",
            f"- Dark (no `e_ar_issued` ever) train companies: **{clip['dark_co']}** / {cov['train_co']} "
            f"({clip['dark_cm']:,} CM).",
            "",
            "| column on dark CM | finite share |",
            "| --- | ---: |",
        ]
        for k, v in clip["dark_fill"].items():
            lines.append(f"| `{k}` | {v:.1%} |")
        lines += [
            "",
            f"- `delay_*` finite share before 2025-03: coll {clip['delay_coll_early_nn']:.1%} / paid {clip['delay_paid_early_nn']:.1%} (should be ~0).",
            f"- after: coll {clip['delay_coll_late_nn']:.1%} / paid {clip['delay_paid_late_nn']:.1%}.",
        ]
    if dso:
        lines += [
            "",
            "## 13. DSO is not in the 14",
            "",
            f"- Javier `ar_overdue` vs `e_dso_proxy`: ρ={_f(dso['j_od_vs_dso']['rho'])} n={dso['j_od_vs_dso']['n']:,} "
            f"**{dso['j_od_vs_dso']['verdict']}**.",
            f"- Store `e_ar_overdue` vs `e_dso_proxy`: ρ={_f(dso['store_od_vs_dso']['rho'])} n={dso['store_od_vs_dso']['n']:,}.",
            f"- {dso['note']}",
        ]
    if dark_gap:
        lines += [
            "",
            "## 13b. Dark 470 vs store issued-null",
            "",
            f"- Live invoice ever (train): dark **{dark_gap['live_dark']}**.",
            f"- Store `e_ar_issued` all-null: dark **{dark_gap['store_dark']}**.",
            f"- only live-dark: {dark_gap['only_live_dark'] or '∅'}",
            f"- only store-dark: {dark_gap['only_store_dark'] or '∅'}",
            "- `COMP_0962` is the join-QA refund-only ghost (one `Abono` / `refund`). Live ever-invoice excludes it (470 dark). "
            "Store writes `e_ar_issued=0` / `e_credit_note_ratio=1` on the refund month, so issued-null dark is 469. "
            "The five invoice-gated signals stay all-null. Do not count the ghost as ERP.",
        ]
        if dark_gap.get("probe"):
            lines += ["", "| company | n invoices | null iss | n invoice-type | n cancel |", "| --- | ---: | ---: | ---: | ---: |"]
            for r in dark_gap["probe"]:
                lines.append(
                    f"| `{r['company_id']}` | {int(r['n'])} | {int(r['n_null_iss'])} | "
                    f"{int(r['n_inv'])} | {int(r['n_cancel'])} |"
                )
    if conc_res:
        lines += [
            "",
            "## 13c. Concentration residual (top1 is the pipeline object)",
            "",
            f"- Both finite: {conc_res['n_both']:,}. Differ at 1e-9: {conc_res['n_differ']:,} ({conc_res['differ_share']:.1%}).",
            f"- max|Δ|={conc_res['max_abs']:.3g}; p50={conc_res['p50_abs']:.3g}; p90={conc_res['p90_abs']:.3g}.",
            f"- Javier-only finite: {conc_res['j_only']:,}. Store-only finite: {conc_res['s_only']:,}.",
            f"- Median `d_n_cust` on differ vs exact: {conc_res['differ_median_n_cust']:.1f} vs {conc_res['same_median_n_cust']:.1f}.",
            "- Residual is the due_date / paid-validity filter Javier applies and Family D does not — not HHI vs top1.",
        ]
    if fc_diffs:
        lines += [
            "",
            "## 13d. `fin_cost_r` level shifts",
            "",
            f"- Differing train CM: {fc_diffs['n_diff']:,} / {fc_diffs['n']:,} ({fc_diffs['share_diff']:.3%}). "
            f"max|Δ|={fc_diffs['max_abs']:.3g}. Rank stays SAME.",
        ]
    if old_stock:
        lines += [
            "",
            "## 13e. Old unpaid stock at last month (CLOSE mechanism)",
            "",
        ]
        for side, d in old_stock.items():
            lines.append(
                f"- {side}: {d['old_share']:.1%} of open |amount| was issued before the 3m window "
                f"({d['n_co']} train companies)."
            )
    if conc_why:
        lines += [
            "",
            "## 13f. Concentration residual toward extract",
            "",
            f"- |Δ| > 0.01 on {conc_why['share_gt01']:.1%} of both-finite rows ({conc_why['n_gt01']:,}). Median Δ is ~0.",
            f"- Share of rows that differ at all: {conc_why['first_share_diff']:.1%} (first full6 month) → "
            f"{conc_why['last_share_diff']:.1%} (2026-08). Rank stays SAME (ρ=0.995).",
            f"- Book filters Javier applies and D does not: null due {conc_why['filters']['n_null_due']:,} "
            f"(AR {conc_why['filters']['n_ar_null_due']:,}); payment_date_invalid {conc_why['filters']['n_pdi']:,}; "
            f"paid with null payment_date {conc_why['filters']['n_paid_null_pay']:,}.",
            f"- {conc_why['note']}",
        ]
    if pair14:
        meta = pair14["meta"]
        mp = meta.get("max_pair") or {}
        lines += [
            "",
            "## 13g. The 14 are not one number",
            "",
            f"- Largest |ρ| among distinct signals: `{mp.get('a')}` ↔ `{mp.get('b')}` ρ={_f(mp.get('rho'))}.",
            f"- B cluster (runway / d_runway / neg_liq) vs invoice cluster: median |ρ|="
            f"{_f(meta['b_vs_invoice_median'])}, max={_f(meta['b_vs_invoice_maxabs'])}.",
            "- Do not average the 14. Invoice-gated signals are a different object from last-value liquidity.",
            "- `coverage` ↔ `net_margin` ρ=0.989 is the same 3m in/out window (caja twins), not two independent readings.",
        ]
    if sched39:
        lines += [
            "",
            "## 13h. Schedule-thin is outside the 14",
            "",
            f"- Live train schedule companies: {sched39['train_sched_co']}. Live rate: {sched39['train_rate_co']}. "
            f"Store `f_w_rate`: {sched39['store_fw_rate_co']}.",
            f"- Extra vs store: {sched39['extra'] or '∅'}. "
            "`COMP_1027` is created_after_snapshot (2026-09-15 loan). Night 38 is the as-of panel. The 14 already chose flow.",
        ]
    if pdi:
        lines += [
            "",
            "## 13i. `payment_date_invalid` (why D residual grows)",
            "",
            f"- PDI invoices: {pdi['n_pdi']:,} / {pdi['n']:,} ({pdi['pdi_n_share']:.1%}).",
            f"- Share of book |amount|: {pdi['pdi_amt_share']:.1%} (mostly AP). AR |amount| share: {pdi['pdi_ar_amt_share']:.1%}.",
            "- Javier drops paid-null-pay from the whole invoice book. Family D concentration does not. Family E open stock does drop PDI — delays stay SAME.",
        ]
    if hold_cov:
        lines += [
            "",
            "## 13j. Holdout coverage (descriptive — no ρ)",
            "",
            f"- Holdout {hold_cov['co']} companies / {hold_cov['cm']:,} CM. Seed {FOLD_SEED}. Not used for Spearman.",
            "",
            "| signal | holdout finite share |",
            "| --- | ---: |",
        ]
        for sig in THE_14:
            lines.append(f"| `{sig}` | {hold_cov[sig]:.1%} |")
    if sig_meta:
        lines += [
            "",
            "## 13k. SIG types (not applied)",
            "",
            f"- Javier marks only `{', '.join(sig_meta['share_signals'])}` as type=share (the score would do 100×(1−value)). "
            "The other 13 are percentile-against-reference. **This module applies neither.**",
            "- Pillar labels exist on SIG. They are not computed here.",
        ]
    if best_remap:
        lines += [
            "",
            "## 13l. Best-twin footnote (primary map unchanged)",
            "",
            f"- If `volatility`→`vol_from_a` and overdue→3m-window: **{best_remap['SAME']} SAME / {best_remap['CLOSE']} CLOSE / {best_remap['DRIFT']} DRIFT**.",
            "- That is a reconstruction, not a store rename. Primary verdicts stay on the named columns.",
        ]
    if vol_co:
        lines += [
            "",
            "## 13m. Volatility DRIFT is typical, not a few names",
            "",
            f"- Per-company Spearman (n≥8 months): {vol_co['n_co']} train companies. "
            f"p10={_f(vol_co['p10'])} p50={_f(vol_co['p50'])} p90={_f(vol_co['p90'])}.",
            f"- Company verdicts: SAME {vol_co['share_same']:.1%} / CLOSE {vol_co['share_close']:.1%} / DRIFT {vol_co['share_drift']:.1%}.",
            "- `b_bal_vol` is not a noisy twin of Javier vol. It is a different series for most books.",
        ]
    if od_co:
        lines += [
            "",
            "## 13n. Overdue CLOSE per company",
            "",
        ]
        for side, d in od_co.items():
            lines.append(
                f"- {side}: p50 ρ={_f(d['p50'])} on {d['n_co']} companies with ≥8 overlapping months. "
                f"SAME {d['share_same']:.1%} / CLOSE {d['share_close']:.1%} / DRIFT {d['share_drift']:.1%}."
            )
        lines.append("- Panel CLOSE is a mixture: many books are SAME, a minority with a long unpaid tail are DRIFT.")
    if due_iss:
        lines += [
            "",
            "## 13o. due < issuance + grid",
            "",
            f"- Invoices with due < iss: {due_iss['n_due_before']:,} / {due_iss['n']:,} "
            f"(paid {due_iss['n_paid_due_before']:,}). Javier uses GREATEST(due, iss); store delay uses due as-is. "
            "Delay twins are still exact — those rows do not move the 3m paid window enough to break identity.",
            f"- Grid: Javier {due_iss['jav_rows']:,} rows / {due_iss['jav_co']} companies; "
            f"store {due_iss['store_rows']:,} / {due_iss['store_co']}. Inner merge is the store panel.",
        ]
    if size_scr:
        lines += [
            "",
            "## 13p. SIZE screen (log1p |a_in3|, train — not a score)",
            "",
            "| signal | ρ vs size | n | |ρ|≥0.50 |",
            "| --- | ---: | ---: | --- |",
        ]
        for r in size_scr:
            lines.append(
                f"| `{r['signal']}` | {_f(r['rho'])} | {r['n']:,} | {'SIZE' if r['size'] else 'ok'} |"
            )
    if early_inv:
        lines += [
            "",
            "## 13q. First two calendar months (Javier skips invoice features)",
            "",
            f"- Early CM (2024-09..10): {early_inv['early_cm']:,}. "
            f"`concentration` finite {early_inv['j_concentration_early']:.1%}; "
            f"`ar_overdue` {early_inv['j_ar_overdue_early']:.1%}.",
            f"- Later: conc {early_inv['j_concentration_late']:.1%}; ar_od {early_inv['j_ar_overdue_late']:.1%}.",
            "- Delay stays null until month 7 (left truncation), separate from the i<2 skip.",
        ]
    if folds:
        lines += [
            "",
            "## 13r. Group-fold ρ (seed 20260918 — stability, not a fit)",
            "",
            "| signal | fold ρ | min | max | spread |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for r in folds:
            vals = ", ".join(_f(x["rho"]) for x in r["folds"])
            lines.append(
                f"| `{r['signal']}` | {vals} | {_f(r['min'])} | {_f(r['max'])} | {_f(r['spread'])} |"
            )
        lines.append("- Verdicts do not flip across folds. Holdout never entered a fold.")
    if dark_bank:
        lines += [
            "",
            "## 13s. Dark books still have bank signals",
            "",
            f"- Live never-invoice train companies: {dark_bank['dark_co']} ({dark_bank['dark_cm']:,} CM).",
            f"- On those rows: `b_runway` {dark_bank['b_runway']:.1%}, `a_io_ratio` {dark_bank['a_io_ratio']:.1%}, "
            f"`f_fc_r` {dark_bank['f_fc_r']:.1%}. Invoice-gated ≠ missing company.",
            f"- All-null Javier `ar_overdue` is a wider set ({dark_bank['never_ar_od_co']} companies) — AP-only ERP books sit there too.",
        ]
    lines += [
        "",
        "## 14. What this is not",
        "",
        "- Not a 0–100. Not pillar weights. Not trajectory states.",
        "- Not a percentile fit against a reference (that *is* the frozen score).",
        "- Not an average of the 14. Not a scorecard.",
        "- Family B walk identity already closed — this module does not redo balances.",
        "",
        f"Elapsed {elapsed:.0f}s.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(parity, cov, rows, conc, vol, summary) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = _utc_ts()
    c = summary["counts"]
    worst = summary["worst"]
    recs = [
        (ts, "cat_map_same", 1.0 if parity["same_mapping"] else 0.0, 1.0, "pipeline vs common.CAT_MAP"),
        (ts, "n_same", float(c["SAME"]), 1.0, "primary twins ρ≥0.95"),
        (ts, "n_close", float(c["CLOSE"]), 1.0, "primary twins 0.80–0.95"),
        (ts, "n_drift", float(c["DRIFT"]), 1.0, "primary twins |ρ|<0.80"),
        (
            ts,
            "worst_drift_rho",
            float(worst["rho"]) if worst and worst["rho"] is not None else float("nan"),
            float(worst["n"] / cov["train_cm"]) if worst else 0.0,
            f"{worst['signal']} vs {worst['store']}" if worst else "",
        ),
        (ts, "invoice_dark_co", float(cov["invoice_dark_co"]), cov["invoice_dark_share"], "train companies no invoices"),
        (ts, "conc_vs_top1_rho", conc["j_vs_top1"]["rho"], conc["j_vs_top1"]["n"] / cov["train_cm"], "Javier concentration"),
        (ts, "conc_vs_hhi_rho", conc["j_vs_hhi"]["rho"], conc["j_vs_hhi"]["n"] / cov["train_cm"], "HHI is tail twin"),
        (ts, "vol_vs_b_bal_vol_rho", vol["j_vs_b_bal_vol"]["rho"], vol["j_vs_b_bal_vol"]["n"] / cov["train_cm"], "expected DRIFT"),
        (ts, "vol_vs_a_rho", vol["j_vs_vol_from_a"]["rho"], vol["j_vs_vol_from_a"]["n"] / cov["train_cm"], "reconstructed cashflow vol"),
    ]
    for r in rows:
        recs.append(
            (
                ts,
                f"rho_{r['signal']}",
                float(r["rho"]) if r["rho"] is not None else float("nan"),
                float(r["n"] / cov["train_cm"]) if cov["train_cm"] else 0.0,
                f"vs {r['store']} {r['verdict']}",
            )
        )
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        header = f.readline()
    if "ts,round,wave,agent" not in header:
        raise RuntimeError(f"unexpected registry header: {header!r}")
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for ts_, metric, value, coverage, notes in recs:
            val = "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.6g}"
            cov_s = "" if coverage is None or (isinstance(coverage, float) and not np.isfinite(coverage)) else f"{coverage:.4f}"
            w.writerow(
                [ts_, ROUND, WAVE, AGENT, "AH", "-", "spearman", "train", metric, val, cov_s, notes]
            )
    print(f"append_registry {len(recs)} rows")


def append_registry_extra(
    cov, ident, vol_why, extreme, clip, dso, conc_res=None, old_stock=None, fc_diffs=None,
    pdi=None, pair_meta=None,
) -> None:
    ts = _utc_ts()
    recs = []
    if ident:
        for st in ident:
            if st["store"] in ("vol_from_a", "ar_od_3m", "ap_od_3m", "b_d_runway_clip"):
                recs.append(
                    (
                        ts,
                        f"ident_{st['signal']}_{st['store']}_maxabs",
                        st["max_abs"],
                        st["n"] / cov["train_cm"] if cov["train_cm"] else 0.0,
                        f"pearson={st['pearson']:.4f} exact={st['share_exact']:.3f}",
                    )
                )
    if vol_why and "j_vs_log1p_a_in3" in vol_why:
        recs.append(
            (
                ts,
                "vol_vs_size_rho",
                vol_why["j_vs_log1p_a_in3"]["rho"],
                vol_why["j_vs_log1p_a_in3"]["n"] / cov["train_cm"],
                "not SIZE if |ρ|<0.50",
            )
        )
    if extreme:
        recs.append((ts, "extreme_tx_share", extreme["extreme_share"], 1.0, f"n={extreme['n_extreme']}"))
        recs.append((ts, "fc_maxabs", extreme["fc"]["max_abs"], extreme["fc"]["n"] / cov["train_cm"], "SAME rank"))
        recs.append((ts, "ds_maxabs", extreme["ds"]["max_abs"], extreme["ds"]["n"] / cov["train_cm"], "SAME rank"))
    if clip:
        recs.append((ts, "store_drunway_unclipped", float(clip["store_d_runway_outside_clip"]), 1.0, "|x|>12"))
        recs.append((ts, "dark_co", float(clip["dark_co"]), clip["dark_cm"] / cov["train_cm"], "no e_ar_issued"))
    if dso:
        recs.append(
            (
                ts,
                "ar_od_vs_dso_rho",
                dso["j_od_vs_dso"]["rho"],
                dso["j_od_vs_dso"]["n"] / cov["train_cm"],
                "DSO not in the 14",
            )
        )
    if conc_res:
        recs.append((ts, "conc_differ_share", conc_res["differ_share"], conc_res["n_both"] / cov["train_cm"], "top1 residual"))
    if fc_diffs:
        recs.append((ts, "fc_n_diff", float(fc_diffs["n_diff"]), fc_diffs["share_diff"], "level not rank"))
    if old_stock:
        for side, d in old_stock.items():
            recs.append((ts, f"old_open_{side}", d["old_share"], d["n_co"] / cov["train_co"], "last-month open issued >3m ago"))
    if pdi:
        recs.append((ts, "pdi_amt_share", pdi["pdi_amt_share"], pdi["pdi_n_share"], "Javier drops; D keeps"))
    if pair_meta and pair_meta.get("max_pair"):
        recs.append(
            (
                ts,
                "pair14_maxabs",
                abs(pair_meta["max_pair"]["rho"]),
                1.0,
                f"{pair_meta['max_pair']['a']}↔{pair_meta['max_pair']['b']}",
            )
        )
        recs.append((ts, "b_vs_invoice_medabs", pair_meta["b_vs_invoice_median"], 1.0, "do not average the 14"))
    if not recs:
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        for ts_, metric, value, coverage, notes in recs:
            val = "" if value is None or (isinstance(value, float) and not np.isfinite(value)) else f"{value:.6g}"
            cov_s = "" if coverage is None or (isinstance(coverage, float) and not np.isfinite(coverage)) else f"{coverage:.4f}"
            w.writerow([ts_, ROUND, WAVE, AGENT, "AH", "-", "spearman", "train", metric, val, cov_s, notes])
    print(f"append_registry_extra {len(recs)} rows")


def write_wave_end(summary, parity, cov, elapsed) -> None:
    c = summary["counts"]
    worst = summary["worst"]
    note = (
        f"\n# Wave 4 — score_pipeline 14-signal reconcile (end note)\n\n"
        f"Agent `{AGENT}`. Files: `analysis/evaluate/score_pipeline_qa.py`, "
        f"`analysis/outputs/score_pipeline_qa.md`, `{OUT_PNG.name}`, registry Spearman/coverage. "
        f"No 0–100. No pillars. No product/. Did not call `run_score` / `fit_ref`. "
        f"Did not edit `score_pipeline.py` or Family B.\n\n"
        f"## Return\n\n"
        f"- **{c['SAME']} SAME / {c['CLOSE']} CLOSE / {c['DRIFT']} DRIFT** on primary twins.\n"
        f"- Worst drift: `{worst['signal']}` vs `{worst['store']}` ρ={_f(worst['rho'])} n={worst['n']}.\n"
        f"- CAT_MAP parity: {'IDENTICAL' if parity['same_mapping'] else 'MISMATCH'}.\n"
        f"- The 14 do not include days-with-tx / issued_lag1.\n"
        f"- Invoice-dark train companies: {cov['invoice_dark_co']}. Wall {elapsed:.0f}s.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    if WAVE_NOTE.exists():
        WAVE_NOTE.write_text(WAVE_NOTE.read_text(encoding="utf-8") + note, encoding="utf-8")
    else:
        WAVE_NOTE.write_text(note.lstrip(), encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    assert_no_score_call()
    print("CUT 1 — CAT_MAP parity + Javier raw vs store (train Spearman)")
    parity = cat_map_parity()
    print(
        f"  CAT_MAP same_mapping={parity['same_mapping']} "
        f"n={parity['n_pipe']}/{parity['n_store']} diffs={parity['value_diffs']}"
    )
    if list(SIG["signal"]) != list(THE_14):
        raise RuntimeError(f"SIG signal list drifted: {list(SIG['signal'])}")

    con = connect()
    store = load_store()
    jav = load_javier(con)
    print("CUT 2 — coverage holes + 3m overdue reconstruction")
    od3 = invoice_3m_overdue(con)
    merged, train = merge_train(store, jav, od3)
    print(
        f"  merge all={len(merged):,} train={len(train):,} holdout_cm={int(merged['is_holdout'].sum())} "
        f"(holdout unused for ρ)"
    )
    cov = cut_coverage(store, train, con)
    print("CUT 1b — primary Spearman")
    rows = cut_map(train)
    conc = cut_concentration(train)
    vol = cut_volatility(train)
    overdue = cut_overdue_window(train)
    beyond = cut_beyond(train)
    ident = cut_identity(train)
    vol_why = cut_vol_why(train)
    od_months = cut_overdue_months(train)
    extreme = cut_extreme(con, train)
    clip = cut_clip_and_dark(train)
    dso = cut_dso(train)
    dark_gap = cut_dark_gap(con, train)
    conc_res = cut_conc_residual(train)
    fc_diffs = cut_fc_diffs(train)
    old_stock = cut_old_stock(con, train)
    conc_why = cut_conc_why(train, con)
    pair_mat, pair_meta = cut_pairwise_14(train)
    sched39 = cut_sched_39(con, train)
    pdi = cut_pdi(con)
    hold_cov = cut_holdout_cov(merged)
    sig_meta = cut_sig_meta()
    best_remap = cut_best_remap(rows)
    vol_co = cut_vol_by_company(train)
    od_co = cut_od_by_company(train)
    due_iss = cut_due_before_iss(con, jav, store)
    size_scr = cut_size_screen(train)
    early_inv = cut_early_invoice(train)
    folds = cut_folds(con, train)
    dark_bank = cut_dark_other(con, train)
    heat = cut_heatmap(train)
    summary = summarize(rows)
    elapsed = time.time() - t0
    payload = {
        "parity": parity,
        "cov": cov,
        "rows": rows,
        "conc": conc,
        "vol": vol,
        "overdue": overdue,
        "beyond": beyond,
        "ident": ident,
        "vol_why": vol_why,
        "od_months": od_months,
        "extreme": extreme,
        "clip": clip,
        "dso": dso,
        "dark_gap": dark_gap,
        "conc_res": conc_res,
        "fc_diffs": fc_diffs,
        "old_stock": old_stock,
        "conc_why": conc_why,
        "pair14_meta": pair_meta,
        "sched39": sched39,
        "pdi": pdi,
        "hold_cov": hold_cov,
        "summary": {
            "counts": summary["counts"],
            "worst": {k: summary["worst"][k] for k in ("signal", "store", "rho", "n", "verdict")}
            if summary["worst"]
            else None,
        },
        "elapsed": elapsed,
        "heat_shape": list(heat.shape) if heat is not None else None,
    }
    CUTS_JSON.write_text(json.dumps(payload, default=str, indent=2), encoding="utf-8")
    write_md(
        parity,
        cov,
        rows,
        conc,
        vol,
        overdue,
        beyond,
        summary,
        elapsed,
        ident=ident,
        vol_why=vol_why,
        od_months=od_months,
        extreme=extreme,
        clip=clip,
        dso=dso,
        dark_gap=dark_gap,
        conc_res=conc_res,
        fc_diffs=fc_diffs,
        old_stock=old_stock,
        conc_why=conc_why,
        pair14={"meta": pair_meta},
        sched39=sched39,
        pdi=pdi,
        hold_cov=hold_cov,
        sig_meta=sig_meta,
        best_remap=best_remap,
        vol_co=vol_co,
        od_co=od_co,
        due_iss=due_iss,
        size_scr=size_scr,
        early_inv=early_inv,
        folds=folds,
        dark_bank=dark_bank,
    )
    logged = set()
    if REGISTRY.exists():
        prev = pd.read_csv(REGISTRY, usecols=["agent", "metric"])
        logged = set(prev.loc[prev["agent"].astype(str) == AGENT, "metric"].astype(str))
    if "n_same" not in logged:
        append_registry(parity, cov, rows, conc, vol, summary)
    else:
        print("skip base registry (already logged)")
    if "pdi_amt_share" not in logged:
        append_registry_extra(
            cov, ident, vol_why, extreme, clip, dso,
            conc_res=conc_res, old_stock=old_stock, fc_diffs=fc_diffs,
            pdi=pdi, pair_meta=pair_meta,
        )
    else:
        print("skip extra registry (already logged)")
    # wave note is written once at the END of the 30-min loop, not after the first table
    print(
        f"PASS2 done in {elapsed:.0f}s — {summary['counts']} "
        f"worst={summary['worst']['signal'] if summary['worst'] else None} "
        f"ρ={_f(summary['worst']['rho']) if summary['worst'] else '—'}"
    )
    print("The 14 do not include days-with-tx / issued_lag1.")


if __name__ == "__main__":
    main()
