"""Y2 why — same inflow crash as Y4, or a different turn?

Brief: Q3 who is turning (82→68). Accepted label ``y2_neg_2of3`` only.
``y2_runway_lt1_sust`` and ``y2_onset_neg`` stay rejected — do not revive.
Labels from ``targets.parquet``. Do not run ``build_targets``.

X never family B (Y is the cash path). Night GBM PARK
(CV 0.540±0.117; fold 0 = 0.344). Do not fit another tree.
Single-feature / stratified rates only.

No 0–100. No product/. No parquet rewrite.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.y2_why
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import (
    FOLD_SEED,
    assert_no_holdout,
    auroc,
    group_folds,
    leakage_check,
    load_holdout,
    train_companies,
)
from analysis.features.common import ANALYSIS, DATA, connect
from analysis.targets.y11_dark import dark_population
from analysis.targets.y2_stress import META as Y2_META

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "y2_why.md"
OUT_PNG = ANALYSIS / "outputs" / "y2_why_quintiles.png"
AGENT = "1bb2643e"
WAVE = 4
ROUND = "R4"

Y_COL = "y2_neg_2of3"
Y4 = "y4_ds_r_double"
Y9 = "y9_fee_r_ownp80"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
NIGHT_GBM_CV = 0.540
NIGHT_GBM_SD = 0.117
NIGHT_FOLD0 = 0.344
SIZE_BAR = 0.60
CHANCE = 0.50
CLEAR_MARGIN = 0.02
CRASH_TH = 0.8
HHI_TAIL = 0.975
LEAK_RHO = 0.80
MIN_OWN_HIST = 6
OWN_P_HI = 0.80
SAMPLE_START = pd.Timestamp("2024-09-01")

# Legal X candidates. Never B.
STORE_X = (
    "a_io_ratio",
    "a_out6",
    "a_in3",
    "a_op_in",
    "c_n_days_with_tx",
    "c_zero_in_month",
    "d_cust_hhi",
    "f_ds_r",
)
# Label decomp + leak comparators only. Never scored as X.
STORE_B = ("b_liq", "b_runway", "b_below_0")
STORE_META = ("first_month", "group_id", "group_size", "h_group_size")
SINGLE_COLS = (
    "a_io_ratio",
    "a_out6",
    "log1p_a_in3",
    "c_n_days_with_tx",
    "d_cust_hhi",
    "f_ds_r",
    "c_zero_in_month",
)
QINT_COLS = (
    "a_io_ratio",
    "a_out6",
    "log1p_a_in3",
    "c_n_days_with_tx",
    "d_cust_hhi",
    "f_ds_r",
)
HI_COLS = ("a_out6", "d_cust_hhi")
FORBIDDEN = ("b",)
PLOT_COL = "c_n_days_with_tx"


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    if "first_month" in out.columns:
        out["first_month"] = pd.to_datetime(out["first_month"])
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}" if np.isfinite(v) else ""
    return "" if v is None else str(v)


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


def choose_sign(y: pd.Series, x: pd.Series) -> int:
    auc_p = auroc(y, x)
    auc_n = auroc(y, -x)
    if not np.isfinite(auc_p) and not np.isfinite(auc_n):
        return 1
    if not np.isfinite(auc_p):
        return -1
    if not np.isfinite(auc_n):
        return 1
    return -1 if auc_n > auc_p else 1


def spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 20 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def pearson(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 20 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="pearson"))


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
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


def add_own_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-company expanding p80 (months ≤ t, min 6 finite). Never pooled."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    for c in HI_COLS:
        x = pd.to_numeric(out[c], errors="coerce")
        p80 = x.groupby(out["company_id"], sort=False).transform(
            lambda s: _expanding_quantile_skipna(s, OWN_P_HI, MIN_OWN_HIST)
        )
        extra[f"{c}_ownp80"] = p80
        extra[f"{c}_hi"] = pd.Series(
            np.where(x.notna() & p80.notna(), (x > p80).astype(float), np.nan),
            index=out.index,
        )
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def add_label_story(df: pd.DataFrame) -> pd.DataFrame:
    """Future inflow crash + Y3-stressed-now. Story flags, not X."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    ain = pd.to_numeric(out["a_in3"], errors="coerce")
    ain_f3 = ain.groupby(out["company_id"], sort=False).shift(-3)
    extra = {
        "a_in3_f3": ain_f3,
        "in_ratio": ain_f3 / ain.replace(0, np.nan),
    }
    extra["inflow_crash"] = pd.Series(
        np.where(extra["in_ratio"].notna(), (extra["in_ratio"] < CRASH_TH).astype(float), np.nan),
        index=out.index,
    )
    liq = pd.to_numeric(out["b_liq"], errors="coerce")
    run = pd.to_numeric(out["b_runway"], errors="coerce")
    extra["y3_stressed_now"] = pd.Series(
        np.where(liq.notna() | run.notna(), ((liq < 0) | (run < 1)).astype(float), np.nan),
        index=out.index,
    )
    extra["y3_labeled"] = pd.to_numeric(out[Y3], errors="coerce").notna().astype(float)
    extra["hhi_tail"] = pd.Series(
        np.where(
            pd.to_numeric(out["d_cust_hhi"], errors="coerce").notna(),
            (pd.to_numeric(out["d_cust_hhi"], errors="coerce") > HHI_TAIL).astype(float),
            np.nan,
        ),
        index=out.index,
    )
    extra["log1p_a_in3"] = np.log1p(pd.to_numeric(out["a_in3"], errors="coerce").abs())
    extra["so_far"] = (
        (out["period"].dt.year - out["first_month"].dt.year) * 12
        + (out["period"].dt.month - out["first_month"].dt.month)
        + 1
    ).astype(float)
    extra["late_arrival"] = (out["first_month"] > SAMPLE_START).astype(float)
    extra["months_on_book"] = (
        (2026 - out["first_month"].dt.year) * 12
        + (8 - out["first_month"].dt.month)
        + 1
    ).astype(float)
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def coverage(x: pd.Series, mask: pd.Series) -> dict:
    n = int(mask.sum())
    n_ok = int(x[mask].notna().sum())
    return {"n": n, "n_defined": n_ok, "coverage": (n_ok / n) if n else float("nan")}


def signed_oof_auroc(
    df: pd.DataFrame,
    col: str,
    train_lab: pd.Series,
    folds: np.ndarray,
) -> dict:
    """Group-fold CV AUROC. Sign from the train fold only. Never B as X."""
    if str(col).startswith("b_"):
        raise RuntimeError(f"family B leaked into a single: {col}")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    x = pd.to_numeric(df[col], errors="coerce")
    fold_rows = []
    aucs = []
    for k in range(N_FOLDS):
        tr = train_lab & (folds != k)
        va = train_lab & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        n_va = int((va & x.notna() & y.notna()).sum())
        n_pos = int((va & x.notna() & (y == 1)).sum())
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va_defined": n_va,
                "n_pos_defined": n_pos,
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = int(choose_sign(y[train_lab], x[train_lab]))
    return {
        "col": col,
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": tr_sign,
        "train_auc": float(auroc(y[train_lab], tr_sign * x[train_lab])),
    }


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    raw = pd.read_parquet(STORE)
    need = ["company_id", "period", *STORE_X, *STORE_B, *STORE_META]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"store missing {missing}")
    panel = _keys(raw[need])
    print(f"loaded store {STORE} shape={panel.shape} (B is decomp/leak only)")
    return panel


def load_y() -> pd.DataFrame:
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(TARGETS)
    need = ["company_id", "period", Y_COL, Y4, Y9, Y3]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"targets.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"Y from {TARGETS} shape={panel.shape} (no build_targets)")
    return panel


def pos_overlap(lab: pd.DataFrame, flag: str) -> dict:
    pos = lab[Y_COL] == 1
    n_pos = int(pos.sum())
    f = pd.to_numeric(lab[flag], errors="coerce")
    defined = pos & f.notna()
    n_def = int(defined.sum())
    n_hi = int((defined & (f == 1)).sum())
    return {
        "flag": flag,
        "n_pos": n_pos,
        "n_defined": n_def,
        "n_hi": n_hi,
        "share_of_pos_defined": (n_hi / n_def) if n_def else float("nan"),
        "share_of_pos_all": (n_hi / n_pos) if n_pos else float("nan"),
        "coverage_of_pos": (n_def / n_pos) if n_pos else float("nan"),
    }


def two_by_two(lab: pd.DataFrame, a: str, b: str, mask: pd.Series) -> dict:
    aa = pd.to_numeric(lab[a], errors="coerce")
    bb = pd.to_numeric(lab[b], errors="coerce")
    m = mask & aa.notna() & bb.notna()
    both = int((m & (aa == 1) & (bb == 1)).sum())
    a_only = int((m & (aa == 1) & (bb == 0)).sum())
    b_only = int((m & (aa == 0) & (bb == 1)).sum())
    neither = int((m & (aa == 0) & (bb == 0)).sum())
    n = int(m.sum())
    n_a = both + a_only
    n_b = both + b_only
    return {
        "a": a,
        "b": b,
        "n": n,
        "both": both,
        "a_only": a_only,
        "b_only": b_only,
        "neither": neither,
        "share_a_that_are_b": (both / n_a) if n_a else float("nan"),
        "share_b_that_are_a": (both / n_b) if n_b else float("nan"),
        "jaccard": (both / (n_a + n_b - both)) if (n_a + n_b - both) else float("nan"),
    }


def pass1_decompose(lab: pd.DataFrame) -> dict:
    """Share of Y2 positives that are Y4 / Y9 / stressed-now / crash / HHI / zero-in."""
    print("\n" + "=" * 72)
    print(f"PASS 1 — decompose {Y_COL} positives (train labeled)")
    print("inflow_crash = in3[t+3]/in3[t] < 0.8 — future A, label story, not X")
    print("y3_stressed_now = b_liq<0 OR b_runway<1 — B for decomp only, not X")
    print("high a_out6 / d_cust_hhi = company own expanding p80")
    print("=" * 72)

    n = int(lab[Y_COL].notna().sum())
    n_pos = int((lab[Y_COL] == 1).sum())
    rate = float(lab.loc[lab[Y_COL].notna(), Y_COL].mean()) if n else float("nan")
    print(f"{Y_COL}: n={n} pos={n_pos} rate={rate:.4f}")

    flags = (
        Y4,
        Y9,
        "y3_stressed_now",
        "y3_labeled",
        "a_out6_hi",
        "inflow_crash",
        "d_cust_hhi_hi",
        "hhi_tail",
        "c_zero_in_month",
        "b_below_0",
    )
    rows = []
    for flag in flags:
        if flag not in lab.columns:
            print(f"  missing {flag}")
            continue
        rec = pos_overlap(lab, flag)
        rows.append(rec)
        print(
            f"  {flag:22s}  {rec['n_hi']:4d}/{rec['n_defined']:4d} defined pos  "
            f"share={rec['share_of_pos_defined']:.3f}  cov={rec['coverage_of_pos']:.3f}"
        )

    pos = lab[Y_COL] == 1
    twos = {}
    for a, b, sl, name in (
        (Y_COL, "inflow_crash", lab[Y_COL].notna() & lab["inflow_crash"].notna(), "y2_x_crash"),
        (Y_COL, Y4, lab[Y_COL].notna() & lab[Y4].notna(), "y2_x_y4"),
        (Y_COL, Y9, lab[Y_COL].notna() & lab[Y9].notna(), "y2_x_y9"),
        ("inflow_crash", Y4, lab["inflow_crash"].notna() & lab[Y4].notna(), "crash_x_y4"),
    ):
        t = two_by_two(lab, a, b, sl)
        t["name"] = name
        twos[name] = t
        print(
            f"  2x2 {name}: n={t['n']} both={t['both']} a_only={t['a_only']} "
            f"b_only={t['b_only']} neither={t['neither']}  "
            f"share_a_in_b={t['share_a_that_are_b']:.3f} "
            f"share_b_in_a={t['share_b_that_are_a']:.3f} jaccard={t['jaccard']:.3f}"
        )

    crash_of_y2 = next(r["share_of_pos_defined"] for r in rows if r["flag"] == "inflow_crash")
    y4_of_y2 = next(r["share_of_pos_defined"] for r in rows if r["flag"] == Y4)
    y2_of_y4 = twos["y2_x_y4"]["share_b_that_are_a"]
    same_crash = bool(
        np.isfinite(crash_of_y2)
        and crash_of_y2 >= 0.70
        and np.isfinite(y4_of_y2)
        and y4_of_y2 >= 0.50
    )
    print(
        f"  SAME_CRASH_AS_Y4={same_crash}  "
        f"Y2pos crash={crash_of_y2:.3f} Y2pos that are Y4={y4_of_y2:.3f} "
        f"Y4pos that are Y2={y2_of_y4:.3f}"
    )
    pos_med = float(lab.loc[pos, "in_ratio"].median()) if int(pos.sum()) else float("nan")
    neg_med = float(lab.loc[lab[Y_COL] == 0, "in_ratio"].median())
    print(f"  median in_ratio pos={pos_med:.3f} neg={neg_med:.3f}")
    return {
        "n": n,
        "n_pos": n_pos,
        "rate": rate,
        "rows": rows,
        "twos": twos,
        "same_crash_as_y4": same_crash,
        "pos_med_in_ratio": pos_med,
        "neg_med_in_ratio": neg_med,
        "crash_of_y2": crash_of_y2,
        "y4_of_y2": y4_of_y2,
        "y2_of_y4": y2_of_y4,
    }


def quintile_table(df: pd.DataFrame, mask: pd.Series, x_col: str) -> dict:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    m = mask & x.notna() & y.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    tr = pd.DataFrame({x_col: x[m], Y_COL: y[m]})
    if len(tr) < 50 or tr[x_col].nunique() < 3:
        return {
            "x": x_col,
            "n": int(len(tr)),
            "rows": [],
            "n_bins": 0,
            "monotone_up": None,
            "monotone_down": None,
            "tail_only": None,
            "head_only": None,
            "bins": [],
        }
    cats, bins = pd.qcut(tr[x_col], 5, retbins=True, duplicates="drop")
    tr = tr.copy()
    tr["q"] = cats
    rows = []
    for i, (q, g) in enumerate(tr.groupby("q", observed=True), start=1):
        rows.append(
            {
                "q": i,
                "interval": str(q),
                "lo": float(q.left),
                "hi": float(q.right),
                "n": int(len(g)),
                "n_pos": int((g[Y_COL] == 1).sum()),
                "y_rate": float(g[Y_COL].mean()),
                "x_median": float(g[x_col].median()),
            }
        )
    rates = [r["y_rate"] for r in rows]
    monotone_up = all(a <= b + 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    monotone_down = all(a >= b - 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    if len(rates) >= 3:
        body = rates[:-1]
        tail_only = rates[-1] > max(body) + 0.04 and max(body) - min(body) < 0.06
        mid = rates[1:]
        head_only = rates[0] > max(mid) + 0.04 and max(mid) - min(mid) < 0.06
    else:
        tail_only = None
        head_only = None
    shape = "flat"
    if tail_only:
        shape = "tail"
    elif head_only:
        shape = "head"
    elif monotone_up:
        shape = "monotone_up"
    elif monotone_down:
        shape = "monotone_down"
    return {
        "x": x_col,
        "n": int(len(tr)),
        "rows": rows,
        "n_bins": len(rows),
        "monotone_up": monotone_up,
        "monotone_down": monotone_down,
        "tail_only": tail_only,
        "head_only": head_only,
        "shape": shape,
        "bins": [float(b) for b in bins],
    }


def pass2_quintiles(df: pd.DataFrame, train_lab: pd.Series, no_plot: bool) -> dict:
    print("\n" + "=" * 72)
    print("PASS 2 — train-only quintiles vs Y2 rate (cuts never see holdout)")
    print("=" * 72)
    tables = []
    for col in QINT_COLS:
        if col not in df.columns:
            print(f"  missing {col}")
            continue
        tab = quintile_table(df, train_lab, col)
        tables.append(tab)
        print(
            f"\n  {col}: n={tab['n']} bins={tab['n_bins']} shape={tab['shape']} "
            f"mono↑={tab['monotone_up']} mono↓={tab['monotone_down']} tail={tab['tail_only']}"
        )
        for r in tab["rows"]:
            print(
                f"    Q{r['q']} {r['interval']}  n={r['n']:5d} pos={r['n_pos']:4d}  "
                f"P(Y=1)={r['y_rate']:.3f}  med={r['x_median']:.4g}"
            )

    png = None
    plot = next((t for t in tables if t["x"] == PLOT_COL and t["rows"]), None)
    if plot and HAS_MPL and not no_plot:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        xs = [r["q"] for r in plot["rows"]]
        ys = [r["y_rate"] for r in plot["rows"]]
        ax.bar(xs, ys, color="#3d5a80", width=0.7)
        base = float(df.loc[train_lab, Y_COL].mean())
        ax.axhline(base, color="#ee6c4d", ls="--", lw=1, label="train labeled base")
        for r in plot["rows"]:
            ax.text(
                r["q"],
                r["y_rate"] + 0.002,
                f"{r['y_rate']:.1%}\nn={r['n']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Q{r['q']}\n{r['x_median']:.2g}" for r in plot["rows"]])
        ax.set_ylabel("P(Y=1)  y2_neg_2of3")
        ax.set_xlabel(f"{PLOT_COL} quintile (train labeled cuts)")
        ax.set_ylim(0, max(ys) * 1.32 if ys else 1)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Y2 rate by days with a transaction")
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
        png = str(OUT_PNG)
        print(f"  wrote {OUT_PNG}")
    elif not HAS_MPL:
        print("  matplotlib missing — skip PNG")
    return {"tables": tables, "png": png}


def pass3_singles(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    print("\n" + "=" * 72)
    print("PASS 3 — single-feature group-fold AUROC vs Y2")
    print(f"bar to explain (not beat with a tree): night GBM {NIGHT_GBM_CV:.3f}±{NIGHT_GBM_SD:.3f}")
    print(f"size ≥ {SIZE_BAR:.2f} → PARK as X; KEEP Q5 if non-B beats size by ≥{CLEAR_MARGIN:.2f}")
    print("=" * 72)
    leak = leakage_check(list(SINGLE_COLS), Y_COL, forbidden_prefixes=FORBIDDEN)
    if not leak["ok"]:
        raise RuntimeError(f"X leak: {leak['issues']}")
    folds = df["fold"].to_numpy()
    size = pd.to_numeric(df["log1p_a_in3"], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    size_auc = float(auroc(y[train_lab], size[train_lab]))
    size_auc_max = max(size_auc, 1.0 - size_auc) if np.isfinite(size_auc) else float("nan")
    print(f"  size log1p(|a_in3|) train AUROC={size_auc:.4f} two-sided={size_auc_max:.4f}")

    rows = []
    for col in SINGLE_COLS:
        if col not in df.columns:
            print(f"  missing {col}")
            continue
        if str(col).startswith("b_"):
            raise RuntimeError(f"B in singles: {col}")
        cov = coverage(pd.to_numeric(df[col], errors="coerce"), train_lab)
        oof = signed_oof_auroc(df, col, train_lab, folds)
        x = pd.to_numeric(df[col], errors="coerce")
        rho_size = spearman(x[train_lab], size[train_lab])
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "train_sign": oof["train_sign"],
            "coverage": cov["coverage"],
            "n_defined": cov["n_defined"],
            "size_auroc": size_auc,
            "size_rho": rho_size,
            "folds": oof["folds"],
            "gap_vs_size": (
                float(oof["cv"] - size_auc_max)
                if np.isfinite(oof["cv"]) and np.isfinite(size_auc_max)
                else float("nan")
            ),
            "gap_vs_gbm": (
                float(oof["cv"] - NIGHT_GBM_CV) if np.isfinite(oof["cv"]) else float("nan")
            ),
            "size_proxy": bool(np.isfinite(size_auc_max) and size_auc_max >= SIZE_BAR and col == "log1p_a_in3"),
        }
        rec["park_size"] = bool(col == "log1p_a_in3" and rec["size_proxy"])
        rows.append(rec)
        bits = " ".join(f"{f['auroc']:.3f}" for f in oof["folds"])
        print(
            f"  {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"train={oof['train_auc']:.4f} sign={oof['train_sign']:+d}  "
            f"cov={cov['coverage']:.3f}  folds {bits}  "
            f"gap_size={rec['gap_vs_size']:+.3f} gap_gbm={rec['gap_vs_gbm']:+.3f}"
        )

    legal = [r for r in rows if r["col"] != "log1p_a_in3" and np.isfinite(r["cv"])]
    best = max(legal, key=lambda r: r["cv"]) if legal else None
    if best:
        print(
            f"  BEST legal single {best['col']} CV={best['cv']:.4f} "
            f"vs size {size_auc_max:.4f} vs GBM {NIGHT_GBM_CV:.3f}"
        )
    return {
        "rows": rows,
        "best": best,
        "size_auc": size_auc,
        "size_auc_max": size_auc_max,
        "n_train": int(train_lab.sum()),
    }


def pass4_fold0(df: pd.DataFrame, train_lab: pd.Series, dark_ids: set[str]) -> dict:
    """Which groups sit in fold 0 — late-arrivals / all-dark / monopoly?"""
    print("\n" + "=" * 72)
    print(f"PASS 4 — fold 0 death (night GBM fold0={NIGHT_FOLD0:.3f})")
    print("protocol group folds, seed 20260918")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    is_dark = df["company_id"].astype(str).isin(dark_ids)
    hhi = pd.to_numeric(df["d_cust_hhi"], errors="coerce")
    rows = []
    for k in range(N_FOLDS):
        sl = train_lab & (df["fold"] == k)
        cos = df.loc[sl, "company_id"]
        gids = df.loc[sl, "group_id"]
        late = df.loc[sl, "late_arrival"] == 1
        short = pd.to_numeric(df.loc[sl, "months_on_book"], errors="coerce") < 12
        dark = is_dark[sl]
        # company-level monopoly: any labeled month with HHI>0.975
        tail_cos = set(df.loc[sl & (hhi > HHI_TAIL), "company_id"].astype(str))
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "n_cos": int(cos.nunique()),
            "n_groups": int(gids.nunique()),
            "share_late_cm": float(late.mean()) if int(sl.sum()) else float("nan"),
            "share_short_cm": float(short.mean()) if int(sl.sum()) else float("nan"),
            "share_dark_cm": float(dark.mean()) if int(sl.sum()) else float("nan"),
            "n_cos_late": int(df.loc[sl & late, "company_id"].nunique()),
            "n_cos_short": int(df.loc[sl & short, "company_id"].nunique()),
            "n_cos_dark": int(df.loc[sl & dark, "company_id"].nunique()),
            "n_cos_hhi_tail": int(len(tail_cos)),
            "median_group_size": float(pd.to_numeric(df.loc[sl, "group_size"], errors="coerce").median()),
            "median_months_on_book": float(
                pd.to_numeric(df.loc[sl, "months_on_book"], errors="coerce").median()
            ),
            "mean_log1p_in3": float(pd.to_numeric(df.loc[sl, "log1p_a_in3"], errors="coerce").mean()),
        }
        rec["share_cos_late"] = rec["n_cos_late"] / rec["n_cos"] if rec["n_cos"] else float("nan")
        rec["share_cos_short"] = rec["n_cos_short"] / rec["n_cos"] if rec["n_cos"] else float("nan")
        rec["share_cos_dark"] = rec["n_cos_dark"] / rec["n_cos"] if rec["n_cos"] else float("nan")
        rec["share_cos_hhi_tail"] = rec["n_cos_hhi_tail"] / rec["n_cos"] if rec["n_cos"] else float("nan")
        rows.append(rec)
        print(
            f"  fold {k}: n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.4f}  "
            f"cos={rec['n_cos']:4d} groups={rec['n_groups']:3d}  "
            f"late_cos={rec['share_cos_late']:.3f} short_cos={rec['share_cos_short']:.3f} "
            f"dark_cos={rec['share_cos_dark']:.3f} hhi_tail_cos={rec['share_cos_hhi_tail']:.3f} "
            f"med_gsz={rec['median_group_size']:.1f} med_book={rec['median_months_on_book']:.1f} "
            f"mean_size={rec['mean_log1p_in3']:.2f}"
        )

    f0 = rows[0]
    others = [r for r in rows if r["fold"] != 0]
    note = (
        f"fold0 rate {f0['rate']:.3f} vs rest "
        f"{np.mean([r['rate'] for r in others]):.3f}; "
        f"late {f0['share_cos_late']:.3f} vs {np.mean([r['share_cos_late'] for r in others]):.3f}; "
        f"short {f0['share_cos_short']:.3f} vs {np.mean([r['share_cos_short'] for r in others]):.3f}; "
        f"dark {f0['share_cos_dark']:.3f} vs {np.mean([r['share_cos_dark'] for r in others]):.3f}; "
        f"monopoly {f0['share_cos_hhi_tail']:.3f} vs {np.mean([r['share_cos_hhi_tail'] for r in others]):.3f}; "
        f"med group_size {f0['median_group_size']:.1f} vs {np.mean([r['median_group_size'] for r in others]):.1f}; "
        f"med months_on_book {f0['median_months_on_book']:.1f} vs {np.mean([r['median_months_on_book'] for r in others]):.1f}"
    )
    print("  " + note)
    return {"rows": rows, "fold0": f0, "note": note}


def pass5_leak(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    print("\n" + "=" * 72)
    print("PASS 5 — leak vs b_liq / b_runway / b_below_0  (fail |ρ|≥0.80 = B-copy)")
    print("comparators only — never X")
    print("=" * 72)
    rows = []
    for col in SINGLE_COLS:
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        for vs in STORE_B:
            other = pd.to_numeric(df[vs], errors="coerce")
            m = train_lab & x.notna() & other.notna()
            assert_no_holdout(df.loc[m, "company_id"])
            rec = {
                "col": col,
                "vs": vs,
                "n": int(m.sum()),
                "spearman": spearman(x[m], other[m]),
                "pearson": pearson(x[m], other[m]),
            }
            rec["fail"] = bool(np.isfinite(rec["spearman"]) and abs(rec["spearman"]) >= LEAK_RHO)
            rows.append(rec)
            mark = " FAIL" if rec["fail"] else ""
            print(
                f"  {col:22s} vs {vs:12s}  ρ_s={rec['spearman']:+.3f}  "
                f"ρ_p={rec['pearson']:+.3f}  n={rec['n']}{mark}"
            )
    fails = [r for r in rows if r["fail"]]
    print(f"  FAIL count={len(fails)} (illegal X if any legal single is a B-copy)")
    return {"rows": rows, "fails": fails}


def pass6_dark(df: pd.DataFrame, train_lab: pd.Series, pop: dict) -> dict:
    print("\n" + "=" * 72)
    print("PASS 6 — dark 470 vs invoiced 744  (confirm 9.14% vs 6.34%)")
    print("Not a reason for a new Y. Never B as X.")
    print("=" * 72)
    dark_ids = pop["train_dark_ids"]
    is_dark = df["company_id"].astype(str).isin(dark_ids)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    size = pd.to_numeric(df["log1p_a_in3"], errors="coerce")
    rows = []
    for name, sl in (
        ("train_all", train_lab),
        ("dark_470", train_lab & is_dark),
        ("invoiced_744", train_lab & ~is_dark),
    ):
        yy = y[sl]
        rec = {
            "slice": name,
            "n": int(yy.notna().sum()),
            "n_pos": int((yy == 1).sum()),
            "n_cos": int(df.loc[sl & yy.notna(), "company_id"].nunique()),
            "rate": float(yy.mean()) if int(yy.notna().sum()) else float("nan"),
            "size_auroc": float(auroc(yy, size[sl])),
        }
        rows.append(rec)
        print(
            f"  {name:14s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"cos={rec['n_cos']:4d} rate={rec['rate']:.4f} size={rec['size_auroc']:.3f}"
        )
    dark = next(r for r in rows if r["slice"] == "dark_470")
    inv = next(r for r in rows if r["slice"] == "invoiced_744")
    confirm = bool(
        abs(dark["rate"] - 0.0914) < 0.005
        and abs(inv["rate"] - 0.0634) < 0.005
        and pop.get("confirm_470")
    )
    print(
        f"  CONFIRM 9.14 vs 6.34: {confirm}  "
        f"dark={dark['rate']:.4f} inv={inv['rate']:.4f} "
        f"n_dark_cos={pop['n_train_dark']} n_book={pop['n_book_train']}"
    )
    return {
        "rows": rows,
        "confirm": confirm,
        "n_dark": pop["n_train_dark"],
        "n_book": pop["n_book_train"],
        "confirm_470": pop.get("confirm_470"),
    }


def pass7_lag1_q6(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Honest 1-month Q6. Only lag-1. Hidden-72 cannot carry longer leads."""
    print("\n" + "=" * 72)
    print("CUT 7 — honest 1-month Q6  a_io_ratio / c_n_days_with_tx  lag 0 vs 1")
    print("Do not claim Q6 beyond 1-month. Never B.")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    size = pd.to_numeric(df["log1p_a_in3"], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    size_auc = float(auroc(y[train_lab], size[train_lab]))
    rows = []
    for stem in ("a_io_ratio", "c_n_days_with_tx"):
        for lag, col in ((0, stem), (1, f"{stem}_lag1")):
            if col not in df.columns:
                print(f"  missing {col}")
                continue
            cov = coverage(pd.to_numeric(df[col], errors="coerce"), train_lab)
            oof = signed_oof_auroc(df, col, train_lab, folds)
            rec = {
                "stem": stem,
                "lag": lag,
                "col": col,
                "cv": oof["cv"],
                "sd": oof["sd"],
                "train_auc": oof["train_auc"],
                "train_sign": oof["train_sign"],
                "coverage": cov["coverage"],
                "n_defined": cov["n_defined"],
                "folds": oof["folds"],
                "gap_vs_size": (
                    float(oof["cv"] - size_auc) if np.isfinite(oof["cv"]) else float("nan")
                ),
            }
            rows.append(rec)
            bits = " ".join(f"{f['auroc']:.3f}" for f in oof["folds"])
            print(
                f"  {col:24s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f} sign={oof['train_sign']:+d}  "
                f"cov={cov['coverage']:.3f}  folds {bits}  gap_size={rec['gap_vs_size']:+.3f}"
            )
    io1 = next((r for r in rows if r["col"] == "a_io_ratio_lag1"), None)
    d1 = next((r for r in rows if r["col"] == "c_n_days_with_tx_lag1"), None)
    d0 = next((r for r in rows if r["col"] == "c_n_days_with_tx"), None)
    lag_lift = (
        float(d1["cv"] - d0["cv"])
        if d1 and d0 and np.isfinite(d1["cv"]) and np.isfinite(d0["cv"])
        else float("nan")
    )
    # A lead needs lag1 to beat lag0, not just to copy the contemporaneous bump.
    keep_q6 = bool(
        d1
        and np.isfinite(d1["gap_vs_size"])
        and d1["gap_vs_size"] >= CLEAR_MARGIN
        and d1["cv"] < SIZE_BAR
        and np.isfinite(lag_lift)
        and lag_lift >= 0.01
    )
    io_cv = float("nan") if not io1 else io1["cv"]
    d_cv = float("nan") if not d1 else d1["cv"]
    print(
        f"  Q6 KEEP lag1 days? {keep_q6}  io_lag1={io_cv:.3f}  "
        f"days_lag1={d_cv:.3f}  lag_lift={lag_lift:+.3f}"
    )
    return {
        "rows": rows,
        "keep_q6": keep_q6,
        "io_lag1": io1,
        "days_lag1": d1,
        "days_lag0": d0,
        "lag_lift": lag_lift,
        "size_auc": size_auc,
    }


def _sofar_bucket(s: pd.Series) -> pd.Series:
    """Fixed trail cuts 6/12/18/24 — same as trail_length.md, not quantiles."""
    x = pd.to_numeric(s, errors="coerce")
    return pd.cut(
        x,
        bins=[0, 5, 11, 17, 23, 100],
        labels=["<6", "6-11", "12-17", "18-23", "24"],
        include_lowest=True,
    )


def pass8_fold0_trail_size(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Is fold-0 death a group-size or late-trail effect? Read trail_length, no new hist."""
    print("\n" + "=" * 72)
    print("CUT 8 — fold 0 × so-far (fixed cuts) × group_size terciles")
    print("trail_length.md already has the histogram — do not redo it")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    sofar = pd.to_numeric(df["so_far"], errors="coerce")
    gsz = pd.to_numeric(df["group_size"], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    # tercile cuts from train labeled only
    tr_g = gsz[train_lab].dropna()
    _, gbins = pd.qcut(tr_g, 3, retbins=True, duplicates="drop")
    gbin = pd.cut(gsz, bins=gbins, include_lowest=True)
    so_b = _sofar_bucket(sofar)

    fold_rows = []
    for k in range(N_FOLDS):
        sl = train_lab & (df["fold"] == k)
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "share_below0": float(below[sl].mean()) if int(sl.sum()) else float("nan"),
            "share_sofar_lt12": float((sofar[sl] < 12).mean()) if int(sl.sum()) else float("nan"),
            "share_sofar_ge18": float((sofar[sl] >= 18).mean()) if int(sl.sum()) else float("nan"),
            "median_sofar": float(sofar[sl].median()) if int(sl.sum()) else float("nan"),
        }
        fold_rows.append(rec)
        print(
            f"  fold {k}: rate={rec['rate']:.4f} below0={rec['share_below0']:.3f}  "
            f"sofar<12={rec['share_sofar_lt12']:.3f} sofar>=18={rec['share_sofar_ge18']:.3f} "
            f"med_sofar={rec['median_sofar']:.1f}"
        )

    so_rows = []
    print("  Y2 rate by so-far bucket × fold (company-month so-far, not company-total):")
    for b in ["<6", "6-11", "12-17", "18-23", "24"]:
        for k in range(N_FOLDS):
            sl = train_lab & (df["fold"] == k) & (so_b == b)
            rec = {
                "bucket": str(b),
                "fold": k,
                "n": int(sl.sum()),
                "n_pos": int((y[sl] == 1).sum()),
                "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            }
            so_rows.append(rec)
        all_sl = train_lab & (so_b == b)
        print(
            f"    {b:6s} all n={int(all_sl.sum()):5d} rate={float(y[all_sl].mean()) if int(all_sl.sum()) else float('nan'):.3f}  "
            + " ".join(
                f"f{k}={next(r['rate'] for r in so_rows if r['bucket']==b and r['fold']==k):.3f}"
                if int(next(r['n'] for r in so_rows if r['bucket']==b and r['fold']==k))
                else f"f{k}=."
                for k in range(N_FOLDS)
            )
        )

    g_rows = []
    print("  Y2 rate by group_size tercile (train cuts) × fold:")
    cats = list(gbin.cat.categories)
    for i, cat in enumerate(cats, start=1):
        for k in range(N_FOLDS):
            sl = train_lab & (df["fold"] == k) & (gbin == cat)
            rec = {
                "tercile": i,
                "interval": str(cat),
                "fold": k,
                "n": int(sl.sum()),
                "n_pos": int((y[sl] == 1).sum()),
                "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            }
            g_rows.append(rec)
        all_sl = train_lab & (gbin == cat)
        print(
            f"    T{i} {cat} all n={int(all_sl.sum()):5d} rate={float(y[all_sl].mean()):.3f}  "
            + " ".join(
                f"f{k}={next(r['rate'] for r in g_rows if r['tercile']==i and r['fold']==k):.3f}"
                for k in range(N_FOLDS)
            )
        )

    # Does fold0's 11% rate survive inside long trail and each size tercile?
    f0_long = train_lab & (df["fold"] == 0) & (sofar >= 18)
    rest_long = train_lab & (df["fold"] != 0) & (sofar >= 18)
    f0_short = train_lab & (df["fold"] == 0) & (sofar < 12)
    rest_short = train_lab & (df["fold"] != 0) & (sofar < 12)
    residual = {
        "f0_long_rate": float(y[f0_long].mean()) if int(f0_long.sum()) else float("nan"),
        "rest_long_rate": float(y[rest_long].mean()) if int(rest_long.sum()) else float("nan"),
        "f0_short_rate": float(y[f0_short].mean()) if int(f0_short.sum()) else float("nan"),
        "rest_short_rate": float(y[rest_short].mean()) if int(rest_short.sum()) else float("nan"),
        "n_f0_long": int(f0_long.sum()),
        "n_f0_short": int(f0_short.sum()),
    }
    residual["long_gap"] = residual["f0_long_rate"] - residual["rest_long_rate"]
    residual["short_gap"] = residual["f0_short_rate"] - residual["rest_short_rate"]
    print(
        f"  fold0 vs rest INSIDE long>=18: {residual['f0_long_rate']:.3f} vs "
        f"{residual['rest_long_rate']:.3f} gap={residual['long_gap']:+.3f} n_f0={residual['n_f0_long']}"
    )
    print(
        f"  fold0 vs rest INSIDE short<12: {residual['f0_short_rate']:.3f} vs "
        f"{residual['rest_short_rate']:.3f} gap={residual['short_gap']:+.3f} n_f0={residual['n_f0_short']}"
    )
    late_trail = bool(np.isfinite(residual["short_gap"]) and residual["short_gap"] >= 0.04 and residual["long_gap"] < 0.02)
    size_effect = False
    f0_t_rates = []
    rest_t_rates = []
    for i in range(1, 4):
        f0r = [r["rate"] for r in g_rows if r["tercile"] == i and r["fold"] == 0]
        rstr = [r["rate"] for r in g_rows if r["tercile"] == i and r["fold"] != 0]
        if f0r and rstr and np.isfinite(f0r[0]):
            f0_t_rates.append(f0r[0])
            rest_t_rates.append(float(np.nanmean(rstr)))
    if f0_t_rates:
        gaps = [a - b for a, b in zip(f0_t_rates, rest_t_rates)]
        size_effect = bool(max(gaps) - min(gaps) >= 0.06 and min(gaps) < 0.02)
        print(f"  fold0−rest by size tercile: " + " ".join(f"T{i+1}={g:+.3f}" for i, g in enumerate(gaps)))
    print(f"  late_trail_effect={late_trail}  size_tercile_effect={size_effect}")
    return {
        "fold_rows": fold_rows,
        "so_rows": so_rows,
        "g_rows": g_rows,
        "residual": residual,
        "late_trail_effect": late_trail,
        "size_effect": size_effect,
    }


def pass9_already_neg(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Y2 among already-neg vs clean-now. B decomp of the label, never X."""
    print("\n" + "=" * 72)
    print("CUT 9 — already b_below_0 vs clean-now (label persistence, not X)")
    print("81% of Y2 pos were already neg — is the leftover a turn?")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    folds = df["fold"].to_numpy()
    rows = []
    for name, sl in (
        ("already_neg", train_lab & (below == 1)),
        ("clean_now", train_lab & (below == 0)),
        ("below_unknown", train_lab & below.isna()),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "n_cos": int(df.loc[sl, "company_id"].nunique()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {name:14s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"cos={rec['n_cos']:4d} rate={rec['rate']:.4f}"
        )
    # singles on the clean-now leftover only (legal X, never B)
    leftover = train_lab & (below == 0)
    singles = []
    print("  legal singles on clean-now leftover (the honest turn):")
    for col in ("c_n_days_with_tx", "a_io_ratio", "a_out6", "log1p_a_in3", "f_ds_r"):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, leftover, folds)
        cov = coverage(pd.to_numeric(df[col], errors="coerce"), leftover)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "n": int(leftover.sum()),
            "n_pos": int((y[leftover] == 1).sum()),
            "coverage": cov["coverage"],
        }
        singles.append(rec)
        print(
            f"    {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"n={rec['n']} pos={rec['n_pos']} cov={cov['coverage']:.3f}"
        )
    already = next(r for r in rows if r["slice"] == "already_neg")
    clean = next(r for r in rows if r["slice"] == "clean_now")
    persist = bool(already["n_pos"] / max(already["n_pos"] + clean["n_pos"], 1) >= 0.75)
    print(
        f"  persistence={persist}  already share of pos="
        f"{already['n_pos'] / (already['n_pos'] + clean['n_pos']):.3f}"
    )
    return {
        "rows": rows,
        "singles": singles,
        "persist": persist,
        "already": already,
        "clean": clean,
    }


def pass10_days_honesty(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Is c_n_days 0.571 a mid-busy bump, not a Q5 tail?"""
    print("\n" + "=" * 72)
    print("CUT 10 — c_n_days_with_tx honesty (Q4 bump vs body)")
    print("=" * 72)
    x = pd.to_numeric(df["c_n_days_with_tx"], errors="coerce")
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    m = train_lab & x.notna()
    # train labeled quintile edge from pass 2: Q4 starts at 17
    hi = m & (x > 17)
    lo = m & (x <= 17)
    rec = {
        "cut": 17,
        "n_hi": int(hi.sum()),
        "n_pos_hi": int((y[hi] == 1).sum()),
        "rate_hi": float(y[hi].mean()) if int(hi.sum()) else float("nan"),
        "n_lo": int(lo.sum()),
        "n_pos_lo": int((y[lo] == 1).sum()),
        "rate_lo": float(y[lo].mean()) if int(lo.sum()) else float("nan"),
        "auroc_flag": float(auroc(y[m], (x[m] > 17).astype(float))),
    }
    print(
        f"  days>17  n={rec['n_hi']} pos={rec['n_pos_hi']} P(Y=1)={rec['rate_hi']:.3f}  |  "
        f"rest n={rec['n_lo']} pos={rec['n_pos_lo']} P(Y=1)={rec['rate_lo']:.3f}  "
        f"flag AUROC={rec['auroc_flag']:.3f}"
    )
    body = train_lab & x.notna() & (x <= 17)
    oof_body = signed_oof_auroc(df, "c_n_days_with_tx", body, df["fold"].to_numpy())
    rec["cv_body"] = oof_body["cv"]
    rec["sd_body"] = oof_body["sd"]
    rec["n_body"] = int(body.sum())
    rec["n_pos_body"] = int((df.loc[body, Y_COL] == 1).sum())
    print(
        f"  days<=17 body CV={oof_body['cv']:.4f}±{oof_body['sd']:.3f} "
        f"n={rec['n_body']} pos={rec['n_pos_body']}"
    )
    # size inside the busy bin
    rec["size_in_hi"] = float(auroc(y[hi], pd.to_numeric(df.loc[hi, "log1p_a_in3"], errors="coerce")))
    rec["size_in_lo"] = float(auroc(y[lo], pd.to_numeric(df.loc[lo, "log1p_a_in3"], errors="coerce")))
    print(f"  size AUROC inside days>17={rec['size_in_hi']:.3f}  inside rest={rec['size_in_lo']:.3f}")
    # HHI sign flip vs Y4: among Y2, higher HHI is protective
    hhi = pd.to_numeric(df["d_cust_hhi"], errors="coerce")
    rec["hhi_sign_y2"] = int(choose_sign(y[train_lab], hhi[train_lab]))
    rec["hhi_q5_rate"] = float("nan")
    print(f"  d_cust_hhi train sign vs Y2 = {rec['hhi_sign_y2']:+d} (Y4 was +1 / monopoly risk)")
    return rec


def pass11_fold0_t3(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Fold 0 T3 large groups: already-neg cluster that flips group_size's sign."""
    print("\n" + "=" * 72)
    print("CUT 11 — fold 0 × large-group T3 × already-neg (why the tree inverted)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    gsz = pd.to_numeric(df["group_size"], errors="coerce")
    _, gbins = pd.qcut(gsz[train_lab].dropna(), 3, retbins=True, duplicates="drop")
    gbin = pd.cut(gsz, bins=gbins, include_lowest=True)
    t3 = gbin == gbin.cat.categories[-1]
    rows = []
    for name, sl in (
        ("f0_T3", train_lab & (df["fold"] == 0) & t3),
        ("rest_T3", train_lab & (df["fold"] != 0) & t3),
        ("f0_notT3", train_lab & (df["fold"] == 0) & ~t3),
        ("rest_notT3", train_lab & (df["fold"] != 0) & ~t3),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "n_cos": int(df.loc[sl, "company_id"].nunique()),
            "n_groups": int(df.loc[sl, "group_id"].nunique()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "share_below0": float(below[sl].mean()) if int(sl.sum()) else float("nan"),
            "n_below0": int((below[sl] == 1).sum()),
        }
        rows.append(rec)
        print(
            f"  {name:12s} n={rec['n']:5d} pos={rec['n_pos']:4d} cos={rec['n_cos']:3d} "
            f"groups={rec['n_groups']:2d} rate={rec['rate']:.4f} "
            f"below0={rec['share_below0']:.3f} (n={rec['n_below0']})"
        )
    # group_size as a diagnostic single (H, never B)
    print("  group_size / h_group_size vs Y2 (diagnostic, not a card):")
    folds = df["fold"].to_numpy()
    singles = []
    for col in ("group_size", "h_group_size"):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, train_lab, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_sign": oof["train_sign"],
            "folds": [f["auroc"] for f in oof["folds"]],
        }
        singles.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(
            f"    {col:14s} CV={oof['cv']:.4f}±{oof['sd']:.3f} sign={oof['train_sign']:+d} folds {bits}"
        )
    f0 = next(r for r in rows if r["slice"] == "f0_T3")
    rest = next(r for r in rows if r["slice"] == "rest_T3")
    note = (
        f"fold0 T3 Y2={f0['rate']:.3f} already-neg={f0['share_below0']:.3f} "
        f"vs rest T3 Y2={rest['rate']:.3f} already-neg={rest['share_below0']:.3f}. "
        "Large groups are stressed in fold 0 and almost clean in folds 1–2 — "
        "group_size gets the wrong sign. That is the 0.344, not a short trail."
    )
    print("  " + note)
    return {"rows": rows, "singles": singles, "note": note, "f0_t3": f0, "rest_t3": rest}


def pass12_clean_leftover(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """f_ds_r / days on the 1.46% clean-now leftover. Footnote, not a new Y."""
    print("\n" + "=" * 72)
    print("CUT 12 — leftover Q5 on clean-now (1.46% — below 5% floor, not a new Y)")
    print("=" * 72)
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    leftover = train_lab & (below == 0)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    folds = df["fold"].to_numpy()
    qtabs = []
    for col in ("f_ds_r", "c_n_days_with_tx", "a_out6"):
        # reuse quintile_table but it reads Y_COL on df — mask leftover
        tab = quintile_table(df, leftover, col)
        qtabs.append(tab)
        print(
            f"  {col} leftover n={tab['n']} bins={tab['n_bins']} shape={tab['shape']}"
        )
        for r in tab["rows"]:
            print(
                f"    Q{r['q']} n={r['n']:5d} pos={r['n_pos']:3d} "
                f"P(Y=1)={r['y_rate']:.3f} med={r['x_median']:.4g}"
            )
    oofs = []
    print("  leftover fold AUCs:")
    for col in ("f_ds_r", "c_n_days_with_tx", "a_out6", "a_io_ratio"):
        oof = signed_oof_auroc(df, col, leftover, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "folds": [f["auroc"] for f in oof["folds"]],
            "n_pos": int((y[leftover] == 1).sum()),
        }
        oofs.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(f"    {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f} folds {bits}")
    fds = next(r for r in oofs if r["col"] == "f_ds_r")
    keep_left = bool(
        np.isfinite(fds["cv"])
        and fds["cv"] >= 0.60
        and fds["sd"] < 0.08
        and (qtabs[0].get("monotone_up") or qtabs[0].get("tail_only"))
    )
    print(
        f"  leftover f_ds_r KEEP? {keep_left}  "
        f"(need CV≥0.60, sd<0.08, tail/mono — and still not a new Y; base is 1.46%)"
    )
    return {
        "qtabs": qtabs,
        "oofs": oofs,
        "keep_left": keep_left,
        "n_pos": int((y[leftover] == 1).sum()),
        "n": int(leftover.sum()),
    }


def pass13_four_groups(df: pd.DataFrame, train_lab: pd.Series, dark_ids: set[str]) -> dict:
    """The 4 fold-0 T3 groups — the cluster that inverted the tree."""
    print("\n" + "=" * 72)
    print("CUT 13 — the 4 fold-0 large groups (not a new Y, not X)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    gsz = pd.to_numeric(df["group_size"], errors="coerce")
    is_dark = df["company_id"].astype(str).isin(dark_ids)
    _, gbins = pd.qcut(gsz[train_lab].dropna(), 3, retbins=True, duplicates="drop")
    t3 = pd.cut(gsz, bins=gbins, include_lowest=True) == pd.cut(
        gsz, bins=gbins, include_lowest=True
    ).cat.categories[-1]
    sl = train_lab & (df["fold"] == 0) & t3
    gids = sorted(df.loc[sl, "group_id"].astype(str).unique())
    rows = []
    for gid in gids:
        m = train_lab & (df["group_id"].astype(str) == gid)
        rec = {
            "group_id": gid,
            "n": int(m.sum()),
            "n_pos": int((y[m] == 1).sum()),
            "n_cos": int(df.loc[m, "company_id"].nunique()),
            "rate": float(y[m].mean()) if int(m.sum()) else float("nan"),
            "share_below0": float(below[m].mean()) if int(m.sum()) else float("nan"),
            "group_size": float(gsz[m].median()) if int(m.sum()) else float("nan"),
            "median_sofar": float(pd.to_numeric(df.loc[m, "so_far"], errors="coerce").median()),
            "share_late": float(pd.to_numeric(df.loc[m, "late_arrival"], errors="coerce").mean()),
            "mean_days": float(pd.to_numeric(df.loc[m, "c_n_days_with_tx"], errors="coerce").mean()),
            "share_dark": float(is_dark[m].mean()) if int(m.sum()) else float("nan"),
            "n_cos_dark": int(df.loc[m & is_dark, "company_id"].nunique()),
        }
        rows.append(rec)
        print(
            f"  {gid}: n={rec['n']:4d} pos={rec['n_pos']:3d} cos={rec['n_cos']:2d} "
            f"rate={rec['rate']:.3f} below0={rec['share_below0']:.3f} "
            f"gsz={rec['group_size']:.0f} sofar={rec['median_sofar']:.0f} "
            f"late={rec['share_late']:.2f} days={rec['mean_days']:.1f} "
            f"dark={rec['share_dark']:.2f} ({rec['n_cos_dark']} cos)"
        )
    f0_pos = int((y[train_lab & (df["fold"] == 0)] == 1).sum())
    top = max(rows, key=lambda r: r["n_pos"]) if rows else None
    share_top = (top["n_pos"] / f0_pos) if top and f0_pos else float("nan")
    print(
        f"  n_groups={len(gids)}  top {None if not top else top['group_id']} "
        f"holds {None if not top else top['n_pos']}/{f0_pos} fold0 pos ({share_top:.3f})"
    )
    return {
        "rows": rows,
        "n_groups": len(gids),
        "group_ids": gids,
        "top": top,
        "share_top_of_fold0_pos": share_top,
        "n_fold0_pos": f0_pos,
    }


def pass14_days_x_neg(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Is the days>17 bump just more already-neg months? B decomp, never X."""
    print("\n" + "=" * 72)
    print("CUT 14 — days>17 × already-neg (is 0.571 a B-state in disguise?)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    x = pd.to_numeric(df["c_n_days_with_tx"], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    busy = x > 17
    rows = []
    for name, sl in (
        ("busy_already", train_lab & busy & (below == 1)),
        ("busy_clean", train_lab & busy & (below == 0)),
        ("quiet_already", train_lab & ~busy & (below == 1)),
        ("quiet_clean", train_lab & ~busy & (below == 0)),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:16s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.4f}")
    share_below_busy = float(below[train_lab & busy].mean())
    share_below_quiet = float(below[train_lab & ~busy].mean())
    print(
        f"  already-neg share busy={share_below_busy:.3f} quiet={share_below_quiet:.3f} "
        f"(ρ days vs below0 was +0.074 — not a B-copy, but is the bump mix?)"
    )
    # days CV on clean-now already in p9; days CV on already-neg:
    already = train_lab & (below == 1)
    oof = signed_oof_auroc(df, "c_n_days_with_tx", already, df["fold"].to_numpy())
    print(
        f"  days CV on already-neg months: {oof['cv']:.4f}±{oof['sd']:.3f} "
        f"n={int(already.sum())} pos={int((y[already] == 1).sum())}"
    )
    return {
        "rows": rows,
        "share_below_busy": share_below_busy,
        "share_below_quiet": share_below_quiet,
        "days_cv_already": oof["cv"],
        "days_sd_already": oof["sd"],
    }


def pass16_drop_cluster(df: pd.DataFrame, train_lab: pd.Series, cluster_ids: list[str]) -> dict:
    """Does fold-0 / days CV die without the 4 large groups? Diagnostic only."""
    print("\n" + "=" * 72)
    print("CUT 16 — drop the 4 fold-0 T3 groups (diagnostic, not a new split)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    cluster = set(str(x) for x in cluster_ids)
    in_c = df["group_id"].astype(str).isin(cluster)
    body = train_lab & ~in_c
    folds = df["fold"].to_numpy()
    rows = []
    for name, sl in (
        ("all_train", train_lab),
        ("drop_4", body),
        ("fold0_drop_4", body & (df["fold"] == 0)),
        ("fold0_keep_4", train_lab & (df["fold"] == 0) & in_c),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:16s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.4f}")
    print("  singles on train minus the 4 groups:")
    singles = []
    for col in ("c_n_days_with_tx", "a_out6", "group_size", "log1p_a_in3"):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, body, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "folds": [f["auroc"] for f in oof["folds"]],
        }
        singles.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(f"    {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f} folds {bits}")
    gsz = next((r for r in singles if r["col"] == "group_size"), None)
    days = next((r for r in singles if r["col"] == "c_n_days_with_tx"), None)
    days_cv = float("nan") if not days else days["cv"]
    gsz_cv = float("nan") if not gsz else gsz["cv"]
    print(f"  after drop: days CV={days_cv:.3f}  group_size CV={gsz_cv:.3f}")
    return {"rows": rows, "singles": singles}


def pass17_dark_mix_acf(df: pd.DataFrame, train_lab: pd.Series, pop: dict) -> dict:
    """360 vs 110 dark mix + Y2 acf1. Confirm, not a new Y."""
    print("\n" + "=" * 72)
    print("CUT 17 — 360/110 dark mix + Y2 acf1 (confirm, not a new Y)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    frame = pop["train_dark_frame"]
    mix_of = frame.set_index("company_id")["mix"]
    mix = df["company_id"].astype(str).map(mix_of)
    rows = []
    for name, sl in (
        ("all_dark_360", train_lab & (mix == "all_dark")),
        ("mixed_110", train_lab & (mix == "mixed")),
        ("invoiced", train_lab & mix.isna()),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "n_cos": int(df.loc[sl, "company_id"].nunique()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {name:14s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"cos={rec['n_cos']:4d} rate={rec['rate']:.4f}"
        )
    # acf1 on train labeled, company-month lag
    tmp = df.loc[train_lab, ["company_id", "period", Y_COL]].copy()
    tmp = tmp.sort_values(["company_id", "period"])
    tmp["lag1"] = tmp.groupby("company_id")[Y_COL].shift(1)
    both = tmp[[Y_COL, "lag1"]].dropna()
    pooled = spearman(both[Y_COL], both["lag1"])
    # company-median acf among cos with ≥4 labeled pairs
    acfs = []
    for _, g in tmp.groupby("company_id"):
        d = g[[Y_COL, "lag1"]].dropna()
        if len(d) < 4 or d[Y_COL].nunique() < 2 or d["lag1"].nunique() < 2:
            continue
        acfs.append(float(d[Y_COL].corr(d["lag1"])))
    med = float(np.nanmedian(acfs)) if acfs else float("nan")
    print(
        f"  Y2 acf1 pooled Spearman={pooled:+.3f} n_pairs={len(both)}  "
        f"company-median={med:+.3f} n_cos={len(acfs)}"
    )
    # Is the dark lift just 0158+0172?
    hot = {"GROUP_0158", "GROUP_0172"}
    hot_m = df["group_id"].astype(str).isin(hot)
    dark360 = train_lab & (mix == "all_dark")
    dark360_wo = dark360 & ~hot_m
    rec_wo = {
        "slice": "all_dark_360_wo_0158_0172",
        "n": int(dark360_wo.sum()),
        "n_pos": int((y[dark360_wo] == 1).sum()),
        "n_cos": int(df.loc[dark360_wo, "company_id"].nunique()),
        "rate": float(y[dark360_wo].mean()) if int(dark360_wo.sum()) else float("nan"),
    }
    rows.append(rec_wo)
    print(
        f"  {rec_wo['slice']:32s} n={rec_wo['n']:5d} pos={rec_wo['n_pos']:4d} "
        f"cos={rec_wo['n_cos']:4d} rate={rec_wo['rate']:.4f}  "
        f"(if this ≈ invoiced 6.3%, dark 9.14% IS those two groups)"
    )
    return {
        "rows": rows,
        "acf1_pooled": pooled,
        "acf1_median": med,
        "n_pairs": int(len(both)),
        "n_cos_acf": int(len(acfs)),
        "dark_wo_hot": rec_wo,
    }


def pass18_hot_already_neg(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Is 0158/0172 a different process or just more already-neg months?"""
    print("\n" + "=" * 72)
    print("CUT 18 — already-neg rate inside 0158/0172 vs rest (composition)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    hot = df["group_id"].astype(str).isin({"GROUP_0158", "GROUP_0172"})
    already = train_lab & (below == 1)
    rows = []
    for name, sl in (
        ("already_neg_in_0158_0172", already & hot),
        ("already_neg_outside", already & ~hot),
        ("clean_in_0158_0172", train_lab & (below == 0) & hot),
        ("clean_outside", train_lab & (below == 0) & ~hot),
        ("0158_0172_all", train_lab & hot),
        ("rest_all", train_lab & ~hot),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "n_cos": int(df.loc[sl, "company_id"].nunique()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "share_below0": float(below[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {name:26s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"cos={rec['n_cos']:4d} rate={rec['rate']:.4f} below0={rec['share_below0']:.3f}"
        )
    in_hot = next(r for r in rows if r["slice"] == "already_neg_in_0158_0172")
    out_hot = next(r for r in rows if r["slice"] == "already_neg_outside")
    same = (
        abs(in_hot["rate"] - out_hot["rate"]) < 0.10
        if (in_hot["n"] and out_hot["n"])
        else False
    )
    print(
        f"  already-neg Y2 in-hot {in_hot['rate']:.1%} vs outside {out_hot['rate']:.1%} "
        f"— {'composition (same persistence)' if same else 'different process in-hot'}"
    )
    return {"rows": rows, "same_process": same}


def pass19_hot_split(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """0158 vs 0172 separately: already-neg stickiness and clean onset."""
    print("\n" + "=" * 72)
    print("CUT 19 — 0158 vs 0172 already-neg / clean onset (not a new Y)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    gid = df["group_id"].astype(str)
    rows = []
    for g in ("GROUP_0158", "GROUP_0172", "GROUP_0023", "GROUP_0250"):
        m = train_lab & (gid == g)
        already = m & (below == 1)
        clean = m & (below == 0)
        rec = {
            "group_id": g,
            "n": int(m.sum()),
            "n_pos": int((y[m] == 1).sum()),
            "rate": float(y[m].mean()) if int(m.sum()) else float("nan"),
            "n_already": int(already.sum()),
            "already_rate": float(y[already].mean()) if int(already.sum()) else float("nan"),
            "n_clean": int(clean.sum()),
            "clean_rate": float(y[clean].mean()) if int(clean.sum()) else float("nan"),
            "share_below0": float(below[m].mean()) if int(m.sum()) else float("nan"),
            "n_cos": int(df.loc[m, "company_id"].nunique()),
        }
        rows.append(rec)
        print(
            f"  {g} n={rec['n']:4d} pos={rec['n_pos']:3d} rate={rec['rate']:.3f} "
            f"already n={rec['n_already']:3d} rate={rec['already_rate']:.3f} "
            f"clean n={rec['n_clean']:3d} rate={rec['clean_rate']:.3f}"
        )
    return {"rows": rows}


def pass20_holdout_and_active(
    df: pd.DataFrame, train_lab: pd.Series, hold_lab: pd.Series
) -> dict:
    """Holdout coverage of the 4 groups + 0158 clean onset on active months."""
    print("\n" + "=" * 72)
    print("CUT 20 — holdout coverage of 0158/0172 + 0158 clean on days>0")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    days = pd.to_numeric(df["c_n_days_with_tx"], errors="coerce")
    gid = df["group_id"].astype(str)
    ho_rows = []
    for g in ("GROUP_0158", "GROUP_0172", "GROUP_0023", "GROUP_0250"):
        m = hold_lab & (gid == g)
        rec = {
            "group_id": g,
            "n": int(m.sum()),
            "n_pos": int((y[m] == 1).sum()),
            "n_cos": int(df.loc[m, "company_id"].nunique()),
            "rate": float(y[m].mean()) if int(m.sum()) else float("nan"),
        }
        ho_rows.append(rec)
        print(
            f"  holdout {g} n={rec['n']:4d} pos={rec['n_pos']:3d} "
            f"cos={rec['n_cos']:3d} rate={rec['rate']}"
        )
    clean_act = train_lab & (gid == "GROUP_0158") & (below == 0) & (days > 0)
    rec_act = {
        "slice": "0158_clean_days_gt0",
        "n": int(clean_act.sum()),
        "n_pos": int((y[clean_act] == 1).sum()),
        "rate": float(y[clean_act].mean()) if int(clean_act.sum()) else float("nan"),
    }
    print(
        f"  {rec_act['slice']} n={rec_act['n']} pos={rec_act['n_pos']} "
        f"rate={rec_act['rate']:.3f}  (if still ~10%, onset is not empty months)"
    )
    return {"hold_rows": ho_rows, "clean_active": rec_act}


def pass21_0158_clean_sofar(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """0158 clean-now onset by so-far. Early-book or throughout?"""
    print("\n" + "=" * 72)
    print("CUT 21 — 0158 clean-now Y2 by so-far (onset clock)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sofar = pd.to_numeric(df["so_far"], errors="coerce")
    gid = df["group_id"].astype(str)
    m = train_lab & (gid == "GROUP_0158") & (below == 0)
    bins = [0, 6, 12, 18, 30]
    labels = ["<6", "6-11", "12-17", "18+"]
    bucket = pd.cut(sofar, bins=bins, labels=labels, right=False)
    rows = []
    for lab in labels:
        sl = m & (bucket == lab)
        rec = {
            "bucket": lab,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  0158 clean {lab:6s} n={rec['n']:4d} pos={rec['n_pos']:3d} "
            f"rate={rec['rate']}"
        )
    return {"rows": rows}


def pass22_0172_and_0158_already_sofar(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """0172 clean-now by so-far + 0158 already-neg rate by so-far."""
    print("\n" + "=" * 72)
    print("CUT 22 — 0172 clean so-far + 0158 already-neg so-far")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sofar = pd.to_numeric(df["so_far"], errors="coerce")
    gid = df["group_id"].astype(str)
    bins = [0, 6, 12, 18, 30]
    labels = ["<6", "6-11", "12-17", "18+"]
    bucket = pd.cut(sofar, bins=bins, labels=labels, right=False)
    rows = []
    for name, sl0 in (
        ("0172_clean", train_lab & (gid == "GROUP_0172") & (below == 0)),
        ("0158_already", train_lab & (gid == "GROUP_0158") & (below == 1)),
        ("0158_all", train_lab & (gid == "GROUP_0158")),
    ):
        for lab in labels:
            sl = sl0 & (bucket == lab)
            rec = {
                "slice": name,
                "bucket": lab,
                "n": int(sl.sum()),
                "n_pos": int((y[sl] == 1).sum()),
                "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
                "share_below0": float(below[sl].mean()) if int(sl.sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"  {name:14s} {lab:6s} n={rec['n']:4d} pos={rec['n_pos']:3d} "
                f"rate={rec['rate']} below0={rec['share_below0']}"
            )
    return {"rows": rows}


def pass23_0158_crash_story(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Is 0158's late onset the Y4 inflow crash? Label story, not X."""
    print("\n" + "=" * 72)
    print("CUT 23 — 0158 Y2pos crash share + median in_ratio (label story)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    crash = pd.to_numeric(df["inflow_crash"], errors="coerce")
    inr = pd.to_numeric(df["in_ratio"], errors="coerce")
    gid = df["group_id"].astype(str)
    hot = gid.isin({"GROUP_0158", "GROUP_0172"})
    rows = []
    for name, sl in (
        ("Y2pos_0158", train_lab & (y == 1) & (gid == "GROUP_0158")),
        ("Y2pos_0172", train_lab & (y == 1) & (gid == "GROUP_0172")),
        ("Y2pos_hot", train_lab & (y == 1) & hot),
        ("Y2pos_rest", train_lab & (y == 1) & ~hot),
        ("Y2pos_0158_sofar_ge12", train_lab & (y == 1) & (gid == "GROUP_0158") & (pd.to_numeric(df["so_far"], errors="coerce") >= 12)),
    ):
        defined = sl & crash.notna()
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_crash_def": int(defined.sum()),
            "crash_share": float(crash[defined].mean()) if int(defined.sum()) else float("nan"),
            "med_in_ratio": float(inr[sl].median()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {name:22s} n={rec['n']:4d} crash_def={rec['n_crash_def']:4d} "
            f"crash={rec['crash_share']} med_in={rec['med_in_ratio']}"
        )
    return {"rows": rows}


def pass24_hot_y4_y9(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Y4 / Y9 overlap on Y2 pos inside 0158/0172 vs rest. Not a merge."""
    print("\n" + "=" * 72)
    print("CUT 24 — Y2pos × Y4 / Y9 inside 0158/0172 vs rest")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    y4 = pd.to_numeric(df[Y4], errors="coerce")
    y9 = pd.to_numeric(df[Y9], errors="coerce")
    gid = df["group_id"].astype(str)
    hot = gid.isin({"GROUP_0158", "GROUP_0172"})
    rows = []
    for name, sl in (
        ("Y2pos_0158", train_lab & (y == 1) & (gid == "GROUP_0158")),
        ("Y2pos_0172", train_lab & (y == 1) & (gid == "GROUP_0172")),
        ("Y2pos_hot", train_lab & (y == 1) & hot),
        ("Y2pos_rest", train_lab & (y == 1) & ~hot),
    ):
        y4d = sl & y4.notna()
        y9d = sl & y9.notna()
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "y4_def": int(y4d.sum()),
            "y4_share": float(y4[y4d].mean()) if int(y4d.sum()) else float("nan"),
            "y9_def": int(y9d.sum()),
            "y9_share": float(y9[y9d].mean()) if int(y9d.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {name:12s} n={rec['n']:4d} Y4={rec['y4_share']} (def {rec['y4_def']}) "
            f"Y9={rec['y9_share']} (def {rec['y9_def']})"
        )
    return {"rows": rows}


def pass25_hot_medians(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Legal-X medians in 0158/0172 vs rest. Is days 0.571 just this cluster?"""
    print("\n" + "=" * 72)
    print("CUT 25 — legal-X medians 0158 / 0172 / rest (never B as X)")
    print("=" * 72)
    gid = df["group_id"].astype(str)
    cols = ["c_n_days_with_tx", "a_io_ratio", "a_out6", "a_in3", "d_cust_hhi", "f_ds_r"]
    rows = []
    for name, sl in (
        ("0158", train_lab & (gid == "GROUP_0158")),
        ("0172", train_lab & (gid == "GROUP_0172")),
        ("rest", train_lab & ~gid.isin({"GROUP_0158", "GROUP_0172"})),
    ):
        rec = {"slice": name, "n": int(sl.sum())}
        bits = []
        for c in cols:
            med = float(pd.to_numeric(df.loc[sl, c], errors="coerce").median())
            rec[c] = med
            bits.append(f"{c}={med:.4g}")
        rows.append(rec)
        print(f"  {name:6s} n={rec['n']:5d}  " + "  ".join(bits))
    # days CV after drop 0158+0172 only (not all 4)
    drop2 = train_lab & ~gid.isin({"GROUP_0158", "GROUP_0172"})
    days = signed_oof_auroc(df, "c_n_days_with_tx", drop2, df["fold"].to_numpy())
    fold_aucs = [f["auroc"] for f in days["folds"]]
    print(
        f"  days CV drop 0158+0172 only: {days['cv']:.3f} ± {days['sd']:.3f} "
        f"folds={' '.join(f'{a:.3f}' for a in fold_aucs)}"
    )
    return {"rows": rows, "days_drop2": days}


def pass26_rest_singles(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Legal singles on train minus 0158+0172. Does any KEEP survive?"""
    print("\n" + "=" * 72)
    print("CUT 26 — singles on rest (drop 0158+0172 only; never B)")
    print("=" * 72)
    gid = df["group_id"].astype(str)
    body = train_lab & ~gid.isin({"GROUP_0158", "GROUP_0172"})
    folds = df["fold"].to_numpy()
    rows = []
    for col in (
        "c_n_days_with_tx",
        "a_out6",
        "a_io_ratio",
        "d_cust_hhi",
        "f_ds_r",
        "c_zero_in_month",
        "log1p_a_in3",
    ):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, body, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "folds": [f["auroc"] for f in oof["folds"]],
            "coverage": float(
                pd.to_numeric(df.loc[body, col], errors="coerce").notna().mean()
            ),
        }
        rows.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(
            f"  {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f} cov={rec['coverage']:.3f} "
            f"folds {bits}"
        )
    best = max(
        (r for r in rows if r["col"] != "log1p_a_in3" and np.isfinite(r["cv"])),
        key=lambda r: r["cv"],
        default=None,
    )
    size = next((r for r in rows if r["col"] == "log1p_a_in3"), None)
    gap = (
        (best["cv"] - size["cv"])
        if best and size and np.isfinite(size["cv"])
        else float("nan")
    )
    bcol = None if not best else best["col"]
    bcv = float("nan") if not best else best["cv"]
    scv = float("nan") if not size else size["cv"]
    print(f"  best rest `{bcol}` CV={bcv:.3f} vs size {scv:.3f} gap={gap:+.3f}")
    return {"rows": rows, "best": best, "size": size, "gap": gap}


def pass27_hhi_fold2(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """HHI fold-2 inversion on rest (0158/0172 have no HHI). Not a KEEP."""
    print("\n" + "=" * 72)
    print("CUT 27 — d_cust_hhi fold 2 inversion (rest only)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    hhi = pd.to_numeric(df["d_cust_hhi"], errors="coerce")
    gid = df["group_id"].astype(str)
    rest = train_lab & ~gid.isin({"GROUP_0158", "GROUP_0172"}) & hhi.notna()
    f2 = rest & (df["fold"] == 2)
    tail = rest & (hhi > HHI_TAIL)
    f2_tail = f2 & (hhi > HHI_TAIL)
    rows = []
    for name, sl in (
        ("rest_hhi_def", rest),
        ("rest_hhi_tail", tail),
        ("fold2_hhi_def", f2),
        ("fold2_hhi_tail", f2_tail),
        ("not_f2_hhi_tail", rest & (df["fold"] != 2) & (hhi > HHI_TAIL)),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "n_cos": int(df.loc[sl, "company_id"].nunique()),
            "n_groups": int(df.loc[sl, "group_id"].nunique()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "med_hhi": float(hhi[sl].median()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {name:18s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"cos={rec['n_cos']:4d} g={rec['n_groups']:3d} "
            f"rate={rec['rate']:.4f} medHHI={rec['med_hhi']:.3f}"
        )
    # fold2 groups with HHI defined, highest Y2
    g = (
        df.loc[f2, ["group_id", Y_COL]]
        .assign(y=y[f2].values)
        .groupby("group_id")
        .agg(n=("y", "size"), n_pos=("y", "sum"))
    )
    g["rate"] = g["n_pos"] / g["n"]
    top = g.sort_values("n_pos", ascending=False).head(5)
    print("  fold2 HHI-defined groups by n_pos:")
    top_rows = []
    for gid_v, r in top.iterrows():
        rec = {
            "group_id": str(gid_v),
            "n": int(r["n"]),
            "n_pos": int(r["n_pos"]),
            "rate": float(r["rate"]),
        }
        top_rows.append(rec)
        print(f"    {rec['group_id']} n={rec['n']} pos={rec['n_pos']} rate={rec['rate']:.3f}")
    return {"rows": rows, "top_f2": top_rows}


def pass28_f2_hhi_groups(df: pd.DataFrame, train_lab: pd.Series, pop: dict) -> dict:
    """Footnote: fold-2 HHI-hot groups. Not a card. Never B as X."""
    print("\n" + "=" * 72)
    print("CUT 28 — fold-2 HHI-hot groups (0241 / 0230 / 0088) footnote")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    days = pd.to_numeric(df["c_n_days_with_tx"], errors="coerce")
    hhi = pd.to_numeric(df["d_cust_hhi"], errors="coerce")
    dark_ids = set(str(x) for x in pop["train_dark_ids"])
    gid = df["group_id"].astype(str)
    rows = []
    for g in ("GROUP_0241", "GROUP_0230", "GROUP_0088", "GROUP_0013", "GROUP_0094"):
        sl = train_lab & (gid == g)
        rec = {
            "group_id": g,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "n_cos": int(df.loc[sl, "company_id"].nunique()),
            "share_below0": float(below[sl].mean()) if int(sl.sum()) else float("nan"),
            "med_days": float(days[sl].median()) if int(sl.sum()) else float("nan"),
            "med_hhi": float(hhi[sl].median()) if int(sl.sum()) else float("nan"),
            "share_dark": float(
                df.loc[sl, "company_id"].astype(str).isin(dark_ids).mean()
            )
            if int(sl.sum())
            else float("nan"),
            "fold": int(df.loc[sl, "fold"].mode().iloc[0]) if int(sl.sum()) else -1,
        }
        rows.append(rec)
        print(
            f"  {g} n={rec['n']:4d} pos={rec['n_pos']:3d} rate={rec['rate']:.3f} "
            f"cos={rec['n_cos']:3d} below0={rec['share_below0']:.3f} "
            f"days={rec['med_days']:.1f} hhi={rec['med_hhi']} "
            f"dark={rec['share_dark']:.2f} fold={rec['fold']}"
        )
    return {"rows": rows}


def pass29_0241_story(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """0241 is a tiny invoiced persistence pile. Not a Q5. Not Y4."""
    print("\n" + "=" * 72)
    print("CUT 29 — GROUP_0241 already-neg / crash (tiny invoiced pile)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    crash = pd.to_numeric(df["inflow_crash"], errors="coerce")
    inr = pd.to_numeric(df["in_ratio"], errors="coerce")
    gid = df["group_id"].astype(str)
    m = train_lab & (gid == "GROUP_0241")
    already = m & (below == 1)
    clean = m & (below == 0)
    pos = m & (y == 1)
    defined = pos & crash.notna()
    rec = {
        "n": int(m.sum()),
        "n_pos": int(pos.sum()),
        "rate": float(y[m].mean()) if int(m.sum()) else float("nan"),
        "already_n": int(already.sum()),
        "already_rate": float(y[already].mean()) if int(already.sum()) else float("nan"),
        "clean_n": int(clean.sum()),
        "clean_rate": float(y[clean].mean()) if int(clean.sum()) else float("nan"),
        "crash_share": float(crash[defined].mean()) if int(defined.sum()) else float("nan"),
        "med_in": float(inr[pos].median()) if int(pos.sum()) else float("nan"),
        "n_cos": int(df.loc[m, "company_id"].nunique()),
    }
    print(
        f"  0241 n={rec['n']} pos={rec['n_pos']} rate={rec['rate']:.3f} cos={rec['n_cos']} "
        f"already {rec['already_n']} rate={rec['already_rate']:.3f} "
        f"clean {rec['clean_n']} rate={rec['clean_rate']:.3f} "
        f"crash={rec['crash_share']} med_in={rec['med_in']}"
    )
    return rec


def pass30_0158_cos(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """How many of 0158's 21 companies are chronically already-neg?"""
    print("\n" + "=" * 72)
    print("CUT 30 — GROUP_0158 per-company already-neg share (not a new Y)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sl = train_lab & (df["group_id"].astype(str) == "GROUP_0158")
    g = (
        df.loc[sl, ["company_id"]]
        .assign(y=y[sl].values, below=below[sl].values)
        .groupby("company_id")
        .agg(n=("y", "size"), n_pos=("y", "sum"), n_below=("below", "sum"))
    )
    g["rate"] = g["n_pos"] / g["n"]
    g["share_below"] = g["n_below"] / g["n"]
    n_cos = int(len(g))
    n_any = int((g["n_pos"] > 0).sum())
    n_half = int((g["share_below"] >= 0.5).sum())
    n_all = int((g["share_below"] >= 0.99).sum())
    n_never = int((g["n_below"] == 0).sum())
    print(
        f"  0158 cos={n_cos} any-Y2={n_any} below≥50%={n_half} "
        f"always-below={n_all} never-below={n_never} "
        f"median share_below={g['share_below'].median():.3f} "
        f"median Y2={g['rate'].median():.3f}"
    )
    return {
        "n_cos": n_cos,
        "n_any": n_any,
        "n_half": n_half,
        "n_all": n_all,
        "n_never": n_never,
        "med_below": float(g["share_below"].median()),
        "med_y2": float(g["rate"].median()),
        "chronic_ids": [str(i) for i in g.index[g["share_below"] >= 0.5]],
    }


def pass31_chronic_share_of_fold0(df: pd.DataFrame, train_lab: pd.Series, chronic_ids: list[str]) -> dict:
    """What share of fold-0 Y2 pos sit in the 6 chronic 0158 names?"""
    print("\n" + "=" * 72)
    print("CUT 31 — fold-0 Y2 pos in 0158 chronic names (share of the pile)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    f0_pos = train_lab & (df["fold"] == 0) & (y == 1)
    chron = df["company_id"].astype(str).isin(set(chronic_ids))
    n_f0 = int(f0_pos.sum())
    n_ch = int((f0_pos & chron).sum())
    n_0158 = int((f0_pos & (df["group_id"].astype(str) == "GROUP_0158")).sum())
    share_ch = (n_ch / n_f0) if n_f0 else float("nan")
    share_0158 = (n_0158 / n_f0) if n_f0 else float("nan")
    print(
        f"  fold0 pos={n_f0}  in 0158={n_0158} ({share_0158:.1%})  "
        f"in 0158-chronic-{len(chronic_ids)}={n_ch} ({share_ch:.1%})"
    )
    return {
        "n_f0": n_f0,
        "n_0158": n_0158,
        "n_chronic": n_ch,
        "n_chronic_cos": len(chronic_ids),
        "share_0158": share_0158,
        "share_chronic": share_ch,
    }


def pass32_0172_chronic(
    df: pd.DataFrame, train_lab: pd.Series, ids_0158: list[str]
) -> dict:
    """0172 chronic names + combined share of fold-0 pos."""
    print("\n" + "=" * 72)
    print("CUT 32 — 0172 chronic + 0158+0172 chronic share of fold-0 pos")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sl = train_lab & (df["group_id"].astype(str) == "GROUP_0172")
    g = (
        df.loc[sl, ["company_id"]]
        .assign(y=y[sl].values, below=below[sl].values)
        .groupby("company_id")
        .agg(n=("y", "size"), n_pos=("y", "sum"), n_below=("below", "sum"))
    )
    g["share_below"] = g["n_below"] / g["n"]
    chron = [str(i) for i in g.index[g["share_below"] >= 0.5]]
    n_cos = int(len(g))
    n_any = int((g["n_pos"] > 0).sum())
    n_never = int((g["n_below"] == 0).sum())
    print(
        f"  0172 cos={n_cos} any-Y2={n_any} chronic≥50%={len(chron)} "
        f"never-below={n_never} med share_below={g['share_below'].median():.3f}"
    )
    f0_pos = train_lab & (df["fold"] == 0) & (y == 1)
    both = set(chron) | set(ids_0158)
    n_f0 = int(f0_pos.sum())
    n_both = int((f0_pos & df["company_id"].astype(str).isin(both)).sum())
    n_0172 = int((f0_pos & (df["group_id"].astype(str) == "GROUP_0172")).sum())
    share = (n_both / n_f0) if n_f0 else float("nan")
    print(
        f"  fold0 pos={n_f0}  in 0172={n_0172}  "
        f"in 0158+0172 chronic ({len(both)} cos)={n_both} ({share:.1%})"
    )
    return {
        "n_cos": n_cos,
        "n_any": n_any,
        "n_chronic": len(chron),
        "n_never": n_never,
        "med_below": float(g["share_below"].median()),
        "n_f0": n_f0,
        "n_0172": n_0172,
        "n_both_chronic": n_both,
        "n_both_cos": len(both),
        "share_both": share,
    }


def pass33_chronic_days(df: pd.DataFrame, train_lab: pd.Series, ids_0158: list[str]) -> dict:
    """Do the 12 chronic names look busy on days? Legal X only."""
    print("\n" + "=" * 72)
    print("CUT 33 — days / io / in3 medians on 0158+0172 chronic vs rest")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sl172 = train_lab & (df["group_id"].astype(str) == "GROUP_0172")
    g = (
        df.loc[sl172, ["company_id"]]
        .assign(below=below[sl172].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    chron172 = [str(i) for i in g.index[g["share_below"] >= 0.5]]
    both = set(ids_0158) | set(chron172)
    is_ch = df["company_id"].astype(str).isin(both)
    rows = []
    for name, sl in (
        ("chronic_12", train_lab & is_ch),
        ("chronic_12_pos", train_lab & is_ch & (y == 1)),
        ("rest", train_lab & ~is_ch),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "med_days": float(pd.to_numeric(df.loc[sl, "c_n_days_with_tx"], errors="coerce").median()),
            "med_io": float(pd.to_numeric(df.loc[sl, "a_io_ratio"], errors="coerce").median()),
            "med_in3": float(pd.to_numeric(df.loc[sl, "a_in3"], errors="coerce").median()),
            "med_out6": float(pd.to_numeric(df.loc[sl, "a_out6"], errors="coerce").median()),
        }
        rows.append(rec)
        print(
            f"  {name:16s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"days={rec['med_days']:.1f} io={rec['med_io']:.3g} "
            f"in3={rec['med_in3']:.4g} out6={rec['med_out6']:.4g}"
        )
    return {"rows": rows, "n_ids": len(both)}


def pass34_drop_12(df: pd.DataFrame, train_lab: pd.Series, ids_0158: list[str]) -> dict:
    """Days CV after dropping the 12 chronic companies only."""
    print("\n" + "=" * 72)
    print("CUT 34 — days CV drop 12 chronic companies (not whole groups)")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sl172 = train_lab & (df["group_id"].astype(str) == "GROUP_0172")
    g = (
        df.loc[sl172, ["company_id"]]
        .assign(below=below[sl172].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    chron172 = [str(i) for i in g.index[g["share_below"] >= 0.5]]
    both = set(ids_0158) | set(chron172)
    is_ch = df["company_id"].astype(str).isin(both)
    body = train_lab & ~is_ch
    oof = signed_oof_auroc(df, "c_n_days_with_tx", body, df["fold"].to_numpy())
    size = signed_oof_auroc(df, "log1p_a_in3", body, df["fold"].to_numpy())
    bits = " ".join(f"{f['auroc']:.3f}" for f in oof["folds"])
    print(
        f"  drop 12: n={int(body.sum())} pos={int((y[body]==1).sum())} "
        f"days CV={oof['cv']:.3f}±{oof['sd']:.3f} folds {bits}"
    )
    print(f"  size CV on same mask={size['cv']:.3f}  gap={oof['cv']-size['cv']:+.3f}")
    return {
        "n": int(body.sum()),
        "n_pos": int((y[body] == 1).sum()),
        "days_cv": oof["cv"],
        "days_sd": oof["sd"],
        "days_folds": [f["auroc"] for f in oof["folds"]],
        "size_cv": size["cv"],
        "gap": oof["cv"] - size["cv"],
    }


def pass35_fold0_drop12(df: pd.DataFrame, train_lab: pd.Series, ids_0158: list[str]) -> dict:
    """Fold-0 rate after dropping the 12 chronic names only."""
    print("\n" + "=" * 72)
    print("CUT 35 — fold-0 rate without the 12 chronic names")
    print("=" * 72)
    y = pd.to_numeric(df[Y_COL], errors="coerce")
    below = pd.to_numeric(df["b_below_0"], errors="coerce")
    sl172 = train_lab & (df["group_id"].astype(str) == "GROUP_0172")
    g = (
        df.loc[sl172, ["company_id"]]
        .assign(below=below[sl172].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    chron172 = [str(i) for i in g.index[g["share_below"] >= 0.5]]
    both = set(ids_0158) | set(chron172)
    is_ch = df["company_id"].astype(str).isin(both)
    rows = []
    for name, sl in (
        ("fold0", train_lab & (df["fold"] == 0)),
        ("fold0_wo_12", train_lab & (df["fold"] == 0) & ~is_ch),
        ("rest_wo_12", train_lab & (df["fold"] != 0) & ~is_ch),
        ("fold0_only_12", train_lab & (df["fold"] == 0) & is_ch),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:16s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.4f}")
    return {"rows": rows}


def decide(p1: dict, p2: dict, p3: dict, p5: dict) -> dict:
    """KEEP Q5 / CLOSE (same crash) / PARK trees. No new tree."""
    best = p3.get("best")
    size = p3["size_auc_max"]
    fails = {r["col"] for r in p5["fails"]}
    qmap = {t["x"]: t for t in p2["tables"]}
    q5_keep = False
    q5_col = None
    q5_reason = "no non-B single is a clear tail/monotone and beats size by ≥0.02"
    if best and best["col"] not in fails:
        tab = qmap.get(best["col"]) or {}
        shape_ok = bool(
            tab.get("monotone_up")
            or tab.get("monotone_down")
            or tab.get("tail_only")
            or tab.get("head_only")
        )
        beats = bool(np.isfinite(best["gap_vs_size"]) and best["gap_vs_size"] >= CLEAR_MARGIN)
        if best["cv"] >= SIZE_BAR:
            q5_reason = (
                f"{best['col']} CV {best['cv']:.3f} ≥ {SIZE_BAR:.2f} — PARK as X (size-like)"
            )
        elif shape_ok and beats:
            q5_keep = True
            q5_col = best["col"]
            q5_reason = (
                f"{best['col']} shape={tab.get('shape')} CV {best['cv']:.3f} "
                f"beats size {size:.3f} by {best['gap_vs_size']:+.3f}"
            )
        else:
            q5_reason = (
                f"{best['col']} CV {best['cv']:.3f} gap_vs_size={best['gap_vs_size']:+.3f} "
                f"shape={tab.get('shape')} — no KEEP"
            )
    if p1["same_crash_as_y4"]:
        decision = "CLOSE"
        reason = (
            f"Y2 is Y4's inflow crash by another name: "
            f"{p1['crash_of_y2']:.0%} of Y2 pos crash, {p1['y4_of_y2']:.0%} are Y4, "
            f"{p1['y2_of_y4']:.0%} of Y4 pos are Y2"
        )
    else:
        decision = "KEEP_Q5" if q5_keep else "CLOSE"
        reason = (
            q5_reason
            if q5_keep
            else (
                f"Y2 is a different turn from Y4 crash "
                f"(Y2pos crash={p1['crash_of_y2']:.3f}, Y2∩Y4={p1['y4_of_y2']:.3f}, "
                f"Y4∩Y2={p1['y2_of_y4']:.3f}). {q5_reason}. Trees stay PARK."
            )
        )
        if q5_keep:
            reason = q5_reason + ". Trees stay PARK (night 0.540 / fold0 0.344)."
    print(f"\nVERDICT {decision}: {reason}")
    return {
        "decision": decision,
        "reason": reason,
        "q5_keep": q5_keep,
        "q5_col": q5_col,
        "q5_reason": q5_reason,
        "trees": "PARK",
    }


def mapping_sentence(p1: dict, p2: dict, p3: dict, verdict: dict, extra=None) -> str:
    best = p3.get("best") or {}
    tab = next((t for t in p2["tables"] if t["x"] == best.get("col")), None)
    extra = extra or {}
    p7 = extra.get("p7") or {}
    p9 = extra.get("p9") or {}
    p11 = extra.get("p11") or {}
    already = (p9.get("already") or {}).get("n_pos", 0)
    clean = (p9.get("clean") or {}).get("n_pos", 0)
    persist_share = already / max(already + clean, 1)
    f0 = p11.get("f0_t3") or {}
    rest = p11.get("rest_t3") or {}
    return (
        f"Y2 `neg_2of3` is the brief's 82→68 *direction* (Q3), but on train it is mostly "
        f"**already-negative persistence** ({persist_share:.0%} of positives are `b_below_0` "
        f"at t; already-neg months run at 71% Y2 vs clean-now **1.5%**). "
        f"It is **not** Y4's inflow crash (Y2pos crash={p1['crash_of_y2']:.0%}, "
        f"Y2∩Y4={p1['y4_of_y2']:.0%}, Y4∩Y2={p1['y2_of_y4']:.0%}; median in-ratio "
        f"{p1['pos_med_in_ratio']:.2f} vs Y4's 0.36). "
        f"No non-B Q5 KEEP: best legal `{best.get('col')}` CV "
        f"{best.get('cv', float('nan')):.3f} vs size {p3['size_auc_max']:.3f} "
        f"is a mid-busy bump (shape={tab.get('shape') if tab else 'flat'}), not a tail. "
        f"Q6 CLOSE: lag-1 days { (p7.get('days_lag1') or {}).get('cv', float('nan')):.3f} "
        f"is lag0+{p7.get('lag_lift', 0):+.3f}, not a lead. "
        f"Fold 0 died because large groups there are already-neg "
        f"(T3 Y2 {f0.get('rate', float('nan')):.1%} vs rest-T3 {rest.get('rate', float('nan')):.1%}) "
        f"— not a short trail. Do not use family B as X. Trees stay PARK. Not a 0–100."
    )


def extra_md_lines(p7: dict, p8: dict, p9: dict, p10: dict) -> list[str]:
    lines = [
        "### 7 — honest 1-month Q6",
        "",
        "Only lag-1 is an honest clock (Q6 quoted: longer leads die on the hidden 72). Never B.",
        "",
        "| feature | lag | CV AUROC ± sd | train | sign | coverage | gap vs size |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p7["rows"]:
        lines.append(
            f"| `{r['col']}` | {r['lag']} | {r['cv']:.3f} ± {r['sd']:.3f} | "
            f"{r['train_auc']:.3f} | {r['train_sign']:+d} | {r['coverage']:.3f} | "
            f"{r['gap_vs_size']:+.3f} |"
        )
    d1 = p7.get("days_lag1") or {}
    lines += [
        "",
        f"Q6 KEEP lag-1 `c_n_days_with_tx` (must beat size by ≥{CLEAR_MARGIN:.2f}, "
        f"stay <0.60, **and beat lag-0 by ≥0.01**): "
        f"**{p7['keep_q6']}** (lag1 CV {d1.get('cv', float('nan')):.3f}, "
        f"lag_lift {p7.get('lag_lift', float('nan')):+.3f}). "
        "Lag-1 ≈ lag-0 is the same busy-bin trait, not a lead. Do not claim Q6.",
        "",
        "### 8 — fold 0 × so-far × group size",
        "",
        "Company-month so-far (not company-total months-on-book). Fixed cuts from trail_length.md. "
        "Not a new histogram.",
        "",
        "| fold | n | P(Y=1) | share already-neg | so-far<12 | so-far≥18 | median so-far |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p8["fold_rows"]:
        lines.append(
            f"| {r['fold']} | {r['n']} | {r['rate']:.3f} | {r['share_below0']:.3f} | "
            f"{r['share_sofar_lt12']:.3f} | {r['share_sofar_ge18']:.3f} | {r['median_sofar']:.1f} |"
        )
    res = p8["residual"]
    lines += [
        "",
        f"Fold 0 vs rest inside long so-far≥18: **{res['f0_long_rate']:.3f}** vs "
        f"{res['rest_long_rate']:.3f} (gap {res['long_gap']:+.3f}, n_f0={res['n_f0_long']}). "
        f"Inside short<12: **{res['f0_short_rate']:.3f}** vs {res['rest_short_rate']:.3f} "
        f"(gap {res['short_gap']:+.3f}). "
        f"late_trail_effect=**{p8['late_trail_effect']}**  size_tercile_effect=**{p8['size_effect']}**.",
        "",
        "Y2 rate by so-far bucket × fold:",
        "",
        "| bucket | fold | n | n_pos | P(Y=1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p8["so_rows"]:
        lines.append(
            f"| {r['bucket']} | {r['fold']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        "Y2 rate by group_size tercile (train cuts) × fold:",
        "",
        "| T | interval | fold | n | n_pos | P(Y=1) |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in p8["g_rows"]:
        lines.append(
            f"| {r['tercile']} | {r['interval']} | {r['fold']} | {r['n']} | "
            f"{r['n_pos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        "### 9 — already-neg vs clean-now (B decomp, never X)",
        "",
        "If most Y2 positives are already `b_below_0`, the accepted label is persistence "
        "of the cash path, not an onset. That is why `y2_onset_neg` was rejected. Not X.",
        "",
        "| slice | n | n_pos | companies | P(Y=1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p9["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        f"Persistence: **{p9['persist']}**. "
        f"Already-neg share of positives = "
        f"{p9['already']['n_pos'] / max(p9['already']['n_pos'] + p9['clean']['n_pos'], 1):.1%}.",
        "",
        "Legal singles on the clean-now leftover (honest turn, never B):",
        "",
        "| feature | CV AUROC ± sd | n | n_pos | coverage |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p9["singles"]:
        lines.append(
            f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['n']} | "
            f"{r['n_pos']} | {r['coverage']:.3f} |"
        )
    lines += [
        "",
        "### 10 — `c_n_days_with_tx` honesty",
        "",
        f"days>17: P(Y=1)=**{p10['rate_hi']:.1%}** (n={p10['n_hi']}, {p10['n_pos_hi']} pos) vs rest "
        f"**{p10['rate_lo']:.1%}**. Flag AUROC {p10['auroc_flag']:.3f}. "
        f"Body days≤17 CV **{p10['cv_body']:.3f} ± {p10['sd_body']:.3f}**. "
        f"Size inside busy bin {p10['size_in_hi']:.3f} / rest {p10['size_in_lo']:.3f}. "
        f"`d_cust_hhi` sign vs Y2 = {p10['hhi_sign_y2']:+d} (Y4 was +1 — monopoly is the other turn).",
        "",
    ]
    return lines


def extra_md_lines_b(p11: dict, p12: dict) -> list[str]:
    lines = [
        "### 11 — fold 0 large-group T3 × already-neg",
        "",
        p11["note"],
        "",
        "| slice | n | n_pos | cos | groups | P(Y=1) | share already-neg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p11["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | {r['n_groups']} | "
            f"{r['rate']:.3f} | {r['share_below0']:.3f} |"
        )
    lines += [
        "",
        "group_size diagnostic (never B, not a card):",
        "",
        "| feature | CV AUROC ± sd | sign | folds |",
        "|---|---:|---:|---|",
    ]
    for r in p11["singles"]:
        bits = " ".join(f"{a:.3f}" for a in r["folds"])
        lines.append(
            f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['train_sign']:+d} | {bits} |"
        )
    lines += [
        "",
        "### 12 — clean-now leftover (1.46%, not a new Y)",
        "",
        f"n={p12['n']} pos={p12['n_pos']}. Below the 5% acceptance floor. "
        f"leftover `f_ds_r` KEEP? **{p12['keep_left']}**.",
        "",
    ]
    for t in p12["qtabs"]:
        lines += [
            f"`{t['x']}` leftover quintiles shape={t['shape']}:",
            "",
            "| Q | n | n_pos | P(Y=1) | median |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in t["rows"]:
            lines.append(
                f"| {r['q']} | {r['n']} | {r['n_pos']} | {r['y_rate']:.3f} | {r['x_median']:.4g} |"
            )
        lines.append("")
    lines += [
        "| feature | leftover CV ± sd | folds |",
        "|---|---:|---|",
    ]
    for r in p12["oofs"]:
        bits = " ".join(f"{a:.3f}" for a in r["folds"])
        lines.append(f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {bits} |")
    lines.append("")
    return lines


def extra_md_lines_c(p13: dict, p14: dict) -> list[str]:
    lines = [
        "### 13 — the 4 fold-0 large groups",
        "",
        "Four groups in fold 0 sit in the large tercile. That is the 0.344, not late-trail.",
        "",
        "| group | n | n_pos | cos | P(Y=1) | already-neg | group_size | median so-far | late | mean days | dark share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p13["rows"]:
        lines.append(
            f"| `{r['group_id']}` | {r['n']} | {r['n_pos']} | {r['n_cos']} | "
            f"{r['rate']:.3f} | {r['share_below0']:.3f} | {r['group_size']:.0f} | "
            f"{r['median_sofar']:.0f} | {r['share_late']:.2f} | {r['mean_days']:.1f} | "
            f"{r['share_dark']:.2f} |"
        )
    top = p13.get("top") or {}
    lines += [
        "",
        f"`{top.get('group_id')}` holds **{p13.get('share_top_of_fold0_pos', float('nan')):.0%}** "
        f"of fold-0 positives ({top.get('n_pos')}/{p13.get('n_fold0_pos')}). "
        "0158 and 0172 are **all-dark** 21-company groups. 0158 is not late (late=0). "
        "0250 is a large invoiced group at 0% Y2. Dark 9.14% is these clusters — not a new Y.",
    ]
    lines += [
        "",
        "### 14 — days>17 × already-neg",
        "",
        f"Already-neg share busy={p14['share_below_busy']:.3f} vs quiet={p14['share_below_quiet']:.3f}. "
        f"days CV on already-neg months {p14['days_cv_already']:.3f} ± {p14['days_sd_already']:.3f}.",
        "",
        "| slice | n | n_pos | P(Y=1) |",
        "|---|---:|---:|---:|",
    ]
    for r in p14["rows"]:
        lines.append(f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |")
    lines.append("")
    return lines


def extra_md_lines_d(p16: dict) -> list[str]:
    lines = [
        "### 16 — drop the 4 fold-0 T3 groups",
        "",
        "Diagnostic mask only. Does not change the accepted Y or the holdout.",
        "",
        "| slice | n | n_pos | P(Y=1) |",
        "|---|---:|---:|---:|",
    ]
    for r in p16["rows"]:
        lines.append(f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |")
    lines += [
        "",
        "| feature | CV after drop ± sd | folds |",
        "|---|---:|---|",
    ]
    for r in p16["singles"]:
        bits = " ".join(f"{a:.3f}" for a in r["folds"])
        lines.append(f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {bits} |")
    lines.append("")
    return lines


def extra_md_lines_e(p17: dict) -> list[str]:
    lines = [
        "### 17 — 360/110 + Y2 acf1",
        "",
        f"Pooled acf1 Spearman **{p17['acf1_pooled']:+.3f}** (n_pairs={p17['n_pairs']}); "
        f"company-median **{p17['acf1_median']:+.3f}** (n={p17['n_cos_acf']}). "
        "High persistence matches the already-neg story. Not a new Y. "
        f"all-dark minus GROUP_0158/0172 is **{p17['dark_wo_hot']['rate']:.1%}** "
        "(below invoiced 6.3%) — the 9.14% dark lift *is* those two groups.",
        "",
        "| slice | n | n_pos | cos | P(Y=1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p17["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | {r['rate']:.3f} |"
        )
    lines.append("")
    return lines


def extra_md_lines_f(p18: dict) -> list[str]:
    lines = [
        "### 18 — already-neg inside 0158/0172 vs rest",
        "",
        "If already-neg Y2 is ~70% both in-hot and outside, fold-0 death is "
        "*more already-neg months in two dark groups*, not a new process.",
        "",
        f"Same process? **{p18['same_process']}**.",
        "",
        "| slice | n | n_pos | cos | P(Y=1) | share below 0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in p18["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | "
            f"{r['rate']:.3f} | {r['share_below0']:.3f} |"
        )
    lines.append("")
    return lines


def extra_md_lines_g(p19: dict) -> list[str]:
    lines = [
        "### 19 — 0158 vs 0172 (already-neg / clean onset)",
        "",
        "Both all-dark. 0158 is stickier already-neg *and* has leftover onset. "
        "0250 is the large invoiced 0% control. Not a renamed Y.",
        "",
        "| group | n | P(Y=1) | already n / rate | clean n / rate | below0 | cos |",
        "|---|---:|---:|---|---|---:|---:|",
    ]
    for r in p19["rows"]:
        lines.append(
            f"| `{r['group_id']}` | {r['n']} | {r['rate']:.3f} | "
            f"{r['n_already']} / {r['already_rate']:.3f} | "
            f"{r['n_clean']} / {r['clean_rate']:.3f} | "
            f"{r['share_below0']:.3f} | {r['n_cos']} |"
        )
    lines.append("")
    return lines


def extra_md_lines_h(p20: dict) -> list[str]:
    act = p20["clean_active"]
    lines = [
        "### 20 — holdout of the 4 groups + 0158 clean on days>0",
        "",
        "Holdout coverage only (LOW_POWER). Not a claim.",
        "",
        "| group | hold n | hold pos | hold cos | hold rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p20["hold_rows"]:
        rate = "nan" if r["rate"] != r["rate"] else f"{r['rate']:.3f}"
        lines.append(
            f"| `{r['group_id']}` | {r['n']} | {r['n_pos']} | {r['n_cos']} | {rate} |"
        )
    lines += [
        "",
        f"0158 clean + days>0: n={act['n']} pos={act['n_pos']} "
        f"rate=**{act['rate']:.1%}**. Onset is not empty-month noise. "
        "None of the 4 groups appear in holdout labeled months.",
        "",
    ]
    return lines


def extra_md_lines_i(p21: dict) -> list[str]:
    lines = [
        "### 21 — 0158 clean-now by so-far",
        "",
        "If onset is only <6, it is a short-book artifact. If it stays high later, "
        "0158 turns while already on book.",
        "",
        "| so-far | n | n_pos | P(Y=1) |",
        "|---|---:|---:|---:|",
    ]
    for r in p21["rows"]:
        rate = "nan" if r["rate"] != r["rate"] else f"{r['rate']:.3f}"
        lines.append(f"| {r['bucket']} | {r['n']} | {r['n_pos']} | {rate} |")
    lines.append("")
    return lines


def extra_md_lines_j(p22: dict) -> list[str]:
    lines = [
        "### 22 — 0172 clean so-far + 0158 already-neg so-far",
        "",
        "Compare 0158's late onset to 0172. 0158 already-neg by so-far shows when "
        "the pile sits below 0.",
        "",
        "| slice | so-far | n | n_pos | P(Y=1) | below0 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in p22["rows"]:
        rate = "nan" if r["rate"] != r["rate"] else f"{r['rate']:.3f}"
        b0 = "nan" if r["share_below0"] != r["share_below0"] else f"{r['share_below0']:.3f}"
        lines.append(
            f"| {r['slice']} | {r['bucket']} | {r['n']} | {r['n_pos']} | {rate} | {b0} |"
        )
    lines.append("")
    return lines


def extra_md_lines_k(p23: dict) -> list[str]:
    lines = [
        "### 23 — 0158 Y2pos inflow-crash share (label story, not X)",
        "",
        "If 0158 late positives sit at in-ratio ~0.36, that cluster is Y4-like even "
        "though panel Y2 is not. Future A only.",
        "",
        "| slice | n | crash defined | crash share | median in-ratio |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in p23["rows"]:
        cs = "nan" if r["crash_share"] != r["crash_share"] else f"{r['crash_share']:.3f}"
        ir = "nan" if r["med_in_ratio"] != r["med_in_ratio"] else f"{r['med_in_ratio']:.2f}"
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_crash_def']} | {cs} | {ir} |"
        )
    lines.append("")
    return lines


def extra_md_lines_l(p24: dict) -> list[str]:
    lines = [
        "### 24 — Y2pos × Y4 / Y9 in 0158/0172 vs rest",
        "",
        "If the cluster's Y2 pos were mostly Y4, fold-0 would be Y4's crash wearing "
        "a group mask. Y9 is a fee check only.",
        "",
        "| slice | n | Y4 defined / share | Y9 defined / share |",
        "|---|---:|---|---|",
    ]
    for r in p24["rows"]:
        y4s = "nan" if r["y4_share"] != r["y4_share"] else f"{r['y4_share']:.3f}"
        y9s = "nan" if r["y9_share"] != r["y9_share"] else f"{r['y9_share']:.3f}"
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['y4_def']} / {y4s} | {r['y9_def']} / {y9s} |"
        )
    lines.append("")
    return lines


def extra_md_lines_m(p25: dict) -> list[str]:
    d = p25["days_drop2"]
    lines = [
        "### 25 — legal-X medians in 0158 / 0172 / rest",
        "",
        f"Days CV after dropping only 0158+0172: **{d['cv']:.3f}** ± {d['sd']:.3f}. "
        "If this falls toward size 0.54, the 0.571 bump was the cluster.",
        "",
        "| slice | n | days | a_io | a_out6 | a_in3 | d_cust_hhi | f_ds_r |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p25["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['c_n_days_with_tx']:.2f} | "
            f"{r['a_io_ratio']:.3g} | {r['a_out6']:.4g} | {r['a_in3']:.4g} | "
            f"{r['d_cust_hhi']:.3g} | {r['f_ds_r']:.3g} |"
        )
    lines.append("")
    return lines


def extra_md_lines_n(p26: dict) -> list[str]:
    best = p26.get("best") or {}
    size = p26.get("size") or {}
    lines = [
        "### 26 — singles on rest (drop 0158+0172)",
        "",
        f"Best legal on rest: `{best.get('col')}` CV **{best.get('cv', float('nan')):.3f}** "
        f"vs size {size.get('cv', float('nan')):.3f} gap={p26['gap']:+.3f}. "
        "KEEP on rest would still need tail/monotone + ≥0.02. None expected.",
        "",
        "| feature | CV ± sd | coverage | folds |",
        "|---|---:|---:|---|",
    ]
    for r in p26["rows"]:
        bits = " ".join(f"{a:.3f}" for a in r["folds"])
        lines.append(
            f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['coverage']:.3f} | {bits} |"
        )
    lines.append("")
    return lines


def extra_md_lines_o(p27: dict) -> list[str]:
    lines = [
        "### 27 — HHI fold-2 inversion (rest; 0158/0172 have no HHI)",
        "",
        "HHI>0.975 is protective on Y2 (Y4's tail). Fold-2 inversion is a second "
        "cluster, not a Q5 KEEP.",
        "",
        "| slice | n | n_pos | cos | groups | P(Y=1) | med HHI |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p27["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | "
            f"{r['n_groups']} | {r['rate']:.3f} | {r['med_hhi']:.3f} |"
        )
    lines += ["", "Fold-2 HHI-defined groups by n_pos:", ""]
    for r in p27["top_f2"]:
        lines.append(f"- `{r['group_id']}` n={r['n']} pos={r['n_pos']} rate={r['rate']:.3f}")
    lines.append("")
    return lines


def extra_md_lines_p(p28: dict) -> list[str]:
    lines = [
        "### 28 — fold-2 HHI-hot groups (footnote, not a card)",
        "",
        "HHI tail is protective on Y2 except fold 2. These groups do not rewrite Q5.",
        "",
        "| group | n | P(Y=1) | cos | below0 | med days | med HHI | dark | fold |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p28["rows"]:
        hhi = "nan" if r["med_hhi"] != r["med_hhi"] else f"{r['med_hhi']:.3f}"
        lines.append(
            f"| `{r['group_id']}` | {r['n']} | {r['rate']:.3f} | {r['n_cos']} | "
            f"{r['share_below0']:.3f} | {r['med_days']:.1f} | {hhi} | "
            f"{r['share_dark']:.2f} | {r['fold']} |"
        )
    lines.append("")
    return lines


def extra_md_lines_q(p29: dict) -> list[str]:
    return [
        "### 29 — GROUP_0241 (tiny invoiced pile, not a card)",
        "",
        f"n={p29['n']} pos={p29['n_pos']} rate={p29['rate']:.1%} cos={p29['n_cos']}. "
        f"Already-neg {p29['already_n']} at {p29['already_rate']:.1%}; "
        f"clean {p29['clean_n']} at {p29['clean_rate']:.1%}. "
        f"Crash share {p29['crash_share']:.1%} med in-ratio {p29['med_in']:.2f} "
        "(not Y4's 0.36). Same persistence, 7 companies. Not Q5.",
        "",
    ]


def extra_md_lines_r(p30: dict) -> list[str]:
    return [
        "### 30 — 0158 per-company already-neg share",
        "",
        f"{p30['n_cos']} companies: {p30['n_any']} ever Y2-pos; "
        f"{p30['n_half']} spend ≥50% of labeled months below 0; "
        f"{p30['n_all']} always-below; {p30['n_never']} never-below. "
        f"Median share-below **{p30['med_below']:.0%}**, median company Y2 "
        f"**{p30['med_y2']:.0%}**. A handful of chronic names, not the whole 21.",
        "",
        "trail_length.md: Y2 short so-far 7.9% vs long 6.2% — fold-0 11.1% is "
        "not that short-book pile. Histogram not redone.",
        "",
    ]


def extra_md_lines_s(p31: dict) -> list[str]:
    return [
        "### 31 — fold-0 positives in the 0158 chronic names",
        "",
        f"Fold-0 Y2 pos={p31['n_f0']}. In 0158: {p31['n_0158']} "
        f"({p31['share_0158']:.0%}). In the {p31['n_chronic_cos']} chronic "
        f"(≥50% months below 0) names: {p31['n_chronic']} "
        f"({p31['share_chronic']:.0%}). A handful of names, not a group law.",
        "",
    ]


def extra_md_lines_t(p32: dict) -> list[str]:
    return [
        "### 32 — 0172 chronic + combined fold-0 share",
        "",
        f"0172: {p32['n_cos']} cos, {p32['n_any']} ever Y2, "
        f"{p32['n_chronic']} chronic, {p32['n_never']} never-below, "
        f"median share-below {p32['med_below']:.0%}. "
        f"0158+0172 chronic ({p32['n_both_cos']} names) hold "
        f"**{p32['share_both']:.0%}** of fold-0 positives "
        f"({p32['n_both_chronic']}/{p32['n_f0']}).",
        "",
    ]


def extra_md_lines_u(p33: dict) -> list[str]:
    lines = [
        "### 33 — legal-X medians on the 12 chronic names",
        "",
        "If days look like rest, the 0.571 single is not even these names.",
        "",
        "| slice | n | n_pos | med days | med io | med in3 | med out6 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p33["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['med_days']:.1f} | "
            f"{r['med_io']:.3g} | {r['med_in3']:.4g} | {r['med_out6']:.4g} |"
        )
    lines.append("")
    return lines


def extra_md_lines_v(p34: dict) -> list[str]:
    bits = " ".join(f"{a:.3f}" for a in p34["days_folds"])
    return [
        "### 34 — drop the 12 chronic companies only",
        "",
        f"n={p34['n']} pos={p34['n_pos']}. Days CV **{p34['days_cv']:.3f}** ± "
        f"{p34['days_sd']:.3f} vs size {p34['size_cv']:.3f} gap={p34['gap']:+.3f}. "
        f"Folds {bits}. If gap < 0.02 the 0.571 single *is* those 12 names.",
        "",
    ]


def extra_md_lines_w(p35: dict) -> list[str]:
    lines = [
        "### 35 — fold-0 rate without the 12 chronic names",
        "",
        "| slice | n | n_pos | P(Y=1) |",
        "|---|---:|---:|---:|",
    ]
    for r in p35["rows"]:
        lines.append(f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |")
    lines.append("")
    return lines


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


def registry_rows(p1, p2, p3, p4, p5, p6, verdict, ts: str, extra=None) -> list[dict]:
    out = []

    def add(model, metric, value, coverage, notes, families="A+C+D+F", split="train"):
        out.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": families,
                "y": Y_COL,
                "model": model,
                "split": split,
                "metric": metric,
                "value": _fmt(value),
                "coverage": (
                    f"{coverage:.4f}"
                    if isinstance(coverage, (int, float)) and np.isfinite(coverage)
                    else ""
                ),
                "notes": notes,
            }
        )

    for r in p1["rows"]:
        add(
            f"y2_why_{r['flag']}",
            "share_of_positives",
            r["share_of_pos_defined"],
            r["coverage_of_pos"],
            f"n_hi={r['n_hi']} n_def={r['n_defined']} n_pos={r['n_pos']}; "
            f"future-A crash and B-stressed are label story, not X; never B as X",
            families="-" if r["flag"] in STORE_B or r["flag"] in ("y3_stressed_now", "y3_labeled", "inflow_crash", Y4, Y9) else "A+C+D",
        )
    add(
        "y2_why_same_crash_as_y4",
        "flag",
        int(p1["same_crash_as_y4"]),
        1.0,
        f"crash_of_y2={p1['crash_of_y2']}; y4_of_y2={p1['y4_of_y2']}; "
        f"y2_of_y4={p1['y2_of_y4']}; jaccard_y4={p1['twos']['y2_x_y4']['jaccard']}",
        families="-",
    )
    for t in p2["tables"]:
        for r in t["rows"]:
            add(
                f"quintile_{t['x']}",
                f"y_rate_q{r['q']}",
                r["y_rate"],
                r["n"] / t["n"] if t["n"] else float("nan"),
                f"train labeled cuts; n={r['n']} pos={r['n_pos']} interval={r['interval']}; "
                f"shape={t['shape']}; never holdout cuts; never B",
                families=t["x"].split("_")[0].upper(),
            )
    for r in p3["rows"]:
        add(
            f"single_{r['col']}",
            "auroc",
            r["cv"],
            r["coverage"],
            f"sign from train fold; CV={r['cv']:.4f}±{r['sd']:.3f}; "
            f"train={r['train_auc']:.4f}; sign={r['train_sign']:+d}; "
            f"gap_size={r['gap_vs_size']:+.3f}; gap_gbm={r['gap_vs_gbm']:+.3f}; "
            f"never B; quote CV not holdout",
            families=r["col"].split("_")[0].upper() if not r["col"].startswith("log") else "A",
            split="cv5_group",
        )
    add(
        "size_log1p_a_in3",
        "auroc",
        p3["size_auc"],
        1.0,
        f"two-sided={p3['size_auc_max']:.4f}; PARK as X if ≥{SIZE_BAR:.2f}",
        families="A",
        split="train",
    )
    add(
        "fold0_y2_rate",
        "y_rate",
        p4["fold0"]["rate"],
        p4["fold0"]["n"] / p3["n_train"] if p3["n_train"] else float("nan"),
        p4["note"],
        families="-",
    )
    for r in p5["rows"]:
        add(
            f"leak_{r['col']}",
            f"spearman_{r['vs']}",
            r["spearman"],
            r["n"] / p3["n_train"] if p3["n_train"] else float("nan"),
            f"pearson={r['pearson']}; fail={r['fail']}; comparator only, never B as X",
            families=f"{r['col'].split('_')[0].upper()}-vs-B",
        )
    for r in p6["rows"]:
        add(
            f"dark_{r['slice']}",
            "y_rate",
            r["rate"],
            1.0,
            f"n={r['n']} pos={r['n_pos']} cos={r['n_cos']} size={r['size_auroc']}; "
            f"confirm={p6['confirm']}; not a new Y",
            families="-",
        )
    add(
        "y2_why_verdict",
        "decision",
        None,
        1.0,
        f"{verdict['decision']}; {verdict['reason']}; trees={verdict['trees']}",
        families="-",
    )
    extra = extra or {}
    p7 = extra.get("p7") or {}
    for r in p7.get("rows") or []:
        add(
            f"single_{r['col']}",
            "auroc",
            r["cv"],
            r["coverage"],
            f"Q6 clock; CV={r['cv']:.4f}±{r['sd']:.3f}; sign={r['train_sign']:+d}; "
            f"gap_size={r['gap_vs_size']:+.3f}; never B; 1-month only",
            families="A" if r["stem"].startswith("a_") else "C",
            split="cv5_group",
        )
    p8 = extra.get("p8") or {}
    if p8.get("residual"):
        add(
            "fold0_long_sofar_gap",
            "rate_gap",
            p8["residual"].get("long_gap"),
            1.0,
            f"f0_long={p8['residual'].get('f0_long_rate')} rest_long={p8['residual'].get('rest_long_rate')}; "
            f"late_trail={p8.get('late_trail_effect')} size_effect={p8.get('size_effect')}",
            families="-",
        )
    p9 = extra.get("p9") or {}
    for r in p9.get("rows") or []:
        add(
            f"y2_{r['slice']}",
            "y_rate",
            r["rate"],
            1.0,
            f"B decomp never X; n={r['n']} pos={r['n_pos']} persist={p9.get('persist')}",
            families="-",
        )
    p10 = extra.get("p10") or {}
    if p10.get("cv_body") is not None:
        add(
            "single_c_n_days_body",
            "auroc",
            p10["cv_body"],
            (p10.get("n_body") or 0) / p3["n_train"] if p3.get("n_train") else float("nan"),
            f"days<=17; CV={p10['cv_body']:.4f}±{p10.get('sd_body', float('nan')):.3f}; "
            f"busy_rate={p10.get('rate_hi')}; body_rate={p10.get('rate_lo')}; never B",
            families="C",
            split="cv5_group",
        )
    p11 = extra.get("p11") or {}
    for r in p11.get("rows") or []:
        add(
            f"fold0_{r['slice']}",
            "y_rate",
            r["rate"],
            1.0,
            f"n={r['n']} pos={r['n_pos']} below0={r['share_below0']}; {p11.get('note', '')}",
            families="-",
        )
    p12 = extra.get("p12") or {}
    for r in p12.get("oofs") or []:
        add(
            f"single_{r['col']}_clean_now",
            "auroc",
            r["cv"],
            1.0,
            f"clean-now leftover; CV={r['cv']:.4f}±{r['sd']:.3f}; n_pos={r['n_pos']}; "
            f"not a new Y; never B",
            families=r["col"].split("_")[0].upper() if not r["col"].startswith("log") else "A",
            split="cv5_group",
        )
    p13 = extra.get("p13") or {}
    for r in p13.get("rows") or []:
        add(
            f"fold0_t3_{r['group_id']}",
            "y_rate",
            r["rate"],
            1.0,
            f"n={r['n']} pos={r['n_pos']} cos={r['n_cos']} below0={r['share_below0']}; "
            f"the 4-group cluster; never B; not a new Y",
            families="-",
        )
    return out


def write_md(started, n_tr, n_pos, rate, p1, p2, p3, p4, p5, p6, verdict, extra=None) -> None:
    best = p3.get("best") or {}
    lines = [
        "# Y2 why — same crash as Y4, or a different turn?",
        "",
        f"- **When:** {started}",
        f"- **Agent:** `{AGENT}`",
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        f"- **Re-run:** `python -m analysis.evaluate.y2_why`",
        f"- **Holdout:** 72 companies, seed {FOLD_SEED}. Coverage only. Rates + singles on train.",
        f"- **Y:** `{Y_COL}` only from `targets.parquet` (train {n_tr} / {n_pos} / **{rate:.2%}**). "
        "Rejected Y2 columns not revived.",
        "- **X candidates:** `a_io_ratio` / `a_out6` / `log1p(a_in3)` / `c_n_days_with_tx` / "
        "`d_cust_hhi` / `f_ds_r` / `c_zero_in_month`. **Never B.** "
        "`b_liq` / `b_runway` / `b_below_0` are decomp + leak only.",
        f"- **Quote:** train group-fold CV (5 folds). Night GBM PARK {NIGHT_GBM_CV:.3f}±{NIGHT_GBM_SD:.3f} "
        f"(fold 0 = {NIGHT_FOLD0:.3f}) is the bar to *explain*, not beat with a tree.",
        "- **Brief:** Q3 turning (82→68) / Q5 why. Q6 only via honest 1-month lag.",
        "- Not bankruptcy. Not a 0–100.",
        "",
        "## Decision",
        "",
        f"**{verdict['decision']}.** {verdict['reason']}",
        "",
        f"Trees stay **{verdict['trees']}**.",
        "",
        "## Pass 1 — decompose positives",
        "",
        "High `a_out6` / `d_cust_hhi` = that company's own expanding p80 (months ≤ t, min 6). "
        "Inflow crash uses future `a_in3` to *name* the label — not X. "
        "`y3_stressed_now` reads family B to decompose the Y (the Y *is* B) — never as X.",
        "",
        "| flag | n_hi / n_defined pos | share of pos | coverage of pos |",
        "|---|---:|---:|---:|",
    ]
    for r in p1["rows"]:
        lines.append(
            f"| `{r['flag']}` | {r['n_hi']} / {r['n_defined']} | "
            f"{r['share_of_pos_defined']:.3f} | {r['coverage_of_pos']:.3f} |"
        )
    lines += [
        "",
        f"Positives median in-ratio **{p1['pos_med_in_ratio']:.2f}** (neg {p1['neg_med_in_ratio']:.2f}). "
        f"Y4 crash positives sat at 0.36 — compare.",
        "",
        "### 2×2 overlap",
        "",
        "| pair | n | both | a only | b only | neither | share a∈b | share b∈a | Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for t in p1["twos"].values():
        lines.append(
            f"| {t['name']} (`{t['a']}` × `{t['b']}`) | {t['n']} | {t['both']} | "
            f"{t['a_only']} | {t['b_only']} | {t['neither']} | "
            f"{t['share_a_that_are_b']:.3f} | {t['share_b_that_are_a']:.3f} | {t['jaccard']:.3f} |"
        )
    lines += [
        "",
        f"Same-crash-as-Y4 flag: **{p1['same_crash_as_y4']}** "
        f"(needs ≥70% of Y2 pos crashing *and* ≥50% of those also Y4).",
        "",
        "## Pass 2 — quintiles (train labeled cuts)",
        "",
    ]
    for t in p2["tables"]:
        lines += [
            f"### `{t['x']}`",
            "",
            f"n={t['n']} bins={t['n_bins']} shape=**{t['shape']}** "
            f"monotone↑={t['monotone_up']} tail_only={t['tail_only']}",
            "",
            "| Q | interval | n | n_pos | P(Y=1) | median X |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for r in t["rows"]:
            lines.append(
                f"| {r['q']} | {r['interval']} | {r['n']} | {r['n_pos']} | "
                f"{r['y_rate']:.3f} | {r['x_median']:.4g} |"
            )
        lines.append("")
    if p2.get("png"):
        lines.append(f"Plot: `{Path(p2['png']).name}`.")
        lines.append("")
    lines += [
        "## Pass 3 — single-feature CV",
        "",
        f"Size `log1p(|a_in3|)` train AUROC **{p3['size_auc']:.3f}** (two-sided {p3['size_auc_max']:.3f}). "
        f"Size ≥ {SIZE_BAR:.2f} → PARK as X. Night GBM {NIGHT_GBM_CV:.3f} is the bar to explain.",
        "",
        "| feature | CV AUROC ± sd | train | sign | coverage | gap vs size | gap vs 0.540 | folds |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in p3["rows"]:
        bits = " ".join(f"{f['auroc']:.3f}" for f in r["folds"])
        lines.append(
            f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['train_auc']:.3f} | "
            f"{r['train_sign']:+d} | {r['coverage']:.3f} | {r['gap_vs_size']:+.3f} | "
            f"{r['gap_vs_gbm']:+.3f} | {bits} |"
        )
    lines += [
        "",
        f"Best legal single: `{best.get('col')}` CV **{best.get('cv', float('nan')):.3f}**.",
        "",
        "## Pass 4 — why fold 0 inverted",
        "",
        p4["note"],
        "",
        "| fold | n | n_pos | P(Y=1) | cos | groups | late cos | short <12 cos | dark cos | HHI>0.975 cos | med group_size | med months-on-book | mean log1p(in3) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p4["rows"]:
        lines.append(
            f"| {r['fold']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} | {r['n_cos']} | "
            f"{r['n_groups']} | {r['share_cos_late']:.3f} | {r['share_cos_short']:.3f} | "
            f"{r['share_cos_dark']:.3f} | {r['share_cos_hhi_tail']:.3f} | "
            f"{r['median_group_size']:.1f} | {r['median_months_on_book']:.1f} | "
            f"{r['mean_log1p_in3']:.2f} |"
        )
    lines += [
        "",
        f"Night tree fold 0 = {NIGHT_FOLD0:.3f}. This table is the group mix, not a new fit.",
        "",
        "## Pass 5 — leak vs family B",
        "",
        f"Fail if Spearman |ρ| ≥ {LEAK_RHO} vs `b_liq` / `b_runway` / `b_below_0` (that is the Y).",
        "",
        "| feature | vs | n | Spearman | Pearson | fail |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in p5["rows"]:
        lines.append(
            f"| `{r['col']}` | `{r['vs']}` | {r['n']} | {r['spearman']:+.3f} | "
            f"{r['pearson']:+.3f} | {r['fail']} |"
        )
    lines += [
        "",
        f"Illegal B-copies: **{len(p5['fails'])}**.",
        "",
        "## Pass 6 — dark 470 vs invoiced 744",
        "",
        "Join-QA population (DuckDB book invoices). Confirm only. Not a new Y.",
        "",
        "| slice | n | n_pos | companies | base rate | size AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in p6["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | "
            f"{r['rate']:.3f} | {r['size_auroc']:.3f} |"
        )
    lines += [
        "",
        f"CONFIRM 9.14% vs 6.34%: **{p6['confirm']}** "
        f"(dark cos={p6['n_dark']}, invoiced={p6['n_book']}, confirm_470={p6['confirm_470']}).",
        "",
        "## Mapping (Q3 / Q5 / Q6)",
        "",
        mapping_sentence(p1, p2, p3, verdict, extra),
        "",
        "## Holdout coverage (LOW_POWER, not a claim)",
        "",
        extra.get("hold_line") or "Holdout labeled coverage only.",
        "",
        "## Parent return",
        "",
        f"- **Y4 crash overlap:** Y2pos crash={p1['crash_of_y2']:.1%}; Y2∩Y4={p1['y4_of_y2']:.1%} "
        f"(cov 19%); Y4∩Y2={p1['y2_of_y4']:.1%}; Jaccard 0.057; pos in-ratio "
        f"{p1['pos_med_in_ratio']:.2f} vs Y4's 0.36. **Different turn. CLOSE, do not merge.**",
        f"- **Best legal single:** `c_n_days_with_tx` CV **{best.get('cv', float('nan')):.3f}** "
        f"vs size {p3['size_auc_max']:.3f} / night GBM 0.540. Beats size by +0.03 but shape is "
        f"flat (Q4 bump); body ≤17 CV 0.426. Drop 12 chronic names → days **0.549** vs size "
        f"0.544 (gap +0.005). The 0.571 *is* those names. No Q5 KEEP. No B-copy (max |ρ| 0.43).",
        f"- **Fold 0 inverted:** rate 11.1% vs rest 6.3%. Not late-trail. 12 chronic dark names "
        f"in GROUP_0158/0172 hold **47%** of fold-0 positives. Drop the 4 T3 groups and fold-0 "
        f"rate is 6.7% = rest; group_size CV 0.375 → 0.570. Trees stay PARK.",
        "- **Q3/Q5 (no family B):** Y2 is already-negative persistence (82% of pos are below 0 at t; "
        "clean-now leftover 1.5%). The 82→68 *direction* is real; a non-B why is not. Dark 9.14% vs "
        "6.34% confirmed — and all-dark minus GROUP_0158/0172 is **5.5%** (below invoiced). "
        "Not a new Y. Q6 CLOSE (lag1 days = lag0).",
        "",
        "## Extra cuts",
        "",
    ]
    if extra and extra.get("md_lines"):
        lines.extend(extra["md_lines"])
    else:
        lines.append("None yet — next iteration in this module (lag-1 Q6, fold-0 trail vs size).")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def run(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Y2 why — crash overlap / singles / fold 0")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--no-md", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    started = datetime.now().isoformat(timespec="minutes")
    wall0 = datetime.now()
    print(f"y2_why start {started} seed={FOLD_SEED} y={Y_COL} agent={AGENT}")
    print("forbidden X = family B; labels from targets.parquet; no build_targets; no tree")
    print("META.forbidden_x_families", Y2_META.get("forbidden_x_families"))
    print(
        "leakage_check singles:",
        leakage_check(list(SINGLE_COLS), Y_COL, forbidden_prefixes=FORBIDDEN),
    )

    hold_ids = load_holdout()
    con = connect()
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    pop = dark_population(con)
    con.close()
    print(
        f"dark pop train_dark={pop['n_train_dark']} book={pop['n_book_train']} "
        f"confirm_470={pop['confirm_470']} 360={pop['n_360_alldark']} 110={pop['n_110_mixed']}"
    )

    store = load_store()
    y = load_y()
    panel = store.merge(y, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    panel = add_own_cuts(panel)
    panel = add_label_story(panel)
    # lags prepared for the next cut (honest 1-month Q6) — computed, scored later
    panel = add_lags(panel, ["a_io_ratio", "c_n_days_with_tx"], (1,))

    is_hold = panel["company_id"].astype(str).isin(hold_ids)
    labeled = panel[Y_COL].notna()
    train_lab = (~is_hold) & labeled & panel["fold"].notna()
    hold_lab = is_hold & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    if set(panel.loc[train_lab, "company_id"].astype(str)) & hold_ids:
        raise RuntimeError("holdout leaked into train labeled")

    n_tr = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, Y_COL] == 1).sum())
    rate = float(panel.loc[train_lab, Y_COL].mean()) if n_tr else float("nan")
    print(
        f"train labeled={n_tr} pos={n_pos} rate={rate:.4f}  "
        f"holdout labeled={int(hold_lab.sum())} pos={int((panel.loc[hold_lab, Y_COL] == 1).sum())} "
        f"LOW_POWER  hold_cos={len(hold_ids)}"
    )

    lab = panel.loc[train_lab].copy()
    assert_no_holdout(lab["company_id"])

    p1 = pass1_decompose(lab)
    p2 = pass2_quintiles(panel, train_lab, args.no_plot)
    p3 = pass3_singles(panel, train_lab)
    p4 = pass4_fold0(panel, train_lab, pop["train_dark_ids"])
    p5 = pass5_leak(panel, train_lab)
    p6 = pass6_dark(panel, train_lab, pop)
    p7 = pass7_lag1_q6(panel, train_lab)
    p8 = pass8_fold0_trail_size(panel, train_lab)
    p9 = pass9_already_neg(panel, train_lab)
    p10 = pass10_days_honesty(panel, train_lab)
    p11 = pass11_fold0_t3(panel, train_lab)
    p12 = pass12_clean_leftover(panel, train_lab)
    p13 = pass13_four_groups(panel, train_lab, pop["train_dark_ids"])
    p14 = pass14_days_x_neg(panel, train_lab)
    p16 = pass16_drop_cluster(panel, train_lab, p13["group_ids"])
    p17 = pass17_dark_mix_acf(panel, train_lab, pop)
    p18 = pass18_hot_already_neg(panel, train_lab)
    p19 = pass19_hot_split(panel, train_lab)
    p20 = pass20_holdout_and_active(panel, train_lab, hold_lab)
    p21 = pass21_0158_clean_sofar(panel, train_lab)
    p22 = pass22_0172_and_0158_already_sofar(panel, train_lab)
    p23 = pass23_0158_crash_story(panel, train_lab)
    p24 = pass24_hot_y4_y9(panel, train_lab)
    p25 = pass25_hot_medians(panel, train_lab)
    p26 = pass26_rest_singles(panel, train_lab)
    p27 = pass27_hhi_fold2(panel, train_lab)
    p28 = pass28_f2_hhi_groups(panel, train_lab, pop)
    p29 = pass29_0241_story(panel, train_lab)
    p30 = pass30_0158_cos(panel, train_lab)
    p31 = pass31_chronic_share_of_fold0(panel, train_lab, p30["chronic_ids"])
    p32 = pass32_0172_chronic(panel, train_lab, p30["chronic_ids"])
    p33 = pass33_chronic_days(panel, train_lab, p30["chronic_ids"])
    p34 = pass34_drop_12(panel, train_lab, p30["chronic_ids"])
    p35 = pass35_fold0_drop12(panel, train_lab, p30["chronic_ids"])
    verdict = decide(p1, p2, p3, p5)

    ho_n = int(hold_lab.sum())
    ho_pos = int((panel.loc[hold_lab, Y_COL] == 1).sum())
    ho_cos = int(panel.loc[hold_lab, "company_id"].nunique())
    extra = {
        "hold_line": (
            f"Holdout labeled `{Y_COL}`: n={ho_n} pos={ho_pos} cos={ho_cos} "
            f"rate={((ho_pos / ho_n) if ho_n else float('nan')):.3f}. "
            "LOW_POWER — not a KEEP claim. Quote train CV only."
        ),
        "md_lines": extra_md_lines(p7, p8, p9, p10)
        + extra_md_lines_b(p11, p12)
        + extra_md_lines_c(p13, p14)
        + extra_md_lines_d(p16)
        + extra_md_lines_e(p17)
        + extra_md_lines_f(p18)
        + extra_md_lines_g(p19)
        + extra_md_lines_h(p20)
        + extra_md_lines_i(p21)
        + extra_md_lines_j(p22)
        + extra_md_lines_k(p23)
        + extra_md_lines_l(p24)
        + extra_md_lines_m(p25)
        + extra_md_lines_n(p26)
        + extra_md_lines_o(p27)
        + extra_md_lines_p(p28)
        + extra_md_lines_q(p29)
        + extra_md_lines_r(p30)
        + extra_md_lines_s(p31)
        + extra_md_lines_t(p32)
        + extra_md_lines_u(p33)
        + extra_md_lines_v(p34)
        + extra_md_lines_w(p35),
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "p12": p12,
        "p13": p13,
        "p14": p14,
        "p16": p16,
        "p17": p17,
        "p18": p18,
        "p19": p19,
        "p20": p20,
        "p21": p21,
        "p22": p22,
        "p23": p23,
        "p24": p24,
        "p25": p25,
        "p26": p26,
        "p27": p27,
        "p28": p28,
        "p29": p29,
        "p30": p30,
        "p31": p31,
        "p32": p32,
        "p33": p33,
        "p34": p34,
        "p35": p35,
    }
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if not args.no_registry:
        rows = registry_rows(p1, p2, p3, p4, p5, p6, verdict, ts, extra=extra)
        append_registry(rows)
        print(f"appended {len(rows)} registry rows (train-only metrics)")
    if not args.no_md:
        write_md(started, n_tr, n_pos, rate, p1, p2, p3, p4, p5, p6, verdict, extra)

    elapsed = (datetime.now() - wall0).total_seconds()
    quote = {
        "n_tr": n_tr,
        "n_pos": n_pos,
        "rate": rate,
        "crash_of_y2": p1["crash_of_y2"],
        "y4_of_y2": p1["y4_of_y2"],
        "y2_of_y4": p1["y2_of_y4"],
        "same_crash_as_y4": p1["same_crash_as_y4"],
        "best_col": (p3.get("best") or {}).get("col"),
        "best_cv": (p3.get("best") or {}).get("cv"),
        "size_auc": p3["size_auc_max"],
        "fold0_rate": p4["fold0"]["rate"],
        "dark_confirm": p6["confirm"],
        "decision": verdict["decision"],
        "keep_q6": p7["keep_q6"],
        "days_lag1": (p7.get("days_lag1") or {}).get("cv"),
        "lag_lift": p7.get("lag_lift"),
        "late_trail_effect": p8["late_trail_effect"],
        "size_effect": p8["size_effect"],
        "already_neg_share": (
            p9["already"]["n_pos"] / max(p9["already"]["n_pos"] + p9["clean"]["n_pos"], 1)
        ),
        "clean_now_rate": p9["clean"]["rate"],
        "f0_t3_rate": p11["f0_t3"]["rate"],
        "rest_t3_rate": p11["rest_t3"]["rate"],
        "leftover_fds_cv": next(r["cv"] for r in p12["oofs"] if r["col"] == "f_ds_r"),
        "n_f0_t3_groups": p13["n_groups"],
        "days_busy_below": p14["share_below_busy"],
        "elapsed_s": elapsed,
    }
    print("\nQUOTE")
    print(json.dumps(quote, indent=2, default=str))
    print(f"elapsed {elapsed:.1f}s — stay on this module for more cuts")
    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "p11": p11,
        "p12": p12,
        "p13": p13,
        "p14": p14,
        "p16": p16,
        "p17": p17,
        "p18": p18,
        "p19": p19,
        "p20": p20,
        "p21": p21,
        "p22": p22,
        "p23": p23,
        "p24": p24,
        "p25": p25,
        "p26": p26,
        "p27": p27,
        "p28": p28,
        "p29": p29,
        "p30": p30,
        "p31": p31,
        "p32": p32,
        "p33": p33,
        "p34": p34,
        "p35": p35,
        "verdict": verdict,
        "quote": quote,
        "n_tr": n_tr,
        "n_pos": n_pos,
        "rate": rate,
        "pop": pop,
    }


if __name__ == "__main__":
    run()
