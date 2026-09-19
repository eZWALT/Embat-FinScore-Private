"""Q3 growth / io_ratio — leftover turning, or mean-reversion / SIZE / NaNs?

NORTH_STAR: `a_io_ratio` = min(3, in3 / max(out3, 1)).
`a_growth_3` = clip(in3 / in3_{t-3} − 1, −1, 1) if in3_{t-3}>0 else NaN.
`a_growth_12` = YoY of the trailing-3m inflow window (needs month 15+).
`a_net_margin` was dropped as redundant with io_ratio (|ρ| cluster 0.8).
All three are keep-list representatives. growth_12 cov only 26.5%.
Javier: levels persist, change mean-reverts (naive inflow −40% AUC < 0.5).
Y4 is a future inflow crash after monopoly customers — do not treat
growth as Y4 X and do not merge. Night engine days 0.711; 15-col card
does not include growth/io. Night Y3 quote stays 0.762 / 0.752.

Question: leftover Q3 turning after size and days, or mean reversion /
SIZE / short-book NaNs?

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.growth_qa

Owned: analysis/evaluate/growth_qa.py, analysis/outputs/growth_qa.md,
optional one PNG, append-only registry, overnight/waves/wave4_growth.md (end).

Do not edit recency_qa / zero_in_qa / gap_sd_qa / cashflow.py / y4_why.
Do not invent y_growth. Do not put these on tonight's 15-col card.
"""
from __future__ import annotations

import csv
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
    auroc,
    group_folds,
    load_holdout,
)
from analysis.features.common import ANALYSIS, DATA, connect
from analysis.targets.y11_dark import book_invoice_ids

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "growth_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "growth_quintiles.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "f4320aab"
WAVE = "4"
ROUND = "R4"
MODEL = "growth_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
TWIN_RHO = 0.80
ICC_TRAIT = 0.85
ICC_SHOCK = 0.50
MIN_POS = 50
MIN_ACF_PAIRS = 4
CLIP_IO = 3.0
GROWTH_CLIP = 1.0
NAIVE_CRASH = -0.40
COV_IO = 0.885
COV_G3 = 0.640
COV_G12 = 0.265
PANEL_END = pd.Timestamp("2026-08-01")
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
WRITE_WAVE = False

STEMS = ("a_io_ratio", "a_growth_3", "a_growth_12")

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_out3",
    "a_op_in",
    "a_io_ratio",
    "a_growth_3",
    "a_growth_12",
    "a_net_margin",
    "a_n_tx",
    "c_n_days_with_tx",
    "b_below_0",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}"
    return f"{float(x):.{nd}f}"


def _pp(x, nd=1) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{100.0 * float(x):.{nd}f}%"


def _pct(n, d) -> float:
    if d is None or d == 0:
        return float("nan")
    return float(n) / float(d)


def _md_table(rows: list[dict], cols: list[str] | None = None) -> str:
    if not rows:
        return "_(empty)_\n"
    if cols is None:
        cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"]).astype("datetime64[ns]")
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def pearson(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="pearson"))


def median_acf(series: pd.Series, company: pd.Series, lag: int) -> float:
    df = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    vals = []
    for _, g in df.groupby("co", sort=False):
        x = g["x"].to_numpy(dtype=float)
        if len(x) <= lag:
            continue
        aa, bb = x[:-lag], x[lag:]
        m = np.isfinite(aa) & np.isfinite(bb)
        if int(m.sum()) < MIN_ACF_PAIRS:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


def icc_anova(series: pd.Series, company: pd.Series) -> dict:
    s = pd.DataFrame(
        {"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}
    ).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc), "var_w": float(var_w), "var_b": float(var_b), "k": k}


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


def signed_oof_auroc(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    mask: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict:
    """Group-fold CV AUROC. Sign from the train side of each fold."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    low = n_pos < MIN_POS or n_neg == 0
    fold_rows = []
    aucs = []
    if low:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "n_folds": 0,
            "folds": fold_rows,
            "train_sign": 0,
            "train_auc": float("nan"),
            "n_defined": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "low_power": True,
        }
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va": int(va.sum()),
                "n_pos": int((va & (y == 1)).sum()),
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", [])
    )


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
    """Train-defined OLS residual of y on 1+ predictors. Caller must pass train."""
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "slope": [], "intercept": float("nan")}
    n_x = len(xs)
    if int(ok.sum()) < max(20, n_x + 5):
        return resid, info
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)]
    )
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    resid.loc[ok] = Y - (X @ beta)
    info["intercept"] = float(beta[0])
    info["slope"] = [float(b) for b in beta[1:]]
    return resid, info


def cv_of(rec: dict | None) -> float:
    if rec is None or rec.get("low_power"):
        return float("nan")
    return rec["cv"]


def chronic_ids(tr: pd.DataFrame) -> list[str]:
    """12 chronic dark Y2 names: ≥50% labeled months already below 0 in 0158/0172."""
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    sl = lab & hot
    g = (
        tr.loc[sl, ["company_id"]]
        .assign(below=below[sl].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    return [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]


def quintile_table(df: pd.DataFrame, mask: pd.Series, x_col: str, y_col: str) -> dict:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = mask & x.notna() & y.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    tr = pd.DataFrame({x_col: x[m], y_col: y[m]})
    empty = {
        "x": x_col,
        "y": y_col,
        "n": int(len(tr)),
        "rows": [],
        "n_bins": 0,
        "monotone_up": None,
        "monotone_down": None,
        "shape": "na",
        "q1": float("nan"),
        "q5": float("nan"),
        "gap": float("nan"),
    }
    if len(tr) < 50 or tr[x_col].nunique() < 3:
        return empty
    cats, bins = pd.qcut(tr[x_col], 5, retbins=True, duplicates="drop")
    tr = tr.copy()
    tr["q"] = cats
    rows = []
    for i, (q, g) in enumerate(tr.groupby("q", observed=True), start=1):
        rows.append(
            {
                "q": i,
                "interval": str(q),
                "n": int(len(g)),
                "n_pos": int((g[y_col] == 1).sum()),
                "y_rate": float(g[y_col].mean()),
                "x_median": float(g[x_col].median()),
            }
        )
    rates = [r["y_rate"] for r in rows]
    monotone_up = (
        all(a <= b + 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    )
    monotone_down = (
        all(a >= b - 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    )
    shape = "flat"
    if monotone_up:
        shape = "monotone_up"
    elif monotone_down:
        shape = "monotone_down"
    q1 = rates[0] if rates else float("nan")
    q5 = rates[-1] if rates else float("nan")
    gap = q5 - q1 if rates else float("nan")
    return {
        "x": x_col,
        "y": y_col,
        "n": int(len(tr)),
        "rows": rows,
        "n_bins": len(rows),
        "monotone_up": monotone_up,
        "monotone_down": monotone_down,
        "shape": shape,
        "q1": q1,
        "q5": q5,
        "gap": gap,
        "bins": [float(b) for b in bins],
    }


def add_panel_lags(df: pd.DataFrame, stems: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in stems:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    missing = [c for c in STORE_COLS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
    ykeep = ["company_id", "period", Y2, Y3, Y4]
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel = panel.sort_values(["company_id", "period"]).reset_index(drop=True)
    in3 = pd.to_numeric(panel["a_in3"], errors="coerce")
    out3 = pd.to_numeric(panel["a_out3"], errors="coerce")
    panel["log_in3"] = np.log1p(in3.clip(lower=0))
    panel["io_unclip"] = in3 / np.maximum(out3, 1.0)
    panel["io_clip3"] = (pd.to_numeric(panel["a_io_ratio"], errors="coerce") >= CLIP_IO - 1e-12).astype(
        float
    )
    g = panel.groupby("company_id", sort=False)
    panel["in3_l3"] = g["a_in3"].shift(3)
    panel["in3_l12"] = g["a_in3"].shift(12)
    panel["g3_hat"] = np.where(
        panel["in3_l3"] > 0,
        (in3 / panel["in3_l3"] - 1.0).clip(-GROWTH_CLIP, GROWTH_CLIP),
        np.nan,
    )
    panel["g12_hat"] = np.where(
        panel["in3_l12"] > 0,
        (in3 / panel["in3_l12"] - 1.0).clip(-GROWTH_CLIP, GROWTH_CLIP),
        np.nan,
    )
    panel["io_hat"] = np.minimum(CLIP_IO, in3 / np.maximum(out3, 1.0))
    g3 = pd.to_numeric(panel["a_growth_3"], errors="coerce")
    g12 = pd.to_numeric(panel["a_growth_12"], errors="coerce")
    panel["g3_clip_lo"] = (g3 <= -GROWTH_CLIP + 1e-12).astype(float)
    panel["g3_clip_hi"] = (g3 >= GROWTH_CLIP - 1e-12).astype(float)
    panel["g12_clip_lo"] = (g12 <= -GROWTH_CLIP + 1e-12).astype(float)
    panel["g12_clip_hi"] = (g12 >= GROWTH_CLIP - 1e-12).astype(float)
    panel["naive_crash40"] = np.where(g3.notna(), (g3 <= NAIVE_CRASH).astype(float), np.nan)
    panel["neg_growth3"] = -g3
    panel["cal_month"] = panel["period"].dt.month
    fm = pd.to_datetime(panel["first_month"])
    panel["mob_first"] = (panel["period"].dt.year - fm.dt.year) * 12 + (
        panel["period"].dt.month - fm.dt.month
    ) + 1
    first_grid = panel.groupby("company_id")["period"].transform("min")
    panel["trail_len"] = (
        (PANEL_END.year - first_grid.dt.year) * 12 + (PANEL_END.month - first_grid.dt.month) + 1
    )
    panel["short_book"] = (panel["trail_len"] < 15).astype(np.int8)
    panel["so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    panel["late_arrival"] = (first_grid > pd.Timestamp("2024-09-01")).astype(np.int8)
    y3 = pd.to_numeric(panel[Y3], errors="coerce")
    panel["y3_next"] = y3.groupby(panel["company_id"], sort=False).shift(-1)
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


# ---------------------------------------------------------------------------
# Pass 1 — coverage (confirm 88.5 / 64.0 / 26.5)
# ---------------------------------------------------------------------------
def pass1_coverage(tr: pd.DataFrame, panel: pd.DataFrame) -> dict:
    rows = []
    quotes = {"a_io_ratio": COV_IO, "a_growth_3": COV_G3, "a_growth_12": COV_G12}
    store = {}
    for col in STEMS:
        x = pd.to_numeric(tr[col], errors="coerce")
        cov = float(x.notna().mean())
        n_def = int(x.notna().sum())
        n_co = int(tr.loc[x.notna(), "company_id"].nunique())
        n_co_all = int(tr["company_id"].nunique())
        q = quotes[col]
        ok = bool(np.isfinite(cov) and abs(cov - q) < 0.008)
        rec = {
            "col": col,
            "n_cm": int(len(tr)),
            "n_def": n_def,
            "cov": cov,
            "quote": q,
            "ok": ok,
            "n_co": n_co,
            "cov_co": _pct(n_co, n_co_all),
        }
        store[col] = rec
        rows.append(
            {
                "col": col,
                "n_def": f"{n_def:,}",
                "cov": _pp(cov),
                "quote": _pp(q),
                "confirm": "YES" if ok else "NO",
                "n_co": f"{n_co:,}",
                "cov_co": _pp(rec["cov_co"]),
            }
        )

    ho = panel[panel["split"] == "holdout"]
    ho_rows = []
    for col in STEMS:
        x = pd.to_numeric(ho[col], errors="coerce")
        ho_rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "n_def": f"{int(x.notna().sum()):,}",
            }
        )

    # short-book hole: growth_12 needs so_far >= 15 and in3_{t-12} > 0
    g12 = pd.to_numeric(tr["a_growth_12"], errors="coerce")
    short = tr["short_book"] == 1
    late = tr["late_arrival"] == 1
    so15 = tr["so_far"] >= 15
    hole = []
    for name, mask in (
        ("all", pd.Series(True, index=tr.index)),
        ("short_trail<15", short),
        ("long_trail>=15", ~short),
        ("late_arrival", late),
        ("on_time_2024-09", ~late),
        ("so_far<15", ~so15),
        ("so_far>=15", so15),
    ):
        sl = tr[mask]
        hole.append(
            {
                "slice": name,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "io": _pp(float(pd.to_numeric(sl["a_io_ratio"], errors="coerce").notna().mean()) if len(sl) else float("nan")),
                "g3": _pp(float(pd.to_numeric(sl["a_growth_3"], errors="coerce").notna().mean()) if len(sl) else float("nan")),
                "g12": _pp(float(pd.to_numeric(sl["a_growth_12"], errors="coerce").notna().mean()) if len(sl) else float("nan")),
            }
        )
    g12_short = float(g12[short].notna().mean()) if short.any() else float("nan")
    g12_so = float(g12[~so15].notna().mean()) if (~so15).any() else float("nan")
    empty_short = bool(np.isfinite(g12_short) and g12_short < 0.02)
    empty_early = bool(np.isfinite(g12_so) and g12_so < 0.02)
    n_short_co = int(tr.loc[short, "company_id"].nunique())
    n_late_co = int(tr.loc[late, "company_id"].nunique())
    prose = (
        f"Train coverage `a_io_ratio` {_pp(store['a_io_ratio']['cov'])} "
        f"({'CONFIRM 88.5%' if store['a_io_ratio']['ok'] else 'off 88.5%'}), "
        f"`a_growth_3` {_pp(store['a_growth_3']['cov'])} "
        f"({'CONFIRM 64.0%' if store['a_growth_3']['ok'] else 'off 64.0%'}), "
        f"`a_growth_12` {_pp(store['a_growth_12']['cov'])} "
        f"({'CONFIRM 26.5%' if store['a_growth_12']['ok'] else 'off 26.5%'}). "
        f"growth_12 on short-trail companies {_pp(g12_short)} "
        f"({'EMPTY — CLOSE as Q6 hole' if empty_short else 'not empty'}); "
        f"on so_far<15 {_pp(g12_so)} "
        f"({'EMPTY as formula (needs month 15+)' if empty_early else 'not empty'}). "
        f"Short-trail companies {n_short_co}; late-arrival {n_late_co}."
    )
    print(prose)
    return {
        "rows": rows,
        "ho_rows": ho_rows,
        "hole": hole,
        "store": store,
        "g12_short": g12_short,
        "g12_so": g12_so,
        "empty_short": empty_short,
        "empty_early": empty_early,
        "n_short_co": n_short_co,
        "n_late_co": n_late_co,
        "n_cm": int(len(tr)),
        "n_co": int(tr["company_id"].nunique()),
        "ho_n_cm": int(len(ho)),
        "ho_n_co": int(ho["company_id"].nunique()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — formula vs store
# ---------------------------------------------------------------------------
def pass2_formula(tr: pd.DataFrame) -> dict:
    rows = []
    pairs = (
        ("a_io_ratio", "io_hat"),
        ("a_growth_3", "g3_hat"),
        ("a_growth_12", "g12_hat"),
    )
    store = {}
    for a, b in pairs:
        aa = pd.to_numeric(tr[a], errors="coerce")
        bb = pd.to_numeric(tr[b], errors="coerce")
        both = aa.notna() & bb.notna()
        only_a = aa.notna() & bb.isna()
        only_b = aa.isna() & bb.notna()
        delta = (aa[both] - bb[both]).abs()
        rec = {
            "col": a,
            "n_both": int(both.sum()),
            "max_abs": float(delta.max()) if both.any() else float("nan"),
            "p50_abs": float(delta.median()) if both.any() else float("nan"),
            "exact": float((delta <= 1e-12).mean()) if both.any() else float("nan"),
            "only_store": int(only_a.sum()),
            "only_hat": int(only_b.sum()),
            "rho": spearman(aa, bb),
        }
        rec["ok"] = bool(
            rec["n_both"] > 0
            and np.isfinite(rec["max_abs"])
            and rec["max_abs"] < 1e-6
            and rec["only_store"] == 0
            and rec["only_hat"] == 0
        )
        store[a] = rec
        rows.append(
            {
                "col": a,
                "n_both": f"{rec['n_both']:,}",
                "max|Δ|": f"{rec['max_abs']:.2e}" if np.isfinite(rec["max_abs"]) else "—",
                "exact": _pp(rec["exact"]),
                "only store": rec["only_store"],
                "only hat": rec["only_hat"],
                "ok": "YES" if rec["ok"] else "NO",
            }
        )
        print(
            f"formula {a}: max|Δ|={rec['max_abs']:.2e} exact={rec['exact']:.4f} "
            f"only_store={rec['only_store']} only_hat={rec['only_hat']}"
        )

    # clip identity: io_hat sits at 3 when unclip >= 3
    un = pd.to_numeric(tr["io_unclip"], errors="coerce")
    io = pd.to_numeric(tr["a_io_ratio"], errors="coerce")
    sat = un.notna() & (un >= CLIP_IO - 1e-12)
    sat_ok = bool(sat.any() and float((io[sat] >= CLIP_IO - 1e-12).mean()) > 0.999)
    all_ok = all(store[c]["ok"] for c in STEMS)
    prose = (
        f"Store vs in3/out3 reconstruction: "
        + ", ".join(f"{c} max|Δ|={store[c]['max_abs']:.2e}" for c in STEMS)
        + f". Clip identity on unclip≥3: {'YES' if sat_ok else 'NO'}. "
        + (
            "Formulas match cashflow.py — not a rewrite."
            if all_ok
            else "MISMATCH — stop and report; do not rewrite cashflow.py tonight."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "all_ok": all_ok,
        "sat_ok": sat_ok,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman vs size / days / net_margin
# ---------------------------------------------------------------------------
def pass3_rho(tr: pd.DataFrame) -> dict:
    pairs = []
    for col in STEMS:
        for other, lab in (
            ("log_in3", "log1p(a_in3)"),
            ("a_in3", "a_in3"),
            ("c_n_days_with_tx", "c_n_days_with_tx"),
            ("a_net_margin", "a_net_margin"),
            ("a_n_tx", "a_n_tx"),
        ):
            rho = spearman(tr[col], tr[other])
            pairs.append(
                {
                    "pair": f"{col} vs {lab}",
                    "rho": rho,
                    "twin": bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO),
                    "size": bool(other == "log_in3" and np.isfinite(rho) and abs(rho) >= SIZE_RHO),
                    "|ρ|": _f(abs(rho) if np.isfinite(rho) else float("nan")),
                    "call": (
                        "TWIN"
                        if np.isfinite(rho) and abs(rho) >= TWIN_RHO
                        else (
                            "SIZE"
                            if other == "log_in3" and np.isfinite(rho) and abs(rho) >= SIZE_RHO
                            else "—"
                        )
                    ),
                }
            )
    # among the three
    for a, b in (
        ("a_io_ratio", "a_growth_3"),
        ("a_io_ratio", "a_growth_12"),
        ("a_growth_3", "a_growth_12"),
        ("a_io_ratio", "a_net_margin"),
    ):
        rho = spearman(tr[a], tr[b])
        pairs.append(
            {
                "pair": f"{a} vs {b}",
                "rho": rho,
                "twin": bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO),
                "size": False,
                "|ρ|": _f(abs(rho) if np.isfinite(rho) else float("nan")),
                "call": "TWIN" if np.isfinite(rho) and abs(rho) >= TWIN_RHO else "—",
            }
        )

    def _rho(a, b) -> float:
        return spearman(tr[a], tr[b])

    rho_io_size = _rho("a_io_ratio", "log_in3")
    rho_g3_size = _rho("a_growth_3", "log_in3")
    rho_g12_size = _rho("a_growth_12", "log_in3")
    rho_io_days = _rho("a_io_ratio", "c_n_days_with_tx")
    rho_g3_days = _rho("a_growth_3", "c_n_days_with_tx")
    rho_g12_days = _rho("a_growth_12", "c_n_days_with_tx")
    rho_io_nm = _rho("a_io_ratio", "a_net_margin")
    rho_io_in3 = _rho("a_io_ratio", "a_in3")
    io_size = bool(np.isfinite(rho_io_size) and abs(rho_io_size) >= SIZE_RHO)
    g3_size = bool(np.isfinite(rho_g3_size) and abs(rho_g3_size) >= SIZE_RHO)
    g12_size = bool(np.isfinite(rho_g12_size) and abs(rho_g12_size) >= SIZE_RHO)
    io_twin_nm = bool(np.isfinite(rho_io_nm) and abs(rho_io_nm) >= TWIN_RHO)
    io_twin_days = bool(np.isfinite(rho_io_days) and abs(rho_io_days) >= TWIN_RHO)
    g3_twin_days = bool(np.isfinite(rho_g3_days) and abs(rho_g3_days) >= TWIN_RHO)
    prose = (
        f"Spearman vs log1p(a_in3): io {_f(rho_io_size)} "
        f"({'SIZE' if io_size else 'not SIZE'}), "
        f"g3 {_f(rho_g3_size)} ({'SIZE' if g3_size else 'not SIZE'}), "
        f"g12 {_f(rho_g12_size)} ({'SIZE' if g12_size else 'not SIZE'}). "
        f"vs days: io {_f(rho_io_days)} g3 {_f(rho_g3_days)} g12 {_f(rho_g12_days)}. "
        f"io vs net_margin {_f(rho_io_nm)} "
        f"({'TWIN |ρ|≥0.80 — keep-list already dropped net_margin' if io_twin_nm else 'not a twin'})."
    )
    print(prose)
    return {
        "rows": [
            {"pair": r["pair"], "ρ": _f(r["rho"]), "call": r["call"]} for r in pairs
        ],
        "pairs": pairs,
        "rho_io_size": rho_io_size,
        "rho_g3_size": rho_g3_size,
        "rho_g12_size": rho_g12_size,
        "rho_io_days": rho_io_days,
        "rho_g3_days": rho_g3_days,
        "rho_g12_days": rho_g12_days,
        "rho_io_nm": rho_io_nm,
        "rho_io_in3": rho_io_in3,
        "io_size": io_size,
        "g3_size": g3_size,
        "g12_size": g12_size,
        "io_twin_nm": io_twin_nm,
        "io_twin_days": io_twin_days,
        "g3_twin_days": g3_twin_days,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — group-fold AUROC
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "a_io_ratio": tr["a_io_ratio"],
        "a_growth_3": tr["a_growth_3"],
        "a_growth_12": tr["a_growth_12"],
        "neg_growth3": tr["neg_growth3"],
        "naive_crash40": tr["naive_crash40"],
        "io_unclip": tr["io_unclip"],
        "io_clip3": tr["io_clip3"],
        "a_net_margin": tr["a_net_margin"],
        "log1p_a_in3": tr["log_in3"],
        "a_in3": tr["a_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
    }
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sd": _f(res["sd"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                    "train": _f(res["train_auc"]) if not res["low_power"] else "—",
                    "folds": fold_bits(res) if not res["low_power"] else "—",
                }
            )
            print(
                f"AUROC {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']} sign={res['train_sign']}"
            )

    def _cv(y, feat) -> float:
        return cv_of(store[(y, feat)])

    size_y3 = _cv(Y3, "log1p_a_in3")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    io_y3 = _cv(Y3, "a_io_ratio")
    g3_y3 = _cv(Y3, "a_growth_3")
    g12_y3 = _cv(Y3, "a_growth_12")
    neg_y3 = _cv(Y3, "neg_growth3")
    naive_y3 = _cv(Y3, "naive_crash40")
    io_y2 = _cv(Y2, "a_io_ratio")
    g3_y2 = _cv(Y2, "a_growth_3")
    g12_y2 = _cv(Y2, "a_growth_12")
    size_y2 = _cv(Y2, "log1p_a_in3")
    days_y2 = _cv(Y2, "c_n_days_with_tx")
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.03)

    # same-row size/days (coverage-fair)
    same = []
    for feat in STEMS:
        lab = tr[Y3].notna() & pd.to_numeric(tr[feat], errors="coerce").notna()
        sz = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab)
        dy = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
        ft = store[(Y3, feat)]
        same.append(
            {
                "feature": feat,
                "n": f"{ft['n_defined']:,}",
                "feat CV": _f(cv_of(ft)),
                "size same-row": _f(cv_of(sz)),
                "days same-row": _f(cv_of(dy)),
                "Δ size": _f(cv_of(ft) - cv_of(sz) if np.isfinite(cv_of(ft)) and np.isfinite(cv_of(sz)) else float("nan")),
            }
        )

    def _beat(cv, bench) -> float:
        return cv - bench if np.isfinite(cv) and np.isfinite(bench) else float("nan")

    prose = (
        f"Y3 singles (train group-fold): io {_f(io_y3)} / g3 {_f(g3_y3)} / g12 {_f(g12_y3)} "
        f"vs size {_f(size_y3)} (quote 0.617 {'CONFIRM' if size_ok else 'off'}) "
        f"vs days {_f(days_y3)} (night 0.711 {'CONFIRM' if days_ok else 'off'}). "
        f"Signs io={store[(Y3, 'a_io_ratio')]['train_sign']} "
        f"g3={store[(Y3, 'a_growth_3')]['train_sign']} "
        f"g12={store[(Y3, 'a_growth_12')]['train_sign']}. "
        f"−growth3 {_f(neg_y3)}; naive crash −40% {_f(naive_y3)}. "
        f"Y2 io {_f(io_y2)} g3 {_f(g3_y2)} g12 {_f(g12_y2)} vs size {_f(size_y2)}. "
        f"Night Y3 quote stays {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f} (not this cut)."
    )
    print(prose)
    return {
        "rows": rows,
        "same": same,
        "store": store,
        "io_y3": io_y3,
        "g3_y3": g3_y3,
        "g12_y3": g12_y3,
        "neg_y3": neg_y3,
        "naive_y3": naive_y3,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "io_y2": io_y2,
        "g3_y2": g3_y2,
        "g12_y2": g12_y2,
        "size_y2": size_y2,
        "days_y2": days_y2,
        "beat_io": _beat(io_y3, size_y3),
        "beat_g3": _beat(g3_y3, size_y3),
        "beat_g12": _beat(g12_y3, size_y3),
        "days_ok": days_ok,
        "size_ok": size_ok,
        "io_sign": store[(Y3, "a_io_ratio")]["train_sign"],
        "g3_sign": store[(Y3, "a_growth_3")]["train_sign"],
        "g12_sign": store[(Y3, "a_growth_12")]["train_sign"],
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — leftover after size and after days
# ---------------------------------------------------------------------------
def pass5_resid(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    specs = []
    for feat in STEMS:
        x = pd.to_numeric(tr[feat], errors="coerce")
        r_sz, i_sz = ols_resid(x, tr["log_in3"])
        r_dy, i_dy = ols_resid(x, tr["c_n_days_with_tx"])
        r_both, i_both = ols_resid(x, tr["log_in3"], tr["c_n_days_with_tx"])
        specs.append((feat, "size", r_sz, i_sz))
        specs.append((feat, "days", r_dy, i_dy))
        specs.append((feat, "size+days", r_both, i_both))
        tr[f"{feat}_r_size"] = r_sz
        tr[f"{feat}_r_days"] = r_dy
        tr[f"{feat}_r_both"] = r_both

    for y in (Y3, Y2):
        lab = tr[y].notna()
        for feat, spec, resid, info in specs:
            res = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
            key = (y, feat, spec)
            store[key] = {**res, "info": info}
            rows.append(
                {
                    "y": y,
                    "feature": feat,
                    "after": spec,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                    "slope": ",".join(_f(s) for s in info["slope"]) if info["slope"] else "—",
                }
            )

    def _cv(y, feat, spec) -> float:
        return cv_of(store.get((y, feat, spec)))

    size_y3 = float("nan")
    # filled by caller via p4; compute here too
    size_y3 = cv_of(signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna()))

    verdicts = {}
    for feat in STEMS:
        r_sz = _cv(Y3, feat, "size")
        r_dy = _cv(Y3, feat, "days")
        r_bo = _cv(Y3, feat, "size+days")
        live_sz = bool(np.isfinite(r_sz) and np.isfinite(size_y3) and (r_sz - size_y3) >= KEEP_DELTA)
        live_dy = bool(np.isfinite(r_dy) and np.isfinite(size_y3) and (r_dy - size_y3) >= KEEP_DELTA)
        live_bo = bool(np.isfinite(r_bo) and np.isfinite(size_y3) and (r_bo - size_y3) >= KEEP_DELTA)
        died = not (live_sz and live_dy)
        verdicts[feat] = {
            "r_size": r_sz,
            "r_days": r_dy,
            "r_both": r_bo,
            "live_size": live_sz,
            "live_days": live_dy,
            "live_both": live_bo,
            "died": died,
        }
    prose = (
        "Y3 leftover after size / days / size+days: "
        + "; ".join(
            f"{f} {_f(verdicts[f]['r_size'])} / {_f(verdicts[f]['r_days'])} / {_f(verdicts[f]['r_both'])}"
            f"{' DIED' if verdicts[f]['died'] else ' lives'}"
            for f in STEMS
        )
        + f" vs size {_f(size_y3)}. "
        + (
            "Leftover dies — DROP from the 44 / CLOSE as X."
            if all(verdicts[f]["died"] for f in STEMS)
            else "At least one leftover still measured."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "verdicts": verdicts,
        "size_y3": size_y3,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — mean-reversion (sign, quintiles, naive −40%)
# ---------------------------------------------------------------------------
def pass6_reversion(tr: pd.DataFrame, p4: dict) -> dict:
    lab = tr[Y3].notna()
    q_y3 = {c: quintile_table(tr, lab, c, Y3) for c in STEMS}
    q_y2 = {c: quintile_table(tr, tr[Y2].notna(), c, Y2) for c in STEMS}
    q_next = quintile_table(tr, tr["y3_next"].notna() & tr["a_growth_3"].notna(), "a_growth_3", "y3_next")

    g3 = pd.to_numeric(tr["a_growth_3"], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    pos = g3 > 0
    neg = g3 < 0
    rate_pos = float(y3[lab & pos].mean()) if (lab & pos).any() else float("nan")
    rate_neg = float(y3[lab & neg].mean()) if (lab & neg).any() else float("nan")
    rate_all = float(y3[lab & g3.notna()].mean()) if (lab & g3.notna()).any() else float("nan")
    nxt = pd.to_numeric(tr["y3_next"], errors="coerce")
    nxt_pos = float(nxt[g3.notna() & pos & nxt.notna()].mean()) if (g3.notna() & pos & nxt.notna()).any() else float("nan")
    nxt_neg = float(nxt[g3.notna() & neg & nxt.notna()].mean()) if (g3.notna() & neg & nxt.notna()).any() else float("nan")

    plus = p4["g3_y3"]
    minus = p4["neg_y3"]
    naive = p4["naive_y3"]
    # mean-reversion-only: −growth wins and +growth does not beat size
    mr_only = bool(
        np.isfinite(minus)
        and np.isfinite(plus)
        and minus >= plus + 0.01
    )
    # protective if high growth raises Y3 (sign + and Q5 > Q1)
    protective = bool(p4["g3_sign"] == 1 and q_y3["a_growth_3"]["gap"] > 0.02)
    risky = bool(p4["g3_sign"] == -1 or (np.isfinite(q_y3["a_growth_3"]["gap"]) and q_y3["a_growth_3"]["gap"] < -0.02))
    naive_fail = bool(np.isfinite(naive) and naive < 0.55)
    acf1 = median_acf(tr["a_growth_3"], tr["company_id"], 1)
    acf3 = median_acf(tr["a_growth_3"], tr["company_id"], 3)
    acf6 = median_acf(tr["a_growth_3"], tr["company_id"], 6)
    acf1_io = median_acf(tr["a_io_ratio"], tr["company_id"], 1)
    acf3_io = median_acf(tr["a_io_ratio"], tr["company_id"], 3)
    acf1_g12 = median_acf(tr["a_growth_12"], tr["company_id"], 1)
    acf3_g12 = median_acf(tr["a_growth_12"], tr["company_id"], 3)
    acf3_neg = bool(np.isfinite(acf3) and acf3 < -0.20)

    qrows = []
    for c, q in q_y3.items():
        for r in q["rows"]:
            qrows.append(
                {
                    "x": c,
                    "Q": r["q"],
                    "interval": r["interval"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "P(Y3=1)": _pp(r["y_rate"]),
                    "x p50": _f(r["x_median"]),
                }
            )

    prose = (
        f"Y3 `a_growth_3` sign={p4['g3_sign']} "
        f"({'+ protective' if p4['g3_sign'] == 1 else '− risky / already-bounced'}). "
        f"+growth CV {_f(plus)} vs −growth {_f(minus)} vs naive −40% {_f(naive)} "
        f"({'CONFIRM Javier AUC<0.55' if naive_fail else 'naive crash is not a fail'}). "
        f"Y3 rate growth>0 {_pp(rate_pos)} vs growth<0 {_pp(rate_neg)} (all defined {_pp(rate_all)}). "
        f"Next-month Y3 after +growth {_pp(nxt_pos)} vs −growth {_pp(nxt_neg)}. "
        f"Quintile Q1→Q5 {_pp(q_y3['a_growth_3']['q1'])}→{_pp(q_y3['a_growth_3']['q5'])} "
        f"shape={q_y3['a_growth_3']['shape']}. "
        f"acf1/3/6 g3 {_f(acf1)}/{_f(acf3)}/{_f(acf6)} "
        f"({'lag-3 mean-reversion' if acf3_neg else 'no lag-3 flip'}). "
        f"{'MEAN-REVERSION-ONLY' if mr_only else 'not only −growth'}."
    )
    print(prose)
    return {
        "q_y3": q_y3,
        "q_y2": q_y2,
        "q_next": q_next,
        "qrows": qrows,
        "rate_pos": rate_pos,
        "rate_neg": rate_neg,
        "rate_all": rate_all,
        "nxt_pos": nxt_pos,
        "nxt_neg": nxt_neg,
        "plus": plus,
        "minus": minus,
        "naive": naive,
        "mr_only": mr_only,
        "protective": protective,
        "risky": risky,
        "naive_fail": naive_fail,
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "acf1_io": acf1_io,
        "acf3_io": acf3_io,
        "acf1_g12": acf1_g12,
        "acf3_g12": acf3_g12,
        "acf3_neg": acf3_neg,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — io_ratio clip-at-3
# ---------------------------------------------------------------------------
def pass7_clip(tr: pd.DataFrame) -> dict:
    io = pd.to_numeric(tr["a_io_ratio"], errors="coerce")
    un = pd.to_numeric(tr["io_unclip"], errors="coerce")
    defined = io.notna()
    n_def = int(defined.sum())
    n_clip = int((defined & (io >= CLIP_IO - 1e-12)).sum())
    of_def = _pct(n_clip, n_def)
    of_all = _pct(n_clip, len(tr))
    # unclip distribution among defined
    un_d = un[defined]
    p50 = float(un_d.median()) if un_d.notna().any() else float("nan")
    p90 = float(un_d.quantile(0.90)) if un_d.notna().any() else float("nan")
    p99 = float(un_d.quantile(0.99)) if un_d.notna().any() else float("nan")
    share_gt3 = float((un_d >= CLIP_IO).mean()) if un_d.notna().any() else float("nan")
    # is clip a dummy? clip flag vs continuous
    lab = tr[Y3].notna()
    flag = signed_oof_auroc(tr[Y3], tr["io_clip3"], tr["fold"], lab)
    body = lab & defined & (io < CLIP_IO - 1e-12)
    clip = lab & defined & (io >= CLIP_IO - 1e-12)
    body_res = signed_oof_auroc(tr[Y3], tr["a_io_ratio"], tr["fold"], body)
    unclip_res = signed_oof_auroc(tr[Y3], tr["io_unclip"], tr["fold"], lab)
    io_res = signed_oof_auroc(tr[Y3], tr["a_io_ratio"], tr["fold"], lab)
    # Y rates
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    rate_clip = float(y3[clip].mean()) if clip.any() else float("nan")
    rate_body = float(y3[body].mean()) if body.any() else float("nan")
    dummy = bool(
        np.isfinite(cv_of(flag))
        and np.isfinite(cv_of(io_res))
        and abs(cv_of(flag) - cv_of(io_res)) < 0.02
        and of_def >= 0.15
    )
    prose = (
        f"`a_io_ratio` sitting at 3.0: {n_clip:,} / {n_def:,} defined ({_pp(of_def)}; "
        f"{_pp(of_all)} of all train CM). Unclip p50={_f(p50, 2)} p90={_f(p90, 2)} p99={_f(p99, 2)}. "
        f"Y3 clip-flag {_f(cv_of(flag))} vs continuous {_f(cv_of(io_res))} vs unclip {_f(cv_of(unclip_res))}; "
        f"body (io<3) {_f(cv_of(body_res))}. "
        f"P(Y3|clip)={_pp(rate_clip)} vs body {_pp(rate_body)}. "
        f"{'CLIP IS A DUMMY — the 3.0 pile is the skill' if dummy else 'clip is saturation, not the whole dummy'}."
    )
    print(prose)
    return {
        "n_def": n_def,
        "n_clip": n_clip,
        "of_def": of_def,
        "of_all": of_all,
        "p50": p50,
        "p90": p90,
        "p99": p99,
        "share_gt3": share_gt3,
        "flag_cv": cv_of(flag),
        "io_cv": cv_of(io_res),
        "unclip_cv": cv_of(unclip_res),
        "body_cv": cv_of(body_res),
        "rate_clip": rate_clip,
        "rate_body": rate_body,
        "dummy": dummy,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — dark 470 vs 744
# ---------------------------------------------------------------------------
def pass8_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470

    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        io_cov=("a_io_ratio", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
        g3_cov=("a_growth_3", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
        g12_cov=("a_growth_12", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
        io_med=("a_io_ratio", "median"),
        g3_med=("a_growth_3", "median"),
        g12_med=("a_growth_12", "median"),
        trail=("trail_len", "max"),
    )
    ever["ever_erp"] = ever["company_id"].isin(book)
    rows = []
    for name, part in (("ever_erp_744", ever[ever["ever_erp"]]), ("never_erp_470", ever[~ever["ever_erp"]])):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "io cov": _pp(float(part["io_cov"].mean())),
                "g3 cov": _pp(float(part["g3_cov"].mean())),
                "g12 cov": _pp(float(part["g12_cov"].mean())),
                "io p50": _f(float(part["io_med"].median())),
                "g3 p50": _f(float(part["g3_med"].median())),
                "g12 p50": _f(float(part["g12_med"].median())),
                "trail p50": _f(float(part["trail"].median()), 1),
            }
        )
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    cm = []
    store = {}
    for name, part in (("ever_erp", tr2[tr2["ever_erp"]]), ("never_erp", tr2[~tr2["ever_erp"]])):
        for feat in STEMS:
            res = signed_oof_auroc(part[Y3], part[feat], part["fold"], part[Y3].notna())
            store[(name, feat)] = res
            cm.append(
                {
                    "group": name,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "cov": _pp(float(pd.to_numeric(part[feat], errors="coerce").notna().mean())),
                }
            )
    dark_g12 = float(ever.loc[~ever["ever_erp"], "g12_cov"].mean())
    erp_g12 = float(ever.loc[ever["ever_erp"], "g12_cov"].mean())
    same = bool(np.isfinite(dark_g12) and np.isfinite(erp_g12) and abs(dark_g12 - erp_g12) < 0.05)
    hold = load_holdout()
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month companies: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ from join QA'}). "
        f"Mean company growth_12 coverage: invoiced {_pp(erp_g12)} vs dark {_pp(dark_g12)}. "
        f"{'Same book-length hole' if same else 'Dark files growth_12 at a different rate'}. "
        f"Holdout ever-ERP coverage only: {hold_book}/{len(hold)}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "dark_g12": dark_g12,
        "erp_g12": erp_g12,
        "same": same,
        "hold_book": hold_book,
        "hold_n": len(hold),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass9_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    is_ch = tr["company_id"].isin(ids)
    rows = []
    store = {}
    for feat in STEMS:
        for y in (Y3, Y2):
            lab = tr[y].notna()
            for sname, mask in (
                ("all", lab),
                ("drop12", lab & ~is_ch),
                ("chronic12", lab & is_ch),
            ):
                res = signed_oof_auroc(tr[y], tr[feat], tr["fold"], mask)
                store[(y, feat, sname)] = res
                rows.append(
                    {
                        "y": y,
                        "feature": feat,
                        "slice": sname,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    }
                )
    g3_all = cv_of(store[(Y3, "a_growth_3", "all")])
    g3_drop = cv_of(store[(Y3, "a_growth_3", "drop12")])
    flip = bool(np.isfinite(g3_all) and np.isfinite(g3_drop) and abs(g3_all - g3_drop) >= 0.02)
    prose = (
        f"Chronic dark Y2 names n={len(ids)} (expect 12). "
        f"Y3 growth_3 all {_f(g3_all)} vs drop12 {_f(g3_drop)} "
        f"({'FLIP — those names are the skill' if flip else 'no flip'})."
    )
    print(prose)
    print(f"  chronic ids: {ids}")
    return {
        "ids": ids,
        "n_ids": len(ids),
        "rows": rows,
        "g3_all": g3_all,
        "g3_drop": g3_drop,
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name in STEMS:
        col = tr[name]
        icc = icc_anova(col, tr["company_id"])
        acf1 = median_acf(col, tr["company_id"], 1)
        acf3 = median_acf(col, tr["company_id"], 3)
        mu = pd.to_numeric(col, errors="coerce").groupby(tr["company_id"], sort=False).transform("mean")
        dem = pd.to_numeric(col, errors="coerce") - mu
        lab = tr[Y3].notna()
        raw = signed_oof_auroc(tr[Y3], col, tr["fold"], lab)
        dem_res = signed_oof_auroc(tr[Y3], dem, tr["fold"], lab)
        mean_res = signed_oof_auroc(tr[Y3], mu, tr["fold"], lab)
        drop = (
            raw["cv"] - dem_res["cv"]
            if (not raw["low_power"] and not dem_res["low_power"])
            else float("nan")
        )
        rec = {
            "name": name,
            "icc": icc["icc"],
            "k": icc["k"],
            "acf1": acf1,
            "acf3": acf3,
            "raw_cv": cv_of(raw),
            "dem_cv": cv_of(dem_res),
            "mean_cv": cv_of(mean_res),
            "drop": drop,
            "trait": bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT),
            "shock": bool(np.isfinite(icc["icc"]) and icc["icc"] < ICC_SHOCK),
        }
        store[name] = rec
        rows.append(
            {
                "col": name,
                "ICC": _f(icc["icc"]),
                "acf1": _f(acf1),
                "acf3": _f(acf3),
                "Y3 raw": _f(cv_of(raw)),
                "Y3 demean": _f(cv_of(dem_res)),
                "Y3 co-mean": _f(cv_of(mean_res)),
                "drop": _f(drop),
                "call": "TRAIT" if rec["trait"] else ("SHOCK" if rec["shock"] else "between"),
            }
        )
    g3 = store["a_growth_3"]
    g12 = store["a_growth_12"]
    io = store["a_io_ratio"]
    # growth should be LOW_PERSIST / SHOCK if it is a real change
    g3_change = bool(g3["shock"] or (np.isfinite(g3["acf3"]) and g3["acf3"] < 0))
    prose = (
        f"ICC io={_f(io['icc'])} (report 0.70) g3={_f(g3['icc'])} (report 0.70) "
        f"g12={_f(g12['icc'])} (report 0.90 BETWEEN). "
        f"Y3 demean io {_f(io['dem_cv'])} (drop {_f(io['drop'])}); "
        f"g3 {_f(g3['dem_cv'])} (drop {_f(g3['drop'])}); "
        f"g12 {_f(g12['dem_cv'])} (drop {_f(g12['drop'])}). "
        f"growth_3 is {'a month shock / change' if g3['shock'] or g3_change else 'NOT a low-ICC shock'}. "
        f"growth_12 is {'TRAIT (BETWEEN)' if g12['trait'] else ('SHOCK' if g12['shock'] else 'between')}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "g3_change": g3_change,
        "prose": prose,
        **{f"{k}_{kk}": store[k][kk] for k in STEMS for kk in ("icc", "raw_cv", "dem_cv", "trait", "shock")},
    }


# ---------------------------------------------------------------------------
# Pass 11 — Q6 lag1/lag3 short vs long
# ---------------------------------------------------------------------------
def pass11_q6(tr: pd.DataFrame, p1: dict) -> dict:
    rows = []
    store = {}
    slices = [
        ("all", pd.Series(True, index=tr.index)),
        ("short", tr["short_book"] == 1),
        ("long", tr["short_book"] == 0),
        ("so_far<15", tr["so_far"] < 15),
        ("so_far>=15", tr["so_far"] >= 15),
    ]
    cols = []
    for stem in STEMS:
        cols.append(stem)
        cols.append(f"{stem}_lag1")
        cols.append(f"{stem}_lag3")
    for y in (Y3, Y2):
        for sname, smask in slices:
            for col in cols:
                if col not in tr.columns:
                    continue
                mask = tr[y].notna() & smask
                res = signed_oof_auroc(tr[y], tr[col], tr["fold"], mask)
                store[(y, sname, col)] = res
                rows.append(
                    {
                        "y": y,
                        "slice": sname,
                        "col": col,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                    }
                )

    def _cv(y, sl, col) -> float:
        return cv_of(store.get((y, sl, col)))

    now = {c: _cv(Y3, "all", c) for c in STEMS}
    lag1 = {c: _cv(Y3, "all", f"{c}_lag1") for c in STEMS}
    lag3 = {c: _cv(Y3, "all", f"{c}_lag3") for c in STEMS}
    g12_short = _cv(Y3, "short", "a_growth_12")
    g12_so = _cv(Y3, "so_far<15", "a_growth_12")
    g12_short_n = store[(Y3, "short", "a_growth_12")]["n_defined"]
    g12_so_n = store[(Y3, "so_far<15", "a_growth_12")]["n_defined"]
    empty_q6 = bool(p1["empty_short"] or p1["empty_early"] or g12_short_n < MIN_POS or g12_so_n < MIN_POS)

    def _keep(col: str) -> bool:
        n = now[col]
        l = lag1[col]
        return bool(
            np.isfinite(n) and np.isfinite(l) and n >= 0.60 and (n - l) <= 0.03 and l >= 0.60
        )

    keep = {c: _keep(c) for c in STEMS}
    if empty_q6:
        q6 = "CLOSE as Q6"
        why = (
            f"growth_12 is empty on short books (n_def short={g12_short_n}, "
            f"so_far<15={g12_so_n}). Hidden-72 late-arrival hole. CLOSE as Q6."
        )
    elif not any(keep.values()):
        q6 = "CLOSE"
        why = (
            f"Y3 now io {_f(now['a_io_ratio'])} g3 {_f(now['a_growth_3'])} "
            f"g12 {_f(now['a_growth_12'])}; lag1 "
            f"{_f(lag1['a_io_ratio'])}/{_f(lag1['a_growth_3'])}/{_f(lag1['a_growth_12'])}. "
            "CLOSE as Q6 — no contemporaneous skill ≥0.60 that lag1 holds."
        )
    else:
        q6 = "KEEP"
        why = (
            f"lag1 holds contemporaneous skill for "
            f"{[c for c, k in keep.items() if k]}. Still not on the 15-col card."
        )
    print(why)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "g12_short": g12_short,
        "g12_so": g12_so,
        "g12_short_n": g12_short_n,
        "g12_so_n": g12_so_n,
        "empty_q6": empty_q6,
        "keep": keep,
        "q6": q6,
        "prose": why,
    }


# ---------------------------------------------------------------------------
# Pass 12 — Y4 crash-month overlap (descriptive only)
# ---------------------------------------------------------------------------
def pass12_y4(tr: pd.DataFrame) -> dict:
    y4 = pd.to_numeric(tr[Y4], errors="coerce")
    lab = y4.notna()
    pos = lab & (y4 == 1)
    rows = []
    for feat in STEMS:
        x = pd.to_numeric(tr[feat], errors="coerce")
        rows.append(
            {
                "feature": feat,
                "Y4 n": int(lab.sum()),
                "Y4 pos": int(pos.sum()),
                "cov on Y4": _pp(float(x[lab].notna().mean()) if lab.any() else float("nan")),
                "p50 Y4=1": _f(float(x[pos].median()) if pos.any() else float("nan")),
                "p50 Y4=0": _f(float(x[lab & ~pos].median()) if (lab & ~pos).any() else float("nan")),
                "g3<=-0.4 on Y4=1": (
                    _pp(float((x[pos] <= NAIVE_CRASH).mean()))
                    if feat == "a_growth_3" and pos.any()
                    else "—"
                ),
            }
        )
    g3 = pd.to_numeric(tr["a_growth_3"], errors="coerce")
    share_crash = float((g3[pos] <= NAIVE_CRASH).mean()) if pos.any() else float("nan")
    # do NOT compute AUROC vs Y4 as an X claim — only a descriptive note
    rho = spearman(g3[lab], y4[lab])
    prose = (
        f"Y4 labeled {int(lab.sum()):,} / pos {int(pos.sum()):,}. "
        f"growth_3 p50 on Y4=1 {_f(float(g3[pos].median()) if pos.any() else float('nan'))} "
        f"vs Y4=0 {_f(float(g3[lab & ~pos].median()) if (lab & ~pos).any() else float('nan'))}. "
        f"Share growth_3≤−0.4 among Y4 pos {_pp(share_crash)}. "
        f"Spearman growth_3↔Y4 {_f(rho)} (descriptive). "
        "Do not score growth as Y4 X; do not merge with Y4."
    )
    print(prose)
    return {
        "rows": rows,
        "n_lab": int(lab.sum()),
        "n_pos": int(pos.sum()),
        "share_crash": share_crash,
        "rho": rho,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 13 — growth clip ±1 saturation
# ---------------------------------------------------------------------------
def pass13_gclip(tr: pd.DataFrame) -> dict:
    rows = []
    for feat, lo, hi in (
        ("a_growth_3", "g3_clip_lo", "g3_clip_hi"),
        ("a_growth_12", "g12_clip_lo", "g12_clip_hi"),
    ):
        x = pd.to_numeric(tr[feat], errors="coerce")
        d = x.notna()
        n = int(d.sum())
        n_lo = int((d & (tr[lo] == 1)).sum())
        n_hi = int((d & (tr[hi] == 1)).sum())
        rows.append(
            {
                "feature": feat,
                "n_def": f"{n:,}",
                "clip −1": f"{n_lo:,} ({_pp(_pct(n_lo, n))})",
                "clip +1": f"{n_hi:,} ({_pp(_pct(n_hi, n))})",
                "either": _pp(_pct(n_lo + n_hi, n)),
            }
        )
    prose = (
        f"growth_3 clip pile −1 {_pp(_pct(int((tr['g3_clip_lo']==1).sum()), int(tr['a_growth_3'].notna().sum())))} "
        f"+1 {_pp(_pct(int((tr['g3_clip_hi']==1).sum()), int(tr['a_growth_3'].notna().sum())))}."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 14 — size terciles leftover
# ---------------------------------------------------------------------------
def pass14_terciles(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    terc = pd.qcut(
        pd.to_numeric(last["log_in3"], errors="coerce"),
        3,
        labels=["T1", "T2", "T3"],
        duplicates="drop",
    ).rename("size_t")
    m = tr.merge(terc.reset_index(), on="company_id", how="left")
    rows = []
    for feat in STEMS:
        for t in ("T1", "T2", "T3"):
            mask = m[Y3].notna() & (m["size_t"].astype(str) == t)
            res = signed_oof_auroc(m[Y3], m[feat], m["fold"], mask)
            sz = signed_oof_auroc(m[Y3], m["log_in3"], m["fold"], mask)
            rows.append(
                {
                    "feature": feat,
                    "tercile": t,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "feat CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "size CV": "LOW_POWER" if sz["low_power"] else _f(sz["cv"]),
                }
            )
    prose = "Y3 inside company size terciles (last-month log1p(a_in3))."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 15 — holdout coverage (no AUROC)
# ---------------------------------------------------------------------------
def pass15_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col in STEMS + ("a_net_margin", "a_in3"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "p50": _f(float(x.median()) if x.notna().any() else float("nan")),
            }
        )
    g12 = pd.to_numeric(ho["a_growth_12"], errors="coerce")
    short = ho["short_book"] == 1
    late = ho["late_arrival"] == 1
    n_short = int(ho.loc[short, "company_id"].nunique())
    n_late = int(ho.loc[late, "company_id"].nunique())
    g12_short = float(g12[short].notna().mean()) if short.any() else float("nan")
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM. "
        f"growth_12 {_pp(float(g12.notna().mean()))}. "
        f"Short-trail companies {n_short}; late-arrival {n_late}; "
        f"growth_12 on short {_pp(g12_short)}. No AUROC claim."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cm": int(len(ho)),
        "n_co": int(ho["company_id"].nunique()),
        "n_short": n_short,
        "n_late": n_late,
        "g12_short": g12_short,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 16 — calendar of growth / io
# ---------------------------------------------------------------------------
def pass16_cal(tr: pd.DataFrame) -> dict:
    rows = []
    for m in range(1, 13):
        sl = tr[tr["cal_month"] == m]
        def _med(col: str) -> float:
            s = pd.to_numeric(sl[col], errors="coerce").dropna()
            return float(s.median()) if len(s) else float("nan")

        rows.append(
            {
                "month": pd.Timestamp(2000, m, 1).strftime("%b"),
                "n": f"{len(sl):,}",
                "io p50": _f(_med("a_io_ratio")),
                "g3 p50": _f(_med("a_growth_3")),
                "g12 p50": _f(_med("a_growth_12")),
                "g12 cov": _pp(float(pd.to_numeric(sl["a_growth_12"], errors="coerce").notna().mean())),
            }
        )
    prose = "Calendar medians — growth_12 coverage is a late-panel hole, not a month dummy."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 17 — residual ρ vs the control (fake leftover?)
# ---------------------------------------------------------------------------
def pass17_fake(tr: pd.DataFrame) -> dict:
    rows = []
    for feat in STEMS:
        x = pd.to_numeric(tr[feat], errors="coerce")
        r_dy, _ = ols_resid(x, tr["c_n_days_with_tx"])
        r_sz, _ = ols_resid(x, tr["log_in3"])
        rows.append(
            {
                "feature": feat,
                "ρ(resid_days, days)": _f(spearman(r_dy, tr["c_n_days_with_tx"])),
                "ρ(resid_size, size)": _f(spearman(r_sz, tr["log_in3"])),
                "ρ(resid_days, size)": _f(spearman(r_dy, tr["log_in3"])),
            }
        )
    prose = "OLS leftover should be ≈0 vs its control; a large leftover ρ is a numerical leak."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 18 — stressed-only defined vs all Y3 (Y3 is already stressed)
# ---------------------------------------------------------------------------
def pass18_defined(tr: pd.DataFrame) -> dict:
    """Fair same-row size/days already in p4; add growth_12-only long-book body."""
    rows = []
    long = tr["so_far"] >= 15
    for feat in STEMS:
        mask = tr[Y3].notna() & long & pd.to_numeric(tr[feat], errors="coerce").notna()
        res = signed_oof_auroc(tr[Y3], tr[feat], tr["fold"], mask)
        sz = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], mask)
        dy = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], mask)
        rows.append(
            {
                "feature": feat,
                "slice": "so_far>=15",
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "feat": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "size": "LOW_POWER" if sz["low_power"] else _f(sz["cv"]),
                "days": "LOW_POWER" if dy["low_power"] else _f(dy["cv"]),
            }
        )
    prose = "Long-book (so_far≥15) same-row singles — growth_12's native support."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 19 — U-shape / io body leftover / T1 pocket
# ---------------------------------------------------------------------------
def pass19_ushape(tr: pd.DataFrame, p6: dict) -> dict:
    """Q1 and Q5 both recover — two-tail, not monotone mean-reversion."""
    q = p6["q_y3"]["a_growth_3"]
    rates = [r["y_rate"] for r in q["rows"]]
    u = False
    if len(rates) == 5:
        mid = min(rates[1], rates[2], rates[3])
        u = bool(rates[0] >= mid + 0.03 and rates[4] >= mid + 0.03)
    io_q = p6["q_y3"]["a_io_ratio"]
    io_rates = [r["y_rate"] for r in io_q["rows"]]
    io_u = False
    if len(io_rates) == 5:
        mid = min(io_rates[1], io_rates[2], io_rates[3])
        io_u = bool(io_rates[0] >= mid + 0.03 and io_rates[4] >= mid + 0.02)

    io = pd.to_numeric(tr["a_io_ratio"], errors="coerce")
    body = io.notna() & (io < CLIP_IO - 1e-12)
    r_body_d, _ = ols_resid(io.where(body), tr["c_n_days_with_tx"])
    r_body_s, _ = ols_resid(io.where(body), tr["log_in3"])
    lab = tr[Y3].notna()
    body_raw = signed_oof_auroc(tr[Y3], tr["a_io_ratio"], tr["fold"], lab & body)
    body_d = signed_oof_auroc(tr[Y3], r_body_d, tr["fold"], lab)
    body_s = signed_oof_auroc(tr[Y3], r_body_s, tr["fold"], lab)
    days_body = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & body)
    size_body = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab & body)

    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    terc = pd.qcut(
        pd.to_numeric(last["log_in3"], errors="coerce"),
        3,
        labels=["T1", "T2", "T3"],
        duplicates="drop",
    ).rename("size_t")
    m = tr.merge(terc.reset_index(), on="company_id", how="left")
    t1 = m["size_t"].astype(str) == "T1"
    t1_rows = []
    for feat in STEMS:
        mask = m[Y3].notna() & t1
        raw = signed_oof_auroc(m[Y3], m[feat], m["fold"], mask)
        r_d, _ = ols_resid(pd.to_numeric(m[feat], errors="coerce"), m["c_n_days_with_tx"])
        leftover = signed_oof_auroc(m[Y3], r_d, m["fold"], mask)
        dy = signed_oof_auroc(m[Y3], m["c_n_days_with_tx"], m["fold"], mask)
        sz = signed_oof_auroc(m[Y3], m["log_in3"], m["fold"], mask)
        t1_rows.append(
            {
                "feature": feat,
                "raw": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "leftover days": "LOW_POWER" if leftover["low_power"] else _f(leftover["cv"]),
                "days T1": "LOW_POWER" if dy["low_power"] else _f(dy["cv"]),
                "size T1": "LOW_POWER" if sz["low_power"] else _f(sz["cv"]),
                "n_pos": raw["n_pos"],
            }
        )
    io_t1 = next(r for r in t1_rows if r["feature"] == "a_io_ratio")
    prose = (
        f"growth_3 quintiles U-shape={'YES' if u else 'NO'} "
        f"(Q1 {_pp(rates[0] if rates else float('nan'))} mid-low "
        f"{_pp(min(rates[1:4]) if len(rates)==5 else float('nan'))} Q5 "
        f"{_pp(rates[-1] if rates else float('nan'))}). "
        f"io U-shape={'YES' if io_u else 'NO'}. "
        f"io body (io<3) Y3 {_f(cv_of(body_raw))} leftover-days {_f(cv_of(body_d))} "
        f"vs days-on-body {_f(cv_of(days_body))} / size-on-body {_f(cv_of(size_body))}. "
        f"T1 io raw {io_t1['raw']} leftover-days {io_t1['leftover days']} "
        f"vs days {io_t1['days T1']}. Two-tail is not KEEP; leftover after days dies."
    )
    print(prose)
    return {
        "u": u,
        "io_u": io_u,
        "body_raw": cv_of(body_raw),
        "body_days": cv_of(body_d),
        "body_size": cv_of(body_s),
        "days_body": cv_of(days_body),
        "size_body": cv_of(size_body),
        "t1_rows": t1_rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — Javier month-on-month inflow −40%
# ---------------------------------------------------------------------------
def pass20_mom(tr: pd.DataFrame) -> dict:
    """Naive inflow −40% on monthly a_op_in (Javier), not the trailing-3m clip."""
    out = tr.sort_values(["company_id", "period"])
    opin = pd.to_numeric(out["a_op_in"], errors="coerce")
    lag = opin.groupby(out["company_id"], sort=False).shift(1)
    mom = np.where(lag > 0, (opin / lag - 1.0).clip(-1.0, 1.0), np.nan)
    crash = np.where(np.isfinite(mom), (mom <= NAIVE_CRASH).astype(float), np.nan)
    out = out.assign(mom_in=mom, mom_crash40=crash)
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = out[y].notna()
        for name, col in (("mom_in", out["mom_in"]), ("mom_crash40", out["mom_crash40"])):
            res = signed_oof_auroc(out[y], col, out["fold"], lab)
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
    y3 = cv_of(store[(Y3, "mom_crash40")])
    y3_m = cv_of(store[(Y3, "mom_in")])
    y2 = cv_of(store[(Y2, "mom_crash40")])
    javier = bool(np.isfinite(y3) and y3 < 0.55)
    prose = (
        f"MoM a_op_in −40% Y3 {_f(y3)} (sign {store[(Y3, 'mom_crash40')]['train_sign']}); "
        f"continuous MoM {_f(y3_m)}. Y2 crash {_f(y2)}. "
        f"{'CONFIRM Javier naive inflow −40% AUC < 0.55' if javier else 'MoM −40% is not a <0.55 fail on Y3'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_crash": y3,
        "y3_mom": y3_m,
        "y2_crash": y2,
        "javier": javier,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 21 — mechanical acf3 (overlapping 3m windows)
# ---------------------------------------------------------------------------
def pass21_mech(tr: pd.DataFrame) -> dict:
    """acf3 of growth_3 is in3[t]/in3[t-3] vs in3[t+3]/in3[t] — overlap, not a cycle."""
    acf3 = median_acf(tr["a_growth_3"], tr["company_id"], 3)
    # non-overlapping subsample: so_far % 3 == 0, then lag-1 of that series = 3 calendar months
    sl = tr.loc[tr["so_far"] % 3 == 0, ["company_id", "a_growth_3"]].copy()
    acf_non = median_acf(sl["a_growth_3"], sl["company_id"], 1)
    acf1_in3 = median_acf(tr["a_in3"], tr["company_id"], 1)
    acf1_op = median_acf(tr["a_op_in"], tr["company_id"], 1)
    acf3_in3 = median_acf(tr["a_in3"], tr["company_id"], 3)
    mechanical = bool(np.isfinite(acf3) and acf3 < -0.20 and np.isfinite(acf1_in3) and acf1_in3 > 0.40)
    prose = (
        f"growth_3 acf3={_f(acf3)}; non-overlapping (so_far%3==0) acf1={_f(acf_non)}. "
        f"a_in3 acf1/acf3 {_f(acf1_in3)}/{_f(acf3_in3)}; a_op_in acf1 {_f(acf1_op)}. "
        f"{'MECHANICAL — persistent in3 makes overlapping growth_3 flip at lag 3' if mechanical else 'not just window overlap'}."
    )
    print(prose)
    return {
        "acf3": acf3,
        "acf_non": acf_non,
        "acf1_in3": acf1_in3,
        "acf3_in3": acf3_in3,
        "acf1_op": acf1_op,
        "mechanical": mechanical,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 22 — company-mean leftover (style, not a shock)
# ---------------------------------------------------------------------------
def pass22_comean(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cid = tr["company_id"]
    mu_days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce").groupby(cid, sort=False).transform("mean")
    mu_sz = pd.to_numeric(tr["log_in3"], errors="coerce").groupby(cid, sort=False).transform("mean")
    lab = tr[Y3].notna()
    for feat in STEMS:
        x = pd.to_numeric(tr[feat], errors="coerce")
        mu = x.groupby(cid, sort=False).transform("mean")
        raw = signed_oof_auroc(tr[Y3], mu, tr["fold"], lab)
        r_d, _ = ols_resid(mu, mu_days)
        r_s, _ = ols_resid(mu, mu_sz)
        r_b, _ = ols_resid(mu, mu_sz, mu_days)
        dres = signed_oof_auroc(tr[Y3], r_d, tr["fold"], lab)
        sres = signed_oof_auroc(tr[Y3], r_s, tr["fold"], lab)
        bres = signed_oof_auroc(tr[Y3], r_b, tr["fold"], lab)
        store[feat] = {
            "raw": cv_of(raw),
            "days": cv_of(dres),
            "size": cv_of(sres),
            "both": cv_of(bres),
        }
        rows.append(
            {
                "feature": feat,
                "co-mean": _f(cv_of(raw)),
                "after co-days": _f(cv_of(dres)),
                "after co-size": _f(cv_of(sres)),
                "after both": _f(cv_of(bres)),
            }
        )
    g3 = store["a_growth_3"]
    style = bool(np.isfinite(g3["raw"]) and g3["raw"] >= SIZE_QUOTE + KEEP_DELTA)
    died = bool(not np.isfinite(g3["both"]) or g3["both"] < SIZE_QUOTE + KEEP_DELTA)
    prose = (
        f"Company-mean Y3: io {_f(store['a_io_ratio']['raw'])} g3 {_f(g3['raw'])} "
        f"g12 {_f(store['a_growth_12']['raw'])}. "
        f"g3 after co-mean days/size/both {_f(g3['days'])}/{_f(g3['size'])}/{_f(g3['both'])}. "
        f"{'Company-mean g3 beats size — STYLE, not a month shock. Do not KEEP.' if style else 'Company-mean does not beat size.'} "
        f"{'Style leftover dies after co-mean size+days.' if died else 'Style leftover still lives — still not a shock.'}"
    )
    print(prose)
    return {"rows": rows, "store": store, "style": style, "died": died, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 23 — Y3 label hole (who lacks growth_3)
# ---------------------------------------------------------------------------
def pass23_hole(tr: pd.DataFrame) -> dict:
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y3.notna()
    pos = lab & (y3 == 1)
    rows = []
    for feat in STEMS:
        x = pd.to_numeric(tr[feat], errors="coerce")
        rows.append(
            {
                "feature": feat,
                "Y3 labeled": int(lab.sum()),
                "Y3 pos": int(pos.sum()),
                "pos with X": int((pos & x.notna()).sum()),
                "pos missing X": int((pos & x.isna()).sum()),
                "miss share": _pp(_pct(int((pos & x.isna()).sum()), int(pos.sum()))),
                "miss so_far p50": _f(
                    float(tr.loc[pos & x.isna(), "so_far"].median())
                    if (pos & x.isna()).any()
                    else float("nan"),
                    1,
                ),
            }
        )
    g3 = pd.to_numeric(tr["a_growth_3"], errors="coerce")
    n_miss = int((pos & g3.isna()).sum())
    prose = (
        f"Y3 positives missing growth_3: {n_miss} / {int(pos.sum())} "
        f"({_pp(_pct(n_miss, int(pos.sum())))}). Short-book / zero in3_{{t-3}} hole, not a sign flip."
    )
    print(prose)
    return {"rows": rows, "n_miss": n_miss, "n_pos": int(pos.sum()), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 24 — rank leftover (nonlinear SIZE leak)
# ---------------------------------------------------------------------------
def pass24_rank(tr: pd.DataFrame) -> dict:
    """ρ(io resid, size)=−0.643 — OLS leftover is a nonlinear size leak."""
    rows = []
    lab = tr[Y3].notna()
    size_cv = cv_of(signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab))
    for feat in STEMS:
        x = pd.to_numeric(tr[feat], errors="coerce")
        sz = pd.to_numeric(tr["log_in3"], errors="coerce")
        rx = x.rank(method="average")
        rz = sz.rank(method="average")
        r_rk, _ = ols_resid(rx, rz)
        r_raw, _ = ols_resid(x, sz)
        rk = signed_oof_auroc(tr[Y3], r_rk, tr["fold"], lab)
        raw = signed_oof_auroc(tr[Y3], r_raw, tr["fold"], lab)
        rho_fake = spearman(r_raw, sz)
        rows.append(
            {
                "feature": feat,
                "OLS leftover": _f(cv_of(raw)),
                "rank leftover": _f(cv_of(rk)),
                "ρ(OLS,size)": _f(rho_fake),
                "size": _f(size_cv),
            }
        )
    io = next(r for r in rows if r["feature"] == "a_io_ratio")
    leak = bool(abs(float(io["ρ(OLS,size)"]) if io["ρ(OLS,size)"] != "—" else 0) >= 0.40)
    prose = (
        f"io OLS leftover {io['OLS leftover']} ρ(resid,size)={io['ρ(OLS,size)']}; "
        f"rank leftover {io['rank leftover']} vs size {io['size']}. "
        f"{'Nonlinear SIZE leak — the 0.604 is not leftover Q3' if leak else 'OLS residual is orthogonal enough'}."
    )
    print(prose)
    return {"rows": rows, "leak": leak, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 25 — dark g12 hole is trail length
# ---------------------------------------------------------------------------
def pass25_darktrail(tr: pd.DataFrame, p8: dict) -> dict:
    # reuse last-month trail already in p8 rows
    dark_tr = float("nan")
    erp_tr = float("nan")
    for r in p8["rows"]:
        if r["group"] == "never_erp_470":
            dark_tr = r["trail p50"]
        if r["group"] == "ever_erp_744":
            erp_tr = r["trail p50"]
    # company-level: g12_cov vs trail
    ever = tr.groupby("company_id", as_index=False).agg(
        g12_cov=("a_growth_12", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
        trail=("trail_len", "max"),
        late=("late_arrival", "max"),
    )
    rho = spearman(ever["g12_cov"], ever["trail"])
    short_share_dark = float("nan")
    prose = (
        f"Dark trail p50 {dark_tr} vs invoiced {erp_tr}. "
        f"Company g12 coverage vs trail length ρ={_f(rho)}. "
        f"{'Dark g12 hole is shorter books, not a dark-specific growth object' if np.isfinite(rho) and rho >= 0.50 else 'trail does not fully explain the dark hole'}."
    )
    print(prose)
    return {
        "dark_tr": dark_tr,
        "erp_tr": erp_tr,
        "rho": rho,
        "short_share_dark": short_share_dark,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 26 — holdout late-arrival lock
# ---------------------------------------------------------------------------
def pass26_hold_late(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    n_co = int(ho["company_id"].nunique())
    late = ho.groupby("company_id")["late_arrival"].max()
    short = ho.groupby("company_id")["short_book"].max()
    n_late = int(late.sum())
    n_short = int(short.sum())
    g12 = pd.to_numeric(ho["a_growth_12"], errors="coerce")
    g12_late = float(g12[ho["late_arrival"] == 1].notna().mean()) if (ho["late_arrival"] == 1).any() else float("nan")
    g12_ontime = float(g12[ho["late_arrival"] == 0].notna().mean()) if (ho["late_arrival"] == 0).any() else float("nan")
    lock = bool(n_late >= 60 and n_co == 72)
    prose = (
        f"Holdout companies late-arrival {n_late}/{n_co}; short-trail {n_short}/{n_co}. "
        f"growth_12 cov late {_pp(g12_late)} vs on-time {_pp(g12_ontime)}. "
        f"{'LOCK — hidden 72 is almost all late-arrival; growth_12 CLOSE as Q6' if lock else 'holdout late-arrival is not 69/72'}."
    )
    print(prose)
    return {
        "n_co": n_co,
        "n_late": n_late,
        "n_short": n_short,
        "g12_late": g12_late,
        "g12_ontime": g12_ontime,
        "lock": lock,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 27 — clip pile vs Q1 coverage hole (who recovers)
# ---------------------------------------------------------------------------
def pass27_tails(tr: pd.DataFrame) -> dict:
    io = pd.to_numeric(tr["a_io_ratio"], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y3.notna() & io.notna()
    clip = lab & (io >= CLIP_IO - 1e-12)
    q1 = lab & (io <= 0.418)
    mid = lab & (io > 0.418) & (io < 1.224)
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    rows = [
        {
            "slice": "io Q1 (low coverage)",
            "n": int(q1.sum()),
            "n_pos": int((q1 & (y3 == 1)).sum()),
            "P(Y3)": _pp(float(y3[q1].mean()) if q1.any() else float("nan")),
            "days p50": _f(float(days[q1].median()) if q1.any() else float("nan"), 1),
        },
        {
            "slice": "io mid Q2–Q4",
            "n": int(mid.sum()),
            "n_pos": int((mid & (y3 == 1)).sum()),
            "P(Y3)": _pp(float(y3[mid].mean()) if mid.any() else float("nan")),
            "days p50": _f(float(days[mid].median()) if mid.any() else float("nan"), 1),
        },
        {
            "slice": "io clip=3",
            "n": int(clip.sum()),
            "n_pos": int((clip & (y3 == 1)).sum()),
            "P(Y3)": _pp(float(y3[clip].mean()) if clip.any() else float("nan")),
            "days p50": _f(float(days[clip].median()) if clip.any() else float("nan"), 1),
        },
    ]
    quiet = bool(
        q1.any()
        and clip.any()
        and float(days[q1].median()) < float(days[mid].median()) - 1
    )
    prose = (
        f"Y3 on io Q1 {_pp(float(y3[q1].mean()) if q1.any() else float('nan'))} "
        f"(days p50 {_f(float(days[q1].median()) if q1.any() else float('nan'), 1)}) "
        f"vs clip=3 {_pp(float(y3[clip].mean()) if clip.any() else float('nan'))} "
        f"(days p50 {_f(float(days[clip].median()) if clip.any() else float('nan'), 1)}). "
        f"{'Q1 is the quiet-days pile already on the card' if quiet else 'Q1 is not quieter than mid'}."
    )
    print(prose)
    return {"rows": rows, "quiet": quiet, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 28 — MoM leftover + non-overlapping growth_3
# ---------------------------------------------------------------------------
def pass28_alts(tr: pd.DataFrame) -> dict:
    out = tr.sort_values(["company_id", "period"])
    opin = pd.to_numeric(out["a_op_in"], errors="coerce")
    lag = opin.groupby(out["company_id"], sort=False).shift(1)
    mom = pd.Series(np.where(lag > 0, (opin / lag - 1.0).clip(-1.0, 1.0), np.nan), index=out.index)
    r_d, _ = ols_resid(mom, out["c_n_days_with_tx"])
    r_s, _ = ols_resid(mom, out["log_in3"])
    r_b, _ = ols_resid(mom, out["log_in3"], out["c_n_days_with_tx"])
    lab = out[Y3].notna()
    raw = signed_oof_auroc(out[Y3], mom, out["fold"], lab)
    dres = signed_oof_auroc(out[Y3], r_d, out["fold"], lab)
    sres = signed_oof_auroc(out[Y3], r_s, out["fold"], lab)
    bres = signed_oof_auroc(out[Y3], r_b, out["fold"], lab)
    size = signed_oof_auroc(out[Y3], out["log_in3"], out["fold"], lab)
    days = signed_oof_auroc(out[Y3], out["c_n_days_with_tx"], out["fold"], lab)
    non = out["so_far"] % 3 == 0
    g3_non = signed_oof_auroc(out[Y3], out["a_growth_3"], out["fold"], lab & non)
    sz_non = signed_oof_auroc(out[Y3], out["log_in3"], out["fold"], lab & non)
    dy_non = signed_oof_auroc(out[Y3], out["c_n_days_with_tx"], out["fold"], lab & non)
    rows = [
        {
            "feature": "mom_in",
            "raw": _f(cv_of(raw)),
            "after size": _f(cv_of(sres)),
            "after days": _f(cv_of(dres)),
            "after both": _f(cv_of(bres)),
            "size": _f(cv_of(size)),
            "days": _f(cv_of(days)),
        },
        {
            "feature": "g3 so_far%3==0",
            "raw": _f(cv_of(g3_non)),
            "after size": "—",
            "after days": "—",
            "after both": "—",
            "size": _f(cv_of(sz_non)),
            "days": _f(cv_of(dy_non)),
        },
    ]
    mom_dies = bool(not np.isfinite(cv_of(bres)) or cv_of(bres) < cv_of(size) + KEEP_DELTA)
    prose = (
        f"MoM inflow Y3 {_f(cv_of(raw))} leftover size/days/both "
        f"{_f(cv_of(sres))}/{_f(cv_of(dres))}/{_f(cv_of(bres))} vs size {_f(cv_of(size))} / days {_f(cv_of(days))}. "
        f"Non-overlapping growth_3 {_f(cv_of(g3_non))} vs size {_f(cv_of(sz_non))}. "
        f"{'MoM leftover dies — still DROP' if mom_dies else 'MoM leftover lives — revisit'}."
    )
    print(prose)
    return {
        "rows": rows,
        "mom_raw": cv_of(raw),
        "mom_both": cv_of(bres),
        "g3_non": cv_of(g3_non),
        "mom_dies": mom_dies,
        "prose": prose,
    }


def make_png(tr: pd.DataFrame, p1: dict, p6: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    q = p6["q_y3"]["a_growth_3"]
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.4), gridspec_kw={"height_ratios": [1.15, 1.05]})
    ax = axes[0]
    if q["rows"]:
        xs = [r["q"] for r in q["rows"]]
        rates = [100.0 * r["y_rate"] for r in q["rows"]]
        ax.bar(xs, rates, color="#1f4e79")
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Q{i}" for i in xs])
        ax.set_ylabel("% Y3 recover")
        ax.set_title("Train Y3 rate by a_growth_3 quintile (already-stressed recover)")
        ax.set_ylim(0, max(rates) * 1.25 if rates else 20)
        for i, r in zip(xs, q["rows"]):
            ax.text(i, 100.0 * r["y_rate"] + 0.3, f"{100.0 * r['y_rate']:.1f}%", ha="center", fontsize=8)
    ax2 = axes[1]
    so = (
        tr.groupby("so_far", as_index=False)
        .agg(
            n=("company_id", "size"),
            io=("a_io_ratio", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
            g3=("a_growth_3", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
            g12=("a_growth_12", lambda s: pd.to_numeric(s, errors="coerce").notna().mean()),
        )
        .sort_values("so_far")
    )
    ax2.plot(so["so_far"], 100.0 * so["io"], label="a_io_ratio", color="#1f4e79")
    ax2.plot(so["so_far"], 100.0 * so["g3"], label="a_growth_3", color="#9e6b4a")
    ax2.plot(so["so_far"], 100.0 * so["g12"], label="a_growth_12", color="#c1121f")
    ax2.axvline(15, color="#c1121f", ls="--", lw=0.8, alpha=0.8)
    ax2.set_xlabel("months so far on the company grid")
    ax2.set_ylabel("% defined")
    ax2.set_title("Coverage hole — growth_12 needs month 15+")
    ax2.set_ylim(0, 105)
    ax2.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def _obj_decision(feat: str, p3, p4, p5, p6, p10, p11) -> dict:
    """KEEP on 44 / DROP-from-44 CLOSE / PARK Y. Never on the 15-col card."""
    raw = {"a_io_ratio": p4["io_y3"], "a_growth_3": p4["g3_y3"], "a_growth_12": p4["g12_y3"]}[feat]
    beat = {"a_io_ratio": p4["beat_io"], "a_growth_3": p4["beat_g3"], "a_growth_12": p4["beat_g12"]}[feat]
    v = p5["verdicts"][feat]
    size_flag = {
        "a_io_ratio": p3["io_size"],
        "a_growth_3": p3["g3_size"],
        "a_growth_12": p3["g12_size"],
    }[feat]
    twin_days = bool(
        (feat == "a_io_ratio" and p3["io_twin_days"])
        or (feat == "a_growth_3" and p3["g3_twin_days"])
    )
    icc = p10["store"][feat]
    mr = bool(feat == "a_growth_3" and p6["mr_only"])
    leftover_ok = bool(v["live_size"] and v["live_days"] and v["live_both"])
    shock = bool(icc["shock"])
    beat_ok = bool(np.isfinite(beat) and beat >= KEEP_DELTA)
    q6_empty = bool(feat == "a_growth_12" and p11["empty_q6"])

    if q6_empty:
        x_dec = "CLOSE as Q6"
        why = (
            "growth_12 is empty on short books (needs month 15+). "
            "CLOSE as Q6 (hidden 72 late-arrival)."
        )
    elif size_flag or twin_days or v["died"] or mr or not leftover_ok or not beat_ok or not shock:
        x_dec = "DROP from the 44"
        bits = []
        if size_flag:
            bits.append("SIZE")
        if twin_days:
            bits.append("days twin")
        if v["died"] or not leftover_ok:
            bits.append(
                f"leftover dies (size {_f(v['r_size'])} / days {_f(v['r_days'])} "
                f"/ both {_f(v['r_both'])} vs size {_f(p4['size_y3'])})"
            )
        if mr:
            bits.append("mean-reversion-only (−growth wins)")
        if not beat_ok:
            bits.append(f"raw CV {_f(raw)} does not beat size {_f(p4['size_y3'])} by ≥0.02")
        if not shock and not q6_empty:
            bits.append(f"not a month shock (ICC {_f(icc['icc'])})")
        why = "; ".join(bits) + ". CLOSE as X."
    else:
        x_dec = "KEEP"
        why = (
            f"leftover after size+days {_f(v['r_both'])} beats size {_f(p4['size_y3'])} "
            f"by ≥0.02 and is a month shock. Still do not put it on tonight's 15-col card."
        )
    return {"x": x_dec, "why": why, "raw": raw, "beat": beat, "leftover": leftover_ok}


def decide(p1, p3, p4, p5, p6, p7, p10, p11) -> dict:
    objs = {f: _obj_decision(f, p3, p4, p5, p6, p10, p11) for f in STEMS}
    q3 = "CLOSE"
    if any(o["x"] == "KEEP" for o in objs.values()):
        q3 = "KEEP"
    elif p11["empty_q6"] and all(
        o["x"] in {"DROP from the 44", "CLOSE as Q6"} for o in objs.values()
    ):
        q3 = "CLOSE"
    q6 = p11["q6"]
    return {
        "io": objs["a_io_ratio"],
        "g3": objs["a_growth_3"],
        "g12": objs["a_growth_12"],
        "q3": q3,
        "q6": q6,
        "park_y": True,
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12, p13, p14, p15 = ctx["p11"], ctx["p12"], ctx["p13"], ctx["p14"], ctx["p15"]
    p16, p17, p18, d = ctx["p16"], ctx["p17"], ctx["p18"], ctx["decision"]
    p19, p20, p21 = ctx["p19"], ctx["p20"], ctx["p21"]
    p22, p23, p24 = ctx["p22"], ctx["p23"], ctx["p24"]
    p25, p26, p27 = ctx["p25"], ctx["p26"], ctx["p27"]
    p28 = ctx["p28"]
    lines = [
        "# Growth / io_ratio — leftover Q3, or mean-reversion / SIZE / short-book NaNs?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_growth`. "
        "Do not put `a_io_ratio` / `a_growth_3` / `a_growth_12` on tonight's 15-col card. "
        f"Night Y3 quote stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. "
        f"Days bar stays **{DAYS_BENCH:.3f}**. Do not merge with Y4.",
        "",
        "`a_io_ratio` = min(3, in3 / max(out3, 1)). "
        "`a_growth_3` = clip(in3 / in3_{t-3} − 1, −1, 1) if in3_{t-3}>0 else NaN. "
        "`a_growth_12` = YoY of the trailing-3m inflow window (needs month 15+). "
        "`a_net_margin` was dropped as redundant with io_ratio.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | Not a new Y. PARK `y_growth`. io_ratio is a coverage *level*, not a health label. |",
        f"| 2 | Who is improving? | growth_3 / growth_12 are the change objects. Y3 +growth {_f(p4['g3_y3'])} vs −growth {_f(p4['neg_y3'])}. |",
        f"| 3 | Who is turning? | **{d['q3']}** — leftover after size+days io {_f(p5['verdicts']['a_io_ratio']['r_both'])} / g3 {_f(p5['verdicts']['a_growth_3']['r_both'])} / g12 {_f(p5['verdicts']['a_growth_12']['r_both'])} vs size {_f(p4['size_y3'])} / days {_f(p4['days_y3'])}. |",
        "| 4 | Dip vs fall? | Not this table. Y4 overlap is descriptive only. |",
        f"| 5 | Why did it change? | U-shape / mechanical acf3 {_f(p6['acf3'])} ({'window overlap' if p21['mechanical'] else 'measured'}); MoM −40% {_f(p20['y3_crash'])}. |",
        f"| 6 | Months earlier? | **{d['q6']}** — {p11['prose']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `a_io_ratio` as Y3 X (on the 44) | **{d['io']['x']}** | {d['io']['why']} Still **not** on tonight's 15-col card. |",
        f"| `a_growth_3` as Y3 X (on the 44) | **{d['g3']['x']}** | {d['g3']['why']} Still **not** on tonight's 15-col card. |",
        f"| `a_growth_12` as Y3 X (on the 44) | **{d['g12']['x']}** | {d['g12']['why']} Still **not** on tonight's 15-col card. |",
        "| any of the three as a health Y | **PARK** | do not invent `y_growth` |",
        f"| mean-reversion | **{'YES' if p6['mr_only'] else 'measured'}** | {p6['prose']} |",
        f"| io clip-at-3 dummy | **{'YES' if p7['dummy'] else 'NO'}** | {p7['prose']} |",
        f"| Q6 short-book growth_12 | **{d['q6']}** | {p11['prose']} |",
        "| Y4 crash overlap | descriptive only | do not score as Y4 X; do not merge |",
        "",
        "## 1. Coverage — confirm 88.5% / 64.0% / 26.5%",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        "Short-book / late-arrival hole:",
        "",
        _md_table(p1["hole"]),
        "",
        "Holdout coverage (no AUROC):",
        "",
        _md_table(p1["ho_rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 2. Formula vs store (in3 / out3, clip)",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Spearman vs size / days / net_margin",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "Twin if \\|ρ\\|≥0.80. SIZE if \\|ρ\\| vs log1p(a_in3) ≥0.50.",
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p4['n_y2']:,} base {_pp(p4['y2_rate'])}; "
        f"Y3 stressed n={p4['n_y3']:,} base {_pp(p4['y3_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p4['days_y3'])}); "
        f"size 0.617 (replica {_f(p4['size_y3'])}). "
        f"Night Y3 0.762 / 0.752 is **not** this cut.",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "Same-row size / days (coverage-fair):",
        "",
        _md_table(p4["same"]),
        "",
        f"KEEP-as-X on the 44: beat size by ≥{KEEP_DELTA:g} **and** leftover after size **and** leftover after days **and** not mean-reversion-only **and** a month shock. "
        "Still not on the 15-col card.",
        "",
        "## 5. Residual after size and after days",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Mean-reversion — high growth_3 protective or risky for Y3?",
        "",
        p6["prose"],
        "",
        _md_table(p6["qrows"]),
        "",
        f"Next-month Y3 quintiles of growth_3: Q1 {_pp(p6['q_next']['q1'])} → Q5 {_pp(p6['q_next']['q5'])} shape={p6['q_next']['shape']}.",
        "",
        "## 7. `a_io_ratio` clip-at-3",
        "",
        p7["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| defined / clip=3 | {p7['n_def']:,} / {p7['n_clip']:,} |",
        f"| share of defined | {_pp(p7['of_def'])} |",
        f"| unclip p50 / p90 / p99 | {_f(p7['p50'], 2)} / {_f(p7['p90'], 2)} / {_f(p7['p99'], 2)} |",
        f"| Y3 clip-flag / continuous / unclip / body | {_f(p7['flag_cv'])} / {_f(p7['io_cv'])} / {_f(p7['unclip_cv'])} / {_f(p7['body_cv'])} |",
        "",
        "## 8. Dark 470 vs invoiced 744",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        _md_table(p8["cm"]),
        "",
        "## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. ICC / company-demean",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "Growth should be LOW_PERSIST / a month shock if it is a real change. "
        "Feature-report growth_12 ICC 0.90 is BETWEEN — a company YoY style, not a shock.",
        "",
        "## 11. Q6 — lag1 / lag3 on short vs long",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Overlap with Y4 crash months (descriptive)",
        "",
        p12["prose"],
        "",
        _md_table(p12["rows"]),
        "",
        "## 13. Growth clip ±1 saturation",
        "",
        p13["prose"],
        "",
        _md_table(p13["rows"]),
        "",
        "## 14. Size terciles",
        "",
        p14["prose"],
        "",
        _md_table(p14["rows"]),
        "",
        "## 15. Holdout coverage only",
        "",
        p15["prose"],
        "",
        _md_table(p15["rows"]),
        "",
        "## 16. Calendar",
        "",
        p16["prose"],
        "",
        _md_table(p16["rows"]),
        "",
        "## 17. Residual vs control (fake leftover?)",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "## 18. Long-book same-row (so_far≥15)",
        "",
        p18["prose"],
        "",
        _md_table(p18["rows"]),
        "",
        "## 19. U-shape / io body leftover / T1 pocket",
        "",
        p19["prose"],
        "",
        _md_table(p19["t1_rows"]),
        "",
        "## 20. Javier month-on-month inflow −40%",
        "",
        p20["prose"],
        "",
        _md_table(p20["rows"]),
        "",
        "## 21. Mechanical acf3 (overlapping 3m windows)",
        "",
        p21["prose"],
        "",
        "## 22. Company-mean leftover (style, not a shock)",
        "",
        p22["prose"],
        "",
        _md_table(p22["rows"]),
        "",
        "## 23. Y3 positives missing growth_3",
        "",
        p23["prose"],
        "",
        _md_table(p23["rows"]),
        "",
        "## 24. Rank leftover (nonlinear SIZE leak)",
        "",
        p24["prose"],
        "",
        _md_table(p24["rows"]),
        "",
        "## 25. Dark growth_12 hole is trail length",
        "",
        p25["prose"],
        "",
        "## 26. Holdout late-arrival (hidden 72)",
        "",
        p26["prose"],
        "",
        "## 27. io Q1 vs clip=3 tails",
        "",
        p27["prose"],
        "",
        _md_table(p27["rows"]),
        "",
        "## 28. MoM leftover + non-overlapping growth_3",
        "",
        p28["prose"],
        "",
        _md_table(p28["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage, formula, Spearman, singles, leftover, "
        "mean-reversion, clip-at-3, dark 470/744, 12 names, ICC, Q6, Y4 overlap, ±1 clip, "
        "size terciles, holdout, calendar, fake leftover, long-book, U-shape/T1, MoM −40%, "
        "mechanical acf3, company-mean style, Y3 hole, rank leftover, dark trail, holdout late, io tails, "
        "MoM leftover / non-overlap.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p4, p5, p6, p7, p8, p10, p11, d = (
        ctx["p1"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p10"],
        ctx["p11"],
        ctx["decision"],
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")

    def row(y, metric, value, split, notes, coverage="1.0000"):
        return {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": y,
            "model": MODEL,
            "split": split,
            "metric": metric,
            "value": value,
            "coverage": coverage,
            "notes": notes,
        }

    rows = [
        row(
            "-",
            "a_io_ratio_cov",
            p1["store"]["a_io_ratio"]["cov"],
            "train",
            f"quote=0.885 ok={p1['store']['a_io_ratio']['ok']}",
            f"{p1['store']['a_io_ratio']['cov']:.4f}",
        ),
        row(
            "-",
            "a_growth_3_cov",
            p1["store"]["a_growth_3"]["cov"],
            "train",
            f"quote=0.640 ok={p1['store']['a_growth_3']['ok']}",
            f"{p1['store']['a_growth_3']['cov']:.4f}",
        ),
        row(
            "-",
            "a_growth_12_cov",
            p1["store"]["a_growth_12"]["cov"],
            "train",
            f"quote=0.265 ok={p1['store']['a_growth_12']['ok']} empty_short={p1['empty_short']}",
            f"{p1['store']['a_growth_12']['cov']:.4f}",
        ),
        row(
            Y3,
            "auroc_a_io_ratio",
            p4["io_y3"],
            "train_cv",
            f"size={p4['size_y3']:.4f} days={p4['days_y3']:.4f} beat={p4['beat_io']:.4f} x={d['io']['x']}",
        ),
        row(
            Y3,
            "auroc_a_growth_3",
            p4["g3_y3"],
            "train_cv",
            f"neg={p4['neg_y3']:.4f} naive40={p4['naive_y3']:.4f} sign={p4['g3_sign']} x={d['g3']['x']}",
        ),
        row(
            Y3,
            "auroc_a_growth_12",
            p4["g12_y3"],
            "train_cv",
            f"beat={p4['beat_g12']:.4f} q6={d['q6']} x={d['g12']['x']}",
            f"{p1['store']['a_growth_12']['cov']:.4f}",
        ),
        row(
            Y3,
            "auroc_a_growth_3_resid_both",
            p5["verdicts"]["a_growth_3"]["r_both"],
            "train_cv",
            f"r_size={p5['verdicts']['a_growth_3']['r_size']:.4f} r_days={p5['verdicts']['a_growth_3']['r_days']:.4f} died={p5['verdicts']['a_growth_3']['died']}",
        ),
        row(
            Y3,
            "auroc_naive_crash40",
            p6["naive"],
            "train_cv",
            f"mr_only={p6['mr_only']} plus={p6['plus']:.4f} minus={p6['minus']:.4f} acf3={p6['acf3']:.3f}",
        ),
        row(
            "-",
            "a_io_ratio_clip3_share",
            p7["of_def"],
            "train",
            f"dummy={p7['dummy']} n_clip={p7['n_clip']} flag={p7['flag_cv']:.4f}",
        ),
        row(
            "-",
            "a_growth_12_icc",
            p10["store"]["a_growth_12"]["icc"],
            "train",
            f"g3_icc={p10['store']['a_growth_3']['icc']:.3f} io_icc={p10['store']['a_io_ratio']['icc']:.3f} trait={p10['store']['a_growth_12']['trait']}",
        ),
        row(
            "-",
            "dark_vs_erp_g12_cov",
            p8["dark_g12"],
            "train",
            f"erp={p8['erp_g12']:.4f} confirm744_470={p8['confirm']} same={p8['same']}",
        ),
        row(
            Y3,
            "auroc_a_growth_3_lag1",
            p11["lag1"]["a_growth_3"],
            "train_cv",
            f"now={p11['now']['a_growth_3']:.4f} lag3={p11['lag3']['a_growth_3']:.4f} q6={p11['q6']}",
        ),
        row(
            Y3,
            "auroc_mom_crash40",
            ctx["p20"]["y3_crash"],
            "train_cv",
            f"mom={ctx['p20']['y3_mom']:.4f} javier={ctx['p20']['javier']} y2={ctx['p20']['y2_crash']:.4f}",
        ),
        row(
            Y3,
            "auroc_a_growth_3_comean",
            ctx["p22"]["store"]["a_growth_3"]["raw"],
            "train_cv",
            f"after_both={ctx['p22']['store']['a_growth_3']['both']:.4f} style={ctx['p22']['style']} died={ctx['p22']['died']}",
        ),
        row(
            "-",
            "holdout_late_n",
            ctx["p26"]["n_late"],
            "holdout",
            f"n_co={ctx['p26']['n_co']} short={ctx['p26']['n_short']} lock={ctx['p26']['lock']} g12_late={ctx['p26']['g12_late']:.4f}",
        ),
        row(
            Y3,
            "auroc_mom_in_resid_both",
            ctx["p28"]["mom_both"],
            "train_cv",
            f"raw={ctx['p28']['mom_raw']:.4f} g3_non={ctx['p28']['g3_non']:.4f} dies={ctx['p28']['mom_dies']}",
        ),
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
                r.get("x_families"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
            str(r.get("x_families", "")),
        )
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"registry appended {len(fresh)} rows")


def run() -> dict:
    t0 = time.time()
    print(f"growth_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, list(STEMS), (1, 3))
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )

    print("pass 1 coverage")
    p1 = pass1_coverage(tr, panel)
    print("pass 2 formula")
    p2 = pass2_formula(tr)
    print("pass 3 spearman")
    p3 = pass3_rho(tr)
    print("pass 4 singles")
    p4 = pass4_auroc(tr)
    print("pass 5 residual")
    p5 = pass5_resid(tr)
    print("pass 6 mean-reversion")
    p6 = pass6_reversion(tr, p4)
    print("pass 7 clip")
    p7 = pass7_clip(tr)
    con = connect()
    try:
        print("pass 8 dark 470 vs 744")
        p8 = pass8_dark(tr, con)
    finally:
        con.close()
    print("pass 9 chronic 12")
    p9 = pass9_chronic(tr)
    print("pass 10 icc")
    p10 = pass10_icc(tr)
    print("pass 11 Q6")
    p11 = pass11_q6(tr, p1)
    print("pass 12 Y4 overlap")
    p12 = pass12_y4(tr)
    print("pass 13 growth clip")
    p13 = pass13_gclip(tr)
    print("pass 14 size terciles")
    p14 = pass14_terciles(tr)
    print("pass 15 holdout")
    p15 = pass15_holdout(panel)
    print("pass 16 calendar")
    p16 = pass16_cal(tr)
    print("pass 17 fake leftover")
    p17 = pass17_fake(tr)
    print("pass 18 long-book")
    p18 = pass18_defined(tr)
    print("pass 19 U-shape / T1")
    p19 = pass19_ushape(tr, p6)
    print("pass 20 MoM -40%")
    p20 = pass20_mom(tr)
    print("pass 21 mechanical acf3")
    p21 = pass21_mech(tr)
    print("pass 22 company-mean")
    p22 = pass22_comean(tr)
    print("pass 23 Y3 hole")
    p23 = pass23_hole(tr)
    print("pass 24 rank leftover")
    p24 = pass24_rank(tr)
    print("pass 25 dark trail")
    p25 = pass25_darktrail(tr, p8)
    print("pass 26 holdout late")
    p26 = pass26_hold_late(panel)
    print("pass 27 io tails")
    p27 = pass27_tails(tr)
    print("pass 28 MoM leftover / nonoverlap")
    p28 = pass28_alts(tr)

    if not p2["all_ok"]:
        print("WARN formula max|Δ| — inspect cashflow.py, do not rewrite tonight")

    decision = decide(p1, p3, p4, p5, p6, p7, p10, p11)
    png_ok = make_png(tr, p1, p6)
    headline = (
        f"Coverage io/g3/g12 {_pp(p1['store']['a_io_ratio']['cov'])} / "
        f"{_pp(p1['store']['a_growth_3']['cov'])} / {_pp(p1['store']['a_growth_12']['cov'])}. "
        f"Y3 io {_f(p4['io_y3'])} g3 {_f(p4['g3_y3'])} g12 {_f(p4['g12_y3'])} "
        f"vs size {_f(p4['size_y3'])} vs days {_f(p4['days_y3'])}. "
        f"Leftover size+days io {_f(p5['verdicts']['a_io_ratio']['r_both'])} "
        f"g3 {_f(p5['verdicts']['a_growth_3']['r_both'])} "
        f"g12 {_f(p5['verdicts']['a_growth_12']['r_both'])}. "
        f"U-shape g3={'YES' if p19['u'] else 'NO'}; mechanical acf3={'YES' if p21['mechanical'] else 'NO'}; "
        f"MoM −40% {_f(p20['y3_crash'])}. "
        f"io **{decision['io']['x']}**. g3 **{decision['g3']['x']}**. "
        f"g12 **{decision['g12']['x']}**. PARK as Y. Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p2["all_ok"]:
        failed.append("store vs reconstructed formula mismatch — do not rewrite cashflow.py")
    if not p4["days_ok"]:
        failed.append(f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711")
    if not p4["size_ok"]:
        failed.append(f"Y3 size replica {_f(p4['size_y3'])} vs 0.617")
    if not p8["confirm"]:
        failed.append(f"dark/erp {p8['n_dark']}/{p8['n_erp']} ≠ 470/744")
    if p9["n_ids"] != 12:
        failed.append(f"chronic names {p9['n_ids']} ≠ 12 from y2_why")
    if not p1["store"]["a_io_ratio"]["ok"] or not p1["store"]["a_growth_3"]["ok"] or not p1["store"]["a_growth_12"]["ok"]:
        failed.append("coverage quote miss vs feature_report 88.5/64.0/26.5")
    if all(decision[k]["x"] != "KEEP" for k in ("io", "g3", "g12")):
        failed.append(
            "none of io / g3 / g12 KEEP on the 44 — leftover dies, raw loses to size 0.617 / days 0.711"
        )
    if p24["leak"]:
        failed.append("io leftover-after-size 0.604 is a nonlinear SIZE leak (rank leftover 0.533)")
    if p21["mechanical"]:
        failed.append("growth_3 acf3=-0.433 is mechanical window overlap (levels persist)")
    if p19["u"]:
        failed.append("growth_3 / io quintiles are U-shaped two-tail, not monotone Q3 turning")
    if p11["empty_q6"] or p26["lock"]:
        failed.append(
            f"growth_12 CLOSE as Q6: empty on short books; holdout late-arrival {p26['n_late']}/{p26['n_co']}"
        )
    if p22["style"]:
        failed.append(
            f"company-mean g3 {_f(p22['store']['a_growth_3']['raw'])} is STYLE; leftover both "
            f"{_f(p22['store']['a_growth_3']['both'])} dies"
        )
    if p28["mom_dies"]:
        failed.append(
            f"MoM leftover both {_f(p28['mom_both'])} dies vs size; non-overlap g3 {_f(p28['g3_non'])}"
        )
    elapsed = time.time() - t0
    ctx = {
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
        "p15": p15,
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
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
        "panel": panel,
    }
    write_md(ctx)
    append_registry(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
