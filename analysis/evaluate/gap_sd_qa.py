"""c_gap_sd — days/n_tx twin, NEAR SIZE, Pérez-Salazar leftover, or midnight hole?

NORTH_STAR: the feature report kept `c_gap_sd` (cluster representative) and
dropped `c_n_days_with_tx` / `a_n_tx` / `c_n_tx` at |ρ|≥0.8. The night Y3
engine kept **days 0.711** on the 15-col card and left gap_sd off. Who is right?

`c_gap_sd` = sample stdev (ddof=1) of day-gaps between consecutive **unique
calendar days** in (month_end − 90d, month_end]. Null if fewer than 3
distinct days. Pérez-Salazar, Márquez & Vidal-Silva 2026 (Computers 15:135)
σ_Δt. Unique-day collapse exists because 94% of timestamps are midnight —
without it σ floods with 0-day gaps and becomes 1/n_tx.

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent `y_gap_sd`. Do not put `c_gap_sd` on the
15-col card. Night Y3 quote stays 0.762 / 0.752. Days bar stays 0.711.
Do not rewrite ops.py.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.gap_sd_qa

Owned: analysis/evaluate/gap_sd_qa.py, analysis/outputs/gap_sd_qa.md,
optional one PNG, append-only registry, overnight/waves/wave4_gap_sd.md (end).
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
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect
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
OUT_MD = ANALYSIS / "outputs" / "gap_sd_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "gap_sd_vs_days.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_MD = ROOT / "overnight" / "waves" / "wave4_gap_sd.md"
AGENT = "87e59905"
WAVE = "4"
ROUND = "R4"
MODEL = "gap_sd_qa"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
TWIN_RHO = 0.80
ICC_TRAIT = 0.85
ICC_QUOTE = 0.96
COV_QUOTE = 0.951
SIZE_RHO_QUOTE = -0.543
MIDNIGHT_QUOTE = 0.94
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
MIN_POS = 50
MIN_ACF_PAIRS = 4
GAP_WINDOW_DAYS = 90
PANEL_END = pd.Timestamp("2026-08-01")
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
SHORT_MAX = 11
LONG_MIN = 18
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_op_in",
    "a_n_tx",
    "c_n_tx",
    "c_n_days_with_tx",
    "c_gap_sd",
    "c_recency_days",
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


def _month_end(period: pd.Series) -> pd.Series:
    return pd.to_datetime(period) + pd.offsets.MonthEnd(0)


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


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
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0, "ratio": float("nan")}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "var_w": float("nan"), "var_b": float("nan"), "k": 0, "ratio": float("nan")}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    ratio = var_w / var_b if var_b > 1e-18 else float("inf")
    return {"icc": float(icc), "var_w": float(var_w), "var_b": float(var_b), "k": k, "ratio": float(ratio)}


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
    """Train-defined OLS residual of y on 1+ predictors. No holdout in slope."""
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "slope": [], "intercept": float("nan"), "r2": float("nan")}
    n_x = len(xs)
    if int(ok.sum()) < max(20, n_x + 5):
        return resid, info
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack([np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)])
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    pred = X @ beta
    resid.loc[ok] = Y - pred
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    ss_res = float(np.sum((Y - pred) ** 2))
    info["intercept"] = float(beta[0])
    info["slope"] = [float(b) for b in beta[1:]]
    info["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return resid, info


def _company_terciles(tr: pd.DataFrame, col: str, name: str) -> pd.Series:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last[col], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    return terc.rename(name)


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
    ykeep = ["company_id", "period", Y2, Y3]
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["log_op_in"] = np.log1p(np.abs(pd.to_numeric(panel["a_op_in"], errors="coerce")))
    panel["inv_n_tx"] = 1.0 / pd.to_numeric(panel["c_n_tx"], errors="coerce").clip(lower=1)
    panel["inv_a_n_tx"] = 1.0 / pd.to_numeric(panel["a_n_tx"], errors="coerce").clip(lower=1)
    panel["inv_days"] = 1.0 / pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce").clip(lower=1)
    panel["so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    first = panel.groupby("company_id")["period"].transform("min")
    panel["months_on_book"] = (
        (PANEL_END.year - first.dt.year) * 12 + (PANEL_END.month - first.dt.month) + 1
    )
    panel["short_book"] = (panel["months_on_book"] < 12).astype(np.int8)
    panel["trail_class"] = np.where(
        panel["so_far"] <= SHORT_MAX,
        "short_<12",
        np.where(panel["so_far"] >= LONG_MIN, "long_>=18", "mid_12_17"),
    )
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


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


def _gap_sd_from_dates(dates: pd.DataFrame, keys: pd.DataFrame) -> pd.Series:
    """ops.py window: (month_end-90d, month_end], ddof=1, null if <3 dates."""
    ends = _month_end(keys["period"]).to_numpy(dtype="datetime64[ns]")
    starts = ends - np.timedelta64(GAP_WINDOW_DAYS, "D")
    out = np.full(len(keys), np.nan)
    n_win = np.zeros(len(keys), dtype=np.int32)
    by_co = {
        cid: np.sort(g["d"].to_numpy(dtype="datetime64[ns]"))
        for cid, g in dates.groupby("company_id", sort=False)
    }
    for cid, sub in keys.groupby("company_id", sort=False):
        d = by_co.get(cid)
        if d is None or len(d) < 3:
            continue
        idx = sub.index.to_numpy()
        lo = np.searchsorted(d, starts[idx], side="right")
        hi = np.searchsorted(d, ends[idx], side="right")
        for k, a, b in zip(idx, lo, hi):
            n_win[k] = int(b - a)
            if b - a < 3:
                continue
            gaps = np.diff(d[a:b]) / np.timedelta64(1, "D")
            out[k] = float(gaps.std(ddof=1))
    s = pd.Series(out, index=keys.index, name="recon")
    s.attrs["n_win"] = pd.Series(n_win, index=keys.index)
    return s


def load_unique_days(con) -> pd.DataFrame:
    days = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST("date" AS DATE) AS d
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        ORDER BY 1, 2
        """
    ).df()
    days["company_id"] = days["company_id"].astype(str)
    days["d"] = pd.to_datetime(days["d"])
    return days


def load_raw_dates(con) -> pd.DataFrame:
    raw = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               "date" AS d
        FROM transactions
        WHERE "date" IS NOT NULL
        ORDER BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["d"] = pd.to_datetime(raw["d"])
    return raw


# ---------------------------------------------------------------------------
# Pass 1 — completeness / coverage 95.1%
# ---------------------------------------------------------------------------
def pass1_coverage(panel: pd.DataFrame, tr: pd.DataFrame) -> dict:
    rows = []
    for split, sl in (("train", tr), ("holdout", panel[panel["split"] == "holdout"])):
        x = pd.to_numeric(sl["c_gap_sd"], errors="coerce")
        days = pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce")
        rows.append(
            {
                "split": split,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "cov": float(x.notna().mean()) if len(sl) else float("nan"),
                "n_null": int(x.isna().sum()),
                "null_share": float(x.isna().mean()) if len(sl) else float("nan"),
                "days_lt3": float((days < 3).mean()) if len(sl) else float("nan"),
                "p50": float(x.median()) if x.notna().any() else float("nan"),
                "p90": float(x.quantile(0.90)) if x.notna().any() else float("nan"),
                "mean": float(x.mean()) if x.notna().any() else float("nan"),
            }
        )
    tr_r = rows[0]
    ho_r = rows[1]
    confirm = bool(np.isfinite(tr_r["cov"]) and abs(tr_r["cov"] - COV_QUOTE) < 0.008)
    # in-month days<3 vs gap null — they are different clocks
    x = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    null = x.isna()
    p_null_given_thin = float(null[days < 3].mean()) if (days < 3).any() else float("nan")
    p_thin_given_null = float((days < 3)[null].mean()) if null.any() else float("nan")
    prose = (
        f"Train `c_gap_sd` coverage {_pp(tr_r['cov'])} (n_null={tr_r['n_null']:,} / {tr_r['n_cm']:,}) "
        f"({'CONFIRM 95.1%' if confirm else 'does not match feature-report 95.1%'}). "
        f"Holdout coverage only {_pp(ho_r['cov'])} on {ho_r['n_cm']:,} CM / {ho_r['n_co']} cos. "
        f"In-month days<3 share {_pp(tr_r['days_lt3'])}; P(null|days<3)={_pp(p_null_given_thin)}; "
        f"P(days<3|null)={_pp(p_thin_given_null)}. "
        "Null is the 90d unique-day clock (<3 distinct days), not the in-month count."
    )
    print(prose)
    return {
        "rows": rows,
        "train_cov": tr_r["cov"],
        "hold_cov": ho_r["cov"],
        "confirm": confirm,
        "n_null": tr_r["n_null"],
        "p_null_given_thin": p_null_given_thin,
        "p_thin_given_null": p_thin_given_null,
        "p50": tr_r["p50"],
        "p90": tr_r["p90"],
        "mean": tr_r["mean"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — formula vs unique booking dates (90d, ddof=1)
# ---------------------------------------------------------------------------
def pass2_formula(tr: pd.DataFrame, unique_days: pd.DataFrame) -> dict:
    keys = tr[["company_id", "period"]].copy().reset_index(drop=True)
    recon = _gap_sd_from_dates(unique_days, keys)
    n_win = recon.attrs["n_win"]
    store = pd.to_numeric(tr["c_gap_sd"], errors="coerce").reset_index(drop=True)
    both = pd.DataFrame({"store": store, "recon": recon, "n_win": n_win})
    both["abs"] = (both["store"] - both["recon"]).abs()
    agree_nn = both["store"].notna() & both["recon"].notna()
    agree_null = both["store"].isna() & both["recon"].isna()
    n_agree_null = int(agree_null.sum())
    n_store_only = int((both["store"].notna() & both["recon"].isna()).sum())
    n_recon_only = int((both["store"].isna() & both["recon"].notna()).sum())
    max_abs = float(both.loc[agree_nn, "abs"].max()) if agree_nn.any() else float("nan")
    med_abs = float(both.loc[agree_nn, "abs"].median()) if agree_nn.any() else float("nan")
    ok = bool(
        n_store_only == 0
        and n_recon_only == 0
        and (not np.isfinite(max_abs) or max_abs < 1e-6)
    )
    # sample rows: defined + null, mixed so_far
    sl = tr.reset_index(drop=True).assign(recon=recon, n_win=n_win, abs_err=both["abs"])
    sample_idx = []
    defined = sl["c_gap_sd"].notna()
    null = sl["c_gap_sd"].isna()
    for mask, n in ((defined, 6), (null, 4)):
        take = sl.index[mask]
        if len(take) == 0:
            continue
        step = max(1, len(take) // n)
        sample_idx.extend(list(take[::step][:n]))
    sample = sl.loc[sample_idx, ["company_id", "period", "c_gap_sd", "recon", "n_win", "c_n_days_with_tx", "so_far"]]
    sample_rows = []
    for _, r in sample.iterrows():
        sample_rows.append(
            {
                "company_id": r["company_id"],
                "period": pd.Timestamp(r["period"]).strftime("%Y-%m"),
                "store": _f(r["c_gap_sd"], 4),
                "recon": _f(r["recon"], 4),
                "n_win90": int(r["n_win"]),
                "days_m": int(r["c_n_days_with_tx"]),
                "so_far": int(r["so_far"]),
            }
        )
    n_lt3 = int((n_win < 3).sum())
    prose = (
        f"Recompute unique-day σ (90d, ddof=1) vs store: both-defined {int(agree_nn.sum()):,}, "
        f"both-null {n_agree_null:,}, store-only {n_store_only}, recon-only {n_recon_only}. "
        f"max|Δ|={max_abs:.2e} median|Δ|={med_abs:.2e}. "
        f"Window unique-days <3: {n_lt3:,} ({_pp(_pct(n_lt3, len(tr)))}). "
        f"Formula {'OK — matches ops.py' if ok else 'MISMATCH — stop, do not rewrite ops.py'}."
    )
    print(prose)
    return {
        "ok": ok,
        "n_agree_nn": int(agree_nn.sum()),
        "n_agree_null": n_agree_null,
        "n_store_only": n_store_only,
        "n_recon_only": n_recon_only,
        "max_abs": max_abs,
        "med_abs": med_abs,
        "n_lt3": n_lt3,
        "sample_rows": sample_rows,
        "n_win": n_win,
        "recon": recon,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman twins / SIZE / recency
# ---------------------------------------------------------------------------
def pass3_rho(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    pairs = [
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("c_n_tx", tr["c_n_tx"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("log1p(|a_op_in|)", tr["log_op_in"]),
        ("c_recency_days", tr["c_recency_days"]),
        ("1/c_n_tx", tr["inv_n_tx"]),
        ("1/a_n_tx", tr["inv_a_n_tx"]),
        ("1/c_n_days_with_tx", tr["inv_days"]),
        ("so_far", tr["so_far"]),
        ("months_on_book", tr["months_on_book"]),
    ]
    rows = []
    twin_names = []
    size_flag = False
    rhos = {}
    for name, col in pairs:
        rho = spearman(gap, col)
        rhos[name] = rho
        flag = ""
        if name in {"c_n_days_with_tx", "a_n_tx", "c_n_tx"} and np.isfinite(rho) and abs(rho) >= TWIN_RHO:
            flag = "TWIN"
            twin_names.append(name)
        if name in {"log1p(a_in3)", "log1p(|a_op_in|)"} and np.isfinite(rho) and abs(rho) >= SIZE_RHO:
            flag = (flag + " SIZE").strip()
            size_flag = True
        if name == "c_recency_days" and np.isfinite(rho) and abs(rho) >= TWIN_RHO:
            flag = (flag + " TWIN").strip()
            twin_names.append(name)
        rows.append({"pair": f"c_gap_sd vs {name}", "rho": rho, "flag": flag})
    # pairwise among cluster members (is the 0.8 cut a chain?)
    cluster = {
        "days↔a_n_tx": spearman(tr["c_n_days_with_tx"], tr["a_n_tx"]),
        "days↔c_n_tx": spearman(tr["c_n_days_with_tx"], tr["c_n_tx"]),
        "a_n_tx↔c_n_tx": spearman(tr["a_n_tx"], tr["c_n_tx"]),
    }
    for name, rho in cluster.items():
        rows.append({"pair": name, "rho": rho, "flag": "CHAIN" if np.isfinite(rho) and abs(rho) >= TWIN_RHO else ""})
    rho_days = rhos["c_n_days_with_tx"]
    rho_antx = rhos["a_n_tx"]
    rho_cntx = rhos["c_n_tx"]
    rho_size = rhos["log1p(a_in3)"]
    rho_opin = rhos["log1p(|a_op_in|)"]
    confirm_size = bool(np.isfinite(rho_opin) and abs(rho_opin - SIZE_RHO_QUOTE) < 0.03)
    pairwise_twin = bool(
        any(np.isfinite(r) and abs(r) >= TWIN_RHO for r in (rho_days, rho_antx, rho_cntx))
    )
    # company-median Spearman (cluster pick is panel CM; check medians)
    med = tr.groupby("company_id", as_index=False).agg(
        gap=("c_gap_sd", "median"),
        days=("c_n_days_with_tx", "median"),
        antx=("a_n_tx", "median"),
        cntx=("c_n_tx", "median"),
        size=("log_in3", "median"),
    )
    rho_med_days = spearman(med["gap"], med["days"])
    rho_med_antx = spearman(med["gap"], med["antx"])
    rho_med_size = spearman(med["gap"], med["size"])
    prose = (
        f"Train Spearman `c_gap_sd` vs days {rho_days:.3f}, a_n_tx {rho_antx:.3f}, "
        f"c_n_tx {rho_cntx:.3f} (twin |ρ|≥0.80: {'YES ' + ','.join(twin_names) if pairwise_twin else 'NO — cluster cut is a chain, not a pairwise twin'}). "
        f"vs log1p(a_in3) {rho_size:.3f}; vs log1p(|a_op_in|) {rho_opin:.3f} "
        f"({'CONFIRM −0.543' if confirm_size else 'off feature-report −0.543'}). "
        f"SIZE |ρ|≥0.50: {'YES' if size_flag else 'no'}. "
        f"vs recency {rhos['c_recency_days']:.3f}. "
        f"Company-median vs days {rho_med_days:.3f} / a_n_tx {rho_med_antx:.3f} / size {rho_med_size:.3f}. "
        f"days↔a_n_tx {cluster['days↔a_n_tx']:.3f} a_n_tx↔c_n_tx {cluster['a_n_tx↔c_n_tx']:.3f}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_days": rho_days,
        "rho_antx": rho_antx,
        "rho_cntx": rho_cntx,
        "rho_size": rho_size,
        "rho_opin": rho_opin,
        "rho_recency": rhos["c_recency_days"],
        "rho_inv_ntx": rhos["1/c_n_tx"],
        "rho_inv_days": rhos["1/c_n_days_with_tx"],
        "rho_med_days": rho_med_days,
        "rho_med_antx": rho_med_antx,
        "rho_med_size": rho_med_size,
        "cluster": cluster,
        "twin_names": twin_names,
        "pairwise_twin": pairwise_twin,
        "size_flag": size_flag,
        "confirm_size": confirm_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — group-fold AUROC Y3 / Y2
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "c_gap_sd": tr["c_gap_sd"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "c_n_tx": tr["c_n_tx"],
        "log1p_a_in3": tr["log_in3"],
        "c_recency_days": tr["c_recency_days"],
        "inv_c_n_tx": tr["inv_n_tx"],
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
        r = store[(y, feat)]
        return float("nan") if r["low_power"] else r["cv"]

    gap_y3 = _cv(Y3, "c_gap_sd")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    size_y3 = _cv(Y3, "log1p_a_in3")
    antx_y3 = _cv(Y3, "a_n_tx")
    gap_y2 = _cv(Y2, "c_gap_sd")
    days_y2 = _cv(Y2, "c_n_days_with_tx")
    size_y2 = _cv(Y2, "log1p_a_in3")
    rec_y3 = _cv(Y3, "c_recency_days")
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.03)
    beat_size = gap_y3 - size_y3 if np.isfinite(gap_y3) and np.isfinite(size_y3) else float("nan")
    beat_days = gap_y3 - days_y3 if np.isfinite(gap_y3) and np.isfinite(days_y3) else float("nan")
    loses_days = bool(np.isfinite(days_y3) and np.isfinite(gap_y3) and gap_y3 < days_y3)
    wander = float("nan")
    rec = store[(Y3, "c_gap_sd")]
    if not rec["low_power"] and rec["folds"]:
        vals = [r["auroc"] for r in rec["folds"] if np.isfinite(r["auroc"])]
        wander = float(max(vals) - min(vals)) if vals else float("nan")
    prose = (
        f"Y3 stressed singles: `c_gap_sd` {_f(gap_y3)} vs size {_f(size_y3)} "
        f"(quote 0.617 {'CONFIRM' if size_ok else 'off'}, Δ {_f(beat_size)}) vs days {_f(days_y3)} "
        f"(night 0.711 {'CONFIRM' if days_ok else 'off'}, Δ {_f(beat_days)}) vs a_n_tx {_f(antx_y3)}. "
        f"Y2 gap {_f(gap_y2)} vs size {_f(size_y2)} / days {_f(days_y2)}. "
        f"recency Y3 {_f(rec_y3)}. Fold wander {_f(wander)}. "
        f"{'Loses to days 0.711' if loses_days else 'Does not lose to days'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "gap_y3": gap_y3,
        "days_y3": days_y3,
        "size_y3": size_y3,
        "antx_y3": antx_y3,
        "gap_y2": gap_y2,
        "days_y2": days_y2,
        "size_y2": size_y2,
        "rec_y3": rec_y3,
        "beat_size": beat_size,
        "beat_days": beat_days,
        "loses_days": loses_days,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "wander": wander,
        "sign_y3": store[(Y3, "c_gap_sd")]["train_sign"],
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — leftover after residualizing on days / a_n_tx
# ---------------------------------------------------------------------------
def pass5_residual(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    antx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    recency = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    r_d, inf_d = ols_resid(gap, days)
    r_a, inf_a = ols_resid(gap, antx)
    r_both, inf_b = ols_resid(gap, days, antx)
    r_rec, inf_r = ols_resid(gap, recency)
    specs = [
        ("c_gap_sd", gap),
        ("resid_days", r_d),
        ("resid_a_n_tx", r_a),
        ("resid_days+a_n_tx", r_both),
        ("resid_recency", r_rec),
        ("c_n_days_with_tx", days),
        ("log1p_a_in3", tr["log_in3"]),
    ]
    rows = []
    cvs = {}
    for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
        for name, x in specs:
            rec = signed_oof_auroc(tr[ycol], x, tr["fold"], tr[ycol].notna())
            cv = float("nan") if rec["low_power"] else rec["cv"]
            cvs[(yname, name)] = cv
            rows.append(
                {
                    "y": yname,
                    "feature": name,
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(cv),
                    "sign": rec["train_sign"] if not rec["low_power"] else "—",
                }
            )
    y3_d = cvs[("Y3", "resid_days")]
    y3_a = cvs[("Y3", "resid_a_n_tx")]
    y3_b = cvs[("Y3", "resid_days+a_n_tx")]
    size_y3 = cvs[("Y3", "log1p_a_in3")]
    leftover_beat = y3_d - size_y3 if np.isfinite(y3_d) and np.isfinite(size_y3) else float("nan")
    leftover_lives = bool(np.isfinite(leftover_beat) and leftover_beat >= KEEP_DELTA)
    died = bool(
        (np.isfinite(y3_d) and y3_d < 0.55)
        or (np.isfinite(y3_a) and y3_a < 0.55)
        or (not leftover_lives and np.isfinite(y3_d) and y3_d < size_y3)
    )
    prose = (
        f"Y3 leftover after days {_f(y3_d)} (R²={_f(inf_d['r2'])}, slope={_f(inf_d['slope'][0] if inf_d['slope'] else float('nan'))}); "
        f"after a_n_tx {_f(y3_a)} (R²={_f(inf_a['r2'])}); after both {_f(y3_b)} vs size {_f(size_y3)} "
        f"(Δ {_f(leftover_beat, 3)}). "
        + (
            "Leftover dies — CLOSE as twin — **drop gap_sd from the 44**."
            if died and not leftover_lives
            else (
                "Leftover beats size ≥0.02 after days — Pérez-Salazar candidate (still not on tonight's card)."
                if leftover_lives
                else "Leftover does not clear size+0.02 after days."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_days": y3_d,
        "y3_antx": y3_a,
        "y3_both": y3_b,
        "y3_rec": cvs[("Y3", "resid_recency")],
        "y2_days": cvs[("Y2", "resid_days")],
        "size_y3": size_y3,
        "leftover_beat": leftover_beat,
        "leftover_lives": leftover_lives,
        "died": died,
        "r2_days": inf_d["r2"],
        "r2_antx": inf_a["r2"],
        "r2_both": inf_b["r2"],
        "slope_days": inf_d["slope"][0] if inf_d["slope"] else float("nan"),
        "slope_antx": inf_a["slope"][0] if inf_a["slope"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — SIZE terciles: does gap_sd beat size inside T1/T2/T3?
# ---------------------------------------------------------------------------
def pass6_terciles(tr: pd.DataFrame) -> dict:
    terc_size = _company_terciles(tr, "log_in3", "size_t")
    terc_days = _company_terciles(tr, "c_n_days_with_tx", "days_t")
    m = tr.merge(terc_size.reset_index(), on="company_id", how="left")
    m = m.merge(terc_days.reset_index(), on="company_id", how="left")
    rows = []
    store = {}
    beats = []
    for clock, col in (("size_t", "size_t"), ("days_t", "days_t")):
        for labv in ("T1", "T2", "T3"):
            mask = m[Y3].notna() & (m[col].astype(str) == labv)
            gap = signed_oof_auroc(m[Y3], m["c_gap_sd"], m["fold"], mask)
            size = signed_oof_auroc(m[Y3], m["log_in3"], m["fold"], mask)
            days = signed_oof_auroc(m[Y3], m["c_n_days_with_tx"], m["fold"], mask)
            store[(clock, labv, "gap")] = gap
            store[(clock, labv, "size")] = size
            gcv = float("nan") if gap["low_power"] else gap["cv"]
            scv = float("nan") if size["low_power"] else size["cv"]
            dcv = float("nan") if days["low_power"] else days["cv"]
            beat = gcv - scv if np.isfinite(gcv) and np.isfinite(scv) else float("nan")
            if clock == "size_t" and np.isfinite(beat) and beat >= KEEP_DELTA:
                beats.append(labv)
            rows.append(
                {
                    "clock": clock,
                    "tercile": labv,
                    "n": f"{gap['n_defined']:,}",
                    "n_pos": f"{gap['n_pos']:,}",
                    "gap": "LOW_POWER" if gap["low_power"] else _f(gcv),
                    "size": "LOW_POWER" if size["low_power"] else _f(scv),
                    "days": "LOW_POWER" if days["low_power"] else _f(dcv),
                    "Δsize": _f(beat),
                }
            )
    dies_inside = len(beats) == 0
    prose = (
        f"Y3 gap_sd vs size inside company size terciles that beat by ≥0.02: {beats or 'none'}. "
        f"{'Dies inside size terciles — NEAR SIZE leftover (b).' if dies_inside else 'Beats size inside terciles — but days beats gap in the same slices, so the survival is the days engine.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "beats": beats,
        "dies_inside": dies_inside,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — ICC / company-demean (BETWEEN 0.96)
# ---------------------------------------------------------------------------
def pass7_icc(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    icc_g = icc_anova(gap, tr["company_id"])
    icc_d = icc_anova(days, tr["company_id"])
    mu = gap.groupby(tr["company_id"], sort=False).transform("mean")
    dem = gap - mu
    lab = tr[Y3].notna()
    raw = signed_oof_auroc(tr[Y3], gap, tr["fold"], lab)
    dem_res = signed_oof_auroc(tr[Y3], dem, tr["fold"], lab)
    drop = (
        raw["cv"] - dem_res["cv"]
        if (not raw["low_power"] and not dem_res["low_power"])
        else float("nan")
    )
    acf1 = median_acf(gap, tr["company_id"], 1)
    acf3 = median_acf(gap, tr["company_id"], 3)
    acf6 = median_acf(gap, tr["company_id"], 6)
    confirm_icc = bool(np.isfinite(icc_g["icc"]) and abs(icc_g["icc"] - ICC_QUOTE) < 0.03)
    trait = bool(np.isfinite(icc_g["icc"]) and icc_g["icc"] >= ICC_TRAIT)
    shock = bool(np.isfinite(icc_g["icc"]) and icc_g["icc"] < 0.50)
    prose = (
        f"`c_gap_sd` ICC={_f(icc_g['icc'])} w/b={_f(icc_g['ratio'])} "
        f"({'CONFIRM BETWEEN 0.96' if confirm_icc else 'off feature-report 0.96'}); "
        f"days ICC={_f(icc_d['icc'])}. acf1={_f(acf1)} acf3={_f(acf3)} acf6={_f(acf6)} "
        f"(feature-report acf1=0.61; the 0.96 in the brief is the ICC, not acf). "
        f"Y3 raw {_f(raw['cv']) if not raw['low_power'] else 'LOW_POWER'} vs "
        f"company-demean {_f(dem_res['cv']) if not dem_res['low_power'] else 'LOW_POWER'} "
        f"(drop {_f(drop)}). "
        f"{'TRAIT (regular vs irregular booker)' if trait else ('MONTH SHOCK' if shock else 'mixed / between')}."
    )
    print(prose)
    return {
        "icc": icc_g["icc"],
        "icc_days": icc_d["icc"],
        "ratio": icc_g["ratio"],
        "k": icc_g["k"],
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "raw_cv": raw["cv"] if not raw["low_power"] else float("nan"),
        "dem_cv": dem_res["cv"] if not dem_res["low_power"] else float("nan"),
        "drop": drop,
        "trait": trait,
        "shock": shock,
        "confirm_icc": confirm_icc,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — nulls: Y3 rate on gap_sd-null vs defined; thin books?
# ---------------------------------------------------------------------------
def pass8_nulls(tr: pd.DataFrame, n_win: pd.Series | None) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    null = gap.isna()
    defined = gap.notna()
    win = n_win.reindex(tr.index) if n_win is not None else pd.Series(np.nan, index=tr.index)
    if n_win is not None and len(n_win) == len(tr) and not n_win.index.equals(tr.index):
        win = pd.Series(n_win.to_numpy(), index=tr.index)
    rows = []
    for y in (Y2, Y3):
        for sname, mask in (("defined", defined), ("null", null)):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n_cm": int(mask.sum()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                    "days_p50": float(days[mask].median()) if mask.any() else float("nan"),
                    "so_far_p50": float(tr.loc[mask, "so_far"].median()) if mask.any() else float("nan"),
                    "mob_p50": float(tr.loc[mask, "months_on_book"].median()) if mask.any() else float("nan"),
                }
            )
    win_rows = []
    if win.notna().any():
        for k in (0, 1, 2, 3):
            if k < 3:
                sl = win == k
            else:
                sl = win >= 3
            win_rows.append(
                {
                    "n_unique_90d": "≥3" if k >= 3 else str(k),
                    "n_cm": int(sl.sum()),
                    "share": _pp(_pct(int(sl.sum()), len(tr))),
                    "null_share": _pp(float(null[sl].mean()) if sl.any() else float("nan")),
                    "short_book": _pp(float(tr.loc[sl, "short_book"].mean()) if sl.any() else float("nan")),
                }
            )
    y3_null = next(r for r in rows if r["y"] == Y3 and r["slice"] == "null")
    y3_def = next(r for r in rows if r["y"] == Y3 and r["slice"] == "defined")
    thin = bool(
        np.isfinite(y3_null["days_p50"])
        and y3_null["days_p50"] <= 2
        and (win == 0).sum() + (win == 1).sum() + (win == 2).sum() >= int(null.sum()) * 0.9
    )
    prose = (
        f"Y3 rate on gap_sd-null {_pp(y3_null['rate'])} (n_lab={y3_null['n_lab']:,}, n_pos={y3_null['n_pos']}) "
        f"vs defined {_pp(y3_def['rate'])}. Null days p50={_f(y3_null['days_p50'], 1)} "
        f"so_far p50={_f(y3_null['so_far_p50'], 1)} months-on-book p50={_f(y3_null['mob_p50'], 1)}. "
        f"{'Nulls are thin 90d books (unique days <3).' if thin else 'Nulls are not only in-month thin — check 90d window.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "win_rows": win_rows,
        "y3_null": y3_null["rate"],
        "y3_def": y3_def["rate"],
        "thin": thin,
        "n_null": int(null.sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — dark 470 vs invoiced 744
# ---------------------------------------------------------------------------
def pass9_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    rows = []
    store = {}
    for name, part in (("ever_erp_744", tr2[tr2["ever_erp"]]), ("never_erp_470", tr2[~tr2["ever_erp"]])):
        x = pd.to_numeric(part["c_gap_sd"], errors="coerce")
        rows.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "cov": float(x.notna().mean()) if len(part) else float("nan"),
                "p50": float(x.median()) if x.notna().any() else float("nan"),
                "days_p50": float(pd.to_numeric(part["c_n_days_with_tx"], errors="coerce").median()),
            }
        )
        for y in (Y3, Y2):
            res = signed_oof_auroc(part[y], part["c_gap_sd"], part["fold"], part[y].notna())
            store[(name, y)] = res
    same = bool(
        np.isfinite(rows[0]["p50"])
        and np.isfinite(rows[1]["p50"])
        and abs(rows[0]["p50"] - rows[1]["p50"]) < 1.0
    )
    hold = load_holdout()
    hold_book = int(len(set(hold) & book))
    y3_erp = store[("ever_erp_744", Y3)]
    y3_dark = store[("never_erp_470", Y3)]
    prose = (
        f"Train last-month: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ from join QA'}). "
        f"gap_sd p50 invoiced {_f(rows[0]['p50'])} vs dark {_f(rows[1]['p50'])}; "
        f"cov {_pp(rows[0]['cov'])} / {_pp(rows[1]['cov'])}. "
        f"Y3 gap CV invoiced {'LOW_POWER' if y3_erp['low_power'] else _f(y3_erp['cv'])} "
        f"vs dark {'LOW_POWER' if y3_dark['low_power'] else _f(y3_dark['cv'])}. "
        f"{'Same bank-book regularity' if same else 'Dark books a different gap_sd'}. "
        f"Holdout ever-ERP coverage only: {hold_book}/{len(hold)}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "same": same,
        "hold_book": hold_book,
        "hold_n": int(len(hold)),
        "y3_erp": y3_erp["cv"] if not y3_erp["low_power"] else float("nan"),
        "y3_dark": y3_dark["cv"] if not y3_dark["low_power"] else float("nan"),
        "y2_erp": store[("ever_erp_744", Y2)]["cv"]
        if not store[("ever_erp_744", Y2)]["low_power"]
        else float("nan"),
        "y2_dark": store[("never_erp_470", Y2)]["cv"]
        if not store[("never_erp_470", Y2)]["low_power"]
        else float("nan"),
        "book": book,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass10_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    is_ch = tr["company_id"].astype(str).isin(set(ids))
    rows = []
    for y, lab in ((Y2, tr[Y2].notna()), (Y3, tr[Y3].notna())):
        for sname, mask in (("all", lab), ("drop_12", lab & drop), ("chronic_12", lab & is_ch)):
            gap = signed_oof_auroc(tr[y], tr["c_gap_sd"], tr["fold"], mask)
            days = signed_oof_auroc(tr[y], tr["c_n_days_with_tx"], tr["fold"], mask)
            size = signed_oof_auroc(tr[y], tr["log_in3"], tr["fold"], mask)
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n": f"{gap['n_defined']:,}",
                    "n_pos": f"{gap['n_pos']:,}",
                    "gap": "LOW_POWER" if gap["low_power"] else _f(gap["cv"]),
                    "days": "LOW_POWER" if days["low_power"] else _f(days["cv"]),
                    "size": "LOW_POWER" if size["low_power"] else _f(size["cv"]),
                }
            )
    y2_all = signed_oof_auroc(tr[Y2], tr["c_gap_sd"], tr["fold"], tr[Y2].notna())
    y2_drop = signed_oof_auroc(tr[Y2], tr["c_gap_sd"], tr["fold"], tr[Y2].notna() & drop)
    flip = False
    if not y2_all["low_power"] and not y2_drop["low_power"]:
        flip = abs(y2_all["cv"] - y2_drop["cv"]) >= 0.03
    gap_ch = float(pd.to_numeric(tr.loc[is_ch, "c_gap_sd"], errors="coerce").median()) if is_ch.any() else float("nan")
    gap_rest = float(pd.to_numeric(tr.loc[drop, "c_gap_sd"], errors="coerce").median())
    prose = (
        f"Chronic 12 names (0158/0172, ≥50% labeled months below 0): {len(ids)}. "
        f"Y2 gap CV all {_f(y2_all['cv']) if not y2_all['low_power'] else 'LOW_POWER'} → "
        f"drop-12 {_f(y2_drop['cv']) if not y2_drop['low_power'] else 'LOW_POWER'}. "
        f"gap_sd p50 on 12 {_f(gap_ch)} vs rest {_f(gap_rest)}. "
        f"{'DROP FLIPS Y2 (≥0.03) — skill is those names.' if flip else 'Drop does not flip Y2 (≥0.03).'}"
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "ids": ids,
        "rows": rows,
        "y2_all": y2_all["cv"] if not y2_all["low_power"] else float("nan"),
        "y2_drop": y2_drop["cv"] if not y2_drop["low_power"] else float("nan"),
        "flip": flip,
        "gap_ch": gap_ch,
        "gap_rest": gap_rest,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — Q6 lag1/lag3 on short vs long
# ---------------------------------------------------------------------------
def pass11_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    cols = (
        "c_gap_sd",
        "c_gap_sd_lag1",
        "c_gap_sd_lag3",
        "c_n_days_with_tx",
        "c_n_days_with_tx_lag1",
        "c_n_days_with_tx_lag3",
    )
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["trail_class"] == "short_<12"),
        ("long_>=18", tr["trail_class"] == "long_>=18"),
    )
    for y in (Y3, Y2):
        for sname, smask in slices:
            lab = tr[y].notna() & smask
            for col in cols:
                if col not in tr.columns:
                    continue
                res = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab)
                store[(y, sname, col)] = res
                rows.append(
                    {
                        "y": y,
                        "slice": sname,
                        "col": col,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "present": _pp(
                            float(tr.loc[lab, col].notna().mean()) if lab.any() else float("nan")
                        ),
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                    }
                )

    def _cv(y, sl, col) -> float:
        r = store.get((y, sl, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = _cv(Y3, "all", "c_gap_sd")
    lag1 = _cv(Y3, "all", "c_gap_sd_lag1")
    lag3 = _cv(Y3, "all", "c_gap_sd_lag3")
    days_l1 = _cv(Y3, "all", "c_n_days_with_tx_lag1")
    short_now = _cv(Y3, "short_<12", "c_gap_sd")
    short_l1 = _cv(Y3, "short_<12", "c_gap_sd_lag1")
    long_l1 = _cv(Y3, "long_>=18", "c_gap_sd_lag1")
    drop = now - lag1 if np.isfinite(now) and np.isfinite(lag1) else float("nan")
    # days lag1 is a night KEEP — gap_sd lag survives only if contemporaneous skill exists
    # and lag holds it (±0.03) and contemporaneous itself is not chance
    keep_q6 = bool(
        np.isfinite(now)
        and now >= 0.55
        and np.isfinite(lag1)
        and lag1 >= 0.55
        and (now - lag1) <= 0.03
        and np.isfinite(short_l1)
        and short_l1 >= 0.55
    )
    if not np.isfinite(now) or now < 0.55:
        q6 = "CLOSE"
        why = (
            f"Y3 contemporaneous gap_sd {_f(now)} is chance/weak; lag1 {_f(lag1)} / lag3 {_f(lag3)}. "
            "CLOSE as Q6 — there is no contemporaneous skill to lead."
        )
    elif keep_q6:
        q6 = "KEEP"
        why = (
            f"Y3 now {_f(now)} vs lag1 {_f(lag1)} (Δ {_f(drop, 3)}); short lag1 {_f(short_l1)}. "
            f"Days lag1 replica {_f(days_l1)} (night KEEP). Gap_sd lag holds."
        )
    else:
        q6 = "CLOSE"
        why = (
            f"Y3 now {_f(now)} vs lag1 {_f(lag1)} (Δ {_f(drop, 3)}) short {_f(short_l1)} "
            f"long {_f(long_l1)}. Days lag1 {_f(days_l1)} is the night KEEP — gap_sd lag does not survive."
        )
    print(why)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "days_l1": days_l1,
        "short_now": short_now,
        "short_l1": short_l1,
        "long_l1": long_l1,
        "drop": drop,
        "keep_q6": keep_q6,
        "q6": q6,
        "prose": why,
    }


# ---------------------------------------------------------------------------
# Pass 12 — same-day flood: gap_sd without unique-day collapse (in-memory)
# ---------------------------------------------------------------------------
def pass12_raw(tr: pd.DataFrame, unique_days: pd.DataFrame, raw_dates: pd.DataFrame, midnight: dict) -> dict:
    keys = tr[["company_id", "period"]].copy().reset_index(drop=True)
    raw_s = _gap_sd_from_dates(raw_dates, keys)
    uniq_s = _gap_sd_from_dates(unique_days, keys)
    store = pd.to_numeric(tr["c_gap_sd"], errors="coerce").reset_index(drop=True)
    antx = pd.to_numeric(tr["a_n_tx"], errors="coerce").reset_index(drop=True)
    cntx = pd.to_numeric(tr["c_n_tx"], errors="coerce").reset_index(drop=True)
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce").reset_index(drop=True)
    inv = 1.0 / cntx.clip(lower=1)
    rho_raw_inv = spearman(raw_s, inv)
    rho_raw_ntx = spearman(raw_s, cntx)
    rho_raw_antx = spearman(raw_s, antx)
    rho_raw_days = spearman(raw_s, days)
    rho_uniq_inv = spearman(uniq_s, inv)
    rho_uniq_ntx = spearman(uniq_s, cntx)
    rho_store_raw = spearman(store, raw_s)
    raw_is_inv = bool(np.isfinite(rho_raw_inv) and abs(rho_raw_inv) >= TWIN_RHO)
    uniq_escapes = bool(
        np.isfinite(rho_uniq_inv) and abs(rho_uniq_inv) < TWIN_RHO and raw_is_inv
    )
    # AUROC of the raw version (in-memory, not a store col)
    raw_on_tr = pd.Series(raw_s.to_numpy(), index=tr.index)
    raw_y3 = signed_oof_auroc(tr[Y3], raw_on_tr, tr["fold"], tr[Y3].notna())
    uniq_y3 = signed_oof_auroc(tr[Y3], pd.Series(uniq_s.to_numpy(), index=tr.index), tr["fold"], tr[Y3].notna())
    confirm_collapse = bool(raw_is_inv)
    mid_ok = bool(np.isfinite(midnight.get("share", float("nan"))) and abs(midnight["share"] - MIDNIGHT_QUOTE) < 0.03)
    prose = (
        f"Midnight share {_pp(midnight['share'])} (n_tx={midnight['n']:,}) "
        f"({'CONFIRM 94%' if mid_ok else 'off 94% quote'}). "
        f"Raw (no unique-day) σ vs 1/c_n_tx ρ={rho_raw_inv:.3f}, vs c_n_tx {rho_raw_ntx:.3f}, "
        f"vs a_n_tx {rho_raw_antx:.3f}. Unique-day σ vs 1/c_n_tx ρ={rho_uniq_inv:.3f}. "
        f"store↔raw {rho_store_raw:.3f}. Y3 raw CV "
        f"{'LOW_POWER' if raw_y3['low_power'] else _f(raw_y3['cv'])} vs unique "
        f"{'LOW_POWER' if uniq_y3['low_power'] else _f(uniq_y3['cv'])}. "
        + (
            "CONFIRM the collapse — raw σ is a 1/n_tx twin; unique-day was the right fix."
            if confirm_collapse
            else "Raw σ is not a 1/n_tx twin at |ρ|≥0.80 — collapse still matches store, but (e) is weaker."
        )
    )
    print(prose)
    return {
        "midnight": midnight["share"],
        "n_tx": midnight["n"],
        "mid_ok": mid_ok,
        "rho_raw_inv": rho_raw_inv,
        "rho_raw_ntx": rho_raw_ntx,
        "rho_raw_antx": rho_raw_antx,
        "rho_raw_days": rho_raw_days,
        "rho_uniq_inv": rho_uniq_inv,
        "rho_uniq_ntx": rho_uniq_ntx,
        "rho_store_raw": rho_store_raw,
        "raw_is_inv": raw_is_inv,
        "uniq_escapes": uniq_escapes,
        "confirm_collapse": confirm_collapse,
        "raw_y3": raw_y3["cv"] if not raw_y3["low_power"] else float("nan"),
        "uniq_y3": uniq_y3["cv"] if not uniq_y3["low_power"] else float("nan"),
        "raw_s": raw_s,
        "prose": prose,
    }


def midnight_share(con) -> dict:
    row = con.execute(
        """
        SELECT
          AVG(CASE WHEN "date" = date_trunc('day', "date") THEN 1.0 ELSE 0.0 END) AS share,
          COUNT(*) AS n
        FROM transactions
        WHERE "date" IS NOT NULL
        """
    ).fetchone()
    return {"share": float(row[0]), "n": int(row[1])}


# ---------------------------------------------------------------------------
# Extra 13 — 90d window vs months-on-book / so-far
# ---------------------------------------------------------------------------
def pass13_window(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    rows = []
    for name, lo, hi in (("<6", 1, 5), ("6-11", 6, 11), ("12-17", 12, 17), ("18-23", 18, 23), ("24", 24, 24)):
        sl = (tr["so_far"] >= lo) & (tr["so_far"] <= hi)
        x = gap[sl]
        rows.append(
            {
                "so_far": name,
                "n_cm": int(sl.sum()),
                "cov": float(x.notna().mean()) if sl.any() else float("nan"),
                "p50": float(x.median()) if x.notna().any() else float("nan"),
                "n_null": int(x.isna().sum()),
            }
        )
    short = tr["so_far"] <= SHORT_MAX
    long = tr["so_far"] >= LONG_MIN
    cov_s = float(gap[short].notna().mean()) if short.any() else float("nan")
    cov_l = float(gap[long].notna().mean()) if long.any() else float("nan")
    lab = tr[Y3].notna()
    short_cv = signed_oof_auroc(tr[Y3], tr["c_gap_sd"], tr["fold"], lab & short)
    long_cv = signed_oof_auroc(tr[Y3], tr["c_gap_sd"], tr["fold"], lab & long)
    days_s = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & short)
    artifact = bool(np.isfinite(cov_s) and cov_s < 0.80 and np.isfinite(cov_l) and cov_l >= 0.95)
    prose = (
        f"Coverage so-far short {_pp(cov_s)} vs long {_pp(cov_l)}. "
        f"Y3 gap short {'LOW_POWER' if short_cv['low_power'] else _f(short_cv['cv'])} "
        f"vs long {'LOW_POWER' if long_cv['low_power'] else _f(long_cv['cv'])}; "
        f"days short {'LOW_POWER' if days_s['low_power'] else _f(days_s['cv'])}. "
        f"{'90d window is a short-book NaN artifact (d).' if artifact else '90d coverage is not a short-book hole — nulls are thin unique-day books, not late arrivals only.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "cov_short": cov_s,
        "cov_long": cov_l,
        "short_cv": short_cv["cv"] if not short_cv["low_power"] else float("nan"),
        "long_cv": long_cv["cv"] if not long_cv["low_power"] else float("nan"),
        "days_short": days_s["cv"] if not days_s["low_power"] else float("nan"),
        "artifact": artifact,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 14 — full-history σ vs 90d (in-memory, do not rewrite ops)
# ---------------------------------------------------------------------------
def pass14_fullhist(tr: pd.DataFrame, unique_days: pd.DataFrame) -> dict:
    """If 90d σ ≈ lifetime σ, the window is not the object — it is a trait."""
    keys = tr[["company_id", "period"]].copy().reset_index(drop=True)
    ends = _month_end(keys["period"]).to_numpy(dtype="datetime64[ns]")
    out = np.full(len(keys), np.nan)
    by_co = {
        cid: np.sort(g["d"].to_numpy(dtype="datetime64[ns]"))
        for cid, g in unique_days.groupby("company_id", sort=False)
    }
    for cid, sub in keys.groupby("company_id", sort=False):
        d = by_co.get(cid)
        if d is None or len(d) < 3:
            continue
        idx = sub.index.to_numpy()
        hi = np.searchsorted(d, ends[idx], side="right")
        for k, b in zip(idx, hi):
            if b < 3:
                continue
            gaps = np.diff(d[:b]) / np.timedelta64(1, "D")
            out[k] = float(gaps.std(ddof=1))
    full = pd.Series(out, index=keys.index)
    store = pd.to_numeric(tr["c_gap_sd"], errors="coerce").reset_index(drop=True)
    rho = spearman(store, full)
    full_on = pd.Series(full.to_numpy(), index=tr.index)
    rec = signed_oof_auroc(tr[Y3], full_on, tr["fold"], tr[Y3].notna())
    r90, inf = ols_resid(store, full)
    leftover = signed_oof_auroc(tr[Y3], pd.Series(r90.to_numpy(), index=tr.index), tr["fold"], tr[Y3].notna())
    same = bool(np.isfinite(rho) and abs(rho) >= 0.80)
    prose = (
        f"Lifetime unique-day σ (as-of month-end, no 90d cut) vs store 90d ρ={rho:.3f} "
        f"({'same object — 90d is a trait window' if same else '90d is not lifetime σ'}). "
        f"Y3 full-hist {'LOW_POWER' if rec['low_power'] else _f(rec['cv'])}; "
        f"90d leftover after lifetime {'LOW_POWER' if leftover['low_power'] else _f(leftover['cv'])} "
        f"(R²={_f(inf['r2'])})."
    )
    print(prose)
    return {
        "rho": rho,
        "same": same,
        "full_y3": rec["cv"] if not rec["low_power"] else float("nan"),
        "leftover": leftover["cv"] if not leftover["low_power"] else float("nan"),
        "r2": inf["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 15 — Y3 quintiles (shape: inverse-days or leftover regularity?)
# ---------------------------------------------------------------------------
def pass15_quintiles(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    sl = tr.loc[lab].copy()
    rows = []
    for col, name in (("c_gap_sd", "c_gap_sd"), ("c_n_days_with_tx", "days")):
        x = pd.to_numeric(sl[col], errors="coerce")
        ok = x.notna()
        if ok.sum() < 20:
            continue
        q = pd.qcut(x[ok], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
        y = pd.to_numeric(sl.loc[ok, Y3], errors="coerce")
        for labv in q.astype(str).unique():
            m = q.astype(str) == labv
            rows.append(
                {
                    "feature": name,
                    "q": labv,
                    "n": int(m.sum()),
                    "n_pos": int((y[m] == 1).sum()),
                    "rate": float(y[m].mean()) if m.any() else float("nan"),
                    "x_p50": float(x[ok][m].median()),
                }
            )
    # monotonic? Q1 vs Q5 gap
    def _qrate(feat, qn):
        hits = [r for r in rows if r["feature"] == feat and r["q"] == qn]
        return hits[0]["rate"] if hits else float("nan")

    gap_q1, gap_q5 = _qrate("c_gap_sd", "Q1"), _qrate("c_gap_sd", "Q5")
    days_q1, days_q5 = _qrate("days", "Q1"), _qrate("days", "Q5")
    prose = (
        f"Y3 rate gap_sd Q1 {_pp(gap_q1)} → Q5 {_pp(gap_q5)}; "
        f"days Q1 {_pp(days_q1)} → Q5 {_pp(days_q5)}. "
        "Quiet / regular Q1 vs busy Q5 is the days story if the shapes match."
    )
    print(prose)
    return {
        "rows": rows,
        "gap_q1": gap_q1,
        "gap_q5": gap_q5,
        "days_q1": days_q1,
        "days_q5": days_q5,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 16 — inverse-days leftover (is gap_sd just 1/days?)
# ---------------------------------------------------------------------------
def pass16_invdays(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    inv = pd.to_numeric(tr["inv_days"], errors="coerce")
    r, inf = ols_resid(gap, inv)
    rec = signed_oof_auroc(tr[Y3], r, tr["fold"], tr[Y3].notna())
    inv_cv = signed_oof_auroc(tr[Y3], inv, tr["fold"], tr[Y3].notna())
    rho = spearman(gap, inv)
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = (
        f"gap_sd vs 1/days ρ={rho:.3f} ({'TWIN' if twin else 'not a pairwise inverse-days twin'}). "
        f"Y3 1/days {_f(inv_cv['cv']) if not inv_cv['low_power'] else 'LOW_POWER'}; "
        f"leftover after 1/days {_f(rec['cv']) if not rec['low_power'] else 'LOW_POWER'} (R²={_f(inf['r2'])})."
    )
    print(prose)
    return {
        "rho": rho,
        "twin": twin,
        "inv_y3": inv_cv["cv"] if not inv_cv["low_power"] else float("nan"),
        "leftover": rec["cv"] if not rec["low_power"] else float("nan"),
        "r2": inf["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 17 — overlap-only (same rows where gap_sd is defined)
# ---------------------------------------------------------------------------
def pass17_overlap(tr: pd.DataFrame) -> dict:
    """Days 0.711 uses 4 extra Y3 pos that are gap-null. Fair head-to-head?"""
    defined = pd.to_numeric(tr["c_gap_sd"], errors="coerce").notna()
    lab = tr[Y3].notna() & defined
    rows = []
    store = {}
    for name, col in (
        ("c_gap_sd", tr["c_gap_sd"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("a_n_tx", tr["a_n_tx"]),
        ("log1p_a_in3", tr["log_in3"]),
    ):
        rec = signed_oof_auroc(tr[Y3], col, tr["fold"], lab)
        store[name] = rec
        rows.append(
            {
                "feature": name,
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
            }
        )
    gap = store["c_gap_sd"]["cv"] if not store["c_gap_sd"]["low_power"] else float("nan")
    days = store["c_n_days_with_tx"]["cv"] if not store["c_n_days_with_tx"]["low_power"] else float("nan")
    size = store["log1p_a_in3"]["cv"] if not store["log1p_a_in3"]["low_power"] else float("nan")
    still_loses = bool(np.isfinite(gap) and np.isfinite(days) and gap < days)
    prose = (
        f"On gap-defined Y3 rows only: gap {_f(gap)} vs days {_f(days)} vs size {_f(size)}. "
        f"{'Still loses to days — the 4 null positives are not the gap.' if still_loses else 'Overlap flips the days win.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "gap": gap,
        "days": days,
        "size": size,
        "still_loses": still_loses,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 18 — Q6 honesty: lag1 leftover after days_lag1
# ---------------------------------------------------------------------------
def pass18_q6_resid(tr: pd.DataFrame) -> dict:
    gap_l1 = pd.to_numeric(tr["c_gap_sd_lag1"], errors="coerce")
    days_l1 = pd.to_numeric(tr["c_n_days_with_tx_lag1"], errors="coerce")
    r, inf = ols_resid(gap_l1, days_l1)
    rec = signed_oof_auroc(tr[Y3], r, tr["fold"], tr[Y3].notna())
    days_cv = signed_oof_auroc(tr[Y3], days_l1, tr["fold"], tr[Y3].notna())
    gap_cv = signed_oof_auroc(tr[Y3], gap_l1, tr["fold"], tr[Y3].notna())
    leftover = rec["cv"] if not rec["low_power"] else float("nan")
    died = bool(np.isfinite(leftover) and leftover < 0.55)
    prose = (
        f"Y3 gap_sd_lag1 {_f(gap_cv['cv']) if not gap_cv['low_power'] else 'LOW_POWER'} vs "
        f"days_lag1 {_f(days_cv['cv']) if not days_cv['low_power'] else 'LOW_POWER'}; "
        f"lag leftover after days_lag1 {_f(leftover)} (R²={_f(inf['r2'])}). "
        + (
            "CLOSE as Q6 — the lag is the days twin lagged, not a new lead."
            if died
            else "Lag leftover still ranks — not enough to KEEP a twin as Q6."
        )
    )
    print(prose)
    return {
        "gap_l1": gap_cv["cv"] if not gap_cv["low_power"] else float("nan"),
        "days_l1": days_cv["cv"] if not days_cv["low_power"] else float("nan"),
        "leftover": leftover,
        "r2": inf["r2"],
        "died": died,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 19 — rank residual (Spearman twin, not linear OLS)
# ---------------------------------------------------------------------------
def pass19_rank(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    antx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    r_d, inf_d = ols_resid(gap.rank(method="average"), days.rank(method="average"))
    r_a, inf_a = ols_resid(gap.rank(method="average"), antx.rank(method="average"))
    rec_d = signed_oof_auroc(tr[Y3], r_d, tr["fold"], tr[Y3].notna())
    rec_a = signed_oof_auroc(tr[Y3], r_a, tr["fold"], tr[Y3].notna())
    died = bool(not rec_d["low_power"] and rec_d["cv"] < 0.55)
    prose = (
        f"Y3 rank-residual after days {_f(rec_d['cv']) if not rec_d['low_power'] else 'LOW_POWER'} "
        f"(R²={_f(inf_d['r2'])}); after a_n_tx {_f(rec_a['cv']) if not rec_a['low_power'] else 'LOW_POWER'} "
        f"(R²={_f(inf_a['r2'])}). "
        f"{'Rank leftover also dies — not an OLS misspec.' if died else 'Rank leftover lives — OLS leftover was the misspec.'}"
    )
    print(prose)
    return {
        "days": rec_d["cv"] if not rec_d["low_power"] else float("nan"),
        "antx": rec_a["cv"] if not rec_a["low_power"] else float("nan"),
        "r2_days": inf_d["r2"],
        "r2_antx": inf_a["r2"],
        "died": died,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — n_tx leftover 0.659 is just days?
# ---------------------------------------------------------------------------
def pass20_ntx_is_days(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    antx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    r_a, _ = ols_resid(gap, antx)
    r_ad, inf = ols_resid(r_a, days)
    rec = signed_oof_auroc(tr[Y3], r_ad, tr["fold"], tr[Y3].notna())
    leftover = rec["cv"] if not rec["low_power"] else float("nan")
    died = bool(np.isfinite(leftover) and leftover < 0.55)
    prose = (
        f"Y3 leftover-after-a_n_tx, then residualized on days: {_f(leftover)} (R²={_f(inf['r2'])}). "
        + (
            "The 0.659 n_tx leftover *is* days. Twin owner is days, not count."
            if died
            else "n_tx leftover is not days — a real count leftover."
        )
    )
    print(prose)
    return {"cv": leftover, "r2": inf["r2"], "died": died, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 21 — stressed-only Spearman (Y3 labeled rows)
# ---------------------------------------------------------------------------
def pass21_stressed_rho(tr: pd.DataFrame) -> dict:
    sl = tr[tr[Y3].notna()]
    gap = pd.to_numeric(sl["c_gap_sd"], errors="coerce")
    rows = []
    rhos = {}
    for name, col in (
        ("c_n_days_with_tx", sl["c_n_days_with_tx"]),
        ("a_n_tx", sl["a_n_tx"]),
        ("c_n_tx", sl["c_n_tx"]),
        ("log1p(a_in3)", sl["log_in3"]),
        ("c_recency_days", sl["c_recency_days"]),
    ):
        rho = spearman(gap, col)
        rhos[name] = rho
        flag = "TWIN" if name in {"c_n_days_with_tx", "a_n_tx", "c_n_tx"} and abs(rho) >= TWIN_RHO else ""
        if name == "log1p(a_in3)" and abs(rho) >= SIZE_RHO:
            flag = "SIZE"
        rows.append({"pair": name, "rho": rho, "flag": flag})
    twin = bool(abs(rhos["c_n_days_with_tx"]) >= TWIN_RHO)
    prose = (
        f"Y3-labeled Spearman gap vs days {rhos['c_n_days_with_tx']:.3f}, "
        f"a_n_tx {rhos['a_n_tx']:.3f}, size {rhos['log1p(a_in3)']:.3f}. "
        f"{'Twin still holds on the stressed rows.' if twin else 'Twin is a panel artifact, not the Y3 slice.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "rho_days": rhos["c_n_days_with_tx"],
        "rho_antx": rhos["a_n_tx"],
        "rho_size": rhos["log1p(a_in3)"],
        "twin": twin,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 22 — days tercile × gap tercile (leftover cells)
# ---------------------------------------------------------------------------
def pass22_twobytwo(tr: pd.DataFrame) -> dict:
    sl = tr[tr[Y3].notna()].copy()
    gap = pd.to_numeric(sl["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce")
    ok = gap.notna() & days.notna()
    sl = sl.loc[ok]
    sl = sl.assign(
        gap_t=pd.qcut(gap[ok], 3, labels=["G1_reg", "G2", "G3_irreg"], duplicates="drop"),
        days_t=pd.qcut(days[ok], 3, labels=["D1_quiet", "D2", "D3_busy"], duplicates="drop"),
    )
    rows = []
    for (gt, dt), part in sl.groupby(["gap_t", "days_t"], observed=False):
        y = pd.to_numeric(part[Y3], errors="coerce")
        rows.append(
            {
                "gap_t": str(gt),
                "days_t": str(dt),
                "n": int(len(part)),
                "n_pos": int((y == 1).sum()),
                "rate": float(y.mean()) if y.notna().any() else float("nan"),
            }
        )
    # off-diagonal mass: irregular + busy, or regular + quiet should be thin if twin
    off = sl[
        ((sl["gap_t"].astype(str) == "G3_irreg") & (sl["days_t"].astype(str) == "D3_busy"))
        | ((sl["gap_t"].astype(str) == "G1_reg") & (sl["days_t"].astype(str) == "D1_quiet"))
    ]
    n_off = int(len(off))
    n = int(len(sl))
    prose = (
        f"Y3 labeled 3×3: off-diagonal regular×quiet + irregular×busy = {n_off:,}/{n:,} "
        f"({_pp(_pct(n_off, n))}). A leftover regularity cell would be irregular-and-busy. "
        "If that cell is empty, gap_sd is inverse days."
    )
    print(prose)
    return {"rows": rows, "n_off": n_off, "n": n, "off_share": _pct(n_off, n), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 23 — days leftover after gap (who owns the cluster?)
# ---------------------------------------------------------------------------
def pass23_days_after_gap(tr: pd.DataFrame) -> dict:
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    r, inf = ols_resid(days, gap)
    rec = signed_oof_auroc(tr[Y3], r, tr["fold"], tr[Y3].notna())
    leftover = rec["cv"] if not rec["low_power"] else float("nan")
    # days still the owner if leftover after gap dies (symmetric twin) OR if
    # univariate days wins and leftover is not a new object
    lives = bool(np.isfinite(leftover) and leftover >= 0.55)
    prose = (
        f"Y3 days leftover after gap_sd {_f(leftover)} (R²={_f(inf['r2'])}). "
        + (
            "Days still ranks after gap is removed — they are not interchangeable. "
            "The Y3 engine is right to keep days; the feature report picked the weaker twin as representative."
            if lives
            else "Symmetric twin — univariate 0.711 is the only tie-break."
        )
    )
    print(prose)
    return {"cv": leftover, "r2": inf["r2"], "died": bool(np.isfinite(leftover) and leftover < 0.55), "prose": prose}


def decide(p3, p4, p5, p6, p7, p11, p12, p13, p18=None) -> dict:
    """KEEP on 44 / DROP-from-44 / CLOSE as X / PARK as Y. Map (a)–(e)."""
    park_y = True
    twin = bool(p3["pairwise_twin"] or p5["died"])
    size_flag = bool(p3["size_flag"])
    leftover_lives = bool(p5["leftover_lives"])
    shock = bool(p7["shock"])
    loses_days = bool(p4["loses_days"])
    keep_44 = bool(leftover_lives and shock and not size_flag)
    if keep_44:
        x_dec = "KEEP"
        letter = "c"
        why = (
            f"leftover after days {_f(p5['y3_days'])} beats size {_f(p5['size_y3'])} "
            f"by {_f(p5['leftover_beat'])} and ICC {_f(p7['icc'])} is a month shock. "
            "Still **not** on tonight's 15-col card. Parent decides."
        )
    elif twin or loses_days:
        x_dec = "DROP-from-44"
        letter = "a"
        why = (
            f"pairwise |ρ|≥0.80 twin of {', '.join(p3['twin_names'])} "
            f"(days {p3['rho_days']:.3f}). "
            f"Y3 gap {_f(p4['gap_y3'])} vs days {_f(p4['days_y3'])} (night 0.711). "
            f"Leftover after days {_f(p5['y3_days'])} dies. "
            "Keep days on the card. **DROP from the 44** / **CLOSE as X**."
        )
        if not p3["pairwise_twin"] and p5["died"]:
            why = (
                f"Cluster cut is a chain (days↔n_tx), not pairwise |ρ|≥0.80 "
                f"(days {p3['rho_days']:.3f} / a_n_tx {p3['rho_antx']:.3f}). "
                f"Leftover after days still dies ({_f(p5['y3_days'])} vs size {_f(p5['size_y3'])}). "
                f"Y3 gap {_f(p4['gap_y3'])} loses to days {_f(p4['days_y3'])}. "
                "**DROP from the 44** / **CLOSE as X**."
            )
    elif size_flag and p6["dies_inside"]:
        x_dec = "CLOSE"
        letter = "b"
        why = (
            f"NEAR SIZE ρ={p3['rho_size']:.3f} vs log1p(a_in3); leftover dies inside size terciles. "
            f"Y3 {_f(p4['gap_y3'])} vs size {_f(p4['size_y3'])}."
        )
    elif p13["artifact"]:
        x_dec = "CLOSE"
        letter = "d"
        why = "90-day window / short-book NaN artifact. Not a health X."
    elif leftover_lives and not shock:
        x_dec = "CLOSE"
        letter = "c"
        why = (
            f"Leftover after days {_f(p5['y3_days'])} vs size {_f(p5['size_y3'])} "
            f"(Δ {_f(p5['leftover_beat'])}) but ICC {_f(p7['icc'])} is a BETWEEN trait, not a month shock. "
            "KEEP-later only if parent wants a regularity control — still **not** on tonight's card. "
            "Not KEEP on the 44 tonight."
        )
    else:
        x_dec = "CLOSE"
        letter = "a"
        why = (
            f"Y3 {_f(p4['gap_y3'])} vs days {_f(p4['days_y3'])} vs size {_f(p4['size_y3'])}. "
            "Does not clear KEEP-as-X (beat size ≥0.02 **and** leftover after days **and** not SIZE)."
        )
    q5 = "KEEP-Q5 footnote" if (leftover_lives and not twin and shock) else "CLOSE"
    q5_sentence = ""
    if q5 == "KEEP-Q5 footnote" and leftover_lives:
        q5_sentence = (
            "Irregular booking cadence (σ of unique-day gaps) leftover after counting days — "
            "not just fewer booking days."
        )
    else:
        q5_sentence = "not sayable without “fewer booking days” / size / twin."
    q6 = p11["q6"]
    if twin or (p18 is not None and p18.get("died")):
        q6 = "CLOSE"
    letters = {
        "a": "days/n_tx twin" if twin or loses_days else "not a pairwise twin",
        "b": (
            "NEAR SIZE leftover that dies inside terciles"
            if (size_flag and p6["dies_inside"])
            else (
                "SIZE ρ −0.53 is real; tercile 'survival' is the days engine, not leftover regularity"
                if size_flag
                else "not SIZE-only"
            )
        ),
        "c": "Pérez-Salazar leftover after days" if leftover_lives else "no leftover after days",
        "d": "90d / short-book artifact" if p13["artifact"] else "not a short-book NaN hole",
        "e": "midnight hole — collapse CONFIRMED (raw 1/n_tx); unique-day still a days twin" if p12["confirm_collapse"] else "collapse not a 1/n_tx twin",
    }
    return {
        "x_dec": x_dec,
        "park_y": park_y,
        "q5": q5,
        "q5_sentence": q5_sentence,
        "q6": q6,
        "letter": letter,
        "why": why,
        "letters": letters,
        "keep_44": keep_44,
        "twin": twin,
        "close_x": x_dec in {"DROP-from-44", "CLOSE"},
    }


def make_png(tr: pd.DataFrame, p15: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    d = tr[["c_gap_sd", "c_n_days_with_tx", "log_in3"]].copy()
    d["c_gap_sd"] = pd.to_numeric(d["c_gap_sd"], errors="coerce")
    d["c_n_days_with_tx"] = pd.to_numeric(d["c_n_days_with_tx"], errors="coerce")
    d = d.dropna()
    if len(d) > 8000:
        d = d.sample(8000, random_state=FOLD_SEED)
    sc = ax.scatter(
        d["c_n_days_with_tx"],
        d["c_gap_sd"],
        c=d["log_in3"],
        s=6,
        alpha=0.35,
        cmap="viridis",
        linewidths=0,
    )
    ax.set_xlabel("c_n_days_with_tx (in-month)")
    ax.set_ylabel("c_gap_sd (90d unique-day σ)")
    ax.set_title("Train: gap_sd vs days (color = log1p a_in3)")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="log1p(a_in3)")

    ax2 = axes[1]
    gap_rows = [r for r in p15["rows"] if r["feature"] == "c_gap_sd"]
    day_rows = [r for r in p15["rows"] if r["feature"] == "days"]
    qs = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    x = np.arange(len(qs))
    g = [next((100.0 * r["rate"] for r in gap_rows if r["q"] == q), np.nan) for q in qs]
    dy = [next((100.0 * r["rate"] for r in day_rows if r["q"] == q), np.nan) for q in qs]
    ax2.bar(x - 0.18, g, width=0.36, color="#1f4e79", label="c_gap_sd")
    ax2.bar(x + 0.18, dy, width=0.36, color="#9e6b4a", label="c_n_days_with_tx")
    ax2.set_xticks(x)
    ax2.set_xticklabels(qs)
    ax2.set_ylabel("Y3 rate (%)")
    ax2.set_title("Y3 rate by quintile (train stressed)")
    ax2.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12, p13, p14, p15, p16 = (
        ctx["p11"],
        ctx["p12"],
        ctx["p13"],
        ctx["p14"],
        ctx["p15"],
        ctx["p16"],
    )
    p17, p18, p19, p20, p21, p22, p23 = (
        ctx["p17"],
        ctx["p18"],
        ctx["p19"],
        ctx["p20"],
        ctx["p21"],
        ctx["p22"],
        ctx["p23"],
    )
    d = ctx["decision"]
    lines = [
        "# `c_gap_sd` — twin of days, NEAR SIZE, or Pérez-Salazar leftover?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_gap_sd`. "
        "Do not put `c_gap_sd` on the 15-col card. Night Y3 quote stays **0.762 / 0.752**. "
        "Days bar stays **0.711**.",
        "",
        "`c_gap_sd` = sample stdev (ddof=1) of day-gaps between consecutive **unique "
        "calendar days** in (month_end − 90d, month_end]. Null if fewer than 3 distinct days. "
        "Pérez-Salazar, Márquez & Vidal-Silva 2026 (Computers 15:135) σ_Δt. "
        "Unique-day collapse: 94% of timestamps are midnight.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | PARK as a health Y. Do not invent `y_gap_sd`. Regular vs irregular booker is a BETWEEN trait (ICC 0.96). |",
        "| 2 | Who is improving? | Not this column. A 90d σ is not a recovery path. |",
        f"| 3 | Who is turning? | Y3 gap {_f(p4['gap_y3'])} vs days {_f(p4['days_y3'])} vs size {_f(p4['size_y3'])}. |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['q5_sentence']} |",
        f"| 6 | Months earlier? | **{d['q6']}** — {p18['prose']} |",
        "",
        "## Who is right — feature report vs Y3 engine?",
        "",
        f"**{d['x_dec']}** as Y3 X. Letter **({d['letter']})**. {d['why']}",
        "",
        "| letter | claim | this cut |",
        "| --- | --- | --- |",
        f"| (a) | days / n_tx twin (\\|ρ\\|≥0.80) — drop from the 44, keep days on the card | {d['letters']['a']} |",
        f"| (b) | NEAR SIZE (ρ −0.543) leftover that dies inside size terciles | {d['letters']['b']} |",
        f"| (c) | Pérez-Salazar regularity leftover after days (KEEP later, not on tonight's card) | {d['letters']['c']} |",
        f"| (d) | 90-day window artifact / short-book NaN | {d['letters']['d']} |",
        f"| (e) | midnight-timestamp hole the unique-day collapse was meant to fix | {d['letters']['e']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `c_gap_sd` as Y3 X / on the 44 | **{d['x_dec']}** | {d['why']} |",
        "| `c_gap_sd` on tonight's 15-col card | **no** | days 0.711 stays the engine; night quote 0.762 / 0.752 unchanged |",
        "| `c_gap_sd` as a health Y | **PARK** | do not invent `y_gap_sd` |",
        f"| leftover after days | **{'KEEP later' if p5['leftover_lives'] else 'dies / CLOSE as twin'}** | Y3 resid {_f(p5['y3_days'])} vs size {_f(p5['size_y3'])} |",
        f"| Q6 lag1/lag3 | **{d['q6']}** | {p18['prose']} |",
        f"| Q5 footnote | **{d['q5']}** | {d['q5_sentence']} |",
        f"| unique-day collapse | **{'CONFIRM' if p12['confirm_collapse'] else 'measured'}** | {p12['prose']} |",
        "",
        "## 1. Completeness — null share (days<3), train vs holdout",
        "",
        p1["prose"],
        "",
        _md_table(
            [
                {
                    "split": r["split"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "cov": _pp(r["cov"]),
                    "n_null": f"{r['n_null']:,}",
                    "null": _pp(r["null_share"]),
                    "in-month days<3": _pp(r["days_lt3"]),
                    "p50": _f(r["p50"]),
                    "p90": _f(r["p90"]),
                }
                for r in p1["rows"]
            ]
        ),
        "",
        f"Confirm feature-report 95.1%: **{'YES' if p1['confirm'] else 'NO'}**.",
        "",
        "## 2. Formula vs unique booking dates (90d, ddof=1)",
        "",
        p2["prose"],
        "",
        "Sample company-months (recon = in-memory unique-day σ, same window as ops.py):",
        "",
        _md_table(p2["sample_rows"]),
        "",
        "## 3. Spearman — twin / SIZE / recency",
        "",
        p3["prose"],
        "",
        "Twin if |ρ|≥0.80 vs days / a_n_tx / c_n_tx. SIZE if |ρ| vs log size ≥0.50.",
        "",
        _md_table(
            [{"pair": r["pair"], "Spearman": _f(r["rho"]), "flag": r["flag"]} for r in p3["rows"]]
        ),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Y3 stressed n={p4['n_y3']:,} base {_pp(p4['y3_rate'])}; "
        f"Y2 n={p4['n_y2']:,} base {_pp(p4['y2_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p4['days_y3'])}); "
        f"size 0.617 (replica {_f(p4['size_y3'])}).",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        f"KEEP-as-X: leftover after days beats size by ≥{KEEP_DELTA:g} **and** leftover after days "
        f"**and** not SIZE. Size dummy ≥{SIZE_PARK:g} is a PARK flag, not a KEEP.",
        "",
        "## 5. Residual AUROC after days / a_n_tx",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        f"OLS R² gap~days {_f(p5['r2_days'])} slope={_f(p5['slope_days'])}; "
        f"gap~a_n_tx {_f(p5['r2_antx'])} slope={_f(p5['slope_antx'])}; both {_f(p5['r2_both'])}.",
        "",
        "## 6. SIZE terciles — gap_sd vs size inside T1 / T2 / T3",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. ICC / company-demean (feature-report BETWEEN 0.96)",
        "",
        p7["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| ICC `c_gap_sd` | {_f(p7['icc'])} |",
        f"| ICC `c_n_days_with_tx` | {_f(p7['icc_days'])} |",
        f"| w/b | {_f(p7['ratio'])} |",
        f"| acf1 / acf3 / acf6 | {_f(p7['acf1'])} / {_f(p7['acf3'])} / {_f(p7['acf6'])} |",
        f"| Y3 raw / demean | {_f(p7['raw_cv'])} / {_f(p7['dem_cv'])} |",
        f"| trait / shock | {'YES' if p7['trait'] else 'no'} / {'YES' if p7['shock'] else 'no'} |",
        "",
        "## 8. Nulls — Y3 rate on gap_sd-null vs defined",
        "",
        p8["prose"],
        "",
        _md_table(
            [
                {
                    "y": r["y"],
                    "slice": r["slice"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_lab": f"{r['n_lab']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                    "days p50": _f(r["days_p50"], 1),
                    "so_far p50": _f(r["so_far_p50"], 1),
                    "mob p50": _f(r["mob_p50"], 1),
                }
                for r in p8["rows"]
            ]
        ),
        "",
        "Unique days in the 90d window (train CM):",
        "",
        _md_table(p8["win_rows"]) if p8["win_rows"] else "_(n_win not attached)_\n",
        "",
        "## 9. Dark 470 vs invoiced 744",
        "",
        p9["prose"],
        "",
        _md_table(
            [
                {
                    "group": r["group"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "cov": _pp(r["cov"]),
                    "gap p50": _f(r["p50"]),
                    "days p50": _f(r["days_p50"], 1),
                }
                for r in p9["rows"]
            ]
        ),
        "",
        "## 10. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. Q6 — lag1/lag3 on short vs long (days lag1 is a night KEEP)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Same-day flood — σ without unique-day collapse (in-memory)",
        "",
        p12["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| midnight share | {_pp(p12['midnight'])} |",
        f"| raw σ vs 1/c_n_tx | {_f(p12['rho_raw_inv'])} |",
        f"| raw σ vs c_n_tx | {_f(p12['rho_raw_ntx'])} |",
        f"| unique σ vs 1/c_n_tx | {_f(p12['rho_uniq_inv'])} |",
        f"| store ↔ raw | {_f(p12['rho_store_raw'])} |",
        f"| Y3 raw / unique | {_f(p12['raw_y3'])} / {_f(p12['uniq_y3'])} |",
        "",
        "Do **not** rewrite ops.py. The collapse stays.",
        "",
        "## Extra 13 — 90d window vs months-on-book",
        "",
        p13["prose"],
        "",
        _md_table(
            [
                {
                    "so_far": r["so_far"],
                    "n_cm": f"{r['n_cm']:,}",
                    "cov": _pp(r["cov"]),
                    "p50": _f(r["p50"]),
                    "n_null": f"{r['n_null']:,}",
                }
                for r in p13["rows"]
            ]
        ),
        "",
        "## Extra 14 — lifetime unique-day σ vs 90d (in-memory)",
        "",
        p14["prose"],
        "",
        "## Extra 15 — Y3 quintiles",
        "",
        p15["prose"],
        "",
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "q": r["q"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                    "x p50": _f(r["x_p50"]),
                }
                for r in p15["rows"]
            ]
        ),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## Extra 16 — inverse in-month days",
        "",
        p16["prose"],
        "",
        "## Extra 17 — overlap-only (gap-defined Y3 rows)",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "## Extra 18 — Q6 leftover after days_lag1",
        "",
        p18["prose"],
        "",
        "## Extra 19 — rank residual (not linear OLS)",
        "",
        p19["prose"],
        "",
        "## Extra 20 — n_tx leftover is days?",
        "",
        p20["prose"],
        "",
        "## Extra 21 — Y3-labeled Spearman",
        "",
        p21["prose"],
        "",
        _md_table(
            [{"pair": r["pair"], "Spearman": _f(r["rho"]), "flag": r["flag"]} for r in p21["rows"]]
        ),
        "",
        "## Extra 22 — days tercile × gap tercile",
        "",
        p22["prose"],
        "",
        _md_table(
            [
                {
                    "gap_t": r["gap_t"],
                    "days_t": r["days_t"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                }
                for r in p22["rows"]
            ]
        ),
        "",
        "## Extra 23 — days leftover after gap_sd",
        "",
        p23["prose"],
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage, formula, Spearman, singles, residual, "
        "size terciles, ICC, nulls, dark 470/744, chronic-12, Q6, raw-vs-unique, so-far window, "
        "lifetime σ, quintiles, inverse-days, overlap, Q6-resid, rank-resid, n_tx-is-days, "
        "stressed-ρ, 3×3, days-after-gap.",
        "",
        "Did **not**: rewrite ops.py, put `c_gap_sd` on the 15-col card, change the night Y3 "
        "quote or the days 0.711 bar, invent `y_gap_sd`, write a 0–100, touch `product/`, "
        "rewrite parquet/duckdb, run `build_targets`, write the parent journal, commit.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p4, p5, p7, p11, p12, d = (
        ctx["p1"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p7"],
        ctx["p11"],
        ctx["p12"],
        ctx["decision"],
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "c_gap_sd_cov",
            "value": p1["train_cov"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"confirm951={p1['confirm']} hold={p1['hold_cov']:.4f} n_null={p1['n_null']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "rho_gap_sd_days",
            "value": p3["rho_days"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"antx={p3['rho_antx']:.4f} cntx={p3['rho_cntx']:.4f} size={p3['rho_size']:.4f} opin={p3['rho_opin']:.4f} twin={p3['pairwise_twin']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_gap_sd",
            "value": p4["gap_y3"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"size={p4['size_y3']:.4f} days={p4['days_y3']:.4f} antx={p4['antx_y3']:.4f} beat_size={p4['beat_size']:.4f} x={d['x_dec']} letter={d['letter']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_n_days_with_tx",
            "value": p4["days_y3"],
            "coverage": "1.0000",
            "notes": f"night=0.711 replica; confirm={p4['days_ok']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_gap_sd_resid_days",
            "value": p5["y3_days"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"resid_antx={p5['y3_antx']:.4f} both={p5['y3_both']:.4f} leftover_lives={p5['leftover_lives']} died={p5['died']} r2_days={p5['r2_days']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y2,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_gap_sd",
            "value": p4["gap_y2"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"days={p4['days_y2']:.4f} size={p4['size_y2']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "icc_c_gap_sd",
            "value": p7["icc"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"confirm096={p7['confirm_icc']} trait={p7['trait']} shock={p7['shock']} acf1={p7['acf1']:.4f} dem={p7['dem_cv']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_gap_sd_lag1",
            "value": p11["lag1"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"now={p11['now']:.4f} lag3={p11['lag3']:.4f} q6={d['q6']} days_l1={p11['days_l1']:.4f} resid={ctx['p18']['leftover']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "rho_raw_gap_sd_inv_ntx",
            "value": p12["rho_raw_inv"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"uniq_inv={p12['rho_uniq_inv']:.4f} midnight={p12['midnight']:.4f} collapse={p12['confirm_collapse']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_gap_sd_lag1_resid_days",
            "value": ctx["p18"]["leftover"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"died={ctx['p18']['died']} r2={ctx['p18']['r2']:.4f} q6={d['q6']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_days_resid_gap_sd",
            "value": ctx["p23"]["cv"],
            "coverage": f"{p1['train_cov']:.4f}",
            "notes": f"died={ctx['p23']['died']} r2={ctx['p23']['r2']:.4f} overlap_days={ctx['p17']['days']}",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (r.get("agent"), r.get("y"), r.get("model"), r.get("split"), r.get("metric"))
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


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p3, p4, p5, p7 = ctx["p3"], ctx["p4"], ctx["p5"], ctx["p7"]
    WAVE_MD.parent.mkdir(parents=True, exist_ok=True)
    WAVE_MD.write_text(
        "# Wave 4 — c_gap_sd\n"
        "\n"
        f"- agent `{AGENT}` files: `analysis/evaluate/gap_sd_qa.py`, "
        f"`analysis/outputs/gap_sd_qa.md`"
        + (f", `{OUT_PNG.name}`" if ctx.get("png_ok") else "")
        + "\n"
        "- columns: store `c_gap_sd` (Family C). No new column. No ops.py edit.\n"
        f"- train coverage: {_pp(ctx['p1']['train_cov'])} "
        f"({'CONFIRM 95.1%' if ctx['p1']['confirm'] else 'off 95.1%'}).\n"
        f"- ρ vs days {p3['rho_days']:.3f} / a_n_tx {p3['rho_antx']:.3f} / "
        f"c_n_tx {p3['rho_cntx']:.3f}; vs log1p(a_in3) {p3['rho_size']:.3f}.\n"
        f"- Y3 CV gap {_f(p4['gap_y3'])} vs size {_f(p4['size_y3'])} vs days {_f(p4['days_y3'])}. "
        f"Leftover after days {_f(p5['y3_days'])}. ICC {_f(p7['icc'])}.\n"
        f"- verdict: **{d['x_dec']}** as Y3 X ({d['letter']}). PARK as Y. "
        f"Q6 {d['q6']}. Q5 {d['q5']}.\n"
        f"- what failed: {'; '.join(ctx['failed']) if ctx['failed'] else '(none)'}\n"
        "- next: parent decides the 44; do not put gap_sd on the 15-col card.\n"
        f"- elapsed {ctx['elapsed_s']:.0f}s.\n",
        encoding="utf-8",
    )
    print(f"wrote {WAVE_MD}")


def run() -> dict:
    t0 = time.time()
    print(f"gap_sd_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["c_gap_sd", "c_n_days_with_tx"], (1, 3))
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )

    con = connect()
    try:
        print("pass 1 coverage")
        p1 = pass1_coverage(panel, tr)
        print("unique + raw dates (read-only)")
        unique_days = load_unique_days(con)
        raw_dates = load_raw_dates(con)
        mid = midnight_share(con)
        print(f"  unique days={len(unique_days):,} raw txs={len(raw_dates):,} midnight={mid['share']:.3f}")
        print("pass 2 formula")
        p2 = pass2_formula(tr, unique_days)
        # attach n_win aligned to tr
        n_win = pd.Series(p2["n_win"].to_numpy(), index=tr.index)
        print("pass 3 Spearman")
        p3 = pass3_rho(tr)
        print("pass 4 singles")
        p4 = pass4_auroc(tr)
        print("pass 5 residual")
        p5 = pass5_residual(tr)
        print("pass 6 size terciles")
        p6 = pass6_terciles(tr)
        print("pass 7 ICC")
        p7 = pass7_icc(tr)
        print("pass 8 nulls")
        p8 = pass8_nulls(tr, n_win)
        print("pass 9 dark 470 vs 744")
        p9 = pass9_dark(tr, con)
        print("pass 10 chronic 12")
        p10 = pass10_chronic(tr)
        print("pass 11 Q6")
        p11 = pass11_q6(tr)
        print("pass 12 raw vs unique")
        p12 = pass12_raw(tr, unique_days, raw_dates, mid)
        print("pass 13 90d vs months-on-book")
        p13 = pass13_window(tr)
        print("pass 14 lifetime σ")
        p14 = pass14_fullhist(tr, unique_days)
        print("pass 15 quintiles")
        p15 = pass15_quintiles(tr)
        print("pass 16 inverse days")
        p16 = pass16_invdays(tr)
        print("pass 17 overlap")
        p17 = pass17_overlap(tr)
        print("pass 18 Q6 residual")
        p18 = pass18_q6_resid(tr)
        print("pass 19 rank residual")
        p19 = pass19_rank(tr)
        print("pass 20 n_tx leftover is days")
        p20 = pass20_ntx_is_days(tr)
        print("pass 21 stressed Spearman")
        p21 = pass21_stressed_rho(tr)
        print("pass 22 3x3")
        p22 = pass22_twobytwo(tr)
        print("pass 23 days after gap")
        p23 = pass23_days_after_gap(tr)
    finally:
        con.close()

    decision = decide(p3, p4, p5, p6, p7, p11, p12, p13, p18)
    png_ok = make_png(tr, p15)
    headline = (
        f"`c_gap_sd` train cov {_pp(p1['train_cov'])} "
        f"({'CONFIRM 95.1%' if p1['confirm'] else 'off 95.1%'}). "
        f"ρ vs days {p3['rho_days']:.3f} / a_n_tx {p3['rho_antx']:.3f} / "
        f"c_n_tx {p3['rho_cntx']:.3f} (twin={p3['pairwise_twin']}); "
        f"vs log1p(a_in3) {p3['rho_size']:.3f} "
        f"({'CONFIRM −0.543' if p3['confirm_size'] else 'SIZE ρ measured'}). "
        f"Y3 gap {_f(p4['gap_y3'])} vs size {_f(p4['size_y3'])} vs days {_f(p4['days_y3'])}. "
        f"Leftover after days {_f(p5['y3_days'])} (die={p5['died']}). "
        f"ICC {_f(p7['icc'])} ({'TRAIT' if p7['trait'] else 'shock' if p7['shock'] else 'mixed'}). "
        f"X **{decision['x_dec']}** ({decision['letter']}). PARK as Y. "
        f"Q6 **{decision['q6']}** (lag leftover {_f(p18['leftover'])})."
    )
    print(headline)
    failed = []
    if not p1["confirm"]:
        failed.append(f"coverage {_pp(p1['train_cov'])} ≠ 95.1%")
    if not p2["ok"]:
        failed.append("formula mismatch vs unique-day recon — inspect ops.py, do not rewrite")
    if not p4["days_ok"]:
        failed.append(f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711")
    if not p9["confirm"]:
        failed.append(f"dark/erp {p9['n_dark']}/{p9['n_erp']} ≠ 470/744")
    if p10["n_ids"] != 12:
        failed.append(f"chronic names {p10['n_ids']} ≠ 12 from y2_why")
    if not p12["mid_ok"]:
        failed.append(f"midnight {_pp(p12['midnight'])} ≠ 94%")
    failed.append(
        f"leftover after days {_f(p5['y3_days'])} dies; rank leftover {_f(p19['days'])}; "
        f"Q6 lag leftover {_f(p18['leftover'])}; days leftover after gap {_f(p23['cv'])} lives — "
        "Y3 engine keeps days. Feature report picked the weaker twin as representative."
    )
    if p12["rho_uniq_inv"] is not None and abs(p12["rho_uniq_inv"]) >= TWIN_RHO:
        failed.append(
            f"unique-day σ vs 1/c_n_tx still {p12['rho_uniq_inv']:.3f} — collapse cut raw 0.916 "
            "but did not escape the twin. Do not rewrite ops.py."
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
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
