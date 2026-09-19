"""Q3 zero-in — leftover quiet after days, or SIZE / inverse-activity?

NORTH_STAR: `c_zero_in_month` = 1 if no transaction with amount > 0
(any category) this month. `c_zero_in_share_6` = mean of that flag over
last ≤6 months (min_periods=1). Feature report: month RARE modal 88.2%,
size ρ −0.508 (SIZE); share_6 keep-list representative, size ρ −0.519,
acf1 0.87 BETWEEN.

Y6 `y6_zero_in_3` is a *future* rejected Y (t+1..t+3 all op_in==0 after
activity; inverse size). Different window — compare Jaccard / ρ only.
Do not revive Y6. Do not invent `y_zero_in`. Do not put zero-in on the
15-col card. Night Y3 quote stays 0.762 / 0.752. Days bar 0.711.

Zero-in is “no inflow this month”, not last-booking age (recency) and
not gap σ. Split the flag: empty grid (n_tx=0) vs all-out (n_tx>0 but
no amount>0).

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not edit ops.py unless a real formula bug.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.zero_in_qa

Owned: analysis/evaluate/zero_in_qa.py, analysis/outputs/zero_in_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_zero_in.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "zero_in_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "zero_in_piles.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "87c77f5b"
WAVE = "4"
ROUND = "R4"
MODEL = "zero_in_qa"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y6 = "y6_zero_in_3"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
TWIN_RHO = 0.80
MODAL_QUOTE = 0.882
SIZE_RHO_QUOTE_M = -0.508
SIZE_RHO_QUOTE_S = -0.519
ACF1_QUOTE = 0.87
ICC_TRAIT = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
PANEL_END = pd.Timestamp("2026-08-01")
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
ROLL = 6
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_op_in",
    "a_n_tx",
    "c_n_tx",
    "c_n_days_with_tx",
    "c_zero_in_month",
    "c_zero_in_share_6",
    "c_recency_days",
    "b_below_0",
    "a_uncat_share",
)

Y_KEEP = (Y2, Y3, Y6)


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


def jaccard(a, b) -> float:
    d = pd.DataFrame(
        {
            "a": pd.to_numeric(a, errors="coerce"),
            "b": pd.to_numeric(b, errors="coerce"),
        }
    ).dropna()
    if d.empty:
        return float("nan")
    aa = d["a"].to_numpy(dtype=float) == 1
    bb = d["b"].to_numpy(dtype=float) == 1
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


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


def cv_of(rec: dict) -> float:
    return float("nan") if rec.get("low_power") else rec["cv"]


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


def _company_terciles(tr: pd.DataFrame, col: str, name: str) -> pd.Series:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last[col], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    return terc.rename(name)


def add_piles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    z = pd.to_numeric(out["c_zero_in_month"], errors="coerce")
    ntx = pd.to_numeric(out["c_n_tx"], errors="coerce")
    days = pd.to_numeric(out["c_n_days_with_tx"], errors="coerce")
    opin = pd.to_numeric(out["a_op_in"], errors="coerce")
    out["empty_month"] = (ntx == 0).astype(np.int8)
    out["all_out"] = ((ntx > 0) & (z == 1)).astype(np.int8)
    out["has_in"] = (z == 0).astype(np.int8)
    out["zero_op_in"] = (opin.fillna(0) == 0).astype(np.int8)
    out["days0"] = (days == 0).astype(np.int8)
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
    optional = {"a_uncat_share"}
    hard = [c for c in missing if c not in optional]
    if hard:
        raise RuntimeError(f"monthly.parquet missing {hard}")
    if missing:
        print(f"store optional missing {missing} — skip those extras")
    have = [c for c in STORE_COLS if c in raw.columns]
    ykeep = ["company_id", "period"]
    for c in Y_KEEP:
        if c in yraw.columns:
            ykeep.append(c)
        else:
            print(f"targets missing {c} — will compute in-module if needed")
    panel = _keys(raw[list(have)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["cal_month"] = panel["period"].dt.month
    panel["cal_year"] = panel["period"].dt.year
    panel["log_in3"] = np.log1p(
        pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0)
    )
    panel["log_abs_opin"] = np.log1p(
        pd.to_numeric(panel["a_op_in"], errors="coerce").abs()
    )
    panel["so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    first = panel.groupby("company_id")["period"].transform("min")
    panel["months_on_book"] = (
        (PANEL_END.year - first.dt.year) * 12 + (PANEL_END.month - first.dt.month) + 1
    )
    panel["short_book"] = (panel["months_on_book"] < 12).astype(np.int8)
    panel = add_piles(panel)
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


def attach_y6_if_missing(panel: pd.DataFrame, con) -> pd.DataFrame:
    if Y6 in panel.columns:
        print(f"{Y6} present in targets.parquet — using store, no assembler")
        return panel
    from analysis.targets.y6_activity import build as build_y6

    print(f"{Y6} missing from targets — in-module y6_activity.build (no assembler)")
    grid = panel[["company_id", "period"]].drop_duplicates()
    y6 = build_y6(con, grid)
    y6 = _keys(y6[["company_id", "period", Y6]])
    return panel.merge(y6, on=["company_id", "period"], how="left")


# ---------------------------------------------------------------------------
# Pass 1 — prevalence + modal 88.2% + holdout coverage
# ---------------------------------------------------------------------------
def pass1_prev(panel: pd.DataFrame, tr: pd.DataFrame) -> dict:
    rows = []
    for split, sl in (("train", tr), ("holdout", panel[panel["split"] == "holdout"])):
        for col in ("c_zero_in_month", "c_zero_in_share_6"):
            x = pd.to_numeric(sl[col], errors="coerce")
            nn = x.dropna()
            if col == "c_zero_in_month":
                modal = float(nn.mode().iloc[0]) if len(nn) else float("nan")
                modal_share = float((nn == modal).mean()) if len(nn) else float("nan")
                prev = float((x == 1).mean()) if len(x) else float("nan")
                n_pos = int((x == 1).sum())
            else:
                modal = float(nn.mode().iloc[0]) if len(nn) else float("nan")
                modal_share = float((nn == modal).mean()) if len(nn) else float("nan")
                prev = float(x.mean()) if len(x) else float("nan")
                n_pos = int((x > 0).sum())
            rows.append(
                {
                    "split": split,
                    "col": col,
                    "n_cm": int(len(sl)),
                    "n_co": int(sl["company_id"].nunique()),
                    "cov": float(x.notna().mean()) if len(x) else float("nan"),
                    "mean": prev,
                    "n_pos": n_pos,
                    "modal": modal,
                    "modal_share": modal_share,
                }
            )
    m = next(r for r in rows if r["split"] == "train" and r["col"] == "c_zero_in_month")
    s = next(r for r in rows if r["split"] == "train" and r["col"] == "c_zero_in_share_6")
    ho_m = next(r for r in rows if r["split"] == "holdout" and r["col"] == "c_zero_in_month")
    ho_s = next(r for r in rows if r["split"] == "holdout" and r["col"] == "c_zero_in_share_6")
    confirm = bool(np.isfinite(m["modal_share"]) and abs(m["modal_share"] - MODAL_QUOTE) < 0.015)
    prose = (
        f"Train `c_zero_in_month` prevalence {_pp(m['mean'])} (n={m['n_pos']:,} / {m['n_cm']:,}). "
        f"Modal {m['modal']:g} share {_pp(m['modal_share'])} "
        f"({'CONFIRM 88.2%' if confirm else 'does not match feature-report 88.2%'}). "
        f"`c_zero_in_share_6` mean {_f(s['mean'], 3)} modal {_f(s['modal'])} share {_pp(s['modal_share'])}. "
        f"Holdout coverage only: month mean {_pp(ho_m['mean'])} share mean {_f(ho_s['mean'], 3)} "
        f"on {ho_m['n_cm']:,} CM / {ho_m['n_co']} cos. No AUROC on holdout."
    )
    print(prose)
    return {
        "rows": rows,
        "month": m,
        "share": s,
        "ho_month": ho_m,
        "ho_share": ho_s,
        "confirm_modal": confirm,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — empty grid vs all-out (which pile is the flag?)
# ---------------------------------------------------------------------------
def pass2_split(tr: pd.DataFrame) -> dict:
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    ntx = pd.to_numeric(tr["c_n_tx"], errors="coerce")
    antx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    n = int(len(tr))
    n_z = int((z == 1).sum())
    n_empty = int((tr["empty_month"] == 1).sum())
    n_allout = int((tr["all_out"] == 1).sum())
    n_has = int((tr["has_in"] == 1).sum())
    n_empty_z = int(((tr["empty_month"] == 1) & (z == 1)).sum())
    n_empty_in = int(((tr["empty_month"] == 1) & (z == 0)).sum())
    n_days0 = int((days == 0).sum())
    n_ntx_dis = int((ntx.fillna(-1) != antx.fillna(-1)).sum())
    empty_of_z = _pct(n_empty_z, n_z)
    allout_of_z = _pct(n_allout, n_z)
    pile = (
        "all-out (txs, no amount>0)"
        if allout_of_z >= 0.55
        else (
            "empty grid (n_tx=0)"
            if empty_of_z >= 0.55
            else "mixed empty + all-out"
        )
    )
    # days==0 should equal empty if days is distinct booking days
    days_eq_empty = float(((days == 0) == (ntx == 0)).mean())
    rows = [
        {"pile": "has_in (amount>0)", "n_cm": n_has, "share_cm": _pp(_pct(n_has, n)), "of_zero_in": "—"},
        {"pile": "empty (n_tx=0)", "n_cm": n_empty, "share_cm": _pp(_pct(n_empty, n)), "of_zero_in": _pp(empty_of_z)},
        {
            "pile": "all-out (n_tx>0, no amount>0)",
            "n_cm": n_allout,
            "share_cm": _pp(_pct(n_allout, n)),
            "of_zero_in": _pp(allout_of_z),
        },
        {"pile": "zero-in (empty ∪ all-out)", "n_cm": n_z, "share_cm": _pp(_pct(n_z, n)), "of_zero_in": "100%"},
        {"pile": "days==0", "n_cm": n_days0, "share_cm": _pp(_pct(n_days0, n)), "of_zero_in": "—"},
    ]
    prose = (
        f"Train zero-in n={n_z:,} / {n:,} ({_pp(_pct(n_z, n))}). "
        f"Empty grid {n_empty:,} ({_pp(_pct(n_empty, n))}; {_pp(empty_of_z)} of the flag). "
        f"All-out {n_allout:,} ({_pp(_pct(n_allout, n))}; {_pp(allout_of_z)} of the flag). "
        f"Empty ∩ has-in = {n_empty_in} (must be 0). a_n_tx≠c_n_tx n={n_ntx_dis}. "
        f"days==0 ≡ n_tx==0 agree {_pp(days_eq_empty)}. "
        f"Flag pile: **{pile}**."
    )
    print(prose)
    return {
        "n": n,
        "n_z": n_z,
        "n_empty": n_empty,
        "n_allout": n_allout,
        "n_has": n_has,
        "n_empty_in": n_empty_in,
        "empty_of_z": empty_of_z,
        "allout_of_z": allout_of_z,
        "pile": pile,
        "n_ntx_dis": n_ntx_dis,
        "days_eq_empty": days_eq_empty,
        "rows": rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — formula vs raw txs (any amount>0)
# ---------------------------------------------------------------------------
def pass3_formula(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          COUNT(*) AS raw_n_tx,
          MAX(CASE WHEN amount > 0 THEN 1 ELSE 0 END) AS raw_has_in,
          SUM(CASE WHEN amount > 0 THEN 1 ELSE 0 END) AS n_pos_amt,
          SUM(CASE WHEN amount = 0 THEN 1 ELSE 0 END) AS n_zero_amt,
          SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) AS n_neg_amt
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["month"])
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])

    grid = tr[
        [
            "company_id",
            "period",
            "c_zero_in_month",
            "c_zero_in_share_6",
            "c_n_tx",
            "empty_month",
            "all_out",
        ]
    ].copy()
    grid = grid.merge(
        raw[["company_id", "period", "raw_n_tx", "raw_has_in", "n_pos_amt", "n_zero_amt", "n_neg_amt"]],
        on=["company_id", "period"],
        how="left",
    )
    grid["raw_n_tx"] = grid["raw_n_tx"].fillna(0).astype(int)
    grid["raw_has_in"] = grid["raw_has_in"].fillna(0).astype(int)
    recon = (grid["raw_has_in"] == 0).astype(int)
    store_z = pd.to_numeric(grid["c_zero_in_month"], errors="coerce").fillna(0).astype(int)
    agree_z = float((recon == store_z).mean())
    agree_n = float((grid["raw_n_tx"] == pd.to_numeric(grid["c_n_tx"], errors="coerce")).mean())
    n_dis_z = int((recon != store_z).sum())

    g = grid.sort_values(["company_id", "period"])
    recon6 = g.groupby("company_id", sort=False)["c_zero_in_month"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(ROLL, min_periods=1).mean()
    )
    store6 = pd.to_numeric(g["c_zero_in_share_6"], errors="coerce")
    agree6 = float(np.nanmax(np.abs(recon6 - store6)) < 1e-9) if len(g) else float("nan")
    max_d6 = float(np.nanmax(np.abs(recon6 - store6))) if len(g) else float("nan")

    # all-out category-all-out: among n_tx>0 & recon=1, share that have only neg amounts
    ao = grid[grid["all_out"] == 1]
    only_neg = int(((ao["n_neg_amt"].fillna(0) > 0) & (ao["n_pos_amt"].fillna(0) == 0)).sum())
    only_zero = int(
        ((ao["n_zero_amt"].fillna(0) > 0) & (ao["n_neg_amt"].fillna(0) == 0) & (ao["n_pos_amt"].fillna(0) == 0)).sum()
    )
    n_ao_raw = int(len(ao))
    bug = bool(agree_z < 0.995 or (np.isfinite(max_d6) and max_d6 > 1e-6))
    prose = (
        f"Store `c_zero_in_month` vs raw (any amount>0) agree {_pp(agree_z)} "
        f"(n_disagree={n_dis_z}). n_tx agree {_pp(agree_n)}. "
        f"share_6 vs rolling-6 of store month max|Δ|={max_d6:.2e}. "
        f"All-out CM {n_ao_raw:,}: only-neg {only_neg:,} ({_pp(_pct(only_neg, n_ao_raw))}), "
        f"only-zero-amt {only_zero:,}. "
        f"{'FORMULA BUG — stop and report, do not patch ops.py tonight.' if bug else 'Not an ops.py bug.'}"
    )
    print(prose)
    return {
        "agree_z": agree_z,
        "agree_n": agree_n,
        "n_dis_z": n_dis_z,
        "max_d6": max_d6,
        "agree6": agree6,
        "only_neg": only_neg,
        "only_zero": only_zero,
        "n_ao_raw": n_ao_raw,
        "bug": bug,
        "prose": prose,
        "raw": raw,
        "grid": grid,
    }


# ---------------------------------------------------------------------------
# Pass 4 — Spearman twins / SIZE
# ---------------------------------------------------------------------------
def pass4_rho(tr: pd.DataFrame) -> dict:
    pairs = [
        ("c_zero_in_month", "c_n_days_with_tx"),
        ("c_zero_in_month", "a_n_tx"),
        ("c_zero_in_month", "c_n_tx"),
        ("c_zero_in_month", "a_op_in"),
        ("c_zero_in_month", "log_in3"),
        ("c_zero_in_month", "log_abs_opin"),
        ("c_zero_in_month", "c_recency_days"),
        ("c_zero_in_month", "empty_month"),
        ("c_zero_in_month", "all_out"),
        ("c_zero_in_month", "zero_op_in"),
        ("c_zero_in_share_6", "c_n_days_with_tx"),
        ("c_zero_in_share_6", "a_n_tx"),
        ("c_zero_in_share_6", "a_op_in"),
        ("c_zero_in_share_6", "log_in3"),
        ("c_zero_in_share_6", "log_abs_opin"),
        ("c_zero_in_share_6", "c_recency_days"),
        ("c_zero_in_share_6", "c_zero_in_month"),
        ("empty_month", "c_n_days_with_tx"),
        ("all_out", "c_n_days_with_tx"),
        ("all_out", "a_n_tx"),
        ("all_out", "log_in3"),
        ("empty_month", "log_in3"),
        ("zero_op_in", "c_zero_in_month"),
    ]
    rows = []
    rhos = {}
    for a, b in pairs:
        r = spearman(tr[a], tr[b])
        rhos[(a, b)] = r
        twin = bool(np.isfinite(r) and abs(r) >= TWIN_RHO)
        size = bool(b in {"log_in3", "log_abs_opin"} and np.isfinite(r) and abs(r) >= SIZE_RHO)
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "ρ": _f(r),
                "twin |ρ|≥0.80": "YES" if twin else "",
                "SIZE |ρ|≥0.50": "YES" if size else "",
            }
        )
    rho_m_size = rhos[("c_zero_in_month", "log_in3")]
    rho_s_size = rhos[("c_zero_in_share_6", "log_in3")]
    rho_m_opin_sz = rhos.get(("c_zero_in_month", "log_abs_opin"), float("nan"))
    rho_s_opin_sz = rhos.get(("c_zero_in_share_6", "log_abs_opin"), float("nan"))
    rho_m_days = rhos[("c_zero_in_month", "c_n_days_with_tx")]
    rho_m_ntx = rhos[("c_zero_in_month", "a_n_tx")]
    rho_m_opin = rhos[("c_zero_in_month", "a_op_in")]
    rho_m_rec = rhos[("c_zero_in_month", "c_recency_days")]
    rho_s_days = rhos[("c_zero_in_share_6", "c_n_days_with_tx")]
    rho_ao_days = rhos[("all_out", "c_n_days_with_tx")]
    rho_em_size = rhos[("empty_month", "log_in3")]
    rho_ao_size = rhos[("all_out", "log_in3")]
    confirm_m = bool(np.isfinite(rho_m_opin_sz) and abs(rho_m_opin_sz - SIZE_RHO_QUOTE_M) < 0.03)
    confirm_s = bool(np.isfinite(rho_s_opin_sz) and abs(rho_s_opin_sz - SIZE_RHO_QUOTE_S) < 0.03)
    size_m = bool(np.isfinite(rho_m_size) and abs(rho_m_size) >= SIZE_RHO)
    size_s = bool(np.isfinite(rho_s_size) and abs(rho_s_size) >= SIZE_RHO)
    size_m_opin = bool(np.isfinite(rho_m_opin_sz) and abs(rho_m_opin_sz) >= SIZE_RHO)
    size_s_opin = bool(np.isfinite(rho_s_opin_sz) and abs(rho_s_opin_sz) >= SIZE_RHO)
    twin_m = bool(
        (np.isfinite(rho_m_days) and abs(rho_m_days) >= TWIN_RHO)
        or (np.isfinite(rho_m_ntx) and abs(rho_m_ntx) >= TWIN_RHO)
    )
    twin_rec = bool(np.isfinite(rho_m_rec) and abs(rho_m_rec) >= TWIN_RHO)
    prose = (
        f"Month vs log1p(a_in3) ρ={_f(rho_m_size)} "
        f"{'SIZE' if size_m else 'not SIZE on the KEEP clock'}. "
        f"Month vs log1p(|a_op_in|) ρ={_f(rho_m_opin_sz)} "
        f"({'CONFIRM −0.508' if confirm_m else 'off feature-report −0.508'}) "
        f"{'SIZE' if size_m_opin else 'not SIZE'}. "
        f"share_6 vs log1p(a_in3) ρ={_f(rho_s_size)} "
        f"{'SIZE' if size_s else 'not SIZE on the KEEP clock'}. "
        f"share_6 vs log1p(|a_op_in|) ρ={_f(rho_s_opin_sz)} "
        f"({'CONFIRM −0.519' if confirm_s else 'off −0.519'}) "
        f"{'SIZE' if size_s_opin else 'not SIZE'}. "
        f"Month vs days {_f(rho_m_days)} n_tx {_f(rho_m_ntx)} a_op_in {_f(rho_m_opin)} "
        f"recency {_f(rho_m_rec)}. "
        f"All-out vs days {_f(rho_ao_days)} size {_f(rho_ao_size)}; empty vs size {_f(rho_em_size)}. "
        f"{'TWIN of days/n_tx' if twin_m else 'not a |ρ|≥0.80 days/n_tx twin'}. "
        f"{'TWIN of recency' if twin_rec else 'not a recency twin'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "rho_m_size": rho_m_size,
        "rho_s_size": rho_s_size,
        "rho_m_opin_sz": rho_m_opin_sz,
        "rho_s_opin_sz": rho_s_opin_sz,
        "rho_m_days": rho_m_days,
        "rho_m_ntx": rho_m_ntx,
        "rho_m_opin": rho_m_opin,
        "rho_m_rec": rho_m_rec,
        "rho_s_days": rho_s_days,
        "rho_ao_days": rho_ao_days,
        "rho_ao_size": rho_ao_size,
        "rho_em_size": rho_em_size,
        "confirm_m": confirm_m,
        "confirm_s": confirm_s,
        "size_m": size_m,
        "size_s": size_s,
        "size_m_opin": size_m_opin,
        "size_s_opin": size_s_opin,
        "twin_m": twin_m,
        "twin_rec": twin_rec,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — group-fold AUROC Y3 / Y2
# ---------------------------------------------------------------------------
def pass5_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "c_zero_in_month": tr["c_zero_in_month"],
        "c_zero_in_share_6": tr["c_zero_in_share_6"],
        "empty_month": tr["empty_month"],
        "all_out": tr["all_out"],
        "has_in": tr["has_in"],
        "zero_op_in": tr["zero_op_in"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "a_op_in": tr["a_op_in"],
        "log1p_a_in3": tr["log_in3"],
        "c_recency_days": tr["c_recency_days"],
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

    size_y3 = _cv(Y3, "log1p_a_in3")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    ntx_y3 = _cv(Y3, "a_n_tx")
    opin_y3 = _cv(Y3, "a_op_in")
    m_y3 = _cv(Y3, "c_zero_in_month")
    s_y3 = _cv(Y3, "c_zero_in_share_6")
    em_y3 = _cv(Y3, "empty_month")
    ao_y3 = _cv(Y3, "all_out")
    m_y2 = _cv(Y2, "c_zero_in_month")
    s_y2 = _cv(Y2, "c_zero_in_share_6")
    size_y2 = _cv(Y2, "log1p_a_in3")
    days_y2 = _cv(Y2, "c_n_days_with_tx")
    beat_m = m_y3 - size_y3 if np.isfinite(m_y3) and np.isfinite(size_y3) else float("nan")
    beat_s = s_y3 - size_y3 if np.isfinite(s_y3) and np.isfinite(size_y3) else float("nan")
    beat_ao = ao_y3 - size_y3 if np.isfinite(ao_y3) and np.isfinite(size_y3) else float("nan")
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) <= 0.02)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) <= 0.02)
    prose = (
        f"Y3 group-fold: month {_f(m_y3)} share_6 {_f(s_y3)} empty {_f(em_y3)} all-out {_f(ao_y3)} "
        f"vs size {_f(size_y3)} (quote 0.617, Δmonth {_f(beat_m)}) "
        f"vs days {_f(days_y3)} (night 0.711, replica {'OK' if days_ok else 'off'}) "
        f"vs n_tx {_f(ntx_y3)} vs a_op_in {_f(opin_y3)}. "
        f"Y2 month {_f(m_y2)} share_6 {_f(s_y2)} vs size {_f(size_y2)} days {_f(days_y2)}. "
        f"Month beats size ≥0.02: {'YES' if np.isfinite(beat_m) and beat_m >= KEEP_DELTA else 'NO'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "m_y3": m_y3,
        "s_y3": s_y3,
        "em_y3": em_y3,
        "ao_y3": ao_y3,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "ntx_y3": ntx_y3,
        "opin_y3": opin_y3,
        "m_y2": m_y2,
        "s_y2": s_y2,
        "size_y2": size_y2,
        "days_y2": days_y2,
        "beat_m": beat_m,
        "beat_s": beat_s,
        "beat_ao": beat_ao,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — leftover after days / n_tx
# ---------------------------------------------------------------------------
def pass6_resid(tr: pd.DataFrame) -> dict:
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    sh = pd.to_numeric(tr["c_zero_in_share_6"], errors="coerce")
    ao = pd.to_numeric(tr["all_out"], errors="coerce")
    em = pd.to_numeric(tr["empty_month"], errors="coerce")
    specs = [
        ("z_resid_days", *ols_resid(z, tr["c_n_days_with_tx"])),
        ("z_resid_ntx", *ols_resid(z, tr["a_n_tx"])),
        ("z_resid_card", *ols_resid(z, tr["c_n_days_with_tx"], tr["a_n_tx"])),
        ("z_resid_opin", *ols_resid(z, tr["a_op_in"])),
        ("sh_resid_days", *ols_resid(sh, tr["c_n_days_with_tx"])),
        ("sh_resid_ntx", *ols_resid(sh, tr["a_n_tx"])),
        ("sh_resid_card", *ols_resid(sh, tr["c_n_days_with_tx"], tr["a_n_tx"])),
        ("ao_resid_days", *ols_resid(ao, tr["c_n_days_with_tx"])),
        ("ao_resid_ntx", *ols_resid(ao, tr["a_n_tx"])),
        ("em_resid_days", *ols_resid(em, tr["c_n_days_with_tx"])),
        ("sh_resid_z", *ols_resid(sh, z)),
    ]
    rows = []
    cvs = {}
    spec_resids = {name: resid for name, resid, _info in specs}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        size = cv_of(signed_oof_auroc(tr[y], tr["log_in3"], tr["fold"], lab))
        for name, resid, info in specs:
            rec = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
            cv = cv_of(rec)
            cvs[(y, name)] = cv
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(cv),
                    "sign": rec["train_sign"] if not rec["low_power"] else "—",
                    "Δsize": _f((cv - size) if np.isfinite(cv) and np.isfinite(size) else float("nan")),
                    "slope": _f(info["slope"][0]) if info["slope"] else "—",
                }
            )
    # honest leftover: all-out among months that already have txs (days already >0)
    busy = tr["empty_month"] == 0
    lab3 = tr[Y3].notna()
    ao_busy = signed_oof_auroc(tr[Y3], tr["all_out"], tr["fold"], lab3 & busy)
    z_busy = signed_oof_auroc(tr[Y3], tr["c_zero_in_month"], tr["fold"], lab3 & busy)
    size_busy = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab3 & busy)
    days_busy = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab3 & busy)
    r_ao_busy, _ = ols_resid(tr["all_out"].where(busy), tr["c_n_days_with_tx"])
    ao_busy_d = signed_oof_auroc(tr[Y3], r_ao_busy, tr["fold"], lab3 & busy)

    y3_z_d = cvs.get((Y3, "z_resid_days"), float("nan"))
    y3_z_n = cvs.get((Y3, "z_resid_ntx"), float("nan"))
    y3_z_c = cvs.get((Y3, "z_resid_card"), float("nan"))
    y3_sh_d = cvs.get((Y3, "sh_resid_days"), float("nan"))
    y3_sh_n = cvs.get((Y3, "sh_resid_ntx"), float("nan"))
    y3_ao_d = cvs.get((Y3, "ao_resid_days"), float("nan"))
    size_y3 = cv_of(signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab3))
    leftover_z = (
        (y3_z_d - size_y3) if np.isfinite(y3_z_d) and np.isfinite(size_y3) else float("nan")
    )
    leftover_sh = (
        (y3_sh_d - size_y3) if np.isfinite(y3_sh_d) and np.isfinite(size_y3) else float("nan")
    )
    leftover_ao = (
        (cv_of(ao_busy) - cv_of(size_busy))
        if np.isfinite(cv_of(ao_busy)) and np.isfinite(cv_of(size_busy))
        else float("nan")
    )
    died_z = bool(np.isfinite(y3_z_d) and y3_z_d < 0.55)
    died_sh = bool(np.isfinite(y3_sh_d) and y3_sh_d < 0.55)
    lives_z = bool(np.isfinite(leftover_z) and leftover_z >= KEEP_DELTA)
    lives_sh = bool(np.isfinite(leftover_sh) and leftover_sh >= KEEP_DELTA)
    lives_ao = bool(np.isfinite(leftover_ao) and leftover_ao >= KEEP_DELTA)
    rho_fake = spearman(spec_resids["z_resid_days"], tr["c_n_days_with_tx"])
    quiet_twin = bool(died_z or (np.isfinite(y3_z_d) and y3_z_d < 0.60 and not lives_z))
    prose = (
        f"Month leftover after days {_f(y3_z_d)} / after n_tx {_f(y3_z_n)} / card {_f(y3_z_c)} "
        f"(Δsize {_f(leftover_z)}). share_6 leftover after days {_f(y3_sh_d)} (Δsize {_f(leftover_sh)}). "
        f"All-out leftover after days {_f(y3_ao_d)}. "
        f"Among n_tx>0: all-out Y3 {_f(cv_of(ao_busy))} vs size {_f(cv_of(size_busy))} "
        f"vs days {_f(cv_of(days_busy))}; month-on-busy {_f(cv_of(z_busy))}; "
        f"all-out resid-days on busy {_f(cv_of(ao_busy_d))}. "
        f"ρ(month-resid, days)={_f(rho_fake)}. "
        + (
            "Leftover dies — CLOSE as quiet twin of days/n_tx."
            if quiet_twin and not (lives_z or lives_sh or lives_ao)
            else (
                "Leftover beats size ≥0.02 after days — later C candidate (not on the card)."
                if (lives_z or lives_sh or lives_ao)
                else "Leftover does not clear size+0.02 after days."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_z_d": y3_z_d,
        "y3_z_n": y3_z_n,
        "y3_z_c": y3_z_c,
        "y3_sh_d": y3_sh_d,
        "y3_sh_n": y3_sh_n,
        "y3_ao_d": y3_ao_d,
        "ao_busy": cv_of(ao_busy),
        "z_busy": cv_of(z_busy),
        "size_busy": cv_of(size_busy),
        "days_busy": cv_of(days_busy),
        "ao_busy_d": cv_of(ao_busy_d),
        "leftover_z": leftover_z,
        "leftover_sh": leftover_sh,
        "leftover_ao": leftover_ao,
        "died_z": died_z,
        "died_sh": died_sh,
        "lives_z": lives_z,
        "lives_sh": lives_sh,
        "lives_ao": lives_ao,
        "quiet_twin": quiet_twin,
        "rho_fake": rho_fake,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — SIZE terciles
# ---------------------------------------------------------------------------
def pass7_terciles(tr: pd.DataFrame) -> dict:
    terc = _company_terciles(tr, "log_in3", "size_t")
    m = tr.merge(terc.reset_index(), on="company_id", how="left")
    rows = []
    store = {}
    for labv in ("T1", "T2", "T3"):
        sl = m[m["size_t"].astype(str) == labv]
        prev_z = float(pd.to_numeric(sl["c_zero_in_month"], errors="coerce").mean()) if len(sl) else float("nan")
        prev_em = float(pd.to_numeric(sl["empty_month"], errors="coerce").mean()) if len(sl) else float("nan")
        prev_ao = float(pd.to_numeric(sl["all_out"], errors="coerce").mean()) if len(sl) else float("nan")
        lab = sl[Y3].notna()
        for feat in ("c_zero_in_month", "c_zero_in_share_6", "empty_month", "all_out", "log_in3", "c_n_days_with_tx"):
            res = signed_oof_auroc(sl[Y3], sl[feat] if feat != "log_in3" else sl["log_in3"], sl["fold"], lab)
            store[(labv, feat)] = res
            rows.append(
                {
                    "tercile": labv,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "zero_in": _pp(prev_z),
                    "empty": _pp(prev_em),
                    "all_out": _pp(prev_ao),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    beats = []
    for labv in ("T1", "T2", "T3"):
        r = store[(labv, "c_zero_in_month")]
        s = store[(labv, "log_in3")]
        if (
            not r["low_power"]
            and not s["low_power"]
            and np.isfinite(r["cv"])
            and np.isfinite(s["cv"])
            and (r["cv"] - s["cv"]) >= KEEP_DELTA
        ):
            beats.append(labv)
    t1_z = store[("T1", "c_zero_in_month")]
    t3_z = store[("T3", "c_zero_in_month")]
    prose = (
        f"Y3 month CV inside size terciles that beat size ≥0.02: {beats or 'none'}. "
        f"T1 (small) {_f(t1_z['cv']) if not t1_z['low_power'] else 'LOW_POWER'}; "
        f"T3 (large) {_f(t3_z['cv']) if not t3_z['low_power'] else 'LOW_POWER'}. "
        "If skill is only T1, it is the inverse-size / Y6 failure mode."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "beats": beats,
        "t1": cv_of(t1_z),
        "t3": cv_of(t3_z),
        "m": m,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — vs y6_zero_in_3 (do not revive)
# ---------------------------------------------------------------------------
def pass8_y6(tr: pd.DataFrame) -> dict:
    if Y6 not in tr.columns:
        return {
            "present": False,
            "prose": "y6_zero_in_3 unavailable even after in-module build.",
            "jac": float("nan"),
            "rho": float("nan"),
            "leak": False,
            "rows": [],
        }
    x = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    sh = pd.to_numeric(tr["c_zero_in_share_6"], errors="coerce")
    y = pd.to_numeric(tr[Y6], errors="coerce")
    both = x.notna() & y.notna()
    jac = jaccard(x[both], y[both])
    rho = spearman(x[both], y[both])
    rho_sh = spearman(sh[both], y[both])
    n_both = int(both.sum())
    n_x1 = int(((x == 1) & both).sum())
    n_y1 = int(((y == 1) & both).sum())
    n_and = int(((x == 1) & (y == 1) & both).sum())
    p_y_given_x = float(y[(x == 1) & both].mean()) if ((x == 1) & both).any() else float("nan")
    p_x_given_y = float(x[(y == 1) & both].mean()) if ((y == 1) & both).any() else float("nan")
    y_rate = float(y[both].mean()) if both.any() else float("nan")
    leak = bool((np.isfinite(jac) and jac >= 0.50) or (np.isfinite(rho) and abs(rho) >= TWIN_RHO))
    # lead: P(Y6 | zero-in now) vs base — still not a Y
    rows = [
        {"item": "overlap n (Y6 defined)", "value": f"{n_both:,}"},
        {"item": "c_zero_in_month=1 on overlap", "value": f"{n_x1:,}"},
        {"item": "y6_zero_in_3=1 on overlap", "value": f"{n_y1:,}"},
        {"item": "intersection", "value": f"{n_and:,}"},
        {"item": "Jaccard", "value": _f(jac)},
        {"item": "Spearman month", "value": _f(rho)},
        {"item": "Spearman share_6", "value": _f(rho_sh)},
        {"item": "P(Y6|zero-in now)", "value": _pp(p_y_given_x)},
        {"item": "P(zero-in now|Y6)", "value": _pp(p_x_given_y)},
        {"item": "Y6 base on overlap", "value": _pp(y_rate)},
    ]
    lab = tr[Y3].notna() & y.notna()
    y6_as_x = signed_oof_auroc(tr[Y3], y, tr["fold"], lab)
    # distinct window: now vs t+1..t+3
    distinct = bool((not leak) or (np.isfinite(jac) and jac < 0.50))
    prose = (
        f"Y6 overlap n={n_both:,}; Jaccard {_f(jac)} Spearman {_f(rho)} (share_6 {_f(rho_sh)}). "
        f"P(Y6|zero-in now)={_pp(p_y_given_x)} vs Y6 base {_pp(y_rate)}. "
        f"Y3 using Y6-as-X {_f(y6_as_x['cv']) if not y6_as_x['low_power'] else 'LOW_POWER'}. "
        f"{'X leaks the rejected Y6' if leak else 'X does not leak y6_zero_in_3 (different window: now vs t+1..t+3)'}. "
        "Do not revive Y6. Do not invent a zero-in Y."
    )
    print(prose)
    return {
        "present": True,
        "jac": jac,
        "rho": rho,
        "rho_sh": rho_sh,
        "leak": leak,
        "distinct": distinct,
        "n_both": n_both,
        "n_x1": n_x1,
        "n_y1": n_y1,
        "n_and": n_and,
        "p_y_given_x": p_y_given_x,
        "p_x_given_y": p_x_given_y,
        "y_rate": y_rate,
        "y3_y6": y6_as_x["cv"] if not y6_as_x["low_power"] else float("nan"),
        "rows": rows,
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

    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_z=("c_zero_in_month", "sum"),
        n_em=("empty_month", "sum"),
        n_ao=("all_out", "sum"),
        share6=("c_zero_in_share_6", "mean"),
    )
    ever["ever_erp"] = ever["company_id"].isin(book)
    ever["z_rate"] = ever["n_z"] / ever["n_cm"]
    rows = []
    for name, part in (("ever_erp_744", ever[ever["ever_erp"]]), ("never_erp_470", ever[~ever["ever_erp"]])):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "ever_zero": _pp(float((part["n_z"] > 0).mean())),
                "zero_cm": _pp(float(part["z_rate"].mean())),
                "empty_cm": _pp(float((part["n_em"] / part["n_cm"]).mean())),
                "all_out_cm": _pp(float((part["n_ao"] / part["n_cm"]).mean())),
                "share6": _f(float(part["share6"].mean()), 3),
            }
        )
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    cm = []
    store = {}
    for name, part in (("ever_erp", tr2[tr2["ever_erp"]]), ("never_erp", tr2[~tr2["ever_erp"]])):
        cm.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "zero": float(pd.to_numeric(part["c_zero_in_month"], errors="coerce").mean()),
                "empty": float(pd.to_numeric(part["empty_month"], errors="coerce").mean()),
                "all_out": float(pd.to_numeric(part["all_out"], errors="coerce").mean()),
            }
        )
        lab = part[Y3].notna()
        res = signed_oof_auroc(part[Y3], part["c_zero_in_month"], part["fold"], lab)
        store[name] = res
    dark_z = float(ever.loc[~ever["ever_erp"], "z_rate"].mean())
    erp_z = float(ever.loc[ever["ever_erp"], "z_rate"].mean())
    same = bool(np.isfinite(dark_z) and np.isfinite(erp_z) and abs(dark_z - erp_z) < 0.03)
    hold = load_holdout()
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ'}). "
        f"Mean company zero-in CM: invoiced {_pp(erp_z)} vs dark {_pp(dark_z)}. "
        f"{'Same bank-book zero-in rate' if same else 'Dark files zero-in at a different rate'}. "
        f"Y3 month CV invoiced {_f(cv_of(store['ever_erp']))} dark {_f(cv_of(store['never_erp']))}. "
        f"Holdout ever-ERP coverage only: {hold_book}/{len(hold)}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "dark_z": dark_z,
        "erp_z": erp_z,
        "same": same,
        "hold_book": hold_book,
        "hold_n": len(hold),
        "erp_cv": cv_of(store["ever_erp"]),
        "dark_cv": cv_of(store["never_erp"]),
        "prose": prose,
        "book": book,
    }


# ---------------------------------------------------------------------------
# Pass 10 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass10_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    rows = []
    store = {}
    for y, feat, col in (
        (Y2, "c_zero_in_month", tr["c_zero_in_month"]),
        (Y2, "c_zero_in_share_6", tr["c_zero_in_share_6"]),
        (Y2, "log1p_a_in3", tr["log_in3"]),
        (Y2, "c_n_days_with_tx", tr["c_n_days_with_tx"]),
        (Y3, "c_zero_in_month", tr["c_zero_in_month"]),
        (Y3, "c_zero_in_share_6", tr["c_zero_in_share_6"]),
    ):
        lab = tr[y].notna()
        full = signed_oof_auroc(tr[y], col, tr["fold"], lab)
        rest = signed_oof_auroc(tr[y], col, tr["fold"], lab & drop)
        store[(y, feat, "full")] = full
        store[(y, feat, "drop")] = rest
        rows.append(
            {
                "y": y,
                "feature": feat,
                "slice": "full",
                "n": f"{full['n_defined']:,}",
                "n_pos": f"{full['n_pos']:,}",
                "CV": "LOW_POWER" if full["low_power"] else _f(full["cv"]),
            }
        )
        rows.append(
            {
                "y": y,
                "feature": feat,
                "slice": "drop_12",
                "n": f"{rest['n_defined']:,}",
                "n_pos": f"{rest['n_pos']:,}",
                "CV": "LOW_POWER" if rest["low_power"] else _f(rest["cv"]),
            }
        )
    z_ch = float(
        pd.to_numeric(tr.loc[tr["company_id"].astype(str).isin(set(ids)), "c_zero_in_month"], errors="coerce").mean()
    ) if ids else float("nan")
    z_rest = float(pd.to_numeric(tr.loc[drop, "c_zero_in_month"], errors="coerce").mean())
    full = store[(Y2, "c_zero_in_month", "full")]
    rest = store[(Y2, "c_zero_in_month", "drop")]
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    prose = (
        f"Chronic 12 names (0158/0172, ≥50% labeled months below 0): {len(ids)}. "
        f"Y2 month CV full {_f(cv_of(full))} → drop-12 {_f(cv_of(rest))}. "
        f"Zero-in share on 12 {_pp(z_ch)} vs rest {_pp(z_rest)}. "
        f"{'DROP FLIPS Y2' if flip else 'Drop does not flip Y2 (≥0.03)'}."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "ids": ids,
        "y2_full": cv_of(full),
        "y2_drop": cv_of(rest),
        "z_ch": z_ch,
        "z_rest": z_rest,
        "flip": flip,
        "rows": rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — ICC / company-demean (share_6 acf1 0.87)
# ---------------------------------------------------------------------------
def pass11_icc(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name, col in (
        ("c_zero_in_month", tr["c_zero_in_month"]),
        ("c_zero_in_share_6", tr["c_zero_in_share_6"]),
        ("empty_month", tr["empty_month"]),
        ("all_out", tr["all_out"]),
    ):
        icc = icc_anova(col, tr["company_id"])
        acf1 = median_acf(col, tr["company_id"], 1)
        acf3 = median_acf(col, tr["company_id"], 3)
        mu = pd.to_numeric(col, errors="coerce").groupby(tr["company_id"], sort=False).transform("mean")
        dem = pd.to_numeric(col, errors="coerce") - mu
        lab = tr[Y3].notna()
        raw = signed_oof_auroc(tr[Y3], col, tr["fold"], lab)
        dem_res = signed_oof_auroc(tr[Y3], dem, tr["fold"], lab)
        drop = (
            raw["cv"] - dem_res["cv"]
            if (not raw["low_power"] and not dem_res["low_power"])
            else float("nan")
        )
        store[name] = {
            "icc": icc,
            "acf1": acf1,
            "acf3": acf3,
            "raw": raw,
            "dem": dem_res,
            "drop": drop,
        }
        rows.append(
            {
                "col": name,
                "ICC": _f(icc["icc"]),
                "acf1": _f(acf1),
                "acf3": _f(acf3),
                "Y3 raw": _f(cv_of(raw)),
                "Y3 demean": _f(cv_of(dem_res)),
                "drop": _f(drop),
            }
        )
    sh = store["c_zero_in_share_6"]
    mo = store["c_zero_in_month"]
    confirm_acf = bool(np.isfinite(sh["acf1"]) and abs(sh["acf1"] - ACF1_QUOTE) < 0.05)
    trait = bool(np.isfinite(sh["icc"]["icc"]) and sh["icc"]["icc"] >= ICC_TRAIT)
    shock = bool(np.isfinite(mo["icc"]["icc"]) and mo["icc"]["icc"] < 0.50)
    prose = (
        f"share_6 ICC={_f(sh['icc']['icc'])} acf1={_f(sh['acf1'])} "
        f"({'CONFIRM 0.87' if confirm_acf else 'off feature-report 0.87'}). "
        f"month ICC={_f(mo['icc']['icc'])} acf1={_f(mo['acf1'])}. "
        f"Y3 share_6 raw {_f(cv_of(sh['raw']))} vs demean {_f(cv_of(sh['dem']))} "
        f"(drop {_f(sh['drop'])}). "
        f"{'TRAIT (BETWEEN company style)' if trait else ('MONTH SHOCK' if shock else 'mixed / between')}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "icc_sh": sh["icc"]["icc"],
        "icc_m": mo["icc"]["icc"],
        "acf1_sh": sh["acf1"],
        "acf1_m": mo["acf1"],
        "confirm_acf": confirm_acf,
        "trait": trait,
        "shock": shock,
        "dem_sh": cv_of(sh["dem"]),
        "raw_sh": cv_of(sh["raw"]),
        "drop_sh": sh["drop"],
        "dem_m": cv_of(mo["dem"]),
        "raw_m": cv_of(mo["raw"]),
        "drop_m": mo["drop"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — Q6 lag1 / lag3 on short vs long books
# ---------------------------------------------------------------------------
def pass12_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["short_book"] == 1),
        ("long_>=12", tr["short_book"] == 0),
        ("so_far>=4", tr["so_far"] >= 4),
    )
    cols = (
        "c_zero_in_month",
        "c_zero_in_month_lag1",
        "c_zero_in_month_lag3",
        "c_zero_in_share_6",
        "c_zero_in_share_6_lag1",
        "c_zero_in_share_6_lag3",
    )
    for y in (Y2, Y3):
        for sname, smask in slices:
            lab = tr[y].notna() & smask
            for col in cols:
                if col not in tr.columns:
                    continue
                res = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab & tr[col].notna())
                store[(y, sname, col)] = res
                prev = (
                    float(pd.to_numeric(tr.loc[lab, col], errors="coerce").mean())
                    if lab.any()
                    else float("nan")
                )
                rows.append(
                    {
                        "y": y,
                        "slice": sname,
                        "col": col,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "x_mean": _pp(prev) if "share" not in col else _f(prev, 3),
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    }
                )

    def _cv(y, sl, col) -> float:
        r = store.get((y, sl, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = _cv(Y3, "all", "c_zero_in_month")
    lag1 = _cv(Y3, "all", "c_zero_in_month_lag1")
    lag3 = _cv(Y3, "all", "c_zero_in_month_lag3")
    now_sh = _cv(Y3, "all", "c_zero_in_share_6")
    lag1_sh = _cv(Y3, "all", "c_zero_in_share_6_lag1")
    lag3_sh = _cv(Y3, "all", "c_zero_in_share_6_lag3")
    short_now = _cv(Y3, "short_<12", "c_zero_in_month")
    short_lag1 = _cv(Y3, "short_<12", "c_zero_in_month_lag1")
    long_lag1 = _cv(Y3, "long_>=12", "c_zero_in_month_lag1")
    short_prev = float(
        pd.to_numeric(tr.loc[tr["short_book"] == 1, "c_zero_in_month"], errors="coerce").mean()
    )
    n_short = int((tr["short_book"] == 1).sum())
    n_short_z = int(
        ((tr["short_book"] == 1) & (pd.to_numeric(tr["c_zero_in_month"], errors="coerce") == 1)).sum()
    )
    empty_short = bool(n_short_z < 10 or (np.isfinite(short_prev) and short_prev < 0.005))
    keep_q6 = bool(
        np.isfinite(lag1)
        and np.isfinite(now)
        and now >= 0.60
        and (now - lag1) <= 0.03
        and lag1 >= 0.55
        and not empty_short
    )
    if not np.isfinite(now) or now < 0.60:
        q6 = "CLOSE"
        why = (
            f"contemporaneous Y3 {_f(now)} loses to size / leftover dies; "
            f"lag1 {_f(lag1)} / lag3 {_f(lag3)} have no leftover to lead."
        )
    elif empty_short:
        q6 = "CLOSE"
        why = f"empty on short books (zero-in share {_pp(short_prev)}); Q6 needs a trail."
    else:
        q6 = "KEEP" if keep_q6 else "CLOSE"
        why = f"now {_f(now)} vs lag1 {_f(lag1)} vs lag3 {_f(lag3)}; share_6 now {_f(now_sh)} lag1 {_f(lag1_sh)}."
    prose = (
        f"Q6 Y3 month now {_f(now)} lag1 {_f(lag1)} lag3 {_f(lag3)}; "
        f"share_6 now {_f(now_sh)} lag1 {_f(lag1_sh)} lag3 {_f(lag3_sh)}; "
        f"short now {_f(short_now)} lag1 {_f(short_lag1)}; long lag1 {_f(long_lag1)}. "
        f"Short-book zero-in share {_pp(short_prev)} ({n_short_z}/{n_short:,}). **{q6}** — {why}"
    )
    print(prose)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "now_sh": now_sh,
        "lag1_sh": lag1_sh,
        "lag3_sh": lag3_sh,
        "short_now": short_now,
        "short_lag1": short_lag1,
        "long_lag1": long_lag1,
        "short_prev": short_prev,
        "empty_short": empty_short,
        "keep_q6": keep_q6,
        "q6": q6,
        "why": why,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 13 — amount>0 vs a_op_in==0 (CAT_MAP hole)
# ---------------------------------------------------------------------------
def pass13_opin_hole(tr: pd.DataFrame) -> dict:
    """amount>0 is any category; a_op_in is CAT_MAP op_in only."""
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce") == 1
    opin0 = pd.to_numeric(tr["a_op_in"], errors="coerce").fillna(0) == 0
    n = int(len(tr))
    both = int((z & opin0).sum())
    z_only = int((z & ~opin0).sum())  # should be ~0 (inflow without op_in? wait: zero-in AND op_in>0 impossible)
    opin0_only = int((~z & opin0).sum())  # has amount>0 but a_op_in==0 (transfer/invest in)
    neither = int((~z & ~opin0).sum())
    rows = [
        {"cell": "zero-in ∩ a_op_in==0", "n_cm": both, "share": _pp(_pct(both, n))},
        {"cell": "zero-in ∩ a_op_in>0", "n_cm": z_only, "share": _pp(_pct(z_only, n))},
        {"cell": "has amount>0 ∩ a_op_in==0", "n_cm": opin0_only, "share": _pp(_pct(opin0_only, n))},
        {"cell": "has amount>0 ∩ a_op_in>0", "n_cm": neither, "share": _pp(_pct(neither, n))},
    ]
    lab = tr[Y3].notna()
    hole = signed_oof_auroc(tr[Y3], (~z & opin0).astype(float), tr["fold"], lab)
    prose = (
        f"amount>0 ∩ a_op_in==0 (transfer/invest-in, not CAT_MAP op_in): {opin0_only:,} "
        f"({_pp(_pct(opin0_only, n))}). zero-in ∩ a_op_in>0 = {z_only} (must be 0). "
        f"Y3 of that hole flag {_f(cv_of(hole))}. "
        "Zero-in is any-category inflow, not the op_in bucket Y6 used."
    )
    print(prose)
    return {
        "both": both,
        "z_only": z_only,
        "opin0_only": opin0_only,
        "neither": neither,
        "hole_cv": cv_of(hole),
        "rows": rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 14 — calendar of empty vs all-out
# ---------------------------------------------------------------------------
def pass14_calendar(tr: pd.DataFrame) -> dict:
    rows = []
    for m in range(1, 13):
        sl = tr[tr["cal_month"] == m]
        rows.append(
            {
                "month": pd.Timestamp(2000, m, 1).strftime("%b"),
                "n_cm": int(len(sl)),
                "zero_in": float(pd.to_numeric(sl["c_zero_in_month"], errors="coerce").mean()),
                "empty": float(pd.to_numeric(sl["empty_month"], errors="coerce").mean()),
                "all_out": float(pd.to_numeric(sl["all_out"], errors="coerce").mean()),
            }
        )
    peak = max(rows, key=lambda r: r["zero_in"] if np.isfinite(r["zero_in"]) else -1)
    trough = min(rows, key=lambda r: r["zero_in"] if np.isfinite(r["zero_in"]) else 9)
    aug = next(r for r in rows if r["month"] == "Aug")
    calendarish = bool(peak["zero_in"] >= trough["zero_in"] + 0.08)
    prose = (
        f"Zero-in peak {peak['month']} {_pp(peak['zero_in'])}, trough {trough['month']} "
        f"{_pp(trough['zero_in'])}. August {_pp(aug['zero_in'])}. "
        f"{'Calendar-shaped (vacation hole?)' if calendarish else 'Not a strong calendar dummy'}."
    )
    print(prose)
    return {
        "rows": [
            {
                "month": r["month"],
                "n_cm": f"{r['n_cm']:,}",
                "zero_in": _pp(r["zero_in"]),
                "empty": _pp(r["empty"]),
                "all_out": _pp(r["all_out"]),
            }
            for r in rows
        ],
        "raw": rows,
        "peak": peak,
        "trough": trough,
        "calendarish": calendarish,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 15 — holdout coverage only
# ---------------------------------------------------------------------------
def pass15_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col in ("c_zero_in_month", "c_zero_in_share_6", "empty_month", "all_out"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "mean": _pp(float(x.mean()) if x.notna().any() else float("nan"))
                if col != "c_zero_in_share_6"
                else _f(float(x.mean()) if x.notna().any() else float("nan"), 3),
            }
        )
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM / {ho['company_id'].nunique()} cos. "
        f"zero-in mean {_pp(float(pd.to_numeric(ho['c_zero_in_month'], errors='coerce').mean()))}; "
        f"empty {_pp(float(pd.to_numeric(ho['empty_month'], errors='coerce').mean()))}; "
        f"all-out {_pp(float(pd.to_numeric(ho['all_out'], errors='coerce').mean()))}. "
        "No AUROC claim."
    )
    print(prose)
    return {"rows": rows, "n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 16 — Y rates on empty / all-out / has-in
# ---------------------------------------------------------------------------
def pass16_rates(tr: pd.DataFrame) -> dict:
    rows = []
    for y in (Y2, Y3):
        for sname, mask in (
            ("has_in", tr["has_in"] == 1),
            ("empty", tr["empty_month"] == 1),
            ("all_out", tr["all_out"] == 1),
            ("zero_in", pd.to_numeric(tr["c_zero_in_month"], errors="coerce") == 1),
        ):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n_cm": int(mask.sum()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                }
            )
    y3_em = next(r for r in rows if r["y"] == Y3 and r["slice"] == "empty")
    y3_ao = next(r for r in rows if r["y"] == Y3 and r["slice"] == "all_out")
    y3_hi = next(r for r in rows if r["y"] == Y3 and r["slice"] == "has_in")
    prose = (
        f"Y3 rate: empty {_pp(y3_em['rate'])} (n_lab={y3_em['n_lab']}) vs "
        f"all-out {_pp(y3_ao['rate'])} vs has-in {_pp(y3_hi['rate'])}. "
        "A leftover Q3 going-quiet would show all-out ≠ empty. Inverse-activity would pile both."
    )
    print(prose)
    return {"rows": rows, "y3_em": y3_em["rate"], "y3_ao": y3_ao["rate"], "y3_hi": y3_hi["rate"], "prose": prose}


# ---------------------------------------------------------------------------
# Extra 17 — all-out category mix (raw txs, train)
# ---------------------------------------------------------------------------
def pass17_allout_mix(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    keys = tr.loc[tr["all_out"] == 1, ["company_id", "period"]]
    if keys.empty:
        return {"rows": [], "prose": "No all-out months.", "n_tx": 0}
    raw = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          category,
          COUNT(*) AS n,
          SUM(amount) AS amt
        FROM transactions
        WHERE "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2, 3
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["month"])
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    hit = raw.merge(keys, on=["company_id", "period"], how="inner")
    mix = (
        hit.groupby("category", as_index=False)
        .agg(n_tx=("n", "sum"), amt=("amt", "sum"), n_cm=("period", "size"))
        .sort_values("n_tx", ascending=False)
    )
    tot = float(mix["n_tx"].sum()) if len(mix) else 0.0
    rows = []
    for _, r in mix.head(12).iterrows():
        rows.append(
            {
                "category": r["category"],
                "n_tx": int(r["n_tx"]),
                "share_tx": _pp(_pct(r["n_tx"], tot)),
                "n_cm": int(r["n_cm"]),
                "net_amt": f"{float(r['amt']):,.0f}",
            }
        )
    top = mix.iloc[0]["category"] if len(mix) else "—"
    prose = (
        f"All-out months: {len(keys):,} CM, {int(tot):,} txs. Top token `{top}`. "
        "If the pile is payroll/tax/fee outflows only, it is category-all-out, not an empty book."
    )
    print(prose)
    return {"rows": rows, "n_tx": int(tot), "n_cm": int(len(keys)), "top": str(top), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 18 — ever-zero-in companies (trait pile)
# ---------------------------------------------------------------------------
def pass18_ever(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_z=("c_zero_in_month", "sum"),
        n_em=("empty_month", "sum"),
        n_ao=("all_out", "sum"),
        share6=("c_zero_in_share_6", "mean"),
        size=("log_in3", "median"),
    )
    ever["kind"] = np.select(
        [
            ever["n_z"] == 0,
            (ever["n_em"] >= ever["n_ao"]) & (ever["n_z"] >= 3),
            (ever["n_ao"] > ever["n_em"]) & (ever["n_z"] >= 3),
        ],
        ["never", "mostly_empty", "mostly_allout"],
        default="rare",
    )
    rows = []
    for k, part in ever.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_co": int(len(part)),
                "share_co": _pp(_pct(len(part), len(ever))),
                "z_rate_p50": _f(float((part["n_z"] / part["n_cm"]).median()), 3),
                "size_p50": _f(float(part["size"].median()), 3),
            }
        )
    n_never = int((ever["kind"] == "never").sum())
    n_em = int((ever["kind"] == "mostly_empty").sum())
    n_ao = int((ever["kind"] == "mostly_allout").sum())
    n_rare = int((ever["kind"] == "rare").sum())
    prose = (
        f"Train companies: never zero-in {n_never}, mostly-empty {n_em}, "
        f"mostly-all-out {n_ao}, rare {n_rare} / {len(ever)}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_never": n_never,
        "n_em": n_em,
        "n_ao": n_ao,
        "n_rare": n_rare,
        "n_co": int(len(ever)),
        "ever": ever,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 19 — leftover of share_6 after the month flag (is the trail anything?)
# ---------------------------------------------------------------------------
def pass19_share_after_month(tr: pd.DataFrame) -> dict:
    sh = pd.to_numeric(tr["c_zero_in_share_6"], errors="coerce")
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    resid, info = ols_resid(sh, z)
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        raw = signed_oof_auroc(tr[y], sh, tr["fold"], lab)
        res = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        mon = signed_oof_auroc(tr[y], z, tr["fold"], lab)
        store[y] = {"raw": raw, "resid": res, "month": mon}
        rows.append(
            {
                "y": y,
                "share_6": _f(cv_of(raw)),
                "month": _f(cv_of(mon)),
                "share_after_month": _f(cv_of(res)),
                "n_pos": f"{res['n_pos']:,}",
            }
        )
    y3 = store[Y3]
    lives = bool(np.isfinite(cv_of(y3["resid"])) and cv_of(y3["resid"]) >= 0.55)
    prose = (
        f"Y3 share_6 {_f(cv_of(y3['raw']))} after the month flag {_f(cv_of(y3['resid']))} "
        f"(slope {_f(info['slope'][0]) if info['slope'] else float('nan')}). "
        f"{'Trail leftover lives' if lives else 'share_6 is the month flag smoothed — no extra trail'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_resid": cv_of(y3["resid"]),
        "y3_raw": cv_of(y3["raw"]),
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — stressed-only Y3 (the 15-col card population)
# ---------------------------------------------------------------------------
def pass20_stressed(tr: pd.DataFrame) -> dict:
    """Y3 is already stressed-only in the store (NaN off-stress). This is the labeled slice."""
    lab = tr[Y3].notna()
    sl = tr[lab]
    rows = []
    for feat in (
        "c_zero_in_month",
        "c_zero_in_share_6",
        "empty_month",
        "all_out",
        "c_n_days_with_tx",
        "a_n_tx",
        "log_in3",
    ):
        res = signed_oof_auroc(tr[Y3], tr[feat], tr["fold"], lab)
        prev = float(pd.to_numeric(sl[feat], errors="coerce").mean()) if len(sl) else float("nan")
        rows.append(
            {
                "feature": feat,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "mean": _f(prev, 3),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    n_z = int((pd.to_numeric(sl["c_zero_in_month"], errors="coerce") == 1).sum())
    n_em = int((sl["empty_month"] == 1).sum())
    n_ao = int((sl["all_out"] == 1).sum())
    prose = (
        f"Y3-labeled (stressed) CM {int(lab.sum()):,}: zero-in {n_z:,} empty {n_em:,} all-out {n_ao:,}. "
        "This is the card population. Zero-in is not on the 15-col card."
    )
    print(prose)
    return {"rows": rows, "n_z": n_z, "n_em": n_em, "n_ao": n_ao, "n_lab": int(lab.sum()), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 21 — empty vs all-out leftover among busy months only
# ---------------------------------------------------------------------------
def pass21_busy_split(tr: pd.DataFrame) -> dict:
    """Among n_tx>0, is all-out leftover after days / size?"""
    busy = tr["empty_month"] == 0
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna() & busy
        for feat, col in (
            ("all_out", tr["all_out"]),
            ("c_zero_in_month", tr["c_zero_in_month"]),
            ("log1p_a_in3", tr["log_in3"]),
            ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
            ("a_n_tx", tr["a_n_tx"]),
            ("a_op_in", tr["a_op_in"]),
        ):
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, feat)] = res
            rows.append(
                {
                    "y": y,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    ao = store[(Y3, "all_out")]
    size = store[(Y3, "log1p_a_in3")]
    days = store[(Y3, "c_n_days_with_tx")]
    beat = (
        ao["cv"] - size["cv"]
        if (not ao["low_power"] and not size["low_power"])
        else float("nan")
    )
    keep_busy = bool(np.isfinite(beat) and beat >= KEEP_DELTA)
    prose = (
        f"Busy-only (n_tx>0) Y3: all-out {_f(cv_of(ao))} vs size {_f(cv_of(size))} "
        f"(Δ {_f(beat)}) vs days {_f(cv_of(days))}. "
        f"{'Would KEEP all-out on the busy slice' if keep_busy else 'All-out on busy months does not beat size — not a leftover Q3'}."
    )
    print(prose)
    return {
        "rows": rows,
        "ao_y3": cv_of(ao),
        "size_y3": cv_of(size),
        "days_y3": cv_of(days),
        "beat": beat,
        "keep_busy": keep_busy,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 22 — feature-report SIZE clock + leftover after log1p(|a_op_in|)
# ---------------------------------------------------------------------------
def pass22_size_clock(tr: pd.DataFrame) -> dict:
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    sh = pd.to_numeric(tr["c_zero_in_share_6"], errors="coerce")
    r_z, _ = ols_resid(z, tr["log_abs_opin"])
    r_sh, _ = ols_resid(sh, tr["log_abs_opin"])
    r_z_d, _ = ols_resid(z, tr["c_n_days_with_tx"], tr["log_abs_opin"])
    lab = tr[Y3].notna()
    rows = []
    store = {}
    for name, col in (
        ("log1p_|a_op_in|", tr["log_abs_opin"]),
        ("z_resid_opin_sz", r_z),
        ("sh_resid_opin_sz", r_sh),
        ("z_resid_days_opin", r_z_d),
    ):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    opin_y3 = cv_of(store["log1p_|a_op_in|"])
    z_left = cv_of(store["z_resid_opin_sz"])
    sh_left = cv_of(store["sh_resid_opin_sz"])
    prose = (
        f"Y3 log1p(|a_op_in|) {_f(opin_y3)} (feature-report SIZE clock). "
        f"Month leftover after that clock {_f(z_left)}; share_6 leftover {_f(sh_left)}; "
        f"month after days+opin {_f(cv_of(store['z_resid_days_opin']))}. "
        "If leftover dies on the report clock too, the −0.508 flag is the same inverse-activity story."
    )
    print(prose)
    return {
        "rows": rows,
        "opin_y3": opin_y3,
        "z_left": z_left,
        "sh_left": sh_left,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 23 — the 10-row hole: zero-in ∩ a_op_in>0
# ---------------------------------------------------------------------------
def pass23_hole10(tr: pd.DataFrame, con) -> dict:
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce") == 1
    opin = pd.to_numeric(tr["a_op_in"], errors="coerce")
    hole = z & (opin.fillna(0) != 0)
    n = int(hole.sum())
    sl = tr.loc[hole, ["company_id", "period", "a_op_in", "c_n_tx", "empty_month", "all_out", Y3]].copy()
    rows = []
    if n:
        keys = sl[["company_id", "period"]].drop_duplicates()
        raw = con.execute(
            """
            SELECT
              CAST(company_id AS VARCHAR) AS company_id,
              CAST(date_trunc('month', "date") AS DATE) AS month,
              category,
              amount
            FROM transactions
            WHERE "date" IS NOT NULL
              AND "date" >= DATE '2024-09-01'
              AND "date" < DATE '2026-09-01'
            """
        ).df()
        raw["company_id"] = raw["company_id"].astype(str)
        raw["period"] = pd.to_datetime(raw["month"])
        hit = raw.merge(keys, on=["company_id", "period"], how="inner")
        mix = (
            hit.groupby("category", as_index=False)
            .agg(n=("amount", "size"), amt=("amount", "sum"), mx=("amount", "max"))
            .sort_values("n", ascending=False)
        )
        for _, r in mix.head(8).iterrows():
            rows.append(
                {
                    "category": r["category"],
                    "n_tx": int(r["n"]),
                    "net": f"{float(r['amt']):,.2f}",
                    "max_amt": f"{float(r['mx']):,.2f}",
                }
            )
        max_opin = float(sl["a_op_in"].max())
        med_opin = float(sl["a_op_in"].median())
        n_empty = int((sl["empty_month"] == 1).sum())
        prose = (
            f"zero-in ∩ a_op_in≠0: {n} train CM. "
            f"a_op_in p50={med_opin:,.2f} max={max_opin:,.2f}; empty among them {n_empty}. "
            "Signed CAT_MAP op_in without any amount>0 (refunds / dust). "
            "Do not patch ops.py — store vs raw amount>0 already agreed 100%."
        )
    else:
        max_opin = float("nan")
        med_opin = float("nan")
        n_empty = 0
        prose = "zero-in ∩ a_op_in≠0 is empty — amount>0 and CAT_MAP op_in agree."
    print(prose)
    return {
        "n": n,
        "n_empty": n_empty,
        "max_opin": max_opin,
        "med_opin": med_opin,
        "rows": rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 24 — T1 leftover after days (the only tercile that beat size)
# ---------------------------------------------------------------------------
def pass24_t1_leftover(tr: pd.DataFrame, p7: dict) -> dict:
    m = p7.get("m")
    if m is None:
        return {"prose": "no tercile frame", "lives": False, "rows": []}
    sl = m[m["size_t"].astype(str) == "T1"].copy()
    z = pd.to_numeric(sl["c_zero_in_month"], errors="coerce")
    sh = pd.to_numeric(sl["c_zero_in_share_6"], errors="coerce")
    r_z, _ = ols_resid(z, sl["c_n_days_with_tx"])
    r_sh, _ = ols_resid(sh, sl["c_n_days_with_tx"])
    lab = sl[Y3].notna()
    rows = []
    store = {}
    for name, col in (
        ("c_zero_in_month", z),
        ("c_zero_in_share_6", sh),
        ("log1p_a_in3", sl["log_in3"]),
        ("c_n_days_with_tx", sl["c_n_days_with_tx"]),
        ("z_resid_days", r_z),
        ("sh_resid_days", r_sh),
        ("empty_month", sl["empty_month"]),
        ("all_out", sl["all_out"]),
    ):
        res = signed_oof_auroc(sl[Y3], col, sl["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    beat = (
        (cv_of(store["z_resid_days"]) - cv_of(store["log1p_a_in3"]))
        if np.isfinite(cv_of(store["z_resid_days"])) and np.isfinite(cv_of(store["log1p_a_in3"]))
        else float("nan")
    )
    z_left = cv_of(store["z_resid_days"])
    days_cv = cv_of(store["c_n_days_with_tx"])
    size_cv = cv_of(store["log1p_a_in3"])
    # T1 size is often inverted (~0.43). Beating a broken size clock is not KEEP.
    lives = bool(
        np.isfinite(z_left)
        and z_left >= 0.60
        and np.isfinite(days_cv)
        and z_left >= days_cv
        and np.isfinite(beat)
        and beat >= KEEP_DELTA
    )
    prose = (
        f"T1-only Y3: month {_f(cv_of(store['c_zero_in_month']))} leftover-days "
        f"{_f(z_left)} vs size {_f(size_cv)} "
        f"(Δ {_f(beat)}) vs days {_f(days_cv)}. "
        f"{'T1 leftover would KEEP' if lives else 'T1 leftover loses to days (size clock is inverted inside T1) — still inverse-activity'}."
    )
    print(prose)
    return {"rows": rows, "beat": beat, "lives": lives, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 25 — Y6 size AUROC (confirm inverse-size failure)
# ---------------------------------------------------------------------------
def pass25_y6_size(tr: pd.DataFrame) -> dict:
    if Y6 not in tr.columns:
        return {"prose": "Y6 missing", "size_auc": float("nan"), "rows": []}
    y = pd.to_numeric(tr[Y6], errors="coerce")
    lab = y.notna()
    rows = []
    store = {}
    for name, col in (
        ("log1p_a_in3", tr["log_in3"]),
        ("log1p_|a_op_in|", tr["log_abs_opin"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("c_zero_in_month", tr["c_zero_in_month"]),
        ("empty_month", tr["empty_month"]),
        ("all_out", tr["all_out"]),
    ):
        res = signed_oof_auroc(y, col, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "sign": res["train_sign"] if not res["low_power"] else "—",
            }
        )
    size_auc = cv_of(store["log1p_a_in3"])
    opin_auc = cv_of(store["log1p_|a_op_in|"])
    inv = bool((np.isfinite(size_auc) and size_auc >= SIZE_PARK) or (np.isfinite(opin_auc) and opin_auc >= SIZE_PARK))
    p_y6_z = float(y[lab & (tr["c_zero_in_month"] == 1)].mean()) if (lab & (tr["c_zero_in_month"] == 1)).any() else float("nan")
    p_y6_em = float(y[lab & (tr["empty_month"] == 1)].mean()) if (lab & (tr["empty_month"] == 1)).any() else float("nan")
    p_y6_ao = float(y[lab & (tr["all_out"] == 1)].mean()) if (lab & (tr["all_out"] == 1)).any() else float("nan")
    prose = (
        f"Y6 vs size {_f(size_auc)} vs log1p(|a_op_in|) {_f(opin_auc)} "
        f"({'CONFIRM inverse-size / Y6 failure mode' if inv else 'size AUROC < 0.60 on this fold'}). "
        f"P(Y6|zero-in)={_pp(p_y6_z)} empty {_pp(p_y6_em)} all-out {_pp(p_y6_ao)}. "
        "Do not revive Y6."
    )
    print(prose)
    return {
        "rows": rows,
        "size_auc": size_auc,
        "opin_auc": opin_auc,
        "inv": inv,
        "p_y6_z": p_y6_z,
        "p_y6_em": p_y6_em,
        "p_y6_ao": p_y6_ao,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 26 — all-out resid 0.614: fake leftover?
# ---------------------------------------------------------------------------
def pass26_ao_fake(tr: pd.DataFrame) -> dict:
    ao = pd.to_numeric(tr["all_out"], errors="coerce")
    resid, info = ols_resid(ao, tr["c_n_days_with_tx"])
    rho = spearman(resid, tr["c_n_days_with_tx"])
    lab = tr[Y3].notna()
    rec = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
    fake = bool(np.isfinite(rho) and abs(rho) >= 0.50)
    prose = (
        f"All-out OLS leftover after days Y3 {_f(cv_of(rec))} "
        f"ρ(resid, days)={_f(rho)} slope={_f(info['slope'][0]) if info['slope'] else float('nan')}. "
        f"{'FAKE leftover — resid still tracks days' if fake else 'resid is orthogonal to days'}. "
        "Busy-only all-out 0.554 is the honest number."
    )
    print(prose)
    return {
        "cv": cv_of(rec),
        "rho": rho,
        "fake": fake,
        "slope": info["slope"][0] if info["slope"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 27 — share_6 leftover after days + size together
# ---------------------------------------------------------------------------
def pass27_share_days_size(tr: pd.DataFrame) -> dict:
    sh = pd.to_numeric(tr["c_zero_in_share_6"], errors="coerce")
    z = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    r_sh, _ = ols_resid(sh, tr["c_n_days_with_tx"], tr["log_in3"])
    r_z, _ = ols_resid(z, tr["c_n_days_with_tx"], tr["log_in3"])
    mu = sh.groupby(tr["company_id"], sort=False).transform("mean")
    dem = sh - mu
    r_dem, _ = ols_resid(dem, tr["c_n_days_with_tx"])
    lab = tr[Y3].notna()
    rows = []
    store = {}
    for name, col in (
        ("sh_resid_days_size", r_sh),
        ("z_resid_days_size", r_z),
        ("sh_demean", dem),
        ("sh_demean_resid_days", r_dem),
    ):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    lives = bool(np.isfinite(cv_of(store["sh_resid_days_size"])) and cv_of(store["sh_resid_days_size"]) >= 0.55)
    prose = (
        f"share_6 leftover after days+size {_f(cv_of(store['sh_resid_days_size']))}; "
        f"month after days+size {_f(cv_of(store['z_resid_days_size']))}; "
        f"demeaned share after days {_f(cv_of(store['sh_demean_resid_days']))} "
        f"(raw demean {_f(cv_of(store['sh_demean']))}). "
        f"{'Still a leftover' if lives else 'Double residual dies — quiet twin + SIZE, nothing left'}."
    )
    print(prose)
    return {
        "rows": rows,
        "sh": cv_of(store["sh_resid_days_size"]),
        "z": cv_of(store["z_resid_days_size"]),
        "dem_d": cv_of(store["sh_demean_resid_days"]),
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 28 — demean leftover honesty + uncat overlap
# ---------------------------------------------------------------------------
def pass28_demean_uncat(tr: pd.DataFrame) -> dict:
    sh = pd.to_numeric(tr["c_zero_in_share_6"], errors="coerce")
    mu = sh.groupby(tr["company_id"], sort=False).transform("mean")
    dem = sh - mu
    r_dem, info = ols_resid(dem, tr["c_n_days_with_tx"])
    rho_days = spearman(r_dem, tr["c_n_days_with_tx"])
    rho_size = spearman(dem, tr["log_in3"])
    lab = tr[Y3].notna()
    rec = signed_oof_auroc(tr[Y3], r_dem, tr["fold"], lab)
    raw = signed_oof_auroc(tr[Y3], dem, tr["fold"], lab)
    fake = bool(np.isfinite(rho_days) and abs(rho_days) >= 0.50)
    rows = [
        {"item": "Y3 demean", "value": _f(cv_of(raw))},
        {"item": "Y3 demean after days", "value": _f(cv_of(rec))},
        {"item": "ρ(demean-resid, days)", "value": _f(rho_days)},
        {"item": "ρ(demean, log1p(a_in3))", "value": _f(rho_size)},
        {"item": "slope vs days", "value": _f(info["slope"][0]) if info["slope"] else "—"},
    ]
    uncat_rows = []
    if "a_uncat_share" in tr.columns:
        rho_u = spearman(tr["all_out"], tr["a_uncat_share"])
        rho_uz = spearman(tr["c_zero_in_month"], tr["a_uncat_share"])
        ao = tr["all_out"] == 1
        u_ao = float(pd.to_numeric(tr.loc[ao, "a_uncat_share"], errors="coerce").mean()) if ao.any() else float("nan")
        u_hi = float(pd.to_numeric(tr.loc[tr["has_in"] == 1, "a_uncat_share"], errors="coerce").mean())
        uncat_rows = [
            {"item": "ρ(all_out, a_uncat_share)", "value": _f(rho_u)},
            {"item": "ρ(zero-in, a_uncat_share)", "value": _f(rho_uz)},
            {"item": "mean uncat on all-out", "value": _pp(u_ao)},
            {"item": "mean uncat on has-in", "value": _pp(u_hi)},
        ]
        twin_u = bool(np.isfinite(rho_u) and abs(rho_u) >= TWIN_RHO)
        uncat_note = (
            f"All-out vs uncat ρ={_f(rho_u)} "
            f"({'TWIN of uncat' if twin_u else 'not an uncat twin'}); "
            f"uncat share all-out {_pp(u_ao)} vs has-in {_pp(u_hi)}."
        )
    else:
        uncat_note = "a_uncat_share not in store."
        twin_u = False
        rho_u = float("nan")
    lives = bool(np.isfinite(cv_of(rec)) and cv_of(rec) >= 0.60 and not fake)
    prose = (
        f"Demeaned share_6 after days {_f(cv_of(rec))} "
        f"ρ(resid,days)={_f(rho_days)} "
        f"({'FAKE days leftover' if fake else 'resid orthogonal to days'}). "
        f"{uncat_note} "
        f"{'Demean leftover would KEEP as a transform — still not the stored column, not on the 44.' if lives else 'Demean leftover is not a KEEP transform of the stored X.'}"
    )
    print(prose)
    return {
        "rows": rows + uncat_rows,
        "dem_d": cv_of(rec),
        "rho_days": rho_days,
        "fake": fake,
        "lives": lives,
        "rho_u": rho_u,
        "prose": prose,
    }


def make_png(p7: dict, p2: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    m = p7.get("m")
    if m is None or m.empty:
        print("no tercile frame — skip PNG")
        return False
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    ax = axes[0]
    labels = ["T1 small", "T2", "T3 large"]
    empty = []
    allout = []
    hasin = []
    for labv in ("T1", "T2", "T3"):
        sl = m[m["size_t"].astype(str) == labv]
        n = max(len(sl), 1)
        empty.append(100.0 * float((sl["empty_month"] == 1).mean()))
        allout.append(100.0 * float((sl["all_out"] == 1).mean()))
        hasin.append(100.0 * float((sl["has_in"] == 1).mean()))
    x = np.arange(3)
    ax.bar(x, empty, color="#9e6b4a", label="empty (n_tx=0)")
    ax.bar(x, allout, bottom=empty, color="#1f4e79", label="all-out (txs, no amount>0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of train company-months")
    ax.set_title("Zero-in pile by size tercile")
    ax.legend(fontsize=8, loc="upper right")

    ax2 = axes[1]
    piles = ["has-in", "empty", "all-out"]
    counts = [p2["n_has"], p2["n_empty"], p2["n_allout"]]
    colors = ["#7aa36a", "#9e6b4a", "#1f4e79"]
    ax2.bar(piles, [100.0 * c / p2["n"] for c in counts], color=colors)
    ax2.set_ylabel("% of train CM")
    ax2.set_title(f"Flag pile: {p2['pile']}")
    for i, c in enumerate(counts):
        ax2.text(i, 100.0 * c / p2["n"] + 0.4, f"{c:,}", ha="center", fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p4, p5, p6, p11, p12) -> dict:
    """KEEP on the 44 / DROP-from-44 / CLOSE as X / PARK as health Y."""
    size_flag = bool(
        p4["size_m"]
        or p4["size_s"]
        or p4.get("size_m_opin")
        or p4.get("size_s_opin")
    )
    twin = bool(p4["twin_m"])
    leftover_dies = bool(p6["quiet_twin"] or (not p6["lives_z"] and not p6["lives_sh"] and not p6["lives_ao"]))
    leftover_ok = bool((p6["lives_z"] or p6["lives_sh"] or p6["lives_ao"]) and not size_flag)
    keep_44 = bool(leftover_ok and not twin)
    if p12.get("q6") == "KEEP" and leftover_dies:
        p12["q6"] = "CLOSE"
        p12["keep_q6"] = False
        p12["why"] = (
            f"leftover after days died; contemporaneous {_f(p12['now'])} loses to size "
            f"{_f(p5['size_y3'])} so lag1 {_f(p12['lag1'])} has nothing honest to lead."
        )
    if keep_44:
        x_month = "KEEP"
        x_share = "KEEP"
        why = (
            f"leftover after days beats size ≥0.02 "
            f"(month Δ {_f(p6['leftover_z'])} share Δ {_f(p6['leftover_sh'])}) "
            "and is not SIZE. Still not on tonight's 15-col card."
        )
    else:
        bits = []
        if size_flag:
            bits.append(
                f"SIZE (month vs a_in3 {_f(p4['rho_m_size'])} vs |a_op_in| {_f(p4.get('rho_m_opin_sz', float('nan')))}; "
                f"share vs a_in3 {_f(p4['rho_s_size'])} vs |a_op_in| {_f(p4.get('rho_s_opin_sz', float('nan')))})"
            )
        if twin:
            bits.append(f"days/n_tx twin (ρ days={_f(p4['rho_m_days'])} n_tx={_f(p4['rho_m_ntx'])})")
        if leftover_dies:
            bits.append(
                f"leftover after days dies (month {_f(p6['y3_z_d'])} share {_f(p6['y3_sh_d'])} "
                f"Δsize {_f(p6['leftover_z'])})"
            )
        why = "; ".join(bits) if bits else (
            f"Y3 month {_f(p5['m_y3'])} does not beat size {_f(p5['size_y3'])} by ≥0.02"
        )
        x_month = "DROP from the 44"
        x_share = "DROP from the 44"
    close_x = not keep_44
    return {
        "keep_44": keep_44,
        "x_month": x_month,
        "x_share": x_share,
        "close_x": close_x,
        "park_y": True,
        "why": why,
        "q3": "KEEP" if keep_44 else "CLOSE",
        "q6": p12["q6"],
        "size": "YES" if size_flag else "NO",
        "twin": "YES" if twin else "NO",
        "leftover": "lives" if leftover_ok else "dies",
        "trait": "YES" if p11.get("trait") else "NO",
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12, p13, p14, p15 = ctx["p11"], ctx["p12"], ctx["p13"], ctx["p14"], ctx["p15"]
    p16, p17, p18, p19, p20, p21 = (
        ctx["p16"],
        ctx["p17"],
        ctx["p18"],
        ctx["p19"],
        ctx["p20"],
        ctx["p21"],
    )
    d = ctx["decision"]
    lines = [
        "# Q3 zero-in — leftover after days, or SIZE / inverse-activity?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_zero_in`. Do not revive `y6_zero_in_3`. "
        "Do not put zero-in on the 15-col card. Night Y3 quote stays 0.762 / 0.752. Days bar 0.711.",
        "",
        "`c_zero_in_month` = 1 if no row with amount > 0 (any category). "
        "`c_zero_in_share_6` = mean of that flag over last ≤6 months (min_periods=1). "
        "Feature report: rare-event flag, modal 88.2%, size ρ −0.508 / −0.519.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Not this flag. PARK as a health Y. Never-zero-in {p18['n_never']} / mostly-empty {p18['n_em']} / mostly-all-out {p18['n_ao']}. |",
        "| 2 | Who is improving? | Not this table. |",
        f"| 3 | Who is turning? | **{d['q3']}** — empty vs all-out is {p2['pile']}. Leftover after days {d['leftover']}. SIZE {d['size']}. |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | Inverse activity / size (Y6 failure mode), not a leftover quiet after days. |",
        f"| 6 | Months earlier? | **{d['q6']}** — {p12['why']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `c_zero_in_month` on the 44 | **{d['x_month']}** | {d['why']} |",
        f"| `c_zero_in_share_6` on the 44 | **{d['x_share']}** | {d['why']} |",
        "| either as a health Y | **PARK** | do not invent `y_zero_in`; do not revive Y6 |",
        "| either on tonight's 15-col card | **no** | night engine stays days 0.711 / n_tx / salary |",
        f"| SIZE (\\|ρ\\| ≥ 0.50) | **{d['size']}** | KEEP clock log1p(a_in3) month {_f(p4['rho_m_size'])} share {_f(p4['rho_s_size'])}; report clock log1p(\\|a_op_in\\|) month {_f(p4.get('rho_m_opin_sz', float('nan')))} share {_f(p4.get('rho_s_opin_sz', float('nan')))} |",
        f"| days / n_tx twin (\\|ρ\\|≥0.80) | **{d['twin']}** | days {_f(p4['rho_m_days'])} n_tx {_f(p4['rho_m_ntx'])} |",
        f"| leftover after days | **{d['leftover']}** | month {_f(p6['y3_z_d'])} share {_f(p6['y3_sh_d'])} Δsize {_f(p6['leftover_z'])} |",
        f"| Q6 lag1 / lag3 | **{d['q6']}** | {p12['prose']} |",
        "| Y6 `y6_zero_in_3` | **PARK / do not revive** | " + p8["prose"] + " |",
        "",
        "## 1. Prevalence + modal 88.2% + holdout coverage",
        "",
        p1["prose"],
        "",
        _md_table(
            [
                {
                    "split": r["split"],
                    "col": r["col"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "cov": _pp(r["cov"]),
                    "mean": _pp(r["mean"]) if r["col"] != "c_zero_in_share_6" else _f(r["mean"], 3),
                    "modal share": _pp(r["modal_share"]),
                }
                for r in p1["rows"]
            ]
        ),
        "",
        "## 2. Empty grid vs all-out — which pile is the flag?",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 3. Formula vs raw txs (any amount>0)",
        "",
        p3["prose"],
        "",
        "## 4. Spearman vs days / n_tx / a_op_in / size / recency",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p5['n_y2']:,} base {_pp(p5['y2_rate'])}; "
        f"Y3 stressed n={p5['n_y3']:,} base {_pp(p5['y3_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p5['days_y3'])}); size 0.617 (replica {_f(p5['size_y3'])}).",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        f"KEEP-as-X on the 44: leftover after days beats size by ≥{KEEP_DELTA:g} **and** not SIZE (|ρ|≥{SIZE_RHO:g}). "
        "Still do not put it on tonight's 15-col card.",
        "",
        "## 6. Residual after days and after `a_n_tx`",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. SIZE terciles",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. vs `y6_zero_in_3` (do not revive)",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]) if p8.get("rows") else "_(Y6 missing)_\n",
        "",
        "## 9. Dark 470 vs invoiced 744",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        _md_table(
            [
                {
                    "group": r["group"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "zero": _pp(r["zero"]),
                    "empty": _pp(r["empty"]),
                    "all_out": _pp(r["all_out"]),
                }
                for r in p9["cm"]
            ]
        ),
        "",
        "## 10. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. ICC / company-demean (share_6 acf1 0.87)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Q6 — lag1 / lag3 on short vs long books",
        "",
        p12["prose"],
        "",
        _md_table(p12["rows"]),
        "",
        "## 13. amount>0 vs `a_op_in==0` (CAT_MAP hole)",
        "",
        p13["prose"],
        "",
        _md_table(p13["rows"]),
        "",
        "## 14. Calendar of empty vs all-out",
        "",
        p14["prose"],
        "",
        _md_table(p14["rows"]),
        "",
        "## 15. Holdout coverage only (no AUROC)",
        "",
        p15["prose"],
        "",
        _md_table(p15["rows"]),
        "",
        "## 16. Y rates on empty / all-out / has-in",
        "",
        p16["prose"],
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
                }
                for r in p16["rows"]
            ]
        ),
        "",
        "## 17. All-out category mix (raw txs)",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "## 18. Ever-zero-in companies",
        "",
        p18["prose"],
        "",
        _md_table(p18["rows"]),
        "",
        "## 19. share_6 leftover after the month flag",
        "",
        p19["prose"],
        "",
        _md_table(p19["rows"]),
        "",
        "## 20. Stressed / Y3-labeled population (the card)",
        "",
        p20["prose"],
        "",
        _md_table(p20["rows"]),
        "",
        "## 21. Busy-only leftover (n_tx>0)",
        "",
        p21["prose"],
        "",
        _md_table(p21["rows"]),
        "",
        "## 22. Feature-report SIZE clock `log1p(|a_op_in|)`",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## 23. The 10-row hole — zero-in ∩ `a_op_in>0`",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]) if ctx["p23"].get("rows") else "_(empty)_\n",
        "",
        "## 24. T1 leftover after days",
        "",
        ctx["p24"]["prose"],
        "",
        _md_table(ctx["p24"]["rows"]),
        "",
        "## 25. Y6 vs size (inverse-size failure mode)",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## 26. All-out resid 0.614 — fake leftover?",
        "",
        ctx["p26"]["prose"],
        "",
        "## 27. share_6 leftover after days + size",
        "",
        ctx["p27"]["prose"],
        "",
        _md_table(ctx["p27"]["rows"]),
        "",
        "## 28. Demean leftover honesty + uncat overlap",
        "",
        ctx["p28"]["prose"],
        "",
        _md_table(ctx["p28"]["rows"]),
        "",
        "## What we did not do",
        "",
        "- Did not edit `ops.py`, `y6_activity.py`, recency/gap_sd/transfer QA, `product/`, parquet / duckdb.",
        "- Did not run `python -m analysis.targets.build_targets`.",
        "- Did not write the parent journal, a 0–100 formula, or a new Y.",
        "- Did not revive `y6_zero_in_3` or put zero-in on the 15-col card.",
        f"- Did not change the night Y3 quote {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f} or days 0.711.",
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Must-do 1–12 plus opin-hole, calendar, holdout, Y rates, all-out mix, ever-kind, share-after-month, stressed card, busy leftover, report SIZE clock, 10-row hole, T1 leftover, Y6 size, fake all-out resid, days+size residual.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p2, p4, p5, p6, p8, p11, p12, d = (
        ctx["p1"],
        ctx["p2"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p8"],
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
            "metric": "c_zero_in_month_prev",
            "value": p1["month"]["mean"],
            "coverage": "1.0000",
            "notes": f"modal={p1['month']['modal_share']:.4f} confirm882={p1['confirm_modal']} pile={p2['pile']} empty={p2['n_empty']} allout={p2['n_allout']}",
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
            "metric": "auroc_c_zero_in_month",
            "value": p5["m_y3"],
            "coverage": "1.0000",
            "notes": f"size={p5['size_y3']:.4f} days={p5['days_y3']:.4f} beat_size={p5['beat_m']:.4f} x={d['x_month']} leftover={p6['y3_z_d']:.4f}",
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
            "metric": "auroc_c_zero_in_share_6",
            "value": p5["s_y3"],
            "coverage": "1.0000",
            "notes": f"size={p5['size_y3']:.4f} leftover_days={p6['y3_sh_d']:.4f} x={d['x_share']}",
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
            "value": p5["days_y3"],
            "coverage": "1.0000",
            "notes": f"night=0.711 replica; size={p5['size_y3']:.4f}",
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
            "metric": "auroc_zero_in_resid_days",
            "value": p6["y3_z_d"],
            "coverage": "1.0000",
            "notes": f"share_resid={p6['y3_sh_d']:.4f} ao_busy={p6['ao_busy']:.4f} leftover_z={p6['leftover_z']:.4f} quiet_twin={p6['quiet_twin']}",
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
            "metric": "rho_zero_in_vs_size",
            "value": p4["rho_m_size"],
            "coverage": "1.0000",
            "notes": f"share={p4['rho_s_size']:.4f} days={p4['rho_m_days']:.4f} ntx={p4['rho_m_ntx']:.4f} SIZE={d['size']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y6,
            "model": MODEL,
            "split": "train",
            "metric": "jaccard_zero_in_vs_y6",
            "value": p8.get("jac", float("nan")),
            "coverage": "1.0000",
            "notes": f"rho={p8.get('rho', float('nan'))} leak={p8.get('leak')} n_both={p8.get('n_both')}",
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
            "metric": "auroc_c_zero_in_month_lag1",
            "value": p12["lag1"],
            "coverage": "1.0000",
            "notes": f"now={p12['now']:.4f} lag3={p12['lag3']:.4f} q6={d['q6']}",
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
            "metric": "icc_c_zero_in_share_6",
            "value": p11["icc_sh"],
            "coverage": "1.0000",
            "notes": f"acf1={p11['acf1_sh']:.3f} confirm087={p11['confirm_acf']} trait={p11['trait']} demean={p11['dem_sh']}",
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
            "metric": "auroc_all_out_busy",
            "value": p6["ao_busy"],
            "coverage": "1.0000",
            "notes": f"empty_of_z={p2['empty_of_z']:.3f} allout_of_z={p2['allout_of_z']:.3f} pile={p2['pile']}",
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
            "metric": "auroc_zero_in_resid_opin_sz",
            "value": ctx["p22"]["z_left"],
            "coverage": "1.0000",
            "notes": f"share_left={ctx['p22']['sh_left']:.4f} opin_y3={ctx['p22']['opin_y3']:.4f} rho_m={p4.get('rho_m_opin_sz', float('nan')):.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y6,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_y6_vs_size",
            "value": ctx["p25"]["size_auc"],
            "coverage": "1.0000",
            "notes": f"opin={ctx['p25']['opin_auc']:.4f} inv={ctx['p25']['inv']} p_y6_z={ctx['p25']['p_y6_z']:.3f}",
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
            "metric": "auroc_share6_resid_days_size",
            "value": ctx["p27"]["sh"],
            "coverage": "1.0000",
            "notes": f"z={ctx['p27']['z']:.4f} dem_days={ctx['p27']['dem_d']:.4f} lives={ctx['p27']['lives']}",
        },
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
    print(f"zero_in_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["c_zero_in_month", "c_zero_in_share_6"],
        (1, 3),
    )
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )

    con = connect()
    try:
        panel = attach_y6_if_missing(panel, con)
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print("pass 1 prevalence")
        p1 = pass1_prev(panel, tr)
        print("pass 2 empty vs all-out")
        p2 = pass2_split(tr)
        print("pass 3 formula")
        p3 = pass3_formula(tr, con)
        print("pass 4 Spearman")
        p4 = pass4_rho(tr)
        print("pass 5 singles")
        p5 = pass5_auroc(tr)
        print("pass 6 leftover")
        p6 = pass6_resid(tr)
        print("pass 7 size terciles")
        p7 = pass7_terciles(tr)
        print("pass 8 vs Y6")
        p8 = pass8_y6(tr)
        print("pass 9 dark 470 vs 744")
        p9 = pass9_dark(tr, con)
        print("pass 10 chronic 12")
        p10 = pass10_chronic(tr)
        print("pass 11 ICC")
        p11 = pass11_icc(tr)
        print("pass 12 Q6")
        p12 = pass12_q6(tr)
        print("pass 13 opin hole")
        p13 = pass13_opin_hole(tr)
        print("pass 14 calendar")
        p14 = pass14_calendar(tr)
        print("pass 15 holdout")
        p15 = pass15_holdout(panel)
        print("pass 16 Y rates")
        p16 = pass16_rates(tr)
        print("pass 17 all-out mix")
        p17 = pass17_allout_mix(tr, con)
        print("pass 18 ever")
        p18 = pass18_ever(tr)
        print("pass 19 share after month")
        p19 = pass19_share_after_month(tr)
        print("pass 20 stressed card")
        p20 = pass20_stressed(tr)
        print("pass 21 busy leftover")
        p21 = pass21_busy_split(tr)
        print("pass 22 SIZE clock")
        p22 = pass22_size_clock(tr)
        print("pass 23 hole 10")
        p23 = pass23_hole10(tr, con)
        print("pass 24 T1 leftover")
        p24 = pass24_t1_leftover(tr, p7)
        print("pass 25 Y6 size")
        p25 = pass25_y6_size(tr)
        print("pass 26 all-out fake leftover")
        p26 = pass26_ao_fake(tr)
        print("pass 27 share after days+size")
        p27 = pass27_share_days_size(tr)
        print("pass 28 demean honesty + uncat")
        p28 = pass28_demean_uncat(tr)
    finally:
        con.close()

    decision = decide(p4, p5, p6, p11, p12)
    png_ok = make_png(p7, p2)
    headline = (
        f"Flag pile **{p2['pile']}** (empty {p2['n_empty']:,} / all-out {p2['n_allout']:,} "
        f"of {p2['n_z']:,} zero-in CM). Modal {_pp(p1['month']['modal_share'])} "
        f"({'CONFIRM 88.2%' if p1['confirm_modal'] else 'off 88.2%'}). "
        f"SIZE month ρ={_f(p4['rho_m_size'])} share ρ={_f(p4['rho_s_size'])}. "
        f"Y3 month {_f(p5['m_y3'])} share_6 {_f(p5['s_y3'])} vs size {_f(p5['size_y3'])} "
        f"(Δ {_f(p5['beat_m'])}) vs days {_f(p5['days_y3'])}. "
        f"Leftover after days month {_f(p6['y3_z_d'])} share {_f(p6['y3_sh_d'])}. "
        f"On the 44: **{decision['x_month']}**. PARK as health Y. Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p1["confirm_modal"]:
        failed.append(f"modal share {_pp(p1['month']['modal_share'])} ≠ 88.2% quote")
    if not p4["confirm_m"]:
        failed.append(
            f"month vs log1p(|a_op_in|) ρ {_f(p4.get('rho_m_opin_sz', float('nan')))} off −0.508 "
            f"(KEEP clock log1p(a_in3)={_f(p4['rho_m_size'])})"
        )
    if not p4["confirm_s"]:
        failed.append(
            f"share_6 vs log1p(|a_op_in|) ρ {_f(p4.get('rho_s_opin_sz', float('nan')))} off −0.519 "
            f"(KEEP clock log1p(a_in3)={_f(p4['rho_s_size'])})"
        )
    if p23["n"] and p3["agree_z"] >= 0.995:
        failed.append(
            f"zero-in ∩ a_op_in>0 n={p23['n']} (CAT_MAP hole, not an ops.py amount>0 bug)"
        )
    if p24.get("lives"):
        failed.append("T1 leftover after days still beats size — re-open leftover")
    if p27.get("lives"):
        failed.append("share_6 leftover after days+size still ≥0.55 — re-open leftover")
    if not p9["confirm"]:
        failed.append(f"dark/erp {p9['n_dark']}/{p9['n_erp']} ≠ 470/744")
    if p10["n_ids"] != 12:
        failed.append(f"chronic names {p10['n_ids']} ≠ 12 from y2_why")
    if p3["bug"]:
        failed.append("formula disagree — inspect ops.py (do not patch tonight)")
    if abs(p5["days_y3"] - DAYS_BENCH) > 0.02 if np.isfinite(p5["days_y3"]) else True:
        failed.append(
            f"Y3 days replica {_f(p5['days_y3'])} vs night 0.711 "
            "(signed fold; published 0.711 may be oriented 1−raw)"
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
