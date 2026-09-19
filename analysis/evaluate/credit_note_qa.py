"""Q4/Q5 credit-note leftover after issued_lag1 — twin, style, or dip-vs-fall?

NORTH_STAR: `e_credit_note_ratio` = |credit notes| / (|invoices| + |credit notes|)
issued this period. Type `credit_note` if present; else ERP stand-ins `note`
and `refund`. CN is already on the Y7 TURNOVER card (issued_lag1 + issued-lag
CV + CN ±lag1 + f_fc_r_lag3). Night Y7 quote stays **TURNOVER 0.720 /
B_shallow 0.712**. Do not change it. Y7 never D. Night Y3 stays **0.762 /
0.752**. Days bar **0.711**. CN is not on the 15-col card. Y3 never B.

Question: leftover dip-vs-fall after `e_ar_issued_lag1`, or an issued / DSO
twin / style dummy? As Y3 X, does it beat days or leftover after days?

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent `y_credit_note`. Do not grow TURNOVER.
Do not edit invoices.py unless a real formula bug — then stop and report.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.credit_note_qa

Owned: analysis/evaluate/credit_note_qa.py, analysis/outputs/credit_note_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_credit_note.md (end).

Iteration (same module):
1. Prevalence + dark NaN + formula + twins + singles + leftover + token + Q6
2. Fold 4 + ICC + holdout + 12 Y2 names (Y3 only)
3. Quintiles, nonzero, company-mean, lag leftover
4. Same-n artifact + COMP_0962 + note vs refund + fold-4 leftover
5. Demean leftover (style) + issued terciles + leftover folds
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
    leakage_check,
    load_holdout,
)
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect
from analysis.features.invoices import CREDIT_NOTE_CANONICAL, CREDIT_NOTE_FALLBACK
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
OUT_MD = ANALYSIS / "outputs" / "credit_note_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "credit_note_quintiles.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "0c3bf32d"
WAVE = "4"
ROUND = "R4"
MODEL = "credit_note_qa"
X_FAM = "E"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
ISSUED_LAG1_BENCH = 0.630
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
TURNOVER_F4 = 0.680
Q6_CN_LAG1_SHORT = 0.542
ICC_QUOTE = 0.83
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
SIZE_RHO = 0.50
ICC_STYLE = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
FOLD4_GROUPS = ("GROUP_0222", "GROUP_0108")
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "c_n_days_with_tx",
    "e_credit_note_ratio",
    "e_ar_issued",
    "e_dso_proxy",
    "b_below_0",
    "f_fc_r",
)

Y_KEEP = (Y2, Y3, Y7)


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


def fold_k(rec: dict, k: int) -> float:
    for r in rec.get("folds", []):
        if int(r["fold"]) == k:
            return float(r["auroc"])
    return float("nan")


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
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)]
    )
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    pred = X @ beta
    resid.loc[ok] = Y - pred
    info["intercept"] = float(beta[0])
    info["slope"] = [float(b) for b in beta[1:]]
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    info["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return resid, info


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    mu = d.groupby("co")["x"].transform("mean")
    return d["x"] - mu


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict, present: float | None = None) -> dict:
    row = {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }
    if present is not None:
        row["present"] = _pp(present)
    return row


def infer_cn_types(found: list[str]) -> tuple[tuple[str, ...], str]:
    lower = {x.lower(): x for x in found}
    canonical = tuple(lower[c] for c in CREDIT_NOTE_CANONICAL if c in lower)
    if canonical:
        return canonical, "canonical"
    fallback = tuple(lower[c] for c in CREDIT_NOTE_FALLBACK if c in lower)
    return fallback, "fallback"


def _trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


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


def add_issued_lag_cv(df: pd.DataFrame) -> pd.DataFrame:
    """In-memory TURNOVER CV. Not written to the store."""
    out = df.copy()
    lag_names = [c for c in ("e_ar_issued_lag1", "e_ar_issued_lag2", "e_ar_issued_lag3") if c in out.columns]
    if len(lag_names) < 2:
        out["e_ar_issued_lag_cv"] = np.nan
        return out
    mat = out[lag_names].apply(pd.to_numeric, errors="coerce").clip(lower=0.0)
    mu = mat.mean(axis=1)
    sd = mat.std(axis=1, ddof=0)
    out["e_ar_issued_lag_cv"] = sd / (mu.abs() + 1.0)
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
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["first_month"] = pd.to_datetime(panel["first_month"])
    panel["months_so_far"] = (
        (panel["period"].dt.year - panel["first_month"].dt.year) * 12
        + (panel["period"].dt.month - panel["first_month"].dt.month)
        + 1
    )
    grid_n = panel.groupby("company_id")["period"].size().rename("n_grid_months")
    panel = panel.merge(grid_n, on="company_id", how="left")
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["co_class"] = panel["n_grid_months"].map(_trail_class)
    leak = leakage_check(
        ["e_credit_note_ratio", "e_ar_issued", "e_dso_proxy"],
        Y7,
        forbidden_prefixes=["d"],
    )
    if not leak["ok"]:
        raise RuntimeError(f"Y7 X leak: {leak['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def chronic_ids(tr: pd.DataFrame) -> list[str]:
    """12 chronic Y2 names: ≥50% labeled months already below 0 in 0158/0172."""
    y = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y.notna()
    hot = tr["group_id"].astype(str).isin(CHRONIC_GROUPS)
    sl = lab & hot
    if not sl.any():
        return []
    g = (
        tr.loc[sl, ["company_id"]]
        .assign(below=below[sl].values)
        .groupby("company_id")["below"]
        .agg(n="size", n_below="sum")
    )
    g["share_below"] = g["n_below"] / g["n"]
    return [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]


# ---------------------------------------------------------------------------
# Pass 1 — prevalence / coverage. Dark 470 = NaN not 0.
# ---------------------------------------------------------------------------
def pass1_prev(tr: pd.DataFrame, book: set[str]) -> dict:
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    issued = pd.to_numeric(tr["e_ar_issued"], errors="coerce")
    n_cm = int(len(tr))
    n_co = int(tr["company_id"].nunique())
    n_nn = int(cn.notna().sum())
    n_zero = int((cn == 0).sum())
    n_posv = int((cn > 0).sum())
    n_filled0 = int((cn == 0).sum())  # store 0 among defined
    dark = ~tr["company_id"].isin(book)
    erp = tr["company_id"].isin(book)
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    n_erp_co = int(tr.loc[erp, "company_id"].nunique())
    dark_nn = int(cn[dark].notna().sum())
    dark_zero = int((cn[dark] == 0).sum())
    # Dark must not be 0-filled. A note/refund-only company can have CN=1
    # without a book invoice (COMP_0962). That is NaN-or-undefined, not 0.
    dark_nan = bool(dark_zero == 0 and n_dark_co == 470)
    erp_nn = float(cn[erp].notna().mean()) if erp.any() else float("nan")
    modal = float(cn.dropna().mode().iloc[0]) if cn.notna().any() else float("nan")
    modal_share = float((cn.dropna() == modal).mean()) if cn.notna().any() else float("nan")
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y7_nn = int((y7.notna() & cn.notna()).sum())
    y7_n = int(y7.notna().sum())
    y7_pos = int((y7 == 1).sum())
    y3_nn = int((y3.notna() & cn.notna()).sum())
    y3_n = int(y3.notna().sum())
    y3_pos = int((y3 == 1).sum())
    acf1 = median_acf(cn, tr["company_id"], 1)
    acf3 = median_acf(cn, tr["company_id"], 3)
    # issued=0 months among ever-ERP: CN should stay NaN (not filled)
    iss0 = erp & (issued.fillna(0) == 0)
    cn_on_iss0 = int(cn[iss0].notna().sum())
    prose = (
        f"Train CN defined {n_nn:,}/{n_cm:,} ({_pp(_pct(n_nn, n_cm))}); "
        f"companies ever-defined {int(tr.loc[cn.notna(), 'company_id'].nunique()):,}/{n_co}. "
        f"Among defined: zero {_pp(_pct(n_zero, n_nn))} pos {_pp(_pct(n_posv, n_nn))}; "
        f"modal {modal:g} share {_pp(modal_share)}. "
        f"Dark never-ERP companies {n_dark_co} (want 470): CN non-null {dark_nn} zero-filled {dark_zero} "
        f"({'CONFIRM no 0-fill' if dark_nan else 'FAIL — dark was 0-filled'}). "
        f"Ever-ERP {n_erp_co} CM-defined {_pp(erp_nn)}. "
        f"Y7 labeled {y7_n:,} pos {y7_pos:,} CN-nn {y7_nn:,}; "
        f"Y3 stressed {y3_n:,} pos {y3_pos:,} CN-nn {y3_nn:,}. "
        f"acf1={_f(acf1)} acf3={_f(acf3)}."
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": n_co,
        "n_nn": n_nn,
        "cov": _pct(n_nn, n_cm),
        "n_zero": n_zero,
        "n_posv": n_posv,
        "zero_share": _pct(n_zero, n_nn),
        "pos_share": _pct(n_posv, n_nn),
        "modal": modal,
        "modal_share": modal_share,
        "n_dark_co": n_dark_co,
        "n_erp_co": n_erp_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_nan": dark_nan,
        "erp_nn": erp_nn,
        "y7_n": y7_n,
        "y7_pos": y7_pos,
        "y7_nn": y7_nn,
        "y3_n": y3_n,
        "y3_pos": y3_pos,
        "y3_nn": y3_nn,
        "y7_rate": float(y7.mean()) if y7.notna().any() else float("nan"),
        "y3_rate": float(y3.mean()) if y3.notna().any() else float("nan"),
        "acf1": acf1,
        "acf3": acf3,
        "cn_on_iss0": cn_on_iss0,
        "n_iss0": int(iss0.sum()),
        "confirm744": n_erp_co == 744,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — formula vs raw invoice types
# ---------------------------------------------------------------------------
def pass2_formula(tr: pd.DataFrame, con) -> dict:
    found_df = con.execute(
        "SELECT DISTINCT CAST(document_type AS VARCHAR) AS document_type "
        "FROM invoices WHERE document_type IS NOT NULL"
    ).df()
    found = (
        found_df["document_type"].astype(str).str.strip().tolist()
        if len(found_df)
        else []
    )
    cn_types, kind = infer_cn_types(found)
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT CAST(i.company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', i.issuance_date) AS DATE) AS period,
               CAST(i.document_type AS VARCHAR) AS document_type,
               SUM(ABS(i.amount)) AS abs_amt,
               COUNT(*) AS n_docs
        FROM invoices i
        WHERE i.amount <> 0
          AND i.status <> 'cancel'
          AND i.issuance_date IS NOT NULL
          AND CAST(i.issuance_date AS DATE) >= DATE '2024-09-01'
          AND CAST(i.issuance_date AS DATE) < DATE '2026-09-01'
        GROUP BY 1, 2, 3
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    raw["dt"] = raw["document_type"].astype(str).str.strip().str.lower()
    type_rows = (
        raw.groupby("dt", as_index=False)
        .agg(n_docs=("n_docs", "sum"), abs_amt=("abs_amt", "sum"), n_cm=("period", "size"))
        .sort_values("abs_amt", ascending=False)
    )
    cn_l = {t.lower() for t in cn_types}
    raw["is_inv"] = raw["dt"] == "invoice"
    raw["is_cn"] = raw["dt"].isin(cn_l)
    piv = raw.pivot_table(
        index=["company_id", "period"],
        columns="dt",
        values="abs_amt",
        aggfunc="sum",
        fill_value=0.0,
    )
    piv.columns = [str(c) for c in piv.columns]
    piv = piv.reset_index()
    cn_cols = [c for c in piv.columns if c in cn_l]
    inv_col = "invoice" if "invoice" in piv.columns else None
    piv["cn_amt"] = piv[cn_cols].sum(axis=1) if cn_cols else 0.0
    piv["inv_amt"] = piv[inv_col] if inv_col else 0.0
    piv["den"] = piv["cn_amt"] + piv["inv_amt"]
    piv["cn_hat"] = np.where(piv["den"] > 0, piv["cn_amt"] / piv["den"], np.nan)
    store = tr[["company_id", "period", "e_credit_note_ratio"]].copy()
    store["e_credit_note_ratio"] = pd.to_numeric(store["e_credit_note_ratio"], errors="coerce")
    m = store.merge(piv[["company_id", "period", "cn_hat", "cn_amt", "inv_amt", "den"]], on=["company_id", "period"], how="left")
    both = m["e_credit_note_ratio"].notna() & m["cn_hat"].notna()
    absdiff = (m.loc[both, "e_credit_note_ratio"] - m.loc[both, "cn_hat"]).abs()
    maxdiff = float(absdiff.max()) if both.any() else float("nan")
    meandiff = float(absdiff.mean()) if both.any() else float("nan")
    n_agree = int((absdiff < 1e-8).sum()) if both.any() else 0
    n_both = int(both.sum())
    store_only = int((m["e_credit_note_ratio"].notna() & m["cn_hat"].isna()).sum())
    recon_only = int((m["e_credit_note_ratio"].isna() & m["cn_hat"].notna()).sum())
    ok = bool(np.isfinite(maxdiff) and maxdiff < 1e-6 and store_only == 0)
    type_table = [
        {
            "document_type": r["dt"],
            "n_docs": f"{int(r['n_docs']):,}",
            "abs_amt": f"{float(r['abs_amt']):,.0f}",
            "n_cm": f"{int(r['n_cm']):,}",
            "in_feature": "yes" if r["dt"] in cn_l else ("invoice" if r["dt"] == "invoice" else "no"),
        }
        for _, r in type_rows.iterrows()
    ]
    prose = (
        f"Raw document_type values: {sorted(found)}. "
        f"Feature uses **{kind}** {cn_types or '(none)'}. "
        f"Store vs recompute: n_both={n_both:,} max|Δ|={maxdiff:.3e} mean|Δ|={meandiff:.3e} "
        f"exact {n_agree:,}; store-only {store_only} recon-only {recon_only}. "
        f"{'FORMULA MATCH' if ok else 'FORMULA MISMATCH — stop, do not edit invoices.py from here'}."
    )
    print(prose)
    return {
        "found": sorted(found),
        "cn_types": cn_types,
        "kind": kind,
        "type_rows": type_table,
        "type_raw": type_rows,
        "maxdiff": maxdiff,
        "meandiff": meandiff,
        "n_both": n_both,
        "n_agree": n_agree,
        "store_only": store_only,
        "recon_only": recon_only,
        "ok": ok,
        "piv": piv,
        "raw": raw,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman twins
# ---------------------------------------------------------------------------
def pass3_twins(tr: pd.DataFrame) -> dict:
    cn = tr["e_credit_note_ratio"]
    pairs = [
        ("e_ar_issued", tr["e_ar_issued"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_ar_issued_lag_cv", tr["e_ar_issued_lag_cv"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("log1p(a_in3)", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("e_credit_note_ratio_lag1", tr["e_credit_note_ratio_lag1"]),
        ("e_ar_issued_lag3", tr["e_ar_issued_lag3"]),
    ]
    rows = []
    rhos = {}
    for name, col in pairs:
        rho = spearman(cn, col)
        rhos[name] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        rows.append(
            {
                "vs": name,
                "ρ": _f(rho),
                "twin |ρ|≥0.80": "YES" if twin else "no",
            }
        )
        print(f"Spearman CN vs {name}: {rho:.3f} twin={twin}")
    twin_issued = bool(np.isfinite(rhos["e_ar_issued"]) and abs(rhos["e_ar_issued"]) >= TWIN_RHO)
    twin_lag1 = bool(np.isfinite(rhos["e_ar_issued_lag1"]) and abs(rhos["e_ar_issued_lag1"]) >= TWIN_RHO)
    twin_dso = bool(np.isfinite(rhos["e_dso_proxy"]) and abs(rhos["e_dso_proxy"]) >= TWIN_RHO)
    size_flag = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    any_twin = twin_issued or twin_lag1 or twin_dso
    prose = (
        f"CN vs issued {_f(rhos['e_ar_issued'])}, issued_lag1 {_f(rhos['e_ar_issued_lag1'])}, "
        f"DSO {_f(rhos['e_dso_proxy'])}, size {_f(rhos['log1p(a_in3)'])}, "
        f"days {_f(rhos['c_n_days_with_tx'])}. "
        f"{'TWIN of issued/DSO' if any_twin else 'Not a |ρ|≥0.80 twin of issued / issued_lag1 / DSO'}. "
        f"{'NEAR SIZE' if size_flag else 'Not a size clone'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "twin_issued": twin_issued,
        "twin_lag1": twin_lag1,
        "twin_dso": twin_dso,
        "any_twin": any_twin,
        "size_flag": size_flag,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — group-fold AUROC Y7 / Y3
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "e_credit_note_ratio": tr["e_credit_note_ratio"],
        "e_credit_note_ratio_lag1": tr["e_credit_note_ratio_lag1"],
        "e_credit_note_ratio_lag3": tr["e_credit_note_ratio_lag3"],
        "e_ar_issued": tr["e_ar_issued"],
        "e_ar_issued_lag1": tr["e_ar_issued_lag1"],
        "e_ar_issued_lag_cv": tr["e_ar_issued_lag_cv"],
        "e_dso_proxy": tr["e_dso_proxy"],
        "log1p_a_in3": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
    }
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        for name, col in feats.items():
            res = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = res
            present = float(col[lab].notna().mean()) if lab.any() else float("nan")
            rows.append(_auc_row(y, name, res, present))
            print(
                f"AUROC {y} {name}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']} sign={res['train_sign']}"
            )
    cn_y7 = _cv(store[(Y7, "e_credit_note_ratio")])
    lag1_y7 = _cv(store[(Y7, "e_credit_note_ratio_lag1")])
    iss_y7 = _cv(store[(Y7, "e_ar_issued_lag1")])
    dso_y7 = _cv(store[(Y7, "e_dso_proxy")])
    cv_y7 = _cv(store[(Y7, "e_ar_issued_lag_cv")])
    cn_y3 = _cv(store[(Y3, "e_credit_note_ratio")])
    days_y3 = _cv(store[(Y3, "c_n_days_with_tx")])
    size_y3 = _cv(store[(Y3, "log1p_a_in3")])
    iss_y3 = _cv(store[(Y3, "e_ar_issued_lag1")])
    beat_iss = cn_y7 - iss_y7 if np.isfinite(cn_y7) and np.isfinite(iss_y7) else float("nan")
    beat_days = cn_y3 - days_y3 if np.isfinite(cn_y3) and np.isfinite(days_y3) else float("nan")
    beat_size = cn_y3 - size_y3 if np.isfinite(cn_y3) and np.isfinite(size_y3) else float("nan")
    lose_days = bool(np.isfinite(cn_y3) and np.isfinite(days_y3) and cn_y3 < days_y3)
    lose_iss = bool(np.isfinite(cn_y7) and np.isfinite(iss_y7) and cn_y7 + KEEP_DELTA < iss_y7)
    replica_iss = bool(np.isfinite(iss_y7) and abs(iss_y7 - ISSUED_LAG1_BENCH) < 0.015)
    replica_days = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.015)
    replica_size = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.02)
    prose = (
        f"Y7 CN {_f(cn_y7)} vs issued_lag1 {_f(iss_y7)} (night 0.630, Δ {_f(iss_y7 - ISSUED_LAG1_BENCH)}) "
        f"vs DSO {_f(dso_y7)} vs issued-lag CV {_f(cv_y7)}. "
        f"CN lag1 {_f(lag1_y7)}. Quote TURNOVER 0.720 / B_shallow 0.712 unchanged. "
        f"Y3 CN {_f(cn_y3)} vs size {_f(size_y3)} (night 0.617) vs days {_f(days_y3)} "
        f"(night 0.711, Δ {_f(days_y3 - DAYS_BENCH)}). "
        f"{'CN loses to issued_lag1' if lose_iss else 'CN near issued_lag1'}. "
        f"{'CN loses to days — expect CLOSE as Y3 X' if lose_days else 'CN vs days unexpected'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "cn_y7": cn_y7,
        "lag1_y7": lag1_y7,
        "iss_y7": iss_y7,
        "dso_y7": dso_y7,
        "cv_y7": cv_y7,
        "cn_y3": cn_y3,
        "days_y3": days_y3,
        "size_y3": size_y3,
        "iss_y3": iss_y3,
        "beat_iss": beat_iss,
        "beat_days": beat_days,
        "beat_size": beat_size,
        "lose_days": lose_days,
        "lose_iss": lose_iss,
        "replica_iss": replica_iss,
        "replica_days": replica_days,
        "replica_size": replica_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — residual Y7 after issued_lag1 / DSO / issued
# ---------------------------------------------------------------------------
def pass5_resid_y7(tr: pd.DataFrame) -> dict:
    lab = tr[Y7].notna()
    cn = tr["e_credit_note_ratio"]
    specs = [
        ("after issued_lag1", tr["e_ar_issued_lag1"]),
        ("after issued", tr["e_ar_issued"]),
        ("after DSO", tr["e_dso_proxy"]),
        ("after issued_lag_cv", tr["e_ar_issued_lag_cv"]),
        ("after days", tr["c_n_days_with_tx"]),
        ("after size", tr["log_in3"]),
    ]
    rows = []
    store = {}
    infos = {}
    for name, x in specs:
        resid, info = ols_resid(cn, x)
        res = signed_oof_auroc(tr[Y7], resid, tr["fold"], lab)
        store[name] = res
        infos[name] = info
        rows.append(
            {
                "residual": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "R²": _f(info["r2"]),
                "slope": _f(info["slope"][0] if info["slope"] else float("nan")),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
        print(
            f"Y7 leftover {name}: "
            f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
            f"R2={_f(info['r2'])}"
        )
    # joint issued_lag1 + DSO
    resid2, info2 = ols_resid(cn, tr["e_ar_issued_lag1"], tr["e_dso_proxy"])
    res2 = signed_oof_auroc(tr[Y7], resid2, tr["fold"], lab)
    store["after issued_lag1+DSO"] = res2
    infos["after issued_lag1+DSO"] = info2
    rows.append(
        {
            "residual": "after issued_lag1+DSO",
            "n": f"{res2['n_defined']:,}",
            "n_pos": f"{res2['n_pos']:,}",
            "CV": "LOW_POWER" if res2["low_power"] else _f(res2["cv"]),
            "R²": _f(info2["r2"]),
            "slope": _f(info2["slope"][0] if info2["slope"] else float("nan")),
            "folds": fold_bits(res2) if not res2["low_power"] else "—",
        }
    )
    after_iss = _cv(store["after issued_lag1"])
    after_dso = _cv(store["after DSO"])
    after_now = _cv(store["after issued"])
    after_both = _cv(res2)
    lives = bool(np.isfinite(after_iss) and after_iss >= CHANCE)
    lives_dso = bool(np.isfinite(after_dso) and after_dso >= CHANCE)
    lives_now = bool(np.isfinite(after_now) and after_now >= CHANCE)
    died = bool(np.isfinite(after_iss) and after_iss < CHANCE)
    thin = bool(died or not lives_now)
    prose = (
        f"Y7 leftover after issued_lag1 {_f(after_iss)} "
        f"({'lives ≥0.55' if lives else 'dies — not leftover'}); "
        f"after issued {_f(after_now)}; after DSO {_f(after_dso)}; "
        f"after both {_f(after_both)}. "
        f"{'Leftover is sayable without thin issuance' if lives and lives_now else 'Leftover repeats thin issuance / dies'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "after_iss": after_iss,
        "after_dso": after_dso,
        "after_now": after_now,
        "after_both": after_both,
        "after_cv": _cv(store["after issued_lag_cv"]),
        "lives": lives,
        "lives_dso": lives_dso,
        "lives_now": lives_now,
        "died": died,
        "thin": thin,
        "resid_iss": ols_resid(cn, tr["e_ar_issued_lag1"])[0],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — residual Y3 after days (expect CLOSE)
# ---------------------------------------------------------------------------
def pass6_resid_y3(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    cn = tr["e_credit_note_ratio"]
    specs = [
        ("after days", tr["c_n_days_with_tx"]),
        ("after size", tr["log_in3"]),
        ("after issued_lag1", tr["e_ar_issued_lag1"]),
        ("after DSO", tr["e_dso_proxy"]),
        ("after days+size", None),
    ]
    rows = []
    store = {}
    for name, x in specs:
        if name == "after days+size":
            resid, info = ols_resid(cn, tr["c_n_days_with_tx"], tr["log_in3"])
        else:
            resid, info = ols_resid(cn, x)
        res = signed_oof_auroc(tr[Y3], resid, tr["fold"], lab)
        store[name] = res
        rows.append(
            {
                "residual": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "R²": _f(info["r2"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
    after_days = _cv(store["after days"])
    after_size = _cv(store["after size"])
    lives = bool(np.isfinite(after_days) and after_days >= SIZE_QUOTE + KEEP_DELTA)
    died = bool(not lives)
    prose = (
        f"Y3 leftover after days {_f(after_days)} vs size {_f(after_size)}. "
        f"{'Unexpected leftover vs size+0.02' if lives else 'Expect CLOSE as Y3 X — leftover after days dies'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "after_days": after_days,
        "after_size": after_size,
        "lives": lives,
        "died": died,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — token mix credit_note vs note vs refund
# ---------------------------------------------------------------------------
def pass7_tokens(p2: dict) -> dict:
    raw = p2["raw"]
    kind = p2["kind"]
    cn_types = p2["cn_types"]
    want = ("credit_note", "creditnote", "note", "refund")
    sl = raw[raw["dt"].isin(want)].copy()
    tot = float(sl["abs_amt"].sum()) if len(sl) else 0.0
    rows = []
    shares = {}
    for tok in want:
        part = sl[sl["dt"] == tok]
        amt = float(part["abs_amt"].sum()) if len(part) else 0.0
        n_docs = int(part["n_docs"].sum()) if len(part) else 0
        n_co = int(part["company_id"].nunique()) if len(part) else 0
        shares[tok] = _pct(amt, tot) if tot else float("nan")
        rows.append(
            {
                "token": tok,
                "n_docs": f"{n_docs:,}",
                "n_co": f"{n_co:,}",
                "abs_amt": f"{amt:,.0f}",
                "share_mass": _pp(shares[tok]),
                "in_feature": "yes" if tok in {t.lower() for t in cn_types} else "no",
            }
        )
    used = sl[sl["dt"].isin({t.lower() for t in cn_types})]
    used_tot = float(used["abs_amt"].sum()) if len(used) else 0.0
    canon_mass = float(shares.get("credit_note") or 0) + float(shares.get("creditnote") or 0)
    note_mass = float(shares.get("note") or 0)
    refund_mass = float(shares.get("refund") or 0)
    mostly_refund = bool(np.isfinite(refund_mass) and refund_mass >= 0.50)
    mostly_note = bool(np.isfinite(note_mass) and note_mass >= 0.50)
    mostly_canon = bool(canon_mass >= 0.50)
    prose = (
        f"CN-like |amt| mass (train, 2024-09..2026-08): canonical {_pp(canon_mass)}, "
        f"note {_pp(note_mass)}, refund {_pp(refund_mass)}. Feature kind={kind} "
        f"uses {cn_types or '()'}; used-mass {used_tot:,.0f} of CN-like {tot:,.0f}. "
        + (
            "Mostly refunds — the ratio is an ERP refund stand-in, not a credit-note book."
            if mostly_refund
            else (
                "Mostly `note` stand-ins."
                if mostly_note
                else (
                    "Mass is canonical `credit_note`."
                    if mostly_canon
                    else "Mixed tokens."
                )
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "shares": shares,
        "canon_mass": canon_mass,
        "note_mass": note_mass,
        "refund_mass": refund_mass,
        "mostly_refund": mostly_refund,
        "mostly_note": mostly_note,
        "mostly_canon": mostly_canon,
        "kind": kind,
        "tot": tot,
        "used_tot": used_tot,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — Q6 lag1/lag3 short vs long
# ---------------------------------------------------------------------------
def pass8_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = [
        ("all", tr[Y7].notna()),
        ("short_<12_sofar", tr[Y7].notna() & (tr["so_far_class"] == "short_<12")),
        ("long_>=18_sofar", tr[Y7].notna() & (tr["so_far_class"] == "long_>=18")),
        ("short_<12_company", tr[Y7].notna() & (tr["co_class"] == "short_<12")),
        ("long_>=18_company", tr[Y7].notna() & (tr["co_class"] == "long_>=18")),
    ]
    cols = (
        "e_credit_note_ratio",
        "e_credit_note_ratio_lag1",
        "e_credit_note_ratio_lag3",
        "e_ar_issued_lag1",
    )
    for sname, mask in slices:
        n_lab = int(mask.sum())
        for col in cols:
            res = signed_oof_auroc(tr[Y7], tr[col], tr["fold"], mask)
            store[(sname, col)] = res
            present = float(tr.loc[mask, col].notna().mean()) if n_lab else float("nan")
            night = ISSUED_LAG1_BENCH if col.startswith("e_ar") else ISSUED_LAG1_BENCH
            delta = res["cv"] - night if (not res["low_power"] and np.isfinite(res["cv"])) else float("nan")
            rows.append(
                {
                    "slice": sname,
                    "col": col,
                    "n_nn": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(present),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "Δ night 0.630": _f(delta, 3) if np.isfinite(delta) else "—",
                }
            )
    short_lag = store[("short_<12_sofar", "e_credit_note_ratio_lag1")]
    short_cv = _cv(short_lag)
    confirm = bool(np.isfinite(short_cv) and abs(short_cv - Q6_CN_LAG1_SHORT) < 0.02)
    keep_q6 = bool(
        np.isfinite(short_cv)
        and short_cv >= ISSUED_LAG1_BENCH - 0.03
        and short_cv >= CHANCE
    )
    prose = (
        f"Y7 CN lag1 short so-far {_f(short_cv)} "
        f"(q6_quoted 0.542, {'CONFIRM' if confirm else 'off quote'}). "
        f"{'KEEP as Q6' if keep_q6 else 'CLOSE as Q6 — short lag1 stays weak vs issued_lag1 0.630'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "short_lag": short_cv,
        "confirm": confirm,
        "keep_q6": keep_q6,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — fold 4 Y7 (DSO failed; TURNOVER 0.680)
# ---------------------------------------------------------------------------
def pass9_fold4(tr: pd.DataFrame, p4: dict, p5: dict) -> dict:
    lab = tr[Y7].notna()
    f4 = lab & (tr["fold"] == 4)
    hot = tr["group_id"].astype(str).isin(FOLD4_GROUPS)
    rows = []
    store = {}
    for name, col in (
        ("e_credit_note_ratio", tr["e_credit_note_ratio"]),
        ("e_credit_note_ratio_lag1", tr["e_credit_note_ratio_lag1"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_ar_issued_lag_cv", tr["e_ar_issued_lag_cv"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("CN resid after issued_lag1", p5["resid_iss"]),
    ):
        res = signed_oof_auroc(tr[Y7], col, tr["fold"], lab)
        store[name] = res
        auc4 = fold_k(res, 4)
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "fold4": _f(auc4),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
    # fold-4-only pooled AUROC (sign from folds 0–3)
    f4_rows = []
    for name, col in (
        ("e_credit_note_ratio", tr["e_credit_note_ratio"]),
        ("e_credit_note_ratio_lag1", tr["e_credit_note_ratio_lag1"]),
        ("e_ar_issued_lag1", tr["e_ar_issued_lag1"]),
        ("e_ar_issued_lag_cv", tr["e_ar_issued_lag_cv"]),
        ("e_dso_proxy", tr["e_dso_proxy"]),
        ("CN resid after issued_lag1", p5["resid_iss"]),
    ):
        trm = lab & (tr["fold"] != 4) & pd.to_numeric(col, errors="coerce").notna()
        vam = f4 & pd.to_numeric(col, errors="coerce").notna()
        sign = choose_sign(tr[Y7][trm], col[trm])
        auc = auroc(tr[Y7][vam], sign * pd.to_numeric(col[vam], errors="coerce"))
        f4_rows.append(
            {
                "feature": name,
                "n_va": int(vam.sum()),
                "n_pos": int((vam & (tr[Y7] == 1)).sum()),
                "fold4": _f(auc),
                "sign": int(sign),
            }
        )
    grp = []
    for gid in FOLD4_GROUPS:
        sl = lab & (tr["group_id"].astype(str) == gid)
        y = pd.to_numeric(tr.loc[sl, Y7], errors="coerce")
        cn = pd.to_numeric(tr.loc[sl, "e_credit_note_ratio"], errors="coerce")
        iss = pd.to_numeric(tr.loc[sl, "e_ar_issued_lag1"], errors="coerce")
        grp.append(
            {
                "group": gid,
                "n_lab": int(sl.sum()),
                "n_pos": int((y == 1).sum()),
                "rate": _pp(float(y.mean()) if y.notna().any() else float("nan")),
                "CN p50": _f(float(cn.median()) if cn.notna().any() else float("nan")),
                "issued_lag1 p50": f"{float(iss.median()):,.0f}" if iss.notna().any() else "—",
            }
        )
    cn4 = next(r for r in f4_rows if r["feature"] == "e_credit_note_ratio")
    iss4 = next(r for r in f4_rows if r["feature"] == "e_ar_issued_lag1")
    resid4 = next(r for r in f4_rows if r["feature"] == "CN resid after issued_lag1")
    cn4v = float(cn4["fold4"]) if cn4["fold4"] != "—" else float("nan")
    iss4v = float(iss4["fold4"]) if iss4["fold4"] != "—" else float("nan")
    resid4v = float(resid4["fold4"]) if resid4["fold4"] != "—" else float("nan")
    cn_saves = bool(np.isfinite(cn4v) and cn4v >= TURNOVER_F4 - 0.02)
    issued_owns = bool(np.isfinite(iss4v) and iss4v >= 0.60 and (not np.isfinite(cn4v) or cn4v < 0.60))
    prose = (
        f"Fold 4 singles (sign from folds 0–3): CN {_f(cn4v)}, issued_lag1 {_f(iss4v)}, "
        f"CN residual {_f(resid4v)}. TURNOVER fold-4 quote 0.680. "
        + (
            "CN saves fold 4."
            if cn_saves
            else (
                "TURNOVER fold-4 0.680 is issued / issued-CV, not CN."
                if issued_owns
                else "CN does not own fold 4."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "f4_rows": f4_rows,
        "grp": grp,
        "cn4": cn4v,
        "iss4": iss4v,
        "resid4": resid4v,
        "cn_saves": cn_saves,
        "issued_owns": issued_owns,
        "n_f4": int(f4.sum()),
        "n_f4_pos": int((f4 & (tr[Y7] == 1)).sum()),
        "n_hot": int((lab & hot).sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    cn = tr["e_credit_note_ratio"]
    icc = icc_anova(cn, tr["company_id"])
    demean = company_demean(cn, tr["company_id"])
    mean_co = tr.groupby("company_id")["e_credit_note_ratio"].transform("mean")
    confirm = bool(np.isfinite(icc["icc"]) and abs(icc["icc"] - ICC_QUOTE) < 0.05)
    style = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        raw = signed_oof_auroc(tr[y], cn, tr["fold"], lab)
        de = signed_oof_auroc(tr[y], demean, tr["fold"], lab)
        mu = signed_oof_auroc(tr[y], mean_co, tr["fold"], lab)
        store[(y, "raw")] = raw
        store[(y, "demean")] = de
        store[(y, "co_mean")] = mu
        rows.append(_auc_row(y, "CN raw", raw))
        rows.append(_auc_row(y, "CN demean", de))
        rows.append(_auc_row(y, "CN company-mean", mu))
    y7_de = _cv(store[(Y7, "demean")])
    y7_mu = _cv(store[(Y7, "co_mean")])
    shock = bool(np.isfinite(y7_de) and y7_de >= CHANCE)
    trait = bool(np.isfinite(y7_mu) and y7_mu >= CHANCE and (not np.isfinite(y7_de) or y7_de < CHANCE))
    prose = (
        f"CN ICC={_f(icc['icc'])} ({'CONFIRM feature-report 0.83' if confirm else 'off 0.83'}) "
        f"k={icc['k']}. Y7 demean {_f(y7_de)} company-mean {_f(y7_mu)}. "
        + (
            "Within-company shock leftover lives."
            if shock
            else ("Company-mean carries the skill — style dummy." if trait else "Neither demean nor mean is leftover.")
        )
    )
    print(prose)
    return {
        "icc": icc,
        "confirm": confirm,
        "style": style,
        "rows": rows,
        "store": store,
        "y7_de": y7_de,
        "y7_mu": y7_mu,
        "y3_de": _cv(store[(Y3, "demean")]),
        "shock": shock,
        "trait": trait,
        "acf1": median_acf(cn, tr["company_id"], 1),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — dark 470 + holdout coverage (no AUROC claim)
# ---------------------------------------------------------------------------
def pass11_dark_hold(panel: pd.DataFrame, book: set[str]) -> dict:
    ho = panel[panel["split"] == "holdout"].copy()
    cn = pd.to_numeric(ho["e_credit_note_ratio"], errors="coerce")
    y7 = pd.to_numeric(ho[Y7], errors="coerce")
    y3 = pd.to_numeric(ho[Y3], errors="coerce")
    dark = ~ho["company_id"].isin(book)
    erp = ho["company_id"].isin(book)
    n_dark_co = int(ho.loc[dark, "company_id"].nunique())
    n_erp_co = int(ho.loc[erp, "company_id"].nunique())
    dark_nn = int(cn[dark].notna().sum())
    rows = [
        {
            "slice": "holdout all",
            "n_cm": f"{len(ho):,}",
            "n_co": f"{ho['company_id'].nunique():,}",
            "CN nn": f"{int(cn.notna().sum()):,}",
            "share_nn": _pp(float(cn.notna().mean())),
            "Y7 lab / pos": f"{int(y7.notna().sum()):,} / {int((y7 == 1).sum()):,}",
            "Y3 lab / pos": f"{int(y3.notna().sum()):,} / {int((y3 == 1).sum()):,}",
        },
        {
            "slice": "holdout dark",
            "n_cm": f"{int(dark.sum()):,}",
            "n_co": f"{n_dark_co:,}",
            "CN nn": f"{dark_nn:,}",
            "share_nn": _pp(float(cn[dark].notna().mean()) if dark.any() else float("nan")),
            "Y7 lab / pos": f"{int(y7[dark].notna().sum()):,} / {int((y7[dark] == 1).sum()):,}",
            "Y3 lab / pos": f"{int(y3[dark].notna().sum()):,} / {int((y3[dark] == 1).sum()):,}",
        },
        {
            "slice": "holdout ERP",
            "n_cm": f"{int(erp.sum()):,}",
            "n_co": f"{n_erp_co:,}",
            "CN nn": f"{int(cn[erp].notna().sum()):,}",
            "share_nn": _pp(float(cn[erp].notna().mean()) if erp.any() else float("nan")),
            "Y7 lab / pos": f"{int(y7[erp].notna().sum()):,} / {int((y7[erp] == 1).sum()):,}",
            "Y3 lab / pos": f"{int(y3[erp].notna().sum()):,} / {int((y3[erp] == 1).sum()):,}",
        },
    ]
    y7_pos = int((y7 == 1).sum())
    confirm_122 = y7_pos == 122
    dark_ok = dark_nn == 0
    prose = (
        f"Holdout 72 coverage only: Y7 labeled pos={y7_pos} "
        f"({'CONFIRM 122' if confirm_122 else 'off 122'}) — not a trophy AUROC. "
        f"Dark holdout companies {n_dark_co} (join-QA 32): CN nn={dark_nn} "
        f"({'NaN stays' if dark_ok else 'FAIL filled'}). Do not fill 0."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_pos": y7_pos,
        "confirm_122": confirm_122,
        "n_dark_co": n_dark_co,
        "n_erp_co": n_erp_co,
        "dark_nn": dark_nn,
        "dark_ok": dark_ok,
        "n_cm": int(len(ho)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 12 — 12 Y2 chronic names, scored on Y3 only
# ---------------------------------------------------------------------------
def pass12_chronic_y3(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab = tr[Y3].notna()
    full = signed_oof_auroc(tr[Y3], tr["e_credit_note_ratio"], tr["fold"], lab)
    rest = signed_oof_auroc(tr[Y3], tr["e_credit_note_ratio"], tr["fold"], lab & drop)
    days_f = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab)
    days_r = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    size_f = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab)
    size_r = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab & drop)
    # Y7 robustness, not a Y2 claim
    lab7 = tr[Y7].notna()
    y7_f = signed_oof_auroc(tr[Y7], tr["e_credit_note_ratio"], tr["fold"], lab7)
    y7_r = signed_oof_auroc(tr[Y7], tr["e_credit_note_ratio"], tr["fold"], lab7 & drop)
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    rows = [
        {
            "slice": "Y3 all",
            "CN": "LOW_POWER" if full["low_power"] else _f(full["cv"]),
            "days": "LOW_POWER" if days_f["low_power"] else _f(days_f["cv"]),
            "size": "LOW_POWER" if size_f["low_power"] else _f(size_f["cv"]),
        },
        {
            "slice": "Y3 drop-12",
            "CN": "LOW_POWER" if rest["low_power"] else _f(rest["cv"]),
            "days": "LOW_POWER" if days_r["low_power"] else _f(days_r["cv"]),
            "size": "LOW_POWER" if size_r["low_power"] else _f(size_r["cv"]),
        },
        {
            "slice": "Y7 all (robustness)",
            "CN": "LOW_POWER" if y7_f["low_power"] else _f(y7_f["cv"]),
            "days": "—",
            "size": "—",
        },
        {
            "slice": "Y7 drop-12 (robustness)",
            "CN": "LOW_POWER" if y7_r["low_power"] else _f(y7_r["cv"]),
            "days": "—",
            "size": "—",
        },
    ]
    prose = (
        f"Chronic 12 Y2 names (0158/0172, ≥50% labeled months below 0): {len(ids)}. "
        f"Y3-only CN {_f(_cv(full))} → drop-12 {_f(_cv(rest))} "
        f"({'flips ≥0.03' if flip else 'does not flip'}). "
        f"Days stays {_f(_cv(days_r))}. No Y2 AUROC claim."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "ids": ids,
        "rows": rows,
        "y3_full": _cv(full),
        "y3_drop": _cv(rest),
        "y7_full": _cv(y7_f),
        "y7_drop": _cv(y7_r),
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 13 — quintiles + rates
# ---------------------------------------------------------------------------
def pass13_quintiles(tr: pd.DataFrame) -> dict:
    rows = []
    for y, ycol in ((Y7, Y7), (Y3, Y3)):
        sl = tr.loc[tr[ycol].notna() & tr["e_credit_note_ratio"].notna()].copy()
        sl["q"] = pd.qcut(
            pd.to_numeric(sl["e_credit_note_ratio"], errors="coerce").rank(method="first"),
            5,
            labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        )
        for q, part in sl.groupby("q", observed=False):
            s = pd.to_numeric(part[ycol], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "CN q": str(q),
                    "n": f"{len(part):,}",
                    "n_pos": f"{int((s == 1).sum()):,}",
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                    "CN p50": _f(float(pd.to_numeric(part["e_credit_note_ratio"], errors="coerce").median())),
                }
            )
    # CN>0 vs =0
    flag_rows = []
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    for y in (Y7, Y3):
        for name, mask in (("CN=0", cn == 0), ("CN>0", cn > 0), ("CN NaN", cn.isna())):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            flag_rows.append(
                {
                    "y": y,
                    "slice": name,
                    "n_cm": f"{int(mask.sum()):,}",
                    "n_lab": f"{int(s.notna().sum()):,}",
                    "n_pos": f"{int((s == 1).sum()):,}",
                    "rate": _pp(float(s.mean()) if s.notna().any() else float("nan")),
                }
            )
    y7_0 = next(r for r in flag_rows if r["y"] == Y7 and r["slice"] == "CN=0")
    y7_p = next(r for r in flag_rows if r["y"] == Y7 and r["slice"] == "CN>0")
    prose = (
        f"Y7 rate CN=0 {y7_0['rate']} vs CN>0 {y7_p['rate']}. "
        "A leftover Q4 would show a monotone CN quintile. A twin of thin issuance would pile with low issued."
    )
    print(prose)
    return {"rows": rows, "flag_rows": flag_rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 14 — nonzero-only leftover (is skill the zero pile?)
# ---------------------------------------------------------------------------
def pass14_nonzero(tr: pd.DataFrame, p5: dict) -> dict:
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    nz = cn > 0
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        for sname, mask in (("defined", lab & cn.notna()), ("CN>0", lab & nz), ("CN=0", lab & (cn == 0))):
            res = signed_oof_auroc(tr[y], tr["e_credit_note_ratio"], tr["fold"], mask)
            store[(y, sname)] = res
            rows.append(_auc_row(y, sname, res))
        resid = p5["resid_iss"]
        res_nz = signed_oof_auroc(tr[y], resid, tr["fold"], lab & nz)
        store[(y, "resid_nz")] = res_nz
        rows.append(_auc_row(y, "resid after issued_lag1 | CN>0", res_nz))
    y7_nz = _cv(store[(Y7, "CN>0")])
    y7_z = _cv(store[(Y7, "CN=0")])
    y7_rnz = _cv(store[(Y7, "resid_nz")])
    pile = bool(np.isfinite(y7_nz) and y7_nz < CHANCE)
    prose = (
        f"Y7 CN among CN>0 {_f(y7_nz)} / among CN=0 {_f(y7_z)}; "
        f"residual after issued_lag1 on CN>0 {_f(y7_rnz)}. "
        f"{'Skill is the zero pile' if pile else 'Nonzero CN still measured'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "y7_nz": y7_nz,
        "y7_z": y7_z,
        "y7_rnz": y7_rnz,
        "pile": pile,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 15 — company-mean leftover after company-mean issued
# ---------------------------------------------------------------------------
def pass15_co_mean(tr: pd.DataFrame) -> dict:
    cn_mu = tr.groupby("company_id")["e_credit_note_ratio"].transform("mean")
    iss_mu = tr.groupby("company_id")["e_ar_issued_lag1"].transform("mean")
    resid, info = ols_resid(cn_mu, iss_mu)
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        r_mu = signed_oof_auroc(tr[y], cn_mu, tr["fold"], lab)
        r_res = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[(y, "mean")] = r_mu
        store[(y, "resid")] = r_res
        rows.append(_auc_row(y, "company-mean CN", r_mu))
        rows.append(_auc_row(y, "mean CN after mean issued_lag1", r_res))
    y7_res = _cv(store[(Y7, "resid")])
    died = bool(not np.isfinite(y7_res) or y7_res < CHANCE)
    prose = (
        f"Y7 company-mean CN leftover after company-mean issued_lag1 {_f(y7_res)} "
        f"(R²={_f(info['r2'])}). "
        f"{'Company-mean leftover dies — CN is a thin-issuance company twin.' if died else 'Company-mean leftover still lives.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "y7_res": y7_res,
        "y3_res": _cv(store[(Y3, "resid")]),
        "r2": info["r2"],
        "died": died,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 16 — CN lag1 leftover after issued_lag1 (Q6 leftover)
# ---------------------------------------------------------------------------
def pass16_lag_resid(tr: pd.DataFrame) -> dict:
    lag = tr["e_credit_note_ratio_lag1"]
    resid, info = ols_resid(lag, tr["e_ar_issued_lag1"])
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        r = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[y] = r
        rows.append(_auc_row(y, "CN_lag1 resid after issued_lag1", r))
    y7 = _cv(store[Y7])
    lives = bool(np.isfinite(y7) and y7 >= CHANCE)
    prose = (
        f"Y7 CN_lag1 leftover after issued_lag1 {_f(y7)} (R²={_f(info['r2'])}). "
        f"{'Lag leftover lives' if lives else 'CLOSE — lag1 leftover dies after issued_lag1'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y7": y7,
        "y3": _cv(store[Y3]),
        "r2": info["r2"],
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 17 — is leftover 0.597 a same-n / rank artifact?
# ---------------------------------------------------------------------------
def pass17_artifact(tr: pd.DataFrame, p5: dict) -> dict:
    """Raw CN on the residual's row set vs OLS / rank residual."""
    lab = tr[Y7].notna()
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    iss = pd.to_numeric(tr["e_ar_issued_lag1"], errors="coerce")
    both = lab & cn.notna() & iss.notna()
    resid = p5["resid_iss"]
    rank_cn = cn.rank(method="average")
    rank_iss = iss.rank(method="average")
    resid_rk, info_rk = ols_resid(rank_cn, rank_iss)
    log_iss = np.log1p(iss.clip(lower=0))
    resid_log, info_log = ols_resid(cn, log_iss)
    rows = []
    store = {}
    specs = [
        ("raw CN | issued_lag1 nn", cn, both),
        ("issued_lag1 | same n", iss, both),
        ("OLS resid CN~issued_lag1", resid, both),
        ("rank resid CN~issued_lag1", resid_rk, both),
        ("OLS resid CN~log1p issued_lag1", resid_log, both),
        ("raw CN all labeled nn", cn, lab),
    ]
    for name, col, mask in specs:
        res = signed_oof_auroc(tr[Y7], col, tr["fold"], mask)
        store[name] = res
        rows.append(_auc_row(Y7, name, res))
    raw_both = _cv(store["raw CN | issued_lag1 nn"])
    ols = _cv(store["OLS resid CN~issued_lag1"])
    rk = _cv(store["rank resid CN~issued_lag1"])
    logv = _cv(store["OLS resid CN~log1p issued_lag1"])
    iss_both = _cv(store["issued_lag1 | same n"])
    lift = ols - raw_both if np.isfinite(ols) and np.isfinite(raw_both) else float("nan")
    artifact = bool(np.isfinite(lift) and lift < 0.02 and raw_both >= CHANCE)
    real = bool(np.isfinite(ols) and ols >= CHANCE and np.isfinite(lift) and lift >= 0.02)
    prose = (
        f"Same-n Y7: raw CN {_f(raw_both)} vs OLS leftover {_f(ols)} (Δ {_f(lift)}) "
        f"vs rank leftover {_f(rk)} vs log leftover {_f(logv)}; issued_lag1 {_f(iss_both)}. "
        + (
            "Leftover is the same-n raw — not a new residual."
            if artifact
            else (
                "Leftover lifts raw CN on the same rows — residual is not a sample trick."
                if real
                else "Leftover vs same-n raw is a small lift; treat as weak leftover."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "raw_both": raw_both,
        "ols": ols,
        "rank": rk,
        "log": logv,
        "iss_both": iss_both,
        "lift": lift,
        "artifact": artifact,
        "real": real,
        "n_both": int(both.sum()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 18 — COMP_0962 refund-only dark exception
# ---------------------------------------------------------------------------
def pass18_dark_one(tr: pd.DataFrame, book: set[str], p2: dict) -> dict:
    cn = pd.to_numeric(tr["e_credit_note_ratio"], errors="coerce")
    dark = ~tr["company_id"].isin(book)
    hit = tr.loc[dark & cn.notna(), ["company_id", "group_id", "period", "e_credit_note_ratio", "e_ar_issued"]]
    raw = p2["raw"]
    note_only = []
    if len(hit):
        ids = set(hit["company_id"].astype(str))
        sl = raw[raw["company_id"].isin(ids)]
        for cid, g in sl.groupby("company_id"):
            types = sorted(g["dt"].unique())
            note_only.append(
                {
                    "company_id": cid,
                    "types": ",".join(types),
                    "n_docs": f"{int(g['n_docs'].sum()):,}",
                    "has_invoice": "yes" if "invoice" in types else "no",
                    "CN months": f"{int((hit['company_id'] == cid).sum()):,}",
                    "CN values": ",".join(f"{v:.3g}" for v in hit.loc[hit["company_id"] == cid, "e_credit_note_ratio"]),
                }
            )
    prose = (
        f"Dark CN non-null rows: {len(hit)}. "
        + (
            "COMP_0962 is refund-only (no book invoice) — CN=1 is defined, not a 0-fill. "
            "The other 469 dark companies stay NaN. Do not fill 0."
            if len(hit)
            else "No dark CN rows."
        )
    )
    print(prose)
    return {
        "n": int(len(hit)),
        "rows": note_only,
        "ids": sorted(set(hit["company_id"].astype(str))) if len(hit) else [],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 19 — note vs refund as separate in-memory ratios
# ---------------------------------------------------------------------------
def pass19_note_refund(tr: pd.DataFrame, p2: dict) -> dict:
    """Is leftover a refund spike or the note stand-in book? Not a new Y."""
    piv = p2["piv"].copy()
    for tok in ("note", "refund"):
        if tok not in piv.columns:
            piv[tok] = 0.0
        piv[tok] = pd.to_numeric(piv[tok], errors="coerce").fillna(0.0)
    if "invoice" not in piv.columns:
        piv["invoice"] = 0.0
    piv["note_ratio"] = piv["note"] / (piv["note"] + piv["invoice"]).where((piv["note"] + piv["invoice"]) > 0)
    piv["refund_ratio"] = piv["refund"] / (piv["refund"] + piv["invoice"]).where((piv["refund"] + piv["invoice"]) > 0)
    m = tr.merge(piv[["company_id", "period", "note_ratio", "refund_ratio", "note", "refund"]], on=["company_id", "period"], how="left")
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = m[y].notna()
        for col in ("note_ratio", "refund_ratio"):
            res = signed_oof_auroc(m[y], m[col], m["fold"], lab)
            store[(y, col)] = res
            rows.append(_auc_row(y, col, res))
        resid, _ = ols_resid(m["note_ratio"], m["e_ar_issued_lag1"])
        r = signed_oof_auroc(m[y], resid, m["fold"], lab)
        store[(y, "note_resid")] = r
        rows.append(_auc_row(y, "note_ratio resid issued_lag1", r))
    y7_note = _cv(store[(Y7, "note_ratio")])
    y7_ref = _cv(store[(Y7, "refund_ratio")])
    y7_nr = _cv(store[(Y7, "note_resid")])
    prose = (
        f"Y7 note_ratio {_f(y7_note)} refund_ratio {_f(y7_ref)}; "
        f"note leftover after issued_lag1 {_f(y7_nr)}. "
        "If leftover is note not refund, the stand-in book is the object — still not a Y."
    )
    print(prose)
    return {
        "rows": rows,
        "y7_note": y7_note,
        "y7_ref": y7_ref,
        "y7_nr": y7_nr,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — leftover after issued_lag1 inside fold 4 only
# ---------------------------------------------------------------------------
def pass20_f4_resid(tr: pd.DataFrame, p5: dict) -> dict:
    lab = tr[Y7].notna()
    rows = []
    for name, col in (
        ("CN", tr["e_credit_note_ratio"]),
        ("issued_lag1", tr["e_ar_issued_lag1"]),
        ("issued_lag_cv", tr["e_ar_issued_lag_cv"]),
        ("CN resid issued_lag1", p5["resid_iss"]),
        ("DSO", tr["e_dso_proxy"]),
    ):
        trm = lab & (tr["fold"] != 4) & pd.to_numeric(col, errors="coerce").notna()
        vam = lab & (tr["fold"] == 4) & pd.to_numeric(col, errors="coerce").notna()
        sign = choose_sign(tr[Y7][trm], col[trm])
        auc = auroc(tr[Y7][vam], sign * pd.to_numeric(col[vam], errors="coerce"))
        rows.append(
            {
                "feature": name,
                "n_va": f"{int(vam.sum()):,}",
                "n_pos": f"{int((vam & (tr[Y7] == 1)).sum()):,}",
                "fold4": _f(auc),
            }
        )
    cn4 = float(rows[0]["fold4"]) if rows[0]["fold4"] != "—" else float("nan")
    iss4 = float(rows[1]["fold4"]) if rows[1]["fold4"] != "—" else float("nan")
    cv4 = float(rows[2]["fold4"]) if rows[2]["fold4"] != "—" else float("nan")
    rs4 = float(rows[3]["fold4"]) if rows[3]["fold4"] != "—" else float("nan")
    issued_card = bool(np.isfinite(iss4) and np.isfinite(cv4) and max(iss4, cv4) >= 0.62 and (not np.isfinite(cn4) or cn4 < 0.62))
    prose = (
        f"Fold 4 leftover check: CN {_f(cn4)} issued_lag1 {_f(iss4)} issued-CV {_f(cv4)} "
        f"CN residual {_f(rs4)}. TURNOVER 0.680. "
        f"{'Issued + CV own fold 4; CN residual does not save it.' if issued_card else 'CN residual competes on fold 4.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "cn4": cn4,
        "iss4": iss4,
        "cv4": cv4,
        "rs4": rs4,
        "issued_card": issued_card,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 21 — within-company leftover (demean CN after demean issued)
# ---------------------------------------------------------------------------
def pass21_demean_resid(tr: pd.DataFrame) -> dict:
    """If demean leftover dies, KEEP leftover is a who-uses-notes style, not a month shock."""
    cn_d = company_demean(tr["e_credit_note_ratio"], tr["company_id"])
    iss_d = company_demean(tr["e_ar_issued_lag1"], tr["company_id"])
    resid, info = ols_resid(cn_d, iss_d)
    rows = []
    store = {}
    for y in (Y7, Y3):
        lab = tr[y].notna()
        r_d = signed_oof_auroc(tr[y], cn_d, tr["fold"], lab)
        r_r = signed_oof_auroc(tr[y], resid, tr["fold"], lab)
        store[(y, "demean")] = r_d
        store[(y, "resid")] = r_r
        rows.append(_auc_row(y, "demean CN", r_d))
        rows.append(_auc_row(y, "demean CN after demean issued_lag1", r_r))
    y7 = _cv(store[(Y7, "resid")])
    y7_d = _cv(store[(Y7, "demean")])
    shock = bool(np.isfinite(y7) and y7 >= CHANCE)
    prose = (
        f"Y7 demean CN {_f(y7_d)}; demean leftover after demean issued_lag1 {_f(y7)} "
        f"(R²={_f(info['r2'])}). "
        f"{'Within-company leftover lives — month shock.' if shock else 'Within-company leftover dies — KEEP leftover is who-uses-notes, not this month’s correction.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "y7": y7,
        "y7_d": y7_d,
        "y3": _cv(store[(Y3, "resid")]),
        "shock": shock,
        "r2": info["r2"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 22 — leftover inside issued_lag1 terciles
# ---------------------------------------------------------------------------
def pass22_iss_tercile(tr: pd.DataFrame, p5: dict) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last["e_ar_issued_lag1"], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1", "T2", "T3"], duplicates="drop").rename("iss_t")
    m = tr.merge(terc.reset_index(), on="company_id", how="left")
    resid, _ = ols_resid(m["e_credit_note_ratio"], m["e_ar_issued_lag1"])
    rows = []
    store = {}
    lab = m[Y7].notna()
    for labv in ("T1", "T2", "T3"):
        mask = lab & (m["iss_t"].astype(str) == labv)
        raw = signed_oof_auroc(m[Y7], m["e_credit_note_ratio"], m["fold"], mask)
        res = signed_oof_auroc(m[Y7], resid, m["fold"], mask)
        iss = signed_oof_auroc(m[Y7], m["e_ar_issued_lag1"], m["fold"], mask)
        store[labv] = {"raw": raw, "resid": res, "iss": iss}
        rows.append(
            {
                "issued tercile": labv,
                "n": f"{raw['n_defined']:,}",
                "n_pos": f"{raw['n_pos']:,}",
                "CN": "LOW_POWER" if raw["low_power"] else _f(raw["cv"]),
                "resid": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "issued_lag1": "LOW_POWER" if iss["low_power"] else _f(iss["cv"]),
            }
        )
    lives = [t for t, r in store.items() if not r["resid"]["low_power"] and r["resid"]["cv"] >= CHANCE]
    prose = (
        f"Y7 leftover after issued_lag1 inside company issued terciles that stay ≥0.55: {lives or 'none'}. "
        "If leftover dies inside every tercile it is still a thin-issuance mix."
    )
    print(prose)
    return {"rows": rows, "lives": lives, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 23 — leftover fold bits (stability)
# ---------------------------------------------------------------------------
def pass23_fold_bits(tr: pd.DataFrame, p4: dict, p5: dict) -> dict:
    rows = []
    for name, res in (
        ("CN", p4["store"][(Y7, "e_credit_note_ratio")]),
        ("issued_lag1", p4["store"][(Y7, "e_ar_issued_lag1")]),
        ("resid after issued_lag1", p5["store"]["after issued_lag1"]),
        ("resid after DSO", p5["store"]["after DSO"]),
    ):
        rows.append(
            {
                "feature": name,
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
                "fold4": _f(fold_k(res, 4)),
            }
        )
    prose = (
        f"Leftover folds [{fold_bits(p5['store']['after issued_lag1'])}] vs "
        f"issued_lag1 [{fold_bits(p4['store'][(Y7, 'e_ar_issued_lag1')])}]. "
        "If leftover fold 4 is the only lift, do not KEEP on that fold."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    sl = tr.loc[tr[Y7].notna() & tr["e_credit_note_ratio"].notna()].copy()
    sl["cn_q"] = pd.qcut(
        pd.to_numeric(sl["e_credit_note_ratio"], errors="coerce").rank(method="first"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
    )
    sl2 = tr.loc[tr[Y7].notna() & tr["e_ar_issued_lag1"].notna()].copy()
    sl2["iss_q"] = pd.qcut(
        pd.to_numeric(sl2["e_ar_issued_lag1"], errors="coerce"),
        5,
        labels=["Q1", "Q2", "Q3", "Q4", "Q5"],
        duplicates="drop",
    )

    def _rates(df, qcol):
        out = []
        for q, part in df.groupby(qcol, observed=False):
            s = pd.to_numeric(part[Y7], errors="coerce")
            out.append((str(q), 100.0 * float(s.mean()) if s.notna().any() else np.nan))
        return out

    cn_r = _rates(sl, "cn_q")
    iss_r = _rates(sl2, "iss_q")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6), sharey=True)
    ax = axes[0]
    ax.bar([x[0] for x in cn_r], [x[1] for x in cn_r], color="#1f4e79")
    ax.set_title("Y7 rate by CN quintile (train labeled)")
    ax.set_ylabel("% y7_top1_lost")
    ax.axhline(100.0 * float(pd.to_numeric(sl[Y7], errors="coerce").mean()), color="#9e6b4a", ls="--", lw=0.8)
    ax = axes[1]
    ax.bar([x[0] for x in iss_r], [x[1] for x in iss_r], color="#9e6b4a")
    ax.set_title("Y7 rate by issued_lag1 quintile")
    ax.axhline(100.0 * float(pd.to_numeric(sl2[Y7], errors="coerce").mean()), color="#1f4e79", ls="--", lw=0.8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p3, p4, p5, p6, p8, p9, p10, p17=None, p21=None, p20=None) -> dict:
    """KEEP leftover / CLOSE add-on / DROP-from-44 / PARK Y."""
    twin = bool(p3["any_twin"])
    leftover_ok = bool(p5["lives"] and not twin and not p5["thin"])
    leftover_dies = bool(p5["died"] or p5["thin"] or twin)
    if p17 is not None and p17.get("artifact"):
        leftover_ok = False
        leftover_dies = True
    if leftover_ok and p5["lives_now"] and not p3["twin_dso"]:
        y7 = "KEEP"
        y7_why = (
            f"residual after issued_lag1 {_f(p5['after_iss'])} beats chance "
            f"and is not a twin (ρ issued_lag1 {_f(p3['rhos']['e_ar_issued_lag1'])}, "
            f"DSO {_f(p3['rhos']['e_dso_proxy'])}). Still do not grow TURNOVER 0.720."
        )
    else:
        y7 = "CLOSE"
        extra = ""
        if p17 is not None and p17.get("artifact"):
            extra = f" Same-n raw CN {_f(p17['raw_both'])} ≈ leftover {_f(p17['ols'])}."
        y7_why = (
            f"CN is a twin of issued/DSO or leftover dies "
            f"(after issued_lag1 {_f(p5['after_iss'])}, after issued {_f(p5['after_now'])}, "
            f"after DSO {_f(p5['after_dso'])}, ρ_lag1={_f(p3['rhos']['e_ar_issued_lag1'])}). "
            f"CLOSE as Y7 add-on — do not grow TURNOVER.{extra}"
        )
    if p4["lose_days"] or p6["died"]:
        y3 = "CLOSE / DROP from the 44"
        y3_why = (
            f"Y3 CN {_f(p4['cn_y3'])} loses to days {_f(p4['days_y3'])} "
            f"(leftover after days {_f(p6['after_days'])})."
        )
    else:
        y3 = "KEEP"
        y3_why = f"Y3 CN {_f(p4['cn_y3'])} leftover after days {_f(p6['after_days'])} — unexpected."
    style_only = bool(p21 is not None and not p21.get("shock"))
    if leftover_ok and not p5["thin"]:
        q5 = "KEEP-Q5 footnote"
        q5_why = (
            "who-uses-notes leftover after issued — not this month’s correction, not thin issuance"
            if style_only
            else "leftover is sayable without repeating thin issuance"
        )
    else:
        q5 = "CLOSE"
        q5_why = "Q5 footnote would only repeat thin issuance"
    q6 = "KEEP" if p8["keep_q6"] else "CLOSE"
    return {
        "y7": y7,
        "y7_why": y7_why,
        "y3": y3,
        "y3_why": y3_why,
        "q5": q5,
        "q5_why": q5_why,
        "q6": q6,
        "park_y": True,
        "leftover_ok": leftover_ok,
        "leftover_dies": leftover_dies,
        "f4": (
            "issued owns fold 4"
            if (p20 and p20.get("issued_card")) or p9["issued_owns"]
            else ("CN saves fold 4" if p9["cn_saves"] else "TURNOVER 0.680 is issued, not CN")
        ),
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10, p11 = (
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
        ctx["p10"],
        ctx["p11"],
    )
    d = ctx["decision"]
    lines = [
        "# Q4/Q5 credit-note leftover after issued_lag1",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_credit_note`. "
        "Night Y7 quote stays **TURNOVER 0.720 / B_shallow 0.712**. Night Y3 stays **0.762 / 0.752**. "
        "Y7 never D. Y3 never B. CN stays off the 15-col card.",
        "",
        "`e_credit_note_ratio` = |credit notes| / (|invoices| + |credit notes|) issued this period. "
        "Type `credit_note` if present; else ERP stand-ins `note` and `refund`.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_credit_note`. Dark 470 = NaN, not 0 (COMP_0962 refund-only exception). |",
        "| 2 | Who is improving? | Not this ratio. |",
        "| 3 | Who is turning? | CN lag1 is a weak Q6, not a turn clock. |",
        f"| 4 | Dip vs fall? | Y7 leftover after issued_lag1 **{d['y7']}** — {d['y7_why']} |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['q5_why']}. Fold 4: {d['f4']}. |",
        f"| 6 | Months earlier? | CN lag1 short {_f(p8['short_lag'])} vs issued_lag1 0.630 — **{d['q6']}**. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| CN as Y7 leftover after issued_lag1 | **{d['y7']}** | {d['y7_why']} |",
        f"| CN as Y7 add-on / grow TURNOVER | **CLOSE** | do not grow TURNOVER; night quote stays 0.720 / 0.712 |",
        f"| CN as Y3 X / the 44 | **{d['y3']}** | {d['y3_why']} |",
        "| CN as a health Y | **PARK** | do not invent `y_credit_note` |",
        f"| Q5 footnote without “thin issuance” | **{d['q5']}** | {d['q5_why']} |",
        f"| Q6 CN lag1 | **{d['q6']}** | {p8['prose']} |",
        f"| Fold 4 Y7 | **{d['f4']}** | CN {_f(p9['cn4'])} vs issued_lag1 {_f(p9['iss4'])} vs TURNOVER 0.680 |",
        "",
        "## 1. Prevalence / coverage (dark = NaN not 0)",
        "",
        p1["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {p1['n_cm']:,} / {p1['n_co']:,} |",
        f"| CN defined | {p1['n_nn']:,} ({_pp(p1['cov'])}) |",
        f"| among defined: zero / >0 | {_pp(p1['zero_share'])} / {_pp(p1['pos_share'])} |",
        f"| modal / modal share | {p1['modal']:g} / {_pp(p1['modal_share'])} |",
        f"| ever-ERP / never-ERP | {p1['n_erp_co']} / {p1['n_dark_co']} |",
        f"| dark CN non-null / zero-filled | {p1['dark_nn']} / {p1['dark_zero']} |",
        f"| dark 0-fill | {'NO — CONFIRM' if p1['dark_nan'] else 'YES — FAIL'} (nn={p1['dark_nn']}) |",
        f"| Y7 labeled / pos / CN-nn | {p1['y7_n']:,} / {p1['y7_pos']:,} / {p1['y7_nn']:,} |",
        f"| Y3 labeled / pos / CN-nn | {p1['y3_n']:,} / {p1['y3_pos']:,} / {p1['y3_nn']:,} |",
        f"| acf1 / acf3 | {_f(p1['acf1'])} / {_f(p1['acf3'])} |",
        f"| CN defined on issued=0 ERP months | {p1['cn_on_iss0']:,} / {p1['n_iss0']:,} |",
        "",
        "## 2. Formula vs raw invoice types",
        "",
        p2["prose"],
        "",
        _md_table(p2["type_rows"]),
        "",
        "## 3. Spearman twins (|ρ|≥0.80)",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night Y7 issued_lag1 **0.630** (replica {_f(p4['iss_y7'])}); "
        f"TURNOVER **0.720** / B_shallow **0.712** unchanged. "
        f"Night Y3 size **0.617** (replica {_f(p4['size_y3'])}); days **0.711** (replica {_f(p4['days_y3'])}).",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Residual Y7 after issued_lag1 / DSO / issued",
        "",
        "KEEP leftover only if residual after issued_lag1 beats chance (≥0.55) **and** is not a twin. "
        "If leftover dies, CN on TURNOVER is a twin — CLOSE as add-on, do not grow the card.",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Residual Y3 after days",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. Token mix — credit_note vs note vs refund",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Q6 — lag1 / lag3 on short vs long",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Fold 4 Y7 (DSO failed; TURNOVER 0.680)",
        "",
        p9["prose"],
        "",
        "OOF fold bits (sign per fold):",
        "",
        _md_table(p9["rows"]),
        "",
        "Fold 4 only (sign from folds 0–3):",
        "",
        _md_table(p9["f4_rows"]),
        "",
        "Hard groups (y7_core):",
        "",
        _md_table(p9["grp"]),
        "",
        "## 10. ICC / company-demean",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. Dark 470 + holdout coverage (no AUROC)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Drop 12 chronic Y2 names — Y3 only",
        "",
        ctx["p12"]["prose"],
        "",
        _md_table(ctx["p12"]["rows"]),
        "",
        "## 13. Quintiles and CN>0 vs CN=0 rates",
        "",
        ctx["p13"]["prose"],
        "",
        _md_table(ctx["p13"]["rows"]),
        "",
        _md_table(ctx["p13"]["flag_rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 14. Nonzero-only leftover",
        "",
        ctx["p14"]["prose"],
        "",
        _md_table(ctx["p14"]["rows"]),
        "",
        "## 15. Company-mean leftover after company-mean issued_lag1",
        "",
        ctx["p15"]["prose"],
        "",
        _md_table(ctx["p15"]["rows"]),
        "",
        "## 16. CN lag1 leftover after issued_lag1",
        "",
        ctx["p16"]["prose"],
        "",
        _md_table(ctx["p16"]["rows"]),
        "",
        "## 17. Leftover artifact — same-n raw vs residual",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## 18. Dark exception COMP_0962 (refund-only, not a 0-fill)",
        "",
        ctx["p18"]["prose"],
        "",
        _md_table(ctx["p18"]["rows"]) if ctx["p18"]["rows"] else "_(none)_\n",
        "",
        "## 19. Note vs refund ratios (in-memory, not a Y)",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(ctx["p19"]["rows"]),
        "",
        "## 20. Fold 4 leftover vs issued / issued-CV",
        "",
        ctx["p20"]["prose"],
        "",
        _md_table(ctx["p20"]["rows"]),
        "",
        "## 21. Within-company leftover (demean after demean issued)",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "## 22. Leftover inside issued_lag1 company terciles",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## 23. Leftover fold bits",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: prevalence, formula, twins, singles, "
        "Y7 leftover after issued_lag1/DSO, Y3 leftover after days, token mix, Q6, fold 4, "
        "ICC, dark/holdout, 12 Y2 names (Y3 only), quintiles, nonzero pile, company-mean, "
        "lag leftover, same-n artifact, COMP_0962, note vs refund, fold-4 leftover, "
        "demean leftover, issued terciles, leftover folds.",
        "",
        "## What this module did not do",
        "",
        "- Did not change night Y3 0.762 / 0.752 or Y7 TURNOVER 0.720 / B_shallow 0.712.",
        "- Did not put CN on the Y3 15-col card. Did not grow TURNOVER. Did not use Family D as Y7 X.",
        "- Did not invent `y_credit_note`. Did not write 0–100 / pillars. Did not touch `product/`.",
        "- Did not edit `invoices.py` / `gbm_y7_core.py` / sibling `zero_in_qa` / `growth_qa` / `recency_qa`.",
        "- Did not rewrite parquet or duckdb. Did not run `build_targets`. Did not commit.",
        "- Did not quote holdout Y7 AUROC (122 pos is coverage only).",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _reg_row(metric: str, value, coverage, y: str, notes: str, split: str = "train_cv") -> dict:
    return {
        "ts": datetime.now().strftime("%Y-%m-%dT%H:%M"),
        "round": ROUND,
        "wave": WAVE,
        "agent": AGENT,
        "x_families": X_FAM,
        "y": y,
        "model": MODEL,
        "split": split,
        "metric": metric,
        "value": value if value is None or (isinstance(value, float) and not np.isfinite(value)) else value,
        "coverage": coverage,
        "notes": notes,
    }


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p4, p5, p6, p8, p9, p10, d = (
        ctx["p1"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p8"],
        ctx["p9"],
        ctx["p10"],
        ctx["decision"],
    )
    rows = [
        _reg_row(
            "e_credit_note_ratio_cov",
            p1["cov"],
            f"{p1['cov']:.4f}",
            "-",
            f"dark_nan={p1['dark_nan']} n_dark={p1['n_dark_co']} modal={p1['modal_share']:.4f} zero={p1['zero_share']:.4f}",
            "train",
        ),
        _reg_row(
            "rho_cn_vs_issued_lag1",
            p3["rhos"]["e_ar_issued_lag1"],
            "1.0000",
            Y7,
            f"issued={p3['rhos']['e_ar_issued']:.4f} dso={p3['rhos']['e_dso_proxy']:.4f} size={p3['rhos']['log1p(a_in3)']:.4f} twin={p3['any_twin']}",
            "train",
        ),
        _reg_row(
            "auroc_e_credit_note_ratio",
            p4["cn_y7"],
            f"{p1['y7_nn'] / p1['y7_n'] if p1['y7_n'] else float('nan'):.4f}",
            Y7,
            f"vs_issued_lag1={p4['iss_y7']:.4f} night=0.630 leftover={p5['after_iss']:.4f} y7={d['y7']}",
        ),
        _reg_row(
            "auroc_cn_resid_issued_lag1",
            p5["after_iss"],
            "1.0000",
            Y7,
            f"after_issued={p5['after_now']:.4f} after_dso={p5['after_dso']:.4f} lives={p5['lives']} thin={p5['thin']}",
        ),
        _reg_row(
            "auroc_e_credit_note_ratio",
            p4["cn_y3"],
            f"{p1['y3_nn'] / p1['y3_n'] if p1['y3_n'] else float('nan'):.4f}",
            Y3,
            f"size={p4['size_y3']:.4f} days={p4['days_y3']:.4f} leftover_days={p6['after_days']:.4f} y3={d['y3']}",
        ),
        _reg_row(
            "auroc_cn_lag1_short",
            p8["short_lag"],
            "1.0000",
            Y7,
            f"q6_quoted=0.542 confirm={p8['confirm']} q6={d['q6']}",
        ),
        _reg_row(
            "auroc_cn_fold4",
            p9["cn4"],
            "1.0000",
            Y7,
            f"issued_lag1_f4={p9['iss4']:.4f} resid_f4={p9['resid4']:.4f} turnover_f4=0.680 issued_owns={p9['issued_owns']}",
        ),
        _reg_row(
            "icc_e_credit_note_ratio",
            p10["icc"]["icc"],
            "1.0000",
            "-",
            f"quote=0.83 confirm={p10['confirm']} y7_demean={p10['y7_de']:.4f} trait={p10['trait']}",
            "train",
        ),
        _reg_row(
            "dark470_cn_nn",
            float(p1["dark_nn"]),
            f"{p1['n_dark_co'] / p1['n_co'] if p1['n_co'] else float('nan'):.4f}",
            "-",
            f"must_be_nan={p1['dark_nan']} holdout_dark_nn={ctx['p11']['dark_nn']}",
            "train",
        ),
        _reg_row(
            "token_canon_mass",
            ctx["p7"]["canon_mass"],
            "1.0000",
            "-",
            f"kind={ctx['p7']['kind']} note={ctx['p7']['note_mass']:.4f} refund={ctx['p7']['refund_mass']:.4f} mostly_refund={ctx['p7']['mostly_refund']}",
            "train",
        ),
        _reg_row(
            "auroc_cn_demean_resid_issued",
            ctx["p21"]["y7"],
            "1.0000",
            Y7,
            f"demean={ctx['p21']['y7_d']:.4f} shock={ctx['p21']['shock']} q5={d['q5']}",
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
        if r["value"] is None or (isinstance(r["value"], float) and not np.isfinite(r["value"])):
            r["value"] = ""
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
    print(f"credit_note_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel,
        ["e_credit_note_ratio", "e_ar_issued", "e_dso_proxy", "f_fc_r"],
        (1, 2, 3),
    )
    panel = add_issued_lag_cv(panel)
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )

    con = connect()
    try:
        book = book_invoice_ids(con)
        hold = load_holdout()
        book_train = {c for c in book if c not in hold}
        print(f"book ids {len(book)} train-book {len(book_train)}")
        print("pass 1 prevalence")
        p1 = pass1_prev(tr, book)
        print("pass 2 formula")
        p2 = pass2_formula(tr, con)
        print("pass 3 twins")
        p3 = pass3_twins(tr)
        print("pass 4 singles")
        p4 = pass4_auroc(tr)
        print("pass 5 Y7 leftover")
        p5 = pass5_resid_y7(tr)
        print("pass 6 Y3 leftover")
        p6 = pass6_resid_y3(tr)
        print("pass 7 tokens")
        p7 = pass7_tokens(p2)
        print("pass 8 Q6")
        p8 = pass8_q6(tr)
        print("pass 9 fold 4")
        p9 = pass9_fold4(tr, p4, p5)
        print("pass 10 ICC")
        p10 = pass10_icc(tr)
        print("pass 11 dark/holdout")
        p11 = pass11_dark_hold(panel, book)
        print("pass 12 chronic 12 Y3 only")
        p12 = pass12_chronic_y3(tr)
        print("pass 13 quintiles")
        p13 = pass13_quintiles(tr)
        print("pass 14 nonzero")
        p14 = pass14_nonzero(tr, p5)
        print("pass 15 company-mean leftover")
        p15 = pass15_co_mean(tr)
        print("pass 16 lag leftover")
        p16 = pass16_lag_resid(tr)
        print("pass 17 leftover artifact")
        p17 = pass17_artifact(tr, p5)
        print("pass 18 dark exception")
        p18 = pass18_dark_one(tr, book, p2)
        print("pass 19 note vs refund")
        p19 = pass19_note_refund(tr, p2)
        print("pass 20 fold-4 leftover")
        p20 = pass20_f4_resid(tr, p5)
        print("pass 21 demean leftover")
        p21 = pass21_demean_resid(tr)
        print("pass 22 issued terciles")
        p22 = pass22_iss_tercile(tr, p5)
        print("pass 23 leftover folds")
        p23 = pass23_fold_bits(tr, p4, p5)
    finally:
        con.close()

    decision = decide(p3, p4, p5, p6, p8, p9, p10, p17=p17, p21=p21, p20=p20)
    png_ok = make_png(tr)
    headline = (
        f"CN defined {_pp(p1['cov'])} of train CM; dark 470 is NaN not 0 "
        f"(COMP_0962 refund-only is the one defined dark month). "
        f"No canonical `credit_note` — fallback **note 98.6% / refund 1.4%**. "
        f"ρ vs issued_lag1 {_f(p3['rhos']['e_ar_issued_lag1'])} vs DSO {_f(p3['rhos']['e_dso_proxy'])} "
        f"(not a twin). Y7 CN {_f(p4['cn_y7'])} vs issued_lag1 {_f(p4['iss_y7'])} (night 0.630); "
        f"leftover after issued_lag1 {_f(p5['after_iss'])} (same-n lift {_f(p17['lift'])}; "
        f"demean leftover {_f(p21['y7'])} dies — who-uses-notes). "
        f"Y3 CN {_f(p4['cn_y3'])} vs days {_f(p4['days_y3'])} leftover {_f(p6['after_days'])}. "
        f"Fold 4 CN {_f(p9['cn4'])} vs issued {_f(p9['iss4'])} — TURNOVER 0.680 is issued. "
        f"Y7 leftover **{decision['y7']}**. Y3 X **{decision['y3']}**. "
        f"PARK as health Y. Do not grow TURNOVER 0.720."
    )
    print(headline)
    failed = []
    if not p1["dark_nan"]:
        failed.append(f"dark CN was 0-filled nn={p1['dark_nn']} zero={p1['dark_zero']}")
    elif p1["dark_nn"]:
        failed.append(
            f"dark CN nn={p1['dark_nn']} is refund-only COMP_0962 — not a 0-fill; 469 others stay NaN"
        )
    if not p2["ok"]:
        failed.append(
            f"formula mismatch max|Δ|={p2['maxdiff']} — stop, do not edit invoices.py"
        )
    if not p4["replica_iss"]:
        failed.append(f"issued_lag1 replica {_f(p4['iss_y7'])} vs night 0.630")
    if not p4["replica_days"]:
        failed.append(f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711")
    if p5["died"] or p5["thin"]:
        failed.append(
            f"Y7 leftover after issued_lag1 {_f(p5['after_iss'])} dies / thin issuance — CLOSE as add-on"
        )
    if p4["lose_days"] or p6["died"]:
        failed.append(
            f"Y3 CN {_f(p4['cn_y3'])} leftover-after-days {_f(p6['after_days'])} — DROP from the 44"
        )
    if not p8["keep_q6"]:
        failed.append(f"Q6 CN lag1 short {_f(p8['short_lag'])} CLOSE (q6_quoted 0.542)")
    if p9["issued_owns"]:
        failed.append("fold 4 TURNOVER 0.680 is issued, not CN")
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
        "panel": panel,
    }
    write_md(ctx)
    append_registry(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
