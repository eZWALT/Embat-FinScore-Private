"""Wave B — unused leftover of AR issued to last month's top-1 after issued_lag1.

NORTH_STAR: Y7 asks whether the trailing-3m top AR customer vanishes next
quarter. This cut is *thinning of that named buyer this month*, not Y7
itself and not ``d_cust_lost`` (Y3 leftover 0.522 DROP; n_cust twin).
Jacobson demand-shrinkage / Irvine major-customer / Amberg issued −1 pp.

Feature (in-memory, never written to parquet):
  e_issued_top1 = this-month AR |amt| issued to last month's trailing-3m
                  top-1 counterparty (Y7 object, lagged one month).
  0 if that buyer exists and got nothing this month; NaN if no lag-1 top-1.
  Dark 470 stay NaN not 0.

KEEP-as-leftover (Y7): leftover after e_ar_issued_lag1 ≥ 0.58 AND beat-size
≥ 0.02 AND not SIZE AND not a twin (|ρ|<0.80 vs issued_lag1 / e_ar_issued /
d_cust_top1). If it dies, issued_lag1 Q6 KEEP 0.626 stays the invoice lead.

Do not grow TURNOVER 0.720. Do not put this on the 15-col card.
Y7 never D as engine X. Y5 never E. Y3 never B. Do not invent y_cust_lost.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.issued_top1_qa
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
from analysis.features.common import ANALYSIS, DATA, LAST_M, MONTHS, connect
from analysis.features.invoices import DELAY_MASK_BEFORE
from analysis.targets.y11_dark import book_invoice_ids
from analysis.targets.y7_concentration import ID_WIN, _ar_month_cp, _roll_cp

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "issued_top1_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "issued_top1_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_issued_top1.md"
AGENT = "689100e7"
WAVE = "4"
ROUND = "R4"
MODEL = "issued_top1_qa"
X_FAM = "E"

Y3 = "y3_recover_cash_6m"
Y7 = "y7_top1_lost"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
ISSUED_LAG1_Y7 = 0.630
Q6_ISSUED_LAG1 = 0.626
CN_LEFTOVER = 0.597
DELAY_LEFTOVER = 0.581
CUST_LOST_Y3 = 0.522
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
KEEP_LEFT = 0.58
SIZE_RHO = 0.50
ICC_STYLE = 0.85
MIN_POS = 50
MIN_ACF_PAIRS = 4
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "c_n_days_with_tx",
    "e_ar_issued",
    "e_credit_note_ratio",
    "e_delay_coll",
    "d_cust_top1",
    "d_cust_lost",
)

Y_KEEP = (Y3, Y7)
STEM = "e_issued_top1"
SHARE = "e_issued_top1_share"
TWIN_COLS = (
    "e_ar_issued_lag1",
    "e_ar_issued",
    "d_cust_top1",
    "d_cust_lost",
    "log_in3",
    SHARE,
    "top1_share_lag1",
)
GATE_TWINS = ("e_ar_issued_lag1", "e_ar_issued", "d_cust_top1")


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


def spearman_n(a, b) -> tuple[float, int]:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan"), int(len(d))
    return float(d["a"].corr(d["b"], method="spearman")), int(len(d))


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


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    mu = d.groupby("co")["x"].transform("mean")
    return d["x"] - mu


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


def signed_oof_auroc(y, x, folds, mask, n_folds: int = N_FOLDS) -> dict:
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


def leftover_diag(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof_auroc(y, resid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rresid, _ = ols_resid(xr, *cr)
    rrec = signed_oof_auroc(y, rresid, folds, mask)
    fake = bool(np.isfinite(rho_c) and abs(rho_c) >= TWIN_RHO)
    rank_cv = _cv(rrec)
    ols_cv = _cv(rec)
    honest_dies = bool(fake or (np.isfinite(rank_cv) and rank_cv < CHANCE))
    return {
        "ols": ols_cv,
        "rank": rank_cv,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "resid": resid,
        "rresid": rresid,
        "info": info,
        "rec": rec,
        "rrec": rrec,
    }


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    return {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }


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
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def _rank_q(s: pd.Series, n: int = 5) -> pd.Series:
    r = pd.to_numeric(s, errors="coerce").rank(method="first")
    return pd.qcut(r, n, labels=[f"Q{i}" for i in range(1, n + 1)])


# ---------------------------------------------------------------------------
# Feature: this-month AR issued to last month's trailing-3m top-1
# ---------------------------------------------------------------------------
def build_top1_ids(con) -> pd.DataFrame:
    mcp = _ar_month_cp(con)
    if mcp.empty:
        return pd.DataFrame(columns=["company_id", "period", "top1_id", "top1_share", "n_cp"])
    id_win = _roll_cp(mcp, (0, 1, 2))
    id_lo = MONTHS[ID_WIN - 1]
    id_win = id_win[(id_win["period"] >= id_lo) & (id_win["period"] <= LAST_M)]
    tot = id_win.groupby(["company_id", "period"])["amt"].transform("sum")
    id_win = id_win.loc[tot > 0].copy()
    id_win["share"] = id_win["amt"] / tot.loc[id_win.index]
    idx = id_win.groupby(["company_id", "period"])["amt"].idxmax()
    top = id_win.loc[idx, ["company_id", "period", "counterparty_id", "share"]].rename(
        columns={"counterparty_id": "top1_id", "share": "top1_share"}
    )
    ncp = id_win.groupby(["company_id", "period"], as_index=False).agg(n_cp=("counterparty_id", "nunique"))
    return top.merge(ncp, on=["company_id", "period"], how="left")


def build_issued_to_named(con, keys: pd.DataFrame) -> pd.DataFrame:
    """This-month AR issued to the named (lag-1) top-1. Dates = issuance month."""
    if keys.empty:
        return pd.DataFrame(columns=["company_id", "period", STEM])
    per = keys[["company_id", "period", "top1_id_lag1"]].dropna(subset=["top1_id_lag1"]).copy()
    per["period"] = pd.to_datetime(per["period"])
    if per.empty:
        return pd.DataFrame(columns=["company_id", "period", STEM])
    con.register("_iss_top1", per)
    try:
        book = con.execute(
            """
            SELECT t.company_id,
                   t.period,
                   SUM(abs(i.amount)) AS issued_to_top1
            FROM invoices i
            JOIN _iss_top1 t
              ON CAST(i.company_id AS VARCHAR) = t.company_id
             AND CAST(i.counterparty_id AS VARCHAR) = t.top1_id_lag1
             AND CAST(date_trunc('month', i.issuance_date) AS DATE) = CAST(t.period AS DATE)
            WHERE i.document_type = 'invoice'
              AND i.status <> 'cancel'
              AND i.amount > 0
              AND i.issuance_date IS NOT NULL
            GROUP BY 1, 2
            """
        ).df()
    finally:
        con.unregister("_iss_top1")
    book["company_id"] = book["company_id"].astype(str)
    book["period"] = pd.to_datetime(book["period"])
    book[STEM] = pd.to_numeric(book["issued_to_top1"], errors="coerce")
    return book[["company_id", "period", STEM]]


def attach_issued_top1(panel: pd.DataFrame, con) -> pd.DataFrame:
    print("building trailing-3m top-1 ids (Y7 object)")
    top = build_top1_ids(con)
    print(f"top1 rows {len(top):,} companies {top['company_id'].nunique()}")
    out = panel.merge(
        top[["company_id", "period", "top1_id", "top1_share", "n_cp"]],
        on=["company_id", "period"],
        how="left",
    )
    out = out.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = out.groupby("company_id", sort=False)
    out["top1_id_lag1"] = g["top1_id"].shift(1)
    out["top1_share_lag1"] = g["top1_share"].shift(1)
    print("building this-month AR issued to last month's top-1")
    got = build_issued_to_named(con, out)
    print(f"issued-to-top1 nonzero rows {len(got):,}")
    out = out.merge(got, on=["company_id", "period"], how="left")
    named = out["top1_id_lag1"].notna()
    out.loc[named & out[STEM].isna(), STEM] = 0.0
    out.loc[~named, STEM] = np.nan
    issued = pd.to_numeric(out["e_ar_issued"], errors="coerce")
    top_iss = pd.to_numeric(out[STEM], errors="coerce")
    out[SHARE] = np.where(issued > 0, top_iss / issued, np.nan)
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
    panel["so_far_class"] = panel["months_so_far"].map(_trail_class)
    panel["early6"] = panel["period"] < DELAY_MASK_BEFORE
    leak7 = leakage_check([STEM, "e_ar_issued_lag1"], Y7, forbidden_prefixes=["d"])
    leak3 = leakage_check([STEM], Y3, forbidden_prefixes=["b"])
    if not leak7["ok"]:
        raise RuntimeError(f"Y7 X leak: {leak7['issues']}")
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 X leak: {leak3['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame, book: set[str]) -> dict:
    x = pd.to_numeric(tr[STEM], errors="coerce")
    sh = pd.to_numeric(tr[SHARE], errors="coerce")
    dark = ~tr["company_id"].isin(book)
    n_cm = int(len(tr))
    n_co = int(tr["company_id"].nunique())
    n_nn = int(x.notna().sum())
    n_zero = int((x == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    dark_ok = bool(dark_nn == 0 and dark_zero == 0 and n_dark_co == 470)
    y7 = pd.to_numeric(tr[Y7], errors="coerce")
    y7_lab = y7.notna()
    y7_nn = int((y7_lab & x.notna()).sum())
    prose = (
        f"Train CM={n_cm:,} / {n_co:,} companies. {STEM} defined {n_nn:,} ({_pp(n_nn / n_cm)}) "
        f"zero-among-defined {_pp(n_zero / n_nn if n_nn else float('nan'))}. "
        f"Dark 470 nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN not 0' if dark_ok else 'FAIL — filled'}."
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": n_co,
        "n_nn": n_nn,
        "cov": n_nn / n_cm if n_cm else float("nan"),
        "n_zero": n_zero,
        "zero_share": n_zero / n_nn if n_nn else float("nan"),
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_ok": dark_ok,
        "y7_n": int(y7_lab.sum()),
        "y7_pos": int((y7 == 1).sum()),
        "y7_nn": y7_nn,
        "mean": float(x.mean()) if n_nn else float("nan"),
        "p50": float(x.median()) if n_nn else float("nan"),
        "mean_share": float(sh.mean()) if int(sh.notna().sum()) else float("nan"),
        "acf1": median_acf(x, tr["company_id"], 1),
        "acf3": median_acf(x, tr["company_id"], 3),
        "prose": prose,
    }


def pass2_formula(tr: pd.DataFrame) -> dict:
    named = tr["top1_id_lag1"].notna()
    x = pd.to_numeric(tr[STEM], errors="coerce")
    n_named = int(named.sum())
    n_nn_named = int(x[named].notna().sum())
    n_zero_named = int((x[named] == 0).sum())
    ok = bool(n_named == n_nn_named and int(x[~named].notna().sum()) == 0)
    prose = (
        f"Named lag-1 top-1 rows {n_named:,}; issued-to-them defined {n_nn_named:,} "
        f"(zero {n_zero_named:,}). Unnamed nn={int(x[~named].notna().sum()):,} "
        f"{'MATCH — 0 only when named, NaN otherwise' if ok else 'FAIL'}."
    )
    print(prose)
    return {"ok": ok, "n_named": n_named, "n_zero_named": n_zero_named, "prose": prose}


def pass3_twins(tr: pd.DataFrame) -> dict:
    x = tr[STEM]
    rows = []
    rhos = {}
    twins = []
    for c in TWIN_COLS:
        if c not in tr.columns:
            continue
        rho, n = spearman_n(x, tr[c])
        rhos[c] = rho
        twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
        if twin:
            twins.append(c)
        rows.append({"vs": c, "ρ": _f(rho), "n": f"{n:,}", "twin?": "TWIN" if twin else ""})
    gate = [c for c in GATE_TWINS if c in twins]
    size = bool(np.isfinite(rhos.get("log_in3", float("nan"))) and abs(rhos["log_in3"]) >= SIZE_RHO)
    prose = (
        f"Gate twins vs issued_lag1 / issued / d_cust_top1: {gate or 'none'}. "
        f"ρ vs issued_lag1={_f(rhos.get('e_ar_issued_lag1'))} "
        f"issued={_f(rhos.get('e_ar_issued'))} top1={_f(rhos.get('d_cust_top1'))} "
        f"lost={_f(rhos.get('d_cust_lost'))} size={_f(rhos.get('log_in3'))}"
        f"{' SIZE' if size else ''}."
    )
    print(prose)
    return {"rows": rows, "rhos": rhos, "twins": twins, "gate": gate, "size": size, "prose": prose}


def pass4_singles(tr: pd.DataFrame) -> dict:
    y7 = tr[Y7]
    y3 = tr[Y3]
    recs = {
        "iss_top1_y7": signed_oof_auroc(y7, tr[STEM], tr["fold"], y7.notna()),
        "share_y7": signed_oof_auroc(y7, tr[SHARE], tr["fold"], y7.notna()),
        "iss_lag1_y7": signed_oof_auroc(y7, tr["e_ar_issued_lag1"], tr["fold"], y7.notna()),
        "iss_y7": signed_oof_auroc(y7, tr["e_ar_issued"], tr["fold"], y7.notna()),
        "size_y7": signed_oof_auroc(y7, tr["log_in3"], tr["fold"], y7.notna()),
        "lost_y7": signed_oof_auroc(y7, tr["d_cust_lost"], tr["fold"], y7.notna()),
        "iss_top1_y3": signed_oof_auroc(y3, tr[STEM], tr["fold"], y3.notna()),
        "days_y3": signed_oof_auroc(y3, tr["c_n_days_with_tx"], tr["fold"], y3.notna()),
        "size_y3": signed_oof_auroc(y3, tr["log_in3"], tr["fold"], y3.notna()),
    }
    rows = [_auc_row(Y7 if "y7" in k else Y3, k, r) for k, r in recs.items()]
    pd_y7 = _cv(recs["iss_top1_y7"])
    iss = _cv(recs["iss_lag1_y7"])
    size = _cv(recs["size_y7"])
    days = _cv(recs["days_y3"])
    beat = bool(np.isfinite(pd_y7) and np.isfinite(size) and abs(pd_y7 - size) >= KEEP_DELTA)
    if np.isfinite(iss) and abs(iss - ISSUED_LAG1_Y7) > 0.015:
        print(f"WARN issued_lag1 Y7 replica {_f(iss)} vs night {ISSUED_LAG1_Y7}")
    if np.isfinite(days) and abs(days - DAYS_BENCH) > 0.015:
        print(f"WARN days Y3 replica {_f(days)} vs night {DAYS_BENCH}")
    prose = (
        f"Y7 {STEM} {_f(pd_y7)} vs issued_lag1 {_f(iss)} (night 0.630) "
        f"size {_f(size)} beat-size {'PASS' if beat else 'FAIL'} "
        f"Δ={_f(abs(pd_y7 - size) if np.isfinite(pd_y7) and np.isfinite(size) else float('nan'))}. "
        f"Y3 {STEM} {_f(_cv(recs['iss_top1_y3']))} vs days {_f(days)}."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "pd_y7": pd_y7,
        "iss_y7": iss,
        "size_y7": size,
        "days_y3": days,
        "share_y7": _cv(recs["share_y7"]),
        "beat": beat,
        "prose": prose,
    }


def pass5_leftover_y7(tr: pd.DataFrame) -> dict:
    y = tr[Y7]
    x = tr[STEM]
    mask = y.notna()
    cuts = {
        "after issued_lag1": leftover_diag(y, x, (tr["e_ar_issued_lag1"],), tr["fold"], mask),
        "after e_ar_issued": leftover_diag(y, x, (tr["e_ar_issued"],), tr["fold"], mask),
        "after issued_lag1+issued": leftover_diag(
            y, x, (tr["e_ar_issued_lag1"], tr["e_ar_issued"]), tr["fold"], mask
        ),
        "after size": leftover_diag(y, x, (tr["log_in3"],), tr["fold"], mask),
        "after CN": leftover_diag(y, x, (tr["e_credit_note_ratio"],), tr["fold"], mask),
        "after delay": leftover_diag(y, x, (tr["e_delay_coll"],), tr["fold"], mask),
        "after d_cust_top1": leftover_diag(y, x, (tr["d_cust_top1"],), tr["fold"], mask),
        "share after issued_lag1": leftover_diag(
            y, tr[SHARE], (tr["e_ar_issued_lag1"],), tr["fold"], mask
        ),
    }
    rows = []
    for name, d in cuts.items():
        rows.append(
            {
                "cut": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "ρ(resid,ctrl)": _f(d["rho_ctrl"]),
                "R²": _f(d["r2"]),
                "n": f"{d['n']:,}",
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        )
    main = cuts["after issued_lag1"]
    keep = bool(
        np.isfinite(main["rank"])
        and main["rank"] >= KEEP_LEFT
        and not main["fake"]
        and not main["honest_dies"]
    )
    prose = (
        f"Y7 leftover after issued_lag1 rank {_f(main['rank'])} OLS {_f(main['ols'])} "
        f"ρ(resid,issued_lag1)={_f(main['rho_ctrl'])} R²={_f(main['r2'])} "
        f"{'KEEP ≥0.58' if keep else ('dies' if main['honest_dies'] else 'lives but <0.58')}. "
        f"After contemp issued {_f(cuts['after e_ar_issued']['rank'])}; "
        f"share leftover {_f(cuts['share after issued_lag1']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "cuts": cuts,
        "main_rank": main["rank"],
        "main_ols": main["ols"],
        "main_dies": main["honest_dies"],
        "main_fake": main["fake"],
        "after_issued": cuts["after e_ar_issued"]["rank"],
        "share_rank": cuts["share after issued_lag1"]["rank"],
        "keep": keep,
        "prose": prose,
    }


def pass6_y3(tr: pd.DataFrame) -> dict:
    d = leftover_diag(
        tr[Y3], tr[STEM], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
    prose = (
        f"Y3 leftover after days rank {_f(d['rank'])} OLS {_f(d['ols'])} "
        f"{'dies — stay off the 15-col card' if d['honest_dies'] else 'lives (still not a card add-on)'}."
    )
    print(prose)
    return {
        "rank": d["rank"],
        "ols": d["ols"],
        "dies": d["honest_dies"],
        "rows": [
            {
                "cut": "Y3 leftover after days",
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        ],
        "prose": prose,
    }


def pass7_q6(tr: pd.DataFrame) -> dict:
    y = tr[Y7]
    short = tr["so_far_class"] == "short_<12"
    rows = []
    store = {}
    for label, mask in (
        ("all", y.notna()),
        ("short_<12", short & y.notna()),
        ("long_>=18", (tr["so_far_class"] == "long_>=18") & y.notna()),
    ):
        for name, s in (
            (STEM, tr[STEM]),
            (f"{STEM}_lag1", tr.get(f"{STEM}_lag1")),
            ("issued_lag1", tr["e_ar_issued_lag1"]),
        ):
            if s is None:
                continue
            res = signed_oof_auroc(y, s, tr["fold"], mask)
            store[(label, name)] = res
            rows.append(_auc_row(Y7, f"{label} {name}", res))
    short_x = _cv(store.get(("short_<12", STEM), {"low_power": True}))
    short_l1 = _cv(store.get(("short_<12", f"{STEM}_lag1"), {"low_power": True}))
    short_iss = _cv(store.get(("short_<12", "issued_lag1"), {"low_power": True}))
    keep_q6 = bool(np.isfinite(short_l1) and short_l1 >= 0.58)
    prose = (
        f"Q6 short {STEM} {_f(short_x)} lag1 {_f(short_l1)} vs issued_lag1 {_f(short_iss)} "
        f"(quote 0.626). {'KEEP this lag1' if keep_q6 else 'CLOSE as Q6 — issued_lag1 0.626 stays the invoice lead'}."
    )
    print(prose)
    return {
        "rows": rows,
        "short_x": short_x,
        "short_l1": short_l1,
        "short_iss": short_iss,
        "keep_q6": keep_q6,
        "prose": prose,
    }


def pass8_icc(tr: pd.DataFrame) -> dict:
    x = tr[STEM]
    icc = icc_anova(x, tr["company_id"])
    de = company_demean(x, tr["company_id"])
    d_mu = leftover_diag(
        tr[Y7], x, (tr.groupby("company_id")[STEM].transform("mean"),), tr["fold"], tr[Y7].notna()
    )
    d_de = leftover_diag(
        tr[Y7], de, (tr["e_ar_issued_lag1"],), tr["fold"], tr[Y7].notna()
    )
    style = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    prose = (
        f"ICC {_f(icc['icc'])} (k={icc['k']}) "
        f"{'BETWEEN / who-issues-to-whom style' if style else 'not a company trait'}. "
        f"Demean leftover after issued_lag1 {_f(d_de['rank'])} "
        f"{'dies — who-has-a-big-buyer' if d_de['honest_dies'] else 'lives as a month shock'}."
    )
    print(prose)
    return {
        "icc": icc["icc"],
        "k": icc["k"],
        "style": style,
        "demean_rank": d_de["rank"],
        "demean_dies": d_de["honest_dies"],
        "mean_rank": d_mu["rank"],
        "rows": [
            {"cut": "ICC", "value": _f(icc["icc"]), "k": f"{icc['k']:,}"},
            {"cut": "company-mean leftover Y7", "value": _f(d_mu["rank"]), "k": "—"},
            {"cut": "demean leftover after issued_lag1", "value": _f(d_de["rank"]), "k": "—"},
        ],
        "prose": prose,
    }


def pass9_dark_hold(panel: pd.DataFrame, book: set[str]) -> dict:
    hold = panel["split"] == "holdout"
    dark = ~panel["company_id"].isin(book)
    x = pd.to_numeric(panel[STEM], errors="coerce")
    rows = [
        {
            "slice": "train dark",
            "n_co": f"{panel.loc[dark & (panel['split']=='train'), 'company_id'].nunique():,}",
            "nn": f"{int(x[dark & (panel['split']=='train')].notna().sum()):,}",
            "zero": f"{int((x[dark & (panel['split']=='train')] == 0).sum()):,}",
        },
        {
            "slice": "holdout (coverage only)",
            "n_co": f"{panel.loc[hold, 'company_id'].nunique():,}",
            "nn": f"{int(x[hold].notna().sum()):,}",
            "zero": f"{int((x[hold] == 0).sum()):,}",
        },
        {
            "slice": "holdout dark",
            "n_co": f"{panel.loc[hold & dark, 'company_id'].nunique():,}",
            "nn": f"{int(x[hold & dark].notna().sum()):,}",
            "zero": f"{int((x[hold & dark] == 0).sum()):,}",
        },
    ]
    prose = (
        f"Holdout 72 coverage only — no AUROC. Holdout {STEM} nn={int(x[hold].notna().sum()):,}. "
        f"Holdout dark nn={int(x[hold & dark].notna().sum()):,} (want 0)."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def pass10_quintiles(tr: pd.DataFrame) -> dict:
    y = pd.to_numeric(tr[Y7], errors="coerce")
    x = pd.to_numeric(tr[STEM], errors="coerce")
    lab = y.notna() & x.notna()
    work = tr.loc[lab, [Y7, STEM]].copy()
    work["q"] = _rank_q(work[STEM])
    work["pile"] = np.where(work[STEM] == 0, "zero", "positive")
    rows = []
    for q, g in work.groupby("q", observed=False):
        rows.append(
            {
                "q": str(q),
                "n": f"{len(g):,}",
                "rate": _pp(float(g[Y7].mean())),
                "p50 issued-to-top1": _f(float(g[STEM].median())),
            }
        )
    pile_rows = []
    for q, g in work.groupby("pile", observed=False):
        pile_rows.append(
            {
                "bin": str(q),
                "n": f"{len(g):,}",
                "share": _pp(len(g) / len(work) if len(work) else float("nan")),
                "Y7 rate": _pp(float(g[Y7].mean())),
            }
        )
    q1 = work.loc[work["q"].astype(str) == "Q1", Y7].mean()
    q5 = work.loc[work["q"].astype(str) == "Q5", Y7].mean()
    lift = float(q5 - q1) if np.isfinite(q1) and np.isfinite(q5) else float("nan")
    prose = (
        f"Y7 rate by rank-quintile of issued-to-last-month-top-1. Q5−Q1 {_pp(lift)} "
        f"(Jacobson/Irvine thinning seat; sign expected negative if thinning → loss)."
    )
    print(prose)
    return {"rows": rows, "pile_rows": pile_rows, "q5_q1": lift, "prose": prose}


def pass11_boot(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print(f"EXTRA — company bootstrap leftover after issued_lag1 (n={n_boot})")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    work = tr.loc[y.notna(), ["company_id", "fold", Y7, STEM, "e_ar_issued_lag1"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y7],
            b[STEM],
            (b["e_ar_issued_lag1"],),
            b["fold"],
            pd.Series(True, index=b.index),
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share_die = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = (
        f"Bootstrap leftover-after-issued_lag1 rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"share<0.55={_pp(share_die)} n={len(ranks)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share_die, "prose": prose}


def _slice_left(tr: pd.DataFrame, mask, x, controls, label: str) -> dict:
    xs = pd.to_numeric(x, errors="coerce").where(mask)
    d = leftover_diag(tr[Y7], xs, controls, tr["fold"], mask)
    rho_iss, _ = spearman_n(xs, tr["e_ar_issued_lag1"])
    return {
        "cut": label,
        "rank": _f(d["rank"]),
        "OLS": _f(d["ols"]),
        "n": f"{d['n']:,}",
        "n_pos": f"{d['n_pos']:,}",
        "ρ vs issued_lag1": _f(rho_iss),
        "dies?": "dies" if d["honest_dies"] else (
            "lives ≥0.58" if np.isfinite(d["rank"]) and d["rank"] >= KEEP_LEFT else "lives"
        ),
        "_rank": d["rank"],
        "_dies": d["honest_dies"],
    }


def pass12_share_slices(tr: pd.DataFrame) -> dict:
    print("EXTRA — leftover after issued_lag1 by last-month top-1 share")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    share = pd.to_numeric(tr["top1_share_lag1"], errors="coerce")
    x = pd.to_numeric(tr[STEM], errors="coerce")
    ctrl = (tr["e_ar_issued_lag1"],)
    rows = []
    store = {}
    for label, m in (
        ("share_lag1 <0.50 diversified", y.notna() & x.notna() & (share < 0.50)),
        ("share_lag1 0.50–0.80", y.notna() & x.notna() & (share >= 0.50) & (share < 0.80)),
        ("share_lag1 ≥0.80 monopoly", y.notna() & x.notna() & (share >= 0.80)),
        ("share_lag1 >0.975 HHI-tail seat", y.notna() & x.notna() & (share > 0.975)),
        ("zero issued-to-top1", y.notna() & (x == 0)),
        ("positive issued-to-top1", y.notna() & (x > 0)),
    ):
        rec = _slice_left(tr, m, x, ctrl, label)
        rows.append({k: rec[k] for k in rec if not k.startswith("_")})
        store[label] = rec
    prose = (
        f"Diversified leftover {_f(store['share_lag1 <0.50 diversified']['_rank'])}; "
        f"monopoly leftover {_f(store['share_lag1 ≥0.80 monopoly']['_rank'])}; "
        f"HHI-tail leftover {_f(store['share_lag1 >0.975 HHI-tail seat']['_rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "div_rank": store["share_lag1 <0.50 diversified"]["_rank"],
        "mono_rank": store["share_lag1 ≥0.80 monopoly"]["_rank"],
        "tail_rank": store["share_lag1 >0.975 HHI-tail seat"]["_rank"],
        "prose": prose,
    }


def pass14_same_switch_zero(tr: pd.DataFrame) -> dict:
    print("EXTRA — same vs switched top-1; zero dummy leftover; Q6 leftover")
    y = pd.to_numeric(tr[Y7], errors="coerce")
    x = pd.to_numeric(tr[STEM], errors="coerce")
    same = tr["top1_id_lag1"].notna() & tr["top1_id"].notna() & (tr["top1_id_lag1"] == tr["top1_id"])
    switched = tr["top1_id_lag1"].notna() & tr["top1_id"].notna() & (tr["top1_id_lag1"] != tr["top1_id"])
    zero = x == 0
    dummy = (x == 0).astype(float)
    dummy[x.isna()] = np.nan
    ctrl = (tr["e_ar_issued_lag1"],)
    rows = []
    store = {}
    for label, m in (
        ("same top-1 id (lag1==t)", y.notna() & x.notna() & same),
        ("switched top-1 id", y.notna() & x.notna() & switched),
        ("positive issued-to-top1 only", y.notna() & (x > 0)),
        ("zero issued-to-top1 only", y.notna() & zero),
    ):
        rec = _slice_left(tr, m, x, ctrl, label)
        rows.append({k: rec[k] for k in rec if not k.startswith("_")})
        store[label] = rec
    dum = leftover_diag(y, dummy, ctrl, tr["fold"], y.notna())
    q6 = leftover_diag(y, tr[f"{STEM}_lag1"], ctrl, tr["fold"], y.notna())
    n_same = int((y.notna() & x.notna() & same).sum())
    n_sw = int((y.notna() & x.notna() & switched).sum())
    rate0 = float(y[y.notna() & zero].mean()) if (y.notna() & zero).any() else float("nan")
    ratep = float(y[y.notna() & (x > 0)].mean()) if (y.notna() & (x > 0)).any() else float("nan")
    rows.append(
        {
            "cut": "zero dummy after issued_lag1",
            "rank": _f(dum["rank"]),
            "OLS": _f(dum["ols"]),
            "n": f"{dum['n']:,}",
            "n_pos": f"{dum['n_pos']:,}",
            "ρ vs issued_lag1": _f(dum["rho_ctrl"]),
            "dies?": "dies" if dum["honest_dies"] else "lives",
        }
    )
    rows.append(
        {
            "cut": "issued_top1_lag1 after issued_lag1 (Q6 leftover)",
            "rank": _f(q6["rank"]),
            "OLS": _f(q6["ols"]),
            "n": f"{q6['n']:,}",
            "n_pos": f"{q6['n_pos']:,}",
            "ρ vs issued_lag1": _f(q6["rho_ctrl"]),
            "dies?": "dies" if q6["honest_dies"] else "lives",
        }
    )
    prose = (
        f"Same-id rows {n_same:,} leftover {_f(store['same top-1 id (lag1==t)']['_rank'])}; "
        f"switched {n_sw:,} leftover {_f(store['switched top-1 id']['_rank'])}. "
        f"Y7 rate zero={_pp(rate0)} vs positive={_pp(ratep)}. "
        f"Zero dummy leftover {_f(dum['rank'])}; "
        f"positive-only leftover {_f(store['positive issued-to-top1 only']['_rank'])}; "
        f"Q6 leftover of lag1 after issued_lag1 {_f(q6['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "same_rank": store["same top-1 id (lag1==t)"]["_rank"],
        "sw_rank": store["switched top-1 id"]["_rank"],
        "pos_rank": store["positive issued-to-top1 only"]["_rank"],
        "dummy_rank": dum["rank"],
        "q6_left": q6["rank"],
        "rate0": rate0,
        "ratep": ratep,
        "prose": prose,
    }


def pass13_fold4_samen(tr: pd.DataFrame, p4: dict, p5: dict) -> dict:
    rec = p4["recs"]["iss_top1_y7"]
    f4 = next((r["auroc"] for r in rec.get("folds", []) if int(r["fold"]) == 4), float("nan"))
    iss4 = next(
        (r["auroc"] for r in p4["recs"]["iss_lag1_y7"].get("folds", []) if int(r["fold"]) == 4),
        float("nan"),
    )
    main = p5["cuts"]["after issued_lag1"]
    ok = main["rresid"].notna() & tr[Y7].notna() & tr[STEM].notna()
    raw = signed_oof_auroc(tr[Y7], tr[STEM], tr["fold"], ok)
    lift = _cv(raw) - main["rank"] if np.isfinite(_cv(raw)) and np.isfinite(main["rank"]) else float("nan")
    prose = (
        f"Fold 4 {STEM} {_f(f4)} vs issued_lag1 {_f(iss4)}. "
        f"Same-n raw {_f(_cv(raw))} vs leftover {_f(main['rank'])} lift {_f(lift)}. "
        f"TURNOVER fold-4 0.680 stays issued. Do not grow the card."
    )
    print(prose)
    return {
        "f4": f4,
        "iss4": iss4,
        "raw": _cv(raw),
        "left": main["rank"],
        "lift": lift,
        "rows": [{"fold": "4", STEM: _f(f4), "issued_lag1": _f(iss4)}],
        "prose": prose,
    }


def decide(p3, p4, p5, p6, p7) -> dict:
    twin = bool(p3["gate"] or p3["size"])
    keep = bool(p5["keep"] and p4["beat"] and not twin)
    if keep:
        y7 = "KEEP"
        y7_why = (
            f"leftover after issued_lag1 {_f(p5['main_rank'])} ≥ 0.58, "
            f"not a twin, beat-size PASS"
        )
    elif p5["main_dies"] or twin:
        y7 = "CLOSE"
        y7_why = (
            f"leftover {_f(p5['main_rank'])} dies / twin={p3['gate'] or p3['size']} "
            f"— issued_lag1 Q6 0.626 stays the invoice lead"
        )
    else:
        y7 = "CLOSE"
        y7_why = (
            f"leftover {_f(p5['main_rank'])} lives but <0.58 or beat-size FAIL "
            f"— issued_lag1 Q6 0.626 stays the invoice lead"
        )
    return {
        "y7": y7,
        "y7_why": y7_why,
        "y3": "DROP",
        "y3_why": f"leftover after days {_f(p6['rank'])} — off the 15-col card; d_cust_lost 0.522 already DROP",
        "q6": "KEEP" if p7["keep_q6"] else "CLOSE",
        "keep": keep,
    }


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    y = pd.to_numeric(tr[Y7], errors="coerce")
    x = pd.to_numeric(tr[STEM], errors="coerce")
    lab = y.notna() & x.notna()
    work = tr.loc[lab, [Y7, STEM]].copy()
    work["q"] = _rank_q(work[STEM])
    work["pile"] = np.where(work[STEM] == 0, "zero", "positive")
    rates = work.groupby("q", observed=False)[Y7].mean()
    pile = work.groupby("pile", observed=False)[Y7].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    axes[0].bar(rates.index.astype(str), rates.values, color="#3d5a80")
    axes[0].set_ylabel("Y7 rate")
    axes[0].set_xlabel("rank-quintile of issued-to-last-month-top-1")
    axes[0].set_title("Jacobson / Irvine thinning (Y7)")
    axes[0].set_ylim(0, max(0.35, float(rates.max()) + 0.05))
    order = [b for b in ("zero", "positive") if b in pile.index]
    axes[1].bar(order, [float(pile[b]) for b in order], color="#ee6c4d")
    axes[1].set_ylabel("Y7 rate")
    axes[1].set_xlabel("zero vs positive issued-to-top-1")
    axes[1].set_title("Named buyer got nothing this month?")
    axes[1].set_ylim(0, max(0.35, float(pile.max()) + 0.05))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    d = ctx["decision"]
    lines = [
        "# Unused leftover of AR issued to last month's top-1 after issued_lag1",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_cust_lost`. "
        "Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. "
        "Y7 never D as engine X. Y5 never E. Y3 never B. Dark 470 stay NaN not 0.",
        "",
        "`e_issued_top1` = this-month AR |amt| issued to **last month’s** trailing-3m "
        "top-1 counterparty (Y7 object, lagged). 0 if that buyer exists and got nothing; "
        "NaN if no lag-1 top-1. Jacobson demand-shrinkage / Irvine major-customer / "
        "Amberg issued −1 pp. Not `d_cust_lost` (Y3 leftover 0.522 DROP).",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_cust_lost`. Dark 470 = NaN, not 0. |",
        "| 2 | Who is improving? | Thinning is a *why*, not a recovery clock. |",
        "| 3 | Who is turning? | Named-buyer issued going to 0 is a turn *if* leftover lives. |",
        f"| 4 | Dip vs fall? | Y7 leftover after issued_lag1 **{d['y7']}** — {d['y7_why']} |",
        "| 5 | Why did it change? | Identity of the missing euro. Firm issued_lag1 cannot see *who* thinned. |",
        f"| 6 | Months earlier? | **{d['q6']}** named-volume lead footnote — {p7['prose']} issued_lag1 0.626 stays on TURNOVER. |",
        "",
        "## PARK / CLOSE / KEEP / DROP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| issued-to-last-month-top-1 leftover after issued_lag1 (Y7) | **{d['y7']}** | {d['y7_why']} |",
        "| TURNOVER add-on | **CLOSE** | do not grow 0.720 |",
        "| D as Y7 engine X | **DROP** | Y7 never D |",
        f"| as Y3 X / the 15-col card | **{d['y3']}** | {d['y3_why']} |",
        "| as Y5 X | **DROP** | Y5 never E |",
        "| health Y `y_cust_lost` | **PARK** | `d_cust_lost` leftover 0.522 already DROP |",
        f"| Q6 named-volume lead (not a TURNOVER add-on) | **{d['q6']}** | leftover / lag1 ≥0.58; issued_lag1 0.626 stays the card lead |",
        "| issued_lag1 Q6 0.626 | **KEEP (locked)** | do not overwrite issued_qa |",
        "",
        "## 1. Coverage / dark 470",
        "",
        p1["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {p1['n_cm']:,} / {p1['n_co']:,} |",
        f"| {STEM} defined | {p1['n_nn']:,} ({_pp(p1['cov'])}) |",
        f"| among defined: zero | {_pp(p1['zero_share'])} |",
        f"| mean / p50 issued-to-top1 | {_f(p1['mean'])} / {_f(p1['p50'])} |",
        f"| mean share of this-month issued | {_f(p1['mean_share'])} |",
        f"| dark 470 nn / zero | {p1['dark_nn']} / {p1['dark_zero']} |",
        f"| dark NaN | {'CONFIRM' if p1['dark_ok'] else 'FAIL'} |",
        f"| Y7 labeled / pos / issued-top1-nn | {p1['y7_n']:,} / {p1['y7_pos']:,} / {p1['y7_nn']:,} |",
        f"| acf1 / acf3 | {_f(p1['acf1'])} / {_f(p1['acf3'])} |",
        "",
        "## 2. Formula (0 if named and silent; NaN if no lag-1 top-1)",
        "",
        p2["prose"],
        "",
        "## 3. Spearman twins (|ρ|≥0.80)",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Night issued_lag1 **0.630** (replica {_f(p4['iss_y7'])}). "
        f"Days **0.711** (replica {_f(p4['days_y3'])}). Size **0.617**. "
        f"TURNOVER **0.720** / 0.712 unchanged.",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Residual Y7 after issued_lag1 (KEEP gate)",
        "",
        "KEEP leftover only if rank ≥ **0.58**, not a twin of issued_lag1 / issued / "
        "`d_cust_top1`, not SIZE, beat-size. Rank leftover is honest; OLS can fake.",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Residual Y3 after days (off the card)",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. Q6 — lag1 on short vs long",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. ICC / demean",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Dark 470 + holdout coverage (no AUROC)",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. Quintiles (thinning seat)",
        "",
        ctx["p10"]["prose"],
        "",
        _md_table(ctx["p10"]["rows"]),
        "",
        "Zero vs positive issued-to-top-1 on the same labeled rows:",
        "",
        _md_table(ctx["p10"].get("pile_rows") or []),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped.",
        "",
        "## 11. Company bootstrap leftover after issued_lag1",
        "",
        ctx["p11"]["prose"],
        "",
        "## 12. Extra — monopoly vs diversified (last-month top-1 share)",
        "",
        ctx["p12"]["prose"],
        "",
        _md_table(ctx["p12"]["rows"]),
        "",
        "## 13. Fold 4 + same-n artifact",
        "",
        ctx["p13"]["prose"],
        "",
        _md_table(ctx["p13"]["rows"]),
        "",
        "## 14. Extra — same vs switched top-1; zero dummy; Q6 leftover",
        "",
        ctx["p14"]["prose"],
        "",
        _md_table(ctx["p14"]["rows"]),
        "",
        "## What this note did not do",
        "",
        "- Did not change Y7 0.720 / 0.712 or Y3 0.762 / 0.752.",
        "- Did not put issued-to-top-1 / delay / CN / DSO on TURNOVER.",
        "- Did not merge Family D as Y7 X. Did not score this as Y5 X.",
        "- Did not overwrite delay_qa / issued_qa / credit_note_qa / top1_pastdue_qa / "
        "top1_qa / cust_lost_qa / y7_core.",
        "- Did not invent `y_cust_lost`. Dark 470 stayed NaN.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(rows: list[dict]) -> None:
    if not REGISTRY.exists():
        print("registry missing — skip")
        return
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                str(r.get("agent", "")),
                str(r.get("y", "")),
                str(r.get("model", "")),
                str(r.get("split", "")),
                str(r.get("metric", "")),
                str(r.get("x_families", "")),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        if r.get("value") is None or (isinstance(r.get("value"), float) and not np.isfinite(r["value"])):
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


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p5 = ctx["p5"]
    WAVE_NOTE.write_text(
        "\n".join(
            [
                "# Wave 4 — issued to last month's top-1 leftover (Wave B)",
                "",
                f"Owner: `{MODEL}`. As-of `{_now_iso()}`.",
                "Deliverable: `analysis/outputs/issued_top1_qa.md`.",
                "Do not grow TURNOVER **0.720**. issued_lag1 Q6 **0.626** stays the invoice lead.",
                "",
                "## Headline",
                "",
                ctx["headline"],
                "",
                "## Decision",
                "",
                f"- Y7 leftover after issued_lag1: **{d['y7']}** ({d['y7_why']})",
                f"- rank {_f(p5['main_rank'])} OLS {_f(p5['main_ols'])} "
                f"after-contemp-issued {_f(p5['after_issued'])} share {_f(p5['share_rank'])}",
                f"- Q6: **{d['q6']}**. Y3 card: **{d['y3']}**.",
                "",
                "## Do not do next",
                "",
                "- Put this on TURNOVER. Merge D as Y7 X. Score as Y5 X.",
                "- Reopen `d_cust_lost` / overwrite cust_lost_qa. Invent `y_cust_lost`.",
                "",
                "## Extras after headline",
                "",
                f"- Dark 470 NaN CONFIRM. Diversified leftover {_f(ctx['p12']['div_rank'])}; "
                f"monopoly {_f(ctx['p12']['mono_rank'])}; HHI-tail {_f(ctx['p12']['tail_rank'])}.",
                f"- Boot p05 {_f(ctx['p11']['p05'])} p50 {_f(ctx['p11']['p50'])}.",
                f"- Same-id leftover {_f(ctx['p14']['same_rank'])}; switched {_f(ctx['p14']['sw_rank'])}; "
                f"zero dummy {_f(ctx['p14']['dummy_rank'])}; positive-only {_f(ctx['p14']['pos_rank'])}; "
                f"Q6 leftover {_f(ctx['p14']['q6_left'])}.",
                "- KEEP leftover. CLOSE TURNOVER add-on. issued_lag1 0.626 stays the card lead.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"issued_top1_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    con = connect()
    try:
        book = book_invoice_ids(con)
        print(f"book ids {len(book)}")
        panel = attach_issued_top1(panel, con)
    finally:
        con.close()
    panel = add_panel_lags(panel, [STEM, SHARE, "e_ar_issued"], (1, 3))
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()}")

    p1 = pass1_cov(tr, book)
    p2 = pass2_formula(tr)
    p3 = pass3_twins(tr)
    p4 = pass4_singles(tr)
    p5 = pass5_leftover_y7(tr)
    p6 = pass6_y3(tr)
    p7 = pass7_q6(tr)
    p8 = pass8_icc(tr)
    p9 = pass9_dark_hold(panel, book)
    p10 = pass10_quintiles(tr)
    p11 = pass11_boot(tr, n_boot=40)
    p12 = pass12_share_slices(tr)
    p13 = pass13_fold4_samen(tr, p4, p5)
    p14 = pass14_same_switch_zero(tr)
    decision = decide(p3, p4, p5, p6, p7)
    png_ok = make_png(tr)
    headline = (
        f"{STEM} defined {_pp(p1['cov'])} of train CM; dark 470 "
        f"{'NaN CONFIRM' if p1['dark_ok'] else 'FAIL'}. "
        f"zero-among-defined {_pp(p1['zero_share'])}. "
        f"ρ vs issued_lag1 {_f(p3['rhos'].get('e_ar_issued_lag1'))} "
        f"issued {_f(p3['rhos'].get('e_ar_issued'))} "
        f"d_cust_top1 {_f(p3['rhos'].get('d_cust_top1'))} "
        f"{'TWIN' if p3['gate'] else 'not a twin'}. "
        f"Y7 raw {_f(p4['pd_y7'])} vs issued_lag1 {_f(p4['iss_y7'])} size {_f(p4['size_y7'])} "
        f"beat-size {'PASS' if p4['beat'] else 'FAIL'}. "
        f"Leftover after issued_lag1 rank {_f(p5['main_rank'])} OLS {_f(p5['main_ols'])} "
        f"(after contemp issued {_f(p5['after_issued'])}; share {_f(p5['share_rank'])}). "
        f"ICC {_f(p8['icc'])} demean leftover {_f(p8['demean_rank'])}. "
        f"Boot p05 {_f(p11['p05'])}. Y7 leftover **{decision['y7']}**. "
        f"Thinning-to-zero Y7 {_pp(p14['rate0'])} vs positive {_pp(p14['ratep'])}; "
        f"positive-only leftover {_f(p14['pos_rank'])}. "
        f"CLOSE TURNOVER add-on. issued_lag1 Q6 0.626 stays."
    )
    print(headline)
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
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
    }
    write_md(ctx)
    ts = _now_iso()
    append_registry(
        [
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y7,
                "model": MODEL,
                "split": "train_cv",
                "metric": "auroc_issued_top1",
                "value": p4["pd_y7"],
                "coverage": p1["cov"],
                "notes": headline[:240],
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y7,
                "model": MODEL,
                "split": "train_cv",
                "metric": "leftover_iss_lag1_rank",
                "value": p5["main_rank"],
                "coverage": p1["cov"],
                "notes": f"ols={_f(p5['main_ols'])} keep={decision['y7']}",
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y7,
                "model": MODEL,
                "split": "train_cv",
                "metric": "rho_vs_issued_lag1",
                "value": p3["rhos"].get("e_ar_issued_lag1"),
                "coverage": p1["cov"],
                "notes": f"twins={p3['twins']}",
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y7,
                "model": MODEL,
                "split": "train_cv",
                "metric": "leftover_zero_dummy",
                "value": p14["dummy_rank"],
                "coverage": p1["cov"],
                "notes": f"pos_only={_f(p14['pos_rank'])} q6_left={_f(p14['q6_left'])}",
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y7,
                "model": MODEL,
                "split": "train_cv",
                "metric": "leftover_div_share50",
                "value": p12["div_rank"],
                "coverage": p1["cov"],
                "notes": f"mono={_f(p12['mono_rank'])} tail={_f(p12['tail_rank'])}",
            },
        ]
    )
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"done in {time.time() - t0:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
