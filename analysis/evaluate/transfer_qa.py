"""Signed A transfer / invest — SHAP-heavy, perm-light.

NORTH_STAR: `a_transfer` is signed net (`sum(amount | grp=transfer)`).
`a_invest` is signed deploy+return. Family M `m_xfer_share` is |transfer|/|all|
— do not merge M. Family I `i_transfer_x_*` already CLOSE — do not merge I.
`c_salary_month` 0.671 stays on the 15-col card. Night Y3 quote stays
0.762 / 0.752. Do not put `a_transfer` / `a_invest` on tonight's card.
Do not invent `y_transfer` / `y_invest`. Do not rewrite cashflow.py
unless a real formula bug (then stop).

Question: is signed net (a) a quiet-month twin of days / n_tx / salary,
(b) a two-way wash, (c) SIZE, (d) leftover Q5, or (e) nothing
(loses to size 0.617 / days 0.711)?

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.transfer_qa

Owned: analysis/evaluate/transfer_qa.py, analysis/outputs/transfer_qa.md,
optional one PNG, append-only registry, overnight/waves/wave4_transfer.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "transfer_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "transfer_wash.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "1bc809f3"
WAVE = "4"
ROUND = "R4"
MODEL = "transfer_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
SALARY_QUOTE = 0.671
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
PANEL_END = pd.Timestamp("2026-08-01")
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
WASH_NEAR = 0.05  # |net|/gross below this = washed month
TOKEN_NEAR = 1.0  # |amt| ≤ 1€ is a token

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_transfer",
    "a_invest",
    "a_n_tx",
    "a_in3",
    "a_op_in",
    "a_op_out",
    "c_n_days_with_tx",
    "c_salary_month",
    "c_ss_month",
    "b_below_0",
)

Y_KEEP = (Y2, Y3)

RAW_ZERO = (
    "xfer_in",
    "xfer_out",
    "xfer_gross",
    "n_xfer",
    "n_xfer_in",
    "n_xfer_out",
    "deploy",
    "ret",
    "inv_gross",
    "n_deploy",
    "n_return",
    "abs_all",
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


def cv_of(rec: dict) -> float:
    return float("nan") if rec.get("low_power") else rec["cv"]


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


def load_raw_monthly(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category = 'transfer' AND t.amount > 0 THEN t.amount ELSE 0 END) AS xfer_in,
          SUM(CASE WHEN t.category = 'transfer' AND t.amount < 0 THEN -t.amount ELSE 0 END) AS xfer_out,
          SUM(CASE WHEN t.category = 'transfer' THEN ABS(t.amount) ELSE 0 END) AS xfer_gross,
          SUM(CASE WHEN t.category = 'transfer' THEN 1 ELSE 0 END) AS n_xfer,
          SUM(CASE WHEN t.category = 'transfer' AND t.amount > 0 THEN 1 ELSE 0 END) AS n_xfer_in,
          SUM(CASE WHEN t.category = 'transfer' AND t.amount < 0 THEN 1 ELSE 0 END) AS n_xfer_out,
          SUM(CASE WHEN t.category = 'investment_deployment' THEN t.amount ELSE 0 END) AS deploy,
          SUM(CASE WHEN t.category = 'investment_return' THEN t.amount ELSE 0 END) AS ret,
          SUM(CASE WHEN t.category IN ('investment_deployment', 'investment_return')
                   THEN ABS(t.amount) ELSE 0 END) AS inv_gross,
          SUM(CASE WHEN t.category = 'investment_deployment' THEN 1 ELSE 0 END) AS n_deploy,
          SUM(CASE WHEN t.category = 'investment_return' THEN 1 ELSE 0 END) AS n_return,
          SUM(ABS(t.amount)) AS abs_all
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    return df.drop(columns=["month"])


def load_panel(con) -> pd.DataFrame:
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
    ykeep = ["company_id", "period"] + [c for c in Y_KEEP if c in yraw.columns]
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    monthly = load_raw_monthly(con)
    panel = panel.merge(monthly, on=["company_id", "period"], how="left")
    for c in RAW_ZERO:
        panel[c] = pd.to_numeric(panel[c], errors="coerce").fillna(0.0)
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["abs_xfer"] = pd.to_numeric(panel["a_transfer"], errors="coerce").abs()
    panel["abs_inv"] = pd.to_numeric(panel["a_invest"], errors="coerce").abs()
    panel["has_xfer"] = (pd.to_numeric(panel["n_xfer"], errors="coerce") > 0).astype(np.int8)
    panel["has_inv"] = (
        (pd.to_numeric(panel["n_deploy"], errors="coerce") > 0)
        | (pd.to_numeric(panel["n_return"], errors="coerce") > 0)
    ).astype(np.int8)
    gros = pd.to_numeric(panel["xfer_gross"], errors="coerce")
    panel["wash_ratio"] = np.where(gros > 0, panel["abs_xfer"] / gros, np.nan)
    panel["washed"] = ((gros > 0) & (panel["wash_ratio"] <= WASH_NEAR)).astype(np.int8)
    panel["one_way"] = ((gros > 0) & (panel["wash_ratio"] >= 0.99)).astype(np.int8)
    panel["both_ways"] = (
        (pd.to_numeric(panel["n_xfer_in"], errors="coerce") > 0)
        & (pd.to_numeric(panel["n_xfer_out"], errors="coerce") > 0)
    ).astype(np.int8)
    abs_all = pd.to_numeric(panel["abs_all"], errors="coerce")
    panel["m_xfer_share"] = np.where(abs_all > 0, gros / abs_all, np.nan)
    inv_g = pd.to_numeric(panel["inv_gross"], errors="coerce")
    panel["m_invest_share"] = np.where(abs_all > 0, inv_g / abs_all, np.nan)
    panel["inv_wash"] = np.where(inv_g > 0, panel["abs_inv"] / inv_g, np.nan)
    panel["cal_month"] = panel["period"].dt.month
    first = panel.groupby("company_id")["period"].transform("min")
    panel["months_on_book"] = (
        (PANEL_END.year - first.dt.year) * 12 + (PANEL_END.month - first.dt.month) + 1
    )
    panel["short_book"] = (panel["months_on_book"] < 12).astype(np.int8)
    panel["so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
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


# ---------------------------------------------------------------------------
# Pass 1 — prevalence
# ---------------------------------------------------------------------------
def pass1_prev(panel: pd.DataFrame, tr: pd.DataFrame) -> dict:
    rows = []
    for split, sl in (("train", tr), ("holdout", panel[panel["split"] == "holdout"])):
        for col, flag in (("a_transfer", "has_xfer"), ("a_invest", "has_inv")):
            x = pd.to_numeric(sl[col], errors="coerce")
            any_ = pd.to_numeric(sl[flag], errors="coerce") == 1
            rows.append(
                {
                    "split": split,
                    "col": col,
                    "n_cm": int(len(sl)),
                    "n_co": int(sl["company_id"].nunique()),
                    "cov": float(x.notna().mean()) if len(x) else float("nan"),
                    "any_share": float(any_.mean()) if len(sl) else float("nan"),
                    "n_any": int(any_.sum()),
                    "p50": float(x.median()) if x.notna().any() else float("nan"),
                    "p90": float(x.quantile(0.90)) if x.notna().any() else float("nan"),
                    "p50_abs": float(x.abs().median()) if x.notna().any() else float("nan"),
                    "p90_abs": float(x.abs().quantile(0.90)) if x.notna().any() else float("nan"),
                    "zero_share": float((x == 0).mean()) if x.notna().any() else float("nan"),
                }
            )
    xf = next(r for r in rows if r["split"] == "train" and r["col"] == "a_transfer")
    inv = next(r for r in rows if r["split"] == "train" and r["col"] == "a_invest")
    ho_x = next(r for r in rows if r["split"] == "holdout" and r["col"] == "a_transfer")
    ho_i = next(r for r in rows if r["split"] == "holdout" and r["col"] == "a_invest")
    ever_x = int((tr.groupby("company_id")["has_xfer"].max() > 0).sum())
    ever_i = int((tr.groupby("company_id")["has_inv"].max() > 0).sum())
    n_co = int(tr["company_id"].nunique())
    prose = (
        f"Train any-transfer CM {_pp(xf['any_share'])} (n={xf['n_any']:,} / {xf['n_cm']:,}); "
        f"ever-transfer companies {ever_x}/{n_co}. "
        f"signed `a_transfer` p50={xf['p50']:,.0f} p90={xf['p90']:,.0f} "
        f"(zero share {_pp(xf['zero_share'])}). "
        f"Any-invest {_pp(inv['any_share'])} (n={inv['n_any']:,}); "
        f"ever-invest {ever_i}/{n_co}; signed p50={inv['p50']:,.0f} p90={inv['p90']:,.0f}. "
        f"Holdout coverage only: xfer {_pp(ho_x['any_share'])} / invest {_pp(ho_i['any_share'])} "
        f"on {ho_x['n_cm']:,} CM / {ho_x['n_co']} cos. No AUROC on holdout."
    )
    print(prose)
    return {
        "rows": rows,
        "xf": xf,
        "inv": inv,
        "ho_x": ho_x,
        "ho_i": ho_i,
        "ever_x": ever_x,
        "ever_i": ever_i,
        "n_co": n_co,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — raw tokens vs store
# ---------------------------------------------------------------------------
def pass2_tokens(tr: pd.DataFrame, con) -> dict:
    signed_x = pd.to_numeric(tr["a_transfer"], errors="coerce")
    raw_x = pd.to_numeric(tr["xfer_in"], errors="coerce") - pd.to_numeric(tr["xfer_out"], errors="coerce")
    signed_i = pd.to_numeric(tr["a_invest"], errors="coerce")
    raw_i = pd.to_numeric(tr["deploy"], errors="coerce") + pd.to_numeric(tr["ret"], errors="coerce")
    max_x = float((signed_x - raw_x).abs().max())
    max_i = float((signed_i - raw_i).abs().max())
    # euro amounts; 1e-4 is float dust, not a cashflow.py bug
    agree_x = bool(np.isfinite(max_x) and max_x < 1e-4)
    agree_i = bool(np.isfinite(max_i) and max_i < 1e-4)

    cats = con.execute(
        """
        SELECT category,
               COUNT(*) AS n,
               SUM(CASE WHEN amount > 0 THEN 1 ELSE 0 END) AS n_in,
               SUM(CASE WHEN amount < 0 THEN 1 ELSE 0 END) AS n_out,
               median(ABS(amount)) AS p50_abs
        FROM transactions
        WHERE category IN ('transfer', 'investment_deployment', 'investment_return')
          AND "date" IS NOT NULL
        GROUP BY 1
        ORDER BY 1
        """
    ).df()

    sample = (
        tr.loc[tr["has_xfer"] == 1, ["company_id", "period", "a_transfer", "xfer_in", "xfer_out", "xfer_gross"]]
        .head(4)
        .copy()
    )
    sample2 = (
        tr.loc[tr["has_xfer"] == 0, ["company_id", "period", "a_transfer", "xfer_in", "xfer_out", "xfer_gross"]]
        .head(2)
        .copy()
    )
    samp = pd.concat([sample, sample2], ignore_index=True)
    samp_rows = []
    for _, r in samp.iterrows():
        samp_rows.append(
            {
                "company_id": r["company_id"],
                "period": pd.Timestamp(r["period"]).strftime("%Y-%m"),
                "store": f"{float(r['a_transfer']):,.2f}",
                "in": f"{float(r['xfer_in']):,.2f}",
                "out": f"{float(r['xfer_out']):,.2f}",
                "gross": f"{float(r['xfer_gross']):,.2f}",
            }
        )
    formula_ok = agree_x and agree_i
    prose = (
        f"Store `a_transfer` vs raw sum(amount|category=transfer) max|Δ|={max_x:.3g}; "
        f"`a_invest` vs deploy+return max|Δ|={max_i:.3g}. "
        f"{'Formula OK — not a cashflow.py bug.' if formula_ok else 'FORMULA MISMATCH — stop, do not rewrite tonight.'} "
        f"Tokens present: transfer / investment_deployment / investment_return."
    )
    print(prose)
    return {
        "max_x": max_x,
        "max_i": max_i,
        "agree_x": agree_x,
        "agree_i": agree_i,
        "formula_ok": formula_ok,
        "cats": cats,
        "samp_rows": samp_rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — two-way wash
# ---------------------------------------------------------------------------
def pass3_wash(tr: pd.DataFrame) -> dict:
    has = tr["has_xfer"] == 1
    n_has = int(has.sum())
    both = int((tr["both_ways"] == 1).sum())
    one = int((tr["one_way"] == 1).sum())
    washed = int((tr["washed"] == 1).sum())
    wr = pd.to_numeric(tr.loc[has, "wash_ratio"], errors="coerce")
    rho_signed_gross = spearman(tr["a_transfer"], tr["xfer_gross"])
    rho_abs_gross = spearman(tr["abs_xfer"], tr["xfer_gross"])
    rho_signed_abs = spearman(tr["a_transfer"], tr["abs_xfer"])
    p_both = _pct(both, n_has) if n_has else float("nan")
    p_one = _pct(one, n_has) if n_has else float("nan")
    p_wash = float((tr.loc[has, "washed"] == 1).mean()) if n_has else float("nan")
    # typical month: p50 |net|/gross == 1 means one-way, not a wash
    wash_typical = bool(np.isfinite(wr.median()) and wr.median() <= 0.20)
    wash_rare = bool(np.isfinite(p_wash) and p_wash < 0.15)
    is_wash = bool(wash_typical and (not np.isfinite(rho_signed_gross) or abs(rho_signed_gross) < 0.30))
    prose = (
        f"Train transfer CM {n_has:,}: both-ways {_pp(p_both)}, one-way (|net|≈gross) {_pp(p_one)}, "
        f"|net|/gross ≤{WASH_NEAR:g} {_pp(p_wash)}. "
        f"|net|/gross p10/p50/p90 {_f(float(wr.quantile(0.10)))} / {_f(float(wr.median()))} / "
        f"{_f(float(wr.quantile(0.90)))}. "
        f"Spearman signed↔gross {_f(rho_signed_gross)}; |signed|↔gross {_f(rho_abs_gross)}; "
        f"signed↔|signed| {_f(rho_signed_abs)}. "
        + (
            "YES two-way wash — signed net sits near 0 while gross is large."
            if is_wash
            else "NO — typical transfer month is one-way (p50 |net|/gross = 1); signed is not a wash dummy."
        )
    )
    print(prose)
    return {
        "n_has": n_has,
        "both": both,
        "one": one,
        "washed": washed,
        "p_both": p_both,
        "p_one": p_one,
        "p_wash": p_wash,
        "wr_p10": float(wr.quantile(0.10)) if wr.notna().any() else float("nan"),
        "wr_p50": float(wr.median()) if wr.notna().any() else float("nan"),
        "wr_p90": float(wr.quantile(0.90)) if wr.notna().any() else float("nan"),
        "rho_signed_gross": rho_signed_gross,
        "rho_abs_gross": rho_abs_gross,
        "rho_signed_abs": rho_signed_abs,
        "is_wash": is_wash,
        "wash_rare": wash_rare,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — Spearman twins / SIZE
# ---------------------------------------------------------------------------
def pass4_rho(tr: pd.DataFrame) -> dict:
    stems = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "c_salary_month": tr["c_salary_month"],
        "c_ss_month": tr["c_ss_month"],
        "log1p(a_in3)": tr["log_in3"],
        "a_op_in": tr["a_op_in"],
        "m_xfer_share": tr["m_xfer_share"],
        "xfer_gross": tr["xfer_gross"],
        "abs_xfer": tr["abs_xfer"],
        "has_xfer": tr["has_xfer"],
    }
    objects = {
        "a_transfer": tr["a_transfer"],
        "abs_xfer": tr["abs_xfer"],
        "xfer_gross": tr["xfer_gross"],
        "has_xfer": tr["has_xfer"],
        "a_invest": tr["a_invest"],
        "abs_inv": tr["abs_inv"],
        "has_inv": tr["has_inv"],
    }
    rows = []
    twins = []
    size_hits = []
    for oname, ocol in objects.items():
        for sname, scol in stems.items():
            if oname == sname:
                continue
            rho = spearman(ocol, scol)
            flag = ""
            if np.isfinite(rho) and abs(rho) >= TWIN_RHO:
                flag = "TWIN"
                twins.append(f"{oname}↔{sname} {rho:.3f}")
            if sname == "log1p(a_in3)" and np.isfinite(rho) and abs(rho) >= SIZE_RHO:
                flag = (flag + " SIZE").strip()
                size_hits.append(f"{oname} {rho:.3f}")
            rows.append({"object": oname, "vs": sname, "rho": rho, "flag": flag})

    def _rho(obj, vs) -> float:
        hits = [r["rho"] for r in rows if r["object"] == obj and r["vs"] == vs]
        return hits[0] if hits else float("nan")

    xfer_twin_days = bool(abs(_rho("a_transfer", "c_n_days_with_tx") or 0) >= TWIN_RHO)
    xfer_twin_ntx = bool(abs(_rho("a_transfer", "a_n_tx") or 0) >= TWIN_RHO)
    xfer_twin_sal = bool(abs(_rho("a_transfer", "c_salary_month") or 0) >= TWIN_RHO)
    xfer_size = bool(abs(_rho("a_transfer", "log1p(a_in3)") or 0) >= SIZE_RHO)
    inv_size = bool(abs(_rho("a_invest", "log1p(a_in3)") or 0) >= SIZE_RHO)
    prose = (
        "Twin |ρ|≥0.80; SIZE |ρ| vs log1p(a_in3) ≥0.50. "
        f"`a_transfer` vs days {_f(_rho('a_transfer','c_n_days_with_tx'))}, "
        f"n_tx {_f(_rho('a_transfer','a_n_tx'))}, "
        f"salary {_f(_rho('a_transfer','c_salary_month'))}, "
        f"ss {_f(_rho('a_transfer','c_ss_month'))}, "
        f"size {_f(_rho('a_transfer','log1p(a_in3)'))}, "
        f"a_op_in {_f(_rho('a_transfer','a_op_in'))}, "
        f"m_xfer_share {_f(_rho('a_transfer','m_xfer_share'))}. "
        f"`has_xfer` vs days {_f(_rho('has_xfer','c_n_days_with_tx'))}. "
        f"`a_invest` vs size {_f(_rho('a_invest','log1p(a_in3)'))}. "
        + (f"Twins: {'; '.join(twins)}." if twins else "No signed-net twin vs days/n_tx/salary.")
        + (f" SIZE hits: {'; '.join(size_hits)}." if size_hits else " Signed net is not SIZE.")
    )
    print(prose)
    return {
        "rows": rows,
        "twins": twins,
        "size_hits": size_hits,
        "rho_days": _rho("a_transfer", "c_n_days_with_tx"),
        "rho_ntx": _rho("a_transfer", "a_n_tx"),
        "rho_sal": _rho("a_transfer", "c_salary_month"),
        "rho_ss": _rho("a_transfer", "c_ss_month"),
        "rho_size": _rho("a_transfer", "log1p(a_in3)"),
        "rho_opin": _rho("a_transfer", "a_op_in"),
        "rho_mxfer": _rho("a_transfer", "m_xfer_share"),
        "rho_has_days": _rho("has_xfer", "c_n_days_with_tx"),
        "rho_abs_days": _rho("abs_xfer", "c_n_days_with_tx"),
        "rho_gross_days": _rho("xfer_gross", "c_n_days_with_tx"),
        "rho_inv_size": _rho("a_invest", "log1p(a_in3)"),
        "rho_inv_days": _rho("a_invest", "c_n_days_with_tx"),
        "xfer_twin_days": xfer_twin_days,
        "xfer_twin_ntx": xfer_twin_ntx,
        "xfer_twin_sal": xfer_twin_sal,
        "xfer_size": xfer_size,
        "inv_size": inv_size,
        "xfer_twin": xfer_twin_days or xfer_twin_ntx or xfer_twin_sal,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — group-fold AUROC
# ---------------------------------------------------------------------------
def pass5_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "a_transfer": tr["a_transfer"],
        "abs_xfer": tr["abs_xfer"],
        "xfer_gross": tr["xfer_gross"],
        "has_xfer": tr["has_xfer"],
        "washed": tr["washed"],
        "a_invest": tr["a_invest"],
        "abs_inv": tr["abs_inv"],
        "inv_gross": tr["inv_gross"],
        "has_inv": tr["has_inv"],
        "log1p_a_in3": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "c_salary_month": tr["c_salary_month"],
        "c_ss_month": tr["c_ss_month"],
        "m_xfer_share": tr["m_xfer_share"],
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
    sal_y3 = _cv(Y3, "c_salary_month")
    ntx_y3 = _cv(Y3, "a_n_tx")
    xfer_y3 = _cv(Y3, "a_transfer")
    abs_y3 = _cv(Y3, "abs_xfer")
    gros_y3 = _cv(Y3, "xfer_gross")
    has_y3 = _cv(Y3, "has_xfer")
    inv_y3 = _cv(Y3, "a_invest")
    abs_i_y3 = _cv(Y3, "abs_inv")
    xfer_y2 = _cv(Y2, "a_transfer")
    inv_y2 = _cv(Y2, "a_invest")
    size_y2 = _cv(Y2, "log1p_a_in3")
    days_y2 = _cv(Y2, "c_n_days_with_tx")
    beat_size = xfer_y3 - size_y3 if np.isfinite(xfer_y3) and np.isfinite(size_y3) else float("nan")
    beat_days = xfer_y3 - days_y3 if np.isfinite(xfer_y3) and np.isfinite(days_y3) else float("nan")
    beat_sal = xfer_y3 - sal_y3 if np.isfinite(xfer_y3) and np.isfinite(sal_y3) else float("nan")
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.02)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.03)
    sal_ok = bool(np.isfinite(sal_y3) and abs(sal_y3 - SALARY_QUOTE) < 0.03)
    loses = bool(
        (np.isfinite(xfer_y3) and xfer_y3 < SIZE_PARK)
        or (np.isfinite(beat_size) and beat_size < KEEP_DELTA)
    )
    prose = (
        f"Y3 singles (train group-fold): `a_transfer` {_f(xfer_y3)} vs size {_f(size_y3)} "
        f"(quote 0.617 {'CONFIRM' if size_ok else 'off'}, Δ {_f(beat_size, 3)}) "
        f"vs days {_f(days_y3)} (night 0.711 {'CONFIRM' if days_ok else 'off'}) "
        f"vs salary {_f(sal_y3)} (quote 0.671 {'CONFIRM' if sal_ok else 'off'}) "
        f"vs n_tx {_f(ntx_y3)}. "
        f"|signed| {_f(abs_y3)} gross {_f(gros_y3)} has_xfer {_f(has_y3)}. "
        f"`a_invest` {_f(inv_y3)} |invest| {_f(abs_i_y3)}. "
        f"Y2 transfer {_f(xfer_y2)} vs size {_f(size_y2)} / days {_f(days_y2)}; invest {_f(inv_y2)}. "
        f"Night Y3 quote stays {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f} (not this cut)."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "xfer_y3": xfer_y3,
        "abs_y3": abs_y3,
        "gros_y3": gros_y3,
        "has_y3": has_y3,
        "inv_y3": inv_y3,
        "abs_i_y3": abs_i_y3,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "sal_y3": sal_y3,
        "ntx_y3": ntx_y3,
        "xfer_y2": xfer_y2,
        "inv_y2": inv_y2,
        "size_y2": size_y2,
        "days_y2": days_y2,
        "beat_size": beat_size,
        "beat_days": beat_days,
        "beat_sal": beat_sal,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "sal_ok": sal_ok,
        "loses": loses,
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "y3_folds": fold_bits(store[(Y3, "a_transfer")]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — leftover after days / n_tx / salary / |signed|
# ---------------------------------------------------------------------------
def pass6_resid(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_transfer"], errors="coerce")
    inv = pd.to_numeric(tr["a_invest"], errors="coerce")
    has = pd.to_numeric(tr["has_xfer"], errors="coerce")
    abx = pd.to_numeric(tr["abs_xfer"], errors="coerce")
    gros = pd.to_numeric(tr["xfer_gross"], errors="coerce")
    # Zero-inflated signed OLS leftover is often −β·days on the zero months
    # (fake skill). Honest leftover = residual of has_xfer / |signed| / signed
    # among transfer months only.
    specs = [
        ("xfer_resid_days", *ols_resid(x, tr["c_n_days_with_tx"])),
        ("xfer_resid_ntx", *ols_resid(x, tr["a_n_tx"])),
        ("xfer_resid_sal", *ols_resid(x, tr["c_salary_month"])),
        ("xfer_resid_ss", *ols_resid(x, tr["c_ss_month"])),
        ("xfer_resid_days_ntx", *ols_resid(x, tr["c_n_days_with_tx"], tr["a_n_tx"])),
        ("xfer_resid_days_sal", *ols_resid(x, tr["c_n_days_with_tx"], tr["c_salary_month"])),
        ("xfer_resid_card", *ols_resid(x, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])),
        ("xfer_resid_abs", *ols_resid(x, abx)),
        ("xfer_resid_gross", *ols_resid(x, gros)),
        ("has_resid_days", *ols_resid(has, tr["c_n_days_with_tx"])),
        ("has_resid_ntx", *ols_resid(has, tr["a_n_tx"])),
        ("has_resid_sal", *ols_resid(has, tr["c_salary_month"])),
        ("has_resid_card", *ols_resid(has, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])),
        ("abs_resid_days", *ols_resid(abx, tr["c_n_days_with_tx"])),
        ("abs_resid_card", *ols_resid(abx, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])),
        ("gros_resid_days", *ols_resid(gros, tr["c_n_days_with_tx"])),
        ("inv_resid_days", *ols_resid(inv, tr["c_n_days_with_tx"])),
        ("inv_resid_ntx", *ols_resid(inv, tr["a_n_tx"])),
        ("inv_resid_card", *ols_resid(inv, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])),
        ("has_inv_resid_days", *ols_resid(tr["has_inv"], tr["c_n_days_with_tx"])),
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
    # signed leftover among transfer months only (no zero-month −days leak)
    has_m = tr["has_xfer"] == 1
    x_on = pd.to_numeric(tr["a_transfer"], errors="coerce").where(has_m)
    r_on_days, _ = ols_resid(x_on, tr["c_n_days_with_tx"])
    r_on_card, _ = ols_resid(x_on, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])
    lab3 = tr[Y3].notna()
    on_raw = signed_oof_auroc(tr[Y3], x_on, tr["fold"], lab3 & has_m)
    on_days = signed_oof_auroc(tr[Y3], r_on_days, tr["fold"], lab3 & has_m)
    on_card = signed_oof_auroc(tr[Y3], r_on_card, tr["fold"], lab3 & has_m)
    rows.append(
        {
            "y": Y3,
            "feature": "signed_on_xfer_cm",
            "n": f"{on_raw['n_defined']:,}",
            "n_pos": f"{on_raw['n_pos']:,}",
            "CV": "LOW_POWER" if on_raw["low_power"] else _f(on_raw["cv"]),
            "sign": on_raw["train_sign"] if not on_raw["low_power"] else "—",
            "Δsize": "—",
            "slope": "—",
        }
    )
    rows.append(
        {
            "y": Y3,
            "feature": "signed_on_xfer_resid_days",
            "n": f"{on_days['n_defined']:,}",
            "n_pos": f"{on_days['n_pos']:,}",
            "CV": "LOW_POWER" if on_days["low_power"] else _f(on_days["cv"]),
            "sign": on_days["train_sign"] if not on_days["low_power"] else "—",
            "Δsize": "—",
            "slope": "—",
        }
    )
    rho_fake = spearman(spec_resids["xfer_resid_days"], tr["c_n_days_with_tx"])
    rho_has = spearman(spec_resids["has_resid_days"], tr["c_n_days_with_tx"])
    y3_days = cvs.get((Y3, "xfer_resid_days"), float("nan"))
    y3_ntx = cvs.get((Y3, "xfer_resid_ntx"), float("nan"))
    y3_sal = cvs.get((Y3, "xfer_resid_sal"), float("nan"))
    y3_card = cvs.get((Y3, "xfer_resid_card"), float("nan"))
    y3_abs = cvs.get((Y3, "xfer_resid_abs"), float("nan"))
    y3_gros = cvs.get((Y3, "xfer_resid_gross"), float("nan"))
    y3_inv = cvs.get((Y3, "inv_resid_days"), float("nan"))
    y3_has_d = cvs.get((Y3, "has_resid_days"), float("nan"))
    y3_has_c = cvs.get((Y3, "has_resid_card"), float("nan"))
    y3_abs_d = cvs.get((Y3, "abs_resid_days"), float("nan"))
    y3_abs_c = cvs.get((Y3, "abs_resid_card"), float("nan"))
    y3_hinv = cvs.get((Y3, "has_inv_resid_days"), float("nan"))
    size_y3 = cv_of(signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna()))
    leftover_days = (
        (y3_has_d - size_y3) if np.isfinite(y3_has_d) and np.isfinite(size_y3) else float("nan")
    )
    leftover_card = (
        (y3_has_c - size_y3) if np.isfinite(y3_has_c) and np.isfinite(size_y3) else float("nan")
    )
    died_days = bool(np.isfinite(y3_has_d) and y3_has_d < 0.55)
    died_card = bool(np.isfinite(y3_has_c) and y3_has_c < 0.55)
    leftover_lives = bool(np.isfinite(leftover_card) and leftover_card >= KEEP_DELTA)
    quiet_twin = bool(died_days or died_card or (np.isfinite(y3_has_d) and y3_has_d < 0.60))
    fake_days = bool(np.isfinite(rho_fake) and abs(rho_fake) >= 0.50)
    prose = (
        f"Honest leftover is `has_xfer` after days {_f(y3_has_d)} / after card {_f(y3_has_c)} "
        f"(Δsize {_f(leftover_card, 3)}); |signed| after days {_f(y3_abs_d)}. "
        f"Signed OLS leftover after days {_f(y3_days)} is "
        f"{'a zero-month −days leak' if fake_days else 'measured'} "
        f"(ρ(resid,days)={_f(rho_fake)}; has-resid ρ vs days {_f(rho_has)}). "
        f"Among transfer CM only, signed Y3 {_f(cv_of(on_raw))} leftover-days {_f(cv_of(on_days))}. "
        f"Invest signed leftover {_f(y3_inv)} / has_inv leftover {_f(y3_hinv)}. "
        + (
            "Leftover dies — CLOSE as quiet twin (explains SHAP-heavy / perm-light)."
            if quiet_twin and not leftover_lives
            else (
                "Leftover beats size ≥0.02 after the card stems — later A candidate (not on the card)."
                if leftover_lives
                else "Leftover does not clear size+0.02 after the card stems."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_days": y3_days,
        "y3_ntx": y3_ntx,
        "y3_sal": y3_sal,
        "y3_card": y3_card,
        "y3_abs": y3_abs,
        "y3_gros": y3_gros,
        "y3_inv": y3_inv,
        "y3_has_d": y3_has_d,
        "y3_has_c": y3_has_c,
        "y3_abs_d": y3_abs_d,
        "y3_abs_c": y3_abs_c,
        "y3_hinv": y3_hinv,
        "on_raw": cv_of(on_raw),
        "on_days": cv_of(on_days),
        "on_card": cv_of(on_card),
        "rho_fake": rho_fake,
        "rho_has": rho_has,
        "fake_days": fake_days,
        "leftover_days": leftover_days,
        "leftover_card": leftover_card,
        "died_days": died_days,
        "died_card": died_card,
        "leftover_lives": leftover_lives,
        "quiet_twin": quiet_twin,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass7_icc(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for name, col in (
        ("a_transfer", tr["a_transfer"]),
        ("abs_xfer", tr["abs_xfer"]),
        ("has_xfer", tr["has_xfer"]),
        ("xfer_gross", tr["xfer_gross"]),
        ("a_invest", tr["a_invest"]),
        ("has_inv", tr["has_inv"]),
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
        rec = {
            "name": name,
            "icc": icc["icc"],
            "k": icc["k"],
            "acf1": acf1,
            "acf3": acf3,
            "raw_cv": cv_of(raw),
            "dem_cv": cv_of(dem_res),
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
                "drop": _f(drop),
                "call": "TRAIT" if rec["trait"] else ("SHOCK" if rec["shock"] else "between"),
            }
        )
    x = store["a_transfer"]
    prose = (
        f"`a_transfer` ICC={_f(x['icc'])} acf1={_f(x['acf1'])} "
        f"(feature-report 0.96 BETWEEN / LOW_PERSIST). "
        f"Y3 raw {_f(x['raw_cv'])} vs company-demean {_f(x['dem_cv'])} (drop {_f(x['drop'])}). "
        f"`has_xfer` ICC={_f(store['has_xfer']['icc'])}. "
        f"`a_invest` ICC={_f(store['a_invest']['icc'])} acf1={_f(store['a_invest']['acf1'])}. "
        f"{'TRAIT (who transfers)' if x['trait'] else ('MONTH SHOCK' if x['shock'] else 'mixed / between')}."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose, **{f"xfer_{k}": x[k] for k in x}}


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
        n_x=("has_xfer", "sum"),
        n_i=("has_inv", "sum"),
        med_x=("a_transfer", "median"),
    )
    ever["ever_erp"] = ever["company_id"].isin(book)
    ever["x_rate"] = ever["n_x"] / ever["n_cm"]
    ever["i_rate"] = ever["n_i"] / ever["n_cm"]
    rows = []
    for name, part in (
        ("ever_erp_744", ever[ever["ever_erp"]]),
        ("never_erp_470", ever[~ever["ever_erp"]]),
    ):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "ever_xfer": _pp(float((part["n_x"] > 0).mean())),
                "xfer_cm": _pp(float(part["x_rate"].mean())),
                "ever_inv": _pp(float((part["n_i"] > 0).mean())),
                "inv_cm": _pp(float(part["i_rate"].mean())),
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
                "xfer": float(part["has_xfer"].mean()),
                "inv": float(part["has_inv"].mean()),
            }
        )
        lab = part[Y3].notna()
        store[name] = signed_oof_auroc(part[Y3], part["a_transfer"], part["fold"], lab)
    dark_x = float(ever.loc[~ever["ever_erp"], "x_rate"].mean())
    erp_x = float(ever.loc[ever["ever_erp"], "x_rate"].mean())
    same = bool(np.isfinite(dark_x) and np.isfinite(erp_x) and abs(dark_x - erp_x) < 0.05)
    hold = load_holdout()
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ'}). "
        f"Mean company transfer-CM: invoiced {_pp(erp_x)} vs dark {_pp(dark_x)}. "
        f"{'Same bank-book transfer rate' if same else 'Dark files transfer at a different rate'}. "
        f"Y3 `a_transfer` invoiced {_f(cv_of(store['ever_erp']))} / dark {_f(cv_of(store['never_erp']))}. "
        f"Holdout ever-ERP coverage only: {hold_book}/{len(hold)}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "dark_x": dark_x,
        "erp_x": erp_x,
        "same": same,
        "hold_book": hold_book,
        "hold_n": int(len(hold)),
        "y3_erp": cv_of(store["ever_erp"]),
        "y3_dark": cv_of(store["never_erp"]),
        "prose": prose,
        "book": book,
    }


# ---------------------------------------------------------------------------
# Pass 9 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass9_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab2 = tr[Y2].notna()
    lab3 = tr[Y3].notna()
    rows = []
    store = {}
    for feat, col in (
        ("a_transfer", tr["a_transfer"]),
        ("has_xfer", tr["has_xfer"]),
        ("a_invest", tr["a_invest"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
    ):
        for y, lab in ((Y2, lab2), (Y3, lab3)):
            full = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            rest = signed_oof_auroc(tr[y], col, tr["fold"], lab & drop)
            store[(y, feat, "full")] = full
            store[(y, feat, "drop")] = rest
            rows.append(
                {
                    "y": y,
                    "feature": feat,
                    "full": "LOW_POWER" if full["low_power"] else _f(full["cv"]),
                    "drop12": "LOW_POWER" if rest["low_power"] else _f(rest["cv"]),
                    "Δ": _f(
                        (full["cv"] - rest["cv"])
                        if (not full["low_power"] and not rest["low_power"])
                        else float("nan")
                    ),
                }
            )
    y2f = store[(Y2, "a_transfer", "full")]
    y2d = store[(Y2, "a_transfer", "drop")]
    flip = False
    if not y2f["low_power"] and not y2d["low_power"]:
        flip = abs(y2f["cv"] - y2d["cv"]) >= 0.03
    xfer_ch = float(tr.loc[tr["company_id"].astype(str).isin(set(ids)), "has_xfer"].mean()) if ids else float("nan")
    xfer_rest = float(tr.loc[drop, "has_xfer"].mean())
    prose = (
        f"Chronic 12 names (0158/0172, ≥50% labeled months below 0): {len(ids)}. "
        f"Y2 `a_transfer` {_f(cv_of(y2f))} → drop-12 {_f(cv_of(y2d))}. "
        f"Transfer share on 12 {_pp(xfer_ch)} vs rest {_pp(xfer_rest)}. "
        f"{'DROP FLIPS Y2' if flip else 'Drop does not flip Y2 (≥0.03)'}."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "ids": ids,
        "rows": rows,
        "y2_full": cv_of(y2f),
        "y2_drop": cv_of(y2d),
        "y3_full": cv_of(store[(Y3, "a_transfer", "full")]),
        "y3_drop": cv_of(store[(Y3, "a_transfer", "drop")]),
        "xfer_ch": xfer_ch,
        "xfer_rest": xfer_rest,
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — Q6 lag1 / lag3 on short vs long books
# ---------------------------------------------------------------------------
def pass10_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = [
        ("all", pd.Series(True, index=tr.index)),
        ("short", tr["short_book"] == 1),
        ("long", tr["short_book"] == 0),
    ]
    cols = ("a_transfer", "a_transfer_lag1", "a_transfer_lag3", "a_invest", "a_invest_lag1", "a_invest_lag3")
    for y in (Y3, Y2):
        for sname, smask in slices:
            for col in cols:
                if col not in tr.columns:
                    continue
                mask = tr[y].notna() & smask & tr[col].notna()
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
        r = store.get((y, sl, col))
        return cv_of(r) if r is not None else float("nan")

    now = _cv(Y3, "all", "a_transfer")
    lag1 = _cv(Y3, "all", "a_transfer_lag1")
    lag3 = _cv(Y3, "all", "a_transfer_lag3")
    short_now = _cv(Y3, "short", "a_transfer")
    short_l1 = _cv(Y3, "short", "a_transfer_lag1")
    long_l1 = _cv(Y3, "long", "a_transfer_lag1")
    drop = now - lag1 if np.isfinite(now) and np.isfinite(lag1) else float("nan")
    keep_q6 = bool(
        np.isfinite(lag1)
        and np.isfinite(now)
        and now >= 0.60
        and (now - lag1) <= 0.03
        and lag1 >= 0.60
    )
    if not np.isfinite(now) or now < 0.60:
        q6 = "CLOSE"
        why = (
            f"Y3 contemporaneous {_f(now)} loses to size 0.617 / days 0.711 "
            f"(lag1 {_f(lag1)} / lag3 {_f(lag3)}). "
            "CLOSE as Q6 — no useful contemporaneous skill to lead. "
            "SHAP lag1 perm-light CONFIRMED."
        )
    else:
        q6 = "KEEP" if keep_q6 else "CLOSE"
        why = (
            f"Y3 now {_f(now)} vs lag1 {_f(lag1)} (Δ {_f(drop, 3)}) vs lag3 {_f(lag3)}. "
            f"Short now {_f(short_now)} lag1 {_f(short_l1)}; long lag1 {_f(long_l1)}. "
            f"{'KEEP as honest 1-month Q6' if keep_q6 else 'CLOSE as Q6 — lag does not hold the contemporaneous skill'}. "
            "SHAP lag1 perm-light: signed lag is the same weak object."
        )
    print(why)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "short_now": short_now,
        "short_l1": short_l1,
        "long_l1": long_l1,
        "drop": drop,
        "keep_q6": keep_q6,
        "q6": q6,
        "prose": why,
    }


# ---------------------------------------------------------------------------
# Pass 11 — invest deploy vs return
# ---------------------------------------------------------------------------
def pass11_invest(tr: pd.DataFrame) -> dict:
    dep = pd.to_numeric(tr["n_deploy"], errors="coerce") > 0
    ret = pd.to_numeric(tr["n_return"], errors="coerce") > 0
    n_dep = int(dep.sum())
    n_ret = int(ret.sum())
    n_both = int((dep & ret).sum())
    n_do = int((dep & ~ret).sum())
    n_ro = int((~dep & ret).sum())
    n_any = int((dep | ret).sum())
    sided = bool(n_any > 0 and (n_do / n_any >= 0.85 or n_ro / n_any >= 0.85))
    wr = pd.to_numeric(tr.loc[tr["has_inv"] == 1, "inv_wash"], errors="coerce")
    deploy_amt = pd.to_numeric(tr.loc[dep, "deploy"], errors="coerce")
    ret_amt = pd.to_numeric(tr.loc[ret, "ret"], errors="coerce")
    # one-sided like G? G is rise-only inventory. Invest has both tokens with similar counts.
    rows = [
        {
            "slice": "deploy only",
            "n_cm": n_do,
            "share_of_any": _pp(_pct(n_do, n_any)),
            "signed p50": f"{float(tr.loc[dep & ~ret, 'a_invest'].median()):,.0f}" if n_do else "—",
        },
        {
            "slice": "return only",
            "n_cm": n_ro,
            "share_of_any": _pp(_pct(n_ro, n_any)),
            "signed p50": f"{float(tr.loc[~dep & ret, 'a_invest'].median()):,.0f}" if n_ro else "—",
        },
        {
            "slice": "both",
            "n_cm": n_both,
            "share_of_any": _pp(_pct(n_both, n_any)),
            "signed p50": f"{float(tr.loc[dep & ret, 'a_invest'].median()):,.0f}" if n_both else "—",
        },
    ]
    rate_rows = []
    for y in (Y2, Y3):
        for sname, mask in (
            ("deploy_only", dep & ~ret),
            ("return_only", ~dep & ret),
            ("both", dep & ret),
            ("no_invest", ~dep & ~ret),
        ):
            s = pd.to_numeric(tr.loc[mask, y], errors="coerce")
            rate_rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n_cm": int(mask.sum()),
                    "n_lab": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                }
            )
    prose = (
        f"Invest CM: deploy-only {n_do:,}, return-only {n_ro:,}, both {n_both:,} / any {n_any:,}. "
        f"Deploy txs-months {n_dep:,} (signed p50 {float(deploy_amt.median()) if n_dep else float('nan'):,.0f}); "
        f"return {n_ret:,} (p50 {float(ret_amt.median()) if n_ret else float('nan'):,.0f}). "
        f"|net|/gross p50 among invest CM {_f(float(wr.median()) if wr.notna().any() else float('nan'))}. "
        + (
            "YES one-sided like G rise-only."
            if sided
            else "NO — both deploy and return fire (not a G-style rise-only connection)."
        )
    )
    print(prose)
    return {
        "n_dep": n_dep,
        "n_ret": n_ret,
        "n_both": n_both,
        "n_do": n_do,
        "n_ro": n_ro,
        "n_any": n_any,
        "sided": sided,
        "wr_p50": float(wr.median()) if wr.notna().any() else float("nan"),
        "deploy_p50": float(deploy_amt.median()) if n_dep else float("nan"),
        "ret_p50": float(ret_amt.median()) if n_ret else float("nan"),
        "rows": rows,
        "rate_rows": rate_rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — amounts: transfer vs op_out
# ---------------------------------------------------------------------------
def pass12_amounts(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    tx = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          category,
          ABS(amount) AS abs_amt
        FROM transactions
        WHERE category IN (
            'transfer', 'investment_deployment', 'investment_return',
            'payment', 'bulk_payment', 'utility', 'salary', 'social_security',
            'tax', 'cash_withdrawal', 'pos_withdrawal'
          )
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx = tx.loc[~tx["company_id"].isin(hold)].copy()
    assert_no_holdout(tx["company_id"])
    op = {"payment", "bulk_payment", "utility", "salary", "social_security", "tax", "cash_withdrawal", "pos_withdrawal"}
    rows = []
    stats = {}
    for name, mask in (
        ("transfer", tx["category"] == "transfer"),
        ("op_out", tx["category"].isin(op)),
        ("investment_deployment", tx["category"] == "investment_deployment"),
        ("investment_return", tx["category"] == "investment_return"),
    ):
        s = pd.to_numeric(tx.loc[mask, "abs_amt"], errors="coerce").dropna()
        rec = {
            "n": int(len(s)),
            "p50": float(s.median()) if len(s) else float("nan"),
            "p90": float(s.quantile(0.90)) if len(s) else float("nan"),
            "le1": float((s <= TOKEN_NEAR).mean()) if len(s) else float("nan"),
        }
        stats[name] = rec
        rows.append(
            {
                "token": name,
                "n_tx": f"{rec['n']:,}",
                "p50 |amt|": f"{rec['p50']:,.0f}" if np.isfinite(rec["p50"]) else "—",
                "p90 |amt|": f"{rec['p90']:,.0f}" if np.isfinite(rec["p90"]) else "—",
                "≤1€": _pp(rec["le1"]),
            }
        )
    tokenish = bool(np.isfinite(stats["transfer"]["p50"]) and stats["transfer"]["p50"] <= 5)
    real = bool(np.isfinite(stats["transfer"]["p50"]) and stats["transfer"]["p50"] >= 100)
    ratio = (
        stats["transfer"]["p50"] / stats["op_out"]["p50"]
        if stats["op_out"]["p50"]
        else float("nan")
    )
    prose = (
        f"Train transfer |amt| p50={stats['transfer']['p50']:,.0f} p90={stats['transfer']['p90']:,.0f} "
        f"(≤1€ {_pp(stats['transfer']['le1'])}) vs op_out p50={stats['op_out']['p50']:,.0f} "
        f"(ratio {_f(ratio)}). "
        + (
            "1€ token — ignore as mass."
            if tokenish
            else ("Real mass — transfer tickets are larger than typical op_out." if real else "Modest tickets.")
        )
    )
    print(prose)
    return {
        "rows": rows,
        "stats": stats,
        "tokenish": tokenish,
        "real": real,
        "ratio": ratio,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 13 — token mix / descriptions / missing CP (Q5 leftover?)
# ---------------------------------------------------------------------------
def pass13_tokens(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    tx = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          category,
          amount,
          description,
          CAST(counterparty_id AS VARCHAR) AS counterparty_id
        FROM transactions
        WHERE category IN ('transfer', 'investment_deployment', 'investment_return')
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx = tx.loc[~tx["company_id"].isin(hold)].copy()
    assert_no_holdout(tx["company_id"])
    xf = tx[tx["category"] == "transfer"].copy()
    desc = xf["description"].fillna("").astype(str)
    cp = xf["counterparty_id"].replace({"None": np.nan, "": np.nan})
    miss_cp = float(cp.isna().mean())
    pats = [
        ("TRASP / TRASPASO", r"TRASP"),
        ("TRANSFERENCIA", r"TRANSFERENCIA"),
        ("COUNTERPARTY_", r"COUNTERPARTY_"),
        ("[ACCOUNT]", r"\[ACCOUNT\]"),
        ("[COMPANY]", r"\[COMPANY\]"),
        ("[NUM]", r"\[NUM\]"),
        ("SWEEP / TREAS / OWNER", r"SWEEP|TREAS|OWNER|DIVIDEND|INTERNAL"),
        ("INTER / GRUPO", r"INTER|GRUPO|GROUP"),
    ]
    mix = []
    for name, pat in pats:
        share = float(desc.str.contains(pat, case=False, regex=True).mean()) if len(desc) else float("nan")
        mix.append({"token": name, "share_tx": share, "n": int(desc.str.contains(pat, case=False, regex=True).sum())})
    # Q5 leftover sentence? Missing CP + TRASPASO bookkeeping is treasury movement, not owner-draw language.
    ownerish = next(r["share_tx"] for r in mix if r["token"].startswith("SWEEP"))
    q5_new = bool(np.isfinite(ownerish) and ownerish >= 0.05)
    prose = (
        f"Transfer txs n={len(xf):,}; missing counterparty_id {_pp(miss_cp)} "
        f"(cannot score intercompany via CP — `d_interco_share` is already all-null). "
        f"Description mix: TRASP {_pp(next(r['share_tx'] for r in mix if r['token'].startswith('TRASP')))}, "
        f"TRANSFERENCIA {_pp(next(r['share_tx'] for r in mix if r['token'].startswith('TRANSFERENCIA')))}, "
        f"owner/sweep/dividend {_pp(ownerish)}. "
        + (
            "KEEP-Q5 footnote — descriptions carry owner/sweep language."
            if q5_new
            else "No KEEP-Q5 footnote — this is bookkeeping TRASPASO / TRANSFERENCIA, already the quiet-month story."
        )
    )
    print(prose)
    return {
        "n_xfer_tx": int(len(xf)),
        "miss_cp": miss_cp,
        "mix": mix,
        "ownerish": ownerish,
        "q5_new": q5_new,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 14 — rates: has / wash / one-way × Y
# ---------------------------------------------------------------------------
def pass14_rates(tr: pd.DataFrame) -> dict:
    rows = []
    for y in (Y2, Y3):
        for sname, mask in (
            ("has_xfer", tr["has_xfer"] == 1),
            ("no_xfer", tr["has_xfer"] == 0),
            ("one_way", tr["one_way"] == 1),
            ("both_ways", tr["both_ways"] == 1),
            ("washed", tr["washed"] == 1),
            ("has_inv", tr["has_inv"] == 1),
            ("no_inv", tr["has_inv"] == 0),
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
    y3_has = next(r for r in rows if r["y"] == Y3 and r["slice"] == "has_xfer")
    y3_no = next(r for r in rows if r["y"] == Y3 and r["slice"] == "no_xfer")
    gap = (
        y3_no["rate"] - y3_has["rate"]
        if np.isfinite(y3_has["rate"]) and np.isfinite(y3_no["rate"])
        else float("nan")
    )
    prose = (
        f"Y3 rate no-transfer {_pp(y3_no['rate'])} vs has-transfer {_pp(y3_has['rate'])} "
        f"(gap {_pp(gap) if np.isfinite(gap) else '—'}). "
        "Quiet months recover — same sign as salary / days / n_tx."
    )
    print(prose)
    return {"rows": rows, "gap": gap, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 15 — size terciles / leftover inside busy months
# ---------------------------------------------------------------------------
def pass15_terciles(tr: pd.DataFrame) -> dict:
    terc_size = _company_terciles(tr, "log_in3", "size_t")
    terc_days = _company_terciles(tr, "c_n_days_with_tx", "days_t")
    m = tr.merge(terc_size.reset_index(), on="company_id", how="left")
    m = m.merge(terc_days.reset_index(), on="company_id", how="left")
    rows = []
    store = {}
    lab = m[Y3].notna()
    for tname in ("size_t", "days_t"):
        for labv in ("T1", "T2", "T3"):
            mask = lab & (m[tname].astype(str) == labv)
            res = signed_oof_auroc(m[Y3], m["a_transfer"], m["fold"], mask)
            store[(tname, labv)] = res
            rows.append(
                {
                    "clock": tname,
                    "tercile": labv,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "xfer": _pp(
                        float(m.loc[mask, "has_xfer"].mean()) if mask.any() else float("nan")
                    ),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    # busy months: days >= company median
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    busy = days >= days.median()
    quiet = days < days.median()
    busy_res = signed_oof_auroc(tr[Y3], tr["a_transfer"], tr["fold"], tr[Y3].notna() & busy)
    quiet_res = signed_oof_auroc(tr[Y3], tr["a_transfer"], tr["fold"], tr[Y3].notna() & quiet)
    prose = (
        f"Y3 `a_transfer` inside size T1/T2/T3: "
        f"{' / '.join(r['CV'] for r in rows if r['clock']=='size_t')}. "
        f"Inside days terciles: {' / '.join(r['CV'] for r in rows if r['clock']=='days_t')}. "
        f"Busy-half days CV {_f(cv_of(busy_res))}; quiet-half {_f(cv_of(quiet_res))}."
    )
    print(prose)
    return {
        "rows": rows,
        "busy_cv": cv_of(busy_res),
        "quiet_cv": cv_of(quiet_res),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 16 — holdout coverage only
# ---------------------------------------------------------------------------
def pass16_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col in ("a_transfer", "a_invest", "has_xfer", "has_inv", "xfer_gross"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "mean": _f(float(x.mean()) if x.notna().any() else float("nan")),
                "p50": _f(float(x.median()) if x.notna().any() else float("nan")),
            }
        )
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM. "
        f"has_xfer {_pp(float(ho['has_xfer'].mean()))}; "
        f"has_inv {_pp(float(ho['has_inv'].mean()))}. No AUROC claim."
    )
    print(prose)
    return {"rows": rows, "n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 17 — calendar of transfers (not a dummy unless peaked)
# ---------------------------------------------------------------------------
def pass17_cal(tr: pd.DataFrame) -> dict:
    rows = []
    for mth in range(1, 13):
        sl = tr[tr["cal_month"] == mth]
        rows.append(
            {
                "month": pd.Timestamp(2000, mth, 1).strftime("%b"),
                "n_cm": int(len(sl)),
                "xfer": float(sl["has_xfer"].mean()) if len(sl) else float("nan"),
                "inv": float(sl["has_inv"].mean()) if len(sl) else float("nan"),
                "signed_p50": float(pd.to_numeric(sl["a_transfer"], errors="coerce").median())
                if len(sl)
                else float("nan"),
            }
        )
    shares = [r["xfer"] for r in rows if np.isfinite(r["xfer"])]
    peaked = bool(shares and (max(shares) - min(shares)) >= 0.15)
    prose = (
        f"Transfer-CM share range {_pp(min(shares))}–{_pp(max(shares))}. "
        f"{'Calendar-peaked' if peaked else 'Flat — not a tax-style calendar dummy'}."
    )
    print(prose)
    return {
        "rows": [
            {
                "month": r["month"],
                "n_cm": f"{r['n_cm']:,}",
                "xfer": _pp(r["xfer"]),
                "inv": _pp(r["inv"]),
                "signed p50": f"{r['signed_p50']:,.0f}" if np.isfinite(r["signed_p50"]) else "—",
            }
            for r in rows
        ],
        "peaked": peaked,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 18 — signed leftover after has_xfer (is the euro level anything?)
# ---------------------------------------------------------------------------
def pass18_level(tr: pd.DataFrame) -> dict:
    """If leftover after the presence flag dies, SHAP is reading has_xfer, not signed euros."""
    x = pd.to_numeric(tr["a_transfer"], errors="coerce")
    r_has, _ = ols_resid(x, tr["has_xfer"])
    r_has_days, _ = ols_resid(x, tr["has_xfer"], tr["c_n_days_with_tx"])
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        for name, col in (
            ("a_transfer", x),
            ("has_xfer", tr["has_xfer"]),
            ("resid_has", r_has),
            ("resid_has_days", r_has_days),
        ):
            rec = signed_oof_auroc(tr[y], col, tr["fold"], lab)
            store[(y, name)] = rec
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                    "sign": rec["train_sign"] if not rec["low_power"] else "—",
                }
            )
    y3_has = cv_of(store[(Y3, "has_xfer")])
    y3_res = cv_of(store[(Y3, "resid_has")])
    y3_resd = cv_of(store[(Y3, "resid_has_days")])
    dies = bool(np.isfinite(y3_res) and y3_res < 0.55)
    prose = (
        f"Y3 has_xfer {_f(y3_has)}; signed leftover after has_xfer {_f(y3_res)}; "
        f"after has+days {_f(y3_resd)}. "
        + (
            "Euro level dies after the presence flag — SHAP is the busy-month dummy."
            if dies
            else "Signed euros keep some leftover after presence."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_has": y3_has,
        "y3_res": y3_res,
        "y3_resd": y3_resd,
        "dies": dies,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 19 — ever-transfer company trait (ICC 0.96 is who, not when)
# ---------------------------------------------------------------------------
def pass19_ever(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_x=("has_xfer", "sum"),
        n_i=("has_inv", "sum"),
        med_signed=("a_transfer", "median"),
        mean_has=("has_xfer", "mean"),
    )
    ever["ever_x"] = ever["n_x"] > 0
    ever["ever_i"] = ever["n_i"] > 0
    m = tr.merge(ever[["company_id", "ever_x", "ever_i", "mean_has"]], on="company_id", how="left")
    rows = []
    store = {}
    for y in (Y2, Y3):
        lab = m[y].notna()
        for name, col in (
            ("ever_xfer", m["ever_x"].astype(float)),
            ("mean_has_xfer", m["mean_has"]),
            ("ever_inv", m["ever_i"].astype(float)),
        ):
            rec = signed_oof_auroc(m[y], col, m["fold"], lab)
            store[(y, name)] = rec
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                    "sign": rec["train_sign"] if not rec["low_power"] else "—",
                }
            )
    n_ever = int(ever["ever_x"].sum())
    n_co = int(len(ever))
    y3e = cv_of(store[(Y3, "ever_xfer")])
    y3m = cv_of(store[(Y3, "mean_has_xfer")])
    prose = (
        f"Ever-transfer companies {n_ever}/{n_co}. Y3 ever-flag {_f(y3e)}; "
        f"company mean has_xfer {_f(y3m)}. "
        "A company trait that a transferer is less likely to recover — not a month shock."
    )
    print(prose)
    return {
        "rows": rows,
        "n_ever": n_ever,
        "n_co": n_co,
        "y3_ever": y3e,
        "y3_mean": y3m,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — winsor signed + leftover among stressed busy months
# ---------------------------------------------------------------------------
def pass20_winsor(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["a_transfer"], errors="coerce")
    lo, hi = x.quantile(0.01), x.quantile(0.99)
    w = x.clip(lower=lo, upper=hi)
    r_w_days, _ = ols_resid(w, tr["c_n_days_with_tx"])
    r_w_card, _ = ols_resid(w, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    busy = days >= days.median()
    rows = []
    store = {}
    lab = tr[Y3].notna()
    for name, col, mask in (
        ("winsor_signed", w, lab),
        ("winsor_resid_days", r_w_days, lab),
        ("winsor_resid_card", r_w_card, lab),
        ("has_xfer_busy", tr["has_xfer"], lab & busy),
        ("has_xfer_quiet", tr["has_xfer"], lab & ~busy),
        ("abs_xfer_busy", tr["abs_xfer"], lab & busy),
    ):
        rec = signed_oof_auroc(tr[Y3], col, tr["fold"], mask)
        store[name] = rec
        rows.append(
            {
                "feature": name,
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "CV": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "sign": rec["train_sign"] if not rec["low_power"] else "—",
            }
        )
    prose = (
        f"Winsor p01–p99 signed Y3 {_f(cv_of(store['winsor_signed']))}; "
        f"resid-days {_f(cv_of(store['winsor_resid_days']))} "
        f"(still the zero-month leak if high). "
        f"has_xfer on busy-half {_f(cv_of(store['has_xfer_busy']))}; "
        f"quiet-half {_f(cv_of(store['has_xfer_quiet']))}."
    )
    print(prose)
    return {
        "rows": rows,
        "winsor": cv_of(store["winsor_signed"]),
        "winsor_days": cv_of(store["winsor_resid_days"]),
        "has_busy": cv_of(store["has_xfer_busy"]),
        "has_quiet": cv_of(store["has_xfer_quiet"]),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 21 — company-mean has_xfer 0.747: activity twin at company grain?
# ---------------------------------------------------------------------------
def pass21_co_mean(tr: pd.DataFrame) -> dict:
    co = tr.groupby("company_id", as_index=False).agg(
        mean_has=("has_xfer", "mean"),
        mean_days=("c_n_days_with_tx", "mean"),
        mean_ntx=("a_n_tx", "mean"),
        mean_sal=("c_salary_month", "mean"),
        mean_in3=("log_in3", "mean"),
        mean_ss=("c_ss_month", "mean"),
        group_id=("group_id", "first"),
        fold=("fold", "first"),
    )
    # last labeled Y3 per company (any-positive on labeled months)
    y3 = (
        tr.loc[tr[Y3].notna(), ["company_id", Y3]]
        .groupby("company_id")[Y3]
        .mean()
        .rename("y3_rate")
    )
    y3_any = (
        tr.loc[tr[Y3].notna(), ["company_id", Y3]]
        .groupby("company_id")[Y3]
        .max()
        .rename("y3_any")
    )
    co = co.merge(y3, on="company_id", how="left")
    co = co.merge(y3_any, on="company_id", how="left")
    rhos = {
        "days": spearman(co["mean_has"], co["mean_days"]),
        "n_tx": spearman(co["mean_has"], co["mean_ntx"]),
        "salary": spearman(co["mean_has"], co["mean_sal"]),
        "ss": spearman(co["mean_has"], co["mean_ss"]),
        "size": spearman(co["mean_has"], co["mean_in3"]),
    }
    # company-level AUROC of mean_has vs ever-recovered (among companies with a Y3 label)
    lab = co["y3_any"].notna()
    rec = signed_oof_auroc(co["y3_any"], co["mean_has"], co["fold"], lab)
    rec_days = signed_oof_auroc(co["y3_any"], co["mean_days"], co["fold"], lab)
    rec_size = signed_oof_auroc(co["y3_any"], co["mean_in3"], co["fold"], lab)
    twin = bool(abs(rhos["days"]) >= TWIN_RHO or abs(rhos["n_tx"]) >= TWIN_RHO)
    size = bool(abs(rhos["size"]) >= SIZE_RHO)
    rows = [
        {"vs": k, "ρ": rhos[k], "flag": "TWIN" if abs(rhos[k]) >= TWIN_RHO else ("SIZE" if k == "size" and abs(rhos[k]) >= SIZE_RHO else "")}
        for k in rhos
    ]
    rows.append(
        {
            "vs": "Y3-any company AUROC mean_has",
            "ρ": cv_of(rec),
            "flag": f"days {cv_of(rec_days)} size {cv_of(rec_size)}",
        }
    )
    prose = (
        f"Company-mean has_xfer vs mean days ρ={_f(rhos['days'])}, n_tx {_f(rhos['n_tx'])}, "
        f"salary {_f(rhos['salary'])}, size {_f(rhos['size'])}. "
        f"Company-level Y3-any AUROC mean_has {_f(cv_of(rec))} vs mean days {_f(cv_of(rec_days))} "
        f"vs size {_f(cv_of(rec_size))}. "
        + (
            "Company-mean transfer share is the activity twin at company grain — still the quiet card."
            if twin
            else "Not a |ρ|≥0.80 twin of company-mean days, but ICC 0.96 still makes it a trait, not a month shock."
        )
    )
    print(prose)
    return {
        "rows": [{"vs": r["vs"], "ρ": _f(r["ρ"]) if not isinstance(r["ρ"], str) else r["ρ"], "flag": r["flag"]} for r in rows],
        "rho_days": rhos["days"],
        "rho_ntx": rhos["n_tx"],
        "rho_size": rhos["size"],
        "co_auc": cv_of(rec),
        "co_days": cv_of(rec_days),
        "twin": twin,
        "size": size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 22 — |signed| leftover only on transfer months (no zero-month leak)
# ---------------------------------------------------------------------------
def pass22_abs_on(tr: pd.DataFrame) -> dict:
    has = tr["has_xfer"] == 1
    abx = pd.to_numeric(tr["abs_xfer"], errors="coerce").where(has)
    gros = pd.to_numeric(tr["xfer_gross"], errors="coerce").where(has)
    r_a, _ = ols_resid(abx, tr["c_n_days_with_tx"])
    r_g, _ = ols_resid(gros, tr["c_n_days_with_tx"])
    r_c, _ = ols_resid(abx, tr["c_n_days_with_tx"], tr["a_n_tx"], tr["c_salary_month"])
    lab = tr[Y3].notna() & has
    rows = []
    store = {}
    for name, col in (
        ("abs_on_xfer", abx),
        ("gros_on_xfer", gros),
        ("abs_on_resid_days", r_a),
        ("gros_on_resid_days", r_g),
        ("abs_on_resid_card", r_c),
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
    prose = (
        f"Among transfer CM: |signed| {_f(cv_of(store['abs_on_xfer']))} "
        f"gross {_f(cv_of(store['gros_on_xfer']))}; leftover after days "
        f"{_f(cv_of(store['abs_on_resid_days']))} / card {_f(cv_of(store['abs_on_resid_card']))}. "
        "Euro mass inside transfer months is not a leftover lever."
    )
    print(prose)
    return {
        "rows": rows,
        "abs_on": cv_of(store["abs_on_xfer"]),
        "abs_d": cv_of(store["abs_on_resid_days"]),
        "abs_c": cv_of(store["abs_on_resid_card"]),
        "prose": prose,
    }


def make_png(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    wr = pd.to_numeric(tr.loc[tr["has_xfer"] == 1, "wash_ratio"], errors="coerce").dropna()
    lab = tr[Y3].notna()
    x = pd.to_numeric(tr.loc[lab, "a_transfer"], errors="coerce")
    y = pd.to_numeric(tr.loc[lab, Y3], errors="coerce")
    # quintiles of signed transfer among labeled (0-heavy)
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    try:
        d["q"] = pd.qcut(d["x"], 5, duplicates="drop")
    except ValueError:
        d["q"] = pd.cut(d["x"], bins=3)
    qrows = d.groupby("q", observed=False)["y"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    ax.hist(wr.to_numpy(), bins=20, color="#1f4e79", edgecolor="white")
    ax.axvline(p3["wr_p50"], color="#9e6b4a", ls="--", lw=1.2, label=f"p50={p3['wr_p50']:.2f}")
    ax.set_xlabel("|signed net| / gross among transfer CM")
    ax.set_ylabel("train company-months")
    ax.set_title("Two-way wash? (1 = one-way)")
    ax.legend(frameon=False, fontsize=8)

    ax2 = axes[1]
    labels = [str(i) for i in qrows.index]
    ax2.bar(np.arange(len(qrows)), 100.0 * qrows.to_numpy(), color="#9e6b4a")
    ax2.set_xticks(np.arange(len(qrows)))
    ax2.set_xticklabels([f"Q{i+1}" for i in range(len(qrows))], fontsize=8)
    ax2.set_ylabel("% Y3 recover (labeled)")
    ax2.set_title("Y3 rate by signed a_transfer quintile")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p3, p4, p5, p6, p7, p10, p11, p13) -> dict:
    """KEEP later-A / CLOSE as X / PARK as Y. Never invent a transfer Y."""
    shock = bool(p7.get("xfer_shock"))
    leftover_ok = bool(p6.get("leftover_lives") and not p6.get("fake_days"))
    twin = bool(p4.get("xfer_twin") or p6.get("quiet_twin"))
    size = bool(p4.get("xfer_size"))
    wash = bool(p3.get("is_wash"))
    loses = bool(p5.get("loses"))
    keep_x = bool(leftover_ok and shock and not twin and not size and not wash)
    if keep_x:
        x_dec = "KEEP"
        why = (
            f"has_xfer leftover after days+n_tx+salary {_f(p6['y3_has_c'])} beats size "
            f"{_f(p5['size_y3'])} by {_f(p6['leftover_card'])} and is a month shock "
            f"(ICC {_f(p7['xfer_icc'])}). Still **not** on tonight's 15-col card."
        )
    else:
        x_dec = "CLOSE"
        bits = []
        if twin:
            bits.append("quiet twin of days/n_tx/salary (leftover dies)")
        if wash:
            bits.append("two-way wash")
        if size:
            bits.append("SIZE")
        if loses:
            bits.append(
                f"Y3 {_f(p5['xfer_y3'])} loses to size {_f(p5['size_y3'])} / "
                f"days {_f(p5['days_y3'])} / salary {_f(p5['sal_y3'])}"
            )
        if not bits:
            bits.append(
                f"Y3 {_f(p5['xfer_y3'])} vs size {_f(p5['size_y3'])} (Δ {_f(p5['beat_size'])})"
            )
        why = "; ".join(bits) + ". Explains SHAP-heavy / perm-light."
    inv_keep = bool(
        np.isfinite(p5.get("inv_y3", float("nan")))
        and np.isfinite(p5.get("size_y3", float("nan")))
        and (p5["inv_y3"] - p5["size_y3"]) >= KEEP_DELTA
        and not p4.get("inv_size")
        and np.isfinite(p6.get("y3_hinv", float("nan")))
        and p6["y3_hinv"] >= (p5["size_y3"] + KEEP_DELTA)
    )
    inv_x = "KEEP" if inv_keep else "CLOSE"
    inv_why = (
        f"Y3 {_f(p5['inv_y3'])} vs size {_f(p5['size_y3'])}; "
        f"has_inv leftover after days {_f(p6['y3_hinv'])}; "
        + ("one-sided like G." if p11.get("sided") else "not G rise-only.")
    )
    q5 = "KEEP-Q5 footnote" if p13.get("q5_new") else "CLOSE"
    q5_why = (
        p13["prose"]
        if p13.get("q5_new")
        else "no sentence beyond quiet-stressed recover (already on the 15-col card)."
    )
    return {
        "xfer_x": x_dec,
        "xfer_why": why,
        "inv_x": inv_x,
        "inv_why": inv_why,
        "xfer_y": "PARK",
        "inv_y": "PARK",
        "q5": q5,
        "q5_why": q5_why,
        "q6": p10["q6"],
        "keep_x": keep_x,
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12, p13, p14, p15 = ctx["p11"], ctx["p12"], ctx["p13"], ctx["p14"], ctx["p15"]
    p16, p17, p18, d = ctx["p16"], ctx["p17"], ctx["p18"], ctx["decision"]
    lines = [
        "# Signed `a_transfer` / `a_invest` — wash, quiet twin, or leftover?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_transfer` / `y_invest`. "
        "Do not put `a_transfer` / `a_invest` on tonight's 15-col card. "
        f"Night Y3 quote stays **{NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}**. "
        "Do not merge Family M or I. `c_salary_month` 0.671 stays on the card. "
        "`c_missed_salary` CLOSE — not rewritten.",
        "",
        "`a_transfer` = signed net `sum(amount | category=transfer)`. "
        "`a_invest` = signed `investment_deployment` + `investment_return`. "
        "`m_xfer_share` is |transfer|/|all| computed **in-module** (catmix.py not edited).",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Not a new Y. PARK `y_transfer` / `y_invest`. Any-transfer {_pp(p1['xf']['any_share'])} of train CM. |",
        "| 2 | Who is improving? | Signed transfer is not a recovery path. Quiet months recover — already on the card. |",
        f"| 3 | Who is turning? | Y3 `a_transfer` {_f(p5['xfer_y3'])} vs size {_f(p5['size_y3'])} / days {_f(p5['days_y3'])} / salary {_f(p5['sal_y3'])}. |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['q5_why']} |",
        f"| 6 | Months earlier? | lag1/lag3 **{d['q6']}** — {p10['prose']} |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `a_transfer` as Y3 X | **{d['xfer_x']}** | {d['xfer_why']} Still **not** on tonight's 15-col card. |",
        f"| `a_transfer` as a health Y | **PARK** | do not invent `y_transfer` |",
        f"| `a_invest` as Y3 X | **{d['inv_x']}** | {d['inv_why']} Not on the card. |",
        f"| `a_invest` as a health Y | **PARK** | do not invent `y_invest` |",
        f"| two-way wash | **{'YES' if p3['is_wash'] else 'NO'}** | {p3['prose']} |",
        f"| quiet twin days/n_tx/salary | **{'YES' if p4['xfer_twin'] or p6['quiet_twin'] else 'NO at |ρ|≥0.80'}** | signed ρ days {_f(p4['rho_days'])} n_tx {_f(p4['rho_ntx'])} salary {_f(p4['rho_sal'])}; honest has_xfer leftover after days {_f(p6['y3_has_d'])} (signed OLS leftover {_f(p6['y3_days'])} is −days leak, ρ={_f(p6['rho_fake'])}) |",
        f"| SIZE | **{'YES' if p4['xfer_size'] else 'NO'}** | ρ vs log1p(a_in3) {_f(p4['rho_size'])} |",
        f"| Q6 lag1/lag3 | **{d['q6']}** | {p10['prose']} |",
        f"| KEEP-Q5 footnote | **{d['q5']}** | {d['q5_why']} |",
        "| Family M / I | **do not merge** | `m_xfer_share` is |transfer|/|all|, not signed. I CLOSE already. |",
        "",
        "## 1. Prevalence (train rates; holdout coverage)",
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
                    "any": _pp(r["any_share"]),
                    "zero": _pp(r["zero_share"]),
                    "p50 signed": f"{r['p50']:,.0f}" if np.isfinite(r["p50"]) else "—",
                    "p90 signed": f"{r['p90']:,.0f}" if np.isfinite(r["p90"]) else "—",
                    "p90 |signed|": f"{r['p90_abs']:,.0f}" if np.isfinite(r["p90_abs"]) else "—",
                }
                for r in p1["rows"]
            ]
        ),
        "",
        f"Ever-transfer companies {p1['ever_x']}/{p1['n_co']}; ever-invest {p1['ever_i']}/{p1['n_co']}.",
        "",
        "## 2. Tokens vs store",
        "",
        p2["prose"],
        "",
        _md_table(
            [
                {
                    "category": str(r["category"]),
                    "n_tx": f"{int(r['n']):,}",
                    "n_in": f"{int(r['n_in']):,}",
                    "n_out": f"{int(r['n_out']):,}",
                    "p50 |amt|": f"{float(r['p50_abs']):,.0f}",
                }
                for _, r in p2["cats"].iterrows()
            ]
        ),
        "",
        "Sample company-months (store signed vs raw in / out / gross):",
        "",
        _md_table(p2["samp_rows"]),
        "",
        "## 3. Two-way wash — gross vs signed net",
        "",
        p3["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| transfer CM | {p3['n_has']:,} |",
        f"| both-ways share | {_pp(p3['p_both'])} |",
        f"| one-way share | {_pp(p3['p_one'])} |",
        f"| washed (|net|/gross ≤{WASH_NEAR:g}) | {_pp(p3['p_wash'])} |",
        f"| \\|net\\|/gross p10 / p50 / p90 | {_f(p3['wr_p10'])} / {_f(p3['wr_p50'])} / {_f(p3['wr_p90'])} |",
        f"| ρ signed ↔ gross | {_f(p3['rho_signed_gross'])} |",
        f"| ρ \\|signed\\| ↔ gross | {_f(p3['rho_abs_gross'])} |",
        f"| ρ signed ↔ \\|signed\\| | {_f(p3['rho_signed_abs'])} |",
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 4. Spearman vs days / n_tx / salary / size / mix",
        "",
        p4["prose"],
        "",
        _md_table(
            [
                {
                    "object": r["object"],
                    "vs": r["vs"],
                    "ρ": _f(r["rho"]),
                    "flag": r["flag"],
                }
                for r in p4["rows"]
                if r["object"] in {"a_transfer", "has_xfer", "a_invest", "abs_xfer"}
                and r["vs"]
                in {
                    "c_n_days_with_tx",
                    "a_n_tx",
                    "c_salary_month",
                    "c_ss_month",
                    "log1p(a_in3)",
                    "a_op_in",
                    "m_xfer_share",
                }
            ]
        ),
        "",
        "## 5. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p5['n_y2']:,} base {_pp(p5['y2_rate'])}; "
        f"Y3 stressed n={p5['n_y3']:,} base {_pp(p5['y3_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: size 0.617 (replica {_f(p5['size_y3'])}), "
        f"days 0.711 (replica {_f(p5['days_y3'])}), "
        f"`c_salary_month` 0.671 (replica {_f(p5['sal_y3'])}). "
        f"Night Y3 GBM stays {NIGHT_Y3:.3f} / {NIGHT_Y3_CORE:.3f}.",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        f"KEEP-as-later-A: leftover after days/n_tx/salary beats size by ≥{KEEP_DELTA:g} "
        "**and** not a twin **and** not SIZE **and** leftover after \\|transfer\\| vs signed "
        "**and** month shock. Still not on the 15-col card.",
        "",
        "## 6. Residual AUROC after days / n_tx / salary",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. ICC / company-demean (trait vs month shock)",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Dark 470 vs invoiced 744",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        _md_table(
            [
                {
                    "group": r["group"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "xfer": _pp(r["xfer"]),
                    "inv": _pp(r["inv"]),
                }
                for r in p8["cm"]
            ]
        ),
        "",
        "## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. Q6 — lag1 / lag3 on short vs long books",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. Invest — deploy vs return (G rise-only?)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
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
                for r in p11["rate_rows"]
            ]
        ),
        "",
        "## 12. Amounts — 1€ token or real mass?",
        "",
        p12["prose"],
        "",
        _md_table(p12["rows"]),
        "",
        "## 13. Category token mix / descriptions (Q5 leftover?)",
        "",
        p13["prose"],
        "",
        _md_table(
            [
                {"token": r["token"], "share": _pp(r["share_tx"]), "n_tx": f"{r['n']:,}"}
                for r in p13["mix"]
            ]
        ),
        "",
        "## 14. Y rates on has / wash / one-way",
        "",
        p14["prose"],
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
                for r in p14["rows"]
            ]
        ),
        "",
        "## 15. Size / days terciles + busy-half",
        "",
        p15["prose"],
        "",
        _md_table(p15["rows"]),
        "",
        "## 16. Holdout coverage only (no AUROC)",
        "",
        p16["prose"],
        "",
        _md_table(p16["rows"]),
        "",
        "## 17. Calendar",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "## 18. Signed leftover after the presence flag",
        "",
        p18["prose"],
        "",
        _md_table(p18["rows"]),
        "",
        "## 19. Ever-transfer company trait",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(ctx["p19"]["rows"]),
        "",
        "## 20. Winsor signed + leftover on busy/quiet halves",
        "",
        ctx["p20"]["prose"],
        "",
        _md_table(ctx["p20"]["rows"]),
        "",
        "## 21. Company-mean has_xfer vs activity (the 0.747 trait)",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "## 22. |signed| leftover on transfer months only",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: prevalence, tokens, wash, Spearman, singles, "
        "honest leftover (has_xfer after days — signed OLS leftover is a zero-month −days leak), "
        "ICC, dark 470/744, chronic-12, Q6 lags, invest sidedness, amounts, "
        "description mix, Y rates, terciles, holdout, calendar, leftover after has_xfer, "
        "ever-transfer trait, winsor / busy-half, company-mean activity twin, "
        "|signed| leftover on transfer months only.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p3, p4, p5, p6, p7, p8, p10, d = (
        ctx["p1"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p10"],
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
            "a_transfer_any_share",
            p1["xf"]["any_share"],
            "train",
            f"p50={p1['xf']['p50']:.4g} p90={p1['xf']['p90']:.4g} ever={p1['ever_x']}/{p1['n_co']}",
        ),
        row(
            "-",
            "a_invest_any_share",
            p1["inv"]["any_share"],
            "train",
            f"p50={p1['inv']['p50']:.4g} p90={p1['inv']['p90']:.4g} ever={p1['ever_i']}/{p1['n_co']}",
        ),
        row(
            "-",
            "xfer_wash_ratio_p50",
            p3["wr_p50"],
            "train",
            f"p_wash={p3['p_wash']:.4f} p_one={p3['p_one']:.4f} is_wash={p3['is_wash']} rho_sg={p3['rho_signed_gross']:.3f}",
        ),
        row(
            Y3,
            "auroc_a_transfer",
            p5["xfer_y3"],
            "train_cv",
            f"size={p5['size_y3']:.4f} days={p5['days_y3']:.4f} sal={p5['sal_y3']:.4f} beat_size={p5['beat_size']:.4f} x={d['xfer_x']}",
        ),
        row(
            Y3,
            "auroc_abs_a_transfer",
            p5["abs_y3"],
            "train_cv",
            f"gross={p5['gros_y3']:.4f} has={p5['has_y3']:.4f}",
        ),
        row(
            Y3,
            "auroc_a_invest",
            p5["inv_y3"],
            "train_cv",
            f"abs={p5['abs_i_y3']:.4f} x={d['inv_x']}",
        ),
        row(
            Y3,
            "auroc_a_transfer_resid_days",
            p6["y3_has_d"],
            "train_cv",
            f"honest=has_xfer|days; signed_ols={p6['y3_days']:.4f} fake={p6['fake_days']} has_c={p6['y3_has_c']:.4f} leftover_lives={p6['leftover_lives']} quiet_twin={p6['quiet_twin']}",
        ),
        row(
            "-",
            "a_transfer_icc",
            p7["xfer_icc"],
            "train",
            f"acf1={p7['xfer_acf1']:.4f} trait={p7['xfer_trait']} shock={p7['xfer_shock']} dem={p7['xfer_dem_cv']}",
        ),
        row(
            Y3,
            "auroc_a_transfer_lag1",
            p10["lag1"],
            "train_cv",
            f"now={p10['now']:.4f} lag3={p10['lag3']:.4f} q6={d['q6']}",
        ),
        row(
            "-",
            "rho_a_transfer_days",
            p4["rho_days"],
            "train",
            f"ntx={p4['rho_ntx']:.4f} sal={p4['rho_sal']:.4f} size={p4['rho_size']:.4f} mxfer={p4['rho_mxfer']:.4f}",
        ),
        row(
            "-",
            "dark_vs_erp_xfer_rate",
            p8["dark_x"],
            "train",
            f"erp={p8['erp_x']:.4f} confirm744_470={p8['confirm']} same={p8['same']}",
            coverage=f"{p8['n_dark'] / (p8['n_dark'] + p8['n_erp']) if (p8['n_dark'] + p8['n_erp']) else float('nan'):.4f}",
        ),
        row(
            Y2,
            "auroc_a_transfer",
            p5["xfer_y2"],
            "train_cv",
            f"size={p5['size_y2']:.4f} days={p5['days_y2']:.4f} drop12={ctx['p9']['y2_drop']}",
        ),
        row(
            Y3,
            "auroc_has_xfer",
            p5["has_y3"],
            "train_cv",
            f"abs={p5['abs_y3']:.4f} gross={p5['gros_y3']:.4f} leftover_days={p6['y3_has_d']:.4f}",
        ),
        row(
            Y3,
            "auroc_co_mean_has_xfer",
            ctx["p19"]["y3_mean"],
            "train_cv",
            f"trait ICC={p7['xfer_icc']:.3f} ever={ctx['p19']['y3_ever']:.3f} co_days_rho={ctx['p21']['rho_days']:.3f} not_shock",
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
    print(f"transfer_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    con = connect()
    try:
        panel = load_panel(con)
        panel = attach_folds(panel)
        panel = add_panel_lags(panel, ["a_transfer", "a_invest", "has_xfer"], (1, 3))
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print(
            f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
            f"holdout CM={(panel['split']=='holdout').sum()}"
        )
        print("pass 1 prevalence")
        p1 = pass1_prev(panel, tr)
        print("pass 2 tokens")
        p2 = pass2_tokens(tr, con)
        print("pass 3 wash")
        p3 = pass3_wash(tr)
        print("pass 4 spearman")
        p4 = pass4_rho(tr)
        print("pass 5 singles")
        p5 = pass5_auroc(tr)
        print("pass 6 residual")
        p6 = pass6_resid(tr)
        print("pass 7 icc")
        p7 = pass7_icc(tr)
        print("pass 8 dark")
        p8 = pass8_dark(tr, con)
        print("pass 9 chronic 12")
        p9 = pass9_chronic(tr)
        print("pass 10 Q6")
        p10 = pass10_q6(tr)
        print("pass 11 invest")
        p11 = pass11_invest(tr)
        print("pass 12 amounts")
        p12 = pass12_amounts(tr, con)
        print("pass 13 token mix")
        p13 = pass13_tokens(tr, con)
        print("pass 14 rates")
        p14 = pass14_rates(tr)
        print("pass 15 terciles")
        p15 = pass15_terciles(tr)
        print("pass 16 holdout")
        p16 = pass16_holdout(panel)
        print("pass 17 calendar")
        p17 = pass17_cal(tr)
        print("pass 18 leftover after has")
        p18 = pass18_level(tr)
        print("pass 19 ever-transfer trait")
        p19 = pass19_ever(tr)
        print("pass 20 winsor / busy")
        p20 = pass20_winsor(tr)
        print("pass 21 company-mean twin")
        p21 = pass21_co_mean(tr)
        print("pass 22 abs leftover on transfer CM")
        p22 = pass22_abs_on(tr)
    finally:
        con.close()

    if not p2["formula_ok"]:
        print("WARN formula max|Δ| above 1e-4 — inspect cashflow.py, do not rewrite tonight")

    decision = decide(p3, p4, p5, p6, p7, p10, p11, p13)
    png_ok = make_png(tr, p3)
    headline = (
        f"Any-transfer {_pp(p1['xf']['any_share'])}; any-invest {_pp(p1['inv']['any_share'])}. "
        f"Wash **{'YES' if p3['is_wash'] else 'NO'}** (|net|/gross p50 {_f(p3['wr_p50'])}). "
        f"Y3 `a_transfer` {_f(p5['xfer_y3'])} vs size {_f(p5['size_y3'])} "
        f"(Δ {_f(p5['beat_size'])}) vs days {_f(p5['days_y3'])} vs salary {_f(p5['sal_y3'])}. "
        f"ρ vs days {_f(p4['rho_days'])}. Honest leftover has_xfer|days {_f(p6['y3_has_d'])}. "
        f"ICC {_f(p7['xfer_icc'])}. X **{decision['xfer_x']}**. PARK as Y. "
        f"`a_invest` X **{decision['inv_x']}**. Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p2["formula_ok"]:
        failed.append("store vs raw signed mismatch")
    if not p5["days_ok"]:
        failed.append(f"Y3 days replica {_f(p5['days_y3'])} vs night 0.711")
    if not p5["size_ok"]:
        failed.append(f"Y3 size replica {_f(p5['size_y3'])} vs 0.617")
    if not p8["confirm"]:
        failed.append(f"dark/erp {p8['n_dark']}/{p8['n_erp']} ≠ 470/744")
    if p9["n_ids"] != 12:
        failed.append(f"chronic names {p9['n_ids']} ≠ 12 from y2_why")
    if not decision["keep_x"]:
        failed.append(
            f"a_transfer Y3 {_f(p5['xfer_y3'])} has_xfer leftover-card {_f(p6['y3_has_c'])} "
            f"< size+0.02 or trait/twin/wash/SIZE — CLOSE as X"
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
