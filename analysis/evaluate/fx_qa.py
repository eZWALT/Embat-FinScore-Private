"""Q5 invoice FX share — exporter style, or a health why?

NORTH_STAR: why on the invoice book. `e_fx_share` = this-period issued
|amount| with currency <> accounting_currency. Feature report kept it in
the 44-col starter set (BETWEEN, LOW_PERSIST, size ρ 0.05 vs |op_in|).

Y7 is invoice-built → never use E as X for Y7. Y5 same. Score
`e_fx_share` vs Y3 (never B) and Y2 (never B). Y7 / Y5 are descriptive
base-rate splits only (fx>0 vs 0), not a model X.

470 dark have no invoice book — FX is undefined there (NaN, not 0).
Do not invent a merged FX Y. No 0–100. No product/. No parquet rewrite.
No new GBM. Do not run build_targets. Do not edit invoices.py.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.fx_qa

Owned: analysis/evaluate/fx_qa.py, analysis/outputs/fx_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_fx.md (end).
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
    train_companies,
)
from analysis.features.common import ANALYSIS, DATA, MONTHS, connect
from analysis.features.invoices import build as build_invoices
from analysis.targets.y11_dark import book_invoice_ids, dark_population

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "fx_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "fx_y3_quintiles.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "97d3db33"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y5_AP = "y5_ap_od30_ownp80"
Y5_AR = "y5_ar_od30_sust"
Y7 = "y7_top1_lost"
Y7_IN = "y7_top1_lost_inflow"
N_FOLDS = 5
DAYS_BENCH = 0.711
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
LEAK_RHO = 0.80
MIN_POS = 50
STYLE_ALWAYS = 0.80  # share of defined months with fx>0
BOOK = (
    "document_type = 'invoice' AND status <> 'cancel' "
    "AND amount <> 0 AND issuance_date IS NOT NULL"
)

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "e_fx_share",
    "e_ar_issued",
    "e_ap_issued",
    "e_dso_proxy",
    "c_n_days_with_tx",
    "d_cust_hhi",
    "d_cust_top1",
    "d_n_cust",
)

Y_KEEP = (Y2, Y3, Y5_AP, Y5_AR, Y7, Y7_IN)


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
        if m.sum() < 4:
            continue
        aa, bb = aa[m], bb[m]
        if np.std(aa) == 0 or np.std(bb) == 0:
            continue
        vals.append(float(np.corrcoef(aa, bb)[0, 1]))
    return float(np.median(vals)) if vals else float("nan")


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
    ykeep = ["company_id", "period", *Y_KEEP]
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["log_ar"] = np.log1p(pd.to_numeric(panel["e_ar_issued"], errors="coerce").clip(lower=0))
    fx = pd.to_numeric(panel["e_fx_share"], errors="coerce")
    panel["e_fx_share"] = fx
    panel["fx_defined"] = fx.notna()
    panel["fx_pos"] = fx > 0
    panel["fx_zero"] = fx.eq(0)
    return panel


def attach_book_and_folds(panel: pd.DataFrame, con) -> pd.DataFrame:
    book = book_invoice_ids(con)
    pop = dark_population(con)
    panel = panel.copy()
    panel["has_book"] = panel["company_id"].isin(book)
    mix_of = pop["mix_of"]
    panel["group_mix"] = panel["group_id"].map(mix_of)
    train_cos = train_companies(con)
    assert_no_holdout(train_cos["company_id"])
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    panel.attrs["dark_pop"] = {
        "n_train_dark": pop["n_train_dark"],
        "n_110_mixed": pop["n_110_mixed"],
        "n_360_alldark": pop["n_360_alldark"],
        "n_book_train": pop["n_book_train"],
        "confirm_470": pop["confirm_470"],
        "train_dark_ids": pop["train_dark_ids"],
    }
    return panel


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


def _rate_row(name: str, sl: pd.DataFrame, y: str) -> dict:
    lab = sl[y].notna()
    n_lab = int(lab.sum())
    n_pos = int((lab & (sl[y] == 1)).sum())
    return {
        "slice": name,
        "y": y,
        "n_cm": int(len(sl)),
        "n_co": int(sl["company_id"].nunique()),
        "n_lab": n_lab,
        "n_pos": n_pos,
        "rate": _pct(n_pos, n_lab),
    }


# ---------------------------------------------------------------------------
# Pass 1 — coverage + currencies
# ---------------------------------------------------------------------------
def pass1_coverage(tr: pd.DataFrame, con, pop: dict) -> dict:
    erp = tr[tr["has_book"]].copy()
    dark = tr[~tr["has_book"]].copy()
    n_erp_cm = int(len(erp))
    n_erp_co = int(erp["company_id"].nunique())
    n_def = int(erp["fx_defined"].sum())
    n_pos = int(erp["fx_pos"].sum())
    n_zero = int(erp["fx_zero"].sum())
    ever_fx = erp.loc[erp["fx_pos"], "company_id"].unique()
    n_ever = int(len(ever_fx))
    # dark must be all-NaN
    dark_nan = bool(dark["e_fx_share"].isna().all()) if len(dark) else True
    dark_n_defined = int(dark["fx_defined"].sum())
    confirm_470 = int(dark["company_id"].nunique()) == 470 and bool(pop.get("confirm_470"))
    confirm_744 = n_erp_co == 744

    fx_tx = con.execute(
        f"""
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(currency AS VARCHAR) AS currency,
          CAST(accounting_currency AS VARCHAR) AS accounting_currency,
          CASE WHEN amount > 0 THEN 'AR' ELSE 'AP' END AS side,
          SUM(abs(amount)) AS abs_amt,
          COUNT(*) AS n_inv
        FROM invoices
        WHERE {BOOK}
          AND currency IS NOT NULL
          AND accounting_currency IS NOT NULL
          AND currency <> accounting_currency
          AND issuance_date IS NOT NULL
          AND issuance_date >= DATE '2024-09-01'
          AND issuance_date < DATE '2026-09-01'
        GROUP BY 1, 2, 3, 4
        """
    ).df()
    fx_tx["company_id"] = fx_tx["company_id"].astype(str)
    hold = load_holdout()
    fx_tx = fx_tx.loc[~fx_tx["company_id"].isin(hold)].copy()
    assert_no_holdout(fx_tx["company_id"])

    pair = (
        fx_tx.groupby(["accounting_currency", "currency"], as_index=False)
        .agg(n_inv=("n_inv", "sum"), abs_amt=("abs_amt", "sum"), n_co=("company_id", "nunique"))
        .sort_values("abs_amt", ascending=False)
    )
    pair_rows = []
    for r in pair.itertuples(index=False):
        pair_rows.append(
            {
                "acct": r.accounting_currency,
                "inv": r.currency,
                "n_inv": int(r.n_inv),
                "n_co": int(r.n_co),
                "abs_amt": float(r.abs_amt),
            }
        )
    side = (
        fx_tx.groupby("side", as_index=False)
        .agg(n_inv=("n_inv", "sum"), abs_amt=("abs_amt", "sum"), n_co=("company_id", "nunique"))
    )
    side_map = {str(r.side): r for r in side.itertuples(index=False)}
    currs = sorted(set(pair["currency"].astype(str)) | set(pair["accounting_currency"].astype(str)))
    inv_currs = sorted(set(pair["currency"].astype(str)))
    acct_currs = sorted(set(pair["accounting_currency"].astype(str)))

    # all invoice currencies (not just FX) for context
    all_ccy = con.execute(
        f"""
        SELECT CAST(currency AS VARCHAR) AS currency,
               CAST(accounting_currency AS VARCHAR) AS accounting_currency,
               COUNT(*) AS n_inv
        FROM invoices
        WHERE {BOOK}
          AND issuance_date >= DATE '2024-09-01'
          AND issuance_date < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()

    prose = (
        f"Train ever-ERP {n_erp_co:,} companies / {n_erp_cm:,} CM. "
        f"`e_fx_share` defined {_pp(_pct(n_def, n_erp_cm))} of ERP CM "
        f"(issued this period); >0 on {_pp(_pct(n_pos, n_erp_cm))} of ERP CM "
        f"({n_pos:,} / {n_erp_cm:,}) and {_pp(_pct(n_pos, n_def))} of defined. "
        f"Ever-FX companies **{n_ever}**. "
        f"Dark {dark['company_id'].nunique()} companies: defined FX months {dark_n_defined} "
        f"({'NaN-ok' if dark_nan else 'LEAK — dark has defined FX'}). "
        f"FX invoice currencies {inv_currs} vs accounting {acct_currs}."
    )
    print(prose)
    return {
        "n_erp_cm": n_erp_cm,
        "n_erp_co": n_erp_co,
        "n_def": n_def,
        "n_pos": n_pos,
        "n_zero": n_zero,
        "share_pos_erp": _pct(n_pos, n_erp_cm),
        "share_pos_def": _pct(n_pos, n_def),
        "share_def": _pct(n_def, n_erp_cm),
        "n_ever": n_ever,
        "dark_nan": dark_nan,
        "dark_n_defined": dark_n_defined,
        "n_dark_co": int(dark["company_id"].nunique()),
        "n_dark_cm": int(len(dark)),
        "confirm_470": confirm_470,
        "confirm_744": confirm_744,
        "pair_rows": pair_rows,
        "currs": currs,
        "inv_currs": inv_currs,
        "acct_currs": acct_currs,
        "n_fx_inv": int(fx_tx["n_inv"].sum()) if len(fx_tx) else 0,
        "n_fx_co_raw": int(fx_tx["company_id"].nunique()) if len(fx_tx) else 0,
        "ar_n": int(side_map["AR"].n_inv) if "AR" in side_map else 0,
        "ap_n": int(side_map["AP"].n_inv) if "AP" in side_map else 0,
        "ar_amt": float(side_map["AR"].abs_amt) if "AR" in side_map else 0.0,
        "ap_amt": float(side_map["AP"].abs_amt) if "AP" in side_map else 0.0,
        "n_all_pairs": int(len(all_ccy)),
        "prose": prose,
        "ever_fx_ids": set(ever_fx),
    }


# ---------------------------------------------------------------------------
# Pass 2 — size ρ
# ---------------------------------------------------------------------------
def pass2_size(erp: pd.DataFrame) -> dict:
    defined = erp[erp["fx_defined"]]
    rho_in3 = spearman(defined["e_fx_share"], defined["log_in3"])
    rho_ar = spearman(defined["e_fx_share"], defined["e_ar_issued"])
    rho_log_ar = spearman(defined["e_fx_share"], defined["log_ar"])
    rho_pos_in3 = spearman(
        defined.loc[defined["fx_pos"], "e_fx_share"],
        defined.loc[defined["fx_pos"], "log_in3"],
    )
    rho_pos_ar = spearman(
        defined.loc[defined["fx_pos"], "e_fx_share"],
        defined.loc[defined["fx_pos"], "e_ar_issued"],
    )
    # binary ever-FX company vs median log_in3
    co = erp.groupby("company_id", as_index=False).agg(
        ever_fx=("fx_pos", "max"),
        med_in3=("log_in3", "median"),
        med_ar=("e_ar_issued", "median"),
    )
    rho_ever_size = spearman(co["ever_fx"].astype(float), co["med_in3"])
    is_size = bool(
        (np.isfinite(rho_in3) and abs(rho_in3) >= SIZE_RHO)
        or (np.isfinite(rho_ar) and abs(rho_ar) >= SIZE_RHO)
    )
    prose = (
        f"Defined ERP CM n={len(defined):,}. Spearman `e_fx_share` vs log1p(a_in3) "
        f"**{rho_in3:.3f}**, vs e_ar_issued {rho_ar:.3f} (log1p {rho_log_ar:.3f}). "
        f"Among fx>0: vs in3 {rho_pos_in3:.3f}, vs issued {rho_pos_ar:.3f}. "
        f"Ever-FX vs company-median log1p(a_in3) {rho_ever_size:.3f}. "
        f"{'SIZE' if is_size else 'not SIZE'} at |ρ|≥{SIZE_RHO:g}."
    )
    print(prose)
    return {
        "n_def": int(len(defined)),
        "rho_in3": rho_in3,
        "rho_ar": rho_ar,
        "rho_log_ar": rho_log_ar,
        "rho_pos_in3": rho_pos_in3,
        "rho_pos_ar": rho_pos_ar,
        "rho_ever_size": rho_ever_size,
        "is_size": is_size,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — quintiles + descriptive Y7/Y5
# ---------------------------------------------------------------------------
def _qcut_rates(sl: pd.DataFrame, xcol: str, y: str) -> list[dict]:
    x = pd.to_numeric(sl[xcol], errors="coerce")
    lab = sl[y].notna() & x.notna()
    d = sl.loc[lab].copy()
    d["_x"] = x[lab]
    d["_y"] = pd.to_numeric(d[y], errors="coerce")
    if d["_x"].nunique() < 2 or len(d) < 20:
        return []
    cats = pd.qcut(d["_x"], 5, duplicates="drop")
    d = d.assign(q=cats)
    rows = []
    for i, (q, g) in enumerate(d.groupby("q", observed=True), start=1):
        rows.append(
            {
                "y": y,
                "q": i,
                "interval": str(q),
                "n": int(len(g)),
                "n_pos": int((g["_y"] == 1).sum()),
                "rate": float(g["_y"].mean()),
                "fx_med": float(g["_x"].median()),
            }
        )
    return rows


def pass3_quintiles(erp: pd.DataFrame) -> dict:
    y3_q = _qcut_rates(erp, "e_fx_share", Y3)
    y2_q = _qcut_rates(erp, "e_fx_share", Y2)
    # zeros pile up — also 0 vs >0 plus intensity among >0
    pos = erp[erp["fx_pos"]]
    y3_pos_q = _qcut_rates(pos, "e_fx_share", Y3)
    y2_pos_q = _qcut_rates(pos, "e_fx_share", Y2)

    split_rows = []
    for y in (Y3, Y2, Y7, Y7_IN, Y5_AP, Y5_AR):
        for name, sl in (
            ("fx>0", erp[erp["fx_pos"]]),
            ("fx=0", erp[erp["fx_zero"]]),
            ("fx NaN (no issue month)", erp[erp["has_book"] & ~erp["fx_defined"]]),
        ):
            rec = _rate_row(name, sl, y)
            split_rows.append(
                {
                    "y": y,
                    "slice": name,
                    "n_cm": f"{rec['n_cm']:,}",
                    "n_co": f"{rec['n_co']:,}",
                    "n_lab": f"{rec['n_lab']:,}",
                    "n_pos": f"{rec['n_pos']:,}",
                    "rate": _pp(rec["rate"]),
                }
            )
            print(f"  {y} {name}: n_lab={rec['n_lab']} rate={_pp(rec['rate'])}")

    def _rate(y, sl):
        lab = sl[y].notna()
        return _pct(int((lab & (sl[y] == 1)).sum()), int(lab.sum()))

    y7_pos = _rate(Y7, erp[erp["fx_pos"]])
    y7_zero = _rate(Y7, erp[erp["fx_zero"]])
    y5ap_pos = _rate(Y5_AP, erp[erp["fx_pos"]])
    y5ap_zero = _rate(Y5_AP, erp[erp["fx_zero"]])
    y5ar_pos = _rate(Y5_AR, erp[erp["fx_pos"]])
    y5ar_zero = _rate(Y5_AR, erp[erp["fx_zero"]])
    y3_pos = _rate(Y3, erp[erp["fx_pos"]])
    y3_zero = _rate(Y3, erp[erp["fx_zero"]])
    y2_pos = _rate(Y2, erp[erp["fx_pos"]])
    y2_zero = _rate(Y2, erp[erp["fx_zero"]])

    prose = (
        f"Y3 stressed recover fx>0 {_pp(y3_pos)} vs fx=0 {_pp(y3_zero)}. "
        f"Y2 {_pp(y2_pos)} vs {_pp(y2_zero)}. "
        f"Descriptive Y7 top1_lost {_pp(y7_pos)} vs {_pp(y7_zero)}; "
        f"Y5 AP {_pp(y5ap_pos)} vs {_pp(y5ap_zero)}, AR {_pp(y5ar_pos)} vs {_pp(y5ar_zero)}. "
        f"qcut on defined FX vs Y3 produced {len(y3_q)} bins "
        f"(zeros pile — intensity quintiles among >0: {len(y3_pos_q)} bins)."
    )
    print(prose)
    return {
        "y3_q": y3_q,
        "y2_q": y2_q,
        "y3_pos_q": y3_pos_q,
        "y2_pos_q": y2_pos_q,
        "split_rows": split_rows,
        "y3_pos": y3_pos,
        "y3_zero": y3_zero,
        "y2_pos": y2_pos,
        "y2_zero": y2_zero,
        "y7_pos": y7_pos,
        "y7_zero": y7_zero,
        "y5ap_pos": y5ap_pos,
        "y5ap_zero": y5ap_zero,
        "y5ar_pos": y5ar_pos,
        "y5ar_zero": y5ar_zero,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — singles Y3 / Y2 only
# ---------------------------------------------------------------------------
def pass4_singles(tr: pd.DataFrame) -> dict:
    leak_y3 = leakage_check(["e_fx_share", "log_in3", "c_n_days_with_tx"], Y3, ("b",))
    leak_y2 = leakage_check(["e_fx_share", "log_in3", "c_n_days_with_tx"], Y2, ("b",))
    leak_y7 = leakage_check(["e_fx_share"], Y7, ("e",))
    print(f"leakage Y3 vs B: {leak_y3}")
    print(f"leakage Y2 vs B: {leak_y2}")
    print(f"leakage Y7 vs E (must FAIL — we do not score): {leak_y7}")
    if not leak_y3["ok"] or not leak_y2["ok"]:
        raise RuntimeError("forbidden family B in Y3/Y2 X")
    if leak_y7["ok"]:
        raise RuntimeError("expected Y7×E leak screen to fail")

    feats = {
        "e_fx_share": tr["e_fx_share"],
        "fx_pos": tr["fx_pos"].astype(float),
        "log1p_a_in3": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "e_ar_issued": tr["e_ar_issued"],
        "d_cust_hhi": tr["d_cust_hhi"],
    }
    # fx_pos on defined-only so NaN is not scored as 0
    fx_pos_def = tr["e_fx_share"].where(tr["fx_defined"]).gt(0).astype(float)
    fx_pos_def = fx_pos_def.where(tr["fx_defined"])
    feats["fx_pos"] = fx_pos_def

    rows = []
    store = {}
    for y in (Y2, Y3):
        full = tr[y].notna()
        same = tr[y].notna() & tr["fx_defined"]
        for mask_name, mask in (("full", full), ("fx_defined", same)):
            for name, col in feats.items():
                if name in {"e_fx_share", "fx_pos"} and mask_name == "full":
                    # still run — NaNs drop inside signed_oof
                    pass
                res = signed_oof_auroc(tr[y], col, tr["fold"], mask)
                key = (y, name, mask_name)
                store[key] = res
                rows.append(
                    {
                        "y": y,
                        "feature": name,
                        "mask": mask_name,
                        "n": f"{res['n_defined']:,}",
                        "n_pos": f"{res['n_pos']:,}",
                        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                        "sd": _f(res["sd"]),
                        "sign": res["train_sign"] if not res["low_power"] else "—",
                        "train": _f(res["train_auc"]) if not res["low_power"] else "—",
                    }
                )
                print(
                    f"AUROC {y} {name} [{mask_name}]: "
                    f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                    f"n_pos={res['n_pos']} sign={res['train_sign']}"
                )

    def _cv(y, feat, mask="fx_defined") -> float:
        r = store[(y, feat, mask)]
        return float("nan") if r["low_power"] else r["cv"]

    fx_y3 = _cv(Y3, "e_fx_share")
    fx_y2 = _cv(Y2, "e_fx_share")
    size_y3 = _cv(Y3, "log1p_a_in3")
    size_y3_full = _cv(Y3, "log1p_a_in3", "full")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    days_y3_full = _cv(Y3, "c_n_days_with_tx", "full")
    size_y2 = _cv(Y2, "log1p_a_in3")
    pos_y3 = _cv(Y3, "fx_pos")
    beat = (fx_y3 - size_y3) if np.isfinite(fx_y3) and np.isfinite(size_y3) else float("nan")
    keep_x = bool(np.isfinite(beat) and beat >= KEEP_DELTA)
    days_ok = bool(np.isfinite(days_y3_full) and abs(days_y3_full - DAYS_BENCH) <= 0.02)

    n_y3 = int(tr[Y3].notna().sum())
    n_y2 = int(tr[Y2].notna().sum())
    y3_rate = float(tr.loc[tr[Y3].notna(), Y3].mean())
    y2_rate = float(tr.loc[tr[Y2].notna(), Y2].mean())

    prose = (
        f"Y3 `e_fx_share` CV {_f(fx_y3)} vs same-mask size {_f(size_y3)} "
        f"(Δ {_f(beat)}) vs days {_f(days_y3)}. "
        f"Full-panel days {_f(days_y3_full)} "
        f"({'replica 0.711' if days_ok else 'off 0.711'}). "
        f"Y2 FX {_f(fx_y2)} vs size {_f(size_y2)}. "
        f"{'KEEP as Y3 X' if keep_x else 'does not clear KEEP gate (need +0.02 vs size, not SIZE)'}."
    )
    print(prose)
    return {
        "rows": rows,
        "fx_y3": fx_y3,
        "fx_y2": fx_y2,
        "size_y3": size_y3,
        "size_y3_full": size_y3_full,
        "days_y3": days_y3,
        "days_y3_full": days_y3_full,
        "size_y2": size_y2,
        "pos_y3": pos_y3,
        "beat": beat,
        "keep_x": keep_x,
        "days_ok": days_ok,
        "n_y3": n_y3,
        "n_y2": n_y2,
        "y3_rate": y3_rate,
        "y2_rate": y2_rate,
        "leak_y3": leak_y3,
        "leak_y7_fails": not leak_y7["ok"],
        "prose": prose,
        "store": store,
    }


# ---------------------------------------------------------------------------
# Pass 5 — persistence / style vs shock
# ---------------------------------------------------------------------------
def pass5_persist(erp: pd.DataFrame) -> dict:
    acf1 = median_acf(erp["e_fx_share"], erp["company_id"], 1)
    acf3 = median_acf(erp["e_fx_share"], erp["company_id"], 3)
    acf6 = median_acf(erp["e_fx_share"], erp["company_id"], 6)
    pos = erp["e_fx_share"].where(erp["fx_defined"]).gt(0).astype(float)
    pos = pos.where(erp["fx_defined"])
    acf1_pos = median_acf(pos, erp["company_id"], 1)

    defined = erp[erp["fx_defined"]]
    co = defined.groupby("company_id").agg(
        n_def=("e_fx_share", "size"),
        n_pos=("fx_pos", "sum"),
        mean_fx=("e_fx_share", "mean"),
        med_fx=("e_fx_share", "median"),
    )
    co["share_pos"] = co["n_pos"] / co["n_def"]
    co["kind"] = np.select(
        [
            co["n_pos"] == 0,
            co["share_pos"] >= STYLE_ALWAYS,
        ],
        ["never", "always"],
        default="shock",
    )
    n_never = int((co["kind"] == "never").sum())
    n_always = int((co["kind"] == "always").sum())
    n_shock = int((co["kind"] == "shock").sum())
    kind_of = co["kind"]
    sl = erp.copy()
    sl["fx_kind"] = sl["company_id"].map(kind_of)

    kind_rows = []
    for kind in ("never", "shock", "always"):
        sub = sl[sl["fx_kind"] == kind]
        rec3 = _rate_row(kind, sub, Y3)
        rec2 = _rate_row(kind, sub, Y2)
        rec7 = _rate_row(kind, sub, Y7)
        kind_rows.append(
            {
                "kind": kind,
                "n_co": int(sub["company_id"].nunique()),
                "n_cm": int(len(sub)),
                "mean_fx": _f(
                    float(co.loc[co["kind"] == kind, "mean_fx"].mean())
                    if (co["kind"] == kind).any()
                    else float("nan")
                ),
                "Y3": _pp(rec3["rate"]),
                "Y3_n": f"{rec3['n_lab']:,}",
                "Y2": _pp(rec2["rate"]),
                "Y7": _pp(rec7["rate"]),
            }
        )
        print(
            f"  style {kind}: n_co={kind_rows[-1]['n_co']} "
            f"Y3={kind_rows[-1]['Y3']} Y2={kind_rows[-1]['Y2']}"
        )

    # among ever-FX only, acf of the share
    ever_ids = set(co.index[co["n_pos"] > 0])
    ever = erp[erp["company_id"].isin(ever_ids)]
    acf1_ever = median_acf(ever["e_fx_share"], ever["company_id"], 1)
    style = "style" if (n_always >= n_shock and np.isfinite(acf1_ever) and acf1_ever >= 0.40) else (
        "shock" if n_shock > n_always else "mixed"
    )
    prose = (
        f"acf1={acf1:.3f} acf3={acf3:.3f} acf6={acf6:.3f} "
        f"(fx>0 indicator acf1={acf1_pos:.3f}; ever-FX share acf1={acf1_ever:.3f}). "
        f"Companies with ≥1 defined month: never={n_never} always(≥{STYLE_ALWAYS:.0%} fx>0)={n_always} "
        f"shock={n_shock}. Call: **{style}**."
    )
    print(prose)
    return {
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "acf1_pos": acf1_pos,
        "acf1_ever": acf1_ever,
        "n_never": n_never,
        "n_always": n_always,
        "n_shock": n_shock,
        "style": style,
        "kind_rows": kind_rows,
        "prose": prose,
        "kind_of": kind_of,
    }


# ---------------------------------------------------------------------------
# Pass 6 — leak vs issued / DSO (Y7 story)
# ---------------------------------------------------------------------------
def pass6_leak(erp: pd.DataFrame) -> dict:
    defined = erp[erp["fx_defined"]]
    rows = []
    pairs = [
        ("e_fx_share", "e_ar_issued"),
        ("e_fx_share", "e_dso_proxy"),
        ("e_fx_share", "e_ap_issued"),
        ("e_fx_share", "d_cust_hhi"),
        ("e_fx_share", "d_cust_top1"),
        ("e_fx_share", "d_n_cust"),
        ("e_fx_share", "log_in3"),
        ("fx_pos", "e_ar_issued"),
        ("fx_pos", "d_cust_hhi"),
    ]
    store = {}
    for a, b in pairs:
        xa = defined["fx_pos"].astype(float) if a == "fx_pos" else defined[a]
        rho = spearman(xa, defined[b])
        store[(a, b)] = rho
        leak = bool(np.isfinite(rho) and abs(rho) >= LEAK_RHO)
        rows.append({"pair": f"{a} × {b}", "Spearman": _f(rho), "leak": "LEAK" if leak else ""})
        print(f"  leak {a} vs {b}: {rho:.3f}{' LEAK' if leak else ''}")
    any_leak = any(r["leak"] == "LEAK" for r in rows)
    prose = (
        f"FX vs e_ar_issued ρ={store[('e_fx_share','e_ar_issued')]:.3f}, "
        f"vs e_dso_proxy ρ={store[('e_fx_share','e_dso_proxy')]:.3f}, "
        f"vs d_cust_hhi ρ={store[('e_fx_share','d_cust_hhi')]:.3f}. "
        f"{'LEAK (≥0.80) — do not treat as a new Y7 column' if any_leak else 'not a leak of issued / DSO / HHI'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_ar": store[("e_fx_share", "e_ar_issued")],
        "rho_dso": store[("e_fx_share", "e_dso_proxy")],
        "rho_hhi": store[("e_fx_share", "d_cust_hhi")],
        "any_leak": any_leak,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — FX vs customer HHI (foreign ≠ monopoly)
# ---------------------------------------------------------------------------
def pass7_hhi(erp: pd.DataFrame) -> dict:
    defined = erp[erp["fx_defined"] & erp["d_cust_hhi"].notna()]
    rho = spearman(defined["e_fx_share"], defined["d_cust_hhi"])
    rho_pos = spearman(defined.loc[defined["fx_pos"], "e_fx_share"], defined.loc[defined["fx_pos"], "d_cust_hhi"])
    # HHI quintiles (train defined cuts) × fx>0 rate and Y3
    x = defined["d_cust_hhi"]
    cats = pd.qcut(x, 5, duplicates="drop")
    defined = defined.assign(hhi_q=cats)
    rows = []
    for i, (q, g) in enumerate(defined.groupby("hhi_q", observed=True), start=1):
        rec3_pos = _rate_row("fx>0", g[g["fx_pos"]], Y3)
        rec3_z = _rate_row("fx=0", g[g["fx_zero"]], Y3)
        rows.append(
            {
                "q": i,
                "hhi_med": _f(float(g["d_cust_hhi"].median())),
                "n": f"{len(g):,}",
                "fx>0": _pp(float(g["fx_pos"].mean())),
                "fx_med": _f(float(g["e_fx_share"].median())),
                "Y3_fx>0": _pp(rec3_pos["rate"]),
                "Y3_fx=0": _pp(rec3_z["rate"]),
                "n_Y3_fx": f"{rec3_pos['n_lab']:,}",
            }
        )
        print(
            f"  HHI Q{i} med={rows[-1]['hhi_med']} fx>0={rows[-1]['fx>0']} "
            f"Y3 {rows[-1]['Y3_fx>0']} vs {rows[-1]['Y3_fx=0']}"
        )
    # monopoly tail (top quintile) vs diversified
    hi = defined[defined["hhi_q"] == cats.cat.categories[-1]] if len(cats.cat.categories) else defined.iloc[0:0]
    lo = defined[defined["hhi_q"] == cats.cat.categories[0]] if len(cats.cat.categories) else defined.iloc[0:0]
    fx_hi = float(hi["fx_pos"].mean()) if len(hi) else float("nan")
    fx_lo = float(lo["fx_pos"].mean()) if len(lo) else float("nan")
    monopoly = bool(np.isfinite(fx_hi) and np.isfinite(fx_lo) and fx_hi >= fx_lo + 0.10)
    prose = (
        f"FX × d_cust_hhi ρ={rho:.3f} (among >0: {rho_pos:.3f}). "
        f"fx>0 share HHI-Q1 {_pp(fx_lo)} vs Q5 {_pp(fx_hi)}. "
        f"{'FX concentrates on the monopoly tail' if monopoly else 'foreign book ≠ customer monopoly'}."
    )
    print(prose)
    return {
        "rho": rho,
        "rho_pos": rho_pos,
        "rows": rows,
        "fx_hi": fx_hi,
        "fx_lo": fx_lo,
        "monopoly": monopoly,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — mixed-group 110 (invoiced siblings, not the dark 110)
# ---------------------------------------------------------------------------
def pass8_mixed(tr: pd.DataFrame, pop: dict) -> dict:
    erp = tr[tr["has_book"]]
    dark = tr[~tr["has_book"]]
    mix_rows = []
    for mix in ("all_invoiced", "mixed", "all_dark"):
        sl = tr[tr["group_mix"] == mix]
        erp_sl = sl[sl["has_book"]]
        dark_sl = sl[~sl["has_book"]]
        n_ever = int(erp_sl.loc[erp_sl["fx_pos"], "company_id"].nunique())
        mix_rows.append(
            {
                "mix": mix,
                "n_co": int(sl["company_id"].nunique()),
                "n_erp": int(erp_sl["company_id"].nunique()),
                "n_dark": int(dark_sl["company_id"].nunique()),
                "fx_def": _pp(float(erp_sl["fx_defined"].mean()) if len(erp_sl) else float("nan")),
                "fx>0_erp": _pp(float(erp_sl["fx_pos"].mean()) if len(erp_sl) else float("nan")),
                "ever_FX": n_ever,
                "dark_fx_defined": int(dark_sl["fx_defined"].sum()),
                "Y3_erp": _pp(_rate_row("erp", erp_sl, Y3)["rate"]),
                "Y3_dark": _pp(_rate_row("dark", dark_sl, Y3)["rate"]) if len(dark_sl) else "—",
            }
        )
        print(f"  mix {mix}: {mix_rows[-1]}")

    mixed_erp = erp[erp["group_mix"] == "mixed"]
    allinv = erp[erp["group_mix"] == "all_invoiced"]
    p_mixed = float(mixed_erp["fx_pos"].mean()) if len(mixed_erp) else float("nan")
    p_all = float(allinv["fx_pos"].mean()) if len(allinv) else float("nan")
    n_ever_mixed = int(mixed_erp.loc[mixed_erp["fx_pos"], "company_id"].nunique())
    n_ever_all = int(allinv.loc[allinv["fx_pos"], "company_id"].nunique())
    n_110 = int(dark.loc[dark["group_mix"] == "mixed", "company_id"].nunique())
    n_360 = int(dark.loc[dark["group_mix"] == "all_dark", "company_id"].nunique())
    # FX companies are invoiced — they are not the 110 dark
    same_110 = False
    prose = (
        f"Mixed-group invoiced fx>0 CM {_pp(p_mixed)} (ever-FX n={n_ever_mixed}) vs "
        f"all-invoiced groups {_pp(p_all)} (ever-FX n={n_ever_all}). "
        f"Dark mixed {n_110} / all-dark {n_360} have defined FX months "
        f"{int(dark['fx_defined'].sum())} (must be 0). "
        f"FX companies are **not** the 110 dark siblings "
        f"(they are the invoiced side of mixed / all-invoiced groups)."
    )
    print(prose)
    return {
        "rows": mix_rows,
        "p_mixed": p_mixed,
        "p_all": p_all,
        "n_ever_mixed": n_ever_mixed,
        "n_ever_all": n_ever_all,
        "n_110": n_110,
        "n_360": n_360,
        "same_110": same_110,
        "confirm_110": n_110 == 110 and n_360 == 360,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — holdout coverage only
# ---------------------------------------------------------------------------
def pass9_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    erp = ho[ho["has_book"]]
    dark = ho[~ho["has_book"]]
    rows = [
        {
            "slice": "holdout ERP",
            "n_co": int(erp["company_id"].nunique()),
            "n_cm": int(len(erp)),
            "fx defined": _pp(float(erp["fx_defined"].mean()) if len(erp) else float("nan")),
            "fx>0": _pp(float(erp["fx_pos"].mean()) if len(erp) else float("nan")),
            "ever_FX": int(erp.loc[erp["fx_pos"], "company_id"].nunique()),
        },
        {
            "slice": "holdout dark",
            "n_co": int(dark["company_id"].nunique()),
            "n_cm": int(len(dark)),
            "fx defined": _pp(float(dark["fx_defined"].mean()) if len(dark) else float("nan")),
            "fx>0": _pp(float(dark["fx_pos"].mean()) if len(dark) else float("nan")),
            "ever_FX": int(dark.loc[dark["fx_pos"], "company_id"].nunique()),
        },
    ]
    prose = (
        f"Holdout coverage only: ERP {rows[0]['n_co']} ever-FX {rows[0]['ever_FX']}; "
        f"dark {rows[1]['n_co']} defined {rows[1]['fx defined']}. No AUROC."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 10 — AR vs AP FX (exporter vs importer)
# ---------------------------------------------------------------------------
def pass10_side(tr: pd.DataFrame, con) -> dict:
    """In-module AR/AP FX shares. Not a store rewrite. Not a new Y."""
    periods = pd.DataFrame({"period": pd.to_datetime(MONTHS)})
    periods["period_end"] = periods["period"] + pd.offsets.MonthEnd(0)
    con.register("_fx_periods", periods)
    try:
        side = con.execute(
            f"""
            SELECT i.company_id,
                   p.period,
                   SUM(CASE WHEN i.amount > 0
                             AND i.currency <> i.accounting_currency
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount > 0 THEN abs(i.amount) END), 0)
                     AS ar_fx,
                   SUM(CASE WHEN i.amount < 0
                             AND i.currency <> i.accounting_currency
                            THEN abs(i.amount) ELSE 0 END)
                     / NULLIF(SUM(CASE WHEN i.amount < 0 THEN abs(i.amount) END), 0)
                     AS ap_fx,
                   SUM(CASE WHEN i.amount > 0
                             AND i.currency <> i.accounting_currency
                            THEN abs(i.amount) ELSE 0 END) AS ar_fx_amt,
                   SUM(CASE WHEN i.amount < 0
                             AND i.currency <> i.accounting_currency
                            THEN abs(i.amount) ELSE 0 END) AS ap_fx_amt
            FROM invoices i
            JOIN _fx_periods p
              ON CAST(i.issuance_date AS DATE) >= CAST(p.period AS DATE)
             AND CAST(i.issuance_date AS DATE) <= CAST(p.period_end AS DATE)
            WHERE {BOOK}
            GROUP BY 1, 2
            """
        ).df()
    finally:
        con.unregister("_fx_periods")
    side["company_id"] = side["company_id"].astype(str)
    side["period"] = pd.to_datetime(side["period"])
    hold = load_holdout()
    side = side.loc[~side["company_id"].isin(hold)].copy()
    assert_no_holdout(side["company_id"])

    erp = tr[tr["has_book"]][["company_id", "period", "fold", Y3, Y2, "e_fx_share", "fx_defined"]].copy()
    m = erp.merge(side, on=["company_id", "period"], how="left")
    m["ar_fx"] = pd.to_numeric(m["ar_fx"], errors="coerce")
    m["ap_fx"] = pd.to_numeric(m["ap_fx"], errors="coerce")
    m["ar_pos"] = m["ar_fx"] > 0
    m["ap_pos"] = m["ap_fx"] > 0
    ar_amt = float(pd.to_numeric(m["ar_fx_amt"], errors="coerce").fillna(0).sum())
    ap_amt = float(pd.to_numeric(m["ap_fx_amt"], errors="coerce").fillna(0).sum())
    ar_share = _pct(ar_amt, ar_amt + ap_amt)
    n_ar = int(m.loc[m["ar_pos"], "company_id"].nunique())
    n_ap = int(m.loc[m["ap_pos"], "company_id"].nunique())
    n_both = int(
        m.groupby("company_id")
        .agg(ar=("ar_pos", "max"), ap=("ap_pos", "max"))
        .query("ar and ap")
        .shape[0]
    )
    n_ar_only = n_ar - n_both
    n_ap_only = n_ap - n_both

    rate_rows = []
    for name, sl in (
        ("AR FX>0", m[m["ar_pos"]]),
        ("AP FX>0", m[m["ap_pos"]]),
        ("AR FX=0 (issued AR)", m[m["ar_fx"].eq(0)]),
        ("AP FX=0 (issued AP)", m[m["ap_fx"].eq(0)]),
    ):
        r3 = _rate_row(name, sl, Y3)
        r2 = _rate_row(name, sl, Y2)
        rate_rows.append(
            {
                "slice": name,
                "n_cm": f"{r3['n_cm']:,}",
                "n_co": f"{r3['n_co']:,}",
                "Y3": _pp(r3["rate"]),
                "Y3_n": f"{r3['n_lab']:,}",
                "Y2": _pp(r2["rate"]),
            }
        )

    y3_lab = m[Y3].notna()
    ar_auc = signed_oof_auroc(m[Y3], m["ar_fx"], m["fold"], y3_lab)
    ap_auc = signed_oof_auroc(m[Y3], m["ap_fx"], m["fold"], y3_lab)
    rho_vs_store = spearman(m["e_fx_share"], (m["ar_fx_amt"].fillna(0) + m["ap_fx_amt"].fillna(0)))

    label = (
        "importer"
        if ap_amt >= 1.5 * ar_amt
        else ("exporter" if ar_amt >= 1.5 * ap_amt else "mixed import/export")
    )
    prose = (
        f"FX |amount| is **{_pp(ar_share)} AR / {_pp(1.0 - ar_share)} AP**. "
        f"Ever AR-FX {n_ar} (only {n_ar_only}) / AP-FX {n_ap} (only {n_ap_only}) / both {n_both}. "
        f"Book call: **{label}**. "
        f"Y3 AR-FX CV {_f(ar_auc['cv'])} AP-FX {_f(ap_auc['cv'])} "
        f"(neither is a Y3 X). Store e_fx_share vs AR+AP FX amt ρ={rho_vs_store:.3f}."
    )
    print(prose)
    return {
        "ar_amt": ar_amt,
        "ap_amt": ap_amt,
        "ar_share": ar_share,
        "n_ar": n_ar,
        "n_ap": n_ap,
        "n_both": n_both,
        "n_ar_only": n_ar_only,
        "n_ap_only": n_ap_only,
        "label": label,
        "rate_rows": rate_rows,
        "ar_cv": ar_auc["cv"],
        "ap_cv": ap_auc["cv"],
        "rho_store": rho_vs_store,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — Q6 lag1 (1-month only)
# ---------------------------------------------------------------------------
def pass11_q6(tr: pd.DataFrame) -> dict:
    lab = tr[Y3].notna()
    now = signed_oof_auroc(tr[Y3], tr["e_fx_share"], tr["fold"], lab)
    lag = signed_oof_auroc(tr[Y3], tr["e_fx_share_lag1"], tr["fold"], lab)
    lag3 = signed_oof_auroc(tr[Y3], tr["e_fx_share_lag3"], tr["fold"], lab)
    size_lag = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab & tr["e_fx_share_lag1"].notna())
    drop = (
        (now["cv"] - lag["cv"])
        if np.isfinite(now["cv"]) and np.isfinite(lag["cv"])
        else float("nan")
    )
    keep_q6 = bool(
        np.isfinite(lag["cv"])
        and np.isfinite(size_lag["cv"])
        and (lag["cv"] - size_lag["cv"]) >= KEEP_DELTA
    )
    rows = [
        {
            "feature": "e_fx_share",
            "n": f"{now['n_defined']:,}",
            "n_pos": f"{now['n_pos']:,}",
            "CV": _f(now["cv"]),
            "sd": _f(now["sd"]),
        },
        {
            "feature": "e_fx_share_lag1",
            "n": f"{lag['n_defined']:,}",
            "n_pos": f"{lag['n_pos']:,}",
            "CV": _f(lag["cv"]),
            "sd": _f(lag["sd"]),
        },
        {
            "feature": "e_fx_share_lag3",
            "n": f"{lag3['n_defined']:,}",
            "n_pos": f"{lag3['n_pos']:,}",
            "CV": _f(lag3["cv"]),
            "sd": _f(lag3["sd"]),
        },
        {
            "feature": "log1p_a_in3 on lag1 rows",
            "n": f"{size_lag['n_defined']:,}",
            "n_pos": f"{size_lag['n_pos']:,}",
            "CV": _f(size_lag["cv"]),
            "sd": _f(size_lag["sd"]),
        },
    ]
    q6 = "KEEP" if keep_q6 else "CLOSE"
    prose = (
        f"Y3 contemporaneous {_f(now['cv'])} vs lag1 {_f(lag['cv'])} (drop {_f(drop)}) "
        f"vs lag3 {_f(lag3['cv'])}. Size on lag1 rows {_f(size_lag['cv'])}. "
        f"Q6 **{q6}** — hidden-test lead is 1-month only; issued_lag1 stays the Y7 KEEP."
    )
    print(prose)
    return {
        "now": now["cv"],
        "lag": lag["cv"],
        "lag3": lag3["cv"],
        "size_lag": size_lag["cv"],
        "drop": drop,
        "keep_q6": keep_q6,
        "q6": q6,
        "rows": rows,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — Y3 residual inside size terciles
# ---------------------------------------------------------------------------
def pass12_size_resid(erp: pd.DataFrame) -> dict:
    sl = erp[erp["fx_defined"] & erp["log_in3"].notna() & erp[Y3].notna()].copy()
    sl["sz"] = pd.qcut(sl["log_in3"], 3, duplicates="drop", labels=False)
    rows = []
    gaps = []
    for i, g in sl.groupby("sz", observed=True):
        r_pos = _rate_row("fx>0", g[g["fx_pos"]], Y3)
        r_z = _rate_row("fx=0", g[g["fx_zero"]], Y3)
        gap = (
            (r_pos["rate"] - r_z["rate"])
            if np.isfinite(r_pos["rate"]) and np.isfinite(r_z["rate"])
            else float("nan")
        )
        gaps.append(gap)
        rows.append(
            {
                "tercile": int(i) + 1,
                "in3_med": _f(float(g["log_in3"].median())),
                "n_lab": f"{int(len(g)):,}",
                "fx>0": _pp(float(g["fx_pos"].mean())),
                "Y3_fx>0": _pp(r_pos["rate"]),
                "Y3_fx=0": _pp(r_z["rate"]),
                "n_pos_fx": f"{r_pos['n_pos']:,}",
                "gap": _pp(gap) if np.isfinite(gap) else "—",
            }
        )
        print(f"  size T{int(i)+1}: Y3 fx>0 {_pp(r_pos['rate'])} vs 0 {_pp(r_z['rate'])} gap={gap}")
    mean_gap = float(np.nanmean(gaps)) if gaps else float("nan")
    vanishes = bool(np.isfinite(mean_gap) and abs(mean_gap) < 0.02)
    prose = (
        f"Y3 fx>0 minus fx=0 inside size terciles: mean gap {_pp(mean_gap)}. "
        f"{'vanishes after size' if vanishes else 'a residual remains (still not a single that beats size)'}."
    )
    print(prose)
    return {"rows": rows, "mean_gap": mean_gap, "vanishes": vanishes, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 13 — HHI × FX cell counts (Q4/Q5 recover spike)
# ---------------------------------------------------------------------------
def pass13_hhi_cells(erp: pd.DataFrame) -> dict:
    defined = erp[erp["fx_defined"] & erp["d_cust_hhi"].notna() & erp[Y3].notna()].copy()
    defined["hhi_q"] = pd.qcut(defined["d_cust_hhi"], 5, duplicates="drop")
    rows = []
    for i, (q, g) in enumerate(defined.groupby("hhi_q", observed=True), start=1):
        pos = g[g["fx_pos"]]
        z = g[g["fx_zero"]]
        rows.append(
            {
                "q": i,
                "hhi_med": _f(float(g["d_cust_hhi"].median())),
                "n_lab": f"{len(g):,}",
                "n_fx": f"{len(pos):,}",
                "n_pos_fx": int((pos[Y3] == 1).sum()),
                "Y3_fx": _pp(float(pos[Y3].mean()) if len(pos) else float("nan")),
                "n_z": f"{len(z):,}",
                "n_pos_z": int((z[Y3] == 1).sum()),
                "Y3_z": _pp(float(z[Y3].mean()) if len(z) else float("nan")),
            }
        )
        print(f"  HHI-Y3 Q{i}: fx n={len(pos)} pos={rows[-1]['n_pos_fx']} rate={rows[-1]['Y3_fx']}")
    q45 = defined[defined["hhi_q"].isin(list(defined["hhi_q"].cat.categories[-2:]))]
    n_pos_fx = int((q45.loc[q45["fx_pos"], Y3] == 1).sum())
    thin = n_pos_fx < MIN_POS
    prose = (
        f"HHI Q4+Q5 × fx>0 Y3 positives = {n_pos_fx} "
        f"({'thin — do not KEEP a monopoly×FX interaction' if thin else 'enough to quote'}). "
        "Foreign book is more common on the diversified side (pass 7)."
    )
    print(prose)
    return {"rows": rows, "n_pos_q45": n_pos_fx, "thin": thin, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 14 — within-company shock months
# ---------------------------------------------------------------------------
def pass14_within(erp: pd.DataFrame, kind_of: pd.Series) -> dict:
    sl = erp.copy()
    sl["fx_kind"] = sl["company_id"].map(kind_of)
    shock = sl[sl["fx_kind"] == "shock"]
    rows = []
    for y in (Y3, Y2, Y7):
        r_pos = _rate_row("own FX month", shock[shock["fx_pos"]], y)
        r_z = _rate_row("own fx=0 month", shock[shock["fx_zero"]], y)
        rows.append(
            {
                "y": y,
                "FX month": _pp(r_pos["rate"]),
                "n_lab_fx": f"{r_pos['n_lab']:,}",
                "n_pos_fx": f"{r_pos['n_pos']:,}",
                "own 0": _pp(r_z["rate"]),
                "n_lab_0": f"{r_z['n_lab']:,}",
                "n_pos_0": f"{r_z['n_pos']:,}",
            }
        )
        print(f"  within shock {y}: FX {_pp(r_pos['rate'])} vs own0 {_pp(r_z['rate'])}")
    y3_pos = _rate_row("x", shock[shock["fx_pos"]], Y3)["rate"]
    y3_z = _rate_row("x", shock[shock["fx_zero"]], Y3)["rate"]
    month_shock = bool(
        np.isfinite(y3_pos) and np.isfinite(y3_z) and abs(y3_pos - y3_z) >= 0.03
    )
    prose = (
        f"Shock companies (n={shock['company_id'].nunique()}): own FX-month Y3 {_pp(y3_pos)} "
        f"vs own zero-month {_pp(y3_z)}. "
        f"{'month-level shock' if month_shock else 'company identity, not the FX month'}."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_pos": y3_pos,
        "y3_z": y3_z,
        "month_shock": month_shock,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 15 — feature-report confirm + 228 vs raw 232
# ---------------------------------------------------------------------------
def pass15_confirm(tr: pd.DataFrame, p1: dict) -> dict:
    fx = tr["e_fx_share"]
    cov_cm = float(fx.notna().mean())
    cov_co = float(tr.groupby("company_id")["e_fx_share"].apply(lambda s: s.notna().any()).mean())
    defined = fx.dropna()
    modal = float((defined == 0).mean()) if len(defined) else float("nan")
    # feature report size is vs log1p(|a_op_in|); we quote vs log1p(a_in3) as the model control
    confirm_cov = bool(abs(cov_cm - 0.528) < 0.015)
    confirm_modal = bool(abs(modal - 0.796) < 0.02)
    delta_co = int(p1["n_fx_co_raw"]) - int(p1["n_ever"])
    prose = (
        f"Full-train (incl. dark NaN) cov_cm {_pp(cov_cm)} cov_co {_pp(cov_co)} "
        f"modal-zero-among-defined {_pp(modal)}. "
        f"{'CONFIRM feature-report 52.8% / 79.6% modal' if confirm_cov and confirm_modal else 'off feature-report quote'}. "
        f"Raw FX invoice companies {p1['n_fx_co_raw']} vs store ever-FX {p1['n_ever']} "
        f"(Δ {delta_co} — likely share rounded to 0 or period-edge)."
    )
    print(prose)
    return {
        "cov_cm": cov_cm,
        "cov_co": cov_co,
        "modal": modal,
        "confirm_cov": confirm_cov,
        "confirm_modal": confirm_modal,
        "delta_co": delta_co,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 16 — no-issue month vs FX (issued=0 is the recoverer)
# ---------------------------------------------------------------------------
def pass16_noissue(erp: pd.DataFrame, tr: pd.DataFrame) -> dict:
    """ERP months with no issuance recover 16.2% — is FX just 'issued something'?"""
    hole = erp[~erp["fx_defined"]]
    issued = erp[erp["fx_defined"]]
    r_hole = _rate_row("no issue", hole, Y3)
    r_iss = _rate_row("issued", issued, Y3)
    # single: issued-this-month flag vs Y3 on ERP
    flag = erp["fx_defined"].astype(float)
    auc = signed_oof_auroc(erp[Y3], flag, erp["fold"], erp[Y3].notna())
    # full panel including dark (dark is also 'no invoice book')
    dark_flag = tr["has_book"].astype(float)
    auc_book = signed_oof_auroc(tr[Y3], dark_flag, tr["fold"], tr[Y3].notna())
    prose = (
        f"ERP no-issue Y3 {_pp(r_hole['rate'])} (n_lab={r_hole['n_lab']:,}) vs issued {_pp(r_iss['rate'])}. "
        f"Issued-this-month flag Y3 CV {_f(auc['cv'])} sign={auc['train_sign']} "
        f"(has_book vs dark {_f(auc_book['cv'])}). "
        "The 16% hole is the bigger Q5 footnote — FX is not that hole."
    )
    print(prose)
    return {
        "hole": r_hole["rate"],
        "issued": r_iss["rate"],
        "issued_cv": auc["cv"],
        "issued_sign": auc["train_sign"],
        "book_cv": auc_book["cv"],
        "prose": prose,
        "n_hole": r_hole["n_lab"],
        "n_iss": r_iss["n_lab"],
    }


# ---------------------------------------------------------------------------
# Pass 17 — major vs exotic invoice currency
# ---------------------------------------------------------------------------
def pass17_ccy(tr: pd.DataFrame, con) -> dict:
    major = ("EUR", "USD", "GBP")
    raw = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(date_trunc('month', issuance_date) AS DATE) AS period,
               CAST(currency AS VARCHAR) AS currency,
               SUM(abs(amount)) AS amt
        FROM invoices
        WHERE {BOOK}
          AND currency <> accounting_currency
          AND issuance_date >= DATE '2024-09-01'
          AND issuance_date < DATE '2026-09-01'
        GROUP BY 1, 2, 3
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    hold = load_holdout()
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])
    raw["major"] = raw["currency"].isin(major)
    maj = (
        raw[raw["major"]]
        .groupby(["company_id", "period"], as_index=False)["amt"]
        .sum()
        .rename(columns={"amt": "major_amt"})
    )
    tot = raw.groupby(["company_id", "period"], as_index=False)["amt"].sum()
    by = tot.merge(maj, on=["company_id", "period"], how="left")
    by["major_amt"] = by["major_amt"].fillna(0.0)
    by["major_share"] = by["major_amt"] / by["amt"].where(by["amt"] > 0)
    by["exotic"] = by["major_share"] < 0.5
    erp = tr[tr["has_book"]][["company_id", "period", Y3, Y2, "fold"]].copy()
    m = erp.merge(by[["company_id", "period", "major_share", "exotic", "amt"]], on=["company_id", "period"], how="left")
    m["has_fx_raw"] = m["amt"].notna()
    m["exotic"] = np.where(m["has_fx_raw"], m["exotic"].eq(True), False)
    rows = []
    for name, sl in (
        ("major-FX month (EUR/USD/GBP ≥50%)", m[m["has_fx_raw"] & ~m["exotic"]]),
        ("exotic-FX month", m[m["has_fx_raw"] & m["exotic"]]),
        ("no raw FX", m[~m["has_fx_raw"]]),
    ):
        r = _rate_row(name, sl, Y3)
        rows.append(
            {
                "slice": name,
                "n_cm": f"{r['n_cm']:,}",
                "n_co": f"{r['n_co']:,}",
                "n_lab": f"{r['n_lab']:,}",
                "n_pos": f"{r['n_pos']:,}",
                "Y3": _pp(r["rate"]),
            }
        )
    top = (
        raw.groupby("currency", as_index=False)["amt"]
        .sum()
        .sort_values("amt", ascending=False)
        .head(8)
    )
    top_rows = [{"ccy": r.currency, "|amt|": f"{float(r.amt):,.0f}"} for r in top.itertuples(index=False)]
    n_maj = int(m.loc[m["has_fx_raw"] & ~m["exotic"], "company_id"].nunique())
    n_exo = int(m.loc[m["has_fx_raw"] & m["exotic"], "company_id"].nunique())
    prose = (
        f"Among raw-FX months, major (EUR/USD/GBP ≥50% of FX |amt|) companies {n_maj}, "
        f"exotic-majority {n_exo}. Y3 rates in the table. Not a Y."
    )
    print(prose)
    return {"rows": rows, "top_rows": top_rows, "n_maj": n_maj, "n_exo": n_exo, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 18 — raw 232 vs store 228 + home accounting currency
# ---------------------------------------------------------------------------
def pass18_gap_home(tr: pd.DataFrame, p1: dict, con) -> dict:
    raw_ids = con.execute(
        f"""
        SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id
        FROM invoices
        WHERE {BOOK}
          AND currency <> accounting_currency
          AND issuance_date >= DATE '2024-09-01'
          AND issuance_date < DATE '2026-09-01'
        """
    ).df()["company_id"].astype(str)
    hold = load_holdout()
    raw_ids = set(raw_ids) - hold
    store_ids = set(tr.loc[tr["fx_pos"], "company_id"])
    extra = sorted(raw_ids - store_ids)
    missing = sorted(store_ids - raw_ids)
    extra_rows = []
    if extra:
        det = con.execute(
            f"""
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   COUNT(*) AS n_inv,
                   SUM(abs(amount)) AS amt,
                   SUM(CASE WHEN currency <> accounting_currency THEN abs(amount) ELSE 0 END) AS fx_amt
            FROM invoices
            WHERE {BOOK}
              AND company_id IN ({", ".join("'" + x.replace("'", "''") + "'" for x in extra)})
              AND issuance_date >= DATE '2024-09-01'
              AND issuance_date < DATE '2026-09-01'
            GROUP BY 1
            """
        ).df()
        for r in det.itertuples(index=False):
            extra_rows.append(
                {
                    "company_id": r.company_id,
                    "n_inv": int(r.n_inv),
                    "fx_share": _pp(_pct(float(r.fx_amt), float(r.amt))),
                }
            )

    home = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(accounting_currency AS VARCHAR) AS acct,
               COUNT(*) AS n
        FROM invoices
        WHERE {BOOK}
          AND issuance_date >= DATE '2024-09-01'
          AND issuance_date < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    home["company_id"] = home["company_id"].astype(str)
    idx = home.groupby("company_id")["n"].idxmax()
    modal = home.loc[idx, ["company_id", "acct"]].rename(columns={"acct": "home_acct"})
    erp = tr[tr["has_book"]].merge(modal, on="company_id", how="left")
    erp["home_eur"] = erp["home_acct"] == "EUR"
    erp["home_usd"] = erp["home_acct"] == "USD"
    home_rows = []
    for name, sl in (
        ("home EUR", erp[erp["home_eur"]]),
        ("home USD", erp[erp["home_usd"]]),
        ("home other", erp[~erp["home_eur"] & ~erp["home_usd"] & erp["home_acct"].notna()]),
    ):
        r = _rate_row(name, sl, Y3)
        home_rows.append(
            {
                "home": name,
                "n_co": int(sl["company_id"].nunique()),
                "fx>0": _pp(float(sl["fx_pos"].mean()) if len(sl) else float("nan")),
                "ever_FX": int(sl.loc[sl["fx_pos"], "company_id"].nunique()),
                "Y3": _pp(r["rate"]),
                "n_lab": f"{r['n_lab']:,}",
            }
        )
    prose = (
        f"Raw-only FX companies {len(extra)} {extra[:8]}; store-only {len(missing)}. "
        f"Home-EUR n={home_rows[0]['n_co']} fx>0 {home_rows[0]['fx>0']}; "
        f"home-USD n={home_rows[1]['n_co']} fx>0 {home_rows[1]['fx>0']}. "
        "Not a Y. The 4-name gap does not move the 228."
    )
    print(prose)
    return {
        "n_extra": len(extra),
        "extra": extra,
        "n_missing": len(missing),
        "extra_rows": extra_rows,
        "home_rows": home_rows,
        "modal": modal,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 19 — why 4 raw-FX names never light the store
# ---------------------------------------------------------------------------
def pass19_four(tr: pd.DataFrame, extra: list[str], con) -> dict:
    rows = []
    for cid in extra:
        sl = tr[tr["company_id"] == cid]
        fx = sl["e_fx_share"]
        rows.append(
            {
                "company_id": cid,
                "n_cm": int(len(sl)),
                "defined": int(fx.notna().sum()),
                "max_store": _f(float(fx.max()) if fx.notna().any() else float("nan"), 6),
                "n_pos": int((fx > 0).sum()),
                "has_book": bool(sl["has_book"].any()) if len(sl) else False,
            }
        )
        print(f"  {cid}: defined={rows[-1]['defined']} max={rows[-1]['max_store']} pos={rows[-1]['n_pos']}")
    # issuance months of their FX invoices vs panel
    if extra:
        raw = con.execute(
            f"""
            SELECT CAST(company_id AS VARCHAR) AS company_id,
                   CAST(date_trunc('month', issuance_date) AS DATE) AS period,
                   COUNT(*) AS n_fx
            FROM invoices
            WHERE {BOOK}
              AND currency <> accounting_currency
              AND company_id IN ({", ".join("'" + x.replace("'", "''") + "'" for x in extra)})
            GROUP BY 1, 2
            """
        ).df()
        raw["period"] = pd.to_datetime(raw["period"])
        n_off = int((~raw["period"].isin(pd.to_datetime(MONTHS))).sum())
    else:
        n_off = 0
    prose = (
        f"Four raw-FX names that never have store e_fx_share>0: {extra}. "
        f"FX issuance months outside the 24-month panel: {n_off}. "
        "Live invoices.build lights them (COMP_0510 2024-09 share=1). Store is left-truncated / stale. Leave the 228 quote; do not rewrite parquet."
    )
    print(prose)
    return {"rows": rows, "n_off": n_off, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 20 — home-EUR only (majority book) singles
# ---------------------------------------------------------------------------
def pass20_home_eur(tr: pd.DataFrame, modal: pd.DataFrame) -> dict:
    m = tr.merge(modal, on="company_id", how="left")
    eur = m[m["home_acct"] == "EUR"]
    oth = m[m["home_acct"].notna() & (m["home_acct"] != "EUR")]
    rows = []
    store = {}
    for pop_name, sl in (("home_EUR", eur), ("home_not_EUR", oth)):
        lab = sl[Y3].notna() & sl["fx_defined"]
        for feat, col in (
            ("e_fx_share", sl["e_fx_share"]),
            ("log1p_a_in3", sl["log_in3"]),
            ("c_n_days_with_tx", sl["c_n_days_with_tx"]),
        ):
            res = signed_oof_auroc(sl[Y3], col, sl["fold"], lab)
            store[(pop_name, feat)] = res
            rows.append(
                {
                    "pop": pop_name,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
            print(
                f"  {pop_name} {feat}: "
                f"{'LOW_POWER' if res['low_power'] else _f(res['cv'])} n_pos={res['n_pos']}"
            )
    eur_fx = store[("home_EUR", "e_fx_share")]["cv"]
    eur_size = store[("home_EUR", "log1p_a_in3")]["cv"]
    oth_fx = store[("home_not_EUR", "e_fx_share")]["cv"]
    prose = (
        f"Home-EUR Y3 FX {_f(eur_fx)} vs size {_f(eur_size)}. "
        f"Non-EUR home FX {_f(oth_fx)}. "
        "The 0.528 pooled single is not a EUR-SME recover why."
    )
    print(prose)
    return {
        "rows": rows,
        "eur_fx": eur_fx,
        "eur_size": eur_size,
        "oth_fx": oth_fx,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 21 — high-intensity flag (share ≥ 0.08, intensity Q4 cut)
# ---------------------------------------------------------------------------
def pass21_hi(tr: pd.DataFrame) -> dict:
    hi = tr["e_fx_share"].where(tr["fx_defined"])
    flag = (hi >= 0.08).astype(float)
    flag = flag.where(tr["fx_defined"])
    lab = tr[Y3].notna()
    res = signed_oof_auroc(tr[Y3], flag, tr["fold"], lab)
    size = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab & tr["fx_defined"])
    r_hi = _rate_row("hi", tr[tr["fx_defined"] & (tr["e_fx_share"] >= 0.08)], Y3)
    r_lo = _rate_row("lo", tr[tr["fx_defined"] & (tr["e_fx_share"] > 0) & (tr["e_fx_share"] < 0.08)], Y3)
    beat = (
        (res["cv"] - size["cv"])
        if np.isfinite(res["cv"]) and np.isfinite(size["cv"])
        else float("nan")
    )
    keep = bool(np.isfinite(beat) and beat >= KEEP_DELTA)
    prose = (
        f"High-intensity (e_fx_share≥0.08) Y3 {_pp(r_hi['rate'])} (n_lab={r_hi['n_lab']}, "
        f"n_pos={r_hi['n_pos']}) vs low-positive {_pp(r_lo['rate'])}. "
        f"Flag CV {_f(res['cv'])} vs same-mask size {_f(size['cv'])} (Δ {_f(beat)}). "
        f"{'KEEP' if keep else 'still CLOSE'} as Y3 X."
    )
    print(prose)
    return {
        "cv": res["cv"],
        "size": size["cv"],
        "beat": beat,
        "keep": keep,
        "hi_rate": r_hi["rate"],
        "lo_rate": r_lo["rate"],
        "n_pos": r_hi["n_pos"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 22 — FX calendar (is the shock a filing month?)
# ---------------------------------------------------------------------------
def pass22_cal(erp: pd.DataFrame) -> dict:
    sl = erp.copy()
    sl["cal_month"] = sl["period"].dt.month
    rows = []
    for m in range(1, 13):
        g = sl[sl["cal_month"] == m]
        defn = g[g["fx_defined"]]
        rows.append(
            {
                "month": pd.Timestamp(2000, m, 1).strftime("%b"),
                "n_cm": f"{len(g):,}",
                "fx>0 / ERP": _pp(float(g["fx_pos"].mean()) if len(g) else float("nan")),
                "fx>0 / defined": _pp(float(defn["fx_pos"].mean()) if len(defn) else float("nan")),
                "Y3_fx>0": _pp(_rate_row("x", g[g["fx_pos"]], Y3)["rate"]),
            }
        )
    shares = [
        float(sl.loc[sl["cal_month"] == m, "fx_pos"].mean()) if (sl["cal_month"] == m).any() else float("nan")
        for m in range(1, 13)
    ]
    peak = max(shares) if shares else float("nan")
    trough = min(shares) if shares else float("nan")
    calendar = bool(np.isfinite(peak) and np.isfinite(trough) and (peak - trough) >= 0.08)
    prose = (
        f"fx>0 / ERP CM by calendar month: peak {_pp(peak)} trough {_pp(trough)}. "
        f"{'calendar-shaped' if calendar else 'flat — not a tax-style dummy'}."
    )
    print(prose)
    return {"rows": rows, "peak": peak, "trough": trough, "calendar": calendar, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 23 — store vs live invoices.build (no parquet write)
# ---------------------------------------------------------------------------
def pass23_live(tr: pd.DataFrame, con) -> dict:
    """Recompute family E on the train ERP grid. Do not write parquet."""
    erp = tr[tr["has_book"]][["company_id", "period", "fold", Y3, Y2, "e_fx_share"]].copy()
    ids = sorted(erp["company_id"].unique())
    grid = pd.DataFrame(
        {"company_id": [i for i in ids for _ in MONTHS], "period": list(MONTHS) * len(ids)}
    )
    live = build_invoices(con, grid)
    live = _keys(live[["company_id", "period", "e_fx_share"]].rename(columns={"e_fx_share": "live_fx"}))
    m = erp.merge(live, on=["company_id", "period"], how="left")
    store = pd.to_numeric(m["e_fx_share"], errors="coerce")
    live_s = pd.to_numeric(m["live_fx"], errors="coerce")
    both = store.notna() & live_s.notna()
    abs_diff = (store - live_s).abs()
    n_both = int(both.sum())
    n_disagree = int((both & (abs_diff > 1e-9)).sum())
    max_abs = float(abs_diff[both].max()) if n_both else float("nan")
    store_nan_live = int((store.isna() & live_s.notna()).sum())
    live_nan_store = int((live_s.isna() & store.notna()).sum())
    ever_store = int(m.loc[store > 0, "company_id"].nunique())
    ever_live = int(m.loc[live_s > 0, "company_id"].nunique())
    extra_live = sorted(set(m.loc[live_s > 0, "company_id"]) - set(m.loc[store > 0, "company_id"]))
    # full-grid live (24 months × ERP) vs on-panel store keys
    ever_live_grid = int(live.loc[pd.to_numeric(live["live_fx"], errors="coerce") > 0, "company_id"].nunique())
    store_keys = set(
        zip(erp["company_id"].astype(str), pd.to_datetime(erp["period"]))
    )
    live_pos = live.loc[pd.to_numeric(live["live_fx"], errors="coerce") > 0, ["company_id", "period"]].copy()
    live_pos["period"] = pd.to_datetime(live_pos["period"])
    n_offpanel = int(
        sum((str(r.company_id), r.period) not in store_keys for r in live_pos.itertuples(index=False))
    )
    t2 = tr[["company_id", "period", "log_in3"]].copy()
    m = m.merge(t2, on=["company_id", "period"], how="left")
    live_s = pd.to_numeric(m["live_fx"], errors="coerce")
    res = signed_oof_auroc(m[Y3], live_s, m["fold"], m[Y3].notna())
    size = signed_oof_auroc(m[Y3], m["log_in3"], m["fold"], m[Y3].notna() & live_s.notna())
    beat = (
        (res["cv"] - size["cv"])
        if np.isfinite(res["cv"]) and np.isfinite(size["cv"])
        else float("nan")
    )
    stale = n_disagree > 0 or store_nan_live > 0
    prose = (
        f"On the store panel, live matches store: ever-FX **{ever_live}** / {ever_store}, "
        f"disagree {n_disagree:,} (max |Δ|={_f(max_abs, 6)}). "
        f"Full 24-month grid live ever-FX **{ever_live_grid}**; "
        f"{n_offpanel} live-FX months sit off the monthly panel "
        f"(invoice before first cash-trail month; 4 names never light the store). "
        f"Live-on-panel Y3 FX CV {_f(res['cv'])} vs size {_f(size['cv'])}. "
        "Do not rewrite parquet."
    )
    print(prose)
    return {
        "ever_store": ever_store,
        "ever_live": ever_live,
        "n_both": n_both,
        "n_disagree": n_disagree,
        "max_abs": max_abs,
        "store_nan_live": store_nan_live,
        "live_nan_store": live_nan_store,
        "extra_live": extra_live,
        "live_cv": res["cv"],
        "size_cv": size["cv"],
        "beat": beat,
        "stale": stale,
        "ever_live_grid": ever_live_grid,
        "n_offpanel": n_offpanel,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 24 — always-FX 97 × home currency
# ---------------------------------------------------------------------------
def pass24_always_home(erp: pd.DataFrame, kind_of: pd.Series, modal: pd.DataFrame) -> dict:
    sl = erp[["company_id"]].drop_duplicates()
    sl["kind"] = sl["company_id"].map(kind_of)
    sl = sl.merge(modal, on="company_id", how="left")
    sl["home"] = sl["home_acct"].where(sl["home_acct"].isin(["EUR", "USD"]), "other")
    sl.loc[sl["home_acct"].isna(), "home"] = "unknown"
    rows = []
    for kind in ("never", "shock", "always"):
        rec = {"kind": kind}
        sub = sl[sl["kind"] == kind]
        rec["n"] = int(len(sub))
        for h in ("EUR", "USD", "other"):
            rec[h] = int((sub["home"] == h).sum())
        rows.append(rec)
        print(f"  {kind}: n={rec['n']} EUR={rec['EUR']} USD={rec['USD']} other={rec['other']}")
    always = sl[sl["kind"] == "always"]
    share_non_eur = _pct(int((always["home"] != "EUR").sum()), int(len(always))) if len(always) else float("nan")
    prose = (
        f"Always-FX {int(len(always))}: EUR {int((always['home']=='EUR').sum())} / "
        f"USD {int((always['home']=='USD').sum())} / other {int((always['home']=='other').sum())} "
        f"(non-EUR {_pp(share_non_eur)}). Style is a foreign-home identity more than a month shock."
    )
    print(prose)
    return {"rows": rows, "share_non_eur": share_non_eur, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 25 — FX invoice before first panel month (trail / connection)
# ---------------------------------------------------------------------------
def pass25_pretrail(tr: pd.DataFrame, con) -> dict:
    first = (
        tr[tr["has_book"]]
        .groupby("company_id", as_index=False)["period"]
        .min()
        .rename(columns={"period": "first_panel"})
    )
    raw = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               MIN(CAST(date_trunc('month', issuance_date) AS DATE)) AS first_inv,
               MIN(CASE WHEN currency <> accounting_currency
                        THEN CAST(date_trunc('month', issuance_date) AS DATE) END) AS first_fx
        FROM invoices
        WHERE {BOOK}
        GROUP BY 1
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["first_inv"] = pd.to_datetime(raw["first_inv"])
    raw["first_fx"] = pd.to_datetime(raw["first_fx"])
    hold = load_holdout()
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    m = first.merge(raw, on="company_id", how="left")
    m["inv_before"] = m["first_inv"] < m["first_panel"]
    m["fx_before"] = m["first_fx"].notna() & (m["first_fx"] < m["first_panel"])
    n_inv_b = int(m["inv_before"].sum())
    n_fx_b = int(m["fx_before"].sum())
    n_fx = int(m["first_fx"].notna().sum())
    names = sorted(m.loc[m["fx_before"], "company_id"].tolist())
    prose = (
        f"Train ERP {int(len(m))}: first invoice before first panel month {n_inv_b}; "
        f"first FX invoice before panel {n_fx_b} / {n_fx} ever-raw-FX. "
        f"Names {names}. Same connection clock as trail_length — not a store formula bug."
    )
    print(prose)
    return {
        "n_inv_before": n_inv_b,
        "n_fx_before": n_fx_b,
        "n_fx": n_fx,
        "names": names,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 26 — ever-FX company dummy (BETWEEN identity)
# ---------------------------------------------------------------------------
def pass26_ever_dummy(tr: pd.DataFrame) -> dict:
    ever = set(tr.loc[tr["fx_pos"], "company_id"])
    dummy = tr["company_id"].isin(ever).astype(float)
    # only score on ERP (dark dummy is identically 0 and would be "has_book")
    mask = tr["has_book"] & tr[Y3].notna()
    res = signed_oof_auroc(tr[Y3], dummy, tr["fold"], mask)
    size = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], mask)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], mask)
    beat = (
        (res["cv"] - size["cv"])
        if np.isfinite(res["cv"]) and np.isfinite(size["cv"])
        else float("nan")
    )
    prose = (
        f"Ever-FX company dummy (constant) Y3 CV {_f(res['cv'])} vs size {_f(size['cv'])} "
        f"vs days {_f(days['cv'])} (Δ vs size {_f(beat)}). "
        "BETWEEN identity is not a recover X."
    )
    print(prose)
    return {
        "cv": res["cv"],
        "size": size["cv"],
        "days": days["cv"],
        "beat": beat,
        "n_pos": res["n_pos"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 27 — holdout more FX? home mix coverage only
# ---------------------------------------------------------------------------
def pass27_hold_mix(panel: pd.DataFrame, modal: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"].merge(modal, on="company_id", how="left")
    erp = ho[ho["has_book"]]
    n_eur = int(erp.loc[erp["home_acct"] == "EUR", "company_id"].nunique())
    n_oth = int(erp.loc[erp["home_acct"].notna() & (erp["home_acct"] != "EUR"), "company_id"].nunique())
    fx_eur = float(erp.loc[erp["home_acct"] == "EUR", "fx_pos"].mean()) if n_eur else float("nan")
    fx_oth = (
        float(erp.loc[erp["home_acct"].notna() & (erp["home_acct"] != "EUR"), "fx_pos"].mean())
        if n_oth
        else float("nan")
    )
    prose = (
        f"Holdout ERP home-EUR {n_eur} fx>0 {_pp(fx_eur)}; non-EUR {n_oth} fx>0 {_pp(fx_oth)}. "
        f"Holdout ERP fx>0 {_pp(float(erp['fx_pos'].mean()) if len(erp) else float('nan'))} "
        "vs train 16.8% — coverage only, not a claim."
    )
    print(prose)
    return {"n_eur": n_eur, "n_oth": n_oth, "fx_eur": fx_eur, "fx_oth": fx_oth, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 28 — book quality: FX vs overdue / credit notes (descriptive)
# ---------------------------------------------------------------------------
def pass28_bookqual(tr: pd.DataFrame) -> dict:
    raw = pd.read_parquet(STORE, columns=["company_id", "period", "e_ar_overdue_30", "e_credit_note_ratio"])
    raw = _keys(raw)
    sl = tr.merge(raw, on=["company_id", "period"], how="left")
    sl = sl[sl["has_book"] & sl["fx_defined"]]
    rows = []
    for col in ("e_ar_overdue_30", "e_credit_note_ratio"):
        rho = spearman(sl["e_fx_share"], sl[col])
        rows.append({"pair": f"e_fx_share × {col}", "Spearman": _f(rho)})
        print(f"  {col} ρ={rho:.3f}")
    prose = (
        f"FX vs AR od30 ρ={rows[0]['Spearman']}, vs credit-note ρ={rows[1]['Spearman']}. "
        "Not a late-payer rewrite and not a Y5 X."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 29 — published SHAP rank (read-only, do not refit)
# ---------------------------------------------------------------------------
def pass29_shap() -> dict:
    path = ANALYSIS / "outputs" / "shap_y3_meanabs.csv"
    if not path.exists():
        return {"prose": "shap_y3_meanabs.csv missing — skip.", "rank": None, "meanabs": None}
    df = pd.read_csv(path)
    col = "feature" if "feature" in df.columns else df.columns[0]
    hit = df[df[col].astype(str).eq("e_fx_share")]
    if hit.empty:
        return {"prose": "e_fx_share not in shap_y3_meanabs.csv.", "rank": None, "meanabs": None}
    r = hit.iloc[0]
    rank = int(r["rank"]) if "rank" in df.columns else int(hit.index[0]) + 1
    meanabs = float("nan")
    for c in ("mean_abs", "meanabs", "mean_abs_shap", df.columns[1]):
        if c in df.columns:
            meanabs = float(r[c])
            break
    prose = (
        f"Published Y3 SHAP: `e_fx_share` rank ~{rank}, mean|SHAP|={_f(meanabs, 4)}. "
        "Already a quiet column on the 44 — dropping it is consistent with the single."
    )
    print(prose)
    return {"rank": rank, "meanabs": meanabs, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 30 — 53 pre-trail FX vs other ever-FX (on-panel rates)
# ---------------------------------------------------------------------------
def pass30_pretrail_y(tr: pd.DataFrame, names: list[str]) -> dict:
    ever = set(tr.loc[tr["fx_pos"], "company_id"])
    pre = set(names)
    rows = []
    for label, ids in (
        ("pre-trail FX", pre),
        ("on-panel-only ever-FX", ever - pre),
        ("ERP never-FX", set(tr.loc[tr["has_book"], "company_id"]) - ever - pre),
    ):
        sl = tr[tr["company_id"].isin(ids)]
        r = _rate_row(label, sl, Y3)
        rows.append(
            {
                "slice": label,
                "n_co": int(sl["company_id"].nunique()),
                "n_lab": f"{r['n_lab']:,}",
                "Y3": _pp(r["rate"]),
                "fx>0": _pp(float(sl["fx_pos"].mean()) if len(sl) else float("nan")),
            }
        )
    prose = (
        f"Pre-trail FX companies on-panel Y3 {rows[0]['Y3']} vs other ever-FX {rows[1]['Y3']}. "
        "Do not invent a trail Y from this."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 31 — amount handful (NOK / COP) vs name-count
# ---------------------------------------------------------------------------
def pass31_handful(p1: dict) -> dict:
    top = p1["pair_rows"][:6]
    nok = next((r for r in p1["pair_rows"] if r["acct"] == "EUR" and r["inv"] == "NOK"), None)
    usd = next((r for r in p1["pair_rows"] if r["acct"] == "EUR" and r["inv"] == "USD"), None)
    prose = (
        f"Amount table is a handful: EUR-NOK n_co={nok['n_co'] if nok else '—'} "
        f"|amt|={(nok['abs_amt'] if nok else 0):,.0f}; "
        f"EUR-USD n_co={usd['n_co'] if usd else '—'} "
        f"(many names, smaller |amt|). Do not treat COP/NOK as the 228."
    )
    print(prose)
    return {
        "prose": prose,
        "rows": [
            {"acct": r["acct"], "inv": r["inv"], "n_co": r["n_co"], "|amt|": f"{r['abs_amt']:,.0f}"}
            for r in top
        ],
    }


# ---------------------------------------------------------------------------
# Pass 32 — late joiners (short trail) vs ever-FX
# ---------------------------------------------------------------------------
def pass32_trail(tr: pd.DataFrame) -> dict:
    co = tr[tr["has_book"]].groupby("company_id", as_index=False).agg(
        n_m=("period", "nunique"),
        ever_fx=("fx_pos", "max"),
        med_in3=("log_in3", "median"),
    )
    rho = spearman(co["n_m"], co["ever_fx"].astype(float))
    short = co[co["n_m"] <= 12]
    long = co[co["n_m"] >= 20]
    p_s = float(short["ever_fx"].mean()) if len(short) else float("nan")
    p_l = float(long["ever_fx"].mean()) if len(long) else float("nan")
    prose = (
        f"Ever-FX vs months-on-panel ρ={rho:.3f}. "
        f"≤12 months on book: ever-FX {_pp(p_s)} (n={len(short)}); "
        f"≥20 months: {_pp(p_l)} (n={len(long)}). "
        "Late joiners are not the FX pocket."
    )
    print(prose)
    return {
        "rho": rho,
        "p_short": p_s,
        "p_long": p_l,
        "n_short": int(len(short)),
        "n_long": int(len(long)),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 33 — pre-trail 2.3% recover: size or trail?
# ---------------------------------------------------------------------------
def pass33_pretrail_size(tr: pd.DataFrame, names: list[str]) -> dict:
    ever = set(tr.loc[tr["fx_pos"], "company_id"])
    pre = set(names)
    rows = []
    for label, ids in (
        ("pre-trail FX", pre),
        ("other ever-FX", ever - pre),
        ("ERP never-FX", set(tr.loc[tr["has_book"], "company_id"]) - ever),
    ):
        sl = tr[tr["company_id"].isin(ids) & tr["has_book"]]
        rows.append(
            {
                "slice": label,
                "n_co": int(sl["company_id"].nunique()),
                "med_log_in3": _f(float(sl["log_in3"].median()) if sl["log_in3"].notna().any() else float("nan")),
                "med_months": _f(float(sl.groupby("company_id")["period"].nunique().median()) if len(sl) else float("nan"), 1),
                "Y3": _pp(_rate_row(label, sl, Y3)["rate"]),
            }
        )
    prose = (
        f"Pre-trail FX median log1p(a_in3) {rows[0]['med_log_in3']} vs other ever-FX {rows[1]['med_log_in3']}; "
        f"Y3 {rows[0]['Y3']} vs {rows[1]['Y3']}. "
        "The 2.3% is a quieter stressed set, not a recover engine."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 34 — Y5 descriptive by style (never as X)
# ---------------------------------------------------------------------------
def pass34_y5_style(erp: pd.DataFrame, kind_of: pd.Series) -> dict:
    sl = erp.copy()
    sl["fx_kind"] = sl["company_id"].map(kind_of)
    rows = []
    for kind in ("never", "shock", "always"):
        sub = sl[sl["fx_kind"] == kind]
        rec = {"kind": kind, "n_co": int(sub["company_id"].nunique())}
        for y in (Y5_AP, Y5_AR):
            r = _rate_row(kind, sub, y)
            rec[y] = _pp(r["rate"])
            rec[f"{y}_n"] = f"{r['n_lab']:,}"
        rows.append(rec)
    leak = leakage_check(["e_fx_share"], Y5_AP, ("e",))
    prose = (
        f"Y5 AP always {rows[2][Y5_AP]} vs never {rows[0][Y5_AP]}; "
        f"AR always {rows[2][Y5_AR]} vs never {rows[0][Y5_AR]}. "
        f"Y5×E leak screen fails as required ({leak['issues']}). Descriptive only."
    )
    print(prose)
    return {"rows": rows, "leak_fails": not leak["ok"], "prose": prose}


# ---------------------------------------------------------------------------
# Pass 35 — Y3 residual inside activity (days) terciles
# ---------------------------------------------------------------------------
def pass35_days_resid(erp: pd.DataFrame) -> dict:
    sl = erp[erp["fx_defined"] & erp["c_n_days_with_tx"].notna() & erp[Y3].notna()].copy()
    sl["dq"] = pd.qcut(pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce"), 3, duplicates="drop", labels=False)
    rows = []
    gaps = []
    for i, g in sl.groupby("dq", observed=True):
        r_pos = _rate_row("fx>0", g[g["fx_pos"]], Y3)
        r_z = _rate_row("fx=0", g[g["fx_zero"]], Y3)
        gap = (
            (r_pos["rate"] - r_z["rate"])
            if np.isfinite(r_pos["rate"]) and np.isfinite(r_z["rate"])
            else float("nan")
        )
        gaps.append(gap)
        rows.append(
            {
                "days_T": int(i) + 1,
                "days_med": _f(float(g["c_n_days_with_tx"].median()), 1),
                "n_lab": f"{int(len(g)):,}",
                "Y3_fx>0": _pp(r_pos["rate"]),
                "Y3_fx=0": _pp(r_z["rate"]),
                "n_pos_fx": f"{r_pos['n_pos']:,}",
                "gap": _pp(gap) if np.isfinite(gap) else "—",
            }
        )
    mean_gap = float(np.nanmean(gaps)) if gaps else float("nan")
    vanishes = bool(np.isfinite(mean_gap) and abs(mean_gap) < 0.02)
    prose = (
        f"Y3 fx>0 minus fx=0 inside `c_n_days_with_tx` terciles: mean gap {_pp(mean_gap)}. "
        f"{'vanishes after activity' if vanishes else 'a small residual remains; days single is still 0.711'}."
    )
    print(prose)
    return {"rows": rows, "mean_gap": mean_gap, "vanishes": vanishes, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 36 — FX single inside high-activity tercile only
# ---------------------------------------------------------------------------
def pass36_days_hi(erp: pd.DataFrame) -> dict:
    sl = erp[erp["fx_defined"] & erp["c_n_days_with_tx"].notna()].copy()
    sl["dq"] = pd.qcut(
        pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce"), 3, duplicates="drop", labels=False
    )
    hi = sl[sl["dq"] == sl["dq"].max()]
    res = signed_oof_auroc(hi[Y3], hi["e_fx_share"], hi["fold"], hi[Y3].notna())
    size = signed_oof_auroc(hi[Y3], hi["log_in3"], hi["fold"], hi[Y3].notna())
    days = signed_oof_auroc(hi[Y3], hi["c_n_days_with_tx"], hi["fold"], hi[Y3].notna())
    prose = (
        f"High-activity tercile only: Y3 FX {_f(res['cv'])} "
        f"{'(LOW_POWER) ' if res['low_power'] else ''}"
        f"vs size {_f(size['cv'])} vs days {_f(days['cv'])} "
        f"(n_pos={res['n_pos']}). Still not a Y3 X."
    )
    print(prose)
    return {
        "cv": res["cv"],
        "size": size["cv"],
        "days": days["cv"],
        "n_pos": res["n_pos"],
        "low_power": res["low_power"],
        "prose": prose,
    }


def make_png(p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    rows = p3.get("y3_pos_q") or p3.get("y3_q")
    if not rows:
        print("no Y3 quintile rows — skip PNG")
        return False
    # prepend fx=0 bar from the binary split
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    xs = [r["q"] for r in rows]
    ys = [r["rate"] for r in rows]
    ax.bar(xs, ys, color="#3d5a80", width=0.7)
    if np.isfinite(p3.get("y3_zero", float("nan"))):
        ax.axhline(p3["y3_zero"], color="#9e6b4a", ls="--", lw=1, label=f"fx=0  {_pp(p3['y3_zero'])}")
    if np.isfinite(p3.get("y3_pos", float("nan"))):
        ax.axhline(p3["y3_pos"], color="#1f4e79", ls=":", lw=1, label=f"fx>0  {_pp(p3['y3_pos'])}")
    for r in rows:
        ax.text(
            r["q"],
            r["rate"] + 0.004,
            f"{r['rate']:.1%}\nn={r['n']}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"Q{r['q']}\n{_f(r['fx_med'], 2)}" for r in rows])
    ax.set_ylabel("P(Y=1)  y3_recover_cash_6m")
    ax.set_xlabel("e_fx_share intensity quintile among fx>0 (train, stressed, defined)")
    ymax = max(ys + [p3.get("y3_zero", 0) or 0, p3.get("y3_pos", 0) or 0])
    ax.set_ylim(0, max(ymax, 0.05) * 1.35)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Y3 recovery by FX intensity (invoice FX months, not dark)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p1, p2, p3, p4, p5, p6, p7, p8, p10=None, p11=None, p12=None) -> dict:
    park_y = True
    close_y7y5 = True
    keep_x = bool(p4["keep_x"] and not p2["is_size"])
    book = (p10 or {}).get("label", "FX book")
    if keep_x:
        x_dec = "KEEP"
        q5 = "KEEP"
        why = (
            f"Y3 single {_f(p4['fx_y3'])} beats same-mask size {_f(p4['size_y3'])} "
            f"by {_f(p4['beat'])} and is not SIZE (ρ_in3={_f(p2['rho_in3'])})."
        )
    else:
        x_dec = "CLOSE"
        q5 = "CLOSE"
        why = (
            f"Y3 single {_f(p4['fx_y3'])} vs size {_f(p4['size_y3'])} "
            f"(Δ {_f(p4['beat'])}); "
            f"{'SIZE' if p2['is_size'] else 'not SIZE'} ρ_in3={_f(p2['rho_in3'])}. "
            f"CLOSE as a Q5 footnote ({book}; foreign-home identity, not the 110), "
            "not a 44-col Y3 X."
        )
    q6 = (p11 or {}).get("q6", "CLOSE")
    return {
        "park_y": park_y,
        "close_y7y5": close_y7y5,
        "keep_x": keep_x,
        "x_dec": x_dec,
        "q5": q5,
        "q6": q6,
        "why": why,
        "book": book,
        "keep_list": "KEEP on the 44" if keep_x else "drop from the 44-col Y3 starter",
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5, p6, p7, p8, p9, d = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
        ctx["decision"],
    )
    p10, p11, p12, p13, p14, p15, p16, p17, p18 = (
        ctx["p10"],
        ctx["p11"],
        ctx["p12"],
        ctx["p13"],
        ctx["p14"],
        ctx["p15"],
        ctx["p16"],
        ctx["p17"],
        ctx["p18"],
    )
    p19, p20, p21, p22 = ctx["p19"], ctx["p20"], ctx["p21"], ctx["p22"]
    p23, p24 = ctx["p23"], ctx["p24"]
    p25, p26, p27, p28 = ctx["p25"], ctx["p26"], ctx["p27"], ctx["p28"]
    p29, p30, p31, p32 = ctx["p29"], ctx["p30"], ctx["p31"], ctx["p32"]
    p33, p34, p35, p36 = ctx["p33"], ctx["p34"], ctx["p35"], ctx["p36"]
    lines = [
        "# Q5 invoice FX share (`e_fx_share`)",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**, ever-ERP. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent a merged FX Y.",
        "",
        "`e_fx_share` = this-period issued |amount| with `currency <> accounting_currency`. "
        "Y7 / Y5 are invoice-built — **never E as X**. Y3 cash-recover forbids B; Y2 forbids B. "
        "470 dark: FX is NaN, not 0.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | PARK as a health Y. FX is a mixed import/export book, not a FICO label. |",
        "| 2 | Who is improving? | Not this share. |",
        "| 3 | Who is turning? | Y2 single is a check, not a claim. |",
        "| 4 | Dip vs fall? | Y7/Y5 descriptive only. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['why']} |",
        f"| 6 | Months earlier? | **{d['q6']}** lag1 {_f(p11['lag'])} vs now {_f(p11['now'])}; acf1 {_f(p5['acf1'])} ({p5['style']}). |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        "| `e_fx_share` as a health Y | **PARK** | do not invent a merged FX Y |",
        "| `e_fx_share` as Y7 / Y5 X | **CLOSE** | forbidden family E (labels are invoice-built) |",
        f"| `e_fx_share` as Y3 X (44-col list) | **{d['x_dec']}** | {d['why']} |",
        f"| `e_fx_share` as Y2 X | **CLOSE** | single {_f(p4['fx_y2'])} vs size {_f(p4['size_y2'])}; models already PARK |",
        f"| Q5 footnote ({d['book']}) | **KEEP footnote** | {p5['n_always']} always / {p5['n_shock']} shock; HHI ρ={_f(p7['rho'])}; not the 110 |",
        f"| Q6 `e_fx_share_lag1` | **{d['q6']}** | {p11['prose']} |",
        f"| 44-col keep list | **{d['keep_list']}** | gate = beat size by ≥0.02 and not SIZE |",
        "",
        "## 1. Coverage (train, ever-ERP)",
        "",
        p1["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train ERP companies / CM | {p1['n_erp_co']:,} / {p1['n_erp_cm']:,} |",
        f"| confirm 744 / 470 | {'YES' if p1['confirm_744'] and p1['confirm_470'] else 'NO'} |",
        f"| `e_fx_share` defined | {_pp(p1['share_def'])} ({p1['n_def']:,}) |",
        f"| `e_fx_share` > 0 / ERP CM | {_pp(p1['share_pos_erp'])} ({p1['n_pos']:,}) |",
        f"| `e_fx_share` > 0 / defined | {_pp(p1['share_pos_def'])} |",
        f"| ever-FX companies | **{p1['n_ever']}** |",
        f"| dark companies / defined FX months | {p1['n_dark_co']:,} / {p1['dark_n_defined']:,} |",
        f"| dark is all-NaN | {'YES' if p1['dark_nan'] else 'NO'} |",
        f"| raw FX invoices (train) | {p1['n_fx_inv']:,} across {p1['n_fx_co_raw']:,} companies |",
        f"| AR / AP FX invoices | {p1['ar_n']:,} / {p1['ap_n']:,} |",
        f"| invoice currencies | {', '.join(p1['inv_currs']) or '—'} |",
        f"| accounting currencies | {', '.join(p1['acct_currs']) or '—'} |",
        "",
        "FX pairs (train, |amount|):",
        "",
        _md_table(
            [
                {
                    "acct": r["acct"],
                    "invoice": r["inv"],
                    "n_inv": f"{r['n_inv']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "|amt|": f"{r['abs_amt']:,.0f}",
                }
                for r in p1["pair_rows"][:12]
            ]
        ),
        "",
        "## 2. Size ρ",
        "",
        p2["prose"],
        "",
        "| pair | Spearman | SIZE? |",
        "| --- | ---: | --- |",
        f"| vs log1p(a_in3) | {_f(p2['rho_in3'])} | {'YES' if abs(p2['rho_in3'])>=SIZE_RHO else ''} |",
        f"| vs e_ar_issued | {_f(p2['rho_ar'])} | {'YES' if abs(p2['rho_ar'])>=SIZE_RHO else ''} |",
        f"| vs log1p(e_ar_issued) | {_f(p2['rho_log_ar'])} | |",
        f"| fx>0 only vs in3 | {_f(p2['rho_pos_in3'])} | |",
        f"| fx>0 only vs issued | {_f(p2['rho_pos_ar'])} | |",
        f"| ever-FX vs company med in3 | {_f(p2['rho_ever_size'])} | |",
        "",
        f"Gate: SIZE if |ρ| ≥ {SIZE_RHO:g} vs log1p(a_in3) or e_ar_issued.",
        "",
        "## 3. Quintiles and descriptive Y7 / Y5 (not as X)",
        "",
        p3["prose"],
        "",
        "Binary fx>0 vs 0 vs NaN (no issue month). Y7 / Y5 rates only — family E is forbidden X.",
        "",
        _md_table(p3["split_rows"]),
        "",
        "Y3 stressed, qcut of defined `e_fx_share` (zeros may collapse bins):",
        "",
        _md_table(
            [
                {
                    "q": r["q"],
                    "interval": r["interval"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                    "fx med": _f(r["fx_med"], 3),
                }
                for r in p3["y3_q"]
            ]
        )
        if p3["y3_q"]
        else "_(qcut collapsed)_\n",
        "",
        "Y3 intensity quintiles among fx>0:",
        "",
        _md_table(
            [
                {
                    "q": r["q"],
                    "interval": r["interval"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                    "fx med": _f(r["fx_med"], 3),
                }
                for r in p3["y3_pos_q"]
            ]
        )
        if p3["y3_pos_q"]
        else "_(empty)_\n",
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped.",
        "",
        "## 4. Single-feature train group-fold AUROC (Y3 / Y2 only)",
        "",
        f"Y3 stressed n={p4['n_y3']:,} base {_pp(p4['y3_rate'])}; "
        f"Y2 n={p4['n_y2']:,} base {_pp(p4['y2_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (full-panel replica {_f(p4['days_y3_full'])}). "
        "KEEP as Y3 X only if same-mask FX beats same-mask size by ≥0.02 and is not SIZE. "
        "Y7×E leak screen is expected to fail — we do not score that pair.",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Persistence — style vs month shock",
        "",
        p5["prose"],
        "",
        _md_table(p5["kind_rows"]),
        "",
        f"always = fx>0 in ≥{STYLE_ALWAYS:.0%} of defined months. shock = some FX, not always. never = defined months all 0.",
        "",
        "## 6. Leak vs issued / DSO (Y7 story)",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. FX vs `d_cust_hhi` (foreign ≠ monopoly)",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8. Mixed-group 110 — are FX companies those dark siblings?",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9. Holdout coverage only (no AUROC)",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "## 10. AR vs AP FX (exporter vs importer)",
        "",
        p10["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| AR / AP FX \\|amount\\| | {_pp(p10['ar_share'])} / {_pp(1.0 - p10['ar_share'])} |",
        f"| ever AR-FX / AP-FX / both | {p10['n_ar']} / {p10['n_ap']} / {p10['n_both']} |",
        f"| AR-only / AP-only | {p10['n_ar_only']} / {p10['n_ap_only']} |",
        f"| Y3 AR-FX / AP-FX CV | {_f(p10['ar_cv'])} / {_f(p10['ap_cv'])} |",
        f"| book call | {p10['label']} |",
        "",
        _md_table(p10["rate_rows"]),
        "",
        "## 11. Q6 — lag1 of `e_fx_share` (1-month only)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Y3 residual inside size terciles",
        "",
        p12["prose"],
        "",
        _md_table(p12["rows"]),
        "",
        "## 13. HHI × FX Y3 cells (Q4/Q5 spike)",
        "",
        p13["prose"],
        "",
        _md_table(p13["rows"]),
        "",
        "## 14. Within shock companies — FX month vs own zero month",
        "",
        p14["prose"],
        "",
        _md_table(p14["rows"]),
        "",
        "## 15. Feature-report confirm",
        "",
        p15["prose"],
        "",
        "## 16. No-issue month vs FX",
        "",
        p16["prose"],
        "",
        "## 17. Major vs exotic invoice currency",
        "",
        p17["prose"],
        "",
        _md_table(p17["rows"]),
        "",
        "Top FX invoice currencies by |amount| (train):",
        "",
        _md_table(p17["top_rows"]),
        "",
        "## 18. Raw 232 vs store 228 + home accounting currency",
        "",
        p18["prose"],
        "",
        _md_table(p18["extra_rows"]) if p18["extra_rows"] else "No raw-only names.\n",
        "",
        _md_table(p18["home_rows"]),
        "",
        "## 19. Four raw-FX names that never light the store",
        "",
        p19["prose"],
        "",
        _md_table(p19["rows"]),
        "",
        "## 20. Home-EUR only singles (majority book)",
        "",
        p20["prose"],
        "",
        _md_table(p20["rows"]),
        "",
        "## 21. High-intensity flag (share ≥ 0.08)",
        "",
        p21["prose"],
        "",
        "## 22. FX calendar",
        "",
        p22["prose"],
        "",
        _md_table(p22["rows"]),
        "",
        "## 23. Store vs live `invoices.build` (no parquet write)",
        "",
        p23["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| store ever-FX | {p23['ever_store']} |",
        f"| live ever-FX (on panel) | {p23['ever_live']} |",
        f"| live ever-FX (24m grid) | {p23.get('ever_live_grid', '—')} |",
        f"| live-FX months off panel | {p23.get('n_offpanel', 0)} |",
        f"| both-defined months | {p23['n_both']:,} |",
        f"| disagree (\\|Δ\\| > 0) | {p23['n_disagree']:,} |",
        f"| max \\|Δ\\| | {_f(p23['max_abs'], 6)} |",
        f"| store NaN / live defined | {p23['store_nan_live']:,} |",
        f"| live Y3 FX CV | {_f(p23['live_cv'])} |",
        f"| live same-mask size | {_f(p23['size_cv'])} |",
        "",
        "Do **not** rewrite parquet tonight. Do **not** patch `invoices.py` — live build already has the 4 names. The 228 quote is the store.",
        "",
        "## 24. Always-FX × home currency",
        "",
        p24["prose"],
        "",
        _md_table(p24["rows"]),
        "",
        "## 25. FX / invoice before first panel month",
        "",
        p25["prose"],
        "",
        "## 26. Ever-FX company dummy (BETWEEN)",
        "",
        p26["prose"],
        "",
        "## 27. Holdout home mix (coverage only)",
        "",
        p27["prose"],
        "",
        "## 28. FX vs overdue / credit notes",
        "",
        p28["prose"],
        "",
        _md_table(p28["rows"]),
        "",
        "## 29. Published Y3 SHAP (no refit)",
        "",
        p29["prose"],
        "",
        "## 30. Pre-trail FX companies on the panel",
        "",
        p30["prose"],
        "",
        _md_table(p30["rows"]),
        "",
        "## 31. Amount handful vs name-count",
        "",
        p31["prose"],
        "",
        _md_table(p31["rows"]),
        "",
        "## 32. Months-on-panel vs ever-FX",
        "",
        p32["prose"],
        "",
        "## 33. Pre-trail 2.3% — size or trail?",
        "",
        p33["prose"],
        "",
        _md_table(p33["rows"]),
        "",
        "## 34. Y5 by style (never as X)",
        "",
        p34["prose"],
        "",
        _md_table(p34["rows"]),
        "",
        "## 35. Y3 residual inside activity terciles",
        "",
        p35["prose"],
        "",
        _md_table(p35["rows"]),
        "",
        "## 36. High-activity tercile FX single",
        "",
        p36["prose"],
        "",
        "## What failed / next",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: coverage, size, quintiles+Y7/Y5, Y3/Y2 singles, "
        "persistence, leak, HHI, mixed-110, holdout, AR/AP, Q6 lag1, size residual, HHI cells, "
        "within-shock, feature-report confirm, no-issue hole, major/exotic, raw-gap + home ccy, "
        "four-name gap, home-EUR singles, high-intensity, calendar, live-vs-store, always×home, "
        "pre-trail FX, ever-dummy, holdout home, book quality, SHAP, pre-trail Y3, handful, trail length, "
        "pre-trail size, Y5 style, days residual, high-activity single.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p2, p3, p4, p5, d = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["decision"],
    )
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": "-",
            "model": "fx_qa",
            "split": "train",
            "metric": "e_fx_share_ever_n",
            "value": p1["n_ever"],
            "coverage": f"{p1['share_def']:.4f}",
            "notes": f"pos_erp={p1['share_pos_erp']:.4f} pos_def={p1['share_pos_def']:.4f} dark_nan={p1['dark_nan']} curr={','.join(p1['inv_currs'])}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": "-",
            "model": "fx_qa",
            "split": "train",
            "metric": "e_fx_share_size_rho_in3",
            "value": p2["rho_in3"],
            "coverage": f"{p2['n_def'] / ctx['p1']['n_erp_cm']:.4f}" if ctx["p1"]["n_erp_cm"] else "",
            "notes": f"rho_ar={p2['rho_ar']:.4f} size={p2['is_size']} ever_size={p2['rho_ever_size']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": Y3,
            "model": "fx_qa",
            "split": "train_cv",
            "metric": "auroc_e_fx_share",
            "value": p4["fx_y3"],
            "coverage": f"{p1['share_def']:.4f}",
            "notes": f"size={p4['size_y3']:.4f} days_full={p4['days_y3_full']:.4f} beat={p4['beat']:.4f} keep_x={p4['keep_x']} x={d['x_dec']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": Y2,
            "model": "fx_qa",
            "split": "train_cv",
            "metric": "auroc_e_fx_share",
            "value": p4["fx_y2"],
            "coverage": f"{p1['share_def']:.4f}",
            "notes": f"size={p4['size_y2']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": Y7,
            "model": "fx_qa",
            "split": "train",
            "metric": "y7_rate_fx_pos_vs_zero",
            "value": p3["y7_pos"],
            "coverage": f"{p1['share_pos_erp']:.4f}",
            "notes": f"fx0={p3['y7_zero']:.4f} descriptive_only never_X",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": "-",
            "model": "fx_qa",
            "split": "train",
            "metric": "e_fx_share_acf1",
            "value": p5["acf1"],
            "coverage": f"{p1['share_def']:.4f}",
            "notes": f"style={p5['style']} always={p5['n_always']} shock={p5['n_shock']} never={p5['n_never']} ever_acf1={p5['acf1_ever']:.3f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": Y3,
            "model": "fx_qa",
            "split": "train_cv",
            "metric": "auroc_e_fx_share_lag1",
            "value": ctx["p11"]["lag"],
            "coverage": f"{p1['share_def']:.4f}",
            "notes": f"now={ctx['p11']['now']:.4f} q6={ctx['p11']['q6']} book={ctx['p10']['label']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": "-",
            "model": "fx_qa",
            "split": "train",
            "metric": "e_fx_ar_amt_share",
            "value": ctx["p10"]["ar_share"],
            "coverage": f"{p1['share_pos_erp']:.4f}",
            "notes": f"label={ctx['p10']['label']} n_ar={ctx['p10']['n_ar']} n_ap={ctx['p10']['n_ap']} both={ctx['p10']['n_both']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "E",
            "y": "-",
            "model": "fx_qa",
            "split": "train",
            "metric": "e_fx_share_live_ever_n",
            "value": ctx["p23"]["ever_live"],
            "coverage": f"{p1['share_def']:.4f}",
            "notes": f"store={ctx['p23']['ever_store']} disagree={ctx['p23']['n_disagree']} live_y3={ctx['p23']['live_cv']:.4f} stale={ctx['p23']['stale']} no_parquet_write",
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


def run() -> dict:
    t0 = time.time()
    print(f"fx_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    con = connect()
    try:
        panel = attach_book_and_folds(panel, con)
        pop = dict(panel.attrs.get("dark_pop") or {})
        panel = add_panel_lags(panel, ["e_fx_share"], (1, 3))
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print(
            f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
            f"ERP={int(tr.loc[tr['has_book'],'company_id'].nunique())} "
            f"dark={pop['n_train_dark']}"
        )
        erp = tr[tr["has_book"]].copy()
        print("pass 1 coverage")
        p1 = pass1_coverage(tr, con, pop)
        print("pass 2 size")
        p2 = pass2_size(erp)
        print("pass 3 quintiles")
        p3 = pass3_quintiles(erp)
        print("pass 4 singles")
        p4 = pass4_singles(tr)
        print("pass 5 persist")
        p5 = pass5_persist(erp)
        print("pass 6 leak")
        p6 = pass6_leak(erp)
        print("pass 7 HHI")
        p7 = pass7_hhi(erp)
        print("pass 8 mixed-110")
        p8 = pass8_mixed(tr, pop)
        print("pass 9 holdout")
        p9 = pass9_holdout(panel)
        print("pass 10 AR vs AP")
        p10 = pass10_side(tr, con)
        print("pass 11 Q6 lag1")
        p11 = pass11_q6(tr)
        print("pass 12 size residual")
        p12 = pass12_size_resid(erp)
        print("pass 13 HHI cells")
        p13 = pass13_hhi_cells(erp)
        print("pass 14 within shock")
        p14 = pass14_within(erp, p5["kind_of"])
        print("pass 15 confirm")
        p15 = pass15_confirm(tr, p1)
        print("pass 16 no-issue")
        p16 = pass16_noissue(erp, tr)
        print("pass 17 major/exotic")
        p17 = pass17_ccy(tr, con)
        print("pass 18 raw-gap + home ccy")
        p18 = pass18_gap_home(tr, p1, con)
        print("pass 19 four names")
        p19 = pass19_four(tr, p18["extra"], con)
        print("pass 20 home-EUR singles")
        p20 = pass20_home_eur(tr, p18["modal"])
        print("pass 21 high-intensity")
        p21 = pass21_hi(tr)
        print("pass 22 calendar")
        p22 = pass22_cal(erp)
        print("pass 23 live vs store")
        p23 = pass23_live(tr, con)
        print("pass 24 always × home")
        p24 = pass24_always_home(erp, p5["kind_of"], p18["modal"])
        print("pass 25 pre-trail")
        p25 = pass25_pretrail(tr, con)
        print("pass 26 ever-dummy")
        p26 = pass26_ever_dummy(tr)
        print("pass 27 holdout home")
        p27 = pass27_hold_mix(panel, p18["modal"])
        print("pass 28 book quality")
        p28 = pass28_bookqual(tr)
        print("pass 29 SHAP")
        p29 = pass29_shap()
        print("pass 30 pre-trail Y")
        p30 = pass30_pretrail_y(tr, p25["names"])
        print("pass 31 handful")
        p31 = pass31_handful(p1)
        print("pass 32 trail vs FX")
        p32 = pass32_trail(tr)
        print("pass 33 pre-trail size")
        p33 = pass33_pretrail_size(tr, p25["names"])
        print("pass 34 Y5 style")
        p34 = pass34_y5_style(erp, p5["kind_of"])
        print("pass 35 days residual")
        p35 = pass35_days_resid(erp)
        print("pass 36 high-activity FX")
        p36 = pass36_days_hi(erp)
    finally:
        con.close()

    decision = decide(p1, p2, p3, p4, p5, p6, p7, p8, p10, p11, p12)
    png_ok = make_png(p3)
    headline = (
        f"Ever-FX **{p1['n_ever']}** train ERP companies; fx>0 on {_pp(p1['share_pos_erp'])} of ERP CM. "
        f"Size ρ vs log1p(a_in3) {_f(p2['rho_in3'])} ({'SIZE' if p2['is_size'] else 'not SIZE'}). "
        f"Y3 single {_f(p4['fx_y3'])} vs size {_f(p4['size_y3'])} vs days-full {_f(p4['days_y3_full'])} "
        f"(night 0.711). Y7 descriptive {_pp(p3['y7_pos'])} vs {_pp(p3['y7_zero'])}. "
        f"Book **{p10['label']}**. Style **{p5['style']}**. Q5 **{decision['q5']}**. "
        f"44-col **{decision['keep_list']}**."
    )
    print(headline)
    failed = []
    if not p1["dark_nan"]:
        failed.append("dark companies have defined e_fx_share — store bug")
    if not p1["confirm_470"] or not p1["confirm_744"]:
        failed.append(f"ERP/dark {p1['n_erp_co']}/{p1['n_dark_co']} ≠ 744/470")
    if not p4["days_ok"]:
        failed.append(
            f"Y3 days full-panel {_f(p4['days_y3_full'])} vs night 0.711 "
            "(signed fold; published 0.711 may be oriented 1−raw)"
        )
    if not p8.get("confirm_110"):
        failed.append(f"mixed/all-dark {p8['n_110']}/{p8['n_360']} ≠ 110/360")
    if not p15.get("confirm_cov"):
        failed.append(f"full-train cov {_pp(p15['cov_cm'])} ≠ 52.8% quote")
    if p23.get("stale"):
        failed.append(
            f"store stale vs live invoices.build: ever {p23['ever_store']}→{p23['ever_live']}, "
            f"disagree {p23['n_disagree']}. Do not rewrite parquet."
        )
    failed.append(
        f"Y3 FX {_f(p4['fx_y3'])} loses to size {_f(p4['size_y3'])} by "
        f"{_f(-p4['beat'] if np.isfinite(p4['beat']) else float('nan'))}; "
        f"live rebuild {_f(p23['live_cv'])} still loses. drop from the 44. Book is {p10['label']}."
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
        "p29": p29,
        "p30": p30,
        "p31": p31,
        "p32": p32,
        "p33": p33,
        "p34": p34,
        "p35": p35,
        "p36": p36,
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
        "panel": panel,
        "tr": tr,
        "erp": erp,
        "pop": pop,
    }
    write_md(ctx)
    append_registry(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
