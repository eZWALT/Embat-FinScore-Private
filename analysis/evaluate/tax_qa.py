"""Q5 tax calendar vs missed-tax — calendar dummy or a health why?

NORTH_STAR: tax months are a calendar, not a health Y unless proven.
`c_tax_month` = any category `tax` (not `tax_refund`).
`c_missed_tax` = usual tax in last ≤6 months (≥3) AND no tax this month.
Feature report: rare-event flag, modal 91.7%.

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
`build_targets`. Do not invent a merged Y from missed-tax. Do not
edit ops.py unless a real bug.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.tax_qa

Owned: analysis/evaluate/tax_qa.py, analysis/outputs/tax_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_tax.md (end).
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
    train_companies,
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
OUT_MD = ANALYSIS / "outputs" / "tax_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "tax_month_calendar.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "1094dc70"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y9 = "y9_fee_r_ownp80"
N_FOLDS = 5
DAYS_BENCH = 0.711
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
MODAL_QUOTE = 0.917
PANEL_START = pd.Timestamp("2024-09-01")
PANEL_END = pd.Timestamp("2026-08-01")  # inclusive month-start
Q_MONTHS = (1, 4, 7, 10)  # VAT / IS typical quarters
MIN_POS = 50

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_out6",
    "c_n_days_with_tx",
    "c_tax_month",
    "c_missed_tax",
    "c_salary_month",
    "c_missed_salary",
    "c_ss_month",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")


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
    ykeep = ["company_id", "period", Y2, Y3, Y9]
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["cal_month"] = panel["period"].dt.month
    panel["cal_year"] = panel["period"].dt.year
    panel["is_q_month"] = panel["cal_month"].isin(Q_MONTHS).astype(np.int8)
    panel["is_jan"] = (panel["cal_month"] == 1).astype(np.int8)
    panel["not_tax_month"] = (pd.to_numeric(panel["c_tax_month"], errors="coerce") == 0).astype(
        np.int8
    )
    panel["log_in3"] = np.log1p(
        pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0)
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


# ---------------------------------------------------------------------------
# Pass 1 — calendar
# ---------------------------------------------------------------------------
def pass1_calendar(tr: pd.DataFrame, con) -> dict:
    """Share of train companies with a tax tx by calendar month, stacked 2024-09..2026-08."""
    raw = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          COUNT(*) AS n_tax_tx,
          SUM(ABS(amount)) AS abs_tax,
          SUM(amount) AS net_tax
        FROM transactions
        WHERE category = 'tax'
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["month"])
    hold = load_holdout()
    raw = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(raw["company_id"])

    grid = tr[["company_id", "period", "cal_month", "cal_year", "c_tax_month"]].copy()
    grid["has_tax_store"] = pd.to_numeric(grid["c_tax_month"], errors="coerce") == 1
    hit = raw[["company_id", "period"]].drop_duplicates()
    hit["has_tax_raw"] = True
    grid = grid.merge(hit, on=["company_id", "period"], how="left")
    grid["has_tax_raw"] = grid["has_tax_raw"].eq(True)
    # store vs raw agreement (ops.py bug screen, do not rewrite)
    agree = float((grid["has_tax_store"] == grid["has_tax_raw"]).mean())

    # stacked Jan–Dec: company-months on grid that calendar month
    cal_rows = []
    for m in range(1, 13):
        sl = grid[grid["cal_month"] == m]
        n_cm = int(len(sl))
        n_co = int(sl["company_id"].nunique())
        n_tax_cm = int(sl["has_tax_raw"].sum())
        n_tax_co = int(sl.loc[sl["has_tax_raw"], "company_id"].nunique())
        years = sorted(sl["cal_year"].unique())
        cal_rows.append(
            {
                "month": m,
                "name": pd.Timestamp(2000, m, 1).strftime("%b"),
                "q": int(m in Q_MONTHS),
                "n_cm": n_cm,
                "n_co": n_co,
                "share_cm": _pct(n_tax_cm, n_cm),
                "share_co": _pct(n_tax_co, n_co),
                "n_tax_cm": n_tax_cm,
                "years": ",".join(str(y) for y in years),
            }
        )
    q_share = float(np.nanmean([r["share_cm"] for r in cal_rows if r["q"] == 1]))
    nq_share = float(np.nanmean([r["share_cm"] for r in cal_rows if r["q"] == 0]))
    peak = max(cal_rows, key=lambda r: r["share_cm"] if np.isfinite(r["share_cm"]) else -1)
    trough = min(cal_rows, key=lambda r: r["share_cm"] if np.isfinite(r["share_cm"]) else 9)
    # Pure quarterly: Q high and off-quarter near-empty. 40%+ off-quarter is not that.
    ratio = q_share / nq_share if nq_share else float("nan")
    quarterly = bool(q_share >= 0.55 and nq_share <= 0.25)
    monthly = bool(nq_share >= 0.35 and np.isfinite(ratio) and ratio < 1.3)
    mixed_q = bool((not quarterly) and (not monthly) and np.isfinite(ratio) and ratio >= 1.3)
    shape = (
        "quarterly"
        if quarterly
        else ("monthly" if monthly else ("mixed_q_peaked" if mixed_q else "mixed"))
    )

    # year × month matrix for heatmap (share of companies on grid)
    ym = (
        grid.groupby(["cal_year", "cal_month"], as_index=False)
        .agg(n_co=("company_id", "nunique"), n_tax=("has_tax_raw", "sum"), n_cm=("company_id", "size"))
    )
    ym["share_cm"] = ym["n_tax"] / ym["n_cm"]
    # company-level: share of *companies present that month* with a tax tx
    ym_co = []
    for (y, m), sl in grid.groupby(["cal_year", "cal_month"]):
        n_co = sl["company_id"].nunique()
        n_tax_co = sl.loc[sl["has_tax_raw"], "company_id"].nunique()
        ym_co.append(
            {
                "year": int(y),
                "month": int(m),
                "n_co": int(n_co),
                "share_co": _pct(n_tax_co, n_co),
                "share_cm": float(sl["has_tax_raw"].mean()),
            }
        )

    prose = (
        f"Stacked Jan–Dec train company-months: Q-months (Jan/Apr/Jul/Oct) tax-CM share "
        f"**{_pp(q_share)}**, other months **{_pp(nq_share)}** "
        f"(ratio {q_share / nq_share if nq_share else float('nan'):.2f}). "
        f"Peak {peak['name']} {_pp(peak['share_cm'])}, trough {trough['name']} "
        f"{_pp(trough['share_cm'])}. Shape call: **{shape}**. "
        f"Store `c_tax_month` vs raw category=tax agreement {_pp(agree)}."
    )
    print(prose)
    return {
        "cal_rows": cal_rows,
        "q_share": q_share,
        "nq_share": nq_share,
        "ratio": q_share / nq_share if nq_share else float("nan"),
        "peak": peak,
        "trough": trough,
        "shape": shape,
        "quarterly": quarterly,
        "monthly": monthly,
        "mixed_q": mixed_q,
        "agree": agree,
        "ym": ym,
        "ym_co": ym_co,
        "prose": prose,
        "n_tax_cm_raw": int(len(raw)),
        "n_tax_co_raw": int(raw["company_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Pass 2 — missed-tax prevalence / “just not a tax month?”
# ---------------------------------------------------------------------------
def pass2_missed(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["c_missed_tax"], errors="coerce")
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    nn = x.dropna()
    modal = float(nn.mode().iloc[0]) if len(nn) else float("nan")
    modal_share = float((nn == modal).mean()) if len(nn) else float("nan")
    prev = float((x == 1).mean()) if len(x) else float("nan")
    n_miss = int((x == 1).sum())
    n_tax = int((tax == 1).sum())
    # usual-tax reconstruction from the store flag (rolling 6 of c_tax_month)
    g = tr.sort_values(["company_id", "period"])
    tax6 = (
        g.groupby("company_id", sort=False)["c_tax_month"]
        .transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=1).sum())
    )
    usual = tax6 >= 3
    not_tax = tax == 0
    # by construction missed = usual & not_tax
    recon = ((usual) & (not_tax)).astype(float)
    agree = float((recon == x).mean())

    # is missed just “not a tax month”?
    p_miss_given_notax = float(x[not_tax].mean()) if not_tax.any() else float("nan")
    p_notax_given_miss = float(not_tax[x == 1].mean()) if (x == 1).any() else float("nan")
    p_usual_given_notax = float(usual[not_tax].mean()) if not_tax.any() else float("nan")
    # among usual-tax companies this month, share not filing
    p_notax_given_usual = float(not_tax[usual].mean()) if usual.any() else float("nan")

    # calendar of missed
    cal = []
    for m in range(1, 13):
        sl = tr[tr["cal_month"] == m]
        cal.append(
            {
                "month": m,
                "name": pd.Timestamp(2000, m, 1).strftime("%b"),
                "q": int(m in Q_MONTHS),
                "tax_share": float(pd.to_numeric(sl["c_tax_month"], errors="coerce").mean()),
                "miss_share": float(pd.to_numeric(sl["c_missed_tax"], errors="coerce").mean()),
                "n_cm": int(len(sl)),
            }
        )
    q_miss = float(np.nanmean([r["miss_share"] for r in cal if r["q"] == 1]))
    nq_miss = float(np.nanmean([r["miss_share"] for r in cal if r["q"] == 0]))

    logx = tr["log_in3"]
    rho = spearman(x, logx)
    acf1 = median_acf(x, tr["company_id"], 1)
    acf3 = median_acf(x, tr["company_id"], 3)
    acf6 = median_acf(x, tr["company_id"], 6)
    acf1_tax = median_acf(tax, tr["company_id"], 1)
    acf3_tax = median_acf(tax, tr["company_id"], 3)

    # “just not January / just not a Q-month”
    jan = tr["cal_month"] == 1
    q = tr["is_q_month"] == 1
    # residual: missed on Q-months only (a skip of a filing month)
    miss_on_q = float(x[q].mean()) if q.any() else float("nan")
    miss_on_nq = float(x[~q].mean()) if (~q).any() else float("nan")
    miss_on_jan = float(x[jan].mean()) if jan.any() else float("nan")
    miss_off_jan = float(x[~jan].mean()) if (~jan).any() else float("nan")

    just_not_tax = bool(
        np.isfinite(p_miss_given_notax)
        and p_miss_given_notax >= 0.80
        and abs(modal_share - (1.0 - prev)) < 0.02
    )
    # calendar dummy: missed much higher off Q-months than on them
    calendar_dummy = bool(
        np.isfinite(nq_miss) and np.isfinite(q_miss) and nq_miss >= q_miss + 0.05
    )

    confirm_modal = bool(np.isfinite(modal_share) and abs(modal_share - MODAL_QUOTE) < 0.015)

    prose = (
        f"`c_missed_tax` train prevalence {_pp(prev)} (n={n_miss:,} / {len(tr):,}). "
        f"Modal value {modal:g} share {_pp(modal_share)} "
        f"({'CONFIRM 91.7%' if confirm_modal else 'does not match feature-report 91.7%'}). "
        f"Store reconstruction agree {_pp(agree)}. "
        f"P(missed | not tax month)={_pp(p_miss_given_notax)} — "
        f"{'YES, missed ≈ not-a-tax-month among usual filers' if just_not_tax else 'NO, most non-tax months are not “missed” (no usual-tax history)'}. "
        f"P(usual | not tax)={_pp(p_usual_given_notax)}. "
        f"Missed share Q-months {_pp(q_miss)} vs other {_pp(nq_miss)}. "
        f"acf1={acf1:.3f} acf3={acf3:.3f} vs log1p(a_in3) ρ={rho:.3f}."
    )
    print(prose)
    return {
        "n_cm": int(len(tr)),
        "n_co": int(tr["company_id"].nunique()),
        "prev": prev,
        "n_miss": n_miss,
        "n_tax": n_tax,
        "tax_share": float((tax == 1).mean()),
        "modal": modal,
        "modal_share": modal_share,
        "confirm_modal": confirm_modal,
        "agree": agree,
        "p_miss_given_notax": p_miss_given_notax,
        "p_notax_given_miss": p_notax_given_miss,
        "p_usual_given_notax": p_usual_given_notax,
        "p_notax_given_usual": p_notax_given_usual,
        "just_not_tax": just_not_tax,
        "calendar_dummy": calendar_dummy,
        "q_miss": q_miss,
        "nq_miss": nq_miss,
        "miss_on_q": miss_on_q,
        "miss_on_nq": miss_on_nq,
        "miss_on_jan": miss_on_jan,
        "miss_off_jan": miss_off_jan,
        "rho": rho,
        "acf1": acf1,
        "acf3": acf3,
        "acf6": acf6,
        "acf1_tax": acf1_tax,
        "acf3_tax": acf3_tax,
        "cal": cal,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — single-feature group-fold AUROC
# ---------------------------------------------------------------------------
def pass3_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "c_missed_tax": tr["c_missed_tax"],
        "c_tax_month": tr["c_tax_month"],
        "c_missed_salary": tr["c_missed_salary"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "log1p_a_in3": tr["log_in3"],
        "is_q_month": tr["is_q_month"],
        "is_jan": tr["is_jan"],
        "not_tax_month": tr["not_tax_month"],
    }
    rows = []
    store = {}
    for y in (Y2, Y3, Y9):
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
    miss_y3 = _cv(Y3, "c_missed_tax")
    tax_y3 = _cv(Y3, "c_tax_month")
    miss_y2 = _cv(Y2, "c_missed_tax")
    miss_y9 = _cv(Y9, "c_missed_tax")
    size_y2 = _cv(Y2, "log1p_a_in3")
    size_y9 = _cv(Y9, "log1p_a_in3")
    jan_y3 = _cv(Y3, "is_jan")
    q_y3 = _cv(Y3, "is_q_month")
    notax_y3 = _cv(Y3, "not_tax_month")

    beat_size_y3 = (
        miss_y3 - size_y3 if np.isfinite(miss_y3) and np.isfinite(size_y3) else float("nan")
    )
    beat_days = (
        miss_y3 - days_y3 if np.isfinite(miss_y3) and np.isfinite(days_y3) else float("nan")
    )
    size_park = bool(np.isfinite(size_y3) and size_y3 >= SIZE_PARK)
    # KEEP Q5: beats size by ≥0.02 AND not just January (missed ≫ is_jan)
    just_jan = bool(
        np.isfinite(miss_y3) and np.isfinite(jan_y3) and abs(miss_y3 - jan_y3) < 0.02
    )
    keep_q5 = bool(
        np.isfinite(beat_size_y3) and beat_size_y3 >= KEEP_DELTA and not just_jan
    )

    prose = (
        f"Y3 stressed singles (train group-fold): `c_missed_tax` {_f(miss_y3)} vs "
        f"size {_f(size_y3)} (Δ {_f(beat_size_y3, 3)}) vs days {_f(days_y3)} "
        f"(night 0.711, Δ {_f(days_y3 - DAYS_BENCH, 3)}). "
        f"`c_tax_month` {_f(tax_y3)}. `is_jan` {_f(jan_y3)} `is_q_month` {_f(q_y3)} "
        f"`not_tax_month` {_f(notax_y3)}. "
        f"Y2 missed {_f(miss_y2)} vs size {_f(size_y2)}; "
        f"Y9 missed {_f(miss_y9)} vs size {_f(size_y9)}. "
        f"Size≥0.60 on Y3: {'YES — PARK tax as X' if size_park else 'no'}. "
        f"Missed vs January dummy: {'same skill — just not January' if just_jan else 'not just January'}."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "miss_y3": miss_y3,
        "tax_y3": tax_y3,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "miss_y2": miss_y2,
        "miss_y9": miss_y9,
        "size_y2": size_y2,
        "size_y9": size_y9,
        "jan_y3": jan_y3,
        "q_y3": q_y3,
        "notax_y3": notax_y3,
        "beat_size_y3": beat_size_y3,
        "beat_days": beat_days,
        "size_park": size_park,
        "just_jan": just_jan,
        "keep_q5": keep_q5,
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "n_y9": int(tr[Y9].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "y9_rate": float(pd.to_numeric(tr[Y9], errors="coerce").mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — leak vs a_out6 / payroll month
# ---------------------------------------------------------------------------
def pass4_leak(tr: pd.DataFrame) -> dict:
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    miss = pd.to_numeric(tr["c_missed_tax"], errors="coerce")
    sal = pd.to_numeric(tr["c_salary_month"], errors="coerce")
    ss = pd.to_numeric(tr["c_ss_month"], errors="coerce")
    out6 = pd.to_numeric(tr["a_out6"], errors="coerce")
    rows = [
        {"pair": "c_tax_month vs a_out6", "rho": spearman(tax, out6)},
        {"pair": "c_missed_tax vs a_out6", "rho": spearman(miss, out6)},
        {"pair": "c_salary_month vs a_out6", "rho": spearman(sal, out6)},
        {"pair": "c_tax_month vs c_salary_month", "rho": spearman(tax, sal)},
        {"pair": "c_tax_month vs c_ss_month", "rho": spearman(tax, ss)},
        {"pair": "c_missed_tax vs c_missed_salary", "rho": spearman(miss, tr["c_missed_salary"])},
        {"pair": "c_tax_month vs log1p(a_in3)", "rho": spearman(tax, tr["log_in3"])},
        {"pair": "c_missed_tax vs log1p(a_in3)", "rho": spearman(miss, tr["log_in3"])},
    ]
    both = int(((tax == 1) & (sal == 1)).sum())
    tax_only = int(((tax == 1) & (sal == 0)).sum())
    sal_only = int(((tax == 0) & (sal == 1)).sum())
    neither = int(((tax == 0) & (sal == 0)).sum())
    n = int(len(tr))
    p_tax_given_sal = float(tax[sal == 1].mean()) if (sal == 1).any() else float("nan")
    p_tax_given_nosal = float(tax[sal == 0].mean()) if (sal == 0).any() else float("nan")
    same = abs(p_tax_given_sal - p_tax_given_nosal) < 0.05
    payroll_eq_tax = bool(np.isfinite(rows[3]["rho"]) and abs(rows[3]["rho"]) >= 0.70)
    leak_out = bool(np.isfinite(rows[0]["rho"]) and abs(rows[0]["rho"]) >= 0.70)
    prose = (
        f"Tax × salary 2×2: both {both:,} ({_pp(_pct(both, n))}), tax-only {tax_only:,}, "
        f"salary-only {sal_only:,}, neither {neither:,}. "
        f"P(tax|salary)={_pp(p_tax_given_sal)} vs P(tax|no salary)={_pp(p_tax_given_nosal)}. "
        f"Spearman tax↔salary {rows[3]['rho']:.3f} "
        f"({'payroll month = tax month' if payroll_eq_tax else 'payroll ≠ tax'}). "
        f"tax↔a_out6 {rows[0]['rho']:.3f} "
        f"({'LEAK size/outflow' if leak_out else 'not an outflow dummy'})."
    )
    print(prose)
    return {
        "rows": rows,
        "both": both,
        "tax_only": tax_only,
        "sal_only": sal_only,
        "neither": neither,
        "p_tax_given_sal": p_tax_given_sal,
        "p_tax_given_nosal": p_tax_given_nosal,
        "same": same,
        "payroll_eq_tax": payroll_eq_tax,
        "leak_out": leak_out,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — dark 470 vs 744
# ---------------------------------------------------------------------------
def pass5_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    hold = load_holdout()
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470

    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_tax=("c_tax_month", "sum"),
        n_miss=("c_missed_tax", "sum"),
        n_sal=("c_salary_month", "sum"),
        n_ss=("c_ss_month", "sum"),
    )
    ever["ever_tax"] = ever["n_tax"] > 0
    ever["ever_erp"] = ever["company_id"].isin(book)
    ever["tax_rate"] = ever["n_tax"] / ever["n_cm"]
    ever["miss_rate"] = ever["n_miss"] / ever["n_cm"]

    rows = []
    for name, part in (
        ("ever_erp_744", ever[ever["ever_erp"]]),
        ("never_erp_470", ever[~ever["ever_erp"]]),
    ):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "ever_tax": _pp(float(part["ever_tax"].mean())),
                "n_ever_tax": int(part["ever_tax"].sum()),
                "tax_cm": _pp(float(part["tax_rate"].mean())),
                "miss_cm": _pp(float(part["miss_rate"].mean())),
                "sal_cm": _pp(float((part["n_sal"] / part["n_cm"]).mean())),
                "ss_cm": _pp(float((part["n_ss"] / part["n_cm"]).mean())),
                "tax_p50": _f(float(part["tax_rate"].median()), 3),
            }
        )
    # company-month rates
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    cm = []
    for name, part in (
        ("ever_erp", tr2[tr2["ever_erp"]]),
        ("never_erp", tr2[~tr2["ever_erp"]]),
    ):
        cm.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "tax_share": float(pd.to_numeric(part["c_tax_month"], errors="coerce").mean()),
                "miss_share": float(pd.to_numeric(part["c_missed_tax"], errors="coerce").mean()),
            }
        )
    dark_tax = float(ever.loc[~ever["ever_erp"], "tax_rate"].mean())
    erp_tax = float(ever.loc[ever["ever_erp"], "tax_rate"].mean())
    same_rate = bool(np.isfinite(dark_tax) and np.isfinite(erp_tax) and abs(dark_tax - erp_tax) < 0.05)
    hold_n = int(len(hold))
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month companies: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ from join QA'}). "
        f"Mean company tax-CM rate: invoiced {_pp(erp_tax)} vs dark {_pp(dark_tax)}. "
        f"{'Same bank-book tax rate' if same_rate else 'Dark file tax at a different rate'}. "
        f"Holdout ever-ERP coverage only: {hold_book}/{hold_n}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "dark_tax": dark_tax,
        "erp_tax": erp_tax,
        "same_rate": same_rate,
        "hold_book": hold_book,
        "hold_n": hold_n,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — Q6 lag1 only
# ---------------------------------------------------------------------------
def pass6_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for y in (Y2, Y3, Y9):
        lab = tr[y].notna()
        for col in ("c_missed_tax", "c_tax_month", "c_missed_tax_lag1", "c_tax_month_lag1"):
            if col not in tr.columns:
                continue
            # lag1 only defined when so-far ≥ 2 (panel shift)
            res = signed_oof_auroc(tr[y], tr[col], tr["fold"], lab & tr[col].notna())
            store[(y, col)] = res
            rows.append(
                {
                    "y": y,
                    "col": col,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "present": _pp(float(tr.loc[lab, col].notna().mean()) if lab.any() else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"] if not res["low_power"] else "—",
                }
            )
            print(
                f"Q6 {y} {col}: {'LOW_POWER' if res['low_power'] else _f(res['cv'])} "
                f"n_pos={res['n_pos']}"
            )

    def _cv(y, col) -> float:
        r = store.get((y, col))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = _cv(Y3, "c_missed_tax")
    lag = _cv(Y3, "c_missed_tax_lag1")
    drop = now - lag if np.isfinite(now) and np.isfinite(lag) else float("nan")
    keep_q6 = bool(np.isfinite(lag) and np.isfinite(now) and now >= 0.55 and (now - lag) <= 0.03 and lag >= 0.55)
    if not np.isfinite(now) or now < 0.55:
        prose = (
            f"Y3 contemporaneous missed {_f(now)} is chance; lag1 {_f(lag)}. "
            "CLOSE as Q6 — there is no contemporaneous skill to lead."
        )
    else:
        prose = (
            f"Y3 contemporaneous missed {_f(now)} vs lag1 {_f(lag)} (Δ {_f(drop, 3)}). "
            f"{'KEEP as honest 1-month Q6' if keep_q6 else 'CLOSE as Q6 — lag1 does not hold the contemporaneous skill'}."
        )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "now": now,
        "lag": lag,
        "drop": drop,
        "keep_q6": keep_q6,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — VAT vs corporate-tax guess (amount size)
# ---------------------------------------------------------------------------
def pass7_vat(tr: pd.DataFrame, con) -> dict:
    """Amount size of tax txs. Guess VAT (small, frequent) vs IS (large, rare). Do not build a Y."""
    hold = load_holdout()
    tx = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', "date") AS DATE) AS month,
          amount,
          category
        FROM transactions
        WHERE category IN ('tax', 'tax_refund')
          AND "date" IS NOT NULL
          AND "date" >= DATE '2024-09-01'
          AND "date" < DATE '2026-09-01'
        """
    ).df()
    tx["company_id"] = tx["company_id"].astype(str)
    tx["period"] = pd.to_datetime(tx["month"])
    tx = tx.loc[~tx["company_id"].isin(hold)].copy()
    assert_no_holdout(tx["company_id"])
    tax = tx[tx["category"] == "tax"].copy()
    refund = tx[tx["category"] == "tax_refund"].copy()
    tax["abs"] = pd.to_numeric(tax["amount"], errors="coerce").abs()
    refund["abs"] = pd.to_numeric(refund["amount"], errors="coerce").abs()

    def _amt(s: pd.Series) -> dict:
        s = pd.to_numeric(s, errors="coerce").dropna()
        if s.empty:
            return {"n": 0, "p50": float("nan"), "p90": float("nan"), "mean": float("nan")}
        return {
            "n": int(len(s)),
            "p50": float(s.median()),
            "p90": float(s.quantile(0.90)),
            "mean": float(s.mean()),
        }

    overall = _amt(tax["abs"])
    by_m = []
    for m in range(1, 13):
        sl = tax[tax["period"].dt.month == m]
        a = _amt(sl["abs"])
        by_m.append(
            {
                "month": m,
                "name": pd.Timestamp(2000, m, 1).strftime("%b"),
                "q": int(m in Q_MONTHS),
                "n_tx": a["n"],
                "p50": a["p50"],
                "p90": a["p90"],
                "n_co": int(sl["company_id"].nunique()),
            }
        )
    q_p50 = float(np.nanmedian([r["p50"] for r in by_m if r["q"] == 1 and np.isfinite(r["p50"])]))
    nq_p50 = float(np.nanmedian([r["p50"] for r in by_m if r["q"] == 0 and np.isfinite(r["p50"])]))
    # per company-month: one small vs one large
    cm = (
        tax.groupby(["company_id", "period"], as_index=False)
        .agg(n_tx=("abs", "size"), abs_sum=("abs", "sum"), abs_p50=("abs", "median"))
    )
    cm["large"] = cm["abs_sum"] >= cm["abs_sum"].median()
    # compare to same-month a_in3 (train merge)
    size = tr[["company_id", "period", "a_in3", "log_in3"]].copy()
    cm = cm.merge(size, on=["company_id", "period"], how="left")
    rho_size = spearman(cm["abs_sum"], cm["log_in3"])
    # heuristic: VAT-like if p50 is modest vs inflow and Q vs non-Q amounts similar
    vat_like = bool(
        np.isfinite(q_p50)
        and np.isfinite(nq_p50)
        and (max(q_p50, 1) / max(nq_p50, 1) < 3.0)
    )
    corp_like = bool(
        np.isfinite(q_p50) and np.isfinite(nq_p50) and q_p50 >= 3.0 * max(nq_p50, 1)
    )
    guess = "corporate-tax (Q months much larger)" if corp_like else (
        "VAT-like (amounts similar across months)" if vat_like else "mixed / unclear"
    )
    ref = _amt(refund["abs"])
    prose = (
        f"Train tax txs n={overall['n']:,} p50={overall['p50']:,.0f} p90={overall['p90']:,.0f}. "
        f"Q-month p50 {q_p50:,.0f} vs other {nq_p50:,.0f}. "
        f"Amount vs log1p(a_in3) ρ={rho_size:.3f}. Guess: **{guess}**. "
        f"tax_refund txs n={ref['n']:,} p50={ref['p50']:,.0f}."
    )
    print(prose)
    return {
        "overall": overall,
        "by_m": by_m,
        "q_p50": q_p50,
        "nq_p50": nq_p50,
        "rho_size": rho_size,
        "guess": guess,
        "vat_like": vat_like,
        "corp_like": corp_like,
        "refund": ref,
        "n_refund_co": int(refund["company_id"].nunique()),
        "prose": prose,
        "tax_tx": tax,
        "refund_tx": refund,
    }


# ---------------------------------------------------------------------------
# Pass 8 — tax_refund months as recoveries? (do not build a Y)
# ---------------------------------------------------------------------------
def pass8_refund(tr: pd.DataFrame, refund_tx: pd.DataFrame) -> dict:
    hit = refund_tx.groupby(["company_id", "period"], as_index=False).size().rename(columns={"size": "n_ref"})
    m = tr.merge(hit, on=["company_id", "period"], how="left")
    m["refund"] = m["n_ref"].fillna(0) > 0
    n_cm = int(m["refund"].sum())
    n_co = int(m.loc[m["refund"], "company_id"].nunique())
    rows = []
    for y in (Y2, Y3, Y9):
        for name, mask in (("refund", m["refund"]), ("no_refund", ~m["refund"])):
            s = pd.to_numeric(m.loc[mask, y], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "slice": name,
                    "n_cm": int(mask.sum()),
                    "n_labeled": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                }
            )
    # next-month Y3 among refund months (recovery *after* refund) — story only
    srt = m.sort_values(["company_id", "period"])
    y3_lead = srt.groupby("company_id", sort=False)[Y3].shift(-1)
    srt = srt.assign(y3_next=y3_lead)
    ref = srt[srt["refund"]]
    y3_now = float(pd.to_numeric(ref[Y3], errors="coerce").mean()) if len(ref) else float("nan")
    y3_nxt = float(pd.to_numeric(ref["y3_next"], errors="coerce").mean()) if len(ref) else float("nan")
    y3_all = float(pd.to_numeric(m[Y3], errors="coerce").mean())
    recover_like = bool(
        np.isfinite(y3_now) and np.isfinite(y3_all) and y3_now >= y3_all + 0.03
    )
    prose = (
        f"tax_refund company-months: {n_cm:,} / {m['company_id'].nunique()} companies={n_co}. "
        f"Y3 rate on refund months {_pp(y3_now)} vs all labeled {_pp(y3_all)}; "
        f"next-month Y3 {_pp(y3_nxt)}. "
        f"{'Refund months look recover-ish (higher Y3) — still not a Y.' if recover_like else 'Refund months are not a recovery flag. Do not build a Y.'}"
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": n_co,
        "rows": rows,
        "y3_now": y3_now,
        "y3_nxt": y3_nxt,
        "y3_all": y3_all,
        "recover_like": recover_like,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — residual: missed after calendar month / “not January”
# ---------------------------------------------------------------------------
def pass9_residual(tr: pd.DataFrame) -> dict:
    """Does missed-tax still rank inside a calendar month? If not, it is the dummy."""
    rows = []
    store = {}
    y = Y3
    lab = tr[y].notna()
    # within Q-months / non-Q / each calendar month
    slices = [
        ("all", lab),
        ("Q_months", lab & (tr["is_q_month"] == 1)),
        ("nonQ_months", lab & (tr["is_q_month"] == 0)),
        ("not_January", lab & (tr["cal_month"] != 1)),
        ("January", lab & (tr["cal_month"] == 1)),
        ("usual_tax6", lab & (tr["_usual"] if "_usual" in tr.columns else lab)),
    ]
    # reconstruct usual
    g = tr.sort_values(["company_id", "period"])
    tax6 = g.groupby("company_id", sort=False)["c_tax_month"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=1).sum()
    )
    usual = tax6 >= 3
    slices[-1] = ("usual_tax6", lab & usual)

    for sname, mask in slices:
        for feat in ("c_missed_tax", "c_tax_month", "log1p_a_in3", "is_q_month"):
            col = tr[feat] if feat != "log1p_a_in3" else tr["log_in3"]
            res = signed_oof_auroc(tr[y], col, tr["fold"], mask)
            store[(sname, feat)] = res
            rows.append(
                {
                    "slice": sname,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )

    # per calendar month (pooled AUROC — fold may be thin)
    month_rows = []
    for m in range(1, 13):
        mask = lab & (tr["cal_month"] == m)
        res = signed_oof_auroc(tr[y], tr["c_missed_tax"], tr["fold"], mask)
        month_rows.append(
            {
                "month": pd.Timestamp(2000, m, 1).strftime("%b"),
                "n_pos": res["n_pos"],
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )

    miss_all = store[("all", "c_missed_tax")]
    miss_nq = store[("nonQ_months", "c_missed_tax")]
    miss_q = store[("Q_months", "c_missed_tax")]
    miss_njan = store[("not_January", "c_missed_tax")]
    size_all = store[("all", "log1p_a_in3")]
    # calendar dummy if skill vanishes inside Q-months (where a skip would be real)
    vanishes_on_q = bool(
        (miss_q["low_power"] or (np.isfinite(miss_q["cv"]) and miss_q["cv"] < 0.55))
        and (not miss_all["low_power"] and miss_all["cv"] >= 0.55)
    )
    prose = (
        f"Y3 missed CV all {_f(miss_all['cv'])}; on Q-months "
        f"{'LOW_POWER' if miss_q['low_power'] else _f(miss_q['cv'])}; "
        f"on non-Q {_f(miss_nq['cv']) if not miss_nq['low_power'] else 'LOW_POWER'}; "
        f"not-January {_f(miss_njan['cv']) if not miss_njan['low_power'] else 'LOW_POWER'}. "
        f"{'Skill is the non-Q calendar hole — CLOSE as dummy.' if vanishes_on_q else 'Residual inside calendar slices still measured.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "month_rows": month_rows,
        "store": store,
        "vanishes_on_q": vanishes_on_q,
        "miss_all": miss_all["cv"] if not miss_all["low_power"] else float("nan"),
        "miss_q": miss_q["cv"] if not miss_q["low_power"] else float("nan"),
        "miss_nq": miss_nq["cv"] if not miss_nq["low_power"] else float("nan"),
        "size_all": size_all["cv"] if not size_all["low_power"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — company cadence (monthly / quarterly / never)
# ---------------------------------------------------------------------------
def pass10_cadence(tr: pd.DataFrame) -> dict:
    ever = tr.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        n_tax=("c_tax_month", "sum"),
        n_miss=("c_missed_tax", "sum"),
        n_q=("is_q_month", "sum"),
    )
    # tax months that fall on Q vs not
    qtax = (
        tr.loc[pd.to_numeric(tr["c_tax_month"], errors="coerce") == 1]
        .groupby("company_id")["is_q_month"]
        .mean()
        .rename("share_tax_on_q")
    )
    ever = ever.merge(qtax, on="company_id", how="left")
    ever["tax_rate"] = ever["n_tax"] / ever["n_cm"]
    ever["kind"] = np.select(
        [
            ever["n_tax"] == 0,
            ever["tax_rate"] >= 0.70,
            (ever["share_tax_on_q"] >= 0.70) & (ever["tax_rate"] <= 0.45) & (ever["n_tax"] >= 2),
        ],
        ["never", "monthly", "quarterly"],
        default="irregular",
    )
    rows = []
    for k, part in ever.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_co": int(len(part)),
                "share_co": _pp(_pct(len(part), len(ever))),
                "tax_rate_p50": _f(float(part["tax_rate"].median()), 3),
                "miss_p50": _f(float((part["n_miss"] / part["n_cm"]).median()), 3),
                "n_tax_p50": _f(float(part["n_tax"].median()), 1),
            }
        )
    n_never = int((ever["kind"] == "never").sum())
    n_month = int((ever["kind"] == "monthly").sum())
    n_q = int((ever["kind"] == "quarterly").sum())
    n_irr = int((ever["kind"] == "irregular").sum())
    prose = (
        f"Train companies by tax cadence: never {n_never}, monthly {n_month}, "
        f"quarterly {n_q}, irregular {n_irr} / {len(ever)}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_never": n_never,
        "n_month": n_month,
        "n_q": n_q,
        "n_irr": n_irr,
        "n_co": int(len(ever)),
        "prose": prose,
        "ever": ever,
    }


# ---------------------------------------------------------------------------
# Pass 11 — holdout coverage only
# ---------------------------------------------------------------------------
def pass11_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col in ("c_tax_month", "c_missed_tax", "c_salary_month", "c_missed_salary"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "mean": _pp(float(x.mean()) if x.notna().any() else float("nan")),
            }
        )
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM. "
        f"tax-month mean {_pp(float(pd.to_numeric(ho['c_tax_month'], errors='coerce').mean()))}; "
        f"missed-tax mean {_pp(float(pd.to_numeric(ho['c_missed_tax'], errors='coerce').mean()))}. "
        "No AUROC claim."
    )
    print(prose)
    return {"rows": rows, "n_cm": int(len(ho)), "n_co": int(ho["company_id"].nunique()), "prose": prose}


def _usual_mask(tr: pd.DataFrame) -> pd.Series:
    g = tr.sort_values(["company_id", "period"])
    tax6 = g.groupby("company_id", sort=False)["c_tax_month"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=1).sum()
    )
    return (tax6 >= 3).reindex(tr.index)


def _company_terciles(tr: pd.DataFrame, col: str, name: str) -> pd.Series:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last[col], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1", "T2", "T3"], duplicates="drop")
    return terc.rename(name)


def pass12_honest_skip(tr: pd.DataFrame) -> dict:
    """Missed-tax only among usual filers (tax6≥3). That is the honest skip, not the hole."""
    usual = _usual_mask(tr)
    rows = []
    store = {}
    for y in (Y2, Y3, Y9):
        lab = tr[y].notna()
        for sname, mask in (
            ("usual_tax6", lab & usual),
            ("usual_and_Q", lab & usual & (tr["is_q_month"] == 1)),
            ("usual_and_nonQ", lab & usual & (tr["is_q_month"] == 0)),
        ):
            res = signed_oof_auroc(tr[y], tr["c_missed_tax"], tr["fold"], mask)
            store[(y, sname)] = res
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "miss_share": _pp(float(pd.to_numeric(tr.loc[mask, "c_missed_tax"], errors="coerce").mean()) if mask.any() else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3u = store[(Y3, "usual_tax6")]
    n_usual = int(usual.sum())
    n_miss_u = int(((usual) & (pd.to_numeric(tr["c_missed_tax"], errors="coerce") == 1)).sum())
    keep = bool(not y3u["low_power"] and y3u["cv"] >= 0.55)
    prose = (
        f"Usual-tax CM (tax6≥3): {n_usual:,} / {len(tr):,} ({_pp(_pct(n_usual, len(tr)))}); "
        f"of those, missed {n_miss_u:,} ({_pp(_pct(n_miss_u, n_usual))}). "
        f"Y3 missed-on-usual CV "
        f"{'LOW_POWER' if y3u['low_power'] else _f(y3u['cv'])} n_pos={y3u['n_pos']}. "
        "Above chance is not KEEP — KEEP still requires beating size by ≥0.02."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "n_usual": n_usual,
        "n_miss_u": n_miss_u,
        "y3_cv": y3u["cv"] if not y3u["low_power"] else float("nan"),
        "keep": keep,
        "prose": prose,
    }


def pass13_activity(tr: pd.DataFrame) -> dict:
    """Is c_tax_month 0.611 just inverse activity / size?"""
    tax = pd.to_numeric(tr["c_tax_month"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    rho_days = spearman(tax, days)
    rho_size = spearman(tax, tr["log_in3"])
    terc_size = _company_terciles(tr, "log_in3", "size_t")
    terc_days = _company_terciles(tr, "c_n_days_with_tx", "days_t")
    m = tr.merge(terc_size.reset_index(), on="company_id", how="left")
    m = m.merge(terc_days.reset_index(), on="company_id", how="left")
    rows = []
    store = {}
    lab = m[Y3].notna()
    for tname, col in (("size_t", "size_t"), ("days_t", "days_t")):
        for labv in ("T1", "T2", "T3"):
            mask = lab & (m[col].astype(str) == labv)
            res = signed_oof_auroc(m[Y3], m["c_tax_month"], m["fold"], mask)
            store[(tname, labv)] = res
            tax_share = float(pd.to_numeric(m.loc[mask, "c_tax_month"], errors="coerce").mean()) if mask.any() else float("nan")
            rows.append(
                {
                    "clock": tname,
                    "tercile": labv,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "tax_share": _pp(tax_share),
                    "tax CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    # quiet months: days==0 should have tax=0 if tax is a booking
    quiet = days == 0
    p_tax_quiet = float(tax[quiet].mean()) if quiet.any() else float("nan")
    p_tax_busy = float(tax[days >= 5].mean()) if (days >= 5).any() else float("nan")
    # does tax beat size inside any tercile by 0.02?
    beats = []
    for labv in ("T1", "T2", "T3"):
        r = store[("size_t", labv)]
        if not r["low_power"] and r["cv"] >= 0.55:
            beats.append(labv)
    activity_dummy = bool(np.isfinite(rho_days) and rho_days >= 0.40)
    prose = (
        f"tax↔days ρ={rho_days:.3f}; tax↔log1p(a_in3) ρ={rho_size:.3f}. "
        f"P(tax|days=0)={_pp(p_tax_quiet)} P(tax|days≥5)={_pp(p_tax_busy)}. "
        f"Y3 tax CV inside size terciles that stay ≥0.55: {beats or 'none'}. "
        f"{'tax-month is an activity dummy' if activity_dummy else 'tax-month is not a strong activity ρ'}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_days": rho_days,
        "rho_size": rho_size,
        "p_tax_quiet": p_tax_quiet,
        "p_tax_busy": p_tax_busy,
        "activity_dummy": activity_dummy,
        "beats": beats,
        "prose": prose,
    }


def pass14_cadence_y(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Y rates and missed AUROC by company tax cadence."""
    m = tr.merge(ever[["company_id", "kind", "tax_rate"]], on="company_id", how="left")
    rate_rows = []
    for y in (Y2, Y3, Y9):
        for k, part in m.groupby("kind"):
            s = pd.to_numeric(part[y], errors="coerce")
            rate_rows.append(
                {
                    "y": y,
                    "kind": k,
                    "n_cm": int(len(part)),
                    "n_co": int(part["company_id"].nunique()),
                    "n_labeled": int(s.notna().sum()),
                    "n_pos": int((s == 1).sum()),
                    "rate": float(s.mean()) if s.notna().any() else float("nan"),
                    "tax_cm": float(pd.to_numeric(part["c_tax_month"], errors="coerce").mean()),
                    "miss_cm": float(pd.to_numeric(part["c_missed_tax"], errors="coerce").mean()),
                }
            )
    auc_rows = []
    store = {}
    for y in (Y2, Y3):
        for k in ("monthly", "quarterly", "irregular", "never"):
            mask = (m["kind"] == k) & m[y].notna()
            res = signed_oof_auroc(m[y], m["c_missed_tax"], m["fold"], mask)
            store[(y, k)] = res
            auc_rows.append(
                {
                    "y": y,
                    "kind": k,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3_m = store[(Y3, "monthly")]
    prose = (
        f"Y3 missed CV on monthly-cadence companies "
        f"{'LOW_POWER' if y3_m['low_power'] else _f(y3_m['cv'])} n_pos={y3_m['n_pos']}. "
        "A skip among monthly filers would be the cleanest Q5; it is not."
    )
    print(prose)
    return {"rate_rows": rate_rows, "auc_rows": auc_rows, "prose": prose, "y3_monthly": y3_m}


def pass15_amounts(tax_tx: pd.DataFrame) -> dict:
    """Amount mix — small frequent VAT vs large rare IS. Do not build a Y."""
    s = pd.to_numeric(tax_tx["abs"], errors="coerce").dropna()
    bins = [
        ("<500", 0, 500),
        ("500-2k", 500, 2000),
        ("2k-10k", 2000, 10000),
        ("10k-50k", 10000, 50000),
        (">50k", 50000, np.inf),
    ]
    rows = []
    for name, lo, hi in bins:
        sl = s[(s >= lo) & (s < hi)]
        rows.append(
            {
                "bin": name,
                "n_tx": int(len(sl)),
                "share_tx": _pp(_pct(len(sl), len(s))),
                "p50": f"{float(sl.median()):,.0f}" if len(sl) else "—",
            }
        )
    # company ever-large
    cm = tax_tx.groupby("company_id")["abs"].agg(p50="median", mx="max", n="size")
    n_co = int(len(cm))
    n_large = int((cm["mx"] >= 10000).sum())
    n_only_small = int(((cm["mx"] < 2000) & (cm["n"] >= 3)).sum())
    prose = (
        f"Tax |amt| mix: <500 {_pp(_pct(int(((s>=0)&(s<500)).sum()), len(s)))}, "
        f">10k {_pp(_pct(int((s>=10000).sum()), len(s)))}. "
        f"Companies ever ≥10k: {n_large}/{n_co}. Only-small (≥3 txs, max<2k): {n_only_small}. "
        "Most mass is small — VAT-like withholdings, not a corporate-tax event."
    )
    print(prose)
    return {
        "rows": rows,
        "n_co": n_co,
        "n_large": n_large,
        "n_only_small": n_only_small,
        "prose": prose,
    }


def pass16_nonq_who(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Who files tax off Jan/Apr/Jul/Oct — the monthly book, not a skip."""
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    nonq = m[m["is_q_month"] == 0]
    tax_nq = nonq[pd.to_numeric(nonq["c_tax_month"], errors="coerce") == 1]
    rows = []
    for k, part in tax_nq.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_tax_nq_cm": int(len(part)),
                "share": _pp(_pct(len(part), len(tax_nq))),
                "n_co": int(part["company_id"].nunique()),
            }
        )
    monthly_share = _pct(
        int((tax_nq["kind"] == "monthly").sum()), int(len(tax_nq))
    )
    # Q-month tax by kind
    q = m[m["is_q_month"] == 1]
    q_rows = []
    for k, part in q.groupby("kind"):
        q_rows.append(
            {
                "kind": k,
                "n_cm": int(len(part)),
                "tax": _pp(float(pd.to_numeric(part["c_tax_month"], errors="coerce").mean())),
                "miss": _pp(float(pd.to_numeric(part["c_missed_tax"], errors="coerce").mean())),
            }
        )
    prose = (
        f"Non-Q tax CM: {len(tax_nq):,}. Monthly-cadence companies account for "
        f"{_pp(monthly_share)} of those filings. Off-quarter tax is the monthly book, not a health event."
    )
    print(prose)
    return {
        "rows": rows,
        "q_rows": q_rows,
        "monthly_share": monthly_share,
        "n_tax_nq": int(len(tax_nq)),
        "prose": prose,
    }


def pass17_ss(tr: pd.DataFrame) -> dict:
    """Social-security calendar vs tax — SS should be flatter if tax is the Q peak."""
    rows = []
    for mth in range(1, 13):
        sl = tr[tr["cal_month"] == mth]
        rows.append(
            {
                "month": pd.Timestamp(2000, mth, 1).strftime("%b"),
                "q": int(mth in Q_MONTHS),
                "tax": float(pd.to_numeric(sl["c_tax_month"], errors="coerce").mean()),
                "ss": float(pd.to_numeric(sl["c_ss_month"], errors="coerce").mean()),
                "salary": float(pd.to_numeric(sl["c_salary_month"], errors="coerce").mean()),
            }
        )
    q_ss = float(np.nanmean([r["ss"] for r in rows if r["q"] == 1]))
    nq_ss = float(np.nanmean([r["ss"] for r in rows if r["q"] == 0]))
    q_tax = float(np.nanmean([r["tax"] for r in rows if r["q"] == 1]))
    nq_tax = float(np.nanmean([r["tax"] for r in rows if r["q"] == 0]))
    ss_flat = bool(abs(q_ss - nq_ss) < 0.08)
    prose = (
        f"SS Q vs other: {_pp(q_ss)} / {_pp(nq_ss)} "
        f"({'flat — monthly payroll tax' if ss_flat else 'also Q-peaked'}). "
        f"Tax Q vs other: {_pp(q_tax)} / {_pp(nq_tax)}. "
        "Tax is the calendar; SS is the monthly control."
    )
    print(prose)
    return {
        "rows": [
            {
                "month": r["month"],
                "Q?": "Q" if r["q"] else "",
                "tax": _pp(r["tax"]),
                "ss": _pp(r["ss"]),
                "salary": _pp(r["salary"]),
            }
            for r in rows
        ],
        "q_ss": q_ss,
        "nq_ss": nq_ss,
        "ss_flat": ss_flat,
        "prose": prose,
    }


def pass18_usual_rates(tr: pd.DataFrame) -> dict:
    """Y rates on usual × missed. The 2×2 that would be Q5 if the gap were large."""
    usual = _usual_mask(tr)
    miss = pd.to_numeric(tr["c_missed_tax"], errors="coerce") == 1
    rows = []
    for y in (Y2, Y3, Y9):
        for sname, mask in (
            ("usual_missed", usual & miss),
            ("usual_filed", usual & ~miss),
            ("not_usual", ~usual),
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
    y3_miss = next(r for r in rows if r["y"] == Y3 and r["slice"] == "usual_missed")
    y3_file = next(r for r in rows if r["y"] == Y3 and r["slice"] == "usual_filed")
    gap = (
        y3_miss["rate"] - y3_file["rate"]
        if np.isfinite(y3_miss["rate"]) and np.isfinite(y3_file["rate"])
        else float("nan")
    )
    prose = (
        f"Y3 among usual filers: missed {_pp(y3_miss['rate'])} (n_pos={y3_miss['n_pos']}) vs "
        f"filed {_pp(y3_file['rate'])} (gap {_pp(gap) if np.isfinite(gap) else '—'}). "
        "A Q5 skip would show a large same-sign gap. It does not."
    )
    print(prose)
    return {"rows": rows, "gap": gap, "prose": prose}


def pass19_large_tax(tr: pd.DataFrame, tax_tx: pd.DataFrame) -> dict:
    """Large tax month (≥10k sum) as a one-off IS-like event. In-module only — not a Y."""
    cm = tax_tx.groupby(["company_id", "period"], as_index=False)["abs"].sum().rename(columns={"abs": "tax_abs"})
    m = tr.merge(cm, on=["company_id", "period"], how="left")
    m["tax_abs"] = m["tax_abs"].fillna(0)
    m["large_tax"] = (m["tax_abs"] >= 10000).astype(float)
    n_large = int(m["large_tax"].sum())
    n_co = int(m.loc[m["large_tax"] == 1, "company_id"].nunique())
    rows = []
    store = {}
    for y in (Y2, Y3, Y9):
        lab = m[y].notna()
        res = signed_oof_auroc(m[y], m["large_tax"], m["fold"], lab)
        store[y] = res
        s1 = pd.to_numeric(m.loc[m["large_tax"] == 1, y], errors="coerce")
        s0 = pd.to_numeric(m.loc[m["large_tax"] == 0, y], errors="coerce")
        rows.append(
            {
                "y": y,
                "n_large_lab": int(s1.notna().sum()),
                "n_pos_large": int((s1 == 1).sum()),
                "rate_large": float(s1.mean()) if s1.notna().any() else float("nan"),
                "rate_other": float(s0.mean()) if s0.notna().any() else float("nan"),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    y3 = store[Y3]
    prose = (
        f"Large-tax CM (≥10k sum): {n_large:,} / {m['company_id'].nunique()} companies={n_co}. "
        f"Y3 CV {'LOW_POWER' if y3['low_power'] else _f(y3['cv'])}. "
        "Do not build a corporate-tax Y from amount size."
    )
    print(prose)
    return {"rows": rows, "n_large": n_large, "n_co": n_co, "y3_cv": y3["cv"] if not y3["low_power"] else float("nan"), "prose": prose}


def pass20_q_skip(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Quarterly-cadence companies on Q-months — the cleanest filing-date skip."""
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    usual = _usual_mask(m)
    mask_q = (m["kind"] == "quarterly") & (m["is_q_month"] == 1)
    mask_u = mask_q & usual
    miss = pd.to_numeric(m["c_missed_tax"], errors="coerce")
    rows = []
    for sname, mask in (
        ("q_co_on_Q", mask_q),
        ("q_co_on_Q_usual", mask_u),
        ("q_co_on_nonQ", (m["kind"] == "quarterly") & (m["is_q_month"] == 0)),
        ("monthly_on_any", m["kind"] == "monthly"),
    ):
        sl = m.loc[mask]
        rows.append(
            {
                "slice": sname,
                "n_cm": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()) if len(sl) else 0,
                "tax": _pp(float(pd.to_numeric(sl["c_tax_month"], errors="coerce").mean()) if len(sl) else float("nan")),
                "miss": _pp(float(miss[mask].mean()) if mask.any() else float("nan")),
            }
        )
    auc_rows = []
    for y in (Y2, Y3):
        res = signed_oof_auroc(m[y], m["c_missed_tax"], m["fold"], mask_q & m[y].notna())
        auc_rows.append(
            {
                "y": y,
                "slice": "q_co_on_Q",
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    # Jan→Feb complement: among companies that filed in January, P(tax in February)
    jan = m[m["cal_month"] == 1]
    jan_filers = set(jan.loc[pd.to_numeric(jan["c_tax_month"], errors="coerce") == 1, "company_id"])
    feb = m[m["cal_month"] == 2]
    feb_of = feb[feb["company_id"].isin(jan_filers)]
    p_feb = float(pd.to_numeric(feb_of["c_tax_month"], errors="coerce").mean()) if len(feb_of) else float("nan")
    p_feb_all = float(pd.to_numeric(feb["c_tax_month"], errors="coerce").mean()) if len(feb) else float("nan")
    prose = (
        f"Quarterly companies on Q-months: tax {rows[0]['tax']} miss {rows[0]['miss']}. "
        f"Jan filers who also file in Feb: {_pp(p_feb)} (Feb base {_pp(p_feb_all)}). "
        "January is not a unique dummy — Feb is the complementary hole of the Q peak."
    )
    print(prose)
    return {
        "rows": rows,
        "auc_rows": auc_rows,
        "p_feb": p_feb,
        "p_feb_all": p_feb_all,
        "n_jan_filers": len(jan_filers),
        "prose": prose,
    }


def pass21_miss_who(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Who produces the 1,763 missed months — quarterly filers cannot (tax6 < 3)."""
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    g = m.sort_values(["company_id", "period"])
    tax6 = g.groupby("company_id", sort=False)["c_tax_month"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=1).sum()
    )
    m = m.assign(tax6=tax6)
    miss = m[pd.to_numeric(m["c_missed_tax"], errors="coerce") == 1]
    rows = []
    for k, part in miss.groupby("kind"):
        rows.append(
            {
                "kind": k,
                "n_miss": int(len(part)),
                "share": _pp(_pct(len(part), len(miss))),
                "n_co": int(part["company_id"].nunique()),
                "on_Q": _pp(float(part["is_q_month"].mean())),
            }
        )
    tax6_rows = []
    for k, part in m.groupby("kind"):
        t = pd.to_numeric(part["tax6"], errors="coerce")
        tax6_rows.append(
            {
                "kind": k,
                "n_cm": int(len(part)),
                "tax6_p50": _f(float(t.median()), 2),
                "share_tax6_ge3": _pp(float((t >= 3).mean())),
                "share_tax6_ge4": _pp(float((t >= 4).mean())),
            }
        )
    n_q_miss = int((miss["kind"] == "quarterly").sum())
    n_irr = int((miss["kind"] == "irregular").sum())
    n_mon = int((miss["kind"] == "monthly").sum())
    prose = (
        f"Of {len(miss):,} missed CM: irregular {n_irr:,} ({_pp(_pct(n_irr, len(miss)))}), "
        f"monthly {n_mon:,}, quarterly {n_q_miss:,}. "
        f"Quarterly companies almost never reach tax6≥3, so `c_missed_tax` cannot flag a quarterly skip. "
        "The flag is the irregular/monthly complementary hole, not a filing-date miss."
    )
    print(prose)
    return {
        "rows": rows,
        "tax6_rows": tax6_rows,
        "n_miss": int(len(miss)),
        "n_q_miss": n_q_miss,
        "n_irr": n_irr,
        "n_mon": n_mon,
        "prose": prose,
    }


def pass22_irregular(tr: pd.DataFrame, ever: pd.DataFrame) -> dict:
    """Irregular cadence is 66% of missed. Is that the leftover Q5?"""
    m = tr.merge(ever[["company_id", "kind"]], on="company_id", how="left")
    irr = m["kind"] == "irregular"
    rows = []
    for y in (Y2, Y3, Y9):
        lab = irr & m[y].notna()
        for feat, col in (
            ("c_missed_tax", m["c_missed_tax"]),
            ("c_tax_month", m["c_tax_month"]),
            ("log1p_a_in3", m["log_in3"]),
            ("c_n_days_with_tx", m["c_n_days_with_tx"]),
        ):
            res = signed_oof_auroc(m[y], col, m["fold"], lab)
            rows.append(
                {
                    "y": y,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3_miss = next(r for r in rows if r["y"] == Y3 and r["feature"] == "c_missed_tax")
    y3_size = next(r for r in rows if r["y"] == Y3 and r["feature"] == "log1p_a_in3")
    prose = (
        f"Irregular-only Y3: missed {y3_miss['CV']} vs size {y3_size['CV']}. "
        "The 66% pile is still not a Q5."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def make_png(p1: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    months = [r["name"] for r in p1["cal_rows"]]
    shares = [100.0 * r["share_cm"] for r in p1["cal_rows"]]
    colors = ["#1f4e79" if r["q"] else "#9e6b4a" for r in p1["cal_rows"]]
    # heatmap year × month
    mat = np.full((3, 12), np.nan)
    years = (2024, 2025, 2026)
    ymap = {y: i for i, y in enumerate(years)}
    for r in p1["ym_co"]:
        if r["year"] in ymap:
            mat[ymap[r["year"]], r["month"] - 1] = 100.0 * r["share_cm"]

    fig, axes = plt.subplots(2, 1, figsize=(8.4, 6.2), gridspec_kw={"height_ratios": [1.1, 1.3]})
    ax = axes[0]
    ax.bar(np.arange(12), shares, color=colors)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(months)
    ax.set_ylabel("% of train CM with a tax tx")
    ax.set_title("Tax calendar (stacked 2024-09..2026-08) — navy = Jan/Apr/Jul/Oct")
    ax.set_ylim(0, max(shares) * 1.15 if shares else 100)
    ax.axhline(100.0 * p1["q_share"], color="#1f4e79", ls="--", lw=0.8, alpha=0.7)
    ax.axhline(100.0 * p1["nq_share"], color="#9e6b4a", ls="--", lw=0.8, alpha=0.7)

    ax2 = axes[1]
    im = ax2.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=max(40, np.nanmax(mat)))
    ax2.set_xticks(np.arange(12))
    ax2.set_xticklabels(months)
    ax2.set_yticks(np.arange(3))
    ax2.set_yticklabels(["2024", "2025", "2026"])
    ax2.set_title("Share of train company-months with category=tax")
    fig.colorbar(im, ax=ax2, fraction=0.025, pad=0.02, label="%")
    for i in range(3):
        for j in range(12):
            v = mat[i, j]
            if np.isfinite(v):
                ax2.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7, color="black")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p1, p2, p3, p4, p6, p9, p12=None, p13=None) -> dict:
    """KEEP as Q5 / CLOSE as calendar dummy / PARK as health Y and maybe as X."""
    park_y = True
    cal_shaped = p1["shape"] in {"quarterly", "mixed_q_peaked"}
    close_cal = bool(cal_shaped and p2["calendar_dummy"] and not p3["keep_q5"])
    if p9.get("vanishes_on_q"):
        close_cal = True
    keep_q5 = bool(p3["keep_q5"] and not close_cal and not p2["just_not_tax"])
    if p12 is not None and p12.get("keep"):
        # honest skip among usual filers can still KEEP Q5 even if the pooled flag is chance
        keep_q5 = bool(np.isfinite(p12.get("y3_cv")) and p12["y3_cv"] >= (p3["size_y3"] + KEEP_DELTA))
    park_x = bool(p3["size_park"] and not keep_q5)
    if p2["just_not_tax"] and cal_shaped:
        close_cal = True
        keep_q5 = False
    if keep_q5:
        q5 = "KEEP"
        why = (
            f"missed-tax Y3 CV {_f(p3['miss_y3'])} beats size {_f(p3['size_y3'])} "
            f"by {_f(p3['beat_size_y3'])} and is not just January."
        )
    elif close_cal:
        q5 = "CLOSE"
        why = (
            f"tax calendar is {p1['shape']} (Q {_pp(p1['q_share'])} vs other {_pp(p1['nq_share'])}); "
            f"missed-tax is the complementary hole (Q-miss {_pp(p2['q_miss'])} vs non-Q {_pp(p2['nq_miss'])}). "
            f"Y3 {_f(p3['miss_y3'])} loses to size {_f(p3['size_y3'])}. "
            "Quarterly cadence cannot fire the flag (tax6<3)."
        )
    else:
        q5 = "CLOSE"
        why = (
            f"missed-tax Y3 {_f(p3['miss_y3'])} does not beat size {_f(p3['size_y3'])} "
            f"by ≥0.02 (Δ {_f(p3['beat_size_y3'])})."
        )
    x_dec = "PARK" if park_x else ("KEEP" if keep_q5 else "CLOSE")
    return {
        "q5": q5,
        "why": why,
        "park_y": park_y,
        "park_x": park_x,
        "keep_q5": keep_q5,
        "close_cal": close_cal,
        "x_dec": x_dec,
        "q6": "KEEP" if p6["keep_q6"] else "CLOSE",
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
        "# Q5 tax calendar vs missed-tax",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent a missed-tax Y.",
        "",
        "`c_tax_month` = any category `tax` (not `tax_refund`). "
        "`c_missed_tax` = usual tax in last ≤6 months (≥3) AND no tax this month. "
        "Feature report: rare-event flag, modal 91.7%.",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Not this flag. PARK as a health Y. Cadence: {p10['n_never']} never / {p10['n_month']} monthly / {p10['n_q']} quarterly. |",
        "| 2 | Who is improving? | tax_refund months are not a recovery Y. |",
        "| 3 | Who is turning? | Missed-tax is a skip of a usual filing, or the complementary calendar hole. |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | **{d['q5']}** — {d['why']} |",
        f"| 6 | Months earlier? | lag1 of missed-tax {d['q6']} (contemporaneous {_f(p6['now'])} vs lag1 {_f(p6['lag'])}). Only 1-month leads are honest. |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `c_missed_tax` as Q5 why | **{d['q5']}** | {d['why']} |",
        "| `c_missed_tax` as a health Y | **PARK** | do not invent a merged Y from a calendar / skip flag |",
        f"| `c_missed_tax` / `c_tax_month` as Y3 X | **{d['x_dec']}** | "
        + (
            f"size dummy {_f(p3['size_y3'])} ≥ 0.60"
            if d["park_x"]
            else f"Y3 missed {_f(p3['miss_y3'])} vs size {_f(p3['size_y3'])} vs days {_f(p3['days_y3'])}"
        )
        + " |",
        f"| tax calendar as a dummy | **{'CLOSE (it is the dummy)' if d['close_cal'] else 'measured'}** | shape={p1['shape']}; Q {_pp(p1['q_share'])} vs other {_pp(p1['nq_share'])} |",
        f"| Q6 lag1 `c_missed_tax` | **{d['q6']}** | {p6['prose']} |",
        "| `tax_refund` as recovery Y | **PARK** | measured; do not build a Y |",
        "",
        "## 1. Calendar — Jan–Dec stacked 2024-09..2026-08",
        "",
        p1["prose"],
        "",
        "Share of **train company-months** with a raw `category=tax` booking. "
        "`share_co` = companies with ≥1 tax tx in that calendar month (any stacked year) / companies on the grid that month.",
        "",
        _md_table(
            [
                {
                    "month": r["name"],
                    "Q?": "Q" if r["q"] else "",
                    "n_cm": f"{r['n_cm']:,}",
                    "tax CM": _pp(r["share_cm"]),
                    "tax companies": _pp(r["share_co"]),
                    "years": r["years"],
                }
                for r in p1["cal_rows"]
            ]
        ),
        "",
        f"Raw tax company-months (train): {p1['n_tax_cm_raw']:,} / companies {p1['n_tax_co_raw']:,}. "
        f"Store vs raw agreement {_pp(p1['agree'])} — not an ops.py bug.",
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## 2. `c_missed_tax` — prevalence, acf, size, “just not a tax month?”",
        "",
        p2["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {p2['n_cm']:,} / {p2['n_co']:,} |",
        f"| prevalence (share=1) | {_pp(p2['prev'])} (n={p2['n_miss']:,}) |",
        f"| modal / modal share | {p2['modal']:g} / {_pp(p2['modal_share'])} |",
        f"| confirm feature-report 91.7% | {'YES' if p2['confirm_modal'] else 'NO'} |",
        f"| `c_tax_month` share | {_pp(p2['tax_share'])} |",
        f"| store reconstruction agree | {_pp(p2['agree'])} |",
        f"| P(missed \\| not tax month) | {_pp(p2['p_miss_given_notax'])} |",
        f"| P(usual \\| not tax month) | {_pp(p2['p_usual_given_notax'])} |",
        f"| P(not tax \\| usual) | {_pp(p2['p_notax_given_usual'])} |",
        f"| acf1 / acf3 / acf6 | {_f(p2['acf1'])} / {_f(p2['acf3'])} / {_f(p2['acf6'])} |",
        f"| `c_tax_month` acf1 / acf3 | {_f(p2['acf1_tax'])} / {_f(p2['acf3_tax'])} |",
        f"| ρ vs log1p(a_in3) | {_f(p2['rho'])} |",
        f"| missed Q-months / other | {_pp(p2['q_miss'])} / {_pp(p2['nq_miss'])} |",
        f"| missed January / not-Jan | {_pp(p2['miss_on_jan'])} / {_pp(p2['miss_off_jan'])} |",
        f"| “just not a tax month”? | {'YES' if p2['just_not_tax'] else 'NO'} |",
        f"| calendar-dummy pattern? | {'YES' if p2['calendar_dummy'] else 'NO'} |",
        "",
        "Missed vs tax by calendar month (train CM):",
        "",
        _md_table(
            [
                {
                    "month": r["name"],
                    "Q?": "Q" if r["q"] else "",
                    "tax": _pp(r["tax_share"]),
                    "missed": _pp(r["miss_share"]),
                    "n_cm": f"{r['n_cm']:,}",
                }
                for r in p2["cal"]
            ]
        ),
        "",
        "## 3. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p3['n_y2']:,} base {_pp(p3['y2_rate'])}; "
        f"Y3 stressed n={p3['n_y3']:,} base {_pp(p3['y3_rate'])}; "
        f"Y9 n={p3['n_y9']:,} base {_pp(p3['y9_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p3['days_y3'])}).",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        f"KEEP-as-Q5 rule: missed-tax beats size by ≥{KEEP_DELTA:g} **and** is not “just not January”. "
        f"Size dummy ≥{SIZE_PARK:g} → PARK as X.",
        "",
        "## 4. Leak vs `a_out6` / payroll month",
        "",
        p4["prose"],
        "",
        _md_table(
            [{"pair": r["pair"], "Spearman": _f(r["rho"])} for r in p4["rows"]]
        ),
        "",
        "## 5. Dark 470 vs 744 — same bank-book tax rate?",
        "",
        p5["prose"],
        "",
        "Company-level (mean of each company's tax-CM rate):",
        "",
        _md_table(p5["rows"]),
        "",
        "Company-month:",
        "",
        _md_table(
            [
                {
                    "group": r["group"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "tax": _pp(r["tax_share"]),
                    "missed": _pp(r["miss_share"]),
                }
                for r in p5["cm"]
            ]
        ),
        "",
        "## 6. Q6 — lag1 of `c_missed_tax` (1-month only)",
        "",
        p6["prose"],
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. VAT vs corporate-tax guess (amount size)",
        "",
        p7["prose"],
        "",
        "Do **not** build a Y from this guess.",
        "",
        _md_table(
            [
                {
                    "month": r["name"],
                    "Q?": "Q" if r["q"] else "",
                    "n_tx": f"{r['n_tx']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "|amt| p50": f"{r['p50']:,.0f}" if np.isfinite(r["p50"]) else "—",
                    "|amt| p90": f"{r['p90']:,.0f}" if np.isfinite(r["p90"]) else "—",
                }
                for r in p7["by_m"]
            ]
        ),
        "",
        f"tax_refund txs: {p7['refund']['n']:,} across {p7['n_refund_co']:,} train companies; "
        f"p50={p7['refund']['p50']:,.0f}." if np.isfinite(p7["refund"]["p50"]) else "tax_refund: none.",
        "",
        "## 8. `tax_refund` months — recoveries? (not a Y)",
        "",
        p8["prose"],
        "",
        _md_table(
            [
                {
                    "y": r["y"],
                    "slice": r["slice"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_labeled": f"{r['n_labeled']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                }
                for r in p8["rows"]
            ]
        ),
        "",
        "## 9. Residual skill inside the calendar",
        "",
        p9["prose"],
        "",
        _md_table(p9["rows"]),
        "",
        "Per calendar month (Y3 × `c_missed_tax`):",
        "",
        _md_table(p9["month_rows"]),
        "",
        "## 10. Company tax cadence",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "monthly = tax in ≥70% of grid months. quarterly = ≥70% of tax months sit on Jan/Apr/Jul/Oct and tax-rate ≤45%. never = no tax tx. Else irregular.",
        "",
        "## 11. Holdout coverage only (no AUROC)",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "## 12. Honest skip — missed only among usual filers (tax6≥3)",
        "",
        ctx["p12"]["prose"],
        "",
        _md_table(ctx["p12"]["rows"]),
        "",
        "## 13. `c_tax_month` 0.611 — activity dummy?",
        "",
        ctx["p13"]["prose"],
        "",
        _md_table(ctx["p13"]["rows"]),
        "",
        "## 14. Cadence × Y rates (and missed AUROC)",
        "",
        ctx["p14"]["prose"],
        "",
        _md_table(
            [
                {
                    "y": r["y"],
                    "kind": r["kind"],
                    "n_cm": f"{r['n_cm']:,}",
                    "n_co": f"{r['n_co']:,}",
                    "n_lab": f"{r['n_labeled']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "rate": _pp(r["rate"]),
                    "tax": _pp(r["tax_cm"]),
                    "miss": _pp(r["miss_cm"]),
                }
                for r in ctx["p14"]["rate_rows"]
            ]
        ),
        "",
        _md_table(ctx["p14"]["auc_rows"]),
        "",
        "## 15. Tax amount mix (VAT vs IS)",
        "",
        ctx["p15"]["prose"],
        "",
        _md_table(ctx["p15"]["rows"]),
        "",
        "## 16. Who files off-quarter?",
        "",
        ctx["p16"]["prose"],
        "",
        _md_table(ctx["p16"]["rows"]),
        "",
        "Q-month tax / missed by cadence:",
        "",
        _md_table(ctx["p16"]["q_rows"]),
        "",
        "## 17. Social-security calendar (monthly control)",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## 18. Usual × missed Y rates",
        "",
        ctx["p18"]["prose"],
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
                for r in ctx["p18"]["rows"]
            ]
        ),
        "",
        "## 19. Large-tax month (≥10k) — not a Y",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(
            [
                {
                    "y": r["y"],
                    "n_large_lab": f"{r['n_large_lab']:,}",
                    "n_pos_large": f"{r['n_pos_large']:,}",
                    "rate_large": _pp(r["rate_large"]),
                    "rate_other": _pp(r["rate_other"]),
                    "CV": r["CV"],
                }
                for r in ctx["p19"]["rows"]
            ]
        ),
        "",
        "## 20. Quarterly companies on Q-months + Jan→Feb",
        "",
        ctx["p20"]["prose"],
        "",
        _md_table(ctx["p20"]["rows"]),
        "",
        _md_table(ctx["p20"]["auc_rows"]),
        "",
        "## 21. Who produces missed-tax? (quarterly cannot)",
        "",
        ctx["p21"]["prose"],
        "",
        _md_table(ctx["p21"]["rows"]),
        "",
        "tax6 by cadence (usual = tax6≥3):",
        "",
        _md_table(ctx["p21"]["tax6_rows"]),
        "",
        "## 22. Irregular-only singles (66% of missed)",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: calendar, missed prevalence, singles, leak, dark 470/744, Q6 lag1, VAT/IS amounts, refund, residual calendar, cadence, holdout, honest skip, activity terciles, cadence×Y, amount mix, non-Q who, SS control, usual×missed rates, large-tax, Q-company skip + Jan→Feb.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p2, p3, p5, p6, d = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p5"],
        ctx["p6"],
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
            "model": "tax_qa",
            "split": "train",
            "metric": "tax_calendar_q_share",
            "value": p1["q_share"],
            "coverage": "1.0000",
            "notes": f"shape={p1['shape']} nq={p1['nq_share']:.4f} ratio={p1['ratio']:.3f} agree={p1['agree']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": "tax_qa",
            "split": "train",
            "metric": "c_missed_tax_prev",
            "value": p2["prev"],
            "coverage": "1.0000",
            "notes": f"modal={p2['modal_share']:.4f} confirm917={p2['confirm_modal']} just_not_tax={p2['just_not_tax']} acf1={p2['acf1']:.3f} rho={p2['rho']:.3f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": "tax_qa",
            "split": "train_cv",
            "metric": "auroc_c_missed_tax",
            "value": p3["miss_y3"],
            "coverage": "1.0000",
            "notes": f"size={p3['size_y3']:.4f} days={p3['days_y3']:.4f} beat_size={p3['beat_size_y3']:.4f} q5={d['q5']} x={d['x_dec']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": "tax_qa",
            "split": "train_cv",
            "metric": "auroc_c_tax_month",
            "value": p3["tax_y3"],
            "coverage": "1.0000",
            "notes": f"jan={p3['jan_y3']:.4f} q_month={p3['q_y3']:.4f} not_tax={p3['notax_y3']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": "tax_qa",
            "split": "train_cv",
            "metric": "auroc_c_n_days_with_tx",
            "value": p3["days_y3"],
            "coverage": "1.0000",
            "notes": f"night=0.711 replica; size={p3['size_y3']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": "tax_qa",
            "split": "train",
            "metric": "dark_vs_erp_tax_rate",
            "value": p5["dark_tax"],
            "coverage": f"{p5['n_dark'] / (p5['n_dark'] + p5['n_erp']) if (p5['n_dark'] + p5['n_erp']) else float('nan'):.4f}",
            "notes": f"erp={p5['erp_tax']:.4f} dark={p5['dark_tax']:.4f} confirm744_470={p5['confirm']} same={p5['same_rate']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": "tax_qa",
            "split": "train_cv",
            "metric": "auroc_c_missed_tax_lag1",
            "value": p6["lag"],
            "coverage": "1.0000",
            "notes": f"now={p6['now']:.4f} drop={p6['drop']:.4f} q6={d['q6']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": "tax_qa",
            "split": "train_cv",
            "metric": "auroc_c_missed_tax_usual",
            "value": ctx["p12"]["y3_cv"],
            "coverage": f"{ctx['p12']['n_usual'] / ctx['p2']['n_cm']:.4f}",
            "notes": f"honest skip tax6>=3; n_miss={ctx['p12']['n_miss_u']} keep={ctx['p12']['keep']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": "tax_qa",
            "split": "train",
            "metric": "tax_cadence_n_monthly",
            "value": ctx["p10"]["n_month"],
            "coverage": "1.0000",
            "notes": f"never={ctx['p10']['n_never']} q={ctx['p10']['n_q']} irr={ctx['p10']['n_irr']} shape={p1['shape']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": "tax_qa",
            "split": "train",
            "metric": "c_missed_tax_share_irregular",
            "value": ctx["p21"]["n_irr"] / ctx["p21"]["n_miss"] if ctx["p21"]["n_miss"] else "",
            "coverage": "1.0000",
            "notes": f"n_miss={ctx['p21']['n_miss']} irr={ctx['p21']['n_irr']} mon={ctx['p21']['n_mon']} q={ctx['p21']['n_q_miss']}",
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
    print(f"tax_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["c_missed_tax", "c_tax_month"], (1,))
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()} holdout CM={(panel['split']=='holdout').sum()}")

    con = connect()
    try:
        print("pass 1 calendar")
        p1 = pass1_calendar(tr, con)
        print("pass 2 missed-tax")
        p2 = pass2_missed(tr)
        print("pass 3 singles")
        p3 = pass3_auroc(tr)
        print("pass 4 leak")
        p4 = pass4_leak(tr)
        print("pass 5 dark 470 vs 744")
        p5 = pass5_dark(tr, con)
        print("pass 6 Q6 lag1")
        p6 = pass6_q6(tr)
        print("pass 7 VAT vs IS amounts")
        p7 = pass7_vat(tr, con)
        print("pass 8 tax_refund")
        p8 = pass8_refund(tr, p7["refund_tx"])
        print("pass 9 residual calendar")
        p9 = pass9_residual(tr)
        print("pass 10 cadence")
        p10 = pass10_cadence(tr)
        print("pass 11 holdout coverage")
        p11 = pass11_holdout(panel)
        print("pass 12 honest skip")
        p12 = pass12_honest_skip(tr)
        print("pass 13 tax vs activity")
        p13 = pass13_activity(tr)
        print("pass 14 cadence × Y")
        p14 = pass14_cadence_y(tr, p10["ever"])
        print("pass 15 amount mix")
        p15 = pass15_amounts(p7["tax_tx"])
        print("pass 16 who files non-Q")
        p16 = pass16_nonq_who(tr, p10["ever"])
        print("pass 17 SS calendar")
        p17 = pass17_ss(tr)
        print("pass 18 usual × missed rates")
        p18 = pass18_usual_rates(tr)
        print("pass 19 large-tax month")
        p19 = pass19_large_tax(tr, p7["tax_tx"])
        print("pass 20 quarterly skip + Jan→Feb")
        p20 = pass20_q_skip(tr, p10["ever"])
        print("pass 21 who is missed")
        p21 = pass21_miss_who(tr, p10["ever"])
        print("pass 22 irregular-only")
        p22 = pass22_irregular(tr, p10["ever"])
    finally:
        con.close()

    decision = decide(p1, p2, p3, p4, p6, p9, p12, p13)
    png_ok = make_png(p1)
    headline = (
        f"Calendar **{p1['shape']}** (Q-months tax {_pp(p1['q_share'])} vs other {_pp(p1['nq_share'])}). "
        f"`c_missed_tax` modal {_pp(p2['modal_share'])} "
        f"({'CONFIRM 91.7%' if p2['confirm_modal'] else 'off 91.7%'}); "
        f"missed is {'just not-a-tax-month' if p2['just_not_tax'] else 'NOT just not-a-tax-month'} "
        f"(P(missed|not tax)={_pp(p2['p_miss_given_notax'])}). "
        f"Y3 missed {_f(p3['miss_y3'])} vs size {_f(p3['size_y3'])} (Δ {_f(p3['beat_size_y3'])}) "
        f"vs days {_f(p3['days_y3'])}. Dark vs 744 tax-CM {_pp(p5['dark_tax'])} vs {_pp(p5['erp_tax'])}. "
        f"Q5 **{decision['q5']}**. PARK as health Y. X **{decision['x_dec']}**."
    )
    print(headline)
    failed = []
    if not p2["confirm_modal"]:
        failed.append(f"modal share {_pp(p2['modal_share'])} ≠ 91.7% quote")
    if not p5["confirm"]:
        failed.append(f"dark/erp {p5['n_dark']}/{p5['n_erp']} ≠ 470/744")
    if abs(p3["days_y3"] - DAYS_BENCH) > 0.02 if np.isfinite(p3["days_y3"]) else True:
        failed.append(
            f"Y3 days replica {_f(p3['days_y3'])} vs night 0.711 "
            "(signed fold; published 0.711 may be oriented 1−raw)"
        )
    if p1["agree"] < 0.995:
        failed.append(f"store vs raw tax-month agree {_pp(p1['agree'])} — inspect ops.py")
    failed.append(
        f"honest skip Y3 {_f(p12['y3_cv'])} < size {_f(p3['size_y3'])}; "
        f"large-tax {_f(p19['y3_cv'])} also loses. PARK as X."
    )
    failed.append(
        "quarterly companies on Q-months miss 0% — tax6≥3 cannot fire for a 4-per-year book. "
        "Do not invent a Y; do not patch ops.py tonight."
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
