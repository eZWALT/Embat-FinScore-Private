"""Factoring / confirming / LOC inventory flags — 44-col QA.

NORTH_STAR Q3: factoring and confirming are working-capital tools, not
utilisation (Y10 PARK). `f_has_factoring` / `f_has_confirming` / `f_has_loc`
are created_at inventory flags (rise-only like G). Feature report: rare
(modal 99.3% / 96.7%), still in the 44-col starter.

Debt schedule QA: inventory is a connection panel, not origination. Do
not redo schedule / util. Do not score these F flags vs Y9. Y3 X never B;
Y2 X never B — vs Y3 / Y2 is legal.

Holdout 72 / seed 20260918: coverage only. Singles on train via
analysis.evaluate.protocol. No parquet rewrite. No new GBM. No 0–100.
Do not run build_targets. Do not invent a merged Y.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.factoring_qa

Owned: analysis/evaluate/factoring_qa.py, analysis/outputs/factoring_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_factoring.md (end).

Iteration (same module):
1. Prevalence + rise-only + size ρ + Y3/Y2 singles + Q3 new-type month
2. Y4 / Y5 leftover / ever-FX 228 overlap + dark 470 vs 744
3. bank_name; confirming as AP-side (Y5 rates, never E as X)
4. Group-dummy screen; size-tercile Q3; first-birth vs add-on
5. Q3 min-n gate (CLOSE false KEEP); Y5 AP after size; dark WC; Empresas; GROUP_0139
6. Zero Y3 on factoring; connect-after-cash; granted p50 outstanding=0; 2025-10 wave
7. Leftover = GROUP_0139; siblings; EUR book; store=live; ICC 0.97
8. Pre/post; never-recover 0/18; size-matched T3 0 vs 7.7%; in-house LOC
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
from analysis.features.common import ANALYSIS, AS_OF, DATA, MONTHS, connect
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
OUT_MD = ANALYSIS / "outputs" / "factoring_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "factoring_has_vs_size.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "fd90198f"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y4 = "y4_ds_r_double"
Y5_AP = "y5_ap_od30_ownp80"
Y5_AR = "y5_ar_od30_sust"
N_FOLDS = 5
DAYS_BENCH = 0.711
Y2_DAYS_BENCH = 0.540
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
Q3_PP = 0.05
MIN_Q3_LAB = 20
MIN_POS = 50
MIN_OWN_HIST = 6
OWN_P_HI = 0.80
OWN_P_LO = 0.20
GROUP_DUMMY_SHARE = 0.25
TYPES = ("factoring", "confirming", "lineofcredit")
FLAGS = ("f_has_factoring", "f_has_confirming", "f_has_loc")
FLAG_OF = {
    "factoring": "f_has_factoring",
    "confirming": "f_has_confirming",
    "lineofcredit": "f_has_loc",
}

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_io_ratio",
    "c_n_days_with_tx",
    "d_supp_hhi",
    "d_cust_hhi",
    "e_fx_share",
    "f_has_factoring",
    "f_has_confirming",
    "f_has_loc",
    "f_new_facility",
    "f_n_facilities",
    "f_ds_r",
    "f_fc_r",
)
Y_KEEP = (Y2, Y3, Y4, Y5_AP, Y5_AR)


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


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return "" if v is None else str(v)


def icc_anova(series: pd.Series, company: pd.Series) -> dict:
    s = pd.DataFrame(
        {"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}
    ).dropna()
    if len(s) < 10 or s["x"].nunique() < 2:
        return {"icc": float("nan"), "k": 0}
    y = s["x"].to_numpy(dtype=float)
    _, inv, counts = np.unique(s["co"].to_numpy(), return_inverse=True, return_counts=True)
    if counts.size < 2:
        return {"icc": float("nan"), "k": 0}
    means = np.bincount(inv, weights=y) / counts
    grand = float(y.mean())
    ssb = float(np.sum(counts * (means - grand) ** 2))
    ssw = float(np.sum((y - means[inv]) ** 2))
    k = int(counts.size)
    n = int(y.size)
    var_b = ssb / max(k - 1, 1)
    var_w = ssw / max(n - k, 1)
    icc = var_b / (var_b + var_w) if (var_b + var_w) > 0 else float("nan")
    return {"icc": float(icc), "k": k}


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
    mask: pd.Series | None = None,
    n_folds: int = N_FOLDS,
) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = y.notna() & x.notna()
    if mask is not None:
        defined = defined & mask
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    if n_pos < MIN_POS or n_neg == 0:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "n_folds": 0,
            "train_sign": 0,
            "train_auc": float("nan"),
            "n_defined": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "low_power": True,
        }
    aucs = []
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        aucs.append(auroc(y[va], sign * x[va]))
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "low_power": False,
    }


def _orient(auc: float) -> tuple[float, int]:
    if auc is None or not np.isfinite(auc):
        return float("nan"), 0
    a = float(auc)
    if a >= 0.5:
        return a, 1
    return 1.0 - a, -1


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


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


def add_own_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """Y5 leftover cell: own expanding p20 io / p80 HHI. Never pooled."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    x = pd.to_numeric(out["a_io_ratio"], errors="coerce")
    p20 = x.groupby(out["company_id"], sort=False).transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P_LO, MIN_OWN_HIST)
    )
    extra["a_io_ratio_lo"] = pd.Series(
        np.where(x.notna() & p20.notna(), (x < p20).astype(float), np.nan),
        index=out.index,
    )
    h = pd.to_numeric(out["d_supp_hhi"], errors="coerce")
    p80 = h.groupby(out["company_id"], sort=False).transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P_HI, MIN_OWN_HIST)
    )
    extra["d_supp_hhi_hi"] = pd.Series(
        np.where(h.notna() & p80.notna(), (h > p80).astype(float), np.nan),
        index=out.index,
    )
    hc = pd.to_numeric(out["d_cust_hhi"], errors="coerce")
    p80c = hc.groupby(out["company_id"], sort=False).transform(
        lambda s: _expanding_quantile_skipna(s, OWN_P_HI, MIN_OWN_HIST)
    )
    extra["d_cust_hhi_hi"] = pd.Series(
        np.where(hc.notna() & p80c.notna(), (hc > p80c).astype(float), np.nan),
        index=out.index,
    )
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def load_products(con) -> pd.DataFrame:
    df = con.execute(
        """
        SELECT CAST(product_id AS VARCHAR) AS product_id,
               CAST(company_id AS VARCHAR) AS company_id,
               CAST(type AS VARCHAR) AS type,
               CAST(label AS VARCHAR) AS label,
               CAST(bank_name AS VARCHAR) AS bank_name,
               CAST(service AS VARCHAR) AS service,
               CAST(currency AS VARCHAR) AS currency,
               CAST(created_at AS TIMESTAMP) AS created_at,
               granted, outstanding, liquidity,
               coalesce(created_after_snapshot, false) AS created_after_snapshot
        FROM debt_products
        WHERE type IN ('factoring', 'confirming', 'lineofcredit')
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["created_month"] = df["created_at"].dt.to_period("M").dt.to_timestamp()
    return df


def attach_new_type_months(panel: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Count of each WC type whose created_at falls inside the month."""
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"]].copy()
    out = panel.copy()
    out["period_m"] = pd.to_datetime(out["period"]).dt.to_period("M").dt.to_timestamp()
    for t, col in (
        ("factoring", "new_factoring"),
        ("confirming", "new_confirming"),
        ("lineofcredit", "new_loc"),
    ):
        sl = live.loc[live["type"] == t, ["company_id", "created_month"]]
        cnt = sl.groupby(["company_id", "created_month"]).size().rename(col)
        out = out.merge(
            cnt.reset_index().rename(columns={"created_month": "period_m"}),
            on=["company_id", "period_m"],
            how="left",
        )
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["new_wc"] = out["new_factoring"] + out["new_confirming"]
    out["new_any_typed"] = out["new_factoring"] + out["new_confirming"] + out["new_loc"]
    # holdout products counted only later; panel already tagged
    _ = hold
    return out.drop(columns=["period_m"])


def load_panel(con) -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE, columns=list(STORE_COLS))
    yraw = pd.read_parquet(TARGETS, columns=["company_id", "period", *Y_KEEP])
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    panel = _keys(raw)
    y = _keys(yraw)
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    for c in FLAGS + ("f_new_facility", "f_n_facilities"):
        panel[c] = pd.to_numeric(panel[c], errors="coerce")
    book = book_invoice_ids(con)
    pop = dark_population(con)
    panel["has_book"] = panel["company_id"].isin(book)
    panel["group_mix"] = panel["group_id"].map(pop["mix_of"])
    train_cos = train_companies(con)
    assert_no_holdout(train_cos["company_id"])
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    panel = add_own_cuts(panel)
    ap = pd.to_numeric(panel[Y5_AP], errors="coerce")
    io_lo = pd.to_numeric(panel["a_io_ratio_lo"], errors="coerce")
    hh = pd.to_numeric(panel["d_supp_hhi_hi"], errors="coerce")
    panel["y5_ap_leftover"] = np.where(
        (ap == 1) & io_lo.notna() & hh.notna() & (io_lo == 0) & (hh == 0),
        1.0,
        np.where((ap == 1) & io_lo.notna() & hh.notna(), 0.0, np.nan),
    )
    panel.attrs["dark_pop"] = pop
    return panel, pop


def _rate(sl: pd.DataFrame, y: str) -> dict:
    lab = pd.to_numeric(sl[y], errors="coerce")
    n_lab = int(lab.notna().sum())
    n_pos = int((lab == 1).sum())
    return {
        "n_cm": int(len(sl)),
        "n_co": int(sl["company_id"].nunique()),
        "n_lab": n_lab,
        "n_pos": n_pos,
        "rate": _pct(n_pos, n_lab),
    }


def _rise_only(tr: pd.DataFrame, col: str) -> dict:
    s = tr.sort_values(["company_id", "period"])
    dlt = s.groupby("company_id")[col].diff()
    n_drop = int((dlt < 0).sum())
    n_rise = int((dlt > 0).sum())
    n_flat = int((dlt == 0).sum())
    return {
        "n_rise": n_rise,
        "n_drop": n_drop,
        "n_flat": n_flat,
        "rise_only": n_drop == 0,
        "acf1": median_acf(s[col], s["company_id"], 1),
        "acf3": median_acf(s[col], s["company_id"], 3),
        "acf6": median_acf(s[col], s["company_id"], 6),
    }


# ---------------------------------------------------------------------------
# Pass 1 — prevalence + rise-only
# ---------------------------------------------------------------------------
def pass1_prevalence(tr: pd.DataFrame, ho: pd.DataFrame, products: pd.DataFrame) -> dict:
    hold = load_holdout()
    raw = products.copy()
    raw_train = raw.loc[~raw["company_id"].isin(hold)]
    live = raw.loc[~raw["created_after_snapshot"]]
    live_tr = live.loc[~live["company_id"].isin(hold)]
    rows = []
    rise = {}
    n_cm = int(len(tr))
    n_co = int(tr["company_id"].nunique())
    for t in TYPES:
        flag = FLAG_OF[t]
        x = pd.to_numeric(tr[flag], errors="coerce").fillna(0)
        ever = set(tr.loc[x > 0, "company_id"])
        last = tr.sort_values("period").groupby("company_id", sort=False).last()
        last_on = int((pd.to_numeric(last[flag], errors="coerce") > 0).sum())
        r = _rise_only(tr, flag)
        rise[t] = r
        modal0 = float((x == 0).mean())
        raw_n = int((raw_train["type"] == t).sum())
        raw_co = int(raw_train.loc[raw_train["type"] == t, "company_id"].nunique())
        after = int(
            ((raw_train["type"] == t) & raw_train["created_after_snapshot"]).sum()
        )
        ho_x = pd.to_numeric(ho[flag], errors="coerce").fillna(0)
        rows.append(
            {
                "type": t,
                "flag": flag,
                "raw_prod_train": raw_n,
                "raw_co_train": raw_co,
                "after_snap": after,
                "ever_n": int(len(ever)),
                "ever_share_co": _pct(len(ever), n_co),
                "cm_on": int((x > 0).sum()),
                "cm_share": float((x > 0).mean()),
                "modal0": modal0,
                "last_on": last_on,
                "n_rise": r["n_rise"],
                "n_drop": r["n_drop"],
                "rise_only": r["rise_only"],
                "acf1": r["acf1"],
                "hold_ever": int(ho.loc[ho_x > 0, "company_id"].nunique()),
                "hold_cm_on": int((ho_x > 0).sum()),
            }
        )
    # any WC (fact or conf)
    any_wc = (tr["f_has_factoring"].fillna(0) > 0) | (tr["f_has_confirming"].fillna(0) > 0)
    any_loc = tr["f_has_loc"].fillna(0) > 0
    n_fac = pd.to_numeric(tr["f_n_facilities"], errors="coerce")
    fac_rise = _rise_only(tr, "f_n_facilities")
    return {
        "n_cm": n_cm,
        "n_co": n_co,
        "n_hold_co": int(ho["company_id"].nunique()),
        "n_hold_cm": int(len(ho)),
        "rows": rows,
        "rise": rise,
        "ever_wc": int(tr.loc[any_wc, "company_id"].nunique()),
        "ever_loc": int(tr.loc[any_loc, "company_id"].nunique()),
        "cm_wc": int(any_wc.sum()),
        "fac_rise": fac_rise,
        "raw_train_n": int(len(raw_train)),
        "live_train_n": int(len(live_tr)),
        "confirm_modal_fact": abs(next(r["modal0"] for r in rows if r["type"] == "factoring") - 0.993) < 0.01,
        "confirm_modal_conf": abs(next(r["modal0"] for r in rows if r["type"] == "confirming") - 0.967) < 0.015,
        "prose": "",
    }


# ---------------------------------------------------------------------------
# Pass 2 — size ρ
# ---------------------------------------------------------------------------
def pass2_size(tr: pd.DataFrame) -> dict:
    log_in3 = pd.to_numeric(tr["log_in3"], errors="coerce")
    ain3 = pd.to_numeric(tr["a_in3"], errors="coerce")
    rows = []
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    last_log = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    for t in TYPES:
        flag = FLAG_OF[t]
        x = pd.to_numeric(tr[flag], errors="coerce")
        rho = spearman(x, log_in3)
        rho_raw = spearman(x, ain3)
        ever = (
            tr.assign(_x=x)
            .groupby("company_id")["_x"]
            .max()
        )
        rho_ever = spearman(ever, last_log.reindex(ever.index))
        rows.append(
            {
                "type": t,
                "flag": flag,
                "rho_login3": rho,
                "rho_ain3": rho_raw,
                "rho_ever_medsize": rho_ever,
                "is_size": bool(
                    (np.isfinite(rho) and abs(rho) >= SIZE_RHO)
                    or (np.isfinite(rho_raw) and abs(rho_raw) >= SIZE_RHO)
                ),
            }
        )
    return {"rows": rows, "any_size": any(r["is_size"] for r in rows)}


# ---------------------------------------------------------------------------
# Pass 3 — singles vs Y3 / Y2
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    leak = leakage_check(FLAGS, Y3, forbidden_prefixes=("b",))
    leak2 = leakage_check(FLAGS, Y2, forbidden_prefixes=("b",))
    feats = {
        **{f: pd.to_numeric(tr[f], errors="coerce") for f in FLAGS},
        "f_new_facility_gt0": (pd.to_numeric(tr["f_new_facility"], errors="coerce") > 0).astype(float),
        "new_factoring_gt0": (tr["new_factoring"] > 0).astype(float),
        "new_confirming_gt0": (tr["new_confirming"] > 0).astype(float),
        "new_wc_gt0": (tr["new_wc"] > 0).astype(float),
        "log1p_a_in3": pd.to_numeric(tr["log_in3"], errors="coerce"),
        "c_n_days_with_tx": pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce"),
    }
    size_rho = {name: spearman(col, tr["log_in3"]) for name, col in feats.items()}
    rows = []
    for y in (Y3, Y2):
        for name, col in feats.items():
            cv = signed_oof_auroc(tr[y], col, tr["fold"])
            pooled = auroc(tr[y], col)
            o_cv, s_cv = _orient(cv["cv"]) if np.isfinite(cv["cv"]) else (cv["cv"], cv["train_sign"])
            # signed_oof already orients per fold; report that CV as oriented
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "cv": cv["cv"],
                    "sd": cv["sd"],
                    "sign": cv["train_sign"],
                    "train_auc": cv["train_auc"],
                    "pooled": pooled,
                    "n_lab": cv["n_defined"],
                    "n_pos": cv["n_pos"],
                    "rho_login3": size_rho[name],
                    "is_size": abs(size_rho[name]) >= SIZE_RHO if np.isfinite(size_rho[name]) else False,
                    "low_power": cv["low_power"],
                }
            )

    def _pick(y, feat):
        return next(r for r in rows if r["y"] == y and r["feature"] == feat)

    size_y3 = _pick(Y3, "log1p_a_in3")
    days_y3 = _pick(Y3, "c_n_days_with_tx")
    size_y2 = _pick(Y2, "log1p_a_in3")
    days_y2 = _pick(Y2, "c_n_days_with_tx")
    verdicts = []
    for r in rows:
        if r["feature"] in {"log1p_a_in3", "c_n_days_with_tx"}:
            continue
        if r["y"] != Y3:
            continue
        beat = (
            r["cv"] - size_y3["cv"]
            if np.isfinite(r["cv"]) and np.isfinite(size_y3["cv"])
            else float("nan")
        )
        keep = bool(
            np.isfinite(beat)
            and beat >= KEEP_DELTA
            and not r["is_size"]
        )
        verdicts.append(
            {
                "feature": r["feature"],
                "cv": r["cv"],
                "size_cv": size_y3["cv"],
                "days_cv": days_y3["cv"],
                "beat_size": beat,
                "is_size": r["is_size"],
                "keep_44": keep,
            }
        )
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    return {
        "rows": rows,
        "verdicts": verdicts,
        "size_y3": size_y3,
        "days_y3": days_y3,
        "size_y2": size_y2,
        "days_y2": days_y2,
        "n_y3": int(y3.notna().sum()),
        "y3_rate": float(y3.mean()) if y3.notna().any() else float("nan"),
        "n_y2": int(y2.notna().sum()),
        "y2_rate": float(y2.mean()) if y2.notna().any() else float("nan"),
        "days_y3_ok": abs(days_y3["cv"] - DAYS_BENCH) < 0.02 if np.isfinite(days_y3["cv"]) else False,
        "leak_y3": leak,
        "leak_y2": leak2,
        "any_keep_44": any(v["keep_44"] for v in verdicts),
    }


# ---------------------------------------------------------------------------
# Pass 4 — Q3 new factoring / confirming month
# ---------------------------------------------------------------------------
def pass4_q3(tr: pd.DataFrame) -> dict:
    rows = []
    terc = pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    tr = tr.assign(_terc=terc)
    flags = {
        "new_factoring": tr["new_factoring"] > 0,
        "new_confirming": tr["new_confirming"] > 0,
        "new_wc": tr["new_wc"] > 0,
        "new_loc": tr["new_loc"] > 0,
        "f_new_facility": pd.to_numeric(tr["f_new_facility"], errors="coerce") > 0,
    }
    # first-birth vs add-on for WC
    s = tr.sort_values(["company_id", "period"]).copy()
    s["had_wc"] = (
        (s["f_has_factoring"].fillna(0) > 0) | (s["f_has_confirming"].fillna(0) > 0)
    ).astype(int)
    prev = s.groupby("company_id")["had_wc"].shift(1)
    s["wc_birth"] = (s["had_wc"] == 1) & (prev.fillna(0) == 0)
    birth_idx = s.index[s["wc_birth"]]
    for name, mask in flags.items():
        n = int(mask.sum())
        n_co = int(tr.loc[mask, "company_id"].nunique())
        rec = {
            "flag": name,
            "n_cm": n,
            "n_co": n_co,
            "share": _pct(n, len(tr)),
        }
        for y in (Y3, Y2):
            on = _rate(tr.loc[mask], y)
            off = _rate(tr.loc[~mask], y)
            rec[f"{y}_on"] = on["rate"]
            rec[f"{y}_off"] = off["rate"]
            rec[f"{y}_n_lab_on"] = on["n_lab"]
            rec[f"{y}_pp"] = (
                on["rate"] - off["rate"]
                if np.isfinite(on["rate"]) and np.isfinite(off["rate"])
                else float("nan")
            )
        # size-controlled: weighted mean of within-tercile deltas
        w_pp = {}
        for y in (Y3, Y2):
            deltas = []
            weights = []
            terc_rows = []
            for tname, g in tr.groupby("_terc", observed=False):
                m = mask.reindex(g.index).fillna(False)
                on = _rate(g.loc[m], y)
                off = _rate(g.loc[~m], y)
                dlt = (
                    on["rate"] - off["rate"]
                    if np.isfinite(on["rate"]) and np.isfinite(off["rate"])
                    else float("nan")
                )
                terc_rows.append(
                    {
                        "flag": name,
                        "tercile": str(tname),
                        "y": y,
                        "n_on": on["n_lab"],
                        "rate_on": on["rate"],
                        "rate_off": off["rate"],
                        "pp": dlt,
                    }
                )
                if np.isfinite(dlt) and on["n_lab"] > 0:
                    deltas.append(dlt)
                    weights.append(on["n_lab"] + off["n_lab"])
            rec[f"{y}_pp_size"] = (
                float(np.average(deltas, weights=weights)) if deltas else float("nan")
            )
            rec[f"_terc_{y}"] = terc_rows
            w_pp[y] = rec[f"{y}_pp_size"]
        rec["q3_keep"] = bool(
            name in {"new_factoring", "new_confirming", "new_wc"}
            and rec.get("y3_recover_cash_6m_n_lab_on", 0) >= MIN_Q3_LAB
            and (
                (np.isfinite(w_pp[Y3]) and abs(w_pp[Y3]) >= Q3_PP)
                or (
                    rec.get("y2_neg_2of3_n_lab_on", 0) >= MIN_Q3_LAB
                    and np.isfinite(w_pp[Y2])
                    and abs(w_pp[Y2]) >= Q3_PP
                )
            )
        )
        rows.append(rec)
    n_new_wc = int(flags["new_wc"].sum())
    n_birth = int(s.loc[s["new_wc"] > 0, "wc_birth"].sum()) if n_new_wc else 0
    # birth vs add-on rates (connection, not origination)
    new_m = s["new_wc"] > 0
    birth_m = new_m & s["wc_birth"]
    addon_m = new_m & ~s["wc_birth"]
    birth_rates = {
        "n": int(birth_m.sum()),
        "y3": _rate(s.loc[birth_m], Y3),
        "y2": _rate(s.loc[birth_m], Y2),
    }
    addon_rates = {
        "n": int(addon_m.sum()),
        "y3": _rate(s.loc[addon_m], Y3),
        "y2": _rate(s.loc[addon_m], Y2),
    }
    cal = (
        s.loc[new_m]
        .groupby("period")
        .agg(
            n_wc=("company_id", "size"),
            n_fact=("new_factoring", lambda x: int((x > 0).sum())),
            n_conf=("new_confirming", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    return {
        "rows": rows,
        "tercile_rows": [r for rec in rows for y in (Y3, Y2) for r in rec[f"_terc_{y}"]],
        "n_new_wc": n_new_wc,
        "n_birth": n_birth,
        "share_new_is_birth": _pct(n_birth, n_new_wc),
        "any_q3_keep": any(r["q3_keep"] for r in rows if r["flag"] in {"new_factoring", "new_confirming", "new_wc"}),
        "birth": birth_rates,
        "addon": addon_rates,
        "calendar": cal,
        "min_lab": MIN_Q3_LAB,
    }


# ---------------------------------------------------------------------------
# Pass 5 — overlap Y4 / Y5 leftover / ever-FX 228
# ---------------------------------------------------------------------------
def pass5_overlap(tr: pd.DataFrame) -> dict:
    ever = {}
    for t in TYPES:
        flag = FLAG_OF[t]
        ever[t] = set(tr.loc[pd.to_numeric(tr[flag], errors="coerce") > 0, "company_id"])
    ever["wc"] = ever["factoring"] | ever["confirming"]
    y4_co = set(tr.loc[pd.to_numeric(tr[Y4], errors="coerce") == 1, "company_id"])
    y5_pos = set(tr.loc[pd.to_numeric(tr[Y5_AP], errors="coerce") == 1, "company_id"])
    y5_left = set(tr.loc[pd.to_numeric(tr["y5_ap_leftover"], errors="coerce") == 1, "company_id"])
    fx_co = set(tr.loc[pd.to_numeric(tr["e_fx_share"], errors="coerce") > 0, "company_id"])
    n_fx = int(len(fx_co))
    confirm_228 = n_fx == 228
    leftover_n = int((pd.to_numeric(tr["y5_ap_leftover"], errors="coerce") == 1).sum())
    leftover_co = int(len(y5_left))
    rows = []
    for name, ids in ever.items():
        rows.append(
            {
                "set": name,
                "n_co": int(len(ids)),
                "y4": int(len(ids & y4_co)),
                "y4_share": _pct(len(ids & y4_co), len(ids)),
                "y5_ap": int(len(ids & y5_pos)),
                "y5_ap_share": _pct(len(ids & y5_pos), len(ids)),
                "y5_left": int(len(ids & y5_left)),
                "y5_left_share": _pct(len(ids & y5_left), len(ids)),
                "fx228": int(len(ids & fx_co)),
                "fx_share": _pct(len(ids & fx_co), len(ids)),
            }
        )
    # month-level leftover overlap (descriptive, never E as X)
    left_m = pd.to_numeric(tr["y5_ap_leftover"], errors="coerce") == 1
    month_rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        on = pd.to_numeric(tr[flag], errors="coerce") > 0
        n_left_on = int((left_m & on).sum())
        month_rows.append(
            {
                "flag": flag,
                "leftover_cm_on": n_left_on,
                "leftover_cm": int(left_m.sum()),
                "share_of_leftover": _pct(n_left_on, int(left_m.sum())),
                "y4_cm_on": int(((pd.to_numeric(tr[Y4], errors="coerce") == 1) & on).sum()),
            }
        )
    return {
        "rows": rows,
        "month_rows": month_rows,
        "n_y4_co": int(len(y4_co)),
        "n_y5_pos_co": int(len(y5_pos)),
        "n_y5_left_cm": leftover_n,
        "n_y5_left_co": leftover_co,
        "n_fx": n_fx,
        "confirm_228": confirm_228,
        "n_train_co": int(tr["company_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Pass 6 — 470 dark vs 744 invoiced
# ---------------------------------------------------------------------------
def pass6_dark(tr: pd.DataFrame, pop: dict) -> dict:
    dark_ids = pop["train_dark_ids"]
    dark = tr[tr["company_id"].isin(dark_ids)]
    erp = tr[~tr["company_id"].isin(dark_ids)]
    n_dark = int(dark["company_id"].nunique())
    n_erp = int(erp["company_id"].nunique())
    confirm = n_dark == 470 and n_erp == 744 and bool(pop.get("confirm_470"))
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        d_ever = int(dark.loc[pd.to_numeric(dark[flag], errors="coerce") > 0, "company_id"].nunique())
        e_ever = int(erp.loc[pd.to_numeric(erp[flag], errors="coerce") > 0, "company_id"].nunique())
        rows.append(
            {
                "type": t,
                "dark_ever": d_ever,
                "dark_share": _pct(d_ever, n_dark),
                "erp_ever": e_ever,
                "erp_share": _pct(e_ever, n_erp),
                "dark_cm": int((pd.to_numeric(dark[flag], errors="coerce") > 0).sum()),
                "erp_cm": int((pd.to_numeric(erp[flag], errors="coerce") > 0).sum()),
            }
        )
    mix_rows = []
    for mix, sl in (
        ("all_dark", tr[tr["group_mix"] == "all_dark"]),
        ("mixed", tr[tr["group_mix"] == "mixed"]),
        ("all_invoiced", tr[tr["group_mix"] == "all_invoiced"]),
    ):
        n_co = int(sl["company_id"].nunique())
        rec = {"mix": mix, "n_co": n_co}
        for t in TYPES:
            flag = FLAG_OF[t]
            rec[t] = int(sl.loc[pd.to_numeric(sl[flag], errors="coerce") > 0, "company_id"].nunique())
        mix_rows.append(rec)
    return {
        "n_dark": n_dark,
        "n_erp": n_erp,
        "confirm_470_744": confirm,
        "n_360": pop.get("n_360_alldark"),
        "n_110": pop.get("n_110_mixed"),
        "rows": rows,
        "mix_rows": mix_rows,
    }


# ---------------------------------------------------------------------------
# Pass 7 — bank_name
# ---------------------------------------------------------------------------
def pass7_banks(products: pd.DataFrame) -> dict:
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)].copy()
    assert_no_holdout(live["company_id"])
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t]
        vc = (
            sl.groupby(sl["bank_name"].fillna("(null)"))
            .agg(n=("product_id", "size"), n_co=("company_id", "nunique"))
            .reset_index()
            .sort_values("n", ascending=False)
        )
        empresas = int(sl["bank_name"].fillna("").str.contains("Empresas", case=False).sum())
        inhouse = int(sl["bank_name"].fillna("").str.contains("In-house|customer-defined|Other", case=False).sum())
        rows.append(
            {
                "type": t,
                "n_prod": int(len(sl)),
                "n_co": int(sl["company_id"].nunique()),
                "n_banks": int(sl["bank_name"].nunique()),
                "empresas": empresas,
                "inhouse_or_other": inhouse,
                "top": [
                    {
                        "bank": str(r.bank_name),
                        "n": int(r.n),
                        "n_co": int(r.n_co),
                    }
                    for r in vc.head(8).itertuples(index=False)
                ],
            }
        )
    return {"rows": rows, "n_live_train": int(len(live))}


# ---------------------------------------------------------------------------
# Pass 8 — confirming as AP-side (Y5 rates, never E as X)
# ---------------------------------------------------------------------------
def pass8_ap(tr: pd.DataFrame) -> dict:
    """Confirming should sit on the AP / supplier side. Descriptive only."""
    leak = leakage_check(["f_has_confirming"], Y5_AP, forbidden_prefixes=("e",))
    # We do NOT score confirming as X for Y5 — leak screen is the reminder.
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        on = pd.to_numeric(tr[flag], errors="coerce") > 0
        rec = {"type": t, "flag": flag, "n_cm_on": int(on.sum())}
        for y in (Y5_AP, Y5_AR):
            a = _rate(tr.loc[on], y)
            b = _rate(tr.loc[~on], y)
            rec[f"{y}_on"] = a["rate"]
            rec[f"{y}_off"] = b["rate"]
            rec[f"{y}_n_on"] = a["n_lab"]
            rec[f"{y}_pp"] = (
                a["rate"] - b["rate"]
                if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                else float("nan")
            )
        left_on = _rate(tr.loc[on], "y5_ap_leftover")
        left_off = _rate(tr.loc[~on], "y5_ap_leftover")
        rec["left_on"] = left_on["rate"]
        rec["left_off"] = left_off["rate"]
        rec["left_pp"] = (
            left_on["rate"] - left_off["rate"]
            if np.isfinite(left_on["rate"]) and np.isfinite(left_off["rate"])
            else float("nan")
        )
        rows.append(rec)
    # ever-company Y5 leftover prevalence
    ever_rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        ever = set(tr.loc[pd.to_numeric(tr[flag], errors="coerce") > 0, "company_id"])
        rest = set(tr["company_id"]) - ever
        left_ever = set(tr.loc[pd.to_numeric(tr["y5_ap_leftover"], errors="coerce") == 1, "company_id"])
        ever_rows.append(
            {
                "type": t,
                "ever": int(len(ever)),
                "ever_and_left": int(len(ever & left_ever)),
                "share": _pct(len(ever & left_ever), len(ever)),
                "rest_share": _pct(len(rest & left_ever), len(rest)),
            }
        )
    conf_pp = next(r["y5_ap_od30_ownp80_pp"] for r in rows if r["type"] == "confirming")
    fact_pp = next(r["y5_ap_od30_ownp80_pp"] for r in rows if r["type"] == "factoring")
    # confirming AP-side if AP rate lift > AR lift and > factoring AP lift
    conf = next(r for r in rows if r["type"] == "confirming")
    ap_side = bool(
        np.isfinite(conf["y5_ap_od30_ownp80_pp"])
        and np.isfinite(conf["y5_ar_od30_sust_pp"])
        and conf["y5_ap_od30_ownp80_pp"] > conf["y5_ar_od30_sust_pp"]
    )
    return {
        "rows": rows,
        "ever_rows": ever_rows,
        "leak_ok": leak["ok"],
        "leak": leak,
        "ap_side": ap_side,
        "conf_ap_pp": conf_pp,
        "fact_ap_pp": fact_pp,
    }


# ---------------------------------------------------------------------------
# Pass 9 — group dummy
# ---------------------------------------------------------------------------
def pass9_group(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        ever = (
            tr.groupby(["company_id", "group_id"], as_index=False)[flag]
            .max()
        )
        ever = ever.loc[pd.to_numeric(ever[flag], errors="coerce") > 0]
        n_co = int(len(ever))
        n_g = int(ever["group_id"].nunique())
        if n_co == 0:
            max_share = float("nan")
            max_g = ""
        else:
            vc = ever.groupby("group_id").size().sort_values(ascending=False)
            max_share = float(vc.iloc[0] / n_co)
            max_g = str(vc.index[0])
        dummy = bool(np.isfinite(max_share) and max_share >= GROUP_DUMMY_SHARE)
        rows.append(
            {
                "type": t,
                "n_co": n_co,
                "n_groups": n_g,
                "max_group_share": max_share,
                "max_group": max_g,
                "group_dummy": dummy,
            }
        )
    return {"rows": rows, "any_dummy": any(r["group_dummy"] for r in rows)}


def pass10_cooccur_size_y(tr: pd.DataFrame) -> dict:
    """Product overlap + size-tercile Y3/Y2 rates for has_* (not new)."""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    fact = last["f_has_factoring"].fillna(0) > 0
    conf = last["f_has_confirming"].fillna(0) > 0
    loc = last["f_has_loc"].fillna(0) > 0
    overlap = {
        "fact_only": int((fact & ~conf & ~loc).sum()),
        "conf_only": int((conf & ~fact & ~loc).sum()),
        "loc_only": int((loc & ~fact & ~conf).sum()),
        "fact_conf": int((fact & conf).sum()),
        "fact_loc": int((fact & loc).sum()),
        "conf_loc": int((conf & loc).sum()),
        "all3": int((fact & conf & loc).sum()),
        "n_last": int(len(last)),
    }
    terc = pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    work = tr.assign(_terc=terc)
    rows = []
    for flag in FLAGS:
        on = pd.to_numeric(work[flag], errors="coerce") > 0
        for y in (Y3, Y2):
            for tname, g in work.groupby("_terc", observed=False):
                m = on.reindex(g.index).fillna(False)
                a = _rate(g.loc[m], y)
                b = _rate(g.loc[~m], y)
                rows.append(
                    {
                        "flag": flag,
                        "y": y,
                        "tercile": str(tname),
                        "n_on": a["n_lab"],
                        "rate_on": a["rate"],
                        "rate_off": b["rate"],
                        "pp": (
                            a["rate"] - b["rate"]
                            if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                            else float("nan")
                        ),
                    }
                )
    return {"overlap": overlap, "rows": rows}


def pass11_y5_size(tr: pd.DataFrame) -> dict:
    """Y5 AP leftover after size — never E as X. Is factoring +19.6pp leftover size?"""
    terc = pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    work = tr.assign(_terc=terc)
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        on = pd.to_numeric(work[flag], errors="coerce") > 0
        for tname, g in work.groupby("_terc", observed=False):
            m = on.reindex(g.index).fillna(False)
            a = _rate(g.loc[m], Y5_AP)
            b = _rate(g.loc[~m], Y5_AP)
            left_a = _rate(g.loc[m], "y5_ap_leftover")
            rows.append(
                {
                    "type": t,
                    "tercile": str(tname),
                    "n_on": a["n_lab"],
                    "ap_on": a["rate"],
                    "ap_off": b["rate"],
                    "pp": (
                        a["rate"] - b["rate"]
                        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                        else float("nan")
                    ),
                    "left_on": left_a["rate"],
                    "left_n": left_a["n_lab"],
                }
            )
        # weighted within-tercile AP pp
        deltas, w = [], []
        for r in rows[-3:]:
            if np.isfinite(r["pp"]) and r["n_on"] > 0:
                deltas.append(r["pp"])
                w.append(r["n_on"])
        rec_pp = float(np.average(deltas, weights=w)) if deltas else float("nan")
        terc_chunk = [r for r in rows if r["type"] == t and r["tercile"] in {"T1", "T2", "T3"}]
        rows.append(
            {
                "type": t,
                "tercile": "size-w",
                "n_on": int(sum(r["n_on"] for r in terc_chunk)),
                "ap_on": float("nan"),
                "ap_off": float("nan"),
                "pp": rec_pp,
                "left_on": float("nan"),
                "left_n": 0,
            }
        )
    fact_w = next(r["pp"] for r in rows if r["type"] == "factoring" and r["tercile"] == "size-w")
    conf_w = next(r["pp"] for r in rows if r["type"] == "confirming" and r["tercile"] == "size-w")
    return {
        "rows": rows,
        "fact_pp_size": fact_w,
        "conf_pp_size": conf_w,
        "fact_survives": bool(np.isfinite(fact_w) and abs(fact_w) >= Q3_PP),
        "conf_survives": bool(np.isfinite(conf_w) and abs(conf_w) >= Q3_PP),
    }


def pass12_dark_wc(tr: pd.DataFrame, pop: dict) -> dict:
    """Dark companies with factoring — WC tool without an invoice book."""
    dark_ids = pop["train_dark_ids"]
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["dark"] = last["company_id"].isin(dark_ids)
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        sl = last.loc[pd.to_numeric(last[flag], errors="coerce") > 0]
        dark = sl[sl["dark"]]
        erp = sl[~sl["dark"]]
        rows.append(
            {
                "type": t,
                "dark_n": int(len(dark)),
                "erp_n": int(len(erp)),
                "dark_med_in3": float(pd.to_numeric(dark["a_in3"], errors="coerce").median()) if len(dark) else float("nan"),
                "erp_med_in3": float(pd.to_numeric(erp["a_in3"], errors="coerce").median()) if len(erp) else float("nan"),
                "dark_med_days": float(pd.to_numeric(dark["c_n_days_with_tx"], errors="coerce").median()) if len(dark) else float("nan"),
                "erp_med_days": float(pd.to_numeric(erp["c_n_days_with_tx"], errors="coerce").median()) if len(erp) else float("nan"),
            }
        )
    # dark without the product, for size context
    dark_last = last[last["dark"]]
    dark_no_wc = dark_last.loc[
        (dark_last["f_has_factoring"].fillna(0) == 0)
        & (dark_last["f_has_confirming"].fillna(0) == 0)
    ]
    return {
        "rows": rows,
        "dark_no_wc_n": int(len(dark_no_wc)),
        "dark_no_wc_med_in3": float(pd.to_numeric(dark_no_wc["a_in3"], errors="coerce").median()) if len(dark_no_wc) else float("nan"),
        "note": "Factoring on dark companies is a connected product, not an invoice-finance trail.",
    }


def pass13_empresas(tr: pd.DataFrame, products: pd.DataFrame) -> dict:
    """Empresas-labelled confirming vs retail bank_name — still not E as X."""
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    conf = live.loc[live["type"] == "confirming"].copy()
    conf["empresas"] = conf["bank_name"].fillna("").str.contains("Empresas", case=False)
    emp_co = set(conf.loc[conf["empresas"], "company_id"])
    ret_co = set(conf.loc[~conf["empresas"], "company_id"]) - emp_co
    both = set(conf.loc[conf["empresas"], "company_id"]) & set(conf.loc[~conf["empresas"], "company_id"])
    rows = []
    for name, ids in (("empresas_only", emp_co - both), ("retail_only", ret_co), ("any_empresas", emp_co)):
        sl = tr[tr["company_id"].isin(ids)]
        on = sl["f_has_confirming"].fillna(0) > 0
        a = _rate(sl.loc[on], Y5_AP)
        y3 = _rate(sl.loc[on], Y3)
        rows.append(
            {
                "slice": name,
                "n_co": int(len(ids)),
                "n_cm_on": int(on.sum()),
                "y5_ap": a["rate"],
                "y5_n": a["n_lab"],
                "y3": y3["rate"],
                "y3_n": y3["n_lab"],
            }
        )
    return {"rows": rows, "n_both": int(len(both))}


def pass14_group0139(tr: pd.DataFrame) -> dict:
    """Largest factoring group is 22% — just under the dummy line."""
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    g = last.loc[last["group_id"] == "GROUP_0139"]
    return {
        "n_co": int(g["company_id"].nunique()),
        "n_fact": int((g["f_has_factoring"].fillna(0) > 0).sum()),
        "n_conf": int((g["f_has_confirming"].fillna(0) > 0).sum()),
        "n_loc": int((g["f_has_loc"].fillna(0) > 0).sum()),
        "med_in3": float(pd.to_numeric(g["a_in3"], errors="coerce").median()) if len(g) else float("nan"),
        "ids": sorted(g.loc[g["f_has_factoring"].fillna(0) > 0, "company_id"].astype(str)),
    }


def pass16_y3_zero(tr: pd.DataFrame) -> dict:
    """Rarity: 0 Y3 recoveries on factoring months still AUROC ≈ 0.50."""
    rows = []
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y3.notna()
    n_pos = int((y3 == 1).sum())
    for t in TYPES:
        flag = FLAG_OF[t]
        on = pd.to_numeric(tr[flag], errors="coerce") > 0
        n_on_lab = int((lab & on).sum())
        n_on_pos = int(((y3 == 1) & on).sum())
        rows.append(
            {
                "type": t,
                "y3_lab_on": n_on_lab,
                "y3_pos_on": n_on_pos,
                "share_of_pos": _pct(n_on_pos, n_pos),
                "rate_on": _pct(n_on_pos, n_on_lab),
            }
        )
    return {"rows": rows, "n_pos": n_pos, "n_lab": int(lab.sum())}


def pass17_connect_after_cash(tr: pd.DataFrame, products: pd.DataFrame) -> dict:
    """created_at vs first cash-trail month — connection, not origination."""
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)].copy()
    first = (
        tr.sort_values("period")
        .groupby("company_id", sort=False)
        .first()[["period", "a_in3"]]
        .reset_index()
        .rename(columns={"period": "first_trail", "a_in3": "first_in3"})
    )
    live = live.merge(first, on="company_id", how="left")
    live["lag_m"] = (
        live["created_at"].dt.to_period("M") - pd.to_datetime(live["first_trail"]).dt.to_period("M")
    ).apply(lambda x: int(x.n) if pd.notna(x) else np.nan)
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t, "lag_m"].dropna()
        rows.append(
            {
                "type": t,
                "n": int(len(sl)),
                "p50": float(sl.median()) if len(sl) else float("nan"),
                "share_after_trail": float((sl >= 0).mean()) if len(sl) else float("nan"),
                "share_before": float((sl < 0).mean()) if len(sl) else float("nan"),
            }
        )
    return {"rows": rows}


def pass18_granted_last(con, tr: pd.DataFrame) -> dict:
    """Last-month snapshot amounts by type. Not utilisation. Not a path."""
    last = pd.Timestamp(MONTHS[-1])
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(type AS VARCHAR) AS type,
               abs(granted) AS granted_abs,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type IN ('factoring', 'confirming', 'lineofcredit')
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df = df.loc[~df["company_id"].isin(hold)]
    assert_no_holdout(df["company_id"])
    rows = []
    for t in TYPES:
        sl = df.loc[df["type"] == t]
        rows.append(
            {
                "type": t,
                "n": int(len(sl)),
                "granted_p50": float(sl["granted_abs"].median()) if len(sl) else float("nan"),
                "out_p50": float(sl["out_abs"].median()) if len(sl) else float("nan"),
                "share_granted_gt1": float((sl["granted_abs"] > 1).mean()) if len(sl) else float("nan"),
                "share_out_gt1": float((sl["out_abs"] > 1).mean()) if len(sl) else float("nan"),
            }
        )
    _ = last
    _ = tr
    return {"rows": rows, "note": "2026-08 extract amounts. Not a 2024 book. Do not revive util."}


def pass19_wave_oct(tr: pd.DataFrame) -> dict:
    """2025-10 connection spike (7 of 20 factoring births)."""
    octm = pd.Timestamp("2025-10-01")
    sl = tr.loc[(tr["period"] == octm) & (tr["new_factoring"] > 0)]
    return {
        "n": int(sl["company_id"].nunique()),
        "n_dark": int((~sl["has_book"]).sum()),
        "med_in3": float(pd.to_numeric(sl["a_in3"], errors="coerce").median()) if len(sl) else float("nan"),
        "y3": _rate(sl, Y3),
        "y2": _rate(sl, Y2),
        "ids": sorted(sl["company_id"].astype(str)),
    }


def pass20_fact_ap_who(tr: pd.DataFrame) -> dict:
    """The 12-ish factoring AP leftovers — how many companies?"""
    on = pd.to_numeric(tr["f_has_factoring"], errors="coerce") > 0
    ap = pd.to_numeric(tr[Y5_AP], errors="coerce") == 1
    left = pd.to_numeric(tr["y5_ap_leftover"], errors="coerce") == 1
    return {
        "n_ap_on": int((on & ap).sum()),
        "n_ap_co": int(tr.loc[on & ap, "company_id"].nunique()),
        "n_left_on": int((on & left).sum()),
        "n_left_co": int(tr.loc[on & left, "company_id"].nunique()),
        "ids": sorted(tr.loc[on & left, "company_id"].astype(str).unique()),
    }


def pass21_leftover_is_group(p14: dict, p20: dict) -> dict:
    """Y5 leftover factoring names vs GROUP_0139 — a group leftover, not a panel law."""
    g_ids = set(p14["ids"])
    left_ids = set(p20["ids"])
    return {
        "n_left": int(len(left_ids)),
        "n_in_g": int(len(left_ids & g_ids)),
        "share": _pct(len(left_ids & g_ids), len(left_ids)),
        "outside": sorted(left_ids - g_ids),
        "group_leftover": bool(len(left_ids) > 0 and len(left_ids & g_ids) / len(left_ids) >= 0.5),
    }


def pass22_siblings(tr: pd.DataFrame) -> dict:
    """Do siblings share WC products? Company tag, not a group book."""
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        on = last.loc[pd.to_numeric(last[flag], errors="coerce") > 0]
        g = on.groupby("group_id").size()
        n_g = int(g.shape[0])
        n_multi = int((g >= 2).sum())
        rows.append(
            {
                "type": t,
                "n_co": int(len(on)),
                "n_groups": n_g,
                "n_multi": n_multi,
                "share_multi": _pct(n_multi, n_g),
            }
        )
    return {"rows": rows}


def pass23_currency(products: pd.DataFrame) -> dict:
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t]
        vc = sl.groupby(sl["currency"].fillna("(null)")).size().sort_values(ascending=False)
        rows.append(
            {
                "type": t,
                "top": [{"ccy": str(i), "n": int(v)} for i, v in vc.head(6).items()],
                "n_ccy": int(sl["currency"].nunique()),
            }
        )
    return {"rows": rows}


def pass25_icc_connected(tr: pd.DataFrame) -> dict:
    """ICC (BETWEEN style) + Y3 singles on months that already have a facility."""
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        icc = icc_anova(tr[flag], tr["company_id"])
        rows.append({"type": t, "icc": icc["icc"], "k": icc["k"], "acf1": median_acf(tr[flag], tr["company_id"], 1)})
    connected = pd.to_numeric(tr["f_n_facilities"], errors="coerce") > 0
    feats = {f: pd.to_numeric(tr[f], errors="coerce") for f in FLAGS}
    feats["log1p_a_in3"] = pd.to_numeric(tr["log_in3"], errors="coerce")
    feats["c_n_days_with_tx"] = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    auc_rows = []
    for name, col in feats.items():
        cv = signed_oof_auroc(tr[Y3], col, tr["fold"], mask=connected)
        auc_rows.append(
            {
                "feature": name,
                "cv": cv["cv"],
                "n_lab": cv["n_defined"],
                "n_pos": cv["n_pos"],
                "low_power": cv["low_power"],
            }
        )
    size_cv = next(r["cv"] for r in auc_rows if r["feature"] == "log1p_a_in3")
    loc = next(r for r in auc_rows if r["feature"] == "f_has_loc")
    return {
        "icc_rows": rows,
        "auc_rows": auc_rows,
        "n_connected": int(connected.sum()),
        "loc_beat": (
            loc["cv"] - size_cv
            if np.isfinite(loc["cv"]) and np.isfinite(size_cv)
            else float("nan")
        ),
        "size_cv": size_cv,
    }


def pass26_y4_size(tr: pd.DataFrame) -> dict:
    """Y4 overlap after size — descriptive, never F as Y4 X (Y4 forbids F)."""
    leak = leakage_check(FLAGS, Y4, forbidden_prefixes=("f",))
    terc = pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    work = tr.assign(_terc=terc)
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        on = pd.to_numeric(work[flag], errors="coerce") > 0
        for tname, g in work.groupby("_terc", observed=False):
            m = on.reindex(g.index).fillna(False)
            a = _rate(g.loc[m], Y4)
            b = _rate(g.loc[~m], Y4)
            rows.append(
                {
                    "type": t,
                    "tercile": str(tname),
                    "n_on": a["n_lab"],
                    "on": a["rate"],
                    "off": b["rate"],
                    "pp": (
                        a["rate"] - b["rate"]
                        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                        else float("nan")
                    ),
                }
            )
    return {"rows": rows, "leak_ok": leak["ok"], "leak": leak}


def pass27_conf_ap_nogroup(tr: pd.DataFrame) -> dict:
    """Confirming Y5 AP after dropping GROUP_0139 — is AP-side a group leftover too?"""
    drop = tr["group_id"] != "GROUP_0139"
    sl = tr.loc[drop]
    on = pd.to_numeric(sl["f_has_confirming"], errors="coerce") > 0
    a = _rate(sl.loc[on], Y5_AP)
    b = _rate(sl.loc[~on], Y5_AP)
    ar_on = _rate(sl.loc[on], Y5_AR)
    ar_off = _rate(sl.loc[~on], Y5_AR)
    fact_on = pd.to_numeric(sl["f_has_factoring"], errors="coerce") > 0
    f_a = _rate(sl.loc[fact_on], Y5_AP)
    f_b = _rate(sl.loc[~fact_on], Y5_AP)
    return {
        "conf_ap_on": a["rate"],
        "conf_ap_off": b["rate"],
        "conf_pp": a["rate"] - b["rate"] if np.isfinite(a["rate"]) and np.isfinite(b["rate"]) else float("nan"),
        "conf_n": a["n_lab"],
        "conf_ar_pp": (
            ar_on["rate"] - ar_off["rate"]
            if np.isfinite(ar_on["rate"]) and np.isfinite(ar_off["rate"])
            else float("nan")
        ),
        "fact_ap_on": f_a["rate"],
        "fact_ap_off": f_b["rate"],
        "fact_pp": f_a["rate"] - f_b["rate"] if np.isfinite(f_a["rate"]) and np.isfinite(f_b["rate"]) else float("nan"),
        "fact_n": f_a["n_lab"],
        "n_dropped": int((tr["group_id"] == "GROUP_0139").sum()),
    }


def pass28_prepost(tr: pd.DataFrame) -> dict:
    """Y3/Y2 on the same companies before vs after first typed connection."""
    rows = []
    s = tr.sort_values(["company_id", "period"]).copy()
    for t in TYPES:
        flag = FLAG_OF[t]
        s["_f"] = pd.to_numeric(s[flag], errors="coerce")
        first = s.loc[s["_f"] > 0].groupby("company_id")["period"].min()
        ever = set(first.index.astype(str))
        sl = s.loc[s["company_id"].isin(ever)].copy()
        sl["first"] = sl["company_id"].map(first)
        before = sl["period"] < sl["first"]
        after = sl["period"] >= sl["first"]
        rec = {"type": t, "n_co": int(len(ever))}
        for y in (Y3, Y2):
            b = _rate(sl.loc[before], y)
            a = _rate(sl.loc[after], y)
            rec[f"{y}_before"] = b["rate"]
            rec[f"{y}_after"] = a["rate"]
            rec[f"{y}_n_b"] = b["n_lab"]
            rec[f"{y}_n_a"] = a["n_lab"]
            rec[f"{y}_pp"] = (
                a["rate"] - b["rate"]
                if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                else float("nan")
            )
        rows.append(rec)
    return {"rows": rows}


def pass29_holdout_cov(ho: pd.DataFrame) -> dict:
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        x = pd.to_numeric(ho[flag], errors="coerce")
        rows.append(
            {
                "type": t,
                "ever": int(ho.loc[x > 0, "company_id"].nunique()),
                "cm": int((x > 0).sum()),
                "n_co": int(ho["company_id"].nunique()),
            }
        )
    return {"rows": rows, "n_co": int(ho["company_id"].nunique()), "n_cm": int(len(ho))}


def pass57_holdout_ids(ho: pd.DataFrame) -> dict:
    """Holdout WC names — coverage only. No AUROC. No fit."""
    last = ho.sort_values("period").groupby("company_id", sort=False).last()
    rows = []
    for cid, r in last.iterrows():
        flags = []
        if r.get("f_has_factoring", 0) > 0:
            flags.append("factoring")
        if r.get("f_has_confirming", 0) > 0:
            flags.append("confirming")
        if r.get("f_has_loc", 0) > 0:
            flags.append("loc")
        if not flags:
            continue
        rows.append(
            {
                "company_id": str(cid),
                "group_id": str(r.get("group_id", "")),
                "has_book": bool(r.get("has_book", False)),
                "flags": "+".join(flags),
                "in3": float(pd.to_numeric(r.get("a_in3"), errors="coerce")),
            }
        )
    rows.sort(key=lambda x: x["company_id"])
    fact = [r for r in rows if "factoring" in r["flags"]]
    gvc = {}
    for r in rows:
        gvc[r["group_id"]] = gvc.get(r["group_id"], 0) + 1
    top_g = sorted(gvc.items(), key=lambda kv: -kv[1])
    return {
        "n": int(len(rows)),
        "rows": rows,
        "n_fact": int(len(fact)),
        "fact_ids": [r["company_id"] for r in fact],
        "fact_flags": fact[0]["flags"] if fact else "",
        "top_group": top_g[0][0] if top_g else "",
        "top_group_n": int(top_g[0][1]) if top_g else 0,
    }


def pass58_holdout_group_unseen(tr: pd.DataFrame, p57: dict) -> dict:
    """Holdout WC groups must be unseen on train (whole-group holdout)."""
    train_g = set(tr["group_id"].astype(str))
    hold_g = {r["group_id"] for r in p57["rows"]}
    overlap = sorted(hold_g & train_g)
    return {
        "n_hold_g": int(len(hold_g)),
        "n_overlap": int(len(overlap)),
        "overlap": overlap,
        "unseen": bool(len(overlap) == 0),
        "top_in_train": p57["top_group"] in train_g,
    }


def pass30_fx_overlap_size(tr: pd.DataFrame) -> dict:
    """Ever-FX 228 ∩ WC — leftover size? Descriptive."""
    fx = set(tr.loc[pd.to_numeric(tr["e_fx_share"], errors="coerce") > 0, "company_id"])
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    last_log = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        ever = set(tr.loc[pd.to_numeric(tr[flag], errors="coerce") > 0, "company_id"])
        both = ever & fx
        only_wc = ever - fx
        rows.append(
            {
                "type": t,
                "n_wc": int(len(ever)),
                "n_both": int(len(both)),
                "share": _pct(len(both), len(ever)),
                "both_med": float(last_log.reindex(list(both)).median()) if both else float("nan"),
                "wc_med": float(last_log.reindex(list(only_wc)).median()) if only_wc else float("nan"),
                "fx_med": float(last_log.reindex(list(fx)).median()) if fx else float("nan"),
            }
        )
    return {"rows": rows, "n_fx": int(len(fx))}


def pass31_prepost_y2_ctrl(tr: pd.DataFrame) -> dict:
    """Factoring Y2 20.3%→10.7% after connect — size / calendar, not a Q3 KEEP."""
    s = tr.sort_values(["company_id", "period"]).copy()
    flag = "f_has_factoring"
    s["_f"] = pd.to_numeric(s[flag], errors="coerce")
    first = s.loc[s["_f"] > 0].groupby("company_id")["period"].min()
    sl = s.loc[s["company_id"].isin(first.index)].copy()
    sl["first"] = sl["company_id"].map(first)
    sl["after"] = sl["period"] >= sl["first"]
    sl["_terc"] = pd.qcut(sl["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    sl["year"] = pd.to_datetime(sl["period"]).dt.year
    terc_rows = []
    for tname, g in sl.groupby("_terc", observed=False):
        b = _rate(g.loc[~g["after"]], Y2)
        a = _rate(g.loc[g["after"]], Y2)
        terc_rows.append(
            {
                "slice": str(tname),
                "n_b": b["n_lab"],
                "n_a": a["n_lab"],
                "before": b["rate"],
                "after": a["rate"],
                "pp": a["rate"] - b["rate"] if np.isfinite(a["rate"]) and np.isfinite(b["rate"]) else float("nan"),
            }
        )
    year_rows = []
    for y, g in sl.groupby("year"):
        b = _rate(g.loc[~g["after"]], Y2)
        a = _rate(g.loc[g["after"]], Y2)
        year_rows.append(
            {
                "slice": str(int(y)),
                "n_b": b["n_lab"],
                "n_a": a["n_lab"],
                "before": b["rate"],
                "after": a["rate"],
                "pp": a["rate"] - b["rate"] if np.isfinite(a["rate"]) and np.isfinite(b["rate"]) else float("nan"),
            }
        )
    # rest of train Y2 by year (calendar base)
    base = []
    work = tr.copy()
    work["year"] = pd.to_datetime(work["period"]).dt.year
    for y, g in work.groupby("year"):
        r = _rate(g, Y2)
        base.append({"year": str(int(y)), "rate": r["rate"], "n": r["n_lab"]})
    return {"terc": terc_rows, "year": year_rows, "base": base}


def pass32_never_recover(tr: pd.DataFrame) -> dict:
    """Are the 18 factoring companies a never-recoverer set (no Y3+ anywhere)?"""
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    ever_pos = set(tr.loc[y3 == 1, "company_id"])
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        ever = set(tr.loc[pd.to_numeric(tr[flag], errors="coerce") > 0, "company_id"])
        rows.append(
            {
                "type": t,
                "n": int(len(ever)),
                "n_ever_y3": int(len(ever & ever_pos)),
                "share": _pct(len(ever & ever_pos), len(ever)),
            }
        )
    rest = set(tr["company_id"]) - set(tr.loc[pd.to_numeric(tr["f_has_factoring"], errors="coerce") > 0, "company_id"])
    return {
        "rows": rows,
        "rest_share": _pct(len(rest & ever_pos), len(rest)),
        "n_ever_y3_co": int(len(ever_pos)),
    }


def pass33_trail(tr: pd.DataFrame) -> dict:
    """Months on the cash trail for WC vs rest."""
    last = tr.sort_values("period").groupby("company_id", sort=False).agg(
        n_mo=("period", "size"),
        med_in3=("a_in3", "median"),
        med_days=("c_n_days_with_tx", "median"),
        fact=("f_has_factoring", "max"),
        conf=("f_has_confirming", "max"),
        loc=("f_has_loc", "max"),
    )
    rows = []
    for name, mask in (
        ("factoring", last["fact"] > 0),
        ("confirming", last["conf"] > 0),
        ("loc", last["loc"] > 0),
        ("rest", (last["fact"] == 0) & (last["conf"] == 0) & (last["loc"] == 0)),
    ):
        sl = last.loc[mask]
        rows.append(
            {
                "slice": name,
                "n": int(len(sl)),
                "p50_mo": float(sl["n_mo"].median()) if len(sl) else float("nan"),
                "p50_in3": float(pd.to_numeric(sl["med_in3"], errors="coerce").median()) if len(sl) else float("nan"),
                "p50_days": float(pd.to_numeric(sl["med_days"], errors="coerce").median()) if len(sl) else float("nan"),
            }
        )
    return {"rows": rows}


def pass34_comp0919(tr: pd.DataFrame) -> dict:
    """The leftover factoring company outside GROUP_0139."""
    sl = tr.loc[tr["company_id"] == "COMP_0919"]
    if sl.empty:
        return {"n": 0}
    last = sl.sort_values("period").iloc[-1]
    return {
        "n": int(len(sl)),
        "group": str(sl["group_id"].iloc[0]),
        "has_book": bool(sl["has_book"].iloc[0]),
        "fact_cm": int((pd.to_numeric(sl["f_has_factoring"], errors="coerce") > 0).sum()),
        "conf_cm": int((pd.to_numeric(sl["f_has_confirming"], errors="coerce") > 0).sum()),
        "y5_ap": int((pd.to_numeric(sl[Y5_AP], errors="coerce") == 1).sum()),
        "y5_left": int((pd.to_numeric(sl["y5_ap_leftover"], errors="coerce") == 1).sum()),
        "y3_pos": int((pd.to_numeric(sl[Y3], errors="coerce") == 1).sum()),
        "y2_pos": int((pd.to_numeric(sl[Y2], errors="coerce") == 1).sum()),
        "last_in3": float(pd.to_numeric(last["a_in3"], errors="coerce")),
        "mix": str(sl["group_mix"].iloc[0]) if "group_mix" in sl.columns else "",
    }


def pass35_nprod(products: pd.DataFrame) -> dict:
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t]
        per = sl.groupby("company_id").size()
        rows.append(
            {
                "type": t,
                "n_co": int(per.shape[0]),
                "p50": float(per.median()) if len(per) else float("nan"),
                "max": int(per.max()) if len(per) else 0,
                "share_multi": float((per >= 2).mean()) if len(per) else float("nan"),
            }
        )
    return {"rows": rows}


def pass36_company_onboard(con, products: pd.DataFrame) -> dict:
    """Product created_at minus company.created_at. Read-only companies. Do not edit companies_qa."""
    hold = load_holdout()
    cos = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(created_at AS TIMESTAMP) AS company_created
        FROM companies
        """
    ).df()
    cos["company_id"] = cos["company_id"].astype(str)
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)].copy()
    live = live.merge(cos, on="company_id", how="left")
    live["lag_d"] = (live["created_at"] - pd.to_datetime(live["company_created"])).dt.total_seconds() / 86400.0
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t, "lag_d"].dropna()
        rows.append(
            {
                "type": t,
                "n": int(len(sl)),
                "p50_d": float(sl.median()) if len(sl) else float("nan"),
                "share_after_co": float((sl >= 0).mean()) if len(sl) else float("nan"),
            }
        )
    return {"rows": rows}


def pass37_y3_coverage(tr: pd.DataFrame) -> dict:
    """Are factoring companies unstressed (no Y3 label) or stressed-never-recover?"""
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    rows = []
    for t in TYPES + ("rest",):
        if t == "rest":
            mask_co = ~tr["company_id"].isin(
                set(tr.loc[pd.to_numeric(tr["f_has_factoring"], errors="coerce") > 0, "company_id"])
                | set(tr.loc[pd.to_numeric(tr["f_has_confirming"], errors="coerce") > 0, "company_id"])
                | set(tr.loc[pd.to_numeric(tr["f_has_loc"], errors="coerce") > 0, "company_id"])
            )
            sl = tr.loc[mask_co]
        else:
            ever = set(tr.loc[pd.to_numeric(tr[FLAG_OF[t]], errors="coerce") > 0, "company_id"])
            sl = tr.loc[tr["company_id"].isin(ever)]
        lab = y3.reindex(sl.index)
        rows.append(
            {
                "slice": t,
                "n_cm": int(len(sl)),
                "n_lab": int(lab.notna().sum()),
                "lab_share": _pct(int(lab.notna().sum()), int(len(sl))),
                "n_pos": int((lab == 1).sum()),
                "rate_among_lab": _pct(int((lab == 1).sum()), int(lab.notna().sum())),
            }
        )
    return {"rows": rows}


def pass38_labels(products: pd.DataFrame) -> dict:
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t]
        vc = sl.groupby(sl["label"].fillna("(null)")).size().sort_values(ascending=False)
        rows.append(
            {
                "type": t,
                "n_lab": int(sl["label"].nunique()),
                "top": [{"label": str(i), "n": int(v)} for i, v in vc.head(6).items()],
            }
        )
    return {"rows": rows}


def pass39_size_matched_never(tr: pd.DataFrame) -> dict:
    """Is 0/18 never-recover just 'large firms rarely recover'?"""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    last = last.loc[last["log_in3"].notna()].copy()
    last["terc"] = pd.qcut(last["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    ever_pos = set(tr.loc[y3 == 1, "company_id"])
    fact = set(tr.loc[pd.to_numeric(tr["f_has_factoring"], errors="coerce") > 0, "company_id"])
    last["ever_y3"] = last.index.astype(str).isin(ever_pos)
    last["fact"] = last.index.astype(str).isin(fact)
    rows = []
    for tname, g in last.groupby("terc", observed=False):
        rest = g.loc[~g["fact"]]
        fa = g.loc[g["fact"]]
        rows.append(
            {
                "tercile": str(tname),
                "n_fact": int(len(fa)),
                "fact_y3": float(fa["ever_y3"].mean()) if len(fa) else float("nan"),
                "n_rest": int(len(rest)),
                "rest_y3": float(rest["ever_y3"].mean()) if len(rest) else float("nan"),
            }
        )
    return {"rows": rows}


def pass40_loc_inhouse(tr: pd.DataFrame, products: pd.DataFrame) -> dict:
    """In-house / Other LOC vs bank LOC — still not a health X."""
    hold = load_holdout()
    live = products.loc[
        (~products["created_after_snapshot"])
        & (~products["company_id"].isin(hold))
        & (products["type"] == "lineofcredit")
    ].copy()
    live["inhouse"] = live["bank_name"].fillna("").str.contains(
        "In-house|customer-defined|Other", case=False
    )
    ih = set(live.loc[live["inhouse"], "company_id"])
    bk = set(live.loc[~live["inhouse"], "company_id"]) - ih
    rows = []
    for name, ids in (("inhouse_or_other", ih), ("bank_loc", bk)):
        sl = tr[tr["company_id"].isin(ids)]
        on = sl["f_has_loc"].fillna(0) > 0
        rows.append(
            {
                "slice": name,
                "n_co": int(len(ids)),
                "y3": _rate(sl.loc[on], Y3),
                "y2": _rate(sl.loc[on], Y2),
            }
        )
    return {"rows": rows}


def pass41_liquidity(products: pd.DataFrame) -> dict:
    """Extract `liquidity` on WC rows — not a path, not util, not a Y."""
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t]
        liq = pd.to_numeric(sl["liquidity"], errors="coerce")
        rows.append(
            {
                "type": t,
                "n": int(len(sl)),
                "n_def": int(liq.notna().sum()),
                "share_def": _pct(int(liq.notna().sum()), int(len(sl))),
                "p50": float(liq.median()) if liq.notna().any() else float("nan"),
                "share_gt1": float((liq.fillna(0) > 1).mean()) if len(sl) else float("nan"),
                "share_zero": float((liq.fillna(0) == 0).mean()) if len(sl) else float("nan"),
            }
        )
    return {"rows": rows, "note": "Snapshot liquidity. PARK with util. Do not build Y10 cousin."}


def pass42_service(products: pd.DataFrame) -> dict:
    hold = load_holdout()
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    rows = []
    for t in TYPES:
        sl = live.loc[live["type"] == t]
        vc = sl.groupby(sl["service"].fillna("(null)")).size().sort_values(ascending=False)
        rows.append(
            {
                "type": t,
                "n": int(len(sl)),
                "n_svc": int(sl["service"].nunique()),
                "custom": int((sl["service"].fillna("").str.lower() == "custom").sum()),
                "top": [{"svc": str(i), "n": int(v)} for i, v in vc.head(5).items()],
            }
        )
    return {"rows": rows}


def pass43_flows(tr: pd.DataFrame) -> dict:
    """Do WC-flag months show observed f_ds_r / f_fc_r? Flow is KEEP; flags are not."""
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        on = pd.to_numeric(tr[flag], errors="coerce") > 0
        for col in ("f_ds_r", "f_fc_r"):
            x = pd.to_numeric(tr[col], errors="coerce")
            a = x[on]
            b = x[~on]
            rows.append(
                {
                    "type": t,
                    "col": col,
                    "n_on": int(a.notna().sum()),
                    "p50_on": float(a.median()) if a.notna().any() else float("nan"),
                    "share_gt0_on": float((a.fillna(0) > 0).mean()) if len(a) else float("nan"),
                    "p50_off": float(b.median()) if b.notna().any() else float("nan"),
                    "share_gt0_off": float((b.fillna(0) > 0).mean()) if len(b) else float("nan"),
                }
            )
    return {"rows": rows}


def pass44_fact_only_flow(tr: pd.DataFrame) -> dict:
    """3 last-month factoring-only companies — is f_ds_r from other facilities?"""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    fact = last["f_has_factoring"].fillna(0) > 0
    conf = last["f_has_confirming"].fillna(0) > 0
    loc = last["f_has_loc"].fillna(0) > 0
    only = set(last.index[fact & ~conf & ~loc].astype(str))
    both = set(last.index[fact & (conf | loc)].astype(str))
    sl_o = tr[tr["company_id"].isin(only)]
    sl_b = tr[tr["company_id"].isin(both)]
    on_o = sl_o["f_has_factoring"].fillna(0) > 0
    on_b = sl_b["f_has_factoring"].fillna(0) > 0
    ds_o = pd.to_numeric(sl_o.loc[on_o, "f_ds_r"], errors="coerce")
    ds_b = pd.to_numeric(sl_b.loc[on_b, "f_ds_r"], errors="coerce")
    return {
        "n_only": int(len(only)),
        "n_both": int(len(both)),
        "ids_only": sorted(only),
        "ds_gt0_only": float((ds_o.fillna(0) > 0).mean()) if len(ds_o) else float("nan"),
        "ds_gt0_both": float((ds_b.fillna(0) > 0).mean()) if len(ds_b) else float("nan"),
        "n_on_only": int(on_o.sum()),
        "n_on_both": int(on_b.sum()),
    }


def pass45_other_debt(con, only_ids: list[str]) -> dict:
    """Other debt types on the 3 factoring-only companies. Read-only."""
    if not only_ids:
        return {"rows": [], "n": 0}
    q = ",".join(["?"] * len(only_ids))
    df = con.execute(
        f"""
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(type AS VARCHAR) AS type,
               COUNT(*) AS n
        FROM debt_products
        WHERE CAST(company_id AS VARCHAR) IN ({q})
          AND NOT coalesce(created_after_snapshot, false)
        GROUP BY 1, 2
        """,
        only_ids,
    ).df()
    return {
        "n": int(len(only_ids)),
        "rows": [
            {"company_id": str(r.company_id), "type": str(r.type), "n": int(r.n)}
            for r in df.itertuples(index=False)
        ],
    }


def pass46_conf_only_ap(tr: pd.DataFrame) -> dict:
    """Confirming without factoring/LOC — cleaner AP-side test."""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    conf = last["f_has_confirming"].fillna(0) > 0
    fact = last["f_has_factoring"].fillna(0) > 0
    loc = last["f_has_loc"].fillna(0) > 0
    only = set(last.index[conf & ~fact & ~loc].astype(str))
    sl = tr[tr["company_id"].isin(only)]
    on = sl["f_has_confirming"].fillna(0) > 0
    return {
        "n_co": int(len(only)),
        "ap": _rate(sl.loc[on], Y5_AP),
        "ap_off": _rate(sl.loc[~on], Y5_AP),
        "ar": _rate(sl.loc[on], Y5_AR),
        "ar_off": _rate(sl.loc[~on], Y5_AR),
        "y3": _rate(sl.loc[on], Y3),
    }


def pass47_conf_stack_ap(tr: pd.DataFrame) -> dict:
    """Confirming AP: only vs stacked (also fact or loc). Within-company off is small-n."""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    conf = last["f_has_confirming"].fillna(0) > 0
    fact = last["f_has_factoring"].fillna(0) > 0
    loc = last["f_has_loc"].fillna(0) > 0
    only = set(last.index[conf & ~fact & ~loc].astype(str))
    stacked = set(last.index[conf & (fact | loc)].astype(str))
    rest = set(last.index[~conf].astype(str))
    rows = []
    for name, ids in (("confirming_only", only), ("confirming_stacked", stacked), ("no_confirming", rest)):
        sl = tr[tr["company_id"].isin(ids)]
        on = sl["f_has_confirming"].fillna(0) > 0 if name != "no_confirming" else sl["company_id"].notna()
        ap = _rate(sl.loc[on], Y5_AP)
        ar = _rate(sl.loc[on], Y5_AR)
        rows.append(
            {
                "slice": name,
                "n_co": int(len(ids)),
                "n_cm": int(on.sum()) if name != "no_confirming" else int(len(sl)),
                "ap": ap["rate"],
                "ap_n": ap["n_lab"],
                "ar": ar["rate"],
                "ar_n": ar["n_lab"],
            }
        )
    only_ap = next(r["ap"] for r in rows if r["slice"] == "confirming_only")
    stack_ap = next(r["ap"] for r in rows if r["slice"] == "confirming_stacked")
    rest_ap = next(r["ap"] for r in rows if r["slice"] == "no_confirming")
    only_ok = (
        np.isfinite(only_ap)
        and np.isfinite(rest_ap)
        and (only_ap - rest_ap) >= 0.02
    )
    return {
        "rows": rows,
        "only_ap": only_ap,
        "stack_ap": stack_ap,
        "rest_ap": rest_ap,
        "only_is_ap": bool(only_ok),
    }


def pass48_stack_ap_size(tr: pd.DataFrame) -> dict:
    """Does stacked-confirming AP lift survive size terciles?"""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    conf = last["f_has_confirming"].fillna(0) > 0
    fact = last["f_has_factoring"].fillna(0) > 0
    loc = last["f_has_loc"].fillna(0) > 0
    stacked = set(last.index[conf & (fact | loc)].astype(str))
    only = set(last.index[conf & ~fact & ~loc].astype(str))
    work = tr.assign(_terc=pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop"))
    rows = []
    for name, ids in (("confirming_only", only), ("confirming_stacked", stacked)):
        on = work["company_id"].isin(ids) & (pd.to_numeric(work["f_has_confirming"], errors="coerce") > 0)
        deltas, w = [], []
        for tname, g in work.groupby("_terc", observed=False):
            m = on.reindex(g.index).fillna(False)
            a = _rate(g.loc[m], Y5_AP)
            b = _rate(g.loc[~m], Y5_AP)
            pp = (
                a["rate"] - b["rate"]
                if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                else float("nan")
            )
            rows.append(
                {
                    "slice": name,
                    "tercile": str(tname),
                    "n_on": a["n_lab"],
                    "ap_on": a["rate"],
                    "ap_off": b["rate"],
                    "pp": pp,
                }
            )
            if np.isfinite(pp) and a["n_lab"] > 0:
                deltas.append(pp)
                w.append(a["n_lab"])
        rows.append(
            {
                "slice": name,
                "tercile": "size-w",
                "n_on": int(sum(r["n_on"] for r in rows if r["slice"] == name and r["tercile"] != "size-w")),
                "ap_on": float("nan"),
                "ap_off": float("nan"),
                "pp": float(np.average(deltas, weights=w)) if deltas else float("nan"),
            }
        )
    stack_w = next(r["pp"] for r in rows if r["slice"] == "confirming_stacked" and r["tercile"] == "size-w")
    only_w = next(r["pp"] for r in rows if r["slice"] == "confirming_only" and r["tercile"] == "size-w")
    return {"rows": rows, "stack_pp_size": stack_w, "only_pp_size": only_w}


def pass49_unused_limit(con) -> dict:
    """(granted-outstanding)/granted on 2026-08 extract. PARK — not util, not a Y."""
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               CAST(type AS VARCHAR) AS type,
               abs(granted) AS granted_abs,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type IN ('factoring', 'confirming', 'lineofcredit')
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df = df.loc[~df["company_id"].isin(hold)]
    assert_no_holdout(df["company_id"])
    rows = []
    for t in TYPES:
        sl = df.loc[df["type"] == t].copy()
        g = sl["granted_abs"]
        o = sl["out_abs"]
        ok = g > 1
        unused = (g - o) / g
        rows.append(
            {
                "type": t,
                "n": int(len(sl)),
                "n_granted": int(ok.sum()),
                "share_unused_ge90": float((unused.loc[ok] >= 0.90).mean()) if ok.any() else float("nan"),
                "unused_p50": float(unused.loc[ok].median()) if ok.any() else float("nan"),
                "share_drawn": float((o > 1).mean()) if len(sl) else float("nan"),
            }
        )
    return {"rows": rows, "note": "PARK with Y10. Connected limit, not a drawn book."}


def pass50_loc_only(tr: pd.DataFrame) -> dict:
    """LOC without factoring/confirming — leftover inverse-size, not a WC tool."""
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    loc = last["f_has_loc"].fillna(0) > 0
    fact = last["f_has_factoring"].fillna(0) > 0
    conf = last["f_has_confirming"].fillna(0) > 0
    only = set(last.index[loc & ~fact & ~conf].astype(str))
    sl = tr[tr["company_id"].isin(only)]
    on = sl["f_has_loc"].fillna(0) > 0
    work = sl.assign(_terc=pd.qcut(sl["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop"))
    terc = []
    for tname, g in work.groupby("_terc", observed=False):
        m = on.reindex(g.index).fillna(False)
        terc.append(
            {
                "tercile": str(tname),
                "n_on": int(m.sum()),
                "y3": _rate(g.loc[m], Y3),
                "y2": _rate(g.loc[m], Y2),
            }
        )
    return {
        "n_co": int(len(only)),
        "y3": _rate(sl.loc[on], Y3),
        "y2": _rate(sl.loc[on], Y2),
        "terc": terc,
    }


def pass51_conf_banks(tr: pd.DataFrame, products: pd.DataFrame) -> dict:
    """Same confirming banks on only vs stacked? Tool identity vs stack identity."""
    hold = load_holdout()
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    conf = last["f_has_confirming"].fillna(0) > 0
    fact = last["f_has_factoring"].fillna(0) > 0
    loc = last["f_has_loc"].fillna(0) > 0
    only = set(last.index[conf & ~fact & ~loc].astype(str))
    stacked = set(last.index[conf & (fact | loc)].astype(str))
    live = products.loc[
        (~products["created_after_snapshot"])
        & (~products["company_id"].isin(hold))
        & (products["type"] == "confirming")
    ]
    rows = []
    for name, ids in (("confirming_only", only), ("confirming_stacked", stacked)):
        sl = live.loc[live["company_id"].isin(ids)]
        vc = (
            sl.groupby(sl["bank_name"].fillna("(null)"))
            .agg(n=("product_id", "size"), n_co=("company_id", "nunique"))
            .reset_index()
            .sort_values("n", ascending=False)
        )
        rows.append(
            {
                "slice": name,
                "n_co": int(len(ids)),
                "n_prod": int(len(sl)),
                "n_banks": int(sl["bank_name"].nunique()),
                "empresas": int(sl["bank_name"].fillna("").str.contains("Empresas", case=False).sum()),
                "top": [
                    {"bank": str(r.bank_name), "n": int(r.n), "n_co": int(r.n_co)}
                    for r in vc.head(5).itertuples(index=False)
                ],
            }
        )
    return {"rows": rows}


def pass52_drawn_factoring(con, tr: pd.DataFrame) -> dict:
    """Drawn factoring (outstanding>1) vs connected-only. Still not a Y."""
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type = 'factoring'
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df = df.loc[~df["company_id"].isin(hold)]
    drawn = set(df.loc[df["out_abs"] > 1, "company_id"])
    connected = set(df["company_id"]) - drawn
    rows = []
    for name, ids in (("drawn", drawn), ("connected_only", connected)):
        sl = tr[tr["company_id"].isin(ids)]
        on = sl["f_has_factoring"].fillna(0) > 0
        rows.append(
            {
                "slice": name,
                "n_co": int(len(ids)),
                "y3": _rate(sl.loc[on], Y3),
                "y2": _rate(sl.loc[on], Y2),
                "ap": _rate(sl.loc[on], Y5_AP),
                "left": _rate(sl.loc[on], "y5_ap_leftover"),
            }
        )
    return {"rows": rows}


def pass53_drawn_size(con, tr: pd.DataFrame) -> dict:
    """Drawn-factoring Y2 / AP after size. n=8 — do not KEEP a pocket."""
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type = 'factoring'
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    drawn = set(df.loc[(~df["company_id"].isin(hold)) & (df["out_abs"] > 1), "company_id"])
    work = tr.assign(_terc=pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop"))
    on = work["company_id"].isin(drawn) & (pd.to_numeric(work["f_has_factoring"], errors="coerce") > 0)
    rows = []
    for y in (Y2, Y5_AP):
        deltas, w = [], []
        for tname, g in work.groupby("_terc", observed=False):
            m = on.reindex(g.index).fillna(False)
            a = _rate(g.loc[m], y)
            b = _rate(g.loc[~m], y)
            pp = (
                a["rate"] - b["rate"]
                if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
                else float("nan")
            )
            rows.append(
                {
                    "y": y,
                    "tercile": str(tname),
                    "n_on": a["n_lab"],
                    "rate_on": a["rate"],
                    "rate_off": b["rate"],
                    "pp": pp,
                }
            )
            if np.isfinite(pp) and a["n_lab"] > 0:
                deltas.append(pp)
                w.append(a["n_lab"])
        rows.append(
            {
                "y": y,
                "tercile": "size-w",
                "n_on": int(sum(r["n_on"] for r in rows if r["y"] == y and r["tercile"] != "size-w")),
                "rate_on": float("nan"),
                "rate_off": float("nan"),
                "pp": float(np.average(deltas, weights=w)) if deltas else float("nan"),
            }
        )
    y2w = next(r["pp"] for r in rows if r["y"] == Y2 and r["tercile"] == "size-w")
    return {"rows": rows, "n_drawn": int(len(drawn)), "y2_pp_size": y2w}


def pass54_drawn_wo_g0139(con, tr: pd.DataFrame) -> dict:
    """Drawn Y2/AP without GROUP_0139 leftover names."""
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type = 'factoring'
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    drawn = set(df.loc[(~df["company_id"].isin(hold)) & (df["out_abs"] > 1), "company_id"])
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    g = set(last.index[last["group_id"].astype(str) == "GROUP_0139"].astype(str))
    wo = drawn - g
    sl = tr[tr["company_id"].isin(wo)]
    on = sl["f_has_factoring"].fillna(0) > 0
    return {
        "n_drawn": int(len(drawn)),
        "n_g": int(len(drawn & g)),
        "n_wo": int(len(wo)),
        "ids_wo": sorted(wo),
        "y2": _rate(sl.loc[on], Y2),
        "ap": _rate(sl.loc[on], Y5_AP),
        "y3": _rate(sl.loc[on], Y3),
    }


def pass55_drawn_confirming(con, tr: pd.DataFrame) -> dict:
    """Drawn confirming (outstanding>1) vs connected-only — same extract PARK."""
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type = 'confirming'
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df = df.loc[~df["company_id"].isin(hold)]
    drawn = set(df.loc[df["out_abs"] > 1, "company_id"])
    connected = set(df["company_id"]) - drawn
    rows = []
    for name, ids in (("drawn", drawn), ("connected_only", connected)):
        sl = tr[tr["company_id"].isin(ids)]
        on = sl["f_has_confirming"].fillna(0) > 0
        rows.append(
            {
                "slice": name,
                "n_co": int(len(ids)),
                "y3": _rate(sl.loc[on], Y3),
                "y2": _rate(sl.loc[on], Y2),
                "ap": _rate(sl.loc[on], Y5_AP),
            }
        )
    return {"rows": rows}


def pass56_drawn_conf_y2_size(con, tr: pd.DataFrame) -> dict:
    """Drawn confirming Y2 after size. PARK if it dies or is leftover size."""
    hold = load_holdout()
    df = con.execute(
        """
        SELECT CAST(company_id AS VARCHAR) AS company_id,
               abs(outstanding) AS out_abs
        FROM debt_products
        WHERE type = 'confirming'
          AND NOT coalesce(created_after_snapshot, false)
          AND CAST(created_at AS DATE) <= DATE '2026-08-31'
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    drawn = set(df.loc[(~df["company_id"].isin(hold)) & (df["out_abs"] > 1), "company_id"])
    work = tr.assign(_terc=pd.qcut(tr["log_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop"))
    on = work["company_id"].isin(drawn) & (pd.to_numeric(work["f_has_confirming"], errors="coerce") > 0)
    rows = []
    deltas, w = [], []
    for tname, g in work.groupby("_terc", observed=False):
        m = on.reindex(g.index).fillna(False)
        a = _rate(g.loc[m], Y2)
        b = _rate(g.loc[~m], Y2)
        pp = (
            a["rate"] - b["rate"]
            if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
            else float("nan")
        )
        rows.append(
            {
                "tercile": str(tname),
                "n_on": a["n_lab"],
                "on": a["rate"],
                "off": b["rate"],
                "pp": pp,
            }
        )
        if np.isfinite(pp) and a["n_lab"] > 0:
            deltas.append(pp)
            w.append(a["n_lab"])
    ppw = float(np.average(deltas, weights=w)) if deltas else float("nan")
    rows.append(
        {
            "tercile": "size-w",
            "n_on": int(sum(r["n_on"] for r in rows)),
            "on": float("nan"),
            "off": float("nan"),
            "pp": ppw,
        }
    )
    return {"rows": rows, "n_drawn": int(len(drawn)), "y2_pp_size": ppw}


def pass24_store_vs_live(tr: pd.DataFrame, products: pd.DataFrame) -> dict:
    """Last-month store flag vs live created_at inventory. No parquet rewrite."""
    hold = load_holdout()
    last_p = pd.to_datetime(tr["period"]).max()
    last = tr.loc[pd.to_datetime(tr["period"]) == last_p]
    live = products.loc[~products["created_after_snapshot"] & ~products["company_id"].isin(hold)]
    live = live.loc[live["created_at"].dt.normalize() <= (last_p + pd.offsets.MonthEnd(0))]
    rows = []
    for t in TYPES:
        flag = FLAG_OF[t]
        store_on = set(last.loc[pd.to_numeric(last[flag], errors="coerce") > 0, "company_id"])
        live_on = set(live.loc[live["type"] == t, "company_id"])
        rows.append(
            {
                "type": t,
                "store": int(len(store_on)),
                "live": int(len(live_on)),
                "only_store": int(len(store_on - live_on)),
                "only_live": int(len(live_on - store_on)),
                "agree": int(len(store_on & live_on)),
            }
        )
    return {"rows": rows, "last": str(pd.Timestamp(last_p).date()), "mismatch": any(r["only_store"] or r["only_live"] for r in rows)}


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["log_in3"] = np.log1p(pd.to_numeric(last["a_in3"], errors="coerce").clip(lower=0))
    last = last.loc[last["log_in3"].notna()].copy()
    last["tercile"] = pd.qcut(last["log_in3"], 3, labels=["T1 small", "T2 mid", "T3 large"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.4))
    xs = np.arange(3)
    w = 0.25
    colors = {"f_has_factoring": "#1f4e79", "f_has_confirming": "#c45911", "f_has_loc": "#548235"}
    labels = {"f_has_factoring": "factoring", "f_has_confirming": "confirming", "f_has_loc": "LOC"}
    ax = axes[0]
    for i, flag in enumerate(FLAGS):
        s = last.groupby("tercile", observed=False)[flag].mean().reindex(
            ["T1 small", "T2 mid", "T3 large"]
        )
        ax.bar(xs + (i - 1) * w, 100.0 * s.to_numpy(dtype=float), width=w, color=colors[flag], label=labels[flag])
    ax.set_xticks(xs)
    ax.set_xticklabels(["T1 small", "T2 mid", "T3 large"])
    ax.set_ylabel("% of train companies (last month)")
    ax.set_title("Has product vs size tercile (train last month)")
    ax.legend(framealpha=0.9, fontsize=8)

    ax = axes[1]
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = tr.loc[y.notna()].copy()
    rates = []
    names = []
    for flag in FLAGS:
        on = pd.to_numeric(lab[flag], errors="coerce") > 0
        rates.append(100.0 * float(y.reindex(lab.index)[on].mean()) if on.any() else 0.0)
        names.append(labels[flag] + "=1")
        rates.append(100.0 * float(y.reindex(lab.index)[~on].mean()) if (~on).any() else 0.0)
        names.append(labels[flag] + "=0")
    ys = np.arange(len(names))
    ax.barh(ys, rates, color=["#1f4e79", "#9dc3e6"] * 3)
    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Y3 recover rate (stressed, train)")
    ax.set_title("Y3 recover rate by inventory flag (fact=1 is 0)")
    fig.suptitle("DROP from 44 — rise-only connection inventory, not health X", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    return True


def _decisions(p1, p2, p3, p4, p9, p47=None) -> list[dict]:
    size_map = {r["flag"]: r for r in p2["rows"]}
    dummy_map = {FLAG_OF[r["type"]]: r for r in p9["rows"]}
    verd_map = {v["feature"]: v for v in p3["verdicts"]}
    out = []
    for t in TYPES:
        flag = FLAG_OF[t]
        row = next(r for r in p1["rows"] if r["type"] == t)
        sz = size_map[flag]
        dum = dummy_map[flag]
        v = verd_map[flag]
        keep44 = bool(v["keep_44"] and not dum["group_dummy"])
        out.append(
            {
                "object": f"{flag} as 44-col Y3 X",
                "decision": "KEEP" if keep44 else "CLOSE",
                "why": (
                    f"Y3 CV {_f(v['cv'])} vs size {_f(v['size_cv'])} "
                    f"(Δ {_f(v['beat_size'])}); SIZE={sz['is_size']}; "
                    f"group_dummy={dum['group_dummy']}; ever_n={row['ever_n']}"
                ),
            }
        )
    q3 = next(r for r in p4["rows"] if r["flag"] == "new_wc")
    out.append(
        {
            "object": "new factoring/confirming month as Q3 footnote",
            "decision": "KEEP footnote" if p4["any_q3_keep"] else "CLOSE",
            "why": (
                f"new_wc size-controlled Y3 Δ={_pp(q3['y3_recover_cash_6m_pp_size'])} "
                f"Y2 Δ={_pp(q3['y2_neg_2of3_pp_size'])}; gate ≥5pp after size"
            ),
        }
    )
    out.append(
        {
            "object": "f_new_facility as 44-col Y3 X",
            "decision": "CLOSE",
            "why": (
                f"Y3 CV {_f(next(v['cv'] for v in p3['verdicts'] if v['feature']=='f_new_facility_gt0'))} "
                f"vs size {_f(p3['size_y3']['cv'])}; connection clock (debt QA Q3 CAUTION)"
            ),
        }
    )
    out.append(
        {
            "object": "has_* as health Y",
            "decision": "PARK",
            "why": "created_at connection inventory, not a health label (Y10 already PARK util)",
        }
    )
    out.append(
        {
            "object": "score flags vs Y9",
            "decision": "CLOSE",
            "why": "Y9 forbids F + a_fin_cost; not scored",
        }
    )
    out.append(
        {
            "object": "has_factoring as Y5 AP why",
            "decision": "CLOSE as X; descriptive only",
            "why": "size-weighted AP +18.9pp is 4 leftover companies, 3 in GROUP_0139 — group leftover, not a panel why",
        }
    )
    out.append(
        {
            "object": "confirming as AP-side footnote (not X)",
            "decision": "KEEP footnote" if (p47 or {}).get("only_is_ap") else "CLOSE",
            "why": (
                "AP lift > AR lift on all confirming (raw +4.4 vs −2.8pp) is stacked WC — "
                f"confirming-only AP {_pp((p47 or {}).get('only_ap', float('nan')))} vs rest "
                f"{_pp((p47 or {}).get('rest_ap', float('nan')))}; stacked "
                f"{_pp((p47 or {}).get('stack_ap', float('nan')))}. Not a 44-col X."
            ),
        }
    )
    out.append(
        {
            "object": "extract liquidity / outstanding as X or Y",
            "decision": "PARK",
            "why": "2026-08 book; outstanding p50=0 on WC; Y10 already PARK util",
        }
    )
    return out


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5, p6, p7, p8, p9 = (
        ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"],
        ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"],
    )
    dec = ctx["decisions"]
    lines = []
    a = lines.append
    a("# Factoring / confirming / LOC inventory (`f_has_*`)")
    a("")
    a(
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. "
        "No parquet rewrite. No new GBM. No `build_targets`. Do not invent a merged Y. "
        "Do not redo schedule / util. Do **not** score these F flags vs Y9."
    )
    a("")
    a("`f_has_factoring` / `f_has_confirming` / `f_has_loc` are as-of `created_at` inventory flags (rise-only like G). NORTH_STAR Q3: factoring/confirming are working-capital tools, not utilisation (Y10 PARK). Debt schedule QA: inventory is a connection panel, not origination.")
    a("")
    a("## Headline")
    a("")
    fact = next(r for r in p1["rows"] if r["type"] == "factoring")
    conf = next(r for r in p1["rows"] if r["type"] == "confirming")
    loc = next(r for r in p1["rows"] if r["type"] == "lineofcredit")
    v_fact = next(v for v in p3["verdicts"] if v["feature"] == "f_has_factoring")
    v_conf = next(v for v in p3["verdicts"] if v["feature"] == "f_has_confirming")
    v_loc = next(v for v in p3["verdicts"] if v["feature"] == "f_has_loc")
    a(
        f"Ever-n train: factoring **{fact['ever_n']}** / confirming **{conf['ever_n']}** / "
        f"LOC **{loc['ever_n']}** of {p1['n_co']} companies. "
        f"CM share {_pp(fact['cm_share'])} / {_pp(conf['cm_share'])} / {_pp(loc['cm_share'])} "
        f"(modal0 {_pp(fact['modal0'])} / {_pp(conf['modal0'])} / {_pp(loc['modal0'])}; "
        f"feature-report 99.3% / 96.7% / 88.3%). "
        f"Rise-only: fact={fact['rise_only']} conf={conf['rise_only']} loc={loc['rise_only']} "
        f"(drops {fact['n_drop']}/{conf['n_drop']}/{loc['n_drop']}). "
        f"Y3 singles {_f(v_fact['cv'])} / {_f(v_conf['cv'])} / {_f(v_loc['cv'])} vs size "
        f"{_f(p3['size_y3']['cv'])} vs days {_f(p3['days_y3']['cv'])} (night 0.711). "
        f"Dark {p6['n_dark']} vs invoiced {p6['n_erp']}: "
        + ", ".join(
            f"{r['type']} {r['dark_ever']}/{r['erp_ever']}" for r in p6["rows"]
        )
        + ". "
        f"44-col: **{'KEEP some' if p3['any_keep_44'] else 'drop all three flags'}**. "
        f"Factoring companies are a never-recoverer set (0/{fact['ever_n']} ever Y3+; "
        "T3 peers 7.7%). Still CLOSE as X — 18-company dummy, AUROC 0.505."
    )
    a("")
    a("## Brief questions")
    a("")
    a("| # | question | what this cut says |")
    a("| --- | --- | --- |")
    a("| 1 | Who is healthy? | **PARK** as a health Y. Connection inventory, not a FICO label. |")
    a("| 2 | Who is improving? | Not these rise-only flags. |")
    a(
        f"| 3 | Who is turning? | new-WC month Q3 {'**KEEP footnote**' if p4['any_q3_keep'] else '**CLOSE**'} "
        f"(size-controlled ≥5pp gate). |"
    )
    a("| 4 | Dip vs fall? | Overlap with Y4 / Y5 leftover is descriptive. |")
    a(
        f"| 5 | Why did it change? | Flags lose to size as Y3 X (Δ "
        f"{_f(v_fact['beat_size'])} / {_f(v_conf['beat_size'])} / {_f(v_loc['beat_size'])}). "
        "Confirming AP-side is stacked WC, not the product. |"
    )
    a("| 6 | Months earlier? | acf of a rise-only flag is persistence of *connection*, not lead. |")
    a("")
    a("## PARK / CLOSE / KEEP")
    a("")
    a("| object | decision | why |")
    a("| --- | --- | --- |")
    for d in dec:
        a(f"| {d['object']} | **{d['decision']}** | {d['why']} |")
    a("")
    a("KEEP-as-44-col-X rule: oriented group-fold CV beats oriented size by ≥0.02 **and** not SIZE (|ρ| vs log1p(a_in3) ≥ 0.5) **and** not a group dummy (largest group ≥25% of ever-companies). Q3 footnote only if a new-factoring/confirming month moves Y3 or Y2 by ≥5pp after size terciles.")
    a("")
    a("## 1. Prevalence (train; holdout count only)")
    a("")
    a(
        f"Train {p1['n_co']:,} companies / {p1['n_cm']:,} CM. "
        f"Holdout {p1['n_hold_co']} companies / {p1['n_hold_cm']:,} CM. "
        f"`f_n_facilities` rises {p1['fac_rise']['n_rise']:,} / drops {p1['fac_rise']['n_drop']:,} "
        f"(rise-only={p1['fac_rise']['rise_only']}; debt QA 555/0). "
        f"Ever WC (fact∪conf) **{p1['ever_wc']}**. "
        f"Feature-report modal confirm fact={p1['confirm_modal_fact']} conf={p1['confirm_modal_conf']}."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "raw prod": r["raw_prod_train"],
                    "raw co": r["raw_co_train"],
                    "after snap": r["after_snap"],
                    "ever n": r["ever_n"],
                    "ever % co": _pp(r["ever_share_co"]),
                    "CM on": r["cm_on"],
                    "CM share": _pp(r["cm_share"]),
                    "modal 0": _pp(r["modal0"]),
                    "last-mo on": r["last_on"],
                    "rises": r["n_rise"],
                    "drops": r["n_drop"],
                    "rise-only": "YES" if r["rise_only"] else "NO",
                    "acf1": _f(r["acf1"]),
                    "hold ever": r["hold_ever"],
                }
                for r in p1["rows"]
            ]
        )
    )
    a("")
    a("## 2. Size ρ vs log1p(a_in3)")
    a("")
    a("SIZE if |ρ| ≥ 0.5. Company-month flag vs log1p(a_in3); ever-flag vs last-month size.")
    a("")
    a(
        _md_table(
            [
                {
                    "flag": r["flag"],
                    "ρ log1p(a_in3)": _f(r["rho_login3"]),
                    "ρ a_in3": _f(r["rho_ain3"]),
                    "ρ ever vs size": _f(r["rho_ever_medsize"]),
                    "SIZE?": "YES" if r["is_size"] else "",
                }
                for r in p2["rows"]
            ]
        )
    )
    a("")
    a("## 3. Single-feature train group-fold AUROC (Y3 / Y2 only)")
    a("")
    a(
        f"Y3 stressed n={p3['n_y3']:,} base {_pp(p3['y3_rate'])}; "
        f"Y2 n={p3['n_y2']:,} base {_pp(p3['y2_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p3['days_y3']['cv'])}"
        f"{'; MATCH' if p3['days_y3_ok'] else '; drift vs 0.711'}). "
        f"Y2 days replica {_f(p3['days_y2']['cv'])} (night 0.540; this is oriented group-fold, not a days mismatch). "
        f"Leak Y3 vs B: {'ok' if p3['leak_y3']['ok'] else p3['leak_y3']['issues']}. "
        "Not scored vs Y9."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "y": r["y"].replace("y3_recover_cash_6m", "Y3").replace("y2_neg_2of3", "Y2"),
                    "feature": r["feature"],
                    "CV": _f(r["cv"]),
                    "sd": _f(r["sd"]),
                    "sign": r["sign"],
                    "train": _f(r["train_auc"]),
                    "n_lab": f"{r['n_lab']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "SIZE": "YES" if r["is_size"] else "",
                    "low_n": "YES" if r["low_power"] else "",
                }
                for r in p3["rows"]
            ]
        )
    )
    a("")
    a(
        "`f_has_loc` 0.546 is leftover inverse-size (ρ 0.255, Y3 size 0.617) — "
        "Δ vs size is −0.070. Factoring 0.505 is a never-recoverer hole that Mann–Whitney "
        "cannot see because the flag is rare. Neither is a 44-col X."
    )
    a("")
    a("Y3 KEEP screen vs size / days:")
    a("")
    a(
        _md_table(
            [
                {
                    "feature": v["feature"],
                    "CV": _f(v["cv"]),
                    "size": _f(v["size_cv"]),
                    "days": _f(v["days_cv"]),
                    "Δ size": _f(v["beat_size"]),
                    "SIZE": "YES" if v["is_size"] else "",
                    "KEEP 44": "YES" if v["keep_44"] else "",
                }
                for v in p3["verdicts"]
            ]
        )
    )
    a("")
    a("## 4. Q3 — new factoring / confirming this month")
    a("")
    a(
        f"`created_at` inside the month, typed. `f_new_facility` is any debt type (debt QA 573 CM / Q3 CAUTION). "
        f"New WC (fact∪conf) {p4['n_new_wc']:,} CM; first-ever WC birth share {_pp(p4['share_new_is_birth'])}. "
        "KEEP footnote only if size-tercile-weighted |Δ| ≥ 5pp on Y3 or Y2."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "flag": r["flag"],
                    "n_cm": r["n_cm"],
                    "n_co": r["n_co"],
                    "share": _pp(r["share"]),
                    "Y3 on": _pp(r["y3_recover_cash_6m_on"]),
                    "Y3 off": _pp(r["y3_recover_cash_6m_off"]),
                    "Y3 pp": _pp(r["y3_recover_cash_6m_pp"]),
                    "Y3 pp|size": _pp(r["y3_recover_cash_6m_pp_size"]),
                    "Y2 on": _pp(r["y2_neg_2of3_on"]),
                    "Y2 off": _pp(r["y2_neg_2of3_off"]),
                    "Y2 pp|size": _pp(r["y2_neg_2of3_pp_size"]),
                    "Q3 KEEP": "YES" if r["q3_keep"] else "",
                }
                for r in p4["rows"]
            ]
        )
    )
    a("")
    a("Within size terciles (new-WC and typed new):")
    a("")
    a(
        _md_table(
            [
                {
                    "flag": r["flag"],
                    "T": r["tercile"],
                    "y": r["y"].replace("y3_recover_cash_6m", "Y3").replace("y2_neg_2of3", "Y2"),
                    "n on": r["n_on"],
                    "rate on": _pp(r["rate_on"]),
                    "rate off": _pp(r["rate_off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p4["tercile_rows"]
                if r["flag"] in {"new_wc", "new_factoring", "new_confirming"}
            ]
        )
    )
    a("")
    a("## 5. Overlap with Y4 / Y5 leftover / ever-FX 228")
    a("")
    a(
        f"Ever-FX companies **{p5['n_fx']}** (confirm 228={p5['confirm_228']}). "
        f"Y4-positive companies {p5['n_y4_co']}. "
        f"Y5 AP leftover (own-p20 io × own-p80 supp HHI neither cell) "
        f"{p5['n_y5_left_cm']} CM / {p5['n_y5_left_co']} companies "
        f"(y5_why quoted 222 leftover AP positives — company-months). "
        "Descriptive only. Confirming vs Y5 is not an X score."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "set": r["set"],
                    "n_co": r["n_co"],
                    "∩ Y4": f"{r['y4']} ({_pp(r['y4_share'])})",
                    "∩ Y5 AP": f"{r['y5_ap']} ({_pp(r['y5_ap_share'])})",
                    "∩ Y5 leftover": f"{r['y5_left']} ({_pp(r['y5_left_share'])})",
                    "∩ FX 228": f"{r['fx228']} ({_pp(r['fx_share'])})",
                }
                for r in p5["rows"]
            ]
        )
    )
    a("")
    a(
        _md_table(
            [
                {
                    "flag": r["flag"],
                    "leftover CM ∧ flag": r["leftover_cm_on"],
                    "share of leftover CM": _pp(r["share_of_leftover"]),
                    "Y4 CM ∧ flag": r["y4_cm_on"],
                }
                for r in p5["month_rows"]
            ]
        )
    )
    a("")
    a("## 6. Dark 470 vs invoiced 744")
    a("")
    a(
        f"confirm 470/744 = {p6['confirm_470_744']}. "
        f"360 all-dark / 110 mixed-group dark. Access ≠ ERP (banking G): these products can exist on dark companies."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "dark ever": f"{r['dark_ever']} ({_pp(r['dark_share'])})",
                    "invoiced ever": f"{r['erp_ever']} ({_pp(r['erp_share'])})",
                    "dark CM": r["dark_cm"],
                    "invoiced CM": r["erp_cm"],
                }
                for r in p6["rows"]
            ]
        )
    )
    a("")
    a(
        _md_table(
            [
                {
                    "mix": r["mix"],
                    "n_co": r["n_co"],
                    "factoring": r["factoring"],
                    "confirming": r["confirming"],
                    "LOC": r["lineofcredit"],
                }
                for r in p6["mix_rows"]
            ]
        )
    )
    a("")
    a("## 7. bank_name of WC products (train, not post-snapshot)")
    a("")
    a("Dictionary: `bank_name` is the connected bank. `Other (customer-defined)` / In-house are not a bank feed.")
    a("")
    for r in p7["rows"]:
        a(
            f"**{r['type']}**: {r['n_prod']} products / {r['n_co']} companies / {r['n_banks']} bank names. "
            f"Empresas-labelled {r['empresas']}; in-house or Other {r['inhouse_or_other']}."
        )
        a("")
        a(
            _md_table(
                [
                    {"bank": t["bank"], "n": t["n"], "n_co": t["n_co"]}
                    for t in r["top"]
                ]
            )
        )
        a("")
    a("## 8. Confirming as AP-side (Y5 rates; never E as X)")
    a("")
    a(
        f"Leak screen confirming vs Y5+E: {'ok (we still do not *score* it as X)' if p8['leak_ok'] else p8['leak']['issues']}. "
        f"Confirming AP-vs-AR lift points AP-side: {p8['ap_side']} "
        f"(AP pp {_pp(p8['conf_ap_pp'])}; factoring AP pp {_pp(p8['fact_ap_pp'])}). "
        "`leftover on` is leftover share among AP-positives with a defined cash×HHI cell "
        "(not a panel rate). Family E columns are not used as X. Size-controlled AP is §11."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "Y5 AP on": _pp(r["y5_ap_od30_ownp80_on"]),
                    "Y5 AP off": _pp(r["y5_ap_od30_ownp80_off"]),
                    "AP pp": _pp(r["y5_ap_od30_ownp80_pp"]),
                    "Y5 AR on": _pp(r["y5_ar_od30_sust_on"]),
                    "Y5 AR off": _pp(r["y5_ar_od30_sust_off"]),
                    "AR pp": _pp(r["y5_ar_od30_sust_pp"]),
                    "leftover on": _pp(r["left_on"]),
                    "leftover pp": _pp(r["left_pp"]),
                }
                for r in p8["rows"]
            ]
        )
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ever co": r["ever"],
                    "ever ∩ leftover": r["ever_and_left"],
                    "share": _pp(r["share"]),
                    "rest share": _pp(r["rest_share"]),
                }
                for r in p8["ever_rows"]
            ]
        )
    )
    a("")
    a("## 9. Group dummy")
    a("")
    a("A rare flag that lives in one or two groups is a group dummy, not a health X.")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ever co": r["n_co"],
                    "n groups": r["n_groups"],
                    "max group share": _pp(r["max_group_share"]),
                    "max group": r["max_group"],
                    "dummy": "YES" if r["group_dummy"] else "",
                }
                for r in p9["rows"]
            ]
        )
    )
    a("")
    a("## Plot")
    a("")
    a(f"`{OUT_PNG.name}`" if ctx.get("png_ok") else "Plot skipped (no matplotlib).")
    a("")
    p10, p11, p12, p13, p14 = ctx["p10"], ctx["p11"], ctx["p12"], ctx["p13"], ctx["p14"]
    ov = p10["overlap"]
    a("## 10. Co-occurrence + size-tercile has_* rates")
    a("")
    a(
        f"Last-month train: fact∩conf {ov['fact_conf']}, fact∩loc {ov['fact_loc']}, "
        f"conf∩loc {ov['conf_loc']}, all three {ov['all3']}. "
        f"Only-fact {ov['fact_only']}, only-conf {ov['conf_only']}, only-loc {ov['loc_only']}. "
        "Most factoring sits with another facility — not a standalone type."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "flag": r["flag"].replace("f_has_", ""),
                    "y": r["y"].replace("y3_recover_cash_6m", "Y3").replace("y2_neg_2of3", "Y2"),
                    "T": r["tercile"],
                    "n on": r["n_on"],
                    "rate on": _pp(r["rate_on"]),
                    "rate off": _pp(r["rate_off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p10["rows"]
                if r["y"] == Y3
            ]
        )
    )
    a("")
    a("Y2 same terciles (has_* months):")
    a("")
    a(
        _md_table(
            [
                {
                    "flag": r["flag"].replace("f_has_", ""),
                    "T": r["tercile"],
                    "n on": r["n_on"],
                    "rate on": _pp(r["rate_on"]),
                    "rate off": _pp(r["rate_off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p10["rows"]
                if r["y"] == Y2
            ]
        )
    )
    a("")
    a("## 11. Y5 AP after size (never E as X)")
    a("")
    a(
        f"Raw factoring AP +19.6pp looked like leftover. Size-weighted AP Δ: "
        f"factoring {_pp(p11['fact_pp_size'])} (survives 5pp={p11['fact_survives']}); "
        f"confirming {_pp(p11['conf_pp_size'])} (survives={p11['conf_survives']}). "
        "Still not an X — Y5 forbids E; this is the AP-side footnote only."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "T": r["tercile"],
                    "n AP-lab on": r["n_on"],
                    "AP on": _pp(r["ap_on"]),
                    "AP off": _pp(r["ap_off"]),
                    "pp": _pp(r["pp"]),
                    "leftover among AP+": _pp(r["left_on"]),
                }
                for r in p11["rows"]
            ]
        )
    )
    a("")
    a("## 12. Dark WC — factoring without an invoice book")
    a("")
    a(
        f"{p12['note']} Dark no-WC n={p12['dark_no_wc_n']}, median a_in3 "
        f"{_f(p12['dark_no_wc_med_in3'], 0)}. Dark factoring is not smaller cash than invoiced factoring."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "dark n": r["dark_n"],
                    "invoiced n": r["erp_n"],
                    "dark med in3": _f(r["dark_med_in3"], 0),
                    "inv med in3": _f(r["erp_med_in3"], 0),
                    "dark med days": _f(r["dark_med_days"], 1),
                    "inv med days": _f(r["erp_med_days"], 1),
                }
                for r in p12["rows"]
            ]
        )
    )
    a("")
    a("## 13. Confirming Empresas vs retail bank_name")
    a("")
    a(
        f"Companies with both Empresas and retail confirming labels: {p13['n_both']}. "
        "Y5 AP on connected months (descriptive)."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n_co": r["n_co"],
                    "CM on": r["n_cm_on"],
                    "Y5 AP": _pp(r["y5_ap"]),
                    "n AP-lab": r["y5_n"],
                    "Y3": _pp(r["y3"]),
                    "n Y3": r["y3_n"],
                }
                for r in p13["rows"]
            ]
        )
    )
    a("")
    a("## 14. GROUP_0139 (22% of ever-factoring)")
    a("")
    a(
        f"Group has {p14['n_co']} train companies last-month; factoring {p14['n_fact']}, "
        f"confirming {p14['n_conf']}, LOC {p14['n_loc']}. Median a_in3 {_f(p14['med_in3'], 0)}. "
        f"Factoring names: {', '.join(p14['ids']) or '—'}. Under the 25% dummy line — still not a KEEP."
    )
    a("")
    p16, p17, p18, p19, p20 = ctx["p16"], ctx["p17"], ctx["p18"], ctx["p19"], ctx["p20"]
    a("## 16. Zero Y3 recoveries on factoring months")
    a("")
    a(
        f"Y3 positives {p16['n_pos']:,} / labeled {p16['n_lab']:,}. "
        "A rare always-off flag among recoverers still has AUROC ≈ 0.50 "
        "(Mann–Whitney lift is n_flag / 2 n_neg). Do not KEEP a 'never recovers' story."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "Y3 lab ∧ flag": r["y3_lab_on"],
                    "Y3 pos ∧ flag": r["y3_pos_on"],
                    "share of pos": _pp(r["share_of_pos"]),
                    "rate on": _pp(r["rate_on"]),
                }
                for r in p16["rows"]
            ]
        )
    )
    a("")
    a("## 17. Connection vs first cash-trail month")
    a("")
    a("Lag = created_at month − first panel month. Positive = connected after the cash trail started.")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n prod": r["n"],
                    "p50 lag mo": _f(r["p50"], 1),
                    "after trail": _pp(r["share_after_trail"]),
                    "before trail": _pp(r["share_before"]),
                }
                for r in p17["rows"]
            ]
        )
    )
    a("")
    a("## 18. Extract granted / outstanding (last-month book, not a path)")
    a("")
    a(p18["note"])
    a("Factoring / confirming outstanding p50 = 0: most WC rows are a connected limit, not a drawn book. Do not treat `f_has_*` as utilisation (Y10 already PARK).")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n": r["n"],
                    "granted p50": _f(r["granted_p50"], 0),
                    "out p50": _f(r["out_p50"], 0),
                    "|granted|>1": _pp(r["share_granted_gt1"]),
                    "|out|>1": _pp(r["share_out_gt1"]),
                }
                for r in p18["rows"]
            ]
        )
    )
    a("")
    a("## 19. 2025-10 factoring connection wave")
    a("")
    a(
        f"{p19['n']} train companies connected a factoring product in 2025-10 "
        f"({p19['n_dark']} dark). Median a_in3 {_f(p19['med_in3'], 0)}. "
        f"Y3 {_pp(p19['y3']['rate'])} n={p19['y3']['n_lab']}; "
        f"Y2 {_pp(p19['y2']['rate'])} n={p19['y2']['n_lab']}. "
        f"Names: {', '.join(p19['ids'])}. A platform connection wave, not Q3 turning."
    )
    a("")
    a("## 20. Who are the factoring AP leftovers")
    a("")
    a(
        f"Factoring ∧ Y5 AP: {p20['n_ap_on']} CM / {p20['n_ap_co']} companies. "
        f"Of those, leftover cell {p20['n_left_on']} CM / {p20['n_left_co']} companies "
        f"({', '.join(p20['ids'])}). Thin — do not invent a Y5-factoring label."
    )
    a("")
    p21, p22, p23 = ctx["p21"], ctx["p22"], ctx["p23"]
    a("## 22. Factoring AP leftover is a group, not a law")
    a("")
    a(
        f"{p21['n_in_g']}/{p21['n_left']} leftover factoring companies sit in GROUP_0139 "
        f"(share {_pp(p21['share'])}). Outside: {', '.join(p21['outside']) or '—'}. "
        f"{'Group leftover — CLOSE as a Y5 why.' if p21['group_leftover'] else 'Not concentrated.'}"
    )
    a("")
    a("## 23. Siblings sharing WC")
    a("")
    a("Groups with ≥2 members on the product. A company tag, not a group book (same shape as schedule QA).")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ever last-mo": r["n_co"],
                    "n groups": r["n_groups"],
                    "groups with ≥2": r["n_multi"],
                    "share groups multi": _pp(r["share_multi"]),
                }
                for r in p22["rows"]
            ]
        )
    )
    a("")
    a("## 24. Product currency")
    a("")
    for r in p23["rows"]:
        a(f"**{r['type']}** ({r['n_ccy']} currencies): " + ", ".join(f"{t['ccy']} {t['n']}" for t in r["top"]))
        a("")
    p24 = ctx["p24"]
    a("## 25. Store vs live last-month inventory")
    a("")
    a(
        f"As-of {p24['last']}. "
        f"{'Mismatch — inspect debt.py / store staleness. Do not rewrite parquet tonight.' if p24['mismatch'] else 'Store flags match live created_at inventory. No parquet rewrite.'}"
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "store": r["store"],
                    "live": r["live"],
                    "agree": r["agree"],
                    "only store": r["only_store"],
                    "only live": r["only_live"],
                }
                for r in p24["rows"]
            ]
        )
    )
    a("")
    p25, p26, p27 = ctx["p25"], ctx["p26"], ctx["p27"]
    a("## 26. ICC and connected-only Y3")
    a("")
    a(
        f"Rise-only flags are BETWEEN (high ICC): connection is a company trait. "
        f"On months with `f_n_facilities>0` ({p25['n_connected']:,} CM), "
        f"`f_has_loc` CV {_f(next(r['cv'] for r in p25['auc_rows'] if r['feature']=='f_has_loc'))} "
        f"vs size {_f(p25['size_cv'])} (Δ {_f(p25['loc_beat'])}). Still CLOSE."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ICC": _f(r["icc"]),
                    "acf1": _f(r["acf1"]),
                    "n cos": r["k"],
                }
                for r in p25["icc_rows"]
            ]
        )
    )
    a("")
    a(
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "CV connected": _f(r["cv"]),
                    "n_lab": f"{r['n_lab']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                }
                for r in p25["auc_rows"]
            ]
        )
    )
    a("")
    a("## 27. Y4 rates after size (never F as X)")
    a("")
    a(
        f"Y4 forbids family F — leak screen ok={p26['leak_ok']} (expected fail). "
        "Rates only. Factoring's raw 33% ever-Y4 overlap is the company set, not a month why."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "T": r["tercile"],
                    "n Y4-lab on": r["n_on"],
                    "Y4 on": _pp(r["on"]),
                    "Y4 off": _pp(r["off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p26["rows"]
            ]
        )
    )
    a("")
    a("## 28. Confirming AP without GROUP_0139")
    a("")
    a(
        f"Drop GROUP_0139 CM ({p27['n_dropped']:,}). Confirming AP {_pp(p27['conf_ap_on'])} vs "
        f"{_pp(p27['conf_ap_off'])} (Δ {_pp(p27['conf_pp'])}, n={p27['conf_n']}); "
        f"AR Δ {_pp(p27['conf_ar_pp'])}. Factoring AP leftover collapses: "
        f"{_pp(p27['fact_ap_on'])} vs {_pp(p27['fact_ap_off'])} (Δ {_pp(p27['fact_pp'])}, n={p27['fact_n']}). "
        "Confirming stays mildly AP-side without the group; factoring's +19pp was the group."
    )
    a("")
    p28, p29, p30 = ctx["p28"], ctx["p29"], ctx["p30"]
    a("## 29. Same companies before vs after first connection")
    a("")
    a("If Q3 turning were real, Y3/Y2 would move after the first typed `created_at`. Same companies, split on first on-flag month.")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n_co": r["n_co"],
                    "Y3 before": f"{_pp(r['y3_recover_cash_6m_before'])} (n={r['y3_recover_cash_6m_n_b']})",
                    "Y3 after": f"{_pp(r['y3_recover_cash_6m_after'])} (n={r['y3_recover_cash_6m_n_a']})",
                    "Y3 pp": _pp(r["y3_recover_cash_6m_pp"]),
                    "Y2 before": f"{_pp(r['y2_neg_2of3_before'])} (n={r['y2_neg_2of3_n_b']})",
                    "Y2 after": f"{_pp(r['y2_neg_2of3_after'])} (n={r['y2_neg_2of3_n_a']})",
                    "Y2 pp": _pp(r["y2_neg_2of3_pp"]),
                }
                for r in p28["rows"]
            ]
        )
    )
    a("")
    a("## 30. Holdout coverage only")
    a("")
    a(
        f"Holdout {p29['n_co']} companies / {p29['n_cm']:,} CM. Not used in any rate, ρ, or AUROC."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ever": r["ever"],
                    "CM on": r["cm"],
                }
                for r in p29["rows"]
            ]
        )
    )
    a("")
    a("## 31. Ever-FX 228 ∩ WC")
    a("")
    a(
        f"Ever-FX {p30['n_fx']}. Factoring almost never overlaps FX "
        f"({p30['rows'][0]['n_both']}/{p30['rows'][0]['n_wc']}). Not an exporter-factoring story."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ever WC": r["n_wc"],
                    "∩ FX": r["n_both"],
                    "share": _pp(r["share"]),
                    "both med log in3": _f(r["both_med"]),
                    "WC-only med": _f(r["wc_med"]),
                    "FX med": _f(r["fx_med"]),
                }
                for r in p30["rows"]
            ]
        )
    )
    a("")
    p31, p32 = ctx["p31"], ctx["p32"]
    a("## 32. Factoring Y2 pre/post after size and year")
    a("")
    a(
        "Raw Y2 20.3%→10.7% after first factoring connection looks like relief. "
        "Within size terciles and calendar years it is not a stable ≥5pp Q3 footnote "
        "(new-month gate already failed). Mean-reversion after a high-Y2 spell, not turning."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n before": r["n_b"],
                    "n after": r["n_a"],
                    "Y2 before": _pp(r["before"]),
                    "Y2 after": _pp(r["after"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p31["terc"] + p31["year"]
            ]
        )
    )
    a("")
    a("Train Y2 base by year (all companies): " + ", ".join(f"{r['year']} {_pp(r['rate'])}" for r in p31["base"]) + ".")
    a("")
    a("## 33. Never-recoverer company set")
    a("")
    a(
        f"Train companies with any Y3=1: {p32['n_ever_y3_co']}. "
        f"Non-factoring ever-recover share {_pp(p32['rest_share'])}. "
        "Factoring 0/18 is a rare company hole, not a month signal — AUROC stays 0.505."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "ever": r["n"],
                    "ever Y3+": r["n_ever_y3"],
                    "share": _pp(r["share"]),
                }
                for r in p32["rows"]
            ]
        )
    )
    a("")
    p33, p34 = ctx["p33"], ctx["p34"]
    a("## 34. Months on book")
    a("")
    a("WC companies are not short-trail names. Connection is late on a full book (pass 17), not a thin onboarding stub.")
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n_co": r["n"],
                    "p50 months": _f(r["p50_mo"], 1),
                    "p50 med in3": _f(r["p50_in3"], 0),
                    "p50 med days": _f(r["p50_days"], 1),
                }
                for r in p33["rows"]
            ]
        )
    )
    a("")
    a("## 35. COMP_0919 — leftover factoring outside GROUP_0139")
    a("")
    if p34.get("n"):
        a(
            f"Group {p34['group']} mix={p34['mix']}; invoiced={p34['has_book']}. "
            f"{p34['n']} CM, factoring-on {p34['fact_cm']}, confirming-on {p34['conf_cm']}. "
            f"Y5 AP {p34['y5_ap']} leftover {p34['y5_left']}; Y3+ {p34['y3_pos']}; Y2+ {p34['y2_pos']}. "
            f"Last a_in3 {_f(p34['last_in3'], 0)}. One name does not make a Y5 law."
        )
    else:
        a("COMP_0919 not on the train panel.")
    a("")
    p35, p36 = ctx["p35"], ctx["p36"]
    a("## 36. How many WC products per company")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n_co": r["n_co"],
                    "p50 n prod": _f(r["p50"], 1),
                    "max": r["max"],
                    "share ≥2": _pp(r["share_multi"]),
                }
                for r in p35["rows"]
            ]
        )
    )
    a("")
    a("## 37. Product `created_at` vs `companies.created_at`")
    a("")
    a("Read-only `companies`. Do not edit `companies_qa.py`. Positive lag = product connected after the company row.")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n": r["n"],
                    "p50 days after company": _f(r["p50_d"], 0),
                    "share after company": _pp(r["share_after_co"]),
                }
                for r in p36["rows"]
            ]
        )
    )
    a("")
    p37, p38 = ctx["p37"], ctx["p38"]
    a("## 38. Y3 label coverage — unstressed vs never-recover")
    a("")
    a("If factoring companies simply lacked a stressed window, Y3 would be unlabeled. They are labeled as often as the rest and still have 0 recoveries.")
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "CM": f"{r['n_cm']:,}",
                    "Y3 labeled": f"{r['n_lab']:,}",
                    "label share": _pp(r["lab_share"]),
                    "Y3+": r["n_pos"],
                    "rate | labeled": _pp(r["rate_among_lab"]),
                }
                for r in p37["rows"]
            ]
        )
    )
    a("")
    a("## 39. Product labels")
    a("")
    for r in p38["rows"]:
        a(f"**{r['type']}** ({r['n_lab']} distinct labels): " + ", ".join(f"{t['label']} ×{t['n']}" for t in r["top"]))
        a("")
    p39 = ctx["p39"]
    a("## 40. Size-matched ever-Y3+ (is 0/18 just large firms?)")
    a("")
    a("Last-month size tercile. Factoring companies vs other companies in the same tercile. If T3 rest still recovers, 0/18 is not leftover size.")
    a("")
    a(
        _md_table(
            [
                {
                    "T": r["tercile"],
                    "n factoring": r["n_fact"],
                    "fact ever Y3+": _pp(r["fact_y3"]),
                    "n rest": r["n_rest"],
                    "rest ever Y3+": _pp(r["rest_y3"]),
                }
                for r in p39["rows"]
            ]
        )
    )
    a("")
    p40 = ctx["p40"]
    a("## 41. In-house LOC vs bank LOC")
    a("")
    a("31 in-house/Other LOC products (pass 7). Descriptive Y3/Y2 on connected months.")
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n_co": r["n_co"],
                    "Y3": _pp(r["y3"]["rate"]),
                    "n Y3": r["y3"]["n_lab"],
                    "Y2": _pp(r["y2"]["rate"]),
                    "n Y2": r["y2"]["n_lab"],
                }
                for r in p40["rows"]
            ]
        )
    )
    a("")
    p41 = ctx["p41"]
    a("## 42. Extract liquidity (not utilisation)")
    a("")
    a(p41["note"])
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n": r["n"],
                    "defined": _pp(r["share_def"]),
                    "p50": _f(r["p50"], 0),
                    ">1": _pp(r["share_gt1"]),
                    "exactly 0": _pp(r["share_zero"]),
                }
                for r in p41["rows"]
            ]
        )
    )
    a("")
    p42 = ctx["p42"]
    a("## 43. `service` codes")
    a("")
    a("Bank service, not a health X. `custom` would be customer-defined (G).")
    a("")
    for r in p42["rows"]:
        a(
            f"**{r['type']}**: {r['n_svc']} services, custom={r['custom']}. "
            + ", ".join(f"{t['svc']} ×{t['n']}" for t in r["top"])
        )
        a("")
    p43 = ctx["p43"]
    a("## 44. Observed `f_ds_r` / `f_fc_r` on WC-flag months")
    a("")
    a("Flow columns stay KEEP. If flag months have no observed service/cost, the inventory is a connected limit, not a used book.")
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "col": r["col"],
                    "n on": r["n_on"],
                    "p50 on": _f(r["p50_on"]),
                    "share>0 on": _pp(r["share_gt0_on"]),
                    "p50 off": _f(r["p50_off"]),
                    "share>0 off": _pp(r["share_gt0_off"]),
                }
                for r in p43["rows"]
            ]
        )
    )
    a("")
    p44 = ctx["p44"]
    a("## 45. Factoring-only vs stacked facilities")
    a("")
    a(
        f"Last-month factoring-only companies: {p44['n_only']} ({', '.join(p44['ids_only']) or '—'}). "
        f"ds>0 on their factoring-on months {_pp(p44['ds_gt0_only'])} (n={p44['n_on_only']}). "
        f"Stacked fact+conf/loc {p44['n_both']}: ds>0 {_pp(p44['ds_gt0_both'])} (n={p44['n_on_both']}). "
        "If only-fact still has ds, the flow is not 'other F flags'."
    )
    a("All three only-fact names also have `loan` rows — observed `f_ds_r` is the loan book, not the factoring flag. Flow KEEP is unchanged.")
    a("")
    p45 = ctx["p45"]
    a("Other `debt_products` types on those names (loan/leasing/…):")
    a("")
    a(
        _md_table(
            [{"company": r["company_id"], "type": r["type"], "n": r["n"]} for r in p45["rows"]]
        )
        if p45["rows"]
        else "_(none)_\n"
    )
    a("")
    p46 = ctx["p46"]
    a("## 46. Confirming-only (no factoring, no LOC) AP-side")
    a("")
    a(
        f"Confirming-only last-month companies: {p46['n_co']}. "
        f"On-months AP {_pp(p46['ap']['rate'])} (n={p46['ap']['n_lab']}) vs off {_pp(p46['ap_off']['rate'])}; "
        f"AR {_pp(p46['ar']['rate'])} vs {_pp(p46['ar_off']['rate'])}. "
        f"Y3 on {_pp(p46['y3']['rate'])} n={p46['y3']['n_lab']}. "
        "If AP lift survives without stacked LOC/factoring, the Q5 footnote is confirming itself. "
        "Within-company off months are pre-connection and small-n — do not read the off rate as a panel."
    )
    a("")
    p47 = ctx["p47"]
    a("## 47. Confirming-only vs stacked vs rest (Y5 AP)")
    a("")
    a(
        f"Confirming-only AP {_pp(p47['only_ap'])} vs stacked {_pp(p47['stack_ap'])} vs rest "
        f"{_pp(p47['rest_ap'])}. only_is_ap={p47['only_is_ap']} (need ≥2pp over rest). "
        "The all-confirming +4.4pp AP story is stacked WC users, not confirming as a tool."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n_co": r["n_co"],
                    "n_cm": r["n_cm"],
                    "Y5 AP": _pp(r["ap"]),
                    "AP n": r["ap_n"],
                    "Y5 AR": _pp(r["ar"]),
                }
                for r in p47["rows"]
            ]
        )
    )
    a("")
    p48 = ctx["p48"]
    a("## 48. Stacked confirming AP after size")
    a("")
    a(
        f"Size-weighted AP pp: confirming-only {_pp(p48['only_pp_size'])}, stacked "
        f"{_pp(p48['stack_pp_size'])}. If stacked also dies after size, Q5 is size/stacking, not the product."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "tercile": r["tercile"],
                    "n_on": r["n_on"],
                    "AP on": _pp(r["ap_on"]),
                    "AP off": _pp(r["ap_off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p48["rows"]
            ]
        )
    )
    a("")
    p49 = ctx["p49"]
    a("## 49. Unused limit share (PARK — not util)")
    a("")
    a(
        "Last-month extract `(granted−outstanding)/granted` given granted>1. "
        "High unused + low drawn = connected limit. PARK with Y10. Do not add a utilisation cousin."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "type": r["type"],
                    "n": r["n"],
                    "n granted>1": r["n_granted"],
                    "unused p50": _pp(r["unused_p50"]),
                    "unused≥90%": _pp(r["share_unused_ge90"]),
                    "drawn |out|>1": _pp(r["share_drawn"]),
                }
                for r in p49["rows"]
            ]
        )
    )
    a("")
    p50 = ctx["p50"]
    a("## 50. LOC-only (no factoring, no confirming)")
    a("")
    a(
        f"LOC-only last-month companies: {p50['n_co']}. "
        f"On-months Y3 {_pp(p50['y3']['rate'])} n={p50['y3']['n_lab']}; "
        f"Y2 {_pp(p50['y2']['rate'])}. Terciles are within LOC-only months (not the panel). "
        "Inverse-size leftover, not a WC-tool dummy."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "tercile": r["tercile"],
                    "n_on": r["n_on"],
                    "Y3": _pp(r["y3"]["rate"]),
                    "Y2": _pp(r["y2"]["rate"]),
                }
                for r in p50["terc"]
            ]
        )
    )
    a("")
    p51 = ctx["p51"]
    a("## 51. Confirming banks: only vs stacked")
    a("")
    a("Same Empresas names on both slices would mean the AP lift is stacking, not a different confirming product.")
    a("")
    for rec in p51["rows"]:
        a(
            f"**{rec['slice']}**: {rec['n_prod']} products / {rec['n_co']} companies / "
            f"{rec['n_banks']} banks. Empresas-labelled {rec['empresas']}."
        )
        a("")
        a(
            _md_table(
                [{"bank": r["bank"], "n": r["n"], "n_co": r["n_co"]} for r in rec["top"]]
            )
            if rec["top"]
            else "_(none)_\n"
        )
        a("")
    p52 = ctx["p52"]
    a("## 52. Drawn factoring vs connected-only")
    a("")
    a(
        "Drawn = last extract outstanding>1. Y3 is still 0 on both slices — draw vs connect does not make a health Y. PARK with util."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n_co": r["n_co"],
                    "Y3": _pp(r["y3"]["rate"]),
                    "Y2": _pp(r["y2"]["rate"]),
                    "Y5 AP": _pp(r["ap"]["rate"]),
                    "leftover": _pp(r["left"]["rate"]),
                }
                for r in p52["rows"]
            ]
        )
    )
    a("")
    p53 = ctx["p53"]
    a("## 53. Drawn factoring Y2 / AP after size")
    a("")
    a(
        f"Drawn companies {p53['n_drawn']}. Size-weighted Y2 pp {_pp(p53['y2_pp_size'])}. "
        "A leftover pocket on n=8 is not a health Y and not a 44-col X."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "y": r["y"],
                    "tercile": r["tercile"],
                    "n_on": r["n_on"],
                    "on": _pp(r["rate_on"]),
                    "off": _pp(r["rate_off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p53["rows"]
            ]
        )
    )
    a("")
    p54 = ctx["p54"]
    a("## 54. Drawn factoring without GROUP_0139")
    a("")
    a(
        f"Drawn ∩ GROUP_0139 = {p54['n_g']} of {p54['n_drawn']}. Remaining {p54['n_wo']}: "
        f"{', '.join(p54['ids_wo']) or '—'}. "
        f"Y2 {_pp(p54['y2']['rate'])} n={p54['y2']['n_lab']}; "
        f"AP {_pp(p54['ap']['rate'])}; Y3 {_pp(p54['y3']['rate'])}."
    )
    a("")
    p55 = ctx["p55"]
    a("## 55. Drawn confirming vs connected-only")
    a("")
    a("Same extract PARK as factoring. If drawn confirming is also a leftover pocket, still not a Y.")
    a("")
    a(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n_co": r["n_co"],
                    "Y3": _pp(r["y3"]["rate"]),
                    "Y2": _pp(r["y2"]["rate"]),
                    "Y5 AP": _pp(r["ap"]["rate"]),
                }
                for r in p55["rows"]
            ]
        )
    )
    a("")
    p56 = ctx["p56"]
    a("## 56. Drawn confirming Y2 after size")
    a("")
    a(
        f"Drawn confirming companies {p56['n_drawn']}. Size-weighted Y2 pp {_pp(p56['y2_pp_size'])}. "
        "Raw +9pp vs connected-only is not a 44-col X (extract book; PARK with util)."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "tercile": r["tercile"],
                    "n_on": r["n_on"],
                    "on": _pp(r["on"]),
                    "off": _pp(r["off"]),
                    "pp": _pp(r["pp"]),
                }
                for r in p56["rows"]
            ]
        )
    )
    a("")
    p57 = ctx["p57"]
    a("## 57. Holdout WC companies (coverage only)")
    a("")
    a(
        f"Last-month holdout companies with any of the three flags: {p57['n']}. "
        f"Factoring holdout: {', '.join(p57['fact_ids']) or '—'} ({p57['fact_flags'] or 'none'}). "
        f"Largest holdout WC group {p57['top_group']} n={p57['top_group_n']} "
        "(different from train GROUP_0139 — hidden test is new groups). "
        "Counts only. No AUROC. No percentile fit."
    )
    a("")
    a(
        _md_table(
            [
                {
                    "company": r["company_id"],
                    "group": r["group_id"],
                    "book": "ERP" if r["has_book"] else "dark",
                    "flags": r["flags"],
                    "a_in3": _f(r["in3"], 0),
                }
                for r in p57["rows"]
            ]
        )
        if p57["rows"]
        else "_(none)_\n"
    )
    a("")
    p58 = ctx["p58"]
    a(
        f"Holdout WC groups unseen on train: {p58['unseen']} "
        f"(n_groups={p58['n_hold_g']}, overlap={p58['n_overlap']}). "
        f"GROUP_0103 in train={p58['top_in_train']}. Whole-group holdout holds."
    )
    a("")
    a("## 21. Q3 birth vs add-on + calendar")
    a("")
    br, ad = p4["birth"], p4["addon"]
    a(
        f"New-WC months: birth {br['n']} (Y3 {_pp(br['y3']['rate'])} n={br['y3']['n_lab']}; "
        f"Y2 {_pp(br['y2']['rate'])}); add-on {ad['n']} "
        f"(Y3 {_pp(ad['y3']['rate'])} n={ad['y3']['n_lab']}; Y2 {_pp(ad['y2']['rate'])}). "
        f"Same connection-clock shape as `f_new_facility` / `g_new`. Q3 gate now requires "
        f"n_lab_on ≥ {p4['min_lab']} — the first-cut KEEP was new-factoring Y3 on 11 labeled months."
    )
    a("")
    if len(p4["calendar"]):
        a(
            _md_table(
                [
                    {
                        "period": pd.Timestamp(r.period).strftime("%Y-%m"),
                        "new WC": int(r.n_wc),
                        "fact": int(r.n_fact),
                        "conf": int(r.n_conf),
                    }
                    for r in p4["calendar"].itertuples(index=False)
                ]
            )
        )
    a("")
    a("## Return card")
    a("")
    a(
        f"- ever-n: factoring {fact['ever_n']}, confirming {conf['ever_n']}, LOC {loc['ever_n']}"
    )
    a(
        f"- Y3 singles vs 0.711 / size {_f(p3['size_y3']['cv'])}: "
        f"fact {_f(v_fact['cv'])} (Δ {_f(v_fact['beat_size'])}), "
        f"conf {_f(v_conf['cv'])} (Δ {_f(v_conf['beat_size'])}), "
        f"loc {_f(v_loc['cv'])} (Δ {_f(v_loc['beat_size'])})"
    )
    a(
        f"- rise-only: fact={fact['rise_only']} conf={conf['rise_only']} loc={loc['rise_only']}"
    )
    a(
        "- dark vs invoiced: "
        + ", ".join(f"{r['type']} {r['dark_ever']} vs {r['erp_ever']}" for r in p6["rows"])
    )
    a(
        f"- drop-from-44: **{'no — a flag KEEP' if p3['any_keep_44'] else 'YES — drop f_has_factoring, f_has_confirming, f_has_loc (and f_new_facility)'}**"
    )
    a("")
    a("## Failed / next")
    a("")
    a("- First-cut Q3 KEEP was new-factoring Y3 0% on 11 labeled months. Min-n=20 + WC-only gate **CLOSE**d it.")
    a("- Factoring AP +19pp after size is GROUP_0139 (3/4 leftover names). Do not invent a Y5-factoring label.")
    a("- 0/18 never-Y3 is real vs T3 peers 7.7% and is still CLOSE as X (18-company dummy; hidden test is new groups).")
    a("- Factoring-only last-month (3 names) all have `loan` rows — `f_ds_r` is the loan book, not a factoring-flow proxy. Flow KEEP stands.")
    a("- Confirming-only AP matches rest (~8–9%); the +4.4pp AP lift is stacked WC. Q5 tool-identity footnote **CLOSE**.")
    a("- Drawn factoring Y2 +21pp after size is GROUP_0139 (3/8). Remaining 5: Y2 0% n=13. PARK, not a pocket Y.")
    a("- Drawn confirming Y2 +8.9pp after size (n=36). AP flat vs connected. Still PARK extract, not 44 X.")
    a("- Holdout factoring is COMP_0269 stacked; largest holdout WC group is not GROUP_0139. Hidden test = new groups. CLOSE as X.")
    a("- Next (not this owner): drop the three flags + `f_new_facility` from `gbm_core` CORE/44 when someone re-cards. Not a parquet rewrite.")
    a("")
    a(
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: prevalence, size, singles, Q3 (min-n), "
        "overlap, dark, banks, AP-side, group dummy, co-occur, Y5|size, dark WC, Empresas, "
        "GROUP_0139, Y3-zero, connect-lag, granted, 2025-10 wave, AP-who, leftover-group, "
        "siblings, currency, store=live, ICC, Y4|size, AP minus GROUP_0139, pre/post, "
        "holdout, FX∩, Y2|size-year, never-recover, trail, COMP_0919, n-prod, company-lag, "
        "Y3-coverage, labels, size-matched never-recover, in-house LOC, liquidity, service, "
        "birth/calendar, other-debt, confirming-only/stacked AP, unused-limit, LOC-only, "
        "confirming banks, drawn factoring/confirming, holdout WC ids, unseen holdout groups."
    )
    a("")
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = _utc_ts()
    p1, p2, p3, p6 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p6"]
    fact = next(r for r in p1["rows"] if r["type"] == "factoring")
    conf = next(r for r in p1["rows"] if r["type"] == "confirming")
    loc = next(r for r in p1["rows"] if r["type"] == "lineofcredit")
    v_fact = next(v for v in p3["verdicts"] if v["feature"] == "f_has_factoring")
    v_conf = next(v for v in p3["verdicts"] if v["feature"] == "f_has_confirming")
    v_loc = next(v for v in p3["verdicts"] if v["feature"] == "f_has_loc")
    rho_f = next(r for r in p2["rows"] if r["type"] == "factoring")
    rows = [
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "factoring_qa",
            "split": "train", "metric": "ever_n_factoring",
            "value": _fmt(fact["ever_n"]),
            "coverage": _fmt(fact["cm_share"]),
            "notes": f"confirming={conf['ever_n']} loc={loc['ever_n']} rise_only={fact['rise_only']}/{conf['rise_only']}/{loc['rise_only']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y3, "model": "factoring_qa",
            "split": "train_cv", "metric": "auroc_f_has_factoring",
            "value": _fmt(v_fact["cv"]),
            "coverage": _fmt(fact["cm_share"]),
            "notes": f"size={v_fact['size_cv']:.4f} days={v_fact['days_cv']:.4f} beat={v_fact['beat_size']:.4f} keep44={v_fact['keep_44']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y3, "model": "factoring_qa",
            "split": "train_cv", "metric": "auroc_f_has_confirming",
            "value": _fmt(v_conf["cv"]),
            "coverage": _fmt(conf["cm_share"]),
            "notes": f"size={v_conf['size_cv']:.4f} beat={v_conf['beat_size']:.4f} keep44={v_conf['keep_44']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y3, "model": "factoring_qa",
            "split": "train_cv", "metric": "auroc_f_has_loc",
            "value": _fmt(v_loc["cv"]),
            "coverage": _fmt(loc["cm_share"]),
            "notes": f"size={v_loc['size_cv']:.4f} beat={v_loc['beat_size']:.4f} keep44={v_loc['keep_44']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": "-", "model": "factoring_qa",
            "split": "train", "metric": "f_has_factoring_size_rho",
            "value": _fmt(rho_f["rho_login3"]),
            "coverage": _fmt(fact["cm_share"]),
            "notes": f"SIZE={rho_f['is_size']} dark={p6['n_dark']} erp={p6['n_erp']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "C", "y": Y3, "model": "factoring_qa",
            "split": "train_cv", "metric": "auroc_c_n_days_with_tx",
            "value": _fmt(p3["days_y3"]["cv"]),
            "coverage": "1.0000",
            "notes": f"replica of published 0.711; size={p3['size_y3']['cv']:.4f}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y3, "model": "factoring_qa",
            "split": "train", "metric": "q3_new_wc_keep",
            "value": "1" if ctx["p4"]["any_q3_keep"] else "0",
            "coverage": _fmt(ctx["p4"]["n_new_wc"] / p1["n_cm"] if p1["n_cm"] else float("nan")),
            "notes": f"birth={ctx['p4']['share_new_is_birth']:.3f} min_lab={ctx['p4']['min_lab']} CLOSE_Q3",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y5_AP, "model": "factoring_qa",
            "split": "train", "metric": "y5_ap_pp_factoring_sizew",
            "value": _fmt(ctx["p11"]["fact_pp_size"]),
            "coverage": _fmt(fact["cm_share"]),
            "notes": f"conf_pp={ctx['p11']['conf_pp_size']:.4f} descriptive_never_X leftover_cm={ctx['p5']['n_y5_left_cm']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y3, "model": "factoring_qa",
            "split": "train", "metric": "ever_factoring_n_y3_pos_companies",
            "value": _fmt(ctx["p32"]["rows"][0]["n_ever_y3"]),
            "coverage": _fmt(fact["ever_share_co"]),
            "notes": f"never_recoverer 0/{fact['ever_n']}; rest_share={ctx['p32']['rest_share']:.3f} CLOSE_44",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y5_AP, "model": "factoring_qa",
            "split": "train", "metric": "q5_confirming_only_is_ap",
            "value": "1" if ctx["p47"]["only_is_ap"] else "0",
            "coverage": _fmt(ctx["p47"]["only_ap"]),
            "notes": (
                f"only={ctx['p47']['only_ap']:.4f} stack={ctx['p47']['stack_ap']:.4f} "
                f"rest={ctx['p47']['rest_ap']:.4f} stack_pp_size={ctx['p48']['stack_pp_size']:.4f} CLOSE_Q5"
            ),
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT,
            "x_families": "F", "y": Y3, "model": "factoring_qa",
            "split": "train", "metric": "drop_from_44",
            "value": "0" if p3["any_keep_44"] else "1",
            "coverage": _fmt(fact["cm_share"]),
            "notes": "drop f_has_factoring,f_has_confirming,f_has_loc,f_new_facility; keep f_ds_r,f_fc_r",
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
    print(f"factoring_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    con = connect()
    products = load_products(con)
    panel, pop = load_panel(con)
    panel = attach_new_type_months(panel, products)
    tr = panel[panel["split"] == "train"].copy()
    ho = panel[panel["split"] == "holdout"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"ERP={int(tr.loc[tr['has_book'],'company_id'].nunique())} "
        f"dark={pop.get('n_train_dark')}"
    )

    print("pass 1 prevalence + rise-only")
    p1 = pass1_prevalence(tr, ho, products)
    for r in p1["rows"]:
        print(
            f"  {r['type']}: ever={r['ever_n']} cm={_pp(r['cm_share'])} "
            f"rise_only={r['rise_only']} drops={r['n_drop']}"
        )

    print("pass 2 size ρ")
    p2 = pass2_size(tr)
    for r in p2["rows"]:
        print(f"  {r['flag']} ρ={r['rho_login3']:.3f} SIZE={r['is_size']}")

    print("pass 3 singles Y3/Y2")
    p3 = pass3_singles(tr)
    print(
        f"  days Y3={p3['days_y3']['cv']:.3f} size={p3['size_y3']['cv']:.3f} "
        f"keep44={p3['any_keep_44']}"
    )
    for v in p3["verdicts"]:
        if v["feature"] in FLAGS:
            print(f"  {v['feature']} CV={v['cv']:.3f} Δsize={v['beat_size']:.3f}")

    print("pass 4 Q3 new typed facility")
    p4 = pass4_q3(tr)
    print(f"  new_wc={p4['n_new_wc']} birth={_pp(p4['share_new_is_birth'])} q3_keep={p4['any_q3_keep']}")

    print("pass 5 overlap Y4 / Y5 leftover / FX")
    p5 = pass5_overlap(tr)
    print(f"  fx={p5['n_fx']} confirm228={p5['confirm_228']} leftover_cm={p5['n_y5_left_cm']}")

    print("pass 6 dark vs 744")
    p6 = pass6_dark(tr, pop)
    print(f"  confirm470/744={p6['confirm_470_744']}")

    print("pass 7 bank_name")
    p7 = pass7_banks(products)
    for r in p7["rows"]:
        print(f"  {r['type']}: banks={r['n_banks']} empresas={r['empresas']} other={r['inhouse_or_other']}")

    print("pass 8 confirming AP-side")
    p8 = pass8_ap(tr)
    print(f"  ap_side={p8['ap_side']} conf_ap_pp={p8['conf_ap_pp']}")

    print("pass 9 group dummy")
    p9 = pass9_group(tr)
    print(f"  any_dummy={p9['any_dummy']}")

    print("pass 10 co-occur + size Y")
    p10 = pass10_cooccur_size_y(tr)
    print(f"  all3={p10['overlap']['all3']} fact_conf={p10['overlap']['fact_conf']}")

    print("pass 11 Y5 AP after size")
    p11 = pass11_y5_size(tr)
    print(f"  fact_pp_size={p11['fact_pp_size']} conf_pp_size={p11['conf_pp_size']}")

    print("pass 12 dark WC")
    p12 = pass12_dark_wc(tr, pop)
    print(f"  dark fact n={p12['rows'][0]['dark_n']} med_in3={p12['rows'][0]['dark_med_in3']}")

    print("pass 13 Empresas")
    p13 = pass13_empresas(tr, products)
    print(f"  both={p13['n_both']}")

    print("pass 14 GROUP_0139")
    p14 = pass14_group0139(tr)
    print(f"  n_fact={p14['n_fact']} ids={p14['ids']}")

    print("pass 16 Y3 zero on factoring")
    p16 = pass16_y3_zero(tr)
    print(f"  fact y3_pos_on={p16['rows'][0]['y3_pos_on']} / {p16['rows'][0]['y3_lab_on']}")

    print("pass 17 connect after cash")
    p17 = pass17_connect_after_cash(tr, products)
    print(f"  fact p50 lag={p17['rows'][0]['p50']} after={p17['rows'][0]['share_after_trail']}")

    print("pass 18 granted last-month extract")
    p18 = pass18_granted_last(con, tr)
    print(f"  fact granted_p50={p18['rows'][0]['granted_p50']}")

    print("pass 19 2025-10 wave")
    p19 = pass19_wave_oct(tr)
    print(f"  oct fact n={p19['n']} dark={p19['n_dark']}")

    print("pass 20 factoring AP who")
    p20 = pass20_fact_ap_who(tr)
    print(f"  ap_co={p20['n_ap_co']} left_co={p20['n_left_co']}")

    print("pass 21 leftover is GROUP_0139?")
    p21 = pass21_leftover_is_group(p14, p20)
    print(f"  left_in_g={p21['n_in_g']}/{p21['n_left']} outside={p21['outside']}")

    print("pass 22 siblings share WC")
    p22 = pass22_siblings(tr)
    print(f"  fact multi-groups={p22['rows'][0]['n_multi']}")

    print("pass 23 currency")
    p23 = pass23_currency(products)
    print(f"  fact n_ccy={p23['rows'][0]['n_ccy']}")

    print("pass 24 store vs live last-month")
    p24 = pass24_store_vs_live(tr, products)
    print(f"  mismatch={p24['mismatch']} last={p24['last']}")

    print("pass 25 ICC + connected-only")
    p25 = pass25_icc_connected(tr)
    print(f"  loc_connected Δsize={p25['loc_beat']} icc_fact={p25['icc_rows'][0]['icc']}")

    print("pass 26 Y4 after size")
    p26 = pass26_y4_size(tr)
    print(f"  leak_ok={p26['leak_ok']} (Y4 forbids F — descriptive)")

    print("pass 27 confirming AP without GROUP_0139")
    p27 = pass27_conf_ap_nogroup(tr)
    print(f"  conf_pp={p27['conf_pp']} fact_pp={p27['fact_pp']} fact_n={p27['fact_n']}")

    print("pass 28 pre vs post first connection")
    p28 = pass28_prepost(tr)
    print(f"  fact Y3 before/after={p28['rows'][0]['y3_recover_cash_6m_before']}/{p28['rows'][0]['y3_recover_cash_6m_after']}")

    print("pass 29 holdout coverage")
    p29 = pass29_holdout_cov(ho)
    print(f"  hold fact ever={p29['rows'][0]['ever']}")

    print("pass 30 FX ∩ WC size")
    p30 = pass30_fx_overlap_size(tr)
    print(f"  fact∩fx={p30['rows'][0]['n_both']}")

    print("pass 31 Y2 pre/post size+year")
    p31 = pass31_prepost_y2_ctrl(tr)
    print(f"  terc pp={[ (r['slice'], r['pp']) for r in p31['terc'] ]}")

    print("pass 32 never-recoverer set")
    p32 = pass32_never_recover(tr)
    print(f"  fact ever Y3+ = {p32['rows'][0]['n_ever_y3']}/{p32['rows'][0]['n']}")

    print("pass 33 trail length")
    p33 = pass33_trail(tr)
    print(f"  fact p50_mo={p33['rows'][0]['p50_mo']} rest={p33['rows'][3]['p50_mo']}")

    print("pass 34 COMP_0919")
    p34 = pass34_comp0919(tr)
    print(f"  group={p34.get('group')} left={p34.get('y5_left')} book={p34.get('has_book')}")

    print("pass 35 n products per company")
    p35 = pass35_nprod(products)
    print(f"  fact multi={p35['rows'][0]['share_multi']} p50={p35['rows'][0]['p50']}")

    print("pass 36 product vs company.created_at")
    p36 = pass36_company_onboard(con, products)
    print(f"  fact p50_d={p36['rows'][0]['p50_d']} after_co={p36['rows'][0]['share_after_co']}")

    print("pass 37 Y3 label coverage")
    p37 = pass37_y3_coverage(tr)
    print(f"  fact lab_share={p37['rows'][0]['lab_share']} rate={p37['rows'][0]['rate_among_lab']}")

    print("pass 38 product labels")
    p38 = pass38_labels(products)
    print(f"  fact labels={p38['rows'][0]['n_lab']}")

    print("pass 39 size-matched never-recover")
    p39 = pass39_size_matched_never(tr)
    print(f"  {p39['rows']}")

    print("pass 40 in-house LOC")
    p40 = pass40_loc_inhouse(tr, products)
    print(f"  inhouse n={p40['rows'][0]['n_co']} bank n={p40['rows'][1]['n_co']}")

    print("pass 41 extract liquidity")
    p41 = pass41_liquidity(products)
    print(f"  fact liq_def={p41['rows'][0]['share_def']} p50={p41['rows'][0]['p50']}")

    print("pass 42 service codes")
    p42 = pass42_service(products)
    print(f"  fact custom={p42['rows'][0]['custom']} n_svc={p42['rows'][0]['n_svc']}")

    print("pass 43 observed flows on WC months")
    p43 = pass43_flows(tr)
    print(f"  fact ds share_gt0={p43['rows'][0]['share_gt0_on']}")

    print("pass 44 factoring-only vs stacked")
    p44 = pass44_fact_only_flow(tr)
    print(f"  only={p44['n_only']} ds_gt0={p44['ds_gt0_only']} both_ds={p44['ds_gt0_both']}")

    print("pass 45 other debt types on factoring-only")
    p45 = pass45_other_debt(con, p44["ids_only"])
    print(f"  other-debt rows={len(p45['rows'])}")

    print("pass 46 confirming-only AP")
    p46 = pass46_conf_only_ap(tr)
    print(f"  n={p46['n_co']} ap={p46['ap']['rate']} ar={p46['ar']['rate']}")

    print("pass 47 confirming only vs stacked AP")
    p47 = pass47_conf_stack_ap(tr)
    print(f"  only_ap={p47['only_ap']} stack={p47['stack_ap']} rest={p47['rest_ap']} only_is_ap={p47['only_is_ap']}")

    print("pass 48 stacked confirming AP after size")
    p48 = pass48_stack_ap_size(tr)
    print(f"  stack_pp_size={p48['stack_pp_size']} only_pp_size={p48['only_pp_size']}")

    print("pass 49 unused limit")
    p49 = pass49_unused_limit(con)
    print(f"  fact unused_p50={p49['rows'][0]['unused_p50']} drawn={p49['rows'][0]['share_drawn']}")

    print("pass 50 LOC-only")
    p50 = pass50_loc_only(tr)
    print(f"  n={p50['n_co']} y3={p50['y3']['rate']} y2={p50['y2']['rate']}")

    print("pass 51 confirming banks only vs stacked")
    p51 = pass51_conf_banks(tr, products)
    print(f"  only_banks={p51['rows'][0]['n_banks']} stack_banks={p51['rows'][1]['n_banks']}")

    print("pass 52 drawn factoring")
    p52 = pass52_drawn_factoring(con, tr)
    print(f"  drawn={p52['rows'][0]['n_co']} connected={p52['rows'][1]['n_co']} y3_drawn={p52['rows'][0]['y3']['rate']}")

    print("pass 53 drawn after size")
    p53 = pass53_drawn_size(con, tr)
    print(f"  n_drawn={p53['n_drawn']} y2_pp_size={p53['y2_pp_size']}")

    print("pass 54 drawn without GROUP_0139")
    p54 = pass54_drawn_wo_g0139(con, tr)
    print(f"  n_wo={p54['n_wo']} n_g={p54['n_g']} y2={p54['y2']['rate']} ap={p54['ap']['rate']}")

    print("pass 55 drawn confirming")
    p55 = pass55_drawn_confirming(con, tr)
    print(f"  drawn={p55['rows'][0]['n_co']} y2={p55['rows'][0]['y2']['rate']} ap={p55['rows'][0]['ap']['rate']}")

    print("pass 56 drawn confirming Y2 after size")
    p56 = pass56_drawn_conf_y2_size(con, tr)
    print(f"  n={p56['n_drawn']} y2_pp_size={p56['y2_pp_size']}")

    print("pass 57 holdout WC ids (coverage)")
    p57 = pass57_holdout_ids(ho)
    print(f"  hold WC last-mo n={p57['n']} fact={p57['fact_ids']} top={p57['top_group']}")

    print("pass 58 holdout WC groups unseen")
    p58 = pass58_holdout_group_unseen(tr, p57)
    print(f"  unseen={p58['unseen']} overlap={p58['overlap']}")

    png_ok = make_png(tr)
    print(f"png={OUT_PNG if png_ok else 'skipped'}")

    decisions = _decisions(p1, p2, p3, p4, p9, p47)
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5,
        "p6": p6, "p7": p7, "p8": p8, "p9": p9,
        "p10": p10, "p11": p11, "p12": p12, "p13": p13, "p14": p14,
        "p16": p16, "p17": p17, "p18": p18, "p19": p19, "p20": p20,
        "p21": p21, "p22": p22, "p23": p23, "p24": p24,
        "p25": p25, "p26": p26, "p27": p27,
        "p28": p28, "p29": p29, "p30": p30, "p31": p31, "p32": p32,
        "p33": p33, "p34": p34, "p35": p35, "p36": p36,
        "p37": p37, "p38": p38, "p39": p39, "p40": p40, "p41": p41, "p42": p42,
        "p43": p43, "p44": p44, "p45": p45, "p46": p46, "p47": p47, "p48": p48,
        "p49": p49, "p50": p50, "p51": p51, "p52": p52, "p53": p53, "p54": p54,
        "p55": p55, "p56": p56, "p57": p57, "p58": p58,
        "decisions": decisions,
        "png_ok": png_ok,
        "elapsed_s": time.time() - t0,
    }
    write_md(ctx)
    print(f"wrote {OUT_MD}")
    append_registry(ctx)
    con.close()
    print(f"factoring_qa done elapsed={ctx['elapsed_s']:.1f}s")
    return ctx


def main() -> int:
    run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
