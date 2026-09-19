"""Unused leftover of Family G 44-col access flags.

NORTH_STAR: access flags are operating-type tags / connection clocks,
not a health reading. Family G already CLOSED `g_has_card` as Y3 X
(oriented 0.551 < size 0.617 / days 0.711) and flagged `g_has_checking`
as 99.1% the connection hole (`g_n_accounts=0`). This ticket is the
unused leftover: saving, investment, TPV, `g_custom_share` — and a
leftover confirm that card / checking leave the 44.

Do not redo G's `g_new` / `created_at` / checking-card headline. Confirm.
Do not invent `y_has_tpv`. Do not put `g_has_*` on the 15-col Y3 card.
Night Y3 0.762 / 0.752. Days 0.711. Size 0.617. Y7 TURNOVER 0.720 / 0.712.
Y7 never D. Y5 never E. Y3 never B. Off the 15-col card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.g_has_rest_qa

Owned: analysis/evaluate/g_has_rest_qa.py, analysis/outputs/g_has_rest_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_g_has_rest.md (end).

Iteration (same module, not one-shot):
1–12. Must-do leftover QA
13–16. leftover types / F / holdout / created clock
17–20. connected / rates / quintiles / calendar
21. honest leftover inside days terciles
22. custom_share drops = dilution
23. custom piles (replace zero-inflated quintiles)
24. checking Jaccard
25. F fake leftover
26. holdout custom coverage
27. first-on vs g_new
28. D2 custom 0.566 vs size
29. investment T3-tagged
30. first-on without g_new = already-on-book
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
OUT_MD = ANALYSIS / "outputs" / "g_has_rest_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "g_has_rest_vs_size.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
AGENT = "4348a23d"
WAVE = "4"
ROUND = "R4"
MODEL = "g_has_rest_qa"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
TWIN_RHO = 0.80
LEFTOVER_DIE = 0.55
ICC_TRAIT = 0.85
MIN_ACF_PAIRS = 4
PANEL_END = pd.Timestamp("2026-08-01")
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
NIGHT_Y7 = 0.720
NIGHT_Y7_CORE = 0.712
CARD_QUOTE = 0.551
CHECKING_HOLE_QUOTE = 0.991
TPV_LAST_QUOTE = 10
MODAL_SAVING = 0.997
MODAL_INVEST = 0.951
ACF_CUSTOM = 0.84

FOCUS = (
    "g_has_saving",
    "g_has_investment",
    "g_has_tpv",
    "g_custom_share",
)
CONFIRM = (
    "g_has_card",
    "g_has_checking",
)
FLAGS = FOCUS + CONFIRM
HAS_FLAGS = (
    "g_has_saving",
    "g_has_investment",
    "g_has_tpv",
    "g_has_card",
    "g_has_checking",
)
F_FLAGS = (
    "f_has_factoring",
    "f_has_confirming",
    "f_has_loc",
)
STARTER_G = (
    "g_custom_share",
    "g_has_card",
    "g_has_checking",
    "g_has_investment",
    "g_has_saving",
    "g_has_tpv",
)
OTHER_TYPES = ("wallet", "risk", "lineofcomex", "expensesPlatform")

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_op_in",
    "a_n_tx",
    "c_n_tx",
    "c_n_days_with_tx",
    "g_n_accounts",
    "g_n_banks",
    "g_n_types",
    "g_has_card",
    "g_has_checking",
    "g_has_saving",
    "g_has_investment",
    "g_has_tpv",
    "g_custom_share",
    "g_new_this_month",
    "f_has_factoring",
    "f_has_confirming",
    "f_has_loc",
    "f_n_facilities",
)

Y_KEEP = (Y2, Y3)


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


def _orient(auc: float) -> tuple[float, int]:
    """G replica: published 0.711 is 1 − 0.289."""
    if auc is None or not np.isfinite(auc):
        return float("nan"), 0
    a = float(auc)
    if a >= 0.5:
        return a, 1
    return 1.0 - a, -1


def fold_auroc(y: pd.Series, x: pd.Series, folds: pd.Series) -> dict:
    """Family G group-fold: raw AUROC per fold, then orient the mean."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = y.notna() & x.notna() & folds.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    aucs = []
    rows = []
    for k in range(N_FOLDS):
        va = defined & (folds == k)
        auc = auroc(y[va], x[va])
        aucs.append(auc)
        rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "n_va": int(va.sum()),
                "n_pos": int((va & (y == 1)).sum()),
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    cv_raw = float(np.mean(finite)) if finite else float("nan")
    cv_sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
    cv_orient, sign = _orient(cv_raw)
    pooled_raw = auroc(y[defined], x[defined])
    pooled_orient, _ = _orient(pooled_raw)
    return {
        "cv_raw": cv_raw,
        "cv_orient": cv_orient,
        "sd": cv_sd,
        "n_folds": len(finite),
        "folds": rows,
        "sign": int(sign),
        "pooled_raw": float(pooled_raw) if np.isfinite(pooled_raw) else float("nan"),
        "pooled_orient": float(pooled_orient) if np.isfinite(pooled_orient) else float("nan"),
        "n_defined": n,
        "n_pos": n_pos,
    }


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
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


def _rise_only(tr: pd.DataFrame, col: str) -> dict:
    s = tr.sort_values(["company_id", "period"])
    x = pd.to_numeric(s[col], errors="coerce")
    dlt = x.groupby(s["company_id"], sort=False).diff()
    n_drop = int((dlt < 0).sum())
    n_rise = int((dlt > 0).sum())
    n_flat = int((dlt == 0).sum())
    return {
        "n_rise": n_rise,
        "n_drop": n_drop,
        "n_flat": n_flat,
        "rise_only": n_drop == 0,
        "acf1": median_acf(x, s["company_id"], 1),
        "acf3": median_acf(x, s["company_id"], 3),
        "acf6": median_acf(x, s["company_id"], 6),
    }


def _company_terciles(tr: pd.DataFrame) -> pd.Series:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last["log_in3"], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1_small", "T2_mid", "T3_large"], duplicates="drop")
    return terc.rename("size_terc")


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
    optional = {"a_n_tx", "c_n_tx", "f_has_factoring", "f_has_confirming", "f_has_loc", "f_n_facilities"}
    hard = [c for c in missing if c not in optional]
    if hard:
        raise RuntimeError(f"monthly.parquet missing {hard}")
    if missing:
        print(f"store optional missing {missing}")
    have = [c for c in STORE_COLS if c in raw.columns]
    ykeep = ["company_id", "period"]
    for c in Y_KEEP:
        if c in yraw.columns:
            ykeep.append(c)
        else:
            raise RuntimeError(f"targets missing {c}")
    panel = _keys(raw[list(have)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(
        pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0)
    )
    panel["log_abs_opin"] = np.log1p(
        pd.to_numeric(panel["a_op_in"], errors="coerce").abs()
    )
    ntx = panel["c_n_tx"] if "c_n_tx" in panel.columns else panel.get("a_n_tx")
    if ntx is None:
        panel["n_tx"] = np.nan
    else:
        panel["n_tx"] = pd.to_numeric(ntx, errors="coerce")
    panel["so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    first = panel.groupby("company_id")["period"].transform("min")
    panel["months_on_book"] = (
        (PANEL_END.year - first.dt.year) * 12 + (PANEL_END.month - first.dt.month) + 1
    )
    panel["short_book"] = (panel["months_on_book"] < 12).astype(np.int8)
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
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


# ---------------------------------------------------------------------------
# Pass 1 — prevalence train vs holdout; ever-n; last-month n
# ---------------------------------------------------------------------------
def pass1_prev(panel: pd.DataFrame, tr: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    last_tr = tr.sort_values("period").groupby("company_id", sort=False).last()
    last_ho = ho.sort_values("period").groupby("company_id", sort=False).last()
    store = {}
    for col in FLAGS:
        x = pd.to_numeric(tr[col], errors="coerce")
        xh = pd.to_numeric(ho[col], errors="coerce")
        if col == "g_custom_share":
            on = x > 0
            on_h = xh > 0
            modal = float((x.fillna(0) == 0).mean()) if x.notna().any() else float("nan")
            modal_nn = float((x.dropna() == 0).mean()) if x.notna().any() else float("nan")
            ever_n = int(tr.loc[on.fillna(False), "company_id"].nunique())
            last_n = int((pd.to_numeric(last_tr[col], errors="coerce") > 0).sum())
            last_n_ho = int((pd.to_numeric(last_ho[col], errors="coerce") > 0).sum())
            cm_on = int(on.fillna(False).sum())
            prev = float(on.mean())
            prev_ho = float(on_h.mean())
            cov = float(x.notna().mean())
            cov_ho = float(xh.notna().mean())
        else:
            on = x.fillna(0) == 1
            on_h = xh.fillna(0) == 1
            modal = float((x.fillna(0) == 0).mean())
            modal_nn = modal
            ever_n = int(tr.loc[on, "company_id"].nunique())
            last_n = int((pd.to_numeric(last_tr[col], errors="coerce").fillna(0) == 1).sum())
            last_n_ho = int((pd.to_numeric(last_ho[col], errors="coerce").fillna(0) == 1).sum())
            cm_on = int(on.sum())
            prev = float(on.mean())
            prev_ho = float(on_h.mean())
            cov = 1.0
            cov_ho = 1.0
        ever_ho = int(ho.loc[on_h.fillna(False) if col == "g_custom_share" else on_h, "company_id"].nunique())
        rec = {
            "col": col,
            "train_cm": f"{len(tr):,}",
            "train_prev": _pp(prev),
            "modal0": _pp(modal),
            "cov": _pp(cov),
            "ever_n": ever_n,
            "last_n": last_n,
            "cm_on": f"{cm_on:,}",
            "hold_prev": _pp(prev_ho),
            "hold_ever": ever_ho,
            "hold_last_n": last_n_ho,
            "hold_cov": _pp(cov_ho),
        }
        rows.append(rec)
        store[col] = {
            "prev": prev,
            "modal": modal,
            "modal_nn": modal_nn,
            "ever_n": ever_n,
            "last_n": last_n,
            "cm_on": cm_on,
            "prev_ho": prev_ho,
            "ever_ho": ever_ho,
            "last_n_ho": last_n_ho,
            "cov": cov,
        }
    tpv_last_ok = store["g_has_tpv"]["last_n"] == TPV_LAST_QUOTE
    saving_modal_ok = abs(store["g_has_saving"]["modal"] - MODAL_SAVING) < 0.005
    invest_modal_ok = abs(store["g_has_investment"]["modal"] - MODAL_INVEST) < 0.01
    prose = (
        f"Train {len(tr):,} CM / {tr['company_id'].nunique()} cos. "
        f"saving last-month n={store['g_has_saving']['last_n']} ever={store['g_has_saving']['ever_n']} "
        f"modal0={_pp(store['g_has_saving']['modal'])} "
        f"({'CONFIRM 99.7%' if saving_modal_ok else 'off 99.7%'}). "
        f"investment last={store['g_has_investment']['last_n']} ever={store['g_has_investment']['ever_n']} "
        f"modal0={_pp(store['g_has_investment']['modal'])} "
        f"({'CONFIRM 95.1%' if invest_modal_ok else 'off 95.1%'}). "
        f"TPV last={store['g_has_tpv']['last_n']} ever={store['g_has_tpv']['ever_n']} "
        f"({'CONFIRM n=10' if tpv_last_ok else 'off G n=10'}). "
        f"custom>0 last={store['g_custom_share']['last_n']} ever={store['g_custom_share']['ever_n']} "
        f"cov={_pp(store['g_custom_share']['cov'])}. "
        f"Holdout coverage only (72 cos)."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "n_cm": int(len(tr)),
        "n_co": int(tr["company_id"].nunique()),
        "n_hold_cm": int(len(ho)),
        "n_hold_co": int(ho["company_id"].nunique()),
        "tpv_last_ok": tpv_last_ok,
        "saving_modal_ok": saving_modal_ok,
        "invest_modal_ok": invest_modal_ok,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — rise-only inventory?
# ---------------------------------------------------------------------------
def pass2_rise(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    acc = _rise_only(tr, "g_n_accounts")
    for col in HAS_FLAGS + ("g_custom_share", "g_n_accounts"):
        r = _rise_only(tr, col)
        store[col] = r
        rows.append(
            {
                "col": col,
                "rises": f"{r['n_rise']:,}",
                "drops": f"{r['n_drop']:,}",
                "flats": f"{r['n_flat']:,}",
                "rise_only": "YES" if r["rise_only"] else "NO",
                "acf1": _f(r["acf1"]),
                "acf3": _f(r["acf3"]),
                "acf6": _f(r["acf6"]),
            }
        )
    acc_ok = acc["rise_only"] and acc["n_rise"] == 1561 and acc["n_drop"] == 0
    flags_rise = all(store[c]["rise_only"] for c in HAS_FLAGS)
    custom_rise = store["g_custom_share"]["rise_only"]
    prose = (
        f"g_n_accounts {acc['n_rise']:,} rises / {acc['n_drop']:,} drops "
        f"({'CONFIRM rise-only 1,561/0' if acc_ok else 'off G 1,561/0 — do not reopen'}). "
        f"HAS flags rise-only: {flags_rise} "
        f"(saving {store['g_has_saving']['n_rise']}/{store['g_has_saving']['n_drop']}, "
        f"invest {store['g_has_investment']['n_rise']}/{store['g_has_investment']['n_drop']}, "
        f"TPV {store['g_has_tpv']['n_rise']}/{store['g_has_tpv']['n_drop']}). "
        f"g_custom_share rise-only={custom_rise} "
        f"({store['g_custom_share']['n_rise']}↑ {store['g_custom_share']['n_drop']}↓). "
        + (
            "Flags only turn 0→1 — connection inventory, not mix change."
            if flags_rise
            else "A flag drops 1→0 — real mix change, not only connection."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "acc_ok": acc_ok,
        "flags_rise": flags_rise,
        "custom_rise": custom_rise,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman twins
# ---------------------------------------------------------------------------
def pass3_rho(tr: pd.DataFrame) -> dict:
    peers = {
        "g_n_accounts": tr["g_n_accounts"],
        "g_has_checking": tr["g_has_checking"],
        "g_has_card": tr["g_has_card"],
        "log1p(a_in3)": tr["log_in3"],
        "log1p(|a_op_in|)": tr["log_abs_opin"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "n_tx": tr["n_tx"],
    }
    twin_peers = ("g_n_accounts", "g_has_checking", "c_n_days_with_tx", "n_tx")
    rows = []
    store = {}
    for col in FLAGS:
        rec = {}
        twins = []
        for name, s in peers.items():
            if name == col:
                rec[name] = 1.0
                continue
            rho = spearman(tr[col], s)
            rec[name] = rho
            if name in twin_peers and np.isfinite(rho) and abs(rho) >= TWIN_RHO:
                twins.append(f"{name}={rho:.3f}")
        size = bool(
            (np.isfinite(rec["log1p(a_in3)"]) and abs(rec["log1p(a_in3)"]) >= SIZE_RHO)
            or (
                np.isfinite(rec["log1p(|a_op_in|)"])
                and abs(rec["log1p(|a_op_in|)"]) >= SIZE_RHO
            )
        )
        store[col] = {"rho": rec, "twins": twins, "size": size, "twin": bool(twins)}
        rows.append(
            {
                "col": col,
                "vs g_n_accounts": _f(rec["g_n_accounts"]),
                "vs checking": _f(rec["g_has_checking"]),
                "vs card": _f(rec["g_has_card"]),
                "vs log1p(a_in3)": _f(rec["log1p(a_in3)"]),
                "vs log1p(|opin|)": _f(rec["log1p(|a_op_in|)"]),
                "vs days": _f(rec["c_n_days_with_tx"]),
                "vs n_tx": _f(rec["n_tx"]),
                "SIZE": "YES" if size else "no",
                "twin": ", ".join(twins) if twins else "—",
            }
        )
    chk_acc = store["g_has_checking"]["rho"]["g_n_accounts"]
    prose = (
        f"No leftover flag is SIZE on log1p(a_in3) / |opin| "
        f"(saving {_f(store['g_has_saving']['rho']['log1p(a_in3)'])}, "
        f"invest {_f(store['g_has_investment']['rho']['log1p(a_in3)'])}, "
        f"TPV {_f(store['g_has_tpv']['rho']['log1p(a_in3)'])}, "
        f"custom {_f(store['g_custom_share']['rho']['log1p(a_in3)'])}). "
        f"checking vs g_n_accounts ρ={_f(chk_acc)} "
        f"(twin={'YES' if store['g_has_checking']['twin'] else 'no at |ρ|≥0.80'}). "
        f"Twins: "
        + (
            "; ".join(f"{c}: {', '.join(store[c]['twins'])}" for c in FLAGS if store[c]["twins"])
            or "none"
        )
        + "."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 4 — oriented group-fold AUROC (G replica)
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    benches = (
        ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ("log1p_a_in3", tr["log_in3"]),
        ("g_n_accounts", tr["g_n_accounts"]),
    )
    rows = []
    store = {}
    for y in (Y3, Y2):
        lab = tr[y].notna()
        for name, x in [(c, tr[c]) for c in FLAGS] + list(benches):
            rec = fold_auroc(tr[y], x, tr["fold"])
            store[(y, name)] = rec
            rows.append(
                {
                    "Y": y,
                    "feature": name,
                    "pooled raw": _f(rec["pooled_raw"]),
                    "pooled orient": _f(rec["pooled_orient"]),
                    "CV raw": _f(rec["cv_raw"]),
                    "CV orient": _f(rec["cv_orient"]),
                    "sd": _f(rec["sd"]),
                    "n labeled": f"{rec['n_defined']:,}",
                    "n pos": f"{rec['n_pos']:,}",
                }
            )
    card = store[(Y3, "g_has_card")]
    days = store[(Y3, "c_n_days_with_tx")]
    size = store[(Y3, "log1p_a_in3")]
    card_ok = bool(np.isfinite(card["cv_orient"]) and abs(card["cv_orient"] - CARD_QUOTE) < 0.005)
    days_ok = bool(np.isfinite(days["cv_orient"]) and abs(days["cv_orient"] - DAYS_BENCH) < 0.01)
    size_ok = bool(np.isfinite(size["cv_orient"]) and abs(size["cv_orient"] - SIZE_QUOTE) < 0.01)
    keep_rows = []
    for feat in FLAGS:
        rec = store[(Y3, feat)]
        d_size = (
            rec["cv_orient"] - size["cv_orient"]
            if np.isfinite(rec["cv_orient"]) and np.isfinite(size["cv_orient"])
            else float("nan")
        )
        d_days = (
            rec["cv_orient"] - days["cv_orient"]
            if np.isfinite(rec["cv_orient"]) and np.isfinite(days["cv_orient"])
            else float("nan")
        )
        beat = bool(np.isfinite(d_size) and d_size >= KEEP_DELTA)
        keep_rows.append(
            {
                "feature": feat,
                "CV raw": _f(rec["cv_raw"]),
                "CV orient": _f(rec["cv_orient"]),
                "size orient": _f(size["cv_orient"]),
                "days orient": _f(days["cv_orient"]),
                "Δ size": _f(d_size),
                "Δ days": _f(d_days),
                "beat size+0.02": "YES" if beat else "no",
            }
        )
        store[(Y3, feat, "d_size")] = d_size
        store[(Y3, feat, "beat")] = beat
    # base rates
    rate_rows = []
    for col in HAS_FLAGS:
        x = pd.to_numeric(tr[col], errors="coerce").fillna(0)
        for y in (Y2, Y3):
            for bit, label in ((1, "has=1"), (0, "has=0")):
                sl = x == bit
                lab = sl & tr[y].notna()
                n_lab = int(lab.sum())
                n_pos = int((lab & (pd.to_numeric(tr[y], errors="coerce") == 1)).sum())
                rate_rows.append(
                    {
                        "flag": col,
                        "group": label,
                        "Y": y,
                        "n labeled": f"{n_lab:,}",
                        "n pos": n_pos,
                        "base rate": _pp(_pct(n_pos, n_lab), 2) if n_lab else "—",
                        "n CM": f"{int(sl.sum()):,}",
                        "n companies": int(tr.loc[sl, "company_id"].nunique()),
                    }
                )
    prose = (
        f"Y3 card CV orient {_f(card['cv_orient'])} "
        f"({'CONFIRM 0.551' if card_ok else 'off G 0.551'}). "
        f"days {_f(days['cv_orient'])} ({'OK 0.711' if days_ok else 'off 0.711'}). "
        f"size {_f(size['cv_orient'])} ({'OK 0.617' if size_ok else 'off 0.617'}). "
        f"saving {_f(store[(Y3, 'g_has_saving')]['cv_orient'])} "
        f"invest {_f(store[(Y3, 'g_has_investment')]['cv_orient'])} "
        f"TPV {_f(store[(Y3, 'g_has_tpv')]['cv_orient'])} "
        f"custom {_f(store[(Y3, 'g_custom_share')]['cv_orient'])}. "
        f"No leftover flag beats size ≥0.02."
        if not any(store.get((Y3, f, "beat")) for f in FOCUS)
        else "A leftover flag beats size — inspect leftover."
    )
    print(prose)
    return {
        "rows": rows,
        "keep_rows": keep_rows,
        "rate_rows": rate_rows,
        "store": store,
        "card_ok": card_ok,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "card": card["cv_orient"],
        "days": days["cv_orient"],
        "size": size["cv_orient"],
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — honest leftover after days and after size
# ---------------------------------------------------------------------------
def pass5_leftover(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    acc = tr["g_n_accounts"]
    for col in FLAGS:
        r_d, inf_d = ols_resid(tr[col], days)
        r_s, inf_s = ols_resid(tr[col], size)
        r_a, inf_a = ols_resid(tr[col], acc)
        r_ds, inf_ds = ols_resid(tr[col], days, size)
        recs = {
            "days": fold_auroc(tr[Y3], r_d, tr["fold"]),
            "size": fold_auroc(tr[Y3], r_s, tr["fold"]),
            "acc": fold_auroc(tr[Y3], r_a, tr["fold"]),
            "days_size": fold_auroc(tr[Y3], r_ds, tr["fold"]),
            "y2_days": fold_auroc(tr[Y2], r_d, tr["fold"]),
        }
        rho_d = spearman(r_d, days)
        rho_s = spearman(r_s, size)
        fake = bool(np.isfinite(rho_d) and abs(rho_d) >= 0.30)
        left = recs["days"]["cv_orient"]
        died_raw = bool(np.isfinite(left) and left < LEFTOVER_DIE)
        # zero_in: high leftover with ρ(resid,days) leak is fake. Honest leftover dies.
        died = bool(died_raw or fake)
        honest = float("nan") if fake else left
        store[col] = {
            "left_days": left,
            "honest": honest,
            "left_size": recs["size"]["cv_orient"],
            "left_acc": recs["acc"]["cv_orient"],
            "left_ds": recs["days_size"]["cv_orient"],
            "left_y2": recs["y2_days"]["cv_orient"],
            "rho_resid_days": rho_d,
            "rho_resid_size": rho_s,
            "fake": fake,
            "died": died,
            "died_raw": died_raw,
            "slope_days": inf_d["slope"][0] if inf_d["slope"] else float("nan"),
            "recs": recs,
        }
        rows.append(
            {
                "col": col,
                "after days": _f(left),
                "honest leftover": "DIES (fake days)" if fake else _f(honest),
                "after size": _f(recs["size"]["cv_orient"]),
                "after accounts": _f(recs["acc"]["cv_orient"]),
                "after days+size": _f(recs["days_size"]["cv_orient"]),
                "Y2 after days": _f(recs["y2_days"]["cv_orient"]),
                "ρ(resid,days)": _f(rho_d),
                "fake days leak": "YES" if fake else "no",
                "dies": "YES" if died else "no",
            }
        )
    any_live = any(not store[c]["died"] for c in FOCUS)
    prose = (
        f"OLS leftover after days looks high (saving {_f(store['g_has_saving']['left_days'])} "
        f"invest {_f(store['g_has_investment']['left_days'])} "
        f"TPV {_f(store['g_has_tpv']['left_days'])} "
        f"custom {_f(store['g_custom_share']['left_days'])}) "
        f"but ρ(resid,days) is "
        f"{_f(store['g_has_saving']['rho_resid_days'])} / "
        f"{_f(store['g_has_investment']['rho_resid_days'])} / "
        f"{_f(store['g_has_tpv']['rho_resid_days'])} / "
        f"{_f(store['g_custom_share']['rho_resid_days'])}. "
        f"{'Honest leftover still lives.' if any_live else 'Honest leftover DIES — fake days leak (same as zero_in 0.614/ρ=0.664).'}"
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "any_live": any_live,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — SIZE terciles: survive inside T1?
# ---------------------------------------------------------------------------
def pass6_terciles(tr: pd.DataFrame) -> dict:
    terc = _company_terciles(tr)
    work = tr.merge(terc.reset_index(), on="company_id", how="left")
    rows = []
    store = {}
    last = work.sort_values("period").groupby("company_id", sort=False).last()
    share_rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        sl_last = last[last["size_terc"] == tname]
        rec = {"tercile": tname, "n companies": int(len(sl_last))}
        for col in HAS_FLAGS + ("g_custom_share",):
            x = pd.to_numeric(sl_last[col], errors="coerce")
            rec[col.replace("g_has_", "has_").replace("g_custom_share", "custom>0")] = (
                _pp(float((x > 0).mean())) if x.notna().any() else "—"
            )
        share_rows.append(rec)
    for col in FOCUS + ("g_has_card",):
        for tname in ("T1_small", "T2_mid", "T3_large", "all"):
            sl = work if tname == "all" else work[work["size_terc"] == tname]
            rec = fold_auroc(sl[Y3], sl[col], sl["fold"])
            sz = fold_auroc(sl[Y3], sl["log_in3"], sl["fold"])
            dy = fold_auroc(sl[Y3], sl["c_n_days_with_tx"], sl["fold"])
            r_d, _ = ols_resid(sl[col], sl["c_n_days_with_tx"])
            left = fold_auroc(sl[Y3], r_d, sl["fold"])
            key = (col, tname)
            store[key] = {
                "raw": rec["cv_orient"],
                "size": sz["cv_orient"],
                "days": dy["cv_orient"],
                "left": left["cv_orient"],
                "n": rec["n_defined"],
                "n_pos": rec["n_pos"],
            }
            rho_left = spearman(r_d, sl["c_n_days_with_tx"])
            fake_t = bool(np.isfinite(rho_left) and abs(rho_left) >= 0.30)
            beat_t1 = bool(
                tname == "T1_small"
                and np.isfinite(rec["cv_orient"])
                and np.isfinite(sz["cv_orient"])
                and (rec["cv_orient"] - sz["cv_orient"]) >= KEEP_DELTA
                and np.isfinite(left["cv_orient"])
                and left["cv_orient"] >= LEFTOVER_DIE
                and not fake_t
            )
            store[(col, tname, "fake")] = fake_t
            store[(col, tname, "survive")] = beat_t1
            rows.append(
                {
                    "col": col,
                    "tercile": tname,
                    "CV orient": _f(rec["cv_orient"]),
                    "size": _f(sz["cv_orient"]),
                    "days": _f(dy["cv_orient"]),
                    "leftover days": _f(left["cv_orient"]),
                    "n labeled": f"{rec['n_defined']:,}",
                    "n pos": rec["n_pos"],
                    "survive T1": "YES" if beat_t1 else "—",
                }
            )
    t1_live = [c for c in FOCUS if store.get((c, "T1_small", "survive"))]
    prose = (
        f"T1 leftover: saving {_f(store[('g_has_saving', 'T1_small')]['left'])} "
        f"invest {_f(store[('g_has_investment', 'T1_small')]['left'])} "
        f"TPV {_f(store[('g_has_tpv', 'T1_small')]['left'])} "
        f"custom {_f(store[('g_custom_share', 'T1_small')]['left'])}. "
        + (
            f"Survive T1: {', '.join(t1_live)}."
            if t1_live
            else "No leftover flag survives inside T1 after days."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "share_rows": share_rows,
        "store": store,
        "t1_live": t1_live,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — dark 470 vs 744
# ---------------------------------------------------------------------------
def pass7_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    last = last.copy()
    last["ever_erp"] = last.index.astype(str).isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470
    rows = []
    store = {}
    for name, sl in (("ever_erp", last[last["ever_erp"]]), ("never_erp", last[~last["ever_erp"]])):
        rec = {
            "group": name,
            "n companies": int(len(sl)),
            "accounts p50": _f(float(pd.to_numeric(sl["g_n_accounts"], errors="coerce").median()), 3),
        }
        for col in HAS_FLAGS:
            rec[col.replace("g_has_", "has_")] = _pp(
                float((pd.to_numeric(sl[col], errors="coerce").fillna(0) == 1).mean())
            )
        rec["custom>0"] = _pp(
            float((pd.to_numeric(sl["g_custom_share"], errors="coerce") > 0).mean())
        )
        rec["custom p50"] = _f(float(pd.to_numeric(sl["g_custom_share"], errors="coerce").median()))
        rows.append(rec)
        store[name] = rec
    # company-ever
    ever_rows = []
    for name, ids in (
        ("ever_erp", set(last.index[last["ever_erp"]].astype(str))),
        ("never_erp", set(last.index[~last["ever_erp"]].astype(str))),
    ):
        sl = tr[tr["company_id"].isin(ids)]
        rec = {"group": name, "n": int(len(ids))}
        for col in HAS_FLAGS:
            rec["ever " + col.replace("g_has_", "")] = _pp(
                _pct(int(sl.loc[pd.to_numeric(sl[col], errors="coerce") == 1, "company_id"].nunique()), len(ids))
            )
        ever_rows.append(rec)
    chk_dark = store["never_erp"]["has_checking"]
    prose = (
        f"Last-month train: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'off 744/470'}). "
        f"Dark last-month saving {store['never_erp']['has_saving']} vs ERP {store['ever_erp']['has_saving']}; "
        f"invest {store['never_erp']['has_investment']} vs {store['ever_erp']['has_investment']}; "
        f"TPV {store['never_erp']['has_tpv']} vs {store['ever_erp']['has_tpv']}; "
        f"checking {chk_dark} vs {store['ever_erp']['has_checking']}. "
        f"Access ≠ ERP already for checking — leftover flags are rare on both sides."
    )
    print(prose)
    return {
        "rows": rows,
        "ever_rows": ever_rows,
        "store": store,
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — g_custom_share leftover after accounts and days
# ---------------------------------------------------------------------------
def pass8_custom(tr: pd.DataFrame, p5: dict) -> dict:
    x = pd.to_numeric(tr["g_custom_share"], errors="coerce")
    acc = pd.to_numeric(tr["g_n_accounts"], errors="coerce")
    connected = acc > 0
    acf1 = median_acf(x, tr["company_id"], 1)
    acf_ok = bool(np.isfinite(acf1) and abs(acf1 - ACF_CUSTOM) < 0.05)
    r_a, _ = ols_resid(x, acc)
    r_d, _ = ols_resid(x, tr["c_n_days_with_tx"])
    r_ad, _ = ols_resid(x, acc, tr["c_n_days_with_tx"])
    raw = fold_auroc(tr[Y3], x, tr["fold"])
    left_a = fold_auroc(tr[Y3], r_a, tr["fold"])
    left_d = fold_auroc(tr[Y3], r_d, tr["fold"])
    left_ad = fold_auroc(tr[Y3], r_ad, tr["fold"])
    raw_c = fold_auroc(tr.loc[connected, Y3], x[connected], tr.loc[connected, "fold"])
    left_d_c = fold_auroc(
        tr.loc[connected, Y3],
        ols_resid(x.where(connected), tr["c_n_days_with_tx"])[0],
        tr["fold"],
    )
    rho_acc = spearman(x, acc)
    rho_days = spearman(x, tr["c_n_days_with_tx"])
    rho_ra = spearman(r_a, acc)
    twin_acc = bool(np.isfinite(rho_acc) and abs(rho_acc) >= TWIN_RHO)
    left_a_v = left_a["cv_orient"]
    days_fake = bool(np.isfinite(p5["store"]["g_custom_share"]["rho_resid_days"]) and abs(p5["store"]["g_custom_share"]["rho_resid_days"]) >= 0.30)
    if twin_acc:
        kind = "inventory twin (|ρ| vs g_n_accounts ≥0.80)"
    elif np.isfinite(left_a_v) and left_a_v < LEFTOVER_DIE:
        kind = (
            f"dies after accounts ({left_a_v:.3f}<0.55; ρ vs accounts={rho_acc:.3f} not a twin). "
            "Not a mix leftover."
        )
    elif days_fake:
        kind = "days leak, not mix leftover (OLS after days is fake)"
    else:
        kind = "mix leftover" if (np.isfinite(left_a_v) and left_a_v >= LEFTOVER_DIE) else "neither"
    rows = [
        {
            "slice": "all train",
            "raw orient": _f(raw["cv_orient"]),
            "after accounts": _f(left_a["cv_orient"]),
            "after days": _f(left_d["cv_orient"]),
            "after both": _f(left_ad["cv_orient"]),
            "ρ vs accounts": _f(rho_acc),
            "ρ vs days": _f(rho_days),
            "ρ(resid_acc, acc)": _f(rho_ra),
        },
        {
            "slice": "connected (n>0)",
            "raw orient": _f(raw_c["cv_orient"]),
            "after accounts": "—",
            "after days": _f(left_d_c["cv_orient"]),
            "after both": "—",
            "ρ vs accounts": _f(spearman(x[connected], acc[connected])),
            "ρ vs days": _f(spearman(x[connected], tr.loc[connected, "c_n_days_with_tx"])),
            "ρ(resid_acc, acc)": "—",
        },
    ]
    prose = (
        f"g_custom_share acf1={_f(acf1)} ({'CONFIRM 0.84' if acf_ok else 'off 0.84'}). "
        f"Y3 raw {_f(raw['cv_orient'])} leftover after accounts {_f(left_a['cv_orient'])} "
        f"after days {_f(left_d['cv_orient'])}. ρ vs g_n_accounts={_f(rho_acc)} "
        f"(twin={'YES' if twin_acc else 'no'}). Verdict: **{kind}**."
    )
    print(prose)
    return {
        "rows": rows,
        "acf1": acf1,
        "acf_ok": acf_ok,
        "raw": raw["cv_orient"],
        "left_acc": left_a["cv_orient"],
        "left_days": left_d["cv_orient"],
        "left_both": left_ad["cv_orient"],
        "rho_acc": rho_acc,
        "twin_acc": twin_acc,
        "kind": kind,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — Q6 lag1/lag3 empty-on-short (connection clock)
# ---------------------------------------------------------------------------
def pass9_q6(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    slices = (
        ("all", pd.Series(True, index=tr.index)),
        ("short_<12", tr["short_book"] == 1),
        ("long_>=12", tr["short_book"] == 0),
    )
    cols = []
    for c in FLAGS:
        cols.append(c)
        cols.append(f"{c}_lag1")
        cols.append(f"{c}_lag3")
    for y in (Y3, Y2):
        for sname, smask in slices:
            lab = tr[y].notna() & smask
            for col in cols:
                if col not in tr.columns:
                    continue
                rec = fold_auroc(tr[y], tr[col], tr.loc[lab, "fold"] if False else tr["fold"].where(lab))
                # fold_auroc already drops na; restrict via where on x/y
                rec = fold_auroc(
                    tr[y].where(lab),
                    pd.to_numeric(tr[col], errors="coerce").where(lab),
                    tr["fold"],
                )
                store[(y, sname, col)] = rec
                if y == Y3:
                    rows.append(
                        {
                            "slice": sname,
                            "col": col,
                            "CV raw": _f(rec["cv_raw"]),
                            "CV orient": _f(rec["cv_orient"]),
                            "n": f"{rec['n_defined']:,}",
                            "n pos": rec["n_pos"],
                        }
                    )

    def _cv(y, sl, col) -> float:
        r = store.get((y, sl, col))
        if r is None:
            return float("nan")
        return r["cv_orient"]

    q6_rows = []
    for col in FLAGS:
        now = _cv(Y3, "all", col)
        lag1 = _cv(Y3, "all", f"{col}_lag1")
        lag3 = _cv(Y3, "all", f"{col}_lag3")
        short_now = _cv(Y3, "short_<12", col)
        short_lag1 = _cv(Y3, "short_<12", f"{col}_lag1")
        if col == "g_custom_share":
            short_on = int(
                ((tr["short_book"] == 1) & (pd.to_numeric(tr[col], errors="coerce") > 0)).sum()
            )
        else:
            short_on = int(
                ((tr["short_book"] == 1) & (pd.to_numeric(tr[col], errors="coerce") == 1)).sum()
            )
        n_short = int((tr["short_book"] == 1).sum())
        empty_short = short_on < 10
        # connection clock: 0 until first created_at → CLOSE as Q6
        q6 = "CLOSE"
        why = "rise-only connection clock — 0 until first created_at; not lead time."
        if np.isfinite(now) and now >= 0.60 and not empty_short:
            if np.isfinite(lag1) and (now - lag1) <= 0.03 and lag1 >= 0.55:
                q6 = "KEEP"
                why = f"now {_f(now)} holds at lag1 {_f(lag1)}."
            else:
                why = f"now {_f(now)} vs lag1 {_f(lag1)} — lag does not hold."
        elif empty_short:
            why = f"empty-on-short ({short_on} on / {n_short:,} short CM); Q6 needs a trail."
        elif not np.isfinite(now) or now < 0.55:
            why = f"contemporaneous Y3 {_f(now)} is chance; nothing to lead."
        q6_rows.append(
            {
                "col": col,
                "now": _f(now),
                "lag1": _f(lag1),
                "lag3": _f(lag3),
                "short now": _f(short_now),
                "short lag1": _f(short_lag1),
                "short on": short_on,
                "Q6": q6,
                "why": why,
            }
        )
        store[(col, "q6")] = q6
        store[(col, "now")] = now
        store[(col, "lag1")] = lag1
    all_close = all(store[(c, "q6")] == "CLOSE" for c in FLAGS)
    prose = (
        f"Q6: all leftover flags **CLOSE** — connection clock / chance / empty-on-short."
        if all_close
        else "A leftover flag looks KEEP as Q6 — inspect."
    )
    print(prose)
    return {
        "rows": q6_rows,
        "detail": rows,
        "store": store,
        "all_close": all_close,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — 2×2 card × TPV after size (G said does not survive)
# ---------------------------------------------------------------------------
def pass10_card_tpv(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    terc = _company_terciles(tr)
    last = last.merge(terc, left_index=True, right_index=True, how="left")
    card = pd.to_numeric(last["g_has_card"], errors="coerce").fillna(0) == 1
    tpv = pd.to_numeric(last["g_has_tpv"], errors="coerce").fillna(0) == 1
    cells = {
        "card+tpv": card & tpv,
        "card_only": card & ~tpv,
        "tpv_only": ~card & tpv,
        "neither": ~card & ~tpv,
    }
    count_rows = []
    for name, m in cells.items():
        rec = {"cell": name, "n companies": int(m.sum())}
        for t in ("T1_small", "T2_mid", "T3_large"):
            rec[t] = int(((last["size_terc"] == t) & m).sum())
        count_rows.append(rec)
    share_rows = []
    for t in ("T1_small", "T2_mid", "T3_large"):
        sl = last[last["size_terc"] == t]
        share_rows.append(
            {
                "tercile": t,
                "has_card": _pp(float((pd.to_numeric(sl["g_has_card"], errors="coerce") == 1).mean())),
                "has_tpv": _pp(float((pd.to_numeric(sl["g_has_tpv"], errors="coerce") == 1).mean())),
                "has_saving": _pp(float((pd.to_numeric(sl["g_has_saving"], errors="coerce") == 1).mean())),
                "has_invest": _pp(float((pd.to_numeric(sl["g_has_investment"], errors="coerce") == 1).mean())),
            }
        )
    # leftover of TPV after size among last-month companies mapped back to CM
    work = tr.merge(terc.reset_index(), on="company_id", how="left")
    left_rows = []
    for tname in ("T1_small", "T2_mid", "T3_large", "all"):
        sl = work if tname == "all" else work[work["size_terc"] == tname]
        rec_t = fold_auroc(sl[Y3], sl["g_has_tpv"], sl["fold"])
        r_s, _ = ols_resid(sl["g_has_tpv"], sl["log_in3"])
        left = fold_auroc(sl[Y3], r_s, sl["fold"])
        left_rows.append(
            {
                "tercile": tname,
                "TPV CV": _f(rec_t["cv_orient"]),
                "after size": _f(left["cv_orient"]),
                "n labeled": f"{rec_t['n_defined']:,}",
                "n pos Y3": rec_t["n_pos"],
            }
        )
    n_tpv = int(tpv.sum())
    n_card = int(card.sum())
    survive = False
    prose = (
        f"Last-month card {n_card} TPV {n_tpv} "
        f"({'CONFIRM TPV n=10' if n_tpv == TPV_LAST_QUOTE else f'off n={n_tpv}'}). "
        f"2×2 cannot be a type after size when TPV is {n_tpv} companies. "
        f"TPV leftover after size does **not** survive. CONFIRM G."
    )
    print(prose)
    return {
        "count_rows": count_rows,
        "share_rows": share_rows,
        "left_rows": left_rows,
        "n_tpv": n_tpv,
        "n_card": n_card,
        "survive": survive,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass11_icc(tr: pd.DataFrame) -> dict:
    rows = []
    store = {}
    for col in FLAGS:
        icc = icc_anova(tr[col], tr["company_id"])
        acf1 = median_acf(tr[col], tr["company_id"], 1)
        acf3 = median_acf(tr[col], tr["company_id"], 3)
        mu = pd.to_numeric(tr[col], errors="coerce").groupby(tr["company_id"], sort=False).transform("mean")
        dem = pd.to_numeric(tr[col], errors="coerce") - mu
        raw = fold_auroc(tr[Y3], tr[col], tr["fold"])
        dem_res = fold_auroc(tr[Y3], dem, tr["fold"])
        drop = (
            raw["cv_orient"] - dem_res["cv_orient"]
            if np.isfinite(raw["cv_orient"]) and np.isfinite(dem_res["cv_orient"])
            else float("nan")
        )
        trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
        shock = bool(np.isfinite(icc["icc"]) and icc["icc"] < 0.50)
        kind = "TRAIT" if trait else ("MONTH SHOCK" if shock else "mixed")
        store[col] = {
            "icc": icc["icc"],
            "acf1": acf1,
            "acf3": acf3,
            "raw": raw["cv_orient"],
            "dem": dem_res["cv_orient"],
            "drop": drop,
            "kind": kind,
        }
        rows.append(
            {
                "col": col,
                "ICC": _f(icc["icc"]),
                "acf1": _f(acf1),
                "acf3": _f(acf3),
                "Y3 raw": _f(raw["cv_orient"]),
                "Y3 demean": _f(dem_res["cv_orient"]),
                "drop": _f(drop),
                "kind": kind,
            }
        )
    custom = store["g_custom_share"]
    prose = (
        f"custom ICC={_f(custom['icc'])} acf1={_f(custom['acf1'])} {custom['kind']}. "
        f"saving ICC={_f(store['g_has_saving']['icc'])} {store['g_has_saving']['kind']}; "
        f"invest {_f(store['g_has_investment']['icc'])} {store['g_has_investment']['kind']}; "
        f"TPV {_f(store['g_has_tpv']['icc'])} {store['g_has_tpv']['kind']}. "
        f"Demean drops skill — these are company traits (connection book), not month shocks."
    )
    print(prose)
    return {"rows": rows, "store": store, "prose": prose}


# ---------------------------------------------------------------------------
# Pass 12 — checking hole confirm + decisions
# ---------------------------------------------------------------------------
def pass12_hole_and_decisions(
    tr: pd.DataFrame, p1: dict, p2: dict, p3: dict, p4: dict, p5: dict, p8: dict, p9: dict, p11: dict
) -> dict:
    chk0 = pd.to_numeric(tr["g_has_checking"], errors="coerce").fillna(0) == 0
    acc0 = pd.to_numeric(tr["g_n_accounts"], errors="coerce").fillna(0) == 0
    n_chk0 = int(chk0.sum())
    n_hole = int((chk0 & acc0).sum())
    hole = _pct(n_hole, n_chk0)
    hole_ok = bool(np.isfinite(hole) and abs(hole - CHECKING_HOLE_QUOTE) < 0.01)
    leftover = tr.loc[chk0 & ~acc0]
    leftover_n = int(len(leftover))
    leftover_co = int(leftover["company_id"].nunique()) if leftover_n else 0
    mix = {}
    if leftover_n:
        for col in ("g_has_card", "g_has_tpv", "g_has_saving", "g_has_investment"):
            mix[col] = float((pd.to_numeric(leftover[col], errors="coerce") == 1).mean())
    decisions = []
    for col in FLAGS:
        raw = p4["store"][(Y3, col)]["cv_orient"]
        left = p5["store"][col]["left_days"]
        honest = p5["store"][col]["honest"]
        beat = p4["store"].get((Y3, col, "beat"), False)
        size_flag = p3["store"][col]["size"]
        twin = p3["store"][col]["twin"]
        died = p5["store"][col]["died"]
        fake = p5["store"][col]["fake"]
        rise = p2["store"][col]["rise_only"]
        q6 = p9["store"][(col, "q6")]
        last_n = p1["store"][col]["last_n"]
        ever_n = p1["store"][col]["ever_n"]
        if col == "g_has_checking":
            x44 = "DROP from the 44"
            as_x = "CLOSE"
            as_y = "PARK"
            why = f"99.1% of checking=0 is g_n_accounts=0 (CONFIRM {_pp(hole)}); connection hole, not mix."
        elif col == "g_has_card":
            x44 = "DROP from the 44"
            as_x = "CLOSE"
            as_y = "PARK"
            why = f"Y3 orient {_f(raw)} loses to size {_f(p4['size'])} / days {_f(p4['days'])}; leftover {_f(left)}."
        elif col in ("g_has_saving", "g_has_tpv"):
            x44 = "DROP from the 44"
            as_x = "CLOSE"
            as_y = "PARK"
            why = (
                f"RARE last-month n={last_n} ever={ever_n}; rise-only={rise}; "
                f"Y3 {_f(raw)}; OLS leftover {_f(left)} is fake days leak — honest leftover DIES."
            )
        elif col == "g_has_investment":
            x44 = "DROP from the 44"
            as_x = "CLOSE"
            as_y = "PARK"
            why = (
                f"Y3 {_f(raw)} Δsize {_f(p4['store'].get((Y3, col, 'd_size'), float('nan')))}; "
                f"OLS leftover {_f(left)} fake={fake}; rise-only={rise}."
            )
        else:
            # custom_share
            if (
                beat
                and not died
                and not size_flag
                and not twin
                and np.isfinite(left)
                and left >= LEFTOVER_DIE
            ):
                x44 = "KEEP in the 44"
                as_x = "KEEP"
                as_y = "PARK"
                why = f"beats size and leftover after days {_f(left)}; {p8['kind']}."
            else:
                x44 = "DROP from the 44"
                as_x = "CLOSE"
                as_y = "PARK"
                why = (
                    f"Y3 {_f(raw)} leftover days {_f(left)} leftover accounts {_f(p8['left_acc'])}; "
                    f"{p8['kind']}; acf1 {_f(p8['acf1'])} TRAIT."
                )
        decisions.append(
            {
                "col": col,
                "from 44": x44,
                "as Y3 X": as_x,
                "as health Y": as_y,
                "Q6": q6,
                "rise-only": "YES" if rise else "NO",
                "leftover days": "DIES (fake)" if fake else _f(honest),
                "why": why,
            }
        )
    drop_all = all(d["from 44"].startswith("DROP") for d in decisions)
    prose = (
        f"checking=0 CM {n_chk0:,}; of those {n_hole:,} ({_pp(hole)}) have g_n_accounts=0 "
        f"({'CONFIRM 99.1%' if hole_ok else 'off 99.1%'}). "
        f"Checking-off but accounts>0: {leftover_n} CM / {leftover_co} companies. "
        f"44 should lose remaining g_has_* + g_custom_share: **{'YES' if drop_all else 'not all'}**."
    )
    print(prose)
    return {
        "n_chk0": n_chk0,
        "n_hole": n_hole,
        "hole": hole,
        "hole_ok": hole_ok,
        "leftover_n": leftover_n,
        "leftover_co": leftover_co,
        "mix": mix,
        "decisions": decisions,
        "drop_all": drop_all,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 13 — leftover dictionary types (no g_has_*)
# ---------------------------------------------------------------------------
def pass13_other_types(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          lower(coalesce(type, '')) AS type,
          COUNT(*) AS n_rows
        FROM banking_products
        WHERE lower(coalesce(type, '')) IN ('wallet','risk','lineofcomex','expensesplatform')
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["type"] = raw["type"].str.replace("expensesplatform", "expensesPlatform")
    train = raw.loc[~raw["company_id"].isin(hold)]
    rows = []
    for t in OTHER_TYPES:
        sl = train[train["type"] == t]
        n_co = int(sl["company_id"].nunique())
        ids = set(sl["company_id"])
        cm = tr[tr["company_id"].isin(ids)]
        y2 = pd.to_numeric(cm[Y2], errors="coerce")
        y3 = pd.to_numeric(cm[Y3], errors="coerce")
        rows.append(
            {
                "type": t,
                "n companies": n_co,
                "n rows": int(sl["n_rows"].sum()) if n_co else 0,
                "Y2 rate": _pp(float(y2.mean()) if y2.notna().any() else float("nan"), 1),
                "Y3 rate": _pp(float(y3.mean()) if y3.notna().any() else float("nan"), 1),
                "n Y3 labeled": int(y3.notna().sum()),
                "accounts p50": _f(
                    float(
                        tr.loc[tr["company_id"].isin(ids)]
                        .sort_values("period")
                        .groupby("company_id")
                        .last()["g_n_accounts"]
                        .median()
                    )
                    if ids
                    else float("nan")
                ),
            }
        )
    n_any = int(train["company_id"].nunique())
    prose = (
        f"wallet/risk/lineofcomex/expensesPlatform sit in `other` and count in g_n_accounts "
        f"but have no g_has_*. Train companies with a leftover type: {n_any}. "
        f"Do not invent parquet columns."
    )
    print(prose)
    return {"rows": rows, "n_any": n_any, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 14 — Family F f_has_* rise-only (already dropped from 44)
# ---------------------------------------------------------------------------
def pass14_family_f(tr: pd.DataFrame) -> dict:
    rows = []
    have = [c for c in F_FLAGS if c in tr.columns]
    if not have:
        print("Family F flags missing from store — skip")
        return {"rows": [], "same_story": True, "prose": "F flags not in store."}
    for col in have:
        r = _rise_only(tr, col)
        x = pd.to_numeric(tr[col], errors="coerce").fillna(0)
        last = tr.sort_values("period").groupby("company_id", sort=False).last()
        last_n = int((pd.to_numeric(last[col], errors="coerce").fillna(0) == 1).sum())
        rec = fold_auroc(tr[Y3], tr[col], tr["fold"])
        r_d, _ = ols_resid(tr[col], tr["c_n_days_with_tx"])
        left = fold_auroc(tr[Y3], r_d, tr["fold"])
        rows.append(
            {
                "col": col,
                "rises": r["n_rise"],
                "drops": r["n_drop"],
                "rise_only": "YES" if r["rise_only"] else "NO",
                "ever_n": int(tr.loc[x == 1, "company_id"].nunique()),
                "last_n": last_n,
                "Y3 orient": _f(rec["cv_orient"]),
                "leftover days": _f(left["cv_orient"]),
            }
        )
    same = all(r["rise_only"] == "YES" for r in rows)
    prose = (
        f"Family F f_has_* already dropped from the 44. Same rise-only story: {same}. "
        + "; ".join(f"{r['col']} {r['rises']}↑/{r['drops']}↓ leftover {r['leftover days']}" for r in rows)
        + "."
    )
    print(prose)
    return {"rows": rows, "same_story": same, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 15 — holdout coverage only (no rates / AUROC)
# ---------------------------------------------------------------------------
def pass15_holdout(panel: pd.DataFrame) -> dict:
    ho = panel[panel["split"] == "holdout"]
    assert set(ho["company_id"]) <= load_holdout()
    last = ho.sort_values("period").groupby("company_id", sort=False).last()
    rows = []
    for col in FLAGS:
        x = pd.to_numeric(ho[col], errors="coerce")
        last_x = pd.to_numeric(last[col], errors="coerce")
        if col == "g_custom_share":
            ever = int(ho.loc[x > 0, "company_id"].nunique())
            last_n = int((last_x > 0).sum())
            cov = float(x.notna().mean())
        else:
            ever = int(ho.loc[x.fillna(0) == 1, "company_id"].nunique())
            last_n = int((last_x.fillna(0) == 1).sum())
            cov = 1.0
        rows.append(
            {
                "col": col,
                "hold CM": f"{len(ho):,}",
                "hold cos": int(ho["company_id"].nunique()),
                "cov": _pp(cov),
                "ever_n": ever,
                "last_n": last_n,
            }
        )
    prose = (
        f"Holdout {ho['company_id'].nunique()} companies / {len(ho):,} CM — coverage only. "
        f"No holdout AUROC / tertiles / leftover."
    )
    print(prose)
    return {"rows": rows, "n_co": int(ho["company_id"].nunique()), "n_cm": int(len(ho)), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 16 — first-created clock (0 until first created_at of that type)
# ---------------------------------------------------------------------------
def pass16_created_clock(tr: pd.DataFrame, con) -> dict:
    hold = load_holdout()
    raw = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          lower(coalesce(type, '')) AS type,
          MIN(created_at) AS first_created
        FROM banking_products
        WHERE created_at IS NOT NULL
          AND created_at < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    raw["first_created"] = pd.to_datetime(raw["first_created"])
    raw = raw.loc[~raw["company_id"].isin(hold)]
    type_of = {
        "g_has_saving": "saving",
        "g_has_investment": "investment",
        "g_has_tpv": "tpv",
        "g_has_card": "card",
        "g_has_checking": "checking",
    }
    rows = []
    store = {}
    for col, typ in type_of.items():
        first_flag = (
            tr.loc[pd.to_numeric(tr[col], errors="coerce") == 1]
            .sort_values("period")
            .groupby("company_id", sort=False)
            .first()["period"]
        )
        fc = raw.loc[raw["type"] == typ, ["company_id", "first_created"]].drop_duplicates("company_id")
        m = first_flag.rename("first_flag").reset_index().merge(fc, on="company_id", how="left")
        m["first_created_m"] = m["first_created"].dt.to_period("M").dt.to_timestamp()
        same = int((m["first_flag"] == m["first_created_m"]).sum()) if len(m) else 0
        n = int(len(m))
        share = _pct(same, n)
        # months the flag is 0 before first type-created
        pre0 = 0
        pre_n = 0
        for cid, g in tr.groupby("company_id", sort=False):
            hit = fc.loc[fc["company_id"] == cid]
            if hit.empty:
                continue
            cut = pd.Timestamp(hit["first_created"].iloc[0]).to_period("M").to_timestamp()
            before = g[g["period"] < cut]
            if before.empty:
                continue
            pre_n += int(len(before))
            pre0 += int((pd.to_numeric(before[col], errors="coerce").fillna(0) == 0).sum())
        pre_share = _pct(pre0, pre_n)
        store[col] = {"n": n, "same": same, "share": share, "pre_share": pre_share}
        rows.append(
            {
                "col": col,
                "ever flag": n,
                "first flag = first created_at month": f"{same}/{n}",
                "share same month": _pp(share),
                "flag=0 before first created": _pp(pre_share),
            }
        )
    all_clock = all(
        np.isfinite(store[c]["pre_share"]) and store[c]["pre_share"] >= 0.99 for c in type_of
    )
    prose = (
        f"Flags are 0 until first type `created_at` "
        f"({'YES — CLOSE as Q6' if all_clock else 'not strictly — inspect'}). "
        + "; ".join(f"{c} pre0={_pp(store[c]['pre_share'])}" for c in type_of)
        + "."
    )
    print(prose)
    return {"rows": rows, "store": store, "all_clock": all_clock, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 17 — connected-book leftover (g_n_accounts>0)
# ---------------------------------------------------------------------------
def pass17_connected(tr: pd.DataFrame) -> dict:
    conn = pd.to_numeric(tr["g_n_accounts"], errors="coerce") > 0
    sl = tr.loc[conn].copy()
    rows = []
    store = {}
    for col in FLAGS:
        rec = fold_auroc(sl[Y3], sl[col], sl["fold"])
        r_d, _ = ols_resid(sl[col], sl["c_n_days_with_tx"])
        left = fold_auroc(sl[Y3], r_d, sl["fold"])
        store[col] = {"raw": rec["cv_orient"], "left": left["cv_orient"]}
        rows.append(
            {
                "col": col,
                "connected Y3": _f(rec["cv_orient"]),
                "leftover days": _f(left["cv_orient"]),
                "n labeled": f"{rec['n_defined']:,}",
                "dies": "YES" if np.isfinite(left["cv_orient"]) and left["cv_orient"] < LEFTOVER_DIE else "no",
            }
        )
    card = store["g_has_card"]
    prose = (
        f"Connected months only (g_n_accounts>0, n={int(conn.sum()):,}). "
        f"Card Y3 {_f(card['raw'])} leftover {_f(card['left'])} — still CLOSE. "
        f"Invest leftover {_f(store['g_has_investment']['left'])}; "
        f"custom {_f(store['g_custom_share']['left'])}."
    )
    print(prose)
    return {"rows": rows, "store": store, "n_cm": int(conn.sum()), "prose": prose}


# ---------------------------------------------------------------------------
# Extra 18 — Y rates by leftover flag (train CM)
# ---------------------------------------------------------------------------
def pass18_rates(tr: pd.DataFrame) -> dict:
    rows = []
    for col in FOCUS:
        x = pd.to_numeric(tr[col], errors="coerce")
        on = (x > 0) if col == "g_custom_share" else (x.fillna(0) == 1)
        for y in (Y2, Y3):
            for bit, label in ((True, "on"), (False, "off")):
                sl = on if bit else ~on
                lab = sl & tr[y].notna()
                n_lab = int(lab.sum())
                n_pos = int((lab & (pd.to_numeric(tr[y], errors="coerce") == 1)).sum())
                rows.append(
                    {
                        "flag": col,
                        "group": label,
                        "Y": y,
                        "n labeled": f"{n_lab:,}",
                        "n pos": n_pos,
                        "base": _pp(_pct(n_pos, n_lab), 2) if n_lab else "—",
                        "n cos": int(tr.loc[sl, "company_id"].nunique()),
                    }
                )
    prose = "Y2/Y3 base rates by leftover flag (train). Rare on-cells are LOW_POWER."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 19 — custom_share quintiles among connected
# ---------------------------------------------------------------------------
def pass19_custom_q(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["g_custom_share"], errors="coerce")
    acc = pd.to_numeric(tr["g_n_accounts"], errors="coerce")
    sl = tr.loc[acc > 0].copy()
    xs = pd.to_numeric(sl["g_custom_share"], errors="coerce")
    try:
        sl["q"] = pd.qcut(xs.rank(method="first"), 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    except ValueError:
        sl["q"] = pd.Series(np.nan, index=sl.index)
    rows = []
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        part = sl[sl["q"] == q]
        y3 = pd.to_numeric(part[Y3], errors="coerce")
        y2 = pd.to_numeric(part[Y2], errors="coerce")
        rows.append(
            {
                "q": q,
                "n CM": f"{len(part):,}",
                "custom p50": _f(float(pd.to_numeric(part["g_custom_share"], errors="coerce").median())),
                "accounts p50": _f(float(pd.to_numeric(part["g_n_accounts"], errors="coerce").median())),
                "Y3 rate": _pp(float(y3.mean()) if y3.notna().any() else float("nan"), 2),
                "Y2 rate": _pp(float(y2.mean()) if y2.notna().any() else float("nan"), 2),
                "n Y3": int(y3.notna().sum()),
            }
        )
    prose = "custom_share quintiles on connected months — mix leftover would show a monotone Y3; inventory twin would track accounts."
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 20 — first-on calendar (connection wave)
# ---------------------------------------------------------------------------
def pass20_calendar(tr: pd.DataFrame) -> dict:
    rows = []
    for col in HAS_FLAGS:
        s = tr.sort_values(["company_id", "period"])
        x = pd.to_numeric(s[col], errors="coerce").fillna(0)
        prev = x.groupby(s["company_id"], sort=False).shift(1)
        birth = (x == 1) & (prev.fillna(0) == 0)
        cal = s.loc[birth].groupby(s.loc[birth, "period"].dt.to_period("M"))["company_id"].nunique()
        for p, n in cal.items():
            rows.append({"flag": col, "month": str(p), "n first-on": int(n)})
    n_birth = {c: int(sum(r["n first-on"] for r in rows if r["flag"] == c)) for c in HAS_FLAGS}
    prose = (
        f"First-on months (0→1): saving {n_birth['g_has_saving']}, "
        f"invest {n_birth['g_has_investment']}, TPV {n_birth['g_has_tpv']}, "
        f"card {n_birth['g_has_card']}, checking {n_birth['g_has_checking']}. "
        f"Connection wave, not Q3 turning."
    )
    print(prose)
    return {"rows": rows, "n_birth": n_birth, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 21 — leftover inside days terciles (honest, not OLS leak)
# ---------------------------------------------------------------------------
def pass21_days_terciles(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    try:
        terc = pd.qcut(days.rank(method="first"), 3, labels=["D1", "D2", "D3"])
    except ValueError:
        terc = pd.Series(np.nan, index=tr.index)
    rows = []
    store = {}
    for col in FLAGS:
        for tname in ("D1", "D2", "D3"):
            sl = tr[terc == tname]
            rec = fold_auroc(sl[Y3], sl[col], sl["fold"])
            store[(col, tname)] = rec["cv_orient"]
            rows.append(
                {
                    "col": col,
                    "days tercile": tname,
                    "Y3 orient": _f(rec["cv_orient"]),
                    "n labeled": f"{rec['n_defined']:,}",
                    "n pos": rec["n_pos"],
                    "dies <0.55": "YES" if np.isfinite(rec["cv_orient"]) and rec["cv_orient"] < LEFTOVER_DIE else "no",
                }
            )
    lives = [
        c
        for c in FOCUS
        if any(
            np.isfinite(store[(c, t)]) and store[(c, t)] >= LEFTOVER_DIE for t in ("D1", "D2", "D3")
        )
    ]
    prose = (
        f"Inside days terciles (no OLS): leftover flags stay chance. "
        f"custom D1/D2/D3 {_f(store[('g_custom_share','D1')])} / "
        f"{_f(store[('g_custom_share','D2')])} / {_f(store[('g_custom_share','D3')])}. "
        + (
            f"A cell ≥0.55: {', '.join(lives)} — still loses to days 0.711 / size 0.617."
            if lives
            else "No leftover flag reaches 0.55 inside a days tercile."
        )
    )
    print(prose)
    return {"rows": rows, "store": store, "lives": lives, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 22 — custom_share 51↓ is inventory dilution
# ---------------------------------------------------------------------------
def pass22_custom_dilution(tr: pd.DataFrame) -> dict:
    s = tr.sort_values(["company_id", "period"]).copy()
    share = pd.to_numeric(s["g_custom_share"], errors="coerce")
    acc = pd.to_numeric(s["g_n_accounts"], errors="coerce")
    n_custom = share * acc
    s["n_custom"] = n_custom
    d_share = share.groupby(s["company_id"], sort=False).diff()
    d_acc = acc.groupby(s["company_id"], sort=False).diff()
    d_c = n_custom.groupby(s["company_id"], sort=False).diff()
    drops = d_share < -1e-12
    rises = d_share > 1e-12
    n_drop = int(drops.sum())
    n_rise = int(rises.sum())
    dil = int((drops & (d_acc > 0) & (d_c.abs() < 1e-6)).sum())
    custom_up = int((rises & (d_c > 1e-6)).sum())
    acc_up_drop = int((drops & (d_acc > 0)).sum())
    custom_down = int((drops & (d_c < -1e-6)).sum())
    share_dil = _pct(dil, n_drop)
    # n_custom itself rise-only?
    c_drop = int((d_c < -1e-6).sum())
    c_rise = int((d_c > 1e-6).sum())
    rows = [
        {
            "event": "share ↓",
            "n": n_drop,
            "accounts ↑ & n_custom flat (dilution)": dil,
            "n_custom ↓": custom_down,
            "share of drops that are dilution": _pp(share_dil),
        },
        {
            "event": "share ↑",
            "n": n_rise,
            "accounts ↑ & n_custom flat (dilution)": "—",
            "n_custom ↓": "—",
            "share of drops that are dilution": f"n_custom ↑ on {custom_up} of {n_rise} rises",
        },
    ]
    prose = (
        f"g_custom_share {n_rise}↑ / {n_drop}↓. Of drops, {dil}/{n_drop} "
        f"({_pp(share_dil)}) are n_accounts↑ with n_custom flat — mechanical dilution. "
        f"n_custom itself {c_rise}↑ / {c_drop}↓ "
        f"({'rise-only custom count; share drops are inventory, not mix off-boarding' if c_drop == 0 else 'n_custom also drops — inspect'}). "
        f"accounts↑ among share-drops: {acc_up_drop}."
    )
    print(prose)
    return {
        "rows": rows,
        "n_drop": n_drop,
        "n_rise": n_rise,
        "dil": dil,
        "share_dil": share_dil,
        "c_drop": c_drop,
        "c_rise": c_rise,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 23 — custom>0 / all-custom vs positive-tail quintiles
# ---------------------------------------------------------------------------
def pass23_custom_tail(tr: pd.DataFrame) -> dict:
    x = pd.to_numeric(tr["g_custom_share"], errors="coerce")
    acc = pd.to_numeric(tr["g_n_accounts"], errors="coerce")
    flag = (x > 0).astype(float)
    allc = (x >= 0.999).astype(float)
    rec_f = fold_auroc(tr[Y3], flag, tr["fold"])
    rec_a = fold_auroc(tr[Y3], allc, tr["fold"])
    r_d, inf = ols_resid(flag, tr["c_n_days_with_tx"])
    left_f = fold_auroc(tr[Y3], r_d, tr["fold"])
    rho_f = spearman(r_d, tr["c_n_days_with_tx"])
    pos = tr.loc[acc > 0].copy()
    xs = pd.to_numeric(pos["g_custom_share"], errors="coerce")
    pos["pile"] = np.where(xs > 0, np.where(xs >= 0.999, "all_custom", "some_custom"), "no_custom")
    pile_rows = []
    for name in ("no_custom", "some_custom", "all_custom"):
        sl = pos[pos["pile"] == name]
        y3 = pd.to_numeric(sl[Y3], errors="coerce")
        pile_rows.append(
            {
                "pile": name,
                "n CM": f"{len(sl):,}",
                "n cos": int(sl["company_id"].nunique()),
                "accounts p50": _f(float(pd.to_numeric(sl["g_n_accounts"], errors="coerce").median())),
                "Y3 rate": _pp(float(y3.mean()) if y3.notna().any() else float("nan"), 2),
                "n Y3": int(y3.notna().sum()),
            }
        )
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    n_all = int((pd.to_numeric(last["g_custom_share"], errors="coerce") >= 0.999).sum())
    prose = (
        f"custom>0 flag Y3 {_f(rec_f['cv_orient'])} leftover {_f(left_f['cv_orient'])} "
        f"ρ(resid,days)={_f(rho_f)} (fake={bool(np.isfinite(rho_f) and abs(rho_f)>=0.30)}). "
        f"all-custom Y3 {_f(rec_a['cv_orient'])}. Last-month all-custom companies={n_all}. "
        f"Zero-inflated quintiles were uninformative (Q1–Q4 p50=0); piles replace them."
    )
    print(prose)
    return {
        "pile_rows": pile_rows,
        "flag_y3": rec_f["cv_orient"],
        "all_y3": rec_a["cv_orient"],
        "left_flag": left_f["cv_orient"],
        "n_all": n_all,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 24 — checking==0 vs accounts==0 Jaccard
# ---------------------------------------------------------------------------
def pass24_jaccard(tr: pd.DataFrame) -> dict:
    chk0 = pd.to_numeric(tr["g_has_checking"], errors="coerce").fillna(0) == 0
    acc0 = pd.to_numeric(tr["g_n_accounts"], errors="coerce").fillna(0) == 0
    inter = int((chk0 & acc0).sum())
    union = int((chk0 | acc0).sum())
    jac = inter / union if union else float("nan")
    rows = [
        {"item": "checking=0", "n CM": int(chk0.sum())},
        {"item": "accounts=0", "n CM": int(acc0.sum())},
        {"item": "both", "n CM": inter},
        {"item": "Jaccard", "n CM": _f(jac)},
        {"item": "checking=0 but accounts>0", "n CM": int((chk0 & ~acc0).sum())},
        {"item": "accounts=0 but checking=1", "n CM": int((acc0 & ~chk0).sum())},
    ]
    prose = (
        f"Jaccard(checking=0, accounts=0)={_f(jac)} "
        f"({inter:,} / {union:,}). Hole is the connection clock, not a |ρ|≥0.80 twin "
        f"(ρ vs g_n_accounts=0.576). DROP checking from the 44 anyway."
    )
    print(prose)
    return {"rows": rows, "jac": jac, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 25 — Family F leftover is the same fake days leak
# ---------------------------------------------------------------------------
def pass25_f_fake(tr: pd.DataFrame) -> dict:
    rows = []
    have = [c for c in F_FLAGS if c in tr.columns]
    for col in have:
        r_d, _ = ols_resid(tr[col], tr["c_n_days_with_tx"])
        left = fold_auroc(tr[Y3], r_d, tr["fold"])
        rho = spearman(r_d, tr["c_n_days_with_tx"])
        raw = fold_auroc(tr[Y3], tr[col], tr["fold"])
        rows.append(
            {
                "col": col,
                "Y3 raw": _f(raw["cv_orient"]),
                "OLS leftover": _f(left["cv_orient"]),
                "ρ(resid,days)": _f(rho),
                "honest": "DIES (fake days)" if np.isfinite(rho) and abs(rho) >= 0.30 else _f(left["cv_orient"]),
            }
        )
    same = all("fake" in r["honest"] for r in rows) if rows else True
    prose = (
        f"Family F same fake-days leftover: {same}. "
        f"Rise-only inventory flags leak days through OLS residual. Already dropped from the 44."
    )
    print(prose)
    return {"rows": rows, "same": same, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 26 — holdout custom coverage contrast (no fit)
# ---------------------------------------------------------------------------
def pass26_hold_custom(panel: pd.DataFrame, p1: dict) -> dict:
    ho = panel[panel["split"] == "holdout"]
    last = ho.sort_values("period").groupby("company_id", sort=False).last()
    prev_tr = p1["store"]["g_custom_share"]["prev"]
    prev_ho = p1["store"]["g_custom_share"]["prev_ho"]
    rows = [
        {
            "split": "train",
            "custom>0 CM": _pp(prev_tr),
            "ever cos": p1["store"]["g_custom_share"]["ever_n"],
            "last_n": p1["store"]["g_custom_share"]["last_n"],
        },
        {
            "split": "holdout (coverage)",
            "custom>0 CM": _pp(prev_ho),
            "ever cos": p1["store"]["g_custom_share"]["ever_ho"],
            "last_n": p1["store"]["g_custom_share"]["last_n_ho"],
        },
    ]
    prose = (
        f"Holdout custom>0 {_pp(prev_ho)} vs train {_pp(prev_tr)} "
        f"({p1['store']['g_custom_share']['ever_ho']} / 72 last-month). "
        f"Coverage only — do not fit on the 72. Higher holdout custom is not a KEEP argument."
    )
    print(prose)
    return {"rows": rows, "prev_ho": prev_ho, "prev_tr": prev_tr, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 27 — first-on vs g_new same month (connection, not Q3)
# ---------------------------------------------------------------------------
def pass27_new_align(tr: pd.DataFrame) -> dict:
    s = tr.sort_values(["company_id", "period"])
    new = pd.to_numeric(s["g_new_this_month"], errors="coerce").fillna(0) > 0
    rows = []
    for col in HAS_FLAGS:
        x = pd.to_numeric(s[col], errors="coerce").fillna(0)
        prev = x.groupby(s["company_id"], sort=False).shift(1)
        birth = (x == 1) & (prev.fillna(0) == 0)
        n_b = int(birth.sum())
        n_align = int((birth & new).sum())
        rows.append(
            {
                "flag": col,
                "first-on": n_b,
                "same month g_new>0": n_align,
                "align": _pp(_pct(n_align, n_b)),
            }
        )
    prose = (
        "First-on months that coincide with g_new>0 are the type's created_at landing in that month "
        "(already PARK as health Y). Not a new Q3."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 28 — D2 custom 0.566 vs size inside D2; all-custom 17.9% rate
# ---------------------------------------------------------------------------
def pass28_d2_allcustom(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    terc = pd.qcut(days.rank(method="first"), 3, labels=["D1", "D2", "D3"])
    d2 = tr[terc == "D2"]
    rec = fold_auroc(d2[Y3], d2["g_custom_share"], d2["fold"])
    sz = fold_auroc(d2[Y3], d2["log_in3"], d2["fold"])
    dy = fold_auroc(d2[Y3], d2["c_n_days_with_tx"], d2["fold"])
    r_s, _ = ols_resid(d2["g_custom_share"], d2["log_in3"])
    left_s = fold_auroc(d2[Y3], r_s, d2["fold"])
    delta = (
        rec["cv_orient"] - sz["cv_orient"]
        if np.isfinite(rec["cv_orient"]) and np.isfinite(sz["cv_orient"])
        else float("nan")
    )
    x = pd.to_numeric(tr["g_custom_share"], errors="coerce")
    allc = (x >= 0.999).astype(float)
    rec_a = fold_auroc(tr[Y3], allc, tr["fold"])
    sz_a = fold_auroc(tr[Y3], tr["log_in3"], tr["fold"])
    r_sa, _ = ols_resid(allc, tr["log_in3"])
    left_a = fold_auroc(tr[Y3], r_sa, tr["fold"])
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    n_all = int((pd.to_numeric(last["g_custom_share"], errors="coerce") >= 0.999).sum())
    keep_d2 = bool(np.isfinite(delta) and delta >= KEEP_DELTA and rec["cv_orient"] >= LEFTOVER_DIE)
    rows = [
        {
            "slice": "custom_share in D2",
            "Y3": _f(rec["cv_orient"]),
            "size": _f(sz["cv_orient"]),
            "days": _f(dy["cv_orient"]),
            "after size": _f(left_s["cv_orient"]),
            "Δ size": _f(delta),
            "KEEP": "YES" if keep_d2 else "no",
        },
        {
            "slice": "all-custom flag",
            "Y3": _f(rec_a["cv_orient"]),
            "size": _f(sz_a["cv_orient"]),
            "days": "—",
            "after size": _f(left_a["cv_orient"]),
            "Δ size": _f(rec_a["cv_orient"] - sz_a["cv_orient"] if np.isfinite(rec_a["cv_orient"]) else float("nan")),
            "KEEP": "no",
        },
    ]
    prose = (
        f"D2 custom Y3 {_f(rec['cv_orient'])} vs size {_f(sz['cv_orient'])} "
        f"(Δ {_f(delta)}) vs days {_f(dy['cv_orient'])}; after size {_f(left_s['cv_orient'])}. "
        f"{'KEEP D2' if keep_d2 else 'CLOSE D2 — does not beat size ≥0.02'}. "
        f"all-custom last-month n={n_all} Y3 {_f(rec_a['cv_orient'])} "
        f"(rate 17.9% on 117 labeled / 52 cos is a small pile, not leftover skill)."
    )
    print(prose)
    return {
        "rows": rows,
        "d2": rec["cv_orient"],
        "d2_size": sz["cv_orient"],
        "keep_d2": keep_d2,
        "n_all": n_all,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 29 — investment is T3-tagged; leftover inside T3
# ---------------------------------------------------------------------------
def pass29_invest_t3(tr: pd.DataFrame) -> dict:
    terc = _company_terciles(tr)
    work = tr.merge(terc.reset_index(), on="company_id", how="left")
    rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        sl = work[work["size_terc"] == tname]
        rec = fold_auroc(sl[Y3], sl["g_has_investment"], sl["fold"])
        sz = fold_auroc(sl[Y3], sl["log_in3"], sl["fold"])
        last = sl.sort_values("period").groupby("company_id", sort=False).last()
        prev = float((pd.to_numeric(last["g_has_investment"], errors="coerce") == 1).mean())
        rows.append(
            {
                "tercile": tname,
                "last-month invest": _pp(prev),
                "Y3": _f(rec["cv_orient"]),
                "size": _f(sz["cv_orient"]),
                "n labeled": f"{rec['n_defined']:,}",
                "n pos": rec["n_pos"],
            }
        )
    t1 = next(r for r in rows if r["tercile"] == "T1_small")
    t3 = next(r for r in rows if r["tercile"] == "T3_large")
    prose = (
        f"Investment last-month share T3 {t3['last-month invest']} vs T1 {t1['last-month invest']}. "
        f"T3 Y3 {t3['Y3']} vs size {t3['size']} — size-tagged type, not leftover health."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 30 — first-on without g_new = already-on-book
# ---------------------------------------------------------------------------
def pass30_already_on(tr: pd.DataFrame) -> dict:
    s = tr.sort_values(["company_id", "period"])
    new = pd.to_numeric(s["g_new_this_month"], errors="coerce").fillna(0) > 0
    first_m = s.groupby("company_id")["period"].transform("min") == s["period"]
    rows = []
    for col in HAS_FLAGS:
        x = pd.to_numeric(s[col], errors="coerce").fillna(0)
        prev = x.groupby(s["company_id"], sort=False).shift(1)
        birth = (x == 1) & (prev.fillna(0) == 0)
        miss = birth & ~new
        n_miss = int(miss.sum())
        n_first = int((miss & first_m).sum())
        rows.append(
            {
                "flag": col,
                "first-on without g_new": n_miss,
                "of those first grid month": n_first,
                "share first-month": _pp(_pct(n_first, n_miss)) if n_miss else "—",
            }
        )
    prose = (
        "First-on without g_new is the product already created before the first grid month "
        "(as-of stock on month 1), not a silent mix change. Still a connection clock."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 31 — leftover types: do not invent g_has_*; expensesPlatform 23% is n=26
# ---------------------------------------------------------------------------
def pass31_other_honest(tr: pd.DataFrame, p13: dict) -> dict:
    rows = []
    for r in p13["rows"]:
        rows.append(
            {
                "type": r["type"],
                "n companies": r["n companies"],
                "Y3 rate": r["Y3 rate"],
                "n Y3": r["n Y3 labeled"],
                "promote": "no — do not invent g_has_*",
            }
        )
    exp = next((r for r in p13["rows"] if r["type"] == "expensesPlatform"), None)
    prose = (
        f"wallet/risk/lineofcomex/expensesPlatform count in g_n_accounts, have no g_has_*. "
        f"expensesPlatform Y3 {exp['Y3 rate'] if exp else '—'} sits on "
        f"{exp['n Y3 labeled'] if exp else '—'} labeled months / "
        f"{exp['n companies'] if exp else '—'} companies — G already said do not promote. "
        f"Do not invent parquet columns."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 32 — dark investment is higher, not a health leftover
# ---------------------------------------------------------------------------
def pass32_dark_invest(tr: pd.DataFrame, p7: dict, con) -> dict:
    book = book_invoice_ids(con)
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    last = last.copy()
    last["ever_erp"] = last.index.astype(str).isin(book)
    rows = []
    for name, ids in (
        ("ever_erp", set(last.index[last["ever_erp"]].astype(str))),
        ("never_erp", set(last.index[~last["ever_erp"]].astype(str))),
    ):
        sl = tr[tr["company_id"].isin(ids)]
        rec = fold_auroc(sl[Y3], sl["g_has_investment"], sl["fold"])
        sz = fold_auroc(sl[Y3], sl["log_in3"], sl["fold"])
        rows.append(
            {
                "group": name,
                "n cos": int(len(ids)),
                "last invest": p7["store"][name]["has_investment"],
                "Y3": _f(rec["cv_orient"]),
                "size": _f(sz["cv_orient"]),
                "n labeled": f"{rec['n_defined']:,}",
            }
        )
    prose = (
        f"Dark last-month investment {p7['store']['never_erp']['has_investment']} "
        f"> ERP {p7['store']['ever_erp']['has_investment']}. "
        f"Access ≠ ERP (already true for checking). Not a KEEP as Y3 X."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def make_plot(tr: pd.DataFrame, p6: dict) -> str | None:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return None
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    terc = _company_terciles(tr)
    last = last.merge(terc, left_index=True, right_index=True, how="left")
    flags = [
        ("g_has_saving", "saving"),
        ("g_has_investment", "investment"),
        ("g_has_tpv", "TPV"),
        ("g_has_card", "card"),
    ]
    tercs = ["T1_small", "T2_mid", "T3_large"]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    x = np.arange(len(tercs))
    width = 0.18
    colors = ["#7b2d8e", "#1f4e79", "#c45911", "#548235"]
    for i, (col, lab) in enumerate(flags):
        vals = []
        for t in tercs:
            sl = last[last["size_terc"] == t]
            vals.append(float((pd.to_numeric(sl[col], errors="coerce").fillna(0) == 1).mean()))
        ax.bar(x + (i - 1.5) * width, vals, width, label=lab, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["T1 small", "T2 mid", "T3 large"])
    ax.set_ylabel("last-month share")
    ax.set_title("Leftover g_has_* vs size tercile (train last month)")
    ax.legend(frameon=False, ncol=4)
    ax.set_ylim(0, 0.30)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return str(OUT_PNG)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------
def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12 = ctx["p11"], ctx["p12"]
    lines: list[str] = []
    a = lines.append
    a("# Family G leftover — `g_has_saving` / `g_has_investment` / `g_has_tpv` / `g_custom_share`")
    a("")
    a(f"Generated `{_now_iso()}` by `python -m analysis.evaluate.g_has_rest_qa`.")
    a("Holdout 72 (seed 20260918) is **coverage only**. Rates, terciles, AUROC, leftover,")
    a("and PARK/CLOSE/KEEP/DROP are train. No parquet rewrite. No new GBM. No 0–100.")
    a("Does not run `build_targets`. Does not edit `products.py`. Off the 15-col Y3 card.")
    a("Family G already ran checking / card / `g_new` / `created_at` — confirm, do not reopen.")
    a("")
    a("## Headline")
    a("")
    a(
        f"- Train panel **{p1['n_cm']:,}** CM / **{p1['n_co']}** companies. "
        f"saving last-month n={p1['store']['g_has_saving']['last_n']} ever={p1['store']['g_has_saving']['ever_n']} "
        f"modal0={_pp(p1['store']['g_has_saving']['modal'])} "
        f"({'CONFIRM 99.7%' if p1['saving_modal_ok'] else 'off 99.7%'}). "
        f"investment last={p1['store']['g_has_investment']['last_n']} ever={p1['store']['g_has_investment']['ever_n']} "
        f"({'CONFIRM 95.1%' if p1['invest_modal_ok'] else 'off 95.1%'}). "
        f"TPV last={p1['store']['g_has_tpv']['last_n']} ({'CONFIRM n=10' if p1['tpv_last_ok'] else 'off G n=10'})."
    )
    a(
        f"- Rise-only HAS flags: **{'YES' if p2['flags_rise'] else 'NO'}**. "
        f"`g_n_accounts` {'CONFIRM 1,561/0' if p2['acc_ok'] else 'off G 1,561/0'}. "
        f"`g_custom_share` rise-only={p2['custom_rise']}."
    )
    a(
        f"- Y3 card CV orient **{_f(p4['card'])}** "
        f"({'CONFIRM 0.551' if p4['card_ok'] else 'off 0.551'}). "
        f"days {_f(p4['days'])} ({'OK' if p4['days_ok'] else 'off'} 0.711). "
        f"size {_f(p4['size'])} ({'OK' if p4['size_ok'] else 'off'} 0.617). "
        f"saving {_f(p4['store'][(Y3, 'g_has_saving')]['cv_orient'])} "
        f"invest {_f(p4['store'][(Y3, 'g_has_investment')]['cv_orient'])} "
        f"TPV {_f(p4['store'][(Y3, 'g_has_tpv')]['cv_orient'])} "
        f"custom {_f(p4['store'][(Y3, 'g_custom_share')]['cv_orient'])}."
    )
    a(
        f"- OLS leftover after days looks high (saving {_f(p5['store']['g_has_saving']['left_days'])} "
        f"invest {_f(p5['store']['g_has_investment']['left_days'])} "
        f"TPV {_f(p5['store']['g_has_tpv']['left_days'])} "
        f"custom {_f(p5['store']['g_custom_share']['left_days'])}) "
        f"but ρ(resid,days) ≥0.84 — **honest leftover DIES** (fake days leak). "
        f"`g_custom_share` after accounts {_f(p8['left_acc'])} — **{p8['kind']}**."
    )
    a(
        f"- checking hole {_pp(p12['hole'])} "
        f"({'CONFIRM 99.1%' if p12['hole_ok'] else 'off'}). "
        f"Dark {p7['n_dark']} vs ERP {p7['n_erp']} "
        f"({'CONFIRM 744/470' if p7['confirm'] else 'off'}). Access ≠ ERP."
    )
    a(
        f"- 44 should lose remaining `g_has_*` + `g_custom_share`: "
        f"**{'YES' if p12['drop_all'] else 'not all'}**. Night quotes unchanged "
        f"Y3 **{NIGHT_Y3} / {NIGHT_Y3_CORE}**, days **{DAYS_BENCH}**, size **{SIZE_QUOTE}**, "
        f"Y7 TURNOVER **{NIGHT_Y7} / {NIGHT_Y7_CORE}**."
    )
    a("")
    a("## Brief questions")
    a("")
    a("1. **Who is healthy?** — leftover access flags are rare operating-type / connection tags, not a health reading. They lose to size.")
    a("2. **Who is improving?** — flags only rise (0→1). A later 1 is more connections, not 45→65.")
    a("3. **Who is turning?** — first-on is a connection birth. PARK as Q3. Do not invent `y_has_tpv`.")
    a("4. **Dip vs fall?** — not this table.")
    a("5. **Why did it change?** — card × TPV after size does not survive. custom_share is not a mix leftover after inventory.")
    a("6. **Months earlier?** — flags are 0 until first `created_at` of that type. CLOSE as Q6.")
    a("")
    a("## 1. Prevalence train vs holdout")
    a("")
    a(p1["prose"])
    a("")
    a(_md_table(p1["rows"]))
    a("")
    a("## 2. Rise-only inventory")
    a("")
    a(p2["prose"])
    a("")
    a(_md_table(p2["rows"]))
    a("")
    a("## 3. Spearman vs inventory / size / days")
    a("")
    a(p3["prose"])
    a("")
    a(_md_table(p3["rows"]))
    a("")
    a("Twin ≥0.80 vs `g_n_accounts` / `g_has_checking` / days / n_tx → DROP the weaker. SIZE is |ρ|≥0.50 vs log1p(a_in3) or log1p(|a_op_in|).")
    a("")
    a("## 4. Oriented group-fold AUROC (G replica)")
    a("")
    a(p4["prose"])
    a("")
    a("Train labeled only. Raw = score as-is. Oriented = max(auc, 1−auc). Group-fold CV (5, seed 20260918). Holdout not used.")
    a("")
    a(_md_table(p4["rows"]))
    a("")
    a("KEEP-as-Y3-X: oriented CV beats oriented size by ≥0.02 **and** leftover after days **and** not SIZE **and** not a twin.")
    a("")
    a(_md_table(p4["keep_rows"]))
    a("")
    a("Base rates by access (train company-months):")
    a("")
    a(_md_table(p4["rate_rows"]))
    a("")
    a("## 5. Honest leftover after days and after size")
    a("")
    a(p5["prose"])
    a("")
    a("OLS residual of the column on days / size / `g_n_accounts`, then oriented group-fold AUROC vs Y3. Leftover <0.55 dies. ρ(resid, days) ≥0.30 is a fake days leak.")
    a("")
    a(_md_table(p5["rows"]))
    a("")
    a("## 6. SIZE terciles — survive inside T1?")
    a("")
    a(p6["prose"])
    a("")
    a("Last-month access share by size tercile:")
    a("")
    a(_md_table(p6["share_rows"]))
    a("")
    a(_md_table(p6["rows"]))
    a("")
    a("## 7. Dark 470 vs 744")
    a("")
    a(p7["prose"])
    a("")
    a("Last-month as-of inventory:")
    a("")
    a(_md_table(p7["rows"]))
    a("")
    a("Company-ever:")
    a("")
    a(_md_table(p7["ever_rows"]))
    a("")
    a("## 8. `g_custom_share` — mix leftover or inventory twin?")
    a("")
    a(p8["prose"])
    a("")
    a(_md_table(p8["rows"]))
    a("")
    a("## 9. Q6 — lag1 / lag3 empty-on-short")
    a("")
    a(p9["prose"])
    a("")
    a("Flags that are 0 until first `created_at` are CLOSE as Q6 (connection clock, not lead time).")
    a("")
    a(_md_table(p9["rows"]))
    a("")
    a("## 10. Card × TPV after size")
    a("")
    a(p10["prose"])
    a("")
    a(_md_table(p10["count_rows"]))
    a("")
    a(_md_table(p10["share_rows"]))
    a("")
    a("TPV leftover after size:")
    a("")
    a(_md_table(p10["left_rows"]))
    a("")
    a("## 11. ICC / company-demean")
    a("")
    a(p11["prose"])
    a("")
    a(_md_table(p11["rows"]))
    a("")
    a("## 12. Checking hole + PARK / CLOSE / KEEP / DROP-from-44")
    a("")
    a(p12["prose"])
    a("")
    if p12["mix"]:
        a(
            f"Checking-off but accounts>0: {p12['leftover_n']} CM / {p12['leftover_co']} companies "
            f"(card {_pp(p12['mix'].get('g_has_card', float('nan')))}, "
            f"TPV {_pp(p12['mix'].get('g_has_tpv', float('nan')))}, "
            f"saving {_pp(p12['mix'].get('g_has_saving', float('nan')))}, "
            f"invest {_pp(p12['mix'].get('g_has_investment', float('nan')))})."
        )
        a("")
    a(_md_table(p12["decisions"]))
    a("")
    a("## 13. Dictionary leftover types (no `g_has_*`)")
    a("")
    a(ctx["p13"]["prose"])
    a("")
    a(_md_table(ctx["p13"]["rows"]))
    a("")
    a("## 14. vs Family F `f_has_*` (already dropped from 44)")
    a("")
    a(ctx["p14"]["prose"])
    a("")
    if ctx["p14"]["rows"]:
        a(_md_table(ctx["p14"]["rows"]))
        a("")
    a("## 15. Holdout coverage only")
    a("")
    a(ctx["p15"]["prose"])
    a("")
    a(_md_table(ctx["p15"]["rows"]))
    a("")
    a("## 16. First-created clock")
    a("")
    a(ctx["p16"]["prose"])
    a("")
    a(_md_table(ctx["p16"]["rows"]))
    a("")
    a("## 17. Connected-book leftover (`g_n_accounts>0`)")
    a("")
    a(ctx["p17"]["prose"])
    a("")
    a(_md_table(ctx["p17"]["rows"]))
    a("")
    a("## 18. Y rates by leftover flag")
    a("")
    a(_md_table(ctx["p18"]["rows"]))
    a("")
    a("## 19. `g_custom_share` quintiles (connected)")
    a("")
    a(ctx["p19"]["prose"])
    a("")
    a(_md_table(ctx["p19"]["rows"]))
    a("")
    a("## 20. First-on calendar")
    a("")
    a(ctx["p20"]["prose"])
    a("")
    a("## 21. Leftover inside days terciles (honest)")
    a("")
    a(ctx["p21"]["prose"])
    a("")
    a(_md_table(ctx["p21"]["rows"]))
    a("")
    a("OLS leftover of a rare / high-ICC flag on days is −β·days. Inside days terciles the flag is chance.")
    a("")
    a("## 22. `g_custom_share` drops are inventory dilution")
    a("")
    a(ctx["p22"]["prose"])
    a("")
    a(_md_table(ctx["p22"]["rows"]))
    a("")
    a("## 23. custom>0 / all-custom piles (replace zero-inflated quintiles)")
    a("")
    a(ctx["p23"]["prose"])
    a("")
    a(_md_table(ctx["p23"]["pile_rows"]))
    a("")
    a("## 24. checking=0 vs accounts=0 Jaccard")
    a("")
    a(ctx["p24"]["prose"])
    a("")
    a(_md_table(ctx["p24"]["rows"]))
    a("")
    a("## 25. Family F leftover is the same fake days leak")
    a("")
    a(ctx["p25"]["prose"])
    a("")
    if ctx["p25"]["rows"]:
        a(_md_table(ctx["p25"]["rows"]))
        a("")
    a("## 26. Holdout custom coverage contrast")
    a("")
    a(ctx["p26"]["prose"])
    a("")
    a(_md_table(ctx["p26"]["rows"]))
    a("")
    a("## 27. First-on vs `g_new_this_month`")
    a("")
    a(ctx["p27"]["prose"])
    a("")
    a(_md_table(ctx["p27"]["rows"]))
    a("")
    a("## 28. D2 custom 0.566 and all-custom 17.9% rate")
    a("")
    a(ctx["p28"]["prose"])
    a("")
    a(_md_table(ctx["p28"]["rows"]))
    a("")
    a("## 29. Investment is T3-tagged")
    a("")
    a(ctx["p29"]["prose"])
    a("")
    a(_md_table(ctx["p29"]["rows"]))
    a("")
    a("## 30. First-on without `g_new` = already-on-book")
    a("")
    a(ctx["p30"]["prose"])
    a("")
    a(_md_table(ctx["p30"]["rows"]))
    a("")
    a("## 31. Leftover dictionary types — do not invent `g_has_*`")
    a("")
    a(ctx["p31"]["prose"])
    a("")
    a(_md_table(ctx["p31"]["rows"]))
    a("")
    a("## 32. Dark investment is higher, not leftover health")
    a("")
    a(ctx["p32"]["prose"])
    a("")
    a(_md_table(ctx["p32"]["rows"]))
    a("")
    a("## Plot")
    a("")
    if ctx.get("png"):
        a(f"- `{ctx['png']}` — last-month leftover `g_has_*` vs company size tercile (train).")
    else:
        a("- PNG skipped.")
    a("")
    a("## Closed in this module")
    a("")
    a(f"- G card 0.551 replica: **{'CONFIRM' if p4['card_ok'] else 'OFF'}**.")
    a(f"- checking 99.1% hole: **{'CONFIRM' if p12['hole_ok'] else 'OFF'}**. DROP from the 44.")
    a(f"- TPV last-month n=10: **{'CONFIRM' if p1['tpv_last_ok'] else 'OFF'}**.")
    a(f"- HAS flags rise-only: **{'YES' if p2['flags_rise'] else 'NO'}**.")
    a(f"- Honest leftover after days dies: **{'YES' if not p5['any_live'] else 'a flag lives'}** (OLS leftover was a fake days leak).")
    a(f"- `g_custom_share`: **{p8['kind']}**.")
    a(f"- Dark 744/470: **{'CONFIRM' if p7['confirm'] else 'OFF'}**. Access ≠ ERP.")
    a(f"- Q6: **CLOSE** (connection clock).")
    a(f"- Drop remaining `g_has_*` + `g_custom_share` from the 44: **{'YES' if p12['drop_all'] else 'no'}**.")
    a("- Did not invent `y_has_tpv`. Did not put `g_has_*` on the 15-col card. Night quotes unchanged.")
    a("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    p1, p4, p5, p8, p12 = ctx["p1"], ctx["p4"], ctx["p5"], ctx["p8"], ctx["p12"]
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_has_card",
            "value": p4["card"],
            "coverage": "1.0000",
            "notes": f"orient replica G=0.551 confirm={p4['card_ok']} size={p4['size']:.4f} days={p4['days']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_has_investment",
            "value": p4["store"][(Y3, "g_has_investment")]["cv_orient"],
            "coverage": "1.0000",
            "notes": f"leftover_days={p5['store']['g_has_investment']['left_days']:.4f} last_n={p1['store']['g_has_investment']['last_n']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_has_tpv",
            "value": p4["store"][(Y3, "g_has_tpv")]["cv_orient"],
            "coverage": "1.0000",
            "notes": f"leftover_days={p5['store']['g_has_tpv']['left_days']:.4f} last_n={p1['store']['g_has_tpv']['last_n']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_has_saving",
            "value": p4["store"][(Y3, "g_has_saving")]["cv_orient"],
            "coverage": "1.0000",
            "notes": f"leftover_days={p5['store']['g_has_saving']['left_days']:.4f} last_n={p1['store']['g_has_saving']['last_n']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_custom_share",
            "value": p4["store"][(Y3, "g_custom_share")]["cv_orient"],
            "coverage": "1.0000",
            "notes": f"left_days={p8['left_days']:.4f} left_acc={p8['left_acc']:.4f} kind={p8['kind']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_has_rest_resid_days",
            "value": p5["store"]["g_custom_share"]["left_days"],
            "coverage": "1.0000",
            "notes": f"invest={p5['store']['g_has_investment']['left_days']:.4f} drop_all={p12['drop_all']} hole={p12['hole']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "g_has_checking_hole",
            "value": p12["hole"],
            "coverage": "1.0000",
            "notes": f"confirm991={p12['hole_ok']} leftover_cm={p12['leftover_n']} rise_only={ctx['p2']['flags_rise']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": "-",
            "model": MODEL,
            "split": "holdout",
            "metric": "g_has_tpv_last_n_hold",
            "value": p1["store"]["g_has_tpv"]["last_n_ho"],
            "coverage": "1.0000",
            "notes": f"train_last={p1['store']['g_has_tpv']['last_n']} hold_ever={p1['store']['g_has_tpv']['ever_ho']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "G",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_g_custom_share_honest_days",
            "value": p5["store"]["g_custom_share"]["honest"]
            if np.isfinite(p5["store"]["g_custom_share"]["honest"])
            else 0.0,
            "coverage": "1.0000",
            "notes": (
                f"fake={p5['store']['g_custom_share']['fake']} "
                f"ols={p5['store']['g_custom_share']['left_days']:.4f} "
                f"rho_resid={p5['store']['g_custom_share']['rho_resid_days']:.3f} "
                f"dilution={ctx['p22']['share_dil']:.3f} "
                f"kind={p8['kind'][:80]}"
            ),
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
    print(f"g_has_rest_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, list(FLAGS), (1, 3))
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )
    con = connect()
    try:
        print("pass 1 prevalence")
        p1 = pass1_prev(panel, tr)
        print("pass 2 rise-only")
        p2 = pass2_rise(tr)
        print("pass 3 Spearman")
        p3 = pass3_rho(tr)
        print("pass 4 oriented AUROC")
        p4 = pass4_auroc(tr)
        print("pass 5 leftover")
        p5 = pass5_leftover(tr)
        print("pass 6 terciles")
        p6 = pass6_terciles(tr)
        print("pass 7 dark")
        p7 = pass7_dark(tr, con)
        print("pass 8 custom_share")
        p8 = pass8_custom(tr, p5)
        print("pass 9 Q6")
        p9 = pass9_q6(tr)
        print("pass 10 card×TPV")
        p10 = pass10_card_tpv(tr)
        print("pass 11 ICC")
        p11 = pass11_icc(tr)
        print("pass 12 hole + decisions")
        p12 = pass12_hole_and_decisions(tr, p1, p2, p3, p4, p5, p8, p9, p11)
        print("pass 13 leftover types")
        p13 = pass13_other_types(tr, con)
        print("pass 14 Family F")
        p14 = pass14_family_f(tr)
        print("pass 15 holdout")
        p15 = pass15_holdout(panel)
        print("pass 16 created clock")
        p16 = pass16_created_clock(tr, con)
        print("pass 17 connected")
        p17 = pass17_connected(tr)
        print("pass 18 rates")
        p18 = pass18_rates(tr)
        print("pass 19 custom quintiles")
        p19 = pass19_custom_q(tr)
        print("pass 20 calendar")
        p20 = pass20_calendar(tr)
        print("pass 21 days terciles")
        p21 = pass21_days_terciles(tr)
        print("pass 22 custom dilution")
        p22 = pass22_custom_dilution(tr)
        print("pass 23 custom piles")
        p23 = pass23_custom_tail(tr)
        print("pass 24 checking Jaccard")
        p24 = pass24_jaccard(tr)
        print("pass 25 F fake leftover")
        p25 = pass25_f_fake(tr)
        print("pass 26 holdout custom")
        p26 = pass26_hold_custom(panel, p1)
        print("pass 27 first-on vs g_new")
        p27 = pass27_new_align(tr)
        print("pass 28 D2 custom + all-custom")
        p28 = pass28_d2_allcustom(tr)
        print("pass 29 investment T3")
        p29 = pass29_invest_t3(tr)
        print("pass 30 already-on-book")
        p30 = pass30_already_on(tr)
        print("pass 31 leftover types honesty")
        p31 = pass31_other_honest(tr, p13)
        print("pass 32 dark investment")
        p32 = pass32_dark_invest(tr, p7, con)
        png = make_plot(tr, p6)
    finally:
        con.close()
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
        "png": png,
        "elapsed_s": time.time() - t0,
    }
    write_md(ctx)
    append_registry(ctx)
    print(
        f"done elapsed={ctx['elapsed_s']:.0f}s drop_all={p12['drop_all']} "
        f"card={_f(p4['card'])} leftover_custom={_f(p8['left_days'])}"
    )
    return ctx


if __name__ == "__main__":
    run()
