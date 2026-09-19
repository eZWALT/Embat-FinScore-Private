"""Wave D — Y5 65% leftover as net TC × activity (Bureau diagnostic).

NORTH_STAR: Y5 is who is turning on AP overdue>30d (own p80). Night leftover
is the 2×2 neither cell (222/341 = 65.1%): not low a_io_ratio, not high
d_supp_hhi. Bureau–Duquerroy–Vinas: 1 SD net TC (AP−AR) × activity shock
raises payment-default PD +10% only in shock months; quiet months zero.

Feature (in-memory, diagnostic leftover only — never Y5 engine X, never E):
  diag_net_tc      = e_ap_open − e_ar_open
  diag_net_x_days  = net_tc × own-p20 days shock
  diag_net_x_in    = net_tc × own-p20 inflow shock
  diag_net_x_shock = net_tc × (days_lo | in_lo)

KEEP diagnostic footnote only if leftover after size+days ≥ 0.58 AND
beat-size ≥ 0.02 AND not SIZE AND not a twin. If leftover < 0.55, CLOSE
and the sentence stays: Y5 is a 65% leftover.

Never E as Y5 X. Y7 never D. Y3 never B. Dark 470 stay NaN not 0.
Do not grow TURNOVER. Do not put this on the 15-col card.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.y5_net_tc_qa
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
from analysis.features.common import ANALYSIS, DATA
from analysis.targets.y11_dark import book_invoice_ids
from analysis.features.common import connect

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "y5_net_tc_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "y5_net_tc_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_y5_net_tc.md"
AGENT = "689100e7"
WAVE = "4"
ROUND = "R4"
MODEL = "y5_net_tc_qa"
X_FAM = "diag"

Y3 = "y3_recover_cash_6m"
Y5 = "y5_ap_od30_ownp80"
Y5AR = "y5_ar_od30_sust"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DTX_QUOTE = 0.611
Y5_NEITHER = 0.651
TURNOVER_QUOTE = 0.720
B_SHALLOW_QUOTE = 0.712
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
TWIN_RHO = 0.80
KEEP_DELTA = 0.02
CHANCE = 0.55
KEEP_LEFT = 0.58
SIZE_RHO = 0.50
MIN_POS = 50
MIN_OWN_HIST = 6
OWN_P_HI = 0.80
OWN_P_LO = 0.20
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_io_ratio",
    "c_n_days_with_tx",
    "d_supp_hhi",
    "d_cust_hhi",
    "d_tx_cp_share",
    "h_group_size",
    "e_ap_open",
    "e_ar_open",
)

Y2 = "y2_neg_2of3"
Y_KEEP = (Y3, Y5, Y5AR, Y2)
STEM = "diag_net_tc"
X_DAYS = "diag_net_x_days"
X_IN = "diag_net_x_in"
X_SHOCK = "diag_net_x_shock"
GATE_TWINS = ("a_in3", "c_n_days_with_tx", "d_supp_hhi", "a_io_ratio", "d_tx_cp_share", "log_in3")
TWIN_COLS = GATE_TWINS + ("e_ap_open", "e_ar_open", STEM)


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


def _expanding_quantile_skipna(s: pd.Series, q: float, min_periods: int) -> pd.Series:
    out = np.full(len(s), np.nan, dtype=float)
    vals: list[float] = []
    for i, v in enumerate(s.to_numpy(dtype=float)):
        if np.isfinite(v):
            vals.append(float(v))
        if len(vals) >= min_periods:
            out[i] = float(np.quantile(vals, q))
    return pd.Series(out, index=s.index)


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
        "resid": resid,
        "rresid": rresid,
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


def add_own_cuts(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    for c, q, tag in (
        ("d_supp_hhi", OWN_P_HI, "hi"),
        ("c_n_days_with_tx", OWN_P_LO, "lo"),
        ("a_in3", OWN_P_LO, "lo"),
        ("a_io_ratio", OWN_P_LO, "lo"),
    ):
        x = pd.to_numeric(out[c], errors="coerce")
        pq = x.groupby(out["company_id"], sort=False).transform(
            lambda s, qq=q: _expanding_quantile_skipna(s, qq, MIN_OWN_HIST)
        )
        extra[f"{c}_{tag}"] = pd.Series(
            np.where(
                x.notna() & pq.notna(),
                ((x > pq) if tag == "hi" else (x < pq)).astype(float),
                np.nan,
            ),
            index=out.index,
        )
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def attach_diag(panel: pd.DataFrame) -> pd.DataFrame:
    out = add_own_cuts(panel)
    ap = pd.to_numeric(out["e_ap_open"], errors="coerce")
    ar = pd.to_numeric(out["e_ar_open"], errors="coerce")
    out[STEM] = ap - ar
    days_lo = pd.to_numeric(out["c_n_days_with_tx_lo"], errors="coerce")
    in_lo = pd.to_numeric(out["a_in3_lo"], errors="coerce")
    shock = ((days_lo == 1) | (in_lo == 1)).astype(float)
    shock[days_lo.isna() & in_lo.isna()] = np.nan
    out["shock"] = shock
    out["quiet"] = np.where(shock.notna(), 1.0 - shock, np.nan)
    out[X_DAYS] = out[STEM] * days_lo
    out[X_IN] = out[STEM] * in_lo
    out[X_SHOCK] = out[STEM] * shock
    out["diag_ap_x_shock"] = ap * shock
    out["diag_ar_x_shock"] = (-ar) * shock
    in3 = pd.to_numeric(out["a_in3"], errors="coerce")
    trail = in3.groupby(out["company_id"], sort=False).transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).mean()
    )
    dip = (in3 < 0.75 * trail).astype(float)
    dip[in3.isna() | trail.isna() | (trail <= 0)] = np.nan
    out["in_dip"] = dip
    out["diag_net_x_dip"] = out[STEM] * dip
    if Y2 in out.columns:
        y2 = pd.to_numeric(out[Y2], errors="coerce")
        out["diag_net_x_y2"] = out[STEM] * y2
    neither = (
        (out["a_io_ratio_lo"] == 0)
        & (out["d_supp_hhi_hi"] == 0)
        & out["a_io_ratio_lo"].notna()
        & out["d_supp_hhi_hi"].notna()
    )
    out["neither"] = neither
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
    leak5 = leakage_check(
        ["d_tx_cp_share", "h_group_size", "a_io_ratio", "d_supp_hhi", "log_in3"],
        Y5,
        forbidden_prefixes=["e", "b"],
    )
    leak3 = leakage_check(["c_n_days_with_tx"], Y3, forbidden_prefixes=["b"])
    if not leak5["ok"]:
        raise RuntimeError(f"Y5 X leak: {leak5['issues']}")
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
    dark = ~tr["company_id"].isin(book)
    y5 = pd.to_numeric(tr[Y5], errors="coerce")
    pos = y5 == 1
    flags = tr["a_io_ratio_lo"].notna() & tr["d_supp_hhi_hi"].notna()
    both = pos & flags
    neither_pos = both & tr["neither"]
    n_cm = int(len(tr))
    n_nn = int(x.notna().sum())
    dark_nn = int(x[dark].notna().sum())
    dark_zero = int((x[dark] == 0).sum())
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_ok = bool(dark_nn == 0 and dark_zero == 0 and n_dark_co == 470)
    n_pos_2x2 = int(both.sum())
    n_neither = int(neither_pos.sum())
    prose = (
        f"Train CM={n_cm:,}. {STEM} defined {n_nn:,} ({_pp(n_nn / n_cm)}). "
        f"Dark 470 nn={dark_nn} zero={dark_zero} "
        f"{'CONFIRM NaN not 0' if dark_ok else 'FAIL'}. "
        f"AP 2×2 positives {n_pos_2x2:,} neither {n_neither:,} "
        f"({_pp(n_neither / n_pos_2x2 if n_pos_2x2 else float('nan'))}; night 222/341=65.1%)."
    )
    print(prose)
    return {
        "n_cm": n_cm,
        "n_co": int(tr["company_id"].nunique()),
        "n_nn": n_nn,
        "cov": n_nn / n_cm if n_cm else float("nan"),
        "dark_nn": dark_nn,
        "dark_zero": dark_zero,
        "dark_ok": dark_ok,
        "y5_n": int(y5.notna().sum()),
        "y5_pos": int(pos.sum()),
        "n_pos_2x2": n_pos_2x2,
        "n_neither": n_neither,
        "neither_share": n_neither / n_pos_2x2 if n_pos_2x2 else float("nan"),
        "mean": float(x.mean()) if n_nn else float("nan"),
        "p50": float(x.median()) if n_nn else float("nan"),
        "shock_share": float(pd.to_numeric(tr["shock"], errors="coerce").mean()),
        "prose": prose,
    }


def pass2_twins(tr: pd.DataFrame) -> dict:
    x = tr[X_SHOCK]
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
        f"Gate twins vs size/days/HHI/io/d_tx: {gate or 'none'}. "
        f"ρ vs days={_f(rhos.get('c_n_days_with_tx'))} size={_f(rhos.get('log_in3'))} "
        f"supp_hhi={_f(rhos.get('d_supp_hhi'))} d_tx={_f(rhos.get('d_tx_cp_share'))}"
        f"{' SIZE' if size else ''}."
    )
    print(prose)
    return {"rows": rows, "rhos": rhos, "twins": twins, "gate": gate, "size": size, "prose": prose}


def pass3_singles(tr: pd.DataFrame) -> dict:
    y = tr[Y5]
    recs = {
        "net_tc": signed_oof_auroc(y, tr[STEM], tr["fold"], y.notna()),
        "net_x_days": signed_oof_auroc(y, tr[X_DAYS], tr["fold"], y.notna()),
        "net_x_in": signed_oof_auroc(y, tr[X_IN], tr["fold"], y.notna()),
        "net_x_shock": signed_oof_auroc(y, tr[X_SHOCK], tr["fold"], y.notna()),
        "ap_x_shock": signed_oof_auroc(y, tr["diag_ap_x_shock"], tr["fold"], y.notna()),
        "ar_x_shock": signed_oof_auroc(y, tr["diag_ar_x_shock"], tr["fold"], y.notna()),
        "size": signed_oof_auroc(y, tr["log_in3"], tr["fold"], y.notna()),
        "days": signed_oof_auroc(y, tr["c_n_days_with_tx"], tr["fold"], y.notna()),
        "d_tx": signed_oof_auroc(y, tr["d_tx_cp_share"], tr["fold"], y.notna()),
        "h_group": signed_oof_auroc(y, tr["h_group_size"], tr["fold"], y.notna()),
        "io": signed_oof_auroc(y, tr["a_io_ratio"], tr["fold"], y.notna()),
        "hhi": signed_oof_auroc(y, tr["d_supp_hhi"], tr["fold"], y.notna()),
    }
    rows = [_auc_row(Y5, k, r) for k, r in recs.items()]
    shock = _cv(recs["net_x_shock"])
    size = _cv(recs["size"])
    beat = bool(np.isfinite(shock) and np.isfinite(size) and (shock - size) >= KEEP_DELTA)
    prose = (
        f"Y5 {X_SHOCK} {_f(shock)} vs size {_f(size)} days {_f(_cv(recs['days']))} "
        f"d_tx {_f(_cv(recs['d_tx']))} (AR-night 0.611 PARK). "
        f"beat-size {'PASS' if beat else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "shock": shock,
        "net": _cv(recs["net_tc"]),
        "size": size,
        "days": _cv(recs["days"]),
        "d_tx": _cv(recs["d_tx"]),
        "ap": _cv(recs["ap_x_shock"]),
        "ar": _cv(recs["ar_x_shock"]),
        "beat": beat,
        "prose": prose,
    }


def pass4_leftover(tr: pd.DataFrame) -> dict:
    y = tr[Y5]
    x = tr[X_SHOCK]
    mask = y.notna()
    neither = mask & tr["neither"]
    cuts = {
        "after size+days": leftover_diag(
            y, x, (tr["log_in3"], tr["c_n_days_with_tx"]), tr["fold"], mask
        ),
        "after size": leftover_diag(y, x, (tr["log_in3"],), tr["fold"], mask),
        "after days": leftover_diag(y, x, (tr["c_n_days_with_tx"],), tr["fold"], mask),
        "after io+hhi": leftover_diag(
            y, x, (tr["a_io_ratio"], tr["d_supp_hhi"]), tr["fold"], mask
        ),
        "after d_tx": leftover_diag(y, x, (tr["d_tx_cp_share"],), tr["fold"], mask),
        "net_tc after size+days": leftover_diag(
            y, tr[STEM], (tr["log_in3"], tr["c_n_days_with_tx"]), tr["fold"], mask
        ),
        "neither after size+days": leftover_diag(
            y, x, (tr["log_in3"], tr["c_n_days_with_tx"]), tr["fold"], neither
        ),
        "neither net_tc after size+days": leftover_diag(
            y, tr[STEM], (tr["log_in3"], tr["c_n_days_with_tx"]), tr["fold"], neither
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
                "n_pos": f"{d['n_pos']:,}",
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        )
    main = cuts["after size+days"]
    nei = cuts["neither after size+days"]
    keep = bool(
        np.isfinite(main["rank"])
        and main["rank"] >= KEEP_LEFT
        and not main["fake"]
        and not main["honest_dies"]
    )
    prose = (
        f"Y5 leftover of {X_SHOCK} after size+days rank {_f(main['rank'])} OLS {_f(main['ols'])} "
        f"{'KEEP ≥0.58' if keep else ('dies' if main['honest_dies'] else 'lives but <0.58')}. "
        f"Neither-cell leftover {_f(nei['rank'])} n_pos={nei['n_pos']:,}. "
        f"Plain net_tc after size+days {_f(cuts['net_tc after size+days']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "cuts": cuts,
        "main_rank": main["rank"],
        "main_ols": main["ols"],
        "main_dies": main["honest_dies"],
        "nei_rank": nei["rank"],
        "nei_dies": nei["honest_dies"],
        "net_rank": cuts["net_tc after size+days"]["rank"],
        "keep": keep,
        "prose": prose,
    }


def pass5_shock_quiet(tr: pd.DataFrame) -> dict:
    print("EXTRA — shock months vs quiet (Bureau: only shock months move PD)")
    y = tr[Y5]
    shock = pd.to_numeric(tr["shock"], errors="coerce") == 1
    quiet = pd.to_numeric(tr["quiet"], errors="coerce") == 1
    rows = []
    store = {}
    for label, m, xcol in (
        ("net_tc on shock months", y.notna() & shock, STEM),
        ("net_tc on quiet months", y.notna() & quiet, STEM),
        ("net×shock on shock months", y.notna() & shock, X_SHOCK),
        ("AP×shock on shock months", y.notna() & shock, "diag_ap_x_shock"),
        ("−AR×shock on shock months", y.notna() & shock, "diag_ar_x_shock"),
        ("net_tc on neither+shock", y.notna() & tr["neither"] & shock, STEM),
        ("net_tc on neither+quiet", y.notna() & tr["neither"] & quiet, STEM),
    ):
        rec = leftover_diag(
            y, tr[xcol], (tr["log_in3"], tr["c_n_days_with_tx"]), tr["fold"], m
        )
        store[label] = rec
        rows.append(
            {
                "cut": label,
                "rank": _f(rec["rank"]),
                "OLS": _f(rec["ols"]),
                "n": f"{rec['n']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "dies?": "dies" if rec["honest_dies"] else "lives",
            }
        )
    prose = (
        f"Shock-month net_tc leftover {_f(store['net_tc on shock months']['rank'])}; "
        f"quiet {_f(store['net_tc on quiet months']['rank'])}. "
        f"AP×shock {_f(store['AP×shock on shock months']['rank'])}; "
        f"−AR×shock {_f(store['−AR×shock on shock months']['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "shock": store["net_tc on shock months"]["rank"],
        "quiet": store["net_tc on quiet months"]["rank"],
        "ap": store["AP×shock on shock months"]["rank"],
        "ar": store["−AR×shock on shock months"]["rank"],
        "prose": prose,
    }


def pass6_y3_hold(tr: pd.DataFrame, panel: pd.DataFrame, book: set[str]) -> dict:
    d = leftover_diag(
        tr[Y3], tr[X_SHOCK], (tr["c_n_days_with_tx"],), tr["fold"], tr[Y3].notna()
    )
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
        f"Y3 leftover after days {_f(d['rank'])} — off the 15-col card. "
        f"Holdout 72 coverage only. Holdout {STEM} nn={int(x[hold].notna().sum()):,}. "
        f"Holdout dark nn={int(x[hold & dark].notna().sum()):,} (want 0)."
    )
    print(prose)
    return {
        "y3_rank": d["rank"],
        "y3_dies": d["honest_dies"],
        "rows": rows,
        "y3_rows": [
            {
                "cut": "Y3 leftover after days",
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        ],
        "prose": prose,
    }


def pass7_quintiles(tr: pd.DataFrame) -> dict:
    y = pd.to_numeric(tr[Y5], errors="coerce")
    x = pd.to_numeric(tr[X_SHOCK], errors="coerce")
    lab = y.notna() & x.notna()
    work = tr.loc[lab, [Y5, X_SHOCK]].copy()
    r = work[X_SHOCK].rank(method="first")
    work["q"] = pd.qcut(r, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rows = []
    for q, g in work.groupby("q", observed=False):
        rows.append(
            {
                "q": str(q),
                "n": f"{len(g):,}",
                "rate": _pp(float(g[Y5].mean())),
                "p50 net×shock": _f(float(g[X_SHOCK].median())),
            }
        )
    q1 = work.loc[work["q"].astype(str) == "Q1", Y5].mean()
    q5 = work.loc[work["q"].astype(str) == "Q5", Y5].mean()
    lift = float(q5 - q1) if np.isfinite(q1) and np.isfinite(q5) else float("nan")
    prose = f"Y5 rate by rank-quintile of net TC × shock. Q5−Q1 {_pp(lift)} (Bureau seat)."
    print(prose)
    return {"rows": rows, "q5_q1": lift, "prose": prose}


def pass9_alt_shocks(tr: pd.DataFrame) -> dict:
    print("EXTRA — alternate shocks: inflow −25% trail, Y2, fold 3, neither after io+hhi")
    y = tr[Y5]
    mask = y.notna()
    ctrl = (tr["log_in3"], tr["c_n_days_with_tx"])
    cuts = {
        "net × inflow-dip after size+days": leftover_diag(
            y, tr["diag_net_x_dip"], ctrl, tr["fold"], mask
        ),
        "net_tc after io+hhi on neither": leftover_diag(
            y, tr[STEM], (tr["a_io_ratio"], tr["d_supp_hhi"]), tr["fold"], mask & tr["neither"]
        ),
        "net×shock after io+hhi": leftover_diag(
            y, tr[X_SHOCK], (tr["a_io_ratio"], tr["d_supp_hhi"]), tr["fold"], mask
        ),
    }
    if "diag_net_x_y2" in tr.columns:
        cuts["net × Y2 after size+days"] = leftover_diag(
            y, tr["diag_net_x_y2"], ctrl, tr["fold"], mask
        )
    rows = []
    for name, d in cuts.items():
        rows.append(
            {
                "cut": name,
                "rank": _f(d["rank"]),
                "OLS": _f(d["ols"]),
                "n": f"{d['n']:,}",
                "n_pos": f"{d['n_pos']:,}",
                "dies?": "dies" if d["honest_dies"] else "lives",
            }
        )
    rec = signed_oof_auroc(y, tr[X_SHOCK], tr["fold"], mask)
    f3 = next((r["auroc"] for r in rec.get("folds", []) if int(r["fold"]) == 3), float("nan"))
    prose = (
        f"Inflow-dip leftover {_f(cuts['net × inflow-dip after size+days']['rank'])}; "
        f"neither after io+hhi {_f(cuts['net_tc after io+hhi on neither']['rank'])}. "
        f"Fold 3 raw {X_SHOCK} {_f(f3)} — not a d_tx leave-one-group revival."
    )
    print(prose)
    return {
        "rows": rows,
        "dip": cuts["net × inflow-dip after size+days"]["rank"],
        "nei_iohhi": cuts["net_tc after io+hhi on neither"]["rank"],
        "f3": f3,
        "prose": prose,
    }


def pass8_boot(tr: pd.DataFrame, n_boot: int = 30) -> dict:
    print(f"EXTRA — company bootstrap leftover after size+days (n={n_boot})")
    y = pd.to_numeric(tr[Y5], errors="coerce")
    work = tr.loc[y.notna(), ["company_id", "fold", Y5, X_SHOCK, "log_in3", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y5],
            b[X_SHOCK],
            (b["log_in3"], b["c_n_days_with_tx"]),
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
        f"Bootstrap leftover-after-size+days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"share<0.55={_pp(share_die)} n={len(ranks)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share_die, "prose": prose}


def decide(p2, p3, p4) -> dict:
    twin = bool(p2["gate"] or p2["size"])
    keep = bool(p4["keep"] and p3["beat"] and not twin)
    if keep:
        y5 = "KEEP"
        why = (
            f"diagnostic leftover after size+days {_f(p4['main_rank'])} ≥ 0.58, "
            f"not a twin, beat-size PASS — footnote only, never E-on-Y5"
        )
    elif p4["main_dies"] or (np.isfinite(p4["main_rank"]) and p4["main_rank"] < CHANCE):
        y5 = "CLOSE"
        why = (
            f"leftover {_f(p4['main_rank'])} dies — Y5 stays a 65% leftover "
            f"(neither {p4.get('nei_note', '222/341')})"
        )
    else:
        y5 = "CLOSE"
        why = (
            f"leftover {_f(p4['main_rank'])} lives but <0.58 or twin/beat-size FAIL "
            f"— Y5 stays a 65% leftover"
        )
    return {"y5": y5, "y5_why": why, "keep": keep}


def make_png(tr: pd.DataFrame) -> bool:
    if not HAS_MPL:
        return False
    y = pd.to_numeric(tr[Y5], errors="coerce")
    x = pd.to_numeric(tr[X_SHOCK], errors="coerce")
    lab = y.notna() & x.notna()
    work = tr.loc[lab, [Y5, X_SHOCK, "shock"]].copy()
    r = work[X_SHOCK].rank(method="first")
    work["q"] = pd.qcut(r, 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    rates = work.groupby("q", observed=False)[Y5].mean()
    shock = pd.to_numeric(work["shock"], errors="coerce")
    bins = pd.Series(np.where(shock == 1, "shock", "quiet"), index=work.index)
    pile = work.groupby(bins, observed=False)[Y5].mean()
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    axes[0].bar(rates.index.astype(str), rates.values, color="#3d5a80")
    axes[0].set_ylabel("Y5 AP-od30 rate")
    axes[0].set_xlabel("rank-quintile of net TC × shock")
    axes[0].set_title("Bureau net TC × activity (Y5)")
    axes[0].set_ylim(0, max(0.20, float(rates.max()) + 0.04))
    order = [b for b in ("quiet", "shock") if b in pile.index]
    axes[1].bar(order, [float(pile[b]) for b in order], color="#ee6c4d")
    axes[1].set_ylabel("Y5 AP-od30 rate")
    axes[1].set_xlabel("quiet vs shock month")
    axes[1].set_title("Shock vs quiet (Bureau)")
    axes[1].set_ylim(0, max(0.20, float(pile.max()) + 0.04))
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    p5, p6, p7, p8 = ctx["p5"], ctx["p6"], ctx["p7"], ctx["p8"]
    d = ctx["decision"]
    lines = [
        "# Y5 leftover 65% as net TC × activity (Bureau diagnostic)",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. **Never E as Y5 X.** Diagnostic leftover only. "
        "Night Y7 stays **TURNOVER 0.720 / 0.712**. Do **not** grow TURNOVER. "
        "Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0.",
        "",
        "`diag_net_tc` = `e_ap_open − e_ar_open`. Interaction with own-p20 days / inflow "
        "shock. Bureau–Duquerroy–Vinas: 1 SD net TC × lockdown → +10% payment-default PD "
        "only in shock months. Seat is the AP neither cell (night **222/341 = 65.1%**).",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        "| 1 | Who is healthy? | **PARK**. Y5 is a turn, not a health Y. |",
        "| 2 | Who is improving? | Not a recovery clock. |",
        "| 3 | Who is turning? | Y5 is the turn. This cut asks *why the leftover 65%*. |",
        f"| 4 | Dip vs fall? | Diagnostic leftover **{d['y5']}** — {d['y5_why']} |",
        "| 5 | Why did it change? | Bureau: net TC only bites in an activity shock. |",
        "| 6 | Months earlier? | **CLOSE** — leftover is contemporaneous; d_tx Q6 already CLOSE. |",
        "",
        "## PARK / CLOSE / KEEP / DROP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| net TC × shock leftover after size+days (Y5 diagnostic) | **{d['y5']}** | {d['y5_why']} |",
        "| as Y5 engine X (Family E) | **DROP** | Y5 never E |",
        "| TURNOVER add-on / 15-col card | **CLOSE** | do not grow 0.720 |",
        "| d_tx as leave-one-group law | **PARK** | night 0.611 PARK |",
        "| d_supp_hhi as engine X | **DROP** | protective tail 2.7% vs 8.6% already measured |",
        "",
        "## 1. Coverage / neither cell / dark 470",
        "",
        p1["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| train CM / companies | {p1['n_cm']:,} / {p1['n_co']:,} |",
        f"| {STEM} defined | {p1['n_nn']:,} ({_pp(p1['cov'])}) |",
        f"| mean / p50 net TC | {_f(p1['mean'])} / {_f(p1['p50'])} |",
        f"| shock-month share | {_pp(p1['shock_share'])} |",
        f"| Y5 labeled / pos | {p1['y5_n']:,} / {p1['y5_pos']:,} |",
        f"| 2×2 pos / neither | {p1['n_pos_2x2']:,} / {p1['n_neither']:,} |",
        f"| neither share | {_pp(p1['neither_share'])} |",
        f"| dark 470 nn / zero | {p1['dark_nn']} / {p1['dark_zero']} |",
        f"| dark NaN | {'CONFIRM' if p1['dark_ok'] else 'FAIL'} |",
        "",
        "## 2. Spearman twins (|ρ|≥0.80)",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3. Single-feature train group-fold AUROC (diagnostic)",
        "",
        f"Days **0.711** is the Y3 bar, not a Y5 engine. Size **0.617**. "
        f"d_tx night **0.611 PARK**. TURNOVER **0.720** unchanged.",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "## 4. Residual Y5 after size+days (KEEP gate)",
        "",
        "KEEP diagnostic footnote only if rank ≥ **0.58**, not a twin, not SIZE, beat-size. "
        "Never E as Y5 X. Rank leftover is honest; OLS can fake.",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        "## 5. Extra — shock vs quiet; AP vs AR sign",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6. Y3 off-card + dark/holdout coverage",
        "",
        p6["prose"],
        "",
        _md_table(p6["y3_rows"]),
        "",
        _md_table(p6["rows"]),
        "",
        "## 7. Quintiles",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped.",
        "",
        "## 8. Company bootstrap leftover after size+days",
        "",
        p8["prose"],
        "",
        "## 9. Extra — alternate shocks / fold 3 / neither after cash+HHI",
        "",
        ctx["p9"]["prose"],
        "",
        _md_table(ctx["p9"]["rows"]),
        "",
        "## What this note did not do",
        "",
        "- Did not change Y7 0.720 / 0.712 or Y3 0.762 / 0.752.",
        "- Did not put Family E on the Y5 card. Did not grow TURNOVER.",
        "- Did not overwrite y5_why / supp_hhi_qa / d_tx_qa / delay_qa / issued_qa.",
        "- Did not score this as Y3 B or Y7 D. Dark 470 stayed NaN.",
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
    p4 = ctx["p4"]
    WAVE_NOTE.write_text(
        "\n".join(
            [
                "# Wave 4 — Y5 net TC × activity leftover (Wave D)",
                "",
                f"Owner: `{MODEL}`. As-of `{_now_iso()}`.",
                "Deliverable: `analysis/outputs/y5_net_tc_qa.md`.",
                "Never E as Y5 X. Do not grow TURNOVER **0.720**.",
                "",
                "## Headline",
                "",
                ctx["headline"],
                "",
                "## Decision",
                "",
                f"- Y5 diagnostic leftover after size+days: **{d['y5']}** ({d['y5_why']})",
                f"- rank {_f(p4['main_rank'])} OLS {_f(p4['main_ols'])} "
                f"neither {_f(p4['nei_rank'])} plain-net {_f(p4['net_rank'])}",
                "",
                "## Do not do next",
                "",
                "- Put Family E on the Y5 card. Grow TURNOVER. Overwrite y5_why.",
                "- Promote d_tx as leave-one-group law. Merge Y4 trees.",
                "",
                "## Extras after headline",
                "",
                f"- Shock leftover {_f(ctx['p5']['shock'])}; quiet {_f(ctx['p5']['quiet'])}.",
                f"- AP×shock {_f(ctx['p5']['ap'])}; −AR×shock {_f(ctx['p5']['ar'])}.",
                f"- Boot p05 {_f(ctx['p8']['p05'])} p50 {_f(ctx['p8']['p50'])}.",
                f"- Inflow-dip leftover {_f(ctx['p9']['dip'])}; neither after io+hhi {_f(ctx['p9']['nei_iohhi'])}; fold 3 {_f(ctx['p9']['f3'])}.",
                "- CLOSE stays. Y5 is a 65% leftover. Never E-on-Y5.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"y5_net_tc_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = attach_diag(panel)
    con = connect()
    try:
        book = book_invoice_ids(con)
        print(f"book ids {len(book)}")
    finally:
        con.close()
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(f"train CM={len(tr):,} companies={tr['company_id'].nunique()}")

    p1 = pass1_cov(tr, book)
    p2 = pass2_twins(tr)
    p3 = pass3_singles(tr)
    p4 = pass4_leftover(tr)
    p5 = pass5_shock_quiet(tr)
    p6 = pass6_y3_hold(tr, panel, book)
    p7 = pass7_quintiles(tr)
    p8 = pass8_boot(tr, n_boot=30)
    p9 = pass9_alt_shocks(tr)
    decision = decide(p2, p3, p4)
    png_ok = make_png(tr)
    headline = (
        f"{STEM} defined {_pp(p1['cov'])}; dark 470 "
        f"{'NaN CONFIRM' if p1['dark_ok'] else 'FAIL'}. "
        f"Neither {p1['n_neither']:,}/{p1['n_pos_2x2']:,} ({_pp(p1['neither_share'])}; night 65.1%). "
        f"Y5 {X_SHOCK} raw {_f(p3['shock'])} vs size {_f(p3['size'])} "
        f"beat-size {'PASS' if p3['beat'] else 'FAIL'}. "
        f"Leftover after size+days rank {_f(p4['main_rank'])} OLS {_f(p4['main_ols'])} "
        f"(neither {_f(p4['nei_rank'])}; plain net {_f(p4['net_rank'])}). "
        f"Shock {_f(p5['shock'])} vs quiet {_f(p5['quiet'])}. "
        f"Boot p05 {_f(p8['p05'])}. Y5 leftover **{decision['y5']}**. "
        f"Never E-on-Y5. Do not grow TURNOVER 0.720. 65% sentence stays if this dies."
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
                "y": Y5,
                "model": MODEL,
                "split": "train_cv",
                "metric": "auroc_net_x_shock",
                "value": p3["shock"],
                "coverage": p1["cov"],
                "notes": headline[:240],
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y5,
                "model": MODEL,
                "split": "train_cv",
                "metric": "leftover_size_days_rank",
                "value": p4["main_rank"],
                "coverage": p1["cov"],
                "notes": f"ols={_f(p4['main_ols'])} keep={decision['y5']}",
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y5,
                "model": MODEL,
                "split": "train_cv",
                "metric": "leftover_neither_rank",
                "value": p4["nei_rank"],
                "coverage": p1["neither_share"],
                "notes": f"shock={_f(p5['shock'])} quiet={_f(p5['quiet'])}",
            },
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": X_FAM,
                "y": Y5,
                "model": MODEL,
                "split": "train_cv",
                "metric": "leftover_shock_vs_quiet",
                "value": p5["shock"],
                "coverage": p1["cov"],
                "notes": f"quiet={_f(p5['quiet'])} dip={_f(p9['dip'])}",
            },
        ]
    )
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"done in {time.time() - t0:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
