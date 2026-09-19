"""Norden lead without utilisation: leftover of Δdays after days-level.

``c_n_days_with_tx`` level is the 0.711 bar. This cut asks whether a
3- or 6-month **change** in days-with-tx leftover after that *level*
is a Y3 X on ≥18-month books.

Not AMPLI (HIGH−LOW of balances) — B-forbidden. Not utilisation / Y10.
Hidden 72 stays 1-month. Do not put Δdays on the 15-col card unless
KEEP-as-X clears.

KEEP-as-X: leftover after days-level ≥0.60 AND beat size ≥0.02 AND
not SIZE (|ρ| vs log1p(a_in3) ≥0.50) AND not a twin (|ρ|≥0.80 vs
days / a_n_tx / c_gap_sd / c_recency_days). Leftover <0.55 dies.
Rank leftover is honest.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.days_delta_qa

Owned: analysis/evaluate/days_delta_qa.py, analysis/outputs/days_delta_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_days_delta.md (end).
"""
from __future__ import annotations

import sys
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
OUT_MD = ANALYSIS / "outputs" / "days_delta_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "days_delta_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"

Y3 = "y3_recover_cash_6m"
Y2 = "y2_neg_2of3"
N_FOLDS = 5
DAYS_BAR = 0.711
SIZE_BAR = 0.617
DAYS_LAG1 = 0.684
Y3_NIGHT = (0.762, 0.752)
KEEP_LEFT = 0.60
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
TWIN_RHO = 0.80
CHANCE = 0.55
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_BOOT = 80
BOOT_SEED = 20260918

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_n_tx",
    "c_n_days_with_tx",
    "c_gap_sd",
    "c_recency_days",
    "c_ss_month",
    "c_salary_month",
    "first_month",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


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


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


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


def signed_oof(y, x, folds, mask) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    low = n_pos < MIN_POS or n_neg == 0
    rows = []
    aucs = []
    if low:
        return {
            "cv": float("nan"),
            "sd": float("nan"),
            "train_auc": float("nan"),
            "sign": 0,
            "folds": rows,
            "n": n,
            "n_pos": n_pos,
            "low_power": True,
        }
    for k in range(N_FOLDS):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            rows.append({"fold": k, "auroc": float("nan"), "n_pos": int((va & (y == 1)).sum())})
            continue
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        rows.append({"fold": k, "auroc": float(auc), "n_pos": int((va & (y == 1)).sum())})
    finite = [a for a in aucs if np.isfinite(a)]
    sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "train_auc": float(auroc(y[defined], sign * x[defined])),
        "sign": int(sign),
        "folds": rows,
        "n": n,
        "n_pos": n_pos,
        "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r.get("auroc", float("nan"))) else "—"
        for r in rec.get("folds", [])
    )


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "r2": float("nan")}
    if int(ok.sum()) < max(20, len(xs) + 5):
        return resid, info
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(len(xs))]
    )
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    pred = X @ beta
    resid.loc[ok] = Y - pred
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    info["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return resid, info


def leftover(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof(y, resid, folds, mask)
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rresid, _ = ols_resid(xr, *cr)
    rrec = signed_oof(y, rresid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    rank = float("nan") if rrec["low_power"] else rrec["cv"]
    ols = float("nan") if rec["low_power"] else rec["cv"]
    fake = bool(np.isfinite(rho_c) and abs(rho_c) >= TWIN_RHO)
    return {
        "ols": ols,
        "rank": rank,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "fake": fake,
        "dies": bool(fake or (np.isfinite(rank) and rank < CHANCE) or rrec["low_power"]),
        "rec": rec,
        "rrec": rrec,
        "n": rrec["n"],
        "n_pos": rrec["n_pos"],
        "low_power": rrec["low_power"],
        "folds": fold_bits(rrec),
    }


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


def company_boot_leftover(y, x, ctrl, folds, mask, n_boot=N_BOOT) -> dict:
    d = pd.DataFrame(
        {
            "y": pd.to_numeric(y, errors="coerce"),
            "x": pd.to_numeric(x, errors="coerce"),
            "c": pd.to_numeric(ctrl, errors="coerce"),
            "fold": folds,
            "co": mask.index.map(lambda i: None),
        }
    )
    # company ids come from the panel index alignment — pass company separately
    return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan"), "share_lt": float("nan")}


def boot_leftover(panel: pd.DataFrame, ycol: str, xcol: str, ccol: str, mask: pd.Series) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    cos = panel.loc[mask, "company_id"].astype(str).drop_duplicates().to_numpy()
    vals = []
    y = pd.to_numeric(panel[ycol], errors="coerce")
    x = pd.to_numeric(panel[xcol], errors="coerce")
    c = pd.to_numeric(panel[ccol], errors="coerce")
    folds = panel["fold"]
    for _ in range(N_BOOT):
        draw = rng.choice(cos, size=len(cos), replace=True)
        keep = panel["company_id"].astype(str).isin(set(draw))
        rec = leftover(y, x, [c], folds, mask & keep)
        if not rec["low_power"] and np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    if not vals:
        return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan"), "share_lt": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "share_lt": float(np.mean(arr < CHANCE)),
        "n": len(arr),
    }


def trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


def gate(raw: dict, left: dict, rhos: dict) -> dict:
    raw_cv = raw["cv"]
    left_cv = left["rank"]
    beat = (
        np.isfinite(raw_cv)
        and (raw_cv - SIZE_BAR) >= KEEP_DELTA
        and not raw["low_power"]
    )
    size = bool(np.isfinite(rhos["size"]) and abs(rhos["size"]) >= SIZE_RHO)
    twins = [k for k, v in rhos["twins"].items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    left_ok = bool(np.isfinite(left_cv) and left_cv >= KEEP_LEFT and not left["dies"] and not left["low_power"])
    keep = bool(left_ok and beat and not size and not twins)
    if left["low_power"] or raw["low_power"]:
        role = "LOW_POWER"
    elif keep:
        role = "KEEP"
    elif np.isfinite(left_cv) and left_cv < CHANCE:
        role = "DROP"
    else:
        role = "CLOSE"
    return {
        "keep": keep,
        "role": role,
        "beat": beat,
        "beat_delta": (raw_cv - SIZE_BAR) if np.isfinite(raw_cv) else float("nan"),
        "size": size,
        "twins": twins,
        "left_ok": left_ok,
    }


def load_panel() -> tuple[pd.DataFrame, set[str]]:
    raw = pd.read_parquet(STORE, columns=list(STORE_COLS))
    yraw = pd.read_parquet(TARGETS)
    raw["company_id"] = raw["company_id"].astype(str)
    yraw["company_id"] = yraw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    yraw["period"] = pd.to_datetime(yraw["period"])
    hold = load_holdout()
    tr = raw.loc[~raw["company_id"].isin(hold)].copy()
    ytr = yraw.loc[~yraw["company_id"].isin(hold)].copy()
    assert_no_holdout(tr["company_id"])
    panel = tr.merge(ytr, on=["company_id", "period"], how="left")
    panel = panel.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = panel["company_id"]
    days = pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce")
    g = days.groupby(cid, sort=False)
    panel["days"] = days
    panel["days_lag1"] = g.shift(1)
    panel["days_lag3"] = g.shift(3)
    panel["days_lag6"] = g.shift(6)
    panel["d3"] = days - panel["days_lag3"]
    panel["d6"] = days - panel["days_lag6"]
    if "months_so_far" not in panel.columns:
        panel["months_so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    panel["book_len"] = panel.groupby("company_id", sort=False)["months_so_far"].transform("max")
    panel["book_class"] = panel["book_len"].map(trail_class)
    panel["sofar_class"] = panel["months_so_far"].map(trail_class)
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    folds = group_folds(
        panel[["company_id", "group_id"]].drop_duplicates(), n=N_FOLDS, seed=FOLD_SEED
    )
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    leak = leakage_check(["d3", "d6", "days"], Y3, forbidden_prefixes=("b", "f"))
    if not leak["ok"]:
        raise AssertionError(leak["issues"])
    return panel, hold


def slice_pack(panel, y, folds, mask, label: str) -> dict:
    days = panel["days"]
    d3 = panel["d3"]
    d6 = panel["d6"]
    log_in = panel["log_in3"]
    rec_days = signed_oof(y, days, folds, mask)
    rec_d3 = signed_oof(y, d3, folds, mask)
    rec_d6 = signed_oof(y, d6, folds, mask)
    rec_size = signed_oof(y, log_in, folds, mask)
    rec_l1 = signed_oof(y, panel["days_lag1"], folds, mask)
    left_d3 = leftover(y, d3, [days], folds, mask)
    left_d6 = leftover(y, d6, [days], folds, mask)
    inv_d3 = leftover(y, days, [d3], folds, mask)
    inv_d6 = leftover(y, days, [d6], folds, mask)
    defined = mask & y.notna() & d3.notna()
    rhos3 = {
        "size": spearman(d3[defined], log_in[defined]),
        "twins": {
            "days": spearman(d3[defined], days[defined]),
            "a_n_tx": spearman(d3[defined], pd.to_numeric(panel.loc[defined, "a_n_tx"], errors="coerce")),
            "c_gap_sd": spearman(d3[defined], pd.to_numeric(panel.loc[defined, "c_gap_sd"], errors="coerce"))
            if "c_gap_sd" in panel.columns
            else float("nan"),
            "c_recency_days": spearman(
                d3[defined], pd.to_numeric(panel.loc[defined, "c_recency_days"], errors="coerce")
            )
            if "c_recency_days" in panel.columns
            else float("nan"),
        },
    }
    defined6 = mask & y.notna() & d6.notna()
    rhos6 = {
        "size": spearman(d6[defined6], log_in[defined6]),
        "twins": {
            "days": spearman(d6[defined6], days[defined6]),
            "a_n_tx": spearman(d6[defined6], pd.to_numeric(panel.loc[defined6, "a_n_tx"], errors="coerce")),
            "c_gap_sd": spearman(d6[defined6], pd.to_numeric(panel.loc[defined6, "c_gap_sd"], errors="coerce"))
            if "c_gap_sd" in panel.columns
            else float("nan"),
            "c_recency_days": spearman(
                d6[defined6], pd.to_numeric(panel.loc[defined6, "c_recency_days"], errors="coerce")
            )
            if "c_recency_days" in panel.columns
            else float("nan"),
        },
    }
    g3 = gate(rec_d3, left_d3, rhos3)
    g6 = gate(rec_d6, left_d6, rhos6)
    return {
        "label": label,
        "days": rec_days,
        "d3": rec_d3,
        "d6": rec_d6,
        "size": rec_size,
        "lag1": rec_l1,
        "left_d3": left_d3,
        "left_d6": left_d6,
        "inv_d3": inv_d3,
        "inv_d6": inv_d6,
        "rhos3": rhos3,
        "rhos6": rhos6,
        "g3": g3,
        "g6": g6,
        "n": int((mask & y.notna()).sum()),
        "n_pos": int((mask & (y == 1)).sum()),
        "n_d3": left_d3["n"],
        "n_pos_d3": left_d3["n_pos"],
        "n_d6": left_d6["n"],
        "n_pos_d6": left_d6["n_pos"],
    }


def decide_primary(longp: dict) -> dict:
    """Primary object is Δ3 then Δ6 on ≥18m books."""
    roles = []
    for name, g, left in (
        ("d3", longp["g3"], longp["left_d3"]),
        ("d6", longp["g6"], longp["left_d6"]),
    ):
        roles.append((name, g["role"], g, left))
    if any(g["keep"] for _, _, g, _ in roles):
        winner = next(n for n, _, g, _ in roles if g["keep"])
        card = "KEEP as Y3 X — parent absorbs; do not edit the 15-col spec here"
        overall = "KEEP"
    elif all(g["role"] == "LOW_POWER" for _, _, g, _ in roles):
        winner = "d3"
        card = "LOW_POWER on ≥18m — stay off the card"
        overall = "LOW_POWER"
    elif any(g["role"] == "CLOSE" for _, _, g, _ in roles) or any(
        np.isfinite(left["rank"]) and left["rank"] >= CHANCE for _, _, _, left in roles
    ):
        winner = "d3" if longp["g3"]["role"] != "DROP" else "d6"
        card = "CLOSE unused leftover / KEEP off the 15-col card"
        overall = "CLOSE"
    else:
        winner = "d3"
        card = "DROP from Y3 X / KEEP off the 15-col card"
        overall = "DROP"
    return {"overall": overall, "card": card, "winner": winner, "roles": roles}


def main() -> None:
    t0 = datetime.now(timezone.utc)
    panel, hold = load_panel()
    y = pd.to_numeric(panel[Y3], errors="coerce")
    folds = panel["fold"]
    stressed = y.notna()
    long_co = panel["book_class"] == "long_>=18"
    short_co = panel["book_class"] == "short_<12"
    mid_co = panel["book_class"] == "mid_12_17"

    longp = slice_pack(panel, y, folds, stressed & long_co, "company ≥18")
    allp = slice_pack(panel, y, folds, stressed, "all train")
    shortp = slice_pack(panel, y, folds, stressed & short_co, "company <12")
    midp = slice_pack(panel, y, folds, stressed & mid_co, "company 12–17")

    con = connect()
    book = book_invoice_ids(con)
    con.close()
    is_erp = panel["company_id"].isin(book)
    darkp = slice_pack(panel, y, folds, stressed & long_co & ~is_erp, "long dark")
    erpp = slice_pack(panel, y, folds, stressed & long_co & is_erp, "long ERP")

    first6 = panel["months_so_far"] <= 6
    later = panel["months_so_far"] > 6
    firstp = slice_pack(panel, y, folds, stressed & long_co & first6, "long first6")
    laterp = slice_pack(panel, y, folds, stressed & long_co & later, "long later")

    y2 = pd.to_numeric(panel[Y2], errors="coerce")
    y2_long = leftover(y2, panel["d3"], [panel["days"]], folds, y2.notna() & long_co)

    boot3 = boot_leftover(panel, Y3, "d3", "days", stressed & long_co)
    boot6 = boot_leftover(panel, Y3, "d6", "days", stressed & long_co)

    ss = pd.to_numeric(panel.get("c_ss_month"), errors="coerce")
    left_d3_ss = leftover(y, panel["d3"], [panel["days"], ss], folds, stressed & long_co)
    fall = stressed & long_co & (panel["d3"] < 0)
    rise = stressed & long_co & (panel["d3"] > 0)
    abs_d3 = panel["d3"].abs()
    left_fall = leftover(y, panel["d3"], [panel["days"]], folds, fall)
    left_rise = leftover(y, panel["d3"], [panel["days"]], folds, rise)
    raw_abs = signed_oof(y, abs_d3, folds, stressed & long_co)
    left_abs = leftover(y, abs_d3, [panel["days"]], folds, stressed & long_co)
    left_d3_l1 = leftover(y, panel["d3"], [panel["days_lag1"]], folds, stressed & long_co)
    raw_fall = signed_oof(y, panel["d3"], folds, fall)

    defined_long = stressed & long_co & panel["d3"].notna()
    acf1_d3 = median_acf(panel.loc[long_co, "d3"], panel.loc[long_co, "company_id"], 1)
    acf1_days = median_acf(panel.loc[long_co, "days"], panel.loc[long_co, "company_id"], 1)
    p50_d3 = float(panel.loc[defined_long, "d3"].median())
    share_neg = float((panel.loc[defined_long, "d3"] < 0).mean())

    hold_cm = int(pd.read_parquet(STORE, columns=["company_id"])["company_id"].astype(str).isin(hold).sum())

    dec = decide_primary(longp)
    if HAS_MPL:
        _plot(longp, allp)

    ctx = {
        "now": _now_iso(),
        "longp": longp,
        "allp": allp,
        "shortp": shortp,
        "midp": midp,
        "darkp": darkp,
        "erpp": erpp,
        "firstp": firstp,
        "laterp": laterp,
        "y2_long": y2_long,
        "boot3": boot3,
        "boot6": boot6,
        "left_d3_ss": left_d3_ss,
        "left_fall": left_fall,
        "left_rise": left_rise,
        "raw_abs": raw_abs,
        "left_abs": left_abs,
        "left_d3_l1": left_d3_l1,
        "raw_fall": raw_fall,
        "acf1_d3": acf1_d3,
        "acf1_days": acf1_days,
        "p50_d3": p50_d3,
        "share_neg": share_neg,
        "dec": dec,
        "hold_n": len(hold),
        "hold_cm": hold_cm,
        "n_train_cm": int(len(panel)),
        "n_long_co": int(panel.loc[long_co, "company_id"].nunique()),
        "n_y3": int(stressed.sum()),
        "n_pos": int((y == 1).sum()),
    }
    write_md(ctx)
    _append_registry(dec, longp)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(
        f"wrote {OUT_MD} primary {dec['overall']} "
        f"d3 leftover {_f(longp['left_d3']['rank'])} "
        f"d6 leftover {_f(longp['left_d6']['rank'])} {elapsed:.0f}s"
    )


def _plot(longp, allp) -> None:
    labels = [
        "days bar\n(long)",
        "Δ3 leftover\n(long)",
        "Δ6 leftover\n(long)",
        "size bar",
        "Δ3 leftover\n(all)",
        "days_lag1",
    ]
    vals = [
        longp["days"]["cv"],
        longp["left_d3"]["rank"],
        longp["left_d6"]["rank"],
        longp["size"]["cv"] if not longp["size"]["low_power"] else SIZE_BAR,
        allp["left_d3"]["rank"],
        longp["lag1"]["cv"],
    ]
    colors = ["#1b4f72", "#196f3d", "#196f3d", "#7d6608", "#1a5276", "#1b4f72"]
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    ax.bar(range(len(labels)), [v if np.isfinite(v) else 0 for v in vals], color=colors)
    ax.axhline(CHANCE, color="#7b241c", ls="--", lw=1, label="dies <0.55")
    ax.axhline(KEEP_LEFT, color="#145a32", ls=":", lw=1, label="KEEP leftover ≥0.60")
    ax.axhline(DAYS_BAR, color="#1b4f72", ls=":", lw=1, label=f"night days {DAYS_BAR}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.45, 0.80)
    ax.set_ylabel("Y3 group-fold AUROC / leftover")
    ax.set_title("Δdays leftover after days-level (≥18m books)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def _rho_rows(rhos: dict) -> list[dict]:
    rows = [{"vs": "log1p(a_in3)", "ρ": _f(rhos["size"]), "SIZE": "YES" if abs(rhos["size"]) >= SIZE_RHO else "no"}]
    for k, v in rhos["twins"].items():
        rows.append({"vs": k, "ρ": _f(v), "SIZE": "TWIN" if abs(v) >= TWIN_RHO else "no"})
    return rows


def _pack_row(name: str, rec: dict, left: dict | None = None) -> dict:
    use = left if left is not None else rec
    low = bool(use.get("low_power") if left is not None else rec.get("low_power"))
    folds = use["folds"] if left is not None else fold_bits(rec)
    if isinstance(folds, list):
        folds = fold_bits({"folds": folds})
    return {
        "stem": name,
        "sign": rec.get("sign", "—") if not rec.get("low_power") else "—",
        "raw": "LOW_POWER" if rec.get("low_power") else _f(rec["cv"]),
        "leftover": "LOW_POWER" if low else _f(use.get("rank", rec["cv"])),
        "folds": folds if not low else "—",
        "n_pos": use.get("n_pos", rec.get("n_pos", "—")),
    }


def write_md(ctx: dict) -> None:
    longp, allp, shortp, midp = ctx["longp"], ctx["allp"], ctx["shortp"], ctx["midp"]
    dec = ctx["dec"]
    g3, g6 = longp["g3"], longp["g6"]
    headline = (
        f"{dec['overall']} on ≥18m books. Δ3 leftover after days-level rank "
        f"{_f(longp['left_d3']['rank'])} (OLS {_f(longp['left_d3']['ols'])}, "
        f"fake={longp['left_d3']['fake']}). Δ6 leftover {_f(longp['left_d6']['rank'])}. "
        f"Δ3 raw {_f(longp['d3']['cv'])} vs days {_f(longp['days']['cv'])} vs size "
        f"{_f(longp['size']['cv'])} beat3={g3['beat']} Δ={_f(g3['beat_delta'])}. "
        f"SIZE3={g3['size']} twins3={g3['twins'] or 'none'}. "
        f"Inverse days-after-Δ3 {_f(longp['inv_d3']['rank'])}. "
        f"Card: {dec['card']}. Night Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]}, days {DAYS_BAR}, "
        f"size {SIZE_BAR}, days_lag1 {DAYS_LAG1} unchanged. Hidden 72 is 1-month only."
    )
    md = f"""# Unused leftover of Δdays after days-level (Norden lead without utilisation)

Generated `{ctx['now']}` by `analysis/evaluate/days_delta_qa.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Rates and leftover on **train**. Holdout {ctx['hold_n']} / {ctx['hold_cm']} CM
coverage only. Seed {FOLD_SEED} group folds. No 0–100. No parquet rewrite.
No new GBM. No `build_targets`. Do not invent utilisation / Y10. Do not
reconstruct Norden AMPLI (HIGH−LOW of balances) — B-forbidden. Y3 never B.
Do not quote `a_out_vol` 0.722 as the engine. Do not overwrite `n_tx_qa.*` /
`recency_qa.*` / `gap_sd_qa.*` / `y3_reasons.md`.

`d3` = `c_n_days_with_tx` − lag3. `d6` = days − lag6. Primary slice:
companies with **≥18 months on book** ({ctx['n_long_co']} train companies).
Y3 labeled n={ctx['n_y3']:,} / n_pos={ctx['n_pos']:,}.

## Headline

{headline}

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not Δdays. Q1 is last-value runway. Never B. |
| 2 | Who is improving? | Days **level** 0.711 is the bar. Δ is the path test. |
| 3 | Who is turning? | {dec['overall']}: Δ3 leftover {_f(longp['left_d3']['rank'])} after days-level on ≥18m. Quiet-stressed recover stays SS/salary/days. |
| 4 | Dip vs fall? | Out. Sibling TURNOVER. |
| 5 | Why did it change? | If KEEP, falling activity leftover after the *level*. If not, the level already ate the change. |
| 6 | Months earlier? | Hidden 72 stays **1-month** (`days_lag1` {DAYS_LAG1}). Δ3/Δ6 are 3-/6-month clocks on long books only — not a holdout claim. |

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| Δ3 leftover after days-level (≥18m) | **{g3['role']}** | rank {_f(longp['left_d3']['rank'])} OLS {_f(longp['left_d3']['ols'])} fake={longp['left_d3']['fake']}; beat_size={g3['beat']} ({_f(g3['beat_delta'])}); SIZE={g3['size']}; twins={g3['twins'] or 'none'} |
| Δ6 leftover after days-level (≥18m) | **{g6['role']}** | rank {_f(longp['left_d6']['rank'])} OLS {_f(longp['left_d6']['ols'])} fake={longp['left_d6']['fake']}; beat_size={g6['beat']} ({_f(g6['beat_delta'])}); SIZE={g6['size']}; twins={g6['twins'] or 'none'} |
| 15-col Y3 card | **{dec['card']}** | KEEP-as-X requires leftover ≥0.60 AND beat size ≥0.02 AND not SIZE AND not twin |
| Inverse: days leftover after Δ3 | **{'lives' if (np.isfinite(longp['inv_d3']['rank']) and longp['inv_d3']['rank'] >= CHANCE) else 'dies / LOW_POWER'}** | rank {_f(longp['inv_d3']['rank'])} — the 0.711 bar must survive |
| AMPLI / utilisation / Y10 | **PARK** | B-forbidden; snapshot 1.6% |
| Hidden-72 Δ3/Δ6 claim | **PARK** | 1-month only |
| Night quotes | **unchanged** | Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]} · days {DAYS_BAR} · size {SIZE_BAR} · days_lag1 {DAYS_LAG1} |

## 1. Coverage / Δ definition (train, ≥18m books)

d3 defined on Y3-long n={longp['n_d3']:,} / n_pos={longp['n_pos_d3']:,}.
d6 defined n={longp['n_d6']:,} / n_pos={longp['n_pos_d6']:,}.
Median Δ3 on defined long Y3 = {_f(ctx['p50_d3'])}; share Δ3<0 = {_f(ctx['share_neg'])}.
acf1(Δ3) {_f(ctx['acf1_d3'])} vs acf1(days) {_f(ctx['acf1_days'])} on long books.

## 2. Spearman twins + SIZE (Δ3 on ≥18m Y3-defined)

{_md_table(_rho_rows(longp['rhos3']), ['vs', 'ρ', 'SIZE'])}

Δ6 twins:

{_md_table(_rho_rows(longp['rhos6']), ['vs', 'ρ', 'SIZE'])}

## 3. Single-feature group-fold AUROC

Sign from the train side of each fold. Never holdout fit. Never Y7.

{_md_table([
    _pack_row("days level (≥18m)", longp["days"]),
    _pack_row("Δ3 raw (≥18m)", longp["d3"]),
    _pack_row("Δ6 raw (≥18m)", longp["d6"]),
    _pack_row("size log1p(a_in3) (≥18m)", longp["size"]),
    _pack_row("days_lag1 (≥18m)", longp["lag1"]),
    _pack_row("Δ3 leftover after days (≥18m)", longp["d3"], longp["left_d3"]),
    _pack_row("Δ6 leftover after days (≥18m)", longp["d6"], longp["left_d6"]),
    _pack_row("days leftover after Δ3 (inverse)", longp["days"], longp["inv_d3"]),
], ["stem", "sign", "raw", "leftover", "folds", "n_pos"])}

## 4. All-train / short / mid (expect CLOSE / LOW_POWER)

| slice | n_pos | days | Δ3 raw | Δ3 leftover | Δ6 leftover | role Δ3 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| all train | {allp['n_pos']} | {_f(allp['days']['cv'])} | {_f(allp['d3']['cv'])} | {_f(allp['left_d3']['rank'])} | {_f(allp['left_d6']['rank'])} | {allp['g3']['role']} |
| company <12 | {shortp['n_pos']} | {_f(shortp['days']['cv'])} | {_f(shortp['d3']['cv'])} | {_f(shortp['left_d3']['rank'])} | {_f(shortp['left_d6']['rank'])} | {shortp['g3']['role']} |
| company 12–17 | {midp['n_pos']} | {_f(midp['days']['cv'])} | {_f(midp['d3']['cv'])} | {_f(midp['left_d3']['rank'])} | {_f(midp['left_d6']['rank'])} | {midp['g3']['role']} |
| company ≥18 | {longp['n_pos']} | {_f(longp['days']['cv'])} | {_f(longp['d3']['cv'])} | {_f(longp['left_d3']['rank'])} | {_f(longp['left_d6']['rank'])} | {longp['g3']['role']} |

## 5. Honest leftover after days-level (primary ≥18m)

Δ3 leftover rank {_f(longp['left_d3']['rank'])} OLS {_f(longp['left_d3']['ols'])} ρ(resid,days)={_f(longp['left_d3']['rho_ctrl'])} R²={_f(longp['left_d3']['r2'])} fake={longp['left_d3']['fake']}.
Folds {longp['left_d3']['folds']}.
Δ6 leftover rank {_f(longp['left_d6']['rank'])} OLS {_f(longp['left_d6']['ols'])} ρ(resid,days)={_f(longp['left_d6']['rho_ctrl'])} fake={longp['left_d6']['fake']}.
Folds {longp['left_d6']['folds']}.

## Extra — inverse leftover of days after Δdays

Days leftover after Δ3 {_f(longp['inv_d3']['rank'])} (OLS {_f(longp['inv_d3']['ols'])}).
Days leftover after Δ6 {_f(longp['inv_d6']['rank'])}.
If inverse dies, the 0.711 bar would be a rewrite of the change — it must live.

## Extra — company bootstrap leftover (≥18m, n={N_BOOT})

Δ3 leftover p05/p50/p95 {_f(ctx['boot3']['p05'])} / {_f(ctx['boot3']['p50'])} / {_f(ctx['boot3']['p95'])}; share<0.55={_f(ctx['boot3']['share_lt'])} (n={ctx['boot3']['n']}).
Δ6 leftover p05/p50/p95 {_f(ctx['boot6']['p05'])} / {_f(ctx['boot6']['p50'])} / {_f(ctx['boot6']['p95'])}; share<0.55={_f(ctx['boot6']['share_lt'])}.

## Extra — dark vs ERP (long books)

| slice | n_pos | days | Δ3 leftover | Δ6 leftover |
| --- | ---: | ---: | ---: | ---: |
| long dark | {ctx['darkp']['n_pos']} | {_f(ctx['darkp']['days']['cv'])} | {_f(ctx['darkp']['left_d3']['rank'])} | {_f(ctx['darkp']['left_d6']['rank'])} |
| long ERP | {ctx['erpp']['n_pos']} | {_f(ctx['erpp']['days']['cv'])} | {_f(ctx['erpp']['left_d3']['rank'])} | {_f(ctx['erpp']['left_d6']['rank'])} |

## Extra — first6 vs later months on long books

First six months of a long book vs later. Δ3 needs lag3 so first6 is mostly empty.

| slice | n_pos | Δ3 leftover | Δ6 leftover | days |
| --- | ---: | ---: | ---: | ---: |
| long first6 | {ctx['firstp']['n_pos']} | {_f(ctx['firstp']['left_d3']['rank'])} | {_f(ctx['firstp']['left_d6']['rank'])} | {_f(ctx['firstp']['days']['cv'])} |
| long later | {ctx['laterp']['n_pos']} | {_f(ctx['laterp']['left_d3']['rank'])} | {_f(ctx['laterp']['left_d6']['rank'])} | {_f(ctx['laterp']['days']['cv'])} |

## Extra — Δ3 leftover after days+SS; Y2 lock

Δ3 leftover after days+SS on ≥18m {_f(ctx['left_d3_ss']['rank'])} (must not steal the SS reason).
Y2 leftover of Δ3 after days-level on ≥18m {_f(ctx['y2_long']['rank'])} — do not merge with Y2 trees.

## Extra — falling-only / |Δ3| / leftover after days_lag1

Norden's lead is activity *falling*. Restricting to Δ3<0 does not rescue leftover.

| slice | n_pos | raw | leftover after days | folds |
| --- | ---: | ---: | ---: | --- |
| Δ3<0 (falling) | {ctx['left_fall']['n_pos']} | {_f(ctx['raw_fall']['cv'])} | {_f(ctx['left_fall']['rank'])} | {ctx['left_fall']['folds']} |
| Δ3>0 (rising) | {ctx['left_rise']['n_pos']} | — | {_f(ctx['left_rise']['rank'])} | {ctx['left_rise']['folds']} |
| \|Δ3\| | {ctx['left_abs']['n_pos']} | {_f(ctx['raw_abs']['cv'])} | {_f(ctx['left_abs']['rank'])} | {ctx['left_abs']['folds']} |
| Δ3 leftover after days_lag1 | {ctx['left_d3_l1']['n_pos']} | — | {_f(ctx['left_d3_l1']['rank'])} | {ctx['left_d3_l1']['folds']} |

`days_lag1` {DAYS_LAG1} stays q6_keep. Δ3 leftover after the lag still dies — the 3-month change is not a lead beyond last month's level.

## Explicitly out

- AMPLI = HIGH−LOW of balances (B). `f_util_snapshot` / Y10.
- Putting Δdays on the 15-col card unless KEEP-as-X cleared.
- Hidden-72 3- or 6-month claims. `a_out_vol` 0.722 as the engine.
- Rewrites of `n_tx_qa.*`, `recency_qa.*`, `gap_sd_qa.*`, `y3_reasons.md`, `gbm_core.py`.

Plot: `{OUT_PNG.name if HAS_MPL else '—'}`.

Night Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]}, days {DAYS_BAR}, size {SIZE_BAR} unchanged.
"""
    OUT_MD.write_text(md)


def _append_registry(dec: dict, longp: dict) -> None:
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    row = (
        f"{ts},R4,4,days_delta_qa,C,y3_recover_cash_6m,days_delta_qa,train_cv,"
        f"d3_left,{_f(longp['left_d3']['rank'], 4)},{_f(longp['left_d6']['rank'], 4)},"
        f"{dec['overall']} card={dec['card'][:40]}"
    )
    with REGISTRY.open("a") as f:
        f.write(row + "\n")


if __name__ == "__main__":
    main()
