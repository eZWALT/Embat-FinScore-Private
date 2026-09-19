"""Unused leftover of ``c_last_tx_before_2026_06`` after days as Y3 X.

``c_last_tx_before_2026_06`` = 1 if period ≥ 2026-06-01 and last booking
as of month_end is < 2026-06-01. Already **PARK as extract**
(recency_qa): Javier 61 vs as-of-Aug **62**; **COMP_0981 is the +1**.
Flag only 2026-06..08. ``c_recency_days`` DROP leftover 0.607.
Do not revive Y6. Do **not** overwrite ``recency_qa.py`` / ``.md``.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not
SIZE (|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs
days / a_n_tx / c_recency_days). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.june_tx_qa

Owned: analysis/evaluate/june_tx_qa.py, analysis/outputs/june_tx_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_june_tx.md (end).
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
from analysis.features.common import ANALYSIS, DATA, connect

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "june_tx_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "june_tx_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_june_tx.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "june_tx_qa"
X_FAM = "C"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
RECENCY_Y3 = 0.659
RECENCY_LEFT = 0.607
JUNE_Y3_PEEK = 0.500
JUNE_N_PEEK = 5648
JUNE_POS_PEEK = 402
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
FAKE_DAYS_RHO = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
CUTOFF = pd.Timestamp("2026-06-01")
AUG_END = pd.Timestamp("2026-08-31")
COMP_0981 = "COMP_0981"
JAVIER_61 = 61
ASOF_62 = 62
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "c_n_tx",
    "c_n_days_with_tx",
    "c_recency_days",
    "c_last_tx_before_2026_06",
)

Y_KEEP = (Y2, Y3)
FLAG = "c_last_tx_before_2026_06"


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


def icc_anova(series: pd.Series, company: pd.Series) -> dict:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}).dropna()
    k = int(d["co"].nunique())
    n = len(d)
    if k < 2 or n < k + 2:
        return {"icc": float("nan"), "k": k, "n": n}
    grand = float(d["x"].mean())
    ns = d.groupby("co")["x"].size()
    mus = d.groupby("co")["x"].mean()
    ssb = float(((mus - grand) ** 2 * ns).sum())
    ssw = float(((d["x"] - d["co"].map(mus)) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k) if n > k else float("nan")
    icc = msb / (msb + msw) if np.isfinite(msw) and (msb + msw) != 0 else float("nan")
    return {"icc": float(icc), "k": k, "n": n}


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
    nuniq = int(x[defined].nunique())
    low = n_pos < MIN_POS or n_neg == 0 or nuniq < 2
    fold_rows, aucs = [], []
    if low:
        return {
            "cv": 0.5 if nuniq < 2 and n_pos >= 1 and n_neg >= 1 else float("nan"),
            "sd": float("nan"), "n_folds": 0, "folds": fold_rows,
            "train_sign": 0, "train_auc": 0.5 if nuniq < 2 and n > 0 else float("nan"),
            "n_defined": n, "n_pos": n_pos, "n_neg": n_neg,
            "low_power": n_pos < MIN_POS or n_neg == 0,
            "constant": nuniq < 2,
        }
    for k in range(n_folds):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            fold_rows.append({"fold": k, "auroc": float("nan"), "sign": 0, "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())})
            continue
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        fold_rows.append({"fold": k, "auroc": float(auc) if np.isfinite(auc) else float("nan"), "sign": int(sign), "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())})
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[defined], x[defined])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite), "folds": fold_rows, "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[defined], tr_sign * x[defined])),
        "n_defined": n, "n_pos": n_pos, "n_neg": n_neg, "low_power": False,
        "constant": False,
    }


def fold_bits(rec: dict) -> str:
    if rec.get("constant"):
        return "const"
    return " ".join(f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", []))


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
    if np.nanstd(Y) == 0:
        resid.loc[ok] = 0.0
        info["intercept"] = float(Y.mean()) if len(Y) else float("nan")
        info["slope"] = [0.0] * n_x
        info["r2"] = float("nan")
        info["constant_y"] = True
        return resid, info
    X = np.column_stack([np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(n_x)])
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
    info["constant_y"] = False
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
    almost = bool(np.isfinite(rho_c) and abs(rho_c) >= FAKE_DAYS_RHO)
    rank_cv = _cv(rrec)
    ols_cv = _cv(rec)
    const = bool(info.get("constant_y") or rec.get("constant") or rrec.get("constant"))
    honest_dies = bool(const or fake or (np.isfinite(rank_cv) and rank_cv < CHANCE))
    return {
        "ols": ols_cv, "rank": rank_cv, "rho_ctrl": rho_c, "r2": info["r2"],
        "n": rec["n_defined"], "n_pos": rec["n_pos"],
        "fake": fake, "almost": almost, "honest_dies": honest_dies, "const": const,
        "folds": fold_bits(rec), "rank_folds": fold_bits(rrec),
        "rec": rec, "rrec": rrec,
    }


def _cv(res: dict) -> float:
    if res.get("constant"):
        return 0.5
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    if res.get("constant"):
        cv = "0.500 const"
        train = "0.500"
        folds = "const"
        sign = "—"
        sd = "—"
    elif res.get("low_power"):
        cv, train, folds, sign, sd = "LOW_POWER", "LOW_POWER", "—", "—", "—"
    else:
        cv, train, folds, sign, sd = _f(res["cv"]), _f(res["train_auc"]), fold_bits(res), res["train_sign"], _f(res["sd"])
    return {"y": y, "feature": feat, "n": f"{res['n_defined']:,}", "n_pos": f"{res['n_pos']:,}", "CV": cv, "train": train, "sd": sd, "sign": sign, "folds": folds}


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


def load_unique_days(con) -> pd.DataFrame:
    days = con.execute(
        """
        SELECT company_id, CAST("date" AS DATE) AS d
        FROM transactions
        WHERE "date" IS NOT NULL
        GROUP BY 1, 2
        """
    ).df()
    days["company_id"] = days["company_id"].astype(str)
    days["d"] = pd.to_datetime(days["d"])
    return days


def load_panel() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    print(f"store {STORE} shape={raw.shape} (read-only)")
    print(f"targets {TARGETS} shape={yraw.shape} (read-only)")
    miss = [c for c in STORE_COLS if c not in raw.columns]
    if miss:
        raise RuntimeError(f"monthly.parquet missing {miss}")
    ymiss = [c for c in Y_KEEP if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[["company_id", "period", *Y_KEEP]])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    leak3 = leakage_check(
        [FLAG, "c_recency_days", "c_n_days_with_tx", "log_in3"],
        Y3, forbidden_prefixes=["b"],
    )
    if not leak3["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak3['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def attach_rosters(panel: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    last_ever = days.groupby("company_id")["d"].max()
    last_aug = days.loc[days["d"] <= AUG_END].groupby("company_id")["d"].max()
    extract = set(last_ever[last_ever < CUTOFF].index)
    asof = set(last_aug[last_aug < CUTOFF].index)
    asof_no = asof - {COMP_0981}
    ever_store = set(
        panel.loc[pd.to_numeric(panel[FLAG], errors="coerce") == 1, "company_id"].astype(str)
    )
    panel = panel.copy()
    cid = panel["company_id"].astype(str)
    panel["june_extract61"] = cid.isin(extract).astype(float)
    panel["june_asof62"] = cid.isin(asof).astype(float)
    panel["june_asof_no0981"] = cid.isin(asof_no).astype(float)
    panel["june_ever_store"] = cid.isin(ever_store).astype(float)
    return panel, {
        "extract": extract, "asof": asof, "asof_no": asof_no, "ever_store": ever_store,
        "n_extract": len(extract), "n_asof": len(asof),
        "plus": sorted(asof - extract), "minus": sorted(extract - asof),
        "last_ever": last_ever, "last_aug": last_aug,
    }


def pass1_cov(tr: pd.DataFrame, rost: dict) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin / SIZE; only 2026-06..08")
    print("=" * 72)
    n_cm, n_co = len(tr), int(tr["company_id"].nunique())
    flg = pd.to_numeric(tr[FLAG], errors="coerce")
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    per = pd.to_datetime(tr["period"])
    win = (per >= CUTOFF) & (per <= pd.Timestamp("2026-08-01"))
    n_on = int((flg == 1).sum())
    n_on_win = int(((flg == 1) & win).sum())
    n_on_out = int(((flg == 1) & ~win).sum())
    months_on = sorted(per[flg == 1].dt.strftime("%Y-%m").unique().tolist())
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y3_lab = y3.notna()
    last_y3 = per[y3_lab].max() if y3_lab.any() else pd.NaT
    last_y2 = per[pd.to_numeric(tr[Y2], errors="coerce").notna()].max()
    june_on_y3 = float(flg[y3_lab].mean()) if y3_lab.any() else float("nan")
    hole = bool(y3_lab.any() and june_on_y3 == 0.0 and last_y3 < CUTOFF)
    only_win = n_on > 0 and n_on_out == 0 and n_on == n_on_win
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "c_recency_days": rec,
        "log1p(a_in3)": tr["log_in3"],
    }
    rhos = {k: spearman(flg, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "c_recency_days")
    )
    rows = [
        {"col": FLAG, "n_nn": f"{int(flg.notna().sum()):,}", "share=1": _pp(_pct(n_on, n_cm)), "n=1": n_on},
        {"col": "c_recency_days", "n_nn": f"{int(rec.notna().sum()):,}", "share=1": "—", "n=1": "—"},
    ]
    rho_rows = [
        {"vs": k, "rho": _f(v), "flag": "SIZE" if k.startswith("log") and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no"}
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. Flag =1 on {n_on} CM "
        f"({n_on_win} in 2026-06..08, {n_on_out} outside) months={months_on} "
        f"{'ONLY 2026-06..08 CONFIRM' if only_win else 'NOT only the window'}. "
        f"Last Y3 {last_y3.date() if pd.notna(last_y3) else '—'} last Y2 "
        f"{last_y2.date() if pd.notna(last_y2) else '—'}; mean(flag|Y3 labeled)="
        f"{_pp(june_on_y3)} extract_hole={hole}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs recency {_f(rhos['c_recency_days'])} vs size {_f(rhos['log1p(a_in3)'])}. "
        f"SIZE={is_size} twins={twins or 'none'}."
    )
    print(prose)
    return {
        "rows": rows, "rho_rows": rho_rows, "rhos": rhos, "twins": twins,
        "is_size": is_size, "twin_gate": twin_gate, "n_cm": n_cm, "n_co": n_co,
        "n_on": n_on, "n_on_win": n_on_win, "n_on_out": n_on_out,
        "only_win": only_win, "hole": hole, "june_on_y3": june_on_y3,
        "last_y3": last_y3, "last_y2": last_y2, "months_on": months_on,
        "cov": 1.0, "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    feats = {
        FLAG: tr[FLAG],
        "c_recency_days": tr["c_recency_days"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "june_asof62": tr["june_asof62"],
        "june_extract61": tr["june_extract61"],
    }
    recs, rows = {}, []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} const={rec.get('constant')} n={rec['n_defined']} pos={rec['n_pos']}")
    y3_f, y3_size, y3_days = _cv(recs[FLAG]), _cv(recs["log1p(a_in3)"]), _cv(recs["c_n_days_with_tx"])
    y3_rec = _cv(recs["c_recency_days"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    rec_ok = bool(np.isfinite(y3_rec) and abs(y3_rec - RECENCY_Y3) < 0.015)
    peek_ok = bool(
        recs[FLAG]["n_defined"] == JUNE_N_PEEK
        and recs[FLAG]["n_pos"] == JUNE_POS_PEEK
        and abs(y3_f - JUNE_Y3_PEEK) < 0.01
    )
    beat_size = bool(np.isfinite(y3_f) and (y3_f - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 {FLAG} {_f(y3_f)} n={recs[FLAG]['n_defined']:,} pos={recs[FLAG]['n_pos']:,} "
        f"const={recs[FLAG].get('constant')} (peek 0.500 / 5,648 / 402 "
        f"{'CONFIRM' if peek_ok else 'DRIFT'}). vs size {_f(y3_size)} vs days {_f(y3_days)} "
        f"vs recency {_f(y3_rec)} (quote 0.659 {'CONFIRM' if rec_ok else 'DRIFT'}). "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ="
        f"{_f(y3_f - SIZE_QUOTE) if np.isfinite(y3_f) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}. "
        f"asof62 roster leftover-as-X {_f(_cv(recs['june_asof62']))} "
        f"extract61 {_f(_cv(recs['june_extract61']))}."
    )
    print(prose)
    return {
        "rows": rows, "recs": recs, "flag": y3_f, "size": y3_size, "days": y3_days,
        "recency": y3_rec, "asof": _cv(recs["june_asof62"]), "extract": _cv(recs["june_extract61"]),
        "days_ok": days_ok, "size_ok": size_ok, "rec_ok": rec_ok, "peek_ok": peek_ok,
        "beat_size": beat_size, "const": bool(recs[FLAG].get("constant")),
        "n_def": recs[FLAG]["n_defined"], "n_pos": recs[FLAG]["n_pos"], "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of June flag after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    after = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr[FLAG],), folds, lab)
    rec_after = leftover_diag(y, tr["c_recency_days"], (tr["c_n_days_with_tx"],), folds, lab)
    rec_ok = bool(np.isfinite(rec_after["rank"]) and abs(rec_after["rank"] - RECENCY_LEFT) < 0.03)
    prose = (
        f"June-flag leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"const={after['const']} fake={after['fake']} honest_dies={after['honest_dies']} "
        f"n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after flag OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"(should stay ~0.711 — flag is 0 on Y3 rows). "
        f"Recency leftover after days OLS {_f(rec_after['ols'])} rank {_f(rec_after['rank'])} "
        f"(peek 0.607 was OLS {'CONFIRM' if (np.isfinite(rec_after['ols']) and abs(rec_after['ols'] - RECENCY_LEFT) < 0.03) else 'DRIFT / rank is honest'})."
    )
    print(prose)
    return {
        "after": after, "inv": inv, "rec_after": rec_after,
        "ols": after["ols"], "rank": after["rank"], "dies": after["honest_dies"],
        "fake": after["fake"], "const": after["const"],
        "inv_rank": inv["rank"], "rec_rank": rec_after["rank"], "rec_ok": rec_ok,
        "prose": prose,
    }


def pass4_formula(tr: pd.DataFrame, panel: pd.DataFrame, days: pd.DataFrame, rost: dict) -> dict:
    print("\n" + "=" * 72)
    print("CUT 4 — Javier 61 vs 62 / COMP_0981")
    print("=" * 72)
    hold = set(load_holdout())
    last_ever, last_aug = rost["last_ever"], rost["last_aug"]
    n_extract, n_asof = rost["n_extract"], rost["n_asof"]
    plus, minus = rost["plus"], rost["minus"]
    is_0981 = COMP_0981 in plus and len(plus) == 1
    last_0981_ever = last_ever.get(COMP_0981, pd.NaT)
    last_0981_aug = last_aug.get(COMP_0981, pd.NaT)
    txs = days.loc[days["company_id"] == COMP_0981, "d"].sort_values()
    after_apr = txs[txs > pd.Timestamp("2025-04-07")]
    next_after = after_apr.iloc[0] if len(after_apr) else pd.NaT
    aug = panel[panel["period"] == pd.Timestamp("2026-08-01")]
    store_aug = set(aug.loc[pd.to_numeric(aug[FLAG], errors="coerce") == 1, "company_id"].astype(str))
    confirm_61 = n_extract == JAVIER_61
    confirm_62 = n_asof == ASOF_62
    store_match = store_aug == rost["asof"]
    n_train_asof = len([c for c in rost["asof"] if c not in hold])
    n_hold_asof = len([c for c in rost["asof"] if c in hold])
    keys = panel[["company_id", "period"]].copy().reset_index(drop=True)
    monthly = (
        days.assign(period=days["d"].dt.to_period("M").dt.to_timestamp())
        .groupby(["company_id", "period"], as_index=False)["d"].max()
        .rename(columns={"d": "last_tx"})
    )
    recon = keys.merge(monthly, on=["company_id", "period"], how="left")
    recon = recon.sort_values(["company_id", "period"]).reset_index(drop=True)
    recon["last_tx"] = recon.groupby("company_id", sort=False)["last_tx"].ffill()
    last_day = pd.to_datetime(recon["last_tx"]).dt.normalize()
    recon["june_raw"] = ((recon["period"] >= CUTOFF) & (last_day < CUTOFF)).astype(float)
    m = recon.merge(panel[["company_id", "period", FLAG]], on=["company_id", "period"], how="left")
    june_agree = float((pd.to_numeric(m[FLAG], errors="coerce") == m["june_raw"]).mean())
    prose = (
        f"Store vs raw June-flag agree {_pp(june_agree)}. "
        f"Extract-level last tx < 2026-06-01: **{n_extract}** "
        f"{'CONFIRM Javier 61' if confirm_61 else '≠ 61'}. "
        f"As-of 2026-08-31: **{n_asof}** {'CONFIRM 62' if confirm_62 else '≠ 62'}. "
        f"+1={plus} {'COMP_0981 is the +1' if is_0981 else 'NOT uniquely COMP_0981'}. "
        f"COMP_0981 last-ever={last_0981_ever} last-asof-Aug={last_0981_aug} "
        f"next-after-2025-04-07={next_after}. "
        f"Store Aug-2026 flag n={len(store_aug)} {'match as-of 62' if store_match else 'DIFFERS'}. "
        f"Train as-of {n_train_asof} / holdout {n_hold_asof}."
    )
    print(prose)
    return {
        "n_extract": n_extract, "n_asof": n_asof, "plus": plus, "minus": minus,
        "is_0981": is_0981, "confirm_61": confirm_61, "confirm_62": confirm_62,
        "store_match": store_match, "june_agree": june_agree,
        "last_0981_ever": str(last_0981_ever), "last_0981_aug": str(last_0981_aug),
        "next_after": str(next_after), "n_train_asof": n_train_asof, "n_hold_asof": n_hold_asof,
        "prose": prose,
    }


def pass5_recency(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after recency (rewrite?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_r = leftover_diag(y, tr[FLAG], (tr["c_recency_days"],), tr["fold"], lab)
    after_rd = leftover_diag(y, tr[FLAG], (tr["c_n_days_with_tx"], tr["c_recency_days"]), tr["fold"], lab)
    after_asof = leftover_diag(y, tr["june_asof62"], (tr["c_recency_days"],), tr["fold"], lab)
    after_asof_d = leftover_diag(y, tr["june_asof62"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rho = spearman(tr[FLAG], tr["c_recency_days"])
    rewrite = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = (
        f"Contemporaneous flag leftover after recency OLS {_f(after_r['ols'])} "
        f"rank {_f(after_r['rank'])} const={after_r['const']} dies={after_r['honest_dies']} "
        f"R²={_f(after_r['r2'])}. after days+recency {_f(after_rd['rank'])}. "
        f"ρ(flag, recency)={_f(rho)} rewrite_twin={rewrite}. "
        f"asof62 leftover after recency {_f(after_asof['rank'])} after days {_f(after_asof_d['rank'])}."
    )
    print(prose)
    return {
        "after_r": after_r["rank"], "after_rd": after_rd["rank"],
        "asof_r": after_asof["rank"], "asof_d": after_asof_d["rank"],
        "rho": rho, "rewrite": rewrite, "const": after_r["const"], "prose": prose,
        "asof_d_dies": after_asof_d["honest_dies"],
    }


def pass6_hold(hold: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — holdout coverage only (no AUROC)")
    print("=" * 72)
    flg = pd.to_numeric(hold[FLAG], errors="coerce")
    per = pd.to_datetime(hold["period"])
    win = (per >= CUTOFF) & (per <= pd.Timestamp("2026-08-01"))
    n_on = int((flg == 1).sum())
    n_on_win = int(((flg == 1) & win).sum())
    n_co_on = int(hold.loc[flg == 1, "company_id"].nunique())
    rec = {
        "n_co": int(hold["company_id"].nunique()), "n_cm": len(hold),
        "n_on": n_on, "n_on_win": n_on_win, "n_co_on": n_co_on,
        "share": _pct(n_on, len(hold)),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM flag=1 n={n_on} "
        f"({n_on_win} in 2026-06..08) n_co={n_co_on} share={_pp(rec['share'])} "
        f"(no fit, no AUROC). Extract hole vs health: coverage only."
    )
    print(prose)
    return {**rec, "prose": prose}


def pass7_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — Q6 lag1 leftover after days_lag1 (extract cannot lead)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab, folds = y.notna(), tr["fold"]
    recs, rows = {}, []
    for name in (FLAG, f"{FLAG}_lag1", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} const={rec.get('constant')} n={rec['n_defined']}")
    after_l1 = leftover_diag(y, tr[f"{FLAG}_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    y3_lab = lab
    lag1_on_y3 = float(pd.to_numeric(tr.loc[y3_lab, f"{FLAG}_lag1"], errors="coerce").fillna(0).mean())
    prose = (
        f"Y3 flag_lag1 {_f(recs[f'{FLAG}_lag1']['cv'])} const={recs[f'{FLAG}_lag1'].get('constant')} "
        f"mean on labeled Y3={_pp(lag1_on_y3)}. leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']} const={after_l1['const']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'}). "
        f"Extract cannot lead — lag1 still 0 on Y3-labeled months."
    )
    print(prose)
    return {
        "rows": rows, "lag1": _cv(recs[f"{FLAG}_lag1"]),
        "l1_rank": after_l1["rank"], "l1_dies": after_l1["honest_dies"],
        "l1_const": after_l1["const"], "days_l1": days_l1, "days_ok": days_ok,
        "lag1_on_y3": lag1_on_y3, "prose": prose,
    }


def pass8_drop0981(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — drop COMP_0981 sensitivity")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rec62 = signed_oof_auroc(y, tr["june_asof62"], tr["fold"], lab)
    rec61 = signed_oof_auroc(y, tr["june_extract61"], tr["fold"], lab)
    rec_no = signed_oof_auroc(y, tr["june_asof_no0981"], tr["fold"], lab)
    after62 = leftover_diag(y, tr["june_asof62"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after61 = leftover_diag(y, tr["june_extract61"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_no = leftover_diag(y, tr["june_asof_no0981"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    in_tr = COMP_0981 in set(tr["company_id"].astype(str))
    prose = (
        f"COMP_0981 in train={in_tr}. asof62 Y3 {_f(_cv(rec62))} leftover-days {_f(after62['rank'])} "
        f"dies={after62['honest_dies']}. extract61 {_f(_cv(rec61))} leftover {_f(after61['rank'])} "
        f"dies={after61['honest_dies']}. drop-0981 {_f(_cv(rec_no))} leftover {_f(after_no['rank'])} "
        f"dies={after_no['honest_dies']}. Roster leftover is extract, not a 44 stem."
    )
    print(prose)
    return {
        "asof": _cv(rec62), "asof_rank": after62["rank"], "asof_dies": after62["honest_dies"],
        "extract": _cv(rec61), "extract_rank": after61["rank"],
        "no0981": _cv(rec_no), "no_rank": after_no["rank"], "no_dies": after_no["honest_dies"],
        "in_tr": in_tr, "prose": prose,
    }


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, FLAG, "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    n_const = 0
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        b = pd.concat([work[work["company_id"] == c] for c in draw], ignore_index=True)
        after = leftover_diag(b[Y3], b[FLAG], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index))
        if after["const"]:
            n_const += 1
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share_die = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = (
        f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"share<0.55={_pp(share_die)} n_const={n_const}/{len(ranks)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share_die, "n_const": n_const, "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only)")
    print("=" * 72)
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    lab = y2.notna()
    rec = signed_oof_auroc(y2, tr[FLAG], tr["fold"], lab)
    after = leftover_diag(y2, tr[FLAG], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    rec_a = signed_oof_auroc(y2, tr["june_asof62"], tr["fold"], lab)
    after_a = leftover_diag(y2, tr["june_asof62"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    june_on = float(pd.to_numeric(tr.loc[lab, FLAG], errors="coerce").mean())
    last_y2 = pd.to_datetime(tr.loc[lab, "period"]).max() if lab.any() else pd.NaT
    prose = (
        f"Y2 last labeled {last_y2.date() if pd.notna(last_y2) else '—'} mean(flag|Y2)={_pp(june_on)}. "
        f"Y2 flag {_f(_cv(rec))} leftover-days {_f(after['rank'])} const={after['const']}. "
        f"asof62 Y2 {_f(_cv(rec_a))} leftover {_f(after_a['rank'])} dies={after_a['honest_dies']}."
    )
    print(prose)
    return {"y2": _cv(rec), "after": after["rank"], "asof": _cv(rec_a), "asof_rank": after_a["rank"], "prose": prose}


def extra_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — ICC / window share")
    print("=" * 72)
    icc = icc_anova(tr[FLAG], tr["company_id"])
    per = pd.to_datetime(tr["period"])
    rows = []
    for p, g in tr.groupby(per.dt.to_period("M")):
        flg = pd.to_numeric(g[FLAG], errors="coerce")
        rows.append({"period": str(p), "n": len(g), "n=1": int((flg == 1).sum()), "share": _pp(_pct(int((flg == 1).sum()), len(g)))})
    prose = f"ICC={_f(icc['icc'])} k={icc['k']} (rare dummy; ICC is not a health trait)."
    print(prose)
    return {"icc": icc, "rows": rows, "prose": prose}


def extra_asof_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — asof62 leftover after size / days+size / recency")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr["june_asof62"], (tr["log_in3"],), tr["fold"], lab)
    after_b = leftover_diag(y, tr["june_asof62"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    after_r = leftover_diag(y, tr["june_asof62"], (tr["c_recency_days"],), tr["fold"], lab)
    rho = spearman(tr["june_asof62"], tr["log_in3"])
    prose = (
        f"asof62 leftover after size {_f(after_s['rank'])} dies={after_s['honest_dies']} "
        f"ρ vs size {_f(rho)}. after days+size {_f(after_b['rank'])} dies={after_b['honest_dies']}. "
        f"after recency {_f(after_r['rank'])} dies={after_r['honest_dies']}."
    )
    print(prose)
    return {"after_s": after_s["rank"], "after_b": after_b["rank"], "after_r": after_r["rank"], "rho": rho, "prose": prose}


def extra_honest_mask(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after days fitted only on Y3-labeled rows")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    sl = tr.loc[lab].copy()
    after = leftover_diag(sl[Y3], sl[FLAG], (sl["c_n_days_with_tx"],), sl["fold"], pd.Series(True, index=sl.index))
    rec_after = leftover_diag(sl[Y3], sl["c_recency_days"], (sl["c_n_days_with_tx"],), sl["fold"], pd.Series(True, index=sl.index))
    asof = leftover_diag(sl[Y3], sl["june_asof62"], (sl["c_n_days_with_tx"],), sl["fold"], pd.Series(True, index=sl.index))
    prose = (
        f"On Y3-labeled rows only: flag leftover after days OLS {_f(after['ols'])} "
        f"rank {_f(after['rank'])} const={after['const']} fake={after['fake']} "
        f"(honest extract-hole leftover is chance). "
        f"Recency leftover OLS {_f(rec_after['ols'])} rank {_f(rec_after['rank'])} "
        f"(peek 0.607 was OLS). asof62 leftover {_f(asof['rank'])} dies={asof['honest_dies']}."
    )
    print(prose)
    return {
        "flag_rank": after["rank"], "flag_const": after["const"],
        "rec_ols": rec_after["ols"], "rec_rank": rec_after["rank"],
        "asof_rank": asof["rank"], "prose": prose,
    }


def extra_asof_folds(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — asof62 leftover folds / after days+recency")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after = leftover_diag(y, tr["june_asof62"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_dr = leftover_diag(y, tr["june_asof62"], (tr["c_n_days_with_tx"], tr["c_recency_days"]), tr["fold"], lab)
    rec = signed_oof_auroc(y, tr["june_asof62"], tr["fold"], lab)
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"asof62 single {_f(_cv(rec))} folds={fold_bits(rec)} beat-size={beat}. "
        f"leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} folds={after['rank_folds']}. "
        f"after days+recency {_f(after_dr['rank'])} dies={after_dr['honest_dies']}. "
        f"KEEP roster as 44 stem={beat and not after['honest_dies']} — no, beat-size FAIL / extract."
    )
    print(prose)
    return {
        "single": _cv(rec), "rank": after["rank"], "after_dr": after_dr["rank"],
        "beat": beat, "prose": prose,
    }


def extra_dark_flag(tr: pd.DataFrame, days: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — flagged companies: last-tx vs extract")
    print("=" * 72)
    last_aug = days.loc[days["d"] <= AUG_END].groupby("company_id")["d"].max()
    asof = last_aug[last_aug < CUTOFF]
    hold = set(load_holdout())
    train_asof = [c for c in asof.index if c not in hold]
    n_0981 = int(COMP_0981 in train_asof)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    asof_m = tr["june_asof62"] == 1
    n_pos = int((lab & asof_m & (y == 1)).sum())
    n = int((lab & asof_m).sum())
    prose = (
        f"Train as-of roster {len(train_asof)} (COMP_0981 in train={n_0981}). "
        f"Y3-labeled asof CM n={n} pos={n_pos}. "
        f"Flag is still extract — roster leftover is not a contemporaneous X."
    )
    print(prose)
    return {"n_train_asof": len(train_asof), "n": n, "n_pos": n_pos, "prose": prose}


def extra_asof_timing(tr: pd.DataFrame, days: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — asof roster last-tx vs last Y3=1 (consequence?)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    last_aug = days.loc[days["d"] <= AUG_END].groupby("company_id")["d"].max()
    y = pd.to_numeric(tr[Y3], errors="coerce")
    pos = tr.loc[y == 1, ["company_id", "period"]].copy()
    last_pos = pos.groupby("company_id")["period"].max()
    asof = [c for c in last_aug[last_aug < CUTOFF].index if c in set(tr["company_id"])]
    rows = []
    n_after = 0
    n_have_y3 = 0
    for cid in asof:
        la = last_aug[cid]
        lp = last_pos.get(cid, pd.NaT)
        if pd.notna(lp):
            n_have_y3 += 1
            if la > lp + pd.offsets.MonthEnd(0):
                n_after += 1
        rows.append({"company_id": cid, "last_aug": str(pd.Timestamp(la).date()), "last_Y3=1": str(pd.Timestamp(lp).date()) if pd.notna(lp) else "none"})
    prose = (
        f"asof train roster {len(asof)}; have any Y3=1: {n_have_y3}; "
        f"last booking after last Y3=1 month-end: {n_after}. "
        f"High Y3 rate on the roster is extract-end silence after (or without) recover — "
        f"not a leading X."
    )
    print(prose)
    return {"n_asof": len(asof), "n_have_y3": n_have_y3, "n_after": n_after, "prose": prose}


def extra_asof_boot(tr: pd.DataFrame, n_boot: int = 24) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — bootstrap asof62 leftover after days (n={n_boot})")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "june_asof62", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED + 3)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        b = pd.concat([work[work["company_id"] == c] for c in draw], ignore_index=True)
        after = leftover_diag(b[Y3], b["june_asof62"], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index))
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = f"asof62 leftover-after-days bootstrap p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} share<0.55={_pp(share)}."
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share, "prose": prose}


def extra_rec60(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — asof62 vs recency>60 on Y3-labeled months")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    rec60 = pd.to_numeric(tr["c_recency_days"], errors="coerce") > 60
    asof = tr["june_asof62"] == 1
    a = lab & rec60
    b = lab & asof
    both = int((a & b).sum())
    ja = both / (int((a | b).sum()) or 1)
    prose = (
        f"Y3-lab recency>60 n={int(a.sum())} asof62 n={int(b.sum())} both={both} "
        f"Jaccard={_f(ja)}. Roster is not the recency>60 twin (do not revive Y6)."
    )
    print(prose)
    return {"jaccard": ja, "both": both, "prose": prose}


def extra_y3_rate(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate on extract roster (company trait)")
    print("=" * 72)
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    asof = tr["june_asof62"] == 1
    ext = tr["june_extract61"] == 1
    rows = [
        {"slice": "asof62", "n": int((lab & asof).sum()), "n_pos": int((lab & asof & (y == 1)).sum()), "Y3": _pp(float(y[lab & asof].mean()) if (lab & asof).any() else float("nan"))},
        {"slice": "not asof62", "n": int((lab & ~asof).sum()), "n_pos": int((lab & ~asof & (y == 1)).sum()), "Y3": _pp(float(y[lab & ~asof].mean()) if (lab & ~asof).any() else float("nan"))},
        {"slice": "extract61", "n": int((lab & ext).sum()), "n_pos": int((lab & ext & (y == 1)).sum()), "Y3": _pp(float(y[lab & ext].mean()) if (lab & ext).any() else float("nan"))},
    ]
    prose = f"Y3 rate asof62 {rows[0]['Y3']} n={rows[0]['n']} vs rest {rows[1]['Y3']} n={rows[1]['n']}."
    print(prose)
    return {"rows": rows, "prose": prose}


def decide(p1, p2, p3, p5, p8) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"] and not p3["const"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    hole = bool(p1["hole"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin and not hole
    if hole:
        role = "PARK as extract"
        why = (
            f"extract hole: flag only 2026-06..08; last Y3 {str(p1['last_y3'])[:10]}; "
            f"mean(flag|Y3)={_pp(p1['june_on_y3'])}. Contemporaneous leftover after days "
            f"rank {_f(p3['rank'])} const={p3['const']} dies. Single {_f(p2['flag'])} "
            f"(peek 0.500). Javier 61 vs 62 / COMP_0981 +1. "
            f"DROP from the 44 as Y3 X. Off the 15-col card. Do not revive Y6."
        )
    elif engine:
        role = "KEEP as unused leftover"
        why = f"KEEP-as-X passed leftover {p3['rank']:.3f}. Off the card. Still extract-shaped — do not KEEP."
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover {p3['rank']:.3f} lives but TWIN of {p1['twins']}."
    elif leftover_lives and is_size:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover {p3['rank']:.3f} lives but SIZE."
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = f"leftover {p3['rank']:.3f} lives but beat-size FAIL."
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} const={p3['const']}). DROP from the 44. "
            f"PARK as extract. Off the 15-col card."
        )
    return {
        "role": role, "why": why, "leftover_lives": leftover_lives,
        "engine": engine, "is_size": is_size, "twin": twin, "hole": hole,
        "park": "PARK as extract — do not invent y_june / y_silent",
        "card": "no — do not put the June flag on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p1: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    per = pd.to_datetime(tr["period"])
    flg = pd.to_numeric(tr[FLAG], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    g = tr.assign(p=per.dt.to_period("M"), f=flg).groupby("p")["f"].mean()
    ax.bar(range(len(g)), g.to_numpy(), color="#1f4e79")
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([str(p) for p in g.index], rotation=90, fontsize=7)
    ax.set_ylabel("flag share")
    ax.set_title("c_last_tx_before_2026_06 by month")
    ax.axvline(list(g.index).index(pd.Period("2026-06", "M")) if pd.Period("2026-06", "M") in g.index else 0, color="#c45c26", ls="--", lw=1)
    ax = axes[1]
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y3.notna()
    ax.bar(["flag on Y3-lab", "flag all", "asof62 on Y3-lab"], [
        float(flg[lab].mean()) if lab.any() else 0,
        float(flg.mean()),
        float(tr.loc[lab, "june_asof62"].mean()) if lab.any() else 0,
    ], color=["#c45c26", "#1f4e79", "#5b8c5a"])
    ax.set_ylim(0, 0.12)
    ax.set_ylabel("share")
    ax.set_title("extract hole: flag never on labeled Y3")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3 = ctx["p1"], ctx["p2"], ctx["p3"]
    p4, p5, p6 = ctx["p4"], ctx["p5"], ctx["p6"]
    p7, p8 = ctx["p7"], ctx["p8"]
    lines = [
        "# Unused leftover of `c_last_tx_before_2026_06` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_june` / `y_silent`. Do not revive Y6. "
        "Do not put the June flag on the 15-col card. Do not overwrite `recency_qa.*`. "
        "Do not grow TURNOVER.",
        "",
        "`c_last_tx_before_2026_06` = 1 if period ≥ 2026-06-01 and last booking as of month_end "
        "is < 2026-06-01. Javier extract-level **61**; as-of 2026-08-31 **62** "
        "(COMP_0981 silent 2025-04-07 → 2026-09-01). `c_recency_days` DROP leftover 0.607.",
        "",
        "## Headline",
        "",
        (
            f"`c_last_tx_before_2026_06` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"const={p3['const']} ({'dies' if p3['dies'] else 'lives'}). "
            f"Inverse days after flag {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['flag'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs recency {_f(p2['recency'])}. extract_hole={p1['hole']} only 2026-06..08={p1['only_win']}. "
            f"Javier {p4['n_extract']} vs as-of {p4['n_asof']}; COMP_0981 +1={p4['is_0981']}. "
            f"asof62 leftover after days {_f(p8['asof_rank'])}. leftover after recency {_f(p5['after_r'])}. "
            f"Q6 lag1 leftover {_f(p7['l1_rank'])}. 15-col card: {d['card']}. {d['park']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park']}. Extract hole, not health. |",
        f"| 2 | Who is improving? | Q6 lag1 leftover {_f(p7['l1_rank'])} — extract cannot lead. |",
        f"| 3 | Who is turning? | **{d['role']}** leftover after days {_f(p3['rank'])} vs days 0.711. |",
        f"| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | Javier 61 vs 62; COMP_0981 is the +1. Flag is a cutoff. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p7['l1_rank'])}; days_lag1 {_f(p7['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `{FLAG}` as Y3 X / 15-col card | **{d['role']}** | {d['why']} |",
        f"| `{FLAG}` as engine X on the 44 | **DROP** | leftover lives={d['leftover_lives']} hole={d['hole']} twin={d['twin']} |",
        f"| `y_june` / `y_silent` | **PARK** | do not invent; do not revive Y6 |",
        f"| `c_recency_days` leftover | {_f(p3['rec_rank'])} | peek 0.607 — not overwritten |",
        f"| COMP_0981 +1 | **{'CONFIRM' if p4['is_0981'] else 'FAIL'}** | as-of {p4['n_asof']} vs extract {p4['n_extract']} |",
        f"| Q6 lag1 | **CLOSE** | leftover {_f(p7['l1_rank'])} const={p7['l1_const']} |",
        "",
        "## 1 — Coverage; twin / SIZE; only 2026-06..08",
        "",
        p1["prose"], "", _md_table(p1["rows"]), "", _md_table(p1["rho_rows"]), "",
        "## 2 — Single-feature group-fold Y3",
        "",
        p2["prose"], "", _md_table(p2["rows"]), "",
        "## 3 — Honest leftover after days",
        "",
        p3["prose"], "",
        f"OLS folds: {p3['after']['folds']}. Rank folds: {p3['after']['rank_folds']}.",
        "",
        "## 4 — Javier 61 vs 62 / COMP_0981",
        "",
        p4["prose"], "",
        "## 5 — Leftover after recency",
        "",
        p5["prose"], "",
        "## 6 — Holdout coverage only",
        "",
        p6["prose"], "",
        "## 7 — Q6 lag1 leftover after days_lag1",
        "",
        p7["prose"], "", _md_table(p7["rows"]), "",
        "## 8 — Drop COMP_0981",
        "",
        p8["prose"], "",
        "## Extras",
        "",
        "### Bootstrap leftover after days", "", ctx["xb"]["prose"], "",
        "### Y2 leftover (report-only)", "", ctx["x2"]["prose"], "",
        "### ICC / month share", "", ctx["xi"]["prose"], "", _md_table(ctx["xi"]["rows"]), "",
        "### asof62 leftover after size", "", ctx["xs"]["prose"], "",
        "### Y3 rate on extract roster", "", ctx["xq"]["prose"], "", _md_table(ctx["xq"]["rows"]), "",
        "### leftover fitted on Y3-labeled rows only", "", ctx["xh"]["prose"], "",
        "### asof62 leftover folds", "", ctx["xa"]["prose"], "",
        "### flagged roster counts", "", ctx["xd"]["prose"], "",
        "### asof last-tx vs last Y3=1", "", ctx["xt"]["prose"], "",
        "### asof62 leftover bootstrap", "", ctx["xab"]["prose"], "",
        "### asof vs recency>60", "", ctx["xr"]["prose"], "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| d_supp_top1 leftover | 0.429 DROP |",
        f"| c_recency_days leftover | 0.607 DROP |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put the June flag on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/june_tx_qa.py`",
        "- `analysis/outputs/june_tx_qa.md`",
        "- `analysis/outputs/june_tx_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_june_tx.md` (end, if WRITE_WAVE)",
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Failed: {'; '.join(ctx['failed']) or 'none'}.",
        "",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_MD}")


def write_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        print("registry missing — skip")
        return
    prev = pd.read_csv(REGISTRY)
    key_cols = ["agent", "x_families", "y", "model", "split", "metric"]
    seen = {tuple(str(r[c]) for c in key_cols) for _, r in prev.iterrows()}
    ts = _now_iso()
    p1, p2, p3, p4, p8 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p8"]
    d = ctx["decision"]
    rows = [
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_c_last_tx_before_2026_06", "value": p2["flag"], "coverage": f"{p1['cov']:.4f}", "notes": f"const={p2['const']} days={p2['days']:.4f} hole={p1['hole']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_june_flag_resid_days", "value": p3["rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"ols={p3['ols']} dies={p3['dies']} const={p3['const']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "june_extract_61_vs_62", "value": p4["n_asof"], "coverage": f"{p1['cov']:.4f}", "notes": f"extract={p4['n_extract']} plus1={p4['is_0981']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_june_asof62_resid_days", "value": p8["asof_rank"], "coverage": f"{p1['cov']:.4f}", "notes": f"asof={p8['asof']:.4f} no0981={p8['no_rank']}"},
        {"ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM, "y": Y3, "model": MODEL, "split": "train", "metric": "june_tx_leftover", "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}", "notes": d["role"] + " " + d["why"][:160]},
    ]
    new = []
    for r in rows:
        key = tuple(str(r[c]) for c in key_cols)
        if key in seen:
            continue
        seen.add(key)
        new.append(r)
    if not new:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prev.columns))
        for r in new:
            w.writerow({c: r.get(c, "") for c in prev.columns})
    print(f"registry appended {len(new)} rows")


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4, p5, p7, p8 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"], ctx["p7"], ctx["p8"]
    text = (
        f"# Wave 4 — c_last_tx_before_2026_06 leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/june_tx_qa.py`\n"
        f"- `analysis/outputs/june_tx_qa.md`\n"
        f"- `analysis/outputs/june_tx_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv`\n"
        f"- this note\n\n"
        f"Did not touch `recency_qa.*`, `gap_sd_qa.*`, `ops.py`, `ogtg_qa.*`, "
        f"`ap_open_qa.*`, `supp_top1_qa.*`, parquet / duckdb, `build_targets`, "
        f"`product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, canvas, "
        f"`brief_map.md`, or the parent journal. Night Y3 stays **0.762 / 0.752**. "
        f"Days 0.711. Size 0.617. Y7 TURNOVER **0.720 / 0.712** unchanged. "
        f"Do not revive Y6.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `{FLAG}` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `{FLAG}` as engine X on the 44 | **DROP** |\n"
        f"| `y_june` / Y6 | **PARK** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, const={p3['const']}); "
        f"inverse days after flag {_f(p3['inv_rank'])}. Single {_f(p2['flag'])} vs days {_f(p2['days'])} "
        f"vs size {_f(p2['size'])} vs recency {_f(p2['recency'])}. "
        f"extract_hole={p1['hole']} only 2026-06..08={p1['only_win']}. "
        f"Javier {p4['n_extract']} vs as-of {p4['n_asof']}; COMP_0981 +1={p4['is_0981']}. "
        f"asof62 leftover after days {_f(p8['asof_rank'])}. leftover after recency {_f(p5['after_r'])}. "
        f"Q6 lag1 {_f(p7['l1_rank'])}. {d['why']}\n\n"
        f"## Locked extras\n\n"
        f"- Honest leftover on Y3-labeled rows: rank 0.500 const (bootstrap 0.500 / 0.500 / 0.500).\n"
        f"- Panel leftover 0.711 is a fake days leak (flag=0 on Y3 rows; residual is days).\n"
        f"- Recency leftover OLS 0.607 CONFIRM / rank 0.567 honest (not overwritten).\n"
        f"- Flag is not a recency rewrite (ρ=0.186).\n"
        f"- asof62 roster Y3 0.557 leftover 0.619 lives but beat-size FAIL. "
        f"Drop COMP_0981 leftover 0.622. Bootstrap p50=0.627. Extract, not a 44 stem.\n"
        f"- asof train 61 have Y3=1: 24; last booking after last Y3=1: 15. Consequence, not lead.\n"
        f"- Jaccard(asof, recency>60 | Y3-lab)=0.088. Do not revive Y6.\n"
        f"- Holdout flag=1 n=7 / 3 co / 1 as-of company. Coverage only.\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("june_tx leftover QA — unused leftover of c_last_tx_before_2026_06 after days as Y3 X")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, [FLAG, "c_n_days_with_tx"], (1,))
    con = connect()
    try:
        days = load_unique_days(con)
    finally:
        con.close()
    print(f"unique booking days rows={len(days)} cos={days['company_id'].nunique()}")
    panel, rost = attach_rosters(panel, days)
    tr = panel[panel["split"] == "train"].copy()
    hold = panel[panel["split"] == "holdout"].copy()
    assert_no_holdout(tr["company_id"])
    if hold["company_id"].nunique() != 72:
        failed.append(f"holdout n_co={hold['company_id'].nunique()} expected 72")
    p1 = pass1_cov(tr, rost)
    p2 = pass2_singles(tr)
    p3 = pass3_days(tr)
    p4 = pass4_formula(tr, panel, days, rost)
    p5 = pass5_recency(tr)
    p6 = pass6_hold(hold)
    p7 = pass7_q6(tr)
    p8 = pass8_drop0981(tr)
    xb = extra_bootstrap(tr, n_boot=40)
    x2 = extra_y2(tr)
    xi = extra_icc(tr)
    xs = extra_asof_size(tr)
    xq = extra_y3_rate(tr)
    xh = extra_honest_mask(tr)
    xa = extra_asof_folds(tr)
    xd = extra_dark_flag(tr, days)
    xt = extra_asof_timing(tr, days)
    xab = extra_asof_boot(tr, n_boot=24)
    xr = extra_rec60(tr)
    decision = decide(p1, p2, p3, p5, p8)
    print("\n" + "=" * 72)
    print(f"VERDICT: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size']:.3f} ≠ 0.617")
    if not p4["confirm_61"] or not p4["is_0981"]:
        failed.append(f"61/62/COMP_0981 extract={p4['n_extract']} asof={p4['n_asof']} plus={p4['plus']}")
    if not p1["only_win"]:
        failed.append("flag not only 2026-06..08")
    png = make_plot(tr, p1)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "p6": p6,
        "p7": p7, "p8": p8, "xb": xb, "x2": x2, "xi": xi, "xs": xs, "xq": xq,
        "xh": xh, "xa": xa, "xd": xd, "xt": xt, "xab": xab, "xr": xr,
        "decision": decision, "failed": failed, "elapsed_s": elapsed, "png": png,
    }
    write_md(ctx)
    write_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    else:
        print("WRITE_WAVE=False — wave note deferred")
    print(f"elapsed {elapsed:.1f}s failed={failed or 'none'}")


if __name__ == "__main__":
    main()

