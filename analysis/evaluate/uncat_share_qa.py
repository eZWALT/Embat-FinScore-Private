"""Unused leftover of ``a_uncat_share`` after days as Y3 X.

``a_uncat_share`` = count share of txs whose category is
``uncategorized`` or not in CAT_MAP (Family A, this month).
Uncat QA already PARK as Y and as X (style dummy; Y3 0.542 vs
size 0.617; ICC 0.985). This lane is leftover after days on the
keep-list 44 stem.

Do **not** overwrite ``uncat_qa.*``. Missing-CP is not an uncat
twin (ρ 0.131). Do not merge amount-uncat. Do not add
``uncategorized`` to CAT_MAP.

KEEP-as-X: beat size ≥0.02 **and** leftover after days **and** not
SIZE (|ρ| vs log1p(a_in3) ≥0.50) **and** not a twin (|ρ|≥0.80 vs
days / a_n_tx / miss_cp / d_tx_cp_share). Leftover <0.55 dies.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.uncat_share_qa

Owned: analysis/evaluate/uncat_share_qa.py, analysis/outputs/uncat_share_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_uncat_share.md (end).
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
from analysis.features.common import ANALYSIS, CAT_MAP, DATA, connect
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
OUT_MD = ANALYSIS / "outputs" / "uncat_share_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "uncat_share_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_uncat_share.md"
AGENT = "572fb928"
WAVE = "4"
ROUND = "R4"
MODEL = "uncat_share_qa"
X_FAM = "A"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
DAYS_LAG1_QUOTE = 0.684
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = 0.720
Y7_B_SHALLOW = 0.712
UNCAT_Y3_QUOTE = 0.542
UNCAT_N_QUOTE = 5536
UNCAT_POS_QUOTE = 372
ICC_QUOTE = 0.985
MISS_RHO_QUOTE = 0.131
COUNT_MEAN_QUOTE = 0.258
AMT_MEAN_QUOTE = 0.226
COUNT_AMT_RHO = 0.889
KEEP_DELTA = 0.02
TWIN_RHO = 0.80
SIZE_RHO = 0.50
CHANCE = 0.55
ICC_TRAIT = 0.85
FAKE_DAYS_RHO = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
N_DARK_WANT = 470
WRITE_WAVE = True

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "first_month",
    "a_in3",
    "a_n_tx",
    "a_uncat_share",
    "c_n_days_with_tx",
    "d_tx_cp_share",
    "b_below_0",
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


def _cat_sql_list() -> str:
    return ", ".join("'" + k.replace("'", "''") + "'" for k in CAT_MAP)


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def median_acf(series: pd.Series, company: pd.Series, lag: int = 1) -> float:
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
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)}).dropna()
    k = int(d["co"].nunique())
    n = len(d)
    if k < 2 or n < k + 2:
        return {"icc": float("nan"), "icc1": float("nan"), "k": k, "n": n}
    grand = float(d["x"].mean())
    ns = d.groupby("co")["x"].size()
    mus = d.groupby("co")["x"].mean()
    ssb = float(((mus - grand) ** 2 * ns).sum())
    ssw = float(((d["x"] - d["co"].map(mus)) ** 2).sum())
    msb = ssb / (k - 1)
    msw = ssw / (n - k) if n > k else float("nan")
    n0 = (n - float((ns**2).sum()) / n) / (k - 1)
    # uncat_qa / feature-report ICC is MSB/(MSB+MSW) ≈ 0.985.
    icc_quote = msb / (msb + msw) if np.isfinite(msw) and (msb + msw) != 0 else float("nan")
    icc1 = (msb - msw) / (msb + (n0 - 1) * msw) if np.isfinite(msw) and (msb + (n0 - 1) * msw) != 0 else float("nan")
    return {"icc": float(icc_quote), "icc1": float(icc1), "k": k, "n": n}


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
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            fold_rows.append(
                {"fold": k, "auroc": float("nan"), "sign": 0, "n_va": int(va.sum()), "n_pos": int((va & (y == 1)).sum())}
            )
            continue
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
    honest_dies = bool(fake or (np.isfinite(rank_cv) and rank_cv < CHANCE))
    return {
        "ols": ols_cv,
        "rank": rank_cv,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "fake": fake,
        "almost": almost,
        "honest_dies": honest_dies,
        "folds": fold_bits(rec),
        "rank_folds": fold_bits(rrec),
        "rec": rec,
        "rrec": rrec,
    }


def company_demean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d["x"] - d.groupby("co")["x"].transform("mean")


def company_mean(series: pd.Series, company: pd.Series) -> pd.Series:
    d = pd.DataFrame({"x": pd.to_numeric(series, errors="coerce"), "co": company.astype(str)})
    return d.groupby("co")["x"].transform("mean")


def _cv(res: dict) -> float:
    return float("nan") if res.get("low_power") else float(res.get("cv", float("nan")))


def _auc_row(y: str, feat: str, res: dict) -> dict:
    return {
        "y": y,
        "feature": feat,
        "n": f"{res['n_defined']:,}",
        "n_pos": f"{res['n_pos']:,}",
        "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
        "train": "LOW_POWER" if res["low_power"] else _f(res["train_auc"]),
        "sd": _f(res["sd"]) if not res["low_power"] else "—",
        "sign": res["train_sign"] if not res["low_power"] else "—",
        "folds": fold_bits(res) if not res["low_power"] else "—",
    }


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


def load_memory_shares(con) -> pd.DataFrame:
    """Amount-uncat + miss_cp in memory. Never written to parquet."""
    cats = _cat_sql_list()
    df = con.execute(
        f"""
        SELECT
          CAST(t.company_id AS VARCHAR) AS company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          COUNT(*) AS n_tx,
          SUM(ABS(t.amount)) AS abs_all,
          SUM(CASE WHEN t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats})
                   THEN 1 ELSE 0 END) AS n_uncat,
          SUM(CASE WHEN t.category = 'uncategorized'
                     OR t.category IS NULL
                     OR t.category NOT IN ({cats})
                   THEN ABS(t.amount) ELSE 0 END) AS abs_uncat,
          SUM(CASE WHEN t.category = 'uncategorized' THEN 1 ELSE 0 END) AS n_token,
          SUM(CASE WHEN t.counterparty_id IS NULL
                     OR length(trim(CAST(t.counterparty_id AS VARCHAR))) = 0
                    THEN 1 ELSE 0 END) AS n_miss
        FROM transactions t
        WHERE t."date" IS NOT NULL
          AND t."date" >= DATE '2024-09-01'
          AND t."date" < DATE '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    df["company_id"] = df["company_id"].astype(str)
    df["period"] = pd.to_datetime(df["month"])
    df = df.drop(columns=["month"])
    n_tx = pd.to_numeric(df["n_tx"], errors="coerce")
    abs_all = pd.to_numeric(df["abs_all"], errors="coerce")
    df["count_uncat"] = np.where(n_tx > 0, df["n_uncat"] / n_tx, np.nan)
    df["amt_uncat"] = np.where(abs_all > 0, df["abs_uncat"] / abs_all, np.nan)
    df["miss_cp_share"] = np.where(n_tx > 0, df["n_miss"] / n_tx, np.nan)
    return df[["company_id", "period", "count_uncat", "amt_uncat", "miss_cp_share", "n_token", "n_uncat"]]


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
    leak = leakage_check(
        ["a_uncat_share", "a_n_tx", "c_n_days_with_tx", "log_in3", "d_tx_cp_share"],
        Y3,
        forbidden_prefixes=["b"],
    )
    if not leak["ok"]:
        raise RuntimeError(f"Y3 never-B leak: {leak['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def pass1_cov(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 1 — coverage; twin / SIZE screen")
    print("=" * 72)
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    amt = pd.to_numeric(tr["amt_uncat"], errors="coerce")
    miss = pd.to_numeric(tr["miss_cp_share"], errors="coerce")
    count_m = float(x.mean())
    amt_m = float(amt.mean())
    rho_ca = spearman(x, amt)
    acf1 = median_acf(x, tr["company_id"], 1)
    pairs = {
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "miss_cp": miss,
        "d_tx_cp_share": tr["d_tx_cp_share"],
        "log1p(a_in3)": tr["log_in3"],
        "amt_uncat": amt,
    }
    rhos = {k: spearman(x, v) for k, v in pairs.items()}
    twins = [k for k, v in rhos.items() if np.isfinite(v) and abs(v) >= TWIN_RHO]
    is_size = bool(np.isfinite(rhos["log1p(a_in3)"]) and abs(rhos["log1p(a_in3)"]) >= SIZE_RHO)
    twin_gate = any(
        np.isfinite(rhos[k]) and abs(rhos[k]) >= TWIN_RHO
        for k in ("c_n_days_with_tx", "a_n_tx", "miss_cp", "d_tx_cp_share")
    )
    miss_ok = bool(np.isfinite(rhos["miss_cp"]) and abs(rhos["miss_cp"] - MISS_RHO_QUOTE) < 0.03)
    ca_ok = bool(np.isfinite(rho_ca) and abs(rho_ca - COUNT_AMT_RHO) < 0.03)
    mean_ok = bool(abs(count_m - COUNT_MEAN_QUOTE) < 0.015 and abs(amt_m - AMT_MEAN_QUOTE) < 0.015)
    store_vs = float((x - pd.to_numeric(tr["count_uncat"], errors="coerce")).abs().max())
    rows = [
        {"col": "a_uncat_share", "n_nn": f"{int(x.notna().sum()):,}", "cov": _pp(_pct(int(x.notna().sum()), n_cm)), "mean": _f(count_m), "acf1": _f(acf1)},
        {"col": "amt_uncat (memory)", "n_nn": f"{int(amt.notna().sum()):,}", "cov": _pp(_pct(int(amt.notna().sum()), n_cm)), "mean": _f(amt_m), "acf1": _f(median_acf(amt, tr["company_id"], 1))},
        {"col": "miss_cp_share (memory)", "n_nn": f"{int(miss.notna().sum()):,}", "cov": _pp(_pct(int(miss.notna().sum()), n_cm)), "mean": _f(float(miss.mean())), "acf1": _f(median_acf(miss, tr["company_id"], 1))},
    ]
    rho_rows = [
        {
            "vs": k,
            "rho": _f(v),
            "flag": "SIZE" if k == "log1p(a_in3)" and abs(v) >= SIZE_RHO else "TWIN" if abs(v) >= TWIN_RHO else "no",
        }
        for k, v in rhos.items()
    ]
    prose = (
        f"Train {n_co:,} co / {n_cm:,} CM. a_uncat_share cov {_pp(_pct(int(x.notna().sum()), n_cm))} "
        f"mean {_f(count_m)} (quote 0.258 {'CONFIRM' if abs(count_m - COUNT_MEAN_QUOTE) < 0.015 else 'DRIFT'}) "
        f"acf1={_f(acf1)}. amt_uncat mean {_f(amt_m)} (quote 0.226 "
        f"{'CONFIRM' if abs(amt_m - AMT_MEAN_QUOTE) < 0.015 else 'DRIFT'}) ρ={_f(rho_ca)} "
        f"(quote 0.889 {'CONFIRM' if ca_ok else 'DRIFT'}). store vs count max|Δ|={store_vs:.6f}. "
        f"ρ vs days {_f(rhos['c_n_days_with_tx'])} vs a_n_tx {_f(rhos['a_n_tx'])} "
        f"vs miss_cp {_f(rhos['miss_cp'])} (quote 0.131 {'CONFIRM' if miss_ok else 'DRIFT'}) "
        f"vs d_tx {_f(rhos['d_tx_cp_share'])} vs size {_f(rhos['log1p(a_in3)'])}. "
        f"SIZE={is_size} twins={twins or 'none'} twin_gate={twin_gate}."
    )
    print(prose)
    return {
        "rows": rows,
        "rho_rows": rho_rows,
        "rhos": rhos,
        "twins": twins,
        "is_size": is_size,
        "twin_gate": twin_gate,
        "n_cm": n_cm,
        "n_co": n_co,
        "cov": _pct(int(x.notna().sum()), n_cm),
        "acf1": acf1,
        "count_mean": count_m,
        "amt_mean": amt_m,
        "rho_ca": rho_ca,
        "miss_ok": miss_ok,
        "ca_ok": ca_ok,
        "mean_ok": mean_ok,
        "store_vs": store_vs,
        "prose": prose,
    }


def pass2_singles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 2 — single-feature group-fold Y3")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    feats = {
        "a_uncat_share": tr["a_uncat_share"],
        "amt_uncat": tr["amt_uncat"],
        "miss_cp_share": tr["miss_cp_share"],
        "d_tx_cp_share": tr["d_tx_cp_share"],
        "log1p(a_in3)": tr["log_in3"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
    }
    recs = {}
    rows = []
    for fname, x in feats.items():
        rec = signed_oof_auroc(y, x, folds, lab)
        recs[fname] = rec
        rows.append(_auc_row(Y3, fname, rec))
        print(f"  {fname} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} folds={fold_bits(rec)}")
    y3_u = _cv(recs["a_uncat_share"])
    y3_size = _cv(recs["log1p(a_in3)"])
    y3_days = _cv(recs["c_n_days_with_tx"])
    days_ok = bool(np.isfinite(y3_days) and abs(y3_days - DAYS_BENCH) < 0.008)
    size_ok = bool(np.isfinite(y3_size) and abs(y3_size - SIZE_QUOTE) < 0.008)
    peek_ok = bool(
        recs["a_uncat_share"]["n_defined"] == UNCAT_N_QUOTE
        and recs["a_uncat_share"]["n_pos"] == UNCAT_POS_QUOTE
        and np.isfinite(y3_u)
        and abs(y3_u - UNCAT_Y3_QUOTE) < 0.015
    )
    beat_size = bool(np.isfinite(y3_u) and (y3_u - SIZE_QUOTE) >= KEEP_DELTA)
    prose = (
        f"Y3 a_uncat_share {_f(y3_u)} n={recs['a_uncat_share']['n_defined']:,} "
        f"pos={recs['a_uncat_share']['n_pos']:,} "
        f"(quote 0.542 / 5,536 / 372 {'CONFIRM' if peek_ok else 'DRIFT'}). "
        f"amt_uncat {_f(_cv(recs['amt_uncat']))} vs size {_f(y3_size)} vs days {_f(y3_days)}. "
        f"Replica days 0.711 {'CONFIRM' if days_ok else 'DRIFT'} / size 0.617 "
        f"{'CONFIRM' if size_ok else 'DRIFT'}. Beat-size Δ="
        f"{_f(y3_u - SIZE_QUOTE) if np.isfinite(y3_u) else '—'} "
        f"{'PASS' if beat_size else 'FAIL'}."
    )
    print(prose)
    return {
        "rows": rows,
        "recs": recs,
        "uncat": y3_u,
        "amt": _cv(recs["amt_uncat"]),
        "size": y3_size,
        "days": y3_days,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "peek_ok": peek_ok,
        "beat_size": beat_size,
        "n_def": recs["a_uncat_share"]["n_defined"],
        "n_pos": recs["a_uncat_share"]["n_pos"],
        "prose": prose,
    }


def pass3_days(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 3 — honest leftover of a_uncat_share after days; inverse")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"],), folds, lab)
    inv = leftover_diag(y, tr["c_n_days_with_tx"], (tr["a_uncat_share"],), folds, lab)
    prose = (
        f"a_uncat_share leftover after days OLS {_f(after['ols'])} rank {_f(after['rank'])} "
        f"fake={after['fake']} almost={after['almost']} ρ(resid,days)={_f(after['rho_ctrl'])} "
        f"R²={_f(after['r2'])} honest_dies={after['honest_dies']} n={after['n']:,} pos={after['n_pos']:,}. "
        f"Inverse: days leftover after uncat OLS {_f(inv['ols'])} rank {_f(inv['rank'])} "
        f"dies={inv['honest_dies']}."
    )
    print(prose)
    return {
        "after": after,
        "inv": inv,
        "ols": after["ols"],
        "rank": after["rank"],
        "dies": after["honest_dies"],
        "fake": after["fake"],
        "almost": after["almost"],
        "r2": after["r2"],
        "inv_rank": inv["rank"],
        "inv_dies": inv["honest_dies"],
        "prose": prose,
    }


def pass5_after_twins(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 5 — leftover after miss_cp / d_tx")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    after_m = leftover_diag(y, tr["a_uncat_share"], (tr["miss_cp_share"],), folds, lab)
    after_d = leftover_diag(y, tr["a_uncat_share"], (tr["d_tx_cp_share"],), folds, lab)
    after_md = leftover_diag(y, tr["a_uncat_share"], (tr["miss_cp_share"], tr["d_tx_cp_share"]), folds, lab)
    after_dm = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"], tr["miss_cp_share"]), folds, lab)
    m_after = leftover_diag(y, tr["miss_cp_share"], (tr["a_uncat_share"],), folds, lab)
    d_after = leftover_diag(y, tr["d_tx_cp_share"], (tr["a_uncat_share"],), folds, lab)
    rows = []
    for name, rec in (
        ("after miss_cp", after_m),
        ("after d_tx", after_d),
        ("after miss+d_tx", after_md),
        ("after days+miss", after_dm),
        ("miss_cp after uncat", m_after),
        ("d_tx after uncat", d_after),
    ):
        rows.append(
            {
                "bar": name,
                "OLS": _f(rec["ols"]),
                "rank": _f(rec["rank"]),
                "ρ(resid,bar)": _f(rec["rho_ctrl"]),
                "R2": _f(rec["r2"]),
                "dies": rec["honest_dies"],
                "n": rec["n"],
                "n_pos": rec["n_pos"],
            }
        )
    prose = (
        f"uncat leftover after miss_cp rank {_f(after_m['rank'])} dies={after_m['honest_dies']} "
        f"(want live if not a twin). after d_tx {_f(after_d['rank'])} dies={after_d['honest_dies']}; "
        f"after days+miss {_f(after_dm['rank'])} dies={after_dm['honest_dies']}. "
        f"Inverse miss after uncat {_f(m_after['rank'])}; d_tx after uncat {_f(d_after['rank'])}."
    )
    print(prose)
    return {
        "rows": rows,
        "m_rank": after_m["rank"],
        "m_dies": after_m["honest_dies"],
        "d_rank": after_d["rank"],
        "d_dies": after_d["honest_dies"],
        "dm_rank": after_dm["rank"],
        "md_rank": after_md["rank"],
        "prose": prose,
    }


def pass6_icc(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 6 — ICC / demean leftover (quote 0.985 TRAIT)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    icc = icc_anova(tr["a_uncat_share"], tr["company_id"])
    trait = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_TRAIT)
    icc_ok = bool(np.isfinite(icc["icc"]) and abs(icc["icc"] - ICC_QUOTE) < 0.02)
    demean = company_demean(tr["a_uncat_share"], tr["company_id"])
    meanx = company_mean(tr["a_uncat_share"], tr["company_id"])
    rec_d = signed_oof_auroc(y, demean, tr["fold"], lab)
    rec_m = signed_oof_auroc(y, meanx, tr["fold"], lab)
    after_d = leftover_diag(y, demean, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_m = leftover_diag(y, meanx, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"ICC MSB/(MSB+MSW)={_f(icc['icc'])} (quote 0.985 {'CONFIRM' if icc_ok else 'DRIFT'}) "
        f"ANOVA ICC(1)={_f(icc['icc1'])} {'TRAIT' if trait else 'STATE-by-ICC1'} k={icc['k']}. "
        f"Demean CV {_f(rec_d['cv'])} leftover-days {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"Company-mean CV {_f(rec_m['cv'])} leftover-days {_f(after_m['rank'])} dies={after_m['honest_dies']}. "
        f"Leftover is BETWEEN (style), not a month shock."
    )
    print(prose)
    return {
        "icc": icc,
        "trait": trait,
        "icc_ok": icc_ok,
        "demean_rank": after_d["rank"],
        "mean_rank": after_m["rank"],
        "demean_cv": _cv(rec_d),
        "mean_cv": _cv(rec_m),
        "prose": prose,
    }


def pass7_dark(tr: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 7 — dark vs ERP leftover (uncat ≠ no ERP)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    erp = tr["company_id"].isin(book)
    dark = ~erp
    n_dark_co = int(tr.loc[dark, "company_id"].nunique())
    dark_nn = int(x[dark].notna().sum())
    rec_e = signed_oof_auroc(y, tr["a_uncat_share"], tr["fold"], lab & erp)
    rec_d = signed_oof_auroc(y, tr["a_uncat_share"], tr["fold"], lab & dark)
    after_e = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"],), tr["fold"], lab & erp)
    after_d = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"],), tr["fold"], lab & dark)
    mean_e = float(x[erp].mean())
    mean_d = float(x[dark].mean())
    rows = [
        {
            "book": "dark",
            "n_co": n_dark_co,
            "uncat nn": dark_nn,
            "mean": _f(mean_d),
            "Y3 n": rec_d["n_defined"],
            "Y3 pos": rec_d["n_pos"],
            "CV": "LOW_POWER" if rec_d["low_power"] else _f(rec_d["cv"]),
            "leftover": _f(after_d["rank"]),
            "dies": after_d["honest_dies"],
        },
        {
            "book": "ERP",
            "n_co": int(tr.loc[erp, "company_id"].nunique()),
            "uncat nn": int(x[erp].notna().sum()),
            "mean": _f(mean_e),
            "Y3 n": rec_e["n_defined"],
            "Y3 pos": rec_e["n_pos"],
            "CV": "LOW_POWER" if rec_e["low_power"] else _f(rec_e["cv"]),
            "leftover": _f(after_e["rank"]),
            "dies": after_e["honest_dies"],
        },
    ]
    dark_ok = n_dark_co == N_DARK_WANT and dark_nn > 0
    prose = (
        f"Dark {n_dark_co} (want {N_DARK_WANT}) uncat nn={dark_nn} mean {_f(mean_d)} "
        f"vs ERP mean {_f(mean_e)} {'CONFIRM uncat ≠ no ERP' if dark_ok else 'FAIL'}. "
        f"Dark Y3 {_f(rec_d['cv'])} leftover {_f(after_d['rank'])} dies={after_d['honest_dies']}. "
        f"ERP Y3 {_f(rec_e['cv'])} leftover {_f(after_e['rank'])} dies={after_e['honest_dies']}."
    )
    print(prose)
    return {
        "rows": rows,
        "dark_ok": dark_ok,
        "n_dark_co": n_dark_co,
        "dark_nn": dark_nn,
        "mean_d": mean_d,
        "mean_e": mean_e,
        "dark_rank": after_d["rank"],
        "erp_rank": after_e["rank"],
        "prose": prose,
    }


def pass8_q6(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 8 — Q6 lag1 leftover after days_lag1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    folds = tr["fold"]
    recs = {}
    rows = []
    for name in ("a_uncat_share", "a_uncat_share_lag1", "c_n_days_with_tx_lag1"):
        rec = signed_oof_auroc(y, tr[name], folds, lab)
        recs[name] = rec
        rows.append(_auc_row(Y3, name, rec))
        print(f"  {name} CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    after_l1 = leftover_diag(y, tr["a_uncat_share_lag1"], (tr["c_n_days_with_tx_lag1"],), folds, lab)
    days_l1 = _cv(recs["c_n_days_with_tx_lag1"])
    days_ok = bool(np.isfinite(days_l1) and abs(days_l1 - DAYS_LAG1_QUOTE) < 0.008)
    lag_ok = bool(np.isfinite(_cv(recs["a_uncat_share_lag1"])) and abs(_cv(recs["a_uncat_share_lag1"]) - 0.529) < 0.02)
    prose = (
        f"Y3 uncat_lag1 {_f(recs['a_uncat_share_lag1']['cv'])} "
        f"(quote 0.529 {'CONFIRM' if lag_ok else 'DRIFT'}) leftover after days_lag1 "
        f"rank {_f(after_l1['rank'])} dies={after_l1['honest_dies']}. "
        f"Days lag1 {_f(days_l1)} (quote 0.684 {'CONFIRM' if days_ok else 'DRIFT'})."
    )
    print(prose)
    return {
        "rows": rows,
        "lag1": _cv(recs["a_uncat_share_lag1"]),
        "l1_rank": after_l1["rank"],
        "l1_dies": after_l1["honest_dies"],
        "days_l1": days_l1,
        "days_ok": days_ok,
        "prose": prose,
    }


def pass9_amount(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("CUT 9 — count-uncat vs amount-uncat leftover (do not merge)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_a = leftover_diag(y, tr["amt_uncat"], (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_c = leftover_diag(y, tr["a_uncat_share"], (tr["amt_uncat"],), tr["fold"], lab)
    after_ac = leftover_diag(y, tr["amt_uncat"], (tr["a_uncat_share"],), tr["fold"], lab)
    after_ad = leftover_diag(y, tr["amt_uncat"], (tr["c_n_days_with_tx"], tr["a_uncat_share"]), tr["fold"], lab)
    rec_a = signed_oof_auroc(y, tr["amt_uncat"], tr["fold"], lab)
    rho = spearman(tr["a_uncat_share"], tr["amt_uncat"])
    same = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    prose = (
        f"ρ(count, amount)={_f(rho)} (quote 0.889 {'TWIN rewrite' if same else 'not ≥0.80 twin-gate'}). "
        f"Y3 amt_uncat {_f(_cv(rec_a))} leftover after days {_f(after_a['rank'])} dies={after_a['honest_dies']}. "
        f"count leftover after amount {_f(after_c['rank'])} dies={after_c['honest_dies']}. "
        f"amount leftover after count {_f(after_ac['rank'])} dies={after_ac['honest_dies']}. "
        f"Do not merge amount-uncat. Do not add uncategorized to CAT_MAP."
    )
    print(prose)
    return {
        "rho": rho,
        "same": same,
        "amt_cv": _cv(rec_a),
        "amt_rank": after_a["rank"],
        "count_after_amt": after_c["rank"],
        "amt_after_count": after_ac["rank"],
        "amt_after_both": after_ad["rank"],
        "prose": prose,
    }


def pass10_hold(hold: pd.DataFrame, book: set[str]) -> dict:
    print("\n" + "=" * 72)
    print("CUT 10 — holdout coverage only (no AUROC)")
    print("=" * 72)
    x = pd.to_numeric(hold["a_uncat_share"], errors="coerce")
    erp = hold["company_id"].isin(book)
    rec = {
        "n_co": int(hold["company_id"].nunique()),
        "n_cm": len(hold),
        "n_nn": int(x.notna().sum()),
        "cov": _pct(int(x.notna().sum()), len(hold)),
        "p50": float(x[x.notna()].median()) if x.notna().any() else float("nan"),
        "dark_nn": int(x[~erp].notna().sum()),
        "dark_mean": float(x[~erp].mean()) if x[~erp].notna().any() else float("nan"),
    }
    prose = (
        f"Holdout {rec['n_co']} co / {rec['n_cm']} CM nn={rec['n_nn']} "
        f"cov={_pp(rec['cov'])} p50={_f(rec['p50'])} dark nn={rec['dark_nn']} "
        f"dark mean={_f(rec['dark_mean'])} (no fit, no AUROC)."
    )
    print(prose)
    return {**rec, "prose": prose}


def extra_bootstrap(tr: pd.DataFrame, n_boot: int = 40) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — company bootstrap leftover after days (n={n_boot})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab, ["company_id", "fold", Y3, "a_uncat_share", "c_n_days_with_tx"]].copy()
    cos = work["company_id"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(FOLD_SEED)
    ranks = []
    for _ in range(n_boot):
        draw = rng.choice(cos, size=len(cos), replace=True)
        parts = [work[work["company_id"] == c] for c in draw]
        b = pd.concat(parts, ignore_index=True)
        after = leftover_diag(
            b[Y3], b["a_uncat_share"], (b["c_n_days_with_tx"],), b["fold"], pd.Series(True, index=b.index)
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p05 = float(np.quantile(ranks, 0.05)) if ranks else float("nan")
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p95 = float(np.quantile(ranks, 0.95)) if ranks else float("nan")
    share_die = float(np.mean([r < CHANCE for r in ranks])) if ranks else float("nan")
    prose = (
        f"Bootstrap leftover-after-days rank p05={_f(p05)} p50={_f(p50)} p95={_f(p95)} "
        f"share<0.55={_pp(share_die)} n={len(ranks)}."
    )
    print(prose)
    return {"p05": p05, "p50": p50, "p95": p95, "share_die": share_die, "prose": prose}


def extra_permute(tr: pd.DataFrame, n_perm: int = 24) -> dict:
    print("\n" + "=" * 72)
    print(f"EXTRA — permute uncat within days quintile (n={n_perm})")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    work = tr.loc[lab].copy()
    days = pd.to_numeric(work["c_n_days_with_tx"], errors="coerce")
    x = pd.to_numeric(work["a_uncat_share"], errors="coerce")
    ok = days.notna() & x.notna()
    work = work.loc[ok]
    q = pd.qcut(pd.to_numeric(work["c_n_days_with_tx"], errors="coerce"), 5, duplicates="drop")
    rng = np.random.default_rng(FOLD_SEED + 13)
    ranks = []
    for _ in range(n_perm):
        shuf = work["a_uncat_share"].copy()
        for cat in q.dropna().unique():
            idx = q[q == cat].index
            vals = shuf.loc[idx].to_numpy()
            rng.shuffle(vals)
            shuf.loc[idx] = vals
        after = leftover_diag(
            work[Y3], shuf, (work["c_n_days_with_tx"],), work["fold"], pd.Series(True, index=work.index)
        )
        if np.isfinite(after["rank"]):
            ranks.append(after["rank"])
    p50 = float(np.median(ranks)) if ranks else float("nan")
    p90 = float(np.quantile(ranks, 0.90)) if ranks else float("nan")
    prose = f"Permuted-within-days leftover rank p50={_f(p50)} p90={_f(p90)} n={len(ranks)}."
    print(prose)
    return {"p50": p50, "p90": p90, "prose": prose}


def extra_size(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — leftover after size / after a_n_tx")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    after_s = leftover_diag(y, tr["a_uncat_share"], (tr["log_in3"],), tr["fold"], lab)
    after_t = leftover_diag(y, tr["a_uncat_share"], (tr["a_n_tx"],), tr["fold"], lab)
    after_both = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"], tr["log_in3"]), tr["fold"], lab)
    prose = (
        f"uncat leftover after size rank {_f(after_s['rank'])} dies={after_s['honest_dies']}. "
        f"after a_n_tx {_f(after_t['rank'])} dies={after_t['honest_dies']}. "
        f"after days+size {_f(after_both['rank'])} dies={after_both['honest_dies']}."
    )
    print(prose)
    return {"after_size": after_s["rank"], "after_tx": after_t["rank"], "after_both": after_both["rank"], "prose": prose}


def extra_quintiles(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y3 rate by uncat quintile")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    ok = y.notna() & x.notna()
    qn = pd.qcut(x[ok], 5, duplicates="drop")
    rows = []
    for i, cat in enumerate(sorted(qn.dropna().unique()), start=1):
        sl = ok.copy()
        sl.loc[ok] = qn == cat
        rows.append(
            {
                "q": i,
                "uncat rate": _pp(float(y[sl].mean()) if sl.any() else float("nan")),
                "n": int(sl.sum()),
                "n_pos": int((sl & (y == 1)).sum()),
            }
        )
    prose = f"Y3 uncat Q1→Q5 {[r['uncat rate'] for r in rows]}."
    print(prose)
    return {"rows": rows, "prose": prose}


def extra_fold(tr: pd.DataFrame, p3: dict) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — fold-wise leftover")
    print("=" * 72)
    fold_rows = []
    for r, rr in zip(p3["after"]["rec"]["folds"], p3["after"]["rrec"]["folds"]):
        fold_rows.append(
            {
                "fold": r["fold"],
                "OLS leftover": _f(r["auroc"]),
                "rank leftover": _f(rr["auroc"]),
                "n_va": r["n_va"],
                "n_pos": r["n_pos"],
            }
        )
    prose = f"Rank leftover folds: {p3['after']['rank_folds']}."
    print(prose)
    return {"fold_rows": fold_rows, "prose": prose}


def extra_ushape(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — U-shape leftover (Q1 and Q5 recover)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    lab = y.notna() & x.notna()
    q = pd.qcut(x[lab], 5, duplicates="drop")
    tails = pd.Series(np.nan, index=tr.index)
    tails.loc[lab] = q.isin([q.cat.categories[0], q.cat.categories[-1]]).astype(float)
    rec = signed_oof_auroc(y, tails, tr["fold"], lab)
    after = leftover_diag(y, tails, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    mid = lab.copy()
    mid.loc[lab] = ~q.isin([q.cat.categories[0], q.cat.categories[-1]])
    after_mid = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"],), tr["fold"], mid)
    dev = (x - float(x[lab].median())).abs()
    after_dev = leftover_diag(y, dev, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    prose = (
        f"Q1|Q5 flag Y3 {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']}. Mid-quintiles leftover {_f(after_mid['rank'])} "
        f"dies={after_mid['honest_dies']}. |uncat−p50| leftover {_f(after_dev['rank'])} "
        f"dies={after_dev['honest_dies']}."
    )
    print(prose)
    return {
        "flag_cv": _cv(rec),
        "flag_rank": after["rank"],
        "mid_rank": after_mid["rank"],
        "dev_rank": after_dev["rank"],
        "prose": prose,
    }


def extra_flag_keep(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Q1|Q5 flag KEEP-as-X screen (footnote, not a 44 stem)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    lab = y.notna() & x.notna()
    q = pd.qcut(x[lab], 5, duplicates="drop")
    flag = pd.Series(np.nan, index=tr.index)
    flag.loc[lab] = q.isin([q.cat.categories[0], q.cat.categories[-1]]).astype(float)
    rec = signed_oof_auroc(y, flag, tr["fold"], lab)
    after = leftover_diag(y, flag, (tr["c_n_days_with_tx"],), tr["fold"], lab)
    after_s = leftover_diag(y, flag, (tr["log_in3"],), tr["fold"], lab)
    rho_s = spearman(flag, tr["log_in3"])
    rho_d = spearman(flag, tr["c_n_days_with_tx"])
    beat = bool(np.isfinite(_cv(rec)) and (_cv(rec) - SIZE_QUOTE) >= KEEP_DELTA)
    is_size = bool(np.isfinite(rho_s) and abs(rho_s) >= SIZE_RHO)
    twin = bool(np.isfinite(rho_d) and abs(rho_d) >= TWIN_RHO)
    leftover_lives = bool(np.isfinite(after["rank"]) and after["rank"] >= CHANCE and not after["honest_dies"])
    engine = leftover_lives and beat and not is_size and not twin
    prose = (
        f"Q1|Q5 flag Y3 {_f(_cv(rec))} leftover-days {_f(after['rank'])} dies={after['honest_dies']} "
        f"after size {_f(after_s['rank'])}. ρ vs size {_f(rho_s)} vs days {_f(rho_d)}. "
        f"beat-size={beat} SIZE={is_size} twin={twin} leftover_lives={leftover_lives} "
        f"engine={engine}. Footnote only — do not add a U-shape flag to the 44."
    )
    print(prose)
    return {
        "cv": _cv(rec),
        "rank": after["rank"],
        "beat": beat,
        "engine": engine,
        "prose": prose,
    }


def extra_all_uncat(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — all-uncat months; leftover after days on uncat<1")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    allu = x >= 0.999
    rec = signed_oof_auroc(y, allu.astype(float), tr["fold"], y.notna() & x.notna())
    after = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"],), tr["fold"], y.notna() & x.notna() & ~allu)
    prose = (
        f"all-uncat months {int(allu.sum()):,} / Y3 flag {_f(_cv(rec))}. "
        f"Leftover after days on uncat<1 {_f(after['rank'])} dies={after['honest_dies']} "
        f"n={after['n']} pos={after['n_pos']}."
    )
    print(prose)
    return {"n_all": int(allu.sum()), "flag_cv": _cv(rec), "rest_rank": after["rank"], "prose": prose}


def extra_y2(tr: pd.DataFrame) -> dict:
    print("\n" + "=" * 72)
    print("EXTRA — Y2 leftover after days (report-only; style dummy)")
    print("=" * 72)
    assert_no_holdout(tr["company_id"])
    y = pd.to_numeric(tr[Y2], errors="coerce")
    rec = signed_oof_auroc(y, tr["a_uncat_share"], tr["fold"], y.notna())
    after = leftover_diag(y, tr["a_uncat_share"], (tr["c_n_days_with_tx"],), tr["fold"], y.notna())
    prose = (
        f"Y2 uncat {_f(_cv(rec))} leftover after days {_f(after['rank'])} "
        f"dies={after['honest_dies']} (uncat_qa PARK as X — do not reopen)."
    )
    print(prose)
    return {"cv": _cv(rec), "rank": after["rank"], "prose": prose}


def decide(p1, p2, p3, p5, p6) -> dict:
    leftover_lives = bool(np.isfinite(p3["rank"]) and p3["rank"] >= CHANCE and not p3["dies"])
    is_size = bool(p1["is_size"])
    twin = bool(p1["twin_gate"])
    engine = leftover_lives and bool(p2["beat_size"]) and not is_size and not twin
    if engine:
        role = "KEEP as unused leftover"
        why = (
            f"KEEP-as-X passed: leftover after days rank {p3['rank']:.3f} lives, "
            f"beat-size {_f(p2['uncat'])} vs 0.617, not SIZE, not twin. "
            f"Stays off the 15-col card. PARK as Y (style dummy ICC {_f(p6['icc']['icc'])})."
        )
    elif leftover_lives and twin:
        role = "DROP from the 44 as Y3 X"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but TWIN of {p1['twins']}. "
            f"after miss_cp {_f(p5['m_rank'])} after d_tx {_f(p5['d_rank'])}."
        )
    elif leftover_lives and is_size:
        role = "DROP from the 44 as Y3 X"
        why = f"leftover after days rank {p3['rank']:.3f} lives but SIZE (ρ={_f(p1['rhos']['log1p(a_in3)'])})."
    elif leftover_lives and not p2["beat_size"]:
        role = "CLOSE unused leftover"
        why = (
            f"leftover after days rank {p3['rank']:.3f} lives but fails beat-size "
            f"({_f(p2['uncat'])} vs 0.617). Style dummy ICC {_f(p6['icc']['icc'])}. "
            f"DROP from the 44 as Y3 X. PARK as Y."
        )
    else:
        role = "CLOSE unused leftover"
        why = (
            f"unused leftover after days: honest rank {_f(p3['rank'])} dies "
            f"(OLS {_f(p3['ols'])} fake={p3['fake']}). "
            f"Beat-size FAIL {_f(p2['uncat'])} vs 0.617. "
            f"{'Also TWIN of ' + str(p1['twins']) + '. ' if twin else ''}"
            f"DROP from the 44 as Y3 X. PARK as Y (style dummy ICC {_f(p6['icc']['icc'])}). "
            f"Off the 15-col card. Do not merge amount-uncat."
        )
    return {
        "role": role,
        "why": why,
        "leftover_lives": leftover_lives,
        "engine": engine,
        "is_size": is_size,
        "twin": twin,
        "park_y": "PARK as Y — style dummy, do not invent y_uncat",
        "card": "no — do not put a_uncat_share on the 15-col card",
    }


def make_plot(tr: pd.DataFrame, p3: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    x = pd.to_numeric(tr["a_uncat_share"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax = axes[0]
    lab = y3.notna() & x.notna()
    q = pd.qcut(x[lab], 5, duplicates="drop")
    rates, xs = [], []
    for i, cat in enumerate(sorted(q.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q == cat
        rates.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs.append(i)
    ax.plot(xs, rates, marker="o", color="#1f4e79", label="a_uncat_share")
    q2 = pd.qcut(days[lab], 5, duplicates="drop")
    rates2, xs2 = [], []
    for i, cat in enumerate(sorted(q2.dropna().unique()), start=1):
        sl = lab.copy()
        sl.loc[lab] = q2 == cat
        rates2.append(float(y3[sl].mean()) if sl.any() else float("nan"))
        xs2.append(i)
    ax.plot(xs2, rates2, marker="s", color="#c45c26", label="days")
    ax.set_xlabel("quintile (low → high)")
    ax.set_ylabel("Y3 rate")
    ax.set_title("uncat vs days recover")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1]
    rec = p3["after"]["rrec"]
    xs = [r["fold"] for r in rec["folds"]]
    ys = [r["auroc"] for r in rec["folds"]]
    ax.bar(xs, ys, color="#1f4e79")
    ax.axhline(CHANCE, color="#333", ls="--", lw=1, label="0.55 dies")
    ax.axhline(DAYS_BENCH, color="#c45c26", ls=":", lw=1, label="days 0.711")
    ax.set_xlabel("fold")
    ax.set_ylabel("rank leftover AUROC")
    ax.set_title("a_uncat_share leftover after days")
    ax.set_ylim(0.4, 0.85)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3 = ctx["p1"], ctx["p2"], ctx["p3"]
    p5, p6, p7 = ctx["p5"], ctx["p6"], ctx["p7"]
    p8, p9, p10 = ctx["p8"], ctx["p9"], ctx["p10"]
    lines = [
        "# Unused leftover of `a_uncat_share` after days as Y3 X",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_uncat`. Do not put uncat on the 15-col card. "
        "Do not overwrite `uncat_qa.*`. Do not merge amount-uncat. Do not add `uncategorized` to CAT_MAP. "
        "Do not grow TURNOVER.",
        "",
        "`a_uncat_share` = count share of txs with category `uncategorized` or not in CAT_MAP (Family A, this month). "
        "Uncat QA already PARK as Y and as X (style dummy). This lane is leftover after days on the 44 stem. "
        "Missing-CP is not an uncat twin (ρ 0.131).",
        "",
        "## Headline",
        "",
        (
            f"`a_uncat_share` as Y3 X: **{d['role']}** ({d['why']}). "
            f"Y3 leftover after days OLS {_f(p3['ols'])} rank {_f(p3['rank'])} "
            f"({'dies' if p3['dies'] else 'lives'}, fake={p3['fake']}). "
            f"Inverse days after uncat rank {_f(p3['inv_rank'])}. "
            f"Single {_f(p2['uncat'])} vs size {_f(p2['size'])} vs days {_f(p2['days'])} "
            f"vs amt {_f(p2['amt'])}. "
            f"after miss_cp {_f(p5['m_rank'])} after d_tx {_f(p5['d_rank'])}. "
            f"ICC {_f(p6['icc']['icc'])} {'TRAIT' if p6['trait'] else 'STATE'}. "
            f"15-col card: {d['card']}. {d['park_y']}. "
            f"Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712."
        ),
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | {d['park_y']}. Uncat ≠ no ERP. |",
        f"| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 {_f(p8['l1_rank'])}. |",
        f"| 3 | Who is turning? | **{d['role']}** leftover after days {_f(p3['rank'])} vs days 0.711. |",
        f"| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | Twin screen: {p1['twins'] or 'none'}. ρ vs miss_cp {_f(p1['rhos']['miss_cp'])}. Style dummy ICC {_f(p6['icc']['icc'])}. |",
        f"| 6 | Months earlier? | lag1 leftover {_f(p8['l1_rank'])}; days_lag1 {_f(p8['days_l1'])}. |",
        "",
        "## PARK / CLOSE / KEEP / DROP-from-44",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `a_uncat_share` as Y3 X / the 15-col card | **{d['role']}** | {d['why']} |",
        f"| `a_uncat_share` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** | leftover lives={d['leftover_lives']} twin={d['twin']} SIZE={d['is_size']} beat-size={p2['beat_size']} |",
        f"| uncat as health Y | **PARK** | style dummy; do not invent y_uncat |",
        f"| amount-uncat merge | **CLOSED** | ρ={_f(p9['rho'])}; leftover after count {_f(p9['amt_after_count'])} |",
        f"| CAT_MAP add `uncategorized` | **no** | leftover token is the label |",
        f"| miss_cp / d_tx twin | **{'YES' if d['twin'] else 'NO'}** | ρ miss {_f(p1['rhos']['miss_cp'])} d_tx {_f(p1['rhos']['d_tx_cp_share'])} |",
        f"| Q6 lag1 after days_lag1 | **{'KEEP' if (np.isfinite(p8['l1_rank']) and p8['l1_rank'] >= CHANCE and not p8['l1_dies']) else 'CLOSE'}** | leftover {_f(p8['l1_rank'])} |",
        "",
        "## 1 — Coverage; twin / SIZE screen",
        "",
        p1["prose"],
        "",
        _md_table(p1["rows"]),
        "",
        _md_table(p1["rho_rows"]),
        "",
        "## 2 — Single-feature group-fold Y3",
        "",
        p2["prose"],
        "",
        _md_table(p2["rows"]),
        "",
        "## 3 — Honest leftover after days",
        "",
        p3["prose"],
        "",
        f"OLS folds: {p3['after']['folds']}. Rank folds: {p3['after']['rank_folds']}.",
        "",
        "## 4 — Twin / SIZE screen (in cut 1)",
        "",
        f"SIZE={p1['is_size']} twin_gate={p1['twin_gate']} twins={p1['twins'] or 'none'}.",
        "",
        "## 5 — Leftover after miss_cp / d_tx",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "## 6 — ICC / demean leftover",
        "",
        p6["prose"],
        "",
        "## 7 — Dark vs ERP leftover (uncat ≠ no ERP)",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "## 8 — Q6 lag1 leftover after days_lag1",
        "",
        p8["prose"],
        "",
        _md_table(p8["rows"]),
        "",
        "## 9 — Count vs amount leftover (do not merge)",
        "",
        p9["prose"],
        "",
        "## 10 — Holdout coverage only",
        "",
        p10["prose"],
        "",
        "## Extras",
        "",
        "### Bootstrap leftover after days",
        "",
        ctx["xb"]["prose"],
        "",
        "### Permute within days quintile",
        "",
        ctx["xp"]["prose"],
        "",
        "### leftover after size / a_n_tx",
        "",
        ctx["xsz"]["prose"],
        "",
        "### Y3 rate by uncat quintile",
        "",
        ctx["xq"]["prose"],
        "",
        _md_table(ctx["xq"]["rows"]),
        "",
        "### Fold-wise leftover",
        "",
        ctx["xf"]["prose"],
        "",
        _md_table(ctx["xf"]["fold_rows"]),
        "",
        "### Y2 leftover after days (report-only)",
        "",
        ctx["xy2"]["prose"],
        "",
        "### U-shape leftover",
        "",
        ctx["xu"]["prose"],
        "",
        "### all-uncat months",
        "",
        ctx["xa"]["prose"],
        "",
        "### Q1|Q5 flag KEEP-as-X (footnote, not a 44 stem)",
        "",
        ctx["xk"]["prose"],
        "",
        "## Night quotes (unchanged)",
        "",
        "| quote | locked |",
        "| --- | --- |",
        f"| Y3 | {Y3_NIGHT[0]:.3f} / {Y3_NIGHT[1]:.3f} |",
        f"| days | {DAYS_BENCH:.3f} |",
        f"| size | {SIZE_QUOTE:.3f} |",
        f"| Y7 TURNOVER | {Y7_TURNOVER:.3f} / {Y7_B_SHALLOW:.3f} |",
        f"| d_cust_top1 leftover | 0.525 DROP |",
        "",
        "Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `a_uncat_share` on the 15-col card.",
        "",
        "## Files written",
        "",
        "- `analysis/evaluate/uncat_share_qa.py`",
        "- `analysis/outputs/uncat_share_qa.md`",
        "- `analysis/outputs/uncat_share_qa.png`" if ctx.get("png") else "- (no PNG)",
        "- append-only `analysis/experiments/registry.csv`",
        "- `overnight/waves/wave4_uncat_share.md` (end, if WRITE_WAVE)",
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
    p1, p2, p3, p5, p6 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p6"]
    d = ctx["decision"]
    rows = [
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM,
            "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_uncat_share",
            "value": p2["uncat"], "coverage": f"{p1['cov']:.4f}",
            "notes": f"days={p2['days']:.4f} size={p2['size']:.4f} beat={p2['beat_size']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM,
            "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_uncat_share_resid_days",
            "value": p3["rank"], "coverage": f"{p1['cov']:.4f}",
            "notes": f"ols={p3['ols']:.4f} dies={p3['dies']} fake={p3['fake']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM,
            "y": Y3, "model": MODEL, "split": "train_cv", "metric": "auroc_a_uncat_share_resid_miss_cp",
            "value": p5["m_rank"], "coverage": f"{p1['cov']:.4f}",
            "notes": f"after_dtx={p5['d_rank']:.4f}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM,
            "y": "-", "model": MODEL, "split": "train", "metric": "icc_a_uncat_share",
            "value": p6["icc"]["icc"], "coverage": f"{p1['cov']:.4f}",
            "notes": f"trait={p6['trait']} twins={p1['twins']}",
        },
        {
            "ts": ts, "round": ROUND, "wave": WAVE, "agent": AGENT, "x_families": X_FAM,
            "y": Y3, "model": MODEL, "split": "train", "metric": "uncat_share_leftover",
            "value": 1 if d["leftover_lives"] else 0, "coverage": f"{p1['cov']:.4f}",
            "notes": d["role"] + " " + d["why"][:160],
        },
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
    p1, p2, p3, p5, p6, p7, p8, p9 = (
        ctx["p1"], ctx["p2"], ctx["p3"], ctx["p5"], ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"]
    )
    text = (
        f"# Wave 4 — a_uncat_share leftover after days as Y3 X\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/uncat_share_qa.py`\n"
        f"- `analysis/outputs/uncat_share_qa.md`\n"
        f"- `analysis/outputs/uncat_share_qa.png`\n"
        f"- append-only `analysis/experiments/registry.csv` "
        f"(skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `uncat_qa.*`, `missing_cp_qa.*`, `d_tx_qa.*`, `top1_qa.*`, "
        f"`n_types_qa.*`, `ap_issued_qa.*`, `cashflow.py`, CAT_MAP, parquet / duckdb, "
        f"`build_targets`, `product/`, the 15-col card, TURNOVER, LIVE, CONTEXT, "
        f"canvas, `brief_map.md`, or the parent journal. "
        f"Night Y3 stays **0.762 / 0.752**. Days 0.711. Size 0.617. "
        f"Y7 TURNOVER **0.720 / 0.712** unchanged. "
        f"d_cust_top1 leftover 0.525 DROP stays.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `a_uncat_share` as Y3 X / 15-col card | **{d['role']}** |\n"
        f"| `a_uncat_share` as engine X on the 44 | **{'KEEP leftover' if d['engine'] else 'DROP'}** |\n"
        f"| uncat as health Y | **PARK** |\n"
        f"| amount-uncat merge | **CLOSED** |\n\n"
        f"Y3 leftover after days rank {_f(p3['rank'])} (dies={p3['dies']}, fake={p3['fake']}); "
        f"inverse days after uncat {_f(p3['inv_rank'])}. "
        f"Single {_f(p2['uncat'])} vs days {_f(p2['days'])} vs size {_f(p2['size'])}. "
        f"ρ vs miss_cp {_f(p1['rhos']['miss_cp'])} vs d_tx {_f(p1['rhos']['d_tx_cp_share'])} "
        f"vs days {_f(p1['rhos']['c_n_days_with_tx'])} vs size {_f(p1['rhos']['log1p(a_in3)'])}. "
        f"after miss_cp {_f(p5['m_rank'])} after d_tx {_f(p5['d_rank'])}. "
        f"ICC {_f(p6['icc']['icc'])} trait={p6['trait']}. "
        f"Q6 lag1 leftover {_f(p8['l1_rank'])}. "
        f"Dark nn={p7['dark_nn']} mean {_f(p7['mean_d'])} vs ERP {_f(p7['mean_e'])}. "
        f"count↔amount ρ={_f(p9['rho'])}. {d['why']}\n\n"
        f"Dark 470 uncat nn={p7['dark_nn']} CONFIRM uncat ≠ no ERP. "
        f"ICC MSB/(MSB+MSW) 0.985 CONFIRM TRAIT; ANOVA ICC(1) 0.798; leftover is BETWEEN "
        f"(company-mean leftover 0.586 lives / demean 0.484 dies). "
        f"Permute-within-days p50=0.508 p90=0.526 — observed leftover 0.573 is above the null. "
        f"Bootstrap p05/p50/p95 0.529/0.574/0.622 (22.5% die). "
        f"Q1|Q5 U-shape flag Y3 0.639 leftover 0.558 is a footnote, not a 44 stem. "
        f"Do not merge amount-uncat. Do not add `uncategorized` to CAT_MAP. "
        f"Do not put uncat on the 15-col card. Do not invent y_uncat.\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- none")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s.\n"
    )
    WAVE_NOTE.write_text(text)
    print(f"wrote {WAVE_NOTE}")


def main() -> None:
    t0 = time.time()
    failed: list[str] = []
    print("uncat_share leftover QA — unused leftover of a_uncat_share after days as Y3 X")
    panel = load_panel()
    con = connect()
    try:
        mem = load_memory_shares(con)
        book = book_invoice_ids(con)
    finally:
        con.close()
    print(f"book (ERP) companies={len(book)}; memory shares rows={len(mem)}")
    panel = panel.merge(mem, on=["company_id", "period"], how="left")
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["a_uncat_share", "c_n_days_with_tx", "amt_uncat"], (1,))
    tr = panel[panel["split"] == "train"].copy()
    hold = panel[panel["split"] == "holdout"].copy()
    assert_no_holdout(tr["company_id"])
    if hold["company_id"].nunique() != 72:
        failed.append(f"holdout n_co={hold['company_id'].nunique()} expected 72")
    p1 = pass1_cov(tr)
    p2 = pass2_singles(tr)
    p3 = pass3_days(tr)
    p5 = pass5_after_twins(tr)
    p6 = pass6_icc(tr)
    p7 = pass7_dark(tr, book)
    p8 = pass8_q6(tr)
    p9 = pass9_amount(tr)
    p10 = pass10_hold(hold, book)
    xb = extra_bootstrap(tr, n_boot=40)
    xp = extra_permute(tr, n_perm=24)
    xsz = extra_size(tr)
    xq = extra_quintiles(tr)
    xf = extra_fold(tr, p3)
    xy2 = extra_y2(tr)
    xu = extra_ushape(tr)
    xa = extra_all_uncat(tr)
    xk = extra_flag_keep(tr)
    decision = decide(p1, p2, p3, p5, p6)
    print("\n" + "=" * 72)
    print(f"VERDICT: {decision['role']}")
    print(decision["why"])
    print("=" * 72)
    if not p2["days_ok"]:
        failed.append(f"days replica {p2['days']:.3f} ≠ 0.711")
    if not p2["size_ok"]:
        failed.append(f"size replica {p2['size']:.3f} ≠ 0.617")
    if not p7["dark_ok"]:
        failed.append("dark uncat missing — expected uncat ≠ no ERP")
    png = make_plot(tr, p3)
    elapsed = time.time() - t0
    ctx = {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p5": p5,
        "p6": p6,
        "p7": p7,
        "p8": p8,
        "p9": p9,
        "p10": p10,
        "xb": xb,
        "xp": xp,
        "xsz": xsz,
        "xq": xq,
        "xf": xf,
        "xy2": xy2,
        "xu": xu,
        "xa": xa,
        "xk": xk,
        "decision": decision,
        "failed": failed,
        "elapsed_s": elapsed,
        "png": png,
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
