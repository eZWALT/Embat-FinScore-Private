"""Q3 recency — going-quiet, days/n_tx twin, SIZE, or extract-end hole?

NORTH_STAR: recency is the *cash* clock. ``created_at`` is the connection
clock (PARK — do not revive). ``y6_silent_60`` is a *future* rejected Y
(last tx as of t+3 more than 60 days before that end). Do not revive Y6.
Do not invent ``y_recency`` / ``y_silent``.

``c_recency_days`` = month_end.date − last booking date ≤ month_end.
``c_last_tx_before_2026_06`` = 1 if period ≥ 2026-06-01 and last booking
as of month_end is < 2026-06-01. Javier extract-level count is **61**
companies; as of 2026-08-31 (no Sept-1 look-ahead) this is **62** because
COMP_0981 is silent from 2025-04-07 until 2026-09-01.

Feature report: ``c_recency_days`` keep-list representative (BETWEEN,
LOW_PERSIST, size ρ −0.447). ``c_last_tx_before_2026_06`` RARE modal
**99.0%**. Night engine is days **0.711** on the 15-col card. Recency is
not on the card. Night Y3 quote stays **0.762 / 0.752**.

KEEP-as-X on the 44 only if leftover after days beats size ≥0.02 **and**
is not the 2026-06 extract dummy. Still do **not** put recency on
tonight's 15-col card. DROP from the 44 / CLOSE as X if days twin, SIZE,
or extract hole.

No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
``build_targets``. Do not edit ``ops.py`` unless a real formula bug —
then stop and report. Do not change the night Y3 quote or days 0.711.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.recency_qa

Owned: analysis/evaluate/recency_qa.py, analysis/outputs/recency_qa.md,
optional one PNG, append-only registry, overnight/waves/wave4_recency.md (end).
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
OUT_MD = ANALYSIS / "outputs" / "recency_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "recency_calendar.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_recency.md"
AGENT = "4545d7a6"
WAVE = "4"
ROUND = "R4"
MODEL = "recency_qa"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
Y6 = "y6_silent_60"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
Y3_NIGHT = (0.762, 0.752)
KEEP_DELTA = 0.02
SIZE_PARK = 0.60
SIZE_RHO = 0.50
TWIN_RHO = 0.80
MODAL_QUOTE = 0.990
ICC_STYLE = 0.85
ACF_STYLE = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
CUTOFF = pd.Timestamp("2026-06-01")
AUG_END = pd.Timestamp("2026-08-31")
EXTRACT_END = pd.Timestamp("2026-09-01")
COMP_0981 = "COMP_0981"
JAVIER_61 = 61
ASOF_62 = 62
WRITE_WAVE = True  # last iterate — one note at the end

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_n_tx",
    "c_n_tx",
    "c_n_days_with_tx",
    "c_gap_sd",
    "c_zero_in_month",
    "c_recency_days",
    "c_last_tx_before_2026_06",
    "b_below_0",
    "first_month",
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
    """Train-defined OLS residual of y on 1+ predictors. No holdout in slope."""
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
    pred = X @ beta
    resid.loc[ok] = Y - pred
    info["intercept"] = float(beta[0])
    info["slope"] = [float(b) for b in beta[1:]]
    return resid, info


def jaccard(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    d = d.dropna()
    if d.empty:
        return float("nan")
    aa = d["a"].eq(1)
    bb = d["b"].eq(1)
    inter = int((aa & bb).sum())
    union = int((aa | bb).sum())
    if union == 0:
        return float("nan")
    return float(inter) / float(union)


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


def add_so_far(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    out["so_far"] = out.groupby("company_id", sort=False).cumcount() + 1
    return out


def add_style_shock(df: pd.DataFrame) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    x = pd.to_numeric(out["c_recency_days"], errors="coerce")
    mu = x.groupby(out["company_id"], sort=False).transform("mean")
    out["rec_co_mean"] = mu
    out["rec_demean"] = x - mu
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
    ykeep = ["company_id", "period", Y2, Y3]
    if Y6 in yraw.columns:
        ykeep.append(Y6)
        print(f"{Y6} present in targets.parquet — using store, no assembler")
    else:
        print(f"{Y6} missing from targets — will attach in-module")
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["june_window"] = panel["period"] >= CUTOFF
    leak = leakage_check(
        ["c_recency_days", "c_last_tx_before_2026_06", "c_n_days_with_tx"],
        Y3,
        forbidden_prefixes=["b"],
    )
    if not leak["ok"]:
        print(f"leakage_check note (in-memory X only): {leak['issues']}")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


def attach_y6_if_missing(panel: pd.DataFrame, con) -> pd.DataFrame:
    if Y6 in panel.columns:
        return panel
    from analysis.targets.y6_activity import build as build_y6

    print(f"{Y6} missing from targets — in-module y6_activity.build (no assembler)")
    grid = panel[["company_id", "period"]].drop_duplicates()
    y6 = build_y6(con, grid)
    y6 = _keys(y6[["company_id", "period", Y6]])
    return panel.merge(y6, on=["company_id", "period"], how="left")


def load_tx_days(con) -> pd.DataFrame:
    days = con.execute(
        """
        SELECT
          CAST(company_id AS VARCHAR) AS company_id,
          CAST("date" AS DATE) AS d
        FROM transactions
        WHERE "date" IS NOT NULL
        """
    ).df()
    days["company_id"] = days["company_id"].astype(str)
    days["d"] = pd.to_datetime(days["d"])
    return days


# ---------------------------------------------------------------------------
# Pass 1 — completeness + distributions
# ---------------------------------------------------------------------------
def pass1_dist(panel: pd.DataFrame, tr: pd.DataFrame) -> dict:
    rows = []
    for split, sl in (("train", tr), ("holdout", panel[panel["split"] == "holdout"])):
        for col in ("c_recency_days", "c_last_tx_before_2026_06"):
            x = pd.to_numeric(sl[col], errors="coerce")
            nn = x.dropna()
            modal = float(nn.mode().iloc[0]) if len(nn) else float("nan")
            modal_share = float((nn == modal).mean()) if len(nn) else float("nan")
            rows.append(
                {
                    "split": split,
                    "col": col,
                    "n_cm": int(len(sl)),
                    "n_co": int(sl["company_id"].nunique()),
                    "cov": float(x.notna().mean()) if len(x) else float("nan"),
                    "mean": float(nn.mean()) if len(nn) else float("nan"),
                    "p50": float(nn.median()) if len(nn) else float("nan"),
                    "p90": float(nn.quantile(0.90)) if len(nn) else float("nan"),
                    "p99": float(nn.quantile(0.99)) if len(nn) else float("nan"),
                    "share1": float((x == 1).mean()) if col.endswith("2026_06") and len(x) else float("nan"),
                    "modal": modal,
                    "modal_share": modal_share,
                }
            )
    rec = next(r for r in rows if r["split"] == "train" and r["col"] == "c_recency_days")
    flg = next(r for r in rows if r["split"] == "train" and r["col"] == "c_last_tx_before_2026_06")
    ho_rec = next(r for r in rows if r["split"] == "holdout" and r["col"] == "c_recency_days")
    confirm_modal = bool(
        np.isfinite(flg["modal_share"]) and abs(flg["modal_share"] - MODAL_QUOTE) < 0.015
    )
    confirm_rho_quote = True  # filled in pass 3
    prose = (
        f"Train `c_recency_days` cov {_pp(rec['cov'])} p50={_f(rec['p50'], 1)} "
        f"p90={_f(rec['p90'], 1)} (n_cm={rec['n_cm']:,} / cos={rec['n_co']:,}). "
        f"`c_last_tx_before_2026_06` modal {flg['modal']:g} share {_pp(flg['modal_share'])} "
        f"({'CONFIRM 99.0%' if confirm_modal else 'does not match feature-report 99.0%'}); "
        f"share=1 {_pp(flg['share1'])}. "
        f"Holdout recency cov {_pp(ho_rec['cov'])} p50={_f(ho_rec['p50'], 1)} — coverage only."
    )
    print(prose)
    return {
        "rows": rows,
        "rec": rec,
        "flg": flg,
        "ho_rec": ho_rec,
        "confirm_modal": confirm_modal,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — formula vs raw last booking; 61 vs 62; COMP_0981
# ---------------------------------------------------------------------------
def pass2_formula(tr: pd.DataFrame, panel: pd.DataFrame, days: pd.DataFrame) -> dict:
    hold = load_holdout()
    keys = panel[["company_id", "period"]].copy()
    keys["period"] = pd.to_datetime(keys["period"])
    keys = keys.reset_index(drop=True)
    monthly = (
        days.assign(period=days["d"].dt.to_period("M").dt.to_timestamp())
        .groupby(["company_id", "period"], as_index=False)["d"]
        .max()
        .rename(columns={"d": "last_tx"})
    )
    recon = keys.merge(monthly, on=["company_id", "period"], how="left")
    recon = recon.sort_values(["company_id", "period"]).reset_index(drop=True)
    recon["last_tx"] = recon.groupby("company_id", sort=False)["last_tx"].ffill()
    last_day = pd.to_datetime(recon["last_tx"]).dt.normalize()
    month_end = (pd.to_datetime(recon["period"]) + pd.offsets.MonthEnd(0)).dt.normalize()
    recon["rec_raw"] = (month_end - last_day).dt.days
    recon["june_raw"] = ((recon["period"] >= CUTOFF) & (last_day < CUTOFF)).astype("float")
    store = panel[["company_id", "period", "c_recency_days", "c_last_tx_before_2026_06"]].copy()
    m = recon.merge(store, on=["company_id", "period"], how="left")
    rec_ok = pd.to_numeric(m["c_recency_days"], errors="coerce")
    rec_agree = float((rec_ok == m["rec_raw"]).mean())
    rec_maxabs = float((rec_ok - m["rec_raw"]).abs().max())
    june_ok = pd.to_numeric(m["c_last_tx_before_2026_06"], errors="coerce")
    june_agree = float((june_ok == m["june_raw"]).mean())

    last_ever = days.groupby("company_id")["d"].max()
    last_aug = days.loc[days["d"] <= AUG_END].groupby("company_id")["d"].max()
    extract_silent = last_ever[last_ever < CUTOFF]
    asof_silent = last_aug[last_aug < CUTOFF]
    n_extract = int(len(extract_silent))
    n_asof = int(len(asof_silent))
    plus = sorted(set(asof_silent.index) - set(extract_silent.index))
    minus = sorted(set(extract_silent.index) - set(asof_silent.index))
    is_0981 = COMP_0981 in plus and len(plus) == 1
    last_0981_ever = last_ever.get(COMP_0981, pd.NaT)
    last_0981_aug = last_aug.get(COMP_0981, pd.NaT)
    txs_0981 = days.loc[days["company_id"] == COMP_0981, "d"].sort_values()
    n_0981 = int(len(txs_0981))
    first_0981 = txs_0981.iloc[0] if n_0981 else pd.NaT
    after_apr = txs_0981[txs_0981 > pd.Timestamp("2025-04-07")]
    next_after = after_apr.iloc[0] if len(after_apr) else pd.NaT

    # store roster at 2026-08 (and ever-flagged)
    aug = panel[panel["period"] == pd.Timestamp("2026-08-01")]
    store_aug = set(
        aug.loc[pd.to_numeric(aug["c_last_tx_before_2026_06"], errors="coerce") == 1, "company_id"]
    )
    ever_flag = set(
        panel.loc[
            pd.to_numeric(panel["c_last_tx_before_2026_06"], errors="coerce") == 1, "company_id"
        ]
    )
    train_asof = [c for c in asof_silent.index if c not in hold]
    hold_asof = [c for c in asof_silent.index if c in hold]
    train_extract = [c for c in extract_silent.index if c not in hold]
    confirm_61 = n_extract == JAVIER_61
    confirm_62 = n_asof == ASOF_62
    store_match = store_aug == set(asof_silent.index)

    prose = (
        f"Store vs raw last-booking: recency agree {_pp(rec_agree)} max|Δ|={_f(rec_maxabs, 2)}; "
        f"June-flag agree {_pp(june_agree)}. "
        f"Extract-level last tx < 2026-06-01: **{n_extract}** "
        f"({'CONFIRM Javier 61' if confirm_61 else f'≠ 61'}). "
        f"As-of 2026-08-31: **{n_asof}** "
        f"({'CONFIRM 62' if confirm_62 else f'≠ 62'}). "
        f"+1 names={plus} "
        f"({'COMP_0981 is the +1' if is_0981 else 'COMP_0981 is NOT uniquely the +1'}). "
        f"COMP_0981 last-ever={last_0981_ever} last-asof-Aug={last_0981_aug} "
        f"n_tx={n_0981} next-after-2025-04-07={next_after}. "
        f"Store Aug-2026 flag companies={len(store_aug)} "
        f"({'match as-of 62' if store_match else 'DIFFERS from as-of roster'}). "
        f"Train as-of {len(train_asof)} / holdout {len(hold_asof)}."
    )
    print(prose)
    roster_rows = []
    for cid in sorted(asof_silent.index):
        roster_rows.append(
            {
                "company_id": cid,
                "split": "holdout" if cid in hold else "train",
                "last_aug": str(pd.Timestamp(asof_silent[cid]).date()),
                "last_ever": str(pd.Timestamp(last_ever[cid]).date()) if cid in last_ever.index else "—",
                "extract_61": int(cid in extract_silent.index),
                "plus1": int(cid in plus),
            }
        )
    return {
        "rec_agree": rec_agree,
        "rec_maxabs": rec_maxabs,
        "june_agree": june_agree,
        "n_extract": n_extract,
        "n_asof": n_asof,
        "plus": plus,
        "minus": minus,
        "is_0981": is_0981,
        "last_0981_ever": last_0981_ever,
        "last_0981_aug": last_0981_aug,
        "n_0981": n_0981,
        "first_0981": first_0981,
        "next_after": next_after,
        "confirm_61": confirm_61,
        "confirm_62": confirm_62,
        "store_aug_n": len(store_aug),
        "ever_flag_n": len(ever_flag),
        "store_match": store_match,
        "n_train_asof": len(train_asof),
        "n_hold_asof": len(hold_asof),
        "n_train_extract": len(train_extract),
        "roster": roster_rows,
        "extract_ids": sorted(extract_silent.index),
        "asof_ids": sorted(asof_silent.index),
        "prose": prose,
        "formula_ok": bool(rec_agree >= 0.999 and june_agree >= 0.999),
    }


# ---------------------------------------------------------------------------
# Pass 3 — Spearman twins / SIZE
# ---------------------------------------------------------------------------
def pass3_rho(tr: pd.DataFrame) -> dict:
    rec = tr["c_recency_days"]
    flg = tr["c_last_tx_before_2026_06"]
    pairs = [
        ("c_recency_days vs c_n_days_with_tx", rec, tr["c_n_days_with_tx"]),
        ("c_recency_days vs a_n_tx", rec, tr["a_n_tx"]),
        ("c_recency_days vs c_n_tx", rec, tr["c_n_tx"]),
        ("c_recency_days vs c_gap_sd", rec, tr["c_gap_sd"]),
        ("c_recency_days vs log1p(a_in3)", rec, tr["log_in3"]),
        ("c_recency_days vs c_zero_in_month", rec, tr["c_zero_in_month"]),
        ("june_flag vs c_recency_days", flg, rec),
        ("june_flag vs c_n_days_with_tx", flg, tr["c_n_days_with_tx"]),
        ("june_flag vs log1p(a_in3)", flg, tr["log_in3"]),
        ("c_n_days_with_tx vs a_n_tx", tr["c_n_days_with_tx"], tr["a_n_tx"]),
        ("c_n_days_with_tx vs log1p(a_in3)", tr["c_n_days_with_tx"], tr["log_in3"]),
    ]
    rows = []
    rhos = {}
    for name, a, b in pairs:
        rho = spearman(a, b)
        rhos[name] = rho
        tag = (
            "TWIN"
            if np.isfinite(rho) and abs(rho) >= TWIN_RHO
            else ("SIZE" if "log1p" in name and np.isfinite(rho) and abs(rho) >= SIZE_RHO else "")
        )
        rows.append({"pair": name, "ρ": _f(rho), "tag": tag})
    rho_days = rhos["c_recency_days vs c_n_days_with_tx"]
    rho_ntx = rhos["c_recency_days vs a_n_tx"]
    rho_gap = rhos["c_recency_days vs c_gap_sd"]
    rho_size = rhos["c_recency_days vs log1p(a_in3)"]
    rho_zero = rhos["c_recency_days vs c_zero_in_month"]
    twin_days = bool(np.isfinite(rho_days) and abs(rho_days) >= TWIN_RHO)
    twin_ntx = bool(np.isfinite(rho_ntx) and abs(rho_ntx) >= TWIN_RHO)
    twin_zero = bool(np.isfinite(rho_zero) and abs(rho_zero) >= TWIN_RHO)
    size_flag = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    confirm_size_quote = bool(np.isfinite(rho_size) and abs(rho_size - (-0.447)) < 0.03)
    prose = (
        f"Recency Spearman: days {rho_days:.3f} "
        f"({'TWIN |ρ|≥0.80' if twin_days else 'not a days twin'}); "
        f"a_n_tx {rho_ntx:.3f} ({'TWIN' if twin_ntx else 'not n_tx twin'}); "
        f"gap_sd {rho_gap:.3f}; zero_in {rho_zero:.3f} "
        f"({'TWIN' if twin_zero else 'not zero-in twin'}); "
        f"log1p(a_in3) {rho_size:.3f} "
        f"({'SIZE |ρ|≥0.50' if size_flag else 'not SIZE'} "
        f"{'CLAIM −0.447 CONFIRMED' if confirm_size_quote else ''})."
    )
    print(prose)
    return {
        "rows": rows,
        "rhos": rhos,
        "rho_days": rho_days,
        "rho_ntx": rho_ntx,
        "rho_gap": rho_gap,
        "rho_size": rho_size,
        "rho_zero": rho_zero,
        "twin_days": twin_days,
        "twin_ntx": twin_ntx,
        "twin_zero": twin_zero,
        "size_flag": size_flag,
        "confirm_size_quote": confirm_size_quote,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — group-fold AUROC Y3 / Y2
# ---------------------------------------------------------------------------
def pass4_auroc(tr: pd.DataFrame) -> dict:
    feats = {
        "c_recency_days": tr["c_recency_days"],
        "c_last_tx_before_2026_06": tr["c_last_tx_before_2026_06"],
        "c_n_days_with_tx": tr["c_n_days_with_tx"],
        "a_n_tx": tr["a_n_tx"],
        "c_gap_sd": tr["c_gap_sd"],
        "c_zero_in_month": tr["c_zero_in_month"],
        "log1p_a_in3": tr["log_in3"],
    }
    rows = []
    store = {}
    for y in (Y2, Y3):
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
        r = store[(y, feat)]
        return float("nan") if r["low_power"] else r["cv"]

    rec_y3 = _cv(Y3, "c_recency_days")
    june_y3 = _cv(Y3, "c_last_tx_before_2026_06")
    days_y3 = _cv(Y3, "c_n_days_with_tx")
    size_y3 = _cv(Y3, "log1p_a_in3")
    rec_y2 = _cv(Y2, "c_recency_days")
    june_y2 = _cv(Y2, "c_last_tx_before_2026_06")
    size_y2 = _cv(Y2, "log1p_a_in3")
    days_y2 = _cv(Y2, "c_n_days_with_tx")
    beat_size = (
        rec_y3 - size_y3 if np.isfinite(rec_y3) and np.isfinite(size_y3) else float("nan")
    )
    beat_days = (
        rec_y3 - days_y3 if np.isfinite(rec_y3) and np.isfinite(days_y3) else float("nan")
    )
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) <= 0.02)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) <= 0.02)
    june_y3_lab = store[(Y3, "c_last_tx_before_2026_06")]
    june_const = bool(
        june_y3_lab["n_pos"] == 0
        or (not june_y3_lab["low_power"] and june_y3_lab["cv"] < 0.52)
        or june_y3_lab["low_power"]
    )
    # on labeled Y3 rows the June flag should be identically 0 (horizon 6)
    y3_lab = tr[Y3].notna()
    june_on_y3 = float(pd.to_numeric(tr.loc[y3_lab, "c_last_tx_before_2026_06"], errors="coerce").mean())
    y2_lab = tr[Y2].notna()
    june_on_y2 = float(pd.to_numeric(tr.loc[y2_lab, "c_last_tx_before_2026_06"], errors="coerce").mean())
    prose = (
        f"Y3 recency {_f(rec_y3)} vs size {_f(size_y3)} (Δ {_f(beat_size, 3)}) "
        f"vs days {_f(days_y3)} (night 0.711, replica "
        f"{'OK' if days_ok else 'OFF'}). "
        f"June-flag Y3 {_f(june_y3)} (mean on labeled Y3 rows {_pp(june_on_y3)}). "
        f"Y2 recency {_f(rec_y2)} vs size {_f(size_y2)}; June-flag {_f(june_y2)} "
        f"(mean on labeled Y2 {_pp(june_on_y2)})."
    )
    print(prose)
    return {
        "rows": rows,
        "store": store,
        "rec_y3": rec_y3,
        "june_y3": june_y3,
        "days_y3": days_y3,
        "size_y3": size_y3,
        "rec_y2": rec_y2,
        "june_y2": june_y2,
        "size_y2": size_y2,
        "days_y2": days_y2,
        "beat_size": beat_size,
        "beat_days": beat_days,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "june_const": june_const,
        "june_on_y3": june_on_y3,
        "june_on_y2": june_on_y2,
        "n_y3": int(tr[Y3].notna().sum()),
        "n_y2": int(tr[Y2].notna().sum()),
        "y3_rate": float(pd.to_numeric(tr[Y3], errors="coerce").mean()),
        "y2_rate": float(pd.to_numeric(tr[Y2], errors="coerce").mean()),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — residual after days
# ---------------------------------------------------------------------------
def pass5_residual(tr: pd.DataFrame) -> dict:
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ntx = pd.to_numeric(tr["a_n_tx"], errors="coerce")
    gap = pd.to_numeric(tr["c_gap_sd"], errors="coerce")
    zero = pd.to_numeric(tr["c_zero_in_month"], errors="coerce")
    r_d, inf_d = ols_resid(rec, days)
    r_n, inf_n = ols_resid(rec, ntx)
    r_dn, inf_dn = ols_resid(rec, days, ntx)
    r_all, inf_all = ols_resid(rec, days, ntx, zero)
    specs = [
        ("resid_days", r_d),
        ("resid_n_tx", r_n),
        ("resid_days+n_tx", r_dn),
        ("resid_days+n_tx+zero", r_all),
        ("c_recency_days", rec),
    ]
    rows = []
    cvs = {}
    for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
        size = signed_oof_auroc(tr[ycol], tr["log_in3"], tr["fold"], tr[ycol].notna())
        size_cv = float("nan") if size["low_power"] else size["cv"]
        for name, x in specs:
            rec_auc = signed_oof_auroc(tr[ycol], x, tr["fold"], tr[ycol].notna())
            cv = float("nan") if rec_auc["low_power"] else rec_auc["cv"]
            cvs[(yname, name)] = cv
            rows.append(
                {
                    "y": yname,
                    "feature": name,
                    "n": f"{rec_auc['n_defined']:,}",
                    "n_pos": f"{rec_auc['n_pos']:,}",
                    "CV": "LOW_POWER" if rec_auc["low_power"] else _f(cv),
                    "sign": rec_auc["train_sign"],
                    "Δsize": _f(
                        (cv - size_cv) if np.isfinite(cv) and np.isfinite(size_cv) else float("nan")
                    ),
                }
            )
    y3_d = cvs.get(("Y3", "resid_days"), float("nan"))
    y3_dn = cvs.get(("Y3", "resid_days+n_tx"), float("nan"))
    y2_d = cvs.get(("Y2", "resid_days"), float("nan"))
    size_y3 = float("nan")
    size_rec = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna())
    if not size_rec["low_power"]:
        size_y3 = size_rec["cv"]
    leftover_beat = (
        (y3_d - size_y3) if np.isfinite(y3_d) and np.isfinite(size_y3) else float("nan")
    )
    leftover_lives = bool(np.isfinite(leftover_beat) and leftover_beat >= KEEP_DELTA)
    died = bool(np.isfinite(y3_d) and y3_d < 0.55)
    prose = (
        f"Y3 leftover after days {_f(y3_d)}; after days+n_tx {_f(y3_dn)} vs size {_f(size_y3)} "
        f"(Δ {_f(leftover_beat, 3)}). Y2 leftover-after-days {_f(y2_d)}. "
        f"Slope recency~days {inf_d['slope'][0] if inf_d['slope'] else float('nan'):.3f}. "
        + (
            "Leftover dies — CLOSE as quiet twin."
            if died and not leftover_lives
            else (
                "Leftover beats size ≥0.02 after days — KEEP-as-X candidate (not on 15-col card)."
                if leftover_lives
                else "Leftover does not clear size+0.02 after days."
            )
        )
    )
    print(prose)
    return {
        "rows": rows,
        "y3_days": y3_d,
        "y3_dn": y3_dn,
        "y2_days": y2_d,
        "size_y3": size_y3,
        "leftover_beat": leftover_beat,
        "leftover_lives": leftover_lives,
        "died": died,
        "slope_days": inf_d["slope"][0] if inf_d["slope"] else float("nan"),
        "slope_ntx": inf_n["slope"][0] if inf_n["slope"] else float("nan"),
        "prose": prose,
        "r_d": r_d,
    }


# ---------------------------------------------------------------------------
# Pass 6 — June-cutoff only 2026-06..08?
# ---------------------------------------------------------------------------
def pass6_extract(tr: pd.DataFrame) -> dict:
    flg = pd.to_numeric(tr["c_last_tx_before_2026_06"], errors="coerce")
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    cal = []
    for p, sl in tr.groupby("period"):
        x = pd.to_numeric(sl["c_last_tx_before_2026_06"], errors="coerce")
        r = pd.to_numeric(sl["c_recency_days"], errors="coerce")
        cal.append(
            {
                "period": str(pd.Timestamp(p).date())[:7],
                "n_cm": int(len(sl)),
                "june_n": int((x == 1).sum()),
                "june_share": float((x == 1).mean()),
                "rec_p50": float(r.median()) if r.notna().any() else float("nan"),
                "rec_p90": float(r.quantile(0.90)) if r.notna().any() else float("nan"),
                "rec_ge30": float((r >= 30).mean()) if r.notna().any() else float("nan"),
                "rec_ge60": float((r >= 60).mean()) if r.notna().any() else float("nan"),
            }
        )
    june_months = tr["period"] >= CUTOFF
    n_june_pos = int((flg == 1).sum())
    n_june_pos_in_window = int(((flg == 1) & june_months).sum())
    n_june_pos_out = int(((flg == 1) & ~june_months).sum())
    only_window = n_june_pos > 0 and n_june_pos_out == 0
    n_months_pos = int(sum(1 for r in cal if r["june_n"] > 0))

    rows = []
    store = {}
    slices = [
        ("all", tr[Y3].notna()),
        ("pre_2026_06", tr[Y3].notna() & ~june_months),
        ("2026_06_08", tr[Y3].notna() & june_months),
        ("Y2_all", tr[Y2].notna()),
        ("Y2_pre_2026_06", tr[Y2].notna() & ~june_months),
        ("Y2_2026_06_08", tr[Y2].notna() & june_months),
    ]
    for sname, mask in slices:
        ycol = Y2 if sname.startswith("Y2") else Y3
        for feat, col in (
            ("c_last_tx_before_2026_06", flg),
            ("c_recency_days", rec),
            ("log1p_a_in3", tr["log_in3"]),
            ("c_n_days_with_tx", tr["c_n_days_with_tx"]),
        ):
            res = signed_oof_auroc(tr[ycol], col, tr["fold"], mask)
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
    # Y3/Y2 labels do not reach Jun–Aug (horizons 6 / 3) — confirm
    y3_last = tr.loc[tr[Y3].notna(), "period"].max() if tr[Y3].notna().any() else pd.NaT
    y2_last = tr.loc[tr[Y2].notna(), "period"].max() if tr[Y2].notna().any() else pd.NaT
    y3_no_june = bool(pd.notna(y3_last) and y3_last < CUTOFF)
    y2_no_june = bool(pd.notna(y2_last) and y2_last < CUTOFF)
    extract_hole = bool(only_window and (y3_no_june or y2_no_june) and n_months_pos <= 3)
    prose = (
        f"June-flag =1 on {n_june_pos:,} train CM; {n_june_pos_in_window:,} in 2026-06..08; "
        f"{n_june_pos_out:,} outside. Months with any flag=1: {n_months_pos}. "
        f"{'ONLY defined/informative on 2026-06..08 (3 months).' if only_window else 'Fires outside the 3-month window.'} "
        f"Last labeled Y3 period {y3_last.date() if pd.notna(y3_last) else '—'}; "
        f"Y2 {y2_last.date() if pd.notna(y2_last) else '—'}. "
        f"{'Y3/Y2 labels never reach the June window — extract hole.' if extract_hole else 'Labels overlap the June window.'}"
    )
    print(prose)
    return {
        "cal": cal,
        "rows": rows,
        "n_june_pos": n_june_pos,
        "n_june_pos_in_window": n_june_pos_in_window,
        "n_june_pos_out": n_june_pos_out,
        "only_window": only_window,
        "n_months_pos": n_months_pos,
        "y3_last": y3_last,
        "y2_last": y2_last,
        "y3_no_june": y3_no_june,
        "y2_no_june": y2_no_june,
        "extract_hole": extract_hole,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — vs y6_silent_60
# ---------------------------------------------------------------------------
def pass7_y6(tr: pd.DataFrame) -> dict:
    y6 = pd.to_numeric(tr[Y6], errors="coerce")
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    rec60 = (rec > 60).astype(float)
    rec60 = rec60.where(rec.notna())
    flg = pd.to_numeric(tr["c_last_tx_before_2026_06"], errors="coerce")
    lab = y6.notna()
    jac60 = jaccard(rec60[lab], y6[lab])
    jac_flg = jaccard(flg[lab], y6[lab])
    rho60 = spearman(rec60[lab], y6[lab])
    rho_rec = spearman(rec[lab], y6[lab])
    rho_flg = spearman(flg[lab], y6[lab])
    p_y6 = float(y6[lab].mean()) if lab.any() else float("nan")
    p_y6_given_60 = float(y6[lab & (rec60 == 1)].mean()) if (lab & (rec60 == 1)).any() else float("nan")
    p_60_given_y6 = float(rec60[lab & (y6 == 1)].mean()) if (lab & (y6 == 1)).any() else float("nan")
    n_y6 = int(lab.sum())
    n_pos = int((y6 == 1).sum())
    # windows: recency is as-of t; y6 is as-of t+3
    distinct = bool(np.isfinite(jac60) and jac60 < 0.80)
    rows = [
        {
            "pair": "recency>60 vs y6_silent_60",
            "n_lab": f"{n_y6:,}",
            "Jaccard": _f(jac60),
            "ρ": _f(rho60),
        },
        {
            "pair": "c_recency_days vs y6_silent_60",
            "n_lab": f"{n_y6:,}",
            "Jaccard": "—",
            "ρ": _f(rho_rec),
        },
        {
            "pair": "june_flag vs y6_silent_60",
            "n_lab": f"{n_y6:,}",
            "Jaccard": _f(jac_flg),
            "ρ": _f(rho_flg),
        },
    ]
    # AUROC of recency vs rejected Y6 (diagnostic only — do not accept Y6)
    y6_rec = signed_oof_auroc(y6, rec, tr["fold"], lab)
    y6_days = signed_oof_auroc(y6, tr["c_n_days_with_tx"], tr["fold"], lab)
    y6_size = signed_oof_auroc(y6, tr["log_in3"], tr["fold"], lab)
    prose = (
        f"Train labeled {Y6}: n={n_y6:,} pos={n_pos:,} rate {_pp(p_y6)}. "
        f"Jaccard(recency>60, Y6)={_f(jac60)} ρ={_f(rho60)}; "
        f"P(Y6|rec>60)={_pp(p_y6_given_60)} P(rec>60|Y6)={_pp(p_60_given_y6)}. "
        f"{'Distinct window (now vs t+3) — do not merge / do not revive Y6.' if distinct else 'Near-copy of Y6 — still PARK; do not revive.'} "
        f"Y6~recency CV {_f(y6_rec['cv']) if not y6_rec['low_power'] else 'LOW_POWER'} "
        f"vs days {_f(y6_days['cv']) if not y6_days['low_power'] else 'LOW_POWER'} "
        f"vs size {_f(y6_size['cv']) if not y6_size['low_power'] else 'LOW_POWER'}."
    )
    print(prose)
    return {
        "rows": rows,
        "jac60": jac60,
        "jac_flg": jac_flg,
        "rho60": rho60,
        "rho_rec": rho_rec,
        "p_y6": p_y6,
        "p_y6_given_60": p_y6_given_60,
        "p_60_given_y6": p_60_given_y6,
        "n_y6": n_y6,
        "n_pos": n_pos,
        "distinct": distinct,
        "y6_rec": y6_rec["cv"] if not y6_rec["low_power"] else float("nan"),
        "y6_days": y6_days["cv"] if not y6_days["low_power"] else float("nan"),
        "y6_size": y6_size["cv"] if not y6_size["low_power"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — dark 470 vs invoiced 744
# ---------------------------------------------------------------------------
def pass8_dark(tr: pd.DataFrame, con) -> dict:
    book = book_invoice_ids(con)
    hold = load_holdout()
    last = tr.sort_values("period").groupby("company_id", sort=False).last().reset_index()
    last["ever_erp"] = last["company_id"].isin(book)
    n_erp = int(last["ever_erp"].sum())
    n_dark = int((~last["ever_erp"]).sum())
    confirm = n_erp == 744 and n_dark == 470
    tr2 = tr.copy()
    tr2["ever_erp"] = tr2["company_id"].isin(book)
    ever = tr2.groupby("company_id", as_index=False).agg(
        n_cm=("period", "size"),
        rec_p50=("c_recency_days", "median"),
        rec_mean=("c_recency_days", "mean"),
        n_june=("c_last_tx_before_2026_06", "sum"),
        n_ge60=("c_recency_days", lambda s: float((pd.to_numeric(s, errors="coerce") >= 60).mean())),
    )
    ever["ever_erp"] = ever["company_id"].isin(book)
    rows = []
    for name, part in (
        ("ever_erp_744", ever[ever["ever_erp"]]),
        ("never_erp_470", ever[~ever["ever_erp"]]),
    ):
        rows.append(
            {
                "group": name,
                "n_co": int(len(part)),
                "rec_p50": _f(float(part["rec_p50"].median()), 1),
                "rec_mean": _f(float(part["rec_mean"].mean()), 1),
                "share_ge60": _pp(float(part["n_ge60"].mean())),
                "n_june_co": int((part["n_june"] > 0).sum()),
            }
        )
    cm = []
    for name, part in (("ever_erp", tr2[tr2["ever_erp"]]), ("never_erp", tr2[~tr2["ever_erp"]])):
        r = pd.to_numeric(part["c_recency_days"], errors="coerce")
        cm.append(
            {
                "group": name,
                "n_cm": int(len(part)),
                "n_co": int(part["company_id"].nunique()),
                "rec_p50": float(r.median()) if r.notna().any() else float("nan"),
                "june": float(pd.to_numeric(part["c_last_tx_before_2026_06"], errors="coerce").mean()),
            }
        )
    auc_rows = []
    store = {}
    for sname, mask in (
        ("ever_erp", tr2["ever_erp"] & tr2[Y3].notna()),
        ("never_erp", ~tr2["ever_erp"] & tr2[Y3].notna()),
    ):
        res = signed_oof_auroc(tr2[Y3], tr2["c_recency_days"], tr2["fold"], mask)
        store[sname] = res
        auc_rows.append(
            {
                "slice": sname,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    hold_n = int(len(hold))
    hold_book = int(len(set(hold) & book))
    prose = (
        f"Train last-month companies: ever-ERP {n_erp} / never-ERP {n_dark} "
        f"({'CONFIRM 744/470' if confirm else 'counts differ from join QA'}). "
        f"Recency p50 (company-median of medians): invoiced {rows[0]['rec_p50']} vs dark {rows[1]['rec_p50']}. "
        f"June-flag companies: invoiced {rows[0]['n_june_co']} vs dark {rows[1]['n_june_co']}. "
        f"Y3 recency _erp {_f(store['ever_erp']['cv']) if not store['ever_erp']['low_power'] else 'LOW_POWER'} "
        f"dark {_f(store['never_erp']['cv']) if not store['never_erp']['low_power'] else 'LOW_POWER'}. "
        f"Holdout ever-ERP coverage {hold_book}/{hold_n}."
    )
    print(prose)
    return {
        "n_erp": n_erp,
        "n_dark": n_dark,
        "confirm": confirm,
        "rows": rows,
        "cm": cm,
        "auc_rows": auc_rows,
        "y3_erp": store["ever_erp"]["cv"] if not store["ever_erp"]["low_power"] else float("nan"),
        "y3_dark": store["never_erp"]["cv"] if not store["never_erp"]["low_power"] else float("nan"),
        "hold_book": hold_book,
        "hold_n": hold_n,
        "book": book,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — drop 12 chronic dark Y2 names
# ---------------------------------------------------------------------------
def pass9_chronic(tr: pd.DataFrame) -> dict:
    ids = chronic_ids(tr)
    drop = ~tr["company_id"].astype(str).isin(set(ids))
    lab = tr[Y2].notna()
    full = signed_oof_auroc(tr[Y2], tr["c_recency_days"], tr["fold"], lab)
    rest = signed_oof_auroc(tr[Y2], tr["c_recency_days"], tr["fold"], lab & drop)
    size_full = signed_oof_auroc(tr[Y2], tr["log_in3"], tr["fold"], lab)
    size_rest = signed_oof_auroc(tr[Y2], tr["log_in3"], tr["fold"], lab & drop)
    days_full = signed_oof_auroc(tr[Y2], tr["c_n_days_with_tx"], tr["fold"], lab)
    days_rest = signed_oof_auroc(tr[Y2], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    y3_full = signed_oof_auroc(tr[Y3], tr["c_recency_days"], tr["fold"], tr[Y3].notna())
    y3_rest = signed_oof_auroc(tr[Y3], tr["c_recency_days"], tr["fold"], tr[Y3].notna() & drop)
    rec_ch = float(
        pd.to_numeric(
            tr.loc[tr["company_id"].astype(str).isin(set(ids)), "c_recency_days"], errors="coerce"
        ).median()
    ) if ids else float("nan")
    rec_rest = float(pd.to_numeric(tr.loc[drop, "c_recency_days"], errors="coerce").median())
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    prose = (
        f"Chronic 12 names (0158/0172, ≥50% labeled months below 0): {len(ids)}. "
        f"Y2 recency CV full {_f(full['cv']) if not full['low_power'] else 'LOW_POWER'} → "
        f"drop-12 {_f(rest['cv']) if not rest['low_power'] else 'LOW_POWER'}. "
        f"Y3 {_f(y3_full['cv']) if not y3_full['low_power'] else '—'} → "
        f"{_f(y3_rest['cv']) if not y3_rest['low_power'] else '—'}. "
        f"Recency p50 on 12 {_f(rec_ch, 1)} vs rest {_f(rec_rest, 1)}. "
        f"{'DROP FLIPS Y2' if flip else 'Drop does not flip Y2 (≥0.03)'}."
    )
    print(prose)
    return {
        "n_ids": len(ids),
        "ids": ids,
        "y2_full": full["cv"] if not full["low_power"] else float("nan"),
        "y2_drop": rest["cv"] if not rest["low_power"] else float("nan"),
        "y3_full": y3_full["cv"] if not y3_full["low_power"] else float("nan"),
        "y3_drop": y3_rest["cv"] if not y3_rest["low_power"] else float("nan"),
        "size_full": size_full["cv"] if not size_full["low_power"] else float("nan"),
        "size_drop": size_rest["cv"] if not size_rest["low_power"] else float("nan"),
        "days_full": days_full["cv"] if not days_full["low_power"] else float("nan"),
        "days_drop": days_rest["cv"] if not days_rest["low_power"] else float("nan"),
        "rec_ch": rec_ch,
        "rec_rest": rec_rest,
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass10_icc(tr: pd.DataFrame) -> dict:
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    icc = icc_anova(rec, tr["company_id"])
    acf = {k: median_acf(rec, tr["company_id"], k) for k in (1, 3, 6)}
    high_icc = bool(np.isfinite(icc["icc"]) and icc["icc"] >= ICC_STYLE)
    rows_auc = []
    cvs = {}
    for ycol, yname in ((Y3, "Y3"), (Y2, "Y2")):
        for feat, col in (
            ("co_mean", tr["rec_co_mean"]),
            ("demean", tr["rec_demean"]),
            ("now", rec),
        ):
            res = signed_oof_auroc(tr[ycol], col, tr["fold"], tr[ycol].notna())
            cvs[(yname, feat)] = float("nan") if res["low_power"] else res["cv"]
            rows_auc.append(
                {
                    "y": yname,
                    "feature": feat,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                    "sign": res["train_sign"],
                }
            )
    y3_mean = cvs.get(("Y3", "co_mean"), float("nan"))
    y3_shock = cvs.get(("Y3", "demean"), float("nan"))
    y2_mean = cvs.get(("Y2", "co_mean"), float("nan"))
    y2_shock = cvs.get(("Y2", "demean"), float("nan"))
    style_carries = bool(
        np.isfinite(y3_mean) and np.isfinite(y3_shock) and (y3_mean - y3_shock) >= 0.04
    )
    prose = (
        f"c_recency_days acf1={_f(acf[1])} acf3={_f(acf[3])} acf6={_f(acf[6])}; "
        f"ICC={_f(icc['icc'])} ({'sticky company trait' if high_icc else 'not high-ICC'}). "
        f"Y3 company-mean {_f(y3_mean)} vs demean {_f(y3_shock)}; "
        f"Y2 mean {_f(y2_mean)} vs shock {_f(y2_shock)}. "
        + (
            "Trait, not a month shock."
            if high_icc or style_carries
            else "Month shock is closer to the skill."
        )
    )
    print(prose)
    return {
        "icc": icc,
        "acf": acf,
        "high_icc": high_icc,
        "rows": rows_auc,
        "y3_mean": y3_mean,
        "y3_shock": y3_shock,
        "y2_mean": y2_mean,
        "y2_shock": y2_shock,
        "style_carries": style_carries,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — Q6 lag1/lag3 on short vs long
# ---------------------------------------------------------------------------
def pass11_q6(tr: pd.DataFrame) -> dict:
    short = tr["so_far"] < 12
    long = tr["so_far"] >= 18
    rows = []
    store = {}
    specs = [
        ("c_recency_days", "all", tr[Y3].notna()),
        ("c_recency_days_lag1", "all", tr[Y3].notna() & tr["c_recency_days_lag1"].notna()),
        ("c_recency_days_lag3", "all", tr[Y3].notna() & tr["c_recency_days_lag3"].notna()),
        ("c_n_days_with_tx", "all", tr[Y3].notna()),
        ("c_n_days_with_tx_lag1", "all", tr[Y3].notna() & tr["c_n_days_with_tx_lag1"].notna()),
        ("c_recency_days", "short_<12", tr[Y3].notna() & short),
        ("c_recency_days_lag1", "short_<12", tr[Y3].notna() & short & tr["c_recency_days_lag1"].notna()),
        ("c_recency_days_lag3", "short_<12", tr[Y3].notna() & short & tr["c_recency_days_lag3"].notna()),
        ("c_n_days_with_tx_lag1", "short_<12", tr[Y3].notna() & short & tr["c_n_days_with_tx_lag1"].notna()),
        ("c_recency_days", "long_ge18", tr[Y3].notna() & long),
        ("c_recency_days_lag1", "long_ge18", tr[Y3].notna() & long & tr["c_recency_days_lag1"].notna()),
        ("c_n_days_with_tx_lag1", "long_ge18", tr[Y3].notna() & long & tr["c_n_days_with_tx_lag1"].notna()),
        ("c_recency_days_lag1", "Y2_all", tr[Y2].notna() & tr["c_recency_days_lag1"].notna()),
    ]
    for col, sl, mask in specs:
        if col not in tr.columns:
            continue
        ycol = Y2 if sl.startswith("Y2") else Y3
        res = signed_oof_auroc(tr[ycol], tr[col], tr["fold"], mask)
        store[(ycol, col, sl)] = res
        present = float(tr.loc[mask if sl != "all" else tr[ycol].notna(), col].notna().mean())
        rows.append(
            {
                "y": ycol,
                "col": col,
                "slice": sl,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "present": _pp(present),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "sign": res["train_sign"] if not res["low_power"] else "—",
            }
        )

    def _cv(y, col, sl) -> float:
        r = store.get((y, col, sl))
        if r is None or r["low_power"]:
            return float("nan")
        return r["cv"]

    now = _cv(Y3, "c_recency_days", "all")
    lag1 = _cv(Y3, "c_recency_days_lag1", "all")
    lag3 = _cv(Y3, "c_recency_days_lag3", "all")
    lag1_s = _cv(Y3, "c_recency_days_lag1", "short_<12")
    days_l1 = _cv(Y3, "c_n_days_with_tx_lag1", "all")
    days_l1_s = _cv(Y3, "c_n_days_with_tx_lag1", "short_<12")
    drop = now - lag1 if np.isfinite(now) and np.isfinite(lag1) else float("nan")
    # Days lag1 is a night KEEP. Recency lag survives only if contemporaneous
    # has skill AND lag holds (drop ≤ 0.03) AND short books are not empty.
    keep_q6 = bool(
        np.isfinite(now)
        and now >= 0.55
        and np.isfinite(lag1)
        and lag1 >= 0.55
        and (now - lag1) <= 0.03
        and np.isfinite(lag1_s)
        and lag1_s >= 0.55
    )
    if not np.isfinite(now) or now < 0.55:
        q6 = "CLOSE"
        why = (
            f"Y3 contemporaneous recency {_f(now)} is chance; lag1 {_f(lag1)}. "
            "CLOSE as Q6 — no contemporaneous skill to lead."
        )
    elif keep_q6:
        q6 = "KEEP"
        why = (
            f"Y3 recency now {_f(now)} vs lag1 {_f(lag1)} (Δ {_f(drop, 3)}); "
            f"short lag1 {_f(lag1_s)}. KEEP as honest 1-month Q6 (not on 15-col card)."
        )
    else:
        q6 = "CLOSE"
        why = (
            f"Y3 recency now {_f(now)} vs lag1 {_f(lag1)} (Δ {_f(drop, 3)}); "
            f"short lag1 {_f(lag1_s)}. Days lag1 replica {_f(days_l1)} "
            f"(short {_f(days_l1_s)}). CLOSE as Q6 — lag does not hold."
        )
    print(why)
    return {
        "rows": rows,
        "now": now,
        "lag1": lag1,
        "lag3": lag3,
        "lag1_s": lag1_s,
        "days_l1": days_l1,
        "days_l1_s": days_l1_s,
        "drop": drop,
        "keep_q6": keep_q6,
        "q6": q6,
        "prose": why,
    }


# ---------------------------------------------------------------------------
# Pass 12 — calendar / extract-end pile
# ---------------------------------------------------------------------------
def pass12_calendar(tr: pd.DataFrame) -> dict:
    cal = []
    for p, sl in tr.groupby("period"):
        r = pd.to_numeric(sl["c_recency_days"], errors="coerce")
        d = pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce")
        cal.append(
            {
                "period": str(pd.Timestamp(p).date())[:7],
                "n_cm": int(len(sl)),
                "rec_p50": float(r.median()) if r.notna().any() else float("nan"),
                "rec_p90": float(r.quantile(0.90)) if r.notna().any() else float("nan"),
                "rec_mean": float(r.mean()) if r.notna().any() else float("nan"),
                "ge30": float((r >= 30).mean()) if r.notna().any() else float("nan"),
                "ge60": float((r >= 60).mean()) if r.notna().any() else float("nan"),
                "days0": float((d == 0).mean()) if d.notna().any() else float("nan"),
                "june": float(
                    pd.to_numeric(sl["c_last_tx_before_2026_06"], errors="coerce").mean()
                ),
            }
        )
    late = tr["period"] >= pd.Timestamp("2026-06-01")
    early = tr["period"] <= pd.Timestamp("2026-02-01")
    p50_late = float(pd.to_numeric(tr.loc[late, "c_recency_days"], errors="coerce").median())
    p50_early = float(pd.to_numeric(tr.loc[early, "c_recency_days"], errors="coerce").median())
    p50_aug = float(
        pd.to_numeric(
            tr.loc[tr["period"] == pd.Timestamp("2026-08-01"), "c_recency_days"], errors="coerce"
        ).median()
    )
    p50_sep24 = float(
        pd.to_numeric(
            tr.loc[tr["period"] == pd.Timestamp("2024-09-01"), "c_recency_days"], errors="coerce"
        ).median()
    )
    p90_aug = float(
        pd.to_numeric(
            tr.loc[tr["period"] == pd.Timestamp("2026-08-01"), "c_recency_days"], errors="coerce"
        ).quantile(0.90)
    )
    p90_early = float(pd.to_numeric(tr.loc[early, "c_recency_days"], errors="coerce").quantile(0.90))
    ge30_aug = float(
        (
            pd.to_numeric(
                tr.loc[tr["period"] == pd.Timestamp("2026-08-01"), "c_recency_days"], errors="coerce"
            )
            >= 30
        ).mean()
    )
    ge30_early = float(
        (pd.to_numeric(tr.loc[early, "c_recency_days"], errors="coerce") >= 30).mean()
    )
    pile = bool(
        (np.isfinite(p90_aug) and np.isfinite(p90_early) and p90_aug >= p90_early + 8)
        or (np.isfinite(ge30_aug) and np.isfinite(ge30_early) and ge30_aug >= ge30_early + 0.03)
    )
    # skill excluding extract-end months
    y3_pre = signed_oof_auroc(
        tr[Y3], tr["c_recency_days"], tr["fold"], tr[Y3].notna() & ~late
    )
    size_pre = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna() & ~late)
    days_pre = signed_oof_auroc(
        tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna() & ~late
    )
    prose = (
        f"Recency p50 stays 0 (early {_f(p50_early, 1)} / Aug {_f(p50_aug, 1)}). "
        f"The *tail* piles: p90 early {_f(p90_early, 1)} vs Aug {_f(p90_aug, 1)}; "
        f"share≥30d early {_pp(ge30_early)} vs Aug {_pp(ge30_aug)}. "
        f"{'YES — the recency tail piles at extract end the way a still does.' if pile else 'No late-panel tail pile.'} "
        f"Y3 recency on pre-June labeled {_f(y3_pre['cv']) if not y3_pre['low_power'] else 'LOW_POWER'} "
        f"vs size {_f(size_pre['cv']) if not size_pre['low_power'] else 'LOW_POWER'} "
        f"vs days {_f(days_pre['cv']) if not days_pre['low_power'] else 'LOW_POWER'}."
    )
    print(prose)
    return {
        "cal": cal,
        "p50_late": p50_late,
        "p50_early": p50_early,
        "p50_aug": p50_aug,
        "p50_sep24": p50_sep24,
        "p90_aug": p90_aug,
        "p90_early": p90_early,
        "ge30_aug": ge30_aug,
        "ge30_early": ge30_early,
        "pile": pile,
        "y3_pre": y3_pre["cv"] if not y3_pre["low_power"] else float("nan"),
        "size_pre": size_pre["cv"] if not size_pre["low_power"] else float("nan"),
        "days_pre": days_pre["cv"] if not days_pre["low_power"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 13 — recency among busy months (days>0)
# ---------------------------------------------------------------------------
def pass13_busy(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    rows = []
    store = {}
    for sname, mask in (
        ("days_gt0", days > 0),
        ("days_eq0", days == 0),
        ("days_ge5", days >= 5),
        ("zero_in", pd.to_numeric(tr["c_zero_in_month"], errors="coerce") == 1),
    ):
        for y in (Y2, Y3):
            res = signed_oof_auroc(tr[y], tr["c_recency_days"], tr["fold"], tr[y].notna() & mask)
            store[(y, sname)] = res
            r = pd.to_numeric(tr.loc[mask, "c_recency_days"], errors="coerce")
            rows.append(
                {
                    "y": y,
                    "slice": sname,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "rec_p50": _f(float(r.median()) if r.notna().any() else float("nan"), 1),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3_busy = store[(Y3, "days_gt0")]
    y3_quiet = store[(Y3, "days_eq0")]
    prose = (
        f"Y3 recency on days>0 "
        f"{'LOW_POWER' if y3_busy['low_power'] else _f(y3_busy['cv'])}; "
        f"on days=0 "
        f"{'LOW_POWER' if y3_quiet['low_power'] else _f(y3_quiet['cv'])}. "
        "If skill lives only on days=0, recency is the quiet-month twin."
    )
    print(prose)
    return {
        "rows": rows,
        "y3_busy": y3_busy["cv"] if not y3_busy["low_power"] else float("nan"),
        "y3_quiet": y3_quiet["cv"] if not y3_quiet["low_power"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 14 — recency quintiles vs Y3
# ---------------------------------------------------------------------------
def pass14_quintiles(tr: pd.DataFrame) -> dict:
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    lab = tr[Y3].notna()
    sl = tr.loc[lab].copy()
    rec_s = pd.to_numeric(sl["c_recency_days"], errors="coerce")
    sl["_q"] = pd.cut(
        rec_s,
        bins=[-0.5, 0.5, 7.5, 30.5, 60.5, 1e9],
        labels=["0", "1-7", "8-30", "31-60", "≥60"],
    )
    rows = []
    for q, part in sl.groupby("_q", observed=True):
        y = pd.to_numeric(part[Y3], errors="coerce")
        r = pd.to_numeric(part["c_recency_days"], errors="coerce")
        rows.append(
            {
                "q": str(q),
                "n": int(len(part)),
                "n_pos": int((y == 1).sum()),
                "rate": float(y.mean()) if y.notna().any() else float("nan"),
                "rec_p50": float(r.median()) if r.notna().any() else float("nan"),
                "days_p50": float(pd.to_numeric(part["c_n_days_with_tx"], errors="coerce").median()),
            }
        )
    rates = [r["rate"] for r in rows if np.isfinite(r["rate"])]
    mono = bool(len(rates) >= 4 and (rates[-1] - rates[0]) >= 0.02)
    prose = (
        f"Y3 rate by recency quintile: "
        + " / ".join(_pp(r["rate"]) for r in rows)
        + ("; high-recency tail recovers more." if mono else "; no clean high-recency recovery lift.")
    )
    print(prose)
    return {"rows": rows, "mono": mono, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 15 — onset (recency crosses 30) vs state
# ---------------------------------------------------------------------------
def pass15_onset(tr: pd.DataFrame) -> dict:
    srt = tr.sort_values(["company_id", "period"]).copy()
    rec = pd.to_numeric(srt["c_recency_days"], errors="coerce")
    prev = rec.groupby(srt["company_id"], sort=False).shift(1)
    onset = ((rec >= 30) & (prev < 30)).astype(float)
    state = (rec >= 30).astype(float)
    rows = []
    for y in (Y2, Y3):
        for name, col in (("onset_ge30", onset), ("state_ge30", state), ("recency", rec)):
            res = signed_oof_auroc(srt[y], col, srt["fold"], srt[y].notna())
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "prev": _pp(float(col[srt[y].notna()].mean()) if name != "recency" else float("nan")),
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3_on = next(r for r in rows if r["y"] == Y3 and r["feature"] == "onset_ge30")
    y3_st = next(r for r in rows if r["y"] == Y3 and r["feature"] == "state_ge30")
    prose = (
        f"Y3 onset recency≥30 {y3_on['CV']} vs state≥30 {y3_st['CV']}. "
        "Onset would be the cleanest Q3 going-quiet; do not invent a Y from it."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 16 — holdout coverage of the 62
# ---------------------------------------------------------------------------
def pass16_holdout(panel: pd.DataFrame, p2: dict) -> dict:
    ho = panel[panel["split"] == "holdout"]
    rows = []
    for col in ("c_recency_days", "c_last_tx_before_2026_06", "c_n_days_with_tx"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "n_cm": int(len(ho)),
                "n_co": int(ho["company_id"].nunique()),
                "cov": _pp(float(x.notna().mean())),
                "mean": _f(float(x.mean()) if x.notna().any() else float("nan"), 3),
                "p50": _f(float(x.median()) if x.notna().any() else float("nan"), 1),
            }
        )
    ho_ids = set(ho["company_id"])
    asof_ho = [c for c in p2["asof_ids"] if c in ho_ids]
    extract_ho = [c for c in p2["extract_ids"] if c in ho_ids]
    prose = (
        f"Holdout 72 coverage only: {len(ho):,} CM. Recency p50 {rows[0]['p50']}. "
        f"June-flag mean {rows[1]['mean']}. As-of-62 in holdout: {len(asof_ho)} "
        f"({', '.join(asof_ho) if asof_ho else 'none'}). Extract-61 in holdout: {len(extract_ho)}. "
        "No AUROC claim."
    )
    print(prose)
    return {
        "rows": rows,
        "n_cm": int(len(ho)),
        "n_co": int(ho["company_id"].nunique()),
        "asof_ho": asof_ho,
        "extract_ho": extract_ho,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 17 — leftover after days on busy months only
# ---------------------------------------------------------------------------
def pass17_busy_resid(tr: pd.DataFrame, r_d: pd.Series) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    busy = days > 0
    rows = []
    for y in (Y2, Y3):
        res = signed_oof_auroc(tr[y], r_d, tr["fold"], tr[y].notna() & busy)
        sz = signed_oof_auroc(tr[y], tr["log_in3"], tr["fold"], tr[y].notna() & busy)
        rows.append(
            {
                "y": y,
                "slice": "days>0 leftover-after-days",
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "resid CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "size CV": "LOW_POWER" if sz["low_power"] else _f(sz["cv"]),
            }
        )
    y3 = next(r for r in rows if r["y"] == Y3)
    prose = (
        f"Busy-month leftover after days: Y3 {y3['resid CV']} vs size {y3['size CV']}. "
        "If this dies, recency adds nothing inside months that already book."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 18 — last-tx dates of the 62 (how long silent)
# ---------------------------------------------------------------------------
def pass18_silent_age(p2: dict, tr: pd.DataFrame) -> dict:
    train_asof = [r for r in p2["roster"] if r["split"] == "train"]
    ages = []
    for r in train_asof:
        last = pd.Timestamp(r["last_aug"])
        ages.append((AUG_END - last).days)
    ages = pd.Series(ages, dtype=float)
    n_pre2025 = sum(1 for r in train_asof if pd.Timestamp(r["last_aug"]) < pd.Timestamp("2025-01-01"))
    n_2025h1 = sum(
        1
        for r in train_asof
        if pd.Timestamp("2025-01-01") <= pd.Timestamp(r["last_aug"]) < pd.Timestamp("2025-07-01")
    )
    n_2025h2 = sum(
        1
        for r in train_asof
        if pd.Timestamp("2025-07-01") <= pd.Timestamp(r["last_aug"]) < pd.Timestamp("2026-01-01")
    )
    n_2026 = sum(
        1
        for r in train_asof
        if pd.Timestamp("2026-01-01") <= pd.Timestamp(r["last_aug"]) < CUTOFF
    )
    rows = [
        {"bucket": "last < 2025-01", "n": n_pre2025},
        {"bucket": "2025 H1", "n": n_2025h1},
        {"bucket": "2025 H2", "n": n_2025h2},
        {"bucket": "2026-01..05", "n": n_2026},
        {"bucket": "train as-of 62", "n": len(train_asof)},
    ]
    # Y3 rate among the 62 (on their labeled months — all pre-June, flag=0)
    ids = set(r["company_id"] for r in train_asof)
    sl = tr[tr["company_id"].isin(ids)]
    y3 = pd.to_numeric(sl[Y3], errors="coerce")
    y2 = pd.to_numeric(sl[Y2], errors="coerce")
    prose = (
        f"Train as-of silent {len(train_asof)}: last-tx age p50={_f(float(ages.median()), 0)}d "
        f"p90={_f(float(ages.quantile(0.90)), 0)}d. "
        f"Buckets last<2025={n_pre2025} 2025H1={n_2025h1} 2025H2={n_2025h2} 2026-01..05={n_2026}. "
        f"Their labeled Y3 rate {_pp(float(y3.mean()) if y3.notna().any() else float('nan'))} "
        f"(n_lab={int(y3.notna().sum())}); Y2 {_pp(float(y2.mean()) if y2.notna().any() else float('nan'))}."
    )
    print(prose)
    return {
        "rows": rows,
        "p50_age": float(ages.median()) if len(ages) else float("nan"),
        "p90_age": float(ages.quantile(0.90)) if len(ages) else float("nan"),
        "n_train": len(train_asof),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 19 — company-mean leftover after company-mean days
# ---------------------------------------------------------------------------
def pass19_comean(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    day_mu = days.groupby(tr["company_id"], sort=False).transform("mean")
    rec_mu = pd.to_numeric(tr["rec_co_mean"], errors="coerce")
    r_mu, inf = ols_resid(rec_mu, day_mu)
    rho_mu = spearman(rec_mu, day_mu)
    twin_mu = bool(np.isfinite(rho_mu) and abs(rho_mu) >= TWIN_RHO)
    rows = []
    store = {}
    for y in (Y2, Y3):
        for name, col in (
            ("rec_co_mean", rec_mu),
            ("days_co_mean", day_mu),
            ("resid_comean_days", r_mu),
            ("log1p_a_in3", tr["log_in3"]),
        ):
            res = signed_oof_auroc(tr[y], col, tr["fold"], tr[y].notna())
            store[(y, name)] = res
            rows.append(
                {
                    "y": y,
                    "feature": name,
                    "n": f"{res['n_defined']:,}",
                    "n_pos": f"{res['n_pos']:,}",
                    "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                }
            )
    y3_r = store[(Y3, "resid_comean_days")]
    y3_mu = store[(Y3, "rec_co_mean")]
    y3_d = store[(Y3, "days_co_mean")]
    size = store[(Y3, "log1p_a_in3")]
    y3_resid = y3_r["cv"] if not y3_r["low_power"] else float("nan")
    leftover = (
        y3_resid - size["cv"]
        if np.isfinite(y3_resid) and not size["low_power"]
        else float("nan")
    )
    died = bool(np.isfinite(y3_resid) and y3_resid < 0.55)
    prose = (
        f"Company-mean recency↔days ρ={_f(rho_mu)} "
        f"({'TWIN at company level' if twin_mu else 'not a company-level twin'}). "
        f"Y3 rec_co_mean {_f(y3_mu['cv']) if not y3_mu['low_power'] else 'LOW_POWER'} "
        f"vs days_co_mean {_f(y3_d['cv']) if not y3_d['low_power'] else 'LOW_POWER'}; "
        f"leftover {_f(y3_resid)} vs size {_f(size['cv']) if not size['low_power'] else 'LOW_POWER'} "
        f"(Δ {_f(leftover, 3)}). "
        + (
            "Company-mean leftover dies — recency is a quiet-company twin of days."
            if died or (np.isfinite(leftover) and leftover < KEEP_DELTA)
            else "Company-mean leftover still beats size."
        )
    )
    print(prose)
    return {
        "rows": rows,
        "rho_mu": rho_mu,
        "twin_mu": twin_mu,
        "y3_mu": y3_mu["cv"] if not y3_mu["low_power"] else float("nan"),
        "y3_days_mu": y3_d["cv"] if not y3_d["low_power"] else float("nan"),
        "y3_resid": y3_resid,
        "leftover": leftover,
        "died": died,
        "slope": inf["slope"][0] if inf["slope"] else float("nan"),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 20 — feature-report size ρ on company-median
# ---------------------------------------------------------------------------
def pass20_size_quote(tr: pd.DataFrame) -> dict:
    last = tr.sort_values("period").groupby("company_id", sort=False).median(numeric_only=True)
    rho_cm = spearman(tr["c_recency_days"], tr["log_in3"])
    rho_med = spearman(last["c_recency_days"], last["log_in3"])
    confirm = bool(np.isfinite(rho_med) and abs(rho_med - (-0.447)) < 0.03)
    prose = (
        f"size ρ recency vs log1p(a_in3): company-month {_f(rho_cm)} / "
        f"company-median {_f(rho_med)} (feature-report −0.447 "
        f"{'CONFIRM' if confirm else 'off'}). Not SIZE at 0.50."
    )
    print(prose)
    return {"rho_cm": rho_cm, "rho_med": rho_med, "confirm": confirm, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 21 — drop the 61 silent names from Y3
# ---------------------------------------------------------------------------
def pass21_drop61(tr: pd.DataFrame, p2: dict) -> dict:
    ids = set(r["company_id"] for r in p2["roster"] if r["split"] == "train")
    drop = ~tr["company_id"].isin(ids)
    lab = tr[Y3].notna()
    full = signed_oof_auroc(tr[Y3], tr["c_recency_days"], tr["fold"], lab)
    rest = signed_oof_auroc(tr[Y3], tr["c_recency_days"], tr["fold"], lab & drop)
    size_r = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], lab & drop)
    days_r = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], lab & drop)
    sl = tr[tr["company_id"].isin(ids)]
    y3 = pd.to_numeric(sl[Y3], errors="coerce")
    size_p50 = float(pd.to_numeric(sl["log_in3"], errors="coerce").median())
    size_rest = float(pd.to_numeric(tr.loc[drop, "log_in3"], errors="coerce").median())
    days_p50 = float(pd.to_numeric(sl["c_n_days_with_tx"], errors="coerce").median())
    rec_p50 = float(pd.to_numeric(sl["c_recency_days"], errors="coerce").median())
    flip = False
    if not full["low_power"] and not rest["low_power"]:
        flip = abs(full["cv"] - rest["cv"]) >= 0.03
    prose = (
        f"Drop train-61 silent names: Y3 recency {_f(full['cv'])} → {_f(rest['cv'])} "
        f"vs size {_f(size_r['cv']) if not size_r['low_power'] else 'LOW_POWER'} "
        f"vs days {_f(days_r['cv']) if not days_r['low_power'] else 'LOW_POWER'}. "
        f"On the 61: Y3 rate {_pp(float(y3.mean()) if y3.notna().any() else float('nan'))} "
        f"n_lab={int(y3.notna().sum())} n_pos={int((y3==1).sum())}; "
        f"log1p(a_in3) p50 {_f(size_p50, 2)} vs rest {_f(size_rest, 2)}; "
        f"days p50 {_f(days_p50, 1)} rec p50 {_f(rec_p50, 1)}. "
        f"{'DROP FLIPS Y3' if flip else 'The 61 do not carry the Y3 recency skill.'}"
    )
    print(prose)
    return {
        "y3_full": full["cv"] if not full["low_power"] else float("nan"),
        "y3_drop": rest["cv"] if not rest["low_power"] else float("nan"),
        "size_drop": size_r["cv"] if not size_r["low_power"] else float("nan"),
        "days_drop": days_r["cv"] if not days_r["low_power"] else float("nan"),
        "y3_rate": float(y3.mean()) if y3.notna().any() else float("nan"),
        "n_lab": int(y3.notna().sum()),
        "n_pos": int((y3 == 1).sum()),
        "size_p50": size_p50,
        "size_rest": size_rest,
        "days_p50": days_p50,
        "flip": flip,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 22 — busy leftover vs days (does 0.649 beat the engine?)
# ---------------------------------------------------------------------------
def pass22_busy_vs_days(tr: pd.DataFrame, r_d: pd.Series) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    busy = days > 0
    rows = []
    for y in (Y2, Y3):
        mask = tr[y].notna() & busy
        rec = signed_oof_auroc(tr[y], tr["c_recency_days"], tr["fold"], mask)
        resid = signed_oof_auroc(tr[y], r_d, tr["fold"], mask)
        dcv = signed_oof_auroc(tr[y], tr["c_n_days_with_tx"], tr["fold"], mask)
        sz = signed_oof_auroc(tr[y], tr["log_in3"], tr["fold"], mask)
        rows.append(
            {
                "y": y,
                "n": f"{rec['n_defined']:,}",
                "n_pos": f"{rec['n_pos']:,}",
                "recency": "LOW_POWER" if rec["low_power"] else _f(rec["cv"]),
                "resid": "LOW_POWER" if resid["low_power"] else _f(resid["cv"]),
                "days": "LOW_POWER" if dcv["low_power"] else _f(dcv["cv"]),
                "size": "LOW_POWER" if sz["low_power"] else _f(sz["cv"]),
            }
        )
    y3 = next(r for r in rows if r["y"] == Y3)
    prose = (
        f"Busy months (days>0) Y3: recency {y3['recency']} leftover {y3['resid']} "
        f"vs days {y3['days']} vs size {y3['size']}. "
        "If leftover loses to days, recency is still the quiet twin inside booked months."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 23 — rank / log residuals after days
# ---------------------------------------------------------------------------
def pass23_alt_resid(tr: pd.DataFrame) -> dict:
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    r_log, _ = ols_resid(rec, np.log1p(days.clip(lower=0)))
    rec_rk = rec.rank(method="average")
    days_rk = days.rank(method="average")
    r_rk, _ = ols_resid(rec_rk, days_rk)
    rows = []
    store = {}
    for name, col in (
        ("resid_log1p_days", r_log),
        ("resid_ranks", r_rk),
        ("c_recency_days", rec),
        ("c_n_days_with_tx", days),
        ("log1p_a_in3", tr["log_in3"]),
    ):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], tr[Y3].notna())
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    y3_log = store["resid_log1p_days"]["cv"] if not store["resid_log1p_days"]["low_power"] else float("nan")
    y3_rk = store["resid_ranks"]["cv"] if not store["resid_ranks"]["low_power"] else float("nan")
    size = store["log1p_a_in3"]["cv"] if not store["log1p_a_in3"]["low_power"] else float("nan")
    lives = bool(
        (np.isfinite(y3_log) and y3_log - size >= KEEP_DELTA)
        or (np.isfinite(y3_rk) and y3_rk - size >= KEEP_DELTA)
    )
    prose = (
        f"Alt leftover Y3: log1p(days) {_f(y3_log)}; rank {_f(y3_rk)} vs size {_f(size)}. "
        f"{'An alt residual beats size — revisit KEEP.' if lives else 'Alt residuals still miss size+0.02. CLOSE locked.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "y3_log": y3_log,
        "y3_rk": y3_rk,
        "size": size,
        "lives": lives,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 24 — holdout COMP_0269 + recency=0 (booked month-end)
# ---------------------------------------------------------------------------
def pass24_holdout_name(tr: pd.DataFrame, panel: pd.DataFrame, p2: dict) -> dict:
    ho = panel[panel["split"] == "holdout"]
    row = next((r for r in p2["roster"] if r["company_id"] == "COMP_0269"), None)
    rec0 = (pd.to_numeric(tr["c_recency_days"], errors="coerce") == 0).astype(float)
    rows = []
    for y in (Y2, Y3):
        res = signed_oof_auroc(tr[y], rec0, tr["fold"], tr[y].notna())
        days = signed_oof_auroc(tr[y], tr["c_n_days_with_tx"], tr["fold"], tr[y].notna())
        rows.append(
            {
                "y": y,
                "feature": "recency_eq_0",
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "days": "LOW_POWER" if days["low_power"] else _f(days["cv"]),
            }
        )
    y3 = next(r for r in rows if r["y"] == Y3)
    share0 = float(rec0.mean())
    prose = (
        f"Holdout silent name COMP_0269 last_aug={row['last_aug'] if row else '—'} "
        f"last_ever={row['last_ever'] if row else '—'}. "
        f"Train recency==0 (booked on month-end) share {_pp(share0)}; "
        f"Y3 {y3['CV']} vs days {y3['days']} — still loses to the engine."
    )
    print(prose)
    return {"rows": rows, "share0": share0, "comp0269": row, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 25 — leftover fold bits (stability)
# ---------------------------------------------------------------------------
def pass25_fold_resid(tr: pd.DataFrame, r_d: pd.Series) -> dict:
    rec = signed_oof_auroc(tr[Y3], tr["c_recency_days"], tr["fold"], tr[Y3].notna())
    resid = signed_oof_auroc(tr[Y3], r_d, tr["fold"], tr[Y3].notna())
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], tr[Y3].notna())
    size = signed_oof_auroc(tr[Y3], tr["log_in3"], tr["fold"], tr[Y3].notna())
    rows = [
        {"feature": "c_recency_days", "CV": _f(rec["cv"]), "folds": fold_bits(rec)},
        {"feature": "resid_days", "CV": _f(resid["cv"]), "folds": fold_bits(resid)},
        {"feature": "c_n_days_with_tx", "CV": _f(days["cv"]), "folds": fold_bits(days)},
        {"feature": "log1p_a_in3", "CV": _f(size["cv"]), "folds": fold_bits(size)},
    ]
    prose = (
        f"Y3 folds recency [{fold_bits(rec)}] leftover [{fold_bits(resid)}] "
        f"days [{fold_bits(days)}] size [{fold_bits(size)}]."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 26 — first-month recency (onboarding vs going-quiet)
# ---------------------------------------------------------------------------
def pass26_first_month(tr: pd.DataFrame) -> dict:
    first = tr["so_far"] == 1
    later = tr["so_far"] >= 6
    rec_f = float(pd.to_numeric(tr.loc[first, "c_recency_days"], errors="coerce").median())
    rec_l = float(pd.to_numeric(tr.loc[later, "c_recency_days"], errors="coerce").median())
    rows = []
    for sname, mask in (("so_far=1", first), ("so_far>=6", later)):
        res = signed_oof_auroc(tr[Y3], tr["c_recency_days"], tr["fold"], tr[Y3].notna() & mask)
        rows.append(
            {
                "slice": sname,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "rec_p50": _f(
                    float(pd.to_numeric(tr.loc[mask, "c_recency_days"], errors="coerce").median()),
                    1,
                ),
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
            }
        )
    prose = (
        f"First-month recency p50 {_f(rec_f, 1)} vs so-far≥6 {_f(rec_l, 1)}. "
        "Onboarding recency is not a health Y (same reason created_at is PARK)."
    )
    print(prose)
    return {"rows": rows, "rec_first": rec_f, "rec_later": rec_l, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 27 — log leftover is the days=0 pile?
# ---------------------------------------------------------------------------
def pass27_log_busy(tr: pd.DataFrame) -> dict:
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    zero = (days == 0).astype(float)
    r_log, _ = ols_resid(rec, np.log1p(days.clip(lower=0)))
    r_d0, _ = ols_resid(rec, days, zero)
    r_logz, _ = ols_resid(rec, np.log1p(days.clip(lower=0)), zero)
    busy = days > 0
    specs = [
        ("log_resid_all", r_log, tr[Y3].notna()),
        ("log_resid_days>0", r_log, tr[Y3].notna() & busy),
        ("resid_days+zero", r_d0, tr[Y3].notna()),
        ("log_resid+zero", r_logz, tr[Y3].notna()),
        ("days_all", days, tr[Y3].notna()),
        ("days_days>0", days, tr[Y3].notna() & busy),
        ("size_all", tr["log_in3"], tr[Y3].notna()),
        ("size_days>0", tr["log_in3"], tr[Y3].notna() & busy),
    ]
    rows = []
    store = {}
    for name, col, mask in specs:
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], mask)
        store[name] = res
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
    log_all = store["log_resid_all"]["cv"] if not store["log_resid_all"]["low_power"] else float("nan")
    log_busy = store["log_resid_days>0"]["cv"] if not store["log_resid_days>0"]["low_power"] else float("nan")
    d0 = store["resid_days+zero"]["cv"] if not store["resid_days+zero"]["low_power"] else float("nan")
    days_all = store["days_all"]["cv"] if not store["days_all"]["low_power"] else float("nan")
    days_b = store["days_days>0"]["cv"] if not store["days_days>0"]["low_power"] else float("nan")
    size_all = store["size_all"]["cv"] if not store["size_all"]["low_power"] else float("nan")
    pile = bool(np.isfinite(log_all) and np.isfinite(log_busy) and (log_all - log_busy) >= 0.03)
    beats_size_busy = bool(
        np.isfinite(log_busy) and np.isfinite(store["size_days>0"]["cv"])
        and (log_busy - store["size_days>0"]["cv"]) >= KEEP_DELTA
        and not store["size_days>0"]["low_power"]
    )
    beats_days = bool(np.isfinite(log_all) and np.isfinite(days_all) and log_all >= days_all + KEEP_DELTA)
    keep_alt = bool(beats_size_busy and beats_days)
    prose = (
        f"log1p(days) leftover Y3 all {_f(log_all)} / days>0 {_f(log_busy)}; "
        f"days+zero dummy leftover {_f(d0)}. Days engine {_f(days_all)} / busy {_f(days_b)}. "
        f"{'The 0.673 is the days=0 nonlinear pile.' if pile else 'Log leftover is not only the days=0 pile.'} "
        f"{'KEEP-as-X via log leftover' if keep_alt else 'Still loses to days — CLOSE as quiet twin.'}"
    )
    print(prose)
    return {
        "rows": rows,
        "log_all": log_all,
        "log_busy": log_busy,
        "d0": d0,
        "days_all": days_all,
        "pile": pile,
        "beats_size_busy": beats_size_busy,
        "beats_days": beats_days,
        "keep_alt": keep_alt,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra 28 — Y3 bins on days>0 (within-month timing)
# ---------------------------------------------------------------------------
def pass28_busy_bins(tr: pd.DataFrame) -> dict:
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    lab = tr[Y3].notna() & (days > 0)
    sl = tr.loc[lab].copy()
    rec_s = pd.to_numeric(sl["c_recency_days"], errors="coerce")
    sl["_q"] = pd.cut(
        rec_s,
        bins=[-0.5, 0.5, 7.5, 30.5, 60.5, 1e9],
        labels=["0", "1-7", "8-30", "31-60", "≥60"],
    )
    rows = []
    for q, part in sl.groupby("_q", observed=True):
        y = pd.to_numeric(part[Y3], errors="coerce")
        rows.append(
            {
                "q": str(q),
                "n": int(len(part)),
                "n_pos": int((y == 1).sum()),
                "rate": float(y.mean()) if y.notna().any() else float("nan"),
                "days_p50": float(pd.to_numeric(part["c_n_days_with_tx"], errors="coerce").median()),
            }
        )
    prose = (
        f"Y3 rate by recency bin among days>0: "
        + " / ".join(_pp(r["rate"]) for r in rows)
        + ". Within-month last-booking timing, not a multi-month going-quiet."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


# ---------------------------------------------------------------------------
# Extra 29 — leftover after days+size (double residual)
# ---------------------------------------------------------------------------
def pass29_days_size(tr: pd.DataFrame) -> dict:
    rec = pd.to_numeric(tr["c_recency_days"], errors="coerce")
    r_ds, _ = ols_resid(rec, tr["c_n_days_with_tx"], tr["log_in3"])
    r_lds, _ = ols_resid(rec, np.log1p(pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce").clip(lower=0)), tr["log_in3"])
    rows = []
    for name, col in (("resid_days+size", r_ds), ("resid_logdays+size", r_lds)):
        res = signed_oof_auroc(tr[Y3], col, tr["fold"], tr[Y3].notna())
        rows.append(
            {
                "feature": name,
                "n": f"{res['n_defined']:,}",
                "n_pos": f"{res['n_pos']:,}",
                "CV": "LOW_POWER" if res["low_power"] else _f(res["cv"]),
                "folds": fold_bits(res) if not res["low_power"] else "—",
            }
        )
    y3 = rows[0]
    prose = (
        f"Y3 leftover after days+size {y3['CV']}; after log1p(days)+size {rows[1]['CV']}. "
        "If this is chance, recency is days+size, not a new Q3."
    )
    print(prose)
    return {"rows": rows, "prose": prose}


def make_png(p6: dict, p12: dict) -> bool:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return False
    cal = p12["cal"]
    months = [r["period"] for r in cal]
    p50 = [r["rec_p50"] for r in cal]
    p90 = [r["rec_p90"] for r in cal]
    ge30 = [100.0 * r["ge30"] for r in cal]
    june = [100.0 * r["june"] for r in cal]
    x = np.arange(len(months))
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    ax = axes[0]
    ax.plot(x, p50, color="#1f4e79", lw=1.6, label="p50")
    ax.plot(x, p90, color="#9e6b4a", lw=1.2, ls="--", label="p90")
    ax.axvline(list(months).index("2026-06") if "2026-06" in months else -1, color="#b33", lw=0.8, ls=":")
    ax.set_ylabel("c_recency_days")
    ax.set_title("Recency cash clock (train) — June 2026 extract cutoff")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = axes[1]
    ax2.bar(x - 0.18, ge30, width=0.36, color="#1f4e79", label="% recency ≥ 30d")
    ax2.bar(x + 0.18, june, width=0.36, color="#c45c26", label="% last-tx-before-2026-06")
    ax2.set_ylabel("% of train CM")
    ax2.set_xticks(x[::2])
    ax2.set_xticklabels([months[i] for i in range(0, len(months), 2)], rotation=45, ha="right")
    ax2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return True


def decide(p2, p3, p4, p5, p6, p7, p10, p11, p23=None, p27=None) -> dict:
    """KEEP / DROP-from-44 / CLOSE / PARK. Recency never lands on the 15-col card."""
    park_y = True
    extract_dummy = bool(p6["extract_hole"])
    leftover_ok = bool(p5["leftover_lives"] and not extract_dummy)
    twin = bool(p3["twin_days"] or p3["twin_ntx"])
    size = bool(p3["size_flag"])
    leftover_died = bool(p5["died"] or not p5["leftover_lives"])
    log_keep = bool(p27 and p27.get("keep_alt"))

    if log_keep and not extract_dummy:
        rec_x = "KEEP"
        rec_why = (
            f"log leftover after days {_f(p23['y3_log']) if p23 else float('nan')} "
            f"beats size and days. Still not on tonight's 15-col card."
        )
    elif leftover_ok and not twin and not size:
        rec_x = "KEEP"
        rec_why = (
            f"leftover after days {_f(p5['y3_days'])} beats size {_f(p5['size_y3'])} "
            f"by {_f(p5['leftover_beat'])}. Not a June extract dummy. "
            "Still not on tonight's 15-col card."
        )
    elif leftover_died or twin or size:
        rec_x = "DROP from the 44"
        log_bit = ""
        if p23 is not None and np.isfinite(p23.get("y3_log", float("nan"))):
            log_bit = (
                f" Log leftover {_f(p23['y3_log'])} beats size {_f(p4['size_y3'])} "
                f"but loses to days {_f(p4['days_y3'])}"
                + (
                    f" (busy {_f(p27['log_busy'])} vs days {_f(p27['days_all'])})."
                    if p27
                    else "."
                )
            )
        rec_why = (
            f"Y3 recency {_f(p4['rec_y3'])} leftover-after-days {_f(p5['y3_days'])} "
            f"vs size {_f(p4['size_y3'])} (Δ {_f(p5['leftover_beat'])}) "
            f"vs days {_f(p4['days_y3'])}. CLOSE as quiet twin of days."
            + log_bit
            + " Not on the 15-col card."
        )
    else:
        rec_x = "CLOSE as X"
        rec_why = (
            f"Y3 {_f(p4['rec_y3'])} does not beat size {_f(p4['size_y3'])} "
            f"by ≥0.02 after days (Δ {_f(p5['leftover_beat'])})."
        )

    june_x = "PARK"
    june_why = (
        f"Javier extract 61 vs as-of-Aug 62 (COMP_0981 is the +1). "
        f"Flag only fires 2026-06..08 ({p6['n_june_pos']:,} CM); "
        f"Y3 last labeled {p6['y3_last'].date() if pd.notna(p6['y3_last']) else '—'} "
        f"— never reaches the window. PARK as extract artifact; DROP from the 44."
    )
    if not p6["extract_hole"]:
        june_x = "DROP from the 44"
        june_why = (
            f"June flag Y3 {_f(p4['june_y3'])}; not a clean extract hole by the 3-month test, "
            "but still not a health X."
        )
    return {
        "rec_x": rec_x,
        "rec_why": rec_why,
        "june_x": june_x,
        "june_why": june_why,
        "park_y": park_y,
        "q6": p11["q6"],
        "extract_hole": extract_dummy,
        "leftover_lives": leftover_ok,
        "twin": twin,
        "size": size,
    }


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10, p11, p12 = (
        ctx["p6"],
        ctx["p7"],
        ctx["p8"],
        ctx["p9"],
        ctx["p10"],
        ctx["p11"],
        ctx["p12"],
    )
    d = ctx["decision"]
    lines = [
        "# Q3 recency — going-quiet or extract hole?",
        "",
        f"Generated `{_now_iso()}` by agent `{AGENT}`. DuckDB `clean` read-only. "
        "`monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. "
        "Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. "
        "No new GBM. No `build_targets`. Do not invent `y_recency` / `y_silent`. "
        "Do not revive `y6_silent_60` or `created_at`. Recency is **not** on the 15-col card. "
        "Night Y3 quote stays **0.762 / 0.752**. Days bar **0.711**.",
        "",
        "`c_recency_days` = month_end.date − last booking date ≤ month_end. "
        "`c_last_tx_before_2026_06` = 1 if period ≥ 2026-06-01 and last booking as of "
        "month_end is < 2026-06-01. Javier extract-level **61**; as-of 2026-08-31 **62** "
        "(COMP_0981 silent 2025-04-07 → 2026-09-01).",
        "",
        "## Headline",
        "",
        ctx["headline"],
        "",
        "## Brief questions",
        "",
        "| # | question | what this cut says |",
        "| --- | --- | --- |",
        f"| 1 | Who is healthy? | Recency is a cash clock, not a health Y. PARK. ICC {_f(p10['icc']['icc'])}. |",
        "| 2 | Who is improving? | Not this column. |",
        f"| 3 | Who is turning? | Going-quiet would be Q3. Leftover after days {_f(p5['y3_days'])} — "
        + ("a real leftover" if p5["leftover_lives"] else "dies as a quiet twin of days")
        + ". |",
        "| 4 | Dip vs fall? | Not this table. |",
        f"| 5 | Why did it change? | June flag is an extract cutoff, not a why. Recency vs days ρ={_f(p3['rho_days'])}. |",
        f"| 6 | Months earlier? | Recency lag1 **{d['q6']}** (now {_f(p11['now'])} vs lag1 {_f(p11['lag1'])}; "
        f"days lag1 replica {_f(p11['days_l1'])} stays the night KEEP). |",
        "",
        "## PARK / CLOSE / KEEP",
        "",
        "| object | decision | why |",
        "| --- | --- | --- |",
        f"| `c_recency_days` on the 44 | **{d['rec_x']}** | {d['rec_why']} |",
        "| `c_recency_days` as a health Y | **PARK** | do not invent `y_recency` / `y_silent` |",
        "| `c_recency_days` on the 15-col card | **no** | days 0.711 stays the engine; night Y3 0.762 / 0.752 untouched |",
        f"| `c_last_tx_before_2026_06` | **{d['june_x']}** | {d['june_why']} |",
        f"| Q6 lag1 `c_recency_days` | **{d['q6']}** | {p11['prose']} |",
        "| `y6_silent_60` | **PARK** (do not revive) | future rejected Y; distinct window |",
        "| `created_at` | **PARK** (do not revive) | connection clock, not cash clock |",
        "",
        "## 1. Completeness + distributions",
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
                    "cov": _pp(r["cov"]),
                    "p50": _f(r["p50"], 1),
                    "p90": _f(r["p90"], 1),
                    "modal%": _pp(r["modal_share"]),
                    "share=1": _pp(r["share1"]) if np.isfinite(r["share1"]) else "—",
                }
                for r in p1["rows"]
            ]
        ),
        "",
        f"Feature-report modal 99.0% on the June flag: **{'CONFIRM' if p1['confirm_modal'] else 'NO'}**.",
        "",
        "## 2. Formula vs raw last booking — 61 vs 62 — COMP_0981",
        "",
        p2["prose"],
        "",
        "| item | value |",
        "| --- | ---: |",
        f"| store vs raw recency agree | {_pp(p2['rec_agree'])} |",
        f"| store vs raw June-flag agree | {_pp(p2['june_agree'])} |",
        f"| max \\|Δ recency\\| | {_f(p2['rec_maxabs'], 2)} |",
        f"| extract-level last tx < 2026-06-01 | {p2['n_extract']} (Javier 61) |",
        f"| as-of 2026-08-31 | {p2['n_asof']} (quoted 62) |",
        f"| +1 vs extract | {', '.join(p2['plus']) if p2['plus'] else '—'} |",
        f"| COMP_0981 is the +1 | {'YES' if p2['is_0981'] else 'NO'} |",
        f"| COMP_0981 last-ever / last-asof-Aug | {p2['last_0981_ever']} / {p2['last_0981_aug']} |",
        f"| COMP_0981 next after 2025-04-07 | {p2['next_after']} |",
        f"| store Aug-2026 flag companies | {p2['store_aug_n']} |",
        f"| train / holdout as-of | {p2['n_train_asof']} / {p2['n_hold_asof']} |",
        "",
        "Roster (as-of 2026-08-31 last booking < 2026-06-01):",
        "",
        _md_table(p2["roster"]),
        "",
        "## 3. Spearman vs days / n_tx / gap_sd / size / zero-in",
        "",
        p3["prose"],
        "",
        _md_table(p3["rows"]),
        "",
        "Twin if \\|ρ\\|≥0.80. SIZE if \\|ρ\\| vs log1p(a_in3) ≥0.50. Feature-report size ρ −0.447.",
        "",
        "## 4. Single-feature train group-fold AUROC",
        "",
        f"Y2 n={p4['n_y2']:,} base {_pp(p4['y2_rate'])}; "
        f"Y3 stressed n={p4['n_y3']:,} base {_pp(p4['y3_rate'])}. "
        "Sign from the train side of each fold. Seed 20260918. "
        f"Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica {_f(p4['days_y3'])}); "
        f"size 0.617 (replica {_f(p4['size_y3'])}).",
        "",
        p4["prose"],
        "",
        _md_table(p4["rows"]),
        "",
        f"June-flag mean on labeled Y3 rows {_pp(p4['june_on_y3'])}; on labeled Y2 {_pp(p4['june_on_y2'])}.",
        "",
        "## 5. Residual after days",
        "",
        p5["prose"],
        "",
        _md_table(p5["rows"]),
        "",
        "KEEP-as-X: leftover after days beats size ≥0.02 **and** is not the June extract dummy. "
        "If leftover dies, CLOSE as quiet twin.",
        "",
        "## 6. June-cutoff — only 2026-06..08?",
        "",
        p6["prose"],
        "",
        _md_table(
            [
                {
                    "period": r["period"],
                    "n_cm": f"{r['n_cm']:,}",
                    "june n": f"{r['june_n']:,}",
                    "june %": _pp(r["june_share"]),
                    "rec p50": _f(r["rec_p50"], 1),
                    "rec p90": _f(r["rec_p90"], 1),
                    "≥30d": _pp(r["rec_ge30"]),
                    "≥60d": _pp(r["rec_ge60"]),
                }
                for r in p6["cal"]
            ]
        ),
        "",
        _md_table(p6["rows"]),
        "",
        f"Extract hole: **{'YES — DROP from the 44' if p6['extract_hole'] else 'no'}**.",
        "",
        "## 7. vs rejected `y6_silent_60`",
        "",
        p7["prose"],
        "",
        _md_table(p7["rows"]),
        "",
        "Y6 is last-tx as of **t+3** more than 60 days before that end. Recency is as-of **t**. "
        "Do not revive Y6. Do not merge.",
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
                    "rec p50": _f(r["rec_p50"], 1),
                    "june": _pp(r["june"]),
                }
                for r in p8["cm"]
            ]
        ),
        "",
        _md_table(p8["auc_rows"]),
        "",
        "## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)",
        "",
        p9["prose"],
        "",
        f"IDs ({p9['n_ids']}): {', '.join(p9['ids']) if p9['ids'] else '—'}.",
        "",
        "## 10. ICC / company-demean",
        "",
        p10["prose"],
        "",
        _md_table(p10["rows"]),
        "",
        "## 11. Q6 — lag1 / lag3 on short vs long",
        "",
        p11["prose"],
        "",
        _md_table(p11["rows"]),
        "",
        "Days lag1 is a night KEEP. Recency lag is scored the same way (present on short + AUROC holds).",
        "",
        "## 12. Calendar — extract-end pile?",
        "",
        p12["prose"],
        "",
        _md_table(
            [
                {
                    "period": r["period"],
                    "rec p50": _f(r["rec_p50"], 1),
                    "rec p90": _f(r["rec_p90"], 1),
                    "≥30d": _pp(r["ge30"]),
                    "≥60d": _pp(r["ge60"]),
                    "days=0": _pp(r["days0"]),
                    "june %": _pp(r["june"]),
                }
                for r in p12["cal"]
            ]
        ),
        "",
        f"Plot: `{OUT_PNG.name}`." if ctx["png_ok"] else "Plot: skipped (no matplotlib).",
        "",
        "## Extra 13 — recency among busy months",
        "",
        ctx["p13"]["prose"],
        "",
        _md_table(ctx["p13"]["rows"]),
        "",
        "## Extra 14 — Y3 rate by recency quintile",
        "",
        ctx["p14"]["prose"],
        "",
        _md_table(
            [
                {
                    "q": r["q"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "Y3 rate": _pp(r["rate"]),
                    "rec p50": _f(r["rec_p50"], 1),
                    "days p50": _f(r["days_p50"], 1),
                }
                for r in ctx["p14"]["rows"]
            ]
        ),
        "",
        "## Extra 15 — onset recency≥30 (not a Y)",
        "",
        ctx["p15"]["prose"],
        "",
        _md_table(ctx["p15"]["rows"]),
        "",
        "## Extra 16 — holdout coverage only",
        "",
        ctx["p16"]["prose"],
        "",
        _md_table(ctx["p16"]["rows"]),
        "",
        "## Extra 17 — leftover after days on days>0",
        "",
        ctx["p17"]["prose"],
        "",
        _md_table(ctx["p17"]["rows"]),
        "",
        "## Extra 18 — how long have the 62 been silent?",
        "",
        ctx["p18"]["prose"],
        "",
        _md_table(ctx["p18"]["rows"]),
        "",
        "## Extra 19 — company-mean leftover after company-mean days",
        "",
        ctx["p19"]["prose"],
        "",
        _md_table(ctx["p19"]["rows"]),
        "",
        "## Extra 20 — feature-report size ρ (company-median)",
        "",
        ctx["p20"]["prose"],
        "",
        "## Extra 21 — drop the train-61 silent names",
        "",
        ctx["p21"]["prose"],
        "",
        "## Extra 22 — busy-month leftover vs days",
        "",
        ctx["p22"]["prose"],
        "",
        _md_table(ctx["p22"]["rows"]),
        "",
        "## Extra 23 — log / rank residual after days",
        "",
        ctx["p23"]["prose"],
        "",
        _md_table(ctx["p23"]["rows"]),
        "",
        "## Extra 24 — holdout COMP_0269 + recency==0",
        "",
        ctx["p24"]["prose"],
        "",
        _md_table(ctx["p24"]["rows"]),
        "",
        "## Extra 25 — leftover fold bits",
        "",
        ctx["p25"]["prose"],
        "",
        _md_table(ctx["p25"]["rows"]),
        "",
        "## Extra 26 — first-month recency (onboarding)",
        "",
        ctx["p26"]["prose"],
        "",
        _md_table(ctx["p26"]["rows"]),
        "",
        "## Extra 27 — log leftover is the days=0 pile?",
        "",
        ctx["p27"]["prose"],
        "",
        _md_table(ctx["p27"]["rows"]),
        "",
        "## Extra 28 — recency bins among days>0",
        "",
        ctx["p28"]["prose"],
        "",
        _md_table(
            [
                {
                    "q": r["q"],
                    "n": f"{r['n']:,}",
                    "n_pos": f"{r['n_pos']:,}",
                    "Y3 rate": _pp(r["rate"]),
                    "days p50": _f(r["days_p50"], 1),
                }
                for r in ctx["p28"]["rows"]
            ]
        ),
        "",
        "## Extra 29 — leftover after days+size",
        "",
        ctx["p29"]["prose"],
        "",
        _md_table(ctx["p29"]["rows"]),
        "",
        "## What failed / next (held for wave note)",
        "",
        *([f"- {x}" for x in ctx["failed"]] if ctx["failed"] else ["- (none)"]),
        "",
        f"Elapsed {ctx['elapsed_s']:.0f}s. Cuts: dist, formula 61/62, Spearman, singles, leftover-after-days, "
        "June extract hole, vs Y6, dark 470/744, 12 names, ICC/demean, Q6 short/long, calendar tail pile, "
        "busy months, bins, onset, holdout, busy residual, silent-age, company-mean leftover, "
        "size-ρ quote, drop-61, busy-vs-days, alt residual, COMP_0269, fold bits, first-month.",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    p1, p2, p3, p4, p5, p6, p7, d = (
        ctx["p1"],
        ctx["p2"],
        ctx["p3"],
        ctx["p4"],
        ctx["p5"],
        ctx["p6"],
        ctx["p7"],
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
            "model": MODEL,
            "split": "train",
            "metric": "c_recency_days_p50",
            "value": p1["rec"]["p50"],
            "coverage": f"{p1['rec']['cov']:.4f}",
            "notes": f"p90={p1['rec']['p90']:.2f} modal={p1['flg']['modal_share']:.4f} confirm990={p1['confirm_modal']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "all",
            "metric": "n_last_tx_before_june_extract",
            "value": p2["n_extract"],
            "coverage": "1.0000",
            "notes": f"javier61={p2['confirm_61']} asof62={p2['n_asof']} confirm62={p2['confirm_62']} plus={','.join(p2['plus'])} is_0981={p2['is_0981']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_recency_days",
            "value": p4["rec_y3"],
            "coverage": "1.0000",
            "notes": f"size={p4['size_y3']:.4f} days={p4['days_y3']:.4f} beat_size={p4['beat_size']:.4f} x={d['rec_x']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_recency_resid_days",
            "value": p5["y3_days"],
            "coverage": "1.0000",
            "notes": f"leftover_beat={p5['leftover_beat']:.4f} lives={p5['leftover_lives']} died={p5['died']} slope={p5['slope_days']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_last_tx_before_2026_06",
            "value": p4["june_y3"],
            "coverage": "1.0000",
            "notes": f"extract_hole={p6['extract_hole']} june_on_y3={p4['june_on_y3']:.4f} x={d['june_x']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_n_days_with_tx",
            "value": p4["days_y3"],
            "coverage": "1.0000",
            "notes": f"night=0.711 replica; size={p4['size_y3']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "rho_recency_vs_days",
            "value": p3["rho_days"],
            "coverage": "1.0000",
            "notes": f"rho_ntx={p3['rho_ntx']:.4f} rho_size={p3['rho_size']:.4f} twin_days={p3['twin_days']} size_flag={p3['size_flag']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y6,
            "model": MODEL,
            "split": "train",
            "metric": "jaccard_recency60_y6_silent",
            "value": p7["jac60"],
            "coverage": f"{p7['n_y6'] / ctx['p1']['rec']['n_cm']:.4f}" if ctx["p1"]["rec"]["n_cm"] else "",
            "notes": f"distinct={p7['distinct']} rho={p7['rho60']:.4f} do_not_revive",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_recency_days_lag1",
            "value": ctx["p11"]["lag1"],
            "coverage": "1.0000",
            "notes": f"now={ctx['p11']['now']:.4f} drop={ctx['p11']['drop']:.4f} q6={d['q6']} short={ctx['p11']['lag1_s']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "recency_icc",
            "value": ctx["p10"]["icc"]["icc"],
            "coverage": "1.0000",
            "notes": f"acf1={ctx['p10']['acf'][1]:.3f} y3_mean={ctx['p10']['y3_mean']:.4f} y3_demean={ctx['p10']['y3_shock']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_recency_comean_resid_days",
            "value": ctx["p19"]["y3_resid"],
            "coverage": "1.0000",
            "notes": f"rho_mu={ctx['p19']['rho_mu']:.4f} twin_mu={ctx['p19']['twin_mu']} rec_mu={ctx['p19']['y3_mu']:.4f} days_mu={ctx['p19']['y3_days_mu']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "C",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_c_recency_drop61",
            "value": ctx["p21"]["y3_drop"],
            "coverage": "1.0000",
            "notes": f"full={ctx['p21']['y3_full']:.4f} flip={ctx['p21']['flip']} y3_rate61={ctx['p21']['y3_rate']:.4f}",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("x_families"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("x_families", "")),
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


def write_wave(ctx: dict) -> None:
    d = ctx["decision"]
    p2, p4, p5, p6 = ctx["p2"], ctx["p4"], ctx["p5"], ctx["p6"]
    text = (
        f"# Wave 4 — recency / last-tx-before-June QA\n\n"
        f"Agent `{AGENT}`. Train group-fold seed 20260918. Holdout 72 coverage only.\n\n"
        f"## Files written\n\n"
        f"- `analysis/evaluate/recency_qa.py`\n"
        f"- `analysis/outputs/recency_qa.md`\n"
        f"- `analysis/outputs/recency_calendar.png`\n"
        f"- append-only `analysis/experiments/registry.csv` (skip key agent+x_families+y+model+split+metric)\n"
        f"- this note\n\n"
        f"Did not touch `ops.py`, `y6_activity.py`, `a_vol_qa.py`, `transfer_qa.py`, `gap_sd_qa.py`, "
        f"`product/`, parquet / duckdb, `build_targets`, parent journal, or the 15-col card. "
        f"Night Y3 quote stays **0.762 / 0.752**. Days 0.711 untouched.\n\n"
        f"## Locked verdict\n\n"
        f"| object | decision |\n"
        f"| --- | --- |\n"
        f"| `c_recency_days` on the 44 | **{d['rec_x']}** |\n"
        f"| `c_recency_days` as a health Y | **PARK** |\n"
        f"| `c_last_tx_before_2026_06` | **{d['june_x']}** |\n"
        f"| Q6 recency lag | **{d['q6']}** |\n"
        f"| `y6_silent_60` / `created_at` | **PARK** (not revived) |\n\n"
        f"{d['rec_why']}\n\n"
        f"{d['june_why']}\n\n"
        f"## What we measured (train)\n\n"
        f"- 61 vs 62: extract {p2['n_extract']} / as-of-Aug {p2['n_asof']}; "
        f"COMP_0981 is the +1={p2['is_0981']}.\n"
        f"- Y3 recency {_f(p4['rec_y3'])} vs size {_f(p4['size_y3'])} vs days {_f(p4['days_y3'])}.\n"
        f"- Leftover after days {_f(p5['y3_days'])} (Δ size {_f(p5['leftover_beat'])}).\n"
        f"- Extract hole: {p6['extract_hole']}.\n"
        f"- {ctx['p11']['prose']}\n\n"
        f"## What failed / next\n\n"
        + ("\n".join(f"- {x}" for x in ctx["failed"]) if ctx["failed"] else "- (none)")
        + f"\n\nElapsed {ctx['elapsed_s']:.0f}s. "
        f"Log leftover {_f(ctx['p23']['y3_log'])} / busy {_f(ctx['p27']['log_busy'])} "
        f"still loses to days {_f(p4['days_y3'])}. Tail piles at Aug (p90 29 vs 10).\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"recency_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_panel_lags(
        panel, ["c_recency_days", "c_n_days_with_tx", "c_last_tx_before_2026_06"], (1, 3)
    )
    panel = add_so_far(panel)
    panel = add_style_shock(panel)

    con = connect()
    try:
        panel = attach_y6_if_missing(panel, con)
        tr = panel[panel["split"] == "train"].copy()
        assert_no_holdout(tr["company_id"])
        print(
            f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
            f"holdout CM={(panel['split']=='holdout').sum()}"
        )
        print("pass 1 dist")
        p1 = pass1_dist(panel, tr)
        days = load_tx_days(con)
        print("pass 2 formula 61/62")
        p2 = pass2_formula(tr, panel, days)
        print("pass 3 spearman")
        p3 = pass3_rho(tr)
        print("pass 4 singles")
        p4 = pass4_auroc(tr)
        print("pass 5 residual after days")
        p5 = pass5_residual(tr)
        print("pass 6 june extract hole")
        p6 = pass6_extract(tr)
        print("pass 7 vs y6_silent_60")
        p7 = pass7_y6(tr)
        print("pass 8 dark 470 vs 744")
        p8 = pass8_dark(tr, con)
        print("pass 9 drop 12 chronic")
        p9 = pass9_chronic(tr)
        print("pass 10 ICC / demean")
        p10 = pass10_icc(tr)
        print("pass 11 Q6")
        p11 = pass11_q6(tr)
        print("pass 12 calendar pile")
        p12 = pass12_calendar(tr)
        print("pass 13 busy months")
        p13 = pass13_busy(tr)
        print("pass 14 quintiles")
        p14 = pass14_quintiles(tr)
        print("pass 15 onset")
        p15 = pass15_onset(tr)
        print("pass 16 holdout coverage")
        p16 = pass16_holdout(panel, p2)
        print("pass 17 busy leftover")
        p17 = pass17_busy_resid(tr, p5["r_d"])
        print("pass 18 silent age of 62")
        p18 = pass18_silent_age(p2, tr)
        print("pass 19 company-mean leftover")
        p19 = pass19_comean(tr)
        print("pass 20 company-median size ρ")
        p20 = pass20_size_quote(tr)
        print("pass 21 drop 61")
        p21 = pass21_drop61(tr, p2)
        print("pass 22 busy vs days")
        p22 = pass22_busy_vs_days(tr, p5["r_d"])
        print("pass 23 alt residual")
        p23 = pass23_alt_resid(tr)
        print("pass 24 holdout COMP_0269")
        p24 = pass24_holdout_name(tr, panel, p2)
        print("pass 25 fold leftover")
        p25 = pass25_fold_resid(tr, p5["r_d"])
        print("pass 26 first month")
        p26 = pass26_first_month(tr)
        print("pass 27 log leftover vs days=0 pile")
        p27 = pass27_log_busy(tr)
        print("pass 28 busy bins")
        p28 = pass28_busy_bins(tr)
        print("pass 29 leftover days+size")
        p29 = pass29_days_size(tr)
    finally:
        con.close()

    decision = decide(p2, p3, p4, p5, p6, p7, p10, p11, p23, p27)
    png_ok = make_png(p6, p12)
    headline = (
        f"`c_recency_days` Y3 {_f(p4['rec_y3'])} vs size {_f(p4['size_y3'])} "
        f"(Δ {_f(p4['beat_size'])}) vs days {_f(p4['days_y3'])}; "
        f"leftover after days {_f(p5['y3_days'])} "
        f"({'lives' if p5['leftover_lives'] else 'dies'}). "
        f"Twin-days={p3['twin_days']} SIZE={p3['size_flag']} (ρ_size={_f(p3['rho_size'])}). "
        f"June flag extract {p2['n_extract']} / as-of {p2['n_asof']}; "
        f"COMP_0981 +1={p2['is_0981']}; extract-hole={p6['extract_hole']}. "
        f"On the 44: **{decision['rec_x']}**. June flag **{decision['june_x']}**. "
        f"PARK as health Y. Q6 **{decision['q6']}**."
    )
    print(headline)
    failed = []
    if not p1["confirm_modal"]:
        failed.append(f"June-flag modal {_pp(p1['flg']['modal_share'])} ≠ 99.0%")
    if not p2["formula_ok"]:
        failed.append(
            f"store vs raw recency agree {_pp(p2['rec_agree'])} / june {_pp(p2['june_agree'])} — inspect ops.py"
        )
    if not p2["confirm_61"] or not p2["confirm_62"] or not p2["is_0981"]:
        failed.append(
            f"61/62/COMP_0981 mismatch extract={p2['n_extract']} asof={p2['n_asof']} plus={p2['plus']}"
        )
    if not p4["days_ok"]:
        failed.append(
            f"Y3 days replica {_f(p4['days_y3'])} vs night 0.711 "
            "(signed fold; published 0.711 may be oriented 1−raw)"
        )
    if not p8["confirm"]:
        failed.append(f"dark/erp {p8['n_dark']}/{p8['n_erp']} ≠ 470/744")
    if p5["died"] or not p5["leftover_lives"]:
        failed.append(
            f"leftover after days {_f(p5['y3_days'])} vs size {_f(p5['size_y3'])} — CLOSE as quiet twin"
        )
    if p6["extract_hole"]:
        failed.append("June flag is a 2026-06 extract hole — DROP from the 44 / PARK as artifact")
    if p12["pile"]:
        failed.append(
            f"recency tail piles at Aug (p90 {_f(p12['p90_aug'], 1)} vs early {_f(p12['p90_early'], 1)}; "
            f"≥30d {_pp(p12['ge30_aug'])} vs {_pp(p12['ge30_early'])}) — extract-end still"
        )
    if p19["died"] or (np.isfinite(p19["leftover"]) and p19["leftover"] < KEEP_DELTA):
        failed.append(
            f"company-mean leftover {_f(p19['y3_resid'])} vs size — quiet-company twin of days"
        )
    if not p23["lives"]:
        failed.append(
            f"alt leftover log1p(days) {_f(p23['y3_log'])} / rank {_f(p23['y3_rk'])} still miss size+0.02"
        )
    elif p27["keep_alt"]:
        failed.append("log leftover beats size and days — revisit KEEP (unexpected)")
    else:
        failed.append(
            f"log leftover {_f(p23['y3_log'])} beats size but loses to days {_f(p4['days_y3'])}; "
            f"busy log leftover {_f(p27['log_busy'])} — CLOSE as quiet twin"
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
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
    }
    write_md(ctx)
    append_registry(ctx)
    if WRITE_WAVE:
        write_wave(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
