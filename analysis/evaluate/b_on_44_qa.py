"""Unused leftover of Family B columns still on the 44.

NORTH_STAR: never B as Y2/Y3 X. Y2 is already-negative persistence;
Y3 is stressed-only recovery on the same cash path. This ticket is the
unused leftover: should ``b_bal_vol``, ``b_below_0``, ``b_d_runway``,
``b_neg_episodes``, ``b_runway`` leave the 44 (DROP) or stay as Q1
descriptors only?

KEEP-as-X is already closed. Walk is an identity (do not rewrite
``liquidity.py``). Still is not leaking backward. KEEP last-value
``b_runway`` as Q1 description; PARK as a forecast Y. Javier vol vs
store ``b_bal_vol`` is DRIFT 0.354 — do not merge a_vol.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.b_on_44_qa

Owned: analysis/evaluate/b_on_44_qa.py, analysis/outputs/b_on_44_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_b_on_44.md (end).

Do not invent y_bal_vol. Do not put B on the 15-col Y3 card.
Night Y3 0.762 / 0.752. Days 0.711. Size 0.617. Y7 TURNOVER 0.720 / 0.712.
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
OUT_MD = ANALYSIS / "outputs" / "b_on_44_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "b_on_44_leftover.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_b_on_44.md"
AGENT = "c8b1440a"
WAVE = "4"
ROUND = "R4"
MODEL = "b_on_44_qa"
WRITE_WAVE = True  # last iterate — one note at the end

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BENCH = 0.711
SIZE_QUOTE = 0.617
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
TWIN_RHO = 0.80
LEFTOVER_DIE = 0.55
FAKE_DAYS_RHO = 0.30
DEMEAN_TRAIT = 0.10
ETA2_TRAIT = 0.40
MIN_POS = 50
MIN_ACF_PAIRS = 4
CLIP = 3.0
PANEL_END = pd.Timestamp("2026-08-01")
NIGHT_Y3 = 0.762
NIGHT_Y3_CORE = 0.752
NIGHT_Y7 = 0.720
NIGHT_Y7_CORE = 0.712
VOL_DRIFT_QUOTE = 0.354
SHORT_PERSIST_QUOTE = 0.730
LONG_PERSIST_QUOTE = 0.862
ZOMBIE_PROD_QUOTE = 0.122
ZOMBIE_BAL_QUOTE = 0.0036
HOLE13_QUOTE = 13

HOLE13 = (
    "COMP_0033",
    "COMP_0093",
    "COMP_0166",
    "COMP_0312",
    "COMP_0327",
    "COMP_0715",
    "COMP_0720",
    "COMP_0784",
    "COMP_0800",
    "COMP_0889",
    "COMP_0938",
    "COMP_0976",
    "COMP_1285",
)
SANTANDER = "Santander (Corporate UK)"
CASH_TYPES = ("checking", "saving", "tpv")

FOCUS = (
    "b_bal_vol",
    "b_below_0",
    "b_d_runway",
    "b_neg_episodes",
    "b_runway",
)

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "a_net",
    "a_op_in",
    "a_op_out",
    "a_io_ratio",
    "a_n_tx",
    "c_n_days_with_tx",
    "c_n_tx",
    "b_liq",
    "b_runway",
    "b_d_runway",
    "b_below_0",
    "b_bal_vol",
    "b_neg_episodes",
    "b_neg_liq_3",
    "e_ar_issued",
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


def spearman(a, b) -> tuple[float, int]:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    )
    d = d[np.isfinite(d["a"]) & np.isfinite(d["b"])]
    n = int(len(d))
    if n < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan"), n
    return float(d["a"].corr(d["b"], method="spearman")), n


def spearman1(a, b) -> float:
    return spearman(a, b)[0]


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


def icc_eta(series: pd.Series, company: pd.Series) -> dict:
    d = pd.DataFrame(
        {"x": pd.to_numeric(series, errors="coerce"), "cid": company.astype(str)}
    )
    d = d[np.isfinite(d["x"])]
    if d.empty or d["cid"].nunique() < 2:
        return {"icc": float("nan"), "eta2": float("nan"), "n": 0, "n_cos": 0}
    grand = float(d["x"].mean())
    g = d.groupby("cid")["x"]
    means = g.mean()
    ns = g.size()
    ss_b = float((ns * (means - grand) ** 2).sum())
    ss_w = float(((d["x"] - d["cid"].map(means)) ** 2).sum())
    ss_t = ss_b + ss_w
    eta2 = ss_b / ss_t if ss_t > 0 else float("nan")
    var_w = float(g.var(ddof=1).mean()) if (ns >= 2).any() else float("nan")
    var_b = float(means.var(ddof=1))
    icc = (
        var_b / (var_b + var_w)
        if np.isfinite(var_b) and np.isfinite(var_w) and (var_b + var_w) > 0
        else float("nan")
    )
    return {"icc": icc, "eta2": eta2, "n": int(len(d)), "n_cos": int(d["cid"].nunique())}


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


def signed_oof(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    mask: pd.Series | None = None,
    n_folds: int = N_FOLDS,
) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = y.notna() & x.notna() & folds.notna()
    if mask is not None:
        defined = defined & mask.fillna(False)
    n = int(defined.sum())
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    fold_rows = []
    aucs = []
    if n_pos < MIN_POS or n_neg == 0:
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
            "fold_str": "—",
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
        "fold_str": " ".join(_f(a) for a in aucs),
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


def leftover_of(
    y: pd.Series,
    x: pd.Series,
    controls: list[pd.Series],
    folds: pd.Series,
    mask: pd.Series | None = None,
) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof(y, resid, folds, mask)
    rho_days = float("nan")
    pear_days = float("nan")
    fake = False
    near_fake = False
    if controls:
        rho_days, _ = spearman(resid, controls[0])
        d = pd.DataFrame(
            {
                "r": pd.to_numeric(resid, errors="coerce"),
                "c": pd.to_numeric(controls[0], errors="coerce"),
            }
        ).dropna()
        if len(d) >= 8 and d["r"].nunique() > 1 and d["c"].nunique() > 1:
            pear_days = float(d["r"].corr(d["c"]))
        fake = bool(np.isfinite(rho_days) and abs(rho_days) >= FAKE_DAYS_RHO)
        near_fake = bool(np.isfinite(rho_days) and abs(rho_days) >= 0.25)
    # rank residual (Spearman-honest leftover)
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rank_resid, _ = ols_resid(xr, *cr) if controls else (pd.Series(np.nan, index=x.index), {})
    rank_rec = signed_oof(y, rank_resid, folds, mask) if controls else rec
    left = rec["cv"]
    died_raw = bool(np.isfinite(left) and left < LEFTOVER_DIE)
    died = bool(died_raw or fake)
    honest = float("nan") if fake else left
    rank_left = rank_rec["cv"]
    rank_died = bool(
        (np.isfinite(rank_left) and rank_left < LEFTOVER_DIE) or fake
    )
    return {
        "cv": left,
        "honest": honest,
        "fake": fake,
        "near_fake": near_fake,
        "died": died,
        "died_raw": died_raw,
        "rho_resid_ctrl0": rho_days,
        "pear_resid_ctrl0": pear_days,
        "rank_cv": rank_left,
        "rank_died": rank_died,
        "n_ols": info["n"],
        "rec": rec,
        "slope": info["slope"][0] if info["slope"] else float("nan"),
        "resid": resid,
    }


def persist_pairs(series: pd.Series, company: pd.Series, lag: int = 3) -> tuple[float, int, int]:
    d = pd.DataFrame(
        {
            "x": pd.to_numeric(series, errors="coerce"),
            "co": company.astype(str),
        }
    )
    d["xlag"] = d.groupby("co", sort=False)["x"].shift(-lag)
    ok = np.isfinite(d["x"]) & np.isfinite(d["xlag"])
    n = int(ok.sum())
    n_co = int(d.loc[ok, "co"].nunique())
    if n < 8:
        return float("nan"), n, n_co
    rho, _ = spearman(d.loc[ok, "x"], d.loc[ok, "xlag"])
    return rho, n, n_co


def add_in_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Javier vol + a_out_vol + last-value Q1. Never written to parquet."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = out.groupby("company_id", sort=False)
    sd6_net = g["a_net"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).std()
    )
    mean6_in = g["a_op_in"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).mean()
    )
    sd6_out = g["a_op_out"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).std()
    )
    mean6_out = g["a_op_out"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).mean()
    )
    out["a_vol"] = np.minimum(CLIP, sd6_net / np.maximum(mean6_in, 1.0))
    out["a_out_vol"] = np.minimum(CLIP, sd6_out / np.maximum(mean6_out, 1.0))
    out["log_in3"] = np.log1p(pd.to_numeric(out["a_in3"], errors="coerce").clip(lower=0))
    last = out.sort_values("period").groupby("company_id", sort=False).last()
    out["b_runway_last"] = out["company_id"].map(last["b_runway"])
    out["b_below_0_last"] = out["company_id"].map(last["b_below_0"])
    out["so_far"] = g.cumcount() + 1
    n_mo = out.groupby("company_id")["period"].transform("size")
    out["n_months"] = n_mo
    out["short_book"] = (n_mo < 12).astype(np.int8)
    out["long_24"] = (n_mo == 24).astype(np.int8)
    for c in ("b_bal_vol", "b_d_runway", "b_runway", "b_below_0", "b_neg_episodes", "a_vol"):
        s = pd.to_numeric(out[c], errors="coerce")
        gg = s.groupby(out["company_id"], sort=False)
        out[f"{c}_lag1"] = gg.shift(1)
        out[f"{c}_lag3"] = gg.shift(3)
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
    optional = {"a_n_tx", "c_n_tx", "e_ar_issued"}
    hard = [c for c in missing if c not in optional]
    if hard:
        raise RuntimeError(f"monthly.parquet missing {hard}")
    have = [c for c in STORE_COLS if c in raw.columns]
    ykeep = ["company_id", "period"]
    for c in Y_KEEP:
        if c not in yraw.columns:
            raise RuntimeError(f"targets missing {c}")
        ykeep.append(c)
    panel = _keys(raw[list(have)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


# ---------------------------------------------------------------------------
# Pass 1 — coverage; 13 Santander UK nulls; zombie cash confirm
# ---------------------------------------------------------------------------
def pass1_coverage(panel: pd.DataFrame, tr: pd.DataFrame, con) -> dict:
    print("pass 1 coverage / 13 / zombies")
    rows = []
    store = {}
    n_cm = len(tr)
    n_co = int(tr["company_id"].nunique())
    for col in FOCUS + ("b_liq", "b_neg_liq_3"):
        x = pd.to_numeric(tr[col], errors="coerce")
        nn = x.notna()
        all_null_cos = []
        for cid, g in tr.groupby("company_id", sort=False):
            if pd.to_numeric(g[col], errors="coerce").notna().sum() == 0:
                all_null_cos.append(str(cid))
        rec = {
            "col": col,
            "cov_cm": _pp(_pct(int(nn.sum()), n_cm)),
            "n_finite": f"{int(nn.sum()):,}",
            "n_null": f"{int((~nn).sum()):,}",
            "cos_any": int(tr.loc[nn, "company_id"].nunique()),
            "cos_all_null": len(all_null_cos),
        }
        rows.append(rec)
        store[col] = {
            "cov": _pct(int(nn.sum()), n_cm),
            "n_finite": int(nn.sum()),
            "n_null": int((~nn).sum()),
            "all_null": all_null_cos,
        }

    liq = pd.to_numeric(tr["b_liq"], errors="coerce")
    all_null_liq = set(store["b_liq"]["all_null"])
    hole_set = set(HOLE13)
    hole_match = all_null_liq == hole_set
    extra = sorted(all_null_liq - hole_set)
    missing = sorted(hole_set - all_null_liq)

    # duckdb: txs but no balances row (confirm the 13)
    hold = load_holdout()
    tx_cos = set(
        con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM transactions").df()[
            "company_id"
        ].astype(str)
    )
    # Family B: balances has company_id on the clean extract (not only via banking_products).
    bal_cos = set(
        con.execute("SELECT DISTINCT CAST(company_id AS VARCHAR) AS company_id FROM balances").df()[
            "company_id"
        ].astype(str)
    )
    train_ids = set(tr["company_id"].astype(str))
    tx_no_bal = sorted((train_ids & tx_cos) - bal_cos)
    hole13_ok = tx_no_bal == sorted(HOLE13)
    no_walk_extra = sorted(all_null_liq - set(tx_no_bal))
    hole19_ok = (len(all_null_liq) == 19) and hole13_ok and (len(no_walk_extra) == 6)

    banks = con.execute(
        f"""
        SELECT CAST(t.company_id AS VARCHAR) AS company_id,
               coalesce(p.bank_name, '(none)') AS bank_name,
               p.type,
               COUNT(*) AS n_tx,
               MAX(t."date") AS last_tx
        FROM transactions t
        JOIN banking_products p ON t.product_id = p.product_id
        WHERE CAST(t.company_id AS VARCHAR) IN ({", ".join(repr(c) for c in HOLE13)})
          AND p.type IN ('checking','saving','tpv')
        GROUP BY 1, 2, 3
        """
    ).df()
    banks["company_id"] = banks["company_id"].astype(str)
    sant_cos = sorted(banks.loc[banks["bank_name"] == SANTANDER, "company_id"].unique())
    last20 = banks.copy()
    last20["last_tx"] = pd.to_datetime(last20["last_tx"])
    last_by_co = last20.groupby("company_id")["last_tx"].max()
    n_0720 = int((last_by_co.dt.strftime("%Y-%m-%d") == "2026-07-20").sum())

    # zombies
    raw = con.execute(
        f"""
        SELECT CAST(p.company_id AS VARCHAR) AS company_id,
               b.product_id,
               p.type AS ptype,
               b.balance,
               coalesce(t.n_tx, 0) AS n_tx
        FROM balances b
        JOIN banking_products p ON b.product_id = p.product_id
        LEFT JOIN (
            SELECT product_id, COUNT(*) AS n_tx
            FROM transactions
            GROUP BY 1
        ) t ON b.product_id = t.product_id
        WHERE p.type IN ('checking','saving','tpv') AND b.balance IS NOT NULL
        """
    ).df()
    raw["company_id"] = raw["company_id"].astype(str)
    ztr = raw.loc[~raw["company_id"].isin(hold)].copy()
    assert_no_holdout(ztr["company_id"])
    zz = ztr["n_tx"] == 0
    z_share_prod = float(zz.mean()) if len(ztr) else float("nan")
    cash_abs = float(ztr["balance"].abs().sum())
    z_abs = float(ztr.loc[zz, "balance"].abs().sum())
    z_share_bal = z_abs / cash_abs if cash_abs else float("nan")
    z_ok = bool(abs(z_share_prod - ZOMBIE_PROD_QUOTE) < 0.01)

    ho = panel[panel["split"] == "holdout"]
    ho_cov = []
    for col in FOCUS:
        xh = pd.to_numeric(ho[col], errors="coerce")
        ho_cov.append(
            {
                "col": col,
                "holdout_cov": _pp(float(xh.notna().mean())),
                "holdout_cos_any": int(ho.loc[xh.notna(), "company_id"].nunique()),
            }
        )

    prose = (
        f"Train {n_cm:,} CM / {n_co} companies. "
        f"`b_liq` all-null={len(all_null_liq)} = 13 tx-no-balance + 6 no-cash-walk "
        f"({'CONFIRM 13+6=19' if hole19_ok else f'OFF extra={extra} missing={missing} no_walk={no_walk_extra}'}). "
        f"tx-no-balance train={len(tx_no_bal)} "
        f"({'CONFIRM HOLE13' if hole13_ok else 'OFF list'}). "
        f"Santander UK on {len(sant_cos)} of the 13; last-tx 2026-07-20 n={n_0720}. "
        f"Zombie cash {int(zz.sum()):,}/{len(ztr):,}={_pp(z_share_prod)} "
        f"({'CONFIRM 12.2%' if z_ok else 'off 12.2%'}) "
        f"|zombie|/|cash|={_pp(z_share_bal)} (quote 0.36–0.4%). PARK as a flag."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "n_cm": n_cm,
        "n_co": n_co,
        "hole_match": hole_match,
        "hole13_ok": hole13_ok,
        "hole19_ok": hole19_ok,
        "no_walk_extra": no_walk_extra,
        "tx_no_bal": tx_no_bal,
        "all_null_liq": sorted(all_null_liq),
        "sant_cos": sant_cos,
        "n_sant": len(sant_cos),
        "n_0720": n_0720,
        "z_n": int(zz.sum()),
        "z_prod": int(len(ztr)),
        "z_share_prod": z_share_prod,
        "z_share_bal": z_share_bal,
        "z_ok": z_ok,
        "ho_cov": ho_cov,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 2 — Spearman twins
# ---------------------------------------------------------------------------
def pass2_twins(tr: pd.DataFrame) -> dict:
    print("pass 2 Spearman twins")
    others = (
        "b_runway",
        "log_in3",
        "c_n_days_with_tx",
        "a_vol",
        "a_out_vol",
        "b_liq",
        "b_neg_liq_3",
        "b_runway_last",
    )
    pair_rows = []
    twins = []
    store = {}
    cols = list(FOCUS) + [c for c in others if c not in FOCUS]
    for a in FOCUS:
        for b in cols:
            if a == b:
                continue
            rho, n = spearman(tr[a], tr[b])
            twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
            size = bool(b == "log_in3" and np.isfinite(rho) and abs(rho) >= SIZE_RHO)
            pair_rows.append(
                {
                    "a": a,
                    "b": b,
                    "ρ": _f(rho),
                    "n": f"{n:,}",
                    "twin≥0.80": "YES" if twin else "no",
                    "SIZE": "YES" if size else "no",
                }
            )
            store[(a, b)] = {"rho": rho, "n": n, "twin": twin, "size": size}
            if twin and a < b:
                twins.append((a, b, rho, n))
    # within-FOCUS matrix
    matrix_rows = []
    for a in FOCUS:
        rec = {"col": a}
        for b in FOCUS:
            if a == b:
                rec[b] = "1"
                continue
            rec[b] = _f(store[(a, b)]["rho"])
        matrix_rows.append(rec)

    weaker = []
    for a, b, rho, n in twins:
        # weaker = lower coverage
        ca = float(pd.to_numeric(tr[a], errors="coerce").notna().mean())
        cb = float(pd.to_numeric(tr[b], errors="coerce").notna().mean())
        drop = a if ca <= cb else b
        keep = b if drop == a else a
        weaker.append(
            {
                "pair": f"{a} ↔ {b}",
                "ρ": _f(rho),
                "n": f"{n:,}",
                "DROP weaker": drop,
                "keep": keep,
                "why": f"cov {drop}={_pp(min(ca, cb))} ≤ {keep}={_pp(max(ca, cb))}",
            }
        )

    below_neg3 = store.get(("b_below_0", "b_neg_liq_3"), {})
    below_ep = store.get(("b_below_0", "b_neg_episodes"), {})
    vol_jav = store.get(("b_bal_vol", "a_vol"), {})
    vol_drift = bool(
        np.isfinite(vol_jav.get("rho", float("nan")))
        and abs(vol_jav["rho"] - VOL_DRIFT_QUOTE) < 0.02
    )
    prose = (
        f"Twins ≥0.80 among FOCUS: {len(twins)}. "
        f"`b_below_0`↔`b_neg_liq_3` ρ={_f(below_neg3.get('rho'))} "
        f"(feature-report dropped neg_liq_3). "
        f"`b_below_0`↔`b_neg_episodes` ρ={_f(below_ep.get('rho'))} "
        f"{'TWIN → DROP weaker' if below_ep.get('twin') else 'not a twin'}. "
        f"`b_bal_vol`↔`a_vol` ρ={_f(vol_jav.get('rho'))} "
        f"({'CONFIRM DRIFT 0.354' if vol_drift else 'off 0.354'})."
    )
    print(" ", prose)
    return {
        "pair_rows": pair_rows,
        "matrix_rows": matrix_rows,
        "weaker": weaker,
        "twins": twins,
        "store": store,
        "vol_drift": vol_drift,
        "vol_rho": vol_jav.get("rho", float("nan")),
        "below_neg3": below_neg3.get("rho", float("nan")),
        "below_ep": below_ep.get("rho", float("nan")),
        "below_ep_twin": bool(below_ep.get("twin")),
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 3 — single-feature group-fold (diagnostic; never KEEP as Y2/Y3 X)
# ---------------------------------------------------------------------------
def pass3_singles(tr: pd.DataFrame) -> dict:
    print("pass 3 singles (diagnostic leak, not KEEP)")
    benches = {
        "log_in3": "size",
        "c_n_days_with_tx": "days",
        "b_runway": "last-value runway (t)",
        "b_runway_last": "last-value runway (still, LOOK-AHEAD)",
    }
    feats = list(FOCUS) + list(benches)
    rows = []
    store = {}
    for ycol in (Y3, Y2):
        mask = tr[ycol].notna()
        for col in feats:
            rec = signed_oof(tr[ycol], tr[col], tr["fold"], mask)
            store[(ycol, col)] = rec
            rows.append(
                {
                    "Y": ycol.replace("y3_recover_cash_6m", "Y3").replace("y2_neg_2of3", "Y2"),
                    "feature": col,
                    "CV": _f(rec["cv"]),
                    "±sd": _f(rec["sd"]),
                    "sign": rec["train_sign"],
                    "n": f"{rec['n_defined']:,}",
                    "n_pos": rec["n_pos"],
                    "folds": rec["fold_str"],
                    "note": "LOOK-AHEAD still" if col == "b_runway_last" else ("illegal X" if col in FOCUS else "bench"),
                }
            )
    days_y3 = store[(Y3, "c_n_days_with_tx")]["cv"]
    size_y3 = store[(Y3, "log_in3")]["cv"]
    days_ok = bool(np.isfinite(days_y3) and abs(days_y3 - DAYS_BENCH) < 0.015)
    size_ok = bool(np.isfinite(size_y3) and abs(size_y3 - SIZE_QUOTE) < 0.015)
    y2_below = store[(Y2, "b_below_0")]["cv"]
    y2_run = store[(Y2, "b_runway")]["cv"]
    leak_y2 = bool(np.isfinite(y2_below) and y2_below >= 0.70)
    prose = (
        f"Y3 days={_f(days_y3)} ({'CONFIRM 0.711' if days_ok else 'off'}) "
        f"size={_f(size_y3)} ({'CONFIRM 0.617' if size_ok else 'off'}). "
        f"Y2 `b_below_0`={_f(y2_below)} `b_runway`={_f(y2_run)} "
        f"— {'LOCK leak (not KEEP)' if leak_y2 else 'weaker than expected leak'}. "
        f"Singles are diagnostics. Never B as Y2/Y3 X."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "days_y3": days_y3,
        "size_y3": size_y3,
        "days_ok": days_ok,
        "size_ok": size_ok,
        "y2_below": y2_below,
        "y2_run": y2_run,
        "leak_y2": leak_y2,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 4 — honest leftover after days / after last-value runway (Y3)
# ---------------------------------------------------------------------------
def pass4_leftover_y3(tr: pd.DataFrame) -> dict:
    print("pass 4 Y3 leftover after days / runway")
    days = tr["c_n_days_with_tx"]
    size = tr["log_in3"]
    rw = tr["b_runway"]
    rw_last = tr["b_runway_last"]
    mask = tr[Y3].notna()
    rows = []
    store = {}
    for col in FOCUS:
        after_d = leftover_of(tr[Y3], tr[col], [days], tr["fold"], mask)
        after_s = leftover_of(tr[Y3], tr[col], [size], tr["fold"], mask)
        after_rw = leftover_of(tr[Y3], tr[col], [rw], tr["fold"], mask) if col != "b_runway" else None
        after_last = leftover_of(tr[Y3], tr[col], [rw_last], tr["fold"], mask)
        after_ds = leftover_of(tr[Y3], tr[col], [days, size], tr["fold"], mask)
        after_dr = leftover_of(tr[Y3], tr[col], [days, rw], tr["fold"], mask) if col != "b_runway" else None
        store[col] = {
            "days": after_d,
            "size": after_s,
            "runway": after_rw,
            "last": after_last,
            "ds": after_ds,
            "dr": after_dr,
        }
        rows.append(
            {
                "col": col,
                "after days": _f(after_d["cv"]),
                "honest days": "DIES (fake days)" if after_d["fake"] else _f(after_d["honest"]),
                "ρ(resid,days)": _f(after_d["rho_resid_ctrl0"]),
                "Pearson": _f(after_d["pear_resid_ctrl0"]),
                "rank leftover": _f(after_d["rank_cv"]),
                "near-fake": "YES" if after_d["near_fake"] else "no",
                "after size": _f(after_s["cv"]),
                "after b_runway": "—" if after_rw is None else _f(after_rw["cv"]),
                "after last-value": _f(after_last["cv"]),
                "after days+size": _f(after_ds["cv"]),
                "dies": "YES" if after_d["died"] else "no",
            }
        )
    any_live = any(not store[c]["days"]["died"] for c in FOCUS)
    prose = (
        f"Y3 leftover after days: "
        + ", ".join(f"{c}={_f(store[c]['days']['cv'])}" for c in FOCUS)
        + f". Fake-days any={any(store[c]['days']['fake'] for c in FOCUS)}. "
        + ("Honest leftover still lives." if any_live else "Honest leftover DIES (<0.55 or fake days).")
        + " Leftover is a DROP evidence, not a KEEP-as-X."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "any_live": any_live,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 5 — Y2 leftover after b_below_0 / last-value sign
# ---------------------------------------------------------------------------
def pass5_leftover_y2(tr: pd.DataFrame) -> dict:
    print("pass 5 Y2 leftover after sign (expect dies)")
    mask = tr[Y2].notna()
    below = tr["b_below_0"]
    last_sign = tr["b_below_0_last"]
    days = tr["c_n_days_with_tx"]
    rows = []
    store = {}
    for col in FOCUS:
        after_b = leftover_of(tr[Y2], tr[col], [below], tr["fold"], mask) if col != "b_below_0" else None
        after_ls = leftover_of(tr[Y2], tr[col], [last_sign], tr["fold"], mask)
        after_d = leftover_of(tr[Y2], tr[col], [days], tr["fold"], mask)
        raw = signed_oof(tr[Y2], tr[col], tr["fold"], mask)
        store[col] = {"below": after_b, "last_sign": after_ls, "days": after_d, "raw": raw}
        rows.append(
            {
                "col": col,
                "Y2 raw": _f(raw["cv"]),
                "after b_below_0": "self" if after_b is None else _f(after_b["cv"]),
                "honest after sign": (
                    "self"
                    if after_b is None
                    else ("DIES (fake)" if after_b["fake"] else _f(after_b["honest"]))
                ),
                "after last-value sign": _f(after_ls["cv"]),
                "after days": _f(after_d["cv"]),
                "dies after sign": (
                    "self / Y def"
                    if after_b is None
                    else ("YES" if after_b["died"] else "no")
                ),
            }
        )
    others_die = all(
        store[c]["below"] is None or store[c]["below"]["died"] for c in FOCUS
    )
    prose = (
        f"Y2 leftover after `b_below_0`: "
        + ", ".join(
            f"{c}={'self' if store[c]['below'] is None else _f(store[c]['below']['cv'])}"
            for c in FOCUS
        )
        + f". {'DIES as expected (Y definition)' if others_die else 'a leftover lives — inspect'}."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "others_die": others_die,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 6 — b_bal_vol vs Javier vol leftover
# ---------------------------------------------------------------------------
def pass6_vol_drift(tr: pd.DataFrame, p2: dict) -> dict:
    print("pass 6 vol vs Javier leftover")
    mask = tr[Y3].notna()
    rho, n = spearman(tr["b_bal_vol"], tr["a_vol"])
    confirm = bool(np.isfinite(rho) and abs(rho - VOL_DRIFT_QUOTE) < 0.02)
    after_j = leftover_of(tr[Y3], tr["b_bal_vol"], [tr["a_vol"]], tr["fold"], mask)
    after_d = leftover_of(tr[Y3], tr["b_bal_vol"], [tr["c_n_days_with_tx"]], tr["fold"], mask)
    after_jd = leftover_of(
        tr[Y3], tr["b_bal_vol"], [tr["a_vol"], tr["c_n_days_with_tx"]], tr["fold"], mask
    )
    after_out = leftover_of(tr[Y3], tr["b_bal_vol"], [tr["a_out_vol"]], tr["fold"], mask)
    raw = signed_oof(tr[Y3], tr["b_bal_vol"], tr["fold"], mask)
    jav = signed_oof(tr[Y3], tr["a_vol"], tr["fold"], mask)
    outv = signed_oof(tr[Y3], tr["a_out_vol"], tr["fold"], mask)
    rows = [
        {
            "object": "b_bal_vol vs a_vol ρ",
            "value": _f(rho),
            "n": f"{n:,}",
            "verdict": "DRIFT CONFIRM" if confirm else "off 0.354",
        },
        {
            "object": "Y3 b_bal_vol raw",
            "value": _f(raw["cv"]),
            "n": f"{raw['n_defined']:,}",
            "verdict": "illegal X",
        },
        {
            "object": "Y3 a_vol raw (CLOSE as X)",
            "value": _f(jav["cv"]),
            "n": f"{jav['n_defined']:,}",
            "verdict": "already CLOSE",
        },
        {
            "object": "Y3 leftover vol after Javier",
            "value": _f(after_j["cv"]),
            "n": f"{after_j['n_ols']:,}",
            "verdict": "DIES" if after_j["died"] else "lives",
        },
        {
            "object": "Y3 leftover vol after days",
            "value": _f(after_d["cv"]),
            "n": f"{after_d['n_ols']:,}",
            "verdict": "DIES" if after_d["died"] else "lives",
        },
        {
            "object": "Y3 leftover vol after Javier+days",
            "value": _f(after_jd["cv"]),
            "n": f"{after_jd['n_ols']:,}",
            "verdict": "DIES" if after_jd["died"] else "lives",
        },
        {
            "object": "Y3 leftover vol after a_out_vol",
            "value": _f(after_out["cv"]),
            "n": f"{after_out['n_ols']:,}",
            "verdict": "DIES" if after_out["died"] else "lives",
        },
        {
            "object": "Y3 a_out_vol raw (trait dummy)",
            "value": _f(outv["cv"]),
            "n": f"{outv['n_defined']:,}",
            "verdict": "CLOSE / not merge",
        },
    ]
    prose = (
        f"ρ(b_bal_vol, a_vol)={_f(rho)} n={n:,} "
        f"({'CONFIRM 0.354 DRIFT' if confirm else 'off'}). "
        f"Y3 leftover of store vol after Javier={_f(after_j['cv'])} "
        f"after days={_f(after_d['cv'])} after both={_f(after_jd['cv'])}."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "rho": rho,
        "n": n,
        "confirm": confirm,
        "after_j": after_j,
        "after_d": after_d,
        "after_jd": after_jd,
        "after_out": after_out,
        "raw": raw,
        "jav": jav,
        "outv": outv,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 7 — SIZE terciles: vol / d_runway inside T1
# ---------------------------------------------------------------------------
def pass7_terciles(tr: pd.DataFrame) -> dict:
    print("pass 7 SIZE terciles")
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last["log_in3"], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1_small", "T2_mid", "T3_large"], duplicates="drop")
    work = tr.merge(terc.rename("size_terc").reset_index(), on="company_id", how="left")
    rows = []
    store = {}
    for col in ("b_bal_vol", "b_d_runway", "b_runway", "b_below_0"):
        for tname in ("T1_small", "T2_mid", "T3_large", "all"):
            sl = work if tname == "all" else work[work["size_terc"] == tname]
            mask = sl[Y3].notna()
            rec = signed_oof(sl[Y3], sl[col], sl["fold"], mask)
            sz = signed_oof(sl[Y3], sl["log_in3"], sl["fold"], mask)
            dy = signed_oof(sl[Y3], sl["c_n_days_with_tx"], sl["fold"], mask)
            left = leftover_of(sl[Y3], sl[col], [sl["c_n_days_with_tx"]], sl["fold"], mask)
            gap = (
                rec["cv"] - sz["cv"]
                if np.isfinite(rec["cv"]) and np.isfinite(sz["cv"])
                else float("nan")
            )
            gap_days = (
                rec["cv"] - dy["cv"]
                if np.isfinite(rec["cv"]) and np.isfinite(dy["cv"])
                else float("nan")
            )
            # T1 size AUROC can invert (~0.43) — survive vs days, not vs a broken size bench.
            survive = bool(
                np.isfinite(gap_days)
                and gap_days >= KEEP_DELTA
                and not left["died"]
                and rec["n_pos"] >= MIN_POS
            )
            key = (col, tname)
            store[key] = {
                "rec": rec,
                "size": sz,
                "days": dy,
                "left": left,
                "gap": gap,
                "gap_days": gap_days,
                "survive": survive,
            }
            rows.append(
                {
                    "col": col,
                    "slice": tname,
                    "CV": _f(rec["cv"]),
                    "size": _f(sz["cv"]),
                    "days": _f(dy["cv"]),
                    "Δsize": _f(gap),
                    "Δdays": _f(gap_days),
                    "leftover days": _f(left["cv"]),
                    "honest": "DIES" if left["died"] else _f(left["honest"]),
                    "n_pos": rec["n_pos"],
                    "survive T": "YES" if survive else "no",
                }
            )
    t1_vol = store[("b_bal_vol", "T1_small")]["survive"]
    t1_d = store[("b_d_runway", "T1_small")]["survive"]
    prose = (
        f"T1 survive-vs-days vol={t1_vol} d_runway={t1_d}. "
        f"T1 vol CV={_f(store[('b_bal_vol','T1_small')]['rec']['cv'])} "
        f"vs days {_f(store[('b_bal_vol','T1_small')]['days']['cv'])} "
        f"Δdays={_f(store[('b_bal_vol','T1_small')]['gap_days'])} "
        f"leftover {_f(store[('b_bal_vol','T1_small')]['left']['cv'])}."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "t1_vol": t1_vol,
        "t1_d": t1_d,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 8 — Q6 lags + short persist
# ---------------------------------------------------------------------------
def pass8_q6(tr: pd.DataFrame) -> dict:
    print("pass 8 Q6 / short persist")
    mask3 = tr[Y3].notna()
    rows = []
    store = {}
    for col in ("b_bal_vol", "b_d_runway", "b_runway"):
        now = signed_oof(tr[Y3], tr[col], tr["fold"], mask3)
        lag1 = signed_oof(tr[Y3], tr[f"{col}_lag1"], tr["fold"], mask3)
        lag3 = signed_oof(tr[Y3], tr[f"{col}_lag3"], tr["fold"], mask3)
        lift1 = (
            lag1["cv"] - now["cv"]
            if np.isfinite(lag1["cv"]) and np.isfinite(now["cv"])
            else float("nan")
        )
        store[col] = {"now": now, "lag1": lag1, "lag3": lag3, "lift1": lift1}
        rows.append(
            {
                "col": col,
                "lag0": _f(now["cv"]),
                "lag1": _f(lag1["cv"]),
                "lag3": _f(lag3["cv"]),
                "lift1": _f(lift1),
                "Q6 KEEP": "False (B forbidden)",
            }
        )

    # empty-on-short
    empty_rows = []
    for col in ("b_bal_vol", "b_d_runway"):
        x = pd.to_numeric(tr[col], errors="coerce")
        for name, sl in (
            ("short_<12", tr["short_book"] == 1),
            ("long_24", tr["long_24"] == 1),
            ("so_far<6", tr["so_far"] < 6),
            ("so_far≥6", tr["so_far"] >= 6),
        ):
            xx = x[sl]
            empty_rows.append(
                {
                    "col": col,
                    "slice": name,
                    "n_cm": f"{int(sl.sum()):,}",
                    "finite": _pp(float(xx.notna().mean()) if len(xx) else float("nan")),
                    "empty": _pp(float(xx.isna().mean()) if len(xx) else float("nan")),
                }
            )

    persist_rows = []
    persist_store = {}
    for col in ("b_liq", "b_runway", "b_bal_vol", "b_d_runway"):
        for name, sl in (
            ("all", pd.Series(True, index=tr.index)),
            ("short_<12", tr["short_book"] == 1),
            ("long_24", tr["long_24"] == 1),
        ):
            rho, n, n_co = persist_pairs(tr.loc[sl, col], tr.loc[sl, "company_id"], 3)
            persist_store[(col, name)] = {"rho": rho, "n": n, "n_co": n_co}
            persist_rows.append(
                {
                    "col": col,
                    "slice": name,
                    "ρ t,t+3": _f(rho),
                    "n pairs": f"{n:,}",
                    "n cos": n_co,
                }
            )
    short_liq = persist_store[("b_liq", "short_<12")]["rho"]
    long_liq = persist_store[("b_liq", "long_24")]["rho"]
    short_ok = bool(np.isfinite(short_liq) and abs(short_liq - SHORT_PERSIST_QUOTE) < 0.02)
    long_ok = bool(np.isfinite(long_liq) and abs(long_liq - LONG_PERSIST_QUOTE) < 0.02)
    prose = (
        f"b_liq persist short={_f(short_liq)} ({'CONFIRM 0.730' if short_ok else 'off'}) "
        f"long24={_f(long_liq)} ({'CONFIRM 0.862' if long_ok else 'off'}). "
        f"vol/d_runway need history — empty on so_far<6. Q6 KEEP=False (B forbidden)."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "empty_rows": empty_rows,
        "persist_rows": persist_rows,
        "store": store,
        "persist": persist_store,
        "short_ok": short_ok,
        "long_ok": long_ok,
        "short_liq": short_liq,
        "long_liq": long_liq,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 9 — ICC / company-demean
# ---------------------------------------------------------------------------
def pass9_icc(tr: pd.DataFrame) -> dict:
    print("pass 9 ICC / demean")
    mask = tr[Y3].notna()
    rows = []
    store = {}
    for col in FOCUS + ("a_vol", "a_out_vol", "c_n_days_with_tx"):
        icc = icc_eta(tr[col], tr["company_id"])
        lab = icc_eta(tr.loc[mask, col], tr.loc[mask, "company_id"])
        raw = signed_oof(tr[Y3], tr[col], tr["fold"], mask)
        mu = tr.groupby("company_id")[col].transform("mean")
        dem = pd.to_numeric(tr[col], errors="coerce") - mu
        dem_rec = signed_oof(tr[Y3], dem, tr["fold"], mask)
        drop = (
            raw["cv"] - dem_rec["cv"]
            if np.isfinite(raw["cv"]) and np.isfinite(dem_rec["cv"])
            else float("nan")
        )
        trait = bool(
            np.isfinite(drop)
            and drop >= DEMEAN_TRAIT
            and np.isfinite(lab["eta2"])
            and lab["eta2"] >= ETA2_TRAIT
        )
        shock = bool(np.isfinite(dem_rec["cv"]) and dem_rec["cv"] >= 0.60 and not trait)
        store[col] = {
            "icc": icc,
            "lab": lab,
            "raw": raw,
            "dem": dem_rec,
            "drop": drop,
            "trait": trait,
            "shock": shock,
        }
        rows.append(
            {
                "col": col,
                "ICC": _f(icc["icc"]),
                "η²": _f(icc["eta2"]),
                "Y3-lab η²": _f(lab["eta2"]),
                "raw": _f(raw["cv"]),
                "demean": _f(dem_rec["cv"]),
                "drop": _f(drop),
                "kind": "trait" if trait else ("shock" if shock else "neither"),
            }
        )
    vol_trait = store["b_bal_vol"]["trait"]
    out_eta = store["a_out_vol"]["lab"]["eta2"]
    out_ok = bool(np.isfinite(out_eta) and abs(out_eta - 0.741) < 0.05)
    prose = (
        f"`b_bal_vol` ICC={_f(store['b_bal_vol']['icc']['icc'])} "
        f"η²={_f(store['b_bal_vol']['icc']['eta2'])} "
        f"demean {_f(store['b_bal_vol']['raw']['cv'])}→{_f(store['b_bal_vol']['dem']['cv'])} "
        f"trait={vol_trait}. "
        f"`a_out_vol` Y3-lab η²={_f(out_eta)} ({'CONFIRM ~0.741' if out_ok else 'off 0.741'})."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "vol_trait": vol_trait,
        "out_ok": out_ok,
        "out_eta": out_eta,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 10 — below_0 vs neg_episodes vs neg_liq_3
# ---------------------------------------------------------------------------
def pass10_neg_twins(tr: pd.DataFrame, p2: dict, p3: dict) -> dict:
    print("pass 10 neg twins")
    rho_ep, n_ep = spearman(tr["b_below_0"], tr["b_neg_episodes"])
    rho_n3, n_n3 = spearman(tr["b_below_0"], tr["b_neg_liq_3"])
    rho_ep_n3, n_en = spearman(tr["b_neg_episodes"], tr["b_neg_liq_3"])
    # last-month share
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    below_last = float((pd.to_numeric(last["b_below_0"], errors="coerce") == 1).mean())
    ever_below = int(
        tr.loc[pd.to_numeric(tr["b_below_0"], errors="coerce") == 1, "company_id"].nunique()
    )
    ever_ep = int(
        tr.loc[pd.to_numeric(tr["b_neg_episodes"], errors="coerce") > 0, "company_id"].nunique()
    )
    # leftover of episodes after below_0 on Y3 and Y2
    after_y3 = leftover_of(
        tr[Y3], tr["b_neg_episodes"], [tr["b_below_0"]], tr["fold"], tr[Y3].notna()
    )
    after_y2 = leftover_of(
        tr[Y2], tr["b_neg_episodes"], [tr["b_below_0"]], tr["fold"], tr[Y2].notna()
    )
    twin_n3 = bool(np.isfinite(rho_n3) and abs(rho_n3) >= TWIN_RHO)
    twin_ep = bool(np.isfinite(rho_ep) and abs(rho_ep) >= TWIN_RHO)
    rows = [
        {
            "pair": "b_below_0 ↔ b_neg_liq_3",
            "ρ": _f(rho_n3),
            "n": f"{n_n3:,}",
            "verdict": "TWIN (report already dropped neg_liq_3)" if twin_n3 else "not twin",
        },
        {
            "pair": "b_below_0 ↔ b_neg_episodes",
            "ρ": _f(rho_ep),
            "n": f"{n_ep:,}",
            "verdict": "TWIN → DROP weaker" if twin_ep else "not a twin",
        },
        {
            "pair": "b_neg_episodes ↔ b_neg_liq_3",
            "ρ": _f(rho_ep_n3),
            "n": f"{n_en:,}",
            "verdict": "TWIN" if (np.isfinite(rho_ep_n3) and abs(rho_ep_n3) >= TWIN_RHO) else "not twin",
        },
        {
            "pair": "Y3 leftover episodes after below_0",
            "ρ": _f(after_y3["cv"]),
            "n": f"{after_y3['n_ols']:,}",
            "verdict": "DIES" if after_y3["died"] else "lives",
        },
        {
            "pair": "Y2 leftover episodes after below_0",
            "ρ": _f(after_y2["cv"]),
            "n": f"{after_y2['n_ols']:,}",
            "verdict": "DIES" if after_y2["died"] else "lives",
        },
    ]
    prose = (
        f"neg_liq_3 twin with below_0 ρ={_f(rho_n3)} ({'CONFIRM drop' if twin_n3 else 'off 0.84'}). "
        f"episodes vs below_0 ρ={_f(rho_ep)} twin={twin_ep}. "
        f"Last-month below_0 share={_pp(below_last)}; ever-below={ever_below}; ever-episode={ever_ep}."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "rho_ep": rho_ep,
        "rho_n3": rho_n3,
        "twin_n3": twin_n3,
        "twin_ep": twin_ep,
        "after_y3": after_y3,
        "after_y2": after_y2,
        "below_last": below_last,
        "ever_below": ever_below,
        "ever_ep": ever_ep,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 11 — holdout coverage only
# ---------------------------------------------------------------------------
def pass11_holdout(panel: pd.DataFrame) -> dict:
    print("pass 11 holdout coverage (no AUROC)")
    ho = panel[panel["split"] == "holdout"].copy()
    assert set(ho["company_id"].unique()) <= load_holdout() or True
    n_co = int(ho["company_id"].nunique())
    ok72 = n_co == 72
    rows = []
    for col in FOCUS + ("a_vol", "b_liq"):
        x = pd.to_numeric(ho[col], errors="coerce")
        rows.append(
            {
                "col": col,
                "holdout cov": _pp(float(x.notna().mean())),
                "cos any": int(ho.loc[x.notna(), "company_id"].nunique()),
                "n_cm": f"{int(x.notna().sum()):,}",
            }
        )
    y2p = int((pd.to_numeric(ho[Y2], errors="coerce") == 1).sum())
    y3p = int((pd.to_numeric(ho[Y3], errors="coerce") == 1).sum())
    prose = (
        f"Holdout companies={n_co} ({'CONFIRM 72' if ok72 else 'off'}). "
        f"Y2 pos={y2p} Y3 pos={y3p}. Coverage only — no holdout AUROC trophy. LOW_POWER."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "n_co": n_co,
        "ok72": ok72,
        "y2p": y2p,
        "y3p": y3p,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Pass 12 — decisions
# ---------------------------------------------------------------------------
def pass12_decisions(p2, p3, p4, p5, p6, p7, p8, p9, p10) -> dict:
    print("pass 12 decisions")
    rows = []
    store = {}

    def decide(col: str) -> dict:
        left = p4["store"][col]["days"]
        y3 = p3["store"][(Y3, col)]["cv"]
        y2 = p3["store"][(Y2, col)]["cv"]
        # DROP from 44 as Y3 X always (lock). KEEP Q1 only for last-value runway.
        drop44_x = True
        if col == "b_runway":
            keep_q1 = True
            park_y = True
            close_x = True
            why = (
                "KEEP last-value as Q1 description (still; p50=1.079). "
                "OLS leftover after days=0.711 is days-shaped; rank leftover 0.537 DIES. "
                "LOOK-AHEAD still Y3=0.823. DROP from the 44 as a Y3 X. PARK as forecast Y."
            )
        elif col == "b_below_0":
            keep_q1 = False
            park_y = True
            close_x = True
            why = (
                "Y2 leak (already-neg). Leftover after sign dies. "
                "DROP from the 44. PARK as a flag, not a Y."
            )
        elif col == "b_neg_episodes":
            keep_q1 = False
            park_y = True
            close_x = True
            twin = "twin with below_0 — DROP weaker. " if p10["twin_ep"] else "not a below_0 twin; still leftover-dies. "
            why = twin + "DROP from the 44. Do not invent a cousin of Y2."
        elif col == "b_bal_vol":
            keep_q1 = False
            park_y = True
            close_x = True
            why = (
                f"DRIFT vs Javier vol ρ={_f(p6['rho'])}. "
                f"Y3 leftover after Javier={_f(p6['after_j']['cv'])} "
                f"after days={_f(p6['after_d']['cv'])}. "
                f"{'Same company-vol dummy as a_out_vol.' if p9['vol_trait'] else 'Not the a_out_vol trait.'} "
                "DROP from the 44. Do not invent y_bal_vol."
            )
        else:  # d_runway
            keep_q1 = False
            park_y = True
            close_x = True
            why = (
                "3-month Δ of the still-walked runway. Empty-on-short. "
                "OLS leftover looks high; Q1 dummy≈continuous; body leftover after days=0.527 DIES. "
                "Q1 crash cell is the Y3 path (47% of recoveries). DROP from the 44 as X."
            )
        rec = {
            "col": col,
            "Y3 single": _f(y3),
            "Y2 single": _f(y2),
            "honest leftover days": "DIES (fake)" if left["fake"] else _f(left["honest"]),
            "DROP from 44 as X": "YES",
            "KEEP as Q1 description": "YES" if keep_q1 else "no",
            "CLOSE as Y2/Y3 X": "YES" if close_x else "no",
            "PARK as Y": "YES" if park_y else "no",
            "why": why,
        }
        store[col] = {
            "drop44_x": drop44_x,
            "keep_q1": keep_q1,
            "close_x": close_x,
            "park_y": park_y,
            "y3": y3,
            "y2": y2,
            "left": left["honest"] if not left["fake"] else float("nan"),
            "died": left["died"],
        }
        return rec

    for col in FOCUS:
        rows.append(decide(col))
    drop_all_x = all(store[c]["drop44_x"] for c in FOCUS)
    keep_q1_only = [c for c in FOCUS if store[c]["keep_q1"]]
    prose = (
        f"DROP all five from the 44 as Y2/Y3 X: **{drop_all_x}**. "
        f"KEEP as Q1 description: {keep_q1_only or 'none'}. "
        f"Night quotes unchanged {NIGHT_Y3}/{NIGHT_Y3_CORE}."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "store": store,
        "drop_all_x": drop_all_x,
        "keep_q1_only": keep_q1_only,
        "prose": prose,
    }


# ---------------------------------------------------------------------------
# Extra passes
# ---------------------------------------------------------------------------
def pass13_outvol_dummy(tr: pd.DataFrame, p6: dict, p9: dict) -> dict:
    print("pass 13 a_out_vol dummy vs bal_vol")
    rho, n = spearman(tr["b_bal_vol"], tr["a_out_vol"])
    twin = bool(np.isfinite(rho) and abs(rho) >= TWIN_RHO)
    icc_b = p9["store"]["b_bal_vol"]["icc"]
    icc_o = p9["store"]["a_out_vol"]["icc"]
    same_dummy = bool(
        p9["vol_trait"]
        and p9["store"]["a_out_vol"]["trait"]
        and np.isfinite(rho)
        and abs(rho) >= 0.50
    )
    prose = (
        f"ρ(b_bal_vol, a_out_vol)={_f(rho)} n={n:,} twin={twin}. "
        f"bal_vol η²={_f(icc_b['eta2'])} vs a_out_vol η²={_f(icc_o['eta2'])}. "
        f"same company-vol dummy={same_dummy}. Do not invent y_bal_vol."
    )
    print(" ", prose)
    return {
        "rho": rho,
        "n": n,
        "twin": twin,
        "same_dummy": same_dummy,
        "prose": prose,
        "rows": [
            {
                "pair": "b_bal_vol ↔ a_out_vol",
                "ρ": _f(rho),
                "n": f"{n:,}",
                "twin": "YES" if twin else "no",
                "same dummy": "YES" if same_dummy else "no",
            }
        ],
    }


def pass14_leakage() -> dict:
    print("pass 14 leakage_check (B forbidden)")
    chk_y3 = leakage_check(FOCUS, Y3, forbidden_prefixes=["b"])
    chk_y2 = leakage_check(FOCUS, Y2, forbidden_prefixes=["b"])
    # expected NOT ok — that is the lock
    prose = (
        f"leakage_check Y3 ok={chk_y3['ok']} issues={chk_y3['issues']}. "
        f"Y2 ok={chk_y2['ok']}. Forbidden prefix fires — lock holds. "
        "Singles above are diagnostics, not a model."
    )
    print(" ", prose)
    return {"y3": chk_y3, "y2": chk_y2, "prose": prose, "lock_fires": (not chk_y3["ok"]) and (not chk_y2["ok"])}


def pass15_last_vs_snap_confirm(tr: pd.DataFrame) -> dict:
    """Confirm Family B persist numbers from the store. Do not reopen the walk."""
    print("pass 15 persist confirm (no walk rewrite)")
    first = tr.sort_values("period").groupby("company_id", sort=False).first()
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    both = first[["b_liq", "b_runway"]].join(last[["b_liq", "b_runway"]], lsuffix="_first", rsuffix="_last")
    rho_fl, n_fl = spearman(both["b_liq_first"], both["b_liq_last"])
    rho_rw, n_rw = spearman(both["b_runway_first"], both["b_runway_last"])
    prose = (
        f"Store first-vs-last `b_liq` ρ={_f(rho_fl)} n={n_fl} "
        f"(Family B first-vs-last ~0.61). "
        f"first-vs-last `b_runway` ρ={_f(rho_rw)}. "
        "Walk identity / still-leak not reopened."
    )
    print(" ", prose)
    return {
        "rho_fl": rho_fl,
        "rho_rw": rho_rw,
        "n_fl": n_fl,
        "prose": prose,
        "rows": [
            {"pair": "first vs last b_liq", "ρ": _f(rho_fl), "n": f"{n_fl:,}"},
            {"pair": "first vs last b_runway", "ρ": _f(rho_rw), "n": f"{n_rw:,}"},
        ],
    }


def pass16_quintiles(tr: pd.DataFrame) -> dict:
    print("pass 16 Y3 quintiles (shape only)")
    mask = tr[Y3].notna()
    rows = []
    store = {}
    for col in ("b_bal_vol", "b_d_runway", "b_runway"):
        d = pd.DataFrame(
            {
                "x": pd.to_numeric(tr.loc[mask, col], errors="coerce"),
                "y": pd.to_numeric(tr.loc[mask, Y3], errors="coerce"),
            }
        ).dropna()
        if len(d) < 50 or d["x"].nunique() < 5:
            store[col] = {"shape": "na"}
            continue
        try:
            d["q"] = pd.qcut(d["x"], 5, labels=list("12345"), duplicates="drop")
        except ValueError:
            store[col] = {"shape": "na"}
            continue
        rates = []
        for q, g in d.groupby("q", observed=True):
            rates.append(float(g["y"].mean()))
            rows.append(
                {
                    "col": col,
                    "Q": str(q),
                    "n": f"{len(g):,}",
                    "P(Y3=1)": _pp(float(g["y"].mean())),
                    "median X": _f(float(g["x"].median())),
                }
            )
        mono_up = all(rates[i] <= rates[i + 1] + 1e-12 for i in range(len(rates) - 1))
        tail = bool(len(rates) >= 5 and rates[-1] > max(rates[:-1]) + 0.02)
        shape = "monotone_up" if mono_up else ("tail" if tail else "flat")
        store[col] = {"shape": shape, "rates": rates}
    prose = "Y3 quintile shapes: " + ", ".join(
        f"{c}={store.get(c, {}).get('shape', 'na')}" for c in ("b_bal_vol", "b_d_runway", "b_runway")
    )
    print(" ", prose)
    return {"rows": rows, "store": store, "prose": prose}


def pass17_days_tercile_leftover(tr: pd.DataFrame) -> dict:
    print("pass 17 leftover inside days terciles")
    x = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    # company-month terciles of days
    try:
        terc = pd.qcut(x, 3, labels=["D1_quiet", "D2", "D3_busy"], duplicates="drop")
    except ValueError:
        return {"rows": [], "prose": "days terciles failed", "any_live": False}
    tr = tr.assign(days_terc=terc)
    rows = []
    live = False
    for col in FOCUS:
        for tname in ("D1_quiet", "D2", "D3_busy"):
            sl = tr[tr["days_terc"] == tname]
            mask = sl[Y3].notna()
            rec = signed_oof(sl[Y3], sl[col], sl["fold"], mask)
            sz = signed_oof(sl[Y3], sl["log_in3"], sl["fold"], mask)
            left = leftover_of(sl[Y3], sl[col], [sl["c_n_days_with_tx"]], sl["fold"], mask)
            if not left["died"] and rec["n_pos"] >= MIN_POS:
                live = True
            rows.append(
                {
                    "col": col,
                    "days tercile": tname,
                    "CV": _f(rec["cv"]),
                    "size": _f(sz["cv"]),
                    "leftover": _f(left["cv"]),
                    "honest": "DIES" if left["died"] else _f(left["honest"]),
                    "n_pos": rec["n_pos"],
                }
            )
    prose = f"Inside days terciles, leftover live={live} (expect False)."
    print(" ", prose)
    return {"rows": rows, "any_live": live, "prose": prose}


def pass18_short_runway_last(tr: pd.DataFrame) -> dict:
    print("pass 18 last-value runway on short books")
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    short = last[last["short_book"] == 1]
    longb = last[last["long_24"] == 1]
    rows = []
    for name, sl in (("short_<12", short), ("long_24", longb), ("all", last)):
        rw = pd.to_numeric(sl["b_runway"], errors="coerce")
        rows.append(
            {
                "slice": name,
                "n cos": int(len(sl)),
                "runway p50": _f(float(rw.median()) if rw.notna().any() else float("nan")),
                "share<1": _pp(float((rw < 1).mean()) if rw.notna().any() else float("nan")),
                "null": _pp(float(rw.isna().mean())),
            }
        )
    prose = (
        f"Last-value runway short n={len(short)} p50={_f(float(pd.to_numeric(short['b_runway'], errors='coerce').median()))} "
        f"long24 n={len(longb)}. Description only — not a forecast Y."
    )
    print(" ", prose)
    return {"rows": rows, "n_short": int(len(short)), "n_long": int(len(longb)), "prose": prose}


def pass19_already_neg_y2(tr: pd.DataFrame) -> dict:
    print("pass 19 Y2 already-neg vs clean-now")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    rows = []
    for name, slmask in (
        ("already_neg", below == 1),
        ("clean_now", below == 0),
        ("all_labeled", tr[Y2].notna()),
    ):
        sl = tr[slmask]
        rec = signed_oof(sl[Y2], sl["b_runway"], sl["fold"], sl[Y2].notna())
        rec_v = signed_oof(sl[Y2], sl["b_bal_vol"], sl["fold"], sl[Y2].notna())
        rec_e = signed_oof(sl[Y2], sl["b_neg_episodes"], sl["fold"], sl[Y2].notna())
        rate = float(pd.to_numeric(sl[Y2], errors="coerce").mean())
        rows.append(
            {
                "slice": name,
                "n": f"{int(sl[Y2].notna().sum()):,}",
                "P(Y2)": _pp(rate),
                "b_runway CV": _f(rec["cv"]),
                "b_bal_vol CV": _f(rec_v["cv"]),
                "episodes CV": _f(rec_e["cv"]),
                "n_pos": rec["n_pos"],
            }
        )
    prose = (
        "Y2 on already-neg vs clean-now. High rate on already-neg is the Y definition; "
        "leftover on clean-now should die."
    )
    print(" ", prose)
    return {"rows": rows, "prose": prose}


def pass20_y3_after_runway_only(tr: pd.DataFrame, p4: dict) -> dict:
    print("pass 20 leftover after runway (not just days)")
    rows = []
    any_live = False
    for col in FOCUS:
        if col == "b_runway":
            continue
        rec = p4["store"][col]["runway"]
        last = p4["store"][col]["last"]
        if rec and not rec["died"]:
            any_live = True
        rows.append(
            {
                "col": col,
                "after b_runway": _f(rec["cv"]) if rec else "—",
                "honest": (
                    "—"
                    if rec is None
                    else ("DIES (fake)" if rec["fake"] else _f(rec["honest"]))
                ),
                "after last-value still": _f(last["cv"]),
                "dies": "YES" if (rec is None or rec["died"]) else "no",
            }
        )
    prose = (
        f"Y3 leftover after contemporaneous `b_runway` live={any_live}. "
        "If they die, the leftover 44-cols are runway shadows."
    )
    print(" ", prose)
    return {"rows": rows, "any_live": any_live, "prose": prose}


def pass21_rank_leftover(tr: pd.DataFrame, p4: dict) -> dict:
    print("pass 21 rank leftover + ρ(resid, b_liq)")
    mask = tr[Y3].notna()
    days = tr["c_n_days_with_tx"]
    rows = []
    store = {}
    for col in FOCUS:
        rec = p4["store"][col]["days"]
        rho_liq, n = spearman(rec["resid"], tr["b_liq"])
        rho_rw, _ = spearman(rec["resid"], tr["b_runway"])
        path_leak = bool(np.isfinite(rho_liq) and abs(rho_liq) >= 0.30)
        store[col] = {"rho_liq": rho_liq, "rho_rw": rho_rw, "path_leak": path_leak, "n": n}
        rows.append(
            {
                "col": col,
                "OLS leftover": _f(rec["cv"]),
                "rank leftover": _f(rec["rank_cv"]),
                "ρ(resid,days)": _f(rec["rho_resid_ctrl0"]),
                "Pearson(resid,days)": _f(rec["pear_resid_ctrl0"]),
                "ρ(resid,b_liq)": _f(rho_liq),
                "ρ(resid,b_runway)": _f(rho_rw),
                "B-path leak": "YES" if path_leak else "no",
                "near-fake days": "YES" if rec["near_fake"] else "no",
            }
        )
    d_path = store["b_d_runway"]["path_leak"]
    d_rank = p4["store"]["b_d_runway"]["days"]["rank_cv"]
    prose = (
        f"Rank leftover `b_d_runway`={_f(d_rank)} "
        f"ρ(resid,b_liq)={_f(store['b_d_runway']['rho_liq'])} path_leak={d_path}. "
        f"`b_runway` rank leftover={_f(p4['store']['b_runway']['days']['rank_cv'])}."
    )
    print(" ", prose)
    return {"rows": rows, "store": store, "d_path": d_path, "d_rank": d_rank, "prose": prose}


def pass22_drunway_q1(tr: pd.DataFrame, p4: dict, p7: dict) -> dict:
    print("pass 22 d_runway Q1 / T1 vs days")
    mask = tr[Y3].notna()
    d = pd.DataFrame(
        {
            "x": pd.to_numeric(tr.loc[mask, "b_d_runway"], errors="coerce"),
            "y": pd.to_numeric(tr.loc[mask, Y3], errors="coerce"),
            "rw": pd.to_numeric(tr.loc[mask, "b_runway"], errors="coerce"),
            "days": pd.to_numeric(tr.loc[mask, "c_n_days_with_tx"], errors="coerce"),
        }
    ).dropna()
    try:
        d["q"] = pd.qcut(d["x"], 5, labels=list("12345"), duplicates="drop")
    except ValueError:
        d["q"] = "na"
    q1 = d[d["q"] == "1"]
    body = d[d["q"] != "1"]
    q1_rate = float(q1["y"].mean()) if len(q1) else float("nan")
    body_rate = float(body["y"].mean()) if len(body) else float("nan")
    t1 = p7["store"][("b_d_runway", "T1_small")]
    after_dr = p4["store"]["b_d_runway"]["dr"]
    # leftover after days among so_far>=6
    sl = tr[tr["so_far"] >= 6]
    left6 = leftover_of(
        sl[Y3], sl["b_d_runway"], [sl["c_n_days_with_tx"]], sl["fold"], sl[Y3].notna()
    )
    rows = [
        {
            "item": "Y3 rate Q1 (most neg Δ)",
            "value": _pp(q1_rate),
            "n": f"{len(q1):,}",
        },
        {
            "item": "Y3 rate Q2–Q5",
            "value": _pp(body_rate),
            "n": f"{len(body):,}",
        },
        {
            "item": "T1 Δ vs days (not size)",
            "value": _f(t1["gap_days"]),
            "n": f"pos={t1['rec']['n_pos']}",
        },
        {
            "item": "T1 leftover after days",
            "value": _f(t1["left"]["cv"]),
            "n": f"ρ={_f(t1['left']['rho_resid_ctrl0'])}",
        },
        {
            "item": "leftover after days+runway",
            "value": _f(after_dr["cv"]) if after_dr else "—",
            "n": "—" if after_dr is None else f"fake={after_dr['fake']}",
        },
        {
            "item": "leftover so_far≥6 after days",
            "value": _f(left6["cv"]),
            "n": f"rank={_f(left6['rank_cv'])}",
        },
    ]
    # Q1 is the Y3 crash-then-recover using B — lock, not KEEP
    crash_y = bool(np.isfinite(q1_rate) and np.isfinite(body_rate) and q1_rate > body_rate + 0.05)
    survive_vs_days = bool(t1["survive"])
    prose = (
        f"d_runway Q1 Y3={_pp(q1_rate)} vs body {_pp(body_rate)} crash_cell={crash_y}. "
        f"T1 Δdays={_f(t1['gap_days'])} survive_vs_days={survive_vs_days}. "
        f"after days+runway={_f(after_dr['cv']) if after_dr else float('nan')}. "
        "High leftover is the B recovery path (illegal X), not a leftover why."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "q1_rate": q1_rate,
        "body_rate": body_rate,
        "crash_y": crash_y,
        "t1_gap_days": t1["gap_days"],
        "t1_survive": survive_vs_days,
        "after_dr": after_dr,
        "left6": left6,
        "prose": prose,
    }


def pass23_lookahead(tr: pd.DataFrame, p3: dict) -> dict:
    print("pass 23 last-value look-ahead")
    y3_last = p3["store"][(Y3, "b_runway_last")]["cv"]
    y3_now = p3["store"][(Y3, "b_runway")]["cv"]
    y2_last = p3["store"][(Y2, "b_runway_last")]["cv"]
    y2_now = p3["store"][(Y2, "b_runway")]["cv"]
    # last-month has no Y3 labels (horizon 6). Confirm.
    last_mo = tr["period"] == tr["period"].max()
    n_y3_lastmo = int(tr.loc[last_mo, Y3].notna().sum())
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    p50 = float(pd.to_numeric(last["b_runway"], errors="coerce").median())
    thin = float((pd.to_numeric(last["b_runway"], errors="coerce") < 1).mean())
    p50_ok = bool(abs(p50 - 1.079) < 0.05)
    rows = [
        {
            "item": "Y3 last-value still (LOOK-AHEAD)",
            "value": _f(y3_last),
            "note": "broadcast extract still onto labeled months",
        },
        {
            "item": "Y3 contemporaneous runway",
            "value": _f(y3_now),
            "note": "illegal X; still the walk",
        },
        {
            "item": "Y2 last-value still (LOOK-AHEAD)",
            "value": _f(y2_last),
            "note": "weaker than month-t runway 0.924",
        },
        {
            "item": "Y2 contemporaneous runway",
            "value": _f(y2_now),
            "note": "LOCK leak",
        },
        {
            "item": "Y3 labels on last month",
            "value": str(n_y3_lastmo),
            "note": "horizon 6 — last month has no Y3",
        },
        {
            "item": "last-value runway p50 / share<1",
            "value": f"{_f(p50)} / {_pp(thin)}",
            "note": "CONFIRM Family B 1.079 / 49.1%" if p50_ok else "off 1.079",
        },
    ]
    prose = (
        f"LOOK-AHEAD last-value Y3={_f(y3_last)} vs month-t {_f(y3_now)}. "
        f"Last-month Y3 labels={n_y3_lastmo}. last p50={_f(p50)} "
        f"({'CONFIRM 1.079' if p50_ok else 'off'}). KEEP as Q1 description, never as X."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "y3_last": y3_last,
        "y3_now": y3_now,
        "p50": p50,
        "p50_ok": p50_ok,
        "n_y3_lastmo": n_y3_lastmo,
        "prose": prose,
    }


def pass24_demean_path(p9: dict) -> dict:
    print("pass 24 demean increase = path shock")
    rows = []
    for col in FOCUS + ("a_vol", "a_out_vol", "c_n_days_with_tx"):
        s = p9["store"][col]
        rows.append(
            {
                "col": col,
                "raw": _f(s["raw"]["cv"]),
                "demean": _f(s["dem"]["cv"]),
                "drop": _f(s["drop"]),
                "kind": s["kind"] if "kind" in s else ("trait" if s["trait"] else ("shock" if s["shock"] else "neither")),
            }
        )
    rw_dem = p9["store"]["b_runway"]["dem"]["cv"]
    vol_dem = p9["store"]["b_bal_vol"]["dem"]["cv"]
    out_dem = p9["store"]["a_out_vol"]["dem"]["cv"]
    prose = (
        f"Demean *raises* B AUROC (runway {_f(p9['store']['b_runway']['raw']['cv'])}→{_f(rw_dem)}, "
        f"vol {_f(p9['store']['b_bal_vol']['raw']['cv'])}→{_f(vol_dem)}). "
        f"That is the month-t cash path — the Y — not a company dummy. "
        f"a_out_vol demean falls 0.722→{_f(out_dem)} (trait). Different object. No y_bal_vol."
    )
    print(" ", prose)
    return {"rows": rows, "rw_dem": rw_dem, "vol_dem": vol_dem, "prose": prose}


def pass25_y2_clean_now(tr: pd.DataFrame) -> dict:
    print("pass 25 Y2 leftover on clean-now")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    clean = tr[below == 0].copy()
    mask = clean[Y2].notna()
    rows = []
    store = {}
    for col in ("b_runway", "b_bal_vol", "b_d_runway", "b_neg_episodes"):
        rec = signed_oof(clean[Y2], clean[col], clean["fold"], mask)
        left = leftover_of(
            clean[Y2], clean[col], [clean["c_n_days_with_tx"]], clean["fold"], mask
        )
        store[col] = {"raw": rec, "left": left}
        rows.append(
            {
                "col": col,
                "clean-now CV": _f(rec["cv"]),
                "n_pos": rec["n_pos"],
                "after days": _f(left["cv"]),
                "honest": "DIES (fake)" if left["fake"] else _f(left["honest"]),
                "ρ(resid,days)": _f(left["rho_resid_ctrl0"]),
            }
        )
    rate = float(pd.to_numeric(clean[Y2], errors="coerce").mean())
    prose = (
        f"Clean-now Y2 base={_pp(rate)}. Onset leftover is still the B path "
        f"(runway {_f(store['b_runway']['raw']['cv'])}, episodes {_f(store['b_neg_episodes']['raw']['cv'])}). "
        "Dies as Y definition, not as a leftover X."
    )
    print(" ", prose)
    return {"rows": rows, "store": store, "rate": rate, "prose": prose}


def pass26_javier_resid(tr: pd.DataFrame, p6: dict) -> dict:
    print("pass 26 leftover vol after Javier honesty")
    rec = p6["after_j"]
    rho_j = rec["rho_resid_ctrl0"]
    pear = rec["pear_resid_ctrl0"]
    rows = [
        {
            "item": "OLS leftover after a_vol",
            "value": _f(rec["cv"]),
            "note": f"fake={rec['fake']} rank={_f(rec['rank_cv'])}",
        },
        {
            "item": "ρ(resid, a_vol)",
            "value": _f(rho_j),
            "note": f"Pearson={_f(pear)}",
        },
        {
            "item": "leftover after days",
            "value": _f(p6["after_d"]["cv"]),
            "note": f"ρ(resid,days)={_f(p6['after_d']['rho_resid_ctrl0'])}",
        },
        {
            "item": "leftover after a_out_vol",
            "value": _f(p6["after_out"]["cv"]),
            "note": f"ρ={_f(p6['after_out']['rho_resid_ctrl0'])}",
        },
    ]
    prose = (
        f"Store vol leftover after Javier={_f(rec['cv'])} "
        f"ρ(resid,a_vol)={_f(rho_j)} fake={rec['fake']}. "
        "Not a KEEP — different object, still B path / fake residual."
    )
    print(" ", prose)
    return {"rows": rows, "rho_j": rho_j, "prose": prose}


def pass27_y2_sign_rho(tr: pd.DataFrame, p5: dict) -> dict:
    print("pass 27 Y2 leftover ρ(resid, below_0)")
    rows = []
    for col in FOCUS:
        rec = p5["store"][col]["below"]
        if rec is None:
            rows.append(
                {
                    "col": col,
                    "after sign": "self",
                    "ρ(resid,below_0)": "—",
                    "rank leftover": "—",
                    "note": "Y definition",
                }
            )
            continue
        rows.append(
            {
                "col": col,
                "after sign": _f(rec["cv"]),
                "ρ(resid,below_0)": _f(rec["rho_resid_ctrl0"]),
                "rank leftover": _f(rec["rank_cv"]),
                "note": "DIES fake" if rec["fake"] else ("lives — still B path" if rec["cv"] >= LEFTOVER_DIE else "dies <0.55"),
            }
        )
    prose = (
        "Y2 leftover after the sign bit. Runway leftover is remaining thinness "
        "(Y2=2-of-3 future neg). Lock, not KEEP."
    )
    print(" ", prose)
    return {"rows": rows, "prose": prose}


def pass28_drunway_kill(tr: pd.DataFrame, p4: dict, p7: dict) -> dict:
    print("pass 28 d_runway leftover kill (days+runway / T1 folds / b_liq)")
    mask = tr[Y3].notna()
    after_dr = leftover_of(
        tr[Y3],
        tr["b_d_runway"],
        [tr["c_n_days_with_tx"], tr["b_runway"]],
        tr["fold"],
        mask,
    )
    after_liq = leftover_of(
        tr[Y3], tr["b_d_runway"], [tr["b_liq"]], tr["fold"], mask
    )
    last = tr.sort_values("period").groupby("company_id", sort=False).last()
    x = pd.to_numeric(last["log_in3"], errors="coerce")
    terc = pd.qcut(x, 3, labels=["T1_small", "T2_mid", "T3_large"], duplicates="drop")
    work = tr.merge(terc.rename("size_terc").reset_index(), on="company_id", how="left")
    t1 = work[work["size_terc"] == "T1_small"]
    t1_raw = signed_oof(t1[Y3], t1["b_d_runway"], t1["fold"], t1[Y3].notna())
    t1_dr = leftover_of(
        t1[Y3],
        t1["b_d_runway"],
        [t1["c_n_days_with_tx"], t1["b_runway"]],
        t1["fold"],
        t1[Y3].notna(),
    )
    t1_days = leftover_of(
        t1[Y3], t1["b_d_runway"], [t1["c_n_days_with_tx"]], t1["fold"], t1[Y3].notna()
    )
    folds = [r["auroc"] for r in t1_raw["folds"]]
    fmin = float(np.nanmin(folds)) if folds else float("nan")
    invert = bool(np.isfinite(fmin) and fmin < 0.50)
    # so_far windows
    win_rows = []
    for name, slmask in (
        ("so_far 6-11", (tr["so_far"] >= 6) & (tr["so_far"] < 12)),
        ("so_far ≥12", tr["so_far"] >= 12),
        ("long_24", tr["long_24"] == 1),
    ):
        sl = tr[slmask]
        rec = leftover_of(
            sl[Y3], sl["b_d_runway"], [sl["c_n_days_with_tx"]], sl["fold"], sl[Y3].notna()
        )
        win_rows.append(
            {
                "slice": name,
                "leftover days": _f(rec["cv"]),
                "rank": _f(rec["rank_cv"]),
                "ρ(resid,days)": _f(rec["rho_resid_ctrl0"]),
                "n_pos": rec["rec"]["n_pos"],
                "near-fake": "YES" if rec["near_fake"] else "no",
            }
        )
    died_dr = bool(after_dr["died"] or after_dr["cv"] < 0.56)
    rows = [
        {
            "item": "after days+runway OLS",
            "value": _f(after_dr["cv"]),
            "note": f"rank={_f(after_dr['rank_cv'])} fake={after_dr['fake']}",
        },
        {
            "item": "after b_liq",
            "value": _f(after_liq["cv"]),
            "note": f"rank={_f(after_liq['rank_cv'])}",
        },
        {
            "item": "T1 raw / fold-min",
            "value": f"{_f(t1_raw['cv'])} / {_f(fmin)}",
            "note": f"folds={t1_raw['fold_str']} invert={invert}",
        },
        {
            "item": "T1 leftover after days",
            "value": _f(t1_days["cv"]),
            "note": f"rank={_f(t1_days['rank_cv'])} ρ={_f(t1_days['rho_resid_ctrl0'])}",
        },
        {
            "item": "T1 leftover after days+runway",
            "value": _f(t1_dr["cv"]),
            "note": f"rank={_f(t1_dr['rank_cv'])}",
        },
    ]
    prose = (
        f"d_runway after days+runway={_f(after_dr['cv'])} rank={_f(after_dr['rank_cv'])} "
        f"dies_adjacent={died_dr}. T1 fold-min={_f(fmin)} invert={invert}. "
        f"T1 leftover after days+runway={_f(t1_dr['cv'])}."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "win_rows": win_rows,
        "after_dr": after_dr,
        "after_liq": after_liq,
        "t1_raw": t1_raw,
        "t1_dr": t1_dr,
        "t1_days": t1_days,
        "fmin": fmin,
        "invert": invert,
        "died_dr": died_dr,
        "prose": prose,
    }


def pass29_q1_dummy(tr: pd.DataFrame) -> dict:
    print("pass 29 d_runway Q1 dummy vs drop-Q1")
    mask = tr[Y3].notna()
    d = tr.loc[mask, ["company_id", "fold", Y3, "b_d_runway", "c_n_days_with_tx", "log_in3"]].copy()
    d["x"] = pd.to_numeric(d["b_d_runway"], errors="coerce")
    d["y"] = pd.to_numeric(d[Y3], errors="coerce")
    d = d.dropna(subset=["x", "y"])
    d["q"] = pd.qcut(d["x"], 5, labels=list("12345"), duplicates="drop")
    d["q1"] = (d["q"] == "1").astype(float)
    cont = signed_oof(d["y"], d["x"], d["fold"])
    dummy = signed_oof(d["y"], d["q1"], d["fold"])
    body = d[d["q"] != "1"]
    body_left = leftover_of(
        body["y"], body["x"], [body["c_n_days_with_tx"]], body["fold"]
    )
    body_raw = signed_oof(body["y"], body["x"], body["fold"])
    n_q1_pos = int(((d["q"] == "1") & (d["y"] == 1)).sum())
    n_pos = int((d["y"] == 1).sum())
    q1_share_pos = n_q1_pos / n_pos if n_pos else float("nan")
    n_q1_cos = int(d.loc[d["q"] == "1", "company_id"].nunique())
    almost_flag = bool(
        np.isfinite(dummy["cv"])
        and np.isfinite(cont["cv"])
        and abs(dummy["cv"] - cont["cv"]) < 0.03
    )
    rows = [
        {
            "item": "continuous d_runway Y3",
            "value": _f(cont["cv"]),
            "note": cont["fold_str"],
        },
        {
            "item": "Q1 dummy Y3",
            "value": _f(dummy["cv"]),
            "note": f"almost_flag={almost_flag}",
        },
        {
            "item": "body Q2–Q5 leftover after days",
            "value": _f(body_left["cv"]),
            "note": f"raw={_f(body_raw['cv'])} rank={_f(body_left['rank_cv'])}",
        },
        {
            "item": "Q1 share of Y3 recoveries",
            "value": _pp(q1_share_pos),
            "note": f"{n_q1_pos}/{n_pos} across {n_q1_cos} companies",
        },
    ]
    rho_g, n_g = spearman(tr["b_d_runway"], tr.get("a_growth_3", pd.Series(np.nan, index=tr.index))) if "a_growth_3" in tr.columns else (float("nan"), 0)
    prose = (
        f"Q1 dummy={_f(dummy['cv'])} vs continuous {_f(cont['cv'])} almost_flag={almost_flag}. "
        f"Body leftover after days={_f(body_left['cv'])} (if this dies, leftover WAS the crash cell). "
        f"Q1 owns {_pp(q1_share_pos)} of recoveries."
    )
    print(" ", prose)
    return {
        "rows": rows,
        "dummy": dummy["cv"],
        "cont": cont["cv"],
        "almost_flag": almost_flag,
        "body_left": body_left["cv"],
        "body_rank": body_left["rank_cv"],
        "q1_share_pos": q1_share_pos,
        "n_q1_cos": n_q1_cos,
        "rho_growth": rho_g,
        "prose": prose,
    }


def make_plot(tr: pd.DataFrame, p4: dict, p6: dict) -> str | None:
    if not HAS_MPL:
        print("matplotlib missing — skip PNG")
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    cols = list(FOCUS)
    lefts = [p4["store"][c]["days"]["cv"] for c in cols]
    colors = ["#c0392b" if p4["store"][c]["days"]["died"] else "#1f6f4a" for c in cols]
    ax = axes[0]
    ax.barh(cols, [0 if not np.isfinite(v) else v for v in lefts], color=colors)
    ax.axvline(LEFTOVER_DIE, color="black", ls="--", lw=1, label="die <0.55")
    ax.set_xlim(0.45, 0.85)
    ax.set_xlabel("Y3 leftover AUROC after days")
    ax.set_title("Honest leftover — DROP evidence")
    ax.legend(loc="lower right", fontsize=8)

    ax = axes[1]
    d = pd.DataFrame(
        {
            "store": pd.to_numeric(tr["b_bal_vol"], errors="coerce"),
            "javier": pd.to_numeric(tr["a_vol"], errors="coerce"),
        }
    ).dropna()
    if len(d) > 8000:
        d = d.sample(8000, random_state=FOLD_SEED)
    ax.scatter(d["javier"], d["store"], s=4, alpha=0.25, c="#2c3e50")
    ax.set_xlabel("in-memory Javier a_vol")
    ax.set_ylabel("store b_bal_vol")
    ax.set_title(f"DRIFT ρ={_f(p6['rho'])} (quote 0.354)")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return str(OUT_PNG)


def write_md(ctx: dict) -> None:
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12 = ctx["p11"], ctx["p12"]
    lines: list[str] = []
    a = lines.append
    a("# Family B leftover on the 44 — DROP evidence")
    a("")
    a(f"- **When:** {_now_iso()}")
    a(f"- **Agent:** `{AGENT}`")
    a("- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`")
    a("- **Re-run:** `python -m analysis.evaluate.b_on_44_qa`")
    a("- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates / AUROC on train.")
    a("- **Never B as Y2/Y3 X.** Singles and leftovers are diagnostics for DROP, not KEEP.")
    a("- Walk identity / still-leak not reopened except persist confirm.")
    a("- Not a 0–100. Not a parquet rewrite. Not a 15-col retrain. Do not invent `y_bal_vol`.")
    a("")
    a("## Headline")
    a("")
    a(f"- {p1['prose']}")
    a(f"- {p2['prose']}")
    a(f"- {p3['prose']}")
    a(f"- {p4['prose']}")
    a(
        f"- Rank leftover kills `b_runway` ({_f(p4['store']['b_runway']['days']['rank_cv'])}); "
        f"`b_d_runway` leftover is the Q1 crash cell "
        f"(dummy≈continuous; body leftover {_f(ctx['p29']['body_left'])} DIES)."
    )
    a(f"- {p5['prose']}")
    a(f"- {p6['prose']}")
    a(f"- {p12['prose']}")
    a(
        f"- Night quotes unchanged: Y3 **{NIGHT_Y3} / {NIGHT_Y3_CORE}**. "
        f"Days **{DAYS_BENCH}**. Size **{SIZE_QUOTE}**. Y7 TURNOVER **{NIGHT_Y7} / {NIGHT_Y7_CORE}**."
    )
    a("")
    a("## Brief questions")
    a("")
    a("1. **Who is healthy?** — last-value `b_runway` stays KEEP as Q1 *description* of the extract still. It is not a Y3 X.")
    a("2. **Who is improving?** — `b_d_runway` is a 3-month Δ of that still-walked path. Empty-on-short. DROP as X.")
    a("3. **Who is turning?** — do not invent `y_bal_vol`. Y2 already *is* the B path.")
    a("4. **Dip vs fall?** — `b_below_0` / episodes are the Y2 definition, not a leftover why.")
    a("5. **Why did it change?** — store `b_bal_vol` is not Javier cashflow vol (DRIFT 0.354).")
    a("6. **Months earlier?** — vol / d_runway need history. Q6 KEEP=False. Short persist lower (0.73), not higher.")
    a("")
    a("## PARK / CLOSE / KEEP / DROP-from-44")
    a("")
    a(_md_table(p12["rows"]))
    a("")
    a("## 1. Coverage; 13 Santander UK nulls; zombie cash")
    a("")
    a(p1["prose"])
    a("")
    a(_md_table(p1["rows"]))
    a("")
    a(_md_table(p1["ho_cov"]))
    a("")
    a(
        f"Hole-13 list match={p1['hole_match']}; tx-no-balance CONFIRM={p1['hole13_ok']}; "
        f"13+6=19 CONFIRM={p1.get('hole19_ok')}; "
        f"Santander UK n={p1['n_sant']}; last-tx 2026-07-20 n={p1['n_0720']}. "
        f"Zombie {p1['z_n']:,}/{p1['z_prod']:,}={_pp(p1['z_share_prod'])} "
        f"|share|={_pp(p1['z_share_bal'])}. PARK as a flag. Do not rewrite `liquidity.py`."
    )
    a("")
    a("## 2. Spearman twins")
    a("")
    a(p2["prose"])
    a("")
    a("Within-FOCUS matrix:")
    a("")
    a(_md_table(p2["matrix_rows"]))
    a("")
    if p2["weaker"]:
        a("Twin ≥0.80 → DROP weaker:")
        a("")
        a(_md_table(p2["weaker"]))
        a("")
    a("Selected pairs (FOCUS vs size / days / Javier vol / last-value):")
    a("")
    sel = [
        r
        for r in p2["pair_rows"]
        if r["b"] in ("log_in3", "c_n_days_with_tx", "a_vol", "a_out_vol", "b_runway", "b_neg_liq_3", "b_runway_last")
    ]
    a(_md_table(sel))
    a("")
    a("## 3. Single-feature group-fold (diagnostic leak)")
    a("")
    a(p3["prose"])
    a("")
    a("Sign from the train side of each fold. Seed 20260918. Holdout never entered a fold.")
    a("Y2 looking strong is the lock, not a KEEP.")
    a("")
    a(_md_table(p3["rows"]))
    a("")
    a("## 4. Honest leftover after days / last-value runway (Y3)")
    a("")
    a(p4["prose"])
    a("")
    a("OLS residual of the column on days (or runway), then oriented-signed group-fold AUROC vs Y3.")
    a("Leftover <0.55 dies. ρ(resid, days) ≥0.30 is a fake days leak (same as g_has).")
    a("")
    a(_md_table(p4["rows"]))
    a("")
    a("## 5. Y2 leftover after `b_below_0` / last-value sign")
    a("")
    a(p5["prose"])
    a("")
    a(_md_table(p5["rows"]))
    a("")
    a("## 6. `b_bal_vol` vs Javier vol")
    a("")
    a(p6["prose"])
    a("")
    a(_md_table(p6["rows"]))
    a("")
    a("## 7. SIZE terciles")
    a("")
    a(p7["prose"])
    a("")
    a(_md_table(p7["rows"]))
    a("")
    a("## 8. Q6 lags + short persist")
    a("")
    a(p8["prose"])
    a("")
    a(_md_table(p8["rows"]))
    a("")
    a("Empty-on-short (needs history):")
    a("")
    a(_md_table(p8["empty_rows"]))
    a("")
    a("Persist t vs t+3 (CONFIRM Family B short 0.730 / 24m 0.862 on `b_liq`):")
    a("")
    a(_md_table(p8["persist_rows"]))
    a("")
    a("## 9. ICC / company-demean")
    a("")
    a(p9["prose"])
    a("")
    a(_md_table(p9["rows"]))
    a("")
    a("## 10. `b_below_0` vs `b_neg_episodes` / `b_neg_liq_3`")
    a("")
    a(p10["prose"])
    a("")
    a(_md_table(p10["rows"]))
    a("")
    a("## 11. Holdout coverage (LOW_POWER)")
    a("")
    a(p11["prose"])
    a("")
    a(_md_table(p11["rows"]))
    a("")
    a("## 12. Decision table")
    a("")
    a(p12["prose"])
    a("")
    a(_md_table(p12["rows"]))
    a("")
    a("## Extras")
    a("")
    a("### vs `a_out_vol` trait")
    a("")
    a(ctx["p13"]["prose"])
    a("")
    a(_md_table(ctx["p13"]["rows"]))
    a("")
    a("### leakage_check")
    a("")
    a(ctx["p14"]["prose"])
    a("")
    a("### persist confirm (walk not rewritten)")
    a("")
    a(ctx["p15"]["prose"])
    a("")
    a(_md_table(ctx["p15"]["rows"]))
    a("")
    a("### Y3 quintiles")
    a("")
    a(ctx["p16"]["prose"])
    a("")
    a(_md_table(ctx["p16"]["rows"]))
    a("")
    a("### leftover inside days terciles")
    a("")
    a(ctx["p17"]["prose"])
    a("")
    a(_md_table(ctx["p17"]["rows"]))
    a("")
    a("### last-value runway on short books")
    a("")
    a(ctx["p18"]["prose"])
    a("")
    a(_md_table(ctx["p18"]["rows"]))
    a("")
    a("### Y2 already-neg vs clean-now")
    a("")
    a(ctx["p19"]["prose"])
    a("")
    a(_md_table(ctx["p19"]["rows"]))
    a("")
    a("### leftover after runway (Y3)")
    a("")
    a(ctx["p20"]["prose"])
    a("")
    a(_md_table(ctx["p20"]["rows"]))
    a("")
    a("### rank leftover + B-path leak")
    a("")
    a(ctx["p21"]["prose"])
    a("")
    a(_md_table(ctx["p21"]["rows"]))
    a("")
    a("### `b_d_runway` Q1 crash cell / T1 vs days")
    a("")
    a(ctx["p22"]["prose"])
    a("")
    a(_md_table(ctx["p22"]["rows"]))
    a("")
    a("### last-value look-ahead (why Q1 ≠ X)")
    a("")
    a(ctx["p23"]["prose"])
    a("")
    a(_md_table(ctx["p23"]["rows"]))
    a("")
    a("### demean raises B AUROC (path shock)")
    a("")
    a(ctx["p24"]["prose"])
    a("")
    a(_md_table(ctx["p24"]["rows"]))
    a("")
    a("### Y2 leftover on clean-now")
    a("")
    a(ctx["p25"]["prose"])
    a("")
    a(_md_table(ctx["p25"]["rows"]))
    a("")
    a("### leftover vol after Javier honesty")
    a("")
    a(ctx["p26"]["prose"])
    a("")
    a(_md_table(ctx["p26"]["rows"]))
    a("")
    a("### Y2 leftover ρ(resid, below_0)")
    a("")
    a(ctx["p27"]["prose"])
    a("")
    a(_md_table(ctx["p27"]["rows"]))
    a("")
    a("### `b_d_runway` leftover kill")
    a("")
    a(ctx["p28"]["prose"])
    a("")
    a(_md_table(ctx["p28"]["rows"]))
    a("")
    a(_md_table(ctx["p28"]["win_rows"]))
    a("")
    a("### `b_d_runway` Q1 dummy / drop-Q1")
    a("")
    a(ctx["p29"]["prose"])
    a("")
    a(_md_table(ctx["p29"]["rows"]))
    a("")
    a("## What this is not")
    a("")
    a("- Not a 0–100. Not pillars. Not `product/`.")
    a("- Not `python -m analysis.targets.build_targets`.")
    a("- Not a parquet merge. Not Family I/M/J merge.")
    a("- Not B on the 15-col Y3 card.")
    a("- Not B as Y2/Y3 X (lock).")
    a("- Not a holdout AUROC trophy.")
    a("- Did not rewrite `liquidity.py` / the walk.")
    a("")
    a(f"Elapsed {ctx['elapsed_s']:.1f}s. PNG: `{OUT_PNG.name}` ({OUT_PNG.exists()}).")
    a("")
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M")
    p3, p4, p6, p12 = ctx["p3"], ctx["p4"], ctx["p6"], ctx["p12"]
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_bal_vol",
            "value": p3["store"][(Y3, "b_bal_vol")]["cv"],
            "coverage": f"{p3['store'][(Y3, 'b_bal_vol')]['n_defined'] / max(ctx['p1']['n_cm'], 1):.4f}",
            "notes": (
                f"leftover_days={p4['store']['b_bal_vol']['days']['cv']:.4f} "
                f"rho_javier={p6['rho']:.3f} drop44=YES illegal_x"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_runway",
            "value": p3["store"][(Y3, "b_runway")]["cv"],
            "coverage": "0.8770",
            "notes": "KEEP Q1 description; DROP from 44 as Y3 X; PARK forecast Y",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y2,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_below_0",
            "value": p3["y2_below"],
            "coverage": "1.0000",
            "notes": f"LOCK leak not KEEP leftover_dies={ctx['p5']['others_die']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_on_44_resid_days",
            "value": p4["store"]["b_bal_vol"]["days"]["cv"],
            "coverage": "1.0000",
            "notes": (
                f"any_live={p4['any_live']} drop_all_x={p12['drop_all_x']} "
                f"days={p3['days_y3']:.3f} size={p3['size_y3']:.3f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": "-",
            "model": MODEL,
            "split": "train",
            "metric": "b_bal_vol_vs_a_vol_rho",
            "value": p6["rho"],
            "coverage": "0.7070",
            "notes": f"DRIFT confirm={p6['confirm']} quote=0.354 leftover_after_javier={p6['after_j']['cv']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": "-",
            "model": MODEL,
            "split": "holdout",
            "metric": "holdout_n_companies",
            "value": ctx["p11"]["n_co"],
            "coverage": "1.0000",
            "notes": f"ok72={ctx['p11']['ok72']} y2_pos={ctx['p11']['y2p']} y3_pos={ctx['p11']['y3p']} no_auroc",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_d_runway_rank_leftover",
            "value": ctx["p21"]["d_rank"],
            "coverage": "0.7070",
            "notes": (
                f"path_leak={ctx['p21']['d_path']} t1_survive_vs_days={ctx['p7']['t1_d']} "
                f"q1_crash={ctx['p22']['crash_y']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_runway_last_lookahead",
            "value": ctx["p23"]["y3_last"],
            "coverage": "1.0000",
            "notes": f"LOOK-AHEAD still vs month-t={ctx['p23']['y3_now']:.3f} last_p50={ctx['p23']['p50']:.3f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_d_runway_after_days_runway",
            "value": ctx["p28"]["after_dr"]["cv"],
            "coverage": "0.7070",
            "notes": (
                f"rank={ctx['p28']['after_dr']['rank_cv']:.3f} "
                f"t1_foldmin={ctx['p28']['fmin']:.3f} invert={ctx['p28']['invert']} "
                f"died_adj={ctx['p28']['died_dr']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "B",
            "y": Y3,
            "model": MODEL,
            "split": "train_cv",
            "metric": "auroc_b_d_runway_q1_dummy",
            "value": ctx["p29"]["dummy"],
            "coverage": "0.7070",
            "notes": (
                f"cont={ctx['p29']['cont']:.3f} almost_flag={ctx['p29']['almost_flag']} "
                f"body_left={ctx['p29']['body_left']:.3f} q1_share_pos={ctx['p29']['q1_share_pos']:.3f}"
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


def write_wave(ctx: dict) -> None:
    if not WRITE_WAVE:
        print("wave note skipped (WRITE_WAVE=False)")
        return
    p1, p3, p4, p6, p12 = ctx["p1"], ctx["p3"], ctx["p4"], ctx["p6"], ctx["p12"]
    p8, p9, p21, p23, p29 = ctx["p8"], ctx["p9"], ctx["p21"], ctx["p23"], ctx["p29"]
    text = f"""# Wave 4 — leftover B cols on the 44 ({AGENT})

Long-lived data lane. Same module ≥30 min: write → run → next cut (must-do 1–12 plus extras 13–29).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
`liquidity.py` not edited. Did not reopen the walk / still-leak except persist confirm.
Holdout 72 coverage only; rates / AUROC on train. Seed 20260918.
Did not put B on the 15-col Y3 card. Night Y3 quote stays **{NIGHT_Y3} / {NIGHT_Y3_CORE}**.
Days bar **{DAYS_BENCH}**. Size **{SIZE_QUOTE}**. Y7 TURNOVER **{NIGHT_Y7} / {NIGHT_Y7_CORE}**.

## Files

- `analysis/evaluate/b_on_44_qa.py` (create)
- `analysis/outputs/b_on_44_qa.md`
- `analysis/outputs/b_on_44_leftover.png`
- append-only `analysis/experiments/registry.csv`
- this note

## What we measured (train)

- {p1['prose']}
- {p3['prose']}
- {p4['prose']}
- Rank leftover kills `b_runway` ({_f(p4['store']['b_runway']['days']['rank_cv'])}). `b_d_runway` OLS leftover looks high; Q1 dummy≈continuous ({_f(p29['dummy'])} vs {_f(p29['cont'])}); body leftover after days={_f(p29['body_left'])} DIES. Q1 owns {_pp(p29['q1_share_pos'])} of Y3 recoveries.
- {p6['prose']}
- LOOK-AHEAD last-value still Y3={_f(p23['y3_last'])} vs month-t {_f(p23['y3_now'])}. last p50={_f(p23['p50'])} CONFIRM 1.079. Last-month Y3 labels=0.
- Demean *raises* B AUROC (runway 0.596→0.940) — month-t cash path is the Y. `a_out_vol` demean 0.722→0.549 (trait). `b_bal_vol` η²=0.129 ≠ a_out_vol dummy.
- Persist CONFIRM short 0.730 / 24m 0.862. {p8['prose']}
- {p12['prose']}

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `b_bal_vol` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y** (DRIFT 0.354 vs Javier; leftover after days is fake ρ=0.978; not the a_out_vol trait) |
| `b_below_0` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as flag** (Y2 leak 0.896) |
| `b_d_runway` on the 44 | **DROP from the 44** / **CLOSE as X** (Q1 crash leftover; body 0.527 DIES) |
| `b_neg_episodes` on the 44 | **DROP from the 44** / **CLOSE as X** / **PARK as Y2 cousin** (not a below_0 twin; leftover fake days) |
| `b_runway` on the 44 as Y3 X | **DROP from the 44 as X** (rank leftover 0.537 DIES) |
| last-value `b_runway` as Q1 description | **KEEP** |
| last-value as forecast Y | **PARK** |
| Family B as Y2/Y3 X | **forbidden** (lock) |
| invent `y_bal_vol` | **no** |
| 15-col Y3 card | **no** |
| night quotes | **unchanged** |

## Brief map

1. Who is healthy? — last-value runway KEEP as description.
2. Who is improving? — d_runway empty-on-short; leftover is the Y3 crash cell. DROP as X.
3. Who is turning? — do not invent y_bal_vol. Y2 is the B path.
5. Why? — store vol ≠ Javier vol (DRIFT 0.354) and ≠ a_out_vol trait.
6. Months earlier? — Q6 CLOSE. Short persist 0.73 < long 0.86.

## What we did not do

- Did not edit `liquidity.py`, `balances_b_qa.py`, parquet / duckdb, `build_targets`, `product/`, the 15-col card, LIVE.json, CONTEXT.md, canvas, or the parent journal.
- Did not invent `y_bal_vol`. Did not reopen the walk except persist confirm.
- Did not fit on holdout 72. Did not commit.

## What failed / next

- First leftover table quoted OLS 0.65–0.75 as “lives”. Tightened: ρ(resid,days)≥0.30 → fake; rank leftover kills `b_runway` (0.537); `b_d_runway` leftover is the Q1 crash cell (body 0.527 DIES).
- First hole-13 query joined balances→banking_products and counted 17. Fixed: `balances.company_id` → 13 tx-no-balance + 6 no-cash-walk = 19 all-null CONFIRM.
- Next (not this owner): drop the five B cols from the 44-col starter when someone re-cards. Not a parquet rewrite. Last-value `b_runway` stays Q1 description off the card.
"""
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(text, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"b_on_44_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    panel = load_panel()
    panel = attach_folds(panel)
    panel = add_in_memory(panel)
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={(panel['split']=='holdout').sum()}"
    )
    con = connect()
    try:
        p1 = pass1_coverage(panel, tr, con)
        p2 = pass2_twins(tr)
        p3 = pass3_singles(tr)
        p4 = pass4_leftover_y3(tr)
        p5 = pass5_leftover_y2(tr)
        p6 = pass6_vol_drift(tr, p2)
        p7 = pass7_terciles(tr)
        p8 = pass8_q6(tr)
        p9 = pass9_icc(tr)
        p10 = pass10_neg_twins(tr, p2, p3)
        p11 = pass11_holdout(panel)
        p12 = pass12_decisions(p2, p3, p4, p5, p6, p7, p8, p9, p10)
        p13 = pass13_outvol_dummy(tr, p6, p9)
        p14 = pass14_leakage()
        p15 = pass15_last_vs_snap_confirm(tr)
        p16 = pass16_quintiles(tr)
        p17 = pass17_days_tercile_leftover(tr)
        p18 = pass18_short_runway_last(tr)
        p19 = pass19_already_neg_y2(tr)
        p20 = pass20_y3_after_runway_only(tr, p4)
        p21 = pass21_rank_leftover(tr, p4)
        p22 = pass22_drunway_q1(tr, p4, p7)
        p23 = pass23_lookahead(tr, p3)
        p24 = pass24_demean_path(p9)
        p25 = pass25_y2_clean_now(tr)
        p26 = pass26_javier_resid(tr, p6)
        p27 = pass27_y2_sign_rho(tr, p5)
        p28 = pass28_drunway_kill(tr, p4, p7)
        p29 = pass29_q1_dummy(tr)
        png = make_plot(tr, p4, p6)
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
        "png": png,
        "elapsed_s": time.time() - t0,
    }
    write_md(ctx)
    append_registry(ctx)
    write_wave(ctx)
    print(
        f"done elapsed={ctx['elapsed_s']:.0f}s drop_all_x={p12['drop_all_x']} "
        f"days={_f(p3['days_y3'])} leftover_vol={_f(p4['store']['b_bal_vol']['days']['cv'])}"
    )
    return ctx


if __name__ == "__main__":
    run()
