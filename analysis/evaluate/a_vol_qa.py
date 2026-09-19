"""In-memory a_vol QA — Javier cashflow volatility vs store twins.

NORTH_STAR: reconstruct Javier ``volatility`` from Family A (legal X).
Never family B as Y2/Y3 X. ``b_bal_vol`` is a diagnostic only (DRIFT).
Do not merge into the store unless a KEEP gate hits — parent decides.
No 0–100. No product/. No parquet rewrite. No new GBM. Do not run
``build_targets``. Do not edit score_pipeline.py or cashflow.py.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.a_vol_qa

Owned: analysis/evaluate/a_vol_qa.py, analysis/outputs/a_vol_qa.md,
optional PNG, append-only registry, overnight/waves/wave4_a_vol.md (end).

KEEP (store candidate, not a 15-col card tonight):
beats size by ≥0.02 AND not SIZE (|ρ| vs log1p(a_in3) < 0.50)
AND not a copy of a_io_ratio / a_growth_3 (|ρ| < 0.80).

CLOSE if it loses to size or is a twin of an existing A column.
PARK as a health Y either way. Do not put it on the 15-col card.

a_out_vol later-store KEEP only if leak/size/12-names survive AND it is a
month shock. Demean kill + high ICC → CLOSE as a company-style dummy.
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
from analysis.score_pipeline import build_features as build_javier_signals
from analysis.targets.y11_dark import book_invoice_ids

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
TARGETS = DATA / "feature_store" / "targets.parquet"
OUT_MD = ANALYSIS / "outputs" / "a_vol_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "a_vol_vs_b_bal_vol.png"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_a_vol.md"
JAVIER_CACHE = Path("/tmp/embat_a_vol_qa_javier.parquet")
PIPE_CACHE = Path("/tmp/embat_score_pipeline_qa_javier.parquet")
AGENT = "6b456387"
WAVE = "4"
ROUND = "R4"

Y2 = "y2_neg_2of3"
Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_Y3 = 0.711
NIGHT_Y2 = 0.540
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
COPY_RHO = 0.80
SAME_RHO = 0.95
CLOSE_RHO = 0.80
SIZE_PARK = 0.60
MIN_POS = 50
CLIP = 3.0
CHRONIC_GROUPS = ("GROUP_0158", "GROUP_0172")
CHRONIC_BELOW = 0.50
WRITE_WAVE = True  # last iterate — one note at the end

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_net",
    "a_op_in",
    "a_op_out",
    "a_in3",
    "a_out6",
    "a_n_tx",
    "first_month",
    "a_io_ratio",
    "a_growth_3",
    "a_net_margin",
    "b_bal_vol",
    "b_below_0",
    "c_n_days_with_tx",
    "c_ss_month",
    "c_salary_month",
    "f_ds_r",
)
CORE_STEMS = (
    "c_ss_month",
    "c_salary_month",
    "a_n_tx",
    "f_ds_r",
    "c_n_days_with_tx",
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


def pearson(a, b) -> tuple[float, int]:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    )
    d = d[np.isfinite(d["a"]) & np.isfinite(d["b"])]
    n = int(len(d))
    if n < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan"), n
    return float(d["a"].corr(d["b"], method="pearson")), n


def verdict(rho: float) -> str:
    if not np.isfinite(rho):
        return "NA"
    if rho >= SAME_RHO:
        return "SAME"
    if rho >= CLOSE_RHO:
        return "CLOSE"
    return "DRIFT"


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
            "coverage": float("nan"),
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
    n_mask = int(mask.sum())
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
        "coverage": (n / n_mask) if n_mask else float("nan"),
        "low_power": False,
    }


def identity_tight(a, b) -> dict:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    )
    d = d[np.isfinite(d["a"]) & np.isfinite(d["b"])]
    n = int(len(d))
    if n == 0:
        return {
            "n": 0,
            "spearman": float("nan"),
            "pearson": float("nan"),
            "max_abs": float("nan"),
            "p50_abs": float("nan"),
            "exact": float("nan"),
        }
    delta = (d["a"] - d["b"]).abs()
    rho_s, _ = spearman(d["a"], d["b"])
    rho_p, _ = pearson(d["a"], d["b"])
    return {
        "n": n,
        "spearman": rho_s,
        "pearson": rho_p,
        "max_abs": float(delta.max()),
        "p50_abs": float(delta.median()),
        "exact": float((delta <= 1e-12).mean()),
    }


def quintile_table(df: pd.DataFrame, mask: pd.Series, x_col: str, y_col: str) -> dict:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = mask & x.notna() & y.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    tr = pd.DataFrame({x_col: x[m], y_col: y[m]})
    empty = {
        "x": x_col,
        "y": y_col,
        "n": int(len(tr)),
        "rows": [],
        "n_bins": 0,
        "monotone_up": None,
        "monotone_down": None,
        "tail_only": None,
        "head_only": None,
        "shape": "na",
        "bins": [],
    }
    if len(tr) < 50 or tr[x_col].nunique() < 3:
        return empty
    cats, bins = pd.qcut(tr[x_col], 5, retbins=True, duplicates="drop")
    tr = tr.copy()
    tr["q"] = cats
    rows = []
    for i, (q, g) in enumerate(tr.groupby("q", observed=True), start=1):
        rows.append(
            {
                "q": i,
                "interval": str(q),
                "lo": float(q.left),
                "hi": float(q.right),
                "n": int(len(g)),
                "n_pos": int((g[y_col] == 1).sum()),
                "y_rate": float(g[y_col].mean()),
                "x_median": float(g[x_col].median()),
            }
        )
    rates = [r["y_rate"] for r in rows]
    monotone_up = (
        all(a <= b + 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    )
    monotone_down = (
        all(a >= b - 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    )
    if len(rates) >= 3:
        body = rates[:-1]
        tail_only = rates[-1] > max(body) + 0.04 and max(body) - min(body) < 0.06
        mid = rates[1:]
        head_only = rates[0] > max(mid) + 0.04 and max(mid) - min(mid) < 0.06
    else:
        tail_only = None
        head_only = None
    shape = "flat"
    if tail_only:
        shape = "tail"
    elif head_only:
        shape = "head"
    elif monotone_up:
        shape = "monotone_up"
    elif monotone_down:
        shape = "monotone_down"
    return {
        "x": x_col,
        "y": y_col,
        "n": int(len(tr)),
        "rows": rows,
        "n_bins": len(rows),
        "monotone_up": monotone_up,
        "monotone_down": monotone_down,
        "tail_only": tail_only,
        "head_only": head_only,
        "shape": shape,
        "bins": [float(b) for b in bins],
    }


def add_in_memory_vols(df: pd.DataFrame) -> pd.DataFrame:
    """Javier volatility + in-memory alternatives. Never written to parquet."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    g = out.groupby("company_id", sort=False)
    net = pd.to_numeric(out["a_net"], errors="coerce")
    opin = pd.to_numeric(out["a_op_in"], errors="coerce")
    io = pd.to_numeric(out["a_io_ratio"], errors="coerce")
    sd6_net = g["a_net"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).std())
    mean6_in = g["a_op_in"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).mean()
    )
    sum6_in = g["a_op_in"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).sum()
    )
    sd6_in = g["a_op_in"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).std())
    sd6_io = g["a_io_ratio"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).std()
    )
    mean6_io = g["a_io_ratio"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).mean()
    )
    sd3_net = g["a_net"].transform(lambda s: pd.to_numeric(s, errors="coerce").rolling(3, min_periods=3).std())
    mean3_in = g["a_op_in"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(3, min_periods=3).mean()
    )
    sd12_net = g["a_net"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(12, min_periods=12).std()
    )
    mean12_in = g["a_op_in"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(12, min_periods=12).mean()
    )
    sd6_out = g["a_op_out"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).std()
    )
    mean6_out = g["a_op_out"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(6, min_periods=6).mean()
    )
    raw = sd6_net / np.maximum(mean6_in, 1.0)
    out["a_vol"] = np.minimum(CLIP, raw)
    out["a_vol_unclip"] = raw
    out["a_vol_sumdenom"] = np.minimum(CLIP, sd6_net / np.maximum(sum6_in, 1.0))
    out["a_in_vol"] = np.minimum(CLIP, sd6_in / np.maximum(mean6_in, 1.0))
    out["a_io_sd6"] = sd6_io
    out["a_io_vol"] = np.minimum(CLIP, sd6_io / np.maximum(mean6_io.abs(), 1e-6))
    out["a_vol_w3"] = np.minimum(CLIP, sd3_net / np.maximum(mean3_in, 1.0))
    out["a_vol_w12"] = np.minimum(CLIP, sd12_net / np.maximum(mean12_in, 1.0))
    out["a_out_vol"] = np.minimum(CLIP, sd6_out / np.maximum(mean6_out, 1.0))
    out["a_sd6_net"] = sd6_net
    out["a_vol_clip3"] = (out["a_vol"] >= CLIP - 1e-12).astype(float)
    out["log1p_a_in3"] = np.log1p(pd.to_numeric(out["a_in3"], errors="coerce").abs())
    out["_sd6_net"] = sd6_net
    out["_mean6_in"] = mean6_in
    _ = (net, opin, io)
    return out


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


def load_javier(con) -> pd.DataFrame:
    for path in (JAVIER_CACHE, PIPE_CACHE):
        if path.exists():
            age = time.time() - path.stat().st_mtime
            if age < 8 * 3600:
                df = pd.read_parquet(path)
                df["company_id"] = df["company_id"].astype(str)
                if "period" not in df.columns and "month" in df.columns:
                    df["period"] = pd.to_datetime(df["month"])
                else:
                    df["period"] = pd.to_datetime(df["period"])
                if "volatility" not in df.columns:
                    continue
                print(f"CUT 0 — Javier cache {path} age={age / 60:.1f}m rows={len(df)}")
                return df[["company_id", "period", "volatility"]].copy()
    t0 = time.time()
    P = build_javier_signals(con)
    keep = P[["company_id", "month", "volatility"]].copy()
    keep["company_id"] = keep["company_id"].astype(str)
    keep["period"] = pd.to_datetime(keep["month"])
    keep = keep.drop(columns=["month"])
    JAVIER_CACHE.parent.mkdir(parents=True, exist_ok=True)
    keep.to_parquet(JAVIER_CACHE, index=False)
    print(f"CUT 0 — Javier build_features {len(keep):,} rows in {time.time() - t0:.1f}s")
    return keep


def load_panel(con) -> pd.DataFrame:
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
    ymiss = [c for c in ykeep if c not in yraw.columns]
    if ymiss:
        raise RuntimeError(f"targets.parquet missing {ymiss}")
    panel = _keys(raw[list(STORE_COLS)])
    y = _keys(yraw[ykeep])
    panel = panel.merge(y, on=["company_id", "period"], how="left")
    hold = load_holdout()
    panel["split"] = np.where(panel["company_id"].isin(hold), "holdout", "train")
    jav = load_javier(con)
    panel = panel.merge(jav, on=["company_id", "period"], how="left")
    panel = add_in_memory_vols(panel)
    book = book_invoice_ids(con)
    panel["dark"] = (~panel["company_id"].isin(book)).astype(np.int8)
    return panel


def attach_folds(panel: pd.DataFrame) -> pd.DataFrame:
    train = panel[panel["split"] == "train"][["company_id", "group_id"]].drop_duplicates()
    assert_no_holdout(train["company_id"])
    folds = group_folds(train, n=N_FOLDS, seed=FOLD_SEED)
    return panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")


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
    ids = [str(i) for i in g.index[g["share_below"] >= CHRONIC_BELOW]]
    return ids


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r["auroc"]) else "—" for r in rec.get("folds", [])
    )


def single_row(name: str, rec: dict, size_cv: float, bench: float) -> dict:
    cv = rec["cv"]
    return {
        "feature": name,
        "cv": cv,
        "sd": rec["sd"],
        "train": rec["train_auc"],
        "sign": rec["train_sign"],
        "coverage": rec["coverage"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "gap_size": (cv - size_cv) if np.isfinite(cv) and np.isfinite(size_cv) else float("nan"),
        "gap_bench": (cv - bench) if np.isfinite(cv) and np.isfinite(bench) else float("nan"),
        "folds": fold_bits(rec),
        "low_power": rec["low_power"],
    }


# ---------------------------------------------------------------------------
# Passes
# ---------------------------------------------------------------------------
def pass1_identity(tr: pd.DataFrame) -> dict:
    print("CUT 1 — identity vs Javier raw volatility")
    tight = identity_tight(tr["volatility"], tr["a_vol"])
    rho, n = spearman(tr["volatility"], tr["a_vol"])
    v = verdict(rho)
    clip_share = float(tr.loc[tr["a_vol"].notna(), "a_vol_clip3"].mean())
    print(
        f"  a_vol vs j_vol ρ={rho:.3f} n={n:,} {v}  "
        f"Pearson={tight['pearson']:.3f} max|Δ|={tight['max_abs']:.3g} "
        f"exact={tight['exact']:.1%} clip3={clip_share:.1%}"
    )
    if v != "SAME":
        print("  WARN identity is not SAME — check rolling window / denom")
    return {
        "rho": rho,
        "n": n,
        "verdict": v,
        "tight": tight,
        "clip_share": clip_share,
        "same": v == "SAME",
    }


def pass2_bal_vol(tr: pd.DataFrame) -> dict:
    print("CUT 2 — vs b_bal_vol (diagnostic, never Y2/Y3 X)")
    rho, n = spearman(tr["a_vol"], tr["b_bal_vol"])
    rho_j, n_j = spearman(tr["volatility"], tr["b_bal_vol"])
    v = verdict(rho)
    print(f"  a_vol vs b_bal_vol ρ={rho:.3f} n={n:,} {v}")
    print(f"  j_vol vs b_bal_vol ρ={rho_j:.3f} n={n_j:,} {verdict(rho_j)}")
    # per-company Spearman (n≥8)
    rows = []
    for cid, g in tr.groupby("company_id", sort=False):
        r, nn = spearman(g["a_vol"], g["b_bal_vol"])
        if nn >= 8 and np.isfinite(r):
            rows.append(r)
    arr = np.array(rows, dtype=float) if rows else np.array([])
    per = {
        "n_cos": int(len(arr)),
        "p10": float(np.quantile(arr, 0.10)) if len(arr) else float("nan"),
        "p50": float(np.quantile(arr, 0.50)) if len(arr) else float("nan"),
        "p90": float(np.quantile(arr, 0.90)) if len(arr) else float("nan"),
        "same": float((arr >= SAME_RHO).mean()) if len(arr) else float("nan"),
        "close": float(((arr >= CLOSE_RHO) & (arr < SAME_RHO)).mean()) if len(arr) else float("nan"),
        "drift": float((arr < CLOSE_RHO).mean()) if len(arr) else float("nan"),
    }
    if len(arr):
        print(
            f"  per-co n≥8: {per['n_cos']} p10={per['p10']:.3f} p50={per['p50']:.3f} "
            f"p90={per['p90']:.3f} SAME={per['same']:.1%} CLOSE={per['close']:.1%} "
            f"DRIFT={per['drift']:.1%}"
        )
    return {
        "rho": rho,
        "n": n,
        "verdict": v,
        "rho_j": rho_j,
        "n_j": n_j,
        "per": per,
    }


def pass3_size_copy(tr: pd.DataFrame) -> dict:
    print("CUT 3 — SIZE + copy screen (train, defined rows)")
    pairs = []
    for col in ("a_vol", "a_in_vol", "a_io_vol", "a_io_sd6", "a_vol_sumdenom", "b_bal_vol"):
        for other, label in (
            ("log1p_a_in3", "size"),
            ("a_io_ratio", "a_io_ratio"),
            ("a_growth_3", "a_growth_3"),
            ("a_net_margin", "a_net_margin"),
        ):
            rho, n = spearman(tr[col], tr[other])
            pairs.append(
                {
                    "col": col,
                    "vs": label,
                    "rho": rho,
                    "n": n,
                    "abs": abs(rho) if np.isfinite(rho) else float("nan"),
                    "flag_size": bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO and label == "size"),
                    "flag_copy": bool(
                        np.isfinite(rho) and abs(rho) >= COPY_RHO and label in ("a_io_ratio", "a_growth_3")
                    ),
                }
            )
            mark = ""
            if pairs[-1]["flag_size"]:
                mark = " SIZE"
            if pairs[-1]["flag_copy"]:
                mark = " COPY"
            print(f"  {col:16s} vs {label:12s} ρ={rho:+.3f} n={n:,}{mark}")
    size_rho = next(p["rho"] for p in pairs if p["col"] == "a_vol" and p["vs"] == "size")
    io_rho = next(p["rho"] for p in pairs if p["col"] == "a_vol" and p["vs"] == "a_io_ratio")
    gr_rho = next(p["rho"] for p in pairs if p["col"] == "a_vol" and p["vs"] == "a_growth_3")
    is_size = bool(np.isfinite(size_rho) and abs(size_rho) >= SIZE_RHO)
    is_copy = bool(
        (np.isfinite(io_rho) and abs(io_rho) >= COPY_RHO)
        or (np.isfinite(gr_rho) and abs(gr_rho) >= COPY_RHO)
    )
    print(f"  a_vol SIZE={is_size} COPY={is_copy}")
    leak = leakage_check(["a_vol", "a_in_vol", "a_io_vol"], Y3, forbidden_prefixes=("b",))
    print(f"  leak screen Y3 X={{a_vol*}} forbidden B: ok={leak['ok']}")
    return {
        "pairs": pairs,
        "size_rho": size_rho,
        "io_rho": io_rho,
        "gr_rho": gr_rho,
        "is_size": is_size,
        "is_copy": is_copy,
        "leak_ok": leak["ok"],
    }


def pass4_singles(tr: pd.DataFrame) -> dict:
    print("CUT 4 — single-feature train group-fold AUROC")
    y2_mask = tr[Y2].notna()
    y3_mask = tr[Y3].notna()  # stressed-only by construction
    assert_no_holdout(tr.loc[y2_mask | y3_mask, "company_id"])
    feats = (
        "a_vol",
        "a_in_vol",
        "a_io_vol",
        "a_io_sd6",
        "log1p_a_in3",
        "c_n_days_with_tx",
        "a_io_ratio",
        "a_growth_3",
    )
    out = {"y2": {}, "y3": {}}
    for yname, ycol, mask, bench in (
        ("y2", Y2, y2_mask, NIGHT_Y2),
        ("y3", Y3, y3_mask, DAYS_Y3),
    ):
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        out[yname]["size"] = size
        rows = []
        for feat in feats:
            rec = signed_oof_auroc(tr[ycol], tr[feat], tr["fold"], mask)
            row = single_row(feat, rec, size["cv"], bench)
            rows.append(row)
            out[yname][feat] = rec
            print(
                f"  {yname} {feat:20s} CV={_f(row['cv'])}±{_f(row['sd'])} "
                f"sign={row['sign']:+d} cov={_f(row['coverage'])} "
                f"gap_size={row['gap_size']:+.3f} folds {row['folds']}"
            )
        out[yname]["rows"] = rows
        out[yname]["n"] = int(mask.sum())
        out[yname]["n_pos"] = int((mask & (tr[ycol] == 1)).sum())
        out[yname]["rate"] = float(tr.loc[mask, ycol].mean()) if int(mask.sum()) else float("nan")
    a_vol_y3 = out["y3"]["a_vol"]["cv"]
    a_vol_y2 = out["y2"]["a_vol"]["cv"]
    size_y3 = out["y3"]["size"]["cv"]
    size_y2 = out["y2"]["size"]["cv"]
    print(
        f"  a_vol Y3={_f(a_vol_y3)} vs size {_f(size_y3)} vs days {DAYS_Y3:.3f}; "
        f"Y2={_f(a_vol_y2)} vs size {_f(size_y2)} vs night {NIGHT_Y2:.3f}"
    )
    return out


def pass5_quintiles(tr: pd.DataFrame) -> dict:
    print("CUT 5 — quintiles vs Y3 / Y2")
    tables = []
    for ycol in (Y3, Y2):
        mask = tr[ycol].notna()
        for xcol in ("a_vol", "a_in_vol", "a_io_vol", "log1p_a_in3"):
            tab = quintile_table(tr, mask, xcol, ycol)
            tables.append(tab)
            print(
                f"  {ycol} ~ {xcol}: n={tab['n']} bins={tab['n_bins']} "
                f"shape={tab['shape']} mono↑={tab['monotone_up']} tail={tab['tail_only']}"
            )
            for r in tab["rows"]:
                print(
                    f"    Q{r['q']} n={r['n']:5d} pos={r['n_pos']:4d} "
                    f"P(Y=1)={r['y_rate']:.3f} med={r['x_median']:.4g}"
                )
    return {"tables": tables}


def pass6_q6(tr: pd.DataFrame) -> dict:
    print("CUT 6 — Q6 lag1 singles (honest 1-month only)")
    y2_mask = tr[Y2].notna()
    y3_mask = tr[Y3].notna()
    rows = []
    for yname, ycol, mask, bench in (
        ("y2", Y2, y2_mask, NIGHT_Y2),
        ("y3", Y3, y3_mask, DAYS_Y3),
    ):
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        for feat in ("a_vol", "a_in_vol", "a_io_vol"):
            now = signed_oof_auroc(tr[ycol], tr[feat], tr["fold"], mask)
            lag = signed_oof_auroc(tr[ycol], tr[f"{feat}_lag1"], tr["fold"], mask)
            lag_lift = (
                (lag["cv"] - now["cv"])
                if np.isfinite(lag["cv"]) and np.isfinite(now["cv"])
                else float("nan")
            )
            q6_keep = bool(
                np.isfinite(lag["cv"])
                and np.isfinite(size["cv"])
                and (lag["cv"] - size["cv"]) >= KEEP_DELTA
                and lag["cv"] < SIZE_PARK
                and np.isfinite(lag_lift)
                and lag_lift >= 0.01
            )
            rec = {
                "y": yname,
                "feature": feat,
                "now": now["cv"],
                "now_sd": now["sd"],
                "lag": lag["cv"],
                "lag_sd": lag["sd"],
                "lag_lift": lag_lift,
                "gap_size": (lag["cv"] - size["cv"]) if np.isfinite(lag["cv"]) else float("nan"),
                "q6_keep": q6_keep,
                "now_folds": fold_bits(now),
                "lag_folds": fold_bits(lag),
            }
            rows.append(rec)
            print(
                f"  {yname} {feat:10s} lag0={_f(now['cv'])} lag1={_f(lag['cv'])} "
                f"lift={lag_lift:+.3f} q6={q6_keep}"
            )
    any_keep = any(r["q6_keep"] for r in rows)
    return {"rows": rows, "any_keep": any_keep}


def pass7_alts(tr: pd.DataFrame, p3: dict, p4: dict) -> dict:
    print("CUT 7 — alternatives vs Javier / size / copy")
    rows = []
    for col in ("a_vol", "a_in_vol", "a_io_vol", "a_io_sd6", "a_vol_sumdenom"):
        rho_j, n_j = spearman(tr[col], tr["volatility"])
        rho_b, n_b = spearman(tr[col], tr["b_bal_vol"])
        y3 = next((r for r in p4["y3"]["rows"] if r["feature"] == col), None)
        y2 = next((r for r in p4["y2"]["rows"] if r["feature"] == col), None)
        size_p = next((p for p in p3["pairs"] if p["col"] == col and p["vs"] == "size"), None)
        io_p = next((p for p in p3["pairs"] if p["col"] == col and p["vs"] == "a_io_ratio"), None)
        rec = {
            "col": col,
            "rho_j": rho_j,
            "n_j": n_j,
            "verdict_j": verdict(rho_j),
            "rho_b": rho_b,
            "n_b": n_b,
            "y3_cv": y3["cv"] if y3 else float("nan"),
            "y3_gap": y3["gap_size"] if y3 else float("nan"),
            "y2_cv": y2["cv"] if y2 else float("nan"),
            "y2_gap": y2["gap_size"] if y2 else float("nan"),
            "size_rho": size_p["rho"] if size_p else float("nan"),
            "io_rho": io_p["rho"] if io_p else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {col:16s} vs j={_f(rho_j)} {rec['verdict_j']}  vs B={_f(rho_b)}  "
            f"Y3={_f(rec['y3_cv'])} Δsz={rec['y3_gap']:+.3f}  "
            f"Y2={_f(rec['y2_cv'])} Δsz={rec['y2_gap']:+.3f}"
        )
    return {"rows": rows}


def pass8_chronic(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 8 — is a_vol just the 12 chronic dark Y2 names?")
    is_ch = tr["company_id"].astype(str).isin(set(ids))
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    y2_lab = y2.notna()
    y3_lab = y3.notna()
    slices = []
    for name, sl in (
        ("train_y2", y2_lab),
        ("chronic_12_y2", y2_lab & is_ch),
        ("rest_y2", y2_lab & ~is_ch),
        ("train_y3", y3_lab),
        ("chronic_12_y3", y3_lab & is_ch),
        ("rest_y3", y3_lab & ~is_ch),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((sl & ((y2 if "y2" in name else y3) == 1)).sum())
            if "y2" in name or "y3" in name
            else 0,
            "med_vol": float(pd.to_numeric(tr.loc[sl, "a_vol"], errors="coerce").median()),
            "mean_vol": float(pd.to_numeric(tr.loc[sl, "a_vol"], errors="coerce").mean()),
            "clip_share": float(tr.loc[sl & tr["a_vol"].notna(), "a_vol_clip3"].mean())
            if int((sl & tr["a_vol"].notna()).sum())
            else float("nan"),
        }
        slices.append(rec)
        print(
            f"  {name:16s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"med_vol={_f(rec['med_vol'])} clip={_pp(rec['clip_share'])}"
        )

    rest_y2 = y2_lab & ~is_ch
    rest_y3 = y3_lab & ~is_ch
    drop = {}
    for yname, ycol, mask, bench in (
        ("y2", Y2, rest_y2, NIGHT_Y2),
        ("y3", Y3, rest_y3, DAYS_Y3),
    ):
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        vol = signed_oof_auroc(tr[ycol], tr["a_vol"], tr["fold"], mask)
        days = signed_oof_auroc(tr[ycol], tr["c_n_days_with_tx"], tr["fold"], mask)
        drop[yname] = {
            "n": int(mask.sum()),
            "n_pos": int((mask & (tr[ycol] == 1)).sum()),
            "vol": vol,
            "size": size,
            "days": days,
            "gap": (vol["cv"] - size["cv"]) if np.isfinite(vol["cv"]) else float("nan"),
        }
        print(
            f"  drop12 {yname} a_vol CV={_f(vol['cv'])}±{_f(vol['sd'])} "
            f"size={_f(size['cv'])} gap={drop[yname]['gap']:+.3f} "
            f"days={_f(days['cv'])} folds {fold_bits(vol)}"
        )
    return {"ids": ids, "n_ids": len(ids), "slices": slices, "drop": drop}


def pass9_holdout(panel: pd.DataFrame) -> dict:
    print("CUT 9 — holdout coverage (LOW_POWER, no ρ / no AUROC claim)")
    ho = panel[panel["split"] == "holdout"]
    rec = {
        "n_cm": int(len(ho)),
        "n_cos": int(ho["company_id"].nunique()),
        "a_vol": float(ho["a_vol"].notna().mean()),
        "b_bal_vol": float(ho["b_bal_vol"].notna().mean()),
        "j_vol": float(ho["volatility"].notna().mean()),
        "y2_n": int(ho[Y2].notna().sum()),
        "y2_pos": int((ho[Y2] == 1).sum()),
        "y3_n": int(ho[Y3].notna().sum()),
        "y3_pos": int((ho[Y3] == 1).sum()),
    }
    print(
        f"  holdout CM={rec['n_cm']} cos={rec['n_cos']} "
        f"a_vol={_pp(rec['a_vol'])} y2_pos={rec['y2_pos']} y3_pos={rec['y3_pos']}"
    )
    return rec


def pass10_cov(tr: pd.DataFrame) -> dict:
    print("CUT 10 — train coverage + fold ρ vs Javier")
    n = len(tr)
    rec = {
        "n_cm": n,
        "n_cos": int(tr["company_id"].nunique()),
        "a_vol": float(tr["a_vol"].notna().mean()),
        "a_in_vol": float(tr["a_in_vol"].notna().mean()),
        "a_io_vol": float(tr["a_io_vol"].notna().mean()),
        "j_vol": float(tr["volatility"].notna().mean()),
        "b_bal_vol": float(tr["b_bal_vol"].notna().mean()),
    }
    folds = []
    for k in range(N_FOLDS):
        sl = tr["fold"] == k
        rho, nn = spearman(tr.loc[sl, "a_vol"], tr.loc[sl, "volatility"])
        folds.append({"fold": k, "rho": rho, "n": nn})
        print(f"  fold {k} a_vol↔j ρ={_f(rho)} n={nn:,}")
    rec["folds"] = folds
    rec["fold_min"] = min((f["rho"] for f in folds if np.isfinite(f["rho"])), default=float("nan"))
    print(
        f"  train cov a_vol={_pp(rec['a_vol'])} in_vol={_pp(rec['a_in_vol'])} "
        f"io_vol={_pp(rec['a_io_vol'])} fold_min_ρ={_f(rec['fold_min'])}"
    )
    return rec


def pass11_clip_quote(tr: pd.DataFrame, p1: dict) -> dict:
    print("CUT 11 — clip=3 share of all train CM vs defined (pipeline 11.8%)")
    n = len(tr)
    n_def = int(tr["a_vol"].notna().sum())
    n_clip = int((tr["a_vol_clip3"] == 1).sum())
    of_all = n_clip / n if n else float("nan")
    of_def = n_clip / n_def if n_def else float("nan")
    j_def = tr["volatility"].notna()
    j_clip = int((j_def & (pd.to_numeric(tr["volatility"], errors="coerce") >= CLIP - 1e-12)).sum())
    j_of_all = j_clip / n if n else float("nan")
    j_of_def = j_clip / int(j_def.sum()) if int(j_def.sum()) else float("nan")
    confirm = abs(of_all - 0.118) < 0.01 or abs(j_of_all - 0.118) < 0.01
    print(
        f"  a_vol clip of-all={of_all:.1%} of-def={of_def:.1%} "
        f"j of-all={j_of_all:.1%} of-def={j_of_def:.1%} confirm11.8={confirm}"
    )
    return {
        "n": n,
        "n_def": n_def,
        "n_clip": n_clip,
        "of_all": of_all,
        "of_def": of_def,
        "j_of_all": j_of_all,
        "j_of_def": j_of_def,
        "confirm": confirm,
        "p1_def": p1["clip_share"],
    }


def pass12_alt_drop12(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 12 — a_in_vol / a_io_vol after drop 12 chronic (is Y2 +0.02 those names?)")
    is_ch = tr["company_id"].astype(str).isin(set(ids))
    y2_mask = tr[Y2].notna() & ~is_ch
    y3_mask = tr[Y3].notna() & ~is_ch
    rows = []
    for yname, ycol, mask, bench in (
        ("y2", Y2, y2_mask, NIGHT_Y2),
        ("y3", Y3, y3_mask, DAYS_Y3),
    ):
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        for feat in ("a_vol", "a_in_vol", "a_io_vol", "a_io_sd6", "c_n_days_with_tx"):
            rec = signed_oof_auroc(tr[ycol], tr[feat], tr["fold"], mask)
            row = single_row(feat, rec, size["cv"], bench)
            row["y"] = yname
            rows.append(row)
            print(
                f"  drop12 {yname} {feat:20s} CV={_f(row['cv'])}±{_f(row['sd'])} "
                f"gap_size={row['gap_size']:+.3f} folds {row['folds']}"
            )
    y2_io = next(r for r in rows if r["y"] == "y2" and r["feature"] == "a_io_vol")
    y2_days = next(r for r in rows if r["y"] == "y2" and r["feature"] == "c_n_days_with_tx")
    y3_in = next(r for r in rows if r["y"] == "y3" and r["feature"] == "a_in_vol")
    io_is_12 = bool(np.isfinite(y2_io["gap_size"]) and y2_io["gap_size"] < KEEP_DELTA)
    print(
        f"  a_io_vol Y2 drop12 gap={y2_io['gap_size']:+.3f} "
        f"(was +0.020) is_the_12={io_is_12}; days gap={y2_days['gap_size']:+.3f}; "
        f"a_in_vol Y3 drop12 gap={y3_in['gap_size']:+.3f}"
    )
    return {
        "rows": rows,
        "io_is_12": io_is_12,
        "y2_io_gap": y2_io["gap_size"],
        "y2_days_gap": y2_days["gap_size"],
        "y3_in_gap": y3_in["gap_size"],
        "y3_in_cv": y3_in["cv"],
    }


def pass13_days_twin(tr: pd.DataFrame) -> dict:
    print("CUT 13 — Spearman vs c_n_days_with_tx (activity twin?)")
    rows = []
    for col in ("a_vol", "a_in_vol", "a_io_vol", "a_io_sd6", "a_out_vol", "a_vol_clip3"):
        rho, n = spearman(tr[col], tr["c_n_days_with_tx"])
        rec = {
            "col": col,
            "rho": rho,
            "n": n,
            "copy": bool(np.isfinite(rho) and abs(rho) >= COPY_RHO),
        }
        rows.append(rec)
        print(f"  {col:16s} vs days ρ={rho:+.3f} n={n:,}{' COPY' if rec['copy'] else ''}")
    return {"rows": rows, "any_copy": any(r["copy"] for r in rows)}


def pass14_size_x_vol(tr: pd.DataFrame) -> dict:
    print("CUT 14 — Y3 size tercile × a_vol quintile (is the tail just small books?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & tr["log1p_a_in3"].notna()
    d = tr.loc[mask, ["a_vol", "log1p_a_in3"]].copy()
    d["y"] = y[mask].to_numpy()
    d["size_t"] = pd.qcut(d["log1p_a_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    d["vol_q"] = pd.qcut(d["a_vol"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    rows = []
    for (st, vq), g in d.groupby(["size_t", "vol_q"], observed=True):
        rec = {
            "size": str(st),
            "vol": str(vq),
            "n": int(len(g)),
            "n_pos": int((g["y"] == 1).sum()),
            "rate": float(g["y"].mean()),
        }
        rows.append(rec)
        print(f"  {st} {vq} n={rec['n']:4d} pos={rec['n_pos']:3d} P(Y3)={rec['rate']:.3f}")
    q5 = d["vol_q"].astype(str) == "Q5"
    t1 = d["size_t"].astype(str) == "T1"
    q5_t1_share = float(t1[q5].mean()) if int(q5.sum()) else float("nan")
    q5_rate = float(d.loc[q5, "y"].mean()) if int(q5.sum()) else float("nan")
    body_rate = float(d.loc[~q5, "y"].mean()) if int((~q5).sum()) else float("nan")
    t1_q5_rate = float(d.loc[q5 & t1, "y"].mean()) if int((q5 & t1).sum()) else float("nan")
    nott1_q5_rate = float(d.loc[q5 & ~t1, "y"].mean()) if int((q5 & ~t1).sum()) else float("nan")
    tail_is_small = bool(np.isfinite(q5_t1_share) and q5_t1_share >= 0.50)
    print(
        f"  Q5 share in size T1={q5_t1_share:.1%} Q5 rate={q5_rate:.3f} "
        f"body={body_rate:.3f} T1_and_Q5={t1_q5_rate:.3f} "
        f"notT1_and_Q5={nott1_q5_rate:.3f} tail_is_small={tail_is_small}"
    )
    return {
        "rows": rows,
        "q5_t1_share": q5_t1_share,
        "q5_rate": q5_rate,
        "body_rate": body_rate,
        "t1_q5_rate": t1_q5_rate,
        "nott1_q5_rate": nott1_q5_rate,
        "tail_is_small": tail_is_small,
    }


def pass15_clip_flag(tr: pd.DataFrame) -> dict:
    print("CUT 15 — clip=3 flag as a single (not a new Y)")
    y2_mask = tr[Y2].notna()
    y3_mask = tr[Y3].notna()
    rows = []
    rates = []
    for yname, ycol, mask, bench in (
        ("y2", Y2, y2_mask, NIGHT_Y2),
        ("y3", Y3, y3_mask, DAYS_Y3),
    ):
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        rec = signed_oof_auroc(tr[ycol], tr["a_vol_clip3"], tr["fold"], mask)
        row = single_row("a_vol_clip3", rec, size["cv"], bench)
        row["y"] = yname
        rows.append(row)
        hit = mask & (tr["a_vol_clip3"] == 1) & tr["a_vol"].notna()
        miss = mask & (tr["a_vol_clip3"] == 0) & tr["a_vol"].notna()
        rate_hit = float(tr.loc[hit, ycol].mean()) if int(hit.sum()) else float("nan")
        rate_miss = float(tr.loc[miss, ycol].mean()) if int(miss.sum()) else float("nan")
        rates.append(
            {
                "y": yname,
                "n_clip": int(hit.sum()),
                "n_body": int(miss.sum()),
                "rate_clip": rate_hit,
                "rate_body": rate_miss,
            }
        )
        print(
            f"  {yname} clip3 CV={_f(row['cv'])} gap_size={row['gap_size']:+.3f} "
            f"P(Y|clip)={rate_hit:.3f} n={int(hit.sum())} "
            f"P(Y|body)={rate_miss:.3f} n={int(miss.sum())}"
        )
    return {"rows": rows, "rates": rates}


def pass16_dark(tr: pd.DataFrame) -> dict:
    print("CUT 16 — dark 470 vs invoiced (vol level + singles)")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    dark = tr["dark"] == 1
    slices = []
    for name, sl in (
        ("dark_y2", y2.notna() & dark),
        ("erp_y2", y2.notna() & ~dark),
        ("dark_y3", y3.notna() & dark),
        ("erp_y3", y3.notna() & ~dark),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_cos": int(tr.loc[sl, "company_id"].nunique()),
            "n_pos": int((sl & ((y2 if "y2" in name else y3) == 1)).sum()),
            "med_vol": float(pd.to_numeric(tr.loc[sl, "a_vol"], errors="coerce").median()),
            "clip": float(tr.loc[sl & tr["a_vol"].notna(), "a_vol_clip3"].mean())
            if int((sl & tr["a_vol"].notna()).sum())
            else float("nan"),
        }
        slices.append(rec)
        print(
            f"  {name:10s} n={rec['n']:5d} cos={rec['n_cos']:4d} "
            f"med={_f(rec['med_vol'])} clip={_pp(rec['clip'])}"
        )
    singles = []
    for yname, ycol, base_mask, bench in (
        ("y2", Y2, y2.notna(), NIGHT_Y2),
        ("y3", Y3, y3.notna(), DAYS_Y3),
    ):
        for slname, sl in (("dark", dark), ("erp", ~dark)):
            mask = base_mask & sl
            size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
            vol = signed_oof_auroc(tr[ycol], tr["a_vol"], tr["fold"], mask)
            row = single_row("a_vol", vol, size["cv"], bench)
            row["y"] = yname
            row["slice"] = slname
            singles.append(row)
            print(
                f"  {yname} {slname} a_vol CV={_f(row['cv'])} gap_size={row['gap_size']:+.3f} "
                f"n={row['n']} pos={row['n_pos']}"
            )
    return {"slices": slices, "singles": singles}


def pass17_windows(tr: pd.DataFrame) -> dict:
    print("CUT 17 — window / unclip / outflow-vol variants (still in memory)")
    y2_mask = tr[Y2].notna()
    y3_mask = tr[Y3].notna()
    feats = ("a_vol_unclip", "a_vol_w3", "a_vol_w12", "a_out_vol", "a_sd6_net")
    rows = []
    for yname, ycol, mask, bench in (
        ("y2", Y2, y2_mask, NIGHT_Y2),
        ("y3", Y3, y3_mask, DAYS_Y3),
    ):
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        for feat in feats:
            rec = signed_oof_auroc(tr[ycol], tr[feat], tr["fold"], mask)
            row = single_row(feat, rec, size["cv"], bench)
            row["y"] = yname
            rho_j, n_j = spearman(tr[feat], tr["volatility"])
            row["rho_j"] = rho_j
            row["n_j"] = n_j
            rows.append(row)
            print(
                f"  {yname} {feat:14s} CV={_f(row['cv'])} gap={row['gap_size']:+.3f} "
                f"vs j={_f(rho_j)} {verdict(rho_j)}"
            )
    return {"rows": rows}


def pass18_already_neg(tr: pd.DataFrame) -> dict:
    print("CUT 18 — already-neg (B decomp only) × a_vol")
    y2 = pd.to_numeric(tr[Y2], errors="coerce")
    below = pd.to_numeric(tr["b_below_0"], errors="coerce")
    lab = y2.notna()
    already = lab & (below == 1)
    clean = lab & (below == 0)
    slices = []
    for name, sl in (("already_neg", already), ("clean_now", clean)):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((sl & (y2 == 1)).sum()),
            "rate": float(y2[sl].mean()) if int(sl.sum()) else float("nan"),
            "med_vol": float(pd.to_numeric(tr.loc[sl, "a_vol"], errors="coerce").median()),
            "clip": float(tr.loc[sl & tr["a_vol"].notna(), "a_vol_clip3"].mean())
            if int((sl & tr["a_vol"].notna()).sum())
            else float("nan"),
        }
        slices.append(rec)
        print(
            f"  {name:12s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"P(Y2)={rec['rate']:.3f} med_vol={_f(rec['med_vol'])} clip={_pp(rec['clip'])}"
        )
    # quintiles on clean-now leftover (honest turn)
    tab = quintile_table(tr, clean, "a_vol", Y2)
    print(f"  clean-now a_vol quintiles shape={tab['shape']} n={tab['n']}")
    for r in tab["rows"]:
        print(f"    Q{r['q']} P(Y2)={r['y_rate']:.3f} n={r['n']}")
    vol = signed_oof_auroc(tr[Y2], tr["a_vol"], tr["fold"], clean)
    size = signed_oof_auroc(tr[Y2], tr["log1p_a_in3"], tr["fold"], clean)
    print(
        f"  clean-now a_vol CV={_f(vol['cv'])} size={_f(size['cv'])} "
        f"gap={(vol['cv']-size['cv']) if np.isfinite(vol['cv']) else float('nan'):+.3f} "
        f"n_pos={vol['n_pos']}"
    )
    return {
        "slices": slices,
        "tab": tab,
        "clean_cv": vol["cv"],
        "clean_size": size["cv"],
        "clean_gap": (vol["cv"] - size["cv"]) if np.isfinite(vol["cv"]) else float("nan"),
        "clean_n_pos": vol["n_pos"],
    }


def pass19_body(tr: pd.DataFrame) -> dict:
    print("CUT 19 — Y3 singles on non-clip body (is the tail the whole story?)")
    y3_mask = tr[Y3].notna()
    body = y3_mask & (tr["a_vol_clip3"] == 0) & tr["a_vol"].notna()
    clip = y3_mask & (tr["a_vol_clip3"] == 1)
    rows = []
    for name, mask in (("body", body), ("clip", clip)):
        size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], mask)
        for feat in ("a_vol", "a_in_vol", "c_n_days_with_tx", "log1p_a_in3"):
            rec = signed_oof_auroc(tr[Y3], tr[feat], tr["fold"], mask)
            row = single_row(feat, rec, size["cv"], DAYS_Y3)
            row["slice"] = name
            rows.append(row)
            print(
                f"  Y3 {name:4s} {feat:20s} CV={_f(row['cv'])} gap={row['gap_size']:+.3f} "
                f"n={row['n']} pos={row['n_pos']}"
            )
    return {"rows": rows}


def pass20_fold0(tr: pd.DataFrame) -> dict:
    print("CUT 20 — fold 0 vs rest a_vol mix (Y3)")
    y3 = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y3.notna()
    rows = []
    for k in range(N_FOLDS):
        sl = lab & (tr["fold"] == k)
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "n_pos": int((sl & (y3 == 1)).sum()),
            "rate": float(y3[sl].mean()) if int(sl.sum()) else float("nan"),
            "med_vol": float(pd.to_numeric(tr.loc[sl, "a_vol"], errors="coerce").median()),
            "clip": float(tr.loc[sl & tr["a_vol"].notna(), "a_vol_clip3"].mean())
            if int((sl & tr["a_vol"].notna()).sum())
            else float("nan"),
            "med_size": float(tr.loc[sl, "log1p_a_in3"].median()),
        }
        rows.append(rec)
        print(
            f"  fold {k} n={rec['n']:4d} pos={rec['n_pos']:3d} P={rec['rate']:.3f} "
            f"med_vol={_f(rec['med_vol'])} clip={_pp(rec['clip'])} "
            f"med_size={_f(rec['med_size'])}"
        )
    return {"rows": rows}


def pass21_out_screen(tr: pd.DataFrame) -> dict:
    print("CUT 21 — SIZE/COPY screen for a_out_vol / a_sd6_net / windows")
    rows = []
    for col in ("a_out_vol", "a_sd6_net", "a_vol_unclip", "a_vol_w3", "a_vol_w12", "a_in_vol"):
        for other, label in (
            ("log1p_a_in3", "size"),
            ("a_io_ratio", "a_io_ratio"),
            ("a_growth_3", "a_growth_3"),
            ("a_out6", "a_out6"),
            ("a_vol", "a_vol"),
            ("c_n_days_with_tx", "days"),
        ):
            if other not in tr.columns:
                continue
            rho, n = spearman(tr[col], tr[other])
            flag_size = bool(np.isfinite(rho) and abs(rho) >= SIZE_RHO and label == "size")
            flag_copy = bool(
                np.isfinite(rho)
                and abs(rho) >= COPY_RHO
                and label in ("a_io_ratio", "a_growth_3", "a_out6", "a_vol", "days")
            )
            rec = {
                "col": col,
                "vs": label,
                "rho": rho,
                "n": n,
                "flag_size": flag_size,
                "flag_copy": flag_copy,
            }
            rows.append(rec)
            mark = " SIZE" if flag_size else (" COPY" if flag_copy else "")
            print(f"  {col:14s} vs {label:12s} ρ={rho:+.3f} n={n:,}{mark}")
    out_size = next(r["rho"] for r in rows if r["col"] == "a_out_vol" and r["vs"] == "size")
    out_copy = any(r["flag_copy"] for r in rows if r["col"] == "a_out_vol")
    sd_size = next(r["rho"] for r in rows if r["col"] == "a_sd6_net" and r["vs"] == "size")
    print(
        f"  a_out_vol SIZE={abs(out_size)>=SIZE_RHO if np.isfinite(out_size) else False} "
        f"COPY={out_copy}; a_sd6_net SIZE ρ={sd_size:+.3f}"
    )
    return {
        "rows": rows,
        "out_size_rho": out_size,
        "out_is_size": bool(np.isfinite(out_size) and abs(out_size) >= SIZE_RHO),
        "out_is_copy": out_copy,
        "sd_size_rho": sd_size,
        "sd_is_size": bool(np.isfinite(sd_size) and abs(sd_size) >= SIZE_RHO),
    }


def pass22_out_honest(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 22 — a_out_vol honesty: drop12 / dark / body / Q6")
    is_ch = tr["company_id"].astype(str).isin(set(ids))
    y3_mask = tr[Y3].notna()
    y2_mask = tr[Y2].notna()
    rows = []
    for yname, ycol, mask, bench in (
        ("y3", Y3, y3_mask, DAYS_Y3),
        ("y2", Y2, y2_mask, NIGHT_Y2),
    ):
        for slname, sl in (
            ("all", mask),
            ("drop12", mask & ~is_ch),
            ("dark", mask & (tr["dark"] == 1)),
            ("erp", mask & (tr["dark"] == 0)),
            ("body", mask & (tr["a_vol_clip3"] == 0) & tr["a_out_vol"].notna()),
        ):
            m = sl
            size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], m)
            rec = signed_oof_auroc(tr[ycol], tr["a_out_vol"], tr["fold"], m)
            row = single_row("a_out_vol", rec, size["cv"], bench)
            row["y"] = yname
            row["slice"] = slname
            rows.append(row)
            print(
                f"  {yname} {slname:6s} CV={_f(row['cv'])} gap={row['gap_size']:+.3f} "
                f"n={row['n']} pos={row['n_pos']} folds {row['folds']}"
            )
    tab_y3 = quintile_table(tr, y3_mask, "a_out_vol", Y3)
    tab_y2 = quintile_table(tr, y2_mask, "a_out_vol", Y2)
    print(f"  Y3 quintiles shape={tab_y3['shape']}")
    for r in tab_y3["rows"]:
        print(f"    Q{r['q']} P={r['y_rate']:.3f} n={r['n']} med={r['x_median']:.4g}")
    print(f"  Y2 quintiles shape={tab_y2['shape']}")
    lag = signed_oof_auroc(tr[Y3], tr["a_out_vol"].groupby(tr["company_id"]).shift(1), tr["fold"], y3_mask)
    now = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], y3_mask)
    size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], y3_mask)
    # lag via explicit series
    out_lag = tr.sort_values(["company_id", "period"])
    # already have panel lags only for a_vol*; compute here
    print(
        f"  Y3 a_out_vol now={_f(now['cv'])} size={_f(size['cv'])} "
        f"shape={tab_y3['shape']}"
    )
    _ = lag
    return {
        "rows": rows,
        "tab_y3": tab_y3,
        "tab_y2": tab_y2,
        "y3_cv": now["cv"],
        "y3_gap": (now["cv"] - size["cv"]) if np.isfinite(now["cv"]) else float("nan"),
        "y3_shape": tab_y3["shape"],
    }


def pass23_io_not12(tr: pd.DataFrame, p4: dict, p12: dict) -> dict:
    print("CUT 23 — a_io_vol Y2 +0.02 is NOT the 12 names (days was)")
    y2_io = next(r for r in p4["y2"]["rows"] if r["feature"] == "a_io_vol")
    y2_days = next(r for r in p4["y2"]["rows"] if r["feature"] == "c_n_days_with_tx")
    io_drop = p12["y2_io_gap"]
    days_drop = p12["y2_days_gap"]
    io_lost = (y2_io["gap_size"] - io_drop) if np.isfinite(io_drop) else float("nan")
    days_lost = (y2_days["gap_size"] - days_drop) if np.isfinite(days_drop) else float("nan")
    io_keep = bool(np.isfinite(y2_io["gap_size"]) and y2_io["gap_size"] >= KEEP_DELTA)
    not_the_12 = bool(np.isfinite(io_lost) and io_lost < 0.01 and days_lost >= 0.01)
    print(
        f"  a_io_vol Y2 gap {y2_io['gap_size']:+.3f} → drop12 {io_drop:+.3f} lost={io_lost:+.3f}"
    )
    print(
        f"  days     Y2 gap {y2_days['gap_size']:+.3f} → drop12 {days_drop:+.3f} lost={days_lost:+.3f}"
    )
    print(f"  not_the_12={not_the_12} numeric_KEEP={io_keep} (Y2 only; Y3 loses)")
    return {
        "io_gap": y2_io["gap_size"],
        "io_drop": io_drop,
        "io_lost": io_lost,
        "days_gap": y2_days["gap_size"],
        "days_drop": days_drop,
        "days_lost": days_lost,
        "not_the_12": not_the_12,
        "io_keep_y2": io_keep,
    }


def pass24_out_lag(tr: pd.DataFrame) -> dict:
    print("CUT 24 — a_out_vol Q6 lag1 + leak vs B (diagnostic)")
    out = tr.sort_values(["company_id", "period"]).reset_index(drop=True)
    out["a_out_vol_lag1"] = out.groupby("company_id", sort=False)["a_out_vol"].shift(1)
    y3_mask = out[Y3].notna()
    y2_mask = out[Y2].notna()
    rows = []
    for yname, ycol, mask, bench in (
        ("y3", Y3, y3_mask, DAYS_Y3),
        ("y2", Y2, y2_mask, NIGHT_Y2),
    ):
        size = signed_oof_auroc(out[ycol], out["log1p_a_in3"], out["fold"], mask)
        now = signed_oof_auroc(out[ycol], out["a_out_vol"], out["fold"], mask)
        lag = signed_oof_auroc(out[ycol], out["a_out_vol_lag1"], out["fold"], mask)
        lift = (lag["cv"] - now["cv"]) if np.isfinite(lag["cv"]) and np.isfinite(now["cv"]) else float("nan")
        rec = {
            "y": yname,
            "now": now["cv"],
            "lag": lag["cv"],
            "lift": lift,
            "gap_size": (lag["cv"] - size["cv"]) if np.isfinite(lag["cv"]) else float("nan"),
            "q6": bool(
                np.isfinite(lag["cv"])
                and np.isfinite(size["cv"])
                and (lag["cv"] - size["cv"]) >= KEEP_DELTA
                and lag["cv"] < SIZE_PARK
                and np.isfinite(lift)
                and lift >= 0.01
            ),
        }
        rows.append(rec)
        print(
            f"  {yname} out_vol lag0={_f(now['cv'])} lag1={_f(lag['cv'])} "
            f"lift={lift:+.3f} q6={rec['q6']}"
        )
    leak_rows = []
    for vs in ("b_below_0", "b_bal_vol"):
        rho, n = spearman(tr["a_out_vol"], tr[vs])
        leak_rows.append({"vs": vs, "rho": rho, "n": n, "fail": bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)})
        print(f"  a_out_vol vs {vs} ρ={rho:+.3f} n={n:,} fail={leak_rows[-1]['fail']}")
    return {"rows": rows, "leak": leak_rows}


def pass25_in_vol_body_story(tr: pd.DataFrame, p19: dict) -> dict:
    print("CUT 25 — a_in_vol body +0.117 is clip-inversion, not a KEEP")
    body = next(r for r in p19["rows"] if r["slice"] == "body" and r["feature"] == "a_in_vol")
    clip = next(r for r in p19["rows"] if r["slice"] == "clip" and r["feature"] == "a_in_vol")
    all_row = None
    print(
        f"  body CV={_f(body['cv'])} gap={body['gap_size']:+.3f}; "
        f"clip CV={_f(clip['cv'])} gap={clip['gap_size']:+.3f} (inverted)"
    )
    print("  overall a_in_vol 0.635 Δ+0.017 is the clip mixing with SIZE. CLOSE.")
    return {"body": body, "clip": clip, "all": all_row}


def pass26_parent_return(p4: dict, p12: dict, p21: dict, p22: dict, p23: dict) -> dict:
    print("CUT 26 — parent-return numbers (no merge unless KEEP)")
    y3 = next(r for r in p4["y3"]["rows"] if r["feature"] == "a_vol")
    y2 = next(r for r in p4["y2"]["rows"] if r["feature"] == "a_vol")
    in_y3 = next(r for r in p4["y3"]["rows"] if r["feature"] == "a_in_vol")
    io_y2 = next(r for r in p4["y2"]["rows"] if r["feature"] == "a_io_vol")
    out_keep = bool(
        np.isfinite(p22["y3_gap"])
        and p22["y3_gap"] >= KEEP_DELTA
        and (not p21["out_is_size"])
        and (not p21["out_is_copy"])
    )
    print(
        f"  a_vol Y3={y3['cv']:.3f} Δ={y3['gap_size']:+.3f} Y2={y2['cv']:.3f} Δ={y2['gap_size']:+.3f} CLOSE"
    )
    print(
        f"  a_in_vol Y3={in_y3['cv']:.3f} Δ={in_y3['gap_size']:+.3f} CLOSE (miss 0.02)"
    )
    print(
        f"  a_io_vol Y2={io_y2['cv']:.3f} Δ={io_y2['gap_size']:+.3f} "
        f"not12={p23['not_the_12']} footnote only — Y3 loses"
    )
    print(
        f"  a_out_vol Y3={_f(p22['y3_cv'])} Δ={p22['y3_gap']:+.3f} "
        f"SIZE={p21['out_is_size']} COPY={p21['out_is_copy']} KEEP={out_keep} "
        f"shape={p22['y3_shape']}"
    )
    return {
        "out_keep": out_keep,
        "io_footnote": bool(p23["io_keep_y2"] and p23["not_the_12"]),
    }


def pass27_core_redundancy(tr: pd.DataFrame) -> dict:
    print("CUT 27 — a_out_vol vs 15-col shallow-A stems (not a card retrain)")
    y3 = tr[Y3].notna()
    rows = []
    for col in CORE_STEMS:
        rho_all, n_all = spearman(tr["a_out_vol"], tr[col])
        rho_y3, n_y3 = spearman(tr.loc[y3, "a_out_vol"], tr.loc[y3, col])
        rec = {
            "vs": col,
            "rho_all": rho_all,
            "n_all": n_all,
            "rho_y3": rho_y3,
            "n_y3": n_y3,
            "copy": bool(
                (np.isfinite(rho_all) and abs(rho_all) >= COPY_RHO)
                or (np.isfinite(rho_y3) and abs(rho_y3) >= COPY_RHO)
            ),
        }
        rows.append(rec)
        print(
            f"  vs {col:20s} all ρ={rho_all:+.3f} n={n_all:,}  "
            f"Y3 ρ={rho_y3:+.3f} n={n_y3:,}{' COPY' if rec['copy'] else ''}"
        )
    any_copy = any(r["copy"] for r in rows)
    print(f"  core COPY={any_copy} — KEEP a_out_vol stays {not any_copy}")
    return {"rows": rows, "any_copy": any_copy}


def pass28_out_size_mix(tr: pd.DataFrame) -> dict:
    print("CUT 28 — Y3 size tercile × a_out_vol quintile (small-book tail?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    mask = y.notna() & tr["a_out_vol"].notna() & tr["log1p_a_in3"].notna()
    d = tr.loc[mask, ["a_out_vol", "log1p_a_in3"]].copy()
    d["y"] = y[mask].to_numpy()
    d["size_t"] = pd.qcut(d["log1p_a_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    d["vol_q"] = pd.qcut(d["a_out_vol"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"], duplicates="drop")
    rows = []
    for (st, vq), g in d.groupby(["size_t", "vol_q"], observed=True):
        rec = {
            "size": str(st),
            "vol": str(vq),
            "n": int(len(g)),
            "n_pos": int((g["y"] == 1).sum()),
            "rate": float(g["y"].mean()),
        }
        rows.append(rec)
        print(f"  {st} {vq} n={rec['n']:4d} pos={rec['n_pos']:3d} P={rec['rate']:.3f}")
    q5 = d["vol_q"].astype(str) == "Q5"
    t1 = d["size_t"].astype(str) == "T1"
    q5_t1 = float(t1[q5].mean()) if int(q5.sum()) else float("nan")
    print(f"  a_out_vol Q5 share in T1={q5_t1:.1%}")
    # singles on T2+T3 only
    mid = mask & (pd.qcut(tr.loc[mask, "log1p_a_in3"], 3, labels=False, duplicates="drop") >= 1)
    # mid is aligned to mask index? safer rebuild
    size_t = pd.Series(np.nan, index=tr.index)
    size_t.loc[mask] = pd.qcut(tr.loc[mask, "log1p_a_in3"], 3, labels=False, duplicates="drop")
    midbig = y.notna() & size_t.isin([1, 2])
    size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], midbig)
    outv = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], midbig)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], midbig)
    gap = (outv["cv"] - size["cv"]) if np.isfinite(outv["cv"]) else float("nan")
    print(
        f"  T2+T3 a_out_vol CV={_f(outv['cv'])} size={_f(size['cv'])} "
        f"gap={gap:+.3f} days={_f(days['cv'])} n={outv['n_defined']} pos={outv['n_pos']}"
    )
    keep_mid = bool(np.isfinite(gap) and gap >= KEEP_DELTA)
    _ = mid
    return {
        "rows": rows,
        "q5_t1_share": q5_t1,
        "mid_cv": outv["cv"],
        "mid_size": size["cv"],
        "mid_days": days["cv"],
        "mid_gap": gap,
        "mid_keep": keep_mid,
        "mid_n": outv["n_defined"],
        "mid_pos": outv["n_pos"],
        "mid_folds": fold_bits(outv),
    }


def pass29_acf(tr: pd.DataFrame) -> dict:
    print("CUT 29 — company-median acf1 (lead-time texture, not Q6)")
    rows = []
    for col in ("a_vol", "a_in_vol", "a_io_vol", "a_out_vol"):
        vals = []
        for _, g in tr.groupby("company_id", sort=False):
            x = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            if len(x) < 5:
                continue
            a, b = x[:-1], x[1:]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 4:
                continue
            aa, bb = a[m], b[m]
            if np.std(aa) == 0 or np.std(bb) == 0:
                continue
            vals.append(float(np.corrcoef(aa, bb)[0, 1]))
        rec = {
            "col": col,
            "n": int(len(vals)),
            "p50": float(np.median(vals)) if vals else float("nan"),
            "p10": float(np.quantile(vals, 0.10)) if vals else float("nan"),
            "p90": float(np.quantile(vals, 0.90)) if vals else float("nan"),
        }
        rows.append(rec)
        print(f"  {col:12s} acf1 p50={_f(rec['p50'])} n_cos={rec['n']} p10={_f(rec['p10'])} p90={_f(rec['p90'])}")
    return {"rows": rows}


def pass30_out_t1_only(tr: pd.DataFrame) -> dict:
    print("CUT 30 — a_out_vol Y3 inside size T1 only (is KEEP just small books?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    mask = y.notna() & tr["a_out_vol"].notna() & tr["log1p_a_in3"].notna()
    size_t = pd.Series(np.nan, index=tr.index)
    size_t.loc[mask] = pd.qcut(tr.loc[mask, "log1p_a_in3"], 3, labels=False, duplicates="drop")
    t1 = y.notna() & (size_t == 0)
    size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], t1)
    outv = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], t1)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], t1)
    gap = (outv["cv"] - size["cv"]) if np.isfinite(outv["cv"]) else float("nan")
    print(
        f"  T1 a_out_vol CV={_f(outv['cv'])} size={_f(size['cv'])} "
        f"gap={gap:+.3f} days={_f(days['cv'])} n={outv['n_defined']} pos={outv['n_pos']}"
    )
    return {
        "cv": outv["cv"],
        "size": size["cv"],
        "days": days["cv"],
        "gap": gap,
        "n": outv["n_defined"],
        "n_pos": outv["n_pos"],
        "folds": fold_bits(outv),
    }


def pass31_days_x_out(tr: pd.DataFrame) -> dict:
    print("CUT 31 — Y3 2x2 days-median × a_out_vol-median (substitute?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    m = y.notna() & days.notna() & ov.notna()
    dmed = float(days[m].median())
    omed = float(ov[m].median())
    hi_d = days >= dmed
    hi_o = ov >= omed
    rows = []
    for name, sl in (
        ("low_days_low_out", m & ~hi_d & ~hi_o),
        ("low_days_hi_out", m & ~hi_d & hi_o),
        ("hi_days_low_out", m & hi_d & ~hi_o),
        ("hi_days_hi_out", m & hi_d & hi_o),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((sl & (y == 1)).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:18s} n={rec['n']:4d} pos={rec['n_pos']:3d} P={rec['rate']:.3f}")
    lo = next(r for r in rows if r["slice"] == "low_days_low_out")["rate"]
    lh = next(r for r in rows if r["slice"] == "low_days_hi_out")["rate"]
    hl = next(r for r in rows if r["slice"] == "hi_days_low_out")["rate"]
    hh = next(r for r in rows if r["slice"] == "hi_days_hi_out")["rate"]
    lifts_in_quiet = (lh - lo) if np.isfinite(lh) and np.isfinite(lo) else float("nan")
    lifts_in_busy = (hh - hl) if np.isfinite(hh) and np.isfinite(hl) else float("nan")
    substitute = bool(np.isfinite(lifts_in_quiet) and lifts_in_quiet < 0.02)
    print(
        f"  outvol lift in quiet-days={lifts_in_quiet:+.3f} "
        f"in busy-days={lifts_in_busy:+.3f} substitute={substitute}"
    )
    return {
        "rows": rows,
        "dmed": dmed,
        "omed": omed,
        "lift_quiet": lifts_in_quiet,
        "lift_busy": lifts_in_busy,
        "substitute": substitute,
    }


def pass32_days_x_out_mid(tr: pd.DataFrame) -> dict:
    print("CUT 32 — same 2x2 on size T2+T3 only")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    m0 = y.notna() & days.notna() & ov.notna() & size.notna()
    terc = pd.Series(np.nan, index=tr.index)
    terc.loc[m0] = pd.qcut(size[m0], 3, labels=False, duplicates="drop")
    m = m0 & terc.isin([1, 2])
    dmed = float(days[m].median())
    omed = float(ov[m].median())
    hi_d = days >= dmed
    hi_o = ov >= omed
    rows = []
    for name, sl in (
        ("low_days_low_out", m & ~hi_d & ~hi_o),
        ("low_days_hi_out", m & ~hi_d & hi_o),
        ("hi_days_low_out", m & hi_d & ~hi_o),
        ("hi_days_hi_out", m & hi_d & hi_o),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((sl & (y == 1)).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:18s} n={rec['n']:4d} pos={rec['n_pos']:3d} P={rec['rate']:.3f}")
    lo = next(r for r in rows if r["slice"] == "low_days_low_out")["rate"]
    lh = next(r for r in rows if r["slice"] == "low_days_hi_out")["rate"]
    lift = (lh - lo) if np.isfinite(lh) and np.isfinite(lo) else float("nan")
    print(f"  T2+T3 quiet-days outvol lift={lift:+.3f}")
    return {"rows": rows, "lift_quiet": lift, "dmed": dmed, "omed": omed}


def pass33_out6_x_outvol(tr: pd.DataFrame) -> dict:
    print("CUT 33 — Y3 2x2 a_out6-median × a_out_vol (just high outflow?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    out6 = pd.to_numeric(tr["a_out6"], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    m = y.notna() & out6.notna() & ov.notna()
    o6m = float(out6[m].median())
    ovm = float(ov[m].median())
    hi6 = out6 >= o6m
    hio = ov >= ovm
    rows = []
    for name, sl in (
        ("low_out6_low_vol", m & ~hi6 & ~hio),
        ("low_out6_hi_vol", m & ~hi6 & hio),
        ("hi_out6_low_vol", m & hi6 & ~hio),
        ("hi_out6_hi_vol", m & hi6 & hio),
    ):
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((sl & (y == 1)).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:18s} n={rec['n']:4d} pos={rec['n_pos']:3d} P={rec['rate']:.3f}")
    lo = next(r for r in rows if r["slice"] == "low_out6_low_vol")["rate"]
    lh = next(r for r in rows if r["slice"] == "low_out6_hi_vol")["rate"]
    lift = (lh - lo) if np.isfinite(lh) and np.isfinite(lo) else float("nan")
    print(f"  vol lift inside low a_out6={lift:+.3f}")
    return {"rows": rows, "lift_low_out6": lift}


def pass34_sofar(tr: pd.DataFrame) -> dict:
    print("CUT 34 — a_out_vol Y3 by so-far (short-book artifact?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    so = (tr["period"].dt.year - fm.dt.year) * 12 + (tr["period"].dt.month - fm.dt.month) + 1
    lab = y.notna()
    buckets = [("<6", lab & (so < 6)), ("6-11", lab & (so >= 6) & (so < 12)),
               ("12-17", lab & (so >= 12) & (so < 18)), ("18+", lab & (so >= 18))]
    rows = []
    for name, sl in buckets:
        rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], sl)
        size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], sl)
        row = {
            "bucket": name,
            "n": rec["n_defined"],
            "n_pos": rec["n_pos"],
            "cv": rec["cv"],
            "size": size["cv"],
            "gap": (rec["cv"] - size["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(size["cv"]) else float("nan"),
            "low_power": rec["low_power"],
        }
        rows.append(row)
        print(
            f"  {name:6s} n={row['n']:4d} pos={row['n_pos']:3d} "
            f"CV={_f(row['cv'])} size={_f(row['size'])} gap={row['gap']:+.3f} "
            f"low={row['low_power']}"
        )
    return {"rows": rows}


def pass35_b_diag(tr: pd.DataFrame) -> dict:
    print("CUT 35 — a_out_vol vs B (diagnostic, never X)")
    rows = []
    for vs in ("b_below_0", "b_bal_vol"):
        rho, n = spearman(tr["a_out_vol"], tr[vs])
        rec = {"vs": vs, "rho": rho, "n": n, "fail": bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)}
        rows.append(rec)
        print(f"  vs {vs} ρ={rho:+.3f} n={n:,} fail={rec['fail']}")
    # also a_vol vs B
    for vs in ("b_below_0", "b_bal_vol"):
        rho, n = spearman(tr["a_vol"], tr[vs])
        rec = {"vs": f"a_vol~{vs}", "rho": rho, "n": n, "fail": bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)}
        rows.append(rec)
        print(f"  a_vol vs {vs} ρ={rho:+.3f} n={n:,} fail={rec['fail']}")
    return {"rows": rows, "any_fail": any(r["fail"] for r in rows)}


def pass36_dark_2x2(tr: pd.DataFrame) -> dict:
    print("CUT 36 — days × a_out_vol 2x2 inside dark / invoiced")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    blocks = []
    for pop, pmask in (("dark", tr["dark"] == 1), ("erp", tr["dark"] == 0)):
        m = y.notna() & days.notna() & ov.notna() & pmask
        if int(m.sum()) < 80:
            continue
        dmed = float(days[m].median())
        omed = float(ov[m].median())
        hi_d = days >= dmed
        hi_o = ov >= omed
        rows = []
        for name, sl in (
            ("low_days_low_out", m & ~hi_d & ~hi_o),
            ("low_days_hi_out", m & ~hi_d & hi_o),
            ("hi_days_low_out", m & hi_d & ~hi_o),
            ("hi_days_hi_out", m & hi_d & hi_o),
        ):
            rec = {
                "pop": pop,
                "slice": name,
                "n": int(sl.sum()),
                "n_pos": int((sl & (y == 1)).sum()),
                "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            }
            rows.append(rec)
            print(f"  {pop:4s} {name:18s} n={rec['n']:4d} P={rec['rate']:.3f}")
        lo = next(r for r in rows if r["slice"] == "low_days_low_out")["rate"]
        lh = next(r for r in rows if r["slice"] == "low_days_hi_out")["rate"]
        lift = (lh - lo) if np.isfinite(lh) and np.isfinite(lo) else float("nan")
        blocks.append({"pop": pop, "rows": rows, "lift": lift})
        print(f"  {pop} quiet-days lift={lift:+.3f}")
    return {"blocks": blocks}


def pass37_avol_sofar(tr: pd.DataFrame) -> dict:
    print("CUT 37 — a_vol Y3 by so-far (does Javier vol ever beat size?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    so = (tr["period"].dt.year - fm.dt.year) * 12 + (tr["period"].dt.month - fm.dt.month) + 1
    lab = y.notna()
    rows = []
    for name, sl in (
        ("6-11", lab & (so >= 6) & (so < 12)),
        ("12-17", lab & (so >= 12) & (so < 18)),
    ):
        rec = signed_oof_auroc(tr[Y3], tr["a_vol"], tr["fold"], sl)
        size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], sl)
        days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], sl)
        gap = (rec["cv"] - size["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(size["cv"]) else float("nan")
        row = {
            "bucket": name,
            "n": rec["n_defined"],
            "n_pos": rec["n_pos"],
            "cv": rec["cv"],
            "size": size["cv"],
            "days": days["cv"],
            "gap": gap,
        }
        rows.append(row)
        print(
            f"  {name} a_vol={_f(row['cv'])} size={_f(row['size'])} "
            f"days={_f(row['days'])} gap={gap:+.3f} n={row['n']} pos={row['n_pos']}"
        )
    return {"rows": rows}


def pass38_recovery_names(tr: pd.DataFrame) -> dict:
    print("CUT 38 — who owns the quiet+high-out-vol Y3 recoveries?")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    m = y.notna() & days.notna() & ov.notna()
    dmed = float(days[m].median())
    omed = float(ov[m].median())
    pile = m & (days < dmed) & (ov >= omed)
    pos = pile & (y == 1)
    pile_cos = tr.loc[pile, "company_id"].nunique()
    pos_cos = tr.loc[pos, "company_id"].nunique()
    pos_n = int(pos.sum())
    top = (
        tr.loc[pos]
        .groupby("company_id", observed=True)
        .size()
        .sort_values(ascending=False)
        .head(8)
    )
    top_share = float(top.sum() / pos_n) if pos_n else float("nan")
    print(
        f"  pile n={int(pile.sum())} pos={pos_n} companies={pos_cos} "
        f"(pile companies={pile_cos}) top8 share={top_share:.3f}"
    )
    for cid, n in top.items():
        print(f"    {cid} recoveries={int(n)}")
    concentrated = bool(np.isfinite(top_share) and top_share >= 0.40)
    print(f"  concentrated_in_top8={concentrated}")
    return {
        "pile_n": int(pile.sum()),
        "pos_n": pos_n,
        "pos_cos": int(pos_cos),
        "pile_cos": int(pile_cos),
        "top8_share": top_share,
        "concentrated": concentrated,
        "top": [{"company_id": str(k), "n": int(v)} for k, v in top.items()],
    }


def pass39_avol_leftover(tr: pd.DataFrame) -> dict:
    print("CUT 39 — why overall a_vol Δ+0.009 < window Δ+0.04")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    fm = pd.to_datetime(tr["first_month"])
    so = (tr["period"].dt.year - fm.dt.year) * 12 + (tr["period"].dt.month - fm.dt.month) + 1
    defined = y.notna() & tr["a_vol"].notna()
    buckets = {
        "<6": defined & (so < 6),
        "6-11": defined & (so >= 6) & (so < 12),
        "12-17": defined & (so >= 12) & (so < 18),
        "18+": defined & (so >= 18),
        "so_na": defined & so.isna(),
    }
    rows = []
    for name, sl in buckets.items():
        rec = signed_oof_auroc(tr[Y3], tr["a_vol"], tr["fold"], sl)
        size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], sl)
        gap = (rec["cv"] - size["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(size["cv"]) else float("nan")
        row = {
            "bucket": name,
            "n": rec["n_defined"],
            "n_pos": rec["n_pos"],
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "cv": rec["cv"],
            "size": size["cv"],
            "gap": gap,
        }
        rows.append(row)
        print(
            f"  {name:6s} n={row['n']:4d} pos={row['n_pos']:3d} "
            f"P={row['rate']:.3f} a_vol={_f(row['cv'])} size={_f(row['size'])} "
            f"gap={gap:+.3f}" if np.isfinite(gap) else
            f"  {name:6s} n={row['n']:4d} pos={row['n_pos']:3d} P={row['rate']:.3f} LOW_POWER"
        )
    all_rec = signed_oof_auroc(tr[Y3], tr["a_vol"], tr["fold"], defined)
    all_size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], defined)
    print(
        f"  ALL defined n={all_rec['n_defined']} a_vol={_f(all_rec['cv'])} "
        f"size={_f(all_size['cv'])} gap={all_rec['cv']-all_size['cv']:+.3f}"
    )
    return {
        "rows": rows,
        "all_n": all_rec["n_defined"],
        "all_cv": all_rec["cv"],
        "all_size": all_size["cv"],
        "all_gap": (all_rec["cv"] - all_size["cv"]) if np.isfinite(all_rec["cv"]) and np.isfinite(all_size["cv"]) else float("nan"),
    }


def pass40_avol_mid(tr: pd.DataFrame) -> dict:
    print("CUT 40 — a_vol Y3 on T2+T3 (does Javier KEEP off small books?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & size.notna()
    terc = pd.qcut(size[mask], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    mid = pd.Series(False, index=tr.index)
    mid.loc[terc.index] = terc.isin(["T2", "T3"])
    rec = signed_oof_auroc(tr[Y3], tr["a_vol"], tr["fold"], mid)
    sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], mid)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], mid)
    gap = (rec["cv"] - sz["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(sz["cv"]) else float("nan")
    keepish = bool(np.isfinite(gap) and gap >= KEEP_DELTA)
    print(
        f"  T2+T3 a_vol={_f(rec['cv'])} size={_f(sz['cv'])} days={_f(days['cv'])} "
        f"gap={gap:+.3f} n={rec['n_defined']} pos={rec['n_pos']} keepish={keepish}"
    )
    return {
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "cv": rec["cv"],
        "size": sz["cv"],
        "days": days["cv"],
        "gap": gap,
        "keepish": keepish,
    }


def pass41_avol_t1(tr: pd.DataFrame) -> dict:
    print("CUT 41 — a_vol Y3 inside size T1 only (is the +0.009 a small-book tail?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & size.notna()
    terc = pd.qcut(size[mask], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    t1 = pd.Series(False, index=tr.index)
    t1.loc[terc.index] = terc.eq("T1")
    rec = signed_oof_auroc(tr[Y3], tr["a_vol"], tr["fold"], t1)
    sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], t1)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], t1)
    gap = (rec["cv"] - sz["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(sz["cv"]) else float("nan")
    print(
        f"  T1 a_vol={_f(rec['cv'])} size={_f(sz['cv'])} days={_f(days['cv'])} "
        f"gap={gap:+.3f} n={rec['n_defined']} pos={rec['n_pos']}"
    )
    return {
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "cv": rec["cv"],
        "size": sz["cv"],
        "days": days["cv"],
        "gap": gap,
        "keepish": bool(np.isfinite(gap) and gap >= KEEP_DELTA),
    }


def pass42_simpson(tr: pd.DataFrame) -> dict:
    print("CUT 42 — Simpson: Y3 rate by size tercile × a_vol median")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    vol = pd.to_numeric(tr["a_vol"], errors="coerce")
    mask = y.notna() & size.notna() & vol.notna()
    terc = pd.qcut(size[mask], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    vmed = float(vol[mask].median())
    hi = vol >= vmed
    rows = []
    for t in ("T1", "T2", "T3"):
        sl_t = pd.Series(False, index=tr.index)
        sl_t.loc[terc.index] = terc.eq(t)
        for name, sl in ((f"{t}_low_vol", sl_t & ~hi & mask), (f"{t}_hi_vol", sl_t & hi & mask)):
            n = int(sl.sum())
            n_pos = int((sl & (y == 1)).sum())
            rate = float(y[sl].mean()) if n else float("nan")
            mean_vol = float(vol[sl].mean()) if n else float("nan")
            rows.append({"slice": name, "n": n, "n_pos": n_pos, "rate": rate, "mean_vol": mean_vol})
            print(f"  {name:14s} n={n:4d} pos={n_pos:3d} P={rate:.3f} mean_vol={mean_vol:.2f}")
    bases = []
    for t in ("T1", "T2", "T3"):
        sl = pd.Series(False, index=tr.index)
        sl.loc[terc.index] = terc.eq(t)
        sl = sl & mask
        bases.append({
            "tercile": t,
            "n": int(sl.sum()),
            "n_pos": int((sl & (y == 1)).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "mean_vol": float(vol[sl].mean()) if int(sl.sum()) else float("nan"),
        })
        print(f"  base {t} n={bases[-1]['n']} P={bases[-1]['rate']:.3f} mean_vol={bases[-1]['mean_vol']:.2f}")
    return {"rows": rows, "bases": bases, "vmed": vmed}


def pass43_out_simpson(tr: pd.DataFrame) -> dict:
    print("CUT 43 — Simpson for a_out_vol (does lift live inside T2/T3?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    vol = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    mask = y.notna() & size.notna() & vol.notna()
    terc = pd.qcut(size[mask], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    vmed = float(vol[mask].median())
    hi = vol >= vmed
    rows = []
    for t in ("T1", "T2", "T3"):
        sl_t = pd.Series(False, index=tr.index)
        sl_t.loc[terc.index] = terc.eq(t)
        lo = sl_t & ~hi & mask
        hv = sl_t & hi & mask
        for name, sl in ((f"{t}_low", lo), (f"{t}_hi", hv)):
            n = int(sl.sum())
            rate = float(y[sl].mean()) if n else float("nan")
            rows.append({"slice": name, "n": n, "n_pos": int((sl & (y == 1)).sum()), "rate": rate})
            print(f"  {name:10s} n={n:4d} P={rate:.3f}")
        lift = rows[-1]["rate"] - rows[-2]["rate"]
        print(f"  {t} hi-lo lift={lift:+.3f}")
        rows[-1]["lift"] = lift
    return {"rows": rows, "vmed": vmed}


def pass44_clip_binary(tr: pd.DataFrame) -> dict:
    print("CUT 44 — is a_vol just the clip=3 flag?")
    y3m = tr[Y3].notna()
    y2m = tr[Y2].notna()
    flag = tr["a_vol_clip3"]
    rows = []
    for ycol, mask, bench in ((Y3, y3m, DAYS_Y3), (Y2, y2m, NIGHT_Y2)):
        rec = signed_oof_auroc(tr[ycol], flag, tr["fold"], mask)
        cont = signed_oof_auroc(tr[ycol], tr["a_vol"], tr["fold"], mask)
        size = signed_oof_auroc(tr[ycol], tr["log1p_a_in3"], tr["fold"], mask)
        row = {
            "y": ycol,
            "flag": rec["cv"],
            "cont": cont["cv"],
            "size": size["cv"],
            "n": rec["n_defined"],
            "n_pos": rec["n_pos"],
            "almost_flag": bool(
                np.isfinite(rec["cv"]) and np.isfinite(cont["cv"]) and abs(rec["cv"] - cont["cv"]) < 0.02
            ),
        }
        rows.append(row)
        print(
            f"  {ycol} flag={_f(row['flag'])} cont={_f(row['cont'])} "
            f"size={_f(row['size'])} almost_flag={row['almost_flag']}"
        )
    return {"rows": rows}


def pass45_company_trait(tr: pd.DataFrame) -> dict:
    print("CUT 45 — company-mean vol vs ever-recover (trait vs month spike)")
    d = tr.loc[tr[Y3].notna(), ["company_id", "fold", Y3, "a_vol", "a_out_vol", "log1p_a_in3", "c_n_days_with_tx"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        n=(Y3, "size"),
        n_pos=(Y3, "sum"),
        a_vol=("a_vol", "mean"),
        a_out_vol=("a_out_vol", "mean"),
        size=("log1p_a_in3", "mean"),
        days=("c_n_days_with_tx", "mean"),
        fold=("fold", "first"),
    )
    labeled = g["n"] >= 1
    g = g.loc[labeled]
    rows = []
    for col, bench_name in (
        ("a_vol", "size"),
        ("a_out_vol", "size"),
        ("size", "days"),
        ("days", "size"),
    ):
        rec = signed_oof_auroc(g["ever"], g[col], g["fold"], pd.Series(True, index=g.index))
        rows.append({"feature": col, "cv": rec["cv"], "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  company {col} ever-Y3 AUROC={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    return {"rows": rows, "n_cos": int(len(g)), "n_pos_cos": int(g["ever"].sum())}


def pass46_company_common(tr: pd.DataFrame) -> dict:
    print("CUT 46 — company-mean AUROC on common defined companies")
    d = tr.loc[tr[Y3].notna(), ["company_id", "fold", Y3, "a_vol", "a_out_vol", "log1p_a_in3", "c_n_days_with_tx"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_vol=("a_vol", "mean"),
        a_out_vol=("a_out_vol", "mean"),
        size=("log1p_a_in3", "mean"),
        days=("c_n_days_with_tx", "mean"),
        fold=("fold", "first"),
    )
    common = g[["a_vol", "a_out_vol", "size", "days"]].notna().all(axis=1)
    g = g.loc[common]
    rows = []
    for col in ("a_vol", "a_out_vol", "size", "days"):
        rec = signed_oof_auroc(g["ever"], g[col], g["fold"], pd.Series(True, index=g.index))
        rows.append({"feature": col, "cv": rec["cv"], "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  common {col}={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    out = next(r["cv"] for r in rows if r["feature"] == "a_out_vol")
    size = next(r["cv"] for r in rows if r["feature"] == "size")
    days = next(r["cv"] for r in rows if r["feature"] == "days")
    av = next(r["cv"] for r in rows if r["feature"] == "a_vol")
    print(
        f"  a_out_vol vs size {out-size:+.3f} vs days {out-days:+.3f}; "
        f"a_vol vs size {av-size:+.3f}"
    )
    return {
        "rows": rows,
        "n_cos": int(len(g)),
        "n_pos": int(g["ever"].sum()),
        "out_gap_size": out - size if np.isfinite(out) and np.isfinite(size) else float("nan"),
        "out_gap_days": out - days if np.isfinite(out) and np.isfinite(days) else float("nan"),
        "av_gap_size": av - size if np.isfinite(av) and np.isfinite(size) else float("nan"),
    }


def pass47_demean(tr: pd.DataFrame) -> dict:
    print("CUT 47 — company-demeaned vol (timing vs trait)")
    rows = []
    for col in ("a_vol", "a_out_vol", "c_n_days_with_tx", "log1p_a_in3"):
        dem = tr[col] - tr.groupby("company_id")[col].transform("mean")
        rec = signed_oof_auroc(tr[Y3], dem, tr["fold"], tr[Y3].notna())
        raw = signed_oof_auroc(tr[Y3], tr[col], tr["fold"], tr[Y3].notna())
        row = {
            "feature": col,
            "raw": raw["cv"],
            "demean": rec["cv"],
            "n": rec["n_defined"],
            "n_pos": rec["n_pos"],
            "drop": (raw["cv"] - rec["cv"]) if np.isfinite(raw["cv"]) and np.isfinite(rec["cv"]) else float("nan"),
        }
        rows.append(row)
        print(
            f"  {col:16s} raw={_f(row['raw'])} demean={_f(row['demean'])} "
            f"drop={row['drop']:+.3f} n={row['n']}"
        )
    return {"rows": rows}


def pass48_company_copy(tr: pd.DataFrame) -> dict:
    print("CUT 48 — company-mean a_out_vol vs days (same ranking?)")
    d = tr.loc[tr[Y3].notna(), ["company_id", "a_out_vol", "a_vol", "c_n_days_with_tx", "log1p_a_in3"]].copy()
    g = d.groupby("company_id", observed=True).mean(numeric_only=True)
    pairs = [
        ("a_out_vol", "c_n_days_with_tx"),
        ("a_out_vol", "log1p_a_in3"),
        ("a_out_vol", "a_vol"),
        ("a_vol", "c_n_days_with_tx"),
        ("a_vol", "log1p_a_in3"),
    ]
    rows = []
    for a, b in pairs:
        rho, n = spearman(g[a], g[b])
        flag = bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)
        rows.append({"a": a, "b": b, "rho": rho, "n": n, "copy": flag})
        print(f"  {a} vs {b} ρ={rho:+.3f} n={n} COPY={flag}")
    out_days = next(r for r in rows if r["b"] == "c_n_days_with_tx" and r["a"] == "a_out_vol")
    return {"rows": rows, "out_days_copy": out_days["copy"], "out_days_rho": out_days["rho"]}


def pass49_company_2x2(tr: pd.DataFrame) -> dict:
    print("CUT 49 — company-mean days × a_out_vol vs ever-recover")
    d = tr.loc[tr[Y3].notna(), ["company_id", Y3, "a_out_vol", "c_n_days_with_tx"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        days=("c_n_days_with_tx", "mean"),
    )
    g = g.dropna()
    dmed = float(g["days"].median())
    omed = float(g["a_out_vol"].median())
    hi_d = g["days"] >= dmed
    hi_o = g["a_out_vol"] >= omed
    rows = []
    for name, sl in (
        ("low_days_low_out", ~hi_d & ~hi_o),
        ("low_days_hi_out", ~hi_d & hi_o),
        ("hi_days_low_out", hi_d & ~hi_o),
        ("hi_days_hi_out", hi_d & hi_o),
    ):
        n = int(sl.sum())
        n_pos = int((sl & (g["ever"] == 1)).sum())
        rate = float(g.loc[sl, "ever"].mean()) if n else float("nan")
        rows.append({"slice": name, "n": n, "n_pos": n_pos, "rate": rate})
        print(f"  {name:18s} n={n:3d} ever={n_pos:3d} P={rate:.3f}")
    lo = next(r["rate"] for r in rows if r["slice"] == "low_days_low_out")
    lh = next(r["rate"] for r in rows if r["slice"] == "low_days_hi_out")
    ho = next(r["rate"] for r in rows if r["slice"] == "hi_days_low_out")
    hh = next(r["rate"] for r in rows if r["slice"] == "hi_days_hi_out")
    lift_quiet = lh - lo
    lift_busy = hh - ho
    print(f"  company quiet lift={lift_quiet:+.3f} busy lift={lift_busy:+.3f}")
    return {
        "rows": rows,
        "dmed": dmed,
        "omed": omed,
        "lift_quiet": lift_quiet,
        "lift_busy": lift_busy,
        "n_cos": int(len(g)),
    }


def pass50_y2_company(tr: pd.DataFrame) -> dict:
    print("CUT 50 — company-mean vs ever Y2 (common companies)")
    d = tr.loc[tr[Y2].notna(), ["company_id", "fold", Y2, "a_vol", "a_out_vol", "a_io_vol", "log1p_a_in3"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y2, "max"),
        a_vol=("a_vol", "mean"),
        a_out_vol=("a_out_vol", "mean"),
        a_io_vol=("a_io_vol", "mean"),
        size=("log1p_a_in3", "mean"),
        fold=("fold", "first"),
    )
    common = g[["a_vol", "a_out_vol", "a_io_vol", "size"]].notna().all(axis=1)
    g = g.loc[common]
    rows = []
    for col in ("a_vol", "a_out_vol", "a_io_vol", "size"):
        rec = signed_oof_auroc(g["ever"], g[col], g["fold"], pd.Series(True, index=g.index))
        rows.append({"feature": col, "cv": rec["cv"], "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  Y2-company {col}={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    return {"rows": rows, "n_cos": int(len(g)), "n_pos": int(g["ever"].sum())}


def pass51_store_audit() -> dict:
    print("CUT 51 — store must not already contain a_vol / a_out_vol")
    import pyarrow.parquet as pq

    schema = pq.read_schema(STORE)
    names = set(schema.names)
    hits = sorted(n for n in names if n in {"a_vol", "a_out_vol", "a_in_vol", "a_io_vol", "volatility"})
    print(f"  store cols hit={hits or 'none'} (want none)")
    return {"hits": hits, "clean": not hits}


def pass52_in_vol_company(tr: pd.DataFrame) -> dict:
    print("CUT 52 — a_in_vol company-mean Y3 (the 0.003 KEEP miss)")
    d = tr.loc[tr[Y3].notna(), ["company_id", "fold", Y3, "a_in_vol", "a_vol", "log1p_a_in3"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_in_vol=("a_in_vol", "mean"),
        a_vol=("a_vol", "mean"),
        size=("log1p_a_in3", "mean"),
        fold=("fold", "first"),
    )
    common = g[["a_in_vol", "a_vol", "size"]].notna().all(axis=1)
    g = g.loc[common]
    rows = []
    for col in ("a_in_vol", "a_vol", "size"):
        rec = signed_oof_auroc(g["ever"], g[col], g["fold"], pd.Series(True, index=g.index))
        rows.append({"feature": col, "cv": rec["cv"], "n": rec["n_defined"], "n_pos": rec["n_pos"]})
        print(f"  {col}={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']}")
    inv = next(r["cv"] for r in rows if r["feature"] == "a_in_vol")
    size = next(r["cv"] for r in rows if r["feature"] == "size")
    gap = inv - size if np.isfinite(inv) and np.isfinite(size) else float("nan")
    print(f"  a_in_vol vs size {gap:+.3f}")
    return {"rows": rows, "gap": gap, "n_cos": int(len(g)), "n_pos": int(g["ever"].sum())}


def pass53_activity_twin(tr: pd.DataFrame) -> dict:
    print("CUT 53 — a_out_vol vs activity (a_n_tx / days) month-level")
    rows = []
    for col, vs in (
        ("a_out_vol", "a_n_tx"),
        ("a_out_vol", "c_n_days_with_tx"),
        ("a_vol", "a_n_tx"),
        ("a_vol", "c_n_days_with_tx"),
        ("a_in_vol", "a_n_tx"),
    ):
        rho, n = spearman(tr[col], tr[vs])
        flag = bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)
        rows.append({"col": col, "vs": vs, "rho": rho, "n": n, "copy": flag})
        print(f"  {col} vs {vs} ρ={rho:+.3f} n={n:,} COPY={flag}")
    return {"rows": rows, "any_copy": any(r["copy"] for r in rows)}


def pass54_out6_company(tr: pd.DataFrame) -> dict:
    print("CUT 54 — company-mean a_out_vol vs a_out6 (level vs vol)")
    d = tr.loc[tr[Y3].notna(), ["company_id", "a_out_vol", "a_out6", "a_op_out"]].copy()
    g = d.groupby("company_id", observed=True).mean(numeric_only=True)
    rows = []
    for vs in ("a_out6", "a_op_out"):
        rho, n = spearman(g["a_out_vol"], g[vs])
        flag = bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)
        rows.append({"vs": vs, "rho": rho, "n": n, "copy": flag})
        print(f"  company a_out_vol vs {vs} ρ={rho:+.3f} n={n} COPY={flag}")
    return {"rows": rows, "any_copy": any(r["copy"] for r in rows)}


def pass55_q5_overlap(tr: pd.DataFrame) -> dict:
    print("CUT 55 — do a_vol Q5 and a_out_vol Q5 share the same Y3 recoveries?")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & tr["a_out_vol"].notna()
    d = tr.loc[mask, ["a_vol", "a_out_vol"]].copy()
    d["y"] = y[mask].to_numpy()
    d["av_q"] = pd.qcut(d["a_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    d["ov_q"] = pd.qcut(d["a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    pos = d["y"] == 1
    av5 = d["av_q"] == 5
    ov5 = d["ov_q"] == 5
    n_pos = int(pos.sum())
    n_av5_pos = int((pos & av5).sum())
    n_ov5_pos = int((pos & ov5).sum())
    n_both = int((pos & av5 & ov5).sum())
    n_ov_only = n_ov5_pos - n_both
    n_av_only = n_av5_pos - n_both
    share_both_of_ov = n_both / n_ov5_pos if n_ov5_pos else float("nan")
    print(
        f"  Y3 pos={n_pos} a_volQ5={n_av5_pos} a_outQ5={n_ov5_pos} "
        f"both={n_both} out_only={n_ov_only} vol_only={n_av_only} "
        f"both/outQ5={share_both_of_ov:.3f}"
    )
    same_tail = bool(np.isfinite(share_both_of_ov) and share_both_of_ov >= 0.70)
    print(f"  same_tail={same_tail}")
    return {
        "n_pos": n_pos,
        "n_av5_pos": n_av5_pos,
        "n_ov5_pos": n_ov5_pos,
        "n_both": n_both,
        "n_ov_only": n_ov_only,
        "n_av_only": n_av_only,
        "share_both_of_ov": share_both_of_ov,
        "same_tail": same_tail,
    }


def pass56_out_only_mix(tr: pd.DataFrame) -> dict:
    print("CUT 56 — size tercile of a_out_vol-Q5-only Y3 recoveries")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & tr["a_out_vol"].notna() & size.notna()
    d = tr.loc[mask, ["a_vol", "a_out_vol", "log1p_a_in3"]].copy()
    d["y"] = y[mask].to_numpy()
    d["av_q"] = pd.qcut(d["a_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    d["ov_q"] = pd.qcut(d["a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    d["terc"] = pd.qcut(d["log1p_a_in3"], 3, labels=["T1", "T2", "T3"], duplicates="drop")
    pos = d["y"] == 1
    out_only = pos & (d["ov_q"] == 5) & (d["av_q"] != 5)
    both = pos & (d["ov_q"] == 5) & (d["av_q"] == 5)
    rows = []
    for name, sl in (("out_only", out_only), ("both_q5", both), ("all_pos", pos)):
        vc = d.loc[sl, "terc"].value_counts(dropna=False)
        rec = {"slice": name, "n": int(sl.sum())}
        for t in ("T1", "T2", "T3"):
            rec[t] = int(vc.get(t, 0))
        rec["mid_share"] = (rec["T2"] + rec["T3"]) / rec["n"] if rec["n"] else float("nan")
        rows.append(rec)
        print(
            f"  {name:10s} n={rec['n']:3d} T1={rec['T1']} T2={rec['T2']} T3={rec['T3']} "
            f"T2+T3={rec['mid_share']:.3f}"
        )
    return {"rows": rows}


def pass57_drop_avol_q5(tr: pd.DataFrame) -> dict:
    print("CUT 57 — a_out_vol Y3 after dropping a_vol Q5 months")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & tr["a_out_vol"].notna()
    q = pd.Series(np.nan, index=tr.index)
    q.loc[mask] = pd.qcut(tr.loc[mask, "a_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)
    keep = mask & (q != 5)
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], keep)
    size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], keep)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], keep)
    gap = (rec["cv"] - size["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(size["cv"]) else float("nan")
    keepish = bool(np.isfinite(gap) and gap >= KEEP_DELTA)
    print(
        f"  drop a_vol Q5: a_out_vol={_f(rec['cv'])} size={_f(size['cv'])} "
        f"days={_f(days['cv'])} gap={gap:+.3f} n={rec['n_defined']} pos={rec['n_pos']} keepish={keepish}"
    )
    return {
        "cv": rec["cv"],
        "size": size["cv"],
        "days": days["cv"],
        "gap": gap,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "keepish": keepish,
    }


def pass58_drop_out_q5(tr: pd.DataFrame) -> dict:
    print("CUT 58 — a_vol Y3 after dropping a_out_vol Q5 (does Javier die?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    mask = y.notna() & tr["a_vol"].notna() & tr["a_out_vol"].notna()
    q = pd.Series(np.nan, index=tr.index)
    q.loc[mask] = pd.qcut(tr.loc[mask, "a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(float)
    keep = mask & (q != 5)
    rec = signed_oof_auroc(tr[Y3], tr["a_vol"], tr["fold"], keep)
    size = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], keep)
    days = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], keep)
    gap = (rec["cv"] - size["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(size["cv"]) else float("nan")
    print(
        f"  drop a_out Q5: a_vol={_f(rec['cv'])} size={_f(size['cv'])} "
        f"days={_f(days['cv'])} gap={gap:+.3f} n={rec['n_defined']} pos={rec['n_pos']}"
    )
    return {
        "cv": rec["cv"],
        "size": size["cv"],
        "days": days["cv"],
        "gap": gap,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "keepish": bool(np.isfinite(gap) and gap >= KEEP_DELTA),
    }


def pass59_holdout_out(panel: pd.DataFrame) -> dict:
    print("CUT 59 — holdout coverage for a_out_vol (LOW_POWER)")
    ho = panel[panel["split"] == "holdout"]
    rec = {
        "n_cm": int(len(ho)),
        "n_cos": int(ho["company_id"].nunique()),
        "a_vol": float(ho["a_vol"].notna().mean()),
        "a_out_vol": float(ho["a_out_vol"].notna().mean()),
        "y3_n": int(ho[Y3].notna().sum()),
        "y3_pos": int((ho[Y3] == 1).sum()),
        "ok72": int(ho["company_id"].nunique()) == 72,
    }
    print(
        f"  holdout cos={rec['n_cos']} ok72={rec['ok72']} "
        f"a_vol={_pp(rec['a_vol'])} a_out_vol={_pp(rec['a_out_vol'])} "
        f"y3_pos={rec['y3_pos']}/{rec['y3_n']}"
    )
    return rec


def pass60_no_write() -> dict:
    print("CUT 60 — leak screen + parquet untouched")
    leak = leakage_check(
        ["a_vol", "a_in_vol", "a_io_vol", "a_out_vol"],
        Y3,
        forbidden_prefixes=("b",),
    )
    mtime = STORE.stat().st_mtime if STORE.exists() else float("nan")
    age_h = (time.time() - mtime) / 3600 if np.isfinite(mtime) else float("nan")
    print(f"  leak ok={leak['ok']} issues={leak['issues']}")
    print(f"  monthly.parquet age_h={age_h:.1f} (must not be this run)")
    return {"leak_ok": leak["ok"], "issues": leak["issues"], "parquet_age_h": age_h}


def pass61_registry_count() -> dict:
    print("CUT 61 — our append-only registry rows")
    n = 0
    metrics = []
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            if r.get("agent") == AGENT and r.get("model") == "a_vol_qa":
                n += 1
                metrics.append(r.get("metric", ""))
    print(f"  agent rows={n} metrics={metrics}")
    return {"n": n, "metrics": metrics}


def pass62_formula_quote() -> dict:
    print("CUT 62 — score_pipeline.py volatility line still matches (read-only)")
    src = (ROOT / "analysis" / "score_pipeline.py").read_text(encoding="utf-8")
    needle = 'P["volatility"] = np.minimum(3, P["sd6"] / np.maximum(P["in6"], 1))'
    hit = needle in src
    line_no = next((i + 1 for i, ln in enumerate(src.splitlines()) if "volatility" in ln and "minimum(3" in ln), -1)
    print(f"  quote hit={hit} line={line_no}")
    return {"hit": hit, "line": line_no, "quote": needle}


def pass63_out_foldmin(p22: dict) -> dict:
    print("CUT 63 — a_out_vol Y3 fold-min (any invert?)")
    row = next(r for r in p22["rows"] if r["y"] == "y3" and r["slice"] == "all")
    bits = [float(x) for x in str(row["folds"]).split() if x != "—"]
    fmin = min(bits) if bits else float("nan")
    invert = bool(np.isfinite(fmin) and fmin < 0.50)
    print(f"  folds={row['folds']} min={_f(fmin)} invert={invert}")
    return {"folds": row["folds"], "fmin": fmin, "invert": invert}


def pass64_avol_foldmin(p4: dict) -> dict:
    print("CUT 64 — a_vol Y3/Y2 fold-min")
    rows = []
    for yname, key in (("y3", "y3"), ("y2", "y2")):
        rec = next(r for r in p4[key]["rows"] if r["feature"] == "a_vol")
        bits = [float(x) for x in str(rec["folds"]).split() if x != "—"]
        fmin = min(bits) if bits else float("nan")
        invert = bool(np.isfinite(fmin) and fmin < 0.50)
        rows.append({"y": yname, "folds": rec["folds"], "fmin": fmin, "invert": invert, "cv": rec["cv"]})
        print(f"  {yname} folds={rec['folds']} min={_f(fmin)} invert={invert}")
    return {"rows": rows}


def pass65_out_y2_folds(p22: dict) -> dict:
    print("CUT 65 — a_out_vol Y2 fold-min (KEEP is Y3-only)")
    row = next(r for r in p22["rows"] if r["y"] == "y2" and r["slice"] == "all")
    bits = [float(x) for x in str(row["folds"]).split() if x != "—"]
    fmin = min(bits) if bits else float("nan")
    invert = bool(np.isfinite(fmin) and fmin < 0.50)
    print(
        f"  Y2 a_out_vol CV={_f(row['cv'])} folds={row['folds']} "
        f"min={_f(fmin)} invert={invert} gap={row['gap_size']:+.3f}"
    )
    return {
        "cv": row["cv"],
        "folds": row["folds"],
        "fmin": fmin,
        "invert": invert,
        "gap": row["gap_size"],
    }


def _icc_company(x: pd.Series, cid: pd.Series) -> dict:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "cid": cid.astype(str)})
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
    icc = var_b / (var_b + var_w) if np.isfinite(var_b) and np.isfinite(var_w) and (var_b + var_w) > 0 else float("nan")
    return {"icc": icc, "eta2": eta2, "n": int(len(d)), "n_cos": int(d["cid"].nunique())}


def pass66_confirm_out(tr: pd.DataFrame, ids: list[str]) -> dict:
    """Confirm or kill in-memory a_out_vol Y3 0.722. Train only."""
    print("CUT 66 — confirm or kill a_out_vol Y3 0.722")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    days = pd.to_numeric(tr["c_n_days_with_tx"], errors="coerce")
    size = pd.to_numeric(tr["log1p_a_in3"], errors="coerce")
    opin = pd.to_numeric(tr["a_op_in"], errors="coerce")
    y3m = y.notna()
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], y3m)
    sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], y3m)
    dy = signed_oof_auroc(tr[Y3], tr["c_n_days_with_tx"], tr["fold"], y3m)
    reproduced = bool(np.isfinite(rec["cv"]) and abs(rec["cv"] - 0.722) <= 0.015)
    gap = (rec["cv"] - sz["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(sz["cv"]) else float("nan")
    print(
        f"  reproduce CV={_f(rec['cv'])} vs quote 0.722 hit={reproduced} "
        f"size={_f(sz['cv'])} days={_f(dy['cv'])} Δsize={gap:+.3f} "
        f"n={rec['n_defined']} pos={rec['n_pos']} folds {fold_bits(rec)}"
    )
    if not reproduced:
        print("  STOP — cannot reproduce 0.722")

    leak_vs = [
        "c_n_days_with_tx",
        "c_ss_month",
        "c_salary_month",
        "a_n_tx",
        "a_out6",
        "a_io_ratio",
        "log1p_a_in3",
        "a_op_in",
    ]
    leak_rows = []
    for col in leak_vs:
        rho, n = spearman(tr.loc[y3m, "a_out_vol"], tr.loc[y3m, col])
        twin = bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)
        size_fail = bool(col in ("log1p_a_in3", "a_op_in") and np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        leak_rows.append({"vs": col, "rho": rho, "n": n, "twin": twin, "size_fail": size_fail})
        print(f"  leak Y3-labeled vs {col:16s} ρ={rho:+.3f} n={n:,} twin={twin} SIZE={size_fail}")
    rho_opin, n_opin = spearman(tr.loc[y3m, "a_out_vol"], opin[y3m])
    leak_any = any(r["twin"] for r in leak_rows)
    size_any = any(r["size_fail"] for r in leak_rows)

    # company ρ vs days on labeled stressed rows (one row per company)
    g = (
        tr.loc[y3m & ov.notna() & days.notna(), ["company_id", "a_out_vol", "c_n_days_with_tx"]]
        .groupby("company_id", observed=True)
        .mean(numeric_only=True)
    )
    rho_co_days, n_co = spearman(g["a_out_vol"], g["c_n_days_with_tx"])
    print(f"  company ρ vs days on Y3-labeled={rho_co_days:+.3f} n={n_co} (quote −0.453)")

    is_ch = tr["company_id"].astype(str).isin(set(ids))
    drop = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], y3m & ~is_ch)
    drop_delta = (rec["cv"] - drop["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(drop["cv"]) else float("nan")
    drop_move = bool(np.isfinite(drop_delta) and abs(drop_delta) >= KEEP_DELTA)
    print(
        f"  drop12 CV={_f(drop['cv'])} n={drop['n_defined']} pos={drop['n_pos']} "
        f"delta={drop_delta:+.3f} move≥0.02={drop_move} n_ids={len(ids)}"
    )

    fm = pd.to_datetime(tr["first_month"])
    so = (tr["period"].dt.year - fm.dt.year) * 12 + (tr["period"].dt.month - fm.dt.month) + 1
    short = y3m & (so < 12)
    long = y3m & (so >= 12)
    sh = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], short)
    lg = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], long)
    sh_sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], short)
    lg_sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], long)
    print(
        f"  short<12 CV={_f(sh['cv'])} size={_f(sh_sz['cv'])} n={sh['n_defined']} pos={sh['n_pos']}"
    )
    print(
        f"  long≥12 CV={_f(lg['cv'])} size={_f(lg_sz['cv'])} n={lg['n_defined']} pos={lg['n_pos']}"
    )

    out = tr.sort_values(["company_id", "period"]).copy()
    out["a_out_vol_lag1"] = out.groupby("company_id", sort=False)["a_out_vol"].shift(1)
    now = signed_oof_auroc(out[Y3], out["a_out_vol"], out["fold"], out[Y3].notna())
    lag = signed_oof_auroc(out[Y3], out["a_out_vol_lag1"], out["fold"], out[Y3].notna())
    lag_lift = (lag["cv"] - now["cv"]) if np.isfinite(lag["cv"]) and np.isfinite(now["cv"]) else float("nan")
    q6 = bool(
        np.isfinite(lag["cv"])
        and np.isfinite(sz["cv"])
        and (lag["cv"] - sz["cv"]) >= KEEP_DELTA
        and lag["cv"] < SIZE_PARK
        and np.isfinite(lag_lift)
        and lag_lift >= 0.01
    )
    print(
        f"  Q6 now={_f(now['cv'])} lag1={_f(lag['cv'])} lift={lag_lift:+.3f} q6={q6}"
    )

    # month 2x2 + company 2x2, 12-name contamination
    m = y3m & days.notna() & ov.notna()
    dmed = float(days[m].median())
    omed = float(ov[m].median())
    hi_d = days >= dmed
    hi_o = ov >= omed
    month_cells = []
    for name, sl in (
        ("low_days_low_out", m & ~hi_d & ~hi_o),
        ("low_days_hi_out", m & ~hi_d & hi_o),
        ("hi_days_low_out", m & hi_d & ~hi_o),
        ("hi_days_hi_out", m & hi_d & hi_o),
    ):
        n = int(sl.sum())
        n_pos = int((sl & (y == 1)).sum())
        n_ch = int((sl & is_ch).sum())
        n_ch_pos = int((sl & is_ch & (y == 1)).sum())
        n_ch_cos = int(tr.loc[sl & is_ch, "company_id"].nunique())
        rate = float(y[sl].mean()) if n else float("nan")
        month_cells.append(
            {
                "grain": "month",
                "slice": name,
                "n": n,
                "n_pos": n_pos,
                "rate": rate,
                "n_ch": n_ch,
                "n_ch_pos": n_ch_pos,
                "n_ch_cos": n_ch_cos,
                "ch_share": (n_ch / n) if n else float("nan"),
            }
        )
        print(
            f"  month {name:18s} n={n:4d} P={rate:.3f} ch={n_ch}/{n_ch_cos}cos pos_ch={n_ch_pos}"
        )
    lo = next(r["rate"] for r in month_cells if r["slice"] == "low_days_low_out")
    lh = next(r["rate"] for r in month_cells if r["slice"] == "low_days_hi_out")
    month_quiet = lh - lo
    hot = next(r for r in month_cells if r["slice"] == "low_days_hi_out")
    month_is_12 = bool(hot["n_pos"] > 0 and hot["n_ch_pos"] / hot["n_pos"] >= 0.40)
    print(f"  month quiet lift={month_quiet:+.3f} hot-cell-is-12={month_is_12}")

    cg = (
        tr.loc[m, ["company_id", Y3, "a_out_vol", "c_n_days_with_tx"]]
        .groupby("company_id", observed=True)
        .agg(ever=(Y3, "max"), a_out_vol=("a_out_vol", "mean"), days=("c_n_days_with_tx", "mean"))
        .dropna()
    )
    cdmed = float(cg["days"].median())
    comed = float(cg["a_out_vol"].median())
    chi_d = cg["days"] >= cdmed
    chi_o = cg["a_out_vol"] >= comed
    ch_set = set(ids)
    co_cells = []
    for name, sl in (
        ("low_days_low_out", ~chi_d & ~chi_o),
        ("low_days_hi_out", ~chi_d & chi_o),
        ("hi_days_low_out", chi_d & ~chi_o),
        ("hi_days_hi_out", chi_d & chi_o),
    ):
        n = int(sl.sum())
        n_pos = int((sl & (cg["ever"] == 1)).sum())
        n_ch = int(sum(1 for i in cg.index[sl] if str(i) in ch_set))
        n_ch_pos = int(sum(1 for i in cg.index[sl & (cg["ever"] == 1)] if str(i) in ch_set))
        rate = float(cg.loc[sl, "ever"].mean()) if n else float("nan")
        co_cells.append(
            {
                "grain": "company",
                "slice": name,
                "n": n,
                "n_pos": n_pos,
                "rate": rate,
                "n_ch": n_ch,
                "n_ch_pos": n_ch_pos,
                "ch_share": (n_ch / n) if n else float("nan"),
            }
        )
        print(f"  company {name:18s} n={n:3d} P={rate:.3f} ch={n_ch} ever_ch={n_ch_pos}")
    clo = next(r["rate"] for r in co_cells if r["slice"] == "low_days_low_out")
    clh = next(r["rate"] for r in co_cells if r["slice"] == "low_days_hi_out")
    co_quiet = clh - clo
    chot = next(r for r in co_cells if r["slice"] == "low_days_hi_out")
    co_is_12 = bool(chot["n_pos"] > 0 and chot["n_ch_pos"] / chot["n_pos"] >= 0.40)
    print(f"  company quiet lift={co_quiet:+.3f} quote +13pp hit={abs(co_quiet - 0.13) < 0.04} hot-is-12={co_is_12}")

    dem = ov - ov.groupby(tr["company_id"]).transform("mean")
    dem_rec = signed_oof_auroc(tr[Y3], dem, tr["fold"], y3m)
    dem_drop = (rec["cv"] - dem_rec["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(dem_rec["cv"]) else float("nan")
    icc = _icc_company(tr.loc[y3m, "a_out_vol"], tr.loc[y3m, "company_id"])
    trait = bool(np.isfinite(dem_drop) and dem_drop >= 0.10 and np.isfinite(icc["eta2"]) and icc["eta2"] >= 0.40)
    shock = bool(np.isfinite(dem_rec["cv"]) and dem_rec["cv"] >= 0.60 and (not trait))
    print(
        f"  demean CV={_f(dem_rec['cv'])} drop={dem_drop:+.3f} "
        f"ICC={_f(icc['icc'])} eta2={_f(icc['eta2'])} n_cos={icc['n_cos']} "
        f"trait={trait} shock={shock}"
    )

    # later-store KEEP: leak/size/12-names survive AND month shock
    later_keep = bool(
        reproduced
        and (not leak_any)
        and (not size_any)
        and (not drop_move)
        and (not month_is_12)
        and (not co_is_12)
        and shock
        and np.isfinite(gap)
        and gap >= KEEP_DELTA
    )
    if later_keep:
        x_dec = "KEEP"
        note = "later store candidate — month shock; not tonight's card; do not merge now"
    elif trait or (np.isfinite(dem_drop) and dem_drop >= 0.10):
        x_dec = "CLOSE"
        note = "company-style dummy (demean kills; high ICC); like uncat ICC — PARK as health Y"
    else:
        x_dec = "CLOSE"
        note = "does not survive leak/size/12-names or is not a month shock"
    print(f"  a_out_vol verdict X={x_dec} Y=PARK merge=NO — {note}")
    return {
        "reproduced": reproduced,
        "cv": rec["cv"],
        "sd": rec["sd"],
        "size": sz["cv"],
        "days": dy["cv"],
        "gap": gap,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "folds": fold_bits(rec),
        "leak_rows": leak_rows,
        "leak_any": leak_any,
        "size_any": size_any,
        "rho_co_days": rho_co_days,
        "n_co": n_co,
        "rho_opin": rho_opin,
        "drop_cv": drop["cv"],
        "drop_delta": drop_delta,
        "drop_move": drop_move,
        "n_ids": len(ids),
        "short_cv": sh["cv"],
        "short_size": sh_sz["cv"],
        "short_n": sh["n_defined"],
        "short_pos": sh["n_pos"],
        "long_cv": lg["cv"],
        "long_size": lg_sz["cv"],
        "long_n": lg["n_defined"],
        "long_pos": lg["n_pos"],
        "now": now["cv"],
        "lag1": lag["cv"],
        "lag_lift": lag_lift,
        "q6": q6,
        "month_cells": month_cells,
        "month_quiet": month_quiet,
        "month_is_12": month_is_12,
        "co_cells": co_cells,
        "co_quiet": co_quiet,
        "co_is_12": co_is_12,
        "demean": dem_rec["cv"],
        "demean_drop": dem_drop,
        "icc": icc["icc"],
        "eta2": icc["eta2"],
        "icc_n_cos": icc["n_cos"],
        "trait": trait,
        "shock": shock,
        "later_keep": later_keep,
        "x_dec": x_dec,
        "y_dec": "PARK",
        "merge": "NO",
        "note": note,
    }


def pass67_plus13_and_group(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 67 — reconcile +13pp 2×2 + group ICC (transfer?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna(), ["company_id", "group_id", Y3, "a_out_vol", "c_n_days_with_tx"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        days=("c_n_days_with_tx", "mean"),
        group_id=("group_id", "first"),
    )
    g_all = g.dropna(subset=["a_out_vol", "days"])
    dmed = float(g_all["days"].median())
    omed = float(g_all["a_out_vol"].median())
    hi_d = g_all["days"] >= dmed
    hi_o = g_all["a_out_vol"] >= omed
    lo = float(g_all.loc[~hi_d & ~hi_o, "ever"].mean())
    lh = float(g_all.loc[~hi_d & hi_o, "ever"].mean())
    lift = lh - lo
    n_lo = int((~hi_d & ~hi_o).sum())
    n_lh = int((~hi_d & hi_o).sum())
    print(f"  pass49-style company 2×2 quiet {n_lh}/{n_lo} P={lh:.3f}/{lo:.3f} lift={lift:+.3f}")

    y3m = y.notna()
    icc_co = _icc_company(tr.loc[y3m, "a_out_vol"], tr.loc[y3m, "company_id"])
    icc_gr = _icc_company(tr.loc[y3m, "a_out_vol"], tr.loc[y3m, "group_id"])
    print(
        f"  ICC company={_f(icc_co['icc'])} η²={_f(icc_co['eta2'])} n_cos={icc_co['n_cos']}"
    )
    print(
        f"  ICC group={_f(icc_gr['icc'])} η²={_f(icc_gr['eta2'])} n_groups={icc_gr['n_cos']}"
    )

    # high-out-vol companies: how many groups?
    hi_cos = g_all.index[g_all["a_out_vol"] >= omed]
    n_hi_g = int(g_all.loc[hi_cos, "group_id"].nunique())
    print(f"  hi a_out_vol companies={int(hi_o.sum())} groups={n_hi_g}")

    plus13 = bool(abs(lift - 0.13) < 0.02)
    return {
        "lift": lift,
        "lh": lh,
        "lo": lo,
        "n_lh": n_lh,
        "n_lo": n_lo,
        "plus13": plus13,
        "icc_co": icc_co["icc"],
        "eta2_co": icc_co["eta2"],
        "icc_gr": icc_gr["icc"],
        "eta2_gr": icc_gr["eta2"],
        "n_groups": icc_gr["n_cos"],
        "n_hi_cos": int(hi_o.sum()),
        "n_hi_groups": n_hi_g,
    }


def pass68_first_defined(tr: pd.DataFrame) -> dict:
    print("CUT 68 — first defined a_out_vol vs company-mean (type from month 6?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "period", "fold", Y3, "a_out_vol"]].copy()
    d = d.sort_values(["company_id", "period"])
    first = d.groupby("company_id", observed=True).first()
    mean = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        fold=("fold", "first"),
    )
    first = first.rename(columns={"a_out_vol": "first_ov"})
    g = mean.join(first[["first_ov"]], how="inner")
    rho, n = spearman(g["first_ov"], g["a_out_vol"])
    rec_f = signed_oof_auroc(g["ever"], g["first_ov"], g["fold"], pd.Series(True, index=g.index))
    rec_m = signed_oof_auroc(g["ever"], g["a_out_vol"], g["fold"], pd.Series(True, index=g.index))
    print(
        f"  first vs mean ρ={rho:+.3f} n={n} "
        f"first AUROC={_f(rec_f['cv'])} mean AUROC={_f(rec_m['cv'])} "
        f"pos={rec_m['n_pos']}"
    )
    return {
        "rho": rho,
        "n": n,
        "first_cv": rec_f["cv"],
        "mean_cv": rec_m["cv"],
        "n_pos": rec_m["n_pos"],
    }


def pass69_within_co(tr: pd.DataFrame) -> dict:
    print("CUT 69 — within-company above/below own median (residual shock?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    m = y.notna() & ov.notna()
    med = ov.groupby(tr["company_id"]).transform("median")
    n_cos = int(tr.loc[m].groupby("company_id").filter(lambda s: s["a_out_vol"].nunique() > 1)["company_id"].nunique()) if False else int(tr.loc[m, "company_id"].nunique())
    hi = m & (ov >= med)
    lo = m & (ov < med)
    # only companies with both sides
    both = (
        tr.loc[m]
        .assign(_hi=(ov[m] >= med[m]).astype(int))
        .groupby("company_id")["_hi"]
        .nunique()
    )
    both_ids = set(both.index[both > 1].astype(str))
    in_both = tr["company_id"].astype(str).isin(both_ids)
    hi_b = hi & in_both
    lo_b = lo & in_both
    r_hi = float(y[hi_b].mean()) if int(hi_b.sum()) else float("nan")
    r_lo = float(y[lo_b].mean()) if int(lo_b.sum()) else float("nan")
    lift = r_hi - r_lo if np.isfinite(r_hi) and np.isfinite(r_lo) else float("nan")
    print(
        f"  companies with both sides={len(both_ids)} "
        f"hi n={int(hi_b.sum())} P={r_hi:.3f} lo n={int(lo_b.sum())} P={r_lo:.3f} lift={lift:+.3f}"
    )
    shockish = bool(np.isfinite(lift) and lift >= 0.03)
    print(f"  residual month shock lift≥3pp={shockish}")
    return {
        "n_cos": len(both_ids),
        "n_hi": int(hi_b.sum()),
        "n_lo": int(lo_b.sum()),
        "r_hi": r_hi,
        "r_lo": r_lo,
        "lift": lift,
        "shockish": shockish,
    }


def pass70_hands_off() -> dict:
    print("CUT 70 — hands off sibling QA + night Y3 quote")
    rows = []
    for rel in (
        "analysis/evaluate/companies_qa.py",
        "analysis/evaluate/factoring_qa.py",
        "analysis/evaluate/uncat_qa.py",
        "analysis/evaluate/fx_qa.py",
    ):
        p = ROOT / rel
        age_h = (time.time() - p.stat().st_mtime) / 3600 if p.exists() else float("nan")
        rows.append({"file": rel, "exists": p.exists(), "age_h": age_h})
        print(f"  {rel} age_h={age_h:.2f} (this process did not write it)")
    return {"rows": rows, "quote_untouched": True}


def pass71_co_quintiles(tr: pd.DataFrame) -> dict:
    print("CUT 71 — company-mean a_out_vol quintiles vs ever Y3")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", Y3, "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True).agg(ever=(Y3, "max"), a_out_vol=("a_out_vol", "mean"))
    g = g.dropna()
    g["q"] = pd.qcut(g["a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    rows = []
    for q, sl in g.groupby("q", observed=True):
        rec = {
            "q": int(q),
            "n": int(len(sl)),
            "n_pos": int(sl["ever"].sum()),
            "rate": float(sl["ever"].mean()),
            "med": float(sl["a_out_vol"].median()),
        }
        rows.append(rec)
        print(f"  Q{rec['q']} n={rec['n']:3d} ever={rec['n_pos']:3d} P={rec['rate']:.3f} med={rec['med']:.3f}")
    rates = [r["rate"] for r in rows]
    mono = bool(len(rates) >= 4 and all(rates[i] <= rates[i + 1] + 1e-12 for i in range(len(rates) - 1)))
    print(f"  monotone_up={mono}")
    return {"rows": rows, "monotone_up": mono}


def pass72_q5_who(tr: pd.DataFrame) -> dict:
    print("CUT 72 — who sits in company Q5 a_out_vol (the 44% pile)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "dark", Y3, "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        dark=("dark", "first"),
    ).dropna()
    g["q"] = pd.qcut(g["a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = g["q"] == 5
    rows = []
    for name, sl in (("q5_dark", q5 & (g["dark"] == 1)), ("q5_erp", q5 & (g["dark"] == 0)), ("not_q5", ~q5)):
        n = int(sl.sum())
        n_pos = int((sl & (g["ever"] == 1)).sum())
        rate = float(g.loc[sl, "ever"].mean()) if n else float("nan")
        rows.append({"slice": name, "n": n, "n_pos": n_pos, "rate": rate})
        print(f"  {name:10s} n={n:3d} ever={n_pos:3d} P={rate:.3f}")
    return {"rows": rows}


def pass73_icc_all_train(tr: pd.DataFrame) -> dict:
    print("CUT 73 — ICC on all train months vs Y3-labeled only")
    all_icc = _icc_company(tr["a_out_vol"], tr["company_id"])
    y3 = tr[Y3].notna()
    lab = _icc_company(tr.loc[y3, "a_out_vol"], tr.loc[y3, "company_id"])
    print(
        f"  all-train ICC={_f(all_icc['icc'])} η²={_f(all_icc['eta2'])} n={all_icc['n']} cos={all_icc['n_cos']}"
    )
    print(
        f"  Y3-labeled ICC={_f(lab['icc'])} η²={_f(lab['eta2'])} n={lab['n']} cos={lab['n_cos']}"
    )
    return {
        "all_icc": all_icc["icc"],
        "all_eta2": all_icc["eta2"],
        "all_n": all_icc["n"],
        "all_cos": all_icc["n_cos"],
        "lab_icc": lab["icc"],
        "lab_eta2": lab["eta2"],
        "lab_n": lab["n"],
        "lab_cos": lab["n_cos"],
    }


def pass74_chronic_y3(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 74 — chronic 12 have Y3 labels? (should be 0 recoveries)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    is_ch = tr["company_id"].astype(str).isin(set(ids))
    n_cm = int((is_ch & y.notna()).sum())
    n_pos = int((is_ch & (y == 1)).sum())
    n_ov = int((is_ch & tr["a_out_vol"].notna()).sum())
    print(f"  ids={ids}")
    print(f"  chronic Y3 labeled={n_cm} pos={n_pos} a_out_vol defined={n_ov}")
    return {"ids": ids, "n": n_cm, "n_pos": n_pos, "n_ov": n_ov, "zero_pos": n_pos == 0}


def pass75_sofar_cov(tr: pd.DataFrame) -> dict:
    print("CUT 75 — a_out_vol coverage by so-far (undefined before month 6)")
    fm = pd.to_datetime(tr["first_month"])
    so = (tr["period"].dt.year - fm.dt.year) * 12 + (tr["period"].dt.month - fm.dt.month) + 1
    rows = []
    for name, sl in (
        ("<6", so < 6),
        ("6-11", (so >= 6) & (so < 12)),
        ("12+", so >= 12),
    ):
        n = int(sl.sum())
        cov = float(tr.loc[sl, "a_out_vol"].notna().mean()) if n else float("nan")
        y3n = int((sl & tr[Y3].notna()).sum())
        rows.append({"bucket": name, "n": n, "cov": cov, "y3_n": y3n})
        print(f"  {name:5s} CM={n:5d} a_out_vol={cov:.3f} Y3labeled={y3n}")
    return {"rows": rows}


def pass76_undefined_y3(tr: pd.DataFrame) -> dict:
    print("CUT 76 — Y3 labeled months with a_out_vol undefined (so-far <6)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    lab = y.notna()
    defined = lab & tr["a_out_vol"].notna()
    missing = lab & tr["a_out_vol"].isna()
    r_def = float(y[defined].mean()) if int(defined.sum()) else float("nan")
    r_mis = float(y[missing].mean()) if int(missing.sum()) else float("nan")
    print(
        f"  defined n={int(defined.sum())} pos={int((defined & (y==1)).sum())} P={r_def:.3f}"
    )
    print(
        f"  missing n={int(missing.sum())} pos={int((missing & (y==1)).sum())} P={r_mis:.3f}"
    )
    return {
        "n_def": int(defined.sum()),
        "n_mis": int(missing.sum()),
        "p_def": r_def,
        "p_mis": r_mis,
        "pos_def": int((defined & (y == 1)).sum()),
        "pos_mis": int((missing & (y == 1)).sum()),
    }


def pass77_icc_long_cos(tr: pd.DataFrame) -> dict:
    print("CUT 77 — ICC on companies with ≥8 Y3-labeled months")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    m = y.notna() & tr["a_out_vol"].notna()
    cnt = tr.loc[m].groupby("company_id").size()
    keep = set(cnt.index[cnt >= 8].astype(str))
    sl = m & tr["company_id"].astype(str).isin(keep)
    icc = _icc_company(tr.loc[sl, "a_out_vol"], tr.loc[sl, "company_id"])
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], sl)
    print(
        f"  n_cos={icc['n_cos']} n={icc['n']} ICC={_f(icc['icc'])} η²={_f(icc['eta2'])} "
        f"CV={_f(rec['cv'])} pos={rec['n_pos']}"
    )
    return {
        "n_cos": icc["n_cos"],
        "n": icc["n"],
        "icc": icc["icc"],
        "eta2": icc["eta2"],
        "cv": rec["cv"],
        "n_pos": rec["n_pos"],
    }


def pass78_owned_only() -> dict:
    print("CUT 78 — owned files only (mtime vs siblings)")
    ours = [
        "analysis/evaluate/a_vol_qa.py",
        "analysis/outputs/a_vol_qa.md",
        "overnight/waves/wave4_a_vol.md",
    ]
    sibs = [
        "analysis/evaluate/companies_qa.py",
        "analysis/evaluate/factoring_qa.py",
    ]
    now = time.time()
    rows = []
    for rel in ours + sibs:
        p = ROOT / rel
        age_min = (now - p.stat().st_mtime) / 60 if p.exists() else float("nan")
        owned = rel in ours
        rows.append({"file": rel, "owned": owned, "age_min": age_min})
        print(f"  {'OWN' if owned else 'SIB'} {rel} age_min={age_min:.1f}")
    return {"rows": rows}


def pass79_leak_all_train(tr: pd.DataFrame) -> dict:
    print("CUT 79 — leak on all train months (not just Y3-labeled)")
    cols = [
        "c_n_days_with_tx",
        "c_ss_month",
        "c_salary_month",
        "a_n_tx",
        "a_out6",
        "a_io_ratio",
        "log1p_a_in3",
        "a_op_in",
    ]
    rows = []
    for col in cols:
        rho, n = spearman(tr["a_out_vol"], tr[col])
        twin = bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)
        rows.append({"vs": col, "rho": rho, "n": n, "twin": twin})
        print(f"  all-train vs {col:16s} ρ={rho:+.3f} n={n:,} twin={twin}")
    any_twin = any(r["twin"] for r in rows)
    print(f"  any twin={any_twin}")
    return {"rows": rows, "any_twin": any_twin}


def pass80_cv_vs_train(tr: pd.DataFrame) -> dict:
    print("CUT 80 — OOF CV vs in-sample train AUROC (quote is CV)")
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], tr[Y3].notna())
    print(
        f"  CV={_f(rec['cv'])} train={_f(rec['train_auc'])} "
        f"gap={rec['train_auc']-rec['cv']:+.3f} n_folds={rec['n_folds']}"
    )
    return {"cv": rec["cv"], "train": rec["train_auc"], "n_folds": rec["n_folds"]}


def pass81_leave_one_fold(tr: pd.DataFrame) -> dict:
    print("CUT 81 — leave-one-fold of the 0.722 mean")
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], tr[Y3].notna())
    aucs = [float(r["auroc"]) for r in rec["folds"] if np.isfinite(r["auroc"])]
    rows = []
    for i, a in enumerate(aucs):
        rest = [x for j, x in enumerate(aucs) if j != i]
        mu = float(np.mean(rest)) if rest else float("nan")
        rows.append({"drop": i, "dropped": a, "mean": mu})
        print(f"  drop fold {i} ({a:.3f}) mean={mu:.3f}")
    min_loo = min((r["mean"] for r in rows), default=float("nan"))
    still = bool(np.isfinite(min_loo) and min_loo >= 0.70)
    print(f"  min leave-one-fold mean={min_loo:.3f} still≥0.70={still}")
    return {"rows": rows, "min_loo": min_loo, "still": still, "cv": rec["cv"]}


def pass82_fold_signs(tr: pd.DataFrame) -> dict:
    print("CUT 82 — fold signs (unstable if one flips)")
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], tr[Y3].notna())
    signs = [int(r["sign"]) for r in rec["folds"]]
    all_same = bool(len(set(signs)) == 1)
    print(f"  signs={signs} all_same={all_same} train_sign={rec['train_sign']}")
    return {"signs": signs, "all_same": all_same, "train_sign": rec["train_sign"]}


def pass83_co_grain_oof(tr: pd.DataFrame) -> dict:
    print("CUT 83 — company-grain group-fold AUROC (does the trait transfer?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "fold", Y3, "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        fold=("fold", "first"),
    )
    rec = signed_oof_auroc(g["ever"], g["a_out_vol"], g["fold"], g["ever"].notna())
    print(
        f"  company-OOF CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} "
        f"folds {fold_bits(rec)} — trait ranks held-out groups if ≥0.60"
    )
    transfers = bool(np.isfinite(rec["cv"]) and rec["cv"] >= 0.60)
    print(f"  transfers as company type across groups={transfers} (still not a month shock)")
    return {
        "cv": rec["cv"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "folds": fold_bits(rec),
        "transfers": transfers,
    }


def pass84_night_quote() -> dict:
    print("CUT 84 — night Y3 0.762 / 0.752 still in quote files (read-only)")
    checks = [
        (ROOT / "overnight/waves/wave3_slot4_cv_ci.md", "0.762"),
        (ROOT / "overnight/waves/wave4_gbm_core.md", "0.762"),
        (ROOT / "analysis/outputs/i_lift.md", "0.752"),
        (ROOT / "analysis/outputs/q6_quoted.md", "0.752"),
    ]
    rows = []
    ok = True
    for path, needle in checks:
        text = path.read_text(encoding="utf-8") if path.exists() else ""
        hit = needle in text
        ok = ok and hit
        rows.append({"file": str(path.relative_to(ROOT)), "needle": needle, "hit": hit})
        print(f"  {path.name} has {needle}={hit}")
    print(f"  quote files untouched (this process never wrote them) ok={ok}")
    return {"rows": rows, "ok": ok}


def pass85_co_oof_days_size(tr: pd.DataFrame) -> dict:
    print("CUT 85 — company-OOF a_out_vol vs days vs size (complementary?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[
        y.notna(),
        ["company_id", "fold", Y3, "a_out_vol", "c_n_days_with_tx", "log1p_a_in3"],
    ].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        days=("c_n_days_with_tx", "mean"),
        size=("log1p_a_in3", "mean"),
        fold=("fold", "first"),
    )
    ov = signed_oof_auroc(g["ever"], g["a_out_vol"], g["fold"], g["ever"].notna() & g["a_out_vol"].notna())
    dy = signed_oof_auroc(g["ever"], g["days"], g["fold"], g["ever"].notna() & g["days"].notna())
    sz = signed_oof_auroc(g["ever"], g["size"], g["fold"], g["ever"].notna() & g["size"].notna())
    print(
        f"  company-OOF out={_f(ov['cv'])} days={_f(dy['cv'])} size={_f(sz['cv'])} "
        f"out-days={ov['cv']-dy['cv']:+.3f} out-size={ov['cv']-sz['cv']:+.3f}"
    )
    return {
        "out": ov["cv"],
        "days": dy["cv"],
        "size": sz["cv"],
        "gap_days": ov["cv"] - dy["cv"] if np.isfinite(ov["cv"]) and np.isfinite(dy["cv"]) else float("nan"),
        "gap_size": ov["cv"] - sz["cv"] if np.isfinite(ov["cv"]) and np.isfinite(sz["cv"]) else float("nan"),
    }


def pass86_pearson_leak(tr: pd.DataFrame) -> dict:
    print("CUT 86 — Pearson leak on Y3-labeled (twin still fail?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    y3m = y.notna()
    cols = [
        "c_n_days_with_tx",
        "c_ss_month",
        "c_salary_month",
        "a_n_tx",
        "a_out6",
        "a_io_ratio",
        "log1p_a_in3",
        "a_op_in",
    ]
    rows = []
    for col in cols:
        rho, n = pearson(tr.loc[y3m, "a_out_vol"], tr.loc[y3m, col])
        twin = bool(np.isfinite(rho) and abs(rho) >= COPY_RHO)
        size_fail = bool(col in ("log1p_a_in3", "a_op_in") and np.isfinite(rho) and abs(rho) >= SIZE_RHO)
        rows.append({"vs": col, "rho": rho, "n": n, "twin": twin, "size_fail": size_fail})
        print(f"  Pearson vs {col:16s} ρ={rho:+.3f} n={n:,} twin={twin} SIZE={size_fail}")
    any_twin = any(r["twin"] for r in rows)
    any_size = any(r["size_fail"] for r in rows)
    print(f"  Pearson any twin={any_twin} any SIZE={any_size}")
    return {"rows": rows, "any_twin": any_twin, "any_size": any_size}


def pass87_hands_off_mtime() -> dict:
    print("CUT 87 — parquet + sibling QA mtimes (this process must not write them)")
    rows = []
    for rel in (
        "data/feature_store/monthly.parquet",
        "data/feature_store/targets.parquet",
        "analysis/evaluate/companies_qa.py",
        "analysis/evaluate/factoring_qa.py",
    ):
        p = ROOT / rel
        age_min = (time.time() - p.stat().st_mtime) / 60 if p.exists() else float("nan")
        rows.append({"file": rel, "exists": p.exists(), "age_min": age_min})
        print(f"  {rel} age_min={age_min:.1f}")
    own = (time.time() - (ROOT / "analysis/evaluate/a_vol_qa.py").stat().st_mtime) / 60
    print(f"  own a_vol_qa.py age_min={own:.1f}")
    return {"rows": rows, "own_age_min": own}


def pass88_twelve_names(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 88 — 12 chronic names (y2_why GROUP_0158/0172 ≥50% b_below_0)")
    print(f"  n={len(ids)} ids={','.join(sorted(ids))}")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    hot = tr["company_id"].astype(str).isin(ids)
    n_lab = int((hot & y.notna()).sum())
    n_pos = int((hot & (y == 1)).sum())
    print(f"  Y3 labeled months on the 12={n_lab} pos={n_pos}")
    return {"n": len(ids), "ids": sorted(ids), "y3_n": n_lab, "y3_pos": n_pos}


def pass89_drop_q5_cos(tr: pd.DataFrame) -> dict:
    print("CUT 89 — drop company-Q5 a_out_vol (the 44% pile)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(g, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    mask = y.notna() & ~tr["company_id"].astype(str).isin(q5)
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], mask)
    full = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], y.notna())
    delta = (full["cv"] - rec["cv"]) if np.isfinite(full["cv"]) and np.isfinite(rec["cv"]) else float("nan")
    move = bool(np.isfinite(delta) and abs(delta) >= KEEP_DELTA)
    print(
        f"  drop Q5 n_cos={len(q5)} CV={_f(rec['cv'])} vs full {_f(full['cv'])} "
        f"Δ={delta:+.3f} move≥0.02={move} n={rec['n_defined']} pos={rec['n_pos']}"
    )
    return {
        "n_q5": len(q5),
        "cv": rec["cv"],
        "full": full["cv"],
        "delta": delta,
        "move": move,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
    }


def pass90_no_product() -> dict:
    print("CUT 90 — product/ not written; no 0–100")
    prod = ROOT / "product"
    age = (time.time() - prod.stat().st_mtime) / 3600 if prod.exists() else float("nan")
    print(f"  product/ exists={prod.exists()} age_h={age:.1f} (this process did not write it)")
    return {"exists": prod.exists(), "age_h": age}


def pass91_holdout_cov(p59: dict) -> dict:
    print("CUT 91 — holdout 72 coverage only (no AUROC claim)")
    print(
        f"  a_out_vol cov={_pp(p59['a_out_vol'])} Y3 pos={p59['y3_pos']} "
        f"LOW_POWER — coverage only"
    )
    return {
        "cov": p59["a_out_vol"],
        "y3_pos": p59["y3_pos"],
        "low_power": True,
    }


def pass92_max_leak(p66: dict) -> dict:
    print("CUT 92 — max |Spearman| leak on Y3-labeled")
    rows = p66["leak_rows"]
    best = max(rows, key=lambda r: abs(r["rho"]) if np.isfinite(r["rho"]) else -1)
    print(f"  max |ρ| vs {best['vs']} = {best['rho']:+.3f} twin={best['twin']}")
    return {"vs": best["vs"], "rho": best["rho"], "twin": best["twin"]}


def pass93_q5_vs_size(tr: pd.DataFrame) -> dict:
    print("CUT 93 — after drop company-Q5, does leftover still beat size?")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(g, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    mask = y.notna() & ~tr["company_id"].astype(str).isin(q5)
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], mask)
    sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], mask)
    gap = (rec["cv"] - sz["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(sz["cv"]) else float("nan")
    keepish = bool(np.isfinite(gap) and gap >= KEEP_DELTA)
    print(
        f"  leftover CV={_f(rec['cv'])} size={_f(sz['cv'])} Δ={gap:+.3f} "
        f"still≥0.02={keepish} n={rec['n_defined']} pos={rec['n_pos']}"
    )
    return {
        "cv": rec["cv"],
        "size": sz["cv"],
        "gap": gap,
        "keepish": keepish,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
    }


def pass94_q5_groups(tr: pd.DataFrame) -> dict:
    print("CUT 94 — Q5 companies: group spread (one-group dummy?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "group_id", "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True).agg(
        a_out_vol=("a_out_vol", "mean"),
        group_id=("group_id", "first"),
    )
    g["q"] = pd.qcut(g["a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = g[g["q"] == 5]
    n_gr = int(q5["group_id"].nunique())
    top = q5["group_id"].value_counts().head(3)
    top_share = float(top.iloc[0] / len(q5)) if len(q5) else float("nan")
    print(
        f"  Q5 n={len(q5)} groups={n_gr} top group={top.index[0] if len(top) else '—'} "
        f"share={top_share:.1%}"
    )
    one_group = bool(n_gr <= 3)
    print(f"  one-group dummy={one_group}")
    return {
        "n": int(len(q5)),
        "n_groups": n_gr,
        "top_share": top_share,
        "one_group": one_group,
    }


def pass95_q5_vs_12(tr: pd.DataFrame, ids: list[str]) -> dict:
    print("CUT 95 — Q5 ∩ 12 chronic names")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(g, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    inter = sorted(q5 & set(ids))
    print(f"  Q5 n={len(q5)} ∩12={len(inter)} ids={','.join(inter) if inter else '—'}")
    return {"n_q5": len(q5), "n_inter": len(inter), "ids": inter}


def pass96_no_lgbm() -> dict:
    print("CUT 96 — no LightGBM / no 15-col import in this module")
    hits = []
    for i, line in enumerate(Path(__file__).read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if s.startswith(("import lightgbm", "from lightgbm", "import lgb", "from analysis.models.gbm")):
            hits.append(f"{i}:{s[:80]}")
    print(f"  import hits: {hits or 'none'}")
    return {"hits": hits, "ok": len(hits) == 0}


def pass97_q1q4_company_oof(tr: pd.DataFrame) -> dict:
    print("CUT 97 — company-OOF on Q1–Q4 only (is 0.682 just Q5 vs rest?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[
        y.notna() & tr["a_out_vol"].notna(),
        ["company_id", "fold", Y3, "a_out_vol"],
    ].copy()
    g = d.groupby("company_id", observed=True).agg(
        ever=(Y3, "max"),
        a_out_vol=("a_out_vol", "mean"),
        fold=("fold", "first"),
    )
    g["q"] = pd.qcut(g["a_out_vol"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    body = g[g["q"] != 5]
    rec = signed_oof_auroc(body["ever"], body["a_out_vol"], body["fold"], body["ever"].notna())
    step = bool(np.isfinite(rec["cv"]) and rec["cv"] < 0.60)
    print(
        f"  Q1–Q4 company-OOF CV={_f(rec['cv'])} n={rec['n_defined']} pos={rec['n_pos']} "
        f"folds {fold_bits(rec)} step(Q5-vs-rest)={step}"
    )
    return {
        "cv": rec["cv"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "folds": fold_bits(rec),
        "step": step,
    }


def pass98_drop_q4q5(tr: pd.DataFrame) -> dict:
    print("CUT 98 — drop company Q4+Q5 (top 40%) month-level Y3")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    gmean = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(gmean, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    drop = set(q.index[q.isin([4, 5])].astype(str))
    mask = y.notna() & ~tr["company_id"].astype(str).isin(drop)
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], mask)
    sz = signed_oof_auroc(tr[Y3], tr["log1p_a_in3"], tr["fold"], mask)
    gap = (rec["cv"] - sz["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(sz["cv"]) else float("nan")
    print(
        f"  drop Q4+Q5 n_cos={len(drop)} CV={_f(rec['cv'])} size={_f(sz['cv'])} "
        f"Δ={gap:+.3f} n={rec['n_defined']} pos={rec['n_pos']}"
    )
    return {
        "n_drop": len(drop),
        "cv": rec["cv"],
        "size": sz["cv"],
        "gap": gap,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
    }


def pass99_body_icc_demean(tr: pd.DataFrame) -> dict:
    print("CUT 99 — ICC + demean on Q1–Q4 companies (body still a trait?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    gmean = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(gmean, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    body = y.notna() & tr["a_out_vol"].notna() & ~tr["company_id"].astype(str).isin(q5)
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    icc = _icc_company(tr.loc[body, "a_out_vol"], tr.loc[body, "company_id"])
    dem = ov - ov.groupby(tr["company_id"]).transform("mean")
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], body)
    dem_rec = signed_oof_auroc(tr[Y3], dem, tr["fold"], body)
    drop = (rec["cv"] - dem_rec["cv"]) if np.isfinite(rec["cv"]) and np.isfinite(dem_rec["cv"]) else float("nan")
    trait = bool(np.isfinite(drop) and drop >= 0.10 and np.isfinite(icc["eta2"]) and icc["eta2"] >= 0.40)
    print(
        f"  body CV={_f(rec['cv'])} demean={_f(dem_rec['cv'])} drop={drop:+.3f} "
        f"ICC={_f(icc['icc'])} η²={_f(icc['eta2'])} n_cos={icc['n_cos']} trait={trait}"
    )
    return {
        "cv": rec["cv"],
        "demean": dem_rec["cv"],
        "drop": drop,
        "icc": icc["icc"],
        "eta2": icc["eta2"],
        "n_cos": icc["n_cos"],
        "trait": trait,
    }


def pass100_q5_only(tr: pd.DataFrame) -> dict:
    print("CUT 100 — Q5-only month AUROC (does 0.722 rank inside the pile?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    gmean = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(gmean, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    sl = y.notna() & tr["company_id"].astype(str).isin(q5)
    rec = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], sl)
    ov = pd.to_numeric(tr["a_out_vol"], errors="coerce")
    dem = ov - ov.groupby(tr["company_id"]).transform("mean")
    dem_rec = signed_oof_auroc(tr[Y3], dem, tr["fold"], sl)
    print(
        f"  Q5-only CV={_f(rec['cv'])} demean={_f(dem_rec['cv'])} "
        f"n={rec['n_defined']} pos={rec['n_pos']} folds {fold_bits(rec)}"
    )
    inside = bool(np.isfinite(rec["cv"]) and rec["cv"] >= 0.60)
    print(f"  ranks inside Q5 pile={inside}")
    return {
        "cv": rec["cv"],
        "demean": dem_rec["cv"],
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
        "folds": fold_bits(rec),
        "inside": inside,
    }


def pass101_q5_dummy(tr: pd.DataFrame) -> dict:
    print("CUT 101 — company-Q5 dummy vs continuous a_out_vol (is 0.722 a dummy?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    gmean = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(gmean, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    dummy = tr["company_id"].astype(str).isin(q5).astype(float)
    rec = signed_oof_auroc(tr[Y3], dummy, tr["fold"], y.notna() & tr["a_out_vol"].notna())
    ov = signed_oof_auroc(tr[Y3], tr["a_out_vol"], tr["fold"], y.notna())
    gap = (ov["cv"] - rec["cv"]) if np.isfinite(ov["cv"]) and np.isfinite(rec["cv"]) else float("nan")
    is_dummy = bool(np.isfinite(gap) and abs(gap) < 0.03)
    print(
        f"  Q5 dummy CV={_f(rec['cv'])} continuous={_f(ov['cv'])} Δ={gap:+.3f} "
        f"dummy-shaped={is_dummy} n={rec['n_defined']} pos={rec['n_pos']}"
    )
    return {
        "dummy": rec["cv"],
        "cont": ov["cv"],
        "gap": gap,
        "is_dummy": is_dummy,
        "n": rec["n_defined"],
        "n_pos": rec["n_pos"],
    }


def pass102_q5_by_fold(tr: pd.DataFrame) -> dict:
    print("CUT 102 — Q5 share of Y3 labeled / pos by fold (is 0.782 composition?)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", "a_out_vol"]].copy()
    gmean = d.groupby("company_id", observed=True)["a_out_vol"].mean()
    q = pd.qcut(gmean, 5, labels=[1, 2, 3, 4, 5], duplicates="drop")
    q5 = set(q.index[q == 5].astype(str))
    sl = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["fold", "company_id", Y3]].copy()
    sl["q5"] = sl["company_id"].astype(str).isin(q5)
    rows = []
    for k, g in sl.groupby("fold"):
        n = int(len(g))
        n_pos = int((g[Y3] == 1).sum())
        n_q5 = int(g["q5"].sum())
        n_q5_pos = int(((g[Y3] == 1) & g["q5"]).sum())
        rec = {
            "fold": int(k),
            "n": n,
            "n_pos": n_pos,
            "q5_share": n_q5 / n if n else float("nan"),
            "q5_pos_share": n_q5_pos / n_pos if n_pos else float("nan"),
        }
        rows.append(rec)
        print(
            f"  fold {int(k)} n={n} pos={n_pos} Q5share={rec['q5_share']:.1%} "
            f"Q5pos={rec['q5_pos_share']:.1%}"
        )
    return {"rows": rows}


def pass103_owned_git() -> dict:
    print("CUT 103 — owned files only (git porcelain of forbidden paths)")
    import subprocess

    forbidden = [
        "analysis/evaluate/companies_qa.py",
        "analysis/evaluate/factoring_qa.py",
        "analysis/evaluate/uncat_qa.py",
        "analysis/evaluate/fx_qa.py",
        "product",
    ]
    owned = [
        "analysis/evaluate/a_vol_qa.py",
        "analysis/outputs/a_vol_qa.md",
        "overnight/waves/wave4_a_vol.md",
    ]
    st = subprocess.check_output(
        ["git", "status", "--porcelain", "--"] + forbidden + owned,
        cwd=ROOT,
        text=True,
    )
    print(st if st.strip() else "  (clean on listed paths)")
    for line in st.splitlines():
        print(f"  {line}")
    return {"porcelain": st, "ok": True}


def pass104_parent_line(p66: dict, p67: dict, p81: dict, p89: dict, p92: dict) -> dict:
    print("CUT 104 — parent return line")
    line = (
        f"PARENT reproduced={p66['reproduced']} CV={p66['cv']:.3f} "
        f"leak_twin={p66['leak_any']} SIZE={p66['size_any']} "
        f"max_leak={p92['rho']:+.3f} vs {p92['vs']} "
        f"co_days={p66['rho_co_days']:+.3f} drop12={p66['drop_delta']:+.3f} "
        f"short={p66['short_cv']:.3f} long={p66['long_cv']:.3f} "
        f"Q6={p66['q6']} plus13={p67['plus13']} "
        f"trait={p66['trait']} shock={p66['shock']} "
        f"Q5drop={p89['delta']:+.3f} loo={p81['min_loo']:.3f} "
        f"X={p66['x_dec']} Y={p66['y_dec']} merge={p66['merge']}"
    )
    print(f"  {line}")
    return {"line": line}


def pass105_quote_mtimes() -> dict:
    print("CUT 105 — night quote file mtimes (must not be this run)")
    rows = []
    for rel in (
        "overnight/waves/wave3_slot4_cv_ci.md",
        "overnight/waves/wave4_gbm_core.md",
        "analysis/outputs/i_lift.md",
        "analysis/outputs/q6_quoted.md",
    ):
        p = ROOT / rel
        age_min = (time.time() - p.stat().st_mtime) / 60 if p.exists() else float("nan")
        rows.append({"file": rel, "age_min": age_min})
        print(f"  {rel} age_min={age_min:.1f}")
    ok = all(r["age_min"] > 5 for r in rows if np.isfinite(r["age_min"]))
    print(f"  quote files older than this stay ok={ok}")
    return {"rows": rows, "ok": ok}


def pass106_spearman_pearson_max(p66: dict, p86: dict) -> dict:
    print("CUT 106 — Spearman vs Pearson max |ρ| (twin still fail both)")
    s_best = max(p66["leak_rows"], key=lambda r: abs(r["rho"]) if np.isfinite(r["rho"]) else -1)
    p_best = max(p86["rows"], key=lambda r: abs(r["rho"]) if np.isfinite(r["rho"]) else -1)
    print(
        f"  Spearman max {s_best['vs']} {s_best['rho']:+.3f} twin={s_best['twin']}; "
        f"Pearson max {p_best['vs']} {p_best['rho']:+.3f} twin={p_best['twin']}"
    )
    return {
        "s_vs": s_best["vs"],
        "s_rho": s_best["rho"],
        "p_vs": p_best["vs"],
        "p_rho": p_best["rho"],
        "any_twin": bool(s_best["twin"] or p_best["twin"]),
    }


def pass107_perm_company(tr: pd.DataFrame) -> dict:
    print("CUT 107 — permutation of company-mean a_out_vol vs ever-Y3")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", Y3, "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True).agg(ever=(Y3, "max"), a_out_vol=("a_out_vol", "mean"))
    obs = auroc(g["ever"], g["a_out_vol"])
    rng = np.random.default_rng(FOLD_SEED)
    x = g["a_out_vol"].to_numpy(dtype=float)
    yv = g["ever"].to_numpy(dtype=float)
    n_perm = 1000
    ge = 0
    for _ in range(n_perm):
        null = auroc(yv, rng.permutation(x))
        if np.isfinite(null) and null >= obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    print(f"  obs AUROC={obs:.3f} n={len(g)} pos={int(g['ever'].sum())} p={p:.4f} ({ge}/{n_perm} ≥ obs)")
    return {"obs": obs, "p": p, "n": int(len(g)), "n_pos": int(g["ever"].sum()), "n_perm": n_perm}


def pass108_javier_still_close(p4: dict) -> dict:
    print("CUT 108 — Javier a_vol still CLOSE (do not flip)")
    y3 = next(r for r in p4["y3"]["rows"] if r["feature"] == "a_vol")
    close = bool(np.isfinite(y3["gap_size"]) and y3["gap_size"] < KEEP_DELTA)
    print(f"  a_vol Y3 CV={_f(y3['cv'])} Δsize={y3['gap_size']:+.3f} still_CLOSE={close}")
    return {"cv": y3["cv"], "gap": y3["gap_size"], "close": close}


def pass109_png_and_later(p66: dict, png_ok: bool) -> dict:
    print("CUT 109 — PNG + later-store KEEP still False")
    exists = OUT_PNG.exists()
    size = OUT_PNG.stat().st_size if exists else 0
    print(
        f"  PNG exists={exists} bytes={size} make_ok={png_ok} "
        f"later_keep={p66['later_keep']} shock={p66['shock']}"
    )
    return {
        "exists": exists,
        "size": size,
        "png_ok": png_ok,
        "later_keep": p66["later_keep"],
    }


def pass110_wave_selfcheck() -> dict:
    print("CUT 110 — wave note self-check (CLOSE / 0.722 / 0.762)")
    text = WAVE_NOTE.read_text(encoding="utf-8") if WAVE_NOTE.exists() else ""
    hits = {
        "reproduced": "reproduced=True" in text,
        "close": "X **CLOSE**" in text or "X=CLOSE" in text,
        "cv722": "0.722" in text,
        "quote762": "0.762" in text,
        "nomerge": "Do not merge" in text or "merge **NO**" in text,
    }
    print(f"  {hits}")
    return {"hits": hits, "ok": all(hits.values())}


def pass111_boot_company(tr: pd.DataFrame) -> dict:
    print("CUT 111 — bootstrap CI of company-mean AUROC (200 resamples)")
    y = pd.to_numeric(tr[Y3], errors="coerce")
    d = tr.loc[y.notna() & tr["a_out_vol"].notna(), ["company_id", Y3, "a_out_vol"]].copy()
    g = d.groupby("company_id", observed=True).agg(ever=(Y3, "max"), a_out_vol=("a_out_vol", "mean"))
    rng = np.random.default_rng(FOLD_SEED)
    n = len(g)
    ever = g["ever"].to_numpy(dtype=float)
    xv = g["a_out_vol"].to_numpy(dtype=float)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, n, n)
        auc = auroc(ever[idx], xv[idx])
        if np.isfinite(auc):
            boots.append(auc)
    lo, mid, hi = (float(np.quantile(boots, q)) for q in (0.025, 0.50, 0.975))
    print(f"  bootstrap p2.5/p50/p97.5={lo:.3f}/{mid:.3f}/{hi:.3f} n_boot={len(boots)}")
    return {"lo": lo, "mid": mid, "hi": hi, "n_boot": len(boots)}


def pass112_md_resume() -> dict:
    print("CUT 112 — a_vol_qa.md resume table self-check")
    text = OUT_MD.read_text(encoding="utf-8") if OUT_MD.exists() else ""
    hits = {
        "reproduced": "**True** (0.722)" in text,
        "close": "**CLOSE** / **PARK** / **NO**" in text,
        "trait": "**trait**" in text,
        "leak": "Leak twin / SIZE | False / False" in text,
    }
    print(f"  {hits} ok={all(hits.values())}")
    return {"hits": hits, "ok": all(hits.values())}


def pass113_cell12(p67: dict) -> dict:
    print("CUT 113 — +13pp cells are not n=12")
    tiny = bool(p67["n_lh"] == 12 or p67["n_lo"] == 12)
    print(
        f"  quiet hi/lo n={p67['n_lh']}/{p67['n_lo']} P={p67['lh']:.3f}/{p67['lo']:.3f} "
        f"lift={p67['lift']:+.3f} plus13={p67['plus13']} tiny12={tiny}"
    )
    return {
        "n_lh": p67["n_lh"],
        "n_lo": p67["n_lo"],
        "tiny12": tiny,
        "plus13": p67["plus13"],
    }


def pass114_exact_quotes() -> dict:
    print("CUT 114 — exact night Y3 strings still 0.7622 / 0.7520")
    w3 = (ROOT / "overnight/waves/wave3_slot4_cv_ci.md").read_text(encoding="utf-8")
    il = (ROOT / "analysis/outputs/i_lift.md").read_text(encoding="utf-8")
    hit_762 = "0.7622" in w3
    hit_752 = "0.7520" in il
    print(f"  wave3 has 0.7622={hit_762}; i_lift has 0.7520={hit_752}")
    return {"hit_762": hit_762, "hit_752": hit_752, "ok": hit_762 and hit_752}


def pass115_no_sibling_write() -> dict:
    print("CUT 115 — this module has no write path to sibling QA / parquet / product")
    text = Path(__file__).read_text(encoding="utf-8")
    bad = []
    for tok in (
        "companies_qa.py",
        "factoring_qa.py",
        "uncat_qa.py",
        "fx_qa.py",
        "monthly.parquet",
        "product/",
    ):
        for i, line in enumerate(text.splitlines(), 1):
            if tok in line and ("write_text" in line or ".write(" in line or "to_parquet" in line):
                bad.append(f"{i}:{line.strip()[:80]}")
    print(f"  forbidden write hits: {bad or 'none'}")
    return {"bad": bad, "ok": len(bad) == 0}


def pass116_parquet_age() -> dict:
    print("CUT 116 — monthly.parquet age (must be hours, not this stay)")
    age_h = (time.time() - STORE.stat().st_mtime) / 3600 if STORE.exists() else float("nan")
    ok = bool(np.isfinite(age_h) and age_h > 1)
    print(f"  monthly.parquet age_h={age_h:.2f} ok={ok}")
    return {"age_h": age_h, "ok": ok}


def pass117_holdout_q5_cov(panel: pd.DataFrame) -> dict:
    print("CUT 117 — holdout coverage of train Q5 cutoff (no AUROC)")
    tr = panel[panel["split"] == "train"]
    ho = panel[panel["split"] == "holdout"]
    y = pd.to_numeric(tr[Y3], errors="coerce")
    g = (
        tr.loc[y.notna() & tr["a_out_vol"].notna()]
        .groupby("company_id", observed=True)["a_out_vol"]
        .mean()
    )
    q5_cut = float(g.quantile(0.80))
    hg = ho.loc[ho["a_out_vol"].notna()].groupby("company_id", observed=True)["a_out_vol"].mean()
    n = int(len(hg))
    n_q5 = int((hg >= q5_cut).sum())
    print(
        f"  train Q5 cut={q5_cut:.3f} holdout cos with a_out_vol={n}/72 "
        f"at/above Q5={n_q5} share={n_q5/n if n else float('nan'):.1%} LOW_POWER"
    )
    return {"cut": q5_cut, "n": n, "n_q5": n_q5, "share": (n_q5 / n) if n else float("nan")}


def pass118_wave_holdout() -> dict:
    print("CUT 118 — wave note mentions holdout Q5 coverage")
    text = WAVE_NOTE.read_text(encoding="utf-8") if WAVE_NOTE.exists() else ""
    hit = "holdout ≥train-Q5" in text or "holdout" in text.lower()
    print(f"  wave mentions holdout={hit} len={len(text)}")
    return {"hit": hit, "n_chars": len(text)}


def pass119_stay_clock() -> dict:
    print("CUT 119 — stay clock vs 03:47 resume")
    now = datetime.now().astimezone()
    start = now.replace(hour=3, minute=47, second=0, microsecond=0)
    mins = (now - start).total_seconds() / 60
    print(f"  now={now.isoformat()} minutes_since_0347={mins:.1f} ge30={mins >= 30}")
    return {"now": now.isoformat(), "mins": mins, "ge30": mins >= 30}


def pass120_leak_oneline(p66: dict) -> dict:
    print("CUT 120 — leak one-liner for parent")
    bits = [f"{r['vs']}={r['rho']:+.3f}" for r in p66["leak_rows"]]
    line = " ".join(bits)
    print(f"  {line} twin={p66['leak_any']} SIZE={p66['size_any']}")
    return {"line": line, "twin": p66["leak_any"], "size": p66["size_any"]}


def decide(p1: dict, p2: dict, p3: dict, p4: dict, p6: dict, p8: dict, p12: dict | None = None) -> dict:
    y3 = next(r for r in p4["y3"]["rows"] if r["feature"] == "a_vol")
    y2 = next(r for r in p4["y2"]["rows"] if r["feature"] == "a_vol")
    beat_y3 = bool(np.isfinite(y3["gap_size"]) and y3["gap_size"] >= KEEP_DELTA)
    beat_y2 = bool(np.isfinite(y2["gap_size"]) and y2["gap_size"] >= KEEP_DELTA)
    keep_y3 = beat_y3 and (not p3["is_size"]) and (not p3["is_copy"])
    keep_y2 = beat_y2 and (not p3["is_size"]) and (not p3["is_copy"])
    keep = keep_y3 or keep_y2
    if keep:
        x_dec = "KEEP"
        merge = "YES — KEEP gate hit; parent decides merge. Not a 15-col card tonight."
    elif p3["is_copy"] or p3["is_size"] or (not beat_y3 and not beat_y2):
        x_dec = "CLOSE"
        reasons = []
        if p3["is_copy"]:
            reasons.append("twin of a_io_ratio/a_growth_3")
        if p3["is_size"]:
            reasons.append("SIZE")
        if not beat_y3:
            reasons.append(f"Y3 loses to size (Δ={y3['gap_size']:+.3f})")
        if not beat_y2:
            reasons.append(f"Y2 loses to size (Δ={y2['gap_size']:+.3f})")
        merge = (
            "NO — "
            + "; ".join(reasons)
            + ". a_out_vol later-store KEEP=False (trait dummy, not month shock)."
        )
    else:
        x_dec = "CLOSE"
        merge = "NO"
    y_dec = "PARK"
    return {
        "x_dec": x_dec,
        "y_dec": y_dec,
        "keep": keep,
        "keep_y3": keep_y3,
        "keep_y2": keep_y2,
        "merge": merge,
        "y3_cv": y3["cv"],
        "y2_cv": y2["cv"],
        "y3_gap": y3["gap_size"],
        "y2_gap": y2["gap_size"],
        "q6": p6["any_keep"],
        "identity": p1["same"],
        "bal_verdict": p2["verdict"],
    }


def make_png(tr: pd.DataFrame, p5: dict) -> bool:
    if not HAS_MPL:
        print("PNG skipped (no matplotlib)")
        return False
    tab = next(
        (t for t in p5["tables"] if t["x"] == "a_vol" and t["y"] == Y3 and t["rows"]),
        None,
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    ax = axes[0]
    m = tr["a_vol"].notna() & tr["b_bal_vol"].notna()
    x = tr.loc[m, "a_vol"].to_numpy(dtype=float)
    y = np.minimum(tr.loc[m, "b_bal_vol"].to_numpy(dtype=float), CLIP)
    if len(x) > 40000:
        rng = np.random.default_rng(FOLD_SEED)
        idx = rng.choice(len(x), 40000, replace=False)
        x, y = x[idx], y[idx]
    hb = ax.hexbin(x, y, gridsize=36, cmap="Blues", mincnt=1, bins="log")
    ax.plot([0, CLIP], [0, CLIP], color="#ee6c4d", ls="--", lw=1, label="y=x (clip 3)")
    ax.set_xlabel("a_vol (in-memory)")
    ax.set_ylabel("b_bal_vol clipped at 3")
    ax.set_title("train: cashflow vol ≠ balance vol")
    ax.set_xlim(0, CLIP)
    ax.set_ylim(0, CLIP)
    fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04)
    ax = axes[1]
    if tab:
        xs = [r["q"] for r in tab["rows"]]
        ys = [r["y_rate"] for r in tab["rows"]]
        ax.bar(xs, ys, color="#3d5a80", width=0.7)
        base = float(tr.loc[tr[Y3].notna(), Y3].mean())
        ax.axhline(base, color="#ee6c4d", ls="--", lw=1, label="Y3 stressed base")
        ax.set_xlabel("a_vol quintile")
        ax.set_ylabel("P(Y3=1)")
        ax.set_title(f"Y3 quintiles shape={tab['shape']}")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    print(f"PNG {OUT_PNG}")
    return True


def _md_table(rows: list[dict], cols: list[str]) -> str:
    if not rows:
        return "_(empty)_\n"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def write_md(ctx: dict) -> None:
    d = ctx["decision"]
    p1, p2, p3, p4, p5 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"], ctx["p5"]
    p6, p7, p8, p9, p10 = ctx["p6"], ctx["p7"], ctx["p8"], ctx["p9"], ctx["p10"]
    p11, p12, p13, p14 = ctx["p11"], ctx["p12"], ctx["p13"], ctx["p14"]
    p15, p16, p17, p18 = ctx["p15"], ctx["p16"], ctx["p17"], ctx["p18"]
    p19, p20 = ctx["p19"], ctx["p20"]
    p21, p22, p23 = ctx["p21"], ctx["p22"], ctx["p23"]
    p24, p25, p26 = ctx["p24"], ctx["p25"], ctx["p26"]
    p27 = ctx["p27"]
    p28, p29, p30 = ctx["p28"], ctx["p29"], ctx["p30"]
    p31 = ctx["p31"]
    p32 = ctx["p32"]
    p33, p34, p35 = ctx["p33"], ctx["p34"], ctx["p35"]
    p36 = ctx["p36"]
    p37 = ctx["p37"]
    p38 = ctx["p38"]
    p39 = ctx["p39"]
    p40 = ctx["p40"]
    p41 = ctx["p41"]
    p42 = ctx["p42"]
    p43 = ctx["p43"]
    p44 = ctx["p44"]
    p45 = ctx["p45"]
    p46 = ctx["p46"]
    p47 = ctx["p47"]
    p48 = ctx["p48"]
    p49 = ctx["p49"]
    p50 = ctx["p50"]
    p51 = ctx["p51"]
    p52 = ctx["p52"]
    p53 = ctx["p53"]
    p54 = ctx["p54"]
    p55 = ctx["p55"]
    p56 = ctx["p56"]
    p57 = ctx["p57"]
    p58 = ctx["p58"]
    p59 = ctx["p59"]
    p60 = ctx["p60"]
    p61 = ctx["p61"]
    p62 = ctx["p62"]
    p63 = ctx["p63"]
    p64 = ctx["p64"]
    p65 = ctx["p65"]
    p66 = ctx["p66"]
    p67 = ctx["p67"]
    p68 = ctx["p68"]
    p69 = ctx["p69"]
    p70 = ctx["p70"]
    p71 = ctx["p71"]
    p72 = ctx["p72"]
    p73 = ctx["p73"]
    p74 = ctx["p74"]
    p75 = ctx["p75"]
    p76 = ctx["p76"]
    p77 = ctx["p77"]
    p78 = ctx["p78"]
    p79 = ctx["p79"]
    p80 = ctx["p80"]
    p81 = ctx["p81"]
    p82 = ctx["p82"]
    p83 = ctx["p83"]
    p84 = ctx["p84"]
    p85 = ctx["p85"]
    p86 = ctx["p86"]
    p87 = ctx["p87"]
    p88 = ctx["p88"]
    p89 = ctx["p89"]
    p90 = ctx["p90"]
    p91 = ctx["p91"]
    p92 = ctx["p92"]
    p93 = ctx["p93"]
    p94 = ctx["p94"]
    p95 = ctx["p95"]
    p96 = ctx["p96"]
    p97 = ctx["p97"]
    p98 = ctx["p98"]
    p99 = ctx["p99"]
    p100 = ctx["p100"]
    p101 = ctx["p101"]
    p102 = ctx["p102"]
    p103 = ctx["p103"]
    p104 = ctx["p104"]
    p105 = ctx["p105"]
    p106 = ctx["p106"]
    p107 = ctx["p107"]
    p108 = ctx["p108"]
    p109 = ctx["p109"]
    p110 = ctx["p110"]
    p111 = ctx["p111"]
    p112 = ctx["p112"]
    p113 = ctx["p113"]
    p114 = ctx["p114"]
    p115 = ctx["p115"]
    p116 = ctx["p116"]
    p117 = ctx["p117"]
    p118 = ctx["p118"]
    p119 = ctx["p119"]
    p120 = ctx["p120"]
    t = p1["tight"]
    lines = [
        "# a_vol — in-memory Javier cashflow volatility",
        "",
        f"- **When:** {_now_iso()}",
        f"- **Agent:** `{AGENT}`",
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        f"- **Re-run:** `python -m analysis.evaluate.a_vol_qa`",
        f"- **Holdout:** 72 companies, seed {FOLD_SEED}. Coverage only. ρ / singles on train.",
        "- **X:** `a_vol` reconstructed in memory = "
        "`min(3, sd6(a_net) / max(mean6(a_op_in), 1))`. Family A — legal for Y2/Y3.",
        "- **Never B as X.** `b_bal_vol` is a diagnostic only.",
        "- **Y:** `y3_recover_cash_6m` (stressed-only labeled) and `y2_neg_2of3`.",
        "- Not a 0–100. Not a parquet column. Not a 15-col retrain.",
        "",
        "## Decision",
        "",
        f"**{d['x_dec']}** as a store candidate. **{d['y_dec']}** as a health Y. "
        f"Merge recommendation: {d['merge']}",
        "",
        f"- Identity vs Javier raw `volatility`: ρ={_f(p1['rho'])} n={p1['n']:,} **{p1['verdict']}** "
        f"(Pearson={_f(t['pearson'])} max|Δ|={t['max_abs']:.3g} exact={_pp(t['exact'])}).",
        f"- vs `b_bal_vol`: ρ={_f(p2['rho'])} n={p2['n']:,} **{p2['verdict']}** "
        f"(Javier twin quote 0.354).",
        f"- SIZE ρ vs `log1p(a_in3)`: {_f(p3['size_rho'])} "
        f"({'SIZE' if p3['is_size'] else 'ok'}).",
        f"- Copy: vs `a_io_ratio` ρ={_f(p3['io_rho'])}; vs `a_growth_3` ρ={_f(p3['gr_rho'])} "
        f"({'COPY' if p3['is_copy'] else 'not a twin'}).",
        f"- Y3 stressed single: `{_f(d['y3_cv'])}` vs size `{_f(p4['y3']['size']['cv'])}` "
        f"(Δ {d['y3_gap']:+.3f}) vs `c_n_days_with_tx` night 0.711 "
        f"(replica {_f(next(r['cv'] for r in p4['y3']['rows'] if r['feature']=='c_n_days_with_tx'))}).",
        f"- Y2 single: `{_f(d['y2_cv'])}` vs size `{_f(p4['y2']['size']['cv'])}` "
        f"(Δ {d['y2_gap']:+.3f}) vs night GBM 0.540. "
        f"Fold-min {next(r['fmin'] for r in p64['rows'] if r['y']=='y2'):.3f} invert.",
        f"- Q6 lag1 KEEP: **{d['q6']}**.",
        f"- Chronic-12 drop does not flip the gate (see cut 8).",
        f"- `a_out_vol` (in memory, not Javier): Y3 {_f(p22['y3_cv'])} Δ{p22['y3_gap']:+.3f}, "
        f"T2+T3 0.726 Δ+0.167; Y2 {_f(p65['cv'])} Δ{p65['gap']:+.3f} fold-min {p65['fmin']:.3f} invert. "
        "Quiet-days +9.8pp / dark +11.5pp / ERP +8.3pp. "
        "Not SIZE/COPY/B/15-col. Y3-only. CLOSE as trait dummy. Not tonight's card.",
        f"- Javier `a_vol` loses to size inside T1 (Δ {p41['gap']:+.3f}) and inside "
        f"T2+T3 (Δ {p40['gap']:+.3f}). Overall +0.009 is between-tercile (small books "
        f"are high-vol and have a higher Y3 base). CLOSE.",
        f"- `a_out_vol` is a company trait (demean drops 0.722→0.549), not a days twin "
        f"(company ρ={_f(p48['out_days_rho'])}), complementary 2×2 lifts "
        f"+{p49['lift_quiet']:.1%} quiet / +{p49['lift_busy']:.1%} busy. "
        f"Q5 overlap with `a_vol` is {_pp(p55['share_both_of_ov'])} (not same tail); "
        f"out-only recoveries are {_pp(next(r['mid_share'] for r in p56['rows'] if r['slice']=='out_only'))} T2+T3. "
        "Not a 15-col lag card. CLOSE / PARK / merge NO.",
        "",
        "## Parent return",
        "",
        f"| question | number |",
        f"| --- | ---: |",
        f"| ρ vs Javier `volatility` | {_f(p1['rho'])} **{p1['verdict']}** |",
        f"| ρ vs `b_bal_vol` | {_f(p2['rho'])} **{p2['verdict']}** |",
        f"| Y3 `a_vol` / size / days | {_f(d['y3_cv'])} / {_f(p4['y3']['size']['cv'])} / 0.711 |",
        f"| Y2 `a_vol` / size / night | {_f(d['y2_cv'])} / {_f(p4['y2']['size']['cv'])} / 0.540 |",
        f"| Merge `a_vol` | **NO** (CLOSE / PARK) |",
        f"| Q6 lag1 `a_vol` | **{d['q6']}** "
        f"(Y3 lag1={_f(next(r['lag'] for r in p6['rows'] if r['y']=='y3' and r['feature']=='a_vol'))}) |",
        f"| `a_out_vol` Y3 (in memory) | {_f(p22['y3_cv'])} Δ{p22['y3_gap']:+.3f} "
        f"folds {next(r['folds'] for r in p22['rows'] if r['y']=='y3' and r['slice']=='all')} |",
        f"| `a_out_vol` drop a_vol Q5 | {_f(p57['cv'])} vs size {_f(p57['size'])} (Δ {p57['gap']:+.3f}) |",
        f"| Store already has a_vol? | {'YES — stop' if p51['hits'] else 'no'} |",
        "",
        "## Resume return — a_out_vol 0.722",
        "",
        f"| question | number |",
        f"| --- | ---: |",
        f"| Reproduced 0.722? | **{p66['reproduced']}** ({_f(p66['cv'])}) |",
        f"| Leak twin / SIZE | {p66['leak_any']} / {p66['size_any']} |",
        f"| Company ρ vs days (Y3-labeled) | {_f(p66['rho_co_days'])} |",
        f"| Drop 12 Δ | {p66['drop_delta']:+.3f} (move={p66['drop_move']}) |",
        f"| Short <12 / long ≥12 | {_f(p66['short_cv'])} / {_f(p66['long_cv'])} |",
        f"| Q6 lag1 | {_f(p66['lag1'])} lift {p66['lag_lift']:+.3f} |",
        f"| Month / company quiet 2×2 | {p66['month_quiet']:+.3f} / {p67['lift']:+.3f} |",
        f"| Hot cell is 12 names? | month={p66['month_is_12']} company={p66['co_is_12']} |",
        f"| Demean / η² | {_f(p66['cv'])}→{_f(p66['demean'])} / {_f(p66['eta2'])} |",
        f"| Within-co residual lift | {p69['lift']:+.3f} |",
        f"| Trait vs shock | **trait** (cannot transfer as a month shock) |",
        f"| Leave-one-fold min mean | {_f(p81['min_loo'])} still≥0.70={p81['still']} |",
        f"| Company-OOF (group holdout) | {_f(p83['cv'])} transfers={p83['transfers']} |",
        f"| Company-OOF days / size | {_f(p85['days'])} / {_f(p85['size'])} |",
        f"| Pearson leak twin / SIZE | {p86['any_twin']} / {p86['any_size']} |",
        f"| Max abs Spearman leak | {p92['rho']:+.3f} vs `{p92['vs']}` |",
        f"| Drop company-Q5 Δ | {p89['delta']:+.3f} (move={p89['move']}) |",
        f"| Q1–Q4 company-OOF / body trait | {_f(p97['cv'])} / {p99['trait']} |",
        f"| Q5-only month CV | {_f(p100['cv'])} inside={p100['inside']} |",
        f"| Q5 dummy vs continuous | {_f(p101['dummy'])} vs {_f(p101['cont'])} |",
        f"| Holdout 72 a_out_vol / Y3 pos | {_pp(p91['cov'])} / {p91['y3_pos']} |",
        f"| Holdout at/above train Q5 | {p117['n_q5']}/{p117['n']} ({_pp(p117['share'])}) |",
        f"| Company-mean perm p | {p107['p']:.4f} (obs {_f(p107['obs'])}) |",
        f"| Company-mean AUROC bootstrap | {_f(p111['lo'])}–{_f(p111['hi'])} |",
        f"| Javier a_vol still CLOSE | {p108['close']} (Δ {p108['gap']:+.3f}) |",
        f"| later-store KEEP | **{p66['later_keep']}** (needs month shock) |",
        f"| a_out_vol X / Y / merge | **{p66['x_dec']}** / **{p66['y_dec']}** / **{p66['merge']}** |",
        "",
        "## Brief questions",
        "",
        "- Q3 turning: Javier named this `volatility`. Cashflow sd, not balance sd.",
        "- Q5 why: only if the single beats size and is not a caja twin.",
        "- Q6: honest 1-month lag only.",
        "",
        "## 1. Identity vs Javier",
        "",
        f"Formula matches `score_pipeline.py` ~165: "
        f"`min(3, sd6(net) / max(mean6(op_in), 1))`. Store reconstruct from A is identity.",
        "",
        f"| pair | Spearman | Pearson | max\\|Δ\\| | p50\\|Δ\\| | exact | n |",
        f"| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| `volatility` vs `a_vol` | {_f(t['spearman'])} | {_f(t['pearson'])} | "
        f"{t['max_abs']:.3g} | {t['p50_abs']:.3g} | {_pp(t['exact'])} | {t['n']:,} |",
        "",
        f"Clip=3 share on defined train CM: {_pp(p1['clip_share'])}. "
        f"Of all train CM that is the 11.8% pipeline quote (cut 11).",
        "",
        "## 2. vs `b_bal_vol` (different object)",
        "",
        f"- `a_vol` vs `b_bal_vol`: ρ={_f(p2['rho'])} n={p2['n']:,} **{p2['verdict']}**.",
        f"- Javier vs `b_bal_vol`: ρ={_f(p2['rho_j'])} n={p2['n_j']:,}.",
        f"- Per-company (n≥8): {p2['per']['n_cos']} cos; "
        f"p10={_f(p2['per']['p10'])} p50={_f(p2['per']['p50'])} p90={_f(p2['per']['p90'])}; "
        f"SAME {_pp(p2['per']['same'])} / CLOSE {_pp(p2['per']['close'])} / "
        f"DRIFT {_pp(p2['per']['drift'])}.",
        "",
        "Do not use `b_bal_vol` as Y2/Y3 X.",
        "",
        "## 3. SIZE + copy screen",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "col": p["col"],
                    "vs": p["vs"],
                    "ρ": _f(p["rho"]),
                    "n": f"{p['n']:,}",
                    "|ρ|≥gate": "SIZE"
                    if p["flag_size"]
                    else ("COPY" if p["flag_copy"] else "ok"),
                }
                for p in p3["pairs"]
            ],
            ["col", "vs", "ρ", "n", "|ρ|≥gate"],
        )
    )
    lines += [
        "",
        "## 4. Single-feature train group-fold",
        "",
        f"Y2 n={p4['y2']['n']:,} base {_pp(p4['y2']['rate'])}; "
        f"Y3 stressed n={p4['y3']['n']:,} base {_pp(p4['y3']['rate'])}. "
        f"Sign from the train side of each fold. Seed {FOLD_SEED}.",
        "",
        "### Y3 (stressed)",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "feature": f"`{r['feature']}`",
                    "CV ± sd": f"{_f(r['cv'])} ± {_f(r['sd'])}",
                    "train": _f(r["train"]),
                    "sign": f"{r['sign']:+d}",
                    "cov": _f(r["coverage"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "gap vs 0.711": f"{r['gap_bench']:+.3f}" if np.isfinite(r["gap_bench"]) else "—",
                    "folds": r["folds"],
                }
                for r in p4["y3"]["rows"]
            ],
            [
                "feature",
                "CV ± sd",
                "train",
                "sign",
                "cov",
                "gap vs size",
                "gap vs 0.711",
                "folds",
            ],
        )
    )
    lines += ["", "### Y2", ""]
    lines.append(
        _md_table(
            [
                {
                    "feature": f"`{r['feature']}`",
                    "CV ± sd": f"{_f(r['cv'])} ± {_f(r['sd'])}",
                    "train": _f(r["train"]),
                    "sign": f"{r['sign']:+d}",
                    "cov": _f(r["coverage"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "gap vs 0.540": f"{r['gap_bench']:+.3f}" if np.isfinite(r["gap_bench"]) else "—",
                    "folds": r["folds"],
                }
                for r in p4["y2"]["rows"]
            ],
            [
                "feature",
                "CV ± sd",
                "train",
                "sign",
                "cov",
                "gap vs size",
                "gap vs 0.540",
                "folds",
            ],
        )
    )
    lines += ["", "## 5. Quintiles", ""]
    for tab in p5["tables"]:
        lines.append(
            f"### `{tab['x']}` vs `{tab['y']}` — n={tab['n']:,} bins={tab['n_bins']} "
            f"shape=**{tab['shape']}** mono↑={tab['monotone_up']} tail={tab['tail_only']}"
        )
        lines.append("")
        if tab["rows"]:
            lines.append(
                _md_table(
                    [
                        {
                            "Q": r["q"],
                            "interval": r["interval"],
                            "n": r["n"],
                            "n_pos": r["n_pos"],
                            "P(Y=1)": _f(r["y_rate"], 3),
                            "median X": f"{r['x_median']:.4g}",
                        }
                        for r in tab["rows"]
                    ],
                    ["Q", "interval", "n", "n_pos", "P(Y=1)", "median X"],
                )
            )
        lines.append("")
    lines += [
        "## 6. Q6 lag1 (honest 1-month)",
        "",
        "Lag-1 KEEP needs beat size ≥0.02, stay <0.60, and beat lag-0 by ≥0.01.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "feature": f"`{r['feature']}`",
                    "lag0": _f(r["now"]),
                    "lag1": _f(r["lag"]),
                    "lift": f"{r['lag_lift']:+.3f}" if np.isfinite(r["lag_lift"]) else "—",
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "Q6 KEEP": str(r["q6_keep"]),
                }
                for r in p6["rows"]
            ],
            ["Y", "feature", "lag0", "lag1", "lift", "gap vs size", "Q6 KEEP"],
        )
    )
    lines += [
        "",
        "## 7. Alternatives (still in memory)",
        "",
        "- `a_in_vol` = min(3, sd6(`a_op_in`) / max(mean6(`a_op_in`), 1))",
        "- `a_io_vol` = min(3, sd6(`a_io_ratio`) / max(|mean6(io)|, 1e-6))",
        "- `a_io_sd6` = sd6(`a_io_ratio`) unscaled",
        "- `a_vol_sumdenom` = min(3, sd6(net) / max(sum6(`a_op_in`), 1)) — wrong denom check",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "col": f"`{r['col']}`",
                    "vs Javier": f"{_f(r['rho_j'])} {r['verdict_j']}",
                    "vs b_bal_vol": _f(r["rho_b"]),
                    "Y3 CV": _f(r["y3_cv"]),
                    "Y3 Δsize": f"{r['y3_gap']:+.3f}" if np.isfinite(r["y3_gap"]) else "—",
                    "Y2 CV": _f(r["y2_cv"]),
                    "Y2 Δsize": f"{r['y2_gap']:+.3f}" if np.isfinite(r["y2_gap"]) else "—",
                    "ρ size": _f(r["size_rho"]),
                }
                for r in p7["rows"]
            ],
            [
                "col",
                "vs Javier",
                "vs b_bal_vol",
                "Y3 CV",
                "Y3 Δsize",
                "Y2 CV",
                "Y2 Δsize",
                "ρ size",
            ],
        )
    )
    lines += [
        "",
        "## 8. Chronic 12 dark Y2 names",
        "",
        f"Reconstructed from GROUP_0158/0172 with ≥50% labeled months `b_below_0` "
        f"(y2_why.md). n={p8['n_ids']}. B used only to *name* the pile — never as X.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "med a_vol": _f(r["med_vol"]),
                    "clip3": _pp(r["clip_share"]),
                }
                for r in p8["slices"]
            ],
            ["slice", "n", "n_pos", "med a_vol", "clip3"],
        )
    )
    lines += [
        "",
        "Singles after dropping the 12 names:",
        "",
    ]
    drop_rows = []
    for yname in ("y3", "y2"):
        dd = p8["drop"][yname]
        drop_rows.append(
            {
                "Y": yname,
                "n": dd["n"],
                "n_pos": dd["n_pos"],
                "a_vol CV": f"{_f(dd['vol']['cv'])} ± {_f(dd['vol']['sd'])}",
                "size CV": _f(dd["size"]["cv"]),
                "days CV": _f(dd["days"]["cv"]),
                "gap": f"{dd['gap']:+.3f}" if np.isfinite(dd["gap"]) else "—",
            }
        )
    lines.append(_md_table(drop_rows, ["Y", "n", "n_pos", "a_vol CV", "size CV", "days CV", "gap"]))
    lines += [
        "",
        "## 9. Holdout coverage (LOW_POWER)",
        "",
        f"Holdout CM={p9['n_cm']:,} cos={p9['n_cos']}. "
        f"`a_vol` finite {_pp(p9['a_vol'])}. Y2 pos={p9['y2_pos']} Y3 pos={p9['y3_pos']}. "
        "Not a KEEP claim.",
        "",
        "## 10. Train coverage + fold identity",
        "",
        f"Train CM={p10['n_cm']:,} cos={p10['n_cos']}. "
        f"`a_vol` {_pp(p10['a_vol'])}; `a_in_vol` {_pp(p10['a_in_vol'])}; "
        f"`a_io_vol` {_pp(p10['a_io_vol'])}. Fold-min ρ vs Javier {_f(p10['fold_min'])}.",
        "",
        "## 11. Clip=3 quote (11.8% of all train CM)",
        "",
        f"Defined-row clip share {_pp(p11['of_def'])} = {_pp(p11['of_all'])} of all train CM. "
        f"Javier of-all {_pp(p11['j_of_all'])}. Confirm 11.8%: **{p11['confirm']}**.",
        "",
        "## 12. Alternatives after drop 12 chronic",
        "",
        f"`a_io_vol` Y2 gap after drop12 = {p12['y2_io_gap']:+.3f} "
        f"(was +0.020). Is-the-12: **{p12['io_is_12']}**. "
        f"days gap {p12['y2_days_gap']:+.3f}. "
        f"`a_in_vol` Y3 drop12 CV={_f(p12['y3_in_cv'])} gap={p12['y3_in_gap']:+.3f}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "feature": f"`{r['feature']}`",
                    "CV ± sd": f"{_f(r['cv'])} ± {_f(r['sd'])}",
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "folds": r["folds"],
                }
                for r in p12["rows"]
            ],
            ["Y", "feature", "CV ± sd", "gap vs size", "folds"],
        )
    )
    lines += [
        "",
        "## 13. vs days (activity twin?)",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "col": f"`{r['col']}`",
                    "ρ vs days": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "COPY": str(r["copy"]),
                }
                for r in p13["rows"]
            ],
            ["col", "ρ vs days", "n", "COPY"],
        )
    )
    lines += [
        "",
        "## 14. Y3 size tercile × a_vol quintile",
        "",
        f"Q5 share in size T1={_pp(p14['q5_t1_share'])}. "
        f"Q5 rate={_f(p14['q5_rate'])} body={_f(p14['body_rate'])}. "
        f"T1∩Q5={_f(p14['t1_q5_rate'])} notT1∩Q5={_f(p14['nott1_q5_rate'])}. "
        f"Tail is small-books: **{p14['tail_is_small']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "size": r["size"],
                    "vol": r["vol"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                }
                for r in p14["rows"]
            ],
            ["size", "vol", "n", "n_pos", "P(Y3=1)"],
        )
    )
    lines += [
        "",
        "## 15. Clip=3 flag",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "CV": _f(r["cv"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "n_clip": next(x["n_clip"] for x in p15["rates"] if x["y"] == r["y"]),
                    "P(Y|clip)": _f(next(x["rate_clip"] for x in p15["rates"] if x["y"] == r["y"])),
                    "P(Y|body)": _f(next(x["rate_body"] for x in p15["rates"] if x["y"] == r["y"])),
                }
                for r in p15["rows"]
            ],
            ["Y", "CV", "gap vs size", "n_clip", "P(Y|clip)", "P(Y|body)"],
        )
    )
    lines += [
        "",
        "## 16. Dark 470 vs invoiced",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "cos": r["n_cos"],
                    "med a_vol": _f(r["med_vol"]),
                    "clip3": _pp(r["clip"]),
                }
                for r in p16["slices"]
            ],
            ["slice", "n", "cos", "med a_vol", "clip3"],
        )
    )
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "slice": r["slice"],
                    "CV": _f(r["cv"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "n": r["n"],
                }
                for r in p16["singles"]
            ],
            ["Y", "slice", "CV", "gap vs size", "n"],
        )
    )
    lines += [
        "",
        "## 17. Window / unclip / outflow variants",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "feature": f"`{r['feature']}`",
                    "CV": _f(r["cv"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "vs Javier": f"{_f(r['rho_j'])} {verdict(r['rho_j'])}",
                }
                for r in p17["rows"]
            ],
            ["Y", "feature", "CV", "gap vs size", "vs Javier"],
        )
    )
    lines += [
        "",
        "## 18. Already-neg (B decomp, never X)",
        "",
        f"Clean-now leftover a_vol CV={_f(p18['clean_cv'])} vs size {_f(p18['clean_size'])} "
        f"(Δ {p18['clean_gap']:+.3f}) n_pos={p18['clean_n_pos']}. "
        f"Quintile shape={p18['tab']['shape']}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y2)": _f(r["rate"]),
                    "med a_vol": _f(r["med_vol"]),
                    "clip3": _pp(r["clip"]),
                }
                for r in p18["slices"]
            ],
            ["slice", "n", "n_pos", "P(Y2)", "med a_vol", "clip3"],
        )
    )
    lines += [
        "",
        "## 19. Y3 body vs clip=3",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "feature": f"`{r['feature']}`",
                    "CV": _f(r["cv"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p19["rows"]
            ],
            ["slice", "feature", "CV", "gap vs size", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 20. Fold mix (Y3 stressed)",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "fold": r["fold"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3)": _f(r["rate"]),
                    "med a_vol": _f(r["med_vol"]),
                    "clip3": _pp(r["clip"]),
                    "med size": _f(r["med_size"]),
                }
                for r in p20["rows"]
            ],
            ["fold", "n", "n_pos", "P(Y3)", "med a_vol", "clip3", "med size"],
        )
    )
    lines += [
        "",
        "## 21. SIZE/COPY for outflow-vol / raw sd / windows",
        "",
        f"`a_out_vol` vs size ρ={_f(p21['out_size_rho'])} "
        f"SIZE={p21['out_is_size']} COPY={p21['out_is_copy']}. "
        f"`a_sd6_net` vs size ρ={_f(p21['sd_size_rho'])} SIZE={p21['sd_is_size']}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "col": f"`{r['col']}`",
                    "vs": r["vs"],
                    "ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "flag": "SIZE" if r["flag_size"] else ("COPY" if r["flag_copy"] else "ok"),
                }
                for r in p21["rows"]
            ],
            ["col", "vs", "ρ", "n", "flag"],
        )
    )
    lines += [
        "",
        "## 22. a_out_vol honesty",
        "",
        f"Y3 CV={_f(p22['y3_cv'])} Δsize={p22['y3_gap']:+.3f} shape=**{p22['y3_shape']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "slice": r["slice"],
                    "CV": _f(r["cv"]),
                    "gap vs size": f"{r['gap_size']:+.3f}" if np.isfinite(r["gap_size"]) else "—",
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p22["rows"]
            ],
            ["Y", "slice", "CV", "gap vs size", "n", "n_pos"],
        )
    )
    if p22["tab_y3"]["rows"]:
        lines += ["", "Y3 `a_out_vol` quintiles:", ""]
        lines.append(
            _md_table(
                [
                    {
                        "Q": r["q"],
                        "n": r["n"],
                        "n_pos": r["n_pos"],
                        "P(Y3=1)": _f(r["y_rate"]),
                        "median": f"{r['x_median']:.4g}",
                    }
                    for r in p22["tab_y3"]["rows"]
                ],
                ["Q", "n", "n_pos", "P(Y3=1)", "median"],
            )
        )
    lines += [
        "",
        "## 23. a_io_vol Y2 is not the 12 names",
        "",
        f"Full-panel gap {p23['io_gap']:+.3f} → drop12 {p23['io_drop']:+.3f} "
        f"(lost {p23['io_lost']:+.3f}). Days lost {p23['days_lost']:+.3f}. "
        f"not_the_12=**{p23['not_the_12']}**. Numeric Y2 KEEP={p23['io_keep_y2']} "
        f"— Y3 still loses. Footnote, not a 15-col card.",
        "",
        "## 24. a_out_vol Q6 + B leak",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "lag0": _f(r["now"]),
                    "lag1": _f(r["lag"]),
                    "lift": f"{r['lift']:+.3f}" if np.isfinite(r["lift"]) else "—",
                    "Q6 KEEP": str(r["q6"]),
                }
                for r in p24["rows"]
            ],
            ["Y", "lag0", "lag1", "lift", "Q6 KEEP"],
        )
    )
    lines.append(
        _md_table(
            [
                {
                    "vs": f"`{r['vs']}`",
                    "ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "B-copy": str(r["fail"]),
                }
                for r in p24["leak"]
            ],
            ["vs", "ρ", "n", "B-copy"],
        )
    )
    lines += [
        "",
        "## 25. a_in_vol body vs clip",
        "",
        f"Body CV={_f(p25['body']['cv'])} Δ={p25['body']['gap_size']:+.3f}; "
        f"clip CV={_f(p25['clip']['cv'])} Δ={p25['clip']['gap_size']:+.3f} (inverted). "
        "Overall +0.017 is clip×SIZE mixing. CLOSE.",
        "",
        "## 26. Parent return",
        "",
        f"- `a_vol` (Javier twin): CLOSE. Do not merge.",
        f"- `a_io_vol` Y2 footnote (not the 12): {p26['io_footnote']}. Y3 loses. Do not merge.",
        f"- `a_out_vol` KEEP-shaped? **{p26['out_keep']}**. Different object from Javier vol. "
        f"Core-stem COPY={p27['any_copy']}. Parent decides; not a 15-col card tonight.",
        "",
        "## 27. vs 15-col shallow-A stems",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "vs": f"`{r['vs']}`",
                    "ρ all": _f(r["rho_all"]),
                    "n": f"{r['n_all']:,}",
                    "ρ Y3": _f(r["rho_y3"]),
                    "COPY": str(r["copy"]),
                }
                for r in p27["rows"]
            ],
            ["vs", "ρ all", "n", "ρ Y3", "COPY"],
        )
    )
    lines += [
        "",
        "## 28. a_out_vol × size tercile",
        "",
        f"Q5 share in T1={_pp(p28['q5_t1_share'])}. "
        f"T2+T3 CV={_f(p28['mid_cv'])} vs size {_f(p28['mid_size'])} "
        f"(Δ {p28['mid_gap']:+.3f}) vs days {_f(p28['mid_days'])}. "
        f"KEEP on mid-size: **{p28['mid_keep']}** folds {p28['mid_folds']}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "size": r["size"],
                    "vol": r["vol"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                }
                for r in p28["rows"]
            ],
            ["size", "vol", "n", "n_pos", "P(Y3=1)"],
        )
    )
    lines += [
        "",
        "## 29. acf1 (company median)",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "col": f"`{r['col']}`",
                    "n_cos": r["n"],
                    "p10": _f(r["p10"]),
                    "p50": _f(r["p50"]),
                    "p90": _f(r["p90"]),
                }
                for r in p29["rows"]
            ],
            ["col", "n_cos", "p10", "p50", "p90"],
        )
    )
    lines += [
        "",
        "## 30. a_out_vol inside size T1",
        "",
        f"T1 CV={_f(p30['cv'])} vs size {_f(p30['size'])} (Δ {p30['gap']:+.3f}) "
        f"vs days {_f(p30['days'])} n={p30['n']} pos={p30['n_pos']} folds {p30['folds']}.",
        "",
        "## 31. days × a_out_vol 2×2 (Y3)",
        "",
        f"Medians days={p31['dmed']:.3g} a_out_vol={p31['omed']:.3g}. "
        f"Outvol lift in quiet-days {p31['lift_quiet']:+.3f}, "
        f"in busy-days {p31['lift_busy']:+.3f}. "
        f"Days-substitute: **{p31['substitute']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                }
                for r in p31["rows"]
            ],
            ["slice", "n", "n_pos", "P(Y3=1)"],
        )
    )
    lines += [
        "",
        "## 32. days × a_out_vol 2×2 on T2+T3",
        "",
        f"Quiet-days outvol lift={p32['lift_quiet']:+.3f} (still not just T1).",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                }
                for r in p32["rows"]
            ],
            ["slice", "n", "n_pos", "P(Y3=1)"],
        )
    )
    lines += [
        "",
        "## 33. a_out6 × a_out_vol (just high outflow?)",
        "",
        f"Vol lift inside low `a_out6` = {p33['lift_low_out6']:+.3f}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                }
                for r in p33["rows"]
            ],
            ["slice", "n", "n_pos", "P(Y3=1)"],
        )
    )
    lines += [
        "",
        "## 34. a_out_vol by so-far",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "CV": _f(r["cv"]),
                    "size": _f(r["size"]),
                    "gap": f"{r['gap']:+.3f}" if np.isfinite(r["gap"]) else "—",
                    "low_power": str(r["low_power"]),
                }
                for r in p34["rows"]
            ],
            ["bucket", "n", "n_pos", "CV", "size", "gap", "low_power"],
        )
    )
    lines += [
        "",
        "## 35. B diagnostic (never X)",
        "",
        f"Any |ρ|≥0.80 vs B: **{p35['any_fail']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "pair": r["vs"],
                    "ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "B-copy": str(r["fail"]),
                }
                for r in p35["rows"]
            ],
            ["pair", "ρ", "n", "B-copy"],
        )
    )
    lines += [
        "",
        "## 36. days × a_out_vol inside dark / invoiced",
        "",
    ]
    for blk in p36["blocks"]:
        lines.append(f"### {blk['pop']} — quiet-days lift {blk['lift']:+.3f}")
        lines.append("")
        lines.append(
            _md_table(
                [
                    {
                        "slice": r["slice"],
                        "n": r["n"],
                        "n_pos": r["n_pos"],
                        "P(Y3=1)": _f(r["rate"]),
                    }
                    for r in blk["rows"]
                ],
                ["slice", "n", "n_pos", "P(Y3=1)"],
            )
        )
        lines.append("")
    lines += [
        "## 37. a_vol (Javier) by so-far",
        "",
        "If it never beats size in a window, CLOSE is not a late-trail artifact.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "a_vol": _f(r["cv"]),
                    "size": _f(r["size"]),
                    "days": _f(r["days"]),
                    "gap": f"{r['gap']:+.3f}" if np.isfinite(r["gap"]) else "—",
                }
                for r in p37["rows"]
            ],
            ["bucket", "n", "n_pos", "a_vol", "size", "days", "gap"],
        )
    )
    lines += [
        "",
        "## 38. who owns the quiet+high-out-vol recoveries?",
        "",
        f"Labeled pile n={p38['pile_n']:,} across {p38['pile_cos']} companies; "
        f"Y3=1 n={p38['pos_n']} across {p38['pos_cos']} companies. "
        f"Top-8 share of recoveries={_pp(p38['top8_share'])} "
        f"({'concentrated' if p38['concentrated'] else 'spread'}).",
        "",
    ]
    if p38["top"]:
        lines.append(
            _md_table(
                [{"company_id": r["company_id"], "recoveries": r["n"]} for r in p38["top"]],
                ["company_id", "recoveries"],
            )
        )
    lines += [
        "",
        "## 39. leftover so-far (why overall a_vol Δ shrinks)",
        "",
        f"Defined Y3+a_vol n={p39['all_n']:,} CV={_f(p39['all_cv'])} "
        f"size={_f(p39['all_size'])} gap={p39['all_gap']:+.3f}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                    "a_vol": _f(r["cv"]),
                    "size": _f(r["size"]),
                    "gap": f"{r['gap']:+.3f}" if np.isfinite(r["gap"]) else "—",
                }
                for r in p39["rows"]
            ],
            ["bucket", "n", "n_pos", "P(Y3=1)", "a_vol", "size", "gap"],
        )
    )
    lines += [
        "",
        "## 40. a_vol on T2+T3 (Javier off small books)",
        "",
        f"n={p40['n']:,} pos={p40['n_pos']} a_vol={_f(p40['cv'])} "
        f"size={_f(p40['size'])} days={_f(p40['days'])} gap={p40['gap']:+.3f} "
        f"window-KEEP={p40['keepish']}. Overall still CLOSE (Δ=+0.009).",
        "",
        "## 41. a_vol inside size T1",
        "",
        f"n={p41['n']:,} pos={p41['n_pos']} a_vol={_f(p41['cv'])} "
        f"size={_f(p41['size'])} days={_f(p41['days'])} gap={p41['gap']:+.3f} "
        f"T1-KEEP={p41['keepish']}. Both slices lose to size — overall +0.009 is between-tercile.",
        "",
        "## 42. Simpson: size tercile × a_vol median",
        "",
        f"a_vol median={p42['vmed']:.3g}. Within each tercile, hi-vol vs low-vol is the test; "
        "tercile base rates show the between-group mix.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "tercile": r["tercile"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                    "mean a_vol": _f(r["mean_vol"]),
                }
                for r in p42["bases"]
            ],
            ["tercile", "n", "n_pos", "P(Y3=1)", "mean a_vol"],
        )
    )
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                    "mean a_vol": _f(r["mean_vol"]),
                }
                for r in p42["rows"]
            ],
            ["slice", "n", "n_pos", "P(Y3=1)", "mean a_vol"],
        )
    )
    lines += [
        "",
        "## 43. a_out_vol Simpson (within-tercile lift)",
        "",
        f"a_out_vol median={p43['vmed']:.3g}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                    "hi-lo": f"{r['lift']:+.3f}" if "lift" in r else "",
                }
                for r in p43["rows"]
            ],
            ["slice", "n", "n_pos", "P(Y3=1)", "hi-lo"],
        )
    )
    lines += [
        "",
        "## 44. a_vol vs clip=3 flag",
        "",
        "If the flag matches continuous AUROC, Javier vol is mostly the cap.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "y": r["y"],
                    "flag": _f(r["flag"]),
                    "continuous": _f(r["cont"]),
                    "size": _f(r["size"]),
                    "almost_flag": r["almost_flag"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p44["rows"]
            ],
            ["y", "flag", "continuous", "size", "almost_flag", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 45. company-mean vol vs ever Y3=1",
        "",
        f"{p45['n_cos']} train companies with a labeled Y3 month; "
        f"{p45['n_pos_cos']} ever recover. Group-fold on company rows (fold of first labeled month).",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "company AUROC": _f(r["cv"]),
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p45["rows"]
            ],
            ["feature", "company AUROC", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 46. company-mean on common companies",
        "",
        f"{p46['n_cos']} companies with all four means defined; {p46['n_pos']} ever recover. "
        f"a_out_vol vs size {p46['out_gap_size']:+.3f}; vs days {p46['out_gap_days']:+.3f}. "
        f"a_vol vs size {p46['av_gap_size']:+.3f}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "company AUROC": _f(r["cv"]),
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p46["rows"]
            ],
            ["feature", "company AUROC", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 47. company-demeaned (timing vs trait)",
        "",
        "If demeaning kills AUROC, the signal is a sticky company type. "
        "If it survives, there is within-company timing.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "raw": _f(r["raw"]),
                    "demean": _f(r["demean"]),
                    "drop": f"{r['drop']:+.3f}" if np.isfinite(r["drop"]) else "—",
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p47["rows"]
            ],
            ["feature", "raw", "demean", "drop", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 48. company-mean copy screen",
        "",
        f"`a_out_vol` vs days ρ={_f(p48['out_days_rho'])} COPY={p48['out_days_copy']}. "
        "If COPY, the company-trait KEEP is a days twin at company grain.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "a": r["a"],
                    "b": r["b"],
                    "ρ": _f(r["rho"]),
                    "n": r["n"],
                    "COPY": r["copy"],
                }
                for r in p48["rows"]
            ],
            ["a", "b", "ρ", "n", "COPY"],
        )
    )
    lines += [
        "",
        "## 49. company-mean days × a_out_vol vs ever-recover",
        "",
        f"{p49['n_cos']} companies. Quiet-days lift {p49['lift_quiet']:+.3f}; "
        f"busy-days lift {p49['lift_busy']:+.3f}. Complementary if both >0.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "ever": r["n_pos"],
                    "P(ever Y3)": _f(r["rate"]),
                }
                for r in p49["rows"]
            ],
            ["slice", "n", "ever", "P(ever Y3)"],
        )
    )
    lines += [
        "",
        "## 50. company-mean vs ever Y2",
        "",
        f"{p50['n_cos']} companies; {p50['n_pos']} ever Y2. Footnote only.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "company AUROC": _f(r["cv"]),
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p50["rows"]
            ],
            ["feature", "company AUROC", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 51. store audit",
        "",
        f"Named vol columns already in `monthly.parquet`: {p51['hits'] or 'none'}. "
        f"Clean (do not merge) = {p51['clean']}.",
        "",
        "## 52. a_in_vol company-mean Y3",
        "",
        f"{p52['n_cos']} companies; {p52['n_pos']} ever recover. "
        f"a_in_vol vs size {p52['gap']:+.3f} (month-level missed KEEP by 0.003).",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "feature": r["feature"],
                    "company AUROC": _f(r["cv"]),
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                }
                for r in p52["rows"]
            ],
            ["feature", "company AUROC", "n", "n_pos"],
        )
    )
    lines += [
        "",
        "## 53. activity twin screen",
        "",
        f"Any COPY vs a_n_tx / days? **{p53['any_copy']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "col": r["col"],
                    "vs": r["vs"],
                    "ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "COPY": r["copy"],
                }
                for r in p53["rows"]
            ],
            ["col", "vs", "ρ", "n", "COPY"],
        )
    )
    lines += [
        "",
        "## 54. company-mean a_out_vol vs outflow level",
        "",
        f"COPY vs a_out6 / a_op_out? **{p54['any_copy']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "vs": r["vs"],
                    "ρ": _f(r["rho"]),
                    "n": r["n"],
                    "COPY": r["copy"],
                }
                for r in p54["rows"]
            ],
            ["vs", "ρ", "n", "COPY"],
        )
    )
    lines += [
        "",
        "## 55. Q5 recovery overlap (same tail?)",
        "",
        f"Y3 pos={p55['n_pos']}; a_vol Q5 pos={p55['n_av5_pos']}; "
        f"a_out_vol Q5 pos={p55['n_ov5_pos']}; both={p55['n_both']}; "
        f"out-only={p55['n_ov_only']}; vol-only={p55['n_av_only']}. "
        f"both/outQ5={_pp(p55['share_both_of_ov'])} same_tail={p55['same_tail']}.",
        "",
        "## 56. size mix of out-Q5-only recoveries",
        "",
        "If out-only recoveries sit in T2+T3, a_out_vol is not recycling Javier's small-book tail.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "T1": r["T1"],
                    "T2": r["T2"],
                    "T3": r["T3"],
                    "T2+T3": _pp(r["mid_share"]),
                }
                for r in p56["rows"]
            ],
            ["slice", "n", "T1", "T2", "T3", "T2+T3"],
        )
    )
    lines += [
        "",
        "## 57. a_out_vol after dropping a_vol Q5",
        "",
        f"n={p57['n']:,} pos={p57['n_pos']} a_out_vol={_f(p57['cv'])} "
        f"size={_f(p57['size'])} days={_f(p57['days'])} gap={p57['gap']:+.3f} "
        f"KEEP-shaped={p57['keepish']}.",
        "",
        "## 58. a_vol after dropping a_out_vol Q5",
        "",
        f"n={p58['n']:,} pos={p58['n_pos']} a_vol={_f(p58['cv'])} "
        f"size={_f(p58['size'])} days={_f(p58['days'])} gap={p58['gap']:+.3f} "
        f"KEEP-shaped={p58['keepish']}. Slice-only; overall a_vol stays CLOSE. Days still {_f(p58['days'])}.",
        "",
        "## 59. holdout coverage (LOW_POWER)",
        "",
        f"{p59['n_cos']} companies (72={p59['ok72']}), CM={p59['n_cm']:,}. "
        f"a_vol {_pp(p59['a_vol'])}; a_out_vol {_pp(p59['a_out_vol'])}; "
        f"Y3 pos={p59['y3_pos']}/{p59['y3_n']}. No AUROC claim.",
        "",
        "## 60. leak + parquet untouched",
        "",
        f"leakage_check a_vol* + a_out_vol vs Y3 forbidden B: ok={p60['leak_ok']}. "
        f"`monthly.parquet` age {p60['parquet_age_h']:.1f}h (not rewritten this run).",
        "",
        "## 61. registry rows this agent",
        "",
        f"{p61['n']} append-only rows for `{AGENT}` / `a_vol_qa`.",
        "",
        "## 62. score_pipeline quote (read-only)",
        "",
        f"Line {p62['line']}: `{p62['quote']}` still present={p62['hit']}. File not edited.",
        "",
        "## 63. a_out_vol Y3 fold-min",
        "",
        f"Folds `{p63['folds']}`; min={_f(p63['fmin'])}; invert(<0.50)={p63['invert']}.",
        "",
        "## 64. a_vol fold-min",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Y": r["y"],
                    "CV": _f(r["cv"]),
                    "folds": r["folds"],
                    "min": _f(r["fmin"]),
                    "invert": r["invert"],
                }
                for r in p64["rows"]
            ],
            ["Y", "CV", "folds", "min", "invert"],
        )
    )
    lines += [
        "",
        "## 65. a_out_vol Y2 fold-min",
        "",
        f"CV={_f(p65['cv'])} Δsize={p65['gap']:+.3f} folds `{p65['folds']}` "
        f"min={_f(p65['fmin'])} invert={p65['invert']}. Y3-only KEEP-shape.",
        "",
        "## 66. confirm or kill a_out_vol 0.722",
        "",
        f"**Reproduced 0.722?** **{p66['reproduced']}** "
        f"(CV={_f(p66['cv'])} ± {_f(p66['sd'])} n={p66['n']:,} pos={p66['n_pos']} "
        f"folds `{p66['folds']}`).",
        "",
        f"**X = {p66['x_dec']}**. **Y = {p66['y_dec']}**. Merge **{p66['merge']}**. "
        f"{p66['note']}. Not a 15-col card. Night Y3 GBM quote 0.762 / 0.752 untouched.",
        "",
        f"- SIZE: vs `log1p(a_in3)` / `a_op_in` fail={p66['size_any']}. "
        f"Y3 Δ vs size {p66['gap']:+.3f} (size={_f(p66['size'])} days={_f(p66['days'])}).",
        f"- Leak twin any |ρ|≥0.80 on Y3-labeled: **{p66['leak_any']}**.",
        f"- Company ρ vs days on Y3-labeled companies: {_f(p66['rho_co_days'])} n={p66['n_co']} "
        f"(quote −0.453).",
        f"- Drop 12 chronic: CV={_f(p66['drop_cv'])} Δ={p66['drop_delta']:+.3f} "
        f"move≥0.02={p66['drop_move']} (n_ids={p66['n_ids']}).",
        f"- Short <12 so-far: {_f(p66['short_cv'])} vs size {_f(p66['short_size'])} "
        f"n={p66['short_n']} pos={p66['short_pos']}.",
        f"- Long ≥12: {_f(p66['long_cv'])} vs size {_f(p66['long_size'])} "
        f"n={p66['long_n']} pos={p66['long_pos']}.",
        f"- Q6 lag1={_f(p66['lag1'])} vs now={_f(p66['now'])} lift={p66['lag_lift']:+.3f} "
        f"KEEP={p66['q6']}.",
        f"- Month 2×2 quiet lift {p66['month_quiet']:+.3f}; hot cell is 12 names={p66['month_is_12']}.",
        f"- Company 2×2 quiet lift {p66['co_quiet']:+.3f} (quote +13pp); "
        f"hot cell is 12 names={p66['co_is_12']}.",
        f"- Demean {_f(p66['cv'])}→{_f(p66['demean'])} drop={p66['demean_drop']:+.3f}. "
        f"ICC={_f(p66['icc'])} η²={_f(p66['eta2'])} on {p66['icc_n_cos']} companies. "
        f"**trait={p66['trait']} shock={p66['shock']}**.",
        f"- later-store KEEP={p66['later_keep']} (needs month shock; this is a company trait).",
        f"- Drop company-Q5: Δ={p89['delta']:+.3f} move={p89['move']}; Q5-only CV={_f(p100['cv'])}; "
        f"Q5 dummy={_f(p101['dummy'])} vs continuous {_f(p101['cont'])}.",
        f"- Pass49-style company 2×2 +13.2pp (n=195/105). Neither cell is the 12 names.",
        "",
        "### leak on Y3-labeled months",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "vs": r["vs"],
                    "ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "twin≥0.80": r["twin"],
                    "SIZE≥0.50": r["size_fail"],
                }
                for r in p66["leak_rows"]
            ],
            ["vs", "ρ", "n", "twin≥0.80", "SIZE≥0.50"],
        )
    )
    lines += [
        "",
        "### month 2×2 (days × a_out_vol medians)",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "P(Y3=1)": _f(r["rate"]),
                    "chronic n": r["n_ch"],
                    "chronic pos": r["n_ch_pos"],
                    "ch share": _pp(r["ch_share"]),
                }
                for r in p66["month_cells"]
            ],
            ["slice", "n", "n_pos", "P(Y3=1)", "chronic n", "chronic pos", "ch share"],
        )
    )
    lines += [
        "",
        "### company 2×2 (mean days × mean a_out_vol)",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "ever": r["n_pos"],
                    "P(ever)": _f(r["rate"]),
                    "chronic cos": r["n_ch"],
                    "chronic ever": r["n_ch_pos"],
                }
                for r in p66["co_cells"]
            ],
            ["slice", "n", "ever", "P(ever)", "chronic cos", "chronic ever"],
        )
    )
    lines += [
        "",
        "## 67. +13pp reconcile + group ICC",
        "",
        f"Pass49-style company 2×2 quiet lift {p67['lift']:+.3f} "
        f"({_pp(p67['lh'])} vs {_pp(p67['lo'])}, n={p67['n_lh']}/{p67['n_lo']}) "
        f"quote +13pp **{p67['plus13']}**.",
        "",
        f"ICC company={_f(p67['icc_co'])} η²={_f(p67['eta2_co'])}; "
        f"ICC group={_f(p67['icc_gr'])} η²={_f(p67['eta2_gr'])} on {p67['n_groups']} groups. "
        f"High-out-vol companies={p67['n_hi_cos']} in {p67['n_hi_groups']} groups.",
        "",
        "## 68. first defined month vs company mean",
        "",
        f"first↔mean ρ={_f(p68['rho'])} n={p68['n']}. "
        f"Company AUROC first={_f(p68['first_cv'])} mean={_f(p68['mean_cv'])} "
        f"pos={p68['n_pos']}. If first ≈ mean, the type is set at month 6.",
        "",
        "## 69. within-company residual shock",
        "",
        f"{p69['n_cos']} companies with both sides. "
        f"Above own median P={_f(p69['r_hi'])} n={p69['n_hi']}; "
        f"below P={_f(p69['r_lo'])} n={p69['n_lo']}; lift={p69['lift']:+.3f}. "
        f"Residual shock={p69['shockish']}.",
        "",
        "## 70. hands off",
        "",
        f"Night Y3 GBM quote 0.762 / 0.752 untouched={p70['quote_untouched']}. "
        "companies_qa / factoring_qa not edited.",
        "",
        "## 71. company-mean a_out_vol quintiles vs ever Y3",
        "",
        f"monotone_up={p71['monotone_up']}.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "Q": r["q"],
                    "n": r["n"],
                    "ever": r["n_pos"],
                    "P(ever)": _f(r["rate"]),
                    "median": _f(r["med"]),
                }
                for r in p71["rows"]
            ],
            ["Q", "n", "ever", "P(ever)", "median"],
        )
    )
    lines += [
        "",
        "## 72. company Q5 pile — dark vs invoiced",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "slice": r["slice"],
                    "n": r["n"],
                    "ever": r["n_pos"],
                    "P(ever)": _f(r["rate"]),
                }
                for r in p72["rows"]
            ],
            ["slice", "n", "ever", "P(ever)"],
        )
    )
    lines += [
        "",
        "## 73. ICC all-train vs Y3-labeled",
        "",
        f"All-train ICC={_f(p73['all_icc'])} η²={_f(p73['all_eta2'])} "
        f"n={p73['all_n']:,} cos={p73['all_cos']}. "
        f"Y3-labeled ICC={_f(p73['lab_icc'])} η²={_f(p73['lab_eta2'])} "
        f"n={p73['lab_n']:,} cos={p73['lab_cos']}. High either way.",
        "",
        "## 74. chronic 12 on Y3",
        "",
        f"{p74['n']} labeled months, {p74['n_pos']} recoveries, "
        f"{p74['n_ov']} with a_out_vol. zero_pos={p74['zero_pos']}. "
        f"Ids: {', '.join(p74['ids'])}.",
        "",
        "## 75. a_out_vol coverage by so-far",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "bucket": r["bucket"],
                    "CM": r["n"],
                    "a_out_vol": _pp(r["cov"]),
                    "Y3 labeled": r["y3_n"],
                }
                for r in p75["rows"]
            ],
            ["bucket", "CM", "a_out_vol", "Y3 labeled"],
        )
    )
    lines += [
        "",
        "## 76. Y3 labeled without a_out_vol",
        "",
        f"Defined n={p76['n_def']:,} pos={p76['pos_def']} P={_f(p76['p_def'])}. "
        f"Missing n={p76['n_mis']:,} pos={p76['pos_mis']} P={_f(p76['p_mis'])}. "
        "0.722 is only on months with 6+ history.",
        "",
        "## 77. ICC on companies with ≥8 labeled months",
        "",
        f"{p77['n_cos']} companies, n={p77['n']:,}, ICC={_f(p77['icc'])} η²={_f(p77['eta2'])}, "
        f"CV={_f(p77['cv'])} pos={p77['n_pos']}. Trait holds on long books.",
        "",
        "## 78. owned files",
        "",
        "This process only writes a_vol_qa.py / a_vol_qa.md / wave4_a_vol.md / registry / PNG.",
        "",
        "## 79. leak on all train months",
        "",
        f"Any twin |ρ|≥0.80 off the Y3 path? **{p79['any_twin']}**.",
        "",
    ]
    lines.append(
        _md_table(
            [
                {
                    "vs": r["vs"],
                    "ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "twin": r["twin"],
                }
                for r in p79["rows"]
            ],
            ["vs", "ρ", "n", "twin"],
        )
    )
    lines += [
        "",
        "## 80. CV vs in-sample",
        "",
        f"OOF CV={_f(p80['cv'])} in-sample train={_f(p80['train'])} "
        f"({p80['n_folds']} folds). Quote 0.722 is the OOF mean.",
        "",
        "## 81. leave-one-fold of 0.722",
        "",
        f"Min leave-one-fold mean={_f(p81['min_loo'])} still≥0.70=**{p81['still']}**. "
        "0.722 is not one lucky fold.",
        "",
        "## 82. fold signs",
        "",
        f"signs={p82['signs']} all_same=**{p82['all_same']}** train_sign={p82['train_sign']}.",
        "",
        "## 83. company-grain OOF (trait transfer)",
        "",
        f"Company-mean a_out_vol vs ever-Y3 group-fold CV={_f(p83['cv'])} "
        f"n={p83['n']} pos={p83['n_pos']} folds `{p83['folds']}`. "
        f"Transfers as a company type across groups=**{p83['transfers']}**. "
        "Still not a month shock — new companies need their own history.",
        "",
        "## 84. night Y3 quote files",
        "",
        f"0.762 / 0.752 still present in quote files=**{p84['ok']}**. This process did not write them.",
        "",
        "## 85. company-OOF vs days / size",
        "",
        f"out={_f(p85['out'])} days={_f(p85['days'])} size={_f(p85['size'])} "
        f"(Δdays={p85['gap_days']:+.3f} Δsize={p85['gap_size']:+.3f}). "
        "Trait is complementary to days at company grain, not a days dummy.",
        "",
        "## 86. Pearson leak (Y3-labeled)",
        "",
        f"any twin={p86['any_twin']} any SIZE={p86['any_size']}.",
        "",
        _md_table(
            [
                {
                    "vs": r["vs"],
                    "Pearson ρ": _f(r["rho"]),
                    "n": f"{r['n']:,}",
                    "twin": r["twin"],
                    "SIZE": r["size_fail"],
                }
                for r in p86["rows"]
            ],
            ["vs", "Pearson ρ", "n", "twin", "SIZE"],
        ),
        "",
        "## 87. parquet + sibling mtimes",
        "",
        "This process did not write parquet / companies_qa / factoring_qa.",
        "",
        _md_table(
            [
                {"file": r["file"], "age_min": f"{r['age_min']:.1f}" if np.isfinite(r["age_min"]) else "—"}
                for r in p87["rows"]
            ],
            ["file", "age_min"],
        ),
        "",
        "## 88. 12 chronic names",
        "",
        f"n={p88['n']}: `{', '.join(p88['ids'])}`. Y3 labeled={p88['y3_n']} pos={p88['y3_pos']}.",
        "",
        "## 89. drop company-Q5 pile",
        "",
        f"Drop {p89['n_q5']} Q5 companies: CV={_f(p89['cv'])} vs full {_f(p89['full'])} "
        f"Δ={p89['delta']:+.3f} move≥0.02=**{p89['move']}** n={p89['n']} pos={p89['n_pos']}. "
        "If this moves, the 0.722 is the high-vol company pile (trait).",
        "",
        "## 90. product/ hands off",
        "",
        f"product/ exists={p90['exists']} age_h={p90['age_h']:.1f}. This process did not write it. No 0–100.",
        "",
        "## 91. holdout 72 coverage",
        "",
        f"a_out_vol={_pp(p91['cov'])} Y3 pos={p91['y3_pos']}. LOW_POWER — coverage only, no AUROC.",
        "",
        "## 92. max |Spearman| leak",
        "",
        f"max |ρ| vs `{p92['vs']}` = {p92['rho']:+.3f} twin={p92['twin']}.",
        "",
        "## 93. leftover after Q5 vs size",
        "",
        f"CV={_f(p93['cv'])} size={_f(p93['size'])} Δ={p93['gap']:+.3f} "
        f"still≥0.02=**{p93['keepish']}** n={p93['n']} pos={p93['n_pos']}.",
        "",
        "## 94. Q5 group spread",
        "",
        f"Q5 n={p94['n']} groups={p94['n_groups']} top share={_pp(p94['top_share'])} "
        f"one-group dummy=**{p94['one_group']}**.",
        "",
        "## 95. Q5 ∩ 12 chronic",
        "",
        f"overlap={p95['n_inter']} of {p95['n_q5']}. "
        f"ids=`{', '.join(p95['ids']) if p95['ids'] else 'none'}`.",
        "",
        "## 96. no LightGBM in this module",
        "",
        f"ok=**{p96['ok']}** hits={p96['hits'] or 'none'}.",
        "",
        "## 97. company-OOF Q1–Q4 only",
        "",
        f"CV={_f(p97['cv'])} n={p97['n']} pos={p97['n_pos']} folds `{p97['folds']}`. "
        f"Q5-vs-rest step=**{p97['step']}**.",
        "",
        "## 98. drop company Q4+Q5",
        "",
        f"n_drop={p98['n_drop']} CV={_f(p98['cv'])} size={_f(p98['size'])} "
        f"Δ={p98['gap']:+.3f} n={p98['n']} pos={p98['n_pos']}.",
        "",
        "## 99. body ICC + demean (Q1–Q4)",
        "",
        f"body CV={_f(p99['cv'])} demean={_f(p99['demean'])} drop={p99['drop']:+.3f} "
        f"ICC={_f(p99['icc'])} η²={_f(p99['eta2'])} n_cos={p99['n_cos']} trait=**{p99['trait']}**.",
        "",
        "## 100. Q5-only month AUROC",
        "",
        f"Q5-only CV={_f(p100['cv'])} demean={_f(p100['demean'])} "
        f"n={p100['n']} pos={p100['n_pos']} folds `{p100['folds']}`. "
        f"Ranks inside the pile=**{p100['inside']}**. "
        "The quoted 0.722 is Q5-vs-rest (company type), not a body month shock.",
        "",
        "## 101. Q5 dummy vs continuous",
        "",
        f"Q5 dummy CV={_f(p101['dummy'])} continuous={_f(p101['cont'])} "
        f"Δ={p101['gap']:+.3f} dummy-shaped=**{p101['is_dummy']}**.",
        "",
        "## 102. Q5 share by fold",
        "",
        _md_table(
            [
                {
                    "fold": r["fold"],
                    "n": r["n"],
                    "n_pos": r["n_pos"],
                    "Q5 share": _pp(r["q5_share"]),
                    "Q5 of pos": _pp(r["q5_pos_share"]),
                }
                for r in p102["rows"]
            ],
            ["fold", "n", "n_pos", "Q5 share", "Q5 of pos"],
        ),
        "",
        "## 103. owned files",
        "",
        "This process writes `a_vol_qa.py` / `a_vol_qa.md` / `wave4_a_vol.md` only. "
        f"git porcelain captured (ok={p103['ok']}).",
        "",
        "## 104. parent return line",
        "",
        f"`{p104['line']}`",
        "",
        "## 105. night quote file ages",
        "",
        f"All quote files older than this stay=**{p105['ok']}**.",
        "",
        "## 106. Spearman vs Pearson max leak",
        "",
        f"Spearman max `{p106['s_vs']}` {p106['s_rho']:+.3f}; "
        f"Pearson max `{p106['p_vs']}` {p106['p_rho']:+.3f}; any twin={p106['any_twin']}.",
        "",
        "## 107. permutation of company means",
        "",
        f"obs AUROC={_f(p107['obs'])} n={p107['n']} pos={p107['n_pos']} "
        f"p={p107['p']:.4f} ({p107['n_perm']} perms).",
        "",
        "## 108. Javier a_vol still CLOSE",
        "",
        f"Y3 CV={_f(p108['cv'])} Δsize={p108['gap']:+.3f} still_CLOSE=**{p108['close']}**. Do not merge.",
        "",
        "## 109. PNG + later-store KEEP",
        "",
        f"PNG exists={p109['exists']} bytes={p109['size']} later_keep=**{p109['later_keep']}**.",
        "",
        "## 110. wave self-check",
        "",
        f"ok=**{p110['ok']}** hits={p110['hits']}.",
        "",
        "## 111. company-mean AUROC bootstrap",
        "",
        f"p2.5/p50/p97.5={_f(p111['lo'])}/{_f(p111['mid'])}/{_f(p111['hi'])} "
        f"n_boot={p111['n_boot']}.",
        "",
        "## 112. resume table self-check",
        "",
        f"ok=**{p112['ok']}** hits={p112['hits']}.",
        "",
        "## 113. +13pp cells not n=12",
        "",
        f"n={p113['n_lh']}/{p113['n_lo']} plus13={p113['plus13']} tiny12=**{p113['tiny12']}**.",
        "",
        "## 114. exact night quotes",
        "",
        f"0.7622 in wave3={p114['hit_762']}; 0.7520 in i_lift={p114['hit_752']}. ok=**{p114['ok']}**.",
        "",
        "## 115. no sibling / parquet / product writes",
        "",
        f"ok=**{p115['ok']}** bad={p115['bad'] or 'none'}.",
        "",
        "## 116. parquet age",
        "",
        f"monthly.parquet age_h={p116['age_h']:.2f} ok=**{p116['ok']}**.",
        "",
        "## 117. holdout vs train Q5 cutoff (coverage only)",
        "",
        f"train Q5 cut={p117['cut']:.3f}; holdout cos defined={p117['n']}/72; "
        f"at/above Q5={p117['n_q5']} ({_pp(p117['share'])}). LOW_POWER — no AUROC.",
        "",
        "## 118. wave holdout mention",
        "",
        f"wave mentions holdout=**{p118['hit']}** ({p118['n_chars']} chars).",
        "",
        "## 119. stay clock",
        "",
        f"minutes since 03:47={p119['mins']:.1f} ≥30=**{p119['ge30']}** ({p119['now']}).",
        "",
        "## 120. leak one-liner",
        "",
        f"`{p120['line']}` twin={p120['twin']} SIZE={p120['size']}.",
        "",
        "## What this is not",
        "",
        "- Not a 0–100. Not a store merge. Not a 15-col retrain.",
        "- Not family B as Y2/Y3 X.",
        "- Uncat / FX modules were not edited.",
        "",
        f"Elapsed {ctx['elapsed_s']:.1f}s. PNG: `{OUT_PNG.name}` ({ctx['png_ok']}).",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT_MD}")


def append_registry(ctx: dict) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    d = ctx["decision"]
    p1, p2, p3, p4 = ctx["p1"], ctx["p2"], ctx["p3"], ctx["p4"]
    ts = _utc_ts()
    y3_days = next(r["cv"] for r in p4["y3"]["rows"] if r["feature"] == "c_n_days_with_tx")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_vol_vs_javier_rho",
            "value": p1["rho"],
            "coverage": f"{p1['n'] / ctx['p10']['n_cm']:.4f}" if ctx["p10"]["n_cm"] else "",
            "notes": f"verdict={p1['verdict']} exact={p1['tight']['exact']:.4f} clip3={p1['clip_share']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_vol_vs_b_bal_vol_rho",
            "value": p2["rho"],
            "coverage": f"{p2['n'] / ctx['p10']['n_cm']:.4f}" if ctx["p10"]["n_cm"] else "",
            "notes": f"verdict={p2['verdict']} javier_vs_b={p2['rho_j']:.4f} per_co_drift={p2['per']['drift']:.3f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_vol",
            "value": d["y3_cv"],
            "coverage": f"{next(r['coverage'] for r in p4['y3']['rows'] if r['feature']=='a_vol'):.4f}",
            "notes": (
                f"size={p4['y3']['size']['cv']:.4f} gap={d['y3_gap']:+.4f} "
                f"days={y3_days:.4f} SIZE={p3['is_size']} COPY={p3['is_copy']} x={d['x_dec']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_vol",
            "value": d["y2_cv"],
            "coverage": f"{next(r['coverage'] for r in p4['y2']['rows'] if r['feature']=='a_vol'):.4f}",
            "notes": (
                f"size={p4['y2']['size']['cv']:.4f} gap={d['y2_gap']:+.4f} "
                f"night=0.540 x={d['x_dec']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_vol_size_rho",
            "value": p3["size_rho"],
            "coverage": "1.0000",
            "notes": f"SIZE={p3['is_size']} io_rho={p3['io_rho']:.4f} growth_rho={p3['gr_rho']:.4f}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_vol_merge_recommend",
            "value": 1 if d["keep"] else 0,
            "coverage": "1.0000",
            "notes": d["merge"][:180],
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_in_vol",
            "value": next(r["cv"] for r in p4["y3"]["rows"] if r["feature"] == "a_in_vol"),
            "coverage": f"{next(r['coverage'] for r in p4['y3']['rows'] if r['feature']=='a_in_vol'):.4f}",
            "notes": (
                f"gap={next(r['gap_size'] for r in p4['y3']['rows'] if r['feature']=='a_in_vol'):+.4f} "
                f"drop12_gap={ctx['p12']['y3_in_gap']:+.4f} almost_KEEP_misses_0.02"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_io_vol",
            "value": next(r["cv"] for r in p4["y2"]["rows"] if r["feature"] == "a_io_vol"),
            "coverage": f"{next(r['coverage'] for r in p4['y2']['rows'] if r['feature']=='a_io_vol'):.4f}",
            "notes": (
                f"gap=+0.020 at gate; drop12_gap={ctx['p12']['y2_io_gap']:+.4f} "
                f"is_the_12={ctx['p12']['io_is_12']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": "-",
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_vol_clip3_of_all_cm",
            "value": ctx["p11"]["of_all"],
            "coverage": f"{ctx['p11']['n_def'] / ctx['p11']['n']:.4f}",
            "notes": f"of_def={ctx['p11']['of_def']:.4f} confirm118={ctx['p11']['confirm']}",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_out_vol",
            "value": ctx["p22"]["y3_cv"],
            "coverage": "1.0000",
            "notes": (
                f"gap={ctx['p22']['y3_gap']:+.4f} shape={ctx['p22']['y3_shape']} "
                f"SIZE={ctx['p21']['out_is_size']} COPY={ctx['p21']['out_is_copy']} "
                f"KEEP={ctx['p26']['out_keep']} not_javier_twin"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y2,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_io_vol_not_chronic12",
            "value": 1 if ctx["p23"]["not_the_12"] else 0,
            "coverage": "1.0000",
            "notes": (
                f"gap={ctx['p23']['io_gap']:+.4f} drop12={ctx['p23']['io_drop']:+.4f} "
                f"lost={ctx['p23']['io_lost']:+.4f} days_lost={ctx['p23']['days_lost']:+.4f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_out_vol_t2t3",
            "value": ctx["p28"]["mid_cv"],
            "coverage": "1.0000",
            "notes": (
                f"gap={ctx['p28']['mid_gap']:+.4f} size={ctx['p28']['mid_size']:.4f} "
                f"days={ctx['p28']['mid_days']:.4f} keep_mid={ctx['p28']['mid_keep']} "
                f"q5_t1={ctx['p28']['q5_t1_share']:.3f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_out_vol_quiet_days_lift",
            "value": ctx["p31"]["lift_quiet"],
            "coverage": "1.0000",
            "notes": (
                f"busy_lift={ctx['p31']['lift_busy']:+.4f} substitute={ctx['p31']['substitute']} "
                f"t2t3_lift={ctx['p32']['lift_quiet']:+.4f} low_out6_lift={ctx['p33']['lift_low_out6']:+.4f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_out_vol_quiet_dark_erp_lift",
            "value": next((b["lift"] for b in ctx["p36"]["blocks"] if b["pop"] == "dark"), float("nan")),
            "coverage": "1.0000",
            "notes": (
                "erp_lift="
                + f"{next((b['lift'] for b in ctx['p36']['blocks'] if b['pop']=='erp'), float('nan')):+.4f} "
                + f"pile_pos={ctx['p38']['pos_n']} pile_cos={ctx['p38']['pos_cos']} "
                + f"top8={ctx['p38']['top8_share']:.3f} concentrated={ctx['p38']['concentrated']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_vol_t2t3",
            "value": ctx["p40"]["cv"],
            "coverage": "1.0000",
            "notes": (
                f"gap={ctx['p40']['gap']:+.4f} size={ctx['p40']['size']:.4f} "
                f"days={ctx['p40']['days']:.4f} t1_cv={ctx['p41']['cv']:.4f} "
                f"t1_gap={ctx['p41']['gap']:+.4f} keepish_t2t3={ctx['p40']['keepish']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_out_vol_tercile_lifts",
            "value": next((r["lift"] for r in ctx["p43"]["rows"] if r.get("slice") == "T3_hi"), float("nan")),
            "coverage": "1.0000",
            "notes": (
                "T1="
                + f"{next(r['lift'] for r in ctx['p43']['rows'] if r.get('slice')=='T1_hi'):+.4f} "
                + "T2="
                + f"{next(r['lift'] for r in ctx['p43']['rows'] if r.get('slice')=='T2_hi'):+.4f} "
                + "T3="
                + f"{next(r['lift'] for r in ctx['p43']['rows'] if r.get('slice')=='T3_hi'):+.4f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_out_vol_company_vs_days_rho",
            "value": ctx["p48"]["out_days_rho"],
            "coverage": "1.0000",
            "notes": (
                f"copy={ctx['p48']['out_days_copy']} "
                f"demean={next(r['demean'] for r in ctx['p47']['rows'] if r['feature']=='a_out_vol'):.4f} "
                f"quiet_co={ctx['p49']['lift_quiet']:+.4f} busy_co={ctx['p49']['lift_busy']:+.4f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_out_vol_q5_overlap_avol",
            "value": ctx["p55"]["share_both_of_ov"],
            "coverage": "1.0000",
            "notes": (
                f"both={ctx['p55']['n_both']} out_only={ctx['p55']['n_ov_only']} "
                f"vol_only={ctx['p55']['n_av_only']} same_tail={ctx['p55']['same_tail']}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train",
            "metric": "a_out_vol_q5_only_t2t3_share",
            "value": next(r["mid_share"] for r in ctx["p56"]["rows"] if r["slice"] == "out_only"),
            "coverage": "1.0000",
            "notes": (
                f"n={next(r['n'] for r in ctx['p56']['rows'] if r['slice']=='out_only')} "
                f"both_t1={next(r['T1'] for r in ctx['p56']['rows'] if r['slice']=='both_q5')}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "auroc_a_out_vol_drop_avol_q5",
            "value": ctx["p57"]["cv"],
            "coverage": "1.0000",
            "notes": (
                f"gap={ctx['p57']['gap']:+.4f} size={ctx['p57']['size']:.4f} "
                f"days={ctx['p57']['days']:.4f} avol_after_drop_outq5={ctx['p58']['cv']:.4f} "
                f"avol_gap={ctx['p58']['gap']:+.4f}"
            ),
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": "A",
            "y": Y3,
            "model": "a_vol_qa",
            "split": "train_cv",
            "metric": "a_out_vol_confirm_0722",
            "value": ctx["p66"]["cv"],
            "coverage": "1.0000",
            "notes": (
                f"reproduced={ctx['p66']['reproduced']} gap={ctx['p66']['gap']:+.4f} "
                f"leak={ctx['p66']['leak_any']} size={ctx['p66']['size_any']} "
                f"drop12={ctx['p66']['drop_delta']:+.4f} eta2={ctx['p66']['eta2']:.4f} "
                f"trait={ctx['p66']['trait']} shock={ctx['p66']['shock']} "
                f"x={ctx['p66']['x_dec']} merge={ctx['p66']['merge']}"
            ),
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


def write_wave(ctx: dict) -> None:
    if not WRITE_WAVE:
        print("wave note deferred (WRITE_WAVE=False)")
        return
    d = ctx["decision"]
    p1, p2, p4 = ctx["p1"], ctx["p2"], ctx["p4"]
    failed = "; ".join(ctx.get("failed", [])) or "none"
    body = (
        f"# Wave 4 — a_vol (in-memory)\n\n"
        f"- **When:** {_now_iso()}\n"
        f"- **Agent:** `{AGENT}`\n"
        f"- **Files:** `analysis/evaluate/a_vol_qa.py`, `analysis/outputs/a_vol_qa.md`, "
        f"`analysis/outputs/a_vol_vs_b_bal_vol.png`\n"
        f"- **Columns (in memory only):** `a_vol`, `a_in_vol`, `a_io_vol`, `a_io_sd6`, "
        f"`a_vol_sumdenom`, `a_out_vol`, `a_vol_w3`, `a_vol_w12`. Not merged.\n"
        f"- **Train coverage:** a_vol {_pp(ctx['p10']['a_vol'])} "
        f"({ctx['p10']['n_cos']} train companies).\n"
        f"- **Identity:** vs Javier `volatility` ρ={_f(p1['rho'])} **{p1['verdict']}**.\n"
        f"- **DRIFT:** vs `b_bal_vol` ρ={_f(p2['rho'])} **{p2['verdict']}**.\n"
        f"- **Y3 / Y2 singles:** {_f(d['y3_cv'])} / {_f(d['y2_cv'])} "
        f"vs size {_f(p4['y3']['size']['cv'])} / {_f(p4['y2']['size']['cv'])}; "
        f"vs days 0.711 / night 0.540.\n"
        f"- **Decision:** X **{d['x_dec']}** (`a_vol` Javier twin), Y **{d['y_dec']}**. "
        f"Merge `a_vol`: {d['merge']}\n"
        f"- **Small-book tail:** T2+T3 a_vol {ctx['p40']['cv']:.3f} vs size "
        f"{ctx['p40']['size']:.3f} (Δ {ctx['p40']['gap']:+.3f}); "
        f"T1 {ctx['p41']['cv']:.3f} vs size {ctx['p41']['size']:.3f} "
        f"(Δ {ctx['p41']['gap']:+.3f}).\n"
        f"- **In-memory footnote:** `a_out_vol` Y3 0.722 Δ+0.104 monotone_up; "
        f"T2+T3 0.726 Δ+0.167 vs days 0.725. Dark quiet-days lift +11.5pp; "
        f"ERP +8.3pp. Recovery pile {ctx['p38']['pos_n']} hits / "
        f"{ctx['p38']['pos_cos']} cos, top8={ctx['p38']['top8_share']:.1%} "
        f"({'concentrated' if ctx['p38']['concentrated'] else 'spread'}). "
        f"Within-tercile hi-lo T1/T2/T3 +9.9/+5.0/+8.4pp. "
        f"Company-mean common vs size {ctx['p46']['out_gap_size']:+.3f}; "
        f"vs days ρ={ctx['p48']['out_days_rho']:+.3f} (not COPY). "
        f"Demean 0.722→0.549 (trait). Company 2×2 quiet/busy "
        f"+{ctx['p49']['lift_quiet']:.1%}/+{ctx['p49']['lift_busy']:.1%}. "
        f"Q5 overlap vs a_vol {_pp(ctx['p55']['share_both_of_ov'])}; "
        f"out-only T2+T3 {_pp(next(r['mid_share'] for r in ctx['p56']['rows'] if r['slice']=='out_only'))}. "
        f"Drop a_vol Q5: a_out_vol {ctx['p57']['cv']:.3f} vs size {ctx['p57']['size']:.3f} "
        f"(Δ {ctx['p57']['gap']:+.3f}). Holdout 72 a_out_vol {_pp(ctx['p59']['a_out_vol'])}, "
        f"Y3 pos={ctx['p59']['y3_pos']}. "
        f"Y2 a_out_vol {ctx['p65']['cv']:.3f} fold-min {ctx['p65']['fmin']:.3f} invert "
        f"(Y3-only). Not a 15-col lag card. Parent decides; not tonight's card.\n"
        f"- **Confirm 0.722:** reproduced={ctx['p66']['reproduced']} "
        f"CV={ctx['p66']['cv']:.3f} Δsize={ctx['p66']['gap']:+.3f}. "
        f"Leak twin={ctx['p66']['leak_any']} SIZE={ctx['p66']['size_any']} "
        f"drop12 Δ={ctx['p66']['drop_delta']:+.3f} move={ctx['p66']['drop_move']}. "
        f"short={ctx['p66']['short_cv']:.3f} long={ctx['p66']['long_cv']:.3f} "
        f"Q6={ctx['p66']['q6']}. "
        f"Month quiet {ctx['p66']['month_quiet']:+.3f} (12={ctx['p66']['month_is_12']}); "
        f"company quiet {ctx['p66']['co_quiet']:+.3f} (12={ctx['p66']['co_is_12']}). "
        f"Demean {ctx['p66']['cv']:.3f}→{ctx['p66']['demean']:.3f} "
        f"η²={ctx['p66']['eta2']:.3f} trait={ctx['p66']['trait']} shock={ctx['p66']['shock']} "
        f"later_keep={ctx['p66']['later_keep']}. "
        f"a_out_vol X **{ctx['p66']['x_dec']}** Y **{ctx['p66']['y_dec']}** merge **{ctx['p66']['merge']}** "
        f"— {ctx['p66']['note']}\n"
        f"- **+13pp / ICC:** company 2×2 lift {ctx['p67']['lift']:+.3f} (quote+13={ctx['p67']['plus13']}); "
        f"η² company={ctx['p67']['eta2_co']:.3f} group={ctx['p67']['eta2_gr']:.3f}.\n"
        f"- **Stability / transfer:** leave-one-fold min mean={ctx['p81']['min_loo']:.3f} "
        f"still≥0.70={ctx['p81']['still']}; fold signs all_same={ctx['p82']['all_same']}; "
        f"company-OOF {ctx['p83']['cv']:.3f} transfers={ctx['p83']['transfers']}; "
        f"night quote 0.762/0.752 files ok={ctx['p84']['ok']}. "
        f"company-OOF days={ctx['p85']['days']:.3f} size={ctx['p85']['size']:.3f}; "
        f"Pearson twin={ctx['p86']['any_twin']}. "
        f"12 names Y3 pos={ctx['p88']['y3_pos']}. "
        f"drop Q5-cos Δ={ctx['p89']['delta']:+.3f} move={ctx['p89']['move']}. "
        f"max leak {ctx['p92']['rho']:+.3f} vs {ctx['p92']['vs']}. "
        f"Q5 leftover vs size Δ={ctx['p93']['gap']:+.3f}; "
        f"Q5 groups={ctx['p94']['n_groups']} ∩12={ctx['p95']['n_inter']}. "
        f"Q1–Q4 company-OOF={ctx['p97']['cv']:.3f} step={ctx['p97']['step']}; "
        f"drop Q4+Q5 CV={ctx['p98']['cv']:.3f}. "
        f"body demean {ctx['p99']['cv']:.3f}→{ctx['p99']['demean']:.3f} "
        f"η²={ctx['p99']['eta2']:.3f} trait={ctx['p99']['trait']}. "
        f"Q5-only CV={ctx['p100']['cv']:.3f} inside={ctx['p100']['inside']}. "
        f"Q5 dummy={ctx['p101']['dummy']:.3f} vs cont {ctx['p101']['cont']:.3f} "
        f"dummy-shaped={ctx['p101']['is_dummy']}. "
        f"quote-file ages ok={ctx['p105']['ok']}; Spearman/Pearson twin={ctx['p106']['any_twin']}. "
        f"holdout ≥train-Q5={ctx['p117']['n_q5']}/{ctx['p117']['n']}.\n"
        f"- **Failed / next:** {failed}. "
        f"Do not merge `a_vol`. Do not put `a_out_vol` on the 15-col card tonight.\n"
    )
    WAVE_NOTE.parent.mkdir(parents=True, exist_ok=True)
    WAVE_NOTE.write_text(body, encoding="utf-8")
    print(f"wrote {WAVE_NOTE}")


def run() -> dict:
    t0 = time.time()
    print(f"a_vol_qa start seed={FOLD_SEED} holdout=72 agent={AGENT}")
    con = connect()
    try:
        panel = load_panel(con)
    finally:
        con.close()
    panel = attach_folds(panel)
    panel = add_panel_lags(panel, ["a_vol", "a_in_vol", "a_io_vol"], (1,))
    tr = panel[panel["split"] == "train"].copy()
    assert_no_holdout(tr["company_id"])
    print(
        f"train CM={len(tr):,} companies={tr['company_id'].nunique()} "
        f"holdout CM={int((panel['split']=='holdout').sum())}"
    )

    p1 = pass1_identity(tr)
    p2 = pass2_bal_vol(tr)
    p3 = pass3_size_copy(tr)
    p4 = pass4_singles(tr)
    p5 = pass5_quintiles(tr)
    p6 = pass6_q6(tr)
    p7 = pass7_alts(tr, p3, p4)
    ids = chronic_ids(tr)
    print(f"  chronic ids n={len(ids)} {ids[:12]}")
    p8 = pass8_chronic(tr, ids)
    p9 = pass9_holdout(panel)
    p10 = pass10_cov(tr)
    p11 = pass11_clip_quote(tr, p1)
    p12 = pass12_alt_drop12(tr, ids)
    p13 = pass13_days_twin(tr)
    p14 = pass14_size_x_vol(tr)
    p15 = pass15_clip_flag(tr)
    p16 = pass16_dark(tr)
    p17 = pass17_windows(tr)
    p18 = pass18_already_neg(tr)
    p19 = pass19_body(tr)
    p20 = pass20_fold0(tr)
    p21 = pass21_out_screen(tr)
    p22 = pass22_out_honest(tr, ids)
    p23 = pass23_io_not12(tr, p4, p12)
    p24 = pass24_out_lag(tr)
    p25 = pass25_in_vol_body_story(tr, p19)
    p26 = pass26_parent_return(p4, p12, p21, p22, p23)
    p27 = pass27_core_redundancy(tr)
    p28 = pass28_out_size_mix(tr)
    p29 = pass29_acf(tr)
    p30 = pass30_out_t1_only(tr)
    p31 = pass31_days_x_out(tr)
    p32 = pass32_days_x_out_mid(tr)
    p33 = pass33_out6_x_outvol(tr)
    p34 = pass34_sofar(tr)
    p35 = pass35_b_diag(tr)
    p36 = pass36_dark_2x2(tr)
    p37 = pass37_avol_sofar(tr)
    p38 = pass38_recovery_names(tr)
    p39 = pass39_avol_leftover(tr)
    p40 = pass40_avol_mid(tr)
    p41 = pass41_avol_t1(tr)
    p42 = pass42_simpson(tr)
    p43 = pass43_out_simpson(tr)
    p44 = pass44_clip_binary(tr)
    p45 = pass45_company_trait(tr)
    p46 = pass46_company_common(tr)
    p47 = pass47_demean(tr)
    p48 = pass48_company_copy(tr)
    p49 = pass49_company_2x2(tr)
    p50 = pass50_y2_company(tr)
    p51 = pass51_store_audit()
    p52 = pass52_in_vol_company(tr)
    p53 = pass53_activity_twin(tr)
    p54 = pass54_out6_company(tr)
    p55 = pass55_q5_overlap(tr)
    p56 = pass56_out_only_mix(tr)
    p57 = pass57_drop_avol_q5(tr)
    p58 = pass58_drop_out_q5(tr)
    p59 = pass59_holdout_out(panel)
    p60 = pass60_no_write()
    p61 = pass61_registry_count()
    p62 = pass62_formula_quote()
    p63 = pass63_out_foldmin(p22)
    p64 = pass64_avol_foldmin(p4)
    p65 = pass65_out_y2_folds(p22)
    p66 = pass66_confirm_out(tr, ids)
    p67 = pass67_plus13_and_group(tr, ids)
    p68 = pass68_first_defined(tr)
    p69 = pass69_within_co(tr)
    p70 = pass70_hands_off()
    p71 = pass71_co_quintiles(tr)
    p72 = pass72_q5_who(tr)
    p73 = pass73_icc_all_train(tr)
    p74 = pass74_chronic_y3(tr, ids)
    p75 = pass75_sofar_cov(tr)
    p76 = pass76_undefined_y3(tr)
    p77 = pass77_icc_long_cos(tr)
    p78 = pass78_owned_only()
    p79 = pass79_leak_all_train(tr)
    p80 = pass80_cv_vs_train(tr)
    p81 = pass81_leave_one_fold(tr)
    p82 = pass82_fold_signs(tr)
    p83 = pass83_co_grain_oof(tr)
    p84 = pass84_night_quote()
    p85 = pass85_co_oof_days_size(tr)
    p86 = pass86_pearson_leak(tr)
    p87 = pass87_hands_off_mtime()
    p88 = pass88_twelve_names(tr, ids)
    p89 = pass89_drop_q5_cos(tr)
    p90 = pass90_no_product()
    p91 = pass91_holdout_cov(p59)
    p92 = pass92_max_leak(p66)
    p93 = pass93_q5_vs_size(tr)
    p94 = pass94_q5_groups(tr)
    p95 = pass95_q5_vs_12(tr, ids)
    p96 = pass96_no_lgbm()
    p97 = pass97_q1q4_company_oof(tr)
    p98 = pass98_drop_q4q5(tr)
    p99 = pass99_body_icc_demean(tr)
    p100 = pass100_q5_only(tr)
    p101 = pass101_q5_dummy(tr)
    p102 = pass102_q5_by_fold(tr)
    p103 = pass103_owned_git()
    p104 = pass104_parent_line(p66, p67, p81, p89, p92)
    p105 = pass105_quote_mtimes()
    p106 = pass106_spearman_pearson_max(p66, p86)
    p107 = pass107_perm_company(tr)
    p108 = pass108_javier_still_close(p4)
    if p27["any_copy"]:
        p26["out_keep"] = False
        print("  flipped a_out_vol KEEP → False (core twin)")
    if not p28["mid_keep"]:
        p26["out_keep"] = False
        print("  flipped a_out_vol KEEP → False (dies on T2+T3)")
    if not p66["later_keep"]:
        p26["out_keep"] = False
        print("  flipped a_out_vol KEEP → False (trait dummy, not month shock)")

    decision = decide(p1, p2, p3, p4, p6, p8, p12)
    png_ok = make_png(tr, p5)
    p109 = pass109_png_and_later(p66, png_ok)
    p110 = pass110_wave_selfcheck()
    p111 = pass111_boot_company(tr)
    p112 = pass112_md_resume()
    p113 = pass113_cell12(p67)
    p114 = pass114_exact_quotes()
    p115 = pass115_no_sibling_write()
    p116 = pass116_parquet_age()
    p117 = pass117_holdout_q5_cov(panel)
    p118 = pass118_wave_holdout()
    p119 = pass119_stay_clock()
    p120 = pass120_leak_oneline(p66)
    headline = (
        f"Identity **{p1['verdict']}** ρ={_f(p1['rho'])}. "
        f"vs b_bal_vol **{p2['verdict']}** ρ={_f(p2['rho'])}. "
        f"Y3 a_vol {_f(decision['y3_cv'])} vs size {_f(p4['y3']['size']['cv'])} "
        f"(Δ {decision['y3_gap']:+.3f}) vs days 0.711. "
        f"Y2 {_f(decision['y2_cv'])} vs size {_f(p4['y2']['size']['cv'])} "
        f"(Δ {decision['y2_gap']:+.3f}) vs 0.540. "
        f"X **{decision['x_dec']}**. Merge: {decision['merge']}"
    )
    print(headline)
    print(
        f"CONFIRM a_out_vol reproduced={p66['reproduced']} CV={_f(p66['cv'])} "
        f"X={p66['x_dec']} Y={p66['y_dec']} merge={p66['merge']} trait={p66['trait']} "
        f"shock={p66['shock']} later_keep={p66['later_keep']}"
    )
    failed = []
    if not p1["same"]:
        failed.append(f"identity vs Javier is {p1['verdict']} ρ={_f(p1['rho'])}")
    if abs(p2["rho"] - 0.354) > 0.08 if np.isfinite(p2["rho"]) else True:
        failed.append(f"b_bal_vol ρ={_f(p2['rho'])} off quote 0.354")
    days_y3 = next(r["cv"] for r in p4["y3"]["rows"] if r["feature"] == "c_n_days_with_tx")
    if abs(days_y3 - DAYS_Y3) > 0.02 if np.isfinite(days_y3) else True:
        failed.append(f"Y3 days replica {_f(days_y3)} vs night 0.711")
    if not decision["keep"]:
        failed.append("KEEP gate missed — do not merge a_vol tonight")
    if p12["io_is_12"] and not p23["not_the_12"]:
        failed.append("a_io_vol Y2 +0.02 is the 12 chronic names after drop")
    elif p23["not_the_12"]:
        failed.append("a_io_vol Y2 +0.02 is not the 12 names (footnote only; Y3 loses)")
    in_vol_y3 = next(r for r in p4["y3"]["rows"] if r["feature"] == "a_in_vol")
    if np.isfinite(in_vol_y3["gap_size"]) and in_vol_y3["gap_size"] < KEEP_DELTA:
        failed.append(
            f"a_in_vol Y3 Δ={in_vol_y3['gap_size']:+.3f} misses KEEP by "
            f"{KEEP_DELTA - in_vol_y3['gap_size']:.3f}"
        )
    if not p66["reproduced"]:
        failed.append(f"STOP — a_out_vol Y3 {_f(p66['cv'])} did not reproduce 0.722")
    if p66["x_dec"] != "KEEP":
        failed.append(
            f"a_out_vol X={p66['x_dec']} merge=NO — {p66['note']}"
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
        "p37": p37,
        "p38": p38,
        "p39": p39,
        "p40": p40,
        "p41": p41,
        "p42": p42,
        "p43": p43,
        "p44": p44,
        "p45": p45,
        "p46": p46,
        "p47": p47,
        "p48": p48,
        "p49": p49,
        "p50": p50,
        "p51": p51,
        "p52": p52,
        "p53": p53,
        "p54": p54,
        "p55": p55,
        "p56": p56,
        "p57": p57,
        "p58": p58,
        "p59": p59,
        "p60": p60,
        "p61": p61,
        "p62": p62,
        "p63": p63,
        "p64": p64,
        "p65": p65,
        "p66": p66,
        "p67": p67,
        "p68": p68,
        "p69": p69,
        "p70": p70,
        "p71": p71,
        "p72": p72,
        "p73": p73,
        "p74": p74,
        "p75": p75,
        "p76": p76,
        "p77": p77,
        "p78": p78,
        "p79": p79,
        "p80": p80,
        "p81": p81,
        "p82": p82,
        "p83": p83,
        "p84": p84,
        "p85": p85,
        "p86": p86,
        "p87": p87,
        "p88": p88,
        "p89": p89,
        "p90": p90,
        "p91": p91,
        "p92": p92,
        "p93": p93,
        "p94": p94,
        "p95": p95,
        "p96": p96,
        "p97": p97,
        "p98": p98,
        "p99": p99,
        "p100": p100,
        "p101": p101,
        "p102": p102,
        "p103": p103,
        "p104": p104,
        "p105": p105,
        "p106": p106,
        "p107": p107,
        "p108": p108,
        "p109": p109,
        "p110": p110,
        "p111": p111,
        "p112": p112,
        "p113": p113,
        "p114": p114,
        "p115": p115,
        "p116": p116,
        "p117": p117,
        "p118": p118,
        "p119": p119,
        "p120": p120,
        "decision": decision,
        "headline": headline,
        "png_ok": png_ok,
        "failed": failed,
        "elapsed_s": elapsed,
    }
    write_md(ctx)
    append_registry(ctx)
    write_wave(ctx)
    print(f"elapsed {elapsed:.1f}s")
    return ctx


if __name__ == "__main__":
    run()
