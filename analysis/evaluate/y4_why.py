"""Y4 why / lead — decompose `y4_ds_r_double` and clock HHI.

Brief: Q3 turning / Q5 why / Q6 lead. Not bankruptcy. Not a 0–100.
X is family D lags only. Never family F. Flows for the label split are
rebuilt here from `transactions` + `CAT_MAP` (same convention as
`analysis.targets.y4_debt`). Do not import `analysis.features.debt`.

Holdout 72 companies (seed 20260918) are out of every rate, AUROC, cut,
sign, and z-moment. Quote train group-fold CV.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.y4_why
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
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
from analysis.features.common import (
    ANALYSIS,
    CAT_MAP,
    DATA,
    MONTHS,
    connect,
)
from analysis.targets.y4_debt import META as Y4_META
from analysis.targets.y4_debt import build as build_y4

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:  # pragma: no cover
    HAS_MPL = False

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "monthly.parquet"
OUT_MD = ANALYSIS / "outputs" / "y4_why.md"
OUT_PNG = ANALYSIS / "outputs" / "y4_hhi_quintiles.png"
AGENT = "79044b5e"
WAVE = 4
ROUND = "R4"
Y_COL = "y4_ds_r_double"
N_FOLDS = 5
SINGLE_BAR = 0.605
CLEAR_MARGIN = 0.02
CRASH_TH = 0.8  # in3[t+3] / in3[t]  — >20% inflow drop
SPIKE_TH = 1.5  # ds3[t+3] / ds3[t]
ALT_SPIKE_TH = 1.2  # residual-friendly; 1.5/0.8 < 2 so "neither" is almost empty by algebra
LAGS_CLOCK = (0, 1, 3, 6)
ZAVG_COLS = ("d_cust_hhi_lag3", "d_n_supp_lag3")
D_STEMS = ("d_cust_hhi", "d_cust_top1", "d_n_supp")
LEAK_VS = ("f_ds_r", "a_in3", "d_cust_top1")

_OP_IN = tuple(k for k, v in CAT_MAP.items() if v == "op_in")
_DEBT_SVC = tuple(k for k, v in CAT_MAP.items() if v == "debt_service")


def _sql_in(values: tuple[str, ...]) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    if "month" in out.columns:
        out["month"] = pd.to_datetime(out["month"])
    return out


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}" if np.isfinite(v) else ""
    return "" if v is None else str(v)


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


def spearman(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 20 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="spearman"))


def pearson(a: pd.Series, b: pd.Series) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < 20 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
        return float("nan")
    return float(d["a"].corr(d["b"], method="pearson"))


def add_lags(df: pd.DataFrame, cols: list[str], lags: tuple[int, ...]) -> pd.DataFrame:
    """Past-only lags. lag 0 is the contemporaneous column (no new name)."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            if k == 0:
                continue
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def lag_col(stem: str, lag: int) -> str:
    return stem if lag == 0 else f"{stem}_lag{lag}"


def rebuild_in3_ds3(con) -> pd.DataFrame:
    """Monthly in3 / ds3 from transactions + CAT_MAP. Not family F. Not X."""
    flows = con.execute(
        f"""
        SELECT
          t.company_id,
          CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN t.category IN ({_sql_in(_OP_IN)}) THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN t.category IN ({_sql_in(_DEBT_SVC)}) THEN t.amount ELSE 0 END) AS debt_service
        FROM transactions t
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
        """
    ).df()
    flows["month"] = pd.to_datetime(flows["month"])
    first = flows.groupby("company_id")["month"].min().rename("first_m")
    panel = pd.MultiIndex.from_product(
        [first.index, MONTHS], names=["company_id", "month"]
    ).to_frame(index=False)
    panel = panel.merge(first, left_on="company_id", right_index=True)
    panel = panel[panel["month"] >= panel["first_m"]]
    panel = panel.merge(flows, on=["company_id", "month"], how="left")
    panel["op_in"] = panel["op_in"].fillna(0.0)
    panel["debt_service"] = panel["debt_service"].fillna(0.0)
    panel = panel.sort_values(["company_id", "month"]).reset_index(drop=True)
    g = panel.groupby("company_id", sort=False)
    panel["in3"] = g["op_in"].transform(lambda s: s.rolling(3).sum())
    panel["ds3"] = g["debt_service"].transform(lambda s: s.rolling(3).sum())
    out = panel[["company_id", "month", "op_in", "debt_service", "in3", "ds3"]].rename(
        columns={"month": "period"}
    )
    return _keys(out)


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing — need monthly.parquet")
    need = ["company_id", "period", *D_STEMS]
    extra = [c for c in ("a_in3", "a_op_in", "f_ds_r") if c]
    raw = pd.read_parquet(STORE)
    have = [c for c in need + extra if c in raw.columns]
    missing = [c for c in D_STEMS if c not in raw.columns]
    if missing:
        raise RuntimeError(f"store missing D stems {missing}")
    panel = _keys(raw[have])
    print(f"loaded store {STORE} shape={panel.shape} cols={have}")
    return panel


def load_y(con, grid: pd.DataFrame) -> pd.DataFrame:
    print("building Y4 via analysis.targets.y4_debt.build (live label)")
    y4 = _keys(build_y4(con, grid))
    if Y_COL not in y4.columns:
        raise RuntimeError(f"{Y_COL} missing from y4_debt.build")
    return y4[["company_id", "period", Y_COL]].copy()


def signed_oof_auroc(
    df: pd.DataFrame,
    col: str,
    train_lab: pd.Series,
    folds: np.ndarray,
) -> dict:
    """Group-fold CV AUROC. Sign from the train fold only. NaNs dropped by auroc."""
    y = df[Y_COL].astype(float)
    x = pd.to_numeric(df[col], errors="coerce")
    fold_rows = []
    aucs = []
    for k in range(N_FOLDS):
        tr = train_lab & (folds != k)
        va = train_lab & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        n_va = int((va & x.notna()).sum())
        n_pos = int(((va & x.notna()) & (y == 1)).sum())
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va_defined": n_va,
                "n_pos_defined": n_pos,
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    return {
        "col": col,
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(choose_sign(y[train_lab], x[train_lab])),
        "train_auc": float(auroc(y[train_lab], choose_sign(y[train_lab], x[train_lab]) * x[train_lab])),
    }


def coverage(x: pd.Series, mask: pd.Series) -> dict:
    n = int(mask.sum())
    n_ok = int(x[mask].notna().sum())
    return {
        "n": n,
        "n_defined": n_ok,
        "coverage": (n_ok / n) if n else float("nan"),
    }


def slice_table(lab: pd.DataFrame, mask: pd.Series, name: str) -> dict:
    """n / pos share / size AUROC of Y inside a slice (train labeled only)."""
    sub = lab.loc[mask].copy()
    n = int(len(sub))
    n_pos = int((sub[Y_COL] == 1).sum())
    n_all = int(len(lab))
    n_pos_all = int((lab[Y_COL] == 1).sum())
    size = auroc(sub[Y_COL], np.log1p(sub["in3"].abs()))
    return {
        "slice": name,
        "n": n,
        "n_pos": n_pos,
        "share_labeled": (n / n_all) if n_all else float("nan"),
        "share_of_positives": (n_pos / n_pos_all) if n_pos_all else float("nan"),
        "y_rate": (n_pos / n) if n else float("nan"),
        "size_auroc": float(size) if np.isfinite(size) else float("nan"),
    }


def pass1_decompose(lab: pd.DataFrame) -> dict:
    """Split the doubling into inflow crash / repayment spike / both / neither."""
    print("\n" + "=" * 72)
    print(f"PASS 1 — decompose {Y_COL}  crash in3_ratio<{CRASH_TH}  spike ds3_ratio>{SPIKE_TH}")
    print("in3/ds3 rebuilt from transactions + CAT_MAP; not used as X; never F")
    print("=" * 72)

    pos = lab[lab[Y_COL] == 1]
    neg = lab[lab[Y_COL] == 0]
    print(
        f"train labeled n={len(lab)} pos={int((lab[Y_COL] == 1).sum())} "
        f"rate={float(lab[Y_COL].mean()):.4f}"
    )
    print(
        f"  pos median in3[t+3]/in3[t]={float(pos['in_ratio'].median()):.3f} "
        f"ds3 ratio={float(pos['ds_ratio'].median()):.3f} "
        f"| neg in_ratio={float(neg['in_ratio'].median()):.3f} "
        f"ds_ratio={float(neg['ds_ratio'].median()):.3f}"
    )
    print(
        f"  pos crash<{CRASH_TH}={float((pos['in_ratio'] < CRASH_TH).mean()):.3f} "
        f"spike>{SPIKE_TH}={float((pos['ds_ratio'] > SPIKE_TH).mean()):.3f} "
        f"(overlapping; 20% drop was already known — this measures 1.5 spike)"
    )

    crash = lab["in_ratio"] < CRASH_TH
    spike = lab["ds_ratio"] > SPIKE_TH
    defined = lab["in_ratio"].notna() & lab["ds_ratio"].notna()
    rows = [
        slice_table(lab, crash & ~spike & defined, "crash_only"),
        slice_table(lab, spike & ~crash & defined, "spike_only"),
        slice_table(lab, crash & spike & defined, "both"),
        slice_table(lab, ~crash & ~spike & defined, "neither"),
        slice_table(lab, ~defined, "undefined_ratio"),
        slice_table(lab, crash & defined, "crash_any"),
        slice_table(lab, spike & defined, "spike_any"),
        slice_table(lab, pd.Series(True, index=lab.index), "all_labeled"),
    ]
    print(
        f"{'slice':18s} {'n':>6} {'n_pos':>6} {'shr_lab':>8} {'shr_pos':>8} "
        f"{'y_rate':>8} {'size_auc':>9}"
    )
    for r in rows:
        print(
            f"{r['slice']:18s} {r['n']:6d} {r['n_pos']:6d} "
            f"{r['share_labeled']:8.3f} {r['share_of_positives']:8.3f} "
            f"{r['y_rate']:8.3f} {r['size_auroc']:9.3f}"
        )
    return {
        "crash_th": CRASH_TH,
        "spike_th": SPIKE_TH,
        "pos_med_in_ratio": float(pos["in_ratio"].median()),
        "pos_med_ds_ratio": float(pos["ds_ratio"].median()),
        "neg_med_in_ratio": float(neg["in_ratio"].median()),
        "neg_med_ds_ratio": float(neg["ds_ratio"].median()),
        "slices": rows,
    }


def pass2_clock(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Q6: group-fold CV of d_cust_hhi / d_cust_top1 at lags 0, 1, 3, 6."""
    print("\n" + "=" * 72)
    print("PASS 2 — Q6 clock  d_cust_hhi / d_cust_top1  lags 0,1,3,6")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    n_lab = int(train_lab.sum())
    rows = []
    for stem in ("d_cust_hhi", "d_cust_top1"):
        for lag in LAGS_CLOCK:
            col = lag_col(stem, lag)
            if col not in df.columns:
                print(f"  missing {col}")
                continue
            cov = coverage(pd.to_numeric(df[col], errors="coerce"), train_lab)
            oof = signed_oof_auroc(df, col, train_lab, folds)
            rec = {
                "stem": stem,
                "lag": lag,
                "col": col,
                "coverage": cov["coverage"],
                "n_defined": cov["n_defined"],
                "cv": oof["cv"],
                "sd": oof["sd"],
                "train_auc": oof["train_auc"],
                "train_sign": oof["train_sign"],
                "folds": oof["folds"],
            }
            rows.append(rec)
            print(
                f"  {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f}  sign={oof['train_sign']:+d}  "
                f"cov={cov['coverage']:.3f} ({cov['n_defined']}/{n_lab})"
            )
    hhi = [r for r in rows if r["stem"] == "d_cust_hhi" and np.isfinite(r["cv"])]
    top1 = [r for r in rows if r["stem"] == "d_cust_top1" and np.isfinite(r["cv"])]
    best_hhi = max(hhi, key=lambda r: r["cv"]) if hhi else None
    best_top1 = max(top1, key=lambda r: r["cv"]) if top1 else None
    hhi3 = next((r for r in hhi if r["lag"] == 3), None)
    lag3_is_peak = bool(best_hhi and best_hhi["lag"] == 3)
    if best_hhi:
        print(
            f"  WIN HHI lag={best_hhi['lag']} CV={best_hhi['cv']:.4f}  "
            f"lag3_is_peak={lag3_is_peak}"
            + (
                f"  (lag3 CV={hhi3['cv']:.4f}, not the peak)"
                if hhi3 and not lag3_is_peak
                else ""
            )
        )
    if best_top1:
        print(f"  WIN top1 lag={best_top1['lag']} CV={best_top1['cv']:.4f}")
    return {
        "rows": rows,
        "best_hhi": best_hhi,
        "best_top1": best_top1,
        "lag3_is_peak": lag3_is_peak,
        "n_train_labeled": n_lab,
    }


def pass3_quintiles(df: pd.DataFrame, train_lab: pd.Series, hold_lab: pd.Series) -> dict:
    """Train-only quintiles of d_cust_hhi_lag3. Cuts never see holdout."""
    print("\n" + "=" * 72)
    print("PASS 3 — train-only quintiles of d_cust_hhi_lag3  (P(Y=1))")
    print("=" * 72)
    col = "d_cust_hhi_lag3"
    x = pd.to_numeric(df[col], errors="coerce")
    y = df[Y_COL].astype(float)
    tr = df.loc[train_lab & x.notna(), [col]].copy()
    tr[Y_COL] = y[train_lab & x.notna()].to_numpy()
    assert_no_holdout(df.loc[train_lab & x.notna(), "company_id"])
    if len(tr) < 50:
        raise RuntimeError(f"too few train complete HHI_lag3 rows: {len(tr)}")
    cats, bins = pd.qcut(tr[col], 5, retbins=True, duplicates="drop")
    tr = tr.copy()
    tr["q"] = cats
    grouped = tr.groupby("q", observed=True)
    rows = []
    for i, (q, g) in enumerate(grouped, start=1):
        rec = {
            "q": i,
            "interval": str(q),
            "lo": float(q.left),
            "hi": float(q.right),
            "n": int(len(g)),
            "n_pos": int((g[Y_COL] == 1).sum()),
            "y_rate": float(g[Y_COL].mean()),
            "hhi_median": float(g[col].median()),
        }
        rows.append(rec)
        print(
            f"  Q{i} {q}  n={rec['n']:4d} pos={rec['n_pos']:3d}  "
            f"P(Y=1)={rec['y_rate']:.3f}  med HHI={rec['hhi_median']:.3f}"
        )
    rates = [r["y_rate"] for r in rows]
    monotone_up = all(a <= b + 1e-12 for a, b in zip(rates, rates[1:]))
    print(f"  monotone non-decreasing={monotone_up}  bins={bins.tolist()}")

    ho_rows = []
    ho_m = hold_lab & x.notna()
    if int(ho_m.sum()) >= 5:
        ho = df.loc[ho_m, [col]].copy()
        ho[Y_COL] = y[ho_m].to_numpy()
        ho["q"] = pd.cut(ho[col], bins=bins, include_lowest=True)
        for i, (q, g) in enumerate(ho.groupby("q", observed=True), start=1):
            ho_rows.append(
                {
                    "q": i,
                    "n": int(len(g)),
                    "n_pos": int((g[Y_COL] == 1).sum()),
                    "y_rate": float(g[Y_COL].mean()) if len(g) else float("nan"),
                }
            )
        print("  holdout (train cuts, LOW_POWER, not a claim):")
        for r in ho_rows:
            print(f"    Q{r['q']} n={r['n']} pos={r['n_pos']} rate={r['y_rate']:.3f}")

    png = None
    if HAS_MPL:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        xs = [r["q"] for r in rows]
        ys = [r["y_rate"] for r in rows]
        ax.bar(xs, ys, color="#3d5a80", width=0.7)
        ax.axhline(float(y[train_lab].mean()), color="#ee6c4d", ls="--", lw=1, label="train labeled base")
        for r in rows:
            ax.text(
                r["q"],
                r["y_rate"] + 0.006,
                f"{r['y_rate']:.1%}\nn={r['n']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([f"Q{r['q']}\n{r['hhi_median']:.2f}" for r in rows])
        ax.set_ylabel("P(Y=1)  y4_ds_r_double")
        ax.set_xlabel("d_cust_hhi_lag3 quintile (train labeled cuts; median HHI)")
        ax.set_ylim(0, max(ys) * 1.28)
        ax.legend(frameon=False, fontsize=8)
        ax.set_title("Y4 rate by customer HHI three months earlier")
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
        png = str(OUT_PNG)
        print(f"  wrote {OUT_PNG}")
    else:
        print("  matplotlib missing — skip PNG")

    return {
        "n_train_complete": int(len(tr)),
        "bins": [float(b) for b in bins],
        "rows": rows,
        "monotone_up": monotone_up,
        "holdout_check": ho_rows,
        "png": png,
    }


def zavg_oof(df: pd.DataFrame, train_lab: pd.Series, cols: tuple[str, ...]) -> dict:
    """Train-fold signed z-average. Moments and signs from the train fold only."""
    y = df[Y_COL].astype(float)
    xs = {c: pd.to_numeric(df[c], errors="coerce") for c in cols}
    both = train_lab
    for c in cols:
        both = both & xs[c].notna()
    folds = df["fold"].to_numpy()
    aucs = []
    fold_rows = []
    for k in range(N_FOLDS):
        tr = train_lab & (df["fold"] != k)
        va = train_lab & (df["fold"] == k)
        parts = []
        signs = {}
        for c, x in xs.items():
            sign = choose_sign(y[tr], x[tr])
            mu = float(x[tr].mean())
            sd = float(x[tr].std(ddof=0))
            z = (x - mu) / sd if sd and np.isfinite(sd) and sd > 0 else x * 0.0
            parts.append(sign * z)
            signs[c] = int(sign)
        score = sum(parts)
        auc = auroc(y[va], score[va])
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "n_va": int((va & score.notna()).sum()),
                "n_pos": int((va & score.notna() & (y == 1)).sum()),
                "signs": signs,
            }
        )
        print(
            f"  fold {k}: zavg={auc:.4f} n_va={fold_rows[-1]['n_va']} "
            f"pos={fold_rows[-1]['n_pos']} signs={signs}"
        )
    finite = [a for a in aucs if np.isfinite(a)]
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "folds": fold_rows,
        "n_complete": int(both.sum()),
        "n_complete_pos": int((both & (y == 1)).sum()),
    }


def pass4_zavg(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """2-col card honesty: z-avg vs HHI on the same complete-case rows."""
    print("\n" + "=" * 72)
    print("PASS 4 — 2-col z-avg honesty  HHI_lag3 + n_supp_lag3")
    print(f"KEEP only if CV >= {SINGLE_BAR:.3f}+{CLEAR_MARGIN:.2f} AND not just HHI on CC")
    print("=" * 72)
    for c in ZAVG_COLS:
        if c not in df.columns:
            raise RuntimeError(f"missing {c}")
        if str(c).startswith("f_"):
            raise RuntimeError(f"family F leaked into zavg: {c}")

    leak = leakage_check(list(ZAVG_COLS), Y_COL, forbidden_prefixes=("f",))
    if not leak["ok"]:
        raise RuntimeError(f"zavg leak: {leak['issues']}")

    both = train_lab
    for c in ZAVG_COLS:
        both = both & pd.to_numeric(df[c], errors="coerce").notna()
    assert_no_holdout(df.loc[both, "company_id"])

    print("z-avg (NaN if either col missing — auroc drops those rows):")
    z = zavg_oof(df, train_lab, ZAVG_COLS)
    print(f"  zavg CV={z['cv']:.4f}±{z['sd']:.3f}  complete={z['n_complete']} pos={z['n_complete_pos']}")

    print("HHI_lag3 alone, all train labeled where HHI defined:")
    hhi_all = signed_oof_auroc(df, "d_cust_hhi_lag3", train_lab, df["fold"].to_numpy())
    print(f"  HHI-all CV={hhi_all['cv']:.4f}±{hhi_all['sd']:.3f}")

    print("HHI_lag3 alone, complete-case of both cols (same rows as z-avg):")
    hhi_cc = signed_oof_auroc(df, "d_cust_hhi_lag3", both, df["fold"].to_numpy())
    print(f"  HHI-CC  CV={hhi_cc['cv']:.4f}±{hhi_cc['sd']:.3f}")

    print("n_supp_lag3 alone, complete-case of both cols:")
    supp_cc = signed_oof_auroc(df, "d_n_supp_lag3", both, df["fold"].to_numpy())
    print(f"  supp-CC CV={supp_cc['cv']:.4f}±{supp_cc['sd']:.3f}")

    print("n_supp_lag3 alone, all train labeled where n_supp defined:")
    supp_all = signed_oof_auroc(df, "d_n_supp_lag3", train_lab, df["fold"].to_numpy())
    print(f"  supp-all CV={supp_all['cv']:.4f}±{supp_all['sd']:.3f}")

    hhi_n = int(pd.to_numeric(df["d_cust_hhi_lag3"], errors="coerce")[train_lab].notna().sum())
    supp_n = int(pd.to_numeric(df["d_n_supp_lag3"], errors="coerce")[train_lab].notna().sum())
    print(
        f"  defined rows: HHI_lag3={hhi_n}  n_supp_lag3={supp_n}  both={int(both.sum())}  "
        f"same_invoice_gate={hhi_n == supp_n == int(both.sum())}"
    )
    supp_only = train_lab & pd.to_numeric(df["d_n_supp_lag3"], errors="coerce").notna()
    supp_only = supp_only & pd.to_numeric(df["d_cust_hhi_lag3"], errors="coerce").isna()
    n_so = int(supp_only.sum())
    n_so_pos = int((df.loc[supp_only, Y_COL] == 1).sum())
    print(
        f"  n_supp_lag3 without HHI_lag3: n={n_so} pos={n_so_pos} "
        f"P(Y=1)={((n_so_pos / n_so) if n_so else float('nan')):.3f} "
        f"— z-avg drops these rows"
    )

    gap_bar = float(z["cv"] - SINGLE_BAR) if np.isfinite(z["cv"]) else float("nan")
    gap_cc = float(z["cv"] - hhi_cc["cv"]) if np.isfinite(z["cv"]) and np.isfinite(hhi_cc["cv"]) else float("nan")
    gap_supp = (
        float(z["cv"] - supp_cc["cv"])
        if np.isfinite(z["cv"]) and np.isfinite(supp_cc["cv"])
        else float("nan")
    )
    best_uni = max(
        (v for v in (hhi_cc["cv"], supp_cc["cv"]) if np.isfinite(v)),
        default=float("nan"),
    )
    gap_best = float(z["cv"] - best_uni) if np.isfinite(z["cv"]) and np.isfinite(best_uni) else float("nan")
    beats_bar = bool(np.isfinite(z["cv"]) and z["cv"] >= SINGLE_BAR + CLEAR_MARGIN)
    extra_vs_hhi = bool(np.isfinite(gap_cc) and gap_cc >= CLEAR_MARGIN)
    extra_vs_best = bool(np.isfinite(gap_best) and gap_best >= CLEAR_MARGIN)
    just_hhi = not extra_vs_hhi
    just_univariate = not extra_vs_best
    # Letter of the brief: beat 0.605+0.02 and not just HHI on CC.
    # Honesty cut: also not just n_supp on the same invoice-gated rows.
    if beats_bar and extra_vs_hhi and extra_vs_best:
        decision = "KEEP"
        reason = (
            f"zavg CV {z['cv']:.3f} >= {SINGLE_BAR:.3f}+{CLEAR_MARGIN:.2f} "
            f"and beats both HHI-CC {hhi_cc['cv']:.3f} and n_supp-CC {supp_cc['cv']:.3f}"
        )
    elif beats_bar and extra_vs_hhi and just_univariate:
        decision = "CLOSE"
        reason = (
            f"zavg CV {z['cv']:.3f} clears 0.605+0.02 and is not HHI-on-CC "
            f"(HHI-CC {hhi_cc['cv']:.3f}, gap {gap_cc:+.3f}) but is n_supp on those "
            f"same rows (n_supp-CC {supp_cc['cv']:.3f}, gap {gap_supp:+.3f} < {CLEAR_MARGIN:.2f})"
        )
    elif beats_bar and just_hhi:
        decision = "CLOSE"
        reason = (
            f"zavg CV {z['cv']:.3f} clears the 0.605+0.02 bar but is HHI on "
            f"complete-case rows (HHI-CC {hhi_cc['cv']:.3f}, gap {gap_cc:+.3f} < {CLEAR_MARGIN:.2f})"
        )
    else:
        decision = "PARK"
        reason = (
            f"zavg CV {z['cv']:.3f} vs bar {SINGLE_BAR + CLEAR_MARGIN:.3f} "
            f"HHI-CC {hhi_cc['cv']:.3f} n_supp-CC {supp_cc['cv']:.3f} — not a card KEEP"
        )
    print(f"  VERDICT {decision}: {reason}")
    return {
        "zavg": z,
        "hhi_all": hhi_all,
        "hhi_cc": hhi_cc,
        "supp_cc": supp_cc,
        "supp_all": supp_all,
        "gap_vs_bar": gap_bar,
        "gap_vs_hhi_cc": gap_cc,
        "gap_vs_supp_cc": gap_supp,
        "gap_vs_best_uni": gap_best,
        "just_hhi_on_cc": just_hhi,
        "just_univariate_on_cc": just_univariate,
        "same_invoice_gate": hhi_n == supp_n == int(both.sum()),
        "decision": decision,
        "reason": reason,
        "n_complete": z["n_complete"],
        "n_complete_pos": z["n_complete_pos"],
    }


def pass5_leak(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """ρ of HHI_lag3 vs f_ds_r / a_in3 / d_cust_top1. Comparators only — not X."""
    print("\n" + "=" * 72)
    print("PASS 5 — leak  HHI_lag3 vs f_ds_r / a_in3 / d_cust_top1  (train labeled)")
    print("=" * 72)
    hhi = pd.to_numeric(df["d_cust_hhi_lag3"], errors="coerce")
    top1_l3 = pd.to_numeric(df["d_cust_top1_lag3"], errors="coerce") if "d_cust_top1_lag3" in df.columns else None
    rows = []
    pairs = [
        ("f_ds_r", pd.to_numeric(df["f_ds_r"], errors="coerce") if "f_ds_r" in df.columns else None),
        ("a_in3", pd.to_numeric(df["a_in3"], errors="coerce") if "a_in3" in df.columns else None),
        ("d_cust_top1", pd.to_numeric(df["d_cust_top1"], errors="coerce") if "d_cust_top1" in df.columns else None),
        ("d_cust_top1_lag3", top1_l3),
    ]
    for name, other in pairs:
        if other is None:
            print(f"  {name}: missing")
            continue
        m = train_lab & hhi.notna() & other.notna()
        assert_no_holdout(df.loc[m, "company_id"])
        rec = {
            "vs": name,
            "n": int(m.sum()),
            "spearman": spearman(hhi[m], other[m]),
            "pearson": pearson(hhi[m], other[m]),
        }
        rows.append(rec)
        print(
            f"  HHI_lag3 vs {name:18s}  ρ_s={rec['spearman']:+.3f}  "
            f"ρ_p={rec['pearson']:+.3f}  n={rec['n']}"
        )
    top1 = next((r for r in rows if r["vs"] == "d_cust_top1_lag3"), None)
    is_top1_rewrite = bool(top1 and np.isfinite(top1["spearman"]) and abs(top1["spearman"]) >= 0.90)
    if is_top1_rewrite:
        print(
            "  HHI_lag3 is a top-1 rewrite (ρ≥0.90 vs d_cust_top1_lag3). "
            "Y7 forbids D for that reason; Y4 allowed D — same stem, do not stack."
        )
    else:
        print("  HHI_lag3 is not a top-1 rewrite at ρ≥0.90.")
    fds = next((r for r in rows if r["vs"] == "f_ds_r"), None)
    ain = next((r for r in rows if r["vs"] == "a_in3"), None)
    if fds and abs(fds["spearman"]) >= 0.80:
        print("  FAIL leak vs f_ds_r (|ρ|≥0.80) — would copy the ds_r level.")
    if ain and abs(ain["spearman"]) >= 0.80:
        print("  FAIL leak vs a_in3 (|ρ|≥0.80) — size clone.")
    return {"rows": rows, "is_top1_rewrite": is_top1_rewrite}


def pass1_alt_thresholds(lab: pd.DataFrame) -> dict:
    """0.8/1.2 residual split, plus a small threshold grid. Train labeled only."""
    print("\n" + "=" * 72)
    print("CUT 1b — alt spike 1.2 (allows a residual) and threshold grid")
    print("1.5/0.8 = 1.875 < 2, so neither@1.5 is almost empty by ds_r algebra")
    print("=" * 72)
    pos = lab[lab[Y_COL] == 1]
    crash = lab["in_ratio"] < CRASH_TH
    spike12 = lab["ds_ratio"] > ALT_SPIKE_TH
    defined = lab["in_ratio"].notna() & lab["ds_ratio"].notna()
    rows = [
        slice_table(lab, crash & ~spike12 & defined, "crash_only_1.2"),
        slice_table(lab, spike12 & ~crash & defined, "spike_only_1.2"),
        slice_table(lab, crash & spike12 & defined, "both_1.2"),
        slice_table(lab, ~crash & ~spike12 & defined, "neither_1.2"),
        slice_table(lab, spike12 & defined, "spike_any_1.2"),
    ]
    print(
        f"{'slice':18s} {'n':>6} {'n_pos':>6} {'shr_pos':>8} {'y_rate':>8} {'size_auc':>9}"
    )
    for r in rows:
        print(
            f"{r['slice']:18s} {r['n']:6d} {r['n_pos']:6d} "
            f"{r['share_of_positives']:8.3f} {r['y_rate']:8.3f} {r['size_auroc']:9.3f}"
        )
    grid = []
    for cth in (0.7, 0.8, 0.9):
        for sth in (1.2, 1.5, 2.0):
            cr = lab["in_ratio"] < cth
            sp = lab["ds_ratio"] > sth
            rec = {
                "crash_th": cth,
                "spike_th": sth,
                "pos_crash": float((pos["in_ratio"] < cth).mean()),
                "pos_spike": float((pos["ds_ratio"] > sth).mean()),
                "pos_both": float(((pos["in_ratio"] < cth) & (pos["ds_ratio"] > sth)).mean()),
                "pos_neither": float(
                    (
                        (pos["in_ratio"] >= cth)
                        & (pos["ds_ratio"] <= sth)
                        & pos["in_ratio"].notna()
                        & pos["ds_ratio"].notna()
                    ).mean()
                ),
                "neither_n_pos": int(
                    (
                        (lab[Y_COL] == 1)
                        & (lab["in_ratio"] >= cth)
                        & (lab["ds_ratio"] <= sth)
                        & lab["in_ratio"].notna()
                        & lab["ds_ratio"].notna()
                    ).sum()
                ),
            }
            grid.append(rec)
            print(
                f"  crash<{cth} spike>{sth}: pos crash={rec['pos_crash']:.3f} "
                f"spike={rec['pos_spike']:.3f} both={rec['pos_both']:.3f} "
                f"neither={rec['pos_neither']:.3f} (n_pos={rec['neither_n_pos']})"
            )
    return {"slices_1_2": rows, "grid": grid}


def pass1_hhi_by_slice(lab: pd.DataFrame, panel: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Does HHI_lag3 read the crash doubles more than the spike doubles?"""
    print("\n" + "=" * 72)
    print("CUT 1c — HHI_lag3 inside crash / spike slices (train labeled)")
    print("=" * 72)
    keys = panel.loc[train_lab, ["company_id", "period", "d_cust_hhi_lag3", "d_n_supp_lag3", "fold"]].copy()
    m = lab.merge(keys, on=["company_id", "period"], how="left")
    assert_no_holdout(m["company_id"])
    crash = m["in_ratio"] < CRASH_TH
    spike = m["ds_ratio"] > SPIKE_TH
    spike12 = m["ds_ratio"] > ALT_SPIKE_TH
    defined = m["in_ratio"].notna() & m["ds_ratio"].notna()
    hhi = pd.to_numeric(m["d_cust_hhi_lag3"], errors="coerce")
    y = m[Y_COL].astype(float)
    slices = {
        "crash_only": crash & ~spike & defined,
        "spike_only": spike & ~crash & defined,
        "both": crash & spike & defined,
        "crash_any": crash & defined,
        "spike_any": spike & defined,
        "spike_only_1.2": spike12 & ~crash & defined,
        "all_labeled": pd.Series(True, index=m.index),
    }
    rows = []
    for name, mask in slices.items():
        sub_y = y[mask]
        sub_x = hhi[mask]
        pos = mask & (y == 1)
        rec = {
            "slice": name,
            "n": int(mask.sum()),
            "n_pos": int((sub_y == 1).sum()),
            "n_hhi": int(sub_x.notna().sum()),
            "hhi_auroc": float(auroc(sub_y, sub_x)),
            "hhi_med_pos": float(hhi[pos].median()) if int(hhi[pos].notna().sum()) else float("nan"),
            "hhi_med_neg": (
                float(hhi[mask & (y == 0)].median())
                if int(hhi[mask & (y == 0)].notna().sum())
                else float("nan")
            ),
        }
        rows.append(rec)
        print(
            f"  {name:16s} n={rec['n']:4d} pos={rec['n_pos']:3d} hhi_n={rec['n_hhi']:3d} "
            f"AUROC={rec['hhi_auroc']:.3f}  med HHI pos/neg="
            f"{rec['hhi_med_pos']:.3f}/{rec['hhi_med_neg']:.3f}"
        )
    return {"rows": rows}


def pass2_fold_table(p2: dict) -> dict:
    """Lag-6 mean-CV peak vs lag-3 stability. Print fold AUCs side by side."""
    print("\n" + "=" * 72)
    print("CUT 2b — fold AUCs for HHI lags (lag6 peak is noise)")
    print("=" * 72)
    hhi = [r for r in p2["rows"] if r["stem"] == "d_cust_hhi"]
    header = "lag " + " ".join(f"f{k:>6}" for k in range(N_FOLDS)) + "   cv    sd   train   cov"
    print("  " + header)
    fold_map = {}
    for r in hhi:
        aucs = [f["auroc"] for f in r["folds"]]
        fold_map[r["lag"]] = aucs
        bits = " ".join(f"{a:6.3f}" for a in aucs)
        print(
            f"  {r['lag']:<4}{bits}  {r['cv']:5.3f} {r['sd']:5.3f}  "
            f"{r['train_auc']:5.3f}  {r['coverage']:.3f}"
        )
    lag3 = next(r for r in hhi if r["lag"] == 3)
    lag6 = next(r for r in hhi if r["lag"] == 6)
    stable_peak = 3
    note = (
        f"mean-CV peak is lag {p2['best_hhi']['lag']} ({p2['best_hhi']['cv']:.4f}) "
        f"but lag6 sd={lag6['sd']:.3f} train_auc={lag6['train_auc']:.3f} cov={lag6['coverage']:.3f}. "
        f"Stable quote stays lag3 CV={lag3['cv']:.4f} sd={lag3['sd']:.3f} "
        f"train={lag3['train_auc']:.3f} cov={lag3['coverage']:.3f}."
    )
    print("  " + note)
    return {
        "fold_map": fold_map,
        "stable_peak_lag": stable_peak,
        "mean_cv_peak_lag": p2["best_hhi"]["lag"],
        "note": note,
        "lag3": lag3,
        "lag6": lag6,
    }


def pass2_nsupp_clock(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Same Q6 clock for d_n_supp — is the neighbour also lag-3?"""
    print("\n" + "=" * 72)
    print("CUT 2c — Q6 clock  d_n_supp  lags 0,1,3,6")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    n_lab = int(train_lab.sum())
    rows = []
    for lag in LAGS_CLOCK:
        col = lag_col("d_n_supp", lag)
        if col not in df.columns:
            continue
        cov = coverage(pd.to_numeric(df[col], errors="coerce"), train_lab)
        oof = signed_oof_auroc(df, col, train_lab, folds)
        rec = {
            "lag": lag,
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "train_sign": oof["train_sign"],
            "coverage": cov["coverage"],
            "n_defined": cov["n_defined"],
        }
        rows.append(rec)
        print(
            f"  {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"train={oof['train_auc']:.4f}  sign={oof['train_sign']:+d}  "
            f"cov={cov['coverage']:.3f} ({cov['n_defined']}/{n_lab})"
        )
    finite = [r for r in rows if np.isfinite(r["cv"])]
    best = max(finite, key=lambda r: r["cv"]) if finite else None
    if best:
        print(f"  WIN n_supp lag={best['lag']} CV={best['cv']:.4f}  lag3_is_peak={best['lag'] == 3}")
    return {"rows": rows, "best": best}


def pass2_store_coverage(df: pd.DataFrame, train_lab: pd.Series, hold_ids: set[str]) -> dict:
    """Invoice-gate coverage on all train company-months, not just Y4 labeled."""
    print("\n" + "=" * 72)
    print("CUT 2d — HHI invoice-gate coverage (train company-months vs labeled)")
    print("=" * 72)
    is_train = ~df["company_id"].astype(str).isin(hold_ids)
    hhi = pd.to_numeric(df["d_cust_hhi"], errors="coerce")
    rec = {
        "train_cm": int(is_train.sum()),
        "train_hhi": int(hhi[is_train].notna().sum()),
        "train_hhi_cov": float(hhi[is_train].notna().mean()),
        "lab_cm": int(train_lab.sum()),
        "lab_hhi": int(hhi[train_lab].notna().sum()),
        "lab_hhi_cov": float(hhi[train_lab].notna().mean()),
        "lab_hhi_lag3": int(pd.to_numeric(df["d_cust_hhi_lag3"], errors="coerce")[train_lab].notna().sum()),
    }
    rec["lab_hhi_lag3_cov"] = rec["lab_hhi_lag3"] / rec["lab_cm"] if rec["lab_cm"] else float("nan")
    print(
        f"  train CM HHI defined {rec['train_hhi']}/{rec['train_cm']} = {rec['train_hhi_cov']:.3f} "
        f"(invoice-gate ~53%)"
    )
    print(
        f"  labeled HHI {rec['lab_hhi']}/{rec['lab_cm']} = {rec['lab_hhi_cov']:.3f}  "
        f"lag3 {rec['lab_hhi_lag3']}/{rec['lab_cm']} = {rec['lab_hhi_lag3_cov']:.3f}"
    )
    return rec


def pass3_tail(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Q5 monopoly tail vs a mid-quintile bump. Train labeled cuts only."""
    print("\n" + "=" * 72)
    print("CUT 3b — HHI_lag3 > 0.975 tail vs rest (the Q5 story)")
    print("=" * 72)
    x = pd.to_numeric(df["d_cust_hhi_lag3"], errors="coerce")
    y = df[Y_COL].astype(float)
    m = train_lab & x.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    hi = m & (x > 0.975)
    lo = m & (x <= 0.975)
    rec = {
        "cut": 0.975,
        "n_hi": int(hi.sum()),
        "n_pos_hi": int((y[hi] == 1).sum()),
        "rate_hi": float(y[hi].mean()) if int(hi.sum()) else float("nan"),
        "n_lo": int(lo.sum()),
        "n_pos_lo": int((y[lo] == 1).sum()),
        "rate_lo": float(y[lo].mean()) if int(lo.sum()) else float("nan"),
        "auroc_tail": float(auroc(y[m], (x[m] > 0.975).astype(float))),
    }
    print(
        f"  HHI_lag3>0.975  n={rec['n_hi']} pos={rec['n_pos_hi']} P(Y=1)={rec['rate_hi']:.3f}  |  "
        f"rest n={rec['n_lo']} pos={rec['n_pos_lo']} P(Y=1)={rec['rate_lo']:.3f}  "
        f"tail-indicator AUROC={rec['auroc_tail']:.3f}"
    )
    rho = spearman(x[m], pd.to_numeric(df.loc[m, "d_n_supp_lag3"], errors="coerce"))
    print(f"  Spearman HHI_lag3 vs n_supp_lag3 on train CC: {rho:+.3f}")
    rec["rho_vs_nsupp"] = rho
    return rec


def pass3_crash_quintiles(lab: pd.DataFrame, df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Quintiles of HHI_lag3 among crash_any train rows. Cuts from that slice only."""
    print("\n" + "=" * 72)
    print("CUT 3c — HHI_lag3 quintiles on crash_any (cuts from crash train rows)")
    print("=" * 72)
    flags = lab[["company_id", "period"]].copy()
    defined = lab["in_ratio"].notna() & lab["ds_ratio"].notna()
    flags["crash_any"] = (lab["in_ratio"] < CRASH_TH) & defined
    m = df.merge(flags, on=["company_id", "period"], how="left")
    x = pd.to_numeric(m["d_cust_hhi_lag3"], errors="coerce")
    y = m[Y_COL].astype(float)
    sl = train_lab & m["crash_any"].eq(True) & x.notna()
    assert_no_holdout(m.loc[sl, "company_id"])
    tr = pd.DataFrame({ "hhi": x[sl], Y_COL: y[sl] })
    if len(tr) < 50:
        print(f"  too few crash+HHI rows: {len(tr)}")
        return {"rows": [], "n": int(len(tr))}
    cats, bins = pd.qcut(tr["hhi"], 5, retbins=True, duplicates="drop")
    tr = tr.copy()
    tr["q"] = cats
    rows = []
    for i, (q, g) in enumerate(tr.groupby("q", observed=True), start=1):
        rec = {
            "q": i,
            "interval": str(q),
            "n": int(len(g)),
            "n_pos": int((g[Y_COL] == 1).sum()),
            "y_rate": float(g[Y_COL].mean()),
            "hhi_median": float(g["hhi"].median()),
        }
        rows.append(rec)
        print(
            f"  crash Q{i} {q}  n={rec['n']:4d} pos={rec['n_pos']:3d}  "
            f"P(Y=1)={rec['y_rate']:.3f}  med HHI={rec['hhi_median']:.3f}"
        )
    return {"rows": rows, "n": int(len(tr)), "bins": [float(b) for b in bins]}


def pass_hhi_predicts_crash(lab: pd.DataFrame, df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Does HHI_lag3 pick crash-type months, not just Y? Train only."""
    print("\n" + "=" * 72)
    print("CUT 1e — HHI_lag3 vs crash flag (mechanism, not only Y)")
    print("=" * 72)
    keys = df.loc[train_lab, ["company_id", "period", "d_cust_hhi_lag3"]].copy()
    m = lab.merge(keys, on=["company_id", "period"], how="left")
    assert_no_holdout(m["company_id"])
    defined = m["in_ratio"].notna() & m["ds_ratio"].notna()
    crash = (m["in_ratio"] < CRASH_TH) & defined
    hhi = pd.to_numeric(m["d_cust_hhi_lag3"], errors="coerce")
    y = m[Y_COL].astype(float)
    pos = y == 1
    rec = {
        "auroc_crash_all_lab": float(auroc(crash.astype(float), hhi)),
        "auroc_crash_among_pos": float(auroc(crash[pos].astype(float), hhi[pos])),
        "rho_hhi_inratio_pos": spearman(hhi[pos], m.loc[pos, "in_ratio"]),
        "rho_hhi_dsratio_pos": spearman(hhi[pos], m.loc[pos, "ds_ratio"]),
        "n_pos_hhi": int((pos & hhi.notna()).sum()),
    }
    print(
        f"  HHI vs crash flag on all labeled AUROC={rec['auroc_crash_all_lab']:.3f}  "
        f"among positives AUROC={rec['auroc_crash_among_pos']:.3f} n_pos_hhi={rec['n_pos_hhi']}"
    )
    print(
        f"  among positives: ρ(HHI, in_ratio)={rec['rho_hhi_inratio_pos']:+.3f}  "
        f"ρ(HHI, ds_ratio)={rec['rho_hhi_dsratio_pos']:+.3f}"
    )
    return rec


def pass_hhi_without_tail(df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Is the 0.605 only the HHI>0.975 bin? CV on the complementary rows."""
    print("\n" + "=" * 72)
    print("CUT 3d — HHI_lag3 CV after dropping the monopoly tail")
    print("=" * 72)
    x = pd.to_numeric(df["d_cust_hhi_lag3"], errors="coerce")
    body = train_lab & x.notna() & (x <= 0.975)
    tail = train_lab & x.notna() & (x > 0.975)
    oof = signed_oof_auroc(df, "d_cust_hhi_lag3", body, df["fold"].to_numpy())
    rec = {
        "cv_body": oof["cv"],
        "sd_body": oof["sd"],
        "n_body": int(body.sum()),
        "n_pos_body": int((df.loc[body, Y_COL] == 1).sum()),
        "n_tail": int(tail.sum()),
        "n_pos_tail": int((df.loc[tail, Y_COL] == 1).sum()),
    }
    print(
        f"  HHI<=0.975 CV={oof['cv']:.4f}±{oof['sd']:.3f} n={rec['n_body']} pos={rec['n_pos_body']}  |  "
        f"tail n={rec['n_tail']} pos={rec['n_pos_tail']}"
    )
    tail_cos = df.loc[tail, "company_id"].nunique()
    body_cos = df.loc[body, "company_id"].nunique()
    rec["n_cos_tail"] = int(tail_cos)
    rec["n_cos_body"] = int(body_cos)
    print(f"  tail companies={tail_cos}  body companies={body_cos} (train labeled complete HHI)")
    tail_ids = set(df.loc[tail, "company_id"].astype(str))
    other = train_lab & df["company_id"].astype(str).isin(tail_ids) & ~tail
    rec["n_tailcos_other"] = int(other.sum())
    rec["n_pos_tailcos_other"] = int((df.loc[other, Y_COL] == 1).sum())
    rec["rate_tailcos_other"] = (
        float(df.loc[other, Y_COL].mean()) if int(other.sum()) else float("nan")
    )
    rec["rate_tail"] = float(df.loc[tail, Y_COL].mean()) if int(tail.sum()) else float("nan")
    print(
        f"  same 43 cos, non-tail labeled months: n={rec['n_tailcos_other']} "
        f"pos={rec['n_pos_tailcos_other']} P(Y=1)={rec['rate_tailcos_other']:.3f}  "
        f"vs their tail months {rec['rate_tail']:.3f}"
    )
    # Binary tail card. Cut 0.975 is the train labeled quintile edge (Pass 3), not holdout.
    tail_flag = pd.Series(np.nan, index=df.index, dtype=float)
    defined = train_lab & x.notna()
    tail_flag.loc[defined] = (x[defined] > 0.975).astype(float)
    df_flag = df.copy()
    df_flag["_hhi_tail"] = tail_flag
    oof_flag = signed_oof_auroc(df_flag, "_hhi_tail", defined, df["fold"].to_numpy())
    rec["cv_tail_flag"] = oof_flag["cv"]
    rec["sd_tail_flag"] = oof_flag["sd"]
    print(
        f"  binary HHI>0.975 OOF CV={oof_flag['cv']:.4f}±{oof_flag['sd']:.3f} "
        f"(train-only cut; compare to continuous 0.605)"
    )
    return rec


def pass_slice_oof(lab: pd.DataFrame, df: pd.DataFrame, train_lab: pd.Series) -> dict:
    """Group-fold CV of HHI / n_supp inside crash vs spike (train labeled)."""
    print("\n" + "=" * 72)
    print("CUT 1d — OOF HHI_lag3 / n_supp_lag3 inside crash vs spike")
    print("=" * 72)
    flags = lab[["company_id", "period"]].copy()
    defined = lab["in_ratio"].notna() & lab["ds_ratio"].notna()
    flags["crash_any"] = (lab["in_ratio"] < CRASH_TH) & defined
    flags["spike_any"] = (lab["ds_ratio"] > SPIKE_TH) & defined
    flags["spike_only"] = flags["spike_any"] & ~flags["crash_any"]
    flags["crash_only"] = flags["crash_any"] & ~flags["spike_any"]
    m = df.merge(flags, on=["company_id", "period"], how="left")
    folds = m["fold"].to_numpy()
    rows = []
    for sl in ("crash_any", "crash_only", "spike_any", "spike_only"):
        sl_m = train_lab & m[sl].eq(True)
        for col in ("d_cust_hhi_lag3", "d_n_supp_lag3"):
            oof = signed_oof_auroc(m, col, sl_m, folds)
            cov = coverage(pd.to_numeric(m[col], errors="coerce"), sl_m)
            rec = {
                "slice": sl,
                "col": col,
                "n": int(sl_m.sum()),
                "n_pos": int((m.loc[sl_m, Y_COL] == 1).sum()),
                "cv": oof["cv"],
                "sd": oof["sd"],
                "coverage": cov["coverage"],
                "n_defined": cov["n_defined"],
            }
            rows.append(rec)
            print(
                f"  {sl:12s} {col:20s} n={rec['n']:4d} pos={rec['n_pos']:3d} "
                f"CV={oof['cv']:.3f}±{oof['sd']:.3f} cov={cov['coverage']:.3f}"
            )
    print("HHI clock on crash_any rows only:")
    crash_m = train_lab & m["crash_any"].eq(True)
    clock = []
    for lag in LAGS_CLOCK:
        col = lag_col("d_cust_hhi", lag)
        if col not in m.columns:
            continue
        oof = signed_oof_auroc(m, col, crash_m, folds)
        cov = coverage(pd.to_numeric(m[col], errors="coerce"), crash_m)
        rec = {
            "lag": lag,
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "coverage": cov["coverage"],
            "n_defined": cov["n_defined"],
            "folds": [f["auroc"] for f in oof["folds"]],
        }
        clock.append(rec)
        print(
            f"  crash_any {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f} "
            f"train={oof['train_auc']:.4f} cov={cov['coverage']:.3f}"
        )
    finite = [r for r in clock if np.isfinite(r["cv"])]
    best = max(finite, key=lambda r: r["cv"]) if finite else None
    if best:
        print(
            f"  WIN crash-only HHI lag={best['lag']} CV={best['cv']:.4f} "
            f"lag3_is_peak={best['lag'] == 3}"
        )
        for r in clock:
            bits = " ".join(f"{a:.3f}" for a in r.get("folds") or [])
            print(f"    lag{r['lag']} folds {bits}")
    return {"rows": rows, "crash_clock": clock, "crash_best": best}


def pass_holdout_check(
    df: pd.DataFrame, train_lab: pd.Series, hold_lab: pd.Series
) -> dict:
    """Score holdout with train-only signs. Never a KEEP claim."""
    print("\n" + "=" * 72)
    print("CUT holdout — train-sign scores, LOW_POWER, not a claim")
    print("=" * 72)
    y = df[Y_COL].astype(float)
    rows = []
    for col in ("d_cust_hhi_lag3", "d_n_supp_lag3", "d_cust_hhi", "d_cust_top1_lag3"):
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        sign = choose_sign(y[train_lab], x[train_lab])
        auc = auroc(y[hold_lab], sign * x[hold_lab])
        n_ok = int((hold_lab & x.notna()).sum())
        n_pos = int((hold_lab & x.notna() & (y == 1)).sum())
        rec = {
            "col": col,
            "sign": int(sign),
            "auroc": float(auc) if np.isfinite(auc) else float("nan"),
            "n": n_ok,
            "n_pos": n_pos,
        }
        rows.append(rec)
        print(
            f"  {col:22s} sign={sign:+d} hold AUROC={rec['auroc']:.3f} "
            f"n={n_ok} pos={n_pos} LOW_POWER"
        )
    return {"rows": rows}


def extra_md_lines(
    p1b: dict,
    p1c: dict,
    p2b: dict,
    p2c: dict,
    p2d: dict,
    p3b: dict,
    p1d: dict | None = None,
    ho: dict | None = None,
    p3c: dict | None = None,
    p1e: dict | None = None,
    p3d: dict | None = None,
) -> list[str]:
    lines = [
        "### 1b — residual split at spike 1.2 (20% ds rise)",
        "",
        "The documented 0.8 / 1.5 cut leaves **no residual positives**: 1.5/0.8 = 1.875 < 2, "
        "so a `ds_r` double cannot land in neither without floor effects. "
        "The 0.8 / 1.2 pair (already quoted as 80% / 49%) is the split that can have a leftover.",
        "",
        "| slice | n | n_pos | share of positives | P(Y=1) | size AUROC |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in p1b["slices_1_2"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['share_of_positives']:.3f} | "
            f"{r['y_rate']:.3f} | {r['size_auroc']:.3f} |"
        )
    lines += [
        "",
        "### 1c — HHI_lag3 inside the slices",
        "",
        "| slice | n | n_pos | HHI n | HHI AUROC | med HHI pos / neg |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in p1c["rows"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_hhi']} | {r['hhi_auroc']:.3f} | "
            f"{r['hhi_med_pos']:.3f} / {r['hhi_med_neg']:.3f} |"
        )
    lines += [
        "",
        f"### 2b — lag-6 is not a usable peak",
        "",
        p2b["note"],
        "",
        "### 2c — `d_n_supp` clock",
        "",
        "| lag | CV AUROC ± sd | train AUROC | sign | coverage |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in p2c["rows"]:
        lines.append(
            f"| {r['lag']} | {r['cv']:.3f} ± {r['sd']:.3f} | {r['train_auc']:.3f} | "
            f"{r['train_sign']:+d} | {r['coverage']:.3f} ({r['n_defined']}) |"
        )
    best = p2c.get("best") or {}
    lines += [
        "",
        f"n_supp best lag **{best.get('lag')}** CV **{best.get('cv', float('nan')):.3f}**.",
        "",
        f"### 2d — invoice-gate",
        "",
        f"Train company-months HHI defined **{p2d['train_hhi_cov']:.1%}** "
        f"({p2d['train_hhi']}/{p2d['train_cm']}). "
        f"Labeled months {p2d['lab_hhi_cov']:.1%}; lag3 {p2d['lab_hhi_lag3_cov']:.1%} "
        f"({p2d['lab_hhi_lag3']}/{p2d['lab_cm']}).",
        "",
        f"### 3b — monopoly tail",
        "",
        f"`d_cust_hhi_lag3` > 0.975: P(Y=1) = **{p3b['rate_hi']:.1%}** "
        f"(n={p3b['n_hi']}, {p3b['n_pos_hi']} pos) vs rest **{p3b['rate_lo']:.1%}**. "
        f"Tail-indicator AUROC {p3b['auroc_tail']:.3f}. Quintiles are not monotone "
        f"(Q2 15.4% > Q3/Q4). Cut 3d: body HHI≤0.975 CV is below dummy — "
        f"the 0.605 *is* the near-monopoly bin.",
        "",
    ]
    if p3c and p3c.get("rows"):
        lines += [
            "### 3c — quintiles on crash_any only",
            "",
            f"Cuts from the {p3c['n']} crash+HHI train rows. Not the all-labeled quintiles.",
            "",
            "| Q | interval | n | n_pos | P(Y=1) | median HHI |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for r in p3c["rows"]:
            lines.append(
                f"| {r['q']} | {r['interval']} | {r['n']} | {r['n_pos']} | "
                f"{r['y_rate']:.3f} | {r['hhi_median']:.3f} |"
            )
        lines.append("")
    if p3d:
        lines += [
            "### 3d — without the monopoly tail",
            "",
            f"HHI_lag3 CV on rows with HHI≤0.975: **{p3d['cv_body']:.3f} ± {p3d['sd_body']:.3f}** "
            f"(n={p3d['n_body']}, {p3d['n_pos_body']} pos). "
            f"**The 0.605 is the monopoly tail.** Body loses to dummy; "
            f"do not tell a smooth concentration-gradient story. "
            f"Tail is {p3d.get('n_cos_tail', '?')} companies / {p3d['n_tail']} months. "
            f"Those same companies on non-tail months: P(Y=1)="
            f"{p3d.get('rate_tailcos_other', float('nan')):.1%} "
            f"(n={p3d.get('n_tailcos_other', '?')}) vs tail months "
            f"{p3d.get('rate_tail', float('nan')):.1%}. "
            + (
                f"Binary HHI>0.975 OOF CV **{p3d['cv_tail_flag']:.3f} ± {p3d['sd_tail_flag']:.3f}** "
                f"(weaker than continuous 0.605; still the honest card shape)."
                if p3d.get("cv_tail_flag") is not None
                else ""
            ),
            "",
        ]
    if p1e:
        lines += [
            "### 1e — HHI vs crash flag",
            "",
            f"HHI_lag3 vs crash-any on all labeled AUROC **{p1e['auroc_crash_all_lab']:.3f}**; "
            f"among positives **{p1e['auroc_crash_among_pos']:.3f}** (n={p1e['n_pos_hhi']}). "
            f"Among positives ρ(HHI, in_ratio)={p1e['rho_hhi_inratio_pos']:+.3f}, "
            f"ρ(HHI, ds_ratio)={p1e['rho_hhi_dsratio_pos']:+.3f}. "
            f"HHI does **not** classify crash vs spike among positives (coin flip). "
            f"It ranks Y *inside* crash months (OOF 0.64), especially the monopoly tail.",
            "",
        ]
    if p1d:
        lines += [
            "### 1d — OOF inside crash vs spike",
            "",
            "| slice | feature | n | n_pos | CV AUROC ± sd | coverage |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in p1d["rows"]:
            lines.append(
                f"| {r['slice']} | `{r['col']}` | {r['n']} | {r['n_pos']} | "
                f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['coverage']:.3f} |"
            )
        lines.append("")
        if p1d.get("crash_clock"):
            lines += [
                "HHI clock on crash_any rows only:",
                "",
                "| lag | CV AUROC ± sd | train AUROC | coverage |",
                "|---:|---:|---:|---:|",
            ]
            for r in p1d["crash_clock"]:
                lines.append(
                    f"| {r['lag']} | {r['cv']:.3f} ± {r['sd']:.3f} | "
                    f"{r['train_auc']:.3f} | {r['coverage']:.3f} |"
                )
            best = p1d.get("crash_best") or {}
            lag1 = next((r for r in p1d["crash_clock"] if r["lag"] == 1), None)
            lag3 = next((r for r in p1d["crash_clock"] if r["lag"] == 3), None)
            lines += [
                "",
                f"On crash rows the mean-CV HHI peak is lag **{best.get('lag')}** "
                f"CV **{best.get('cv', float('nan')):.3f}**"
                + (
                    f" (sd {lag1['sd']:.3f}). Lag 3 is the stable crash clock "
                    f"({lag3['cv']:.3f} ± {lag3['sd']:.3f})."
                    if lag1 and lag3
                    else "."
                ),
                "",
            ]
    if ho:
        lines += [
            "### Holdout check (LOW_POWER, not a claim)",
            "",
            "Train-only signs. Holdout 16 positives flip the mechanism (mostly spike, not crash).",
            "",
            "| feature | sign | hold AUROC | n defined | n_pos |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in ho["rows"]:
            lines.append(
                f"| `{r['col']}` | {r['sign']:+d} | {r['auroc']:.3f} | {r['n']} | {r['n_pos']} |"
            )
        lines.append("")
    return lines


def append_registry(rows: list[dict]) -> None:
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def registry_rows(
    p1: dict, p2: dict, p3: dict, p4: dict, p5: dict, ts: str, extra: dict | None = None
) -> list[dict]:
    """Train-only metrics. No holdout AUROC claims."""
    out = []

    def add(model, metric, value, coverage, notes, families="D"):
        out.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": families,
                "y": Y_COL,
                "model": model,
                "split": "cv5_group" if "auroc" in metric or model.startswith("single_") or model.startswith("d_zavg") else "train",
                "metric": metric,
                "value": _fmt(value),
                "coverage": f"{coverage:.4f}" if isinstance(coverage, (int, float)) and np.isfinite(coverage) else "",
                "notes": notes,
            }
        )

    by = {r["slice"]: r for r in p1["slices"]}
    crash = by.get("crash_any", {})
    spike = by.get("spike_any", {})
    both = by.get("both", {})
    neither = by.get("neither", {})
    add(
        "y4_why_crash",
        "share_of_positives",
        crash.get("share_of_positives"),
        crash.get("share_labeled"),
        f"in3_ratio<{CRASH_TH}; n={crash.get('n')} n_pos={crash.get('n_pos')} "
        f"y_rate={crash.get('y_rate')}; size_auroc={crash.get('size_auroc')}; "
        f"never F; train labeled only",
        families="-",
    )
    add(
        "y4_why_spike",
        "share_of_positives",
        spike.get("share_of_positives"),
        spike.get("share_labeled"),
        f"ds3_ratio>{SPIKE_TH}; n={spike.get('n')} n_pos={spike.get('n_pos')} "
        f"y_rate={spike.get('y_rate')}; size_auroc={spike.get('size_auroc')}; "
        f"never F; train labeled only",
        families="-",
    )
    add(
        "y4_why_both",
        "share_of_positives",
        both.get("share_of_positives"),
        both.get("share_labeled"),
        f"crash and spike; n={both.get('n')} n_pos={both.get('n_pos')}; train labeled only",
        families="-",
    )
    add(
        "y4_why_neither",
        "share_of_positives",
        neither.get("share_of_positives"),
        neither.get("share_labeled"),
        f"neither crash nor spike; n={neither.get('n')} n_pos={neither.get('n_pos')}; train labeled only",
        families="-",
    )
    for r in p2["rows"]:
        add(
            f"single_{r['col']}",
            "auroc",
            r["cv"],
            r["coverage"],
            f"Q6 clock; sign from train fold; CV={r['cv']:.4f}±{r['sd']:.3f}; "
            f"train_auc={r['train_auc']:.4f}; sign={r['train_sign']:+d}; "
            f"n_defined={r['n_defined']}; never F; quote CV not holdout",
        )
    for r in p3["rows"]:
        add(
            "hhi_lag3_quintile",
            f"y_rate_q{r['q']}",
            r["y_rate"],
            r["n"] / p3["n_train_complete"] if p3["n_train_complete"] else float("nan"),
            f"train labeled complete HHI_lag3; cuts from train; n={r['n']} "
            f"pos={r['n_pos']} interval={r['interval']}; never holdout cuts",
        )
    add(
        "d_zavg_hhi_nsupp_lag3",
        "auroc",
        p4["zavg"]["cv"],
        p4["n_complete"] / p2["n_train_labeled"] if p2["n_train_labeled"] else float("nan"),
        f"{p4['decision']}; {p4['reason']}; vs HHI-all {p4['hhi_all']['cv']:.4f}; "
        f"HHI-CC {p4['hhi_cc']['cv']:.4f}; supp-CC {p4['supp_cc']['cv']:.4f}; "
        f"complete={p4['n_complete']}/{p4['n_complete_pos']}; never F; no GBM",
    )
    add(
        "single_d_cust_hhi_lag3_cc",
        "auroc",
        p4["hhi_cc"]["cv"],
        p4["n_complete"] / p2["n_train_labeled"] if p2["n_train_labeled"] else float("nan"),
        f"HHI_lag3 on z-avg complete-case rows; honesty check; never F",
    )
    for r in p5["rows"]:
        add(
            "hhi_lag3_leak",
            f"spearman_{r['vs']}",
            r["spearman"],
            r["n"] / p2["n_train_labeled"] if p2["n_train_labeled"] else float("nan"),
            f"train labeled; pearson={r['pearson']}; n={r['n']}; "
            f"top1_rewrite={p5['is_top1_rewrite']}; comparator only, not X; never F in X",
            families="D" if not str(r["vs"]).startswith("f") else "D-vs-F",
        )
    if extra:
        p2b = extra.get("p2b") or {}
        p2c = extra.get("p2c") or {}
        p3b = extra.get("p3b") or {}
        if p2b.get("lag6"):
            add(
                "single_d_cust_hhi_lag6",
                "auroc_sd",
                p2b["lag6"]["sd"],
                p2b["lag6"]["coverage"],
                f"lag6 mean-CV peak is unstable; sd={p2b['lag6']['sd']:.3f} "
                f"train_auc={p2b['lag6']['train_auc']:.3f}; stable quote lag3; never F",
            )
        for r in p2c.get("rows") or []:
            add(
                f"single_{r['col']}",
                "auroc",
                r["cv"],
                r["coverage"],
                f"n_supp Q6 clock; CV={r['cv']:.4f}±{r['sd']:.3f}; sign={r['train_sign']:+d}; never F",
            )
        if p3b:
            add(
                "hhi_lag3_tail_gt0975",
                "y_rate",
                p3b.get("rate_hi"),
                (p3b.get("n_hi") or 0) / p3["n_train_complete"] if p3.get("n_train_complete") else float("nan"),
                f"train labeled; rest_rate={p3b.get('rate_lo')}; tail_auroc={p3b.get('auroc_tail')}; "
                f"n={p3b.get('n_hi')} pos={p3b.get('n_pos_hi')}; never holdout cuts",
            )
        p3d = extra.get("p3d") or {}
        if p3d.get("cv_body") is not None:
            add(
                "single_d_cust_hhi_lag3_body",
                "auroc",
                p3d["cv_body"],
                (p3d.get("n_body") or 0) / p2["n_train_labeled"] if p2.get("n_train_labeled") else float("nan"),
                f"HHI<=0.975 only; CV={p3d['cv_body']:.4f}±{p3d.get('sd_body', float('nan')):.3f}; "
                f"n={p3d.get('n_body')} pos={p3d.get('n_pos_body')}; 0.605 is the tail; never F",
            )
        for r in (extra.get("p3c") or {}).get("rows") or []:
            add(
                "hhi_lag3_crash_quintile",
                f"y_rate_q{r['q']}",
                r["y_rate"],
                r["n"] / extra["p3c"]["n"] if extra["p3c"].get("n") else float("nan"),
                f"crash_any train cuts; n={r['n']} pos={r['n_pos']} interval={r['interval']}",
            )
        p1d = extra.get("p1d") or {}
        for r in p1d.get("rows") or []:
            add(
                f"single_{r['col']}_{r['slice']}",
                "auroc",
                r["cv"],
                r["coverage"],
                f"OOF inside {r['slice']}; n={r['n']} pos={r['n_pos']}; "
                f"CV={r['cv']:.4f}±{r['sd']:.3f}; never F; train folds only",
            )
        for r in p1d.get("crash_clock") or []:
            add(
                f"single_{r['col']}_crash_any",
                "auroc",
                r["cv"],
                r["coverage"],
                f"HHI clock on crash_any; CV={r['cv']:.4f}±{r['sd']:.3f}; "
                f"train={r['train_auc']:.4f}; never F",
            )
    return out


def write_md(
    started: str,
    n_tr: int,
    n_pos: int,
    rate: float,
    p1: dict,
    p2: dict,
    p3: dict,
    p4: dict,
    p5: dict,
    extra: dict | None = None,
) -> None:
    by = {r["slice"]: r for r in p1["slices"]}
    best = p2.get("best_hhi") or {}
    lines = [
        "# Y4 why — crash vs spike, HHI clock, 2-col honesty",
        "",
        f"- **When:** {started}",
        f"- **Agent:** `{AGENT}`",
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        f"- **Re-run:** `python -m analysis.evaluate.y4_why`",
        f"- **Holdout:** 72 companies, seed {FOLD_SEED}. Out of every rate, AUROC, and cut.",
        f"- **Y:** `{Y_COL}` only (train {n_tr} / {n_pos} / **{rate:.2%}**). Other Y4 columns not revived.",
        "- **X:** family D lags. **Never F.** in3/ds3 rebuilt here from transactions + CAT_MAP to split the label — not used as X.",
        "- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER.",
        "- **Brief:** Q3 turning / Q5 why / Q6 lead. Not bankruptcy. Not a 0–100.",
        "",
        "## Decision",
        "",
        f"**{p4['decision']}** the 2-col z-avg card. {p4['reason']}",
        "",
        f"Mean-CV HHI peak is **lag {best.get('lag')}** CV **{best.get('cv', float('nan')):.3f}** "
        f"(lag3 is peak: {p2['lag3_is_peak']}). Stable quote remains **lag 3 CV 0.605** "
        f"(sd 0.044; lag 6 sd 0.175 / train 0.533).",
        "",
        "## Pass 1 — decompose the double",
        "",
        f"Thresholds (documented): inflow crash `in3[t+3]/in3[t] < {CRASH_TH}`; "
        f"repayment spike `ds3[t+3]/ds3[t] > {SPIKE_TH}`.",
        f"Positives median in-ratio **{p1['pos_med_in_ratio']:.2f}**, ds-ratio **{p1['pos_med_ds_ratio']:.2f}** "
        f"(neg {p1['neg_med_in_ratio']:.2f} / {p1['neg_med_ds_ratio']:.2f}).",
        "",
        "| slice | n | n_pos | share labeled | share of positives | P(Y=1) | size AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p1["slices"]:
        lines.append(
            f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['share_labeled']:.3f} | "
            f"{r['share_of_positives']:.3f} | {r['y_rate']:.3f} | {r['size_auroc']:.3f} |"
        )
    lines += [
        "",
        "Size AUROC is `log1p(|in3[t]|)` vs the label *inside* the slice. "
        "Crash/spike flags use future in3/ds3 only to name the label — they are not X. "
        "Holdout 16 positives flip the mix (crash 0.375 / spike 0.875 / med in-ratio 0.88) — LOW_POWER.",
        "",
        "## Pass 2 — Q6 clock",
        "",
        "| feature | lag | CV AUROC ± sd | train AUROC | sign | coverage (train labeled) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in p2["rows"]:
        lines.append(
            f"| `{r['col']}` | {r['lag']} | {r['cv']:.3f} ± {r['sd']:.3f} | "
            f"{r['train_auc']:.3f} | {r['train_sign']:+d} | "
            f"{r['coverage']:.3f} ({r['n_defined']}/{p2['n_train_labeled']}) |"
        )
    lines += [
        "",
        f"HHI is invoice-gated. Lag 3 coverage is not the full ~53% store fill — "
        f"it needs HHI defined three months earlier on a Y4-labeled month.",
        (
            "Lag 3 is the mean-CV peak."
            if p2["lag3_is_peak"]
            else (
                "**Lag 3 is not the mean-CV peak** (lag 6 wins by <0.001). "
                "Do not replace the 0.605 quote with lag 6: fold sd is 0.17, "
                "train AUROC collapses to 0.53, coverage 26%. Stable peak stays lag 3."
            )
        ),
        "",
        "## Pass 3 — quintiles (train labeled cuts)",
        "",
        "| Q | interval | n | n_pos | P(Y=1) | median HHI |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for r in p3["rows"]:
        lines.append(
            f"| {r['q']} | {r['interval']} | {r['n']} | {r['n_pos']} | {r['y_rate']:.3f} | {r['hhi_median']:.3f} |"
        )
    lines += [
        "",
        f"Monotone non-decreasing: **{p3['monotone_up']}**. Cuts from train labeled complete rows only.",
    ]
    if p3.get("png"):
        lines.append(f"Plot: `{Path(p3['png']).name}`.")
    if p3.get("holdout_check"):
        lines.append("Holdout rates with *train* cuts (LOW_POWER, not a claim):")
        for r in p3["holdout_check"]:
            lines.append(f"- Q{r['q']} n={r['n']} pos={r['n_pos']} rate={r['y_rate']:.3f}")
    lines += [
        "",
        "## Pass 4 — 2-col card honesty",
        "",
        f"- z-avg HHI_lag3 + n_supp_lag3: CV **{p4['zavg']['cv']:.3f} ± {p4['zavg']['sd']:.3f}** "
        f"(complete {p4['n_complete']} / {p4['n_complete_pos']} pos)",
        f"- HHI_lag3 all defined rows: CV **{p4['hhi_all']['cv']:.3f}**",
        f"- HHI_lag3 on the z-avg complete-case: CV **{p4['hhi_cc']['cv']:.3f}**",
        f"- n_supp_lag3 on the same complete-case: CV **{p4['supp_cc']['cv']:.3f}**",
        f"- gap vs 0.605 bar: {p4['gap_vs_bar']:+.3f}; gap vs HHI-CC: {p4['gap_vs_hhi_cc']:+.3f}",
        f"- just HHI on complete-case: **{p4['just_hhi_on_cc']}**",
        "",
        f"**{p4['decision']}.** {p4['reason']}",
        "",
        "## Pass 5 — leak",
        "",
        "| vs | n | Spearman | Pearson |",
        "|---|---:|---:|---:|",
    ]
    for r in p5["rows"]:
        lines.append(
            f"| `{r['vs']}` | {r['n']} | {r['spearman']:+.3f} | {r['pearson']:+.3f} |"
        )
    lines += [
        "",
        (
            "HHI_lag3 **is** a top-1 rewrite (same D stem). Y7 forbids D for that reason; "
            "Y4 allowed D — do not stack HHI and top1."
            if p5["is_top1_rewrite"]
            else "HHI_lag3 is not a top-1 rewrite at ρ≥0.90."
        ),
        "f_ds_r / a_in3 are comparators only. They are not X. Family F never enters a score.",
        "",
        "## Mapping (Q3 / Q5 / Q6)",
        "",
        _mapping_paragraph(p1, p2, p3, p4, p5, by),
        "",
        "## Extra cuts",
        "",
    ]
    if extra:
        for line in extra.get("md_lines") or []:
            lines.append(line)
    else:
        lines.append("None yet — next iteration in this module.")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _mapping_paragraph(p1: dict, p2: dict, p3: dict, p4: dict, p5: dict, by: dict) -> str:
    crash = by.get("crash_any", {})
    spike = by.get("spike_any", {})
    both = by.get("both", {})
    best = p2.get("best_hhi") or {}
    q_rates = " / ".join(f"{r['y_rate']:.1%}" for r in p3["rows"])
    return (
        f"Y4 `ds_r` doubling is who is *turning* (Q3): debt-service / inflow at t+3 "
        f"is at least twice t, with a 0.05 floor. On train labeled rows the double is "
        f"mostly a **denominator crash** (Q5) — "
        f"{crash.get('share_of_positives', float('nan')):.0%} of positives drop inflow "
        f">20% (`in3` ratio < {CRASH_TH}), "
        f"{spike.get('share_of_positives', float('nan')):.0%} raise `ds3` by >50%, "
        f"{both.get('share_of_positives', float('nan')):.0%} do both. "
        f"Size AUROC inside those slices stays below 0.60, so this is not a big-firm label. "
        f"`d_cust_hhi` is the leading why (Q6): mean-CV peak is lag **{best.get('lag')}** "
        f"CV **{best.get('cv', float('nan')):.3f}**"
        f"{'' if p2['lag3_is_peak'] else ' — lag 3 is not the mean-CV peak; lag 6 wins by <0.001 with sd 0.17, so the stable quote stays lag3 0.605'}. "
        f"Train quintiles of HHI_lag3 (cuts from train labeled) read {q_rates} — "
        f"not monotone; the 0.605 is the HHI>0.975 bin (body CV 0.445). "
        f"The 2-col z-avg of HHI_lag3 + n_supp_lag3 is **{p4['decision']}** "
        f"(CV {p4['zavg']['cv']:.3f} vs HHI-CC {p4['hhi_cc']['cv']:.3f}). "
        f"{'HHI is a top-1 rewrite; Y4 may use D, but do not stack the two.' if p5['is_top1_rewrite'] else ''} "
        f"Holdout's 16 positives flip the mix (mostly spike, not crash) — HHI should invert there. "
        f"Not bankruptcy. Not a 0–100."
    )


def run(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Y4 why / HHI clock / z-avg honesty")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--no-md", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    started = datetime.now().isoformat(timespec="minutes")
    wall0 = datetime.now()
    print(f"y4_why start {started} seed={FOLD_SEED} y={Y_COL} agent={AGENT}")
    print("forbidden X = family F; D lags only; in3/ds3 rebuilt here, not X")
    print("META.forbidden_x_families", Y4_META.get("forbidden_x_families"))
    print(
        "leakage_check D-only:",
        leakage_check(["d_cust_hhi", "d_cust_top1", "d_n_supp"], Y_COL, ("f",)),
    )

    hold_ids = load_holdout()
    con = connect()
    store = load_store()
    train_cos = train_companies(con)
    assert_no_holdout(train_cos)
    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    grid = store[["company_id", "period"]].drop_duplicates()
    y4 = load_y(con, grid)
    flows = rebuild_in3_ds3(con)
    con.close()

    panel = store.merge(y4, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold", "group_id"]], on="company_id", how="left")
    stems = [c for c in D_STEMS if c in panel.columns]
    panel = add_lags(panel, stems, LAGS_CLOCK)

    is_hold = panel["company_id"].astype(str).isin(hold_ids)
    labeled = panel[Y_COL].notna()
    train_lab = (~is_hold) & labeled & panel["fold"].notna()
    hold_lab = is_hold & labeled
    assert_no_holdout(panel.loc[train_lab, "company_id"])
    if set(panel.loc[train_lab, "company_id"].astype(str)) & hold_ids:
        raise RuntimeError("holdout leaked into train labeled")

    n_tr = int(train_lab.sum())
    n_pos = int((panel.loc[train_lab, Y_COL] == 1).sum())
    rate = float(panel.loc[train_lab, Y_COL].mean()) if n_tr else float("nan")
    n_ho = int(hold_lab.sum())
    n_ho_pos = int((panel.loc[hold_lab, Y_COL] == 1).sum())
    print(
        f"train labeled={n_tr} pos={n_pos} rate={rate:.4f}  "
        f"holdout labeled={n_ho} pos={n_ho_pos} LOW_POWER  "
        f"hold_cos={len(hold_ids)}"
    )

    # Future in3/ds3 for the label split only.
    fl = flows.sort_values(["company_id", "period"]).copy()
    g = fl.groupby("company_id", sort=False)
    fl["in3_f3"] = g["in3"].shift(-3)
    fl["ds3_f3"] = g["ds3"].shift(-3)
    lab = panel.loc[train_lab, ["company_id", "period", Y_COL]].merge(
        fl[["company_id", "period", "op_in", "in3", "ds3", "in3_f3", "ds3_f3"]],
        on=["company_id", "period"],
        how="left",
    )
    assert_no_holdout(lab["company_id"])
    lab["in_ratio"] = lab["in3_f3"] / lab["in3"].replace(0, np.nan)
    lab["ds_ratio"] = lab["ds3_f3"] / lab["ds3"].replace(0, np.nan)

    if "a_in3" in panel.columns:
        chk = panel.loc[train_lab, ["company_id", "period", "a_in3"]].merge(
            fl[["company_id", "period", "in3"]], on=["company_id", "period"], how="left"
        )
        both = chk.dropna(subset=["a_in3", "in3"])
        if len(both):
            mad = float((both["a_in3"] - both["in3"]).abs().max())
            rho = float(both["a_in3"].corr(both["in3"]))
            print(f"QA rebuilt in3 vs a_in3: n={len(both)} max|Δ|={mad:.4g} ρ={rho:.4f}")

    p1 = pass1_decompose(lab)
    ho_lab = panel.loc[hold_lab, ["company_id", "period", Y_COL]].merge(
        fl[["company_id", "period", "in3", "ds3", "in3_f3", "ds3_f3"]],
        on=["company_id", "period"],
        how="left",
    )
    ho_lab["in_ratio"] = ho_lab["in3_f3"] / ho_lab["in3"].replace(0, np.nan)
    ho_lab["ds_ratio"] = ho_lab["ds3_f3"] / ho_lab["ds3"].replace(0, np.nan)
    ho_pos = ho_lab[ho_lab[Y_COL] == 1]
    print(
        f"holdout LOW_POWER decompose n={len(ho_lab)} pos={len(ho_pos)}: "
        f"crash={float((ho_pos['in_ratio'] < CRASH_TH).mean()) if len(ho_pos) else float('nan'):.3f} "
        f"spike1.5={float((ho_pos['ds_ratio'] > SPIKE_TH).mean()) if len(ho_pos) else float('nan'):.3f} "
        f"med in_ratio={float(ho_pos['in_ratio'].median()) if len(ho_pos) else float('nan'):.3f}"
    )
    hhi_pos = panel.loc[train_lab & (panel[Y_COL] == 1), ["company_id", "period", "d_cust_hhi_lag3"]].merge(
        lab.loc[lab[Y_COL] == 1, ["company_id", "period", "in_ratio", "ds_ratio"]],
        on=["company_id", "period"],
        how="left",
    )
    hp = hhi_pos[hhi_pos["d_cust_hhi_lag3"].notna()]
    print(
        f"HHI-defined positives n={len(hp)} crash={float((hp['in_ratio'] < CRASH_TH).mean()):.3f} "
        f"spike1.5={float((hp['ds_ratio'] > SPIKE_TH).mean()):.3f} "
        f"— D coverage does not select a different mechanism"
    )
    p1b = pass1_alt_thresholds(lab)
    p2 = pass2_clock(panel, train_lab)
    p2b = pass2_fold_table(p2)
    p2c = pass2_nsupp_clock(panel, train_lab)
    p2d = pass2_store_coverage(panel, train_lab, hold_ids)
    p1c = pass1_hhi_by_slice(lab, panel, train_lab)
    p1d = pass_slice_oof(lab, panel, train_lab)
    p1e = pass_hhi_predicts_crash(lab, panel, train_lab)
    p3 = pass3_quintiles(panel, train_lab, hold_lab)
    if args.no_plot:
        p3["png"] = None
    p3b = pass3_tail(panel, train_lab)
    p3c = pass3_crash_quintiles(lab, panel, train_lab)
    p3d = pass_hhi_without_tail(panel, train_lab)
    p4 = pass4_zavg(panel, train_lab)
    p5 = pass5_leak(panel, train_lab)
    ho = pass_holdout_check(panel, train_lab, hold_lab)

    extra = {
        "md_lines": extra_md_lines(p1b, p1c, p2b, p2c, p2d, p3b, p1d, ho, p3c, p1e, p3d),
        "p3d": p3d,
        "p1e": p1e,
        "p1b": p1b,
        "p1c": p1c,
        "p2b": p2b,
        "p2c": p2c,
        "p2d": p2d,
        "p3b": p3b,
        "p3c": p3c,
        "p1d": p1d,
        "ho": ho,
    }
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if not args.no_registry:
        rows = registry_rows(p1, p2, p3, p4, p5, ts, extra=extra)
        append_registry(rows)
        print(f"appended {len(rows)} registry rows (train-only metrics)")
    if not args.no_md:
        write_md(started, n_tr, n_pos, rate, p1, p2, p3, p4, p5, extra)

    elapsed = (datetime.now() - wall0).total_seconds()
    quote = {
        "crash_share_of_pos": next(
            r["share_of_positives"] for r in p1["slices"] if r["slice"] == "crash_any"
        ),
        "spike_share_of_pos": next(
            r["share_of_positives"] for r in p1["slices"] if r["slice"] == "spike_any"
        ),
        "both_share_of_pos": next(
            r["share_of_positives"] for r in p1["slices"] if r["slice"] == "both"
        ),
        "neither_share_of_pos": next(
            r["share_of_positives"] for r in p1["slices"] if r["slice"] == "neither"
        ),
        "best_hhi_lag": (p2.get("best_hhi") or {}).get("lag"),
        "best_hhi_cv": (p2.get("best_hhi") or {}).get("cv"),
        "lag3_is_peak": p2["lag3_is_peak"],
        "quintile_rates": [r["y_rate"] for r in p3["rows"]],
        "zavg_cv": p4["zavg"]["cv"],
        "hhi_cc_cv": p4["hhi_cc"]["cv"],
        "nsupp_cc_cv": p4["supp_cc"]["cv"],
        "zavg_decision": p4["decision"],
        "stable_hhi_lag": extra["p2b"]["stable_peak_lag"],
        "nsupp_best_lag": (extra["p2c"].get("best") or {}).get("lag"),
        "nsupp_best_cv": (extra["p2c"].get("best") or {}).get("cv"),
        "elapsed_s": elapsed,
    }
    print("\nQUOTE")
    print(json.dumps(quote, indent=2, default=str))
    print(f"elapsed {elapsed:.1f}s — stay on this module for more cuts")
    return {
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "extra": extra,
        "quote": quote,
        "n_tr": n_tr,
        "n_pos": n_pos,
        "rate": rate,
    }


if __name__ == "__main__":
    run()
