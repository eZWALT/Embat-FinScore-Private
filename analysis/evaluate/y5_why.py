"""Y5 why — cash-stress vs supplier-tail vs leftover (Q5).

Brief: payment behaviour. Accepted labels only:
``y5_ap_od30_ownp80`` (8.5%) and ``y5_ar_od30_sust`` (7.2%).
``y5_ap_delay_up15`` stays rejected. Labels from ``targets.parquet``.
Do not run ``build_targets``. X never family E (that built the Y).

XGB Y5 is PARK (``xgb_panel.py``): does not beat the best single;
holdout inverted. Quote the single, not the tree.

Night honest singles (min_cov=0.25, thin F discarded):
AP ``h_group_size`` train 0.5994; AR ``d_tx_cp_share`` train 0.6114.
Pooled AR tagging-hole 17.4% is fold-3 clustered — quote the single,
not a leave-one-group law.

No 0–100. No product/. No parquet rewrite. No new GBM/XGB.
Single-feature / stratified rates only. Holdout 72 is coverage only.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.y5_why
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
from analysis.features.common import ANALYSIS, DATA, connect
from analysis.targets.y5_payment import META as Y5_META

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
OUT_MD = ANALYSIS / "outputs" / "y5_why.md"
OUT_PNG = ANALYSIS / "outputs" / "y5_why_quintiles.png"
AGENT = "234af73a"
WAVE = 4
ROUND = "R4"

Y_AP = "y5_ap_od30_ownp80"
Y_AR = "y5_ar_od30_sust"
Y_DELAY = "y5_ap_delay_up15"  # rejected — coverage only, never KEEP
Y2 = "y2_neg_2of3"
Y4 = "y4_ds_r_double"
Y9 = "y9_fee_r_ownp80"
N_FOLDS = 5
CLEAR_MARGIN = 0.02
LEAK_RHO = 0.80
SIZE_AUROC = 0.60
CHANCE = 0.50
MIN_OWN_HIST = 6
OWN_P_HI = 0.80
OWN_P_LO = 0.20

# Night quotes from xgb_panel (min_cov=0.25). Thin-F first-run winners are PARK.
NIGHT_AP = {"col": "h_group_size", "train_auc": 0.5994, "note": "min_cov=0.25"}
NIGHT_AR = {"col": "d_tx_cp_share", "train_auc": 0.6114, "note": "min_cov=0.25"}
NIGHT_AP_THIN = {"col": "f_w_rate_lag1", "train_auc": 0.6675, "note": "PARK thin F ~1.7%"}
NIGHT_AR_THIN = {"col": "f_months_to_next_pay_lag3", "train_auc": 0.6766, "note": "PARK thin F"}

STORE_X = (
    "a_out6",
    "a_in3",
    "a_io_ratio",
    "c_n_days_with_tx",
    "c_gap_sd",
    "d_supp_hhi",
    "d_cust_hhi",
    "d_tx_cp_share",
    "d_n_supp",
    "d_n_cust",
    "d_supp_top1",
    "h_group_size",
)
# Comparators only — never X, never in SINGLE_COLS.
STORE_E = (
    "e_ap_overdue_30",
    "e_ar_overdue_30",
    "e_delay_paid",
    "e_delay_coll",
    "e_ap_overdue",
    "e_ar_overdue",
)
SINGLE_COLS = (
    "d_supp_hhi",
    "d_cust_hhi",
    "a_out6",
    "log1p_a_in3",
    "c_n_days_with_tx",
    "a_io_ratio",
    "c_gap_sd",
    "d_tx_cp_share",
    "h_group_size",
)
QINT_COLS = (
    "d_supp_hhi",
    "d_cust_hhi",
    "a_out6",
    "log1p_a_in3",
    "c_n_days_with_tx",
    "a_io_ratio",
    "h_group_size",
    "d_tx_cp_share",
)
HI_COLS = ("d_supp_hhi", "d_cust_hhi", "c_gap_sd", "a_out6")
LO_COLS = ("a_io_ratio",)
LAG_STEMS = ("d_supp_hhi", "a_io_ratio", "d_cust_hhi", "d_tx_cp_share", "h_group_size")
LEAK_VS = (
    "e_ap_overdue_30",
    "e_ar_overdue_30",
    "e_delay_paid",
    "e_delay_coll",
)
FORBIDDEN = Y5_META.get("forbidden_x_families") or ("e",)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    return out


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return f"{float(v):.6g}" if np.isfinite(v) else ""
    return "" if v is None else str(v)


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
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in cols:
        s = pd.to_numeric(out[c], errors="coerce")
        g = s.groupby(cid, sort=False)
        for k in lags:
            extra[f"{c}_lag{k}"] = g.shift(k)
    if extra:
        out = pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)
    return out


def add_own_cuts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-company expanding p80 / p20 (months ≤ t, min 6 finite). Never pooled."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    for c in HI_COLS:
        x = pd.to_numeric(out[c], errors="coerce")
        p80 = x.groupby(out["company_id"], sort=False).transform(
            lambda s: _expanding_quantile_skipna(s, OWN_P_HI, MIN_OWN_HIST)
        )
        extra[f"{c}_ownp80"] = p80
        extra[f"{c}_hi"] = pd.Series(
            np.where(x.notna() & p80.notna(), (x > p80).astype(float), np.nan),
            index=out.index,
        )
    for c in LO_COLS:
        x = pd.to_numeric(out[c], errors="coerce")
        p20 = x.groupby(out["company_id"], sort=False).transform(
            lambda s: _expanding_quantile_skipna(s, OWN_P_LO, MIN_OWN_HIST)
        )
        extra[f"{c}_ownp20"] = p20
        extra[f"{c}_lo"] = pd.Series(
            np.where(x.notna() & p20.notna(), (x < p20).astype(float), np.nan),
            index=out.index,
        )
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def coverage(x: pd.Series, mask: pd.Series) -> dict:
    n = int(mask.sum())
    n_ok = int(x[mask].notna().sum())
    return {"n": n, "n_defined": n_ok, "coverage": (n_ok / n) if n else float("nan")}


def signed_oof_auroc(
    df: pd.DataFrame,
    col: str,
    y_col: str,
    train_lab: pd.Series,
    folds: np.ndarray,
) -> dict:
    """Group-fold CV AUROC. Sign from the train fold only."""
    if str(col).startswith("e_"):
        raise RuntimeError(f"family E leaked into a single: {col}")
    y = pd.to_numeric(df[y_col], errors="coerce")
    x = pd.to_numeric(df[col], errors="coerce")
    fold_rows = []
    aucs = []
    for k in range(N_FOLDS):
        tr = train_lab & (folds != k)
        va = train_lab & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        n_va = int((va & x.notna() & y.notna()).sum())
        n_pos = int((va & x.notna() & (y == 1)).sum())
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
    tr_sign = int(choose_sign(y[train_lab], x[train_lab]))
    return {
        "col": col,
        "y": y_col,
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": tr_sign,
        "train_auc": float(auroc(y[train_lab], tr_sign * x[train_lab])),
    }


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    raw = pd.read_parquet(STORE)
    need = ["company_id", "period", *STORE_X, *STORE_E]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"store missing {missing}")
    panel = _keys(raw[need])
    print(f"loaded store {STORE} shape={panel.shape} (E is comparator only)")
    return panel


def load_y() -> pd.DataFrame:
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(TARGETS)
    need = ["company_id", "period", Y_AP, Y_AR, Y_DELAY, Y2, Y4, Y9]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"targets.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"Y from {TARGETS} shape={panel.shape} (no build_targets)")
    return panel


def pos_overlap(lab: pd.DataFrame, y_col: str, flag: str) -> dict:
    pos = lab[y_col] == 1
    n_pos = int(pos.sum())
    f = pd.to_numeric(lab[flag], errors="coerce")
    defined = pos & f.notna()
    n_def = int(defined.sum())
    n_hi = int((defined & (f == 1)).sum())
    return {
        "flag": flag,
        "y": y_col,
        "n_pos": n_pos,
        "n_defined": n_def,
        "n_hi": n_hi,
        "share_of_pos_defined": (n_hi / n_def) if n_def else float("nan"),
        "share_of_pos_all": (n_hi / n_pos) if n_pos else float("nan"),
        "coverage_of_pos": (n_def / n_pos) if n_pos else float("nan"),
    }


def pass1_decompose(lab: pd.DataFrame) -> dict:
    """Share of Y5 positives that are also Y2 / Y4 / Y9 / cash / HHI / gap."""
    print("\n" + "=" * 72)
    print("PASS 1 — decompose Y5 positives (train labeled)")
    print("high = company own expanding p80; low a_io = own p20. Never pooled.")
    print("Never E as a flag.")
    print("=" * 72)

    flags_ap = (Y2, Y4, Y9, "a_io_ratio_lo", "d_supp_hhi_hi", "c_gap_sd_hi")
    flags_ar = (Y2, Y4, Y9, "a_io_ratio_lo", "d_cust_hhi_hi", "c_gap_sd_hi")
    rows = []
    twos = {}
    for y_col, flags in ((Y_AP, flags_ap), (Y_AR, flags_ar)):
        n = int(lab[y_col].notna().sum())
        n_pos = int((lab[y_col] == 1).sum())
        rate = float(lab.loc[lab[y_col].notna(), y_col].mean()) if n else float("nan")
        print(f"\n{y_col}: n={n} pos={n_pos} rate={rate:.4f}")
        for flag in flags:
            if flag not in lab.columns:
                print(f"  missing {flag}")
                continue
            rec = pos_overlap(lab, y_col, flag)
            rows.append(rec)
            print(
                f"  {flag:22s}  {rec['n_hi']:4d}/{rec['n_defined']:4d} defined pos  "
                f"share={rec['share_of_pos_defined']:.3f}  "
                f"cov={rec['coverage_of_pos']:.3f}"
            )

        pos = lab[y_col] == 1
        hhi_col = "d_supp_hhi_hi" if y_col == Y_AP else "d_cust_hhi_hi"
        both = pos & lab["a_io_ratio_lo"].notna() & lab[hhi_col].notna()
        lo = lab.loc[both, "a_io_ratio_lo"] == 1
        hh = lab.loc[both, hhi_col] == 1
        twos[y_col] = {
            "n": int(both.sum()),
            "cash_only": int((lo & ~hh).sum()),
            "hhi_only": int((~lo & hh).sum()),
            "both": int((lo & hh).sum()),
            "neither": int((~lo & ~hh).sum()),
        }
        t = twos[y_col]
        print(
            f"  2x2 cash×HHI among pos (n={t['n']}): "
            f"cash_only={t['cash_only']} hhi_only={t['hhi_only']} "
            f"both={t['both']} neither={t['neither']}"
        )

    stories = {}
    for y_col in (Y_AP, Y_AR):
        hhi_flag = "d_supp_hhi_hi" if y_col == Y_AP else "d_cust_hhi_hi"
        cash = next(
            r["share_of_pos_defined"]
            for r in rows
            if r["y"] == y_col and r["flag"] == "a_io_ratio_lo"
        )
        hhi = next(
            r["share_of_pos_defined"]
            for r in rows
            if r["y"] == y_col and r["flag"] == hhi_flag
        )
        neither = twos[y_col]["neither"] / twos[y_col]["n"] if twos[y_col]["n"] else float("nan")
        if np.isfinite(cash) and cash >= 0.50 and (not np.isfinite(hhi) or cash >= hhi):
            story = "cash_stress"
        elif np.isfinite(hhi) and hhi >= 0.50 and (not np.isfinite(cash) or hhi > cash + 0.10):
            story = "supplier_tail" if y_col == Y_AP else "customer_tail"
        else:
            story = "unexplained_leftover"
        stories[y_col] = {
            "story": story,
            "cash_share": cash,
            "hhi_share": hhi,
            "neither_share": neither,
        }
        print(
            f"  STORY {y_col}: {story}  cash={cash:.3f} hhi={hhi:.3f} "
            f"neither={neither:.3f}"
        )
    return {"rows": rows, "twos": twos, "stories": stories}


def quintile_table(df: pd.DataFrame, mask: pd.Series, x_col: str, y_col: str) -> dict:
    x = pd.to_numeric(df[x_col], errors="coerce")
    y = pd.to_numeric(df[y_col], errors="coerce")
    m = mask & x.notna() & y.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    tr = pd.DataFrame({x_col: x[m], y_col: y[m]})
    if len(tr) < 50 or tr[x_col].nunique() < 3:
        return {
            "x": x_col,
            "y": y_col,
            "n": int(len(tr)),
            "rows": [],
            "n_bins": 0,
            "monotone_up": None,
            "monotone_down": None,
            "tail_only": None,
            "head_only": None,
            "bins": [],
        }
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
    monotone_up = all(a <= b + 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    monotone_down = all(a >= b - 1e-12 for a, b in zip(rates, rates[1:])) if len(rates) > 1 else None
    if len(rates) >= 3:
        body = rates[:-1]
        tail_only = rates[-1] > max(body) + 0.04 and max(body) - min(body) < 0.06
        mid = rates[1:]
        head_only = rates[0] > max(mid) + 0.04 and max(mid) - min(mid) < 0.06
    else:
        tail_only = None
        head_only = None
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
        "bins": [float(b) for b in bins],
    }


def pass2_quintiles(df: pd.DataFrame, train_by_y: dict[str, pd.Series], no_plot: bool) -> dict:
    print("\n" + "=" * 72)
    print("PASS 2 — train-only quintiles (cuts never see holdout)")
    print("=" * 72)
    tables = []
    for x_col in QINT_COLS:
        for y_col in (Y_AP, Y_AR):
            tab = quintile_table(df, train_by_y[y_col], x_col, y_col)
            tables.append(tab)
            print(
                f"  {x_col} vs {y_col}: n={tab['n']} bins={tab['n_bins']} "
                f"up={tab['monotone_up']} down={tab['monotone_down']} "
                f"tail={tab['tail_only']} head={tab['head_only']}"
            )
            for r in tab["rows"]:
                print(
                    f"    Q{r['q']} {r['interval']} n={r['n']:4d} pos={r['n_pos']:3d} "
                    f"P(Y=1)={r['y_rate']:.3f} med={r['x_median']:.4g}"
                )

    png = None
    ap = next(t for t in tables if t["x"] == "h_group_size" and t["y"] == Y_AP)
    ar = next(t for t in tables if t["x"] == "d_tx_cp_share" and t["y"] == Y_AR)
    if HAS_MPL and not no_plot and (ap["rows"] or ar["rows"]):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharey=False)
        for ax, tab, title, xlab in (
            (axes[0], ap, "y5_ap_od30_ownp80", "h_group_size quintile (train cuts)"),
            (axes[1], ar, "y5_ar_od30_sust", "d_tx_cp_share quintile (train cuts)"),
        ):
            if not tab["rows"]:
                ax.set_visible(False)
                continue
            xs = [r["q"] for r in tab["rows"]]
            ys = [r["y_rate"] for r in tab["rows"]]
            ax.bar(xs, ys, color="#3d5a80", width=0.72)
            base = float(df.loc[train_by_y[tab["y"]], tab["y"]].mean())
            ax.axhline(base, color="#ee6c4d", ls="--", lw=1, label="train labeled base")
            for r in tab["rows"]:
                ax.text(
                    r["q"],
                    r["y_rate"] + 0.003,
                    f"{r['y_rate']:.1%}\nn={r['n']}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Q{r['q']}\n{r['x_median']:.2f}" for r in tab["rows"]])
            ax.set_ylabel("P(Y=1)")
            ax.set_xlabel(xlab)
            ax.set_ylim(0, max(ys) * 1.35 if ys else 1)
            ax.legend(frameon=False, fontsize=7)
            ax.set_title(title)
        fig.suptitle("Y5 rate by night single quintile (not HHI)", fontsize=11)
        fig.tight_layout()
        OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)
        png = str(OUT_PNG)
        print(f"  wrote {OUT_PNG}")
    elif no_plot:
        print("  --no-plot")
    else:
        print("  matplotlib missing — skip PNG")
    return {"tables": tables, "png": png}


def _size_auroc_of_feature(df: pd.DataFrame, col: str, mask: pd.Series) -> dict:
    """Does this column rank large vs small firms? Train-median a_in3 cut."""
    x = pd.to_numeric(df[col], errors="coerce")
    size = pd.to_numeric(df["a_in3"], errors="coerce")
    m = mask & x.notna() & size.notna()
    if int(m.sum()) < 50:
        return {"auroc": float("nan"), "rho": float("nan")}
    med = float(size[m].median())
    large = (size[m] > med).astype(float)
    auc = auroc(large, x[m])
    auc2 = max(auc, 1.0 - auc) if np.isfinite(auc) else float("nan")
    return {
        "auroc": float(auc2) if np.isfinite(auc2) else float("nan"),
        "rho": spearman(x[m], np.log1p(size[m].abs())),
    }


def pass3_singles(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    print("\n" + "=" * 72)
    print("PASS 3 — single-feature train group-fold AUROC")
    print("KEEP non-E if monotone/tail AND beats size by ≥0.02 AND not SIZE_PARK")
    print("Never E as X.")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    rows = []
    for y_col in (Y_AP, Y_AR):
        lab = train_by_y[y_col]
        print(f"\n{y_col} n_lab={int(lab.sum())}")
        for col in SINGLE_COLS:
            if col not in df.columns:
                print(f"  missing {col}")
                continue
            if str(col).startswith("e_"):
                raise RuntimeError(f"E column in SINGLE_COLS: {col}")
            leak = leakage_check([col], y_col, forbidden_prefixes=FORBIDDEN)
            if not leak["ok"]:
                print(f"  SKIP {col}: {leak['issues']}")
                continue
            cov = coverage(pd.to_numeric(df[col], errors="coerce"), lab)
            oof = signed_oof_auroc(df, col, y_col, lab, folds)
            size_bin = _size_auroc_of_feature(df, col, lab)
            rec = {
                "y": y_col,
                "col": col,
                "cv": oof["cv"],
                "sd": oof["sd"],
                "train_auc": oof["train_auc"],
                "train_sign": oof["train_sign"],
                "coverage": cov["coverage"],
                "n_defined": cov["n_defined"],
                "size_auroc": size_bin["auroc"],
                "size_rho": size_bin["rho"],
                "size_park": bool(
                    np.isfinite(size_bin["auroc"]) and size_bin["auroc"] >= SIZE_AUROC
                ),
                "folds": oof["folds"],
            }
            rows.append(rec)
            park = " SIZE_PARK" if rec["size_park"] else ""
            print(
                f"  {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f}  sign={oof['train_sign']:+d}  "
                f"cov={cov['coverage']:.3f}  sizeAUC={size_bin['auroc']:.3f} "
                f"ρ={size_bin['rho']:+.3f}{park}"
            )
    return {"rows": rows}


def pass4_leak(df: pd.DataFrame, train_any: pd.Series) -> dict:
    print("\n" + "=" * 72)
    print("PASS 4 — leak vs E columns that define the label  (fail |ρ|≥0.80)")
    print("comparators only — never X. Cannot use E to score persistence.")
    print("=" * 72)
    rows = []
    xcols = [c for c in SINGLE_COLS if c in df.columns]
    for col in xcols:
        x = pd.to_numeric(df[col], errors="coerce")
        for vs in LEAK_VS:
            other = pd.to_numeric(df[vs], errors="coerce")
            m = train_any & x.notna() & other.notna()
            assert_no_holdout(df.loc[m, "company_id"])
            rec = {
                "col": col,
                "vs": vs,
                "n": int(m.sum()),
                "spearman": spearman(x[m], other[m]),
                "pearson": pearson(x[m], other[m]),
            }
            rec["fail"] = bool(np.isfinite(rec["spearman"]) and abs(rec["spearman"]) >= LEAK_RHO)
            rows.append(rec)
            mark = " FAIL" if rec["fail"] else ""
            print(
                f"  {col:22s} vs {vs:18s}  ρ_s={rec['spearman']:+.3f}  "
                f"ρ_p={rec['pearson']:+.3f}  n={rec['n']}{mark}"
            )
    fails = [r for r in rows if r["fail"]]
    print(f"  FAIL count={len(fails)}  (E rewrite if any non-E |ρ|≥{LEAK_RHO})")
    return {"rows": rows, "n_fail": len(fails)}


def pass5_holdout(df: pd.DataFrame, is_hold: pd.Series) -> dict:
    """Holdout n / n_pos only. Do not quote holdout AUROC as a keep."""
    print("\n" + "=" * 72)
    print("PASS 5 — holdout coverage (LOW_POWER, not a claim)")
    print("=" * 72)
    rows = []
    for y_col in (Y_AP, Y_AR, Y_DELAY):
        y = pd.to_numeric(df[y_col], errors="coerce")
        lab = is_hold & y.notna()
        rec = {
            "y": y_col,
            "n": int(lab.sum()),
            "n_pos": int((is_hold & (y == 1)).sum()),
            "n_cos": int(df.loc[lab, "company_id"].nunique()),
            "n_pos_cos": int(df.loc[is_hold & (y == 1), "company_id"].nunique()),
            "rate": float(y[lab].mean()) if int(lab.sum()) else float("nan"),
            "flag": "LOW_POWER",
        }
        rows.append(rec)
        print(
            f"  {y_col:22s} n={rec['n']:4d} pos={rec['n_pos']:3d} "
            f"cos={rec['n_cos']:3d} pos_cos={rec['n_pos_cos']:3d} "
            f"rate={rec['rate']:.3f} LOW_POWER"
        )
    return {"rows": rows}


def pass6_lag1(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Honest 1-month Q6. Do not claim a quarter lead."""
    print("\n" + "=" * 72)
    print("PASS 6 — Q6 clock  lag1 of d_supp_hhi / a_io_ratio (honest 1m only)")
    print("Q6 quoted: only 1-month leads transfer. Do not claim t3.")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    rows = []
    for y_col in (Y_AP, Y_AR):
        lab = train_by_y[y_col]
        stems = (
            ("d_supp_hhi", "a_io_ratio", "h_group_size")
            if y_col == Y_AP
            else ("d_cust_hhi", "a_io_ratio", "d_tx_cp_share")
        )
        for stem in stems:
            for lag, col in ((0, stem), (1, f"{stem}_lag1")):
                if col not in df.columns:
                    continue
                if str(col).startswith("e_"):
                    raise RuntimeError(f"E in Q6: {col}")
                cov = coverage(pd.to_numeric(df[col], errors="coerce"), lab)
                oof = signed_oof_auroc(df, col, y_col, lab, folds)
                rec = {
                    "y": y_col,
                    "stem": stem,
                    "lag": lag,
                    "col": col,
                    "cv": oof["cv"],
                    "sd": oof["sd"],
                    "train_auc": oof["train_auc"],
                    "train_sign": oof["train_sign"],
                    "coverage": cov["coverage"],
                    "n_defined": cov["n_defined"],
                    "folds": oof["folds"],
                }
                rows.append(rec)
                print(
                    f"  {y_col:22s} {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                    f"train={oof['train_auc']:.4f}  sign={oof['train_sign']:+d}  "
                    f"cov={cov['coverage']:.3f}"
                )
    return {"rows": rows}


def decide(p1: dict, p2: dict, p3: dict, p4: dict, p14: dict | None = None) -> dict:
    """KEEP / CLOSE / PARK a non-E Q5 column. Trees stay PARK."""
    leak_fail = {r["col"] for r in p4["rows"] if r["fail"]}
    qmap = {(t["x"], t["y"]): t for t in p2["tables"]}
    cands = []
    for y_col in (Y_AP, Y_AR):
        size = next((r for r in p3["rows"] if r["y"] == y_col and r["col"] == "log1p_a_in3"), None)
        size_cv = float(size["cv"]) if size and np.isfinite(size["cv"]) else CHANCE
        night = NIGHT_AP if y_col == Y_AP else NIGHT_AR
        night_row = next(
            (r for r in p3["rows"] if r["y"] == y_col and r["col"] == night["col"]),
            None,
        )
        for r in p3["rows"]:
            if r["y"] != y_col:
                continue
            if str(r["col"]).startswith("e_"):
                continue
            tab = qmap.get((r["col"], y_col))
            shape = bool(
                tab
                and (
                    tab.get("monotone_up")
                    or tab.get("monotone_down")
                    or tab.get("tail_only")
                    or tab.get("head_only")
                )
            )
            # shape comes from train quintiles when the column was cut
            gap_size = float(r["cv"] - size_cv) if np.isfinite(r["cv"]) else float("nan")
            if r["col"] in leak_fail:
                dec = "PARK"
                why = f"leak |ρ|≥{LEAK_RHO} vs E (that built the Y)"
            elif r["size_park"]:
                dec = "PARK"
                why = f"size AUROC {r['size_auroc']:.3f} ≥ {SIZE_AUROC}"
            elif r["col"] == "log1p_a_in3":
                dec = "PARK"
                why = "size proxy itself"
            elif np.isfinite(r["cv"]) and r["cv"] < CHANCE + CLEAR_MARGIN:
                dec = "CLOSE"
                why = f"near chance (CV {r['cv']:.3f})"
            elif (
                r["col"] == "d_tx_cp_share"
                and y_col == Y_AR
                and p14
                and (p14.get("presence_rewrite") or p14.get("intensity_dead"))
                and np.isfinite(gap_size)
                and gap_size >= CLEAR_MARGIN
                and shape
            ):
                dec = "KEEP"
                why = (
                    f"Q5 unnamed-cp head: CV {r['cv']:.3f} beats size {size_cv:.3f} "
                    f"by {gap_size:+.3f}; intensity on cp>0 {p14.get('ar_pos_cv', float('nan')):.3f} "
                    f"(dead={p14.get('intensity_dead')}). PARK as model X; KEEP the sentence."
                )
            elif np.isfinite(gap_size) and gap_size >= CLEAR_MARGIN and shape:
                dec = "KEEP"
                why = (
                    f"shape+CV {r['cv']:.3f} beats size {size_cv:.3f} by {gap_size:+.3f}"
                )
            elif np.isfinite(gap_size) and gap_size >= CLEAR_MARGIN and not shape:
                dec = "CLOSE"
                why = (
                    f"beats size by {gap_size:+.3f} but no monotone/tail quintile "
                    f"(CV {r['cv']:.3f})"
                )
            else:
                dec = "CLOSE"
                why = (
                    f"no clear Q5 (CV {r['cv']:.3f} vs size {size_cv:.3f}, "
                    f"gap {gap_size:+.3f}, shape={shape})"
                )
            cands.append(
                {
                    **r,
                    "gap_vs_size": gap_size,
                    "size_cv": size_cv,
                    "shape": shape,
                    "decision": dec,
                    "reason": why,
                    "night_col": night["col"],
                    "night_train": night["train_auc"],
                    "night_cv": (night_row or {}).get("cv", float("nan")),
                }
            )

    keepers = [c for c in cands if c["decision"] == "KEEP"]
    by_y = {}
    for y_col in (Y_AP, Y_AR):
        yc = [c for c in cands if c["y"] == y_col]
        k = [c for c in yc if c["decision"] == "KEEP"]
        best = max(
            (c for c in yc if np.isfinite(c["cv"]) and c["col"] != "log1p_a_in3"),
            key=lambda c: c["cv"],
            default=None,
        )
        legal = max(
            (c for c in yc if c["decision"] != "PARK" and np.isfinite(c["cv"])
             and c["col"] != "log1p_a_in3"),
            key=lambda c: c["cv"],
            default=None,
        )
        head = "KEEP" if k else (
            "CLOSE" if any(c["decision"] == "CLOSE" for c in yc) else "PARK"
        )
        by_y[y_col] = {
            "headline": head,
            "keepers": k,
            "best": best,
            "legal": legal,
            "story": p1["stories"][y_col]["story"],
        }

    if keepers:
        headline = "KEEP"
    elif any(by_y[y]["headline"] == "CLOSE" for y in (Y_AP, Y_AR)):
        headline = "CLOSE"
    else:
        headline = "PARK"

    print("\n" + "=" * 72)
    print(f"VERDICT {headline}  trees stay PARK")
    for y_col in (Y_AP, Y_AR):
        print(f"  {y_col} story={by_y[y_col]['story']} headline={by_y[y_col]['headline']}")
        for c in cands:
            if c["y"] != y_col:
                continue
            print(f"    {c['col']:22s} {c['decision']:5s}  {c['reason']}")
    print("=" * 72)
    return {
        "headline": headline,
        "cands": cands,
        "by_y": by_y,
        "trees": "PARK",
    }


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
    p1: dict,
    p2: dict,
    p3: dict,
    p4: dict,
    p5: dict,
    p6: dict,
    verdict: dict,
    ts: str,
    extra: dict | None = None,
) -> list[dict]:
    out = []

    def add(model, y, metric, value, coverage, notes, families="A"):
        out.append(
            {
                "ts": ts,
                "round": ROUND,
                "wave": WAVE,
                "agent": AGENT,
                "x_families": families,
                "y": y,
                "model": model,
                "split": "cv5_group" if metric == "auroc" else "train",
                "metric": metric,
                "value": _fmt(value),
                "coverage": (
                    f"{coverage:.4f}"
                    if isinstance(coverage, (int, float)) and np.isfinite(coverage)
                    else ""
                ),
                "notes": notes,
            }
        )

    for y_col, st in p1["stories"].items():
        add(
            "y5_why_story",
            y_col,
            "story",
            {"cash_stress": 0, "supplier_tail": 1, "customer_tail": 1, "unexplained_leftover": 0.5}.get(
                st["story"], -1
            ),
            1.0,
            f"{st['story']}; cash={st['cash_share']:.3f} hhi={st['hhi_share']:.3f} "
            f"neither={st['neither_share']:.3f}; never E; no GBM",
            families="A-vs-D",
        )
    for r in p1["rows"]:
        fam = "-" if r["flag"] in (Y2, Y4, Y9) else ("A" if "io" in r["flag"] or r["flag"].startswith("a_") else ("C" if r["flag"].startswith("c_") else "D"))
        add(
            f"y5_why_{r['flag']}",
            r["y"],
            "share_of_positives",
            r["share_of_pos_defined"],
            r["coverage_of_pos"],
            f"own-cut flag; n_hi={r['n_hi']} n_def={r['n_defined']} n_pos={r['n_pos']}; "
            f"train labeled only; never E",
            families=fam,
        )
    for r in p3["rows"]:
        fam = r["col"][0].upper() if r["col"] and r["col"][0].isalpha() else "A"
        if r["col"].startswith("log1p"):
            fam = "A"
        add(
            f"single_{r['col']}",
            r["y"],
            "auroc",
            r["cv"],
            r["coverage"],
            f"group-fold; sign from train fold; CV={r['cv']:.4f}±{r['sd']:.3f}; "
            f"sign={r['train_sign']:+d}; sizeAUC={r['size_auroc']:.3f}; "
            f"size_park={r['size_park']}; never E; quote CV not holdout",
            families=fam,
        )
    for r in p4["rows"]:
        add(
            f"leak_{r['col']}",
            Y_AP,
            f"spearman_{r['vs']}",
            r["spearman"],
            r["n"] / max(r["n"], 1),
            f"train any-Y5; pearson={r['pearson']}; fail={r['fail']}; "
            f"E comparator only, not X",
            families=f"{r['col'][0].upper()}-vs-E" if r["col"] else "vs-E",
        )
    for r in p5["rows"]:
        add(
            "y5_holdout_coverage",
            r["y"],
            "n_pos",
            r["n_pos"],
            1.0,
            f"LOW_POWER; n={r['n']} pos={r['n_pos']} cos={r['n_cos']} "
            f"pos_cos={r['n_pos_cos']}; not an AUROC claim",
            families="-",
        )
    for r in p6["rows"]:
        fam = r["col"][0].upper() if r["col"] else "A"
        add(
            f"single_{r['col']}",
            r["y"],
            "auroc",
            r["cv"],
            r["coverage"],
            f"Q6 honest 1m; lag={r['lag']}; CV={r['cv']:.4f}±{r['sd']:.3f}; never t3; never E",
            families=fam,
        )
    for t in p2["tables"]:
        fam = t["x"][0].upper() if t["x"] else "A"
        if t["x"].startswith("log1p"):
            fam = "A"
        for r in t["rows"]:
            add(
                f"{t['x']}_quintile",
                t["y"],
                f"y_rate_q{r['q']}",
                r["y_rate"],
                r["n"] / t["n"] if t["n"] else float("nan"),
                f"train labeled cuts; n={r['n']} pos={r['n_pos']} interval={r['interval']}; "
                f"up={t['monotone_up']} down={t['monotone_down']} tail={t['tail_only']}",
                families=fam,
            )
    add(
        "y5_why_verdict",
        Y_AP,
        "decision",
        {"KEEP": 1, "CLOSE": 0, "PARK": -1}.get(verdict["headline"], -9),
        1.0,
        f"{verdict['headline']}; trees={verdict['trees']}; "
        f"AP={verdict['by_y'][Y_AP]['headline']}/{verdict['by_y'][Y_AP]['story']}; "
        f"AR={verdict['by_y'][Y_AR]['headline']}/{verdict['by_y'][Y_AR]['story']}",
        families="A+D+C",
    )
    if extra:
        for r in extra.get("registry") or []:
            add(
                r["model"],
                r.get("y", Y_AP),
                r["metric"],
                r["value"],
                r.get("coverage", 1.0),
                r.get("notes", ""),
                families=r.get("families", "A"),
            )
    return out


def _extra_registry(
    p7, p8, p10, p11, p12, p13, p14=None, p15=None, p16=None, p17=None, p18=None,
    p19=None, p20=None, p21=None, p22=None, p23=None, p24=None,
    p25=None, p26=None, p27=None, p28=None, p29=None, p30=None, p31=None, p32=None,
    p33=None, p34=None, p35=None, p36=None, p37=None, p38=None, p39=None, p40=None,
    p41=None, p42=None, p43=None,
) -> list[dict]:
    rows = []
    for r in p7["rows"]:
        rows.append(
            {
                "model": f"{r['col']}_tail_gt0975",
                "y": r["y"],
                "metric": "y_rate",
                "value": r["rate_hi"],
                "coverage": 1.0,
                "notes": (
                    f"Y4-shape tail; rest={r['rate_lo']}; tail_auroc={r['auroc_tail']}; "
                    f"n={r['n_hi']} pos={r['n_pos_hi']}; never E"
                ),
                "families": "D",
            }
        )
    for y_col, blk in p8.items():
        rows.append(
            {
                "model": "y5_neither_cell",
                "y": y_col,
                "metric": "share_of_positives",
                "value": blk["n"] / blk["n_2x2"] if blk["n_2x2"] else float("nan"),
                "coverage": 1.0,
                "notes": f"neither low-io nor high HHI; n={blk['n']}/{blk['n_2x2']}",
                "families": "-",
            }
        )
    for r in p10["rows"]:
        rows.append(
            {
                "model": f"single_{r['col']}_{r['slice']}",
                "y": r["y"],
                "metric": "auroc",
                "value": r["cv"],
                "coverage": 1.0,
                "notes": f"size residual; n={r['n']} pos={r['n_pos']} rate={r['rate']}",
                "families": r["col"][0].upper(),
            }
        )
    for r in p11["rows"]:
        if r["slice"] != "short_<12":
            continue
        rows.append(
            {
                "model": f"single_{r['col']}_short",
                "y": r["y"],
                "metric": "auroc",
                "value": r["cv"],
                "coverage": r["coverage"],
                "notes": (
                    f"Q6 short so-far<12; lag={r['lag']}; CV={r['cv']:.4f}±{r['sd']:.3f}; "
                    f"n_pos={r['n_pos']}; never t3"
                ),
                "families": r["col"][0].upper(),
            }
        )
    for r in p12["rows"]:
        rows.append(
            {
                "model": "y5_company_trait",
                "y": r["y"],
                "metric": "spearman",
                "value": r["rho"],
                "coverage": 1.0,
                "notes": (
                    f"x={r['x']}; n_cos={r['n_cos']} ever_pos={r['n_ever_pos']}; "
                    f"med_x pos/neg={r['x_pos']}/{r['x_neg']}"
                ),
                "families": "-",
            }
        )
    rows.append(
        {
            "model": "y5_ap_ar_overlap",
            "y": Y_AP,
            "metric": "spearman",
            "value": p13["rho"],
            "coverage": 1.0,
            "notes": (
                f"both_pos={p13['n_both_pos']} ap_only={p13['n_ap_only']} "
                f"ar_only={p13['n_ar_only']}"
            ),
            "families": "-",
        }
    )
    if p14:
        rows.append(
            {
                "model": "d_tx_cp_share_vs_anycp",
                "y": Y_AR,
                "metric": "auroc_gap",
                "value": p14["gap_vs_binary"],
                "coverage": 1.0,
                "notes": (
                    f"presence_rewrite={p14['presence_rewrite']}; "
                    f"intensity_dead={p14['intensity_dead']}; "
                    f"any_cp={p14['ar_bin_cv']:.4f}; all={p14['ar_all_cv']:.4f}; "
                    f"cp>0={p14['ar_pos_cv']:.4f}"
                ),
                "families": "D",
            }
        )
    if p15:
        for r in p15["rows"]:
            rows.append(
                {
                    "model": "h_group_size_ge18",
                    "y": r["y"],
                    "metric": "y_rate",
                    "value": r["rate_hi"],
                    "coverage": 1.0,
                    "notes": (
                        f"n={r['n_hi']} pos={r['n_pos_hi']}; rest={r['rate_lo']}; "
                        f"flag_auc={r['auroc_tail']}"
                    ),
                    "families": "H",
                }
            )
    if p16:
        for r in p16["rows"]:
            rows.append(
                {
                    "model": f"single_{r['col']}_neither",
                    "y": r["y"],
                    "metric": "auroc",
                    "value": r["cv"],
                    "coverage": 1.0,
                    "notes": f"neither leftover; n={r['n']} pos={r['n_pos']}; CV={r['cv']:.4f}±{r['sd']:.3f}",
                    "families": r["col"][0].upper(),
                }
            )
    if p17:
        for r in p17["rows"]:
            rows.append(
                {
                    "model": f"y5_cash_present_{r['slice']}",
                    "y": r["y"],
                    "metric": "share_days_gt0",
                    "value": r["share_days_gt0"],
                    "coverage": 1.0,
                    "notes": (
                        f"n={r['n']}; in3_def={r['share_in3_def']}; "
                        f"med_days={r['med_days']}; CLOSE missing-cash"
                    ),
                    "families": "-",
                }
            )
    if p18:
        rows.append(
            {
                "model": "y5_ar_tagging_hole",
                "y": Y_AR,
                "metric": "y_rate",
                "value": p18.get("hole_rate"),
                "coverage": 1.0,
                "notes": (
                    f"tagging_hole={p18.get('tagging_hole')}; "
                    f"named={p18.get('named_rate')}; thin={p18.get('thin_rate')}; "
                    f"med_n_cust={p18.get('med_n_cust')}"
                ),
                "families": "D",
            }
        )
        rows.append(
            {
                "model": "y5_ap_tagging_hole",
                "y": Y_AP,
                "metric": "y_rate",
                "value": p18.get("hole_rate_ap"),
                "coverage": 1.0,
                "notes": (
                    f"tagging_hole_ap={p18.get('tagging_hole_ap')}; "
                    f"named={p18.get('named_rate_ap')}; thin={p18.get('thin_rate_ap')}; "
                    f"med_n_supp={p18.get('med_n_supp')}"
                ),
                "families": "D",
            }
        )
    rows.extend(_registry_p19_p20(p19, p20))
    if p21:
        rows.append(
            {
                "model": "y5_ar_hole_lag1_short",
                "y": Y_AR,
                "metric": "auroc",
                "value": p21.get("short_cv"),
                "coverage": 1.0,
                "notes": (
                    f"Q6 hole_lag1 so-far<12 CV={p21.get('short_cv')}; "
                    f"long={p21.get('long_cv')}; all={p21.get('all_cv')}; never t3"
                ),
                "families": "D",
            }
        )
    if p22:
        rows.append(
            {
                "model": "y5_ar_hole_pos_share",
                "y": Y_AR,
                "metric": "share_of_pos",
                "value": p22.get("share_of_ar_pos"),
                "coverage": 1.0,
                "notes": (
                    f"hole_pos={p22.get('n_hole_pos')}/{p22.get('n_ar_pos')}; "
                    f"n_cos={p22.get('n_cos_hole_pos')} top1={p22.get('top1')} "
                    f"top3={p22.get('top3')} hhi={p22.get('hhi')} "
                    f"fragile={p22.get('fragile')}"
                ),
                "families": "D",
            }
        )
    if p23:
        rows.append(
            {
                "model": "single_d_tx_cp_share_hi_cust",
                "y": Y_AR,
                "metric": "auroc",
                "value": p23.get("hi_cv"),
                "coverage": 1.0,
                "notes": f"hi_cust CV; lo_cust={p23.get('lo_cv')}",
                "families": "D",
            }
        )
    if p24:
        rows.append(
            {
                "model": "single_d_tx_cp_share_diff1",
                "y": Y_AR,
                "metric": "auroc",
                "value": p24.get("cv"),
                "coverage": 1.0,
                "notes": "turning-month first difference; 1m only",
                "families": "D",
            }
        )
    if p25:
        rows.append(
            {
                "model": "y5_ar_short_hole_rate",
                "y": Y_AR,
                "metric": "y_rate",
                "value": p25.get("short_hole_rate"),
                "coverage": 1.0,
                "notes": (
                    f"short_named={p25.get('short_named_rate')}; "
                    f"long_hole={p25.get('long_hole_rate')}; "
                    f"long_named={p25.get('long_named_rate')}"
                ),
                "families": "D",
            }
        )
    if p26:
        for r in p26.get("rows") or []:
            rows.append(
                {
                    "model": f"single_{r['col']}_ap_leftover",
                    "y": Y_AP,
                    "metric": "auroc",
                    "value": r["cv"],
                    "coverage": 1.0,
                    "notes": (
                        f"sizeAUC={r['size_auroc']:.3f}; park={r['size_park']}; "
                        f"gap={r['gap_vs_size']:+.3f}"
                    ),
                    "families": "D",
                }
            )
    if p27:
        for r in p27.get("rows") or []:
            rows.append(
                {
                    "model": "d_tx_cp_share_vs_size",
                    "y": Y_AR,
                    "metric": "spearman",
                    "value": r["rho"],
                    "coverage": 1.0,
                    "notes": f"vs {r['col']}",
                    "families": "D",
                }
            )
    if p28:
        rows.append(
            {
                "model": "y5_ar_hole_vs_E",
                "y": Y_AR,
                "metric": "spearman",
                "value": p28.get("max_abs_rho_e"),
                "coverage": 1.0,
                "notes": f"leak_fail={p28.get('leak_fail')}; max|ρ| vs E",
                "families": "D",
            }
        )
    if p29 and p29.get("table"):
        t = p29["table"]
        rows.append(
            {
                "model": "d_tx_cp_share_hi_cust_head",
                "y": Y_AR,
                "metric": "head_only",
                "value": 1.0 if t.get("head_only") else 0.0,
                "coverage": 1.0,
                "notes": (
                    f"n={t.get('n')} up={t.get('monotone_up')} down={t.get('monotone_down')} "
                    f"tail={t.get('tail_only')} head={t.get('head_only')}"
                ),
                "families": "D",
            }
        )
    if p30:
        rows.append(
            {
                "model": "y5_ar_hole_fold_range",
                "y": Y_AR,
                "metric": "rate_range",
                "value": (
                    (p30.get("max_rate") - p30.get("min_rate"))
                    if np.isfinite(p30.get("max_rate", float("nan")))
                    and np.isfinite(p30.get("min_rate", float("nan")))
                    else float("nan")
                ),
                "coverage": 1.0,
                "notes": f"min={p30.get('min_rate')} max={p30.get('max_rate')}",
                "families": "D",
            }
        )
    if p31:
        rows.append(
            {
                "model": "y5_ar_hole_drop_heavy_fold",
                "y": Y_AR,
                "metric": "y_rate",
                "value": p31.get("hole_rate_wo"),
                "coverage": 1.0,
                "notes": (
                    f"heavy={p31.get('heavy_fold')} share_pos={p31.get('share_pos_heavy')}; "
                    f"named={p31.get('named_rate_wo')}; cv_wo={p31.get('cv_wo')}; "
                    f"survives={p31.get('survives')}"
                ),
                "families": "D",
            }
        )
    if p32:
        rows.append(
            {
                "model": "y5_ar_heavy_fold_who",
                "y": Y_AR,
                "metric": "n_companies",
                "value": p32.get("n_cos"),
                "coverage": 1.0,
                "notes": (
                    f"med_group={p32.get('med_group')}; med_share={p32.get('med_share')}; "
                    f"med_ncust={p32.get('med_ncust')}; hole_month_share={p32.get('share_hole_months')}"
                ),
                "families": "H+D",
            }
        )
    if p33:
        for r in p33.get("rows") or []:
            rows.append(
                {
                    "model": "d_tx_cp_share_med_by_fold",
                    "y": Y_AR,
                    "metric": "median",
                    "value": r["med_share"],
                    "coverage": 1.0,
                    "notes": (
                        f"fold={r['fold']} n={r['n']} cos={r['n_cos']} "
                        f"share0={r['share_eq0']} rate={r['rate']}"
                    ),
                    "families": "D",
                }
            )
    if p34:
        for r in p34.get("rows") or []:
            if r.get("book") != "hi":
                continue
            rows.append(
                {
                    "model": "y5_ar_zero_hi_cust_by_fold",
                    "y": Y_AR,
                    "metric": "y_rate",
                    "value": r["rate"],
                    "coverage": 1.0,
                    "notes": f"fold={r['fold']} n={r['n']} pos={r['n_pos']}",
                    "families": "D",
                }
            )
    if p35:
        rows.append(
            {
                "model": "y5_ap_rate_fold_range",
                "y": Y_AP,
                "metric": "rate_range",
                "value": (
                    (p35.get("max_rate") - p35.get("min_rate"))
                    if np.isfinite(p35.get("max_rate", float("nan")))
                    and np.isfinite(p35.get("min_rate", float("nan")))
                    else float("nan")
                ),
                "coverage": 1.0,
                "notes": f"min={p35.get('min_rate')} max={p35.get('max_rate')}",
                "families": "-",
            }
        )
    if p36:
        for r in p36.get("rows") or []:
            rows.append(
                {
                    "model": "y5_ap_neither_share_by_fold",
                    "y": Y_AP,
                    "metric": "share_of_pos",
                    "value": r["share_neither"],
                    "coverage": 1.0,
                    "notes": f"fold={r['fold']} neither={r['n_neither']}/{r['n_both_def']}",
                    "families": "-",
                }
            )
    if p37:
        for r in p37.get("rows") or []:
            rows.append(
                {
                    "model": "y5_ar_neither_share_by_fold",
                    "y": Y_AR,
                    "metric": "share_of_pos",
                    "value": r["share_neither"],
                    "coverage": 1.0,
                    "notes": f"fold={r['fold']} neither={r['n_neither']}/{r['n_both_def']}",
                    "families": "-",
                }
            )
    if p38:
        rows.append(
            {
                "model": "y5_ar_hole_in_neither",
                "y": Y_AR,
                "metric": "share_of_pos",
                "value": p38.get("share_neither"),
                "coverage": 1.0,
                "notes": f"hole_pos={p38.get('n_hole_pos')} neither={p38.get('n_neither')}",
                "families": "D",
            }
        )
    if p39:
        rows.append(
            {
                "model": "y5_ar_hole_pos_neither",
                "y": Y_AR,
                "metric": "n",
                "value": p39.get("neither"),
                "coverage": 1.0,
                "notes": (
                    f"n={p39.get('n')} cash_only={p39.get('cash_only')} "
                    f"hhi_only={p39.get('hhi_only')} both={p39.get('both')}"
                ),
                "families": "D",
            }
        )
    if p40:
        for r in p40.get("rows") or []:
            rows.append(
                {
                    "model": "y5_ar_so_far_by_fold",
                    "y": Y_AR,
                    "metric": "median",
                    "value": r["med_so_far"],
                    "coverage": 1.0,
                    "notes": f"fold={r['fold']} short={r['share_short']} long={r['share_long']}",
                    "families": "-",
                }
            )
    if p41:
        for r in p41.get("rows") or []:
            rows.append(
                {
                    "model": "y5_ap_so_far_by_fold",
                    "y": Y_AP,
                    "metric": "median",
                    "value": r["med_so_far"],
                    "coverage": 1.0,
                    "notes": f"fold={r['fold']} rate={r['rate']} short={r['share_short']}",
                    "families": "-",
                }
            )
    if p42:
        for r in p42.get("rows") or []:
            rows.append(
                {
                    "model": "single_so_far",
                    "y": r["y"],
                    "metric": "auroc",
                    "value": r["cv"],
                    "coverage": 1.0,
                    "notes": f"tenure single; sign={r['sign']}; sizeAUC={r['size_auroc']}",
                    "families": "-",
                }
            )
    if p43:
        for r in p43.get("rows") or []:
            rows.append(
                {
                    "model": "h_group_size_ge18_by_fold",
                    "y": Y_AP,
                    "metric": "y_rate",
                    "value": r["rate_hi"],
                    "coverage": 1.0,
                    "notes": f"fold={r['fold']} n={r['n_hi']} pos={r['n_pos_hi']} rest={r['rate_rest']}",
                    "families": "H",
                }
            )
    return rows


def _registry_p19_p20(p19: dict | None, p20: dict | None) -> list[dict]:
    rows = []
    if p19:
        rows.append(
            {
                "model": "y5_ar_hole_vs_cash",
                "y": Y_AR,
                "metric": "y_rate",
                "value": p19.get("hole_ok_rate"),
                "coverage": 1.0,
                "notes": (
                    f"cash_independent={p19.get('cash_independent')}; "
                    f"hole_cash={p19.get('hole_cash_rate')}; "
                    f"named_ok={p19.get('named_ok_rate')}; "
                    f"named_cash={p19.get('named_cash_rate')}"
                ),
                "families": "D",
            }
        )
    if p20:
        rows.append(
            {
                "model": "y5_ar_hole_flag_hi_cust",
                "y": Y_AR,
                "metric": "auroc",
                "value": p20.get("cv"),
                "coverage": 1.0,
                "notes": (
                    f"hi_cust only; CV={p20.get('cv')}±{p20.get('sd')}; "
                    f"n={p20.get('n_hi_cust')} pos={p20.get('n_pos_hi_cust')}; "
                    f"ever_hole_cos={p20.get('n_cos_ever_hole')}/"
                    f"{p20.get('n_cos_hi_cust')} "
                    f"rate_ever={p20.get('rate_ever_hole')} "
                    f"rate_never={p20.get('rate_never_hole')}; "
                    f"holdout_hole n={p20.get('ho_n_hole')} pos={p20.get('ho_n_hole_pos')} LOW_POWER"
                ),
                "families": "D",
            }
        )
    return rows


def write_md(
    started: str,
    n_tr: dict,
    p1: dict,
    p2: dict,
    p3: dict,
    p4: dict,
    p5: dict,
    p6: dict,
    verdict: dict,
    extra: dict | None = None,
) -> None:
    ap_best = (verdict["by_y"][Y_AP].get("best") or {})
    ar_best = (verdict["by_y"][Y_AR].get("best") or {})
    lines = [
        "# Y5 why — cash-stress, supplier-tail, or leftover",
        "",
        f"- **When:** {started}",
        f"- **Agent:** `{AGENT}`",
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        f"- **Re-run:** `python -m analysis.evaluate.y5_why`",
        f"- **Holdout:** 72 companies, seed {FOLD_SEED}. Coverage only. Rates + singles on train.",
        f"- **Y:** `{Y_AP}` / `{Y_AR}` from `targets.parquet` (not rebuilt). "
        f"`{Y_DELAY}` rejected — coverage only. "
        f"Train AP {n_tr[Y_AP]['n']} / {n_tr[Y_AP]['n_pos']} / **{n_tr[Y_AP]['rate']:.2%}**; "
        f"AR {n_tr[Y_AR]['n']} / {n_tr[Y_AR]['n_pos']} / **{n_tr[Y_AR]['rate']:.2%}**.",
        "- **X candidates:** `d_supp_hhi` / `d_cust_hhi` / `a_out6` / `log1p(a_in3)` / "
        "`c_n_days_with_tx` / `a_io_ratio` / `c_gap_sd` + night singles "
        "`h_group_size` / `d_tx_cp_share`. **Never E.**",
        "- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER. Trees stay PARK.",
        "- **Brief:** Q5 why / honest 1-month Q6. Not bankruptcy. Not a 0–100.",
        "",
        "## Decision",
        "",
        f"**{verdict['headline']}** a non-E Q5 sentence on AR only. Trees stay **PARK**. "
        f"AP is **{verdict['by_y'][Y_AP]['story']}** / {verdict['by_y'][Y_AP]['headline']}. "
        f"AR cash/HHI is leftover; the usable why is the night single "
        f"`d_tx_cp_share` ({verdict['by_y'][Y_AR]['headline']}).",
        "",
        f"Best non-E AP `{ap_best.get('col')}` CV **{ap_best.get('cv', float('nan')):.3f}** "
        f"(night `{NIGHT_AP['col']}` train {NIGHT_AP['train_auc']:.3f}, "
        f"this-run CV {ap_best.get('night_cv', float('nan')):.3f}) — "
        f"beats size but quintiles are not monotone (large groups 2.9% are protective).",
        f"Best non-E AR `{ar_best.get('col')}` CV **{ar_best.get('cv', float('nan')):.3f}** "
        f"(night `{NIGHT_AR['col']}` train {NIGHT_AR['train_auc']:.3f}) — "
        f"zero-named-cp head 12.8% plus intensity on cp>0 (CV 0.580). "
        f"Cut 18: among above-median `d_n_cust`, unnamed bank months are **17.4%** vs **6.3%** named — "
        f"a tagging hole, not invoice thinness. "
        + (
            f"AP analogue is **false** "
            f"({(extra or {}).get('p18', {}).get('hole_rate_ap', float('nan')):.1%} vs "
            f"{(extra or {}).get('p18', {}).get('named_rate_ap', float('nan')):.1%} named among many-supplier months)."
            if (extra or {}).get("p18")
            else ""
        )
        + (
            f" Cut 19: hole is cash-independent "
            f"({(extra or {}).get('p19', {}).get('hole_ok_rate', float('nan')):.1%} "
            f"vs named-ok {(extra or {}).get('p19', {}).get('named_ok_rate', float('nan')):.1%})."
            if (extra or {}).get("p19")
            else ""
        )
        + (
            f" Cut 21 hole_lag1 short CV "
            f"{(extra or {}).get('p21', {}).get('short_cv', float('nan')):.3f} — Q6 stays CLOSE."
            if (extra or {}).get("p21")
            else ""
        )
        + (
            f" Hi-cust Q1 unnamed **19.5%** vs Q5 3.4% (head-only). "
            f"Hole is {((extra or {}).get('p22') or {}).get('share_of_ar_pos', float('nan')):.0%} of AR pos "
            f"across {((extra or {}).get('p22') or {}).get('n_cos_hole_pos')} companies "
            f"(fragile={((extra or {}).get('p22') or {}).get('fragile')})."
            if (extra or {}).get("p22")
            else ""
        )
        + (
            f" Cut 31: fold {((extra or {}).get('p31') or {}).get('heavy_fold')} owns "
            f"{((extra or {}).get('p31') or {}).get('share_pos_heavy', float('nan')):.0%} of hole pos; "
            f"without it hole {((extra or {}).get('p31') or {}).get('hole_rate_wo', float('nan')):.1%} "
            f"vs named {((extra or {}).get('p31') or {}).get('named_rate_wo', float('nan')):.1%} "
            f"(survives={((extra or {}).get('p31') or {}).get('survives')})."
            if (extra or {}).get("p31")
            else ""
        )
        + " Cut 34: fold 3 zero+hi-cust is 25.1% (48/191); fold 2 zero+hi is 3.3% (2/60). "
        "KEEP the night single; CLOSE 17.4% as a leave-one-group law. "
        "Tenure (so-far) is CV 0.55/0.56 — not a book-age rewrite.",
        "",
        "Thin-F first-run winners (`f_w_rate_lag1` 0.668 / `f_months_to_next_pay_lag3` 0.677) "
        "stay PARK — 1.7% debt-schedule panel, not a cash/D quote.",
        "",
        "We **cannot** use family E to show “already overdue last month”. "
        "Non-E |ρ| vs `e_ap_overdue_30` / `e_ar_overdue_30` / delay stays <0.18, "
        "so cash and HHI are not overdue rewrites. AP has no monotone/tail non-E "
        "that clears size+0.02. AR does: unnamed / thinly-named bank counterparties. "
        "**Q6 CLOSE:** `d_tx_cp_share` dies on so-far<12 (CV 0.445 / lag1 0.431, 75 pos). "
        "Only the 1-month clock is even eligible, and it does not transfer to short books.",
        "",
        "## Pass 1 — decompose positives",
        "",
        "High flags are that company's own expanding p80 (months ≤ t, min 6). "
        "Low `a_io_ratio` is own p20. Y2 / Y4 / Y9 are accepted labels from the same parquet. Never E.",
        "",
        "| Y | flag | n_hi / n_defined pos | share of pos | coverage of pos |",
        "|---|---|---:|---:|---:|",
    ]
    for r in p1["rows"]:
        lines.append(
            f"| `{r['y']}` | `{r['flag']}` | {r['n_hi']} / {r['n_defined']} | "
            f"{r['share_of_pos_defined']:.3f} | {r['coverage_of_pos']:.3f} |"
        )
    lines += [
        "",
        "### 2×2 low-`a_io_ratio` × high counterpart HHI among positives",
        "",
    ]
    for y_col in (Y_AP, Y_AR):
        t = p1["twos"][y_col]
        st = p1["stories"][y_col]
        lines.append(
            f"- `{y_col}` n={t['n']}: cash_only={t['cash_only']} hhi_only={t['hhi_only']} "
            f"both={t['both']} neither={t['neither']} "
            f"(neither **{st['neither_share']:.1%}**). Read: **{st['story']}**."
        )
    lines += [
        "",
        "## Pass 2 — quintiles (train labeled cuts)",
        "",
    ]
    for tab in p2["tables"]:
        lines += [
            f"### `{tab['x']}` vs `{tab['y']}`",
            "",
            f"n={tab['n']} bins={tab['n_bins']} up={tab['monotone_up']} "
            f"down={tab['monotone_down']} tail_only={tab['tail_only']} "
            f"head_only={tab['head_only']}",
            "",
            "| Q | interval | n | n_pos | P(Y=1) | median X |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for r in tab["rows"]:
            lines.append(
                f"| {r['q']} | {r['interval']} | {r['n']} | {r['n_pos']} | "
                f"{r['y_rate']:.3f} | {r['x_median']:.4g} |"
            )
        lines.append("")
    if p2.get("png"):
        lines.append(f"Plot: `{Path(p2['png']).name}`.")
        lines.append("")
    lines += [
        "## Pass 3 — single-feature CV",
        "",
        f"Night honest singles (quote these, not the tree): AP `{NIGHT_AP['col']}` "
        f"train {NIGHT_AP['train_auc']:.3f}; AR `{NIGHT_AR['col']}` "
        f"train {NIGHT_AR['train_auc']:.3f}. Chance = 0.50. "
        f"Size = `log1p(a_in3)`. PARK a column as X if its size AUROC ≥ {SIZE_AUROC}.",
        "",
        "| Y | feature | CV AUROC ± sd | train | sign | coverage | size AUROC | size ρ |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in p3["rows"]:
        lines.append(
            f"| `{r['y']}` | `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | "
            f"{r['train_auc']:.3f} | {r['train_sign']:+d} | {r['coverage']:.3f} | "
            f"{r['size_auroc']:.3f} | {r['size_rho']:+.3f} |"
        )
    lines += [
        "",
        "### Column letters",
        "",
        "| Y | feature | decision | reason |",
        "|---|---|---|---|",
    ]
    for c in verdict["cands"]:
        lines.append(
            f"| `{c['y']}` | `{c['col']}` | **{c['decision']}** | {c['reason']} |"
        )
    lines += [
        "",
        "## Pass 4 — leak vs E (comparators only)",
        "",
        f"Fail if Spearman |ρ| ≥ {LEAK_RHO} vs `e_ap_overdue_30` / `e_ar_overdue_30` / delay. "
        "That would be the Y. **Never E as X.** "
        "A failed leak is a rewrite, not a why.",
        "",
        "| feature | vs | n | Spearman | Pearson | fail |",
        "|---|---|---:|---:|---:|---|",
    ]
    for r in p4["rows"]:
        lines.append(
            f"| `{r['col']}` | `{r['vs']}` | {r['n']} | {r['spearman']:+.3f} | "
            f"{r['pearson']:+.3f} | {r['fail']} |"
        )
    lines += [
        "",
        f"FAIL count: **{p4['n_fail']}**.",
        "",
        "## Pass 5 — holdout coverage (LOW_POWER)",
        "",
        "Do not quote holdout AUROC as a keep. Expect LOW_POWER (14 / 8 pos in the power table).",
        "",
        "| Y | n | n_pos | companies | pos companies | rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in p5["rows"]:
        lines.append(
            f"| `{r['y']}` | {r['n']} | {r['n_pos']} | {r['n_cos']} | "
            f"{r['n_pos_cos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        "## Pass 6 — honest 1-month Q6",
        "",
        "Only lag-1 is transferable (see `q6_quoted.md`). Do not claim a quarter lead.",
        "",
        "| Y | feature | lag | CV AUROC ± sd | train | sign | coverage |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in p6["rows"]:
        lines.append(
            f"| `{r['y']}` | `{r['col']}` | {r['lag']} | {r['cv']:.3f} ± {r['sd']:.3f} | "
            f"{r['train_auc']:.3f} | {r['train_sign']:+d} | {r['coverage']:.3f} |"
        )
    lines += [
        "",
        "## Mapping (Q5 / Q6)",
        "",
        _mapping_paragraph(p1, p3, p6, verdict),
        "",
        "## Extra cuts",
        "",
    ]
    if extra and extra.get("md_lines"):
        lines.extend(extra["md_lines"])
    else:
        lines.append("Next iteration in this module.")
    lines.append("")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _mapping_paragraph(p1: dict, p3: dict, p6: dict, verdict: dict) -> str:
    ap_st = p1["stories"][Y_AP]
    ar_st = p1["stories"][Y_AR]
    ap_best = verdict["by_y"][Y_AP].get("best") or {}
    ar_best = verdict["by_y"][Y_AR].get("best") or {}
    supp1 = next(
        (r for r in p6["rows"] if r["y"] == Y_AP and r["col"] == "d_supp_hhi_lag1"),
        None,
    )
    io1 = next(
        (r for r in p6["rows"] if r["y"] == Y_AP and r["col"] == "a_io_ratio_lag1"),
        None,
    )
    tx1 = next(
        (r for r in p6["rows"] if r["y"] == Y_AR and r["col"] == "d_tx_cp_share_lag1"),
        None,
    )
    return (
        f"Y5 is who is *turning* on invoice overdue>30d (Q5 payment behaviour; "
        f"Hirshleifer PastDue% / Banque de France >30d). "
        f"AP train positives are **{ap_st['story']}** "
        f"(low `a_io_ratio` {ap_st['cash_share']:.0%} of pos; high `d_supp_hhi` "
        f"{ap_st['hhi_share']:.0%}; 2×2 neither {ap_st['neither_share']:.0%}). "
        f"AR train positives are **{ar_st['story']}** "
        f"(low `a_io_ratio` {ar_st['cash_share']:.0%}; high `d_cust_hhi` "
        f"{ar_st['hhi_share']:.0%}; neither {ar_st['neither_share']:.0%}). "
        f"Best non-E AP `{ap_best.get('col')}` CV {ap_best.get('cv', float('nan')):.3f} "
        f"vs night `{NIGHT_AP['col']}` {NIGHT_AP['train_auc']:.3f} and chance 0.50. "
        f"Best non-E AR `{ar_best.get('col')}` CV {ar_best.get('cv', float('nan')):.3f} "
        f"vs night `{NIGHT_AR['col']}` {NIGHT_AR['train_auc']:.3f}. "
        f"Lag-1 `d_supp_hhi` {(supp1 or {}).get('cv', float('nan')):.3f}; "
        f"lag-1 `a_io_ratio` {(io1 or {}).get('cv', float('nan')):.3f}; "
        f"lag-1 `d_tx_cp_share` {(tx1 or {}).get('cv', float('nan')):.3f} "
        f"(short so-far<12 0.431 — Q6 CLOSE). "
        f"AR Q5 sentence (no E): months with a thick invoice book but almost no named "
        f"bank counterparties run 17.4% sustained AR-od30 vs 6.3% when the bank trail names them. "
        f"Cannot use E to show last-month overdue persistence "
        f"(non-E |ρ| vs e_* stays <0.18). "
        f"Trees stay PARK. Verdict **{verdict['headline']}**. Not a 0–100."
    )


def pass7_hhi_tail(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Y4-style monopoly tail. Train labeled only. Expect a null on AP/AR."""
    print("\n" + "=" * 72)
    print("CUT 7 — HHI > 0.975 tail vs rest (Y4 monopoly shape, unused here)")
    print("=" * 72)
    rows = []
    for y_col, col in ((Y_AP, "d_supp_hhi"), (Y_AR, "d_cust_hhi"), (Y_AP, "d_cust_hhi")):
        x = pd.to_numeric(df[col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        m = train_by_y[y_col] & x.notna() & y.notna()
        assert_no_holdout(df.loc[m, "company_id"])
        hi = m & (x > 0.975)
        lo = m & (x <= 0.975)
        rec = {
            "y": y_col,
            "col": col,
            "cut": 0.975,
            "n_hi": int(hi.sum()),
            "n_pos_hi": int((y[hi] == 1).sum()),
            "rate_hi": float(y[hi].mean()) if int(hi.sum()) else float("nan"),
            "n_lo": int(lo.sum()),
            "n_pos_lo": int((y[lo] == 1).sum()),
            "rate_lo": float(y[lo].mean()) if int(lo.sum()) else float("nan"),
            "auroc_tail": float(auroc(y[m], (x[m] > 0.975).astype(float))),
        }
        rows.append(rec)
        print(
            f"  {y_col:22s} {col:14s}>0.975  n={rec['n_hi']} pos={rec['n_pos_hi']} "
            f"P(Y=1)={rec['rate_hi']:.3f}  | rest n={rec['n_lo']} "
            f"P(Y=1)={rec['rate_lo']:.3f}  tail AUROC={rec['auroc_tail']:.3f}"
        )
    return {"rows": rows}


def pass8_neither(lab: pd.DataFrame) -> dict:
    """What sits in the 2×2 neither cell (not low-io, not high HHI)?"""
    print("\n" + "=" * 72)
    print("CUT 8 — neither cell (not low a_io, not high counterpart HHI)")
    print("=" * 72)
    out = {}
    for y_col, hhi_col in ((Y_AP, "d_supp_hhi_hi"), (Y_AR, "d_cust_hhi_hi")):
        pos = lab[y_col] == 1
        both = pos & lab["a_io_ratio_lo"].notna() & lab[hhi_col].notna()
        neither = both & (lab["a_io_ratio_lo"] == 0) & (lab[hhi_col] == 0)
        n = int(neither.sum())
        print(f"\n  {y_col} neither n={n} / 2x2 pos {int(both.sum())}")
        flags = {
            Y2: lab[Y2],
            Y4: lab[Y4],
            Y9: lab[Y9],
            "c_gap_sd_hi": lab["c_gap_sd_hi"],
            "a_out6_hi": lab["a_out6_hi"] if "a_out6_hi" in lab.columns else None,
        }
        rows = []
        for name, s in flags.items():
            if s is None:
                continue
            s = pd.to_numeric(s, errors="coerce")
            d = neither & s.notna()
            rec = {
                "flag": name,
                "n_defined": int(d.sum()),
                "n_hi": int((d & (s == 1)).sum()),
                "share": float(s[d].mean()) if int(d.sum()) else float("nan"),
            }
            rows.append(rec)
            print(f"    {name:20s} {rec['n_hi']:4d}/{rec['n_defined']:4d}  share={rec['share']:.3f}")
        out[y_col] = {"n": n, "n_2x2": int(both.sum()), "rows": rows}
    return out


def pass9_folds(p3: dict) -> dict:
    """Fold AUCs for night singles + days. Wide sd = do not KEEP."""
    print("\n" + "=" * 72)
    print("CUT 9 — fold AUCs for night singles / days / HHI")
    print("=" * 72)
    want = {
        Y_AP: ("h_group_size", "c_n_days_with_tx", "d_supp_hhi", "a_io_ratio", "d_tx_cp_share"),
        Y_AR: ("d_tx_cp_share", "c_n_days_with_tx", "d_cust_hhi", "a_io_ratio", "h_group_size"),
    }
    rows = []
    for y_col, cols in want.items():
        print(f"\n  {y_col}")
        print("    " + "feat".ljust(22) + " ".join(f"f{k:>6}" for k in range(N_FOLDS)) + "   cv    sd")
        for col in cols:
            r = next((x for x in p3["rows"] if x["y"] == y_col and x["col"] == col), None)
            if not r:
                continue
            aucs = [f["auroc"] for f in r["folds"]]
            bits = " ".join(f"{a:6.3f}" for a in aucs)
            print(f"    {col:22s}{bits}  {r['cv']:5.3f} {r['sd']:5.3f}")
            rows.append({"y": y_col, "col": col, "folds": aucs, "cv": r["cv"], "sd": r["sd"]})
    return {"rows": rows}


def pass10_size_residual(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Does the night single / days still rank Y inside a_in3 terciles?"""
    print("\n" + "=" * 72)
    print("CUT 10 — night single / days inside log1p(a_in3) terciles")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    size = pd.to_numeric(df["log1p_a_in3"], errors="coerce")
    rows = []
    for y_col, cols in (
        (Y_AP, ("h_group_size", "c_n_days_with_tx", "d_tx_cp_share")),
        (Y_AR, ("d_tx_cp_share", "c_n_days_with_tx", "h_group_size")),
    ):
        lab = train_by_y[y_col]
        tr = lab & size.notna()
        assert_no_holdout(df.loc[tr, "company_id"])
        _, bins = pd.qcut(size[tr], 3, retbins=True, duplicates="drop")
        terc = pd.Series(pd.cut(size, bins=bins, include_lowest=True), index=df.index)
        print(f"  {y_col} size tercile edges: {bins.tolist()}")
        for i, q in enumerate(sorted(terc.dropna().unique()), start=1):
            sl = lab & (terc == q)
            y = pd.to_numeric(df.loc[sl, y_col], errors="coerce")
            for col in cols:
                oof = signed_oof_auroc(df, col, y_col, sl, folds)
                rec = {
                    "y": y_col,
                    "slice": f"size_T{i}",
                    "col": col,
                    "n": int(sl.sum()),
                    "n_pos": int((y == 1).sum()),
                    "rate": float(y.mean()) if int(sl.sum()) else float("nan"),
                    "cv": oof["cv"],
                    "sd": oof["sd"],
                }
                rows.append(rec)
                print(
                    f"    T{i} {col:22s} n={rec['n']:4d} pos={rec['n_pos']:3d} "
                    f"rate={rec['rate']:.3f} CV={oof['cv']:.3f}"
                )
    return {"rows": rows}


def pass11_short_q6(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Lag1 coverage + CV on so-far<12. Only 1m is transferable."""
    print("\n" + "=" * 72)
    print("CUT 11 — short-book lag1 (so-far<12). Honest Q6 only.")
    print("=" * 72)
    per = df.sort_values(["company_id", "period"])
    first = per.groupby("company_id")["period"].transform("min")
    so_far = (
        (per["period"].dt.year - first.dt.year) * 12
        + (per["period"].dt.month - first.dt.month)
        + 1
    )
    df = df.copy()
    df["_so_far"] = so_far.reindex(df.index)
    folds = df["fold"].to_numpy()
    rows = []
    for y_col in (Y_AP, Y_AR):
        lab = train_by_y[y_col]
        short = lab & (df["_so_far"] < 12)
        long_ = lab & (df["_so_far"] >= 18)
        stems = (
            ("d_supp_hhi", "a_io_ratio", "h_group_size")
            if y_col == Y_AP
            else ("d_cust_hhi", "a_io_ratio", "d_tx_cp_share")
        )
        for sl_name, sl in (("short_<12", short), ("long_>=18", long_), ("all", lab)):
            n = int(sl.sum())
            n_pos = int((pd.to_numeric(df.loc[sl, y_col], errors="coerce") == 1).sum())
            print(f"  {y_col} {sl_name}: n={n} pos={n_pos}")
            for stem in stems:
                for lag, col in ((0, stem), (1, f"{stem}_lag1")):
                    if col not in df.columns:
                        continue
                    cov = coverage(pd.to_numeric(df[col], errors="coerce"), sl)
                    oof = signed_oof_auroc(df, col, y_col, sl, folds)
                    rec = {
                        "y": y_col,
                        "slice": sl_name,
                        "col": col,
                        "lag": lag,
                        "n": n,
                        "n_pos": n_pos,
                        "cv": oof["cv"],
                        "sd": oof["sd"],
                        "coverage": cov["coverage"],
                        "n_defined": cov["n_defined"],
                    }
                    rows.append(rec)
                    print(
                        f"    {col:22s} cov={cov['coverage']:.3f} "
                        f"CV={oof['cv']:.3f}±{oof['sd']:.3f} n_def={cov['n_defined']}"
                    )
    return {"rows": rows}


def pass12_company_trait(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Company-level: is Y5 a small-group trait or a turning month?"""
    print("\n" + "=" * 72)
    print("CUT 12 — company-level group-size / tx-cp vs Y5 rate (Q3 honesty)")
    print("=" * 72)
    rows = []
    for y_col, xcol in ((Y_AP, "h_group_size"), (Y_AR, "d_tx_cp_share")):
        lab = train_by_y[y_col]
        work = pd.DataFrame(
            {
                "company_id": df.loc[lab, "company_id"].astype(str),
                "y": pd.to_numeric(df.loc[lab, y_col], errors="coerce"),
                "x": pd.to_numeric(df.loc[lab, xcol], errors="coerce"),
            }
        ).dropna()
        g = work.groupby("company_id", sort=False).agg(
            n=("y", "size"),
            y_rate=("y", "mean"),
            x_mean=("x", "mean"),
            n_pos=("y", "sum"),
        )
        g = g[g["n"] >= 3]
        rho = spearman(g["x_mean"], g["y_rate"])
        ever = g["n_pos"] > 0
        rec = {
            "y": y_col,
            "x": xcol,
            "n_cos": int(len(g)),
            "n_ever_pos": int(ever.sum()),
            "rho": rho,
            "x_pos": float(g.loc[ever, "x_mean"].median()) if int(ever.sum()) else float("nan"),
            "x_neg": float(g.loc[~ever, "x_mean"].median()) if int((~ever).sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {y_col} vs {xcol}: n_cos={rec['n_cos']} ever_pos={rec['n_ever_pos']} "
            f"ρ={rho:+.3f}  med X pos/neg cos {rec['x_pos']:.3g}/{rec['x_neg']:.3g}"
        )
    return {"rows": rows}


def pass13_label_overlap(df: pd.DataFrame, train_row: pd.Series) -> dict:
    """AP ∩ AR same month — one story or two labels?"""
    print("\n" + "=" * 72)
    print("CUT 13 — AP ∩ AR overlap (train)")
    print("=" * 72)
    ap = pd.to_numeric(df[Y_AP], errors="coerce")
    ar = pd.to_numeric(df[Y_AR], errors="coerce")
    both = train_row & ap.notna() & ar.notna()
    rec = {
        "n_both_lab": int(both.sum()),
        "n_ap_only": int((train_row & (ap == 1) & (ar == 0)).sum()),
        "n_ar_only": int((train_row & (ap == 0) & (ar == 1)).sum()),
        "n_both_pos": int((train_row & (ap == 1) & (ar == 1)).sum()),
        "rho": spearman(ap[both], ar[both]),
        "n_ap_pos": int((train_row & (ap == 1)).sum()),
        "n_ar_pos": int((train_row & (ar == 1)).sum()),
    }
    print(
        f"  both-labeled n={rec['n_both_lab']} both-pos={rec['n_both_pos']} "
        f"AP-only={rec['n_ap_only']} AR-only={rec['n_ar_only']} ρ={rec['rho']:+.3f}"
    )
    return rec


def pass14_txcp_honesty(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is AR 0.576 just 'no named counterparty' vs share intensity?"""
    print("\n" + "=" * 72)
    print("CUT 14 — d_tx_cp_share honesty (zero-head vs intensity)")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    x = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    df = df.copy()
    df["_any_cp"] = pd.Series(np.where(x.notna(), (x > 0).astype(float), np.nan), index=df.index)
    rows = []
    rates = []
    for y_col in (Y_AP, Y_AR):
        lab = train_by_y[y_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        for name, sl in (
            ("cp_eq_0", lab & (x == 0)),
            ("cp_gt_0", lab & (x > 0)),
        ):
            yy = y[sl]
            rec = {
                "y": y_col,
                "slice": name,
                "n": int(yy.notna().sum()),
                "n_pos": int((yy == 1).sum()),
                "rate": float(yy.mean()) if int(yy.notna().sum()) else float("nan"),
            }
            rates.append(rec)
            print(
                f"  {y_col:22s} {name:10s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
                f"rate={rec['rate']:.4f}"
            )
        for col in ("_any_cp", "d_tx_cp_share"):
            oof = signed_oof_auroc(df, col, y_col, lab, folds)
            rec = {
                "y": y_col,
                "col": col,
                "slice": "all_labeled",
                "cv": oof["cv"],
                "sd": oof["sd"],
                "n_defined": int((lab & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
                "folds": [f["auroc"] for f in oof["folds"]],
            }
            rows.append(rec)
            print(
                f"  {y_col:22s} {col:16s} all  CV={oof['cv']:.4f}±{oof['sd']:.3f}"
            )
        pos_lab = lab & (x > 0)
        oof_p = signed_oof_auroc(df, "d_tx_cp_share", y_col, pos_lab, folds)
        rec = {
            "y": y_col,
            "col": "d_tx_cp_share",
            "slice": "cp_gt_0",
            "cv": oof_p["cv"],
            "sd": oof_p["sd"],
            "n_defined": int(pos_lab.sum()),
            "folds": [f["auroc"] for f in oof_p["folds"]],
        }
        rows.append(rec)
        print(
            f"  {y_col:22s} d_tx_cp_share     cp>0 CV={oof_p['cv']:.4f}±{oof_p['sd']:.3f} "
            f"n={int(pos_lab.sum())} pos={int((y[pos_lab] == 1).sum())}"
        )
    ar_bin = next(r for r in rows if r["y"] == Y_AR and r["col"] == "_any_cp")
    ar_all = next(r for r in rows if r["y"] == Y_AR and r["col"] == "d_tx_cp_share" and r["slice"] == "all_labeled")
    ar_pos = next(r for r in rows if r["y"] == Y_AR and r["slice"] == "cp_gt_0")
    gap_bin = (
        float(ar_all["cv"] - ar_bin["cv"])
        if np.isfinite(ar_all["cv"]) and np.isfinite(ar_bin["cv"])
        else float("nan")
    )
    presence_rewrite = bool(np.isfinite(gap_bin) and gap_bin < CLEAR_MARGIN)
    intensity_dead = bool(np.isfinite(ar_pos["cv"]) and ar_pos["cv"] < CHANCE + CLEAR_MARGIN)
    print(
        f"  AR gap continuous vs any-cp {gap_bin:+.3f}  "
        f"presence_rewrite={presence_rewrite} intensity_dead={intensity_dead}"
    )
    return {
        "rows": rows,
        "rates": rates,
        "gap_vs_binary": gap_bin,
        "presence_rewrite": presence_rewrite,
        "intensity_dead": intensity_dead,
        "ar_bin_cv": ar_bin["cv"],
        "ar_all_cv": ar_all["cv"],
        "ar_pos_cv": ar_pos["cv"],
    }


def pass15_group_tail(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """AP Q5 is 2.9% — is 'group size ≥ 18' a protective tail?"""
    print("\n" + "=" * 72)
    print("CUT 15 — h_group_size ≥ 18 protective tail (AP)")
    print("=" * 72)
    x = pd.to_numeric(df["h_group_size"], errors="coerce")
    rows = []
    for y_col in (Y_AP, Y_AR):
        y = pd.to_numeric(df[y_col], errors="coerce")
        m = train_by_y[y_col] & x.notna() & y.notna()
        assert_no_holdout(df.loc[m, "company_id"])
        hi = m & (x >= 18)
        lo = m & (x < 18)
        rec = {
            "y": y_col,
            "n_hi": int(hi.sum()),
            "n_pos_hi": int((y[hi] == 1).sum()),
            "rate_hi": float(y[hi].mean()) if int(hi.sum()) else float("nan"),
            "n_lo": int(lo.sum()),
            "n_pos_lo": int((y[lo] == 1).sum()),
            "rate_lo": float(y[lo].mean()) if int(lo.sum()) else float("nan"),
            "auroc_tail": float(auroc(y[m], (x[m] < 18).astype(float))),
            "n_cos_hi": int(df.loc[hi, "company_id"].nunique()),
        }
        rows.append(rec)
        print(
            f"  {y_col:22s} size>=18 n={rec['n_hi']} pos={rec['n_pos_hi']} "
            f"P(Y=1)={rec['rate_hi']:.3f}  | rest {rec['rate_lo']:.3f}  "
            f"small-group flag AUROC={rec['auroc_tail']:.3f} cos={rec['n_cos_hi']}"
        )
    return {"rows": rows}


def pass16_neither_singles(df: pd.DataFrame, lab: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Do night singles still rank Y inside the 2×2 neither leftover?"""
    print("\n" + "=" * 72)
    print("CUT 16 — night singles on the neither leftover")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    rows = []
    for y_col, hhi_col in ((Y_AP, "d_supp_hhi_hi"), (Y_AR, "d_cust_hhi_hi")):
        keys = lab.loc[
            (lab[y_col].notna())
            & lab["a_io_ratio_lo"].notna()
            & lab[hhi_col].notna()
            & (lab["a_io_ratio_lo"] == 0)
            & (lab[hhi_col] == 0),
            ["company_id", "period"],
        ]
        keyset = set(
            zip(keys["company_id"].astype(str), pd.to_datetime(keys["period"]))
        )
        sl = pd.Series(
            [
                (str(cid), pd.Timestamp(per)) in keyset
                for cid, per in zip(df["company_id"], df["period"])
            ],
            index=df.index,
        )
        sl = sl & train_by_y[y_col]
        y = pd.to_numeric(df.loc[sl, y_col], errors="coerce")
        print(f"  {y_col} neither labeled n={int(sl.sum())} pos={int((y == 1).sum())}")
        cols = ("h_group_size", "d_tx_cp_share", "c_n_days_with_tx")
        for col in cols:
            oof = signed_oof_auroc(df, col, y_col, sl, folds)
            rec = {
                "y": y_col,
                "col": col,
                "n": int(sl.sum()),
                "n_pos": int((y == 1).sum()),
                "cv": oof["cv"],
                "sd": oof["sd"],
            }
            rows.append(rec)
            print(
                f"    {col:22s} CV={oof['cv']:.3f}±{oof['sd']:.3f} "
                f"n={rec['n']} pos={rec['n_pos']}"
            )
    return {"rows": rows}


def pass17_cash_present(lab: pd.DataFrame) -> dict:
    """Join-QA close: leftover positives still have a tx that month."""
    print("\n" + "=" * 72)
    print("CUT 17 — leftover still has cash (CLOSE 'died because cash missing')")
    print("=" * 72)
    rows = []
    days = pd.to_numeric(lab["c_n_days_with_tx"], errors="coerce")
    inn = pd.to_numeric(lab["a_in3"], errors="coerce") if "a_in3" in lab.columns else None
    for y_col, hhi_col in ((Y_AP, "d_supp_hhi_hi"), (Y_AR, "d_cust_hhi_hi")):
        pos = lab[y_col] == 1
        both = pos & lab["a_io_ratio_lo"].notna() & lab[hhi_col].notna()
        neither = both & (lab["a_io_ratio_lo"] == 0) & (lab[hhi_col] == 0)
        for name, sl in (("all_pos", pos), ("neither_pos", neither)):
            rec = {
                "y": y_col,
                "slice": name,
                "n": int(sl.sum()),
                "share_days_gt0": float((days[sl] > 0).mean()) if int(sl.sum()) else float("nan"),
                "share_in3_def": (
                    float(inn[sl].notna().mean()) if inn is not None and int(sl.sum()) else float("nan")
                ),
                "med_days": float(days[sl].median()) if int(sl.sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"  {y_col:22s} {name:12s} n={rec['n']:4d}  "
                f"days>0={rec['share_days_gt0']:.3f}  in3_def={rec['share_in3_def']:.3f}  "
                f"med_days={rec['med_days']:.1f}"
            )
    return {"rows": rows}


def pass18_tagging_vs_invoice(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is low d_tx_cp_share just few invoice customers, or a bank-tag hole?"""
    print("\n" + "=" * 72)
    print("CUT 18 — d_tx_cp_share × d_n_cust (tagging hole vs invoice thinness)")
    print("Never E. d_n_cust is D, invoice-gated.")
    print("=" * 72)
    if "d_n_cust" not in df.columns:
        print("  missing d_n_cust")
        return {"rows": []}
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    m = lab & share.notna() & nc.notna() & y.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    med_n = float(nc[m].median())
    low_share = share <= 0.01
    hi_cust = nc > med_n
    slices = {
        "low_share_hi_cust": m & low_share & hi_cust,
        "low_share_lo_cust": m & low_share & ~hi_cust,
        "hi_share_hi_cust": m & ~low_share & hi_cust,
        "hi_share_lo_cust": m & ~low_share & ~hi_cust,
    }
    rows = []
    print(f"  train AR labeled complete n={int(m.sum())} med d_n_cust={med_n:.3g}")
    for name, sl in slices.items():
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"    {name:22s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.3f}"
        )
    hole = next(r for r in rows if r["slice"] == "low_share_hi_cust")
    thin = next(r for r in rows if r["slice"] == "low_share_lo_cust")
    named = next(r for r in rows if r["slice"] == "hi_share_hi_cust")
    tagging_hole = bool(
        np.isfinite(hole["rate"])
        and np.isfinite(named["rate"])
        and hole["rate"] >= named["rate"] + 0.04
    )
    print(
        f"  tagging_hole={tagging_hole}  "
        f"low-share+hi-cust {hole['rate']:.3f} vs named+hi-cust {named['rate']:.3f} "
        f"vs low-share+lo-cust {thin['rate']:.3f}"
    )

    # AP analogue: many suppliers × unnamed bank trail.
    ap_rows = []
    tagging_hole_ap = False
    hole_ap = thin_ap = named_ap = float("nan")
    med_s = float("nan")
    if "d_n_supp" in df.columns:
        lab_ap = train_by_y[Y_AP]
        ns = pd.to_numeric(df["d_n_supp"], errors="coerce")
        ya = pd.to_numeric(df[Y_AP], errors="coerce")
        ma = lab_ap & share.notna() & ns.notna() & ya.notna()
        assert_no_holdout(df.loc[ma, "company_id"])
        med_s = float(ns[ma].median())
        low_s = share <= 0.01
        hi_s = ns > med_s
        ap_slices = {
            "low_share_hi_supp": ma & low_s & hi_s,
            "low_share_lo_supp": ma & low_s & ~hi_s,
            "hi_share_hi_supp": ma & ~low_s & hi_s,
            "hi_share_lo_supp": ma & ~low_s & ~hi_s,
        }
        print(f"  train AP labeled complete n={int(ma.sum())} med d_n_supp={med_s:.3g}")
        for name, sl in ap_slices.items():
            rec = {
                "y": Y_AP,
                "slice": name,
                "n": int(sl.sum()),
                "n_pos": int((ya[sl] == 1).sum()),
                "rate": float(ya[sl].mean()) if int(sl.sum()) else float("nan"),
            }
            ap_rows.append(rec)
            print(
                f"    {name:22s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.3f}"
            )
        hole_ap = next(r["rate"] for r in ap_rows if r["slice"] == "low_share_hi_supp")
        thin_ap = next(r["rate"] for r in ap_rows if r["slice"] == "low_share_lo_supp")
        named_ap = next(r["rate"] for r in ap_rows if r["slice"] == "hi_share_hi_supp")
        tagging_hole_ap = bool(
            np.isfinite(hole_ap) and np.isfinite(named_ap) and hole_ap >= named_ap + 0.04
        )
        print(
            f"  AP tagging_hole={tagging_hole_ap}  "
            f"low-share+hi-supp {hole_ap:.3f} vs named+hi-supp {named_ap:.3f}"
        )
    return {
        "rows": rows,
        "med_n_cust": med_n,
        "tagging_hole": tagging_hole,
        "hole_rate": hole["rate"],
        "named_rate": named["rate"],
        "thin_rate": thin["rate"],
        "ap_rows": ap_rows,
        "med_n_supp": med_s,
        "tagging_hole_ap": tagging_hole_ap,
        "hole_rate_ap": hole_ap,
        "named_rate_ap": named_ap,
        "thin_rate_ap": thin_ap,
    }


def pass19_hole_vs_cash(df: pd.DataFrame, train_by_y: dict[str, pd.Series], p18: dict) -> dict:
    """Is the AR tagging hole just own-low cash? Never E."""
    print("\n" + "=" * 72)
    print("CUT 19 — AR tagging hole × own-low a_io_ratio (cash rewrite?)")
    print("Never E. Cash flag is company expanding p20.")
    print("=" * 72)
    empty = {
        "rows": [],
        "cash_independent": False,
        "hole_ok_rate": float("nan"),
        "hole_cash_rate": float("nan"),
        "named_ok_rate": float("nan"),
        "named_cash_rate": float("nan"),
    }
    if "d_n_cust" not in df.columns or "a_io_ratio_lo" not in df.columns:
        print("  missing d_n_cust or a_io_ratio_lo")
        return empty
    med_n = p18.get("med_n_cust")
    if med_n is None or not np.isfinite(med_n):
        print("  no train median d_n_cust")
        return empty
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    cash = pd.to_numeric(df["a_io_ratio_lo"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    hi_cust = nc > float(med_n)
    hole = (share <= 0.01) & hi_cust
    base = lab & share.notna() & nc.notna() & y.notna() & cash.notna() & hi_cust
    assert_no_holdout(df.loc[base, "company_id"])
    slices = {
        "hole_cash": base & hole & (cash == 1),
        "hole_ok": base & hole & (cash == 0),
        "named_cash": base & ~hole & (cash == 1),
        "named_ok": base & ~hole & (cash == 0),
    }
    rows = []
    print(f"  train AR hi-cust + cash-defined n={int(base.sum())} med d_n_cust={med_n:.3g}")
    rates = {}
    for name, sl in slices.items():
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        rates[name] = rec["rate"]
        print(
            f"    {name:12s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.3f}"
        )
    hole_ok = rates.get("hole_ok", float("nan"))
    named_ok = rates.get("named_ok", float("nan"))
    cash_independent = bool(
        np.isfinite(hole_ok) and np.isfinite(named_ok) and hole_ok >= named_ok + 0.04
    )
    print(
        f"  cash_independent={cash_independent}  "
        f"hole+ok-cash {hole_ok:.3f} vs named+ok-cash {named_ok:.3f}"
    )
    return {
        "rows": rows,
        "cash_independent": cash_independent,
        "hole_ok_rate": hole_ok,
        "hole_cash_rate": rates.get("hole_cash", float("nan")),
        "named_ok_rate": named_ok,
        "named_cash_rate": rates.get("named_cash", float("nan")),
    }


def pass20_hole_trait_holdout(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    is_hold: pd.Series,
    p18: dict,
) -> dict:
    """Company ever-hole vs never; hole-flag CV on hi-cust; holdout n only."""
    print("\n" + "=" * 72)
    print("CUT 20 — tagging hole as company trait + holdout coverage")
    print("Holdout is n_pos only. Never quote holdout AUROC.")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    out = {
        "cv": float("nan"),
        "sd": float("nan"),
        "n_hi_cust": 0,
        "n_pos_hi_cust": 0,
        "n_cos_hi_cust": 0,
        "n_cos_ever_hole": 0,
        "rate_ever_hole": float("nan"),
        "rate_never_hole": float("nan"),
        "ho_n": 0,
        "ho_n_pos": 0,
        "ho_n_hole": 0,
        "ho_n_hole_pos": 0,
    }
    if med_n is None or not np.isfinite(med_n) or "d_n_cust" not in df.columns:
        print("  skip")
        return out
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    hi_cust = nc > float(med_n)
    hole = (share <= 0.01) & hi_cust
    work = df.copy()
    work["_hole"] = pd.Series(
        np.where(share.notna() & nc.notna(), hole.astype(float), np.nan),
        index=df.index,
    )
    lab = train_by_y[Y_AR]
    hi_lab = lab & share.notna() & nc.notna() & y.notna() & hi_cust
    assert_no_holdout(df.loc[hi_lab, "company_id"])
    out["n_hi_cust"] = int(hi_lab.sum())
    out["n_pos_hi_cust"] = int((y[hi_lab] == 1).sum())
    folds = work["fold"].to_numpy()
    oof = signed_oof_auroc(work, "_hole", Y_AR, hi_lab, folds)
    out["cv"] = oof["cv"]
    out["sd"] = oof["sd"]
    print(
        f"  hole-flag CV on hi-cust train: {oof['cv']:.3f}±{oof['sd']:.3f} "
        f"n={out['n_hi_cust']} pos={out['n_pos_hi_cust']} sign={oof['train_sign']}"
    )

    sub = pd.DataFrame(
        {
            "company_id": df.loc[hi_lab, "company_id"].astype(str),
            "y": y[hi_lab],
            "hole": work.loc[hi_lab, "_hole"],
        }
    )
    g = sub.groupby("company_id", sort=False).agg(
        n=("y", "size"),
        n_pos=("y", "sum"),
        n_hole=("hole", "sum"),
        y_rate=("y", "mean"),
    )
    ever = g["n_hole"] > 0
    out["n_cos_hi_cust"] = int(len(g))
    out["n_cos_ever_hole"] = int(ever.sum())
    out["rate_ever_hole"] = float(g.loc[ever, "y_rate"].mean()) if int(ever.sum()) else float("nan")
    out["rate_never_hole"] = (
        float(g.loc[~ever, "y_rate"].mean()) if int((~ever).sum()) else float("nan")
    )
    print(
        f"  companies with a hi-cust AR month: {out['n_cos_hi_cust']} "
        f"ever-hole {out['n_cos_ever_hole']} "
        f"mean company-rate ever {out['rate_ever_hole']:.3f} "
        f"never {out['rate_never_hole']:.3f}"
    )

    ho = is_hold & y.notna()
    ho_hole = ho & share.notna() & nc.notna() & hole
    out["ho_n"] = int(ho.sum())
    out["ho_n_pos"] = int((y[ho] == 1).sum())
    out["ho_n_hole"] = int(ho_hole.sum())
    out["ho_n_hole_pos"] = int((y[ho_hole] == 1).sum())
    print(
        f"  holdout AR labeled n={out['ho_n']} pos={out['ho_n_pos']} "
        f"hole n={out['ho_n_hole']} pos={out['ho_n_hole_pos']} LOW_POWER"
    )
    return out


def pass21_hole_lag1(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Honest 1-month Q6: last month's tagging hole. Never t3. Never E."""
    print("\n" + "=" * 72)
    print("CUT 21 — hole_lag1 as 1-month Q6 (short vs long books)")
    print("Only lag-1 is transferable. Never claim a quarter.")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    rows = []
    out = {"rows": rows, "short_cv": float("nan"), "long_cv": float("nan"), "all_cv": float("nan")}
    if med_n is None or not np.isfinite(med_n) or "d_n_cust" not in df.columns:
        print("  skip")
        return out
    work = df.sort_values(["company_id", "period"]).copy()
    share = pd.to_numeric(work["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(work["d_n_cust"], errors="coerce")
    hole = pd.Series(
        np.where(share.notna() & nc.notna(), ((share <= 0.01) & (nc > float(med_n))).astype(float), np.nan),
        index=work.index,
    )
    work["_hole"] = hole
    work["_hole_lag1"] = work.groupby("company_id", sort=False)["_hole"].shift(1)
    first = work.groupby("company_id", sort=False)["period"].transform("min")
    work["_so_far"] = (
        (work["period"].dt.year - first.dt.year) * 12
        + (work["period"].dt.month - first.dt.month)
        + 1
    )
    aligned = df.copy()
    key = df[["company_id", "period"]].copy()
    mapped = work.set_index(["company_id", "period"])[["_hole", "_hole_lag1", "_so_far"]]
    joined = key.merge(mapped, left_on=["company_id", "period"], right_index=True, how="left")
    aligned["_hole"] = joined["_hole"].to_numpy()
    aligned["_hole_lag1"] = joined["_hole_lag1"].to_numpy()
    aligned["_so_far"] = joined["_so_far"].to_numpy()

    folds = aligned["fold"].to_numpy()
    lab = train_by_y[Y_AR]
    short = lab & (aligned["_so_far"] < 12)
    long_ = lab & (aligned["_so_far"] >= 18)
    for sl_name, sl in (("short_<12", short), ("long_>=18", long_), ("all", lab)):
        assert_no_holdout(aligned.loc[sl, "company_id"])
        n = int(sl.sum())
        n_pos = int((pd.to_numeric(aligned.loc[sl, Y_AR], errors="coerce") == 1).sum())
        print(f"  {Y_AR} {sl_name}: n={n} pos={n_pos}")
        for col, lag in (("_hole", 0), ("_hole_lag1", 1)):
            cov = coverage(pd.to_numeric(aligned[col], errors="coerce"), sl)
            oof = signed_oof_auroc(aligned, col, Y_AR, sl, folds)
            rec = {
                "y": Y_AR,
                "slice": sl_name,
                "col": col,
                "lag": lag,
                "n": n,
                "n_pos": n_pos,
                "cv": oof["cv"],
                "sd": oof["sd"],
                "coverage": cov["coverage"],
                "n_defined": cov["n_defined"],
            }
            rows.append(rec)
            print(
                f"    {col:14s} cov={cov['coverage']:.3f} "
                f"CV={oof['cv']:.3f}±{oof['sd']:.3f} n_def={cov['n_defined']}"
            )
            if sl_name == "short_<12" and lag == 1:
                out["short_cv"] = oof["cv"]
            if sl_name == "long_>=18" and lag == 1:
                out["long_cv"] = oof["cv"]
            if sl_name == "all" and lag == 1:
                out["all_cv"] = oof["cv"]
    return out


def pass22_hole_concentration(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Are hole positives 3 companies or a panel pattern?"""
    print("\n" + "=" * 72)
    print("CUT 22 — hole-positive concentration across companies")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    out = {
        "n_ar_pos": 0,
        "n_hole_pos": 0,
        "share_of_ar_pos": float("nan"),
        "n_cos_hole_pos": 0,
        "top1": float("nan"),
        "top3": float("nan"),
        "hhi": float("nan"),
        "fragile": None,
    }
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return out
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    hole = (share <= 0.01) & (nc > float(med_n))
    pos = lab & (y == 1)
    hole_pos = pos & share.notna() & nc.notna() & hole
    assert_no_holdout(df.loc[pos, "company_id"])
    n_pos = int(pos.sum())
    n_hp = int(hole_pos.sum())
    out["n_ar_pos"] = n_pos
    out["n_hole_pos"] = n_hp
    out["share_of_ar_pos"] = (n_hp / n_pos) if n_pos else float("nan")
    g = (
        df.loc[hole_pos, "company_id"]
        .astype(str)
        .value_counts()
    )
    out["n_cos_hole_pos"] = int(len(g))
    if n_hp:
        shares = g.to_numpy(dtype=float) / n_hp
        out["top1"] = float(shares[0]) if len(shares) else float("nan")
        out["top3"] = float(shares[:3].sum()) if len(shares) else float("nan")
        out["hhi"] = float((shares ** 2).sum())
    out["fragile"] = bool(np.isfinite(out["top3"]) and out["top3"] >= 0.50)
    print(
        f"  hole pos {n_hp}/{n_pos} = {out['share_of_ar_pos']:.3f} of AR pos; "
        f"n_cos={out['n_cos_hole_pos']} top1={out['top1']:.3f} top3={out['top3']:.3f} "
        f"hhi={out['hhi']:.3f} fragile={out['fragile']}"
    )
    return out


def pass23_intensity_hi_cust(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Continuous d_tx_cp_share on thick vs thin invoice books."""
    print("\n" + "=" * 72)
    print("CUT 23 — d_tx_cp_share intensity on hi-cust vs lo-cust")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    rows = []
    out = {"rows": rows, "hi_cv": float("nan"), "lo_cv": float("nan")}
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return out
    lab = train_by_y[Y_AR]
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    folds = df["fold"].to_numpy()
    for name, sl in (
        ("hi_cust", lab & nc.notna() & y.notna() & (nc > float(med_n))),
        ("lo_cust", lab & nc.notna() & y.notna() & (nc <= float(med_n))),
    ):
        assert_no_holdout(df.loc[sl, "company_id"])
        oof = signed_oof_auroc(df, "d_tx_cp_share", Y_AR, sl, folds)
        rec = {
            "slice": name,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "cv": oof["cv"],
            "sd": oof["sd"],
            "sign": oof["train_sign"],
        }
        rows.append(rec)
        print(
            f"  {name:8s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"CV={oof['cv']:.3f}±{oof['sd']:.3f} sign={oof['train_sign']}"
        )
        if name == "hi_cust":
            out["hi_cv"] = oof["cv"]
        else:
            out["lo_cv"] = oof["cv"]
    return out


def pass24_txcp_diff1(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Turning-month: first difference of d_tx_cp_share. 1m only."""
    print("\n" + "=" * 72)
    print("CUT 24 — Δ d_tx_cp_share (lag0 − lag1) as turning-month")
    print("=" * 72)
    if "d_tx_cp_share_lag1" not in df.columns:
        print("  missing lag1")
        return {"rows": [], "cv": float("nan")}
    work = df.copy()
    x0 = pd.to_numeric(work["d_tx_cp_share"], errors="coerce")
    x1 = pd.to_numeric(work["d_tx_cp_share_lag1"], errors="coerce")
    work["_dtx_diff1"] = x0 - x1
    folds = work["fold"].to_numpy()
    per = work.sort_values(["company_id", "period"])
    first = per.groupby("company_id")["period"].transform("min")
    so_far = (
        (per["period"].dt.year - first.dt.year) * 12
        + (per["period"].dt.month - first.dt.month)
        + 1
    )
    work["_so_far"] = so_far.reindex(work.index)
    rows = []
    lab = train_by_y[Y_AR]
    short = lab & (work["_so_far"] < 12)
    out_cv = float("nan")
    for sl_name, sl in (("all", lab), ("short_<12", short)):
        assert_no_holdout(work.loc[sl, "company_id"])
        oof = signed_oof_auroc(work, "_dtx_diff1", Y_AR, sl, folds)
        y = pd.to_numeric(work.loc[sl, Y_AR], errors="coerce")
        cov = coverage(work["_dtx_diff1"], sl)
        rec = {
            "slice": sl_name,
            "n": int(sl.sum()),
            "n_pos": int((y == 1).sum()),
            "cv": oof["cv"],
            "sd": oof["sd"],
            "sign": oof["train_sign"],
            "coverage": cov["coverage"],
        }
        rows.append(rec)
        print(
            f"  {sl_name:10s} CV={oof['cv']:.3f}±{oof['sd']:.3f} "
            f"sign={oof['train_sign']} cov={cov['coverage']:.3f} "
            f"n={rec['n']} pos={rec['n_pos']}"
        )
        if sl_name == "all":
            out_cv = oof["cv"]
    return {"rows": rows, "cv": out_cv}


def _so_far(df: pd.DataFrame) -> pd.Series:
    per = df.sort_values(["company_id", "period"])
    first = per.groupby("company_id")["period"].transform("min")
    so = (
        (per["period"].dt.year - first.dt.year) * 12
        + (per["period"].dt.month - first.dt.month)
        + 1
    )
    return so.reindex(df.index)


def pass25_short_hole_rate(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Contemporaneous hole rate on so-far<12 thick books. Q5 vs Q6."""
    print("\n" + "=" * 72)
    print("CUT 25 — short-book contemporaneous hole rate (Q5 ≠ Q6)")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    empty = {"rows": [], "short_hole_rate": float("nan"), "short_named_rate": float("nan")}
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return empty
    work = df.copy()
    work["_so_far"] = _so_far(work)
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(work["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(work["d_n_cust"], errors="coerce")
    y = pd.to_numeric(work[Y_AR], errors="coerce")
    hi = nc > float(med_n)
    hole = (share <= 0.01) & hi
    short = lab & (work["_so_far"] < 12) & share.notna() & nc.notna() & y.notna()
    long_ = lab & (work["_so_far"] >= 18) & share.notna() & nc.notna() & y.notna()
    rows = []
    rates = {}
    for book, sl in (("short_<12", short), ("long_>=18", long_)):
        for name, m in (
            (f"{book}_hi_hole", sl & hole),
            (f"{book}_hi_named", sl & hi & ~hole),
        ):
            rec = {
                "slice": name,
                "n": int(m.sum()),
                "n_pos": int((y[m] == 1).sum()),
                "rate": float(y[m].mean()) if int(m.sum()) else float("nan"),
            }
            rows.append(rec)
            rates[name] = rec["rate"]
            print(
                f"    {name:22s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.3f}"
            )
    print(
        f"  short hi hole {rates.get('short_<12_hi_hole', float('nan')):.3f} vs "
        f"named {rates.get('short_<12_hi_named', float('nan')):.3f}"
    )
    return {
        "rows": rows,
        "short_hole_rate": rates.get("short_<12_hi_hole", float("nan")),
        "short_named_rate": rates.get("short_<12_hi_named", float("nan")),
        "long_hole_rate": rates.get("long_>=18_hi_hole", float("nan")),
        "long_named_rate": rates.get("long_>=18_hi_named", float("nan")),
    }


def pass26_ap_leftover_d(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
) -> dict:
    """AP leftover: unused D columns as singles. Never E. Size-PARK if ≥0.60."""
    print("\n" + "=" * 72)
    print("CUT 26 — AP leftover extra D singles (d_n_supp / d_supp_top1)")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    lab = train_by_y[Y_AP]
    size = signed_oof_auroc(df, "log1p_a_in3", Y_AP, lab, folds)
    rows = []
    for col in ("d_n_supp", "d_supp_top1", "d_supp_hhi"):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, Y_AP, lab, folds)
        size_bin = _size_auroc_of_feature(df, col, lab)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "sign": oof["train_sign"],
            "size_auroc": size_bin["auroc"],
            "size_park": bool(
                np.isfinite(size_bin["auroc"]) and size_bin["auroc"] >= SIZE_AUROC
            ),
            "gap_vs_size": float(oof["cv"] - size["cv"]) if np.isfinite(oof["cv"]) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {col:16s} CV={oof['cv']:.3f}±{oof['sd']:.3f} sign={oof['train_sign']} "
            f"sizeAUC={rec['size_auroc']:.3f} park={rec['size_park']} gap={rec['gap_vs_size']:+.3f}"
        )
    return {"rows": rows, "size_cv": size["cv"]}


def pass27_txcp_vs_size(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is d_tx_cp_share a size rewrite? Spearman vs in3 / group / n_cust."""
    print("\n" + "=" * 72)
    print("CUT 27 — d_tx_cp_share vs size (train AR labeled)")
    print("=" * 72)
    lab = train_by_y[Y_AR]
    assert_no_holdout(df.loc[lab, "company_id"])
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    rows = []
    for col, series in (
        ("log1p_a_in3", pd.to_numeric(df["log1p_a_in3"], errors="coerce")),
        ("h_group_size", pd.to_numeric(df["h_group_size"], errors="coerce")),
        ("d_n_cust", pd.to_numeric(df["d_n_cust"], errors="coerce")),
        ("c_n_days_with_tx", pd.to_numeric(df["c_n_days_with_tx"], errors="coerce")),
    ):
        rho = spearman(share[lab], series[lab])
        rec = {"col": col, "rho": rho}
        rows.append(rec)
        print(f"  ρ(d_tx_cp_share, {col:18s}) = {rho:+.3f}")
    return {"rows": rows}


def pass28_hole_vs_e_and_y(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Hole flag vs E (leak) and vs Y2/Y9 (other accepted labels). Never use E as X."""
    print("\n" + "=" * 72)
    print("CUT 28 — hole flag vs E (leak) and Y2/Y9")
    print("E is comparator only. Fail |ρ|≥0.80.")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    rows = []
    out = {"rows": rows, "leak_fail": False, "max_abs_rho_e": float("nan")}
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return out
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    hole = pd.Series(
        np.where(share.notna() & nc.notna(), ((share <= 0.01) & (nc > float(med_n))).astype(float), np.nan),
        index=df.index,
    )
    m = lab & hole.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    rhos = []
    for col in LEAK_VS:
        if col not in df.columns:
            continue
        rho = spearman(hole[m], pd.to_numeric(df.loc[m, col], errors="coerce"))
        fail = bool(np.isfinite(rho) and abs(rho) >= LEAK_RHO)
        rec = {"kind": "E", "col": col, "rho": rho, "fail": fail}
        rows.append(rec)
        if np.isfinite(rho):
            rhos.append(abs(rho))
        print(f"  hole vs {col:22s} ρ={rho:+.3f} fail={fail}")
    for col in (Y2, Y9, Y_AP):
        if col not in df.columns:
            continue
        rho = spearman(hole[m], pd.to_numeric(df.loc[m, col], errors="coerce"))
        rec = {"kind": "Y", "col": col, "rho": rho, "fail": False}
        rows.append(rec)
        print(f"  hole vs {col:22s} ρ={rho:+.3f}")
    out["max_abs_rho_e"] = max(rhos) if rhos else float("nan")
    out["leak_fail"] = bool(any(r["fail"] for r in rows if r["kind"] == "E"))
    print(f"  leak_fail={out['leak_fail']} max|ρ|_E={out['max_abs_rho_e']:.3f}")
    return out


def pass29_hi_cust_quintiles(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Head-only on thick books? Same Q5 shape restricted to hi-cust."""
    print("\n" + "=" * 72)
    print("CUT 29 — d_tx_cp_share quintiles on hi-cust AR only")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return {"table": None}
    lab = train_by_y[Y_AR]
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    hi = lab & nc.notna() & (nc > float(med_n))
    tab = quintile_table(df, hi, "d_tx_cp_share", Y_AR)
    print(
        f"  n={tab['n']} bins={tab['n_bins']} "
        f"up={tab['monotone_up']} down={tab['monotone_down']} "
        f"tail={tab['tail_only']} head={tab['head_only']}"
    )
    for r in tab.get("rows") or []:
        print(
            f"    Q{r['q']} n={r['n']:4d} pos={r['n_pos']:3d} "
            f"rate={r['y_rate']:.3f} medX={r['x_median']:.4g}"
        )
    return {"table": tab}


def pass30_hole_by_fold(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Is the 17.4% hole rate stable across group folds?"""
    print("\n" + "=" * 72)
    print("CUT 30 — hole rate by group fold (train)")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    rows = []
    out = {"rows": rows, "min_rate": float("nan"), "max_rate": float("nan")}
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return out
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    hole = (share <= 0.01) & (nc > float(med_n))
    named = (share > 0.01) & (nc > float(med_n))
    folds = pd.to_numeric(df["fold"], errors="coerce")
    rates = []
    for k in range(N_FOLDS):
        sl = lab & (folds == k) & share.notna() & nc.notna() & y.notna()
        assert_no_holdout(df.loc[sl, "company_id"])
        h = sl & hole
        nmd = sl & named
        rec = {
            "fold": k,
            "n_hole": int(h.sum()),
            "n_pos_hole": int((y[h] == 1).sum()),
            "rate_hole": float(y[h].mean()) if int(h.sum()) else float("nan"),
            "n_named": int(nmd.sum()),
            "rate_named": float(y[nmd].mean()) if int(nmd.sum()) else float("nan"),
        }
        rows.append(rec)
        if np.isfinite(rec["rate_hole"]):
            rates.append(rec["rate_hole"])
        print(
            f"  fold {k} hole n={rec['n_hole']:4d} pos={rec['n_pos_hole']:3d} "
            f"rate={rec['rate_hole']:.3f}  named rate={rec['rate_named']:.3f}"
        )
    out["min_rate"] = min(rates) if rates else float("nan")
    out["max_rate"] = max(rates) if rates else float("nan")
    print(f"  hole-rate range {out['min_rate']:.3f}–{out['max_rate']:.3f}")
    return out


def pass31_drop_fold3(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
    p30: dict,
) -> dict:
    """Does the hole / night single survive if the heavy fold is dropped?"""
    print("\n" + "=" * 72)
    print("CUT 31 — drop the fold that owns hole positives")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    rows30 = p30.get("rows") or []
    heavy = None
    if rows30:
        heavy = max(rows30, key=lambda r: r.get("n_pos_hole") or 0)["fold"]
    out = {
        "heavy_fold": heavy,
        "hole_rate_wo": float("nan"),
        "named_rate_wo": float("nan"),
        "cv_wo": float("nan"),
        "sd_wo": float("nan"),
        "n_pos_heavy": 0,
        "share_pos_heavy": float("nan"),
        "survives": None,
    }
    if heavy is None or med_n is None or not np.isfinite(med_n):
        print("  skip")
        return out
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    folds = pd.to_numeric(df["fold"], errors="coerce")
    hole = (share <= 0.01) & (nc > float(med_n))
    named = (share > 0.01) & (nc > float(med_n))
    base = lab & share.notna() & nc.notna() & y.notna()
    wo = base & (folds != heavy)
    assert_no_holdout(df.loc[wo, "company_id"])
    h = wo & hole
    nmd = wo & named
    heavy_pos = int((base & hole & (folds == heavy) & (y == 1)).sum())
    all_hole_pos = int((base & hole & (y == 1)).sum())
    out["n_pos_heavy"] = heavy_pos
    out["share_pos_heavy"] = (heavy_pos / all_hole_pos) if all_hole_pos else float("nan")
    out["hole_rate_wo"] = float(y[h].mean()) if int(h.sum()) else float("nan")
    out["named_rate_wo"] = float(y[nmd].mean()) if int(nmd.sum()) else float("nan")
    oof = signed_oof_auroc(df, "d_tx_cp_share", Y_AR, wo, df["fold"].to_numpy())
    out["cv_wo"] = oof["cv"]
    out["sd_wo"] = oof["sd"]
    out["survives"] = bool(
        np.isfinite(out["hole_rate_wo"])
        and np.isfinite(out["named_rate_wo"])
        and out["hole_rate_wo"] >= out["named_rate_wo"] + 0.04
        and np.isfinite(out["cv_wo"])
        and out["cv_wo"] >= CHANCE + CLEAR_MARGIN
    )
    print(
        f"  heavy fold={heavy} owns {heavy_pos}/{all_hole_pos} hole pos "
        f"({out['share_pos_heavy']:.3f})"
    )
    print(
        f"  without fold {heavy}: hole {out['hole_rate_wo']:.3f} "
        f"(n={int(h.sum())} pos={int((y[h]==1).sum())}) vs named {out['named_rate_wo']:.3f}; "
        f"d_tx_cp_share CV {out['cv_wo']:.3f}±{out['sd_wo']:.3f} survives={out['survives']}"
    )
    return out


def pass32_heavy_fold_who(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
    p31: dict,
) -> dict:
    """Who sits in the fold that owns the hole? Never sibling_h.py."""
    print("\n" + "=" * 72)
    print("CUT 32 — companies in the heavy hole fold (describe only)")
    print("=" * 72)
    heavy = p31.get("heavy_fold")
    out = {
        "n_cos": 0,
        "med_group": float("nan"),
        "med_share": float("nan"),
        "med_ncust": float("nan"),
        "share_hole_months": float("nan"),
    }
    if heavy is None:
        print("  skip")
        return out
    lab = train_by_y[Y_AR]
    folds = pd.to_numeric(df["fold"], errors="coerce")
    sl = lab & (folds == heavy)
    assert_no_holdout(df.loc[sl, "company_id"])
    cos = df.loc[sl, "company_id"].astype(str).unique()
    out["n_cos"] = int(len(cos))
    out["med_group"] = float(pd.to_numeric(df.loc[sl, "h_group_size"], errors="coerce").median())
    out["med_share"] = float(pd.to_numeric(df.loc[sl, "d_tx_cp_share"], errors="coerce").median())
    out["med_ncust"] = float(pd.to_numeric(df.loc[sl, "d_n_cust"], errors="coerce").median())
    share = pd.to_numeric(df.loc[sl, "d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df.loc[sl, "d_n_cust"], errors="coerce")
    med_n = p18.get("med_n_cust")
    if med_n is not None and np.isfinite(med_n):
        hole = (share <= 0.01) & (nc > float(med_n))
        out["share_hole_months"] = float(hole.mean()) if int(hole.notna().sum()) else float("nan")
    else:
        out["share_hole_months"] = float("nan")
    print(
        f"  fold {heavy}: n_cos={out['n_cos']} med group={out['med_group']:.3g} "
        f"med d_tx_cp_share={out['med_share']:.3g} med d_n_cust={out['med_ncust']:.3g} "
            f"share_low_cp_hi_cust_months={out['share_hole_months']:.3f}"
    )
    return out


def pass33_share_by_fold(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is fold 3 just the unnamed-cp cluster (treatment group), not a fluke?"""
    print("\n" + "=" * 72)
    print("CUT 33 — median d_tx_cp_share by fold (AR train)")
    print("=" * 72)
    lab = train_by_y[Y_AR]
    assert_no_holdout(df.loc[lab, "company_id"])
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    folds = pd.to_numeric(df["fold"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    rows = []
    for k in range(N_FOLDS):
        sl = lab & (folds == k) & share.notna()
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "n_cos": int(df.loc[sl, "company_id"].astype(str).nunique()),
            "med_share": float(share[sl].median()) if int(sl.sum()) else float("nan"),
            "share_eq0": float((share[sl] == 0).mean()) if int(sl.sum()) else float("nan"),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  fold {k}: n={rec['n']:4d} cos={rec['n_cos']:3d} "
            f"med_share={rec['med_share']:.3f} share=0 {rec['share_eq0']:.3f} "
            f"P(Y=1)={rec['rate']:.3f}"
        )
    return {"rows": rows}


def pass34_zero_by_fold_book(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """Fold 2 has many zeros — thin book or same hole?"""
    print("\n" + "=" * 72)
    print("CUT 34 — share=0 rate by fold × hi/lo d_n_cust")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    rows = []
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return {"rows": rows}
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    folds = pd.to_numeric(df["fold"], errors="coerce")
    hi = nc > float(med_n)
    for k in range(N_FOLDS):
        for name, book in (("hi", hi), ("lo", ~hi)):
            sl = lab & (folds == k) & (share == 0) & nc.notna() & y.notna() & book
            rec = {
                "fold": k,
                "book": name,
                "n": int(sl.sum()),
                "n_pos": int((y[sl] == 1).sum()),
                "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"  fold {k} zero+{name} n={rec['n']:4d} pos={rec['n_pos']:3d} "
                f"rate={rec['rate']:.3f}"
            )
    return {"rows": rows}


def pass35_ap_rate_by_fold(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is AP leftover a one-fold spike too?"""
    print("\n" + "=" * 72)
    print("CUT 35 — AP leftover rate by fold (uniform?)")
    print("=" * 72)
    lab = train_by_y[Y_AP]
    assert_no_holdout(df.loc[lab, "company_id"])
    y = pd.to_numeric(df[Y_AP], errors="coerce")
    folds = pd.to_numeric(df["fold"], errors="coerce")
    rows = []
    rates = []
    for k in range(N_FOLDS):
        sl = lab & (folds == k) & y.notna()
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "n_pos": int((y[sl] == 1).sum()),
            "n_cos": int(df.loc[sl, "company_id"].astype(str).nunique()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        if np.isfinite(rec["rate"]):
            rates.append(rec["rate"])
        print(
            f"  fold {k}: n={rec['n']:4d} pos={rec['n_pos']:3d} "
            f"cos={rec['n_cos']:3d} rate={rec['rate']:.3f}"
        )
    out = {
        "rows": rows,
        "min_rate": min(rates) if rates else float("nan"),
        "max_rate": max(rates) if rates else float("nan"),
    }
    print(f"  AP rate range {out['min_rate']:.3f}–{out['max_rate']:.3f}")
    return out


def pass36_ap_neither_by_fold(lab: pd.DataFrame) -> dict:
    """Share of AP positives that are leftover (neither cash nor HHI) by fold."""
    print("\n" + "=" * 72)
    print("CUT 36 — AP neither-leftover share of positives by fold")
    print("=" * 72)
    hhi = "d_supp_hhi_hi"
    rows = []
    for k in range(N_FOLDS):
        sl = (lab["fold"] == k) & (lab[Y_AP] == 1)
        both = sl & lab["a_io_ratio_lo"].notna() & lab[hhi].notna()
        neither = both & (lab["a_io_ratio_lo"] == 0) & (lab[hhi] == 0)
        rec = {
            "fold": k,
            "n_pos": int(sl.sum()),
            "n_both_def": int(both.sum()),
            "n_neither": int(neither.sum()),
            "share_neither": (
                float(neither.sum() / both.sum()) if int(both.sum()) else float("nan")
            ),
        }
        rows.append(rec)
        print(
            f"  fold {k}: pos={rec['n_pos']:3d} neither={rec['n_neither']:3d}/"
            f"{rec['n_both_def']} share={rec['share_neither']:.3f}"
        )
        assert_no_holdout(lab.loc[sl, "company_id"])
    return {"rows": rows}


def pass37_ar_neither_by_fold(lab: pd.DataFrame) -> dict:
    """AR leftover share of positives by fold — is fold 3 all hole, not leftover?"""
    print("\n" + "=" * 72)
    print("CUT 37 — AR neither-leftover share of positives by fold")
    print("=" * 72)
    hhi = "d_cust_hhi_hi"
    rows = []
    for k in range(N_FOLDS):
        sl = (lab["fold"] == k) & (lab[Y_AR] == 1)
        both = sl & lab["a_io_ratio_lo"].notna() & lab[hhi].notna()
        neither = both & (lab["a_io_ratio_lo"] == 0) & (lab[hhi] == 0)
        rec = {
            "fold": k,
            "n_pos": int(sl.sum()),
            "n_both_def": int(both.sum()),
            "n_neither": int(neither.sum()),
            "share_neither": (
                float(neither.sum() / both.sum()) if int(both.sum()) else float("nan")
            ),
        }
        rows.append(rec)
        print(
            f"  fold {k}: pos={rec['n_pos']:3d} neither={rec['n_neither']:3d}/"
            f"{rec['n_both_def']} share={rec['share_neither']:.3f}"
        )
        assert_no_holdout(lab.loc[sl, "company_id"])
    return {"rows": rows}


def pass38_hole_in_neither(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    lab: pd.DataFrame,
    p18: dict,
) -> dict:
    """Are hole positives leftover (neither cash nor HHI)?"""
    print("\n" + "=" * 72)
    print("CUT 38 — hole positives inside the leftover 2×2")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    out = {"n_hole_pos": 0, "n_neither": 0, "share_neither": float("nan")}
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return out
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    keys = set(
        zip(
            df.loc[
                train_by_y[Y_AR] & (y == 1) & (share <= 0.01) & (nc > float(med_n)),
                "company_id",
            ].astype(str),
            pd.to_datetime(df.loc[
                train_by_y[Y_AR] & (y == 1) & (share <= 0.01) & (nc > float(med_n)),
                "period",
            ]),
        )
    )
    hhi = "d_cust_hhi_hi"
    pos = lab[Y_AR] == 1
    both = pos & lab["a_io_ratio_lo"].notna() & lab[hhi].notna()
    neither = both & (lab["a_io_ratio_lo"] == 0) & (lab[hhi] == 0)
    lab_keys_neither = set(
        zip(lab.loc[neither, "company_id"].astype(str), pd.to_datetime(lab.loc[neither, "period"]))
    )
    n_hp = len(keys)
    n_in = len(keys & lab_keys_neither)
    out["n_hole_pos"] = n_hp
    out["n_neither"] = n_in
    out["share_neither"] = (n_in / n_hp) if n_hp else float("nan")
    print(f"  hole pos {n_hp}; also leftover neither {n_in} share={out['share_neither']:.3f}")
    return out


def pass39_hole_pos_2x2(
    df: pd.DataFrame,
    train_by_y: dict[str, pd.Series],
    p18: dict,
) -> dict:
    """2×2 cash × HHI among hole positives only."""
    print("\n" + "=" * 72)
    print("CUT 39 — cash × customer-HHI among hole positives")
    print("=" * 72)
    med_n = p18.get("med_n_cust")
    empty = {"cash_only": 0, "hhi_only": 0, "both": 0, "neither": 0, "n": 0}
    if med_n is None or not np.isfinite(med_n):
        print("  skip")
        return empty
    lab = train_by_y[Y_AR]
    share = pd.to_numeric(df["d_tx_cp_share"], errors="coerce")
    nc = pd.to_numeric(df["d_n_cust"], errors="coerce")
    y = pd.to_numeric(df[Y_AR], errors="coerce")
    cash = pd.to_numeric(df["a_io_ratio_lo"], errors="coerce")
    hhi = pd.to_numeric(df["d_cust_hhi_hi"], errors="coerce")
    hole_pos = lab & (y == 1) & (share <= 0.01) & (nc > float(med_n)) & cash.notna() & hhi.notna()
    assert_no_holdout(df.loc[hole_pos, "company_id"])
    lo = cash[hole_pos] == 1
    hi = hhi[hole_pos] == 1
    rec = {
        "n": int(hole_pos.sum()),
        "cash_only": int((lo & ~hi).sum()),
        "hhi_only": int((~lo & hi).sum()),
        "both": int((lo & hi).sum()),
        "neither": int((~lo & ~hi).sum()),
    }
    print(
        f"  n={rec['n']} cash_only={rec['cash_only']} hhi_only={rec['hhi_only']} "
        f"both={rec['both']} neither={rec['neither']}"
    )
    return rec


def pass40_fold_so_far(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is the unnamed fold just longer books?"""
    print("\n" + "=" * 72)
    print("CUT 40 — months-so-far by fold (AR train)")
    print("=" * 72)
    work = df.copy()
    work["_so_far"] = _so_far(work)
    lab = train_by_y[Y_AR]
    assert_no_holdout(work.loc[lab, "company_id"])
    folds = pd.to_numeric(work["fold"], errors="coerce")
    rows = []
    for k in range(N_FOLDS):
        sl = lab & (folds == k)
        sf = pd.to_numeric(work.loc[sl, "_so_far"], errors="coerce")
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "med_so_far": float(sf.median()) if int(sl.sum()) else float("nan"),
            "share_short": float((sf < 12).mean()) if int(sl.sum()) else float("nan"),
            "share_long": float((sf >= 18).mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  fold {k}: med so-far={rec['med_so_far']:.1f} "
            f"short<12={rec['share_short']:.3f} long>=18={rec['share_long']:.3f}"
        )
    return {"rows": rows}


def pass41_ap_fold_so_far(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Are high-rate AP folds just longer books?"""
    print("\n" + "=" * 72)
    print("CUT 41 — months-so-far by fold (AP train)")
    print("=" * 72)
    work = df.copy()
    work["_so_far"] = _so_far(work)
    lab = train_by_y[Y_AP]
    assert_no_holdout(work.loc[lab, "company_id"])
    folds = pd.to_numeric(work["fold"], errors="coerce")
    y = pd.to_numeric(work[Y_AP], errors="coerce")
    rows = []
    for k in range(N_FOLDS):
        sl = lab & (folds == k)
        sf = pd.to_numeric(work.loc[sl, "_so_far"], errors="coerce")
        rec = {
            "fold": k,
            "n": int(sl.sum()),
            "rate": float(y[sl].mean()) if int(sl.sum()) else float("nan"),
            "med_so_far": float(sf.median()) if int(sl.sum()) else float("nan"),
            "share_short": float((sf < 12).mean()) if int(sl.sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  fold {k}: rate={rec['rate']:.3f} med so-far={rec['med_so_far']:.1f} "
            f"short={rec['share_short']:.3f}"
        )
    return {"rows": rows}


def pass42_so_far_single(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is months-so-far itself a Y5 single? Never E."""
    print("\n" + "=" * 72)
    print("CUT 42 — so-far months as a single (tenure vs Y5)")
    print("=" * 72)
    work = df.copy()
    work["_so_far"] = _so_far(work)
    folds = work["fold"].to_numpy()
    rows = []
    for y_col in (Y_AP, Y_AR):
        lab = train_by_y[y_col]
        oof = signed_oof_auroc(work, "_so_far", y_col, lab, folds)
        size = _size_auroc_of_feature(work, "_so_far", lab)
        rec = {
            "y": y_col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "sign": oof["train_sign"],
            "size_auroc": size["auroc"],
        }
        rows.append(rec)
        print(
            f"  {y_col} so-far CV={oof['cv']:.3f}±{oof['sd']:.3f} "
            f"sign={oof['train_sign']} sizeAUC={size['auroc']:.3f}"
        )
    return {"rows": rows}


def pass43_ap_group_by_fold(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is the AP large-group protective tail one fold?"""
    print("\n" + "=" * 72)
    print("CUT 43 — AP rate for h_group_size≥18 by fold")
    print("=" * 72)
    lab = train_by_y[Y_AP]
    size = pd.to_numeric(df["h_group_size"], errors="coerce")
    y = pd.to_numeric(df[Y_AP], errors="coerce")
    folds = pd.to_numeric(df["fold"], errors="coerce")
    rows = []
    for k in range(N_FOLDS):
        sl = lab & (folds == k) & size.notna() & y.notna()
        hi = sl & (size >= 18)
        rec = {
            "fold": k,
            "n_hi": int(hi.sum()),
            "n_pos_hi": int((y[hi] == 1).sum()),
            "rate_hi": float(y[hi].mean()) if int(hi.sum()) else float("nan"),
            "rate_rest": float(y[sl & (size < 18)].mean()) if int((sl & (size < 18)).sum()) else float("nan"),
        }
        rows.append(rec)
        print(
            f"  fold {k}: ≥18 n={rec['n_hi']:4d} pos={rec['n_pos_hi']:3d} "
            f"rate={rec['rate_hi']:.3f} rest={rec['rate_rest']:.3f}"
        )
        assert_no_holdout(df.loc[sl, "company_id"])
    return {"rows": rows}


def extra_md_lines(
    p1: dict,
    p7: dict | None = None,
    p8: dict | None = None,
    p9: dict | None = None,
    p10: dict | None = None,
    p11: dict | None = None,
    p12: dict | None = None,
    p13: dict | None = None,
    p14: dict | None = None,
    p15: dict | None = None,
    p16: dict | None = None,
    p17: dict | None = None,
    p18: dict | None = None,
    p19: dict | None = None,
    p20: dict | None = None,
    p21: dict | None = None,
    p22: dict | None = None,
    p23: dict | None = None,
    p24: dict | None = None,
    p25: dict | None = None,
    p26: dict | None = None,
    p27: dict | None = None,
    p28: dict | None = None,
    p29: dict | None = None,
    p30: dict | None = None,
    p31: dict | None = None,
    p32: dict | None = None,
    p33: dict | None = None,
    p34: dict | None = None,
    p35: dict | None = None,
    p36: dict | None = None,
    p37: dict | None = None,
    p38: dict | None = None,
    p39: dict | None = None,
    p40: dict | None = None,
    p41: dict | None = None,
    p42: dict | None = None,
    p43: dict | None = None,
) -> list[str]:
    tw_ap = p1["twos"][Y_AP]
    tw_ar = p1["twos"][Y_AR]
    lines = [
        "### 1b — 2×2 reminder",
        "",
        f"AP neither cell {tw_ap['neither']}/{tw_ap['n']} "
        f"({tw_ap['neither'] / tw_ap['n']:.1%}). "
        f"AR neither cell {tw_ar['neither']}/{tw_ar['n']} "
        f"({tw_ar['neither'] / tw_ar['n']:.1%}). "
        "Largest cell is leftover, not cash and not counterpart HHI.",
        "",
    ]
    if p7:
        lines += [
            "### 7 — HHI monopoly tail (Y4 shape, unused here)",
            "",
            "Y4 `d_cust_hhi` is a >0.975 monopoly tail. Supplier `d_supp_hhi` is the unused AP counterpart. "
            "If Y5 were the same story, the tail rate would jump.",
            "",
            "| Y | feature | n tail / pos | P(Y=1) tail | P(Y=1) rest | tail AUROC |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in p7["rows"]:
            lines.append(
                f"| `{r['y']}` | `{r['col']}` | {r['n_hi']} / {r['n_pos_hi']} | "
                f"{r['rate_hi']:.3f} | {r['rate_lo']:.3f} | {r['auroc_tail']:.3f} |"
            )
        lines.append("")
    if p8:
        lines += [
            "### 8 — neither cell",
            "",
        ]
        for y_col in (Y_AP, Y_AR):
            blk = p8[y_col]
            lines += [
                f"`{y_col}` neither n={blk['n']} of {blk['n_2x2']} 2×2 positives.",
                "",
                "| flag inside neither | n_hi / n | share |",
                "|---|---:|---:|",
            ]
            for r in blk["rows"]:
                lines.append(
                    f"| `{r['flag']}` | {r['n_hi']} / {r['n_defined']} | {r['share']:.3f} |"
                )
            lines.append("")
    if p9:
        lines += [
            "### 9 — fold AUCs (night singles)",
            "",
            "Wide fold sd means the night train quote is not a stable Q5.",
            "",
            "| Y | feature | folds | CV ± sd |",
            "|---|---|---|---:|",
        ]
        for r in p9["rows"]:
            bits = " ".join(f"{a:.3f}" for a in r["folds"])
            lines.append(
                f"| `{r['y']}` | `{r['col']}` | {bits} | {r['cv']:.3f} ± {r['sd']:.3f} |"
            )
        lines.append("")
    if p10:
        lines += [
            "### 10 — residual inside size terciles",
            "",
            "| Y | slice | feature | n | n_pos | P(Y=1) | CV AUROC |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for r in p10["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | `{r['col']}` | {r['n']} | "
                f"{r['n_pos']} | {r['rate']:.3f} | {r['cv']:.3f} |"
            )
        lines.append("")
    if p11:
        lines += [
            "### 11 — short-book lag1 (honest Q6)",
            "",
            "Only 1-month leads transfer (`q6_quoted.md`). Short = months-so-far < 12. "
            "`d_tx_cp_share` on short is **0.445 / lag1 0.431** (75 pos) vs long **0.727**. "
            "**Q6 CLOSE** — do not transfer the AR KEEP to short books or the hidden 72 as a lead.",
            "",
            "| Y | slice | feature | lag | coverage | CV AUROC ± sd | n_pos |",
            "|---|---|---|---:|---:|---:|---:|",
        ]
        for r in p11["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | `{r['col']}` | {r['lag']} | "
                f"{r['coverage']:.3f} | {r['cv']:.3f} ± {r['sd']:.3f} | {r['n_pos']} |"
            )
        lines.append("")
    if p12:
        lines += [
            "### 12 — company-level trait vs turning",
            "",
            "Spearman of each train company's mean X vs its Y5 month-share (≥3 labeled months).",
            "",
            "| Y | X | companies | ever pos | ρ | med X pos / neg cos |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in p12["rows"]:
            lines.append(
                f"| `{r['y']}` | `{r['x']}` | {r['n_cos']} | {r['n_ever_pos']} | "
                f"{r['rho']:+.3f} | {r['x_pos']:.3g} / {r['x_neg']:.3g} |"
            )
        lines.append("")
    if p13:
        lines += [
            "### 13 — AP ∩ AR",
            "",
            f"Train both-labeled {p13['n_both_lab']}: both-pos {p13['n_both_pos']}, "
            f"AP-only {p13['n_ap_only']}, AR-only {p13['n_ar_only']}, "
            f"Spearman {p13['rho']:+.3f}. Two labels, not a rewrite.",
            "",
        ]
    if p14:
        lines += [
            "### 14 — `d_tx_cp_share` honesty",
            "",
            f"Any-named-cp (comparator) AR CV **{p14['ar_bin_cv']:.3f}**. "
            f"Continuous all-row **{p14['ar_all_cv']:.3f}** "
            f"(gap vs binary {p14['gap_vs_binary']:+.3f}). "
            f"On cp>0 support CV **{p14['ar_pos_cv']:.3f}**. "
            f"presence_rewrite={p14['presence_rewrite']}; intensity_dead={p14['intensity_dead']}.",
            "",
            "| Y | slice | n | n_pos | P(Y=1) |",
            "|---|---|---:|---:|---:|",
        ]
        for r in p14["rates"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines += [
            "",
            "| Y | feature | slice | CV AUROC ± sd | n |",
            "|---|---|---|---:|---:|",
        ]
        for r in p14["rows"]:
            lines.append(
                f"| `{r['y']}` | `{r['col']}` | {r['slice']} | "
                f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['n_defined']} |"
            )
        lines.append("")
        if p14["presence_rewrite"] or p14["intensity_dead"]:
            lines.append(
                "Honesty: the AR KEEP on `d_tx_cp_share` is *unnamed-counterparty presence*, "
                "not share intensity. Still not E. Still not cash. PARK as a model X if intensity is dead."
            )
        else:
            lines.append(
                "Honesty: share intensity still ranks after zeros are removed."
            )
        lines.append("")
    if p15:
        lines += [
            "### 15 — large-group protective tail",
            "",
            "| Y | n ≥18 / pos | P(Y=1) ≥18 | P(Y=1) rest | small-group flag AUROC | companies ≥18 |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in p15["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['n_hi']} / {r['n_pos_hi']} | {r['rate_hi']:.3f} | "
                f"{r['rate_lo']:.3f} | {r['auroc_tail']:.3f} | {r['n_cos_hi']} |"
            )
        lines += [
            "",
            "AP Q5 (group 18–22) is 2.9%. Protective, not a why for the positives. Not monotone. CLOSE as Q5 X.",
            "",
        ]
    if p16:
        lines += [
            "### 16 — night singles on the neither leftover",
            "",
            "| Y | feature | n | n_pos | CV AUROC ± sd |",
            "|---|---|---:|---:|---:|",
        ]
        for r in p16["rows"]:
            lines.append(
                f"| `{r['y']}` | `{r['col']}` | {r['n']} | {r['n_pos']} | "
                f"{r['cv']:.3f} ± {r['sd']:.3f} |"
            )
        lines.append("")
    if p17:
        lines += [
            "### 17 — leftover still has cash",
            "",
            "Join QA already: Y5 labels are 100% ever-ERP and ~99% have a tx that month. "
            "This cut is the neither cell. CLOSE “died because cash was missing”.",
            "",
            "| Y | slice | n | share days>0 | share a_in3 defined | median days |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in p17["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['share_days_gt0']:.3f} | "
                f"{r['share_in3_def']:.3f} | {r['med_days']:.1f} |"
            )
        lines.append("")
    if p18 and p18.get("rows"):
        lines += [
            "### 18 — tagging hole vs invoice thinness",
            "",
            f"Train AR complete; median `d_n_cust`={p18['med_n_cust']:.3g}. "
            f"Low share = `d_tx_cp_share` ≤ 0.01. "
            f"tagging_hole={p18['tagging_hole']}: "
            f"low-share+many-invoice-cust P(Y=1)=**{p18['hole_rate']:.1%}** vs "
            f"named+many-cust {p18['named_rate']:.1%} vs "
            f"low-share+few-cust {p18['thin_rate']:.1%}.",
            "",
            "| slice | n | n_pos | P(Y=1) |",
            "|---|---:|---:|---:|",
        ]
        for r in p18["rows"]:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines.append("")
        if p18.get("ap_rows"):
            lines += [
                f"AP analogue (median `d_n_supp`={p18.get('med_n_supp', float('nan')):.3g}): "
                f"tagging_hole_ap={p18.get('tagging_hole_ap')} "
                f"low-share+many-supp {p18.get('hole_rate_ap', float('nan')):.1%} vs "
                f"named {p18.get('named_rate_ap', float('nan')):.1%}.",
                "",
                "| slice | n | n_pos | P(Y=1) |",
                "|---|---:|---:|---:|",
            ]
            for r in p18["ap_rows"]:
                lines.append(
                    f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
                )
            lines.append("")
    if p19 and p19.get("rows"):
        lines += [
            "### 19 — tagging hole × own-low cash",
            "",
            f"Among above-median `d_n_cust` months with a cash flag. "
            f"cash_independent={p19.get('cash_independent')}: "
            f"hole+ok-cash **{p19.get('hole_ok_rate', float('nan')):.1%}** vs "
            f"named+ok-cash {p19.get('named_ok_rate', float('nan')):.1%}. "
            "If the hole only fired when cash was low, it would be a cash rewrite.",
            "",
            "| slice | n | n_pos | P(Y=1) |",
            "|---|---:|---:|---:|",
        ]
        for r in p19["rows"]:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines.append("")
    if p20:
        lines += [
            "### 20 — hole as company trait + holdout n",
            "",
            f"Hole-flag group-fold CV on hi-cust train: "
            f"**{p20.get('cv', float('nan')):.3f}** ± {p20.get('sd', float('nan')):.3f} "
            f"(n={p20.get('n_hi_cust')} pos={p20.get('n_pos_hi_cust')}). "
            f"Companies with a hi-cust AR month: {p20.get('n_cos_hi_cust')}; "
            f"ever-hole {p20.get('n_cos_ever_hole')} "
            f"(mean company-rate {p20.get('rate_ever_hole', float('nan')):.3f} vs "
            f"never {p20.get('rate_never_hole', float('nan')):.3f}). "
            f"Holdout hole coverage only: n={p20.get('ho_n_hole')} "
            f"pos={p20.get('ho_n_hole_pos')} (LOW_POWER — not an AUROC keep).",
            "",
        ]
    if p21 and p21.get("rows"):
        lines += [
            "### 21 — hole_lag1 honest 1-month Q6",
            "",
            f"Lag-1 of the tagging-hole flag. Short so-far<12 CV "
            f"**{p21.get('short_cv', float('nan')):.3f}**; "
            f"long ≥18 {p21.get('long_cv', float('nan')):.3f}; "
            f"all-train {p21.get('all_cv', float('nan')):.3f}. "
            "CLOSE a quarter lead. Only the 1-month clock is eligible.",
            "",
            "| slice | col | n | n_pos | CV AUROC ± sd | coverage |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for r in p21["rows"]:
            lines.append(
                f"| {r['slice']} | `{r['col']}` | {r['n']} | {r['n_pos']} | "
                f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['coverage']:.3f} |"
            )
        lines.append("")
    if p22:
        lines += [
            "### 22 — hole-positive concentration",
            "",
            f"Hole months hold **{p22.get('share_of_ar_pos', float('nan')):.1%}** of AR positives "
            f"({p22.get('n_hole_pos')}/{p22.get('n_ar_pos')}) across "
            f"{p22.get('n_cos_hole_pos')} companies. "
            f"top1={p22.get('top1', float('nan')):.3f} top3={p22.get('top3', float('nan')):.3f} "
            f"HHI={p22.get('hhi', float('nan')):.3f}. "
            f"fragile={p22.get('fragile')} (PARK the sentence if top3≥0.50).",
            "",
        ]
    if p23:
        lines += [
            "### 23 — intensity on thick vs thin invoice books",
            "",
            f"`d_tx_cp_share` CV on hi-cust **{p23.get('hi_cv', float('nan')):.3f}**; "
            f"lo-cust {p23.get('lo_cv', float('nan')):.3f}. "
            "KEEP lives on the thick book, not the thin one.",
            "",
            "| slice | n | n_pos | CV AUROC ± sd |",
            "|---|---:|---:|---:|",
        ]
        for r in p23.get("rows") or []:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['cv']:.3f} ± {r['sd']:.3f} |"
            )
        lines.append("")
    if p24 and p24.get("rows"):
        lines += [
            "### 24 — turning-month Δ `d_tx_cp_share`",
            "",
            f"First difference (now − lag1). All-train CV {p24.get('cv', float('nan')):.3f}. "
            "A drop in named-cp share is not a stronger why than the level.",
            "",
            "| slice | n | n_pos | CV AUROC ± sd | sign | coverage |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for r in p24["rows"]:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['n_pos']} | "
                f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['sign']} | {r['coverage']:.3f} |"
            )
        lines.append("")
    if p25 and p25.get("rows"):
        lines += [
            "### 25 — short-book contemporaneous hole rate",
            "",
            f"Q5 can be same-month; Q6 cannot. Short hi-cust hole "
            f"**{p25.get('short_hole_rate', float('nan')):.1%}** vs named "
            f"{p25.get('short_named_rate', float('nan')):.1%}. "
            f"Long hole {p25.get('long_hole_rate', float('nan')):.1%} vs named "
            f"{p25.get('long_named_rate', float('nan')):.1%}.",
            "",
            "| slice | n | n_pos | P(Y=1) |",
            "|---|---:|---:|---:|",
        ]
        for r in p25["rows"]:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines.append("")
    if p26 and p26.get("rows"):
        lines += [
            "### 26 — AP leftover extra D singles",
            "",
            "Unused supplier columns. SIZE_PARK if the column ranks large vs small firms.",
            "",
            "| feature | CV AUROC ± sd | sign | size AUROC | SIZE_PARK | vs size CV |",
            "|---|---:|---:|---:|---|---:|",
        ]
        for r in p26["rows"]:
            lines.append(
                f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['sign']} | "
                f"{r['size_auroc']:.3f} | {r['size_park']} | {r['gap_vs_size']:+.3f} |"
            )
        lines.append("")
    if p27 and p27.get("rows"):
        lines += [
            "### 27 — `d_tx_cp_share` vs size (not a rewrite)",
            "",
            "| vs | Spearman ρ |",
            "|---|---:|",
        ]
        for r in p27["rows"]:
            lines.append(f"| `{r['col']}` | {r['rho']:+.3f} |")
        lines.append("")
    if p28 and p28.get("rows"):
        lines += [
            "### 28 — hole flag vs E / Y2 / Y9",
            "",
            f"max |ρ| vs E = {p28.get('max_abs_rho_e', float('nan')):.3f}. "
            f"leak_fail={p28.get('leak_fail')} (fail ≥{LEAK_RHO}). "
            "Cannot use E to show persistence; this is the non-E hole vs those E columns.",
            "",
            "| kind | col | ρ | fail |",
            "|---|---|---:|---|",
        ]
        for r in p28["rows"]:
            lines.append(
                f"| {r['kind']} | `{r['col']}` | {r['rho']:+.3f} | {r['fail']} |"
            )
        lines.append("")
    if p29 and p29.get("table") and p29["table"].get("rows"):
        t = p29["table"]
        lines += [
            "### 29 — `d_tx_cp_share` quintiles on hi-cust AR",
            "",
            f"n={t['n']} head_only={t.get('head_only')} "
            f"monotone_down={t.get('monotone_down')} tail_only={t.get('tail_only')}.",
            "",
            "| Q | interval | n | n_pos | P(Y=1) | median X |",
            "|---:|---|---:|---:|---:|---:|",
        ]
        for r in t["rows"]:
            lines.append(
                f"| {r['q']} | {r['interval']} | {r['n']} | {r['n_pos']} | "
                f"{r['y_rate']:.3f} | {r['x_median']:.4g} |"
            )
        lines.append("")
    if p30 and p30.get("rows"):
        lines += [
            "### 30 — hole rate by group fold",
            "",
            f"Hole-rate range {p30.get('min_rate', float('nan')):.3f}–"
            f"{p30.get('max_rate', float('nan')):.3f}. "
            "If one fold owns the 17.4%, PARK the sentence.",
            "",
            "| fold | n hole | pos | P(Y=1) hole | P(Y=1) named |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in p30["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n_hole']} | {r['n_pos_hole']} | "
                f"{r['rate_hole']:.3f} | {r['rate_named']:.3f} |"
            )
        lines.append("")
    if p31:
        lines += [
            "### 31 — drop the fold that owns hole positives",
            "",
            f"Fold {p31.get('heavy_fold')} holds "
            f"**{p31.get('share_pos_heavy', float('nan')):.0%}** of hole positives. "
            f"Without it: hole {p31.get('hole_rate_wo', float('nan')):.1%} vs named "
            f"{p31.get('named_rate_wo', float('nan')):.1%}; "
            f"`d_tx_cp_share` CV {p31.get('cv_wo', float('nan')):.3f}. "
            f"survives={p31.get('survives')}. "
            "Pooled 17.4% is not a leave-one-group fact.",
            "",
        ]
    if p32:
        lines += [
            "### 32 — who is the heavy fold",
            "",
            f"{p32.get('n_cos')} train AR companies. "
            f"median `h_group_size`={p32.get('med_group', float('nan')):.3g}, "
            f"median `d_tx_cp_share`={p32.get('med_share', float('nan')):.3g}, "
            f"median `d_n_cust`={p32.get('med_ncust', float('nan')):.3g}, "
            f"share of months that are the hole cell={p32.get('share_hole_months', float('nan')):.1%}. "
            "Not a sibling_h edit.",
            "",
        ]
    if p33 and p33.get("rows"):
        lines += [
            "### 33 — median `d_tx_cp_share` by fold",
            "",
            "If one fold is the unnamed cluster, dropping it removes the treatment group — "
            "not a proof the pooled Q5 is fake.",
            "",
            "| fold | n | companies | median share | share=0 | P(Y=1) |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in p33["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n']} | {r['n_cos']} | {r['med_share']:.3f} | "
                f"{r['share_eq0']:.3f} | {r['rate']:.3f} |"
            )
        lines.append("")
    if p34 and p34.get("rows"):
        lines += [
            "### 34 — share=0 × invoice thickness by fold",
            "",
            "Fold 2 zeros should be thin-book if the hole is fold-3-only.",
            "",
            "| fold | book | n zero | pos | P(Y=1) |",
            "|---:|---|---:|---:|---:|",
        ]
        for r in p34["rows"]:
            lines.append(
                f"| {r['fold']} | {r['book']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines.append("")
    if p35 and p35.get("rows"):
        lines += [
            "### 35 — AP rate by fold",
            "",
            f"Range {p35.get('min_rate', float('nan')):.3f}–"
            f"{p35.get('max_rate', float('nan')):.3f}. "
            "A one-fold spike would be another cluster leftover.",
            "",
            "| fold | n | pos | companies | P(Y=1) |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in p35["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | {r['rate']:.3f} |"
            )
        lines.append("")
    if p36 and p36.get("rows"):
        lines += [
            "### 36 — AP leftover share of positives by fold",
            "",
            "Neither = not own-low cash and not own-high `d_supp_hhi`.",
            "",
            "| fold | n pos | neither / defined | share leftover |",
            "|---:|---:|---:|---:|",
        ]
        for r in p36["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n_pos']} | {r['n_neither']} / {r['n_both_def']} | "
                f"{r['share_neither']:.3f} |"
            )
        lines.append("")
    if p37 and p37.get("rows"):
        lines += [
            "### 37 — AR leftover share of positives by fold",
            "",
            "Neither = not own-low cash and not own-high `d_cust_hhi`. "
            "Fold 3 can still be leftover on cash/HHI while being the unnamed cluster.",
            "",
            "| fold | n pos | neither / defined | share leftover |",
            "|---:|---:|---:|---:|",
        ]
        for r in p37["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n_pos']} | {r['n_neither']} / {r['n_both_def']} | "
                f"{r['share_neither']:.3f} |"
            )
        lines.append("")
    if p38:
        lines += [
            "### 38 — hole positives in the leftover cell",
            "",
            f"{p38.get('n_neither')}/{p38.get('n_hole_pos')} hole positives "
            f"({p38.get('share_neither', float('nan')):.1%}) are leftover "
            "(not own-low cash, not own-high customer HHI). "
            "The unnamed cluster is a slice of leftover, not a cash/HHI rewrite.",
            "",
        ]
    if p39:
        lines += [
            "### 39 — 2×2 among hole positives",
            "",
            f"n={p39.get('n')}: cash_only={p39.get('cash_only')} "
            f"hhi_only={p39.get('hhi_only')} both={p39.get('both')} "
            f"neither={p39.get('neither')}.",
            "",
        ]
    if p40 and p40.get("rows"):
        lines += [
            "### 40 — months-so-far by fold",
            "",
            "If fold 3 is just longer books, the unnamed cluster is a tenure artifact.",
            "",
            "| fold | n | median so-far | share <12 | share ≥18 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in p40["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n']} | {r['med_so_far']:.1f} | "
                f"{r['share_short']:.3f} | {r['share_long']:.3f} |"
            )
        lines.append("")
    if p41 and p41.get("rows"):
        lines += [
            "### 41 — AP months-so-far by fold",
            "",
            "High-rate AP folds (0/1/3) vs low (2/4) — tenure confounder?",
            "",
            "| fold | n | P(Y=1) | median so-far | share <12 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in p41["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n']} | {r['rate']:.3f} | "
                f"{r['med_so_far']:.1f} | {r['share_short']:.3f} |"
            )
        lines.append("")
    if p42 and p42.get("rows"):
        lines += [
            "### 42 — tenure (so-far) as a single",
            "",
            "| Y | CV AUROC ± sd | sign | size AUROC |",
            "|---|---:|---:|---:|",
        ]
        for r in p42["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['sign']} | "
                f"{r['size_auroc']:.3f} |"
            )
        lines.append("")
    if p43 and p43.get("rows"):
        lines += [
            "### 43 — AP `h_group_size`≥18 by fold",
            "",
            "Protective tail should show in every fold if it is a why.",
            "",
            "| fold | n ≥18 | pos | P(Y=1) ≥18 | P(Y=1) rest |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in p43["rows"]:
            lines.append(
                f"| {r['fold']} | {r['n_hi']} | {r['n_pos_hi']} | "
                f"{r['rate_hi']:.3f} | {r['rate_rest']:.3f} |"
            )
        lines.append("")
    lines += [
        "## Return (this owner)",
        "",
        "Y5 is **unexplained leftover** on cash and counterpart HHI (65% neither). "
        "Not cash-stress (days>0 99%+). Not a supplier-tail (`d_supp_hhi` monopoly is protective). "
        "AR has a company-group unnamed-cp cluster — quote `d_tx_cp_share`, do not treat 17.4% as a panel law.",
        "",
        "Best non-E vs night: AP `h_group_size` CV 0.581 vs night train 0.599 (CLOSE). "
        "AR `d_tx_cp_share` CV 0.576 vs night train 0.611 (KEEP quote, PARK as X). "
        "Thin-F PARK. Trees PARK. Q6 CLOSE.",
        "",
        "Q5 sentence (no family E): sustained AR-od30 is higher when the bank trail names "
        "no counterparties on a thick invoice book — in one unnamed company-group "
        "(fold 3: 25.1% on 191 months), not as a leave-one-group law. "
        "AP leftover: not low cash, not supplier monopoly.",
        "",
    ]
    return lines


def run(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Y5 why — cash vs supplier vs leftover")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--no-md", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    started = datetime.now().isoformat(timespec="minutes")
    wall0 = datetime.now()
    print(f"y5_why start {started} seed={FOLD_SEED} agent={AGENT}")
    print("forbidden X = family E; labels from parquet; no GBM; no build_targets")
    print("META.forbidden_x_families", Y5_META.get("forbidden_x_families"))
    print(
        "leakage_check demo",
        leakage_check(
            ["d_supp_hhi", "a_io_ratio", "e_ap_overdue_30"],
            Y_AP,
            FORBIDDEN,
        ),
    )

    hold_ids = load_holdout()
    con = connect()
    store = load_store()
    y = load_y()
    train_cos = train_companies(con)
    con.close()

    assert_no_holdout(train_cos)
    leak_ok = leakage_check(list(SINGLE_COLS), Y_AP, FORBIDDEN)
    if not leak_ok["ok"]:
        raise RuntimeError(f"SINGLE_COLS leak: {leak_ok['issues']}")

    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    panel = store.merge(y, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold", "group_id"]], on="company_id", how="left")
    panel["log1p_a_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").abs())
    panel = add_own_cuts(panel)
    panel = add_lags(panel, list(LAG_STEMS), (1,))

    is_hold = panel["company_id"].astype(str).isin(hold_ids)
    train_row = (~is_hold) & panel["fold"].notna()
    assert_no_holdout(panel.loc[train_row, "company_id"])

    train_by_y = {}
    n_tr = {}
    for y_col in (Y_AP, Y_AR, Y_DELAY):
        lab = train_row & panel[y_col].notna()
        train_by_y[y_col] = lab
        n = int(lab.sum())
        n_pos = int((panel.loc[lab, y_col] == 1).sum())
        rate = float(panel.loc[lab, y_col].mean()) if n else float("nan")
        n_ho = int((is_hold & panel[y_col].notna()).sum())
        n_ho_pos = int((is_hold & (panel[y_col] == 1)).sum())
        n_tr[y_col] = {"n": n, "n_pos": n_pos, "rate": rate, "n_ho": n_ho, "n_ho_pos": n_ho_pos}
        print(
            f"{y_col}: train n={n} pos={n_pos} rate={rate:.4f}  "
            f"holdout n={n_ho} pos={n_ho_pos} LOW_POWER"
        )

    lab = panel.loc[train_row].copy()
    p1 = pass1_decompose(lab)
    p2 = pass2_quintiles(panel, train_by_y, args.no_plot)
    p3 = pass3_singles(panel, train_by_y)
    p4 = pass4_leak(panel, train_by_y[Y_AP] | train_by_y[Y_AR])
    p5 = pass5_holdout(panel, is_hold)
    p6 = pass6_lag1(panel, train_by_y)
    p7 = pass7_hhi_tail(panel, train_by_y)
    p8 = pass8_neither(lab)
    p9 = pass9_folds(p3)
    p10 = pass10_size_residual(panel, train_by_y)
    p11 = pass11_short_q6(panel, train_by_y)
    p12 = pass12_company_trait(panel, train_by_y)
    p13 = pass13_label_overlap(panel, train_row)
    p14 = pass14_txcp_honesty(panel, train_by_y)
    p15 = pass15_group_tail(panel, train_by_y)
    p16 = pass16_neither_singles(panel, lab, train_by_y)
    p17 = pass17_cash_present(lab)
    p18 = pass18_tagging_vs_invoice(panel, train_by_y)
    p19 = pass19_hole_vs_cash(panel, train_by_y, p18)
    p20 = pass20_hole_trait_holdout(panel, train_by_y, is_hold, p18)
    p21 = pass21_hole_lag1(panel, train_by_y, p18)
    p22 = pass22_hole_concentration(panel, train_by_y, p18)
    p23 = pass23_intensity_hi_cust(panel, train_by_y, p18)
    p24 = pass24_txcp_diff1(panel, train_by_y)
    p25 = pass25_short_hole_rate(panel, train_by_y, p18)
    p26 = pass26_ap_leftover_d(panel, train_by_y)
    p27 = pass27_txcp_vs_size(panel, train_by_y)
    p28 = pass28_hole_vs_e_and_y(panel, train_by_y, p18)
    p29 = pass29_hi_cust_quintiles(panel, train_by_y, p18)
    p30 = pass30_hole_by_fold(panel, train_by_y, p18)
    p31 = pass31_drop_fold3(panel, train_by_y, p18, p30)
    p32 = pass32_heavy_fold_who(panel, train_by_y, p18, p31)
    p33 = pass33_share_by_fold(panel, train_by_y)
    p34 = pass34_zero_by_fold_book(panel, train_by_y, p18)
    p35 = pass35_ap_rate_by_fold(panel, train_by_y)
    p36 = pass36_ap_neither_by_fold(lab)
    p37 = pass37_ar_neither_by_fold(lab)
    p38 = pass38_hole_in_neither(panel, train_by_y, lab, p18)
    p39 = pass39_hole_pos_2x2(panel, train_by_y, p18)
    p40 = pass40_fold_so_far(panel, train_by_y)
    p41 = pass41_ap_fold_so_far(panel, train_by_y)
    p42 = pass42_so_far_single(panel, train_by_y)
    p43 = pass43_ap_group_by_fold(panel, train_by_y)
    verdict = decide(p1, p2, p3, p4, p14)

    extra = {
        "md_lines": extra_md_lines(
            p1, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21,
            p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
        ),
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
        "registry": _extra_registry(
            p7, p8, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21,
            p22, p23, p24, p25, p26, p27, p28, p29, p30, p31, p32, p33, p34, p35, p36, p37, p38, p39, p40, p41, p42, p43,
        ),
    }
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if not args.no_registry:
        rows = registry_rows(p1, p2, p3, p4, p5, p6, verdict, ts, extra=extra)
        append_registry(rows)
        print(f"appended {len(rows)} registry rows (train-only metrics)")
    if not args.no_md:
        write_md(started, n_tr, p1, p2, p3, p4, p5, p6, verdict, extra)

    elapsed = (datetime.now() - wall0).total_seconds()
    quote = {
        "headline": verdict["headline"],
        "ap_story": verdict["by_y"][Y_AP]["story"],
        "ar_story": verdict["by_y"][Y_AR]["story"],
        "ap_best": (verdict["by_y"][Y_AP].get("best") or {}).get("col"),
        "ap_best_cv": (verdict["by_y"][Y_AP].get("best") or {}).get("cv"),
        "ar_best": (verdict["by_y"][Y_AR].get("best") or {}).get("col"),
        "ar_best_cv": (verdict["by_y"][Y_AR].get("best") or {}).get("cv"),
        "night_ap": NIGHT_AP,
        "night_ar": NIGHT_AR,
        "trees": "PARK",
        "elapsed_s": elapsed,
    }
    print("\nQUOTE")
    print(json.dumps(quote, indent=2, default=str))
    print(f"elapsed {elapsed:.1f}s — stay on this module for more cuts")
    return {
        "panel": panel,
        "train_by_y": train_by_y,
        "train_row": train_row,
        "is_hold": is_hold,
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
        "verdict": verdict,
        "n_tr": n_tr,
        "quote": quote,
        "started": started,
        "wall0": wall0,
    }


if __name__ == "__main__":
    run()
