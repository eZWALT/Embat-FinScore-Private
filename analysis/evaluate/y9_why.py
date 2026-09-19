"""Y9 why — mix shift vs outflow tail (Q3 turning / Q5 why).

Verdict (train group-fold): **CLOSE** Family M. Not an outflow tail, not a
usable mix shift. Best raw ``m_fin_share`` 0.618 is leak vs ``a_fin_cost``.
Best legal ``m_int_share`` 0.525 loses to ``a_out6`` 0.565. Merge **no**.

Brief: fee-interest pressure. Y9 GBM is PARK (own-p80 CV 0.554 loses to
``a_out6`` 0.565). Label stays. X never Family F / ``a_fin_cost``.
Family M is built **in memory** — not merged into parquet / FAMILIES.

No 0–100. No product/. No parquet rewrite. No new GBM. No build_targets.
Single-feature / stratified rates only. Holdout 72 is coverage only.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.y9_why
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
from analysis.features.catmix import MONTHLY_COLS, build as build_m
from analysis.features.common import ANALYSIS, DATA, connect
from analysis.targets.y11_dark import dark_population
from analysis.targets.y9_fees import META as Y9_META

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
OUT_MD = ANALYSIS / "outputs" / "y9_why.md"
OUT_PNG = ANALYSIS / "outputs" / "y9_fee_share_quintiles.png"
AGENT = "0511f2af"
WAVE = 4
ROUND = "R4"

Y_OWN = "y9_fee_r_ownp80"
Y_SPIKE = "y9_fee_spike"
Y2 = "y2_neg_2of3"
Y4 = "y4_ds_r_double"
N_FOLDS = 5
OUT6_BAR = 0.565
CLEAR_MARGIN = 0.02
LEAK_RHO = 0.80
SIZE_AUROC = 0.60
MIN_OWN_HIST = 6
OWN_P = 0.80
STORE_COLS = (
    "a_out6",
    "a_in3",
    "a_uncat_share",
    "a_fin_cost",
    "f_fc_r",
    "a_op_in",
    "a_out3",
)
M_EVAL = (
    "m_fee_share",
    "m_int_share",
    "m_fin_share",
    "m_fee_n_share",
    "m_uncat_share",
)
SINGLE_COLS = (
    "m_fee_share",
    "m_int_share",
    "m_fin_share",
    "m_fee_n_share",
    "a_out6",
    "a_out3",
    "log1p_a_in3",
    "a_op_in",
    "a_uncat_share",
    "m_uncat_share",
)
HI_COLS = ("a_out6", "m_fee_share", "m_int_share", "a_uncat_share")
LEAK_VS = ("a_fin_cost", "f_fc_r")


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


def add_own_p80(df: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    """Per-company expanding p80 (months ≤ t, min 6 finite). Never pooled, never holdout."""
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    extra = {}
    for c in cols:
        x = pd.to_numeric(out[c], errors="coerce")
        p80 = x.groupby(out["company_id"], sort=False).transform(
            lambda s: _expanding_quantile_skipna(s, OWN_P, MIN_OWN_HIST)
        )
        extra[f"{c}_ownp80"] = p80
        extra[f"{c}_hi"] = pd.Series(
            np.where(x.notna() & p80.notna(), (x > p80).astype(float), np.nan),
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
    need = ["company_id", "period", *STORE_COLS]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"store missing {missing}")
    if any(c.startswith("m_") for c in raw.columns):
        raise RuntimeError("store already has m_* — this module must not assume a merge")
    panel = _keys(raw[need])
    print(f"loaded store {STORE} shape={panel.shape} (no m_*)")
    return panel


def load_y() -> pd.DataFrame:
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(TARGETS)
    need = ["company_id", "period", Y_OWN, Y_SPIKE, Y2, Y4]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"targets.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"Y from {TARGETS} shape={panel.shape} (no build_targets)")
    return panel


def load_mix(con, grid: pd.DataFrame) -> pd.DataFrame:
    print("building Family M in memory — not writing parquet / FAMILIES")
    part = build_m(con, grid)
    part = _keys(part)
    keep = ["company_id", "period", *M_EVAL]
    extra_n = [c for c in ("m_fee_n_share", "m_uncat_share") if c in part.columns]
    keep = list(dict.fromkeys(keep + extra_n))
    missing = [c for c in M_EVAL if c not in part.columns]
    if missing:
        raise RuntimeError(f"catmix.build missing {missing}")
    print(f"mix in-memory shape={part.shape} monthly_cols={len(MONTHLY_COLS)}")
    return part[keep]


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
    """Share of Y9 positives that are also high outflow / mix / Y2 / Y4."""
    print("\n" + "=" * 72)
    print("PASS 1 — decompose Y9 positives (train labeled)")
    print("high = company own expanding p80 (min 6), never pooled, never holdout")
    print("=" * 72)

    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        n = int(lab[y_col].notna().sum())
        n_pos = int((lab[y_col] == 1).sum())
        rate = float(lab.loc[lab[y_col].notna(), y_col].mean()) if n else float("nan")
        print(f"\n{y_col}: n={n} pos={n_pos} rate={rate:.4f}")
        for flag in (
            "a_out6_hi",
            "m_fee_share_hi",
            "m_int_share_hi",
            "a_uncat_share_hi",
            Y2,
            Y4,
        ):
            rec = pos_overlap(lab, y_col, flag)
            rows.append(rec)
            print(
                f"  {flag:22s}  {rec['n_hi']:4d}/{rec['n_defined']:4d} defined pos  "
                f"share={rec['share_of_pos_defined']:.3f}  "
                f"cov={rec['coverage_of_pos']:.3f}"
            )

        # 2×2: high outflow vs high fee-share among own-p80 positives
        pos = lab[y_col] == 1
        both_def = pos & lab["a_out6_hi"].notna() & lab["m_fee_share_hi"].notna()
        ho = lab.loc[both_def, "a_out6_hi"] == 1
        hf = lab.loc[both_def, "m_fee_share_hi"] == 1
        n_b = int(both_def.sum())
        print(
            f"  2x2 out×fee among pos (n={n_b}): "
            f"out_only={int((ho & ~hf).sum())} "
            f"fee_only={int((~ho & hf).sum())} "
            f"both={int((ho & hf).sum())} "
            f"neither={int((~ho & ~hf).sum())}"
        )

    # story: spend tail if most pos are high a_out6; mix if fee/int without out
    own_rows = [r for r in rows if r["y"] == Y_OWN]
    out_sh = next(r["share_of_pos_defined"] for r in own_rows if r["flag"] == "a_out6_hi")
    fee_sh = next(r["share_of_pos_defined"] for r in own_rows if r["flag"] == "m_fee_share_hi")
    twos = {}
    for y_col in (Y_OWN, Y_SPIKE):
        pos = lab[y_col] == 1
        both_def = pos & lab["a_out6_hi"].notna() & lab["m_fee_share_hi"].notna()
        ho = lab.loc[both_def, "a_out6_hi"] == 1
        hf = lab.loc[both_def, "m_fee_share_hi"] == 1
        twos[y_col] = {
            "n": int(both_def.sum()),
            "out_only": int((ho & ~hf).sum()),
            "fee_only": int((~ho & hf).sum()),
            "both": int((ho & hf).sum()),
            "neither": int((~ho & ~hf).sum()),
        }
    neither_share = twos[Y_OWN]["neither"] / twos[Y_OWN]["n"] if twos[Y_OWN]["n"] else float("nan")
    if np.isfinite(out_sh) and out_sh >= 0.50 and (not np.isfinite(fee_sh) or fee_sh <= out_sh):
        story = "outflow_tail"
    elif np.isfinite(fee_sh) and fee_sh >= 0.50 and (not np.isfinite(out_sh) or fee_sh > out_sh + 0.10):
        story = "mix_shift"
    else:
        story = "mixed"
    print(f"  STORY (own-p80): {story}  out_share={out_sh:.3f} fee_share={fee_sh:.3f}")
    print(f"  neither cell share of 2x2 own-p80 pos: {neither_share:.3f}")
    return {
        "rows": rows,
        "story": story,
        "out_share": out_sh,
        "fee_share": fee_sh,
        "twos": twos,
        "neither_share": neither_share,
    }


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
            "tail_only": None,
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
    if len(rates) >= 3:
        body = rates[:-1]
        tail_only = rates[-1] > max(body) + 0.04 and max(body) - min(body) < 0.06
    else:
        tail_only = None
    return {
        "x": x_col,
        "y": y_col,
        "n": int(len(tr)),
        "rows": rows,
        "n_bins": len(rows),
        "monotone_up": monotone_up,
        "tail_only": tail_only,
        "bins": [float(b) for b in bins],
    }


def pass2_quintiles(df: pd.DataFrame, train_by_y: dict[str, pd.Series], no_plot: bool) -> dict:
    print("\n" + "=" * 72)
    print("PASS 2 — train-only quintiles (cuts never see holdout)")
    print("=" * 72)
    tables = []
    for x_col in ("m_fee_share", "m_int_share", "a_out6"):
        for y_col in (Y_OWN, Y_SPIKE):
            tab = quintile_table(df, train_by_y[y_col], x_col, y_col)
            tables.append(tab)
            print(
                f"  {x_col} vs {y_col}: n={tab['n']} bins={tab['n_bins']} "
                f"monotone={tab['monotone_up']} tail_only={tab['tail_only']}"
            )
            for r in tab["rows"]:
                print(
                    f"    Q{r['q']} {r['interval']} n={r['n']:4d} pos={r['n_pos']:3d} "
                    f"P(Y=1)={r['y_rate']:.3f} med={r['x_median']:.4g}"
                )

    png = None
    fee_own = next(t for t in tables if t["x"] == "m_fee_share" and t["y"] == Y_OWN)
    fee_spk = next(t for t in tables if t["x"] == "m_fee_share" and t["y"] == Y_SPIKE)
    if HAS_MPL and not no_plot and fee_own["rows"]:
        fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=False)
        for ax, tab, title in (
            (axes[0], fee_own, "y9_fee_r_ownp80"),
            (axes[1], fee_spk, "y9_fee_spike"),
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
                    r["y_rate"] + 0.004,
                    f"{r['y_rate']:.1%}\nn={r['n']}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
            ax.set_xticks(xs)
            ax.set_xticklabels([f"Q{r['q']}" for r in tab["rows"]])
            ax.set_ylabel("P(Y=1)")
            ax.set_xlabel("m_fee_share quintile (train cuts)")
            ax.set_ylim(0, max(ys) * 1.32 if ys else 1)
            ax.legend(frameon=False, fontsize=7)
            ax.set_title(title)
        fig.suptitle("Y9 rate by fee activity-share quintile (train cuts; Q1 is mostly zero)", fontsize=11)
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


def pass3_singles(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    print("\n" + "=" * 72)
    print("PASS 3 — single-feature train group-fold AUROC")
    print(f"KEEP mix as Q5 if CV >= a_out6 + {CLEAR_MARGIN:.2f} and not an F-copy")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        print(f"\n{y_col} n_lab={int(lab.sum())}")
        for col in SINGLE_COLS:
            if col not in df.columns:
                print(f"  missing {col}")
                continue
            leak = leakage_check(
                [col],
                y_col,
                forbidden_prefixes=Y9_META.get("forbidden_x_prefixes"),
            )
            if not leak["ok"] and col not in LEAK_VS:
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
                f"  {col:18s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f}  sign={oof['train_sign']:+d}  "
                f"cov={cov['coverage']:.3f}  sizeAUC={size_bin['auroc']:.3f} "
                f"ρ={size_bin['rho']:+.3f}{park}"
            )
    return {"rows": rows}


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
    return {"auroc": float(auc2) if np.isfinite(auc2) else float("nan"), "rho": spearman(x[m], np.log1p(size[m].abs()))}


def pass4_leak(df: pd.DataFrame, train_any: pd.Series) -> dict:
    print("\n" + "=" * 72)
    print("PASS 4 — leak screen vs a_fin_cost / f_fc_r  (fail |ρ|≥0.80)")
    print("comparators only — never X")
    print("=" * 72)
    rows = []
    for col in M_EVAL + ("a_out6", "a_uncat_share"):
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
                f"  {col:16s} vs {vs:12s}  ρ_s={rec['spearman']:+.3f}  "
                f"ρ_p={rec['pearson']:+.3f}  n={rec['n']}{mark}"
            )
    fee_int = {
        "spearman": spearman(
            pd.to_numeric(df.loc[train_any, "m_fee_share"], errors="coerce"),
            pd.to_numeric(df.loc[train_any, "m_int_share"], errors="coerce"),
        ),
        "pearson": pearson(
            pd.to_numeric(df.loc[train_any, "m_fee_share"], errors="coerce"),
            pd.to_numeric(df.loc[train_any, "m_int_share"], errors="coerce"),
        ),
    }
    print(
        f"  m_fee_share vs m_int_share  ρ_s={fee_int['spearman']:+.3f}  "
        f"ρ_p={fee_int['pearson']:+.3f}  (substitutes if |ρ| high)"
    )
    return {"rows": rows, "fee_vs_int": fee_int}


def pass5_dark(df: pd.DataFrame, train_by_y: dict[str, pd.Series], dark_ids: set[str]) -> dict:
    """Y9 base on 470 dark vs 744 invoiced. Never D/E as X."""
    print("\n" + "=" * 72)
    print("PASS 5 — 470 dark vs 744 invoiced  (rates only; never D/E as X)")
    print("=" * 72)
    is_dark = df["company_id"].astype(str).isin(dark_ids)
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        for name, sl in (
            ("train_all", lab),
            ("dark_470", lab & is_dark),
            ("invoiced_744", lab & ~is_dark),
        ):
            y = pd.to_numeric(df.loc[sl, y_col], errors="coerce")
            rec = {
                "y": y_col,
                "slice": name,
                "n": int(y.notna().sum()),
                "n_pos": int((y == 1).sum()),
                "n_cos": int(df.loc[sl & y.notna(), "company_id"].nunique()),
                "rate": float(y.mean()) if int(y.notna().sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"  {y_col:20s} {name:14s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
                f"cos={rec['n_cos']:4d} rate={rec['rate']:.4f}"
            )
    return {"rows": rows, "n_dark": len(dark_ids)}


def pass6_lag1(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Honest 1-month Q6: lag1 of fee/int share vs contemporaneous. Not t3."""
    print("\n" + "=" * 72)
    print("PASS 6 — Q6 clock  lag1 of m_fee_share / m_int_share (honest 1m only)")
    print("t3 mix already CLOSE (overlap). Do not claim a fee lead from t3.")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        for stem in ("m_fee_share", "m_int_share", "m_fee_n_share"):
            for lag, col in ((0, stem), (1, f"{stem}_lag1")):
                if col not in df.columns:
                    continue
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
                }
                rows.append(rec)
                print(
                    f"  {y_col:20s} {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                    f"train={oof['train_auc']:.4f}  sign={oof['train_sign']:+d}  "
                    f"cov={cov['coverage']:.3f}"
                )
    return {"rows": rows}


def pass7_zero_leak(df: pd.DataFrame, train_any: pd.Series) -> dict:
    """Is Spearman 0.83 vs a_fin_cost just zeros lining up?"""
    print("\n" + "=" * 72)
    print("CUT 7 — zero-inflation leak  (ρ on support where both > 0)")
    print("=" * 72)
    rows = []
    for col in ("m_fee_share", "m_int_share", "m_fin_share", "m_fee_n_share"):
        if col not in df.columns:
            continue
        x = pd.to_numeric(df[col], errors="coerce")
        for vs in LEAK_VS:
            other = pd.to_numeric(df[vs], errors="coerce")
            both = train_any & x.notna() & other.notna()
            pos = both & (x > 0) & (other > 0)
            rec = {
                "col": col,
                "vs": vs,
                "n_all": int(both.sum()),
                "n_pos": int(pos.sum()),
                "rho_all": spearman(x[both], other[both]),
                "rho_pos": spearman(x[pos], other[pos]),
                "share_x_zero": float((x[both] == 0).mean()) if int(both.sum()) else float("nan"),
                "share_vs_zero": float((other[both] == 0).mean()) if int(both.sum()) else float("nan"),
            }
            rec["fail_all"] = bool(np.isfinite(rec["rho_all"]) and abs(rec["rho_all"]) >= LEAK_RHO)
            rec["fail_pos"] = bool(np.isfinite(rec["rho_pos"]) and abs(rec["rho_pos"]) >= LEAK_RHO)
            rows.append(rec)
            print(
                f"  {col:16s} vs {vs:12s}  all ρ={rec['rho_all']:+.3f} (n={rec['n_all']})  "
                f"support ρ={rec['rho_pos']:+.3f} (n={rec['n_pos']})  "
                f"x=0 {rec['share_x_zero']:.2%} vs=0 {rec['share_vs_zero']:.2%}  "
                f"fail_all={rec['fail_all']} fail_pos={rec['fail_pos']}"
            )
    # any-fee vs any-fin-cost: is the binary a rewrite?
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    m = train_any & fee.notna() & fin.notna()
    any_fee = (fee[m] > 0).astype(float)
    any_fin = (fin[m] > 0).astype(float)
    rec_bin = {
        "n": int(m.sum()),
        "auroc_anyfee_vs_anyfin": float(auroc(any_fin, any_fee)),
        "agree": float((any_fee == any_fin).mean()),
        "fee_only": int(((any_fee == 1) & (any_fin == 0)).sum()),
        "fin_only": int(((any_fee == 0) & (any_fin == 1)).sum()),
    }
    print(
        f"  any-fee vs any-a_fin_cost: AUROC={rec_bin['auroc_anyfee_vs_anyfin']:.3f} "
        f"agree={rec_bin['agree']:.3f} fee_only={rec_bin['fee_only']} "
        f"fin_only={rec_bin['fin_only']} (interest-only months)"
    )
    return {"rows": rows, "binary": rec_bin}


def pass8_any_fee(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is 0.613 just 'had a fee this month' vs mix intensity?"""
    print("\n" + "=" * 72)
    print("CUT 8 — any-fee flag vs continuous m_fee_share / fee>0 support")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    df = df.copy()
    df["_any_fee"] = pd.Series(np.where(fee.notna(), (fee > 0).astype(float), np.nan), index=df.index)
    rows = []
    rates = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        for name, sl in (
            ("fee_eq_0", lab & (fee == 0)),
            ("fee_gt_0", lab & (fee > 0)),
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
            print(f"  {y_col:20s} {name:10s} n={rec['n']:5d} pos={rec['n_pos']:4d} rate={rec['rate']:.4f}")
        for col in ("_any_fee", "m_fee_share"):
            oof = signed_oof_auroc(df, col, y_col, lab, folds)
            rec = {
                "y": y_col,
                "col": col,
                "slice": "all_labeled",
                "cv": oof["cv"],
                "sd": oof["sd"],
                "train_auc": oof["train_auc"],
                "n_defined": int((lab & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
            }
            rows.append(rec)
            print(
                f"  {y_col:20s} {col:16s} all  CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f}"
            )
        # continuous share only among fee>0
        pos_lab = lab & (fee > 0)
        oof_p = signed_oof_auroc(df, "m_fee_share", y_col, pos_lab, folds)
        rec = {
            "y": y_col,
            "col": "m_fee_share",
            "slice": "fee_gt_0",
            "cv": oof_p["cv"],
            "sd": oof_p["sd"],
            "train_auc": oof_p["train_auc"],
            "n_defined": int(pos_lab.sum()),
        }
        rows.append(rec)
        print(
            f"  {y_col:20s} m_fee_share      fee>0 CV={oof_p['cv']:.4f}±{oof_p['sd']:.3f}  "
            f"n={int(pos_lab.sum())} pos={int((y[pos_lab] == 1).sum())}"
        )
    return {"rows": rows, "rates": rates}


def pass9_outflow_residual(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Does mix still rank Y9 after removing the spend tail?"""
    print("\n" + "=" * 72)
    print("CUT 9 — m_fee_share OOF inside a_out6 terciles / not-high-outflow")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    out = pd.to_numeric(df["a_out6"], errors="coerce")
    hi = pd.to_numeric(df["a_out6_hi"], errors="coerce")
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        tr = lab & out.notna()
        assert_no_holdout(df.loc[tr, "company_id"])
        cats, bins = pd.qcut(out[tr], 3, retbins=True, duplicates="drop")
        terc = pd.Series(pd.cut(out, bins=bins, include_lowest=True), index=df.index)
        print(f"  {y_col} a_out6 tercile edges (train labeled): {bins.tolist()}")
        for i, q in enumerate(sorted(terc.dropna().unique()), start=1):
            sl = lab & (terc == q)
            oof = signed_oof_auroc(df, "m_fee_share", y_col, sl, folds)
            oof_out = signed_oof_auroc(df, "a_out6", y_col, sl, folds)
            y = pd.to_numeric(df.loc[sl, y_col], errors="coerce")
            rec = {
                "y": y_col,
                "slice": f"out_T{i}",
                "n": int(sl.sum()),
                "n_pos": int((y == 1).sum()),
                "rate": float(y.mean()) if int(sl.sum()) else float("nan"),
                "fee_cv": oof["cv"],
                "fee_sd": oof["sd"],
                "out_cv": oof_out["cv"],
            }
            rows.append(rec)
            print(
                f"  {y_col:20s} T{i} n={rec['n']:5d} pos={rec['n_pos']:4d} "
                f"rate={rec['rate']:.3f}  fee CV={oof['cv']:.3f}  out CV={oof_out['cv']:.3f}"
            )
        not_hi = lab & (hi == 0)
        oof_n = signed_oof_auroc(df, "m_fee_share", y_col, not_hi, folds)
        y_n = pd.to_numeric(df.loc[not_hi, y_col], errors="coerce")
        rec = {
            "y": y_col,
            "slice": "not_high_out",
            "n": int(not_hi.sum()),
            "n_pos": int((y_n == 1).sum()),
            "rate": float(y_n.mean()) if int(not_hi.sum()) else float("nan"),
            "fee_cv": oof_n["cv"],
            "fee_sd": oof_n["sd"],
            "out_cv": float("nan"),
        }
        rows.append(rec)
        print(
            f"  {y_col:20s} not-high-out n={rec['n']:5d} pos={rec['n_pos']:4d} "
            f"rate={rec['rate']:.3f}  fee CV={oof_n['cv']:.3f}"
        )
    return {"rows": rows}


def pass10_neither(lab: pd.DataFrame) -> dict:
    """What sits in the 2×2 neither cell (not high out, not high fee-share)?"""
    print("\n" + "=" * 72)
    print("CUT 10 — neither cell (own-p80 pos, not high out, not high fee-share)")
    print("=" * 72)
    y_col = Y_OWN
    pos = lab[y_col] == 1
    both = pos & lab["a_out6_hi"].notna() & lab["m_fee_share_hi"].notna()
    neither = both & (lab["a_out6_hi"] == 0) & (lab["m_fee_share_hi"] == 0)
    n = int(neither.sum())
    print(f"  neither n={n} / 2x2 pos {int(both.sum())}")
    flags = {
        "m_int_share_hi": lab["m_int_share_hi"],
        "a_uncat_share_hi": lab["a_uncat_share_hi"],
        Y2: lab[Y2],
        Y4: lab[Y4],
        "any_fee": (pd.to_numeric(lab["m_fee_share"], errors="coerce") > 0).astype(float),
        "any_int": (pd.to_numeric(lab["m_int_share"], errors="coerce") > 0).astype(float),
    }
    rows = []
    for name, s in flags.items():
        s = pd.to_numeric(s, errors="coerce")
        d = neither & s.notna()
        rec = {
            "flag": name,
            "n_defined": int(d.sum()),
            "n_hi": int((d & (s == 1)).sum()),
            "share": float(s[d].mean()) if int(d.sum()) else float("nan"),
        }
        rows.append(rec)
        print(f"  {name:20s} {rec['n_hi']:4d}/{rec['n_defined']:4d}  share={rec['share']:.3f}")
    # own-p80 of fee-share is near 0 — "high" is any crumb
    p80 = pd.to_numeric(lab.loc[pos, "m_fee_share_ownp80"], errors="coerce")
    print(
        f"  own-p80 of m_fee_share among Y9 pos: "
        f"p50={float(p80.median()):.4g} p90={float(p80.quantile(0.9)):.4g} "
        f"share_eq0={float((p80 == 0).mean()):.3f} n={int(p80.notna().sum())}"
    )
    return {
        "n": n,
        "n_2x2": int(both.sum()),
        "rows": rows,
        "fee_p80_med": float(p80.median()) if int(p80.notna().sum()) else float("nan"),
        "fee_p80_p90": float(p80.quantile(0.9)) if int(p80.notna().sum()) else float("nan"),
        "fee_p80_zero": float((p80 == 0).mean()) if int(p80.notna().sum()) else float("nan"),
    }


def pass11_dark_mix(df: pd.DataFrame, train_by_y: dict[str, pd.Series], pop: dict) -> dict:
    """360 all-dark vs 110 mixed-dark Y9 bases. Never D/E as X."""
    print("\n" + "=" * 72)
    print("CUT 11 — 360 all-dark vs 110 mixed-dark  (never D/E as X)")
    print("=" * 72)
    dark = pop["train_dark_frame"]
    mix_of = dark.set_index("company_id")["mix"]
    sl_360 = df["company_id"].map(mix_of) == "all_dark"
    sl_110 = df["company_id"].map(mix_of) == "mixed"
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        for name, sl in (("all_dark_360", lab & sl_360), ("mixed_dark_110", lab & sl_110)):
            y = pd.to_numeric(df.loc[sl, y_col], errors="coerce")
            rec = {
                "y": y_col,
                "slice": name,
                "n": int(y.notna().sum()),
                "n_pos": int((y == 1).sum()),
                "n_cos": int(df.loc[sl & y.notna(), "company_id"].nunique()),
                "rate": float(y.mean()) if int(y.notna().sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"  {y_col:20s} {name:16s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
                f"cos={rec['n_cos']:4d} rate={rec['rate']:.4f}"
            )
    return {"rows": rows}


def pass12_label_overlap(df: pd.DataFrame, train_row: pd.Series, is_hold: pd.Series) -> dict:
    """own-p80 ∩ spike; holdout coverage of mix (not a claim)."""
    print("\n" + "=" * 72)
    print("CUT 12 — label overlap + holdout mix coverage (LOW_POWER)")
    print("=" * 72)
    own = pd.to_numeric(df[Y_OWN], errors="coerce")
    spk = pd.to_numeric(df[Y_SPIKE], errors="coerce")
    both = train_row & own.notna() & spk.notna()
    rec = {
        "n_both_lab": int(both.sum()),
        "n_own_only": int((train_row & (own == 1) & (spk == 0)).sum()),
        "n_spk_only": int((train_row & (own == 0) & (spk == 1)).sum()),
        "n_both_pos": int((train_row & (own == 1) & (spk == 1)).sum()),
        "rho": spearman(own[both], spk[both]),
    }
    print(
        f"  train both-labeled n={rec['n_both_lab']}  both-pos={rec['n_both_pos']}  "
        f"own-only={rec['n_own_only']} spike-only={rec['n_spk_only']}  "
        f"ρ={rec['rho']:+.3f}"
    )
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    ho = is_hold
    rec["hold_cm"] = int(ho.sum())
    rec["hold_fee_cov"] = float(fee[ho].notna().mean()) if int(ho.sum()) else float("nan")
    rec["hold_own_n"] = int((ho & own.notna()).sum())
    rec["hold_own_pos"] = int((ho & (own == 1)).sum())
    rec["hold_own_fee_cov"] = (
        float(fee[ho & own.notna()].notna().mean()) if rec["hold_own_n"] else float("nan")
    )
    print(
        f"  holdout mix coverage {rec['hold_fee_cov']:.3f} (cm={rec['hold_cm']}); "
        f"own-p80 labeled {rec['hold_own_n']} pos={rec['hold_own_pos']} "
        f"fee defined {rec['hold_own_fee_cov']:.3f} LOW_POWER"
    )
    # train-sign holdout AUROC — coverage diagnostic, not a KEEP
    sign = choose_sign(own[train_row & own.notna()], fee[train_row & own.notna()])
    rec["hold_fee_auroc"] = float(auroc(own[ho], sign * fee[ho]))
    rec["hold_sign"] = int(sign)
    print(
        f"  holdout m_fee_share AUROC={rec['hold_fee_auroc']:.3f} "
        f"sign={sign:+d} LOW_POWER not a claim"
    )
    return rec


def pass13_count_honesty(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is m_fee_n_share mix intensity or a fee-presence rewrite of a_fin_cost>0?"""
    print("\n" + "=" * 72)
    print("CUT 13 — m_fee_n_share honesty (presence rewrite vs ticket intensity)")
    print("any-fin-cost is a comparator only — never X")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    fee_n = pd.to_numeric(df["m_fee_n_share"], errors="coerce") if "m_fee_n_share" in df.columns else None
    fee_a = pd.to_numeric(df["m_fee_share"], errors="coerce")
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    df = df.copy()
    df["_any_fin"] = pd.Series(np.where(fin.notna(), (fin > 0).astype(float), np.nan), index=df.index)
    df["_any_fee"] = pd.Series(np.where(fee_a.notna(), (fee_a > 0).astype(float), np.nan), index=df.index)
    rows = []
    lab = train_by_y[Y_OWN]
    rho_an = spearman(fee_n[lab], fee_a[lab]) if fee_n is not None else float("nan")
    print(f"  train labeled Spearman m_fee_n_share vs m_fee_share: {rho_an:+.3f}")

    for col in ("_any_fin", "_any_fee", "m_fee_n_share", "m_fee_share"):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, Y_OWN, lab, folds)
        rec = {
            "col": col,
            "slice": "all",
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "n": int((lab & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
        }
        rows.append(rec)
        print(
            f"  {col:16s} all  CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"train={oof['train_auc']:.4f}  n={rec['n']}"
        )

    pos = lab & (fee_a > 0)
    for col in ("m_fee_n_share", "m_fee_share"):
        if col not in df.columns:
            continue
        oof = signed_oof_auroc(df, col, Y_OWN, pos, folds)
        rec = {
            "col": col,
            "slice": "fee_gt_0",
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "n": int(pos.sum()),
        }
        rows.append(rec)
        print(
            f"  {col:16s} fee>0 CV={oof['cv']:.4f}±{oof['sd']:.3f}  n={rec['n']}"
        )

    any_fin = next(r for r in rows if r["col"] == "_any_fin" and r["slice"] == "all")
    nshare = next((r for r in rows if r["col"] == "m_fee_n_share" and r["slice"] == "all"), None)
    nshare_pos = next((r for r in rows if r["col"] == "m_fee_n_share" and r["slice"] == "fee_gt_0"), None)
    gap_vs_presence = (
        float(nshare["cv"] - any_fin["cv"])
        if nshare and np.isfinite(nshare["cv"]) and np.isfinite(any_fin["cv"])
        else float("nan")
    )
    presence_rewrite = bool(
        np.isfinite(gap_vs_presence) and gap_vs_presence < CLEAR_MARGIN
    )
    intensity_dead = bool(
        nshare_pos and np.isfinite(nshare_pos["cv"]) and nshare_pos["cv"] < OUT6_BAR
    )
    print(
        f"  gap vs any-fin-cost presence: {gap_vs_presence:+.3f}  "
        f"presence_rewrite={presence_rewrite}  intensity_dead={intensity_dead}"
    )
    return {
        "rows": rows,
        "rho_n_vs_amt": rho_an,
        "gap_vs_presence": gap_vs_presence,
        "presence_rewrite": presence_rewrite,
        "intensity_dead": intensity_dead,
        "any_fin_cv": any_fin["cv"],
        "nshare_cv": nshare["cv"] if nshare else float("nan"),
        "nshare_pos_cv": nshare_pos["cv"] if nshare_pos else float("nan"),
    }


def pass14_presence_clock(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """any-fin lag1 (comparator) + a_out6 quintile × any-fee rates."""
    print("\n" + "=" * 72)
    print("CUT 14 — any-fin lag1 clock + outflow quintile × any-fee")
    print("any-fin is a comparator (that is the Y). Never X.")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    out = pd.to_numeric(df["a_out6"], errors="coerce")
    df = df.copy()
    df["_any_fin"] = pd.Series(np.where(fin.notna(), (fin > 0).astype(float), np.nan), index=df.index)
    df = add_lags(df, ["_any_fin"], (1,))
    clock = []
    lab = train_by_y[Y_OWN]
    for col in ("_any_fin", "_any_fin_lag1"):
        oof = signed_oof_auroc(df, col, Y_OWN, lab, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "folds": [f["auroc"] for f in oof["folds"]],
        }
        clock.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(
            f"  {col:18s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"train={oof['train_auc']:.4f}  folds {bits}"
        )

    # a_out6 quintiles × any-fee (train labeled own-p80)
    tr = lab & out.notna() & fee.notna()
    assert_no_holdout(df.loc[tr, "company_id"])
    cats, bins = pd.qcut(out[tr], 5, retbins=True, duplicates="drop")
    work = pd.DataFrame(
        {
            "q": cats,
            Y_OWN: pd.to_numeric(df.loc[tr, Y_OWN], errors="coerce"),
            "any_fee": (fee[tr] > 0).astype(float),
        }
    )
    cross = []
    print("  a_out6 Q × any-fee  P(Y=1)")
    for i, (q, g) in enumerate(work.groupby("q", observed=True), start=1):
        for fee_on, sl in ((0, g[g["any_fee"] == 0]), (1, g[g["any_fee"] == 1])):
            rec = {
                "q": i,
                "any_fee": fee_on,
                "n": int(len(sl)),
                "n_pos": int((sl[Y_OWN] == 1).sum()),
                "rate": float(sl[Y_OWN].mean()) if len(sl) else float("nan"),
            }
            cross.append(rec)
        z = next(r for r in cross if r["q"] == i and r["any_fee"] == 0)
        o = next(r for r in cross if r["q"] == i and r["any_fee"] == 1)
        print(
            f"    Q{i} no-fee n={z['n']:4d} rate={z['rate']:.3f}  |  "
            f"fee n={o['n']:4d} rate={o['rate']:.3f}"
        )

    # fold table for a_out6 / m_fee_share on own-p80
    fold_cmp = []
    for col in ("a_out6", "m_fee_share"):
        oof = signed_oof_auroc(df, col, Y_OWN, lab, folds)
        fold_cmp.append(
            {
                "col": col,
                "cv": oof["cv"],
                "sd": oof["sd"],
                "folds": [f["auroc"] for f in oof["folds"]],
            }
        )
        bits = " ".join(f"{a:.3f}" for a in fold_cmp[-1]["folds"])
        print(f"  {col:18s} folds {bits}  cv={oof['cv']:.3f}")

    lag1 = next(r for r in clock if r["col"] == "_any_fin_lag1")
    now = next(r for r in clock if r["col"] == "_any_fin")
    q6 = bool(np.isfinite(lag1["cv"]) and lag1["cv"] >= OUT6_BAR + CLEAR_MARGIN)
    print(
        f"  any-fin lag1 CV={lag1['cv']:.3f} vs now {now['cv']:.3f} vs a_out6 {OUT6_BAR}  "
        f"honest-Q6 KEEP={q6}"
    )
    return {
        "clock": clock,
        "cross": cross,
        "fold_cmp": fold_cmp,
        "lag1_cv": lag1["cv"],
        "now_cv": now["cv"],
        "q6_keep": q6,
        "bins": [float(b) for b in bins],
    }


def pass15_interest_only(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """486 interest-without-fee months: does Y9 care which fin_cost token?"""
    print("\n" + "=" * 72)
    print("CUT 15 — fee vs interest-only vs neither (train labeled)")
    print("=" * 72)
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    inte = pd.to_numeric(df["m_int_share"], errors="coerce")
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        slices = {
            "neither": lab & (fee == 0) & (inte == 0),
            "fee_only": lab & (fee > 0) & (inte == 0),
            "int_only": lab & (fee == 0) & (inte > 0),
            "both_tokens": lab & (fee > 0) & (inte > 0),
        }
        for name, sl in slices.items():
            yy = y[sl]
            rec = {
                "y": y_col,
                "slice": name,
                "n": int(yy.notna().sum()),
                "n_pos": int((yy == 1).sum()),
                "rate": float(yy.mean()) if int(yy.notna().sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"  {y_col:20s} {name:12s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
                f"rate={rec['rate']:.4f}"
            )
    return {"rows": rows}


def pass16_dark_presence(
    df: pd.DataFrame, train_by_y: dict[str, pd.Series], pop: dict
) -> dict:
    """Is the 110 mixed-dark spike gap just more fee/interest presence?"""
    print("\n" + "=" * 72)
    print("CUT 16 — 360 vs 110 × any-fin  (never D/E as X)")
    print("=" * 72)
    dark = pop["train_dark_frame"]
    mix_of = dark.set_index("company_id")["mix"]
    sl_360 = df["company_id"].map(mix_of) == "all_dark"
    sl_110 = df["company_id"].map(mix_of) == "mixed"
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    any_fin = fin > 0
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        for popn, sl_pop in (("all_dark_360", sl_360), ("mixed_dark_110", sl_110)):
            for name, sl in (
                (f"{popn}_nofin", lab & sl_pop & ~any_fin & fin.notna()),
                (f"{popn}_anyfin", lab & sl_pop & any_fin),
            ):
                yy = y[sl]
                rec = {
                    "y": y_col,
                    "slice": name,
                    "n": int(yy.notna().sum()),
                    "n_pos": int((yy == 1).sum()),
                    "rate": float(yy.mean()) if int(yy.notna().sum()) else float("nan"),
                }
                rows.append(rec)
                print(
                    f"  {y_col:20s} {name:22s} n={rec['n']:5d} pos={rec['n_pos']:4d} "
                    f"rate={rec['rate']:.4f}"
                )
    return {"rows": rows}


def pass17_company_trait(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Company-level: is Y9 a always-fee trait or a turning month?"""
    print("\n" + "=" * 72)
    print("CUT 17 — company-level any-fin rate vs Y9 rate (Q3 honesty)")
    print("=" * 72)
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        work = pd.DataFrame(
            {
                "company_id": df.loc[lab, "company_id"].astype(str),
                "y": pd.to_numeric(df.loc[lab, y_col], errors="coerce"),
                "any_fin": (fin[lab] > 0).astype(float),
            }
        ).dropna()
        g = work.groupby("company_id", sort=False).agg(
            n=("y", "size"),
            y_rate=("y", "mean"),
            fin_rate=("any_fin", "mean"),
            n_pos=("y", "sum"),
        )
        g = g[g["n"] >= 3]
        rho = spearman(g["fin_rate"], g["y_rate"])
        ever_y = g["n_pos"] > 0
        rec = {
            "y": y_col,
            "n_cos": int(len(g)),
            "n_ever_pos": int(ever_y.sum()),
            "rho": rho,
            "fin_rate_pos": float(g.loc[ever_y, "fin_rate"].mean()) if int(ever_y.sum()) else float("nan"),
            "fin_rate_neg": float(g.loc[~ever_y, "fin_rate"].mean()) if int((~ever_y).sum()) else float("nan"),
            "y_rate_always_fin": float(g.loc[g["fin_rate"] >= 0.99, "y_rate"].mean())
            if int((g["fin_rate"] >= 0.99).sum())
            else float("nan"),
            "y_rate_never_fin": float(g.loc[g["fin_rate"] <= 0.01, "y_rate"].mean())
            if int((g["fin_rate"] <= 0.01).sum())
            else float("nan"),
            "n_always_fin": int((g["fin_rate"] >= 0.99).sum()),
            "n_never_fin": int((g["fin_rate"] <= 0.01).sum()),
        }
        rows.append(rec)
        print(
            f"  {y_col}: n_cos={rec['n_cos']} ever_pos={rec['n_ever_pos']}  "
            f"ρ(fin_rate, y_rate)={rho:+.3f}  "
            f"fin_rate pos/neg cos {rec['fin_rate_pos']:.2f}/{rec['fin_rate_neg']:.2f}  "
            f"Y rate always-fin (n={rec['n_always_fin']})={rec['y_rate_always_fin']:.3f}  "
            f"never-fin (n={rec['n_never_fin']})={rec['y_rate_never_fin']:.3f}"
        )
    return {"rows": rows}


def pass18_always_fin_intensity(
    df: pd.DataFrame, train_by_y: dict[str, pd.Series]
) -> dict:
    """Among always-fin companies, does mix intensity still beat a_out6?"""
    print("\n" + "=" * 72)
    print("CUT 18 — m_fee_share OOF on always-fin companies only")
    print("presence is saturated; leftover would be true mix intensity")
    print("=" * 72)
    folds = df["fold"].to_numpy()
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    by_y = {}
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        work = pd.DataFrame(
            {
                "company_id": df.loc[lab, "company_id"].astype(str),
                "any_fin": (fin[lab] > 0).astype(float),
            }
        ).dropna()
        fin_rate = work.groupby("company_id", sort=False)["any_fin"].mean()
        always = set(fin_rate[fin_rate >= 0.99].index.astype(str))
        sl = lab & df["company_id"].astype(str).isin(always)
        assert_no_holdout(df.loc[sl, "company_id"])
        y = pd.to_numeric(df.loc[sl, y_col], errors="coerce")
        print(
            f"  {y_col} always-fin cos={len(always)} months={int(sl.sum())} "
            f"pos={int((y == 1).sum())} rate={float(y.mean()) if int(sl.sum()) else float('nan'):.4f}"
        )
        rows = []
        for col in ("m_fee_share", "m_fee_n_share", "m_int_share", "a_out6", "log1p_a_in3"):
            if col not in df.columns:
                continue
            oof = signed_oof_auroc(df, col, y_col, sl, folds)
            rec = {
                "y": y_col,
                "col": col,
                "cv": oof["cv"],
                "sd": oof["sd"],
                "train_auc": oof["train_auc"],
                "n": int((sl & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
            }
            rows.append(rec)
            print(
                f"    {col:16s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f}  n={rec['n']}"
            )
        fee = next((r for r in rows if r["col"] == "m_fee_share"), None)
        out = next((r for r in rows if r["col"] == "a_out6"), None)
        gap = (
            float(fee["cv"] - out["cv"])
            if fee and out and np.isfinite(fee["cv"]) and np.isfinite(out["cv"])
            else float("nan")
        )
        keep = bool(
            fee
            and np.isfinite(fee["cv"])
            and fee["cv"] >= OUT6_BAR + CLEAR_MARGIN
            and np.isfinite(gap)
            and gap >= CLEAR_MARGIN
        )
        print(f"    gap={gap:+.3f} KEEP_intensity={keep}")
        by_y[y_col] = {
            "rows": rows,
            "n_cos": len(always),
            "n": int(sl.sum()),
            "n_pos": int((y == 1).sum()),
            "rate": float(y.mean()) if int(sl.sum()) else float("nan"),
            "gap": gap,
            "keep_intensity": keep,
        }
    own = by_y[Y_OWN]
    return {
        "by_y": by_y,
        "rows": own["rows"],
        "n_cos": own["n_cos"],
        "n": own["n"],
        "n_pos": own["n_pos"],
        "rate": own["rate"],
        "gap": own["gap"],
        "keep_intensity": own["keep_intensity"],
    }


def pass19_fee_x_out(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Fee-share bins × outflow quintiles — is any leftover mix inside spend?"""
    print("\n" + "=" * 72)
    print("CUT 19 — fee-share bins × a_out6 quintiles (train labeled)")
    print("fee bins collapse (zeros); leftover would be intensity inside spend")
    print("=" * 72)
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    out = pd.to_numeric(df["a_out6"], errors="coerce")
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        m = lab & fee.notna() & out.notna()
        assert_no_holdout(df.loc[m, "company_id"])
        y = pd.to_numeric(df.loc[m, y_col], errors="coerce")
        out_q = pd.qcut(out[m], 5, duplicates="drop")
        # zeros dominate fee-share; use no-fee / mid / top among fee>0
        fee_bin = pd.Series(np.where(fee[m] <= 0, "no_fee", "fee"), index=df.index[m])
        pos_fee = fee[m] > 0
        if int(pos_fee.sum()) >= 30 and fee[m][pos_fee].nunique() >= 2:
            mid = float(fee[m][pos_fee].median())
            fee_bin = pd.Series(
                np.where(
                    fee[m] <= 0,
                    "no_fee",
                    np.where(fee[m] > mid, "fee_hi", "fee_lo"),
                ),
                index=df.index[m],
            )
        work = pd.DataFrame(
            {"y": y.to_numpy(), "out_q": out_q.astype(str).to_numpy(), "fee_bin": fee_bin.to_numpy()}
        )
        print(f"  {y_col}")
        for (oq, fb), g in work.groupby(["out_q", "fee_bin"], sort=True):
            rec = {
                "y": y_col,
                "out_q": str(oq),
                "fee_bin": str(fb),
                "n": int(len(g)),
                "n_pos": int((g["y"] == 1).sum()),
                "rate": float(g["y"].mean()) if len(g) else float("nan"),
            }
            rows.append(rec)
            print(
                f"    out {oq}  {fb:7s}  n={rec['n']:4d}  "
                f"pos={rec['n_pos']:4d}  rate={rec['rate']:.3f}"
            )
        # within-outflow: fee_hi vs no_fee gap
        for oq, g in work.groupby("out_q", sort=True):
            z = g[g["fee_bin"] == "no_fee"]
            h = g[g["fee_bin"] == "fee_hi"] if "fee_hi" in set(g["fee_bin"]) else g[g["fee_bin"] == "fee"]
            if len(z) and len(h):
                gap = float(h["y"].mean() - z["y"].mean())
                print(f"    out {oq}  fee_hi−no_fee gap={gap:+.3f}")
    return {"rows": rows}


def _onset_masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    work = df[["company_id", "period"]].copy()
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    work["any_fin"] = pd.Series(
        np.where(fin.notna(), (fin > 0).astype(float), np.nan), index=df.index
    )
    first = (
        work.loc[work["any_fin"] == 1]
        .groupby("company_id", sort=False)["period"]
        .min()
    )
    work["first_fin"] = work["company_id"].map(first)
    return {
        "onset_first_fin": (work["any_fin"] == 1)
        & (work["period"] == work["first_fin"]),
        "later_fin": (work["any_fin"] == 1)
        & work["first_fin"].notna()
        & (work["period"] > work["first_fin"]),
        "later_off": (work["any_fin"] == 0)
        & work["first_fin"].notna()
        & (work["period"] > work["first_fin"]),
        "before_first_fin": work["first_fin"].notna()
        & (work["period"] < work["first_fin"])
        & work["any_fin"].notna(),
        "never_fin": work["first_fin"].isna() & work["any_fin"].notna(),
    }


def pass20_onset(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """First any-fin month vs later-fin vs later-off (turning-on vs trait)."""
    print("\n" + "=" * 72)
    print("CUT 20 — first-fin onset vs later-fin vs later-off (train labeled)")
    print("Y9 is t+1..t+3; onset at t is almost algebraic for spike. Never X.")
    print("=" * 72)
    slices = _onset_masks(df)
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        y = pd.to_numeric(df[y_col], errors="coerce")
        print(f"  {y_col}")
        for name, sl in slices.items():
            m = lab & sl
            assert_no_holdout(df.loc[m, "company_id"])
            yy = y[m]
            rec = {
                "y": y_col,
                "slice": name,
                "n": int(m.sum()),
                "n_pos": int((yy == 1).sum()),
                "rate": float(yy.mean()) if int(m.sum()) else float("nan"),
            }
            rows.append(rec)
            print(
                f"    {name:18s} n={rec['n']:5d} pos={rec['n_pos']:4d}  "
                f"rate={rec['rate']:.3f}"
            )
    own = {r["slice"]: r for r in rows if r["y"] == Y_OWN}
    spike = {r["slice"]: r for r in rows if r["y"] == Y_SPIKE}
    onset_vs_later = (
        own["onset_first_fin"]["rate"] - own["later_fin"]["rate"]
        if own["onset_first_fin"]["n"] and own["later_fin"]["n"]
        else float("nan")
    )
    n_pos_own = sum(r["n_pos"] for r in rows if r["y"] == Y_OWN)
    print(f"  own-p80 onset−later_fin={onset_vs_later:+.3f}")
    print(
        f"  spike onset {spike['onset_first_fin']['rate']:.3f} vs later_fin "
        f"{spike['later_fin']['rate']:.3f} vs later_off {spike['later_off']['rate']:.3f}"
    )
    print(
        f"  own-p80 pos mass: onset={own['onset_first_fin']['n_pos']}/{n_pos_own} "
        f"later_fin={own['later_fin']['n_pos']}/{n_pos_own} "
        f"before={own['before_first_fin']['n_pos']}/{n_pos_own}"
    )
    return {
        "rows": rows,
        "onset_vs_later_own": onset_vs_later,
        "spike_onset": spike["onset_first_fin"]["rate"],
        "spike_later_fin": spike["later_fin"]["rate"],
        "spike_later_off": spike["later_off"]["rate"],
        "own_pos_onset": own["onset_first_fin"]["n_pos"],
        "own_pos_later_fin": own["later_fin"]["n_pos"],
        "own_pos_before": own["before_first_fin"]["n_pos"],
        "own_pos_n": n_pos_own,
    }


def pass21_later_and_before(
    df: pd.DataFrame, train_by_y: dict[str, pd.Series]
) -> dict:
    """Singles after dropping onset; before-first-fin is the honest turning window."""
    print("\n" + "=" * 72)
    print("CUT 21 — singles on later_fin (presence mass) and before_first_fin (turning)")
    print("Y9 looks at t+1..t+3; before-first-fin mix is still ~0. Never X.")
    print("=" * 72)
    masks = _onset_masks(df)
    folds = df["fold"].to_numpy()
    cols = ("m_fee_share", "m_int_share", "a_out6", "a_out3", "log1p_a_in3")
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        for sl_name in ("later_fin", "before_first_fin"):
            sl = lab & masks[sl_name]
            n = int(sl.sum())
            n_pos = int((pd.to_numeric(df.loc[sl, y_col], errors="coerce") == 1).sum())
            print(f"  {y_col} {sl_name} n={n} pos={n_pos}")
            if n < 40 or n_pos < 8:
                print("    skip (thin)")
                continue
            assert_no_holdout(df.loc[sl, "company_id"])
            for col in cols:
                if col not in df.columns:
                    continue
                oof = signed_oof_auroc(df, col, y_col, sl, folds)
                rec = {
                    "y": y_col,
                    "slice": sl_name,
                    "col": col,
                    "cv": oof["cv"],
                    "sd": oof["sd"],
                    "train_auc": oof["train_auc"],
                    "n": int((sl & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
                    "n_pos": n_pos,
                }
                rows.append(rec)
                print(
                    f"    {col:16s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                    f"train={oof['train_auc']:.4f}  n={rec['n']}"
                )
    later = [r for r in rows if r["y"] == Y_OWN and r["slice"] == "later_fin"]
    fee = next((r for r in later if r["col"] == "m_fee_share"), None)
    out = next((r for r in later if r["col"] == "a_out6"), None)
    gap = (
        float(fee["cv"] - out["cv"])
        if fee and out and np.isfinite(fee["cv"]) and np.isfinite(out["cv"])
        else float("nan")
    )
    keep_later = bool(
        fee
        and np.isfinite(fee["cv"])
        and fee["cv"] >= OUT6_BAR + CLEAR_MARGIN
        and np.isfinite(gap)
        and gap >= CLEAR_MARGIN
    )
    print(f"  later_fin own-p80 fee vs a_out6 gap={gap:+.3f} KEEP_later={keep_later}")
    before = [r for r in rows if r["y"] == Y_OWN and r["slice"] == "before_first_fin"]
    best_before = max(
        (r for r in before if np.isfinite(r["cv"]) and r["col"] != "m_fee_share"),
        key=lambda r: r["cv"],
        default=None,
    )
    sl = train_by_y[Y_OWN] & _onset_masks(df)["before_first_fin"]
    before_folds = []
    before_size = float("nan")
    if int(sl.sum()) >= 40:
        oof = signed_oof_auroc(df, "a_out6", Y_OWN, sl, folds)
        before_folds = [f["auroc"] for f in oof["folds"]]
        y = pd.to_numeric(df.loc[sl, Y_OWN], errors="coerce")
        size = pd.to_numeric(df.loc[sl, "log1p_a_in3"], errors="coerce")
        before_size = float(auroc(y, size))
        bits = " ".join(f"{a:.3f}" for a in before_folds)
        print(f"  before a_out6 folds {bits}  sizeAUC={before_size:.3f}")
    keep_before = bool(
        best_before
        and np.isfinite(best_before["cv"])
        and best_before["cv"] >= OUT6_BAR + CLEAR_MARGIN
        and np.isfinite(best_before.get("sd", float("nan")))
        and best_before["sd"] <= 0.05
        and (not np.isfinite(before_size) or before_size < SIZE_AUROC)
        and int(sl.sum()) >= 800
    )
    if best_before:
        print(
            f"  before_first_fin best `{best_before['col']}` CV={best_before['cv']:.3f} "
            f"sd={best_before['sd']:.3f} KEEP_before={keep_before} "
            f"(thin n={int(sl.sum())}; mix is 0.50; not a mix merge)"
        )
    return {
        "rows": rows,
        "later_gap": gap,
        "keep_later": keep_later,
        "best_before": best_before,
        "keep_before": keep_before,
        "before_folds": before_folds,
        "before_size": before_size,
    }


def pass22_later_off_and_before_q(
    df: pd.DataFrame, train_by_y: dict[str, pd.Series]
) -> dict:
    """later_off singles (resume window) + before_first_fin a_out6 quintiles."""
    print("\n" + "=" * 72)
    print("CUT 22 — later_off leftover + before_first_fin a_out6 quintiles")
    print("later_off = had fin before, a_fin_cost=0 this month. Mix ~0. Never X.")
    print("=" * 72)
    masks = _onset_masks(df)
    folds = df["fold"].to_numpy()
    cols = ("m_fee_share", "a_out6", "a_out3", "log1p_a_in3")
    rows = []
    for y_col in (Y_OWN, Y_SPIKE):
        lab = train_by_y[y_col]
        sl = lab & masks["later_off"]
        n = int(sl.sum())
        n_pos = int((pd.to_numeric(df.loc[sl, y_col], errors="coerce") == 1).sum())
        print(f"  {y_col} later_off n={n} pos={n_pos}")
        if n < 40 or n_pos < 8:
            print("    skip (thin)")
            continue
        assert_no_holdout(df.loc[sl, "company_id"])
        for col in cols:
            if col not in df.columns:
                continue
            oof = signed_oof_auroc(df, col, y_col, sl, folds)
            rec = {
                "y": y_col,
                "col": col,
                "cv": oof["cv"],
                "sd": oof["sd"],
                "train_auc": oof["train_auc"],
                "n": int((sl & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
                "n_pos": n_pos,
                "folds": [f["auroc"] for f in oof["folds"]],
            }
            rows.append(rec)
            bits = " ".join(f"{a:.3f}" for a in rec["folds"])
            print(
                f"    {col:16s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
                f"train={oof['train_auc']:.4f}  folds {bits}"
            )
    sl = train_by_y[Y_OWN] & masks["before_first_fin"]
    qtab = quintile_table(df, sl, "a_out6", Y_OWN)
    print(
        f"  before a_out6 quintiles n={qtab['n']} bins={qtab['n_bins']} "
        f"monotone={qtab['monotone_up']} tail_only={qtab['tail_only']}"
    )
    for r in qtab["rows"]:
        print(
            f"    Q{r['q']} n={r['n']:4d} pos={r['n_pos']:3d}  "
            f"rate={r['y_rate']:.3f}  med={r['x_median']:.4g}"
        )
    best_off = max(
        (r for r in rows if r["y"] == Y_OWN and np.isfinite(r["cv"])),
        key=lambda r: r["cv"],
        default=None,
    )
    keep_off = bool(
        best_off
        and best_off["col"].startswith("m_")
        and best_off["cv"] >= OUT6_BAR + CLEAR_MARGIN
    )
    print(
        f"  later_off best `{best_off['col'] if best_off else None}` "
        f"CV={best_off['cv'] if best_off else float('nan'):.3f} KEEP_mix={keep_off}"
    )
    return {
        "rows": rows,
        "before_q": qtab,
        "best_off": best_off,
        "keep_off": keep_off,
    }


def pass23_months_to_onset(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is before-first-fin Y9 just 'close to first fee' (forward label)?"""
    print("\n" + "=" * 72)
    print("CUT 23 — months until first-fin vs Y9 on the before slice")
    print("comparator only; nearer-to-onset is the forward label. Never X.")
    print("=" * 72)
    work = df[["company_id", "period"]].copy()
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    any_fin = pd.Series(
        np.where(fin.notna(), (fin > 0).astype(float), np.nan), index=df.index
    )
    first = (
        work.loc[any_fin == 1]
        .groupby("company_id", sort=False)["period"]
        .min()
    )
    work["first_fin"] = work["company_id"].map(first)
    # integer month gap; period is month-start
    gap = (work["first_fin"] - work["period"]).dt.days / 30.44
    df = df.copy()
    df["_months_to_first_fin"] = pd.Series(gap.to_numpy(), index=df.index)
    df["_near_onset"] = (df["_months_to_first_fin"] <= 3).astype(float)
    masks = _onset_masks(df)
    folds = df["fold"].to_numpy()
    sl = train_by_y[Y_OWN] & masks["before_first_fin"]
    assert_no_holdout(df.loc[sl, "company_id"])
    n_cos = int(df.loc[sl, "company_id"].nunique())
    print(f"  before_first_fin months={int(sl.sum())} companies={n_cos}")
    rows = []
    for col, sign_hint in (("_months_to_first_fin", -1), ("_near_onset", 1)):
        oof = signed_oof_auroc(df, col, Y_OWN, sl, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "train_sign": oof["train_sign"],
            "folds": [f["auroc"] for f in oof["folds"]],
        }
        rows.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(
            f"  {col:22s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"sign={oof['train_sign']:+d}  folds {bits}"
        )
        del sign_hint
    # rate by months to onset
    y = pd.to_numeric(df.loc[sl, Y_OWN], errors="coerce")
    mleft = pd.to_numeric(df.loc[sl, "_months_to_first_fin"], errors="coerce")
    by = []
    for lo, hi, name in (
        (0, 1.5, "1m"),
        (1.5, 3.5, "2-3m"),
        (3.5, 6.5, "4-6m"),
        (6.5, 99, "7m+"),
    ):
        m = (mleft > lo) & (mleft <= hi)
        rec = {
            "bin": name,
            "n": int(m.sum()),
            "n_pos": int((y[m] == 1).sum()),
            "rate": float(y[m].mean()) if int(m.sum()) else float("nan"),
        }
        by.append(rec)
        print(
            f"    {name:6s} n={rec['n']:4d} pos={rec['n_pos']:3d}  rate={rec['rate']:.3f}"
        )
    clock = next(r for r in rows if r["col"] == "_months_to_first_fin")
    algebraic = bool(np.isfinite(clock["cv"]) and clock["cv"] >= 0.70)
    print(f"  months_to_first_fin CV={clock['cv']:.3f} algebraic_clock={algebraic}")
    return {
        "rows": rows,
        "by_gap": by,
        "n_cos": n_cos,
        "n": int(sl.sum()),
        "clock_cv": clock["cv"],
        "algebraic": algebraic,
    }


def pass24_later_intensity(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """On later_fin (presence saturated), is fee-share intensity monotone?"""
    print("\n" + "=" * 72)
    print("CUT 24 — later_fin m_fee_share quintiles (presence already on)")
    print("=" * 72)
    masks = _onset_masks(df)
    tables = []
    for y_col in (Y_OWN, Y_SPIKE):
        sl = train_by_y[y_col] & masks["later_fin"]
        tab = quintile_table(df, sl, "m_fee_share", y_col)
        tables.append(tab)
        print(
            f"  {y_col} n={tab['n']} bins={tab['n_bins']} "
            f"monotone={tab['monotone_up']} tail_only={tab['tail_only']}"
        )
        for r in tab["rows"]:
            print(
                f"    Q{r['q']} n={r['n']:4d} pos={r['n_pos']:4d}  "
                f"rate={r['y_rate']:.3f}  med={r['x_median']:.4g}"
            )
    return {"tables": tables}


def pass25_last_pre_fee(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Within company: last pre-fee month vs earlier — is a_out6 a ramp?"""
    print("\n" + "=" * 72)
    print("CUT 25 — last pre-fee month vs earlier (within company, train)")
    print("the 64 algebraic Y=1 months vs earlier pre-fee. Never X.")
    print("=" * 72)
    work = df[["company_id", "period"]].copy()
    fin = pd.to_numeric(df["a_fin_cost"], errors="coerce")
    out = pd.to_numeric(df["a_out6"], errors="coerce")
    work["any_fin"] = pd.Series(
        np.where(fin.notna(), (fin > 0).astype(float), np.nan), index=df.index
    )
    work["a_out6"] = out
    first = (
        work.loc[work["any_fin"] == 1]
        .groupby("company_id", sort=False)["period"]
        .min()
    )
    work["first_fin"] = work["company_id"].map(first)
    work["months_to"] = (work["first_fin"] - work["period"]).dt.days / 30.44
    lab = train_by_y[Y_OWN]
    before = lab & work["first_fin"].notna() & (work["period"] < work["first_fin"])
    last = before & (work["months_to"] <= 1.5)
    earlier = before & (work["months_to"] > 1.5)
    y = pd.to_numeric(df[Y_OWN], errors="coerce")
    print(
        f"  last n={int(last.sum())} pos={int((y[last] == 1).sum())}  "
        f"earlier n={int(earlier.sum())} pos={int((y[earlier] == 1).sum())}"
    )
    last_out = out[last]
    early_out = out[earlier]
    recs = []
    for name, s in (("last_pre", last_out), ("earlier_pre", early_out)):
        recs.append(
            {
                "slice": name,
                "n": int(s.notna().sum()),
                "median": float(s.median()) if int(s.notna().sum()) else float("nan"),
                "p80": float(s.quantile(0.8)) if int(s.notna().sum()) else float("nan"),
            }
        )
        print(f"    {name} n={recs[-1]['n']} med={recs[-1]['median']:.4g} p80={recs[-1]['p80']:.4g}")
    # paired: companies with both last and earlier
    last_med = (
        work.loc[last]
        .groupby("company_id", sort=False)["a_out6"]
        .median()
    )
    early_med = (
        work.loc[earlier]
        .groupby("company_id", sort=False)["a_out6"]
        .median()
    )
    both = pd.concat([last_med.rename("last"), early_med.rename("earlier")], axis=1).dropna()
    n_both = int(len(both))
    if n_both:
        both["delta"] = both["last"] - both["earlier"]
        share_up = float((both["delta"] > 0).mean())
        med_delta = float(both["delta"].median())
    else:
        share_up = float("nan")
        med_delta = float("nan")
    print(
        f"  paired cos={n_both}  share last>earlier={share_up:.3f}  "
        f"median delta={med_delta:.4g}"
    )
    return {
        "rows": recs,
        "n_both": n_both,
        "share_up": share_up,
        "med_delta": med_delta,
        "n_last": int(last.sum()),
        "n_earlier": int(earlier.sum()),
    }


def pass26_fee_acf(df: pd.DataFrame, train_row: pd.Series) -> dict:
    """Amount-share acf1 — Family M notes said ≈0. Honest Q6 persistence."""
    print("\n" + "=" * 72)
    print("CUT 26 — train acf1 of m_fee_share / m_int_share / a_out6")
    print("mix cannot answer Q6 if amount-share acf1 ≈ 0")
    print("=" * 72)
    assert_no_holdout(df.loc[train_row, "company_id"])
    work = df.loc[train_row, ["company_id", "period", "m_fee_share", "m_int_share", "a_out6"]].copy()
    work = work.sort_values(["company_id", "period"])
    rows = []
    for col in ("m_fee_share", "m_int_share", "a_out6"):
        s = pd.to_numeric(work[col], errors="coerce")
        lag = s.groupby(work["company_id"], sort=False).shift(1)
        d = pd.DataFrame({"x": s, "lag": lag}).dropna()
        rho = float(d["x"].corr(d["lag"])) if len(d) >= 20 else float("nan")
        # share of companies with |acf1|>=0.3 (min 6 pairs)
        acfs = []
        for _, g in work.groupby("company_id", sort=False):
            xx = pd.to_numeric(g[col], errors="coerce")
            if int(xx.notna().sum()) < 6:
                continue
            a = float(xx.autocorr(lag=1))
            if np.isfinite(a):
                acfs.append(a)
        rec = {
            "col": col,
            "acf1": rho,
            "n_pairs": int(len(d)),
            "n_cos": len(acfs),
            "med_company": float(np.median(acfs)) if acfs else float("nan"),
            "share_persist": float(np.mean([abs(a) >= 0.30 for a in acfs])) if acfs else float("nan"),
        }
        rows.append(rec)
        print(
            f"  {col:16s} pooled_acf1={rho:+.3f}  n={rec['n_pairs']}  "
            f"med_company={rec['med_company']:+.3f}  "
            f"share_|acf|>=0.3={rec['share_persist']:.3f} ({rec['n_cos']} cos)"
        )
    fee = next(r for r in rows if r["col"] == "m_fee_share")
    persist = bool(np.isfinite(fee["med_company"]) and abs(fee["med_company"]) >= 0.30)
    print(
        f"  Q6 mix persist? {persist}  (quote company-median acf1={fee['med_company']:+.3f}; "
        f"pooled {fee['acf1']:+.3f} is zeros lining up)"
    )
    return {
        "rows": rows,
        "fee_acf1": fee["acf1"],
        "fee_med_company": fee["med_company"],
        "persist": persist,
    }


def pass27_out_within_size(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Is a_out6's 0.565 just size? Rates by outflow Q inside inflow terciles."""
    print("\n" + "=" * 72)
    print("CUT 27 — a_out6 quintiles inside log1p(a_in3) terciles (train own-p80)")
    print("if Q1 shield dies inside size, the 0.565 bar is size not spend")
    print("=" * 72)
    lab = train_by_y[Y_OWN]
    out = pd.to_numeric(df["a_out6"], errors="coerce")
    size = pd.to_numeric(df["log1p_a_in3"], errors="coerce")
    y = pd.to_numeric(df[Y_OWN], errors="coerce")
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    m = lab & out.notna() & size.notna() & y.notna()
    assert_no_holdout(df.loc[m, "company_id"])
    work = pd.DataFrame(
        {
            "out": out[m],
            "size": size[m],
            "y": y[m],
            "any_fee": (fee[m] > 0).astype(float),
        }
    )
    work["size_t"] = pd.qcut(work["size"], 3, duplicates="drop")
    rows = []
    print("  P(Y=1) by size tercile × a_out6 Q")
    for i, (st, g) in enumerate(work.groupby("size_t", observed=True), start=1):
        try:
            g = g.copy()
            g["out_q"] = pd.qcut(g["out"], 5, duplicates="drop")
        except ValueError:
            continue
        for j, (oq, h) in enumerate(g.groupby("out_q", observed=True), start=1):
            rec = {
                "size_t": i,
                "out_q": j,
                "n": int(len(h)),
                "n_pos": int((h["y"] == 1).sum()),
                "rate": float(h["y"].mean()),
                "fee_rate": float(h["any_fee"].mean()) if len(h) else float("nan"),
            }
            rows.append(rec)
        bits = " ".join(
            f"Q{r['out_q']}={r['rate']:.3f}/fee{r['fee_rate']:.2f}(n={r['n']})"
            for r in rows
            if r["size_t"] == i
        )
        print(f"    size T{i} n={len(g)}  {bits}")
    # Q1 vs rest inside each size tercile
    gaps = []
    for i in sorted({r["size_t"] for r in rows}):
        q1 = next((r for r in rows if r["size_t"] == i and r["out_q"] == 1), None)
        rest_n = sum(r["n"] for r in rows if r["size_t"] == i and r["out_q"] != 1)
        rest_pos = sum(r["n_pos"] for r in rows if r["size_t"] == i and r["out_q"] != 1)
        rest_rate = rest_pos / rest_n if rest_n else float("nan")
        gap = (q1["rate"] - rest_rate) if q1 and np.isfinite(rest_rate) else float("nan")
        gaps.append({"size_t": i, "q1": q1["rate"] if q1 else float("nan"), "rest": rest_rate, "gap": gap})
        print(f"    size T{i} Q1-rest gap={gap:+.3f}")
    shield_inside = any(np.isfinite(g["gap"]) and g["gap"] <= -0.04 for g in gaps)
    print(f"  Q1 shield survives inside size terciles? {shield_inside}")
    return {"rows": rows, "gaps": gaps, "shield_inside": shield_inside}


def pass28_large_fee_out(df: pd.DataFrame, train_by_y: dict[str, pd.Series]) -> dict:
    """Among large + any-fee months, does a_out6 or mix leftover remain?"""
    print("\n" + "=" * 72)
    print("CUT 28 — own-p80 singles on large (in3 T3) ∩ any-fee")
    print("if a_out6 dies here, the 0.565 bar is the no-fee/small shield")
    print("=" * 72)
    lab = train_by_y[Y_OWN]
    size = pd.to_numeric(df["log1p_a_in3"], errors="coerce")
    fee = pd.to_numeric(df["m_fee_share"], errors="coerce")
    m0 = lab & size.notna() & fee.notna()
    assert_no_holdout(df.loc[m0, "company_id"])
    t3_cut = float(size[m0].quantile(2 / 3))
    sl = m0 & (size >= t3_cut) & (fee > 0)
    y = pd.to_numeric(df.loc[sl, Y_OWN], errors="coerce")
    print(
        f"  n={int(sl.sum())} pos={int((y == 1).sum())} rate="
        f"{float(y.mean()) if int(sl.sum()) else float('nan'):.4f}  t3_cut={t3_cut:.4g}"
    )
    folds = df["fold"].to_numpy()
    rows = []
    for col in ("m_fee_share", "m_int_share", "a_out6", "a_out3", "log1p_a_in3"):
        oof = signed_oof_auroc(df, col, Y_OWN, sl, folds)
        rec = {
            "col": col,
            "cv": oof["cv"],
            "sd": oof["sd"],
            "train_auc": oof["train_auc"],
            "n": int((sl & pd.to_numeric(df[col], errors="coerce").notna()).sum()),
            "folds": [f["auroc"] for f in oof["folds"]],
        }
        rows.append(rec)
        bits = " ".join(f"{a:.3f}" for a in rec["folds"])
        print(
            f"    {col:16s} CV={oof['cv']:.4f}±{oof['sd']:.3f}  "
            f"train={oof['train_auc']:.4f}  folds {bits}"
        )
    fee_r = next(r for r in rows if r["col"] == "m_fee_share")
    out_r = next(r for r in rows if r["col"] == "a_out6")
    keep = bool(
        np.isfinite(fee_r["cv"])
        and fee_r["cv"] >= OUT6_BAR + CLEAR_MARGIN
        and fee_r["cv"] - out_r["cv"] >= CLEAR_MARGIN
    )
    print(f"  KEEP mix on large∩fee? {keep}  fee={fee_r['cv']:.3f} out={out_r['cv']:.3f}")
    return {
        "rows": rows,
        "n": int(sl.sum()),
        "n_pos": int((y == 1).sum()),
        "rate": float(y.mean()) if int(sl.sum()) else float("nan"),
        "keep": keep,
        "t3_cut": t3_cut,
    }


def extra_md_lines(
    p1: dict,
    p7: dict,
    p8: dict,
    p9: dict,
    p10: dict,
    p11: dict,
    p12: dict,
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
) -> list[str]:
    tw = p1.get("twos", {}).get(Y_OWN, {})
    lines = [
        "### 1b — 2×2 high-out × high-fee among own-p80 positives",
        "",
        f"n={tw.get('n')}  out_only={tw.get('out_only')}  fee_only={tw.get('fee_only')}  "
        f"both={tw.get('both')}  neither={tw.get('neither')} "
        f"(neither share **{p1.get('neither_share', float('nan')):.1%}**).",
        "Largest cell is *neither*. Y9 is not 'spent more', and own-p80 high-fee is not most of it.",
        "",
        "### 7 — leak on the positive support",
        "",
        "All-row Spearman vs `a_fin_cost` can be zeros lining up (no fee month ⇒ no fin_cost).",
        "",
        "| feature | vs | n all | ρ all | n support | ρ support | fail all / support |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for r in p7["rows"]:
        lines.append(
            f"| `{r['col']}` | `{r['vs']}` | {r['n_all']} | {r['rho_all']:+.3f} | "
            f"{r['n_pos']} | {r['rho_pos']:+.3f} | {r['fail_all']} / {r['fail_pos']} |"
        )
    b = p7["binary"]
    lines += [
        "",
        f"Any-fee vs any-`a_fin_cost`: AUROC **{b['auroc_anyfee_vs_anyfin']:.3f}**, "
        f"agree {b['agree']:.1%}, fee-only {b['fee_only']}, fin-only {b['fin_only']} "
        "(interest without a fee).",
        "",
        "### 8 — any-fee vs intensity",
        "",
        "| Y | slice | n | n_pos | P(Y=1) |",
        "|---|---|---:|---:|---:|",
    ]
    for r in p8["rates"]:
        lines.append(
            f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        "| Y | feature | slice | CV AUROC ± sd | n |",
        "|---|---|---|---:|---:|",
    ]
    for r in p8["rows"]:
        lines.append(
            f"| `{r['y']}` | `{r['col']}` | {r['slice']} | "
            f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['n_defined']} |"
        )
    lines += [
        "",
        "### 9 — mix inside outflow terciles",
        "",
        "| Y | slice | n | n_pos | P(Y=1) | fee-share CV | a_out6 CV |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in p9["rows"]:
        out_s = f"{r['out_cv']:.3f}" if np.isfinite(r.get("out_cv", float("nan"))) else "—"
        lines.append(
            f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} | "
            f"{r['fee_cv']:.3f} | {out_s} |"
        )
    lines += [
        "",
        "### 10 — neither cell",
        "",
        f"n={p10['n']} of {p10['n_2x2']} 2×2 own-p80 positives. "
        f"Median own-p80 of `m_fee_share` among Y9 pos = **{p10['fee_p80_med']:.4g}** "
        f"(p90 {p10['fee_p80_p90']:.4g}; share of own-p80 that is 0: {p10['fee_p80_zero']:.1%}). "
        "'High fee-share' is often any crumb above a zero history.",
        "",
        "| flag inside neither | n_hi / n | share |",
        "|---|---:|---:|",
    ]
    for r in p10["rows"]:
        lines.append(f"| `{r['flag']}` | {r['n_hi']} / {r['n_defined']} | {r['share']:.3f} |")
    lines += [
        "",
        "### 11 — 360 vs 110 (never D/E as X)",
        "",
        "| Y | slice | n | n_pos | companies | base rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in p11["rows"]:
        lines.append(
            f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        "### 12 — own-p80 ∩ spike; holdout coverage",
        "",
        f"Train both-labeled {p12['n_both_lab']}: both-pos {p12['n_both_pos']}, "
        f"own-only {p12['n_own_only']}, spike-only {p12['n_spk_only']}, "
        f"Spearman {p12['rho']:+.3f}. Two labels, not a rewrite.",
        f"Holdout mix coverage {p12['hold_fee_cov']:.1%} "
        f"(own-p80 labeled {p12['hold_own_n']} / {p12['hold_own_pos']} pos). "
        f"Holdout `m_fee_share` AUROC {p12['hold_fee_auroc']:.3f} (LOW_POWER, not a claim).",
        "",
    ]
    if p13:
        lines += [
            "### 13 — `m_fee_n_share` honesty",
            "",
            f"Spearman vs amount `m_fee_share` **{p13['rho_n_vs_amt']:+.3f}**. "
            f"Any-`a_fin_cost` presence (comparator, never X) CV **{p13['any_fin_cv']:.3f}**. "
            f"`m_fee_n_share` all-row CV **{p13['nshare_cv']:.3f}** "
            f"(gap vs presence {p13['gap_vs_presence']:+.3f}). "
            f"On fee>0 support CV **{p13['nshare_pos_cv']:.3f}**. "
            f"presence_rewrite={p13['presence_rewrite']}; intensity_dead={p13['intensity_dead']}.",
            "",
            "| feature | slice | CV AUROC ± sd | n |",
            "|---|---|---:|---:|",
        ]
        for r in p13["rows"]:
            lines.append(
                f"| `{r['col']}` | {r['slice']} | {r['cv']:.3f} ± {r['sd']:.3f} | {r['n']} |"
            )
        lines.append("")
        if p13["presence_rewrite"] or p13["intensity_dead"]:
            lines.append(
                "Honesty: the numeric KEEP on `m_fee_n_share` is fee-*presence*, not mix intensity. "
                "Same raw fee tickets that build `a_fin_cost`. **Do not merge.**"
            )
        else:
            lines.append(
                "Honesty: count share still has intensity after presence is removed. "
                "Merge remains a parent recommendation only — do not rewrite parquet."
            )
        lines.append("")
    if p14:
        lines += [
            "### 14 — presence clock and outflow × any-fee",
            "",
            f"Any-fin-cost now CV **{p14['now_cv']:.3f}**; lag-1 **{p14['lag1_cv']:.3f}**. "
            f"Honest 1-month Q6 KEEP vs `a_out6`+0.02: **{p14['q6_keep']}**. "
            "Presence is contemporaneous (same as fee-share). Not a lead.",
            "",
            "| a_out6 Q | no-fee n / P(Y=1) | any-fee n / P(Y=1) |",
            "|---:|---:|---:|",
        ]
        qs = sorted({r["q"] for r in p14["cross"]})
        for q in qs:
            z = next(r for r in p14["cross"] if r["q"] == q and r["any_fee"] == 0)
            o = next(r for r in p14["cross"] if r["q"] == q and r["any_fee"] == 1)
            lines.append(
                f"| {q} | {z['n']} / {z['rate']:.3f} | {o['n']} / {o['rate']:.3f} |"
            )
        lines += [
            "",
            "Low-outflow Q1 is protective **only among no-fee months**. "
            "Any-fee months sit near the 14–22% Y9 rate in every outflow quintile. "
            "`a_out6` is a no-activity shield, not a spend-tail why.",
            "",
        ]
        if p14.get("fold_cmp"):
            lines += [
                "Fold AUCs (train group-fold, own-p80):",
                "",
            ]
            for r in p14["fold_cmp"]:
                bits = " / ".join(f"{a:.3f}" for a in r.get("folds") or [])
                lines.append(f"- `{r['col']}` folds {bits}  cv={r['cv']:.3f}")
            lines.append("")
    if p15:
        lines += [
            "### 15 — which fin_cost token",
            "",
            "| Y | slice | n | n_pos | P(Y=1) |",
            "|---|---|---:|---:|---:|",
        ]
        for r in p15["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines += [
            "",
            "Interest-only months match fee-only on own-p80 (~18% vs neither 8%). "
            "The Y9 lump is **any fin_cost token present**, not fee vs interest intensity. "
            "`m_int_share` as a continuous share still loses (0.525) because interest is rare; "
            "the binary *presence* is the whole signal.",
            "",
        ]
    if p16:
        lines += [
            "### 16 — 360 vs 110 × any-fin (never D/E)",
            "",
            "| Y | slice | n | n_pos | P(Y=1) |",
            "|---|---|---:|---:|---:|",
        ]
        for r in p16["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines += [
            "",
            "Own-p80 360 vs 110 is mostly presence (nofin ~7.5–7.9% both sides). "
            "Spike is not: mixed-dark nofin 21.4% vs all-dark nofin 12.9%. "
            "Do not put D/E on that gap. Not a merge reason.",
            "",
        ]
    if p17:
        lines += [
            "### 17 — company-level trait vs turning",
            "",
            "Spearman of each train company's any-fin month-share vs its Y9 month-share "
            "(≥3 labeled months). High ρ = always-fee firms, not turning.",
            "",
            "| Y | companies | ever pos | ρ | fin-rate pos / neg cos | Y rate always-fin | Y rate never-fin |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for r in p17["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['n_cos']} | {r['n_ever_pos']} | {r['rho']:+.3f} | "
                f"{r['fin_rate_pos']:.2f} / {r['fin_rate_neg']:.2f} | "
                f"{r['y_rate_always_fin']:.3f} (n={r['n_always_fin']}) | "
                f"{r['y_rate_never_fin']:.3f} (n={r['n_never_fin']}) |"
            )
        lines += [
            "",
            "Never-fin own-p80 ≈ 0 is algebraic (0 cannot exceed own p80; spike NaNs a zero base). "
            "Useful number: own-p80 ρ=+0.32 (mild company trait); spike ρ=−0.03 (turning, not a firm type). "
            "Label stays. Still cannot put presence in X.",
            "",
        ]
    if p18:
        lines += [
            "### 18 — intensity on always-fin companies",
            "",
            f"Train companies with any-fin on ≥99% of own-p80 labeled months: "
            f"{p18['n_cos']} cos, {p18['n']} months, {p18['n_pos']} pos, "
            f"rate {p18['rate']:.1%}. Presence is saturated.",
            "",
            "| Y | feature | CV AUROC ± sd | n |",
            "|---|---|---:|---:|",
        ]
        for y_col, block in (p18.get("by_y") or {Y_OWN: p18}).items():
            for r in block.get("rows") or p18.get("rows") or []:
                lines.append(
                    f"| `{r.get('y', y_col)}` | `{r['col']}` | "
                    f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['n']} |"
                )
        lines += [
            "",
            f"Gap `m_fee_share` vs `a_out6` **{p18['gap']:+.3f}**. "
            f"KEEP intensity (must also clear 0.565+0.02, not just a dead `a_out6`)? "
            f"**{p18['keep_intensity']}**. "
            "`a_out6` collapses to ~0.51 once no-fee months are gone — that is the shield. "
            "Fee-share leftover 0.54 is not a Q5 card. Still do not merge.",
            "",
        ]
    if p19:
        lines += [
            "### 19 — fee-share bins × outflow quintiles",
            "",
            "Fee-share collapses (mostly zero). Bins are no-fee / fee-lo / fee-hi (median split on fee>0).",
            "",
            "| Y | outflow bin | fee bin | n | n_pos | P(Y=1) |",
            "|---|---|---|---:|---:|---:|",
        ]
        fee_ord = {"no_fee": 0, "fee_lo": 1, "fee": 1, "fee_hi": 2}
        own19 = [r for r in p19["rows"] if r["y"] == Y_OWN]
        def _out_lo(s: str) -> float:
            try:
                return float(str(s).split(",")[0].lstrip("(").lstrip("["))
            except ValueError:
                return 0.0
        own19 = sorted(own19, key=lambda r: (_out_lo(r["out_q"]), fee_ord.get(r["fee_bin"], 9)))
        for r in own19:
            lines.append(
                f"| `{r['y']}` | {r['out_q']} | {r['fee_bin']} | "
                f"{r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines += [
            "",
            "If fee-hi vs no-fee stays large in every outflow bin, leftover mix is still presence, not spend. "
            "Do not merge.",
            "",
        ]
    if p20:
        lines += [
            "### 20 — first-fin onset vs later (never X)",
            "",
            f"Own-p80 onset − later_fin **{p20['onset_vs_later_own']:+.3f}**. "
            f"Spike onset {p20['spike_onset']:.3f} / later_fin {p20['spike_later_fin']:.3f} / "
            f"later_off {p20['spike_later_off']:.3f}.",
            "",
            "| Y | slice | n | n_pos | P(Y=1) |",
            "|---|---|---:|---:|---:|",
        ]
        for r in p20["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines += [
            "",
            "Onset vs later_fin tests turning-on vs presence-as-trait. "
            "later_off vs later_fin is the turning-off residual. Comparator only. "
            f"Own-p80 pos mass: onset {p20.get('own_pos_onset')}/{p20.get('own_pos_n')}, "
            f"later_fin {p20.get('own_pos_later_fin')}/{p20.get('own_pos_n')}, "
            f"before {p20.get('own_pos_before')}/{p20.get('own_pos_n')}.",
            "",
        ]
    if p21:
        lines += [
            "### 21 — later_fin leftover and before-first-fin Q6",
            "",
            f"later_fin `m_fee_share` vs `a_out6` gap **{p21['later_gap']:+.3f}**. "
            f"KEEP_later (must clear 0.565+0.02 too)? **{p21['keep_later']}**.",
            "",
            "| Y | slice | feature | CV AUROC ± sd | n |",
            "|---|---|---|---:|---:|",
        ]
        for r in p21["rows"]:
            lines.append(
                f"| `{r['y']}` | {r['slice']} | `{r['col']}` | "
                f"{r['cv']:.3f} ± {r['sd']:.3f} | {r['n']} |"
            )
        bb = p21.get("best_before") or {}
        lines += [
            "",
            f"before_first_fin best `{bb.get('col')}` CV **{bb.get('cv', float('nan')):.3f}** "
            f"± {bb.get('sd', float('nan')):.3f} (folds "
            f"{' '.join(f'{a:.3f}' for a in (p21.get('before_folds') or []))}; "
            f"sizeAUC {p21.get('before_size', float('nan')):.3f}). "
            f"KEEP_before **{p21.get('keep_before')}**. "
            "n=349 / 64 pos is LOW_POWER; mix is 0.50. "
            "Do not claim a Q6 lead. Do not merge M.",
            "",
        ]
    if p22:
        lines += [
            "### 22 — later_off leftover + before a_out6 quintiles",
            "",
            f"later_off best `{((p22.get('best_off') or {}).get('col'))}` "
            f"CV **{((p22.get('best_off') or {}).get('cv', float('nan'))):.3f}**. "
            f"KEEP mix? **{p22.get('keep_off')}**.",
            "",
            "| Y | feature | CV AUROC ± sd | n |",
            "|---|---|---:|---:|",
        ]
        for r in p22["rows"]:
            lines.append(
                f"| `{r['y']}` | `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['n']} |"
            )
        bq = p22.get("before_q") or {}
        lines += [
            "",
            f"before_first_fin `a_out6` quintiles n={bq.get('n')} "
            f"bins={bq.get('n_bins')} monotone={bq.get('monotone_up')} "
            f"tail_only={bq.get('tail_only')} (LOW_POWER).",
            "",
            "| Q | n | n_pos | P(Y=1) | median a_out6 |",
            "|---:|---:|---:|---:|---:|",
        ]
        for r in bq.get("rows") or []:
            lines.append(
                f"| {r['q']} | {r['n']} | {r['n_pos']} | {r['y_rate']:.3f} | {r['x_median']:.4g} |"
            )
        lines += [
            "",
            "later_off mix should be ~0.50. before quintiles stay LOW_POWER. Do not merge.",
            "",
        ]
    if p23:
        lines += [
            "### 23 — months until first-fin (before slice, never X)",
            "",
            f"{p23['n']} months / {p23['n_cos']} companies. "
            f"`months_to_first_fin` CV **{p23['clock_cv']:.3f}**. "
            f"algebraic_clock={p23['algebraic']}.",
            "",
            "| feature | CV AUROC ± sd | sign |",
            "|---|---:|---:|",
        ]
        for r in p23["rows"]:
            lines.append(
                f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['train_sign']:+d} |"
            )
        lines += [
            "",
            "| months to first fin | n | n_pos | P(Y=1) |",
            "|---|---:|---:|---:|",
        ]
        for r in p23["by_gap"]:
            lines.append(
                f"| {r['bin']} | {r['n']} | {r['n_pos']} | {r['rate']:.3f} |"
            )
        lines += [
            "",
            "If the clock is strong, the 0.66 `a_out6` leftover is proximity to onset "
            "(Y looks at t+1..t+3), not a Q6 mix lead. Do not merge.",
            "",
        ]
    if p24:
        lines += [
            "### 24 — later_fin fee-share quintiles (presence on)",
            "",
            "If this is flat, leftover mix intensity is dead. Single CV 0.550 already loses the KEEP bar.",
            "",
        ]
        for tab in p24["tables"]:
            lines += [
                f"`{tab['x']}` vs `{tab['y']}` n={tab['n']} bins={tab['n_bins']} "
                f"monotone={tab['monotone_up']} tail_only={tab['tail_only']}",
                "",
                "| Q | n | n_pos | P(Y=1) | median fee-share |",
                "|---:|---:|---:|---:|---:|",
            ]
            for r in tab["rows"]:
                lines.append(
                    f"| {r['q']} | {r['n']} | {r['n_pos']} | {r['y_rate']:.3f} | {r['x_median']:.4g} |"
                )
            lines.append("")
        lines += [
            "Tail-only (+4–9 pp in Q5). Not monotone. Not a Q5 mix card. Still do not merge.",
            "",
        ]
    if p25:
        lines += [
            "### 25 — last pre-fee vs earlier outflow (within company)",
            "",
            f"Paired companies {p25['n_both']}: share last>earlier **{p25['share_up']:.1%}**, "
            f"median Δ `a_out6` {p25['med_delta']:.4g}. "
            "If this is a ramp, the 0.66 leftover is spend-before-onset, still not mix, still not Q6.",
            "",
            "| slice | n | median a_out6 | p80 |",
            "|---|---:|---:|---:|",
        ]
        for r in p25["rows"]:
            lines.append(
                f"| {r['slice']} | {r['n']} | {r['median']:.4g} | {r['p80']:.4g} |"
            )
        lines += ["", "Do not merge. Do not put this clock in X.", ""]
    if p26:
        lines += [
            "### 26 — mix persistence (acf1, train)",
            "",
            f"`m_fee_share` company-median acf1 **{p26.get('fee_med_company', float('nan')):+.3f}** "
            f"(pooled {p26['fee_acf1']:+.3f} is zeros lining up). "
            f"persist={p26.get('persist')}. Mix cannot answer Q6.",
            "",
            "| feature | pooled acf1 | pairs | median company acf1 | share |acf|≥0.3 |",
            "|---|---:|---:|---:|---:|",
        ]
        for r in p26["rows"]:
            lines.append(
                f"| `{r['col']}` | {r['acf1']:+.3f} | {r['n_pairs']} | "
                f"{r['med_company']:+.3f} | {r['share_persist']:.3f} |"
            )
        lines += ["", "t3 mix already CLOSE. Lag-1 single does not clear 0.565+0.02. Not Q6.", ""]
    if p27:
        lines += [
            "### 27 — outflow quintiles inside size terciles",
            "",
            f"Q1 shield inside size? **{p27.get('shield_inside')}**. "
            "`a_out6` size AUROC 0.891 — if the Q1 gap dies inside terciles, the 0.565 bar is size.",
            "",
            "| size T | out Q | n | n_pos | P(Y=1) | any-fee share |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for r in p27["rows"]:
            lines.append(
                f"| {r['size_t']} | {r['out_q']} | {r['n']} | {r['n_pos']} | "
                f"{r['rate']:.3f} | {r['fee_rate']:.3f} |"
            )
        lines += ["", "| size T | Q1 rate | rest rate | gap |", "|---:|---:|---:|---:|"]
        for g in p27.get("gaps") or []:
            lines.append(
                f"| {g['size_t']} | {g['q1']:.3f} | {g['rest']:.3f} | {g['gap']:+.3f} |"
            )
        lines += ["", "Comparator only. Do not merge M. Do not treat `a_out6` as a clean Q5 why.", ""]
    if p28:
        lines += [
            "### 28 — large ∩ any-fee leftover",
            "",
            f"n={p28['n']} pos={p28['n_pos']} rate {p28['rate']:.1%}. "
            f"KEEP mix? **{p28['keep']}**.",
            "",
            "| feature | CV AUROC ± sd | n |",
            "|---|---:|---:|",
        ]
        for r in p28["rows"]:
            lines.append(
                f"| `{r['col']}` | {r['cv']:.3f} ± {r['sd']:.3f} | {r['n']} |"
            )
        lines += [
            "",
            "If both mix and `a_out6` are ~0.50–0.54 here, the night 0.565 bar is the "
            "no-fee / small-activity shield. Still do not merge.",
            "",
        ]
    return lines


def _extra_registry(p7, p8, p9, p10, p11, p12, p13=None, p14=None, p15=None, p16=None, p17=None) -> list[dict]:
    rows = []
    for r in p7["rows"]:
        rows.append(
            {
                "model": f"leak_support_{r['col']}",
                "y": Y_OWN,
                "metric": f"spearman_{r['vs']}_pos",
                "value": r["rho_pos"],
                "coverage": r["n_pos"] / r["n_all"] if r["n_all"] else float("nan"),
                "notes": (
                    f"support both>0; rho_all={r['rho_all']}; fail_pos={r['fail_pos']}; "
                    f"n_pos={r['n_pos']}; zero-inflation check"
                ),
                "families": "M-vs-F",
            }
        )
    for r in p8["rows"]:
        rows.append(
            {
                "model": f"single_{r['col']}_{r['slice']}",
                "y": r["y"],
                "metric": "auroc",
                "value": r["cv"],
                "coverage": 1.0,
                "notes": f"any-fee vs intensity; CV={r['cv']:.4f}±{r['sd']:.3f}; n={r['n_defined']}",
                "families": "M",
            }
        )
    for r in p9["rows"]:
        rows.append(
            {
                "model": f"m_fee_share_{r['slice']}",
                "y": r["y"],
                "metric": "auroc",
                "value": r["fee_cv"],
                "coverage": 1.0,
                "notes": (
                    f"outflow residual; n={r['n']} pos={r['n_pos']} rate={r['rate']}; "
                    f"out_cv={r.get('out_cv')}"
                ),
                "families": "M",
            }
        )
    rows.append(
        {
            "model": "y9_neither_cell",
            "y": Y_OWN,
            "metric": "share_of_positives",
            "value": p10["n"] / p10["n_2x2"] if p10["n_2x2"] else float("nan"),
            "coverage": 1.0,
            "notes": f"neither high-out nor high-fee; n={p10['n']}/{p10['n_2x2']}",
            "families": "-",
        }
    )
    for r in p11["rows"]:
        rows.append(
            {
                "model": f"y9_base_{r['slice']}",
                "y": r["y"],
                "metric": "base_rate",
                "value": r["rate"],
                "coverage": 1.0,
                "notes": f"n={r['n']} pos={r['n_pos']} cos={r['n_cos']}; never D/E",
                "families": "-",
            }
        )
    rows.append(
        {
            "model": "y9_own_spike_overlap",
            "y": Y_OWN,
            "metric": "spearman",
            "value": p12["rho"],
            "coverage": 1.0,
            "notes": (
                f"both_pos={p12['n_both_pos']} own_only={p12['n_own_only']} "
                f"spk_only={p12['n_spk_only']}; hold_fee_auc={p12['hold_fee_auroc']} LOW_POWER"
            ),
            "families": "-",
        }
    )
    if p13:
        rows.append(
            {
                "model": "m_fee_n_share_vs_anyfin",
                "y": Y_OWN,
                "metric": "auroc_gap",
                "value": p13["gap_vs_presence"],
                "coverage": 1.0,
                "notes": (
                    f"presence_rewrite={p13['presence_rewrite']}; "
                    f"intensity_dead={p13['intensity_dead']}; "
                    f"any_fin_cv={p13['any_fin_cv']:.4f}; nshare={p13['nshare_cv']:.4f}; "
                    f"nshare_fee>0={p13['nshare_pos_cv']:.4f}; comparator only, never X"
                ),
                "families": "M-vs-F",
            }
        )
        for r in p13["rows"]:
            rows.append(
                {
                    "model": f"single_{r['col']}_{r['slice']}",
                    "y": Y_OWN,
                    "metric": "auroc",
                    "value": r["cv"],
                    "coverage": 1.0,
                    "notes": f"count honesty; CV={r['cv']:.4f}±{r['sd']:.3f}; n={r['n']}",
                    "families": "M" if not str(r["col"]).startswith("_any") else "cmp",
                }
            )
    if p14:
        rows.append(
            {
                "model": "any_fin_lag1",
                "y": Y_OWN,
                "metric": "auroc",
                "value": p14["lag1_cv"],
                "coverage": 1.0,
                "notes": (
                    f"comparator only; now={p14['now_cv']:.4f}; q6_keep={p14['q6_keep']}; never X"
                ),
                "families": "cmp",
            }
        )
        for r in p14["cross"]:
            rows.append(
                {
                    "model": f"out_q{r['q']}_fee{r['any_fee']}",
                    "y": Y_OWN,
                    "metric": "y_rate",
                    "value": r["rate"],
                    "coverage": 1.0,
                    "notes": f"n={r['n']} pos={r['n_pos']}; train cuts; never F as X",
                    "families": "A-x-M",
                }
            )
    if p15:
        for r in p15["rows"]:
            rows.append(
                {
                    "model": f"y9_token_{r['slice']}",
                    "y": r["y"],
                    "metric": "base_rate",
                    "value": r["rate"],
                    "coverage": 1.0,
                    "notes": f"n={r['n']} pos={r['n_pos']}; fee vs interest-only; train labeled",
                    "families": "M",
                }
            )
    if p16:
        for r in p16["rows"]:
            rows.append(
                {
                    "model": f"y9_darkfin_{r['slice']}",
                    "y": r["y"],
                    "metric": "base_rate",
                    "value": r["rate"],
                    "coverage": 1.0,
                    "notes": f"n={r['n']} pos={r['n_pos']}; never D/E as X",
                    "families": "-",
                }
            )
    if p17:
        for r in p17["rows"]:
            rows.append(
                {
                    "model": "y9_company_fin_vs_y",
                    "y": r["y"],
                    "metric": "spearman",
                    "value": r["rho"],
                    "coverage": 1.0,
                    "notes": (
                        f"n_cos={r['n_cos']} ever_pos={r['n_ever_pos']}; "
                        f"always_fin_yrate={r['y_rate_always_fin']}; "
                        f"never_fin_yrate={r['y_rate_never_fin']}"
                    ),
                    "families": "-",
                }
            )
    return rows


def decide(p1: dict, p3: dict, p4: dict, p13: dict | None = None) -> dict:
    """KEEP / CLOSE / PARK a mix column as Q5. Merge rec is almost certainly no."""
    own = [r for r in p3["rows"] if r["y"] == Y_OWN]
    out = next((r for r in own if r["col"] == "a_out6"), None)
    out_cv = float(out["cv"]) if out and np.isfinite(out["cv"]) else OUT6_BAR
    leak_fail = {r["col"] for r in p4["rows"] if r["fail"]}
    cands = []
    for r in own:
        if not r["col"].startswith("m_"):
            continue
        gap = float(r["cv"] - out_cv) if np.isfinite(r["cv"]) else float("nan")
        if r["col"] in leak_fail:
            dec = "PARK"
            why = f"leak |ρ|≥{LEAK_RHO} vs F / a_fin_cost"
        elif r["size_park"]:
            dec = "PARK"
            why = f"size AUROC {r['size_auroc']:.3f} ≥ {SIZE_AUROC}"
        elif r["col"] == "m_fin_share":
            dec = "PARK"
            why = "caution: CAT_MAP lump; prefer fee+int; known ρ≈0.96 vs fin/inflow"
        elif (
            r["col"] == "m_fee_n_share"
            and p13
            and (p13.get("presence_rewrite") or p13.get("intensity_dead"))
        ):
            dec = "PARK"
            why = (
                f"numeric gap vs a_out6 {gap:+.3f} but fee-presence rewrite "
                f"(gap vs any-fin {p13.get('gap_vs_presence', float('nan')):+.3f}; "
                f"fee>0 CV {p13.get('nshare_pos_cv', float('nan')):.3f})"
            )
        elif np.isfinite(gap) and gap >= CLEAR_MARGIN:
            dec = "KEEP"
            why = f"CV {r['cv']:.3f} beats a_out6 {out_cv:.3f} by {gap:+.3f}"
        else:
            dec = "CLOSE"
            why = f"ties or loses to outflow (CV {r['cv']:.3f} vs a_out6 {out_cv:.3f}, gap {gap:+.3f})"
        cands.append({**r, "gap_vs_out6": gap, "decision": dec, "reason": why})

    keepers = [c for c in cands if c["decision"] == "KEEP"]
    best_raw = max(
        (c for c in cands if np.isfinite(c["cv"])),
        key=lambda c: c["cv"],
        default=None,
    )
    best_legal = max(
        (c for c in cands if c["decision"] != "PARK" and np.isfinite(c["cv"])),
        key=lambda c: c["cv"],
        default=None,
    )
    if keepers:
        best = max(keepers, key=lambda c: c["cv"])
        headline = "KEEP"
        merge = (
            f"YES-recommend only: `{best['col']}` beats a_out6 by ≥0.02 on {Y_OWN}. "
            "Do not rewrite parquet; parent merge review."
        )
    else:
        headline = "CLOSE" if any(c["decision"] == "CLOSE" for c in cands) else "PARK"
        merge = "NO — do not merge Family M. No m_* clears a_out6 + 0.02 without leak/size."
        if all(c["decision"] == "PARK" for c in cands):
            headline = "PARK"
    story = p1["story"]
    print("\n" + "=" * 72)
    print(f"VERDICT {headline}  story={story}")
    for c in cands:
        print(f"  {c['col']:16s} {c['decision']:5s}  {c['reason']}")
    print(f"  MERGE: {merge}")
    # Spike side is not the KEEP gate; report vs a_op_in (GBM single 0.563)
    spike = [r for r in p3["rows"] if r["y"] == Y_SPIKE]
    op = next((r for r in spike if r["col"] == "a_op_in"), None)
    op_cv = float(op["cv"]) if op and np.isfinite(op["cv"]) else 0.563
    print(f"  spike GBM single a_op_in={op_cv:.3f} (size-parked; not a mix KEEP gate)")
    for r in spike:
        if not r["col"].startswith("m_"):
            continue
        gap = float(r["cv"] - op_cv) if np.isfinite(r["cv"]) else float("nan")
        leak = r["col"] in leak_fail
        print(
            f"  spike {r['col']:16s} CV={r['cv']:.3f} vs a_op_in {gap:+.3f}  "
            f"{'LEAK' if leak else 'ok'}"
        )
    print("=" * 72)
    return {
        "headline": headline,
        "story": story,
        "cands": cands,
        "out6_cv": out_cv,
        "merge": merge,
        "best_mix": best_raw,
        "best_legal": best_legal,
        "spike_op_in": op_cv,
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

    def add(model, y, metric, value, coverage, notes, families="M"):
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

    add(
        "y9_why_story",
        Y_OWN,
        "story",
        {"outflow_tail": 0, "mix_shift": 1, "mixed": 0.5}.get(p1["story"], -1),
        1.0,
        f"{p1['story']}; out_share={p1['out_share']:.3f} fee_share={p1['fee_share']:.3f}; "
        f"never F; M in-memory; no GBM",
        families="M-vs-A",
    )
    for r in p1["rows"]:
        add(
            f"y9_why_{r['flag']}",
            r["y"],
            "share_of_positives",
            r["share_of_pos_defined"],
            r["coverage_of_pos"],
            f"own-p80 high flag; n_hi={r['n_hi']} n_def={r['n_defined']} n_pos={r['n_pos']}; "
            f"train labeled only; never F",
            families="-" if r["flag"] in (Y2, Y4) else ("A" if r["flag"].startswith("a_") else "M"),
        )
    for r in p3["rows"]:
        fam = "A" if r["col"].startswith("a_") or r["col"].startswith("log1p") else "M"
        add(
            f"single_{r['col']}",
            r["y"],
            "auroc",
            r["cv"],
            r["coverage"],
            f"group-fold; sign from train fold; CV={r['cv']:.4f}±{r['sd']:.3f}; "
            f"sign={r['train_sign']:+d}; sizeAUC={r['size_auroc']:.3f}; "
            f"size_park={r['size_park']}; never F; quote CV not holdout",
            families=fam,
        )
    for r in p4["rows"]:
        add(
            f"leak_{r['col']}",
            Y_OWN,
            f"spearman_{r['vs']}",
            r["spearman"],
            r["n"] / max(r["n"], 1),
            f"train any-Y9; pearson={r['pearson']}; fail={r['fail']}; comparator only, not X",
            families="M-vs-F" if r["col"].startswith("m_") else "A-vs-F",
        )
    add(
        "m_fee_vs_m_int",
        Y_OWN,
        "spearman",
        p4["fee_vs_int"]["spearman"],
        1.0,
        f"substitutes?; pearson={p4['fee_vs_int']['pearson']}; train any-Y9",
        families="M",
    )
    for r in p5["rows"]:
        add(
            f"y9_base_{r['slice']}",
            r["y"],
            "base_rate",
            r["rate"],
            1.0,
            f"n={r['n']} pos={r['n_pos']} cos={r['n_cos']}; never D/E as X",
            families="-",
        )
    for r in p6["rows"]:
        add(
            f"single_{r['col']}",
            r["y"],
            "auroc",
            r["cv"],
            r["coverage"],
            f"Q6 honest 1m; lag={r['lag']}; CV={r['cv']:.4f}±{r['sd']:.3f}; never t3",
            families="M",
        )
    for t in p2["tables"]:
        for r in t["rows"]:
            add(
                f"{t['x']}_quintile",
                t["y"],
                f"y_rate_q{r['q']}",
                r["y_rate"],
                r["n"] / t["n"] if t["n"] else float("nan"),
                f"train labeled cuts; n={r['n']} pos={r['n_pos']} interval={r['interval']}; "
                f"monotone={t['monotone_up']} tail_only={t['tail_only']}",
                families="M" if t["x"].startswith("m_") else "A",
            )
    add(
        "y9_why_verdict",
        Y_OWN,
        "decision",
        {"KEEP": 1, "CLOSE": 0, "PARK": -1}.get(verdict["headline"], -9),
        1.0,
        f"{verdict['headline']}; {verdict['merge']}; out6_cv={verdict['out6_cv']:.4f}",
        families="M",
    )
    if extra:
        for r in extra.get("registry") or []:
            add(
                r["model"],
                r.get("y", Y_OWN),
                r["metric"],
                r["value"],
                r.get("coverage", 1.0),
                r.get("notes", ""),
                families=r.get("families", "M"),
            )
    return out


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
    best = verdict.get("best_mix") or {}
    lines = [
        "# Y9 why — mix shift or outflow tail",
        "",
        f"- **When:** {started}",
        f"- **Agent:** `{AGENT}`",
        f"- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        f"- **Re-run:** `python -m analysis.evaluate.y9_why`",
        f"- **Holdout:** 72 companies, seed {FOLD_SEED}. Coverage only. Rates + singles on train.",
        f"- **Y:** `{Y_OWN}` / `{Y_SPIKE}` from `targets.parquet` (not rebuilt). "
        f"Train own-p80 {n_tr[Y_OWN]['n']} / {n_tr[Y_OWN]['n_pos']} / **{n_tr[Y_OWN]['rate']:.2%}**; "
        f"spike {n_tr[Y_SPIKE]['n']} / {n_tr[Y_SPIKE]['n_pos']} / **{n_tr[Y_SPIKE]['rate']:.2%}**.",
        "- **X candidates:** Family M in memory (`m_fee_share`, `m_int_share`; `m_fin_share` caution) "
        "+ `a_out6` / `log1p(a_in3)` / `a_uncat_share`. **Never F. Never `a_fin_cost`. Never D/E.**",
        "- **Quote:** train group-fold CV (5 folds). Holdout is LOW_POWER.",
        "- **Brief:** Q3 turning / Q5 why. Q6 only via honest 1-month lag (t3 mix already CLOSE).",
        "- Not bankruptcy. Not a 0–100.",
        "",
        "## Decision",
        "",
        f"**{verdict['headline']}** Family M as a Y9 Q5 column. "
        "Y9 is **not an outflow tail** and **not a usable mix shift**. Merge: **no**.",
        "",
        verdict["merge"],
        "",
        f"- Best raw `{best.get('col')}` CV **{best.get('cv', float('nan')):.3f}** vs "
        f"`a_out6` **{verdict['out6_cv']:.3f}** — leak (ρ≥0.80 vs `a_fin_cost`). "
        "Legal leftover `m_int_share` 0.525 CLOSE.",
        "- Mix that beats outflow equals any-`a_fin_cost>0` (0.610 vs `m_fee_share` 0.613). "
        "That is the Y. PARK. Never F / `a_fin_cost` as X.",
        "- First-fin month: 47.5% own-p80 / 86.3% spike. 75% of own-p80 pos are later-fin. "
        "The 64 before-first-fin positives are **exactly** 1 month before onset. Not Q6.",
        "- GBM PARK: `a_out6` 0.565 is a no-fee shield (Q1 no-fee 6.7% vs any-fee 22.9%); "
        "it dies on later-fin (0.51) and on large∩any-fee (0.530). "
        "Shield survives inside size terciles. Spike `a_op_in` 0.563 is size (0.94).",
        "- later-fin fee-share is tail-only (15.8% → 22.5%; single 0.550). "
        "Fee vs int ρ=+0.11. Company-median fee-share acf1 −0.055.",
        "- Label stays. Dark = invoiced 14.1%. Never D/E.",
        "",
        "## Pass 1 — decompose positives",
        "",
        "High flags are that company's own expanding p80 (months ≤ t, min 6 finite). "
        "Not a pooled cut. Y2 / Y4 are accepted labels from the same parquet.",
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
        f"Own-p80 positives that are high-`a_out6`: **{p1['out_share']:.1%}**. "
        f"High-`m_fee_share`: **{p1['fee_share']:.1%}**. "
        f"Read: **{p1['story']}** — "
        + (
            "Y9 is mostly a spend tail, not a fee mix shift."
            if p1["story"] == "outflow_tail"
            else (
                "Y9 is a fee/interest mix shift, not just spent-more."
                if p1["story"] == "mix_shift"
                else "neither story dominates."
            )
        ),
        "",
        "## Pass 2 — quintiles (train labeled cuts)",
        "",
    ]
    for tab in p2["tables"]:
        lines += [
            f"### `{tab['x']}` vs `{tab['y']}`",
            "",
            f"n={tab['n']} bins={tab['n_bins']} monotone={tab['monotone_up']} "
            f"tail_only={tab['tail_only']}",
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
        f"`a_out6` on `{Y_OWN}` is the GBM single to beat (**{verdict['out6_cv']:.3f}**, night quote 0.565). "
        f"KEEP a mix column only if it beats that by ≥{CLEAR_MARGIN:.2f} and is not an F-copy / size proxy.",
        "",
        "### Mix-column letters",
        "",
        "| feature | decision | reason |",
        "|---|---|---|",
    ]
    for c in verdict["cands"]:
        lines.append(f"| `{c['col']}` | **{c['decision']}** | {c['reason']} |")
    lines += [
        "",
        "## Pass 4 — leak vs F",
        "",
        f"Fail if Spearman |ρ| ≥ {LEAK_RHO} vs `a_fin_cost` or `f_fc_r` (that is the Y).",
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
        f"`m_fee_share` vs `m_int_share`: Spearman **{p4['fee_vs_int']['spearman']:+.3f}**, "
        f"Pearson {p4['fee_vs_int']['pearson']:+.3f}. "
        + (
            "Substitutes — do not stack."
            if abs(p4["fee_vs_int"]["spearman"]) >= 0.70
            else "Not substitutes; fee and interest can be told apart."
        ),
        "",
        "## Pass 5 — dark 470 vs invoiced 744",
        "",
        "Join-QA population (DuckDB book invoices). **Never D/E as X.**",
        "",
        "| Y | slice | n | n_pos | companies | base rate |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for r in p5["rows"]:
        lines.append(
            f"| `{r['y']}` | {r['slice']} | {r['n']} | {r['n_pos']} | {r['n_cos']} | {r['rate']:.3f} |"
        )
    lines += [
        "",
        "## Pass 6 — honest 1-month Q6",
        "",
        "Only lag-1 is an honest clock (Q6 quoted: longer leads die on the hidden 72). "
        "Do not claim a fee lead from t3 mix.",
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
        "## Mapping (Q3 / Q5 / Q6)",
        "",
        _mapping_paragraph(p1, p3, p6, verdict),
        "",
        "## Parent return",
        "",
        "- Y9 is **not an outflow tail** (31% of own-p80 pos high `a_out6`; 2×2 neither 42%). "
        "It is **not a usable mix shift**: columns that beat `a_out6` 0.565 are fee-presence "
        "rewrites of `a_fin_cost>0` (any-fin 0.610; `m_fee_share` 0.613).",
        "- Best raw single `m_fin_share` **0.618** — leak ρ 0.875 vs `a_fin_cost`. "
        "Best legal `m_int_share` **0.525** (loses to outflow). Leak fail ≥0.80: "
        "`m_fee_share` 0.830, `m_fin_share` 0.875. `m_fee_n_share` 0.739 is a presence rewrite.",
        "- Merge Family M: **no**. KEEP gate (beat `a_out6` by ≥0.02 and not F-copy) is not met.",
        "- Label stays. GBM PARK explained (no-fee shield + size `a_op_in`). Not Q6 "
        "(lag1 0.571; company-median acf1 −0.055; 64 before-first-fin pos are exactly 1m before onset).",
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
    del p3
    fee0 = next((r for r in p6["rows"] if r["y"] == Y_OWN and r["col"] == "m_fee_share"), None)
    fee1 = next((r for r in p6["rows"] if r["y"] == Y_OWN and r["col"] == "m_fee_share_lag1"), None)
    return (
        f"Y9 is who is *turning* on fee+interest / inflow (Q3, FinRegLab NSF/fee). "
        f"The why (Q5) on train is **not an outflow tail** "
        f"({p1['out_share']:.0%} of own-p80 positives are high `a_out6`; "
        f"2×2 neither is {p1.get('neither_share', float('nan')):.0%}). "
        f"It is also **not a usable mix shift**: `m_fee_share` CV "
        f"{(fee0 or {}).get('cv', float('nan')):.3f} looks like it beats `a_out6` "
        f"{verdict['out6_cv']:.3f}, but that is fee-*presence* "
        f"(any-`a_fin_cost` CV 0.610, gap +0.001; on fee>0 support mix drops to 0.55). "
        f"`m_fee_share` / `m_fin_share` fail the |ρ|≥0.80 leak vs `a_fin_cost`. "
        f"Fee and interest shares are not substitutes (ρ 0.11). "
        f"Lag-1 fee-share {(fee1 or {}).get('cv', float('nan')):.3f} does not clear "
        f"`a_out6`+0.02 — not Q6. The 64 before-first-fin own-p80 positives are "
        f"exactly 1 month before onset (algebraic forward label). t3 mix already CLOSE. "
        f"Verdict **{verdict['headline']}**. Merge: no. Never F. Not a 0–100."
    )


def run(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Y9 why — mix vs outflow tail")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--no-md", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    started = datetime.now().isoformat(timespec="minutes")
    wall0 = datetime.now()
    print(f"y9_why start {started} seed={FOLD_SEED} agent={AGENT}")
    print("forbidden X = F / a_fin_cost / a_fc; M in-memory; no GBM; no build_targets")
    print("META.forbidden_x_prefixes", Y9_META.get("forbidden_x_prefixes"))
    print(
        "leakage_check demo",
        leakage_check(
            ["m_fee_share", "a_out6", "a_fin_cost"],
            Y_OWN,
            Y9_META.get("forbidden_x_prefixes"),
        ),
    )

    hold_ids = load_holdout()
    con = connect()
    store = load_store()
    y = load_y()
    mix = load_mix(con, store[["company_id", "period"]])
    train_cos = train_companies(con)
    pop = dark_population(con)
    con.close()

    assert_no_holdout(train_cos)
    print(
        f"dark pop confirm_470={pop['confirm_470']} train_dark={pop['n_train_dark']} "
        f"invoiced={pop['n_book_train']}"
    )
    if not pop["confirm_470"]:
        print("WARN: train dark != 470 — still using DuckDB set, never hardcoded")

    folds = group_folds(train_cos, n=N_FOLDS, seed=FOLD_SEED)
    panel = store.merge(y, on=["company_id", "period"], how="left")
    panel = panel.merge(mix, on=["company_id", "period"], how="left")
    panel = panel.merge(folds[["company_id", "fold", "group_id"]], on="company_id", how="left")
    panel["log1p_a_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").abs())
    panel = add_own_p80(panel, HI_COLS)
    panel = add_lags(panel, ["m_fee_share", "m_int_share", "m_fee_n_share"], (1,))

    is_hold = panel["company_id"].astype(str).isin(hold_ids)
    train_row = (~is_hold) & panel["fold"].notna()
    assert_no_holdout(panel.loc[train_row, "company_id"])

    train_by_y = {}
    n_tr = {}
    for y_col in (Y_OWN, Y_SPIKE):
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
    # keep both Ys on the same train company-month frame for decompose
    p1 = pass1_decompose(lab)
    p2 = pass2_quintiles(panel, train_by_y, args.no_plot)
    p3 = pass3_singles(panel, train_by_y)
    p4 = pass4_leak(panel, train_by_y[Y_OWN] | train_by_y[Y_SPIKE])
    p5 = pass5_dark(panel, train_by_y, pop["train_dark_ids"])
    p6 = pass6_lag1(panel, train_by_y)
    p7 = pass7_zero_leak(panel, train_by_y[Y_OWN] | train_by_y[Y_SPIKE])
    p8 = pass8_any_fee(panel, train_by_y)
    p9 = pass9_outflow_residual(panel, train_by_y)
    p10 = pass10_neither(lab)
    p11 = pass11_dark_mix(panel, train_by_y, pop)
    p12 = pass12_label_overlap(panel, train_row, is_hold)
    p13 = pass13_count_honesty(panel, train_by_y)
    p14 = pass14_presence_clock(panel, train_by_y)
    p15 = pass15_interest_only(panel, train_by_y)
    p16 = pass16_dark_presence(panel, train_by_y, pop)
    p17 = pass17_company_trait(panel, train_by_y)
    p18 = pass18_always_fin_intensity(panel, train_by_y)
    p19 = pass19_fee_x_out(panel, train_by_y)
    p20 = pass20_onset(panel, train_by_y)
    p21 = pass21_later_and_before(panel, train_by_y)
    p22 = pass22_later_off_and_before_q(panel, train_by_y)
    p23 = pass23_months_to_onset(panel, train_by_y)
    p24 = pass24_later_intensity(panel, train_by_y)
    p25 = pass25_last_pre_fee(panel, train_by_y)
    p26 = pass26_fee_acf(panel, train_row)
    p27 = pass27_out_within_size(panel, train_by_y)
    p28 = pass28_large_fee_out(panel, train_by_y)
    verdict = decide(p1, p3, p4, p13)

    extra = {
        "md_lines": extra_md_lines(
            p1, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21, p22, p23, p24, p25, p26, p27, p28
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
        "registry": _extra_registry(p7, p8, p9, p10, p11, p12, p13, p14, p15, p16, p17),
    }
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if not args.no_registry:
        rows = registry_rows(p1, p2, p3, p4, p5, p6, verdict, ts, extra=extra)
        append_registry(rows)
        print(f"appended {len(rows)} registry rows (train-only metrics)")
    if not args.no_md:
        write_md(started, n_tr, p1, p2, p3, p4, p5, p6, verdict, extra)

    elapsed = (datetime.now() - wall0).total_seconds()
    best = verdict.get("best_mix") or {}
    legal = verdict.get("best_legal") or {}
    quote = {
        "story": p1["story"],
        "out_share": p1["out_share"],
        "fee_share": p1["fee_share"],
        "neither_share": p1.get("neither_share"),
        "headline": verdict["headline"],
        "out6_cv": verdict["out6_cv"],
        "best_raw": best.get("col"),
        "best_raw_cv": best.get("cv"),
        "best_legal": legal.get("col"),
        "best_legal_cv": legal.get("cv"),
        "gap_raw": best.get("gap_vs_out6"),
        "merge": verdict["merge"],
        "elapsed_s": elapsed,
    }
    if quote["headline"] != "CLOSE":
        print(f"WARN: expected CLOSE, got {quote['headline']}")
    if quote.get("best_legal") not in (None, "m_int_share"):
        print(f"WARN: expected best_legal m_int_share, got {quote.get('best_legal')}")
    if quote.get("best_raw") not in (None, "m_fin_share", "m_fee_share"):
        print(f"WARN: unexpected best_raw {quote.get('best_raw')}")
    if quote["headline"] != "CLOSE":
        print(f"WARN: expected CLOSE, got {quote['headline']}")
    if quote.get("best_legal") not in (None, "m_int_share"):
        print(f"WARN: expected best_legal m_int_share, got {quote.get('best_legal')}")
    if quote.get("best_raw") not in (None, "m_fin_share", "m_fee_share"):
        print(f"WARN: unexpected best_raw {quote.get('best_raw')}")
    print("\nQUOTE")
    print(json.dumps(quote, indent=2, default=str))
    print(f"elapsed {elapsed:.1f}s — stay on this module for more cuts")
    return {
        "panel": panel,
        "train_by_y": train_by_y,
        "train_row": train_row,
        "is_hold": is_hold,
        "pop": pop,
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
        "verdict": verdict,
        "n_tr": n_tr,
        "quote": quote,
        "started": started,
    }


if __name__ == "__main__":
    run()
