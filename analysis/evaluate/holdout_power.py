"""Holdout AUROC power for accepted binary Ys. No model fitting.

For each accepted binary that already has a ``build()``, count train and
holdout labeled n / positives / base rate, then simulate a bootstrap 95%
CI width for AUROC if the true AUC were 0.65 at the holdout positive
count. Holdout companies never enter a fit (there is no fit).

    python -m analysis.evaluate.holdout_power
"""
from __future__ import annotations

import csv
import importlib
import math
import sys
from datetime import datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import auroc, load_holdout
from analysis.features.common import ANALYSIS, DATA, connect, train_mask
from analysis.features.grid import monthly_grid

REGISTRY = ANALYSIS / "experiments" / "registry.csv"
STORE = DATA / "feature_store" / "targets.parquet"
AGENT = "22a1f47e"

TRUE_AUC = 0.65
N_BOOT = 4000
BOOT_SEED = 20260918
LOW_POWER_POS = 30
# A true-0.65 model is distinguishable from chance if the typical 95% CI
# sits above 0.50. Width 0.20 → lower bound ≈ 0.55.
USABLE_MAX_WIDTH = 0.20

# Frozen accepted binaries that already have a build() (wave 1–2 notes).
# y6_missed_payroll is ACCEPTED in wave2_slot7 even if y_acceptance.csv
# still marks it shipped=0.
ACCEPTED_BINARY = (
    "y2_neg_2of3",
    "y5_ap_od30_ownp80",
    "y5_ar_od30_sust",
    "y4_ds_r_double",
    "y3_recover_cash_6m",
    "y6_missed_payroll",
)

Y_MODULES = (
    ("analysis.targets.y2_stress", ("y2_neg_2of3",)),
    ("analysis.targets.y5_payment", ("y5_ap_od30_ownp80", "y5_ar_od30_sust")),
    ("analysis.targets.y4_debt", ("y4_ds_r_double",)),
    ("analysis.targets.y3_recovery", ("y3_recover_cash_6m",)),
    ("analysis.targets.y6_activity", ("y6_missed_payroll",)),
)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    out["period"] = pd.to_datetime(out["period"])
    return out


def load_y_panel() -> pd.DataFrame:
    """Prefer assembled targets.parquet; otherwise call each module build()."""
    need = set(ACCEPTED_BINARY)
    if STORE.exists():
        raw = pd.read_parquet(STORE)
        have = set(raw.columns)
        if need <= have and {"company_id", "period"} <= have:
            panel = _keys(raw[["company_id", "period", *ACCEPTED_BINARY]])
            print(f"Y panel from {STORE} shape={panel.shape}")
            return panel
        missing = sorted(need - have)
        print(f"{STORE} missing {missing}; falling back to build()")

    con = connect()
    grid = monthly_grid(con)
    grid = _keys(grid[["company_id", "period"]])
    panel = grid.copy()
    for modname, cols in Y_MODULES:
        mod = importlib.import_module(modname)
        if not hasattr(mod, "build"):
            raise RuntimeError(f"{modname} has no build()")
        part = _keys(mod.build(con, grid.copy()))
        extra = [c for c in cols if c in part.columns]
        missing = [c for c in cols if c not in part.columns]
        if missing:
            raise RuntimeError(f"{modname} build() missing {missing}")
        panel = panel.merge(part[["company_id", "period", *extra]], on=["company_id", "period"], how="left")
        print(f"built {modname} cols={extra}")
    con.close()
    return panel


def split_counts(df: pd.DataFrame, col: str, hold: set[str]) -> dict:
    """Labeled-row counts on train vs holdout. n is non-null Y only."""
    y = pd.to_numeric(df[col], errors="coerce")
    cid = df["company_id"].astype(str)
    is_hold = cid.isin(hold)
    rows = {}
    for name, mask in (("train", ~is_hold), ("hold", is_hold)):
        lab = mask & y.notna()
        yy = y[lab]
        n = int(lab.sum())
        n_pos = int((yy == 1).sum())
        n_neg = int((yy == 0).sum())
        n_cos = int(cid[lab].nunique())
        n_pos_cos = int(cid[lab & (y == 1)].nunique())
        rows[name] = {
            "n": n,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "base_rate": (n_pos / n) if n else float("nan"),
            "n_cos": n_cos,
            "n_pos_cos": n_pos_cos,
        }
    rows["hold_months"] = int(is_hold.sum())
    rows["train_months"] = int((~is_hold).sum())
    return rows


def binormal_gap(auc: float) -> float:
    """Mean gap for two unit-variance Gaussians with P(S+ > S-) = auc."""
    if not 0.0 < auc < 1.0:
        raise ValueError("auc must be in (0, 1)")
    return math.sqrt(2.0) * NormalDist().inv_cdf(auc)


def simulate_binormal(n_pos: int, n_neg: int, auc: float, rng: np.random.Generator):
    d = binormal_gap(auc)
    y = np.concatenate([np.zeros(n_neg, dtype=float), np.ones(n_pos, dtype=float)])
    s = np.concatenate([rng.normal(0.0, 1.0, n_neg), rng.normal(d, 1.0, n_pos)])
    return y, s


def auroc_np(y: np.ndarray, s: np.ndarray) -> float:
    """Mann–Whitney AUROC with 0.5 for ties (same as protocol.auroc)."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(s, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    y = y[mask]
    s = s[mask]
    pos = s[y == 1.0]
    neg = s[y == 0.0]
    n1 = int(pos.size)
    n0 = int(neg.size)
    if n1 == 0 or n0 == 0:
        return float("nan")
    neg.sort()
    less = np.searchsorted(neg, pos, side="left")
    more = np.searchsorted(neg, pos, side="right")
    return float((less.sum() + 0.5 * (more - less).sum()) / (n1 * n0))


def bootstrap_auroc_ci(
    n_pos: int,
    n_neg: int,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    auc: float = TRUE_AUC,
) -> tuple[float, float, float, int]:
    """Parametric bootstrap 95% CI for AUROC at holdout n_pos / n_neg.

    Repeatedly draw binormal scores with true AUC ``auc`` and take the
    2.5–97.5 percentile range of the AUROC estimator. That range is the
    CI width a holdout evaluation would see; it stays centered on 0.65
    instead of wandering with one lucky sample.
    """
    if n_pos < 1 or n_neg < 1:
        return float("nan"), float("nan"), float("nan"), 0
    aucs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        y, s = simulate_binormal(n_pos, n_neg, auc, rng)
        aucs[i] = auroc_np(y, s)
    ok = aucs[np.isfinite(aucs)]
    if ok.size < 50:
        return float("nan"), float("nan"), float("nan"), int(ok.size)
    lo, hi = np.quantile(ok, [0.025, 0.975])
    return float(lo), float(hi), float(hi - lo), int(ok.size)


def hanley_mcneil_width(n_pos: int, n_neg: int, auc: float) -> float:
    """Analytic 95% CI width (Hanley & McNeil 1982). Sanity check only."""
    if n_pos < 1 or n_neg < 1 or not 0.0 < auc < 1.0:
        return float("nan")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    if var <= 0:
        return float("nan")
    return float(2.0 * 1.959963984540054 * math.sqrt(var))


def power_for_y(col: str, counts: dict, rng: np.random.Generator) -> dict:
    ho = counts["hold"]
    n_pos = int(ho["n_pos"])
    n_neg = int(ho["n_neg"])
    low = n_pos < LOW_POWER_POS
    if n_pos < 1 or n_neg < 1:
        lo = hi = width = float("nan")
        n_ok = 0
        sim_auc = float("nan")
    else:
        lo, hi, width, n_ok = bootstrap_auroc_ci(n_pos, n_neg, rng, N_BOOT, TRUE_AUC)
        sim_auc = 0.5 * (lo + hi)
    analytic = hanley_mcneil_width(n_pos, n_neg, TRUE_AUC)
    usable = (not low) and np.isfinite(width) and width <= USABLE_MAX_WIDTH
    return {
        "y": col,
        "train_n": counts["train"]["n"],
        "train_pos": counts["train"]["n_pos"],
        "train_rate": counts["train"]["base_rate"],
        "train_cos": counts["train"]["n_cos"],
        "train_pos_cos": counts["train"]["n_pos_cos"],
        "hold_n": ho["n"],
        "hold_pos": n_pos,
        "hold_neg": n_neg,
        "hold_rate": ho["base_rate"],
        "hold_cos": ho["n_cos"],
        "hold_pos_cos": ho["n_pos_cos"],
        "hold_months": counts["hold_months"],
        "coverage": (ho["n"] / counts["hold_months"]) if counts["hold_months"] else float("nan"),
        "sim_auc": sim_auc,
        "ci_lo": lo,
        "ci_hi": hi,
        "ci_width": width,
        "analytic_width": analytic,
        "n_boot_ok": n_ok,
        "flag": "LOW_POWER" if low else "OK",
        "usable": bool(usable),
    }


def verdict(rows: list[dict]) -> str:
    usable = [r["y"] for r in rows if r["usable"]]
    ok = [r["y"] for r in rows if r["flag"] != "LOW_POWER"]
    if usable:
        return (
            "Holdout model claims tonight: "
            + ", ".join(usable)
            + f" (hold_pos>={LOW_POWER_POS} and bootstrap 95% CI width<={USABLE_MAX_WIDTH:.2f} at true AUC={TRUE_AUC})."
        )
    if ok:
        return (
            "No Y is tight enough for a holdout AUROC claim tonight "
            f"(need width<={USABLE_MAX_WIDTH:.2f}). "
            + ", ".join(ok)
            + " clear the n_pos>=30 floor only."
        )
    return (
        "No accepted binary is usable for holdout AUROC claims tonight: "
        f"every Y has holdout positives < {LOW_POWER_POS} (LOW_POWER). "
        "Use train group-fold CV; treat holdout AUROC as a noisy diagnostic."
    )


def format_table(rows: list[dict]) -> str:
    header = (
        "| y | train n / pos / rate | hold n / pos / rate | hold pos cos | "
        "AUC=0.65 boot 95% CI | width | analytic | flag |"
    )
    sep = (
        "|---|---:|---:|---:|---|---:|---:|---|"
    )
    lines = [header, sep]
    for r in rows:
        tr = f"{r['train_n']:,} / {r['train_pos']:,} / {r['train_rate']:.1%}"
        ho = f"{r['hold_n']:,} / {r['hold_pos']:,} / {r['hold_rate']:.1%}"
        if np.isfinite(r["ci_width"]):
            ci = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            w = f"{r['ci_width']:.3f}"
        else:
            ci = "—"
            w = "—"
        aw = f"{r['analytic_width']:.3f}" if np.isfinite(r["analytic_width"]) else "—"
        lines.append(
            f"| `{r['y']}` | {tr} | {ho} | {r['hold_pos_cos']} | {ci} | {w} | {aw} | {r['flag']} |"
        )
    return "\n".join(lines)


def append_registry(rows: list[dict]) -> None:
    """Append rows. Skip keys (agent, y, model, split, metric) already present."""
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    with REGISTRY.open("r", encoding="utf-8") as f:
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
        print("registry: nothing new to append")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow({k: r.get(k, "") for k in header})
    print(f"appended {len(fresh)} registry rows")


def _fmt(v) -> str:
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        return f"{float(v):.6g}"
    return "" if v is None else str(v)


def _reg_row(ts: str, y: str, metric: str, value, coverage, notes: str) -> dict:
    return {
        "ts": ts,
        "round": "R3",
        "wave": 3,
        "agent": AGENT,
        "x_families": "-",
        "y": y,
        "model": "holdout_power",
        "split": "holdout",
        "metric": metric,
        "value": _fmt(value),
        "coverage": (
            f"{float(coverage):.4f}"
            if isinstance(coverage, (int, float, np.floating)) and np.isfinite(coverage)
            else ""
        ),
        "notes": notes,
    }


def registry_rows(ts: str, rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        notes = (
            f"true_auc={TRUE_AUC}; n_boot={N_BOOT}; seed={BOOT_SEED}; "
            f"sim_auc={r['sim_auc']:.4f}; ci=[{r['ci_lo']:.4f},{r['ci_hi']:.4f}]; "
            f"analytic_w={r['analytic_width']:.4f}; "
            f"train={r['train_n']}/{r['train_pos']}/{r['train_rate']:.4f}; "
            f"hold_cos={r['hold_cos']}; hold_pos_cos={r['hold_pos_cos']}; "
            f"flag={r['flag']}; usable={int(r['usable'])}"
        )
        cov = r["coverage"]
        out.append(_reg_row(ts, r["y"], "n", r["hold_n"], cov, notes))
        out.append(_reg_row(ts, r["y"], "n_pos", r["hold_pos"], cov, notes))
        out.append(_reg_row(ts, r["y"], "base_rate", r["hold_rate"], cov, notes))
        out.append(_reg_row(ts, r["y"], "auroc_ci95_width", r["ci_width"], cov, notes))
        out.append(_reg_row(ts, r["y"], "low_power", int(r["flag"] == "LOW_POWER"), cov, notes))
    return out


def run() -> pd.DataFrame:
    _chk = np.random.default_rng(0)
    y0, s0 = simulate_binormal(30, 70, TRUE_AUC, _chk)
    if abs(auroc_np(y0, s0) - auroc(y0, s0)) > 1e-12:
        raise RuntimeError("auroc_np does not match protocol.auroc")

    hold = load_holdout()
    panel = load_y_panel()
    is_tr = train_mask(panel["company_id"])
    print(
        f"panel={panel.shape} train_cm={int(is_tr.sum())} hold_cm={int((~is_tr).sum())} "
        f"hold_cos={len(hold)}"
    )

    rng = np.random.default_rng(BOOT_SEED)
    rows = []
    for col in ACCEPTED_BINARY:
        if col not in panel.columns:
            raise RuntimeError(f"missing {col}")
        counts = split_counts(panel, col, hold)
        rec = power_for_y(col, counts, rng)
        rows.append(rec)
        print(
            f"{col}: train {rec['train_n']}/{rec['train_pos']} ({rec['train_rate']:.3f}) "
            f"hold {rec['hold_n']}/{rec['hold_pos']} ({rec['hold_rate']:.3f}) "
            f"width={rec['ci_width']:.3f} {rec['flag']}"
        )

    tbl = format_table(rows)
    print()
    print(tbl)
    print()
    print(verdict(rows))

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    append_registry(registry_rows(ts, rows))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run()
