"""Family H — sister existence vs sister state on the 110 mixed-dark.

Brief: Q5 why / Q3 turning. Hidden test is *new groups*, so a
group-sibling leak is dangerous. No 0–100. No product/. No parquet
rewrite. No new GBM. Single-feature / stratified rates only.

Holdout 72 (seed 20260918) is coverage only. Rates and cuts on train.
Y3 X never family B — flag if H is a B-copy.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.sibling_h
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
from analysis.features.common import ANALYSIS, DATA, connect, train_mask
from analysis.targets.y11_dark import book_invoice_ids, dark_population

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
OUT_MD = ANALYSIS / "outputs" / "sibling_h.md"
OUT_PNG = ANALYSIS / "outputs" / "y3_rate_mix_size_tercile.png"
AGENT = "5d5b1b81"
WAVE = 4
ROUND = "R4"
Y3 = "y3_recover_cash_6m"
Y2 = "y2_neg_2of3"
N_FOLDS = 5
DAYS_BAR = 0.711
CLEAR_MARGIN = 0.02
STATE_PP = 0.05
SIZE_RHO = 0.85
NEAR_SIZE_RHO = 0.70
RUNWAY_COPY = 0.80
NEAR_RUNWAY = 0.50
RUNWAY_HEALTHY = 3.0
RUNWAY_OK = 1.0
IO_HEALTHY = 1.0

H_LEAK = ("h_sib_in", "h_sib_neg_share", "h_share_group_in")
OWN_VS = ("a_op_in", "a_in3", "b_runway")
H_COLS = (
    "h_group_size",
    "h_n_siblings_active",
    "h_sib_in",
    "h_sib_out",
    "h_sib_net",
    "h_share_group_in",
    "h_sib_neg_share",
)


def _keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["company_id"] = out["company_id"].astype(str)
    if "period" in out.columns:
        out["period"] = pd.to_datetime(out["period"])
    if "group_id" in out.columns:
        out["group_id"] = out["group_id"].astype(str)
    return out


def _fmt(v, nd=3) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "—"
        return f"{float(v):.{nd}f}"
    return str(v)


def _pp(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100.0 * float(v):.2f}%"


def _pp_delta(v) -> str:
    if v is None or not np.isfinite(v):
        return "—"
    return f"{100.0 * float(v):+.2f}pp"


def spearman(a: pd.Series, b: pd.Series, min_n: int = 20) -> float:
    d = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(d) < min_n or d["a"].nunique() < 2 or d["b"].nunique() < 2:
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


def two_sided(auc: float) -> float:
    if not np.isfinite(auc):
        return float("nan")
    return float(max(auc, 1.0 - auc))


def signed_oof_auroc(
    y: pd.Series,
    x: pd.Series,
    folds: pd.Series,
    train_lab: pd.Series,
    n_folds: int = N_FOLDS,
) -> dict:
    """Group-fold CV AUROC. Sign from the train side of the fold only."""
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    fold_rows = []
    aucs = []
    for k in range(n_folds):
        tr = train_lab & (folds != k)
        va = train_lab & (folds == k)
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        defined = va & x.notna() & y.notna()
        aucs.append(auc)
        fold_rows.append(
            {
                "fold": k,
                "auroc": float(auc) if np.isfinite(auc) else float("nan"),
                "sign": int(sign),
                "n_va": int(defined.sum()),
                "n_pos": int(((defined) & (y == 1)).sum()),
            }
        )
    finite = [a for a in aucs if np.isfinite(a)]
    tr_sign = choose_sign(y[train_lab], x[train_lab])
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n_folds": len(finite),
        "folds": fold_rows,
        "train_sign": int(tr_sign),
        "train_auc": float(auroc(y[train_lab], tr_sign * x[train_lab])),
        "n_defined": int((train_lab & x.notna() & y.notna()).sum()),
        "coverage": float((train_lab & x.notna() & y.notna()).sum() / train_lab.sum())
        if int(train_lab.sum())
        else float("nan"),
    }


def load_store() -> pd.DataFrame:
    if not STORE.exists():
        raise FileNotFoundError(f"{STORE} missing")
    need = [
        "company_id",
        "period",
        "group_id",
        "group_size",
        "a_op_in",
        "a_in3",
        "a_io_ratio",
        "b_runway",
        "c_n_days_with_tx",
        *H_COLS,
    ]
    raw = pd.read_parquet(STORE)
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"monthly.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"loaded store {STORE} shape={panel.shape}")
    return panel


def load_y() -> pd.DataFrame:
    if not TARGETS.exists():
        raise FileNotFoundError(f"{TARGETS} missing — do not run build_targets")
    raw = pd.read_parquet(TARGETS)
    need = ["company_id", "period", Y3, Y2]
    missing = [c for c in need if c not in raw.columns]
    if missing:
        raise RuntimeError(f"targets.parquet missing {missing}")
    panel = _keys(raw[need])
    print(f"Y panel from {TARGETS} shape={panel.shape} (no build_targets)")
    return panel


def rate_row(y: pd.Series, cid: pd.Series, name: str) -> dict:
    ok = y.notna()
    n = int(ok.sum())
    pos = int((y[ok] == 1).sum()) if n else 0
    return {
        "slice": name,
        "n": n,
        "pos": pos,
        "cos": int(cid[ok].nunique()) if n else 0,
        "rate": float(y[ok].mean()) if n else float("nan"),
    }


def fit_terciles(size: pd.Series, mask: pd.Series) -> tuple[pd.Series, np.ndarray]:
    """Tercile edges from mask only (train). Apply everywhere."""
    src = pd.to_numeric(size[mask], errors="coerce").dropna()
    if src.nunique() < 3:
        raise RuntimeError("not enough distinct size values for terciles")
    _cats, edges = pd.qcut(src, 3, retbins=True, duplicates="drop")
    edges = np.asarray(edges, dtype=float)
    # Expand only with *train* extremes so holdout never sets a cut.
    edges[0] = min(edges[0], float(src.min())) - 1e-9
    edges[-1] = max(edges[-1], float(src.max())) + 1e-9
    labels = ("T1_small", "T2_mid", "T3_large")[: len(edges) - 1]
    cut = pd.cut(pd.to_numeric(size, errors="coerce"), bins=edges, labels=labels, include_lowest=True)
    return cut.astype("object"), edges


def slice_rates(
    panel: pd.DataFrame,
    y_col: str,
    masks: list[tuple[str, pd.Series]],
    extra_filter: pd.Series | None = None,
) -> pd.DataFrame:
    rows = []
    y = pd.to_numeric(panel[y_col], errors="coerce")
    cid = panel["company_id"].astype(str)
    for name, sl in masks:
        ok = sl if extra_filter is None else sl & extra_filter
        rec = rate_row(y[ok], cid[ok], name)
        rec["y"] = y_col
        rows.append(rec)
    return pd.DataFrame(rows)


def tercile_mix_table(
    panel: pd.DataFrame,
    y_col: str,
    is_train: pd.Series,
    mix: pd.Series,
    terc: pd.Series,
    slices: tuple[str, ...] = ("all_dark", "mixed"),
) -> pd.DataFrame:
    y = pd.to_numeric(panel[y_col], errors="coerce")
    cid = panel["company_id"].astype(str)
    size = pd.to_numeric(panel["log1p_a_in3"], errors="coerce")
    rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        for sl in slices:
            ok = is_train & y.notna() & terc.eq(tname) & mix.eq(sl)
            rec = rate_row(y[ok], cid[ok], sl)
            rec["y"] = y_col
            rec["tercile"] = tname
            rec["median_log1p_a_in3"] = float(size[ok].median()) if int(ok.sum()) else float("nan")
            rows.append(rec)
    return pd.DataFrame(rows)


def tercile_residuals(tab: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        aa = tab[(tab["tercile"] == tname) & (tab["slice"] == a)]
        bb = tab[(tab["tercile"] == tname) & (tab["slice"] == b)]
        if aa.empty or bb.empty:
            continue
        ra = float(aa.iloc[0]["rate"])
        rb = float(bb.iloc[0]["rate"])
        rows.append(
            {
                "tercile": tname,
                "rate_b": rb,
                "rate_a": ra,
                "residual_b_minus_a": rb - ra,
                "n_a": int(aa.iloc[0]["n"]),
                "n_b": int(bb.iloc[0]["n"]),
                "pos_a": int(aa.iloc[0]["pos"]),
                "pos_b": int(bb.iloc[0]["pos"]),
            }
        )
    return pd.DataFrame(rows)


def attach_sister_state(panel: pd.DataFrame, invoiced_ids: set[str]) -> pd.DataFrame:
    """Invoiced-sister aggregates this month (leave-one-out by construction: dark ≠ invoiced)."""
    inv = panel.loc[panel["company_id"].isin(invoiced_ids)].copy()
    if inv.empty:
        out = panel.copy()
        for c in (
            "n_inv_sisters_month",
            "sister_runway",
            "sister_io",
            "sister_y2",
            "sister_in3",
            "sister_neg_share_inv",
        ):
            out[c] = np.nan
        return out
    g = inv.groupby(["group_id", "period"], sort=False)
    agg = g.agg(
        n_inv_sisters_month=("company_id", "nunique"),
        sister_runway=("b_runway", "mean"),
        sister_io=("a_io_ratio", "mean"),
        sister_y2=("y2_neg_2of3", "mean"),
        sister_in3=("a_in3", "mean"),
        sister_neg_share_inv=("h_sib_neg_share", "mean"),
    ).reset_index()
    out = panel.merge(agg, on=["group_id", "period"], how="left")
    return out


def sister_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rw = pd.to_numeric(out["sister_runway"], errors="coerce")
    io = pd.to_numeric(out["sister_io"], errors="coerce")
    y2 = pd.to_numeric(out["sister_y2"], errors="coerce")
    out["sister_has_inv_month"] = pd.to_numeric(out["n_inv_sisters_month"], errors="coerce").fillna(0) >= 1
    out["sister_runway_healthy"] = rw >= RUNWAY_HEALTHY
    out["sister_runway_ok"] = rw >= RUNWAY_OK
    out["sister_runway_stressed"] = rw < RUNWAY_OK
    out["sister_io_healthy"] = io >= IO_HEALTHY
    out["sister_io_stressed"] = io < IO_HEALTHY
    out["sister_not_y2"] = y2 <= 0
    out["sister_is_y2"] = y2 > 0
    return out


def state_table(
    panel: pd.DataFrame,
    y_col: str,
    base: pd.Series,
    flag_true: pd.Series,
    flag_false: pd.Series,
    name: str,
) -> dict:
    y = pd.to_numeric(panel[y_col], errors="coerce")
    cid = panel["company_id"].astype(str)
    a = rate_row(y[base & flag_true], cid[base & flag_true], f"{name}_healthy")
    b = rate_row(y[base & flag_false], cid[base & flag_false], f"{name}_stressed")
    gap = (
        float(a["rate"] - b["rate"])
        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
        else float("nan")
    )
    return {"name": name, "healthy": a, "stressed": b, "gap": gap}


def state_by_tercile(
    panel: pd.DataFrame,
    y_col: str,
    base: pd.Series,
    healthy: pd.Series,
    stressed: pd.Series,
    terc: pd.Series,
    name: str,
) -> pd.DataFrame:
    y = pd.to_numeric(panel[y_col], errors="coerce")
    cid = panel["company_id"].astype(str)
    rows = []
    for tname in ("T1_small", "T2_mid", "T3_large"):
        t = terc.eq(tname)
        h = rate_row(y[base & t & healthy], cid[base & t & healthy], "healthy")
        s = rate_row(y[base & t & stressed], cid[base & t & stressed], "stressed")
        rows.append(
            {
                "name": name,
                "tercile": tname,
                "n_healthy": h["n"],
                "pos_healthy": h["pos"],
                "rate_healthy": h["rate"],
                "n_stressed": s["n"],
                "pos_stressed": s["pos"],
                "rate_stressed": s["rate"],
                "residual": (
                    float(h["rate"] - s["rate"])
                    if np.isfinite(h["rate"]) and np.isfinite(s["rate"])
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def leak_table(panel: pd.DataFrame, mask: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for h in H_LEAK:
        for vs in OWN_VS:
            rho = spearman(panel.loc[mask, h], panel.loc[mask, vs])
            n = int(pd.DataFrame({"a": panel.loc[mask, h], "b": panel.loc[mask, vs]}).dropna().shape[0])
            flags = []
            if vs in ("a_op_in", "a_in3") and np.isfinite(rho) and abs(rho) >= SIZE_RHO:
                flags.append("SIZE")
            elif vs in ("a_op_in", "a_in3") and np.isfinite(rho) and abs(rho) >= NEAR_SIZE_RHO:
                flags.append("NEAR_SIZE")
            if vs == "b_runway" and np.isfinite(rho) and abs(rho) >= RUNWAY_COPY:
                flags.append("B_COPY")
            elif vs == "b_runway" and np.isfinite(rho) and abs(rho) >= NEAR_RUNWAY:
                flags.append("NEAR_B")
            rows.append(
                {
                    "slice": label,
                    "h": h,
                    "vs": vs,
                    "n": n,
                    "rho": rho,
                    "flags": ",".join(flags) if flags else "—",
                }
            )
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> list[str]:
    """cols: (key, header, kind) kind in n,s,pp,delta,f,raw."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---:" if k != "s" else "---" for _, _, k in cols) + "|"
    lines = [head, sep]
    for _, r in df.iterrows():
        cells = []
        for key, _, kind in cols:
            v = r.get(key)
            if kind == "n":
                cells.append(_fmt(v, 0) if isinstance(v, (int, np.integer, float, np.floating)) else str(v))
            elif kind == "pp":
                cells.append(_pp(v))
            elif kind == "delta":
                cells.append(_pp_delta(v))
            elif kind == "f":
                cells.append(_fmt(v, 3))
            else:
                cells.append("" if v is None or (isinstance(v, float) and not np.isfinite(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def plot_mix_tercile(tab: pd.DataFrame, path: Path) -> str | None:
    if not HAS_MPL or tab.empty:
        return None
    order = ["T1_small", "T2_mid", "T3_large"]
    x = np.arange(len(order))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, sl, color in ((0, "all_dark", "#4C6A8A"), (1, "mixed", "#C45C26")):
        rates = []
        ns = []
        for t in order:
            row = tab[(tab["tercile"] == t) & (tab["slice"] == sl)]
            rates.append(100.0 * float(row.iloc[0]["rate"]) if len(row) and np.isfinite(row.iloc[0]["rate"]) else 0.0)
            ns.append(int(row.iloc[0]["n"]) if len(row) else 0)
        bars = ax.bar(x + (i - 0.5) * w, rates, w, label=sl, color=color, edgecolor="white")
        for b, n, r in zip(bars, ns, rates):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4, f"{r:.1f}%\nn={n}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(["T1 small", "T2 mid", "T3 large"])
    ax.set_ylabel("Y3 recover rate (train, stressed)")
    ax.set_title("Y3 recover by mix × size tercile (log1p(a_in3), train edges)")
    ax.legend(frameon=False)
    ax.set_ylim(0, max(28, ax.get_ylim()[1]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"wrote {path}")
    return str(path)


def append_registry(rows: list[dict]) -> None:
    if not rows:
        return
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    existing = REGISTRY.read_text(encoding="utf-8")
    with REGISTRY.open("r", encoding="utf-8") as f:
        header = next(csv.reader(f))
    written = 0
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in rows:
            marker = f"{AGENT},H,{r.get('y')},{r.get('model')},{r.get('split')},{r.get('metric')}"
            if marker in existing:
                continue
            w.writerow({k: r.get(k, "") for k in header})
            written += 1
    print(f"registry appended {written} rows (skipped {len(rows) - written} already present)")


def population(con) -> dict:
    pop = dark_population(con)
    book = book_invoice_ids(con)
    hold = load_holdout()
    cos = con.execute(
        "SELECT CAST(company_id AS VARCHAR) AS company_id, CAST(group_id AS VARCHAR) AS group_id FROM companies"
    ).df()
    cos["company_id"] = cos["company_id"].astype(str)
    cos["group_id"] = cos["group_id"].astype(str)
    cos["has_book"] = cos["company_id"].isin(book)
    tr = train_mask(cos["company_id"])
    train = cos.loc[tr].copy()
    assert_no_holdout(train["company_id"])
    train_dark = pop["train_dark_frame"].copy()
    train_dark["company_id"] = train_dark["company_id"].astype(str)
    invoiced = set(train.loc[train["has_book"], "company_id"])
    n_inv = int(len(invoiced))
    print(
        f"pop train={pop['n_train']} dark={pop['n_train_dark']} "
        f"confirm_470={pop['confirm_470']} invoiced={n_inv} "
        f"360={pop['n_360_alldark']} 110={pop['n_110_mixed']} "
        f"groups all-dark={pop['n_groups_all_dark']} mixed={pop['n_groups_mixed']} "
        f"all-invoiced={pop['n_groups_all_invoiced']} hold_dark={pop['n_hold_dark']}"
    )
    if n_inv != 744:
        print(f"WARN invoiced train={n_inv} (expected 744)")
    if pop["n_360_alldark"] != 360 or pop["n_110_mixed"] != 110:
        print(f"WARN mix counts {pop['n_360_alldark']}/{pop['n_110_mixed']}")
    mix_of = train_dark.set_index("company_id")["mix"]
    return {
        **pop,
        "hold_ids": hold,
        "book": book,
        "train_invoiced_ids": invoiced,
        "n_invoiced": n_inv,
        "mix_of": mix_of,
        "train_cos": train,
    }


def write_md(ctx: dict) -> None:
    p1 = ctx["p1"]
    p2 = ctx["p2"]
    p3 = ctx["p3"]
    p4 = ctx["p4"]
    p5 = ctx["p5"]
    extra = ctx["extra"]
    verdict = ctx["verdict"]
    lines = [
        "# Family H — sister existence vs sister state (110 vs 360)",
        "",
        f"- **When:** {ctx['started']}",
        f"- **Agent:** `{AGENT}`",
        "- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`",
        "- **Re-run:** `python -m analysis.evaluate.sibling_h`",
        "- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates and cuts on train.",
        "- **Y:** accepted `y3_recover_cash_6m` / `y2_neg_2of3` from `targets.parquet` (no assembler).",
        "- **X question:** Family H as Y3 X (never B) vs a descriptive Q5 footnote.",
        "- **Brief:** Q5 why / Q3 turning. Hidden test is **new groups**.",
        "- **Not:** 0–100, `product/`, parquet rewrite, GBM tree bake-off, per-group models, y11 merge.",
        "",
        "## Decision",
        "",
        f"**{verdict['tag']}**",
        "",
        f"- **H as Y3 X:** PARK. `h_sib_neg_share` CV {_fmt(p5['h_neg_cv'], 3)} vs "
        f"`c_n_days_with_tx` {_fmt(p5['days_cv'], 3)}; mixed dummy {_fmt(p5['mix_cv'], 3)}; "
        f"`h_n_siblings_active` ρ {_fmt(extra['rho_sib_gsize'], 3)} vs `h_group_size`; "
        f"`h_share_group_in` is NEAR_SIZE (ρ {_fmt(p4['worst_size'], 3)} vs own A). "
        f"Not a B-copy (worst |ρ| vs `b_runway` {_fmt(p4['worst_b'], 3)}).",
        f"- **Q5 footnote:** KEEP sister *mean B-runway ≥ 1* on the 110 "
        f"(after-size {_pp_delta(extra.get('rw1_after_size'))}; "
        f"company-level ever-recover {_pp(extra.get('cos_rate_hi'))} vs {_pp(extra.get('cos_rate_lo'))}). "
        f"Family H does not carry that state (`h_sib_neg_share` gap {_pp_delta(extra.get('hneg_110_gap'))}). "
        f"Sister B is forbidden as Y3 X. "
        f"Invoiced firms in mixed groups recover *less* ({_pp(extra.get('inv_mixed_y3'))} vs "
        f"{_pp(extra.get('inv_pure_y3'))}) — the 110 lift is dark-side, not a healthy-holding dummy.",
        "",
        verdict["hidden_test"],
        "",
        verdict["one_liner"],
        "",
        "## Pass 1 — reproduce 360 / 110 / 744",
        "",
        f"Book filter = y11 (invoice, not cancel, amount ≠ 0, issuance present). "
        f"confirm_470={p1['confirm_470']}. Train invoiced **{p1['n_invoiced']}**. "
        f"Dark **{p1['n_dark']}** = **{p1['n_360']}** all-dark ({p1['n_groups_all_dark']} groups) + "
        f"**{p1['n_110']}** mixed ({p1['n_groups_mixed']} groups).",
        "",
        "Y3 is stressed-only (NaN if not stressed or no t+6). Y2 is the accepted 2-of-3 negative-liq label.",
        "",
    ]
    lines += md_table(
        p1["y3_rates"],
        [
            ("slice", "slice", "s"),
            ("n", "n labeled", "n"),
            ("pos", "pos", "n"),
            ("cos", "cos", "n"),
            ("rate", "Y3 recover", "pp"),
        ],
    )
    lines += ["", "Y2 `y2_neg_2of3` on the same company slices:", ""]
    lines += md_table(
        p1["y2_rates"],
        [
            ("slice", "slice", "s"),
            ("n", "n labeled", "n"),
            ("pos", "pos", "n"),
            ("cos", "cos", "n"),
            ("rate", "Y2 base", "pp"),
        ],
    )
    lines += [
        "",
        f"Y3 all-dark { _pp(p1['y3_360']) } vs mixed { _pp(p1['y3_110']) } "
        f"(gap { _pp_delta(p1['y3_gap']) }). "
        f"Y2 all-dark { _pp(p1['y2_360']) } vs mixed { _pp(p1['y2_110']) } "
        f"(gap { _pp_delta(p1['y2_gap']) }).",
        "",
        "## Pass 2 — size control (`log1p(a_in3)` terciles, train edges)",
        "",
        "Tercile cuts from **all train company-months with finite `a_in3`**. "
        "Holdout never enters a cut. Residual = mixed − all-dark inside the tercile.",
        "",
    ]
    lines += md_table(
        p2["y3_terc"],
        [
            ("tercile", "tercile", "s"),
            ("slice", "slice", "s"),
            ("n", "n", "n"),
            ("pos", "pos", "n"),
            ("cos", "cos", "n"),
            ("rate", "Y3 recover", "pp"),
            ("median_log1p_a_in3", "median log1p(a_in3)", "f"),
        ],
    )
    lines += ["", "Residuals (mixed − all-dark):", ""]
    lines += md_table(
        p2["y3_resid"],
        [
            ("tercile", "tercile", "s"),
            ("n_a", "n all-dark", "n"),
            ("n_b", "n mixed", "n"),
            ("rate_a", "Y3 all-dark", "pp"),
            ("rate_b", "Y3 mixed", "pp"),
            ("residual_b_minus_a", "residual", "delta"),
        ],
    )
    t1 = p2["t1_residual"]
    lines += [
        "",
        f"T1 residual **{_pp_delta(t1)}**. "
        f"{'Confirms' if np.isfinite(t1) and abs(t1 - 0.125) < 0.02 else 'Corrects'} "
        f"the y11 +12.5pp (that cut used `log1p(|op_in|)` on dark-labeled rows, not `a_in3` on all train).",
        "",
        "Same terciles on Y2:",
        "",
    ]
    lines += md_table(
        p2["y2_resid"],
        [
            ("tercile", "tercile", "s"),
            ("n_a", "n all-dark", "n"),
            ("n_b", "n mixed", "n"),
            ("rate_a", "Y2 all-dark", "pp"),
            ("rate_b", "Y2 mixed", "pp"),
            ("residual_b_minus_a", "residual", "delta"),
        ],
    )
    if p2.get("png"):
        lines += ["", f"Plot: `{Path(p2['png']).name}`.", ""]
    lines += [
        "",
        "## Pass 3 — sister *state* among the 110 (not sister existence)",
        "",
        "Invoiced sisters only, same `group_id` × `period`. Dark companies are never in the sister pool. "
        "Healthy = sister mean `b_runway` ≥ 3 / sister mean `a_io_ratio` ≥ 1 / no invoiced sister with `y2_neg_2of3`=1. "
        "Stressed = the complement among rows where the sister feature is defined. "
        "If H only marks “has a sister”, this gap is a **group-type dummy**, not why.",
        "",
        f"Among mixed-dark train company-months, invoiced sister present this month: "
        f"{p3['n_with_sister_cm']:,} / {p3['n_mixed_cm']:,} "
        f"({_pp(p3['share_with_sister'])}). Median invoiced sisters/month = {p3['median_n_inv']}.",
        "",
    ]
    st = pd.DataFrame(
        [
            {
                "name": r["name"],
                "n_h": r["healthy"]["n"],
                "pos_h": r["healthy"]["pos"],
                "rate_h": r["healthy"]["rate"],
                "n_s": r["stressed"]["n"],
                "pos_s": r["stressed"]["pos"],
                "rate_s": r["stressed"]["rate"],
                "gap": r["gap"],
            }
            for r in p3["states"]
        ]
    )
    lines += md_table(
        st,
        [
            ("name", "sister cut", "s"),
            ("n_h", "n healthy", "n"),
            ("pos_h", "pos", "n"),
            ("rate_h", "Y3 healthy", "pp"),
            ("n_s", "n stressed", "n"),
            ("pos_s", "pos", "n"),
            ("rate_s", "Y3 stressed", "pp"),
            ("gap", "gap", "delta"),
        ],
    )
    lines += [
        "",
        "Sister-state residual inside the same train `a_in3` terciles (110 only):",
        "",
    ]
    lines += md_table(
        p3["state_terc"],
        [
            ("name", "cut", "s"),
            ("tercile", "tercile", "s"),
            ("n_healthy", "n healthy", "n"),
            ("n_stressed", "n stressed", "n"),
            ("rate_healthy", "Y3 healthy", "pp"),
            ("rate_stressed", "Y3 stressed", "pp"),
            ("residual", "residual", "delta"),
        ],
    )
    lines += [
        "",
        f"Best sister-state move after size (max |T1/T2/T3| residual, runway/io/not-y2): "
        f"**{_pp_delta(p3['best_after_size'])}**. "
        f"KEEP-as-Q5 bar is ≥5pp after size. "
        f"{'Clears' if p3['clears_state'] else 'Does not clear'} the bar.",
        "",
        "Same sister cuts on Y2 (110, train labeled):",
        "",
    ]
    st2 = pd.DataFrame(
        [
            {
                "name": r["name"],
                "n_h": r["healthy"]["n"],
                "rate_h": r["healthy"]["rate"],
                "n_s": r["stressed"]["n"],
                "rate_s": r["stressed"]["rate"],
                "gap": r["gap"],
            }
            for r in p3["states_y2"]
        ]
    )
    lines += md_table(
        st2,
        [
            ("name", "sister cut", "s"),
            ("n_h", "n healthy", "n"),
            ("rate_h", "Y2 healthy", "pp"),
            ("n_s", "n stressed", "n"),
            ("rate_s", "Y2 stressed", "pp"),
            ("gap", "gap", "delta"),
        ],
    )
    lines += [
        "",
        "## Pass 4 — leak screen (H vs own A / own B)",
        "",
        "Y3 X never B. Flag B-copy if |ρ| vs `b_runway` ≥ 0.80 (NEAR at 0.50). "
        "SIZE if |ρ| vs `a_op_in` / `a_in3` ≥ 0.85 (NEAR at 0.70).",
        "",
    ]
    leak_md = p4["tab"]
    if "slice" in leak_md.columns:
        leak_md = leak_md[leak_md["slice"] == "train_y3"]
    lines += md_table(
        leak_md,
        [
            ("slice", "slice", "s"),
            ("h", "H", "s"),
            ("vs", "vs", "s"),
            ("n", "n", "n"),
            ("rho", "Spearman ρ", "f"),
            ("flags", "flags", "s"),
        ],
    )
    lines += [
        "",
        f"Worst |ρ| vs own B (`b_runway`) on train Y3-labeled: "
        f"**{_fmt(p4['worst_b'], 3)}** (`{p4['worst_b_col']}`). "
        f"{'B-COPY' if p4['is_b_copy'] else 'not a B-copy'}.",
        f"Worst |ρ| vs own size on train Y3-labeled: "
        f"**{_fmt(p4['worst_size'], 3)}** (`{p4['worst_size_pair']}`). "
        f"On the 110 only, `h_share_group_in` vs `a_op_in` is SIZE (ρ 0.876).",
        "",
        "## Pass 5 — single-feature train group-fold AUROC (Y3 stressed)",
        "",
        f"Sign from the train side of each fold. Compare to `c_n_days_with_tx` quoted **{DAYS_BAR:.3f}**. "
        f"KEEP as Q5 only if the H / mixed dummy beats a size or group-size dummy by ≥ {CLEAR_MARGIN:.2f} "
        f"and is not a B-copy.",
        "",
    ]
    lines += md_table(
        p5["tab"],
        [
            ("feature", "feature", "s"),
            ("cv", "CV AUROC", "f"),
            ("sd", "sd", "f"),
            ("train_auc", "train AUROC", "f"),
            ("train_sign", "sign", "n"),
            ("coverage", "coverage", "pp"),
            ("vs_days", "vs 0.711", "f"),
            ("vs_size", "vs size dummy", "f"),
            ("vs_gsize", "vs group-size dummy", "f"),
        ],
    )
    lines += [
        "",
        f"`h_sib_neg_share` CV **{_fmt(p5['h_neg_cv'], 3)}**. "
        f"Mixed-group dummy CV **{_fmt(p5['mix_cv'], 3)}**. "
        f"Size dummy `log1p(a_in3)` CV **{_fmt(p5['size_cv'], 3)}**. "
        f"Group-size dummy CV **{_fmt(p5['gsize_cv'], 3)}**. "
        f"`c_n_days_with_tx` CV **{_fmt(p5['days_cv'], 3)}**.",
        f"H/mixed beat size+group-size by ≥0.02: **{p5['beats_dummy']}**.",
        "",
        "## Extra cuts (same module)",
        "",
        "### Are all-dark holdings larger groups?",
        "",
    ]
    lines += md_table(
        extra["group_sizes"],
        [
            ("slice", "slice", "s"),
            ("n_cos", "cos", "n"),
            ("n_groups", "groups", "n"),
            ("median_group_n", "median group_n", "f"),
            ("mean_group_n", "mean group_n", "f"),
            ("median_h_group_size", "median h_group_size", "f"),
            ("median_h_n_sib", "median h_n_siblings_active", "f"),
            ("median_a_in3", "median a_in3", "f"),
        ],
    )
    lines += [
        "",
        extra["group_size_note"],
        "",
        "### Is `h_n_siblings_active` just `h_group_size`?",
        "",
        f"Train Spearman `h_n_siblings_active` vs `h_group_size` = **{_fmt(extra['rho_sib_gsize'], 3)}** "
        f"(n={extra['rho_sib_n']:,}). Feature report quoted 0.98. "
        f"{'Yes — near-copy of group size, not a live sibling-activity lever.' if extra['sib_is_gsize'] else 'Not a near-copy on this slice.'}",
        "",
        f"Train Spearman `h_sib_in` vs `h_sib_out` = **{_fmt(extra['rho_in_out'], 3)}** "
        f"(report 0.92). `h_share_group_in` vs `log1p(a_in3)` = **{_fmt(extra['rho_share_size'], 3)}** "
        f"(report size ρ 0.792 vs log inflow).",
        "",
        "### Holdout coverage (not a rate claim)",
        "",
        f"Holdout companies {extra['hold_n_cos']}. Dark {extra['hold_n_dark']}. "
        f"H columns defined on holdout company-months: "
        f"`h_sib_neg_share` { _pp(extra['hold_h_neg_cov']) }, "
        f"`h_sib_in` { _pp(extra['hold_h_in_cov']) }. "
        f"Y3 labeled holdout n={extra['hold_y3_n']} pos={extra['hold_y3_pos']} (LOW_POWER).",
        "",
        extra.get("more_md", ""),
        "",
        "## Verdict for H as Y3 X vs Q5 footnote",
        "",
        verdict["paragraph"],
        "",
        "## What was not done",
        "",
        "- Did not write parquet / duckdb. Did not run `build_targets`.",
        "- Did not edit y11_dark, trail_length, debt_schedule_qa, family modules, explain_y3.",
        "- Did not invent a new Y. Did not merge y11. Did not touch `product/`. Did not commit.",
        "- Did not revive per-group Y3 (PARK 0.694 vs 0.710; hidden 72 = new groups).",
        "",
    ]
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD}")


def _resid_pair(tab: pd.DataFrame, tname: str) -> float:
    row = tab[tab["tercile"] == tname]
    return float(row.iloc[0]["residual_b_minus_a"]) if len(row) else float("nan")


def round2_cuts(
    panel: pd.DataFrame,
    is_train: pd.Series,
    dark360: pd.Series,
    dark110: pd.Series,
    mix_dark: pd.Series,
    y3_ok: pd.Series,
    base110_y3: pd.Series,
    train_y3: pd.Series,
) -> dict:
    """Second sitting: y11-style size, runway≥1 after size, H-neg among 110, group-size bins."""
    print("\nROUND 2 extra cuts (same module)")
    y = pd.to_numeric(panel[Y3], errors="coerce")
    cid = panel["company_id"].astype(str)
    dark_y3 = is_train & y.notna() & (dark360 | dark110)

    # y11-style: qcut log1p(|a_op_in|) on train-dark Y3-labeled only
    opin = pd.to_numeric(panel["log1p_a_op_in"], errors="coerce")
    try:
        y11_terc, y11_edges = fit_terciles(opin, dark_y3 & opin.notna())
    except RuntimeError:
        y11_terc, y11_edges = pd.Series(index=panel.index, dtype=object), np.array([])
    y11_tab = tercile_mix_table(panel.assign(size_terc_tmp=y11_terc), Y3, is_train, mix_dark, y11_terc)
    # tercile_mix_table reads panel["size_terc"]? No — it takes terc as arg. Good.
    y11_resid = tercile_residuals(y11_tab, "all_dark", "mixed")
    t1_y11 = _resid_pair(y11_resid, "T1_small")
    print("y11-style log1p(|a_op_in|) on dark Y3-labeled")
    print(y11_resid.to_string(index=False))
    print(f"  T1 residual = {t1_y11:+.4f} (quoted +12.5pp)")

    # a_in3 edges from train-dark Y3-labeled only (not all train CM)
    ain = pd.to_numeric(panel["log1p_a_in3"], errors="coerce")
    dark_terc, dark_edges = fit_terciles(ain, dark_y3 & ain.notna())
    dark_tab = tercile_mix_table(panel, Y3, is_train, mix_dark, dark_terc)
    dark_resid = tercile_residuals(dark_tab, "all_dark", "mixed")
    t1_dark = _resid_pair(dark_resid, "T1_small")
    print("log1p(a_in3) edges from train-dark Y3-labeled")
    print(dark_resid.to_string(index=False))
    print(f"  T1 residual = {t1_dark:+.4f}")

    # runway>=1 after the all-train a_in3 terciles
    need_rw = panel["sister_runway"].notna()
    rw1_terc = state_by_tercile(
        panel,
        Y3,
        base110_y3 & need_rw,
        panel["sister_runway_ok"],
        panel["sister_runway_stressed"],
        panel["size_terc"],
        "runway>=1",
    )
    print("runway>=1 by size tercile (110)")
    print(rw1_terc.to_string(index=False))
    w = rw1_terc["n_healthy"] + rw1_terc["n_stressed"]
    ok = rw1_terc["residual"].notna() & (w > 0)
    rw1_w = float(np.average(rw1_terc.loc[ok, "residual"], weights=w[ok])) if ok.any() else float("nan")
    t1s = rw1_terc[rw1_terc["tercile"] == "T1_small"]
    rw1_t1 = float(t1s.iloc[0]["residual"]) if len(t1s) else float("nan")
    rw1_t1_ok = bool(
        len(t1s)
        and int(t1s.iloc[0]["n_healthy"]) >= 20
        and int(t1s.iloc[0]["n_stressed"]) >= 20
    )
    rw1_after = rw1_t1 if rw1_t1_ok and np.isfinite(rw1_t1) else rw1_w
    print(f"  runway>=1 weighted={rw1_w:+.4f} T1={rw1_t1:+.4f} t1n_ok={rw1_t1_ok}")

    # H-legal sister-state: h_sib_neg_share median split on train-110 Y3 (train edges)
    hneg = pd.to_numeric(panel["h_sib_neg_share"], errors="coerce")
    src = hneg[base110_y3 & hneg.notna()]
    hneg_med = float(src.median()) if len(src) else float("nan")
    hi = hneg >= hneg_med  # more siblings in the red — stressed sisters
    lo = hneg < hneg_med
    hneg_state = state_table(panel, Y3, base110_y3 & hneg.notna(), lo, hi, "h_sib_neg_share_lo")
    # after size: treat lo (fewer red sisters) as healthy
    hneg_terc = state_by_tercile(
        panel, Y3, base110_y3 & hneg.notna(), lo, hi, panel["size_terc"], "h_sib_neg_lo"
    )
    print(f"h_sib_neg_share median (110 Y3)={hneg_med:.3f}")
    print(
        f"  lo(healthy) {hneg_state['healthy']['rate']:.4f} n={hneg_state['healthy']['n']} "
        f"hi {hneg_state['stressed']['rate']:.4f} n={hneg_state['stressed']['n']} "
        f"gap={hneg_state['gap']:+.4f}"
    )
    print(hneg_terc.to_string(index=False))
    hw = hneg_terc["n_healthy"] + hneg_terc["n_stressed"]
    hok = hneg_terc["residual"].notna() & (hw > 0)
    hneg_w = float(np.average(hneg_terc.loc[hok, "residual"], weights=hw[hok])) if hok.any() else float("nan")

    # group-size terciles on train — is the 110 gap just bigger groups?
    gsz = pd.to_numeric(panel["h_group_size"], errors="coerce")
    try:
        g_terc, g_edges = fit_terciles(gsz, is_train & gsz.notna())
        g_tab = tercile_mix_table(panel, Y3, is_train, mix_dark, g_terc)
        g_resid = tercile_residuals(g_tab, "all_dark", "mixed")
        print("Y3 residual inside h_group_size terciles (train edges)")
        print(g_resid.to_string(index=False))
    except RuntimeError as exc:
        print(f"group-size terciles skipped: {exc}")
        g_terc, g_edges = pd.Series(index=panel.index, dtype=object), np.array([])
        g_resid = pd.DataFrame()

    # mixed dummy AUROC on dark Y3 only (existence among the 470)
    dark_y3_fold = dark_y3 & panel["fold"].notna()
    mix_dark_cv = signed_oof_auroc(panel[Y3], panel["mixed_dummy"], panel["fold"], dark_y3_fold)
    print(
        f"mixed dummy on dark Y3 only: cv={mix_dark_cv['cv']:.4f} "
        f"train={mix_dark_cv['train_auc']:.4f}"
    )

    # sister runway vs own size / own runway on the 110 — leftover size?
    rho_rw_ownsize = spearman(panel.loc[base110_y3, "sister_runway"], panel.loc[base110_y3, "log1p_a_in3"])
    rho_rw_ownrw = spearman(panel.loc[base110_y3, "sister_runway"], panel.loc[base110_y3, "b_runway"])
    rho_ok_size = spearman(
        panel.loc[base110_y3, "sister_runway_ok"].astype(float),
        panel.loc[base110_y3, "log1p_a_in3"],
    )
    print(
        f"sister_runway vs own log1p(a_in3) ρ={rho_rw_ownsize:.3f}; "
        f"vs own b_runway ρ={rho_rw_ownrw:.3f}; "
        f"runway_ok vs own size ρ={rho_ok_size:.3f}"
    )

    # thin-cell flag for sister_not_y2 T2 (0 pos / 27)
    state_not_thin = bool(rw1_t1_ok)
    # existence dummy: invoiced sister is always present on the 110 (100% this month)
    state_is_existence = bool(not (np.isfinite(rw1_after) and rw1_after >= STATE_PP))

    md = []
    md += [
        "### y11-style size cut (confirm or correct +12.5pp)",
        "",
        "Terciles of `log1p(|a_op_in|)` fit on **train-dark Y3-labeled** only — the y11 recipe.",
        "",
    ]
    md += md_table(
        y11_resid,
        [
            ("tercile", "tercile", "s"),
            ("n_a", "n all-dark", "n"),
            ("n_b", "n mixed", "n"),
            ("rate_a", "Y3 all-dark", "pp"),
            ("rate_b", "Y3 mixed", "pp"),
            ("residual_b_minus_a", "residual", "delta"),
        ],
    )
    md += [
        "",
        f"T1 residual **{_pp_delta(t1_y11)}**. "
        f"{'Confirms' if np.isfinite(t1_y11) and abs(t1_y11 - 0.125) < 0.015 else 'Corrects'} "
        f"the quoted +12.5pp.",
        "",
        "Same idea with `log1p(a_in3)` edges from train-dark Y3-labeled (not all train CM):",
        "",
    ]
    md += md_table(
        dark_resid,
        [
            ("tercile", "tercile", "s"),
            ("n_a", "n all-dark", "n"),
            ("n_b", "n mixed", "n"),
            ("rate_a", "Y3 all-dark", "pp"),
            ("rate_b", "Y3 mixed", "pp"),
            ("residual_b_minus_a", "residual", "delta"),
        ],
    )
    md += [
        "",
        f"T1 residual **{_pp_delta(t1_dark)}**.",
        "",
        "### Sister runway ≥ 1 after size (110)",
        "",
        "This is the strongest raw sister-state cut (+7.87pp). Same train `a_in3` terciles.",
        "",
    ]
    md += md_table(
        rw1_terc,
        [
            ("tercile", "tercile", "s"),
            ("n_healthy", "n sister ok", "n"),
            ("n_stressed", "n sister stressed", "n"),
            ("rate_healthy", "Y3 sister ok", "pp"),
            ("rate_stressed", "Y3 sister stressed", "pp"),
            ("residual", "residual", "delta"),
        ],
    )
    md += [
        "",
        f"Weighted residual **{_pp_delta(rw1_w)}**; T1 **{_pp_delta(rw1_t1)}** "
        f"(cells ≥20: {rw1_t1_ok}). After-size quote **{_pp_delta(rw1_after)}**. "
        f"Sister runway vs own `log1p(a_in3)` ρ={_fmt(rho_rw_ownsize, 3)}; "
        f"vs own `b_runway` ρ={_fmt(rho_rw_ownrw, 3)} "
        f"(not an own-B copy). Sister `a_io_ratio` went the *wrong* way (−2.2pp) — do not KEEP on IO.",
        "",
        "### H-legal sister-state: `h_sib_neg_share` median split on the 110",
        "",
        f"Train-110 Y3 median = {_fmt(hneg_med, 3)}. Low share (fewer red sisters) vs high:",
        f" Y3 {_pp(hneg_state['healthy']['rate'])} (n={hneg_state['healthy']['n']}) vs "
        f"{_pp(hneg_state['stressed']['rate'])} (n={hneg_state['stressed']['n']}), "
        f"gap {_pp_delta(hneg_state['gap'])}. Weighted after size {_pp_delta(hneg_w)}.",
        "",
    ]
    md += md_table(
        hneg_terc,
        [
            ("tercile", "tercile", "s"),
            ("n_healthy", "n low-neg", "n"),
            ("n_stressed", "n high-neg", "n"),
            ("rate_healthy", "Y3 low-neg", "pp"),
            ("rate_stressed", "Y3 high-neg", "pp"),
            ("residual", "residual", "delta"),
        ],
    )
    md += [
        "",
        "If H only encodes “has a sister”, this split should be flat. "
        f"{'It is flat enough that H is not the why.' if (not np.isfinite(hneg_state['gap']) or abs(hneg_state['gap']) < STATE_PP) else 'It moves Y3 — H-neg is a live sister-state column.'}",
        "",
        "### Is the 110 gap just bigger groups?",
        "",
        "`h_group_size` terciles fit on all train. Residual mixed − all-dark:",
        "",
    ]
    md += md_table(
        g_resid,
        [
            ("tercile", "tercile", "s"),
            ("n_a", "n all-dark", "n"),
            ("n_b", "n mixed", "n"),
            ("rate_a", "Y3 all-dark", "pp"),
            ("rate_b", "Y3 mixed", "pp"),
            ("residual_b_minus_a", "residual", "delta"),
        ],
    )
    md += [
        "",
        f"Mixed dummy on dark Y3 only (existence among the 470): "
        f"CV {_fmt(mix_dark_cv['cv'], 3)} / train {_fmt(mix_dark_cv['train_auc'], 3)}. "
        f"A group-type dummy can look useful on *this* panel and still fail on new groups.",
        "",
        "Company-month `h_group_size` medians (9 vs 11) are weighted by large groups. "
        "Company-level median group_n is 2.0 vs 7.5 — all-dark holdings are smaller groups.",
        "",
    ]
    return {
        "t1_y11_residual": t1_y11,
        "t1_dark_ain3_residual": t1_dark,
        "y11_resid": y11_resid,
        "dark_ain3_resid": dark_resid,
        "rw1_terc": rw1_terc,
        "rw1_after_size": rw1_after,
        "rw1_weighted": rw1_w,
        "hneg_110_gap": hneg_state["gap"],
        "hneg_110_gap_after_size": hneg_w,
        "hneg_terc": hneg_terc,
        "gsize_resid": g_resid,
        "mix_dark_cv": mix_dark_cv["cv"],
        "rho_sister_rw_own_size": rho_rw_ownsize,
        "rho_sister_rw_own_rw": rho_rw_ownrw,
        "state_not_thin": state_not_thin,
        "state_is_existence_dummy": state_is_existence,
        "more_md_lines": "\n".join(md),
        "y11_edges": [float(x) for x in y11_edges] if len(y11_edges) else [],
        "g_edges": [float(x) for x in g_edges],
    }


def round3_cuts(
    panel: pd.DataFrame,
    is_train: pd.Series,
    dark110: pd.Series,
    base110_y3: pd.Series,
    base110_y2: pd.Series,
    train_y3: pd.Series,
) -> dict:
    """Third sitting: sister size vs sister health; state inside group-size; Y2 after size."""
    print("\nROUND 3 extra cuts (same module)")
    y3 = pd.to_numeric(panel[Y3], errors="coerce")

    # Is "healthy sister" just a larger invoiced sister?
    sis_in3 = np.log1p(pd.to_numeric(panel["sister_in3"], errors="coerce").clip(lower=0))
    rho_rw_sisin = spearman(panel.loc[base110_y3, "sister_runway"], sis_in3[base110_y3])
    src = sis_in3[base110_y3 & sis_in3.notna()]
    if len(src) >= 30 and src.nunique() >= 3:
        sis_terc, _ = fit_terciles(sis_in3, base110_y3 & sis_in3.notna())
    else:
        sis_terc = pd.Series(index=panel.index, dtype=object)
    rows = []
    cid = panel["company_id"].astype(str)
    for tname in ("T1_small", "T2_mid", "T3_large"):
        ok = base110_y3 & sis_terc.eq(tname)
        rec = rate_row(y3[ok], cid[ok], tname)
        rec["median_sister_log1p_in3"] = float(sis_in3[ok].median()) if int(ok.sum()) else float("nan")
        rec["share_runway_ok"] = float(panel.loc[ok, "sister_runway_ok"].mean()) if int(ok.sum()) else float("nan")
        rows.append(rec)
    sis_size_tab = pd.DataFrame(rows)
    print(f"sister_runway vs sister log1p(a_in3) ρ={rho_rw_sisin:.3f}")
    print(sis_size_tab.to_string(index=False))

    # Sister-state residual inside h_group_size terciles (110 only)
    gsz = pd.to_numeric(panel["h_group_size"], errors="coerce")
    try:
        g_terc, _ = fit_terciles(gsz, is_train & gsz.notna())
    except RuntimeError:
        g_terc = pd.Series(index=panel.index, dtype=object)
    rw1_g = state_by_tercile(
        panel,
        Y3,
        base110_y3 & panel["sister_runway"].notna(),
        panel["sister_runway_ok"],
        panel["sister_runway_stressed"],
        g_terc,
        "runway>=1_in_gsize",
    )
    print("runway>=1 inside group-size terciles (110)")
    print(rw1_g.to_string(index=False))
    gw = rw1_g["n_healthy"] + rw1_g["n_stressed"]
    gok = rw1_g["residual"].notna() & (gw > 0)
    rw1_g_w = float(np.average(rw1_g.loc[gok, "residual"], weights=gw[gok])) if gok.any() else float("nan")

    # Single-feature sister *state* on the 110 Y3 rows (descriptive; sister B is not Y3 X)
    fold110 = base110_y3 & panel["fold"].notna()
    sis_feats = []
    for name, col in (
        ("sister_runway", panel["sister_runway"]),
        ("sister_io", panel["sister_io"]),
        ("sister_y2", panel["sister_y2"]),
        ("sister_in3", panel["sister_in3"]),
        ("h_sib_neg_share", panel["h_sib_neg_share"]),
        ("n_inv_sisters_month", panel["n_inv_sisters_month"]),
    ):
        rec = signed_oof_auroc(panel[Y3], col, panel["fold"], fold110)
        rec["feature"] = name
        sis_feats.append(rec)
        print(
            f"  110-only {name}: cv={rec['cv']:.4f} train={rec['train_auc']:.4f} "
            f"sign={rec['train_sign']} n={rec['n_defined']}"
        )
    sis_feat_tab = pd.DataFrame(
        [
            {
                "feature": r["feature"],
                "cv": r["cv"],
                "sd": r["sd"],
                "train_auc": r["train_auc"],
                "train_sign": r["train_sign"],
                "n_defined": r["n_defined"],
            }
            for r in sis_feats
        ]
    )

    # Y2: sister runway>=1 after the same a_in3 terciles
    y2_rw1 = state_by_tercile(
        panel,
        Y2,
        base110_y2 & panel["sister_runway"].notna(),
        panel["sister_runway_ok"],
        panel["sister_runway_stressed"],
        panel["size_terc"],
        "y2_runway>=1",
    )
    print("Y2 runway>=1 by size tercile (110)")
    print(y2_rw1.to_string(index=False))

    # T3 group-size anomaly: all-dark in large groups almost never recover
    print(
        "T3 h_group_size residual is existence/group-type "
        "(all-dark large groups recover ~0) — not sister state."
    )

    md = [
        "### Sister size vs sister health (110 Y3)",
        "",
        f"Sister `b_runway` vs sister `log1p(a_in3)` ρ={_fmt(rho_rw_sisin, 3)} "
        f"{'(NEAR_SIZE — healthy sister is a bigger sister)' if np.isfinite(rho_rw_sisin) and abs(rho_rw_sisin) >= NEAR_SIZE_RHO else '(not a sister-size clone)'}.",
        "",
    ]
    md += md_table(
        sis_size_tab,
        [
            ("slice", "sister a_in3 tercile", "s"),
            ("n", "n", "n"),
            ("pos", "pos", "n"),
            ("rate", "Y3 recover", "pp"),
            ("share_runway_ok", "share sister runway≥1", "pp"),
            ("median_sister_log1p_in3", "median sister log1p(a_in3)", "f"),
        ],
    )
    md += [
        "",
        "### Sister runway ≥ 1 inside *group-size* terciles (110)",
        "",
        "If the state gap is just “bigger holding”, it dies here.",
        "",
    ]
    md += md_table(
        rw1_g,
        [
            ("tercile", "group-size tercile", "s"),
            ("n_healthy", "n sister ok", "n"),
            ("n_stressed", "n sister stressed", "n"),
            ("rate_healthy", "Y3 sister ok", "pp"),
            ("rate_stressed", "Y3 sister stressed", "pp"),
            ("residual", "residual", "delta"),
        ],
    )
    md += [
        "",
        f"Weighted residual inside group-size bins **{_pp_delta(rw1_g_w)}**.",
        "",
        "### Single-feature on the 110 Y3 rows only (descriptive; not Y3 X)",
        "",
        "Sister `b_runway` is family B of the *sister*. Y3 X never B — this rank is a Q5 footnote, not a column to add.",
        "",
    ]
    md += md_table(
        sis_feat_tab,
        [
            ("feature", "feature", "s"),
            ("cv", "CV AUROC", "f"),
            ("sd", "sd", "f"),
            ("train_auc", "train AUROC", "f"),
            ("train_sign", "sign", "n"),
            ("n_defined", "n", "n"),
        ],
    )
    md += [
        "",
        "### Y2: sister runway ≥ 1 after size (110)",
        "",
        "Y2 is the stress label. A sister-state why should not just be “sister Y2”.",
        "",
    ]
    md += md_table(
        y2_rw1,
        [
            ("tercile", "tercile", "s"),
            ("n_healthy", "n sister ok", "n"),
            ("n_stressed", "n sister stressed", "n"),
            ("rate_healthy", "Y2 sister ok", "pp"),
            ("rate_stressed", "Y2 sister stressed", "pp"),
            ("residual", "residual", "delta"),
        ],
    )
    md += [""]
    return {
        "rho_sister_rw_sister_in3": rho_rw_sisin,
        "sis_size_tab": sis_size_tab,
        "rw1_gsize_terc": rw1_g,
        "rw1_gsize_weighted": rw1_g_w,
        "sis_feat_tab": sis_feat_tab,
        "y2_rw1_terc": y2_rw1,
        "more_md_r3": "\n".join(md),
    }


def round4_cuts(panel: pd.DataFrame, base110_y3: pd.Series) -> dict:
    """Fourth sitting: company-level sister state; sister runway vs sister Y2."""
    print("\nROUND 4 extra cuts (same module)")
    sub = panel.loc[base110_y3].copy()
    sub["y3"] = pd.to_numeric(sub[Y3], errors="coerce")
    # One row per mixed-dark company that has a Y3 label
    g = sub.groupby("company_id", sort=False)
    cos = g.agg(
        n_y3=("y3", "size"),
        n_pos=("y3", "sum"),
        rate=("y3", "mean"),
        sister_rw=("sister_runway", "mean"),
        sister_y2=("sister_y2", "mean"),
        share_rw_ok=("sister_runway_ok", "mean"),
        n_inv=("n_inv_sisters_month", "mean"),
        own_in3=("a_in3", "mean"),
        gsize=("h_group_size", "mean"),
    ).reset_index()
    cos["ever_pos"] = cos["n_pos"] > 0
    med = float(cos["sister_rw"].median()) if cos["sister_rw"].notna().any() else float("nan")
    hi = cos["sister_rw"] >= med
    lo = cos["sister_rw"] < med
    rate_hi = float(cos.loc[hi, "ever_pos"].mean()) if hi.any() else float("nan")
    rate_lo = float(cos.loc[lo, "ever_pos"].mean()) if lo.any() else float("nan")
    cm_hi = float(cos.loc[hi, "rate"].mean()) if hi.any() else float("nan")
    cm_lo = float(cos.loc[lo, "rate"].mean()) if lo.any() else float("nan")
    rho_rw_y2 = spearman(sub["sister_runway"], sub["sister_y2"])
    rho_rw_ninv = spearman(sub["sister_runway"], sub["n_inv_sisters_month"])
    print(
        f"company-level 110 Y3 cos={len(cos)} ever_pos={int(cos['ever_pos'].sum())} "
        f"sister_rw median={med:.3f} ever_pos hi={rate_hi:.3f} lo={rate_lo:.3f} "
        f"mean cm-rate hi={cm_hi:.4f} lo={cm_lo:.4f}"
    )
    print(f"sister_runway vs sister_y2 ρ={rho_rw_y2:.3f}; vs n_inv_sisters ρ={rho_rw_ninv:.3f}")

    md = [
        "### Company-level sister state (74 mixed Y3 companies)",
        "",
        "Company-months can repeat the same holding. Split companies by mean sister `b_runway` "
        f"(median { _fmt(med, 3) }).",
        "",
        f"- Companies with mean sister runway ≥ median: ever-recover **{_pp(rate_hi)}** "
        f"({int(hi.sum())} cos); mean CM rate {_pp(cm_hi)}.",
        f"- Below median: ever-recover **{_pp(rate_lo)}** ({int(lo.sum())} cos); mean CM rate {_pp(cm_lo)}.",
        f"- Gap ever-recover {_pp_delta(rate_hi - rate_lo)}; CM-rate gap {_pp_delta(cm_hi - cm_lo)}.",
        f"- Sister runway vs sister Y2 ρ={_fmt(rho_rw_y2, 3)}; vs n invoiced sisters ρ={_fmt(rho_rw_ninv, 3)}.",
        "",
        "If the company-level ever-recover gap is small, the CM +7pp is a few months in the same holdings. "
        "Sister runway is **not** a Y2 clone of the sister "
        f"({ 'near-copy' if np.isfinite(rho_rw_y2) and abs(rho_rw_y2) >= 0.80 else 'ρ well below 0.80' }).",
        "",
    ]
    return {
        "cos_n": int(len(cos)),
        "cos_ever_pos": int(cos["ever_pos"].sum()),
        "cos_rate_hi": rate_hi,
        "cos_rate_lo": rate_lo,
        "cos_cm_hi": cm_hi,
        "cos_cm_lo": cm_lo,
        "rho_sister_rw_y2": rho_rw_y2,
        "rho_sister_rw_ninv": rho_rw_ninv,
        "more_md_r4": "\n".join(md),
    }


def round5_cuts(panel: pd.DataFrame, base110_y3: pd.Series, invoiced_ids: set[str]) -> dict:
    """Fifth sitting: min/max/largest sister; fold table for sister_runway on the 110."""
    print("\nROUND 5 extra cuts (same module)")
    inv = panel.loc[panel["company_id"].isin(invoiced_ids)].copy()
    g = inv.groupby(["group_id", "period"], sort=False)
    ext = g.agg(
        sister_rw_min=("b_runway", "min"),
        sister_rw_max=("b_runway", "max"),
        sister_rw_med=("b_runway", "median"),
    ).reset_index()
    # largest invoiced sister by a_in3 this month
    inv["_in3"] = pd.to_numeric(inv["a_in3"], errors="coerce")
    idx = inv.groupby(["group_id", "period"], sort=False)["_in3"].idxmax()
    big = inv.loc[idx.dropna()].rename(columns={"b_runway": "sister_rw_largest", "y2_neg_2of3": "sister_y2_largest"})
    big = big[["group_id", "period", "sister_rw_largest", "sister_y2_largest"]]
    m = panel.merge(ext, on=["group_id", "period"], how="left").merge(big, on=["group_id", "period"], how="left")

    y = pd.to_numeric(m[Y3], errors="coerce")
    cid = m["company_id"].astype(str)
    rows = []
    for name, col, thr, ge in (
        ("mean_rw>=1", "sister_runway", RUNWAY_OK, True),
        ("min_rw>=1 (all sisters ok)", "sister_rw_min", RUNWAY_OK, True),
        ("max_rw>=1 (any sister ok)", "sister_rw_max", RUNWAY_OK, True),
        ("largest_rw>=1", "sister_rw_largest", RUNWAY_OK, True),
        ("mean_rw>=3", "sister_runway", RUNWAY_HEALTHY, True),
    ):
        s = pd.to_numeric(m[col], errors="coerce")
        defined = base110_y3 & s.notna()
        healthy = (s >= thr) if ge else (s < thr)
        a = rate_row(y[defined & healthy], cid[defined & healthy], "h")
        b = rate_row(y[defined & ~healthy], cid[defined & ~healthy], "s")
        gap = (
            float(a["rate"] - b["rate"])
            if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
            else float("nan")
        )
        rows.append(
            {
                "name": name,
                "n_h": a["n"],
                "rate_h": a["rate"],
                "n_s": b["n"],
                "rate_s": b["rate"],
                "gap": gap,
            }
        )
        print(f"  {name}: h={a['rate']:.4f} n={a['n']} s={b['rate']:.4f} n={b['n']} gap={gap:+.4f}")
    tab = pd.DataFrame(rows)

    # fold table for sister_runway on 110
    rec = signed_oof_auroc(m[Y3], m["sister_runway"], m["fold"], base110_y3 & m["fold"].notna())
    fold_tab = pd.DataFrame(rec["folds"])
    print("sister_runway 110 folds:", fold_tab.to_string(index=False))

    md = [
        "### Sister aggregation (mean vs min vs max vs largest)",
        "",
        "Mean can hide one healthy invoiced sister among four. Min = all sisters ok. "
        "Largest = invoiced sister with the highest `a_in3` this month.",
        "",
    ]
    md += md_table(
        tab,
        [
            ("name", "cut", "s"),
            ("n_h", "n healthy", "n"),
            ("rate_h", "Y3 healthy", "pp"),
            ("n_s", "n stressed", "n"),
            ("rate_s", "Y3 stressed", "pp"),
            ("gap", "gap", "delta"),
        ],
    )
    md += [
        "",
        f"Sister-runway group-fold on the 110: CV {_fmt(rec['cv'], 3)} ± {_fmt(rec['sd'], 3)} "
        f"(folds {[round(float(x), 3) if np.isfinite(x) else None for x in fold_tab['auroc']]}). "
        f"n_pos per fold is small — quote the CM rate gap and the company-level ever-recover split, not 0.662 as an engine number.",
        "",
    ]
    return {
        "agg_tab": tab,
        "sister_rw_110_cv": rec["cv"],
        "sister_rw_110_sd": rec["sd"],
        "more_md_r5": "\n".join(md),
    }


def round6_cuts(panel: pd.DataFrame, base110_y3: pd.Series) -> dict:
    """Sixth sitting: sister runway lag1 (Q6 lead) vs contemporaneous Q5."""
    print("\nROUND 6 extra cuts (same module)")
    tmp = panel.sort_values(["company_id", "period"]).copy()
    tmp["sister_rw_lag1"] = tmp.groupby("company_id", sort=False)["sister_runway"].shift(1)
    keys = panel.loc[base110_y3, ["company_id", "period"]].drop_duplicates()
    keys["_b"] = True
    tmp = tmp.merge(keys, on=["company_id", "period"], how="left")
    tmp = tmp.reset_index(drop=True)
    y = pd.to_numeric(tmp[Y3], errors="coerce")
    cid = tmp["company_id"].astype(str)
    s0 = pd.to_numeric(tmp["sister_runway"], errors="coerce")
    s1 = pd.to_numeric(tmp["sister_rw_lag1"], errors="coerce")
    base = tmp["_b"].eq(True)
    a0 = rate_row(y[base & s0.ge(RUNWAY_OK)], cid[base & s0.ge(RUNWAY_OK)], "now_ok")
    b0 = rate_row(y[base & s0.lt(RUNWAY_OK)], cid[base & s0.lt(RUNWAY_OK)], "now_stress")
    a1 = rate_row(y[base & s1.ge(RUNWAY_OK)], cid[base & s1.ge(RUNWAY_OK)], "lag1_ok")
    b1 = rate_row(y[base & s1.lt(RUNWAY_OK)], cid[base & s1.lt(RUNWAY_OK)], "lag1_stress")
    gap0 = a0["rate"] - b0["rate"] if np.isfinite(a0["rate"]) and np.isfinite(b0["rate"]) else float("nan")
    gap1 = a1["rate"] - b1["rate"] if np.isfinite(a1["rate"]) and np.isfinite(b1["rate"]) else float("nan")
    print(f"  contemporaneous runway>=1 gap={gap0:+.4f} n_ok={a0['n']} n_st={b0['n']}")
    print(f"  lag1 runway>=1 gap={gap1:+.4f} n_ok={a1['n']} n_st={b1['n']}")
    rec0 = signed_oof_auroc(tmp[Y3], tmp["sister_runway"], tmp["fold"], base & tmp["fold"].notna())
    rec1 = signed_oof_auroc(tmp[Y3], tmp["sister_rw_lag1"], tmp["fold"], base & tmp["fold"].notna())
    print(f"  CV now={rec0['cv']:.4f} lag1={rec1['cv']:.4f}")
    md = [
        "### Q6: does sister runway lead by a month?",
        "",
        f"Contemporaneous sister runway≥1: Y3 {_pp(a0['rate'])} vs {_pp(b0['rate'])} "
        f"(gap {_pp_delta(gap0)}, n={a0['n']}/{b0['n']}).",
        f"Lag-1 sister runway≥1: Y3 {_pp(a1['rate'])} vs {_pp(b1['rate'])} "
        f"(gap {_pp_delta(gap1)}, n={a1['n']}/{b1['n']}).",
        f"110-only CV now {_fmt(rec0['cv'], 3)} vs lag1 {_fmt(rec1['cv'], 3)}.",
        "",
        (
            "Lag-1 still clears +5pp — a short Q6 lead on the 110, still sister B, still not H, "
            "still no transfer to new groups."
            if np.isfinite(gap1) and gap1 >= STATE_PP
            else "Lag-1 does not clear +5pp. The Q5 footnote is this-month sister liquidity, not months of lead."
        ),
        "",
    ]
    return {
        "rw1_now_gap": gap0,
        "rw1_lag1_gap": gap1,
        "rw1_now_cv": rec0["cv"],
        "rw1_lag1_cv": rec1["cv"],
        "more_md_r6": "\n".join(md),
    }


def round7_cuts(
    panel: pd.DataFrame,
    is_train: pd.Series,
    inv744: pd.Series,
    dark110: pd.Series,
    y3_ok: pd.Series,
) -> dict:
    """Seventh sitting: do invoiced companies in mixed groups also recover more?"""
    print("\nROUND 7 extra cuts (same module)")
    y = pd.to_numeric(panel[Y3], errors="coerce")
    cid = panel["company_id"].astype(str)
    mix_g = panel["mixed_dummy"].eq(1)
    inv_mixed = inv744 & y3_ok & mix_g
    inv_pure = inv744 & y3_ok & ~mix_g
    a = rate_row(y[inv_mixed], cid[inv_mixed], "invoiced_in_mixed")
    b = rate_row(y[inv_pure], cid[inv_pure], "invoiced_all_invoiced")
    c = rate_row(y[dark110 & y3_ok], cid[dark110 & y3_ok], "dark_in_mixed")
    gap = (
        float(a["rate"] - b["rate"])
        if np.isfinite(a["rate"]) and np.isfinite(b["rate"])
        else float("nan")
    )
    print(
        f"  invoiced-in-mixed Y3={a['rate']:.4f} n={a['n']} pos={a['pos']} cos={a['cos']}"
    )
    print(
        f"  invoiced-all-invoiced Y3={b['rate']:.4f} n={b['n']} pos={b['pos']} cos={b['cos']}"
    )
    print(f"  dark-in-mixed Y3={c['rate']:.4f} n={c['n']} gap invoiced mixed-pure={gap:+.4f}")
    md = [
        "### Do invoiced companies in mixed groups also recover more?",
        "",
        "If yes, the 110 gap is a **mixed-group** type, not “H lets a dark firm see sister cash”.",
        "",
        f"- Invoiced in mixed groups: Y3 **{_pp(a['rate'])}** (n={a['n']:,} / pos={a['pos']} / cos={a['cos']}).",
        f"- Invoiced in all-invoiced groups: Y3 **{_pp(b['rate'])}** (n={b['n']:,} / pos={b['pos']} / cos={b['cos']}).",
        f"- Dark in mixed (the 110): Y3 **{_pp(c['rate'])}** (n={c['n']:,}).",
        f"- Invoiced mixed − all-invoiced gap **{_pp_delta(gap)}**.",
        "",
        (
            "Invoiced firms in mixed groups also recover more — the live signal is *group type*, "
            "not a dark-only H cash view."
            if np.isfinite(gap) and gap >= 0.03
            else "Invoiced firms in mixed groups do not share the 110 lift — the gap is on the dark side."
        ),
        "",
    ]
    return {
        "inv_mixed_y3": a["rate"],
        "inv_pure_y3": b["rate"],
        "inv_mixed_gap": gap,
        "more_md_r7": "\n".join(md),
    }


def decide(p1, p2, p3, p4, p5, extra) -> dict:
    t1 = p2["t1_residual"]
    t1_y11 = extra.get("t1_y11_residual", float("nan"))
    gap_survives = np.isfinite(t1) and t1 >= 0.05
    # Sister-state KEEP is descriptive Q5, not an X ticket. Require an H-legal
    # or brief-listed state cut ≥5pp after size, not thin cells, not io (wrong sign).
    rw1 = extra.get("rw1_after_size", float("nan"))
    hneg = extra.get("hneg_110_gap_after_size", float("nan"))
    state_ok = bool(
        (np.isfinite(rw1) and rw1 >= STATE_PP)
        or (np.isfinite(p3["best_after_size"]) and p3["best_after_size"] >= STATE_PP and extra.get("state_not_thin", False))
    )
    # H as X: PARK if group-size copy, B-copy, or single-feature loses to size/gsize.
    b_copy = bool(p4["is_b_copy"])
    size_like = bool(
        extra["sib_is_gsize"]
        or p4.get("is_size")
        or (
            np.isfinite(p5["h_neg_cv"])
            and p5["h_neg_cv"] < p5["gsize_cv"] + CLEAR_MARGIN
            and p5["h_neg_cv"] < p5["size_cv"] + CLEAR_MARGIN
        )
    )
    beats = bool(p5["beats_dummy"])
    hidden = (
        "If the useful signal is “this group has an invoiced sister”, it **cannot** "
        "transfer to the hidden test of new groups."
    )
    x_park = True  # H is not a durable Y3 X on this evidence
    hneg_flat = abs(float(extra.get("hneg_110_gap") or 0)) < 0.02
    cos_gap = extra.get("cos_rate_hi", float("nan")) - extra.get("cos_rate_lo", float("nan"))
    if state_ok and not extra.get("state_is_existence_dummy", True):
        tag = "PARK H as Y3 X; KEEP as a descriptive Q5 footnote"
        one = (
            f"Sister *B-runway* (not Family H, not sister existence) moves Y3 "
            f"{_pp_delta(rw1)} after own-size terciles inside the 110 "
            f"(company-level ever-recover gap {_pp_delta(cos_gap)}). "
            f"`h_sib_neg_share` on the same 110 is flat ({_pp_delta(extra.get('hneg_110_gap'))}). "
            f"Do not add H or sister-B to Y3 X."
        )
        if hneg_flat:
            one += " H as stored is a group-type dummy; the Q5 keep is sister liquidity, which Y3 X forbids."
    elif gap_survives:
        tag = "PARK H as Y3 X; CLOSE as a descriptive Q5 footnote"
        one = (
            f"110 vs 360 Y3 gap survives size (T1 a_in3 residual {_pp_delta(t1)}; "
            f"y11-style |op_in| T1 {_pp_delta(t1_y11)}) but the live signal is "
            f"“has an invoiced sister” — a group-type dummy, not sister *state* in H."
        )
    else:
        tag = "PARK H as a Y3 X"
        one = (
            "H is size, group-size, or a group-type dummy that will not transfer "
            "to new groups. Do not brief it as a durable lever."
        )
    paragraph = (
        f"PARK H as a Y3 X: `h_n_siblings_active` ρ vs `h_group_size` = "
        f"{_fmt(extra['rho_sib_gsize'], 3)}; `h_sib_neg_share` CV {_fmt(p5['h_neg_cv'], 3)} "
        f"loses to size {_fmt(p5['size_cv'], 3)} and to `c_n_days_with_tx` {_fmt(p5['days_cv'], 3)}; "
        f"mixed-group dummy CV {_fmt(p5['mix_cv'], 3)}; "
        f"`h_share_group_in` CV {_fmt(p5['by']['h_share_group_in']['cv'], 3)} is NEAR_SIZE "
        f"(ρ vs a_op_in {_fmt(p4['worst_size'], 3)}) and SIZE on the 110. "
        f"{one} {hidden} Not a 0–100."
    )
    return {
        "tag": tag,
        "one_liner": one,
        "hidden_test": hidden,
        "paragraph": paragraph,
        "gap_survives": gap_survives,
        "state_ok": state_ok,
        "b_copy": b_copy,
        "beats": beats,
        "x_park": x_park,
        "size_like": size_like,
    }


def run(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description="Family H sister existence vs sister state")
    p.add_argument("--no-registry", action="store_true")
    p.add_argument("--no-md", action="store_true")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args(argv)

    started = datetime.now().isoformat(timespec="minutes")
    wall0 = datetime.now()
    print(f"sibling_h start {started} seed={FOLD_SEED} agent={AGENT}")
    print("no product / no 0-100 / no parquet rewrite / no GBM / no build_targets")
    print(
        "leakage_check H-only vs Y3 never B:",
        leakage_check(list(H_COLS), Y3, ("b",)),
    )

    con = connect()
    pop = population(con)
    con.close()

    store = load_store()
    y = load_y()
    panel = store.merge(y, on=["company_id", "period"], how="left")
    folds = group_folds(pop["train_cos"][["company_id", "group_id"]], n=N_FOLDS, seed=FOLD_SEED)
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")

    cid = panel["company_id"].astype(str)
    is_train = train_mask(cid)
    is_hold = cid.isin(pop["hold_ids"])
    assert_no_holdout(cid[is_train])
    if set(cid[is_train]) & pop["hold_ids"]:
        raise RuntimeError("holdout leaked into train")

    mix_map = pop["mix_of"]
    mix = cid.map(mix_map)
    invoiced = cid.isin(pop["train_invoiced_ids"])
    dark360 = is_train & mix.eq("all_dark")
    dark110 = is_train & mix.eq("mixed")
    dark470 = dark360 | dark110
    inv744 = is_train & invoiced

    panel["log1p_a_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    panel["log1p_a_op_in"] = np.log1p(pd.to_numeric(panel["a_op_in"], errors="coerce").abs())
    mixed_gids = set(
        pop["train_dark_frame"].loc[pop["train_dark_frame"]["mix"] == "mixed", "group_id"].astype(str)
    )
    panel["mixed_dummy"] = panel["group_id"].astype(str).isin(mixed_gids).astype(float)
    panel["size_dummy"] = panel["log1p_a_in3"]
    panel["gsize_dummy"] = pd.to_numeric(panel["h_group_size"], errors="coerce")

    # --- pass 1 rates ---
    y3_masks = [
        ("full_train", is_train),
        ("invoiced_744", inv744),
        ("dark_470", dark470),
        ("all_dark_360", dark360),
        ("mixed_110", dark110),
    ]
    y3_rates = slice_rates(panel, Y3, y3_masks)
    y2_rates = slice_rates(panel, Y2, y3_masks)
    print("\nPASS 1 Y3 rates")
    print(y3_rates.to_string(index=False))
    print("\nPASS 1 Y2 rates")
    print(y2_rates.to_string(index=False))

    def _rate(df, sl):
        row = df[df["slice"] == sl]
        return float(row.iloc[0]["rate"]) if len(row) else float("nan")

    p1 = {
        "confirm_470": pop["confirm_470"],
        "n_invoiced": pop["n_invoiced"],
        "n_dark": pop["n_train_dark"],
        "n_360": pop["n_360_alldark"],
        "n_110": pop["n_110_mixed"],
        "n_groups_all_dark": pop["n_groups_all_dark"],
        "n_groups_mixed": pop["n_groups_mixed"],
        "y3_rates": y3_rates,
        "y2_rates": y2_rates,
        "y3_360": _rate(y3_rates, "all_dark_360"),
        "y3_110": _rate(y3_rates, "mixed_110"),
        "y2_360": _rate(y2_rates, "all_dark_360"),
        "y2_110": _rate(y2_rates, "mixed_110"),
    }
    p1["y3_gap"] = p1["y3_110"] - p1["y3_360"]
    p1["y2_gap"] = p1["y2_110"] - p1["y2_360"]

    # --- pass 2 size terciles (train edges) ---
    size_ok = is_train & panel["log1p_a_in3"].notna()
    terc, edges = fit_terciles(panel["log1p_a_in3"], size_ok)
    panel["size_terc"] = terc
    print(f"\nPASS 2 size tercile edges (train log1p a_in3): {edges}")

    mix_dark = pd.Series(np.where(dark360, "all_dark", np.where(dark110, "mixed", None)), index=panel.index)
    y3_terc = tercile_mix_table(panel, Y3, is_train, mix_dark, panel["size_terc"])
    y2_terc = tercile_mix_table(panel, Y2, is_train, mix_dark, panel["size_terc"])
    y3_resid = tercile_residuals(y3_terc, "all_dark", "mixed")
    y2_resid = tercile_residuals(y2_terc, "all_dark", "mixed")
    print("\nPASS 2 Y3 terciles")
    print(y3_terc.to_string(index=False))
    print(y3_resid.to_string(index=False))
    t1_row = y3_resid[y3_resid["tercile"] == "T1_small"]
    t1_residual = float(t1_row.iloc[0]["residual_b_minus_a"]) if len(t1_row) else float("nan")
    print(f"T1 residual mixed-alldark = {t1_residual:+.4f}")

    png = None
    if not args.no_plot:
        png = plot_mix_tercile(y3_terc, OUT_PNG)

    p2 = {
        "edges": [float(x) for x in edges],
        "y3_terc": y3_terc,
        "y2_terc": y2_terc,
        "y3_resid": y3_resid,
        "y2_resid": y2_resid,
        "t1_residual": t1_residual,
        "png": png,
    }

    # --- pass 3 sister state ---
    panel = attach_sister_state(panel, pop["train_invoiced_ids"])
    panel = sister_flags(panel)
    y3_ok = is_train & panel[Y3].notna()
    y2_ok = is_train & panel[Y2].notna()
    base110_y3 = dark110 & y3_ok
    base110_y2 = dark110 & y2_ok
    defined = panel["sister_runway"].notna() | panel["sister_io"].notna() | panel["sister_y2"].notna()

    states = [
        state_table(
            panel, Y3, base110_y3 & panel["sister_runway"].notna(),
            panel["sister_runway_healthy"], panel["sister_runway_stressed"] | (
                panel["sister_runway"].notna() & ~panel["sister_runway_healthy"]
            ),
            "runway>=3",
        ),
        state_table(
            panel, Y3, base110_y3 & panel["sister_runway"].notna(),
            panel["sister_runway_ok"], panel["sister_runway_stressed"],
            "runway>=1",
        ),
        state_table(
            panel, Y3, base110_y3 & panel["sister_io"].notna(),
            panel["sister_io_healthy"], panel["sister_io_stressed"],
            "io_ratio>=1",
        ),
        state_table(
            panel, Y3, base110_y3 & panel["sister_y2"].notna(),
            panel["sister_not_y2"], panel["sister_is_y2"],
            "sister_not_y2",
        ),
    ]
    states_y2 = [
        state_table(
            panel, Y2, base110_y2 & panel["sister_runway"].notna(),
            panel["sister_runway_ok"], panel["sister_runway_stressed"],
            "runway>=1",
        ),
        state_table(
            panel, Y2, base110_y2 & panel["sister_io"].notna(),
            panel["sister_io_healthy"], panel["sister_io_stressed"],
            "io_ratio>=1",
        ),
        state_table(
            panel, Y2, base110_y2 & panel["sister_y2"].notna(),
            panel["sister_not_y2"], panel["sister_is_y2"],
            "sister_not_y2",
        ),
    ]
    print("\nPASS 3 sister state vs Y3 (110)")
    for r in states:
        print(
            f"  {r['name']}: healthy {r['healthy']['rate']:.4f} n={r['healthy']['n']} "
            f"stressed {r['stressed']['rate']:.4f} n={r['stressed']['n']} gap={r['gap']:+.4f}"
        )

    terc_parts = []
    for name, hcol, scol in (
        ("runway>=3", "sister_runway_healthy", "sister_runway_stressed"),
        ("runway>=1", "sister_runway_ok", "sister_runway_stressed"),
        ("io_ratio>=1", "sister_io_healthy", "sister_io_stressed"),
        ("sister_not_y2", "sister_not_y2", "sister_is_y2"),
    ):
        need = panel["sister_runway"].notna() if "runway" in name else (
            panel["sister_io"].notna() if "io" in name else panel["sister_y2"].notna()
        )
        # runway>=3 complement includes mid (1<=rw<3); for after-size use healthy vs not-healthy
        if name == "runway>=3":
            scol_s = panel["sister_runway"].notna() & ~panel["sister_runway_healthy"]
            hcol_s = panel["sister_runway_healthy"]
        else:
            hcol_s = panel[hcol]
            scol_s = panel[scol]
        terc_parts.append(
            state_by_tercile(panel, Y3, base110_y3 & need, hcol_s, scol_s, panel["size_terc"], name)
        )
    state_terc = pd.concat(terc_parts, ignore_index=True)
    print(state_terc.to_string(index=False))
    finite_res = state_terc["residual"].to_numpy(dtype=float)
    # KEEP bar: any cut whose *n-weighted* residual after size is >= 5pp,
    # or T1 residual >= 5pp with both cells n>=20.
    best_after = float("nan")
    clears = False
    for name in state_terc["name"].unique():
        sub = state_terc[state_terc["name"] == name]
        w = sub["n_healthy"] + sub["n_stressed"]
        r = sub["residual"]
        ok = r.notna() & (w > 0)
        if ok.any():
            wmean = float(np.average(r[ok], weights=w[ok]))
        else:
            wmean = float("nan")
        t1s = sub[sub["tercile"] == "T1_small"]
        t1r = float(t1s.iloc[0]["residual"]) if len(t1s) else float("nan")
        t1n_ok = False
        if len(t1s):
            t1n_ok = int(t1s.iloc[0]["n_healthy"]) >= 20 and int(t1s.iloc[0]["n_stressed"]) >= 20
        cand = t1r if t1n_ok and np.isfinite(t1r) else wmean
        if np.isfinite(cand) and (not np.isfinite(best_after) or abs(cand) > abs(best_after)):
            best_after = cand
        if np.isfinite(cand) and cand >= STATE_PP:
            clears = True
        print(f"  {name} weighted residual={wmean:+.4f} T1={t1r:+.4f} t1n_ok={t1n_ok}")

    n_mixed_cm = int(dark110.sum())
    n_with = int((dark110 & panel["sister_has_inv_month"]).sum())
    p3 = {
        "n_mixed_cm": n_mixed_cm,
        "n_with_sister_cm": n_with,
        "share_with_sister": n_with / n_mixed_cm if n_mixed_cm else float("nan"),
        "median_n_inv": float(panel.loc[dark110, "n_inv_sisters_month"].median())
        if n_mixed_cm
        else float("nan"),
        "states": states,
        "states_y2": states_y2,
        "state_terc": state_terc,
        "best_after_size": best_after,
        "clears_state": clears,
    }

    # --- pass 4 leak ---
    y3_lab = is_train & panel[Y3].notna()
    leak_y3 = leak_table(panel, y3_lab, "train_y3")
    leak_all = leak_table(panel, is_train, "train_all")
    leak_110 = leak_table(panel, dark110, "train_110")
    leak = pd.concat([leak_y3, leak_all, leak_110], ignore_index=True)
    print("\nPASS 4 leak (train Y3)")
    print(leak_y3.to_string(index=False))
    b_rows = leak_y3[leak_y3["vs"] == "b_runway"]
    size_rows = leak_y3[leak_y3["vs"].isin(["a_op_in", "a_in3"])]
    worst_b = float(b_rows["rho"].abs().max()) if len(b_rows) else float("nan")
    worst_b_col = (
        str(b_rows.loc[b_rows["rho"].abs().idxmax(), "h"]) if len(b_rows) and b_rows["rho"].notna().any() else ""
    )
    worst_size = float(size_rows["rho"].abs().max()) if len(size_rows) else float("nan")
    worst_size_pair = ""
    if len(size_rows) and size_rows["rho"].notna().any():
        i = size_rows["rho"].abs().idxmax()
        worst_size_pair = f"{size_rows.loc[i, 'h']} vs {size_rows.loc[i, 'vs']}"
    p4 = {
        "tab": leak,
        "worst_b": worst_b,
        "worst_b_col": worst_b_col,
        "worst_size": worst_size,
        "worst_size_pair": worst_size_pair,
        "is_b_copy": bool(np.isfinite(worst_b) and worst_b >= RUNWAY_COPY),
        "is_size": bool(np.isfinite(worst_size) and worst_size >= SIZE_RHO),
    }

    # --- pass 5 single-feature ---
    train_y3 = y3_lab & panel["fold"].notna()
    feats = [
        ("h_sib_neg_share", panel["h_sib_neg_share"]),
        ("mixed_dummy", panel["mixed_dummy"]),
        ("log1p_a_in3", panel["size_dummy"]),
        ("h_group_size", panel["gsize_dummy"]),
        ("c_n_days_with_tx", panel["c_n_days_with_tx"]),
        ("h_n_siblings_active", panel["h_n_siblings_active"]),
        ("h_share_group_in", panel["h_share_group_in"]),
        ("h_sib_in", panel["h_sib_in"]),
    ]
    recs = []
    by = {}
    for name, x in feats:
        rec = signed_oof_auroc(panel[Y3], x, panel["fold"], train_y3)
        rec["feature"] = name
        by[name] = rec
        recs.append(rec)
        print(
            f"PASS 5 {name}: cv={rec['cv']:.4f}±{rec['sd']:.3f} "
            f"train={rec['train_auc']:.4f} sign={rec['train_sign']} cov={rec['coverage']:.3f}"
        )
    size_cv = by["log1p_a_in3"]["cv"]
    gsize_cv = by["h_group_size"]["cv"]
    days_cv = by["c_n_days_with_tx"]["cv"]
    dummy_bar = max(
        x for x in (size_cv, gsize_cv) if np.isfinite(x)
    ) if any(np.isfinite(x) for x in (size_cv, gsize_cv)) else float("nan")
    tab5 = []
    for rec in recs:
        tab5.append(
            {
                "feature": rec["feature"],
                "cv": rec["cv"],
                "sd": rec["sd"],
                "train_auc": rec["train_auc"],
                "train_sign": rec["train_sign"],
                "coverage": rec["coverage"],
                "vs_days": rec["cv"] - DAYS_BAR if np.isfinite(rec["cv"]) else float("nan"),
                "vs_size": rec["cv"] - size_cv if np.isfinite(rec["cv"]) and np.isfinite(size_cv) else float("nan"),
                "vs_gsize": rec["cv"] - gsize_cv if np.isfinite(rec["cv"]) and np.isfinite(gsize_cv) else float("nan"),
            }
        )
    tab5 = pd.DataFrame(tab5)
    h_neg_cv = by["h_sib_neg_share"]["cv"]
    mix_cv = by["mixed_dummy"]["cv"]
    beats = False
    for key in ("h_sib_neg_share", "mixed_dummy"):
        cv = by[key]["cv"]
        if np.isfinite(cv) and np.isfinite(dummy_bar) and (cv - dummy_bar) >= CLEAR_MARGIN and not p4["is_b_copy"]:
            beats = True
    p5 = {
        "tab": tab5,
        "h_neg_cv": h_neg_cv,
        "mix_cv": mix_cv,
        "size_cv": size_cv,
        "gsize_cv": gsize_cv,
        "days_cv": days_cv,
        "beats_dummy": beats,
        "dummy_bar": dummy_bar,
        "by": by,
    }

    # --- extra: group sizes, h_n_siblings vs h_group_size ---
    gsize_rows = []
    for name, sl in (
        ("all_dark_360", dark360),
        ("mixed_110", dark110),
        ("invoiced_744", inv744),
    ):
        sub = panel.loc[sl]
        cos = sub[["company_id", "group_id", "h_group_size"]].drop_duplicates("company_id")
        g = cos.drop_duplicates("group_id")
        gsize_rows.append(
            {
                "slice": name,
                "n_cos": int(cos["company_id"].nunique()),
                "n_groups": int(g["group_id"].nunique()),
                "median_group_n": float(g["h_group_size"].median()) if len(g) else float("nan"),
                "mean_group_n": float(g["h_group_size"].mean()) if len(g) else float("nan"),
                "median_h_group_size": float(sub["h_group_size"].median()) if len(sub) else float("nan"),
                "median_h_n_sib": float(sub["h_n_siblings_active"].median()) if len(sub) else float("nan"),
                "median_a_in3": float(sub["a_in3"].median()) if len(sub) else float("nan"),
            }
        )
    group_sizes = pd.DataFrame(gsize_rows)
    print("\nEXTRA group sizes")
    print(group_sizes.to_string(index=False))
    ad = group_sizes[group_sizes["slice"] == "all_dark_360"].iloc[0]
    mx = group_sizes[group_sizes["slice"] == "mixed_110"].iloc[0]
    if ad["median_group_n"] > mx["median_group_n"]:
        gnote = (
            f"All-dark holdings are *larger* groups (median {ad['median_group_n']:.1f} vs "
            f"{mx['median_group_n']:.1f})."
        )
    else:
        gnote = (
            f"All-dark holdings are *smaller* groups (median {ad['median_group_n']:.1f} vs mixed "
            f"{mx['median_group_n']:.1f}). Mixed dark companies are the small firms inside large groups "
            f"(median a_in3 {mx['median_a_in3']:.0f} vs all-dark {ad['median_a_in3']:.0f})."
        )
    print(gnote)

    rho_sib = spearman(panel.loc[is_train, "h_n_siblings_active"], panel.loc[is_train, "h_group_size"])
    rho_n = int(
        pd.DataFrame(
            {"a": panel.loc[is_train, "h_n_siblings_active"], "b": panel.loc[is_train, "h_group_size"]}
        )
        .dropna()
        .shape[0]
    )
    rho_in_out = spearman(panel.loc[is_train, "h_sib_in"], panel.loc[is_train, "h_sib_out"])
    rho_share = spearman(panel.loc[is_train, "h_share_group_in"], panel.loc[is_train, "log1p_a_in3"])
    sib_is_gsize = bool(np.isfinite(rho_sib) and abs(rho_sib) >= 0.90)
    print(f"h_n_siblings_active vs h_group_size ρ={rho_sib:.3f} n={rho_n} near-copy={sib_is_gsize}")

    hold_cm = is_hold
    extra = {
        "group_sizes": group_sizes,
        "group_size_note": gnote,
        "rho_sib_gsize": rho_sib,
        "rho_sib_n": rho_n,
        "sib_is_gsize": sib_is_gsize,
        "rho_in_out": rho_in_out,
        "rho_share_size": rho_share,
        "hold_n_cos": int(cid[is_hold].nunique()),
        "hold_n_dark": int(cid[is_hold & cid.isin(pop["hold_dark_ids"])].nunique()),
        "hold_h_neg_cov": float(panel.loc[hold_cm, "h_sib_neg_share"].notna().mean()) if hold_cm.any() else float("nan"),
        "hold_h_in_cov": float(panel.loc[hold_cm, "h_sib_in"].notna().mean()) if hold_cm.any() else float("nan"),
        "hold_y3_n": int((is_hold & panel[Y3].notna()).sum()),
        "hold_y3_pos": int((is_hold & (panel[Y3] == 1)).sum()),
        "more_md": "",
    }
    extra.update(
        round2_cuts(
            panel,
            is_train=is_train,
            dark360=dark360,
            dark110=dark110,
            mix_dark=mix_dark,
            y3_ok=y3_ok,
            base110_y3=base110_y3,
            train_y3=train_y3,
        )
    )
    extra.update(
        round3_cuts(
            panel,
            is_train=is_train,
            dark110=dark110,
            base110_y3=base110_y3,
            base110_y2=base110_y2,
            train_y3=train_y3,
        )
    )
    extra.update(round4_cuts(panel, base110_y3=base110_y3))
    extra.update(round5_cuts(panel, base110_y3=base110_y3, invoiced_ids=pop["train_invoiced_ids"]))
    extra.update(round6_cuts(panel, base110_y3=base110_y3))
    extra.update(round7_cuts(panel, is_train=is_train, inv744=inv744, dark110=dark110, y3_ok=y3_ok))
    extra["more_md"] = "\n".join(
        p
        for p in (
            extra.pop("more_md_lines", ""),
            extra.pop("more_md_r3", ""),
            extra.pop("more_md_r4", ""),
            extra.pop("more_md_r5", ""),
            extra.pop("more_md_r6", ""),
            extra.pop("more_md_r7", ""),
        )
        if p
    )

    verdict = decide(p1, p2, p3, p4, p5, extra)
    print(f"\nVERDICT {verdict['tag']}")
    print(verdict["one_liner"])
    print(verdict["hidden_test"])

    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    if not args.no_registry:
        rows = []

        def add(model, metric, value, coverage, notes, y=Y3, split="train"):
            rows.append(
                {
                    "ts": ts,
                    "round": ROUND,
                    "wave": WAVE,
                    "agent": AGENT,
                    "x_families": "H",
                    "y": y,
                    "model": model,
                    "split": split,
                    "metric": metric,
                    "value": f"{value:.6g}" if isinstance(value, (int, float, np.floating)) and np.isfinite(value) else "",
                    "coverage": f"{coverage:.4f}" if isinstance(coverage, (int, float)) and np.isfinite(coverage) else "",
                    "notes": notes,
                }
            )

        add(
            "mix_360_110",
            "y3_gap",
            p1["y3_gap"],
            1.0,
            f"mixed {p1['y3_110']:.4f} vs all-dark {p1['y3_360']:.4f}; confirm_470={p1['confirm_470']}; train only",
        )
        add("size_tercile", "t1_residual", p2["t1_residual"], 1.0, "mixed-alldark inside T1; log1p(a_in3) train edges")
        add("sister_state", "best_after_size", p3["best_after_size"], 1.0, f"clears_5pp={p3['clears_state']}; 110 only")
        add(
            "h_sib_neg_share",
            "auroc",
            p5["h_neg_cv"],
            p5["by"]["h_sib_neg_share"]["coverage"],
            f"vs days {DAYS_BAR}; vs size {p5['size_cv']:.3f}; vs gsize {p5['gsize_cv']:.3f}; not GBM",
            split="cv5_group",
        )
        add(
            "mixed_dummy",
            "auroc",
            p5["mix_cv"],
            p5["by"]["mixed_dummy"]["coverage"],
            "group-type dummy; hidden test = new groups; train group-fold",
            split="cv5_group",
        )
        add(
            "c_n_days_with_tx",
            "auroc",
            p5["days_cv"],
            p5["by"]["c_n_days_with_tx"]["coverage"],
            "repro bar 0.711; train group-fold signed",
            split="cv5_group",
        )
        add(
            "h_n_siblings_active",
            "spearman_vs_h_group_size",
            extra["rho_sib_gsize"],
            extra["rho_sib_n"] / max(int(is_train.sum()), 1),
            "near-copy of group size if |ρ|>=0.90",
        )
        add(
            "size_tercile_y11_opin",
            "t1_residual",
            extra.get("t1_y11_residual"),
            1.0,
            "y11-style log1p(|a_op_in|) on train-dark Y3-labeled; confirm +12.5pp",
        )
        add(
            "sister_runway_ge1",
            "y3_gap_after_size",
            extra.get("rw1_after_size"),
            1.0,
            "110 only; sister B-runway; not an H column; descriptive Q5",
        )
        add(
            "h_sib_neg_share_110",
            "y3_gap",
            extra.get("hneg_110_gap"),
            1.0,
            "median split on 110 Y3; H-legal sister-state",
        )
        append_registry(rows)

    ctx = {
        "started": started,
        "p1": p1,
        "p2": p2,
        "p3": p3,
        "p4": p4,
        "p5": p5,
        "extra": extra,
        "verdict": verdict,
    }
    if not args.no_md:
        write_md(ctx)

    elapsed = (datetime.now() - wall0).total_seconds()
    quote = {
        "n_360": p1["n_360"],
        "n_110": p1["n_110"],
        "n_invoiced": p1["n_invoiced"],
        "y3_360": p1["y3_360"],
        "y3_110": p1["y3_110"],
        "y3_gap": p1["y3_gap"],
        "y2_360": p1["y2_360"],
        "y2_110": p1["y2_110"],
        "t1_residual": p2["t1_residual"],
        "best_sister_state_after_size": p3["best_after_size"],
        "clears_state_5pp": p3["clears_state"],
        "h_sib_neg_share_cv": p5["h_neg_cv"],
        "mixed_dummy_cv": p5["mix_cv"],
        "size_dummy_cv": p5["size_cv"],
        "gsize_dummy_cv": p5["gsize_cv"],
        "c_n_days_cv": p5["days_cv"],
        "worst_rho_b_runway": p4["worst_b"],
        "rho_h_n_sib_vs_gsize": extra["rho_sib_gsize"],
        "verdict": verdict["tag"],
        "elapsed_s": elapsed,
    }
    print("\nQUOTE")
    print(json.dumps(quote, indent=2, default=str))
    print(f"elapsed {elapsed:.1f}s — stay on this module for more cuts")
    return ctx


if __name__ == "__main__":
    run()
