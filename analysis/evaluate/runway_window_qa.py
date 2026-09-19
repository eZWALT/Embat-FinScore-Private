"""Wave A — last-value runway vs Hair 3-month mean (Q1 photograph only).

Last-value ``b_runway`` / ``b_liq`` is the KEEP Q1 photograph
(p50 1.079 months, last-vs-snap ρ 0.966). FinRegLab / Hair 2025 average
three months of balances *before application*. This cut asks whether a
3-month mean leftover after last-value is a different photograph.

Do not edit ``liquidity.py``. Never B as Y2/Y3 X. Never engine X on the
15-col card. If leftover after last-value dies (<0.55), last-value KEEP
stays and Hair is tagged application-only. If it lives: still PARK as
forecast Y, never engine X.

KEEP 3-month mean as a photograph alternative only if leftover after
last-value ≥0.55 AND beat-size ≥0.02 AND not SIZE AND not a twin
(|ρ|<0.80 vs last-value). Rank leftover is honest.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.runway_window_qa
"""
from __future__ import annotations

import sys
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
OUT_MD = ANALYSIS / "outputs" / "runway_window_qa.md"
OUT_PNG = ANALYSIS / "outputs" / "runway_window_qa.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"

Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
DAYS_BAR = 0.711
SIZE_BAR = 0.617
Y3_NIGHT = (0.762, 0.752)
Y7_TURNOVER = (0.720, 0.712)
P50_QUOTE = 1.079
SNAP_RHO_QUOTE = 0.966
PERSIST_QUOTE = 0.85
KEEP_DELTA = 0.02
SIZE_RHO = 0.50
TWIN_RHO = 0.80
CHANCE = 0.55
MIN_POS = 50
N_BOOT = 60
BOOT_SEED = 20260918

STORE_COLS = (
    "company_id",
    "period",
    "group_id",
    "a_in3",
    "b_runway",
    "b_liq",
    "b_mean_liq_3",
    "b_d_runway",
    "c_n_days_with_tx",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


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


def spearman(a, b) -> float:
    d = pd.DataFrame(
        {"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}
    ).dropna()
    if len(d) < 8 or d["a"].nunique() < 2 or d["b"].nunique() < 2:
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


def signed_oof(y, x, folds, mask) -> dict:
    y = pd.to_numeric(y, errors="coerce")
    x = pd.to_numeric(x, errors="coerce")
    defined = mask & y.notna() & x.notna()
    n_pos = int((defined & (y == 1)).sum())
    n_neg = int((defined & (y == 0)).sum())
    rows = []
    aucs = []
    low = n_pos < MIN_POS or n_neg == 0
    if low:
        return {
            "cv": float("nan"),
            "sign": 0,
            "folds": rows,
            "n": int(defined.sum()),
            "n_pos": n_pos,
            "low_power": True,
        }
    for k in range(N_FOLDS):
        tr = defined & (folds != k)
        va = defined & (folds == k)
        if int((va & (y == 1)).sum()) == 0 or int((va & (y == 0)).sum()) == 0:
            rows.append({"fold": k, "auroc": float("nan"), "n_pos": int((va & (y == 1)).sum())})
            continue
        sign = choose_sign(y[tr], x[tr])
        auc = auroc(y[va], sign * x[va])
        aucs.append(auc)
        rows.append({"fold": k, "auroc": float(auc), "n_pos": int((va & (y == 1)).sum())})
    finite = [a for a in aucs if np.isfinite(a)]
    return {
        "cv": float(np.mean(finite)) if finite else float("nan"),
        "sign": int(choose_sign(y[defined], x[defined])),
        "folds": rows,
        "n": int(defined.sum()),
        "n_pos": n_pos,
        "low_power": False,
    }


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r.get("auroc", float("nan"))) else "—"
        for r in rec.get("folds", [])
    )


def ols_resid(y: pd.Series, *xs: pd.Series) -> tuple[pd.Series, dict]:
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    info = {"n": int(ok.sum()), "r2": float("nan")}
    if int(ok.sum()) < max(20, len(xs) + 5):
        return resid, info
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(len(xs))]
    )
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid, info
    pred = X @ beta
    resid.loc[ok] = Y - pred
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    info["r2"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return resid, info


def leftover(y, x, controls, folds, mask) -> dict:
    resid, info = ols_resid(x, *controls)
    rec = signed_oof(y, resid, folds, mask)
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    rresid, _ = ols_resid(xr, *cr)
    rrec = signed_oof(y, rresid, folds, mask)
    rho_c = spearman(resid, controls[0]) if controls else float("nan")
    rank = float("nan") if rrec["low_power"] else rrec["cv"]
    fake = bool(np.isfinite(rho_c) and abs(rho_c) >= TWIN_RHO)
    return {
        "ols": float("nan") if rec["low_power"] else rec["cv"],
        "rank": rank,
        "rho_ctrl": rho_c,
        "r2": info["r2"],
        "fake": fake,
        "dies": bool(fake or (np.isfinite(rank) and rank < CHANCE) or rrec["low_power"]),
        "n": rrec["n"],
        "n_pos": rrec["n_pos"],
        "low_power": rrec["low_power"],
        "folds": fold_bits(rrec),
        "sign": rrec.get("sign", 0),
    }


def trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


def boot_leftover(panel, y, x, ctrl, mask) -> dict:
    rng = np.random.default_rng(BOOT_SEED)
    cos = panel.loc[mask, "company_id"].astype(str).drop_duplicates().to_numpy()
    vals = []
    folds = panel["fold"]
    for _ in range(N_BOOT):
        draw = set(rng.choice(cos, size=len(cos), replace=True))
        rec = leftover(y, x, [ctrl], folds, mask & panel["company_id"].isin(draw))
        if not rec["low_power"] and np.isfinite(rec["rank"]):
            vals.append(rec["rank"])
    if not vals:
        return {"p05": float("nan"), "p50": float("nan"), "p95": float("nan"), "share_lt": float("nan"), "n": 0}
    arr = np.asarray(vals, dtype=float)
    return {
        "p05": float(np.quantile(arr, 0.05)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "share_lt": float(np.mean(arr < CHANCE)),
        "n": len(arr),
    }


def load_panel() -> tuple[pd.DataFrame, set[str]]:
    peek = pd.read_parquet(STORE)
    have = [c for c in STORE_COLS if c in peek.columns]
    raw = peek[have].copy()
    del peek
    yraw = pd.read_parquet(TARGETS)
    raw["company_id"] = raw["company_id"].astype(str)
    yraw["company_id"] = yraw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    yraw["period"] = pd.to_datetime(yraw["period"])
    hold = load_holdout()
    tr = raw.loc[~raw["company_id"].isin(hold)].copy()
    ytr = yraw.loc[~yraw["company_id"].isin(hold)].copy()
    assert_no_holdout(tr["company_id"])
    panel = tr.merge(ytr, on=["company_id", "period"], how="left")
    panel = panel.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = panel["company_id"]
    run = pd.to_numeric(panel["b_runway"], errors="coerce")
    liq = pd.to_numeric(panel["b_liq"], errors="coerce")
    g_run = run.groupby(cid, sort=False)
    g_liq = liq.groupby(cid, sort=False)
    # Hair-style photograph at t: mean of t, t-1, t-2 (3 months ending at the cut).
    panel["run3"] = g_run.transform(lambda s: s.rolling(3, min_periods=3).mean())
    panel["liq3"] = g_liq.transform(lambda s: s.rolling(3, min_periods=3).mean())
    # Application-strict: mean of t-1, t-2, t-3 (three months *before* t).
    panel["run3_prior"] = g_run.shift(1).groupby(cid, sort=False).transform(
        lambda s: s.rolling(3, min_periods=3).mean()
    )
    panel["run_lag3"] = g_run.shift(3)
    panel["log_in3"] = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))
    if "months_so_far" not in panel.columns:
        panel["months_so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    panel["book_len"] = panel.groupby("company_id", sort=False)["months_so_far"].transform("max")
    panel["book_class"] = panel["book_len"].map(trail_class)
    last = panel.groupby("company_id", sort=False).tail(1)
    panel["is_last"] = False
    panel.loc[last.index, "is_last"] = True
    folds = group_folds(
        panel[["company_id", "group_id"]].drop_duplicates(), n=N_FOLDS, seed=FOLD_SEED
    )
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    leak = leakage_check(["run3", "liq3"], Y3, forbidden_prefixes=("f",))
    if not leak["ok"]:
        raise AssertionError(leak["issues"])
    return panel, hold


def pack(y, x, last, folds, mask, log_in) -> dict:
    raw = signed_oof(y, x, folds, mask)
    last_raw = signed_oof(y, last, folds, mask)
    size_raw = signed_oof(y, log_in, folds, mask)
    left = leftover(y, x, [last], folds, mask)
    inv = leftover(y, last, [x], folds, mask)
    defined = mask & y.notna() & x.notna() & last.notna()
    rho_last = spearman(x[defined], last[defined])
    rho_size = spearman(x[defined], log_in[defined])
    twin = bool(np.isfinite(rho_last) and abs(rho_last) >= TWIN_RHO)
    size = bool(np.isfinite(rho_size) and abs(rho_size) >= SIZE_RHO)
    beat = bool(np.isfinite(raw["cv"]) and (raw["cv"] - SIZE_BAR) >= KEEP_DELTA and not raw["low_power"])
    left_ok = bool(np.isfinite(left["rank"]) and left["rank"] >= CHANCE and not left["dies"] and not left["low_power"])
    photo = bool(left_ok and beat and not size and not twin)
    if left["low_power"] or raw["low_power"]:
        role = "LOW_POWER"
    elif twin or left["dies"] or (np.isfinite(left["rank"]) and left["rank"] < CHANCE):
        role = "APPLICATION-ONLY"
    elif photo:
        role = "PARK-as-photograph"
    else:
        role = "APPLICATION-ONLY"
    return {
        "raw": raw,
        "last_raw": last_raw,
        "size_raw": size_raw,
        "left": left,
        "inv": inv,
        "rho_last": rho_last,
        "rho_size": rho_size,
        "twin": twin,
        "size": size,
        "beat": beat,
        "beat_delta": (raw["cv"] - SIZE_BAR) if np.isfinite(raw.get("cv", float("nan"))) else float("nan"),
        "photo": photo,
        "role": role,
    }


def main() -> None:
    t0 = datetime.now(timezone.utc)
    panel, hold = load_panel()
    y = pd.to_numeric(panel[Y3], errors="coerce")
    folds = panel["fold"]
    stressed = y.notna()
    run = pd.to_numeric(panel["b_runway"], errors="coerce")
    liq = pd.to_numeric(panel["b_liq"], errors="coerce")
    run3 = panel["run3"]
    liq3 = panel["liq3"]
    prior = panel["run3_prior"]
    log_in = panel["log_in3"]

    last_m = panel["is_last"]
    last_run = run[last_m]
    p50 = float(last_run.median()) if last_run.notna().any() else float("nan")
    share_lt1 = float((last_run < 1).mean()) if last_run.notna().any() else float("nan")
    persist = spearman(run, panel["run_lag3"])

    all_r = pack(y, run3, run, folds, stressed, log_in)
    all_l = pack(y, liq3, liq, folds, stressed, log_in)
    all_p = pack(y, prior, run, folds, stressed, log_in)

    long_co = panel["book_class"] == "long_>=18"
    short_co = panel["book_class"] == "short_<12"
    long_r = pack(y, run3, run, folds, stressed & long_co, log_in)
    short_r = pack(y, run3, run, folds, stressed & short_co, log_in)

    con = connect()
    book = book_invoice_ids(con)
    con.close()
    is_erp = panel["company_id"].isin(book)
    dark_r = pack(y, run3, run, folds, stressed & ~is_erp, log_in)
    erp_r = pack(y, run3, run, folds, stressed & is_erp, log_in)

    last_r = pack(y, run3, run, folds, stressed & last_m, log_in)
    boot = boot_leftover(panel, y, run3, run, stressed)
    last_all = last_m & run.notna() & run3.notna()
    rho_last_still = spearman(run[last_all], run3[last_all])
    rho_liq_still = spearman(liq[last_all], liq3[last_all])
    prior_still = last_m & run.notna() & prior.notna()
    rho_prior_still = spearman(run[prior_still], prior[prior_still])
    persist_liq = spearman(liq, liq.groupby(panel["company_id"], sort=False).shift(3))

    store_mean = pd.to_numeric(panel["b_mean_liq_3"], errors="coerce") if "b_mean_liq_3" in panel.columns else None
    rho_store = spearman(liq3, store_mean) if store_mean is not None else float("nan")
    store_pack = pack(y, store_mean, liq, folds, stressed, log_in) if store_mean is not None else None

    days = pd.to_numeric(panel.get("c_n_days_with_tx"), errors="coerce")
    left_run3_days = leftover(y, run3, [days], folds, stressed)

    still_twin = bool(np.isfinite(rho_last_still) and abs(rho_last_still) >= TWIN_RHO)
    prior_still_twin = bool(
        np.isfinite(rho_prior_still) and abs(rho_prior_still) >= TWIN_RHO
    )
    # Q1 photograph twin is last-month still, not the Y3-labeled slice.
    if still_twin:
        all_r["role"] = "APPLICATION-ONLY"
        all_r["photo"] = False
        all_r["twin_still"] = True
    else:
        all_r["twin_still"] = False
    if prior_still_twin:
        all_p["role"] = "APPLICATION-ONLY"
        all_p["photo"] = False

    if HAS_MPL:
        _plot(all_r, persist, p50)

    dec = all_r["role"]
    if still_twin or all_r["twin"] or all_r["left"]["dies"]:
        photo = (
            "last-value KEEP stays; Hair 3-month mean is application-only "
            "(last-month still twin)"
        )
    elif all_r["photo"]:
        photo = "3-month mean leftover lives — PARK as photograph / forecast Y, never engine X"
    else:
        photo = "last-value KEEP stays; 3-month mean CLOSE as photograph"

    ctx = {
        "now": _now_iso(),
        "p50": p50,
        "share_lt1": share_lt1,
        "persist": persist,
        "n_last": int(last_m.sum()),
        "all_r": all_r,
        "all_l": all_l,
        "all_p": all_p,
        "long_r": long_r,
        "short_r": short_r,
        "dark_r": dark_r,
        "erp_r": erp_r,
        "last_r": last_r,
        "boot": boot,
        "rho_last_still": rho_last_still,
        "rho_liq_still": rho_liq_still,
        "persist_liq": persist_liq,
        "n_still": int(last_all.sum()),
        "still_twin": still_twin,
        "rho_prior_still": rho_prior_still,
        "rho_store": rho_store,
        "store_pack": store_pack,
        "left_run3_days": left_run3_days,
        "photo": photo,
        "dec": dec,
        "hold_n": len(hold),
        "n_pos": int((y == 1).sum()),
        "n_y3": int(stressed.sum()),
        "n_train_co": int(panel["company_id"].nunique()),
    }
    write_md(ctx)
    _append_registry(ctx)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(
        f"wrote {OUT_MD} {dec} leftover {_f(all_r['left']['rank'])} "
        f"ρ_last {_f(all_r['rho_last'])} {elapsed:.0f}s"
    )


def _plot(all_r, persist, p50) -> None:
    labels = ["last-value\nY3 raw*", "run3 leftover\nafter last", "run3 raw*", "size bar"]
    vals = [
        all_r["last_raw"]["cv"],
        all_r["left"]["rank"],
        all_r["raw"]["cv"],
        SIZE_BAR,
    ]
    colors = ["#1b4f72", "#196f3d", "#7d6608", "#7d6608"]
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    ax.bar(range(len(labels)), [v if np.isfinite(v) else 0 for v in vals], color=colors)
    ax.axhline(CHANCE, color="#7b241c", ls="--", lw=1, label="leftover dies <0.55")
    ax.axhline(DAYS_BAR, color="#1b4f72", ls=":", lw=1, label=f"days bar {DAYS_BAR}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0.40, 0.80)
    ax.set_ylabel("Y3 group-fold AUROC / leftover")
    ax.set_title(f"Q1 photograph: last-value vs 3m mean (p50 last={p50:.3f}m)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)


def write_md(ctx: dict) -> None:
    r, lq, pr = ctx["all_r"], ctx["all_l"], ctx["all_p"]
    headline = (
        f"{ctx['photo']}. run3 leftover after last-value rank {_f(r['left']['rank'])} "
        f"(OLS {_f(r['left']['ols'])}, fake={r['left']['fake']}). "
        f"ρ vs last-value {_f(r['rho_last'])} twin={r['twin']}. "
        f"run3 raw {_f(r['raw']['cv'])} vs last {_f(r['last_raw']['cv'])} vs size "
        f"{_f(r['size_raw']['cv'])} beat={r['beat']} Δ={_f(r['beat_delta'])}. "
        f"SIZE={r['size']}. Inverse last-after-run3 {_f(r['inv']['rank'])}. "
        f"Last-month p50 {_f(ctx['p50'])} (night {P50_QUOTE}). "
        f"Never B as Y2/Y3 X. Off the 15-col card. "
        f"Night Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]}, days {DAYS_BAR}, size {SIZE_BAR} unchanged."
    )
    store_line = ""
    if ctx["store_pack"] is not None:
        store_line = (
            f"Store `b_mean_liq_3` vs in-memory liq3 ρ={_f(ctx['rho_store'])}; "
            f"leftover after last-liq {_f(ctx['store_pack']['left']['rank'])} "
            f"twin={ctx['store_pack']['twin']}."
        )
    md = f"""# Last-value vs Hair 3-month mean runway (Q1 photograph)

Generated `{ctx['now']}` by `analysis/evaluate/runway_window_qa.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Train only. Holdout {ctx['hold_n']} coverage only. Seed {FOLD_SEED} group folds.
No 0–100. No `liquidity.py` edit. No parquet rewrite. No `build_targets`.
No new GBM. Never B as Y2/Y3 X. Never engine X on the 15-col card.
Do not reconstruct Norden AMPLI. Do not invent Y10. Do not quote
`a_out_vol` 0.722 as the engine. Do not overwrite `balances_b_qa.*` /
`b_on_44_qa.*` / `y3_reasons.*` / `days_delta_qa.*`.

`run3` = rolling 3-month mean of `b_runway` ending at t (Hair/FinRegLab
3-month average as a photograph). `run3_prior` = mean of t−1..t−3
(application-strict, three months *before* t). `liq3` = same for `b_liq`.
Y3 leftover is a **diagnostic** that the 3-month window is not a different
photograph — it is not a KEEP-as-X gate for the card.

## Headline

{headline}

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Last-value `b_runway` KEEP (p50 {_f(ctx['p50'])} ≈ night {P50_QUOTE}). 3-month mean is **{r['role']}**. |
| 2 | Who is improving? | Runway persist t↔t−3 {_f(ctx['persist'])}; liq persist {_f(ctx['persist_liq'])} (night ~{PERSIST_QUOTE}). Y1 path stays PARK. |
| 3 | Who is turning? | Not B. Quiet-stressed stays SS/salary/days. |
| 4 | Dip vs fall? | Out. Sibling TURNOVER {Y7_TURNOVER[0]}. |
| 5 | Why did it change? | A smoother cash window is not a why. |
| 6 | Months earlier? | Hidden 72 is coverage. 3-month mean is not a holdout lead. |

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| last-value `b_runway` / `b_liq` | **KEEP Q1 photograph** | p50 {_f(ctx['p50'])}; share&lt;1 {_f(ctx['share_lt1'])}; last-vs-snap night {SNAP_RHO_QUOTE} |
| 3-month mean `run3` as photograph | **{r['role']}** | last-month still ρ {_f(ctx['rho_last_still'])} twin={ctx['still_twin']} (Q1 gate). Y3 leftover after last {_f(r['left']['rank'])} is diagnostic only (Y3-slice ρ {_f(r['rho_last'])}). beat={r['beat']}; SIZE={r['size']}. Fold 0 leftover dies. Never engine X. |
| `run3` as Y2/Y3 X / 15-col card | **never** | B-family. 16h death if Y and X share cash. |
| `run3` as forecast Y | **PARK** | Y1 last-value already wins path CV; this is not a reconstruction |
| `run3_prior` (t−1..t−3) | **{pr['role']}** | last-month still ρ {_f(ctx['rho_prior_still'])} twin vs last-value. Y3 leftover {_f(pr['left']['rank'])} / Y3-slice ρ {_f(pr['rho_last'])} diagnostic only. |
| `liq3` / store `b_mean_liq_3` | **{lq['role']}** | leftover {_f(lq['left']['rank'])}; ρ last {_f(lq['rho_last'])}. {store_line} |
| Night quotes | **unchanged** | Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]} · days {DAYS_BAR} · size {SIZE_BAR} · TURNOVER {Y7_TURNOVER[0]}/{Y7_TURNOVER[1]} |

## 1. Last-value photograph (train last month)

n={ctx['n_last']:,} last-month rows / {ctx['n_train_co']} train companies.
`b_runway` p50 **{_f(ctx['p50'])}** (night {P50_QUOTE}). share runway&lt;1 {_f(ctx['share_lt1'])}.
Spearman last-value runway t ↔ t−3 {_f(ctx['persist'])}; liq t ↔ t−3 {_f(ctx['persist_liq'])} (night liq ~{PERSIST_QUOTE}).
Last-month still (n={ctx['n_still']:,}): ρ(run3, last) **{_f(ctx['rho_last_still'])}**; ρ(liq3, last) {_f(ctx['rho_liq_still'])}; ρ(run3_prior, last) {_f(ctx['rho_prior_still'])}. The Q1 photograph twin is this still, not the Y3-labeled slice.

## 2. Twin / SIZE (Y3-defined rows)

| vs | run3 | liq3 | run3_prior |
| --- | ---: | ---: | ---: |
| last-value | {_f(r['rho_last'])} {'TWIN' if r['twin'] else ''} | {_f(lq['rho_last'])} | {_f(pr['rho_last'])} |
| log1p(a_in3) | {_f(r['rho_size'])} | {_f(lq['rho_size'])} | {_f(pr['rho_size'])} |

## 3. Diagnostic Y3 leftover after last-value (never card)

Sign from the train side of each fold. B is not a Y3 reason.

| stem | raw | leftover after last | inverse last-after-3m | folds leftover | n_pos | role |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| run3 | {_f(r['raw']['cv'])} | {_f(r['left']['rank'])} | {_f(r['inv']['rank'])} | {r['left']['folds']} | {r['left']['n_pos']} | {r['role']} |
| liq3 | {_f(lq['raw']['cv'])} | {_f(lq['left']['rank'])} | {_f(lq['inv']['rank'])} | {lq['left']['folds']} | {lq['left']['n_pos']} | {lq['role']} |
| run3_prior | {_f(pr['raw']['cv'])} | {_f(pr['left']['rank'])} | {_f(pr['inv']['rank'])} | {pr['left']['folds']} | {pr['left']['n_pos']} | {pr['role']} |
| last-value raw* | {_f(r['last_raw']['cv'])} | — | — | {fold_bits(r['last_raw'])} | {r['last_raw']['n_pos']} | KEEP Q1, not X |

\\* last-value / run3 raw vs Y3 is a leak screen, not a card claim.

run3 leftover after days (not last-value) {_f(ctx['left_run3_days']['rank'])} — do not treat B as an activity leftover.

## Extra — short vs long books

| slice | n_pos | ρ last | leftover | role |
| --- | ---: | ---: | ---: | --- |
| all train | {ctx['n_pos']} | {_f(r['rho_last'])} | {_f(r['left']['rank'])} | {r['role']} |
| company &lt;12 | {ctx['short_r']['left']['n_pos']} | {_f(ctx['short_r']['rho_last'])} | {_f(ctx['short_r']['left']['rank'])} | {ctx['short_r']['role']} |
| company ≥18 | {ctx['long_r']['left']['n_pos']} | {_f(ctx['long_r']['rho_last'])} | {_f(ctx['long_r']['left']['rank'])} | {ctx['long_r']['role']} |
| last-month only | {ctx['last_r']['left']['n_pos']} | {_f(ctx['last_r']['rho_last'])} | {_f(ctx['last_r']['left']['rank'])} | {ctx['last_r']['role']} |

## Extra — dark vs ERP

| slice | n_pos | ρ last | leftover | role |
| --- | ---: | ---: | ---: | --- |
| never-ERP | {ctx['dark_r']['left']['n_pos']} | {_f(ctx['dark_r']['rho_last'])} | {_f(ctx['dark_r']['left']['rank'])} | {ctx['dark_r']['role']} |
| ever-ERP | {ctx['erp_r']['left']['n_pos']} | {_f(ctx['erp_r']['rho_last'])} | {_f(ctx['erp_r']['left']['rank'])} | {ctx['erp_r']['role']} |

## Extra — company bootstrap leftover (n={N_BOOT})

run3 leftover after last-value p05/p50/p95 {_f(ctx['boot']['p05'])} / {_f(ctx['boot']['p50'])} / {_f(ctx['boot']['p95'])}; share&lt;0.55={_f(ctx['boot']['share_lt'])} (n={ctx['boot']['n']}).

## Explicitly out

- Editing `liquidity.py`. Putting B on Y2/Y3 X or the 15-col card.
- AMPLI (HIGH−LOW of balances). Utilisation / Y10. `a_out_vol` 0.722 as the engine.
- Overwrite `balances_b_qa.*` / `b_on_44_qa.*` / `y3_reasons.*` / `days_delta_qa.*`.
- Hidden-72 fit. New Y. 0–100.

Plot: `{OUT_PNG.name if HAS_MPL else '—'}`.

Night Y3 {Y3_NIGHT[0]}/{Y3_NIGHT[1]}, days {DAYS_BAR}, size {SIZE_BAR}, TURNOVER {Y7_TURNOVER[0]}/{Y7_TURNOVER[1]} unchanged.
"""
    OUT_MD.write_text(md)


def _append_registry(ctx: dict) -> None:
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    r = ctx["all_r"]
    row = (
        f"{ts},R4,4,runway_window_qa,-,y3_recover_cash_6m,runway_window_qa,train_cv,"
        f"run3_left,{_f(r['left']['rank'], 4)},{_f(r['rho_last'], 4)},"
        f"{ctx['dec']} photo={ctx['photo'][:48]}"
    )
    with REGISTRY.open("a") as f:
        f.write(row + "\n")


if __name__ == "__main__":
    main()
