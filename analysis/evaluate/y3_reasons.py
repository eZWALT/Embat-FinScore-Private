"""Siddiqi / CFPB-style Y3 reasons on leftover-KEEP stems only.

Read-only store. Does not rewrite gbm_core, y3_importances, shap_y3,
leftover QA files, or the 15-col historical spec. No 0–100. Never B as X.
Holdout 72 is coverage only. Seed 20260918.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.y3_reasons
"""
from __future__ import annotations

import csv
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
OUT_MD = ANALYSIS / "outputs" / "y3_reasons.md"
OUT_PNG = ANALYSIS / "outputs" / "y3_reasons.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
WAVE_NOTE = ROOT / "overnight" / "waves" / "wave4_y3_reasons.md"

Y3 = "y3_recover_cash_6m"
N_FOLDS = 5
MIN_POS = 50
DAYS_BAR = 0.711
SIZE_BAR = 0.617
SS_LEFT = 0.635
SAL_LEFT = 0.603
Y3_NIGHT = (0.762, 0.752)
DAYS_LAG1 = 0.684
SS_LAG1_LEFT = 0.631
SAL_LAG1_LEFT = 0.606
ISSUED_LAG1 = 0.626
PERM_SS = 0.034

KEEP_STEMS = (
    "c_ss_month",
    "c_salary_month",
    "c_n_days_with_tx",
    "c_ss_month_lag1",
    "c_ss_month_lag3",
    "c_salary_month_lag1",
    "c_salary_month_lag3",
    "c_n_days_with_tx_lag1",
    "c_n_days_with_tx_lag3",
)
WILL_NOT = (
    "a_n_tx",
    "a_op_in",
    "a_transfer",
    "a_transfer_lag1",
    "a_transfer_lag3",
    "e_dso_proxy",
    "f_ds_r",
    "f_ds_r_lag1",
    "c_gap_sd",
    "a_out_vol",
    "h_n_siblings_active_lag3",
)


def _f(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


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
    low = n_pos < MIN_POS or n_neg == 0
    rows = []
    aucs = []
    if low:
        return {
            "cv": float("nan"),
            "folds": rows,
            "n": int(defined.sum()),
            "n_pos": n_pos,
            "low_power": True,
            "sign": 0,
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
        "folds": rows,
        "n": int(defined.sum()),
        "n_pos": n_pos,
        "low_power": False,
        "sign": int(choose_sign(y[defined], x[defined])),
    }


def ols_resid(y: pd.Series, *xs: pd.Series) -> pd.Series:
    cols = {"y": pd.to_numeric(y, errors="coerce")}
    for i, x in enumerate(xs):
        cols[f"x{i}"] = pd.to_numeric(x, errors="coerce")
    d = pd.DataFrame(cols)
    ok = d.notna().all(axis=1)
    resid = pd.Series(np.nan, index=d.index, dtype=float)
    if int(ok.sum()) < max(20, len(xs) + 5):
        return resid
    Y = d.loc[ok, "y"].to_numpy(dtype=float)
    X = np.column_stack(
        [np.ones(int(ok.sum()))] + [d.loc[ok, f"x{i}"].to_numpy(dtype=float) for i in range(len(xs))]
    )
    try:
        beta, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
    except np.linalg.LinAlgError:
        return resid
    resid.loc[ok] = Y - (X @ beta)
    return resid


def leftover_rank(y, x, controls, folds, mask) -> dict:
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    cr = [pd.to_numeric(c, errors="coerce").rank(method="average") for c in controls]
    resid = ols_resid(xr, *cr)
    rec = signed_oof(y, resid, folds, mask)
    rho = spearman(resid, controls[0]) if controls else float("nan")
    return {**rec, "rho_ctrl": rho, "fake": bool(np.isfinite(rho) and abs(rho) >= 0.80)}


def fold_bits(rec: dict) -> str:
    return " ".join(
        f"{r['auroc']:.3f}" if np.isfinite(r.get("auroc", float("nan"))) else "—"
        for r in rec.get("folds", [])
    )


def add_lags(df: pd.DataFrame, stems: list[str], lags=(1, 3)) -> pd.DataFrame:
    out = df.sort_values(["company_id", "period"]).reset_index(drop=True)
    cid = out["company_id"]
    extra = {}
    for c in stems:
        if c not in out.columns:
            continue
        s = pd.to_numeric(out[c], errors="coerce")
        for k in lags:
            extra[f"{c}_lag{k}"] = s.groupby(cid, sort=False).shift(k)
    return pd.concat([out, pd.DataFrame(extra, index=out.index)], axis=1)


def trail_class(n) -> str:
    if n is None or (isinstance(n, float) and not np.isfinite(n)):
        return "na"
    n = int(n)
    if n < 12:
        return "short_<12"
    if n >= 18:
        return "long_>=18"
    return "mid_12_17"


def load_panel() -> tuple[pd.DataFrame, pd.Series]:
    raw = pd.read_parquet(STORE)
    yraw = pd.read_parquet(TARGETS)
    raw["company_id"] = raw["company_id"].astype(str)
    yraw["company_id"] = yraw["company_id"].astype(str)
    raw["period"] = pd.to_datetime(raw["period"])
    yraw["period"] = pd.to_datetime(yraw["period"])
    hold = load_holdout()
    tr = raw.loc[~raw["company_id"].isin(hold)].copy()
    ytr = yraw.loc[~yraw["company_id"].isin(hold)].copy()
    assert_no_holdout(tr["company_id"])
    assert_no_holdout(ytr["company_id"])
    panel = tr.merge(ytr, on=["company_id", "period"], how="left")
    stems = ["c_ss_month", "c_salary_month", "c_n_days_with_tx", "a_in3"]
    panel = add_lags(panel, stems)
    if "months_so_far" not in panel.columns:
        panel = panel.sort_values(["company_id", "period"])
        panel["months_so_far"] = panel.groupby("company_id", sort=False).cumcount() + 1
    panel["trail_class"] = panel["months_so_far"].map(trail_class)
    book_len = panel.groupby("company_id", sort=False)["months_so_far"].transform("max")
    panel["book_len"] = book_len
    panel["book_class"] = book_len.map(trail_class)
    folds = group_folds(
        panel[["company_id", "group_id"]].drop_duplicates(), n=N_FOLDS, seed=FOLD_SEED
    )
    panel = panel.merge(folds[["company_id", "fold"]], on="company_id", how="left")
    leak = leakage_check(KEEP_STEMS, Y3, forbidden_prefixes=("b",))
    if not leak["ok"]:
        raise AssertionError(leak["issues"])
    return panel, pd.Series(list(hold))


def main() -> None:
    t0 = datetime.now(timezone.utc)
    panel, hold = load_panel()
    y = pd.to_numeric(panel[Y3], errors="coerce")
    stressed = y.notna()
    days = pd.to_numeric(panel["c_n_days_with_tx"], errors="coerce")
    ss = pd.to_numeric(panel["c_ss_month"], errors="coerce").fillna(0)
    sal = pd.to_numeric(panel["c_salary_month"], errors="coerce").fillna(0)
    folds = panel["fold"]
    log_in = np.log1p(pd.to_numeric(panel["a_in3"], errors="coerce").clip(lower=0))

    days_rec = signed_oof(y, days, folds, stressed)
    ss_rec = signed_oof(y, ss, folds, stressed)
    sal_rec = signed_oof(y, sal, folds, stressed)
    size_rec = signed_oof(y, log_in, folds, stressed)
    ss_left = leftover_rank(y, ss, [days], folds, stressed)
    sal_left = leftover_rank(y, sal, [days], folds, stressed)
    days_lag1 = pd.to_numeric(panel["c_n_days_with_tx_lag1"], errors="coerce")
    ss_lag1 = pd.to_numeric(panel["c_ss_month_lag1"], errors="coerce")
    sal_lag1 = pd.to_numeric(panel["c_salary_month_lag1"], errors="coerce")
    days_l1 = signed_oof(y, days_lag1, folds, stressed)
    ss_l1_left = leftover_rank(y, ss_lag1, [days_lag1], folds, stressed)
    sal_l1_left = leftover_rank(y, sal_lag1, [days_lag1], folds, stressed)

    con = connect()
    book = book_invoice_ids(con)
    con.close()
    is_erp = panel["company_id"].isin(book)
    dark_left_ss = leftover_rank(y, ss, [days], folds, stressed & ~is_erp)
    erp_left_ss = leftover_rank(y, ss, [days], folds, stressed & is_erp)
    dark_left_sal = leftover_rank(y, sal, [days], folds, stressed & ~is_erp)
    erp_left_sal = leftover_rank(y, sal, [days], folds, stressed & is_erp)
    dark_days = signed_oof(y, days, folds, stressed & ~is_erp)
    erp_days = signed_oof(y, days, folds, stressed & is_erp)

    short = panel["trail_class"] == "short_<12"
    longb = panel["trail_class"] == "long_>=18"
    short_days = signed_oof(y, days, folds, stressed & short)
    long_days = signed_oof(y, days, folds, stressed & longb)
    short_ss = leftover_rank(y, ss, [days], folds, stressed & short)
    long_ss = leftover_rank(y, ss, [days], folds, stressed & longb)
    short_sal = leftover_rank(y, sal, [days], folds, stressed & short)
    long_sal = leftover_rank(y, sal, [days], folds, stressed & longb)
    short_l1 = signed_oof(y, days_lag1, folds, stressed & short)
    long_l1 = signed_oof(y, days_lag1, folds, stressed & longb)
    co_short = panel["book_class"] == "short_<12"
    co_long = panel["book_class"] == "long_>=18"
    cshort_days = signed_oof(y, days, folds, stressed & co_short)
    clong_days = signed_oof(y, days, folds, stressed & co_long)
    cshort_ss = leftover_rank(y, ss, [days], folds, stressed & co_short)
    clong_ss = leftover_rank(y, ss, [days], folds, stressed & co_long)
    cshort_sal = leftover_rank(y, sal, [days], folds, stressed & co_short)
    clong_sal = leftover_rank(y, sal, [days], folds, stressed & co_long)
    cshort_l1 = signed_oof(y, days_lag1, folds, stressed & co_short)
    clong_l1 = signed_oof(y, days_lag1, folds, stressed & co_long)
    co_mid = panel["book_class"] == "mid_12_17"
    cmid_days = signed_oof(y, days, folds, stressed & co_mid)
    cmid_ss = leftover_rank(y, ss, [days], folds, stressed & co_mid)
    cmid_sal = leftover_rank(y, sal, [days], folds, stressed & co_mid)
    cmid_l1 = signed_oof(y, days_lag1, folds, stressed & co_mid)
    days_lag3 = pd.to_numeric(panel["c_n_days_with_tx_lag3"], errors="coerce")
    ss_lag3 = pd.to_numeric(panel["c_ss_month_lag3"], errors="coerce")
    sal_lag3 = pd.to_numeric(panel["c_salary_month_lag3"], errors="coerce")
    days_l3 = signed_oof(y, days_lag3, folds, stressed)
    ss_l3_left = leftover_rank(y, ss_lag3, [days_lag3], folds, stressed)
    sal_l3_left = leftover_rank(y, sal_lag3, [days_lag3], folds, stressed)
    ss_after_both = leftover_rank(y, ss, [days, sal], folds, stressed)
    sal_after_both = leftover_rank(y, sal, [days, ss], folds, stressed)

    hold_cm = 0
    hold_co = int(hold.nunique()) if hasattr(hold, "nunique") else len(set(hold))
    raw_all = pd.read_parquet(STORE, columns=["company_id"])
    hold_cm = int(raw_all["company_id"].astype(str).isin(set(hold.astype(str))).sum())

    if HAS_MPL:
        labels = [
            "days bar",
            "SS leftover",
            "salary leftover",
            "size bar",
            "n_tx leftover (DROP)",
            "ds_r leftover (DROP)",
        ]
        vals = [
            days_rec["cv"],
            ss_left["cv"],
            sal_left["cv"],
            size_rec["cv"],
            0.538,
            0.528,
        ]
        colors = ["#1b4f72", "#196f3d", "#196f3d", "#7d6608", "#922b21", "#922b21"]
        fig, ax = plt.subplots(figsize=(8.2, 4.2))
        ax.bar(range(len(labels)), vals, color=colors)
        ax.axhline(0.55, color="#7b241c", ls="--", lw=1, label="leftover dies <0.55")
        ax.axhline(DAYS_BAR, color="#1b4f72", ls=":", lw=1, label=f"night days {DAYS_BAR}")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylim(0.45, 0.80)
        ax.set_ylabel("Y3 group-fold AUROC / leftover")
        ax.set_title("Y3 card reasons: leftover-KEEP vs published DROPs")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=120)
        plt.close(fig)

    now = datetime.now().astimezone().isoformat(timespec="seconds")
    md = _render(
        now=now,
        days_rec=days_rec,
        ss_rec=ss_rec,
        sal_rec=sal_rec,
        size_rec=size_rec,
        ss_left=ss_left,
        sal_left=sal_left,
        days_l1=days_l1,
        ss_l1_left=ss_l1_left,
        sal_l1_left=sal_l1_left,
        dark_left_ss=dark_left_ss,
        erp_left_ss=erp_left_ss,
        dark_left_sal=dark_left_sal,
        erp_left_sal=erp_left_sal,
        dark_days=dark_days,
        erp_days=erp_days,
        short_days=short_days,
        long_days=long_days,
        short_ss=short_ss,
        long_ss=long_ss,
        short_sal=short_sal,
        long_sal=long_sal,
        short_l1=short_l1,
        long_l1=long_l1,
        cshort_days=cshort_days,
        clong_days=clong_days,
        cshort_ss=cshort_ss,
        clong_ss=clong_ss,
        cshort_sal=cshort_sal,
        clong_sal=clong_sal,
        cshort_l1=cshort_l1,
        clong_l1=clong_l1,
        cmid_days=cmid_days,
        cmid_ss=cmid_ss,
        cmid_sal=cmid_sal,
        cmid_l1=cmid_l1,
        days_l3=days_l3,
        ss_l3_left=ss_l3_left,
        sal_l3_left=sal_l3_left,
        ss_after_both=ss_after_both,
        sal_after_both=sal_after_both,
        n_y3=int(stressed.sum()),
        n_pos=int((y == 1).sum()),
        hold_co=hold_co,
        hold_cm=hold_cm,
        png=OUT_PNG.name if HAS_MPL else None,
    )
    OUT_MD.write_text(md)
    _append_registry(
        {
            "ss_left": ss_left["cv"],
            "sal_left": sal_left["cv"],
            "days": days_rec["cv"],
            "ss_l1": ss_l1_left["cv"],
        }
    )
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    print(f"wrote {OUT_MD} leftover SS {_f(ss_left['cv'])} salary {_f(sal_left['cv'])} {elapsed:.0f}s")


def _render(**k) -> str:
    def row(name, rec, leftover=None):
        left = leftover or rec
        return (
            f"| {name} | {left.get('sign', rec.get('sign', '—'))} | "
            f"{_f(rec['cv'])} | {_f(left['cv'])} | {fold_bits(left)} | "
            f"{left['n_pos']} |"
        )

    png = k["png"]
    return f"""# Y3 card reasons — leftover-KEEP stems only

Generated `{k['now']}` by `analysis/evaluate/y3_reasons.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Rates and leftover on **train** stressed months. Holdout 72 is coverage only
({k['hold_co']} companies / {k['hold_cm']} CM — no fit). Seed {FOLD_SEED}.
No 0–100. No `gbm_core.py`. No rewrite of `y3_importances.md` / `shap_y3.md`.
Y3 never B. Do not quote `a_out_vol` 0.722 as the engine.

Siddiqi (2017): reasons = factors actually scored; prefer 8–15 stems, but do
**not** pad DROPped SHAP names to hit 15. CFPB Circular 2023-03: reasons must
“relate to and accurately describe the factors actually considered or scored”;
the closest checklist item is not enough.

**Quote this:** among already-stressed months, recover (45→65) is more likely
when the book is quiet — no SS, no salary, fewer booking days. Those three
families are the only leftover-KEEP reasons. Historical SHAP still names
`a_n_tx` (perm 0.030); leftover QA DROPped it. We will not say it.

Night quotes locked: Y3 **{Y3_NIGHT[0]} / {Y3_NIGHT[1]}**. Days bar **{DAYS_BAR}**.
Size bar **{SIZE_BAR}**. `q6_keep` = `e_ar_issued_lag1` **{ISSUED_LAG1}** /
`c_n_days_with_tx_lag1` **{DAYS_LAG1}** / `c_ss_month_lag1` leftover **{SS_LAG1_LEFT}**.

Train Y3 labeled n={k['n_y3']:,} / n_pos={k['n_pos']:,}. Replica days {_f(k['days_rec']['cv'])}
(night {DAYS_BAR}); SS leftover {_f(k['ss_left']['cv'])} (night {SS_LEFT}); salary leftover
{_f(k['sal_left']['cv'])} (night {SAL_LEFT}).

---

## Judge-quotable reasons (factors actually scored)

Sign is − for every KEEP stem: among already-stressed months, **quiet** raises
P(recover). That is the 45→65 direction. It is not “more activity is healthier.”

| # | reason we will say | stem | sign | leftover / bar | brief Q |
|---|--------------------|------|:----:|----------------|---------|
| 1 | No social-security booking this month | `c_ss_month` | − | leftover after days **{SS_LEFT}** (replica {_f(k['ss_left']['cv'])}) | Q3 turning, Q5 why |
| 2 | No salary booking this month | `c_salary_month` | − | leftover after days **{SAL_LEFT}** (replica {_f(k['sal_left']['cv'])}) | Q3 turning, Q5 why |
| 3 | Fewer distinct booking days this month | `c_n_days_with_tx` | − | **bar {DAYS_BAR}** (replica {_f(k['days_rec']['cv'])}) | Q2 improving, Q3 turning |
| 4 | Fewer booking days last month | `c_n_days_with_tx_lag1` | − | **q6_keep {DAYS_LAG1}** (replica {_f(k['days_l1']['cv'])}) | Q6 months earlier |
| 5 | No SS booking last month | `c_ss_month_lag1` | − | leftover after days_lag1 **{SS_LAG1_LEFT}** (replica {_f(k['ss_l1_left']['cv'])}) | Q6 months earlier |
| 6 | No salary booking last month | `c_salary_month_lag1` | − | leftover after days_lag1 **{SAL_LAG1_LEFT}** (replica {_f(k['sal_l1_left']['cv'])}) | Q6 footnote (lives; thinner than SS) |
| 7 | Same stems at t−3 | `*_lag3` | − | SS leftover after days_lag3 replica {_f(k['ss_l3_left']['cv'])}; salary {_f(k['sal_l3_left']['cv'])}; days_lag3 {_f(k['days_l3']['cv'])} | Q6 CLOSE on company-short books |

Plain sentences a judge can read aloud:

1. **Who is healthy (Q1)?** Last-value `b_runway` / cash-buffer days. Not a card
   reason. Never B as Y3 X.
2. **Who is improving (Q2)?** Fewer booking days now than a busy stressed month
   (`c_n_days_with_tx` −, bar 0.711).
3. **Who is turning (Q3)?** Already-stressed months that go quiet on SS and
   salary (`c_ss_month` leftover 0.635, `c_salary_month` leftover 0.603). Engine
   0.762 / 15-col historical 0.752.
4. **Dip vs fall (Q4)?** Not this card. Invoice TURNOVER 0.720 is the sibling.
5. **Why did it change (Q5)?** Payroll-like drains stopped. SS leftover after
   days+salary still lives (published 0.633). Quiet is de-escalation, not size
   (`a_in3` leftover 0.521 DROP).
6. **How many months earlier (Q6)?** One month: days_lag1 0.684 and SS_lag1
   leftover 0.631. Hidden 72 stays a 1-month claim. `issued_lag1` 0.626 is Y7,
   not a Y3 reason.

---

## What we will NOT say

Historical SHAP (`y3_importances.md` 00:16; `shap_y3.md` 00:17) still ranks
these. Leftover QAs DROPped them as **card** stems. CFPB: do not check the
closest leftover name.

| dropped SHAP name | why it is not a reason | published leftover / flag |
|-------------------|------------------------|---------------------------|
| `a_n_tx` | days twin (ρ 0.938); perm ΔAUROC 0.030 is the trap | leftover after days **0.538** DROP |
| `a_op_in` | SIZE clone (ρ vs log inflow 0.998) | size bar 0.617; `a_in3` leftover **0.521** |
| `a_transfer` / `_lag1` / `_lag3` | company trait (ICC 0.956); perm ≈ 0 | leftover **0.579** CLOSE |
| `e_dso_proxy` | invoice sibling; sign flip SHAP − vs univ + | leftover after days **0.474** — see `lit_invoice` |
| `f_ds_r` / `_lag1` | unused leftover; twin of euro debt service | leftover **0.528** DROP |
| `c_gap_sd` | days twin (ρ −0.905) | leftover after days **0.535** DROP |
| `h_n_siblings_active_lag3` | sister *existence*, not sister health | Family H PARK as Y3 X |
| `a_out_vol` | company trait (demean 0.549); Javier vol DRIFT | raw **0.722** is not the engine |
| any `b_*` | honest Y≠X; walk is an identity | never B as Y2/Y3 X |
| `created_at` / `g_has_*` / `g_n_accounts` / `f_has_*` | connection clocks | `g_n_accounts` leftover **0.428** |
| `f_util_snapshot` | last-month-only 1.6% | Y10 impossible |
| `c_missed_salary` | skip, not presence | **0.513** CLOSE |
| `c_tax_month` | leftover after days dies | **0.520** DROP |

Do not pad the historical 15-col card back to 15 with those names. Siddiqi’s
8–15 is a preference; leftover QA left **three families**. The 15-col **0.752**
is a locked night quote, not a license to re-advertise DROPped SHAP.

---

## Six questions — card vs photograph

| # | question | what the card says | what it does not say |
|---|---------|--------------------|----------------------|
| 1 | Who is healthy? | Nothing. Q1 is last-value runway (Farrell buffer days). | Do not reason from B. |
| 2 | Who is improving? | Days − among already-stressed. | Not `c_gap_sd` irregularity. |
| 3 | Who is turning? | SS − and salary − (quiet-stressed recover). | Not bigger credits (Hair/FinRegLab PD). |
| 4 | Dip vs fall? | Out. Sibling TURNOVER. | Not DSO. |
| 5 | Why did it change? | Payroll-like drains off; activity clock down. | Not transfer / n_tx / ds_r / vol. |
| 6 | Months earlier? | days_lag1 + SS_lag1. | Not utilisation. Hidden 72 = 1 month. |

---

## Replica singles (train group-fold; sign −)

| stem | sign | raw CV | leftover after days | leftover folds | n_pos |
|------|:----:|-------:|--------------------:|----------------|------:|
{row("`c_n_days_with_tx`", k["days_rec"], k["days_rec"])}
{row("`c_ss_month`", k["ss_rec"], k["ss_left"])}
{row("`c_salary_month`", k["sal_rec"], k["sal_left"])}
{row("`log1p(a_in3)` size bar", k["size_rec"], k["size_rec"])}
{row("`c_n_days_with_tx_lag1`", k["days_l1"], k["days_l1"])}
{row("`c_ss_month_lag1` after days_lag1", k["ss_l1_left"], k["ss_l1_left"])}
{row("`c_salary_month_lag1` after days_lag1", k["sal_l1_left"], k["sal_l1_left"])}
{row("`c_n_days_with_tx_lag3`", k["days_l3"], k["days_l3"])}
{row("`c_ss_month_lag3` after days_lag3", k["ss_l3_left"], k["ss_l3_left"])}
{row("`c_salary_month_lag3` after days_lag3", k["sal_l3_left"], k["sal_l3_left"])}
{row("`c_ss_month` after days+salary", k["ss_after_both"], k["ss_after_both"])}
{row("`c_salary_month` after days+SS", k["sal_after_both"], k["sal_after_both"])}

---

## Extras — fold-wise reason stability

Leftover-after-days rank, five group folds (seed {FOLD_SEED}). KEEP if the
**mean** leftover stays ≥0.55 and at least four folds live. Do not drop a
stem because one fold dips.

| stem | leftover folds | mean | min | n_pos/fold |
|------|----------------|-----:|----:|------------|
| `c_ss_month` | {fold_bits(k["ss_left"])} | {_f(k["ss_left"]["cv"])} | {_f(min((r["auroc"] for r in k["ss_left"]["folds"] if np.isfinite(r["auroc"])), default=float("nan")))} | {" ".join(str(r["n_pos"]) for r in k["ss_left"]["folds"])} |
| `c_salary_month` | {fold_bits(k["sal_left"])} | {_f(k["sal_left"]["cv"])} | {_f(min((r["auroc"] for r in k["sal_left"]["folds"] if np.isfinite(r["auroc"])), default=float("nan")))} | {" ".join(str(r["n_pos"]) for r in k["sal_left"]["folds"])} |
| `c_n_days_with_tx` raw | {fold_bits(k["days_rec"])} | {_f(k["days_rec"]["cv"])} | {_f(min((r["auroc"] for r in k["days_rec"]["folds"] if np.isfinite(r["auroc"])), default=float("nan")))} | {" ".join(str(r["n_pos"]) for r in k["days_rec"]["folds"])} |

Published SS leftover folds 0.646 0.582 0.698 0.617 0.630 (spread 0.116).
Published salary leftover folds 0.550 0.573 0.708 0.618 0.567. Salary fold 0
sits on the 0.55 line — KEEP the **mean** 0.603, do not drop the stem.

---

## Extras — dark 470 vs ERP

Reasons are bank-book flags. Dark companies have no ERP invoices; they still
have SS/salary/days on the treasury trail. Do not require an invoice to say
quiet-stressed.

| slice | days raw | SS leftover | salary leftover | n_pos |
|-------|---------:|------------:|----------------:|------:|
| never-ERP (dark) | {_f(k["dark_days"]["cv"])} | {_f(k["dark_left_ss"]["cv"])} | {_f(k["dark_left_sal"]["cv"])} | {k["dark_days"]["n_pos"]} |
| ever-ERP | {_f(k["erp_days"]["cv"])} | {_f(k["erp_left_ss"]["cv"])} | {_f(k["erp_left_sal"]["cv"])} | {k["erp_days"]["n_pos"]} |

Published: SS leftover invoiced **0.677** / dark **0.574**; salary leftover ERP
**0.634** / dark **0.547** (dark salary leftover dies). Card sentence on a
**dark** book: say SS + days; do not lean on salary alone.

---

## Extras — short vs long books

`q6_keep` is a 1-month claim. Lag3 is CLOSE on company-short books
(days_lag3 nn 36.3%). Contemporaneous days/SS are SIGNAL on short books, not
lead time.

| slice | days raw | SS leftover | salary leftover | days_lag1 | n_pos |
|-------|---------:|------------:|----------------:|----------:|------:|
| so-far <12 (month clock) | {_f(k["short_days"]["cv"])} | {_f(k["short_ss"]["cv"])} | {_f(k["short_sal"]["cv"])} | {_f(k["short_l1"]["cv"])} | {k["short_days"]["n_pos"]} |
| so-far ≥18 (month clock) | {_f(k["long_days"]["cv"])} | {_f(k["long_ss"]["cv"])} | {_f(k["long_sal"]["cv"])} | {_f(k["long_l1"]["cv"])} | {k["long_days"]["n_pos"]} |
| company book <12 | {_f(k["cshort_days"]["cv"])} | {_f(k["cshort_ss"]["cv"])} | {_f(k["cshort_sal"]["cv"])} | {_f(k["cshort_l1"]["cv"])} | {k["cshort_days"]["n_pos"]} |
| company book 12–17 | {_f(k["cmid_days"]["cv"])} | {_f(k["cmid_ss"]["cv"])} | {_f(k["cmid_sal"]["cv"])} | {_f(k["cmid_l1"]["cv"])} | {k["cmid_days"]["n_pos"]} |
| company book ≥18 | {_f(k["clong_days"]["cv"])} | {_f(k["clong_ss"]["cv"])} | {_f(k["clong_sal"]["cv"])} | {_f(k["clong_l1"]["cv"])} | {k["clong_days"]["n_pos"]} |

so-far ≥18 Y3 labels are almost empty (need t+1..t+6). Company book <12 is
LOW_POWER (recover Y needs a future window). **Company-long** (≥18) is the
honest long-book slice — days 0.714 / SS leftover 0.632 / salary leftover 0.596
/ days_lag1 0.685. Hidden 72: =24 books **4.2%**. Do not fit. Do not claim a
3-month Y3 lead there.

355 / 402 Y3 positives sit on company-long books (88%). Company-short and
mid books are LOW_POWER for recover. The card is a **long-book reason list**.

---

## Extras — size tercile (published leftover; not re-fit)

Salary leftover after days **dies on T1** (0.399) and lives on T2 (0.620) /
T3 (0.653). SS leftover after days lives on T1 (0.576) and T2+T3 (0.616).
On the smallest stressed firms, say **days + SS**, not salary. Size bar
0.617 is not a reason.

---

## Explicitly out of this file

- A 0–100 formula, reason-code integers, ECOA adverse-action letters.
- Refit of the 278-col tree or the historical 15-col A (0.752 stays a quote).
- Family I / M merge. Wave D NSF token count. Wave A/B runway / Δdays.
- Invoice reasons (`e_dso_proxy`, issued, CN) — see `lit_invoice` / TURNOVER.
- Bankruptcy / Altman language.

{f"Plot: `{png}`." if png else ""}

Elapsed read-only. Night Y3 0.762/0.752, days 0.711, size 0.617 unchanged.
"""


def _append_registry(metrics: dict) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().astimezone().isoformat(timespec="seconds")
    row = (
        f"{ts},R4,4,y3_reasons,-,y3_recover_cash_6m,y3_reasons,n/a,"
        f"ss_left,{_f(metrics['ss_left'], 4)},{_f(metrics['sal_left'], 4)},"
        f"days={_f(metrics['days'])} ss_l1={_f(metrics['ss_l1'])} KEEP stems only"
    )
    with REGISTRY.open("a", newline="") as f:
        f.write(row + "\n")


if __name__ == "__main__":
    main()
