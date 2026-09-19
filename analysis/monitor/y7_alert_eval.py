"""Alert-grade evaluation of the Y7 (top customer lost) card: precision / recall / lift.

Reuses the night's panel + TURNOVER card + shallow LightGBM. Adds what the night
did not report: precision at a flag rate, lift over base rate, a stricter 6-month
label, a one-line rule baseline, and group-bootstrap intervals.
Train = group-fold OOF. Holdout is fit-on-all-train, one look.
"""
from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from analysis.evaluate.protocol import FOLD_SEED, auroc, group_folds, load_holdout, train_companies
from analysis.features.common import LAST_M, MONTHS, connect
from analysis.models.gbm_y7_core import (
    CORE_TURNOVER, Y_COL, _fit_shallow, _resolve_cols, prepare_panel,
)
from analysis.models.gbm_y7y8 import _keys, load_store, load_y
from analysis.targets.y7_concentration import _ar_month_cp, _invoice_top1_panel, _roll_cp

RNG = np.random.default_rng(7)
con = connect()
store, _ = load_store(con)
store = _keys(store)
train = train_companies(con)
folds = group_folds(train, n=5, seed=FOLD_SEED)
grid = store[["company_id", "period"]].drop_duplicates()
y7, _y8, _ = load_y(con, grid)
y7 = _keys(y7[["company_id", "period", "y7_top1_lost", "y7_top1_lost_inflow"]])
panel = prepare_panel(store, y7[["company_id", "period", Y_COL]], folds)
panel = panel.merge(y7[["company_id", "period", "y7_top1_lost_inflow"]], on=["company_id", "period"], how="left")

# --- extra labels / rule from raw invoices (same helper the night used)
top = _invoice_top1_panel(con)                       # company, period, top1_id, top1_share, nxt_amt
mcp = _ar_month_cp(con)
top = _keys(top.rename(columns={"period": "period"}))
# amount to the same customer in t+4..t+6 (did it come back?)
later = _roll_cp(mcp, (-4, -5, -6)).rename(columns={"amt": "later_amt", "counterparty_id": "top1_id"})
top = top.merge(later, on=["company_id", "period", "top1_id"], how="left")
top["later_amt"] = top["later_amt"].fillna(0.0)
top["can_6m"] = top["period"] <= (LAST_M - pd.DateOffset(months=6))
top["y7_strict6"] = np.where(top["can_6m"], ((top["nxt_amt"] <= 0) & (top["later_amt"] <= 0)).astype(float), np.nan)
# rule: last month's top-1 (window t-3..t-1) got nothing in month t
prev = top[["company_id", "period", "top1_id"]].copy()
prev["period"] = prev["period"] + pd.DateOffset(months=1)
prev = prev.rename(columns={"top1_id": "prev_top1"})
this = mcp.rename(columns={"month": "period", "counterparty_id": "prev_top1", "amt": "amt_now"})
rule = prev.merge(this, on=["company_id", "period", "prev_top1"], how="left")
rule["rule_quiet"] = (rule["amt_now"].fillna(0.0) <= 0).astype(float)
extra = top[["company_id", "period", "nxt_amt", "later_amt", "y7_strict6"]].merge(
    rule[["company_id", "period", "rule_quiet"]], on=["company_id", "period"], how="left")
panel = panel.merge(_keys(extra), on=["company_id", "period"], how="left")

hold = load_holdout()
panel["is_hold"] = panel["company_id"].isin(hold)
assert 'group_id' in panel.columns, panel.columns.tolist()[:10]

cols = _resolve_cols(panel, CORE_TURNOVER)
lab = panel[Y_COL].notna() & ~panel["is_hold"]
tr = panel[lab].copy().reset_index(drop=True)
print(f"train labeled rows {len(tr)}  companies {tr.company_id.nunique()}  groups {tr.group_id.nunique()}  base {tr[Y_COL].mean():.4f}")

# --- OOF scores
tr["oof"] = np.nan
for k in range(5):
    fit = tr[tr["fold"] != k]; ev = tr["fold"] == k
    clf = _fit_shallow(fit[cols], fit[Y_COL].astype(int))
    tr.loc[ev, "oof"] = clf.predict_proba(tr.loc[ev, cols])[:, 1]

def prec_rec(df, ycol, score, frac):
    d = df[df[ycol].notna() & df[score].notna()]
    cut = d[score].quantile(1 - frac)
    flag = d[score] >= cut
    tp = ((d[ycol] == 1) & flag).sum()
    p = tp / max(flag.sum(), 1); r = tp / max((d[ycol] == 1).sum(), 1); b = (d[ycol] == 1).mean()
    return p, r, b

def boot(df, fn, n=300):
    groups = df["group_id"].unique()
    idx = {g: np.flatnonzero((df["group_id"] == g).to_numpy()) for g in groups}
    out = []
    for _ in range(n):
        pick = RNG.choice(groups, len(groups), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        try:
            out.append(fn(df.iloc[rows]))
        except Exception:
            pass
    a = np.array(out, dtype=float)
    return np.nanpercentile(a, [2.5, 97.5])

print("\n=== OOF AUROC by label (same TURNOVER scores) ===")
for ycol in (Y_COL, "y7_top1_lost_inflow", "y7_strict6"):
    d = tr[tr[ycol].notna()]
    ci = boot(d, lambda x: auroc(x[ycol], x["oof"]), 200)
    print(f"{ycol:22s} n={len(d):6d} pos={int(d[ycol].sum()):5d} base={d[ycol].mean():.3f}  AUROC={auroc(d[ycol], d['oof']):.3f}  95%CI[{ci[0]:.3f},{ci[1]:.3f}]  (label defined on {d.company_id.nunique()} cos)")

print("\n=== precision / recall / lift when flagging top X% of company-months by OOF score ===")
for ycol in (Y_COL, "y7_top1_lost_inflow", "y7_strict6"):
    d = tr[tr[ycol].notna()]
    for frac in (0.05, 0.10, 0.20, 0.30):
        p, r, b = prec_rec(d, ycol, "oof", frac)
        ci = boot(d, lambda x, f=frac: prec_rec(x, ycol, "oof", f)[0], 150)
        print(f"{ycol:22s} flag {int(frac*100):2d}%  precision={p:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  base={b:.3f}  lift={p/b:.2f}x  recall={r:.3f}")

print("\n=== one-line rule: last month's top customer got NO invoice this month (rule_quiet=1) ===")
for ycol in (Y_COL, "y7_top1_lost_inflow", "y7_strict6"):
    d = tr[tr[ycol].notna() & tr["rule_quiet"].notna()]
    flag = d["rule_quiet"] == 1
    tp = ((d[ycol] == 1) & flag).sum()
    p = tp / max(flag.sum(), 1); r = tp / max((d[ycol] == 1).sum(), 1); b = d[ycol].mean()
    ci = boot(d, lambda x: ((x[ycol] == 1) & (x["rule_quiet"] == 1)).sum() / max((x["rule_quiet"] == 1).sum(), 1), 150)
    print(f"{ycol:22s} rows={len(d):6d} flagged={flag.mean():.3f}  precision={p:.3f} [{ci[0]:.3f},{ci[1]:.3f}]  base={b:.3f}  lift={p/b:.2f}x  recall={r:.3f}  AUROC(rule)={auroc(d[ycol], d['rule_quiet']):.3f}")

print("\n=== does the 'lost' customer come back? (share of y7_top1_lost=1 rows with billing in t+4..t+6) ===")
d = tr[(tr[Y_COL] == 1) & tr["y7_strict6"].notna()]
print(f"rows with t+6 observable: {len(d)}  returned within t+4..t+6: {(d['later_amt'] > 0).mean():.3f}")
d2 = tr[tr["y7_strict6"].notna()]
print(f"3-month label base {d2[Y_COL].mean():.3f} vs 6-month strict base {d2['y7_strict6'].mean():.3f}")

print("\n=== company-level: flagged at top-10% score, how many distinct companies, per-company-year false alarms ===")
d = tr[tr[Y_COL].notna()].copy()
cut = d["oof"].quantile(0.90)
d["flag"] = d["oof"] >= cut
fl = d[d["flag"]]
print(f"flagged rows {len(fl)} ({len(fl)/len(d):.3f}) across {fl.company_id.nunique()} of {d.company_id.nunique()} companies; false-alarm share {(fl[Y_COL]==0).mean():.3f}")
ep = fl.sort_values(["company_id", "period"])
new_ep = (ep.groupby("company_id")["period"].diff().dt.days.fillna(999) > 40)
print(f"distinct alert episodes {int(new_ep.sum())}; episodes per flagged company {new_ep.sum()/fl.company_id.nunique():.2f}")

print("\n=== HOLDOUT (one look; fit on all train; same fixed OOF top-10% cut) ===")
clf = _fit_shallow(tr[cols], tr[Y_COL].astype(int))
h = panel[panel["is_hold"] & panel[Y_COL].notna()].copy()
h["s"] = clf.predict_proba(h[cols])[:, 1]
flag = h["s"] >= cut
tp = ((h[Y_COL] == 1) & flag).sum()
print(f"holdout rows {len(h)} cos {h.company_id.nunique()} pos {int(h[Y_COL].sum())} base {h[Y_COL].mean():.3f}  AUROC {auroc(h[Y_COL], h['s']):.3f}  flagged {flag.mean():.3f}  precision {tp/max(flag.sum(),1):.3f}  recall {tp/max((h[Y_COL]==1).sum(),1):.3f}")
