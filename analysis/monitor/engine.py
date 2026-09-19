"""The monitor (plan step 3): control charts, comparisons, alerts. Deterministic, as-of, parameters from monitor_params.json.

    run_monitor(detail, items, store, con, group_of, params) -> Monitor

Four comparisons, all on within-company (or within-group) change, never on levels:
  own_history        company score (and three category scores) vs its own lagged median, robust EWMA/CUSUM, 3-of-4 persistence
  cluster            the company's gap to the median of its behaviour cluster, same chart (a change of relative standing)
  group_own_history  group mean score vs its own history; floor sqrt(a + b/n) so small groups get wide limits
  group_vs_groups    the group's 3-month change against funnel limits fitted on train groups (needs at least MIN_GROUP members)
An alert is the ONSET of a persistent signal that is also material (moved at least MATERIAL points from the baseline).
Materiality is fixed a priori (not tuned on outcomes); evaluate.py reports how the results move with it.
Alert kinds: score_deterioration / score_improvement (two-sided), category_drop, going_dark, top_customer_quiet (decided in the plan).
Group alerts reuse the score kinds with entity type "group".
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from product.score import spec
from product.score.explain import _guard_sentence, eur, item_sentence

from . import behaviour, control, routing, topcustomer
from .fit import CATEGORIES, MIN_GROUP, group_mean_series, wide

MATERIAL = {"score": 8.0, "cluster_gap": 8.0, "payment_history": 10.0, "amounts_owed": 10.0, "stability": 10.0, "group": 8.0}
SEVERITY_ACT, SEVERITY_WATCH = 20.0, 12.0   # points away from the baseline
RULE = f"{control.PERSIST_K} of the last {control.PERSIST_N} months"
ITEM_NAMES = [i.name for i in spec.ITEMS]
MOVER_MIN = 0.5      # contribution points
ITEM_MOVE_MIN = 5.0  # item points (0-100)


@dataclass
class Monitor:
    grid: pd.DatetimeIndex
    charts: dict = field(default_factory=dict)         # company_id -> list of ControlChart dicts
    group_charts: dict = field(default_factory=dict)   # group_id -> list of ControlChart dicts
    cluster: pd.Series = None                          # company_id -> cluster id (NaN: trail too short)
    vs_cluster: dict = field(default_factory=dict)     # company_id -> ClusterMembership dict
    clusters: list = field(default_factory=list)       # ClusterIndex rows
    alerts: pd.DataFrame = None                        # one row per alert (dicts in `alert`), plus company_id, period, kind
    floors: dict = field(default_factory=dict)


# ---------------------------------------------------------------- charts
def _round(a, nd=2):
    return [None if not np.isfinite(v) else round(float(v), nd) + 0.0 for v in a]


def chart_json(comparison: str, metric: str, grid, x, ch: dict, method: dict) -> dict | None:
    """ControlChart (contract type) from arrays, trimmed to the months where the chart exists."""
    center = np.asarray(ch["center"], dtype=float)
    ok = np.flatnonzero(np.isfinite(center))
    if len(ok) == 0:
        return None
    s = slice(ok[0], None)
    out = {"comparison": comparison, "metric": metric, "months": [f"{p:%Y-%m}" for p in grid[s]], "values": _round(np.asarray(x, dtype=float)[s], 1),
           "center": _round(center[s], 1), "lower": _round(np.asarray(ch["lower"])[s], 1), "upper": _round(np.asarray(ch["upper"])[s], 1),
           "signal": ["low" if v < 0 else "high" if v > 0 else "none" for v in ch["signal"][s]], "persistent": [bool(v) for v in ch["persistent"][s]],
           "method": method}
    if "ewma_value" in ch:
        out["ewma"] = _round(np.asarray(ch["ewma_value"])[s], 1)
        out["cusum_low"] = _round(np.asarray(ch["cusum_low"])[s], 2)
        out["cusum_high"] = _round(np.asarray(ch["cusum_high"])[s], 2)
    return out


def material_flag(ch: dict, x: np.ndarray, material: float) -> np.ndarray:
    """Persistent signal that is also material: the smoothed level sits at least `material` points from the baseline, on the signal's side."""
    gap = np.asarray(ch["ewma_value"]) - np.asarray(ch["center"])
    d = ch["persistent"]
    return np.where((d != 0) & (np.sign(np.nan_to_num(gap)) == d) & (np.abs(np.nan_to_num(gap)) >= material), d, 0)


def score_flags(detail: pd.DataFrame, floors: dict, col: str = "score", floor_key: str = "score", material: float | None = None) -> pd.DataFrame:
    """Alert onsets on one score column, for every company: company_id, period, direction (-1 down / +1 up), gap, months_flagged."""
    grid = pd.date_range(detail["period"].min(), detail["period"].max(), freq="MS")
    w = wide(detail, col, grid)
    material = MATERIAL[floor_key] if material is None else material
    rows = []
    for cid, x in zip(w.index, w.to_numpy(dtype=float)):
        ch = control.chart(x, floors[floor_key])
        flag = material_flag(ch, x, material)
        for t in control.onsets(flag):
            d = int(flag[t])
            rows.append((cid, grid[t], d, float(ch["ewma_value"][t] - ch["center"][t]), int((ch["signal"][max(0, t - control.PERSIST_N + 1): t + 1] == d).sum())))
    return pd.DataFrame(rows, columns=["company_id", "period", "direction", "gap", "months_flagged"])


# ---------------------------------------------------------------- reasons
def _movers(detail, contrib, pts, items, i_now: int, i_prev: int | None, direction: int, cat: str | None = None, top: int = 4) -> list[dict]:
    """The items whose points moved most in the alert's direction since the baseline month (3 months before)."""
    out = []
    row_i = items.iloc[i_now]
    if i_prev is not None:
        d = contrib[i_now] - contrib[i_prev]
        dp = pts[i_now] - pts[i_prev]  # the item itself must have moved (not only the category weights around it)
        for j in np.argsort(d * direction * -1):  # largest movement in the direction first
            n = ITEM_NAMES[j]
            if direction > 0 and n in ("ds_increase", "fc_increase"):
                continue  # their sentence describes a rise; an improvement here is only "no longer rising"
            if not np.isfinite(d[j]) or d[j] * direction < MOVER_MIN or dp[j] * direction < ITEM_MOVE_MIN or (cat and spec.ITEM_CATEGORY[n] != cat):
                continue
            s, e = item_sentence(n, row_i)
            out.append({"item": n, "points": float(d[j]), "sentence": s, "eur": None if not np.isfinite(e) else float(e)})
    if cat is None and i_prev is not None:
        g_now = detail["score"].iat[i_now] - detail["score_raw"].iat[i_now]
        g_prev = detail["score"].iat[i_prev] - detail["score_raw"].iat[i_prev]
        dg = g_now - g_prev
        if np.isfinite(dg) and dg * direction >= MOVER_MIN and detail["guard"].iat[i_now]:
            s, e = _guard_sentence({**row_i.to_dict(), **detail.iloc[i_now].to_dict()})
            out.append({"item": "guard", "points": float(dg), "sentence": s, "eur": None if not np.isfinite(e) else float(e)})
    if cat is None and direction > 0 and i_prev is not None and not detail["guard"].iat[i_now] and detail["guard"].iat[i_prev]:
        out.append({"item": "guard", "points": float(detail["score"].iat[i_now] - detail["score"].iat[i_prev]), "sentence": "The activity cap no longer applies: bank activity has resumed.", "eur": None})
    if not out and "reasons" in detail.columns and direction < 0:
        # no single item moved enough (a spread-out fall): fall back to why the score is not higher, restricted to the category if there is one
        out = [x for x in detail["reasons"].iat[i_now] if x["item"] == "guard" or not cat or spec.ITEM_CATEGORY.get(x["item"]) == cat]
    out.sort(key=lambda x: -abs(x["points"]))
    return out[:top]


def _to_reason(x: dict, items_row) -> dict:
    from product.score.export import _reason

    return _reason(x, items_row, 1)


def _severity(gap: float) -> str:
    a = abs(gap)
    return "act" if a >= SEVERITY_ACT else "watch" if a >= SEVERITY_WATCH else "info"


# ---------------------------------------------------------------- the run
def run_monitor(detail: pd.DataFrame, items: pd.DataFrame, store: pd.DataFrame, con, group_of: pd.Series, params: dict,
                top_customer: bool = True) -> Monitor:
    """`detail`/`items`: score_store(..., return_items=True) with the same row order. `group_of`: company_id -> group_id."""
    detail = detail.reset_index(drop=True)
    items = items.reset_index(drop=True)
    assert (detail[["company_id", "period"]].values == items[["company_id", "period"]].values).all()
    floors = params["control"]["floors"]
    method = params["control"]["method"]
    grid = pd.date_range(detail["period"].min(), detail["period"].max(), freq="MS")
    mon = Monitor(grid=grid, floors=floors)
    pos = {(c, p): i for i, (c, p) in enumerate(zip(detail["company_id"], detail["period"]))}
    contrib = detail[[f"contrib_{n}" for n in ITEM_NAMES]].to_numpy(dtype=float)
    pts = detail[[f"pts_{n}" for n in ITEM_NAMES]].to_numpy(dtype=float)
    score_w = wide(detail, "score", grid)
    cat_w = {k: wide(detail, f"cat_{k}", grid) for k in CATEGORIES}
    guard_w = wide(detail.assign(g=detail["guard"].astype(str).eq("dark").astype(float)), "g", grid).fillna(0.0)

    # --- clusters
    cparams = params["clusters"]
    clusterer = behaviour.Clusterer(cparams["model"])
    member = clusterer.assign(behaviour.behaviour_table(store))
    member = member.reindex(score_w.index)
    mon.cluster = member
    k = cparams["model"]["k"]
    counts = member.value_counts()
    mon.clusters = [{"cluster_id": str(c), "label": cparams["model"]["labels"][c], "description": cparams["model"]["descriptions"][c],
                     "n_companies": int(counts.get(float(c), 0)), "n_companies_train_fit": int(cparams["model"]["sizes"][c])} for c in range(k)]
    peer_med = pd.DataFrame({c: score_w.loc[member.index[member == c]].median() for c in range(k)}).T
    ref = cparams["reference"]

    alerts: list[dict] = []
    onset_kw = dict(quiet_months=2)

    def add(cid, t, kind, direction, severity, gap, reasons, evidence, months_flagged, entity="company", rank=None, title="", summary="", owner=None, action=None):
        eid = str(cid)
        alerts.append({"company_id": eid, "period": grid[t] if isinstance(t, (int, np.integer)) else t, "kind": kind, "alert": {
            "alert_id": f"{kind}:{entity}:{eid}:{(grid[t] if isinstance(t, (int, np.integer)) else t):%Y-%m}" + (f":{evidence['category']}" if "category" in evidence else ""),
            "entity": {"type": entity, "id": eid, "name": None}, "month": f"{(grid[t] if isinstance(t, (int, np.integer)) else t):%Y-%m}",
            "kind": kind, "direction": direction, "severity": severity, "title": title, "summary": summary, "reasons": reasons,
            "persistence": {"rule": RULE, "months_flagged": int(months_flagged)}, "owner": owner, "action": action,
            "evidence": evidence, "rank_score": rank}})

    # --- company charts and score / category alerts
    for cid in score_w.index:
        xs = score_w.loc[cid].to_numpy(dtype=float)
        if not np.isfinite(xs).any():
            continue
        charts = []
        ch = control.chart(xs, floors["score"])
        c = chart_json("own_history", "score", grid, xs, ch, method)
        if c:
            charts.append(c)
        flag = material_flag(ch, xs, MATERIAL["score"])
        dark = guard_w.loc[cid].to_numpy()
        for t in control.onsets(flag, **onset_kw):
            d = int(flag[t])
            if dark[max(0, t - 2): t + 1].any():
                continue  # a going-dark alert covers it
            i_now = pos.get((cid, grid[t]))
            i_prev = pos.get((cid, grid[t - 3])) if t >= 3 else None
            if i_now is None:
                continue
            gap = float(ch["ewma_value"][t] - ch["center"][t])
            mv = _movers(detail, contrib, pts, items, i_now, i_prev, d)
            reasons = [_to_reason(x, items.iloc[i_now]) for x in mv]
            owner, action = routing.route_item(mv[0]["item"] if mv else None)
            kind, direction = ("score_deterioration", "risk") if d < 0 else ("score_improvement", "opportunity")
            if d > 0:
                action = routing.IMPROVE_ACTION
                owner = "cfo"
            now, base = float(xs[t]), float(ch["center"][t])
            add(cid, t, kind, direction, _severity(gap), gap, reasons,
                {"score": round(now, 1), "baseline": round(base, 1), "gap_points": round(gap, 1), "chart": "own_history"}, int((ch["signal"][max(0, t - control.PERSIST_N + 1): t + 1] == d).sum()),
                title=("Score is deteriorating against its own history" if d < 0 else "Score is improving against its own history"),
                summary=(f"Score {now:.0f} against a usual {base:.0f} ({gap:+.0f} points), outside its normal range in {RULE}. "
                         + (mv[0]["sentence"] if mv else "")).strip(),
                owner=owner, action=action)
        # cluster comparison: gap to the cluster median
        cl = member.get(cid)
        if pd.notna(cl):
            gap_x = xs - peer_med.loc[int(cl)].to_numpy(dtype=float)
            chg = control.chart(gap_x, floors["cluster_gap"])
            cj = chart_json("cluster", "score", grid, gap_x, chg, method)
            if cj:
                charts.append(cj)
        # category charts and drops
        for cat in CATEGORIES:
            xc = cat_w[cat].loc[cid].to_numpy(dtype=float)
            if np.isfinite(xc).sum() < control.MIN_PRIOR + control.BASELINE_LAG:
                continue
            chc = control.chart(xc, floors[cat])
            cj = chart_json("own_history", cat, grid, xc, chc, method)
            if cj:
                charts.append(cj)
            fc = material_flag(chc, xc, MATERIAL[cat])
            for t in control.onsets(fc, **onset_kw):
                d = int(fc[t])
                if d > 0 or dark[max(0, t - 2): t + 1].any() or flag[max(0, t - 2): t + 1].any():
                    continue  # category drops only; a score alert in the same window already carries the reasons
                i_now = pos.get((cid, grid[t]))
                i_prev = pos.get((cid, grid[t - 3])) if t >= 3 else None
                if i_now is None:
                    continue
                gap = float(chc["ewma_value"][t] - chc["center"][t])
                mv = _movers(detail, contrib, pts, items, i_now, i_prev, -1, cat=cat)
                owner, action = routing.route_item(mv[0]["item"] if mv else None)
                label = spec.CATEGORY_LABELS[cat]
                add(cid, t, "category_drop", "risk", _severity(gap * 0.5 if cat != "amounts_owed" else gap), gap, [_to_reason(x, items.iloc[i_now]) for x in mv],
                    {"category": cat, "category_score": round(float(xc[t]), 1), "baseline": round(float(chc["center"][t]), 1), "gap_points": round(gap, 1)},
                    int((chc["signal"][max(0, t - control.PERSIST_N + 1): t + 1] == d).sum()),
                    title=f"{label} is dropping against its own history",
                    summary=(f"{label} {xc[t]:.0f}/100 against a usual {chc['center'][t]:.0f}, outside its normal range in {RULE}. " + (mv[0]["sentence"] if mv else "")).strip(),
                    owner=owner, action=action)
        # going dark: first month of a dark run
        for t in np.flatnonzero(dark > 0):
            if not dark[max(0, t - 2): t].any():
                i_now = pos.get((cid, grid[t]))
                if i_now is None:
                    continue
                row_i, row_d = items.iloc[i_now], detail.iloc[i_now]
                s, _ = _guard_sentence({**row_i.to_dict(), **row_d.to_dict()})
                add(cid, t, "going_dark", "risk", "act", float(row_d["score"] - row_d["score_raw"]),
                    [_to_reason({"item": "guard", "points": float(row_d["score"] - row_d["score_raw"]), "sentence": s, "eur": None}, row_i)],
                    {"recency_days": int(row_i["recency_days"]), "score": round(float(row_d["score"]), 1), "score_pre_cap": round(float(row_d["score_raw"]), 1)}, 1,
                    title="Going dark: no bank movement for 60 days", summary=s, owner="treasurer", action=routing.DARK_ACTION)
        mon.charts[cid] = charts

    # --- vs cluster snapshot (latest scored month)
    last = detail[detail["score"].notna()].sort_values("period").groupby("company_id").tail(1).set_index("company_id")
    for cid, r in last.iterrows():
        cl = member.get(cid)
        if pd.isna(cl):
            continue
        rows = []
        for metric, col in [("score", "score")] + [(c, f"cat_{c}") for c in CATEGORIES]:
            v = r[col]
            rf = ref[str(int(cl))][metric]
            if not np.isfinite(v):
                rows.append({"metric": metric, "percentile": None, "robust_z": None})
                continue
            pct = float(np.mean(np.asarray(rf["quantiles"]) <= v) * 100.0)
            z = (v - rf["median"]) / max(rf["scale"], 1e-9)
            rows.append({"metric": metric, "percentile": round(pct, 0), "robust_z": round(float(z), 2)})
        mon.vs_cluster[cid] = {"cluster_id": str(int(cl)), "month": f"{r['period']:%Y-%m}", "vs_cluster": rows}

    # --- groups
    gp = params["group"]
    mean, n = group_mean_series(score_w, group_of)
    for gid in mean.index:
        nn = n.loc[gid].to_numpy(dtype=float)
        if nn.max() < gp["min_size"]:
            continue
        x = mean.loc[gid].to_numpy(dtype=float)
        floor = np.sqrt(gp["scale_law"]["a"] + gp["scale_law"]["b"] / np.maximum(nn, 1.0))
        ch = control.chart(x, floor)
        d3 = pd.Series(x).diff(3).to_numpy()
        fn = control.funnel_chart(d3, nn, gp["delta3_law"]["median"], gp["delta3_law"]["a"], gp["delta3_law"]["b"], gp["delta3_law"]["z_limit"], gp["min_size"])
        charts = [c for c in (chart_json("group_own_history", "score", grid, x, ch, method),
                              chart_json("group_vs_groups", "score", grid, d3, fn, {"name": "funnel_delta3", "params": {"z": gp["delta3_law"]["z_limit"], "min_members": gp["min_size"], **{k: round(v, 3) for k, v in gp["delta3_law"].items() if k in ("a", "b", "median")}}}))
                  if c]
        mon.group_charts[str(gid)] = charts
        flag = np.where(nn >= gp["min_size"], material_flag(ch, x, MATERIAL["group"]), 0)
        members = list(group_of.index[group_of == gid])
        for t in control.onsets(flag, **onset_kw):
            d = int(flag[t])
            sc_now = score_w.loc[[m for m in members if m in score_w.index], grid[t]]
            sc_prev = score_w.loc[[m for m in members if m in score_w.index], grid[t - 3]] if t >= 3 else sc_now * np.nan
            ch3 = (sc_now - sc_prev).dropna().sort_values(ascending=d > 0)
            worst = [{"company_id": m, "change_3m": round(float(v), 1)} for m, v in ch3.head(3).items()]
            gap = float(ch["ewma_value"][t] - ch["center"][t])
            kind, direction = ("score_deterioration", "risk") if d < 0 else ("score_improvement", "opportunity")
            add(str(gid), t, kind, direction, _severity(gap), gap, [],
                {"chart": "group_own_history", "n_companies": int(nn[t]), "mean_score": round(float(x[t]), 1), "baseline": round(float(ch["center"][t]), 1),
                 "gap_points": round(gap, 1), "outside_funnel_vs_other_groups": bool(fn["persistent"][t] == d),
                 "members_moving_most": ", ".join(f"{w['company_id']} ({w['change_3m']:+.1f})" for w in worst)},
                int((ch["signal"][max(0, t - control.PERSIST_N + 1): t + 1] == d).sum()), entity="group",
                title=("Group mean score is deteriorating" if d < 0 else "Group mean score is improving"),
                summary=(f"Mean score of {int(nn[t])} companies {x[t]:.0f} against a usual {ch['center'][t]:.0f} ({gap:+.0f} points)"
                         + (", also beyond what other groups of this size do" if fn["persistent"][t] == d else "") + "."),
                owner="cfo", action=routing.GROUP_ACTION)

    # --- top customer went quiet (a rule; the model only ranks)
    if top_customer:
        ev = topcustomer.onset_events(con)
        try:
            rk = topcustomer.rank_scores(store)
            ev = ev.merge(rk, on=["company_id", "period"], how="left")
        except (FileNotFoundError, OSError):
            ev["rank_score"] = np.nan
        cut = params.get("rank_model", {}).get("top_decile_cut")
        grid_set = set(grid)
        for r in ev.itertuples(index=False):
            if r.period not in grid_set or r.company_id not in score_w.index:
                continue
            rank = None if pd.isna(r.rank_score) else round(float(r.rank_score), 4)
            sev = "act" if (rank is not None and cut is not None and rank >= cut) else "info" if r.months_billed_of_3 >= 3 else "watch"
            open_e = None if pd.isna(r.open_receivable_eur) else float(r.open_receivable_eur)
            sent = (f"Customer {r.customer_id}, {r.share:.0%} of last quarter's billing ({eur(r.last_quarter_amount)}), has not been invoiced this month"
                    + (f"; {eur(open_e)} is still open with them." if open_e else "."))
            reason = {"item": "top_customer", "label": "Top customer stopped billing", "points": 0.0, "value": round(float(r.share), 3), "unit": "share",
                      "eur": None if pd.isna(r.last_quarter_amount) else round(float(r.last_quarter_amount), 0), "sentence": sent}
            add(r.company_id, r.period, "top_customer_quiet", "risk", sev, 0.0, [reason],
                {"counterparty_id": r.customer_id, "share_last_quarter": round(float(r.share), 3), "last_quarter_amount": round(float(r.last_quarter_amount), 0),
                 "months_billed_of_last_3": int(r.months_billed_of_3), "months_quiet": 1, "open_receivable_eur": None if open_e is None else round(open_e, 0)}, 1,
                rank=rank, title="Top customer stopped billing", summary=sent + " Review exposure and collections.",
                owner="collections", action=routing.top_customer_action(r.customer_id, open_e))
    mon.alerts = pd.DataFrame(alerts, columns=["company_id", "period", "kind", "alert"]).sort_values(["period", "company_id", "kind"]).reset_index(drop=True)
    return mon
