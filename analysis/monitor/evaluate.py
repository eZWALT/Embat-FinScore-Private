"""Evaluate the monitor on TRAIN companies (holdout untouched): false-alarm rate, lead time, dips, clusters, top-customer alert.

    PYTHONUTF8=1 PYTHONPATH=<repo>/.venv/Lib/site-packages:<repo> python -m analysis.monitor.evaluate

Monitoring claim only. Nothing is fitted on the outcomes here; the accepted outcomes are used the way the night used them,
as later events to date the alerts against. Writes evaluation.md and monitor_stats.json (quoted next to alerts in the bundle).

Alert-outcome protocol (per accepted outcome, train companies only):
  - the score is recomputed WITHOUT the feature families the outcome's label is built from (spec.OUTCOME_FORBIDDEN_FAMILIES)
  - risk alerts = onsets of a persistent, material fall in that score (own history) or a going-dark guard
  - an alert at month a is "followed" if the outcome label is 1 in some month of [a, a + HORIZON]; lead time = first such month - a
  - chance level = the same fraction over every chartable company-month (an alert on a random month)
  - false-alarm rate = 1 - followed rate. Lift = followed rate / chance rate. Lift near 1 means the alert carries no information
    about that outcome. Group-bootstrap intervals over group_id.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from product.score import spec
from . import control, engine, fit as mfit, topcustomer

OUT_MD = Path(__file__).resolve().parent / "evaluation.md"
STATS_PATH = Path(__file__).resolve().parent / "monitor_stats.json"
HORIZON = 6
N_BOOT = 300
SEED = 20260919
MATERIALS = (0.0, 4.0, 8.0, 12.0)


def _boot_ci(flag: np.ndarray, groups: np.ndarray, n=N_BOOT, seed=SEED) -> tuple[float, float]:
    """95% group-bootstrap interval of a mean of 0/1 flags (flag NaN rows dropped)."""
    ok = np.isfinite(flag)
    flag, groups = flag[ok], groups[ok]
    if len(flag) == 0:
        return (np.nan, np.nan)
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    uniq, start = np.unique(g, return_index=True)
    s = np.add.reduceat(flag[order], start)
    c = np.diff(np.append(start, len(g)))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(uniq), len(uniq))
        vals.append(s[pick].sum() / max(c[pick].sum(), 1))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def chart_bundle(detail: pd.DataFrame, floors: dict) -> dict:
    """Per company: grid arrays needed to re-flag at any materiality. Returns {cid: (ch, x)} and the grid."""
    grid = pd.date_range(detail["period"].min(), detail["period"].max(), freq="MS")
    w = mfit.wide(detail, "score", grid)
    dark = mfit.wide(detail.assign(g=detail["guard"].astype(str).eq("dark").astype(float)), "g", grid).fillna(0.0)
    out = {}
    for cid, x in zip(w.index, w.to_numpy(dtype=float)):
        out[cid] = (control.chart(x, floors["score"]), x, dark.loc[cid].to_numpy())
    return {"grid": grid, "charts": out}


def risk_onsets(cb: dict, material: float) -> pd.DataFrame:
    """Risk alerts (score fall onset not covered by a dark run, and going-dark onsets): company_id, t (grid index), kind."""
    rows = []
    for cid, (ch, x, dark) in cb["charts"].items():
        flag = engine.material_flag(ch, x, material)
        for t in control.onsets(np.where(flag < 0, flag, 0), quiet_months=2):
            if not dark[max(0, t - 2): t + 1].any():
                rows.append((cid, t, "score_deterioration"))
        for t in np.flatnonzero(dark > 0):
            if not dark[max(0, t - 2): t].any():
                rows.append((cid, t, "going_dark"))
    return pd.DataFrame(rows, columns=["company_id", "t", "kind"])


def up_onsets(cb: dict, material: float) -> pd.DataFrame:
    rows = []
    for cid, (ch, x, dark) in cb["charts"].items():
        flag = engine.material_flag(ch, x, material)
        rows += [(cid, t, "score_improvement") for t in control.onsets(np.where(flag > 0, flag, 0), quiet_months=2)]
    return pd.DataFrame(rows, columns=["company_id", "t", "kind"])


def followed_table(y_w: pd.DataFrame, grid) -> tuple[np.ndarray, np.ndarray]:
    """For every (company, month t): lead = months to the first y=1 in [t, t+HORIZON] (NaN if none), and whether y is observable
    over the full window (else the row cannot be judged). Returns (lead, judged) as arrays shaped like y_w."""
    y = y_w.to_numpy(dtype=float)
    n_c, n_t = y.shape
    lead = np.full((n_c, n_t), np.nan)
    judged = np.zeros((n_c, n_t), dtype=bool)
    for t in range(n_t):
        hi = min(n_t, t + HORIZON + 1)
        win = y[:, t:hi]
        judged[:, t] = (hi - t == HORIZON + 1) & (np.isfinite(win).sum(axis=1) >= 1)  # a full window with at least one observed label
        first = np.argmax(np.nan_to_num(win) > 0, axis=1)
        has = (np.nan_to_num(win) > 0).any(axis=1)
        lead[:, t] = np.where(has, first, np.nan)
    return lead, judged


def outcome_report(detail_by_outcome: dict, outcomes: pd.DataFrame, groups: pd.Series, floors: dict, train: set[str], materials=MATERIALS) -> pd.DataFrame:
    """One row per outcome and materiality: alerts, followed rate, chance rate, lift, false-alarm rate, median lead time."""
    rows = []
    for oc, det in detail_by_outcome.items():
        det = det[det["company_id"].isin(train)]
        cb = chart_bundle(det, floors)
        grid = cb["grid"]
        ids = list(cb["charts"])
        idx = {c: i for i, c in enumerate(ids)}
        yw = mfit.wide(outcomes[["company_id", "period", oc]].rename(columns={oc: "y"}).merge(det[["company_id", "period"]], on=["company_id", "period"]), "y", grid).reindex(ids)
        lead, judged = followed_table(yw, grid)
        chartable = np.zeros_like(judged)
        for cid, (ch, x, dark) in cb["charts"].items():
            chartable[idx[cid]] = np.isfinite(ch["center"]) & np.isfinite(x)
        base_rows = judged & chartable
        chance = float(np.mean(np.isfinite(lead[base_rows]))) if base_rows.any() else np.nan
        chance_gid = np.repeat(groups.reindex(ids).astype(str).to_numpy()[:, None], len(grid), axis=1)[base_rows]
        for m in materials:
            al = risk_onsets(cb, m)
            if al.empty:
                continue
            ri = al["company_id"].map(idx).to_numpy()
            ok = judged[ri, al["t"].to_numpy()]
            ri, tt = ri[ok], al["t"].to_numpy()[ok]
            ld = lead[ri, tt]
            foll = np.isfinite(ld)
            gid = groups.reindex([ids[i] for i in ri]).astype(str).to_numpy()
            lo, hi = _boot_ci(foll.astype(float), gid)
            rows.append({"outcome": oc, "material": m, "alerts": int(len(ri)), "followed": float(foll.mean()) if len(ri) else np.nan, "ci_lo": lo, "ci_hi": hi,
                         "chance": chance, "lift": float(foll.mean() / chance) if chance else np.nan, "false_alarm_rate": float(1 - foll.mean()),
                         "false_alarm_chance": float(1 - chance), "median_lead_months": float(np.median(ld[foll])) if foll.any() else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    from analysis.evaluate.protocol import train_companies
    from analysis.features.common import connect
    from analysis.models.gbm_y7y8 import _keys, load_store, load_y
    from product.score import fit as score_fit
    from product.score import validate as sv
    from product.score.run import score_store

    con = connect()
    store, _ = load_store(con)
    store = _keys(store)
    train_df = train_companies(con)
    train = set(train_df["company_id"].astype(str))
    params = mfit.load()
    floors = params["control"]["floors"]
    ref = score_fit.load_reference()
    comp = con.execute("SELECT company_id, group_id FROM companies").df()
    comp["company_id"] = comp["company_id"].astype(str)
    groups = comp.set_index("company_id")["group_id"].astype(str)

    print("scoring...")
    detail_full, items = score_store(store, ref, con=con, return_items=True)
    detail_full["company_id"] = detail_full["company_id"].astype(str)
    outcomes = sv.load_outcomes(store)
    outcomes["company_id"] = outcomes["company_id"].astype(str)
    by_outcome = {}
    for oc, fam in spec.OUTCOME_FORBIDDEN_FAMILIES.items():
        by_outcome[oc] = score_store(store, ref, with_reasons=False, drop_families=frozenset(fam))
        by_outcome[oc]["company_id"] = by_outcome[oc]["company_id"].astype(str)
    print("outcome report...")
    rep = outcome_report(by_outcome, outcomes, groups, floors, train)
    rep.to_csv(Path(__file__).resolve().parent / "evaluation_outcomes.csv", index=False)

    # ---- volume (train companies, full score)
    tr_detail = detail_full[detail_full["company_id"].isin(train)]
    cb = chart_bundle(tr_detail, floors)
    charted = sum(int(np.isfinite(ch["center"]).sum()) for ch, x, d in cb["charts"].values())
    vol = {}
    for m in MATERIALS:
        dn, up = risk_onsets(cb, m), up_onsets(cb, m)
        vol[m] = {"deterioration": int((dn.kind == "score_deterioration").sum()), "going_dark": int((dn.kind == "going_dark").sum()), "improvement": int(len(up)),
                  "per_company_year_risk": len(dn) / charted * 12, "per_company_year_up": len(up) / charted * 12}

    # ---- dips: one-month falls of >= 8 points, do they become alerts?
    piv = mfit.wide(tr_detail, "score", cb["grid"])
    dn8 = risk_onsets(cb, engine.MATERIAL["score"])
    alerted = set(zip(dn8["company_id"], dn8["t"]))
    dips = kept = 0
    for cid, x in zip(piv.index, piv.to_numpy(dtype=float)):
        d = np.diff(x)
        for t in np.flatnonzero(d <= -engine.MATERIAL["score"]) + 1:
            if not np.isfinite(control.chart(x, floors["score"])["center"][t]):
                continue
            dips += 1
            kept += any((cid, u) in alerted for u in range(t, min(t + 4, len(x))))
    dip_share = kept / max(dips, 1)

    # ---- clusters
    tab = engine.behaviour.behaviour_table(store)
    member = engine.behaviour.Clusterer(params["clusters"]["model"]).assign(tab)
    tt = tab.join(member.rename("cluster")).dropna(subset=["cluster"])
    tt = tt[tt.index.isin(train)]
    eta = float(1 - sum(((g["log_inflow"] - g["log_inflow"].mean()) ** 2).sum() for _, g in tt.groupby("cluster")) / ((tt["log_inflow"] - tt["log_inflow"].mean()) ** 2).sum())
    from sklearn.metrics import adjusted_rand_score

    ari = []
    tr_tab = tab[tab.index.isin(train)]
    rng = np.random.default_rng(SEED)
    for s in range(3):
        idx = rng.permutation(len(tr_tab))
        half = [tr_tab.iloc[idx[: len(idx) // 2]], tr_tab.iloc[idx[len(idx) // 2:]]]
        fits = [engine.behaviour.Clusterer.fit(h, k=params["clusters"]["model"]["k"]) for h in half]
        cl_a = engine.behaviour.Clusterer({k: v for k, v in fits[0].items() if k != "train_assignment"}).assign(tr_tab)
        cl_b = engine.behaviour.Clusterer({k: v for k, v in fits[1].items() if k != "train_assignment"}).assign(tr_tab)
        ok = cl_a.notna() & cl_b.notna()
        ari.append(float(adjusted_rand_score(cl_a[ok], cl_b[ok])))
    last = tr_detail[tr_detail["score"].notna()].sort_values("period").groupby("company_id").tail(1).set_index("company_id")
    ctab = tt.join(last[["score"]]).groupby("cluster").agg(n=("log_inflow", "size"), mean_score=("score", "mean"), mean_log_inflow=("log_inflow", "mean"))
    n_train_months = charted

    # ---- top customer: rule and severity, out-of-fold ranks
    grid7 = store[["company_id", "period"]].drop_duplicates()
    y7, _y8, _ = load_y(con, grid7)
    y7 = _keys(y7[["company_id", "period", "y7_top1_lost"]])
    oof = topcustomer.oof_rank_scores(store, y7, train_df)
    ev = topcustomer.onset_events(con)
    ev_all = topcustomer.quiet_events(con)
    for e in (ev, ev_all):
        e["period"] = pd.to_datetime(e["period"])
    y = y7[y7["company_id"].isin(train) & y7["y7_top1_lost"].notna()]
    base = float(y["y7_top1_lost"].mean())

    def top_stats(e, label):
        d = y.merge(e[["company_id", "period"]].assign(flag=1), on=["company_id", "period"], how="left")
        d["flag"] = d["flag"].fillna(0)
        f = d[d["flag"] == 1]
        gid = groups.reindex(f["company_id"]).to_numpy()
        lo, hi = _boot_ci(f["y7_top1_lost"].to_numpy(dtype=float), gid)
        return {"label": label, "flagged_rows": int(len(f)), "flag_rate": float(d["flag"].mean()), "precision": float(f["y7_top1_lost"].mean()), "ci": [lo, hi],
                "lift": float(f["y7_top1_lost"].mean() / base), "recall": float(((d["flag"] == 1) & (d["y7_top1_lost"] == 1)).sum() / (d["y7_top1_lost"] == 1).sum())}
    tc_all, tc_onset = top_stats(ev_all, "every quiet month"), top_stats(ev, "onset only")
    evo = ev.merge(oof, on=["company_id", "period"], how="left").merge(y, on=["company_id", "period"], how="inner")
    cut = params["rank_model"]["top_decile_cut"]
    sev = np.where(evo["rank_score"] >= cut, "act", np.where(evo["months_billed_of_3"] >= 3, "info", "watch"))
    evo["severity"] = sev
    sev_tab = evo.groupby("severity").agg(n=("y7_top1_lost", "size"), precision=("y7_top1_lost", "mean")).reset_index()

    # ---- group funnel calibration on train groups
    gp = params["group"]
    score_w = mfit.wide(tr_detail, "score", cb["grid"])
    mean, n = mfit.group_mean_series(score_w, groups.reindex(score_w.index).where(lambda s: s != "nan"))
    out_share, tot = [], 0
    for gid in mean.index:
        nn = n.loc[gid].to_numpy(dtype=float)
        if nn.max() < gp["min_size"]:
            continue
        d3 = pd.Series(mean.loc[gid].to_numpy(dtype=float)).diff(3).to_numpy()
        fn = control.funnel_chart(d3, nn, gp["delta3_law"]["median"], gp["delta3_law"]["a"], gp["delta3_law"]["b"], gp["delta3_law"]["z_limit"], gp["min_size"])
        judged = np.isfinite(d3) & (nn >= gp["min_size"])
        out_share.append((int(judged.sum()), int((fn["signal"][judged] != 0).sum()), int((fn["persistent"][judged] != 0).sum())))
    o = np.array(out_share)
    funnel = {"group_months": int(o[:, 0].sum()), "outside_share": float(o[:, 1].sum() / o[:, 0].sum()), "persistent_share": float(o[:, 2].sum() / o[:, 0].sum())}

    # ---- assemble headline
    m8 = rep[rep["material"] == engine.MATERIAL["score"]]
    headline = {
        "false_alarm_rate": float(m8["false_alarm_rate"].mean()), "false_alarm_rate_at_chance": float(m8["false_alarm_chance"].mean()),
        "median_lead_time_months": float(m8["median_lead_months"].median()),
        "lift_range": [float(m8["lift"].min()), float(m8["lift"].max())],
    }
    stats = {
        "fitted_on": "train companies only (holdout untouched)", "horizon_months": HORIZON, "material_points": engine.MATERIAL["score"],
        "headline": headline, "volume_per_company_year": vol[engine.MATERIAL["score"]],
        "dips_that_persist": {"one_month_falls_8pts": dips, "became_alert_within_3_months": kept, "share": dip_share},
        "top_customer": {"base_rate": base, "every_quiet_month": tc_all, "onset_only": tc_onset, "by_severity": sev_tab.to_dict("records"),
                         "top_decile_cut": cut},
        "clusters": {"k": params["clusters"]["model"]["k"], "silhouette_by_k": params["clusters"]["model"]["silhouette_by_k"], "eta2_log_inflow": eta,
                     "half_split_ari": ari, "table": ctab.reset_index().to_dict("records")},
        "group_funnel": funnel,
        "note": (f"Measured on train companies against the accepted outcomes (score recomputed without the outcome's own feature families). "
                 f"Lift near 1 means the alert says nothing about that outcome."),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=1, ensure_ascii=False, default=float, allow_nan=False) + "\n", encoding="utf8")
    write_md(stats, rep, vol)
    print(json.dumps(stats["headline"], indent=1))
    return 0


def write_md(stats: dict, rep: pd.DataFrame, vol: dict) -> None:
    h = stats["headline"]
    L = ["# Monitor evaluation (train companies, holdout untouched)", "",
         "Generated by `python -m analysis.monitor.evaluate`. **Monitoring claim only**: the alerts flag a company that has moved away from its own",
         "normal, with the reason and the amount behind it; the tables below say how much each alert says about the accepted outcomes. Read them",
         "with the lift column: near 1 means no information about that outcome.", "",
         "## Headline", "",
         f"- Material score falls (>= {stats['material_points']:.0f} points below the company's own baseline, 3 of the last 4 months): mean false-alarm rate "
         f"**{h['false_alarm_rate']:.0%}** across the eight accepted outcomes, against {h['false_alarm_rate_at_chance']:.0%} for an alert on a random month "
         f"(followed = the outcome label is 1 within {stats['horizon_months']} months of the alert).",
         f"- Lift over chance per outcome: {h['lift_range'][0]:.2f} to {h['lift_range'][1]:.2f}. Median lead time from alert to the outcome label: **{h['median_lead_time_months']:.0f} months** "
         "(alerts that are followed).", ""]
    v = stats["volume_per_company_year"]
    L += ["## Volume", "", f"At {stats['material_points']:.0f} points: {v['per_company_year_risk']:.2f} risk alerts and {v['per_company_year_up']:.2f} improvement alerts per company-year "
          f"({v['deterioration']} deteriorations, {v['going_dark']} going dark, {v['improvement']} improvements on train).", ""]
    d = stats["dips_that_persist"]
    L += ["## Dips that do not become alerts", "", f"One-month falls of 8 points or more: {d['one_month_falls_8pts']}. {d['share']:.0%} of them became an alert within 3 months; the rest reverted "
          "or stayed inside the company's normal range and raised nothing (this is what the persistence rule is for).", ""]
    L += ["## Alerts against the accepted outcomes", "", "Risk alerts = score falls and going-dark. Score recomputed without the families the outcome is built from.", "",
          "| outcome | material | alerts | followed (95% CI) | chance | lift | false alarms | median lead (months) |", "|---|---|---|---|---|---|---|---|"]
    for r in rep.itertuples():
        L.append(f"| {r.outcome} | {r.material:.0f} | {r.alerts} | {r.followed:.0%} ({r.ci_lo:.0%}-{r.ci_hi:.0%}) | {r.chance:.0%} | {r.lift:.2f} | {r.false_alarm_rate:.0%} | {r.median_lead_months:.0f} |")
    L += ["", "Materiality (points) was fixed a priori at 8; the other rows show how the result moves.", ""]
    t = stats["top_customer"]
    L += ["## Top customer went quiet", "", f"Base rate of losing the top customer in the labelled months: {t['base_rate']:.0%}.", "",
          "| trigger | flagged rows | flag rate | precision (95% CI) | lift | recall |", "|---|---|---|---|---|---|"]
    for k in ("every_quiet_month", "onset_only"):
        x = t[k]
        L.append(f"| {x['label']} | {x['flagged_rows']} | {x['flag_rate']:.0%} | {x['precision']:.0%} ({x['ci'][0]:.0%}-{x['ci'][1]:.0%}) | {x['lift']:.2f} | {x['recall']:.0%} |")
    L += ["", "By severity (the model score is out-of-fold and only ranks; `act` = top decile of onset events, `info` = the customer billed in all 3 of the last 3 months):", "",
          "| severity | events | precision |", "|---|---|---|"] + [f"| {s['severity']} | {s['n']} | {s['precision']:.0%} |" for s in t["by_severity"]]
    c = stats["clusters"]
    L += ["", "## Behaviour clusters", "", f"k = {c['k']} chosen by silhouette ({c['silhouette_by_k']}). Structure is weak (silhouette < 0.2): clusters are peer groups for a comparison, "
          f"not segments. Size signal left in the clusters: eta^2 with log inflow = {c['eta2_log_inflow']:.3f} (0.23 before regressing size out). Stability, adjusted Rand between fits on random "
          f"halves of train: {', '.join(f'{a:.2f}' for a in c['half_split_ari'])}.", "", "| cluster | companies (train) | mean latest score | mean log inflow |", "|---|---|---|---|"]
    L += [f"| {int(r['cluster'])} | {r['n']} | {r['mean_score']:.1f} | {r['mean_log_inflow']:.2f} |" for r in c["table"]]
    g = stats["group_funnel"]
    L += ["", "## Group funnel", "", f"Train group-months judged (groups of at least 3): {g['group_months']}. Outside the 3-sigma funnel: {g['outside_share']:.1%}; persistent (3 of 4): {g['persistent_share']:.1%}."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf8")


if __name__ == "__main__":
    raise SystemExit(main())
