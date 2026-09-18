"""Python port of the score pipeline in 02_score_salud_financiera.Rmd.

Reads data/embat.duckdb (see build_db.py). Same logic as the notebook:
monthly signals -> percentile scoring against a frozen reference -> 5 pillars -> score
-> trajectory/states. No look-ahead: every month uses only data up to that month.
"""
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "embat.duckdb"
AS_OF = pd.Timestamp("2026-09-01")
MONTHS = pd.date_range("2024-09-01", "2026-08-01", freq="MS")
LAST_M = MONTHS[-1]
MI = {m: i + 1 for i, m in enumerate(MONTHS)}

CAT_MAP = {
    "collection": "op_in", "bulk_collection": "op_in", "cash_settlement": "op_in", "cash_settlements": "op_in",
    "pos_settlement": "op_in", "collection_refund": "op_in",
    "payment": "op_out", "bulk_payment": "op_out", "utility": "op_out", "payment_refund": "op_out",
    "salary": "op_out", "social_security": "op_out", "tax": "op_out", "tax_refund": "op_out",
    "cash_withdrawal": "op_out", "pos_withdrawal": "op_out",
    "fee": "fin_cost", "interest_charge": "fin_cost",
    "debt_repayment": "debt_service",
    "transfer": "transfer", "investment_deployment": "invest", "investment_return": "invest",
}

SIG = pd.DataFrame({
    "signal": ["runway", "d_runway", "neg_liq", "coverage", "net_margin", "volatility", "growth",
               "concentration", "fin_cost_r", "debt_serv_r", "ar_overdue", "ap_overdue", "delay_coll", "delay_paid"],
    "pillar": ["liquidez", "liquidez", "liquidez", "caja", "caja", "estabilidad", "estabilidad",
               "estabilidad", "financiera", "financiera", "cobros", "cobros", "cobros", "cobros"],
    "direction": [1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
    "type": ["pct", "pct", "share"] + ["pct"] * 11,
})
PILLAR_W = {"liquidez": 0.30, "caja": 0.25, "estabilidad": 0.15, "financiera": 0.10, "cobros": 0.20}


def connect(schema_first="clean"):
    con = duckdb.connect(str(DB_PATH), read_only=True)
    con.execute(f"SET search_path = '{schema_first},main'")   # las tablas limpias tapan a las crudas
    return con


# --------------------------------------------------------------------------- features
def _monthly_flows(con):
    cm = pd.DataFrame({"category": list(CAT_MAP), "grp": list(CAT_MAP.values())})
    con.register("cat_map", cm)
    df = con.execute("""
        SELECT t.company_id, CAST(date_trunc('month', t."date") AS DATE) AS month,
          SUM(CASE WHEN c.grp = 'op_in' THEN t.amount ELSE 0 END) AS op_in,
          -SUM(CASE WHEN c.grp = 'op_out' THEN t.amount ELSE 0 END) AS op_out,
          -SUM(CASE WHEN c.grp = 'fin_cost' THEN t.amount ELSE 0 END) AS fin_cost,
          -SUM(CASE WHEN c.grp = 'debt_service' THEN t.amount ELSE 0 END) AS debt_service,
          COUNT(*) AS n_tx
        FROM transactions t LEFT JOIN cat_map c ON t.category = c.category
        WHERE t."date" < TIMESTAMP '2026-09-01'
        GROUP BY 1, 2
    """).df()
    df["month"] = pd.to_datetime(df["month"])
    return df


def _liquidity(con):
    bal = con.execute("""
        SELECT b.product_id, p.company_id, b.balance
        FROM balances b JOIN banking_products p ON b.product_id = p.product_id
        WHERE p.type IN ('checking', 'saving', 'tpv') AND b.balance IS NOT NULL
    """).df()
    con.register("bal_products", bal[["product_id"]])
    flow = con.execute("""
        SELECT product_id, CAST(date_trunc('month', "date") AS DATE) AS month, SUM(amount) AS flow
        FROM transactions WHERE product_id IN (SELECT product_id FROM bal_products) GROUP BY 1, 2
    """).df()
    flow["month"] = pd.to_datetime(flow["month"])
    months = list(MONTHS) + [LAST_M + pd.DateOffset(months=1)]
    grid = pd.MultiIndex.from_product([bal["product_id"].unique(), months], names=["product_id", "month"]).to_frame(index=False)
    grid = grid.merge(flow, on=["product_id", "month"], how="left").fillna({"flow": 0.0})
    grid = grid.merge(bal, on="product_id").sort_values(["product_id", "month"]).reset_index(drop=True)
    g = grid.groupby("product_id")["flow"]
    after = g.transform("sum") - g.cumsum()          # movimientos posteriores al mes
    grid["end_bal"] = grid["balance"] - after
    grid = grid[grid["month"] <= LAST_M]
    return grid.groupby(["company_id", "month"], as_index=False)["end_bal"].sum().rename(columns={"end_bal": "liq"})


def _invoice_features(con):
    k = con.execute("""
        SELECT company_id, CAST(issuance_date AS DATE) AS iss,
               GREATEST(CAST(due_date AS DATE), CAST(issuance_date AS DATE)) AS due,
               CASE WHEN status = 'paid' THEN CAST(payment_date AS DATE) END AS paid_dt,
               amount, counterparty_id
        FROM invoices
        WHERE document_type = 'invoice' AND status <> 'cancel' AND amount <> 0
          AND issuance_date IS NOT NULL AND due_date IS NOT NULL
          AND NOT (status = 'paid' AND NOT (payment_date >= issuance_date AND payment_date <= TIMESTAMP '2026-09-01'))
    """).df()
    for c in ("iss", "due", "paid_dt"):
        k[c] = pd.to_datetime(k[c])
    k["side"] = np.where(k["amount"] > 0, "AR", "AP")
    k["abs_amt"] = k["amount"].abs()
    k["delay_d"] = (k["paid_dt"] - k["due"]).dt.days
    k["w_delay"] = k["delay_d"] * k["abs_amt"]

    out = []
    for i, m in enumerate(MONTHS):
        if i < 2:
            continue
        e = m + pd.offsets.MonthEnd(0)
        w0 = m - pd.DateOffset(months=2)
        o = k[(k["iss"] >= w0) & (k["iss"] <= e) & (k["paid_dt"].isna() | (k["paid_dt"] > e))]
        tot = o.groupby(["company_id", "side"])["abs_amt"].sum().rename("open_amt")
        ov = o[o["due"] < e].groupby(["company_id", "side"])["abs_amt"].sum().rename("ov_amt")
        x = pd.concat([tot, ov], axis=1)
        x["ov_amt"] = x["ov_amt"].fillna(0.0)
        x["od_share"] = x["ov_amt"] / x["open_amt"]
        p = k[k["paid_dt"].notna() & (k["paid_dt"] >= w0) & (k["paid_dt"] <= e)]
        gp = p.groupby(["company_id", "side"])[["w_delay", "abs_amt"]].sum()
        x["delay"] = (gp["w_delay"] / gp["abs_amt"]).clip(-30, 120)
        x = x.reset_index()
        ar = x[x["side"] == "AR"][["company_id", "od_share", "delay"]].rename(columns={"od_share": "ar_overdue", "delay": "delay_coll"})
        ap = x[x["side"] == "AP"][["company_id", "od_share", "delay"]].rename(columns={"od_share": "ap_overdue", "delay": "delay_paid"})
        res = ar.merge(ap, on="company_id", how="outer")
        if i >= 5:
            c = k[(k["side"] == "AR") & (k["iss"] >= m - pd.DateOffset(months=5)) & (k["iss"] <= e) & k["counterparty_id"].notna()]
            a = c.groupby(["company_id", "counterparty_id"])["abs_amt"].sum().groupby("company_id")
            conc = (a.max() / a.sum()).rename("concentration").reset_index()
            res = res.merge(conc, on="company_id", how="outer")
        else:
            res["concentration"] = np.nan
        if i < 6:  # faltan facturas emitidas antes de 2024-09 y pagadas tarde -> sesgo a retrasos cortos
            res[["delay_coll", "delay_paid"]] = np.nan
        res["month"] = m
        out.append(res)
    return pd.concat(out, ignore_index=True)


def build_features(con):
    mth = _monthly_flows(con)
    first = mth.groupby("company_id")["month"].min().rename("first_m")
    grid = pd.MultiIndex.from_product([first.index, MONTHS], names=["company_id", "month"]).to_frame(index=False)
    grid = grid.merge(first, left_on="company_id", right_index=True)
    grid = grid[grid["month"] >= grid["first_m"]]
    P = grid.merge(mth, on=["company_id", "month"], how="left")
    for v in ("op_in", "op_out", "fin_cost", "debt_service", "n_tx"):
        P[v] = P[v].fillna(0.0)
    P = P.sort_values(["company_id", "month"]).reset_index(drop=True)

    g = P.groupby("company_id")
    roll = lambda col, w, f: g[col].transform(lambda s: getattr(s.rolling(w), f)())
    P["in3"] = roll("op_in", 3, "sum"); P["out3"] = roll("op_out", 3, "sum")
    P["fc3"] = roll("fin_cost", 3, "sum"); P["ds3"] = roll("debt_service", 3, "sum")
    P["net_m"] = P["op_in"] - P["op_out"]
    g = P.groupby("company_id")
    P["sd6"] = g["net_m"].transform(lambda s: s.rolling(6).std())
    P["in6"] = g["op_in"].transform(lambda s: s.rolling(6).mean())
    P["in3_l3"] = g["in3"].shift(3)

    in3, out3 = P["in3"], P["out3"]
    P["coverage"] = np.minimum(3, in3 / np.maximum(out3, 1))
    P["net_margin"] = np.where(in3.isna(), np.nan, np.where(in3 > 0, ((in3 - out3) / in3.where(in3 > 0)).clip(-1, 1), -1.0))
    P["growth"] = np.where(P["in3_l3"] > 0, (in3 / P["in3_l3"].where(P["in3_l3"] > 0) - 1).clip(-1, 1), np.nan)
    P["volatility"] = np.minimum(3, P["sd6"] / np.maximum(P["in6"], 1))
    P["fin_cost_r"] = (P["fc3"] / np.maximum(in3, 1)).clip(0, 1)
    P["debt_serv_r"] = (P["ds3"] / np.maximum(in3, 1)).clip(0, 2)

    P = P.merge(_liquidity(con), on=["company_id", "month"], how="left")
    P = P.sort_values(["company_id", "month"]).reset_index(drop=True)
    P["runway"] = (P["liq"] / np.maximum(P["out3"] / 3, 1)).clip(-6, 24)
    g = P.groupby("company_id")
    P["d_runway"] = (P["runway"] - g["runway"].shift(3)).clip(-12, 12)
    neg = pd.Series(np.where(P["liq"].isna(), np.nan, (P["liq"] < 0).astype(float)), index=P.index)
    P["neg_liq"] = neg.groupby(P["company_id"]).transform(lambda s: s.rolling(3).mean())

    P = P.merge(_invoice_features(con), on=["company_id", "month"], how="left")
    return P.sort_values(["company_id", "month"]).reset_index(drop=True)


# --------------------------------------------------------------------------- scoring
def fit_ref(P, sig=SIG, max_n=20000, seed=2026):
    rng = np.random.default_rng(seed)
    ref = {}
    for s in sig["signal"]:
        v = P[s].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if len(v) > max_n:
            v = rng.choice(v, max_n, replace=False)
        ref[s] = np.sort(v)
    return ref


def run_score(P, ref, sig=SIG, pw=PILLAR_W):
    L = P[["company_id", "month"] + list(sig["signal"])].melt(id_vars=["company_id", "month"], var_name="signal", value_name="value")
    L = L[np.isfinite(L["value"])].merge(sig, on="signal")
    L["sig_score"] = np.nan
    for s in sig.loc[sig["type"] == "pct", "signal"]:
        m = (L["signal"] == s).to_numpy()
        r = ref[s]
        v = L.loc[m, "value"].to_numpy()
        pct = (np.searchsorted(r, v, "left") + np.searchsorted(r, v, "right")) / 2 / len(r)
        L.loc[m, "sig_score"] = 100 * np.where(L.loc[m, "direction"].to_numpy() == 1, pct, 1 - pct)
    sh = L["type"] == "share"
    L.loc[sh, "sig_score"] = 100 * (1 - L.loc[sh, "value"])

    Pl = L.groupby(["company_id", "month", "pillar"], as_index=False).agg(pillar_score=("sig_score", "mean"), n_sig=("sig_score", "size"))
    Pl["w"] = Pl["pillar"].map(pw)
    Pl["w_eff"] = Pl["w"] / Pl.groupby(["company_id", "month"])["w"].transform("sum")
    Pl["ws"] = Pl["w_eff"] * Pl["pillar_score"]
    Sc = Pl.groupby(["company_id", "month"], as_index=False).agg(score=("ws", "sum"), cov_w=("w", "sum"))
    Sc = Sc[Sc["cov_w"] >= 0.5].reset_index(drop=True)
    L = L.merge(Pl[["company_id", "month", "pillar", "n_sig", "w_eff"]], on=["company_id", "month", "pillar"])
    L["contrib"] = L["w_eff"] / L["n_sig"] * L["sig_score"]
    Pl["mi"] = Pl["month"].map(MI)
    return {"Sc": Sc, "L": L, "Pl": Pl}


def _slope(y):
    ok = ~np.isnan(y)
    if ok.sum() < 4:
        return np.nan
    x = np.arange(len(y))[ok].astype(float)
    yy = y[ok]
    return ((x - x.mean()) * (yy - yy.mean())).sum() / ((x - x.mean()) ** 2).sum()


def make_traj(Sc):
    grid = pd.MultiIndex.from_product([Sc["company_id"].unique(), MONTHS], names=["company_id", "month"]).to_frame(index=False)
    T = grid.merge(Sc, on=["company_id", "month"], how="left").sort_values(["company_id", "month"]).reset_index(drop=True)
    T["mi"] = T["month"].map(MI)
    g = T.groupby("company_id")["score"]
    T["delta3"] = T["score"] - g.shift(3)
    T["trend6"] = 5 * g.transform(lambda s: s.rolling(6, min_periods=4).apply(_slope, raw=True))
    T["base6"] = g.transform(lambda s: s.rolling(6, min_periods=3).median()).groupby(T["company_id"]).shift(1)
    T["traj_component"] = (50 + 2.5 * T["trend6"]).clip(0, 100)
    T["traj_score"] = np.where(T["trend6"].isna(), T["score"], 0.75 * T["score"] + 0.25 * T["traj_component"])
    s, t = T["score"], T["trend6"]
    T["state"] = np.select(
        [(s >= 65) & (t.isna() | (t > -8)), t >= 8, (s < 40) | ((t <= -8) & (s < 50)), t <= -8],
        ["Sólida", "Mejorando", "Deteriorada", "Empezando a torcerse"], default="Estable").astype(object)
    T.loc[s.isna(), "state"] = None
    return T


def pillar_moves(Pl):
    lag = Pl[["company_id", "pillar", "mi", "pillar_score"]].copy()
    lag["mi"] += 3
    Pd = Pl[["company_id", "pillar", "mi", "pillar_score"]].merge(lag, on=["company_id", "pillar", "mi"], suffixes=("", "_lag"))
    Pd["d3"] = Pd["pillar_score"] - Pd["pillar_score_lag"]
    return Pd


def worst_pillar(Pl):
    Pd = pillar_moves(Pl)
    idx = Pd.groupby(["company_id", "mi"])["d3"].idxmin()
    w = Pd.loc[idx, ["company_id", "mi", "pillar", "d3"]].rename(columns={"pillar": "worst_pillar", "d3": "worst_pillar_d3"})
    n = Pd.assign(dn=(Pd["d3"] <= -8).astype(int)).groupby(["company_id", "mi"], as_index=False)["dn"].sum().rename(columns={"dn": "n_down"})
    return w.merge(n, on=["company_id", "mi"])


def alerts(T, worst):
    A = T[["company_id", "month", "mi", "score", "delta3", "state"]].merge(worst, on=["company_id", "mi"], how="left")
    A = A.sort_values(["company_id", "mi"]).reset_index(drop=True)
    prev = A.groupby("company_id")["state"].shift(1)
    A["alert"] = ((A["delta3"] <= -10) | ((A["state"] == "Deteriorada") & (prev != "Deteriorada") & prev.notna())
                  | (A["worst_pillar_d3"] <= -25)).fillna(False)
    a = A["alert"].astype(int)
    prior = a.groupby(A["company_id"]).shift(1, fill_value=0) | a.groupby(A["company_id"]).shift(2, fill_value=0)
    A["alert_new"] = A["alert"] & ~prior.astype(bool)
    return A


def auc(score, y):
    d = pd.DataFrame({"s": score, "y": y}).dropna()
    n1, n0 = int((d["y"] == 1).sum()), int((d["y"] == 0).sum())
    if n1 == 0 or n0 == 0:
        return np.nan
    r = d["s"].rank()
    return (r[d["y"] == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
