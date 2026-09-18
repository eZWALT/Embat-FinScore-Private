"""Runs the Python score pipeline end to end and prints the validation diagnostics."""
import sys
import time

import numpy as np
import pandas as pd

import score_pipeline as sp

pd.set_option("display.width", 200, "display.max_columns", 30)


def spearman(a, b):
    return a.rank().corr(b.rank())

t0 = time.time()
con = sp.connect()

# ------------------------------------------------------------------ features
P = sp.build_features(con)
print(f"[{time.time()-t0:5.1f}s] panel: {len(P):,} filas, {P.company_id.nunique()} empresas")
P.to_pickle(sys.argv[1] if len(sys.argv) > 1 else "panel.pkl")

print("\n== cobertura de senales (% no nulo) ==")
print((P[list(sp.SIG.signal)].notna().mean() * 100).round(1).to_string())

# ------------------------------------------------------------------ comprobacion de signo de facturas
chk = con.execute("""
  WITH i AS (SELECT company_id, SUM(CASE WHEN amount>0 THEN amount END) ar, SUM(CASE WHEN amount<0 THEN -amount END) ap
             FROM invoices WHERE document_type='invoice' AND status='paid' GROUP BY 1)
  SELECT * FROM i""").df()
cash = P.groupby("company_id")[["op_in", "op_out"]].sum().reset_index()
chk = chk.merge(cash, on="company_id")
print("\n== signo facturas: spearman ==")
print({"AR~cobros": round(spearman(chk.ar, chk.op_in), 3), "AP~cobros": round(spearman(chk.ap, chk.op_in), 3),
       "AP~pagos": round(spearman(chk.ap, chk.op_out), 3), "AR~pagos": round(spearman(chk.ar, chk.op_out), 3)})

print("\n== liquidez reconstruida ==")
print({"pct_meses_con_liq": round(P.liq.notna().mean(), 3), "pct_liq_negativa": round((P.liq.dropna() < 0).mean(), 3)})

# ------------------------------------------------------------------ score
ref = sp.fit_ref(P)
res = sp.run_score(P, ref)
Sc = res["Sc"]
chk2 = res["L"].groupby(["company_id", "month"])["contrib"].sum().reset_index().merge(Sc, on=["company_id", "month"])
print("\nmax |suma contribuciones - score| =", float((chk2.contrib - chk2.score).abs().max()))
print("\n== distribucion del score ==")
print(Sc.score.describe().round(1).to_string())
med = Sc.groupby("month").score.median()
print("mediana por mes (deriva temporal?):", med.round(1).iloc[::3].to_dict())

T = sp.make_traj(Sc)
print("\n== estados ultimo mes ==")
print(T[T.month == sp.LAST_M].state.value_counts(dropna=False).to_string())

# ------------------------------------------------------------------ estabilidad por grupo
comp = con.execute("SELECT company_id, group_id FROM companies").df()
rng = np.random.default_rng(42)
grp = comp.group_id.unique()
fold_of = dict(zip(grp, rng.permutation(np.tile(np.arange(5), len(grp) // 5 + 1)[: len(grp)])))
comp["fold"] = comp.group_id.map(fold_of)
Pf = P.merge(comp[["company_id", "fold"]], on="company_id")
rows = []
for k in range(5):
    rk = sp.fit_ref(Pf[Pf.fold != k])
    sk = sp.run_score(Pf[Pf.fold == k], rk)["Sc"].merge(Sc, on=["company_id", "month"], suffixes=("_ho", "_full"))
    rows.append((k, sk.company_id.nunique(), round(spearman(sk.score_ho, sk.score_full), 4), round((sk.score_ho - sk.score_full).abs().mean(), 2)))
print("\n== estabilidad por grupo (fold, empresas, spearman, MAE) ==")
for r in rows: print(r)

# ------------------------------------------------------------------ eventos proxy y AUC
F = P[["company_id", "month", "in3", "in3_l3", "growth"]].copy()
F["in3_f6"] = F.groupby("company_id")["in3"].shift(-6)
floor = P["in3"].quantile(0.10)
ok = F["in3"] >= floor
F["crash"] = np.where(F["in3_f6"].isna(), np.nan, (ok & (F["in3_f6"] <= 0.6 * F["in3"])).astype(float))
F["boom"] = np.where(F["in3_f6"].isna(), np.nan, (ok & (F["in3_f6"] >= 1.5 * F["in3"])).astype(float))
Ev = T[["company_id", "month", "score", "trend6", "delta3", "traj_score"]].merge(F[["company_id", "month", "crash", "boom", "growth"]], on=["company_id", "month"])
Ev = Ev[Ev.score.notna() & Ev.crash.notna()]
print(f"\n== eventos: {len(Ev):,} empresa-mes | tasa caida {Ev.crash.mean():.3f} | tasa subida {Ev.boom.mean():.3f}")
tab = []
for pred in ("score", "trend6", "delta3", "traj_score", "growth"):
    tab.append((pred, round(sp.auc(-Ev[pred], Ev.crash), 3), round(sp.auc(Ev[pred], Ev.boom), 3)))
print(pd.DataFrame(tab, columns=["predictor", "AUC caida (-x)", "AUC subida (+x)"]).to_string(index=False))

# ------------------------------------------------------------------ anticipacion (score sin 'growth')
sig_ex = sp.SIG[sp.SIG.signal != "growth"]
res_ex = sp.run_score(P, ref, sig_ex)
T_ex = sp.make_traj(res_ex["Sc"])
A = sp.alerts(T_ex, sp.worst_pillar(res_ex["Pl"]))
onset = F[F.in3_l3.notna() & (F.in3_l3 >= floor) & (F.in3 <= 0.6 * F.in3_l3)].groupby("company_id").month.min().rename("E").reset_index()
cand = onset.merge(A[A.alert_new][["company_id", "month"]].rename(columns={"month": "A"}), on="company_id")
cand = cand[(cand.A <= cand.E) & (cand.A >= cand.E - pd.DateOffset(months=6))]
lead = cand.groupby(["company_id", "E"], as_index=False).A.min()
lead["lead_m"] = lead.E.map(sp.MI) - lead.A.map(sp.MI)
print(f"\n== anticipacion ==\nempresas con caida de actividad: {len(onset)}")
print(f"con alerta en los 6 meses previos (o mismo mes): {len(lead)} ({100*len(lead)/max(len(onset),1):.0f}%)")
print(f"con alerta >=1 mes antes: {(lead.lead_m>=1).sum()} ({100*(lead.lead_m>=1).sum()/max(len(onset),1):.0f}%) | antelacion mediana: {lead.lead_m.median()}")
Al = A[A.score.notna()][["company_id", "month", "alert_new"]].merge(onset, on="company_id", how="left")
Al["crash6"] = Al.E.notna() & (Al.E > Al.month) & (Al.E <= Al.month + pd.DateOffset(months=6))
base = Al.crash6.mean()
print("tasa de caida a 6m tras alerta vs sin alerta (base %.3f):" % base)
print(Al.groupby("alert_new").crash6.mean().round(3).to_string())
print(f"\nTotal {time.time()-t0:.0f}s")
