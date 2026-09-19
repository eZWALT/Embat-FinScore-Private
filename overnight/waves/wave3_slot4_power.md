# Wave 3 · slot 4 — holdout AUROC power

- **Files:** `analysis/evaluate/holdout_power.py` (owned). This note. 30 registry rows appended (`model=holdout_power`).
- **Not edited:** `analysis/models/baselines.py`, other agents' files, `product/`.
- **Re-run:** `python -m analysis.evaluate.holdout_power` (idempotent registry keys: agent, y, model, split, metric).
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Holdout:** 72 companies / 1,073 company-months. Never used to fit anything (there is no fit).
- **Y source:** `data/feature_store/targets.parquet` (same `build()` columns). Fallback is each module `build()`.

Accepted binaries checked: `y2_neg_2of3`, `y5_ap_od30_ownp80`, `y5_ar_od30_sust`, `y4_ds_r_double`, `y3_recover_cash_6m`, `y6_missed_payroll`.

## Method

Labeled rows only (Y not null). Train = not in `analysis/splits/holdout_companies.csv`.

Bootstrap: 4,000 parametric draws of binormal scores with **true AUC = 0.65**, `n_pos` = holdout positives, `n_neg` = holdout negatives. 95% CI = 2.5–97.5 percentile of the AUROC estimator. Seed 20260918. Hanley–McNeil analytic width is the sanity check (same n_pos / n_neg / 0.65).

`LOW_POWER` = holdout positives < 30.

## Table

| y | train n / pos / rate | hold n / pos / rate | hold labeled cos | hold pos cos | AUC=0.65 boot 95% CI | width | analytic | flag |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `y2_neg_2of3` | 17,356 / 1,271 / 7.3% | 857 / **23** / 2.7% | 72 | 6 | [0.541, 0.760] | 0.219 | 0.247 | **LOW_POWER** |
| `y5_ap_od30_ownp80` | 4,905 / 418 / 8.5% | 195 / **14** / 7.2% | 28 | 8 | [0.495, 0.794] | 0.299 | 0.322 | **LOW_POWER** |
| `y5_ar_od30_sust` | 3,315 / 237 / 7.1% | 165 / **8** / 4.8% | 23 | 5 | [0.446, 0.835] | 0.389 | 0.423 | **LOW_POWER** |
| `y4_ds_r_double` | 2,370 / 329 / 13.9% | 135 / **16** / 11.9% | 25 | 9 | [0.505, 0.791] | 0.286 | 0.307 | **LOW_POWER** |
| `y3_recover_cash_6m` | 5,648 / 402 / 7.1% | 235 / **14** / 6.0% | 33 | 7 | [0.496, 0.790] | 0.294 | 0.321 | **LOW_POWER** |
| `y6_missed_payroll` | 6,004 / 367 / 6.1% | 271 / **7** / 2.6% | 25 | 2 | [0.433, 0.838] | 0.405 | 0.448 | **LOW_POWER** |

y2 holdout 857 / 23 / 2.68% matches baselines. Train rates match `y_acceptance.csv`.

## Recommendation — none usable for holdout model claims tonight

Every accepted binary is **LOW_POWER**. Do not publish holdout AUROC as a bake-off result. A true 0.65 model has a 95% CI **0.22–0.41 wide**; several intervals include 0.50, so holdout cannot tell 0.65 from chance, let alone rank two models that differ by 0.05.

Worse than the raw n_pos: events sit in **2–9 holdout companies**. Protocol bootstrap-over-companies would be wider still. `y6_missed_payroll` is 7 events in 2 companies.

Holdout prevalence also shifts down vs train (y2 7.3% → 2.7%; y6 6.1% → 2.6%; y5_ar 7.1% → 4.8%, under the 5% floor). Javier −score 0.776 uses family B on the same 23-event y2 holdout — not a fair no-B baseline, and not a powered one.

**Tonight:** train group-fold CV only. Train positives are adequate (237–1,271). Prefer `y2_neg_2of3` (1,271 pos / 170 cos) then `y5_ap_od30_ownp80` (418) / `y3_recover_cash_6m` (402) / `y6_missed_payroll` (367) / `y4_ds_r_double` (329). Treat any holdout AUROC as a noisy diagnostic.

## What failed

- No accepted Y has holdout pos ≥ 30.
- y2: 23 pos / 6 companies, rate 2.7%. Explains noisy single-feature AUROCs (0.86 sibling count vs 0.42 train-best).
- y6: 7 pos / 2 companies. Dead for holdout.
- y5_ar: 8 pos and holdout rate 4.8% (below acceptance floor).
- Company-month iid CI is already too wide; company-clustered CI is not even worth quoting.

## Next idea

Do not retune these Ys to chase holdout events. If a holdout claim is required later, add a higher-rate sustained binary or pool a few close Ys — and report company-clustered CIs on train CV, not a 23-event holdout AUROC.
