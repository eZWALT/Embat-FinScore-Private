# Wave 4 — XGB Y4 `y4_ds_r_double` (never F)

- **When:** 2026-09-19 00:52–01:22 CEST
- **Agent:** `f248c993`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Holdout:** 72 companies, seed 20260918. Never fitted.
- **Re-run:** `python -m analysis.models.xgb_y4 --variant xgb_d3_n50`

Y4 is brief **Q3 (turning)** / **Q5 (why)**: `ds_r` at t+3 ≥ 2× `ds_r` at t, both defined, `ds_r_t > 0.05`. Not bankruptcy. Not a 0–100.

## Files written

- `analysis/models/xgb_y4.py` (create — did not append into `xgb_panel.py`)
- `analysis/outputs/y4_xgb.md`
- this note
- append-only rows on `analysis/experiments/registry.csv` (agent `f248c993`, R4 / wave 4)

Did not edit `xgb_panel.py`, `gbm_y7_core.py`, `gbm_y1.py`, `match.py`, `catmix.py`, parquet, `product/`. No commit. No 0–100.

## Setup

- **Y:** `y4_ds_r_double` only. Re-checked `data/feature_store/y_acceptance.csv`: still the only accepted Y4 (13.88%). `y4_ds_r_gt05_sust` / `y4_new_facility_after_dip` / `y4_ogtg_appear` still rejected — not revived.
- **X:** families A B C D E G H. **Never F.** Dropped `a_fin_cost*`, `a_debt*` (ds_r numerator — would copy the label), and `m_debt*` if present (not in the store; catmix not merged).
- **Lags:** 1 and 3, past-only.
- **Split:** 5 `group_folds`, seed 20260918. 72 holdout companies out of every fit and sign pick.
- **Train:** 2,370 labeled / 329 pos / 13.88% / grid coverage 0.112.
- **KEEP bar:** dummy 0.50 and best *fixed* OOF single + 0.02. Trees must not collapse to 1–10. Quote CV, not holdout.

## Screens

- Leak vs `f_ds_r` / lags on train labeled: max \|ρ\| = **0.411** (`c_n_days_with_tx`). Pass (< 0.80).
- Size `log1p(|a_in3|)` AUROC = **0.493**. Pass (< 0.60).

## Table (the quote)

| variant | CV ± sd | vs dummy | vs single | n_x | collapsed | decision |
|---|---:|---:|---|---:|---|---|
| xgb_es 400+ES depth-4 | **0.585 ± 0.039** | +0.085 | `d_cust_hhi_lag3` 0.605 (**−0.020**) | 241 | **yes** (min 7) | **PARK** |
| xgb_d3_n50 | **0.561 ± 0.050** | +0.061 | 0.605 (**−0.044**) | 241 | no | **PARK** |
| lgb_d3_n50 | **0.540 ± 0.043** | +0.040 | 0.605 (**−0.065**) | 241 | no | **PARK** |
| xgb_d3_n50_nosize | 0.551 ± 0.048 | +0.051 | 0.605 (−0.054) | 199 | no | **PARK** |
| lgb_d3_n50_nosize | 0.544 ± 0.041 | +0.044 | 0.605 (−0.061) | 190 | no | **PARK** |
| xgb_d3_n50_ratios | 0.509 ± 0.063 | +0.009 | 0.605 (−0.095) | 146 | no | **PARK** |
| xgb_d3_n50_d | 0.492 ± 0.039 | **−0.008** | 0.605 (−0.113) | 27 | no | **PARK** |
| xgb_d3_n50_d2 | 0.515 ± 0.028 | +0.015 | 0.605 (−0.090) | 6 | no | **PARK** |
| d_zavg (not a tree) | 0.634 ± 0.076 | +0.134 | 0.605 (**+0.029**) | 2 | no | **PARK** (next idea) |

Holdout check (not a claim): **n_pos=16** LOW_POWER, XGB AUROC **0.467** (inverts). Same pattern as Y5 XGB.

Complete-case on `d_cust_hhi_lag3` defined rows: XGB 0.581 vs single 0.605.

## Verdict

**PARK.** No tree beats the best single by 0.02. 400+ES collapsed (do not KEEP that tree count). Early registry KEEP rows vs a noisy train-pick of 0.508 are superseded by the 00:58+ rows that use the fixed OOF single.

The label stays accepted. The honest signal is **`d_cust_hhi_lag3` (+)** — concentrated customers three months earlier. Neighbour `d_n_supp_lag3` (−). Not inflow size. Not a ds_r leak. OOF lead: HHI 0.583 / 0.603 / **0.605** at t / t−1 / t−3 (Q6). HHI is sticky (acf1 0.95) but company-median HHI is only 0.545 vs month Y — lag3 is not just a company type.

Label mechanism (train only, from the Y4 flow panel — not used as X): positives have median `in3[t+3]/in3[t] = 0.36` vs 1.10 for negatives, and median `ds3` ratio 1.15 vs 0.98. **80%** of positives are an inflow drop >20%; 49% are a debt-service rise >20%. HHI is the leading why of a future denominator crash, not a repayment spike.

SHAP names (train only, d3_n50): `a_growth_3`, `group_size`, `a_growth_3_lag1`, `a_n_tx`, `a_growth_12_lag3`, `a_out3`, `h_share_group_in_lag3`, `c_n_tx`, `d_n_cust`, `a_n_tx_lag1`. Growth / activity / group scale — not debt-service pressure.

## What failed

- Per-fold train-best pick averaged 0.508 because fold 0 picked `e_ap_open` (val 0.445). That is not “best single.”
- D-only XGB scored **below dummy** — trees wash out HHI.
- 2-stem XGB (`xgb_d3_n50_d2`, 6 cols): CV **0.515** (loses to the z-average 0.634 and to the single 0.605). Trees are the wrong tool on this Y.
- Ratios-only (no euro SIZE, no counts) is a coin flip.

## Next idea

`--variant d_zavg` in `xgb_y4.py` reproduces a train-only signed z-average of `d_cust_hhi_lag3` (+) and `d_n_supp_lag3` (−): CV **0.634 ± 0.076** (gap +0.029 vs 0.605). Fold 2 is 0.518. Holdout **0.368** on n_pos=16 because those two D lags are defined on only 21 holdout labeled rows (2 pos). Next idea if someone wants a Y4 *signal* — still no F, still no 0–100, still not holdout-ready. Do not retune XGB on this bake-off.

## Y4 why / HHI clock (01:25–01:32) — agent `79044b5e`

Long-lived child on `analysis/evaluate/y4_why.py`. Did not edit `xgb_y4.py`, parquet, `product/`. No 0–100. No new GBM. No commit.

**Re-run:** `python -m analysis.evaluate.y4_why`

Holdout 72 out of every rate, AUROC, and cut. Seed 20260918. Never F. in3/ds3 rebuilt from transactions + CAT_MAP inside the module (QA vs `a_in3`: max|Δ|=0).

### Return

| item | number |
|---|---|
| crash share of positives (`in3` ratio < 0.8) | **79.6%** (262/329) |
| spike share (`ds3` ratio > 1.5) | **34.3%** (113/329) |
| both | **15.2%** |
| neither | **0** (1.5/0.8 < 2 — algebra, not a discovery) |
| best HHI lag (mean CV) | lag **6** 0.6052 — **not usable** (sd 0.175, train 0.533, cov 26%) |
| stable HHI lag | **lag 3 CV 0.605 ± 0.044** (cov 35.8%, 848/2370) |
| quintiles P(Y=1) | 8.2 / 15.4 / 10.6 / 11.8 / **22.4%** (not monotone) |
| z-avg card | **CLOSE** — CV 0.634 vs HHI-CC 0.605 (+0.029) but vs n_supp-CC 0.631 (+0.003) |

### Passes

1. **Decompose.** Train 2370 / 329 / 13.88%. Pos med in-ratio 0.36, ds-ratio 1.15 (known). Crash-only 64% of pos, Y rate 31.9%; spike-only 19%, Y rate 26.4%; both 15%, Y rate **98%**. Size AUROC inside slices < 0.60 (crash 0.386 inverse). Holdout 16 pos **flip** the mix (crash 38% / spike 88% / med in-ratio 0.88).
2. **Q6 clock.** HHI / top1 at lags 0,1,3,6. Lag 3 is **not** the mean-CV peak (lag 6 +0.0005). Quote lag 3. Invoice-gate: train CM HHI **42.2%** (not ~53%); labeled 44.2%; lag3 35.8%. On crash rows HHI OOF **0.639**; on spike_any **0.524**. Crash-slice clock mean-CV peak is lag 1 (0.663, sd 0.139); stable crash clock still lag 3 (0.639 ± 0.070).
3. **Quintiles.** Train-labeled cuts. Not monotone. **The 0.605 is the monopoly tail:** HHI>0.975 P(Y=1)=22.1% (172 months / **43 companies**) vs rest 11.5%. Body HHI≤0.975 CV **0.445** (loses to dummy). Those 43 companies on their *other* labeled months: P(Y=1)=**15.3%** (242 / 37) — partly a type, still a month-level jump to 22.1%. Crash-only quintiles: 25 / 26 / 31 / 31 / **54%**.
4. **2-col honesty.** z-avg 0.634 ± 0.076, signs +HHI / −n_supp from train folds. HHI and n_supp are **not** the same invoice gate (848 vs 986). z-avg is n_supp on the 848 HHI rows (n_supp-CC 0.631). n_supp without HHI: 138 rows, Y rate 18.8% — dropped by the card. **CLOSE**, not KEEP. Fold 2 still 0.518. Holdout HHI 0.526 / n_supp 0.087 on 2 pos.
5. **Leak.** HHI_lag3 vs `f_ds_r` ρ_s=0.161 / ρ_p=0.234; vs `a_in3` −0.293 / −0.134; vs `d_cust_top1_lag3` **0.991 / 0.979**. **Top-1 rewrite.** Y7 forbids D for that reason; Y4 allowed D — do not stack.

### Q3 / Q5 / Q6

Y4 doubling is who is *turning* (Q3). On train it is mostly a denominator crash (Q5). HHI three months earlier is the lead (Q6) **only as a near-monopoly flag**, not a concentration gradient, and only for crash-type doubles — holdout’s 16 events are spike-led so HHI should invert. Not bankruptcy. Not a 0–100.

### Files

- `analysis/evaluate/y4_why.py` (create)
- `analysis/outputs/y4_why.md`
- `analysis/outputs/y4_hhi_quintiles.png`
- append-only `analysis/experiments/registry.csv` (agent `79044b5e`, train-only)

### What failed / next

Lag 6 and the 2-col card both look like wins on the mean and are not. Do not KEEP z-avg. Do not write a gradient story from the quintiles. Binary `d_cust_hhi_lag3 > 0.975` OOF **0.578** (weaker than continuous 0.605) — the 0.605 *is* that tail, but the continuous rank still wins the number. Still never F, still quote CV, still LOW_POWER holdout.
