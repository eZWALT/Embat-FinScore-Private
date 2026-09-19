# Wave 4 — FX invoice share (`e_fx_share`)

Agent `97d3db33`. Train / seed 20260918. Holdout 72 coverage only.
Owned: `analysis/evaluate/fx_qa.py`, `analysis/outputs/fx_qa.md`,
`analysis/outputs/fx_y3_quintiles.png`, append-only registry.
Did not edit invoices.py, parquet, product/, gbm_y7*, y2_why, uncat, score_pipeline.

## Quote

- Ever-FX **228** train ERP (fx>0 on 16.8% of 13,554 ERP CM; 20.4% of defined).
  Dark 470: FX all-NaN. Feature-report 52.8% / 79.6% modal CONFIRMED.
- Size ρ vs log1p(a_in3) **0.060** (vs e_ar_issued 0.122) — not SIZE.
- Y3 single **0.528** vs same-mask size **0.628** (Δ −0.100) vs days-full **0.711**
  (night replica). Gate fails. Live rebuild on the store panel matches (0.528).
- Y7 descriptive (not X): top1_lost 30.5% vs 26.8%. Y5 AP/AR slightly *lower* on FX.

## Verdict

| object | call |
| --- | --- |
| health Y | **PARK** — do not invent a merged FX Y |
| Y7 / Y5 X | **CLOSE** — forbidden family E |
| Y3 X / 44-col list | **CLOSE** — drop `e_fx_share` from the starter 44 |
| Y2 X | **CLOSE** — 0.509 vs size 0.565 |
| Q6 lag1 | **CLOSE** — 0.535 vs size 0.627; issued_lag1 stays the Y7 KEEP |
| Q5 footnote | **KEEP footnote** — mixed import/export (44% AR / 56% AP), foreign-home identity, not the 110 |

## What the later cuts added

- Foreign ≠ monopoly (HHI ρ −0.133; fx>0 is 26% in Q1 vs 12% in Q5).
- Not the mixed-group 110 dark siblings (those stay NaN). Mixed invoiced 17.9% vs all-invoiced 16.2%.
- Style vs shock: 97 always / 131 shock / 515 never; acf1 −0.019. Within shock companies the FX *month* is not the recoverer (5.6% vs own 6.8%). Always-FX is 46% non-EUR home.
- 4 raw-only names (COMP_0510/0708/1159/1194): FX invoices sit before first cash-trail month. Live `invoices.build` on a 24-month grid lights 232; on-panel overlap matches the store (228, Δ=0). 215 off-panel FX months / 53 companies with first FX before panel. Do **not** rewrite parquet.
- SHAP rank ~69, mean\|SHAP\| 0.057 — already quiet on the 44.

## Failed / next

- High-intensity (≥0.08) Y3 13.9% vs 3.8% but CV 0.539, n_pos=27. Do not KEEP an intensity flag.
- HHI Q4/Q5 × FX recover 16–19% is 18 positives — thin, no interaction X.
- Next (not this owner): drop `e_fx_share` from `gbm_core` CORE list when someone re-cards the 44. Not a parquet rewrite.
