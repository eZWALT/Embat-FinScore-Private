# Wave 4 — Y2 why (end note)

Agent `1bb2643e`. Lane: `analysis/evaluate/y2_why.py`. No commit. No parquet/duckdb rewrite. No `product/`. No 0–100. No GBM/XGB. Did not run `build_targets`. Never B as X. Did not edit y5_why, y9_why, balances_b_qa, y2_stress, liquidity.

## Files written

- `analysis/evaluate/y2_why.py` — create; write→run cuts 1–29 in this module
- `analysis/outputs/y2_why.md` — tables + CLOSE
- `analysis/outputs/y2_why_quintiles.png` — `c_n_days_with_tx` quintiles vs Y2 rate
- `overnight/waves/wave4_y2_why.md` — this note
- `analysis/experiments/registry.csv` — append-only (agent `1bb2643e`, first full run)

## Columns / coverage (train)

Y `y2_neg_2of3` from `targets.parquet` only. Train labeled **17356 / 1271 / 7.32%**. Holdout 857/23/72 LOW_POWER. Legal X: `a_io_ratio` `a_out6` `log1p(a_in3)` `c_n_days_with_tx` `d_cust_hhi` `f_ds_r` `c_zero_in_month`. Family B decomp/leak only.

## What failed

- **Same crash as Y4:** no. Y2pos crash=35.4% (med in-ratio 1.03 vs Y4's 0.36). Y2∩Y4=12.7% (Y4 defined on 19% of Y2 pos). Y4∩Y2=9.5%. Jaccard 0.057. CLOSE, do not merge.
- **Q5 KEEP (non-B):** no. Best legal `c_n_days_with_tx` CV **0.571** vs size **0.540** / night GBM 0.540. Gap +0.031 but shape flat (Q4 bump). Body ≤17 CV 0.426. Drop 0158+0172 → days **0.557** (gap +0.017, below KEEP). Rest-only best is `d_cust_hhi` 0.564 vs size 0.555 (+0.009), fold 2 = 0.325.
- **Q6:** lag1 days 0.575, lag_lift +0.004. CLOSE. Do not claim beyond 1-month.
- **Leak:** 0 B-copies. Max |ρ| `a_out6` vs `b_runway` −0.432.
- **Trees:** stay PARK. Fold 0 = 0.344 is GROUP_0158/0172 (all-dark, 21-cos, train-only), not late-trail. trail_length.md already has Y2 short 7.9% vs long 6.2%.
- **Dark 9.14% vs 6.34%:** confirm. All-dark minus 0158/0172 is **5.5%** (below invoiced). Not a new Y. Do not rename.

## Fold 0

Rate 11.1% vs rest 6.3%. T3 (group_size>13) 17.3% vs fold1 T3 0.2%. 0158: 34.5% Y2, late=0, 40.6% of fold-0 pos, already-neg 84%, clean onset 9.8% and *later* on book (12–17: 15.6%). 0172: 22%, late, clean onset *early* (8.4% <6). Drop the 4 T3 groups: fold-0 rate → 6.65%; group_size CV 0.375 → 0.570. Night tree learned large≈safe on folds 1–4.

## Q3/Q5 sentence (no family B)

Y2 is already-negative persistence (82% of positives are below 0 at t; clean-now leftover 1.5%). The 82→68 *direction* is real; a non-B why is not.

## Next idea (parent)

Do not fit another tree. Do not merge with Y4. Do not rename via Y11. Optional: a *company* persistence card (how long already-neg) — that is still B-derived, so it is a footnote, not an X.

---

## End note (cuts 17–30)

360/110 confirm (9.49% / 8.17% / invoiced 6.34%). Y2 acf1 pooled +0.880 / company-median +0.686. All-dark minus 0158/0172 = **5.53%**. 0158/0172 already-neg Y2 85% vs rest 69%; clean-in-hot 7.8% vs 1.2%. 0158 onset is *late* on book (12–17: 15.6%); 0172 onset is *early* (<6: 8.4%). Neither cluster is Y4 (0158 crash 33%, med in 1.11; 0172 Y4 share 0). Days CV without them 0.557. 0158 is not 21 chronic names: 14/21 ever Y2, 6 ≥50% below 0, 8 never-below. 0172: 6 chronic / 15 never-below. Combined **12 names hold 47% of fold-0 positives**. Drop those 12: days CV 0.571→**0.549** vs size 0.544 (gap +0.005). The 0.571 single *is* those names. Fold-0 is a handful of busy, chronically-negative dark names sitting in a large group the tree treated as safe. Cut 35: fold-0 without the 12 is **6.22%** vs rest 6.42%; the 12 names themselves are **81.9%** Y2 (216 months / 177 pos).
