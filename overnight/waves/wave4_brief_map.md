# Wave 4 — brief map (six-question compile)

Agent `e4a91c7b`. Train group-fold seed 20260918. Holdout 72 coverage only. No fit. No parquet rewrite.

## Files written

- `analysis/evaluate/brief_map.py` (compile / check only)
- `analysis/outputs/brief_map.md` (morning-quotable map)
- `analysis/outputs/brief_map_grid.png` (2×3 six-question grid)
- append-only `analysis/experiments/registry.csv` (compile rows; skip key agent+x_families+y+model+split+metric)
- this note

Did not touch `dso_qa.py`, `b_on_44_qa.py`, `supp_hhi_qa.py`, parquet, duckdb, `build_targets`, `product/`, the 15-col Y3 card, LIVE.json, CONTEXT.md, canvas, parent journal, or sibling QA scripts. Night quotes unchanged.

## Locked night quotes (do not change)

- Y3 shallow 278 **0.762 ± 0.016**. 15-col A **0.752 ± 0.037**. Days **0.711**. Size **0.617**.
- Y7 TURNOVER **0.720** / B_shallow **0.712**. Do not quote 278-col 0.663 or holdout 0.680. Do not quote a_out_vol 0.722 as the engine.
- CN leftover after issued_lag1 **KEEP 0.597** — do not grow TURNOVER.
- delay_coll leftover after DSO **KEEP 0.581** — do not grow TURNOVER.

## What the map says (six sentences)

1. Healthy is last-value `b_runway` (ρ 0.966; persist 0.85), not `companies.csv` (0.480), not `created_at` (73.6%).
2. Improving is thin `y1_in_h3` only (0.817 vs hist 0.856); liq path PARK.
3. Turning 45→65 is Y3: 0.762 / 0.752 beating days 0.711; Family I CLOSE 0.7615; no new card cols.
4. Dip vs fall is Y7 TURNOVER 0.720 / issued_lag1; Y2 ≠ Y4 (Jaccard 0.057); Y4 monopoly tail 0.605; Y8 PARK.
5. Why: CN leftover KEEP 0.597; delay leftover KEEP 0.581; Y5 leftover 65%; missing-CP is the d_tx twin; Family J KEEP-Q5 not Y3 X; Family M CLOSE.
6. Months earlier: KEEP issued_lag1 0.626 and days_lag1 0.684; CLOSE F lag3, Y4 HHI, delay, pending, DPO, supp HHI, growth_12. Hidden 72 = 1-month claims only.

## Contradictions resolved (later leftover wins)

Feature-report 44 is the starter, not the morning list. Settled DROP-from-44 as of 04:50: e_fx, f_has_*, c_gap_sd, c_recency_days, a_io_ratio, a_growth_3, c_zero_in_*, e_credit_note_ratio as Y3 X, delay/overdue as Y3 X, e_pending_amt_share, remaining g_has_* + g_custom_share, e_dpo_proxy, d_supp_hhi.

Role splits (not fights): CN/delay KEEP leftover vs CLOSE TURNOVER add-on vs DROP as Y3 X; Family J KEEP-Q5 not Y3 X; Y4 HHI KEEP as monopoly tail vs CLOSE as Q6; Family H PARK as X / KEEP-Q5 footnote; a_out_vol 0.722 is a trait, not the engine.

## In-flight / after-cut

- **B-on-44** and **DSO** files exist after CONTEXT 04:50. Verdicts not invented. Not on the 04:50 drop list. `b_runway` already KEEP as Q1 description; TURNOVER already drops DSO.
- **`d_tx_qa.md` 05:00** (after the cut). Do not fold `d_tx_cp_share` into the 04:50 drop list. After-cut if morning reads it: KEEP 0.611 quote, PARK as X, DROP from the 44 as Y3 X.

## Compile

52 map rows. Source token+number misses **0**. 44-drop leftover confirm **0**. Holdout n=72. Registry 67e8ef01 / 0.762239 ok. 15-col stems not in the drop list. No GBM. No `build_targets`.

## What failed / next

- First compile had 11 source-string misses (y3_importances does not print 0.762; shap_y7 does not print 56%). Retied to CONTEXT / i_lift / q6_quoted / sibling_h (`+6.97pp`) / debt_schedule (`-5.55`).
- Table-level KEEP+DROP parser saw 0 collisions — leftover tables do not share object strings. The resolved table is the hunt.
- Morning still cannot write a 0–100, quote holdout AUROC, or settle B / DSO.
- Next is not this owner: parent reads B-on-44 / DSO / (optional) d_tx after the 04:50 cut.

## What this module did not do

No 0–100. No product/. No parquet merge. No Family I/M/J merge. No 15-col add. No TURNOVER grow. No holdout fit. No commit. No parent journal / LIVE / canvas.
