# Wave 4 — growth / io_ratio QA

Agent `f4320aab`. Train group-fold seed 20260918. Holdout 72 coverage only.

## Files written

- `analysis/evaluate/growth_qa.py`
- `analysis/outputs/growth_qa.md`
- `analysis/outputs/growth_quintiles.png`
- append-only `analysis/experiments/registry.csv` (skip key agent+y+model+split+metric+x_families)
- this note

Did not touch `recency_qa.py`, `zero_in_qa.py`, `gap_sd_qa.py`, `cashflow.py`, `y4_why.py`, `product/`, parquet / duckdb, `build_targets`, parent journal, or the 15-col card. Night Y3 quote stays **0.762 / 0.752**. Days bar stays **0.711**.

## Columns / coverage (train)

- `a_io_ratio` 88.5% (CONFIRM feature-report). Formula vs in3/out3 max|Δ|=0.
- `a_growth_3` 64.0% (CONFIRM). 137 / 402 Y3 positives lack it (short book / zero in3_{t-3}).
- `a_growth_12` 26.5% (CONFIRM). **0.0%** on short-trail and on so_far<15. Holdout 16.5%; 69/72 late-arrival.

## Locked verdict

| object | decision |
| --- | --- |
| `a_io_ratio` on the 44 | **DROP from the 44** / CLOSE as X |
| `a_growth_3` on the 44 | **DROP from the 44** / CLOSE as X |
| `a_growth_12` on the 44 | **CLOSE as Q6** |
| any as a health Y | **PARK** (do not invent `y_growth`) |
| leftover after size+days | **dies** (io 0.527 / g3 0.522 / g12 0.399 vs size 0.617) |
| mean-reversion | **U-shape + mechanical acf3**, not monotone-only |
| clip-at-3 dummy | **NO** (14.9% of defined sit at 3.0; flag 0.522) |
| Q6 | **CLOSE as Q6** |

Y3 io **0.565** / g3 **0.515** / g12 **0.526** vs size **0.617** (CONFIRM) vs days **0.711** (CONFIRM). Same-row size still wins (Δ −0.052 / −0.068 / −0.071). Signs all −1. io is a **net_margin twin** (ρ=0.989), not SIZE (ρ=0.355) and not a days twin (ρ=0.172).

io leftover-after-size 0.604 is a **nonlinear SIZE leak** (ρ(resid,size)=−0.643; rank leftover 0.533). Body (io<3) leftover-days 0.619 loses to days-on-body 0.737.

growth_3 quintiles U-shape 9.6% / 4.0% / 9.4%. acf3=−0.433 is **mechanical** (persistent in3 acf1=0.655; even non-overlapping acf1=−0.381). Company-mean g3 0.664 is STYLE; leftover after co-mean size+days 0.531 dies. MoM inflow leftover both 0.563 dies. Naive trailing-3m −40% 0.553; MoM −40% 0.585 / Y2 0.485.

growth_12 ICC 0.904 TRAIT. Empty on short books. Dark 16.7% vs 744 22.5% is shorter trail (ρ vs trail 0.897). Holdout late-arrival **69/72** — hidden-test hole. Do not merge with Y4 (ρ=0.078; Y4=1 has *higher* current growth_3).

12 chronic Y2 names: no flip (0.515 → 0.514). Dark 744/470 CONFIRM. Clip pile is saturation, not a dummy.

## What failed / extras that stayed CLOSE

- KEEP gate needs leftover after size **and** days beating size ≥0.02 **and** a month shock. None of the three clear it.
- T1 pocket: io 0.541 leftover-days 0.453 vs days 0.603 — not a KEEP footnote.
- Do not put these on tonight's 15-col card. Night engine stays days / SS / salary / n_tx / f_ds_r.

## Next (parent)

DROP `a_io_ratio` and `a_growth_3` from the 44. CLOSE `a_growth_12` as Q6 (short-book / hidden 72). PARK as Ys. Do not invent `y_growth`. Do not merge with Y4. Days 0.711 and night Y3 0.762 / 0.752 unchanged.
