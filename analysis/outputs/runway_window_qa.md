# Last-value vs Hair 3-month mean runway (Q1 photograph)

Generated `2026-09-19T07:53:03+02:00` by `analysis/evaluate/runway_window_qa.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Train only. Holdout 72 coverage only. Seed 20260918 group folds.
No 0–100. No `liquidity.py` edit. No parquet rewrite. No `build_targets`.
No new GBM. Never B as Y2/Y3 X. Never engine X on the 15-col card.
Do not reconstruct Norden AMPLI. Do not invent Y10. Do not quote
`a_out_vol` 0.722 as the engine. Do not overwrite `balances_b_qa.*` /
`b_on_44_qa.*` / `y3_reasons.*` / `days_delta_qa.*`.

`run3` = rolling 3-month mean of `b_runway` ending at t (Hair/FinRegLab
3-month average as a photograph). `run3_prior` = mean of t−1..t−3
(application-strict, three months *before* t). `liq3` = same for `b_liq`.
Y3 leftover is a **diagnostic** that the 3-month window is not a different
photograph — it is not a KEEP-as-X gate for the card.

## Headline

last-value KEEP stays; Hair 3-month mean is application-only (last-month still twin). run3 leftover after last-value rank 0.608 (OLS 0.625, fake=False). ρ vs last-value 0.792 twin=False. run3 raw 0.663 vs last 0.596 vs size 0.617 beat=True Δ=0.046. SIZE=False. Inverse last-after-run3 0.594. Last-month p50 1.079 (night 1.079). Never B as Y2/Y3 X. Off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Last-value `b_runway` KEEP (p50 1.079 ≈ night 1.079). 3-month mean is **APPLICATION-ONLY**. |
| 2 | Who is improving? | Runway persist t↔t−3 0.757; liq persist 0.848 (night ~0.85). Y1 path stays PARK. |
| 3 | Who is turning? | Not B. Quiet-stressed stays SS/salary/days. |
| 4 | Dip vs fall? | Out. Sibling TURNOVER 0.72. |
| 5 | Why did it change? | A smoother cash window is not a why. |
| 6 | Months earlier? | Hidden 72 is coverage. 3-month mean is not a holdout lead. |

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| last-value `b_runway` / `b_liq` | **KEEP Q1 photograph** | p50 1.079; share&lt;1 0.484; last-vs-snap night 0.966 |
| 3-month mean `run3` as photograph | **APPLICATION-ONLY** | last-month still ρ 0.946 twin=True (Q1 gate). Y3 leftover after last 0.608 is diagnostic only (Y3-slice ρ 0.792). beat=True; SIZE=False. Fold 0 leftover dies. Never engine X. |
| `run3` as Y2/Y3 X / 15-col card | **never** | B-family. 16h death if Y and X share cash. |
| `run3` as forecast Y | **PARK** | Y1 last-value already wins path CV; this is not a reconstruction |
| `run3_prior` (t−1..t−3) | **APPLICATION-ONLY** | last-month still ρ 0.873 twin vs last-value. Y3 leftover 0.641 / Y3-slice ρ 0.690 diagnostic only. |
| `liq3` / store `b_mean_liq_3` | **APPLICATION-ONLY** | leftover 0.591; ρ last 0.916. Store `b_mean_liq_3` vs in-memory liq3 ρ=1.000; leftover after last-liq 0.591 twin=True. |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.72/0.712 |

## 1. Last-value photograph (train last month)

n=1,214 last-month rows / 1214 train companies.
`b_runway` p50 **1.079** (night 1.079). share runway&lt;1 0.484.
Spearman last-value runway t ↔ t−3 0.757; liq t ↔ t−3 0.848 (night liq ~0.85).
Last-month still (n=1,193): ρ(run3, last) **0.946**; ρ(liq3, last) 0.942; ρ(run3_prior, last) 0.873. The Q1 photograph twin is this still, not the Y3-labeled slice.

## 2. Twin / SIZE (Y3-defined rows)

| vs | run3 | liq3 | run3_prior |
| --- | ---: | ---: | ---: |
| last-value | 0.792  | 0.916 | 0.690 |
| log1p(a_in3) | -0.000 | 0.489 | -0.026 |

## 3. Diagnostic Y3 leftover after last-value (never card)

Sign from the train side of each fold. B is not a Y3 reason.

| stem | raw | leftover after last | inverse last-after-3m | folds leftover | n_pos | role |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| run3 | 0.663 | 0.608 | 0.594 | 0.538 0.646 0.648 0.626 0.580 | 334 | APPLICATION-ONLY |
| liq3 | 0.538 | 0.591 | 0.570 | 0.648 0.582 0.596 0.572 0.558 | 391 | APPLICATION-ONLY |
| run3_prior | 0.681 | 0.641 | 0.633 | 0.537 0.706 0.692 0.657 0.616 | 313 | APPLICATION-ONLY |
| last-value raw* | 0.596 | — | — | 0.662 0.580 0.601 0.527 0.609 | 391 | KEEP Q1, not X |

\* last-value / run3 raw vs Y3 is a leak screen, not a card claim.

run3 leftover after days (not last-value) 0.557 — do not treat B as an activity leftover.

## Extra — short vs long books

| slice | n_pos | ρ last | leftover | role |
| --- | ---: | ---: | ---: | --- |
| all train | 402 | 0.792 | 0.608 | APPLICATION-ONLY |
| company &lt;12 | 0 | 0.545 | — | LOW_POWER |
| company ≥18 | 312 | 0.792 | 0.599 | PARK-as-photograph |
| last-month only | 0 | — | — | LOW_POWER |

## Extra — dark vs ERP

| slice | n_pos | ρ last | leftover | role |
| --- | ---: | ---: | ---: | --- |
| never-ERP | 114 | 0.840 | 0.564 | APPLICATION-ONLY |
| ever-ERP | 220 | 0.751 | 0.621 | PARK-as-photograph |

## Extra — company bootstrap leftover (n=60)

run3 leftover after last-value p05/p50/p95 0.580 / 0.610 / 0.638; share&lt;0.55=0.000 (n=60).

## Explicitly out

- Editing `liquidity.py`. Putting B on Y2/Y3 X or the 15-col card.
- AMPLI (HIGH−LOW of balances). Utilisation / Y10. `a_out_vol` 0.722 as the engine.
- Overwrite `balances_b_qa.*` / `b_on_44_qa.*` / `y3_reasons.*` / `days_delta_qa.*`.
- Hidden-72 fit. New Y. 0–100.

Plot: `runway_window_qa.png`.

Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.72/0.712 unchanged.
