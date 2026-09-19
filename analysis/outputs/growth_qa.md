# Growth / io_ratio — leftover Q3, or mean-reversion / SIZE / short-book NaNs?

Generated `2026-09-19T04:31:30+02:00` by agent `f4320aab`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_growth`. Do not put `a_io_ratio` / `a_growth_3` / `a_growth_12` on tonight's 15-col card. Night Y3 quote stays **0.762 / 0.752**. Days bar stays **0.711**. Do not merge with Y4.

`a_io_ratio` = min(3, in3 / max(out3, 1)). `a_growth_3` = clip(in3 / in3_{t-3} − 1, −1, 1) if in3_{t-3}>0 else NaN. `a_growth_12` = YoY of the trailing-3m inflow window (needs month 15+). `a_net_margin` was dropped as redundant with io_ratio.

## Headline

Coverage io/g3/g12 88.5% / 64.0% / 26.5%. Y3 io 0.565 g3 0.515 g12 0.526 vs size 0.617 vs days 0.711. Leftover size+days io 0.527 g3 0.522 g12 0.399. U-shape g3=YES; mechanical acf3=YES; MoM −40% 0.585. io **DROP from the 44**. g3 **DROP from the 44**. g12 **CLOSE as Q6**. PARK as Y. Q6 **CLOSE as Q6**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not a new Y. PARK `y_growth`. io_ratio is a coverage *level*, not a health label. |
| 2 | Who is improving? | growth_3 / growth_12 are the change objects. Y3 +growth 0.515 vs −growth 0.515. |
| 3 | Who is turning? | **CLOSE** — leftover after size+days io 0.527 / g3 0.522 / g12 0.399 vs size 0.617 / days 0.711. |
| 4 | Dip vs fall? | Not this table. Y4 overlap is descriptive only. |
| 5 | Why did it change? | U-shape / mechanical acf3 -0.433 (window overlap); MoM −40% 0.585. |
| 6 | Months earlier? | **CLOSE as Q6** — growth_12 is empty on short books (n_def short=0, so_far<15=0). Hidden-72 late-arrival hole. CLOSE as Q6. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `a_io_ratio` as Y3 X (on the 44) | **DROP from the 44** | leftover dies (size 0.604 / days 0.573 / both 0.527 vs size 0.617); raw CV 0.565 does not beat size 0.617 by ≥0.02; not a month shock (ICC 0.702). CLOSE as X. Still **not** on tonight's 15-col card. |
| `a_growth_3` as Y3 X (on the 44) | **DROP from the 44** | leftover dies (size 0.526 / days 0.528 / both 0.522 vs size 0.617); raw CV 0.515 does not beat size 0.617 by ≥0.02; not a month shock (ICC 0.704). CLOSE as X. Still **not** on tonight's 15-col card. |
| `a_growth_12` as Y3 X (on the 44) | **CLOSE as Q6** | growth_12 is empty on short books (needs month 15+). CLOSE as Q6 (hidden 72 late-arrival). Still **not** on tonight's 15-col card. |
| any of the three as a health Y | **PARK** | do not invent `y_growth` |
| mean-reversion | **measured** | Y3 `a_growth_3` sign=-1 (− risky / already-bounced). +growth CV 0.515 vs −growth 0.515 vs naive −40% 0.553 (naive crash is not a fail). Y3 rate growth>0 6.5% vs growth<0 6.6% (all defined 6.6%). Next-month Y3 after +growth 6.7% vs −growth 6.2%. Quintile Q1→Q5 9.6%→9.4% shape=flat. acf1/3/6 g3 0.523/-0.433/-0.071 (lag-3 mean-reversion). not only −growth. |
| io clip-at-3 dummy | **NO** | `a_io_ratio` sitting at 3.0: 2,798 / 18,729 defined (14.9%; 13.2% of all train CM). Unclip p50=0.97 p90=7.45 p99=140745.03. Y3 clip-flag 0.522 vs continuous 0.565 vs unclip 0.564; body (io<3) 0.604. P(Y3|clip)=12.7% vs body 6.7%. clip is saturation, not the whole dummy. |
| Q6 short-book growth_12 | **CLOSE as Q6** | growth_12 is empty on short books (n_def short=0, so_far<15=0). Hidden-72 late-arrival hole. CLOSE as Q6. |
| Y4 crash overlap | descriptive only | do not score as Y4 X; do not merge |

## 1. Coverage — confirm 88.5% / 64.0% / 26.5%

Train coverage `a_io_ratio` 88.5% (CONFIRM 88.5%), `a_growth_3` 64.0% (CONFIRM 64.0%), `a_growth_12` 26.5% (CONFIRM 26.5%). growth_12 on short-trail companies 0.0% (EMPTY — CLOSE as Q6 hole); on so_far<15 0.0% (EMPTY as formula (needs month 15+)). Short-trail companies 452; late-arrival 779.

| col | n_def | cov | quote | confirm | n_co | cov_co |
| --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | 18,729 | 88.5% | 88.5% | YES | 1,214 | 100.0% |
| a_growth_3 | 13,532 | 64.0% | 64.0% | YES | 1,188 | 97.9% |
| a_growth_12 | 5,600 | 26.5% | 26.5% | YES | 727 | 59.9% |


Short-book / late-arrival hole:

| slice | n_cm | n_co | io | g3 | g12 |
| --- | --- | --- | --- | --- | --- |
| all | 21157 | 1214 | 88.5% | 64.0% | 26.5% |
| short_trail<15 | 4299 | 452 | 79.0% | 43.8% | 0.0% |
| long_trail>=15 | 16858 | 762 | 91.0% | 69.1% | 33.2% |
| late_arrival | 10717 | 779 | 85.5% | 56.6% | 15.2% |
| on_time_2024-09 | 10440 | 435 | 91.7% | 71.5% | 38.0% |
| so_far<15 | 14967 | 1214 | 83.8% | 53.8% | 0.0% |
| so_far>=15 | 6190 | 762 | 100.0% | 88.6% | 90.5% |


Holdout coverage (no AUROC):

| col | n_cm | n_co | cov | n_def |
| --- | --- | --- | --- | --- |
| a_io_ratio | 1073 | 72 | 86.6% | 929 |
| a_growth_3 | 1073 | 72 | 61.3% | 658 |
| a_growth_12 | 1073 | 72 | 16.5% | 177 |


Plot: `growth_quintiles.png`.

## 2. Formula vs store (in3 / out3, clip)

Store vs in3/out3 reconstruction: a_io_ratio max|Δ|=0.00e+00, a_growth_3 max|Δ|=0.00e+00, a_growth_12 max|Δ|=0.00e+00. Clip identity on unclip≥3: YES. Formulas match cashflow.py — not a rewrite.

| col | n_both | max|Δ| | exact | only store | only hat | ok |
| --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | 18,729 | 0.00e+00 | 100.0% | 0 | 0 | YES |
| a_growth_3 | 13,532 | 0.00e+00 | 100.0% | 0 | 0 | YES |
| a_growth_12 | 5,600 | 0.00e+00 | 100.0% | 0 | 0 | YES |


## 3. Spearman vs size / days / net_margin

Spearman vs log1p(a_in3): io 0.355 (not SIZE), g3 0.348 (not SIZE), g12 0.485 (not SIZE). vs days: io 0.172 g3 0.209 g12 0.333. io vs net_margin 0.989 (TWIN |ρ|≥0.80 — keep-list already dropped net_margin).

| pair | ρ | call |
| --- | --- | --- |
| a_io_ratio vs log1p(a_in3) | 0.355 | — |
| a_io_ratio vs a_in3 | 0.355 | — |
| a_io_ratio vs c_n_days_with_tx | 0.172 | — |
| a_io_ratio vs a_net_margin | 0.989 | TWIN |
| a_io_ratio vs a_n_tx | 0.187 | — |
| a_growth_3 vs log1p(a_in3) | 0.348 | — |
| a_growth_3 vs a_in3 | 0.348 | — |
| a_growth_3 vs c_n_days_with_tx | 0.209 | — |
| a_growth_3 vs a_net_margin | 0.346 | — |
| a_growth_3 vs a_n_tx | 0.221 | — |
| a_growth_12 vs log1p(a_in3) | 0.485 | — |
| a_growth_12 vs a_in3 | 0.485 | — |
| a_growth_12 vs c_n_days_with_tx | 0.333 | — |
| a_growth_12 vs a_net_margin | 0.421 | — |
| a_growth_12 vs a_n_tx | 0.339 | — |
| a_io_ratio vs a_growth_3 | 0.370 | — |
| a_io_ratio vs a_growth_12 | 0.454 | — |
| a_growth_3 vs a_growth_12 | 0.476 | — |
| a_io_ratio vs a_net_margin | 0.989 | TWIN |


Twin if \|ρ\|≥0.80. SIZE if \|ρ\| vs log1p(a_in3) ≥0.50.

## 4. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711); size 0.617 (replica 0.617). Night Y3 0.762 / 0.752 is **not** this cut.

Y3 singles (train group-fold): io 0.565 / g3 0.515 / g12 0.526 vs size 0.617 (quote 0.617 CONFIRM) vs days 0.711 (night 0.711 CONFIRM). Signs io=-1 g3=-1 g12=-1. −growth3 0.515; naive crash −40% 0.553. Y2 io 0.473 g3 0.516 g12 0.504 vs size 0.552. Night Y3 quote stays 0.762 / 0.752 (not this cut).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_io_ratio | 5,528 | 391 | 0.565 | 0.043 | -1 | 0.563 | 0.523 0.578 0.632 0.550 0.541 |
| y3_recover_cash_6m | a_growth_3 | 4,006 | 265 | 0.515 | 0.046 | -1 | 0.510 | 0.573 0.528 0.518 0.445 0.511 |
| y3_recover_cash_6m | a_growth_12 | 910 | 79 | 0.526 | 0.160 | -1 | 0.564 | 0.619 0.682 0.311 0.614 0.402 |
| y3_recover_cash_6m | neg_growth3 | 4,006 | 265 | 0.515 | 0.046 | 1 | 0.510 | 0.573 0.528 0.518 0.445 0.511 |
| y3_recover_cash_6m | naive_crash40 | 4,006 | 265 | 0.553 | 0.038 | 1 | 0.552 | 0.598 0.542 0.551 0.498 0.577 |
| y3_recover_cash_6m | io_unclip | 5,528 | 391 | 0.564 | 0.043 | -1 | 0.562 | 0.522 0.577 0.633 0.550 0.540 |
| y3_recover_cash_6m | io_clip3 | 5,648 | 402 | 0.522 | 0.018 | 1 | 0.525 | 0.498 0.538 0.514 0.518 0.542 |
| y3_recover_cash_6m | a_net_margin | 5,528 | 391 | 0.557 | 0.035 | -1 | 0.553 | 0.518 0.570 0.610 0.548 0.540 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.619 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 | 0.652 0.732 0.697 0.699 0.737 |
| y2_neg_2of3 | a_io_ratio | 14,968 | 1,044 | 0.473 | 0.069 | 1 | 0.509 | 0.438 0.571 0.481 0.384 0.490 |
| y2_neg_2of3 | a_growth_3 | 10,304 | 699 | 0.516 | 0.045 | 1 | 0.514 | 0.523 0.488 0.583 0.522 0.462 |
| y2_neg_2of3 | a_growth_12 | 3,568 | 222 | 0.504 | 0.067 | -1 | 0.533 | 0.536 0.447 0.429 0.516 0.593 |
| y2_neg_2of3 | neg_growth3 | 10,304 | 699 | 0.516 | 0.045 | -1 | 0.514 | 0.523 0.488 0.583 0.522 0.462 |
| y2_neg_2of3 | naive_crash40 | 10,304 | 699 | 0.523 | 0.033 | -1 | 0.522 | 0.525 0.514 0.577 0.503 0.494 |
| y2_neg_2of3 | io_unclip | 14,968 | 1,044 | 0.472 | 0.070 | 1 | 0.512 | 0.438 0.566 0.485 0.376 0.492 |
| y2_neg_2of3 | io_clip3 | 17,356 | 1,271 | 0.483 | 0.043 | 1 | 0.519 | 0.485 0.474 0.511 0.416 0.528 |
| y2_neg_2of3 | a_net_margin | 14,968 | 1,044 | 0.469 | 0.069 | 1 | 0.510 | 0.432 0.560 0.480 0.377 0.499 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.522 0.610 0.595 0.520 0.513 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 | 0.641 0.569 0.648 0.584 0.549 |


Same-row size / days (coverage-fair):

| feature | n | feat CV | size same-row | days same-row | Δ size |
| --- | --- | --- | --- | --- | --- |
| a_io_ratio | 5,528 | 0.565 | 0.617 | 0.718 | -0.052 |
| a_growth_3 | 4,006 | 0.515 | 0.583 | 0.723 | -0.068 |
| a_growth_12 | 910 | 0.526 | 0.597 | 0.709 | -0.071 |


KEEP-as-X on the 44: beat size by ≥0.02 **and** leftover after size **and** leftover after days **and** not mean-reversion-only **and** a month shock. Still not on the 15-col card.

## 5. Residual after size and after days

Y3 leftover after size / days / size+days: a_io_ratio 0.604 / 0.573 / 0.527 DIED; a_growth_3 0.526 / 0.528 / 0.522 DIED; a_growth_12 0.397 / 0.550 / 0.399 DIED vs size 0.617. Leftover dies — DROP from the 44 / CLOSE as X.

| y | feature | after | n | n_pos | CV | sign | slope |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_io_ratio | size | 5,528 | 391 | 0.604 | 1 | 0.238 |
| y3_recover_cash_6m | a_io_ratio | days | 5,528 | 391 | 0.573 | 1 | 0.042 |
| y3_recover_cash_6m | a_io_ratio | size+days | 5,528 | 391 | 0.527 | 1 | 0.287,-0.044 |
| y3_recover_cash_6m | a_growth_3 | size | 4,006 | 265 | 0.526 | 1 | 0.078 |
| y3_recover_cash_6m | a_growth_3 | days | 4,006 | 265 | 0.528 | 1 | 0.016 |
| y3_recover_cash_6m | a_growth_3 | size+days | 4,006 | 265 | 0.522 | 1 | 0.080,-0.002 |
| y3_recover_cash_6m | a_growth_12 | size | 910 | 79 | 0.397 | 1 | 0.086 |
| y3_recover_cash_6m | a_growth_12 | days | 910 | 79 | 0.550 | 1 | 0.026 |
| y3_recover_cash_6m | a_growth_12 | size+days | 910 | 79 | 0.399 | 1 | 0.085,0.001 |
| y2_neg_2of3 | a_io_ratio | size | 14,968 | 1,044 | 0.491 | -1 | 0.238 |
| y2_neg_2of3 | a_io_ratio | days | 14,968 | 1,044 | 0.479 | -1 | 0.042 |
| y2_neg_2of3 | a_io_ratio | size+days | 14,968 | 1,044 | 0.426 | 1 | 0.287,-0.044 |
| y2_neg_2of3 | a_growth_3 | size | 10,304 | 699 | 0.495 | -1 | 0.078 |
| y2_neg_2of3 | a_growth_3 | days | 10,304 | 699 | 0.478 | -1 | 0.016 |
| y2_neg_2of3 | a_growth_3 | size+days | 10,304 | 699 | 0.468 | -1 | 0.080,-0.002 |
| y2_neg_2of3 | a_growth_12 | size | 3,568 | 222 | 0.564 | -1 | 0.086 |
| y2_neg_2of3 | a_growth_12 | days | 3,568 | 222 | 0.565 | -1 | 0.026 |
| y2_neg_2of3 | a_growth_12 | size+days | 3,568 | 222 | 0.565 | -1 | 0.085,0.001 |


## 6. Mean-reversion — high growth_3 protective or risky for Y3?

Y3 `a_growth_3` sign=-1 (− risky / already-bounced). +growth CV 0.515 vs −growth 0.515 vs naive −40% 0.553 (naive crash is not a fail). Y3 rate growth>0 6.5% vs growth<0 6.6% (all defined 6.6%). Next-month Y3 after +growth 6.7% vs −growth 6.2%. Quintile Q1→Q5 9.6%→9.4% shape=flat. acf1/3/6 g3 0.523/-0.433/-0.071 (lag-3 mean-reversion). not only −growth.

| x | Q | interval | n | n_pos | P(Y3=1) | x p50 |
| --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | 1 | (-2.6189999999999998, 0.418] | 1,106 | 124 | 11.2% | 0.061 |
| a_io_ratio | 2 | (0.418, 0.825] | 1,105 | 82 | 7.4% | 0.670 |
| a_io_ratio | 3 | (0.825, 1.002] | 1,106 | 46 | 4.2% | 0.935 |
| a_io_ratio | 4 | (1.002, 1.224] | 1,105 | 52 | 4.7% | 1.072 |
| a_io_ratio | 5 | (1.224, 3.0] | 1,106 | 87 | 7.9% | 1.811 |
| a_growth_3 | 1 | (-1.001, -0.471] | 802 | 77 | 9.6% | -0.759 |
| a_growth_3 | 2 | (-0.471, -0.116] | 801 | 40 | 5.0% | -0.268 |
| a_growth_3 | 3 | (-0.116, 0.144] | 801 | 32 | 4.0% | 0.006 |
| a_growth_3 | 4 | (0.144, 0.783] | 801 | 41 | 5.1% | 0.361 |
| a_growth_3 | 5 | (0.783, 1.0] | 801 | 75 | 9.4% | 1.000 |
| a_growth_12 | 1 | (-1.001, -0.665] | 182 | 22 | 12.1% | -0.916 |
| a_growth_12 | 2 | (-0.665, -0.252] | 182 | 22 | 12.1% | -0.449 |
| a_growth_12 | 3 | (-0.252, 0.0619] | 182 | 9 | 4.9% | -0.084 |
| a_growth_12 | 4 | (0.0619, 0.71] | 182 | 8 | 4.4% | 0.291 |
| a_growth_12 | 5 | (0.71, 1.0] | 182 | 18 | 9.9% | 1.000 |


Next-month Y3 quintiles of growth_3: Q1 8.1% → Q5 9.5% shape=flat.

## 7. `a_io_ratio` clip-at-3

`a_io_ratio` sitting at 3.0: 2,798 / 18,729 defined (14.9%; 13.2% of all train CM). Unclip p50=0.97 p90=7.45 p99=140745.03. Y3 clip-flag 0.522 vs continuous 0.565 vs unclip 0.564; body (io<3) 0.604. P(Y3|clip)=12.7% vs body 6.7%. clip is saturation, not the whole dummy.

| item | value |
| --- | ---: |
| defined / clip=3 | 18,729 / 2,798 |
| share of defined | 14.9% |
| unclip p50 / p90 / p99 | 0.97 / 7.45 / 140745.03 |
| Y3 clip-flag / continuous / unclip / body | 0.522 / 0.565 / 0.564 / 0.604 |

## 8. Dark 470 vs invoiced 744

Train last-month companies: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean company growth_12 coverage: invoiced 22.5% vs dark 16.7%. Dark files growth_12 at a different rate. Holdout ever-ERP coverage only: 40/72.

| group | n_co | io cov | g3 cov | g12 cov | io p50 | g3 p50 | g12 p50 | trail p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 86.8% | 60.5% | 22.5% | 0.974 | -0.005 | -0.031 | 20.0 |
| never_erp_470 | 470 | 84.7% | 55.2% | 16.7% | 0.954 | -0.001 | -0.059 | 16.5 |


| group | feature | n | n_pos | CV | cov |
| --- | --- | --- | --- | --- | --- |
| ever_erp | a_io_ratio | 3,543 | 257 | 0.594 | 89.0% |
| ever_erp | a_growth_3 | 2,588 | 178 | 0.441 | 65.4% |
| ever_erp | a_growth_12 | 610 | 58 | 0.525 | 28.2% |
| never_erp | a_io_ratio | 1,985 | 134 | 0.421 | 87.6% |
| never_erp | a_growth_3 | 1,418 | 87 | 0.437 | 61.3% |
| never_erp | a_growth_12 | 300 | 21 | LOW_POWER | 23.4% |


## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)

Chronic dark Y2 names n=12 (expect 12). Y3 growth_3 all 0.515 vs drop12 0.514 (no flip).

| y | feature | slice | n | n_pos | CV |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_io_ratio | all | 5,528 | 391 | 0.565 |
| y3_recover_cash_6m | a_io_ratio | drop12 | 5,374 | 391 | 0.566 |
| y3_recover_cash_6m | a_io_ratio | chronic12 | 154 | 0 | LOW_POWER |
| y2_neg_2of3 | a_io_ratio | all | 14,968 | 1,044 | 0.473 |
| y2_neg_2of3 | a_io_ratio | drop12 | 14,776 | 881 | 0.478 |
| y2_neg_2of3 | a_io_ratio | chronic12 | 192 | 163 | 0.505 |
| y3_recover_cash_6m | a_growth_3 | all | 4,006 | 265 | 0.515 |
| y3_recover_cash_6m | a_growth_3 | drop12 | 3,886 | 265 | 0.514 |
| y3_recover_cash_6m | a_growth_3 | chronic12 | 120 | 0 | LOW_POWER |
| y2_neg_2of3 | a_growth_3 | all | 10,304 | 699 | 0.516 |
| y2_neg_2of3 | a_growth_3 | drop12 | 10,148 | 565 | 0.516 |
| y2_neg_2of3 | a_growth_3 | chronic12 | 156 | 134 | 0.448 |
| y3_recover_cash_6m | a_growth_12 | all | 910 | 79 | 0.526 |
| y3_recover_cash_6m | a_growth_12 | drop12 | 886 | 79 | 0.523 |
| y3_recover_cash_6m | a_growth_12 | chronic12 | 24 | 0 | LOW_POWER |
| y2_neg_2of3 | a_growth_12 | all | 3,568 | 222 | 0.504 |
| y2_neg_2of3 | a_growth_12 | drop12 | 3,520 | 180 | 0.522 |
| y2_neg_2of3 | a_growth_12 | chronic12 | 48 | 42 | LOW_POWER |


## 10. ICC / company-demean

ICC io=0.702 (report 0.70) g3=0.704 (report 0.70) g12=0.904 (report 0.90 BETWEEN). Y3 demean io 0.586 (drop -0.021); g3 0.553 (drop -0.039); g12 0.592 (drop -0.066). growth_3 is a month shock / change. growth_12 is TRAIT (BETWEEN).

| col | ICC | acf1 | acf3 | Y3 raw | Y3 demean | Y3 co-mean | drop | call |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | 0.702 | 0.551 | -0.143 | 0.565 | 0.586 | 0.439 | -0.021 | between |
| a_growth_3 | 0.704 | 0.523 | -0.433 | 0.515 | 0.553 | 0.664 | -0.039 | between |
| a_growth_12 | 0.904 | 0.525 | -0.340 | 0.526 | 0.592 | 0.647 | -0.066 | TRAIT |


Growth should be LOW_PERSIST / a month shock if it is a real change. Feature-report growth_12 ICC 0.90 is BETWEEN — a company YoY style, not a shock.

## 11. Q6 — lag1 / lag3 on short vs long

growth_12 is empty on short books (n_def short=0, so_far<15=0). Hidden-72 late-arrival hole. CLOSE as Q6.

| y | slice | col | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all | a_io_ratio | 5,528 | 391 | 0.565 | -1 |
| y3_recover_cash_6m | all | a_io_ratio_lag1 | 5,078 | 355 | 0.534 | -1 |
| y3_recover_cash_6m | all | a_io_ratio_lag3 | 4,212 | 313 | 0.507 | -1 |
| y3_recover_cash_6m | all | a_growth_3 | 4,006 | 265 | 0.515 | -1 |
| y3_recover_cash_6m | all | a_growth_3_lag1 | 3,618 | 235 | 0.467 | 1 |
| y3_recover_cash_6m | all | a_growth_3_lag3 | 2,875 | 192 | 0.407 | -1 |
| y3_recover_cash_6m | all | a_growth_12 | 910 | 79 | 0.526 | -1 |
| y3_recover_cash_6m | all | a_growth_12_lag1 | 663 | 57 | 0.537 | -1 |
| y3_recover_cash_6m | all | a_growth_12_lag3 | 202 | 15 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_io_ratio | 282 | 24 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_io_ratio_lag1 | 191 | 15 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_io_ratio_lag3 | 76 | 8 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_growth_3 | 63 | 6 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_growth_3_lag1 | 31 | 2 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_growth_3_lag3 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_growth_12 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_growth_12_lag1 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | short | a_growth_12_lag3 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | long | a_io_ratio | 5,246 | 367 | 0.567 | -1 |
| y3_recover_cash_6m | long | a_io_ratio_lag1 | 4,887 | 340 | 0.536 | -1 |
| y3_recover_cash_6m | long | a_io_ratio_lag3 | 4,136 | 305 | 0.507 | -1 |
| y3_recover_cash_6m | long | a_growth_3 | 3,943 | 259 | 0.518 | -1 |
| y3_recover_cash_6m | long | a_growth_3_lag1 | 3,587 | 233 | 0.494 | 1 |
| y3_recover_cash_6m | long | a_growth_3_lag3 | 2,875 | 192 | 0.407 | -1 |
| y3_recover_cash_6m | long | a_growth_12 | 910 | 79 | 0.526 | -1 |
| y3_recover_cash_6m | long | a_growth_12_lag1 | 663 | 57 | 0.537 | -1 |
| y3_recover_cash_6m | long | a_growth_12_lag3 | 202 | 15 | LOW_POWER | — |
| y3_recover_cash_6m | so_far<15 | a_io_ratio | 4,584 | 309 | 0.561 | -1 |
| y3_recover_cash_6m | so_far<15 | a_io_ratio_lag1 | 4,134 | 273 | 0.529 | -1 |
| y3_recover_cash_6m | so_far<15 | a_io_ratio_lag3 | 3,268 | 231 | 0.524 | -1 |
| y3_recover_cash_6m | so_far<15 | a_growth_3 | 3,120 | 196 | 0.466 | 1 |
| y3_recover_cash_6m | so_far<15 | a_growth_3_lag1 | 2,729 | 166 | 0.516 | 1 |
| y3_recover_cash_6m | so_far<15 | a_growth_3_lag3 | 1,982 | 122 | 0.435 | -1 |
| y3_recover_cash_6m | so_far<15 | a_growth_12 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | so_far<15 | a_growth_12_lag1 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | so_far<15 | a_growth_12_lag3 | 0 | 0 | LOW_POWER | — |
| y3_recover_cash_6m | so_far>=15 | a_io_ratio | 944 | 82 | 0.545 | -1 |
| y3_recover_cash_6m | so_far>=15 | a_io_ratio_lag1 | 944 | 82 | 0.487 | -1 |
| y3_recover_cash_6m | so_far>=15 | a_io_ratio_lag3 | 944 | 82 | 0.595 | 1 |
| y3_recover_cash_6m | so_far>=15 | a_growth_3 | 886 | 69 | 0.560 | -1 |
| y3_recover_cash_6m | so_far>=15 | a_growth_3_lag1 | 889 | 69 | 0.365 | 1 |
| y3_recover_cash_6m | so_far>=15 | a_growth_3_lag3 | 893 | 70 | 0.392 | 1 |
| y3_recover_cash_6m | so_far>=15 | a_growth_12 | 910 | 79 | 0.526 | -1 |
| y3_recover_cash_6m | so_far>=15 | a_growth_12_lag1 | 663 | 57 | 0.537 | -1 |
| y3_recover_cash_6m | so_far>=15 | a_growth_12_lag3 | 202 | 15 | LOW_POWER | — |
| y2_neg_2of3 | all | a_io_ratio | 14,968 | 1,044 | 0.473 | 1 |
| y2_neg_2of3 | all | a_io_ratio_lag1 | 13,776 | 938 | 0.474 | 1 |
| y2_neg_2of3 | all | a_io_ratio_lag3 | 11,477 | 748 | 0.480 | 1 |
| y2_neg_2of3 | all | a_growth_3 | 10,304 | 699 | 0.516 | 1 |
| y2_neg_2of3 | all | a_growth_3_lag1 | 9,428 | 626 | 0.512 | 1 |
| y2_neg_2of3 | all | a_growth_3_lag3 | 7,795 | 517 | 0.532 | 1 |
| y2_neg_2of3 | all | a_growth_12 | 3,568 | 222 | 0.504 | -1 |
| y2_neg_2of3 | all | a_growth_12_lag1 | 2,919 | 184 | 0.479 | -1 |
| y2_neg_2of3 | all | a_growth_12_lag3 | 1,746 | 112 | 0.550 | 1 |
| y2_neg_2of3 | short | a_io_ratio | 1,958 | 106 | 0.582 | 1 |
| y2_neg_2of3 | short | a_io_ratio_lag1 | 1,526 | 72 | 0.594 | 1 |
| y2_neg_2of3 | short | a_io_ratio_lag3 | 747 | 23 | LOW_POWER | — |
| y2_neg_2of3 | short | a_growth_3 | 659 | 20 | LOW_POWER | — |
| y2_neg_2of3 | short | a_growth_3_lag1 | 450 | 8 | LOW_POWER | — |
| y2_neg_2of3 | short | a_growth_3_lag3 | 163 | 2 | LOW_POWER | — |
| y2_neg_2of3 | short | a_growth_12 | 0 | 0 | LOW_POWER | — |
| y2_neg_2of3 | short | a_growth_12_lag1 | 0 | 0 | LOW_POWER | — |
| y2_neg_2of3 | short | a_growth_12_lag3 | 0 | 0 | LOW_POWER | — |
| y2_neg_2of3 | long | a_io_ratio | 13,010 | 938 | 0.465 | 1 |
| y2_neg_2of3 | long | a_io_ratio_lag1 | 12,250 | 866 | 0.469 | 1 |
| y2_neg_2of3 | long | a_io_ratio_lag3 | 10,730 | 725 | 0.478 | 1 |
| y2_neg_2of3 | long | a_growth_3 | 9,645 | 679 | 0.511 | 1 |
| y2_neg_2of3 | long | a_growth_3_lag1 | 8,978 | 618 | 0.512 | 1 |
| y2_neg_2of3 | long | a_growth_3_lag3 | 7,632 | 515 | 0.534 | 1 |
| y2_neg_2of3 | long | a_growth_12 | 3,568 | 222 | 0.504 | -1 |
| y2_neg_2of3 | long | a_growth_12_lag1 | 2,919 | 184 | 0.479 | -1 |
| y2_neg_2of3 | long | a_growth_12_lag3 | 1,746 | 112 | 0.550 | 1 |
| y2_neg_2of3 | so_far<15 | a_io_ratio | 11,039 | 806 | 0.472 | 1 |
| y2_neg_2of3 | so_far<15 | a_io_ratio_lag1 | 9,847 | 700 | 0.474 | 1 |
| y2_neg_2of3 | so_far<15 | a_io_ratio_lag3 | 7,548 | 510 | 0.478 | 1 |
| y2_neg_2of3 | so_far<15 | a_growth_3 | 6,777 | 478 | 0.539 | 1 |
| y2_neg_2of3 | so_far<15 | a_growth_3_lag1 | 5,890 | 402 | 0.527 | 1 |
| y2_neg_2of3 | so_far<15 | a_growth_3_lag3 | 4,227 | 291 | 0.471 | 1 |
| y2_neg_2of3 | so_far<15 | a_growth_12 | 0 | 0 | LOW_POWER | — |
| y2_neg_2of3 | so_far<15 | a_growth_12_lag1 | 0 | 0 | LOW_POWER | — |
| y2_neg_2of3 | so_far<15 | a_growth_12_lag3 | 0 | 0 | LOW_POWER | — |
| y2_neg_2of3 | so_far>=15 | a_io_ratio | 3,929 | 238 | 0.471 | 1 |
| y2_neg_2of3 | so_far>=15 | a_io_ratio_lag1 | 3,929 | 238 | 0.470 | 1 |
| y2_neg_2of3 | so_far>=15 | a_io_ratio_lag3 | 3,929 | 238 | 0.468 | 1 |
| y2_neg_2of3 | so_far>=15 | a_growth_3 | 3,527 | 221 | 0.543 | -1 |
| y2_neg_2of3 | so_far>=15 | a_growth_3_lag1 | 3,538 | 224 | 0.516 | -1 |
| y2_neg_2of3 | so_far>=15 | a_growth_3_lag3 | 3,568 | 226 | 0.477 | 1 |
| y2_neg_2of3 | so_far>=15 | a_growth_12 | 3,568 | 222 | 0.504 | -1 |
| y2_neg_2of3 | so_far>=15 | a_growth_12_lag1 | 2,919 | 184 | 0.479 | -1 |
| y2_neg_2of3 | so_far>=15 | a_growth_12_lag3 | 1,746 | 112 | 0.550 | 1 |


## 12. Overlap with Y4 crash months (descriptive)

Y4 labeled 2,370 / pos 329. growth_3 p50 on Y4=1 0.079 vs Y4=0 -0.082. Share growth_3≤−0.4 among Y4 pos 21.6%. Spearman growth_3↔Y4 0.078 (descriptive). Do not score growth as Y4 X; do not merge with Y4.

| feature | Y4 n | Y4 pos | cov on Y4 | p50 Y4=1 | p50 Y4=0 | g3<=-0.4 on Y4=1 |
| --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | 2370 | 329 | 100.0% | 0.991 | 1.020 | — |
| a_growth_3 | 2370 | 329 | 77.7% | 0.079 | -0.082 | 21.6% |
| a_growth_12 | 2370 | 329 | 26.3% | -0.244 | -0.162 | — |


## 13. Growth clip ±1 saturation

growth_3 clip pile −1 4.9% +1 18.3%.

| feature | n_def | clip −1 | clip +1 | either |
| --- | --- | --- | --- | --- |
| a_growth_3 | 13,532 | 667 (4.9%) | 2,483 (18.3%) | 23.3% |
| a_growth_12 | 5,600 | 468 (8.4%) | 1,144 (20.4%) | 28.8% |


## 14. Size terciles

Y3 inside company size terciles (last-month log1p(a_in3)).

| feature | tercile | n | n_pos | feat CV | size CV |
| --- | --- | --- | --- | --- | --- |
| a_io_ratio | T1 | 1,533 | 261 | 0.541 | 0.427 |
| a_io_ratio | T2 | 2,006 | 72 | 0.388 | 0.634 |
| a_io_ratio | T3 | 1,989 | 58 | 0.378 | 0.620 |
| a_growth_3 | T1 | 998 | 169 | 0.540 | 0.427 |
| a_growth_3 | T2 | 1,506 | 50 | 0.345 | 0.634 |
| a_growth_3 | T3 | 1,502 | 46 | LOW_POWER | 0.620 |
| a_growth_12 | T1 | 239 | 57 | 0.470 | 0.427 |
| a_growth_12 | T2 | 341 | 15 | LOW_POWER | 0.634 |
| a_growth_12 | T3 | 330 | 7 | LOW_POWER | 0.620 |


## 15. Holdout coverage only

Holdout 72 coverage only: 1,073 CM. growth_12 16.5%. Short-trail companies 29; late-arrival 69; growth_12 on short 0.0%. No AUROC claim.

| col | n_cm | n_co | cov | p50 |
| --- | --- | --- | --- | --- |
| a_io_ratio | 1073 | 72 | 86.6% | 1.054 |
| a_growth_3 | 1073 | 72 | 61.3% | 0.036 |
| a_growth_12 | 1073 | 72 | 16.5% | -0.045 |
| a_net_margin | 1073 | 72 | 86.6% | 0.051 |
| a_in3 | 1073 | 72 | 86.6% | 339930.360 |


## 16. Calendar

Calendar medians — growth_12 coverage is a late-panel hole, not a month dummy.

| month | n | io p50 | g3 p50 | g12 p50 | g12 cov |
| --- | --- | --- | --- | --- | --- |
| Jan | 1,768 | 0.996 | 0.044 | -0.149 | 25.1% |
| Feb | 1,881 | 0.962 | 0.003 | -0.077 | 25.3% |
| Mar | 1,925 | 0.943 | -0.119 | -0.076 | 29.8% |
| Apr | 1,945 | 0.971 | -0.075 | -0.051 | 31.0% |
| May | 1,966 | 0.989 | -0.059 | -0.065 | 33.1% |
| Jun | 1,976 | 0.987 | 0.076 | -0.083 | 33.8% |
| Jul | 2,010 | 0.979 | 0.055 | -0.069 | 33.5% |
| Aug | 2,047 | 0.937 | 0.000 | -0.094 | 33.2% |
| Sep | 1,299 | 0.976 | -0.055 | — | 0.0% |
| Oct | 1,381 | 0.953 | -0.124 | — | 0.0% |
| Nov | 1,428 | 0.971 | -0.043 | -0.110 | 27.7% |
| Dec | 1,531 | 0.988 | 0.067 | -0.126 | 28.6% |


## 17. Residual vs control (fake leftover?)

OLS leftover should be ≈0 vs its control; a large leftover ρ is a numerical leak.

| feature | ρ(resid_days, days) | ρ(resid_size, size) | ρ(resid_days, size) |
| --- | --- | --- | --- |
| a_io_ratio | -0.304 | -0.643 | 0.062 |
| a_growth_3 | 0.007 | -0.034 | 0.253 |
| a_growth_12 | 0.019 | 0.005 | 0.309 |


## 18. Long-book same-row (so_far≥15)

Long-book (so_far≥15) same-row singles — growth_12's native support.

| feature | slice | n | n_pos | feat | size | days |
| --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | so_far>=15 | 944 | 82 | 0.545 | 0.596 | 0.703 |
| a_growth_3 | so_far>=15 | 886 | 69 | 0.560 | 0.581 | 0.713 |
| a_growth_12 | so_far>=15 | 910 | 79 | 0.526 | 0.597 | 0.709 |


## 19. U-shape / io body leftover / T1 pocket

growth_3 quintiles U-shape=YES (Q1 9.6% mid-low 4.0% Q5 9.4%). io U-shape=YES. io body (io<3) Y3 0.604 leftover-days 0.619 vs days-on-body 0.737 / size-on-body 0.621. T1 io raw 0.541 leftover-days 0.453 vs days 0.603. Two-tail is not KEEP; leftover after days dies.

| feature | raw | leftover days | days T1 | size T1 | n_pos |
| --- | --- | --- | --- | --- | --- |
| a_io_ratio | 0.541 | 0.453 | 0.603 | 0.427 | 261 |
| a_growth_3 | 0.540 | 0.553 | 0.603 | 0.427 | 169 |
| a_growth_12 | 0.470 | 0.578 | 0.603 | 0.427 | 57 |


## 20. Javier month-on-month inflow −40%

MoM a_op_in −40% Y3 0.585 (sign 1); continuous MoM 0.600. Y2 crash 0.485. MoM −40% is not a <0.55 fail on Y3.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | mom_in | 4,933 | 283 | 0.600 | -1 |
| y3_recover_cash_6m | mom_crash40 | 4,933 | 283 | 0.585 | 1 |
| y2_neg_2of3 | mom_in | 12,969 | 955 | 0.493 | -1 |
| y2_neg_2of3 | mom_crash40 | 12,969 | 955 | 0.485 | 1 |


## 21. Mechanical acf3 (overlapping 3m windows)

growth_3 acf3=-0.433; non-overlapping (so_far%3==0) acf1=-0.381. a_in3 acf1/acf3 0.655/-0.091; a_op_in acf1 0.004. MECHANICAL — persistent in3 makes overlapping growth_3 flip at lag 3.

## 22. Company-mean leftover (style, not a shock)

Company-mean Y3: io 0.439 g3 0.664 g12 0.647. g3 after co-mean days/size/both 0.569/0.537/0.531. Company-mean g3 beats size — STYLE, not a month shock. Do not KEEP. Style leftover dies after co-mean size+days.

| feature | co-mean | after co-days | after co-size | after both |
| --- | --- | --- | --- | --- |
| a_io_ratio | 0.439 | 0.526 | 0.733 | 0.594 |
| a_growth_3 | 0.664 | 0.569 | 0.537 | 0.531 |
| a_growth_12 | 0.647 | 0.592 | 0.574 | 0.569 |


## 23. Y3 positives missing growth_3

Y3 positives missing growth_3: 137 / 402 (34.1%). Short-book / zero in3_{t-3} hole, not a sign flip.

| feature | Y3 labeled | Y3 pos | pos with X | pos missing X | miss share | miss so_far p50 |
| --- | --- | --- | --- | --- | --- | --- |
| a_io_ratio | 5648 | 402 | 391 | 11 | 2.7% | 2.0 |
| a_growth_3 | 5648 | 402 | 265 | 137 | 34.1% | 5.0 |
| a_growth_12 | 5648 | 402 | 79 | 323 | 80.3% | 8.0 |


## 24. Rank leftover (nonlinear SIZE leak)

io OLS leftover 0.604 ρ(resid,size)=-0.643; rank leftover 0.533 vs size 0.617. Nonlinear SIZE leak — the 0.604 is not leftover Q3.

| feature | OLS leftover | rank leftover | ρ(OLS,size) | size |
| --- | --- | --- | --- | --- |
| a_io_ratio | 0.604 | 0.533 | -0.643 | 0.617 |
| a_growth_3 | 0.526 | 0.510 | -0.034 | 0.617 |
| a_growth_12 | 0.397 | 0.401 | 0.005 | 0.617 |


## 25. Dark growth_12 hole is trail length

Dark trail p50 16.5 vs invoiced 20.0. Company g12 coverage vs trail length ρ=0.897. Dark g12 hole is shorter books, not a dark-specific growth object.

## 26. Holdout late-arrival (hidden 72)

Holdout companies late-arrival 69/72; short-trail 29/72. growth_12 cov late 14.7% vs on-time 41.7%. LOCK — hidden 72 is almost all late-arrival; growth_12 CLOSE as Q6.

## 27. io Q1 vs clip=3 tails

Y3 on io Q1 11.2% (days p50 13.0) vs clip=3 12.7% (days p50 12.0). Q1 is the quiet-days pile already on the card.

| slice | n | n_pos | P(Y3) | days p50 |
| --- | --- | --- | --- | --- |
| io Q1 (low coverage) | 1106 | 124 | 11.2% | 13.0 |
| io mid Q2–Q4 | 3316 | 180 | 5.4% | 19.0 |
| io clip=3 | 330 | 42 | 12.7% | 12.0 |


## 28. MoM leftover + non-overlapping growth_3

MoM inflow Y3 0.600 leftover size/days/both 0.590/0.563/0.563 vs size 0.617 / days 0.711. Non-overlapping growth_3 0.521 vs size 0.631. MoM leftover dies — still DROP.

| feature | raw | after size | after days | after both | size | days |
| --- | --- | --- | --- | --- | --- | --- |
| mom_in | 0.600 | 0.590 | 0.563 | 0.563 | 0.617 | 0.711 |
| g3 so_far%3==0 | 0.521 | — | — | — | 0.631 | 0.714 |


## What failed / next (held for wave note)

- none of io / g3 / g12 KEEP on the 44 — leftover dies, raw loses to size 0.617 / days 0.711
- io leftover-after-size 0.604 is a nonlinear SIZE leak (rank leftover 0.533)
- growth_3 acf3=-0.433 is mechanical window overlap (levels persist)
- growth_3 / io quintiles are U-shaped two-tail, not monotone Q3 turning
- growth_12 CLOSE as Q6: empty on short books; holdout late-arrival 69/72
- company-mean g3 0.664 is STYLE; leftover both 0.531 dies
- MoM leftover both 0.563 dies vs size; non-overlap g3 0.521

Elapsed 10s. Cuts: coverage, formula, Spearman, singles, leftover, mean-reversion, clip-at-3, dark 470/744, 12 names, ICC, Q6, Y4 overlap, ±1 clip, size terciles, holdout, calendar, fake leftover, long-book, U-shape/T1, MoM −40%, mechanical acf3, company-mean style, Y3 hole, rank leftover, dark trail, holdout late, io tails, MoM leftover / non-overlap.
