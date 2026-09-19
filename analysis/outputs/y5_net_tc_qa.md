# Y5 leftover 65% as net TC × activity (Bureau diagnostic)

Generated `2026-09-19T09:10:03+02:00` by agent `689100e7`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. **Never E as Y5 X.** Diagnostic leftover only. Night Y7 stays **TURNOVER 0.720 / 0.712**. Do **not** grow TURNOVER. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0.

`diag_net_tc` = `e_ap_open − e_ar_open`. Interaction with own-p20 days / inflow shock. Bureau–Duquerroy–Vinas: 1 SD net TC × lockdown → +10% payment-default PD only in shock months. Seat is the AP neither cell (night **222/341 = 65.1%**).

## Headline

diag_net_tc defined 64.1%; dark 470 NaN CONFIRM. Neither 222/341 (65.1%; night 65.1%). Y5 diag_net_x_shock raw 0.468 vs size 0.556 beat-size FAIL. Leftover after size+days rank 0.534 OLS 0.516 (neither 0.450; plain net 0.466). Shock 0.524 vs quiet 0.466. Boot p05 0.469. Y5 leftover **CLOSE**. Never E-on-Y5. Do not grow TURNOVER 0.720. 65% sentence stays if this dies.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK**. Y5 is a turn, not a health Y. |
| 2 | Who is improving? | Not a recovery clock. |
| 3 | Who is turning? | Y5 is the turn. This cut asks *why the leftover 65%*. |
| 4 | Dip vs fall? | Diagnostic leftover **CLOSE** — leftover 0.534 dies — Y5 stays a 65% leftover (neither 222/341) |
| 5 | Why did it change? | Bureau: net TC only bites in an activity shock. |
| 6 | Months earlier? | **CLOSE** — leftover is contemporaneous; d_tx Q6 already CLOSE. |

## PARK / CLOSE / KEEP / DROP

| object | decision | why |
| --- | --- | --- |
| net TC × shock leftover after size+days (Y5 diagnostic) | **CLOSE** | leftover 0.534 dies — Y5 stays a 65% leftover (neither 222/341) |
| as Y5 engine X (Family E) | **DROP** | Y5 never E |
| TURNOVER add-on / 15-col card | **CLOSE** | do not grow 0.720 |
| d_tx as leave-one-group law | **PARK** | night 0.611 PARK |
| d_supp_hhi as engine X | **DROP** | protective tail 2.7% vs 8.6% already measured |

## 1. Coverage / neither cell / dark 470

Train CM=21,157. diag_net_tc defined 13,554 (64.1%). Dark 470 nn=0 zero=0 CONFIRM NaN not 0. AP 2×2 positives 341 neither 222 (65.1%; night 222/341=65.1%).

| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| diag_net_tc defined | 13,554 (64.1%) |
| mean / p50 net TC | -19133850.841 / 0.000 |
| shock-month share | 29.4% |
| Y5 labeled / pos | 4,905 / 418 |
| 2×2 pos / neither | 341 / 222 |
| neither share | 65.1% |
| dark 470 nn / zero | 0 / 0 |
| dark NaN | CONFIRM |

## 2. Spearman twins (|ρ|≥0.80)

Gate twins vs size/days/HHI/io/d_tx: none. ρ vs days=-0.059 size=-0.101 supp_hhi=0.121 d_tx=-0.027.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| a_in3 | -0.101 | 9,834 |  |
| c_n_days_with_tx | -0.059 | 9,834 |  |
| d_supp_hhi | 0.121 | 8,963 |  |
| a_io_ratio | 0.016 | 9,834 |  |
| d_tx_cp_share | -0.027 | 9,758 |  |
| log_in3 | -0.101 | 9,834 |  |
| e_ap_open | -0.012 | 9,834 |  |
| e_ar_open | -0.324 | 9,834 |  |
| diag_net_tc | 0.525 | 9,834 |  |


## 3. Single-feature train group-fold AUROC (diagnostic)

Days **0.711** is the Y3 bar, not a Y5 engine. Size **0.617**. d_tx night **0.611 PARK**. TURNOVER **0.720** unchanged.

Y5 diag_net_x_shock 0.468 vs size 0.556 days 0.540 d_tx 0.532 (AR-night 0.611 PARK). beat-size FAIL.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ap_od30_ownp80 | net_tc | 4,905 | 418 | 0.465 | 0.051 | 1 | 0.533 0.474 0.428 0.402 0.486 |
| y5_ap_od30_ownp80 | net_x_days | 4,905 | 418 | 0.477 | 0.018 | 1 | 0.480 0.477 0.453 0.472 0.502 |
| y5_ap_od30_ownp80 | net_x_in | 4,574 | 385 | 0.469 | 0.034 | 1 | 0.501 0.488 0.431 0.434 0.494 |
| y5_ap_od30_ownp80 | net_x_shock | 4,905 | 418 | 0.468 | 0.047 | 1 | 0.531 0.473 0.418 0.427 0.491 |
| y5_ap_od30_ownp80 | ap_x_shock | 4,905 | 418 | 0.512 | 0.025 | -1 | 0.488 0.487 0.511 0.541 0.534 |
| y5_ap_od30_ownp80 | ar_x_shock | 4,905 | 418 | 0.482 | 0.028 | 1 | 0.478 0.486 0.491 0.438 0.515 |
| y5_ap_od30_ownp80 | size | 4,905 | 418 | 0.556 | 0.084 | 1 | 0.674 0.539 0.466 0.495 0.606 |
| y5_ap_od30_ownp80 | days | 4,905 | 418 | 0.540 | 0.083 | 1 | 0.641 0.449 0.473 0.530 0.607 |
| y5_ap_od30_ownp80 | d_tx | 4,894 | 418 | 0.532 | 0.095 | -1 | 0.581 0.384 0.489 0.604 0.600 |
| y5_ap_od30_ownp80 | h_group | 4,905 | 418 | 0.581 | 0.086 | -1 | 0.517 0.647 0.488 0.563 0.693 |
| y5_ap_od30_ownp80 | io | 4,905 | 418 | 0.525 | 0.039 | -1 | 0.553 0.477 0.564 0.543 0.489 |
| y5_ap_od30_ownp80 | hhi | 4,902 | 418 | 0.539 | 0.044 | -1 | 0.536 0.575 0.533 0.470 0.579 |


## 4. Residual Y5 after size+days (KEEP gate)

KEEP diagnostic footnote only if rank ≥ **0.58**, not a twin, not SIZE, beat-size. Never E as Y5 X. Rank leftover is honest; OLS can fake.

Y5 leftover of diag_net_x_shock after size+days rank 0.534 OLS 0.516 dies. Neither-cell leftover 0.450 n_pos=222. Plain net_tc after size+days 0.466.

| cut | rank | OLS | ρ(resid,ctrl) | R² | n | n_pos | dies? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after size+days | 0.534 | 0.516 | 0.481 | 0.008 | 4,905 | 418 | dies |
| after size | 0.534 | 0.570 | 0.948 | 0.005 | 4,905 | 418 | dies |
| after days | 0.521 | 0.525 | -0.908 | 0.000 | 4,905 | 418 | dies |
| after io+hhi | 0.522 | 0.549 | 0.084 | 0.001 | 4,902 | 418 | dies |
| after d_tx | 0.494 | 0.539 | -0.946 | 0.001 | 4,894 | 418 | dies |
| net_tc after size+days | 0.466 | 0.517 | 0.551 | 0.016 | 4,905 | 418 | dies |
| neither after size+days | 0.450 | 0.458 | 0.481 | 0.008 | 2,379 | 222 | dies |
| neither net_tc after size+days | 0.432 | 0.458 | 0.551 | 0.016 | 2,379 | 222 | dies |


## 5. Extra — shock vs quiet; AP vs AR sign

Shock-month net_tc leftover 0.524; quiet 0.466. AP×shock 0.408; −AR×shock 0.428.

| cut | rank | OLS | n | n_pos | dies? |
| --- | --- | --- | --- | --- | --- |
| net_tc on shock months | 0.524 | 0.525 | 1,565 | 119 | dies |
| net_tc on quiet months | 0.466 | 0.496 | 3,340 | 299 | dies |
| net×shock on shock months | 0.523 | 0.535 | 1,565 | 119 | dies |
| AP×shock on shock months | 0.408 | 0.526 | 1,565 | 119 | dies |
| −AR×shock on shock months | 0.428 | 0.536 | 1,565 | 119 | dies |
| net_tc on neither+shock | — | — | 545 | 44 | lives |
| net_tc on neither+quiet | 0.470 | 0.467 | 1,834 | 178 | dies |


## 6. Y3 off-card + dark/holdout coverage

Y3 leftover after days 0.624 — off the 15-col card. Holdout 72 coverage only. Holdout diag_net_tc nn=582. Holdout dark nn=0 (want 0).

| cut | rank | OLS | dies? |
| --- | --- | --- | --- |
| Y3 leftover after days | 0.624 | 0.718 | dies |


| slice | n_co | nn | zero |
| --- | --- | --- | --- |
| train dark | 470 | 0 | 0 |
| holdout (coverage only) | 72 | 582 | 35 |
| holdout dark | 32 | 0 | 0 |


## 7. Quintiles

Y5 rate by rank-quintile of net TC × shock. Q5−Q1 1.6% (Bureau seat).

| q | n | rate | p50 net×shock |
| --- | --- | --- | --- |
| Q1 | 981 | 6.8% | -207497.440 |
| Q2 | 981 | 9.0% | 0.000 |
| Q3 | 981 | 7.5% | 0.000 |
| Q4 | 981 | 10.8% | 0.000 |
| Q5 | 981 | 8.5% | 25832.630 |


Plot: `y5_net_tc_qa.png`.

## 8. Company bootstrap leftover after size+days

Bootstrap leftover-after-size+days rank p05=0.469 p50=0.539 p95=0.562 share<0.55=70.0% n=30.

## 9. Extra — alternate shocks / fold 3 / neither after cash+HHI

Inflow-dip leftover 0.539; neither after io+hhi 0.441. Fold 3 raw diag_net_x_shock 0.427 — not a d_tx leave-one-group revival.

| cut | rank | OLS | n | n_pos | dies? |
| --- | --- | --- | --- | --- | --- |
| net × inflow-dip after size+days | 0.539 | 0.543 | 4,761 | 404 | dies |
| net_tc after io+hhi on neither | 0.441 | 0.482 | 2,379 | 222 | dies |
| net×shock after io+hhi | 0.522 | 0.549 | 4,902 | 418 | dies |
| net × Y2 after size+days | 0.473 | 0.497 | 4,904 | 418 | dies |


## What this note did not do

- Did not change Y7 0.720 / 0.712 or Y3 0.762 / 0.752.
- Did not put Family E on the Y5 card. Did not grow TURNOVER.
- Did not overwrite y5_why / supp_hhi_qa / d_tx_qa / delay_qa / issued_qa.
- Did not score this as Y3 B or Y7 D. Dark 470 stayed NaN.
