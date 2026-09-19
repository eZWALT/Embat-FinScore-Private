# `c_gap_sd` — twin of days, NEAR SIZE, or Pérez-Salazar leftover?

Generated `2026-09-19T04:23:33+02:00` by agent `87e59905`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_gap_sd`. Do not put `c_gap_sd` on the 15-col card. Night Y3 quote stays **0.762 / 0.752**. Days bar stays **0.711**.

`c_gap_sd` = sample stdev (ddof=1) of day-gaps between consecutive **unique calendar days** in (month_end − 90d, month_end]. Null if fewer than 3 distinct days. Pérez-Salazar, Márquez & Vidal-Silva 2026 (Computers 15:135) σ_Δt. Unique-day collapse: 94% of timestamps are midnight.

## Headline

`c_gap_sd` train cov 95.1% (CONFIRM 95.1%). ρ vs days -0.905 / a_n_tx -0.866 / c_n_tx -0.866 (twin=True); vs log1p(a_in3) -0.532 (CONFIRM −0.543). Y3 gap 0.686 vs size 0.617 vs days 0.711. Leftover after days 0.535 (die=True). ICC 0.963 (TRAIT). X **DROP-from-44** (a). PARK as Y. Q6 **CLOSE** (lag leftover 0.483).

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as a health Y. Do not invent `y_gap_sd`. Regular vs irregular booker is a BETWEEN trait (ICC 0.96). |
| 2 | Who is improving? | Not this column. A 90d σ is not a recovery path. |
| 3 | Who is turning? | Y3 gap 0.686 vs days 0.711 vs size 0.617. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | **CLOSE** — not sayable without “fewer booking days” / size / twin. |
| 6 | Months earlier? | **CLOSE** — Y3 gap_sd_lag1 0.669 vs days_lag1 0.684; lag leftover after days_lag1 0.483 (R²=0.414). CLOSE as Q6 — the lag is the days twin lagged, not a new lead. |

## Who is right — feature report vs Y3 engine?

**DROP-from-44** as Y3 X. Letter **(a)**. pairwise |ρ|≥0.80 twin of c_n_days_with_tx, a_n_tx, c_n_tx (days -0.905). Y3 gap 0.686 vs days 0.711 (night 0.711). Leftover after days 0.535 dies. Keep days on the card. **DROP from the 44** / **CLOSE as X**.

| letter | claim | this cut |
| --- | --- | --- |
| (a) | days / n_tx twin (\|ρ\|≥0.80) — drop from the 44, keep days on the card | days/n_tx twin |
| (b) | NEAR SIZE (ρ −0.543) leftover that dies inside size terciles | SIZE ρ −0.53 is real; tercile 'survival' is the days engine, not leftover regularity |
| (c) | Pérez-Salazar regularity leftover after days (KEEP later, not on tonight's card) | no leftover after days |
| (d) | 90-day window artifact / short-book NaN | not a short-book NaN hole |
| (e) | midnight-timestamp hole the unique-day collapse was meant to fix | midnight hole — collapse CONFIRMED (raw 1/n_tx); unique-day still a days twin |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `c_gap_sd` as Y3 X / on the 44 | **DROP-from-44** | pairwise |ρ|≥0.80 twin of c_n_days_with_tx, a_n_tx, c_n_tx (days -0.905). Y3 gap 0.686 vs days 0.711 (night 0.711). Leftover after days 0.535 dies. Keep days on the card. **DROP from the 44** / **CLOSE as X**. |
| `c_gap_sd` on tonight's 15-col card | **no** | days 0.711 stays the engine; night quote 0.762 / 0.752 unchanged |
| `c_gap_sd` as a health Y | **PARK** | do not invent `y_gap_sd` |
| leftover after days | **dies / CLOSE as twin** | Y3 resid 0.535 vs size 0.617 |
| Q6 lag1/lag3 | **CLOSE** | Y3 gap_sd_lag1 0.669 vs days_lag1 0.684; lag leftover after days_lag1 0.483 (R²=0.414). CLOSE as Q6 — the lag is the days twin lagged, not a new lead. |
| Q5 footnote | **CLOSE** | not sayable without “fewer booking days” / size / twin. |
| unique-day collapse | **CONFIRM** | Midnight share 94.3% (n_tx=2,556,068) (CONFIRM 94%). Raw (no unique-day) σ vs 1/c_n_tx ρ=0.916, vs c_n_tx -0.916, vs a_n_tx -0.916. Unique-day σ vs 1/c_n_tx ρ=0.867. store↔raw 0.967. Y3 raw CV 0.681 vs unique 0.686. CONFIRM the collapse — raw σ is a 1/n_tx twin; unique-day was the right fix. |

## 1. Completeness — null share (days<3), train vs holdout

Train `c_gap_sd` coverage 95.1% (n_null=1,038 / 21,157) (CONFIRM 95.1%). Holdout coverage only 96.4% on 1,073 CM / 72 cos. In-month days<3 share 12.6%; P(null|days<3)=39.0%; P(days<3|null)=100.0%. Null is the 90d unique-day clock (<3 distinct days), not the in-month count.

| split | n_cm | n_co | cov | n_null | null | in-month days<3 | p50 | p90 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,157 | 1,214 | 95.1% | 1,038 | 4.9% | 12.6% | 1.356 | 6.324 |
| holdout | 1,073 | 72 | 96.4% | 39 | 3.6% | 8.9% | 1.278 | 4.733 |


Confirm feature-report 95.1%: **YES**.

## 2. Formula vs unique booking dates (90d, ddof=1)

Recompute unique-day σ (90d, ddof=1) vs store: both-defined 20,119, both-null 1,038, store-only 0, recon-only 0. max|Δ|=0.00e+00 median|Δ|=0.00e+00. Window unique-days <3: 1,038 (4.9%). Formula OK — matches ops.py.

Sample company-months (recon = in-memory unique-day σ, same window as ops.py):

| company_id | period | store | recon | n_win90 | days_m | so_far |
| --- | --- | --- | --- | --- | --- | --- |
| COMP_0001 | 2026-01 | 1.4142 | 1.4142 | 3 | 3 | 1 |
| COMP_0218 | 2026-01 | 3.2026 | 3.2026 | 28 | 12 | 13 |
| COMP_0438 | 2025-06 | 1.1721 | 1.1721 | 46 | 12 | 10 |
| COMP_0645 | 2025-10 | 4.0000 | 4.0000 | 8 | 2 | 9 |
| COMP_0848 | 2024-10 | 0.8653 | 0.8653 | 42 | 22 | 2 |
| COMP_1066 | 2026-07 | 0.5376 | 0.5376 | 76 | 24 | 13 |
| COMP_0002 | 2025-07 | — | — | 0 | 0 | 11 |
| COMP_0325 | 2026-05 | — | — | 2 | 0 | 21 |
| COMP_0611 | 2025-06 | — | — | 2 | 0 | 9 |
| COMP_0962 | 2026-07 | — | — | 0 | 0 | 12 |


## 3. Spearman — twin / SIZE / recency

Train Spearman `c_gap_sd` vs days -0.905, a_n_tx -0.866, c_n_tx -0.866 (twin |ρ|≥0.80: YES c_n_days_with_tx,a_n_tx,c_n_tx). vs log1p(a_in3) -0.532; vs log1p(|a_op_in|) -0.543 (CONFIRM −0.543). SIZE |ρ|≥0.50: YES. vs recency 0.462. Company-median vs days -0.952 / a_n_tx -0.903 / size -0.585. days↔a_n_tx 0.938 a_n_tx↔c_n_tx 1.000.

Twin if |ρ|≥0.80 vs days / a_n_tx / c_n_tx. SIZE if |ρ| vs log size ≥0.50.

| pair | Spearman | flag |
| --- | --- | --- |
| c_gap_sd vs c_n_days_with_tx | -0.905 | TWIN |
| c_gap_sd vs a_n_tx | -0.866 | TWIN |
| c_gap_sd vs c_n_tx | -0.866 | TWIN |
| c_gap_sd vs log1p(a_in3) | -0.532 | SIZE |
| c_gap_sd vs log1p(|a_op_in|) | -0.543 | SIZE |
| c_gap_sd vs c_recency_days | 0.462 |  |
| c_gap_sd vs 1/c_n_tx | 0.867 |  |
| c_gap_sd vs 1/a_n_tx | 0.867 |  |
| c_gap_sd vs 1/c_n_days_with_tx | 0.905 |  |
| c_gap_sd vs so_far | 0.015 |  |
| c_gap_sd vs months_on_book | 0.026 |  |
| days↔a_n_tx | 0.938 | CHAIN |
| days↔c_n_tx | 0.938 | CHAIN |
| a_n_tx↔c_n_tx | 1.000 | CHAIN |


## 4. Single-feature train group-fold AUROC

Y3 stressed n=5,648 base 7.1%; Y2 n=17,356 base 7.3%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711); size 0.617 (replica 0.617).

Y3 stressed singles: `c_gap_sd` 0.686 vs size 0.617 (quote 0.617 CONFIRM, Δ 0.070) vs days 0.711 (night 0.711 CONFIRM, Δ -0.025) vs a_n_tx 0.703. Y2 gap 0.577 vs size 0.552 / days 0.571. recency Y3 0.659. Fold wander 0.090. Loses to days 0.711.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_gap_sd | 5,574 | 398 | 0.686 | 0.038 | 1 | 0.693 | 0.630 0.721 0.690 0.672 0.720 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | c_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_recency_days | 5,648 | 402 | 0.659 | 0.051 | 1 | 0.665 | 0.590 0.666 0.666 0.730 0.642 |
| y3_recover_cash_6m | inv_c_n_tx | 5,648 | 402 | 0.703 | 0.034 | 1 | 0.714 | 0.652 0.733 0.697 0.699 0.737 |
| y2_neg_2of3 | c_gap_sd | 16,559 | 1,207 | 0.577 | 0.040 | -1 | 0.585 | 0.620 0.519 0.609 0.569 0.568 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | c_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | c_recency_days | 17,356 | 1,271 | 0.546 | 0.039 | -1 | 0.541 | 0.557 0.572 0.580 0.540 0.483 |
| y2_neg_2of3 | inv_c_n_tx | 17,356 | 1,271 | 0.599 | 0.044 | -1 | 0.601 | 0.641 0.570 0.648 0.584 0.549 |


KEEP-as-X: leftover after days beats size by ≥0.02 **and** leftover after days **and** not SIZE. Size dummy ≥0.6 is a PARK flag, not a KEEP.

## 5. Residual AUROC after days / a_n_tx

Y3 leftover after days 0.535 (R²=0.414, slope=-0.267); after a_n_tx 0.659 (R²=0.067); after both 0.543 vs size 0.617 (Δ -0.081). Leftover dies — CLOSE as twin — **drop gap_sd from the 44**.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| Y3 | c_gap_sd | 5,574 | 398 | 0.686 | 1 |
| Y3 | resid_days | 5,574 | 398 | 0.535 | -1 |
| Y3 | resid_a_n_tx | 5,574 | 398 | 0.659 | 1 |
| Y3 | resid_days+a_n_tx | 5,574 | 398 | 0.543 | -1 |
| Y3 | resid_recency | 5,574 | 398 | 0.628 | 1 |
| Y3 | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 |
| Y3 | log1p_a_in3 | 5,528 | 391 | 0.617 | -1 |
| Y2 | c_gap_sd | 16,559 | 1,207 | 0.577 | -1 |
| Y2 | resid_days | 16,559 | 1,207 | 0.521 | 1 |
| Y2 | resid_a_n_tx | 16,559 | 1,207 | 0.557 | -1 |
| Y2 | resid_days+a_n_tx | 16,559 | 1,207 | 0.520 | 1 |
| Y2 | resid_recency | 16,559 | 1,207 | 0.568 | -1 |
| Y2 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 1 |
| Y2 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 1 |


OLS R² gap~days 0.414 slope=-0.267; gap~a_n_tx 0.067 slope=-0.004; both 0.420.

## 6. SIZE terciles — gap_sd vs size inside T1 / T2 / T3

Y3 gap_sd vs size inside company size terciles that beat by ≥0.02: ['T1', 'T2', 'T3']. Beats size inside terciles — but days beats gap in the same slices, so the survival is the days engine.

| clock | tercile | n | n_pos | gap | size | days | Δsize |
| --- | --- | --- | --- | --- | --- | --- | --- |
| size_t | T1 | 1,507 | 266 | 0.574 | 0.427 | 0.603 | 0.147 |
| size_t | T2 | 2,039 | 72 | 0.654 | 0.634 | 0.673 | 0.020 |
| size_t | T3 | 2,028 | 60 | 0.731 | 0.620 | 0.733 | 0.111 |
| days_t | T1 | 1,510 | 268 | 0.547 | 0.531 | 0.590 | 0.016 |
| days_t | T2 | 2,106 | 78 | 0.588 | 0.633 | 0.608 | -0.045 |
| days_t | T3 | 1,958 | 52 | 0.696 | LOW_POWER | 0.676 | — |


## 7. ICC / company-demean (feature-report BETWEEN 0.96)

`c_gap_sd` ICC=0.963 w/b=0.039 (CONFIRM BETWEEN 0.96); days ICC=0.985. acf1=0.610 acf3=-0.035 acf6=-0.072 (feature-report acf1=0.61; the 0.96 in the brief is the ICC, not acf). Y3 raw 0.686 vs company-demean 0.566 (drop 0.121). TRAIT (regular vs irregular booker).

| item | value |
| --- | ---: |
| ICC `c_gap_sd` | 0.963 |
| ICC `c_n_days_with_tx` | 0.985 |
| w/b | 0.039 |
| acf1 / acf3 / acf6 | 0.610 / -0.035 / -0.072 |
| Y3 raw / demean | 0.686 / 0.566 |
| trait / shock | YES / no |

## 8. Nulls — Y3 rate on gap_sd-null vs defined

Y3 rate on gap_sd-null 5.4% (n_lab=74, n_pos=4) vs defined 7.1%. Null days p50=0.0 so_far p50=11.5 months-on-book p50=23.0. Nulls are thin 90d books (unique days <3).

| y | slice | n_cm | n_lab | n_pos | rate | days p50 | so_far p50 | mob p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | defined | 20,119 | 16,559 | 1,207 | 7.3% | 15.0 | 9.0 | 23.0 |
| y2_neg_2of3 | null | 1,038 | 797 | 64 | 8.0% | 0.0 | 11.5 | 23.0 |
| y3_recover_cash_6m | defined | 20,119 | 5,574 | 398 | 7.1% | 15.0 | 9.0 | 23.0 |
| y3_recover_cash_6m | null | 1,038 | 74 | 4 | 5.4% | 0.0 | 11.5 | 23.0 |


Unique days in the 90d window (train CM):

| n_unique_90d | n_cm | share | null_share | short_book |
| --- | --- | --- | --- | --- |
| 0 | 391 | 1.8% | 100.0% | 2.3% |
| 1 | 324 | 1.5% | 100.0% | 15.1% |
| 2 | 323 | 1.5% | 100.0% | 16.1% |
| ≥3 | 20119 | 95.1% | 0.0% | 14.2% |


## 9. Dark 470 vs invoiced 744

Train last-month: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). gap_sd p50 invoiced 1.430 vs dark 1.202; cov 95.5% / 94.4%. Y3 gap CV invoiced 0.690 vs dark 0.696. Same bank-book regularity. Holdout ever-ERP coverage only: 40/72.

| group | n_cm | n_co | cov | gap p50 | days p50 |
| --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 13,554 | 744 | 95.5% | 1.430 | 14.0 |
| never_erp_470 | 7,603 | 470 | 94.4% | 1.202 | 15.0 |


## 10. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)

Chronic 12 names (0158/0172, ≥50% labeled months below 0): 12. Y2 gap CV all 0.577 → drop-12 0.556. gap_sd p50 on 12 0.807 vs rest 1.374. Drop does not flip Y2 (≥0.03).

| y | slice | n | n_pos | gap | days | size |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | all | 16,559 | 1,207 | 0.577 | 0.571 | 0.552 |
| y2_neg_2of3 | drop_12 | 16,343 | 1,030 | 0.556 | 0.549 | 0.545 |
| y2_neg_2of3 | chronic_12 | 216 | 177 | 0.523 | 0.535 | 0.514 |
| y3_recover_cash_6m | all | 5,574 | 398 | 0.686 | 0.711 | 0.617 |
| y3_recover_cash_6m | drop_12 | 5,412 | 398 | 0.682 | 0.708 | 0.617 |
| y3_recover_cash_6m | chronic_12 | 162 | 0 | LOW_POWER | LOW_POWER | LOW_POWER |


## 11. Q6 — lag1/lag3 on short vs long (days lag1 is a night KEEP)

Y3 now 0.686 vs lag1 0.669 (Δ 0.017); short lag1 0.681. Days lag1 replica 0.684 (night KEEP). Gap_sd lag holds.

| y | slice | col | n | n_pos | present | CV | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all | c_gap_sd | 5,574 | 398 | 98.7% | 0.686 | 1 |
| y3_recover_cash_6m | all | c_gap_sd_lag1 | 5,542 | 390 | 98.1% | 0.669 | 1 |
| y3_recover_cash_6m | all | c_gap_sd_lag3 | 4,932 | 336 | 87.3% | 0.659 | 1 |
| y3_recover_cash_6m | all | c_n_days_with_tx | 5,648 | 402 | 100.0% | 0.711 | -1 |
| y3_recover_cash_6m | all | c_n_days_with_tx_lag1 | 5,648 | 402 | 100.0% | 0.684 | -1 |
| y3_recover_cash_6m | all | c_n_days_with_tx_lag3 | 5,078 | 355 | 89.9% | 0.666 | -1 |
| y3_recover_cash_6m | short_<12 | c_gap_sd | 3,684 | 249 | 99.0% | 0.691 | 1 |
| y3_recover_cash_6m | short_<12 | c_gap_sd_lag1 | 3,653 | 243 | 98.1% | 0.681 | 1 |
| y3_recover_cash_6m | short_<12 | c_gap_sd_lag3 | 3,042 | 192 | 81.7% | 0.688 | 1 |
| y3_recover_cash_6m | short_<12 | c_n_days_with_tx | 3,723 | 252 | 100.0% | 0.696 | -1 |
| y3_recover_cash_6m | short_<12 | c_n_days_with_tx_lag1 | 3,723 | 252 | 100.0% | 0.684 | -1 |
| y3_recover_cash_6m | short_<12 | c_n_days_with_tx_lag3 | 3,153 | 205 | 84.7% | 0.691 | -1 |
| y3_recover_cash_6m | long_>=18 | c_gap_sd | 208 | 16 | 97.7% | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_gap_sd_lag1 | 208 | 16 | 97.7% | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_gap_sd_lag3 | 209 | 16 | 98.1% | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_n_days_with_tx | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_n_days_with_tx_lag1 | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover_cash_6m | long_>=18 | c_n_days_with_tx_lag3 | 213 | 16 | 100.0% | LOW_POWER | — |
| y2_neg_2of3 | all | c_gap_sd | 16,559 | 1,207 | 95.4% | 0.577 | -1 |
| y2_neg_2of3 | all | c_gap_sd_lag1 | 15,417 | 1,089 | 88.8% | 0.582 | -1 |
| y2_neg_2of3 | all | c_gap_sd_lag3 | 13,126 | 889 | 75.6% | 0.596 | -1 |
| y2_neg_2of3 | all | c_n_days_with_tx | 17,356 | 1,271 | 100.0% | 0.571 | 1 |
| y2_neg_2of3 | all | c_n_days_with_tx_lag1 | 16,161 | 1,151 | 93.1% | 0.575 | 1 |
| y2_neg_2of3 | all | c_n_days_with_tx_lag3 | 13,776 | 938 | 79.4% | 0.583 | 1 |
| y2_neg_2of3 | short_<12 | c_gap_sd | 10,688 | 827 | 95.5% | 0.559 | -1 |
| y2_neg_2of3 | short_<12 | c_gap_sd_lag1 | 9,520 | 710 | 85.1% | 0.561 | -1 |
| y2_neg_2of3 | short_<12 | c_gap_sd_lag3 | 7,175 | 512 | 64.1% | 0.566 | -1 |
| y2_neg_2of3 | short_<12 | c_n_days_with_tx | 11,186 | 885 | 100.0% | 0.544 | 1 |
| y2_neg_2of3 | short_<12 | c_n_days_with_tx_lag1 | 9,991 | 765 | 89.3% | 0.545 | 1 |
| y2_neg_2of3 | short_<12 | c_n_days_with_tx_lag3 | 7,606 | 552 | 68.0% | 0.546 | 1 |
| y2_neg_2of3 | long_>=18 | c_gap_sd | 1,786 | 118 | 93.6% | 0.651 | -1 |
| y2_neg_2of3 | long_>=18 | c_gap_sd_lag1 | 1,797 | 117 | 94.2% | 0.664 | -1 |
| y2_neg_2of3 | long_>=18 | c_gap_sd_lag3 | 1,828 | 117 | 95.8% | 0.682 | -1 |
| y2_neg_2of3 | long_>=18 | c_n_days_with_tx | 1,908 | 119 | 100.0% | 0.680 | 1 |
| y2_neg_2of3 | long_>=18 | c_n_days_with_tx_lag1 | 1,908 | 119 | 100.0% | 0.682 | 1 |
| y2_neg_2of3 | long_>=18 | c_n_days_with_tx_lag3 | 1,908 | 119 | 100.0% | 0.697 | 1 |


## 12. Same-day flood — σ without unique-day collapse (in-memory)

Midnight share 94.3% (n_tx=2,556,068) (CONFIRM 94%). Raw (no unique-day) σ vs 1/c_n_tx ρ=0.916, vs c_n_tx -0.916, vs a_n_tx -0.916. Unique-day σ vs 1/c_n_tx ρ=0.867. store↔raw 0.967. Y3 raw CV 0.681 vs unique 0.686. CONFIRM the collapse — raw σ is a 1/n_tx twin; unique-day was the right fix.

| item | value |
| --- | ---: |
| midnight share | 94.3% |
| raw σ vs 1/c_n_tx | 0.916 |
| raw σ vs c_n_tx | -0.916 |
| unique σ vs 1/c_n_tx | 0.867 |
| store ↔ raw | 0.967 |
| Y3 raw / unique | 0.681 / 0.686 |

Do **not** rewrite ops.py. The collapse stays.

## Extra 13 — 90d window vs months-on-book

Coverage so-far short 95.8% vs long 92.5%. Y3 gap short 0.691 vs long LOW_POWER; days short 0.696. 90d coverage is not a short-book hole — nulls are thin unique-day books, not late arrivals only.

| so_far | n_cm | cov | p50 | n_null |
| --- | --- | --- | --- | --- |
| <6 | 6,068 | 93.5% | 1.357 | 395 |
| 6-11 | 6,406 | 98.1% | 1.340 | 124 |
| 12-17 | 4,740 | 95.3% | 1.404 | 223 |
| 18-23 | 3,508 | 92.7% | 1.333 | 255 |
| 24 | 435 | 90.6% | 1.288 | 41 |


## Extra 14 — lifetime unique-day σ vs 90d (in-memory)

Lifetime unique-day σ (as-of month-end, no 90d cut) vs store 90d ρ=0.921 (same object — 90d is a trait window). Y3 full-hist 0.678; 90d leftover after lifetime 0.639 (R²=0.432).

## Extra 15 — Y3 quintiles

Y3 rate gap_sd Q1 2.9% → Q5 16.6%; days Q1 17.0% → Q5 2.4%. Quiet / regular Q1 vs busy Q5 is the days story if the shapes match.

| feature | q | n | n_pos | rate | x p50 |
| --- | --- | --- | --- | --- | --- |
| c_gap_sd | Q1 | 1,116 | 32 | 2.9% | 0.322 |
| c_gap_sd | Q5 | 1,115 | 185 | 16.6% | 5.466 |
| c_gap_sd | Q4 | 1,115 | 78 | 7.0% | 1.977 |
| c_gap_sd | Q2 | 1,119 | 47 | 4.2% | 0.801 |
| c_gap_sd | Q3 | 1,109 | 56 | 5.0% | 1.111 |
| days | Q5 | 975 | 23 | 2.4% | 28.000 |
| days | Q1 | 1,235 | 210 | 17.0% | 4.000 |
| days | Q2 | 1,171 | 85 | 7.3% | 12.000 |
| days | Q4 | 1,107 | 32 | 2.9% | 22.000 |
| days | Q3 | 1,160 | 52 | 4.5% | 19.000 |


Plot: `gap_sd_vs_days.png`.

## Extra 16 — inverse in-month days

gap_sd vs 1/days ρ=0.905 (TWIN). Y3 1/days 0.711; leftover after 1/days 0.585 (R²=0.393).

## Extra 17 — overlap-only (gap-defined Y3 rows)

On gap-defined Y3 rows only: gap 0.686 vs days 0.719 vs size 0.621. Still loses to days — the 4 null positives are not the gap.

| feature | n | n_pos | CV |
| --- | --- | --- | --- |
| c_gap_sd | 5,574 | 398 | 0.686 |
| c_n_days_with_tx | 5,574 | 398 | 0.719 |
| a_n_tx | 5,574 | 398 | 0.711 |
| log1p_a_in3 | 5,461 | 387 | 0.621 |


## Extra 18 — Q6 leftover after days_lag1

Y3 gap_sd_lag1 0.669 vs days_lag1 0.684; lag leftover after days_lag1 0.483 (R²=0.414). CLOSE as Q6 — the lag is the days twin lagged, not a new lead.

## Extra 19 — rank residual (not linear OLS)

Y3 rank-residual after days 0.523 (R²=0.815); after a_n_tx 0.542 (R²=0.747). Rank leftover also dies — not an OLS misspec.

## Extra 20 — n_tx leftover is days?

Y3 leftover-after-a_n_tx, then residualized on days: 0.446 (R²=0.280). The 0.659 n_tx leftover *is* days. Twin owner is days, not count.

## Extra 21 — Y3-labeled Spearman

Y3-labeled Spearman gap vs days -0.949, a_n_tx -0.884, size -0.468. Twin still holds on the stressed rows.

| pair | Spearman | flag |
| --- | --- | --- |
| c_n_days_with_tx | -0.949 | TWIN |
| a_n_tx | -0.884 | TWIN |
| c_n_tx | -0.884 | TWIN |
| log1p(a_in3) | -0.468 |  |
| c_recency_days | 0.456 |  |


## Extra 22 — days tercile × gap tercile

Y3 labeled 3×3: off-diagonal regular×quiet + irregular×busy = 20/5,574 (0.4%). A leftover regularity cell would be irregular-and-busy. If that cell is empty, gap_sd is inverse days.

| gap_t | days_t | n | n_pos | rate |
| --- | --- | --- | --- | --- |
| G1_reg | D1_quiet | 9 | 5 | 55.6% |
| G1_reg | D2 | 317 | 17 | 5.4% |
| G1_reg | D3_busy | 1,544 | 42 | 2.7% |
| G2 | D1_quiet | 238 | 34 | 14.3% |
| G2 | D2 | 1,443 | 65 | 4.5% |
| G2 | D3_busy | 165 | 2 | 1.2% |
| G3_irreg | D1_quiet | 1,727 | 228 | 13.2% |
| G3_irreg | D2 | 120 | 5 | 4.2% |
| G3_irreg | D3_busy | 11 | 0 | 0.0% |


## Extra 23 — days leftover after gap_sd

Y3 days leftover after gap_sd 0.658 (R²=0.414). Days still ranks after gap is removed — they are not interchangeable. The Y3 engine is right to keep days; the feature report picked the weaker twin as representative.

## What failed / next (held for wave note)

- leftover after days 0.535 dies; rank leftover 0.523; Q6 lag leftover 0.483; days leftover after gap 0.658 lives — Y3 engine keeps days. Feature report picked the weaker twin as representative.
- unique-day σ vs 1/c_n_tx still 0.867 — collapse cut raw 0.916 but did not escape the twin. Do not rewrite ops.py.

Elapsed 8s. Cuts: coverage, formula, Spearman, singles, residual, size terciles, ICC, nulls, dark 470/744, chronic-12, Q6, raw-vs-unique, so-far window, lifetime σ, quintiles, inverse-days, overlap, Q6-resid, rank-resid, n_tx-is-days, stressed-ρ, 3×3, days-after-gap.

Did **not**: rewrite ops.py, put `c_gap_sd` on the 15-col card, change the night Y3 quote or the days 0.711 bar, invent `y_gap_sd`, write a 0–100, touch `product/`, rewrite parquet/duckdb, run `build_targets`, write the parent journal, commit.
