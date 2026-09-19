# Q3 recency — going-quiet or extract hole?

Generated `2026-09-19T04:25:53+02:00` by agent `4545d7a6`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_recency` / `y_silent`. Do not revive `y6_silent_60` or `created_at`. Recency is **not** on the 15-col card. Night Y3 quote stays **0.762 / 0.752**. Days bar **0.711**.

`c_recency_days` = month_end.date − last booking date ≤ month_end. `c_last_tx_before_2026_06` = 1 if period ≥ 2026-06-01 and last booking as of month_end is < 2026-06-01. Javier extract-level **61**; as-of 2026-08-31 **62** (COMP_0981 silent 2025-04-07 → 2026-09-01).

## Headline

`c_recency_days` Y3 0.659 vs size 0.617 (Δ 0.042) vs days 0.711; leftover after days 0.607 (dies). Twin-days=False SIZE=False (ρ_size=-0.413). June flag extract 61 / as-of 62; COMP_0981 +1=True; extract-hole=True. On the 44: **DROP from the 44**. June flag **PARK**. PARK as health Y. Q6 **CLOSE**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Recency is a cash clock, not a health Y. PARK. ICC 0.926. |
| 2 | Who is improving? | Not this column. |
| 3 | Who is turning? | Going-quiet would be Q3. Leftover after days 0.607 — dies as a quiet twin of days. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | June flag is an extract cutoff, not a why. Recency vs days ρ=-0.615. |
| 6 | Months earlier? | Recency lag1 **CLOSE** (now 0.659 vs lag1 0.619; days lag1 replica 0.684 stays the night KEEP). |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `c_recency_days` on the 44 | **DROP from the 44** | Y3 recency 0.659 leftover-after-days 0.607 vs size 0.617 (Δ -0.010) vs days 0.711. CLOSE as quiet twin of days. Log leftover 0.673 beats size 0.617 but loses to days 0.711 (busy 0.680 vs days 0.711). Not on the 15-col card. |
| `c_recency_days` as a health Y | **PARK** | do not invent `y_recency` / `y_silent` |
| `c_recency_days` on the 15-col card | **no** | days 0.711 stays the engine; night Y3 0.762 / 0.752 untouched |
| `c_last_tx_before_2026_06` | **PARK** | Javier extract 61 vs as-of-Aug 62 (COMP_0981 is the +1). Flag only fires 2026-06..08 (203 CM); Y3 last labeled 2026-02-01 — never reaches the window. PARK as extract artifact; DROP from the 44. |
| Q6 lag1 `c_recency_days` | **CLOSE** | Y3 recency now 0.659 vs lag1 0.619 (Δ 0.040); short lag1 0.623. Days lag1 replica 0.684 (short 0.684). CLOSE as Q6 — lag does not hold. |
| `y6_silent_60` | **PARK** (do not revive) | future rejected Y; distinct window |
| `created_at` | **PARK** (do not revive) | connection clock, not cash clock |

## 1. Completeness + distributions

Train `c_recency_days` cov 100.0% p50=0.0 p90=11.0 (n_cm=21,157 / cos=1,214). `c_last_tx_before_2026_06` modal 0 share 99.0% (CONFIRM 99.0%); share=1 1.0%. Holdout recency cov 100.0% p50=0.0 — coverage only.

| split | col | n_cm | n_co | cov | p50 | p90 | modal% | share=1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | c_recency_days | 21,157 | 1,214 | 100.0% | 0.0 | 11.0 | 58.5% | — |
| train | c_last_tx_before_2026_06 | 21,157 | 1,214 | 100.0% | 0.0 | 0.0 | 99.0% | 1.0% |
| holdout | c_recency_days | 1,073 | 72 | 100.0% | 0.0 | 4.0 | 61.3% | — |
| holdout | c_last_tx_before_2026_06 | 1,073 | 72 | 100.0% | 0.0 | 0.0 | 99.3% | 0.7% |


Feature-report modal 99.0% on the June flag: **CONFIRM**.

## 2. Formula vs raw last booking — 61 vs 62 — COMP_0981

Store vs raw last-booking: recency agree 100.0% max|Δ|=0.00; June-flag agree 100.0%. Extract-level last tx < 2026-06-01: **61** (CONFIRM Javier 61). As-of 2026-08-31: **62** (CONFIRM 62). +1 names=['COMP_0981'] (COMP_0981 is the +1). COMP_0981 last-ever=2026-09-01 00:00:00 last-asof-Aug=2025-04-07 00:00:00 n_tx=161 next-after-2025-04-07=2026-09-01 00:00:00. Store Aug-2026 flag companies=62 (match as-of 62). Train as-of 61 / holdout 1.

| item | value |
| --- | ---: |
| store vs raw recency agree | 100.0% |
| store vs raw June-flag agree | 100.0% |
| max \|Δ recency\| | 0.00 |
| extract-level last tx < 2026-06-01 | 61 (Javier 61) |
| as-of 2026-08-31 | 62 (quoted 62) |
| +1 vs extract | COMP_0981 |
| COMP_0981 is the +1 | YES |
| COMP_0981 last-ever / last-asof-Aug | 2026-09-01 00:00:00 / 2025-04-07 00:00:00 |
| COMP_0981 next after 2025-04-07 | 2026-09-01 00:00:00 |
| store Aug-2026 flag companies | 62 |
| train / holdout as-of | 61 / 1 |

Roster (as-of 2026-08-31 last booking < 2026-06-01):

| company_id | split | last_aug | last_ever | extract_61 | plus1 |
| --- | --- | --- | --- | --- | --- |
| COMP_0002 | train | 2025-04-14 | 2025-04-14 | 1 | 0 |
| COMP_0032 | train | 2026-01-01 | 2026-01-01 | 1 | 0 |
| COMP_0055 | train | 2026-05-31 | 2026-05-31 | 1 | 0 |
| COMP_0060 | train | 2026-04-29 | 2026-04-29 | 1 | 0 |
| COMP_0062 | train | 2026-04-30 | 2026-04-30 | 1 | 0 |
| COMP_0080 | train | 2026-05-22 | 2026-05-22 | 1 | 0 |
| COMP_0147 | train | 2025-04-22 | 2025-04-22 | 1 | 0 |
| COMP_0148 | train | 2026-03-18 | 2026-03-18 | 1 | 0 |
| COMP_0170 | train | 2026-04-02 | 2026-04-02 | 1 | 0 |
| COMP_0188 | train | 2026-04-16 | 2026-04-16 | 1 | 0 |
| COMP_0223 | train | 2025-12-19 | 2025-12-19 | 1 | 0 |
| COMP_0259 | train | 2025-12-24 | 2025-12-24 | 1 | 0 |
| COMP_0261 | train | 2026-01-08 | 2026-01-08 | 1 | 0 |
| COMP_0269 | holdout | 2025-10-09 | 2025-10-09 | 1 | 0 |
| COMP_0288 | train | 2026-02-01 | 2026-02-01 | 1 | 0 |
| COMP_0307 | train | 2025-09-11 | 2025-09-11 | 1 | 0 |
| COMP_0325 | train | 2026-04-15 | 2026-04-15 | 1 | 0 |
| COMP_0337 | train | 2026-04-16 | 2026-04-16 | 1 | 0 |
| COMP_0342 | train | 2025-12-11 | 2025-12-11 | 1 | 0 |
| COMP_0343 | train | 2025-10-17 | 2025-10-17 | 1 | 0 |
| COMP_0379 | train | 2026-05-14 | 2026-05-14 | 1 | 0 |
| COMP_0398 | train | 2026-05-02 | 2026-05-02 | 1 | 0 |
| COMP_0434 | train | 2025-10-30 | 2025-10-30 | 1 | 0 |
| COMP_0440 | train | 2026-05-31 | 2026-05-31 | 1 | 0 |
| COMP_0475 | train | 2025-06-13 | 2025-06-13 | 1 | 0 |
| COMP_0492 | train | 2025-10-13 | 2025-10-13 | 1 | 0 |
| COMP_0503 | train | 2026-02-04 | 2026-02-04 | 1 | 0 |
| COMP_0606 | train | 2026-02-27 | 2026-02-27 | 1 | 0 |
| COMP_0624 | train | 2025-11-17 | 2025-11-17 | 1 | 0 |
| COMP_0625 | train | 2026-05-05 | 2026-05-05 | 1 | 0 |
| COMP_0681 | train | 2025-08-06 | 2025-08-06 | 1 | 0 |
| COMP_0692 | train | 2026-05-28 | 2026-05-28 | 1 | 0 |
| COMP_0712 | train | 2025-05-29 | 2025-05-29 | 1 | 0 |
| COMP_0715 | train | 2026-01-28 | 2026-01-28 | 1 | 0 |
| COMP_0720 | train | 2026-05-25 | 2026-05-25 | 1 | 0 |
| COMP_0729 | train | 2025-12-11 | 2025-12-11 | 1 | 0 |
| COMP_0806 | train | 2026-03-31 | 2026-03-31 | 1 | 0 |
| COMP_0840 | train | 2025-07-22 | 2025-07-22 | 1 | 0 |
| COMP_0857 | train | 2026-05-19 | 2026-05-19 | 1 | 0 |
| COMP_0869 | train | 2026-02-07 | 2026-02-07 | 1 | 0 |
| COMP_0870 | train | 2026-04-29 | 2026-04-29 | 1 | 0 |
| COMP_0889 | train | 2026-05-15 | 2026-05-15 | 1 | 0 |
| COMP_0916 | train | 2025-10-26 | 2025-10-26 | 1 | 0 |
| COMP_0938 | train | 2025-09-02 | 2025-09-02 | 1 | 0 |
| COMP_0962 | train | 2026-04-20 | 2026-04-20 | 1 | 0 |
| COMP_0980 | train | 2026-04-30 | 2026-04-30 | 1 | 0 |
| COMP_0981 | train | 2025-04-07 | 2026-09-01 | 0 | 1 |
| COMP_1000 | train | 2026-03-31 | 2026-03-31 | 1 | 0 |
| COMP_1004 | train | 2026-01-09 | 2026-01-09 | 1 | 0 |
| COMP_1046 | train | 2025-12-11 | 2025-12-11 | 1 | 0 |
| COMP_1065 | train | 2026-05-22 | 2026-05-22 | 1 | 0 |
| COMP_1068 | train | 2026-02-19 | 2026-02-19 | 1 | 0 |
| COMP_1099 | train | 2025-04-14 | 2025-04-14 | 1 | 0 |
| COMP_1121 | train | 2026-03-17 | 2026-03-17 | 1 | 0 |
| COMP_1143 | train | 2025-12-12 | 2025-12-12 | 1 | 0 |
| COMP_1148 | train | 2026-04-20 | 2026-04-20 | 1 | 0 |
| COMP_1152 | train | 2026-03-02 | 2026-03-02 | 1 | 0 |
| COMP_1159 | train | 2026-02-06 | 2026-02-06 | 1 | 0 |
| COMP_1237 | train | 2026-05-04 | 2026-05-04 | 1 | 0 |
| COMP_1245 | train | 2026-04-30 | 2026-04-30 | 1 | 0 |
| COMP_1255 | train | 2025-07-21 | 2025-07-21 | 1 | 0 |
| COMP_1265 | train | 2025-12-23 | 2025-12-23 | 1 | 0 |


## 3. Spearman vs days / n_tx / gap_sd / size / zero-in

Recency Spearman: days -0.615 (not a days twin); a_n_tx -0.579 (not n_tx twin); gap_sd 0.462; zero_in 0.443 (not zero-in twin); log1p(a_in3) -0.413 (not SIZE ).

| pair | ρ | tag |
| --- | --- | --- |
| c_recency_days vs c_n_days_with_tx | -0.615 |  |
| c_recency_days vs a_n_tx | -0.579 |  |
| c_recency_days vs c_n_tx | -0.579 |  |
| c_recency_days vs c_gap_sd | 0.462 |  |
| c_recency_days vs log1p(a_in3) | -0.413 |  |
| c_recency_days vs c_zero_in_month | 0.443 |  |
| june_flag vs c_recency_days | 0.186 |  |
| june_flag vs c_n_days_with_tx | -0.163 |  |
| june_flag vs log1p(a_in3) | -0.149 |  |
| c_n_days_with_tx vs a_n_tx | 0.938 | TWIN |
| c_n_days_with_tx vs log1p(a_in3) | 0.595 | SIZE |


Twin if \|ρ\|≥0.80. SIZE if \|ρ\| vs log1p(a_in3) ≥0.50. Feature-report size ρ −0.447.

## 4. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711); size 0.617 (replica 0.617).

Y3 recency 0.659 vs size 0.617 (Δ 0.042) vs days 0.711 (night 0.711, replica OK). June-flag Y3 0.500 (mean on labeled Y3 rows 0.0%). Y2 recency 0.546 vs size 0.552; June-flag 0.500 (mean on labeled Y2 0.0%).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_recency_days | 17,356 | 1,271 | 0.546 | 0.039 | -1 | 0.541 | 0.557 0.572 0.580 0.540 0.483 |
| y2_neg_2of3 | c_last_tx_before_2026_06 | 17,356 | 1,271 | 0.500 | 0.000 | 1 | 0.500 | 0.500 0.500 0.500 0.500 0.500 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 | 0.641 0.569 0.648 0.584 0.549 |
| y2_neg_2of3 | c_gap_sd | 16,559 | 1,207 | 0.577 | 0.040 | -1 | 0.585 | 0.620 0.519 0.609 0.569 0.568 |
| y2_neg_2of3 | c_zero_in_month | 17,356 | 1,271 | 0.522 | 0.027 | -1 | 0.521 | 0.526 0.539 0.555 0.504 0.486 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.523 0.609 0.595 0.520 0.513 |
| y3_recover_cash_6m | c_recency_days | 5,648 | 402 | 0.659 | 0.051 | 1 | 0.665 | 0.590 0.666 0.666 0.730 0.642 |
| y3_recover_cash_6m | c_last_tx_before_2026_06 | 5,648 | 402 | 0.500 | 0.000 | 1 | 0.500 | 0.500 0.500 0.500 0.500 0.500 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | c_gap_sd | 5,574 | 398 | 0.686 | 0.038 | 1 | 0.693 | 0.630 0.721 0.690 0.672 0.720 |
| y3_recover_cash_6m | c_zero_in_month | 5,648 | 402 | 0.580 | 0.015 | 1 | 0.584 | 0.571 0.582 0.593 0.559 0.595 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |


June-flag mean on labeled Y3 rows 0.0%; on labeled Y2 0.0%.

## 5. Residual after days

Y3 leftover after days 0.607; after days+n_tx 0.613 vs size 0.617 (Δ -0.010). Y2 leftover-after-days 0.553. Slope recency~days -1.085. Leftover does not clear size+0.02 after days.

| y | feature | n | n_pos | CV | sign | Δsize |
| --- | --- | --- | --- | --- | --- | --- |
| Y3 | resid_days | 5,648 | 402 | 0.607 | -1 | -0.010 |
| Y3 | resid_n_tx | 5,648 | 402 | 0.557 | 1 | -0.059 |
| Y3 | resid_days+n_tx | 5,648 | 402 | 0.613 | -1 | -0.003 |
| Y3 | resid_days+n_tx+zero | 5,648 | 402 | 0.622 | -1 | 0.006 |
| Y3 | c_recency_days | 5,648 | 402 | 0.659 | 1 | 0.042 |
| Y2 | resid_days | 17,356 | 1,271 | 0.553 | 1 | 0.001 |
| Y2 | resid_n_tx | 17,356 | 1,271 | 0.474 | 1 | -0.078 |
| Y2 | resid_days+n_tx | 17,356 | 1,271 | 0.550 | 1 | -0.002 |
| Y2 | resid_days+n_tx+zero | 17,356 | 1,271 | 0.540 | 1 | -0.012 |
| Y2 | c_recency_days | 17,356 | 1,271 | 0.546 | -1 | -0.006 |


KEEP-as-X: leftover after days beats size ≥0.02 **and** is not the June extract dummy. If leftover dies, CLOSE as quiet twin.

## 6. June-cutoff — only 2026-06..08?

June-flag =1 on 203 train CM; 203 in 2026-06..08; 0 outside. Months with any flag=1: 3. ONLY defined/informative on 2026-06..08 (3 months). Last labeled Y3 period 2026-02-01; Y2 2026-05-01. Y3/Y2 labels never reach the June window — extract hole.

| period | n_cm | june n | june % | rec p50 | rec p90 | ≥30d | ≥60d |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2024-09 | 435 | 0 | 0.0% | 0.0 | 4.0 | 0.0% | 0.0% |
| 2024-10 | 473 | 0 | 0.0% | 0.0 | 9.0 | 1.7% | 0.0% |
| 2024-11 | 483 | 0 | 0.0% | 1.0 | 9.0 | 2.1% | 0.4% |
| 2024-12 | 525 | 0 | 0.0% | 0.0 | 11.0 | 1.9% | 0.6% |
| 2025-01 | 636 | 0 | 0.0% | 0.0 | 7.0 | 1.6% | 0.5% |
| 2025-02 | 677 | 0 | 0.0% | 0.0 | 7.0 | 1.8% | 0.3% |
| 2025-03 | 714 | 0 | 0.0% | 0.0 | 11.0 | 3.2% | 1.3% |
| 2025-04 | 733 | 0 | 0.0% | 0.0 | 8.0 | 2.5% | 1.2% |
| 2025-05 | 752 | 0 | 0.0% | 1.0 | 9.9 | 3.5% | 1.2% |
| 2025-06 | 762 | 0 | 0.0% | 0.0 | 10.0 | 3.7% | 1.6% |
| 2025-07 | 796 | 0 | 0.0% | 0.0 | 9.5 | 3.0% | 1.6% |
| 2025-08 | 833 | 0 | 0.0% | 2.0 | 16.0 | 5.2% | 1.8% |
| 2025-09 | 864 | 0 | 0.0% | 0.0 | 8.0 | 3.0% | 2.3% |
| 2025-10 | 908 | 0 | 0.0% | 0.0 | 11.0 | 4.1% | 1.7% |
| 2025-11 | 945 | 0 | 0.0% | 2.0 | 12.0 | 4.0% | 2.4% |
| 2025-12 | 1,006 | 0 | 0.0% | 0.0 | 12.0 | 5.0% | 2.6% |
| 2026-01 | 1,132 | 0 | 0.0% | 1.0 | 10.0 | 4.2% | 2.5% |
| 2026-02 | 1,204 | 0 | 0.0% | 1.0 | 11.0 | 4.6% | 2.8% |
| 2026-03 | 1,211 | 0 | 0.0% | 0.0 | 8.0 | 4.9% | 3.6% |
| 2026-04 | 1,212 | 0 | 0.0% | 0.0 | 10.0 | 4.4% | 3.5% |
| 2026-05 | 1,214 | 0 | 0.0% | 2.0 | 13.0 | 5.8% | 3.5% |
| 2026-06 | 1,214 | 78 | 6.4% | 0.0 | 15.0 | 6.4% | 4.5% |
| 2026-07 | 1,214 | 64 | 5.3% | 0.0 | 15.0 | 7.6% | 5.3% |
| 2026-08 | 1,214 | 61 | 5.0% | 0.0 | 29.4 | 10.0% | 6.6% |


| slice | feature | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| all | c_last_tx_before_2026_06 | 5,648 | 402 | 0.500 |
| all | c_recency_days | 5,648 | 402 | 0.659 |
| all | log1p_a_in3 | 5,528 | 391 | 0.617 |
| all | c_n_days_with_tx | 5,648 | 402 | 0.711 |
| pre_2026_06 | c_last_tx_before_2026_06 | 5,648 | 402 | 0.500 |
| pre_2026_06 | c_recency_days | 5,648 | 402 | 0.659 |
| pre_2026_06 | log1p_a_in3 | 5,528 | 391 | 0.617 |
| pre_2026_06 | c_n_days_with_tx | 5,648 | 402 | 0.711 |
| 2026_06_08 | c_last_tx_before_2026_06 | 0 | 0 | LOW_POWER |
| 2026_06_08 | c_recency_days | 0 | 0 | LOW_POWER |
| 2026_06_08 | log1p_a_in3 | 0 | 0 | LOW_POWER |
| 2026_06_08 | c_n_days_with_tx | 0 | 0 | LOW_POWER |
| Y2_all | c_last_tx_before_2026_06 | 17,356 | 1,271 | 0.500 |
| Y2_all | c_recency_days | 17,356 | 1,271 | 0.546 |
| Y2_all | log1p_a_in3 | 14,968 | 1,044 | 0.552 |
| Y2_all | c_n_days_with_tx | 17,356 | 1,271 | 0.571 |
| Y2_pre_2026_06 | c_last_tx_before_2026_06 | 17,356 | 1,271 | 0.500 |
| Y2_pre_2026_06 | c_recency_days | 17,356 | 1,271 | 0.546 |
| Y2_pre_2026_06 | log1p_a_in3 | 14,968 | 1,044 | 0.552 |
| Y2_pre_2026_06 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 |
| Y2_2026_06_08 | c_last_tx_before_2026_06 | 0 | 0 | LOW_POWER |
| Y2_2026_06_08 | c_recency_days | 0 | 0 | LOW_POWER |
| Y2_2026_06_08 | log1p_a_in3 | 0 | 0 | LOW_POWER |
| Y2_2026_06_08 | c_n_days_with_tx | 0 | 0 | LOW_POWER |


Extract hole: **YES — DROP from the 44**.

## 7. vs rejected `y6_silent_60`

Train labeled y6_silent_60: n=17,515 pos=525 rate 3.0%. Jaccard(recency>60, Y6)=0.378 ρ=0.550; P(Y6|rec>60)=69.1% P(rec>60|Y6)=45.5%. Distinct window (now vs t+3) — do not merge / do not revive Y6. Y6~recency CV 0.904 vs days 0.895 vs size 0.783.

| pair | n_lab | Jaccard | ρ |
| --- | --- | --- | --- |
| recency>60 vs y6_silent_60 | 17,515 | 0.378 | 0.550 |
| c_recency_days vs y6_silent_60 | 17,515 | — | 0.265 |
| june_flag vs y6_silent_60 | 17,515 | 0.000 | — |


Y6 is last-tx as of **t+3** more than 60 days before that end. Recency is as-of **t**. Do not revive Y6. Do not merge.

## 8. Dark 470 vs invoiced 744

Train last-month companies: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Recency p50 (company-median of medians): invoiced 0.0 vs dark 0.0. June-flag companies: invoiced 43 vs dark 35. Y3 recency _erp 0.683 dark 0.632. Holdout ever-ERP coverage 40/72.

| group | n_co | rec_p50 | rec_mean | share_ge60 | n_june_co |
| --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 0.0 | 5.2 | 1.9% | 43 |
| never_erp_470 | 470 | 0.0 | 7.2 | 2.8% | 35 |


| group | n_cm | n_co | rec p50 | june |
| --- | --- | --- | --- | --- |
| ever_erp | 13,554 | 744 | 0.0 | 0.8% |
| never_erp | 7,603 | 470 | 0.0 | 1.3% |


| slice | n | n_pos | CV |
| --- | --- | --- | --- |
| ever_erp | 3,618 | 264 | 0.683 |
| never_erp | 2,030 | 138 | 0.632 |


## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)

Chronic 12 names (0158/0172, ≥50% labeled months below 0): 12. Y2 recency CV full 0.546 → drop-12 0.537. Y3 0.659 → 0.657. Recency p50 on 12 0.0 vs rest 0.0. Drop does not flip Y2 (≥0.03).

IDs (12): COMP_0188, COMP_0254, COMP_0356, COMP_0577, COMP_0679, COMP_0750, COMP_0885, COMP_0910, COMP_0911, COMP_0947, COMP_0969, COMP_1213.

## 10. ICC / company-demean

c_recency_days acf1=-0.192 acf3=0.250 acf6=0.316; ICC=0.926 (sticky company trait). Y3 company-mean 0.754 vs demean 0.602; Y2 mean 0.576 vs shock 0.526. Trait, not a month shock.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| Y3 | co_mean | 5,648 | 402 | 0.754 | 1 |
| Y3 | demean | 5,648 | 402 | 0.602 | -1 |
| Y3 | now | 5,648 | 402 | 0.659 | 1 |
| Y2 | co_mean | 17,356 | 1,271 | 0.576 | -1 |
| Y2 | demean | 17,356 | 1,271 | 0.526 | 1 |
| Y2 | now | 17,356 | 1,271 | 0.546 | -1 |


## 11. Q6 — lag1 / lag3 on short vs long

Y3 recency now 0.659 vs lag1 0.619 (Δ 0.040); short lag1 0.623. Days lag1 replica 0.684 (short 0.684). CLOSE as Q6 — lag does not hold.

| y | col | slice | n | n_pos | present | CV | sign |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | c_recency_days | all | 5,648 | 402 | 100.0% | 0.659 | 1 |
| y3_recover_cash_6m | c_recency_days_lag1 | all | 5,648 | 402 | 100.0% | 0.619 | 1 |
| y3_recover_cash_6m | c_recency_days_lag3 | all | 5,078 | 355 | 89.9% | 0.616 | 1 |
| y3_recover_cash_6m | c_n_days_with_tx | all | 5,648 | 402 | 100.0% | 0.711 | -1 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | all | 5,648 | 402 | 100.0% | 0.684 | -1 |
| y3_recover_cash_6m | c_recency_days | short_<12 | 3,723 | 252 | 100.0% | 0.646 | 1 |
| y3_recover_cash_6m | c_recency_days_lag1 | short_<12 | 3,723 | 252 | 100.0% | 0.623 | 1 |
| y3_recover_cash_6m | c_recency_days_lag3 | short_<12 | 3,153 | 205 | 100.0% | 0.631 | 1 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | short_<12 | 3,723 | 252 | 100.0% | 0.684 | -1 |
| y3_recover_cash_6m | c_recency_days | long_ge18 | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover_cash_6m | c_recency_days_lag1 | long_ge18 | 213 | 16 | 100.0% | LOW_POWER | — |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | long_ge18 | 213 | 16 | 100.0% | LOW_POWER | — |
| y2_neg_2of3 | c_recency_days_lag1 | Y2_all | 16,161 | 1,151 | 100.0% | 0.546 | -1 |


Days lag1 is a night KEEP. Recency lag is scored the same way (present on short + AUROC holds).

## 12. Calendar — extract-end pile?

Recency p50 stays 0 (early 0.0 / Aug 0.0). The *tail* piles: p90 early 10.0 vs Aug 29.4; share≥30d early 3.4% vs Aug 10.0%. YES — the recency tail piles at extract end the way a still does. Y3 recency on pre-June labeled 0.659 vs size 0.617 vs days 0.711.

| period | rec p50 | rec p90 | ≥30d | ≥60d | days=0 | june % |
| --- | --- | --- | --- | --- | --- | --- |
| 2024-09 | 0.0 | 4.0 | 0.0% | 0.0% | 0.0% | 0.0% |
| 2024-10 | 0.0 | 9.0 | 1.7% | 0.0% | 1.3% | 0.0% |
| 2024-11 | 1.0 | 9.0 | 2.1% | 0.4% | 2.1% | 0.0% |
| 2024-12 | 0.0 | 11.0 | 1.9% | 0.6% | 1.7% | 0.0% |
| 2025-01 | 0.0 | 7.0 | 1.6% | 0.5% | 1.3% | 0.0% |
| 2025-02 | 0.0 | 7.0 | 1.8% | 0.3% | 2.1% | 0.0% |
| 2025-03 | 0.0 | 11.0 | 3.2% | 1.3% | 2.9% | 0.0% |
| 2025-04 | 0.0 | 8.0 | 2.5% | 1.2% | 2.5% | 0.0% |
| 2025-05 | 1.0 | 9.9 | 3.5% | 1.2% | 3.1% | 0.0% |
| 2025-06 | 0.0 | 10.0 | 3.7% | 1.6% | 3.7% | 0.0% |
| 2025-07 | 0.0 | 9.5 | 3.0% | 1.6% | 2.1% | 0.0% |
| 2025-08 | 2.0 | 16.0 | 5.2% | 1.8% | 4.6% | 0.0% |
| 2025-09 | 0.0 | 8.0 | 3.0% | 2.3% | 3.0% | 0.0% |
| 2025-10 | 0.0 | 11.0 | 4.1% | 1.7% | 3.3% | 0.0% |
| 2025-11 | 2.0 | 12.0 | 4.0% | 2.4% | 4.0% | 0.0% |
| 2025-12 | 0.0 | 12.0 | 5.0% | 2.6% | 4.4% | 0.0% |
| 2026-01 | 1.0 | 10.0 | 4.2% | 2.5% | 3.7% | 0.0% |
| 2026-02 | 1.0 | 11.0 | 4.6% | 2.8% | 4.7% | 0.0% |
| 2026-03 | 0.0 | 8.0 | 4.9% | 3.6% | 4.8% | 0.0% |
| 2026-04 | 0.0 | 10.0 | 4.4% | 3.5% | 4.4% | 0.0% |
| 2026-05 | 2.0 | 13.0 | 5.8% | 3.5% | 5.5% | 0.0% |
| 2026-06 | 0.0 | 15.0 | 6.4% | 4.5% | 6.4% | 6.4% |
| 2026-07 | 0.0 | 15.0 | 7.6% | 5.3% | 6.8% | 5.3% |
| 2026-08 | 0.0 | 29.4 | 10.0% | 6.6% | 10.0% | 5.0% |


Plot: `recency_calendar.png`.

## Extra 13 — recency among busy months

Y3 recency on days>0 0.643; on days=0 LOW_POWER. If skill lives only on days=0, recency is the quiet-month twin.

| y | slice | n | n_pos | rec_p50 | CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | days_gt0 | 16,764 | 1,236 | 0.0 | 0.545 |
| y3_recover_cash_6m | days_gt0 | 5,536 | 372 | 0.0 | 0.643 |
| y2_neg_2of3 | days_eq0 | 592 | 35 | 77.0 | LOW_POWER |
| y3_recover_cash_6m | days_eq0 | 112 | 30 | 77.0 | LOW_POWER |
| y2_neg_2of3 | days_ge5 | 13,927 | 1,077 | 0.0 | 0.526 |
| y3_recover_cash_6m | days_ge5 | 4,978 | 263 | 0.0 | 0.593 |
| y2_neg_2of3 | zero_in | 1,958 | 95 | 15.0 | 0.416 |
| y3_recover_cash_6m | zero_in | 313 | 85 | 15.0 | 0.399 |


## Extra 14 — Y3 rate by recency quintile

Y3 rate by recency quintile: 4.4% / 8.9% / 22.1% / 33.9% / 17.8%; high-recency tail recovers more.

| q | n | n_pos | Y3 rate | rec p50 | days p50 |
| --- | --- | --- | --- | --- | --- |
| 0 | 3,508 | 153 | 4.4% | 0.0 | 21.0 |
| 1-7 | 1,744 | 156 | 8.9% | 1.0 | 13.0 |
| 8-30 | 289 | 64 | 22.1% | 13.0 | 2.0 |
| 31-60 | 62 | 21 | 33.9% | 39.0 | 0.0 |
| ≥60 | 45 | 8 | 17.8% | 91.0 | 0.0 |


## Extra 15 — onset recency≥30 (not a Y)

Y3 onset recency≥30 0.524 vs state≥30 0.530. Onset would be the cleanest Q3 going-quiet; do not invent a Y from it.

| y | feature | n | n_pos | prev | CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | onset_ge30 | 17,356 | 1,271 | 1.6% | 0.494 |
| y2_neg_2of3 | state_ge30 | 17,356 | 1,271 | 3.6% | 0.505 |
| y2_neg_2of3 | recency | 17,356 | 1,271 | — | 0.546 |
| y3_recover_cash_6m | onset_ge30 | 5,648 | 402 | 1.2% | 0.524 |
| y3_recover_cash_6m | state_ge30 | 5,648 | 402 | 2.0% | 0.530 |
| y3_recover_cash_6m | recency | 5,648 | 402 | — | 0.659 |


## Extra 16 — holdout coverage only

Holdout 72 coverage only: 1,073 CM. Recency p50 0.0. June-flag mean 0.007. As-of-62 in holdout: 1 (COMP_0269). Extract-61 in holdout: 1. No AUROC claim.

| col | n_cm | n_co | cov | mean | p50 |
| --- | --- | --- | --- | --- | --- |
| c_recency_days | 1073 | 72 | 100.0% | 4.145 | 0.0 |
| c_last_tx_before_2026_06 | 1073 | 72 | 100.0% | 0.007 | 0.0 |
| c_n_days_with_tx | 1073 | 72 | 100.0% | 14.726 | 15.0 |


## Extra 17 — leftover after days on days>0

Busy-month leftover after days: Y3 0.649 vs size 0.612. If this dies, recency adds nothing inside months that already book.

| y | slice | n | n_pos | resid CV | size CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | days>0 leftover-after-days | 16,764 | 1,236 | 0.560 | 0.550 |
| y3_recover_cash_6m | days>0 leftover-after-days | 5,536 | 372 | 0.649 | 0.612 |


## Extra 18 — how long have the 62 been silent?

Train as-of silent 61: last-tx age p50=205d p90=406d. Buckets last<2025=0 2025H1=6 2025H2=17 2026-01..05=38. Their labeled Y3 rate 23.2% (n_lab=263); Y2 3.8%.

| bucket | n |
| --- | --- |
| last < 2025-01 | 0 |
| 2025 H1 | 6 |
| 2025 H2 | 17 |
| 2026-01..05 | 38 |
| train as-of 62 | 61 |


## Extra 19 — company-mean leftover after company-mean days

Company-mean recency↔days ρ=-0.792 (not a company-level twin). Y3 rec_co_mean 0.754 vs days_co_mean 0.730; leftover 0.462 vs size 0.617 (Δ -0.155). Company-mean leftover dies — recency is a quiet-company twin of days.

| y | feature | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| y2_neg_2of3 | rec_co_mean | 17,356 | 1,271 | 0.576 |
| y2_neg_2of3 | days_co_mean | 17,356 | 1,271 | 0.599 |
| y2_neg_2of3 | resid_comean_days | 17,356 | 1,271 | 0.575 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 |
| y3_recover_cash_6m | rec_co_mean | 5,648 | 402 | 0.754 |
| y3_recover_cash_6m | days_co_mean | 5,648 | 402 | 0.730 |
| y3_recover_cash_6m | resid_comean_days | 5,648 | 402 | 0.462 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 |


## Extra 20 — feature-report size ρ (company-median)

size ρ recency vs log1p(a_in3): company-month -0.413 / company-median -0.495 (feature-report −0.447 off). Not SIZE at 0.50.

## Extra 21 — drop the train-61 silent names

Drop train-61 silent names: Y3 recency 0.659 → 0.645 vs size 0.637 vs days 0.716. On the 61: Y3 rate 23.2% n_lab=263 n_pos=61; log1p(a_in3) p50 7.89 vs rest 12.59; days p50 2.0 rec p50 9.0. The 61 do not carry the Y3 recency skill.

## Extra 22 — busy-month leftover vs days

Busy months (days>0) Y3: recency 0.643 leftover 0.649 vs days 0.699 vs size 0.612. If leftover loses to days, recency is still the quiet twin inside booked months.

| y | n | n_pos | recency | resid | days | size |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | 16,764 | 1,236 | 0.545 | 0.560 | 0.571 | 0.550 |
| y3_recover_cash_6m | 5,536 | 372 | 0.643 | 0.649 | 0.699 | 0.612 |


## Extra 23 — log / rank residual after days

Alt leftover Y3: log1p(days) 0.673; rank 0.567 vs size 0.617. An alt residual beats size — revisit KEEP.

| feature | n | n_pos | CV |
| --- | --- | --- | --- |
| resid_log1p_days | 5,648 | 402 | 0.673 |
| resid_ranks | 5,648 | 402 | 0.567 |
| c_recency_days | 5,648 | 402 | 0.659 |
| c_n_days_with_tx | 5,648 | 402 | 0.711 |
| log1p_a_in3 | 5,528 | 391 | 0.617 |


## Extra 24 — holdout COMP_0269 + recency==0

Holdout silent name COMP_0269 last_aug=2025-10-09 last_ever=2025-10-09. Train recency==0 (booked on month-end) share 58.5%; Y3 0.627 vs days 0.711 — still loses to the engine.

| y | feature | n | n_pos | CV | days |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | recency_eq_0 | 17,356 | 1,271 | 0.537 | 0.571 |
| y3_recover_cash_6m | recency_eq_0 | 5,648 | 402 | 0.627 | 0.711 |


## Extra 25 — leftover fold bits

Y3 folds recency [0.590 0.666 0.666 0.730 0.642] leftover [0.577 0.676 0.551 0.550 0.681] days [0.665 0.738 0.700 0.715 0.740] size [0.565 0.632 0.683 0.543 0.661].

| feature | CV | folds |
| --- | --- | --- |
| c_recency_days | 0.659 | 0.590 0.666 0.666 0.730 0.642 |
| resid_days | 0.607 | 0.577 0.676 0.551 0.550 0.681 |
| c_n_days_with_tx | 0.711 | 0.665 0.738 0.700 0.715 0.740 |
| log1p_a_in3 | 0.617 | 0.565 0.632 0.683 0.543 0.661 |


## Extra 26 — first-month recency (onboarding)

First-month recency p50 0.0 vs so-far≥6 0.0. Onboarding recency is not a health Y (same reason created_at is PARK).

| slice | n | n_pos | rec_p50 | CV |
| --- | --- | --- | --- | --- |
| so_far=1 | 0 | 0 | 0.0 | LOW_POWER |
| so_far>=6 | 4,212 | 313 | 0.0 | 0.656 |


## Extra 27 — log leftover is the days=0 pile?

log1p(days) leftover Y3 all 0.673 / days>0 0.680; days+zero dummy leftover 0.553. Days engine 0.711 / busy 0.699. Log leftover is not only the days=0 pile. Still loses to days — CLOSE as quiet twin.

| feature | n | n_pos | CV | folds |
| --- | --- | --- | --- | --- |
| log_resid_all | 5,648 | 402 | 0.673 | 0.620 0.724 0.668 0.633 0.721 |
| log_resid_days>0 | 5,536 | 372 | 0.680 | 0.638 0.723 0.655 0.652 0.731 |
| resid_days+zero | 5,648 | 402 | 0.553 | 0.546 0.579 0.565 0.510 0.567 |
| log_resid+zero | 5,648 | 402 | 0.569 | 0.551 0.596 0.590 0.505 0.603 |
| days_all | 5,648 | 402 | 0.711 | 0.665 0.738 0.700 0.715 0.740 |
| days_days>0 | 5,536 | 372 | 0.699 | 0.653 0.740 0.665 0.694 0.744 |
| size_all | 5,528 | 391 | 0.617 | 0.565 0.632 0.683 0.543 0.661 |
| size_days>0 | 5,422 | 361 | 0.612 | 0.544 0.643 0.657 0.555 0.661 |


## Extra 28 — recency bins among days>0

Y3 rate by recency bin among days>0: 4.4% / 8.9% / 22.2%. Within-month last-booking timing, not a multi-month going-quiet.

| q | n | n_pos | Y3 rate | days p50 |
| --- | --- | --- | --- | --- |
| 0 | 3,508 | 153 | 4.4% | 21.0 |
| 1-7 | 1,744 | 156 | 8.9% | 13.0 |
| 8-30 | 284 | 63 | 22.2% | 2.0 |


## Extra 29 — leftover after days+size

Y3 leftover after days+size 0.592; after log1p(days)+size 0.678. If this is chance, recency is days+size, not a new Q3.

| feature | n | n_pos | CV | folds |
| --- | --- | --- | --- | --- |
| resid_days+size | 5,528 | 391 | 0.592 | 0.543 0.647 0.627 0.485 0.658 |
| resid_logdays+size | 5,528 | 391 | 0.678 | 0.626 0.727 0.699 0.616 0.722 |


## What failed / next (held for wave note)

- leftover after days 0.607 vs size 0.617 — CLOSE as quiet twin
- June flag is a 2026-06 extract hole — DROP from the 44 / PARK as artifact
- recency tail piles at Aug (p90 29.4 vs early 10.0; ≥30d 10.0% vs 3.4%) — extract-end still
- company-mean leftover 0.462 vs size — quiet-company twin of days
- log leftover 0.673 beats size but loses to days 0.711; busy log leftover 0.680 — CLOSE as quiet twin

Elapsed 7s. Cuts: dist, formula 61/62, Spearman, singles, leftover-after-days, June extract hole, vs Y6, dark 470/744, 12 names, ICC/demean, Q6 short/long, calendar tail pile, busy months, bins, onset, holdout, busy residual, silent-age, company-mean leftover, size-ρ quote, drop-61, busy-vs-days, alt residual, COMP_0269, fold bits, first-month.
