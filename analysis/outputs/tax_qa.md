# Q5 tax calendar vs missed-tax

Generated `2026-09-19T03:02:50+02:00` by agent `1094dc70`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent a missed-tax Y.

`c_tax_month` = any category `tax` (not `tax_refund`). `c_missed_tax` = usual tax in last ≤6 months (≥3) AND no tax this month. Feature report: rare-event flag, modal 91.7%.

## Headline

Calendar **mixed_q_peaked** (Q-months tax 69.2% vs other 43.4%). `c_missed_tax` modal 91.7% (CONFIRM 91.7%); missed is NOT just not-a-tax-month (P(missed|not tax)=17.3%). Y3 missed 0.511 vs size 0.617 (Δ -0.105) vs days 0.711. Dark vs 744 tax-CM 48.7% vs 51.8%. Q5 **CLOSE**. PARK as health Y. X **PARK**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not this flag. PARK as a health Y. Cadence: 116 never / 406 monthly / 183 quarterly. |
| 2 | Who is improving? | tax_refund months are not a recovery Y. |
| 3 | Who is turning? | Missed-tax is a skip of a usual filing, or the complementary calendar hole. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | **CLOSE** — tax calendar is mixed_q_peaked (Q 69.2% vs other 43.4%); missed-tax is the complementary hole (Q-miss 2.2% vs non-Q 10.9%). Y3 0.511 loses to size 0.617. Quarterly cadence cannot fire the flag (tax6<3). |
| 6 | Months earlier? | lag1 of missed-tax CLOSE (contemporaneous 0.511 vs lag1 0.504). Only 1-month leads are honest. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `c_missed_tax` as Q5 why | **CLOSE** | tax calendar is mixed_q_peaked (Q 69.2% vs other 43.4%); missed-tax is the complementary hole (Q-miss 2.2% vs non-Q 10.9%). Y3 0.511 loses to size 0.617. Quarterly cadence cannot fire the flag (tax6<3). |
| `c_missed_tax` as a health Y | **PARK** | do not invent a merged Y from a calendar / skip flag |
| `c_missed_tax` / `c_tax_month` as Y3 X | **PARK** | size dummy 0.617 ≥ 0.60 |
| tax calendar as a dummy | **CLOSE (it is the dummy)** | shape=mixed_q_peaked; Q 69.2% vs other 43.4% |
| Q6 lag1 `c_missed_tax` | **CLOSE** | Y3 contemporaneous missed 0.511 is chance; lag1 0.504. CLOSE as Q6 — there is no contemporaneous skill to lead. |
| `tax_refund` as recovery Y | **PARK** | measured; do not build a Y |

## 1. Calendar — Jan–Dec stacked 2024-09..2026-08

Stacked Jan–Dec train company-months: Q-months (Jan/Apr/Jul/Oct) tax-CM share **69.2%**, other months **43.4%** (ratio 1.59). Peak Oct 72.4%, trough Aug 38.6%. Shape call: **mixed_q_peaked**. Store `c_tax_month` vs raw category=tax agreement 100.0%.

Share of **train company-months** with a raw `category=tax` booking. `share_co` = companies with ≥1 tax tx in that calendar month (any stacked year) / companies on the grid that month.

| month | Q? | n_cm | tax CM | tax companies | years |
| --- | --- | --- | --- | --- | --- |
| Jan | Q | 1,768 | 67.9% | 71.8% | 2025,2026 |
| Feb |  | 1,881 | 39.5% | 46.2% | 2025,2026 |
| Mar |  | 1,925 | 43.1% | 50.3% | 2025,2026 |
| Apr | Q | 1,945 | 68.6% | 74.2% | 2025,2026 |
| May |  | 1,966 | 43.7% | 50.9% | 2025,2026 |
| Jun |  | 1,976 | 42.5% | 49.3% | 2025,2026 |
| Jul | Q | 2,010 | 67.9% | 74.1% | 2025,2026 |
| Aug |  | 2,047 | 38.6% | 47.0% | 2025,2026 |
| Sep |  | 1,299 | 45.0% | 49.4% | 2024,2025 |
| Oct | Q | 1,381 | 72.4% | 75.8% | 2024,2025 |
| Nov |  | 1,428 | 46.1% | 53.0% | 2024,2025 |
| Dec |  | 1,531 | 48.9% | 54.0% | 2024,2025 |


Raw tax company-months (train): 10,953 / companies 1,098. Store vs raw agreement 100.0% — not an ops.py bug.

Plot: `tax_month_calendar.png`.

## 2. `c_missed_tax` — prevalence, acf, size, “just not a tax month?”

`c_missed_tax` train prevalence 8.3% (n=1,763 / 21,157). Modal value 0 share 91.7% (CONFIRM 91.7%). Store reconstruction agree 100.0%. P(missed | not tax month)=17.3% — NO, most non-tax months are not “missed” (no usual-tax history). P(usual | not tax)=17.3%. Missed share Q-months 2.2% vs other 10.9%. acf1=0.049 acf3=0.067 vs log1p(a_in3) ρ=0.007.

| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| prevalence (share=1) | 8.3% (n=1,763) |
| modal / modal share | 0 / 91.7% |
| confirm feature-report 91.7% | YES |
| `c_tax_month` share | 51.8% |
| store reconstruction agree | 100.0% |
| P(missed \| not tax month) | 17.3% |
| P(usual \| not tax month) | 17.3% |
| P(not tax \| usual) | 18.0% |
| acf1 / acf3 / acf6 | 0.049 / 0.067 / -0.108 |
| `c_tax_month` acf1 / acf3 | -0.125 / 0.277 |
| ρ vs log1p(a_in3) | 0.007 |
| missed Q-months / other | 2.2% / 10.9% |
| missed January / not-Jan | 2.6% / 8.9% |
| “just not a tax month”? | NO |
| calendar-dummy pattern? | YES |

Missed vs tax by calendar month (train CM):

| month | Q? | tax | missed | n_cm |
| --- | --- | --- | --- | --- |
| Jan | Q | 67.9% | 2.6% | 1,768 |
| Feb |  | 39.5% | 15.6% | 1,881 |
| Mar |  | 43.1% | 12.5% | 1,925 |
| Apr | Q | 68.6% | 2.2% | 1,945 |
| May |  | 43.7% | 13.6% | 1,966 |
| Jun |  | 42.5% | 11.9% | 1,976 |
| Jul | Q | 67.9% | 2.7% | 2,010 |
| Aug |  | 38.6% | 14.4% | 2,047 |
| Sep |  | 45.0% | 6.4% | 1,299 |
| Oct | Q | 72.4% | 1.4% | 1,381 |
| Nov |  | 46.1% | 6.2% | 1,428 |
| Dec |  | 48.9% | 6.4% | 1,531 |


## 3. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%; Y9 n=9,591 base 14.1%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711).

Y3 stressed singles (train group-fold): `c_missed_tax` 0.511 vs size 0.617 (Δ -0.105) vs days 0.711 (night 0.711, Δ 0.000). `c_tax_month` 0.611. `is_jan` 0.490 `is_q_month` 0.495 `not_tax_month` 0.611. Y2 missed 0.514 vs size 0.552; Y9 missed 0.506 vs size 0.534. Size≥0.60 on Y3: YES — PARK tax as X. Missed vs January dummy: not just January.

| y | feature | n | n_pos | CV | sd | sign | train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_missed_tax | 17,356 | 1,271 | 0.514 | 0.014 | -1 | 0.515 |
| y2_neg_2of3 | c_tax_month | 17,356 | 1,271 | 0.535 | 0.026 | 1 | 0.541 |
| y2_neg_2of3 | c_missed_salary | 17,356 | 1,271 | 0.494 | 0.009 | 1 | 0.503 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 |
| y2_neg_2of3 | is_q_month | 17,356 | 1,271 | 0.502 | 0.005 | -1 | 0.503 |
| y2_neg_2of3 | is_jan | 17,356 | 1,271 | 0.510 | 0.007 | 1 | 0.508 |
| y2_neg_2of3 | not_tax_month | 17,356 | 1,271 | 0.535 | 0.026 | -1 | 0.541 |
| y3_recover_cash_6m | c_missed_tax | 5,648 | 402 | 0.511 | 0.017 | 1 | 0.510 |
| y3_recover_cash_6m | c_tax_month | 5,648 | 402 | 0.611 | 0.031 | -1 | 0.617 |
| y3_recover_cash_6m | c_missed_salary | 5,648 | 402 | 0.513 | 0.010 | 1 | 0.512 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 |
| y3_recover_cash_6m | is_q_month | 5,648 | 402 | 0.495 | 0.012 | 1 | 0.503 |
| y3_recover_cash_6m | is_jan | 5,648 | 402 | 0.490 | 0.028 | -1 | 0.508 |
| y3_recover_cash_6m | not_tax_month | 5,648 | 402 | 0.611 | 0.031 | 1 | 0.617 |
| y9_fee_r_ownp80 | c_missed_tax | 9,591 | 1,350 | 0.506 | 0.011 | -1 | 0.506 |
| y9_fee_r_ownp80 | c_tax_month | 9,591 | 1,350 | 0.534 | 0.041 | 1 | 0.535 |
| y9_fee_r_ownp80 | c_missed_salary | 9,591 | 1,350 | 0.505 | 0.006 | 1 | 0.505 |
| y9_fee_r_ownp80 | c_n_days_with_tx | 9,591 | 1,350 | 0.565 | 0.057 | 1 | 0.567 |
| y9_fee_r_ownp80 | log1p_a_in3 | 9,591 | 1,350 | 0.534 | 0.018 | 1 | 0.534 |
| y9_fee_r_ownp80 | is_q_month | 9,591 | 1,350 | 0.512 | 0.015 | 1 | 0.512 |
| y9_fee_r_ownp80 | is_jan | 9,591 | 1,350 | 0.505 | 0.012 | 1 | 0.505 |
| y9_fee_r_ownp80 | not_tax_month | 9,591 | 1,350 | 0.534 | 0.041 | -1 | 0.535 |


KEEP-as-Q5 rule: missed-tax beats size by ≥0.02 **and** is not “just not January”. Size dummy ≥0.6 → PARK as X.

## 4. Leak vs `a_out6` / payroll month

Tax × salary 2×2: both 6,037 (28.5%), tax-only 4,916, salary-only 2,447, neither 7,757. P(tax|salary)=71.2% vs P(tax|no salary)=38.8%. Spearman tax↔salary 0.317 (payroll ≠ tax). tax↔a_out6 0.380 (not an outflow dummy).

| pair | Spearman |
| --- | --- |
| c_tax_month vs a_out6 | 0.380 |
| c_missed_tax vs a_out6 | 0.044 |
| c_salary_month vs a_out6 | 0.375 |
| c_tax_month vs c_salary_month | 0.317 |
| c_tax_month vs c_ss_month | 0.330 |
| c_missed_tax vs c_missed_salary | 0.075 |
| c_tax_month vs log1p(a_in3) | 0.357 |
| c_missed_tax vs log1p(a_in3) | 0.007 |


## 5. Dark 470 vs 744 — same bank-book tax rate?

Train last-month companies: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean company tax-CM rate: invoiced 51.8% vs dark 48.7%. Same bank-book tax rate. Holdout ever-ERP coverage only: 40/72.

Company-level (mean of each company's tax-CM rate):

| group | n_co | ever_tax | n_ever_tax | tax_cm | miss_cm | sal_cm | ss_cm | tax_p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 92.6% | 689 | 51.8% | 8.2% | 41.7% | 49.1% | 0.500 |
| never_erp_470 | 470 | 87.0% | 409 | 48.7% | 6.5% | 37.1% | 38.3% | 0.458 |


Company-month:

| group | n_cm | n_co | tax | missed |
| --- | --- | --- | --- | --- |
| ever_erp | 13,554 | 744 | 52.5% | 8.9% |
| never_erp | 7,603 | 470 | 50.5% | 7.3% |


## 6. Q6 — lag1 of `c_missed_tax` (1-month only)

Y3 contemporaneous missed 0.511 is chance; lag1 0.504. CLOSE as Q6 — there is no contemporaneous skill to lead.

| y | col | n | n_pos | present | CV | sign |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_missed_tax | 17,356 | 1,271 | 100.0% | 0.514 | -1 |
| y2_neg_2of3 | c_tax_month | 17,356 | 1,271 | 100.0% | 0.535 | 1 |
| y2_neg_2of3 | c_missed_tax_lag1 | 16,161 | 1,151 | 93.1% | 0.514 | -1 |
| y2_neg_2of3 | c_tax_month_lag1 | 16,161 | 1,151 | 93.1% | 0.543 | 1 |
| y3_recover_cash_6m | c_missed_tax | 5,648 | 402 | 100.0% | 0.511 | 1 |
| y3_recover_cash_6m | c_tax_month | 5,648 | 402 | 100.0% | 0.611 | -1 |
| y3_recover_cash_6m | c_missed_tax_lag1 | 5,648 | 402 | 100.0% | 0.504 | 1 |
| y3_recover_cash_6m | c_tax_month_lag1 | 5,648 | 402 | 100.0% | 0.597 | -1 |
| y9_fee_r_ownp80 | c_missed_tax | 9,591 | 1,350 | 100.0% | 0.506 | -1 |
| y9_fee_r_ownp80 | c_tax_month | 9,591 | 1,350 | 100.0% | 0.534 | 1 |
| y9_fee_r_ownp80 | c_missed_tax_lag1 | 9,591 | 1,350 | 100.0% | 0.492 | -1 |
| y9_fee_r_ownp80 | c_tax_month_lag1 | 9,591 | 1,350 | 100.0% | 0.527 | 1 |


## 7. VAT vs corporate-tax guess (amount size)

Train tax txs n=52,237 p50=419 p90=31,880. Q-month p50 676 vs other 377. Amount vs log1p(a_in3) ρ=0.455. Guess: **VAT-like (amounts similar across months)**. tax_refund txs n=2 p50=66,617.

Do **not** build a Y from this guess.

| month | Q? | n_tx | n_co | |amt| p50 | |amt| p90 |
| --- | --- | --- | --- | --- | --- |
| Jan | Q | 4,580 | 813 | 1,368 | 45,383 |
| Feb |  | 2,960 | 556 | 465 | 28,928 |
| Mar |  | 3,880 | 609 | 332 | 30,452 |
| Apr | Q | 6,008 | 899 | 658 | 41,076 |
| May |  | 4,611 | 618 | 175 | 12,468 |
| Jun |  | 4,852 | 599 | 197 | 18,159 |
| Jul | Q | 6,884 | 900 | 512 | 40,497 |
| Aug |  | 3,868 | 571 | 210 | 20,018 |
| Sep |  | 2,832 | 427 | 422 | 27,905 |
| Oct | Q | 5,336 | 688 | 694 | 34,970 |
| Nov |  | 2,963 | 501 | 464 | 21,192 |
| Dec |  | 3,463 | 543 | 632 | 34,944 |


tax_refund txs: 2 across 1 train companies; p50=66,617.

## 8. `tax_refund` months — recoveries? (not a Y)

tax_refund company-months: 2 / 1214 companies=1. Y3 rate on refund months — vs all labeled 7.1%; next-month Y3 —. Refund months are not a recovery flag. Do not build a Y.

| y | slice | n_cm | n_labeled | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | refund | 2 | 0 | 0 | — |
| y2_neg_2of3 | no_refund | 21,155 | 17,356 | 1,271 | 7.3% |
| y3_recover_cash_6m | refund | 2 | 0 | 0 | — |
| y3_recover_cash_6m | no_refund | 21,155 | 5,648 | 402 | 7.1% |
| y9_fee_r_ownp80 | refund | 2 | 0 | 0 | — |
| y9_fee_r_ownp80 | no_refund | 21,155 | 9,591 | 1,350 | 14.1% |


## 9. Residual skill inside the calendar

Y3 missed CV all 0.511; on Q-months 0.532; on non-Q 0.476; not-January 0.493. Residual inside calendar slices still measured.

| slice | feature | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| all | c_missed_tax | 5,648 | 402 | 0.511 |
| all | c_tax_month | 5,648 | 402 | 0.611 |
| all | log1p_a_in3 | 5,528 | 391 | 0.617 |
| all | is_q_month | 5,648 | 402 | 0.495 |
| Q_months | c_missed_tax | 1,870 | 135 | 0.532 |
| Q_months | c_tax_month | 1,870 | 135 | 0.639 |
| Q_months | log1p_a_in3 | 1,801 | 131 | 0.631 |
| Q_months | is_q_month | 1,870 | 135 | 0.500 |
| nonQ_months | c_missed_tax | 3,778 | 267 | 0.476 |
| nonQ_months | c_tax_month | 3,778 | 267 | 0.598 |
| nonQ_months | log1p_a_in3 | 3,727 | 260 | 0.609 |
| nonQ_months | is_q_month | 3,778 | 267 | 0.500 |
| not_January | c_missed_tax | 4,932 | 357 | 0.493 |
| not_January | c_tax_month | 4,932 | 357 | 0.603 |
| not_January | log1p_a_in3 | 4,825 | 347 | 0.616 |
| not_January | is_q_month | 4,932 | 357 | 0.508 |
| January | c_missed_tax | 716 | 45 | LOW_POWER |
| January | c_tax_month | 716 | 45 | LOW_POWER |
| January | log1p_a_in3 | 703 | 44 | LOW_POWER |
| January | is_q_month | 716 | 45 | LOW_POWER |
| usual_tax6 | c_missed_tax | 3,431 | 161 | 0.566 |
| usual_tax6 | c_tax_month | 3,431 | 161 | 0.566 |
| usual_tax6 | log1p_a_in3 | 3,431 | 161 | 0.596 |
| usual_tax6 | is_q_month | 3,431 | 161 | 0.479 |


Per calendar month (Y3 × `c_missed_tax`):

| month | n_pos | CV |
| --- | --- | --- |
| Jan | 45 | LOW_POWER |
| Feb | 53 | 0.447 |
| Mar | 22 | LOW_POWER |
| Apr | 20 | LOW_POWER |
| May | 22 | LOW_POWER |
| Jun | 28 | LOW_POWER |
| Jul | 37 | LOW_POWER |
| Aug | 30 | LOW_POWER |
| Sep | 24 | LOW_POWER |
| Oct | 33 | LOW_POWER |
| Nov | 44 | LOW_POWER |
| Dec | 44 | LOW_POWER |


## 10. Company tax cadence

Train companies by tax cadence: never 116, monthly 406, quarterly 183, irregular 509 / 1214.

| kind | n_co | share_co | tax_rate_p50 | miss_p50 | n_tax_p50 |
| --- | --- | --- | --- | --- | --- |
| irregular | 509 | 41.9% | 0.417 | 0.125 | 6.0 |
| monthly | 406 | 33.4% | 0.946 | 0.000 | 18.0 |
| never | 116 | 9.6% | 0.000 | 0.000 | 0.0 |
| quarterly | 183 | 15.1% | 0.333 | 0.000 | 6.0 |


monthly = tax in ≥70% of grid months. quarterly = ≥70% of tax months sit on Jan/Apr/Jul/Oct and tax-rate ≤45%. never = no tax tx. Else irregular.

## 11. Holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM. tax-month mean 51.3%; missed-tax mean 4.9%. No AUROC claim.

| col | n_cm | n_co | cov | mean |
| --- | --- | --- | --- | --- |
| c_tax_month | 1073 | 72 | 100.0% | 51.3% |
| c_missed_tax | 1073 | 72 | 100.0% | 4.9% |
| c_salary_month | 1073 | 72 | 100.0% | 37.8% |
| c_missed_salary | 1073 | 72 | 100.0% | 1.4% |


## 12. Honest skip — missed only among usual filers (tax6≥3)

Usual-tax CM (tax6≥3): 9,819 / 21,157 (46.4%); of those, missed 1,763 (18.0%). Y3 missed-on-usual CV 0.566 n_pos=161. Above chance is not KEEP — KEEP still requires beating size by ≥0.02.

| y | slice | n | n_pos | miss_share | CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | usual_tax6 | 7,892 | 643 | 17.7% | 0.536 |
| y2_neg_2of3 | usual_and_Q | 2,604 | 203 | 4.8% | 0.491 |
| y2_neg_2of3 | usual_and_nonQ | 5,288 | 440 | 24.1% | 0.557 |
| y3_recover_cash_6m | usual_tax6 | 3,431 | 161 | 15.1% | 0.566 |
| y3_recover_cash_6m | usual_and_Q | 1,136 | 54 | 3.6% | 0.599 |
| y3_recover_cash_6m | usual_and_nonQ | 2,295 | 107 | 20.8% | 0.553 |
| y9_fee_r_ownp80 | usual_tax6 | 5,561 | 836 | 19.1% | 0.514 |
| y9_fee_r_ownp80 | usual_and_Q | 1,917 | 295 | 5.1% | 0.512 |
| y9_fee_r_ownp80 | usual_and_nonQ | 3,644 | 541 | 26.5% | 0.515 |


## 13. `c_tax_month` 0.611 — activity dummy?

tax↔days ρ=0.428; tax↔log1p(a_in3) ρ=0.357. P(tax|days=0)=0.0% P(tax|days≥5)=60.1%. Y3 tax CV inside size terciles that stay ≥0.55: ['T3']. tax-month is an activity dummy.

| clock | tercile | n | n_pos | tax_share | tax CV |
| --- | --- | --- | --- | --- | --- |
| size_t | T1 | 1,572 | 269 | 45.3% | 0.543 |
| size_t | T2 | 2,044 | 73 | 61.4% | 0.538 |
| size_t | T3 | 2,032 | 60 | 77.7% | 0.612 |
| days_t | T1 | 1,575 | 271 | 43.7% | 0.537 |
| days_t | T2 | 2,112 | 79 | 61.5% | 0.581 |
| days_t | T3 | 1,961 | 52 | 79.5% | 0.622 |


## 14. Cadence × Y rates (and missed AUROC)

Y3 missed CV on monthly-cadence companies 0.509 n_pos=55. A skip among monthly filers would be the cleanest Q5; it is not.

| y | kind | n_cm | n_co | n_lab | n_pos | rate | tax | miss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | irregular | 9,051 | 509 | 7,390 | 386 | 5.2% | 37.8% | 12.9% |
| y2_neg_2of3 | monthly | 7,172 | 406 | 5,954 | 635 | 10.7% | 90.7% | 5.3% |
| y2_neg_2of3 | never | 1,529 | 116 | 1,156 | 107 | 9.3% | 0.0% | 0.0% |
| y2_neg_2of3 | quarterly | 3,405 | 183 | 2,856 | 143 | 5.0% | 30.0% | 6.4% |
| y3_recover_cash_6m | irregular | 9,051 | 509 | 2,122 | 234 | 11.0% | 37.8% | 12.9% |
| y3_recover_cash_6m | monthly | 7,172 | 406 | 2,464 | 55 | 2.2% | 90.7% | 5.3% |
| y3_recover_cash_6m | never | 1,529 | 116 | 173 | 27 | 15.6% | 0.0% | 0.0% |
| y3_recover_cash_6m | quarterly | 3,405 | 183 | 889 | 86 | 9.7% | 30.0% | 6.4% |
| y9_fee_r_ownp80 | irregular | 9,051 | 509 | 4,174 | 547 | 13.1% | 37.8% | 12.9% |
| y9_fee_r_ownp80 | monthly | 7,172 | 406 | 3,311 | 520 | 15.7% | 90.7% | 5.3% |
| y9_fee_r_ownp80 | never | 1,529 | 116 | 483 | 40 | 8.3% | 0.0% | 0.0% |
| y9_fee_r_ownp80 | quarterly | 3,405 | 183 | 1,623 | 243 | 15.0% | 30.0% | 6.4% |


| y | kind | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| y2_neg_2of3 | monthly | 5,954 | 635 | 0.511 |
| y2_neg_2of3 | quarterly | 2,856 | 143 | 0.517 |
| y2_neg_2of3 | irregular | 7,390 | 386 | 0.519 |
| y2_neg_2of3 | never | 1,156 | 107 | 0.500 |
| y3_recover_cash_6m | monthly | 2,464 | 55 | 0.509 |
| y3_recover_cash_6m | quarterly | 889 | 86 | 0.434 |
| y3_recover_cash_6m | irregular | 2,122 | 234 | 0.501 |
| y3_recover_cash_6m | never | 173 | 27 | LOW_POWER |


## 15. Tax amount mix (VAT vs IS)

Tax |amt| mix: <500 51.9%, >10k 17.9%. Companies ever ≥10k: 754/1098. Only-small (≥3 txs, max<2k): 119. Most mass is small — VAT-like withholdings, not a corporate-tax event.

| bin | n_tx | share_tx | p50 |
| --- | --- | --- | --- |
| <500 | 27136 | 51.9% | 76 |
| 500-2k | 8512 | 16.3% | 1,037 |
| 2k-10k | 7244 | 13.9% | 4,112 |
| 10k-50k | 5642 | 10.8% | 21,107 |
| >50k | 3703 | 7.1% | 136,280 |


## 16. Who files off-quarter?

Non-Q tax CM: 6,055. Monthly-cadence companies account for 69.2% of those filings. Off-quarter tax is the monthly book, not a health event.

| kind | n_tax_nq_cm | share | n_co |
| --- | --- | --- | --- |
| irregular | 1748 | 28.9% | 469 |
| monthly | 4189 | 69.2% | 406 |
| quarterly | 118 | 1.9% | 86 |


Q-month tax / missed by cadence:

| kind | n_cm | tax | miss |
| --- | --- | --- | --- |
| irregular | 3042 | 55.1% | 3.9% |
| monthly | 2407 | 96.3% | 1.8% |
| never | 512 | 0.0% | 0.0% |
| quarterly | 1143 | 79.0% | 0.0% |


## 17. Social-security calendar (monthly control)

SS Q vs other: 45.8% / 45.5% (flat — monthly payroll tax). Tax Q vs other: 69.2% / 43.4%. Tax is the calendar; SS is the monthly control.

| month | Q? | tax | ss | salary |
| --- | --- | --- | --- | --- |
| Jan | Q | 67.9% | 44.6% | 39.5% |
| Feb |  | 39.5% | 45.1% | 40.0% |
| Mar |  | 43.1% | 46.4% | 41.2% |
| Apr | Q | 68.6% | 46.4% | 41.4% |
| May |  | 43.7% | 46.5% | 39.2% |
| Jun |  | 42.5% | 46.2% | 39.6% |
| Jul | Q | 67.9% | 45.9% | 40.3% |
| Aug |  | 38.6% | 45.1% | 39.4% |
| Sep |  | 45.0% | 45.0% | 40.1% |
| Oct | Q | 72.4% | 46.4% | 40.7% |
| Nov |  | 46.1% | 45.4% | 40.1% |
| Dec |  | 48.9% | 44.6% | 39.6% |


## 18. Usual × missed Y rates

Y3 among usual filers: missed 8.5% (n_pos=44) vs filed 4.0% (gap 4.5%). A Q5 skip would show a large same-sign gap. It does not.

| y | slice | n_cm | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | usual_missed | 1,763 | 1,398 | 67 | 4.8% |
| y2_neg_2of3 | usual_filed | 8,056 | 6,494 | 576 | 8.9% |
| y2_neg_2of3 | not_usual | 11,338 | 9,464 | 628 | 6.6% |
| y3_recover_cash_6m | usual_missed | 1,763 | 518 | 44 | 8.5% |
| y3_recover_cash_6m | usual_filed | 8,056 | 2,913 | 117 | 4.0% |
| y3_recover_cash_6m | not_usual | 11,338 | 2,217 | 241 | 10.9% |
| y9_fee_r_ownp80 | usual_missed | 1,763 | 1,064 | 137 | 12.9% |
| y9_fee_r_ownp80 | usual_filed | 8,056 | 4,497 | 699 | 15.5% |
| y9_fee_r_ownp80 | not_usual | 11,338 | 4,030 | 514 | 12.8% |


## 19. Large-tax month (≥10k) — not a Y

Large-tax CM (≥10k sum): 5,399 / 1214 companies=788. Y3 CV 0.589. Do not build a corporate-tax Y from amount size.

| y | n_large_lab | n_pos_large | rate_large | rate_other | CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | 4,448 | 390 | 8.8% | 6.8% | 0.534 |
| y3_recover_cash_6m | 1,874 | 68 | 3.6% | 8.9% | 0.589 |
| y9_fee_r_ownp80 | 2,470 | 408 | 16.5% | 13.2% | 0.527 |


## 20. Quarterly companies on Q-months + Jan→Feb

Quarterly companies on Q-months: tax 79.0% miss 0.0%. Jan filers who also file in Feb: 49.1% (Feb base 39.5%). January is not a unique dummy — Feb is the complementary hole of the Q peak.

| slice | n_cm | n_co | tax | miss |
| --- | --- | --- | --- | --- |
| q_co_on_Q | 1143 | 183 | 79.0% | 0.0% |
| q_co_on_Q_usual | 152 | 70 | 100.0% | 0.0% |
| q_co_on_nonQ | 2262 | 183 | 5.2% | 9.7% |
| monthly_on_any | 7172 | 406 | 90.7% | 5.3% |


| y | slice | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| y2_neg_2of3 | q_co_on_Q | 960 | 45 | LOW_POWER |
| y3_recover_cash_6m | q_co_on_Q | 305 | 28 | LOW_POWER |


## 21. Who produces missed-tax? (quarterly cannot)

Of 1,763 missed CM: irregular 1,165 (66.1%), monthly 379, quarterly 219. Quarterly companies almost never reach tax6≥3, so `c_missed_tax` cannot flag a quarterly skip. The flag is the irregular/monthly complementary hole, not a filing-date miss.

| kind | n_miss | share | n_co | on_Q |
| --- | --- | --- | --- | --- |
| irregular | 1165 | 66.1% | 310 | 10.2% |
| monthly | 379 | 21.5% | 153 | 11.3% |
| quarterly | 219 | 12.4% | 71 | 0.0% |


tax6 by cadence (usual = tax6≥3):

| kind | n_cm | tax6_p50 | share_tax6_ge3 | share_tax6_ge4 |
| --- | --- | --- | --- | --- |
| irregular | 9051 | 2.00 | 36.6% | 19.7% |
| monthly | 7172 | 6.00 | 84.5% | 77.2% |
| never | 1529 | 0.00 | 0.0% | 0.0% |
| quarterly | 3405 | 2.00 | 13.1% | 1.1% |


## 22. Irregular-only singles (66% of missed)

Irregular-only Y3: missed 0.501 vs size 0.585. The 66% pile is still not a Q5.

| y | feature | n | n_pos | CV |
| --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_missed_tax | 7,390 | 386 | 0.519 |
| y2_neg_2of3 | c_tax_month | 7,390 | 386 | 0.540 |
| y2_neg_2of3 | log1p_a_in3 | 6,404 | 299 | 0.463 |
| y2_neg_2of3 | c_n_days_with_tx | 7,390 | 386 | 0.558 |
| y3_recover_cash_6m | c_missed_tax | 2,122 | 234 | 0.501 |
| y3_recover_cash_6m | c_tax_month | 2,122 | 234 | 0.536 |
| y3_recover_cash_6m | log1p_a_in3 | 2,076 | 229 | 0.585 |
| y3_recover_cash_6m | c_n_days_with_tx | 2,122 | 234 | 0.635 |
| y9_fee_r_ownp80 | c_missed_tax | 4,174 | 547 | 0.510 |
| y9_fee_r_ownp80 | c_tax_month | 4,174 | 547 | 0.536 |
| y9_fee_r_ownp80 | log1p_a_in3 | 4,174 | 547 | 0.451 |
| y9_fee_r_ownp80 | c_n_days_with_tx | 4,174 | 547 | 0.564 |


## What failed / next (held for wave note)

- honest skip Y3 0.566 < size 0.617; large-tax 0.589 also loses. PARK as X.
- quarterly companies on Q-months miss 0% — tax6≥3 cannot fire for a 4-per-year book. Do not invent a Y; do not patch ops.py tonight.

Elapsed 6s. Cuts: calendar, missed prevalence, singles, leak, dark 470/744, Q6 lag1, VAT/IS amounts, refund, residual calendar, cadence, holdout, honest skip, activity terciles, cadence×Y, amount mix, non-Q who, SS control, usual×missed rates, large-tax, Q-company skip + Jan→Feb.
