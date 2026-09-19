# Q3/Q5 missed-salary — shock, quiet twin, or calendar dummy?

Generated `2026-09-19T04:12:29+02:00` by agent `0fd41cbf`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent a missed-salary Y. Do not put `c_missed_salary` on the 15-col card. Night Y3 quote stays **0.762 / 0.752**.

`c_missed_salary` = usual salary in last ≤6 months (rolling sum ≥ 3, min_periods=1, **includes current month**) AND `c_salary_month` = 0. Token is raw `category = 'salary'`. Feature report: RARE, modal 97.1%. `c_salary_month` / `c_ss_month` already sit on the 15-col Y3 card (quiet-stressed recover). Tax cousin CLOSED as calendar dummy — not redone.

## Headline

`c_missed_salary` train prev 2.9% (modal 97.1%, CONFIRM 97.1%). Calendar **monthly**. Y3 missed 0.513 vs size 0.617 (Δ -0.104) vs days 0.711. ICC 0.736 (between). Y6 Jaccard 0.365. X **CLOSE**. PARK as health Y. Q6 **CLOSE**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not this flag. PARK as a health Y. Cadence: 431 never / 410 monthly / 161 irregular. |
| 2 | Who is improving? | Not this flag. A skip is not a recovery. |
| 3 | Who is turning? | Missed usual payroll this month. Y3 CV 0.513 vs size 0.617 / days 0.711. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | **CLOSE** — Y3 0.513 vs size 0.617 (Δ -0.104) vs days 0.711. `c_salary_month` already on the 15-col card at 0.671; missed-salary does not beat that bar or size. |
| 6 | Months earlier? | lag1/lag3 **CLOSE** — contemporaneous Y3 0.513 is chance; lag1 0.512 / lag3 0.486 have nothing to lead. Short books are not empty (72 miss / 2,970 CM, 2.4%) but Y3 there is LOW_POWER. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `c_missed_salary` as Y3 / Q3-Q5 X | **CLOSE** | Y3 0.513 vs size 0.617 (Δ -0.104) vs days 0.711. `c_salary_month` already on the 15-col card at 0.671; missed-salary does not beat that bar or size. Still **not** on tonight's 15-col card. |
| `c_missed_salary` as a health Y | **PARK** | do not invent `y_missed_salary`; Y6 future-miss already rejected |
| missed-salary as calendar dummy | **no — not Q/Aug peaked like tax** | shape=monthly; Q-sal 40.5% vs other 39.9% |
| Q6 lag1/lag3 `c_missed_salary` | **CLOSE** | Q6 Y3 now 0.513 lag1 0.512 lag3 0.486; short now — lag1 —; long lag1 0.512. Short-book miss share 2.4%. **CLOSE** — contemporaneous Y3 0.513 is chance; lag1 0.512 / lag3 0.486 have nothing to lead. Short books are not empty (72 miss / 2,970 CM, 2.4%) but Y3 there is LOW_POWER. |
| leak twin | **none** | Leak screen |ρ|≥0.80 = twin; |ρ| vs log1p(a_in3) ≥0.50 = SIZE. No |ρ|≥0.80 twin vs the listed stems. Not a size clone. vs c_missed_tax ρ=0.075 (tax cousin quote 0.075). |
| vs rejected `y6_missed_payroll` | **distinct** | Jaccard 0.365 ρ 0.505 |

## 1. Completeness + base rate (train rates; holdout coverage)

Train `c_salary_month` share 40.1% (n=8,484 / 21,157). `c_missed_salary` prevalence 2.9% (n=623); modal 0 share 97.1% (CONFIRM 97.1%). Holdout coverage only: missed mean 1.4% on 1,073 CM / 72 cos. No AUROC on holdout.

| split | col | n_cm | n_co | cov | prev | n_pos | modal | modal% |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | c_salary_month | 21,157 | 1,214 | 100.0% | 40.1% | 8,484 | 0 | 59.9% |
| train | c_missed_salary | 21,157 | 1,214 | 100.0% | 2.9% | 623 | 0 | 97.1% |
| holdout | c_salary_month | 1,073 | 72 | 100.0% | 37.8% | 406 | 0 | 62.2% |
| holdout | c_missed_salary | 1,073 | 72 | 100.0% | 1.4% | 15 | 0 | 98.6% |


Confirm feature-report modal 97.1%: **YES**.

## 2. Formula vs raw `category = 'salary'`

Store `c_salary_month` vs raw category=salary agreement 100.0%. Reconstructed missed (roll-6 include current, min_periods=1) vs store 100.0%. Include-vs-exclude-current disagree on 144 CM (incl-only 0, excl-only 144). Quirk: the current 0 occupies a slot in the 6-month usual-sum, so include-current is stricter (needs ≥3 salary in the other 5). Exclude-current would add those excl-only months — do not rewrite ops.py.

Raw salary CM (train): 8,484 / companies 783. Formula OK: **YES**.

Sample company-months (window is last ≤6 including current, 1=salary):

| company_id | period | store_sal | raw_sal | store_miss | recon_incl | recon_excl | win_sal | win_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| COMP_0001 | 2026-06-01 | 0 | 0 | 1 | 1 | 1 | 1,1,1,1,1,0 | 6 |
| COMP_0004 | 2026-07-01 | 0 | 0 | 1 | 1 | 1 | 0,1,1,1,1,0 | 6 |
| COMP_0001 | 2026-01-01 | 1 | 1 | 0 | 0 | 0 | 1 | 1 |
| COMP_0001 | 2026-02-01 | 1 | 1 | 0 | 0 | 0 | 1,1 | 2 |
| COMP_0039 | 2026-04-01 | 0 | 0 | 0 | 0 | 1 | 1,1,0,0,0,0 | 6 |
| COMP_0042 | 2025-04-01 | 0 | 0 | 0 | 0 | 1 | 1,0,0,0,1,0 | 6 |


## 3. Calendar — month-of-year / August / weekday of last salary

Salary CM share Q-months 40.5% vs other 39.9% (ratio 1.01). Missed Q 2.8% vs other 2.9%; Aug missed 3.8% vs other-month mean 2.7%. Peak salary Apr 41.4%, trough May 39.2%. Peak missed Jun 4.1%. Shape **monthly**. Last-salary weekday weekend share 0.8%.

| month | Q? | n_cm | salary | missed |
| --- | --- | --- | --- | --- |
| Jan | Q | 1,768 | 39.5% | 2.7% |
| Feb |  | 1,881 | 40.0% | 2.8% |
| Mar |  | 1,925 | 41.2% | 2.9% |
| Apr | Q | 1,945 | 41.4% | 3.0% |
| May |  | 1,966 | 39.2% | 3.7% |
| Jun |  | 1,976 | 39.6% | 4.1% |
| Jul | Q | 2,010 | 40.3% | 3.5% |
| Aug | Aug | 2,047 | 39.4% | 3.8% |
| Sep |  | 1,299 | 40.1% | 1.5% |
| Oct | Q | 1,381 | 40.7% | 2.0% |
| Nov |  | 1,428 | 40.1% | 2.0% |
| Dec |  | 1,531 | 39.6% | 2.0% |


Weekday of last in-month salary booking (train):

| wd | n_cm | share |
| --- | --- | --- |
| Mon | 1560 | 0.18387553041018387 |
| Tue | 1356 | 0.15983026874115983 |
| Wed | 1391 | 0.16395568128241395 |
| Thu | 2280 | 0.26874115983026875 |
| Fri | 1827 | 0.21534653465346534 |
| Sat | 40 | 0.004714757190004715 |
| Sun | 30 | 0.003536067892503536 |


Plot: `salary_calendar.png`.

## 4. Co-occurrence with SS / tax / days (quiet month?)

Missed CM: P(ss)=46.9% vs salary-month 83.4%; P(days=0)=4.0% vs all 4.2%; days p50 missed 17.0 vs salary 20.0 vs all 14.0. Missed ∩ ss 292; missed ∩ tax 364. Not a pure quiet-month (still some activity / SS).

| slice | n | P(ss) | P(tax) | days p50 | P(days=0) |
| --- | --- | --- | --- | --- | --- |
| all | 21,157 | 45.7% | 51.8% | 14.0 | 4.2% |
| missed | 623 | 46.9% | 58.4% | 17.0 | 4.0% |
| salary | 8,484 | 83.4% | 71.2% | 20.0 | 0.0% |
| no_salary | 12,673 | 20.4% | 38.8% | 9.0 | 7.0% |


## 5. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 stressed n=5,648 base 7.1%; Y9 n=9,591 base 14.1%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711). Size quote 0.617 (replica 0.617). Night Y3 GBM **0.762 / core 0.752** — not re-fit.

Y3 stressed singles: `c_missed_salary` 0.513 vs size 0.617 (Δ -0.104) vs days 0.711 (night 0.711, replica Δ 0.000). `c_salary_month` 0.671 `c_ss_month` 0.693 `not_salary` 0.671 `is_aug` 0.506. Y2 missed 0.494 vs size 0.552 / days 0.571. Y9 missed 0.505. Fold wander Y3 missed 0.024. KEEP-gate closed.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_missed_salary | 17,356 | 1,271 | 0.494 | 0.009 | 1 | 0.503 | 0.508 0.485 0.495 0.486 0.496 |
| y2_neg_2of3 | c_salary_month | 17,356 | 1,271 | 0.522 | 0.055 | 1 | 0.513 | 0.511 0.572 0.535 0.432 0.559 |
| y2_neg_2of3 | c_ss_month | 17,356 | 1,271 | 0.539 | 0.043 | -1 | 0.531 | 0.509 0.562 0.516 0.604 0.505 |
| y2_neg_2of3 | c_missed_tax | 17,356 | 1,271 | 0.514 | 0.014 | -1 | 0.515 | 0.505 0.502 0.525 0.504 0.534 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 | 0.623 0.539 0.612 0.565 0.517 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 | 0.523 0.609 0.595 0.520 0.513 |
| y2_neg_2of3 | is_aug | 17,356 | 1,271 | 0.497 | 0.004 | 1 | 0.500 | 0.494 0.499 0.491 0.499 0.499 |
| y2_neg_2of3 | is_q_month | 17,356 | 1,271 | 0.502 | 0.005 | -1 | 0.503 | 0.506 0.498 0.496 0.502 0.506 |
| y2_neg_2of3 | not_salary | 17,356 | 1,271 | 0.522 | 0.055 | -1 | 0.513 | 0.511 0.572 0.535 0.432 0.559 |
| y3_recover_cash_6m | c_missed_salary | 5,648 | 402 | 0.513 | 0.010 | 1 | 0.512 | 0.531 0.510 0.507 0.507 0.508 |
| y3_recover_cash_6m | c_salary_month | 5,648 | 402 | 0.671 | 0.050 | -1 | 0.672 | 0.611 0.659 0.745 0.689 0.652 |
| y3_recover_cash_6m | c_ss_month | 5,648 | 402 | 0.693 | 0.031 | -1 | 0.690 | 0.673 0.667 0.745 0.687 0.694 |
| y3_recover_cash_6m | c_missed_tax | 5,648 | 402 | 0.511 | 0.017 | 1 | 0.510 | 0.532 0.504 0.487 0.517 0.515 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | is_aug | 5,648 | 402 | 0.506 | 0.013 | 1 | 0.506 | 0.513 0.522 0.504 0.486 0.503 |
| y3_recover_cash_6m | is_q_month | 5,648 | 402 | 0.495 | 0.012 | 1 | 0.503 | 0.514 0.487 0.494 0.498 0.483 |
| y3_recover_cash_6m | not_salary | 5,648 | 402 | 0.671 | 0.050 | 1 | 0.672 | 0.611 0.659 0.745 0.689 0.652 |
| y9_fee_r_ownp80 | c_missed_salary | 9,591 | 1,350 | 0.505 | 0.006 | 1 | 0.505 | 0.513 0.499 0.508 0.505 0.500 |
| y9_fee_r_ownp80 | c_salary_month | 9,591 | 1,350 | 0.534 | 0.052 | 1 | 0.530 | 0.566 0.585 0.500 0.556 0.461 |
| y9_fee_r_ownp80 | c_ss_month | 9,591 | 1,350 | 0.533 | 0.026 | 1 | 0.532 | 0.539 0.570 0.517 0.535 0.503 |
| y9_fee_r_ownp80 | c_missed_tax | 9,591 | 1,350 | 0.506 | 0.011 | -1 | 0.506 | 0.500 0.510 0.513 0.516 0.490 |
| y9_fee_r_ownp80 | c_n_days_with_tx | 9,591 | 1,350 | 0.565 | 0.057 | 1 | 0.567 | 0.586 0.618 0.468 0.585 0.568 |
| y9_fee_r_ownp80 | log1p_a_in3 | 9,591 | 1,350 | 0.534 | 0.018 | 1 | 0.534 | 0.563 0.539 0.521 0.523 0.523 |
| y9_fee_r_ownp80 | is_aug | 9,591 | 1,350 | 0.504 | 0.008 | -1 | 0.504 | 0.513 0.508 0.496 0.508 0.494 |
| y9_fee_r_ownp80 | is_q_month | 9,591 | 1,350 | 0.512 | 0.015 | 1 | 0.512 | 0.529 0.524 0.493 0.513 0.500 |
| y9_fee_r_ownp80 | not_salary | 9,591 | 1,350 | 0.534 | 0.052 | -1 | 0.530 | 0.566 0.585 0.500 0.556 0.461 |


KEEP-as-X gate: beat size by ≥0.02 **and** not a calendar / size / 12-name / salary_month twin. Size dummy ≥0.6 is a bar, not a keep.

## 6. Leak screens

Leak screen |ρ|≥0.80 = twin; |ρ| vs log1p(a_in3) ≥0.50 = SIZE. No |ρ|≥0.80 twin vs the listed stems. Not a size clone. vs c_missed_tax ρ=0.075 (tax cousin quote 0.075).

| pair | Spearman | flag |
| --- | --- | --- |
| c_missed_salary vs c_salary_month | -0.143 |  |
| c_missed_salary vs c_ss_month | 0.004 |  |
| c_missed_salary vs c_n_days_with_tx | 0.038 |  |
| c_missed_salary vs a_n_tx | 0.041 |  |
| c_missed_salary vs a_out6 | 0.060 |  |
| c_missed_salary vs log1p(a_in3) | 0.042 |  |
| c_missed_salary vs c_missed_tax | 0.075 |  |
| c_missed_salary vs c_tax_month | 0.023 |  |
| c_missed_salary vs not_salary | 0.143 |  |
| c_salary_month vs c_n_days_with_tx | 0.419 |  |


## 7. ICC / company-demean (trait vs month shock)

`c_missed_salary` ICC=0.736 (k=1214); `c_salary_month` ICC=0.978. Y3 raw 0.513 vs company-demean 0.551 (drop -0.039). Ever-miss companies 233/1214. mixed / between.

| item | value |
| --- | ---: |
| ICC `c_missed_salary` | 0.736 |
| ICC `c_salary_month` | 0.978 |
| Y3 raw / demean | 0.513 / 0.551 |
| ever-miss companies | 233 / 1,214 |
| trait / shock | False / False |

Uncat-style ICC 0.985 is a bookkeeping trait. Feature-report missed ICC 0.74 is BETWEEN.

## 8. Dark 470 vs 744 invoiced

Train last-month: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean company miss-CM: invoiced 2.7% vs dark 3.1%. Same bank-book miss rate. Holdout ever-ERP coverage only: 40/72.

| group | n_co | ever_sal | sal_cm | miss_cm | ss_cm |
| --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 65.3% | 41.7% | 2.7% | 49.1% |
| never_erp_470 | 470 | 63.2% | 37.1% | 3.1% | 38.3% |


| group | n_cm | n_co | salary | missed |
| --- | --- | --- | --- | --- |
| ever_erp | 13,554 | 744 | 42.4% | 2.7% |
| never_erp | 7,603 | 470 | 36.0% | 3.3% |


Y3 missed CV invoiced 0.514 / dark 0.496.

## 9. Drop 12 chronic dark Y2 names (GROUP_0158 / GROUP_0172)

Chronic 12 names (0158/0172, ≥50% labeled months below 0): 12. Y2 missed CV full 0.494 → drop-12 0.492. Size 0.552 → 0.545; days 0.571 → 0.549. Miss share on 12 5.6% vs rest 2.9%. Drop does not flip Y2 (≥0.03).

n_ids=12. Y2 full 0.494 → drop-12 0.492.

## 10. Complementary 2×2 vs `c_salary_month` and days terciles

2×2 missed×salary: both 0 (must be 0), miss-only 623, salary-only 8,484, neither 12,050 / 21,157. Y3 missed CV inside size terciles: T1=0.514, T2=0.506, T3=0.482. Among not-salary months Y3 0.490.

| y | slice | n_cm | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | missed | 623 | 465 | 40 | 8.6% |
| y2_neg_2of3 | salary | 8,484 | 7,030 | 546 | 7.8% |
| y2_neg_2of3 | no_sal_not_miss | 12,050 | 9,861 | 685 | 6.9% |
| y3_recover_cash_6m | missed | 623 | 198 | 23 | 11.6% |
| y3_recover_cash_6m | salary | 8,484 | 3,000 | 85 | 2.8% |
| y3_recover_cash_6m | no_sal_not_miss | 12,050 | 2,450 | 294 | 12.0% |
| y9_fee_r_ownp80 | missed | 623 | 359 | 62 | 17.3% |
| y9_fee_r_ownp80 | salary | 8,484 | 3,953 | 626 | 15.8% |
| y9_fee_r_ownp80 | no_sal_not_miss | 12,050 | 5,279 | 662 | 12.5% |


| clock | tercile | n | n_pos | miss | CV |
| --- | --- | --- | --- | --- | --- |
| size_t | T1 | 1,572 | 269 | 3.2% | 0.514 |
| size_t | T2 | 2,044 | 73 | 3.3% | 0.506 |
| size_t | T3 | 2,032 | 60 | 3.9% | 0.482 |
| days_t | T1 | 1,575 | 271 | 3.7% | 0.517 |
| days_t | T2 | 2,112 | 79 | 2.8% | 0.509 |
| days_t | T3 | 1,961 | 52 | 4.0% | 0.517 |


## 11. Q6 — lag1 / lag3 on short vs long books

Q6 Y3 now 0.513 lag1 0.512 lag3 0.486; short now — lag1 —; long lag1 0.512. Short-book miss share 2.4%. **CLOSE** — contemporaneous Y3 0.513 is chance; lag1 0.512 / lag3 0.486 have nothing to lead. Short books are not empty (72 miss / 2,970 CM, 2.4%) but Y3 there is LOW_POWER.

| y | slice | col | n | n_pos | x_mean | CV |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | all | c_missed_salary | 17,356 | 1,271 | 2.7% | 0.494 |
| y2_neg_2of3 | all | c_missed_salary_lag1 | 16,161 | 1,151 | 2.6% | 0.492 |
| y2_neg_2of3 | all | c_missed_salary_lag3 | 13,776 | 938 | 2.5% | 0.500 |
| y2_neg_2of3 | all | c_salary_month | 17,356 | 1,271 | 40.5% | 0.522 |
| y2_neg_2of3 | short_<12 | c_missed_salary | 1,833 | 147 | 1.0% | 0.505 |
| y2_neg_2of3 | short_<12 | c_missed_salary_lag1 | 1,497 | 108 | 0.7% | 0.503 |
| y2_neg_2of3 | short_<12 | c_missed_salary_lag3 | 830 | 46 | 0.2% | LOW_POWER |
| y2_neg_2of3 | short_<12 | c_salary_month | 1,833 | 147 | 39.5% | 0.461 |
| y2_neg_2of3 | long_>=12 | c_missed_salary | 15,523 | 1,124 | 2.9% | 0.494 |
| y2_neg_2of3 | long_>=12 | c_missed_salary_lag1 | 14,664 | 1,043 | 2.8% | 0.500 |
| y2_neg_2of3 | long_>=12 | c_missed_salary_lag3 | 12,946 | 892 | 2.6% | 0.500 |
| y2_neg_2of3 | long_>=12 | c_salary_month | 15,523 | 1,124 | 40.6% | 0.487 |
| y2_neg_2of3 | so_far>=4 | c_missed_salary | 13,776 | 938 | 3.4% | 0.500 |
| y2_neg_2of3 | so_far>=4 | c_missed_salary_lag1 | 13,776 | 938 | 3.0% | 0.500 |
| y2_neg_2of3 | so_far>=4 | c_missed_salary_lag3 | 13,776 | 938 | 2.5% | 0.500 |
| y2_neg_2of3 | so_far>=4 | c_salary_month | 13,776 | 938 | 41.2% | 0.534 |
| y3_recover_cash_6m | all | c_missed_salary | 5,648 | 402 | 3.5% | 0.513 |
| y3_recover_cash_6m | all | c_missed_salary_lag1 | 5,648 | 402 | 3.2% | 0.512 |
| y3_recover_cash_6m | all | c_missed_salary_lag3 | 5,078 | 355 | 2.7% | 0.486 |
| y3_recover_cash_6m | all | c_salary_month | 5,648 | 402 | 53.1% | 0.671 |
| y3_recover_cash_6m | short_<12 | c_missed_salary | 146 | 13 | 0.7% | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_missed_salary_lag1 | 146 | 13 | 0.0% | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_missed_salary_lag3 | 53 | 1 | 0.0% | LOW_POWER |
| y3_recover_cash_6m | short_<12 | c_salary_month | 146 | 13 | 43.8% | LOW_POWER |
| y3_recover_cash_6m | long_>=12 | c_missed_salary | 5,502 | 389 | 3.6% | 0.513 |
| y3_recover_cash_6m | long_>=12 | c_missed_salary_lag1 | 5,502 | 389 | 3.3% | 0.512 |
| y3_recover_cash_6m | long_>=12 | c_missed_salary_lag3 | 5,025 | 354 | 2.7% | 0.486 |
| y3_recover_cash_6m | long_>=12 | c_salary_month | 5,502 | 389 | 53.4% | 0.669 |
| y3_recover_cash_6m | so_far>=4 | c_missed_salary | 5,078 | 355 | 3.9% | 0.514 |
| y3_recover_cash_6m | so_far>=4 | c_missed_salary_lag1 | 5,078 | 355 | 3.6% | 0.513 |
| y3_recover_cash_6m | so_far>=4 | c_missed_salary_lag3 | 5,078 | 355 | 2.7% | 0.486 |
| y3_recover_cash_6m | so_far>=4 | c_salary_month | 5,078 | 355 | 53.7% | 0.669 |


## 12. vs rejected `y6_missed_payroll` (future t+1..t+3)

Y6 overlap n=6,004; Jaccard 0.365 Spearman 0.505. P(Y6|missed)=47.7% vs Y6 base 6.1%. Y3 using Y6-as-X 0.626. X does not leak y6_missed_payroll (different window: now vs t+1..t+3).

| item | value |
| --- | --- |
| overlap n | 6,004 |
| c_missed_salary=1 on overlap | 470 |
| y6_missed_payroll=1 on overlap | 367 |
| intersection | 224 |
| Jaccard | 0.365 |
| Spearman | 0.505 |
| P(Y6|missed X) | 47.7% |
| P(missed X|Y6) | 61.0% |
| Y6 base on overlap | 6.1% |


Y6 looks forward. `c_missed_salary` is contemporaneous X. Do not merge them.

## 13. Amounts — payroll mass or 1€ token?

Train salary txs n=40,988 |amt| p50=1,977 p90=44,224 share≤1€ 0.3%. On salary months, salary / |out-proxy| p50 8.8%. SS txs p50=2,366. real payroll mass (not a 1€ token).

| bin | n_tx | share_tx | p50 |
| --- | --- | --- | --- |
| <=1 | 121 | 0.3% | 0 |
| 1-100 | 2625 | 6.4% | 45 |
| 100-1k | 9107 | 22.2% | 402 |
| 1k-5k | 16840 | 41.1% | 1,946 |
| >5k | 12295 | 30.0% | 24,455 |


Salary companies 783. SS |amt| p50 2,366.

## 14. Company payroll cadence

Train payroll cadence: never 431, monthly 410, irregular 161, rare 212 / 1214.

| kind | n_co | share_co | sal_p50 | miss_p50 |
| --- | --- | --- | --- | --- |
| irregular | 161 | 13.3% | 0.444 | 0.130 |
| monthly | 410 | 33.8% | 1.000 | 0.000 |
| never | 431 | 35.5% | 0.000 | 0.000 |
| rare | 212 | 17.5% | 0.100 | 0.000 |


monthly = salary in ≥70% of grid months. irregular = 25–70% and ≥3 salary months. never = none. Else rare.

## 15. Honest skip — usual (sal6≥3) × missed

Usual-salary CM (sal6≥3): 7,478 / 21,157; of those missed 623 (8.3%). Y3 missed-on-usual 0.591. Y3 rate usual-missed 11.6% vs usual-paid 2.7%.

| y | slice | n | n_lab | n_pos | rate | CV |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | usual_sal6 | 5,987 | 5,987 | 441 | 7.4% | 0.475 |
| y2_neg_2of3 | usual_missed | 465 | 465 | 40 | 8.6% | LOW_POWER |
| y2_neg_2of3 | usual_paid | 5,522 | 5,522 | 401 | 7.3% | 0.500 |
| y3_recover_cash_6m | usual_sal6 | 2,896 | 2,896 | 95 | 3.3% | 0.591 |
| y3_recover_cash_6m | usual_missed | 198 | 198 | 23 | 11.6% | LOW_POWER |
| y3_recover_cash_6m | usual_paid | 2,698 | 2,698 | 72 | 2.7% | 0.500 |
| y9_fee_r_ownp80 | usual_sal6 | 4,001 | 4,001 | 619 | 15.5% | 0.508 |
| y9_fee_r_ownp80 | usual_missed | 359 | 359 | 62 | 17.3% | 0.500 |
| y9_fee_r_ownp80 | usual_paid | 3,642 | 3,642 | 557 | 15.3% | 0.500 |


## 16. Who produces missed-salary?

Of 623 missed CM: monthly 172, irregular 393. A monthly booker skip is the cleanest Q5; count it separately from rare/irregular holes.

| kind | n_miss | share | n_co | days_p50 |
| --- | --- | --- | --- | --- |
| irregular | 393 | 63.1% | 128 | 18.0 |
| monthly | 172 | 27.6% | 81 | 14.0 |
| rare | 58 | 9.3% | 24 | 16.0 |


## 17. Holdout coverage only (no AUROC)

Holdout 72 coverage only: 1,073 CM. salary-month mean 37.8%; missed-salary mean 1.4%. No AUROC claim.

| col | n_cm | n_co | cov | mean |
| --- | --- | --- | --- | --- |
| c_salary_month | 1073 | 72 | 100.0% | 37.8% |
| c_missed_salary | 1073 | 72 | 100.0% | 1.4% |
| c_ss_month | 1073 | 72 | 100.0% | 45.6% |
| c_n_days_with_tx | 1073 | 72 | 100.0% | 14.726 |


## 18. Y3 fold table (wander)

Y3 fold wander missed 0.024. Signs from train side of each fold.

| fold | missed | salary | days | size | n_pos_miss |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.531 | 0.611 | 0.665 | 0.565 | 54 |
| 1 | 0.510 | 0.659 | 0.738 | 0.632 | 93 |
| 2 | 0.507 | 0.745 | 0.700 | 0.683 | 66 |
| 3 | 0.507 | 0.689 | 0.715 | 0.543 | 82 |
| 4 | 0.508 | 0.652 | 0.740 | 0.661 | 107 |


## 19. SS packet when salary is missed

P(ss|missed)=46.9% vs P(ss|salary)=83.4%. SS often continues when salary is missed (token skip, not payroll halt).

| slice | n | P(ss) | P(tax) |
| --- | --- | --- | --- |
| missed | 623 | 46.9% | 58.4% |
| salary | 8484 | 83.4% | 71.2% |
| ss_no_salary | 2590 | 100.0% | 59.3% |


## 20. Residual after quiet months

Y3 missed on days≥3 0.513 (vs all 0.513). If skill vanishes once the month is busy, it is the quiet-month twin already on the 15-col card.

| slice | n | n_pos | miss | CV |
| --- | --- | --- | --- | --- |
| all | 5,648 | 402 | 3.5% | 0.513 |
| days>=1 | 5,536 | 372 | 3.5% | 0.511 |
| days>=3 | 5,259 | 310 | 3.6% | 0.513 |
| days>=5 | 4,978 | 263 | 3.7% | 0.511 |
| has_ss | 3,383 | 99 | 3.3% | 0.510 |
| no_ss | 2,265 | 303 | 3.8% | 0.514 |


## 21. Usual-only singles vs size / days

Usual-only Y3: missed 0.591 vs size 0.641 (Δ -0.050) vs days 0.719. Still loses to size on the usual slice — not a leftover Q5.

| y | feature | n | n_pos | CV | sd |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | c_missed_salary | 5,987 | 441 | 0.475 | 0.033 |
| y2_neg_2of3 | c_salary_month | 5,987 | 441 | 0.475 | 0.033 |
| y2_neg_2of3 | log1p_a_in3 | 5,987 | 441 | 0.566 | 0.046 |
| y2_neg_2of3 | c_n_days_with_tx | 5,987 | 441 | 0.528 | 0.132 |
| y2_neg_2of3 | c_ss_month | 5,987 | 441 | 0.543 | 0.074 |
| y3_recover_cash_6m | c_missed_salary | 2,896 | 95 | 0.591 | 0.017 |
| y3_recover_cash_6m | c_salary_month | 2,896 | 95 | 0.591 | 0.017 |
| y3_recover_cash_6m | log1p_a_in3 | 2,896 | 95 | 0.641 | 0.041 |
| y3_recover_cash_6m | c_n_days_with_tx | 2,896 | 95 | 0.719 | 0.134 |
| y3_recover_cash_6m | c_ss_month | 2,896 | 95 | 0.609 | 0.057 |


## 22. Monthly-cadence skip

Monthly-cadence Y3 missed 0.511 vs size on monthly 0.757; irregular 0.591. A monthly skip would be the cleanest Q5; it is not.

| y | kind | n | n_pos | miss | CV |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | monthly | 5,853 | 459 | 1.9% | 0.490 |
| y3_recover_cash_6m | monthly | 2,473 | 55 | 1.8% | 0.511 |
| y2_neg_2of3 | irregular | 2,578 | 220 | 11.8% | 0.464 |
| y3_recover_cash_6m | irregular | 1,041 | 53 | 12.9% | 0.591 |
| y2_neg_2of3 | rare | 3,192 | 208 | 1.5% | 0.491 |
| y3_recover_cash_6m | rare | 926 | 85 | 2.2% | 0.523 |
| y2_neg_2of3 | never | 5,733 | 384 | 0.0% | 0.500 |
| y3_recover_cash_6m | never | 1,208 | 209 | 0.0% | 0.500 |


## 23. Exclude-current reconstruction (in-module; do not rewrite ops.py)

Exclude-current missed n=767 vs store 623 (+144 quirk months). Y3 CV 0.516. The include-current quirk does not hide a KEEP. Do not patch ops.py.

| y | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- |
| y2_neg_2of3 | 17,356 | 1,271 | 0.490 | 1 |
| y3_recover_cash_6m | 5,648 | 402 | 0.516 | 1 |


## 24. Last salary amount before a miss

Last salary |amt| before a miss p50=4,512 vs in-month salary p50=20,839. Share of misses after a ≤5€ last salary 1.1%. Usual payroll before a miss is mass, not a token.

## 25. P(Y6 | missed X) by cadence / trail

P(Y6|missed X)=47.7% vs Y6 base on overlap. A now-miss often precedes the rejected 3-month future miss. Still PARK — do not merge.

| slice | n_overlap | n_miss | P(Y6|miss) | Y6 base |
| --- | --- | --- | --- | --- |
| all_overlap | 6004 | 470 | 47.7% | 6.1% |
| monthly | 4867 | 112 | 8.0% | 0.5% |
| irregular | 1052 | 306 | 54.2% | 25.8% |
| short | 412 | 21 | 28.6% | 4.4% |
| long | 5592 | 449 | 48.6% | 6.2% |


## 26. Company-mean vs demean as Y3 X

Y3 raw 0.513 / company-mean 0.530 / demean 0.551. Demean rising toward chance-plus is not a shock KEEP — still loses to size.

| feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- |
| raw | 5,648 | 402 | 0.513 | 1 |
| co_mean | 5,648 | 402 | 0.530 | -1 |
| demean | 5,648 | 402 | 0.551 | 1 |


## 27. June peak residual (not August)

Y3 missed off-June 0.514; off-Aug 0.515. June is a small peak, not a tax-style calendar dummy.

| slice | n | n_pos | miss | CV |
| --- | --- | --- | --- | --- |
| June | 351 | 28 | 4.3% | LOW_POWER |
| not_June | 5,297 | 374 | 3.5% | 0.514 |
| Aug | 362 | 30 | 3.6% | LOW_POWER |
| not_Aug | 5,286 | 372 | 3.5% | 0.515 |


## 28. Payroll halt vs salary-token skip

Halt (missed, no SS) n=331; token-skip (missed, SS continues) n=292. Y3 halt-as-X 0.517; token-as-X 0.504 vs size 0.617. Neither split is a KEEP.

| y | slice | n_cm | n_lab | n_pos | rate | CV |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | halt_no_ss | 331 | 238 | 17 | 7.1% | LOW_POWER |
| y2_neg_2of3 | token_ss_continues | 292 | 227 | 23 | 10.1% | LOW_POWER |
| y2_neg_2of3 | not_miss | 20534 | 16891 | 1231 | 7.3% | 0.500 |
| y3_recover_cash_6m | halt_no_ss | 331 | 87 | 18 | 20.7% | LOW_POWER |
| y3_recover_cash_6m | token_ss_continues | 292 | 111 | 5 | 4.5% | LOW_POWER |
| y3_recover_cash_6m | not_miss | 20534 | 5450 | 379 | 7.0% | 0.500 |


## 29. First miss vs continuation streak

First miss n=358; continuation n=265. Y3 first-miss-as-X 0.498. Onset of a skip is not a turning X.

| y | slice | n_cm | n_lab | n_pos | rate | CV |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | first_miss | 358 | 275 | 23 | 8.4% | LOW_POWER |
| y2_neg_2of3 | cont_miss | 265 | 190 | 17 | 8.9% | LOW_POWER |
| y3_recover_cash_6m | first_miss | 358 | 122 | 10 | 8.2% | LOW_POWER |
| y3_recover_cash_6m | cont_miss | 265 | 76 | 13 | 17.1% | LOW_POWER |


## 30. Jaccard vs `not_salary` (the 15-col lever)

Jaccard(missed, not_salary)=0.049 (n_miss=623 / n_notsal=12,673); ρ=0.143. Most non-salary months are not “missed” (no usual history). `not_salary` Y3 0.671 is the quiet-month lever already on the card; missed is the rare usual-skip leftover and it is chance.

## 31. Size of usual-missers (rate gap vs size pile)

Usual-missed log1p(a_in3) p50 12.952 vs usual-paid 13.453. Missers are smaller — the Y3 rate gap is a size pile.

| slice | n_cm | log_in3_p50 | days_p50 | Y3 rate |
| --- | --- | --- | --- | --- |
| usual_missed | 623 | 12.952 | 17.0 | 11.6% |
| usual_paid | 6855 | 13.453 | 20.0 | 2.7% |
| not_usual | 13679 | 11.628 | 10.0 | 11.2% |


## 32. Holdout cadence coverage only

Holdout cadence coverage only: never 36, monthly 22, irregular 9, rare 5 / 72. No AUROC.

| kind | n_co | sal_p50 | miss_p50 |
| --- | --- | --- | --- |
| irregular | 9 | 0.571 | 0.000 |
| monthly | 22 | 1.000 | 0.000 |
| never | 36 | 0.000 | 0.000 |
| rare | 5 | 0.083 | 0.000 |


## What this cut did not do

- Did not rewrite `ops.py`, `tax_qa.*`, `uncat_qa.py`, `companies_qa.py`, `a_vol_qa.py`, `factoring_qa.py`.
- Did not run `python -m analysis.targets.build_targets` or rewrite parquet / duckdb.
- Did not touch `product/` or write a 0–100 / pillars.
- Did not invent `y_missed_salary` or merge with Y6.
- Did not put `c_missed_salary` on the 15-col card.
- Did not change the night Y3 quote 0.762 / 0.752.
- Did not redo the CLOSED tax calendar.
- Did not write the parent journal.

## What failed / next (held for wave note)

- X CLOSE: Y3 0.513 vs size 0.617 / days 0.711
- usual-only Y3 0.591 vs size 0.641 (Δ -0.050)
- exclude-current Y3 0.516 n=767; do not rewrite ops.py
- halt Y3 0.517 / first-miss 0.498; missers smaller=True

Elapsed 6s. Must-do 1–13 plus cadence, usual-skip, who, holdout, fold wander, SS packet, quiet residual, usual-vs-size, monthly skip, exclude-current quirk, last-amt, Y6 lead, co-mean, June residual, halt vs token, miss streak, not-salary Jaccard, misser size, holdout cadence.
