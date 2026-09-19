# Q5 invoice FX share (`e_fx_share`)

Generated `2026-09-19T03:32:14+02:00` by agent `97d3db33`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**, ever-ERP. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent a merged FX Y.

`e_fx_share` = this-period issued |amount| with `currency <> accounting_currency`. Y7 / Y5 are invoice-built — **never E as X**. Y3 cash-recover forbids B; Y2 forbids B. 470 dark: FX is NaN, not 0.

## Headline

Ever-FX **228** train ERP companies; fx>0 on 16.8% of ERP CM. Size ρ vs log1p(a_in3) 0.060 (not SIZE). Y3 single 0.528 vs size 0.628 vs days-full 0.711 (night 0.711). Y7 descriptive 30.5% vs 26.8%. Book **mixed import/export**. Style **shock**. Q5 **CLOSE**. 44-col **drop from the 44-col Y3 starter**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as a health Y. FX is a mixed import/export book, not a FICO label. |
| 2 | Who is improving? | Not this share. |
| 3 | Who is turning? | Y2 single is a check, not a claim. |
| 4 | Dip vs fall? | Y7/Y5 descriptive only. |
| 5 | Why did it change? | **CLOSE** — Y3 single 0.528 vs size 0.628 (Δ -0.100); not SIZE ρ_in3=0.060. CLOSE as a Q5 footnote (mixed import/export; foreign-home identity, not the 110), not a 44-col Y3 X. |
| 6 | Months earlier? | **CLOSE** lag1 0.535 vs now 0.528; acf1 -0.019 (shock). |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `e_fx_share` as a health Y | **PARK** | do not invent a merged FX Y |
| `e_fx_share` as Y7 / Y5 X | **CLOSE** | forbidden family E (labels are invoice-built) |
| `e_fx_share` as Y3 X (44-col list) | **CLOSE** | Y3 single 0.528 vs size 0.628 (Δ -0.100); not SIZE ρ_in3=0.060. CLOSE as a Q5 footnote (mixed import/export; foreign-home identity, not the 110), not a 44-col Y3 X. |
| `e_fx_share` as Y2 X | **CLOSE** | single 0.509 vs size 0.565; models already PARK |
| Q5 footnote (mixed import/export) | **KEEP footnote** | 97 always / 131 shock; HHI ρ=-0.133; not the 110 |
| Q6 `e_fx_share_lag1` | **CLOSE** | Y3 contemporaneous 0.528 vs lag1 0.535 (drop -0.007) vs lag3 0.523. Size on lag1 rows 0.627. Q6 **CLOSE** — hidden-test lead is 1-month only; issued_lag1 stays the Y7 KEEP. |
| 44-col keep list | **drop from the 44-col Y3 starter** | gate = beat size by ≥0.02 and not SIZE |

## 1. Coverage (train, ever-ERP)

Train ever-ERP 744 companies / 13,554 CM. `e_fx_share` defined 82.5% of ERP CM (issued this period); >0 on 16.8% of ERP CM (2,280 / 13,554) and 20.4% of defined. Ever-FX companies **228**. Dark 470 companies: defined FX months 0 (NaN-ok). FX invoice currencies ['AED', 'AOA', 'AUD', 'BRL', 'CAD', 'CHF', 'CLP', 'CNY', 'COP', 'CZK', 'DKK', 'EUR', 'GBP', 'HKD', 'HUF', 'IDR', 'ILS', 'INR', 'ISK', 'JPY', 'MAD', 'MXN', 'MYR', 'NAD', 'NOK', 'PEN', 'PHP', 'PLN', 'SAR', 'SEK', 'SGD', 'THB', 'TRY', 'USD', 'ZAR'] vs accounting ['AED', 'ARS', 'AUD', 'BRL', 'CAD', 'CLP', 'COP', 'DKK', 'EUR', 'GBP', 'INR', 'MXN', 'MYR', 'MZN', 'NAD', 'NZD', 'PEN', 'SGD', 'USD'].

| item | value |
| --- | ---: |
| train ERP companies / CM | 744 / 13,554 |
| confirm 744 / 470 | YES |
| `e_fx_share` defined | 82.5% (11,176) |
| `e_fx_share` > 0 / ERP CM | 16.8% (2,280) |
| `e_fx_share` > 0 / defined | 20.4% |
| ever-FX companies | **228** |
| dark companies / defined FX months | 470 / 0 |
| dark is all-NaN | YES |
| raw FX invoices (train) | 34,168 across 232 companies |
| AR / AP FX invoices | 10,022 / 24,146 |
| invoice currencies | AED, AOA, AUD, BRL, CAD, CHF, CLP, CNY, COP, CZK, DKK, EUR, GBP, HKD, HUF, IDR, ILS, INR, ISK, JPY, MAD, MXN, MYR, NAD, NOK, PEN, PHP, PLN, SAR, SEK, SGD, THB, TRY, USD, ZAR |
| accounting currencies | AED, ARS, AUD, BRL, CAD, CLP, COP, DKK, EUR, GBP, INR, MXN, MYR, MZN, NAD, NZD, PEN, SGD, USD |

FX pairs (train, |amount|):

| acct | invoice | n_inv | n_co | |amt| |
| --- | --- | --- | --- | --- |
| EUR | NOK | 421 | 7 | 674,035,236 |
| USD | COP | 25 | 2 | 455,182,650 |
| EUR | COP | 11 | 4 | 311,878,189 |
| EUR | USD | 13,499 | 135 | 228,334,489 |
| GBP | USD | 2,055 | 13 | 161,231,532 |
| EUR | CLP | 9 | 3 | 127,938,396 |
| COP | USD | 148 | 4 | 100,824,298 |
| EUR | IDR | 3 | 1 | 84,651,405 |
| BRL | EUR | 11 | 1 | 69,755,071 |
| SGD | USD | 145 | 1 | 65,066,598 |
| PEN | CLP | 16 | 1 | 63,214,170 |
| USD | EUR | 3,404 | 19 | 31,796,320 |


## 2. Size ρ

Defined ERP CM n=11,176. Spearman `e_fx_share` vs log1p(a_in3) **0.060**, vs e_ar_issued 0.122 (log1p 0.122). Among fx>0: vs in3 -0.375, vs issued -0.404. Ever-FX vs company-median log1p(a_in3) 0.177. not SIZE at |ρ|≥0.5.

| pair | Spearman | SIZE? |
| --- | ---: | --- |
| vs log1p(a_in3) | 0.060 |  |
| vs e_ar_issued | 0.122 |  |
| vs log1p(e_ar_issued) | 0.122 | |
| fx>0 only vs in3 | -0.375 | |
| fx>0 only vs issued | -0.404 | |
| ever-FX vs company med in3 | 0.177 | |

Gate: SIZE if |ρ| ≥ 0.5 vs log1p(a_in3) or e_ar_issued.

## 3. Quintiles and descriptive Y7 / Y5 (not as X)

Y3 stressed recover fx>0 7.9% vs fx=0 5.3%. Y2 7.9% vs 6.3%. Descriptive Y7 top1_lost 30.5% vs 26.8%; Y5 AP 7.1% vs 9.1%, AR 5.2% vs 7.9%. qcut on defined FX vs Y3 produced 1 bins (zeros pile — intensity quintiles among >0: 5 bins).

Binary fx>0 vs 0 vs NaN (no issue month). Y7 / Y5 rates only — family E is forbidden X.

| y | slice | n_cm | n_co | n_lab | n_pos | rate |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | fx>0 | 2,280 | 228 | 481 | 38 | 7.9% |
| y3_recover_cash_6m | fx=0 | 8,896 | 676 | 2,594 | 138 | 5.3% |
| y3_recover_cash_6m | fx NaN (no issue month) | 2,378 | 334 | 543 | 88 | 16.2% |
| y2_neg_2of3 | fx>0 | 2,280 | 228 | 1,867 | 147 | 7.9% |
| y2_neg_2of3 | fx=0 | 8,896 | 676 | 7,402 | 470 | 6.3% |
| y2_neg_2of3 | fx NaN (no issue month) | 2,378 | 334 | 2,006 | 98 | 4.9% |
| y7_top1_lost | fx>0 | 2,280 | 228 | 1,542 | 471 | 30.5% |
| y7_top1_lost | fx=0 | 8,896 | 676 | 5,758 | 1,545 | 26.8% |
| y7_top1_lost | fx NaN (no issue month) | 2,378 | 334 | 164 | 133 | 81.1% |
| y7_top1_lost_inflow | fx>0 | 2,280 | 228 | 1,360 | 96 | 7.1% |
| y7_top1_lost_inflow | fx=0 | 8,896 | 676 | 5,249 | 291 | 5.5% |
| y7_top1_lost_inflow | fx NaN (no issue month) | 2,378 | 334 | 130 | 33 | 25.4% |
| y5_ap_od30_ownp80 | fx>0 | 2,280 | 228 | 1,095 | 78 | 7.1% |
| y5_ap_od30_ownp80 | fx=0 | 8,896 | 676 | 3,739 | 340 | 9.1% |
| y5_ap_od30_ownp80 | fx NaN (no issue month) | 2,378 | 334 | 71 | 0 | 0.0% |
| y5_ar_od30_sust | fx>0 | 2,280 | 228 | 847 | 44 | 5.2% |
| y5_ar_od30_sust | fx=0 | 8,896 | 676 | 2,453 | 193 | 7.9% |
| y5_ar_od30_sust | fx NaN (no issue month) | 2,378 | 334 | 15 | 0 | 0.0% |


Y3 stressed, qcut of defined `e_fx_share` (zeros may collapse bins):

| q | interval | n | n_pos | rate | fx med |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.001, 1.0] | 3,075 | 176 | 5.7% | 0.000 |


Y3 intensity quintiles among fx>0:

| q | interval | n | n_pos | rate | fx med |
| --- | --- | --- | --- | --- | --- |
| 1 | (-0.000999256, 0.000904] | 97 | 2 | 2.1% | 0.000 |
| 2 | (0.000904, 0.015] | 96 | 5 | 5.2% | 0.005 |
| 3 | (0.015, 0.0832] | 96 | 4 | 4.2% | 0.039 |
| 4 | (0.0832, 0.486] | 96 | 14 | 14.6% | 0.194 |
| 5 | (0.486, 1.0] | 96 | 13 | 13.5% | 0.852 |


Plot: `fx_y3_quintiles.png`.

## 4. Single-feature train group-fold AUROC (Y3 / Y2 only)

Y3 stressed n=5,648 base 7.1%; Y2 n=17,356 base 7.3%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (full-panel replica 0.711). KEEP as Y3 X only if same-mask FX beats same-mask size by ≥0.02 and is not SIZE. Y7×E leak screen is expected to fail — we do not score that pair.

Y3 `e_fx_share` CV 0.528 vs same-mask size 0.628 (Δ -0.100) vs days 0.731. Full-panel days 0.711 (replica 0.711). Y2 FX 0.509 vs size 0.565. does not clear KEEP gate (need +0.02 vs size, not SIZE).

| y | feature | mask | n | n_pos | CV | sd | sign | train |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | e_fx_share | full | 9,269 | 617 | 0.509 | 0.028 | 1 | 0.515 |
| y2_neg_2of3 | fx_pos | full | 9,269 | 617 | 0.518 | 0.023 | 1 | 0.520 |
| y2_neg_2of3 | log1p_a_in3 | full | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 |
| y2_neg_2of3 | c_n_days_with_tx | full | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 |
| y2_neg_2of3 | e_ar_issued | full | 11,275 | 715 | 0.541 | 0.067 | 1 | 0.527 |
| y2_neg_2of3 | d_cust_hhi | full | 7,130 | 445 | 0.564 | 0.161 | -1 | 0.535 |
| y2_neg_2of3 | e_fx_share | fx_defined | 9,269 | 617 | 0.509 | 0.028 | 1 | 0.515 |
| y2_neg_2of3 | fx_pos | fx_defined | 9,269 | 617 | 0.518 | 0.023 | 1 | 0.520 |
| y2_neg_2of3 | log1p_a_in3 | fx_defined | 8,163 | 519 | 0.565 | 0.093 | 1 | 0.532 |
| y2_neg_2of3 | c_n_days_with_tx | fx_defined | 9,269 | 617 | 0.561 | 0.060 | 1 | 0.548 |
| y2_neg_2of3 | e_ar_issued | fx_defined | 9,269 | 617 | 0.516 | 0.077 | 1 | 0.514 |
| y2_neg_2of3 | d_cust_hhi | fx_defined | 6,807 | 435 | 0.561 | 0.159 | -1 | 0.536 |
| y3_recover_cash_6m | e_fx_share | full | 3,075 | 176 | 0.528 | 0.052 | 1 | 0.537 |
| y3_recover_cash_6m | fx_pos | full | 3,075 | 176 | 0.523 | 0.047 | 1 | 0.532 |
| y3_recover_cash_6m | log1p_a_in3 | full | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 |
| y3_recover_cash_6m | c_n_days_with_tx | full | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 |
| y3_recover_cash_6m | e_ar_issued | full | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.685 |
| y3_recover_cash_6m | d_cust_hhi | full | 2,485 | 141 | 0.595 | 0.101 | 1 | 0.585 |
| y3_recover_cash_6m | e_fx_share | fx_defined | 3,075 | 176 | 0.528 | 0.052 | 1 | 0.537 |
| y3_recover_cash_6m | fx_pos | fx_defined | 3,075 | 176 | 0.523 | 0.047 | 1 | 0.532 |
| y3_recover_cash_6m | log1p_a_in3 | fx_defined | 3,016 | 170 | 0.628 | 0.015 | -1 | 0.625 |
| y3_recover_cash_6m | c_n_days_with_tx | fx_defined | 3,075 | 176 | 0.731 | 0.061 | -1 | 0.733 |
| y3_recover_cash_6m | e_ar_issued | fx_defined | 3,075 | 176 | 0.667 | 0.061 | -1 | 0.662 |
| y3_recover_cash_6m | d_cust_hhi | fx_defined | 2,364 | 111 | 0.615 | 0.097 | 1 | 0.603 |


## 5. Persistence — style vs month shock

acf1=-0.019 acf3=-0.055 acf6=-0.097 (fx>0 indicator acf1=0.107; ever-FX share acf1=-0.019). Companies with ≥1 defined month: never=515 always(≥80% fx>0)=97 shock=131. Call: **shock**.

| kind | n_co | n_cm | mean_fx | Y3 | Y3_n | Y2 | Y7 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| never | 515 | 9428 | 0.000 | 6.8% | 2,636 | 5.7% | 25.9% |
| shock | 131 | 2362 | 0.069 | 8.2% | 644 | 8.2% | 36.9% |
| always | 97 | 1747 | 0.369 | 9.5% | 338 | 7.6% | 31.7% |


always = fx>0 in ≥80% of defined months. shock = some FX, not always. never = defined months all 0.

## 6. Leak vs issued / DSO (Y7 story)

FX vs e_ar_issued ρ=0.122, vs e_dso_proxy ρ=0.146, vs d_cust_hhi ρ=-0.133. not a leak of issued / DSO / HHI.

| pair | Spearman | leak |
| --- | --- | --- |
| e_fx_share × e_ar_issued | 0.122 |  |
| e_fx_share × e_dso_proxy | 0.146 |  |
| e_fx_share × e_ap_issued | 0.277 |  |
| e_fx_share × d_cust_hhi | -0.133 |  |
| e_fx_share × d_cust_top1 | -0.125 |  |
| e_fx_share × d_n_cust | 0.132 |  |
| e_fx_share × log_in3 | 0.060 |  |
| fx_pos × e_ar_issued | 0.148 |  |
| fx_pos × d_cust_hhi | -0.148 |  |


## 7. FX vs `d_cust_hhi` (foreign ≠ monopoly)

FX × d_cust_hhi ρ=-0.133 (among >0: 0.219). fx>0 share HHI-Q1 26.1% vs Q5 12.0%. foreign book ≠ customer monopoly.

| q | hhi_med | n | fx>0 | fx_med | Y3_fx>0 | Y3_fx=0 | n_Y3_fx |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.068 | 1,707 | 26.1% | 0.000 | 4.3% | 2.9% | 92 |
| 2 | 0.238 | 1,706 | 29.7% | 0.000 | 3.6% | 3.0% | 111 |
| 3 | 0.464 | 1,706 | 22.6% | 0.000 | 5.1% | 3.2% | 98 |
| 4 | 0.808 | 1,706 | 15.1% | 0.000 | 16.4% | 4.4% | 55 |
| 5 | 1.000 | 1,707 | 12.0% | 0.000 | 18.8% | 6.5% | 48 |


## 8. Mixed-group 110 — are FX companies those dark siblings?

Mixed-group invoiced fx>0 CM 17.9% (ever-FX n=68) vs all-invoiced groups 16.2% (ever-FX n=160). Dark mixed 110 / all-dark 360 have defined FX months 0 (must be 0). FX companies are **not** the 110 dark siblings (they are the invoiced side of mixed / all-invoiced groups).

| mix | n_co | n_erp | n_dark | fx_def | fx>0_erp | ever_FX | dark_fx_defined | Y3_erp | Y3_dark |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_invoiced | 514 | 514 | 0 | 84.8% | 16.2% | 160 | 0 | 8.9% | — |
| mixed | 340 | 230 | 110 | 78.3% | 17.9% | 68 | 0 | 5.0% | 12.1% |
| all_dark | 360 | 0 | 360 | — | — | 0 | 0 | — | 5.2% |


## 9. Holdout coverage only (no AUROC)

Holdout coverage only: ERP 40 ever-FX 17; dark 32 defined 0.0%. No AUROC.

| slice | n_co | n_cm | fx defined | fx>0 | ever_FX |
| --- | --- | --- | --- | --- | --- |
| holdout ERP | 40 | 582 | 89.7% | 35.6% | 17 |
| holdout dark | 32 | 491 | 0.0% | 0.0% | 0 |


## 10. AR vs AP FX (exporter vs importer)

FX |amount| is **44.1% AR / 55.9% AP**. Ever AR-FX 142 (only 18) / AP-FX 210 (only 86) / both 124. Book call: **mixed import/export**. Y3 AR-FX CV 0.522 AP-FX 0.529 (neither is a Y3 X). Store e_fx_share vs AR+AP FX amt ρ=0.992.

| item | value |
| --- | ---: |
| AR / AP FX \|amount\| | 44.1% / 55.9% |
| ever AR-FX / AP-FX / both | 142 / 210 / 124 |
| AR-only / AP-only | 18 / 86 |
| Y3 AR-FX / AP-FX CV | 0.522 / 0.529 |
| book call | mixed import/export |

| slice | n_cm | n_co | Y3 | Y3_n | Y2 |
| --- | --- | --- | --- | --- | --- |
| AR FX>0 | 1,102 | 142 | 6.3% | 223 | 6.0% |
| AP FX>0 | 1,989 | 210 | 8.4% | 403 | 8.1% |
| AR FX=0 (issued AR) | 7,481 | 635 | 3.6% | 2,195 | 6.7% |
| AP FX=0 (issued AP) | 8,840 | 685 | 5.4% | 2,612 | 6.4% |


## 11. Q6 — lag1 of `e_fx_share` (1-month only)

Y3 contemporaneous 0.528 vs lag1 0.535 (drop -0.007) vs lag3 0.523. Size on lag1 rows 0.627. Q6 **CLOSE** — hidden-test lead is 1-month only; issued_lag1 stays the Y7 KEEP.

| feature | n | n_pos | CV | sd |
| --- | --- | --- | --- | --- |
| e_fx_share | 3,075 | 176 | 0.528 | 0.052 |
| e_fx_share_lag1 | 3,055 | 180 | 0.535 | 0.055 |
| e_fx_share_lag3 | 2,723 | 158 | 0.523 | 0.050 |
| log1p_a_in3 on lag1 rows | 3,001 | 175 | 0.627 | 0.031 |


## 12. Y3 residual inside size terciles

Y3 fx>0 minus fx=0 inside size terciles: mean gap 2.5%. a residual remains (still not a single that beats size).

| tercile | in3_med | n_lab | fx>0 | Y3_fx>0 | Y3_fx=0 | n_pos_fx | gap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 10.946 | 1,006 | 13.3% | 11.2% | 8.8% | 15 | 2.4% |
| 2 | 13.227 | 1,004 | 11.8% | 5.9% | 3.4% | 7 | 2.5% |
| 3 | 14.802 | 1,006 | 21.4% | 6.0% | 3.5% | 13 | 2.5% |


## 13. HHI × FX Y3 cells (Q4/Q5 spike)

HHI Q4+Q5 × fx>0 Y3 positives = 18 (thin — do not KEEP a monopoly×FX interaction). Foreign book is more common on the diversified side (pass 7).

| q | hhi_med | n_lab | n_fx | n_pos_fx | Y3_fx | n_z | n_pos_z | Y3_z |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.087 | 473 | 98 | 4 | 4.1% | 375 | 11 | 2.9% |
| 2 | 0.250 | 473 | 111 | 6 | 5.4% | 362 | 10 | 2.8% |
| 3 | 0.459 | 472 | 92 | 3 | 3.3% | 380 | 13 | 3.4% |
| 4 | 0.789 | 473 | 55 | 9 | 16.4% | 418 | 18 | 4.3% |
| 5 | 1.000 | 473 | 48 | 9 | 18.8% | 425 | 28 | 6.6% |


## 14. Within shock companies — FX month vs own zero month

Shock companies (n=131): own FX-month Y3 5.6% vs own zero-month 6.8%. company identity, not the FX month.

| y | FX month | n_lab_fx | n_pos_fx | own 0 | n_lab_0 | n_pos_0 |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 5.6% | 179 | 10 | 6.8% | 396 | 27 |
| y2_neg_2of3 | 8.9% | 529 | 47 | 7.9% | 1,171 | 92 |
| y7_top1_lost | 29.3% | 447 | 131 | 38.7% | 878 | 340 |


## 15. Feature-report confirm

Full-train (incl. dark NaN) cov_cm 52.8% cov_co 61.2% modal-zero-among-defined 79.6%. CONFIRM feature-report 52.8% / 79.6% modal. Raw FX invoice companies 232 vs store ever-FX 228 (Δ 4 — likely share rounded to 0 or period-edge).

## 16. No-issue month vs FX

ERP no-issue Y3 16.2% (n_lab=543) vs issued 5.7%. Issued-this-month flag Y3 CV 0.607 sign=-1 (has_book vs dark 0.460). The 16% hole is the bigger Q5 footnote — FX is not that hole.

## 17. Major vs exotic invoice currency

Among raw-FX months, major (EUR/USD/GBP ≥50% of FX |amt|) companies 211, exotic-majority 62. Y3 rates in the table. Not a Y.

| slice | n_cm | n_co | n_lab | n_pos | Y3 |
| --- | --- | --- | --- | --- | --- |
| major-FX month (EUR/USD/GBP ≥50%) | 2,003 | 211 | 451 | 34 | 7.5% |
| exotic-FX month | 277 | 62 | 30 | 4 | 13.3% |
| no raw FX | 11,274 | 686 | 3,137 | 226 | 7.2% |


Top FX invoice currencies by |amount| (train):

| ccy | |amt| |
| --- | --- |
| COP | 767,544,839 |
| NOK | 674,038,161 |
| USD | 587,010,532 |
| CLP | 210,802,232 |
| EUR | 147,613,399 |
| IDR | 84,651,405 |
| MXN | 28,548,366 |
| AOA | 25,250,000 |


## 18. Raw 232 vs store 228 + home accounting currency

Raw-only FX companies 4 ['COMP_0510', 'COMP_0708', 'COMP_1159', 'COMP_1194']; store-only 0. Home-EUR n=658 fx>0 11.3%; home-USD n=23 fx>0 52.8%. Not a Y. The 4-name gap does not move the 228.

| company_id | n_inv | fx_share |
| --- | --- | --- |
| COMP_1159 | 1518 | 1.2% |
| COMP_0510 | 42 | 15.6% |
| COMP_1194 | 1007 | 1.3% |
| COMP_0708 | 1425 | 0.0% |


| home | n_co | fx>0 | ever_FX | Y3 | n_lab |
| --- | --- | --- | --- | --- | --- |
| home EUR | 658 | 11.3% | 151 | 7.2% | 3,429 |
| home USD | 23 | 52.8% | 20 | 10.1% | 69 |
| home other | 63 | 67.4% | 57 | 9.2% | 120 |


## 19. Four raw-FX names that never light the store

Four raw-FX names that never have store e_fx_share>0: ['COMP_0510', 'COMP_0708', 'COMP_1159', 'COMP_1194']. FX issuance months outside the 24-month panel: 0. Live invoices.build lights them (COMP_0510 2024-09 share=1). Store is left-truncated / stale. Leave the 228 quote; do not rewrite parquet.

| company_id | n_cm | defined | max_store | n_pos | has_book |
| --- | --- | --- | --- | --- | --- |
| COMP_0510 | 21 | 20 | 0.000000 | 0 | True |
| COMP_0708 | 9 | 9 | 0.000000 | 0 | True |
| COMP_1159 | 14 | 7 | 0.000000 | 0 | True |
| COMP_1194 | 12 | 11 | 0.000000 | 0 | True |


## 20. Home-EUR only singles (majority book)

Home-EUR Y3 FX 0.527 vs size 0.640. Non-EUR home FX —. The 0.528 pooled single is not a EUR-SME recover why.

| pop | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| home_EUR | e_fx_share | 2,902 | 159 | 0.527 | 1 |
| home_EUR | log1p_a_in3 | 2,856 | 155 | 0.640 | -1 |
| home_EUR | c_n_days_with_tx | 2,902 | 159 | 0.747 | -1 |
| home_not_EUR | e_fx_share | 173 | 17 | LOW_POWER | — |
| home_not_EUR | log1p_a_in3 | 160 | 15 | LOW_POWER | — |
| home_not_EUR | c_n_days_with_tx | 173 | 17 | LOW_POWER | — |


## 21. High-intensity flag (share ≥ 0.08)

High-intensity (e_fx_share≥0.08) Y3 13.9% (n_lab=194, n_pos=27) vs low-positive 3.8%. Flag CV 0.539 vs same-mask size 0.628 (Δ -0.089). still CLOSE as Y3 X.

## 22. FX calendar

fx>0 / ERP CM by calendar month: peak 17.9% trough 15.5%. flat — not a tax-style dummy.

| month | n_cm | fx>0 / ERP | fx>0 / defined | Y3_fx>0 |
| --- | --- | --- | --- | --- |
| Jan | 1,155 | 15.5% | 18.9% | 9.7% |
| Feb | 1,214 | 16.0% | 19.4% | 11.3% |
| Mar | 1,231 | 16.7% | 20.0% | 0.0% |
| Apr | 1,242 | 16.7% | 20.1% | 0.0% |
| May | 1,251 | 17.9% | 21.7% | 0.0% |
| Jun | 1,258 | 17.4% | 20.8% | 7.1% |
| Jul | 1,265 | 17.8% | 21.4% | 5.7% |
| Aug | 1,277 | 16.7% | 20.7% | 3.8% |
| Sep | 842 | 16.2% | 19.7% | 6.9% |
| Oct | 899 | 16.8% | 20.5% | 7.5% |
| Nov | 926 | 16.6% | 20.6% | 10.7% |
| Dec | 994 | 17.2% | 20.7% | 11.4% |


## 23. Store vs live `invoices.build` (no parquet write)

On the store panel, live matches store: ever-FX **228** / 228, disagree 0 (max |Δ|=0.000000). Full 24-month grid live ever-FX **232**; 215 live-FX months sit off the monthly panel (invoice before first cash-trail month; 4 names never light the store). Live-on-panel Y3 FX CV 0.528 vs size 0.628. Do not rewrite parquet.

| item | value |
| --- | ---: |
| store ever-FX | 228 |
| live ever-FX (on panel) | 228 |
| live ever-FX (24m grid) | 232 |
| live-FX months off panel | 215 |
| both-defined months | 11,176 |
| disagree (\|Δ\| > 0) | 0 |
| max \|Δ\| | 0.000000 |
| store NaN / live defined | 0 |
| live Y3 FX CV | 0.528 |
| live same-mask size | 0.628 |

Do **not** rewrite parquet tonight. Do **not** patch `invoices.py` — live build already has the 4 names. The 228 quote is the store.

## 24. Always-FX × home currency

Always-FX 97: EUR 52 / USD 11 / other 34 (non-EUR 46.4%). Style is a foreign-home identity more than a month shock.

| kind | n | EUR | USD | other |
| --- | --- | --- | --- | --- |
| never | 515 | 506 | 3 | 6 |
| shock | 131 | 99 | 9 | 23 |
| always | 97 | 52 | 11 | 34 |


## 25. FX / invoice before first panel month

Train ERP 744: first invoice before first panel month 324; first FX invoice before panel 53 / 232 ever-raw-FX. Names ['COMP_0014', 'COMP_0057', 'COMP_0104', 'COMP_0110', 'COMP_0168', 'COMP_0185', 'COMP_0238', 'COMP_0247', 'COMP_0270', 'COMP_0275', 'COMP_0280', 'COMP_0283', 'COMP_0311', 'COMP_0323', 'COMP_0339', 'COMP_0350', 'COMP_0352', 'COMP_0365', 'COMP_0404', 'COMP_0406', 'COMP_0469', 'COMP_0472', 'COMP_0477', 'COMP_0488', 'COMP_0510', 'COMP_0523', 'COMP_0559', 'COMP_0566', 'COMP_0576', 'COMP_0580', 'COMP_0643', 'COMP_0666', 'COMP_0708', 'COMP_0783', 'COMP_0790', 'COMP_0799', 'COMP_0836', 'COMP_0854', 'COMP_0912', 'COMP_0919', 'COMP_0921', 'COMP_0956', 'COMP_0959', 'COMP_0976', 'COMP_1085', 'COMP_1108', 'COMP_1158', 'COMP_1159', 'COMP_1161', 'COMP_1163', 'COMP_1190', 'COMP_1194', 'COMP_1225']. Same connection clock as trail_length — not a store formula bug.

## 26. Ever-FX company dummy (BETWEEN)

Ever-FX company dummy (constant) Y3 CV 0.527 vs size 0.623 vs days 0.730 (Δ vs size -0.096). BETWEEN identity is not a recover X.

## 27. Holdout home mix (coverage only)

Holdout ERP home-EUR 30 fx>0 17.8%; non-EUR 10 fx>0 85.1%. Holdout ERP fx>0 35.6% vs train 16.8% — coverage only, not a claim.

## 28. FX vs overdue / credit notes

FX vs AR od30 ρ=0.046, vs credit-note ρ=0.175. Not a late-payer rewrite and not a Y5 X.

| pair | Spearman |
| --- | --- |
| e_fx_share × e_ar_overdue_30 | 0.046 |
| e_fx_share × e_credit_note_ratio | 0.175 |


## 29. Published Y3 SHAP (no refit)

Published Y3 SHAP: `e_fx_share` rank ~69, mean|SHAP|=0.0573. Already a quiet column on the 44 — dropping it is consistent with the single.

## 30. Pre-trail FX companies on the panel

Pre-trail FX companies on-panel Y3 2.3% vs other ever-FX 9.1%. Do not invent a trail Y from this.

| slice | n_co | n_lab | Y3 | fx>0 |
| --- | --- | --- | --- | --- |
| pre-trail FX | 53 | 88 | 2.3% | 71.8% |
| on-panel-only ever-FX | 179 | 908 | 9.1% | 51.8% |
| ERP never-FX | 512 | 2,622 | 6.8% | 0.0% |


## 31. Amount handful vs name-count

Amount table is a handful: EUR-NOK n_co=7 |amt|=674,035,236; EUR-USD n_co=135 (many names, smaller |amt|). Do not treat COP/NOK as the 228.

| acct | inv | n_co | |amt| |
| --- | --- | --- | --- |
| EUR | NOK | 7 | 674,035,236 |
| USD | COP | 2 | 455,182,650 |
| EUR | COP | 4 | 311,878,189 |
| EUR | USD | 135 | 228,334,489 |
| GBP | USD | 13 | 161,231,532 |
| EUR | CLP | 3 | 127,938,396 |


## 32. Months-on-panel vs ever-FX

Ever-FX vs months-on-panel ρ=0.021. ≤12 months on book: ever-FX 36.0% (n=211); ≥20 months: 30.7% (n=440). Late joiners are not the FX pocket.

## 33. Pre-trail 2.3% — size or trail?

Pre-trail FX median log1p(a_in3) 13.041 vs other ever-FX 13.039; Y3 2.3% vs 9.1%. The 2.3% is a quieter stressed set, not a recover engine.

| slice | n_co | med_log_in3 | med_months | Y3 |
| --- | --- | --- | --- | --- |
| pre-trail FX | 53 | 13.041 | 9.0 | 2.3% |
| other ever-FX | 179 | 13.039 | 24.0 | 9.1% |
| ERP never-FX | 516 | 12.193 | 20.0 | 6.8% |


## 34. Y5 by style (never as X)

Y5 AP always 5.5% vs never 9.0%; AR always 4.5% vs never 7.8%. Y5×E leak screen fails as required (["forbidden prefix 'e_': ['e_fx_share']"]). Descriptive only.

| kind | n_co | y5_ap_od30_ownp80 | y5_ap_od30_ownp80_n | y5_ar_od30_sust | y5_ar_od30_sust_n |
| --- | --- | --- | --- | --- | --- |
| never | 515 | 9.0% | 3,200 | 7.8% | 2,077 |
| shock | 131 | 9.6% | 920 | 7.6% | 643 |
| always | 97 | 5.5% | 785 | 4.5% | 595 |


## 35. Y3 residual inside activity terciles

Y3 fx>0 minus fx=0 inside `c_n_days_with_tx` terciles: mean gap 3.3%. a small residual remains; days single is still 0.711.

| days_T | days_med | n_lab | Y3_fx>0 | Y3_fx=0 | n_pos_fx | gap |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8.0 | 1,063 | 13.3% | 11.0% | 18 | 2.3% |
| 2 | 18.0 | 1,142 | 6.3% | 3.0% | 12 | 3.3% |
| 3 | 25.0 | 870 | 5.1% | 1.0% | 8 | 4.1% |


## 36. High-activity tercile FX single

High-activity tercile only: Y3 FX — (LOW_POWER) vs size — vs days — (n_pos=21). Still not a Y3 X.

## What failed / next

- Y3 FX 0.528 loses to size 0.628 by 0.100; live rebuild 0.528 still loses. drop from the 44. Book is mixed import/export.

Elapsed 4s. Cuts: coverage, size, quintiles+Y7/Y5, Y3/Y2 singles, persistence, leak, HHI, mixed-110, holdout, AR/AP, Q6 lag1, size residual, HHI cells, within-shock, feature-report confirm, no-issue hole, major/exotic, raw-gap + home ccy, four-name gap, home-EUR singles, high-intensity, calendar, live-vs-store, always×home, pre-trail FX, ever-dummy, holdout home, book quality, SHAP, pre-trail Y3, handful, trail length, pre-trail size, Y5 style, days residual, high-activity single.
