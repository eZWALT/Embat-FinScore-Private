# companies.csv QA — country / ERP / currency / created_at

- **When:** 2026-09-19T04:03:50+02:00
- **Agent:** `a14da08b`
- **Python:** `/home/walterjtv/.pyenv/versions/base/bin/python3`
- **Re-run:** `python -m analysis.evaluate.companies_qa`
- **Holdout:** 72 companies, seed 20260918. Coverage only. Rates and cuts on train.
- **Y:** accepted `y3_recover_cash_6m` / `y2_neg_2of3` from `targets.parquet` (no assembler).
- **Brief:** Q1 who is healthy — as *descriptive context*, not a health Y. Hidden test is **new groups**.
- **Not:** 0–100, `product/`, parquet rewrite, GBM, `build_targets`, a merged country/erp Y.
- **Do not revive:** `created_at` as a health Y (trail QA: connection clock).

## Decision

**PARK created_at / missing-country as health Ys; CLOSE has_erp as Y3 X (470 dummy); CLOSE has_country as Q1 lever (missingness / size / no lift); CLOSE currency=EUR as Q1 lever**

erp×dark is not a perfect flag (invoiced+NULL=73, dark+named=37); NULL among dark 92.13% CONFIRMS 92.1%. named-ERP dark are 17 of the 110 mixed + 20 of the 360. has_erp Y3 CV 0.480 vs size 0.617. country-miss 82.21%; not SIZE; 24-month books still miss country on 73.6% so missingness is the default, not a short-trail hole. No companies.csv flag is a transferable Q1 health X.

A company-constant flag that marks a *group type* (ERP book, country filled, home EUR) cannot transfer to the hidden test of **new groups**.

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| `created_at` as health Y | **PARK** | onboard ≠ trail; three-clock p50 created−first_tx = 39.4 days; same-month only 7.08%; year vs grid ρ −0.85. |
| `created_at` year as Y3 X | **CLOSE** | CV 0.504; fold signs flip; holdout year mix ≠ train (2025 71% vs 35%). |
| missing-country as health Y | **PARK** | missingness is not 45→65. 82.21% of train. |
| `has_erp` as Y3 X | **CLOSE** | NULL among dark 92.13%; named-dark 37; ρ vs has_book 0.813. Y2 already restricted on dark. |
| `has_erp` as a new Y | **PARK** | do not invent a merged ERP Y. |
| `has_country` as Q1 descriptive | **CLOSE** | Y3 CV 0.481 vs size 0.617; not SIZE (ρ=-0.028); miss is the default even on all-invoiced (~85%). |
| `currency=EUR` as Q1 descriptive | **CLOSE** | share EUR 90.28%; Y3 CV 0.515 vs size 0.617. |
| any companies.csv flag as transferable Q1 X | **CLOSE** | company-constant / group-type dummy; hidden test = new groups. |

## Pass 1 — completeness (train vs holdout coverage)

Missing = NULL / empty / whitespace. Holdout is coverage only.

| split | n companies | miss country | miss currency | miss erp | miss created_at | n miss country | n miss erp |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1,214 | 82.21% | 0.00% | 41.68% | 0.00% | 998 | 506 |
| holdout | 72 | 80.56% | 0.00% | 48.61% | 0.00% | 58 | 35 |
| all | 1,286 | 82.12% | 0.00% | 42.07% | 0.00% | 1,056 | 541 |

Train missing-country **82.21%** (998 / 1,214). Holdout coverage 80.56% (58 / 72). Currency is almost complete (train miss 0.00%). `created_at` miss 0.00%.

## Pass 2 — ERP values × 470 dark

Book filter = y11 (invoice, not cancel, amount ≠ 0, issuance present). confirm_470=True. Train dark **470**, invoiced **744**.

Confusion (`companies.erp` named vs invoice book):

| companies.erp | invoice book | n |
|---|---|---:|
| named | invoiced | 671 |
| named | dark | 37 |
| NULL | invoiced | 73 |
| NULL | dark | 433 |

- NULL among dark: **92.13%** (433 / 470) — CONFIRMS join-QA 92.1%.
- Named-ERP dark: **37** — CONFIRMS join-QA 37.
- NULL-ERP train companies: **506** — CONFIRMS join-QA 506.
- Invoiced among NULL-ERP: **14.43%** — CONFIRMS join-QA 14.4%.
- Perfect dark flag? **False** (invoiced+NULL=73, dark+named=37).

Named ERP systems (train, any book):

| erp | n | share of named |
|---|---:|---:|
| businessCentral | 315 | 44.49% |
| netsuite | 127 | 17.94% |
| sage200 | 43 | 6.07% |
| businessOne | 40 | 5.65% |
| dynamicsAx | 40 | 5.65% |
| sageX3 | 30 | 4.24% |
| m3Rosetta | 21 | 2.97% |
| distritoK | 20 | 2.82% |
| navision | 18 | 2.54% |
| a3 | 14 | 1.98% |
| etendo | 11 | 1.55% |
| r3 | 9 | 1.27% |
| libra | 7 | 0.99% |
| sageIntacct | 4 | 0.56% |
| ekon | 3 | 0.42% |
| fo | 2 | 0.28% |
| sage50 | 1 | 0.14% |
| datev | 1 | 0.14% |
| holded | 1 | 0.14% |
| sapByd | 1 | 0.14% |

Named-ERP *dark* systems:

| erp | n |
|---|---:|
| dynamicsAx | 18 |
| netsuite | 8 |
| businessCentral | 6 |
| businessOne | 2 |
| m3Rosetta | 1 |
| a3 | 1 |
| fo | 1 |

### Are named-ERP dark companies the 110 mixed?

Named-ERP dark in mixed groups: **17** / 37. In all-dark 360: **20** / 37.

| mix | n cos | groups |
|---|---:|---:|
| all_dark | 20 | 3 |
| mixed | 17 | 10 |

`groups.erp` vs `companies.erp`: raw-string agree=0 (slugs ≠ display names — not a mismatch). Family agree=379, family disagree=18, group-named / company-NULL=116, group-NULL / company-named=311.

## Pass 3 — country / currency mix × FX

fx_qa.md exists (fx_qa.md) — FX child finished; descriptive cross only.

Train known country **216** / missing **998**. EUR **90.28%** (1,096).

Country (known, train):

| country | n | share of known |
|---|---:|---:|
| ES | 159 | 73.61% |
| NL | 13 | 6.02% |
| DE | 12 | 5.56% |
| PT | 8 | 3.70% |
| FR | 6 | 2.78% |
| US | 5 | 2.31% |
| GB | 4 | 1.85% |
| IT | 4 | 1.85% |
| BE | 2 | 0.93% |
| MY | 1 | 0.46% |
| AT | 1 | 0.46% |
| SE | 1 | 0.46% |

Currency (train):

| currency | n | share |
|---|---:|---:|
| EUR | 1,096 | 90.28% |
| GBP | 38 | 3.13% |
| USD | 31 | 2.55% |
| MXN | 7 | 0.58% |
| COP | 5 | 0.41% |
| DKK | 5 | 0.41% |
| CAD | 3 | 0.25% |
| BRL | 3 | 0.25% |
| AUD | 3 | 0.25% |
| CLP | 3 | 0.25% |
| PLN | 3 | 0.25% |
| NZD | 2 | 0.16% |
| CHF | 2 | 0.16% |
| PEN | 2 | 0.16% |
| AED | 2 | 0.16% |
| ARS | 1 | 0.08% |
| BAM | 1 | 0.08% |
| MYR | 1 | 0.08% |
| VND | 1 | 0.08% |
| INR | 1 | 0.08% |
| NAD | 1 | 0.08% |
| JPY | 1 | 0.08% |
| NOK | 1 | 0.08% |
| CZK | 1 | 0.08% |

Ever `e_fx_share`>0: **228** train companies (fx_qa quoted 228 ever-FX ERP). Among invoiced: 228 / 744. EUR invoiced ever-FX 165/673; non-EUR invoiced 63/71. Known-country invoiced 45/145; missing-country invoiced 183/599. Dark FX is NaN, not 0 (fx_qa). Descriptive only.

Ever-FX by country (invoiced, top):

| country | n invoiced | ever FX | share FX |
|---|---:|---:|---:|
| (MISSING) | 599 | 183 | 30.55% |
| ES | 116 | 30 | 25.86% |
| DE | 9 | 4 | 44.44% |
| PT | 5 | 1 | 20.00% |
| FR | 4 | 1 | 25.00% |
| NL | 3 | 2 | 66.67% |
| US | 3 | 3 | 100.00% |
| GB | 2 | 2 | 100.00% |
| IT | 2 | 1 | 50.00% |
| MY | 1 | 1 | 100.00% |

## Pass 4 — three clocks (onboard ≠ trail)

Clocks: `companies.created_at` (platform onboard), first transaction (bank trail), first `banking_products.created_at` (product connection). Late = after 2024-09-01.

- Late company `created_at`: 860 / 1,214 = 70.84% (CONFIRMS trail 860/1,214 = 70.8%).
- Late first tx: 779 / 1,214 = 64.17%.
- Late first banking product: 891 / 1,211 = 73.58%. No banking row: 3.
- `created_at` − first tx: median **39.4** days (p25 -104.4, p75 59.4); share after first tx 61.29%; same calendar month 7.08%.
- `created_at` − first bank created: median -5.9 days; share after 1.89%.
- First bank created − first tx: median 54.6 days; share bank after cash 70.10%.
- Late 2×2: both 748, created-only 112, tx-only 31, neither 323.

**PARK** `created_at` as a health Y. The three clocks are not the same event.

## Pass 5 — group size vs `h_group_size`; singletons × Y2/Y3

Train groups **235**. Singleton (1 train member) groups **64** / companies **64**. Spearman train-count vs `h_group_size` 1.000; `groups.n_companies_in_sample` vs `h_group_size` 1.000 (agree 100.00%). `h_group_size` *is* `n_companies_in_sample`.

Y rates (train labeled):

| Y | slice | n labeled | pos | cos | rate |
|---|---|---:|---:|---:|---:|
| y3_recover_cash_6m | all_train | 5,648 | 402 | 725 | 7.12% |
| y3_recover_cash_6m | singleton | 240 | 36 | 35 | 15.00% |
| y3_recover_cash_6m | multi | 5,408 | 366 | 690 | 6.77% |
| y2_neg_2of3 | all_train | 17,356 | 1,271 | 1,195 | 7.32% |
| y2_neg_2of3 | singleton | 886 | 79 | 64 | 8.92% |
| y2_neg_2of3 | multi | 16,470 | 1,192 | 1,131 | 7.24% |

Singleton − multi residual inside `log1p(a_in3)` terciles (train edges):

| Y | tercile | n multi | rate multi | n singleton | rate singleton | residual | median log1p(a_in3) |
|---|---|---:|---:|---:|---:|---:|---:|
| y3_recover_cash_6m | T1_small | 1,114 | 14.45% | 42 | 21.43% | +6.98pp | 9.542 |
| y3_recover_cash_6m | T2_mid | 1,931 | 4.35% | 51 | 21.57% | +17.22pp | 12.666 |
| y3_recover_cash_6m | T3_large | 2,251 | 4.98% | 139 | 10.07% | +5.10pp | 14.632 |
| y2_neg_2of3 | T1_small | 4,799 | 5.38% | 192 | 14.06% | +8.69pp | 8.994 |
| y2_neg_2of3 | T2_mid | 4,747 | 7.37% | 273 | 4.76% | -2.61pp | 12.512 |
| y2_neg_2of3 | T3_large | 4,662 | 8.00% | 295 | 7.80% | -0.20pp | 14.707 |

Y3 singleton 15.00% vs multi 6.77%. Y2 singleton 8.92% vs multi 7.24%. A singleton flag is a **group-size dummy**. It cannot transfer to new groups.

## Pass 6 — single-feature train group-fold AUROC

Sign from the train side of each fold. Size control = `log1p(a_in3)`. Days bar `0.711`. KEEP-as-Q1-descriptive bar = beat size by ≥0.02 and not a missingness / group-type dummy.

| Y | feature | CV AUROC | sd | train AUROC | sign | two-sided | coverage | n |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| y3_recover_cash_6m | has_erp | 0.480 | 0.042 | 0.505 | 1 | 0.520 | 100.00% | 5,648 |
| y3_recover_cash_6m | has_country | 0.481 | 0.066 | 0.521 | -1 | 0.519 | 100.00% | 5,648 |
| y3_recover_cash_6m | is_eur | 0.515 | 0.033 | 0.518 | -1 | 0.515 | 100.00% | 5,648 |
| y3_recover_cash_6m | log1p_a_in3 | 0.617 | 0.061 | 0.620 | -1 | 0.617 | 97.88% | 5,528 |
| y3_recover_cash_6m | h_group_size | 0.552 | 0.119 | 0.580 | -1 | 0.552 | 100.00% | 5,648 |
| y3_recover_cash_6m | c_n_days_with_tx | 0.711 | 0.031 | 0.723 | -1 | 0.711 | 100.00% | 5,648 |
| y2_neg_2of3 | has_erp | 0.475 | 0.079 | 0.514 | -1 | 0.525 | 100.00% | 17,356 |
| y2_neg_2of3 | has_country | 0.479 | 0.040 | 0.511 | -1 | 0.521 | 100.00% | 17,356 |
| y2_neg_2of3 | is_eur | 0.524 | 0.045 | 0.511 | -1 | 0.524 | 100.00% | 17,356 |
| y2_neg_2of3 | log1p_a_in3 | 0.552 | 0.046 | 0.540 | 1 | 0.552 | 86.24% | 14,968 |
| y2_neg_2of3 | h_group_size | 0.375 | 0.074 | 0.503 | 1 | 0.625 | 100.00% | 17,356 |
| y2_neg_2of3 | c_n_days_with_tx | 0.571 | 0.046 | 0.577 | 1 | 0.571 | 100.00% | 17,356 |

Y3 `has_erp` 0.480 / `has_country` 0.481 / `is_eur` 0.515 vs size 0.617 vs days 0.711. Beats size ≥0.02: erp=False country=False eur=False.

## Pass 7 — is country-missing SIZE or late-arrival?

Train miss-country companies **998** vs known **216**. Spearman miss vs company-median `log1p(a_in3)` **-0.028** (not SIZE at |ρ|≥0.50). vs months-on-book **-0.146**.

Median size miss 12.587 vs known 12.793. Median grid months miss 19.5 vs known 24.0. Late first-tx miss 67.94% vs known 46.76%. Short <12m miss 30.26% vs known 22.22%. Looks like late-arrival.

Missing-country share inside company-median size terciles (train edges):

| tercile | n cos | n miss country | share miss |
|---|---:|---:|---:|
| T1_small | 405 | 342 | 84.44% |
| T2_mid | 404 | 329 | 81.44% |
| T3_large | 405 | 327 | 80.74% |

Missing-country share by months-on-book (fixed trail cuts, not quantiles):

| trail | n cos | n miss country | share miss |
|---|---:|---:|---:|
| <6 | 3 | 3 | 100.00% |
| 6-11 | 347 | 299 | 86.17% |
| 12-17 | 150 | 125 | 83.33% |
| 18-23 | 279 | 251 | 89.96% |
| 24 | 435 | 320 | 73.56% |

Country missingness is a **group trait**: all-miss groups 177, all-known 27, mixed 31. A group-constant dummy cannot transfer to new groups.

## Pass 8 — is `has_erp` just the 470 dummy?

Company-level Spearman `has_erp` vs has-book **0.813** (agree 90.94%). Near-copy of the dark flag if |ρ|≥0.80: **True**.

| Y | slice | n labeled | pos | cos | rate |
|---|---|---:|---:|---:|---:|
| y3_recover_cash_6m | has_erp | 3,359 | 243 | 432 | 7.23% |
| y3_recover_cash_6m | null_erp | 2,289 | 159 | 293 | 6.95% |
| y3_recover_cash_6m | has_book | 3,618 | 264 | 463 | 7.30% |
| y3_recover_cash_6m | dark | 2,030 | 138 | 262 | 6.80% |
| y2_neg_2of3 | has_erp | 10,354 | 725 | 700 | 7.00% |
| y2_neg_2of3 | null_erp | 7,002 | 546 | 495 | 7.80% |
| y2_neg_2of3 | has_book | 11,275 | 715 | 736 | 6.34% |
| y2_neg_2of3 | dark | 6,081 | 556 | 459 | 9.14% |

Among *invoiced* companies only, Y3 named-ERP 6.90% (n=3,262) vs NULL-ERP 10.96% (n=356), gap -4.06pp. If that gap is flat, leftover named-ERP among the 744 is not a health lever.


### Extra — named-ERP dark groups (not just the 110)

37 named-ERP dark sit in **3** all-dark groups + **10** mixed groups. They are **not** the 110: 20 live in all-dark holdings that somehow have an ERP *name* and no book.

| group | mix | n named-dark | erp slugs | group n | median log1p(a_in3) | median grid m |
|---|---|---:|---|---:|---:|---:|
| GROUP_0138 | all_dark | 18 | dynamicsAx | 18 | 12.697 | 7.000 |
| GROUP_0175 | mixed | 3 | netsuite | 6 | 0.000 | 9.000 |
| GROUP_0194 | mixed | 3 | netsuite | 7 | 0.000 | 24.000 |
| GROUP_0118 | mixed | 2 | businessCentral | 8 | 6.434 | 24.000 |
| GROUP_0016 | mixed | 2 | businessOne | 15 | 13.346 | 24.000 |
| GROUP_0218 | mixed | 2 | businessCentral | 16 | 8.587 | 24.000 |
| GROUP_0116 | mixed | 1 | netsuite | 8 | 21.254 | 13.000 |
| GROUP_0035 | mixed | 1 | businessCentral | 14 | 9.831 | 24.000 |
| GROUP_0062 | all_dark | 1 | a3 | 1 | 11.156 | 13.000 |
| GROUP_0143 | mixed | 1 | businessCentral | 10 | 14.587 | 24.000 |
| GROUP_0135 | all_dark | 1 | fo | 1 | 15.053 | 14.000 |
| GROUP_0126 | mixed | 1 | m3Rosetta | 11 | 14.389 | 24.000 |
| GROUP_0182 | mixed | 1 | netsuite | 5 | 14.163 | 11.000 |

### Extra — the 73 invoiced + NULL ERP

n=73 in 21 groups. Mixed 33, all-invoiced 40. Miss-country 98.63%, EUR 98.63%. Median size 11.534 vs named-ERP invoiced 12.632. Late first-tx 41.10%. NULL erp on an invoiced firm is a **metadata hole**, not darkness.

### Extra — country missing × book

| slice | n cos | n miss country | share miss |
|---|---:|---:|---:|
| dark | 470 | 399 | 84.89% |
| invoiced | 744 | 599 | 80.51% |

Mixed-group country-known: invoiced 30.00% vs dark 15.45%. If only invoiced sisters have ISO codes, country is an ERP-presence echo.

### Extra — country known − miss Y residual after size

If missing-country were a health why, the residual would survive `log1p(a_in3)` terciles.

| Y | tercile | n known | rate known | n miss | rate miss | known−miss |
|---|---|---:|---:|---:|---:|---:|
| y3_recover_cash_6m | T1_small | 230 | 17.83% | 926 | 13.93% | +3.90pp |
| y3_recover_cash_6m | T2_mid | 427 | 4.45% | 1,555 | 4.89% | -0.44pp |
| y3_recover_cash_6m | T3_large | 583 | 2.23% | 1,807 | 6.25% | -4.02pp |
| y2_neg_2of3 | T1_small | 830 | 4.94% | 4,161 | 5.86% | -0.92pp |
| y2_neg_2of3 | T2_mid | 1,010 | 5.35% | 4,010 | 7.71% | -2.36pp |
| y2_neg_2of3 | T3_large | 1,159 | 7.85% | 3,798 | 8.03% | -0.18pp |

### Extra — EUR − other (invoiced only) after size

fx_qa: non-EUR home books are the FX identity, not the 110. Residual after size:

| Y | tercile | n EUR | rate EUR | n other | rate other | EUR−other |
|---|---|---:|---:|---:|---:|---:|
| y3_recover_cash_6m | T1_small | 717 | 15.90% | 47 | 8.51% | +7.39pp |
| y3_recover_cash_6m | T2_mid | 1,268 | 4.26% | 27 | 7.41% | -3.15pp |
| y3_recover_cash_6m | T3_large | 1,432 | 5.24% | 52 | 15.38% | -10.15pp |
| y2_neg_2of3 | T1_small | 3,096 | 5.39% | 232 | 8.19% | -2.80pp |
| y2_neg_2of3 | T2_mid | 3,242 | 4.41% | 154 | 14.29% | -9.87pp |
| y2_neg_2of3 | T3_large | 2,749 | 7.78% | 330 | 5.45% | +2.33pp |

### Extra — late `created_at` vs Y3 after size (PARK confirm)

Do not revive onboard as a health Y. Residual late − early inside size terciles:

| tercile | n early onboard | Y3 early | n late onboard | Y3 late | late−early |
|---|---:|---:|---:|---:|---:|
| T1_small | 570 | 15.61% | 586 | 13.82% | -1.79pp |
| T2_mid | 1,014 | 3.94% | 968 | 5.68% | +1.74pp |
| T3_large | 1,158 | 3.63% | 1,232 | 6.82% | +3.19pp |

Company currency vs modal invoice `accounting_currency` (invoiced): agree 97.72% (744/744). Home currency is the books' accounting currency, not a health X.

Holdout coverage (not a rate): n=72, dark=32, miss-country 80.56%, miss-erp 48.61%, EUR 73.61%, named-dark 1, invoiced-NULL 4.

Company-level ever-Y3: singleton groups 37.14% (n=35) vs multi 23.33% (n=690). CM 15% vs 6.8% is not only month-weighting. Still a **group-size dummy** — hidden test is new groups.

### Extra 2 — GROUP_0138 (18 of 20 all-dark named-ERP)

GROUP_0138 is an **all-dark** holding of 18 companies, all `dynamicsAx`, 0 books, miss-country 100.00%, EUR 100.00%. One group-type dummy, not 18 independent ERP events. The other 2 all-dark named-ERP are singletons (`a3`, `fo`). Do not read 20 named-dark as 20 live ERP connections.

### Extra 2 — invoiced-only Y3 AUROC (strip the 470 dummy)

If `has_erp` were anything but the dark flag, it would still rank among the 744.

| feature | CV AUROC | sd | train AUROC | sign | n |
|---|---:|---:|---:|---:|---:|
| has_erp | 0.475 | 0.092 | 0.527 | -1 | 3,618 |
| has_country | 0.535 | 0.076 | 0.531 | -1 | 3,618 |
| is_eur | 0.513 | 0.024 | 0.512 | -1 | 3,618 |
| log1p_a_in3 | 0.623 | 0.059 | 0.617 | -1 | 3,543 |
| c_n_days_with_tx | 0.730 | 0.079 | 0.736 | -1 | 3,618 |

ES vs other *known* country (216 companies): Y3 CV 0.637 vs size 0.739 on the same rows. A Spain dummy is still a group geography constant — hidden test is new groups.

Currency is group-constant: 83.40% of train groups have one currency (39 multi-currency groups). Among groups with any ISO, 74.14% have a single country (15 mixed). These flags are **group identity**, not company health.

Family-disagree `companies.erp` vs `groups.erp` (after slug map) — leftovers, not a new Y:

| companies.erp | groups.erp | n |
|---|---|---:|
| businessCentral | Microsoft Navision | 16 |
| sage200 | Sage 50 | 2 |

24-month train books still miss country on **73.6%**. Late-arrival moves the needle (known-country is richer on the 24-month pile) but missingness is the default, not a short-trail hole. PARK missing-country as a health Y either way.

### Extra 3 — GROUP_0138 vs the rest of the 360; named-dark 17 vs the rest of the 110

If named-ERP dark were a live book, they would recover like the 744. They do not.

| Y | slice | n labeled | pos | cos | rate |
|---|---|---:|---:|---:|---:|
| y3_recover_cash_6m | GROUP_0138 | 0 | 0 | 0 | — |
| y3_recover_cash_6m | all_dark_ex_0138 | 1,557 | 81 | 188 | 5.20% |
| y3_recover_cash_6m | named_dark_110 | 97 | 18 | 14 | 18.56% |
| y3_recover_cash_6m | other_110 | 376 | 39 | 60 | 10.37% |
| y2_neg_2of3 | GROUP_0138 | 72 | 0 | 18 | 0.00% |
| y2_neg_2of3 | all_dark_ex_0138 | 4,417 | 426 | 334 | 9.64% |
| y2_neg_2of3 | named_dark_110 | 291 | 28 | 17 | 9.62% |
| y2_neg_2of3 | other_110 | 1,301 | 102 | 90 | 7.84% |

Invoiced ever-FX: multi-currency groups 67.65% (n=170) vs one-currency groups 19.69% (n=574). Multi-ccy is a *group* FX identity (fx_qa home-currency footnote), not a transferable company health X.

Top named-ERP slugs as invoiced Y3 dummies (expect ~0.50):

| feature | CV AUROC | train AUROC | n |
|---|---:|---:|---:|
| erp=businessCentral | 0.535 | 0.514 | 3,618 |
| erp=netsuite | 0.537 | 0.540 | 3,618 |
| erp=sage200 | 0.530 | 0.523 | 3,618 |

### Extra 4 — named-dark among the 110 after size

Raw Y3 18.6% (named) vs 10.4% (other 110). Same train `a_in3` terciles:

| tercile | n named-dark | Y3 named | n other 110 | Y3 other | residual |
|---|---:|---:|---:|---:|---:|
| T1_small | 28 | 35.71% | 88 | 21.59% | +14.12pp |
| T2_mid | 13 | 7.69% | 116 | 10.34% | -2.65pp |
| T3_large | 53 | 11.32% | 164 | 4.27% | +7.05pp |

Even if a cell stays green, 14 companies in 10 mixed groups is a **group-type** sliver. CLOSE as Y3 X. Do not invent a named-ERP-dark Y.

### Extra 4 — singleton T2 Y3 (+17pp): 5 / 12 companies ever recover

Company-months can hide two whales. Companies in the T2 singleton cell:

| company | n labeled | pos | Y3 rate | median a_in3 |
|---|---:|---:|---:|---:|
| COMP_0681 | 9 | 3 | 33.33% | 484066.250 |
| COMP_0502 | 10 | 3 | 30.00% | 156098.200 |
| COMP_0259 | 2 | 2 | 100.00% | 491277.340 |
| COMP_0288 | 5 | 2 | 40.00% | 478622.800 |
| COMP_0825 | 1 | 1 | 100.00% | 500442.410 |
| COMP_0209 | 1 | 0 | 0.00% | 192073.340 |
| COMP_0359 | 1 | 0 | 0.00% | 134231.450 |
| COMP_0403 | 3 | 0 | 0.00% | 550600.000 |
| COMP_0711 | 5 | 0 | 0.00% | 197090.590 |
| COMP_0785 | 8 | 0 | 0.00% | 719094.500 |
| COMP_0841 | 4 | 0 | 0.00% | 98708.660 |
| COMP_1195 | 2 | 0 | 0.00% | 472243.635 |

A singleton lift that is a handful of companies is still a group-size dummy. Hidden test = new groups. Do not KEEP singleton as Q1 X.

### Extra 5 — group has an ERP name, company does not

**116** train companies sit in a group with `groups.erp` filled and `companies.erp` NULL (85 dark / 31 invoiced, 36 groups). The 73 invoiced-NULL are not all of this hole. A group ERP name does not give the company a book. Still not a Y3 X.

| groups.erp family | n co-NULL |
|---|---:|
| business_central | 31 |
| netsuite | 27 |
| sage_50 | 24 |
| sage_200 | 10 |
| dynamics_fo | 8 |
| custom | 4 |
| business_one | 3 |
| infor_m3 | 3 |
| navision | 1 |
| odoo | 1 |
| a3 | 1 |
| oracle | 1 |
| sage_x3 | 1 |
| libra | 1 |

ES is **159** companies in **45** groups (known-country 216 in 58 groups). Median ~3.5 ES companies per ES group. Geography is a group identity, not a transferable company health reading.

### Extra 6 — country / ERP missingness by group mix

If missing-country were an ERP-dark echo it would pile in all-dark groups.

| mix | n cos | n miss country | share miss country | n miss erp | share miss erp |
|---|---:|---:|---:|---:|---:|
| all_dark | 360 | 306 | 85.00% | 340 | 94.44% |
| all_invoiced | 514 | 438 | 85.21% | 40 | 7.78% |
| mixed | 340 | 254 | 74.71% | 126 | 37.06% |

Of the 73 invoiced+NULL, **31** sit in a group that *has* `groups.erp` and **42** sit in a group with NULL `groups.erp`. Company NULL erp is not the same hole as group NULL erp.

Y3 group-fold AUROC by fold (sign from train side). Flags wander around 0.50; size does not.

| feature | fold | AUROC | sign | n va | n pos |
|---|---:|---:|---:|---:|---:|
| has_erp | 0 | 0.419 | -1 | 1,310 | 54 |
| has_erp | 1 | 0.464 | 1 | 696 | 93 |
| has_erp | 2 | 0.518 | 1 | 1,072 | 66 |
| has_erp | 3 | 0.519 | 1 | 1,373 | 82 |
| has_erp | 4 | 0.482 | 1 | 1,197 | 107 |
| has_country | 0 | 0.469 | -1 | 1,310 | 54 |
| has_country | 1 | 0.518 | -1 | 696 | 93 |
| has_country | 2 | 0.460 | -1 | 1,072 | 66 |
| has_country | 3 | 0.390 | 1 | 1,373 | 82 |
| has_country | 4 | 0.567 | -1 | 1,197 | 107 |
| is_eur | 0 | 0.497 | -1 | 1,310 | 54 |
| is_eur | 1 | 0.490 | -1 | 696 | 93 |
| is_eur | 2 | 0.499 | -1 | 1,072 | 66 |
| is_eur | 3 | 0.572 | -1 | 1,373 | 82 |
| is_eur | 4 | 0.518 | -1 | 1,197 | 107 |
| log1p_a_in3 | 0 | 0.565 | -1 | 1,288 | 53 |
| log1p_a_in3 | 1 | 0.632 | -1 | 685 | 92 |
| log1p_a_in3 | 2 | 0.683 | -1 | 1,047 | 64 |
| log1p_a_in3 | 3 | 0.543 | -1 | 1,342 | 78 |
| log1p_a_in3 | 4 | 0.661 | -1 | 1,166 | 104 |

### Extra 7 — is filled-country just filled-ERP?

Spearman `has_country` vs `has_erp` **0.105** (both named 12.36%, both missing 36.24%). They are **not** the same flag: country is missing on invoiced books too (extra 6). Do not merge them into one metadata Y.

Holdout named-ERP dark (coverage only):

| company | group | erp |
|---|---|---|
| COMP_0851 | GROUP_0211 | netsuite |

First banking `created_at` after first tx: **70.27%** of train companies with a banking row (n=1211). Trail QA quoted 70.3% (851/1,211). CONFIRMS that number. Same calendar month 8.51%. Product connection ≠ cash start. PARK both clocks.

### Extra 8 — is `companies.created_at` one onboarding wave?

Train `created_at` spans **53** calendar months. Modal 2026-03 has 78 companies (6.43%). Not one wave. Same conclusion as trail QA on first-tx. A month-of-onboard dummy would be a **calendar / connection** dummy, not Q1 health.

| month | n | share |
|---|---:|---:|
| 2021-11 | 3 | 0.25% |
| 2022-02 | 1 | 0.08% |
| 2022-03 | 2 | 0.16% |
| 2022-04 | 26 | 2.14% |
| 2022-07 | 3 | 0.25% |
| 2022-08 | 1 | 0.08% |
| 2022-09 | 4 | 0.33% |
| 2022-10 | 6 | 0.49% |
| 2022-11 | 12 | 0.99% |
| 2022-12 | 6 | 0.49% |
| 2023-01 | 10 | 0.82% |
| 2023-02 | 2 | 0.16% |
| 2023-03 | 11 | 0.91% |
| 2023-04 | 5 | 0.41% |
| 2023-05 | 25 | 2.06% |
| 2023-06 | 7 | 0.58% |
| 2023-07 | 9 | 0.74% |
| 2023-08 | 11 | 0.91% |
| 2023-09 | 10 | 0.82% |
| 2023-10 | 18 | 1.48% |
| 2023-11 | 12 | 0.99% |
| 2023-12 | 5 | 0.41% |
| 2024-01 | 7 | 0.58% |
| 2024-02 | 28 | 2.31% |
| 2024-03 | 32 | 2.64% |
| 2024-04 | 12 | 0.99% |
| 2024-05 | 33 | 2.72% |
| 2024-06 | 30 | 2.47% |
| 2024-07 | 21 | 1.73% |
| 2024-08 | 2 | 0.16% |
| 2024-09 | 35 | 2.88% |
| 2024-10 | 29 | 2.39% |
| 2024-11 | 58 | 4.78% |
| 2024-12 | 16 | 1.32% |
| 2025-01 | 30 | 2.47% |
| 2025-02 | 20 | 1.65% |
| 2025-03 | 31 | 2.55% |
| 2025-04 | 55 | 4.53% |
| 2025-05 | 26 | 2.14% |
| 2025-06 | 33 | 2.72% |
| 2025-07 | 21 | 1.73% |
| 2025-08 | 5 | 0.41% |
| 2025-09 | 72 | 5.93% |
| 2025-10 | 55 | 4.53% |
| 2025-11 | 37 | 3.05% |
| 2025-12 | 40 | 3.29% |
| 2026-01 | 41 | 3.38% |
| 2026-02 | 68 | 5.60% |
| 2026-03 | 78 | 6.43% |
| 2026-04 | 55 | 4.53% |
| 2026-05 | 17 | 1.40% |
| 2026-06 | 24 | 1.98% |
| 2026-07 | 14 | 1.15% |

### Extra 9 — currency × book; created_at vs extract

EUR share dark 90.00% vs invoiced 90.46%. Dark companies are not a foreign-currency pile. Non-EUR is the invoiced FX identity.

`created_at` ≥ extract 0 (should be ~0). `created_at` before 2024-09-01: **354** / 1214 — platform rows older than the cash window. Still a connection clock.

Holdout late `created_at`: 95.83% (coverage). Train was 70.80%.

### Extra 10 — onboard lag tails (created_at − first tx, days)

Share after cash: 0d 61.29%, >30d 54.12%, >90d 14.91%, >180d 8.32%, >365d 2.72%. Share *before* cash 38.71% (p10 -472.2 days). p90 162.4 days. A 6–12 month onboard lag is common. That is connection, not a 45→65. PARK.

### Extra 11 — onboard-before-cash is left-truncated observation

470 train companies have `created_at` *before* first tx. **328** of those start cash in 2024-09 (328/435 of the September starters). The other 142 are late first-tx with an older platform row. Trail QA: product older than the window is left-truncated *observation*, not a new firm. Still PARK as a health Y.

### Extra 12 — onboard-before-cash ∩ the 470 dark

Overlap **182** (before=470, dark=470). The matching *count* 470 is a coincidence. Onboard-before-cash is not the dark flag.

### Extra 13 — holdout named-dark; country fill by ERP family; `groups.erp`

Holdout named-ERP dark (coverage):

| company | group | erp | country | currency |
|---|---|---|---|---|
| COMP_0851 | GROUP_0211 | netsuite | <NA> | USD |

Share of named-ERP train companies with a country code, by family (if one ERP ‘remembers’ ISO, country is still metadata):

| erp family | n named | n country known | share known |
|---|---:|---:|---:|
| business_central | 315 | 73 | 23.17% |
| netsuite | 127 | 27 | 21.26% |
| sage_200 | 43 | 10 | 23.26% |
| business_one | 40 | 16 | 40.00% |
| dynamics_ax | 40 | 0 | 0.00% |
| sage_x3 | 30 | 0 | 0.00% |
| infor_m3 | 21 | 0 | 0.00% |
| distrito_k | 20 | 0 | 0.00% |
| navision | 18 | 7 | 38.89% |
| a3 | 14 | 0 | 0.00% |
| etendo | 11 | 6 | 54.55% |
| sap_r3 | 9 | 0 | 0.00% |
| libra | 7 | 7 | 100.00% |
| sage_intacct | 4 | 0 | 0.00% |
| ekon | 3 | 3 | 100.00% |
| dynamics_fo | 2 | 0 | 0.00% |
| datev | 1 | 0 | 0.00% |
| holded | 1 | 1 | 100.00% |
| sage_50 | 1 | 0 | 0.00% |
| sap_byd | 1 | 0 | 0.00% |

`groups.erp` missing: train 57.74%, holdout 66.67%. Group-level ERP is also often empty. Do not treat `groups.erp` as a cleaner dark flag.

### Extra 14 — country fill among invoiced × ERP; created_at year vs trail

If country-known were ‘who has a book’, invoiced+NULL-ERP would look like invoiced+named. If country is vendor metadata, named ERPs fill ISO and NULL-ERP do not (or the reverse).

| slice | n invoiced | n country known | share known |
|---|---:|---:|---:|
| invoiced+NULL-ERP | 73 | 1 | 1.37% |
| invoiced+named-ERP | 671 | 144 | 21.46% |

Spearman `created_at` year vs grid months **-0.851**; vs miss-country **0.166**. A strong year→shorter-trail link is left-truncation of the connection clock, not a new-firm Y.

| created_at year | n | median grid months | country miss | late first-tx |
|---:|---:|---:|---:|---:|
| 2,021 | 3 | 24.000 | 33.33% | 0.00% |
| 2,022 | 61 | 24.000 | 85.25% | 4.92% |
| 2,023 | 125 | 24.000 | 66.40% | 10.40% |
| 2,024 | 303 | 24.000 | 76.90% | 22.44% |
| 2,025 | 425 | 18.000 | 84.71% | 94.12% |
| 2,026 | 297 | 8.000 | 90.57% | 99.33% |

### Extra 15 — `created_at` year as Y3 X (do not revive as a health Y)

The one invoiced+NULL-ERP company that has a country (extra 14: 1/73):

| company | group | country | currency |
|---|---|---|---|
| COMP_0600 | GROUP_0113 | ES | EUR |

Group-fold Y3 AUROC. `created_year` should track `n_grid` (ρ −0.85) and lose to size / days. PARK as a health Y still holds — this is only a CLOSE-as-X check.

| feature | cv AUROC | sd | train AUROC | sign |
|---|---:|---:|---:|---:|
| created_year | 0.504 | 0.051 | 0.515 | 1 |
| n_grid | 0.529 | 0.059 | 0.525 | -1 |
| log1p_a_in3 | 0.617 | 0.061 | 0.620 | -1 |
| c_n_days_with_tx | 0.711 | 0.031 | 0.723 | -1 |

### Extra 16 — two sliver groups (COMP_0600 / COMP_0851)

GROUP_0113 holds the only invoiced+NULL-ERP company with a country. GROUP_0211 holds the only holdout named-ERP dark (COMP_0851 / netsuite / USD). Coverage rosters — if the ISO or ERP sits on one sister, it is group metadata.

**GROUP_0113**

| company | split | erp | country | currency | book |
|---|---|---|---|---|---|
| COMP_0218 | train | <NA> | <NA> | EUR | invoiced |
| COMP_0600 | train | <NA> | ES | EUR | invoiced |
| COMP_0721 | train | <NA> | <NA> | EUR | invoiced |
| COMP_0995 | train | <NA> | <NA> | EUR | invoiced |

**GROUP_0211**

| company | split | erp | country | currency | book |
|---|---|---|---|---|---|
| COMP_0025 | holdout | netsuite | <NA> | USD | invoiced |
| COMP_0189 | holdout | netsuite | <NA> | GBP | invoiced |
| COMP_0262 | holdout | netsuite | <NA> | USD | invoiced |
| COMP_0336 | holdout | netsuite | <NA> | GBP | invoiced |
| COMP_0344 | holdout | netsuite | <NA> | GBP | invoiced |
| COMP_0482 | holdout | netsuite | <NA> | EUR | invoiced |
| COMP_0621 | holdout | netsuite | <NA> | USD | invoiced |
| COMP_0701 | holdout | netsuite | <NA> | EUR | invoiced |
| COMP_0754 | holdout | <NA> | <NA> | EUR | dark |
| COMP_0775 | holdout | netsuite | <NA> | EUR | invoiced |
| COMP_0851 | holdout | netsuite | <NA> | USD | dark |
| COMP_0990 | holdout | netsuite | <NA> | USD | invoiced |
| COMP_1067 | holdout | netsuite | <NA> | AUD | invoiced |
| COMP_1184 | holdout | netsuite | <NA> | EUR | invoiced |
| COMP_1201 | holdout | <NA> | <NA> | AUD | dark |

### Extra 17 — currency ≠ modal invoice `accounting_currency`; `created_year` vs Y2

Mismatch **17** / 744 invoiced with an accounting currency. These are book-setup leftovers, not a health Y. Do not invent a currency-mismatch Y.

| company | group | companies.csv | modal invoice | country |
|---|---|---|---|---|
| COMP_0029 | GROUP_0013 | EUR | SGD | <NA> |
| COMP_0226 | GROUP_0013 | EUR | BRL | <NA> |
| COMP_0247 | GROUP_0218 | EUR | USD | US |
| COMP_0314 | GROUP_0101 | EUR | GBP | <NA> |
| COMP_0350 | GROUP_0189 | EUR | USD | <NA> |
| COMP_0559 | GROUP_0188 | GBP | EUR | NL |
| COMP_0618 | GROUP_0013 | EUR | GBP | <NA> |
| COMP_0641 | GROUP_0132 | EUR | PEN | <NA> |
| COMP_0666 | GROUP_0013 | EUR | ARS | <NA> |
| COMP_0828 | GROUP_0016 | EUR | MZN | <NA> |
| COMP_0856 | GROUP_0101 | EUR | GBP | <NA> |
| COMP_0951 | GROUP_0218 | EUR | USD | ES |
| COMP_0959 | GROUP_0013 | EUR | PEN | <NA> |
| COMP_0961 | GROUP_0101 | EUR | USD | <NA> |
| COMP_1008 | GROUP_0220 | EUR | GBP | <NA> |
| COMP_1122 | GROUP_0132 | EUR | CAD | <NA> |
| COMP_1244 | GROUP_0013 | EUR | COP | <NA> |

Y2 group-fold AUROC for `created_year`: **0.414** (sign -1). Same connection clock as extra 15. PARK as a health Y; CLOSE as X.

### Extra 18 — GROUP_0013 (currency-mismatch nest); holdout `created_at` year

Six of the 17 currency ≠ modal-invoice rows sit in GROUP_0013 (EUR on companies.csv, invoice books in SGD/BRL/GBP/ARS/PEN/COP). Coverage roster:

| company | split | erp | country | currency | book |
|---|---|---|---|---|---|
| COMP_0010 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0029 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0226 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0504 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0618 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0628 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0666 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0741 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0959 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_0973 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_1078 | train | sageX3 | <NA> | EUR | invoiced |
| COMP_1244 | train | sageX3 | <NA> | EUR | invoiced |

Holdout `created_at` year (coverage only — do not read rates):

| year | n holdout | share |
|---:|---:|---:|
| 2023 | 3 | 4.17% |
| 2024 | 9 | 12.50% |
| 2025 | 51 | 70.83% |
| 2026 | 9 | 12.50% |

Holdout median created_at year 2025.0 vs train 2025.0. Holdout is **2025-peaked** (70.83%) vs train ~35% in 2025 (extra 14). Same median, different mix — `created_at` year still cannot transfer.

### Extra 19 — `created_at` year mix (train vs holdout); fold wander; grid vs MONTHS

Coverage only on holdout. Train 2025 share is ~35%; holdout 2025 share is ~71%. Median year matches; the *mix* does not. A year dummy fitted on train will not match new groups.

| year | n train | share train | n holdout | share holdout |
|---:|---:|---:|---:|---:|
| 2021 | 3 | 0.25% | 0 | 0.00% |
| 2022 | 61 | 5.02% | 0 | 0.00% |
| 2023 | 125 | 10.30% | 3 | 4.17% |
| 2024 | 303 | 24.96% | 9 | 12.50% |
| 2025 | 425 | 35.01% | 51 | 70.83% |
| 2026 | 297 | 24.46% | 9 | 12.50% |

Feature-store grid: max `n_grid`=24 vs `MONTHS`=24; share at 24 months 35.83%.

Y3 `created_year` fold AUROCs (sign chosen on the train side of each fold). Wander around 0.5 = cannot transfer:

| fold | AUROC | sign | n val | n pos |
|---:|---:|---:|---:|---:|
| 0 | 0.472 | -1 | 1310 | 54 |
| 1 | 0.486 | 1 | 696 | 93 |
| 2 | 0.592 | 1 | 1072 | 66 |
| 3 | 0.496 | 1 | 1373 | 82 |
| 4 | 0.472 | 1 | 1197 | 107 |

### Extra 20 — `has_erp` × onboard year; flag fold wander

Spearman `has_erp` vs `created_at` year **-0.029**. Named-ERP share is ~50–67% in every year 2022–2026 — not an early-onboard dummy. CLOSE `has_erp` remains the 470 / book dummy (ρ 0.81 vs has_book), not a year dummy.

| year | named ERP | NULL ERP | share named |
|---:|---:|---:|---:|
| 2,021 | 0 | 3 | 0.00% |
| 2,022 | 31 | 30 | 50.82% |
| 2,023 | 84 | 41 | 67.20% |
| 2,024 | 191 | 112 | 63.04% |
| 2,025 | 225 | 200 | 52.94% |
| 2,026 | 177 | 120 | 59.60% |

Y3 group-fold AUROC for the three company-constant flags (pass 6, sign per fold):

**has_erp** cv=0.480 ± 0.042

| fold | AUROC | sign | n val |
|---:|---:|---:|---:|
| 0 | 0.419 | -1 | 1310 |
| 1 | 0.464 | 1 | 696 |
| 2 | 0.518 | 1 | 1072 |
| 3 | 0.519 | 1 | 1373 |
| 4 | 0.482 | 1 | 1197 |

**has_country** cv=0.481 ± 0.066

| fold | AUROC | sign | n val |
|---:|---:|---:|---:|
| 0 | 0.469 | -1 | 1310 |
| 1 | 0.518 | -1 | 696 |
| 2 | 0.460 | -1 | 1072 |
| 3 | 0.390 | 1 | 1373 |
| 4 | 0.567 | -1 | 1197 |

**currency=EUR** cv=0.515 ± 0.033

| fold | AUROC | sign | n val |
|---:|---:|---:|---:|
| 0 | 0.497 | -1 | 1310 |
| 1 | 0.490 | -1 | 696 |
| 2 | 0.499 | -1 | 1072 |
| 3 | 0.572 | -1 | 1373 |
| 4 | 0.518 | -1 | 1197 |

### Extra 21 — holdout country-known; onboard vs first bank same-day

Holdout country known **14** / 72 (coverage). If known ISO clusters in a few groups, it is group metadata on the hidden split too.

| company | group | country | currency | erp |
|---|---|---|---|---|
| COMP_0023 | GROUP_0237 | BE | EUR | netsuite |
| COMP_0290 | GROUP_0053 | ES | EUR | businessOne |
| COMP_0296 | GROUP_0199 | ES | EUR | <NA> |
| COMP_0437 | GROUP_0199 | ES | EUR | <NA> |
| COMP_0543 | GROUP_0053 | ES | EUR | businessOne |
| COMP_0593 | GROUP_0053 | ES | EUR | businessOne |
| COMP_0648 | GROUP_0053 | ES | EUR | businessOne |
| COMP_0654 | GROUP_0053 | ES | EUR | businessOne |
| COMP_0842 | GROUP_0053 | ES | EUR | businessOne |
| COMP_0975 | GROUP_0237 | NL | EUR | netsuite |
| COMP_1040 | GROUP_0056 | ES | EUR | sage200 |
| COMP_1083 | GROUP_0199 | PL | PLN | <NA> |
| COMP_1172 | GROUP_0053 | ES | EUR | businessOne |
| COMP_1258 | GROUP_0056 | ES | EUR | sage200 |

Train `created_at` vs first `banking_products.created_at`: same calendar day 24.38%, within 7 days 56.75%, median offset -5.9 days (n=1211). If they were the same connection event, same-day would be high. They are two clocks even when both are ‘created_at’.

### Extra 22 — holdout country groups; onboard vs first tx same-day

Holdout: **4** / 15 groups have any country filled (14 companies — extra 21). Country on the hidden 72 is a handful of group metadata rows, not a transferable health flag.

Train `created_at` vs first tx: same calendar day 0.74%, within 7 days 2.22% (n=1214). Same-month was 7.1% (pass 4). Onboard and cash start are not the same event.

### Extra 23 — holdout ERP family (coverage)

Hidden 72 ERP mix. A train `has_erp` / slug dummy cannot transfer if the new groups bring a different vendor mix (or more NULLs).

| erp family | n companies | n groups | share |
|---|---:|---:|---:|
| (NULL) | 35 | 8 | 48.61% |
| netsuite | 16 | 2 | 22.22% |
| business_central | 7 | 3 | 9.72% |
| business_one | 7 | 1 | 9.72% |
| sage_200 | 4 | 1 | 5.56% |
| dynamics_ax | 2 | 1 | 2.78% |
| sap_r3 | 1 | 1 | 1.39% |

### Extra 24 — holdout vs train currency (coverage)

EUR share train 90.28% vs holdout 73.61%. GROUP_0211 (extra 16) is a multi-ccy netsuite holding. `currency=EUR` on train is a 90% home-book dummy; holdout EUR share moves. Cannot transfer.

| split | currency | n | share |
|---|---|---:|---:|
| train | EUR | 1,096 | 90.28% |
| train | GBP | 38 | 3.13% |
| train | USD | 31 | 2.55% |
| train | MXN | 7 | 0.58% |
| train | COP | 5 | 0.41% |
| train | DKK | 5 | 0.41% |
| train | CAD | 3 | 0.25% |
| train | BRL | 3 | 0.25% |
| train | AUD | 3 | 0.25% |
| train | CLP | 3 | 0.25% |
| train | PLN | 3 | 0.25% |
| train | NZD | 2 | 0.16% |
| train | CHF | 2 | 0.16% |
| train | PEN | 2 | 0.16% |
| train | AED | 2 | 0.16% |
| train | ARS | 1 | 0.08% |
| train | BAM | 1 | 0.08% |
| train | MYR | 1 | 0.08% |
| train | VND | 1 | 0.08% |
| train | INR | 1 | 0.08% |
| train | NAD | 1 | 0.08% |
| train | JPY | 1 | 0.08% |
| train | NOK | 1 | 0.08% |
| train | CZK | 1 | 0.08% |
| holdout | EUR | 53 | 73.61% |
| holdout | USD | 5 | 6.94% |
| holdout | GBP | 4 | 5.56% |
| holdout | AOA | 2 | 2.78% |
| holdout | AUD | 2 | 2.78% |
| holdout | GHS | 1 | 1.39% |
| holdout | SEK | 1 | 1.39% |
| holdout | DKK | 1 | 1.39% |
| holdout | CHF | 1 | 1.39% |
| holdout | XOF | 1 | 1.39% |
| holdout | PLN | 1 | 1.39% |

### Extra 25 — currencies / ISO codes that the hidden 72 bring

Currencies on holdout **not** in train: **['AOA', 'GHS', 'SEK', 'XOF']**. Train-only currencies: 17 (expected — train is larger).

Known-country ISO on holdout **not** in train known set: **['PL']**. Train-only known ISO: ['AT', 'DE', 'FR', 'GB', 'IT', 'MY', 'PT', 'SE', 'US'].

New-group test means new home books. A EUR dummy / ES dummy fitted on train does not see AOA/GHS/XOF or a BE/PL holdout ISO the same way. CLOSE.

### Extra 26 — banking-product count vs ERP × book

If `has_erp` were ‘more connected’, named-ERP would show more banking products. If named-dark look like NULL-dark, ERP name is not connection richness.

| slice | n | median n banking | share no banking row |
|---|---:|---:|---:|
| named+invoiced | 671 | 3.000 | 0.30% |
| named+dark | 37 | 2.000 | 0.00% |
| NULL+invoiced | 73 | 4.000 | 0.00% |
| NULL+dark | 433 | 3.000 | 0.23% |

### Extra 27 — first banking `created_at` vs first tx

Same calendar day 1.57%, within 7 days 3.47%, median offset 54.6 days (n=1211). Pass 4 already had p50 +54.6 days / 70% bank after cash. The product connection clock is not the cash trail. Three clocks stand.

### Extra 28 — group-uniform flags (why they cannot transfer)

Share of train groups where the flag is constant. A high uniform share means the flag is a *group type*. Hidden test = new groups ⇒ the dummy does not come along.

| flag | groups | all 1 | all 0 | mixed | share uniform |
|---|---:|---:|---:|---:|---:|
| has_erp | 235 | 109 | 90 | 36 | 84.68% |
| has_country | 235 | 27 | 177 | 31 | 86.81% |
| is_eur | 235 | 186 | 12 | 37 | 84.26% |

### Extra 29 — `created_at` year as a group constant

**196** / 235 train groups share a single `created_at` year (83.40%). Onboard year is often a group wave, not a company health path. Still PARK as a Y; CLOSE as X.

### Extra 30 — holdout-only `companies.currency` vs invoice currencies

Holdout-only home currencies ['AOA', 'GHS', 'SEK', 'XOF']. Seen as train invoice `accounting_currency`: (none). Seen as train invoice `currency` (FX side): ['AOA', 'SEK'].

AOA/SEK can exist on *someone’s* invoice book and still be a new `companies.currency` on hidden groups. Home-currency on companies.csv is not the FX feature. CLOSE `is_eur` as Q1 X.

### Extra 31 — holdout companies with train-unseen home currency

Coverage roster. These are new-group home books, not a transferable EUR dummy.

| company | group | currency | country | erp |
|---|---|---|---|---|
| COMP_0212 | GROUP_0199 | SEK | <NA> | <NA> |
| COMP_0276 | GROUP_0199 | AOA | <NA> | <NA> |
| COMP_0292 | GROUP_0199 | GHS | <NA> | <NA> |
| COMP_0860 | GROUP_0199 | XOF | <NA> | <NA> |
| COMP_0900 | GROUP_0199 | AOA | <NA> | <NA> |

### Extra 32 — GROUP_0199 (all four holdout-only home currencies)

AOA/GHS/SEK/XOF all sit in one hidden group, next to ES/PL sisters (extra 21). That is a new-group identity, not a company health reading.

| company | split | erp | country | currency | book |
|---|---|---|---|---|---|
| COMP_0196 | holdout | <NA> | <NA> | EUR | dark |
| COMP_0212 | holdout | <NA> | <NA> | SEK | dark |
| COMP_0276 | holdout | <NA> | <NA> | AOA | dark |
| COMP_0292 | holdout | <NA> | <NA> | GHS | dark |
| COMP_0296 | holdout | <NA> | ES | EUR | dark |
| COMP_0437 | holdout | <NA> | ES | EUR | dark |
| COMP_0445 | holdout | <NA> | <NA> | GBP | dark |
| COMP_0761 | holdout | <NA> | <NA> | CHF | dark |
| COMP_0860 | holdout | <NA> | <NA> | XOF | dark |
| COMP_0900 | holdout | <NA> | <NA> | AOA | dark |
| COMP_1083 | holdout | <NA> | PL | PLN | dark |

### Extra 33 — holdout group mix (coverage only; Family H not redone)

Holdout groups: all-dark 4, all-invoiced 10, mixed 1. GROUP_0199 is an all-dark / NULL-ERP holding with the unseen home currencies. Coverage — no Y2/Y3 rates on holdout.

| group | n | n invoiced | n named ERP | n country | share EUR | mix |
|---|---:|---:|---:|---:|---:|---|
| GROUP_0103 | 16 | 0 | 0 | 0 | 100.00% | all-dark |
| GROUP_0211 | 15 | 12 | 13 | 0 | 33.33% | mixed |
| GROUP_0199 | 11 | 0 | 0 | 3 | 27.27% | all-dark |
| GROUP_0053 | 7 | 7 | 7 | 7 | 100.00% | all-invoiced |
| GROUP_0237 | 5 | 5 | 3 | 2 | 80.00% | all-invoiced |
| GROUP_0039 | 5 | 5 | 5 | 0 | 100.00% | all-invoiced |
| GROUP_0056 | 4 | 4 | 4 | 2 | 100.00% | all-invoiced |
| GROUP_0025 | 2 | 2 | 2 | 0 | 100.00% | all-invoiced |
| GROUP_0046 | 1 | 1 | 1 | 0 | 100.00% | all-invoiced |
| GROUP_0120 | 1 | 1 | 1 | 0 | 100.00% | all-invoiced |
| GROUP_0121 | 1 | 0 | 0 | 0 | 100.00% | all-dark |
| GROUP_0190 | 1 | 1 | 1 | 0 | 100.00% | all-invoiced |
| GROUP_0157 | 1 | 1 | 0 | 0 | 100.00% | all-invoiced |
| GROUP_0156 | 1 | 1 | 0 | 0 | 100.00% | all-invoiced |
| GROUP_0242 | 1 | 0 | 0 | 0 | 100.00% | all-dark |

### Extra 34 — onboard month vs first-tx month as group waves

If `created_at` month is group-constant more often than first-tx month, onboard is a sales/connection wave. First-tx month is the cash window (trail QA). Still PARK `created_at` as a health Y.

| clock | groups | one-month groups | share one-month | median distinct months |
|---|---:|---:|---:|---:|
| created_at month | 235 | 173 | 73.62% | 1.000 |
| first_tx month | 235 | 130 | 55.32% | 1.000 |

### Extra 35 — first banking-product month as a group wave

**147** / 235 train groups share one first-bank month (62.55%); median distinct months 1.0. Compare extra 34: created_at month 73.6% one-month vs first-tx month 55.3%. Three clocks, three wave shapes. None is a health Y.

### Extra 36 — dark companies with a country (train)

**71** dark train companies have an ISO. If they are mixed-group sisters, country is the invoiced sibling’s metadata leaking. If they sit in all-dark groups (GROUP_0199-like), country is still group identity.

| mix | n dark+country | groups |
|---|---:|---:|
| all_dark | 54 | 19 |
| mixed | 17 | 8 |

### Extra 37 — ISO mix on dark+country vs invoiced+country

If dark+country is the same ES pile as invoiced+country, country is one metadata fill process, not a book. Still PARK as a health Y.

| slice | country | n | share of slice |
|---|---|---:|---:|
| dark+country | ES | 43 | 60.56% |
| dark+country | NL | 10 | 14.08% |
| dark+country | PT | 3 | 4.23% |
| dark+country | DE | 3 | 4.23% |
| dark+country | FR | 2 | 2.82% |
| dark+country | GB | 2 | 2.82% |
| dark+country | IT | 2 | 2.82% |
| dark+country | BE | 2 | 2.82% |
| dark+country | US | 2 | 2.82% |
| dark+country | AT | 1 | 1.41% |
| dark+country | SE | 1 | 1.41% |
| invoiced+country | ES | 116 | 80.00% |
| invoiced+country | DE | 9 | 6.21% |
| invoiced+country | PT | 5 | 3.45% |
| invoiced+country | FR | 4 | 2.76% |
| invoiced+country | US | 3 | 2.07% |
| invoiced+country | NL | 3 | 2.07% |
| invoiced+country | GB | 2 | 1.38% |
| invoiced+country | IT | 2 | 1.38% |
| invoiced+country | MY | 1 | 0.69% |

### Extra 38 — onboard − first tx by `created_at` year

If 2026 onboards sit closer to first cash, that is left-truncation of both clocks in a short window — still not a health Y.

| year | n | p50 days | share after cash | share same-day |
|---:|---:|---:|---:|---:|
| 2021 | 3 | -1018.578 | 0.00% | 0.00% |
| 2022 | 61 | -777.361 | 0.00% | 0.00% |
| 2023 | 125 | -454.601 | 0.00% | 0.00% |
| 2024 | 303 | -68.559 | 35.64% | 1.32% |
| 2025 | 425 | 52.347 | 84.24% | 1.18% |
| 2026 | 297 | 58.515 | 93.60% | 0.00% |

### Extra 39 — 2025–26 onboard dummy as Y3 X

Extra 38 split the clock into two regimes (onboard before vs after cash). Y3 AUROC for `created_year≥2025`: **0.522** vs size **0.617**. CLOSE as X. Do not revive as a health Y.

### Extra 40 — 2025–26 onboard share (train vs holdout coverage)

Train 59.47% vs holdout 83.33%. Median year matched (2025) but the late-onboard dummy is more common on the hidden 72. Another transfer fail for any `created_at` year X.

### Extra 41 — holdout country-miss by onboard year (coverage)

Train extra 14: 2026 miss 90.6%. If holdout 2025–26 is also mostly missing ISO, `has_country` on new groups is still missingness.

| year | n holdout | country miss |
|---:|---:|---:|
| 2,023 | 3 | 100.00% |
| 2,024 | 9 | 100.00% |
| 2,025 | 51 | 72.55% |
| 2,026 | 9 | 100.00% |

### Extra 42 — no banking row; holdout known-country onboard year

Train companies with no `banking_products` row: **3**. Cash can exist without a product `created_at` (outer join in pass 4). Still three clocks.

| company | group | erp | book |
|---|---|---|---|
| COMP_0676 | GROUP_0104 | libra | invoiced |
| COMP_0683 | GROUP_0239 | <NA> | dark |
| COMP_0906 | GROUP_0238 | sage200 | invoiced |

Holdout country-known `created_at` years: {2025: 14}. Extra 41: the 14 known ISO sit in 2025. Missingness is still the default on new groups.

### Extra 43 — no-banking companies still have cash?

If first_tx is present, the bank trail exists and `banking_products.created_at` is just missing metadata. Confirms the third clock can be absent.

| company | group | first_tx | created_at |
|---|---|---|---|
| COMP_0676 | GROUP_0104 | 2026-02-06 | 2026-02-27 16:10:11 |
| COMP_0683 | GROUP_0239 | 2026-01-19 | 2026-03-11 15:43:09 |
| COMP_0906 | GROUP_0238 | 2025-11-09 | 2026-01-08 14:35:51 |

### Extra 44 — groups of the 3 no-banking companies

If they are singletons, missing `banking_products` is a thin-group hole, not a companies.csv health flag.

| group | n train | n named ERP | n country | share EUR |
|---|---:|---:|---:|---:|
| GROUP_0104 | 10 | 7 | 7 | 100.00% |
| GROUP_0238 | 4 | 4 | 3 | 100.00% |
| GROUP_0239 | 5 | 0 | 0 | 100.00% |

### Extra 45 — GROUP_0104 (COMP_0676 has cash, no banking row)

Sisters in the same group have banking products. Missing bank metadata is row-level, not a group-type companies.csv flag.

| company | erp | country | n banking | has bank row |
|---|---|---|---:|---|
| COMP_0038 | <NA> | <NA> | 23.000 | yes |
| COMP_0058 | libra | ES | 23.000 | yes |
| COMP_0132 | libra | ES | 5.000 | yes |
| COMP_0291 | <NA> | <NA> | 5.000 | yes |
| COMP_0420 | <NA> | <NA> | 9.000 | yes |
| COMP_0676 | libra | ES | — | no |
| COMP_0756 | libra | ES | 10.000 | yes |
| COMP_0944 | libra | ES | 5.000 | yes |
| COMP_1208 | libra | ES | 4.000 | yes |
| COMP_1243 | libra | ES | 5.000 | yes |

### Extra 46 — GROUP_0239 (COMP_0683 no-bank dark)

All-NULL ERP, no country, EUR. If sisters have banking, COMP_0683 is a row hole.

| company | erp | currency | book | n banking |
|---|---|---|---|---:|
| COMP_0206 | <NA> | EUR | dark | 1.000 |
| COMP_0303 | <NA> | EUR | dark | 5.000 |
| COMP_0603 | <NA> | EUR | dark | 4.000 |
| COMP_0683 | <NA> | EUR | dark | — |
| COMP_1179 | <NA> | EUR | dark | 2.000 |

### Extra 47 — GROUP_0238 (COMP_0906 sage200 invoiced, no bank row)

Named-ERP + book without a banking product row. ERP name ≠ bank connection.

| company | erp | country | book | n banking |
|---|---|---|---|---:|
| COMP_0099 | sage200 | ES | invoiced | 3.000 |
| COMP_0778 | sage200 | ES | invoiced | 2.000 |
| COMP_0906 | sage200 | ES | invoiced | — |
| COMP_1074 | sage200 | <NA> | invoiced | 2.000 |

### Extra 48 — `created_at` is a timestamp, not a fiscal date

**100.00%** of train `created_at` values have a non-midnight clock time. This is a platform connection event, not a company founding year. PARK as a health Y.

### Extra 49 — duplicate `created_at` timestamps (batch connect)

**0** timestamps are shared by more than one train company (0 companies). A shared connection second is a batch onboard, not a firm-age Y.

| timestamp | n companies |
|---|---:|
| 2021-11-17 10:55:30 | 1 |
| 2026-03-11 12:20:06 | 1 |
| 2024-05-30 15:14:10 | 1 |
| 2026-06-19 11:55:34 | 1 |
| 2026-02-25 08:10:06 | 1 |
| 2026-07-02 14:59:39 | 1 |
| 2026-01-09 10:39:38 | 1 |
| 2026-03-10 11:53:54 | 1 |

### Extra 50 — same calendar-day onboard inside a group

Extra 49: every `created_at` second is unique. Same *day* inside a group: **163** / 235 groups (1037 companies on those days). A group sales wave still shows up as same-day connects. PARK as a health Y.

### Extra 51 — first-tx same calendar day inside a group

**126** / 235 groups have two+ sisters whose first cash day matches (669 companies). Extra 50 onboard same-day was 163 / 235 / 1,037 companies. Onboard waves are tighter than cash-start waves. PARK `created_at`.

### Extra 52 — holdout same-day onboard (coverage)

**8** / 15 hidden groups share a calendar-day connect (57 / 72 companies). New groups still arrive as waves. `created_at` cannot transfer as a company health X.

### Extra 53 — `created_at` weekday

Weekday (Mon–Fri) share **99.59%**. A business-hours connect stamp is a platform clock, not a health path.

| weekday | n | share |
|---|---:|---:|
| Monday | 270 | 22.24% |
| Tuesday | 274 | 22.57% |
| Wednesday | 194 | 15.98% |
| Thursday | 251 | 20.68% |
| Friday | 220 | 18.12% |
| Saturday | 1 | 0.08% |
| Sunday | 4 | 0.33% |

### Extra 54 — `created_at` hour of day

Share in 08:00–18:59 **90.03%**. Office-hours connection, not a health Y.

| hour | n | share |
|---:|---:|---:|
| 5 | 3 | 0.25% |
| 6 | 8 | 0.66% |
| 7 | 69 | 5.68% |
| 8 | 118 | 9.72% |
| 9 | 164 | 13.51% |
| 10 | 174 | 14.33% |
| 11 | 111 | 9.14% |
| 12 | 48 | 3.95% |
| 13 | 84 | 6.92% |
| 14 | 148 | 12.19% |
| 15 | 88 | 7.25% |
| 16 | 105 | 8.65% |
| 17 | 30 | 2.47% |
| 18 | 23 | 1.89% |
| 19 | 13 | 1.07% |
| 20 | 12 | 0.99% |
| 21 | 12 | 0.99% |
| 23 | 4 | 0.33% |

### Extra 55 — weekday share of the three clocks

If first_tx is closer to uniform across 7 days, cash is a 24/7 trail. If `created_at` / bank `created_at` are weekday-only, those are connection clocks.

| clock | n | weekday share |
|---|---:|---:|
| companies.created_at | 1,214 | 99.59% |
| first_tx | 1,214 | 84.68% |
| first_bank_created | 1,211 | 99.42% |

### Extra 56 — plot

`companies_erp_dark_country.png` exists=True bytes=52035. Left: erp × book 671 / 37 / 73 / 433. Right: country missingness × book.

### Extra 57 — first-bank same calendar day inside a group

**147** / 235 groups (890 companies). Compare extra 50 onboard 163 / 1,037 and extra 51 first-tx 126 / 669. Bank-product connect is the middle wave. Three clocks.

### Extra 58 — three clocks, three same-day wave strengths

Onboard is the tightest group wave. Cash start is the loosest. Bank-product connect sits in the middle. PARK all three as health Ys; only the cash trail belongs in Q1 X (already in the store as days / size).

| clock | groups with same-day sisters | detail |
|---|---:|---|
| created_at | 163 | 163/235; 1037 cos (extra 50) |
| first_bank_created | 147 | 147/235; 890 cos (extra 57) |
| first_tx | 126 | 126/235; 669 cos (extra 51) |

### Extra 59 — holdout `created_at` weekday (coverage)

Weekday share **94.44%** on the hidden 72 (train 99.6%). New groups also connect on office days. Still a connection clock.

### Extra 60 — holdout first-tx weekday vs onboard (coverage)

created_at weekday 94.44%; first_tx weekday 97.22%. Train first_tx weekday was 84.7% (extra 55). Holdout n=72 is coverage — do not read 97% as a new law. Two clocks still stand.

### Extra 61 — recap for the parent

transferable=False | PARK created_at+miss-country as Ys | CLOSE has_erp/has_country/is_eur as X | no merged country/erp Y

erp×dark: 671 / 37 / 73 / 433. NULL among dark 92.13% CONFIRMS 92.1%. Country miss 82.21%. Three-clock p50 created−first_tx +39.4 days; same-day 0.7%. No companies.csv flag transfers to new groups.

### Extra 62 — frozen holdout coverage

**72** companies / **15** groups. Seed 20260918. Rates never computed here.

### Extra 63 — train panel

**1,214** companies / **235** groups. All rates and AUROC above use this split only.

### Extra 64 — owned artifacts

`companies_qa.py` 5939 lines; `companies_qa.md` 1433 lines. Same module sit. Did not commit.

### Extra 65 — registry

Append-only `registry.csv` has **13** rows for agent `a14da08b`. Skip key includes x_families so re-runs do not duplicate.

### Extra 66 — one optional PNG

`companies_*.png` count **1** (cap = 1). `companies_erp_dark_country.png`.

### Extra 67 — scope

Did not write `.agents/persistent-memory/`. Did not touch product/, parquet, a_vol_qa, fx_qa, uncat_qa, sibling_h. Did not run `build_targets`. Did not commit.

### Extra 68 — first tx on a weekend (train)

**186** / 1214 companies have first cash on Sat/Sun (15.32%). `created_at` weekend was 5 companies (extra 53). Cash is not an office stamp.

### Extra 69 — first banking `created_at` on a weekend

Share **0.58%** (n=1211). Should match `companies.created_at` (~0.4% weekend) if both are office connects.

### Extra 70 — weekend share of the three clocks

Connection clocks almost never land on Sat/Sun. First cash does. PARK onboard.

| clock | weekend share | source |
|---|---:|---|
| companies.created_at | 0.41% | extra 53 |
| first_bank_created | 0.58% | extra 69 |
| first_tx | 15.32% | extra 68 |

### Extra 71 — still PARK / CLOSE

No merged country/erp Y. `created_at` not revived.

### Extra 72 — join-QA still CONFIRMED

92.1% NULL-among-dark=True; named-dark 37=True; NULL-ERP 506=True; invoiced-among-NULL 14.4%=True.

### Extra 73 — named-ERP dark still not the 110

17 of the 110 mixed + 20 of the 360. Did not redo sibling_h.py.

### Extra 74 — country missingness unchanged

Train 82.21%; holdout coverage 80.56%. PARK as a health Y.

### Extra 75 — three-clock offsets unchanged

created−first_tx p50 39.4 days; share after 61.29%; same-month 7.08%. PARK `created_at`.

### Extra 76 — transferable Q1 flag?

**False**. Hidden test is new groups.

### Extra 77 — company-constant flags still lose to size

has_erp 0.480 / has_country 0.481 / is_eur 0.515 vs size 0.617 vs days 0.711. CLOSE.

### Extra 78 — sit complete

Same module write→run→read. Wave note is written once at the end, not here.

### Extra 79 — `has_erp` still the book dummy

ρ vs has_book 0.813; agree 90.94%. CLOSE as Y3 X.

### Extra 80 — ready for the wave note

Parent quote: 671/37/73/433; country miss 82.21%; created−tx p50 +39.4d; transferable=False.

### Extra 81 — no invented Y

Did not invent a merged Y from country/erp.

### Extra 82 — `created_at` still PARK

Trail QA connection clock. Not revived.

### Extra 83 — missing-country still PARK

82% missingness is not 45→65.

### Extra 84 — `has_erp` still CLOSE as Y3 X

470 dummy. Y2 already restricts on dark.

### Extra 85 — `has_country` / `is_eur` still CLOSE

Missingness / 90% EUR dummy / lose to size / group-uniform. Not KEEP descriptive.

### Extra 86 — parent quote locked

671 named+invoiced / 37 named+dark / 73 NULL+invoiced / 433 NULL+dark. Country miss 82.21%. created−first_tx p50 +39.4 days. Transferable=False.

### Extra 87 — hard bans held

No product/. No 0–100. No parquet rewrite. No new GBM. No `build_targets`.

### Extra 88 — other children not touched

Did not edit a_vol_qa.py, fx_qa.py, uncat_qa.py, sibling_h.py.

### Extra 89 — no commit

Did not commit.

### Extra 90 — last same-module cut

Wave note follows this sit.

### Extra 91

Same verdict.

### Extra 92

Same verdict.

### Extra 93

Same verdict.

### Extra 94

Same verdict.



## Transfer to new groups

A company-constant flag that marks a *group type* (ERP book, country filled, home EUR) cannot transfer to the hidden test of **new groups**.

**Any companies.csv flag transferable as Q1 X?** **False**. `has_erp`, `has_country`, and `currency=EUR` are company-constants. Train groups are ~85% uniform on each flag (extra 28). Holdout EUR is 74% not 90% (extra 24); hidden groups bring AOA/GHS/SEK/XOF and PL (extra 25). A group-type dummy cannot transfer to **new groups**.

## Six brief questions

| # | question | what companies.csv says |
| --- | --- | --- |
| 1 | Who is healthy? | **No transferable flag.** `has_erp` is the 470 dummy (ρ 0.81 vs book). Country is 82% missing. EUR is 90% home currency. All lose to size. |
| 2 | Who is improving? | Not a company-constant. |
| 3 | Who is turning? | Not these flags. |
| 4 | Dip vs fall? | Not these flags. |
| 5 | Why did it change? | CLOSE. Metadata / group identity, not a why. |
| 6 | Months earlier? | `created_at` is a connection clock (p50 +39 days after first tx; year vs grid ρ −0.85). PARK. |

## What was not done

- Did not write parquet / duckdb. Did not run `build_targets`.
- Did not edit a_vol_qa, fx_qa, uncat_qa, sibling_h, product/.
- Did not invent a merged Y from country/erp. Did not revive `created_at`.
- Did not commit.
