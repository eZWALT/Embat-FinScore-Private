# Factoring / confirming / LOC inventory (`f_has_*`)

Generated `2026-09-19T03:58:07+02:00` by agent `fd90198f`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent a merged Y. Do not redo schedule / util. Do **not** score these F flags vs Y9.

`f_has_factoring` / `f_has_confirming` / `f_has_loc` are as-of `created_at` inventory flags (rise-only like G). NORTH_STAR Q3: factoring/confirming are working-capital tools, not utilisation (Y10 PARK). Debt schedule QA: inventory is a connection panel, not origination.

## Headline

Ever-n train: factoring **18** / confirming **61** / LOC **186** of 1214 companies. CM share 0.7% / 3.3% / 11.7% (modal0 99.3% / 96.7% / 88.3%; feature-report 99.3% / 96.7% / 88.3%). Rise-only: fact=True conf=True loc=True (drops 0/0/0). Y3 singles 0.505 / 0.485 / 0.546 vs size 0.617 vs days 0.711 (night 0.711). Dark 470 vs invoiced 744: factoring 10/8, confirming 28/33, lineofcredit 74/112. 44-col: **drop all three flags**. Factoring companies are a never-recoverer set (0/18 ever Y3+; T3 peers 7.7%). Still CLOSE as X — 18-company dummy, AUROC 0.505.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Connection inventory, not a FICO label. |
| 2 | Who is improving? | Not these rise-only flags. |
| 3 | Who is turning? | new-WC month Q3 **CLOSE** (size-controlled ≥5pp gate). |
| 4 | Dip vs fall? | Overlap with Y4 / Y5 leftover is descriptive. |
| 5 | Why did it change? | Flags lose to size as Y3 X (Δ -0.112 / -0.132 / -0.070). Confirming AP-side is stacked WC, not the product. |
| 6 | Months earlier? | acf of a rise-only flag is persistence of *connection*, not lead. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| f_has_factoring as 44-col Y3 X | **CLOSE** | Y3 CV 0.505 vs size 0.617 (Δ -0.112); SIZE=False; group_dummy=False; ever_n=18 |
| f_has_confirming as 44-col Y3 X | **CLOSE** | Y3 CV 0.485 vs size 0.617 (Δ -0.132); SIZE=False; group_dummy=False; ever_n=61 |
| f_has_loc as 44-col Y3 X | **CLOSE** | Y3 CV 0.546 vs size 0.617 (Δ -0.070); SIZE=False; group_dummy=False; ever_n=186 |
| new factoring/confirming month as Q3 footnote | **CLOSE** | new_wc size-controlled Y3 Δ=-2.0% Y2 Δ=-0.1%; gate ≥5pp after size |
| f_new_facility as 44-col Y3 X | **CLOSE** | Y3 CV 0.515 vs size 0.617; connection clock (debt QA Q3 CAUTION) |
| has_* as health Y | **PARK** | created_at connection inventory, not a health label (Y10 already PARK util) |
| score flags vs Y9 | **CLOSE** | Y9 forbids F + a_fin_cost; not scored |
| has_factoring as Y5 AP why | **CLOSE as X; descriptive only** | size-weighted AP +18.9pp is 4 leftover companies, 3 in GROUP_0139 — group leftover, not a panel why |
| confirming as AP-side footnote (not X) | **CLOSE** | AP lift > AR lift on all confirming (raw +4.4 vs −2.8pp) is stacked WC — confirming-only AP 8.8% vs rest 8.3%; stacked 13.4%. Not a 44-col X. |
| extract liquidity / outstanding as X or Y | **PARK** | 2026-08 book; outstanding p50=0 on WC; Y10 already PARK util |

KEEP-as-44-col-X rule: oriented group-fold CV beats oriented size by ≥0.02 **and** not SIZE (|ρ| vs log1p(a_in3) ≥ 0.5) **and** not a group dummy (largest group ≥25% of ever-companies). Q3 footnote only if a new-factoring/confirming month moves Y3 or Y2 by ≥5pp after size terciles.

## 1. Prevalence (train; holdout count only)

Train 1,214 companies / 21,157 CM. Holdout 72 companies / 1,073 CM. `f_n_facilities` rises 555 / drops 0 (rise-only=True; debt QA 555/0). Ever WC (fact∪conf) **65**. Feature-report modal confirm fact=True conf=True.

| type | raw prod | raw co | after snap | ever n | ever % co | CM on | CM share | modal 0 | last-mo on | rises | drops | rise-only | acf1 | hold ever |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factoring | 23 | 18 | 0 | 18 | 1.5% | 151 | 0.7% | 99.3% | 18 | 17 | 0 | YES | 0.778 | 1 |
| confirming | 205 | 61 | 2 | 61 | 5.0% | 692 | 3.3% | 96.7% | 61 | 50 | 0 | YES | 0.864 | 9 |
| lineofcredit | 500 | 189 | 4 | 186 | 15.3% | 2482 | 11.7% | 88.3% | 186 | 135 | 0 | YES | 0.691 | 17 |


## 2. Size ρ vs log1p(a_in3)

SIZE if |ρ| ≥ 0.5. Company-month flag vs log1p(a_in3); ever-flag vs last-month size.

| flag | ρ log1p(a_in3) | ρ a_in3 | ρ ever vs size | SIZE? |
| --- | --- | --- | --- | --- |
| f_has_factoring | 0.083 | 0.083 | 0.099 |  |
| f_has_confirming | 0.156 | 0.156 | 0.178 |  |
| f_has_loc | 0.255 | 0.255 | 0.253 |  |


## 3. Single-feature train group-fold AUROC (Y3 / Y2 only)

Y3 stressed n=5,648 base 7.1%; Y2 n=17,356 base 7.3%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711; MATCH). Y2 days replica 0.571 (night 0.540; this is oriented group-fold, not a days mismatch). Leak Y3 vs B: ok. Not scored vs Y9.

| y | feature | CV | sd | sign | train | n_lab | n_pos | SIZE | low_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Y3 | f_has_factoring | 0.505 | 0.004 | -1 | 0.505 | 5,648 | 402 |  |  |
| Y3 | f_has_confirming | 0.485 | 0.020 | -1 | 0.502 | 5,648 | 402 |  |  |
| Y3 | f_has_loc | 0.546 | 0.028 | -1 | 0.546 | 5,648 | 402 |  |  |
| Y3 | f_new_facility_gt0 | 0.515 | 0.005 | -1 | 0.516 | 5,648 | 402 |  |  |
| Y3 | new_factoring_gt0 | 0.501 | 0.001 | -1 | 0.501 | 5,648 | 402 |  |  |
| Y3 | new_confirming_gt0 | 0.499 | 0.002 | -1 | 0.500 | 5,648 | 402 |  |  |
| Y3 | new_wc_gt0 | 0.501 | 0.003 | -1 | 0.501 | 5,648 | 402 |  |  |
| Y3 | log1p_a_in3 | 0.617 | 0.061 | -1 | 0.620 | 5,528 | 391 | YES |  |
| Y3 | c_n_days_with_tx | 0.711 | 0.031 | -1 | 0.723 | 5,648 | 402 | YES |  |
| Y2 | f_has_factoring | 0.496 | 0.004 | 1 | 0.501 | 17,356 | 1,271 |  |  |
| Y2 | f_has_confirming | 0.504 | 0.018 | 1 | 0.511 | 17,356 | 1,271 |  |  |
| Y2 | f_has_loc | 0.523 | 0.024 | 1 | 0.524 | 17,356 | 1,271 |  |  |
| Y2 | f_new_facility_gt0 | 0.501 | 0.017 | -1 | 0.505 | 17,356 | 1,271 |  |  |
| Y2 | new_factoring_gt0 | 0.499 | 0.001 | 1 | 0.500 | 17,356 | 1,271 |  |  |
| Y2 | new_confirming_gt0 | 0.501 | 0.001 | -1 | 0.501 | 17,356 | 1,271 |  |  |
| Y2 | new_wc_gt0 | 0.500 | 0.002 | -1 | 0.501 | 17,356 | 1,271 |  |  |
| Y2 | log1p_a_in3 | 0.552 | 0.046 | 1 | 0.540 | 14,968 | 1,044 | YES |  |
| Y2 | c_n_days_with_tx | 0.571 | 0.046 | 1 | 0.577 | 17,356 | 1,271 | YES |  |


`f_has_loc` 0.546 is leftover inverse-size (ρ 0.255, Y3 size 0.617) — Δ vs size is −0.070. Factoring 0.505 is a never-recoverer hole that Mann–Whitney cannot see because the flag is rare. Neither is a 44-col X.

Y3 KEEP screen vs size / days:

| feature | CV | size | days | Δ size | SIZE | KEEP 44 |
| --- | --- | --- | --- | --- | --- | --- |
| f_has_factoring | 0.505 | 0.617 | 0.711 | -0.112 |  |  |
| f_has_confirming | 0.485 | 0.617 | 0.711 | -0.132 |  |  |
| f_has_loc | 0.546 | 0.617 | 0.711 | -0.070 |  |  |
| f_new_facility_gt0 | 0.515 | 0.617 | 0.711 | -0.102 |  |  |
| new_factoring_gt0 | 0.501 | 0.617 | 0.711 | -0.116 |  |  |
| new_confirming_gt0 | 0.499 | 0.617 | 0.711 | -0.118 |  |  |
| new_wc_gt0 | 0.501 | 0.617 | 0.711 | -0.116 |  |  |


## 4. Q3 — new factoring / confirming this month

`created_at` inside the month, typed. `f_new_facility` is any debt type (debt QA 573 CM / Q3 CAUTION). New WC (fact∪conf) 80 CM; first-ever WC birth share 68.8%. KEEP footnote only if size-tercile-weighted |Δ| ≥ 5pp on Y3 or Y2.

| flag | n_cm | n_co | share | Y3 on | Y3 off | Y3 pp | Y3 pp|size | Y2 on | Y2 off | Y2 pp|size | Q3 KEEP |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| new_factoring | 20 | 18 | 0.1% | 0.0% | 7.1% | -7.1% | -5.1% | 12.5% | 7.3% | 13.3% |  |
| new_confirming | 69 | 53 | 0.3% | 6.9% | 7.1% | -0.2% | 1.9% | 2.1% | 7.3% | -2.8% |  |
| new_wc | 80 | 61 | 0.4% | 5.4% | 7.1% | -1.7% | -2.0% | 5.2% | 7.3% | -0.1% |  |
| new_loc | 217 | 159 | 1.0% | 0.0% | 7.2% | -7.2% | -7.1% | 6.2% | 7.3% | -4.9% |  |
| f_new_facility | 573 | 309 | 2.7% | 1.8% | 7.3% | -5.5% | -5.8% | 4.8% | 7.4% | -3.9% |  |


Within size terciles (new-WC and typed new):

| flag | T | y | n on | rate on | rate off | pp |
| --- | --- | --- | --- | --- | --- | --- |
| new_factoring | T1 | Y3 | 0 | — | 14.7% | — |
| new_factoring | T2 | Y3 | 3 | 0.0% | 4.8% | -4.8% |
| new_factoring | T3 | Y3 | 8 | 0.0% | 5.3% | -5.3% |
| new_factoring | T1 | Y2 | 0 | — | 5.7% | — |
| new_factoring | T2 | Y2 | 3 | 33.3% | 7.2% | 26.1% |
| new_factoring | T3 | Y2 | 12 | 8.3% | 8.0% | 0.3% |
| new_confirming | T1 | Y3 | 1 | 0.0% | 14.7% | -14.7% |
| new_confirming | T2 | Y3 | 4 | 25.0% | 4.8% | 20.2% |
| new_confirming | T3 | Y3 | 23 | 0.0% | 5.3% | -5.3% |
| new_confirming | T1 | Y2 | 1 | 0.0% | 5.7% | -5.7% |
| new_confirming | T2 | Y2 | 8 | 12.5% | 7.2% | 5.3% |
| new_confirming | T3 | Y2 | 37 | 0.0% | 8.0% | -8.0% |
| new_wc | T1 | Y3 | 1 | 0.0% | 14.7% | -14.7% |
| new_wc | T2 | Y3 | 7 | 14.3% | 4.8% | 9.5% |
| new_wc | T3 | Y3 | 28 | 0.0% | 5.3% | -5.3% |
| new_wc | T1 | Y2 | 1 | 0.0% | 5.7% | -5.7% |
| new_wc | T2 | Y2 | 11 | 18.2% | 7.2% | 11.0% |
| new_wc | T3 | Y2 | 44 | 2.3% | 8.0% | -5.8% |


## 5. Overlap with Y4 / Y5 leftover / ever-FX 228

Ever-FX companies **228** (confirm 228=True). Y4-positive companies 117. Y5 AP leftover (own-p20 io × own-p80 supp HHI neither cell) 222 CM / 122 companies (y5_why quoted 222 leftover AP positives — company-months). Descriptive only. Confirming vs Y5 is not an X score.

| set | n_co | ∩ Y4 | ∩ Y5 AP | ∩ Y5 leftover | ∩ FX 228 |
| --- | --- | --- | --- | --- | --- |
| factoring | 18 | 6 (33.3%) | 6 (33.3%) | 5 (27.8%) | 1 (5.6%) |
| confirming | 61 | 11 (18.0%) | 18 (29.5%) | 13 (21.3%) | 6 (9.8%) |
| lineofcredit | 186 | 43 (23.1%) | 38 (20.4%) | 20 (10.8%) | 29 (15.6%) |
| wc | 65 | 13 (20.0%) | 19 (29.2%) | 13 (20.0%) | 6 (9.2%) |


| flag | leftover CM ∧ flag | share of leftover CM | Y4 CM ∧ flag |
| --- | --- | --- | --- |
| f_has_factoring | 11 | 5.0% | 6 |
| f_has_confirming | 22 | 9.9% | 17 |
| f_has_loc | 33 | 14.9% | 84 |


## 6. Dark 470 vs invoiced 744

confirm 470/744 = True. 360 all-dark / 110 mixed-group dark. Access ≠ ERP (banking G): these products can exist on dark companies.

| type | dark ever | invoiced ever | dark CM | invoiced CM |
| --- | --- | --- | --- | --- |
| factoring | 10 (2.1%) | 8 (1.1%) | 81 | 70 |
| confirming | 28 (6.0%) | 33 (4.4%) | 255 | 437 |
| lineofcredit | 74 (15.7%) | 112 (15.1%) | 898 | 1584 |


| mix | n_co | factoring | confirming | LOC |
| --- | --- | --- | --- | --- |
| all_dark | 360 | 10 | 24 | 63 |
| mixed | 340 | 4 | 12 | 48 |
| all_invoiced | 514 | 4 | 25 | 75 |


## 7. bank_name of WC products (train, not post-snapshot)

Dictionary: `bank_name` is the connected bank. `Other (customer-defined)` / In-house are not a bank feed.

**factoring**: 23 products / 18 companies / 9 bank names. Empresas-labelled 20; in-house or Other 0.

| bank | n | n_co |
| --- | --- | --- |
| Banco Santander Empresas | 6 | 5 |
| Bankinter Empresas | 5 | 3 |
| Banco Sabadell Empresas | 4 | 4 |
| Banco Santander | 3 | 3 |
| Abanca Empresas | 1 | 1 |
| Banco Sabadell T. sec - CAL Empresas | 1 | 1 |
| BBVA Net Cash Empresas | 1 | 1 |
| Caixabank Empresas | 1 | 1 |


**confirming**: 203 products / 61 companies / 17 bank names. Empresas-labelled 146; in-house or Other 0.

| bank | n | n_co |
| --- | --- | --- |
| Banco Santander Empresas | 54 | 21 |
| Caixabank Empresas | 37 | 19 |
| Banco Santander | 33 | 7 |
| Bankinter Empresas | 22 | 14 |
| BBVA Net Cash Empresas | 14 | 12 |
| Bankinter | 7 | 5 |
| Caixabank | 6 | 3 |
| RuralVia Empresas | 6 | 5 |


**lineofcredit**: 496 products / 186 companies / 32 bank names. Empresas-labelled 340; in-house or Other 31.

| bank | n | n_co |
| --- | --- | --- |
| Banco Santander Empresas | 112 | 59 |
| BBVA Net Cash Empresas | 50 | 41 |
| Bankinter Empresas | 45 | 32 |
| Banco Sabadell Empresas | 45 | 35 |
| Caixabank Empresas | 44 | 36 |
| BBVA | 27 | 14 |
| In-house bank | 26 | 12 |
| Caixabank | 17 | 14 |


## 8. Confirming as AP-side (Y5 rates; never E as X)

Leak screen confirming vs Y5+E: ok (we still do not *score* it as X). Confirming AP-vs-AR lift points AP-side: True (AP pp 4.4%; factoring AP pp 19.6%). `leftover on` is leftover share among AP-positives with a defined cash×HHI cell (not a panel rate). Family E columns are not used as X. Size-controlled AP is §11.

| type | Y5 AP on | Y5 AP off | AP pp | Y5 AR on | Y5 AR off | AR pp | leftover on | leftover pp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| factoring | 27.9% | 8.4% | 19.6% | 4.9% | 7.2% | -2.3% | 91.7% | 27.5% |
| confirming | 12.7% | 8.3% | 4.4% | 4.5% | 7.3% | -2.8% | 81.5% | 17.8% |
| lineofcredit | 10.2% | 8.2% | 2.0% | 8.4% | 6.9% | 1.5% | 56.9% | -9.9% |


| type | ever co | ever ∩ leftover | share | rest share |
| --- | --- | --- | --- | --- |
| factoring | 18 | 5 | 27.8% | 9.8% |
| confirming | 61 | 13 | 21.3% | 9.5% |
| lineofcredit | 186 | 20 | 10.8% | 9.9% |


## 9. Group dummy

A rare flag that lives in one or two groups is a group dummy, not a health X.

| type | ever co | n groups | max group share | max group | dummy |
| --- | --- | --- | --- | --- | --- |
| factoring | 18 | 14 | 22.2% | GROUP_0139 |  |
| confirming | 61 | 36 | 6.6% | GROUP_0095 |  |
| lineofcredit | 186 | 104 | 3.8% | GROUP_0246 |  |


## Plot

`factoring_has_vs_size.png`

## 10. Co-occurrence + size-tercile has_* rates

Last-month train: fact∩conf 14, fact∩loc 12, conf∩loc 44, all three 11. Only-fact 3, only-conf 14, only-loc 141. Most factoring sits with another facility — not a standalone type.

| flag | y | T | n on | rate on | rate off | pp |
| --- | --- | --- | --- | --- | --- | --- |
| factoring | Y3 | T1 | 0 | — | 14.7% | — |
| factoring | Y3 | T2 | 15 | 0.0% | 4.8% | -4.8% |
| factoring | Y3 | T3 | 40 | 0.0% | 5.4% | -5.4% |
| confirming | Y3 | T1 | 17 | 5.9% | 14.8% | -9.0% |
| confirming | Y3 | T2 | 68 | 4.4% | 4.8% | -0.4% |
| confirming | Y3 | T3 | 166 | 7.2% | 5.1% | 2.1% |
| loc | Y3 | T1 | 57 | 12.3% | 14.8% | -2.6% |
| loc | Y3 | T2 | 246 | 2.4% | 5.1% | -2.7% |
| loc | Y3 | T3 | 671 | 3.4% | 6.0% | -2.6% |


Y2 same terciles (has_* months):

| flag | T | n on | rate on | rate off | pp |
| --- | --- | --- | --- | --- | --- |
| factoring | T1 | 2 | 0.0% | 5.7% | -5.7% |
| factoring | T2 | 21 | 9.5% | 7.2% | 2.3% |
| factoring | T3 | 78 | 11.5% | 7.9% | 3.6% |
| confirming | T1 | 39 | 12.8% | 5.7% | 7.2% |
| confirming | T2 | 105 | 7.6% | 7.2% | 0.4% |
| confirming | T3 | 355 | 12.7% | 7.6% | 5.0% |
| loc | T1 | 159 | 11.3% | 5.5% | 5.8% |
| loc | T2 | 516 | 7.2% | 7.2% | -0.1% |
| loc | T3 | 1140 | 10.8% | 7.2% | 3.6% |


## 11. Y5 AP after size (never E as X)

Raw factoring AP +19.6pp looked like leftover. Size-weighted AP Δ: factoring 18.9% (survives 5pp=True); confirming 3.7% (survives=False). Still not an X — Y5 forbids E; this is the AP-side footnote only.

| type | T | n AP-lab on | AP on | AP off | pp | leftover among AP+ |
| --- | --- | --- | --- | --- | --- | --- |
| factoring | T1 | 0 | — | 6.8% | — | — |
| factoring | T2 | 20 | 20.0% | 8.1% | 11.9% | 100.0% |
| factoring | T3 | 23 | 34.8% | 9.8% | 24.9% | 87.5% |
| factoring | size-w | 43 | — | — | 18.9% | — |
| confirming | T1 | 18 | 11.1% | 6.8% | 4.3% | 0.0% |
| confirming | T2 | 76 | 13.2% | 8.0% | 5.2% | 80.0% |
| confirming | T3 | 134 | 12.7% | 10.0% | 2.7% | 93.3% |
| confirming | size-w | 228 | — | — | 3.7% | — |
| lineofcredit | T1 | 39 | 5.1% | 6.9% | -1.8% | 0.0% |
| lineofcredit | T2 | 223 | 7.6% | 8.3% | -0.6% | 41.2% |
| lineofcredit | T3 | 441 | 12.0% | 9.5% | 2.5% | 65.0% |
| lineofcredit | size-w | 703 | — | — | 1.3% | — |


## 12. Dark WC — factoring without an invoice book

Factoring on dark companies is a connected product, not an invoice-finance trail. Dark no-WC n=439, median a_in3 267905. Dark factoring is not smaller cash than invoiced factoring.

| type | dark n | invoiced n | dark med in3 | inv med in3 | dark med days | inv med days |
| --- | --- | --- | --- | --- | --- | --- |
| factoring | 10 | 8 | 3720906 | 2179761 | 24.0 | 17.0 |
| confirming | 28 | 33 | 5263278 | 1250581 | 23.0 | 21.0 |
| lineofcredit | 74 | 112 | 2273281 | 1121124 | 22.5 | 20.5 |


## 13. Confirming Empresas vs retail bank_name

Companies with both Empresas and retail confirming labels: 6. Y5 AP on connected months (descriptive).

| slice | n_co | CM on | Y5 AP | n AP-lab | Y3 | n Y3 |
| --- | --- | --- | --- | --- | --- | --- |
| empresas_only | 45 | 417 | 13.8% | 123 | 8.2% | 147 |
| retail_only | 10 | 200 | 13.2% | 91 | 2.9% | 105 |
| any_empresas | 51 | 492 | 12.4% | 137 | 9.3% | 150 |


## 14. GROUP_0139 (22% of ever-factoring)

Group has 5 train companies last-month; factoring 4, confirming 4, LOC 1. Median a_in3 530546. Factoring names: COMP_0065, COMP_0686, COMP_1171, COMP_1191. Under the 25% dummy line — still not a KEEP.

## 16. Zero Y3 recoveries on factoring months

Y3 positives 402 / labeled 5,648. A rare always-off flag among recoverers still has AUROC ≈ 0.50 (Mann–Whitney lift is n_flag / 2 n_neg). Do not KEEP a 'never recovers' story.

| type | Y3 lab ∧ flag | Y3 pos ∧ flag | share of pos | rate on |
| --- | --- | --- | --- | --- |
| factoring | 55 | 0 | 0.0% | 0.0% |
| confirming | 255 | 17 | 4.2% | 6.7% |
| lineofcredit | 987 | 36 | 9.0% | 3.6% |


## 17. Connection vs first cash-trail month

Lag = created_at month − first panel month. Positive = connected after the cash trail started.

| type | n prod | p50 lag mo | after trail | before trail |
| --- | --- | --- | --- | --- |
| factoring | 23 | 7.0 | 100.0% | 0.0% |
| confirming | 203 | 5.0 | 79.3% | 20.7% |
| lineofcredit | 496 | 2.0 | 82.9% | 17.1% |


## 18. Extract granted / outstanding (last-month book, not a path)

2026-08 extract amounts. Not a 2024 book. Do not revive util.
Factoring / confirming outstanding p50 = 0: most WC rows are a connected limit, not a drawn book. Do not treat `f_has_*` as utilisation (Y10 already PARK).

| type | n | granted p50 | out p50 | |granted|>1 | |out|>1 |
| --- | --- | --- | --- | --- | --- |
| factoring | 23 | 800000 | 0 | 91.3% | 43.5% |
| confirming | 203 | 1010000 | 0 | 73.9% | 35.0% |
| lineofcredit | 496 | 300000 | 59594 | 87.3% | 76.8% |


## 19. 2025-10 factoring connection wave

7 train companies connected a factoring product in 2025-10 (3 dark). Median a_in3 2520565. Y3 0.0% n=7; Y2 28.6% n=7. Names: COMP_0065, COMP_0686, COMP_0806, COMP_0877, COMP_1171, COMP_1191, COMP_1216. A platform connection wave, not Q3 turning.

## 20. Who are the factoring AP leftovers

Factoring ∧ Y5 AP: 12 CM / 5 companies. Of those, leftover cell 11 CM / 4 companies (COMP_0686, COMP_0919, COMP_1171, COMP_1191). Thin — do not invent a Y5-factoring label.

## 22. Factoring AP leftover is a group, not a law

3/4 leftover factoring companies sit in GROUP_0139 (share 75.0%). Outside: COMP_0919. Group leftover — CLOSE as a Y5 why.

## 23. Siblings sharing WC

Groups with ≥2 members on the product. A company tag, not a group book (same shape as schedule QA).

| type | ever last-mo | n groups | groups with ≥2 | share groups multi |
| --- | --- | --- | --- | --- |
| factoring | 18 | 14 | 2 | 14.3% |
| confirming | 61 | 36 | 17 | 47.2% |
| lineofcredit | 186 | 104 | 41 | 39.4% |


## 24. Product currency

**factoring** (2 currencies): EUR 22, USD 1

**confirming** (2 currencies): EUR 199, USD 4

**lineofcredit** (2 currencies): EUR 492, USD 4

## 25. Store vs live last-month inventory

As-of 2026-08-01. Store flags match live created_at inventory. No parquet rewrite.

| type | store | live | agree | only store | only live |
| --- | --- | --- | --- | --- | --- |
| factoring | 18 | 18 | 18 | 0 | 0 |
| confirming | 61 | 61 | 61 | 0 | 0 |
| lineofcredit | 186 | 186 | 186 | 0 | 0 |


## 26. ICC and connected-only Y3

Rise-only flags are BETWEEN (high ICC): connection is a company trait. On months with `f_n_facilities>0` (4,854 CM), `f_has_loc` CV 0.440 vs size 0.599 (Δ -0.159). Still CLOSE.

| type | ICC | acf1 | n cos |
| --- | --- | --- | --- |
| factoring | 0.969 | 0.778 | 1214 |
| confirming | 0.983 | 0.864 | 1214 |
| lineofcredit | 0.989 | 0.691 | 1214 |


| feature | CV connected | n_lab | n_pos |
| --- | --- | --- | --- |
| f_has_factoring | 0.515 | 1,876 | 71 |
| f_has_confirming | 0.592 | 1,876 | 71 |
| f_has_loc | 0.440 | 1,876 | 71 |
| log1p_a_in3 | 0.599 | 1,856 | 70 |
| c_n_days_with_tx | 0.732 | 1,876 | 71 |


## 27. Y4 rates after size (never F as X)

Y4 forbids family F — leak screen ok=False (expected fail). Rates only. Factoring's raw 33% ever-Y4 overlap is the company set, not a month why.

| type | T | n Y4-lab on | Y4 on | Y4 off | pp |
| --- | --- | --- | --- | --- | --- |
| factoring | T1 | 0 | — | 12.2% | — |
| factoring | T2 | 4 | 0.0% | 16.0% | -16.0% |
| factoring | T3 | 35 | 17.1% | 12.8% | 4.3% |
| confirming | T1 | 6 | 0.0% | 12.4% | -12.4% |
| confirming | T2 | 38 | 5.3% | 16.4% | -11.2% |
| confirming | T3 | 138 | 10.9% | 13.3% | -2.4% |
| lineofcredit | T1 | 48 | 8.3% | 12.7% | -4.3% |
| lineofcredit | T2 | 184 | 9.8% | 17.6% | -7.8% |
| lineofcredit | T3 | 504 | 12.3% | 13.6% | -1.3% |


## 28. Confirming AP without GROUP_0139

Drop GROUP_0139 CM (102). Confirming AP 10.9% vs 8.3% (Δ 2.5%, n=175); AR Δ -1.9%. Factoring AP leftover collapses: 18.2% vs 8.4% (Δ 9.8%, n=11). Confirming stays mildly AP-side without the group; factoring's +19pp was the group.

## 29. Same companies before vs after first connection

If Q3 turning were real, Y3/Y2 would move after the first typed `created_at`. Same companies, split on first on-flag month.

| type | n_co | Y3 before | Y3 after | Y3 pp | Y2 before | Y2 after | Y2 pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| factoring | 18 | 0.0% (n=59) | 0.0% (n=55) | 0.0% | 20.3% (n=123) | 10.7% (n=103) | -9.6% |
| confirming | 61 | 0.0% (n=153) | 6.7% (n=255) | 6.7% | 6.1% (n=343) | 12.1% (n=522) | 5.9% |
| lineofcredit | 186 | 0.9% (n=228) | 3.6% (n=987) | 2.8% | 5.3% (n=723) | 10.2% (n=1926) | 5.0% |


## 30. Holdout coverage only

Holdout 72 companies / 1,073 CM. Not used in any rate, ρ, or AUROC.

| type | ever | CM on |
| --- | --- | --- |
| factoring | 1 | 11 |
| confirming | 9 | 97 |
| lineofcredit | 17 | 154 |


## 31. Ever-FX 228 ∩ WC

Ever-FX 228. Factoring almost never overlaps FX (1/18). Not an exporter-factoring story.

| type | ever WC | ∩ FX | share | both med log in3 | WC-only med | FX med |
| --- | --- | --- | --- | --- | --- | --- |
| factoring | 18 | 1 | 5.6% | 16.503 | 14.716 | 13.149 |
| confirming | 61 | 6 | 9.8% | 16.518 | 14.706 | 13.149 |
| lineofcredit | 186 | 29 | 15.6% | 14.512 | 14.236 | 13.149 |


## 32. Factoring Y2 pre/post after size and year

Raw Y2 20.3%→10.7% after first factoring connection looks like relief. Within size terciles and calendar years it is not a stable ≥5pp Q3 footnote (new-month gate already failed). Mean-reversion after a high-Y2 spell, not turning.

| slice | n before | n after | Y2 before | Y2 after | pp |
| --- | --- | --- | --- | --- | --- |
| T1 | 35 | 29 | 37.1% | 6.9% | -30.2% |
| T2 | 33 | 33 | 24.2% | 27.3% | 3.0% |
| T3 | 21 | 39 | 4.8% | 0.0% | -4.8% |
| 2024 | 18 | 2 | 22.2% | 0.0% | -22.2% |
| 2025 | 82 | 37 | 20.7% | 13.5% | -7.2% |
| 2026 | 23 | 64 | 17.4% | 9.4% | -8.0% |


Train Y2 base by year (all companies): 2024 9.9%, 2025 7.6%, 2026 6.1%.

## 33. Never-recoverer company set

Train companies with any Y3=1: 174. Non-factoring ever-recover share 14.5%. Factoring 0/18 is a rare company hole, not a month signal — AUROC stays 0.505.

| type | ever | ever Y3+ | share |
| --- | --- | --- | --- |
| factoring | 18 | 0 | 0.0% |
| confirming | 61 | 6 | 9.8% |
| lineofcredit | 186 | 15 | 8.1% |


## 34. Months on book

WC companies are not short-trail names. Connection is late on a full book (pass 17), not a thin onboarding stub.

| slice | n_co | p50 months | p50 med in3 | p50 med days |
| --- | --- | --- | --- | --- |
| factoring | 18 | 14.5 | 3251621 | 22.5 |
| confirming | 61 | 20.0 | 2358635 | 22.5 |
| loc | 186 | 20.0 | 1742992 | 22.0 |
| rest | 1008 | 20.0 | 209939 | 13.0 |


## 35. COMP_0919 — leftover factoring outside GROUP_0139

Group GROUP_0230 mix=all_invoiced; invoiced=True. 12 CM, factoring-on 10, confirming-on 10. Y5 AP 1 leftover 1; Y3+ 0; Y2+ 0. Last a_in3 14690578. One name does not make a Y5 law.

## 36. How many WC products per company

| type | n_co | p50 n prod | max | share ≥2 |
| --- | --- | --- | --- | --- |
| factoring | 18 | 1.0 | 3 | 16.7% |
| confirming | 61 | 2.0 | 16 | 59.0% |
| lineofcredit | 186 | 2.0 | 15 | 57.0% |


## 37. Product `created_at` vs `companies.created_at`

Read-only `companies`. Do not edit `companies_qa.py`. Positive lag = product connected after the company row.

| type | n | p50 days after company | share after company |
| --- | --- | --- | --- |
| factoring | 23 | 64 | 100.0% |
| confirming | 203 | 11 | 100.0% |
| lineofcredit | 496 | 14 | 100.0% |


## 38. Y3 label coverage — unstressed vs never-recover

If factoring companies simply lacked a stressed window, Y3 would be unlabeled. They are labeled as often as the rest and still have 0 recoveries.

| slice | CM | Y3 labeled | label share | Y3+ | rate | labeled |
| --- | --- | --- | --- | --- | --- |
| factoring | 280 | 114 | 40.7% | 0 | 0.0% |
| confirming | 1,048 | 408 | 38.9% | 17 | 4.2% |
| lineofcredit | 3,223 | 1,215 | 37.7% | 38 | 3.1% |
| rest | 17,569 | 4,309 | 24.5% | 358 | 8.3% |


## 39. Product labels

**factoring** (3 distinct labels): FACTORING_01 ×18, FACTORING_02 ×3, FACTORING_03 ×2

**confirming** (16 distinct labels): CONFIRMING_01 ×60, CONFIRMING_02 ×36, CONFIRMING_03 ×29, CONFIRMING_04 ×18, CONFIRMING_05 ×15, CONFIRMING_06 ×11

**lineofcredit** (15 distinct labels): LINEOFCREDIT_01 ×186, LINEOFCREDIT_02 ×106, LINEOFCREDIT_03 ×68, LINEOFCREDIT_04 ×47, LINEOFCREDIT_05 ×28, LINEOFCREDIT_06 ×16

## 40. Size-matched ever-Y3+ (is 0/18 just large firms?)

Last-month size tercile. Factoring companies vs other companies in the same tercile. If T3 rest still recovers, 0/18 is not leftover size.

| T | n factoring | fact ever Y3+ | n rest | rest ever Y3+ |
| --- | --- | --- | --- | --- |
| T1 | 2 | 0.0% | 403 | 26.8% |
| T2 | 3 | 0.0% | 401 | 9.0% |
| T3 | 13 | 0.0% | 392 | 7.7% |


## 41. In-house LOC vs bank LOC

31 in-house/Other LOC products (pass 7). Descriptive Y3/Y2 on connected months.

| slice | n_co | Y3 | n Y3 | Y2 | n Y2 |
| --- | --- | --- | --- | --- | --- |
| inhouse_or_other | 15 | — | 0 | 0.0% | 10 |
| bank_loc | 171 | 3.6% | 987 | 10.3% | 1916 |


## 42. Extract liquidity (not utilisation)

Snapshot liquidity. PARK with util. Do not build Y10 cousin.

| type | n | defined | p50 | >1 | exactly 0 |
| --- | --- | --- | --- | --- | --- |
| factoring | 23 | 87.0% | 746243 | 87.0% | 13.0% |
| confirming | 203 | 75.4% | 476690 | 73.4% | 24.6% |
| lineofcredit | 496 | 82.3% | 151291 | 81.2% | 17.9% |


## 43. `service` codes

Bank service, not a health X. `custom` would be customer-defined (G).

**factoring**: 8 services, custom=0. santander_emp ×9, bankinter_emp ×5, sabadell_emp ×4, abanca_emp ×1, caixa_emp ×1

**confirming**: 14 services, custom=0. santander_emp ×87, caixa_emp ×42, bankinter_emp ×29, bbva_emp ×18, ruralvia_emp ×6

**lineofcredit**: 25 services, custom=5. santander_emp ×126, bbva_emp ×77, bankinter_emp ×61, caixa_emp ×59, sabadell_emp ×55

## 44. Observed `f_ds_r` / `f_fc_r` on WC-flag months

Flow columns stay KEEP. If flag months have no observed service/cost, the inventory is a connected limit, not a used book.

| type | col | n on | p50 on | share>0 on | p50 off | share>0 off |
| --- | --- | --- | --- | --- | --- | --- |
| factoring | f_ds_r | 149 | 0.028 | 84.8% | 0.000 | 25.1% |
| factoring | f_fc_r | 149 | 0.002 | 94.0% | 0.000 | 62.8% |
| confirming | f_ds_r | 669 | 0.018 | 69.5% | 0.000 | 24.0% |
| confirming | f_fc_r | 669 | 0.002 | 92.8% | 0.000 | 62.1% |
| lineofcredit | f_ds_r | 2370 | 0.024 | 64.9% | 0.000 | 20.2% |
| lineofcredit | f_fc_r | 2370 | 0.002 | 91.5% | 0.000 | 59.3% |


## 45. Factoring-only vs stacked facilities

Last-month factoring-only companies: 3 (COMP_0511, COMP_0769, COMP_1025). ds>0 on their factoring-on months 95.0% (n=20). Stacked fact+conf/loc 15: ds>0 83.2% (n=131). If only-fact still has ds, the flow is not 'other F flags'.
All three only-fact names also have `loan` rows — observed `f_ds_r` is the loan book, not the factoring flag. Flow KEEP is unchanged.

Other `debt_products` types on those names (loan/leasing/…):

| company | type | n |
| --- | --- | --- |
| COMP_1025 | loan | 1 |
| COMP_0769 | loan | 3 |
| COMP_1025 | factoring | 1 |
| COMP_0511 | loan | 6 |
| COMP_0511 | factoring | 3 |
| COMP_0769 | factoring | 1 |
| COMP_1025 | guarantee | 1 |


## 46. Confirming-only (no factoring, no LOC) AP-side

Confirming-only last-month companies: 14. On-months AP 8.8% (n=34) vs off 37.5%; AR 5.6% vs 25.0%. Y3 on 10.0% n=60. If AP lift survives without stacked LOC/factoring, the Q5 footnote is confirming itself. Within-company off months are pre-connection and small-n — do not read the off rate as a panel.

## 47. Confirming-only vs stacked vs rest (Y5 AP)

Confirming-only AP 8.8% vs stacked 13.4% vs rest 8.3%. only_is_ap=False (need ≥2pp over rest). The all-confirming +4.4pp AP story is stacked WC users, not confirming as a tool.

| slice | n_co | n_cm | Y5 AP | AP n | Y5 AR |
| --- | --- | --- | --- | --- | --- |
| confirming_only | 14 | 182 | 8.8% | 34 | 5.6% |
| confirming_stacked | 47 | 510 | 13.4% | 194 | 4.4% |
| no_confirming | 1153 | 20109 | 8.3% | 4611 | 7.2% |


## 48. Stacked confirming AP after size

Size-weighted AP pp: confirming-only -0.3%, stacked 4.3%. If stacked also dies after size, Q5 is size/stacking, not the product.

| slice | tercile | n_on | AP on | AP off | pp |
| --- | --- | --- | --- | --- | --- |
| confirming_only | T1 | 4 | 25.0% | 6.8% | 18.2% |
| confirming_only | T2 | 12 | 0.0% | 8.2% | -8.2% |
| confirming_only | T3 | 18 | 11.1% | 10.2% | 0.9% |
| confirming_only | size-w | 34 | — | — | -0.3% |
| confirming_stacked | T1 | 14 | 7.1% | 6.8% | 0.3% |
| confirming_stacked | T2 | 64 | 15.6% | 7.9% | 7.7% |
| confirming_stacked | T3 | 116 | 12.9% | 10.0% | 2.9% |
| confirming_stacked | size-w | 194 | — | — | 4.3% |


## 49. Unused limit share (PARK — not util)

Last-month extract `(granted−outstanding)/granted` given granted>1. High unused + low drawn = connected limit. PARK with Y10. Do not add a utilisation cousin.

| type | n | n granted>1 | unused p50 | unused≥90% | drawn |out|>1 |
| --- | --- | --- | --- | --- | --- |
| factoring | 23 | 21 | 100.0% | 52.4% | 43.5% |
| confirming | 203 | 150 | 100.0% | 65.3% | 35.0% |
| lineofcredit | 496 | 433 | 54.8% | 37.6% | 76.8% |


## 50. LOC-only (no factoring, no confirming)

LOC-only last-month companies: 141. On-months Y3 3.3% n=760; Y2 10.6%. Terciles are within LOC-only months (not the panel). Inverse-size leftover, not a WC-tool dummy.

| tercile | n_on | Y3 | Y2 |
| --- | --- | --- | --- |
| T1 | 603 | 5.4% | 9.2% |
| T2 | 593 | 1.5% | 13.7% |
| T3 | 650 | 3.6% | 7.0% |


## 51. Confirming banks: only vs stacked

Same Empresas names on both slices would mean the AP lift is stacking, not a different confirming product.

**confirming_only**: 32 products / 14 companies / 8 banks. Empresas-labelled 29.

| bank | n | n_co |
| --- | --- | --- |
| Caixabank Empresas | 11 | 5 |
| Bankinter Empresas | 8 | 6 |
| Banco Santander Empresas | 5 | 4 |
| BBVA Net Cash Empresas | 3 | 2 |
| Cajamar | 2 | 2 |


**confirming_stacked**: 171 products / 47 companies / 16 banks. Empresas-labelled 117.

| bank | n | n_co |
| --- | --- | --- |
| Banco Santander Empresas | 49 | 17 |
| Banco Santander | 33 | 7 |
| Caixabank Empresas | 26 | 14 |
| Bankinter Empresas | 14 | 8 |
| BBVA Net Cash Empresas | 11 | 10 |


## 52. Drawn factoring vs connected-only

Drawn = last extract outstanding>1. Y3 is still 0 on both slices — draw vs connect does not make a health Y. PARK with util.

| slice | n_co | Y3 | Y2 | Y5 AP | leftover |
| --- | --- | --- | --- | --- | --- |
| drawn | 8 | 0.0% | 27.0% | 35.5% | 90.9% |
| connected_only | 10 | 0.0% | 1.5% | 8.3% | 100.0% |


## 53. Drawn factoring Y2 / AP after size

Drawn companies 8. Size-weighted Y2 pp 21.1%. A leftover pocket on n=8 is not a health Y and not a 44-col X.

| y | tercile | n_on | on | off | pp |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | T1 | 2 | 0.0% | 5.7% | -5.7% |
| y2_neg_2of3 | T2 | 12 | 16.7% | 7.2% | 9.5% |
| y2_neg_2of3 | T3 | 21 | 38.1% | 7.9% | 30.2% |
| y2_neg_2of3 | size-w | 35 | — | — | 21.1% |
| y5_ap_od30_ownp80 | T1 | 0 | — | 6.8% | — |
| y5_ap_od30_ownp80 | T2 | 12 | 33.3% | 8.0% | 25.3% |
| y5_ap_od30_ownp80 | T3 | 19 | 36.8% | 9.9% | 27.0% |
| y5_ap_od30_ownp80 | size-w | 31 | — | — | 26.3% |


## 54. Drawn factoring without GROUP_0139

Drawn ∩ GROUP_0139 = 3 of 8. Remaining 5: COMP_0058, COMP_0511, COMP_0769, COMP_0855, COMP_1068. Y2 0.0% n=13; AP 14.3%; Y3 0.0%.

## 55. Drawn confirming vs connected-only

Same extract PARK as factoring. If drawn confirming is also a leftover pocket, still not a Y.

| slice | n_co | Y3 | Y2 | Y5 AP |
| --- | --- | --- | --- | --- |
| drawn | 36 | 4.4% | 16.7% | 13.0% |
| connected_only | 25 | 8.5% | 7.3% | 12.5% |


## 56. Drawn confirming Y2 after size

Drawn confirming companies 36. Size-weighted Y2 pp 8.9%. Raw +9pp vs connected-only is not a 44-col X (extract book; PARK with util).

| tercile | n_on | on | off | pp |
| --- | --- | --- | --- | --- |
| T1 | 16 | 0.0% | 5.7% | -5.7% |
| T2 | 37 | 0.0% | 7.3% | -7.3% |
| T3 | 199 | 20.6% | 7.5% | 13.1% |
| size-w | 252 | — | — | 8.9% |


## 57. Holdout WC companies (coverage only)

Last-month holdout companies with any of the three flags: 19. Factoring holdout: COMP_0269 (factoring+confirming+loc). Largest holdout WC group GROUP_0103 n=6 (different from train GROUP_0139 — hidden test is new groups). Counts only. No AUROC. No percentile fit.

| company | group | book | flags | a_in3 |
| --- | --- | --- | --- | --- |
| COMP_0023 | GROUP_0237 | ERP | loc | 700891 |
| COMP_0091 | GROUP_0237 | ERP | loc | 479381 |
| COMP_0128 | GROUP_0103 | dark | confirming+loc | 37525 |
| COMP_0255 | GROUP_0025 | ERP | loc | 1087982 |
| COMP_0269 | GROUP_0156 | ERP | factoring+confirming+loc | 0 |
| COMP_0447 | GROUP_0190 | ERP | loc | 1903570 |
| COMP_0452 | GROUP_0103 | dark | confirming | 1725118 |
| COMP_0556 | GROUP_0237 | ERP | loc | 0 |
| COMP_0654 | GROUP_0053 | ERP | loc | 220627 |
| COMP_0700 | GROUP_0103 | dark | confirming+loc | 4550562 |
| COMP_0774 | GROUP_0103 | dark | loc | 274500 |
| COMP_0952 | GROUP_0103 | dark | confirming | 410636 |
| COMP_0975 | GROUP_0237 | ERP | loc | 226059 |
| COMP_1036 | GROUP_0121 | dark | confirming+loc | 70732483 |
| COMP_1049 | GROUP_0025 | ERP | confirming+loc | 3056244 |
| COMP_1091 | GROUP_0103 | dark | confirming+loc | 5568656 |
| COMP_1106 | GROUP_0056 | ERP | confirming+loc | 737860 |
| COMP_1232 | GROUP_0237 | ERP | loc | 73117 |
| COMP_1258 | GROUP_0056 | ERP | loc | 52437 |


Holdout WC groups unseen on train: True (n_groups=8, overlap=0). GROUP_0103 in train=False. Whole-group holdout holds.

## 21. Q3 birth vs add-on + calendar

New-WC months: birth 55 (Y3 6.9% n=29; Y2 2.4%); add-on 25 (Y3 0.0% n=8; Y2 12.5%). Same connection-clock shape as `f_new_facility` / `g_new`. Q3 gate now requires n_lab_on ≥ 20 — the first-cut KEEP was new-factoring Y3 on 11 labeled months.

| period | new WC | fact | conf |
| --- | --- | --- | --- |
| 2024-10 | 1 | 0 | 1 |
| 2024-11 | 1 | 1 | 0 |
| 2025-01 | 1 | 0 | 1 |
| 2025-02 | 4 | 0 | 4 |
| 2025-03 | 2 | 0 | 2 |
| 2025-05 | 1 | 0 | 1 |
| 2025-06 | 5 | 0 | 5 |
| 2025-07 | 3 | 0 | 3 |
| 2025-09 | 3 | 0 | 3 |
| 2025-10 | 11 | 7 | 6 |
| 2025-11 | 6 | 2 | 5 |
| 2025-12 | 5 | 1 | 4 |
| 2026-01 | 2 | 1 | 1 |
| 2026-02 | 6 | 1 | 6 |
| 2026-03 | 2 | 1 | 2 |
| 2026-04 | 3 | 1 | 3 |
| 2026-05 | 2 | 1 | 1 |
| 2026-06 | 4 | 0 | 4 |
| 2026-07 | 11 | 2 | 10 |
| 2026-08 | 7 | 2 | 7 |


## Return card

- ever-n: factoring 18, confirming 61, LOC 186
- Y3 singles vs 0.711 / size 0.617: fact 0.505 (Δ -0.112), conf 0.485 (Δ -0.132), loc 0.546 (Δ -0.070)
- rise-only: fact=True conf=True loc=True
- dark vs invoiced: factoring 10 vs 8, confirming 28 vs 33, lineofcredit 74 vs 112
- drop-from-44: **YES — drop f_has_factoring, f_has_confirming, f_has_loc (and f_new_facility)**

## Failed / next

- First-cut Q3 KEEP was new-factoring Y3 0% on 11 labeled months. Min-n=20 + WC-only gate **CLOSE**d it.
- Factoring AP +19pp after size is GROUP_0139 (3/4 leftover names). Do not invent a Y5-factoring label.
- 0/18 never-Y3 is real vs T3 peers 7.7% and is still CLOSE as X (18-company dummy; hidden test is new groups).
- Factoring-only last-month (3 names) all have `loan` rows — `f_ds_r` is the loan book, not a factoring-flow proxy. Flow KEEP stands.
- Confirming-only AP matches rest (~8–9%); the +4.4pp AP lift is stacked WC. Q5 tool-identity footnote **CLOSE**.
- Drawn factoring Y2 +21pp after size is GROUP_0139 (3/8). Remaining 5: Y2 0% n=13. PARK, not a pocket Y.
- Drawn confirming Y2 +8.9pp after size (n=36). AP flat vs connected. Still PARK extract, not 44 X.
- Holdout factoring is COMP_0269 stacked; largest holdout WC group is not GROUP_0139. Hidden test = new groups. CLOSE as X.
- Next (not this owner): drop the three flags + `f_new_facility` from `gbm_core` CORE/44 when someone re-cards. Not a parquet rewrite.

Elapsed 8s. Cuts: prevalence, size, singles, Q3 (min-n), overlap, dark, banks, AP-side, group dummy, co-occur, Y5|size, dark WC, Empresas, GROUP_0139, Y3-zero, connect-lag, granted, 2025-10 wave, AP-who, leftover-group, siblings, currency, store=live, ICC, Y4|size, AP minus GROUP_0139, pre/post, holdout, FX∩, Y2|size-year, never-recover, trail, COMP_0919, n-prod, company-lag, Y3-coverage, labels, size-matched never-recover, in-house LOC, liquidity, service, birth/calendar, other-debt, confirming-only/stacked AP, unused-limit, LOC-only, confirming banks, drawn factoring/confirming, holdout WC ids, unseen holdout groups.

