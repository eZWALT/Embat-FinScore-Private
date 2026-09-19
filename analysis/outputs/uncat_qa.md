# Q5 uncat readability — count vs amount

Generated `2026-09-19T03:26:00+02:00` by agent `ec17da1b`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent a merged Y from uncat. Family M CLOSED — amount-uncat is in-memory only.

`a_uncat_share` = share of txs with category `uncategorized` OR not in CAT_MAP (count). Amount-uncat = `sum(|amt| | same rule) / sum(|amt|)` computed here, not merged. Feature report: keep-list, cov 95.8%, size ρ −0.023, acf1 0.27, ICC 0.99 BETWEEN.

## Headline

Count mean 0.258 vs amount-uncat 0.226 (ρ 0.889; leftover tokens 1, only `uncategorized`). Y3 amount 0.530 count 0.542 vs size 0.617 (Δ -0.075) vs days 0.711. Count acf1 0.270 ICC 0.985 (style). Dark vs 744 count-uncat 31.7% vs 25.0%. Y2 amount 0.602 vs size 0.552 (style mean 0.586 / shock 0.527). Q5 **CLOSE**. PARK as health Y. X **PARK**. Q6 **CLOSE**.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Uncat is not a health Y. PARK. |
| 2 | Who is improving? | Not this share. |
| 3 | Who is turning? | A month-shock of uncategorized euros would be Q3; a style dummy is not. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | **CLOSE** — Y3 amount 0.530 loses to size 0.617 (Δ -0.075). Y2 amount 0.602 beats size 0.552 by 0.050 but is a bookkeeping-style dummy (ICC 0.978, acf1 0.050; company-mean 0.586 vs shock 0.527). Not Q5 readability of a *change*. PARK as X (Y2 0.602 ≥ 0.60). |
| 6 | Months earlier? | lag1 CLOSE (count now 0.542 / lag1 0.529; amount now 0.530 / lag1 0.520). Honest 1-month only. |

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| amount-uncat as Q5 why | **CLOSE** | Y3 amount 0.530 loses to size 0.617 (Δ -0.075). Y2 amount 0.602 beats size 0.552 by 0.050 but is a bookkeeping-style dummy (ICC 0.978, acf1 0.050; company-mean 0.586 vs shock 0.527). Not Q5 readability of a *change*. PARK as X (Y2 0.602 ≥ 0.60). |
| `a_uncat_share` as Q5 why | **CLOSE** | count ICC 0.985 acf1 0.270; style dummy |
| uncat as a health Y | **PARK** | do not invent a merged Y from opacity |
| uncat as Y3 X | **PARK** | size dummy 0.617 ≥ 0.6 |
| Q6 lag1 | **CLOSE** | Y3 count now 0.542 vs lag1 0.529 (Δ 0.013); amount now 0.530 vs lag1 0.520 (Δ 0.010). CLOSE as Q6 — lag1 does not hold contemporaneous skill (or contemporaneous is chance). |
| leftover CAT_MAP add | **CLOSE (only `uncategorized`)** | Universe tokens 22 (CAT_MAP size 22; seen mapped 21; leftover tokens 1). CONFIRM catmix: only leftover token is `uncategorized`. Train leftover txs 593,523 / € 138,777,893,069. Never-seen CAT_MAP keys: ['cash_settlements']. Frequent leftover to consider adding: 0 (design note only — CAT_MAP not edited). |
| Family M merge | **CLOSED** | do not merge mix; amount-uncat stays in-memory |
| Y2 amount-uncat 0.602 | **PARK as X** | beats size 0.050 but style (mean 0.586 / shock 0.527); y2_why is in flight |
| holding-style footnote | **measured** | group ICC of company-median 0.866; do not edit sibling_h |

## 1. Amount-mass vs count-share

Train CM 21,157. Store `a_uncat_share` cov 95.8% (CONFIRM 95.8%); recomputed count vs store max|Δ|=0.000000 mean|Δ|=0.000000. Count mean 0.258 vs amount-uncat mean 0.226 (Spearman 0.889; catmix quote ρ≈0.889). Pooled ticket uncat 24.6% vs euro uncat 29.9%. Months amount−count ≥0.20: 1,452; count−amount ≥0.20: 2,491. Leftover (not the `uncategorized` token) txs 0 / € 0. Size ρ vs a_in3 count 0.001 amount 0.032; vs log1p(|a_op_in|) count -0.023 (CONFIRM −0.023) amount 0.005. Amount-uncat is not systematically heavier than count.

| item | count `a_uncat_share` | amount-uncat (memory) |
| --- | ---: | ---: |
| coverage | 95.8% | 95.8% |
| mean | 0.258 | 0.226 |
| p50 | 0.125 | 0.033 |
| p90 | 0.794 | 0.840 |
| size ρ vs log1p(a_in3) | 0.001 | 0.032 |
| pooled ticket / euro share | 24.6% | 29.9% |
| Spearman count↔amount | 0.889 | Pearson 0.852 |
| store vs recomputed max\|Δ\| | 0.000000 | — |
| months amount−count ≥0.20 | 1,452 | (count hides euros) |
| months count−amount ≥0.20 | 2,491 | (many small uncat tickets) |
| leftover txs / € (not the token) | 0 | 0 |
| size ρ vs log1p(\|a_op_in\|) | -0.023 | 0.005 |

Amount-uncat is **not** merged. `m_uncat_n_share` stays parked as a rewrite of `a_uncat_share`.

## 2. Leftover categories (outside CAT_MAP)

Universe tokens 22 (CAT_MAP size 22; seen mapped 21; leftover tokens 1). CONFIRM catmix: only leftover token is `uncategorized`. Train leftover txs 593,523 / € 138,777,893,069. Never-seen CAT_MAP keys: ['cash_settlements']. Frequent leftover to consider adding: 0 (design note only — CAT_MAP not edited).

Top 15 leftover tokens by **count** (train):

| category | n | |amt| | n_co | share_n | share_|amt| |
| --- | --- | --- | --- | --- | --- |
| uncategorized | 593,523 | 138,777,893,069 | 1,145 | 100.0% | 100.0% |


Top 15 leftover tokens by **|amount|** (train):

| category | n | |amt| | n_co | share_n | share_|amt| |
| --- | --- | --- | --- | --- | --- |
| uncategorized | 593,523 | 138,777,893,069 | 1,145 | 100.0% | 100.0% |


### CAT_MAP design note (do not edit CAT_MAP)

No frequent leftover besides the `uncategorized` token. Do not add `uncategorized` to CAT_MAP — that would zero the readability hole. Never-seen mapped keys: `cash_settlements`.

## 3. Quintiles vs Y2 / Y3 / Y7 / Y9 base rates

Count-share quintiles vs Y2 rising, Y3 noise, Y7 noise, Y9 inverted-U. Amount-uncat quintiles vs Y2 rising, Y3 noise, Y7 rising, Y9 noise.

Cuts from **train** company-months with a finite feature. Holdout never enters a cut.

### `a_uncat_share` (count)

| q | n_cm | p50 | Y2 | Y3 | Y7 | Y9 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,116 | 0.000 | 5.9% | 6.2% | 24.2% | 14.9% |
| 2 | 4,116 | 0.125 | 6.8% | 4.2% | 29.3% | 15.9% |
| 3 | 3,982 | 0.330 | 8.1% | 7.1% | 33.9% | 17.0% |
| 4 | 4,054 | 0.794 | 10.3% | 13.5% | 32.6% | 9.9% |


### amount-uncat (in memory)

| q | n_cm | p50 | Y2 | Y3 | Y7 | Y9 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 8,107 | 0.000 | 5.1% | 6.9% | 24.7% | 15.2% |
| 2 | 4,054 | 0.033 | 7.0% | 3.5% | 26.8% | 14.7% |
| 3 | 4,053 | 0.253 | 9.6% | 6.3% | 33.9% | 15.6% |
| 4 | 4,054 | 0.840 | 10.3% | 14.6% | 34.7% | 12.2% |


Plot: `uncat_quintiles.png`.

## 4. Single-feature train group-fold AUROC

Y2 n=17,356 base 7.3%; Y3 n=5,648 base 7.1%; Y7 n=7,464 base 28.8%; Y9 n=9,591 base 14.1%. Sign from the train side of each fold. Seed 20260918. Night quote: `c_n_days_with_tx` 0.711 on Y3 (replica 0.711).

Y3 singles: count 0.542 amount-uncat 0.530 vs size 0.617 (best Δsize -0.075) vs days 0.711 (night 0.711, CONFIRM). Best uncat is a_uncat_share. Size≥0.60 on Y3: YES — PARK uncat as X.

| y | feature | n | n_pos | CV | sd | sign | train |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | a_uncat_share | 16,764 | 1,236 | 0.584 | 0.031 | 1 | 0.576 |
| y2_neg_2of3 | amt_uncat | 16,764 | 1,236 | 0.602 | 0.036 | 1 | 0.593 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 0.571 | 0.046 | 1 | 0.577 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 0.046 | 1 | 0.540 |
| y2_neg_2of3 | a_n_tx | 17,356 | 1,271 | 0.598 | 0.044 | 1 | 0.601 |
| y3_recover_cash_6m | a_uncat_share | 5,536 | 372 | 0.542 | 0.046 | 1 | 0.534 |
| y3_recover_cash_6m | amt_uncat | 5,536 | 372 | 0.530 | 0.043 | 1 | 0.524 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.714 |
| y7_top1_lost | a_uncat_share | 7,370 | 2,115 | 0.550 | 0.034 | 1 | 0.554 |
| y7_top1_lost | amt_uncat | 7,370 | 2,115 | 0.556 | 0.043 | 1 | 0.559 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.511 |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.536 |
| y7_top1_lost | a_n_tx | 7,464 | 2,149 | 0.452 | 0.032 | -1 | 0.516 |
| y9_fee_r_ownp80 | a_uncat_share | 9,134 | 1,336 | 0.520 | 0.029 | -1 | 0.522 |
| y9_fee_r_ownp80 | amt_uncat | 9,134 | 1,336 | 0.511 | 0.029 | -1 | 0.512 |
| y9_fee_r_ownp80 | c_n_days_with_tx | 9,591 | 1,350 | 0.565 | 0.057 | 1 | 0.567 |
| y9_fee_r_ownp80 | log1p_a_in3 | 9,591 | 1,350 | 0.534 | 0.018 | 1 | 0.534 |
| y9_fee_r_ownp80 | a_n_tx | 9,591 | 1,350 | 0.563 | 0.046 | 1 | 0.565 |


KEEP-as-Q5 rule: amount-uncat beats size by ≥0.02 **and** is not a style dummy (ICC ≥ 0.85 and month acf1 < 0.4). Size dummy ≥0.6 → PARK as X.

## 5. Persistence — style vs month shock

Count acf1=0.270 (CONFIRM ~0.27) acf3=0.121 acf6=0.037; ICC=0.985 (feature-report 0.99). Amount acf1=0.050 ICC=0.978. Company-median count: always-messy 206/1214, always-clean 309, shock (low median, max≥0.50) 111. Amount always-messy 101, shock 169. Count is STYLE dummy (high ICC + modest month acf). Amount is STYLE dummy.

| item | count | amount |
| --- | ---: | ---: |
| acf1 / acf3 / acf6 | 0.270 / 0.121 / 0.037 | 0.050 / 0.025 / -0.054 |
| ICC | 0.985 | 0.978 |
| company-median p50 | 0.135 | 0.043 |
| always-messy (med≥0.40, sd≤0.15) | 206 | 101 |
| always-clean (med≤0.05, sd≤0.10) | 309 | — |
| shock (med≤0.15, max≥0.50) | 111 | 169 |
| style dummy? | YES | YES |

Style = high company ICC + modest month acf: the firm is *always* messy, not a month that changed. That answers “who keeps a dirty book”, not Q5 “why did it change”.

## 6. Dark 470 vs 744 — uncat ≠ no ERP

Train last-month companies: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Mean company count-uncat: invoiced 25.0% vs dark 31.7%; amount-uncat 21.4% vs 29.4%. Dark and invoiced file uncat at different rates. Holdout ever-ERP coverage only: 40/72.

Company-level (mean of each company's uncat rate):

| group | n_co | mean_count | p50_count | mean_amt | p50_amt |
| --- | --- | --- | --- | --- | --- |
| ever_erp_744 | 744 | 0.250 | 0.111 | 0.214 | 0.019 |
| never_erp_470 | 470 | 0.317 | 0.184 | 0.294 | 0.107 |


Company-month:

| group | n_cm | n_co | count | amount |
| --- | --- | --- | --- | --- |
| ever_erp | 13,554 | 744 | 23.8% | 20.4% |
| never_erp | 7,603 | 470 | 29.3% | 26.6% |


## 7. Q6 — lag1 (honest 1-month only)

Y3 count now 0.542 vs lag1 0.529 (Δ 0.013); amount now 0.530 vs lag1 0.520 (Δ 0.010). CLOSE as Q6 — lag1 does not hold contemporaneous skill (or contemporaneous is chance).

| y | col | n | n_pos | present | CV | sign |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | a_uncat_share | 16,764 | 1,236 | 96.6% | 0.584 | 1 |
| y2_neg_2of3 | amt_uncat | 16,764 | 1,236 | 96.6% | 0.602 | 1 |
| y2_neg_2of3 | a_uncat_share_lag1 | 15,632 | 1,120 | 90.1% | 0.580 | 1 |
| y2_neg_2of3 | amt_uncat_lag1 | 15,632 | 1,120 | 90.1% | 0.599 | 1 |
| y2_neg_2of3 | c_n_days_with_tx | 17,356 | 1,271 | 100.0% | 0.571 | 1 |
| y2_neg_2of3 | c_n_days_with_tx_lag1 | 16,161 | 1,151 | 93.1% | 0.575 | 1 |
| y3_recover_cash_6m | a_uncat_share | 5,536 | 372 | 98.0% | 0.542 | 1 |
| y3_recover_cash_6m | amt_uncat | 5,536 | 372 | 98.0% | 0.530 | 1 |
| y3_recover_cash_6m | a_uncat_share_lag1 | 5,546 | 383 | 98.2% | 0.529 | 1 |
| y3_recover_cash_6m | amt_uncat_lag1 | 5,546 | 383 | 98.2% | 0.520 | 1 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 100.0% | 0.711 | -1 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 100.0% | 0.684 | -1 |
| y7_top1_lost | a_uncat_share | 7,370 | 2,115 | 98.7% | 0.550 | 1 |
| y7_top1_lost | amt_uncat | 7,370 | 2,115 | 98.7% | 0.556 | 1 |
| y7_top1_lost | a_uncat_share_lag1 | 7,163 | 2,041 | 96.0% | 0.552 | 1 |
| y7_top1_lost | amt_uncat_lag1 | 7,163 | 2,041 | 96.0% | 0.561 | 1 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 100.0% | 0.458 | -1 |
| y7_top1_lost | c_n_days_with_tx_lag1 | 7,253 | 2,072 | 97.2% | 0.457 | -1 |
| y9_fee_r_ownp80 | a_uncat_share | 9,134 | 1,336 | 95.2% | 0.520 | -1 |
| y9_fee_r_ownp80 | amt_uncat | 9,134 | 1,336 | 95.2% | 0.511 | -1 |
| y9_fee_r_ownp80 | a_uncat_share_lag1 | 9,176 | 1,329 | 95.7% | 0.515 | -1 |
| y9_fee_r_ownp80 | amt_uncat_lag1 | 9,176 | 1,329 | 95.7% | 0.508 | -1 |
| y9_fee_r_ownp80 | c_n_days_with_tx | 9,591 | 1,350 | 100.0% | 0.565 | 1 |
| y9_fee_r_ownp80 | c_n_days_with_tx_lag1 | 9,591 | 1,350 | 100.0% | 0.559 | 1 |


## 8. Uncat months vs `m_*` missingness (in memory)

Empty-tx months (a_uncat_share NaN = m_* all NaN): 889 / 21,157. High-uncat Q5 vs Q1 `m_coll_vs_pay` missing 25.5% vs 16.5%; e_ar_issued missing 41.5% vs 33.4%. High-uncat is not just a mix-missingness dummy.

Quintiles of `a_uncat_share` vs definition rates of mix / invoice / match columns. `m_*` is not in parquet (Family M CLOSED). `m_mix_ok` = month has |amount|>0; `m_coll_vs_pay` needs collection+payment euros.

| q | n_cm | p50_count | p50_amt | mix_defined | coll_pay_defined | e_ar_defined | d_cust_defined | j_pay_defined |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 8116 | 0.000 | 0.000 | 100.0% | 91.2% | 67.8% | 56.7% | 0.0% |
| 2 | 4116 | 0.125 | 0.043 | 100.0% | 98.1% | 68.2% | 57.1% | 0.0% |
| 3 | 3982 | 0.330 | 0.189 | 100.0% | 92.9% | 59.8% | 49.5% | 0.0% |
| 4 | 4054 | 0.794 | 0.804 | 100.0% | 74.5% | 58.5% | 47.6% | 0.0% |


## 9. Holdout coverage only (no AUROC)

Holdout coverage only (no AUROC): 72 companies / 1,073 CM. Count defined 97.3%, amount 97.3%.

| col | n_cm | n_co | defined | mean | p50 |
| --- | --- | --- | --- | --- | --- |
| count | 1073 | 72 | 97.3% | 0.336 | 0.213 |
| amount | 1073 | 72 | 97.3% | 0.295 | 0.083 |


## 10. Company-mean (style) vs demeaned (shock)

Y2 amount company-mean 0.586 vs demean (shock) 0.527. Y3 mean 0.605 vs shock 0.470. Y2 skill is the company style, not a month shock — not Q5 change.

| y | feature | n | n_pos | CV | sd | sign |
| --- | --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | count_co_mean | 17,356 | 1,271 | 0.573 | 0.042 | 1 |
| y2_neg_2of3 | count_demean | 16,764 | 1,236 | 0.528 | 0.025 | 1 |
| y2_neg_2of3 | amt_co_mean | 17,356 | 1,271 | 0.586 | 0.053 | 1 |
| y2_neg_2of3 | amt_demean | 16,764 | 1,236 | 0.527 | 0.029 | 1 |
| y3_recover_cash_6m | count_co_mean | 5,648 | 402 | 0.570 | 0.055 | 1 |
| y3_recover_cash_6m | count_demean | 5,536 | 372 | 0.522 | 0.039 | -1 |
| y3_recover_cash_6m | amt_co_mean | 5,648 | 402 | 0.605 | 0.063 | 1 |
| y3_recover_cash_6m | amt_demean | 5,536 | 372 | 0.470 | 0.040 | -1 |
| y7_top1_lost | count_co_mean | 7,464 | 2,149 | 0.555 | 0.051 | 1 |
| y7_top1_lost | count_demean | 7,370 | 2,115 | 0.520 | 0.025 | 1 |
| y7_top1_lost | amt_co_mean | 7,464 | 2,149 | 0.592 | 0.068 | 1 |
| y7_top1_lost | amt_demean | 7,370 | 2,115 | 0.505 | 0.038 | -1 |
| y9_fee_r_ownp80 | count_co_mean | 9,591 | 1,350 | 0.530 | 0.030 | -1 |
| y9_fee_r_ownp80 | count_demean | 9,134 | 1,336 | 0.482 | 0.035 | -1 |
| y9_fee_r_ownp80 | amt_co_mean | 9,591 | 1,350 | 0.533 | 0.036 | -1 |
| y9_fee_r_ownp80 | amt_demean | 9,134 | 1,336 | 0.522 | 0.024 | 1 |


If company-mean carries the AUROC and the demeaned month shock is chance, uncat answers “who keeps a dirty book”, not Q5 “why did this month change”.

## 11. Y2 0.602 vs `a_n_tx` (activity)

Spearman amount-uncat↔a_n_tx 0.252 (count↔n_tx 0.181; amount↔days 0.232). Amount-uncat is not an activity clone (|ρ|<0.50). Y2 night single a_n_tx was 0.598 vs amount 0.602 — almost a tie.

Within train terciles of `a_n_tx`, amount-uncat AUROC on Y2:

| ntx_tercile | n | n_pos | Y2_amt_CV |
| --- | --- | --- | --- |
| 1 | 5,779 | 277 | 0.574 |
| 2 | 5,556 | 396 | 0.589 |
| 3 | 5,429 | 563 | 0.595 |


Y2 rate by n_tx tercile × amount-uncat quartile (train cuts):

| ntx_tercile | amt_q | n_cm | n_labeled | Y2 | p50_amt | p50_ntx |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 3229 | 2753 | 3.9% | 0.000 | 5.0 |
| 1 | 2 | 1061 | 925 | 3.5% | 0.004 | 13.0 |
| 1 | 3 | 845 | 693 | 6.1% | 0.146 | 10.0 |
| 1 | 4 | 1694 | 1408 | 6.7% | 0.862 | 8.0 |
| 2 | 1 | 1334 | 1113 | 6.6% | 0.000 | 43.0 |
| 2 | 2 | 2006 | 1699 | 5.1% | 0.004 | 48.0 |
| 2 | 3 | 1642 | 1366 | 7.8% | 0.146 | 52.0 |
| 2 | 4 | 1723 | 1378 | 9.3% | 0.739 | 49.0 |
| 3 | 1 | 504 | 393 | 6.1% | 0.000 | 154.0 |
| 3 | 2 | 2000 | 1672 | 7.5% | 0.006 | 182.0 |
| 3 | 3 | 2580 | 2063 | 11.7% | 0.128 | 222.0 |
| 3 | 4 | 1650 | 1301 | 13.3% | 0.735 | 181.0 |


## 12. Zero pile — why qcut made 4 bins

Defined count months 20,268; exact-zero 21.1% (n=4,279) — qcut drops a quintile (4 bins). Amount exact-zero 4,279.

| col | band | n_cm | Y2 | Y3 | Y7 | Y9 |
| --- | --- | --- | --- | --- | --- | --- |
| count | zero | 4,279 | 4.4% | 11.5% | 23.9% | 14.0% |
| count | low_pos | 7,995 | 7.2% | 3.4% | 26.9% | 15.9% |
| count | high_pos | 7,994 | 9.2% | 9.4% | 33.3% | 13.6% |
| amt | zero | 4,279 | 4.4% | 11.5% | 23.9% | 14.0% |
| amt | low_pos | 7,995 | 6.4% | 3.5% | 26.0% | 15.5% |
| amt | high_pos | 7,994 | 10.0% | 9.2% | 34.4% | 14.0% |


## 13. Hide-euro months (amount − count ≥ 0.20)

Hide-euro months 1,452 / aligned 16,325 / small-ticket 2,491. Hide-euro dummy Y2 0.495 Y3 0.527.

| slice | n_cm | n_co | p50_count | p50_amt | y2_neg_2of3 | y3_recover_cash_6m | y7_top1_lost | y9_fee_r_ownp80 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hide_euros | 1452 | 435 | 0.325 | 0.730 | 7.2% | 14.2% | 38.2% | 18.5% |
| small_tickets | 2491 | 589 | 0.452 | 0.069 | 6.5% | 6.9% | 26.3% | 12.9% |
| aligned | 16325 | 1196 | 0.078 | 0.012 | 7.5% | 6.3% | 28.3% | 14.6% |


Count does hide some large uncategorized euros (1.4k months), but the hide dummy is not a Y3 signal. Do not rewrite `a_uncat_share` to amount.

## 14. `cash_settlements` alias (do not edit CAT_MAP)

`cash_settlement` (mapped op_in) n=48,236; `cash_settlements` (also in CAT_MAP, never seen) n=0. Do not drop the plural alias tonight — it is a Javier synonym, not a leftover.

## 15. Dark 470 vs 744 inside size terciles

Dark count-uncat gap raw 6.7%. Size terciles of log1p(a_in3) on train (holdout never in cuts).

| size_tercile | group | n_cm | n_co | count | amount | p50_in3 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | erp | 4076 | 417 | 29.2% | 27.6% | 9621 |
| 1 | dark | 2167 | 242 | 34.6% | 33.0% | 3908 |
| 2 | erp | 4172 | 486 | 21.2% | 17.3% | 258631 |
| 2 | dark | 2071 | 276 | 28.6% | 25.5% | 298778 |
| 3 | erp | 3818 | 386 | 20.5% | 15.8% | 2385597 |
| 3 | dark | 2425 | 260 | 24.7% | 21.7% | 2718841 |


Uncat ≠ no ERP: dark books are still mostly categorized. They are messier, and the gap is checked inside size terciles so it is not just smaller firms.

## 16. Company-median amount-uncat vs ever-Y2

Company-median amount-uncat vs company Y2 rate Spearman 0.097 (n_co=1,195). Style identity, not a month why.

| amt_med_q | n_co | p50_med_amt | ever_Y2 | mean_Y2_cm |
| --- | --- | --- | --- | --- |
| 1 | 299 | 0.000 | 10.7% | 6.0% |
| 2 | 299 | 0.006 | 10.7% | 5.5% |
| 3 | 298 | 0.132 | 16.8% | 8.4% |
| 4 | 299 | 0.766 | 18.7% | 9.5% |


## 17. Always-messy vs always-clean companies

Always-messy 206 vs always-clean 309 train companies (median count-uncat ≥0.40 & sd≤0.15 vs ≤0.05 & sd≤0.10). A type, not a month.

| group | n_co | dark | p50_in3 | p50_med_c | ever_Y2 | Y2_cm | ever_Y3 | Y3_cm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| always_messy | 206 | 51.5% | 296141 | 0.791 | 20.1% | 10.7% | 38.5% | 29.2% |
| always_clean | 309 | 36.9% | 267414 | 0.000 | 12.0% | 6.8% | 19.2% | 10.8% |
| other | 699 | 35.8% | 340994 | 0.158 | 13.5% | 6.7% | 23.8% | 15.4% |


## 18. Calendar of uncat (seasonal dummy?)

Stacked Jan–Dec count-uncat range 2.0%. No strong calendar dummy (range <8pp).

| month | n_cm | count | amount | Y2 |
| --- | --- | --- | --- | --- |
| Jan | 1768 | 26.5% | 23.5% | 8.4% |
| Feb | 1881 | 26.7% | 23.4% | 7.8% |
| Mar | 1925 | 25.9% | 22.5% | 6.7% |
| Apr | 1945 | 25.7% | 22.3% | 5.9% |
| May | 1966 | 26.5% | 22.9% | 5.8% |
| Jun | 1976 | 25.4% | 21.9% | 8.0% |
| Jul | 2010 | 24.7% | 22.1% | 7.3% |
| Aug | 2047 | 25.8% | 23.0% | 7.4% |
| Sep | 1299 | 25.5% | 22.3% | 7.6% |
| Oct | 1381 | 24.8% | 21.5% | 7.5% |
| Nov | 1428 | 25.5% | 22.4% | 8.2% |
| Dec | 1531 | 25.8% | 23.3% | 8.5% |


## 19. Uncategorized euro sign (in vs out)

Train uncategorized euros: inflow 52.8% / outflow 47.2% (€ in 73,274,008,121 / out 65,503,884,948). Company in-share p50 0.477; mostly-in 331 / mostly-out 373 / n=1145.

Do not invent an uncat-inflow Y. Sign is a readability footnote.

## 20. Y3 amount-uncat after `c_n_days_with_tx` terciles

Y3 amount-uncat inside `c_n_days_with_tx` terciles (train labeled). If residual is chance, uncat adds nothing after the 0.711 days bar.

| days_tercile | p50_days | n | n_pos | Y3_amt_CV | Y3_rate |
| --- | --- | --- | --- | --- | --- |
| 1 | 7.0 | 1,936 | 241 | 0.529 | 12.4% |
| 2 | 19.0 | 1,880 | 87 | 0.593 | 4.6% |
| 3 | 25.0 | 1,720 | 44 | LOW_POWER | 2.6% |


## 21. Months-on-book — onboarding mess?

Months 1–3 count-uncat 28.6% vs months 13+ 22.8%. Onboarding hole — early books are messier. Style from month 1, not a trail that gets labeled later.

| so_far | n_cm | n_co | count | amount | Y2 |
| --- | --- | --- | --- | --- | --- |
| 1-3 | 3642 | 1214 | 28.6% | 25.0% | 9.3% |
| 4-6 | 3637 | 1214 | 28.3% | 24.9% | 8.2% |
| 7-12 | 6059 | 1204 | 26.1% | 23.6% | 6.6% |
| 13-18 | 4590 | 833 | 22.4% | 19.2% | 6.3% |
| 19-24 | 3229 | 677 | 23.4% | 19.9% | 5.9% |


## 22. Always-messy Y3 label coverage (stressed density)

Y3 is labeled only among stressed. Always-messy Y3-coverage minus clean -15.3%. A higher recover *rate among labeled* can still be a type (more often stressed), not Q5 readability.

| group | n_co | Y3_label_cov | mean_n_pos |
| --- | --- | --- | --- |
| always_messy | 206 | 12.4% | 0.27 |
| always_clean | 309 | 27.7% | 0.34 |


## 23. Within-company early vs late (selection vs cleaning)

Companies with ≥13 months: n=812. Within-company late−early count-uncat mean Δ -0.018 (share cleaner>5pp 34.2%; worse 25.0%); early↔late Spearman 0.651. Short-book (<13) CM count-uncat 33.8% vs long-book 24.2%. Within-company path is mixed.

| item | value |
| --- | ---: |
| long companies (≥13m) | 812 |
| short-book companies | 381 |
| late−early mean Δ | -0.018 |
| share cleaner >5pp | 34.2% |
| share worse >5pp | 25.0% |
| early↔late Spearman | 0.651 |
| short-book count-uncat | 33.8% |
| long-book count-uncat | 24.2% |

## 24. All-uncat months (amount ≥ 0.95)

All-uncat months (amount≥0.95): 1,280 / 20,268. If Y rates match the high quintile, the tail is the style dummy, not a new event.

| slice | n_cm | n_co | dark | y2_neg_2of3 | y3_recover_cash_6m | y7_top1_lost | y9_fee_r_ownp80 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_uncat>=0.95 | 1280 | 300 | 44.7% | 10.9% | 24.6% | 38.4% | 10.6% |
| mixed | 8208 | 974 | 40.2% | 9.5% | 7.0% | 32.8% | 14.4% |
| zero | 4279 | 577 | 33.4% | 4.4% | 11.5% | 23.9% | 14.0% |


## 25. Group ICC of company-median uncat

Company-median count-uncat ICC across group_id 0.866 (amount 0.861). Multi-company groups 171: mono-messy 17, mono-clean 13. Holding style.

## 26. All-dark 360 vs mixed-dark 110 (uncat ≠ no ERP)

Train last-month: invoiced 744 / all-dark 360 / mixed-dark 110 (CONFIRM 744/360/110). Uncat is not a dark-only hole.

| group | n_co | mean_count | mean_amt |
| --- | --- | --- | --- |
| invoiced_744 | 744 | 0.250 | 0.214 |
| all_dark_360 | 360 | 0.315 | 0.297 |
| mixed_dark_110 | 110 | 0.323 | 0.286 |


## 27. Uncat vs mapped ticket size

Pooled |amt|/tx uncat 233821 vs mapped 178301 (1.31×). Typical-company mean ticket uncat 5138 vs mapped 10314. Fat tail: a few huge uncat txs lift pooled euros; typical uncat tickets are smaller.

| kind | n | |amt| | pooled mean/tx | co-median of mean | co-median of p50 |
| --- | --- | --- | --- | --- | --- |
| mapped | 1,822,584 | 324,968,099,877 | 178,301 | 10,314 | 826 |
| uncat | 593,523 | 138,777,893,069 | 233,821 | 5,138 | 386 |


## 28. Pending vs uncat

`a_pending_share` vs count-uncat Spearman 0.007 (amount 0.033); share exactly 0 97.3%. High-uncat pending mean 0.1% vs low 0.1%. Uncat is not a pending dummy (booked txs can be uncategorized).

Feature report: `a_pending_share` is near-zero variance. If it does not clone uncat, opacity is a booked-but-uncategorized style, not a settlement lag.

## 29. Holding mean vs within-group residual

Y2 group-mean amount-uncat 0.558 vs within-group residual 0.552. Within-group residual still has skill; not only a holding dummy.

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | group_mean_amt | 17,356 | 1,271 | 0.558 | 1 |
| y2_neg_2of3 | within_group_resid | 16,764 | 1,236 | 0.552 | 1 |
| y3_recover_cash_6m | group_mean_amt | 5,648 | 402 | 0.596 | 1 |
| y3_recover_cash_6m | within_group_resid | 5,536 | 372 | 0.533 | 1 |


## 30. Uncat vs mapped row flags (do not edit clean_flags)

Uncat vs mapped row flags (train txs). If booked and described, opacity is a category-style hole, not DQ.

| kind | n | pending | booked | is_dup | is_extreme | product_known | empty_desc | no_cp |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uncat | 593,523 | 0.4% | 96.3% | 4.7% | 0.0% | 99.9% | 0.0% | 91.8% |
| mapped | 1,822,584 | 0.2% | 99.2% | 3.9% | 0.0% | 100.0% | 0.0% | 89.6% |


Accounting status (top):

| uncat | accounting_status | n |
| --- | --- | --- |
| 0 | nan | 1,057,591 |
| 1 | nan | 417,163 |
| 0 | RECONCILIATION_COMPLETED | 415,211 |
| 0 | DISCARDED | 243,008 |
| 1 | RECONCILIATION_COMPLETED | 86,957 |
| 0 | PENDING | 66,682 |
| 1 | DISCARDED | 53,614 |
| 1 | PENDING | 26,224 |


## 31. value_date lag (settlement delay?)

value_date−date p50 uncat 0.0 vs mapped 0.0. Same settlement lag — uncat is not a value-date delay.

| kind | n | p50 | mean | share_0 | share_gt2 |
| --- | --- | --- | --- | --- | --- |
| uncat | 593,421 | 0.0 | -0.12 | 89.6% | 2.8% |
| mapped | 1,822,491 | 0.0 | -0.24 | 87.9% | 3.6% |


## 32. Uncat by banking product type (do not edit G)

Uncat share by product type (train txs). Card 48.0% vs checking 24.5%. Do not edit Family G.

| type | n | uncat_n | uncat_share |
| --- | --- | --- | --- |
| checking | 2,140,232 | 523,960 | 24.5% |
| (unknown) | 174,609 | 23,750 | 13.6% |
| card | 76,684 | 36,783 | 48.0% |
| tpv | 17,911 | 5,224 | 29.2% |
| expensesPlatform | 5,070 | 2,666 | 52.6% |
| wallet | 1,540 | 1,103 | 71.6% |
| saving | 38 | 15 | 39.5% |
| lineofcomex | 23 | 22 | 95.7% |


## 33. Is Y2 0.602 just `g_has_card`?

Y2 `g_has_card` 0.467 vs amount-uncat 0.602 (Spearman -0.045). Uncat is not just has-card (banking_g CLOSE as Y3 X stands).

| y | feature | n | n_pos | CV | sign |
| --- | --- | --- | --- | --- | --- |
| y2_neg_2of3 | g_has_card | 17,356 | 1,271 | 0.467 | -1 |
| y2_neg_2of3 | amt_uncat | 16,764 | 1,236 | 0.602 | 1 |
| y2_neg_2of3 | log1p_a_in3 | 14,968 | 1,044 | 0.552 | 1 |
| y2_neg_2of3 | g_has_checking | 17,356 | 1,271 | 0.546 | -1 |
| y3_recover_cash_6m | g_has_card | 5,648 | 402 | 0.551 | -1 |
| y3_recover_cash_6m | amt_uncat | 5,536 | 372 | 0.530 | 1 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | -1 |
| y3_recover_cash_6m | g_has_checking | 5,648 | 402 | 0.490 | -1 |


Do not edit `banking_g_qa.py`. Card 0.551 was already CLOSE as Y3 X.

## 34. Amount-uncat Y2 on checking-only (no card)

Amount-uncat Y2 on no-card company-months: 0.606 (has-card 0.637). If skill survives, the style is on checking books too (card is 48% uncat but 3% of txs).

| slice | n_cm | n_co | n | n_pos | Y2_amt_CV | mean_amt |
| --- | --- | --- | --- | --- | --- | --- |
| no_card | 18369 | 1168 | 14,580 | 1,083 | 0.606 | 0.244 |
| has_card | 2788 | 197 | 2,184 | 153 | 0.637 | 0.112 |


## 35. Inflow vs outflow uncat (messy AP vs both-sides style)

Y2 inflow-uncat 0.588 vs outflow-uncat 0.605 (pooled amount 0.602). Both sides — style on the whole book, not a messy-AP hole.

| feat | Y | mean | n | n_pos | CV |
| --- | --- | --- | --- | --- | --- |
| inflow | Y2 | 0.225 | 15,398 | 1,176 | 0.588 |
| inflow | Y3 | 0.225 | 5,335 | 317 | 0.576 |
| outflow | Y2 | 0.227 | 16,612 | 1,228 | 0.605 |
| outflow | Y3 | 0.227 | 5,524 | 370 | 0.469 |
| amount | Y2 | 0.226 | 16,764 | 1,236 | 0.602 |
| amount | Y3 | 0.226 | 5,536 | 372 | 0.530 |


## 36. Amount-uncat residual after `a_n_tx`

Amount-uncat residual after log1p(a_n_tx) (slope 0.004): Y2 0.583 Y3 0.618. If Y2 stays ~0.60, activity does not explain the 0.602.

| Y | n | n_pos | resid_CV |
| --- | --- | --- | --- |
| Y2 | 16,764 | 1,236 | 0.583 |
| Y3 | 5,536 | 372 | 0.618 |


## 37. Is the n_tx residual just size?

n_tx-residual ↔ size Spearman -0.151 (↔ amt 0.910). Y3 resid 0.618 sign 1 vs size 0.617 sign -1. Residual is not a size clone; still PARK (raw amount 0.530).

| item | value |
| --- | --- |
| resid↔size Spearman | -0.151 |
| resid↔amt Spearman | 0.910 |
| Y3 resid CV | 0.618 |
| Y3 size CV | 0.617 |
| Y3 resid train sign | 1 |
| Y3 size train sign | -1 |


## 38. Amount-uncat Y2 inside size terciles

Amount-uncat Y2 inside log1p(a_in3) terciles. Skill in more than one size band — not only a large-book dummy.

| size_tercile | p50_log_in3 | n | n_pos | Y2_amt_CV | mean_amt |
| --- | --- | --- | --- | --- | --- |
| 1 | 9.56 | 4,878 | 281 | 0.582 | 0.289 |
| 2 | 12.63 | 4,807 | 354 | 0.642 | 0.198 |
| 3 | 14.78 | 4,732 | 380 | 0.614 | 0.181 |


## 39. Amount-uncat Y2 on mature books

Amount-uncat Y2 on months 13+ 0.638 (months 1–3 0.582). Survives on mature books — not only an onboarding hole.

| slice | n_cm | n_co | n | n_pos | Y2_amt_CV |
| --- | --- | --- | --- | --- | --- |
| months_1-3 | 3642 | 1214 | 3,510 | 325 | 0.582 |
| months_13+ | 7819 | 833 | 5,115 | 329 | 0.638 |
| all | 21157 | 1214 | 16,764 | 1,236 | 0.602 |


## 40. Y2 amount-uncat on invoiced 744 vs dark 470

Y2 amount-uncat invoiced 0.566 vs dark 0.605. Skill on both sides — uncat ≠ no ERP, and the style is not a dark-only dummy.

| slice | n_cm | n_co | n | n_pos | Y2_amt_CV | mean_amt |
| --- | --- | --- | --- | --- | --- | --- |
| invoiced_744 | 13554 | 744 | 10,934 | 692 | 0.566 | 0.204 |
| dark_470 | 7603 | 470 | 5,830 | 544 | 0.605 | 0.266 |


## 41. Always-uncat counterparties (mapping hole vs random miss)

Missing-CP uncat 91.8%; named sticky (≥80%) 5.8% of all uncat / 70.4% of named-CP uncat. Company ICC still says style — this is a labeling habit across many CPs, not a few always-uncat vendors (and not a month miss).

| cp_kind | n_pairs | uncat_txs | share_n | share_|amt| |
| --- | --- | --- | --- | --- |
| missing_cp | 1214 | 544,872 | 91.8% | 94.0% |
| named_sticky≥80% | 9047 | 34,226 | 5.8% | 5.4% |
| named_mixed_20-80 | 2247 | 12,835 | 2.2% | 0.5% |
| named_rare_≤20% | 565 | 1,590 | 0.3% | 0.1% |


## 42. Description length (described but unlabeled)

Description length p50 uncat 37 vs mapped 39. Same length — uncat is labeled-as-uncategorized, not a blank memo.

| kind | n | p10 | p50 | p90 | share_lt8 |
| --- | --- | --- | --- | --- | --- |
| mapped | 1,822,584 | 15 | 39 | 87 | 1.7% |
| uncat | 593,523 | 15 | 37 | 116 | 1.8% |


## 43. Y2 after dropping all-uncat months

Y2 amount-uncat after dropping amount≥0.95 months: 0.601 (all-uncat-only 0.396). Skill survives without the 100% tail — not only the all-uncat dummy.

| slice | n_cm | n | n_pos | Y2_amt_CV |
| --- | --- | --- | --- | --- |
| drop_alluncat | 18988 | 15,728 | 1,123 | 0.601 |
| alluncat_only | 1280 | 1,036 | 113 | 0.396 |
| all_defined | 20268 | 16,764 | 1,236 | 0.602 |


## 44. Weekday of uncat vs mapped

Weekday uncat-share range 7.5%. No weekday dummy (range <8pp) — not a weekend batch hole.

| dow | n | uncat_n | uncat_share |
| --- | --- | --- | --- |
| Sun | 50,817 | 15,489 | 30.5% |
| Mon | 524,328 | 120,515 | 23.0% |
| Tue | 466,280 | 110,684 | 23.7% |
| Wed | 430,754 | 106,723 | 24.8% |
| Thu | 431,937 | 106,910 | 24.8% |
| Fri | 441,937 | 112,329 | 25.4% |
| Sat | 70,054 | 20,873 | 29.8% |


## 45. Invoiced-only style vs shock

Invoiced-only Y2 company-mean 0.557 vs demean 0.525 (now 0.566). ERP books still carry style, not a month change — y2_why should not treat uncat as a why.

| feat | n | n_pos | Y2_CV |
| --- | --- | --- | --- |
| style | 11,275 | 715 | 0.557 |
| shock | 10,934 | 692 | 0.525 |
| now | 10,934 | 692 | 0.566 |


## 46. Mapped-category richness vs uncat (CAT_MAP design note)

Mapped CAT_MAP tokens per month vs count-uncat Spearman -0.214. If high-uncat months still show many mapped tokens, leftover is a labeling habit on top of a used map — do not add `uncategorized` to CAT_MAP.

| q | n_cm | p50_uncat | p50_mapped_cats |
| --- | --- | --- | --- |
| 1 | 8116 | 0.000 | 7.0 |
| 2 | 4116 | 0.125 | 8.0 |
| 3 | 3982 | 0.330 | 6.0 |
| 4 | 4054 | 0.794 | 3.0 |


Do not edit CAT_MAP. Do not add `uncategorized`.

## What failed / next

- Y2 amount 0.602 is company-mean style (0.586) not month shock (0.527)
- CAT_MAP leftover tokens: 1 (`uncategorized` only) — do not add that token; never-seen key `cash_settlements` is a Javier alias
- High-uncat vs Q1 `m_coll_vs_pay` missing 25.5% vs 16.5% — not a mix-missingness dummy
- Card txs 48.0% uncat vs checking 24.5%; Y2 g_has_card 0.467 vs amount 0.602; checking-only Y2 0.606
- Uncat rows booked 96.3% / empty desc 0 — category-style hole, not DQ
- Inflow vs outflow Y2 0.588 / 0.605 (both-sides style)
- Amount residual after log1p(a_n_tx) Y2 0.583 Y3 0.618; resid↔size ρ -0.151 (not size)
- Y2 inside size terciles survives=True; mature-book Y2 0.638
- Y2 invoiced 0.566 vs dark 0.605 (both sides) — uncat ≠ no ERP
- Missing-CP uncat 91.8%; named sticky 5.8% (of named 70.4%) — labeling habit, not vendor hole
- Desc length p50 uncat 37 vs mapped 39 (same memo)
- Y2 after drop all-uncat months 0.601
- Weekday uncat range 7.5% (no weekday dummy)
- Invoiced-only Y2 style 0.557 vs shock 0.525 (style carries)
- Mapped-cat richness ↔ count-uncat ρ -0.214 — CAT_MAP leftover CLOSE

Elapsed 12s. Cuts: amount vs count, leftover cats, quintiles, singles, style/shock, dark 470/744, Q6 lag1, mix missingness, holdout, demean, activity, zero pile, hide-euro, CAT_MAP alias, dark×size, company-median Y2, messy-company cards, calendar, uncat sign, Y3 after days, months-on-book, messy Y3 label coverage, within early/late, all-uncat months, group ICC, 360/110, ticket size, pending vs uncat, group residual, row flags, value_date lag, product type, card dummy, checking-only, inflow/outflow, n_tx residual, resid vs size, size terciles, mature books, dark vs invoiced Y2, sticky CP, desc length, drop all-uncat, weekday, invoiced style.

