# Unused leftover of `a_uncat_share` after days as Y3 X

Generated `2026-09-19T06:10:42+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_uncat`. Do not put uncat on the 15-col card. Do not overwrite `uncat_qa.*`. Do not merge amount-uncat. Do not add `uncategorized` to CAT_MAP. Do not grow TURNOVER.

`a_uncat_share` = count share of txs with category `uncategorized` or not in CAT_MAP (Family A, this month). Uncat QA already PARK as Y and as X (style dummy). This lane is leftover after days on the 44 stem. Missing-CP is not an uncat twin (ρ 0.131).

## Headline

`a_uncat_share` as Y3 X: **CLOSE unused leftover** (leftover after days rank 0.573 lives but fails beat-size (0.542 vs 0.617). Style dummy ICC 0.985. DROP from the 44 as Y3 X. PARK as Y.). Y3 leftover after days OLS 0.559 rank 0.573 (lives, fake=False). Inverse days after uncat rank 0.712. Single 0.542 vs size 0.617 vs days 0.711 vs amt 0.530. after miss_cp 0.533 after d_tx 0.536. ICC 0.985 TRAIT. 15-col card: no — do not put a_uncat_share on the 15-col card. PARK as Y — style dummy, do not invent y_uncat. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — style dummy, do not invent y_uncat. Uncat ≠ no ERP. |
| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 0.557. |
| 3 | Who is turning? | **CLOSE unused leftover** leftover after days 0.573 vs days 0.711. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | Twin screen: ['amt_uncat']. ρ vs miss_cp 0.131. Style dummy ICC 0.985. |
| 6 | Months earlier? | lag1 leftover 0.557; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `a_uncat_share` as Y3 X / the 15-col card | **CLOSE unused leftover** | leftover after days rank 0.573 lives but fails beat-size (0.542 vs 0.617). Style dummy ICC 0.985. DROP from the 44 as Y3 X. PARK as Y. |
| `a_uncat_share` as engine X on the 44 | **DROP** | leftover lives=True twin=False SIZE=False beat-size=False |
| uncat as health Y | **PARK** | style dummy; do not invent y_uncat |
| amount-uncat merge | **CLOSED** | ρ=0.889; leftover after count 0.466 |
| CAT_MAP add `uncategorized` | **no** | leftover token is the label |
| miss_cp / d_tx twin | **NO** | ρ miss 0.131 d_tx -0.141 |
| Q6 lag1 after days_lag1 | **KEEP** | leftover 0.557 |

## 1 — Coverage; twin / SIZE screen

Train 1,214 co / 21,157 CM. a_uncat_share cov 95.8% mean 0.258 (quote 0.258 CONFIRM) acf1=0.270. amt_uncat mean 0.226 (quote 0.226 CONFIRM) ρ=0.889 (quote 0.889 CONFIRM). store vs count max|Δ|=0.000000. ρ vs days 0.180 vs a_n_tx 0.181 vs miss_cp 0.131 (quote 0.131 CONFIRM) vs d_tx -0.141 vs size 0.001. SIZE=False twins=['amt_uncat'] twin_gate=False.

| col | n_nn | cov | mean | acf1 |
| --- | --- | --- | --- | --- |
| a_uncat_share | 20,268 | 95.8% | 0.258 | 0.270 |
| amt_uncat (memory) | 20,268 | 95.8% | 0.226 | 0.050 |
| miss_cp_share (memory) | 20,268 | 95.8% | 0.855 | 0.535 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.180 | no |
| a_n_tx | 0.181 | no |
| miss_cp | 0.131 | no |
| d_tx_cp_share | -0.141 | no |
| log1p(a_in3) | 0.001 | no |
| amt_uncat | 0.889 | TWIN |


## 2 — Single-feature group-fold Y3

Y3 a_uncat_share 0.542 n=5,536 pos=372 (quote 0.542 / 5,536 / 372 CONFIRM). amt_uncat 0.530 vs size 0.617 vs days 0.711. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.075 FAIL.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_uncat_share | 5,536 | 372 | 0.542 | 0.534 | 0.046 | 1 | 0.528 0.542 0.553 0.607 0.478 |
| y3_recover_cash_6m | amt_uncat | 5,536 | 372 | 0.530 | 0.524 | 0.043 | 1 | 0.513 0.493 0.550 0.596 0.498 |
| y3_recover_cash_6m | miss_cp_share | 5,536 | 372 | 0.554 | 0.556 | 0.040 | 1 | 0.517 0.530 0.528 0.585 0.608 |
| y3_recover_cash_6m | d_tx_cp_share | 5,643 | 402 | 0.534 | 0.541 | 0.055 | -1 | 0.449 0.532 0.534 0.551 0.601 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

a_uncat_share leftover after days OLS 0.559 rank 0.573 fake=False almost=False ρ(resid,days)=0.111 R²=0.000 honest_dies=False n=5,536 pos=372. Inverse: days leftover after uncat OLS 0.702 rank 0.712 dies=False.

OLS folds: 0.532 0.559 0.571 0.627 0.506. Rank folds: 0.549 0.580 0.579 0.634 0.521.

## 4 — Twin / SIZE screen (in cut 1)

SIZE=False twin_gate=False twins=['amt_uncat'].

## 5 — Leftover after miss_cp / d_tx

uncat leftover after miss_cp rank 0.533 dies=True (want live if not a twin). after d_tx 0.536 dies=True; after days+miss 0.562 dies=False. Inverse miss after uncat 0.543; d_tx after uncat 0.520.

| bar | OLS | rank | ρ(resid,bar) | R2 | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after miss_cp | 0.535 | 0.533 | -0.091 | 0.019 | True | 5536 | 372 |
| after d_tx | 0.532 | 0.536 | 0.086 | 0.020 | True | 5536 | 372 |
| after miss+d_tx | 0.533 | 0.537 | -0.093 | 0.021 | True | 5536 | 372 |
| after days+miss | 0.543 | 0.562 | 0.095 | 0.019 | False | 5536 | 372 |
| miss_cp after uncat | 0.542 | 0.543 | -0.323 | 0.019 | True | 5536 | 372 |
| d_tx after uncat | 0.524 | 0.520 | 0.297 | 0.020 | True | 5536 | 372 |


## 6 — ICC / demean leftover

ICC MSB/(MSB+MSW)=0.985 (quote 0.985 CONFIRM) ANOVA ICC(1)=0.798 TRAIT k=1214. Demean CV 0.522 leftover-days 0.484 dies=True. Company-mean CV 0.570 leftover-days 0.586 dies=False. Leftover is BETWEEN (style), not a month shock.

## 7 — Dark vs ERP leftover (uncat ≠ no ERP)

Dark 470 (want 470) uncat nn=7206 mean 0.293 vs ERP mean 0.238 CONFIRM uncat ≠ no ERP. Dark Y3 0.548 leftover 0.575 dies=False. ERP Y3 0.543 leftover 0.578 dies=False.

| book | n_co | uncat nn | mean | Y3 n | Y3 pos | CV | leftover | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dark | 470 | 7206 | 0.293 | 1972 | 124 | 0.548 | 0.575 | False |
| ERP | 744 | 13062 | 0.238 | 3564 | 248 | 0.543 | 0.578 | False |


## 8 — Q6 lag1 leftover after days_lag1

Y3 uncat_lag1 0.529 (quote 0.529 CONFIRM) leftover after days_lag1 rank 0.557 dies=False. Days lag1 0.684 (quote 0.684 CONFIRM).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_uncat_share | 5,536 | 372 | 0.542 | 0.534 | 0.046 | 1 | 0.528 0.542 0.553 0.607 0.478 |
| y3_recover_cash_6m | a_uncat_share_lag1 | 5,546 | 383 | 0.529 | 0.526 | 0.041 | 1 | 0.493 0.536 0.542 0.586 0.485 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 9 — Count vs amount leftover (do not merge)

ρ(count, amount)=0.889 (quote 0.889 TWIN rewrite). Y3 amt_uncat 0.530 leftover after days 0.569 dies=False. count leftover after amount 0.522 dies=True. amount leftover after count 0.466 dies=True. Do not merge amount-uncat. Do not add uncategorized to CAT_MAP.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=1044 cov=97.3% p50=0.213 dark nn=478 dark mean=0.261 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.529 p50=0.574 p95=0.622 share<0.55=22.5% n=40.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.508 p90=0.526 n=24.

### leftover after size / a_n_tx

uncat leftover after size rank 0.542 dies=True. after a_n_tx 0.575 dies=False. after days+size 0.573 dies=False.

### Y3 rate by uncat quintile

Y3 uncat Q1→Q5 ['10.6%', '2.4%', '4.1%', '5.3%', '11.2%'].

| q | uncat rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 10.6% | 1108 | 118 |
| 2 | 2.4% | 1107 | 27 |
| 3 | 4.1% | 1156 | 47 |
| 4 | 5.3% | 1058 | 56 |
| 5 | 11.2% | 1107 | 124 |


### Fold-wise leftover

Rank leftover folds: 0.549 0.580 0.579 0.634 0.521.

| fold | OLS leftover | rank leftover | n_va | n_pos |
| --- | --- | --- | --- | --- |
| 0 | 0.532 | 0.549 | 1292 | 51 |
| 1 | 0.559 | 0.580 | 674 | 87 |
| 2 | 0.571 | 0.579 | 1055 | 58 |
| 3 | 0.627 | 0.634 | 1348 | 74 |
| 4 | 0.506 | 0.521 | 1167 | 102 |


### Y2 leftover after days (report-only)

Y2 uncat 0.584 leftover after days 0.576 dies=False (uncat_qa PARK as X — do not reopen).

### U-shape leftover

Q1|Q5 flag Y3 0.639 leftover after days 0.558 dies=False. Mid-quintiles leftover 0.643 dies=False. |uncat−p50| leftover 0.615 dies=False.

### all-uncat months

all-uncat months 531 / Y3 flag 0.506. Leftover after days on uncat<1 0.568 dies=False n=5494 pos=365.

### Q1|Q5 flag KEEP-as-X (footnote, not a 44 stem)

Q1|Q5 flag Y3 0.639 leftover-days 0.558 dies=False after size 0.605. ρ vs size -0.222 vs days -0.282. beat-size=True SIZE=False twin=False leftover_lives=True engine=True. Footnote only — do not add a U-shape flag to the 44.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| d_cust_top1 leftover | 0.525 DROP |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `a_uncat_share` on the 15-col card.

## Files written

- `analysis/evaluate/uncat_share_qa.py`
- `analysis/outputs/uncat_share_qa.md`
- `analysis/outputs/uncat_share_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_uncat_share.md` (end, if WRITE_WAVE)

Elapsed 30s. Failed: none.

