# Unused leftover of `j_pay_match` after days as Y3 X

Generated `2026-09-19T06:35:51+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Family J computed **in-memory** (not merged, not in FAMILIES). Rates and AUROC on **train**. Holdout 72 coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_pay_match`. Do not put J on the 15-col card. Do not overwrite `match.py`. Do not grow TURNOVER. 470 never-ERP stay NaN not 0. Y5 never E.

`j_pay_match` = share of paid-this-month book invoices with a greedy 1-1 bank match (|Δ| ≤ 0.01 €). Family J is **KEEP-Q5 diagnostic, not a Y3 X**.

## Headline

`j_pay_match` as Y3 X: **KEEP-Q5 only** (leftover after days rank 0.556 lives but beat-size FAIL (0.567 vs 0.617). J stays KEEP-Q5 diagnostic, not a Y3 X. Do not merge J.). Y3 leftover after days OLS 0.560 rank 0.556 (lives, fake=False). Inverse days after pay_match 0.710. Single 0.567 vs size 0.617 vs days 0.711 vs pending 0.484 vs iss 0.542. after pending 0.566 twin=False. Dark 470 NaN=True. Q6 lag1 leftover 0.586. Y5 leftover after size 0.533 (report-only). 15-col card: no — do not put J on the 15-col card. Merge: do not merge J. KEEP-Q5 diagnostic. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | KEEP-Q5 diagnostic. Dark 470 = NaN. |
| 2 | Who is improving? | Q6 lag1 leftover 0.586. |
| 3 | Who is turning? | **KEEP-Q5 only** leftover after days 0.556 vs days 0.711. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | KEEP-Q5 diagnostic — match rate on paid invoices, not a recover X. |
| 6 | Months earlier? | lag1 leftover 0.586; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP / KEEP-Q5

| object | decision | why |
| --- | --- | --- |
| `j_pay_match` as Y3 X / 15-col card | **KEEP-Q5 only** | leftover after days rank 0.556 lives but beat-size FAIL (0.567 vs 0.617). J stays KEEP-Q5 diagnostic, not a Y3 X. Do not merge J. |
| `j_pay_match` as engine X on the 44 | **DROP** | leftover lives=True twin=False SIZE=False beat-size=False |
| Family J merge into parquet | **do not merge J** | KEEP-as-X FAIL |
| Q5 diagnostic | **KEEP-Q5** | match rate on ever-ERP paid months |
| `y_pay_match` | **PARK** | do not invent |
| vs pending twin | **NO** | ρ=-0.093 |
| Q6 lag1 | **KEEP** | leftover 0.586 |

## 1 — Coverage; 470 NaN; twin / SIZE

Train 1,214 co / 21,157 CM. j_pay_match nn=9,648 cov 45.6% (ever-ERP 71.2%). Dark 470 nn=0 zero=0 has_book=0.000 CONFIRM NaN. ρ vs days 0.082 vs a_n_tx 0.091 vs pending -0.093 (peek −0.093 CONFIRM) vs iss 0.678 (peek 0.678 CONFIRM) vs size 0.000. SIZE=False twins=none.

| col | n_nn | cov | mean |
| --- | --- | --- | --- |
| j_pay_match | 9,648 | 45.6% | 0.392 |
| j_iss_match | 11,176 | 52.8% | 0.236 |
| j_has_book | 21,157 | 100.0% | 0.603 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.082 | no |
| a_n_tx | 0.091 | no |
| e_pending_amt_share | -0.093 | no |
| j_iss_match | 0.678 | no |
| log1p(a_in3) | 0.000 | no |


## 2 — Single-feature group-fold Y3

Y3 j_pay_match 0.567 n=2,646 pos=151 (peek 0.567 / 2,646 / 151 CONFIRM). vs size 0.617 vs days 0.711 vs pending 0.484 vs iss 0.542. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.050 FAIL.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | j_pay_match | 2,646 | 151 | 0.567 | 0.574 | 0.130 | -1 | 0.373 0.636 0.506 0.705 0.618 |
| y3_recover_cash_6m | j_iss_match | 3,075 | 176 | 0.542 | 0.533 | 0.056 | -1 | 0.479 0.569 0.517 0.624 0.520 |
| y3_recover_cash_6m | e_pending_amt_share | 3,446 | 239 | 0.484 | 0.523 | 0.102 | 1 | 0.571 0.458 0.381 0.609 0.399 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

j_pay_match leftover after days OLS 0.560 rank 0.556 fake=False almost=False ρ(resid,days)=0.041 R²=0.000 honest_dies=False n=2,646 pos=151. Inverse: days leftover after pay_match OLS 0.715 rank 0.710 dies=False.

OLS folds: 0.369 0.624 0.500 0.700 0.609. Rank folds: 0.370 0.617 0.499 0.693 0.599.

## 4 — Twin / SIZE (in cut 1)

SIZE=False twin_gate=False twins=none.

## 5 — Leftover after pending

pay leftover after pending rank 0.566 dies=False ρ=-0.093 twin=False (peek −0.093 / not a twin). after iss 0.524 dies=True; after days+pending 0.552 dies=False. pending leftover after pay_match 0.359 (Y3; pending Y3 leftover after days was 0.443).

| bar | OLS | rank | ρ(resid,bar) | R2 | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after pending | 0.563 | 0.566 | -0.014 | 0.004 | False | 2646 | 151 |
| after iss | 0.553 | 0.524 | 0.214 | 0.408 | True | 2599 | 142 |
| after days+pending | 0.559 | 0.552 | 0.068 | 0.004 | False | 2646 | 151 |
| pending after pay_match | 0.357 | 0.359 | -0.017 | 0.004 | True | 2646 | 151 |


## 6 — Dark 470 stay NaN; ERP leftover

Dark 470 (want 470) pay_match nn=0 CONFIRM NaN not 0. ERP Y3 0.567 leftover after days 0.556 dies=False.

| book | n_co | nn | Y3 n | Y3 pos | CV |
| --- | --- | --- | --- | --- | --- |
| dark | 470 | 0 | 0 | 0 | LOW_POWER |
| ERP | 744 | 9648 | 2646 | 151 | 0.567 |


## 7 — Q6 lag1 leftover after days_lag1

Y3 pay_match_lag1 0.596 leftover after days_lag1 rank 0.586 dies=False. Days lag1 0.684 (quote 0.684 CONFIRM).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | j_pay_match | 2,646 | 151 | 0.567 | 0.574 | 0.130 | -1 | 0.373 0.636 0.506 0.705 0.618 |
| y3_recover_cash_6m | j_pay_match_lag1 | 2,589 | 144 | 0.596 | 0.591 | 0.104 | -1 | 0.463 0.667 0.529 0.723 0.598 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 8 — Y5 leftover after size (report-only)

Y5 pay_match 0.532 leftover after size 0.533 dies=True (report-only; Y5 never E). after days 0.529 after pending 0.528. folds=0.489 0.631 0.511 0.521 0.509 fold3=0.521 hole_flag=False. d_tx leftover after J was Y5 0.588 ρ 0.262 — not overwritten.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ar_od30_sust | j_pay_match | 3,017 | 200 | 0.532 | 0.516 | 0.056 | 1 | 0.489 0.631 0.511 0.521 0.509 |
| y5_ar_od30_sust | log1p(a_in3) | 3,315 | 237 | 0.469 | 0.504 | 0.024 | 1 | 0.510 0.461 0.455 0.450 0.469 |


## 9 — Holdout coverage only

Holdout 72 co / 1073 CM nn=481 cov=44.8% p50=0.409 dark nn=0 (no fit, no AUROC). Do not merge J.

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.389 p50=0.570 p95=0.637 share<0.55=37.5% n=40.

### Y3 quintiles (Q5 diagnostic)

Y3 pay_match Q1→Q5 ['10.4%', '4.0%', '4.4%', '3.8%', '5.9%'] (KEEP-Q5 diagnostic, not a 44 stem).

| q | Y3 rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 10.4% | 540 | 56 |
| 2 | 4.0% | 520 | 21 |
| 3 | 4.4% | 528 | 23 |
| 4 | 3.8% | 529 | 20 |
| 5 | 5.9% | 529 | 31 |


### leftover after size / a_n_tx

pay leftover after size 0.562 dies=False. after a_n_tx 0.551 dies=False. after days+size 0.559 dies=False.

### j_has_book leftover

j_has_book Y3 0.504 leftover after days 0.621 dies=False. KEEP-Q5 miss flag, not a Y3 X.

### Y2 leftover

Y2 pay_match 0.544 leftover after days 0.552 dies=False.

### unmatched complement

j_pay_unmatched Y3 0.567 leftover after days 0.556 dies=False (exact complement — PARK, do not emit).

### Q1 low-match dummy

Q1 dummy Y3 0.586 leftover after days 0.554 dies=False after days+size 0.535. Low-match recover is a Q5 footnote, not a 44 stem.

### leftover after days+iss

leftover after days+iss 0.526 dies=True. after days+pending+iss 0.533 dies=True.

### ICC

ICC=0.925 k=691 n=9648 (acf1 quote 0.226 — not a TRAIT like pending 0.97).

### Y5 pending leftover after J

Y5 pending leftover after pay_match 0.531 dies=True. Y5 pay leftover after pending 0.528. ρ=-0.093. Quoted Y5 leftover 0.588 ρ 0.262 was d_tx after J — not pending after J. Not a twin.

### lag1 leftover after now-days

lag1 leftover after now-days 0.587 dies=False. lag1 single 0.596 beat-size=False. Q6 CLOSE as 44 stem.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| June last-tx leftover | 0.500 PARK extract |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put J on the 15-col card.

## Files written

- `analysis/evaluate/pay_match_qa.py`
- `analysis/outputs/pay_match_qa.md`
- `analysis/outputs/pay_match_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_pay_match.md` (end, if WRITE_WAVE)

Elapsed 20s. Failed: none.

