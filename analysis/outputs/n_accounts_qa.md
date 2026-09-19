# Unused leftover of `g_n_accounts` after days as Y3 X

Generated `2026-09-19T07:18:33+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_n_accounts`. Do not put g_n_accounts on the 15-col card. Do not overwrite `banking_g_qa.*`, `g_has_rest_qa.*`. Do not grow TURNOVER.

`g_n_accounts` = COUNT(product_id) as-of period_end. Rise-only connection inventory. `g_has_*` already DROP. `g_new` PARK as health Y. `g_has_checking` is 99.1% the connection hole.

## Headline

CLOSE leftover-after-days rank 0.428 (OLS 0.562, fake=False). Y3 n_accounts 0.581 vs days 0.711 vs size 0.617 vs checking 0.490 vs f_n_types 0.578. SIZE=False twin_gate=False (twin of g_n_banks ρ 0.880). Inverse days-after-n_accounts 0.707. Rise-only 1561/0 CONFIRM. Checking hole 99.1% CONFIRM; leftover after checking 0.586 / after days+checking 0.430. Dark/ERP last p50=3/3 CONFIRM. Card: **CLOSE unused leftover** / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_n_accounts (connection clock). Access ≠ ERP. Dark p50=3 = ERP p50=3. |
| 2 | Who is improving? | leftover after days 0.428. Rise-only is more connections, not 45→65. |
| 3 | Who is turning? | **CLOSE unused leftover** vs days 0.711. `g_new` stays PARK. |
| 4 | Dip vs fall? | Rise-only 1561/0 — no drop. |
| 5 | Why did it change? | after checking 0.586; hole 99.1%. |
| 6 | Months earlier? | lag1 leftover 0.460; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `g_n_accounts` as Y3 X / 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.428 dies (OLS 0.562 fake=False). Rise-only 1561/0. DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER. |
| `g_n_accounts` as engine X on the 44 | **DROP** | leftover lives=False twin=False SIZE=False beat-size=False |
| rise-only inventory | **YES** | 1561/0 |
| rewrite of checking hole | **NO** | leftover after checking 0.586 hole=99.1% |
| `y_n_accounts` | **PARK** | do not invent |
| Q6 lag1 / TURNOVER | **CLOSE** | leftover 0.460; do not grow 0.720 |
| `g_new` as health Y | **PARK (locked)** | do not reopen banking_g |

## 1 — Coverage; twin / SIZE

Train 1,214 co / 21,157 CM. g_n_accounts nn=21,157 cov 100.0% eq0 12.4% p50=2.000. Dark panel p50=2.000 ERP panel p50=2.000 (last-month p50=3 is cut 7). ρ vs days 0.406 vs a_n_tx 0.404 vs checking 0.576 vs f_n_types 0.460 vs facilities 0.464 vs size 0.421. SIZE=False gate_twins=none all_twins=['g_n_banks'].

| col | n_nn | cov | eq0 | p50 |
| --- | --- | --- | --- | --- |
| g_n_accounts | 21,157 | 100.0% | 12.4% | 2.000 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.406 | no |
| a_n_tx | 0.404 | no |
| g_has_checking | 0.576 | no |
| g_has_card | 0.401 | no |
| g_has_tpv | 0.089 | no |
| g_has_saving | 0.019 | no |
| g_has_investment | 0.289 | no |
| g_n_banks | 0.880 | TWIN |
| g_n_types | 0.679 | no |
| f_n_types | 0.460 | no |
| f_n_facilities | 0.464 | no |
| log1p(a_in3) | 0.421 | no |


## 2 — Single-feature group-fold Y3

Y3 g_n_accounts 0.581 n=5,648 pos=402 (banking_g oriented 0.585 CONFIRM). vs size 0.617 vs days 0.711 vs checking 0.490 vs f_n_types 0.578 (peek 0.578 CONFIRM). Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.036 FAIL.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | g_n_accounts | 5,648 | 402 | 0.581 | 0.585 | 0.092 | -1 | 0.630 0.599 0.640 0.420 0.616 |
| y3_recover_cash_6m | g_has_checking | 5,648 | 402 | 0.490 | 0.506 | 0.035 | -1 | 0.461 0.461 0.538 0.477 0.515 |
| y3_recover_cash_6m | g_n_banks | 5,648 | 402 | 0.612 | 0.615 | 0.068 | -1 | 0.678 0.601 0.662 0.505 0.615 |
| y3_recover_cash_6m | g_n_types | 5,648 | 402 | 0.552 | 0.554 | 0.043 | -1 | 0.532 0.530 0.601 0.503 0.594 |
| y3_recover_cash_6m | f_n_types | 5,648 | 402 | 0.578 | 0.583 | 0.057 | -1 | 0.638 0.574 0.630 0.504 0.543 |
| y3_recover_cash_6m | f_n_facilities | 5,648 | 402 | 0.579 | 0.585 | 0.054 | -1 | 0.637 0.576 0.627 0.508 0.546 |
| y3_recover_cash_6m | g_new_this_month | 5,648 | 402 | 0.492 | 0.501 | 0.005 | 1 | 0.494 0.498 0.489 0.493 0.485 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

g_n_accounts leftover after days OLS 0.562 rank 0.428 fake=False almost=False ρ(resid,days)=-0.227 R²=0.144 honest_dies=True n=5,648 pos=402. Inverse: days leftover after n_accounts OLS 0.704 rank 0.707 dies=False.

OLS folds: 0.529 0.538 0.479 0.713 0.552. Rank folds: 0.451 0.478 0.414 0.309 0.490.

## 4 — Twin / SIZE (in cut 1)

SIZE=False twin_gate=False gate_twins=none.

## 5 — Rise-only

rises=1,561 drops=0 flats=18,382 (peek 1,561/0 CONFIRM). rise-only=True. Rise-month dummy Y3 0.493 leftover after days 0.684 dies=True.

## 6 — Leftover after g_has_checking

checking=0 and n_accounts=0: 99.1% (peek 99.1% CONFIRM). ρ(n_accounts, checking)=0.576. leftover after checking rank 0.586 dies=False rewrite=False. after days+checking 0.430.

| bar | OLS | rank | R2 | dies |
| --- | --- | --- | --- | --- |
| after checking | 0.583 | 0.586 | 0.083 | False |
| after days+checking | 0.566 | 0.430 | 0.214 | True |
| checking after n_accounts | 0.569 | 0.569 | 0.083 | False |


## 7 — Dark vs ERP

Dark 470 (want 470) last-month p50=3.000 ERP last p50=3.000 CONFIRM p50=3 both. Access ≠ ERP. Dark leftover 0.447 ERP leftover 0.399.

| book | n_co | p50 | last p50 | Y3 | leftover |
| --- | --- | --- | --- | --- | --- |
| dark | 470 | 2.000 | 3.000 | 0.593 | 0.447 |
| ERP | 744 | 2.000 | 3.000 | 0.594 | 0.399 |


## 8 — Q6 lag1 leftover after days_lag1

Y3 n_accounts_lag1 0.582 leftover after days_lag1 rank 0.460 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | g_n_accounts | 5,648 | 402 | 0.581 | 0.585 | 0.092 | -1 | 0.630 0.599 0.640 0.420 0.616 |
| y3_recover_cash_6m | g_n_accounts_lag1 | 5,648 | 402 | 0.582 | 0.587 | 0.084 | -1 | 0.625 0.593 0.648 0.436 0.605 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 9 — Ever-n vs this-month count

Ever n_accounts>0 companies 1209/1214. Ever-max Y3 0.618 leftover after days 0.419. Company-mean Y3 0.601 leftover 0.406. Month leftover after ever-max 0.532 dies=True.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=1073 cov=100.0% p50=2.000 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.399 p50=0.433 p95=0.545 share<0.55=96.2% n=80.

### leftover after f_n_types / facilities

n_accounts leftover after f_n_types 0.533 after facilities 0.531. f_n_types leftover after days 0.534 (peek 0.534 CONFIRM).

### Y3 at n=0 vs connected

Y3 n=0 8.2% n=466 vs connected 7.0% n=5182. zero-dummy Y3 0.491 leftover after days 0.674 dies=False.

| slice | n | n_pos | Y3 rate |
| --- | --- | --- | --- |
| n_accounts==0 | 466 | 38 | 8.2% |
| n_accounts>0 | 5182 | 364 | 7.0% |


### Y2 leftover

Y2 n_accounts 0.432 leftover after days 0.557 dies=False.

### leftover after g_n_banks

ρ(n_accounts, n_banks)=0.880 TWIN. leftover after banks 0.469 dies=True. after days+banks 0.550. g_n_banks Y3 0.612 leftover after days 0.542 dies=True.

### rise / zero dummy leftover

rise dummy leftover OLS 0.684 rank 0.684 fake=True ρ(resid,days)=-0.805. zero dummy leftover OLS 0.674 rank 0.674 fake=False ρ(resid,days)=0.744. High OLS leftover on dummies is a fake days leak unless rank also lives.

### leftover on connected months

connected-only Y3 0.596 n=5,182 leftover after days 0.428 dies=True.

### ICC / acf1

ICC=0.983 (quote 0.98 BETWEEN) median company Pearson acf1=0.804 (quote 0.80 CONFIRM) n_co=901.

### checking leftover after days

checking leftover after days rank 0.674 OLS 0.674 fake=False dies=False. zero dummy leftover after checking 0.491 dies=True. zero dummy leftover after days 0.674 fake=False ρ(resid,days)=0.744.

### fold-wise leftover

Rank leftover folds 0.451 0.478 0.414 0.309 0.490 — fold 4 0.420 single pulls leftover down; mean 0.428 dies.

| fold | rank leftover | n_va | n_pos |
| --- | --- | --- | --- |
| 0 | 0.451 | 1310 | 54 |
| 1 | 0.478 | 696 | 93 |
| 2 | 0.414 | 1072 | 66 |
| 3 | 0.309 | 1373 | 82 |
| 4 | 0.490 | 1197 | 107 |


### first6 / later / last-month leftover

first6 leftover 0.533 later 0.444 last-month —.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| first6 | 0.594 | 0.533 | 1844 | 118 | True |
| later | 0.580 | 0.444 | 3804 | 284 | True |
| last-month | — | — | 0 | 0 | False |


### last Y3-labeled month leftover

last Y3-labeled month Y3 0.574 leftover after days 0.455 dies=True n=725 pos=154.

### leftover after g_n_types / g_new

leftover after g_n_types 0.555 dies=False. after g_new 0.581 dies=False. after days+g_n_types 0.443.

### Y7 leftover after days

Y7 n_accounts 0.446 leftover after days 0.441 dies=True. Do not grow TURNOVER 0.720.

### leftover after size; ρ vs lag1

leftover after size 0.532 dies=True. after days+size 0.434 dies=True. ρ vs own lag1=0.930 TWIN — BETWEEN snapshot.

### leftover after a_n_tx / days+checking+banks

leftover after a_n_tx 0.430 dies=True. after days+checking+banks 0.550 dies=False. g_n_types leftover after days 0.564 dies=False.

### short vs long leftover

short leftover 0.520 long leftover 0.562.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| short_<12 | 0.597 | 0.520 | 4074 | 276 | True |
| long_>=12 | 0.525 | 0.562 | 1574 | 126 | False |


### long / n>=3 leftover

long leftover OLS 0.575 rank 0.562 fake=False. n>=3 Y3 0.582 leftover 0.617 dies=False. after days+checking+banks rank 0.550 OLS 0.609 fake=False.

### n>=3 vs KEEP-as-X gate

n>=3 Y3 0.582 n=2,798 pos=164 vs size 0.556 vs days 0.721. leftover after days 0.617 fake=False. after days+size 0.620. after banks 0.599. beat-size FAIL — leftover on n>=3 does not pass KEEP-as-X.

### n>=3 leftover after days+banks

n>=3 leftover after days+banks 0.626 dies=False fake=False. Twin leftover of banks on the n>=3 slice.

### count-bin leftover after days

Count-bin leftover after days: 0 is the hole dummy; 5+ leftover is not a KEEP seat.

| bin | Y3 | leftover | fake | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.491 | 0.674 | False | 5648 | 402 |
| 1-2 | 0.546 | 0.557 | False | 5648 | 402 |
| 3-4 | 0.463 | 0.641 | False | 5648 | 402 |
| 5+ | 0.550 | 0.577 | False | 5648 | 402 |


### leftover of n_accounts − n_banks

n_accounts−n_banks Y3 0.544 leftover after days 0.552 dies=False fake=False. after days+banks 0.534.

### leftover on ever-rise vs always-flat

ever-rise leftover 0.438 always-flat leftover 0.428. Rise-clock companies do not unlock leftover after days.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| ever_rise | 0.571 | 0.438 | 4048 | 258 | True |
| always_flat | 0.583 | 0.428 | 1600 | 144 | True |


### leftover of log1p(n_accounts)

log1p(n_accounts) Y3 0.581 leftover after days 0.428 dies=True fake=False. Transform does not revive leftover.

### leftover of banks after days+n_accounts

banks leftover after days+n_accounts 0.548 dies=True fake=False. after days 0.542. after n_accounts 0.579. Banks has no leftover once the count is in — cluster, not a second seat.

### leftover of month-to-month Δn

Δn Y3 0.492 leftover after days 0.684 dies=True fake=True. after days+count 0.631. Month shock of a rise-only clock is not leftover after days.

### leftover on varying vs constant companies

varying leftover 0.438 constant leftover 0.428. BETWEEN trait still dies after days.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| varying | 0.571 | 0.438 | 4048 | 258 | True |
| constant | 0.583 | 0.428 | 1600 | 144 | True |


### leftover after days+f_n_types / n>=5

leftover after days+f_n_types 0.446 dies=True. n>=3 leftover after days+checking 0.597. after days+f_n_types+banks 0.551. n>=5 Y3 0.592 leftover 0.693 dies=False beat-size FAIL.

### n>=5 leftover vs KEEP-as-X

n>=5 Y3 0.592 n=1,574 pos=77 vs size 0.453 vs days 0.745. leftover after days 0.693 fake=False. after days+banks 0.469. after days+size 0.637. after days+checking 0.678. beat-size FAIL — n>=5 leftover does not pass KEEP-as-X vs locked size 0.617.

### leftover on days>0 / n_accounts÷n_banks

days>0 leftover 0.434 Y3 0.576 n=5,536 dies=True. ratio n/banks Y3 0.472 leftover 0.558 dies=False. leftover after days+size+checking 0.444.

### leftover after days + other g_has_*

leftover after days+all g_has_* 0.432 dies=True. g_has_* already DROP; count leftover after each still dies or is days.

| control | ctrl_Y3 | leftover_after_days+ctrl | dies |
| --- | --- | --- | --- |
| g_has_checking | 0.490 | 0.430 | True |
| g_has_card | 0.551 | 0.442 | True |
| g_has_tpv | 0.498 | 0.426 | True |
| g_has_saving | 0.499 | 0.428 | True |
| g_has_investment | 0.509 | 0.421 | True |


### checking leftover after days+n_accounts

checking leftover after days 0.674; after days+n_accounts 0.459 dies=True fake=False. checking=1 leftover 0.428 Y3 0.596 n=5,178 pos=364 beat-size FAIL.

### leftover on ever-max>=5 companies

ever-max>=5 Y3 0.556 n=2,199 pos=110 vs size 0.453 vs days 0.678. leftover after days 0.606 dies=False. after days+banks 0.581. beat-size FAIL.

### leftover excluding fold 4 / banks>=2

ex-fold4 Y3 0.621 leftover 0.542 dies=True n=4,275. banks>=2 leftover 0.611 Y3 0.545 n=3,467. Weak fold does not hide a KEEP leftover.

### leftover of g_n_types / g_new after n_accounts

g_n_types leftover after days+n_accounts 0.537 dies=True. g_new leftover after days+n_accounts 0.623 dies=False. count leftover after days+types+banks 0.556. g_new=1 leftover — g_new=0 leftover 0.433.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| g_new=1 | — | — | 243 | 15 | False |
| g_new=0 | 0.582 | 0.433 | 5193 | 369 | True |


### leftover on g_n_types>=2 / last-3 / banks>=2+size

g_n_types>=2 leftover — Y3 — n=1,229. last-3 labeled leftover 0.417 Y3 0.579 n=1,762. banks>=2 leftover after days+size 0.622 Y3 0.545 beat-size FAIL.

### leftover on single-bank / never-zero / ever-hole

single-bank leftover 0.619 never-zero 0.532 ever-hole 0.459 n_1_2 0.587 chk1_n>=3 0.616.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| single_bank | 0.441 | 0.619 | 1715 | 192 | False |
| never_zero | 0.631 | 0.532 | 3085 | 189 | True |
| ever_hole | 0.524 | 0.459 | 2590 | 213 | True |
| n_1_2 | 0.566 | 0.587 | 2384 | 200 | False |
| chk1_n>=3 | 0.582 | 0.616 | 2794 | 164 | False |


### leftover after days+tx+size / by year

leftover after days+a_n_tx+size 0.435 dies=True. 2024 leftover — 2025 leftover 0.464 2026 leftover 0.574.

| year | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| 2024 | — | — | 485 | 22 | False |
| 2025 | 0.594 | 0.464 | 4208 | 305 | True |
| 2026 | 0.525 | 0.574 | 955 | 75 | False |


### never-zero leftover vs KEEP-as-X

never-zero Y3 0.631 n=3,085 pos=189 vs same-n size 0.681 vs days 0.733. leftover after days 0.532 dies=True. after days+size 0.431. after days+banks 0.522. beat locked-size FAIL beat same-n size FAIL — leftover still dies, not KEEP.

### leftover on large vs small / first labeled

small leftover 0.526 large leftover 0.429 first-labeled leftover 0.459. leftover after days+checking+size+banks 0.548 dies=True.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| small | 0.557 | 0.526 | 2764 | 255 | True |
| large | 0.564 | 0.429 | 2764 | 136 | True |
| first_labeled | 0.546 | 0.459 | 725 | 91 | True |


### leftover on high/low days / card=1 / ge3∩never0

low-days leftover 0.446 high-days 0.486 card=1 — ge3∩never0 0.624. leftover after days+facilities+banks 0.551. 1:1 accounts=banks Y3 0.540 leftover 0.566 dies=False.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| low_days | 0.519 | 0.446 | 2977 | 319 | True |
| high_days | 0.550 | 0.486 | 2671 | 83 | True |
| card=1 | — | — | 981 | 33 | False |
| ge3_never0 | 0.563 | 0.624 | 1633 | 77 | False |


### n>=3 ∩ never-zero leftover vs KEEP-as-X

ge3∩never0 Y3 0.563 n=1,633 pos=77 vs size 0.577 vs days 0.677. leftover after days 0.624 after days+banks 0.642 after days+size 0.642. beat-size FAIL. >=6 labeled leftover 0.425 Y3 0.573 n=4,869.

### leftover on 2026 / saving=1 / rise∩n>=3

2026 leftover 0.574 after banks 0.575. saving=1 leftover —. rise∩n>=3 leftover 0.622 after banks 0.610. leftover after days+checking+a_n_tx 0.435.

| slice | Y3 | leftover | after_banks | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| 2026 | 0.525 | 0.574 | 0.575 | 955 | 75 |
| saving=1 | — | — | — | 9 | 0 |
| rise_ge3 | 0.571 | 0.622 | 0.610 | 2204 | 117 |


### leftover on invest=1 / sb∩n>=2 / types=1

invest=1 leftover — sb∩n>=2 0.632 types=1 0.425. leftover after days+types+checking 0.437.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| invest=1 | — | — | 328 | 13 | False |
| sb_n>=2 | 0.451 | 0.632 | 601 | 65 | False |
| types=1 | 0.572 | 0.425 | 3953 | 315 | True |


### leftover on banks>=3 / tx split / full G cluster

banks>=3 leftover 0.664 low-tx 0.549 high-tx 0.555. leftover after days+banks+types+checking 0.558 dies=False.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| banks>=3 | 0.555 | 0.664 | 2113 | 78 | False |
| low_tx | 0.441 | 0.549 | 2834 | 308 | True |
| high_tx | 0.585 | 0.555 | 2814 | 94 | False |


### banks>=3 leftover vs KEEP-as-X

banks>=3 Y3 0.555 n=2,113 pos=78 vs same-n size 0.569 vs days 0.716. leftover after days 0.664 after days+size 0.658 after days+banks 0.539. beat locked-size FAIL beat same-n FAIL — slice leftover is not KEEP-as-X.

### leftover on Q1 vs rest / days+facilities+size

Q1 leftover 0.530 Q2-4 leftover 0.412. leftover after days+facilities+size 0.447 dies=True.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| Q1 | 0.562 | 0.530 | 1803 | 120 | True |
| Q2-4 | 0.587 | 0.412 | 3845 | 282 | True |


### leftover on H1 vs H2 / kitchen-sink controls

H1 leftover 0.455 H2 leftover 0.430. leftover after days+tx+checking+size 0.444 dies=True.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| H1 | 0.571 | 0.455 | 2818 | 190 | True |
| H2 | 0.588 | 0.430 | 2830 | 212 | True |


### leftover on group size / first-year

singleton leftover — multi-group 0.425 first-2024 0.424 first-2025 0.531.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| singleton_group | — | — | 240 | 36 | False |
| multi_group | 0.589 | 0.425 | 5408 | 366 | True |
| first_2024 | 0.583 | 0.424 | 4079 | 276 | True |
| first_2025 | 0.581 | 0.531 | 1554 | 123 | True |


## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| e_ar_open leftover | 0.527 DROP |
| f_n_types leftover | 0.534 DROP |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `g_n_accounts` on the 15-col card.

## Files written

- `analysis/evaluate/n_accounts_qa.py`
- `analysis/outputs/n_accounts_qa.md`
- `analysis/outputs/n_accounts_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_n_accounts.md` (end, if WRITE_WAVE)

Elapsed 59s. Failed: none.

