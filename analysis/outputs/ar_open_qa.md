# Unused leftover of `e_ar_open` after days as Y3 X

Generated `2026-09-19T06:49:58+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ar_open`. Do not put e_ar_open on the 15-col card. Do not overwrite `dso_qa.*`, `issued_qa.*`, `pending_qa.*`, `ap_open_qa.*`. Y5 never E. Do not grow TURNOVER. Dark 470 stay NaN not 0.

`e_ar_open` = unpaid AR |amount| stock at period end. `e_dso_proxy` = open / this-period issued (already DROP leftover 0.474). `e_ar_issued` CLOSE leftover 0.608 (fake days clone).

## Headline

CLOSE leftover-after-days rank 0.527 (OLS 0.719, fake=True). Y3 open 0.587 vs days 0.711 vs size 0.617 vs issued 0.687 vs DSO 0.564 vs ap_open 0.569. SIZE=False twin=False. Inverse days-after-open 0.706. Leftover after issued 0.433 (rewrite). Leftover after DSO 0.557. Zero-stock Y3 12.6% vs 5.4% is a days rewrite. Card: **CLOSE unused leftover** / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_ar_open. Dark 470 = NaN. |
| 2 | Who is improving? | leftover after days 0.527. |
| 3 | Who is turning? | **CLOSE unused leftover** vs days 0.711. Do not claim TURNOVER. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | after issued 0.433; after DSO 0.557. |
| 6 | Months earlier? | lag1 leftover 0.529; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `e_ar_open` as Y3 X / 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.527 dies (OLS 0.719 fake=True). Issued rewrite leftover 0.433. DROP from the 44 as Y3 X. Off the 15-col card. Do not grow TURNOVER. |
| `e_ar_open` as engine X on the 44 | **DROP** | leftover lives=False twin=False SIZE=False beat-size=False |
| rewrite of issued | **YES** | leftover after issued 0.433 ρ=0.697 |
| DSO numerator | **NO** | leftover after DSO 0.557 |
| vs `e_ap_open` | **NO** | ρ=0.682 |
| `y_ar_open` | **PARK** | do not invent |
| Q6 lag1 / TURNOVER | **CLOSE** | leftover 0.529; do not grow 0.720 |

## 1 — Coverage; twin / SIZE

Train 1,214 co / 21,157 CM. e_ar_open nn=13,554 cov 64.1%. Dark 470 nn=0 zero=0 CONFIRM NaN. ρ vs days 0.347 vs a_n_tx 0.385 vs issued 0.697 vs DSO 0.600 vs pending 0.271 vs ap_open 0.682 (peek 0.682 CONFIRM) vs size 0.431. SIZE=False gate_twins=none all_twins=none.

| col | n_nn | cov | eq0 | p50 |
| --- | --- | --- | --- | --- |
| e_ar_open | 13,554 | 64.1% | 27.5% | 43377.255 |
| e_ar_issued | 13,555 | — | — | — |
| e_dso_proxy | 8,583 | — | — | — |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | 0.347 | no |
| a_n_tx | 0.385 | no |
| e_ar_issued | 0.697 | no |
| e_dso_proxy | 0.600 | no |
| e_pending_amt_share | 0.271 | no |
| e_ap_open | 0.682 | no |
| log1p(a_in3) | 0.431 | no |


## 2 — Single-feature group-fold Y3

Y3 e_ar_open 0.587 n=3,618 pos=264 (ap_open peek 0.587 CONFIRM). vs size 0.617 vs days 0.711 vs issued 0.687 (peek 0.687 CONFIRM) vs DSO 0.564 vs ap_open 0.569 (sibling 0.569). Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=-0.030 FAIL.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_ar_open | 3,618 | 264 | 0.587 | 0.595 | 0.101 | -1 | 0.544 0.738 0.547 0.476 0.629 |
| y3_recover_cash_6m | log1p(e_ar_open) | 3,618 | 264 | 0.587 | 0.595 | 0.101 | -1 | 0.544 0.738 0.547 0.476 0.629 |
| y3_recover_cash_6m | e_ar_issued | 3,618 | 264 | 0.687 | 0.685 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.552 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y3_recover_cash_6m | e_ap_open | 3,618 | 264 | 0.569 | 0.570 | 0.109 | -1 | 0.638 0.674 0.461 0.443 0.632 |
| y3_recover_cash_6m | e_pending_amt_share | 3,446 | 239 | 0.484 | 0.523 | 0.102 | 1 | 0.571 0.458 0.381 0.609 0.399 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

e_ar_open leftover after days OLS 0.719 rank 0.527 fake=True almost=True ρ(resid,days)=-0.817 R²=0.000 honest_dies=True n=3,618 pos=264. log1p leftover rank 0.527 dies=True. Inverse: days leftover after ar_open OLS 0.729 rank 0.706 dies=False.

OLS folds: 0.722 0.776 0.628 0.768 0.699. Rank folds: 0.495 0.655 0.530 0.406 0.547.

## 4 — Twin / SIZE (in cut 1)

SIZE=False twin_gate=False gate_twins=none.

## 5 — Leftover after issued

open leftover after issued rank 0.433 dies=True R²=0.422 ρ=0.697 rewrite=True. after days+issued 0.434 dies=True. issued leftover after open 0.678 (issued leftover after days was 0.608 fake).

| bar | OLS | rank | ρ(resid,bar) | R2 | dies |
| --- | --- | --- | --- | --- | --- |
| after issued | 0.611 | 0.433 | -0.422 | 0.422 | True |
| after days+issued | 0.578 | 0.434 | 0.568 | 0.422 | True |
| issued after open | 0.668 | 0.678 | 0.183 | 0.422 | False |


## 6 — Leftover after DSO

open leftover after DSO rank 0.557 dies=False R²=0.000 ρ=0.600 numerator=False. after days+DSO 0.423. DSO leftover after open 0.576. DSO leftover after days OLS 0.474 rank 0.569 (peek 0.474 is OLS CONFIRM).

| bar | OLS | rank | R2 | dies |
| --- | --- | --- | --- | --- |
| after DSO | 0.453 | 0.557 | 0.000 | False |
| after days+DSO | 0.655 | 0.423 | 0.000 | True |
| DSO after open | 0.569 | 0.576 | 0.000 | False |
| DSO after days | 0.474 | 0.569 | 0.000 | False |


## 7 — Dark 470 stay NaN; ERP leftover

Dark 470 (want 470) ar_open nn=0 CONFIRM NaN not 0. ERP Y3 0.587 leftover after days 0.527 dies=True.

| book | n_co | nn | Y3 n | Y3 pos | CV |
| --- | --- | --- | --- | --- | --- |
| dark | 470 | 0 | 0 | 0 | LOW_POWER |
| ERP | 744 | 13554 | 3618 | 264 | 0.587 |


## 8 — Q6 lag1 leftover after days_lag1

Y3 ar_open_lag1 0.582 leftover after days_lag1 rank 0.529 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM). Y7 leftover after issued_lag1 0.456 (report-only; do not claim TURNOVER).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_ar_open | 3,618 | 264 | 0.587 | 0.595 | 0.101 | -1 | 0.544 0.738 0.547 0.476 0.629 |
| y3_recover_cash_6m | e_ar_open_lag1 | 3,618 | 264 | 0.582 | 0.587 | 0.107 | -1 | 0.541 0.752 0.546 0.466 0.603 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.669 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |


## 9 — vs `e_ap_open`

ρ(ar_open, ap_open)=0.682 (peek 0.682 different object). Y3 ap_open 0.569 (sibling leftover after days 0.418 — not overwritten). ar leftover after ap 0.573 dies=False. sibling file present=True.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=582 cov=54.2% p50=132888.770 dark nn=0 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.397 p50=0.523 p95=0.578 share<0.55=70.0% n=80.

### leftover after pending / days+issued+DSO

open leftover after pending 0.579 dies=True. after days+issued+DSO 0.603 dies=True.

### leftover after size

open leftover after size 0.535 dies=True. after days+size 0.516 dies=True.

### Y3 quintiles

Y3 ar_open Q1→Q5 ['10.3%', '4.3%', '5.5%', '5.9%'].

| q | Y3 rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 10.3% | 1450 | 150 |
| 2 | 4.3% | 721 | 31 |
| 3 | 5.5% | 723 | 40 |
| 4 | 5.9% | 724 | 43 |


### Y2 leftover

Y2 ar_open 0.572 leftover after days 0.532 dies=True.

### DSO leftover OLS vs rank (peek 0.474)

DSO leftover after days OLS 0.474 rank 0.569 fake=False (peek 0.474 is OLS CONFIRM). Rank leftover of DSO is not the locked quote.

### Y3 at open==0 vs positive

Y3 open==0 12.6% n=963 vs open>0 5.4% n=2655. zero-dummy Y3 0.596 leftover after days 0.479 dies=True.

| slice | n | n_pos | Y3 rate |
| --- | --- | --- | --- |
| open==0 | 963 | 121 | 12.6% |
| open>0 | 2655 | 143 | 5.4% |


### issued leftover after days (same-n)

Same-n n=3618: issued leftover after days rank 0.608 OLS 0.694 fake=True (peek 0.608 CONFIRM). open leftover same-n 0.527 OLS 0.719 fake=True. Open is weaker than the issued fake-days clone.

### fold-wise leftover after days

Rank leftover folds 0.495 0.655 0.530 0.406 0.547 — fold 1 0.655 lives, fold 4 0.406 dies; mean 0.527 dies.

| fold | rank leftover | n_va | n_pos |
| --- | --- | --- | --- |
| 0 | 0.495 | 575 | 39 |
| 1 | 0.655 | 306 | 47 |
| 2 | 0.530 | 793 | 46 |
| 3 | 0.406 | 995 | 56 |
| 4 | 0.547 | 949 | 76 |


### ICC / acf1

ICC=0.990 (quote 0.99 BETWEEN) acf1=0.944 (quote 0.77) k=744 n=13554. BETWEEN / sticky stock — not a month-to-month lead.

### leftover after pending (honest rank)

open leftover after pending rank 0.579 OLS 0.473 ρ(resid,pending)=-0.890 resid_twin=True. Honest rank lives — leftover_diag fake flag is residual-twin of pending, not a days leak.

### median company Pearson acf1

median company Pearson acf1=0.770 n_co=657 (quote 0.77 CONFIRM). Pooled Spearman lag was 0.944 — not the report acf.

### leftover on open>0 only

open>0 only Y3 0.420 n=2,655 pos=143 leftover after days rank 0.586 dies=True.

### Y7 leftover after days (no TURNOVER)

Y7 ar_open 0.465 leftover after days 0.472 dies=True. leftover after issued_lag1 0.456. Do not grow TURNOVER 0.720.

### p99-clipped leftover

p99=199497396 clip Y3 0.587 leftover after days 0.527 dies=True — tail is not the leftover.

### Q1 dummy leftover

Q1 dummy Y3 0.595 leftover after days 0.458 dies=True. Low-open spike is a days rewrite, not a stock dummy.

### open>0 leftover rank vs fake-days

open>0 leftover after days OLS 0.717 rank 0.586 fake=True ρ(resid,days)=-0.817. Honest rank lives — leftover_diag dies=True because OLS residual still twins days. Inverse days after open 0.713. Thin leftover on positive stock still fails beat-size (Y3 0.420).

### Q6 lag3 leftover

Y3 ar_open_lag3 0.555 leftover after days_lag3 0.441 dies=True. No TURNOVER seat.

### first6 vs later leftover

first6 leftover 0.564 later leftover 0.432 — stock does not lead on early books.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| first6 | 0.597 | 0.564 | 1154 | 73 | True |
| later | 0.584 | 0.432 | 2464 | 191 | True |


### ρ vs own lag1

ρ(e_ar_open, lag1)=0.944 twin=True — BETWEEN stock (ICC 0.99). Not a lead, a snapshot.

### short vs long leftover

short leftover 0.560 long leftover 0.480.

| slice | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| short_<12 | 0.618 | 0.560 | 2565 | 172 | True |
| long_>=12 | 0.473 | 0.480 | 1053 | 92 | True |


### DSO leftover after issued / open after days+issued+size

DSO leftover after issued rank 0.564 dies=False. open leftover after days+issued+size 0.438 dies=True.

### is open==0 just low-days?

Y3-labeled days p50 open==0 11.000 vs open>0 18.000. ρ(zero-dummy, days)=-0.317 — zero stock is thinner activity, not a leftover.

| slice | n | days mean | days p50 | Y3 |
| --- | --- | --- | --- | --- |
| open==0 | 963 | 12.441 | 11.000 | 12.6% |
| open>0 | 2655 | 17.471 | 18.000 | 5.4% |


### leftover after days+ap_open

open leftover after days+ap_open rank 0.545 dies=True. Thin leftover after ap alone (0.573) is days.

### leftover after delay / overdue

open leftover after delay 0.512 dies=True ρ=0.097. after overdue 0.420 dies=True ρ=-0.166. after days+delay 0.478.

### leftover by origin month

Origin leftover after days — no month keeps a living leftover with power.

| origin | n | n_pos | leftover |
| --- | --- | --- | --- |
| 2024-10 | 48 | 2 | LOW_POWER |
| 2024-11 | 208 | 11 | LOW_POWER |
| 2024-12 | 229 | 9 | LOW_POWER |
| 2025-01 | 255 | 9 | LOW_POWER |
| 2025-02 | 269 | 14 | LOW_POWER |
| 2025-03 | 324 | 22 | LOW_POWER |
| 2025-04 | 328 | 20 | LOW_POWER |
| 2025-05 | 336 | 22 | LOW_POWER |
| 2025-06 | 351 | 28 | LOW_POWER |
| 2025-07 | 392 | 37 | LOW_POWER |
| 2025-08 | 362 | 30 | LOW_POWER |
| 2025-09 | 368 | 24 | LOW_POWER |
| 2025-10 | 386 | 31 | LOW_POWER |
| 2025-11 | 401 | 33 | LOW_POWER |
| 2025-12 | 436 | 35 | LOW_POWER |
| 2026-01 | 461 | 36 | LOW_POWER |
| 2026-02 | 494 | 39 | LOW_POWER |


### company-median BETWEEN leftover

company-median open Y3 0.608 leftover after days 0.551 dies=True. BETWEEN trait is not a leftover after days.

### leftover by size tercile

Size-tercile leftover after days: T1 0.560, T2 —, T3 0.400.

| tercile | Y3 | leftover | n | n_pos | dies |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.606 | 0.560 | 1220 | 142 | True |
| 2 | — | — | 1200 | 46 | True |
| 3 | 0.553 | 0.400 | 1123 | 69 | True |


### rebuilt DSO leftover after days

open/issued Y3 0.564 leftover after days 0.569 dies=False ρ vs DSO 1.000 (should be 1). Rebuilt DSO leftover after days OLS 0.474 — DSO quote 0.474 is OLS.

### Δopen leftover after days

Δopen Y3 0.581 leftover after days 0.523 dies=True. Month-to-month stock change is not a leftover.

### leftover after a_n_tx / AP issued / overdue_30

open leftover after a_n_tx 0.515 dies=True. after e_ap_issued 0.530 dies=True. after overdue_30 0.416 dies=True. overdue_30 Y3 0.629.

### leftover on last 6 Y3 origins

last6 origins Y3 0.548 leftover after days 0.471 dies=True n=1580 pos=135.

### leftover after fx / credit-note; intensity

open leftover after fx 0.593 dies=False. after credit-note 0.569 dies=False. open/a_in3 Y3 0.524 leftover after days 0.435 dies=True.

### leftover after days+fx / days+credit-note

open leftover after days+fx 0.541 dies=True. after days+credit-note 0.531 dies=True. overdue_30 leftover after days 0.596 (report-only; not this seat).

### first6 / T1 leftover after issued

first6 leftover after issued 0.563 dies=False. T1 leftover after issued 0.558 dies=False. Thin first6 leftover after days is still issued volume.

### first6 / T1 leftover after days+issued

first6 leftover after days+issued 0.535 dies=True. T1 leftover after days+issued 0.538 dies=True.

## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| j_pay_match leftover | 0.556 KEEP-Q5 |
| e_ar_issued leftover | 0.608 CLOSE fake |
| e_dso_proxy leftover | 0.474 DROP |
| e_ap_open leftover | 0.418 CLOSE |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `e_ar_open` on the 15-col card.

## Files written

- `analysis/evaluate/ar_open_qa.py`
- `analysis/outputs/ar_open_qa.md`
- `analysis/outputs/ar_open_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_ar_open.md` (end, if WRITE_WAVE)

Elapsed 33s. Failed: none.

