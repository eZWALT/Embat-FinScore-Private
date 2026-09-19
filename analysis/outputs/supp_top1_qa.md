# Unused leftover of `d_supp_top1` after days as Y3 X

Generated `2026-09-19T06:25:02+02:00` by agent `572fb928`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_supp_top1`. Do not put supp_top1 on the 15-col card. Do not overwrite `supp_hhi_qa.*`, `top1_qa.*`, `n_supp_qa.*`. Y5 never E. Do not grow TURNOVER. Javier concentration is **top1**, not HHI.

`d_supp_top1` = share of AP invoice |amount| from the single largest supplier (Family D, trailing 6m). Incomplete 6m books are NaN. Dark 470 stay NaN not 0. `d_supp_hhi` DROP as weaker rewrite (ρ 0.987). `d_cust_top1` DROP leftover 0.525 (ρ 0.117 — different object).

## Headline

`d_supp_top1` as Y3 X: **CLOSE unused leftover** (unused leftover after days: honest rank 0.429 dies (OLS 0.427 fake=False). Also TWIN of ['d_supp_hhi']. DROP from the 44 as Y3 X. Y5 protective-tail footnote KEEP. Do not invent y_supp_top1. Off the 15-col card.). Y3 leftover after days OLS 0.427 rank 0.429 (dies, fake=False). Inverse days after supp_top1 rank 0.718. Single 0.641 vs size 0.617 vs days 0.711 vs HHI 0.653 vs n_supp 0.699 vs cust_top1 0.590. after HHI OLS 0.403 / rank 0.555 (rewrite leftover dies on OLS, R²=0.955) after n_supp 0.396 after days+HHI 0.424. Bootstrap leftover-after-days 0.363 / 0.438 / 0.600. Y5 footnote **KEEP**. 15-col card: no — do not put d_supp_top1 on the 15-col card. PARK as Y — do not invent y_supp_top1. Night quotes unchanged: Y3 0.762 / 0.752, days 0.711, size 0.617, TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | PARK as Y — do not invent y_supp_top1. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Q6 lag1 leftover after days_lag1 0.478. |
| 3 | Who is turning? | **CLOSE unused leftover** leftover after days 0.429 vs days 0.711. |
| 4 | Dip vs fall? | Not this table. |
| 5 | Why did it change? | Twin screen: ['d_supp_hhi']. Y5 protective tail KEEP. |
| 6 | Months earlier? | lag1 leftover 0.478; days_lag1 0.684. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| `d_supp_top1` as Y3 X / the 15-col card | **CLOSE unused leftover** | unused leftover after days: honest rank 0.429 dies (OLS 0.427 fake=False). Also TWIN of ['d_supp_hhi']. DROP from the 44 as Y3 X. Y5 protective-tail footnote KEEP. Do not invent y_supp_top1. Off the 15-col card. |
| `d_supp_top1` as engine X on the 44 | **DROP** | leftover lives=False twin=True SIZE=False beat-size=True |
| Y5 protective tail HHI>0.975 | **KEEP** | 2.7% vs 8.6%; top1 tail 2.1% |
| `y_supp_top1` | **PARK** | do not invent a concentration Y |
| twin of HHI / n_supp | 0.987 / -0.661 | leftover after HHI 0.555 after n_supp 0.396 |
| same object as `d_cust_top1` | **NO** | ρ=0.117 |
| Q6 lag1 after days_lag1 | **CLOSE** | leftover 0.478 |

## 1 — Coverage; twin / SIZE screen

Train 1,214 co / 21,157 CM. d_supp_top1 cov 50.1% acf1=0.711. Dark 470 nn=0 zero=0 CONFIRM NaN. Calendar incomplete nn=0 CONFIRM NaN. ρ vs days -0.402 vs a_n_tx -0.421 vs HHI 0.987 (peek 0.987 CONFIRM) vs n_supp -0.661 vs cust_top1 0.117 (peek 0.117 CONFIRM) vs size -0.323. SIZE=False twins=['d_supp_hhi'] twin_gate=True.

| col | n_nn | cov | acf1 |
| --- | --- | --- | --- |
| d_supp_top1 | 10,595 | 50.1% | 0.711 |
| d_supp_hhi | 10,595 | 50.1% | 0.745 |
| d_n_supp | 11,293 | 53.4% | 0.906 |


| vs | rho | flag |
| --- | --- | --- |
| c_n_days_with_tx | -0.402 | no |
| a_n_tx | -0.421 | no |
| d_supp_hhi | 0.987 | TWIN |
| d_n_supp | -0.661 | no |
| d_cust_top1 | 0.117 | no |
| log1p(a_in3) | -0.323 | no |


## 2 — Single-feature group-fold Y3

Y3 d_supp_top1 0.641 n=2,877 pos=203 (peek 0.641 / 2,877 / 203 CONFIRM). vs size 0.617 vs days 0.711 vs HHI 0.653 vs n_supp 0.699 vs cust_top1 0.590. Replica days 0.711 CONFIRM / size 0.617 CONFIRM. Beat-size Δ=0.024 PASS.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_supp_top1 | 2,877 | 203 | 0.641 | 0.605 | 0.102 | 1 | 0.787 0.702 0.561 0.613 0.542 |
| y3_recover_cash_6m | d_supp_hhi | 2,877 | 203 | 0.653 | 0.623 | 0.091 | 1 | 0.789 0.689 0.573 0.643 0.570 |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.685 | 0.116 | -1 | 0.820 0.722 0.514 0.764 0.674 |
| y3_recover_cash_6m | d_cust_top1 | 2,485 | 141 | 0.590 | 0.590 | 0.116 | 1 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.620 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.723 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.714 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |


## 3 — Honest leftover after days

d_supp_top1 leftover after days OLS 0.427 rank 0.429 (peek 0.429 CONFIRM) fake=False almost=False ρ(resid,days)=-0.016 R²=0.163 honest_dies=True n=2,877 pos=203. Inverse: days leftover after supp_top1 OLS 0.721 rank 0.718 dies=False.

OLS folds: 0.290 0.404 0.512 0.488 0.440. Rank folds: 0.295 0.403 0.521 0.492 0.436.

## 4 — Twin / SIZE screen (in cut 1)

SIZE=False twin_gate=True twins=['d_supp_hhi'].

## 5 — Leftover after HHI / n_supp / days+HHI

top1 leftover after HHI OLS 0.403 rank 0.555 dies=False (want die — rewrite R²=0.955; OLS leftover is the honest rewrite residual when R²≥0.95). after n_supp OLS 0.629 rank 0.396 dies=True; after days+HHI 0.424 dies=True; after cust_top1 0.606 dies=False. HHI after top1 OLS 0.525 rank 0.582 (supp_hhi leftover after top1 OLS 0.525 — not overwritten).

| bar | OLS | rank | ρ(resid,bar) | R2 | dies | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after HHI | 0.403 | 0.555 | 0.186 | 0.955 | False | 2877 | 203 |
| after n_supp | 0.629 | 0.396 | -0.517 | 0.130 | True | 2877 | 203 |
| after days+HHI | 0.404 | 0.424 | 0.011 | 0.955 | True | 2877 | 203 |
| after cust_top1 | 0.608 | 0.606 | 0.017 | 0.011 | False | 2464 | 138 |
| HHI after top1 | 0.525 | 0.582 | -0.112 | 0.955 | False | 2877 | 203 |


## 6 — Dark 470 stay NaN; ERP leftover

Dark 470 (want 470) top1 nn=0 CONFIRM NaN not 0. ERP Y3 0.641 leftover after days 0.429 dies=True.

| book | n_co | nn | Y3 n | Y3 pos | CV |
| --- | --- | --- | --- | --- | --- |
| dark | 470 | 0 | 0 | 0 | LOW_POWER |
| ERP | 744 | 10595 | 2877 | 203 | 0.641 |


## 7 — Q6 lag1 leftover after days_lag1

Y3 supp_top1_lag1 0.635 leftover after days_lag1 rank 0.478 dies=True. Days lag1 0.684 (quote 0.684 CONFIRM).

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_supp_top1 | 2,877 | 203 | 0.641 | 0.605 | 0.102 | 1 | 0.787 0.702 0.561 0.613 0.542 |
| y3_recover_cash_6m | d_supp_top1_lag1 | 2,695 | 195 | 0.635 | 0.600 | 0.096 | 1 | 0.761 0.713 0.571 0.585 0.544 |
| y3_recover_cash_6m | c_n_days_with_tx_lag1 | 5,648 | 402 | 0.684 | 0.694 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |


## 8 — Y5 leftover after size; protective tail

Y5 top1 0.530 leftover after size 0.473 dies=True (report-only; Y5 never E). leftover after days 0.454. HHI>0.975 Y5 2.7% n=73 vs rest 8.6% (quote 2.7% / 8.6% CONFIRM). top1>0.975 Y5 2.1% vs rest 8.7% same tail. Y5 protective-tail footnote KEEP.

| y | feature | n | n_pos | CV | train | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y5_ap_od30_ownp80 | d_supp_top1 | 4,902 | 418 | 0.530 | 0.518 | 0.043 | -1 | 0.526 0.565 0.523 0.464 0.571 |
| y5_ap_od30_ownp80 | d_supp_hhi | 4,902 | 418 | 0.539 | 0.526 | 0.044 | -1 | 0.536 0.575 0.533 0.470 0.579 |
| y5_ap_od30_ownp80 | log1p(a_in3) | 4,905 | 418 | 0.556 | 0.545 | 0.084 | 1 | 0.674 0.539 0.466 0.495 0.606 |


## 9 — vs `d_cust_top1`

ρ(supp_top1, cust_top1)=0.117 (peek 0.117 CONFIRM) different object. Y3 cust_top1 0.590. supp leftover after cust 0.606 dies=False.

## 10 — Holdout coverage only

Holdout 72 co / 1073 CM nn=527 cov=49.1% p50=0.483 dark nn=0 (no fit, no AUROC).

## Extras

### Bootstrap leftover after days

Bootstrap leftover-after-days rank p05=0.363 p50=0.438 p95=0.600 share<0.55=72.5% n=40.

### Permute within days quintile

Permuted-within-days leftover rank p50=0.522 p90=0.559 n=24.

### ICC / demean

ICC=0.959 TRAIT k=741. Demean leftover-days 0.524 dies=True. Company-mean leftover-days 0.581 dies=False.

### leftover after size / a_n_tx

top1 leftover after size 0.609 dies=False. after a_n_tx 0.431 dies=True. after days+size 0.474 dies=True.

### Y3 rate by supp_top1 quintile

Y3 supp_top1 Q1→Q5 ['3.1%', '5.2%', '9.9%', '6.6%', '10.4%'].

| q | Y3 rate | n | n_pos |
| --- | --- | --- | --- |
| 1 | 3.1% | 576 | 18 |
| 2 | 5.2% | 575 | 30 |
| 3 | 9.9% | 576 | 57 |
| 4 | 6.6% | 574 | 38 |
| 5 | 10.4% | 576 | 60 |


### HHI rewrite leftover

rewrite leftover after HHI OLS 0.403 rank 0.555 R²=0.955 ρ(resid,HHI)=0.186. OLS-dies=True rewrite_dead=True (rank 0.555 is chance on a near-collinear residual; do not KEEP it). OLS resid leftover after days 0.402 dies=True. (top1−HHI) leftover after days 0.430 after HHI 0.530.

### Y5 tail grid

Y5 tail grid: [{'stem': 'HHI', 'cut': 0.9, 'hi_rate': '2.5%', 'rest_rate': '8.7%', 'n_hi': 160, 'n_rest': 4742, 'protective': True}, {'stem': 'top1', 'cut': 0.9, 'hi_rate': '6.6%', 'rest_rate': '8.6%', 'n_hi': 289, 'n_rest': 4613, 'protective': True}, {'stem': 'HHI', 'cut': 0.95, 'hi_rate': '2.1%', 'rest_rate': '8.7%', 'n_hi': 96, 'n_rest': 4806, 'protective': True}, {'stem': 'top1', 'cut': 0.95, 'hi_rate': '2.0%', 'rest_rate': '8.7%', 'n_hi': 153, 'n_rest': 4749, 'protective': True}, {'stem': 'HHI', 'cut': 0.975, 'hi_rate': '2.7%', 'rest_rate': '8.6%', 'n_hi': 73, 'n_rest': 4829, 'protective': True}, {'stem': 'top1', 'cut': 0.975, 'hi_rate': '2.1%', 'rest_rate': '8.7%', 'n_hi': 94, 'n_rest': 4808, 'protective': True}, {'stem': 'HHI', 'cut': 0.99, 'hi_rate': '1.7%', 'rest_rate': '8.6%', 'n_hi': 59, 'n_rest': 4843, 'protective': True}, {'stem': 'top1', 'cut': 0.99, 'hi_rate': '2.9%', 'rest_rate': '8.6%', 'n_hi': 68, 'n_rest': 4834, 'protective': True}]. Overlap HHI>0.975 ∩ top1>0.975 both=73 only_HHI=0 only_top1=21. Same tail object.

| stem | cut | hi_rate | rest_rate | n_hi | n_rest | protective |
| --- | --- | --- | --- | --- | --- | --- |
| HHI | 0.9 | 2.5% | 8.7% | 160 | 4742 | True |
| top1 | 0.9 | 6.6% | 8.6% | 289 | 4613 | True |
| HHI | 0.95 | 2.1% | 8.7% | 96 | 4806 | True |
| top1 | 0.95 | 2.0% | 8.7% | 153 | 4749 | True |
| HHI | 0.975 | 2.7% | 8.6% | 73 | 4829 | True |
| top1 | 0.975 | 2.1% | 8.7% | 94 | 4808 | True |
| HHI | 0.99 | 1.7% | 8.6% | 59 | 4843 | True |
| top1 | 0.99 | 2.9% | 8.6% | 68 | 4834 | True |


### Q5 dummy leftover

Q5 dummy Y3 0.571 leftover after days 0.584 dies=False. Q1 dummy leftover after days 0.596 dies=False. Do not KEEP a Q5 dummy as a 44 stem.

### Y2 leftover (report-only)

Y2 top1 0.463 leftover after days 0.507 dies=True after size 0.478 (report-only; not a Y2 engine).

### company-mean trait leftover

company-mean Y3 0.668 leftover after days+size 0.576 dies=False. leftover after HHI-mean 0.580 ρ(mean_top1, mean_HHI)=0.990 (trait twin). Do not KEEP the company mean.

### first_month vintage leftover

early first_month leftover after days 0.586 n=1613 pos=115 dies=False. late leftover 0.519 n=1264 pos=88 dies=True.

### Q5 dummy leftover detail

Q5 leftover after days OLS 0.584 rank 0.584 fake=False almost=True folds=0.376 0.668 0.648 0.611 0.619 n=2877 pos=203. after days+size 0.583 dies=False. after HHI-Q5 0.572 dies=True. after days+HHI-Q5 0.727 dies=True. KEEP dummy as 44 stem=True — no, leftover barely lives and is a bin of the twin.

### early first_month leftover detail

early Y3 0.693 leftover after days OLS 0.589 rank 0.586 fake=False folds=0.713 0.625 0.667 0.570 0.357 n=1613 pos=115. after days+HHI 0.437 dies=True. after days+size 0.580 dies=False. KEEP early slice=True — no, leftover barely lives and dies after HHI.

### bootstrap leftover after HHI

Bootstrap leftover-after-HHI rank p05=0.423 p50=0.541 p95=0.584 OLS p50=0.452 share<0.55=66.7% n=24.

### n_supp>=3 / monopoly dummy

n_supp>=3 n=2661 Y3 0.627 leftover after days 0.381 dies=True. monopoly n_supp==1 n=83 Y3 0.512 leftover after days 0.704 dies=True after days+HHI 0.668 dies=False.

### >0.975 tail dummy leftover

top1>0.975 dummy Y3 0.483 leftover after days 0.712 dies=False. Y5 0.507 leftover after size 0.547 dies=True (HHI tail leftover after size 0.550). Tail dummy is a footnote, not a 44 stem. Y5 never E. Y3 leftover after days fake=False almost=True folds=0.706 0.801 0.644 0.686 0.726 n=2877 pos=203. Y3 tail leftover after days+HHI-tail 0.729 dies=True.

### Y5 only-top1 vs HHI subset

Y5 both 2.7% n=73; only_top1 0.0% n=21; only_HHI n=0; neither 8.7%. HHI cut is the stricter subset; extra 21 are still protective vs rest. Footnote stays on HHI>0.975.

| slice | n | Y5 | n_pos |
| --- | --- | --- | --- |
| both HHI∩top1 >0.975 | 73 | 2.7% | 2 |
| only top1>0.975 | 21 | 0.0% | 0 |
| only HHI>0.975 | 0 | — | 0 |
| neither | 4808 | 8.7% | 416 |


## Night quotes (unchanged)

| quote | locked |
| --- | --- |
| Y3 | 0.762 / 0.752 |
| days | 0.711 |
| size | 0.617 |
| Y7 TURNOVER | 0.720 / 0.712 |
| h_sib_neg leftover | 0.453 DROP |

Do not quote a_out_vol 0.722. Do not grow TURNOVER. Do not put `d_supp_top1` on the 15-col card.

## Files written

- `analysis/evaluate/supp_top1_qa.py`
- `analysis/outputs/supp_top1_qa.md`
- `analysis/outputs/supp_top1_qa.png`
- append-only `analysis/experiments/registry.csv`
- `overnight/waves/wave4_supp_top1.md` (end, if WRITE_WAVE)

Elapsed 38s. Failed: none.

