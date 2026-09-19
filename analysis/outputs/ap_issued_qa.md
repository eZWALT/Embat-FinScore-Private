# Unused leftover of `e_ap_issued` after `c_n_days_with_tx`

Generated `2026-09-19T06:16:40+02:00` by agent `e8b2c0d4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_ap_issued`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Do **not** put e_ap_issued on the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. AR issued leftover 0.608 CLOSED — do not overwrite issued_qa.

`e_ap_issued` = this-period AP issuance volume (amount<0 invoices). Feature report: 64.1% cov, acf1 0.17, w/b 0.86, ICC 0.54 LOW_PERSIST (not BETWEEN).

## Headline

CLOSE leftover-after-days rank 0.591 (OLS 0.714, fake=True). Y3 ap 0.675 vs days 0.711 vs size 0.617 vs AR 0.687. SIZE=True twin=False. Inverse days-after-ap 0.683. Leftover after AR 0.578 (after AR+days 0.524 dies). After d_n_supp 0.523. both>0 leftover-days 0.471. ap>0 leftover 0.532. zero-light leftover 0.370. Boot leftover-days p50=0.584 p05=0.544. Card: CLOSE unused leftover / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| ap leftover after days (Y3 X) | **CLOSE** | Y3 leftover after days rank 0.591 OLS 0.714 dies; twin=False SIZE=True. Inverse days-after-ap 0.683. Contemporaneous AP stays off the 15-col card. Do not grow TURNOVER. |
| 15-col Y3 card stem | **KEEP off the card** | do not put e_ap_issued on the card |
| TURNOVER add-on | **CLOSE** | do not grow 0.720 |
| AR issued leftover | **CLOSE (locked)** | rank 0.608 fake days clone |
| health Y `y_ap_issued` | **PARK** | do not invent y_ap_issued |

## 1. Coverage / 470 vs ERP / SIZE ρ

Train e_ap_issued nn=13,555 cov=64.1% (feature report 64.1%). Dark never-ERP 470 (want 470): nn=1 zero=1 pos=0 BOOK stub. Ever-ERP 744 nn=13,554 of which zero=2,725. p50=19591 p99=34156859 max=62714613082. ρ vs log1p(a_in3)=0.514 n=12067 SIZE. acf1=0.170 (quote 0.17) ICC=0.537 (feature-report ICC 0.54 CONFIRM; the 0.86 is w/b not ICC) k=745.

| slice | n_cm | nn | cov | eq0 | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| all train | 21,157 | 13,555 | 64.1% | 20.1% | 19591 | 34156859 |
| ever-ERP | 13,554 | 13,554 | 100.0% | 20.1% | — | — |
| dark 470 | 7,603 | 1 | 0.0% | 100.0% | — | — |


## 2. Spearman twins / SIZE

ap_issued vs days ρ=0.459 (not a twin). vs a_n_tx 0.504 vs e_ar_issued 0.672 vs DPO -0.109 vs d_n_supp 0.760 vs log1p(a_in3) 0.514 (SIZE). vs ap_lag1 0.859. TWIN |ρ|≥0.80: e_ap_issued_lag1. Gate twins: none.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| c_n_days_with_tx | 0.459 | 13,555 |  |
| a_n_tx | 0.504 | 13,555 |  |
| e_ar_issued | 0.672 | 13,555 |  |
| e_dpo_proxy | -0.109 | 10,829 |  |
| d_n_supp | 0.760 | 11,293 |  |
| log1p(a_in3) | 0.514 | 12,067 |  |
| e_ap_issued_lag1 | 0.859 | 12,810 | TWIN |
| a_in3 | 0.514 | 12,067 |  |


## 3. Single-feature group-fold Y3 / Y7

Y3 e_ap_issued 0.675 vs days 0.711 (CONFIRM 0.711) vs size 0.617 (CONFIRM 0.617) vs e_ar_issued 0.687 (CONFIRM 0.687) vs DPO 0.627. ap_lag1 0.656. Y7 ap 0.570 ap_lag1 0.555. Beat size 0.058.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_ap_issued | 3,618 | 264 | 0.675 | 0.117 | -1 | 0.811 0.656 0.492 0.712 0.704 |
| y3_recover_cash_6m | e_ap_issued_lag1 | 3,618 | 264 | 0.656 | 0.104 | -1 | 0.779 0.642 0.494 0.701 0.666 |
| y3_recover_cash_6m | log1p_ap | 3,618 | 264 | 0.675 | 0.117 | -1 | 0.811 0.656 0.492 0.712 0.704 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | e_ar_issued | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | e_dpo_proxy | 3,015 | 174 | 0.627 | 0.112 | 1 | 0.797 0.510 0.605 0.669 0.554 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | d_n_supp | 3,003 | 221 | 0.699 | 0.116 | -1 | 0.820 0.722 0.514 0.764 0.674 |
| y7_top1_lost | e_ap_issued | 7,464 | 2,149 | 0.570 | 0.058 | -1 | 0.562 0.483 0.572 0.589 0.644 |
| y7_top1_lost | e_ap_issued_lag1 | 7,253 | 2,072 | 0.555 | 0.058 | -1 | 0.553 0.464 0.557 0.579 0.621 |
| y7_top1_lost | log1p_ap | 7,464 | 2,149 | 0.570 | 0.058 | -1 | 0.562 0.483 0.572 0.589 0.644 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | e_ar_issued | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 |
| y7_top1_lost | e_dpo_proxy | 7,038 | 1,929 | 0.450 | 0.051 | -1 | 0.497 0.403 0.510 0.438 0.403 |
| y7_top1_lost | a_n_tx | 7,464 | 2,149 | 0.452 | 0.032 | -1 | 0.461 0.482 0.476 0.436 0.405 |
| y7_top1_lost | d_n_supp | 6,702 | 1,871 | 0.552 | 0.029 | -1 | 0.565 0.509 0.547 0.588 0.549 |


## 4. Honest leftover after days (Y3)

Y3 leftover after days rank 0.591 OLS 0.714 ρ(resid,days)=-0.907 R²=0.000 FALSE clone — dies. Inverse: days leftover after ap rank 0.683 OLS 0.729 survives — keep the 0.711 bar.

| bar | rank | OLS | ρ(resid,ctrl) | R² | fake? | n |
| --- | --- | --- | --- | --- | --- | --- |
| ap leftover after days | 0.591 | 0.714 | -0.907 | 0.000 | FALSE clone | 3,618 |
| days leftover after ap | 0.683 | 0.729 | 0.431 | 0.000 |  | 3,618 |


## 5. Leftover after e_ar_issued

Y3 leftover after e_ar_issued rank 0.578 OLS 0.557 ρ=-0.422 not just AR. After AR+days 0.524 fake=False.

| bar | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| after e_ar_issued | 0.578 | 0.557 | -0.422 |  | 3,618 |
| after AR+days | 0.524 | 0.717 | -0.464 |  | 3,618 |


## 6. Y5 leftover after size (report only)

Y5 leftover after size rank 0.563 OLS 0.549 raw 0.585. Y5 never E — no AUROC as card X, leak_ok=False.

| slice | rank | OLS | ρ | raw | n |
| --- | --- | --- | --- | --- | --- |
| Y5 leftover after size (report only) | 0.563 | 0.549 | -0.939 | 0.585 | 4,905 |


## 7. Dark 470

Dark never-ERP 470 nn=1 zero=1 pos=0. BOOK stub or check.

## 8. Q6 lag1 leftover after days_lag1

Q6 Y7 ap_lag1 on short 0.543 (AR issued_lag1 KEEP 0.626 locked). ap_lag1 leftover after days_lag1 short rank 0.547 fake=True. Contemporaneous leftover after days_lag1 short 0.568 fake=True. Do not claim a TURNOVER seat.

| slice | CV | n | n_pos |
| --- | --- | --- | --- |
| Q6 Y7 ap_lag1 short | 0.543 | 4,210 | 1,260 |
| Q6 Y3 ap_lag1 leftover after days_lag1 short |  |  |  |
| Q6 Y3 contemporaneous leftover after days_lag1 short |  |  |  |


## 9. vs e_ap_issued_lag1

Store has e_ap_issued_lag1=False; panel lag computed either way. Y3 ap_lag1 0.656. Now leftover after lag1 rank 0.614 OLS 0.677 ρ=0.816. Y7 leftover after lag1 0.542 — do not grow TURNOVER 0.720.

| bar | rank | OLS | ρ | raw_lag1 | fake? |
| --- | --- | --- | --- | --- | --- |
| Y3 now leftover after ap_lag1 | 0.614 | 0.677 | 0.816 | 0.656 | FALSE clone |
| Y7 now leftover after ap_lag1 | 0.542 | 0.569 | 0.816 | — | FALSE clone |


## Extra — leftover after size / DPO

Leftover after size 0.620. After DPO 0.635 (DPO already decided — do not reopen). After days+size 0.586.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| after size | 0.620 | 0.618 | -0.939 | FALSE clone |
| after DPO | 0.635 | 0.600 | 0.021 |  |
| after days+size | 0.586 | 0.449 | 0.117 |  |


## 10. Bootstrap leftover-after-days

Bootstrap leftover-after-days rank p05=0.544 p50=0.584 p95=0.630 n=40/40.

| boot | p05 | p50 | p95 |
| --- | --- | --- | --- |
| n=40/40 | 0.544 | 0.584 | 0.630 |


## Extra — holdout coverage

Holdout 72 coverage only: 1,073 CM / 72 companies. ap cov 54.2%; dark nn=0. No AUROC.

| slice | n_cm | companies | cov | dark_nn |
| --- | --- | --- | --- | --- |
| holdout 72 | 1,073 | 72 | 54.2% | 0 |


## Extra — ap>0 leftover

ap>0 leftover after days rank 0.532 OLS 0.721 ρ=-0.907 fake=True raw 0.648 days 0.732.

| slice | rank | OLS | ρ | raw | days | fake? |
| --- | --- | --- | --- | --- | --- | --- |
| ap>0 leftover-days | 0.532 | 0.721 | -0.907 | 0.648 | 0.732 | FALSE clone |


## Extra — log1p(ap)

log1p(ap) Y3 0.675 leftover-after-days rank 0.591 OLS 0.575 ρ=-0.018 fake=False.

| slice | rank | OLS | ρ | raw | fake? |
| --- | --- | --- | --- | --- | --- |
| log1p(ap) leftover-days | 0.591 | 0.575 | -0.018 | 0.675 |  |


## Extra — intensity

ap/days intensity Y3 0.633 leftover-after-days rank 0.582 fake=False.

| slice | rank | OLS | ρ | raw | fake? |
| --- | --- | --- | --- | --- | --- |
| ap/days leftover-days | 0.582 | 0.640 | -0.632 | 0.633 |  |


## Extra — SIZE terciles

SIZE terciles leftover after days: T1 0.532 T2+T3 0.540 T3 —.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| T1 | 0.532 | 0.606 | -0.907 | FALSE clone | 847 |
| T2 | 0.371 | 0.619 | -0.907 | FALSE clone | 1,361 |
| T3 | — | — | -0.907 | FALSE clone | 1,410 |
| T2+T3 | 0.540 | 0.650 | -0.907 | FALSE clone | 2,771 |


## Extra — company-median ρ

Company-median ρ ap vs days 0.532 vs log1p(a_in3) 0.594 SIZE (feature report flagged company-median VIF / size).

| pair | ρ | n |
| --- | --- | --- |
| company-median vs days | 0.532 | 745 |
| company-median vs log1p(a_in3) | 0.594 | 745 |


## Extra — leftover after d_n_supp / a_n_tx

Leftover after d_n_supp rank 0.523 OLS 0.675 ρ=-0.793 fake=False. After supp+days 0.513. After a_n_tx 0.580 fake=True.

| bar | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| after d_n_supp | 0.523 | 0.675 | -0.793 |  | 3,003 |
| after supp+days | 0.513 | 0.727 | -0.668 |  | 3,003 |
| after a_n_tx | 0.580 | 0.708 | 0.844 | FALSE clone | 3,618 |


## Extra — both AP>0 and AR>0

both>0 leftover-days rank 0.471 fake=True; leftover-AR 0.469; days-after-ap 0.695.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| ap>0 & ar>0 leftover-days | 0.471 | 0.692 | -0.907 | FALSE clone | 2,358 |
| ap>0 & ar>0 leftover-AR | 0.469 | 0.548 | -0.422 |  | 2,358 |
| days leftover after ap on both>0 | 0.695 | 0.692 | 0.431 |  | 2,358 |


## Extra — so_far leftover after days

so_far leftover after days: short 0.562 mid 0.605 long —.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| short_<12 | 0.562 | 0.707 | -0.907 | FALSE clone | 2,339 |
| mid_12_17 | 0.605 | 0.717 | -0.907 | FALSE clone | 1,141 |
| long_>=18 | — | — | -0.907 | FALSE clone | 138 |


## Extra — per-fold leftover

Per-fold rank leftover after days: 0.745 0.522 0.445 0.619 0.624.

| fold | rank leftover |
| --- | --- |
| 0 | 0.745 |
| 1 | 0.522 |
| 2 | 0.445 |
| 3 | 0.619 |
| 4 | 0.624 |


## Extra — log1p twins / leftover after size / AR / lag1

log1p leftover after size 0.620 after AR 0.578 after lag1 0.614. Inverse days-after-log1p 0.683. ρ vs size 0.514 SIZE; vs lag1 0.859 TWIN.

| bar | rank | OLS | ρ | n |
| --- | --- | --- | --- | --- |
| log1p leftover after size | 0.620 | 0.625 | 0.141 | 3,543 |
| log1p leftover after AR | 0.578 | 0.673 | 0.658 | 3,618 |
| log1p leftover after log1p_lag1 | 0.614 | 0.627 | 0.269 | 3,618 |
| days leftover after log1p | 0.683 | 0.683 | 0.117 | 3,618 |
| ρ log1p vs log1p(a_in3) |  |  | 0.514 | 12,067 |
| ρ log1p vs log1p_lag1 |  |  | 0.859 | 12,810 |
| ρ log1p vs days |  |  | 0.459 | 13,555 |


## Extra — Q6 mid / long (do not claim TURNOVER)

Q6 mid Y7 ap_lag1 0.557 leftover 0.592; long Y7 0.494 leftover —. Do not claim a TURNOVER seat.

| slice | Y7 ap_lag1 | Y3 leftover days_lag1 | Y7 leftover days_lag1 | fake? |
| --- | --- | --- | --- | --- |
| short_<12 | 0.543 | 0.547 | 0.523 | FALSE clone |
| mid_12_17 | 0.557 | 0.592 | 0.568 | FALSE clone |
| long_>=18 | 0.494 | — | 0.560 | FALSE clone |


## Extra — ICC formulas vs quote 0.86

ICC ANOVA 0.537 ICC(1) 0.009 between-share 0.063 means-formula 0.062 (quote 0.86). n0=18.2 k=745.

| formula | ICC |
| --- | --- |
| ANOVA MSB/(MSB+MSW) | 0.537 |
| ICC(1) (MSB-MSW)/(MSB+(n0-1)MSW) | 0.009 |
| SSB/(SSB+SSW) share | 0.063 |
| var(means)/(var(means)+mean within) | 0.062 |
| n0 / k | 18.2 / 745 |


## Extra — inverse AR leftover / Y7 leftover after AR

Inverse AR leftover after AP 0.640 fake=False. Y7 AP leftover after AR 0.529 — do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| AR leftover after AP (Y3) | 0.640 | 0.669 | 0.483 |  |
| AP leftover after AR (Y7) | 0.529 | 0.585 | -0.422 |  |


## Extra — ever-ERP / early-late leftover

ever-ERP leftover-days 0.591 fake=True; early — late 0.595.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| ever-ERP leftover-days | 0.591 | 0.714 | -0.907 | FALSE clone | 3,618 |
| ever-ERP early (pre delay mask) | — | — | -0.907 | FALSE clone | 666 |
| ever-ERP late | 0.595 | 0.724 | -0.907 | FALSE clone | 2,952 |


## Extra — ap/(ap+ar) share leftover

ap/(ap+ar) Y3 0.615 leftover-days 0.588 fake=False; leftover-AR 0.510.

| bar | rank | OLS | raw | ρ | fake? |
| --- | --- | --- | --- | --- | --- |
| ap/(ap+ar) leftover-days | 0.588 | 0.568 | 0.615 | 0.058 |  |
| ap/(ap+ar) leftover-AR | 0.510 | 0.615 | — | -0.643 |  |


## Extra — AP-median quintiles leftover

AP-median quintiles leftover-days Q1 0.579 Q5 — Q4+Q5 0.564.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| Q1 | 0.579 | 0.681 | -0.907 | FALSE clone | 717 |
| Q5 | — | — | -0.907 | FALSE clone | 692 |
| Q1+Q2 | 0.474 | 0.653 | -0.907 | FALSE clone | 1,306 |
| Q4+Q5 | 0.564 | 0.765 | -0.907 | FALSE clone | 1,570 |


## Extra — fold 0 vs rest leftover

fold0 leftover-days — fake=True; other folds 0.553 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| fold==0 leftover-days | — | — | -0.907 | FALSE clone | 575 |
| fold!=0 leftover-days | 0.553 | 0.727 | -0.907 | FALSE clone | 3,043 |


## Extra — persistence / company-median vs d_n_supp

≥6 AP months leftover 0.591 fake=True; <6 —. Company-median ρ vs d_n_supp 0.789.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| ≥6 AP months leftover-days | 0.591 | 0.714 | -0.907 | FALSE clone | 3,618 |
| <6 AP months leftover-days | — | — | -0.907 | FALSE clone | 0 |
| company-median ρ vs d_n_supp |  |  | 0.789 |  | 744 |


## Extra — bootstrap leftover after AR

Bootstrap leftover-after-AR rank p05=0.541 p50=0.580 p95=0.626 n=20/20.

| boot | p05 | p50 | p95 |
| --- | --- | --- | --- |
| n=20/20 leftover-AR | 0.541 | 0.580 | 0.626 |


## Extra — w/b vs ICC (feature-report 0.86 is w/b)

w/b 0.862 (quote 0.86 CONFIRM); ICC 0.537 (quote 0.54 CONFIRM). AP is LOW_PERSIST, not BETWEEN.

| stat | value |
| --- | --- |
| var_w / var_b | 0.862 |
| ICC = var_b/(var_b+var_w) | 0.537 |
| feature-report w/b | 0.86 |
| feature-report ICC | 0.54 |


## Extra — winsor99 leftover

winsor99 Y3 0.675 leftover-days 0.591 ρ=-0.476 fake=False.

| bar | rank | OLS | raw | ρ | fake? |
| --- | --- | --- | --- | --- | --- |
| winsor99 leftover-days | 0.591 | 0.615 | 0.675 | -0.476 |  |


## Extra — lag3 leftover

lag3 Y3 0.636 leftover-days 0.550 fake=False; after lag1 0.545.

| bar | rank | OLS | raw | ρ | fake? |
| --- | --- | --- | --- | --- | --- |
| lag3 leftover-days | 0.550 | 0.713 | 0.636 | -0.783 |  |
| lag3 leftover after lag1 | 0.545 | 0.621 | — | 0.747 |  |


## Extra — Y7 leftover after days / mid fake / days after AP+AR

Y7 leftover after days 0.565 fake=True. mid leftover 0.605 fake=True. days after AP+AR 0.670. intensity after size 0.586.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days | 0.565 | 0.471 | -0.907 | FALSE clone |
| mid so_far leftover-days | 0.605 | 0.717 | -0.907 | FALSE clone |
| days leftover after AP+AR | 0.670 | 0.727 | 0.438 |  |
| intensity leftover after size | 0.586 | 0.624 | -0.939 | FALSE clone |


## Extra — ap>0 after supp / log1p after days+size

ap>0 leftover after supp 0.498; after supp+days+AR 0.588. log1p leftover after days+size 0.586 fake=False.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| ap>0 leftover after d_n_supp | 0.498 | 0.697 | -0.793 |  |
| ap>0 leftover after supp+days+AR | 0.588 | 0.723 | -0.476 |  |
| log1p leftover after days+size | 0.586 | 0.582 | 0.008 |  |


## Extra — DPO-defined leftover / ρ(rank-resid, days)

DPO-defined leftover-days 0.532 fake=True; after DPO+days 0.545. ρ(rank-resid, days)=0.081.

| bar | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| DPO-defined leftover-days | 0.532 | 0.721 | -0.907 | FALSE clone | 3,015 |
| DPO-defined leftover after DPO+days | 0.545 | 0.707 | 0.002 |  | 3,015 |
| ρ(rank-resid, days) |  |  | 0.081 |  | 3,618 |


## Extra — KEEP-as-X scorecard

KEEP-as-X scorecard: beat size PASS; leftover-after-days FAIL (rank 0.591 looks alive, OLS 0.714 is a days clone ρ=-0.907); SIZE FAIL 0.514; gate twins PASS; leftover after AR thin. Do not put e_ap_issued on the 15-col card.

| gate | value | pass? |
| --- | --- | --- |
| beat size ≥0.02 | 0.058 | PASS |
| leftover after days (honest, not fake, ≥0.55) | 0.591 | FAIL (fake days clone) |
| not SIZE |ρ| vs log1p(a_in3) <0.50 | 0.514 | FAIL |
| not twin vs days / n_tx / AR / DPO / d_n_supp | none | PASS (lag1 twin 0.859 is not a gate) |
| leftover after AR (not a rewrite) | 0.578 | thin 0.578 / after AR+days 0.524 dies |


## Extra — country leftover / vs locked AR issued

country leftover-days ES — fake=True; not-ES —; missing-country 0.599 fake=True. Twin of AR leftover 0.608.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| ES | — | — | -0.907 | FALSE clone | 831 |
| not-ES known | — | — | -0.907 | FALSE clone | 105 |
| has country | 0.554 | 0.689 | -0.907 | FALSE clone | 936 |
| missing country | 0.599 | 0.703 | -0.907 | FALSE clone | 2,682 |
| AR issued leftover (locked) | 0.608 | fake days clone | -0.849 | FALSE clone | locked |


## Extra — ERP vendor / zero-share leftover

ERP leftover-days businessCentral:0.606 netsuite:0.592 sage200:—. zero-heavy 0.586 zero-light 0.370.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| erp=businessCentral | 0.606 | 0.698 | -0.907 | FALSE clone | 1,420 |
| erp=netsuite | 0.592 | 0.713 | -0.907 | FALSE clone | 457 |
| erp=sage200 | — | — | -0.907 | FALSE clone | 292 |
| zero-share≥0.5 leftover-days | 0.586 | 0.684 | -0.907 | FALSE clone | 657 |
| zero-share<0.2 leftover-days | 0.370 | 0.738 | -0.907 | FALSE clone | 2,543 |


## Extra — leftover after open / overdue / days+AR+size

Leftover after e_ap_open 0.677 ρ_raw=0.757; after overdue 0.652 ρ_raw=-0.372; after days+AR+size 0.533; after open+days 0.611; zero-light leftover-AR 0.477.

| bar | rank | OLS | ρ | raw ρ |
| --- | --- | --- | --- | --- |
| leftover after e_ap_open | 0.677 | 0.537 | -0.535 | 0.757 |
| leftover after e_ap_overdue | 0.652 | 0.634 | -0.921 | -0.372 |
| leftover after days+AR+size | 0.533 | 0.447 | 0.097 | — |
| leftover after open+days | 0.611 | 0.670 | -0.548 | — |
| zero-light leftover after AR | 0.477 | 0.581 | -0.422 | — |


## Extra — never-zero companies leftover

never-zero leftover-days 0.281 fake=True; days-after-ap 0.629; leftover-AR 0.594.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| never-zero leftover-days | 0.281 | 0.640 | -0.907 | FALSE clone | 1,733 |
| days leftover after AP on never-zero | 0.629 | 0.603 | 0.431 |  | 1,733 |
| never-zero leftover-AR | 0.594 | 0.586 | -0.422 |  | 1,733 |


## Extra — has_ap dummy / has-any-zero leftover

has_ap dummy Y3 0.602 leftover-days 0.533 fake=False. has-any-zero leftover 0.461 (never-zero was 0.281).

| slice | rank | OLS | raw | ρ | fake? |
| --- | --- | --- | --- | --- | --- |
| has_ap dummy leftover-days | 0.533 | 0.533 | 0.602 | -0.445 |  |
| has-any-zero leftover-days | 0.461 | 0.731 | — | -0.907 | FALSE clone |


## Extra — issued/open AP turnover leftover (not TURNOVER family)

issued/open Y3 0.662 leftover-days 0.625 fake=True; after issued 0.601. Do not claim a TURNOVER seat.

| bar | rank | OLS | raw | ρ | fake? |
| --- | --- | --- | --- | --- | --- |
| issued/open leftover-days | 0.625 | 0.736 | 0.662 | 0.950 | FALSE clone |
| issued/open leftover after issued | 0.601 | 0.662 | — | 0.425 |  |


## Extra — leftover after locked AR issued_lag1 (do not grow TURNOVER)

Y3 leftover after AR_lag1 0.593; Y7 0.447; after AR_lag1+days 0.533. ρ vs AR_lag1 0.634. Do not grow TURNOVER 0.720.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y3 leftover after AR_lag1 | 0.593 | 0.456 | -0.333 |  |
| Y7 leftover after AR_lag1 | 0.447 | 0.541 | -0.333 |  |
| Y3 leftover after AR_lag1+days | 0.533 | 0.708 | -0.428 |  |
| ρ(ap, AR_lag1) |  |  | 0.634 |  |


## Extra — leftover after AP_lag1 + AR_lag1

Y3 leftover after both lag1s 0.592; Y7 0.517. Contemporaneous AP adds nothing on top of the two lag1s. Do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y3 leftover after AP_lag1+AR_lag1 | 0.592 | 0.469 | 0.123 |  |
| Y7 leftover after AP_lag1+AR_lag1 | 0.517 | 0.542 | 0.123 |  |


## Extra — leftover after group-mean days / AP

Leftover after group-mean days 0.648; after group-mean AP 0.668; after group-days+days 0.602.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| leftover after group-mean days | 0.648 | 0.571 | -0.634 |  |
| leftover after group-mean AP | 0.668 | 0.576 | -0.545 |  |
| leftover after group-days + days | 0.602 | 0.665 | -0.112 |  |


## Extra — leftover after card KEEP leftovers (ss / salary / days)

Leftover after c_ss_month 0.641 (card KEEP 0.635); after c_salary_month 0.644 (card KEEP 0.603); after ss+salary+days 0.601. Stay off the 15-col card.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| leftover after c_ss_month | 0.641 | 0.788 | 0.826 | FALSE clone |
| leftover after c_salary_month | 0.644 | 0.765 | 0.826 | FALSE clone |
| leftover after ss+salary+days | 0.601 | 0.678 | 0.805 | FALSE clone |


## What failed / next (held for wave note)

- Y3 leftover after days rank 0.591 OLS 0.714 fake=True
- inverse days leftover 0.683
- leftover after AR 0.578 rewrite=False
- Y5 leftover after size 0.563 leak_ok=False
- Q6 ap_lag1 leftover 0.547 now 0.568
- leftover after ap_lag1 0.614 Y7 0.542
- leftover after size 0.620 DPO 0.635
- boot leftover p05=0.544 p50=0.584 p95=0.630
- ap>0 leftover 0.532 fake=True
- log1p leftover 0.591 fake=False
- intensity leftover 0.582 fake=False
- terciles T1 0.532 T2+T3 0.540
- co-median SIZE=True ρ=0.594
- after d_n_supp 0.523 dies=True after n_tx 0.580
- both>0 leftover-days 0.471 leftover-AR 0.469
- so_far short 0.562 mid 0.605 long —
- per-fold leftover 0.745 0.522 0.445 0.619 0.624
- log1p after size 0.620 SIZE=True twin_lag=True
- Q6 mid Y7 0.557 long Y7 0.494
- ICC(1) 0.009 share 0.063 vs quote 0.86
- AR leftover after AP 0.640 Y7 leftover after AR 0.529
- ever-ERP leftover 0.591 early — late 0.595
- ap/(ap+ar) leftover 0.588 fake=False raw 0.615
- quintiles Q1 0.579 Q5 — hi 0.564
- fold0 leftover — fake=True rest 0.553 fake=True
- ≥6mo leftover 0.591 fake=True ρ vs supp 0.789
- boot leftover-AR p05=0.541 p50=0.580 p95=0.626
- w/b 0.862 ICC 0.537 (0.86 is w/b; ICC 0.54 CONFIRM)
- winsor leftover 0.591 fake=False
- lag3 leftover 0.550 after lag1 0.545
- Y7 leftover-days 0.565 mid 0.605 fake=True days-after-AP+AR 0.670
- ap>0 after supp 0.498 triple 0.588 log1p days+size 0.586
- DPO-defined leftover 0.532 after DPO+days 0.545 ρ(rank-resid,days)=0.081
- country ES leftover — fake=True missing 0.599 fake=True
- zero-heavy leftover 0.586 zero-light 0.370
- leftover after open 0.677 overdue 0.652 days+AR+size 0.533 open+days 0.611 zero-light-AR 0.477
- never-zero leftover-days 0.281 leftover-AR 0.594 days-after-ap 0.629
- has_ap leftover 0.533 has-any-zero 0.461
- issued/open leftover 0.625 after issued 0.601
- leftover after AR_lag1 Y3 0.593 Y7 0.447 +days 0.533
- leftover after both lag1s Y3 0.592 Y7 0.517
- leftover after group-mean days 0.648 group-AP 0.668 +days 0.602
- leftover after ss 0.641 salary 0.644 ss+salary+days 0.601
- card: CLOSE unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put e_ap_issued on the 15-col card

Elapsed 14s. Night quotes unchanged. Do not grow TURNOVER. Do not put e_ap_issued on the 15-col card.
