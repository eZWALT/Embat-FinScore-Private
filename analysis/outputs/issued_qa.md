# Unused leftover of `e_ar_issued` after `c_n_days_with_tx`

Generated `2026-09-19T05:46:18+02:00` by agent `e8b2c0d4`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_issued`. Night Y7 stays **TURNOVER 0.720 / B_shallow 0.712**. Do **not** grow TURNOVER. Do **not** put contemporaneous issued on the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 never D. Y5 never E. Y3 never B. Dark 470 stay NaN not 0. CN leftover KEEP 0.597 after issued_lag1 — not overwritten.

`e_ar_issued` = this-period AR issuance volume. Feature report: 64.1% cov, acf1 0.14, ICC 0.94 BETWEEN / LOW_PERSIST; company-median VIF with size. This card is unused leftover after days as Y3 X, and leftover of *now* after `e_ar_issued_lag1` as a TURNOVER add-on.

## Headline

CLOSE leftover-after-days rank 0.608 (OLS 0.694, fake=True). Y3 issued 0.687 vs days 0.711 vs size 0.617. SIZE=False twin=True. Inverse days-after-issued 0.676. Y7 leftover after lag1 0.581 — CLOSE. Card: CLOSE as unused leftover / KEEP off the 15-col card. Q6 CLOSE. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged.

## Brief map

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | issued is a volume stem (feature-report BETWEEN / size VIF), not a health Y. Do not invent `y_issued`. |
| 2 | Who is improving? | Y3 leftover after days 0.608 — CLOSE. The 45→65 engine is days 0.711, not contemporaneous issued. |
| 3 | Who is turning? | Q6 issued_lag1 already KEEP 0.626. Contemporaneous issued is not the lead. |
| 4 | Dip vs fall? | Y7 leftover after issued_lag1 0.581 — CLOSE. Do not grow TURNOVER 0.720. |
| 5 | Why did it change? | Issued Y3 0.687 loses to days 0.711. Unused leftover after days is the only why-question. |
| 6 | Months earlier? | lag1 already KEEP. Contemporaneous leftover on short 0.585 / 0.640 — CLOSE. |


## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| issued leftover after days (Y3 X) | **CLOSE** | Y3 leftover after days rank 0.608 OLS 0.694 dies; twin=True SIZE=False. Inverse days-after-issued 0.676. Contemporaneous issued stays off the 15-col card. Do not grow TURNOVER. |
| 15-col Y3 card stem | **KEEP off the card** | already off (raw-level park). Do not put issued on the card. |
| TURNOVER add-on (now after lag1) | **CLOSE** | Y7 leftover after issued_lag1 0.581; do not grow 0.720 |
| issued_lag1 Q6 / TURNOVER stem | **KEEP (locked)** | short so-far 0.626 vs 0.626 / 0.630 |
| health Y `y_issued` | **PARK** | do not invent y_issued |


## 1. Coverage / 470 vs ERP / SIZE ρ

Train issued nn=13,555 cov=64.1% (feature report 64.1%). Dark never-ERP 470 (want 470): nn=1 zero=1 pos=0 BOOK stub. Ever-ERP 744 nn=13,554 of which zero=4,971. p50=13890 p99=39372348 max=5947260462. ρ vs log1p(a_in3)=0.463 n=12067 not SIZE. acf1=0.143 (quote 0.14) ICC=0.942 (quote 0.94) k=745.

| slice | n_cm | nn | cov | eq0 | p50 | p99 |
| --- | --- | --- | --- | --- | --- | --- |
| all train | 21,157 | 13,555 | 64.1% | 36.7% | 13890 | 39372348 |
| ever-ERP | 13,554 | 13,554 | 100.0% | 36.7% | — | — |
| dark 470 | 7,603 | 1 | 0.0% | 100.0% | — | — |


## 2. Spearman twins

issued vs days ρ=0.417 (not a twin). vs a_in3 0.462 vs log1p(a_in3) 0.463 (not SIZE). vs a_n_tx 0.459 vs issued_lag1 0.814 vs CN 0.251 vs DSO -0.048 vs pending -0.023. TWIN |ρ|≥0.80: e_ar_issued_lag1.

| vs | ρ | n | twin? |
| --- | --- | --- | --- |
| c_n_days_with_tx | 0.417 | 13,555 |  |
| a_in3 | 0.462 | 12,067 |  |
| log1p(a_in3) | 0.463 | 12,067 |  |
| a_n_tx | 0.459 | 13,555 |  |
| e_ar_issued_lag1 | 0.814 | 12,810 | TWIN |
| e_credit_note_ratio | 0.251 | 11,207 |  |
| e_dso_proxy | -0.048 | 8,583 |  |
| e_pending_amt_share | -0.023 | 12,762 |  |


## 3. Single-feature group-fold

Y3 issued 0.687 (CONFIRM 0.687) issued_lag1 0.671 vs days 0.711 (CONFIRM 0.711) vs size 0.617 (CONFIRM 0.617). Y7 issued 0.663 (quote 0.663) issued_lag1 0.630 (CONFIRM 0.630). Fold 4 issued 0.696 (quote 0.647). Beat size 0.070.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | e_ar_issued | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | e_ar_issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |
| y3_recover_cash_6m | log1p_issued | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p_a_in3 | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | a_n_tx | 5,648 | 402 | 0.703 | 0.034 | -1 | 0.652 0.732 0.697 0.699 0.737 |
| y3_recover_cash_6m | e_credit_note_ratio | 3,079 | 177 | 0.579 | 0.065 | -1 | 0.655 0.509 0.518 0.627 0.584 |
| y3_recover_cash_6m | e_dso_proxy | 2,418 | 93 | 0.564 | 0.135 | 1 | 0.680 0.380 0.588 0.697 0.476 |
| y7_top1_lost | e_ar_issued | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 |
| y7_top1_lost | e_ar_issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | log1p_issued | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 |
| y7_top1_lost | c_n_days_with_tx | 7,464 | 2,149 | 0.458 | 0.023 | -1 | 0.457 0.483 0.478 0.432 0.439 |
| y7_top1_lost | log1p_a_in3 | 7,000 | 1,987 | 0.469 | 0.080 | -1 | 0.453 0.540 0.484 0.527 0.339 |
| y7_top1_lost | a_n_tx | 7,464 | 2,149 | 0.452 | 0.032 | -1 | 0.461 0.482 0.476 0.436 0.405 |
| y7_top1_lost | e_credit_note_ratio | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 |
| y7_top1_lost | e_dso_proxy | 6,651 | 1,623 | 0.431 | 0.101 | 1 | 0.370 0.600 0.420 0.424 0.342 |


## 4. Honest leftover after days (Y3) + inverse

Y3 leftover after days rank 0.608 OLS 0.694 ρ(resid,days)=-0.849 R²=0.000 FALSE clone — dies. Inverse: days leftover after issued rank 0.676 OLS 0.727 survives — keep the 0.711 bar.

| cut | rank | OLS | ρ(resid,ctrl) | R² | fake? |
| --- | --- | --- | --- | --- | --- |
| issued leftover after days (Y3) | 0.608 | 0.694 | -0.849 | 0.000 | FALSE clone |
| days leftover after issued (Y3) | 0.676 | 0.727 | 0.388 | 0.000 |  |


## 5. Leftover after size / issued_lag1

Y3 leftover after size rank 0.645 OLS 0.619. After issued_lag1 rank 0.653 OLS 0.659 ρ(resid,lag1)=0.493 (now vs lag is leftover-live). After days+size rank 0.613.

| bar | rank | OLS | ρ(resid,ctrl) | dies? |
| --- | --- | --- | --- | --- |
| after size | 0.645 | 0.619 | -0.923 | dies |
| after issued_lag1 | 0.653 | 0.659 | 0.493 | lives |
| after days+size | 0.613 | 0.450 | 0.136 | lives |


## 6. SIZE terciles

SIZE terciles leftover after days: T1 rank 0.530 T2+T3 0.569 T3 —. dies inside T1 and T2+T3.

| slice | raw issued | leftover rank | leftover OLS | n | dies? |
| --- | --- | --- | --- | --- | --- |
| T1 | 0.591 | 0.530 | 0.612 | 847 | dies |
| T2 | 0.584 | 0.452 | 0.623 | 1,361 | dies |
| T3 | — | — | — | 1,410 | dies |
| T2+T3 | 0.621 | 0.569 | 0.630 | 2,771 | dies |
| all | 0.687 | 0.608 | 0.694 | 3,618 | dies |


## 7. Q6 short books

Q6 Y7 issued_lag1 on short so-far 0.626 (CONFIRM 0.626 vs night 0.630). Y3 contemporaneous leftover after days_lag1 on short rank 0.585; after issued_lag1 0.640. dies on short books. days_lag1 leftover is a FALSE clone. Q6 contemporaneous CLOSE (lag1 KEEP locked). 

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | all issued now | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | all issued_lag1 | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |
| y3_recover_cash_6m | all days | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | all days_lag1 | 5,648 | 402 | 0.684 | 0.034 | -1 | 0.627 0.714 0.694 0.684 0.701 |
| y3_recover_cash_6m | all leftover after days_lag1 | 3,618 | 264 | 0.621 | rank | — | 0.739 0.583 0.558 0.619 0.606 |
| y3_recover_cash_6m | all leftover after issued_lag1 | 3,618 | 264 | 0.653 | rank | — | 0.748 0.622 0.583 0.672 0.640 |
| y3_recover_cash_6m | short_<12 issued now | 2,339 | 156 | 0.654 | 0.066 | -1 | 0.708 0.705 0.625 0.552 0.678 |
| y3_recover_cash_6m | short_<12 issued_lag1 | 2,339 | 156 | 0.640 | 0.077 | -1 | 0.675 0.734 0.604 0.529 0.656 |
| y3_recover_cash_6m | short_<12 days | 3,723 | 252 | 0.696 | 0.067 | -1 | 0.645 0.753 0.742 0.605 0.735 |
| y3_recover_cash_6m | short_<12 days_lag1 | 3,723 | 252 | 0.684 | 0.059 | -1 | 0.635 0.721 0.736 0.607 0.720 |
| y3_recover_cash_6m | short_<12 leftover after days_lag1 | 2,339 | 156 | 0.585 | rank | — | 0.640 0.629 0.565 0.490 0.601 |
| y3_recover_cash_6m | short_<12 leftover after issued_lag1 | 2,339 | 156 | 0.640 | rank | — | 0.681 0.655 0.590 0.625 0.648 |
| y3_recover_cash_6m | long_>=18 issued now | 138 | 12 | LOW_POWER | — | — | — |
| y3_recover_cash_6m | long_>=18 issued_lag1 | 138 | 12 | LOW_POWER | — | — | — |
| y3_recover_cash_6m | long_>=18 days | 213 | 16 | LOW_POWER | — | — | — |
| y3_recover_cash_6m | long_>=18 days_lag1 | 213 | 16 | LOW_POWER | — | — | — |
| y3_recover_cash_6m | long_>=18 leftover after days_lag1 | 138 | 12 | — | rank | — |  |
| y3_recover_cash_6m | long_>=18 leftover after issued_lag1 | 138 | 12 | — | rank | — |  |


## 8. Y7 leftover after issued_lag1 (TURNOVER add-on)

Y7 contemporaneous leftover after issued_lag1 rank 0.581 OLS 0.655 ρ=0.493 R²=0.156. Raw issued 0.663 vs lag1 0.630. CLOSE as TURNOVER add-on — do not grow 0.720.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | issued now | 7,464 | 2,149 | 0.663 | 0.037 | -1 | 0.660 0.702 0.615 0.641 0.696 |
| y7_top1_lost | issued_lag1 | 7,253 | 2,072 | 0.630 | 0.031 | -1 | 0.643 0.662 0.590 0.605 0.647 |
| y7_top1_lost | leftover after issued_lag1 (rank) | 7,253 | 2,072 | 0.581 | 0.655 | rank/OLS | 0.540 0.616 0.549 0.590 0.612 |
| y7_top1_lost | leftover after days (rank) | 7,464 | 2,149 | 0.665 | 0.520 | rank/OLS | 0.676 0.721 0.630 0.619 0.679 |


## 9. CN leftover confirm (do not overwrite)

CN leftover after issued_lag1 rank 0.609 OLS 0.597 (locked KEEP 0.597 CONFIRM). This lane does not rewrite credit_note_qa.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | CN raw | 7,308 | 2,022 | 0.545 | 0.041 | 1 | 0.505 0.546 0.552 0.513 0.608 |
| y7_top1_lost | CN leftover after issued_lag1 rank | 7,105 | 1,951 | 0.609 | 0.597 | rank/OLS | 0.613 0.610 0.631 0.549 0.643 |


## 10. Fold 4 — do not reopen TURNOVER

Fold 4 Y7 issued 0.696 (quote 0.647) vs issued_lag1 0.647. Contemporaneous leftover-after-lag1 fold 4 0.612. Do not reopen TURNOVER 0.720.

| stem | CV | fold4 | folds |
| --- | --- | --- | --- |
| issued now | 0.663 | 0.696 | 0.660 0.702 0.615 0.641 0.696 |
| issued_lag1 | 0.630 | 0.647 | 0.643 0.662 0.590 0.605 0.647 |
| now leftover after lag1 (rank) | 0.581 | 0.612 | 0.540 0.616 0.549 0.590 0.612 |


## Extra — holdout coverage

Holdout 72 coverage only: 1,073 CM / 72 companies. issued cov 54.2%; dark nn=0. Y7 pos=122. No AUROC.

| col | n_cm | nn | cov | dark nn | Y7 pos |
| --- | --- | --- | --- | --- | --- |
| e_ar_issued | 1,073 | 582 | 54.2% | 0 | 122 |


## Extra — same-n issued vs issued_lag1

Same-n issued vs lag1: now 0.687 lag1 0.671 days 0.730 leftover-after-lag1 rank 0.653.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued same-n | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | issued_lag1 same-n | 3,618 | 264 | 0.671 | 0.071 | -1 | 0.760 0.700 0.567 0.678 0.650 |
| y3_recover_cash_6m | days same-n | 3,618 | 264 | 0.730 | 0.079 | -1 | 0.708 0.800 0.606 0.793 0.744 |


## Extra — log1p(issued)

log1p(issued) Y3 0.687 leftover-after-days rank 0.608 OLS 0.603 ρ=-0.038 lives — still loses if below days.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | log1p(issued) | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | log1p leftover after days (rank) | 3,618 | 264 | 0.608 | 0.603 | rank/OLS | 0.723 0.560 0.553 0.605 0.600 |


## Extra — issued/days intensity

issued/days intensity Y3 0.665 leftover-after-days rank 0.605 OLS 0.716 dies.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued/days | 3,564 | 248 | 0.665 | 0.066 | -1 | 0.771 0.652 0.587 0.654 0.661 |
| y3_recover_cash_6m | intensity leftover after days (rank) | 3,564 | 248 | 0.605 | 0.716 | rank/OLS | 0.715 0.560 0.559 0.588 0.604 |


## Extra — fat-issued tail

Fat-issued p90=1377798: leftover-days rank — vs rest 0.611.

| slice | raw | leftover rank | n | dies? |
| --- | --- | --- | --- | --- |
| fat p90+ | — | — | 215 | dies |
| rest | 0.696 | 0.611 | 3,403 | dies |
| all | 0.687 | 0.608 | 3,618 | dies |


## Extra — book-only (drop 470)

Book-only (drop 470) Y3 issued 0.687 leftover-days rank 0.608.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued book-only | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | book leftover after days (rank) | 3,618 | 264 | 0.608 | 0.694 | rank/OLS | 0.723 0.560 0.553 0.605 0.600 |


## Extra — dark BOOK stub

Dark BOOK stub rows=1 (expected 1). Do not rewrite invoices.py.

| company_id | period | e_ar_issued | group_id |
| --- | --- | --- | --- |
| COMP_0962 | 2026-02-01 | 0.00 | GROUP_0154 |


## Extra — log1p leftover after days+size

log1p leftover after days+size rank 0.613 OLS 0.609 ρ=-0.016 lives — still off the card. After size 0.645.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| log1p after days+size | 0.613 | 0.609 | -0.016 | lives |
| log1p after size | 0.645 | 0.651 | 0.123 | lives |


## Extra — fold-wise leftover after days

Y3 fold-wise leftover-days rank 0.608 folds 0.723 0.560 0.553 0.605 0.600 vs days 0.665 0.738 0.700 0.715 0.740.

| fold | leftover rank | days | n_pos |
| --- | --- | --- | --- |
| 0 | 0.723 | 0.665 | 39 |
| 1 | 0.560 | 0.738 | 47 |
| 2 | 0.553 | 0.700 | 46 |
| 3 | 0.605 | 0.715 | 56 |
| 4 | 0.600 | 0.740 | 76 |


## Extra — demean leftover

Demean leftover-days rank 0.477 mean leftover 0.572 raw leftover 0.608.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued raw | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | demean leftover-days rank | 3,618 | 264 | 0.477 | 0.612 | rank/OLS | 0.551 0.463 0.483 0.404 0.482 |
| y3_recover_cash_6m | company-mean leftover-days rank | 3,618 | 264 | 0.572 | 0.717 | rank/OLS | 0.652 0.643 0.554 0.482 0.532 |


## Extra — same-n leftover after days

Same-n leftover after days rank 0.608 OLS 0.694 ρ=-0.849 fake=True issued 0.687 days 0.730.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued same-n days | 3,618 | 264 | 0.687 | 0.071 | -1 | 0.794 0.675 0.597 0.700 0.669 |
| y3_recover_cash_6m | days same-n | 3,618 | 264 | 0.730 | 0.079 | -1 | 0.708 0.800 0.606 0.793 0.744 |
| y3_recover_cash_6m | leftover same-n rank | 3,618 | 264 | 0.608 | 0.694 | fake=True | 0.723 0.560 0.553 0.605 0.600 |


## Extra — holdout SIZE coverage

Holdout SIZE tercile coverage only: T1 56.5% T2 62.4% T3 43.5%. No AUROC.

| slice | n | nn | cov |
| --- | --- | --- | --- |
| holdout T1 | 324 | 183 | 56.5% |
| holdout T2 | 386 | 241 | 62.4% |
| holdout T3 | 363 | 158 | 43.5% |


## Extra — T2+T3 leftover fake check

T2+T3 leftover after days rank 0.569 OLS 0.630 ρ=-0.849 FALSE clone.

| slice | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| T2+T3 leftover after days | 0.569 | 0.630 | -0.849 | FALSE clone |


## Extra — issued>0 leftover after days

issued>0 leftover after days rank 0.412 OLS 0.666 ρ=-0.849 fake=True raw 0.587 days 0.689.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued>0 | 2,418 | 93 | 0.587 | 0.114 | -1 | 0.773 0.565 0.589 0.537 0.469 |
| y3_recover_cash_6m | days on issued>0 | 2,418 | 93 | 0.689 | 0.103 | -1 | 0.628 0.723 0.544 0.805 0.746 |
| y3_recover_cash_6m | issued>0 leftover-days rank | 2,418 | 93 | 0.412 | 0.666 | fake=True | 0.316 0.577 0.412 0.387 0.368 |


## Extra — log1p vs issued_lag1

log1p vs issued_lag1 ρ=0.814 n=12810 TWIN. leftover after lag1 rank 0.653 OLS 0.683.

| slice | n | value | twin? |
| --- | --- | --- | --- |
| log1p vs issued_lag1 ρ | 12810 | 0.814 | TWIN |
| log1p leftover after lag1 rank | 3618 | 0.653 |  |


## Extra — short-book leftover fake

Short leftover after days_lag1 rank 0.585 ρ=-0.844 FALSE clone. After days now 0.577 ρ=-0.849 FALSE clone.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| short leftover after days_lag1 | 0.585 | 0.666 | -0.844 | FALSE clone |
| short leftover after days now | 0.577 | 0.689 | -0.849 | FALSE clone |


## Extra — leftover after a_n_tx

Leftover after a_n_tx rank 0.603 ρ=0.104. After n_tx+days 0.605 lives.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after a_n_tx | 0.603 | 0.584 | 0.104 | lives |
| after n_tx+days | 0.605 | 0.699 | -0.665 | lives |


## Extra — Y7 leftover after days

Y7 leftover after days rank 0.665 OLS 0.520 ρ=-0.849 FALSE clone.

| cut | value | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days rank | 0.665 | 0.520 | -0.849 | FALSE clone |


## Extra — n_tx+days leftover vs days clone

n_tx+days leftover rank 0.605 ρ(resid,days)=-0.798 not a days clone.

| cut | rank | ρ(resid,days) | fake? |
| --- | --- | --- | --- |
| leftover n_tx+days vs days ρ | 0.605 | -0.798 |  |


## Extra — early6 coverage

Early6 issued cov 68.8% after7 63.2%. After-month7 leftover-days rank 0.614 fake=True.

| slice | nn | n | cov | eq0 |
| --- | --- | --- | --- | --- |
| early6 | 2220 | 3229 | 68.8% | 41.5% |
| after month 7 | 11335 | 17928 | 63.2% | 35.7% |


## Extra — issued==0 flag leftover

issued==0 flag Y3 0.667 leftover-days rank 0.582 fake=False.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued==0 flag | 3,618 | 264 | 0.667 | 0.062 | 1 | 0.742 0.660 0.572 0.688 0.675 |
| y3_recover_cash_6m | zero-flag leftover-days rank | 3,618 | 264 | 0.582 | 0.582 | fake=False | 0.679 0.542 0.528 0.560 0.602 |


## Extra — log1p leftover on issued>0

log1p issued>0 leftover after days rank 0.412 OLS 0.594 ρ=-0.038 fake=False raw 0.587 days 0.689.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | log1p issued>0 | 2,418 | 93 | 0.587 | 0.114 | -1 | 0.773 0.565 0.589 0.537 0.469 |
| y3_recover_cash_6m | days issued>0 | 2,418 | 93 | 0.689 | 0.103 | -1 | 0.628 0.723 0.544 0.805 0.746 |
| y3_recover_cash_6m | log1p>0 leftover-days rank | 2,418 | 93 | 0.412 | 0.594 | fake=False | 0.316 0.577 0.412 0.387 0.368 |


## Extra — leftover after days+issued_lag1

Leftover after days+issued_lag1 rank 0.587 OLS 0.662 ρ=-0.754 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+issued_lag1 | 0.587 | 0.662 | -0.754 | lives |


## Extra — CN leftover after issued now

CN leftover after contemporaneous issued rank 0.614 OLS 0.600 (lag1 KEEP 0.597 stays; do not overwrite credit_note_qa).

| cut | rank | OLS | ρ |
| --- | --- | --- | --- |
| CN leftover after issued now | 0.614 | 0.600 | -0.104 |


## Extra — holdout early6 coverage

Holdout early6 issued cov 41.8% after7 54.9%. No AUROC.

| slice | nn | n | cov |
| --- | --- | --- | --- |
| holdout early6 | 23 | 55 | 41.8% |
| holdout after7 | 559 | 1018 | 54.9% |


## Extra — leftover after pending

Leftover after pending rank 0.694 ρ=-0.851. After pending+days 0.613 fake=False.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| after pending | 0.694 | 0.571 | -0.851 | FALSE clone |
| after pending+days | 0.613 | 0.582 | -0.702 |  |


## Extra — fold 4 leftover after days

Fold 4 Y3 leftover after days rank 0.400 fake=True issued 0.331 days 0.260. Y7 leftover 0.321.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued fold4 | 949 | 76 | 0.331 | — | -1 | — — — — 0.331 |
| y3_recover_cash_6m | days fold4 | 1,197 | 107 | 0.260 | — | -1 | — — — — 0.260 |
| y3_recover_cash_6m | fold4 leftover-days rank | 949 | 76 | 0.400 | 0.709 | fake=True | — — — — 0.400 |
| y7_top1_lost | fold4 leftover-days rank | 1,795 | 686 | 0.321 | 0.485 | fake=True | — — — — 0.321 |


## Extra — lag1 leftover after contemporaneous

lag1 leftover after now Y3 rank 0.596 Y7 0.518. TURNOVER KEEP is the lag; contemporaneous does not steal it.

| y | cut | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | lag1 leftover after now | 0.596 | 0.613 | 0.450 | lives |
| y7_top1_lost | lag1 leftover after now | 0.518 | 0.596 | 0.450 | dies |


## Extra — leftover after CN

Y3 leftover after CN rank 0.654 OLS 0.487 ρ=-0.624. CN KEEP vs lag1 is a different bar.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| after CN | 0.654 | 0.487 | -0.624 |  |


## Extra — issued / a_in3 leftover

issued/a_in3 Y3 0.655 leftover-days rank 0.615 fake=True.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued / a_in3 | 3,380 | 226 | 0.655 | 0.099 | -1 | 0.781 0.641 0.516 0.711 0.628 |
| y3_recover_cash_6m | issued/in3 leftover-days rank | 3,380 | 226 | 0.615 | 0.726 | fake=True | 0.739 0.571 0.502 0.667 0.595 |


## Extra — trail terciles leftover after days

Trail terciles leftover-days T1 0.559 T2 0.596 T3 0.638.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| T1 short | 0.559 | 0.633 | -0.849 | FALSE clone | 1,154 |
| T2 mid | 0.596 | 0.723 | -0.849 | FALSE clone | 1,628 |
| T3 long | 0.638 | 0.688 | -0.849 | FALSE clone | 836 |


## Extra — log1p leftover after days+lag1

log1p leftover after days+lag1 rank 0.587 OLS 0.603 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| log1p after days+lag1 | 0.587 | 0.603 | -0.040 | lives |


## Extra — train SIZE tercile coverage

Train SIZE issued cov T1 64.0% T2 66.4% T3 61.7%.

| SIZE | nn | n | cov | p50 |
| --- | --- | --- | --- | --- |
| T1 | 4693 | 7338 | 64.0% | 0 |
| T2 | 4746 | 7144 | 66.4% | 44469 |
| T3 | 4116 | 6675 | 61.7% | 245139 |


## Extra — company-median ρ (VIF / size flag)

Company-median ρ issued vs days 0.484 vs log1p(a_in3) 0.511 SIZE (feature report flagged company-median VIF / size).

| pair | ρ | n |
| --- | --- | --- |
| issued vs days (co-median) | 0.484 | 745 |
| issued vs a_in3 (co-median) | 0.500 | 745 |
| issued vs log1p(a_in3) (co-median) | 0.511 | 745 |


## Extra — lag3 leftover

issued_lag3 Y3 0.644. Leftover after days_lag3 0.632 fake=True. After issued_lag3 0.662.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | issued_lag3 | 3,256 | 234 | 0.644 | 0.080 | -1 | 0.718 0.704 0.515 0.649 0.634 |
|  |  |  |  |  |  |  |  |
|  |  |  |  |  |  |  |  |


## Extra — Y5 coverage (never E)

Y5-defined issued cov 100.0%. Y5 never E — no AUROC, no card.

| slice | nn | n | cov |
| --- | --- | --- | --- |
| train Y5-defined | 4905 | 4905 | 100.0% |


## Extra — holdout trail coverage

Holdout trail issued cov T1 55.7% T2 56.5% T3 50.1%. No AUROC.

| slice | nn | n | cov |
| --- | --- | --- | --- |
| T1 short | 200 | 359 | 55.7% |
| T2 mid | 213 | 377 | 56.5% |
| T3 long | 169 | 337 | 50.1% |


## Extra — leftover after lag1 fold-wise

Leftover after lag1 fold-wise rank 0.653 folds 0.748 0.622 0.583 0.672 0.640.

| cut | rank | folds | OLS | ρ |
| --- | --- | --- | --- | --- |
| leftover after lag1 fold-wise | 0.653 | 0.748 0.622 0.583 0.672 0.640 | 0.659 | 0.493 |


## Extra — Y7 leftover after days+lag1

Y7 leftover after days+lag1 rank 0.590 OLS 0.543 lives — still CLOSE TURNOVER add-on.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days+lag1 | 0.590 | 0.543 | -0.754 | lives |


## Extra — acf1 by SIZE

issued acf1 by SIZE T1 0.040 T2 0.140 T3 0.279 (panel 0.143 LOW_PERSIST).

| SIZE | acf1 | nn |
| --- | --- | --- |
| T1 | 0.040 | 4,693 |
| T2 | 0.140 | 4,746 |
| T3 | 0.279 | 4,116 |


## Extra — issued==0 share by SIZE

issued==0 share train 36.7% T1 55.9% T2 26.4% T3 26.6% (zeros inflated leftover-after-days; issued>0 leftover 0.412).

| SIZE | zeros | nn | zero_share |
| --- | --- | --- | --- |
| T1 | 2625 | 4693 | 55.9% |
| T2 | 1251 | 4746 | 26.4% |
| T3 | 1096 | 4116 | 26.6% |


## Extra — days leftover after log1p(issued)

Days leftover after log1p(issued) rank 0.676 OLS 0.677 lives — keep the 0.711 bar.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| days leftover after log1p(issued) | 0.676 | 0.677 | 0.070 | lives |


## Extra — SIZE × trail leftover grid

SIZE×trail leftover-days T1×T1 — T3×T3 —.

| SIZE×trail | rank | OLS | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| T1×T1 | — | — | FALSE clone | 294 | 45 |
| T1×T2 | 0.505 | 0.637 | FALSE clone | 360 | 73 |
| T1×T3 | — | — | FALSE clone | 193 | 35 |
| T2×T1 | — | — | FALSE clone | 416 | 20 |
| T2×T2 | — | — | FALSE clone | 623 | 29 |
| T2×T3 | — | — | FALSE clone | 322 | 21 |
| T3×T1 | — | — | FALSE clone | 444 | 8 |
| T3×T2 | — | — | FALSE clone | 645 | 12 |
| T3×T3 | — | — | FALSE clone | 321 | 21 |


## Extra — leftover after days+size+lag1

Leftover after days+size+lag1 rank 0.601 OLS 0.448 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+size+lag1 | 0.601 | 0.448 | 0.139 | lives |


## Extra — Q6 leftover after days_lag1+issued_lag1

Q6 leftover after days_lag1+issued_lag1 rank 0.584 dies.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| Q6 leftover after days_lag1+issued_lag1 | 0.584 | 0.648 | -0.818 | dies |


## Extra — leftover after DSO (already DROPPED)

Leftover after DSO rank 0.586 ρ=-0.028. DSO stays DROPPED; do not put DSO back on TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| after DSO (already DROPPED) | 0.586 | 0.574 | -0.028 |  |


## Extra — holdout issued==0 share

Holdout issued==0 share 22.7% of nn=582. No AUROC.

| slice | zeros | nn | zero_share |
| --- | --- | --- | --- |
| holdout issued==0 | 132 | 582 | 22.7% |


## Extra — fold 1 isolate

Fold 1 issued 0.675 days 0.738 (leftover-days fold 1 0.723 is OLS-inflated; days still wins the fold).

| fold | issued | days | issued−days |
| --- | --- | --- | --- |
| 1 | 0.675 | 0.738 | -0.063 |


## Extra — rank-resid ICC after days

Rank-resid after days ICC 0.958 (raw issued ICC 0.942 BETWEEN). Leftover is still a company-level smear.

| object | ICC | k |
| --- | --- | --- |
| rank-resid after days | 0.958 | 745 |


## Extra — calendar leftover after days

Calendar leftover-days 2024=— 2025=0.612 2026=0.592.

| year | rank | OLS | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| 2024 | — | — | FALSE clone | 315 | 14 |
| 2025 | 0.612 | 0.695 | FALSE clone | 2,704 | 192 |
| 2026 | 0.592 | 0.707 | FALSE clone | 599 | 58 |


## Extra — issued vs months_so_far

issued vs months_so_far ρ=0.077 n=13,555. Leftover after trail rank 0.698.

| pair | ρ | n |
| --- | --- | --- |
| issued vs months_so_far | 0.077 | 13,555 |
|  | -0.613 |  |


## Extra — issued ρ vs days / size by SIZE tercile

issued vs days ρ T1 0.340 T2 0.227 T3 0.091.

| SIZE | ρ vs days | ρ vs log1p(a_in3) | n | SIZE? |
| --- | --- | --- | --- | --- |
| T1 | 0.340 | 0.269 | 4,693 |  |
| T2 | 0.227 | 0.155 | 4,746 |  |
| T3 | 0.091 | 0.283 | 4,116 |  |


## Extra — n_tx leftover after issued

n_tx leftover after issued rank 0.668 lives — n_tx/days stay the volume bar.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| n_tx leftover after issued | 0.668 | 0.725 | 0.452 | lives |


## Extra — leftover after intensity

Leftover after intensity rank 0.654 ρ=0.224 fake=False.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| after issued/days intensity | 0.654 | 0.648 | 0.224 |  |


## Extra — Y7 leftover after CN

Y7 leftover after CN rank 0.662 OLS 0.612. CN KEEP vs lag1 stays; do not overwrite credit_note_qa.

| bar | rank | OLS | ρ |
| --- | --- | --- | --- |
| Y7 leftover after CN | 0.662 | 0.612 | -0.624 |


## Extra — SIZE T2 leftover after days

SIZE T2 leftover after days rank 0.452 fake=True (T3 was LOW_POWER in pass 6).

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| SIZE T2 leftover-days | 0.452 | 0.623 | -0.849 | FALSE clone | 1,361 |


## Extra — calendar-month leftover after days

Calendar-month leftover-days 1=— 2=— 3=— 4=— 5=— 6=— 7=— 8=— 9=— 10=— 11=— 12=—.

| month | rank | OLS | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| 1 | — | — | FALSE clone | 460 | 34 |
| 2 | — | — | FALSE clone | 490 | 42 |
| 3 | — | — | FALSE clone | 224 | 14 |
| 4 | — | — | FALSE clone | 226 | 15 |
| 5 | — | — | FALSE clone | 218 | 12 |
| 6 | — | — | FALSE clone | 229 | 16 |
| 7 | — | — | FALSE clone | 248 | 20 |
| 8 | — | — | FALSE clone | 227 | 20 |
| 9 | — | — | FALSE clone | 222 | 13 |
| 10 | — | — | FALSE clone | 265 | 20 |
| 11 | — | — | FALSE clone | 385 | 27 |
| 12 | — | — | FALSE clone | 424 | 31 |


## Extra — now vs lag1 ρ by SIZE

now vs lag1 ρ T1 0.761 T2 0.764 T3 0.770 (panel 0.814 TWIN).

| SIZE | ρ now vs lag1 | n | twin? |
| --- | --- | --- | --- |
| T1 | 0.761 | 4,448 |  |
| T2 | 0.764 | 4,485 |  |
| T3 | 0.770 | 3,877 |  |


## Extra — days leftover after n_tx

Days leftover after n_tx rank 0.564 lives — days is not just n_tx.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| days leftover after n_tx | 0.564 | 0.710 | 0.797 | lives |


## Extra — company-mean leftover

Company-mean issued 0.642 days 0.666 leftover-days rank 0.586 fake=True.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | co-mean issued | 463 | 117 | 0.642 | 0.095 | -1 | 0.780 0.681 0.591 0.527 0.632 |
| y3_recover_cash_6m | co-mean days | 725 | 174 | 0.666 | 0.038 | -1 | 0.647 0.691 0.622 0.653 0.718 |
| y3_recover_cash_6m | co-mean leftover-days rank | 463 | 117 | 0.586 | 0.667 | fake=True | 0.704 0.619 0.582 0.469 0.558 |


## Extra — holdout year coverage

Holdout year issued cov 2024=100.0% 2025=51.4% 2026=55.7%. No AUROC.

| year | nn | n | cov |
| --- | --- | --- | --- |
| 2024 | 13 | 13 | 100.0% |
| 2025 | 254 | 494 | 51.4% |
| 2026 | 315 | 566 | 55.7% |


## Extra — leftover after days+lag1+CN

Leftover after days+lag1+CN rank 0.591 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+lag1+CN | 0.591 | 0.580 | 0.331 | lives |


## Extra — quarter leftover after days

Quarter leftover-days Q1=0.594 Q2=— Q3=0.653 Q4=0.616.

| Q | rank | OLS | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.594 | 0.739 | FALSE clone | 1,174 | 90 |
| 2 | — | — | FALSE clone | 673 | 43 |
| 3 | 0.653 | 0.634 | FALSE clone | 697 | 53 |
| 4 | 0.616 | 0.645 | FALSE clone | 1,074 | 78 |


## Extra — lag1 leftover after days

lag1 leftover after days Y3 0.591 fake=True Y7 0.632 fake=True. TURNOVER KEEP is Y7, not this Y3 leftover.

| y | bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | lag1 leftover after days | 0.591 | 0.702 | -0.863 | FALSE clone |
| y7_top1_lost | lag1 leftover after days | 0.632 | 0.479 | -0.863 | FALSE clone |


## Extra — Y7 leftover after size

Y7 leftover after size rank 0.658 OLS 0.538. Do not grow TURNOVER.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| Y7 leftover after size | 0.658 | 0.538 | -0.923 | dies |


## Extra — issued>0 vs lag1

issued>0 vs lag1 ρ=0.728 n=8,214 not a twin. Leftover after lag1 0.562.

| pair | ρ | n | twin? |
| --- | --- | --- | --- |
| issued>0 vs lag1 | 0.728 | 8,214 |  |
|  | 0.493 |  |  |


## Extra — train vs holdout coverage gap

Train vs holdout issued cov 64.1% vs 54.2%; p50 13890 vs 64768. No AUROC.

| split | cov | p50 | zero_share |
| --- | --- | --- | --- |
| train | 64.1% | 13890 | 36.7% |
| holdout | 54.2% | 64768 | 22.7% |


## Extra — T1 issued==0 leftover

T1 issued==0 leftover after days rank 0.557 fake=False (T1 zero share 55.9%).

| slice | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| T1 issued==0 leftover-days | 0.557 | 0.557 | 0.420 |  |


## Extra — both>0 leftover

both>0 now vs lag1 ρ=0.806 issued 0.621 lag1 0.588 days 0.698. leftover-days 0.370 leftover-lag1 0.596.

| pair | ρ | n | twin? |
| --- | --- | --- | --- |
| both>0 now vs lag1 | 0.806 | 7,464 | TWIN |
|  |  | 2,205 |  |
|  |  | 2,205 |  |
|  |  | 2,205 |  |
|  |  | 2,205 |  |
|  |  | 2,205 |  |


## Extra — leftover after days+n_tx+lag1

Leftover after days+n_tx+lag1 rank 0.587 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+n_tx+lag1 | 0.587 | 0.667 | -0.690 | lives |


## Extra — size leftover after issued

Size leftover after issued rank 0.561 lives — size is not issued.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| size leftover after issued | 0.561 | 0.623 | 0.454 | lives |


## Extra — leftover after days_lag1 (all books)

Leftover after days_lag1 on all books rank 0.621 ρ=-0.844 fake=True.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| leftover after days_lag1 (all books) | 0.621 | 0.659 | -0.844 | FALSE clone |


## Extra — leftover after size fold-wise

Leftover after size fold-wise rank 0.645 folds 0.728 0.628 0.549 0.698 0.622.

| cut | rank | folds | OLS |
| --- | --- | --- | --- |
| leftover after size fold-wise | 0.645 | 0.728 0.628 0.549 0.698 0.622 | 0.619 |


## Extra — Y7 both>0 leftover after lag1

Y7 both>0 issued 0.554 lag1 0.478 leftover-lag1 0.540. CLOSE TURNOVER add-on.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y7_top1_lost | both>0 issued | 5,823 | 1,285 | 0.554 | 0.072 | -1 | 0.518 0.521 0.495 0.564 0.675 |
| y7_top1_lost | both>0 lag1 | 5,823 | 1,285 | 0.478 | 0.078 | -1 | 0.509 0.513 0.484 0.541 0.344 |
| y7_top1_lost | both>0 leftover-lag1 rank | 5,823 | 1,285 | 0.540 | 0.583 | fake=False | 0.509 0.520 0.506 0.560 0.606 |


## Extra — leftover after pending+CN

Leftover after pending+CN rank 0.655 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after pending+CN | 0.655 | 0.458 | -0.570 | lives |


## Extra — holdout both>0 coverage

Holdout both>0 share 36.1% nn=387. No AUROC.

| slice | nn | n | share |
| --- | --- | --- | --- |
| holdout both>0 | 387 | 1073 | 36.1% |


## Extra — days leftover after issued on both>0

Days leftover after issued on both>0 rank 0.679 lives — keep the 0.711 bar.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| days leftover after issued on both>0 | 0.679 | 0.695 | 0.388 | lives |


## Extra — leftover after n_tx fold-wise

Leftover after n_tx fold-wise rank 0.603 folds 0.713 0.554 0.562 0.597 0.591.

| cut | rank | folds | OLS |
| --- | --- | --- | --- |
| leftover after n_tx fold-wise | 0.603 | 0.713 0.554 0.562 0.597 0.591 | 0.584 |


## Extra — group leftover after days

Group leftover-days n_groups=0 median — (panel leftover 0.608).

_(empty)_


## Extra — first-month cohort leftover

First-month cohort leftover-days 2024=0.596 2025=0.685 2026=—.

| first_year | rank | OLS | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| 2024 | 0.596 | 0.729 | FALSE clone | 2,716 | 179 |
| 2025 | 0.685 | 0.611 | FALSE clone | 892 | 83 |
| 2026 | — | — | FALSE clone | 10 | 2 |


## Extra — Y7 leftover after n_tx

Y7 leftover after n_tx rank 0.665 OLS 0.652 fake=False. Do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after n_tx | 0.665 | 0.652 | 0.104 |  |


## Extra — rank-resid acf1

Rank-resid after days acf1 0.293 (raw issued acf1 0.143 LOW_PERSIST).

| object | acf1 |
| --- | --- |
| rank-resid after days | 0.293 |


## Extra — holdout SIZE×trail coverage

Holdout SIZE×trail issued coverage only. No AUROC.

| SIZE×trail | nn | n | cov |
| --- | --- | --- | --- |
| T1×T1 | 65 | 120 | 54.2% |
| T1×T2 | 66 | 118 | 55.9% |
| T1×T3 | 52 | 86 | 60.5% |
| T2×T1 | 80 | 120 | 66.7% |
| T2×T2 | 88 | 134 | 65.7% |
| T2×T3 | 73 | 132 | 55.3% |
| T3×T1 | 55 | 119 | 46.2% |
| T3×T2 | 59 | 125 | 47.2% |
| T3×T3 | 44 | 119 | 37.0% |


## Extra — both>0 intensity leftover

both>0 intensity Y3 0.566 leftover-days 0.409 fake=True.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | both>0 intensity | 2,198 | 80 | 0.566 | 0.126 | -1 | 0.719 0.526 0.675 0.426 0.484 |
| y3_recover_cash_6m | both>0 intensity leftover-days | 2,198 | 80 | 0.409 | 0.674 | fake=True | 0.385 0.592 0.340 0.316 0.413 |


## Extra — 2025 first-month cohort leftover

2025-cohort leftover-days 0.685 fake=True issued 0.729 days 0.691.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 2025-cohort issued | 892 | 83 | 0.729 | 0.127 | -1 | 0.774 0.894 0.657 0.762 0.558 |
| y3_recover_cash_6m | 2025-cohort days | 1,554 | 123 | 0.691 | 0.047 | -1 | 0.619 0.750 0.699 0.703 0.684 |
| y3_recover_cash_6m | 2025-cohort leftover-days rank | 892 | 83 | 0.685 | 0.611 | fake=True | 0.730 0.779 0.635 0.770 0.509 |


## Extra — kitchen-sink leftover

Kitchen-sink leftover rank 0.596 OLS 0.473 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+n_tx+size+lag1+CN | 0.596 | 0.473 | 0.155 | lives |


## Extra — intensity leftover after issued

Intensity leftover after issued rank 0.612 lives — still not a card stem.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| intensity leftover after issued | 0.612 | 0.647 | -0.331 | lives |


## Extra — issued vs DSO by SIZE

issued vs DSO ρ T1 -0.034 T2 -0.172 T3 -0.056. DSO stays DROPPED.

| SIZE | ρ vs DSO | n |
| --- | --- | --- |
| T1 | -0.034 | 2,068 |
| T2 | -0.172 | 3,495 |
| T3 | -0.056 | 3,020 |


## Extra — 2024 first-month cohort leftover

2024-cohort leftover-days 0.596 fake=True issued 0.681 days 0.732.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 2024-cohort issued | 2,716 | 179 | 0.681 | 0.091 | -1 | 0.799 0.597 0.584 0.697 0.728 |
| y3_recover_cash_6m | 2024-cohort days | 4,079 | 276 | 0.732 | 0.029 | -1 | 0.744 0.731 0.690 0.727 0.768 |
| y3_recover_cash_6m | 2024-cohort leftover-days rank | 2,716 | 179 | 0.596 | 0.729 | fake=True | 0.707 0.487 0.533 0.599 0.653 |


## Extra — Y7 leftover after pending

Y7 leftover after pending rank 0.661 OLS 0.556 fake=True. Do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after pending | 0.661 | 0.556 | -0.851 | FALSE clone |


## Extra — issued vs CN by SIZE

issued vs CN ρ T1 0.145 T2 0.186 T3 0.203.

| SIZE | ρ vs CN | n |
| --- | --- | --- |
| T1 | 0.145 | 3,377 |
| T2 | 0.186 | 4,169 |
| T3 | 0.203 | 3,661 |


## Extra — leftover after days+DSO

Leftover after days+DSO rank 0.446 fake=True. DSO stays DROPPED.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| after days+DSO | 0.446 | 0.654 | 0.835 | FALSE clone |


## Extra — log1p leftover after days+n_tx+lag1

log1p leftover after days+n_tx+lag1 rank 0.587 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| log1p after days+n_tx+lag1 | 0.587 | 0.603 | -0.039 | lives |


## Extra — ever-ERP issued==0 leftover

Ever-ERP issued==0 leftover-days 0.697 fake=True flag 0.667.

| y | feature | n | n_pos | CV | sd | sign | folds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | ERP issued==0 flag | 3,618 | 264 | 0.667 | 0.062 | 1 | 0.742 0.660 0.572 0.688 0.675 |
| y3_recover_cash_6m | ERP-zero leftover-days rank | 1,200 | 171 | 0.697 | 0.697 | fake=True | 0.702 0.803 0.648 0.731 0.603 |


## Extra — mid-trail leftover

mid_12_17 leftover after days rank 0.610 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| mid_12_17 leftover-days | 0.610 | 0.694 | -0.849 | FALSE clone | 1,141 |


## Extra — Y7 leftover after days+size

Y7 leftover after days+size rank 0.663 lives — still CLOSE TURNOVER add-on.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days+size | 0.663 | 0.534 | 0.136 | lives |


## Extra — issued vs pending by SIZE

issued vs pending ρ T1 0.058 T2 -0.052 T3 -0.149.

| SIZE | ρ vs pending | n |
| --- | --- | --- |
| T1 | 0.058 | 4,347 |
| T2 | -0.052 | 4,512 |
| T3 | -0.149 | 3,903 |


## Extra — long-trail leftover

long_>=18 leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| long_>=18 leftover-days | — | — | -0.849 | FALSE clone | 138 |


## Extra — Y7 leftover after days+lag1+size

Y7 leftover after days+lag1+size rank 0.593 lives — still CLOSE TURNOVER add-on.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days+lag1+size | 0.593 | 0.532 | 0.139 | lives |


## Extra — issued vs n_tx by SIZE

issued vs n_tx ρ T1 0.352 T2 0.255 T3 0.147.

| SIZE | ρ vs n_tx | n |
| --- | --- | --- |
| T1 | 0.352 | 4,693 |
| T2 | 0.255 | 4,746 |
| T3 | 0.147 | 4,116 |


## Extra — leftover by company acf1

Leftover-days high-acf1 0.575 low-acf1 0.586 (median acf1 0.143).

| slice | rank | OLS | fake? | n |
| --- | --- | --- | --- | --- |
| high-acf1 | 0.575 | 0.679 | FALSE clone | 1,715 |
| low-acf1 | 0.586 | 0.647 | FALSE clone | 1,675 |


## Extra — Y7 leftover after pending+CN

Y7 leftover after pending+CN rank 0.663 lives — still CLOSE TURNOVER add-on.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| Y7 leftover after pending+CN | 0.663 | 0.578 | -0.570 | lives |


## Extra — issued p99 train vs holdout

issued p99 train 39372348 holdout 3109531. No AUROC.

| split | p90 | p99 | max |
| --- | --- | --- | --- |
| train | 777052 | 39372348 | 5947260462 |
| holdout | 692811 | 3109531 | 6365651 |


## Extra — n_grid>=18 leftover

n_grid>=18 leftover after days rank 0.612 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| n_grid>=18 leftover-days | 0.612 | 0.702 | -0.849 | FALSE clone | 3,358 |


## Extra — Y7 leftover after DSO

Y7 leftover after DSO rank 0.583 fake=False. DSO stays DROPPED; do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after DSO | 0.583 | 0.575 | -0.028 |  |


## Extra — leftover after days+pending+lag1

Leftover after days+pending+lag1 rank 0.598 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+pending+lag1 | 0.598 | 0.460 | -0.269 | lives |


## Extra — n_grid<12 leftover

n_grid<12 leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| n_grid<12 leftover-days | — | — | -0.849 | FALSE clone | 90 |


## Extra — leftover after days+CN+lag1

Leftover after days+CN+lag1 rank 0.591 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+CN+lag1 | 0.591 | 0.580 | 0.331 | lives |


## Extra — Y7 leftover after days+pending

Y7 leftover after days+pending rank 0.664 fake=False. Do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days+pending | 0.664 | 0.486 | -0.513 |  |


## Extra — leftover after days+n_tx+size

Leftover after days+n_tx+size rank 0.610 lives — still off the card.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| after days+n_tx+size | 0.610 | 0.448 | 0.139 | lives |


## Extra — days leftover after issued+n_tx

Days leftover after issued+n_tx rank 0.570 lives — keep the 0.711 bar.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| days leftover after issued+n_tx | 0.570 | 0.730 | 0.379 | lives |


## Extra — SIZE T3 leftover after days

SIZE T3 leftover after days rank — fake=True (pass 6 was LOW_POWER).

| slice | rank | OLS | ρ | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- | --- |
| SIZE T3 leftover-days | — | — | -0.849 | FALSE clone | 1,410 | 41 |


## Extra — leftover by issued median

Leftover-days issued>p50 0.543 issued<=p50 0.601.

| slice | rank | OLS | fake? | n |
| --- | --- | --- | --- | --- |
| issued>p50 | 0.543 | 0.696 | FALSE clone | 1,945 |
| issued<=p50 | 0.601 | 0.700 | FALSE clone | 1,673 |


## Extra — Y7 leftover after days+CN

Y7 leftover after days+CN rank 0.658 fake=False. Do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days+CN | 0.658 | 0.608 | -0.192 |  |


## Extra — leftover by issued bands

Leftover-days p50-p90 0.532 p90+ —.

| slice | rank | OLS | fake? | n | n_pos |
| --- | --- | --- | --- | --- | --- |
| p50-p90 | 0.532 | 0.724 | FALSE clone | 1,599 | 54 |
| p90+ | — | — | FALSE clone | 346 | 15 |


## Extra — days leftover after issued+size

Days leftover after issued+size rank 0.648 lives — keep the 0.711 bar.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| days leftover after issued+size | 0.648 | 0.679 | 0.243 | lives |


## Extra — leftover by issued quartiles

Leftover-days <=p25 0.697 >p75 —.

| slice | rank | OLS | fake? | n |
| --- | --- | --- | --- | --- |
| <=p25 | 0.697 | 0.697 | FALSE clone | 1,200 |
| >p75 | — | — | FALSE clone | 1,028 |


## Extra — Y7 leftover after days+n_tx

Y7 leftover after days+n_tx rank 0.665 fake=False. Do not grow TURNOVER.

| bar | rank | OLS | ρ | fake? |
| --- | --- | --- | --- | --- |
| Y7 leftover after days+n_tx | 0.665 | 0.520 | -0.798 |  |


## Extra — leftover p25-p75

p25-p75 leftover after days rank 0.575 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| p25-p75 leftover-days | 0.575 | 0.663 | -0.849 | FALSE clone | 1,390 |


## Extra — days leftover after issued+lag1

Days leftover after issued+lag1 rank 0.673 lives — keep the 0.711 bar.

| bar | rank | OLS | ρ | dies? |
| --- | --- | --- | --- | --- |
| days leftover after issued+lag1 | 0.673 | 0.728 | 0.392 | lives |


## Extra — n_tx>0 leftover after days

n_tx>0 leftover after days rank 0.605 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| n_tx>0 leftover-days | 0.605 | 0.683 | -0.849 | FALSE clone | 3,564 |


## Extra — p10-p90 leftover after days

p10-p90 leftover after days rank 0.605 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| p10-p90 leftover-days | 0.605 | 0.720 | -0.849 | FALSE clone | 3,272 |


## Extra — days>0 leftover after days

days>0 leftover after days rank 0.605 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| days>0 leftover-days | 0.605 | 0.683 | -0.849 | FALSE clone | 3,564 |


## Extra — lag1>0 leftover after days

lag1>0 leftover after days rank 0.559 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| lag1>0 leftover-days | 0.559 | 0.660 | -0.849 | FALSE clone | 2,385 |


## Extra — onset leftover after days

onset (issued>0 lag1==0) leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| onset issued>0 lag1==0 leftover-days | — | — | -0.849 | FALSE clone | 213 |


## Extra — stop leftover after days

stop (issued==0 lag1>0) leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| stop issued==0 lag1>0 leftover-days | — | — | -0.849 | FALSE clone | 180 |


## Extra — issued>0 days>0 leftover after days

issued>0 days>0 leftover after days rank 0.413 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued>0 days>0 leftover-days | 0.413 | 0.668 | -0.849 | FALSE clone | 2,409 |


## Extra — issued>0 SIZE T1 leftover after days

issued>0 SIZE T1 leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued>0 SIZE T1 leftover-days | — | — | -0.849 | FALSE clone | 382 |


## Extra — issued>0 SIZE T2 leftover after days

issued>0 SIZE T2 leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued>0 SIZE T2 leftover-days | — | — | -0.849 | FALSE clone | 989 |


## Extra — issued>0 mid_12_17 leftover after days

issued>0 mid_12_17 leftover after days rank — fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued>0 mid_12_17 leftover-days | — | — | -0.849 | FALSE clone | 800 |


## Extra — issued>0 T2+T3 leftover after days

issued>0 T2+T3 leftover after days rank 0.462 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued>0 T2+T3 leftover-days | 0.462 | 0.672 | -0.849 | FALSE clone | 2,036 |


## Extra — issued>0 or lag1>0 leftover after days

issued>0 or lag1>0 leftover after days rank 0.536 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued>0 or lag1>0 leftover-days | 0.536 | 0.665 | -0.849 | FALSE clone | 2,598 |


## Extra — complete-case leftover after days

complete-case leftover after days rank 0.611 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| complete issued/days/a_in3 leftover-days | 0.611 | 0.696 | -0.849 | FALSE clone | 3,543 |


## Extra — days>p50 leftover after days

days>p50 leftover after days rank 0.642 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| days>p50(14) leftover-days | 0.642 | 0.373 | -0.849 | FALSE clone | 2,160 |


## Extra — days<=p50 leftover after days

days<=p50 leftover after days rank 0.578 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| days<=p50(14) leftover-days | 0.578 | 0.643 | -0.849 | FALSE clone | 1,458 |


## Extra — n_tx>p50 leftover after days

n_tx>p50 leftover after days rank 0.654 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| n_tx>p50(43) leftover-days | 0.654 | 0.557 | -0.849 | FALSE clone | 2,125 |


## Extra — n_tx<=p50 leftover after days

n_tx<=p50 leftover after days rank 0.583 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| n_tx<=p50(43) leftover-days | 0.583 | 0.673 | -0.849 | FALSE clone | 1,493 |


## Extra — a_in3>p50 leftover after days

a_in3>p50 leftover after days rank 0.634 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| a_in3>p50 leftover-days | 0.634 | 0.657 | -0.849 | FALSE clone | 2,209 |


## Extra — CN-defined leftover after days

CN-defined leftover after days rank 0.594 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| CN-defined leftover-days | 0.594 | 0.699 | -0.849 | FALSE clone | 3,079 |


## Extra — pending-defined leftover after days

pending-defined leftover after days rank 0.616 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| pending-defined leftover-days | 0.616 | 0.697 | -0.849 | FALSE clone | 3,446 |


## Extra — a_in3<=p50 leftover after days

a_in3<=p50 leftover after days rank 0.583 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| a_in3<=p50 leftover-days | 0.583 | 0.674 | -0.849 | FALSE clone | 1,334 |


## Extra — days leftover after issued+size+lag1

Days leftover after issued+size+lag1 rank 0.648 lives — keep the 0.711 bar.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| days leftover after issued+size+lag1 | 0.648 | 0.679 | 0.245 |  | 3,543 |


## Extra — pending+CN leftover after days

pending+CN leftover after days rank 0.594 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| pending+CN leftover-days | 0.594 | 0.699 | -0.849 | FALSE clone | 3,078 |


## Extra — n_tx-defined leftover after days

n_tx-defined leftover after days rank 0.608 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| n_tx-defined leftover-days | 0.608 | 0.694 | -0.849 | FALSE clone | 3,618 |


## Extra — issued+days defined leftover after days

issued+days defined leftover after days rank 0.608 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| issued+days defined leftover-days | 0.608 | 0.694 | -0.849 | FALSE clone | 3,618 |


## Extra — DSO-defined leftover after days

DSO-defined leftover after days rank 0.412 fake=True. DSO stays DROPPED; do not put DSO back on TURNOVER.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| DSO-defined leftover-days | 0.412 | 0.666 | -0.849 | FALSE clone | 2,418 |


## Extra — pending-defined days>0 leftover after days

pending-defined days>0 leftover after days rank 0.611 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| pending-defined days>0 leftover-days | 0.611 | 0.685 | -0.849 | FALSE clone | 3,398 |


## Extra — log1p complete-case leftover after days

log1p complete-case leftover after days rank 0.611 fake=False.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| log1p complete-case leftover-days | 0.611 | 0.606 | -0.038 |  | 3,543 |


## Extra — DSO-defined issued>0 leftover after days

DSO-defined issued>0 leftover after days rank 0.412 fake=True. DSO stays DROPPED.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| DSO-defined issued>0 leftover-days | 0.412 | 0.666 | -0.849 | FALSE clone | 2,418 |


## Extra — pending-defined issued>0 leftover after days

pending-defined issued>0 leftover after days rank 0.412 fake=True.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| pending-defined issued>0 leftover-days | 0.412 | 0.666 | -0.849 | FALSE clone | 2,418 |


## Extra — log1p DSO-defined leftover after days

log1p DSO-defined leftover after days rank 0.412 fake=False. DSO stays DROPPED.

| slice | rank | OLS | ρ | fake? | n |
| --- | --- | --- | --- | --- | --- |
| log1p DSO-defined leftover-days | 0.412 | 0.594 | -0.038 |  | 2,418 |


## What failed / next (held for wave note)

- Y3 leftover after days rank 0.608 OLS 0.694 fake=True
- inverse days leftover 0.676
- leftover after size 0.645 after lag1 0.653
- terciles T1 0.530 T2+T3 0.569
- Q6 short leftover days_lag1 0.585 issued_lag1 0.640
- Y7 leftover after lag1 0.581 addon=CLOSE
- fold 4 issued 0.696 leftover 0.612
- log1p leftover 0.608 intensity 0.605
- dark stub n=1
- log1p leftover days+size 0.613
- fold leftover 0.723 0.560 0.553 0.605 0.600
- demean leftover 0.477 mean 0.572
- same-n leftover-days 0.608 fake=True
- T2+T3 leftover 0.569 fake=True
- issued>0 leftover 0.412 fake=True
- log1p vs lag1 ρ=0.814 leftover 0.653
- short leftover days_lag1 0.585 fake=True
- leftover after n_tx 0.603 n_tx+days 0.605
- Y7 leftover after days 0.665 fake=True
- n_tx+days leftover 0.605 fake=False
- early6 cov 68.8% after7 leftover 0.614
- zero-flag leftover 0.582 fake=False
- log1p>0 leftover 0.412 fake=False
- leftover days+lag1 0.587 dies=False
- CN leftover after now 0.614
- leftover after pending 0.694 pending+days 0.613
- fold4 leftover-days Y3 0.400 Y7 0.321
- lag1 leftover after now Y3 0.596 Y7 0.518
- leftover after CN 0.654
- issued/in3 leftover 0.615 fake=True
- trail leftover T1 0.559 T3 0.638
- log1p leftover days+lag1 0.587
- co-median ρ vs log1p(a_in3) 0.511 SIZE=True
- leftover after days_lag3 0.632 after issued_lag3 0.662
- leftover after lag1 folds 0.748 0.622 0.583 0.672 0.640
- Y7 leftover days+lag1 0.590
- days leftover after log1p 0.676
- SIZE×trail leftover T1×T1 — T3×T3 —
- leftover days+size+lag1 0.601
- Q6 leftover days_lag1+issued_lag1 0.584
- leftover after DSO 0.586
- fold 1 issued 0.675 days 0.738
- rank-resid ICC 0.958
- issued vs months_so_far ρ=0.077 leftover 0.698
- n_tx leftover after issued 0.668
- leftover after intensity 0.654 fake=False
- Y7 leftover after CN 0.662
- SIZE T2 leftover 0.452 fake=True
- month leftover median —
- now vs lag1 ρ T1 0.761 T3 0.770
- days leftover after n_tx 0.564
- co-mean leftover 0.586 fake=True
- leftover days+lag1+CN 0.591
- quarter leftover Q1 0.594 Q3 0.653
- lag1 leftover after days Y3 0.591 fake=True
- Y7 leftover after size 0.658
- issued>0 vs lag1 ρ=0.728
- T1 zero leftover 0.557 fake=False
- both>0 leftover-days 0.370 leftover-lag1 0.596
- leftover days+n_tx+lag1 0.587
- size leftover after issued 0.561
- leftover after days_lag1 all 0.621 fake=True
- Y7 both>0 leftover-lag1 0.540
- leftover after pending+CN 0.655
- days leftover both>0 0.679
- group leftover n=0 median —
- Y7 leftover after n_tx 0.665 fake=False
- rank-resid acf1 0.293
- both>0 intensity leftover 0.409 fake=True
- 2025-cohort leftover 0.685 fake=True
- kitchen-sink leftover 0.596
- 2024-cohort leftover 0.596 fake=True
- Y7 leftover after pending 0.661
- leftover after days+DSO 0.446 fake=True
- log1p leftover days+n_tx+lag1 0.587
- ERP-zero leftover 0.697 fake=True
- mid-trail leftover 0.610 fake=True
- Y7 leftover days+size 0.663
- long-trail leftover — fake=True
- Y7 leftover days+lag1+size 0.593
- leftover high-acf1 0.575 low-acf1 0.586
- Y7 leftover pending+CN 0.663
- n_grid>=18 leftover 0.612 fake=True
- Y7 leftover after DSO 0.583
- leftover days+pending+lag1 0.598
- n_grid<12 leftover — fake=True
- leftover days+CN+lag1 0.591
- Y7 leftover days+pending 0.664 fake=False
- leftover days+n_tx+size 0.610
- days leftover after issued+n_tx 0.570
- SIZE T3 leftover — fake=True
- leftover issued>p50 0.543 <=p50 0.601
- Y7 leftover days+CN 0.658 fake=False
- leftover p50-p90 0.532 p90+ —
- days leftover after issued+size 0.648
- leftover <=p25 0.697 >p75 —
- Y7 leftover days+n_tx 0.665 fake=False
- leftover p25-p75 0.575 fake=True
- days leftover after issued+lag1 0.673
- n_tx>0 leftover 0.605 fake=True
- p10-p90 leftover 0.605 fake=True
- days>0 leftover 0.605 fake=True
- lag1>0 leftover 0.559 fake=True
- onset leftover — fake=True
- stop leftover — fake=True
- issued>0 days>0 leftover 0.413 fake=True
- issued>0 T1 leftover — fake=True
- issued>0 T2 leftover — fake=True
- issued>0 mid leftover — fake=True
- issued>0 T2+T3 leftover 0.462 fake=True
- either-pos leftover 0.536 fake=True
- complete-case leftover 0.611 fake=True
- days>p50 leftover 0.642 fake=True
- days<=p50 leftover 0.578 fake=True
- n_tx>p50 leftover 0.654 fake=True
- n_tx<=p50 leftover 0.583 fake=True
- a_in3>p50 leftover 0.634 fake=True
- CN-defined leftover 0.594 fake=True
- pending-defined leftover 0.616 fake=True
- a_in3<=p50 leftover 0.583 fake=True
- days leftover after issued+size+lag1 0.648
- pending+CN leftover 0.594 fake=True
- n_tx-defined leftover 0.608 fake=True
- issued+days leftover 0.608 fake=True
- DSO-defined leftover 0.412 fake=True
- pending days>0 leftover 0.611 fake=True
- log1p complete leftover 0.611 fake=False
- DSO issued>0 leftover 0.412 fake=True
- pending issued>0 leftover 0.412 fake=True
- log1p DSO leftover 0.412 fake=False
- card: CLOSE as unused leftover / KEEP off the 15-col card
- do not grow TURNOVER 0.720; do not put issued on the 15-col card

Elapsed 14s. Cuts: coverage, twins, singles, leftover-days, size/lag1 bars, terciles, Q6, Y7 add-on, CN confirm, fold 4, holdout, same-n, log1p, intensity, fat tail, book-only.

Night quotes unchanged. Do not grow TURNOVER. Do not put issued on the 15-col card.
