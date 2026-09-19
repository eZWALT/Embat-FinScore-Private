# Unused leftover of `d_cust_hhi` on the 44 as Y3 X

Generated `2026-09-19T05:10:50+02:00` by agent `b4e81c2a`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage / mix only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. No `build_targets`. Do not invent `y_cust_hhi`. Do not reopen Y4 trees. Off the 15-col Y3 card. Night Y3 stays **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**. Y7 never D. Y5 never E. Y3 never B. Do not grow TURNOVER.

`d_cust_hhi` = Herfindahl of AR invoice counterparties in the trailing 6-month window (Family D). Needs identified customer CPs. Dark 470 stay **NaN not 0**. Javier 14: concentration is **top1**, not HHI. Y4 already KEEP this as a monopoly-tail single (lag3 0.605; body 0.445; ρ vs top1_lag3 0.991) and PARK trees. This ticket is leftover as Y3 X / engine X on the 44.

## Headline

**DROP** as engine X. Y3 **DROP** leftover-after-days 0.419. Y4 engine X **DROP** leftover after top1_lag3 0.549 body 0.445 (CONFIRM 0.445). Y4 tail footnote **KEEP** (0.221 vs 0.115; lag3 0.605). not SIZE (ρ=-0.179); twin of d_cust_top1 ρ=0.994 / lag3 0.994. Object: top1 rewrite; Y4 0.605 is the >0.975 monopoly tail, not a gradient. 44 should lose `d_cust_hhi` as engine X; Y4 tail footnote may stay. Q6 CLOSE. Night quotes unchanged: Y3 0.762 / 0.752; days 0.711; size 0.617; TURNOVER 0.720 / 0.712.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | **PARK** as a health Y. Do not invent `y_cust_hhi`. Dark 470 = NaN, not 0. |
| 2 | Who is improving? | Not a customer-HHI gradient. Body CV dies. |
| 3 | Who is turning? | **DROP** as Y3 X — leftover after days 0.419 vs days 0.711. |
| 4 | Dip vs fall? | Y4 tail footnote **KEEP** 0.605; engine X **DROP** leftover after top1_lag3 0.549; body 0.445. Trees PARK. |
| 5 | Why did it change? | Twin of top1 ρ=0.994. Object: top1 rewrite; Y4 0.605 is the >0.975 monopoly tail, not a gradient. Supp HHI is a different (protective) object. |
| 6 | Months earlier? | **CLOSE** — short lag3 present 21.7% n_pos=42 LOW_POWER — quote 21.7% / 42 pos. |

## PARK / CLOSE / KEEP / DROP-from-44

| object | decision | why |
| --- | --- | --- |
| d_cust_hhi as Y3 X / the 15-col card | DROP | twin of `d_cust_top1` ρ=0.994 / lag3 0.994 — HHI is the weaker rewrite. Y3 0.595 leftover-after-days 0.419. Stays off the 15-col card. |
| d_cust_hhi as engine X on the 44 | DROP from the 44 as engine X | twin or leftover dies as X. Y3 leftover 0.419; Y4 leftover 0.549 body 0.445. Javier concentration is top1. Y4 tail footnote may stay. |
| d_cust_hhi_lag3 as Y4 tail footnote | KEEP | Y4 lag3 CV 0.605 (quote 0.605). Tail >0.975 0.221 vs 0.115; body 0.445 CONFIRM True. Trees stay PARK. |
| Y4 trees / 2-col z-avg | PARK / CLOSE | Trees PARK (lose to the 0.605 single). z-avg was CLOSE. Do not reopen. Do not put on the card. |
| twin of d_cust_top1 / lag3 | DROP weaker (HHI) as X | ρ(HHI_lag3, top1_lag3)=0.994 (quote 0.991; CONFIRM). ρ(HHI, top1)=0.994. TWIN ≥0.80 — DROP weaker `d_cust_hhi` as engine X (Javier concentration is top1). Mean CV HHI 0.600 top1 0.597. Y4 leftover after top1_lag3 0.549. Y4 tail footnote may KEEP; engine X on the 44 may DROP. |
| same object as d_supp_hhi? | NO — different object | ρ(cust HHI, supp HHI)=0.158 — not a twin. Y4 customer tail is a crash 0.221 vs 0.115. Y5 supplier tail is protective 0.027 vs 0.086. CONFIRM different object — do not merge with supp HHI / Y4.. |
| y_cust_hhi / reopen Y4 trees | PARK | do not invent y_cust_hhi. Do not reopen Y4 trees. |
| Q6 lag1/lag3 on short books | CLOSE | short lag3 present 21.7% n_pos=42 LOW_POWER — quote 21.7% / 42 pos. |
| ICC / trait vs month shock | TRAIT | ICC 0.982 η² 0.821 k=674 (TRAIT ≥0.85). Y3 demean 0.404 mean 0.609. Y4 demean 0.561 mean 0.460. |
| KEEP-as-X gate (beat size+0.02, leftover, not SIZE, not twin) | FAIL | beat-size Y3=False Y4=True; leftover Y3 lives=False Y4 lives=False; SIZE=False twin=True. |


## 1. Coverage; 470 dark NaN vs invoice-book; ever-n

Train `d_cust_hhi` coverage 42.2% (8,928/21,157); ever-n 674 companies. Dark 470 (want 470): HHI non-null 0 zero-filled 0 top1 nn 0. 470 stay NaN not 0: CONFIRM. ERP n_cust==0 months 2,365; HHI defined on those 0 (want 0 — n=0 keeps HHI NaN). acf1=0.768 size ρ=-0.179.

| split | cm | companies | HHI nn | cov | NaN | ever-n | ever-ERP / never | dark nn / zero | ERP nn / NaN | HHI==0 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 21,157 | 1,214 | 8,928 | 42.2% | 12,229 | 674 | 744 / 470 | 0 / 0 | 8,928 / 4,626 | 0 |
| holdout | 1,073 | 72 | 502 | 46.8% | 571 | 40 | 40 / 32 | 0 / 0 | 502 / 80 | 0 |


| item | value |
| --- | ---: |
| train CM / companies | 21,157 / 1,214 |
| d_cust_hhi defined | 8,928 (42.2%) |
| ever-n companies | 674 |
| never-ERP companies | 470 (want 470; confirm_470=True) |
| dark HHI non-null / zero-filled | 0 / 0 |
| dark 0-fill | NO — CONFIRM |
| ERP n_cust==0 / HHI defined there | 2,365 / 0 |
| holdout coverage (check only) | 46.8% |
| acf1 / acf3 | 0.768 / 0.272 |
| size ρ vs log1p(a_in3) | -0.179 |

## 2. Spearman twins (|ρ|≥0.80)

ρ vs `d_cust_top1` 0.994 n=8,928; HHI_lag3↔top1_lag3 0.994 n=7,155 (quote 0.991; CONFIRM). TWIN — HHI is the weaker rewrite. vs `d_supp_hhi` 0.158 (not a twin of supp). vs `d_tx_cp_share` -0.113 not a twin of d_tx. vs size -0.179 (not SIZE). vs `d_n_cust` -0.837 (count rewrite |ρ|≥0.80 — not the official KEEP twin list). Javier: concentration is top1.

| vs | n | Spearman | Pearson | twin ≥0.80 |
| --- | --- | --- | --- | --- |
| d_cust_top1 | 8,928 | 0.994 | 0.983 | YES |
| d_cust_top1_lag3 | 6,866 | 0.861 | 0.845 | YES |
| d_cust_hhi_lag3 | 6,866 | 0.874 | 0.865 | no |
| d_supp_hhi | 8,788 | 0.158 | 0.158 | no |
| d_tx_cp_share | 8,916 | -0.113 | -0.106 | no |
| log1p(a_in3) | 8,538 | -0.179 | -0.203 | no |
| c_n_days_with_tx | 8,928 | -0.300 | -0.279 | no |
| d_n_cust | 8,928 | -0.837 | -0.397 | COUNT |
| d_n_supp | 8,928 | -0.319 | -0.204 | no |
| a_io_ratio | 8,538 | -0.064 | 0.015 | no |
| HHI_lag3 vs top1_lag3 | 7,155 | 0.994 | 0.983 | YES |


## 3. Y4 quintiles + >0.975 tail. Body CV ~0.445 CONFIRM

Y4 `d_cust_hhi_lag3` >0.975 P(Y=1)=0.221 (n=172, 38 pos) vs rest 0.115 (quote 22.1% / 11.5%, n=172/38) — CONFIRM crash tail. Body HHI≤0.975 CV 0.445 ± 0.132 (n=676, 78 pos; quote 0.445) — CONFIRM. The 0.605 is the monopoly tail. Body loses to dummy.. Quintiles monotone=False; Q5 rate 0.224. Y3 body 0.354. Y5 cust body 0.434. Y5 supp tail 0.027 vs 0.086 (protective object).

Train labeled cuts. Not monotone unless noted. Body CV is signed group-fold on HHI≤0.975.

| y | x | Q | interval | n | n_pos | P(Y=1) | median HHI |
| --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | d_cust_hhi_lag3 | 1 | (0.00531, 0.242] | 170 | 14 | 0.082 | 0.1118 |
| y4_ds_r_double | d_cust_hhi_lag3 | 2 | (0.242, 0.412] | 169 | 26 | 0.154 | 0.3308 |
| y4_ds_r_double | d_cust_hhi_lag3 | 3 | (0.412, 0.645] | 170 | 18 | 0.106 | 0.5132 |
| y4_ds_r_double | d_cust_hhi_lag3 | 4 | (0.645, 0.975] | 169 | 20 | 0.118 | 0.7818 |
| y4_ds_r_double | d_cust_hhi_lag3 | 5 | (0.975, 1.0] | 170 | 38 | 0.224 | 1.0000 |
| y4_ds_r_double | d_cust_hhi | 1 | (0.00531, 0.223] | 210 | 22 | 0.105 | 0.1125 |
| y4_ds_r_double | d_cust_hhi | 2 | (0.223, 0.391] | 209 | 25 | 0.120 | 0.3176 |
| y4_ds_r_double | d_cust_hhi | 3 | (0.391, 0.628] | 209 | 21 | 0.100 | 0.5010 |
| y4_ds_r_double | d_cust_hhi | 4 | (0.628, 0.969] | 209 | 28 | 0.134 | 0.7600 |
| y4_ds_r_double | d_cust_hhi | 5 | (0.969, 1.0] | 210 | 41 | 0.195 | 1.0000 |
| y3_recover_cash_6m | d_cust_hhi | 1 | (0.00225, 0.158] | 497 | 26 | 0.052 | 0.0854 |
| y3_recover_cash_6m | d_cust_hhi | 2 | (0.158, 0.355] | 498 | 16 | 0.032 | 0.2516 |
| y3_recover_cash_6m | d_cust_hhi | 3 | (0.355, 0.609] | 496 | 19 | 0.038 | 0.4630 |
| y3_recover_cash_6m | d_cust_hhi | 4 | (0.609, 0.976] | 497 | 33 | 0.066 | 0.7994 |
| y3_recover_cash_6m | d_cust_hhi | 5 | (0.976, 1.0] | 497 | 47 | 0.095 | 1.0000 |
| y5_ap_od30_ownp80 | d_cust_hhi | 1 | (0.0017599999999999998, 0.129] | 892 | 57 | 0.064 | 0.0632 |
| y5_ap_od30_ownp80 | d_cust_hhi | 2 | (0.129, 0.318] | 892 | 90 | 0.101 | 0.2117 |
| y5_ap_od30_ownp80 | d_cust_hhi | 3 | (0.318, 0.564] | 892 | 87 | 0.098 | 0.4201 |
| y5_ap_od30_ownp80 | d_cust_hhi | 4 | (0.564, 0.945] | 892 | 78 | 0.087 | 0.7632 |
| y5_ap_od30_ownp80 | d_cust_hhi | 5 | (0.945, 1.0] | 892 | 69 | 0.077 | 1.0000 |


| y | x | n tail / pos | P(Y=1) tail | P(Y=1) rest | tail AUROC | body CV | body n / pos | shape |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | d_cust_hhi_lag3 | 172 / 38 | 0.221 | 0.115 | 0.572 | 0.445 | 676 / 78 | crash |
| y4_ds_r_double | d_cust_hhi | 204 / 40 | 0.196 | 0.115 | 0.556 | 0.572 | 843 / 97 | crash |
| y3_recover_cash_6m | d_cust_hhi | 502 / 47 | 0.094 | 0.047 | 0.570 | 0.354 | 1983 / 94 | crash |
| y3_recover_cash_6m | d_cust_hhi_lag3 | 394 / 30 | 0.076 | 0.058 | 0.526 | 0.417 | 1468 / 85 | crash |
| y5_ap_od30_ownp80 | d_cust_hhi | 780 / 61 | 0.078 | 0.087 | 0.492 | 0.434 | 3680 / 320 | protective |
| y5_ap_od30_ownp80 | d_supp_hhi | 73 / 2 | 0.027 | 0.086 | 0.494 | 0.535 | 4829 / 416 | protective |


Y4 crash tail CONFIRM vs quote 22.1% / 11.5%: **True**. Body CV CONFIRM vs 0.445: **True** (0.445). Body dead: **True**.

## 4. Single-feature train group-fold AUROC

Sign from the train side of each fold. Seed 20260918. Night Y3 size **0.617** (replica 0.617; CONFIRM True); days **0.711** (replica 0.711; CONFIRM True). Y4 customer HHI lag3 night **0.605** (replica 0.605; CONFIRM True). Do not quote holdout.

Y3 HHI 0.595 vs size 0.617 (night 0.617; CONFIRM) days 0.711 (night 0.711; CONFIRM) top1 0.590 top1_lag3 0.541. Y4 HHI 0.583 HHI_lag3 0.605 (night 0.605; CONFIRM) top1_lag3 0.604 size 0.507 days 0.553. Beat-size Y3=False Y4=True. Beat-days Y3=False. Weaker of the Y4 twin pair: d_cust_hhi_lag3 (Javier concentration is top1).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_cust_hhi | 2,485 | 141 | 0.595 | 0.101 | 1 | 0.585 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | d_cust_hhi_lag1 | 2,309 | 135 | 0.578 | 0.090 | 1 | 0.570 | 0.693 0.445 0.554 0.599 0.597 |
| y3_recover_cash_6m | d_cust_hhi_lag3 | 1,862 | 115 | 0.547 | 0.084 | 1 | 0.533 | 0.638 0.457 0.478 0.533 0.629 |
| y3_recover_cash_6m | d_cust_top1 | 2,485 | 141 | 0.590 | 0.116 | 1 | 0.590 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | d_cust_top1_lag3 | 1,862 | 115 | 0.541 | 0.101 | 1 | 0.536 | 0.646 0.405 0.483 0.538 0.632 |
| y3_recover_cash_6m | d_supp_hhi | 2,877 | 203 | 0.653 | 0.091 | 1 | 0.623 | 0.789 0.689 0.573 0.643 0.570 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | 0.061 | -1 | 0.620 | 0.565 0.632 0.683 0.543 0.661 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | 0.031 | -1 | 0.723 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.077 | -1 | 0.650 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | d_tx_cp_share | 5,643 | 402 | 0.534 | 0.055 | -1 | 0.541 | 0.449 0.532 0.534 0.551 0.601 |
| y4_ds_r_double | d_cust_hhi | 1,047 | 137 | 0.583 | 0.074 | 1 | 0.574 | 0.614 0.648 0.523 0.644 0.487 |
| y4_ds_r_double | d_cust_hhi_lag1 | 986 | 130 | 0.603 | 0.081 | 1 | 0.586 | 0.664 0.686 0.525 0.631 0.508 |
| y4_ds_r_double | d_cust_hhi_lag3 | 848 | 116 | 0.605 | 0.044 | 1 | 0.592 | 0.677 0.610 0.575 0.563 0.598 |
| y4_ds_r_double | d_cust_top1 | 1,047 | 137 | 0.583 | 0.072 | 1 | 0.573 | 0.613 0.648 0.525 0.640 0.488 |
| y4_ds_r_double | d_cust_top1_lag3 | 848 | 116 | 0.604 | 0.043 | 1 | 0.590 | 0.666 0.613 0.588 0.547 0.604 |
| y4_ds_r_double | d_supp_hhi | 1,161 | 160 | 0.525 | 0.091 | 1 | 0.556 | 0.611 0.492 0.404 0.621 0.497 |
| y4_ds_r_double | log1p(a_in3) | 2,370 | 329 | 0.507 | 0.018 | -1 | 0.506 | 0.478 0.516 0.511 0.526 0.505 |
| y4_ds_r_double | c_n_days_with_tx | 2,370 | 329 | 0.553 | 0.046 | -1 | 0.562 | 0.565 0.576 0.471 0.573 0.579 |
| y4_ds_r_double | d_n_cust | 1,218 | 171 | 0.566 | 0.089 | -1 | 0.591 | 0.608 0.490 0.497 0.702 0.532 |
| y4_ds_r_double | d_tx_cp_share | 2,370 | 329 | 0.496 | 0.030 | -1 | 0.505 | 0.498 0.493 0.542 0.489 0.457 |
| y5_ap_od30_ownp80 | d_cust_hhi | 4,460 | 381 | 0.439 | 0.095 | 1 | 0.505 | 0.340 0.479 0.561 0.470 0.343 |
| y5_ap_od30_ownp80 | d_cust_hhi_lag1 | 4,443 | 374 | 0.431 | 0.084 | 1 | 0.502 | 0.333 0.460 0.534 0.470 0.358 |
| y5_ap_od30_ownp80 | d_cust_hhi_lag3 | 4,214 | 349 | 0.433 | 0.078 | 1 | 0.502 | 0.346 0.460 0.529 0.470 0.361 |
| y5_ap_od30_ownp80 | d_cust_top1 | 4,460 | 381 | 0.428 | 0.082 | 1 | 0.510 | 0.348 0.484 0.441 0.526 0.341 |
| y5_ap_od30_ownp80 | d_cust_top1_lag3 | 4,214 | 349 | 0.435 | 0.076 | 1 | 0.505 | 0.352 0.465 0.525 0.474 0.361 |
| y5_ap_od30_ownp80 | d_supp_hhi | 4,902 | 418 | 0.539 | 0.044 | -1 | 0.526 | 0.536 0.575 0.533 0.470 0.579 |
| y5_ap_od30_ownp80 | log1p(a_in3) | 4,905 | 418 | 0.556 | 0.084 | 1 | 0.545 | 0.674 0.539 0.466 0.495 0.606 |
| y5_ap_od30_ownp80 | c_n_days_with_tx | 4,905 | 418 | 0.540 | 0.083 | 1 | 0.540 | 0.641 0.449 0.473 0.530 0.607 |
| y5_ap_od30_ownp80 | d_n_cust | 4,905 | 418 | 0.466 | 0.092 | 1 | 0.526 | 0.325 0.542 0.531 0.510 0.423 |
| y5_ap_od30_ownp80 | d_tx_cp_share | 4,894 | 418 | 0.532 | 0.095 | -1 | 0.552 | 0.581 0.384 0.489 0.604 0.600 |


## 5. Honest leftover after days (Y3)

OLS residual of `d_cust_hhi` on the bar (train-defined slope). Leftover <0.55 dies. If leftover looks high, ρ(resid, days) is the fake-days leak screen.

Y3 leftover after days 0.419 (dies <0.55 / fake-days); ρ(resid, days)=-0.004 (not a fake-days leak). after top1 0.563; after size 0.566; after days+size 0.448. Y4 leftover after top1_lag3 0.549 (dies <0.55); after tail-flag 0.464. Rank-ortho days 0.450 top1 0.461.

| y | residual | n | n_pos | CV | R² | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | after days | 2,485 | 141 | 0.419 | 0.078 | 0.317 0.353 0.429 0.524 0.472 |
| y3_recover_cash_6m | after size | 2,459 | 138 | 0.566 | 0.041 | 0.696 0.488 0.546 0.576 0.526 |
| y3_recover_cash_6m | after days+size | 2,459 | 138 | 0.448 | 0.084 | 0.320 0.380 0.548 0.517 0.476 |
| y3_recover_cash_6m | after d_cust_top1 | 2,485 | 141 | 0.563 | 0.966 | 0.604 0.652 0.554 0.472 0.533 |
| y3_recover_cash_6m | after d_supp_hhi | 2,464 | 138 | 0.578 | 0.025 | 0.684 0.429 0.586 0.610 0.584 |
| y3_recover_cash_6m | rank-resid after days | 2,485 | 141 | 0.450 | — | 0.317 0.360 0.571 0.519 0.482 |
| y4_ds_r_double | after d_cust_top1_lag3 | 848 | 116 | 0.549 | 0.966 | 0.574 0.635 0.417 0.569 0.548 |
| y4_ds_r_double | after tail-flag | 848 | 116 | 0.464 | 0.475 | 0.383 0.375 0.616 0.533 0.412 |
| y4_ds_r_double | after size | 848 | 116 | 0.612 | 0.045 | 0.704 0.656 0.585 0.550 0.567 |
| y4_ds_r_double | after days | 848 | 116 | 0.558 | 0.065 | 0.680 0.427 0.591 0.545 0.545 |
| y4_ds_r_double | after d_supp_hhi | 828 | 111 | 0.606 | 0.022 | 0.668 0.592 0.602 0.582 0.587 |
| y4_ds_r_double | rank-resid after top1_lag3 | 848 | 116 | 0.461 | — | 0.529 0.546 0.371 0.384 0.476 |
| y4_ds_r_double | after d_cust_top1 | 1,047 | 137 | 0.530 | 0.966 | 0.592 0.514 0.440 0.543 0.560 |
| y4_ds_r_double | after days | 1,047 | 137 | 0.556 | 0.078 | 0.628 0.514 0.544 0.638 0.455 |


## 6. Leftover after top1 lag3 (Y4). Twin → DROP HHI as X

ρ(HHI_lag3, top1_lag3)=0.994 (quote 0.991; CONFIRM). ρ(HHI, top1)=0.994. TWIN ≥0.80 — DROP weaker `d_cust_hhi` as engine X (Javier concentration is top1). Mean CV HHI 0.600 top1 0.597. Y4 leftover after top1_lag3 0.549. Y4 tail footnote may KEEP; engine X on the 44 may DROP.

## 7. SIZE terciles — tail inside T1?

Y4 T1 tail P(Y=1)=0.147 vs rest 0.086 (n_tail=68 pos=10). Crash tail survives inside T1. Y3 T1 tail 0.162 vs rest 0.099 raw 0.566.

| y | x | tercile | n | n_pos | n tail / pos | P(Y=1) tail | P(Y=1) rest | raw CV | body CV |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_cust_hhi | T1 | 430 | 53 | 167 / 27 | 0.162 | 0.099 | 0.566 | LOW_POWER |
| y3_recover_cash_6m | d_cust_hhi | T2 | 939 | 40 | 191 / 10 | 0.052 | 0.040 | LOW_POWER | LOW_POWER |
| y3_recover_cash_6m | d_cust_hhi | T3 | 1090 | 45 | 134 / 8 | 0.060 | 0.039 | LOW_POWER | LOW_POWER |
| y4_ds_r_double | d_cust_hhi_lag3 | T1 | 138 | 16 | 68 / 10 | 0.147 | 0.086 | LOW_POWER | LOW_POWER |
| y4_ds_r_double | d_cust_hhi_lag3 | T2 | 313 | 52 | 60 / 18 | 0.300 | 0.134 | 0.637 | LOW_POWER |
| y4_ds_r_double | d_cust_hhi_lag3 | T3 | 397 | 48 | 44 / 10 | 0.227 | 0.108 | LOW_POWER | LOW_POWER |
| y5_ap_od30_ownp80 | d_cust_hhi | T1 | 1082 | 74 | 275 / 18 | 0.065 | 0.069 | 0.433 | 0.415 |
| y5_ap_od30_ownp80 | d_cust_hhi | T2 | 1744 | 145 | 327 / 26 | 0.080 | 0.084 | 0.437 | 0.553 |
| y5_ap_od30_ownp80 | d_cust_hhi | T3 | 1634 | 162 | 178 / 17 | 0.096 | 0.100 | 0.489 | 0.502 |


## 8. Q6 lag1 / lag3 on short books

Y4 short lag3 present 21.7% (290/1334; quote 21.7%) n_pos=42 (quote 42) — CONFIRM LOW_POWER. short lag1 n_pos=54 0.536. Y4 all lag3 0.605 n=848. Q6 stays CLOSE.

| y | slice | feature | n | n_pos | present | CV | sd | sign |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | short_<12 | d_cust_hhi | 1,435 | 71 | 38.5% | 0.600 | 0.103 | 1 |
| y3_recover_cash_6m | short_<12 | d_cust_hhi_lag1 | 1,263 | 66 | 33.9% | 0.612 | 0.133 | 1 |
| y3_recover_cash_6m | short_<12 | d_cust_hhi_lag3 | 817 | 50 | 21.9% | 0.615 | 0.165 | 1 |
| y3_recover_cash_6m | long_>=18 | d_cust_hhi | 111 | 6 | 52.1% | LOW_POWER | — | — |
| y3_recover_cash_6m | long_>=18 | d_cust_hhi_lag1 | 112 | 6 | 52.6% | LOW_POWER | — | — |
| y3_recover_cash_6m | long_>=18 | d_cust_hhi_lag3 | 109 | 6 | 51.2% | LOW_POWER | — | — |
| y3_recover_cash_6m | all | d_cust_hhi | 2,485 | 141 | 44.0% | 0.595 | 0.101 | 1 |
| y3_recover_cash_6m | all | d_cust_hhi_lag1 | 2,309 | 135 | 40.9% | 0.578 | 0.090 | 1 |
| y3_recover_cash_6m | all | d_cust_hhi_lag3 | 1,862 | 115 | 33.0% | 0.547 | 0.084 | 1 |
| y4_ds_r_double | short_<12 | d_cust_hhi | 481 | 58 | 36.1% | 0.475 | 0.046 | 1 |
| y4_ds_r_double | short_<12 | d_cust_hhi_lag1 | 423 | 54 | 31.7% | 0.536 | 0.052 | 1 |
| y4_ds_r_double | short_<12 | d_cust_hhi_lag3 | 290 | 42 | 21.7% | LOW_POWER | — | — |
| y4_ds_r_double | long_>=18 | d_cust_hhi | 157 | 18 | 52.9% | LOW_POWER | — | — |
| y4_ds_r_double | long_>=18 | d_cust_hhi_lag1 | 157 | 18 | 52.9% | LOW_POWER | — | — |
| y4_ds_r_double | long_>=18 | d_cust_hhi_lag3 | 158 | 18 | 53.2% | LOW_POWER | — | — |
| y4_ds_r_double | all | d_cust_hhi | 1,047 | 137 | 44.2% | 0.583 | 0.074 | 1 |
| y4_ds_r_double | all | d_cust_hhi_lag1 | 986 | 130 | 41.6% | 0.603 | 0.081 | 1 |
| y4_ds_r_double | all | d_cust_hhi_lag3 | 848 | 116 | 35.8% | 0.605 | 0.044 | 1 |


## 9. ICC / company-demean (trait vs month shock)

ICC 0.982 η² 0.821 k=674 (TRAIT ≥0.85). Y3 demean 0.404 mean 0.609. Y4 demean 0.561 mean 0.460.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | company-demean | 2,485 | 141 | 0.404 | 0.072 | 1 | 0.513 | 0.307 0.362 0.419 0.494 0.436 |
| y3_recover_cash_6m | company-mean | 3,397 | 205 | 0.609 | 0.049 | 1 | 0.611 | 0.592 0.620 0.532 0.643 0.657 |
| y4_ds_r_double | company-demean | 1,047 | 137 | 0.561 | 0.063 | 1 | 0.540 | 0.545 0.671 0.524 0.546 0.517 |
| y4_ds_r_double | company-mean | 1,402 | 196 | 0.460 | 0.071 | 1 | 0.521 | 0.458 0.505 0.542 0.355 0.439 |


## 10. Holdout mix flip (coverage / mix only)

Holdout 72 is coverage / mix only. Do not quote holdout AUROC. Y4 train pos crash 79.6% spike 35.6%. Holdout pos n=16 crash 37.5% spike1.5 87.5% spike1.2 93.8% ds_nn=16 med in-ratio 0.878 (quote 38% / 88%) — CONFIRM mix flip. y3_recover_cash_6m labeled 235 pos 14 HHI-nn 83 lag3 67. y4_ds_r_double labeled 135 pos 16 HHI-nn 32 lag3 21.

| y | n labeled | n_pos | HHI nn | HHI_lag3 nn | n tail / pos |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 235 | 14 | 83 | 67 | 4 / 0 |
| y4_ds_r_double | 135 | 16 | 32 | 21 | 3 / 0 |


| split | n labeled / pos | crash <0.8 | spike >1.5 | med in-ratio pos |
| --- | --- | --- | --- | --- |
| train | 2370 / 329 | 79.6% | 35.6% | 0.363 |
| holdout | 135 / 16 | 37.5% | 87.5% | 0.878 |


## 11. vs supp HHI — different object?

ρ(cust HHI, supp HHI)=0.158 — not a twin. Y4 customer tail is a crash 0.221 vs 0.115. Y5 supplier tail is protective 0.027 vs 0.086. CONFIRM different object — do not merge with supp HHI / Y4..

| object | P(Y=1) tail | P(Y=1) rest | shape |
| --- | --- | --- | --- |
| Y4 customer HHI_lag3 >0.975 | 0.221 | 0.115 | crash |
| Y5 supplier HHI >0.975 | 0.027 | 0.086 | protective |
| Y5 customer HHI >0.975 | 0.078 | 0.087 | protective |
| Y3 customer HHI >0.975 | 0.094 | 0.047 | crash |


## Extra A. 2-col z-avg of top1 + HHI

2-col z-avg of HHI + top1 is a twin stack. Y4 z-avg was CLOSE. Do not put this on any card. Y3 0.591 Y4 0.606.

| y | zavg CV | HHI-CC | top1-CC | gap vs HHI | just HHI | n / pos |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 0.591 | 0.595 | 0.590 | -0.004 | True | 2485 / 141 |
| y4_ds_r_double | 0.606 | 0.605 | 0.604 | 0.001 | True | 848 / 116 |


CLOSE. Do not put on the card.

## Extra B. Same-n leftover after top1

Same-n HHI vs top1: Y4 leftover after top1_lag3 0.549 R²=0.966. Y3 leftover after top1 0.563 R²=0.966. Near-identity — leftover after top1 is not a new object.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI same-n | 2,485 | 141 | 0.595 | 0.101 | 1 | 0.585 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | top1 same-n | 2,485 | 141 | 0.590 | 0.116 | 1 | 0.590 | 0.738 0.411 0.588 0.607 0.606 |
| y3_recover_cash_6m | HHI resid after top1 | 2,485 | 141 | 0.563 | 0.068 | 1 | 0.547 | 0.604 0.652 0.554 0.472 0.533 |
| y4_ds_r_double | HHI same-n | 848 | 116 | 0.605 | 0.044 | 1 | 0.592 | 0.677 0.610 0.575 0.563 0.598 |
| y4_ds_r_double | top1 same-n | 848 | 116 | 0.604 | 0.043 | 1 | 0.590 | 0.666 0.613 0.588 0.547 0.604 |
| y4_ds_r_double | HHI resid after top1 | 848 | 116 | 0.549 | 0.081 | 1 | 0.548 | 0.574 0.635 0.417 0.569 0.548 |


## Extra C. Tail companies on non-tail months

Companies that ever hit cust HHI_lag3>0.975: 288. Y4 tail months 0.221. If those same books are quieter on non-tail months, the object is the bin, not a gradient. Y4 non-tail-on-tail-cos 0.112.

| y | tail cos | tail months / pos | P(Y=1) tail mo | non-tail mo / pos | P(Y=1) non-tail mo |
| --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | 161 | 502 / 47 | 0.094 | 479 / 36 | 0.075 |
| y4_ds_r_double | 59 | 172 / 38 | 0.221 | 152 / 17 | 0.112 |


## Extra D. Y3 body leftover after days

Y3 body HHI 0.354 leftover-after-days 0.344 vs days 0.668 R²=0.078. Body leftover dies after days — not a monopoly-tail leftover as Y3 X.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI body ≤0.975 | 1,983 | 94 | 0.354 | 0.128 | 1 | 0.525 | 0.149 0.345 0.491 0.366 0.420 |
| y3_recover_cash_6m | days on body | 1,983 | 94 | 0.668 | 0.143 | -1 | 0.685 | 0.549 0.772 0.499 0.837 0.685 |
| y3_recover_cash_6m | HHI body resid after days | 1,983 | 94 | 0.344 | 0.143 | -1 | 0.526 | 0.165 0.234 0.503 0.455 0.362 |


## Extra E. Inverse leftover (days after HHI; top1 after HHI)

Inverse: days leftover after HHI 0.714 R²=0.078 (days 0.711 should survive). top1 leftover after HHI 0.536 R²=0.966 (near-identity twin — leftover of top1 after HHI is not a new object).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | days resid after HHI | 2,485 | 141 | 0.714 | 0.115 | -1 | 0.716 | 0.686 0.777 0.526 0.812 0.769 |
| y4_ds_r_double | top1_lag3 resid after HHI_lag3 | 848 | 116 | 0.536 | 0.086 | -1 | 0.536 | 0.546 0.627 0.393 0.555 0.558 |


## Extra F. Leftover after `d_n_cust`

ρ(HHI, n_cust)=-0.837. Y3 n_cust 0.653 leftover after n_cust 0.608 R²=0.157.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | d_n_cust | 3,003 | 221 | 0.653 | 0.077 | -1 | 0.650 | 0.742 0.662 0.530 0.665 0.666 |
| y3_recover_cash_6m | HHI resid after n_cust | 2,485 | 141 | 0.608 | 0.097 | 1 | 0.602 | 0.754 0.481 0.615 0.605 0.586 |
| y4_ds_r_double | d_n_cust | 1,218 | 171 | 0.566 | 0.089 | -1 | 0.591 | 0.608 0.490 0.497 0.702 0.532 |
| y4_ds_r_double | HHI resid after n_cust | 1,047 | 137 | 0.567 | 0.074 | 1 | 0.557 | 0.611 0.648 0.506 0.598 0.472 |


## Extra G. Company-mean leftover after days

Y3 company-mean HHI 0.609 leftover after days 0.537 R²=0.104. Style leftover dies — who-is-concentrated is days.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | company-mean HHI | 3,397 | 205 | 0.609 | 0.049 | 1 | 0.611 | 0.592 0.620 0.532 0.643 0.657 |
| y3_recover_cash_6m | mean resid after days | 3,397 | 205 | 0.537 | 0.015 | 1 | 0.540 | 0.527 0.539 0.520 0.541 0.559 |


## Extra H. Binary tail vs continuous (Y4)

Y4 continuous 0.605 vs binary tail flag 0.578 (y4_why binary OOF 0.578). Continuous still wins; the honest card shape is still the bin.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | HHI_lag3 continuous | 848 | 116 | 0.605 | 0.044 | 1 | 0.592 | 0.677 0.610 0.575 0.563 0.598 |
| y4_ds_r_double | HHI_lag3 >0.975 flag | 848 | 116 | 0.578 | 0.071 | 1 | 0.572 | 0.585 0.656 0.481 0.535 0.632 |


## Extra I. Dark 470 — HHI defined?

Dark 470 HHI defined 0 zero 0. CONFIRM NaN not 0 — HHI needs invoice CPs.. COMP_0962 HHI nn=0.

| slice | cm | HHI nn | HHI==0 | top1 nn | supp HHI nn |
| --- | --- | --- | --- | --- | --- |
| dark 470 CM | 7,603 | 0 | 0 | 0 | 0 |
| invoice-book CM | 13,554 | 8928 | 0 | 8928 | 10595 |
| COMP_0962 (refund ghost) | 13 | 0 | 0 | 0 | 0 |


## Extra J. Y3 T2/T3 pocket leftover after days

Y3 T2+T3 HHI 0.555 leftover-after-days 0.383 vs days 0.688. Pocket dies after days — not leftover turning.

| slice | n / pos | HHI CV | days CV | leftover days |
| --- | --- | --- | --- | --- |
| T1 | 430 / 53 | 0.566 | 0.615 | 0.513 |
| T2 | 939 / 40 | LOW_POWER | 0.634 | LOW_POWER |
| T3 | 1090 / 45 | LOW_POWER | 0.726 | LOW_POWER |
| T2+T3 | 2029 / 85 | 0.555 | 0.688 | 0.383 |


## Extra K. Short-book leftover after the honest bar

Y4 short lag3 leftover after top1_lag3 —. Y3 short leftover after days 0.452. Q6 CLOSE — short-book lag is not leftover after the honest bar.

| y | feat | bar | lag CV | leftover | n / pos | R² |
| --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | d_cust_hhi_lag3 | top1_lag3 | LOW_POWER | LOW_POWER | 290 / 42 | 0.966 |
| y3_recover_cash_6m | d_cust_hhi_lag3 | days | 0.615 | 0.350 | 817 / 50 | 0.065 |
| y3_recover_cash_6m | d_cust_hhi | days | 0.600 | 0.452 | 1435 / 71 | 0.078 |


## Extra L. Leftover after n_cust + days

Y3 leftover after n_cust+days 0.550 R²=0.195. Rank-ortho 0.551. Still ranks after n_cust+days — but twin of top1, still DROP as X.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI resid after n_cust+days | 2,485 | 141 | 0.550 | 0.104 | 1 | 0.548 | 0.693 0.420 0.600 0.540 0.497 |
| y3_recover_cash_6m | rank resid n_cust then days | 2,485 | 141 | 0.551 | 0.091 | 1 | 0.558 | 0.533 0.415 0.666 0.565 0.576 |


## Extra M. Holdout 16 positives — spike 1.5 vs 1.2

Holdout Y4 pos n=16 ds defined 16. crash 37.5% (quote 38%). spike>1.5 87.5% among-defined 87.5% spike>1.2 93.8% (quote 88%). CONFIRM 1.5. Coverage / mix only.

| company | period | in_ratio | ds_ratio | crash<0.8 | spike>1.5 | spike>1.2 | HHI_lag3 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| COMP_0269 | 2025-03 | 32.178 | 337.293 | False | True | True | — |
| COMP_0755 | 2025-05 | 1.609 | 3.620 | False | True | True | — |
| COMP_1211 | 2025-05 | 0.406 | 5.800 | True | True | True | — |
| COMP_0755 | 2025-06 | 1.320 | 6.174 | False | True | True | — |
| COMP_0700 | 2025-08 | 0.765 | 7.228 | True | True | True | — |
| COMP_0700 | 2025-09 | 0.878 | 2.336 | False | True | True | — |
| COMP_0755 | 2025-12 | 0.716 | 1.727 | True | True | True | — |
| COMP_0755 | 2026-01 | 0.976 | 2.950 | False | True | True | — |
| COMP_0952 | 2026-01 | 0.406 | 1.169 | True | False | False | — |
| COMP_0447 | 2026-02 | 1.360 | 3.588 | False | True | True | 0.437 |
| COMP_0774 | 2026-02 | 0.477 | 1.460 | True | False | True | — |
| COMP_0952 | 2026-02 | 0.559 | 1.510 | True | True | True | — |
| COMP_0276 | 2026-03 | — | 3.079 | — | True | True | — |
| COMP_0447 | 2026-03 | 1.749 | 4.291 | False | True | True | 0.503 |
| COMP_0276 | 2026-04 | — | 2.116 | — | True | True | — |
| COMP_0900 | 2026-04 | — | 2.128 | — | True | True | — |


## Extra N. n_cust leftover 0.608 — fake-days?

Y3 leftover after n_cust 0.608 R²=0.157 ρ(resid, days)=-0.217 (not a fake-days leak through n_cust). after n_cust+days 0.550 R²=0.195. after n_cust+days+top1 0.445 R²=0.968 (dies — count+activity+top1 eat the 0.608). ρ(HHI, n_cust)=-0.837 is a count rewrite, not the official KEEP twin list.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI resid after n_cust | 2,485 | 141 | 0.608 | 0.097 | 1 | 0.602 | 0.754 0.481 0.615 0.605 0.586 |
| y3_recover_cash_6m | HHI resid after n_cust+days | 2,485 | 141 | 0.550 | 0.104 | 1 | 0.548 | 0.693 0.420 0.600 0.540 0.497 |
| y3_recover_cash_6m | HHI resid after n_cust+days+top1 | 2,485 | 141 | 0.445 | 0.024 | -1 | 0.513 | 0.480 0.438 0.458 0.417 0.434 |


## Extra O. Y4 body leftover after top1

Y4 body leftover after top1_lag3 0.495 (body raw 0.445, top1 0.393) R²=0.966. Body leftover after top1 dies — no residual gradient under the tail.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | HHI_lag3 body | 676 | 78 | 0.445 | 0.132 | 1 | 0.528 | 0.331 0.385 0.630 0.534 0.343 |
| y4_ds_r_double | top1_lag3 on body | 676 | 78 | 0.393 | 0.069 | 1 | 0.524 | 0.351 0.390 0.351 0.513 0.361 |
| y4_ds_r_double | HHI body resid after top1 | 676 | 78 | 0.495 | 0.070 | -1 | 0.511 | 0.522 0.574 0.405 0.440 0.535 |


## Extra P. Days leftover after HHI (same-n)

On HHI-defined Y3 rows: HHI 0.595 days 0.730 days leftover after HHI 0.714 R²=0.078. Days survive after HHI — HHI does not eat the 0.711 bar.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI same-n | 2,485 | 141 | 0.595 | 0.101 | 1 | 0.585 | 0.740 0.454 0.581 0.608 0.590 |
| y3_recover_cash_6m | days same-n | 2,485 | 141 | 0.730 | 0.104 | -1 | 0.724 | 0.716 0.785 0.556 0.817 0.775 |
| y3_recover_cash_6m | days resid after HHI | 2,485 | 141 | 0.714 | 0.115 | -1 | 0.716 | 0.686 0.777 0.526 0.812 0.769 |


## Extra Q. Holdout HHI coverage on crash vs spike pos

Holdout Y4 pos HHI_lag3 defined 2/16. Crash pos n=6 lag3 nn=0. Spike pos n=14 lag3 nn=2. Tail almost empty on the hidden 72 — HHI cannot invert the mix flip. LOW_POWER.

| slice | n | HHI_lag3 nn | tail >0.975 |
| --- | --- | --- | --- |
| all hold Y4 pos | 16 | 2 | 0 |
| crash pos | 6 | 0 | 0 |
| spike>1.5 pos | 14 | 2 | 0 |


## Extra R. Y3 HHI_lag3 leftover after days

Y3 HHI_lag3 0.547 leftover after days 0.527 R²=0.065. Lag3 leftover dies — Q6 is not a leftover lead as Y3 X.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | HHI_lag3 | 1,862 | 115 | 0.547 | 0.084 | 1 | 0.533 | 0.638 0.457 0.478 0.533 0.629 |
| y3_recover_cash_6m | HHI_lag3 resid after days | 1,862 | 115 | 0.527 | 0.091 | -1 | 0.532 | 0.404 0.642 0.544 0.571 0.476 |


## Extra S. Y4 contemporaneous body leftover after days

Y4 contemporaneous body 0.572 leftover after days 0.569 vs days 0.526 R²=0.078; leftover after top1 0.473 R²=0.966. Lag3 body was 0.445. Now-body leftover lives — unexpected.

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | HHI now body ≤0.975 | 843 | 97 | 0.572 | 0.131 | 1 | 0.533 | 0.526 0.750 0.591 0.602 0.390 |
| y4_ds_r_double | days on now-body | 843 | 97 | 0.526 | 0.067 | -1 | 0.551 | 0.459 0.514 0.471 0.563 0.621 |
| y4_ds_r_double | now-body resid after days | 843 | 97 | 0.569 | 0.132 | 1 | 0.529 | 0.536 0.745 0.595 0.593 0.378 |
| y4_ds_r_double | now-body resid after top1 | 843 | 97 | 0.473 | 0.027 | -1 | 0.500 | 0.497 0.429 0.476 0.493 0.472 |


## Extra T. Y4 leftover after the >0.975 flag

Y4 leftover after the >0.975 flag 0.464 (dies — the 0.605 is the bin). Leftover after top1_lag3 0.549. Do not tell a smooth concentration-gradient story.

| residual | CV | note |
| --- | --- | --- |
| after tail-flag | 0.464 | 0.605 minus the >0.975 bin |
| after top1_lag3 | 0.549 | near-identity R²=0.966 |


## Extra U. Now-body leftover after days — fake-top1?

Y4 now-body leftover after days 0.569 ρ(resid, top1)=0.946 (FAKE-TOP1 leak — the 0.569 is the twin through days). after days+top1 0.480 R²=0.967 (dies — days leftover was the twin).

| y | feature | n | n_pos | CV | sd | sign | train | folds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y4_ds_r_double | now-body resid after days | 843 | 97 | 0.569 | 0.132 | 1 | 0.529 | 0.536 0.745 0.595 0.593 0.378 |
| y4_ds_r_double | now-body resid after days+top1 | 843 | 97 | 0.480 | 0.020 | -1 | 0.501 | 0.481 0.446 0.482 0.497 0.493 |


## What failed / next (held for wave note)

- twin of d_cust_top1 ρ=0.994 / lag3 0.994 — DROP weaker HHI as X
- Y3 leftover after days 0.419 dies <0.55
- Y4 leftover after top1_lag3 0.549; body 0.445
- Y4 body CV 0.445 CONFIRM vs 0.445
- Y4 tail CONFIRM crash 0.221 vs 0.115
- Q6 short lag3 LOW_POWER present=0.217 pos=42
- supp HHI is a different object (protective) — do not merge
- same-n leftover after top1 is a near-identity (R²≥0.90)
- Y3 body leftover after days 0.344 dies
- Y3 T2+T3 leftover after days 0.383 dies
- Q6 short leftover after the honest bar dies — CLOSE
- holdout mix flip CONFIRM crash=0.375 spike=0.875
- Y3 leftover after n_cust+days+top1 0.445 dies
- Y4 body leftover after top1 0.495 dies
- days leftover after HHI 0.714 survives — days is the bar
- Y3 HHI_lag3 leftover after days 0.527 dies
- Y4 leftover after tail-flag 0.464 dies — 0.605 is the bin
- Y4 now-body leftover after days 0.569 ρ(resid,top1)=0.946; after days+top1 0.480

Elapsed 5s. Cuts 1–12 plus extras (z-avg, same-n leftover, tail-company months, Y3 body leftover, inverse leftover, n_cust leftover, style, binary tail, dark 470, Y3 T2/T3 pocket, short-book Q6 leftover, n_cust+days, holdout 16, n_cust fake-days, Y4 body after top1, days same-n, holdout HHI mix, Y3 lag3 leftover, Y4 now-body leftover, tail-flag leftover, now-body fake-top1).

## What this module did not do

- Did not change night Y3 0.762 / 0.752, days 0.711, size 0.617, or Y7 TURNOVER 0.720 / 0.712.
- Did not put customer HHI on the 15-col Y3 card. Did not grow TURNOVER.
- Did not reopen Y4 trees. Did not invent `y_cust_hhi`. Did not merge with Y4 / Family I/M/J.
- Did not use Family E as Y5 X. Did not use Family B as Y3 X. Did not use Family F as Y4 X.
- Did not write 0–100 / pillars. Did not touch `product/`.
- Did not rewrite parquet or duckdb. Did not run `build_targets`.
- Did not fit on holdout 72. Did not commit. Did not write the parent journal / LIVE / canvas.
