# Family G leftover — `g_has_saving` / `g_has_investment` / `g_has_tpv` / `g_custom_share`

Generated `2026-09-19T04:43:50+02:00` by `python -m analysis.evaluate.g_has_rest_qa`.
Holdout 72 (seed 20260918) is **coverage only**. Rates, terciles, AUROC, leftover,
and PARK/CLOSE/KEEP/DROP are train. No parquet rewrite. No new GBM. No 0–100.
Does not run `build_targets`. Does not edit `products.py`. Off the 15-col Y3 card.
Family G already ran checking / card / `g_new` / `created_at` — confirm, do not reopen.

## Headline

- Train panel **21,157** CM / **1214** companies. saving last-month n=9 ever=9 modal0=99.7% (CONFIRM 99.7%). investment last=88 ever=88 (CONFIRM 95.1%). TPV last=10 (CONFIRM n=10).
- Rise-only HAS flags: **YES**. `g_n_accounts` CONFIRM 1,561/0. `g_custom_share` rise-only=False.
- Y3 card CV orient **0.551** (CONFIRM 0.551). days 0.711 (OK 0.711). size 0.617 (OK 0.617). saving 0.501 invest 0.509 TPV 0.502 custom 0.530.
- OLS leftover after days looks high (saving 0.713 invest 0.683 TPV 0.708 custom 0.649) but ρ(resid,days) ≥0.84 — **honest leftover DIES** (fake days leak). `g_custom_share` after accounts 0.548 — **dies after accounts (0.548<0.55; ρ vs accounts=0.064 not a twin). Not a mix leftover.**.
- checking hole 99.1% (CONFIRM 99.1%). Dark 470 vs ERP 744 (CONFIRM 744/470). Access ≠ ERP.
- 44 should lose remaining `g_has_*` + `g_custom_share`: **YES**. Night quotes unchanged Y3 **0.762 / 0.752**, days **0.711**, size **0.617**, Y7 TURNOVER **0.72 / 0.712**.

## Brief questions

1. **Who is healthy?** — leftover access flags are rare operating-type / connection tags, not a health reading. They lose to size.
2. **Who is improving?** — flags only rise (0→1). A later 1 is more connections, not 45→65.
3. **Who is turning?** — first-on is a connection birth. PARK as Q3. Do not invent `y_has_tpv`.
4. **Dip vs fall?** — not this table.
5. **Why did it change?** — card × TPV after size does not survive. custom_share is not a mix leftover after inventory.
6. **Months earlier?** — flags are 0 until first `created_at` of that type. CLOSE as Q6.

## 1. Prevalence train vs holdout

Train 21,157 CM / 1214 cos. saving last-month n=9 ever=9 modal0=99.7% (CONFIRM 99.7%). investment last=88 ever=88 modal0=95.1% (CONFIRM 95.1%). TPV last=10 ever=10 (CONFIRM n=10). custom>0 last=121 ever=121 cov=87.6%. Holdout coverage only (72 cos).

| col | train_cm | train_prev | modal0 | cov | ever_n | last_n | cm_on | hold_prev | hold_ever | hold_last_n | hold_cov |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 21,157 | 0.3% | 99.7% | 100.0% | 9 | 9 | 68 | 0.0% | 0 | 0 | 100.0% |
| g_has_investment | 21,157 | 4.9% | 95.1% | 100.0% | 88 | 88 | 1,040 | 3.8% | 5 | 5 | 100.0% |
| g_has_tpv | 21,157 | 0.4% | 99.6% | 100.0% | 10 | 10 | 95 | 0.0% | 0 | 0 | 100.0% |
| g_custom_share | 21,157 | 8.6% | 91.4% | 87.6% | 121 | 121 | 1,820 | 21.0% | 19 | 19 | 90.8% |
| g_has_card | 21,157 | 13.2% | 86.8% | 100.0% | 197 | 197 | 2,788 | 5.3% | 8 | 8 | 100.0% |
| g_has_checking | 21,157 | 87.5% | 12.5% | 100.0% | 1208 | 1208 | 18,516 | 90.8% | 72 | 72 | 100.0% |


## 2. Rise-only inventory

g_n_accounts 1,561 rises / 0 drops (CONFIRM rise-only 1,561/0). HAS flags rise-only: True (saving 9/0, invest 80/0, TPV 9/0). g_custom_share rise-only=False (51↑ 51↓). Flags only turn 0→1 — connection inventory, not mix change.

| col | rises | drops | flats | rise_only | acf1 | acf3 | acf6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 9 | 0 | 19,934 | YES | 0.639 | 0.642 | — |
| g_has_investment | 80 | 0 | 19,863 | YES | 0.800 | 0.685 | 0.442 |
| g_has_tpv | 9 | 0 | 19,934 | YES | 0.896 | 0.685 | 0.354 |
| g_has_card | 151 | 0 | 19,792 | YES | 0.773 | 0.685 | 0.389 |
| g_has_checking | 773 | 0 | 19,170 | YES | 0.685 | 0.663 | 0.389 |
| g_custom_share | 51 | 51 | 17,228 | NO | 0.844 | 0.461 | 0.307 |
| g_n_accounts | 1,561 | 0 | 18,382 | YES | 0.804 | 0.700 | 0.505 |


## 3. Spearman vs inventory / size / days

No leftover flag is SIZE on log1p(a_in3) / |opin| (saving -0.010, invest 0.188, TPV 0.039, custom 0.021). checking vs g_n_accounts ρ=0.576 (twin=no at |ρ|≥0.80). Twins: none.

| col | vs g_n_accounts | vs checking | vs card | vs log1p(a_in3) | vs log1p(|opin|) | vs days | vs n_tx | SIZE | twin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 0.019 | 0.021 | -0.022 | -0.010 | -0.011 | -0.039 | -0.040 | no | — |
| g_has_investment | 0.289 | 0.085 | 0.211 | 0.188 | 0.170 | 0.165 | 0.164 | no | — |
| g_has_tpv | 0.089 | 0.023 | -0.003 | 0.039 | 0.046 | 0.086 | 0.083 | no | — |
| g_custom_share | 0.064 | 0.012 | -0.048 | 0.021 | 0.017 | 0.042 | 0.079 | no | — |
| g_has_card | 0.401 | 0.142 | 1.000 | 0.151 | 0.155 | 0.267 | 0.225 | no | — |
| g_has_checking | 0.576 | 1.000 | 0.142 | 0.031 | 0.037 | 0.064 | 0.043 | no | — |


Twin ≥0.80 vs `g_n_accounts` / `g_has_checking` / days / n_tx → DROP the weaker. SIZE is |ρ|≥0.50 vs log1p(a_in3) or log1p(|a_op_in|).

## 4. Oriented group-fold AUROC (G replica)

Y3 card CV orient 0.551 (CONFIRM 0.551). days 0.711 (OK 0.711). size 0.617 (OK 0.617). saving 0.501 invest 0.509 TPV 0.502 custom 0.530. No leftover flag beats size ≥0.02.

Train labeled only. Raw = score as-is. Oriented = max(auc, 1−auc). Group-fold CV (5, seed 20260918). Holdout not used.

| Y | feature | pooled raw | pooled orient | CV raw | CV orient | sd | n labeled | n pos |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | g_has_saving | 0.499 | 0.501 | 0.499 | 0.501 | 0.002 | 5,648 | 402 |
| y3_recover_cash_6m | g_has_investment | 0.486 | 0.514 | 0.491 | 0.509 | 0.021 | 5,648 | 402 |
| y3_recover_cash_6m | g_has_tpv | 0.498 | 0.502 | 0.498 | 0.502 | 0.004 | 5,648 | 402 |
| y3_recover_cash_6m | g_custom_share | 0.525 | 0.525 | 0.530 | 0.530 | 0.031 | 5,182 | 364 |
| y3_recover_cash_6m | g_has_card | 0.451 | 0.549 | 0.449 | 0.551 | 0.034 | 5,648 | 402 |
| y3_recover_cash_6m | g_has_checking | 0.494 | 0.506 | 0.494 | 0.506 | 0.036 | 5,648 | 402 |
| y3_recover_cash_6m | c_n_days_with_tx | 0.277 | 0.723 | 0.289 | 0.711 | 0.031 | 5,648 | 402 |
| y3_recover_cash_6m | log1p_a_in3 | 0.380 | 0.620 | 0.383 | 0.617 | 0.061 | 5,528 | 391 |
| y3_recover_cash_6m | g_n_accounts | 0.415 | 0.585 | 0.419 | 0.581 | 0.092 | 5,648 | 402 |
| y2_neg_2of3 | g_has_saving | 0.499 | 0.501 | 0.499 | 0.501 | 0.002 | 17,356 | 1,271 |
| y2_neg_2of3 | g_has_investment | 0.500 | 0.500 | 0.496 | 0.504 | 0.011 | 17,356 | 1,271 |
| y2_neg_2of3 | g_has_tpv | 0.498 | 0.502 | 0.498 | 0.502 | 0.003 | 17,356 | 1,271 |
| y2_neg_2of3 | g_custom_share | 0.529 | 0.529 | 0.516 | 0.516 | 0.052 | 14,836 | 1,038 |
| y2_neg_2of3 | g_has_card | 0.497 | 0.503 | 0.508 | 0.508 | 0.050 | 17,356 | 1,271 |
| y2_neg_2of3 | g_has_checking | 0.478 | 0.522 | 0.454 | 0.546 | 0.063 | 17,356 | 1,271 |
| y2_neg_2of3 | c_n_days_with_tx | 0.577 | 0.577 | 0.571 | 0.571 | 0.046 | 17,356 | 1,271 |
| y2_neg_2of3 | log1p_a_in3 | 0.540 | 0.540 | 0.552 | 0.552 | 0.046 | 14,968 | 1,044 |
| y2_neg_2of3 | g_n_accounts | 0.501 | 0.501 | 0.471 | 0.529 | 0.098 | 17,356 | 1,271 |


KEEP-as-Y3-X: oriented CV beats oriented size by ≥0.02 **and** leftover after days **and** not SIZE **and** not a twin.

| feature | CV raw | CV orient | size orient | days orient | Δ size | Δ days | beat size+0.02 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 0.499 | 0.501 | 0.617 | 0.711 | -0.116 | -0.211 | no |
| g_has_investment | 0.491 | 0.509 | 0.617 | 0.711 | -0.108 | -0.203 | no |
| g_has_tpv | 0.498 | 0.502 | 0.617 | 0.711 | -0.115 | -0.210 | no |
| g_custom_share | 0.530 | 0.530 | 0.617 | 0.711 | -0.087 | -0.181 | no |
| g_has_card | 0.449 | 0.551 | 0.617 | 0.711 | -0.065 | -0.160 | no |
| g_has_checking | 0.494 | 0.506 | 0.617 | 0.711 | -0.111 | -0.206 | no |


Base rates by access (train company-months):

| flag | group | Y | n labeled | n pos | base rate | n CM | n companies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | has=1 | y2_neg_2of3 | 41 | 0 | 0.00% | 68 | 9 |
| g_has_saving | has=0 | y2_neg_2of3 | 17,315 | 1271 | 7.34% | 21,089 | 1214 |
| g_has_saving | has=1 | y3_recover_cash_6m | 9 | 0 | 0.00% | 68 | 9 |
| g_has_saving | has=0 | y3_recover_cash_6m | 5,639 | 402 | 7.13% | 21,089 | 1214 |
| g_has_investment | has=1 | y2_neg_2of3 | 788 | 57 | 7.23% | 1,040 | 88 |
| g_has_investment | has=0 | y2_neg_2of3 | 16,568 | 1214 | 7.33% | 20,117 | 1206 |
| g_has_investment | has=1 | y3_recover_cash_6m | 328 | 13 | 3.96% | 1,040 | 88 |
| g_has_investment | has=0 | y3_recover_cash_6m | 5,320 | 389 | 7.31% | 20,117 | 1206 |
| g_has_tpv | has=1 | y2_neg_2of3 | 67 | 1 | 1.49% | 95 | 10 |
| g_has_tpv | has=0 | y2_neg_2of3 | 17,289 | 1270 | 7.35% | 21,062 | 1213 |
| g_has_tpv | has=1 | y3_recover_cash_6m | 25 | 0 | 0.00% | 95 | 10 |
| g_has_tpv | has=0 | y3_recover_cash_6m | 5,623 | 402 | 7.15% | 21,062 | 1213 |
| g_has_card | has=1 | y2_neg_2of3 | 2,196 | 154 | 7.01% | 2,788 | 197 |
| g_has_card | has=0 | y2_neg_2of3 | 15,160 | 1117 | 7.37% | 18,369 | 1168 |
| g_has_card | has=1 | y3_recover_cash_6m | 981 | 33 | 3.36% | 2,788 | 197 |
| g_has_card | has=0 | y3_recover_cash_6m | 4,667 | 369 | 7.91% | 18,369 | 1168 |
| g_has_checking | has=1 | y2_neg_2of3 | 14,821 | 1034 | 6.98% | 18,516 | 1208 |
| g_has_checking | has=0 | y2_neg_2of3 | 2,535 | 237 | 9.35% | 2,641 | 779 |
| g_has_checking | has=1 | y3_recover_cash_6m | 5,178 | 364 | 7.03% | 18,516 | 1208 |
| g_has_checking | has=0 | y3_recover_cash_6m | 470 | 38 | 8.09% | 2,641 | 779 |


## 5. Honest leftover after days and after size

OLS leftover after days looks high (saving 0.713 invest 0.683 TPV 0.708 custom 0.649) but ρ(resid,days) is 0.989 / -0.840 / -0.978 / 0.840. Honest leftover DIES — fake days leak (same as zero_in 0.614/ρ=0.664).

OLS residual of the column on days / size / `g_n_accounts`, then oriented group-fold AUROC vs Y3. Leftover <0.55 dies. ρ(resid, days) ≥0.30 is a fake days leak.

| col | after days | honest leftover | after size | after accounts | after days+size | Y2 after days | ρ(resid,days) | fake days leak | dies |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 0.713 | DIES (fake days) | 0.618 | 0.582 | 0.719 | 0.569 | 0.989 | YES | YES |
| g_has_investment | 0.683 | DIES (fake days) | 0.601 | 0.560 | 0.687 | 0.569 | -0.840 | YES | YES |
| g_has_tpv | 0.708 | DIES (fake days) | 0.614 | 0.578 | 0.687 | 0.574 | -0.978 | YES | YES |
| g_custom_share | 0.649 | DIES (fake days) | 0.557 | 0.548 | 0.639 | 0.611 | 0.840 | YES | YES |
| g_has_card | 0.599 | DIES (fake days) | 0.527 | 0.512 | 0.601 | 0.552 | -0.595 | YES | YES |
| g_has_checking | 0.674 | DIES (fake days) | 0.596 | 0.569 | 0.526 | 0.608 | -0.741 | YES | YES |


## 6. SIZE terciles — survive inside T1?

T1 leftover: saving 0.609 invest 0.580 TPV 0.603 custom 0.631. No leftover flag survives inside T1 after days.

Last-month access share by size tercile:

| tercile | n companies | has_saving | has_investment | has_tpv | has_card | has_checking | custom>0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T1_small | 405 | 0.7% | 3.7% | 0.5% | 7.9% | 99.5% | 9.1% |
| T2_mid | 404 | 0.2% | 3.7% | 0.5% | 19.1% | 99.3% | 9.9% |
| T3_large | 405 | 1.2% | 14.3% | 1.5% | 21.7% | 99.8% | 10.9% |


| col | tercile | CV orient | size | days | leftover days | n labeled | n pos | survive T1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | T1_small | 0.503 | 0.509 | 0.603 | 0.609 | 1,572 | 269 | — |
| g_has_saving | T2_mid | 0.500 | 0.634 | 0.673 | 0.673 | 2,044 | 73 | — |
| g_has_saving | T3_large | 0.500 | 0.620 | 0.733 | 0.733 | 2,032 | 60 | — |
| g_has_saving | all | 0.501 | 0.617 | 0.711 | 0.713 | 5,648 | 402 | — |
| g_has_investment | T1_small | 0.514 | 0.509 | 0.603 | 0.580 | 1,572 | 269 | — |
| g_has_investment | T2_mid | 0.509 | 0.634 | 0.673 | 0.660 | 2,044 | 73 | — |
| g_has_investment | T3_large | 0.529 | 0.620 | 0.733 | 0.699 | 2,032 | 60 | — |
| g_has_investment | all | 0.509 | 0.617 | 0.711 | 0.683 | 5,648 | 402 | — |
| g_has_tpv | T1_small | 0.500 | 0.509 | 0.603 | 0.603 | 1,572 | 269 | — |
| g_has_tpv | T2_mid | 0.501 | 0.634 | 0.673 | 0.672 | 2,044 | 73 | — |
| g_has_tpv | T3_large | 0.504 | 0.620 | 0.733 | 0.724 | 2,032 | 60 | — |
| g_has_tpv | all | 0.502 | 0.617 | 0.711 | 0.708 | 5,648 | 402 | — |
| g_custom_share | T1_small | 0.532 | 0.509 | 0.603 | 0.631 | 1,446 | 247 | — |
| g_custom_share | T2_mid | 0.567 | 0.634 | 0.673 | 0.567 | 1,904 | 70 | — |
| g_custom_share | T3_large | 0.541 | 0.620 | 0.733 | 0.648 | 1,832 | 47 | — |
| g_custom_share | all | 0.530 | 0.617 | 0.711 | 0.649 | 5,182 | 364 | — |
| g_has_card | T1_small | 0.519 | 0.509 | 0.603 | 0.564 | 1,572 | 269 | — |
| g_has_card | T2_mid | 0.511 | 0.634 | 0.673 | 0.610 | 2,044 | 73 | — |
| g_has_card | T3_large | 0.601 | 0.620 | 0.733 | 0.555 | 2,032 | 60 | — |
| g_has_card | all | 0.551 | 0.617 | 0.711 | 0.599 | 5,648 | 402 | — |


## 7. Dark 470 vs 744

Last-month train: ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Dark last-month saving 0.4% vs ERP 0.9%; invest 8.9% vs 6.2%; TPV 0.2% vs 1.2%; checking 99.4% vs 99.6%. Access ≠ ERP already for checking — leftover flags are rare on both sides.

Last-month as-of inventory:

| group | n companies | accounts p50 | has_saving | has_investment | has_tpv | has_card | has_checking | custom>0 | custom p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ever_erp | 744 | 3.000 | 0.9% | 6.2% | 1.2% | 17.3% | 99.6% | 11.8% | 0.000 |
| never_erp | 470 | 3.000 | 0.4% | 8.9% | 0.2% | 14.5% | 99.4% | 7.0% | 0.000 |


Company-ever:

| group | n | ever saving | ever investment | ever tpv | ever card | ever checking |
| --- | --- | --- | --- | --- | --- | --- |
| ever_erp | 744 | 0.9% | 6.2% | 1.2% | 17.3% | 99.6% |
| never_erp | 470 | 0.4% | 8.9% | 0.2% | 14.5% | 99.4% |


## 8. `g_custom_share` — mix leftover or inventory twin?

g_custom_share acf1=0.844 (CONFIRM 0.84). Y3 raw 0.530 leftover after accounts 0.548 after days 0.649. ρ vs g_n_accounts=0.064 (twin=no). Verdict: **dies after accounts (0.548<0.55; ρ vs accounts=0.064 not a twin). Not a mix leftover.**.

| slice | raw orient | after accounts | after days | after both | ρ vs accounts | ρ vs days | ρ(resid_acc, acc) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all train | 0.530 | 0.548 | 0.649 | 0.576 | 0.064 | 0.042 | 0.844 |
| connected (n>0) | 0.530 | — | 0.649 | — | 0.064 | 0.042 | — |


## 9. Q6 — lag1 / lag3 empty-on-short

Q6: all leftover flags **CLOSE** — connection clock / chance / empty-on-short.

Flags that are 0 until first `created_at` are CLOSE as Q6 (connection clock, not lead time).

| col | now | lag1 | lag3 | short now | short lag1 | short on | Q6 | why |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 0.501 | 0.501 | 0.501 | 0.500 | 0.500 | 29 | CLOSE | contemporaneous Y3 0.501 is chance; nothing to lead. |
| g_has_investment | 0.509 | 0.510 | 0.510 | 0.503 | 0.503 | 88 | CLOSE | contemporaneous Y3 0.509 is chance; nothing to lead. |
| g_has_tpv | 0.502 | 0.502 | 0.502 | 0.500 | 0.500 | 11 | CLOSE | contemporaneous Y3 0.502 is chance; nothing to lead. |
| g_custom_share | 0.530 | 0.533 | 0.537 | 0.503 | 0.542 | 175 | CLOSE | contemporaneous Y3 0.530 is chance; nothing to lead. |
| g_has_card | 0.551 | 0.551 | 0.556 | 0.532 | 0.519 | 234 | CLOSE | rise-only connection clock — 0 until first created_at; not lead time. |
| g_has_checking | 0.506 | 0.513 | 0.517 | 0.607 | 0.642 | 2190 | CLOSE | contemporaneous Y3 0.506 is chance; nothing to lead. |


## 10. Card × TPV after size

Last-month card 197 TPV 10 (CONFIRM TPV n=10). 2×2 cannot be a type after size when TPV is 10 companies. TPV leftover after size does **not** survive. CONFIRM G.

| cell | n companies | T1_small | T2_mid | T3_large |
| --- | --- | --- | --- | --- |
| card+tpv | 3 | 2 | 1 | 0 |
| card_only | 194 | 30 | 76 | 88 |
| tpv_only | 7 | 0 | 1 | 6 |
| neither | 1010 | 373 | 326 | 311 |


| tercile | has_card | has_tpv | has_saving | has_invest |
| --- | --- | --- | --- | --- |
| T1_small | 7.9% | 0.5% | 0.7% | 3.7% |
| T2_mid | 19.1% | 0.5% | 0.2% | 3.7% |
| T3_large | 21.7% | 1.5% | 1.2% | 14.3% |


TPV leftover after size:

| tercile | TPV CV | after size | n labeled | n pos Y3 |
| --- | --- | --- | --- | --- |
| T1_small | 0.500 | 0.509 | 1,572 | 269 |
| T2_mid | 0.501 | 0.635 | 2,044 | 73 |
| T3_large | 0.504 | 0.626 | 2,032 | 60 |
| all | 0.502 | 0.614 | 5,648 | 402 |


## 11. ICC / company-demean

custom ICC=0.998 acf1=0.844 TRAIT. saving ICC=0.981 TRAIT; invest 0.980 TRAIT; TPV 0.965 TRAIT. Demean drops skill — these are company traits (connection book), not month shocks.

| col | ICC | acf1 | acf3 | Y3 raw | Y3 demean | drop | kind |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | 0.981 | 0.639 | 0.642 | 0.501 | 0.501 | 0.000 | TRAIT |
| g_has_investment | 0.980 | 0.800 | 0.685 | 0.509 | 0.512 | -0.003 | TRAIT |
| g_has_tpv | 0.965 | 0.896 | 0.685 | 0.502 | 0.501 | 0.001 | TRAIT |
| g_custom_share | 0.998 | 0.844 | 0.461 | 0.530 | 0.507 | 0.023 | TRAIT |
| g_has_card | 0.987 | 0.773 | 0.685 | 0.551 | 0.500 | 0.051 | TRAIT |
| g_has_checking | 0.887 | 0.685 | 0.663 | 0.506 | 0.528 | -0.022 | TRAIT |


## 12. Checking hole + PARK / CLOSE / KEEP / DROP-from-44

checking=0 CM 2,641; of those 2,618 (99.1%) have g_n_accounts=0 (CONFIRM 99.1%). Checking-off but accounts>0: 23 CM / 8 companies. 44 should lose remaining g_has_* + g_custom_share: **YES**.

Checking-off but accounts>0: 23 CM / 8 companies (card 47.8%, TPV 4.3%, saving 0.0%, invest 8.7%).

| col | from 44 | as Y3 X | as health Y | Q6 | rise-only | leftover days | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | DROP from the 44 | CLOSE | PARK | CLOSE | YES | DIES (fake) | RARE last-month n=9 ever=9; rise-only=True; Y3 0.501; OLS leftover 0.713 is fake days leak — honest leftover DIES. |
| g_has_investment | DROP from the 44 | CLOSE | PARK | CLOSE | YES | DIES (fake) | Y3 0.509 Δsize -0.108; OLS leftover 0.683 fake=True; rise-only=True. |
| g_has_tpv | DROP from the 44 | CLOSE | PARK | CLOSE | YES | DIES (fake) | RARE last-month n=10 ever=10; rise-only=True; Y3 0.502; OLS leftover 0.708 is fake days leak — honest leftover DIES. |
| g_custom_share | DROP from the 44 | CLOSE | PARK | CLOSE | NO | DIES (fake) | Y3 0.530 leftover days 0.649 leftover accounts 0.548; dies after accounts (0.548<0.55; ρ vs accounts=0.064 not a twin). Not a mix leftover.; acf1 0.844 TRAIT. |
| g_has_card | DROP from the 44 | CLOSE | PARK | CLOSE | YES | DIES (fake) | Y3 orient 0.551 loses to size 0.617 / days 0.711; leftover 0.599. |
| g_has_checking | DROP from the 44 | CLOSE | PARK | CLOSE | YES | DIES (fake) | 99.1% of checking=0 is g_n_accounts=0 (CONFIRM 99.1%); connection hole, not mix. |


## 13. Dictionary leftover types (no `g_has_*`)

wallet/risk/lineofcomex/expensesPlatform sit in `other` and count in g_n_accounts but have no g_has_*. Train companies with a leftover type: 37. Do not invent parquet columns.

| type | n companies | n rows | Y2 rate | Y3 rate | n Y3 labeled | accounts p50 |
| --- | --- | --- | --- | --- | --- | --- |
| wallet | 20 | 33 | 10.8% | 3.6% | 111 | 5.000 |
| risk | 5 | 25 | 0.0% | 0.0% | 17 | 11.000 |
| lineofcomex | 7 | 17 | 0.0% | 0.0% | 35 | 7.000 |
| expensesPlatform | 7 | 23 | 0.0% | 23.1% | 26 | 7.000 |


## 14. vs Family F `f_has_*` (already dropped from 44)

Family F f_has_* already dropped from the 44. Same rise-only story: True. f_has_factoring 17↑/0↓ leftover 0.703; f_has_confirming 50↑/0↓ leftover 0.697; f_has_loc 135↑/0↓ leftover 0.601.

| col | rises | drops | rise_only | ever_n | last_n | Y3 orient | leftover days |
| --- | --- | --- | --- | --- | --- | --- | --- |
| f_has_factoring | 17 | 0 | YES | 18 | 18 | 0.505 | 0.703 |
| f_has_confirming | 50 | 0 | YES | 61 | 61 | 0.503 | 0.697 |
| f_has_loc | 135 | 0 | YES | 186 | 186 | 0.546 | 0.601 |


## 15. Holdout coverage only

Holdout 72 companies / 1,073 CM — coverage only. No holdout AUROC / tertiles / leftover.

| col | hold CM | hold cos | cov | ever_n | last_n |
| --- | --- | --- | --- | --- | --- |
| g_has_saving | 1,073 | 72 | 100.0% | 0 | 0 |
| g_has_investment | 1,073 | 72 | 100.0% | 5 | 5 |
| g_has_tpv | 1,073 | 72 | 100.0% | 0 | 0 |
| g_custom_share | 1,073 | 72 | 90.8% | 19 | 19 |
| g_has_card | 1,073 | 72 | 100.0% | 8 | 8 |
| g_has_checking | 1,073 | 72 | 100.0% | 72 | 72 |


## 16. First-created clock

Flags are 0 until first type `created_at` (YES — CLOSE as Q6). g_has_saving pre0=100.0%; g_has_investment pre0=100.0%; g_has_tpv pre0=100.0%; g_has_card pre0=100.0%; g_has_checking pre0=100.0%.

| col | ever flag | first flag = first created_at month | share same month | flag=0 before first created |
| --- | --- | --- | --- | --- |
| g_has_saving | 9 | 9/9 | 100.0% | 100.0% |
| g_has_investment | 88 | 80/88 | 90.9% | 100.0% |
| g_has_tpv | 10 | 10/10 | 100.0% | 100.0% |
| g_has_card | 197 | 166/197 | 84.3% | 100.0% |
| g_has_checking | 1208 | 874/1208 | 72.4% | 100.0% |


## 17. Connected-book leftover (`g_n_accounts>0`)

Connected months only (g_n_accounts>0, n=18,539). Card Y3 0.556 leftover 0.592 — still CLOSE. Invest leftover 0.684; custom 0.649.

| col | connected Y3 | leftover days | n labeled | dies |
| --- | --- | --- | --- | --- |
| g_has_saving | 0.501 | 0.717 | 5,182 | no |
| g_has_investment | 0.510 | 0.684 | 5,182 | no |
| g_has_tpv | 0.502 | 0.712 | 5,182 | no |
| g_custom_share | 0.530 | 0.649 | 5,182 | no |
| g_has_card | 0.556 | 0.592 | 5,182 | no |
| g_has_checking | 0.501 | 0.716 | 5,182 | no |


## 18. Y rates by leftover flag

| flag | group | Y | n labeled | n pos | base | n cos |
| --- | --- | --- | --- | --- | --- | --- |
| g_has_saving | on | y2_neg_2of3 | 41 | 0 | 0.00% | 9 |
| g_has_saving | off | y2_neg_2of3 | 17,315 | 1271 | 7.34% | 1214 |
| g_has_saving | on | y3_recover_cash_6m | 9 | 0 | 0.00% | 9 |
| g_has_saving | off | y3_recover_cash_6m | 5,639 | 402 | 7.13% | 1214 |
| g_has_investment | on | y2_neg_2of3 | 788 | 57 | 7.23% | 88 |
| g_has_investment | off | y2_neg_2of3 | 16,568 | 1214 | 7.33% | 1206 |
| g_has_investment | on | y3_recover_cash_6m | 328 | 13 | 3.96% | 88 |
| g_has_investment | off | y3_recover_cash_6m | 5,320 | 389 | 7.31% | 1206 |
| g_has_tpv | on | y2_neg_2of3 | 67 | 1 | 1.49% | 10 |
| g_has_tpv | off | y2_neg_2of3 | 17,289 | 1270 | 7.35% | 1213 |
| g_has_tpv | on | y3_recover_cash_6m | 25 | 0 | 0.00% | 10 |
| g_has_tpv | off | y3_recover_cash_6m | 5,623 | 402 | 7.15% | 1213 |
| g_custom_share | on | y2_neg_2of3 | 1,466 | 154 | 10.50% | 121 |
| g_custom_share | off | y2_neg_2of3 | 15,890 | 1117 | 7.03% | 1159 |
| g_custom_share | on | y3_recover_cash_6m | 371 | 42 | 11.32% | 121 |
| g_custom_share | off | y3_recover_cash_6m | 5,277 | 360 | 6.82% | 1159 |


## 19. `g_custom_share` quintiles (connected)

custom_share quintiles on connected months — mix leftover would show a monotone Y3; inventory twin would track accounts.

| q | n CM | custom p50 | accounts p50 | Y3 rate | Y2 rate | n Y3 |
| --- | --- | --- | --- | --- | --- | --- |
| Q1 | 3,708 | 0.000 | 2.000 | 8.68% | 5.33% | 1071 |
| Q2 | 3,708 | 0.000 | 2.000 | 7.87% | 6.20% | 1080 |
| Q3 | 3,707 | 0.000 | 3.000 | 6.89% | 7.89% | 1074 |
| Q4 | 3,708 | 0.000 | 2.000 | 5.18% | 7.29% | 1061 |
| Q5 | 3,708 | 0.000 | 3.000 | 6.36% | 8.25% | 896 |


## 20. First-on calendar

First-on months (0→1): saving 9, invest 88, TPV 10, card 197, checking 1208. Connection wave, not Q3 turning.

## 21. Leftover inside days terciles (honest)

Inside days terciles (no OLS): leftover flags stay chance. custom D1/D2/D3 0.524 / 0.566 / 0.526. A cell ≥0.55: g_custom_share — still loses to days 0.711 / size 0.617.

| col | days tercile | Y3 orient | n labeled | n pos | dies <0.55 |
| --- | --- | --- | --- | --- | --- |
| g_has_saving | D1 | 0.504 | 1,290 | 212 | YES |
| g_has_saving | D2 | 0.500 | 1,965 | 121 | YES |
| g_has_saving | D3 | 0.500 | 2,393 | 69 | YES |
| g_has_investment | D1 | 0.507 | 1,290 | 212 | YES |
| g_has_investment | D2 | 0.526 | 1,965 | 121 | YES |
| g_has_investment | D3 | 0.517 | 2,393 | 69 | YES |
| g_has_tpv | D1 | 0.500 | 1,290 | 212 | YES |
| g_has_tpv | D2 | 0.500 | 1,965 | 121 | YES |
| g_has_tpv | D3 | 0.504 | 2,393 | 69 | YES |
| g_custom_share | D1 | 0.524 | 1,164 | 192 | YES |
| g_custom_share | D2 | 0.566 | 1,812 | 112 | no |
| g_custom_share | D3 | 0.526 | 2,206 | 60 | YES |
| g_has_card | D1 | 0.500 | 1,290 | 212 | YES |
| g_has_card | D2 | 0.519 | 1,965 | 121 | YES |
| g_has_card | D3 | 0.545 | 2,393 | 69 | YES |
| g_has_checking | D1 | 0.502 | 1,290 | 212 | YES |
| g_has_checking | D2 | 0.505 | 1,965 | 121 | YES |
| g_has_checking | D3 | 0.520 | 2,393 | 69 | YES |


OLS leftover of a rare / high-ICC flag on days is −β·days. Inside days terciles the flag is chance.

## 22. `g_custom_share` drops are inventory dilution

g_custom_share 51↑ / 51↓. Of drops, 51/51 (100.0%) are n_accounts↑ with n_custom flat — mechanical dilution. n_custom itself 65↑ / 0↓ (rise-only custom count; share drops are inventory, not mix off-boarding). accounts↑ among share-drops: 51.

| event | n | accounts ↑ & n_custom flat (dilution) | n_custom ↓ | share of drops that are dilution |
| --- | --- | --- | --- | --- |
| share ↓ | 51 | 51 | 0 | 100.0% |
| share ↑ | 51 | — | — | n_custom ↑ on 51 of 51 rises |


## 23. custom>0 / all-custom piles (replace zero-inflated quintiles)

custom>0 flag Y3 0.525 leftover 0.705 ρ(resid,days)=-0.820 (fake=True). all-custom Y3 0.518. Last-month all-custom companies=42. Zero-inflated quintiles were uninformative (Q1–Q4 p50=0); piles replace them.

| pile | n CM | n cos | accounts p50 | Y3 rate | n Y3 |
| --- | --- | --- | --- | --- | --- |
| no_custom | 16,719 | 1129 | 2.000 | 6.69% | 4811 |
| some_custom | 1,115 | 79 | 5.000 | 8.27% | 254 |
| all_custom | 705 | 52 | 2.000 | 17.95% | 117 |


## 24. checking=0 vs accounts=0 Jaccard

Jaccard(checking=0, accounts=0)=0.991 (2,618 / 2,641). Hole is the connection clock, not a |ρ|≥0.80 twin (ρ vs g_n_accounts=0.576). DROP checking from the 44 anyway.

| item | n CM |
| --- | --- |
| checking=0 | 2641 |
| accounts=0 | 2618 |
| both | 2618 |
| Jaccard | 0.991 |
| checking=0 but accounts>0 | 23 |
| accounts=0 but checking=1 | 0 |


## 25. Family F leftover is the same fake days leak

Family F same fake-days leftover: True. Rise-only inventory flags leak days through OLS residual. Already dropped from the 44.

| col | Y3 raw | OLS leftover | ρ(resid,days) | honest |
| --- | --- | --- | --- | --- |
| f_has_factoring | 0.505 | 0.703 | -0.976 | DIES (fake days) |
| f_has_confirming | 0.503 | 0.697 | -0.892 | DIES (fake days) |
| f_has_loc | 0.546 | 0.601 | -0.625 | DIES (fake days) |


## 26. Holdout custom coverage contrast

Holdout custom>0 21.0% vs train 8.6% (19 / 72 last-month). Coverage only — do not fit on the 72. Higher holdout custom is not a KEEP argument.

| split | custom>0 CM | ever cos | last_n |
| --- | --- | --- | --- |
| train | 8.6% | 121 | 121 |
| holdout (coverage) | 21.0% | 19 | 19 |


## 27. First-on vs `g_new_this_month`

First-on months that coincide with g_new>0 are the type's created_at landing in that month (already PARK as health Y). Not a new Q3.

| flag | first-on | same month g_new>0 | align |
| --- | --- | --- | --- |
| g_has_saving | 9 | 9 | 100.0% |
| g_has_investment | 88 | 81 | 92.0% |
| g_has_tpv | 10 | 10 | 100.0% |
| g_has_card | 197 | 168 | 85.3% |
| g_has_checking | 1208 | 890 | 73.7% |


## 28. D2 custom 0.566 and all-custom 17.9% rate

D2 custom Y3 0.566 vs size 0.555 (Δ 0.011) vs days 0.575; after size 0.559. CLOSE D2 — does not beat size ≥0.02. all-custom last-month n=42 Y3 0.518 (rate 17.9% on 117 labeled / 52 cos is a small pile, not leftover skill).

| slice | Y3 | size | days | after size | Δ size | KEEP |
| --- | --- | --- | --- | --- | --- | --- |
| custom_share in D2 | 0.566 | 0.555 | 0.575 | 0.559 | 0.011 | no |
| all-custom flag | 0.518 | 0.617 | — | 0.588 | -0.098 | no |


## 29. Investment is T3-tagged

Investment last-month share T3 14.3% vs T1 3.7%. T3 Y3 0.529 vs size 0.620 — size-tagged type, not leftover health.

| tercile | last-month invest | Y3 | size | n labeled | n pos |
| --- | --- | --- | --- | --- | --- |
| T1_small | 3.7% | 0.514 | 0.509 | 1,572 | 269 |
| T2_mid | 3.7% | 0.509 | 0.634 | 2,044 | 73 |
| T3_large | 14.3% | 0.529 | 0.620 | 2,032 | 60 |


## 30. First-on without `g_new` = already-on-book

First-on without g_new is the product already created before the first grid month (as-of stock on month 1), not a silent mix change. Still a connection clock.

| flag | first-on without g_new | of those first grid month | share first-month |
| --- | --- | --- | --- |
| g_has_saving | 0 | 0 | — |
| g_has_investment | 7 | 7 | 100.0% |
| g_has_tpv | 0 | 0 | — |
| g_has_card | 29 | 29 | 100.0% |
| g_has_checking | 318 | 318 | 100.0% |


## 31. Leftover dictionary types — do not invent `g_has_*`

wallet/risk/lineofcomex/expensesPlatform count in g_n_accounts, have no g_has_*. expensesPlatform Y3 23.1% sits on 26 labeled months / 7 companies — G already said do not promote. Do not invent parquet columns.

| type | n companies | Y3 rate | n Y3 | promote |
| --- | --- | --- | --- | --- |
| wallet | 20 | 3.6% | 111 | no — do not invent g_has_* |
| risk | 5 | 0.0% | 17 | no — do not invent g_has_* |
| lineofcomex | 7 | 0.0% | 35 | no — do not invent g_has_* |
| expensesPlatform | 7 | 23.1% | 26 | no — do not invent g_has_* |


## 32. Dark investment is higher, not leftover health

Dark last-month investment 8.9% > ERP 6.2%. Access ≠ ERP (already true for checking). Not a KEEP as Y3 X.

| group | n cos | last invest | Y3 | size | n labeled |
| --- | --- | --- | --- | --- | --- |
| ever_erp | 744 | 6.2% | 0.513 | 0.623 | 3,618 |
| never_erp | 470 | 8.9% | 0.503 | 0.644 | 2,030 |


## Plot

- `/home/walterjtv/Escritorio/SIDE/Embat-FinScore-Private/analysis/outputs/g_has_rest_vs_size.png` — last-month leftover `g_has_*` vs company size tercile (train).

## Closed in this module

- G card 0.551 replica: **CONFIRM**.
- checking 99.1% hole: **CONFIRM**. DROP from the 44.
- TPV last-month n=10: **CONFIRM**.
- HAS flags rise-only: **YES**.
- Honest leftover after days dies: **YES** (OLS leftover was a fake days leak).
- `g_custom_share`: **dies after accounts (0.548<0.55; ρ vs accounts=0.064 not a twin). Not a mix leftover.**.
- Dark 744/470: **CONFIRM**. Access ≠ ERP.
- Q6: **CLOSE** (connection clock).
- Drop remaining `g_has_*` + `g_custom_share` from the 44: **YES**.
- Did not invent `y_has_tpv`. Did not put `g_has_*` on the 15-col card. Night quotes unchanged.
