# Unused leftover of contemporaneous `a_fin_cost` after `c_n_days_with_tx`

Generated `2026-09-19T07:33:39+02:00` by agent `b17e9c44`. DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only. Rates and AUROC on **train**. Holdout 72 is coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite. No new GBM. Do not invent a Y. Do not merge M. Do not put `a_fin_cost` on the 15-col card. Do not grow TURNOVER. Do not overwrite `y9_why.*` / `fc_r_qa.*` / `ds_r_qa.*` / `catmix.py` / `util_snap_qa.*` / `ogtg_qa.*` / `n_types_qa.*`. Y3 never B. Night Y3 **0.762 / 0.752**. Days **0.711**. Size **0.617**. Y7 TURNOVER **0.720 / 0.712**.

`a_fin_cost` = -sum(amount | grp = fin_cost). Contemporaneous `f_fc_r` already DROP leftover after days 0.449. KEEP `f_fc_r_lag3` on TURNOVER. Y9 is the fee label — leftover as Y3 X, not as Y9 X.

## Headline

CLOSE as Y3 X. Leftover after days **CLOSE** rank 0.483 OLS 0.683 fake=True. Y3 native 0.634 vs days 0.711 vs size 0.617 vs f_fc_r 0.559. beat_size=0.018. Twin=['f_fin_cost'] ρ days 0.531 f_fin_cost 1.000 f_fc_r 0.663. After f_fc_r 0.634. Q6 CLOSE. CLOSE unused leftover / DROP from the 44 as Y3 X. Night Y3 0.762/0.752, days 0.711, size 0.617, TURNOVER 0.720/0.712 unchanged. Do not put a_fin_cost on the 15-col card. KEEP f_fc_r_lag3 on TURNOVER.

## Brief questions

| q | cut |
| --- | --- |
| 1 Who is healthy? | Y3 leftover after days 0.483 (dies). Native 0.634 vs size 0.617. |
| 2 Who is improving? | Within-company path rises 6033 drops 5961 (flow, not inventory). |
| 3 Who is turning? | Y9 forbids fee raw material as X. ρ vs Y9 0.155. Leftover is Y3 X, not Y9 X. |
| 4 Dip vs fall? | Y2 leftover after days 0.455. Do not grow TURNOVER 0.720. |
| 5 Why did it change? | After f_fc_r 0.634 after f_fin_cost 0.635. Twin=['f_fin_cost']. |
| 6 Months earlier? | lag1 leftover after days_lag1 0.469. Q6 CLOSE. |


## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| a_fin_cost leftover after days | **CLOSE** | rank 0.483 OLS 0.683 fake=True |
| as Y3 X (not on 15-col card) | **CLOSE unused leftover / DROP from the 44 as Y3 X** | leftover after days 0.483 dies; beat_size=0.018; twin=['f_fin_cost']. |
| Twin / SIZE | twin=YES SIZE=no | ρ days 0.531 f_fin_cost 1.000 f_fc_r 0.663 size 0.430 |
| Q6 lag leftover | **CLOSE** | Q6 lag1 leftover after days_lag1 0.469 ρ=-0.840 fake=True. Days lag1 0.684 (0.684 CONFIRM). lag3 leftover after days_lag3 0.459. Q6 CLOSE. |
| f_fc_r_lag3 on TURNOVER | **KEEP** | Do not rip; contemporaneous f_fc_r leftover 0.449 already DROP. |
| as Y9 X / Family M | **CLOSE** | Y9 descriptive only (not as Y9 X): ρ a_fin_cost~Y9 0.155 Jaccard(hi,Y9+)=0.085. Leftover after days on Y9-defined 0.533. Do not treat a_fin_cost as Y9 X. Do not merge M. |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · TURNOVER 0.720/0.712 |


## 1. Coverage — flow vs snapshot

Train a_fin_cost defined 21,157/21,157 (100.0%). zeros 9,209 (43.5%) >0 11,943 (56.4%). last-month-only=False periods_with_def=24 last-nn 1214. median 10.000 p95 9226.068. acf1 -0.060 acf3 -0.011. ever>0 1026/1214 companies. ρ vs log1p(a_in3) 0.430 (not SIZE).

| col | n_nn | cov | share>0 | p50 | last_only |
| --- | --- | --- | --- | --- | --- |
| a_fin_cost | 21,157 | 100.0% | 56.4% | 10.000 | False |


Spearman twins |ρ|≥0.80 on a_fin_cost: ['f_fin_cost']. vs days 0.531 vs a_n_tx 0.545 vs f_fin_cost 1.000 vs f_fc_r 0.663 vs a_debt_service 0.331 vs size 0.430.

| vs | ρ | twin |
| --- | --- | --- |
| c_n_days_with_tx | 0.531 |  |
| a_n_tx | 0.545 |  |
| f_fin_cost | 1.000 | YES |
| f_fc_r | 0.663 |  |
| a_debt_service | 0.331 |  |
| f_ds_r | 0.291 |  |
| log1p(a_in3) | 0.430 |  |


## 2. Single-feature train group-fold AUROC

Y3 native a_fin_cost 0.634 n=5648 n_pos=402. vs days 0.711 (0.711 CONFIRM) vs size 0.617 (0.617 CONFIRM) vs f_fc_r 0.559 vs f_fin_cost 0.634. beat_size=0.018.

| y | feature | n | n_pos | CV | sign | folds |
| --- | --- | --- | --- | --- | --- | --- |
| y3_recover_cash_6m | a_fin_cost | 5,648 | 402 | 0.634 | -1 | 0.607 0.666 0.703 0.583 0.613 |
| y3_recover_cash_6m | f_fin_cost | 5,648 | 402 | 0.634 | -1 | 0.607 0.666 0.703 0.583 0.613 |
| y3_recover_cash_6m | f_fc_r | 5,528 | 391 | 0.559 | -1 | 0.550 0.541 0.703 0.500 0.500 |
| y3_recover_cash_6m | a_debt_service | 5,648 | 402 | 0.613 | -1 | 0.570 0.643 0.650 0.591 0.613 |
| y3_recover_cash_6m | c_n_days_with_tx | 5,648 | 402 | 0.711 | -1 | 0.665 0.738 0.700 0.715 0.740 |
| y3_recover_cash_6m | log1p(a_in3) | 5,528 | 391 | 0.617 | -1 | 0.565 0.632 0.683 0.543 0.661 |
| y2_neg_2of3 | a_fin_cost | 17,356 | 1,271 | 0.546 | 1 | 0.536 0.644 0.522 0.492 0.536 |


## 3. Honest leftover after days + inverse + f_fc_r / f_fin_cost

Honest leftover after days rank 0.483 OLS 0.683 ρ(resid,days)=-0.845 fake=True. After f_fc_r 0.634 after f_fin_cost 0.635 after a_debt_service 0.596 after days+f_fc_r 0.529. Inverse days after a_fin_cost 0.673. Honest leftover DIES.

| cut | OLS | rank | ρctrl | fake | n_pos |
| --- | --- | --- | --- | --- | --- |
| after days | 0.683 | 0.483 | -0.845 | True | 402 |
| after f_fc_r | 0.618 | 0.634 | -0.047 | False | 391 |
| after f_fin_cost | 0.634 | 0.635 | -1.000 | True | 402 |
| after a_debt_service | 0.583 | 0.596 | 0.014 | False | 402 |
| after days+f_fc_r | 0.641 | 0.529 | -0.677 | False | 391 |
| after days+f_fin_cost | 0.707 | 0.633 | -0.970 | True | 402 |
| after size | 0.602 | 0.592 | -0.871 | True | 391 |
| days after a_fin_cost | 0.711 | 0.673 | 0.505 | False | 402 |


## 4. Path (flow, not inventory)

Within-company a_fin_cost path: rises 6033 drops 5961 same 7949 pairs 19943 rise-only=False (flow, not inventory).

## 5. Dark vs ERP leftover

Last-month ever-ERP 744 / never-ERP 470 (CONFIRM 744/470). Leftover after days invoiced 0.536 dark 0.472.

| book | n_co | n_nn | share>0 | leftover | fake |
| --- | --- | --- | --- | --- | --- |
| invoiced | 744 | 13554 | 58.0% | 0.536 | True |
| dark | 470 | 7603 | 53.7% | 0.472 | True |


## 6. Q6 — lag leftover after days_lag1

Q6 lag1 leftover after days_lag1 0.469 ρ=-0.840 fake=True. Days lag1 0.684 (0.684 CONFIRM). lag3 leftover after days_lag3 0.459. Q6 CLOSE.

| cut | rank | fake |
| --- | --- | --- |
| a_fin_cost_lag1 after days_lag1 | 0.469 | True |
| a_fin_cost_lag3 after days_lag3 | 0.459 | True |
| days_lag1 single | 0.684 | False |


## 7. Y9 descriptive (not as Y9 X)

Y9 descriptive only (not as Y9 X): ρ a_fin_cost~Y9 0.155 Jaccard(hi,Y9+)=0.085. Leftover after days on Y9-defined 0.533. Do not treat a_fin_cost as Y9 X. Do not merge M.

Holdout coverage only (no fit): 72 co / 1,073 CM, defined 1073 >0 544.

Replica f_fc_r leftover after days 0.449 (quote 0.449 CONFIRM).

Plot: `fin_cost_qa.png`.

## What failed / next

- no replica miss; leftover after days 0.483 dies and is a fake days leak (ρ=-0.845; OLS 0.683). Exact twin of f_fin_cost ρ=1.000. beat_size +0.018 fails. CLOSE leftover / DROP from the 44. Do not put a_fin_cost on the 15-col card. KEEP f_fc_r_lag3 on TURNOVER.

## Later extras (same module)

- Company bootstrap leftover-after-days n=32 p05/p50/p95 0.431 / 0.472 / 0.554.
- On a_fin_cost>0 leftover after days 0.582 n_pos=197 ρ=-0.845 fake=True. >0 dummy leftover after days 0.456 fake=False.
- BETWEEN leftover of company-mean a_fin_cost after days-mean 0.416 ρ=-0.745 fake=False; WITHIN leftover 0.548 fake=False.
- Y2 leftover after days 0.455 ρ=-0.845 fake=True n_pos=1271.
- Exact rewrite: a_fin_cost==f_fin_cost on 21,157/21,157 (100.0%) max|Δ|=0.000000. f_fin_cost leftover after days 0.483 fake=True (same object as a_fin_cost).
- log1p(a_fin_cost) leftover after days 0.483 ρ=0.006 fake=False; after days+size 0.536.
- Monthly a_fin_cost/a_in3 leftover after days 0.536 ρ=0.986 fake=True; after f_fc_r 0.575; ρ vs f_fc_r 0.778 (3m rate vs month ratio).
- Leftover after days+a_n_tx 0.479 ρ=-0.849 fake=True; after a_n_tx alone 0.480.
- Inverse leftover of f_fc_r after a_fin_cost 0.508 ρ=0.645 fake=False; after a_fin_cost+days 0.459 (f_fc_r leftover after days quote 0.449).
- Leftover-after-days rank folds 0.518 0.560 0.368 0.483 0.487 rank 0.483 ρ=-0.845 fake=True.
- Permute a_fin_cost leftover-after-days null p50 0.542 p(obs≥null)=1.000 obs=0.483.
- Leftover after days by SIZE tercile: T1 0.502 / T2 0.439 / T3 0.575.
- Leftover after f_ds_r 0.604 ρ=0.665 fake=False; after days+f_ds_r+f_fc_r 0.523; inverse a_debt_service leftover after a_fin_cost 0.559.
- Ever-fee company dummy leftover after days 0.627 fake=False; a_fin_cost leftover after days on ever>0 companies 0.454 n_pos=349.
- Leave-one-group leftover-after-days n=235 min/med/max 0.477 / 0.483 / 0.546.
- SIZE T3 leftover-after-days folds 0.531 0.699 0.652 0.473 0.518 rank 0.575 n_pos=126 ρ=-0.845 fake=True.
- Q6 lag1 leftover after contemporaneous days 0.446 ρ=-0.855 fake=True; lag3 after days 0.508.
- log1p leftover after days invoiced 0.536 ρ=0.006 fake=False dark 0.472 ρ=0.006 fake=False.
- Ever-fee dummy leftover after days+size 0.610 ρ=-0.597 fake=False; after days+f_fc_r 0.457. Amount leftover after ever-dummy 0.614; after ever+days 0.483. Inverse days after ever-dummy 0.698.
- Ever-fee dummy leftover-after-days folds 0.567 0.624 0.533 0.661 0.748 rank 0.627; Y2 0.582.
- High-fee p90 dummy (cut=2307.436) leftover after days 0.638 ρ=-0.646 fake=False; after days+size 0.612.
- On f_fc_r>0 leftover after days 0.510 n_pos=274 ρ=-0.845 fake=True; after f_fc_r+days 0.523.
- High-fee dummy leftover after days+f_fc_r 0.593; after days+size+f_fc_r 0.590; Y2 after days 0.508; amount leftover after high-fee+days 0.539.
- High-fee dummy leftover-after-days folds 0.628 0.648 0.583 0.691 0.642 rank 0.638; after days+size folds 0.586 0.621 0.594 0.633 0.627 rank 0.612.
- Ever-fee dummy leftover after days invoiced 0.675 dark 0.565.
- Company bootstrap ever-fee dummy leftover-after-days n=24 p05/p50/p95 0.579 / 0.634 / 0.661.
- High-fee dummy leftover after days+a_n_tx 0.631; after days+f_ds_r 0.654; lag1 leftover after days_lag1 0.619; inverse f_fc_r leftover after high-fee dummy 0.547.
- High-fee dummy leftover after days invoiced 0.675 dark 0.588.
- On high-fee months leftover of amount after days — n_pos=25 ρ=-0.845 fake=True.
- p80-fee dummy leftover after days 0.556 folds 0.619 0.511 0.477 0.625 0.547; after days+size+f_fc_r 0.523; Jaccard vs Y9 0.076 (Y9 is the fee label — not as Y9 X).
- log1p leftover after days+f_fc_r 0.529 ρ=0.012 fake=False; after days+f_fc_r+size 0.530.
- High-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r 0.588 folds 0.555 0.606 0.615 0.606 0.557 ρ=-0.488 fake=False.
- MoM Δ a_fin_cost leftover after days 0.515 ρ=0.739 fake=False; after days+lag1 0.527 (Q2 path).
- Rolling-3m a_fin_cost leftover after f_fc_r 0.665 ρ vs f_fc_r 0.803; after days 0.461 fake=True.
- Leftover after days short_<12 0.444 n_pos=252; long_>=18 — n_pos=16.
- Y2 high-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r 0.480 n_pos=1044.
- a_debt_service leftover after days 0.484 fake=False; a_fin_cost leftover after days+a_debt_service 0.480; on f_ds_r>0 leftover after days 0.473 n_pos=66.
- a_fin_cost/(a_fin_cost+a_debt_service) leftover after days 0.539 ρ=0.286 fake=False; after f_fc_r 0.607.
- Leave-one-group high-fee dummy leftover-after-days n=235 min/med/max 0.629 / 0.638 / 0.649.
- Amount leftover after f_fc_r_lag3 0.656; after days+f_fc_r_lag3 0.555; high-fee dummy leftover after days+f_fc_r_lag3 0.620 (KEEP lag3 on TURNOVER — do not rip).
- High-fee dummy leftover after days+size+f_fc_r+a_n_tx+f_ds_r+a_debt_service 0.587 folds 0.559 0.608 0.615 0.599 0.551.
- Amount leftover after days+f_fc_r_lag3 folds 0.525 0.584 0.572 0.596 0.497 rank 0.555; leftover after days on so-far≥6 0.540 n_pos=313.
- High-fee dummy leftover after days+f_fc_r_lag3+size 0.584 folds 0.578 0.569 0.615 0.552 0.605 ρ=-0.581 fake=False.
- Amount leftover after days+f_fc_r+f_fc_r_lag3 0.528 folds 0.499 0.569 0.529 0.543 0.498; high-fee dummy leftover after days+f_fc_r_lag3+size+a_n_tx 0.580 folds 0.550 0.574 0.611 0.560 0.604.
- On ever-high-fee companies amount leftover after days 0.471 n_pos=102; ever-high dummy leftover after days 0.530 fake=False.
- High-fee dummy leftover after days+f_fc_r+f_fc_r_lag3+size 0.597 folds 0.605 0.587 0.619 0.579 0.595; Y2 after days+f_fc_r_lag3 0.500; log1p leftover after days+f_fc_r+f_fc_r_lag3 0.527.
- High-fee dummy BETWEEN leftover after days-mean 0.532 ρ=-0.396 fake=False; WITHIN 0.514.
- Amount leftover after days on so-far≥6 invoiced 0.536 n_pos=208; dark 0.515 n_pos=105.
- High-fee dummy leftover after days on so-far≥6 0.657 n_pos=313; amount leftover after days on never-high-fee companies 0.470 n_pos=300.
- High-fee dummy leftover after days+f_fc_r on so-far≥6 0.616 n_pos=313; Y2 leftover after days on so-far≥6 0.527.
- Amount leftover after days+f_fc_r on so-far≥6 0.522 n_pos=313; high-fee dummy leftover after days+size on so-far≥6 0.612.
- High-fee dummy leftover after days+size+f_fc_r on so-far≥6 0.601 folds 0.610 0.591 0.618 0.588 0.598; amount leftover after days+f_fc_r_lag3 on so-far≥6 0.555.

Elapsed 59s.

Did **not**: overwrite `y9_why.*` / `fc_r_qa.*` / `ds_r_qa.*` / `catmix.py` / `util_snap_qa.*` / `ogtg_qa.*` / `n_types_qa.*`, edit `debt.py` / `gbm_core.py`, put a_fin_cost on the 15-col card, grow TURNOVER, invent a Y, merge M, write 0–100, fit holdout, touch `product/`, run `build_targets`.

