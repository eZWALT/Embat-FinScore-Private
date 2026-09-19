# Unused leftover of Δdays after days-level (Norden lead without utilisation)

Generated `2026-09-19T07:45:06+02:00` by `analysis/evaluate/days_delta_qa.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Rates and leftover on **train**. Holdout 72 / 1073 CM
coverage only. Seed 20260918 group folds. No 0–100. No parquet rewrite.
No new GBM. No `build_targets`. Do not invent utilisation / Y10. Do not
reconstruct Norden AMPLI (HIGH−LOW of balances) — B-forbidden. Y3 never B.
Do not quote `a_out_vol` 0.722 as the engine. Do not overwrite `n_tx_qa.*` /
`recency_qa.*` / `gap_sd_qa.*` / `y3_reasons.md`.

`d3` = `c_n_days_with_tx` − lag3. `d6` = days − lag6. Primary slice:
companies with **≥18 months on book** (714 train companies).
Y3 labeled n=5,648 / n_pos=402.

## Headline

DROP on ≥18m books. Δ3 leftover after days-level rank 0.484 (OLS 0.526, fake=False). Δ6 leftover 0.517. Δ3 raw 0.557 vs days 0.714 vs size 0.609 beat3=False Δ=-0.060. SIZE3=False twins3=none. Inverse days-after-Δ3 0.712. Card: DROP from Y3 X / KEEP off the 15-col card. Night Y3 0.762/0.752, days 0.711, size 0.617, days_lag1 0.684 unchanged. Hidden 72 is 1-month only.

## Brief questions

| # | question | what this cut says |
| --- | --- | --- |
| 1 | Who is healthy? | Not Δdays. Q1 is last-value runway. Never B. |
| 2 | Who is improving? | Days **level** 0.711 is the bar. Δ is the path test. |
| 3 | Who is turning? | DROP: Δ3 leftover 0.484 after days-level on ≥18m. Quiet-stressed recover stays SS/salary/days. |
| 4 | Dip vs fall? | Out. Sibling TURNOVER. |
| 5 | Why did it change? | If KEEP, falling activity leftover after the *level*. If not, the level already ate the change. |
| 6 | Months earlier? | Hidden 72 stays **1-month** (`days_lag1` 0.684). Δ3/Δ6 are 3-/6-month clocks on long books only — not a holdout claim. |

## KEEP / CLOSE / DROP / PARK

| object | decision | why |
| --- | --- | --- |
| Δ3 leftover after days-level (≥18m) | **DROP** | rank 0.484 OLS 0.526 fake=False; beat_size=False (-0.060); SIZE=False; twins=none |
| Δ6 leftover after days-level (≥18m) | **DROP** | rank 0.517 OLS 0.522 fake=False; beat_size=False (-0.032); SIZE=False; twins=none |
| 15-col Y3 card | **DROP from Y3 X / KEEP off the 15-col card** | KEEP-as-X requires leftover ≥0.60 AND beat size ≥0.02 AND not SIZE AND not twin |
| Inverse: days leftover after Δ3 | **lives** | rank 0.712 — the 0.711 bar must survive |
| AMPLI / utilisation / Y10 | **PARK** | B-forbidden; snapshot 1.6% |
| Hidden-72 Δ3/Δ6 claim | **PARK** | 1-month only |
| Night quotes | **unchanged** | Y3 0.762/0.752 · days 0.711 · size 0.617 · days_lag1 0.684 |

## 1. Coverage / Δ definition (train, ≥18m books)

d3 defined on Y3-long n=4,722 / n_pos=327.
d6 defined n=3,669 / n_pos=274.
Median Δ3 on defined long Y3 = 0.000; share Δ3<0 = 0.408.
acf1(Δ3) 0.139 vs acf1(days) 0.145 on long books.

## 2. Spearman twins + SIZE (Δ3 on ≥18m Y3-defined)

| vs | ρ | SIZE |
| --- | --- | --- |
| log1p(a_in3) | 0.015 | no |
| days | 0.139 | no |
| a_n_tx | 0.068 | no |
| c_gap_sd | -0.003 | no |
| c_recency_days | -0.113 | no |


Δ6 twins:

| vs | ρ | SIZE |
| --- | --- | --- |
| log1p(a_in3) | 0.056 | no |
| days | 0.191 | no |
| a_n_tx | 0.122 | no |
| c_gap_sd | -0.057 | no |
| c_recency_days | -0.174 | no |


## 3. Single-feature group-fold AUROC

Sign from the train side of each fold. Never holdout fit. Never Y7.

| stem | sign | raw | leftover | folds | n_pos |
| --- | --- | --- | --- | --- | --- |
| days level (≥18m) | -1 | 0.714 | 0.714 | 0.672 0.748 0.691 0.725 0.736 | 355 |
| Δ3 raw (≥18m) | -1 | 0.557 | 0.557 | 0.557 0.520 0.574 0.576 0.561 | 327 |
| Δ6 raw (≥18m) | -1 | 0.585 | 0.585 | 0.587 0.612 0.558 0.579 0.591 | 274 |
| size log1p(a_in3) (≥18m) | -1 | 0.609 | 0.609 | 0.559 0.628 0.647 0.550 0.663 | 351 |
| days_lag1 (≥18m) | -1 | 0.685 | 0.685 | 0.632 0.731 0.677 0.691 0.696 | 355 |
| Δ3 leftover after days (≥18m) | -1 | 0.557 | 0.484 | 0.517 0.462 0.467 0.474 0.501 | 327 |
| Δ6 leftover after days (≥18m) | -1 | 0.585 | 0.517 | 0.526 0.529 0.508 0.510 0.513 | 274 |
| days leftover after Δ3 (inverse) | -1 | 0.714 | 0.712 | 0.681 0.756 0.667 0.727 0.730 | 327 |


## 4. All-train / short / mid (expect CLOSE / LOW_POWER)

| slice | n_pos | days | Δ3 raw | Δ3 leftover | Δ6 leftover | role Δ3 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| all train | 402 | 0.711 | 0.556 | 0.487 | 0.518 | DROP |
| company <12 | 13 | — | — | — | — | LOW_POWER |
| company 12–17 | 34 | — | — | — | — | LOW_POWER |
| company ≥18 | 355 | 0.714 | 0.557 | 0.484 | 0.517 | DROP |

## 5. Honest leftover after days-level (primary ≥18m)

Δ3 leftover rank 0.484 OLS 0.526 ρ(resid,days)=-0.199 R²=0.074 fake=False.
Folds 0.517 0.462 0.467 0.474 0.501.
Δ6 leftover rank 0.517 OLS 0.522 ρ(resid,days)=-0.225 fake=False.
Folds 0.526 0.529 0.508 0.510 0.513.

## Extra — inverse leftover of days after Δdays

Days leftover after Δ3 0.712 (OLS 0.695).
Days leftover after Δ6 0.712.
If inverse dies, the 0.711 bar would be a rewrite of the change — it must live.

## Extra — company bootstrap leftover (≥18m, n=80)

Δ3 leftover p05/p50/p95 0.450 / 0.486 / 0.529; share<0.55=1.000 (n=80).
Δ6 leftover p05/p50/p95 0.451 / 0.503 / 0.547; share<0.55=0.950.

## Extra — dark vs ERP (long books)

| slice | n_pos | days | Δ3 leftover | Δ6 leftover |
| --- | ---: | ---: | ---: | ---: |
| long dark | 118 | 0.701 | 0.422 | 0.562 |
| long ERP | 237 | 0.740 | 0.459 | 0.466 |

## Extra — first6 vs later months on long books

First six months of a long book vs later. Δ3 needs lag3 so first6 is mostly empty.

| slice | n_pos | Δ3 leftover | Δ6 leftover | days |
| --- | ---: | ---: | ---: | ---: |
| long first6 | 81 | 0.535 | — | 0.640 |
| long later | 274 | 0.493 | 0.517 | 0.732 |

## Extra — Δ3 leftover after days+SS; Y2 lock

Δ3 leftover after days+SS on ≥18m 0.514 (must not steal the SS reason).
Y2 leftover of Δ3 after days-level on ≥18m 0.524 — do not merge with Y2 trees.

## Extra — falling-only / |Δ3| / leftover after days_lag1

Norden's lead is activity *falling*. Restricting to Δ3<0 does not rescue leftover.

| slice | n_pos | raw | leftover after days | folds |
| --- | ---: | ---: | ---: | --- |
| Δ3<0 (falling) | 146 | 0.626 | 0.458 | 0.504 0.363 0.452 0.575 0.398 |
| Δ3>0 (rising) | 119 | — | 0.580 | 0.630 0.575 0.515 0.602 0.578 |
| abs(Δ3) | 327 | 0.543 | 0.577 | 0.646 0.435 0.567 0.655 0.579 |
| Δ3 leftover after days_lag1 | 327 | — | 0.536 | 0.543 0.494 0.556 0.553 0.536 |

`days_lag1` 0.684 stays q6_keep. Δ3 leftover after the lag still dies — the 3-month change is not a lead beyond last month's level.
Rising leftover 0.580 is unused leftover below the 0.60 KEEP bar, fold-noisy, and the **wrong sign story** for quiet-stressed recover. Do not KEEP it. abs(Δ3) leftover 0.577 is the same: lives thin, fold 1 dies (0.435).

## Explicitly out

- AMPLI = HIGH−LOW of balances (B). `f_util_snapshot` / Y10.
- Putting Δdays on the 15-col card unless KEEP-as-X cleared.
- Hidden-72 3- or 6-month claims. `a_out_vol` 0.722 as the engine.
- Rewrites of `n_tx_qa.*`, `recency_qa.*`, `gap_sd_qa.*`, `y3_reasons.md`, `gbm_core.py`.

Plot: `days_delta_qa.png`.

Night Y3 0.762/0.752, days 0.711, size 0.617 unchanged.
