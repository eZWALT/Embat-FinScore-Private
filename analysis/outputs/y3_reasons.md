# Y3 card reasons — leftover-KEEP stems only

Generated `2026-09-19T07:41:51+02:00` by `analysis/evaluate/y3_reasons.py`.
DuckDB `clean` read-only. `monthly.parquet` / `targets.parquet` read-only.
Rates and leftover on **train** stressed months. Holdout 72 is coverage only
(72 companies / 1073 CM — no fit). Seed 20260918.
No 0–100. No `gbm_core.py`. No rewrite of `y3_importances.md` / `shap_y3.md`.
Y3 never B. Do not quote `a_out_vol` 0.722 as the engine.

Siddiqi (2017): reasons = factors actually scored; prefer 8–15 stems, but do
**not** pad DROPped SHAP names to hit 15. CFPB Circular 2023-03: reasons must
“relate to and accurately describe the factors actually considered or scored”;
the closest checklist item is not enough.

**Quote this:** among already-stressed months, recover (45→65) is more likely
when the book is quiet — no SS, no salary, fewer booking days. Those three
families are the only leftover-KEEP reasons. Historical SHAP still names
`a_n_tx` (perm 0.030); leftover QA DROPped it. We will not say it.

Night quotes locked: Y3 **0.762 / 0.752**. Days bar **0.711**.
Size bar **0.617**. `q6_keep` = `e_ar_issued_lag1` **0.626** /
`c_n_days_with_tx_lag1` **0.684** / `c_ss_month_lag1` leftover **0.631**.

Train Y3 labeled n=5,648 / n_pos=402. Replica days 0.711
(night 0.711); SS leftover 0.635 (night 0.635); salary leftover
0.603 (night 0.603).

---

## Judge-quotable reasons (factors actually scored)

Sign is − for every KEEP stem: among already-stressed months, **quiet** raises
P(recover). That is the 45→65 direction. It is not “more activity is healthier.”

| # | reason we will say | stem | sign | leftover / bar | brief Q |
|---|--------------------|------|:----:|----------------|---------|
| 1 | No social-security booking this month | `c_ss_month` | − | leftover after days **0.635** (replica 0.635) | Q3 turning, Q5 why |
| 2 | No salary booking this month | `c_salary_month` | − | leftover after days **0.603** (replica 0.603) | Q3 turning, Q5 why |
| 3 | Fewer distinct booking days this month | `c_n_days_with_tx` | − | **bar 0.711** (replica 0.711) | Q2 improving, Q3 turning |
| 4 | Fewer booking days last month | `c_n_days_with_tx_lag1` | − | **q6_keep 0.684** (replica 0.684) | Q6 months earlier |
| 5 | No SS booking last month | `c_ss_month_lag1` | − | leftover after days_lag1 **0.631** (replica 0.631) | Q6 months earlier |
| 6 | No salary booking last month | `c_salary_month_lag1` | − | leftover after days_lag1 **0.606** (replica 0.606) | Q6 footnote (lives; thinner than SS) |
| 7 | Same stems at t−3 | `*_lag3` | − | SS leftover after days_lag3 replica 0.620; salary 0.587; days_lag3 0.666 | Q6 CLOSE on company-short books |

Plain sentences a judge can read aloud:

1. **Who is healthy (Q1)?** Last-value `b_runway` / cash-buffer days. Not a card
   reason. Never B as Y3 X.
2. **Who is improving (Q2)?** Fewer booking days now than a busy stressed month
   (`c_n_days_with_tx` −, bar 0.711).
3. **Who is turning (Q3)?** Already-stressed months that go quiet on SS and
   salary (`c_ss_month` leftover 0.635, `c_salary_month` leftover 0.603). Engine
   0.762 / 15-col historical 0.752.
4. **Dip vs fall (Q4)?** Not this card. Invoice TURNOVER 0.720 is the sibling.
5. **Why did it change (Q5)?** Payroll-like drains stopped. SS leftover after
   days+salary still lives (published 0.633). Quiet is de-escalation, not size
   (`a_in3` leftover 0.521 DROP).
6. **How many months earlier (Q6)?** One month: days_lag1 0.684 and SS_lag1
   leftover 0.631. Hidden 72 stays a 1-month claim. `issued_lag1` 0.626 is Y7,
   not a Y3 reason.

---

## What we will NOT say

Historical SHAP (`y3_importances.md` 00:16; `shap_y3.md` 00:17) still ranks
these. Leftover QAs DROPped them as **card** stems. CFPB: do not check the
closest leftover name.

| dropped SHAP name | why it is not a reason | published leftover / flag |
|-------------------|------------------------|---------------------------|
| `a_n_tx` | days twin (ρ 0.938); perm ΔAUROC 0.030 is the trap | leftover after days **0.538** DROP |
| `a_op_in` | SIZE clone (ρ vs log inflow 0.998) | size bar 0.617; `a_in3` leftover **0.521** |
| `a_transfer` / `_lag1` / `_lag3` | company trait (ICC 0.956); perm ≈ 0 | leftover **0.579** CLOSE |
| `e_dso_proxy` | invoice sibling; sign flip SHAP − vs univ + | leftover after days **0.474** — see `lit_invoice` |
| `f_ds_r` / `_lag1` | unused leftover; twin of euro debt service | leftover **0.528** DROP |
| `c_gap_sd` | days twin (ρ −0.905) | leftover after days **0.535** DROP |
| `h_n_siblings_active_lag3` | sister *existence*, not sister health | Family H PARK as Y3 X |
| `a_out_vol` | company trait (demean 0.549); Javier vol DRIFT | raw **0.722** is not the engine |
| any `b_*` | honest Y≠X; walk is an identity | never B as Y2/Y3 X |
| `created_at` / `g_has_*` / `g_n_accounts` / `f_has_*` | connection clocks | `g_n_accounts` leftover **0.428** |
| `f_util_snapshot` | last-month-only 1.6% | Y10 impossible |
| `c_missed_salary` | skip, not presence | **0.513** CLOSE |
| `c_tax_month` | leftover after days dies | **0.520** DROP |

Do not pad the historical 15-col card back to 15 with those names. Siddiqi’s
8–15 is a preference; leftover QA left **three families**. The 15-col **0.752**
is a locked night quote, not a license to re-advertise DROPped SHAP.

---

## Six questions — card vs photograph

| # | question | what the card says | what it does not say |
|---|---------|--------------------|----------------------|
| 1 | Who is healthy? | Nothing. Q1 is last-value runway (Farrell buffer days). | Do not reason from B. |
| 2 | Who is improving? | Days − among already-stressed. | Not `c_gap_sd` irregularity. |
| 3 | Who is turning? | SS − and salary − (quiet-stressed recover). | Not bigger credits (Hair/FinRegLab PD). |
| 4 | Dip vs fall? | Out. Sibling TURNOVER. | Not DSO. |
| 5 | Why did it change? | Payroll-like drains off; activity clock down. | Not transfer / n_tx / ds_r / vol. |
| 6 | Months earlier? | days_lag1 + SS_lag1. | Not utilisation. Hidden 72 = 1 month. |

---

## Replica singles (train group-fold; sign −)

| stem | sign | raw CV | leftover after days | leftover folds | n_pos |
|------|:----:|-------:|--------------------:|----------------|------:|
| `c_n_days_with_tx` | -1 | 0.711 | 0.711 | 0.665 0.738 0.700 0.715 0.740 | 402 |
| `c_ss_month` | -1 | 0.693 | 0.635 | 0.646 0.582 0.698 0.617 0.630 | 402 |
| `c_salary_month` | -1 | 0.671 | 0.603 | 0.550 0.573 0.708 0.618 0.567 | 402 |
| `log1p(a_in3)` size bar | -1 | 0.617 | 0.617 | 0.565 0.632 0.683 0.543 0.661 | 391 |
| `c_n_days_with_tx_lag1` | -1 | 0.684 | 0.684 | 0.627 0.714 0.694 0.684 0.701 | 402 |
| `c_ss_month_lag1` after days_lag1 | -1 | 0.631 | 0.631 | 0.636 0.561 0.695 0.615 0.649 | 402 |
| `c_salary_month_lag1` after days_lag1 | -1 | 0.606 | 0.606 | 0.591 0.568 0.707 0.629 0.535 | 402 |
| `c_n_days_with_tx_lag3` | -1 | 0.666 | 0.666 | 0.637 0.710 0.658 0.658 0.665 | 355 |
| `c_ss_month_lag3` after days_lag3 | -1 | 0.620 | 0.620 | 0.622 0.512 0.699 0.619 0.647 | 355 |
| `c_salary_month_lag3` after days_lag3 | -1 | 0.587 | 0.587 | 0.542 0.544 0.690 0.618 0.540 | 355 |
| `c_ss_month` after days+salary | -1 | 0.633 | 0.633 | 0.653 0.585 0.681 0.608 0.640 | 402 |
| `c_salary_month` after days+SS | -1 | 0.591 | 0.591 | 0.540 0.567 0.678 0.612 0.559 | 402 |

---

## Extras — fold-wise reason stability

Leftover-after-days rank, five group folds (seed 20260918). KEEP if the
**mean** leftover stays ≥0.55 and at least four folds live. Do not drop a
stem because one fold dips.

| stem | leftover folds | mean | min | n_pos/fold |
|------|----------------|-----:|----:|------------|
| `c_ss_month` | 0.646 0.582 0.698 0.617 0.630 | 0.635 | 0.582 | 54 93 66 82 107 |
| `c_salary_month` | 0.550 0.573 0.708 0.618 0.567 | 0.603 | 0.550 | 54 93 66 82 107 |
| `c_n_days_with_tx` raw | 0.665 0.738 0.700 0.715 0.740 | 0.711 | 0.665 | 54 93 66 82 107 |

Published SS leftover folds 0.646 0.582 0.698 0.617 0.630 (spread 0.116).
Published salary leftover folds 0.550 0.573 0.708 0.618 0.567. Salary fold 0
sits on the 0.55 line — KEEP the **mean** 0.603, do not drop the stem.

---

## Extras — dark 470 vs ERP

Reasons are bank-book flags. Dark companies have no ERP invoices; they still
have SS/salary/days on the treasury trail. Do not require an invoice to say
quiet-stressed.

| slice | days raw | SS leftover | salary leftover | n_pos |
|-------|---------:|------------:|----------------:|------:|
| never-ERP (dark) | 0.704 | 0.574 | 0.547 | 138 |
| ever-ERP | 0.730 | 0.677 | 0.634 | 264 |

Published: SS leftover invoiced **0.677** / dark **0.574**; salary leftover ERP
**0.634** / dark **0.547** (dark salary leftover dies). Card sentence on a
**dark** book: say SS + days; do not lean on salary alone.

---

## Extras — short vs long books

`q6_keep` is a 1-month claim. Lag3 is CLOSE on company-short books
(days_lag3 nn 36.3%). Contemporaneous days/SS are SIGNAL on short books, not
lead time.

| slice | days raw | SS leftover | salary leftover | days_lag1 | n_pos |
|-------|---------:|------------:|----------------:|----------:|------:|
| so-far <12 (month clock) | 0.696 | 0.642 | 0.612 | 0.684 | 252 |
| so-far ≥18 (month clock) | — | — | — | — | 16 |
| company book <12 | — | — | — | — | 13 |
| company book 12–17 | — | — | — | — | 34 |
| company book ≥18 | 0.714 | 0.632 | 0.596 | 0.685 | 355 |

so-far ≥18 Y3 labels are almost empty (need t+1..t+6). Company book <12 is
LOW_POWER (recover Y needs a future window). **Company-long** (≥18) is the
honest long-book slice — days 0.714 / SS leftover 0.632 / salary leftover 0.596
/ days_lag1 0.685. Hidden 72: =24 books **4.2%**. Do not fit. Do not claim a
3-month Y3 lead there.

355 / 402 Y3 positives sit on company-long books (88%). Company-short and
mid books are LOW_POWER for recover. The card is a **long-book reason list**.

---

## Extras — size tercile (published leftover; not re-fit)

Salary leftover after days **dies on T1** (0.399) and lives on T2 (0.620) /
T3 (0.653). SS leftover after days lives on T1 (0.576) and T2+T3 (0.616).
On the smallest stressed firms, say **days + SS**, not salary. Size bar
0.617 is not a reason.

---

## Explicitly out of this file

- A 0–100 formula, reason-code integers, ECOA adverse-action letters.
- Refit of the 278-col tree or the historical 15-col A (0.752 stays a quote).
- Family I / M merge. Wave D NSF token count. Wave A/B runway / Δdays.
- Invoice reasons (`e_dso_proxy`, issued, CN) — see `lit_invoice` / TURNOVER.
- Bankruptcy / Altman language.

Plot: `y3_reasons.png`.

Elapsed read-only. Night Y3 0.762/0.752, days 0.711, size 0.617 unchanged.
