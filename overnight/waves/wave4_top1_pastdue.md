# Wave 4 — top-1 AR PastDue% leftover (Wave A)

Owner: `top1_pastdue_qa`. As-of `2026-09-19T07:44:34+02:00`.
Deliverable: `analysis/outputs/top1_pastdue_qa.md`.
Do not grow TURNOVER **0.720**. Delay footnote 0.581 stays.

## Headline

top-1 PastDue defined 32.6% of train CM; dark 470 NaN CONFIRM. mean 0.586 (Hirshleifer 27%). ρ vs delay 0.278 DSO 0.291 overdue 0.858 TWIN. Y7 raw 0.612 vs issued_lag1 0.630 size 0.469 beat-size PASS. Leftover after issued_lag1+CN rank 0.576 OLS 0.621 (issued-only 0.567; after delay 0.597). ICC 0.918 demean leftover 0.569. Boot p05 0.551. Y7 leftover **CLOSE**. Do not grow TURNOVER 0.720. Delay footnote 0.581 stays.

## Decision

- Y7 leftover after issued_lag1+CN: **CLOSE** (leftover 0.576 dies / twin=['e_ar_overdue'] — keep the delay footnote 0.581)
- rank 0.576 OLS 0.621 issued-only 0.567 after-delay 0.597
- Q6: **KEEP**. Y3 card: **DROP**.

## Do not do next

- Put this on TURNOVER. Merge D as Y7 X. Score as Y5 X.
- Overwrite delay_qa / credit_note_qa. Invent `y_delay`.

## Extras after headline

- Twin of firm `e_ar_overdue` (not delay). After overdue leftover dies; after delay leftover 0.597.
- Diversified share<0.50 leftover 0.615 ρ-overdue 0.784; monopoly 0.519 ρ 0.939.
- PastDue30 leftover 0.556; gap(named−overdue) 0.543; triple+overdue 0.524.
- Boot after delay p05 0.558 p50 0.601. Interior leftover 0.527.
- Diversified after issued+CN+overdue 0.454 twins=none.
- CLOSE stays. Delay footnote 0.581. Do not grow TURNOVER 0.720.
