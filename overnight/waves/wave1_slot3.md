# Wave 1 slot 3 — Family E (receivables / payables)

## Files written

- `analysis/features/invoices.py` — `SOURCE_TABLES=["invoices"]`, `FAMILY="e"`, `build(con, grid)`.
- This note. Smoke script stayed in `/tmp/slot3_smoke.py` (not in repo).

## Decisions

- AR = `amount > 0`, AP = `amount < 0` (journal 2210 sign check).
- Book filter: `document_type='invoice'`, `status<>'cancel'`, `amount<>0`. Dates used only if `<= period_end`. Open reconstructs from issuance/payment; does **not** trust current `status='overdue'` (that is as-of extract).
- `payment_date_invalid` rows stay in issuance volume, drop out of open / overdue / delay / pending (timing unknown). `is_extreme` kept (not dropped).
- Delay: amount-weighted `(paid_dt - due)` on payments in a trailing 3 months (90 days on a weekly grid), clip `[-30, 120]`. Null when `period < 2025-03-01` (first 6 calendar months of the panel — left truncation). Matches `score_pipeline._invoice_features` (Spearman 1.0, median abs diff 0).
- DSO/DPO = open / this-period issued (months outstanding). Null if issued is 0.
- No `credit_note` type in the dump. `e_credit_note_ratio` uses ERP stand-ins `note` + `refund` ("Abono" / "Factura correctiva").
- `e_pending_amt_share` is reconstructed (0 if paid by period end, else `|amount|`) over invoices issued by period end. Snapshot `pending_amount` would leak later collections.
- No holdout fitting (no percentiles / bins / centroids / `REF`).

## Columns (`e_*`)

`e_ar_open`, `e_ap_open`, `e_ar_overdue`, `e_ap_overdue`, `e_ar_overdue_30`, `e_ap_overdue_30`, `e_delay_coll`, `e_delay_paid`, `e_dso_proxy`, `e_dpo_proxy`, `e_credit_note_ratio`, `e_pending_amt_share`, `e_fx_share`, `e_ar_issued`, `e_ap_issued`.

## Train coverage

Panel: 21,157 train company-months / 1,214 train companies (holdout 1,073 cm left in the frame, unused). 744 train companies have invoices (40 holdout); 470 train companies are all-NaN on family E (no ERP invoices).

| column | pct train cm non-null | pct train companies any | n cm | train p50 |
|---|---:|---:|---:|---:|
| e_ar_open | 64.1% | 61.3% | 13,554 | 43,377 |
| e_ap_open | 64.1% | 61.3% | 13,554 | 49,042 |
| e_ar_overdue | 46.4% | 54.9% | 9,820 | 0.890 |
| e_ap_overdue | 57.5% | 61.0% | 12,173 | 0.680 |
| e_ar_overdue_30 | 46.4% | 54.9% | 9,820 | 0.581 |
| e_ap_overdue_30 | 57.5% | 61.0% | 12,173 | 0.397 |
| e_delay_coll | 31.9% | 48.9% | 6,752 | 2.58 d |
| e_delay_paid | 40.7% | 56.4% | 8,602 | 1.09 d |
| e_dso_proxy | 40.6% | 55.4% | 8,583 | 1.74 mo |
| e_dpo_proxy | 51.2% | 61.0% | 10,829 | 1.78 mo |
| e_credit_note_ratio | 53.0% | 61.3% | 11,207 | 0.00 |
| e_pending_amt_share | 60.3% | 61.3% | 12,762 | 0.314 |
| e_fx_share | 52.8% | 61.2% | 11,176 | 0.00 |
| e_ar_issued | 64.1% | 61.4% | 13,555 | 13,890 |
| e_ap_issued | 64.1% | 61.4% | 13,555 | 19,591 |

Among the 13,554 train invoice-active company-months: open/issued are 100% filled (0 before first invoice); after the delay mask, `e_delay_coll` is 59.6% non-null and `e_delay_paid` 75.9%. Mean AR overdue share 0.69, of which >30 days 0.53.

## What failed / caveats

- Nothing failed the contract smoke (`/tmp/slot3_smoke.py`): prefix, keys, no dups, no look-ahead on June 2025 issued, delay mask, `overdue_30 <= overdue`, weekly 2-company sample.
- No literal `credit_note` document type — ratio is the `note`/`refund` stand-in (amount-weighted; ~35% of invoice-active months have ratio > 0).
- DSO/DPO means are unusable (train mean DSO ~860 months, max ~2e6) because `is_extreme` invoices and tiny monthly issued against a large book. p50 (~1.7–1.8 months) is the real signal; ~4.5% / 7.2% of invoice-active months have DSO/DPO > 24.
- A company can issue invoices in a month it is off the transaction grid (107 issuers in 2025-06). Those invoices still enter later open stock once the company is on the grid.
- Open stock is left-truncated (no pre-2024-09 invoices). Only delay is masked for that; DSO is biased low early.

## Next idea

- Fixed clip (or `log1p`) on DSO/DPO at 24–36 months — same style as the delay clip, not a fit.
- Optional `e_extreme_share` so models can down-weight COMP_0629-style 1e9 plugs without dropping rows.
- Add issuance trend (`e_ar_issued` vs trailing 3m) as the plan asked; out of this slot’s listed columns.
- Y5 can sit on `e_ap_overdue_30` / `e_delay_paid` without touching this family as X.
