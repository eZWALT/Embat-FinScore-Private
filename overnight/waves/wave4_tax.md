# Wave 4 — tax calendar / missed-tax QA (1094dc70)

Long-lived data lane. Same module ≥30 min: write → run → next cut (passes 1–22).
No 0–100. No product/. No parquet rewrite. No new GBM. No `build_targets`.
`ops.py` not edited (store vs raw tax-month agree 100%). Holdout 72 coverage only; rates / AUROC on train. Seed 20260918.

## Files

- `analysis/evaluate/tax_qa.py` (create)
- `analysis/outputs/tax_qa.md`
- `analysis/outputs/tax_month_calendar.png`
- append-only `analysis/experiments/registry.csv`

## What we measured (train)

- Calendar stacked Jan–Dec (2024-09..2026-08): Q-months (Jan/Apr/Jul/Oct) tax-CM **69.2%** vs other **43.4%** (ratio 1.59). Peak Oct 72.4%, trough Aug 38.6%. **mixed_q_peaked**, not pure quarterly (off-quarter is still 43%) and not monthly. SS control is flat 45.8% / 45.5%.
- Cadence: never 116 / monthly 406 / quarterly 183 / irregular 509 of 1,214. Off-quarter tax CM: 69.2% from the monthly book.
- `c_missed_tax` prevalence **8.3%** (1,763 / 21,157). Modal 0 share **91.7% CONFIRM**. acf1=0.05. ρ vs log1p(a_in3)=0.007. P(missed | not tax month)=**17.3%** — not “just not a tax month” (most non-tax months have no usual-tax history). Missed Q-months 2.2% vs other 10.9% — complementary hole.
- Definition trap: usual = tax6≥3. Quarterly tax6 p50=2, share≥3 only 13.1%. Quarterly companies on Q-months miss **0%**. Of 1,763 missed CM: irregular 66.1% / monthly 21.5% / quarterly 12.4% (all quarterly misses are off-Q).
- Singles vs Y3 stressed: `c_missed_tax` **0.511**, `c_tax_month` **0.611** (= `not_tax_month`), size **0.617**, days **0.711** (night replica). Y2 missed 0.514 / Y9 0.506. Honest skip (usual only) 0.566 still < size. Irregular-only 0.501 vs size 0.585. Large-tax month (≥10k) 0.589. **Does not beat size by ≥0.02.**
- Payroll ≠ tax: Spearman tax↔salary 0.317; P(tax|salary)=71.2% vs 38.8%. tax↔a_out6 0.380 — not an outflow leak.
- Dark 470 vs 744: tax-CM **48.7% vs 51.8%** — same bank-book tax rate. Ever-tax 87.0% vs 92.6%. CONFIRM 744/470.
- Q6 lag1 missed 0.504 vs contemporaneous 0.511 — no skill to lead. CLOSE as Q6.
- VAT vs IS: tax txs n=52,237 p50=419; 51.9% <500, 17.9% >10k. Q-month p50 676 vs other 377. **VAT-like**. `tax_refund` = 2 txs / 1 company — not a recovery Y.

## PARK / CLOSE / KEEP

| object | decision |
| --- | --- |
| `c_missed_tax` as Q5 why | **CLOSE** (calendar dummy / complementary hole; Y3 0.511 loses to size 0.617) |
| `c_missed_tax` as a health Y | **PARK** — do not invent a merged Y |
| `c_missed_tax` / `c_tax_month` as Y3 X | **PARK** — size dummy 0.617 ≥ 0.60; tax-month is activity (ρ vs days 0.43) |
| tax calendar as a dummy | **CLOSE** — it is the dummy (mixed Q-peaked; SS flat) |
| Q6 lag1 `c_missed_tax` | **CLOSE** |
| `tax_refund` as recovery Y | **PARK** (n=2) |
| large-tax / IS amount as Y | **PARK** (0.589 < size; do not build) |

## What failed / next

- First shape call said “quarterly” on a 1.59 ratio with 43% off-quarter. Tightened to **mixed_q_peaked**.
- Honest-skip 0.566 looked KEEP vs chance; KEEP rule is beat size by ≥0.02. Still CLOSE.
- `c_missed_tax` cannot flag a quarterly filing-date skip (`tax6≥3`). Do not patch `ops.py` tonight.
- **Next (not this lane):** if someone trims the 44-col starter, drop `c_missed_tax` (calendar hole) and treat `c_tax_month` as the activity twin of `c_n_days_with_tx`. Do not invent a missed-tax Y.
