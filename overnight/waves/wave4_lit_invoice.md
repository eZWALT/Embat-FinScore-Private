# Wave 4 — invoice literature (as-of ~07:35)

Owner: lit invoice / trade-credit / concentration / fee-as-label / Q4–Q6.
Deliverable: `analysis/outputs/lit_invoice.md` + `lit_invoice_cluster.csv`.
Sibling cash-flow scorecards → `analysis/outputs/lit_cashflow.md`. Do not grow TURNOVER **0.720**. Hidden 72 never fit.

## Cluster

12 ultra papers, all URLs fetched. Starters verified (BdF 227/8, Hirshleifer w25553, Pérez-Salazar 2026, FinRegLab 2025 fee slice). Beyond: Jacobson 2015, Boissay–Gropp 2013, Irvine 2016, Campello–Gao 2017, **Ellingsen–Jacobson–von Schedvin 2016 (52m contracts: AP = volume not days)**, García-Appendini 2013 (+0.5 pp AR/sales per 1 SD cash), Bitetto et al. 2024, **Bureau–Duquerroy–Vinas BdF WP 851 / RCFS 2024**. Companions: Amberg 2021 (full PDF), Costello 2020 / Lian 2017 / Barrot 2016 (abstract_only).

Pérez-Salazar is the only **CONTRADICT** (synthetic supplier-HHI = fragility; our Y5 tail is protective 2.7% vs 8.6%). CN ratio is a **literature gap**. Everything else **SAME** vs the night locks.

## Do not do

- Grow TURNOVER. Rewrite `gbm_core.py`. Merge parquet. `build_targets`. Fit hidden 72.
- Touch leftover QA files (`op_out_qa`, `fin_cost_qa`, `ap_overdue_qa`, `n_accounts_qa`, `util_snap_qa`, `cust_lost_qa`, …).
- Put DSO / delay / CN / AP overdue / HHI / `d_cust_lost` on a card.
- Merge Family M (Y9 is the fee label). Merge J. Fill dark 470 with 0.
- Bankruptcy classifiers. Collections-ops “this invoice will be late.”

## Three pitches the parent should absorb (in order)

1. **Wave A — top-1 PastDue% leftover after issued_lag1+CN** (`top1_pastdue_qa`). Hirshleifer’s object, not firm-level delay. Acceptance leftover ≥0.58, ρ vs `e_delay_coll` <0.80, dark 470 NaN. If it dies, the delay footnote stays.
2. **Wave B — AR issued to last month’s top-1 after firm issued_lag1, on Y7 rows** (`issued_top1_qa`). Jacobson demand-shrinkage / Irvine major-customer / Amberg issued −1 pp. **Not** `d_cust_lost` (Y3 leftover 0.522 already died; n_cust twin). Never D-on-Y7.
3. **Wave D — Y5 65% leftover as net TC × activity** (`y5_net_tc_qa`). Bureau: 1 SD (AP−AR) → +10% payment-default PD **only in shock months**. Diagnostic only — **never E as Y5 X**. If leftover dies, 65% stays the sentence.

Wave C (CN note vs refund / top-1 notes) is the footnote cut if A/B/D need a fourth seat. No PD paper exists; Boissay 16.2% vs 2.1% is the only number.

## Night numbers this wave must not move

Y7 0.720 / 0.712. Y3 0.762 / 0.752. days 0.711. size 0.617. Q6 issued_lag1 0.626 / days_lag1 0.684 / ss_lag1 0.631. Y4 tail 0.605. Y5 leftover 65%. CN 0.597. delay 0.581. DSO drop 0.474 / 0.452.
