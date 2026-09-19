# Wave 4 — missing-CP leftover

- **When:** 2026-09-19T04:16:17+02:00
- **Agent:** `c91e4b2a`
- **Files:** `analysis/evaluate/missing_cp_qa.py`, `analysis/outputs/missing_cp_qa.md`, `analysis/outputs/missing_cp_vs_uncat.png`, append-only `analysis/experiments/registry.csv`.
- **Columns:** in-memory `miss_cp_share` / `miss_cp_amt` / `miss_mapped_share` (not written to parquet). Store `d_tx_cp_share` replica only.
- **Train coverage:** cm 95.8%; txs miss 90.1% / |amt| 96.4%.
- **Verdict:** X **CLOSE**; Y **PARK**; Q5 **CLOSE**; Q6 **CLOSE**.
- **Numbers:** uncat 2×2 miss-among-uncat 91.8% (mapped still miss 89.6%); ρ vs uncat 0.131; ρ vs d_tx -0.947; Y3 miss 0.554 vs size 0.617 vs days 0.711; leftover after both 0.509.
- **What failed:** d_tx_cp_share twin ρ=-0.947 — CLOSE as (b); leftover after uncat/dtx dies — CLOSE as twin; Y5 fold-3 pile again — CLOSE as (e); invoiced-744 still d_tx twin ρ=-0.919 — not only the 470; Missing-CP txs: uncategorized 25.0% / transfer 6.6% / salary 1.8% / tax 2.2% / collection 21.8% / payment 13.2%. Not just uncat+transfer — other mapped cats also miss CP.; Same-month invoice CP fill mean 0.999 vs tx named share 0.252 (ρ -0.014; n=11,015). Hole is bank-book tagging, not the invoice book (Family D HHI lives on invoices). Do not invent a COMP_* ↔ COUNTERPARTY_* map.
- **Next idea:** parent decides any later-D merge. Do not put on the 15-col card. Y7 never D.
- **Did not:** product/, 0–100, parquet rewrite, `build_targets`, counterparties.py, uncat_qa.py, y5_why.py, parent journal, commit.
