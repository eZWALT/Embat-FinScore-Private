# 2026-09-19 11:54 — Supabase schema review

- **Author:** Codex
- **When:** 2026-09-19 11:54 CEST

## What was reviewed

- `data/data_dictionary.md`, the eight CSV declarations and `analysis/build_db.py` / `analysis/clean_db.py`.
- `product/score/DATA_CONTRACT.md`, its TypeScript contract and the sample score bundle.
- The new web repository boundary in `product/web/src/lib/data/`.
- Current Supabase import, database-size and Data API guidance.

## Findings and decisions

- The eight Git LFS objects total about 646 MB before Postgres/index overhead. A full raw import will exceed Supabase Free's current 500 MB database-size limit; use a paid project or upload only the product/analytics layer.
- In this checkout the CSV paths are Git LFS pointer files, not the actual CSV contents. An import needs the LFS objects or another local source folder.
- Use four boundaries: private `ingest` staging, private cleaned `core`, versioned `analytics`, and a narrow read-only `api` schema for the web/Data API.
- Unify `banking_products` and `debt_products` behind a `products` supertable. Otherwise `transactions.product_id` and `balances.product_id` cannot have one valid foreign key to the current two-table union.
- Preserve the cleaning policy and `dq_log`; nullable/duplicate transaction and invoice identifiers require surrogate primary keys plus partial unique indexes, not naive natural primary keys.
- Counterparties have no master data and do not map to `company_id`; they can be synthesized as counterparties but cannot be treated as scored companies/customers/suppliers yet.
- Persist computed score data by immutable run (`schema_version`, `scorecard_version`, generated time, source hashes, spec/disclaimer) and normalized company-month/category/item/reason rows. Derive latest-company and group summaries as views instead of duplicating the JSON bundle indexes.
- Expose only purpose-built API views/RPCs with explicit grants and RLS. Keep raw narratives and internal analytics out of the public Data API.
- The existing `ScoreRepository` boundary supports replacing the local-bundle adapter with a Supabase adapter without changing the dashboard service.

## Still unknown

- Whether the first hosted version needs all raw transactions/invoices or only the computed score/product tables.
- Which connected Supabase organization and EU region to use, and whether the user accepts the live project cost.
- Where the real LFS CSV objects/full generated bundle are available to this checkout.
