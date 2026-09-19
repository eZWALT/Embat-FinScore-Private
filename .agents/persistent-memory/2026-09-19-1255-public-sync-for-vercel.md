# 2026-09-19-1255 — public repo sync for Vercel

- **Author:** Walter (with Cursor agent)
- **When:** 2026-09-19 ~12:55 CEST

## Decision

The hosted demo has to go on the **public** GitHub repo (`eZWALT/Embat-FinScore`). The private sibling cannot be used as the Vercel source. This is the late publish: copy the working tree onto the public face so Ruben can deploy.

## What is copied

Tracked files from `Embat-FinScore-Private` at `26e6037` plus the 12:50 product-definition note, except:

- The raw CSVs (`data/transactions.csv` 472 MB, `data/invoices.csv` 173 MB, and the other dump files). They are tracked on the private repo by accident of history; public `.gitignore` keeps them out. Dataset stays the official zip.
- `LICENSE` (public stays Apache 2.0; private is MIT).
- Private README identity ("private working repo / nothing implemented"). Public README is rewritten for the current product.

Javier's **full export bundle** is not in either git (53 MB, generated). The committed 12-company `product/score/sample_bundle/` is what the first Vercel build reads. He is still producing the final data; a later sync can swap `SCORE_BUNDLE_DIR` or replace the sample.

## Vercel

Root Directory = `product/web`. The app reads `../score/sample_bundle` (and two fallbacks). No secrets, no env required for the sample.

## Still unknown

- Whether Vercel includes files outside `product/web` on this account; the path fallbacks cover the usual monorepo case.
- When the full bundle lands and how it is hosted (Blob / `BUNDLE_URL`).
