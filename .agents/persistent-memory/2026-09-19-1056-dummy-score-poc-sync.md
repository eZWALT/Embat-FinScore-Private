# 2026-09-19-1056 — agentic context + dummy score + POC reconcile

- **Author:** agent (Aura)
- **When:** 2026-09-19 ~10:56 CEST
- **Repo:** Embat-FinScore-Private (`main`)

Walter's same-morning verdict (`2026-09-19-1100-product-verdict-and-poc.md`)
stands: analysis stops for product work; 01+02 merged; Ruben deploys
`product/`; Walter iterates `poc/`; Javi owns the score. This note only
adds the score side and the bind.

## What changed

Brought the working tree in line with the plan and with the Streamlit
shell that was sitting untracked.

- **Step 2 started (not done):** v0 dummy FICO-like card in `product/score/`.
  Journal of the card itself: `2026-09-19-1048-dummy-fico-v0.md`.
  Latest-month train `score_3m` p10/p50/p90 = 17/47/80.
- **POC:** `poc/` (Streamlit Overview / Sentinel / Portfolio) is now part
  of the tree. Portfolio reads `product/score/outputs/monthly_scores.parquet`
  when present. Sentinel is still an empty chat.
- **AGENTS.md** pointers only: v0 score command, `poc/`. No live status
  in that file. Step 2 is not marked done (no Y validation, no
  `score_new`, no € reasons).

Commit / pull / push on `origin/main` after this note. Public sibling
`../Embat-FinScore` is left alone (late sync).

## Decisions

- Dummy score is the number the POC may show. Weights a priori; holdout
  out of the percentile fit; going-dark cap stays.
- Score ↔ product contract for v0 is the column table in
  `product/score/README.md` (no € reasons, no owner/action yet).
- Do not treat `poc/` as the hosted demo (step 5). Ruben's Vercel + DB
  path is still `product/web/`.
- Do not commit score/store parquets (`*.parquet` gitignored).

## Still unknown

- Group-fold AUROC of the dummy card vs the eight accepted Ys.
- Whether no-invoice companies should be score-capped in the UI.
- Submission `score_new(csv_folder)` format.
