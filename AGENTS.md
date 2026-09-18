# AGENTS

Private working repo for **HackSpain 2026 · X Ray (Embat)**.
Public face (late sync only): `../Embat-FinScore`.

Build in this repo. Do not grow this file with live status.

FICO-like **company health score**. Four goals: signals → 0–100 index → explainability → company-facing web + LLM. Details: `.agents/persistent-memory/2026-09-18-2020-four-goals.md`.

## Read first

1. This file (pointers only).
2. `.agents/persistent-memory/` — journal. Start with `2026-09-18-initial-context.md`, then newer dated files.
3. `data/data_dictionary.md` when touching data.

## Pointers

| What | Where |
|------|--------|
| Challenge brief | https://claude.ai/artifact/8N8Q7QMjprCUWxGAiJaWoP?sk=5wYke4E8ukAw6afs6TrG1g |
| Track dataset zip | https://f5xe6kyx7jpysotw.public.blob.vercel-storage.com/output_hackspain_data.zip |
| Field dictionary | `data/data_dictionary.md` |
| Dataset notes | `data/README.md` |
| 1. Signals | `analysis/` |
| 2–3. Score 0–100 + explain | `product/score/` |
| 4. Web + LLM (company user) | `product/web/` |
| App folder | `product/README.md` — Docker/runtime still empty |
| Public sibling | `../Embat-FinScore` (GitHub: `eZWALT/Embat-FinScore`) |

## Memory (three teammates)

After meaningful work, **add a new file** (do not rewrite history in old ones):

```text
.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md
```

Each entry: author, timestamp, what changed, decisions, still-unknown.
Put facts that can rot in the journal, not here.
