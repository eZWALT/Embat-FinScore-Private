# 2026-09-18 — initial context

- **Author:** agent (bootstrap), for the three hackathon teammates
- **When:** 2026-09-18 evening, Madrid / HackSpain day 1
- **Repo:** `Embat-FinScore-Private` (this tree). Work here.

## Repos

| Repo | Visibility | Role |
|------|------------|------|
| `eZWALT/Embat-FinScore-Private` | private | **Working repo.** Build Aura here so the weekend work is not copied off the public tree. |
| `eZWALT/Embat-FinScore` | public | Late-publish / judges face. Keep in sync **at the end**, not commit-by-commit. |

Do not develop new features on the public repo until the team decides to publish.

## Event and challenge

- Event: [HackSpain 2026](https://hackspain.com/), 18–20 September 2026, Madrid.
- Track: Embat **X Ray**. Working name in this repo: **Aura**.
- Official brief: https://claude.ai/artifact/8N8Q7QMjprCUWxGAiJaWoP?sk=5wYke4E8ukAw6afs6TrG1g
- Track question: *¿Puede el dinero decir cómo está una empresa?*

Embat gives a 24-month treasury trail. Build a **financial-health score** (the engine) and a **sellable product on top**. Not a bankruptcy classifier.

Brief example (not our output): Northbrook Foods 45→65 vs Velasco Industrial 82→68. Same month-24 still can hide who is the better risk.

Six questions (per company, per month): who is healthy; who is improving; who is turning; dip vs fall; why it changed; how many months earlier it was visible.

Four jobs: read the trail; score **trajectory** (not only last month); explain the number; build something someone would pay for. Hint for buyer: the company that already hands you the data.

Judging (brief): equal weight on being right, being on time, and being worth something. Simple model + clear product beats a clever number with nothing on top.

Hidden test: **60–80 companies the system must never train on** (leaderboard).

Delivery required: hidden-test scores; both directions; trajectory; explanation; product on top; identified buyer; navigable demo (laptop-only notebook does not count). Bonus: measured lead time; a monitor that alerts unprompted.

Possible products in the brief (none chosen): credit marketplace, financial insurance, working-capital line, recommendation agent, sector signal, other. Last brief heading **Qué ponemos nosotros** was not captured from the iframe.

## Dataset

- Zip: https://f5xe6kyx7jpysotw.public.blob.vercel-storage.com/output_hackspain_data.zip (~187 MB)
- Synthetic SME treasury, 2024-09-01 → 2026-09-01. No real companies/accounts/people.
- 250 groups, 1,286 companies.
- Files: `groups` 250 · `companies` 1,286 · `banking_products` 5,987 · `debt_products` 2,239 · `debt_schedule_config` 87 · `transactions` 2,556,437 · `invoices` 897,894 · `balances` 7,996 (2026-09-01).
- Join: `company_id`. Products: `product_id`. Counterparties share one ID space.
- Docs: `data/data_dictionary.md` (committed). CSVs/zip are gitignored.
- A local extract already exists on this machine at `../Embat-FinScore/data/raw/output/` (not copied here). Re-download or copy into `data/raw/` on this repo when needed.

## Files in this repo at bootstrap

```text
AGENTS.md                          # pointers only
.agents/persistent-memory/         # this journal
README.md                          # Aura + short map
assets/finscore-banner.png         # provisional banner
data/data_dictionary.md
data/README.md
analysis/README.md                 # six-question checklist, no results
product/                           # uv + Docker scaffold, no score yet
LICENSE                            # MIT
```

`product` prints name/version only. No model, no demo, no buyer, no train/hidden split.

## How to journal

Three people will use agents on this repo. After meaningful work, add
`.agents/persistent-memory/YYYY-MM-DD-HHmm-<slug>.md` with author + decisions.
Do not dump live status into `AGENTS.md`.
