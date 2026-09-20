# Monetization strategy — Health Sentinel

- **Author:** agent (with Walter), 2026-09-20 09:45
- **What this is:** who pays, for what unit, at what price, in what order. Extends the buyer rationale in `2026-09-19-embat-business-context.md` (Embat premium module) with the bank / multi-company angle.
- **Rule kept:** we sell *explainable and monitorable*, never *predictive*. Nothing below prices a default probability. Numbers marked *illustrative* are assumptions to test, not facts.

## 1. The buyer is never a single company

A single SME will not pay for a score of itself: low willingness to pay, high acquisition cost, and it already sees its own cash. The value appears when someone watches **many** companies and has to decide **which one to look at first**. Three buyers do that, and all three already hold (or can legally receive) the treasury trail the score needs:

| Buyer | Why they pay | What they already have | Where it lands |
|---|---|---|---|
| **Embat** (primary) | Retention and expansion of its 300+ mid-market groups; a differentiator against Kyriba, Agicap, Nomentia; more reasons to open TellMe daily | Bank + ERP + invoice data of every client, entity-scoped permissions, TellMe skills (Silent / Guided / Ask), a Risk Management module to slot into | Premium module + TellMe skill. Embat resells; we license |
| **Multi-entity groups and PE sponsors** (through Embat) | A holding with 5–24 subsidiaries needs a ranked portfolio, not 24 dashboards; PE ops teams need the same across portfolio companies | Already Embat customers (the ICP: 5+ entities, 5+ banks) | The "Grupo" view, group charts, group-vs-groups funnel |
| **Banks and lenders** (secondary, via Embat's bank channel) | EBA loan-origination guidelines require early-warning indicators on the SME book; relationship managers need a reason to call; cross-sell of credit lines and investment loans | The bank sees only its own accounts; Embat sees all banks of the company. With consent, the bank gets the *consolidated* picture | Portfolio monitoring API/feed + consented "Sugerencia comercial" leads |

Later, same product, different feed: factoring / confirming providers (top-customer-quiet is exactly their risk), credit insurers, auditors. Not now.

## 2. Value metric: per monitored legal entity per month

- Matches the unit of analysis (`company_id`) and Embat's own complexity driver (entities, banks), so it scales with the customer's size without a sales conversation.
- Not per user: the CFO, the treasurer and Cobros all need to see the same alert; per-seat pricing would fight adoption.
- Not per alert: volume is low by design (0.53 risk + 0.22 improvement alerts per company-year) and pricing on alerts rewards noise.
- Not a share of credit originated on the Embat side: that would make Embat a credit intermediary. Origination fees exist only in the bank channel, paid by the bank.

## 3. Packaging

| Tier | Who | What is in it (all exists today) | Price (*illustrative*) |
|---|---|---|---|
| **Score** (included in Embat) | Every Embat entity | 0–100 score, trajectory state, confidence flag, top-4 reasons with the € behind each. Silent mode only. | €0 — this is the hook and the data moat; it makes the number a habit |
| **Sentinel** (add-on) | Groups that act on alerts | Five alert kinds with owner + action, Vigilancia posts (3 months, precomputed), control charts (own history, cluster, group), forecast fan, Pregunta chat over score + records, drag-a-period explanation. Guided mode. | €30–60 per entity per month, floor €300 per customer per month |
| **Portfolio** (add-on) | Holdings with 10+ entities, PE ops | Everything above plus group funnel, cluster peer comparison, ranked portfolio, CSV/API export of scores and alerts, monthly board summary | €20–40 per entity per month on top, volume-discounted from 25 entities |
| **Bank feed** (separate contract) | Banks / lenders | Consented per-company feed: score, trajectory, alert onsets, top-customer-quiet, going-dark. No raw transactions. Plus qualified offer leads (size band + "Ofrecer" suggestion) | €3–8 per monitored company per month with a floor; lead fee €150–400 per accepted lead or 5–15 bps of drawn amount, whichever the bank prefers |

Why the free tier is free: percentiles are fitted on Embat's own train population; the more entities scored, the better the reference and the harder it is for a competitor to replicate. The paid tiers sell the **workflow** (owner, action, chat, charts), not the number.

## 4. Illustrative economics (assumptions, to be tested in a pilot)

- Embat base: 300 customers × ~8 entities ≈ 2,400 entities. 30 % attach on Sentinel at €45 → 90 customers × 8 × €45 × 12 ≈ **€390 k ARR**. Portfolio on the top 20 holdings × 15 entities × €30 × 12 ≈ **€110 k**. One bank pilot on 2,000 monitored SMEs at €5 → **€120 k**, plus leads.
- Marginal cost is close to zero: the monthly run is a batch (`product.score.export` + Neon load, minutes), Vigilancia posts are precomputed (no LLM on first paint), Neon at demo scale is < €5/month and at full core facts €20–50/month, chat turns cost cents. Gross margin > 90 %.
- The real cost is the monthly run on a new CSV drop and the Spanish production copy; both are already built.

## 5. Bank channel, done right

- **Consent and scope.** The company (Embat's client) opts in per bank; the bank receives the derived layer (`analytics` / `api`), never `core` records. Permissions stay entity-scoped like TellMe's.
- **Positioning.** *Monitoring and engagement*, not credit decisioning. This keeps it out of IRB model governance and out of the high-risk "creditworthiness of natural persons" AI Act category (these are companies, and the score is not used to grant or deny credit). The EBA Guidelines on loan origination and monitoring (section 8: early warning indicators, watch lists) are the buying trigger for the risk team; the relationship-manager team buys the leads.
- **What we can claim to a bank, with the numbers we have:** the alert with measured lift is *top customer went quiet* (56 % of flagged company-months lose the customer vs 29 % base, lift 1.9, `analysis/monitor/evaluation.md`). Score-fall alerts are descriptive (what moved, how much, in €), not early warnings of failure. Say both sentences in the deck.
- **Opportunity side is the revenue side.** Alerts are two-sided; `score_improvement` and the size band feed "Sugerencia comercial: ofrecer préstamo de inversión / línea de crédito". Banks pay for that lead; the company sees an offer that fits its trajectory. This is the only place a success fee makes sense.

## 6. Deal shape with Embat

Three options, in order of preference for a hackathon team:

1. **Build-in (acqui-hire / paid build).** Embat takes method, evaluation, Spanish copy and the team; the module ships as a TellMe skill. Simplest for Embat's security posture (ISO 27001, SOC 2, one vendor).
2. **License + revenue share.** We keep the scorecard and monitor (`product/score`, `analysis/monitor`), Embat embeds and resells; 20–30 % of module revenue back to us. Needs a data-processing agreement and the monthly run inside Embat's perimeter.
3. **Partner app on Embat's API.** Lowest friction, weakest moat, and our own `analytics` store would have to leave Embat's perimeter. Only as a fallback.

Our leverage: the method is documented and audit-ready (`METHOD.md`, `DATA_CONTRACT.md`, `validation.md`), it runs on data Embat already holds, and it does not overlap Risk Management (counterparty exposure) or forecasting (cash): it adds the *health trajectory of the entity itself* and the routing to an owner.

## 7. Sequence

| When | What | Success signal |
|---|---|---|
| 0–90 days | Pilot inside Embat with 10 multi-entity groups (Score free, Sentinel on). Weekly Vigilancia posts; measure acknowledged and acted alerts | ≥ 50 % of `act` alerts acknowledged within a week; treasurers open Sentinel ≥ 2×/week |
| 3–6 months | Roll Sentinel to the base as a paid add-on; Portfolio to the top holdings. Price test: two entity prices across cohorts | Attach ≥ 25 %; NRR uplift visible on Sentinel cohort |
| 6–12 months | One bank pilot through Embat's partner channel with consented feed + leads; add supplier-side top-counterparty alert (untested today) | Bank renews; lead acceptance ≥ 10 % |
| Ongoing | Same run on each new CSV drop; refit reference once a year, never on holdout | No manual steps in the monthly run |

## 8. KPIs to price against

Attach rate; entities monitored; alerts acknowledged / acted (by owner); offers shown → accepted; chat turns per active user; churn of Sentinel customers vs non-Sentinel; time from alert onset to action. Not: AUROC on outcomes (we do not sell prediction).

## 9. Risks

| Risk | Mitigation |
|---|---|
| Embat builds it internally | Speed and completeness: method, copy, evaluation, and the Spanish wording are done; offer option 1 |
| Overlap with Risk Management / TellMe forecasting | Position as entity health + owner routing; do not sell counterparty exposure or cash forecasts |
| False-alarm fatigue kills perceived value | Volume is 0.53 risk alerts per company-year; persistence 3-of-4 and onset-only are already in; report false-alarm rate on train |
| Bank reads the score as a credit model | Contract language: monitoring and engagement only; no raw records; two-sentence claim above |
| Data rights for the bank feed | Per-bank consent by the company, revocable; derived data only |
| Hidden-test surprises | No claims on the hidden test; `score_new` handles short trails and no-invoice companies with the confidence flag |

## Still unknown

Embat's real module ACV and attach rates (public pricing is "contact us"); whether Embat's bank partner channel already carries data products; whether banks would accept the consented derived feed under their vendor rules; supplier-side alert not tested.
