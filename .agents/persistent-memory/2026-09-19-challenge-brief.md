# HackSpain 2026 — X Ray (Embat) challenge brief

Source: <https://claude.ai/artifact/8N8Q7QMjprCUWxGAiJaWoP?sk=5wYke4E8ukAw6afs6TrG1g>  
Captured: 2026-09-19. Treat the live artifact as canonical if it changes.

## Brief

Build from 24 months of financial activity a company health score that identifies financial health and trajectory. The goal is not bankruptcy prediction: read financial behaviour in both directions, before it becomes obvious.

The score is the engine. The deliverable must also be a product, service or tool that someone would pay for.

## Questions the system must answer, per company and per month

1. Who is healthy — identify exceptionally solid companies, not only troubled ones.
2. Who is improving — a company moving from 45 to 65 can be a stronger future bet despite mediocre current levels.
3. Who is starting to deteriorate — a company moving from 82 to 68 may still look healthy today but its behaviour has changed.
4. Is it a dip or a decline — distinguish a one-off weak cash month from structural deterioration.
5. Why did it change — state which signal moved and when.
6. When was it visible — measure how many months before the change was detected.

## Required system capabilities

1. Read the trail: bank movements, issued/received invoices, payment behaviour, financing cost and debt balances.
2. Produce a score that captures trajectory rather than only the latest snapshot, and generalizes to unseen companies.
3. Explain the score and its month-on-month change.
4. Build a sellable product on top of the score and identify its buyer.

## Dataset

- 1,286 synthetic companies in 250 business groups.
- 24 months: September 2024 to September 2026.
- Nine CSV files; no real company, account or personal data.
- `groups.csv`: one business group per row; a group has 1–24 companies (median 2).
- `companies.csv`: company ID, group, country, currency, ERP and onboarding date.
- `banking_products.csv`: current, card, POS, savings, investment and expense-platform accounts.
- `debt_products.csv`: loans, leasing, credit lines, mortgages, renting, factoring, confirming and guarantees.
- `debt_schedule_config.csv`: amortization schedule parameters.
- `transactions.csv`: 24 months of bank activity.
- `invoices.csv`: issued/received ERP invoices, due/paid dates, outstanding amount, status and counterparty.
- `balances.csv`: account/product balances as of 1 September 2026.
- `data_dictionary.md`: field definitions.

## Delivery requirements

| Requirement | Status |
| --- | --- |
| Predict the hidden test companies for the leaderboard | Required |
| Detect improvement and deterioration | Required |
| Reflect trajectory, not only the latest snapshot | Required |
| Explain score and changes for any company | Required |
| Product built on the score | Required |
| Identified buyer and buyer rationale | Required |
| Navigable demo | Required |
| Measured anticipation: months of early detection | Bonus |
| Proactive monitor that alerts on material moves | Bonus |

## Evaluation rubric

Three equally weighted blocks:

### Accuracy

- Generalization: works on unseen companies.
- Trajectory: captures direction, not only current level.
- Both sides: recognizes improvement and deterioration.

### Timeliness

- Anticipation: detects change before it is obvious; quantify months ahead.
- Stability: separates a temporary dip from a real deterioration.
- Monitor: extra credit when it alerts proactively.

### Value

- Product: goes beyond a number.
- Buyer: clear payer and economic rationale; Embat, the data provider, is the obvious candidate.
- Explanation: score is understandable.
- Craft: polished, usable demo.

## Suggested product directions in the brief

- Credit marketplace.
- Financial insurance / trade-credit coverage with premiums adapting to the score.
- Working-capital financing with dynamically recalculated limits.
- Recommendation agent that suggests weekly actions: renegotiate a supplier, refinance debt, improve collections.
- Sector-level investment signal.
- Dynamic pricing, supplier scoring, a shareable trust seal or a terms comparator.

## Strategic implication for this project

An integrated **Health Sentinel** fits strongly if it is built on a validated, explainable, bidirectional trajectory score. It should notify the right financial owner about a material, persistent change; explain the drivers; propose a specific action; and measure lead time. It covers the monitoring bonus and transforms the score into a sellable Embat add-on rather than a standalone dashboard.
