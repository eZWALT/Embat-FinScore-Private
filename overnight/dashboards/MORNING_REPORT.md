# Embat X Ray — what the money trail told us overnight

- **As of:** Saturday 19 September 2026, 09:10 CEST
- **Track:** HackSpain 2026 · Embat X Ray
- **Audience:** the team and anyone we pitch to. No prior knowledge of the code assumed.
- **Scope of the night:** find signals, build honest targets, explain them. **No 0–100 score yet, no product, no web.** Those come after the team agrees on this evidence.
- **Companion views:** Cursor canvas `xray-morning-report` and `overnight/dashboards/morning.html`

---

## 1. The question, in one paragraph

Embat asked: *can the money say how a company is doing?* We have 24 months of bank movements, invoices, debts and balances for 1,286 small and medium companies (synthetic, but shaped like real treasury data). The brief gives two pictures: Northbrook Foods, whose health goes **45 → 65** (a recovery), and Velasco Industrial, whose health goes **82 → 68** (a deterioration). Both may look similar in the last month. So the object is the **trajectory**, not the snapshot, and every answer has to come with a reason and with how early it was visible.

Judges will score companies we have never seen. So we froze 72 companies at the start of the night and never used them to fit anything. Every number below comes from the other 1,214 companies, measured with a method that always tests on companies the model did not learn from.

---

## 2. How to read the numbers

Three ideas repeat through the report. Read this box once.

**Ranking score (AUROC).** Every model or signal is graded on a 0.5–1.0 scale. **0.5** means a coin flip. **1.0** means it separates perfectly. A practical way to read 0.76: take one company-month that later recovered and one that did not; the model puts the recovering one first 76 times out of 100. In credit scoring, 0.70–0.80 on a hard target is a respectable model; anything under 0.60 is close to guessing.

**Two bars every signal must clear.**
- **The size bar (0.617).** "How big is the company" (three months of operating inflows) already ranks recovery at 0.617. Any signal that does not beat that by at least 0.02 is just measuring size.
- **The activity bar (0.711).** "How many distinct days this month had any bank movement" already ranks recovery at 0.711. Most things that look predictive are secretly this activity clock. So for each candidate signal we first strip out what the activity clock explains and measure what is **left over**. If the leftover is under **0.55**, the signal adds nothing new.

**Twins.** If a candidate moves almost identically to something we already keep (correlation above 0.80), it is the same information with a different name. We do not count it twice.

Two more facts about the data:
- **470 of the 1,214 training companies have no invoices** (no ERP connected). We call them "dark". Any reading that needs invoices is silent for them, and we leave that silence as a blank, not a zero.
- **The hidden test is small.** 72 companies is enough to check coverage, not to prove a rare event. We quote the training cross-validation, never the hidden-set number.

---

## 3. The story of the night

**We threw away a good-looking number.** The first overnight run reported a ranking score of 0.86. It was built by predicting a company's future cash from its own cash columns — the target and the input were the same thing. That measures persistence, not health. We erased it and set one rule for the night: **the outcome we predict may never be built from the columns we allow the model to see.**

**We built the trail.** Every company, every month, 118 measurements from all eight tables: cash flows, reconstructed balances, operational regularity, customer and supplier mix, invoicing behaviour, debt and financing, bank products, and group siblings. 22,230 company-months.

**We defined honest outcomes.** Nine yes/no events plus six continuous paths, each with a fixed rule taken from the credit literature, none tuned on our data. The ones that matter tonight:

| Outcome | Plain meaning | How often it happens |
|---------|---------------|----------------------|
| **Recovery** | The company is in cash stress now (negative cash, or under one month of runway) and, inside the next six months, enjoys three consecutive months with at least three months of runway. This is the 45 → 65 picture. | 7.1% of stressed months |
| **Losing the top customer** | The customer who billed the most in the last quarter bills nothing in the next quarter. This is a dip-vs-fall signal from the invoice book. | 28.8% of company-months with an invoice book |
| Staying underwater | Cash negative in two of the next three months. | 7.3% |
| Debt service doubling | Debt repayments relative to inflows double. | 13.9% |
| Paying suppliers late | Own payables over 30 days late above the company's usual level. | 8.5% |
| Fee pressure | Bank fees and interest spike relative to the company's own history. | 14.1% / 19.1% |

**Two readings survived every test.** Everything else we tried was either size in disguise, a twin of the activity clock, or noise. That is not a failure. It is the finding: a treasury trail carries two clean health messages, and the rest is bookkeeping style.

---

## 4. The two readings

### Reading 1 — "Quiet stress recovers": the 45 → 65 engine

**What it predicts.** Among companies already in cash stress, which ones will be back to three months of runway within half a year.

**How well.** Ranking score **0.762** with the full 118-measurement model (five-fold, always tested on unseen company groups). A stripped-down version with only fifteen measurements scores **0.752**. The activity clock alone scores 0.711; company size alone 0.617.

**What it says, in words.** Among stressed months, the ones that recover are the **quiet** ones:

| What we will say to a company | Behind it | Strength after removing the activity clock |
|-------------------------------|-----------|----------------------------------------------|
| "No social-security payment was booked this month" | a yes/no flag on the transaction category | **0.635** |
| "No payroll payment was booked this month" | a yes/no flag on the transaction category | **0.603** |
| "Fewer distinct days with bank movements this month" | count of booking dates | **0.711** (this is the activity clock itself) |
| "Fewer movement days last month" | same, one month earlier | **0.684** |
| "No social-security payment last month" | same flag, one month earlier | **0.631** |

Every arrow points the same way: **less payroll-type outflow and less activity → more likely to recover**. That is de-escalation at the bottom, not "bigger firms bounce back". It is also, we must say plainly, partly mean reversion: a quiet stressed month can be the floor rather than a healthy choice. Our own baseline showed that naive "inflows fell 40%" alerts mean-revert (they score *below* a coin flip), so the recovery engine is built with that honesty in mind.

For companies with no invoices, and for the smallest third of companies, say **movement days + social security**; the payroll flag stops adding information there.

**What we will not say.** Historical feature-importance plots also rank "number of transactions", "operating inflows", "transfers", "days sales outstanding", "debt-service ratio" and "outflow volatility". Each of those failed one of the three tests: transaction count is a twin of movement days; operating inflows is company size; transfers and volatility are stable company traits, not this month's news; DSO and debt-service ratio add nothing once activity is removed. Reasons given to a company must be factors that actually carry weight, so those names stay off the list.

### Reading 2 — "The big customer goes quiet": the dip-vs-fall engine

**What it predicts.** Whether the company's largest customer of the last quarter will bill nothing in the next quarter.

**How well.** Ranking score **0.720** with five measurements; **0.712** with a six-measurement version built for explanation. Company size scores 0.469 here — this reading is not about size at all.

**What it says.** The lead is **last month's invoicing volume** and how jumpy that volume has been, plus the share of credit notes and the financing-cost ratio three months back. The classic metric people reach for — **days sales outstanding (how long customers take to pay)** — is **not** on the card. It ranked first in a naive model and was *worse than a coin flip* on the fifth of companies with the shortest collection times. Once we removed it, the weakest fold improved from 0.556 to 0.680.

Two footnotes we can quote but do not add to the card: **credit-note share** (leftover 0.597: companies that issue many corrective invoices) and **how late customers actually pay** (leftover 0.581). Both describe *who the company is* more than *what changed this month*.

A late result from this morning: **how much the top customer was billed this month** is a strong leftover signal (0.789) — because the event is mostly *thinning to zero*: when last quarter's top customer received no invoice this month, 58% are gone next quarter versus 8% when they were still billed. That is a reason we can show a company. It does not replace the card, because it is nearly the outcome itself one month early.

---

## 5. The six questions, answered

| # | Brief question | Our answer this morning | How sure |
|---|----------------|-------------------------|----------|
| 1 | **Who is healthy?** | Months of cash runway on the latest balance (end-of-month cash ÷ average monthly outflow). The median company holds **1.08 months ≈ 32 days** of runway — the same order as JPMorgan Institute's 27-day US small-business median. | The last month *is* the photograph (correlation 0.97 with the snapshot); it persists three months out (0.85). Company metadata (country, ERP flag, account-opening date) adds nothing. |
| 2 | **Who is improving?** | Only a thin inflow forecast three months out beats the historical mean. Cash itself is best forecast by its last value. | Honest but thin. We do not claim to reconstruct the cash path with a model. |
| 3 | **Who is turning (45 → 65)?** | Reading 1: quiet stressed months recover. | **0.762 / 0.752** vs activity 0.711 and size 0.617. |
| 4 | **Dip or fall?** | Reading 2: thin, jumpy invoicing precedes losing the top customer. The mirror image (82 → 68, cash going and staying negative) is real as an event but **we could not explain it** without using the cash columns themselves; 82% of those companies are already underwater when the alert would fire. | **0.720 / 0.712** for the customer-loss side. Deterioration side: honest failure. |
| 5 | **Why did it change?** | For recovery: payroll-type drains stopped and activity fell. For customer loss: last month's invoicing thinned. Concentration matters only at the extreme — a company whose customers are essentially one buyer (concentration above 0.975) has a 22% chance of a debt-service shock vs 12% otherwise. | Reasons are the five sentences in Reading 1 plus the invoicing lead. Everything else we tested as a "why" is size, style, or a twin. |
| 6 | **How many months earlier?** | **One month.** Last month's invoicing volume (0.626), last month's movement days (0.684) and last month's social-security flag (0.631) carry the lead. Three-month leads only exist on companies with long histories, which the hidden test mostly lacks (only 4% of hidden companies have a full 24 months). | Solid one-month claim. No longer claim on the hidden set. |

---

## 6. What the literature says about our findings

Two long-lived agents spent the morning reading 23 papers (URLs fetched, no invented citations) and mapping each night result to them.

| Our result | What the research says | Verdict |
|------------|------------------------|---------|
| Months of runway on the latest balance is the health photograph | JPMorgan Chase Institute 2016: "cash buffer days" — median 27 for US small business. A three-month average (as in FinRegLab / Hair 2025 loan-application scoring) is the same picture, smoother (correlation 0.95). | **Same** |
| Quiet stressed months recover | Bank-statement scoring papers (FinRegLab 2025, Yao 2017, Ng 2025, Norden & Weber 2010) all predict *default*, and there "more credits" is good. Nobody scores recovery from stress. | **New** |
| Strip the activity clock before crediting a signal | Published scorecards quote raw ranking scores. Ng 2025 reports 0.81–0.85 without this check. | **New referee** |
| Invoicing volume, not DSO, predicts customer loss | Banque de France 2019: "the increase in days sales outstanding is not related to the probability of default"; Ellingsen et al. on 52 million contracts: weaker buyers owe more because they *buy more*, not because they stretch days. | **Same** |
| How late customers pay is a footnote, not a driver | Hirshleifer et al. 2019: buyer past-due share predicts *buyer* default six months out; only >30 days late moves risk (BdF). | **Same, as footnote** |
| Concentrated supplier base is *protective* here (2.7% vs 8.6% late-payment risk) | Pérez-Salazar 2026 (synthetic data): supplier concentration = fragility. | **Contradicts** — their data is simulated; ours points the other way |
| Outflow volatility is a company trait, not a month signal | Lundmark et al. 2020: volatile new ventures exit; survivors do not become less volatile. | **Same** |
| Credit-note share as a signal | No paper measures it against default. | **Gap** |
| Credit-line utilisation as an early warning | Norden & Weber: usage rises ~12 months before default. | **Hole in our data** — we only have one month of utilisation |

---

## 7. What we tested and set aside, and why that is the point

We ran roughly forty "leftover" audits overnight, one per candidate signal. Each asked: after removing size and the activity clock, is anything left, and is it new? Groups of answers:

- **Twins of the activity clock.** Transaction count, gap regularity, recency of the last movement, change in movement days. Same information as "movement days"; not counted twice.
- **Company size in disguise.** Operating inflows, operating outflows, number of suppliers, number of customers. They rank recovery only because bigger companies recover more.
- **Stable traits, not monthly news.** Transfers between own accounts, outflow volatility, share of uncategorised transactions, pending-invoice share. They tell you *what kind of company* this is (very stable across months), not *what changed*.
- **Connection artefacts.** Number of bank accounts, product flags (cards, savings, factoring lines), account-opening dates. These only ever go up as companies connect more accounts; they date the connection, not the health.
- **Invoice-side rewrites.** Days sales / payables outstanding, open receivables, open payables, overdue payables, this month's issuance, debt-service amount, finance-cost amount. Each is either a rewrite of last month's invoicing or a twin of a ratio we already dropped.
- **Snapshots we cannot use.** Credit-line utilisation and "outstanding above granted" exist only for the last month (1.6% coverage). A utilisation early warning — the strongest one in the banking literature — is a hole in this dataset.

The night also parked several *models*: a per-group version of the recovery model (worse than global, and the hidden test is new groups), tree models for debt-service shock, late payment and fee pressure (each loses to a single hand-picked signal), and weekly time-series forecasts (lose to the historical mean).

---

## 8. What we will not claim

- A 0–100 index or pillar weights. That needs a team decision on top of this evidence.
- A ranking score on the hidden 72 companies as proof of anything. Too few events.
- A lead time longer than one month on the hidden set.
- That we can explain the 82 → 68 deterioration without using the cash columns.
- Reasons drawn from importance plots that failed the leftover test.
- Bankruptcy or default language. This is a health reading, not a default model.

---

## 9. What this sets up

**Still this morning.** Two literature agents keep running until about 10:00 (Y5 late-payment leftover; confirming the dataset has no NSF/overdraft token). No further model fitting.

**When the team opens the score and product goals:**
- The 0–100 should be a **small, monotone scorecard** on the five recovery reasons (social security, payroll, movement days, and their one-month lags) plus a separate customer-loss reading from last month's invoicing. Not a 118-column tree. Not an average of legacy ratios.
- Reason codes shown to a company are the plain sentences in Reading 1 and the invoicing lead in Reading 2.
- Then: hidden-test scores, both directions, a trajectory view, and the company-facing web + assistant.

---

## 10. Glossary — variable names that appear in the technical files

| Name in code | Plain meaning |
|--------------|---------------|
| `b_runway` | Months of cash runway: reconstructed end-of-month cash ÷ average monthly outflow (last 3 months) |
| `b_liq` | Reconstructed end-of-month cash |
| `c_n_days_with_tx` | Number of distinct days in the month with any bank movement ("the activity clock") |
| `c_ss_month` | 1 if a social-security payment was booked this month |
| `c_salary_month` | 1 if a payroll payment was booked this month |
| `*_lag1`, `*_lag3` | The same measurement one / three months earlier |
| `log1p(a_in3)` | Company size proxy: operating inflows over the last three months |
| `e_ar_issued_lag1` | Last month's invoicing volume to customers |
| `e_credit_note_ratio` | Share of this month's issuance that is credit notes / corrective invoices |
| `e_delay_coll` | How late customers pay, in days past due, amount-weighted, last three months |
| `e_dso_proxy` | Days sales outstanding: open receivables ÷ this month's issuance (months) |
| `f_fc_r_lag3` | Financing cost as a share of inflows, three months ago |
| `d_cust_hhi` | Customer concentration (1 = a single customer) |
| Y3 `y3_recover_cash_6m` | The recovery outcome (Reading 1) |
| Y7 `y7_top1_lost` | The top-customer-lost outcome (Reading 2) |
| TURNOVER | The five-measurement Y7 card (invoicing volume, its variability, credit notes, financing cost) |
| "leftover after days" | Ranking score of a signal after removing what the activity clock explains |
| "beat size" | Signal's own ranking score minus the size proxy's 0.617 |

---

## 11. Pointers for the technical reader

| What | Where |
|------|--------|
| Brief (working copy) | `overnight/NORTH_STAR.md` |
| Live technical ledger (every audit, every number) | `overnight/dashboards/CONTEXT.md` |
| Recovery reasons, Siddiqi style | `analysis/outputs/y3_reasons.md` |
| Recovery importance plots | `analysis/outputs/y3_importances.md` |
| Customer-loss importance plots | `analysis/outputs/shap_y7.md` |
| Outcome acceptance table | `data/feature_store/y_acceptance.csv` |
| Feature dictionary | `data/feature_store/feature_dictionary.md` |
| Literature | `analysis/outputs/lit_cashflow.md`, `analysis/outputs/lit_invoice.md` |
| Experiment registry | `analysis/experiments/registry.csv` |
| Frozen holdout | `analysis/splits/holdout_companies.csv` (seed 20260918) |
