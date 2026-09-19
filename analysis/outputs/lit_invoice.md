# Invoice / trade-credit literature cluster

Owner: invoice-side literature (dip-vs-fall, why-it-changed, months-earlier).
As-of: 2026-09-19 ~07:40 CEST (pass 3; Ellingsen in, Lian to companion). Hidden 72 never used to fit. No 0–100. No `product/`.
Sibling: bank-statement scorecards, last-value liquidity, Y3 activity, FICO methodology → **see `lit_cashflow`**. Do not grow TURNOVER **0.720**.

Object is a **treasury + invoice trail**, not a bankruptcy classifier. Papers below are kept only when they measure a PD / early-warning / trade-credit number that maps onto Y7 / Y4 tail / Y5 leftover / DSO drop / delay+CN footnotes / Q6 lags / Y9-as-label / J KEEP-Q5.

Night lock this note is not allowed to move: Y7 TURNOVER **0.720** / B_shallow **0.712** (`issued_lag1` + issued-lag CV + CN ±lag1 + `f_fc_r_lag3`). No DSO on the card. CN leftover after issued_lag1 **KEEP 0.597**. delay_coll leftover after DSO **KEEP 0.581**. DSO leftover after days **0.474** / after issued **0.452** — DROP. Y4 HHI **>0.975** footnote **0.605**; HHI as engine X DROP (twin top1 ρ **0.994**). Y5 leftover **65%**; supp HHI protective **2.7% vs 8.6%**; `d_tx` **0.611 PARK** (fold 3). Y8 excuse **PARK**. Y9 is the fee label. Q6 KEEP: issued_lag1 **0.626**, days_lag1 **0.684**, ss_lag1 **0.631**. `e_ap_overdue` leftover **0.584** lives, beat-size **FAIL +0.008**. J `pay_match` KEEP-Q5, leftover **0.556** thin. Dark **470 stay NaN**. `d_cust_lost` leftover after days **0.522 DROP** (Y3; twin of `d_n_cust` ρ **0.861**) — not Wave B.

---


## 1. Ultra cluster (12)

| # | paper | year | venue | url | why ultra | Q |
|---|---|---|---|---|---|---|
| 1 | Banque de France, “Do late customer payments impact companies’ probability of default?” | 2019 | *Bulletin* 227/8 | [en](https://www.banque-france.fr/en/publications-and-statistics/publications/do-late-customer-payments-impact-companies-probability-default) · [PDF](https://publications.banque-france.fr/sites/default/files/medias/documents/819416_bdf227-8_late_customer_payment_vfinale.pdf) | >30d late vs legal 60d **OR 1.42 (~+40% PD)**; ≤30d **OR ~1.09 (no real effect)**; **“The increase in days sales outstanding is not related to the probability of default.”** Only **8/100** failing firms exposed. | Q4, Q5 |
| 2 | Hirshleifer, Li, Lourie, Ruchti, “Do Trade Creditors Possess Private Information?” | 2019 / rev. 2023 | NBER w25553 | [html](https://www.nber.org/papers/w25553) · [PDF](https://www.nber.org/system/files/working_papers/w25553/w25553.pdf) | Buyer **PastDue%** (mean **27%** of trade credit) vs next-**6-month** default: Q5 vs Q1 **+107% bankruptcy odds** (log-odds **0.73**), +52% going-concern, −23% rating-upgrade odds. Stronger for low-liquidity buyers. | Q4, Q5, Q6 |
| 3 | Pérez-Salazar, Márquez, Vidal-Silva, “Algorithmic Profiling of Operational Risk…” | 2026 | *Computers* 15(2):135 | [mdpi](https://www.mdpi.com/2073-431X/15/2/135) | Exists. **Synthetic** 5,000 logs. Supplier-payment **HHI = “supply chain fragility.”** XGB AUC **0.94** vs logit **0.75**; they say operational consistency beats revenue. **Different object** from Y5 (and opposite sign). | Q5 (contradict) |
| 4 | FinRegLab / Howell, Matsumoto, Cochran, “Sharpening the Focus” | 2025-06 | FinRegLab white paper | [landing](https://finreglab.org/research/sharpening-the-focus-using-cash-flow-data-to-underwrite-financially-constrained-businesses/) · [PDF](https://finreglab.org/wp-content/uploads/2025/06/FinRegLab_06-03-2025_Sharpening-the-Focus.pdf) | NSF / overdraft / daily-pay (MCA) counts are **distress indicators**, not extra X once the fee event is the Y. Invoice-side slice only. Full bank-statement scorecard → **see `lit_cashflow`**. | Q3, Q5 |
| 5 | Jacobson & von Schedvin, “Trade Credit and the Propagation of Corporate Failure” | 2015 | *Econometrica* 83(4) | [doi](https://doi.org/10.3982/ecta12148) | Customer failure raises supplier annual bankruptcy risk by **~2 pp (~+100% at the 2% mean)**. Channel is **credit loss + demand shrinkage**. Cash-rich / profitable suppliers absorb. | Q4, Q5 |
| 6 | Boissay & Gropp, “Payment Defaults and Interfirm Liquidity Provision” | 2013 (ECB WP 753 / 2007) | *Review of Finance* 17(6) | [doi](https://doi.org/10.1093/rof/rfs045) · [ECB PDF](https://www.ecb.europa.eu/pub/pdf/scpwps/ecbwp753.pdf) | French CIPE trade-bill defaults: **18.5%** of firms default ≥1×/quarter. Reasons: **disagreement 16.2%** vs **illiquidity 2.1%** (omission 1%, insolvency 0.4%). Constrained firms pass **>1/4** of liquidity shocks; large/liquid firms **stop the chain**. | Q5, J |
| 7 | Irvine, Park, Yıldızhan, “Customer-Base Concentration, Profitability, and the Relationship Life Cycle” | 2016 | *The Accounting Review* 91(3) | [doi](https://doi.org/10.2308/accr-51246) · [WP PDF](https://mpra.ub.uni-muenchen.de/53886/1/MPRA_paper_53886.pdf) | Not monotone: negative-OM CC mean **14.2%** vs positive-OM **9.0%** (t **−27.6**). WP Table 9: **Rank(CC)** raises IPO failure over **5 and 7 years** (weaker at 7). Table 10: Rank(CC)+, AGE×Rank(CC)−. Loss of a major customer is the crash. | Q4, Q5 |
| 8 | Campello & Gao, “Customer concentration and loan contract terms” | 2017 | *JFE* 123(1) | [doi](https://doi.org/10.1016/j.jfineco.2016.03.010) · [Nova PDF](https://www2.novasbe.unl.pt/Portals/0/Research/documents/murillo_campelo.pdf?ver=2019-04-09-144422-130) | +1 sd → **+10 bp** spread (**6%** of 179 bp mean), **+0.2** covenants (vs 1.8), **−2 months** maturity (vs 46). Intensified by customer distress and TC use. Banks price the **tail**, not a linear HHI card. | Q4 |
| 9 | Ellingsen, Jacobson, von Schedvin, “Trade Credit: Contract-Level Evidence Contradicts Current Theories” | 2016 | Riksbank WP 315 | [PDF](https://www.riksbank.se/Documents/Rapporter/Working_papers/2016/rap_wp315_160126.pdf) | **52m** invoices, 51 suppliers, 199k buyers. Weaker buyers have more AP because they **buy more**, not because they stretch days. Contract duration **unrelated** to buyer finances; overdue is a **minor fraction** of payables. Volume, not DSO. | Q4, Q5 |
| 10 | García-Appendini & Montoriol-Garriga, “Firms as liquidity providers” | 2013 | *JFE* 109(1) | [doi](https://doi.org/10.1016/j.jfineco.2013.02.010) · [Caixa WP PDF](https://www.caixabankresearch.com/sites/default/files/content/file/2016/08/1210aa-en.pdf) | Crisis: AR/sales **−3 pp** on average. Cash-rich **raised** TC: **+1 SD cash → +0.5 pp** quarterly AR/sales (~**$9.8m**/yr, **~11%** of LOC). Constrained buyers **took** more. Insurance, not fragility. (Cuñat *RFS* 2007.) | Q5 |
| 11 | Bitetto, Cerchiello, Filomeni, Tanda, Tarantino, “Can we trust machine learning…” | 2024 | *RQFA* 63 | [doi](https://doi.org/10.1007/s11156-024-01278-0) | Invoice-lending book (Italian SMEs). INV: **Delinquency** (mean **1.62%**) and **Outstanding** dominate. They **drop Outstanding** because it is **correlated with Turnover** and keep Delinquency + size. Contemporaneous invoice behaviour beats thin FS. | Q4, Q5, Q6 |
| 12 | Bureau, Duquerroy, Vinas, “Activity shocks and corporate liquidity: the role of trade credit” | 2023 / 2024 | BdF WP 851 → *RCFS* 13(3) | [HTML](https://www.banque-france.fr/en/publications-and-statistics/publications/corporate-liquidity-during-covid-19-crisis-trade-credit-channel) · [PDF](https://publications.banque-france.fr/sites/default/files/medias/documents/wp851_0.pdf) · [blog](https://www.banque-france.fr/en/publications-and-statistics/publications/activity-shocks-amplifier-effect-trade-credit) | **>170k** French firms, daily supplier-payment defaults. **1 SD** net TC (AP−AR) × lockdown → **+0.31 pp / +10%** monthly payment-default PD (baseline **3%**); retail up to **+30%**. In normal months net TC **does not** move PD. Cash / factoring hedges. Horizon **≤2 months**. | Q3, Q5, Q6 |

Twelve is the cap. Not a dump.

### Companions (wave support; not a 13th cluster row)

| paper | year | venue | url | verified | use |
|---|---|---|---|---|---|
| Amberg, Jacobson, von Schedvin, Townsend, “Curbing Shocks to Corporate Liquidity: The Role of Trade Credit” | 2021 | *JPE* 129(1) | [doi](https://doi.org/10.1086/711403) · [MIT PDF](https://hdl.handle.net/1721.1/130501) | **yes** (full text) | Liquidity shock → **more AP drawn**, **less issued** to customers. Compounded TC adjustments ≈ cash. AP +**2.8 pp** by 2012; AR −**1 pp**; postpone-pay **+1.7 pp**. Mechanism is partly **overdue**. Wave A/B. |
| Costello, “Credit Market Disruptions and Liquidity Spillover Effects in the Supply Chain” | 2020 | *JPE* 128(9):3434–3468 | [doi](https://doi.org/10.1086/708736) · [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3258029) | **abstract_only** (JPE/SSRN body not opened; Stanford WP 404) | Bank shock → less TC **and** fewer goods. **Least-important customers** lose credit first; downstream credit-risk spike + employment drop. Wave B (who thins). Do not quote a coefficient we did not see. |
| Lian, “Financial distress and customer-supplier relationships” | 2017 | *JCF* 43 | [doi](https://doi.org/10.1016/j.jcorpfin.2017.02.006) | **abstract_only** | Supplier PD rises with major-customer distress and **persists up to two years**. Why Q6 cannot quote 3-month / 2-year leads on hidden 72. |
| Barrot, “Trade Credit and Industry Dynamics” | 2016 | *JF* 71(5) | [doi](https://doi.org/10.1111/jofi.12371) · [MIT record](https://dspace.mit.edu/handle/1721.1/108627) | **abstract_only** (PDF 405/timeout) | French trucking reform restricting TC supply → corporate default PD **−25%**, persistent, liquidity-constrained. Companion to BdF late-pay, not a 13th row. |

---

## 2. Already-cited — verified? what we used them for vs what they actually say

### Banque de France Bulletin 227/8 — **VERIFIED** (HTML + both FR/EN PDFs)

- **Plan / brief_map used:** “AP late >30 days +~40% PD; ≤30 no effect; DSO growth alone not predictive.” Also: Y5 leftover is “what cash/HHI do not explain.”
- **They actually say:** late *customer* payments (supplier’s AR DSO vs the French **60-day legal ceiling**). All late >60d: **OR 1.24 (~+25% PD)**. Split: **<30d late OR 1.094** (CI almost includes 1); **>30d late OR 1.424 [1.355–1.495]**. Univariate failure rate almost **2×** when DSO **>90d** vs at the 60-day limit — but in the score, **annual DSO growth is unrelated to PD** (sales-strategy vs customer-distress mix). Deteriorated solvency / liquidity / profitability have **OR ≥ 4**. Only **8/100** failing firms are even *exposed* to this late-pay risk; **¾** of those 8 are the >30d bucket.
- **We used it correctly** for the 30-day cut and the DSO-growth claim. **We over-read it** if we treat Y5 leftover 65% as “BdF already measured 65%.” Their leftover is **92% of failures are not late-pay events**. Same *shape*, different denominator.

### Hirshleifer, Li, Lourie, Ruchti NBER w25553 — **VERIFIED** (nber.org + PDF; title is *stock-returns / private information*, not “PastDue% doubles PD”)

- **Plan used:** “PastDue% top vs bottom quintile **doubles** 6-month bankruptcy odds.”
- **They actually say:** proprietary supplier AR files → buyer AP for **4,176** public firms, 2002–17; mean PastDue% **27%**. Next-six-month outcomes: Q5 vs Q1 **log-odds bankruptcy +0.73 (odds +107%)**, going-concern **+0.42 (+52%)**, rating change **−0.26 (−23%)**. Stronger for **low-liquidity / distressed** buyers. Late pay = inability, not cheap investment finance.
- **Close enough:** +107% is “a bit more than double.” Keep the number, drop the paraphrase “doubles.”
- **Object mismatch:** they predict the *buyer’s* default from the *supplier’s* AR tape. Our `e_delay_coll` is the *supplier’s* collection delay. Our Y5 is *our* AP overdue own-p80. Same clock, three seats.

### Pérez-Salazar, Márquez, Vidal-Silva *Computers* 15(2):135 (2026) — **VERIFIED** (mdpi.com)

- **Plan used:** “operational volatility (sd of inter-transaction time) and **supplier HHI beat revenue** for micro-enterprise solvency.”
- **They actually say:** **synthetic** 5,000-entity logs (GitHub `cvidalmsu/micro-enterprise-solvency-data`). Dependency vector = **HHI on supplier payments; “High concentration indicates supply chain fragility.”** Headline is operational **consistency / variance**, not a measured PD on real defaults. AUC 0.94 is in-sample on data they generated.
- **CONTRADICT / different object:** our Y5 supplier-HHI **>0.975 tail is protective (2.7% vs 8.6%)**, leftover after size **0.525 dies**, twin of `d_supp_top1` ρ **0.987**. Their HHI is a simulated fragility feature; ours is a real AP-behaviour leftover. Do not import their sign.

### FinRegLab 2025 “Sharpening the Focus” — **VERIFIED** (finreglab.org PDF, June 2025)

- **NORTH_STAR / plan used:** NSF counts and low/negative balances as the strongest cash-flow distress flags; Y9 = fee/interest turning.
- **They actually say (invoice-adjacent slice):** among originated fintech loans (N≈38k), distress flags are **# insufficient-funds / overdraft**, **# low or negative ending balances**, and **daily-pay / MCA** withdrawals. Mean NSF among borrowers **0.05**. Higher withdrawals, balance vol, daily-pay, and frequent low balances raise default; credits and balances lower it. Young / low-FICO gain more from cash-flow X.
- **Invoice-side use:** Y9 **is** the fee/interest event (`m_fin` ρ **0.875** with the label). Do **not** merge Family M. Sibling `lit_cashflow` owns the rest: 3-month **application** averages, last-value balances, NSF *as X*, Yao / Norden / Siddiqi. Do not re-score those here.

---

## 3. Mapping SAME / CONTRADICT / NEW

| night object | literature | verdict |
|---|---|---|
| **Y7 TURNOVER 0.720** (thin / jumpy **issued_lag1**, no DSO) | **Ellingsen–Jacobson–von Schedvin:** 52m contracts — AP moves with **input volume**, not days; overdue is a **minor fraction**. Jacobson 2015: demand **shrinkage** (~**+2 pp / +100%** supplier PD). Bitetto: **Outstanding correlated with Turnover** — they drop it and keep **Delinquency** (mean 1.62%). BdF: DSO *growth* is not PD. Amberg: liquidity-short firms **contract issued**. | **SAME** on “volume / issuance, not DSO days.” **Do not grow 0.720.** Fold 4 is short-DSO *churn*; more E columns are the BdF mistake. |
| **DSO leftover dies (0.474 / 0.452)** | BdF heading: *“The increase in days sales outstanding is not related to the probability of default.”* SHAP #1 on the 278-col card was the poison (short-DSO fifth **0.410**). | **SAME.** Confirmed leftover dies. DSO stays **PARK as a health Y**, off TURNOVER. |
| **delay leftover 0.581 after DSO** | Hirshleifer PastDue% (not DSO days) predicts 6-month buyer default; BdF **>30d** moves PD, **≤30d** does not. Our delay vs DSO ρ **0.214** — not a twin. Amberg: overdue is the **mechanism**, +1.7 pp postpone. | **SAME as a Q5 footnote, not an add-on.** ICC **0.922** / demean **0.521** = *who-pays-late* style, Hirshleifer’s cross-section, not a month shock. Q6 CLOSE (empty first 6 months). |
| **CN leftover 0.597 after issued_lag1** | Still **no measured-PD paper** on credit-note *ratio* (second-pass search: collections theses use dispute *flags* to predict invoice lateness, not firm PD). Closest: Boissay **disagreement 16.2%** vs illiquidity **2.1%** (dispute defaults **€11.6k** vs illiquidity **~€40k**). Our CN vs issued_lag1 ρ **0.237**, demean leftover **0.509 dies**. | **NEW / gap.** KEEP footnote = *who-uses-notes*, not this month’s correction. Fallback **note 98.6% / refund 1.4%**. Do not invent `y_credit_note`. |
| **Y4 HHI>0.975 tail KEEP 0.605; HHI as X DROP** | Irvine: concentration is **tail / young / loss-making**, not a linear scorecard (body in our data CV **0.445**; quintiles not monotone). Campello–Gao: banks price +1 sd as **+10 bp**, not a GBM feature. Twin top1 ρ **0.994**. | **SAME.** Footnote the monopoly tail. Do not reopen Y4 trees. Q6 HHI_lag3 **CLOSE** (21.7% on short). |
| **Y5 leftover 65%; supp HHI protective 2.7 vs 8.6** | BdF: only **8/100** failures are late-pay exposed. Pérez-Salazar: HHI = fragility (**CONTRADICT sign**, synthetic). García-Appendini / Cuñat / Boissay: liquid / concentrated suppliers **absorb**. **Bureau–Duquerroy–Vinas:** 1 SD net TC (AP−AR) × activity shock → **+10%** *payment-default* PD; in quiet months net TC is **zero**. Cash / AR-finance hedges. | **SAME leftover.** **CONTRADICT Pérez-Salazar.** Protective tail is insurance, not a crash. The unnamed 65% is the Bureau object (net TC × dip), not HHI. `d_tx` 0.611 stays **PARK** (fold 3 owns **85.9%**). |
| **`e_ap_overdue` leftover 0.584, beat-size FAIL +0.008** | BdF ≤30d **no effect**; Hirshleifer needs PastDue% **intensity**, not a stock share. Bureau’s default dummy is the **Y**, not an X. | **SAME: off the card.** Rank leftover lives; it does not beat size. Twin of `e_ap_overdue_30` ρ **0.849**. |
| **Q6 issued_lag1 0.626 / days_lag1 0.684 / ss_lag1 0.631** | Hirshleifer’s horizon is **6 months**. Bureau’s TC-amplifier is **≤2 months** then reverses. Bitetto: contemporaneous INV delinquency already informs. Lian (companion, abstract): customer-distress spillover lasts **up to 2 years** — we cannot use that on hidden-72 (**1-month claims only**). | **SAME on 1-month issued.** **CLOSE** F lag3 / Y4 HHI_lag3 / delay. Do not quote 3-month leads on the hidden 72. |
| **Y9 is the fee label** | FinRegLab NSF / daily-pay **are** the distress event. BdF chart also has **interest charge (high vs low)** as a PD factor. | **SAME.** Do not merge Family M. `f_fc_r_lag3` stays on TURNOVER; contemporaneous `f_fc_r` stays DROP. |
| **Y8 excuse PARK** | Jacobson / Lian / Costello need a **customer-failure or bank-shock join**. PLOS ONE (Berloco et al. 2021) needs a **bank-transfer graph**. We have **no invoice↔bank FK**. `y8_inv` already had cash on **98.1%** and still lost to `a_in12`. | **SAME PARK.** Not “no FK so we could not try.” We tried; the join is not the excuse. |
| **Family J pay_match KEEP-Q5, leftover 0.556 thin** | Boissay: most trade-bill “defaults” are **disputes (16.2%)**, not illiquidity (**2.1%**). Amount-match rate **35.7% vs 0.5% random** is a real book, not a recovered FK. | **SAME.** Thin leftover is what the literature predicts. Do not merge. Dark 470 **NaN**. |
| **Dark 470 stay NaN on invoice features** | Bitetto / Hirshleifer / BdF / Bureau are **invoice-conditioned** samples. Filling 0 would invent a book. | **SAME.** |
| **`d_cust_lost` leftover 0.522 (Y3)** | Irvine / Jacobson care about **loss of a major customer / demand shrinkage**, not a quarterly headcount of counterparties. Our lost is a **`d_n_cust` twin** (ρ **0.861**). | **SAME DROP as Y3 X.** Wave B is **euros to last month’s top-1 after issued_lag1 on Y7**, not this column. |

---

## 4. Grounding (judge-quotable)

1. The night engine is a **trajectory on a treasury + invoice trail**, not a bankruptcy score: Y7 asks whether the **top customer is about to vanish**, Y4 whether a **monopoly book is about to crash inflow**, Y5 whether **we** start paying >30d late — none of those is “will this firm enter insolvency proceedings.”
2. Banque de France already separated **late-pay intensity** from **DSO growth**: >30d late raises PD ~40% (OR 1.42); ≤30d does not; **annual DSO growth is unrelated to PD**. Our DSO leftover after days **0.474** and after issued **0.452** is that sentence on this panel.
3. That is why TURNOVER **drops DSO** and still prints **0.720 / fold-4 0.680**: Ellingsen et al. already showed AP/AR stocks move with **invoice volume**, not days; the SHAP-#1 DSO card was **worse than chance on the short-DSO fifth (0.410)**. Growing the card with more E columns repeats the BdF mistake.
4. Hirshleifer’s PastDue% (Q5 vs Q1, **+107% 6-month bankruptcy odds**) is the **footnote**, not the stem: our delay leftover after DSO is **0.581** and is a **who-pays-late trait** (ICC 0.922), empty as Q6 until month 7.
5. Credit notes have **no PD paper** with a number. Our CN leftover **0.597** after issued_lag1 is **who-uses-notes** (demean 0.509 dies). Boissay’s **16.2% disagreement vs 2.1% illiquidity** is why a note ratio is not a crash lever and must not grow **0.720**.
6. Customer HHI is a **monopoly tail**, not an engine X: Irvine’s life-cycle (costly when young / losing), Campello–Gao’s **+10 bp** loan price, our HHI>0.975 **22.1% vs 11.5%** with body CV **0.445**. Twin top1 ρ **0.994** — keep the footnote, drop the column.
7. Supplier HHI is **not** Pérez-Salazar fragility. García-Appendini / Cuñat / Boissay describe **insurance by liquid suppliers**. We measured the protective tail **2.7% vs 8.6%**. Bureau–Duquerroy–Vinas name the rest: **net trade-credit (AP−AR) only moves payment-default PD in an activity shock** (+10% per 1 SD; quiet months zero). Y5’s honest sentence is the **65% leftover**, not a supplier crash.
8. Hidden 72 is **1-month lead time**. Issued_lag1 **0.626** (present on **95.2%** of short Y7 rows) is the only invoice Q6 that survives. HHI_lag3, F lag3, and delay do not. Bureau’s amplifier itself dies after **two months**.

---

## 5. Next waves (4)

Each wave is **not** a redo of Y5 / Y7 TURNOVER / Y8 / Y9. Forbidden: grow TURNOVER, rewrite `gbm_core.py`, merge parquet, fit hidden 72, invent `y_dso` / `y_delay` / `y_credit_note` / `y_pay_match` / `y_cust_lost`. Do not reopen `d_cust_lost` as Y3 X (leftover **0.522** already died).

### Wave A — Top-1 customer PastDue% as a Y7 leftover (not on the card)

- **Owner files:** new `analysis/evaluate/top1_pastdue_qa.py` + `analysis/outputs/top1_pastdue_qa.md`. Read-only store + `clean.invoices`.
- **Forbidden X:** Family **D as Y7 engine X**; `e_dso_proxy`; putting delay / CN back on TURNOVER; Family M; Y5 E (Y5 never E).
- **Acceptance:** train group-fold leftover AUROC of **top-1 AR PastDue%** (or share of that customer’s open >30d) **after** `issued_lag1` + CN **≥ 0.58**, ρ vs `e_delay_coll` **< 0.80**, beat-size, dark 470 **NaN**. If leftover dies, CLOSE and keep the delay footnote.
- **Why literature asks:** Hirshleifer’s object is the **buyer’s** PastDue% on the **supplier’s** tape, 6 months ahead. BdF’s +40% is **customer** lateness, not our DSO. Amberg: overdue is how TC adjusts (+1.7 pp postpone). We have `counterparty_id` on invoices.
- **Why not a redo:** Y5 is *our* AP own-p80. Delay leftover is *firm-level* who-pays-late. This is **which customer** is already late **before** top-1 share goes to 0.

### Wave B — Issued-to-top-1 thinning (identity of the volume, still never D-on-Y7)

- **Owner files:** `analysis/evaluate/issued_top1_qa.py` + md. In-memory only.
- **Forbidden X:** `d_cust_hhi` / `d_cust_top1` / **`d_cust_lost`** as engine X; Y4 trees; grow TURNOVER; B as Y2/Y3 X. Do not overwrite `cust_lost_qa` / `top1_qa`.
- **Acceptance:** leftover of **AR issued to last month’s top-1** after firm-level `issued_lag1` on **Y7 labeled rows** **≥ 0.58**, or the same leftover on **Y4 tail rows** after the HHI>0.975 flag. Twin screen vs `d_cust_top1` **and** vs `d_cust_lost`. Dark 470 NaN.
- **Why literature asks:** Jacobson’s second channel is **demand shrinkage**; Irvine’s crash is **loss of a major customer**; Amberg: AR issued contracts **1 pp** under a liquidity shock; Costello (abstract): **least-important** customers lose credit first — so the euro that vanishes is **named**. Firm-level issued_lag1 cannot see *who* thinned.
- **Why not a redo:** not HHI level (already DROP), not TURNOVER add-on, not Y4 trees, **not** `d_cust_lost` (Y3 leftover **0.522**, n_cust twin). Identity of the missing euro on the Y7 / Y4-tail seats.

### Wave C — Credit-note *on the top-1 counterparty* (footnote only)

- **Owner files:** `analysis/evaluate/cn_top1_qa.py` + md.
- **Forbidden X:** grow TURNOVER; invent `y_credit_note`; CN as Y3 X; merge J.
- **Acceptance:** leftover of **CN |amt| share on last month’s top-1** after firm-level CN **and** issued_lag1 **≥ 0.58**. Twin screen vs firm CN. If leftover **< 0.55**, CLOSE — the 0.597 is the who-uses-notes trait.
- **Why literature asks:** Boissay’s defaults are **mostly disputes** (16.2% vs 2.1% illiquidity; dispute tickets **€11.6k** vs illiquidity **~€40k**). Still **no PD paper** on credit-note ratios.
- **Why not a redo:** `credit_note_qa` already split **note 0.595 / refund 0.496** leftover after issued_lag1 and showed demean dies. Do **not** redo that firm-level cut. This asks *which customer* gets the notes.

### Wave D — Y5 leftover 65%: net trade-credit × activity (Bureau), never E-on-Y5

- **Owner files:** `analysis/evaluate/y5_net_tc_qa.py` + md. **Never E as Y5 engine X** (Y5 forbids E). Diagnostic leftover only; use **D / A** if a model column is needed.
- **Forbidden X:** `d_supp_hhi` as engine X; merge Y4; `d_tx` as leave-one-group law; Family E on the Y5 GBM; grow TURNOVER. Do not overwrite `y5_why`.
- **Acceptance:** on the **AP neither cell (222/341 = 65.1%)** of `y5_ap_od30_ownp80` (not low-`a_io_ratio`, not high-`d_supp_hhi`), leftover of **net TC** (`e_ap_open − e_ar_open`) **× a quiet/activity flag** (days or inflow dip) **≥ 0.58** after the cash+HHI dummy, beat-size, not fold-3 `d_tx`. If nothing clears, the 65% stays the sentence.
- **Why literature asks:** Bureau–Duquerroy–Vinas: 1 SD net TC raises **payment-default** PD **+10% only in the shock months**; quiet months **zero**. That is our Y5 object (we start paying suppliers late) and our leftover (cash and HHI already taken). García-Appendini: +1 SD cash → **+0.5 pp** AR/sales in a crisis (insurance). Pérez-Salazar said the opposite on synthetic data.
- **Why not a redo of Y5:** not the supplier-HHI tail (2.7 vs 8.6 already measured), not the 0.611 `d_tx` quote, trees stay PARK, E stays off the Y5 card.

---

## 6. Explicitly out

- **Bankruptcy-only classifiers** (Altman Z, Omega Score, generic “SME default AUC 0.94” blogs).
- **DSO as engine X** — already DROP; BdF already said growth is not PD; leftover died.
- **Growing TURNOVER 0.720** — CN, delay, DSO, AP overdue, DPO, pending, FX stay off.
- **Cash-flow scorecards / last-value liquidity / Y3 activity / FICO methodology** — **see `lit_cashflow`** (Yao 2017, Ng et al. 2025, FinRegLab *as bank-statement X*, Pedregal–Trapero Tobit).
- **Generic “customer concentration and default” blogs**; Dhaliwal cost-of-equity without a PD number (Campello–Gao already prices the tail).
- **Ciampi 2018 IJBM** — “payment behaviour” is **bank overdraft >60d**, not invoices. Near-miss → `lit_cashflow`.
- **PLOS ONE Berloco et al. 2021** — 3-month trade-credit *network* EWS that needs a **bank-transfer graph**. That is the Y8 join we do not have. Near-miss, not a card.
- **Marouani 2014 SSRN** — thin preprint, not used.
- **Pérez-Salazar operational-volatility / `c_gap_sd`** — already DROP as a days twin (ρ −0.905). Their vol claim is not an invoice lever.
- **Collections-ops “predict this invoice will be late”** (Hilti thesis LightGBM AUC >0.85; Dima et al. 2026 synthetic 26k invoices AUC 0.804). Different object: one invoice’s lateness, not company trajectory. Dispute flags there support Wave C, not a cluster row.
- **`d_cust_lost` as Y3 X** — leftover after days **0.522** already died (2026-09-19 07:27).
- **Costello / Amberg as engine papers** — companions for Waves A/B; Costello coefficients **UNVERIFIED** beyond the abstract.

---

## What this note did not do

- Did not change Y3 0.762 / 0.752 or Y7 0.720 / 0.712.
- Did not put anything on the 15-col card. Did not grow TURNOVER.
- Did not run `build_targets`, rewrite `gbm_core.py`, merge parquet, or open leftover QA files to edit them.
- Did not invent citations. Every ultra-cluster URL was fetched. Lian 2017 and Costello 2020 table coefficients beyond the abstract are **abstract-verified** — marked as such.
- Did not cover bank-statement NSF-as-X, last-value liquidity, or FICO methodology (sibling `lit_cashflow`).
