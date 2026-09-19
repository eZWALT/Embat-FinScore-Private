# Cash-flow underwriting literature — ultra cluster

As-of `2026-09-19` morning (live fetches; no invented cites).
Owner: cash-flow underwriting + FICO-like SME / bank-statement scorecards.
Sibling: invoice / concentration / trade-credit / DSO / NSF-as-invoice / lead-time → **see `lit_invoice`**. One-line pointers only below.

Object: a FICO-like **company-health reading of a 24-month treasury trail**. Not a bankruptcy classifier. Trajectory (45→65 recover vs 82→68 deteriorate), not last-month snapshot. No `product/`. No 0–100.

Night engine this note maps onto (do not change):

- Q1 KEEP last-value `b_runway` / `b_liq` (walk is an identity; never B as Y2/Y3 X).
- Y2 trees PARK: already-negative persistence, not Y4’s crash.
- Y3 engine **0.762** / 15-col **0.752**; quiet-stressed recover (SS/salary −); days bar **0.711**; size bar **0.617**; `a_in3` DROP as engine X leftover **0.521**.
- 15-col card KEEP after leftover QAs: `c_ss_month` leftover **0.635**, `c_salary_month` **0.603**, `c_n_days_with_tx` + lags. `a_n_tx` DROP. `f_ds_r` DROP from card.
- Javier 14: 11 SAME / 2 CLOSE / 1 DRIFT (vol). `a_out_vol` **0.722** is a company trait (demean **0.549**) — not the engine.
- Referee: leftover-after-days (beat size ≥0.02 AND leftover after days AND not SIZE AND not twin). Literature often quotes raw AUROC without this.
- Connection clocks (`g_has_*`, `g_n_accounts` leftover **0.428**, `created_at`, `f_has_*`) are not health.
- `f_util_snapshot` last-month-only **1.6%**; Y10 utilisation impossible.
- Honest referee: Y never from the same columns as allowed X (the 16h 0.86 died on this).

Cluster size: **11** (10 ultra + 1 contrast). Not a dump.

---

## 1. Ultra cluster

| # | paper | year | venue | url | why ultra (one sentence) | our Q |
|---|-------|------|-------|-----|--------------------------|-------|
| 1 | FinRegLab / Howell, Matsumoto, Cochran, *Sharpening the Focus: Using Cash-Flow Data to Underwrite Financially Constrained Businesses* | 2025 | FinRegLab empirical white paper (Jun) | [finreglab.org](https://finreglab.org/research/sharpening-the-focus-using-cash-flow-data-to-underwrite-financially-constrained-businesses/) · [PDF](https://finreglab.org/wp-content/uploads/2025/06/FinRegLab_06-03-2025_Sharpening-the-Focus.pdf) | Bank-statement SMB default on 38k fintech loans: credits, withdrawals, balances, NSF, low/neg ending balances, daily-pay MCAs; 3-month average *before application*. | Q1, Q3, Q5 |
| 2 | Yao, Levy-Chapira, Margaryan, *Checking account activity and credit default risk of enterprises* | 2017 | arXiv:1707.00757 | [abs](https://arxiv.org/abs/1707.00757) | French bank: checking-account activity beats financial ratios for corporate default; they **normalise for account size**; credit-line violations + cash inflows are the named levers. | Q3, Q5 |
| 3 | Ng, Chu, Lim, Boon, Low, Tan, *AI-BAAM: AI-Driven Bank Statement Analytics as Alternative Data for Malaysian MSME Credit Scoring* | 2025 | arXiv:2510.16066 (ICAIF FinRem) | [abs](https://arxiv.org/abs/2510.16066) · [html](https://arxiv.org/html/2510.16066) | MSME bank-statement scorecard; WOE/IV screen with **IV ≥ 0.5 = leakage smell** (cites Siddiqi 2017); quotes raw AUROC 0.806 blended / 0.763 statement-only. | Q5 (referee) |
| 4 | Norden & Weber, *Credit Line Usage, Checking Account Activity, and Default Risk of Bank Borrowers* | 2010 | *Review of Financial Studies* 23(10):3665–3699 | [doi](https://doi.org/10.1093/rfs/hhq061) · [Mannheim PDF](https://madoc.bib.uni-mannheim.de/2983/1/SSRN_id1101548_pub137.pdf) | Existing-borrower **monitoring** (not application): usage, limit violations, and cash inflows go abnormal ~**12 months** before default; amplitude warns ~5 months; especially useful for small firms. | Q3, Q6 |
| 5 | Mester, Nakamura, Renault, *Transactions Accounts and Loan Monitoring* | 2007 | *Review of Financial Studies* 20(3):529–556 | [doi](https://doi.org/10.1093/rfs/hhl018) · [Philly Fed WP PDF](https://www.philadelphiafed.org/-/media/frbp/assets/working-papers/2005/wp05-14.pdf) | Canadian SME: monthly checking balances track collateral (AR+inventory); borrowings in excess of collateral predict downgrades; last-value balances are the monitor. | Q1, Q6 |
| 6 | Farrell & Wheat, *Cash is King: Flows, Balances, and Buffer Days* | 2016 | JPMorgan Chase Institute | [page](https://www.jpmorganchase.com/institute/all-topics/business-growth-and-entrepreneurship/report-cash-flows-balances-and-buffer-days) · [PDF](https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/institute/pdf/jpmc-institute-small-business-report.pdf) | Health, not PD: **cash buffer days** = avg daily balance / avg daily outflow; median **27 days** (p25=13, p75=62) on 597k SMEs. Same object as `b_runway`. | Q1 |
| 7 | Farrell, Wheat, Grandet, *Facing Uncertainty: Small Business Cash Flow Patterns in 25 U.S. Cities* | 2019 | JPMorgan Chase Institute (Aug) | [page](https://www.jpmorganchase.com/institute/all-topics/business-growth-and-entrepreneurship/facing-uncertainty-small-business-cash-flow-patterns-in-25-us-cities) · [PDF](https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/institute/pdf/institute-facing-uncertainty.pdf) | 24-month-style **trajectory**: 7 regularity patterns on 290k deposit accounts; irregular firms exit more; thin buffer **plus** irregular is the worst cell. | Q1, Q2, Q3 |
| 8 | FinRegLab + CRA, *The Use of Cash-Flow Data in Underwriting Credit: Empirical Research Findings* | 2019 | FinRegLab (Jul) | [page](https://finreglab.org/research/the-use-of-cash-flow-data-in-underwriting-credit-empirical-research-findings/) · [PDF](https://finreglab.org/wp-content/uploads/2023/12/FinRegLab_2019-07-25_Research-Report_The-Use-of-Cash-Flow-Data-in-Underwriting-Credit_Empirical-Research-Findings.pdf) | Six lenders (incl. Kabbage SMB): cash-flow-only AUC **0.592–0.725** (one thin 0.572); usually ≥ bureau; combined lifts (one named combo **0.758** vs FICO+attrs **0.720**). Thin-file. | Q3, Q5 |
| 9 | Siddiqi, *Intelligent Credit Scoring: Building and Implementing Better Credit Risk Scorecards* (2nd ed.) | 2017 | Wiley / SAS | [Wiley](https://www.wiley.com/en-us/Intelligent+Credit+Scoring%3A+Building+and+Implementing+Better+Credit+Risk+Scorecards%2C+2nd+Edition-p-9781119282334) | The transferable scorecard practice: **8–15** characteristic risk profile, WOE, IV, reasons codes, application vs **behavior** scoring of an existing book. | Q5 |
| 10 | Pedregal & Trapero, *Censored Data Forecasting: Applying Tobit Exponential Smoothing with Time Aggregation* | 2024 | arXiv:2409.05412 | [abs](https://arxiv.org/abs/2409.05412) | Tobit ETS for **inventory stockouts**, not credit-line draw vs granted. Why we PARKED weekly TS vs historical mean — the planned LOC censoring object does not exist here. | Q2 (PARK) |
| C | **CONTRAST** Altman, *Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy* | 1968 | *Journal of Finance* 23(4):589–609 | [doi](https://doi.org/10.1111/j.1540-6261.1968.tb00843.x) | One-year **statement snapshot → bankruptcy PD**. The thing we are not building. | — |

Invoice / trade-credit / DSO / concentration (Hirshleifer 2019; Banque de France Bulletin 227/8; Pérez-Salazar 2026): **see `lit_invoice`**.

---

## 2. Already-cited: verified? used-for vs actually-says

All five plan-§9 cash-flow starting points **exist**. Fetched abstracts / PDFs. None UNVERIFIED.

| cite | verified? | what the night plan used them for | what they actually say |
|------|-----------|-----------------------------------|------------------------|
| FinRegLab 2025 *Sharpening the Focus* | **YES** — PDF + landing page 2025-06-03/04. Authors Howell, Matsumoto, Cochran. Draws from Hair et al. (2025) NBER w33367 (same data; **not a second cluster row**). | NSF/fee + low/neg balances as Y9 / family-B distress. | **Application** underwriting. Variables **averaged across the three months prior to application**. Higher credits and balances → lower default; withdrawals, credit/balance SD, daily-pay MCAs, frequent low/neg balances, NSF → higher default. +1 SD balances (~$64k) ≈ −2 pp default for young firms vs −1 pp for older. Cash-flow AUC gain 0.011 overall, **0.023** for low-score × young. This is default PD, not 45→65 recovery. |
| Yao et al. 2017 arXiv 1707.00757 | **YES** — arXiv abs 2017-07-03. Yao, Levy-Chapira, Margaryan (École Polytechnique / Société Générale). | Checking-account activity beats ratios; credit-line violations + inflows. | Confirmed. Account-only boosting AUC **79.19–79.66%** vs financial+managerial **76.17%**; merged **84.24%**. They **normalise to eliminate account size**. They cite Norden & Weber (2010), Mester et al. (2007), Jiménez et al. (2009). Still a **default** paper. |
| Ng et al. 2025 arXiv 2510.16066 | **YES** — arXiv html. Ng/Chu/Lim/Boon/Low/Tan, AI Lens, Kuala Lumpur. 611 Malaysian MSME applicants. | Adopt IV ≥ 0.5 as a leakage smell. | Confirmed, citing Siddiqi (2017): IV <0.02 none; 0.02–0.1 weak; 0.1–0.3 medium; 0.3–0.5 strong; **≥0.5 suspiciously high / potential leakage**. Statement-only LR AUROC **0.763** vs application-only **0.647**; blended **0.806**. They quote **raw AUROC**. No leftover-after-activity referee. LR beat trees on n=366 — not a license to grow our 278-col engine. |
| Pedregal & Trapero 2024 arXiv 2409.05412 | **YES** — arXiv abs 2024-09-09. | Tobit ETS for censored series; planned use = LOC drawn vs `granted`. | **Inventory / lost-sales** censoring (hourly/daily stockouts). Published twin: Pedregal, Trapero, Holgado, *IJPR* Dec 2025, doi [10.1080/00207543.2025.2600521](https://doi.org/10.1080/00207543.2025.2600521), same demand-planning object. **Not** a credit-line paper. Our Y10 utilisation is impossible (snapshot 1.6%); weekly TS already lost to the historical mean. PARK stands. |
| Siddiqi 2017 (at most 1–2 methodological) | **YES** — Wiley 2nd ed. page + SAS excerpts. Ng quotes the IV≥0.5 rule from this book. Full book not re-read cover-to-cover; the transferable claims below are in the fetched excerpts + Ng’s citation. | Reasons codes, WOE, thin-file. | Scorecards should be an **8–15 characteristic risk profile** (not 4–5, not 278). Distinguishes **application** scoring from **behavior** scoring of existing accounts. Reasons must be the factors actually scored. IV ≥ 0.5 is the leakage smell Ng imported. Thin-file is a segment, not a license to treat `created_at` as health. |

Companion verified, **not** extra cluster rows:

- **Hair, Howell, Johnson, Matsumoto 2025**, NBER w33367 *Modernizing Access to Credit for Younger Entrepreneurs: From FICO to Cash Flow*. Fetched PDF. Academic source of FinRegLab 2025. Same 3-or-6-month statement average; “most fintechs use **<10** basic measures.” ROC-AUC lift vs random-guess benchmark **+28%** for owners ≤35 vs **+4.7%** over 35. Cites Mester 2007, Norden 2010, Khandani 2010, Frost 2019 as *existing-borrower monitoring* — the literature already splits application vs behavior. [nber.org/papers/w33367](https://www.nber.org/papers/w33367)
- **Jiménez, López, Saurina 2009**, *RFS* 22(12):5069–5098. Fetched Banco de España WP PDF. Credit-line **usage rises as condition worsens**; ~10% per year aging decline. Twin of Norden’s utilisation lead. We cannot run it (Y10 impossible). [BdE WP](https://www.bde.es/f/webbde/SES/Secciones/Publicaciones/PublicacionesSeriadas/DocumentosTrabajo/08/Fic/dt0821e.pdf)
- **Farrell & Wheat 2017**, *The Ups and Downs of Small Business Employment* (JPMC Institute; PDF titled *Making Payroll*). Fetched PDF. Employer SMEs: payroll is a material outflow (median **18%** of outflows); median cash buffer **18 days** vs **27** for all SMEs; **61.8%** have unstable payroll. Not a 12th cluster row — companion of #6/#7. [page](https://www.jpmorganchase.com/institute/all-topics/business-growth-and-entrepreneurship/report-small-business-payroll) · [PDF](https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/institute/pdf/institute-small-business-payroll-report.pdf)
- **Sufi 2009**, *RFS* 22(3):1057–1088 *Bank Lines of Credit in Corporate Finance*. Fetched FDIC WP. Lines substitute for cash **only** for high-cash-flow firms; covenant violations (~36%) cut access. Same utilisation hole as Norden: we cannot score the line, so last-value cash (Q1) is the constrained firm’s actual liquidity. Not a cluster row. [doi](https://doi.org/10.1093/revfin/hhm007)
- **Ciampi 2018**, *IJBM* 13(4):57–80 *Using Prior Payment Behavior Variables for Small Enterprise Default Prediction Modelling*. Fetched CCSE PDF. 980 Italian SEs. “Payment behaviour” here is **bank loans past due and/or overdrawn >60 days** (stock, year-end), not invoices — sibling correctly parked it. Companion of Norden/Y2: already-overdrawn exposure predicts default and the lift grows at 2–3 year horizons / smaller firms. Still a **default** paper; not a 12th cluster row. [doi](https://doi.org/10.5539/ijbm.v13n4p57) · [PDF](https://ccsenet.org/journal/index.php/ijbm/article/download/73976/40931)
- **Farrell, Wheat, Mac 2018**, JPMC Institute *Growth, Vitality, and Cash Flows: High-Frequency Evidence from 1 Million Small Businesses*. Fetched PDF. 1.3M firms / 3.1B txs, Oct 2012–Feb 2018. **Source of the seven cash-flow patterns** that 2019 Facing Uncertainty later maps onto cities. Key sentences: new firms “achieve more stable and regular cash flow patterns over time, **or exit**”; “volatile expenses (relative to revenues) are much more likely to exit”; “Small businesses can and do **mitigate irregular cash flows by holding more cash**.” That last line is our Q1 photograph plus their irregularity typology — not a license to put `a_out_vol` on Y3. Not a 12th cluster row (would twin #7). [page](https://www.jpmorganchase.com/institute/all-topics/business-growth-and-entrepreneurship/report-growth-vitality-cash-flows) · [PDF](https://www.jpmorganchase.com/content/dam/jpmc/jpmorgan-chase-and-co/institute/pdf/institute-growth-vitality-cash-flows.pdf)
- **Farrell, Wheat, Mac 2020**, JPMC Institute *Small Business Cash Liquidity in 25 Metro Areas* (Apr). Fetched page. Later-sample median buffer **15 days** (only 40% >21 days); SF/SJ/Seattle 18 vs Atlanta/Orlando 11. Do not treat the 2016 **27-day** median as eternal — our p50 1.079 months ≈ 32 days still sits next to 2016, not 2020. Companion of #6, not a cluster row. [page](https://www.jpmorganchase.com/institute/all-topics/business-growth-and-entrepreneurship/small-business-cash-liquidity-in-25-metro-areas)
- **Nemoto, Yoshino, Okubo, Inaba, Yanagisawa 2018**, ADBI WP **857** *Credit Risk Reduction Effect on Small and Medium-Sized Enterprise Finance through the Use of Bank Account Information*. Fetched Econstor full text + ADB landing (WP number confirmed 857; do not use the nearby `adbi-wp859.pdf`, a different paper). Japanese RDB, n≈42k. 79 account ratios (end-month deposits/loans/net deposits, min/max/SD, **÷ sales or loans**). Out-sample Accuracy Ratio: FS **65.1** / account **64.6** / hybrid **71.4**. Smallest bucket (<¥30m): account **59.5** beats FS **55.3**. SAME as Yao (checking complements ratios; size-normalise). Still **default PD**. Not a 12th cluster row. [ADB](https://www.adb.org/publications/credit-risk-reduction-effect-sme-finance-through-bank-account-information) · [Econstor](http://hdl.handle.net/10419/190278)
- **Formisano, Gallucci, Modina, Pietrovito 2016 / Modina, Pietrovito, Gallucci, Formisano 2023.** CASMEF WP 4 (Nov 2016, fetched LUISS PDF) → *QREF* 89:254–268 (2023, doi fetched). Italian co-operative banks; WP: 113 banks / ~12k firms, 2012–13. Checking-account X = **credit-line usage, overdraft days, n overdraft accounts, consecutive months overdrawn**. Overdrafts + usage lift default accuracy **~10%**. 0→1 overdraft day ≈ **+1 pp** PD. Journal version: 13,081 firms / 111 banks; usage and violations predict 1- and 2-year PD after FS + bank-time FE. Companion of Norden / Wave D — **not** a 12th cluster row (would twin #4). [WP PDF](https://www.luiss.it/sites/default/files/media-documents/1604.pdf) · [doi](https://doi.org/10.1016/j.qref.2023.04.008)
- **Ehling & Haushalter 2014**, Banco de España WP 1412 *When does cash matter? Evidence for private firms*. Fetched NHH/BdE abstract + body. 180k Norwegian private firms, **annual statements**, 2000–09. Small-firm cash stock predicts sales/assets/survival **only around negative industry or macro shocks**. SAME as Farrell that last-value cash is the small-firm hedge; not a bank-statement trail (would twin #6 if clustered). [BdE](https://ideas.repec.org/p/bde/wpaper/1412.html) · [NHH](http://hdl.handle.net/11250/95412)
- **La Rocca, Staglianò, La Rocca, Cariola, Skatova 2019**, *Small Business Economics* 53(4). Fetched RePEc + doi page. European SME **statement cash holdings** raise operating performance (precautionary). Same FS-cash family as Ehling, not a treasury trail. Not a cluster row. [doi](https://doi.org/10.1007/s11187-018-0100-y)

---

## 3. Mapping — SAME / CONTRADICT / NEW vs KEEP-DROP-PARK

### Last-value Q1 `b_runway` / `b_liq`

- **SAME as Farrell & Wheat 2016.** Cash buffer days = balance / outflow. Ours is months: `b_runway = clip(liq / max(out3/3, 1), −6, 24)`. Night p50 **1.079 months ≈ 32 days** sits next to their median **27 days** (2016), not the later 2020 metro median of **15**. Last-vs-snap ρ **0.966**: the 2026-09 still *is* the photograph. Walk identity (median |resid| 9.1e-12). Facing Uncertainty Finding 3: thin buffer + irregular = worst survival cell — that is a **health** sentence, not a PD. 2018: firms “mitigate irregular cash flows by holding more cash.”
- **SAME as Mester 2007** that the bank’s privilege is the **monthly last-value balance** as a monitor.
- **CONTRADICT FinRegLab/Hair 2025 if we put B on Y2/Y3 X.** They average 3 months of balances *as application X for default*. We KEEP last-value as Q1 **description** and **forbid B as Y2/Y3 X** (honest leftover after days dies; `b_below_0` Y2 0.896 is the lock, not a KEEP). Hair’s 3-month mean is a different job (origination PD) than our last-value photograph.
- **NEW:** the walk is an identity. Literature never had to reconstruct backwards from a single still. Do not rewrite `liquidity.py`.

### Quiet-stressed recover (Y3 0.762 / 0.752; SS/salary −; days 0.711)

- **NEW vs the default literature; SAME as JPMC payroll.** FinRegLab/Hair: higher credits and balances → *lower* default. Yao: low/unstable inflows raise default. Norden: inflows fall ~12 months before default. Our Y3 is the **opposite slice**: among *already-stressed* months, quiet firms (no SS, no salary, fewer days-with-tx) are the ones whose runway later prints ≥3 for three months. All top-10 SHAP signs are −. That is de-escalation / mean reversion at the bottom — the 45→65 direction Javier already saw when “inflow −40%” failed. Farrell & Wheat 2017 give the cash-buffer reason: employer payroll is **18%** of outflows and cuts median buffer from **27 → 18 days**. `c_ss_month` leftover after days **0.635** and `c_salary_month` **0.603** are that drain, not a skip (`c_missed_salary` 0.513 CLOSE).
- **CONTRADICT raw “more activity = healthier”.** Hair’s most powerful variables are deposits and balances (size-like). Withdrawals are their distress sign. Our size bar **0.617** loses to days **0.711**; `a_in3` leftover after days **0.521** dies. `a_op_out` leftover after days **0.586 lives** but is **SIZE** (ρ vs log1p(a_in3) **0.741**) and a twin of `a_out3` (**0.907**) — CLOSE unused leftover / DROP from the 44 (07:30 QA). Hair would quote the raw 0.678; leftover-after-days plus SIZE/twin is why we do not. Quiet is not “small,” and euro withdrawals are not a Y3 leftover.
- **PARTIAL SAME as Farrell/Wheat/Mac 2018 + Facing Uncertainty 2019** that the object is a **cash-flow pattern over time**, not a last-month snapshot. 2018: firms “transition from less regular … to more regular … **or exit**” — vitality language for 45→65 vs 82→68. **CONTRADICT if we treat their irregularity as our days.** Irregularity is timing consistency (7 patterns). We DROPPED `c_gap_sd` as a days twin (ρ −0.905; leftover after days 0.535 dies). Days leftover after gap **0.658** still lives — activity level, not irregularity. Do not revive `c_gap_sd`. 2018’s “volatile expenses → exit” is a **survival** sentence; our `a_out_vol` 0.722 demean 0.549 is a trait, not that month’s shock.
- **SAME as Siddiqi** that an 8–15 stem card is the FICO-like object. 15-col A **0.752** beats days 0.711. After leftover QAs the KEEP stems are `c_ss_month` (0.635 leftover), `c_salary_month` (0.603), `c_n_days_with_tx` + lags. `a_n_tx` DROP (days twin ρ 0.938). `f_ds_r` DROP (unused leftover 0.528). Do not rewrite `gbm_core.py`. Family I CLOSE (best add 0.7615 < 0.772).

### Leftover-after-days referee vs raw AUROC

- **NEW. Literature does not have this gate.** Ng 0.806, Yao 0.79, FinRegLab 2019 0.592–0.725, FinRegLab 2025 +0.011 AUC — all **raw**. Our 16h 0.86 died because Y and X shared `net`/`overdue`. Ng’s IV≥0.5 (Siddiqi: “suspicious / check for over-predicting”) is the closest published smell test; leftover-after-days is stricter and on-brief. The residual is a **referee**, not a new X — do not feed days-residuals back into the card (that is a different leakage).
- **SAME as Yao** that you must **normalise size** before quoting activity. `a_in3` leftover 0.521 is that lesson. Size stays a *bar* (0.617), not a card stem.
- **CONTRADICT quoting `a_out_vol` 0.722 as the engine.** Hair/FinRegLab put credit/balance SD in the distress kit and quote raw coefficients. Our 0.722 reproduces; demean **0.549** (η² 0.741) — company trait, not a month shock. Javier vol DRIFT 0.354 vs store `b_bal_vol`. X CLOSE. Y PARK. Night quote stays 0.762 / 0.752.

### Y2 trees PARK (already-negative persistence)

- **SAME as Norden/Mester/Ciampi** that once the account is already in the red, the *level* is the story. 82% of our Y2 positives are already below 0 at *t*; clean-now leftover 1.5%. Best legal single days **0.571** → **0.549** after dropping 12 chronic dark names. Ciampi’s overdrawn>60d stock is that lock as a **default** X; we cannot use B as Y2 X without recreating it.
- **CONTRADICT merging Y2 with Y4.** Jaccard 0.057. Y2 median in-ratio 1.03 vs Y4 0.36. Literature’s default path is closer to Y4’s crash + Norden’s usage run-up than to Y2’s already-neg lock. Trees stay PARK. Do not reopen.

### Utilisation impossible / connection clocks not health

- **CONTRADICT capability, not the finding.** Norden (usage → 80–100% in the last year), Jiménez (usage rises as condition worsens), Yao (limit violations) are the classic 6–12 month lead. Sufi 2009 adds: lines are liquidity only for high-cash-flow firms. We have `f_util_snapshot` last-month-only **1.6%**, native Y3 n_pos=0, fillna0 leftover 0.711 is a fake days leak. Y10 **impossible**. Do not invent a utilisation Y. Last-value cash is the constrained firm’s actual buffer (Sufi + Farrell).
- **Norden AMPLI is not our days.** Fetched Mannheim PDF: `AMPLI = HIGH − LOW` of the **account balance** (euro range). Median AMPLI of defaulters drops from ~€1,000 about **five months** before default. That is a **B-family** object. We forbid B as Y2/Y3 X, so we cannot score ΔAMPLI. Wave B’s Δdays leftover is the **legal activity-clock** leftover of that paper, not a reconstruction of HIGH–LOW. Same PDF: on accounts **without a credit line**, the cumulative number of overdrafts (ΔCUMOVER) is the **only** significant account-activity predictor. That is Wave D’s object, not utilisation.
- **CONTRADICT treating `created_at` / `g_has_*` / `g_n_accounts` / `f_has_*` as Hair’s “firm age.”** Hair’s young firm is **<5 years operating history**. Our `created_at` after 2024-09 is **73.6%** — a **connection clock**. `g_n_accounts` leftover after days **0.428**; rise-only 1561/0. Access ≠ ERP (dark p50 accounts = 3 = invoiced). PARK as health Ys. CLOSE as X.

### Pedregal Tobit ETS

- **CONTRADICT the planned use.** Censoring level in the paper is inventory, not `granted`. Weekly ETS/SARIMAX already lost to the historical mean. PARK. No Python port of UComp required tonight.

### Scorecard reasons (Siddiqi)

- **SAME** that reasons must be the factors actually scored. Current `y3_importances.md` still lists `a_n_tx` (perm 0.030) and `f_ds_r`. Those are **DROP from the card**. A judge-facing reason list that still names them fails Siddiqi/ECOA-style practice even though we are not writing a 0–100 tonight.

---

## 4. Grounding (judge-quotable)

The contest asks whether a 24-month treasury trail can say how a company is — 45→65 versus 82→68 — not whether it will file. The cash-flow literature that actually uses bank statements is almost all **origination PD** on a **3-month average** (FinRegLab 2025 / Hair 2025; FinRegLab 2019 AUC 0.592–0.725) or **existing-borrower default monitoring** with credit-line usage as the 12-month lead (Norden 2010; Mester 2007; Formisano/Modina overdraft days). A 2021–2024 search for *portfolio-monitoring SME cash-flow scorecards* returns the same two families plus JPMC vitality/survival — not a FICO-like health trajectory. We are building the missing object: a **behavior-style health reading** of an already-open book, which is the job Siddiqi (2017) calls behavior scoring and Farrell & Wheat (2016) call cash-buffer vitality.

Healthy, tonight, is last-value reconstructed cash / `b_runway` on the 2026-09 still (last-vs-snap ρ 0.966; persist 0.85; p50 1.079 months ≈ JPMC 2016’s **27** cash-buffer days, not the later 2020 metro median of **15**). That photograph is Q1. It is not Y2/Y3 X. Putting balances into the recovery model would repeat Hair’s application recipe and recreate the 16h mistake (Y and X from the same cash). 2018 Growth/Vitality: firms hold more cash to **mitigate irregular flows**, then either regularise or exit — the contest pictures, not a PD.

Turning toward 45→65 is Y3 quiet-stressed recover: 278-col shallow 0.762 ± 0.016, 15-col A 0.752 ± 0.037, beating days 0.711 and size 0.617. Among already-stressed months, SS leftover after days 0.635 and salary leftover 0.603 live; `a_in3` leftover 0.521 dies. Farrell & Wheat 2017: employer payroll is 18% of outflows and cuts median buffer 27→18 days — so SS/salary − is de-escalating a known drain, not “more activity is healthier.” The default literature would have expected bigger credits to help. They do not, once the firm is already on the floor.

The published papers quote raw AUROC. Ours died at 0.86 when Y shared `net`/`overdue` with X. Ng’s IV≥0.5 is a leakage smell; leftover-after-days (beat size ≥0.02, leftover after days, not SIZE, not twin) is the referee a judge can audit. `a_out_vol` 0.722 is what raw AUROC looks like when the feature is a company trait (demean 0.549). Do not quote it as the engine.

Norden’s 12-month lead is real and it is **utilisation plus inflows**. We cannot score utilisation (Y10 last-month 1.6%). Q6 tonight is days_lag1 (short 0.684 / 100%) and SS_lag1 leftover 0.631 — one-month claims, hidden 72 included. Connection clocks (`created_at` 73.6%, `g_n_accounts` leftover 0.428, `f_has_*`) are not Hair’s firm age and are not health.

A FICO-like later score, if the team ever writes one, is an 8–15 monotone card on the KEEP stems (SS, salary, days + lags), with reasons equal to those stems. It is not Altman’s Z, not a 278-column tree, and not a 0–100 tonight.

---

## 5. Next waves (3–4)

Literature asks for these. None reopens a parked Y. None rewrites `gbm_core.py`. None merges parquet. None touches leftover QA files already in flight.

### Wave A — Last-value vs 3-month mean runway (Q1 photograph only)

- **Owner files:** `analysis/evaluate/runway_window_qa.py` → `analysis/outputs/runway_window_qa.md` (new). Do not edit `liquidity.py`.
- **Forbidden X families:** B as Y2/Y3 X. No new Y. No holdout fit.
- **Acceptance:** Spearman and leftover-after-last-value of a 3-month mean `b_runway` / `b_liq` (Hair’s recipe) vs last-value. If leftover dies (<0.55), last-value KEEP stays and Hair’s 3-month average is tagged application-only. If it lives, still **PARK as forecast Y**, never engine X.
- **Why literature asks:** FinRegLab/Hair 2025 average three months of balances as the cash-flow X; Farrell & Wheat average daily balances to get buffer days. We only published last-value.
- **Why not a redo:** Y1 liquidity path already PARK (last-value wins CV OOF; Spearman t↔t+3 = 0.85). This is a Q1 description bake-off, not a reconstruction model.

### Wave B — Norden lead without utilisation (Δdays leftover after days-level)

- **Owner files:** `analysis/evaluate/days_delta_qa.py` → `analysis/outputs/days_delta_qa.md` (new). May append `q6_quoted.md` footnotes only if the parent asks.
- **Forbidden X families:** F (no `f_util_snapshot`, no `f_ds_r` back on the card). B. `created_at`. Hidden-72 claims beyond one month.
- **Acceptance:** on train books with months-on-book ≥18, leftover of a 3- or 6-month **change** in `c_n_days_with_tx` after the **level** of days. KEEP only if leftover ≥0.60 AND beat-size ≥0.02 AND not a twin of days (ρ<0.80) AND not SIZE. Otherwise CLOSE. Hidden 72 stays 1-month.
- **Why literature asks:** Norden’s 12-month signal is usage + inflows; AMPLI (HIGH−LOW of **balances**) warns ~5 months. Jiménez is usage. We cannot score usage or AMPLI (B forbidden). A 3-/6-month **change in days-with-tx**, leftover after the *level* of days, is the legal activity-clock leftover — not a recreation of HIGH−LOW.
- **Why not a redo:** not Y10 (impossible). not F lag3 (already CLOSE, empty until so-far≥6). not Y2 trees (already-neg persistence). not Y4 crash (Jaccard 0.057).

### Wave C — Siddiqi reasons on the post-leftover KEEP card (Q5)

- **Owner files:** `analysis/outputs/y3_reasons.md` (new). Read-only on existing SHAP / leftover numbers. **Do not rewrite `gbm_core.py`.** Do not refit.
- **Forbidden X families:** quoting `a_n_tx`, `f_ds_r`, `a_op_in`, `e_dso_proxy`, `a_transfer` (leftover 0.579 / ICC 0.956 TRAIT), B, I, M, J as reasons. Historical SHAP still ranks those (00:16 `y3_importances.md`); leftover QAs DROPped them as card stems. No new column on the 15-col card.
- **Acceptance:** a judge-facing reason list whose named levers are only the leftover-KEEP stems (`c_ss_month`, `c_salary_month`, `c_n_days_with_tx` ± lags) with the published leftovers (0.635 / 0.603 / days 0.711). `y3_importances.md` stays the 50-tree historical SHAP; this file is the **card** reasons.
- **Why literature asks:** Siddiqi — reasons = factors actually scored; 8–15 characteristic profile. CFPB Circular 2023-03 (fetched): reasons must “relate to and accurately describe the factors actually considered or scored”; closest-checklist is not enough. Current importances still advertise `a_n_tx` (perm 0.030) after it was DROPped.
- **Why not a redo:** not a new Y3 engine; not Family I merge (0.7615 already CLOSE); not a 0–100.

### Wave D — NSF-like **count** leftover after days (FinRegLab distress, not Y9 trees)

- **Owner files:** a scratch extract (not Family M, not a parquet merge) + `analysis/outputs/nsf_count_qa.md`. In-memory only.
- **Forbidden X families:** M (fee/fin shares **are** the Y9 label, ρ 0.875). B (`b_below_0` / `b_neg_episodes` are the lock). F. Family I (`i_below0_x_payroll` already CLOSE). Do not score vs Y9. Do not invent `y_nsf`. Dictionary has **no NSF/overdraft token** — count must come from raw `description`/`category` strings, not reconstructed balances.
- **Acceptance:** leftover-after-days of an NSF/overdraft **count** (not euro fee) ≥0.60 AND beat-size ≥0.02 AND ρ vs Y9 <0.80 AND not SIZE. Else CLOSE / PARK. Trees stay PARK. **Do not reconstruct negative-balance days from B** — that is the Y2 lock (`b_below_0`).
- **Why literature asks:** FinRegLab 2025’s three distress flags are NSF counts, low/neg ending balances, daily-pay MCAs. Formisano/Modina: overdraft **days** and consecutive months overdrawn lift PD ~10% after FS (0→1 day ≈ +1 pp). Norden (same PDF as #4): on accounts **without a credit line**, ΔCUMOVER is the only significant activity predictor. We have no line (Y10 1.6%). We have the balance photograph (Q1) and Y9 as a fee/interest **label**. We do not have an NSF *token* count.
- **Why not a redo:** Y9 GBM already PARK (own-p80 0.554 loses to `a_out6` 0.565). Family M CLOSE. This is one leftover gate on a count, not a new fee model.

---

## 6. Explicitly out

- **Bankruptcy-only / Z-score families.** Altman 1968 is the one contrast. Altman–Balzano 2024 *Bouncing Back* (JSBM) is statement-ratio recovery + DSO — **see `lit_invoice`**, not this cluster.
- **Consumer FICO / UltraFICO / Experian–VantageScore 2024 cash-flow consumer scores / FICO SBSS 0–300.** SBSS is an application bureau score (owner FICO + statements), not a 24-month treasury health reading. Khandani, Kim, Lo 2010 (*JBF*) is consumer credit-card delinquency. Berg, Burg, Gombović, Puri 2020 *RFS* is device/digital-footprint consumer default (AUC 0.736 combined). Hair cites both as monitoring prior; neither transfers to SME treasury. CFPB Circulars 2022-03 / 2023-03 are consumer ECOA; the *reasons-must-be-scored-factors* sentence transfers (Wave C) — the circulars themselves are not cluster rows. CFPB §1071 is data-collection, not a scorecard.
- **Stevenson & Pond 2016** (*SEF*): banker *interviews* on UK/DE SME lending process, not bank-statement variables. Ciampi SLR 2021 cites it next to Norden; we opened it and left it out.
- **Berger, Frame, Miller 2005** *JMCB* 37(2):191–222. Fetched Atlanta Fed WP. Bank *adoption* of small-business credit scoring (availability, price, risk of credits <$100k). No bank-statement variables. Not a treasury-health paper.
- **OCC/CFPB/Fed/FDIC/NCUA 2019** Interagency Statement on Alternative Data. Fetched PDF. Cash-flow data as **consumer** ability-to-repay (income and expenses over time). Policy cousin of FinRegLab 2019; not a 24-month company-health reading. Nakamura & Roszbach 2016 (Philly Fed WP 16-14): internal ratings vs bureau, not checking-account X.
- **Generic ML-on-tabular / OCR / BigTech platform scores.** Ng’s template-matching / LLM extraction chapter is out. Frost et al. 2019 BIS WP 779 (Mercado Libre / Ant e-commerce ratings) is platform data, not a treasury trail — Hair cites it as monitoring prior; it does not transfer. Djeundje, Crook, Calabrese, Hamid 2021 *ESWA* (fetched Edinburgh PDF): consumer **email + psychometric** thin-file, not SME treasury. Vendor blogs (Ocrolus, Prestatech pSCORE, Experian MY i-SCORE) confirm industry already splits origination vs portfolio monitoring; they are not papers.
- **Credit-line utilisation papers as engine X.** Jiménez 2009 is real; we cannot score it. Mentioned under Norden, not a cluster row.
- **Invoice / concentration / trade-credit / DSO / NSF-as-invoice.** Hirshleifer PastDue%, Banque de France >30d, Pérez-Salazar operational-volatility × supplier HHI: **see `lit_invoice`**.
- **Weekly Tobit ETS ports, 0–100 formulas, `product/`, leftover QA files in flight** (`op_out_qa`, `fin_cost_qa`, `ap_overdue_qa`, `n_accounts_qa`, `util_snap_qa`, …).

---

## Method note

Live search + page fetch (Exa, then Cursor web search/fetch after Exa rate-limit). Every cluster URL was opened. Invented citations = failure; none added. Hair 2025 and Jiménez 2009 verified as companions, not dumped into the table. Pedregal 2025 *IJPR* verified as the published twin of the 2024 arXiv.

Machine-readable: `analysis/outputs/lit_cashflow_cluster.csv`.
