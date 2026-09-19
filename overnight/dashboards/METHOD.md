# Method — what we did, from scratch

- **As of:** 2026-09-19 09:25 CEST
- **Read this if:** you want the problem definition and the math behind every number in `MORNING_REPORT.md`. You do not need `CONTEXT.md` unless you want the audit trail for one specific line.

## 1. The problem

Panel: companies $i \in \mathcal{C}$ (1,286) × months $t \in \{1,\dots,24\}$ (Sep 2024 → Aug 2026). Six questions per company-month: healthy? improving? turning? dip or fall? why? how early?

Turned into **supervised prediction with a constraint**:

- An **outcome** $y_{i,t} \in \{0,1\}$ observable only in the future window $t+1 \dots t+h$.
- **Inputs** $x_{i,t} \in \mathbb{R}^p$ computed only from events dated $\le$ end of month $t$ (no look-ahead).
- A function $f$ that ranks company-months with $y=1$ above $y=0$ on companies it never saw.
- **Honesty rule:** columns used to define $y$ may not appear in $x$.

The first overnight number (0.86) violated the rule — it predicted future cash from current cash. Cash today vs cash in three months has Spearman $\rho = 0.85$ on its own. That is persistence, not health. Deleted.

## 2. Data objects

Eight tables on `company_id`. Transactions (dated, signed, categorised bank movements). Invoices (counterparty, amounts, issue/due/paid dates, credit notes). Balances (one still on 2026-09-01). Debt products, banking products, companies, groups.

Two facts: 470 of 1,214 training companies have **no invoices** (invoice inputs are NaN for them, never 0). There is **no key** from an invoice to its bank payment.

## 3. Inputs

118 columns, eight families, each a deterministic function of events $\le$ month-end $t$.

- Activity clock: $\texttt{days}_{i,t} = |\{\text{distinct booking dates in } t\}|$.
- Flags: $\texttt{ss}_{i,t} = \mathbb{1}[\exists\ \text{transaction with category = social\_security in } t]$; same for salary.
- Size proxy: $s_{i,t} = \log\big(1 + \sum_{k=0}^{2} \text{op\_inflow}_{i,t-k}\big)$.
- Reconstructed cash (family B): $\text{liq}_{i,t} = \text{liq}_{i,T} - \sum_{u=t+1}^{T} \text{net}_{i,u}$. An identity (residual $\approx 10^{-12}$ €).
- Runway: $\text{run}_{i,t} = \mathrm{clip}\big(\text{liq}_{i,t} / \max(\tfrac13\sum_{k=0}^{2}\text{out}_{i,t-k},\,1),\,-6,\,24\big)$.
- Invoicing: $\text{issued}_{i,t}$, its lag, coefficient of variation over lags, credit-note share, DSO proxy $=\text{open receivables}/\text{issued}_{i,t}$.
- Every stem also at $t-1$ and $t-3$.

## 4. Outcomes

Fixed rules from the literature, never tuned on our data, each with a forbidden family.

**Y3 — recovery (45→65).**

$$\text{stressed}_{i,t} = \mathbb{1}[\text{liq}_{i,t} < 0 \ \lor\ \text{run}_{i,t} < 1]$$

$$y^{(3)}_{i,t} = \mathbb{1}\big[\exists\,k\in\{1..4\}:\ \text{run}_{i,t+k}\ge 3 \land \text{run}_{i,t+k+1}\ge 3 \land \text{run}_{i,t+k+2}\ge 3\big]$$

Defined only when stressed and $t+6 \le T$. 7.1% positive (402 / 5,648). Forbidden: family B.

**Y7 — top customer lost (dip vs fall).**

$$c^*_{i,t} = \arg\max_c \sum_{u=t-2}^{t} |\text{AR issued to } c|,\qquad y^{(7)}_{i,t} = \mathbb{1}\Big[\sum_{u=t+1}^{t+3} \text{AR issued to } c^*_{i,t} = 0\Big]$$

28.8% positive. Forbidden: family D.

Others (Y2 cash negative 2 of next 3 months; Y4 debt-service ratio doubles; Y5 own payables >30 days late above own p80; Y9 fee/interest spike) follow the same shape. Their models did not beat a single input; they stay as describable labels.

## 5. Splits and metric

**Holdout:** 72 companies / 15 groups, seed 20260918, frozen; never used to fit anything.

**CV:** 5-fold GroupKFold by corporate group on the 1,214 training companies. Mean ± sd across folds.

**AUROC:**

$$\mathrm{AUROC}(f,y) = \Pr[f(x_a) > f(x_b)] + \tfrac12\Pr[f(x_a)=f(x_b)],\quad a\sim\{y=1\},\ b\sim\{y=0\}$$

0.5 coin flip, 1.0 perfect. Threshold- and base-rate-free. Single column $z$: own score $=\max(\mathrm{AUROC}(z,y), \mathrm{AUROC}(-z,y))$.

## 6. The referee — three gates

**Gate 1 — beat size.** $\mathrm{AUROC}(z) - \mathrm{AUROC}(s) \ge 0.02$ and $|\rho_S(z,s)| < 0.50$.

**Gate 2 — leftover after the activity clock.** Within fold, rank-transform and regress

$$\mathrm{rank}(z) = a + b\,\mathrm{rank}(\texttt{days}) + r,\qquad \text{leftover}(z) = \mathrm{AUROC}(r, y)$$

Below 0.55 → nothing new. Check $\rho(r,\texttt{days})$; if the residual still tracks days, the OLS figure is fake and the rank figure is quoted.

**Gate 3 — not a twin.** $\max_j |\rho_S(z, k_j)| < 0.80$ against every kept signal $k_j$.

Diagnostics: **ICC** $= \sigma^2_{\text{between}} / (\sigma^2_{\text{between}} + \sigma^2_{\text{within}})$ — above ~0.90 the column is a company trait, not monthly news; also test the demeaned $z_{i,t} - \bar z_i$. **Bootstrap** over companies on the leftover.

~40 candidates audited. Two readings survived.

## 7. Models

LightGBM, 50 trees, depth 3. Deeper / early-stopped specs collapsed to 1–7 trees.

- Y3, all allowed columns: **0.762 ± 0.016**. Y3, 15 columns: **0.752 ± 0.037**. Clock alone 0.711. Size alone 0.617.
- Y7, 5 columns (issued$_{t-1}$, its CV, credit-note share $t$ and $t-1$, financing cost $t-3$): **0.720 ± 0.034**. Adding DSO drops the weakest fold 0.680 → 0.556 (DSO scores 0.410 on the fastest-collecting fifth).

## 8. Explanation

TreeSHAP: $f(x) = \phi_0 + \sum_j \phi_j(x)$; rank by mean $|\phi_j|$, sign from $\mathrm{corr}(\phi_j, x_j)$; durability by permutation ΔAUROC. Then the Siddiqi / CFPB rule: a reason shown to a company must carry weight *and* pass the gates. SHAP names `a_n_tx` (clock twin ρ 0.94), `a_op_in` (size ρ 0.998), transfers (ICC 0.956), DSO (below chance on a fifth) — all excluded. Reasons: no social security (leftover 0.635), no payroll (0.603), fewer movement days (0.711), and their one-month lags. All signs negative.

## 9. Spec sheet — every headline number pinned to rows, label, inputs, estimator, split

Checked against `analysis/evaluate/protocol.py`, `analysis/models/gbm_core.py`, `analysis/models/gbm_y7_core.py`, `analysis/evaluate/ss_qa.py`.

**Common.** Row $r=(i,t)$, company × calendar month; $T$ = 2026-08. Train companies: 1,214 (21,157 rows). Folds: sort distinct `group_id`, shuffle with seed 20260918, fold $= \text{position} \bmod 5$; every company inherits its group's fold. AUROC with average ranks for ties:

$$\mathrm{AUROC}(R) = \frac{\sum_{r:y_r=1}\text{rank}(f_r) - n_1(n_1+1)/2}{n_1 n_0} = \Pr[f_a > f_b] + \tfrac12\Pr[f_a=f_b]$$

Out-of-fold: fit on folds $\ne k$, score fold $k$, AUROC on fold $k$; **report mean ± sd over the 5 folds**.

**(1) Y3 = 0.762 ± 0.016.** Rows $R^{(3)}$: train rows with $\text{stressed}_{i,t}=\mathbb{1}[\text{liq}_{i,t}<0 \lor \text{run}_{i,t}<1]$ and $t+6\le T$; $|R^{(3)}|=5{,}648$. Label $y^{(3)}_{i,t}=\mathbb{1}[\exists k\in\{1..4\}: \min(\text{run}_{t+k},\text{run}_{t+k+1},\text{run}_{t+k+2})\ge 3]$; $n_1=402$. Inputs: all store columns in families A,C,D,E,F,G,H at lags 0,1,3 (278 columns); **B excluded**. Estimator: LightGBM, `n_estimators=50, max_depth=3, num_leaves=8, learning_rate=0.05`, class weight $n_0/n_1$; score = predicted probability.

**(2) Y3 = 0.752 ± 0.037.** Same rows/label/estimator/folds. Inputs = 5 stems × lags {0,1,3}: `c_ss_month`, `c_salary_month`, `c_n_days_with_tx`, `a_n_tx`, `f_ds_r`. Historical card as measured; `a_n_tx` and `f_ds_r` later failed the leftover gate, so only the other three are quoted as reasons. Not re-fit.

**(3) Single-column bars on Y3.** No model. Per fold: choose sign $\sigma_k\in\{\pm1\}$ maximising AUROC on fit rows; score $f_r=\sigma_k z_r$ on fold $k$; mean over folds. `c_n_days_with_tx` → **0.711** (sign −). $\log(1+\sum_{k=0}^{2}\text{op\_inflow}_{t-k})$ → **0.617** (sign −).

**(4) Leftover after the clock.** Candidate $z$, control $d=$ `c_n_days_with_tx`, rows $R^{(3)}$. $\tilde z=\text{rank}(z)$, $\tilde d=\text{rank}(d)$ over all rows. Per fold, least squares on fit rows $\tilde z=\alpha+\beta\tilde d+\varepsilon$; residual on scored rows $\rho_r=\tilde z_r-(\hat\alpha+\hat\beta\tilde d_r)$; sign on fit rows; AUROC on fold $k$; mean over folds. Gate: $<0.55$ adds nothing. Also report $\text{Spearman}(\rho,d)$; if $|\cdot|\ge0.80$ the raw-value residual is a copy of $d$ and only the rank version is quoted. `c_ss_month`: own 0.693, leftover **0.635**. `c_salary_month`: own 0.671, leftover **0.603**. `a_n_tx`: leftover 0.538 (fails).

**(5) Y7 = 0.720 ± 0.034.** Rows $R^{(7)}$: train rows with ≥1 customer invoice in $t-2..t$ and $t+3\le T$; $|R^{(7)}|=7{,}464$. With $A_{i,c,u}$ = |amount| invoiced by $i$ to customer $c$ in month $u$: $c^*_{i,t}=\arg\max_c\sum_{u=t-2}^{t}A_{i,c,u}$; $y^{(7)}_{i,t}=\mathbb{1}[\sum_{u=t+1}^{t+3}A_{i,c^*_{i,t},u}=0]$; $n_1=2{,}149$. Inputs (five): `e_ar_issued_lag1`, `e_ar_issued_lag_cv`, `e_credit_note_ratio`, `e_credit_note_ratio_lag1`, `f_fc_r_lag3`; **D excluded**. Same estimator and folds. Fold scores 0.745, 0.765, 0.713, 0.698, 0.680. Single column `e_ar_issued_lag1` alone: 0.630 (0.626 on short-history companies).

**(6) Y7 leftover of top-customer billing = 0.789.** Recipe (4) on $R^{(7)}$ restricted to rows where $z$ is defined (40.6%): $z=A_{i,c^*_{i,t-1},t}$, control $d=$ `e_ar_issued_lag1`. Event rate 58% when $z=0$ vs 8% when $z>0$ — nearly the label one month early; shown as a reason, not added to the card.

**(7) Diagnostics.** $\rho_S$ = Spearman over rows where both defined; twin gate $|\rho_S|\ge0.80$; size gate $|\rho_S(z,\text{size})|\ge0.50$. $\text{ICC}(z)=\mathrm{Var}_i(\bar z_i)/(\mathrm{Var}_i(\bar z_i)+\overline{\mathrm{Var}_t(z_{i,t}\mid i)})$; demeaned test re-runs (4) on $z_{i,t}-\bar z_i$.

**(8) Not modelled.** Y2, Y4, Y5, Y8, Y9: labels defined and accepted; no model beat a single column under this protocol; no model AUROC is claimed for them.

## 10. Result

Two clean messages. Recovery from stress is predictable (0.76); the reason is "the book went quiet" — de-escalation, partly mean reversion. Losing the biggest customer is predictable (0.72) from the volume and jumpiness of last month's invoicing, not from payment delay. Health today is months of runway on the latest balance (median 1.08 ≈ 32 days). Defensible lead: one month. The 82→68 deterioration is a real event we could not explain without the cash columns — reported as a failure.
