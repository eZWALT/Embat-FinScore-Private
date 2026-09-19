# Family B — balances still / reconstruction QA

Generated `2026-09-19T00:46` UTC by `python -m analysis.evaluate.balances_b_qa`.
Holdout 72 (seed 20260918) is **coverage only**. Rates, Spearman,
and PARK/CLOSE/KEEP are train. No parquet rewrite. No new GBM. No 0–100.
Does not invent a Y from `b_liq` / `b_runway`. Does not use Family B as X for Y2/Y3.
Family G is a connection clock — this module does not redo it.

## Headline

- Raw `balances`: **7,996 rows / 1,273 companies**. As-of 2026-09-01: 7,980 (99.8%). Not the still: 16 rows on 4 late-August dates (dictionary: closest prior day). Null balance sentinels: 3.
- Identity `b_liq + after-flows = snapshot`: train n=20,941 CM / 1,195 companies. median |residual| = **9.095e-12** euro; |resid|>1€ 0.0%; |resid|>1% of |snap| 0.0%. **CLOSE the reconstruction** — the walk is an identity.
- Walk types = checking + saving + tpv. Excluded |snapshot| share **21.0%** (debt orphans in `balances`, plus card / investment / leftover bank types). Custom *checking* is **in** the walk (261 products, |bal| 1.605e+09) — Family B filters type, not service.
- First-month `g_n_accounts=0` is 63.7% of train companies (G 63.7% replica). On those first months `b_liq` is finite 97.8%, null 2.2%, exactly 0 0.1%. The walk ignores `created_at` — unconnected ≠ missing cash.
- Spearman(`b_liq_t`, `b_liq_t+3`) train = **0.848** (n=17,356; CONFIRM ~0.85). Short <12: 0.730 (n=1,833). 24-month books: 0.862 (n=9,093). Short books are not *more* persistent than long books. Long books still have a path (first-vs-snap weaker than last-vs-snap / range).
- Last-value Q1 (train last month): `b_liq` p50=8.202e+04, share<0 4.7%; `b_runway` p50=1.079, share<1 49.1%. Not a Y2/Y3 model.
- Holdout coverage (descriptive): 72/72 have a balance row (100.0%); cash-walk snapshot 72/72.
- Zombie cash products (0 txs, train): 538/4,408 (12.2%) holding 0.4% of |cash snapshot|.
- `b_bal_vol` vs log1p(a_in3) ρ=-0.282 (not SIZE; |ρ|≥0.50).
- Late-book pre-first-tx months: **none** (grid starts at first tx). Late on-book neg 7.4% vs on-time 8.1%. First-month neg late 8.9% vs on-time 11.5% — not a left-trunc red pile.
- Short persist after dropping age<3: 0.736 vs long 0.866. Later arrivals first-vs-snap 0.619 vs on-time 0.606 (still-leak ranking: no).
- Last-month vs cash snapshot ρ=0.966; |snap−b_liq| is Sep flows (median 200, identity max 9.313e-09). Last>first on 48.6% of train; 24-month neg share 11.5% → 5.1% inside the same 435 (neg→pos 38 / pos→neg 10). 13 train companies have checking txs (8,645) and no snapshot (1 still live in the last 30d, COMP_0312) — null `b_liq`, do not invent a 0-still. Steps closer to the still 51.8%. ρ(company-median, last)=0.808 (typical month ≠ still). Unique walk companies 1,267 (train 1,195); 2026-07-20 last cash-tx on 46.2% of the 13 vs 0.3% of photographed walk — hole stamp.

## Brief questions

1. **Who is healthy?** — last-value `b_liq` / `b_runway` is the night Q1 *description*. It is the 2026-09 still walked backward, not a forecast.
2. **Who is improving?** — last>first `b_liq` is a coin flip (~49%). The *negative tail* shrinks toward extract (11.5%→5.1% on the same 435). Description only; not a new Y. Y1 liq path stays PARK.
3. **Who is turning?** — do not invent a Y from B (circular 16h death). Y2 already *is* the B path.
4. **Dip vs fall?** — reconstruction identity, not a dip detector.
5. **Why did it change?** — excluded snapshot cash is mostly debt orphans + investment, not a health why.
6. **Months earlier?** — short books persist *less* (0.73 vs 0.86), not more. First-vs-snap is ~0.61 on long and short. The still is the *last* month; it does not leak a constant ranking backward. Lead time on liquidity is the walk through later flows, not a photograph of 2024.

## 1. Raw `balances`

| item | n |
| --- | ---: |
| rows | 7,996 |
| companies | 1,273 (train 1201 / holdout 72; universe 1286) |
| product_id | 7,996 |
| distinct dates | 5 (2026-08-25 → 2026-09-01) |
| date = 2026-09-01 | 7,980 (99.8%) |
| not 2026-09-01 | 16 |
| null balance (sentinel) | 3 |
| product_known=false | 29 |

Not everything is 2026-09-01. The dictionary already allowed the closest prior day. Those 16 late-August rows are real extract jitter, not a second still.

Dates:

| date | n |
| --- | ---: |
| 2026-08-25 | 2 |
| 2026-08-27 | 5 |
| 2026-08-28 | 8 |
| 2026-08-29 | 1 |
| 2026-09-01 | 7980 |

Product mix (banking type, or debt type when the balance row has no `banking_products` join):

| bucket | type | rows | companies | sum balance | sum |balance| | null bal |
| --- | --- | ---: | ---: | --- | --- | ---: |
| checking | checking | 4647 | 1267 | 7.964e+09 | 8.151e+09 | 3 |
| debt_orphan | loan | 1006 | 235 | -1.37e+09 | 1.37e+09 | 0 |
| card | card | 794 | 205 | 9.249e+05 | 1.444e+06 | 0 |
| debt_orphan | lineofcredit | 532 | 204 | 7.971e+06 | 2.856e+08 | 0 |
| debt_orphan | confirming | 227 | 70 | -5.375e+07 | 5.67e+07 | 0 |
| investment | investment | 185 | 92 | 2.423e+08 | 2.424e+08 | 0 |
| debt_orphan | leasing | 176 | 44 | -6.041e+07 | 6.041e+07 | 0 |
| debt_orphan | guarantee | 149 | 50 | -8.441e+07 | 8.441e+07 | 0 |
| debt_orphan | mortgage | 60 | 11 | -8.387e+06 | 8.387e+06 | 0 |
| debt_orphan | renting | 34 | 14 | -3.296e+05 | 3.296e+05 | 0 |
| wallet | wallet | 32 | 20 | 4.262e+05 | 4.262e+05 | 0 |
| unknown_orphan | (no product) | 29 | 19 | 3.466e+07 | 5.548e+07 | 0 |
| tpv | tpv | 25 | 10 | 0 | 0 | 0 |
| risk | risk | 25 | 5 | -1.969e+06 | 2.002e+06 | 0 |
| debt_orphan | factoring | 23 | 19 | 3.23e+06 | 1.279e+07 | 0 |
| expensesPlatform | expensesPlatform | 23 | 7 | 2.273e+05 | 2.273e+05 | 0 |
| lineofcomex | lineofcomex | 19 | 8 | -4.873e+06 | 4.93e+06 | 0 |
| saving | saving | 10 | 9 | 8.228e+07 | 8.228e+07 | 0 |

## 2. Identity

Formula: `end_bal_t = snapshot − sum(checking/saving/tpv flows with month > t)`. Snapshot-day txs sit in 2026-09. Train company-months with both `b_liq` and a cash snapshot:

| check | n | median \|err\| | p90 | max | share >1€ | share >1% \|snap\| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| store `b_liq` + after vs snap | 20,941 | 9.095e-12 | 6.985e-10 | 7.153e-07 | 0.0% | 0.0% |
| store vs `liquidity._reconstruct` | 20,941 | 0 | 0 | 0 | — | — |
| store vs product-level walk | 20,941 | 0 | 0 | 0 | — | — |
| last-month recovered vs snap | 1,195 | 0 | 1.164e-10 | 1.192e-07 | — | — |

**CLOSE.** Residuals are numerical noise. Do not rewrite `liquidity.py`. Do not treat the walk as a feature that *learns* history — it is the still minus later booked flows.

PARK any idea of using the snapshot as a Y. That would score the extract photograph.

## 3. Walk vs excluded

Walk |cash| = 8.233e+09 (79.0% of all |balances|). Excluded |cash| share 21.0%. Positive-cash share excluded 5.2%.
Train companies with a balance row but no cash-walk snapshot: 6 / 1,201.

Piles (n_co here **sums across types** — a company with checking+saving is counted twice. Unique membership is §26):

| pile | rows | companies (type-sum) | sum bal | sum |bal| | positive bal |
| --- | ---: | ---: | --- | --- | --- |
| walk | 4682 | 1358 | 8.046e+09 | 8.233e+09 | 8.139e+09 |
| debt_orphan | 2207 | 647 | -1.567e+09 | 1.879e+09 | 1.563e+08 |
| bank_excluded | 1078 | 342 | 2.371e+08 | 2.514e+08 | 2.442e+08 |
| unknown_orphan | 29 | 19 | 3.466e+07 | 5.548e+07 | 4.507e+07 |

By type:

| type | in walk? | rows | companies | sum bal | sum |bal| |
| --- | --- | ---: | ---: | --- | --- |
| checking | True | 4647 | 1339 | 7.964e+09 | 8.151e+09 |
| loan | False | 1006 | 235 | -1.37e+09 | 1.37e+09 |
| card | False | 794 | 205 | 9.249e+05 | 1.444e+06 |
| lineofcredit | False | 532 | 204 | 7.971e+06 | 2.856e+08 |
| confirming | False | 227 | 70 | -5.375e+07 | 5.67e+07 |
| investment | False | 185 | 97 | 2.423e+08 | 2.424e+08 |
| leasing | False | 176 | 44 | -6.041e+07 | 6.041e+07 |
| guarantee | False | 149 | 50 | -8.441e+07 | 8.441e+07 |
| mortgage | False | 60 | 11 | -8.387e+06 | 8.387e+06 |
| renting | False | 34 | 14 | -3.296e+05 | 3.296e+05 |
| wallet | False | 32 | 20 | 4.262e+05 | 4.262e+05 |
| (unknown) | False | 29 | 19 | 3.466e+07 | 5.548e+07 |
| tpv | True | 25 | 10 | 0 | 0 |
| risk | False | 25 | 5 | -1.969e+06 | 2.002e+06 |
| expensesPlatform | False | 23 | 7 | 2.273e+05 | 2.273e+05 |
| factoring | False | 23 | 19 | 3.23e+06 | 1.279e+07 |
| lineofcomex | False | 19 | 8 | -4.873e+06 | 4.93e+06 |
| saving | True | 10 | 9 | 8.228e+07 | 8.228e+07 |

Debt product outstanding sitting in `balances` is the bulk of the excluded pile (loan / LOC / confirming…). Card and investment are the named *bank* exclusions. TPV is in the walk but the extract stock is all-zero. Custom checking is walked.

## 4. Coverage (train)

| set | n companies |
| --- | ---: |
| train on the monthly grid | 1,214 |
| with a `balances` row | 1,201 |
| with any tx | 1,214 |
| with a cash-walk snapshot | 1,195 |
| ever `g_n_accounts>0` | 1,209 |
| tx but no balance | 13 |
| balance but no cash-walk | 6 |
| cash-walk but never G-connected | 2 |

First-month unconnected (`g_n_accounts=0`): 773 / 1,214 = 63.7%. `b_liq` on those rows: finite 97.8%, null 2.2%, zero 0.1%, negative 8.9%, p50=1.049e+05.
All first-month `b_liq` null share 1.6%. CM with `g_n_accounts=0`: 2,618; `b_liq` finite 97.4%, null 2.6%, zero 0.2%, p50=8.04e+04.

The 63.7% hole is G's connection clock, not a missing `b_liq`. Unconnected first months usually already have a reconstructed cash path.

## 5. Persistence — short vs long

Train companies: short <12 first-tx trail 350; 24-month grid 435; late first tx 779.

| slice | lag | n pairs | n companies | Spearman |
| --- | --- | ---: | ---: | --- |
| all | 1 | 19746 | 1195 | 0.909 |
| all | 3 | 17356 | 1195 | 0.848 |
| short_<12 | 1 | 2505 | 336 | 0.843 |
| short_<12 | 3 | 1833 | 336 | 0.730 |
| long_24 | 1 | 9959 | 433 | 0.915 |
| long_24 | 3 | 9093 | 433 | 0.862 |
| late_first_tx | 1 | 9787 | 762 | 0.903 |
| late_first_tx | 3 | 8263 | 762 | 0.833 |
| from_2024_09 | 1 | 9959 | 433 | 0.915 |
| from_2024_09 | 3 | 9093 | 433 | 0.862 |

Company first / last reconstructed cash vs the extract snapshot:

| slice | n | ρ first vs snap | ρ last vs snap | ρ first vs last | p50 range/|snap| | p50 |first−snap|/|snap| | p50 |last−snap|/|snap| |
| --- | ---: | --- | --- | --- | --- | --- | --- |
| all | 1214 | 0.6163 | 0.9665 | 0.6125 | 2.248 | 0.7867 | 0.001492 |
| short_<12 | 350 | 0.5917 | 0.9255 | 0.5529 | 1.843 | 0.7995 | 0.009204 |
| long_24 | 435 | 0.6062 | 0.9832 | 0.6297 | 2.873 | 0.8173 | 0.001179 |
| late_first_tx | 779 | 0.6194 | 0.9574 | 0.6009 | 1.946 | 0.765 | 0.002033 |
| from_2024_09 | 435 | 0.6062 | 0.9832 | 0.6297 | 2.873 | 0.8173 | 0.001179 |

**KEEP last-value Q1 as a description of the extract still** (last-vs-snap ρ≈0.97). It is **not** 'everyone is the still all the way back': first-vs-snap is ~0.61 on both short and long books. Short 3-month persist is *lower* (0.73), not higher. Do not promote a 3-month liq forecast (already PARK).

## 6. Q1 last-value honesty (description only)

Last grid month per train company — the month-24 still the brief warned about. Not a model. Not a Y.

| series | n | null | p10 | p25 | p50 | p75 | p90 | share <0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| last `b_liq` | 1195 | 1.6% | 203.5 | 1.345e+04 | 8.202e+04 | 4.661e+05 | 2.056e+06 | 4.7% |
| last `b_runway` | 1195 | 1.6% | 0.001 | 0.129 | 1.079 | 10.804 | 24.000 | 4.7% |

Last-month runway < 1: 49.1%. Last-month stressed (`b_liq<0` or `b_runway<1`) among non-null liq: 49.1%. Runway sitting on the clip walls: floor −6 0.7%, ceiling 24 20.3%.
Company-month (all train): `b_liq` < 0 7.8%; runway < 1 48.3%; stressed 43.5%.

Do **not** predict Y2/Y3 with Family B. Y2 is already 2-of-3 negative reconstructed months; Y3 is stressed-only recovery. Using B as X is the contract leak.

## 7. Holdout coverage (count only)

| item | n |
| --- | ---: |
| holdout companies | 72 |
| with a balance row | 72 (100.0%) |
| with a cash-walk snapshot | 72 (100.0%) |
| holdout CM | 1,073 |
| `b_liq` non-null CM | 100.0% |
| companies with any `b_liq` | 72 |

## 8. Zombie snapshot cash

Train cash-walk products with **zero** txs in the extract: 538 / 4,408 = 12.2%. |zombie| / |cash snap| = 0.4% (zombie signed sum 1.648e+07 vs cash 7.204e+09). Companies with any zombie cash product: 271. Companies whose *entire* cash snapshot is zombie: 2 (signed sum 3.46e+05).

| type | products | zombies | zombie share | zombie bal | zombie |bal| | all bal |
| --- | ---: | ---: | --- | --- | --- | --- |
| checking | 4373 | 533 | 12.2% | 1.648e+07 | 2.663e+07 | 7.122e+09 |
| saving | 10 | 3 | 30.0% | 100 | 100 | 8.228e+07 |
| tpv | 25 | 2 | 8.0% | 0 | 0 | 0 |

Zombie cash is a still of unused accounts, not a trail. Small |share| means it does not move company-sum `b_liq` much — but a dead checking account can hide inside a live sum.

## 9. Is `b_bal_vol` SIZE?

Train CM coverage 70.7% (6-month roll). p50=0.525, p90=8.150, share>1 35.0%. Spearman vs log1p(a_in3)=-0.282; vs a_in3=-0.282; vs `b_liq`=0.231; vs |b_liq|=0.251. Last-month vs log size -0.229.
**not SIZE** — vol is not company scale. Still not a Y (it is a roll of the same still-walked path).

## 10. Negatives on late books (left-trunc + still)

Train CM `b_liq<0`: 1,640 (7.8%). Late-book months *before* first tx: 0 CM / 0 companies; neg —; constant reconstructed path —. Late on-book neg 7.4%; on-time-book neg 8.1%.
First-month neg: late 8.9% vs on-time 11.5%. Late last-month neg 4.4%.

Negative share by calendar month (train):

| month | n CM | n neg | share neg | n late-book CM |
| --- | ---: | ---: | --- | --- |
| 2024-09 | 435 | 50 | 11.5% | 0 |
| 2024-10 | 473 | 53 | 11.2% | 38 |
| 2024-11 | 483 | 52 | 10.8% | 48 |
| 2024-12 | 525 | 48 | 9.1% | 90 |
| 2025-01 | 636 | 69 | 10.8% | 201 |
| 2025-02 | 677 | 70 | 10.3% | 242 |
| 2025-03 | 714 | 68 | 9.5% | 279 |
| 2025-04 | 733 | 65 | 8.9% | 298 |
| 2025-05 | 752 | 62 | 8.2% | 317 |
| 2025-06 | 762 | 65 | 8.5% | 327 |
| 2025-07 | 796 | 68 | 8.5% | 361 |
| 2025-08 | 833 | 62 | 7.4% | 398 |
| 2025-09 | 864 | 64 | 7.4% | 429 |
| 2025-10 | 908 | 71 | 7.8% | 473 |
| 2025-11 | 945 | 70 | 7.4% | 510 |
| 2025-12 | 1006 | 76 | 7.6% | 571 |
| 2026-01 | 1132 | 94 | 8.3% | 697 |
| 2026-02 | 1204 | 88 | 7.3% | 769 |
| 2026-03 | 1211 | 96 | 7.9% | 776 |
| 2026-04 | 1212 | 93 | 7.7% | 777 |
| 2026-05 | 1214 | 81 | 6.7% | 779 |
| 2026-06 | 1214 | 59 | 4.9% | 779 |
| 2026-07 | 1214 | 60 | 4.9% | 779 |
| 2026-08 | 1214 | 56 | 4.6% | 779 |

The official monthly grid **starts at first tx** — there are no pre-connection company-months to hold a constant opening photograph. Left-truncation was already applied by the grid, not by Family B. See §12 for early-on-book (age 0–2) negatives.

## 11. Why short persist is lower

Still-leak predicted short books *more* like the snapshot. Observed Spearman t vs t+3 is **lower** on short (0.730) than on 24-month books. After dropping the first 3 on-book months: short 0.736 (n=830) vs long 0.866. Dropping birth months does not close the short-vs-long gap — short books are noisier, not flatter.
First-month vs snap ranking: late arrivals 0.619 vs on-time 0.606 (later arrivals do **not** rank more like the still). First→second month |Δliq|/|liq| p50: short 0.297 vs not-short 0.157.

| trail | months used | n pairs | n companies | Spearman t,t+3 |
| --- | --- | ---: | ---: | --- |
| <6 | all_months | 4 | 3 |  |
| <6 | age>=3 | 0 | 0 |  |
| 6-11 | all_months | 1829 | 333 | 0.729 |
| 6-11 | age>=3 | 830 | 326 | 0.736 |
| 12-17 | all_months | 1626 | 147 | 0.861 |
| 12-17 | age>=3 | 1185 | 147 | 0.854 |
| 18-23 | all_months | 4804 | 279 | 0.859 |
| 18-23 | age>=3 | 3967 | 279 | 0.858 |
| 24 | all_months | 9093 | 433 | 0.862 |
| 24 | age>=3 | 7794 | 433 | 0.866 |

First-month vs snapshot by first-tx calendar (train):

| first tx | n | ρ first vs snap | p50 |first−snap|/|snap| | first-month neg |
| --- | ---: | --- | --- | --- |
| 2024-09 | 435 | 0.6062 | 0.8173 | 11.5% |
| 2024-10 | 38 | 0.1579 | 1.029 | 13.2% |
| 2024-11 | 10 | 0.9394 | 0.4881 | 0.0% |
| 2024-12 | 42 | 0.7238 | 0.8662 | 9.5% |
| 2025-01 | 111 | 0.6214 | 0.8028 | 9.0% |
| 2025-02 | 41 | 0.6854 | 0.7226 | 4.9% |
| 2025-03 | 37 | 0.5556 | 0.6599 | 2.7% |
| 2025-04 | 19 | 0.5316 | 0.6774 | 10.5% |
| 2025-05 | 19 | 0.6351 | 0.8693 | 26.3% |
| 2025-06 | 10 | 0.4788 | 0.9448 | 0.0% |
| 2025-07 | 34 | 0.5852 | 0.6602 | 2.9% |
| 2025-08 | 37 | 0.887 | 0.5066 | 5.4% |
| 2025-09 | 31 | 0.6417 | 0.8853 | 12.9% |
| 2025-10 | 44 | 0.7147 | 0.9072 | 4.5% |
| 2025-11 | 37 | 0.3569 | 1.321 | 13.5% |
| 2025-12 | 61 | 0.6247 | 0.8202 | 6.6% |
| 2026-01 | 126 | 0.4729 | 0.7559 | 13.5% |
| 2026-02 | 72 | 0.7809 | 0.6879 | 5.6% |
| 2026-03 | 7 |  | 0.2932 | 0.0% |
| 2026-04 | 1 |  | 0.3336 | 0.0% |
| 2026-05 | 2 |  | 1.437 | 50.0% |

## 12. Early-on-book negatives (age, not pre-tx)

monthly grid starts at first_tx — there are no pre-tx company-months.
 `g_n_accounts=0` months that are `b_liq<0`: 10.4% vs connected months 7.4%.

| book | age | n CM | n companies | share neg |
| --- | --- | ---: | ---: | --- |
| late | age0 | 779 | 779 | 8.9% |
| late | age1-2 | 1558 | 779 | 10.4% |
| late | age3-5 | 2332 | 779 | 8.4% |
| late | age6+ | 6048 | 769 | 6.0% |
| ontime | age0 | 435 | 435 | 11.5% |
| ontime | age1-2 | 870 | 435 | 10.8% |
| ontime | age3-5 | 1305 | 435 | 9.8% |
| ontime | age6+ | 7830 | 435 | 7.4% |

First 3 vs last 3 months within company:

| slice | n | n head CM | head neg | n tail CM | tail neg |
| --- | ---: | --- | --- | --- | --- |
| late | 779 | 2337 | 9.9% | 2337 | 4.3% |
| short_<12 | 350 | 1050 | 10.5% | 1050 | 5.8% |
| long_24 | 435 | 1305 | 11.0% | 1305 | 5.7% |
| ontime | 435 | 1305 | 11.0% | 1305 | 5.7% |

Late books are more negative in their first three months than their last three — **but so are 24-month on-time books**. That is cash accumulating toward a healthier extract still, not late-arrival left-trunc (late age0 8.9% is *below* on-time age0 11.5%). **CLOSE** early-grid `b_below_0` as Q6. See §17.

## 13. Runway < 1 anatomy

Last-month train, non-null: runway<1 48.4% of which `liq>=0` 44.4% and `liq<0` 4.7%. Spearman(runway, log1p a_in3)=-0.386 (not SIZE). Implied monthly burn among interior (unclipped) runways: p10=1.2e+04, p50=2.214e+05.

| group | n | share of nn | liq p50 | runway p50 |
| --- | ---: | --- | --- | --- |
| all_nn | 1195 | 100.0% | 8.202e+04 | 1.079 |
| rw<1 | 587 | 49.1% | 2.992e+04 | 0.1239 |
| liq<0 | 56 | 4.7% | -1.015e+05 | -0.2467 |
| liq>=0 & rw<1 | 531 | 44.4% | 3.805e+04 | 0.1576 |
| rw=24 wall | 243 | 20.3% | 9.044e+04 | 24 |
| rw=-6 wall | 8 | 0.7% | -7.666e+05 | -6 |

Half the last-month book is `runway<1` mostly because **positive cash over a fat 3-month burn**, not because 49% are below zero (only ~5% are). The 24-month clip wall is the other tail (idle / tiny out3). This is a description of the still, not a license to invent a runway Y.

## 14. Coverage holes

Tx but no `balances` row: 13 train companies (`b_liq` all-null on last month: True). Balance row but no cash-walk snapshot: 6.

| company | kind | last g_n_accounts | last b_liq | last a_in3 |
| --- | --- | --- | --- | --- |
| COMP_0033 | tx_no_balance | 3 |  | 3.202e+05 |
| COMP_0093 | tx_no_balance | 1 |  | 4.086e+04 |
| COMP_0166 | tx_no_balance | 2 |  | 5.39e+05 |
| COMP_0312 | tx_no_balance | 1 |  | 3.864e+04 |
| COMP_0327 | tx_no_balance | 1 |  | 0 |
| COMP_0715 | tx_no_balance | 1 |  | 0 |
| COMP_0720 | tx_no_balance | 1 |  | 0 |
| COMP_0784 | tx_no_balance | 1 |  | 1.96e+05 |
| COMP_0800 | tx_no_balance | 1 |  | 5.502e+04 |
| COMP_0889 | tx_no_balance | 1 |  | 0 |
| COMP_0938 | tx_no_balance | 1 |  | 0 |
| COMP_0976 | tx_no_balance | 3 |  | 2.622e+06 |
| COMP_1285 | tx_no_balance | 1 |  | 7.521e+04 |
| COMP_0676 | balance_no_cash_walk | 0 |  | 1.116e+05 |
| COMP_0683 | balance_no_cash_walk | 0 |  | 4.905e+06 |
| COMP_0906 | balance_no_cash_walk | 0 |  | 0 |
| COMP_0962 | balance_no_cash_walk | 2 |  | 0 |
| COMP_1121 | balance_no_cash_walk | 3 |  | 0 |
| COMP_1254 | balance_no_cash_walk | 3 |  | 1.601e+05 |

No-cash-walk companies — what sits in `balances`:

| company | type | n | sum bal |
| --- | --- | ---: | --- |
| COMP_0676 | lineofcredit | 1 | 2.737e+04 |
| COMP_1121 | loan | 2 | -2.814e+04 |
| COMP_1254 | wallet | 3 | 2.459e+04 |
| COMP_1254 | (unknown) | 1 | 0 |
| COMP_0683 | lineofcredit | 5 | -2.239e+06 |
| COMP_0906 | loan | 1 | -1.818e+05 |
| COMP_0962 | card | 1 | -48.45 |
| COMP_0906 | lineofcredit | 1 | 1.203e+04 |

## 15. Last month vs snapshot = September flows

Train last-grid-month (2026-08) with a cash snap: 1,195. median |snap − b_liq| = 200; share |diff|<1€ 40.3%. ρ(last, snap)=0.966. | (snap−b_liq) − Sep cash flow | median 5.826e-13, max 9.313e-09. Companies with no Sep cash-product flow: 477.
Last-value Q1 **is** the still, minus extract-month txs. That is why last-value wins Y1 liq and why a snapshot Y is PARK.

## 16. Last-value = the still

Company last month vs cash snapshot: n=1,195, ρ=0.966, p50 |liq−snap|/|snap|=0.001492, p90=0.342. Last month ≠ 2026-08: 0. **Yes — last-value Q1 is the extract still.**

## 17. Accumulation toward the still (not late-arrival red)

Share of train companies with last `b_liq` > first `b_liq`: 48.6%. On the 24-month cohort, calendar neg share 11.5% → 5.1% (declines inside the same 435 companies). Last>first is a coin flip — the red decline is a *tail* recovery, not a richer book. 24-month crossings: neg→pos 38, pos→neg 10, stayed neg 12, stayed pos 373 (n=433).

| slice | n | share last>first | first-month neg | last-month neg |
| --- | ---: | --- | --- | --- |
| all | 1195 | 48.6% | 10.0% | 4.7% |
| long_24 | 433 | 48.7% | 11.5% | 5.1% |
| late | 762 | 48.6% | 9.1% | 4.5% |
| short_<12 | 336 | 44.3% | 9.8% | 6.8% |

Neg share on the **same** 24-month train companies (composition held fixed):

| month | n CM | n neg | share neg |
| --- | ---: | ---: | --- |
| 2024-09 | 435 | 50 | 11.5% |
| 2024-10 | 435 | 48 | 11.0% |
| 2024-11 | 435 | 46 | 10.6% |
| 2024-12 | 435 | 38 | 8.7% |
| 2025-01 | 435 | 45 | 10.3% |
| 2025-02 | 435 | 45 | 10.3% |
| 2025-03 | 435 | 43 | 9.9% |
| 2025-04 | 435 | 39 | 9.0% |
| 2025-05 | 435 | 35 | 8.0% |
| 2025-06 | 435 | 38 | 8.7% |
| 2025-07 | 435 | 37 | 8.5% |
| 2025-08 | 435 | 31 | 7.1% |
| 2025-09 | 435 | 30 | 6.9% |
| 2025-10 | 435 | 31 | 7.1% |
| 2025-11 | 435 | 33 | 7.6% |
| 2025-12 | 435 | 33 | 7.6% |
| 2026-01 | 435 | 31 | 7.1% |
| 2026-02 | 435 | 27 | 6.2% |
| 2026-03 | 435 | 33 | 7.6% |
| 2026-04 | 435 | 30 | 6.9% |
| 2026-05 | 435 | 33 | 7.6% |
| 2026-06 | 435 | 27 | 6.2% |
| 2026-07 | 435 | 25 | 5.7% |
| 2026-08 | 435 | 22 | 5.1% |

## 18. Runway persist vs liq persist

Spearman t vs t+3: `b_liq` 0.848 vs `b_runway` 0.757. Runway is **not** stickier than `b_liq` — last-value Q1 should quote the stock (or both), not treat runway as a smoother path.

| slice | series | n pairs | Spearman t,t+3 |
| --- | --- | ---: | --- |
| all | b_liq | 17356 | 0.848 |
| all | b_runway | 14968 | 0.757 |
| short_<12 | b_liq | 1833 | 0.730 |
| short_<12 | b_runway | 1163 | 0.669 |
| long_24 | b_liq | 9093 | 0.862 |
| long_24 | b_runway | 8227 | 0.763 |

## 19. The 13 with txs and no `balances` row

Cash-type txs on those 13: 8,645 txs / 13 companies. Family B **misses** cash-product history that has no extract snapshot — those `b_liq` stay null. Not a liquidity.py rewrite (no snapshot to walk from).

| type | n tx | n companies | sum amount |
| --- | ---: | ---: | --- |
| checking | 8645 | 13 | 1.243e+06 |

## 20. Timing of the 13 unphotographed checking books

Cash products on the 13: 18 (unphotographed 18). Last cash-tx ≥30d before extract: 12/13; ≥90d quiet: 4; still live in the last 30d: 1. Median last cash-tx 2026-07-20.

| company | cash products | n tx | first tx | last tx | quiet ≥30d | sum amount |
| --- | ---: | ---: | --- | --- | --- | --- |
| COMP_0033 | 3 | 792 | 2025-10-19 | 2026-06-17 | True | 6.51e+04 |
| COMP_0093 | 1 | 124 | 2025-12-19 | 2026-07-20 | True | -1128 |
| COMP_0166 | 2 | 131 | 2025-12-19 | 2026-07-20 | True | 6513 |
| COMP_0312 | 1 | 1684 | 2025-12-22 | 2026-08-07 | False | -6.753e+04 |
| COMP_0327 | 1 | 962 | 2025-10-18 | 2026-06-09 | True | 1.466e+05 |
| COMP_0715 | 1 | 63 | 2025-07-08 | 2026-01-28 | True | -1.466e+04 |
| COMP_0720 | 1 | 751 | 2025-09-29 | 2026-05-25 | True | 1.713e+05 |
| COMP_0784 | 1 | 140 | 2025-12-31 | 2026-07-20 | True | -601.3 |
| COMP_0800 | 1 | 278 | 2025-12-19 | 2026-07-20 | True | -4725 |
| COMP_0889 | 1 | 1806 | 2024-09-30 | 2026-05-15 | True | 3.75e+04 |
| COMP_0938 | 1 | 722 | 2024-09-03 | 2025-09-02 | True | 5.104e+05 |
| COMP_0976 | 3 | 1079 | 2025-12-18 | 2026-07-20 | True | 3.948e+05 |
| COMP_1285 | 1 | 113 | 2025-12-31 | 2026-07-20 | True | -281 |

Some of the 13 still transact in the last 30 days and have no photograph. That is an extract hole, not a closed account. Still **do not** invent a 0-snapshot in `liquidity.py`.

## 21. Does the path approach the still?

|b_liq − snap| is |remaining after-flows|. Share of month-to-month steps that get *closer* to the snapshot: 51.8% (farther 38.2%). Company-level p50 of that share 0.545; majority-approach companies 59.2%.

| slice | n | n steps | closer | farther |
| --- | ---: | --- | --- | --- |
| all | 1195 | 19746 | 51.8% | 38.2% |
| short_<12 | 336 | 2505 | 55.5% | 39.2% |
| long_24 | 433 | 9959 | 50.7% | 39.1% |

The walk oscillates — remaining flow sums do not shrink every month. That is a real cash path, not a monotone zoom into the still.

## 22. Custom checking inside the walk

Train cash-walk |snapshot|: custom service 101 companies / 47 all-custom, |share| 11.2% (8.295e+08 of 7.391e+09). Family B already walks these (type filter only). Not a hole. Not a G redo.

Largest custom cash snapshots (train):

| company | custom |bal| | cash |bal| |
| --- | --- | --- |
| COMP_1105 | 6.278e+08 | 6.278e+08 |
| COMP_0469 | 7.372e+07 | 7.372e+07 |
| COMP_1141 | 3.721e+07 | 3.75e+07 |
| COMP_0573 | 1.75e+07 | 1.75e+07 |
| COMP_1018 | 1.52e+07 | 1.52e+07 |
| COMP_1003 | 9.032e+06 | 9.032e+06 |
| COMP_0828 | 6.682e+06 | 6.682e+06 |
| COMP_0283 | 6.162e+06 | 6.354e+06 |

## 23. Company-median vs last (is the typical month the still?)

Train companies with a cash path n=1,195. ρ(median, last)=0.808; ρ(median, snap)=0.804; ρ(median, first)=0.814; ρ(first, last)=0.613. p50 |median−last|/|last|=0.438.
The typical month is **not** the still (ρ(median,last) < 0.90). Last-value Q1 is the extract photograph; the rest of the book is a different ranking. That is a real trail, not everyone being the 2026-09 still.

## 24. Negative-month stickiness (not a Y2 model)

Train companies ever `b_liq<0`: 242 (19.9%). 405 negative runs; length p50=2, p90=10; share of runs that are a single month 42.5%; share ≥3 months 45.7%. P(neg next | neg now)=0.780; P(enter | ok)=0.016.
Negative months tend to come in streaks. That is already Y2's 2-of-3; do not invent a cousin.

## 25. September flow vs snapshot

Train last-month companies with a cash snap: 1,195. median |Sep cash flow|=200; p50 |Sep|/|snap|=0.001; p90=0.342; share >10% of |snap| 19.3% (n=231); zero Sep flow 477. ρ(last, snap) all=0.966; fat-Sep 0.799; thin-Sep 1.000. |Sep|/|snap| vs log size ρ=0.418 (not SIZE).
Last-value differs from the still by a typically small extract-month flow. On the fat-Sep fifth, ranking peels (0.80 vs 1.00). Q1 last-value is the still *plus* extract-month noise, not a SIZE artifact.

## 26. Unique pile companies + last-tx cluster on the 13

Unique companies with a walk row: 1,267 (train 1,195). Walk ∩ debt-orphan: 370 (same company can sit in both piles). Train excluded-only (balance but no cash walk): 6.

| pile | unique companies | of which train |
| --- | ---: | ---: |
| walk | 1267 | 1195 |
| bank_excluded | 284 | 272 |
| debt_orphan | 374 | 354 |
| unknown_orphan | 19 | 16 |
| any_excluded | 495 | 470 |
| walk ∩ debt_orphan | 370 | 350 |
| walk-only (no excluded row) | 778 | 731 |
| excluded-only (no walk) | 6 | 6 |

Last cash-tx on **2026-07-20**: 46.2% of the 13 holes vs 0.3% of train photographed walk. That date is a hole stamp — not the typical photographed last-tx. Still do not invent a 0-snapshot. Distinct banks on the 13 cash products: 9. Santander (Corporate UK) on 6 of the 13; 6/6 of the 2026-07-20 last-tx books are that bank — the cluster is one bank extract.

| slice | n | median last cash-tx | n on 2026-07-20 | share 07-20 | n in 2026-07 | share Jul | n in 2026-08 | share Aug |
| --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| 13 holes | 13 | 2026-07-20 | 6 | 46.2% | 6 | 46.2% | 1 | 7.7% |
| train photographed walk | 1194 | 2026-09-01 | 4 | 0.3% | 36 | 3.0% | 298 | 25.0% |
| train any cash-tx | 1209 | 2026-09-01 | 10 | 0.8% | 42 | 3.5% | 299 | 24.7% |

Last cash-tx dates on the 13:

| last cash-tx | n |
| --- | ---: |
| 2026-07-20 | 6 |
| 2025-09-02 | 1 |
| 2026-05-15 | 1 |
| 2026-01-28 | 1 |
| 2026-05-25 | 1 |
| 2026-06-09 | 1 |
| 2026-06-17 | 1 |
| 2026-08-07 | 1 |

Cash-product banks on the 13 (not a G redo):

| bank | type | n companies | n products |
| --- | --- | ---: | ---: |
| Santander (Corporate UK) | checking | 6 | 9 |
| ING | checking | 2 | 2 |
| Crédit Industriel et Commercial | checking | 1 | 1 |
| Banca Sella | checking | 1 | 1 |
| ABN AMRO Bank N.V. | checking | 1 | 1 |
| Commerzbank AG | checking | 1 | 1 |
| BNP Paribas - Corporate | checking | 1 | 1 |
| Societe Generale Entreprises | checking | 1 | 1 |
| BNL Corporate | checking | 1 | 1 |

Do not rewrite `liquidity.py` around a 2026-07-20 cutoff. Extract hole, not a walk identity bug.

## PARK / CLOSE / KEEP

| object | decision | why |
| --- | --- | --- |
| reconstruction walk | **CLOSE** | identity; median |resid| is noise; do not rewrite `liquidity.py` |
| snapshot as a Y | **PARK** | would score the 2026-09 photograph |
| last-value `b_liq` / `b_runway` as Q1 description | **KEEP** | last-vs-snap ≈ 0.97 (thin-Sep = 1.00; fat-Sep fifth 0.80); first-vs-snap ≈ 0.61; persist 0.85 is a real path |
| last-value as a 3-month forecast (Y1 liq) | **PARK** (CONFIRM) | already lost to last-value on CV OOF |
| Family B as X for Y2 / Y3 | **forbidden** | contract; Y2/Y3 *are* the B path |
| invent a Y from `b_liq` / `b_runway` / neg-run length | **PARK** | circular 16h death; Y2 already is the streak |
| zombie cash as a health flag | **PARK** | unused-account still; small |share| |
| `b_bal_vol` as X | **not SIZE**; still a roll of the walk | same still-walked path |
| early-grid negatives on late books as Q6 | **CLOSE** | late age0 is *less* red than on-time; head>tail red is accumulation on every cohort |
| snapshot / last-value as a *forecast* Y | **PARK** | last month *is* the still minus Sep flows |
| rewrite `liquidity.py` for the 13 nulls | **no** | they have checking txs but no extract still; walking from 0 would invent a snapshot |
| 2026-07-20 last-tx as a second still / cutoff Y | **PARK** | hole timing, not a health signal |
| redo Family G | **no** | connection clock already closed |

## Plot

- `analysis/outputs/balances_b_residual.png` — identity residual hist + recovered vs snapshot (train sample).

## Closed in this module

- Raw still: almost all 2026-09-01; 16 closest-prior-day rows.
- Identity of the walk: measured on train company-months.
- Excluded snapshot cash: debt orphans + named bank types.
- 63.7% first-month G hole is not a null `b_liq`.
- Spearman t vs t+3 on short vs 24-month books.
- Q1 last-value distribution; no B→Y2/Y3 model.
- Zombies / vol-SIZE / left-trunc negatives.
- Short persist is lower, not higher; first-vs-snap is not the still.
- Runway<1 is fat burn over positive cash, not 49% below zero.
- Last-value minus snapshot = September cash-product flows.
- Head>tail red is a shrinking *negative tail* on the 24-month cohort (last>first is a coin flip).
- Runway persist vs liq persist; 13 no-balance checking books (no still → no walk).
- Approach-to-still rate; custom checking already inside the walk.
- Typical month ≠ still (median vs last); neg-run lengths; Sep flow size.
- Unique pile n_co (pass-3 type-sum overcount); 2026-07-20 last-tx vs photographed books.

