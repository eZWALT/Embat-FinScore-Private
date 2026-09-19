# Wave 4 — Y5 why (end note)

Agent `234af73a`. Lane: `analysis/evaluate/y5_why.py`. No commit. No parquet/duckdb rewrite. No `product/`. No 0–100. No GBM/XGB. Did not run `build_targets`. Never E as X.

## Files written

- `analysis/evaluate/y5_why.py` — create; write→run cuts 1–18 in this module
- `analysis/outputs/y5_why.md` — tables + KEEP/CLOSE/PARK
- `analysis/outputs/y5_why_quintiles.png` — AP `h_group_size` / AR `d_tx_cp_share` quintiles
- `overnight/waves/wave4_y5_why.md` — this note
- `analysis/experiments/registry.csv` — append-only (agent `234af73a`)

Did not edit sibling_h, y9_why, banking_g_qa, xgb_panel, invoices, parquet, product/.

## Verdict

Y5 is **not cash-stress** and **not a supplier/customer HHI tail**. 65% of positives sit in the 2×2 neither cell. `d_supp_hhi` monopoly (>0.975) is **protective** (2.7% vs 8.6%) — the unused Y4 counterpart does not copy. Leftover positives still have a tx (days>0 99–100%; median 17–19). CLOSE “died because cash was missing”. Cannot use E to prove last-month overdue persistence (|ρ| vs e_* < 0.18).

**AP CLOSE.** Best non-E = night `h_group_size` CV **0.581** (train 0.599). Beats size +0.025 but not monotone (size≥18 is 2.9% vs rest 10.4%). Trees stay PARK.

**AR KEEP** a Q5 sentence on `d_tx_cp_share` (the night D single, train 0.611 / CV **0.576**, +0.108 vs size 0.469). Head-only: Q1 12.8%. Not a zero rewrite (cp>0 CV 0.580). Cut 18: thick invoice book + unnamed bank trail **17.4%** vs named **6.3%** — tagging hole, not invoice thinness. AP analogue is **false** (10.9% vs 10.1% with many suppliers).

**Q6 CLOSE.** Lag1 `d_tx_cp_share` 0.563 all-train; short so-far<12 **0.431** (75 pos). Do not transfer a quarter lead. Thin-F first-run winners stay PARK (1.7% panel).

Holdout coverage only: AP 195/14, AR 165/8, LOW_POWER. Not an AUROC claim.

## Q5 sentence (no family E)

Sustained AR overdue-30 is higher when the bank trail names almost no counterparties even though the invoice book is thick (17.4% vs 6.3%). AP overdue is leftover: not low cash, not supplier monopoly.

## Next idea (parent)

Do not put E on Y5. Do not revive XGB Y5. Optional: Family J match rates as a Q5 footnote on the tagging hole — not this owner.

---

## End note (cuts 19–34)

Same owner. Write→run in `y5_why.py` only. Registry appended again (246 rows).

AP analogue tagging hole is **false** (10.9% vs 10.1%). Hole × cash is independent (16.2% vs 7.3% named-ok). Hole vs E max |ρ|=0.105. 27% of AR pos; 28 companies; top3=23% (not a 3-firm clique). Hi-cust Q1 **19.5%** vs Q5 3.4%. `d_tx_cp_share` is not size (ρ vs `a_in3` −0.063). Extra AP D columns (`d_n_supp`, `d_supp_top1`) are SIZE_PARK.

**Fold honesty:** fold 3 owns 55/64 hole pos. Without it 7.8% vs 7.0%. Fold 3 is 111 companies, median named-cp share 0.066, 33% zeros. Fold 3 zero+hi-cust **25.1%** (48/191); fold 2 zero+hi **3.3%** (2/60). KEEP the night single; CLOSE 17.4% as a leave-one-group law. Q6 still CLOSE (hole_lag1 short 0.480). Holdout hole n=54 pos=7 — coverage only.

### Q5 sentence (no family E)

Sustained AR-od30 is higher when the bank trail names no counterparties on a thick invoice book — in the unnamed company-group (fold 3: 25.1% on 191 months), not as a leave-one-group panel law. AP leftover: not low cash, not supplier monopoly.

Cut 35: AP leftover is also group-clustered (fold rates 4.4–12.3%), not a cash/HHI why.

Cut 36: leftover *share of AP positives* is 58–71% in every fold — composition is leftover everywhere; only the event rate clusters.

Cut 37: fold 3 AR leftover share is still 67% — the unnamed cluster is leftover on cash/HHI, not a cash rewrite.

Cut 38: 28/64 hole positives (44%) sit in the leftover neither cell (below the 65% AR-pos base) — hole is leftover-adjacent, not a cash rewrite.

Cut 39: among 52 hole pos with cash+HHI flags: neither=28 cash_only=11 hhi_only=8 both=5 — leftover is still the largest cell.

Cut 40: fold 3 median so-far is 14 months, same as the other folds — not a long-book artifact.

Cut 41: AP high-rate folds are not longer books (med so-far 12–14). Leftover AP is not a tenure rewrite.

Cut 42: months-so-far as a single is CV 0.546 AP / 0.560 AR — not a tenure rewrite; not SIZE_PARK.

Cut 43: AP `h_group_size`≥18 is protective in every fold that has large groups (0.5–7.0% vs rest 7–16%). Not a why for the positives. CLOSE.

`y5_why.md` now ends with a Return block: leftover, night singles vs this-run CV, Q5 sentence with fold-3 honesty. PNG `y5_why_quintiles.png`. No commit.
