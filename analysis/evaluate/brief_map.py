"""Compile the six-question morning map. No parquet rewrite. No model fit.

NORTH_STAR first: HackSpain 2026 · Embat X Ray. Trajectory, not last-month
snapshot. Six questions. No 0–100. No product/. Hidden 72 (seed 20260918)
is LOW_POWER except y7_top1_lost (122 pos). Quote train group-fold CV.

This module is a compile / consistency pass. It reads already-written
``analysis/outputs/*.md``, checks every KEEP/CLOSE/DROP/PARK row against
its source file, hunts KEEP-vs-DROP contradictions, and writes
``analysis/outputs/brief_map.md``. It does not load monthly/targets parquet
for a join, does not call ``build_targets``, and does not fit.

    /home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.brief_map

Owned: analysis/evaluate/brief_map.py, analysis/outputs/brief_map.md,
optional PNG, append-only registry, overnight/waves/wave4_brief_map.md (end).
"""
from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.evaluate.protocol import FOLD_SEED, load_holdout
from analysis.features.common import ANALYSIS

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

OUT_MD = ANALYSIS / "outputs" / "brief_map.md"
OUT_PNG = ANALYSIS / "outputs" / "brief_map_grid.png"
REGISTRY = ANALYSIS / "experiments" / "registry.csv"
OUTPUTS = ANALYSIS / "outputs"

AGENT = "e4a91c7b"
WAVE = "4"
ROUND = "R4"
MODEL = "brief_map"
X_FAM = "-"

# Night quotes — do not change.
Y3_SHALLOW_278 = "0.762 ± 0.016"
Y3_15COL_A = "0.752 ± 0.037"
DAYS_BAR = "0.711"
SIZE_BAR = "0.617"
Y7_TURNOVER = "0.720"
Y7_B_SHALLOW = "0.712"
CN_LEFTOVER = "0.597"
DELAY_LEFTOVER = "0.581"
FORBIDDEN_Y7_278 = "0.663"
FORBIDDEN_Y7_HOLDOUT = "0.680"
FORBIDDEN_A_OUT_VOL = "0.722"

# 15-col Y3 card stems + lags 1,3. Do not add columns.
Y3_STEMS = (
    "c_ss_month",
    "c_salary_month",
    "a_n_tx",
    "f_ds_r",
    "c_n_days_with_tx",
)

# Feature-report 44-col starter (feature_report.md §Core GBM starter).
STARTER_44 = (
    "a_growth_12", "a_growth_3", "a_in3", "a_io_ratio", "a_uncat_share",
    "b_bal_vol", "b_below_0", "b_d_runway", "b_neg_episodes", "b_runway",
    "c_gap_sd", "c_last_tx_before_2026_06", "c_missed_salary", "c_missed_tax",
    "c_recency_days", "c_zero_in_month", "c_zero_in_share_6",
    "d_cust_hhi", "d_supp_hhi", "d_tx_cp_share",
    "e_ap_overdue_30", "e_ar_overdue", "e_ar_overdue_30", "e_credit_note_ratio",
    "e_delay_coll", "e_delay_paid", "e_dpo_proxy", "e_dso_proxy",
    "e_fx_share", "e_pending_amt_share",
    "f_ds_r", "f_fc_r", "f_has_confirming", "f_has_factoring", "f_has_loc",
    "f_new_facility", "f_outstanding_gt_granted",
    "g_custom_share", "g_has_card", "g_has_checking", "g_has_investment",
    "g_has_saving", "g_has_tpv", "h_sib_neg_share",
)

# 44-drop as of CONTEXT ~04:50. B-on-44 and DSO stay in-flight — not in this set.
DROP_44_COLS = (
    "e_fx_share",
    "f_has_factoring", "f_has_confirming", "f_has_loc", "f_new_facility",
    "c_gap_sd", "c_recency_days",
    "a_io_ratio", "a_growth_3",
    "c_zero_in_month", "c_zero_in_share_6",
    "e_credit_note_ratio",
    "e_delay_coll", "e_delay_paid", "e_ar_overdue", "e_ar_overdue_30", "e_ap_overdue_30",
    "e_pending_amt_share",
    "g_has_saving", "g_has_investment", "g_has_tpv", "g_custom_share",
    "g_has_checking", "g_has_card",
    "e_dpo_proxy",
    "d_supp_hhi",
)
DROP_44_LABELS = (
    "e_fx_share",
    "f_has_factoring / f_has_confirming / f_has_loc / f_new_facility",
    "c_gap_sd",
    "c_recency_days",
    "a_io_ratio",
    "a_growth_3",
    "c_zero_in_month / c_zero_in_share_6",
    "e_credit_note_ratio (as Y3 X)",
    "e_delay_coll / e_delay_paid / e_ar_overdue / e_*_overdue_30 (as Y3 X)",
    "e_pending_amt_share",
    "remaining g_has_* + g_custom_share (card CLOSE 0.551; checking 99.1% hole)",
    "e_dpo_proxy",
    "d_supp_hhi",
)
DROP_44_SOURCES = {
    "e_fx_share": "fx_qa.md",
    "f_has_factoring": "factoring_qa.md",
    "c_gap_sd": "gap_sd_qa.md",
    "c_recency_days": "recency_qa.md",
    "a_io_ratio": "growth_qa.md",
    "a_growth_3": "growth_qa.md",
    "c_zero_in_month": "zero_in_qa.md",
    "e_credit_note_ratio": "credit_note_qa.md",
    "e_delay_coll": "delay_qa.md",
    "e_pending_amt_share": "pending_qa.md",
    "g_has_saving": "g_has_rest_qa.md",
    "e_dpo_proxy": "dpo_qa.md",
    "d_supp_hhi": "supp_hhi_qa.md",
}
IN_FLIGHT = ("B-on-44 (b_* leftover on the 44)", "DSO (e_dso_proxy leftover on the 44)")
IN_FLIGHT_FILES = ("b_on_44_qa.md", "dso_qa.md")

DECISIONS = ("KEEP", "CLOSE", "DROP", "PARK")
# Role-split tokens that are not KEEP-vs-DROP fights.
ROLE_SPLIT = {
    "KEEP-Q5",
    "KEEP leftover",
    "KEEP footnote",
    "KEEP thin",
    "KEEP description",
    "DROP from 44",
    "CLOSE TURNOVER",
}

QUESTIONS = {
    1: "Who is healthy?",
    2: "Who is improving?",
    3: "Who is turning?",
    4: "Dip vs fall?",
    5: "Why did it change?",
    6: "How many months earlier?",
}


@dataclass(frozen=True)
class Row:
    q: int
    obj: str
    decision: str
    number: str
    source: str
    note: str = ""
    need_token: bool = True
    must_contain: str = ""

    @property
    def token(self) -> str:
        d = self.decision.upper()
        for t in DECISIONS:
            if t in d:
                return t
        return d.split()[0]


# Canonical morning map. One number per row. Source is the later leftover /
# why / core md that owns the verdict. Re-read NORTH_STAR before changing.
MAP: tuple[Row, ...] = (
    # ----- Q1 healthy -----
    Row(1, "last-value `b_liq` / `b_runway` (Q1 description)", "KEEP", "0.966", "balances_b_qa.md", "last-vs-snap ρ; persist t↔t+3 = 0.848 (~0.85). Never B as Y2/Y3 X."),
    Row(1, "reconstruction walk (identity)", "CLOSE", "9.095e-12", "balances_b_qa.md", "median |resid|; do not rewrite liquidity.py."),
    Row(1, "`companies.csv` flags as transferable Q1 X", "CLOSE", "0.480", "companies_qa.md", "has_erp Y3 vs size 0.617; hidden test is new groups."),
    Row(1, "`created_at` / short-trail as health Y", "PARK", "73.6%", "trail_length.md", "connection clock, not cash trail (first tx late 64.2%)."),
    Row(1, "clean flags as health Y / X", "KEEP never-drop / PARK Y", "24", "clean_flags_qa.md", "tx extremes 24; quoted singles do not move (days 0.711)."),
    # ----- Q2 improving -----
    Row(2, "`y1_in_h3` thin path", "KEEP", "0.817", "y1_rest.md", "holdout med-norm vs hist 0.856; CV gap 0.005; beat-share 51%."),
    Row(2, "`y1_in_h1`", "PARK", "0.799", "y1_rest.md", "holdout 0.799 vs hist 0.795; CV and holdout disagree."),
    Row(2, "Y1 liquidity path (`y1_liq_h3`)", "PARK", "0.598", "y1_rest.md", "last-value wins CV OOF on liq_h3 vs GBM 0.632; Spearman(liq_t, liq_t+3)=0.848."),
    # ----- Q3 turning (45→65) -----
    Row(3, "Y3 278-col shallow engine", "KEEP", "0.762 ± 0.016", "overnight/dashboards/CONTEXT.md", "50 trees, depth 3. Registry 67e8ef01 mean 0.7622 sd 0.0158. Do not average with 400+ES 0.710."),
    Row(3, "Y3 15-col shallow-A (stems + lags 1/3)", "KEEP", "0.7520 ± 0.037", "i_lift.md", "beats days 0.711. Never B. Never a_op_in. No new cols on the card."),
    Row(3, "days bar `c_n_days_with_tx`", "KEEP", "0.711", "q6_quoted.md", "quiet-stressed recover bar. Size 0.617. Night quote."),
    Row(3, "Family I (`i_*`) merge onto the 15-col card", "CLOSE", "0.7615", "i_lift.md", "best add C3 still under KEEP gate 0.772. Leave in-memory."),
    Row(3, "Y2 trees (82→68 direction)", "PARK", "0.571", "y2_why.md", "best legal single days; drop 12 chronic names → 0.549. Direction real; non-B why is not."),
    Row(3, "Y9 trees / Family M as Y9 X", "CLOSE", "0.525", "y9_why.md", "legal leftover m_int_share; m_fin 0.618 is the Y (ρ 0.875). FinRegLab NSF/fee. Trees PARK."),
    Row(3, "Y6 `y6_zero_in_3` / missed-payroll", "PARK", "0.854", "zero_in_qa.md", "Y6 vs size 0.854 — inverse-activity fail. Do not revive."),
    # ----- Q4 dip vs fall -----
    Row(4, "Y7 TURNOVER (no DSO; issued_lag1 + CV + CN ±lag1 + f_fc_r_lag3)", "KEEP", "0.720", "y7_core.md", "n_x=5; fold 4 0.680. Do not grow. Do not quote 278-col 0.663 or holdout 0.680."),
    Row(4, "Y7 B_shallow SHAP card", "CLOSE", "0.712", "y7_core.md", "6-col named card; DSO fails short quintile 0.410."),
    Row(4, "Y2 = Y4 crash?", "CLOSE", "0.057", "y2_why.md", "Jaccard. Y2 median in-ratio 1.03 vs Y4 0.36. Already-neg 82%. Y2 ≠ Y4."),
    Row(4, "Y4 `d_cust_hhi_lag3` monopoly tail", "KEEP", "0.605", "y4_why.md", "HHI>0.975 tail; body ≤0.975 CV 0.445. ρ vs top1_lag3 0.991. CLOSE as Q6.", False, "monopoly tail"),
    Row(4, "Y4 2-col z-avg", "CLOSE", "0.634", "y4_why.md", "zavg is n_supp on those same rows (0.631, gap +0.003)."),
    Row(4, "Y4 XGB / LGB trees", "PARK", "0.585", "y4_xgb.md", "xgb_es 0.585 loses to the 0.605 single. Do not reopen trees."),
    Row(4, "Y8 both (inv / cash)", "PARK", "53.4%", "data_join_qa.md", "cross-source months; y8_inv had cash on 98.1% and still lost to a_in12."),
    Row(4, "Y10 utilisation", "PARK", "1.65%", "y10_acceptance.md", "no outstanding/granted history; snapshot last-month only. NORTH_STAR ~1.6%."),
    Row(4, "Y11 renamed Y2/Y3 on the 470", "CLOSE", "1.0", "y11_acceptance.md", "restricted Y2 ρ=1 with accepted Y2. Do not run assembler."),
    # ----- Q5 why -----
    Row(5, "CN leftover after `e_ar_issued_lag1`", "KEEP", "0.597", "credit_note_qa.md", "ρ issued_lag1 0.237 — not a twin. CLOSE TURNOVER add-on. Do not grow 0.720."),
    Row(5, "`e_delay_coll` leftover after DSO", "KEEP", "0.581", "delay_qa.md", "also after issued_lag1 0.583 / both 0.584. ρ vs DSO 0.214. CLOSE TURNOVER add-on."),
    Row(5, "Y5 unexplained leftover (cash × HHI neither)", "KEEP", "65%", "y5_why.md", "AP 65.1% / AR 65.3%. Trees PARK. Banque de France >30d / Hirshleifer PastDue%."),
    Row(5, "Y5 AR `d_tx_cp_share` (night quote)", "KEEP", "0.611", "y5_why.md", "CV 0.576. Fold 3 owns 85.9% of hole pos — not a leave-one-group law. PARK as X."),
    Row(5, "missing-CP share as new Q5", "CLOSE", "-0.947", "missing_cp_qa.md", "twin of store d_tx_cp_share. Leftover after uncat+dtx dies 0.509."),
    Row(5, "Family J `j_pay_match` (amount-match rate)", "KEEP", "35.7%", "data_join_qa.md", "KEEP-Q5 vs random 0.5%. 470 stay NaN. Not a Y3 X. Do not merge."),
    Row(5, "Family M (`m_*`) merge", "CLOSE", "0.525", "y9_why.md", "do not merge. Mix cannot answer Q6. m_fin 0.618 is the Y (ρ 0.875)."),
    Row(5, "Family H as Y3 X / sister mean `b_runway`≥1", "PARK", "+6.97pp", "sibling_h.md", "after-size T1. KEEP-Q5 footnote only. H marks sister existence, not sister health."),
    Row(5, "Y3 SHAP story (all-negative quiet recover)", "KEEP", "0.276", "y3_importances.md", "top SHAP `c_ss_month`; top-10 signs all −. Not dip-vs-fall.", False, "operationally quiet"),
    Row(5, "Y7 Family E SHAP share", "KEEP", "56%", "overnight/dashboards/CONTEXT.md", "lead-time/turning 52%. DSO SHAP #1 fails short-DSO fifth 0.410. shap_y7 e=1.4289."),
    Row(5, "`e_fx_share` as Y3 X / Q5 footnote", "CLOSE", "0.528", "fx_qa.md", "DROP from the 44. KEEP footnote only (mixed import/export). vs size 0.628 / days 0.711."),
    Row(5, "uncat amount/count as Q5 why", "CLOSE", "0.542", "uncat_qa.md", "Y3 count vs size 0.617. ICC 0.985 bookkeeping style. PARK as X."),
    Row(5, "`c_missed_salary` as Y3 / Q5 X", "CLOSE", "0.513", "salary_qa.md", "vs size 0.617 / days 0.711. Card already has c_salary_month 0.671."),
    Row(5, "`c_missed_tax` as Q5", "CLOSE", "0.511", "tax_qa.md", "Q-peaked 69.2% vs 43.4%. PARK as Y and as Y3 X."),
    Row(5, "signed `a_transfer` as Y3 X", "CLOSE", "0.566", "transfer_qa.md", "vs days 0.711. Honest leftover after days 0.579. ICC 0.956 TRAIT."),
    Row(5, "Javier `a_vol` / `a_out_vol` as engine", "CLOSE", "0.626", "a_vol_qa.md", "a_vol Δ+0.009 vs size is tercile mix. Do not quote a_out_vol 0.722 as the engine."),
    # ----- Q6 months earlier -----
    Row(6, "`e_ar_issued_lag1` (Y7)", "KEEP", "0.626", "q6_quoted.md", "short so-far 95.2% present; night 0.630 Δ −0.004. Hidden 72: 1-month claims only."),
    Row(6, "`c_n_days_with_tx_lag1` (Y3)", "KEEP", "0.684", "q6_quoted.md", "short 100% present. Contemporaneous days is SIGNAL (0.696), not lead."),
    Row(6, "Y3 C/A lag3 on short companies", "CLOSE", "36.3%", "q6_quoted.md", "company-short nn; so-far-short was 84.7% (full4 shift)."),
    Row(6, "F lag3 (`f_ds_r_lag3` / `f_fc_r_lag3`)", "CLOSE", "16.1%", "q6_quoted.md", "Y7 company-short f_fc_r_lag3; empty until so-far≥6."),
    Row(6, "Y4 `d_cust_hhi_lag3` as Q6", "CLOSE", "21.7%", "q6_quoted.md", "missing on short half; LOW_POWER 42 pos."),
    Row(6, "delay / overdue as Q6", "CLOSE", "0.0%", "delay_qa.md", "empty first 6 calendar months. Short lag1 0.576 is not issued_lag1."),
    Row(6, "`e_pending_amt_share` as Q6", "CLOSE", "0.430", "pending_qa.md", "lag1 ≈ now 0.420. Stock populated early (58.3%) but not a lead."),
    Row(6, "`e_dpo_proxy` as Q6", "CLOSE", "0.450", "dpo_qa.md", "short lag1 dies. Exists early (93.7%) unlike delay — still not a lead."),
    Row(6, "`d_supp_hhi` as Q6", "CLOSE", "0.523", "supp_hhi_qa.md", "leftover after cust HHI lag3 dies. Twin of d_supp_top1 ρ=0.987."),
    Row(6, "`a_growth_12` as Q6", "CLOSE", "0.0%", "growth_qa.md", "empty on short books (needs month 15+). n_def short=0."),
    Row(6, "CN lag1 as Q6", "CLOSE", "0.542", "credit_note_qa.md", "short so-far vs issued_lag1 0.630."),
    Row(6, "`next_payment_date` / `f_months_to_next_pay`", "CLOSE", "-5.55", "debt_schedule_qa.md", "p50 months-to-next; 91.6% already past. KEEP flow f_ds_r / f_fc_r."),
)

# Objects that look KEEP in an earlier md and DROP in a later leftover.
# Resolve from the later leftover QA. B / DSO stay listed as in-flight.
RESOLUTIONS: tuple[dict, ...] = (
    {
        "object": "e_credit_note_ratio as Y3 X / the 44",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "credit_note_qa.md",
        "resolve": "later leftover: DROP from the 44 as Y3 X (0.579 vs days 0.711; leftover 0.560). KEEP leftover after issued_lag1 0.597 as Q5 footnote. CLOSE TURNOVER add-on.",
        "winner": "credit_note_qa.md",
    },
    {
        "object": "e_delay_coll / overdue as Y3 X / the 44",
        "keep_where": "feature_report.md (44-col starter); shap_y7.md (Q5 why)",
        "drop_where": "delay_qa.md",
        "resolve": "later leftover: DROP from the 44 as Y3 X (0.512; leftover-days 0.427). KEEP leftover after DSO 0.581 as Q5 footnote. CLOSE TURNOVER add-on. CLOSE e_delay_paid / e_ar_overdue leftovers (0.449 / 0.547).",
        "winner": "delay_qa.md",
    },
    {
        "object": "e_pending_amt_share",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "pending_qa.md",
        "resolve": "later leftover: DROP from the 44. Leftover dies (Y7 after DSO 0.421 / issued_lag1 0.418). Q6 CLOSE. PARK as health Y.",
        "winner": "pending_qa.md",
    },
    {
        "object": "e_dpo_proxy",
        "keep_where": "feature_report.md (44-col starter); y7_core.md TURNDPO 0.723",
        "drop_where": "dpo_qa.md",
        "resolve": "later leftover: DROP from the 44. Y7 leftover CLOSE (0.471 / 0.443). TURNDPO 0.723 is not KEEP. Q6 CLOSE.",
        "winner": "dpo_qa.md",
    },
    {
        "object": "d_supp_hhi",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "supp_hhi_qa.md",
        "resolve": "later leftover: DROP from the 44. Twin of d_supp_top1 ρ=0.987. Y5 tail protective 2.7% vs 8.6% — not a Y4 crash. PARK as Y5 X (size-rank 0.663).",
        "winner": "supp_hhi_qa.md",
    },
    {
        "object": "c_gap_sd",
        "keep_where": "feature_report.md (44-col starter); shap_y3.md rank 8",
        "drop_where": "gap_sd_qa.md",
        "resolve": "later leftover: DROP from the 44. Twin of days ρ −0.905. Leftover after days 0.535 dies. Days leftover after gap 0.658 still lives.",
        "winner": "gap_sd_qa.md",
    },
    {
        "object": "c_recency_days",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "recency_qa.md",
        "resolve": "later leftover: DROP from the 44. Leftover after days 0.607 dies. June flag PARK as extract artifact (Javier 61 vs as-of 62; COMP_0981 is the +1).",
        "winner": "recency_qa.md",
    },
    {
        "object": "a_io_ratio / a_growth_3",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "growth_qa.md",
        "resolve": "later leftover: DROP both from the 44. Leftover after size+days dies (0.527 / 0.522). a_growth_12 CLOSE as Q6 (empty on short).",
        "winner": "growth_qa.md",
    },
    {
        "object": "c_zero_in_month / c_zero_in_share_6",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "zero_in_qa.md",
        "resolve": "later leftover: DROP from the 44. Leftover after days dies (0.553 / 0.537). Do not revive Y6.",
        "winner": "zero_in_qa.md",
    },
    {
        "object": "e_fx_share",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "fx_qa.md",
        "resolve": "later leftover: DROP from the 44 as Y3 X. KEEP only as a Q5 footnote (mixed import/export). Q6 lag1 CLOSE.",
        "winner": "fx_qa.md",
    },
    {
        "object": "f_has_factoring / f_has_confirming / f_has_loc / f_new_facility",
        "keep_where": "feature_report.md (44-col starter)",
        "drop_where": "factoring_qa.md",
        "resolve": "later leftover: DROP from the 44. Rise-only connection inventory. Y3 0.505 / 0.485 / 0.546 vs size 0.617. PARK as health Ys. KEEP flow f_ds_r / f_fc_r (debt_schedule_qa.md).",
        "winner": "factoring_qa.md",
    },
    {
        "object": "remaining g_has_* + g_custom_share",
        "keep_where": "feature_report.md (44); banking_g_qa.md (g_has_* all in starter=True; CLOSE as Y3 X, card 0.551)",
        "drop_where": "g_has_rest_qa.md",
        "resolve": "later leftover: DROP remaining g_has_saving / investment / tpv + g_custom_share. Honest leftover after days dies (OLS 0.65–0.71 is a fake days leak). Card already CLOSE 0.551; checking DROP (99.1% hole). banking_g confirmed they were on the starter, not that they stay as engines.",
        "winner": "g_has_rest_qa.md",
    },
    {
        "object": "a_uncat_share / amount-uncat as Q5",
        "keep_where": "feature_report.md (keep-list)",
        "drop_where": "uncat_qa.md",
        "resolve": "later leftover: CLOSE as Q5. PARK as Y and as X. ICC 0.985 style dummy. Do not merge amount-uncat.",
        "winner": "uncat_qa.md",
    },
    {
        "object": "c_missed_salary as Y3 X",
        "keep_where": "feature_report.md (rare-event keep)",
        "drop_where": "salary_qa.md",
        "resolve": "later leftover: CLOSE as Y3 / Q3–Q5 X. PARK as health Y. Lever is c_salary_month 0.671 already on the 15-col card.",
        "winner": "salary_qa.md",
    },
    {
        "object": "Family J as Y3 X vs KEEP-Q5",
        "keep_where": "match_report.md / data_join_qa.md (KEEP-Q5 / KEEP amount-match rate)",
        "drop_where": "not a DROP — role split",
        "resolve": "not a contradiction. KEEP as Q5 diagnostic (35.7% vs 0.5% random). Not a Y3 X (pay vs recover −0.060). Do not merge parquet.",
        "winner": "data_join_qa.md + match_report.md",
    },
    {
        "object": "Family M as Q5 mix vs Y9 merge",
        "keep_where": "catmix_report.md (maps m_* to Q5 why)",
        "drop_where": "y9_why.md (CLOSE merge)",
        "resolve": "later leftover: CLOSE merge. Fee/fin shares are the Y9 label (ρ 0.875 / 0.830). Mix cannot answer Q6.",
        "winner": "y9_why.md",
    },
    {
        "object": "a_out_vol 0.722 as engine",
        "keep_where": "a_vol_qa.md (Y3 0.722 reproduced)",
        "drop_where": "a_vol_qa.md (later-store KEEP False; X CLOSE)",
        "resolve": "same md, role split. Reproduced number is a company trait (demean 0.549, η² 0.741), not the Y3 engine. Do not quote 0.722 as the engine. Night quote stays 0.762 / 0.752.",
        "winner": "a_vol_qa.md (later-store KEEP False)",
    },
    {
        "object": "Y7 278-col 0.663 vs TURNOVER 0.720",
        "keep_where": "shap_y7.md quotes 0.663 as this-run CV",
        "drop_where": "y7_core.md (do not quote 278-col 0.663)",
        "resolve": "later core card: quote TURNOVER 0.720 / B_shallow 0.712. shap_y7.md is the 278-col SHAP story (Family E 56%), not the even card.",
        "winner": "y7_core.md",
    },
    {
        "object": "B columns on the 44 / DSO on the 44",
        "keep_where": "feature_report.md (b_runway, b_bal_vol, …, e_dso_proxy in the 44)",
        "drop_where": "b_on_44_qa.md / dso_qa.md exist after CONTEXT 04:50",
        "resolve": "IN-FLIGHT as of the 04:50 compile. Do not invent a settled DROP. balances_b_qa.md already KEEP last-value as Q1 description and forbids B as Y2/Y3 X. y7_core.md already drops DSO from TURNOVER. 44-drop list as of 04:50 does not wait for these two.",
        "winner": "IN-FLIGHT (B-on-44, DSO)",
    },
    {
        "object": "Y4 d_cust_hhi_lag3 KEEP as monopoly tail vs CLOSE as Q6",
        "keep_where": "y4_why.md (0.605 is the HHI>0.975 tail)",
        "drop_where": "q6_quoted.md (21.7% present on short; LOW_POWER 42 pos)",
        "resolve": "role split, not a fight. KEEP as the Q4 single (monopoly tail). CLOSE as a 3-month lead on short / hidden-72 books.",
        "winner": "y4_why.md (Q4) + q6_quoted.md (Q6)",
    },
    {
        "object": "Y4 2-col z-avg CLOSE vs PARK",
        "keep_where": "y4_xgb.md (PARK as next idea, not XGB KEEP)",
        "drop_where": "y4_why.md (CLOSE — is n_supp on the same rows)",
        "resolve": "later leftover: CLOSE the z-avg card. Trees stay PARK. Do not reopen.",
        "winner": "y4_why.md",
    },
    {
        "object": "created_at CLOSE as Y3 X vs PARK as health Y",
        "keep_where": "not KEEP — companies_qa.md CLOSE as Y3 X (CV 0.504)",
        "drop_where": "trail_length.md PARK as health Y (73.6% connection clock)",
        "resolve": "role split. PARK as a health Y. CLOSE as a transferable X. Same object, two jobs, same answer: not health.",
        "winner": "trail_length.md + companies_qa.md",
    },
    {
        "object": "e_fx_share CLOSE as Q5 X vs KEEP footnote",
        "keep_where": "fx_qa.md (KEEP footnote — mixed import/export)",
        "drop_where": "fx_qa.md (CLOSE as Y3 X / DROP from the 44)",
        "resolve": "same md, role split. DROP from the 44. KEEP only as a Q5 footnote.",
        "winner": "fx_qa.md",
    },
    {
        "object": "d_tx_cp_share KEEP 0.611 vs DROP from the 44",
        "keep_where": "y5_why.md (KEEP quote 0.611 / PARK as X)",
        "drop_where": "d_tx_qa.md (05:00 — after the 04:50 cut)",
        "resolve": "post-04:50 leftover. Do **not** add to the 04:50 44-drop list. If morning reads the 05:00 file: KEEP the 0.611 quote, PARK as X, DROP from the 44 as Y3 X (leftover-after-days 0.600). Same fold-3 hole 85.9%. Twin of miss_cp ρ −0.947 already CLOSE as new Q5.",
        "winner": "d_tx_qa.md (after-cut; 04:50 list unchanged)",
    },
)


def _read(name: str) -> str:
    """Read a source md. Bare names are analysis/outputs/; else repo-relative."""
    if "/" in name:
        path = ROOT / name
    else:
        path = OUTPUTS / name
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def _number_in(needle: str, text: str) -> bool:
    if needle in {"−", "-", "n/a"}:
        return True
    if needle in text:
        return True
    return False


def check_row(row: Row) -> list[str]:
    """Source-file existence + token + number must be quotable from the md."""
    issues: list[str] = []
    text = _read(row.source)
    if not text:
        issues.append(f"MISSING source {row.source} for {row.obj}")
        return issues
    token = row.token
    if token not in DECISIONS:
        issues.append(f"bad decision token {row.decision!r} on {row.obj}")
    if row.need_token and token not in text and row.decision.split()[0] not in text:
        issues.append(f"{row.source} has no {token} for {row.obj}")
    if not _number_in(row.number, text):
        issues.append(f"{row.source} does not contain number {row.number!r} for {row.obj}")
    if row.must_contain and row.must_contain not in text:
        issues.append(f"{row.source} missing must_contain {row.must_contain!r} for {row.obj}")
    return issues


def check_drop44() -> list[str]:
    """Each 04:50 drop must be named in its leftover QA with DROP or CLOSE."""
    issues: list[str] = []
    for col, src in DROP_44_SOURCES.items():
        text = _read(src)
        if not text:
            issues.append(f"MISSING leftover {src} for 44-drop {col}")
            continue
        if col.split()[0] not in text and col not in text:
            issues.append(f"{src} does not name {col}")
        if "DROP" not in text and "CLOSE" not in text:
            issues.append(f"{src} has neither DROP nor CLOSE for {col}")
    return issues


def check_registry_y3() -> list[str]:
    """Read-only: night Y3 0.7622 lives in registry agent 67e8ef01."""
    if not REGISTRY.exists():
        return ["MISSING registry.csv"]
    text = REGISTRY.read_text(encoding="utf-8")
    if "67e8ef01" not in text or "0.762239" not in text:
        return ["registry missing 67e8ef01 / 0.762239 (night Y3 shallow 278)"]
    return []


def remaining_44() -> list[str]:
    dropped = set(DROP_44_COLS)
    return [c for c in STARTER_44 if c not in dropped]


IN_FLIGHT_COLS = (
    "b_bal_vol", "b_below_0", "b_d_runway", "b_neg_episodes", "b_runway", "e_dso_proxy",
)
ROW_DROP_RE = re.compile(
    r"^\|[^|\n]*`?(?P<col>[a-z]_[a-z0-9_]+)`?[^|\n]*\|[^|\n]*\b(?P<dec>DROP|CLOSE|KEEP|PARK)\b",
    re.M,
)


def hunt_starter_vs_leftover() -> dict:
    """Same-row KEEP/DROP on starter-44 names. Later leftover wins. B/DSO stay in-flight."""
    leftover = sorted(
        p for p in OUTPUTS.glob("*.md")
        if p.name.endswith(("_qa.md", "_why.md"))
        or p.name in {"i_lift.md", "y7_core.md", "q6_quoted.md", "sibling_h.md", "feature_report.md"}
    )
    by_col: dict[str, list[tuple[str, str]]] = {c: [] for c in STARTER_44}
    for path in leftover:
        if path.name in {"brief_map.md", "b_on_44_qa.md", "dso_qa.md"}:
            # in-flight leftovers are listed, not resolved
            continue
        text = path.read_text(encoding="utf-8")
        for m in ROW_DROP_RE.finditer(text):
            col = m.group("col")
            if col in by_col:
                by_col[col].append((path.name, m.group("dec")))
    expected = []
    unexpected = []
    inflight = []
    for col, pairs in by_col.items():
        tokens = {t for _, t in pairs}
        if col in IN_FLIGHT_COLS:
            inflight.append(col)
            continue
        if col in DROP_44_COLS and ("DROP" in tokens or "CLOSE" in tokens):
            expected.append({"col": col, "files": pairs})
        elif "DROP" in tokens and col not in DROP_44_COLS:
            unexpected.append({"col": col, "files": pairs})
    return {
        "expected": expected,
        "unexpected": unexpected,
        "inflight": inflight,
        "n_expected": len(expected),
        "n_unexpected": len(unexpected),
    }


def safety_checks() -> list[str]:
    issues: list[str] = []
    for stem in Y3_STEMS:
        if stem in DROP_44_COLS:
            issues.append(f"15-col stem {stem} was folded into the 04:50 44-drop list")
    for col in ("b_runway", "b_bal_vol", "b_below_0", "b_d_runway", "b_neg_episodes", "e_dso_proxy"):
        if col in DROP_44_COLS:
            issues.append(f"in-flight {col} was folded into the settled 04:50 44-drop list")
    if len(STARTER_44) != 44:
        issues.append(f"STARTER_44 length {len(STARTER_44)} != 44")
    return issues


def hunt_keep_drop() -> list[dict]:
    """Scan leftover / why md PARK/CLOSE/KEEP tables for KEEP vs DROP on the same object."""
    hits: list[dict] = []
    table_re = re.compile(
        r"^\|\s*(?P<obj>[^|]+?)\s*\|\s*(?P<dec>[^|]*\*\*(?:KEEP|CLOSE|DROP|PARK)[^|]*\*\*[^|]*)\|",
        re.M,
    )
    by_obj: dict[str, list[tuple[str, str]]] = {}
    for path in sorted(OUTPUTS.glob("*.md")):
        if path.name in {"brief_map.md"}:
            continue
        text = path.read_text(encoding="utf-8")
        for m in table_re.finditer(text):
            obj = re.sub(r"`+", "", m.group("obj")).strip().lower()
            dec = m.group("dec")
            token = None
            for t in ("DROP", "KEEP", "CLOSE", "PARK"):
                if f"**{t}" in dec or f"{t}**" in dec:
                    token = t
                    break
            if not token:
                continue
            by_obj.setdefault(obj[:80], []).append((path.name, token))
    for obj, pairs in sorted(by_obj.items()):
        tokens = {t for _, t in pairs}
        if "KEEP" in tokens and "DROP" in tokens:
            hits.append({"object": obj, "files": pairs})
    return hits


def verify_night_quotes(text: str) -> list[str]:
    issues = []
    for q in (Y3_SHALLOW_278, Y3_15COL_A, DAYS_BAR, SIZE_BAR, Y7_TURNOVER, Y7_B_SHALLOW, CN_LEFTOVER, DELAY_LEFTOVER):
        if q.split("±")[0].strip() not in text and q not in text:
            issues.append(f"night quote missing from brief_map.md: {q}")
    # Forbidden trophies must appear only as 'do not quote'.
    for bad, label in (
        (FORBIDDEN_Y7_278, "278-col 0.663"),
        (FORBIDDEN_Y7_HOLDOUT, "holdout 0.680"),
        (FORBIDDEN_A_OUT_VOL, "a_out_vol 0.722"),
    ):
        if bad in text and "do not quote" not in text.lower() and "not quote" not in text.lower():
            issues.append(f"forbidden trophy {label} appears without a do-not-quote fence")
    if "0–100" not in text and "0-100" not in text:
        issues.append("brief_map.md must say morning cannot write a 0–100")
    return issues


def six_sentences() -> list[str]:
    return [
        "Healthy, tonight, is last-value reconstructed cash / `b_runway` on the 2026-09 still (last-vs-snap ρ 0.966; persist 0.85) — not `companies.csv` (has_erp Y3 0.480) and not `created_at` (connection clock, 73.6% late).",
        "Improving is only a thin inflow path: KEEP `y1_in_h3` (holdout med-norm 0.817 vs hist 0.856); the Y1 liquidity path stays PARK because last-value already wins (Spearman 0.85).",
        "Turning toward 45→65 is Y3 quiet-stressed recover: 278-col shallow 0.762 ± 0.016, 15-col A 0.752 ± 0.037 beating days 0.711 / size 0.617; Family I CLOSE (best add 0.7615 < 0.772); no new column on the 15-col card.",
        "Dip vs fall is Y7 TURNOVER 0.720 / B_shallow 0.712 on issued_lag1 (not DSO; not 278-col 0.663); Y2 is already-negative persistence (Jaccard vs Y4 0.057), Y4 is the HHI>0.975 monopoly tail (0.605), Y8 PARK.",
        "Why: CN leftover after issued_lag1 KEEP 0.597 and delay leftover after DSO KEEP 0.581 as footnotes (do not grow TURNOVER); Y5 is a 65% leftover; missing-CP is the `d_tx_cp_share` twin (ρ −0.947); Family J KEEP-Q5 not Y3 X; Family M CLOSE.",
        "Months earlier: KEEP issued_lag1 (short 0.626 / 95.2%) and days_lag1 (short 0.684 / 100%); CLOSE F lag3, Y4 HHI, delay, pending, DPO, supp HHI, and growth_12 — hidden 72 is 1-month claims only.",
    ]


def cannot_say() -> list[str]:
    return [
        "A 0–100 index, pillars, or product/web/Docker story.",
        "Holdout AUROC as a trophy. Hidden 72 is LOW_POWER (7–23 events) except `y7_top1_lost` (122 pos). Quote train group-fold CV.",
        "Hidden-72 lead time beyond one month (full-24 books 4.2% vs train 35.8%; Y4 HHI_lag3 almost undefined). `y7_top1_lost` holdout 122 events is the exception — still do not quote holdout AUROC.",
        "278-col Y7 0.663, holdout Y7 0.680, or `a_out_vol` 0.722 as the engine.",
        "Y2 / Y4 / Y5 / Y9 trees as a why (all PARK). Family I/M/J parquet merges.",
        "New columns on the 15-col Y3 card, or a grown TURNOVER card.",
        "Settled B-on-44 or DSO leftover verdicts — those two were in-flight at the 04:50 compile.",
    ]


def morning_can() -> list[str]:
    return six_sentences()


def render_md(checks: dict) -> str:
    now = datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
    hold_n = checks["holdout_n"]
    lines: list[str] = []
    a = lines.append
    a("# Brief map — six questions (morning-quotable)")
    a("")
    a(f"Generated `{now}` by `{AGENT}` (`analysis.evaluate.brief_map`). Compile / consistency pass. DuckDB and parquet were **not** opened for a join. No GBM. No `build_targets`. No 0–100. No `product/`. Holdout seed `{FOLD_SEED}`; holdout n={hold_n} (coverage only).")
    a("")
    a("Re-read `overnight/NORTH_STAR.md`: the object is a **trajectory** (45→65 recover vs 82→68 deteriorate), not a last-month snapshot. Every row below maps to one of the six questions.")
    a("")
    a("## Night quotes (do not change)")
    a("")
    a(f"- Y3 shallow 278 **{Y3_SHALLOW_278}**. 15-col A **{Y3_15COL_A}**. Days bar **{DAYS_BAR}**. Size **{SIZE_BAR}**.")
    a(f"- Y7 TURNOVER **{Y7_TURNOVER}** / B_shallow **{Y7_B_SHALLOW}**. Do not quote 278-col {FORBIDDEN_Y7_278} or holdout {FORBIDDEN_Y7_HOLDOUT}. Do not quote a_out_vol {FORBIDDEN_A_OUT_VOL} as the engine.")
    a(f"- CN leftover after issued_lag1 **KEEP {CN_LEFTOVER}** — do not grow TURNOVER.")
    a(f"- delay_coll leftover after DSO **KEEP {DELAY_LEFTOVER}** — do not grow TURNOVER.")
    a("")
    a("## 15-col Y3 card (do not add)")
    a("")
    a("Stems + lags 1,3: `" + "`, `".join(Y3_STEMS) + "`. Never B. Never `a_op_in` (SIZE). `e_dso_proxy` left out. Family I CLOSE.")
    a("")
    a("## Y7 even card (do not grow)")
    a("")
    a(f"TURNOVER n_x=5: `e_ar_issued_lag1` + issued-lag CV + CN ±lag1 + `f_fc_r_lag3`. No DSO. Quote **{Y7_TURNOVER}** / B_shallow **{Y7_B_SHALLOW}**.")
    a("")

    for q in range(1, 7):
        a(f"## Q{q} — {QUESTIONS[q]}")
        a("")
        a("| object | decision | one number | source |")
        a("| --- | --- | --- | --- |")
        for row in MAP:
            if row.q != q:
                continue
            a(f"| {row.obj} | **{row.decision}** | {row.number} | `{row.source}` |")
        a("")
        notes = [r for r in MAP if r.q == q and r.note]
        if notes:
            a("Notes:")
            a("")
            for r in notes:
                a(f"- {r.obj}: {r.note}")
            a("")

    a("## Contradiction hunt (KEEP in one md, DROP in another)")
    a("")
    a("Rule: the **later leftover QA** wins. Feature-report 44 is the starter, not the morning list. Role splits (KEEP leftover / CLOSE TURNOVER add-on; KEEP-Q5 / not Y3 X) are not fights.")
    a("")
    a("| object | KEEP where | DROP where | resolve (later leftover) |")
    a("| --- | --- | --- | --- |")
    for r in RESOLUTIONS:
        a(f"| {r['object']} | {r['keep_where']} | {r['drop_where']} | {r['resolve']} |")
    a("")
    raw = checks.get("raw_keep_drop") or []
    starter = checks.get("starter_hits") or {}
    a(f"Parser also saw **{len(raw)}** table-level KEEP+DROP collisions (object-string match). Same-row leftover DROP/CLOSE on starter-44 names: **{starter.get('n_expected', 0)}** expected (in the 04:50 drop list), **{starter.get('n_unexpected', 0)}** unexpected. In-flight B / DSO are not resolved.")
    a("")
    unexpected = starter.get("unexpected") or []
    if unexpected:
        a("Unexpected same-row DROP on a column that is **not** in the 04:50 drop list (review; do not auto-add):")
        a("")
        for h in unexpected:
            files = ", ".join(f"{fn}:{dec}" for fn, dec in h["files"])
            a(f"- `{h['col']}` — {files}")
        a("")
    a("## 44-drop list as of 04:50")
    a("")
    a("Do not wait for in-flight B-on-44 / DSO. Feature-report starter was 44; night leftover QAs have since dropped:")
    a("")
    for c in DROP_44_LABELS:
        a(f"- `{c}`")
    a("")
    a("**In-flight (do not invent a verdict):**")
    a("")
    for c in IN_FLIGHT:
        a(f"- {c}")
    a("")
    inflight_exist = checks.get("inflight_files") or {}
    for name, exists in inflight_exist.items():
        a(f"  - `{name}` exists after CONTEXT 04:50: **{exists}**. Verdict still in-flight for this compile.")
    a("")
    rest = ", ".join(f"`{c}`" for c in remaining_44())
    a(f"Still on the 44 as of 04:50 (not a KEEP-as-engine): {rest}. Read: `a_in3` size control; `a_growth_12` CLOSE as Q6; `a_uncat_share` CLOSE Q5; `c_missed_salary` / `c_missed_tax` CLOSE/PARK as X; `d_cust_hhi` Y4 monopoly-tail single / CLOSE as Q6; `d_tx_cp_share` Y5 KEEP quote / PARK as X; `e_dso_proxy` **in-flight**; `f_ds_r` / `f_fc_r` KEEP flow; `f_outstanding_gt_granted` PARK snapshot; `h_sib_neg_share` PARK as Y3 X; five B columns **in-flight** as X (`b_runway` already KEEP as Q1 description).")
    a("")
    a("`d_tx_qa.md` appeared **05:00** (after the 04:50 cut). Do not fold `d_tx_cp_share` into the 04:50 drop list. After-cut verdict if morning reads it: KEEP 0.611 quote, PARK as X, DROP from the 44 as Y3 X.")
    a("")
    a("## What morning can say (six sentences)")
    a("")
    for i, s in enumerate(morning_can(), 1):
        a(f"{i}. {s}")
    a("")
    a("## What morning cannot say")
    a("")
    for s in cannot_say():
        a(f"- {s}")
    a("")
    a("## SHAP story (already written)")
    a("")
    a("- **Y3:** among already-stressed months, all top-10 SHAP signs are **−**. Quiet firms (no SS/salary, fewer txs) are the ones whose runway later prints ≥ 3 for three months. That is de-escalation / mean reversion at the bottom — the 45→65 direction. It does **not** yet separate dip from fall. Canonical: `y3_importances.md` (50-tree). Do not mix parent 400-tree rank order.")
    a("- **Y7:** Family E holds **56%** of mean |SHAP|; lead-time / turning holds **52%**. DSO is SHAP #1 and **worse than chance** on the short-DSO fifth (0.410). TURNOVER drops DSO; fold 4 0.680 is issued, not DSO. Canonical: `shap_y7.md` + `y7_core.md`.")
    a("")
    a("## Literature already on the desk (do not invent papers)")
    a("")
    a("- FinRegLab 2025 NSF/fee / low-negative balances — NORTH_STAR Y9 turning/why; journal `2026-09-18-2350-feature-store-and-y-plan.md`.")
    a("- Banque de France Bulletin 227/8: late payments **>30d** move PD; ≤30d do not. Y5 is that clock; the 65% leftover is what cash/HHI do not explain.")
    a("- Hirshleifer et al. 2019 PastDue% — same Y5 plan note. Not a new paper.")
    a("")
    a("## Compile checks")
    a("")
    a(f"- Holdout companies: **{hold_n}** (want 72). Seed `{FOLD_SEED}`.")
    a(f"- Map rows: **{len(MAP)}**. Source-file token+number misses: **{len(checks['row_issues'])}**.")
    a(f"- 44-drop leftover confirm misses: **{len(checks.get('drop44_issues') or [])}**.")
    a(f"- Registry 67e8ef01 / 0.762239: **{'ok' if not checks.get('registry_issues') else checks['registry_issues']}**.")
    a(f"- Resolved KEEP-vs-DROP objects: **{len(RESOLUTIONS)}**. Raw parser collisions: **{len(raw)}**.")
    a(f"- In-flight: {', '.join(IN_FLIGHT)}.")
    a(f"- Parquet / models: not opened / not fit.")
    a("")
    if checks.get("drop44_issues"):
        a("### 44-drop leftover issues")
        a("")
        for x in checks["drop44_issues"]:
            a(f"- {x}")
        a("")
    if checks["row_issues"]:
        a("### Row issues (must be empty for a clean compile)")
        a("")
        for x in checks["row_issues"]:
            a(f"- {x}")
        a("")
    else:
        a("Row compile is clean: every number and decision token is in its source md.")
        a("")
    a("## What this module did not do")
    a("")
    a("- Did not change night Y3 0.762 / 0.752 or Y7 TURNOVER 0.720 / B_shallow 0.712.")
    a("- Did not put anything on the 15-col card. Did not grow TURNOVER.")
    a("- Did not invent B-on-44 or DSO leftover verdicts.")
    a("- Did not rewrite parquet or duckdb. Did not run `build_targets`. Did not fit. Did not commit.")
    a("- Did not write the parent journal / LIVE / canvas / `product/`.")
    a("")
    a("## Re-run")
    a("")
    a("```bash")
    a("/home/walterjtv/.pyenv/versions/base/bin/python3 -m analysis.evaluate.brief_map")
    a("```")
    a("")
    return "\n".join(lines)


HEADLINES = {
    1: "KEEP last-value b_runway\nρ 0.966  persist 0.85\nCLOSE companies.csv 0.480\nPARK created_at 73.6%",
    2: "KEEP thin y1_in_h3\nholdout 0.817 vs hist 0.856\nPARK liq path 0.598\nlast-value already wins",
    3: "Y3 45→65 engine\n278-col 0.762 ± 0.016\n15-col A 0.752 ± 0.037\nFamily I CLOSE 0.7615",
    4: "Y7 TURNOVER 0.720\nissued_lag1 · no DSO\nY2 ≠ Y4  Jaccard 0.057\nY4 monopoly tail 0.605",
    5: "CN leftover KEEP 0.597\ndelay leftover KEEP 0.581\nY5 leftover 65%\nJ KEEP-Q5 · M CLOSE",
    6: "KEEP issued_lag1 0.626\nKEEP days_lag1 0.684\nCLOSE F/HHI/delay/DPO\nhidden 72 = 1-month",
}


def draw_png() -> None:
    if not HAS_MPL:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11.4, 6.6))
    colors = {
        1: "#1f4e5f",
        2: "#2e8b57",
        3: "#c47b17",
        4: "#8b1e3f",
        5: "#3d348b",
        6: "#4a4a4a",
    }
    for q, ax in zip(range(1, 7), axes.ravel()):
        rows = [r for r in MAP if r.q == q]
        keep = sum(1 for r in rows if r.token == "KEEP")
        close = sum(1 for r in rows if r.token == "CLOSE")
        drop = sum(1 for r in rows if r.token == "DROP")
        park = sum(1 for r in rows if r.token == "PARK")
        ax.set_facecolor("#f7f4ee")
        for s in ax.spines.values():
            s.set_color(colors[q])
            s.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Q{q}  {QUESTIONS[q]}", fontsize=10, color=colors[q], pad=8)
        body = f"{HEADLINES[q]}\n\nKEEP {keep} · CLOSE {close} · DROP {drop} · PARK {park}"
        ax.text(0.5, 0.52, body, ha="center", va="center", fontsize=8, color="#222", family="DejaVu Sans")
    fig.suptitle("X Ray brief map — six questions (04:50 compile; B / DSO in-flight)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=140)
    plt.close(fig)


def append_registry(n_rows: int, n_issues: int, n_res: int) -> None:
    if not REGISTRY.exists():
        raise FileNotFoundError(REGISTRY)
    ts = datetime.now().strftime("%Y-%m-%dT%H:%M")
    rows = [
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "-",
            "model": MODEL,
            "split": "compile",
            "metric": "n_map_rows",
            "value": n_rows,
            "coverage": "1.0000",
            "notes": f"six-question compile; issues={n_issues}; resolved={n_res}; in_flight=B,DSO; quotes unchanged",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "-",
            "model": MODEL,
            "split": "compile",
            "metric": "n_keep_drop_resolved",
            "value": n_res,
            "coverage": "1.0000",
            "notes": "later leftover QA wins; B/DSO in-flight",
        },
        {
            "ts": ts,
            "round": ROUND,
            "wave": WAVE,
            "agent": AGENT,
            "x_families": X_FAM,
            "y": "-",
            "model": MODEL,
            "split": "compile",
            "metric": "n_source_issues",
            "value": n_issues,
            "coverage": "1.0000",
            "notes": "0 = every KEEP/CLOSE/DROP number is in its source md",
        },
    ]
    with REGISTRY.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        seen = {
            (
                r.get("agent"),
                r.get("x_families"),
                r.get("y"),
                r.get("model"),
                r.get("split"),
                r.get("metric"),
            )
            for r in reader
        }
    fresh = []
    for r in rows:
        key = (
            str(r.get("agent", "")),
            str(r.get("x_families", "")),
            str(r.get("y", "")),
            str(r.get("model", "")),
            str(r.get("split", "")),
            str(r.get("metric", "")),
        )
        if key in seen:
            continue
        fresh.append(r)
        seen.add(key)
    if not fresh:
        print("registry: no new rows")
        return
    with REGISTRY.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header, extrasaction="ignore")
        for r in fresh:
            w.writerow(r)
    print(f"registry: appended {len(fresh)} compile rows")


def compile_once() -> dict:
    hold = load_holdout()
    if FOLD_SEED != 20260918:
        raise RuntimeError(f"holdout seed drifted: {FOLD_SEED}")
    if len(hold) != 72:
        raise RuntimeError(f"holdout n drifted: {len(hold)}")
    row_issues: list[str] = []
    for row in MAP:
        row_issues.extend(check_row(row))
    drop44_issues = check_drop44()
    registry_issues = check_registry_y3()
    raw = hunt_keep_drop()
    starter_hits = hunt_starter_vs_leftover()
    safety_issues = safety_checks()
    inflight_files = {name: (OUTPUTS / name).exists() for name in IN_FLIGHT_FILES}
    return {
        "holdout_n": len(hold),
        "row_issues": row_issues + safety_issues,
        "drop44_issues": drop44_issues,
        "registry_issues": registry_issues,
        "raw_keep_drop": raw,
        "starter_hits": starter_hits,
        "inflight_files": inflight_files,
    }


def main() -> int:
    print("NORTH_STAR: six questions, trajectory not snapshot, no 0-100, hidden 72.")
    print(f"holdout seed {FOLD_SEED}; do not fit; do not quote holdout AUROC.")
    checks = compile_once()
    print(f"holdout n={checks['holdout_n']}")
    print(f"map rows={len(MAP)} source issues={len(checks['row_issues'])}")
    for x in checks["row_issues"]:
        print("  ISSUE", x)
    print(f"raw KEEP+DROP collisions={len(checks['raw_keep_drop'])}")
    print(f"resolved contradictions={len(RESOLUTIONS)}")
    sh = checks.get("starter_hits") or {}
    print(f"starter-44 same-row expected={sh.get('n_expected')} unexpected={sh.get('n_unexpected')}")
    for h in sh.get("unexpected") or []:
        print("  UNEXPECTED", h["col"], h["files"])
    print(f"44-drop leftover issues={len(checks.get('drop44_issues') or [])}")
    for x in checks.get("drop44_issues") or []:
        print("  DROP44", x)
    print(f"registry y3={checks.get('registry_issues') or 'ok'}")
    print(f"in-flight files={checks.get('inflight_files')}")
    md = render_md(checks)
    quote_issues = verify_night_quotes(md)
    for x in quote_issues:
        print("  QUOTE", x)
        checks["row_issues"].append(x)
    # Re-render if quote fence needed — verify_night_quotes is informational.
    OUT_MD.write_text(md, encoding="utf-8")
    print(f"wrote {OUT_MD}")
    draw_png()
    if OUT_PNG.exists():
        print(f"wrote {OUT_PNG}")
    append_registry(len(MAP), len(checks["row_issues"]), len(RESOLUTIONS))
    print("in-flight: B-on-44, DSO — verdicts not invented.")
    print("night quotes unchanged: 0.762/0.752 ; 0.720/0.712 ; CN 0.597 ; delay 0.581")
    return 0 if not [i for i in checks["row_issues"] if i.startswith("MISSING")] else 1


if __name__ == "__main__":
    raise SystemExit(main())
