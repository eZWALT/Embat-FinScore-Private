# 2026-09-19-0740 — lit_invoice pass 3 (Ellingsen in)

- **Author:** agent (Cursor, 689100e7)
- **When:** 2026-09-19 ~07:40 CEST

## What changed

Tightened the invoice cluster without growing past 12.

- Swapped **Lian 2017** (abstract_only, 2-year contagion) out of the
  ultra table → companion. Q6 CLOSE still cites it.
- Swapped in **Ellingsen–Jacobson–von Schedvin 2016** Riksbank WP 315
  (full PDF): 52m contracts; AP moves with **input volume**, not days;
  overdue is a minor fraction. Locks “do not grow TURNOVER with DSO.”
- Added companions: Barrot 2016 (abstract −25% PD; PDF not opened),
  Lian. Amberg 2021 and Costello 2020 stay companions.
- Campello–Gao +10 bp / +0.2 covenants / −2 months confirmed on Nova
  SBE PDF (6% of 179 bp mean; 46-month maturity).
- García-Appendini numbers from Caixa WP: crisis AR/sales −3 pp;
  +1 SD cash → +0.5 pp quarterly AR/sales (~$9.8m / ~11% of LOC).
- Bitetto: they drop Outstanding because it is correlated with Turnover;
  keep Delinquency (mean 1.62%).
- Wave C is now **CN on top-1**, not a redo of note vs refund (0.595 /
  0.496 already in `credit_note_qa`).
- Wave D names the **222/341 AP neither cell**.
- Marouani 2014 SSRN confirmed = the thin preprint already OUT.

## Decisions

- Cap stays 12. No dump. Pérez-Salazar still the only CONTRADICT.
- CN ratio still a literature gap.
- Do not grow TURNOVER 0.720.

## Still unknown

- Costello table coefficients (PDF still closed).
- Whether top-1 PastDue% leftover clears 0.58.
- Whether net TC × dip names the Y5 65% leftover.
