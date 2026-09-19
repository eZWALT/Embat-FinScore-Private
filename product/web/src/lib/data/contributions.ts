import { CATEGORY_LABELS } from "./plain-language";
import { createScoreRepository } from "./repository";
import type { CategoryId } from "./types";

const COMPANY_ID = /^COMP_[0-9]{4}$/;

const CATEGORY_ORDER: CategoryId[] = ["payment_history", "amounts_owed", "stability", "new_credit", "mix"];

/** What each category brought to the index in one month. `points` are points of the 100; `score` is null when the category was not observable. */
export interface MonthContribution {
  month: string;
  categories: { id: CategoryId; label: string; score: number | null; points: number }[];
  /** Points removed by the safety guard (score minus score before the cap). 0 when no guard applied. */
  guardAdjustment: number;
}

/** Per-month category contributions of one company, keyed by month. Read from whichever repository is configured. */
export async function getCompanyContributions(companyId: string): Promise<Record<string, MonthContribution>> {
  if (!COMPANY_ID.test(companyId)) throw new Error(`Invalid company id: ${companyId}`);

  const detail = await createScoreRepository().getCompany(companyId);
  const byMonth: Record<string, MonthContribution> = {};
  for (const record of detail.months) {
    const adjustment = record.score - record.score_pre_cap;
    byMonth[record.month] = {
      month: record.month,
      categories: CATEGORY_ORDER.map((id) => ({
        id,
        label: CATEGORY_LABELS[id],
        score: record.categories[id]?.score ?? null,
        points: Number(record.categories[id]?.contribution ?? 0),
      })),
      guardAdjustment: Math.abs(adjustment) >= 0.05 ? adjustment : 0,
    };
  }
  return byMonth;
}
