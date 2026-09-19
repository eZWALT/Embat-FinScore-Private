import { createScoreRepository } from "./repository";
import {
  CATEGORY_LABELS,
  MONITORING_DISCLAIMER,
  topReasonInSpanish,
} from "./plain-language";
import type {
  CategoryId,
  DashboardCompany,
  DashboardData,
  ScoreRepository,
} from "./types";

const CATEGORY_ORDER: CategoryId[] = [
  "payment_history",
  "amounts_owed",
  "stability",
  "new_credit",
  "mix",
];

export async function getDashboardData(
  repository: ScoreRepository = createScoreRepository(),
): Promise<DashboardData> {
  if (repository.getDashboardSnapshot) {
    return repository.getDashboardSnapshot();
  }

  const [manifest, index] = await Promise.all([
    repository.getManifest(),
    repository.listCompanies(),
  ]);

  if (index.length === 0) {
    throw new Error("The score bundle contains no companies");
  }

  const companies = await Promise.all(
    index.map(async (summary): Promise<DashboardCompany> => {
      const detail = await repository.getCompany(summary.company_id);
      const latest = detail.months.at(-1);

      if (!latest) {
        throw new Error(`Company ${summary.company_id} has no scored months`);
      }

      return {
        companyId: detail.company_id,
        groupId: detail.group_id,
        country: detail.country,
        currency: detail.currency,
        erp: detail.erp,
        latestMonth: detail.latest_month,
        score: latest.score,
        delta1m: summary.delta_1m,
        delta3m: summary.delta_3m,
        trajectory: latest.trajectory,
        confidence: latest.confidence,
        confidenceNote: latest.confidence_note,
        coverage: latest.coverage,
        topReason: topReasonInSpanish(summary.guard, latest.reasons?.[0]),
        scoreHistory: detail.months.map(({ month, score }) => ({ month, score })),
        categories: CATEGORY_ORDER.map((id) => ({
          id,
          label: CATEGORY_LABELS[id] ?? manifest.spec.categories[id]?.label ?? id,
          score: latest.categories[id]?.score ?? null,
        })),
      };
    }),
  );

  companies.sort((a, b) => b.score - a.score);

  return {
    asOfMonth: manifest.as_of_month,
    isSample: manifest.is_sample,
    disclaimer: MONITORING_DISCLAIMER,
    companies,
  };
}
