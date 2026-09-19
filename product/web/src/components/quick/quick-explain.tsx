import { companyLabel, formatPoints, groupLabel, trajectoryLabel } from "@/components/group/labels";
import type { MonthRange } from "@/components/quick/quick-chart";
import type { DashboardCompany } from "@/lib/data/types";
import { formatMonth } from "@/lib/format-month";

/** The question sent to the same Ask chat as Profundo: the numbers are ours, the reasons are the agent's. */
export function buildPrompt(companies: DashboardCompany[], range: MonthRange) {
  const lines = companies.map((company) => {
    const inRange = company.scoreHistory.filter((point) => point.month >= range.from && point.month <= range.to);
    const first = inRange[0];
    const last = inRange.at(-1);
    if (!first || !last) return `- ${companyLabel(company.companyId)}: sin puntuación en el periodo`;
    const delta = last.score - first.score;
    const group = company.groupId ? ` (${groupLabel(company.groupId)})` : "";
    return `- ${companyLabel(company.companyId)}${group}: ${first.score.toFixed(0)} en ${formatMonth(first.month)} → ${last.score.toFixed(0)} en ${formatMonth(last.month)} (${formatPoints(delta)} pts), ${trajectoryLabel(company.trajectory)}`;
  });
  return `Periodo seleccionado: ${formatMonth(range.from)} → ${formatMonth(range.to)}.\nEmpresas:\n${lines.join("\n")}\n\nExplica qué pasó en ese periodo y por qué.`;
}
