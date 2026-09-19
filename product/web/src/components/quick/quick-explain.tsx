import { formatPoints } from "@/components/group/labels";
import type { MonthRange } from "@/components/quick/quick-chart";
import type { DashboardCompany } from "@/lib/data/types";

/** The question sent to the same Ask chat as Profundo: the numbers are ours, the reasons are the agent's. */
export function buildPrompt(companies: DashboardCompany[], range: MonthRange) {
  const lines = companies.map((company) => {
    const inRange = company.scoreHistory.filter((point) => point.month >= range.from && point.month <= range.to);
    const first = inRange[0];
    const last = inRange.at(-1);
    if (!first || !last) return `- ${company.companyId}: sin puntuación en el periodo`;
    const delta = last.score - first.score;
    return `- ${company.companyId}${company.groupId ? ` (${company.groupId})` : ""}: ${first.score.toFixed(0)} en ${first.month} → ${last.score.toFixed(0)} en ${last.month} (${formatPoints(delta)} pts), hoy ${company.trajectory}`;
  });
  return `Periodo seleccionado: ${range.from} → ${range.to}.\nEmpresas:\n${lines.join("\n")}\n\nExplica qué pasó en ese periodo y por qué.`;
}
