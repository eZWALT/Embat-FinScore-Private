import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany } from "@/lib/data/types";

export const SERIES_COLORS = [
  "oklch(0.52 0.19 250)",
  "oklch(0.55 0.16 145)",
  "oklch(0.58 0.18 35)",
  "oklch(0.52 0.18 300)",
  "oklch(0.5 0.14 200)",
  "oklch(0.55 0.19 20)",
  "oklch(0.48 0.12 80)",
  "oklch(0.5 0.16 340)",
] as const;

/** Same order as SERIES_COLORS. The chat names lines with these words. */
export const SERIES_COLOR_LABELS = [
  "azul",
  "verde",
  "naranja",
  "violeta",
  "turquesa",
  "rojo",
  "oliva",
  "rosa",
] as const;

export function buildChartRows(selected: DashboardCompany[]) {
  const months = [
    ...new Set(selected.flatMap((company) => (company.scoreHistory ?? []).map((point) => point.month))),
  ].sort();

  return months.map((month) => {
    const row: Record<string, string | number | null> = {
      month,
      label: formatMonth(month),
    };
    for (const company of selected) {
      row[company.companyId] =
        (company.scoreHistory ?? []).find((point) => point.month === month)?.score ?? null;
    }
    return row;
  });
}
