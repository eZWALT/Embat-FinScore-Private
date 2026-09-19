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

export function yDomainForSelection(
  rows: Record<string, string | number | null>[],
  companyIds: string[],
): [number, number] {
  let min = Infinity;
  let max = -Infinity;
  for (const row of rows) {
    for (const id of companyIds) {
      const value = row[id];
      if (typeof value !== "number" || Number.isNaN(value)) continue;
      min = Math.min(min, value);
      max = Math.max(max, value);
    }
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 100];
  if (min === max) {
    return [Math.max(0, min - 5), Math.min(100, max + 5)];
  }
  const pad = Math.max(2, (max - min) * 0.1);
  return [Math.max(0, Math.floor(min - pad)), Math.min(100, Math.ceil(max + pad))];
}
