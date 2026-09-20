const MONTHS_SHORT = [
  "ene",
  "feb",
  "mar",
  "abr",
  "may",
  "jun",
  "jul",
  "ago",
  "sept",
  "oct",
  "nov",
  "dic",
] as const;

const MONTHS_FULL = [
  "enero",
  "febrero",
  "marzo",
  "abril",
  "mayo",
  "junio",
  "julio",
  "agosto",
  "septiembre",
  "octubre",
  "noviembre",
  "diciembre",
] as const;

function parts(month: string): { year: string; monthNumber: string; index: number } {
  const [year, monthNumber] = month.split("-");
  return { year, monthNumber, index: Number(monthNumber) - 1 };
}

/** "2026-08" → "agosto 2026" — badges, titles, tooltips. */
export function formatMonth(month: string) {
  const { year, monthNumber, index } = parts(month);
  return `${MONTHS_FULL[index] ?? monthNumber} ${year}`;
}

/** "2026-08" → "ago 2026" — tight chart axes (no apostrophe). */
export function formatMonthShort(month: string) {
  const { year, monthNumber, index } = parts(month);
  return `${MONTHS_SHORT[index] ?? monthNumber} ${year}`;
}

/** "2026-08" → "hasta agosto 2026" — header badges for the scored cut-off. */
export function formatAsOf(month: string) {
  return `hasta ${formatMonth(month)}`;
}

/** "2026-08" → ["ago", "2026"] for two-line column headers. */
export function splitMonth(month: string): [string, string] {
  const { year, monthNumber, index } = parts(month);
  return [MONTHS_SHORT[index] ?? monthNumber, year];
}

const MONTH_NAME_INDEX: Record<string, number> = (() => {
  const map: Record<string, number> = { sep: 8 };
  MONTHS_SHORT.forEach((name, index) => {
    map[name] = index;
  });
  MONTHS_FULL.forEach((name, index) => {
    map[name] = index;
  });
  return map;
})();

/** Accepts `2026-08`, `ago 2026`, `agosto 2026`. Returns `YYYY-MM` or null. */
export function parseMonth(value: string | null | undefined): string | null {
  if (!value) return null;
  const raw = value.trim();
  if (/^\d{4}-\d{2}$/.test(raw)) return raw;
  const match = raw
    .toLowerCase()
    .normalize("NFD")
    .replace(/\p{Diacritic}/gu, "")
    .match(/^(ene|feb|mar|abr|may|jun|jul|ago|sept?|oct|nov|dic|enero|febrero|marzo|abril|mayo|junio|julio|agosto|septiembre|octubre|noviembre|diciembre)\s+(\d{4})$/);
  if (!match) return null;
  const index = MONTH_NAME_INDEX[match[1]];
  if (index == null) return null;
  return `${match[2]}-${String(index + 1).padStart(2, "0")}`;
}
