const MONTHS = [
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

export function formatMonth(month: string) {
  const [year, monthNumber] = month.split("-");
  const index = Number(monthNumber) - 1;
  const label = MONTHS[index] ?? monthNumber;
  return `${label} ’${year.slice(2)}`;
}
