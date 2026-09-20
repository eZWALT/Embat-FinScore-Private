const COMPANY_ID = /^COMP_(\d{4})$/;
const GROUP_ID = /^GROUP_(\d{4})$/;

export function companyLabel(id: string): string {
  const match = COMPANY_ID.exec(id);
  return match ? `Empresa ${match[1]}` : id;
}

export function groupLabel(id: string): string {
  const match = GROUP_ID.exec(id);
  return match ? `Grupo ${match[1]}` : id;
}

export function entityLabel(id: string): string {
  if (GROUP_ID.test(id)) return groupLabel(id);
  if (COMPANY_ID.test(id)) return companyLabel(id);
  return id;
}

export function relabelEntities(text: string): string {
  return text
    .replace(/COMP_\d{4}/g, (id) => companyLabel(id))
    .replace(/GROUP_\d{4}/g, (id) => groupLabel(id))
    .replace(/COUNTERPARTY_(\d+)/g, (_match, digits: string) => `cliente ${digits}`);
}

export function companySearchText(company: {
  companyId: string;
  groupId?: string | null;
}): string {
  return [
    company.companyId,
    companyLabel(company.companyId),
    company.groupId,
    company.groupId ? groupLabel(company.groupId) : "",
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
}

export function formatDecimal(value: number, digits = 1): string {
  return new Intl.NumberFormat("es-ES", {
    minimumFractionDigits: 0,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatSigned(value: number, digits = 1): string {
  const text = formatDecimal(Math.abs(value), digits);
  if (value > 0) return `+${text}`;
  if (value < 0) return `−${text}`;
  return text;
}

/** Javi money: `14 k€`, `1,2 M€`, `164 €`. Other ISO codes: `14 k AED`. */
export function formatMoney(value: number, currency: string | null = "EUR"): string {
  const code = (typeof currency === "string" && currency ? currency : "EUR").toUpperCase();
  const abs = Math.abs(value);
  const sign = value < 0 ? "−" : "";
  const amount = (n: number, digits: number) => `${sign}${formatDecimal(n, digits)}`;
  if (abs >= 1_000_000) {
    const body = amount(abs / 1_000_000, 1);
    return code === "EUR" ? `${body} M€` : `${body} M ${code}`;
  }
  if (abs >= 1000) {
    const digits = abs >= 10_000 ? 0 : 1;
    const body = amount(abs / 1000, digits);
    return code === "EUR" ? `${body} k€` : `${body} k ${code}`;
  }
  const body = amount(abs, 0);
  return code === "EUR" ? `${body} €` : `${body} ${code}`;
}
