import type {
  AlertKind,
  AlertSeverity,
  Confidence,
  Guard,
  Owner,
  Trajectory,
} from "@/lib/data/types";

export const trajectoryLabels: Record<Trajectory, string> = {
  improving: "Mejorando",
  stable: "Estable",
  dip: "Caída puntual",
  deteriorating: "Deteriorándose",
  "insufficient history": "Historial insuficiente",
};

export const confidenceLabels: Record<Confidence, string> = {
  high: "Alta",
  medium: "Media",
  low: "Baja",
};

export const guardLabels: Record<Guard, string> = {
  dark: "Sin movimientos",
  fading: "Entradas en caída",
};

export const severityLabels: Record<AlertSeverity, string> = {
  act: "Actuar",
  watch: "Vigilar",
  info: "Informativa",
};

export const ownerLabels: Record<Owner, string> = {
  treasurer: "Tesorero",
  cfo: "CFO",
  collections: "Cobros",
};

export const kindLabels: Record<AlertKind, string> = {
  score_deterioration: "Deterioro del score",
  score_improvement: "Mejora del score",
  category_drop: "Caída de una categoría",
  going_dark: "Sin movimientos bancarios",
  top_customer_quiet: "Cliente principal sin facturar",
};

/** Spanish label per score item; falls back to the bundle label when unknown. */
export const reasonLabels: Record<string, string> = {
  delay_paid: "Retraso en pagos a proveedores",
  delay_coll: "Retraso en cobros de clientes",
  ap_overdue30: "Facturas de proveedores vencidas",
  ar_overdue30: "Facturas de clientes vencidas",
  runway: "Cobertura de caja",
  neg_liq: "Periodos con liquidez negativa",
  neg_episodes: "Episodios de caja negativa",
  ds_ratio: "Servicio de deuda",
  fc_ratio: "Costes financieros",
  months_observed: "Longitud del historial",
  active_share: "Recurrencia de entradas de caja",
  out_vol: "Volatilidad de salidas",
  ds_increase: "Aumento del servicio de deuda",
  fc_increase: "Aumento de costes financieros",
  cust_tail: "Concentración de clientes",
  credit_note: "Peso de las notas de crédito",
  guard: "Límite de seguridad aplicado",
  top_customer: "Cliente principal sin facturar",
};

const monthFormatter = new Intl.DateTimeFormat("es-ES", { month: "short", year: "2-digit" });
const monthOnlyFormatter = new Intl.DateTimeFormat("es-ES", { month: "short" });

function monthDate(month: string) {
  const [year, monthNumber] = month.split("-").map(Number);
  return new Date(Date.UTC(year, monthNumber - 1, 1));
}

/** "2026-08" -> "ago ’26" */
export function formatMonth(month: string) {
  return monthFormatter.format(monthDate(month)).replace(" ", " ’");
}

/** "2026-08" -> ["ago", "26"] for two-line column headers. */
export function splitMonth(month: string): [string, string] {
  return [monthOnlyFormatter.format(monthDate(month)).replace(".", ""), month.slice(2, 4)];
}

export function formatEur(value: number, currency: string | null = "EUR") {
  return new Intl.NumberFormat("es-ES", {
    style: "currency",
    currency: currency ?? "EUR",
    notation: "compact",
    maximumFractionDigits: 1,
  }).format(value);
}

export function formatPoints(value: number, signed = true) {
  const text = Math.abs(value).toFixed(1).replace(".", ",");
  if (!signed) return text;
  return value > 0 ? `+${text}` : value < 0 ? `−${text}` : text;
}

/** 0 = red, 50 = amber, 100 = green. Same lightness and chroma in both themes. */
export function scoreColor(score: number | null): string | undefined {
  if (score === null || Number.isNaN(score)) return undefined;
  const s = Math.max(0, Math.min(100, score));
  const hue = s <= 50 ? 25 + (s / 50) * 60 : 85 + ((s - 50) / 50) * 60;
  return `oklch(0.74 0.15 ${hue.toFixed(0)})`;
}

export const severityBadgeVariant: Record<AlertSeverity, "destructive" | "secondary" | "outline"> = {
  act: "destructive",
  watch: "secondary",
  info: "outline",
};
