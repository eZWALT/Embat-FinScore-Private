import type { CategoryId } from "./types";

export const CATEGORY_LABELS: Record<CategoryId, string> = {
  payment_history: "Historial de pagos",
  amounts_owed: "Liquidez y deuda",
  stability: "Estabilidad",
  new_credit: "Nuevo crédito",
  mix: "Combinación de clientes",
};

const REASON_LABELS: Record<string, string> = {
  delay_paid: "El retraso en pagos a proveedores",
  delay_coll: "El retraso en cobros de clientes",
  ap_overdue30: "Las facturas de proveedores vencidas",
  ar_overdue30: "Las facturas de clientes vencidas",
  runway: "La cobertura de caja",
  neg_liq: "Los periodos con liquidez negativa",
  neg_episodes: "Los episodios de caja negativa",
  ds_ratio: "El servicio de deuda",
  fc_ratio: "Los costes financieros",
  months_observed: "La longitud del historial",
  active_share: "La recurrencia de entradas de caja",
  out_vol: "La volatilidad de salidas",
  ds_increase: "El aumento del servicio de deuda",
  fc_increase: "El aumento de costes financieros",
  cust_tail: "La concentración de clientes",
  credit_note: "El peso de las notas de crédito",
};

export function topReasonInSpanish(
  guard: "dark" | "fading" | null,
  reason: { item: string; points: number; eur: number | null } | undefined,
) {
  if (guard === "dark") {
    return "No hay movimientos bancarios recientes; la puntuación se limita a 30 por seguridad.";
  }

  if (guard === "fading") {
    return "Las entradas de caja han caído frente al histórico propio; la puntuación se limita a 50.";
  }

  if (!reason) return null;

  const label = REASON_LABELS[reason.item] ?? "Esta señal";
  const impact = Math.abs(reason.points).toFixed(1);
  const amount = reason.eur
    ? ` La exposición observada es ${new Intl.NumberFormat("es-ES", {
        style: "currency",
        currency: "EUR",
        notation: "compact",
        maximumFractionDigits: 1,
      }).format(reason.eur)}.`
    : "";

  return `${label} representa la principal dimensión que limita la puntuación (${impact} puntos).${amount}`;
}

export const MONITORING_DISCLAIMER =
  "Puntuación documentada y explicable calculada sobre el historial de tesorería de la empresa. Es una ayuda de monitorización, no un predictor de insolvencia; en los resultados aceptados no supera una referencia de tamaño empresarial.";
