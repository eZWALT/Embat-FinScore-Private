import { LocalBundleRepository } from "./local-bundle-repository";
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

const CATEGORY_LABELS: Record<CategoryId, string> = {
  payment_history: "Historial de pagos",
  amounts_owed: "Liquidez y deuda",
  stability: "Estabilidad",
  new_credit: "Nuevo crédito",
  mix: "Mix de clientes",
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

function topReasonInSpanish(
  guard: "dark" | "fading" | null,
  reason: { item: string; points: number; eur: number | null } | undefined,
) {
  if (guard === "dark") {
    return "No hay movimientos bancarios recientes; el score se limita a 30 por seguridad.";
  }

  if (guard === "fading") {
    return "Las entradas de caja han caído frente al histórico propio; el score se limita a 50.";
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

  return `${label} representa la principal dimensión que limita el score (${impact} puntos).${amount}`;
}

export async function getDashboardData(
  repository: ScoreRepository = new LocalBundleRepository(),
): Promise<DashboardData> {
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
          label: CATEGORY_LABELS[id] ?? manifest.spec.categories[id].label,
          score: latest.categories[id].score,
        })),
      };
    }),
  );

  companies.sort((a, b) => b.score - a.score);

  return {
    asOfMonth: manifest.as_of_month,
    isSample: manifest.is_sample,
    disclaimer:
      "Score documentado y explicable calculado sobre el historial de tesorería de la empresa. Es una ayuda de monitorización, no un predictor de insolvencia; en los resultados aceptados no supera un baseline de tamaño empresarial.",
    companies,
  };
}
