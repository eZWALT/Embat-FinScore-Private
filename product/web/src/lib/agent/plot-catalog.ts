import { entityLabel } from "@/lib/display";
import { createScoreRepository } from "@/lib/data/repository";
import { CATEGORY_LABELS } from "@/lib/data/plain-language";
import type { CategoryId } from "@/lib/data/types";

import { neonQuery } from "./neon-sql";
import type { PlotSpec } from "./plot-spec";

/** Charts that already exist on the pages or in Javi’s monitor. No free-form series. */
export const PLOT_KINDS = [
  "score_history",
  "score_compare",
  "categories",
  "control_own",
  "control_cluster",
  "control_group",
  "forecast_fan",
  "group_members",
] as const;

export type PlotKind = (typeof PLOT_KINDS)[number];

const CONTROL_METRICS = ["score", "payment_history", "amounts_owed", "stability"] as const;
type ControlMetric = (typeof CONTROL_METRICS)[number];

export type PlotRequest = {
  kind: PlotKind;
  company_id?: string;
  group_id?: string;
  company_ids?: string[];
  metric?: ControlMetric;
};

function repo() {
  return createScoreRepository();
}

function asPlot(spec: PlotSpec): PlotSpec {
  return spec;
}

export async function buildCatalogPlot(req: PlotRequest): Promise<PlotSpec | { error: string }> {
  try {
    switch (req.kind) {
      case "score_history":
        return await scoreHistory(need(req.company_id, "company_id"));
      case "score_compare":
        return await scoreCompare(req.company_ids?.length ? req.company_ids : req.company_id ? [req.company_id] : []);
      case "categories":
        return await categories(need(req.company_id, "company_id"));
      case "control_own":
        return await controlChart(need(req.company_id, "company_id"), "own_history", req.metric ?? "score");
      case "control_cluster":
        return await controlChart(need(req.company_id, "company_id"), "cluster", req.metric ?? "score");
      case "control_group":
        return await controlChart(need(req.group_id, "group_id"), "group_own_history", "score");
      case "forecast_fan":
        return await forecastFan(need(req.company_id, "company_id"));
      case "group_members":
        return await groupMembers(need(req.group_id, "group_id"));
      default:
        return { error: "ese gráfico no está en el catálogo" };
    }
  } catch (error) {
    return { error: error instanceof Error ? error.message : String(error) };
  }
}

function need(value: string | undefined, name: string): string {
  if (!value) throw new Error(name.startsWith("group") ? "hace falta un grupo" : "hace falta una empresa");
  return value;
}

async function scoreHistory(companyId: string): Promise<PlotSpec> {
  const detail = await repo().getCompany(companyId);
  if (!detail.months.length) throw new Error(`${entityLabel(companyId)} no tiene meses puntuados`);
  return asPlot({
    title: `Índice de ${entityLabel(companyId)}`,
    x: detail.months.map((row) => row.month),
    series: { Índice: detail.months.map((row) => row.score) },
    kind: "line",
    y_label: "0–100",
  });
}

async function scoreCompare(ids: string[]): Promise<PlotSpec> {
  const unique = [...new Set(ids)].slice(0, 8);
  if (!unique.length) throw new Error("hacen falta empresas");
  const details = await Promise.all(unique.map((id) => repo().getCompany(id)));
  const months = [...new Set(details.flatMap((detail) => detail.months.map((row) => row.month)))].sort();
  const series: Record<string, (number | null)[]> = {};
  for (const detail of details) {
    const byMonth = new Map(detail.months.map((row) => [row.month, row.score]));
    series[entityLabel(detail.company_id)] = months.map((month) => byMonth.get(month) ?? null);
  }
  return asPlot({
    title: unique.length === 1 ? `Índice de ${entityLabel(unique[0])}` : "Índice de las empresas",
    x: months,
    series,
    kind: "line",
    y_label: "0–100",
  });
}

async function categories(companyId: string): Promise<PlotSpec> {
  const detail = await repo().getCompany(companyId);
  const latest = detail.months.at(-1);
  if (!latest) throw new Error(`${entityLabel(companyId)} no tiene meses puntuados`);
  const order: CategoryId[] = ["payment_history", "amounts_owed", "stability", "new_credit", "mix"];
  return asPlot({
    title: `Categorías · ${entityLabel(companyId)}`,
    x: order.map((id) => CATEGORY_LABELS[id]),
    series: { Puntos: order.map((id) => latest.categories[id]?.score ?? null) },
    kind: "bar",
    y_label: "0–100",
  });
}

async function controlChart(
  entityId: string,
  comparison: "own_history" | "cluster" | "group_own_history" | "group_vs_groups",
  metric: ControlMetric,
): Promise<PlotSpec> {
  const entityType = entityId.startsWith("GROUP") ? "group" : "company";
  const rows = await neonQuery<{
    months: string[];
    values: (number | null)[] | null;
    center: (number | null)[] | null;
    lower: (number | null)[] | null;
    upper: (number | null)[] | null;
  }>(
    `SELECT months, values, center, lower, upper
     FROM analytics.control_charts c
     JOIN api.current_run r ON r.run_id = c.run_id
     WHERE c.entity_type = $1 AND c.entity_id = $2 AND c.comparison = $3 AND c.metric = $4`,
    [entityType, entityId, comparison, metric],
  );
  const chart = rows[0];
  if (!chart?.months?.length) {
    throw new Error("este gráfico de control no está cargado (hacen falta 7 meses; un grupo, 3 empresas)");
  }
  const title =
    comparison === "cluster"
      ? `${entityLabel(entityId)} frente a su grupo de pares`
      : comparison.startsWith("group")
        ? `Media de ${entityLabel(entityId)}`
        : `${entityLabel(entityId)} frente a su histórico`;
  return asPlot({
    title,
    x: chart.months,
    series: {
      Valor: chart.values ?? chart.months.map(() => null),
      Centro: chart.center ?? chart.months.map(() => null),
    },
    kind: "line",
    band:
      chart.lower && chart.upper
        ? { lower: chart.lower, upper: chart.upper }
        : null,
    y_label: "0–100",
  });
}

async function forecastFan(companyId: string): Promise<PlotSpec> {
  const forecasts = await neonQuery<{ forecast_id: number; origin_month: string }>(
    `SELECT f.forecast_id, f.origin_month
     FROM analytics.forecasts f
     JOIN api.current_run r ON r.run_id = f.run_id
     WHERE f.company_id = $1`,
    [companyId],
  );
  const forecast = forecasts[0];
  if (!forecast) throw new Error("el abanico no está cargado (hacen falta 4 meses puntuados)");
  const points = await neonQuery<{
    month: string;
    median: number;
    lo80: number;
    hi80: number;
  }>(
    `SELECT month, median, lo80, hi80 FROM analytics.forecast_points WHERE forecast_id = $1 ORDER BY month`,
    [forecast.forecast_id],
  );
  const detail = await repo().getCompany(companyId);
  const history = detail.months.filter((row) => row.month <= forecast.origin_month).slice(-6);
  const x = [...history.map((row) => row.month), ...points.map((row) => row.month)];
  return asPlot({
    title: `Abanico de ${entityLabel(companyId)}`,
    x,
    series: {
      Índice: [...history.map((row) => row.score), ...points.map(() => null)],
      Mediana: [...history.map(() => null), ...points.map((row) => row.median)],
    },
    kind: "line",
    band: {
      lower: [...history.map(() => null), ...points.map((row) => row.lo80)],
      upper: [...history.map(() => null), ...points.map((row) => row.hi80)],
    },
    y_label: "0–100",
  });
}

async function groupMembers(groupId: string): Promise<PlotSpec> {
  const store = repo();
  const [groups, companies] = await Promise.all([store.listGroups(), store.listCompanies()]);
  const group = groups.find((row) => row.group_id === groupId);
  if (!group) throw new Error(`${entityLabel(groupId)} no está en esta corrida`);
  const members = group.company_ids
    .map((id) => companies.find((company) => company.company_id === id))
    .filter((row): row is NonNullable<typeof row> => Boolean(row))
    .sort((a, b) => a.score - b.score);
  if (!members.length) throw new Error("este grupo no tiene empresas puntuadas");
  return asPlot({
    title: `Empresas de ${entityLabel(groupId)}`,
    x: members.map((row) => entityLabel(row.company_id)),
    series: { Índice: members.map((row) => row.score) },
    kind: "bar",
    y_label: "0–100",
  });
}
