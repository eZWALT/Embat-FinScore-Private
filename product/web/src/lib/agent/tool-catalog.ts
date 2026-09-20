import type { ComponentType, SVGProps } from "react";

import { entityLabel } from "@/lib/display";
import {
  ArrowsRightLeftIcon,
  BellAlertIcon,
  BuildingOffice2Icon,
  ChartBarIcon,
  ChartPieIcon,
  CircleStackIcon,
  PresentationChartLineIcon,
  ScaleIcon,
  SignalIcon,
  UserGroupIcon,
  WrenchScrewdriverIcon,
} from "@heroicons/react/24/outline";

type ToolIcon = ComponentType<SVGProps<SVGSVGElement>>;

export const TOOL_LABELS = {
  list_companies: "Listar empresas",
  get_company: "Leer índice",
  explain_change: "Explicar el cambio",
  get_group: "Leer grupo",
  get_alerts: "Leer alertas",
  get_control_chart: "Vs su histórico",
  compare_with_cluster: "Comparar con pares",
  get_forecast: "Abanico",
  query_clean_db: "Leer registros",
  plot_series: "Gráfico",
} as const;

export type ToolName = keyof typeof TOOL_LABELS;

export const TOOL_ICONS: Record<ToolName, ToolIcon> = {
  list_companies: BuildingOffice2Icon,
  get_company: ChartBarIcon,
  explain_change: ArrowsRightLeftIcon,
  get_group: UserGroupIcon,
  get_alerts: BellAlertIcon,
  get_control_chart: SignalIcon,
  compare_with_cluster: ScaleIcon,
  get_forecast: ChartPieIcon,
  query_clean_db: CircleStackIcon,
  plot_series: PresentationChartLineIcon,
};

export function toolLabel(name: string): string {
  return TOOL_LABELS[name as ToolName] ?? name;
}

export function toolIcon(name: string): ToolIcon {
  return TOOL_ICONS[name as ToolName] ?? WrenchScrewdriverIcon;
}

export function toolInputSummary(name: string, input: unknown): string {
  if (!input || typeof input !== "object") return "";
  const row = input as Record<string, unknown>;
  if (name === "query_clean_db" && typeof row.sql === "string") {
    return row.sql.replace(/\s+/g, " ").trim().slice(0, 160);
  }
  const bits = Object.entries(row)
    .filter(([, value]) => value != null && value !== "")
    .map(([key, value]) => `${key}=${Array.isArray(value) ? value.join(",") : String(value)}`);
  return bits.join(" · ");
}

export function toolOutputSummary(name: string, output: unknown): string {
  if (output == null) return "";
  if (typeof output !== "object") return String(output).slice(0, 160);
  const row = output as Record<string, unknown>;
  if (typeof row.error === "string") {
    if (row.error === "cap") return "Ya lo tienes · responde con lo recuperado";
    if (row.error === "records not mounted" || row.error === "registros no montados") return "Registros no montados";
    return row.error;
  }
  switch (name) {
    case "get_company":
      return [
        typeof row.empresa === "string"
          ? row.empresa
          : typeof row.company_id === "string"
            ? entityLabel(row.company_id)
            : null,
        row.month,
        row.score != null ? `índice ${row.score}` : null,
        row.trajectory,
      ]
        .filter(Boolean)
        .join(" · ");
    case "explain_change":
      return [
        typeof row.empresa === "string" ? row.empresa : row.company_id,
        (row.desde ?? row.from) && (row.hasta ?? row.to)
          ? `${row.desde ?? row.from} → ${row.hasta ?? row.to}`
          : null,
        (row.cambio ?? row.change) != null ? `${row.cambio ?? row.change} pts` : null,
      ]
        .filter(Boolean)
        .join(" · ");
    case "get_alerts": {
      const n = row.n_matching ?? (Array.isArray(row.alerts) ? row.alerts.length : 0);
      const scoped = Array.isArray(row.scoped_to) && row.scoped_to.length ? ` · ${row.scoped_to.length} entidades` : "";
      return `${n} alertas${scoped}`;
    }
    case "get_group":
      return [
        typeof row.grupo === "string" ? row.grupo : row.group_id,
        (row.n_empresas ?? row.n_companies) != null ? `${row.n_empresas ?? row.n_companies} empresas` : null,
        (row.media ?? row.latest_mean_score) != null ? `media ${row.media ?? row.latest_mean_score}` : null,
      ]
        .filter(Boolean)
        .join(" · ");
    case "list_companies":
      return `${Array.isArray(row.companies) ? row.companies.length : 0} empresas · ${row.as_of ?? ""}`.trim();
    case "get_control_chart":
      return [row.comparison, row.metric, Array.isArray(row.months) ? `${row.months.length} meses` : null]
        .filter(Boolean)
        .join(" · ");
    case "compare_with_cluster":
      return typeof row.grupo_pares === "string"
        ? row.grupo_pares
        : row.cluster && typeof row.cluster === "object" && "label" in row.cluster
          ? String((row.cluster as { label?: string }).label)
          : "grupo de pares";
    case "get_forecast":
      return [
        row.metodo ?? row.method,
        row.origen ?? row.origin_month,
        (row.horizonte_meses ?? row.horizon_months) != null ? `${row.horizonte_meses ?? row.horizon_months} meses` : null,
      ]
        .filter(Boolean)
        .join(" · ");
    case "query_clean_db":
      return `${row.rows ?? 0} filas`;
    case "plot_series":
      return typeof row.title === "string" ? row.title : typeof row.kind === "string" ? row.kind : `${row.points ?? 0} puntos`;
    default:
      return "";
  }
}

export function toolTimingMs(output: unknown): number | null {
  if (!output || typeof output !== "object") return null;
  const value = (output as { timing_ms?: unknown }).timing_ms;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
