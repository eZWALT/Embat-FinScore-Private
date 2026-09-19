export const TOOL_LABELS = {
  list_companies: "Listar empresas",
  get_company: "Leer índice",
  explain_change: "Explicar el cambio",
  get_group: "Leer grupo",
  get_alerts: "Leer alertas",
  get_control_chart: "Control chart",
  compare_with_cluster: "Comparar con pares",
  get_forecast: "Abanico",
  query_clean_db: "Consultar registros",
  plot_series: "Dibujar gráfico",
} as const;

export type ToolName = keyof typeof TOOL_LABELS;

export function toolLabel(name: string): string {
  return TOOL_LABELS[name as ToolName] ?? name;
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
  if (typeof row.error === "string") return row.error;
  switch (name) {
    case "get_company":
      return [row.company_id, row.month, row.score != null ? `índice ${row.score}` : null, row.trajectory]
        .filter(Boolean)
        .join(" · ");
    case "explain_change":
      return [row.company_id, row.from && row.to ? `${row.from} → ${row.to}` : null, row.change != null ? `${row.change} pts` : null]
        .filter(Boolean)
        .join(" · ");
    case "get_alerts":
      return `${row.n_matching ?? (Array.isArray(row.alerts) ? row.alerts.length : 0)} alertas`;
    case "get_group":
      return [row.group_id, row.n_companies != null ? `${row.n_companies} empresas` : null, row.latest_mean_score != null ? `media ${row.latest_mean_score}` : null]
        .filter(Boolean)
        .join(" · ");
    case "list_companies":
      return `${Array.isArray(row.companies) ? row.companies.length : 0} empresas · ${row.as_of ?? ""}`.trim();
    case "get_control_chart":
      return [row.comparison, row.metric, Array.isArray(row.months) ? `${row.months.length} meses` : null]
        .filter(Boolean)
        .join(" · ");
    case "compare_with_cluster":
      return row.cluster && typeof row.cluster === "object" && "label" in row.cluster
        ? String((row.cluster as { label?: string }).label)
        : "grupo de pares";
    case "get_forecast":
      return [row.method, row.origin_month, row.horizon_months != null ? `${row.horizon_months} meses` : null]
        .filter(Boolean)
        .join(" · ");
    case "query_clean_db":
      return `${row.rows ?? 0} filas`;
    case "plot_series":
      return typeof row.title === "string" ? row.title : `${row.points ?? 0} puntos`;
    default:
      return "";
  }
}

export function toolTimingMs(output: unknown): number | null {
  if (!output || typeof output !== "object") return null;
  const value = (output as { timing_ms?: unknown }).timing_ms;
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}
