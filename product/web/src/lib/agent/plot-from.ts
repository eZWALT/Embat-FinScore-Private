import type { ModelMessage } from "ai";

import type { PlotSpec } from "./plot-spec";

/** Tools whose JSON result may be drawn. The model points at one; the server reads the numbers. */
export const PLOT_SOURCES = [
  "get_group",
  "get_alerts",
  "list_companies",
  "get_company",
  "explain_change",
  "compare_with_cluster",
  "query_clean_db",
] as const;

export type PlotSource = (typeof PLOT_SOURCES)[number];

export type PlotFromRequest = {
  source: PlotSource;
  path?: string;
  x: string;
  y: string[];
  kind: "bar" | "line" | "pie";
  sort?: "asc" | "desc" | "none";
  ref_line?: string;
  badge?: string;
  title: string;
};

const MAX_POINTS = 50;
const MAX_SLICES = 8;

type Row = Record<string, unknown>;

/** Latest JSON output of `source` in this turn (or earlier turns kept in the thread). */
export function findToolResult(messages: ModelMessage[], source: string): unknown {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i];
    if (message.role !== "tool" || !Array.isArray(message.content)) continue;
    for (const part of message.content) {
      if (part.type !== "tool-result" || part.toolName !== source) continue;
      const output = part.output as { type?: string; value?: unknown } | undefined;
      if (!output) continue;
      if (output.type === "json") return output.value;
      if (output.type === "text" && typeof output.value === "string") {
        try {
          return JSON.parse(output.value);
        } catch {
          return undefined;
        }
      }
    }
  }
  return undefined;
}

function walk(value: unknown, path?: string): unknown {
  if (!path) return value;
  let node = value;
  for (const key of path.split(".")) {
    if (node == null || typeof node !== "object") return undefined;
    node = (node as Row)[key];
  }
  return node;
}

/** First array of objects inside the result when `path` is not given. */
function firstArray(value: unknown): { rows: Row[]; path: string } | null {
  if (Array.isArray(value)) return { rows: value as Row[], path: "" };
  if (!value || typeof value !== "object") return null;
  const row = value as Row;
  // query_clean_db → {columns, data: [[...]]}
  if (Array.isArray(row.columns) && Array.isArray(row.data)) {
    const columns = row.columns as string[];
    return {
      rows: (row.data as unknown[][]).map((cells) => Object.fromEntries(columns.map((c, i) => [c, cells[i]]))),
      path: "data",
    };
  }
  for (const [key, child] of Object.entries(row)) {
    if (Array.isArray(child) && child.length && child.every((item) => item && typeof item === "object" && !Array.isArray(item))) {
      return { rows: child as Row[], path: key };
    }
  }
  return null;
}

function toNumber(value: unknown): number | null {
  if (typeof value === "number") return Number.isFinite(value) ? value : null;
  if (typeof value === "string") {
    const parsed = Number(value.replace(/\s|€|%/g, "").replace(",", "."));
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function yLabel(columns: string[]): string | null {
  const joined = columns.join(" ").toLowerCase();
  if (/score|puntuaci|indice|índice|pts|puntos|media|percentil/.test(joined)) return "0–100";
  if (/importe|eur|amount|saldo|caja|€/.test(joined)) return "€";
  if (/pct|share|cuota|porcentaje|%/.test(joined)) return "%";
  return null;
}

export function buildPlotFromResult(
  req: PlotFromRequest,
  messages: ModelMessage[],
): { plot: PlotSpec; source_path: string } | { error: string } {
  const result = findToolResult(messages, req.source);
  if (result === undefined) return { error: `primero llama a ${req.source}; no hay resultado que dibujar` };
  if (result && typeof result === "object" && typeof (result as Row).error === "string") {
    return { error: `${req.source} devolvió un error, no hay datos que dibujar` };
  }

  const scoped = walk(result, req.path);
  const table = firstArray(scoped ?? result);
  if (!table || !table.rows.length) return { error: `no hay una lista de filas en ${req.source}${req.path ? `.${req.path}` : ""}` };
  const rows = table.rows;
  const available = Object.keys(rows[0]);
  const missing = [req.x, ...req.y].filter((column) => !(column in rows[0]));
  if (missing.length) return { error: `columnas desconocidas: ${missing.join(", ")}. Disponibles: ${available.join(", ")}` };

  let points = rows
    .map((row) => ({
      label: String(row[req.x] ?? ""),
      values: req.y.map((column) => toNumber(row[column])),
      badge: req.badge ? row[req.badge] : undefined,
    }))
    .filter((point) => point.label && point.values.some((value) => value != null));
  if (!points.length) return { error: `las columnas ${req.y.join(", ")} no son numéricas` };
  if (points.length > MAX_POINTS) return { error: `demasiadas filas (${points.length}); filtra antes a ${MAX_POINTS} o menos` };

  const sort = req.sort ?? "none";
  if (sort !== "none") {
    points = [...points].sort((a, b) => ((a.values[0] ?? 0) - (b.values[0] ?? 0)) * (sort === "asc" ? 1 : -1));
  }

  let kind = req.kind;
  let note: string | undefined;
  if (kind === "pie") {
    const flat = points.flatMap((point) => point.values);
    const isShare = req.y.length === 1 && points.length <= MAX_SLICES && flat.every((value) => value == null || value >= 0);
    const looksLikeScore = yLabel(req.y) === "0–100";
    if (!isShare || looksLikeScore) {
      kind = "bar";
      note = looksLikeScore
        ? "un índice 0–100 no es una cuota: se dibuja en barras"
        : `un pastel necesita una sola serie no negativa con ≤ ${MAX_SLICES} porciones: se dibuja en barras`;
    }
  }

  const series: Record<string, (number | null)[]> = {};
  req.y.forEach((column, i) => {
    series[column] = points.map((point) => point.values[i]);
  });

  const ref_lines: { label: string; value: number }[] = [];
  let highlight: string[] = [];
  if (req.ref_line) {
    const scalar = toNumber(walk(result, req.ref_line));
    if (scalar == null) return { error: `${req.ref_line} no es un número en ${req.source}` };
    ref_lines.push({ label: refLabel(req.ref_line), value: scalar });
    highlight = points.filter((point) => (point.values[0] ?? Infinity) < scalar).map((point) => point.label);
  }

  const badges: Record<string, string> = {};
  if (req.badge) {
    for (const point of points) {
      const value = point.badge;
      const on = typeof value === "number" ? value > 0 : typeof value === "boolean" ? value : typeof value === "string" ? value.trim() !== "" && value !== "0" : false;
      if (on) badges[point.label] = badgeLabel(req.badge, value);
    }
  }

  return {
    plot: {
      title: note ? `${req.title} (${note})` : req.title,
      x: points.map((point) => point.label),
      series,
      kind,
      y_label: yLabel(req.y),
      ref_lines,
      highlight,
      badges,
    },
    source_path: `${req.source}${table.path ? `.${table.path}` : ""}`,
  };
}

function refLabel(path: string): string {
  const key = path.split(".").at(-1) ?? path;
  if (/mean|media/.test(key)) return "Media del grupo";
  if (/min/.test(key)) return "Mínimo";
  if (/max/.test(key)) return "Máximo";
  return key;
}

function badgeLabel(column: string, value: unknown): string {
  if (/alert/.test(column)) return typeof value === "number" && value > 1 ? `${value} alertas` : "alerta";
  if (/guard|dark/.test(column)) return "en silencio";
  return typeof value === "string" ? value : column;
}
