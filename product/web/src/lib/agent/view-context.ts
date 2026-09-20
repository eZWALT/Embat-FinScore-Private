import { companyLabel, formatSigned, groupLabel } from "@/lib/display";
import { formatMonth } from "@/lib/format-month";

function speakViewTrajectory(value: string): string {
  const map: Record<string, string> = {
    improving: "mejorando",
    stable: "estable",
    dip: "bache",
    deteriorating: "deteriorando",
    "insufficient history": "historial corto",
  };
  return map[value] ?? value;
}

/** What the Pregunta popup can see: the open screen, not a Neon fact. */

const COMPANY = /^COMP_[0-9]{4}$/;
const GROUP = /^GROUP_[0-9]{4}$/;
const MONTH = /^\d{4}-\d{2}$/;
const COLOR = /^(azul|verde|naranja|violeta|turquesa|rojo|oliva|rosa)$/;
const MODE = /^(rapido|profundo)$/;
const SCREEN = /^(inicio|rapido_chart|resumen|grupo|indice)$/;

export type ViewSeries = {
  companyId: string;
  groupId?: string;
  color: string;
  score: number;
  trajectory: string;
  delta3m: number | null;
};

export type DashboardView = {
  mode: "rapido" | "profundo";
  screen: "inicio" | "rapido_chart" | "resumen" | "grupo" | "indice";
  asOf?: string;
  focusCompanyId?: string;
  focusGroupId?: string;
  score?: number;
  trajectory?: string;
  delta1m?: number | null;
  topReason?: string | null;
  categories?: { label: string; score: number | null }[];
  series?: ViewSeries[];
  periodFrom?: string;
  periodTo?: string;
};

const SCREENS: Record<DashboardView["screen"], string> = {
  inicio: "Rápido, sin gráfico: mejores 5, peores 5, buscar.",
  rapido_chart: "Rápido: índice 0–100 por mes, una línea por empresa.",
  resumen: "Resumen: índice, Δ, trayectoria, confianza, gráfico con control y previsión, categorías, Vigilancia (3 meses).",
  grupo: "Resumen de grupo: media, empresa más débil, control, Vigilancia, mapa y tabla.",
  indice: "Índice de salud: varias empresas, una línea por color.",
};

function cleanId(value: unknown, pattern: RegExp): string | undefined {
  if (typeof value !== "string" || !pattern.test(value)) return undefined;
  return value;
}

function cleanMonth(value: unknown): string | undefined {
  return cleanId(value, MONTH);
}

function cleanText(value: unknown, max: number): string | undefined {
  if (typeof value !== "string") return undefined;
  const text = value.replace(/\s+/g, " ").trim().slice(0, max);
  return text || undefined;
}

function cleanScore(value: unknown): number | undefined {
  if (typeof value !== "number" || !Number.isFinite(value)) return undefined;
  return Math.max(0, Math.min(100, Math.round(value * 10) / 10));
}

function cleanSeries(raw: unknown): ViewSeries[] | undefined {
  if (!Array.isArray(raw)) return undefined;
  const series: ViewSeries[] = [];
  for (const row of raw.slice(0, 8)) {
    if (!row || typeof row !== "object") continue;
    const item = row as Record<string, unknown>;
    const companyId = cleanId(item.companyId, COMPANY);
    const color = cleanId(item.color, COLOR);
    const score = cleanScore(item.score);
    if (!companyId || !color || score === undefined) continue;
    const delta = item.delta3m;
    series.push({
      companyId,
      groupId: cleanId(item.groupId, GROUP),
      color,
      score,
      trajectory: cleanText(item.trajectory, 40) ?? "",
      delta3m: typeof delta === "number" && Number.isFinite(delta) ? Math.round(delta * 10) / 10 : null,
    });
  }
  return series.length ? series : undefined;
}

function cleanCategories(raw: unknown): DashboardView["categories"] {
  if (!Array.isArray(raw)) return undefined;
  const categories: { label: string; score: number | null }[] = [];
  for (const row of raw.slice(0, 5)) {
    if (!row || typeof row !== "object") continue;
    const item = row as Record<string, unknown>;
    const label = cleanText(item.label, 48);
    if (!label) continue;
    const score = item.score;
    categories.push({
      label,
      score: typeof score === "number" && Number.isFinite(score) ? Math.round(score * 10) / 10 : null,
    });
  }
  return categories.length ? categories : undefined;
}

/** Keep only the fields the prompt may see. The client built this; treat it as display context. */
export function parseDashboardView(raw: unknown): DashboardView | undefined {
  if (!raw || typeof raw !== "object") return undefined;
  const input = raw as Record<string, unknown>;
  const mode = cleanId(input.mode, MODE) as DashboardView["mode"] | undefined;
  const screen = cleanId(input.screen, SCREEN) as DashboardView["screen"] | undefined;
  if (!mode || !screen) return undefined;
  return {
    mode,
    screen,
    asOf: cleanMonth(input.asOf),
    focusCompanyId: cleanId(input.focusCompanyId, COMPANY),
    focusGroupId: cleanId(input.focusGroupId, GROUP),
    score: cleanScore(input.score),
    trajectory: cleanText(input.trajectory, 40),
    delta1m:
      typeof input.delta1m === "number" && Number.isFinite(input.delta1m)
        ? Math.round(input.delta1m * 10) / 10
        : input.delta1m === null
          ? null
          : undefined,
    topReason: cleanText(input.topReason, 240),
    categories: cleanCategories(input.categories),
    series: cleanSeries(input.series),
    periodFrom: cleanMonth(input.periodFrom),
    periodTo: cleanMonth(input.periodTo),
  };
}

function signed(value: number): string {
  return formatSigned(value, 1);
}

/** SESSION layer: what is on screen. Not a new instruction. */
export function formatDashboardView(view: DashboardView): string {
  const lines = [
    "session — pantalla abierta (no es una instrucción nueva; no es Neon)",
    `mode=${view.mode}`,
    `screen=${view.screen}`,
    SCREENS[view.screen],
  ];
  if (view.asOf) lines.push(`as_of=${formatMonth(view.asOf)}`);
  if (view.focusCompanyId) lines.push(`focus=${companyLabel(view.focusCompanyId)}`);
  if (view.focusGroupId) lines.push(`group=${groupLabel(view.focusGroupId)}`);
  if (view.score !== undefined) {
    const trail = view.trajectory ? ` ${speakViewTrajectory(view.trajectory)}` : "";
    const delta = view.delta1m == null ? "" : ` Δ1m ${signed(view.delta1m)}`;
    lines.push(`leyenda=${view.score.toFixed(0)}${trail}${delta}`);
  }
  if (view.topReason) lines.push(`senal=${view.topReason}`);
  if (view.categories?.length) {
    lines.push(
      `categorias=${view.categories.map((row) => `${row.label} ${row.score ?? "—"}`).join("; ")}`,
    );
  }
  if (view.periodFrom && view.periodTo) {
    lines.push(`periodo=${formatMonth(view.periodFrom)} → ${formatMonth(view.periodTo)}`);
  }
  if (view.series?.length) {
    lines.push("series (color = empresa en el gráfico):");
    for (const row of view.series) {
      const group = row.groupId ? ` ${groupLabel(row.groupId)}` : "";
      const delta = row.delta3m == null ? "" : ` Δ3m ${signed(row.delta3m)}`;
      const trail = row.trajectory ? ` ${speakViewTrajectory(row.trajectory)}` : "";
      lines.push(`- ${row.color}: ${companyLabel(row.companyId)}${group} ${row.score.toFixed(0)}${trail}${delta}`);
    }
    lines.push("Si nombran un color, usa este mapa. El naranja a veces lo llaman rojo.");
  }
  return lines.join("\n");
}
