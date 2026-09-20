import { tool } from "ai";
import { z } from "zod";

import { entityLabel, formatDecimal, formatMoney, formatSigned, relabelEntities } from "@/lib/display";
import { formatMonth, formatMonthShort, parseMonth } from "@/lib/format-month";
import { CATEGORY_LABELS, REASON_LABELS } from "@/lib/data/plain-language";
import { createScoreRepository } from "@/lib/data/repository";
import type {
  Alert,
  AlertKind,
  AlertSeverity,
  CategoryId,
  CompanyDetail,
  Comparison,
  GroupRow,
  MonthRecord,
  Reason,
  ScoreRepository,
} from "@/lib/data/types";

import { neonQuery, coreIsMounted } from "./neon-sql";
import { buildCatalogPlot, PLOT_KINDS } from "./plot-catalog";
import { checkSql, toCoreSql, UnsafeQuery } from "./sql-guard";

const ALERT_KINDS = [
  "score_deterioration",
  "score_improvement",
  "category_drop",
  "going_dark",
  "top_customer_quiet",
] as const satisfies readonly AlertKind[];

const ALERT_SEVERITIES = ["info", "watch", "act"] as const satisfies readonly AlertSeverity[];

const COMPARISONS = [
  "own_history",
  "cluster",
  "group_own_history",
  "group_vs_groups",
] as const satisfies readonly Comparison[];

const METRICS = [
  "score",
  "payment_history",
  "amounts_owed",
  "stability",
  "new_credit",
  "mix",
] as const satisfies readonly ("score" | CategoryId)[];

type ItemRow = {
  value: number | null;
  points: number;
  contribution: number;
  delta: number | null;
};

type ScoreExtras = {
  slope3: number | null;
  slope6: number | null;
  change_guard: number | null;
  guard_adjustment: number | null;
  items: Record<string, ItemRow>;
};

type ClusterMeta = {
  cluster_id: string;
  label: string;
  description: string | null;
  n_companies: number | null;
  n_companies_train_fit: number | null;
};

type Json = Record<string, unknown>;

function repo(): ScoreRepository {
  return createScoreRepository();
}

function round(value: number | null | undefined, digits = 1): number | null {
  if (value == null || Number.isNaN(Number(value))) return null;
  return Number(Number(value).toFixed(digits));
}

function asError(error: unknown): Json {
  if (error instanceof Error) return { error: error.message };
  return { error: String(error) };
}

function slimReason(reason: Reason, currency: string | null = "EUR") {
  return {
    points: reason.points,
    eur: reason.eur != null ? formatMoney(reason.eur, currency) : null,
    sentence: reason.sentence,
  };
}

function speakMonth(month: string | null | undefined, long = false): string | undefined {
  if (!month) return undefined;
  const key = parseMonth(month) ?? month;
  if (!/^\d{4}-\d{2}$/.test(key)) return month;
  return long ? formatMonth(key) : formatMonthShort(key);
}

function speakScore(value: number | null | undefined): string | undefined {
  if (value == null || Number.isNaN(Number(value))) return undefined;
  return formatDecimal(Number(value), 1);
}

function speakDelta(value: number | null | undefined): string | undefined {
  if (value == null || Number.isNaN(Number(value))) return undefined;
  return formatSigned(Number(value), 1);
}

function monthKey(month: string | null | undefined): string | null {
  return parseMonth(month);
}

function asCompanyId(raw: string): string {
  const token = raw.match(/COMP_(\d{1,4})/i);
  if (token) return `COMP_${token[1].padStart(4, "0")}`;
  const empresa = raw.match(/^(?:Empresa\s+)?(\d{1,4})$/i);
  if (empresa) return `COMP_${empresa[1].padStart(4, "0")}`;
  return raw;
}

function asGroupId(raw: string): string {
  const token = raw.match(/GROUP_(\d{1,4})/i);
  if (token) return `GROUP_${token[1].padStart(4, "0")}`;
  const grupo = raw.match(/^(?:Grupo\s+)?(\d{1,4})$/i);
  if (grupo) return `GROUP_${grupo[1].padStart(4, "0")}`;
  return raw;
}

function asEntityId(raw: string): string {
  if (/GROUP_|Grupo\s/i.test(raw)) return asGroupId(raw);
  return asCompanyId(raw);
}

function speakGuard(guard: string | null | undefined): string {
  if (guard === "dark") return "sin movimientos (tope 30)";
  if (guard === "fading") return "entradas hundidas (tope 50)";
  return "sin tope";
}

function speakTrajectory(value: string | null | undefined): string | undefined {
  const map: Record<string, string> = {
    improving: "mejorando",
    stable: "estable",
    dip: "bache",
    deteriorating: "deteriorando",
    "insufficient history": "historial corto",
  };
  return value ? (map[value] ?? value) : undefined;
}

function speakConfidence(value: string | null | undefined): string | undefined {
  if (value === "high") return "alta";
  if (value === "medium") return "media";
  if (value === "low") return "baja";
  return value ?? undefined;
}

function speakOwner(value: string | null | undefined): string | undefined {
  if (value === "treasurer") return "Tesorero";
  if (value === "cfo") return "CFO";
  if (value === "collections") return "Cobros";
  return value ?? undefined;
}

function speakSeverity(value: string | null | undefined): string | undefined {
  if (value === "act") return "actuar";
  if (value === "watch") return "vigilar";
  if (value === "info") return "informativa";
  return value ?? undefined;
}

/** Full reasons on the focus month, the last 3, or a move of ≥2 pts. Period: scores only — reasons sit on the payload. */
function scoreHistory(
  months: MonthRecord[],
  focusMonth: string,
  span = 18,
  currency: string | null = "EUR",
  scoresOnly = false,
) {
  const focusIdx = months.findIndex((row) => row.month === focusMonth);
  const lookback = span >= 12 ? 6 : 1;
  const from = Math.max(0, months.length - span, focusIdx >= 0 ? focusIdx - lookback : 0);
  const window = months.slice(from);
  return window.map((row, index) => {
    const prev = window[index - 1] ?? months[from - 1];
    const moved = prev != null && Math.abs(row.score - prev.score) >= 2;
    const focus = row.month === focusMonth;
    const recent = from + index >= months.length - 3;
    const base: Json = {
      month: speakMonth(row.month),
      score: speakScore(row.score),
    };
    const trajectory = speakTrajectory(row.trajectory);
    const guard = speakGuard(row.guard);
    if (trajectory) base.trajectory = trajectory;
    if (guard && !(scoresOnly && guard === "sin tope")) base.guard = guard;
    if (scoresOnly || (!moved && !focus && !recent)) return base;
    return {
      ...base,
      reasons: (row.reasons ?? []).map((reason) => slimReason(reason, currency)),
      change_reasons: (row.change_reasons ?? []).map((reason) => slimReason(reason, currency)),
    };
  });
}

function slimItems(items: Record<string, ItemRow>): Record<string, ItemRow> {
  const rows = Object.entries(items);
  if (rows.length <= 8) return items;
  return Object.fromEntries(
    rows
      .sort((a, b) => Math.abs(Number(b[1].delta ?? 0)) - Math.abs(Number(a[1].delta ?? 0)))
      .slice(0, 8),
  );
}

function itemsFromMonth(rec: MonthRecord): Record<string, ItemRow> {
  const raw = (rec as MonthRecord & { items?: Record<string, ItemRow> }).items;
  if (!raw || typeof raw !== "object") return {};
  return slimItems(raw);
}

const EVIDENCE_KEYS: Record<string, string> = {
  baseline: "habitual",
  category: "categoría",
  category_score: "puntos_categoría",
  chart: "gráfico",
  counterparty_id: "contrapartida",
  gap_points: "brecha_pts",
  last_quarter_amount: "facturado_trimestre",
  mean_score: "media",
  members_moving_most: "miembros",
  months_billed_of_last_3: "meses_facturados_de_3",
  months_quiet: "meses_en_silencio",
  n_companies: "n_empresas",
  open_receivable_eur: "pendiente_cobro",
  outside_funnel_vs_other_groups: "fuera_embudo",
  recency_days: "días",
  score: "índice",
  score_pre_cap: "sin_tope",
  share_last_quarter: "cuota_trimestre",
};

const EVIDENCE_MONEY = new Set(["last_quarter_amount", "open_receivable_eur"]);
const EVIDENCE_PCT = new Set(["share_last_quarter"]);
const EVIDENCE_SCORE = new Set(["score", "score_pre_cap", "baseline", "mean_score", "category_score", "gap_points"]);
const CHART_LABELS: Record<string, string> = {
  own_history: "vs su histórico",
  cluster: "vs pares",
  group_own_history: "grupo vs su histórico",
  group_vs_groups: "grupo vs otros grupos",
};

function slimEvidence(evidence: Alert["evidence"], currency: string | null = "EUR") {
  const out: Record<string, string | number | boolean> = {};
  for (const [key, value] of Object.entries(evidence ?? {})) {
    if (value == null) continue;
    const name = EVIDENCE_KEYS[key] ?? key;
    if (typeof value === "string") {
      if (key === "category") out[name] = CATEGORY_LABELS[value as CategoryId] ?? value;
      else if (key === "chart") out[name] = CHART_LABELS[value] ?? value;
      else out[name] = relabelEntities(value);
      continue;
    }
    if (typeof value === "number") {
      if (EVIDENCE_MONEY.has(key)) out[name] = formatMoney(value, currency);
      else if (EVIDENCE_PCT.has(key)) out[name] = `${formatDecimal(value * 100, 0)} %`;
      else if (EVIDENCE_SCORE.has(key)) out[name] = formatDecimal(value, 1);
      else out[name] = value;
      continue;
    }
    if (typeof value === "boolean") out[name] = value;
  }
  return out;
}

function slimCategories(categories: MonthRecord["categories"]) {
  return Object.entries(categories).map(([id, row]) => ({
    name: CATEGORY_LABELS[id as CategoryId] ?? id,
    score: speakScore(row.score),
    pts: round(row.contribution, 1),
  }));
}

function slimAlert(alert: Alert) {
  const flagged = alert.persistence?.months_flagged;
  return {
    entity: entityLabel(alert.entity.id),
    month: speakMonth(alert.month, true),
    title: alert.title,
    reasons: (alert.reasons ?? []).map((reason) => slimReason(reason)).slice(0, 2),
    owner: speakOwner(alert.owner),
    severity: speakSeverity(alert.severity),
    action: alert.action,
    persistence: flagged != null ? `${flagged} de los últimos 4 meses` : "3 de los últimos 4 meses",
    evidence: slimEvidence(alert.evidence),
  };
}

function toolStepOutput(result: { output?: unknown; result?: unknown }): unknown {
  return result.output ?? result.result;
}

function companyPayloads(
  steps: { toolResults?: { toolName: string; output?: unknown; result?: unknown }[] }[],
): Record<string, unknown>[] {
  const out: Record<string, unknown>[] = [];
  for (const step of steps) {
    for (const result of step.toolResults ?? []) {
      if (result.toolName !== "get_company") continue;
      const payload = toolStepOutput(result);
      if (payload && typeof payload === "object" && !Array.isArray(payload)) {
        out.push(payload as Record<string, unknown>);
      }
    }
  }
  return out;
}

function retrievedCompanyOk(
  steps: { toolResults?: { toolName: string; output?: unknown; result?: unknown }[] }[],
): boolean {
  return companyPayloads(steps).some((row) => typeof row.error !== "string" && row.score != null);
}

function companyHasChangeReasons(
  steps: { toolResults?: { toolName: string; output?: unknown; result?: unknown }[] }[],
): boolean {
  return companyPayloads(steps).some(
    (row) => Array.isArray(row.change_reasons) && row.change_reasons.length > 0,
  );
}

function monthRecord(
  detail: CompanyDetail,
  month?: string | null,
): MonthRecord | { error: string; scored_months?: string[] } {
  const months = detail.months;
  if (!months.length) return { error: `${entityLabel(detail.company_id)} no tiene mes puntuado` };
  if (month) {
    const key = monthKey(month) ?? month;
    const rec = months.find((row) => row.month === key);
    if (!rec) {
      return { error: `${entityLabel(detail.company_id)} no tiene mes puntuado ${speakMonth(month, true) ?? month}`, scored_months: months.map((m) => speakMonth(m.month)) };
    }
    return rec;
  }
  return months[months.length - 1];
}

async function loadScoreExtras(companyId: string, month: string): Promise<ScoreExtras | null> {
  try {
    const rows = await neonQuery<{
      slope3: number | null;
      slope6: number | null;
      change_guard: number | null;
      guard_adjustment: number | null;
      item_id: string | null;
      value: number | null;
      points: number | null;
      contribution: number | null;
      delta: number | null;
    }>(
      `SELECT s.slope3, s.slope6, s.change_guard, s.guard_adjustment,
              i.item_id, i.value, i.points, i.contribution, i.delta
       FROM analytics.company_scores s
       JOIN api.current_run r ON r.run_id = s.run_id
       LEFT JOIN analytics.score_items i ON i.score_id = s.score_id
       WHERE s.company_id = $1 AND s.month = $2`,
      [companyId, month],
    );
    if (!rows.length) return null;
    const first = rows[0];
    const items: Record<string, ItemRow> = {};
    for (const row of rows) {
      if (!row.item_id) continue;
      items[row.item_id] = {
        value: round(row.value, 2),
        points: Number(row.points ?? 0),
        contribution: Number(row.contribution ?? 0),
        delta: round(row.delta, 2),
      };
    }
    return {
      slope3: round(first.slope3, 2),
      slope6: round(first.slope6, 2),
      change_guard: round(first.change_guard, 2),
      guard_adjustment: round(first.guard_adjustment, 2),
      items,
    };
  } catch {
    return null;
  }
}

async function loadCompanyCluster(companyId: string) {
  try {
    const clusterRows = await neonQuery<{
      cluster_id: string;
      month: string;
      label: string;
      description: string | null;
      n_companies: number | null;
      n_companies_train_fit: number | null;
    }>(
      `SELECT cc.cluster_id, cc.month, cl.label, cl.description, cl.n_companies, cl.n_companies_train_fit
       FROM analytics.company_cluster cc
       JOIN api.current_run r ON r.run_id = cc.run_id
       JOIN analytics.clusters cl ON cl.run_id = cc.run_id AND cl.cluster_id = cc.cluster_id
       WHERE cc.company_id = $1`,
      [companyId],
    );
    const cluster = clusterRows[0];
    if (!cluster) return null;
    const vs = await neonQuery<{ metric: string; percentile: number | null; robust_z: number | null }>(
      `SELECT metric, percentile, robust_z
       FROM analytics.cluster_vs v
       JOIN api.current_run r ON r.run_id = v.run_id
       WHERE v.company_id = $1`,
      [companyId],
    );
    return {
      cluster_id: cluster.cluster_id,
      month: cluster.month,
      vs_cluster: vs,
      meta: {
        cluster_id: cluster.cluster_id,
        label: cluster.label,
        description: cluster.description,
        n_companies: cluster.n_companies,
        n_companies_train_fit: cluster.n_companies_train_fit,
      } satisfies ClusterMeta,
    };
  } catch {
    return null;
  }
}

async function loadClusterQualityNote(): Promise<string | null> {
  try {
    const rows = await neonQuery<{ note: string | null }>(
      `SELECT q.note
       FROM analytics.cluster_quality q
       JOIN api.current_run r ON r.run_id = q.run_id`,
    );
    return rows[0]?.note ?? null;
  } catch {
    return null;
  }
}

const list_companies = tool({
  description:
    "Lista empresas con puntuación (último mes): índice, trayectoria, confianza, tope, nº de alertas. group_id opcional. Solo si no hay empresa en la sesión.",
  inputSchema: z.object({
    group_id: z.string().optional().describe("Grupo 0126"),
    limit: z.number().int().min(1).max(200).optional().describe("Por defecto 30"),
  }),
  execute: async ({ group_id, limit = 30 }) => {
    try {
      const store = repo();
      const [manifest, companies] = await Promise.all([store.getManifest(), store.listCompanies()]);
      const gid = group_id ? asGroupId(group_id) : undefined;
      const rows = companies
        .filter((company) => !gid || company.group_id === gid)
        .sort((a, b) => a.score - b.score)
        .slice(0, limit)
        .map((company) => ({
          empresa: entityLabel(company.company_id),
          grupo: company.group_id ? entityLabel(company.group_id) : undefined,
          score: speakScore(company.score),
          trajectory: speakTrajectory(company.trajectory),
          confidence: speakConfidence(company.confidence),
          guard: speakGuard(company.guard),
          delta_3m: speakDelta(company.delta_3m),
          n_alerts: company.n_alerts,
          max_alert_severity: speakSeverity(company.max_alert_severity),
        }));
      return { as_of: speakMonth(manifest.as_of_month, true), companies: rows };
    } catch (error) {
      return asError(error);
    }
  },
});

function createGetCompany(session?: ToolSession) {
  const historySpan = session?.historySpan ?? (session?.period ? 6 : 4);
  return tool({
    description:
      "Índice, tope, trayectoria, reasons y change_reasons de UNA empresa. Historial: periodo = tramo+1 mes (máx. 12); si no, 4. Omite month salvo un mes concreto. Sin clúster.",
    inputSchema: z.object({
      company_id: z.string().describe("Empresa 0030"),
      month: z.string().optional().describe("YYYY-MM; omite salvo un mes concreto"),
    }),
    execute: async ({ company_id, month }) => {
      try {
        const id = asCompanyId(company_id);
        const detail = await repo().getCompany(id);
        const rec = monthRecord(detail, month);
        if ("error" in rec) return rec;
        const currency = detail.currency ?? "EUR";
        const reasons = (rec.reasons ?? []).map((reason) => slimReason(reason, currency));
        const change = (rec.change_reasons ?? []).map((reason) => slimReason(reason, currency));
        const period = Boolean(session?.period);
        const payload: Json = {
          empresa: entityLabel(detail.company_id),
          grupo: period || !detail.group_id ? undefined : entityLabel(detail.group_id),
          month: speakMonth(rec.month, true),
          score: speakScore(rec.score),
          guard: speakGuard(rec.guard),
          trajectory: speakTrajectory(rec.trajectory),
          reasons,
          score_history: scoreHistory(detail.months, rec.month, historySpan, currency, period),
        };
        if (!period) {
          payload.confidence = speakConfidence(rec.confidence);
          payload.confidence_note = rec.confidence_note;
          payload.categories = slimCategories(rec.categories);
        }
        if (detail.currency && detail.currency !== "EUR") payload.currency = detail.currency;
        if (change.length) payload.change_reasons = change;
        if (rec.guard && rec.score_pre_cap != null) payload.sin_el_tope = speakScore(rec.score_pre_cap);
        if ((rec.trail_months ?? 24) < 12) payload.trail_months = rec.trail_months;
        if (!reasons.length) {
          const extras = await loadScoreExtras(id, rec.month);
          payload.items = extras?.items ? slimItems(extras.items) : itemsFromMonth(rec);
        }
        return payload;
      } catch (error) {
        return asError(error);
      }
    },
  });
}

const explain_change = tool({
  description:
    "Por qué se movió vs el mes anterior (deltas, change_reasons, tope). Solo si get_company no trajo change_reasons.",
  inputSchema: z.object({
    company_id: z.string().describe("Empresa 0030"),
    month: z.string().optional().describe("YYYY-MM; omite = último mes"),
  }),
    execute: async ({ company_id, month }) => {
      try {
        const id = asCompanyId(company_id);
        const detail = await repo().getCompany(id);
      const months = detail.months;
      if (!months.length) return { error: "sin mes puntuado" };
      let idx = months.length - 1;
      if (month) {
        const key = monthKey(month) ?? month;
        idx = months.findIndex((row) => row.month === key);
        if (idx < 0) return { error: `sin mes puntuado ${speakMonth(month, true) ?? month}` };
      }
      if (idx === 0) return { error: "primer mes puntuado, no hay anterior" };
      const cur = months[idx];
      const prev = months[idx - 1];
      const extras = await loadScoreExtras(id, cur.month);
      const itemSource = extras?.items ?? itemsFromMonth(cur);
      const deltas = Object.fromEntries(
        Object.entries(itemSource)
          .filter(([, item]) => item.delta != null)
          .map(([key, item]) => [REASON_LABELS[key] ?? key, item.delta])
          .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1]))),
      );
      return {
        empresa: entityLabel(id),
        desde: speakMonth(prev.month, true),
        hasta: speakMonth(cur.month, true),
        indice_desde: speakScore(prev.score),
        indice_hasta: speakScore(cur.score),
        cambio: speakDelta(cur.score - prev.score),
        trayectoria_desde: speakTrajectory(prev.trajectory),
        trayectoria_hasta: speakTrajectory(cur.trajectory),
        tope_desde: speakGuard(prev.guard),
        tope_hasta: speakGuard(cur.guard),
        cambio_tope: extras?.change_guard ?? null,
        deltas,
        change_reasons: (cur.change_reasons ?? []).map((reason) => slimReason(reason, detail.currency)),
      };
    } catch (error) {
      return asError(error);
    }
  },
});

async function findGroup(groupId: string): Promise<GroupRow | undefined> {
  const groups = await repo().listGroups();
  return groups.find((group) => group.group_id === groupId);
}

const get_group = tool({
  description:
    "Grupo: miembros (Empresa, índice, tope), media, historial. Alertas: get_alerts.",
  inputSchema: z.object({
    group_id: z.string().describe("Grupo 0126"),
  }),
  execute: async ({ group_id }) => {
    try {
      const id = asGroupId(group_id);
      const store = repo();
      const [group, companies, manifest] = await Promise.all([
        findGroup(id),
        store.listCompanies(),
        store.getManifest(),
      ]);
      if (!group) return { error: `${entityLabel(id)} no está en esta corrida` };
      const byId = new Map(companies.map((company) => [company.company_id, company]));
      const members = group.company_ids
        .map((id) => byId.get(id))
        .filter((row): row is NonNullable<typeof row> => Boolean(row))
        .sort((a, b) => a.score - b.score)
        .map((row) => ({
          empresa: entityLabel(row.company_id),
          score: speakScore(row.score),
          trajectory: speakTrajectory(row.trajectory),
          confidence: speakConfidence(row.confidence),
          guard: speakGuard(row.guard),
          delta_3m: speakDelta(row.delta_3m),
          n_alerts: row.n_alerts,
        }));
      const hist = manifest.months
        .map((month, index) => ({
          month: speakMonth(month),
          mean_score: speakScore(group.mean_scores[index] ?? null),
        }))
        .filter((row) => row.mean_score != null);
      return {
        grupo: entityLabel(id),
        n_empresas: group.n_companies,
        media: speakScore(group.latest_mean_score),
        minimo: speakScore(group.latest_min_score),
        empresa_min: group.latest_min_company_id ? entityLabel(group.latest_min_company_id) : null,
        limites: group.limits_available,
        miembros: members,
        historial_media: hist.slice(-12),
        graficos: (group.control ?? []).map((chart) => CHART_LABELS[chart.comparison] ?? chart.comparison),
      };
    } catch (error) {
      return asError(error);
    }
  },
});

export type ToolSession = {
  companyId?: string;
  groupId?: string;
  named?: string[];
  stats?: boolean;
  period?: boolean;
  historySpan?: number;
};

function alertScope(session?: ToolSession): Set<string> {
  const seen = new Set<string>();
  if (session?.companyId) seen.add(session.companyId);
  if (session?.groupId) seen.add(session.groupId);
  for (const id of session?.named ?? []) seen.add(id);
  return seen;
}

function createGetAlerts(session?: ToolSession) {
  const scope = alertScope(session);
  return tool({
    description:
      "Alertas, más nuevas primero. Una llamada. Sin empresa en SESSION ni nombrada: error sin empresa. Cita title y action tal cual.",
    inputSchema: z.object({
      entity_id: z.string().optional(),
      kinds: z.array(z.enum(ALERT_KINDS)).optional(),
      severities: z.array(z.enum(ALERT_SEVERITIES)).optional(),
      since_month: z.string().optional().describe("YYYY-MM"),
      limit: z.number().int().min(1).max(200).optional().describe("Por defecto 30"),
    }),
    execute: async ({ entity_id, kinds, severities, since_month, limit = 30 }) => {
      try {
        const feed = await repo().getAlerts();
        let alerts = feed.alerts;
        const wanted = new Set(scope);
        if (entity_id) wanted.add(asEntityId(entity_id));
        if (!wanted.size) {
          return {
            error: "sin empresa",
            ask: "¿Sobre qué Empresa o Grupo? Elige una en Rápido o nómbrala (Empresa 0030).",
          };
        }
        const scoped = [...wanted];
        alerts = alerts.filter((alert) => wanted.has(alert.entity.id));
        if (kinds?.length) alerts = alerts.filter((alert) => kinds.includes(alert.kind));
        if (severities?.length) alerts = alerts.filter((alert) => severities.includes(alert.severity));
        if (since_month) {
          const since = monthKey(since_month) ?? since_month;
          alerts = alerts.filter((alert) => alert.month >= since);
        }
        const cap = entity_id || scope.size ? limit : Math.min(limit, 8);
        const out = alerts.slice(0, cap).map(slimAlert);
        const payload: Json = { n_matching: alerts.length, alerts: out, scoped_to: scoped.map(entityLabel) };
        if (session?.stats) payload.stats = feed.stats;
        return payload;
      } catch (error) {
        return asError(error);
      }
    },
  });
}

const get_control_chart = tool({
  description:
    "Gráfico de control: ¿bache o deterioro? Persistencia: 3 de los últimos 4. Empresa: own_history|cluster. Grupo: group_own_history|group_vs_groups.",
  inputSchema: z.object({
      entity_id: z.string().describe("Empresa 0030 o Grupo 0126"),
    comparison: z.enum(COMPARISONS).optional(),
    metric: z.enum(METRICS).optional(),
  }),
  execute: async ({ entity_id, comparison = "own_history", metric = "score" }) => {
    try {
      const id = asEntityId(entity_id);
      const entityType = id.startsWith("GROUP") ? "group" : "company";
      const charts = await neonQuery<{
        comparison: string;
        metric: string;
        months: string[];
        values: (number | null)[] | null;
        center: (number | null)[] | null;
        lower: (number | null)[] | null;
        upper: (number | null)[] | null;
        ewma: (number | null)[] | null;
        cusum_low: (number | null)[] | null;
        cusum_high: (number | null)[] | null;
        signal: string[] | null;
        persistent: boolean[] | null;
        method: { name?: string } | string | null;
      }>(
        `SELECT comparison, metric, months, values, center, lower, upper, ewma, cusum_low, cusum_high, signal, persistent, method
         FROM analytics.control_charts c
         JOIN api.current_run r ON r.run_id = c.run_id
         WHERE c.entity_type = $1 AND c.entity_id = $2`,
        [entityType, id],
      );
      if (!charts.length) {
        return {
          error: "sin gráfico de control",
          hint: "Hacen falta 7 meses puntuados; un grupo, 3 miembros.",
        };
      }
      const chart = charts.find((row) => row.comparison === comparison && row.metric === metric);
      if (!chart) {
        return {
          error: "sin ese gráfico (hacen falta 7 meses; grupos, 3 miembros)",
          available: charts.map((row) => [row.comparison, row.metric]),
        };
      }
      const method =
        typeof chart.method === "object" && chart.method && "name" in chart.method
          ? chart.method.name
          : chart.method;
      const n = chart.months.length;
      const from = Math.max(0, n - 8);
      const slice = <T,>(values: T[] | null | undefined) => (Array.isArray(values) ? values.slice(from) : values);
      return {
        comparison: chart.comparison,
        metric: chart.metric,
        months: (slice(chart.months) ?? []).map((month) => speakMonth(month) ?? month),
        values: slice(chart.values),
        center: slice(chart.center),
        lower: slice(chart.lower),
        upper: slice(chart.upper),
        ewma: slice(chart.ewma),
        signal: slice(chart.signal),
        persistent: slice(chart.persistent),
        persistent_now: Array.isArray(chart.persistent) ? chart.persistent[n - 1] ?? null : null,
        persistent_rule: "3 de los últimos 4",
        method,
      };
    } catch (error) {
      return { error: "sin gráfico de control", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const compare_with_cluster = tool({
  description:
    "Grupo de pares (no segmento): etiqueta, tamaño, percentil y z robusta. Solo si preguntan cómo se sitúa frente a pares.",
  inputSchema: z.object({
    company_id: z.string().describe("Empresa 0030"),
  }),
  execute: async ({ company_id }) => {
    try {
      const id = asCompanyId(company_id);
      const [cluster, qualityNote] = await Promise.all([
        loadCompanyCluster(id),
        loadClusterQualityNote(),
      ]);
      if (!cluster) {
        return { error: "clúster no cargado (historial < 6 meses)" };
      }
      return {
        empresa: entityLabel(id),
        grupo_pares: cluster.meta.label,
        descripcion: cluster.meta.description,
        n_empresas: cluster.meta.n_companies,
        quality_note: qualityNote,
        month: speakMonth(cluster.month, true),
        vs_pares: cluster.vs_cluster.map((row) => ({
          metrica: row.metric === "score" ? "índice" : (CATEGORY_LABELS[row.metric as CategoryId] ?? row.metric),
          percentil: row.percentile,
          z: row.robust_z,
        })),
      };
    } catch (error) {
      return { error: "clúster no cargado", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const get_forecast = tool({
  description:
    "Abanico 1–6 meses (mediana, bandas 50 % y 80 %). method=naive_last: qué tan lejos suele moverse, no hacia dónde.",
  inputSchema: z.object({
    company_id: z.string().describe("Empresa 0030"),
  }),
  execute: async ({ company_id }) => {
    try {
      const id = asCompanyId(company_id);
      const forecasts = await neonQuery<{
        metric: string;
        method: string;
        origin_month: string;
        horizon_months: number;
        naive_last: number | null;
        skill_vs_naive: number | null;
        note: string | null;
        forecast_id: number;
      }>(
        `SELECT f.forecast_id, f.metric, f.method, f.origin_month, f.horizon_months, f.naive_last, f.skill_vs_naive, f.note
         FROM analytics.forecasts f
         JOIN api.current_run r ON r.run_id = f.run_id
         WHERE f.company_id = $1`,
        [id],
      );
      const forecast = forecasts[0];
      if (!forecast) {
        return { error: "previsión no cargada (historial < 4 meses)" };
      }
      const points = await neonQuery<{
        month: string;
        median: number;
        lo50: number;
        hi50: number;
        lo80: number;
        hi80: number;
      }>(
        `SELECT month, median, lo50, hi50, lo80, hi80
         FROM analytics.forecast_points
         WHERE forecast_id = $1
         ORDER BY month`,
        [forecast.forecast_id],
      );
      return {
        empresa: entityLabel(id),
        metrica: forecast.metric === "score" ? "índice" : forecast.metric,
        metodo: forecast.method,
        origen: speakMonth(forecast.origin_month, true),
        horizonte_meses: forecast.horizon_months,
        naive_last: speakScore(forecast.naive_last),
        nota: forecast.note,
        puntos: points.map((row) => ({
          month: speakMonth(row.month),
          mediana: speakScore(row.median),
          lo50: speakScore(row.lo50),
          hi50: speakScore(row.hi50),
          lo80: speakScore(row.lo80),
          hi80: speakScore(row.hi80),
        })),
      };
    } catch (error) {
      return { error: "previsión no cargada", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const query_clean_db = tool({
  description:
    "Un SELECT de solo lectura sobre registros limpios (clean.* → core.*). Siempre WHERE company_id y LIMIT ≤ 200. Nunca para reconstruir un índice.",
  inputSchema: z.object({
    sql: z.string().describe("Un SELECT o WITH … SELECT. Usa clean.* (se reescribe a core.*)."),
  }),
  execute: async ({ sql }) => {
    try {
      const guarded = checkSql(sql);
      if (!(await coreIsMounted())) {
        return { error: "registros no montados" };
      }
      const rewritten = toCoreSql(guarded);
      const data = await neonQuery<Record<string, unknown>>(rewritten);
      const columns = data[0] ? Object.keys(data[0]) : [];
      return { rows: data.length, columns, data };
    } catch (error) {
      if (error instanceof UnsafeQuery) return { error: `rejected: ${error.message}` };
      const first = error instanceof Error ? error.message.split("\n")[0] : String(error);
      if (/does not exist|not mounted|permission denied/i.test(first)) {
        return { error: "registros no montados" };
      }
      return { error: first };
    }
  },
});

const plot_series = tool({
  description:
    "Un gráfico del catálogo. El servidor pone los números. Sin series tecleadas.",
  inputSchema: z.object({
    kind: z.enum(PLOT_KINDS),
    company_id: z.string().optional().describe("Empresa 0030"),
    group_id: z.string().optional().describe("Grupo 0126"),
    company_ids: z.array(z.string()).max(8).optional(),
    metric: z.enum(["score", "payment_history", "amounts_owed", "stability"]).optional(),
  }),
  execute: async (input) => {
    const plot = await buildCatalogPlot({
      ...input,
      company_id: input.company_id ? asCompanyId(input.company_id) : input.company_id,
      group_id: input.group_id ? asGroupId(input.group_id) : input.group_id,
      company_ids: input.company_ids?.map(asCompanyId),
    });
    if ("error" in plot) return plot;
    return { ok: true, title: plot.title, points: plot.x.length, kind: input.kind, plot };
  },
});

/**
 * Distinct entities allowed per tool in one Pregunta turn.
 * One company = one get_company (history is in the payload), not one per month.
 * A period chart can name up to four companies.
 */
export const TOOL_CALL_CAP: Record<string, number> = {
  list_companies: 1,
  get_company: 4,
  explain_change: 2,
  get_group: 2,
  get_alerts: 1,
  get_control_chart: 2,
  compare_with_cluster: 2,
  get_forecast: 2,
  query_clean_db: 2,
  plot_series: 1,
};

const ENTITY_TOOLS = new Set([
  "get_company",
  "explain_change",
  "get_alerts",
  "get_control_chart",
  "compare_with_cluster",
  "get_forecast",
  "get_group",
]);

function toolEntityId(input: unknown): string {
  if (input == null || typeof input !== "object") return "";
  const row = input as Record<string, unknown>;
  for (const key of ["company_id", "entity_id", "group_id"]) {
    if (typeof row[key] === "string" && row[key].trim()) return String(row[key]).trim();
  }
  return "";
}

/** Same company + same tool, any month, is one retrieval. */
export function toolCallKey(name: string, input: unknown): string {
  const entity = toolEntityId(input);
  if (ENTITY_TOOLS.has(name)) return `${name}:${entity || "_"}`;
  return `${name}:${inputKey(input)}`;
}

function inputKey(input: unknown): string {
  if (input == null || typeof input !== "object") return JSON.stringify(input ?? null);
  const row = input as Record<string, unknown>;
  const ordered: Record<string, unknown> = {};
  for (const key of Object.keys(row).sort()) {
    if (row[key] !== undefined) ordered[key] = row[key];
  }
  return JSON.stringify(ordered);
}

export function countToolCallsByName(steps: { toolCalls?: { toolName: string }[] }[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const step of steps) {
    for (const call of step.toolCalls ?? []) {
      counts.set(call.toolName, (counts.get(call.toolName) ?? 0) + 1);
    }
  }
  return counts;
}

/** First model step: score path only. Records join if the user asked about invoices / debt. */
const OPENING_TOOLS = new Set(["get_company", "get_alerts", "get_group", "plot_series"]);

export function activeToolsUnderCap(
  names: string[],
  steps: {
    toolCalls?: { toolName: string }[];
    toolResults?: { toolName: string; output?: unknown; result?: unknown }[];
  }[],
  options?: { records?: boolean; plots?: boolean; alerts?: boolean; group?: boolean },
): string[] {
  const counts = countToolCallsByName(steps);
  const hasCompany = retrievedCompanyOk(steps);
  const hasChangeReasons = companyHasChangeReasons(steps);
  const opening = new Set(OPENING_TOOLS);
  if (options?.records) opening.add("query_clean_db");
  if (!options?.plots) opening.delete("plot_series");
  if (!options?.alerts) opening.delete("get_alerts");
  if (!options?.group) opening.delete("get_group");
  return names.filter((name) => {
    if (name === "plot_series" && !options?.plots) return false;
    if (name === "get_alerts" && !options?.alerts) return false;
    if (name === "get_group" && !options?.group) return false;
    if (steps.length === 0 && !opening.has(name)) return false;
    if ((counts.get(name) ?? 0) >= (TOOL_CALL_CAP[name] ?? 2)) return false;
    if (name === "explain_change" && hasChangeReasons) return false;
    if (name === "list_companies" && hasCompany) return false;
    return true;
  });
}

export function totalToolCalls(steps: { toolCalls?: { toolName: string }[] }[]): number {
  return steps.reduce((sum, step) => sum + (step.toolCalls?.length ?? 0), 0);
}

/** Completed steps / calls after which the next step must write, not call again. */
export const TOOL_STEP_BUDGET = 4;
export const TOOL_CALL_BUDGET = 8;

function retrievedAlertsOk(
  steps: { toolResults?: { toolName: string; output?: unknown; result?: unknown }[] }[],
): boolean {
  for (const step of steps) {
    for (const result of step.toolResults ?? []) {
      if (result.toolName !== "get_alerts") continue;
      const payload = toolStepOutput(result);
      if (payload && typeof payload === "object" && !Array.isArray(payload) && !("error" in payload)) return true;
    }
  }
  return false;
}

export function shouldForceTextStep(
  steps: {
    toolCalls?: { toolName: string }[];
    toolResults?: { toolName: string; output?: unknown; result?: unknown }[];
  }[],
  options?: { alertsOnly?: boolean; companyOnly?: boolean },
): boolean {
  if (steps.length >= TOOL_STEP_BUDGET || totalToolCalls(steps) >= TOOL_CALL_BUDGET) return true;
  const scored = companyPayloads(steps).filter((row) => typeof row.error !== "string" && row.score != null);
  if (scored.length >= 4) return true;
  if (options?.companyOnly && scored.length >= 1) return true;
  return Boolean(options?.alertsOnly && retrievedAlertsOk(steps));
}

function withMemoize<T extends Record<string, { execute?: (...args: never[]) => unknown }>>(tools: T): T {
  const cache = new Map<string, Promise<unknown>>();
  return Object.fromEntries(
    Object.entries(tools).map(([name, definition]) => {
      const execute = definition.execute;
      if (typeof execute !== "function") return [name, definition];
      return [
        name,
        {
          ...definition,
          execute: ((input: never, options: never) => {
            const key = toolCallKey(name, input);
            const hit = cache.get(key);
            if (hit) return hit;
            const used = [...cache.keys()].filter((entry) => entry.startsWith(`${name}:`)).length;
            if (used >= (TOOL_CALL_CAP[name] ?? 2)) {
              const denied = Promise.resolve({
                error: "cap",
                tool: name,
                message: "Ya tienes este dato. Responde con lo recuperado.",
              });
              cache.set(`${name}:cap:${used}`, denied);
              return denied;
            }
            const pending = Promise.resolve(execute(input, options));
            cache.set(key, pending);
            return pending;
          }) as (typeof definition)["execute"],
        },
      ];
    }),
  ) as T;
}

function withTiming<T extends { execute?: (...args: never[]) => unknown }>(name: string, definition: T): T {
  const execute = definition.execute;
  if (typeof execute !== "function") return definition;
  return {
    ...definition,
    execute: (async (input: never, options: never) => {
      const started = performance.now();
      const result = await execute(input, options);
      const timing_ms = Math.round(performance.now() - started);
      if (result && typeof result === "object" && !Array.isArray(result)) {
        return { ...(result as object), timing_ms };
      }
      return { result, timing_ms };
    }) as T["execute"],
  };
}

function timeAll<T extends Record<string, { execute?: (...args: never[]) => unknown }>>(tools: T): T {
  return Object.fromEntries(
    Object.entries(tools).map(([name, definition]) => [name, withTiming(name, definition)]),
  ) as T;
}

export function chatTools(session?: ToolSession) {
  return withMemoize(
    timeAll({
      list_companies,
      get_company: createGetCompany(session),
      explain_change,
      get_group,
      get_alerts: createGetAlerts(session),
      get_control_chart,
      compare_with_cluster,
      get_forecast,
      query_clean_db,
      plot_series,
    }),
  );
}

/** Explicación rápida: solo lo necesario para decir qué movió la puntuación. */
export function quickTools(session?: ToolSession) {
  return withMemoize(
    timeAll({
      get_company: createGetCompany(session),
      explain_change,
      get_alerts: createGetAlerts(session),
    }),
  );
}

export function sentinelTools(session?: ToolSession) {
  return withMemoize(
    timeAll({
      get_company: createGetCompany(session),
      get_group,
      get_alerts: createGetAlerts(session),
      get_control_chart,
      plot_series,
    }),
  );
}
