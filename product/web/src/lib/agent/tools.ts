import { tool } from "ai";
import { z } from "zod";

import { entityLabel, relabelEntities } from "@/lib/display";
import { CATEGORY_LABELS } from "@/lib/data/plain-language";
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

function slimReason(reason: Reason) {
  return {
    points: reason.points,
    eur: reason.eur,
    sentence: reason.sentence,
  };
}

function speakGuard(guard: string | null | undefined): string | undefined {
  if (guard === "dark") return "sin movimientos (tope 30)";
  if (guard === "fading") return "entradas hundidas (tope 50)";
  return undefined;
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

/** Full reasons on the focus month, the last 3, or a move of ≥2 pts. One call covers a period. */
function scoreHistory(months: MonthRecord[], focusMonth: string) {
  const focusIdx = months.findIndex((row) => row.month === focusMonth);
  const from = Math.max(0, months.length - 18, focusIdx >= 0 ? focusIdx - 6 : 0);
  const window = months.slice(from);
  return window.map((row, index) => {
    const prev = window[index - 1] ?? months[from - 1];
    const moved = prev != null && Math.abs(row.score - prev.score) >= 2;
    const focus = row.month === focusMonth;
    const recent = from + index >= months.length - 3;
    const base: Json = {
      month: row.month,
      score: row.score,
    };
    const trajectory = speakTrajectory(row.trajectory);
    const guard = speakGuard(row.guard);
    if (trajectory) base.trajectory = trajectory;
    if (guard) base.guard = guard;
    if (!moved && !focus && !recent) return base;
    return {
      ...base,
      reasons: (row.reasons ?? []).map(slimReason),
      change_reasons: (row.change_reasons ?? []).map(slimReason),
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

function slimEvidence(evidence: Alert["evidence"]) {
  const out: Record<string, string | number | boolean | null> = {};
  for (const [key, value] of Object.entries(evidence ?? {})) {
    out[key] = typeof value === "string" ? relabelEntities(value) : value;
  }
  return out;
}

function slimCategories(categories: MonthRecord["categories"]) {
  return Object.entries(categories).map(([id, row]) => ({
    name: CATEGORY_LABELS[id as CategoryId] ?? id,
    score: row.score,
    pts: row.contribution,
  }));
}

function slimAlert(alert: Alert) {
  return {
    alert_id: alert.alert_id,
    entity: entityLabel(alert.entity.id),
    month: alert.month,
    title: alert.title,
    summary: alert.summary,
    reasons: (alert.reasons ?? []).map(slimReason),
    owner: speakOwner(alert.owner),
    severity: speakSeverity(alert.severity),
    action: alert.action,
    persistence: alert.persistence,
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
  if (!months.length) return { error: `${detail.company_id} has no scored month` };
  if (month) {
    const rec = months.find((row) => row.month === month);
    if (!rec) {
      return { error: `${detail.company_id} has no scored month ${month}`, scored_months: months.map((m) => m.month) };
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
    group_id: z.string().optional().describe("GROUP_xxxx"),
    limit: z.number().int().min(1).max(200).optional().describe("Default 30"),
  }),
  execute: async ({ group_id, limit = 30 }) => {
    try {
      const store = repo();
      const [manifest, companies] = await Promise.all([store.getManifest(), store.listCompanies()]);
      const rows = companies
        .filter((company) => !group_id || company.group_id === group_id)
        .sort((a, b) => a.score - b.score)
        .slice(0, limit)
        .map((company) => ({
          company_id: company.company_id,
          group_id: company.group_id,
          score: company.score,
          trajectory: speakTrajectory(company.trajectory),
          confidence: speakConfidence(company.confidence),
          guard: speakGuard(company.guard),
          delta_3m: company.delta_3m,
          n_alerts: company.n_alerts,
          max_alert_severity: speakSeverity(company.max_alert_severity),
        }));
      return { as_of: manifest.as_of_month, companies: rows };
    } catch (error) {
      return asError(error);
    }
  },
});

const get_company = tool({
  description:
    "Índice, trayectoria, tope, confianza, categorías, reasons y change_reasons de UNA empresa. score_history trae reasons en el mes pedido, los 3 últimos y los que se movieron ≥2 pts: una llamada cubre un periodo. No pases month salvo un mes concreto. No llames explain_change si ya hay change_reasons. Sin clúster (usa compare_with_cluster).",
  inputSchema: z.object({
    company_id: z.string().describe("COMP_xxxx"),
    month: z.string().optional().describe("YYYY-MM; omite salvo un mes concreto"),
  }),
  execute: async ({ company_id, month }) => {
    try {
      const detail = await repo().getCompany(company_id);
      const rec = monthRecord(detail, month);
      if ("error" in rec) return rec;
      const extras = await loadScoreExtras(company_id, rec.month);
      const reasons = (rec.reasons ?? []).map(slimReason);
      const change = (rec.change_reasons ?? []).map(slimReason);
      const payload: Json = {
        company_id: detail.company_id,
        group_id: detail.group_id,
        currency: detail.currency,
        month: rec.month,
        score: rec.score,
        score_pre_cap: rec.score_pre_cap,
        guard: speakGuard(rec.guard),
        trajectory: speakTrajectory(rec.trajectory),
        confidence: speakConfidence(rec.confidence),
        confidence_note: rec.confidence_note,
        categories: slimCategories(rec.categories),
        reasons,
        change_reasons: change,
        score_history: scoreHistory(detail.months, rec.month),
        alert_ids: detail.alert_ids,
      };
      if (extras?.slope3 != null) payload.slope3 = extras.slope3;
      if (extras?.slope6 != null) payload.slope6 = extras.slope6;
      if ((rec.trail_months ?? 24) < 12) payload.trail_months = rec.trail_months;
      if (!reasons.length) {
        payload.items = extras?.items ? slimItems(extras.items) : itemsFromMonth(rec);
      }
      return payload;
    } catch (error) {
      return asError(error);
    }
  },
});

const explain_change = tool({
  description:
    "Por qué se movió vs el mes anterior (deltas, change_reasons, tope). Solo si get_company no trajo change_reasons.",
  inputSchema: z.object({
    company_id: z.string().describe("COMP_xxxx"),
    month: z.string().optional().describe("YYYY-MM, default latest"),
  }),
  execute: async ({ company_id, month }) => {
    try {
      const detail = await repo().getCompany(company_id);
      const months = detail.months;
      if (!months.length) return { error: "no scored month" };
      let idx = months.length - 1;
      if (month) {
        idx = months.findIndex((row) => row.month === month);
        if (idx < 0) return { error: `no scored month ${month}` };
      }
      if (idx === 0) return { error: "first scored month, no previous month" };
      const cur = months[idx];
      const prev = months[idx - 1];
      const extras = await loadScoreExtras(company_id, cur.month);
      const itemSource = extras?.items ?? itemsFromMonth(cur);
      const deltas = Object.fromEntries(
        Object.entries(itemSource)
          .filter(([, item]) => item.delta != null)
          .map(([key, item]) => [key, item.delta])
          .sort((a, b) => Math.abs(Number(b[1])) - Math.abs(Number(a[1]))),
      );
      return {
        company_id,
        from: prev.month,
        to: cur.month,
        score_from: prev.score,
        score_to: cur.score,
        change: round(cur.score - prev.score, 1),
        trajectory_from: speakTrajectory(prev.trajectory),
        trajectory_to: speakTrajectory(cur.trajectory),
        guard_from: speakGuard(prev.guard),
        guard_to: speakGuard(cur.guard),
        change_guard: extras?.change_guard ?? null,
        item_deltas: deltas,
        change_reasons: (cur.change_reasons ?? []).map(slimReason),
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
    "Resumen del grupo: miembros (índice, trayectoria, tope), media/mínimo, historial de la media, limits_available (3+), alert_ids.",
  inputSchema: z.object({
    group_id: z.string().describe("GROUP_xxxx"),
  }),
  execute: async ({ group_id }) => {
    try {
      const store = repo();
      const [group, companies, manifest] = await Promise.all([
        findGroup(group_id),
        store.listCompanies(),
        store.getManifest(),
      ]);
      if (!group) return { error: `${group_id} not in current score run` };
      const byId = new Map(companies.map((company) => [company.company_id, company]));
      const members = group.company_ids
        .map((id) => byId.get(id))
        .filter((row): row is NonNullable<typeof row> => Boolean(row))
        .map((row) => ({
          company_id: row.company_id,
          score: row.score,
          trajectory: speakTrajectory(row.trajectory),
          confidence: speakConfidence(row.confidence),
          guard: speakGuard(row.guard),
          delta_3m: row.delta_3m,
          n_alerts: row.n_alerts,
        }))
        .sort((a, b) => a.score - b.score);
      const hist = manifest.months
        .map((month, index) => ({ month, mean_score: group.mean_scores[index] ?? null }))
        .filter((row) => row.mean_score != null);
      return {
        group_id,
        n_companies: group.n_companies,
        latest_mean_score: group.latest_mean_score,
        latest_min_score: group.latest_min_score,
        latest_min_company_id: group.latest_min_company_id ? entityLabel(group.latest_min_company_id) : null,
        limits_available: group.limits_available,
        members,
        mean_score_history: hist.slice(-12),
        alert_ids: group.alert_ids,
        control_charts: group.control?.map((chart) => chart.comparison) ?? [],
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
      "Alertas del feed, más nuevas primero. Una sola llamada. Omite entity_id si SESSION ya tiene empresa y grupo: el servidor acota. Sin id nunca devuelve el feed entero. Cita title y action tal cual.",
    inputSchema: z.object({
      entity_id: z.string().optional(),
      kinds: z.array(z.enum(ALERT_KINDS)).optional(),
      severities: z.array(z.enum(ALERT_SEVERITIES)).optional(),
      since_month: z.string().optional().describe("YYYY-MM"),
      limit: z.number().int().min(1).max(200).optional().describe("Default 30"),
    }),
    execute: async ({ entity_id, kinds, severities, since_month, limit = 30 }) => {
      try {
        const feed = await repo().getAlerts();
        let alerts = feed.alerts;
        const wanted = new Set(scope);
        if (entity_id) wanted.add(entity_id);
        const scoped = [...wanted];
        if (wanted.size) alerts = alerts.filter((alert) => wanted.has(alert.entity.id));
        if (kinds?.length) alerts = alerts.filter((alert) => kinds.includes(alert.kind));
        if (severities?.length) alerts = alerts.filter((alert) => severities.includes(alert.severity));
        if (since_month) alerts = alerts.filter((alert) => alert.month >= since_month);
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
    "Gráfico de control: ¿bache o deterioro? persistent (3 de los últimos 4) es la regla. Empresa: own_history|cluster. Grupo: group_own_history|group_vs_groups.",
  inputSchema: z.object({
    entity_id: z.string().describe("COMP_xxxx or GROUP_xxxx"),
    comparison: z.enum(COMPARISONS).optional(),
    metric: z.enum(METRICS).optional(),
  }),
  execute: async ({ entity_id, comparison = "own_history", metric = "score" }) => {
    try {
      const entityType = entity_id.startsWith("GROUP") ? "group" : "company";
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
        [entityType, entity_id],
      );
      if (!charts.length) {
        return {
          error: "control charts not loaded",
          hint: "ScoreRepository.listGroups sets control: null; analytics.control_charts has no row for this entity (needs 7 scored months; groups need 3 members).",
        };
      }
      const chart = charts.find((row) => row.comparison === comparison && row.metric === metric);
      if (!chart) {
        return {
          error: "no such chart (needs 7 scored months; groups need 3 members)",
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
        months: slice(chart.months),
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
      return { error: "control charts not loaded", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const compare_with_cluster = tool({
  description:
    "Grupo de pares (no segmento): etiqueta, tamaño, percentil y z robusta. Solo si preguntan cómo se sitúa frente a pares.",
  inputSchema: z.object({
    company_id: z.string().describe("COMP_xxxx"),
  }),
  execute: async ({ company_id }) => {
    try {
      const [cluster, qualityNote] = await Promise.all([
        loadCompanyCluster(company_id),
        loadClusterQualityNote(),
      ]);
      if (!cluster) {
        return { error: "cluster not loaded (trail under 6 months, or analytics.company_cluster empty)" };
      }
      return {
        company_id,
        cluster: cluster.meta,
        quality_note: qualityNote,
        month: cluster.month,
        vs_cluster: cluster.vs_cluster,
      };
    } catch (error) {
      return { error: "cluster not loaded", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const get_forecast = tool({
  description:
    "Abanico 1–6 meses (mediana, bandas 50 % y 80 %). method=naive_last: qué tan lejos suele moverse, no hacia dónde.",
  inputSchema: z.object({
    company_id: z.string().describe("COMP_xxxx"),
  }),
  execute: async ({ company_id }) => {
    try {
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
        [company_id],
      );
      const forecast = forecasts[0];
      if (!forecast) {
        return { error: "forecast not loaded (under 4 scored months, or analytics.forecasts empty)" };
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
        metric: forecast.metric,
        method: forecast.method,
        origin_month: forecast.origin_month,
        horizon_months: forecast.horizon_months,
        naive_last: forecast.naive_last,
        skill_vs_naive: forecast.skill_vs_naive,
        note: forecast.note,
        points,
      };
    } catch (error) {
      return { error: "forecast not loaded", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const query_clean_db = tool({
  description:
    "Un SELECT de solo lectura sobre registros limpios (clean.* → core.*). Siempre WHERE company_id y LIMIT ≤ 200. Nunca para reconstruir un índice.",
  inputSchema: z.object({
    sql: z.string().describe("One SELECT or WITH … SELECT. Use clean.* (rewritten to core.*)."),
  }),
  execute: async ({ sql }) => {
    try {
      const guarded = checkSql(sql);
      if (!(await coreIsMounted())) {
        return { error: "records not mounted" };
      }
      const rewritten = toCoreSql(guarded);
      const data = await neonQuery<Record<string, unknown>>(rewritten);
      const columns = data[0] ? Object.keys(data[0]) : [];
      return { rows: data.length, columns, data };
    } catch (error) {
      if (error instanceof UnsafeQuery) return { error: `rejected: ${error.message}` };
      const first = error instanceof Error ? error.message.split("\n")[0] : String(error);
      if (/does not exist|not mounted|permission denied/i.test(first)) {
        return { error: "records not mounted" };
      }
      return { error: first };
    }
  },
});

const plot_series = tool({
  description:
    "Un gráfico del catálogo. El servidor pone los números. kind: score_history | score_compare | categories | control_own | control_cluster | control_group | forecast_fan | group_members. Sin series tecleadas.",
  inputSchema: z.object({
    kind: z.enum(PLOT_KINDS),
    company_id: z.string().optional().describe("COMP_xxxx"),
    group_id: z.string().optional().describe("GROUP_xxxx"),
    company_ids: z.array(z.string()).max(8).optional(),
    metric: z.enum(["score", "payment_history", "amounts_owed", "stability"]).optional(),
  }),
  execute: async (input) => {
    const plot = await buildCatalogPlot(input);
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
  options?: { records?: boolean; plots?: boolean; alerts?: boolean },
): string[] {
  const counts = countToolCallsByName(steps);
  const hasCompany = retrievedCompanyOk(steps);
  const hasChangeReasons = companyHasChangeReasons(steps);
  const opening = new Set(OPENING_TOOLS);
  if (options?.records) opening.add("query_clean_db");
  if (!options?.plots) opening.delete("plot_series");
  if (!options?.alerts) opening.delete("get_alerts");
  return names.filter((name) => {
    if (name === "plot_series" && !options?.plots) return false;
    if (name === "get_alerts" && !options?.alerts) return false;
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
  options?: { alertsOnly?: boolean },
): boolean {
  if (steps.length >= TOOL_STEP_BUDGET || totalToolCalls(steps) >= TOOL_CALL_BUDGET) return true;
  const scored = companyPayloads(steps).filter((row) => typeof row.error !== "string" && row.score != null);
  if (scored.length >= 4) return true;
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
      get_company,
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
      get_company,
      explain_change,
      get_alerts: createGetAlerts(session),
    }),
  );
}

export function sentinelTools(session?: ToolSession) {
  return withMemoize(
    timeAll({
      get_company,
      get_group,
      get_alerts: createGetAlerts(session),
      get_control_chart,
      plot_series,
    }),
  );
}
