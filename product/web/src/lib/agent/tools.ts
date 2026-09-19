import { tool } from "ai";
import { z } from "zod";

import { createScoreRepository } from "@/lib/data/repository";
import type {
  AlertKind,
  AlertSeverity,
  CategoryId,
  CompanyDetail,
  Comparison,
  GroupRow,
  MonthRecord,
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
    "List scored companies (latest month) with score, trajectory, confidence, guard, alert count. Optional group_id filters to one group. Use it to resolve which companies exist.",
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
          trajectory: company.trajectory,
          confidence: company.confidence,
          guard: company.guard,
          delta_3m: company.delta_3m,
          n_alerts: company.n_alerts,
          max_alert_severity: company.max_alert_severity,
        }));
      return { as_of: manifest.as_of_month, companies: rows };
    } catch (error) {
      return asError(error);
    }
  },
});

const get_company = tool({
  description:
    "Score, trajectory, guard, confidence, categories, items, reasons (with EUR) and change reasons of one company for one month (default: latest). Also returns the score history and cluster membership when loaded.",
  inputSchema: z.object({
    company_id: z.string().describe("COMP_xxxx"),
    month: z.string().optional().describe("YYYY-MM, default latest"),
  }),
  execute: async ({ company_id, month }) => {
    try {
      const detail = await repo().getCompany(company_id);
      const rec = monthRecord(detail, month);
      if ("error" in rec) return rec;
      const extras = await loadScoreExtras(company_id, rec.month);
      const cluster = await loadCompanyCluster(company_id);
      return {
        company_id: detail.company_id,
        group_id: detail.group_id,
        country: detail.country,
        currency: detail.currency,
        erp: detail.erp,
        first_month: detail.first_month,
        month: rec.month,
        score: rec.score,
        score_pre_cap: rec.score_pre_cap,
        guard: rec.guard,
        trajectory: rec.trajectory,
        slope3: extras?.slope3 ?? null,
        slope6: extras?.slope6 ?? null,
        confidence: rec.confidence,
        confidence_note: rec.confidence_note,
        coverage: rec.coverage,
        trail_months: rec.trail_months,
        categories: rec.categories,
        items: extras?.items ?? {},
        reasons: rec.reasons,
        change_reasons: rec.change_reasons,
        score_history: detail.months.map((row) => ({
          month: row.month,
          score: row.score,
          trajectory: row.trajectory,
          guard: row.guard,
        })),
        cluster: cluster
          ? { cluster_id: cluster.cluster_id, month: cluster.month, vs_cluster: cluster.vs_cluster }
          : null,
        alert_ids: detail.alert_ids,
      };
    } catch (error) {
      return asError(error);
    }
  },
});

const explain_change = tool({
  description:
    "Why the score moved from the previous month: signed per-item deltas, change_reasons, guard effect. Default month: latest.",
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
      const deltas = Object.fromEntries(
        Object.entries(extras?.items ?? {})
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
        trajectory_from: prev.trajectory,
        trajectory_to: cur.trajectory,
        guard_from: prev.guard,
        guard_to: cur.guard,
        change_guard: extras?.change_guard ?? null,
        item_deltas: deltas,
        change_reasons: cur.change_reasons,
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
    "Group summary: members with their latest score/trajectory/guard, group mean and min, mean-score history, whether funnel limits exist (3+ scored members), and the group's alert ids.",
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
          trajectory: row.trajectory,
          confidence: row.confidence,
          guard: row.guard,
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
        latest_min_company_id: group.latest_min_company_id,
        limits_available: group.limits_available,
        members,
        mean_score_history: hist,
        alert_ids: group.alert_ids,
        control_charts: group.control?.map((chart) => chart.comparison) ?? [],
      };
    } catch (error) {
      return asError(error);
    }
  },
});

const get_alerts = tool({
  description:
    "Alerts from the feed, newest first. Filter by entity (company or group id), kinds (score_deterioration, score_improvement, category_drop, going_dark, top_customer_quiet), severities (info, watch, act), and since_month (YYYY-MM). Each alert has title, summary, reasons with EUR, owner, action, evidence, persistence.",
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
      if (entity_id) alerts = alerts.filter((alert) => alert.entity.id === entity_id);
      if (kinds?.length) alerts = alerts.filter((alert) => kinds.includes(alert.kind));
      if (severities?.length) alerts = alerts.filter((alert) => severities.includes(alert.severity));
      if (since_month) alerts = alerts.filter((alert) => alert.month >= since_month);
      const out = alerts.slice(0, limit).map(({ rank_score: _rank, ...alert }) => alert);
      return { stats: feed.stats, n_matching: alerts.length, alerts: out };
    } catch (error) {
      return asError(error);
    }
  },
});

const get_control_chart = tool({
  description:
    "Control chart series for a company (comparison own_history|cluster; metric score|payment_history|amounts_owed|stability) or a group (comparison group_own_history|group_vs_groups; metric score). Returns months, values, center, lower, upper, ewma, signal per month and the persistent flag.",
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
      return {
        comparison: chart.comparison,
        metric: chart.metric,
        months: chart.months,
        values: chart.values,
        center: chart.center,
        lower: chart.lower,
        upper: chart.upper,
        ewma: chart.ewma,
        cusum_low: chart.cusum_low,
        cusum_high: chart.cusum_high,
        signal: chart.signal,
        persistent: chart.persistent,
        method,
      };
    } catch (error) {
      return { error: "control charts not loaded", detail: error instanceof Error ? error.message : String(error) };
    }
  },
});

const compare_with_cluster = tool({
  description:
    "Peer-group comparison: the company's behaviour cluster (label, description, size, quality caveat) and its percentile / robust z inside that cluster for score and categories at the latest month.",
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
    "Score fan 1-6 months ahead (median, 50% and 80% bands) with the naive-last baseline. Method is naive_last: it says how far the score usually moves, not which way.",
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
    "Run one read-only SELECT over the cleaned records (schema `clean`: companies, groups, transactions, invoices, balances, banking_products, debt_products, debt_schedule_config, dq_log). Always filter by company_id and use LIMIT (max 200 rows). Hosted as Neon `core` (same table names; banking/debt products live in core.products). Never recompute a score.",
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
    "Draw one catalog chart. The server fills every number from the score run. kind: score_history | score_compare | categories | control_own | control_cluster | control_group | forecast_fan | group_members. Do not pass series or typed values.",
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

/** Distinct calls allowed per tool in one Pregunta turn. Two companies = two reads, not four. */
export const TOOL_CALL_CAP: Record<string, number> = {
  list_companies: 1,
  get_company: 2,
  explain_change: 2,
  get_group: 2,
  get_alerts: 2,
  get_control_chart: 2,
  compare_with_cluster: 2,
  get_forecast: 2,
  query_clean_db: 2,
  plot_series: 1,
};

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

export function activeToolsUnderCap(names: string[], steps: { toolCalls?: { toolName: string }[] }[]): string[] {
  const counts = countToolCallsByName(steps);
  return names.filter((name) => (counts.get(name) ?? 0) < (TOOL_CALL_CAP[name] ?? 2));
}

export function totalToolCalls(steps: { toolCalls?: { toolName: string }[] }[]): number {
  return steps.reduce((sum, step) => sum + (step.toolCalls?.length ?? 0), 0);
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
            const key = `${name}:${inputKey(input)}`;
            const hit = cache.get(key);
            if (hit) return hit;
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

export function chatTools() {
  return withMemoize(
    timeAll({
      list_companies,
      get_company,
      explain_change,
      get_group,
      get_alerts,
      get_control_chart,
      compare_with_cluster,
      get_forecast,
      query_clean_db,
      plot_series,
    }),
  );
}

/** Explicación rápida: solo lo necesario para decir qué movió la puntuación. */
export function quickTools() {
  return {
    get_company,
    explain_change,
    get_alerts,
  };
}

export function sentinelTools() {
  return withMemoize(
    timeAll({
      get_company,
      get_group,
      get_alerts,
      get_control_chart,
      plot_series,
    }),
  );
}
