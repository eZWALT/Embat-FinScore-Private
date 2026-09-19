import { neon } from "@neondatabase/serverless";

import {
  CATEGORY_LABELS,
  MONITORING_DISCLAIMER,
  topReasonInSpanish,
} from "./plain-language";
import type {
  Alert,
  AlertFeed,
  AlertKind,
  AlertSeverity,
  CategoryId,
  CompanyDetail,
  CompanyIndexRow,
  DashboardCompany,
  DashboardData,
  EntityType,
  GroupRow,
  Manifest,
  MonthRecord,
  Owner,
  Reason,
  ScoreRepository,
  Trajectory,
  Confidence,
} from "./types";

const COMPANY_ID = /^COMP_[0-9]{4}$/;

const CATEGORY_ORDER: CategoryId[] = [
  "payment_history",
  "amounts_owed",
  "stability",
  "new_credit",
  "mix",
];

type Sql = ReturnType<typeof neon>;
type Row = Record<string, unknown>;

function rows<T>(result: unknown): T[] {
  return result as T[];
}

function connect(): Sql {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL is not set");
  }
  return neon(url);
}

export class NeonScoreRepository implements ScoreRepository {
  constructor(private readonly sql: Sql = connect()) {}

  async getManifest(): Promise<Manifest> {
    const row = rows<Manifest>(await this.sql`
      SELECT as_of_month, months, is_sample, counts, disclaimer, spec
      FROM api.manifest
    `)[0];
    if (!row) throw new Error("No score run loaded");
    return {
      as_of_month: row.as_of_month,
      months: row.months ?? [],
      is_sample: row.is_sample,
      counts: row.counts,
      disclaimer: row.disclaimer,
      spec: row.spec,
    };
  }

  async listCompanies(): Promise<CompanyIndexRow[]> {
    return rows<Row>(await this.sql`
      SELECT
        company_id, group_id, latest_month, score, trajectory, confidence,
        guard, delta_1m, delta_3m, top_reason, cluster_id, n_alerts,
        max_alert_severity, sparkline
      FROM api.current_index
      ORDER BY company_id
    `).map(mapIndexRow);
  }

  async getCompany(companyId: string): Promise<CompanyDetail> {
    if (!COMPANY_ID.test(companyId)) {
      throw new Error(`Invalid company id: ${companyId}`);
    }

    const profile = rows<CompanyDetail>(await this.sql`
      SELECT p.company_id, p.group_id, p.country, p.currency, p.erp,
             p.first_month, p.latest_month, p.alert_ids
      FROM analytics.company_profiles p
      JOIN api.current_run r ON r.run_id = p.run_id
      WHERE p.company_id = ${companyId}
    `)[0];
    if (!profile) {
      throw new Error(`Company not found: ${companyId}`);
    }

    const months = rows<{
      score_id: number;
      month: string;
      score: number;
      score_pre_cap: number;
      guard: CompanyDetail["months"][number]["guard"];
      trajectory: string;
      confidence: string;
      confidence_note: string | null;
      coverage: number;
      trail_months: number;
    }>(await this.sql`
      SELECT
        s.score_id, s.month, s.score, s.score_pre_cap, s.guard, s.trajectory,
        s.confidence, s.confidence_note, s.coverage, s.trail_months
      FROM analytics.company_scores s
      JOIN api.current_run r ON r.run_id = s.run_id
      WHERE s.company_id = ${companyId}
      ORDER BY s.month
    `);

    const scoreIds = months.map((m) => Number(m.score_id));
    const categories = scoreIds.length
      ? await this.sql.query(
          `SELECT score_id, category_id, score, contribution
           FROM analytics.score_categories
           WHERE score_id = ANY($1::bigint[])`,
          [scoreIds],
        )
      : [];
    const reasons = scoreIds.length
      ? await this.sql.query(
          `SELECT score_id, kind, position, item, label, points, value, unit, eur, sentence
           FROM analytics.score_reasons
           WHERE score_id = ANY($1::bigint[])
           ORDER BY score_id, kind, position`,
          [scoreIds],
        )
      : [];

    const catsByScore = new Map<number, MonthRecord["categories"]>();
    for (const row of categories as Array<{
      score_id: number;
      category_id: CategoryId;
      score: number | null;
      contribution: number;
    }>) {
      const current = catsByScore.get(Number(row.score_id)) ?? emptyCategories();
      current[row.category_id] = { score: row.score, contribution: row.contribution };
      catsByScore.set(Number(row.score_id), current);
    }

    const reasonsByScore = new Map<number, { level: Reason[]; change: Reason[] }>();
    for (const row of reasons as Array<{
      score_id: number;
      kind: "level" | "change";
      item: string;
      label: string;
      points: number;
      value: number;
      unit: string;
      eur: number | null;
      sentence: string;
    }>) {
      const bucket = reasonsByScore.get(Number(row.score_id)) ?? { level: [], change: [] };
      const reason: Reason = {
        item: row.item,
        label: row.label,
        points: row.points,
        value: row.value,
        unit: row.unit,
        eur: row.eur,
        sentence: row.sentence,
      };
      if (row.kind === "change") bucket.change.push(reason);
      else bucket.level.push(reason);
      reasonsByScore.set(Number(row.score_id), bucket);
    }

    return {
      company_id: profile.company_id,
      group_id: profile.group_id,
      country: profile.country,
      currency: profile.currency,
      erp: profile.erp,
      first_month: profile.first_month,
      latest_month: profile.latest_month,
      months: months.map((m) => {
        const id = Number(m.score_id);
        const bucket = reasonsByScore.get(id);
        return {
          month: m.month,
          score: Number(m.score),
          score_pre_cap: Number(m.score_pre_cap),
          guard: m.guard ?? null,
          trajectory: m.trajectory as Trajectory,
          confidence: m.confidence as Confidence,
          confidence_note: m.confidence_note,
          coverage: Number(m.coverage),
          trail_months: Number(m.trail_months),
          categories: catsByScore.get(id) ?? emptyCategories(),
          reasons: bucket?.level,
          change_reasons: bucket?.change,
        } satisfies MonthRecord;
      }),
      alert_ids: profile.alert_ids ?? [],
    };
  }

  async listGroups(): Promise<GroupRow[]> {
    return rows<Row>(await this.sql`
      SELECT
        g.group_id,
        g.n_companies,
        g.latest_mean_score,
        g.latest_min_score,
        g.latest_min_company_id,
        g.mean_scores,
        g.limits_available,
        g.alert_ids,
        COALESCE(
          array_agg(m.company_id ORDER BY m.company_id) FILTER (WHERE m.company_id IS NOT NULL),
          '{}'::text[]
        ) AS company_ids
      FROM analytics.groups_index g
      JOIN api.current_run r ON r.run_id = g.run_id
      LEFT JOIN analytics.group_members m
        ON m.run_id = g.run_id AND m.group_id = g.group_id
      GROUP BY
        g.group_id, g.n_companies, g.latest_mean_score, g.latest_min_score,
        g.latest_min_company_id, g.mean_scores, g.limits_available, g.alert_ids
      ORDER BY g.n_companies DESC, g.group_id
    `).map((row) => ({
      group_id: String(row.group_id),
      company_ids: (row.company_ids as string[]) ?? [],
      n_companies: Number(row.n_companies),
      latest_mean_score: row.latest_mean_score === null ? null : Number(row.latest_mean_score),
      latest_min_score: row.latest_min_score === null ? null : Number(row.latest_min_score),
      latest_min_company_id: (row.latest_min_company_id as string | null) ?? null,
      mean_scores: ((row.mean_scores as (number | null)[]) ?? []).map((value) =>
        value === null ? null : Number(value),
      ),
      limits_available: Boolean(row.limits_available),
      control: null,
      alert_ids: (row.alert_ids as string[]) ?? [],
    }));
  }

  async getAlerts(): Promise<AlertFeed> {
    const run = rows<{
      as_of_month: string;
      detail_from_month: string | null;
    }>(await this.sql`
      SELECT as_of_month, detail_from_month FROM api.current_run
    `)[0];
    if (!run) throw new Error("No score run loaded");

    const alerts = rows<Row>(await this.sql`
      SELECT
        a.alert_id, a.entity_type, a.entity_id, a.entity_name, a.month, a.kind,
        a.direction, a.severity, a.title, a.summary, a.owner, a.action,
        a.persistence_rule, a.months_flagged, a.rank_score, a.evidence,
        COALESCE(
          (
            SELECT jsonb_agg(
              jsonb_build_object(
                'item', ar.item,
                'label', ar.label,
                'points', ar.points,
                'value', ar.value,
                'unit', ar.unit,
                'eur', ar.eur,
                'sentence', ar.sentence
              ) ORDER BY ar.position
            )
            FROM analytics.alert_reasons ar
            WHERE ar.run_id = a.run_id AND ar.alert_id = a.alert_id
          ),
          '[]'::jsonb
        ) AS reasons
      FROM analytics.alerts a
      JOIN api.current_run r ON r.run_id = a.run_id
      ORDER BY a.month DESC, a.alert_id
    `).map(mapAlert);

    return {
      as_of_month: run.as_of_month,
      from_month: run.detail_from_month ?? alerts.at(-1)?.month ?? run.as_of_month,
      stats: {
        false_alarm_rate: null,
        false_alarm_rate_at_chance: null,
        median_lead_time_months: null,
        top_customer_precision: null,
        top_customer_base_rate: null,
        alerts_per_company_year: null,
        note: "Alert stats live in the export bundle; Neon serves the alert rows.",
      },
      alerts,
    };
  }

  async getDashboardSnapshot(): Promise<DashboardData> {
    const [manifest, companies] = await Promise.all([
      this.getManifest(),
      this.sql`SELECT * FROM api.dashboard_companies ORDER BY score DESC`,
    ]);

    return {
      asOfMonth: manifest.as_of_month,
      isSample: manifest.is_sample,
      disclaimer: MONITORING_DISCLAIMER,
      companies: rows<DashboardRow>(companies).map(mapDashboardRow),
    };
  }
}

interface DashboardRow {
  company_id: string;
  group_id: string | null;
  country: string | null;
  currency: string | null;
  erp: string | null;
  latest_month: string;
  score: number;
  delta_1m: number | null;
  delta_3m: number | null;
  trajectory: Trajectory;
  confidence: Confidence;
  confidence_note: string | null;
  coverage: number;
  top_reason: string | null;
  guard: "dark" | "fading" | null;
  reason_item: string | null;
  reason_points: number | null;
  reason_eur: number | null;
  score_history: { month: string; score: number }[];
  categories: Record<CategoryId, { score: number | null; contribution: number }>;
}

function mapDashboardRow(row: DashboardRow): DashboardCompany {
  return {
    companyId: row.company_id,
    groupId: row.group_id,
    country: row.country,
    currency: row.currency,
    erp: row.erp,
    latestMonth: row.latest_month,
    score: Number(row.score),
    delta1m: row.delta_1m === null ? null : Number(row.delta_1m),
    delta3m: row.delta_3m === null ? null : Number(row.delta_3m),
    trajectory: row.trajectory,
    confidence: row.confidence,
    confidenceNote: row.confidence_note,
    coverage: Number(row.coverage),
    topReason: topReasonInSpanish(
      row.guard,
      row.reason_item
        ? { item: row.reason_item, points: Number(row.reason_points), eur: row.reason_eur }
        : undefined,
    ),
    scoreHistory: row.score_history,
    categories: CATEGORY_ORDER.map((id) => ({
      id,
      label: CATEGORY_LABELS[id],
      score: row.categories?.[id]?.score ?? null,
    })),
  };
}

function mapIndexRow(row: Row): CompanyIndexRow {
  return {
    company_id: String(row.company_id),
    group_id: (row.group_id as string | null) ?? null,
    latest_month: String(row.latest_month),
    score: Number(row.score),
    trajectory: row.trajectory as Trajectory,
    confidence: row.confidence as Confidence,
    guard: (row.guard as CompanyIndexRow["guard"]) ?? null,
    delta_1m: row.delta_1m === null || row.delta_1m === undefined ? null : Number(row.delta_1m),
    delta_3m: row.delta_3m === null || row.delta_3m === undefined ? null : Number(row.delta_3m),
    top_reason: (row.top_reason as string | null) ?? null,
    cluster_id: (row.cluster_id as string | null) ?? null,
    n_alerts: Number(row.n_alerts ?? 0),
    max_alert_severity: (row.max_alert_severity as CompanyIndexRow["max_alert_severity"]) ?? null,
    scores: ((row.sparkline as (number | null)[]) ?? []).map((value) =>
      value === null || value === undefined ? null : Number(value),
    ),
  };
}

function mapAlert(row: Row): Alert {
  const reasons = (row.reasons as Reason[] | null) ?? [];
  return {
    alert_id: String(row.alert_id),
    entity: {
      type: row.entity_type as EntityType,
      id: String(row.entity_id),
      name: (row.entity_name as string | null) ?? null,
    },
    month: String(row.month),
    kind: row.kind as AlertKind,
    direction: row.direction as Alert["direction"],
    severity: row.severity as AlertSeverity,
    title: String(row.title),
    summary: String(row.summary),
    reasons,
    persistence: {
      rule: String(row.persistence_rule ?? ""),
      months_flagged: Number(row.months_flagged ?? 0),
    },
    owner: row.owner as Owner,
    action: String(row.action),
    evidence: (row.evidence as Alert["evidence"]) ?? {},
    rank_score: row.rank_score === null || row.rank_score === undefined ? null : Number(row.rank_score),
  };
}

function emptyCategories(): MonthRecord["categories"] {
  return {
    payment_history: { score: null, contribution: 0 },
    amounts_owed: { score: null, contribution: 0 },
    stability: { score: null, contribution: 0 },
    new_credit: { score: null, contribution: 0 },
    mix: { score: null, contribution: 0 },
  };
}

