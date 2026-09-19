import { neonQuery } from "@/lib/agent/neon-sql";

import { type ChartRow, type ControlSeries, toSeries } from "./monitor-service";
import type { AlertSeverity, Confidence, Guard, Trajectory } from "./types";

const GROUP_ID = /^GROUP_[0-9]{4}$/;

export interface GroupMemberRow {
  companyId: string;
  score: number;
  delta1m: number | null;
  delta3m: number | null;
  trajectory: Trajectory;
  confidence: Confidence;
  guard: Guard | null;
  nAlerts: number;
  maxAlertSeverity: AlertSeverity | null;
  /** Aligned with `GroupOverview.months`. */
  scores: (number | null)[];
}

export interface GroupOverview {
  groupId: string;
  asOfMonth: string;
  months: string[];
  nCompanies: number;
  meanScore: number | null;
  minScore: number | null;
  minCompanyId: string | null;
  /** Aligned with `months`. */
  meanScores: (number | null)[];
  /** True from 3 scored members. Below that only the mean is drawn: no limits, no alerts. */
  limitsAvailable: boolean;
  /** `own`: the mean against its own history. `vsGroups`: its 3-month change against funnel limits of similar-size groups. */
  control: { own: ControlSeries | null; vsGroups: ControlSeries | null };
  /** Weakest first. */
  members: GroupMemberRow[];
}

const num = (value: unknown) => (value === null || value === undefined ? null : Number(value));
const numbers = (list: unknown) => ((list as unknown[] | null) ?? []).map(num);

/** Everything the group screen draws except its alerts: the mean series, control charts and the members with their monthly scores. */
export async function getGroupOverview(groupId: string): Promise<GroupOverview> {
  if (!GROUP_ID.test(groupId)) throw new Error(`Invalid group id: ${groupId}`);

  const [manifest, groups, members, charts] = await Promise.all([
    neonQuery<{ as_of_month: string; months: string[] | null }>("SELECT as_of_month, months FROM api.manifest"),
    neonQuery<Record<string, unknown>>(
      `SELECT g.n_companies, g.latest_mean_score, g.latest_min_score, g.latest_min_company_id, g.mean_scores, g.limits_available
       FROM analytics.groups_index g
       JOIN api.current_run r ON r.run_id = g.run_id
       WHERE g.group_id = $1`,
      [groupId],
    ),
    neonQuery<Record<string, unknown>>(
      `SELECT company_id, score, trajectory, confidence, guard, delta_1m, delta_3m, n_alerts, max_alert_severity, sparkline
       FROM api.current_index
       WHERE group_id = $1
       ORDER BY score, company_id`,
      [groupId],
    ),
    neonQuery<ChartRow & Record<string, unknown>>(
      `SELECT c.comparison, c.months, c.values, c.center, c.lower, c.upper, c.signal, c.persistent
       FROM analytics.control_charts c
       JOIN api.current_run r ON r.run_id = c.run_id
       WHERE c.entity_type = 'group' AND c.entity_id = $1 AND c.metric = 'score'
         AND c.comparison IN ('group_own_history', 'group_vs_groups')`,
      [groupId],
    ),
  ]);

  const group = groups[0];
  if (!group) throw new Error(`Group not found: ${groupId}`);

  return {
    groupId,
    asOfMonth: manifest[0]?.as_of_month ?? "",
    months: manifest[0]?.months ?? [],
    nCompanies: Number(group.n_companies),
    meanScore: num(group.latest_mean_score),
    minScore: num(group.latest_min_score),
    minCompanyId: (group.latest_min_company_id as string | null) ?? null,
    meanScores: numbers(group.mean_scores),
    limitsAvailable: Boolean(group.limits_available),
    control: {
      own: toSeries(charts.find((row) => row.comparison === "group_own_history")),
      vsGroups: toSeries(charts.find((row) => row.comparison === "group_vs_groups")),
    },
    members: members.map((row) => ({
      companyId: String(row.company_id),
      score: Number(row.score),
      delta1m: num(row.delta_1m),
      delta3m: num(row.delta_3m),
      trajectory: row.trajectory as Trajectory,
      confidence: row.confidence as Confidence,
      guard: (row.guard as Guard | null) ?? null,
      nAlerts: Number(row.n_alerts ?? 0),
      maxAlertSeverity: (row.max_alert_severity as AlertSeverity | null) ?? null,
      scores: numbers(row.sparkline),
    })),
  };
}
