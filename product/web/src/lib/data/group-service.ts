import { LocalBundleRepository } from "./local-bundle-repository";
import type {
  Alert,
  AlertSeverity,
  Confidence,
  Guard,
  Reason,
  ScoreRepository,
  Trajectory,
} from "./types";

const GROUP_ID = /^GROUP_[0-9]{4}$/;
const COMPANY_ID = /^COMP_[0-9]{4}$/;

export interface GroupOption {
  groupId: string;
  nCompanies: number;
  meanScore: number | null;
}

export interface GroupAlertView {
  alertId: string;
  entityId: string;
  month: string;
  kind: Alert["kind"];
  direction: Alert["direction"];
  severity: AlertSeverity;
  title: string;
  summary: string;
  owner: Alert["owner"];
  action: string;
  reasons: GroupReasonView[];
}

export interface GroupReasonView {
  item: string;
  label: string;
  points: number;
  value: number | null;
  unit: string | null;
  eur: number | null;
  sentence: string;
}

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
  /** Aligned with `GroupMapData.months`. */
  scores: (number | null)[];
}

export interface GroupCompanyPanel {
  companyId: string;
  currency: string | null;
  latestMonth: string;
  score: number;
  scorePreCap: number;
  guard: Guard | null;
  trajectory: Trajectory;
  confidence: Confidence;
  confidenceNote: string | null;
  coverage: number;
  scoreHistory: { month: string; score: number }[];
  alertMonths: { month: string; severity: AlertSeverity }[];
  reasons: GroupReasonView[];
  alerts: GroupAlertView[];
}

export interface GroupMapData {
  asOfMonth: string;
  isSample: boolean;
  disclaimer: string;
  months: string[];
  groupOptions: GroupOption[];
  group: {
    groupId: string;
    nCompanies: number;
    meanScore: number | null;
    minScore: number | null;
    minCompanyId: string | null;
    meanScores: (number | null)[];
    limitsAvailable: boolean;
    alerts: GroupAlertView[];
  };
  members: GroupMemberRow[];
  company: GroupCompanyPanel | null;
}

function toReasonView(reason: Reason): GroupReasonView {
  return {
    item: reason.item,
    label: reason.label,
    points: reason.points,
    value: reason.value,
    unit: reason.unit,
    eur: reason.eur,
    sentence: reason.sentence,
  };
}

function toAlertView(alert: Alert): GroupAlertView {
  return {
    alertId: alert.alert_id,
    entityId: alert.entity.id,
    month: alert.month,
    kind: alert.kind,
    direction: alert.direction,
    severity: alert.severity,
    title: alert.title,
    summary: alert.summary,
    owner: alert.owner,
    action: alert.action,
    reasons: alert.reasons.map(toReasonView),
  };
}

function resolveAlerts(ids: string[], byId: Map<string, Alert>): GroupAlertView[] {
  const found: Alert[] = [];
  for (const id of ids) {
    const alert = byId.get(id);
    if (alert) found.push(alert);
  }
  // Newest first, then severity act > watch > info.
  const rank: Record<AlertSeverity, number> = { act: 0, watch: 1, info: 2 };
  found.sort(
    (a, b) => b.month.localeCompare(a.month) || rank[a.severity] - rank[b.severity],
  );
  return found.map(toAlertView);
}

/**
 * View model for the Group Health Map. Reads the three indexes and only the selected company's detail file.
 * Unknown or missing ids fall back to the largest group and its weakest member.
 */
export async function getGroupMapData(
  groupId?: string,
  companyId?: string,
  repository: ScoreRepository = new LocalBundleRepository(),
): Promise<GroupMapData> {
  const [manifest, companies, groups, feed] = await Promise.all([
    repository.getManifest(),
    repository.listCompanies(),
    repository.listGroups(),
    repository.getAlerts(),
  ]);

  if (groups.length === 0) {
    throw new Error("The score bundle contains no groups");
  }

  const sortedGroups = [...groups].sort(
    (a, b) => b.n_companies - a.n_companies || a.group_id.localeCompare(b.group_id),
  );
  const groupOptions: GroupOption[] = sortedGroups.map((group) => ({
    groupId: group.group_id,
    nCompanies: group.n_companies,
    meanScore: group.latest_mean_score,
  }));

  const requestedGroup =
    groupId && GROUP_ID.test(groupId)
      ? sortedGroups.find((group) => group.group_id === groupId)
      : undefined;
  const group = requestedGroup ?? sortedGroups[0];

  const companyById = new Map(companies.map((row) => [row.company_id, row]));
  const members: GroupMemberRow[] = group.company_ids
    .map((id) => companyById.get(id))
    .filter((row) => row !== undefined)
    .map((row) => ({
      companyId: row.company_id,
      score: row.score,
      delta1m: row.delta_1m,
      delta3m: row.delta_3m,
      trajectory: row.trajectory,
      confidence: row.confidence,
      guard: row.guard,
      nAlerts: row.n_alerts,
      maxAlertSeverity: row.max_alert_severity,
      scores: row.scores,
    }))
    .sort((a, b) => a.score - b.score || a.companyId.localeCompare(b.companyId));

  const memberIds = new Set(members.map((member) => member.companyId));
  const selectedCompanyId =
    companyId && COMPANY_ID.test(companyId) && memberIds.has(companyId)
      ? companyId
      : group.latest_min_company_id && memberIds.has(group.latest_min_company_id)
        ? group.latest_min_company_id
        : members[0]?.companyId ?? null;

  const alertById = new Map(feed.alerts.map((alert) => [alert.alert_id, alert]));

  let company: GroupCompanyPanel | null = null;
  if (selectedCompanyId) {
    const detail = await repository.getCompany(selectedCompanyId);
    const latest = detail.months.at(-1);
    if (!latest) {
      throw new Error(`Company ${selectedCompanyId} has no scored months`);
    }
    const alerts = resolveAlerts(detail.alert_ids, alertById);
    company = {
      companyId: detail.company_id,
      currency: detail.currency,
      latestMonth: detail.latest_month,
      score: latest.score,
      scorePreCap: latest.score_pre_cap,
      guard: latest.guard,
      trajectory: latest.trajectory,
      confidence: latest.confidence,
      confidenceNote: latest.confidence_note,
      coverage: latest.coverage,
      scoreHistory: detail.months.map(({ month, score }) => ({ month, score })),
      alertMonths: alerts.map(({ month, severity }) => ({ month, severity })),
      reasons: (latest.reasons ?? []).slice(0, 4).map(toReasonView),
      alerts,
    };
  }

  return {
    asOfMonth: manifest.as_of_month,
    isSample: manifest.is_sample,
    disclaimer: manifest.disclaimer,
    months: manifest.months,
    groupOptions,
    group: {
      groupId: group.group_id,
      nCompanies: group.n_companies,
      meanScore: group.latest_mean_score,
      minScore: group.latest_min_score,
      minCompanyId: group.latest_min_company_id,
      meanScores: group.mean_scores,
      limitsAvailable: group.limits_available,
      alerts: resolveAlerts(group.alert_ids, alertById),
    },
    members,
    company,
  };
}
