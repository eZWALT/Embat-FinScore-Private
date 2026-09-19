import { neonQuery } from "@/lib/agent/neon-sql";

import type { AlertKind, AlertSeverity, Owner } from "./types";

const COMPANY_ID = /^COMP_[0-9]{4}$/;
const GROUP_ID = /^GROUP_[0-9]{4}$/;
const MAX_ROWS = 300;

export interface AlertReasonRow {
  item: string;
  label: string;
  points: number;
  eur: number | null;
  sentence: string;
}

/** One alert as the table shows it: what happened, why, who owns it and what to do. */
export interface AlertRow {
  alertId: string;
  entityType: "company" | "group";
  entityId: string;
  month: string;
  kind: AlertKind;
  direction: "risk" | "opportunity";
  severity: AlertSeverity;
  title: string;
  summary: string;
  owner: Owner;
  action: string;
  /** Rule text ("3 de los últimos 4 meses") and how many months it has been flagged. */
  persistence: { rule: string; monthsFlagged: number };
  reasons: AlertReasonRow[];
}

export interface EntityAlerts {
  asOfMonth: string;
  fromMonth: string | null;
  alerts: AlertRow[];
}

type Scope = { company: string } | { group: string };

const SEVERITY_RANK: Record<AlertSeverity, number> = { act: 0, watch: 1, info: 2 };

/**
 * Alerts of the current run for one company, or for one group: the group's own alerts plus those of its members.
 * Newest month first, then act > watch > info.
 */
export async function getEntityAlerts(scope: Scope): Promise<EntityAlerts> {
  const [kind, id] = "company" in scope ? (["company", scope.company] as const) : (["group", scope.group] as const);
  if (!(kind === "company" ? COMPANY_ID : GROUP_ID).test(id)) throw new Error(`Invalid ${kind} id: ${id}`);

  const [run, rows] = await Promise.all([
    neonQuery<{ as_of_month: string; detail_from_month: string | null }>(
      "SELECT as_of_month, detail_from_month FROM api.current_run",
    ),
    neonQuery<Record<string, unknown>>(
      `SELECT
         a.alert_id, a.entity_type, a.entity_id, a.month, a.kind, a.direction, a.severity,
         a.title, a.summary, a.owner, a.action, a.persistence_rule, a.months_flagged,
         COALESCE(
           (SELECT jsonb_agg(jsonb_build_object(
                     'item', ar.item, 'label', ar.label, 'points', ar.points,
                     'eur', ar.eur, 'sentence', ar.sentence) ORDER BY ar.position)
            FROM analytics.alert_reasons ar
            WHERE ar.run_id = a.run_id AND ar.alert_id = a.alert_id),
           '[]'::jsonb) AS reasons
       FROM analytics.alerts a
       JOIN api.current_run r ON r.run_id = a.run_id
       WHERE ($1 = 'company' AND a.entity_type = 'company' AND a.entity_id = $2)
          OR ($1 = 'group' AND (
                (a.entity_type = 'group' AND a.entity_id = $2)
                OR (a.entity_type = 'company' AND a.entity_id IN (
                      SELECT m.company_id FROM analytics.group_members m
                      WHERE m.run_id = a.run_id AND m.group_id = $2))))
       ORDER BY a.month DESC, a.alert_id
       LIMIT ${MAX_ROWS}`,
      [kind, id],
    ),
  ]);

  const alerts = rows.map(
    (row): AlertRow => ({
      alertId: String(row.alert_id),
      entityType: row.entity_type as AlertRow["entityType"],
      entityId: String(row.entity_id),
      month: String(row.month),
      kind: row.kind as AlertKind,
      direction: row.direction as AlertRow["direction"],
      severity: row.severity as AlertSeverity,
      title: String(row.title),
      summary: String(row.summary),
      owner: row.owner as Owner,
      action: String(row.action),
      persistence: { rule: String(row.persistence_rule ?? ""), monthsFlagged: Number(row.months_flagged ?? 0) },
      reasons: ((row.reasons as AlertReasonRow[] | null) ?? []).map((reason) => ({
        item: reason.item,
        label: reason.label,
        points: Number(reason.points ?? 0),
        eur: reason.eur === null || reason.eur === undefined ? null : Number(reason.eur),
        sentence: reason.sentence,
      })),
    }),
  );
  alerts.sort((a, b) => b.month.localeCompare(a.month) || SEVERITY_RANK[a.severity] - SEVERITY_RANK[b.severity]);

  return { asOfMonth: run[0]?.as_of_month ?? "", fromMonth: run[0]?.detail_from_month ?? null, alerts };
}
