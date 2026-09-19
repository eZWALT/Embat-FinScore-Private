export type Trajectory =
  | "improving"
  | "stable"
  | "dip"
  | "deteriorating"
  | "insufficient history";

export type Confidence = "high" | "medium" | "low";

/** "dark": no bank booking for 60 days, score capped at 30. "fading": inflows collapsed, capped at 50. */
export type Guard = "dark" | "fading";

export type CategoryId =
  | "payment_history"
  | "amounts_owed"
  | "stability"
  | "new_credit"
  | "mix";

export interface CompanyIndexRow {
  company_id: string;
  group_id: string | null;
  latest_month: string;
  score: number;
  trajectory: Trajectory;
  confidence: Confidence;
  guard: Guard | null;
  delta_1m: number | null;
  delta_3m: number | null;
  top_reason: string | null;
  cluster_id: string | null;
  n_alerts: number;
  max_alert_severity: AlertSeverity | null;
  /** Aligned with `Manifest.months`; null where the company had no score yet. */
  scores: (number | null)[];
}

export interface CompanyIndex {
  months: string[];
  companies: CompanyIndexRow[];
}

export interface Reason {
  item: string;
  label: string;
  points: number;
  value: number;
  unit: string;
  eur: number | null;
  sentence: string;
}

export interface MonthRecord {
  month: string;
  score: number;
  score_pre_cap: number;
  guard: Guard | null;
  trajectory: Trajectory;
  confidence: Confidence;
  confidence_note: string | null;
  coverage: number;
  trail_months: number;
  categories: Record<CategoryId, { score: number | null; contribution: number }>;
  reasons?: Reason[];
  change_reasons?: Reason[];
}

export interface CompanyDetail {
  company_id: string;
  group_id: string | null;
  country: string | null;
  currency: string | null;
  erp: string | null;
  first_month: string;
  latest_month: string;
  months: MonthRecord[];
  /** Keys into alerts.json, in the feed window. */
  alert_ids: string[];
}

export interface Manifest {
  as_of_month: string;
  /** Every month with a score for at least one company; index rows are aligned to it. */
  months: string[];
  is_sample: boolean;
  counts: { companies: number; groups: number; company_months: number };
  disclaimer: string;
  spec: {
    categories: Record<
      CategoryId,
      { label: string; nominal_weight: number; effective_weight: number }
    >;
  };
}

/* ---------------------------------------------------------------- groups and alerts (schema 1.1.0) */

export type Comparison = "own_history" | "cluster" | "group_own_history" | "group_vs_groups";

export interface ControlChart {
  comparison: Comparison;
  metric: "score" | CategoryId;
  months: string[];
  values: (number | null)[];
  center: (number | null)[];
  lower: (number | null)[];
  upper: (number | null)[];
  ewma?: (number | null)[];
  cusum_low?: (number | null)[];
  cusum_high?: (number | null)[];
  signal: ("none" | "low" | "high")[];
  persistent: boolean[];
  method: { name: string; params: Record<string, number | string> };
}

/** One entry of groups.json `groups[]`. */
export interface GroupRow {
  group_id: string;
  company_ids: string[];
  n_companies: number;
  latest_mean_score: number | null;
  latest_min_score: number | null;
  latest_min_company_id: string | null;
  /** Aligned with `Manifest.months`. */
  mean_scores: (number | null)[];
  /** True from 3 scored members. Below that, draw only the mean, no limits and no alerts. */
  limits_available: boolean;
  control: ControlChart[] | null;
  alert_ids: string[];
}

export interface GroupIndex {
  months: string[];
  groups: GroupRow[];
}

export type EntityType = "company" | "group" | "customer" | "supplier";
export type AlertSeverity = "info" | "watch" | "act";
export type AlertKind =
  | "score_deterioration"
  | "score_improvement"
  | "category_drop"
  | "going_dark"
  | "top_customer_quiet";
export type Owner = "treasurer" | "cfo" | "collections";

export interface Alert {
  alert_id: string;
  entity: { type: EntityType; id: string; name?: string | null };
  month: string;
  kind: AlertKind;
  direction: "risk" | "opportunity";
  severity: AlertSeverity;
  title: string;
  summary: string;
  reasons: Reason[];
  persistence: { rule: string; months_flagged: number };
  owner: Owner;
  action: string;
  evidence: Record<string, number | string | boolean | null>;
  rank_score: number | null;
}

export interface AlertFeed {
  as_of_month: string;
  from_month: string;
  stats: {
    false_alarm_rate: number | null;
    false_alarm_rate_at_chance: number | null;
    median_lead_time_months: number | null;
    top_customer_precision: number | null;
    top_customer_base_rate: number | null;
    alerts_per_company_year: number | null;
    note: string;
  };
  alerts: Alert[];
}

export interface ScoreRepository {
  getManifest(): Promise<Manifest>;
  listCompanies(): Promise<CompanyIndexRow[]>;
  getCompany(companyId: string): Promise<CompanyDetail>;
  listGroups(): Promise<GroupRow[]>;
  getAlerts(): Promise<AlertFeed>;
}

export interface DashboardCompany {
  companyId: string;
  groupId: string | null;
  country: string | null;
  currency: string | null;
  erp: string | null;
  latestMonth: string;
  score: number;
  delta1m: number | null;
  delta3m: number | null;
  trajectory: Trajectory;
  confidence: Confidence;
  confidenceNote: string | null;
  coverage: number;
  topReason: string | null;
  scoreHistory: { month: string; score: number }[];
  categories: { id: CategoryId; label: string; score: number | null }[];
}

export interface DashboardData {
  asOfMonth: string;
  isSample: boolean;
  disclaimer: string;
  companies: DashboardCompany[];
}
