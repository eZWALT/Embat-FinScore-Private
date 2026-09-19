export type Trajectory =
  | "improving"
  | "stable"
  | "dip"
  | "deteriorating"
  | "insufficient history";

export type Confidence = "high" | "medium" | "low";

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
  guard: "dark" | "fading" | null;
  delta_1m: number | null;
  delta_3m: number | null;
  top_reason: string | null;
}

export interface CompanyIndex {
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
}

export interface Manifest {
  as_of_month: string;
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

export interface ScoreRepository {
  getManifest(): Promise<Manifest>;
  listCompanies(): Promise<CompanyIndexRow[]>;
  getCompany(companyId: string): Promise<CompanyDetail>;
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
