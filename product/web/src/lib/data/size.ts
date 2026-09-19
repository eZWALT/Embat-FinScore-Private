import { coreIsMounted, neonQuery } from "@/lib/agent/neon-sql";

const COMPANY_ID = /^COMP_[0-9]{4}$/;
const MONTH = /^[0-9]{4}-(0[1-9]|1[0-2])$/;

/** Months averaged into the monthly figures, ending at the company's latest scored month. Same window as the score's 3-month items. */
const WINDOW_MONTHS = 3;

/** Transaction categories of operating inflows and outflows. Mirrors CAT_MAP in analysis/features/common.py, which feeds a_op_in / a_op_out. */
const OP_IN = ["collection", "bulk_collection", "cash_settlement", "cash_settlements", "pos_settlement", "collection_refund"];
const OP_OUT = [
  "payment",
  "bulk_payment",
  "utility",
  "payment_refund",
  "salary",
  "social_security",
  "tax",
  "tax_refund",
  "cash_withdrawal",
  "pos_withdrawal",
];

/** How big a company is, from its own bank records. Amounts are in the company's currency, monthly averages over the window. */
export interface CompanySize {
  /** Mean monthly operating inflow. The size measure the score's own size baseline uses. */
  monthlyInflow: number;
  monthlyOutflow: number;
  /** Cash on the latest balance snapshot, counted accounts only. Null without a snapshot. */
  cash: number | null;
  /** First and last month of the window, YYYY-MM. */
  from: string;
  to: string;
}

function monthStart(month: string, offset: number): Date {
  const [year, mon] = month.split("-").map(Number);
  return new Date(Date.UTC(year, mon - 1 + offset, 1));
}

const isoMonth = (date: Date) => date.toISOString().slice(0, 7);

/**
 * Size of one company from `core.transactions` and `core.balances`, for the window of months ending at `month`.
 * Read-only aggregates on existing tables; nothing is stored. Needs the core schema to be loaded.
 */
export async function getCompanySize(companyId: string, month: string): Promise<CompanySize> {
  if (!COMPANY_ID.test(companyId)) throw new Error(`Invalid company id: ${companyId}`);
  if (!MONTH.test(month)) throw new Error(`Invalid month: ${month}`);
  if (!(await coreIsMounted())) throw new Error("Unavailable: the core schema is not loaded");

  const start = monthStart(month, -(WINDOW_MONTHS - 1));
  const end = monthStart(month, 1);

  const [flows, balances] = await Promise.all([
    neonQuery<{ inflow: number | null; outflow: number | null }>(
      `SELECT
         SUM(CASE WHEN category = ANY($4::text[]) THEN amount ELSE 0 END) AS inflow,
         -SUM(CASE WHEN category = ANY($5::text[]) THEN amount ELSE 0 END) AS outflow
       FROM core.transactions
       WHERE company_id = $1 AND booked_at >= $2::date AND booked_at < $3::date`,
      [companyId, start.toISOString().slice(0, 10), end.toISOString().slice(0, 10), OP_IN, OP_OUT],
    ),
    neonQuery<{ cash: number | null }>(
      `SELECT SUM(countable) AS cash
       FROM core.balances
       WHERE company_id = $1
         AND NOT COALESCE(balance_sentinel, false)
         AND as_of_date = (SELECT MAX(as_of_date) FROM core.balances WHERE company_id = $1)`,
      [companyId],
    ),
  ]);

  const row = flows[0];
  const cash = balances[0]?.cash;
  return {
    monthlyInflow: Number(row?.inflow ?? 0) / WINDOW_MONTHS,
    monthlyOutflow: Number(row?.outflow ?? 0) / WINDOW_MONTHS,
    cash: cash === null || cash === undefined ? null : Number(cash),
    from: isoMonth(start),
    to: month,
  };
}
