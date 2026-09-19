import { neonQuery } from "@/lib/agent/neon-sql";

const COMPANY_ID = /^COMP_[0-9]{4}$/;

export type ControlSignal = "none" | "low" | "high";

/** One control chart, aligned month by month. `own`: the score against its own normal. `cluster`: the gap to the peer cluster's median. */
export interface ControlSeries {
  months: string[];
  values: (number | null)[];
  center: (number | null)[];
  lower: (number | null)[];
  upper: (number | null)[];
  signal: ControlSignal[];
  /** True where the crossing has held 3 of the last 4 months. */
  persistent: boolean[];
}

export interface ForecastFan {
  originMonth: string;
  points: { month: string; median: number; lo80: number; hi80: number }[];
}

export interface CompanyMonitor {
  own: ControlSeries | null;
  cluster: ControlSeries | null;
  forecast: ForecastFan | null;
}

interface ChartRow {
  comparison: string;
  months: string[];
  values: (number | null)[] | null;
  center: (number | null)[] | null;
  lower: (number | null)[] | null;
  upper: (number | null)[] | null;
  signal: ControlSignal[] | null;
  persistent: boolean[] | null;
}

const num = (value: number | null | undefined) => (value === null || value === undefined ? null : Number(value));

function toSeries(row: ChartRow | undefined): ControlSeries | null {
  if (!row?.months?.length) return null;
  const pad = <T,>(list: T[] | null, fill: T) => row.months.map((_, i) => list?.[i] ?? fill);
  return {
    months: row.months,
    values: row.months.map((_, i) => num(row.values?.[i])),
    center: row.months.map((_, i) => num(row.center?.[i])),
    lower: row.months.map((_, i) => num(row.lower?.[i])),
    upper: row.months.map((_, i) => num(row.upper?.[i])),
    signal: pad(row.signal, "none" as ControlSignal),
    persistent: pad(row.persistent, false),
  };
}

/** Score control charts (own history, cluster gap) and the forecast fan of one company, from the current run. */
export async function getCompanyMonitor(companyId: string): Promise<CompanyMonitor> {
  if (!COMPANY_ID.test(companyId)) throw new Error(`Invalid company id: ${companyId}`);

  const [charts, forecasts] = await Promise.all([
    neonQuery<ChartRow & Record<string, unknown>>(
      `SELECT c.comparison, c.months, c.values, c.center, c.lower, c.upper, c.signal, c.persistent
       FROM analytics.control_charts c
       JOIN api.current_run r ON r.run_id = c.run_id
       WHERE c.entity_type = 'company' AND c.entity_id = $1 AND c.metric = 'score'
         AND c.comparison IN ('own_history', 'cluster')`,
      [companyId],
    ),
    neonQuery<{ forecast_id: number; origin_month: string }>(
      `SELECT f.forecast_id, f.origin_month
       FROM analytics.forecasts f
       JOIN api.current_run r ON r.run_id = f.run_id
       WHERE f.company_id = $1`,
      [companyId],
    ),
  ]);

  let forecast: ForecastFan | null = null;
  const head = forecasts[0];
  if (head) {
    const points = await neonQuery<{ month: string; median: number; lo80: number; hi80: number }>(
      `SELECT month, median, lo80, hi80 FROM analytics.forecast_points WHERE forecast_id = $1 ORDER BY month`,
      [head.forecast_id],
    );
    if (points.length) {
      forecast = {
        originMonth: head.origin_month,
        points: points.map((p) => ({
          month: p.month,
          median: Number(p.median),
          lo80: Number(p.lo80),
          hi80: Number(p.hi80),
        })),
      };
    }
  }

  return {
    own: toSeries(charts.find((row) => row.comparison === "own_history")),
    cluster: toSeries(charts.find((row) => row.comparison === "cluster")),
    forecast,
  };
}
