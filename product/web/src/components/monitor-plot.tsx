"use client";

import { useEffect, useMemo, useState } from "react";

import { cn } from "cn";

import { PlotCard, PlotChart, SignalLegend, controlRows, type PlotRow } from "@/components/plot-parts";
import { Segmented, type SegmentedOption } from "@/components/segmented";
import type { MonthContribution } from "@/lib/data/contributions";
import type { DashboardCompany } from "@/lib/data/types";
import type { CompanyMonitor, ControlSeries } from "@/lib/data/monitor-service";

type PlotView = "score" | "own" | "peers" | "cluster";

const VIEWS: SegmentedOption<PlotView>[] = [
  { value: "score", label: "Índice" },
  { value: "own", label: "vs tendencia histórica" },
  { value: "peers", label: "vs empresas del grupo" },
  { value: "cluster", label: "vs empresas similares" },
];

const COPY: Record<PlotView, { title: string; unit: string }> = {
  score: { title: "Evolución de la puntuación", unit: "Índice" },
  own: { title: "Índice vs tendencia histórica", unit: "Índice" },
  peers: { title: "Índice vs empresas del grupo", unit: "Índice" },
  cluster: { title: "Diferencia vs empresas similares", unit: "Diferencia con empresas similares" },
};

/** A group of fewer scored companies than this has no meaningful spread to compare against. */
const MIN_PEERS = 3;

function quantile(sorted: number[], q: number) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** The company's score against the companies of its own group, month by month: median and P10–P90. Computed here, not part of the monitor bundle. */
function peerSeries(company: DashboardCompany, everyone: DashboardCompany[]): ControlSeries | null {
  if (!company.groupId) return null;
  const byMonth = new Map<string, number[]>();
  for (const other of everyone) {
    if (other.groupId !== company.groupId) continue;
    for (const point of other.scoreHistory) {
      const list = byMonth.get(point.month);
      if (list) list.push(point.score);
      else byMonth.set(point.month, [point.score]);
    }
  }
  const months = company.scoreHistory.map((point) => point.month).filter((month) => (byMonth.get(month)?.length ?? 0) >= MIN_PEERS);
  if (months.length === 0) return null;
  const score = new Map(company.scoreHistory.map((point) => [point.month, point.score]));
  const stats = months.map((month) => {
    const sorted = byMonth.get(month)!.slice().sort((a, b) => a - b);
    return { median: quantile(sorted, 0.5), lower: quantile(sorted, 0.1), upper: quantile(sorted, 0.9) };
  });
  const values = months.map((month) => score.get(month) ?? null);
  return {
    months,
    values,
    center: stats.map((s) => s.median),
    lower: stats.map((s) => s.lower),
    upper: stats.map((s) => s.upper),
    signal: values.map((value, i) => (value === null ? "none" : value < stats[i].lower ? "low" : value > stats[i].upper ? "high" : "none")),
    persistent: values.map(() => false),
  };
}

function scoreRows(company: DashboardCompany, monitor: CompanyMonitor | null, showForecast: boolean): PlotRow[] {
  const rows: PlotRow[] = company.scoreHistory.map((point) => ({ month: point.month, value: point.score }));
  const fan = showForecast ? monitor?.forecast : null;
  if (!fan) return rows;
  const origin = rows.find((row) => row.month === fan.originMonth);
  if (origin) {
    // Start the forecast line on the last real point so the two segments join.
    origin.median = origin.value;
    origin.band = [origin.value as number, origin.value as number];
  }
  for (const point of fan.points) {
    if (point.month <= fan.originMonth) continue;
    rows.push({ month: point.month, median: point.median, band: [point.lo80, point.hi80] });
  }
  return rows;
}

/**
 * The main plot of the company view. Score history, or a control chart of the score against the company's own history,
 * against the companies of its group, or against similar companies. The forecast fan can be laid over the score.
 */
export function MonitorPlot({
  company,
  companies,
}: {
  company: DashboardCompany;
  companies: DashboardCompany[];
}) {
  const [view, setView] = useState<PlotView>("score");
  const [showForecast, setShowForecast] = useState(true);
  const [loaded, setLoaded] = useState<{ companyId: string; monitor: CompanyMonitor } | null>(null);
  const [failure, setFailure] = useState<{ companyId: string; message: string } | null>(null);

  const [contributed, setContributed] = useState<{
    companyId: string;
    byMonth: Record<string, MonthContribution>;
  } | null>(null);

  const contributions = contributed?.companyId === company.companyId ? contributed.byMonth : null;
  const monitor = loaded?.companyId === company.companyId ? loaded.monitor : null;
  const failed = !monitor && failure?.companyId === company.companyId ? failure.message : null;
  const needsMonitor = view === "own" || view === "cluster" || (view === "score" && showForecast);

  useEffect(() => {
    if (!needsMonitor || monitor) return;
    let cancelled = false;
    fetch(`/api/monitor?company=${encodeURIComponent(company.companyId)}`)
      .then(async (res) => {
        const body = (await res.json()) as CompanyMonitor & { error?: string };
        if (!res.ok) throw new Error(body.error || res.statusText);
        if (!cancelled) setLoaded({ companyId: company.companyId, monitor: body });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setFailure({ companyId: company.companyId, message: error instanceof Error ? error.message : "No se pudo cargar el gráfico" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [needsMonitor, monitor, company.companyId]);

  // What each category contributed, month by month, for the tooltip. Without it the tooltip still shows the index.
  useEffect(() => {
    if (contributions) return;
    let cancelled = false;
    fetch(`/api/contributions?company=${encodeURIComponent(company.companyId)}`)
      .then(async (res) => {
        if (!res.ok) return;
        const byMonth = (await res.json()) as Record<string, MonthContribution>;
        if (!cancelled) setContributed({ companyId: company.companyId, byMonth });
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [contributions, company.companyId]);

  const peers = useMemo(() => (view === "peers" ? peerSeries(company, companies) : null), [view, company, companies]);
  const series = view === "own" ? monitor?.own : view === "cluster" ? monitor?.cluster : view === "peers" ? peers : null;
  const isControl = view !== "score";
  const loading = needsMonitor && !monitor && !failed;
  const noFan = view === "score" && showForecast && monitor !== null && monitor.forecast === null;
  const empty = isControl && !loading && !failed && !series;

  const rows = useMemo(
    () => (isControl ? (series ? controlRows(series) : []) : scoreRows(company, monitor, showForecast)),
    [isControl, series, company, monitor, showForecast],
  );

  const copy = COPY[view];
  const notice = failed
    ? failed
    : noFan
      ? "Esta empresa aún no tiene predicción: hacen falta al menos 4 meses puntuados."
      : empty
        ? view === "peers"
          ? "Hace falta un grupo con al menos 3 empresas puntuadas para comparar."
          : "Este gráfico necesita más historial (al menos 7 meses puntuados) o no está disponible para esta empresa."
        : null;

  return (
    <PlotCard
      title={copy.title}
      views={<Segmented options={VIEWS} value={view} onChange={setView} ariaLabel="Vista del gráfico" className="min-w-max" />}
      aside={
        view === "score" ? (
          <button
            type="button"
            role="switch"
            aria-checked={showForecast}
            onClick={() => setShowForecast((on) => !on)}
            className="group inline-flex items-center gap-2 rounded-md text-xs text-muted-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
          >
            <span
              aria-hidden="true"
              className={cn("relative h-4 w-7 rounded-full border transition-colors", showForecast ? "border-foreground bg-foreground" : "bg-muted")}
            >
              <span
                className={cn(
                  "absolute top-0.5 size-2.5 rounded-full transition-all",
                  showForecast ? "left-[calc(100%-0.75rem)] bg-background" : "left-0.5 bg-muted-foreground",
                )}
              />
            </span>
            <span className={cn("font-medium", showForecast && "text-foreground")}>Predicción</span>
          </button>
        ) : null
      }
      notice={notice}
    >
      <PlotChart
        rows={rows}
        control={isControl}
        valueLabel={copy.unit}
        centerLabel={view === "peers" ? "Mediana del grupo" : "Normalidad"}
        zeroLine={view === "cluster"}
        forecast={view === "score" && showForecast && !!monitor?.forecast}
        loading={loading}
        hidden={empty}
        contributions={contributions}
      />
      {isControl && series ? <SignalLegend persistent={view !== "peers"} /> : null}
    </PlotCard>
  );
}
