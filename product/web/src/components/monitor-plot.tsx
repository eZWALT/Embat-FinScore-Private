"use client";

import { useEffect, useMemo, useState } from "react";
import { CalendarDays } from "lucide-react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

import { cn } from "cn";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ClientOnly } from "@/components/client-only";
import { Segmented, type SegmentedOption } from "@/components/segmented";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany } from "@/lib/data/types";
import type { CompanyMonitor, ControlSeries } from "@/lib/data/monitor-service";

type PlotView = "score" | "own" | "peers" | "cluster";

const VIEWS: SegmentedOption<PlotView>[] = [
  { value: "score", label: "Índice" },
  { value: "own", label: "Intra-empresa" },
  { value: "peers", label: "Inter-empresa" },
  { value: "cluster", label: "Inter-cluster" },
];

const COPY: Record<PlotView, { title: string; hint: string; unit: string }> = {
  score: {
    title: "Evolución de la puntuación",
    hint: "La dirección y persistencia importan tanto como el nivel actual.",
    unit: "Índice",
  },
  own: {
    title: "Control intra-empresa",
    hint: "El índice frente a su propia normalidad reciente. Fuera de la banda es un cambio inusual para esta empresa.",
    unit: "Índice",
  },
  peers: {
    title: "Control inter-empresa",
    hint: "El índice frente al resto de empresas: mediana y banda P10–P90 de cada mes.",
    unit: "Índice",
  },
  cluster: {
    title: "Control inter-cluster",
    hint: "Diferencia con la mediana de su cluster de empresas con comportamiento parecido, en puntos. Fuera de la banda cambia su posición relativa.",
    unit: "Diferencia con el cluster",
  },
};

const MIN_PEERS = 5;

function quantile(sorted: number[], q: number) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

/** The company's score against every scored company, month by month: median and P10–P90. Computed here, not part of the monitor bundle. */
function peerSeries(company: DashboardCompany, everyone: DashboardCompany[]): ControlSeries | null {
  const byMonth = new Map<string, number[]>();
  for (const other of everyone) {
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

type Row = Record<string, string | number | boolean | [number, number] | null>;

function controlRows(series: ControlSeries): Row[] {
  return series.months.map((month, i) => {
    const lower = series.lower[i];
    const upper = series.upper[i];
    return {
      month,
      value: series.values[i],
      center: series.center[i],
      band: lower !== null && upper !== null ? [lower, upper] : null,
      signal: series.signal[i],
      persistent: series.persistent[i],
    };
  });
}

function scoreRows(company: DashboardCompany, monitor: CompanyMonitor | null, showForecast: boolean): Row[] {
  const rows: Row[] = company.scoreHistory.map((point) => ({ month: point.month, value: point.score }));
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

type DotProps = { cx?: number; cy?: number; index?: number; payload?: Row };

function signalDot({ cx, cy, index, payload }: DotProps) {
  const key = `dot-${index}`;
  if (cx === undefined || cy === undefined || !payload || payload.signal === "none" || payload.signal === undefined) {
    return <g key={key} />;
  }
  const persistent = payload.persistent === true;
  return (
    <circle
      key={key}
      cx={cx}
      cy={cy}
      r={persistent ? 5 : 4}
      fill={persistent ? "var(--destructive)" : "var(--background)"}
      stroke="var(--destructive)"
      strokeWidth={2}
    />
  );
}

/**
 * The main plot of the deep view. Score history, or a control chart of the score against the company's own history,
 * against every other company, or against its cluster. The forecast fan can be laid over the score.
 */
export function MonitorPlot({
  company,
  companies,
  asOfMonth,
}: {
  company: DashboardCompany;
  companies: DashboardCompany[];
  asOfMonth: string;
}) {
  const [view, setView] = useState<PlotView>("score");
  const [showForecast, setShowForecast] = useState(false);
  const [loaded, setLoaded] = useState<{ companyId: string; monitor: CompanyMonitor } | null>(null);
  const [failure, setFailure] = useState<{ companyId: string; message: string } | null>(null);

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

  const config = useMemo(
    () =>
      ({
        value: { label: COPY[view].unit, color: "var(--chart-1)" },
        center: { label: view === "peers" ? "Mediana" : "Normalidad", color: "var(--muted-foreground)" },
        median: { label: "Predicción (mediana)", color: "var(--chart-2)" },
      }) satisfies ChartConfig,
    [view],
  );

  const copy = COPY[view];

  return (
    <Card id="evolucion" className="scroll-mt-20">
      <CardHeader className="gap-3">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <CardTitle className="text-base">{copy.title}</CardTitle>
          <Badge variant="outline" className="gap-1.5 font-normal text-muted-foreground">
            <CalendarDays className="size-3" />
            hasta {formatMonth(asOfMonth)}
          </Badge>
        </div>
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <div className="max-w-full overflow-x-auto">
            <Segmented options={VIEWS} value={view} onChange={setView} ariaLabel="Vista del gráfico" className="min-w-max" />
          </div>
          {view === "score" ? (
            <button
              type="button"
              role="switch"
              aria-checked={showForecast}
              onClick={() => setShowForecast((on) => !on)}
              className="group inline-flex items-center gap-2 rounded-md text-xs text-muted-foreground outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
            >
              <span
                aria-hidden="true"
                className={cn(
                  "relative h-4 w-7 rounded-full border transition-colors",
                  showForecast ? "border-foreground bg-foreground" : "bg-muted",
                )}
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
          ) : null}
        </div>
        <p className="text-sm text-muted-foreground">
          {copy.hint}
          {view === "score" && showForecast && monitor?.forecast
            ? " Línea discontinua: mediana prevista; banda: intervalo del 80 %. Es una referencia, no una garantía."
            : ""}
        </p>
      </CardHeader>
      <CardContent>
        {failed || empty || noFan ? (
          <p className="mb-3 rounded-lg border border-dashed px-3 py-2 text-sm text-muted-foreground">
            {failed
              ? failed
              : noFan
                ? "Esta empresa aún no tiene predicción: hacen falta al menos 4 meses puntuados."
                : view === "peers"
                  ? "No hay suficientes empresas puntuadas en estos meses para comparar."
                  : "Este gráfico necesita más historial (al menos 7 meses puntuados) o no está disponible para esta empresa."}
          </p>
        ) : null}
        <div className={cn("h-[300px] w-full transition-opacity", loading && "opacity-50")} aria-busy={loading}>
          <ClientOnly fallback={<div className="h-full w-full" />}>
            {empty ? null : (
              <ChartContainer config={config} className="h-full w-full aspect-auto">
                <ComposedChart data={rows} margin={{ top: 12, right: 8, left: -12, bottom: 0 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="month" tickFormatter={(month: string) => formatMonth(month)} tickLine={false} axisLine={false} minTickGap={28} />
                  <YAxis
                    domain={isControl ? ["auto", "auto"] : [0, 100]}
                    ticks={isControl ? undefined : [0, 25, 50, 75, 100]}
                    tickFormatter={isControl ? (value: number) => String(Math.round(value)) : undefined}
                    tickLine={false}
                    axisLine={false}
                    width={40}
                  />
                  {!isControl ? <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" /> : null}
                  {view === "cluster" ? <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="4 4" /> : null}
                  <ChartTooltip
                    cursor={{ stroke: "var(--foreground)", strokeOpacity: 0.4, strokeWidth: 1.5 }}
                    content={
                      <ChartTooltipContent
                        indicator="line"
                        labelFormatter={(_, payload) => {
                          const month = payload?.[0]?.payload?.month;
                          return month ? formatMonth(String(month)) : "";
                        }}
                      />
                    }
                  />
                  <Area
                    dataKey="band"
                    type="monotone"
                    stroke="none"
                    fill={isControl ? "var(--chart-1)" : "var(--chart-2)"}
                    fillOpacity={0.14}
                    tooltipType="none"
                    isAnimationActive={false}
                    connectNulls={false}
                  />
                  {isControl ? (
                    <Line dataKey="center" type="monotone" stroke="var(--color-center)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} activeDot={false} isAnimationActive={false} />
                  ) : null}
                  {view === "score" && showForecast && monitor?.forecast ? (
                    <Line dataKey="median" type="monotone" stroke="var(--color-median)" strokeWidth={2.5} strokeDasharray="6 4" dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
                  ) : null}
                  <Line
                    dataKey="value"
                    type="monotone"
                    stroke="var(--color-value)"
                    strokeWidth={2.5}
                    dot={isControl ? signalDot : false}
                    activeDot={{ r: 5 }}
                    isAnimationActive={false}
                    connectNulls={false}
                  />
                </ComposedChart>
              </ChartContainer>
            )}
          </ClientOnly>
        </div>
        {isControl && series ? (
          <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
            <span className="inline-flex items-center gap-1.5">
              <span className="size-2.5 rounded-full border-2 border-destructive bg-background" aria-hidden="true" />
              Fuera de la banda
            </span>
            {view !== "peers" ? (
              <span className="inline-flex items-center gap-1.5">
                <span className="size-2.5 rounded-full border-2 border-destructive bg-destructive" aria-hidden="true" />
                Persistente (3 de los últimos 4 meses)
              </span>
            ) : null}
          </p>
        ) : null}
      </CardContent>
    </Card>
  );
}
