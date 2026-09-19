"use client";

import type { ReactNode } from "react";
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
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { formatMonth } from "@/lib/format-month";
import type { ControlSeries } from "@/lib/data/monitor-service";

/** One month of a plot. `band` is a [low, high] range drawn as a shaded area. */
export type PlotRow = Record<string, string | number | boolean | [number, number] | null>;

export function controlRows(series: ControlSeries): PlotRow[] {
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

type DotProps = { cx?: number; cy?: number; index?: number; payload?: PlotRow };

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

/** Card frame shared by the company and group plots: title, as-of badge, view switch, an aside and the explanation. */
export function PlotCard({
  title,
  asOfMonth,
  views,
  aside,
  hint,
  notice,
  children,
}: {
  title: string;
  asOfMonth: string;
  views?: ReactNode;
  aside?: ReactNode;
  hint: ReactNode;
  notice?: string | null;
  children: ReactNode;
}) {
  return (
    <Card id="evolucion" className="scroll-mt-20">
      <CardHeader className="gap-3">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <CardTitle className="text-base">{title}</CardTitle>
          <Badge variant="outline" className="gap-1.5 font-normal text-muted-foreground">
            <CalendarDays className="size-3" />
            hasta {formatMonth(asOfMonth)}
          </Badge>
        </div>
        {views || aside ? (
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <div className="max-w-full overflow-x-auto">{views}</div>
            {aside}
          </div>
        ) : null}
        <p className="text-sm text-muted-foreground">{hint}</p>
      </CardHeader>
      <CardContent>
        {notice ? (
          <p className="mb-3 rounded-lg border border-dashed px-3 py-2 text-sm text-muted-foreground">{notice}</p>
        ) : null}
        {children}
      </CardContent>
    </Card>
  );
}

/**
 * The plot itself. Plain lines on a fixed 0–100 scale, or a control chart: the value against a band and a centre line,
 * with the months outside the band marked in red (filled when the signal has persisted).
 */
export function PlotChart({
  rows,
  control,
  valueLabel,
  centerLabel = "Normalidad",
  zeroLine = false,
  forecast = false,
  loading = false,
  hidden = false,
}: {
  rows: PlotRow[];
  control: boolean;
  valueLabel: string;
  centerLabel?: string;
  /** Draw the 0 line: the values are differences around zero. */
  zeroLine?: boolean;
  forecast?: boolean;
  loading?: boolean;
  /** Draw nothing (the card shows why). */
  hidden?: boolean;
}) {
  const config = {
    value: { label: valueLabel, color: "var(--chart-1)" },
    center: { label: centerLabel, color: "var(--muted-foreground)" },
    median: { label: "Predicción (mediana)", color: "var(--chart-2)" },
  } satisfies ChartConfig;

  return (
    <div className={cn("h-[300px] w-full transition-opacity", loading && "opacity-50")} aria-busy={loading}>
      <ClientOnly fallback={<div className="h-full w-full" />}>
        {hidden ? null : (
          <ChartContainer config={config} className="h-full w-full aspect-auto">
            <ComposedChart data={rows} margin={{ top: 12, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="month" tickFormatter={(month: string) => formatMonth(month)} tickLine={false} axisLine={false} minTickGap={28} />
              <YAxis
                domain={control ? ["auto", "auto"] : [0, 100]}
                ticks={control ? undefined : [0, 25, 50, 75, 100]}
                tickFormatter={control ? (value: number) => String(Math.round(value)) : undefined}
                tickLine={false}
                axisLine={false}
                width={40}
              />
              {!control ? <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" /> : null}
              {zeroLine ? <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="4 4" /> : null}
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
                fill={control ? "var(--chart-1)" : "var(--chart-2)"}
                fillOpacity={0.14}
                tooltipType="none"
                isAnimationActive={false}
                connectNulls={false}
              />
              {control ? (
                <Line dataKey="center" type="monotone" stroke="var(--color-center)" strokeWidth={1.5} strokeDasharray="5 4" dot={false} activeDot={false} isAnimationActive={false} />
              ) : null}
              {forecast ? (
                <Line dataKey="median" type="monotone" stroke="var(--color-median)" strokeWidth={2.5} strokeDasharray="6 4" dot={false} activeDot={{ r: 4 }} isAnimationActive={false} />
              ) : null}
              <Line
                dataKey="value"
                type="monotone"
                stroke="var(--color-value)"
                strokeWidth={2.5}
                dot={control ? signalDot : false}
                activeDot={{ r: 5 }}
                isAnimationActive={false}
                connectNulls={false}
              />
            </ComposedChart>
          </ChartContainer>
        )}
      </ClientOnly>
    </div>
  );
}

/** Legend under a control chart. */
export function SignalLegend({ persistent = true }: { persistent?: boolean }) {
  return (
    <p className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] text-muted-foreground">
      <span className="inline-flex items-center gap-1.5">
        <span className="size-2.5 rounded-full border-2 border-destructive bg-background" aria-hidden="true" />
        Fuera de la banda
      </span>
      {persistent ? (
        <span className="inline-flex items-center gap-1.5">
          <span className="size-2.5 rounded-full border-2 border-destructive bg-destructive" aria-hidden="true" />
          Persistente (3 de los últimos 4 meses)
        </span>
      ) : null}
    </p>
  );
}
