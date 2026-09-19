"use client";

import type { ReactNode } from "react";
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

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ClientOnly } from "@/components/client-only";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { formatDecimal, formatSigned } from "@/lib/display";
import { formatMonth } from "@/lib/format-month";
import type { MonthContribution } from "@/lib/data/contributions";
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

function tooltipMonth(_: unknown, payload?: readonly { payload?: PlotRow }[]) {
  const month = payload?.[0]?.payload?.month;
  return month ? formatMonth(String(month)) : "";
}

/** The plot's own tooltip (month and values) with what each category contributed to the index that month underneath. */
function ContributionTooltip({
  contributions,
  ...props
}: React.ComponentProps<typeof ChartTooltipContent> & { contributions: Record<string, MonthContribution> }) {
  if (!props.active || !props.payload?.length) return null;
  const month = props.payload[0]?.payload?.month;
  const detail = month ? contributions[String(month)] : undefined;

  return (
    <div className="grid min-w-56 gap-2 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <ChartTooltipContent
        {...props}
        indicator="line"
        labelFormatter={tooltipMonth}
        className="min-w-0 border-0 bg-transparent p-0 shadow-none"
      />
      {detail ? (
        <div className="grid gap-1 border-t border-border/50 pt-1.5">
          <p className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground">Aportación al índice (pts)</p>
          {detail.categories.map((category) => (
            <div key={category.id} className="grid grid-cols-[minmax(0,1fr)_3rem_2.5rem] items-center gap-2">
              <span className="truncate text-muted-foreground">{category.label}</span>
              <span className="h-1.5 overflow-hidden rounded-full bg-muted" aria-hidden="true">
                <span
                  className="block h-full rounded-full bg-[var(--chart-1)]"
                  style={{ width: `${Math.max(0, Math.min(100, category.points))}%` }}
                />
              </span>
              <span className="text-right font-mono tabular-nums">
                {category.score === null ? "—" : formatDecimal(category.points, 1)}
              </span>
            </div>
          ))}
          {detail.guardAdjustment !== 0 ? (
            <div className="grid grid-cols-[minmax(0,1fr)_2.5rem] items-center gap-2 text-destructive">
              <span className="truncate">Tope de seguridad</span>
              <span className="text-right font-mono tabular-nums">{formatSigned(detail.guardAdjustment, 1)}</span>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

/** Card frame shared by the company and group plots: title, view switch and an aside. */
export function PlotCard({
  title,
  views,
  aside,
  notice,
  children,
}: {
  title: string;
  views?: ReactNode;
  aside?: ReactNode;
  notice?: string | null;
  children: ReactNode;
}) {
  return (
    <Card id="evolucion" className="scroll-mt-20">
      <CardHeader className="gap-3">
        <CardTitle className="text-base">{title}</CardTitle>
        {views || aside ? (
          <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
            <div className="max-w-full overflow-x-auto">{views}</div>
            {aside}
          </div>
        ) : null}
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
 * The plot itself. Score lines and score control charts sit on a fixed 0–100 axis; only charts of differences
 * (`zeroLine`) fit their axis. A control chart draws the value against a band and a centre line, with the months
 * outside the band marked in red (filled when the signal has persisted).
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
  contributions = null,
}: {
  rows: PlotRow[];
  control: boolean;
  valueLabel: string;
  centerLabel?: string;
  /** The values are differences around zero, not scores: draw the 0 line and let the axis fit. Score plots are always 0–100. */
  zeroLine?: boolean;
  forecast?: boolean;
  loading?: boolean;
  /** Draw nothing (the card shows why). */
  hidden?: boolean;
  /** Per-month category contributions, by month. Adds them to the tooltip of a score plot. */
  contributions?: Record<string, MonthContribution> | null;
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
                domain={zeroLine ? ["auto", "auto"] : [0, 100]}
                ticks={zeroLine ? undefined : [0, 25, 50, 75, 100]}
                tickFormatter={zeroLine ? (value: number) => String(Math.round(value)) : undefined}
                tickLine={false}
                axisLine={false}
                width={40}
              />
              {!zeroLine ? <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" /> : null}
              {zeroLine ? <ReferenceLine y={0} stroke="var(--border)" strokeDasharray="4 4" /> : null}
              <ChartTooltip
                cursor={{ stroke: "var(--foreground)", strokeOpacity: 0.4, strokeWidth: 1.5 }}
                content={
                  contributions && !zeroLine ? (
                    <ContributionTooltip contributions={contributions} />
                  ) : (
                    <ChartTooltipContent indicator="line" labelFormatter={tooltipMonth} />
                  )
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
