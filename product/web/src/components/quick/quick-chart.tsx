"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceArea,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

import { ClientOnly } from "@/components/client-only";
import { buildChartRows, SERIES_COLORS } from "@/components/quick/series";
import { type MonthRange, useRangeDrag } from "@/components/use-range-drag";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany } from "@/lib/data/types";

export type { MonthRange };

/** Score lines for the chosen companies. Press and drag across months to select a period. */
export function QuickChart({
  companies,
  range,
  onRangeChange,
}: {
  companies: DashboardCompany[];
  range: MonthRange | null;
  onRangeChange: (range: MonthRange | null) => void;
}) {
  const rows = useMemo(() => buildChartRows(companies), [companies]);
  const config = useMemo(() => {
    const next: ChartConfig = {};
    companies.forEach((company, index) => {
      next[company.companyId] = { label: company.companyId, color: SERIES_COLORS[index % SERIES_COLORS.length] };
    });
    return next;
  }, [companies]);

  const months = useMemo(() => rows.map((row) => String(row.month)), [rows]);
  const { shaded, handlers } = useRangeDrag(months, range, onRangeChange);

  return (
    <div className="h-[min(52vh,440px)] w-full cursor-crosshair touch-none select-none" aria-label="Gráfico de puntuación. Arrastra para seleccionar un periodo.">
      <ClientOnly fallback={<div className="h-full w-full" />}>
        <ChartContainer config={config} className="aspect-auto h-full w-full">
          <LineChart data={rows} margin={{ top: 12, right: 16, left: -12, bottom: 0 }} {...handlers}>
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="month"
              tickFormatter={(month: string) => formatMonth(month)}
              tickLine={false}
              axisLine={false}
              minTickGap={28}
            />
            <YAxis domain={[0, 100]} ticks={[0, 25, 50, 75, 100]} tickLine={false} axisLine={false} width={36} />
            <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" />
            <ChartTooltip
              cursor={{ stroke: "var(--foreground)", strokeOpacity: 0.4, strokeWidth: 1.5 }}
              content={
                <ChartTooltipContent
                  indicator="dot"
                  labelFormatter={(_, payload) => {
                    const month = payload?.[0]?.payload?.month;
                    return month ? formatMonth(String(month)) : "";
                  }}
                />
              }
            />
            {shaded ? (
              <ReferenceArea
                x1={shaded[0]}
                x2={shaded[1]}
                fill="var(--foreground)"
                fillOpacity={0.08}
                stroke="var(--foreground)"
                strokeOpacity={0.35}
                strokeDasharray="4 3"
              />
            ) : null}
            {companies.map((company, index) => (
              <Line
                key={company.companyId}
                dataKey={company.companyId}
                type="monotone"
                stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                strokeWidth={2.5}
                dot={false}
                connectNulls={false}
                activeDot={{ r: 4 }}
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ChartContainer>
      </ClientOnly>
    </div>
  );
}
