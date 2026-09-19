"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
import { buildChartRows, SERIES_COLORS, yDomainForSelection } from "@/components/health-score-view";
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany } from "@/lib/data/types";

export interface MonthRange {
  from: string;
  to: string;
}

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
  const ids = useMemo(() => companies.map((company) => company.companyId), [companies]);
  // Round the fitted range to whole tens (or twenties) so the axis ticks are readable numbers.
  const { yDomain, yTicks } = useMemo(() => {
    const [min, max] = yDomainForSelection(rows, ids);
    const step = max - min > 50 ? 20 : 10;
    const lo = Math.max(0, Math.floor(min / step) * step);
    const hi = Math.min(100, Math.ceil(max / step) * step);
    const ticks: number[] = [];
    for (let value = lo; value <= hi; value += step) ticks.push(value);
    return { yDomain: [lo, hi] as [number, number], yTicks: ticks };
  }, [rows, ids]);
  const config = useMemo(() => {
    const next: ChartConfig = {};
    companies.forEach((company, index) => {
      next[company.companyId] = { label: company.companyId, color: SERIES_COLORS[index % SERIES_COLORS.length] };
    });
    return next;
  }, [companies]);

  type Drag = { anchor: string; head: string } | null;
  const [drag, setDragState] = useState<Drag>(null);
  const dragRef = useRef<Drag>(null);
  // The ref lets the mouse handlers see the latest drag before React has re-rendered.
  function setDrag(next: Drag) {
    dragRef.current = next;
    setDragState(next);
  }

  function monthAt(state: { activeTooltipIndex?: number | string | null; activeLabel?: string | number | null }) {
    const index = Number(state.activeTooltipIndex);
    if (Number.isInteger(index) && rows[index]) return String(rows[index].month);
    return state.activeLabel != null ? String(state.activeLabel) : null;
  }

  function finish() {
    const current = dragRef.current;
    if (!current) return;
    setDrag(null);
    if (current.anchor === current.head) {
      onRangeChange(null);
      return;
    }
    const [from, to] = [current.anchor, current.head].sort();
    onRangeChange({ from, to });
  }

  // Releasing the button outside the plot area must still end the selection.
  useEffect(() => {
    if (!drag) return;
    window.addEventListener("mouseup", finish);
    return () => window.removeEventListener("mouseup", finish);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drag === null]);

  const shaded = drag
    ? ([drag.anchor, drag.head].sort() as [string, string])
    : range
      ? ([range.from, range.to] as [string, string])
      : null;

  return (
    <div className="h-[min(52vh,440px)] w-full cursor-crosshair touch-none select-none" aria-label="Gráfico de puntuación. Arrastra para seleccionar un periodo.">
      <ClientOnly fallback={<div className="h-full w-full" />}>
        <ChartContainer config={config} className="aspect-auto h-full w-full">
          <LineChart
            data={rows}
            margin={{ top: 12, right: 16, left: -12, bottom: 0 }}
            onMouseDown={(state) => {
              const month = monthAt(state);
              if (month) setDrag({ anchor: month, head: month });
            }}
            onMouseMove={(state) => {
              if (!dragRef.current) return;
              const month = monthAt(state);
              if (month && month !== dragRef.current.head) setDrag({ ...dragRef.current, head: month });
            }}
            onMouseUp={finish}
            onTouchStart={(state) => {
              const month = monthAt(state);
              if (month) setDrag({ anchor: month, head: month });
            }}
            onTouchMove={(state) => {
              if (!dragRef.current) return;
              const month = monthAt(state);
              if (month && month !== dragRef.current.head) setDrag({ ...dragRef.current, head: month });
            }}
            onTouchEnd={finish}
          >
            <CartesianGrid vertical={false} strokeDasharray="3 3" />
            <XAxis
              dataKey="month"
              tickFormatter={(month: string) => formatMonth(month)}
              tickLine={false}
              axisLine={false}
              minTickGap={28}
            />
            <YAxis domain={yDomain} ticks={yTicks} tickLine={false} axisLine={false} width={36} />
            {yDomain[0] <= 50 && yDomain[1] >= 50 ? (
              <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" />
            ) : null}
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
