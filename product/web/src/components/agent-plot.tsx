"use client";

import { Bar, BarChart, Line, LineChart, XAxis, YAxis } from "recharts";

import { ChartContainer, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import type { PlotSpec } from "@/lib/agent/plot-spec";
import type { WatcherSparkPoint } from "@/lib/agent/watcher-post";
import { formatMonth } from "@/lib/format-month";

const sparkConfig = {
  score: { label: "Índice", color: "var(--chart-1)" },
} satisfies ChartConfig;

export function WatcherSpark({ id, points }: { id: string; points: WatcherSparkPoint[] }) {
  return (
    <div className="pt-1">
      <p className="mb-1 font-mono text-[10px] text-muted-foreground">{id}</p>
      <ChartContainer config={sparkConfig} className="aspect-auto h-16 w-full">
        <LineChart data={points} margin={{ top: 4, right: 4, left: 0, bottom: 0 }}>
          <XAxis dataKey="month" hide />
          <YAxis domain={[0, 100]} hide />
          <Line type="monotone" dataKey="score" stroke="var(--color-score)" strokeWidth={1.5} dot={false} />
        </LineChart>
      </ChartContainer>
    </div>
  );
}

function tickMonth(value: string) {
  return /^\d{4}-\d{2}$/.test(value) ? formatMonth(value) : value;
}

export function AgentPlot({ spec }: { spec: PlotSpec }) {
  const names = Object.keys(spec.series);
  const data = spec.x.map((month, i) => {
    const row: Record<string, string | number | null> = { month };
    for (const name of names) row[name] = spec.series[name][i] ?? null;
    if (spec.band) {
      row._lo = spec.band.lower[i] ?? null;
      row._hi = spec.band.upper[i] ?? null;
    }
    return row;
  });
  const config: ChartConfig = Object.fromEntries(
    names.map((name, i) => [name, { label: name, color: `var(--chart-${(i % 5) + 1})` }]),
  );
  const Chart = spec.kind === "bar" ? BarChart : LineChart;

  return (
    <div className="space-y-1 pt-1">
      <p className="text-xs font-medium">{spec.title}</p>
      <ChartContainer config={config} className="aspect-auto h-44 w-full">
        <Chart data={data} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
          <XAxis dataKey="month" tick={{ fontSize: 10 }} interval="preserveStartEnd" tickFormatter={tickMonth} />
          <YAxis domain={[0, 100]} width={28} tick={{ fontSize: 10 }} />
          <ChartTooltip content={<ChartTooltipContent />} />
          {spec.band ? (
            <>
              <Line type="monotone" dataKey="_lo" stroke="var(--muted-foreground)" strokeWidth={1} dot={false} connectNulls={false} />
              <Line type="monotone" dataKey="_hi" stroke="var(--muted-foreground)" strokeWidth={1} dot={false} connectNulls={false} />
            </>
          ) : null}
          {names.map((name) =>
            spec.kind === "bar" ? (
              <Bar key={name} dataKey={name} fill={`var(--color-${name})`} radius={[4, 4, 0, 0]} />
            ) : (
              <Line
                key={name}
                type="monotone"
                dataKey={name}
                stroke={`var(--color-${name})`}
                strokeWidth={1.5}
                dot={false}
                connectNulls={false}
              />
            ),
          )}
        </Chart>
      </ChartContainer>
    </div>
  );
}
