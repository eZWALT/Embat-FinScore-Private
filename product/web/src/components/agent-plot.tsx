"use client";

import { Bar, BarChart, Cell, LabelList, Line, LineChart, Pie, PieChart, ReferenceLine, XAxis, YAxis } from "recharts";

import { PlotExpand, plotHeight } from "@/components/plot-expand";
import { ChartContainer, ChartLegend, ChartLegendContent, ChartTooltip, ChartTooltipContent, type ChartConfig } from "@/components/ui/chart";
import type { PlotSpec } from "@/lib/agent/plot-spec";
import type { WatcherSparkPoint } from "@/lib/agent/watcher-post";
import { entityLabel } from "@/lib/display";
import { formatMonthShort } from "@/lib/format-month";

const sparkConfig = {
  score: { label: "Índice", color: "var(--chart-1)" },
} satisfies ChartConfig;

export function WatcherSpark({ id, points }: { id: string; points: WatcherSparkPoint[] }) {
  return (
    <div className="pt-1">
      <p className="mb-1 text-[10px] text-muted-foreground" title={id}>
        {entityLabel(id)}
      </p>
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
  return /^\d{4}-\d{2}$/.test(value) ? formatMonthShort(value) : value;
}

function tickLabel(value: string) {
  return /^\d{4}-\d{2}$/.test(value) ? formatMonthShort(value) : entityLabel(value);
}

const HIGHLIGHT = "var(--destructive)";

export function AgentPlot({ spec }: { spec: PlotSpec }) {
  const names = Object.keys(spec.series);
  const highlight = new Set(spec.highlight ?? []);
  const badges = spec.badges ?? {};
  const data = spec.x.map((label, i) => {
    const row: Record<string, string | number | null> = { month: label, _badge: badges[label] ?? "" };
    for (const name of names) row[name] = spec.series[name][i] ?? null;
    if (spec.band) {
      row._lo = spec.band.lower[i] ?? null;
      row._hi = spec.band.upper[i] ?? null;
    }
    return row;
  });
  const config: ChartConfig = Object.fromEntries(
    names.map((name, i) => [name, { label: entityLabel(name), color: `var(--chart-${(i % 5) + 1})` }]),
  );
  const yDomain: [number, number] | ["auto", "auto"] = spec.y_label === "0–100" ? [0, 100] : ["auto", "auto"];

  if (spec.kind === "pie") {
    const key = names[0];
    const slices = data.map((row, i) => ({ ...row, fill: `var(--chart-${(i % 5) + 1})` }));
    const pieConfig: ChartConfig = Object.fromEntries(
      spec.x.map((label, i) => [label, { label: entityLabel(label), color: `var(--chart-${(i % 5) + 1})` }]),
    );
    return (
      <PlotExpand title={spec.title} chrome="row" className="pt-1">
        {(frame) => (
          <ChartContainer config={pieConfig} className={plotHeight(frame, "aspect-auto h-44 w-full")}>
            <PieChart margin={{ top: 4, right: 4, left: 4, bottom: 4 }}>
              <ChartTooltip content={<ChartTooltipContent nameKey="month" hideLabel />} />
              <Pie data={slices} dataKey={key} nameKey="month" innerRadius="45%" strokeWidth={2} />
              <ChartLegend content={<ChartLegendContent nameKey="month" />} />
            </PieChart>
          </ChartContainer>
        )}
      </PlotExpand>
    );
  }

  const Chart = spec.kind === "bar" ? BarChart : LineChart;

  return (
    <PlotExpand title={spec.title} chrome="row" className="pt-1">
      {(frame) => (
        <ChartContainer config={config} className={plotHeight(frame, "aspect-auto h-44 w-full")}>
          <Chart data={data} margin={{ top: 14, right: 8, left: 0, bottom: 0 }}>
            <XAxis
              dataKey="month"
              tick={{ fontSize: 10 }}
              interval={spec.kind === "bar" && data.length <= 12 ? 0 : "preserveStartEnd"}
              tickFormatter={spec.kind === "bar" ? tickLabel : tickMonth}
            />
            <YAxis domain={yDomain} width={spec.y_label === "€" ? 48 : 28} tick={{ fontSize: 10 }} />
            <ChartTooltip content={<ChartTooltipContent labelFormatter={(value) => tickLabel(String(value))} />} />
            {spec.ref_lines?.map((line) => (
              <ReferenceLine
                key={line.label}
                y={line.value}
                stroke="var(--muted-foreground)"
                strokeDasharray="4 3"
                label={{ value: `${line.label} ${line.value.toLocaleString("es-ES", { maximumFractionDigits: 1 })}`, position: "insideTopRight", fontSize: 10, fill: "var(--muted-foreground)" }}
              />
            ))}
            {spec.band ? (
              <>
                <Line type="monotone" dataKey="_lo" stroke="var(--muted-foreground)" strokeWidth={1} dot={false} connectNulls={false} />
                <Line type="monotone" dataKey="_hi" stroke="var(--muted-foreground)" strokeWidth={1} dot={false} connectNulls={false} />
              </>
            ) : null}
            {names.map((name, seriesIndex) =>
              spec.kind === "bar" ? (
                <Bar key={name} dataKey={name} fill={`var(--color-${name})`} radius={[4, 4, 0, 0]}>
                  {seriesIndex === 0 && highlight.size
                    ? data.map((row) => (
                        <Cell key={String(row.month)} fill={highlight.has(String(row.month)) ? HIGHLIGHT : `var(--color-${name})`} />
                      ))
                    : null}
                  {seriesIndex === 0 && Object.keys(badges).length ? (
                    <LabelList dataKey="_badge" position="top" fontSize={9} fill="var(--foreground)" />
                  ) : null}
                </Bar>
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
      )}
    </PlotExpand>
  );
}
