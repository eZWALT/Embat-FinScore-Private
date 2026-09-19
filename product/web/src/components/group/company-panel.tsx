"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { AlertList } from "@/components/group/alert-list";
import {
  confidenceLabels,
  formatEur,
  formatMonth,
  formatPoints,
  reasonLabels,
  trajectoryLabels,
} from "@/components/group/labels";
import { GuardBadge } from "@/components/group/member-table";
import type { GroupCompanyPanel } from "@/lib/data/group-service";

const scoreChartConfig = {
  score: { label: "Índice de salud", color: "var(--chart-1)" },
} satisfies ChartConfig;

const severityColor = {
  act: "var(--destructive)",
  watch: "var(--chart-2)",
  info: "var(--chart-4)",
} as const;

export function CompanyPanel({ company }: { company: GroupCompanyPanel | null }) {
  if (!company) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Empresa</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">Este grupo no tiene empresas con puntuación.</p>
        </CardContent>
      </Card>
    );
  }

  const history = company.scoreHistory.map((point) => ({ ...point, label: formatMonth(point.month) }));
  const scoreByMonth = new Map(company.scoreHistory.map((point) => [point.month, point.score]));
  const marks = company.alertMonths
    .filter((mark) => scoreByMonth.has(mark.month))
    .map((mark) => ({ ...mark, label: formatMonth(mark.month), score: scoreByMonth.get(mark.month)! }));

  return (
    <Card>
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <CardTitle className="font-mono text-base">{company.companyId}</CardTitle>
          <span className="text-xs text-muted-foreground">último mes {formatMonth(company.latestMonth)}</span>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="font-mono text-3xl font-medium tabular-nums">{company.score.toFixed(0)}</span>
          <span className="text-sm text-muted-foreground">/ 100</span>
          {company.guard && company.scorePreCap > company.score && (
            <span className="text-xs text-muted-foreground">
              (sin límite sería {company.scorePreCap.toFixed(0)})
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="secondary">{trajectoryLabels[company.trajectory]}</Badge>
          <Badge variant="outline">Confianza {confidenceLabels[company.confidence].toLowerCase()}</Badge>
          <GuardBadge guard={company.guard} />
        </div>
        {company.confidenceNote && (
          <p className="text-xs text-muted-foreground">{company.confidenceNote}</p>
        )}
      </CardHeader>

      <CardContent className="space-y-6">
        <div>
          <ChartContainer config={scoreChartConfig} className="h-[220px] w-full aspect-auto">
            <LineChart data={history} margin={{ top: 12, right: 8, left: -20, bottom: 0 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={28} />
              <YAxis domain={[0, 100]} tickLine={false} axisLine={false} ticks={[0, 25, 50, 75, 100]} />
              <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" />
              <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="line" />} />
              <Line
                dataKey="score"
                type="monotone"
                stroke="var(--color-score)"
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 5 }}
                isAnimationActive={false}
              />
              {marks.map((mark, index) => (
                <ReferenceDot
                  key={`${mark.month}-${index}`}
                  x={mark.label}
                  y={mark.score}
                  r={5}
                  fill={severityColor[mark.severity]}
                  stroke="var(--background)"
                  strokeWidth={1.5}
                />
              ))}
            </LineChart>
          </ChartContainer>
          <p className="mt-1 text-[11px] text-muted-foreground">
            Los puntos marcan los meses con alerta ({marks.length}).
          </p>
        </div>

        <div>
          <h3 className="mb-2 text-sm font-medium">Por qué la puntuación no es más alta</h3>
          {company.reasons.length === 0 ? (
            <p className="text-sm text-muted-foreground">Sin razones detalladas para este mes.</p>
          ) : (
            <ul className="divide-y">
              {company.reasons.map((reason) => (
                <li key={reason.item} className="flex items-start justify-between gap-3 py-2 first:pt-0 last:pb-0">
                  <div className="min-w-0">
                    <p className="text-sm">{reasonLabels[reason.item] ?? reason.label}</p>
                    <p className="text-xs text-muted-foreground">{reason.sentence}</p>
                  </div>
                  <div className="shrink-0 text-right font-mono text-xs tabular-nums">
                    <p>{formatPoints(reason.points)} pts</p>
                    <p className="text-muted-foreground">
                      {reason.eur !== null ? formatEur(reason.eur, company.currency) : "—"}
                    </p>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h3 className="mb-2 text-sm font-medium">Alertas de la empresa</h3>
          <AlertList
            alerts={company.alerts}
            currency={company.currency}
            emptyText="Sin alertas en la ventana de detalle."
          />
        </div>
      </CardContent>
    </Card>
  );
}
