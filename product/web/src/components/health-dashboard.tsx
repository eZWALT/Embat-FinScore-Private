"use client";

import { useState } from "react";
import {
  ArrowDownRight,
  ArrowUpRight,
  Building2,
  CalendarDays,
  Minus,
  ShieldCheck,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  XAxis,
  YAxis,
} from "recharts";

import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { HealthSidebar } from "@/components/health-sidebar";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Separator } from "@/components/ui/separator";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import type { DashboardData, Trajectory } from "@/lib/data/types";

const scoreChartConfig = {
  score: { label: "Health Score", color: "var(--chart-1)" },
} satisfies ChartConfig;

const categoryChartConfig = {
  score: { label: "Puntuación", color: "var(--chart-2)" },
} satisfies ChartConfig;

const trajectoryLabels: Record<Trajectory, string> = {
  improving: "Mejorando",
  stable: "Estable",
  dip: "Caída puntual",
  deteriorating: "Deteriorándose",
  "insufficient history": "Historial insuficiente",
};

const confidenceLabels = { high: "Alta", medium: "Media", low: "Baja" } as const;

const monthFormatter = new Intl.DateTimeFormat("es-ES", {
  month: "short",
  year: "2-digit",
});

function formatMonth(month: string) {
  const [year, monthNumber] = month.split("-").map(Number);
  return monthFormatter
    .format(new Date(Date.UTC(year, monthNumber - 1, 1)))
    .replace(" ", " ’");
}

function Delta({ value }: { value: number | null }) {
  if (value === null) return <Minus className="size-4" />;
  const Icon = value > 0 ? ArrowUpRight : value < 0 ? ArrowDownRight : Minus;
  return (
    <span className="inline-flex items-center gap-1 font-mono text-sm tabular-nums">
      <Icon className="size-4" />
      {value > 0 ? "+" : ""}{value.toFixed(1)} pts
    </span>
  );
}

export function HealthDashboard({ data }: { data: DashboardData }) {
  const defaultCompany =
    data.companies.find((company) => company.trajectory === "improving") ?? data.companies[0];
  const [companyId, setCompanyId] = useState(defaultCompany.companyId);
  const company =
    data.companies.find((candidate) => candidate.companyId === companyId) ?? defaultCompany;

  const scoreHistory = company.scoreHistory.map((point) => ({
    ...point,
    label: formatMonth(point.month),
  }));

  return (
    <SidebarProvider>
      <HealthSidebar data={data} companyId={companyId} onCompanyChange={setCompanyId} />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b bg-background/95 px-4 backdrop-blur sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="h-4" />
            <div className="min-w-0">
              <p className="truncate font-mono text-sm font-medium">{company.companyId}</p>
              <p className="truncate text-xs text-muted-foreground">{company.groupId ?? "Sin grupo"}</p>
            </div>
          </div>
          <Badge variant="outline" className="font-mono text-[11px] font-normal text-muted-foreground">
            {formatMonth(data.asOfMonth)}
          </Badge>
        </header>

        <main className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-10">
          <section id="resumen" className="scroll-mt-20">
          <div className="max-w-2xl">
            <div className="mb-3 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
              <ShieldCheck className="size-3.5" />
              Monitor de salud financiera
            </div>
            <h1 className="text-balance text-3xl font-semibold tracking-tight sm:text-4xl">
              Entiende qué cambia antes de que se convierta en un problema.
            </h1>
            <p className="mt-3 text-sm leading-6 text-muted-foreground sm:text-base">
              Score explicable de 0 a 100 calculado a partir de la tesorería, deuda y comportamiento de pagos.
            </p>
          </div>
        </section>

        <Separator className="my-8" />

        <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Health Score</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span className="font-mono text-4xl font-medium tracking-tight tabular-nums">{company.score.toFixed(0)}</span>
                <span className="text-sm text-muted-foreground">/ 100</span>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Cambio mensual</CardTitle>
            </CardHeader>
            <CardContent className="flex h-11 items-center">
              <Delta value={company.delta1m} />
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Trayectoria</CardTitle>
            </CardHeader>
            <CardContent className="flex h-11 items-center">
              <Badge variant="secondary">{trajectoryLabels[company.trajectory]}</Badge>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Confianza</CardTitle>
            </CardHeader>
            <CardContent className="flex h-11 items-center justify-between gap-3">
              <span className="font-medium">{confidenceLabels[company.confidence]}</span>
              <span className="font-mono text-xs text-muted-foreground">{Math.round(company.coverage * 100)}% cobertura</span>
            </CardContent>
          </Card>
        </section>

        <section className="mt-4 grid gap-4 xl:grid-cols-[1.6fr_1fr]">
          <Card id="evolucion" className="scroll-mt-20">
            <CardHeader className="gap-1">
              <div className="flex items-center justify-between gap-4">
                <CardTitle className="text-base">Evolución del score</CardTitle>
                <Badge variant="outline" className="gap-1.5 font-normal text-muted-foreground">
                  <CalendarDays className="size-3" />
                  hasta {formatMonth(data.asOfMonth)}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">La dirección y persistencia importan tanto como el nivel actual.</p>
            </CardHeader>
            <CardContent>
              <ChartContainer config={scoreChartConfig} className="h-[300px] w-full aspect-auto">
                <LineChart data={scoreHistory} margin={{ top: 12, right: 8, left: -20, bottom: 0 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={28} />
                  <YAxis domain={[0, 100]} tickLine={false} axisLine={false} ticks={[0, 25, 50, 75, 100]} />
                  <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" />
                  <ChartTooltip
                    cursor={false}
                    content={<ChartTooltipContent indicator="line" />}
                  />
                  <Line
                    dataKey="score"
                    type="monotone"
                    stroke="var(--color-score)"
                    strokeWidth={2.5}
                    dot={false}
                    activeDot={{ r: 5 }}
                  />
                </LineChart>
              </ChartContainer>
            </CardContent>
          </Card>

          <Card id="categorias" className="scroll-mt-20">
            <CardHeader className="gap-1">
              <CardTitle className="text-base">Score por categoría</CardTitle>
              <p className="text-sm text-muted-foreground">Qué dimensiones sostienen o limitan el resultado actual.</p>
            </CardHeader>
            <CardContent>
              <ChartContainer config={categoryChartConfig} className="h-[300px] w-full aspect-auto">
                <BarChart data={company.categories} layout="vertical" margin={{ top: 4, right: 8, left: 8, bottom: 0 }}>
                  <CartesianGrid horizontal={false} strokeDasharray="3 3" />
                  <XAxis type="number" domain={[0, 100]} hide />
                  <YAxis
                    dataKey="label"
                    type="category"
                    tickLine={false}
                    axisLine={false}
                    width={112}
                    tick={{ fontSize: 11 }}
                  />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent hideLabel />} />
                  <Bar dataKey="score" fill="var(--color-score)" radius={[0, 5, 5, 0]} barSize={18} />
                </BarChart>
              </ChartContainer>
            </CardContent>
          </Card>
        </section>

        <section id="senales" className="mt-4 grid scroll-mt-20 gap-4 lg:grid-cols-[1.6fr_1fr]">
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Principal señal a revisar</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm leading-6 text-muted-foreground">
                {company.topReason ?? "No hay una señal dominante para este periodo."}
              </p>
            </CardContent>
          </Card>
          <Card>
            <CardContent className="flex min-h-28 items-center gap-4 pt-6">
              <div className="grid size-10 shrink-0 place-items-center rounded-lg bg-muted">
                <Building2 className="size-4" />
              </div>
              <div className="min-w-0">
                <p className="font-mono text-sm font-medium">{company.companyId}</p>
                <p className="mt-1 truncate text-xs text-muted-foreground">
                  {company.groupId ?? "Sin grupo"} · {company.country ?? "País no disponible"} · {company.erp ?? "Sin ERP"}
                </p>
              </div>
            </CardContent>
          </Card>
        </section>

        <p className="mt-8 max-w-4xl text-xs leading-5 text-muted-foreground">
          {data.disclaimer}
        </p>
        </main>
      </SidebarInset>
    </SidebarProvider>
  );
}
