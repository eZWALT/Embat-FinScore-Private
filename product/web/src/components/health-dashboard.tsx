"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowDownRight,
  ArrowUpRight,
  CalendarDays,
  ChartNoAxesCombined,
  Eye,
  Layers,
  Minus,
  TriangleAlert,
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
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ClientOnly } from "@/components/client-only";
import { ModeToggle, type AnalysisMode } from "@/components/mode-toggle";
import { QuickAnalysis } from "@/components/quick/quick-analysis";
import { confidenceLabels, scoreColor, trajectoryLabels } from "@/components/group/labels";
import { HealthScoreChat } from "@/components/health-score-chat";
import { HealthScoreView } from "@/components/health-score-view";
import { HealthSidebar, type AppView } from "@/components/health-sidebar";
import { ResumenVigilancia } from "@/components/resumen-vigilancia";
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Separator } from "@/components/ui/separator";
import { SidebarInset, SidebarProvider, SidebarTrigger } from "@/components/ui/sidebar";
import { formatMonth } from "@/lib/format-month";
import type { DashboardData } from "@/lib/data/types";

const scoreChartConfig = {
  score: { label: "Índice de salud", color: "var(--chart-1)" },
} satisfies ChartConfig;

const categoryChartConfig = {
  score: { label: "Puntuación", color: "var(--chart-2)" },
} satisfies ChartConfig;

function Delta({ value }: { value: number | null }) {
  if (value === null) return <span className="text-sm text-muted-foreground">—</span>;
  const Icon = value > 0 ? ArrowUpRight : value < 0 ? ArrowDownRight : Minus;
  return (
    <span className="inline-flex items-center gap-1 font-mono text-sm tabular-nums">
      <Icon className="size-4" />
      {value > 0 ? "+" : ""}{value.toFixed(1)} pts
    </span>
  );
}

function queryFor(params: Record<string, string | null>) {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) if (value) query.set(key, value);
  return query.size ? `?${query}` : "";
}

export function HealthDashboard({
  data,
  openChat = false,
  initialCompanyId,
  initialMode = "quick",
}: {
  data: DashboardData;
  openChat?: boolean;
  initialCompanyId?: string;
  initialMode?: AnalysisMode;
}) {
  const defaultCompany =
    data.companies.find((company) => company.companyId === initialCompanyId) ??
    data.companies.find((company) => company.trajectory === "improving") ??
    data.companies[0];
  const [companyId, setCompanyId] = useState(defaultCompany.companyId);
  const [view, setView] = useState<AppView>("overview");
  const [chatOpen, setChatOpen] = useState(openChat);
  const [mode, setMode] = useState<AnalysisMode>(initialMode);
  const company =
    data.companies.find((candidate) => candidate.companyId === companyId) ?? defaultCompany;

  useEffect(() => {
    if (openChat) setChatOpen(true);
  }, [openChat]);

  useEffect(() => {
    if (window.location.hash === "#vigilancia") setView("overview");
  }, []);

  // The company lives in the URL so a shared link or a link out of Grupos keeps it, without refetching.
  function selectCompany(next: string) {
    setCompanyId(next);
    window.history.replaceState(null, "", `/${queryFor({ company: next })}`);
  }

  function changeMode(next: AnalysisMode) {
    setMode(next);
    window.history.replaceState(null, "", next === "quick" ? "/" : `/${queryFor({ modo: "profundo", company: companyId })}`);
  }

  function goVigilancia() {
    setView("overview");
    requestAnimationFrame(() => {
      document.getElementById("vigilancia")?.scrollIntoView({ behavior: "smooth" });
    });
  }

  const scoreHistory = company.scoreHistory.map((point) => ({
    ...point,
    label: formatMonth(point.month),
  }));

  if (mode === "quick") {
    return <QuickAnalysis data={data} mode={mode} onModeChange={changeMode} />;
  }

  return (
    <SidebarProvider>
      <HealthSidebar
        data={data}
        companyId={companyId}
        onCompanyChange={selectCompany}
        view={view}
        onViewChange={setView}
        onVigilancia={goVigilancia}
      />
      <SidebarInset>
        <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b bg-background/95 px-4 backdrop-blur sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <SidebarTrigger />
            <Separator orientation="vertical" className="h-4" />
            <div className="min-w-0">
              <p className="truncate font-mono text-sm font-medium">
                {view === "health-score" ? "Índice de salud" : company.companyId}
              </p>
              <p className="truncate text-xs text-muted-foreground">
                {view === "health-score"
                  ? `${data.companies.length} empresas`
                  : (company.groupId ?? "Sin grupo")}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <ModeToggle mode={mode} onChange={changeMode} />
            <Badge variant="outline" className="hidden font-mono text-[11px] font-normal text-muted-foreground sm:inline-flex">
              {formatMonth(data.asOfMonth)}
            </Badge>
          </div>
        </header>

        <main
          className={
            view === "health-score"
              ? "w-full px-4 py-6 sm:px-6 lg:px-8 lg:py-8"
              : "mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8 lg:py-10"
          }
        >
          {view === "health-score" ? (
            <HealthScoreView data={data} initialCompanyId={company.companyId} />
          ) : (
            <>
          <section id="resumen" className="flex scroll-mt-20 flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
            <div className="min-w-0">
              <h1 className="font-mono text-2xl font-semibold tracking-tight">{company.companyId}</h1>
              <p className="mt-1 text-sm text-muted-foreground">
                {[company.groupId ?? "Sin grupo", company.country, company.erp].filter(Boolean).join(" · ")}
              </p>
            </div>
            <nav aria-label="Ir a otras vistas de esta empresa" className="flex flex-wrap gap-2">
              {company.groupId ? (
                <Button asChild variant="outline" size="sm">
                  <Link href={`/grupos${queryFor({ group: company.groupId, company: company.companyId })}`}>
                    <Layers data-icon="inline-start" />
                    Ver grupo
                  </Link>
                </Button>
              ) : null}
              <Button variant="outline" size="sm" onClick={() => setView("health-score")}>
                <ChartNoAxesCombined data-icon="inline-start" />
                Comparar
              </Button>
              <Button variant="outline" size="sm" onClick={goVigilancia}>
                <Eye data-icon="inline-start" />
                Vigilancia
              </Button>
            </nav>
          </section>

        <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
          <Card>
            <CardHeader className="pb-2">
              <CardTitle className="text-xs font-medium text-muted-foreground">Índice de salud</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex items-baseline gap-2">
                <span
                  className="size-2.5 self-center rounded-full"
                  style={{ background: scoreColor(company.score) }}
                  aria-hidden="true"
                />
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

        <section
          id="senales"
          aria-label="Principal señal a revisar"
          className="mt-4 flex scroll-mt-20 items-start gap-3 rounded-xl border bg-card px-4 py-3"
        >
          <TriangleAlert className="mt-0.5 size-4 shrink-0 text-muted-foreground" aria-hidden="true" />
          <div className="min-w-0">
            <p className="text-xs font-medium text-muted-foreground">Principal señal a revisar</p>
            <p className="mt-1 text-sm leading-6">
              {company.topReason ?? "No hay una señal dominante para este periodo."}
            </p>
          </div>
        </section>

        <section className="mt-4 grid gap-4 xl:grid-cols-[1.6fr_1fr]">
          <Card id="evolucion" className="scroll-mt-20">
            <CardHeader className="gap-1">
              <div className="flex items-center justify-between gap-4">
                <CardTitle className="text-base">Evolución de la puntuación</CardTitle>
                <Badge variant="outline" className="gap-1.5 font-normal text-muted-foreground">
                  <CalendarDays className="size-3" />
                  hasta {formatMonth(data.asOfMonth)}
                </Badge>
              </div>
              <p className="text-sm text-muted-foreground">La dirección y persistencia importan tanto como el nivel actual.</p>
            </CardHeader>
            <CardContent>
              <ClientOnly fallback={<div className="h-[300px] w-full" />}>
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
              </ClientOnly>
            </CardContent>
          </Card>

          <Card id="categorias" className="scroll-mt-20">
            <CardHeader className="gap-1">
              <CardTitle className="text-base">Puntuación por categoría</CardTitle>
              <p className="text-sm text-muted-foreground">Qué dimensiones sostienen o limitan el resultado actual.</p>
            </CardHeader>
            <CardContent>
              <ClientOnly fallback={<div className="h-[300px] w-full" />}>
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
              </ClientOnly>
            </CardContent>
          </Card>
        </section>

        <ResumenVigilancia companyId={company.companyId} />

        <p className="mt-8 max-w-4xl text-xs leading-5 text-muted-foreground">
          {data.disclaimer}
        </p>
            </>
          )}
        </main>
        <HealthScoreChat
          key={`${view}:${view === "overview" ? company.companyId : "indice"}`}
          companyId={view === "overview" ? company.companyId : undefined}
          groupId={view === "overview" ? company.groupId ?? undefined : undefined}
          asOf={data.asOfMonth}
          open={chatOpen}
          onOpenChange={setChatOpen}
        />
      </SidebarInset>
    </SidebarProvider>
  );
}
