"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { ArrowDownRight, ArrowUpRight, Layers, Minus, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { AppHeader } from "@/components/app-header";
import { CompanyAlerts } from "@/components/company-alerts";
import { EntityCombobox } from "@/components/entity-combobox";
import { HealthIndexHelp } from "@/components/health-index-help";
import { ModeToggle, type AnalysisMode } from "@/components/mode-toggle";
import { MonitorPlot } from "@/components/monitor-plot";
import { QuickAnalysis } from "@/components/quick/quick-analysis";
import { Segmented, type SegmentedOption } from "@/components/segmented";
import { confidenceLabels, scoreColor, trajectoryLabels } from "@/components/group/labels";
import { HealthScoreChat } from "@/components/health-score-chat";
import { ProductCredit } from "@/components/product-credit";
import { HealthScoreView } from "@/components/health-score-view";
import { type MonthRange } from "@/components/quick/quick-chart";
import { SERIES_COLOR_LABELS } from "@/components/quick/series";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { DashboardView } from "@/lib/agent/view-context";
import type { DashboardCompany, DashboardData } from "@/lib/data/types";

type DeepView = "overview" | "health-score";

const DEEP_VIEWS: SegmentedOption<DeepView>[] = [
  { value: "overview", label: "Resumen" },
  { value: "health-score", label: "Comparar" },
];

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

function legendSeries(companies: DashboardCompany[]) {
  return companies.map((company, index) => ({
    companyId: company.companyId,
    groupId: company.groupId ?? undefined,
    color: SERIES_COLOR_LABELS[index % SERIES_COLOR_LABELS.length],
    score: company.score,
    trajectory: trajectoryLabels[company.trajectory],
    delta3m: company.delta3m,
  }));
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
  const [view, setView] = useState<DeepView>("overview");
  const [chatOpen, setChatOpen] = useState(openChat);
  const [mode, setMode] = useState<AnalysisMode>(initialMode);
  const [quickCompanies, setQuickCompanies] = useState<DashboardData["companies"]>([]);
  const [indiceCompanies, setIndiceCompanies] = useState<DashboardCompany[]>([]);
  const [quickRange, setQuickRange] = useState<MonthRange | null>(null);
  const [seedPrompt, setSeedPrompt] = useState<string | undefined>();
  const [seedKey, setSeedKey] = useState(0);
  const company =
    data.companies.find((candidate) => candidate.companyId === companyId) ?? defaultCompany;

  useEffect(() => {
    if (openChat) setChatOpen(true);
  }, [openChat]);

  useEffect(() => {
    if (window.location.hash === "#alerts") {
      setView("overview");
      requestAnimationFrame(() => document.getElementById("alerts")?.scrollIntoView());
    }
  }, []);

  // The company lives in the URL so a shared link or a link out of Grupos keeps it, without refetching.
  function selectCompany(next: string) {
    setCompanyId(next);
    window.history.replaceState(null, "", `/${queryFor({ company: next })}`);
  }

  function changeMode(next: AnalysisMode) {
    setMode(next);
    setSeedPrompt(undefined);
    window.history.replaceState(null, "", next === "quick" ? "/" : `/${queryFor({ modo: "profundo", company: companyId })}`);
  }

  function explainRange(prompt: string) {
    setSeedPrompt(prompt);
    setSeedKey((key) => key + 1);
    setChatOpen(true);
  }

  const chatCompany =
    mode === "deep"
      ? view === "overview"
        ? company
        : undefined
      : quickCompanies.length === 1
        ? quickCompanies[0]
        : undefined;

  const dashboardView = useMemo((): DashboardView => {
    if (mode === "quick") {
      if (quickCompanies.length === 0) {
        return { mode: "rapido", screen: "inicio", asOf: data.asOfMonth };
      }
      return {
        mode: "rapido",
        screen: "rapido_chart",
        asOf: data.asOfMonth,
        series: legendSeries(quickCompanies),
        periodFrom: quickRange?.from,
        periodTo: quickRange?.to,
      };
    }
    if (view === "health-score") {
      return {
        mode: "profundo",
        screen: "indice",
        asOf: data.asOfMonth,
        series: legendSeries(indiceCompanies),
      };
    }
    return {
      mode: "profundo",
      screen: "resumen",
      asOf: data.asOfMonth,
      focusCompanyId: company.companyId,
      focusGroupId: company.groupId ?? undefined,
      score: company.score,
      trajectory: trajectoryLabels[company.trajectory],
      delta1m: company.delta1m,
      topReason: company.topReason,
      categories: company.categories.map((row) => ({ label: row.label, score: row.score })),
    };
  }, [mode, view, data.asOfMonth, quickCompanies, quickRange, indiceCompanies, company]);

  return (
    <div className="flex min-h-svh flex-col">
      <AppHeader>
        <ModeToggle mode={mode} onChange={changeMode} />
      </AppHeader>

      {mode === "quick" ? (
        <div key="quick" className="flex flex-1 flex-col animate-in fade-in-0 duration-300 motion-reduce:animate-none">
          <QuickAnalysis
            data={data}
            onCompaniesChange={setQuickCompanies}
            onRangeChange={setQuickRange}
            onExplain={explainRange}
          />
        </div>
      ) : (
        <main
          key="deep"
          className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 animate-in fade-in-0 duration-300 motion-reduce:animate-none sm:px-6 lg:px-8 lg:py-8"
        >
          <div className="mb-6 flex flex-wrap items-end justify-between gap-3">
            {view === "overview" ? (
              <div className="w-full sm:w-72">
                <p className="mb-1.5 text-xs font-medium text-muted-foreground">Empresa analizada</p>
                <EntityCombobox
                  ariaLabel="Empresa analizada"
                  options={data.companies.map((candidate) => ({
                    value: candidate.companyId,
                    label: candidate.companyId,
                    detail: `${candidate.score.toFixed(0)} pts`,
                  }))}
                  value={companyId}
                  onChange={(value) => value && selectCompany(value)}
                  placeholder="Elige una empresa"
                  searchPlaceholder="Buscar empresa (p. ej. 0462)"
                />
              </div>
            ) : (
              <span />
            )}
            <Segmented options={DEEP_VIEWS} value={view} onChange={setView} ariaLabel="Vista del análisis profundo" />
          </div>

          {view === "health-score" ? (
            <HealthScoreView
              data={data}
              initialCompanyId={company.companyId}
              onSelectionChange={setIndiceCompanies}
              onExplain={explainRange}
            />
          ) : (
            <>
              <section id="resumen" className="flex scroll-mt-20 flex-col gap-3 lg:flex-row lg:items-end lg:justify-between">
                <div className="min-w-0">
                  <h1 className="font-mono text-2xl font-semibold tracking-tight">{company.companyId}</h1>
                  <p className="mt-1 text-sm text-muted-foreground">
                    {[company.groupId ?? "Sin grupo", company.country, company.erp].filter(Boolean).join(" · ")}
                  </p>
                </div>
                {company.groupId ? (
                  <nav aria-label="Ir a otras vistas de esta empresa" className="flex flex-wrap gap-2">
                    <Button asChild variant="outline" size="sm">
                      <Link href={`/grupos${queryFor({ group: company.groupId, company: company.companyId })}`}>
                        <Layers data-icon="inline-start" />
                        Ver grupo
                      </Link>
                    </Button>
                  </nav>
                ) : null}
              </section>

              <section className="mt-6 grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
                <Card>
                  <CardHeader className="pb-2">
                    <CardTitle className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                      Índice de salud
                      <HealthIndexHelp />
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <div
                          tabIndex={0}
                          className="flex w-fit cursor-help items-baseline gap-2 rounded-md outline-none focus-visible:ring-3 focus-visible:ring-ring/50"
                        >
                          <span
                            className="size-2.5 self-center rounded-full"
                            style={{ background: scoreColor(company.score) }}
                            aria-hidden="true"
                          />
                          <span className="font-mono text-4xl font-medium tracking-tight tabular-nums underline decoration-muted-foreground/40 decoration-dotted underline-offset-[6px]">
                            {company.score.toFixed(0)}
                          </span>
                          <span className="text-sm text-muted-foreground">/ 100</span>
                        </div>
                      </TooltipTrigger>
                      <TooltipContent side="bottom" align="start" className="min-w-60 flex-col items-stretch gap-2 p-3">
                        <p className="text-[11px] font-medium opacity-70">Puntuación por categoría</p>
                        <ul className="space-y-1.5">
                          {company.categories.map((category) => (
                            <li key={category.id}>
                              <div className="flex items-baseline justify-between gap-4">
                                <span>{category.label}</span>
                                <span className="font-mono tabular-nums">
                                  {category.score === null ? "—" : category.score.toFixed(0)}
                                </span>
                              </div>
                              <div className="mt-1 h-1 rounded-full bg-background/25">
                                <div
                                  className="h-full rounded-full bg-background"
                                  style={{ width: `${Math.max(0, Math.min(100, category.score ?? 0))}%` }}
                                />
                              </div>
                            </li>
                          ))}
                        </ul>
                      </TooltipContent>
                    </Tooltip>
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

              <section className="mt-4 grid items-start gap-4 xl:grid-cols-[1.6fr_1fr]">
                <MonitorPlot company={company} companies={data.companies} asOfMonth={data.asOfMonth} />
                <CompanyAlerts companyId={company.companyId} />
              </section>

              <ProductCredit className="mt-8" />
            </>
          )}
        </main>
      )}
      <HealthScoreChat
        key={`${mode}:${view}:${chatCompany?.companyId ?? "index"}`}
        companyId={chatCompany?.companyId}
        groupId={chatCompany?.groupId ?? undefined}
        asOf={data.asOfMonth}
        seedPrompt={seedPrompt}
        seedKey={seedKey}
        view={dashboardView}
        open={chatOpen}
        onOpenChange={setChatOpen}
      />
    </div>
  );
}
