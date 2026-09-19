"use client";

import { useMemo, useState } from "react";
import { Activity, MousePointerClick, Search, TrendingDown, TrendingUp, X, type LucideIcon } from "lucide-react";
import { cn } from "cn";

import { HealthScoreChat } from "@/components/health-score-chat";
import { ModeToggle, type AnalysisMode } from "@/components/mode-toggle";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { formatPoints } from "@/components/group/labels";
import { SERIES_COLORS } from "@/components/health-score-view";
import { QuickChart, type MonthRange } from "@/components/quick/quick-chart";
import { buildPrompt } from "@/components/quick/quick-explain";
import { QuickList } from "@/components/quick/quick-list";
import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany, DashboardData } from "@/lib/data/types";

type Tab = "top" | "bottom" | "search";

const MAX_PICKED = 8;

const TABS: { id: Tab; label: string; hint: string; icon: LucideIcon }[] = [
  { id: "top", label: "Mejores 5", hint: "Las cinco empresas con mayor índice hoy", icon: TrendingUp },
  { id: "bottom", label: "Peores 5", hint: "Las cinco con menor índice hoy", icon: TrendingDown },
  { id: "search", label: "Buscar", hint: "Por nombre o grupo, con lista ordenable", icon: Search },
];

/** Quick mode: pick a set of companies, see their score lines, drag over a period to get it explained. */
export function QuickAnalysis({
  data,
  mode,
  onModeChange,
}: {
  data: DashboardData;
  mode: AnalysisMode;
  onModeChange: (mode: AnalysisMode) => void;
}) {
  const [tab, setTab] = useState<Tab | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [range, setRange] = useState<MonthRange | null>(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [seedPrompt, setSeedPrompt] = useState<string | undefined>();
  const [seedKey, setSeedKey] = useState(0);

  const byScore = useMemo(
    () => data.companies.slice().sort((a, b) => b.score - a.score || a.companyId.localeCompare(b.companyId)),
    [data.companies],
  );

  const selected: DashboardCompany[] = useMemo(() => {
    if (tab === "top") return byScore.slice(0, 5);
    if (tab === "bottom") return byScore.slice(-5).reverse();
    if (tab === "search") {
      return picked
        .map((id) => data.companies.find((company) => company.companyId === id))
        .filter((company): company is DashboardCompany => Boolean(company));
    }
    return [];
  }, [tab, byScore, picked, data.companies]);

  function selectRange(next: MonthRange | null) {
    if (!next || selected.length === 0) {
      setRange(null);
      setSeedPrompt(undefined);
      return;
    }
    setRange(next);
    setSeedPrompt(buildPrompt(selected, next));
    setSeedKey((key) => key + 1);
    setChatOpen(true);
  }

  function choose(next: Tab) {
    setTab(next);
    selectRange(null);
  }

  function toggle(companyId: string) {
    selectRange(null);
    setPicked((current) =>
      current.includes(companyId)
        ? current.filter((id) => id !== companyId)
        : current.length >= MAX_PICKED
          ? current
          : [...current, companyId],
    );
  }

  return (
    <div className="flex min-h-svh flex-col bg-background">
      <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <div className="grid size-8 shrink-0 place-items-center rounded-lg border bg-background">
            <Activity className="size-4" aria-hidden="true" />
          </div>
          <span className="hidden text-sm font-semibold sm:inline">Centinela de salud</span>
        </div>
        <ModeToggle mode={mode} onChange={onModeChange} />
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="hidden font-mono text-[11px] font-normal text-muted-foreground sm:inline-flex">
            {formatMonth(data.asOfMonth)}
          </Badge>
          <ThemeSwitcher />
        </div>
      </header>

      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-4 px-4 py-6 pb-24 sm:px-6 lg:py-8">
        {tab === null ? (
          <section
            aria-label="Elige qué quieres ver"
            className="grid flex-1 place-items-center rounded-2xl border border-dashed px-4 py-16"
          >
            <div className="w-full max-w-3xl text-center">
              <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl">¿Qué quieres ver?</h1>
              <div className="mt-8 grid gap-3 sm:grid-cols-3">
                {TABS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    onClick={() => choose(option.id)}
                    className="group flex flex-col items-center gap-3 rounded-xl border bg-card px-4 py-6 text-center outline-none transition-colors hover:border-foreground/40 hover:bg-muted/50 focus-visible:ring-3 focus-visible:ring-ring/50"
                  >
                    <option.icon className="size-5 text-muted-foreground group-hover:text-foreground" aria-hidden="true" />
                    <span className="text-base font-medium">{option.label}</span>
                    <span className="text-xs leading-5 text-muted-foreground">{option.hint}</span>
                  </button>
                ))}
              </div>
            </div>
          </section>
        ) : (
          <>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div role="tablist" aria-label="Qué quieres ver" className="flex items-center rounded-lg border bg-card p-0.5">
                {TABS.map((option) => (
                  <button
                    key={option.id}
                    type="button"
                    role="tab"
                    aria-selected={tab === option.id}
                    onClick={() => choose(option.id)}
                    className={cn(
                      "inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-sm font-medium transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
                      tab === option.id ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground",
                    )}
                  >
                    <option.icon className="size-3.5" aria-hidden="true" />
                    {option.label}
                  </button>
                ))}
              </div>
              {selected.length > 0 && !range ? (
                <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <MousePointerClick className="size-3.5" aria-hidden="true" />
                  Arrastra sobre el gráfico para explicar un periodo
                </p>
              ) : null}
              {range ? (
                <Button type="button" variant="ghost" size="sm" onClick={() => selectRange(null)}>
                  <X data-icon="inline-start" />
                  Quitar selección
                </Button>
              ) : null}
            </div>

            <div className={cn("grid gap-4", tab === "search" && "lg:grid-cols-[19rem_minmax(0,1fr)]")}>
              {tab === "search" ? (
                <QuickList companies={data.companies} picked={picked} max={MAX_PICKED} onToggle={toggle} />
              ) : null}

              <div className="min-w-0 space-y-3 max-lg:order-first">
                {selected.length === 0 ? (
                  <div className="grid h-[min(52vh,440px)] place-items-center rounded-xl border border-dashed px-4 text-center text-sm text-muted-foreground">
                    Marca una o más empresas de la lista para ver su evolución.
                  </div>
                ) : (
                  <>
                    <QuickChart companies={selected} range={range} onRangeChange={selectRange} />
                    <ul className="flex flex-wrap gap-2" aria-label="Empresas en el gráfico">
                      {selected.map((company, index) => (
                        <li
                          key={company.companyId}
                          className="inline-flex h-7 items-center gap-1.5 rounded-full border px-2.5 font-mono text-xs"
                        >
                          <span
                            className="size-1.5 rounded-full"
                            style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }}
                            aria-hidden="true"
                          />
                          {company.companyId}
                          <span className="text-muted-foreground tabular-nums">{company.score.toFixed(0)}</span>
                          {company.delta3m !== null ? (
                            <span className="text-muted-foreground tabular-nums">{formatPoints(company.delta3m)}</span>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </>
                )}
              </div>
            </div>

          </>
        )}

        <p className="text-xs leading-5 text-muted-foreground">{data.disclaimer}</p>
      </main>

      <HealthScoreChat
        companyId={selected.length === 1 ? selected[0].companyId : undefined}
        groupId={selected.length === 1 ? (selected[0].groupId ?? undefined) : undefined}
        asOf={data.asOfMonth}
        seedPrompt={seedPrompt}
        seedKey={seedKey}
        open={chatOpen}
        onOpenChange={setChatOpen}
      />
    </div>
  );
}
