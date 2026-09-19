"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { MousePointerClick, Search, TrendingDown, TrendingUp, X, type LucideIcon } from "lucide-react";
import { cn } from "cn";

import { Button } from "@/components/ui/button";
import { formatPoints } from "@/components/group/labels";
import { SERIES_COLORS } from "@/components/health-score-view";
import { QuickChart, type MonthRange } from "@/components/quick/quick-chart";
import { buildPrompt } from "@/components/quick/quick-explain";
import { QuickList } from "@/components/quick/quick-list";
import type { DashboardCompany, DashboardData } from "@/lib/data/types";

type Tab = "top" | "bottom" | "search";

const MAX_PICKED = 8;

const OPTIONS: { id: Tab; label: string; hint: string; icon: LucideIcon }[] = [
  { id: "top", label: "Mejores 5", hint: "Las cinco empresas con mayor índice hoy", icon: TrendingUp },
  { id: "bottom", label: "Peores 5", hint: "Las cinco con menor índice hoy", icon: TrendingDown },
  { id: "search", label: "Buscar", hint: "Por nombre o grupo", icon: Search },
];

/**
 * Quick mode body. One screen: the three options stay where they are and only ease upwards once you
 * choose; the chart (and, for search, the list) open underneath. Period explain uses the shared Pregunta popup.
 */
export function QuickAnalysis({
  data,
  onCompaniesChange,
  onRangeChange,
  onExplain,
}: {
  data: DashboardData;
  onCompaniesChange?: (companies: DashboardCompany[]) => void;
  onRangeChange?: (range: MonthRange | null) => void;
  onExplain?: (prompt: string) => void;
}) {
  const [tab, setTab] = useState<Tab | null>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [range, setRange] = useState<MonthRange | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const onCompaniesChangeRef = useRef(onCompaniesChange);
  onCompaniesChangeRef.current = onCompaniesChange;

  const idle = tab === null;

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

  useEffect(() => {
    onCompaniesChangeRef.current?.(selected);
  }, [selected]);

  function selectRange(next: MonthRange | null) {
    if (!next || selected.length === 0) {
      setRange(null);
      onRangeChange?.(null);
      return;
    }
    setRange(next);
    onRangeChange?.(next);
    onExplain?.(buildPrompt(selected, next));
  }

  function choose(next: Tab) {
    if (next === tab) return;
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
    <div className="mx-auto flex w-full max-w-6xl flex-1 flex-col px-4 pb-24 sm:px-6">
      {/* Eases the options up from the middle of the empty space once something is chosen. */}
      <div
        aria-hidden="true"
        className="shrink-0 transition-[height] duration-500 ease-out motion-reduce:transition-none"
        style={{ height: idle ? "clamp(1.5rem, 18vh, 9rem)" : "1.25rem" }}
      />

      <div
        className={cn(
          "grid transition-[grid-template-rows,opacity] duration-300 ease-out motion-reduce:transition-none",
          idle ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
        )}
      >
        <div className="overflow-hidden">
          <h1 className="pb-6 text-center text-balance text-2xl font-semibold tracking-tight sm:text-3xl">
            ¿Qué quieres ver?
          </h1>
        </div>
      </div>

      <div role="group" aria-label="Qué quieres ver" className="grid gap-3 sm:grid-cols-3">
        {OPTIONS.map((option) => {
          const active = tab === option.id;
          const isSearch = option.id === "search";
          const shell = cn(
            "group flex flex-col items-center justify-center rounded-xl border text-center outline-none transition-[padding,gap,background-color,border-color,color,box-shadow] duration-300 ease-out motion-reduce:transition-none",
            idle ? "gap-3 px-4 py-6" : "gap-1.5 px-3 py-3",
            active
              ? "border-foreground bg-foreground text-background shadow-sm"
              : "bg-card hover:border-foreground/40 hover:bg-muted/50",
          );
          const body = (
            <>
              <option.icon
                className={cn("size-5 shrink-0 transition-colors", active ? "" : "text-muted-foreground group-hover:text-foreground")}
                aria-hidden="true"
              />
              <span className="text-base font-medium">{option.label}</span>
              {isSearch ? (
                <input
                  ref={searchRef}
                  value={query}
                  onChange={(event) => {
                    setQuery(event.target.value);
                    choose("search");
                  }}
                  onFocus={() => choose("search")}
                  placeholder="Nombre o grupo…"
                  aria-label="Buscar empresa por nombre o grupo"
                  autoComplete="off"
                  className={cn(
                    "h-8 w-full rounded-md border bg-background px-2.5 text-center text-sm text-foreground outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50",
                    active && "border-transparent",
                  )}
                />
              ) : (
                <span
                  className={cn(
                    "block overflow-hidden text-xs leading-5 transition-[max-height,opacity] duration-300 ease-out motion-reduce:transition-none",
                    idle ? "max-h-10 opacity-100" : "max-h-0 opacity-0",
                    active ? "text-background/70" : "text-muted-foreground",
                  )}
                >
                  {option.hint}
                </span>
              )}
            </>
          );
          return isSearch ? (
            <div key={option.id} className={cn(shell, "cursor-text")} onClick={() => searchRef.current?.focus()}>
              {body}
            </div>
          ) : (
            <button key={option.id} type="button" aria-pressed={active} onClick={() => choose(option.id)} className={cn(shell, "focus-visible:ring-3 focus-visible:ring-ring/50")}>
              {body}
            </button>
          );
        })}
      </div>

      {tab !== null ? (
        <div key={tab} className="mt-5 flex flex-1 flex-col gap-4 animate-in fade-in-0 slide-in-from-bottom-1 duration-300 motion-reduce:animate-none">
          <div className={cn("grid gap-4", tab === "search" && "lg:grid-cols-[19rem_minmax(0,1fr)]")}>
            {tab === "search" ? (
              <QuickList companies={data.companies} query={query} picked={picked} max={MAX_PICKED} onToggle={toggle} />
            ) : null}

            <div className="min-w-0 space-y-2 max-lg:order-first">
              {/* Fixed-height line so the hint and the clear button never shift the chart. */}
              <div className="flex h-8 items-center justify-end">
                {range ? (
                  <Button type="button" variant="ghost" size="sm" onClick={() => selectRange(null)}>
                    <X data-icon="inline-start" />
                    Quitar selección
                  </Button>
                ) : selected.length > 0 ? (
                  <p className="flex items-center gap-1.5 text-xs text-muted-foreground">
                    <MousePointerClick className="size-3.5" aria-hidden="true" />
                    Arrastra sobre el gráfico para explicar un periodo
                  </p>
                ) : null}
              </div>

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
                          <span className="text-muted-foreground tabular-nums" title="Cambio en 3 meses">
                            {formatPoints(company.delta3m)}
                          </span>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </div>
          </div>
        </div>
      ) : null}

      <p className="mt-auto pt-8 text-xs leading-5 text-muted-foreground">{data.disclaimer}</p>
    </div>
  );
}
