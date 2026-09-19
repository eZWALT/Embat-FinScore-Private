"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Search, TrendingDown, TrendingUp, type LucideIcon } from "lucide-react";
import { cn } from "cn";

import { ProductCredit } from "@/components/product-credit";
import { ComparePanel } from "@/components/quick/compare-panel";
import { type MonthRange } from "@/components/quick/quick-chart";
import { QuickList } from "@/components/quick/quick-list";
import type { DashboardCompany, DashboardData } from "@/lib/data/types";
import { HALF_LIFE_MONTHS, recencyWeightedMean } from "@/lib/quick-ranking";

type Tab = "top" | "bottom" | "search";

const MAX_PICKED = 8;

const OPTIONS: { id: Tab; label: string; hint: string; icon: LucideIcon }[] = [
  { id: "search", label: "Buscar", hint: "Por nombre o grupo", icon: Search },
  { id: "top", label: "Mejores 5", hint: "Mayor media del índice, con más peso a los meses recientes", icon: TrendingUp },
  { id: "bottom", label: "Peores 5", hint: "Menor media del índice, con más peso a los meses recientes", icon: TrendingDown },
];

/**
 * Quick mode body. One screen: Buscar leads (wide, left) with Mejores 5 / Peores 5 as smaller cards beside it. They
 * stay where they are and only ease upwards once you choose; the chart (and, for search, the list) open underneath.
 * Period explain uses the shared Pregunta popup.
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
  const [tab, setTab] = useState<Tab | null>("search");
  const [picked, setPicked] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const searchRef = useRef<HTMLInputElement>(null);
  const onCompaniesChangeRef = useRef(onCompaniesChange);
  onCompaniesChangeRef.current = onCompaniesChange;

  const idle = tab === null;

  // Best and worst are ranked by the recency-weighted mean of the whole history, not by last month alone.
  const weighted = useMemo(() => {
    const means = new Map<string, number>();
    for (const company of data.companies) {
      const mean = recencyWeightedMean(company.scoreHistory, data.asOfMonth);
      if (mean !== null) means.set(company.companyId, mean);
    }
    return means;
  }, [data.companies, data.asOfMonth]);

  const ranked = useMemo(
    () =>
      data.companies
        .filter((company) => weighted.has(company.companyId))
        .sort((a, b) => weighted.get(b.companyId)! - weighted.get(a.companyId)! || a.companyId.localeCompare(b.companyId)),
    [data.companies, weighted],
  );

  const selected: DashboardCompany[] = useMemo(() => {
    if (tab === "top") return ranked.slice(0, 5);
    if (tab === "bottom") return ranked.slice(-5).reverse();
    if (tab === "search") {
      return picked
        .map((id) => data.companies.find((company) => company.companyId === id))
        .filter((company): company is DashboardCompany => Boolean(company));
    }
    return [];
  }, [tab, ranked, picked, data.companies]);

  useEffect(() => {
    onCompaniesChangeRef.current?.(selected);
  }, [selected]);

  // Opens as if Buscar had been clicked: the search box is ready to type in.
  useEffect(() => {
    searchRef.current?.focus();
  }, []);

  function choose(next: Tab) {
    if (next === tab) return;
    setTab(next);
    onRangeChange?.(null);
  }

  function toggle(companyId: string) {
    onRangeChange?.(null);
    setPicked((current) =>
      current.includes(companyId)
        ? current.filter((id) => id !== companyId)
        : current.length >= MAX_PICKED
          ? current
          : [...current, companyId],
    );
  }

  return (
    <div className="mx-auto flex w-full max-w-[96rem] flex-1 flex-col px-4 pb-16 sm:px-6 lg:px-10">
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

      <div role="group" aria-label="Qué quieres ver" className="grid gap-3 sm:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)_minmax(0,1fr)]">
        {OPTIONS.map((option) => {
          const active = tab === option.id;
          const isSearch = option.id === "search";
          const shell = cn(
            "group flex flex-col items-center justify-center rounded-xl border text-center outline-none transition-[padding,gap,background-color,border-color,color,box-shadow] duration-300 ease-out motion-reduce:transition-none",
            isSearch
              ? idle
                ? "gap-4 px-6 py-10"
                : "gap-1.5 px-4 py-3"
              : idle
                ? "gap-2 px-4 py-6"
                : "gap-1 px-3 py-3",
            active
              ? "border-foreground bg-foreground text-background shadow-sm"
              : isSearch
                ? "border-foreground/30 bg-card shadow-sm hover:border-foreground/60"
                : "bg-card hover:border-foreground/40 hover:bg-muted/50",
          );
          const body = (
            <>
              <option.icon
                className={cn(
                  "shrink-0 transition-[color,width,height]",
                  isSearch && idle ? "size-7" : "size-5",
                  active ? "" : "text-muted-foreground group-hover:text-foreground",
                )}
                aria-hidden="true"
              />
              <span className={cn("font-medium", isSearch ? "text-lg" : "text-sm")}>{option.label}</span>
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
                    "w-full rounded-md border bg-background px-3 text-center text-foreground outline-none transition-[height,font-size,colors] placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50",
                    idle ? "h-12 max-w-xl text-base" : "h-9 text-sm",
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
          <ComparePanel
            selected={selected}
            list={
              tab === "search" ? (
                <QuickList companies={data.companies} query={query} picked={picked} max={MAX_PICKED} onToggle={toggle} />
              ) : undefined
            }
            onRangeChange={onRangeChange}
            onExplain={onExplain}
          />
        </div>
      ) : null}

      <ProductCredit className="mt-auto pt-8" />
    </div>
  );
}
