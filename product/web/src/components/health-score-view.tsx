"use client";

import { useEffect, useMemo, useState } from "react";
import { Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { HealthIndexHelp } from "@/components/health-index-help";
import { Input } from "@/components/ui/input";
import { ComparePanel } from "@/components/quick/compare-panel";
import { QuickList } from "@/components/quick/quick-list";
import type { DashboardCompany, DashboardData } from "@/lib/data/types";

const MAX_SERIES = 8;

function defaultSelection(companies: DashboardCompany[], first?: string): string[] {
  const picks: string[] = [];
  if (first && companies.some((company) => company.companyId === first)) picks.push(first);
  const improving = companies.find((company) => company.trajectory === "improving");
  const deteriorating = companies.find((company) => company.trajectory === "deteriorating");
  if (improving && !picks.includes(improving.companyId)) picks.push(improving.companyId);
  if (deteriorating && !picks.includes(deteriorating.companyId)) picks.push(deteriorating.companyId);
  for (const company of companies) {
    if (picks.length >= 3) break;
    if (!picks.includes(company.companyId)) picks.push(company.companyId);
  }
  return picks;
}

/**
 * Deep comparison of several companies. The search, its sorting and the chart are the Rápido ones
 * (`QuickList` + `ComparePanel`): picking a result draws it on the chart beside the list, without leaving the page.
 */
export function HealthScoreView({
  data,
  initialCompanyId,
  onSelectionChange,
  onExplain,
}: {
  data: DashboardData;
  initialCompanyId?: string;
  onSelectionChange?: (companies: DashboardCompany[]) => void;
  onExplain?: (prompt: string) => void;
}) {
  const [picked, setPicked] = useState<string[]>(() => defaultSelection(data.companies, initialCompanyId));
  const [query, setQuery] = useState("");

  const selected = useMemo(
    () =>
      picked
        .map((id) => data.companies.find((company) => company.companyId === id))
        .filter((company): company is DashboardCompany => Boolean(company)),
    [data.companies, picked],
  );

  useEffect(() => {
    onSelectionChange?.(selected);
  }, [selected, onSelectionChange]);

  function toggle(companyId: string) {
    setPicked((current) =>
      current.includes(companyId)
        ? current.filter((id) => id !== companyId)
        : current.length >= MAX_SERIES
          ? current
          : [...current, companyId],
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <p className="mb-2 flex items-center gap-1.5 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
          Índice de salud
          <HealthIndexHelp />
        </p>
        <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl">Comparar empresas</h1>
        <p className="mt-2 text-sm text-muted-foreground">
          Busca por nombre o grupo y marca hasta {MAX_SERIES} empresas para ver su evolución juntas.
        </p>
      </div>

      <ComparePanel
        selected={selected}
        onExplain={onExplain}
        list={
          <div className="flex min-h-0 flex-col gap-3">
            <div className="flex items-center gap-2">
              <div className="relative min-w-0 flex-1">
                <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
                <Input
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Nombre o grupo…"
                  aria-label="Buscar empresa por nombre o grupo"
                  autoComplete="off"
                  className="pl-8"
                />
              </div>
              {picked.length > 0 ? (
                <Button type="button" variant="ghost" size="sm" onClick={() => setPicked([])}>
                  Limpiar
                </Button>
              ) : null}
            </div>
            <QuickList companies={data.companies} query={query} picked={picked} max={MAX_SERIES} onToggle={toggle} />
          </div>
        }
      />
    </div>
  );
}
