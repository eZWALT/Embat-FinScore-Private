"use client";

import { useMemo, useState } from "react";
import { Search, X } from "lucide-react";
import {
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
import {
  ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import { Input } from "@/components/ui/input";
import { formatMonth } from "@/lib/format-month";
import type { DashboardCompany, DashboardData } from "@/lib/data/types";

const MAX_SERIES = 8;

const SERIES_COLORS = [
  "oklch(0.52 0.19 250)",
  "oklch(0.55 0.16 145)",
  "oklch(0.58 0.18 35)",
  "oklch(0.52 0.18 300)",
  "oklch(0.5 0.14 200)",
  "oklch(0.55 0.19 20)",
  "oklch(0.48 0.12 80)",
  "oklch(0.5 0.16 340)",
] as const;

function defaultSelection(companies: DashboardCompany[]): string[] {
  const picks: string[] = [];
  const improving = companies.find((company) => company.trajectory === "improving");
  const deteriorating = companies.find((company) => company.trajectory === "deteriorating");
  if (improving) picks.push(improving.companyId);
  if (deteriorating && deteriorating.companyId !== improving?.companyId) {
    picks.push(deteriorating.companyId);
  }
  for (const company of companies) {
    if (picks.length >= 3) break;
    if (!picks.includes(company.companyId)) picks.push(company.companyId);
  }
  return picks;
}

function buildChartRows(selected: DashboardCompany[]) {
  const months = [
    ...new Set(selected.flatMap((company) => (company.scoreHistory ?? []).map((point) => point.month))),
  ].sort();

  return months.map((month) => {
    const row: Record<string, string | number | null> = {
      month,
      label: formatMonth(month),
    };
    for (const company of selected) {
      row[company.companyId] =
        (company.scoreHistory ?? []).find((point) => point.month === month)?.score ?? null;
    }
    return row;
  });
}

export function HealthScoreView({ data }: { data: DashboardData }) {
  const [selectedIds, setSelectedIds] = useState<string[]>(() => defaultSelection(data.companies));
  const [query, setQuery] = useState("");

  const selected = useMemo(
    () =>
      selectedIds
        .map((id) => data.companies.find((company) => company.companyId === id))
        .filter((company): company is DashboardCompany => Boolean(company)),
    [data.companies, selectedIds],
  );

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (needle.length < 2) return [];
    return data.companies
      .filter((company) => {
        if (selectedIds.includes(company.companyId)) return false;
        const haystack = `${company.companyId} ${company.groupId ?? ""}`.toLowerCase();
        return haystack.includes(needle);
      })
      .slice(0, 8);
  }, [data.companies, query, selectedIds]);

  const chartConfig = useMemo(() => {
    const config: ChartConfig = {};
    selected.forEach((company, index) => {
      config[company.companyId] = {
        label: company.companyId,
        color: SERIES_COLORS[index % SERIES_COLORS.length],
      };
    });
    return config;
  }, [selected]);

  const chartRows = useMemo(() => buildChartRows(selected), [selected]);
  const atCap = selectedIds.length >= MAX_SERIES;

  function addCompany(companyId: string) {
    setSelectedIds((current) => {
      if (current.includes(companyId) || current.length >= MAX_SERIES) return current;
      return [...current, companyId];
    });
    setQuery("");
  }

  function removeCompany(companyId: string) {
    setSelectedIds((current) => current.filter((id) => id !== companyId));
  }

  return (
    <div className="space-y-4">
      <div>
        <p className="mb-2 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
          Health Score
        </p>
        <h1 className="text-balance text-2xl font-semibold tracking-tight sm:text-3xl">
          Evolución del índice 0–100
        </h1>
      </div>

      <Card>
        <CardHeader className="flex-row items-center justify-between gap-4 space-y-0">
          <CardTitle className="text-base">Empresas</CardTitle>
          <div className="flex items-center gap-3">
            <p className="text-xs text-muted-foreground">
              {selectedIds.length}/{MAX_SERIES}
            </p>
            {selectedIds.length > 0 ? (
              <Button type="button" variant="ghost" size="sm" onClick={() => setSelectedIds([])}>
                Limpiar
              </Button>
            ) : null}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="relative max-w-md">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 size-3.5 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Añadir empresa (COMP_0001…)"
              className="pl-8"
              aria-label="Añadir empresa"
              disabled={atCap}
            />
            {matches.length > 0 ? (
              <ul className="absolute z-20 mt-1 max-h-56 w-full overflow-y-auto rounded-lg border bg-popover p-1 shadow-md">
                {matches.map((company) => (
                  <li key={company.companyId}>
                    <button
                      type="button"
                      onClick={() => addCompany(company.companyId)}
                      className="flex w-full items-center justify-between gap-3 rounded-md px-2 py-1.5 text-left text-sm hover:bg-muted"
                    >
                      <span className="font-mono text-[13px]">{company.companyId}</span>
                      <span className="font-mono text-xs tabular-nums text-muted-foreground">
                        {company.score.toFixed(0)}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            ) : null}
          </div>

          {query.trim().length >= 2 && matches.length === 0 && !atCap ? (
            <p className="text-xs text-muted-foreground">Ninguna empresa coincide con “{query}”.</p>
          ) : null}

          {atCap ? (
            <p className="text-xs text-muted-foreground">
              Quita una empresa para añadir otra.
            </p>
          ) : null}

          {selected.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {selected.map((company, index) => (
                <Badge
                  key={company.companyId}
                  variant="outline"
                  className="h-7 gap-1.5 rounded-full border px-2.5 font-mono text-[12px]"
                >
                  <span
                    className="size-1.5 shrink-0 rounded-full"
                    style={{ background: SERIES_COLORS[index % SERIES_COLORS.length] }}
                    aria-hidden="true"
                  />
                  {company.companyId}
                  <button
                    type="button"
                    className="-mr-1 rounded-full p-0.5 hover:bg-muted"
                    onClick={() => removeCompany(company.companyId)}
                    aria-label={`Quitar ${company.companyId}`}
                  >
                    <X className="size-3" />
                  </button>
                </Badge>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">
              Añade una o más empresas para ver el gráfico.
            </p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Evolución del Health Score</CardTitle>
          <p className="text-sm text-muted-foreground">
            Escala fija 0–100. Los huecos son meses sin score.
          </p>
        </CardHeader>
        <CardContent>
          {selected.length === 0 ? (
            <div className="grid h-[min(70vh,560px)] place-items-center rounded-lg border border-dashed text-sm text-muted-foreground">
              Selecciona al menos una empresa para ver el gráfico.
            </div>
          ) : (
            <ClientOnly fallback={<div className="h-[min(70vh,560px)] w-full" />}>
              <ChartContainer config={chartConfig} className="h-[min(70vh,560px)] w-full aspect-auto">
                <LineChart data={chartRows} margin={{ top: 12, right: 16, left: -12, bottom: 0 }}>
                  <CartesianGrid vertical={false} strokeDasharray="3 3" />
                  <XAxis dataKey="label" tickLine={false} axisLine={false} minTickGap={28} />
                  <YAxis domain={[0, 100]} tickLine={false} axisLine={false} ticks={[0, 25, 50, 75, 100]} />
                  <ReferenceLine y={50} stroke="var(--border)" strokeDasharray="4 4" />
                  <ChartTooltip cursor={false} content={<ChartTooltipContent indicator="line" />} />
                  {selected.map((company, index) => (
                    <Line
                      key={company.companyId}
                      dataKey={company.companyId}
                      type="monotone"
                      stroke={SERIES_COLORS[index % SERIES_COLORS.length]}
                      strokeWidth={2.5}
                      dot={false}
                      connectNulls={false}
                      activeDot={{ r: 4 }}
                    />
                  ))}
                </LineChart>
              </ChartContainer>
            </ClientOnly>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
