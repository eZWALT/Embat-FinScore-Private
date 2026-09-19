"use client";

import { useMemo, useState } from "react";
import { ArrowDown, ArrowUp, Check, Search } from "lucide-react";
import { cn } from "cn";

import { Button } from "@/components/ui/button";
import { formatPoints, scoreColor } from "@/components/group/labels";
import type { DashboardCompany } from "@/lib/data/types";

type SortKey = "score" | "delta1m" | "delta3m" | "id";

const SORTS: { key: SortKey; label: string }[] = [
  { key: "score", label: "Índice" },
  { key: "delta1m", label: "Δ 1 m" },
  { key: "delta3m", label: "Δ 3 m" },
  { key: "id", label: "Nombre" },
];

const PAGE = 100;

function sortValue(company: DashboardCompany, key: SortKey) {
  if (key === "score") return company.score;
  if (key === "delta1m") return company.delta1m;
  if (key === "delta3m") return company.delta3m;
  return company.companyId;
}

/** Searchable, sortable list of every company. Picking rows draws them on the chart. */
export function QuickList({
  companies,
  picked,
  max,
  onToggle,
}: {
  companies: DashboardCompany[];
  picked: string[];
  max: number;
  onToggle: (companyId: string) => void;
}) {
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ key: SortKey; dir: "asc" | "desc" }>({ key: "score", dir: "desc" });
  const [shown, setShown] = useState(PAGE);

  const rows = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const filtered = needle
      ? companies.filter((company) => `${company.companyId} ${company.groupId ?? ""}`.toLowerCase().includes(needle))
      : companies.slice();
    const sign = sort.dir === "asc" ? 1 : -1;
    return filtered.sort((a, b) => {
      const x = sortValue(a, sort.key);
      const y = sortValue(b, sort.key);
      // Companies without the value (no 3-month change yet) always go last.
      if (x === null && y === null) return 0;
      if (x === null) return 1;
      if (y === null) return -1;
      if (typeof x === "string" && typeof y === "string") return sign * x.localeCompare(y);
      return sign * ((x as number) - (y as number));
    });
  }, [companies, query, sort]);

  const atCap = picked.length >= max;

  function chooseSort(key: SortKey) {
    setShown(PAGE);
    setSort((current) =>
      current.key === key
        ? { key, dir: current.dir === "asc" ? "desc" : "asc" }
        : { key, dir: key === "id" ? "asc" : "desc" },
    );
  }

  return (
    <div className="flex min-h-0 flex-col gap-3">
      <div className="relative">
        <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground" aria-hidden="true" />
        <input
          value={query}
          onChange={(event) => {
            setQuery(event.target.value);
            setShown(PAGE);
          }}
          placeholder="Buscar por nombre o grupo"
          aria-label="Buscar empresa por nombre o grupo"
          autoComplete="off"
          className="h-9 w-full rounded-lg border border-input bg-transparent pr-2.5 pl-8 text-sm outline-none transition-colors placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30"
        />
      </div>

      <div className="flex flex-wrap items-center gap-1" role="group" aria-label="Ordenar por">
        {SORTS.map((option) => {
          const active = sort.key === option.key;
          return (
            <Button
              key={option.key}
              type="button"
              size="sm"
              variant={active ? "secondary" : "ghost"}
              className="h-7 gap-1 px-2 text-xs"
              aria-pressed={active}
              onClick={() => chooseSort(option.key)}
            >
              {option.label}
              {active ? sort.dir === "asc" ? <ArrowUp className="size-3" /> : <ArrowDown className="size-3" /> : null}
            </Button>
          );
        })}
      </div>

      <div className="flex items-center justify-between px-2.5 text-[11px] font-medium text-muted-foreground" aria-hidden="true">
        <span className="pl-6">Empresa</span>
        <span className="flex gap-4">
          <span>Δ 3 m</span>
          <span className="w-10 text-right">Índice</span>
        </span>
      </div>
      <ul
        aria-label="Empresas"
        className="-mt-1.5 max-h-[min(46vh,420px)] divide-y overflow-y-auto rounded-lg border"
      >
        {rows.slice(0, shown).map((company) => {
          const isPicked = picked.includes(company.companyId);
          const disabled = !isPicked && atCap;
          return (
            <li key={company.companyId}>
              <button
                type="button"
                disabled={disabled}
                aria-pressed={isPicked}
                onClick={() => onToggle(company.companyId)}
                className={cn(
                  "flex w-full items-center gap-2.5 px-2.5 py-2 text-left text-sm transition-colors hover:bg-muted/60 disabled:cursor-not-allowed disabled:opacity-40",
                  isPicked && "bg-muted",
                )}
              >
                <span
                  className={cn(
                    "grid size-4 shrink-0 place-items-center rounded border",
                    isPicked ? "border-foreground bg-foreground text-background" : "border-input",
                  )}
                  aria-hidden="true"
                >
                  {isPicked ? <Check className="size-3" /> : null}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate font-mono text-[13px]">{company.companyId}</span>
                  <span className="block truncate text-xs text-muted-foreground">{company.groupId ?? "Sin grupo"}</span>
                </span>
                <span className="shrink-0 text-right font-mono text-xs tabular-nums text-muted-foreground">
                  {company.delta3m === null ? "—" : formatPoints(company.delta3m)}
                </span>
                <span className="inline-flex w-10 shrink-0 items-center justify-end gap-1.5 font-mono text-sm tabular-nums">
                  <span className="size-2 rounded-full" style={{ background: scoreColor(company.score) }} aria-hidden="true" />
                  {company.score.toFixed(0)}
                </span>
              </button>
            </li>
          );
        })}
        {rows.length === 0 ? (
          <li className="px-3 py-6 text-center text-sm text-muted-foreground">Ninguna empresa coincide.</li>
        ) : null}
        {rows.length > shown ? (
          <li className="p-1.5">
            <Button type="button" variant="ghost" size="sm" className="w-full" onClick={() => setShown((count) => count + PAGE)}>
              Mostrar más ({rows.length - shown})
            </Button>
          </li>
        ) : null}
      </ul>
      <p className="text-xs text-muted-foreground">
        {rows.length} empresas · marca hasta {max} para compararlas.
      </p>
    </div>
  );
}
