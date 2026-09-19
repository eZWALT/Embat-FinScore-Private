"use client";

import { useState, type ReactNode } from "react";
import { MousePointerClick, X } from "lucide-react";
import { cn } from "cn";

import { Button } from "@/components/ui/button";
import { formatPoints } from "@/components/group/labels";
import { SERIES_COLORS } from "@/components/quick/series";
import { OfferStrip } from "@/components/offers/offer-guidance";
import { QuickChart, type MonthRange } from "@/components/quick/quick-chart";
import { buildPrompt } from "@/components/quick/quick-explain";
import type { DashboardCompany } from "@/lib/data/types";

/**
 * Score lines for the chosen companies, with the list slot on the left when there is one. Rápido and the deep
 * comparison both render this, so a search result opens the same chart in the same place in either mode.
 * Dragging across the chart selects a period; the parent gets the range and the prompt that explains it.
 */
export function ComparePanel({
  selected,
  list,
  emptyText = "Marca una o más empresas de la lista para ver su evolución.",
  onRangeChange,
  onExplain,
}: {
  selected: DashboardCompany[];
  list?: ReactNode;
  emptyText?: string;
  onRangeChange?: (range: MonthRange | null) => void;
  onExplain?: (prompt: string) => void;
}) {
  // A period belongs to the companies it was drawn for: pick another set and it goes away.
  const selectionKey = selected.map((company) => company.companyId).join("|");
  const [drawn, setDrawn] = useState<{ key: string; range: MonthRange } | null>(null);
  const range = drawn?.key === selectionKey ? drawn.range : null;

  function selectRange(next: MonthRange | null) {
    if (!next || selected.length === 0) {
      setDrawn(null);
      onRangeChange?.(null);
      return;
    }
    setDrawn({ key: selectionKey, range: next });
    onRangeChange?.(next);
    onExplain?.(buildPrompt(selected, next));
  }

  return (
    <div className={cn("grid gap-4", list && "lg:grid-cols-[19rem_minmax(0,1fr)]")}>
      {list}

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
            {emptyText}
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
            <OfferStrip companies={selected} />
          </>
        )}
      </div>
    </div>
  );
}
