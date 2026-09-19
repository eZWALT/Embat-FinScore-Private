"use client";

import { useMemo } from "react";
import { ChevronRight, Handshake } from "lucide-react";

import { cn } from "cn";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { Posture, Tone } from "@/config/offers";
import { guidanceFor } from "@/lib/offers";
import type { DashboardCompany } from "@/lib/data/types";

const TONE: Record<Tone, string> = {
  positive: "bg-emerald-500/15 text-emerald-700 dark:text-emerald-400",
  neutral: "bg-muted text-foreground",
  caution: "bg-amber-500/15 text-amber-700 dark:text-amber-400",
  negative: "bg-destructive/10 text-destructive",
};

export function PostureBadge({ posture, className }: { posture: Posture; className?: string }) {
  return (
    <span className={cn("inline-flex h-5 items-center rounded-full px-2 text-[11px] font-semibold", TONE[posture.tone], className)}>
      {posture.label}
    </span>
  );
}

/** Deep view: the posture, how to negotiate, and each product that fits with the reasons it fits. */
export function OfferGuidanceCard({ company }: { company: DashboardCompany }) {
  const guidance = useMemo(() => guidanceFor(company), [company]);
  const { posture, notes, suggestions } = guidance;

  return (
    <Card id="ofertas" className="scroll-mt-20">
      <CardHeader className="gap-2">
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
          <CardTitle className="flex items-center gap-2 text-base">
            <Handshake className="size-4 text-muted-foreground" aria-hidden="true" />
            Sugerencia comercial
          </CardTitle>
          <PostureBadge posture={posture} className="h-6 px-2.5 text-xs" />
        </div>
        <p className="text-sm font-medium">{posture.headline}</p>
        <p className="text-sm text-muted-foreground">{posture.negotiation}</p>
        {notes.length > 0 ? (
          <ul className="space-y-0.5 text-xs text-muted-foreground">
            {notes.map((note) => (
              <li key={note}>· {note}</li>
            ))}
          </ul>
        ) : null}
      </CardHeader>
      <CardContent>
        {suggestions.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {posture.id === "protect" ? "Ningún producto nuevo en esta situación." : "Ningún producto concreto encaja ahora con estos números."}
          </p>
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {suggestions.map(({ product, because }) => (
              <li key={product.id} className="rounded-xl border bg-background px-4 py-3">
                <p className="text-sm font-medium">{product.name}</p>
                <p className="mt-0.5 text-sm text-muted-foreground">{product.pitch}</p>
                <ul className="mt-2 space-y-0.5 text-xs text-muted-foreground">
                  {because.map((line) => (
                    <li key={line}>· {line}</li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        )}
        <p className="mt-4 text-[11px] leading-5 text-muted-foreground">
          Orientación comercial a partir del índice y sus categorías; no es una decisión de crédito. Umbrales y productos en{" "}
          <code className="font-mono">config/offers.ts</code>.
        </p>
      </CardContent>
    </Card>
  );
}

/** Quick view: closed by default, one small line per company on the chart. */
export function OfferStrip({ companies }: { companies: DashboardCompany[] }) {
  const rows = useMemo(() => companies.map((company) => ({ company, guidance: guidanceFor(company) })), [companies]);
  if (rows.length === 0) return null;

  return (
    <details className="group rounded-xl border bg-card/50 text-sm">
      <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2 text-xs font-medium text-muted-foreground outline-none hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 [&::-webkit-details-marker]:hidden">
        <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" aria-hidden="true" />
        Sugerencias comerciales
      </summary>
      <ul className="grid gap-2 border-t p-3 sm:grid-cols-2 lg:grid-cols-3">
        {rows.map(({ company, guidance }) => (
          <li key={company.companyId} className="min-w-0 rounded-lg border bg-background px-3 py-2">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-mono text-xs">{company.companyId}</span>
              <PostureBadge posture={guidance.posture} />
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {guidance.suggestions.length > 0
                ? guidance.suggestions
                    .slice(0, 2)
                    .map((suggestion) => suggestion.product.name)
                    .join(" · ")
                : guidance.posture.headline}
            </p>
          </li>
        ))}
      </ul>
    </details>
  );
}
