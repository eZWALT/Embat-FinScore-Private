"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronRight, Handshake } from "lucide-react";

import { cn } from "cn";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import { companyLabel } from "@/components/group/labels";
import { MAGNITUDES, type Posture, type Tone } from "@/config/offers";
import { guidanceFor } from "@/lib/offers";
import type { CompanySize } from "@/lib/data/size";
import type { DashboardCompany } from "@/lib/data/types";
import { formatMoney } from "@/lib/display";

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

/** The size bands the suggestions are graded by, by monthly inflow. The company's own band is filled in when known. */
function SizeBands({ current, currency }: { current?: string; currency: string | null }) {
  return (
    <ul className="flex flex-wrap gap-1.5" aria-label="Tamaños de empresa (ingresos mensuales)">
      {MAGNITUDES.map((band, index) => {
        const next = MAGNITUDES[index + 1];
        const range = next
          ? index === 0
            ? `< ${formatMoney(next.minMonthlyInflow, currency)}`
            : `${formatMoney(band.minMonthlyInflow, currency)} – ${formatMoney(next.minMonthlyInflow, currency)}`
          : `> ${formatMoney(band.minMonthlyInflow, currency)}`;
        const active = band.id === current;
        return (
          <li
            key={band.id}
            className={cn(
              "inline-flex h-6 items-center gap-1.5 rounded-full border px-2 text-[11px]",
              active ? "border-foreground bg-foreground text-background" : "text-muted-foreground",
            )}
            aria-current={active ? "true" : undefined}
          >
            <span className="font-medium">{band.label}</span>
            <span className={cn("font-mono tabular-nums", active ? "text-background/70" : "text-muted-foreground/70")}>{range}</span>
          </li>
        );
      })}
    </ul>
  );
}

/** Deep view: the posture and each product that fits. Hover a product for the numbers that make it fit. */
export function OfferGuidanceCard({ company }: { company: DashboardCompany }) {
  const [loaded, setLoaded] = useState<{ key: string; size: CompanySize | null } | null>(null);
  const key = `${company.companyId}|${company.latestMonth}`;
  const settled = loaded?.key === key;
  const size = settled ? loaded.size : null;

  // The company's size comes from its bank records. Without them the card still lists the bands and says why it cannot place the company.
  useEffect(() => {
    let cancelled = false;
    fetch(`/api/size?company=${encodeURIComponent(company.companyId)}&month=${encodeURIComponent(company.latestMonth)}`)
      .then(async (res) => {
        const body = res.ok ? ((await res.json()) as CompanySize) : null;
        if (!cancelled) setLoaded({ key, size: body });
      })
      .catch(() => {
        if (!cancelled) setLoaded({ key, size: null });
      });
    return () => {
      cancelled = true;
    };
  }, [key, company.companyId, company.latestMonth]);

  const guidance = useMemo(() => guidanceFor(company, size), [company, size]);
  const { posture, suggestions, magnitude, tooSmallFor } = guidance;

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
        <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 pt-1">
          <div className="min-w-0 space-y-1.5">
            <p
              className="text-xs text-muted-foreground"
              title={size ? `Media mensual de los últimos 3 meses (${size.from} → ${size.to}), según los movimientos bancarios.` : undefined}
            >
              {size ? (
                magnitude ? (
                  <>
                    Tamaño: <span className="font-medium text-foreground">{magnitude.label}</span> · ingresos de ≈{" "}
                    {formatMoney(size.monthlyInflow, company.currency)} al mes
                  </>
                ) : (
                  "Sin ingresos recientes: no se puede estimar el tamaño."
                )
              ) : settled ? (
                "No se puede situar a esta empresa por tamaño: los movimientos bancarios no están disponibles en la base de datos."
              ) : (
                "Calculando el tamaño…"
              )}
            </p>
            <SizeBands current={magnitude?.id} currency={company.currency} />
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* The list waits for the size: it decides which products fit and their amounts, and showing it first makes it jump. */}
        {!settled ? (
          <ul aria-busy="true" aria-label="Calculando sugerencias" className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {[0, 1, 2].map((slot) => (
              <li key={slot}>
                <Skeleton className="h-[7.25rem] rounded-xl" />
              </li>
            ))}
          </ul>
        ) : suggestions.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {posture.id === "protect"
              ? "Ningún producto nuevo en esta situación."
              : tooSmallFor > 0
                ? "Los productos que encajan con su situación son demasiado grandes para el tamaño de esta empresa."
                : "Ningún producto concreto encaja ahora con estos números."}
          </p>
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
            {suggestions.map(({ product, because, amount }) => (
              <li
                key={product.id}
                className="rounded-xl border bg-background px-4 py-3"
                title={[...because, ...(amount ? [`Importe: ${amount.basis}`] : [])].join(" · ")}
              >
                <p className="text-sm font-medium">{product.name}</p>
                <p className="mt-0.5 text-sm text-muted-foreground">{product.pitch}</p>
                {amount ? (
                  <p className="mt-2 flex flex-wrap items-baseline gap-x-2 text-sm">
                    <span className="font-mono font-medium tabular-nums">
                      ≈ {formatMoney(amount.low, company.currency)}
                      {amount.high > amount.low ? ` – ${formatMoney(amount.high, company.currency)}` : ""}
                    </span>
                    <span className="text-xs text-muted-foreground">{amount.basis}</span>
                  </p>
                ) : null}
              </li>
            ))}
          </ul>
        )}
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
              <span className="truncate text-xs" title={company.companyId}>{companyLabel(company.companyId)}</span>
              <PostureBadge posture={guidance.posture} />
            </div>
            {guidance.suggestions.length > 0 ? (
              <p className="mt-1 text-xs text-muted-foreground">
                {guidance.suggestions
                  .slice(0, 2)
                  .map((suggestion) => suggestion.product.name)
                  .join(" · ")}
              </p>
            ) : null}
          </li>
        ))}
      </ul>
    </details>
  );
}
