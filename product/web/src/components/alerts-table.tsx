"use client";

import { useEffect, useMemo, useState } from "react";
import { ChevronLeft, ChevronRight, TrendingDown, TrendingUp } from "lucide-react";

import { cn } from "cn";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Segmented, type SegmentedOption } from "@/components/segmented";
import {
  companyLabel,
  formatEur,
  formatMonth,
  formatPoints,
  groupLabel,
  kindLabels,
  ownerLabels,
  reasonLabels,
  relabelEntities,
  severityBadgeVariant,
  severityLabels,
} from "@/components/group/labels";
import type { AlertRow, EntityAlerts } from "@/lib/data/alerts-service";
import type { AlertSeverity } from "@/lib/data/types";

type Filter = "all" | AlertSeverity;

const PAGE = 8;

export function SeverityBadge({ severity }: { severity: AlertSeverity }) {
  return <Badge variant={severityBadgeVariant[severity]}>{severityLabels[severity]}</Badge>;
}

type Scope = { company: string } | { group: string };

/**
 * Alerts of a company, or of a group and its members, as one table: when, what happened, why, and who should do what.
 * Filter by level; each row says how long it has been flagged.
 */
export function AlertsTable({ scope, onSelectCompany }: { scope: Scope; onSelectCompany?: (companyId: string) => void }) {
  const scopeKey = "company" in scope ? scope.company : scope.group;
  const isGroup = "group" in scope;
  const [result, setResult] = useState<{ key: string; data: EntityAlerts | null; error: string | null } | null>(null);
  const [filter, setFilter] = useState<Filter>("all");
  const [page, setPage] = useState(0);

  useEffect(() => {
    let cancelled = false;
    const query = new URLSearchParams({ [isGroup ? "group" : "company"]: scopeKey });
    fetch(`/api/alerts?${query}`)
      .then(async (res) => {
        const body = (await res.json()) as EntityAlerts & { error?: string };
        if (!res.ok) throw new Error(body.error || res.statusText);
        if (!cancelled) setResult({ key: scopeKey, data: body, error: null });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setResult({ key: scopeKey, data: null, error: error instanceof Error ? error.message : "No se pudieron cargar las alerts" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [scopeKey, isGroup]);

  const current = result?.key === scopeKey ? result : null;
  const alerts = useMemo(() => current?.data?.alerts ?? [], [current]);

  const counts = useMemo(() => {
    const tally: Record<Filter, number> = { all: alerts.length, act: 0, watch: 0, info: 0 };
    for (const alert of alerts) tally[alert.severity] += 1;
    return tally;
  }, [alerts]);

  const options: SegmentedOption<Filter>[] = [
    { value: "all", label: `Todas · ${counts.all}` },
    { value: "act", label: `${severityLabels.act} · ${counts.act}` },
    { value: "watch", label: `${severityLabels.watch} · ${counts.watch}` },
    { value: "info", label: `${severityLabels.info} · ${counts.info}` },
  ];

  const visible = filter === "all" ? alerts : alerts.filter((alert) => alert.severity === filter);
  const pages = Math.max(1, Math.ceil(visible.length / PAGE));
  const at = Math.min(page, pages - 1);
  const first = at * PAGE;

  return (
    <Card id="alerts" className="scroll-mt-20">
      <CardHeader className="gap-3">
        <CardTitle className="text-base">Alertas</CardTitle>
        {alerts.length > 0 ? (
          <div className="max-w-full overflow-x-auto">
            <Segmented
              options={options}
              value={filter}
              onChange={(next) => {
                setFilter(next);
                setPage(0);
              }}
              ariaLabel="Filtrar por nivel"
              className="min-w-max"
            />
          </div>
        ) : null}
      </CardHeader>
      <CardContent>
        {current?.error ? (
          <p className="text-sm text-destructive">{current.error}</p>
        ) : !current ? (
          <p className="text-sm text-muted-foreground">Cargando…</p>
        ) : alerts.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            {isGroup ? "Sin alertas en el grupo ni en sus empresas en este periodo." : "Sin alertas para esta empresa en este periodo."}
          </p>
        ) : visible.length === 0 ? (
          <p className="text-sm text-muted-foreground">No hay alertas de este nivel.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[52rem] border-collapse text-left text-sm">
              <thead className="text-xs text-muted-foreground">
                <tr className="border-b [&>th]:px-3 [&>th]:pb-2 [&>th]:font-medium [&>th]:whitespace-nowrap">
                  <th scope="col" className="w-32">Nivel</th>
                  <th scope="col" className="w-[26%]">Qué pasa</th>
                  <th scope="col" className="w-[30%]">Por qué</th>
                  <th scope="col">Qué hacer</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {visible.slice(first, first + PAGE).map((alert) => (
                  <AlertLine key={alert.alertId} alert={alert} showEntity={isGroup} onSelectCompany={onSelectCompany} />
                ))}
              </tbody>
            </table>
            {pages > 1 ? (
              <nav aria-label="Paginación de alertas" className="mt-3 flex items-center justify-between gap-3 border-t pt-3">
                <p className="text-xs text-muted-foreground">
                  {first + 1}–{Math.min(first + PAGE, visible.length)} de {visible.length}
                </p>
                <div className="flex items-center gap-1">
                  <Button type="button" variant="outline" size="icon-sm" aria-label="Página anterior" disabled={at === 0} onClick={() => setPage(at - 1)}>
                    <ChevronLeft />
                  </Button>
                  <span className="min-w-14 text-center font-mono text-xs tabular-nums text-muted-foreground">
                    {at + 1} / {pages}
                  </span>
                  <Button type="button" variant="outline" size="icon-sm" aria-label="Página siguiente" disabled={at === pages - 1} onClick={() => setPage(at + 1)}>
                    <ChevronRight />
                  </Button>
                </div>
              </nav>
            ) : null}
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function AlertLine({
  alert,
  showEntity,
  onSelectCompany,
}: {
  alert: AlertRow;
  showEntity: boolean;
  onSelectCompany?: (companyId: string) => void;
}) {
  const risk = alert.direction === "risk";
  const DirectionIcon = risk ? TrendingDown : TrendingUp;
  // The summary already says what the top-customer reason says.
  const reasons = alert.reasons.filter((reason) => reason.item !== "top_customer").slice(0, 3);
  const months = alert.persistence.monthsFlagged;

  return (
    <tr className="align-top [&>td]:px-3 [&>td]:py-3">
      <td>
        <p className="font-mono text-xs tabular-nums text-muted-foreground">{formatMonth(alert.month)}</p>
        <div className="mt-1.5">
          <SeverityBadge severity={alert.severity} />
        </div>
        <p className={cn("mt-1.5 inline-flex items-center gap-1 text-xs", risk ? "text-destructive" : "text-emerald-700 dark:text-emerald-400")}>
          <DirectionIcon className="size-3.5" aria-hidden="true" />
          {risk ? "Riesgo" : "Oportunidad"}
        </p>
      </td>
      <td>
        <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{kindLabels[alert.kind]}</p>
        <p className="mt-0.5 font-medium leading-snug">{relabelEntities(alert.title)}</p>
        {showEntity ? (
          alert.entityType === "company" && onSelectCompany ? (
            <button
              type="button"
              onClick={() => onSelectCompany(alert.entityId)}
              className="mt-1 font-mono text-xs underline decoration-muted-foreground/40 underline-offset-2 outline-none hover:decoration-foreground focus-visible:ring-2 focus-visible:ring-ring"
            >
              {companyLabel(alert.entityId)}
            </button>
          ) : (
            <p className="mt-1 text-xs text-muted-foreground">
              {alert.entityType === "group" ? groupLabel(alert.entityId) : companyLabel(alert.entityId)}
            </p>
          )
        ) : null}
        {months > 0 ? (
          <p className="mt-1 text-xs text-muted-foreground">
            {months} {months === 1 ? "mes" : "meses"} señalada{alert.persistence.rule ? ` · ${alert.persistence.rule}` : ""}
          </p>
        ) : null}
      </td>
      <td>
        <p className="leading-snug">{relabelEntities(alert.summary)}</p>
        {reasons.length > 0 ? (
          <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
            {reasons.map((reason) => (
              <li key={reason.item} className="flex flex-wrap items-baseline gap-x-2">
                <span className="text-foreground">{reasonLabels[reason.item] ?? reason.label}</span>
                {reason.points !== 0 ? <span className="font-mono tabular-nums">{formatPoints(reason.points)} pts</span> : null}
                {reason.eur !== null ? <span className="font-mono tabular-nums">{formatEur(reason.eur)}</span> : null}
              </li>
            ))}
          </ul>
        ) : null}
      </td>
      <td>
        <p className="text-xs font-medium text-muted-foreground">{ownerLabels[alert.owner]}</p>
        <p className="mt-0.5 leading-snug">{relabelEntities(alert.action)}</p>
      </td>
    </tr>
  );
}
