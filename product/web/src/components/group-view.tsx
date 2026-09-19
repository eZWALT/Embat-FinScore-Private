"use client";

import { useEffect, useState } from "react";
import { ArrowDownRight, ArrowUpRight, Minus } from "lucide-react";

import { AlertsTable } from "@/components/alerts-table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { GroupPlot } from "@/components/group-plot";
import { HeatmapLegend, ScoreHeatmap } from "@/components/group/score-heatmap";
import { companyLabel, formatPoints, groupLabel, scoreColor } from "@/components/group/labels";
import type { GroupOverview } from "@/lib/data/group-service";

function lastChange(values: (number | null)[]): number | null {
  const present = values.filter((value): value is number => value !== null);
  return present.length >= 2 ? present[present.length - 1] - present[present.length - 2] : null;
}

/**
 * The deep view of a group, laid out like the company view: indicators, main plot, alerts. Below them, the members as a
 * heatmap and a table. Picking a member opens that company in place.
 */
export function GroupView({
  groupId,
  highlightCompanyId,
  onSelectCompany,
}: {
  groupId: string;
  highlightCompanyId: string | null;
  onSelectCompany: (companyId: string) => void;
}) {
  const [result, setResult] = useState<{ groupId: string; overview: GroupOverview | null; error: string | null } | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetch(`/api/group?id=${encodeURIComponent(groupId)}`)
      .then(async (res) => {
        const body = (await res.json()) as GroupOverview & { error?: string };
        if (!res.ok) throw new Error(body.error || res.statusText);
        if (!cancelled) setResult({ groupId, overview: body, error: null });
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setResult({ groupId, overview: null, error: error instanceof Error ? error.message : "No se pudo cargar el grupo" });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [groupId]);

  const current = result?.groupId === groupId ? result : null;
  const overview = current?.overview ?? null;

  const header = (
    <section id="resumen" className="flex scroll-mt-20 flex-col gap-1">
      <h1 className="text-2xl font-semibold tracking-tight" title={groupId}>
        {groupLabel(groupId)}
      </h1>
      <p className="text-sm text-muted-foreground">
        {overview
          ? `${overview.nCompanies} ${overview.nCompanies === 1 ? "empresa" : "empresas"}${
              overview.members.length !== overview.nCompanies ? ` · ${overview.members.length} con puntuación` : ""
            }`
          : "Cargando el grupo…"}
      </p>
    </section>
  );

  if (!overview) {
    return (
      <>
        {header}
        <p className={current?.error ? "mt-6 text-sm text-destructive" : "mt-6 text-sm text-muted-foreground"}>
          {current?.error ?? "Cargando…"}
        </p>
      </>
    );
  }

  const change = lastChange(overview.meanScores);
  const ChangeIcon = change === null || change === 0 ? Minus : change > 0 ? ArrowUpRight : ArrowDownRight;
  const min = overview.minScore;

  return (
    <>
      {header}

      <section className="mt-6 grid gap-4 sm:grid-cols-3">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Media del grupo</CardTitle>
          </CardHeader>
          <CardContent className="flex items-center justify-between gap-3">
            <div className="flex items-baseline gap-2">
              {overview.meanScore !== null ? (
                <span className="size-2.5 self-center rounded-full" style={{ background: scoreColor(overview.meanScore) }} aria-hidden="true" />
              ) : null}
              <span className="font-mono text-4xl font-medium tracking-tight tabular-nums">
                {overview.meanScore === null ? "—" : overview.meanScore.toFixed(0)}
              </span>
              <span className="text-sm text-muted-foreground">/ 100</span>
            </div>
            {change === null ? (
              <span className="text-sm text-muted-foreground">—</span>
            ) : (
              <span className="inline-flex items-center gap-1 font-mono text-sm tabular-nums text-muted-foreground" title="Cambio mensual">
                <ChangeIcon className="size-4" />
                {formatPoints(change)} pts
              </span>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Empresa más débil</CardTitle>
          </CardHeader>
          <CardContent className="flex h-11 items-center justify-between gap-3">
            <span className="font-mono text-2xl font-medium tabular-nums">{min === null ? "—" : min.toFixed(0)}</span>
            {overview.minCompanyId ? (
              <button
                type="button"
                onClick={() => onSelectCompany(overview.minCompanyId!)}
                className="font-mono text-xs text-muted-foreground underline decoration-muted-foreground/40 underline-offset-2 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring"
              >
                {companyLabel(overview.minCompanyId)}
              </button>
            ) : null}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-xs font-medium text-muted-foreground">Empresas</CardTitle>
          </CardHeader>
          <CardContent className="flex h-11 items-center">
            <span className="font-mono text-2xl font-medium tabular-nums">{overview.nCompanies}</span>
          </CardContent>
        </Card>
      </section>

      <div className="mt-4 space-y-4">
        <GroupPlot overview={overview} />
        <AlertsTable scope={{ group: groupId }} onSelectCompany={onSelectCompany} />

        <Card>
          <CardHeader>
            <div className="flex flex-wrap items-center justify-between gap-3">
              <CardTitle className="text-base">Empresas del grupo</CardTitle>
              <HeatmapLegend />
            </div>
          </CardHeader>
          <CardContent>
            <ScoreHeatmap
              months={overview.months}
              meanScores={overview.meanScores}
              members={overview.members}
              selectedCompanyId={highlightCompanyId}
              onSelectCompany={onSelectCompany}
            />
          </CardContent>
        </Card>
      </div>
    </>
  );
}
