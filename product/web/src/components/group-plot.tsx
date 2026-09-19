"use client";

import { useMemo, useState } from "react";

import { PlotCard, PlotChart, SignalLegend, controlRows, type PlotRow } from "@/components/plot-parts";
import { Segmented, type SegmentedOption } from "@/components/segmented";
import type { GroupOverview } from "@/lib/data/group-service";

type GroupView = "mean" | "own" | "groups";

const COPY: Record<GroupView, { title: string; hint: string; unit: string; center: string }> = {
  mean: {
    title: "Evolución de la media del grupo",
    hint: "Media del índice de las empresas puntuadas del grupo, mes a mes.",
    unit: "Media del grupo",
    center: "Normalidad",
  },
  own: {
    title: "Control intra-grupo",
    hint: "La media del grupo frente a su propia normalidad reciente. Fuera de la banda es un cambio inusual para este grupo.",
    unit: "Media del grupo",
    center: "Normalidad",
  },
  groups: {
    title: "Control inter-grupo",
    hint: "Cambio de la media en 3 meses frente a los límites de grupos de tamaño parecido: los grupos pequeños tienen límites más anchos.",
    unit: "Cambio en 3 meses",
    center: "Esperado",
  },
};

/** The main plot of the group view: the mean, and its two control charts when the group has 3 or more scored members. */
export function GroupPlot({ overview }: { overview: GroupOverview }) {
  const [view, setView] = useState<GroupView>("mean");

  const options: SegmentedOption<GroupView>[] = overview.limitsAvailable
    ? [
        { value: "mean", label: "Media" },
        { value: "own", label: "Intra-grupo" },
        { value: "groups", label: "Inter-grupo" },
      ]
    : [];
  const active: GroupView = options.some((option) => option.value === view) ? view : "mean";

  const series = active === "own" ? overview.control.own : active === "groups" ? overview.control.vsGroups : null;
  const isControl = active !== "mean";
  const empty = isControl && !series;

  const rows = useMemo<PlotRow[]>(
    () =>
      isControl
        ? series
          ? controlRows(series)
          : []
        : overview.months.map((month, i) => ({ month, value: overview.meanScores[i] ?? null })),
    [isControl, series, overview.months, overview.meanScores],
  );

  const copy = COPY[active];
  return (
    <PlotCard
      title={copy.title}
      asOfMonth={overview.asOfMonth}
      views={
        options.length > 0 ? (
          <Segmented options={options} value={active} onChange={setView} ariaLabel="Vista del gráfico" className="min-w-max" />
        ) : undefined
      }
      hint={copy.hint}
      notice={
        empty
          ? "Este gráfico de control necesita más historial (al menos 7 meses puntuados)."
          : !overview.limitsAvailable
            ? "Grupo pequeño: solo media, sin límites ni alerts de grupo (hacen falta al menos 3 empresas puntuadas)."
            : null
      }
    >
      <PlotChart rows={rows} control={isControl} valueLabel={copy.unit} centerLabel={copy.center} zeroLine={active === "groups"} hidden={empty} />
      {isControl && series ? <SignalLegend /> : null}
    </PlotCard>
  );
}
