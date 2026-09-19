"use client";

import { useRouter } from "next/navigation";
import Link from "next/link";
import { useState, useTransition } from "react";
import { ArrowLeft, Building2, Eye, Layers } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { EntityCombobox } from "@/components/entity-combobox";
import { Separator } from "@/components/ui/separator";
import { ProductCredit } from "@/components/product-credit";
import { ProductNav } from "@/components/product-nav";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { AlertList } from "@/components/group/alert-list";
import { CompanyPanel } from "@/components/group/company-panel";
import { formatMonth } from "@/components/group/labels";
import { MemberTable } from "@/components/group/member-table";
import { HeatmapLegend, ScoreHeatmap } from "@/components/group/score-heatmap";
import type { GroupMapData } from "@/lib/data/group-service";

export function GroupHealthMap({ data }: { data: GroupMapData }) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [pendingCompanyId, setPendingCompanyId] = useState<string | null>(null);

  const selectedCompanyId =
    isPending && pendingCompanyId ? pendingCompanyId : data.company?.companyId ?? null;

  const navigate = (groupId: string, companyId?: string) => {
    const query = new URLSearchParams({ group: groupId });
    if (companyId) query.set("company", companyId);
    startTransition(() => {
      router.push(`/grupos?${query.toString()}`, { scroll: false });
    });
  };

  const selectCompany = (companyId: string) => {
    if (companyId === data.company?.companyId) return;
    setPendingCompanyId(companyId);
    navigate(data.group.groupId, companyId);
  };

  const selectGroup = (groupId: string) => {
    if (groupId === data.group.groupId) return;
    setPendingCompanyId(null);
    navigate(groupId);
  };

  return (
    <div className="min-h-screen">
      <header className="sticky top-0 z-10 border-b bg-background/95 backdrop-blur">
        <div className="flex h-14 items-center justify-between px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/"
              className="inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
            >
              <ArrowLeft className="size-3.5" />
              Panel de empresa
            </Link>
            <Separator orientation="vertical" className="h-4" />
            <div className="min-w-0">
              <p className="truncate text-sm font-medium">Mapa de salud por grupo</p>
              <p className="truncate font-mono text-xs text-muted-foreground">{data.group.groupId}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="outline" className="font-mono text-[11px] font-normal text-muted-foreground">
              {formatMonth(data.asOfMonth)}
              {data.isSample ? " · muestra" : ""}
            </Badge>
            <ThemeSwitcher />
          </div>
        </div>
        <div className="flex items-center border-t px-4 py-1.5 sm:px-6">
          <ProductNav current="/grupos" />
        </div>
      </header>

      <main
        className="mx-auto w-full max-w-7xl px-4 py-8 sm:px-6 lg:px-8"
        aria-busy={isPending}
        data-pending={isPending || undefined}
      >
        <section className="flex flex-col gap-4">
          <div className="max-w-2xl">
            <div className="mb-2 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-muted-foreground">
              <Layers className="size-3.5" />
              Grupos de empresas
            </div>
            <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
              Puntuación de cada empresa del grupo, mes a mes.
            </h1>
            <p className="mt-2 text-sm leading-6 text-muted-foreground">
              Índice explicable y monitorizable de 0 a 100. Las filas son las empresas del grupo,
              ordenadas de menor a mayor puntuación; las columnas, los meses con dato.
            </p>
          </div>
          <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
            <div className="w-full sm:w-80">
              <p className="mb-1.5 text-xs font-medium text-muted-foreground">Grupo analizado</p>
              <EntityCombobox
                ariaLabel="Grupo analizado"
                options={data.groupOptions.map((option) => ({
                  value: option.groupId,
                  label: option.groupId,
                  detail: `${option.nCompanies} emp. · ${option.meanScore === null ? "sin media" : `media ${option.meanScore.toFixed(0)}`}`,
                }))}
                value={data.group.groupId}
                onChange={(next) => next && selectGroup(next)}
                placeholder="Elige un grupo"
                searchPlaceholder="Buscar grupo (p. ej. 0142)"
              />
            </div>
            <nav aria-label="Ir a otras vistas de este grupo" className="flex flex-wrap gap-2">
              {data.company ? (
                <>
                  <Button asChild variant="outline" size="sm">
                    <Link href={`/?company=${data.company.companyId}`}>
                      <Building2 data-icon="inline-start" />
                      Ver {data.company.companyId}
                    </Link>
                  </Button>
                  <Button asChild variant="outline" size="sm">
                    <Link href={`/?company=${data.company.companyId}#vigilancia`}>
                      <Eye data-icon="inline-start" />
                      Vigilancia
                    </Link>
                  </Button>
                </>
              ) : null}
            </nav>
          </div>
        </section>

        <section className="mt-6 grid gap-4 sm:grid-cols-3">
          <Card size="sm">
            <CardHeader className="pb-1">
              <CardTitle className="text-xs font-medium text-muted-foreground">Empresas</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="font-mono text-2xl font-medium tabular-nums">{data.group.nCompanies}</span>
              {data.members.length !== data.group.nCompanies && (
                <span className="ml-2 text-xs text-muted-foreground">{data.members.length} con puntuación</span>
              )}
            </CardContent>
          </Card>
          <Card size="sm">
            <CardHeader className="pb-1">
              <CardTitle className="text-xs font-medium text-muted-foreground">Media del grupo</CardTitle>
            </CardHeader>
            <CardContent>
              <span className="font-mono text-2xl font-medium tabular-nums">
                {data.group.meanScore === null ? "—" : data.group.meanScore.toFixed(0)}
              </span>
            </CardContent>
          </Card>
          <Card size="sm">
            <CardHeader className="pb-1">
              <CardTitle className="text-xs font-medium text-muted-foreground">Mínimo del grupo</CardTitle>
            </CardHeader>
            <CardContent className="flex items-baseline gap-2">
              <span className="font-mono text-2xl font-medium tabular-nums">
                {data.group.minScore === null ? "—" : data.group.minScore.toFixed(0)}
              </span>
              {data.group.minCompanyId && (
                <span className="font-mono text-xs text-muted-foreground">{data.group.minCompanyId}</span>
              )}
            </CardContent>
          </Card>
        </section>

        <Card className={`mt-4 transition-opacity ${isPending ? "opacity-70" : ""}`}>
          <CardHeader className="gap-1">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <CardTitle className="text-base">Mapa de calor de la puntuación</CardTitle>
              <HeatmapLegend />
            </div>
            <p className="text-sm text-muted-foreground">
              Primera fila: media del grupo. Pasa el cursor por una celda para ver empresa, mes y puntuación; haz clic para
              seleccionar la empresa.
            </p>
          </CardHeader>
          <CardContent>
            <ScoreHeatmap
              months={data.months}
              meanScores={data.group.meanScores}
              members={data.members}
              selectedCompanyId={selectedCompanyId}
              onSelectCompany={selectCompany}
            />
          </CardContent>
        </Card>

        <section className={`mt-4 grid gap-4 2xl:grid-cols-[1.15fr_1fr] transition-opacity ${isPending ? "opacity-70" : ""}`}>
          <Card>
            <CardHeader className="gap-1">
              <CardTitle className="text-base">Empresas del grupo</CardTitle>
              <p className="text-sm text-muted-foreground">Ordenadas por puntuación actual, de menor a mayor.</p>
            </CardHeader>
            <CardContent>
              <MemberTable
                members={data.members}
                selectedCompanyId={selectedCompanyId}
                onSelectCompany={selectCompany}
              />
            </CardContent>
          </Card>
          <CompanyPanel company={data.company} />
        </section>

        <Card className="mt-4">
          <CardHeader className="gap-1">
            <CardTitle className="text-base">Señales del grupo</CardTitle>
            <p className="text-sm text-muted-foreground">
              {data.group.limitsAvailable
                ? "Este grupo tiene límites tipo embudo: su media se vigila frente a su propio histórico y frente a otros grupos de tamaño similar."
                : "Grupo pequeño: solo media, sin límites ni alertas de grupo."}
            </p>
          </CardHeader>
          <CardContent>
            <AlertList
              alerts={data.group.alerts}
              showEntity
              emptyText={
                data.group.limitsAvailable
                  ? "Sin alertas de grupo en la ventana de detalle."
                  : "Las alertas de grupo requieren al menos 3 empresas con puntuación."
              }
            />
          </CardContent>
        </Card>

        <ProductCredit className="mt-8" />
      </main>
    </div>
  );
}
