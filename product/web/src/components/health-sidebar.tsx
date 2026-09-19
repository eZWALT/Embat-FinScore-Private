"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  Activity,
  BarChart3,
  ChartNoAxesCombined,
  Database,
  Layers,
  LayoutDashboard,
  ScanSearch,
} from "lucide-react";

import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  SidebarSeparator,
  useSidebar,
} from "@/components/ui/sidebar";
import { ThemeSwitcher } from "@/components/theme-switcher";
import type { DashboardData } from "@/lib/data/types";

export type AppView = "overview" | "health-score";

const views = [
  { id: "overview" as const, label: "Resumen", icon: LayoutDashboard },
  { id: "health-score" as const, label: "Health Score", icon: ChartNoAxesCombined },
];

const overviewAnchors = [
  { label: "Evolución", href: "#evolucion", icon: Activity },
  { label: "Categorías", href: "#categorias", icon: BarChart3 },
  { label: "Señales", href: "#senales", icon: ScanSearch },
] as const;

type OverviewHref = (typeof overviewAnchors)[number]["href"] | "#resumen";

function isOverviewHref(hash: string): hash is OverviewHref {
  return hash === "#resumen" || overviewAnchors.some((item) => item.href === hash);
}

export function HealthSidebar({
  data,
  companyId,
  onCompanyChange,
  view,
  onViewChange,
}: {
  data: DashboardData;
  companyId: string;
  onCompanyChange: (companyId: string) => void;
  view: AppView;
  onViewChange: (view: AppView) => void;
}) {
  const { setOpenMobile } = useSidebar();
  const [activeHref, setActiveHref] = useState<OverviewHref>("#resumen");

  useEffect(() => {
    const syncActiveHref = () => {
      const { hash } = window.location;
      setActiveHref(isOverviewHref(hash) ? hash : "#resumen");
    };

    syncActiveHref();
    window.addEventListener("hashchange", syncActiveHref);

    return () => window.removeEventListener("hashchange", syncActiveHref);
  }, []);

  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader className="gap-3 p-3">
        <div className="flex h-10 items-center gap-3 px-1">
          <div className="grid size-8 shrink-0 place-items-center rounded-lg border bg-background">
            <Activity className="size-4" aria-hidden="true" />
          </div>
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <p className="truncate text-sm font-semibold leading-none">Health Sentinel</p>
            <p className="mt-1 truncate text-xs text-sidebar-foreground/60">Embat Financial Intelligence</p>
          </div>
        </div>

        {view === "overview" ? (
          <div className="group-data-[collapsible=icon]:hidden">
            <label className="mb-1.5 block px-1 text-[11px] font-medium text-sidebar-foreground/60" htmlFor="sidebar-company">
              Empresa analizada
            </label>
            <Select
              value={companyId}
              onValueChange={(value) => {
                onCompanyChange(value);
                setOpenMobile(false);
              }}
            >
              <SelectTrigger id="sidebar-company" className="w-full bg-background">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {data.companies.map((company) => (
                  <SelectItem key={company.companyId} value={company.companyId}>
                    {company.companyId} · {company.score.toFixed(0)} pts
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        ) : null}
      </SidebarHeader>

      <SidebarSeparator />

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navegación</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {views.map((item) => (
                <SidebarMenuItem key={item.id}>
                  <SidebarMenuButton
                    isActive={view === item.id}
                    tooltip={item.label}
                    onClick={() => {
                      onViewChange(item.id);
                      if (item.id === "overview") {
                        setActiveHref("#resumen");
                      }
                      setOpenMobile(false);
                    }}
                  >
                    <item.icon />
                    <span>{item.label}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
              <SidebarMenuItem>
                <SidebarMenuButton asChild tooltip="Grupos">
                  <Link href="/grupos" onClick={() => setOpenMobile(false)}>
                    <Layers />
                    <span>Grupos</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
              {overviewAnchors.map((item) => (
                <SidebarMenuItem key={item.href}>
                  <SidebarMenuButton
                    asChild
                    isActive={view === "overview" && activeHref === item.href}
                    tooltip={item.label}
                  >
                    <a
                      href={item.href}
                      aria-current={
                        view === "overview" && activeHref === item.href ? "location" : undefined
                      }
                      onClick={() => {
                        onViewChange("overview");
                        setActiveHref(item.href);
                        setOpenMobile(false);
                      }}
                    >
                      <item.icon />
                      <span>{item.label}</span>
                    </a>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="gap-3 p-3">
        <div className="flex items-center gap-2 px-1 text-xs text-sidebar-foreground/60 group-data-[collapsible=icon]:justify-center">
          <Database className="size-3.5 shrink-0" />
          <span className="group-data-[collapsible=icon]:hidden">Neon · {data.asOfMonth}</span>
        </div>
        <div className="group-data-[collapsible=icon]:hidden">
          <ThemeSwitcher />
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
