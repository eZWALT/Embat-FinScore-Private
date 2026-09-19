"use client";

import {
  Activity,
  ChartNoAxesCombined,
  Database,
  Eye,
  LayoutDashboard,
  MessageSquare,
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
  { id: "health-score" as const, label: "Índice de salud", icon: ChartNoAxesCombined },
];

export function HealthSidebar({
  data,
  companyId,
  onCompanyChange,
  view,
  onViewChange,
  onOpenChat,
  onVigilancia,
}: {
  data: DashboardData;
  companyId: string;
  onCompanyChange: (companyId: string) => void;
  view: AppView;
  onViewChange: (view: AppView) => void;
  onOpenChat?: () => void;
  onVigilancia?: () => void;
}) {
  const { setOpenMobile } = useSidebar();

  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader className="gap-3 p-3">
        <div className="flex h-10 items-center gap-3 px-1">
          <div className="grid size-8 shrink-0 place-items-center rounded-lg border bg-background">
            <Activity className="size-4" aria-hidden="true" />
          </div>
          <div className="min-w-0 group-data-[collapsible=icon]:hidden">
            <p className="truncate text-sm font-semibold leading-none">Centinela de salud</p>
            <p className="mt-1 truncate text-xs text-sidebar-foreground/60">Inteligencia financiera Embat</p>
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
                      setOpenMobile(false);
                    }}
                  >
                    <item.icon />
                    <span>{item.label}</span>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
              <SidebarMenuItem>
                <SidebarMenuButton
                  isActive={view === "overview"}
                  tooltip="Vigilancia"
                  onClick={() => {
                    onVigilancia?.();
                    setOpenMobile(false);
                  }}
                >
                  <Eye />
                  <span>Vigilancia</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton
                  tooltip="Consultas"
                  onClick={() => {
                    onOpenChat?.();
                    setOpenMobile(false);
                  }}
                >
                  <MessageSquare />
                  <span>Consultas</span>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="gap-3 p-3">
        <div className="flex items-center gap-2 px-1 text-xs text-sidebar-foreground/60 group-data-[collapsible=icon]:justify-center">
          <Database className="size-3.5 shrink-0" />
          <span className="group-data-[collapsible=icon]:hidden">Datos · {data.asOfMonth}</span>
        </div>
        <div className="group-data-[collapsible=icon]:hidden">
          <ThemeSwitcher />
        </div>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
