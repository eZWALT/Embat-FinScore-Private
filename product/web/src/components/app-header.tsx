"use client";

import Link from "next/link";
import type { ReactNode } from "react";
import { Activity } from "lucide-react";

import { ThemeIconToggle } from "@/components/theme-switcher";

/** The one header of the platform: brand on the left, page controls and the theme switch on the right. */
export function AppHeader({ children }: { children?: ReactNode }) {
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b bg-background/95 px-4 backdrop-blur sm:px-6">
      <Link href="/" className="flex items-center gap-2.5 rounded-lg outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
        <span className="grid size-8 shrink-0 place-items-center rounded-lg border bg-background">
          <Activity className="size-4" aria-hidden="true" />
        </span>
        <span className="hidden text-sm font-semibold sm:inline">Centinela de salud</span>
      </Link>
      <div className="flex items-center gap-3">
        {children}
        <ThemeIconToggle />
      </div>
    </header>
  );
}
