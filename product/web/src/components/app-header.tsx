"use client";

import Image from "next/image";
import Link from "next/link";
import { ViewTransition, type ReactNode } from "react";

import { ThemeIconToggle } from "@/components/theme-switcher";

/** The one header of the platform: brand on the left, page controls and the theme switch on the right. */
export function AppHeader({ children }: { children?: ReactNode }) {
  return (
    <header className="sticky top-0 z-10 flex h-14 items-center justify-between border-b bg-background/95 px-4 backdrop-blur sm:px-6">
      <Link href="/" className="flex items-center rounded-lg outline-none focus-visible:ring-3 focus-visible:ring-ring/50">
        {/* Black-on-white JPG: multiply drops the white on the light theme; inverted + screen does the same on dark. */}
        <ViewTransition name="sentinel-logo" share="logo-morph" default="none">
          <Image
            src="/landing/sentinel-logo.jpg"
            alt="Sentinel"
            width={1024}
            height={341}
            className="h-10 w-auto mix-blend-multiply sm:h-11 dark:mix-blend-screen dark:invert"
            priority
          />
        </ViewTransition>
      </Link>
      <div className="flex items-center gap-3">
        {children}
        <ThemeIconToggle />
      </div>
    </header>
  );
}
