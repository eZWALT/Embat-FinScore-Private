"use client";

import { Moon, Sun } from "lucide-react";
import { useSyncExternalStore } from "react";

import { Button } from "@/components/ui/button";

type Theme = "light" | "dark";

const subscribe = (callback: () => void) => {
  window.addEventListener("theme-change", callback);
  return () => window.removeEventListener("theme-change", callback);
};

const getTheme = (): Theme =>
  document.documentElement.classList.contains("dark") ? "dark" : "light";

const getServerTheme = (): Theme => "light";

function applyTheme(nextTheme: Theme) {
  document.documentElement.classList.toggle("dark", nextTheme === "dark");
  document.documentElement.style.colorScheme = nextTheme;
  window.localStorage.setItem("theme", nextTheme);
  window.dispatchEvent(new Event("theme-change"));
}

export function ThemeSwitcher() {
  const theme = useSyncExternalStore(subscribe, getTheme, getServerTheme);

  return (
    <div
      className="flex items-center rounded-lg border bg-card p-0.5"
      aria-label="Apariencia"
    >
      <Button
        type="button"
        variant={theme === "light" ? "secondary" : "ghost"}
        size="sm"
        className="h-7 gap-1.5 px-2 sm:px-2.5"
        aria-label="Usar tema claro"
        aria-pressed={theme === "light"}
        onClick={() => applyTheme("light")}
      >
        <Sun className="size-3.5" />
        <span className="text-xs">Claro</span>
      </Button>
      <Button
        type="button"
        variant={theme === "dark" ? "secondary" : "ghost"}
        size="sm"
        className="h-7 gap-1.5 px-2 sm:px-2.5"
        aria-label="Usar tema oscuro"
        aria-pressed={theme === "dark"}
        onClick={() => applyTheme("dark")}
      >
        <Moon className="size-3.5" />
        <span className="text-xs">Oscuro</span>
      </Button>
    </div>
  );
}

/** One-button theme switch for tight headers. */
export function ThemeIconToggle() {
  const theme = useSyncExternalStore(subscribe, getTheme, getServerTheme);
  const next: Theme = theme === "dark" ? "light" : "dark";

  return (
    <Button
      type="button"
      variant="ghost"
      size="icon-sm"
      aria-label={next === "light" ? "Usar tema claro" : "Usar tema oscuro"}
      onClick={() => applyTheme(next)}
    >
      {theme === "dark" ? <Sun /> : <Moon />}
    </Button>
  );
}
