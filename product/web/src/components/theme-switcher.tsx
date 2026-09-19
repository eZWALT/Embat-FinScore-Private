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

export function ThemeSwitcher() {
  const theme = useSyncExternalStore(subscribe, getTheme, getServerTheme);

  const chooseTheme = (nextTheme: Theme) => {
    document.documentElement.classList.toggle("dark", nextTheme === "dark");
    document.documentElement.style.colorScheme = nextTheme;
    window.localStorage.setItem("theme", nextTheme);
    window.dispatchEvent(new Event("theme-change"));
  };

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
        onClick={() => chooseTheme("light")}
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
        onClick={() => chooseTheme("dark")}
      >
        <Moon className="size-3.5" />
        <span className="text-xs">Oscuro</span>
      </Button>
    </div>
  );
}
