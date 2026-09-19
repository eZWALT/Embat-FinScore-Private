"use client";

import { cn } from "cn";

export type AnalysisMode = "quick" | "deep";

const OPTIONS: { value: AnalysisMode; label: string }[] = [
  { value: "quick", label: "Rápido" },
  { value: "deep", label: "Profundo" },
];

/** Switches between the quick overview and the deep analysis. */
export function ModeToggle({ mode, onChange }: { mode: AnalysisMode; onChange: (mode: AnalysisMode) => void }) {
  return (
    <div role="group" aria-label="Modo de análisis" className="flex items-center rounded-lg border bg-card p-0.5">
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={mode === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "h-7 rounded-md px-3 text-xs font-medium transition-colors outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
            mode === option.value ? "bg-foreground text-background" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
