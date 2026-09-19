"use client";

import { cn } from "cn";

export type AnalysisMode = "quick" | "deep";

const OPTIONS: { value: AnalysisMode; label: string }[] = [
  { value: "quick", label: "Rápido" },
  { value: "deep", label: "Profundo" },
];

/** Switches between the quick overview and the deep analysis. The active pill slides between the two. */
export function ModeToggle({ mode, onChange }: { mode: AnalysisMode; onChange: (mode: AnalysisMode) => void }) {
  return (
    <div role="group" aria-label="Modo de análisis" className="relative grid grid-cols-2 rounded-lg border bg-card p-0.5">
      <span
        aria-hidden="true"
        className={cn(
          "absolute inset-y-0.5 left-0.5 w-[calc(50%-0.125rem)] rounded-md bg-foreground transition-transform duration-300 ease-out motion-reduce:transition-none",
          mode === "deep" && "translate-x-full",
        )}
      />
      {OPTIONS.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={mode === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "relative h-7 min-w-[4.5rem] rounded-md px-3 text-xs font-medium transition-colors duration-300 outline-none focus-visible:ring-3 focus-visible:ring-ring/50",
            mode === option.value ? "text-background" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
