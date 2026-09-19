"use client";

import { Segmented, type SegmentedOption } from "@/components/segmented";

export type AnalysisMode = "quick" | "deep";

const OPTIONS: SegmentedOption<AnalysisMode>[] = [
  { value: "quick", label: "Rápido" },
  { value: "deep", label: "Profundo" },
];

/** Switches between the quick overview and the deep analysis. */
export function ModeToggle({ mode, onChange }: { mode: AnalysisMode; onChange: (mode: AnalysisMode) => void }) {
  return <Segmented options={OPTIONS} value={mode} onChange={onChange} ariaLabel="Modo de análisis" />;
}
