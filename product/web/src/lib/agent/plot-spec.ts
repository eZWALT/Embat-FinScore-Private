export type PlotSpec = {
  title: string;
  x: string[];
  series: Record<string, (number | null)[]>;
  kind?: "line" | "bar" | "pie";
  markers?: string[];
  band?: { lower: (number | null)[]; upper: (number | null)[] } | null;
  y_label?: string | null;
  /** Horizontal reference lines the server computed (e.g. the group mean). */
  ref_lines?: { label: string; value: number }[];
  /** x categories drawn in the accent colour (e.g. members below the mean). */
  highlight?: string[];
  /** Small tag on an x category (e.g. "alerta"). */
  badges?: Record<string, string>;
};

export function isPlotSpec(value: unknown): value is PlotSpec {
  if (!value || typeof value !== "object") return false;
  const plot = value as PlotSpec;
  return Array.isArray(plot.x) && !!plot.series && typeof plot.series === "object" && typeof plot.title === "string";
}
