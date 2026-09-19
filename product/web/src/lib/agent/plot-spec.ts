export type PlotSpec = {
  title: string;
  x: string[];
  series: Record<string, (number | null)[]>;
  kind?: "line" | "bar";
  markers?: string[];
  band?: { lower: (number | null)[]; upper: (number | null)[] } | null;
  y_label?: string | null;
};

export function isPlotSpec(value: unknown): value is PlotSpec {
  if (!value || typeof value !== "object") return false;
  const plot = value as PlotSpec;
  return Array.isArray(plot.x) && !!plot.series && typeof plot.series === "object" && typeof plot.title === "string";
}
