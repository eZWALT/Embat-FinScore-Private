"use client";

import { CircleHelp, X } from "lucide-react";
import { Dialog } from "radix-ui";

import { Button } from "@/components/ui/button";

interface MethodItem {
  item: string;
  calculation: string;
  rationale: string;
}

interface MethodCategory {
  category: string;
  weight: string;
  items: MethodItem[];
}

/** Mirrors product/score/METHOD.md §3.1 and spec.py (Spanish copy). Update all three together. */
const METHOD: MethodCategory[] = [
  {
    category: "Historial de pagos",
    weight: "35%",
    items: [
      {
        item: "Días de retraso al pagar (proveedores)",
        calculation: "Días entre el vencimiento y el pago de la factura, ponderados por importe, 3 meses",
        rationale: "Análogo directo al historial de pagos: con qué fiabilidad la empresa paga sus obligaciones.",
      },
      {
        item: "Días de retraso al cobrar (clientes)",
        calculation: "Mismo cálculo, aplicado a las facturas emitidas",
        rationale: "El cobro tardío es uno de los principales motores de tensión de liquidez.",
      },
      {
        item: "Pagos pendientes > 30 días vencidos",
        calculation: "Porcentaje de facturas a pagar abiertas con más de 30 días de vencimiento, 3 meses",
        rationale: "Umbral del Banque de France: un retraso superior a 30 días aumenta de forma significativa el riesgo de impago.",
      },
      {
        item: "Cobros pendientes > 30 días vencidos",
        calculation: "Porcentaje de facturas a cobrar abiertas con más de 30 días de vencimiento, 3 meses",
        rationale: "El mismo indicador de riesgo aplicado a la entrada de caja.",
      },
    ],
  },
  {
    category: "Deuda y liquidez",
    weight: "30%",
    items: [
      {
        item: "Meses de salidas cubiertos por la caja",
        calculation: "Caja a fin de mes ÷ salida operativa mensual media (ventana de 6 meses, acotado entre -6 y 24), 3 meses",
        rationale: "Mide la autonomía sin ingresos (análogo a la deuda pendiente).",
      },
      {
        item: "Cierres de mes con caja negativa",
        calculation: "Porcentaje de los últimos 3 cierres de mes con caja < 0",
        rationale: "Señal directa de tensión operativa.",
      },
      {
        item: "Veces que la caja se volvió negativa",
        calculation: "Inicios de caja negativa en los últimos 6 cierres de mes",
        rationale: "Distingue un déficit aislado de un problema estructural recurrente.",
      },
      {
        item: "Servicio de la deuda / entradas",
        calculation: "Amortización de deuda de 3 meses ÷ entradas operativas de 3 meses, 3 meses",
        rationale: "Evalúa la carga total de deuda frente a la capacidad de entrada de caja.",
      },
      {
        item: "Comisiones e intereses bancarios / entradas",
        calculation: "Comisiones + intereses ÷ entradas operativas, 3 meses",
        rationale: "Evalúa la tensión del coste de financiación.",
      },
    ],
  },
  {
    category: "Antigüedad y estabilidad",
    weight: "15%",
    items: [
      {
        item: "Meses de historial",
        calculation: "Fijo: 100 × min(meses, 12) ÷ 12",
        rationale: "Los historiales cortos implican más incertidumbre (véase el indicador de confianza).",
      },
      {
        item: "Meses con entrada de dinero",
        calculation: "Fijo: 100 × porcentaje de los últimos 6 meses con entradas positivas",
        rationale: "Salvaguarda: las empresas que pierden velocidad de entradas pierden puntos de salud.",
      },
      {
        item: "Volatilidad de las salidas",
        calculation: "Desviación típica ÷ media de las salidas operativas mensuales en 6 meses (limitado a 3)",
        rationale: "Los picos irregulares de gasto complican la planificación financiera.",
      },
    ],
  },
  {
    category: "Crédito nuevo",
    weight: "10% → efectivo 5%",
    items: [
      {
        item: "Servicio de la deuda al alza",
        calculation: "Aumento de (servicio de la deuda ÷ entradas) frente a la misma ventana de hace 6 meses (0 si bajó)",
        rationale: "Aproximación operativa a la asunción de nueva deuda.",
      },
      {
        item: "Comisiones e intereses al alza",
        calculation: "Aumento de (comisiones + intereses ÷ entradas) frente a la misma ventana de hace 6 meses",
        rationale: "Aproximación a la expansión del uso de crédito o a la escalada de penalizaciones.",
      },
    ],
  },
  {
    category: "Mix de clientes",
    weight: "10%",
    items: [
      {
        item: "Dependencia de un solo cliente",
        calculation: "Concentración de clientes (HHI), contando solo la parte por encima de 0,975, 3 meses",
        rationale: "Se centra estrictamente en la cola extrema (riesgo de «comprador único»).",
      },
      {
        item: "Abonos / facturación",
        calculation: "Porcentaje de la facturación total revertido por notas de abono, 3 meses",
        rationale: "Tasas altas de reversión indican disputas con clientes o errores de facturación.",
      },
    ],
  },
];

/** "?" next to an Índice de salud heading. Click opens how the index is built. */
export function HealthIndexHelp({ className }: { className?: string }) {
  return (
    <Dialog.Root>
      <Dialog.Trigger asChild>
        <button
          type="button"
          aria-label="Cómo se calcula el índice de salud"
          className={
            "inline-grid size-5 shrink-0 place-items-center rounded-full text-muted-foreground outline-none transition-colors hover:text-foreground focus-visible:ring-3 focus-visible:ring-ring/50 " +
            (className ?? "")
          }
        >
          <CircleHelp className="size-4" aria-hidden="true" />
        </button>
      </Dialog.Trigger>
      <Dialog.Portal>
        <Dialog.Overlay className="fixed inset-0 z-50 bg-black/40 data-open:animate-in data-open:fade-in-0 data-closed:animate-out data-closed:fade-out-0" />
        <Dialog.Content aria-describedby={undefined} className="fixed left-1/2 top-1/2 z-50 flex max-h-[min(88vh,52rem)] w-[calc(100vw-2rem)] max-w-4xl -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border bg-popover text-popover-foreground shadow-lg outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95">
          <div className="flex items-start justify-between gap-4 border-b p-5">
            <div className="min-w-0">
              <Dialog.Title className="text-base font-semibold tracking-tight">
                Cómo se calcula el Índice de salud
              </Dialog.Title>
            </div>
            <Dialog.Close asChild>
              <Button type="button" variant="ghost" size="icon-sm" aria-label="Cerrar">
                <X />
              </Button>
            </Dialog.Close>
          </div>

          <div className="overflow-y-auto p-5">
            <div className="overflow-x-auto rounded-lg border">
              <table className="w-full min-w-[46rem] border-collapse text-left text-sm">
                <thead className="bg-muted/50 text-xs text-muted-foreground">
                  <tr>
                    <th scope="col" className="w-40 px-3 py-2 font-medium">Categoría (peso)</th>
                    <th scope="col" className="w-52 px-3 py-2 font-medium">Indicador</th>
                    <th scope="col" className="px-3 py-2 font-medium">Cálculo</th>
                    <th scope="col" className="px-3 py-2 font-medium">Justificación / análogo FICO</th>
                  </tr>
                </thead>
                {METHOD.map((group) => (
                  <tbody key={group.category} className="border-t">
                    {group.items.map((row, index) => (
                      <tr key={row.item} className={index > 0 ? "border-t border-dashed" : undefined}>
                        {index === 0 ? (
                          <th scope="row" rowSpan={group.items.length} className="px-3 py-2.5 align-top font-medium">
                            {group.category}
                            <span className="mt-0.5 block font-mono text-xs font-normal text-muted-foreground">{group.weight}</span>
                          </th>
                        ) : null}
                        <td className="px-3 py-2.5 align-top font-medium">{row.item}</td>
                        <td className="px-3 py-2.5 align-top text-muted-foreground">{row.calculation}</td>
                        <td className="px-3 py-2.5 align-top text-muted-foreground">{row.rationale}</td>
                      </tr>
                    ))}
                  </tbody>
                ))}
              </table>
            </div>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
