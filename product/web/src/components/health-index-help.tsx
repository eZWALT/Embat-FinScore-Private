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

/** Mirrors product/score/METHOD.md §3.1 and spec.py. Update all three together. */
const METHOD: MethodCategory[] = [
  {
    category: "Payment History",
    weight: "35%",
    items: [
      {
        item: "Days paid after due date (Suppliers)",
        calculation: "Amount-weighted days between due date and invoice payment, 3-mo",
        rationale: "Direct analogue to payment history: how reliably the company pays its obligations.",
      },
      {
        item: "Days customers pay after due date",
        calculation: "Same calculation, applied to issued invoices",
        rationale: "Late collection is a primary driver of liquidity stress.",
      },
      {
        item: "Payables > 30 days overdue",
        calculation: "Share of open payables > 30 days past due, 3-mo",
        rationale: "Banque de France threshold: lateness beyond 30 days significantly increases default risk.",
      },
      {
        item: "Receivables > 30 days overdue",
        calculation: "Share of open receivables > 30 days past due, 3-mo",
        rationale: "Same risk indicator applied to incoming cash flow.",
      },
    ],
  },
  {
    category: "Amounts Owed & Liquidity",
    weight: "30%",
    items: [
      {
        item: "Months of outflows covered by cash",
        calculation: "Month-end cash ÷ mean monthly operating outflow (6-mo window, clipped -6 to 24), 3-mo",
        rationale: "Measures runway without income (analogue of amounts owed).",
      },
      {
        item: "Month-ends with negative cash",
        calculation: "Share of last 3 month-ends where cash < 0",
        rationale: "Direct operational stress signal.",
      },
      {
        item: "Times cash turned negative",
        calculation: "Onsets of negative cash over the last 6 month-ends",
        rationale: "Distinguishes a single isolated deficit from a recurring structural issue.",
      },
      {
        item: "Debt service / inflows",
        calculation: "3-month debt repayment ÷ 3-month operating inflow, 3-mo",
        rationale: "Evaluates overall debt burden against inflow capacity.",
      },
      {
        item: "Bank fees & interest / inflows",
        calculation: "Fees + interest ÷ operating inflow, 3-mo",
        rationale: "Evaluates cost of funding strain.",
      },
    ],
  },
  {
    category: "Length & Stability",
    weight: "15%",
    items: [
      {
        item: "Months of history",
        calculation: "Fixed: 100 × min(months, 12) ÷ 12",
        rationale: "Shorter histories carry higher uncertainty (see the confidence flag).",
      },
      {
        item: "Months with incoming money",
        calculation: "Fixed: 100 × share of the last 6 months with positive inflow",
        rationale: "Safety guard: companies losing inflow velocity lose health points.",
      },
      {
        item: "Outflow volatility",
        calculation: "Standard deviation ÷ mean of monthly operating outflows over 6 months (capped at 3)",
        rationale: "Irregular spending spikes complicate financial planning.",
      },
    ],
  },
  {
    category: "New Credit",
    weight: "10% → effective 5%",
    items: [
      {
        item: "Debt service rising",
        calculation: "Rise of (debt service ÷ inflows) vs. the same window 6 months prior (0 if it fell)",
        rationale: "Operational proxy for taking on new debt exposure.",
      },
      {
        item: "Fees and interest rising",
        calculation: "Rise of (fees + interest ÷ inflows) vs. the same window 6 months prior",
        rationale: "Proxy for expanding credit usage or penalty escalation.",
      },
    ],
  },
  {
    category: "Customer Mix",
    weight: "10%",
    items: [
      {
        item: "Dependence on a single customer",
        calculation: "Customer concentration (HHI), counting only the part above 0.975, 3-mo",
        rationale: "Focuses strictly on the extreme tail (“single buyer” risk).",
      },
      {
        item: "Credit notes / billing",
        calculation: "Share of total billing reversed by credit notes, 3-mo",
        rationale: "High reversal rates indicate customer disputes or billing errors.",
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
        <Dialog.Content className="fixed left-1/2 top-1/2 z-50 flex max-h-[min(88vh,52rem)] w-[calc(100vw-2rem)] max-w-4xl -translate-x-1/2 -translate-y-1/2 flex-col rounded-xl border bg-popover text-popover-foreground shadow-lg outline-none data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95">
          <div className="flex items-start justify-between gap-4 border-b p-5">
            <div className="min-w-0">
              <Dialog.Title className="text-base font-semibold tracking-tight">
                How the Health Index is calculated
              </Dialog.Title>
              <Dialog.Description className="mt-1 text-sm leading-6 text-muted-foreground">
                A FICO-inspired framework. Item values are computed looking back from each month (“3-mo” is the mean of
                the last three monthly values). Definitions and category weights are fixed in <code className="font-mono text-xs">spec.py</code>;
                nothing is fitted.
              </Dialog.Description>
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
                    <th scope="col" className="w-40 px-3 py-2 font-medium">Category (weight)</th>
                    <th scope="col" className="w-52 px-3 py-2 font-medium">Metric item</th>
                    <th scope="col" className="px-3 py-2 font-medium">Calculation</th>
                    <th scope="col" className="px-3 py-2 font-medium">Rationale / FICO analogue</th>
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
            <p className="mt-3 text-xs leading-5 text-muted-foreground">
              New Credit is thin (the trail barely observes it), so its weight is halved and the 5 points are spread over
              the other categories. The index is a monitoring aid, not an insolvency predictor.
            </p>
          </div>
        </Dialog.Content>
      </Dialog.Portal>
    </Dialog.Root>
  );
}
