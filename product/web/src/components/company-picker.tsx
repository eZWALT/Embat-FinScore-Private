"use client";

import { useState } from "react";
import { ChevronsUpDown, Search } from "lucide-react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { cn } from "cn";

import { companyLabel } from "@/components/group/labels";
import { QuickList } from "@/components/quick/quick-list";
import type { DashboardCompany } from "@/lib/data/types";

/**
 * Company chooser of the deep view. The same search and sortable list as Comparar (`QuickList`), so companies can be
 * ordered by index, monthly or 3-month change, or name before picking one.
 */
export function CompanyPicker({
  companies,
  value,
  onChange,
  className,
}: {
  companies: DashboardCompany[];
  value: string;
  onChange: (companyId: string) => void;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const selected = companies.find((company) => company.companyId === value);

  return (
    <PopoverPrimitive.Root
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) setQuery("");
      }}
    >
      <PopoverPrimitive.Trigger asChild>
        <button
          type="button"
          aria-haspopup="dialog"
          aria-expanded={open}
          aria-label="Empresa analizada"
          className={cn(
            "flex h-9 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-background px-2.5 text-left text-sm outline-none transition-colors hover:bg-muted/50 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30 dark:hover:bg-input/50",
            className,
          )}
        >
          <span className="truncate text-[13px]">{companyLabel(value)}</span>
          <span className="flex shrink-0 items-center gap-1.5 text-muted-foreground">
            {selected ? <span className="font-mono text-xs tabular-nums">{selected.score.toFixed(0)} pts</span> : null}
            <ChevronsUpDown className="size-3.5" aria-hidden="true" />
          </span>
        </button>
      </PopoverPrimitive.Trigger>

      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          align="start"
          sideOffset={6}
          className="z-50 w-[min(24rem,calc(100vw-2rem))] origin-(--radix-popover-content-transform-origin) space-y-3 rounded-lg bg-popover p-3 text-popover-foreground shadow-md ring-1 ring-foreground/10 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95"
        >
          <div className="relative">
            <Search
              className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-muted-foreground"
              aria-hidden="true"
            />
            <input
              autoFocus
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Nombre o grupo…"
              aria-label="Buscar empresa por nombre o grupo"
              autoComplete="off"
              className="h-8 w-full rounded-md border bg-background pr-2 pl-8 text-sm outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
            />
          </div>
          <QuickList
            single
            companies={companies}
            query={query}
            picked={[value]}
            max={1}
            onToggle={(companyId) => {
              onChange(companyId);
              setOpen(false);
            }}
          />
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
