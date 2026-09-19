"use client";

import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Popover as PopoverPrimitive } from "radix-ui";
import { Check, ChevronsUpDown, Search, X } from "lucide-react";
import { cn } from "cn";

export interface ComboOption {
  value: string;
  label: string;
  /** Right-aligned secondary text, e.g. "91 pts". */
  detail?: string;
}

const MAX_RENDERED = 100;

export function EntityCombobox({
  options,
  value,
  onChange,
  placeholder,
  searchPlaceholder,
  emptyText = "Sin resultados.",
  clearLabel,
  ariaLabel,
  className,
}: {
  options: ComboOption[];
  value: string | null;
  onChange: (value: string | null) => void;
  placeholder: string;
  searchPlaceholder: string;
  emptyText?: string;
  /** When set, the popover offers a first row that clears the selection. */
  clearLabel?: string;
  ariaLabel: string;
  className?: string;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const listRef = useRef<HTMLUListElement>(null);
  const listId = useId();

  const selected = options.find((option) => option.value === value) ?? null;

  const matches = useMemo(() => {
    const needle = query.trim().toLowerCase();
    if (!needle) return options;
    return options.filter(
      (option) =>
        option.label.toLowerCase().includes(needle) || option.value.toLowerCase().includes(needle),
    );
  }, [options, query]);

  const shown = matches.slice(0, MAX_RENDERED);
  const rows: { value: string | null; label: string; detail?: string }[] = [
    ...(clearLabel && !query ? [{ value: null, label: clearLabel }] : []),
    ...shown,
  ];

  useEffect(() => {
    listRef.current?.querySelector<HTMLElement>('[data-active="true"]')?.scrollIntoView({ block: "nearest" });
  }, [active, open]);

  function choose(next: string | null) {
    onChange(next);
    setOpen(false);
  }

  return (
    <PopoverPrimitive.Root
      open={open}
      onOpenChange={(next) => {
        setOpen(next);
        if (next) {
          setQuery("");
          const at = rows.findIndex((row) => row.value === value);
          setActive(Math.max(0, at));
        }
      }}
    >
      <PopoverPrimitive.Trigger asChild>
        <button
          type="button"
          role="combobox"
          aria-expanded={open}
          aria-controls={listId}
          aria-label={ariaLabel}
          className={cn(
            "flex h-9 w-full min-w-0 items-center justify-between gap-2 rounded-lg border border-input bg-background px-2.5 text-left text-sm outline-none transition-colors hover:bg-muted/50 focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 dark:bg-input/30 dark:hover:bg-input/50",
            className,
          )}
        >
          <span className={cn("truncate", selected ? "text-[13px]" : "text-muted-foreground")}>
            {selected?.label ?? placeholder}
          </span>
          <span className="flex shrink-0 items-center gap-1.5 text-muted-foreground">
            {selected?.detail ? <span className="font-mono text-xs tabular-nums">{selected.detail}</span> : null}
            <ChevronsUpDown className="size-3.5" aria-hidden="true" />
          </span>
        </button>
      </PopoverPrimitive.Trigger>

      <PopoverPrimitive.Portal>
        <PopoverPrimitive.Content
          align="start"
          sideOffset={6}
          className="z-50 w-(--radix-popover-trigger-width) min-w-64 origin-(--radix-popover-content-transform-origin) rounded-lg bg-popover p-1 text-popover-foreground shadow-md ring-1 ring-foreground/10 data-open:animate-in data-open:fade-in-0 data-open:zoom-in-95 data-closed:animate-out data-closed:fade-out-0 data-closed:zoom-out-95"
        >
          <div className="relative border-b p-1 pb-1.5">
            <Search
              className="pointer-events-none absolute top-1/2 left-3 size-3.5 -translate-y-[60%] text-muted-foreground"
              aria-hidden="true"
            />
            <input
              autoFocus
              value={query}
              onChange={(event) => {
                setQuery(event.target.value);
                setActive(0);
              }}
              onKeyDown={(event) => {
                if (event.key === "ArrowDown") {
                  event.preventDefault();
                  setActive((index) => Math.min(rows.length - 1, index + 1));
                } else if (event.key === "ArrowUp") {
                  event.preventDefault();
                  setActive((index) => Math.max(0, index - 1));
                } else if (event.key === "Enter") {
                  event.preventDefault();
                  const row = rows[active];
                  if (row) choose(row.value);
                }
              }}
              placeholder={searchPlaceholder}
              aria-label={searchPlaceholder}
              autoComplete="off"
              className="h-8 w-full rounded-md bg-transparent pr-2 pl-7 text-sm outline-none placeholder:text-muted-foreground"
            />
          </div>

          {rows.length === 0 ? (
            <p className="px-2 py-6 text-center text-sm text-muted-foreground">{emptyText}</p>
          ) : (
            <ul ref={listRef} id={listId} role="listbox" aria-label={ariaLabel} className="max-h-72 overflow-y-auto pt-1">
              {rows.map((row, index) => {
                const isSelected = row.value === value;
                return (
                  <li
                    key={row.value ?? "__clear__"}
                    role="option"
                    aria-selected={isSelected}
                    data-active={index === active}
                    onMouseMove={() => setActive(index)}
                    onClick={() => choose(row.value)}
                    className={cn(
                      "flex cursor-pointer items-center gap-2 rounded-md px-2 py-1.5 text-sm",
                      index === active && "bg-muted",
                      row.value === null && "text-muted-foreground",
                    )}
                  >
                    {row.value === null ? (
                      <X className="size-3.5 shrink-0" aria-hidden="true" />
                    ) : (
                      <Check className={cn("size-3.5 shrink-0", !isSelected && "invisible")} aria-hidden="true" />
                    )}
                    <span className={cn("min-w-0 flex-1 truncate", row.value !== null && "text-[13px]")}>
                      {row.label}
                    </span>
                    {row.detail ? (
                      <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">{row.detail}</span>
                    ) : null}
                  </li>
                );
              })}
              {matches.length > MAX_RENDERED ? (
                <li className="px-2 py-1.5 text-center text-xs text-muted-foreground" aria-hidden="true">
                  {MAX_RENDERED} de {matches.length}: sigue escribiendo para acotar.
                </li>
              ) : null}
            </ul>
          )}
        </PopoverPrimitive.Content>
      </PopoverPrimitive.Portal>
    </PopoverPrimitive.Root>
  );
}
