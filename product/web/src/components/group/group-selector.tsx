"use client";

import { useMemo, useState } from "react";

import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { GroupOption } from "@/lib/data/group-service";

function optionLabel(option: GroupOption) {
  const mean = option.meanScore === null ? "sin media" : `media ${option.meanScore.toFixed(0)}`;
  const companies = option.nCompanies === 1 ? "1 empresa" : `${option.nCompanies} empresas`;
  return `${option.groupId} · ${companies} · ${mean}`;
}

export function GroupSelector({
  options,
  value,
  onChange,
}: {
  options: GroupOption[];
  value: string;
  onChange: (groupId: string) => void;
}) {
  const [filter, setFilter] = useState("");

  const visible = useMemo(() => {
    const needle = filter.trim().toLowerCase();
    if (!needle) return options;
    return options.filter(
      (option) => option.groupId === value || option.groupId.toLowerCase().includes(needle),
    );
  }, [filter, options, value]);

  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
      <Input
        value={filter}
        onChange={(event) => setFilter(event.target.value)}
        placeholder="Filtrar grupos (p. ej. 0142)"
        aria-label="Filtrar grupos"
        className="sm:w-56"
      />
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger aria-label="Grupo" className="w-full bg-background font-mono text-xs sm:w-80">
          <SelectValue />
        </SelectTrigger>
        <SelectContent position="popper" align="start" className="max-h-80">
          {visible.map((option) => (
            <SelectItem key={option.groupId} value={option.groupId} className="font-mono text-xs">
              {optionLabel(option)}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      <span className="text-xs text-muted-foreground">
        {visible.length} de {options.length} grupos
      </span>
    </div>
  );
}
