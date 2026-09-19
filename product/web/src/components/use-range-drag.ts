"use client";

import { useEffect, useRef, useState } from "react";

export interface MonthRange {
  from: string;
  to: string;
}

type ChartState = { activeTooltipIndex?: number | string | null; activeLabel?: string | number | null };
type Drag = { anchor: string; head: string } | null;

/**
 * Press and drag across the months of a recharts chart to select a period. `months` are the chart's x values in row
 * order. Spread `handlers` on the chart and shade `shaded` (the drag in progress, else the committed `range`).
 * Releasing on one month clears the selection.
 */
export function useRangeDrag(
  months: string[],
  range: MonthRange | null,
  onRangeChange: (range: MonthRange | null) => void,
) {
  const [drag, setDragState] = useState<Drag>(null);
  const dragRef = useRef<Drag>(null);
  // The ref lets the mouse handlers see the latest drag before React has re-rendered.
  function setDrag(next: Drag) {
    dragRef.current = next;
    setDragState(next);
  }

  function monthAt(state: ChartState) {
    const index = Number(state.activeTooltipIndex);
    if (Number.isInteger(index) && months[index]) return months[index];
    return state.activeLabel != null ? String(state.activeLabel) : null;
  }

  function start(state: ChartState) {
    const month = monthAt(state);
    if (month) setDrag({ anchor: month, head: month });
  }

  function move(state: ChartState) {
    if (!dragRef.current) return;
    const month = monthAt(state);
    if (month && month !== dragRef.current.head) setDrag({ ...dragRef.current, head: month });
  }

  function finish() {
    const current = dragRef.current;
    if (!current) return;
    setDrag(null);
    if (current.anchor === current.head) {
      onRangeChange(null);
      return;
    }
    const [from, to] = [current.anchor, current.head].sort();
    onRangeChange({ from, to });
  }

  // Releasing the button outside the plot area must still end the selection.
  useEffect(() => {
    if (!drag) return;
    window.addEventListener("mouseup", finish);
    return () => window.removeEventListener("mouseup", finish);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drag === null]);

  const shaded = drag
    ? ([drag.anchor, drag.head].sort() as [string, string])
    : range
      ? ([range.from, range.to] as [string, string])
      : null;

  return {
    shaded,
    handlers: {
      onMouseDown: start,
      onMouseMove: move,
      onMouseUp: finish,
      onTouchStart: start,
      onTouchMove: move,
      onTouchEnd: finish,
    },
  };
}
