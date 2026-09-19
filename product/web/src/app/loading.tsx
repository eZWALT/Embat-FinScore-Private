import { ViewTransition } from "react";

import { AppHeader } from "@/components/app-header";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * What shows while the dashboard loads: the app's own header (so the frame does not change when the data arrives) and a
 * skeleton with the shape of Rápido. It fades out when the page is ready.
 */
export default function Loading() {
  return (
    <ViewTransition exit="route-out" default="none">
      <div className="flex min-h-svh flex-col" aria-busy="true">
        <AppHeader />
        <div className="mx-auto flex w-full max-w-[96rem] flex-1 flex-col px-4 pt-5 sm:px-6 lg:px-10">
          <div className="grid gap-3 sm:grid-cols-[minmax(0,2.4fr)_minmax(0,1fr)_minmax(0,1fr)]">
            <Skeleton className="h-[7.5rem] rounded-xl" />
            <Skeleton className="hidden h-[7.5rem] rounded-xl sm:block" />
            <Skeleton className="hidden h-[7.5rem] rounded-xl sm:block" />
          </div>
          <div className="mt-5 grid gap-4 lg:grid-cols-[19rem_minmax(0,1fr)]">
            <div className="hidden space-y-3 lg:block">
              <Skeleton className="h-7 w-2/3" />
              <Skeleton className="h-[min(46vh,420px)] rounded-lg" />
            </div>
            <div className="space-y-2">
              <Skeleton className="ml-auto h-8 w-64" />
              <Skeleton className="h-[min(52vh,440px)] rounded-xl" />
            </div>
          </div>
        </div>
      </div>
    </ViewTransition>
  );
}
