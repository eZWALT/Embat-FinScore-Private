/// <reference types="react/experimental" />
import "react";

declare module "react" {
  /** Exported by the React that Next.js bundles (see the "View transitions" guide); @types/react 19.1 only types it as `unstable_ViewTransition`. */
  export const ViewTransition: ExoticComponent<ViewTransitionProps>;
}
