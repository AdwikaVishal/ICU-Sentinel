import type { ReactNode } from "react";

import { SectionCard } from "./common/SectionCard";

export {
  EmptyState as EmptyPanel,
  ErrorState as ErrorPanel,
  LoadingState as LoadingPanel,
} from "./common/StateViews";

/**
 * Legacy section wrapper retained for the original research views.
 * New pages use SectionCard directly.
 */
export function Section({
  title,
  description,
  actions,
  children,
  id,
}: {
  title: string;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
}) {
  return (
    <SectionCard title={title} description={description} actions={actions} id={id}>
      {children}
    </SectionCard>
  );
}
