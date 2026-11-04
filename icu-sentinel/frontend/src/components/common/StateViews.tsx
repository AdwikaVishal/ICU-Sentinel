import { AlertCircle, Inbox } from "lucide-react";
import type { ReactNode } from "react";

export function LoadingState({
  label = "Loading…",
  skeleton = false,
}: {
  label?: string;
  skeleton?: boolean;
}) {
  if (skeleton) {
    return (
      <div className="state-panel" role="status" aria-live="polite">
        <div className="skeleton-stack" aria-hidden="true">
          <div className="skeleton skeleton--wide" />
          <div className="skeleton skeleton--short" />
          <div className="skeleton" />
        </div>
        <span className="sr-only">{label}</span>
      </div>
    );
  }

  return (
    <div className="state-panel" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  );
}

export function ErrorState({
  title = "Could not load this view",
  error,
  onRetry,
}: {
  title?: string;
  error: Error | string;
  onRetry?: () => void;
}) {
  const message = typeof error === "string" ? error : error.message;
  return (
    <div className="state-panel state-panel--error" role="alert">
      <div>
        <strong>
          <AlertCircle
            size={15}
            aria-hidden="true"
            style={{ display: "inline", marginRight: 6, verticalAlign: "-2px" }}
          />
          {title}
        </strong>
        <p>{message}</p>
      </div>
      {onRetry ? (
        <button type="button" className="btn btn--quiet" onClick={onRetry}>
          Try again
        </button>
      ) : null}
    </div>
  );
}

export function EmptyState({
  title,
  children,
  icon,
}: {
  title: string;
  children?: ReactNode;
  icon?: ReactNode;
}) {
  return (
    <div className="state-panel state-panel--empty">
      <div>
        <strong>{title}</strong>
        {children ? <p>{children}</p> : null}
      </div>
      <span className="metric-card__icon metric-card__icon--info" aria-hidden="true">
        {icon ?? <Inbox size={19} />}
      </span>
    </div>
  );
}
