import type { ReactNode } from "react";

export function MetricCard({
  icon,
  value,
  label,
  hint,
  tone = "green",
}: {
  icon: ReactNode;
  value: string;
  label: string;
  hint?: string;
  tone?: "green" | "cream" | "info" | "lilac";
}) {
  return (
    <article className="metric-card">
      <span
        className={`metric-card__icon metric-card__icon--${tone}`}
        aria-hidden="true"
      >
        {icon}
      </span>
      <div className="metric-card__body">
        <span className="metric-card__value">{value}</span>
        <span className="metric-card__label">{label}</span>
        {hint ? <span className="metric-card__hint">{hint}</span> : null}
      </div>
    </article>
  );
}
