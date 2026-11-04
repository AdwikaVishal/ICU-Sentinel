import { ArrowUpRight } from "lucide-react";
import type { ReactNode } from "react";
import { Link } from "react-router-dom";

export function KeyModuleCard({
  icon,
  title,
  description,
  cta,
  to,
  tone = "green",
  tag,
}: {
  icon: ReactNode;
  title: string;
  description: string;
  cta: string;
  to: string;
  tone?: "green" | "cream" | "lilac";
  tag: string;
}) {
  const toneClass =
    tone === "cream"
      ? " module-card--cream"
      : tone === "lilac"
        ? " module-card--lilac"
        : "";

  return (
    <Link to={to} className={`module-card${toneClass}`}>
      <div className="module-card__top">
        <span className="module-card__icon" aria-hidden="true">
          {icon}
        </span>
        <span className="module-card__tag">{tag}</span>
      </div>
      <h3>{title}</h3>
      <p>{description}</p>
      <span className="module-card__cta">
        {cta}
        <ArrowUpRight size={15} aria-hidden="true" />
      </span>
    </Link>
  );
}
