import type { ReactNode } from "react";

export function SectionCard({
  title,
  description,
  actions,
  children,
  id,
  className,
  headingLevel = 2,
}: {
  title: ReactNode;
  description?: ReactNode;
  actions?: ReactNode;
  children: ReactNode;
  id?: string;
  className?: string;
  headingLevel?: 2 | 3;
}) {
  const Heading = headingLevel === 3 ? "h3" : "h2";
  return (
    <section className={["card", className].filter(Boolean).join(" ")} id={id}>
      <header className="card__header">
        <div className="card__header-copy">
          <Heading>{title}</Heading>
          {description ? <p>{description}</p> : null}
        </div>
        {actions ? <div className="card__actions">{actions}</div> : null}
      </header>
      {children}
    </section>
  );
}
