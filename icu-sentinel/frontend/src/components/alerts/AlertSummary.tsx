import { BellRing, CheckCircle2, Eye, Layers } from "lucide-react";

export function AlertSummary({
  active,
  acknowledged,
  cleared,
  total,
}: {
  active: number | null;
  acknowledged: number | null;
  cleared: number | null;
  total: number | null;
}) {
  const cards = [
    {
      label: "Active Alerts",
      value: active,
      icon: BellRing,
      tone: "",
    },
    {
      label: "Acknowledged",
      value: acknowledged,
      icon: Eye,
      tone: "alert-summary-card__icon--warning",
    },
    {
      label: "Cleared",
      value: cleared,
      icon: CheckCircle2,
      tone: "alert-summary-card__icon--success",
    },
    {
      label: "Total Episodes",
      value: total,
      icon: Layers,
      tone: "alert-summary-card__icon--neutral",
    },
  ];

  return (
    <div className="alert-summary-grid">
      {cards.map(({ label, value, icon: Icon, tone }) => (
        <article className="alert-summary-card" key={label}>
          <span className={`alert-summary-card__icon ${tone}`} aria-hidden="true">
            <Icon size={19} />
          </span>
          <div>
            <span className="alert-summary-card__value">
              {value === null ? "—" : value.toLocaleString("en-US")}
            </span>
            <span className="alert-summary-card__label">{label}</span>
          </div>
        </article>
      ))}
    </div>
  );
}
