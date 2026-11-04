import { Activity, FlaskConical, HeartPulse, UserRound } from "lucide-react";

import type { ObservationValue } from "../../api/types";
import { EmptyState } from "../common/StateViews";

const UNIT_LABELS: Record<string, string> = {
  beats_per_minute: "bpm",
  breaths_per_minute: "/min",
  percent: "%",
  deg_c: "°C",
  mm_hg: "mmHg",
  mmol_per_l: "mmol/L",
  mmol_per_dl: "mmol/dL",
  mg_per_dl: "mg/dL",
  iu_per_l: "IU/L",
  ng_per_ml: "ng/mL",
  g_per_dl: "g/dL",
  count_times_10_3_per_ul: "×10³/µL",
  seconds: "s",
  years: "years",
  hours: "h",
  unitless: "",
  code: "",
};

const CATEGORY_META: Record<
  string,
  { label: string; icon: typeof HeartPulse; order: number }
> = {
  vital_signs: { label: "Vital signs", icon: HeartPulse, order: 0 },
  laboratory: { label: "Laboratory", icon: FlaskConical, order: 1 },
  static_demographic: { label: "Patient profile", icon: UserRound, order: 2 },
};

function formatValue(observation: ObservationValue): string {
  if (observation.value === null) return "Missing";
  const value = observation.value;
  if (observation.parameter.toLowerCase().includes("ph")) {
    return value.toFixed(2);
  }
  if (
    observation.unit === "code" ||
    observation.unit === "years" ||
    observation.unit === "hours"
  ) {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }
  if (Math.abs(value) >= 100) return value.toFixed(0);
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(1);
}

function formatUnit(observation: ObservationValue): string {
  if (!observation.unit) return "";
  const label = UNIT_LABELS[observation.unit] ?? observation.unit;
  return label ? ` ${label}` : "";
}

export function ClinicalObservationGrid({
  observations,
}: {
  observations: ObservationValue[];
}) {
  if (observations.length === 0) {
    return (
      <EmptyState
        title="No observation row for this hour"
        icon={<Activity size={18} />}
      >
        The replay remains valid, but this hour has no recorded measurements.
        Missing values are never treated as normal values.
      </EmptyState>
    );
  }

  const groups = new Map<string, ObservationValue[]>();
  for (const observation of observations) {
    const category = observation.category || "other";
    const list = groups.get(category) ?? [];
    list.push(observation);
    groups.set(category, list);
  }

  const ordered = [...groups.entries()].sort(([a], [b]) => {
    const orderA = CATEGORY_META[a]?.order ?? 99;
    const orderB = CATEGORY_META[b]?.order ?? 99;
    return orderA - orderB;
  });

  return (
    <div className="observation-groups">
      {ordered.map(([category, items]) => {
        const meta = CATEGORY_META[category] ?? {
          label: category.replaceAll("_", " "),
          icon: Activity,
          order: 99,
        };
        const Icon = meta.icon;
        const available = items.filter((item) => item.available).length;
        return (
          <section key={category} aria-labelledby={`obs-${category}`}>
            <h3 className="observation-group__header" id={`obs-${category}`}>
              <Icon size={15} aria-hidden="true" />
              {meta.label}
              <span className="observation-group__count">
                {available}/{items.length} recorded
              </span>
            </h3>
            <div className="observation-grid">
              {items.map((observation) => (
                <article
                  key={observation.parameter}
                  className={`observation-tile${
                    observation.available ? "" : " observation-tile--missing"
                  }`}
                >
                  <span className="observation-tile__name" title={observation.display_name}>
                    {observation.display_name}
                  </span>
                  <span className="observation-tile__value">
                    {observation.available
                      ? formatValue(observation)
                      : "Not recorded"}
                    {observation.available ? (
                      <span className="observation-tile__unit">
                        {formatUnit(observation)}
                      </span>
                    ) : null}
                  </span>
                  <span className="observation-tile__foot">
                    <span className="observation-tile__status">
                      {observation.available ? "Recorded" : "Missing"}
                    </span>
                    <span className="tiny muted numeric">
                      {observation.parameter}
                    </span>
                  </span>
                </article>
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}
