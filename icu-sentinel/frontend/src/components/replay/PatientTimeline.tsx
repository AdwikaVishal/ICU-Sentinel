import type { AlertRecord, TimelineResponse } from "../../api/types";

export function PatientTimeline({
  timeline,
  hour,
  alerts,
  onSelect,
  busy,
}: {
  timeline: TimelineResponse;
  hour: number | null;
  alerts: AlertRecord[];
  onSelect: (hour: number) => void;
  busy?: boolean;
}) {
  const hours = timeline.hours;
  if (hours.length === 0) {
    return (
      <div className="timeline__labels">
        <span>No recorded timeline hours are available.</span>
      </div>
    );
  }

  const first = hours[0].hour;
  const last = hours[hours.length - 1].hour;

  const alertHours = new Set<number>();
  for (const alert of alerts) {
    for (let value = alert.first_hour; value <= alert.last_hour; value += 1) {
      alertHours.add(value);
    }
  }

  return (
    <div className="timeline">
      <div className="timeline__track-wrap">
        <div
          className="timeline__track"
          role="group"
          aria-label="Hourly patient timeline"
        >
          {hours.map((item) => {
            const selected = item.hour === hour;
            const hasAlert = alertHours.has(item.hour);
            const label = `Hour ${item.hour}: ${
              item.has_observations
                ? `${item.available_observation_count} recorded observations`
                : "no recorded observation row"
            }${hasAlert ? ", inside an alert episode" : ""}`;
            return (
              <button
                key={item.hour}
                type="button"
                className={[
                  "timeline__hour",
                  item.has_observations ? "" : "timeline__hour--no-data",
                  selected ? "timeline__hour--selected" : "",
                  hasAlert ? "timeline__hour--alert" : "",
                ]
                  .filter(Boolean)
                  .join(" ")}
                title={label}
                aria-label={label}
                aria-pressed={selected}
                onClick={() => onSelect(item.hour)}
                disabled={busy}
              />
            );
          })}
        </div>
        <div className="timeline__labels">
          <span>ICU admission · hour {first}</span>
          <span>Latest recorded hour {last}</span>
        </div>
      </div>

      <div className="timeline__events">
        <span>
          <span className="timeline__legend-dot" aria-hidden="true" />
          Recorded hour
        </span>
        <span>
          <span
            className="timeline__legend-dot timeline__legend-dot--selected"
            aria-hidden="true"
          />
          Selected hour
        </span>
        <span>
          <span
            className="timeline__legend-dot timeline__legend-dot--alert"
            aria-hidden="true"
          />
          Alert event
        </span>
        <span>
          <span
            className="timeline__legend-dot"
            style={{
              background:
                "repeating-linear-gradient(135deg,#f0f3ee,#f0f3ee 2px,#fafbf9 2px,#fafbf9 4px)",
            }}
            aria-hidden="true"
          />
          No observation row
        </span>
      </div>
    </div>
  );
}
