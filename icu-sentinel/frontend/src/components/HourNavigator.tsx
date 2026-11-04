import type { TimelineResponse } from "../api/types";

export function HourNavigator({
  timeline,
  hour,
  busy,
  onSelect,
  onPrevious,
  onNext,
  onReset,
  onEnd,
}: {
  timeline: TimelineResponse;
  hour: number | null;
  busy: boolean;
  onSelect: (hour: number) => void;
  onPrevious: () => void;
  onNext: () => void;
  onReset: () => void;
  onEnd: () => void;
}) {
  const hours = timeline.hours;
  const first = hours[0]?.hour ?? 0;
  const last = hours[hours.length - 1]?.hour ?? 0;
  const currentIndex = hours.findIndex((item) => item.hour === hour);

  return (
    <div className="hour-navigator">
      <div className="hour-navigator__row">
        <button
          type="button"
          className="button button--quiet"
          onClick={onReset}
          disabled={busy || hour === first}
        >
          Start
        </button>
        <button
          type="button"
          className="button button--quiet"
          onClick={onPrevious}
          disabled={busy || currentIndex <= 0}
        >
          ← Previous
        </button>
        <label className="hour-navigator__slider">
          <span className="field__label">
            Hour {hour ?? "–"} of {first}–{last}
          </span>
          <input
            type="range"
            min={first}
            max={last}
            step={1}
            value={hour ?? first}
            aria-label="Replay hour"
            onChange={(event) => onSelect(Number(event.target.value))}
            disabled={busy || hours.length === 0}
          />
        </label>
        <button
          type="button"
          className="button button--quiet"
          onClick={onNext}
          disabled={busy || currentIndex === -1 || currentIndex >= hours.length - 1}
        >
          Next →
        </button>
        <button
          type="button"
          className="button button--quiet"
          onClick={onEnd}
          disabled={busy || hour === last}
        >
          End
        </button>
      </div>

      <div className="timeline-strip" role="group" aria-label="Recorded hours">
        {hours.map((item) => {
          const selected = item.hour === hour;
          const label = `Hour ${item.hour}: ${
            item.has_observations
              ? `${item.available_observation_count} observations`
              : "no recorded observations"
          }`;
          return (
            <button
              key={item.hour}
              type="button"
              className={[
                "timeline-strip__cell",
                item.has_observations ? "has-data" : "no-data",
                selected ? "is-selected" : "",
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
      <p className="muted small">
        Each cell is one recorded hour. Dimmed cells have no observation row; they
        are not treated as normal values.
      </p>
    </div>
  );
}
