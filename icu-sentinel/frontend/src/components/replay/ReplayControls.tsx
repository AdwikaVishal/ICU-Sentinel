import {
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
} from "lucide-react";

export function ReplayControls({
  hour,
  minHour,
  maxHour,
  totalHours,
  playing,
  busy,
  navigating,
  evaluateAlerts,
  missingHour,
  onPrevious,
  onNext,
  onPlayToggle,
  onRestart,
  onSeek,
  onEvaluateToggle,
  onEvaluateNow,
}: {
  hour: number | null;
  minHour: number | null;
  maxHour: number | null;
  totalHours: number;
  playing: boolean;
  busy: boolean;
  navigating: boolean;
  evaluateAlerts: boolean;
  missingHour: boolean;
  onPrevious: () => void;
  onNext: () => void;
  onPlayToggle: () => void;
  onRestart: () => void;
  onSeek: (hour: number) => void;
  onEvaluateToggle: (value: boolean) => void;
  onEvaluateNow: () => void;
}) {
  const first = minHour ?? 0;
  const last = maxHour ?? first;
  const span = Math.max(last - first, 1);
  const progress =
    hour === null ? 0 : Math.min(100, Math.max(0, ((hour - first) / span) * 100));

  return (
    <section className="replay-controls" aria-labelledby="replay-controls-title">
      <div className="replay-controls__top">
        <div className="replay-hour">
          <span className="replay-hour__label" id="replay-controls-title">
            Replay position
          </span>
          <span className="replay-hour__value">
            Hour {hour ?? "—"}{" "}
            <span>/ {totalHours > 0 ? totalHours : "—"}</span>
          </span>
        </div>

        <div className="replay-buttons">
          <button
            type="button"
            className="icon-button"
            onClick={onRestart}
            disabled={busy || hour === first}
            aria-label="Restart replay at the first hour"
            title="Restart"
          >
            <RotateCcw size={17} />
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={onPrevious}
            disabled={busy || hour === null || hour <= first}
            aria-label="Previous hour"
            title="Previous hour"
          >
            <ChevronLeft size={19} />
          </button>
          <button
            type="button"
            className="replay-play"
            onClick={onPlayToggle}
            disabled={busy || hour === null || last === first}
            aria-label={playing ? "Pause replay" : "Play replay"}
            title={playing ? "Pause" : "Play"}
          >
            {playing ? <Pause size={18} /> : <Play size={18} />}
          </button>
          <button
            type="button"
            className="icon-button"
            onClick={onNext}
            disabled={busy || hour === null || hour >= last}
            aria-label="Next hour"
            title="Next hour"
          >
            <ChevronRight size={19} />
          </button>
        </div>
      </div>

      <div className="slider-block">
        <div className="slider-block__labels">
          <span>Hour {first}</span>
          <span>Hour {last}</span>
        </div>
        <input
          type="range"
          className="slider"
          min={first}
          max={last}
          step={1}
          value={hour ?? first}
          style={{ "--progress": `${progress}%` } as React.CSSProperties}
          onChange={(event) => onSeek(Number(event.target.value))}
          disabled={busy || hour === null || last === first}
          aria-label="Replay hour"
          aria-valuetext={`Hour ${hour ?? first} of ${last}`}
        />
      </div>

      <div className="replay-controls__footer">
        <span>
          {navigating
            ? "Updating replay hour…"
            : missingHour
              ? "This hour has no recorded observation row."
              : "Prediction uses observations up to this hour only."}
        </span>
        <label className="checkbox-pill">
          <input
            type="checkbox"
            checked={evaluateAlerts}
            onChange={(event) => onEvaluateToggle(event.target.checked)}
          />
          <span>Evaluate alert rule</span>
        </label>
        {evaluateAlerts ? (
          <button
            type="button"
            className="btn btn--quiet btn--sm"
            onClick={onEvaluateNow}
            disabled={busy || hour === null}
          >
            Evaluate this hour
          </button>
        ) : null}
      </div>
    </section>
  );
}
