import type { AlertRecord, PredictionResult, ReplayState } from "../api/types";
import { StatusBadge, SourceBadge } from "./Badges";

function formatScore(value: number | null): string {
  return value === null ? "—" : value.toFixed(3);
}

function formatClass(value: number | null): string {
  return value === null ? "—" : String(value);
}

export function PredictionPanel({ state }: { state: ReplayState }) {
  const alert: AlertRecord | null = state.alert;
  const predictions: PredictionResult[] = state.predictions.length > 0
    ? state.predictions
    : [state.prediction];

  return (
    <div className="prediction">
      {predictions.map((prediction) => (
        <div key={prediction.model.name} className="prediction__card">
          <div className="prediction__header">
            <span className="prediction__label">
              {prediction.model.display_name ?? prediction.model.name}
            </span>
            <SourceBadge source={prediction.source} />
          </div>

          <div className="prediction__score">
            <span className="prediction__value">
              {prediction.available ? formatScore(prediction.probability) : "—"}
            </span>
            <span className="prediction__caption">
              {prediction.available
                ? "Estimated probability-like score for this hour. Not a diagnosis."
                : prediction.unavailable_reason ?? "Prediction unavailable for this hour."}
            </span>
          </div>

          <dl className="facts">
            <div>
              <dt>Prediction status</dt>
              <dd>{state.prediction_status.replaceAll("_", " ")}</dd>
            </div>
            <div>
              <dt>Configured threshold</dt>
              <dd>
                {prediction.threshold !== null ? formatScore(prediction.threshold) : "—"}
              </dd>
            </div>
            <div>
              <dt>Predicted class</dt>
              <dd>{formatClass(prediction.predicted_class)}</dd>
            </div>
            <div>
              <dt>Model version</dt>
              <dd>
                {prediction.model.version}
              </dd>
            </div>
            <div>
              <dt>Explanation</dt>
              <dd>{state.explanation_available ? "Available" : "Not available"}</dd>
            </div>
          </dl>

          {state.alert_error ? (
            <p className="inline-error" role="alert">
              Alert evaluation could not complete: {state.alert_error}
            </p>
          ) : null}
        </div>
      ))}

      <dl className="facts facts--shared">
        <div>
          <dt>Alert evaluation</dt>
          <dd>
            {state.alert_status.replaceAll("_", " ")}
            {alert ? (
              <>
                {" "}
                <StatusBadge status={alert.status} />{" "}
                <span className="muted small">
                  episode {alert.first_hour}–{alert.last_hour}
                </span>
              </>
            ) : null}
          </dd>
        </div>
        <div>
          <dt>Retrospective label</dt>
          <dd>
            {state.retrospective_label_available
              ? state.retrospective_label === 1
                ? "Sepsis label present (evaluation only)"
                : "No sepsis label (evaluation only)"
              : "Not available"}
          </dd>
        </div>
      </dl>

      <p className="muted small">{state.label_disclosure}</p>
    </div>
  );
}
