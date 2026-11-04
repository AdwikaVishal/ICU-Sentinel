import { Activity, Info } from "lucide-react";

import type { AlertThresholdInfo, ReplayState } from "../../api/types";
import { SourceBadge, StatusBadge } from "../Badges";

export function RiskScoreCard({
  state,
  threshold,
}: {
  state: ReplayState;
  threshold: AlertThresholdInfo | null;
}) {
  const prediction = state.prediction;
  const probability = prediction.available ? prediction.probability : null;
  const configuredThreshold = threshold?.threshold ?? prediction.threshold;
  const riskPercent =
    probability === null ? 0 : Math.min(100, Math.max(0, probability * 100));
  const aboveThreshold =
    probability !== null &&
    configuredThreshold !== null &&
    probability >= configuredThreshold;

  const riskColor =
    probability === null
      ? "var(--muted-light)"
      : aboveThreshold
        ? "var(--critical)"
        : "var(--green-700)";

  return (
    <section className="risk-card" aria-labelledby="risk-score-title">
      <div className="risk-card__top">
        <div>
          <h2 id="risk-score-title">Risk Score</h2>
          <p>Current hourly prediction</p>
        </div>
        <span className="badge badge--soft-green">
          <Activity size={12} aria-hidden="true" />
          Hour {state.hour}
        </span>
      </div>

      <div
        className="risk-donut"
        style={
          {
            "--risk-value": riskPercent,
            "--risk-color": riskColor,
          } as React.CSSProperties
        }
        role="img"
        aria-label={
          probability === null
            ? "Risk probability unavailable"
            : `Risk probability ${probability.toFixed(3)}`
        }
      >
        <div className="risk-donut__inner">
          <span className="risk-donut__value">
            {probability === null ? "—" : probability.toFixed(3)}
          </span>
          <span className="risk-donut__label">
            {probability === null ? "Unavailable" : "Probability"}
          </span>
        </div>
      </div>

      <div className="risk-card__badges">
        {probability === null ? (
          <span className="risk-badge risk-badge--unavailable">
            Prediction unavailable
          </span>
        ) : aboveThreshold ? (
          <span className="risk-badge risk-badge--high">
            Above configured threshold
          </span>
        ) : (
          <span className="risk-badge risk-badge--low">
            Below configured threshold
          </span>
        )}
        {state.alert ? <StatusBadge status={state.alert.status} /> : null}
      </div>

      <dl className="risk-facts">
        <div>
          <dt>Model score</dt>
          <dd className="numeric">
            {probability === null ? "—" : probability.toFixed(4)}
          </dd>
        </div>
        <div>
          <dt>Threshold</dt>
          <dd className="numeric">
            {configuredThreshold === null ? "—" : configuredThreshold.toFixed(2)}
          </dd>
        </div>
        <div>
          <dt>Alert evaluation</dt>
          <dd>{state.alert_status.replaceAll("_", " ")}</dd>
        </div>
        <div>
          <dt>Explanation</dt>
          <dd>{state.explanation_available ? "Available" : "Not available"}</dd>
        </div>
        <div>
          <dt>Provider</dt>
          <dd>
            {prediction.model.display_name ?? prediction.model.name}{" "}
            <span className="muted tiny">v{prediction.model.version}</span>
          </dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>
            <SourceBadge source={prediction.source} />
          </dd>
        </div>
      </dl>

      {probability === null && prediction.unavailable_reason ? (
        <p className="inline-error" role="status" style={{ marginTop: 12 }}>
          {prediction.unavailable_reason}
        </p>
      ) : null}

      {state.alert_error ? (
        <p className="inline-error" role="alert" style={{ marginTop: 12 }}>
          Alert evaluation could not complete: {state.alert_error}
        </p>
      ) : null}

      <p className="chart-footnote">
        <Info size={13} aria-hidden="true" />
        A model score is not a diagnosis and does not confirm sepsis. The
        configured threshold is a demonstration value, not a validated clinical
        cut-off.
      </p>
    </section>
  );
}
