import { ArrowUpRight, Brain, Info, Play } from "lucide-react";
import { Link } from "react-router-dom";

import type { AlertRecord, AlertThresholdInfo } from "../../api/types";
import { SourceBadge, StatusBadge } from "../Badges";

function formatTime(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function AlertDetails({
  alert,
  threshold,
  busy,
  onAcknowledge,
  onClear,
}: {
  alert: AlertRecord | null;
  threshold: AlertThresholdInfo | null;
  busy: boolean;
  onAcknowledge: (alertId: string) => void;
  onClear: (alertId: string) => void;
}) {
  if (!alert) {
    return (
      <aside className="alert-detail" aria-label="Alert details">
        <div className="alert-detail__empty">
          Select an alert episode to review its lifecycle, risk score and linked
          replay evidence.
        </div>
      </aside>
    );
  }

  const riskIsAboveThreshold = alert.peak_probability >= alert.threshold;
  const replayHref = `/replay?patient=${encodeURIComponent(alert.patient_id)}&hour=${alert.first_hour}`;
  const explanationHref = `/explanations?patient=${encodeURIComponent(alert.patient_id)}&hour=${alert.first_hour}`;

  return (
    <aside className="alert-detail" aria-label="Selected alert details">
      <div className="alert-detail__top">
        <div>
          <div className="alert-detail__patient">{alert.patient_id}</div>
          <div className="alert-detail__id">{alert.alert_id}</div>
        </div>
        <StatusBadge status={alert.status} />
      </div>

      <div className="alert-detail__score">
        <strong>{alert.peak_probability.toFixed(3)}</strong>
        <span>Peak risk probability</span>
        {riskIsAboveThreshold ? (
          <span className="risk-badge risk-badge--high">
            Above configured threshold
          </span>
        ) : (
          <span className="risk-badge risk-badge--low">
            Below configured threshold
          </span>
        )}
      </div>

      <dl className="alert-detail__facts">
        <div>
          <dt>Trigger hour</dt>
          <dd>Hour {alert.first_hour}</dd>
        </div>
        <div>
          <dt>Episode range</dt>
          <dd>
            {alert.first_hour}–{alert.last_hour} h
          </dd>
        </div>
        <div>
          <dt>Threshold</dt>
          <dd className="numeric">{alert.threshold.toFixed(2)}</dd>
        </div>
        <div>
          <dt>Occurrences</dt>
          <dd className="numeric">{alert.occurrence_count}</dd>
        </div>
        <div>
          <dt>Rule</dt>
          <dd>{alert.rule_id}</dd>
        </div>
        <div>
          <dt>Threshold source</dt>
          <dd>{alert.threshold_source}</dd>
        </div>
        <div>
          <dt>Created</dt>
          <dd>{formatTime(alert.created_at)}</dd>
        </div>
        <div>
          <dt>Last updated</dt>
          <dd>{formatTime(alert.updated_at)}</dd>
        </div>
      </dl>

      <div className="row" style={{ gap: 6 }}>
        <SourceBadge source={alert.prediction_source} />
        <span className="tiny muted">
          {alert.model_name} · v{alert.model_version}
        </span>
      </div>

      {alert.cleared_reason ? (
        <p className="inline-success">
          Cleared reason: <strong>{alert.cleared_reason}</strong>
        </p>
      ) : null}

      <div className="alert-detail__links">
        <Link className="detail-link" to={replayHref}>
          <Play size={16} aria-hidden="true" />
          <span>Open associated replay hour</span>
          <ArrowUpRight size={14} aria-hidden="true" />
        </Link>
        <Link className="detail-link" to={explanationHref}>
          <Brain size={16} aria-hidden="true" />
          <span>Open explanation for this hour</span>
          <ArrowUpRight size={14} aria-hidden="true" />
        </Link>
      </div>

      <div className="row-actions">
        <button
          type="button"
          className="btn btn--quiet btn--sm"
          onClick={() => onAcknowledge(alert.alert_id)}
          disabled={busy || alert.status !== "ACTIVE"}
        >
          Acknowledge alert
        </button>
        <button
          type="button"
          className="btn btn--danger btn--sm"
          onClick={() => onClear(alert.alert_id)}
          disabled={busy || alert.status === "CLEARED"}
        >
          Clear alert
        </button>
      </div>

      {threshold?.disclosure ? (
        <p className="muted tiny">{threshold.disclosure}</p>
      ) : null}

      <p className="chart-footnote">
        <Info size={13} aria-hidden="true" />
        {alert.is_synthetic
          ? "Synthetic demonstration episode. This is a workflow flag, not a diagnosis."
          : "Research workflow flag. Not a diagnosis or treatment recommendation."}
      </p>
    </aside>
  );
}
