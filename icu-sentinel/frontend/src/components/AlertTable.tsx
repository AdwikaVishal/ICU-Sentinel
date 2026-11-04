import type { AlertRecord, AlertThresholdInfo } from "../api/types";
import { StatusBadge } from "./Badges";

export function ThresholdPanel({
  threshold,
}: {
  threshold: AlertThresholdInfo | null;
}) {
  if (!threshold) {
    return (
      <p className="muted">
        The alert threshold is not available from the configured alert engine.
      </p>
    );
  }

  return (
    <div className="threshold">
      <div className="threshold__value">
        <span className="prediction__label">Configured threshold</span>
        <span className="prediction__value">{threshold.threshold.toFixed(2)}</span>
        <span className="badge badge--muted">
          {threshold.clinically_validated ? "Clinically validated" : "Not clinically validated"}
        </span>
      </div>
      <dl className="facts">
        <div>
          <dt>Rule</dt>
          <dd>{threshold.rule_id}</dd>
        </div>
        <div>
          <dt>Source / version</dt>
          <dd>
            {threshold.source} · {threshold.version}
          </dd>
        </div>
        <div>
          <dt>Deduplication window</dt>
          <dd>{threshold.dedup_window_hours} h</dd>
        </div>
        <div>
          <dt>Clear below threshold</dt>
          <dd>{threshold.clear_on_below_threshold ? "Enabled" : "Disabled"}</dd>
        </div>
      </dl>
      <p className="muted small">{threshold.disclosure}</p>
    </div>
  );
}

export function AlertTable({
  alerts,
  emptyTitle,
  busyId,
  onAcknowledge,
  onClear,
}: {
  alerts: AlertRecord[];
  emptyTitle: string;
  busyId: string | null;
  onAcknowledge: (alertId: string) => void;
  onClear: (alertId: string) => void;
}) {
  if (alerts.length === 0) {
    return (
      <div className="state-panel state-panel--empty">
        <div>
          <strong>{emptyTitle}</strong>
          <p>
            No alert episodes match this view. An empty list is not evidence that a
            patient is stable.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th scope="col">Patient</th>
            <th scope="col">Episode</th>
            <th scope="col">Peak score</th>
            <th scope="col">Threshold</th>
            <th scope="col">Occurrences</th>
            <th scope="col">Status</th>
            <th scope="col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert) => {
            const busy = busyId === alert.alert_id;
            return (
              <tr key={alert.alert_id}>
                <th scope="row">
                  {alert.patient_id}
                  {alert.is_synthetic ? (
                    <span className="badge badge--synthetic">Synthetic</span>
                  ) : null}
                </th>
                <td>
                  {alert.first_hour}–{alert.last_hour} h
                  <span className="muted small"> · updated {alert.last_hour} h</span>
                </td>
                <td className="numeric">{alert.peak_probability.toFixed(3)}</td>
                <td className="numeric">{alert.threshold.toFixed(2)}</td>
                <td className="numeric">{alert.occurrence_count}</td>
                <td>
                  <StatusBadge status={alert.status} />
                  {alert.cleared_reason ? (
                    <div className="muted small">{alert.cleared_reason}</div>
                  ) : null}
                </td>
                <td>
                  <div className="row-actions">
                    <button
                      type="button"
                      className="button button--quiet"
                      onClick={() => onAcknowledge(alert.alert_id)}
                      disabled={busy || alert.status !== "ACTIVE"}
                    >
                      Acknowledge
                    </button>
                    <button
                      type="button"
                      className="button button--quiet"
                      onClick={() => onClear(alert.alert_id)}
                      disabled={busy || alert.status === "CLEARED"}
                    >
                      Clear
                    </button>
                  </div>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
