import type { AlertRecord } from "../../api/types";
import { StatusBadge } from "../Badges";
import { EmptyState } from "../common/StateViews";

function formatTime(value: string): string {
  return new Date(value).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function AlertTable({
  alerts,
  selectedId,
  busyId,
  onSelect,
  onAcknowledge,
  onClear,
}: {
  alerts: AlertRecord[];
  selectedId: string | null;
  busyId: string | null;
  onSelect: (alert: AlertRecord) => void;
  onAcknowledge: (alertId: string) => void;
  onClear: (alertId: string) => void;
}) {
  if (alerts.length === 0) {
    return (
      <EmptyState title="No alert episodes match this view">
        An empty list is not evidence that a patient is stable. Change the filter
        or evaluate a replay hour to generate episodes.
      </EmptyState>
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">
          Alert episodes with patient, status, probability, trigger hour,
          threshold and lifecycle actions.
        </caption>
        <thead>
          <tr>
            <th scope="col">Patient</th>
            <th scope="col">Status</th>
            <th scope="col">Risk probability</th>
            <th scope="col">Triggered</th>
            <th scope="col">Threshold</th>
            <th scope="col">Last updated</th>
            <th scope="col">Actions</th>
          </tr>
        </thead>
        <tbody>
          {alerts.map((alert) => {
            const busy = busyId === alert.alert_id;
            const selected = selectedId === alert.alert_id;
            return (
              <tr
                key={alert.alert_id}
                className={selected ? "is-selected" : undefined}
                onClick={() => onSelect(alert)}
              >
                <th scope="row">
                  <button
                    type="button"
                    className="btn btn--ghost btn--sm"
                    style={{ padding: 0, minHeight: 0, fontWeight: 650 }}
                    onClick={(event) => {
                      event.stopPropagation();
                      onSelect(alert);
                    }}
                  >
                    {alert.patient_id}
                  </button>
                  {alert.is_synthetic ? (
                    <span className="badge badge--synthetic">Synthetic</span>
                  ) : null}
                </th>
                <td>
                  <StatusBadge status={alert.status} />
                </td>
                <td className="numeric">
                  {alert.peak_probability.toFixed(3)}
                  <span className="tiny muted">
                    {" "}
                    / {alert.threshold.toFixed(2)}
                  </span>
                </td>
                <td>
                  Hour {alert.first_hour}
                  {alert.last_hour !== alert.first_hour
                    ? `–${alert.last_hour}`
                    : ""}
                  <span className="tiny muted">
                    {" "}
                    · occurrences {alert.occurrence_count}
                  </span>
                </td>
                <td className="numeric">{alert.threshold.toFixed(2)}</td>
                <td>{formatTime(alert.updated_at)}</td>
                <td>
                  <div className="row-actions">
                    <button
                      type="button"
                      className="btn btn--quiet btn--sm"
                      onClick={(event) => {
                        event.stopPropagation();
                        onAcknowledge(alert.alert_id);
                      }}
                      disabled={busy || alert.status !== "ACTIVE"}
                    >
                      {busy ? "Saving…" : "Acknowledge"}
                    </button>
                    <button
                      type="button"
                      className="btn btn--quiet btn--sm"
                      onClick={(event) => {
                        event.stopPropagation();
                        onClear(alert.alert_id);
                      }}
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
