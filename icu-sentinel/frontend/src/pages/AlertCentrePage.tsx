import { ErrorPanel, LoadingPanel, Section } from "../components/AsyncState";
import { AlertTable, ThresholdPanel } from "../components/AlertTable";
import { useAlertCentre } from "../hooks/useExplanation";

export function AlertCentrePage() {
  const centre = useAlertCentre();

  return (
    <div className="page">
      <div className="page__heading">
        <div>
          <h2>Alert Centre</h2>
          <p className="muted">
            Alert episodes are workflow flags produced by a configurable threshold.
            They are not diagnoses and do not confirm sepsis.
          </p>
        </div>
        <button type="button" className="button button--quiet" onClick={centre.reload}>
          Refresh
        </button>
      </div>

      {centre.loading ? <LoadingPanel label="Loading alert episodes…" /> : null}
      {centre.error ? <ErrorPanel error={centre.error} onRetry={centre.reload} /> : null}

      {!centre.loading && !centre.error ? (
        <>
          <Section
            title="Threshold configuration"
            description="Configured demonstration rule. Threshold selection never uses the ground-truth label."
          >
            <ThresholdPanel threshold={centre.threshold} />
          </Section>

          {centre.actionError ? (
            <p className="inline-error" role="alert">
              Alert action failed: {centre.actionError}
            </p>
          ) : null}

          <Section
            title={`Active alert episodes (${centre.active?.count ?? 0})`}
            description="One ongoing episode per patient and rule. Later scores inside the deduplication window update the episode instead of creating duplicates."
          >
            <AlertTable
              alerts={centre.active?.alerts ?? []}
              emptyTitle="No active alert episodes"
              busyId={centre.actionId}
              onAcknowledge={centre.acknowledge}
              onClear={(alertId) =>
                centre.clear(alertId, "manual_clear_from_alert_centre")
              }
            />
          </Section>

          <Section
            title={`Acknowledged and cleared history (${centre.history?.count ?? 0})`}
            description="Acknowledged and cleared episodes remain available for retrospective review."
          >
            <AlertTable
              alerts={centre.history?.alerts ?? []}
              emptyTitle="No acknowledged or cleared episodes"
              busyId={centre.actionId}
              onAcknowledge={centre.acknowledge}
              onClear={(alertId) =>
                centre.clear(alertId, "manual_clear_from_alert_centre")
              }
            />
          </Section>

          <Section
            title="How this workflow behaves"
            description="Documented lifecycle and deduplication rules for the demonstration."
          >
            <ul className="notes">
              <li>
                <strong>ACTIVE</strong> — the configured threshold was reached and the
                episode has not been reviewed or cleared.
              </li>
              <li>
                <strong>ACKNOWLEDGED</strong> — a reviewer has seen the episode. It is
                removed from the active list but not deleted.
              </li>
              <li>
                <strong>CLEARED</strong> — the episode was cleared manually, superseded
                by a new episode outside the deduplication window, or cleared because
                the score fell below the configured threshold.
              </li>
              <li>
                An episode is updated rather than duplicated while the next score is
                within the configured deduplication window. Navigating backwards in
                replay never creates another episode.
              </li>
              <li>
                The threshold is configurable and is not clinically validated. Alert
                status is an application workflow state, not a clinical finding.
              </li>
            </ul>
          </Section>
        </>
      ) : null}
    </div>
  );
}
