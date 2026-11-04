import { useEffect, useState } from "react";

import { EmptyPanel, ErrorPanel, LoadingPanel, Section } from "../components/AsyncState";
import { CausalityPanel } from "../components/CausalityPanel";
import { SyntheticNote } from "../components/Disclaimer";
import { HourNavigator } from "../components/HourNavigator";
import { ObservationTable } from "../components/ObservationTable";
import { PatientSelector } from "../components/PatientSelector";
import { PredictionPanel } from "../components/PredictionPanel";
import { SourceBadge } from "../components/Badges";
import { usePatients, useProviderStatus } from "../hooks/useReplayResources";
import { useReplay } from "../hooks/useReplay";

export function PatientReplayPage() {
  const patients = usePatients();
  const provider = useProviderStatus();
  const [selected, setSelected] = useState<string | null>(null);
  const replay = useReplay(selected);

  useEffect(() => {
    if (!selected && patients.data && patients.data.patients.length > 0) {
      setSelected(patients.data.patients[0].patient_id);
    }
  }, [patients.data, selected]);

  return (
    <div className="page">
      <div className="page__heading">
        <div>
          <h2>Patient Replay</h2>
          <p className="muted">
            Step through recorded hours. The prediction at each hour uses only that
            hour and earlier observations.
          </p>
        </div>
      </div>

      {provider.data ? (
        <div className="provider-strip">
          <span className="muted small">Prediction provider</span>
          <strong>{provider.data.provider_name}</strong>
          {provider.data.model ? <SourceBadge source={provider.data.model.source} /> : null}
          <span className="muted small">
            {provider.data.available
              ? "Available"
              : provider.data.reason ?? "Unavailable"}
          </span>
        </div>
      ) : null}

      {patients.loading ? <LoadingPanel label="Loading synthetic demo patients…" /> : null}
      {patients.error ? (
        <ErrorPanel error={patients.error} onRetry={patients.reload} />
      ) : null}

      {patients.data && patients.data.patients.length === 0 ? (
        <EmptyPanel title="No patients are available">
          Configure a processed data directory or enable the synthetic demonstration
          cohort.
        </EmptyPanel>
      ) : null}

      {patients.data && patients.data.patients.length > 0 ? (
        <Section
          title="Select a patient and hour"
          description="Synthetic demo patients are labelled throughout. Missing hours are shown explicitly."
        >
          <div className="toolbar">
            <PatientSelector
              patients={patients.data.patients}
              value={selected}
              onChange={(patientId) => setSelected(patientId)}
              disabled={replay.loading}
            />
            <label className="field field--inline">
              <input
                type="checkbox"
                checked={replay.evaluateAlerts}
                onChange={(event) => replay.setEvaluateAlerts(event.target.checked)}
              />
              <span>
                Evaluate the configured alert rule when navigating
              </span>
            </label>
            {replay.evaluateAlerts && replay.hour !== null ? (
              <button
                type="button"
                className="button"
                onClick={() => replay.goToHour(replay.hour as number)}
                disabled={replay.navigating}
              >
                Evaluate this hour
              </button>
            ) : null}
          </div>

          {replay.timeline ? (
            <HourNavigator
              timeline={replay.timeline}
              hour={replay.hour}
              busy={replay.loading || replay.navigating}
              onSelect={replay.goToHour}
              onPrevious={replay.previousHour}
              onNext={replay.nextHour}
              onReset={replay.reset}
              onEnd={replay.goToEnd}
            />
          ) : null}
        </Section>
      ) : null}

      {replay.loading ? <LoadingPanel label="Building the causal prefix and score…" /> : null}
      {replay.error ? <ErrorPanel error={replay.error} onRetry={replay.reload} /> : null}

      {replay.state ? (
        <>
          <SyntheticNote isSynthetic={replay.state.is_synthetic} />
          {replay.navigating ? (
            <p className="muted small" role="status">
              Updating replay hour…
            </p>
          ) : null}

          <div className="grid grid--2">
            <Section
              title={`Hour ${replay.state.hour} model output`}
              description="Probability-like model score with explicit availability and provenance."
            >
              <PredictionPanel state={replay.state} />
            </Section>

            <Section
              title="Temporal causality"
              description="Automated leakage tests cover this guarantee; these values come from the served prediction."
            >
              <CausalityPanel causality={replay.state.causality} />
            </Section>
          </div>

          <Section
            title="Recorded observations"
            description="Only values recorded at this hour are shown as recorded. Missing values stay missing."
          >
            <ObservationTable observations={replay.state.observations} />
          </Section>
        </>
      ) : null}
    </div>
  );
}
