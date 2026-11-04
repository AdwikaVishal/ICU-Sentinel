import { ArrowUpRight, Brain, ExternalLink, Info } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

import { alertsApi } from "../api/client";
import type { AlertListResponse } from "../api/types";
import { CausalityPanel } from "../components/CausalityPanel";
import { PatientSelector } from "../components/PatientSelector";
import { SourceBadge } from "../components/Badges";
import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";
import {
  EmptyState,
  ErrorState,
  LoadingState,
} from "../components/common/StateViews";
import { ClinicalObservationGrid } from "../components/replay/ClinicalObservationGrid";
import { PatientTimeline } from "../components/replay/PatientTimeline";
import { ReplayControls } from "../components/replay/ReplayControls";
import { RiskScoreCard } from "../components/replay/RiskScoreCard";
import { useAsync } from "../hooks/useAsync";
import { useReplay } from "../hooks/useReplay";
import {
  usePatients,
  useProviderStatus,
  useThreshold,
} from "../hooks/useReplayResources";

const PLAY_INTERVAL_MS = 1400;

export function ReplayPage() {
  const patients = usePatients();
  const provider = useProviderStatus();
  const threshold = useThreshold();
  const [selected, setSelected] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedPatient = searchParams.get("patient");
  const requestedHourRaw = searchParams.get("hour");
  const requestedHour =
    requestedHourRaw !== null && Number.isFinite(Number(requestedHourRaw))
      ? Number(requestedHourRaw)
      : null;
  const appliedInitialHour = useRef(false);

  const replay = useReplay(selected);

  const patientAlerts = useAsync<AlertListResponse>(
    (signal) =>
      alertsApi.list({ patientId: selected as string, limit: 500 }, signal),
    [selected, replay.state?.alert?.updated_at, replay.state?.alert_status],
    Boolean(selected),
  );

  useEffect(() => {
    if (selected || !patients.data || patients.data.patients.length === 0) return;
    const candidate =
      requestedPatient &&
      patients.data.patients.some(
        (patient) => patient.patient_id === requestedPatient,
      )
        ? requestedPatient
        : patients.data.patients[0].patient_id;
    setSelected(candidate);
  }, [patients.data, requestedPatient, selected]);

  useEffect(() => {
    setPlaying(false);
  }, [selected]);

  useEffect(() => {
    if (appliedInitialHour.current) return;
    if (requestedHour === null || replay.timeline === null) return;
    const first = replay.timeline.hours[0]?.hour;
    const last = replay.timeline.hours[replay.timeline.hours.length - 1]?.hour;
    if (first === undefined || last === undefined) return;
    appliedInitialHour.current = true;
    if (requestedHour >= first && requestedHour <= last) {
      replay.goToHour(requestedHour);
    }
  }, [replay, requestedHour]);

  const minHour = replay.timeline?.hours[0]?.hour ?? null;
  const maxHour = replay.timeline
    ? (replay.timeline.hours[replay.timeline.hours.length - 1]?.hour ?? null)
    : null;

  useEffect(() => {
    if (!playing) return;
    if (replay.hour === null || minHour === null || maxHour === null) {
      setPlaying(false);
      return;
    }
    if (replay.hour >= maxHour) {
      setPlaying(false);
      return;
    }
    const timer = window.setInterval(() => {
      if (!replay.navigating) replay.nextHour();
    }, PLAY_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [
    maxHour,
    minHour,
    playing,
    replay,
    replay.hour,
    replay.navigating,
    replay.nextHour,
  ]);

  const handleSelect = (patientId: string) => {
    setPlaying(false);
    setSelected(patientId);
    appliedInitialHour.current = true;
    const next = new URLSearchParams(searchParams);
    next.set("patient", patientId);
    next.delete("hour");
    setSearchParams(next, { replace: true });
  };

  const handlePlayToggle = () => {
    if (playing) {
      setPlaying(false);
      return;
    }
    if (replay.hour !== null && maxHour !== null && replay.hour >= maxHour) {
      replay.reset();
    }
    setPlaying(true);
  };

  const selectedPatient = patients.data?.patients.find(
    (patient) => patient.patient_id === selected,
  );
  const alertCount = patientAlerts.data?.count ?? 0;
  const explanationHref =
    selected && replay.hour !== null
      ? `/explanations?patient=${encodeURIComponent(selected)}&hour=${replay.hour}`
      : "/explanations";

  return (
    <div className="page">
      <PageHeader
        eyebrow={
          <>
            <ActivityDot /> Retrospective monitoring
          </>
        }
        title="Patient Replay"
        description="Explore a patient's ICU timeline and understand how predicted sepsis risk evolves over time."
        actions={
          provider.data ? (
            <span className="badge badge--soft-green">
              <Info size={12} aria-hidden="true" />
              {provider.data.available
                ? provider.data.provider_name
                : (provider.data.reason ?? "Provider unavailable")}
            </span>
          ) : null
        }
      />

      {provider.error ? (
        <ErrorState
          title="Prediction provider status is unavailable"
          error={provider.error}
          onRetry={provider.reload}
        />
      ) : null}

      {patients.loading ? <LoadingState label="Loading demo patients…" /> : null}
      {patients.error ? (
        <ErrorState error={patients.error} onRetry={patients.reload} />
      ) : null}

      {patients.data && patients.data.patients.length === 0 ? (
        <EmptyState title="No patients are available">
          Configure a processed data directory or enable the synthetic
          demonstration cohort.
        </EmptyState>
      ) : null}

      {patients.data && patients.data.patients.length > 0 ? (
        <>
          <section className="selector-card" aria-label="Patient selection">
            <PatientSelector
              patients={patients.data.patients}
              value={selected}
              onChange={handleSelect}
              disabled={replay.loading}
            />
            <div>
              <div className="selector-card__title">
                <Info size={17} aria-hidden="true" />
                <span>Replay context</span>
              </div>
              <p className="selector-card__hint">
                Predictions are generated from a causal prefix ending at the
                selected hour. Future observations are never used.
              </p>
              {selectedPatient ? (
                <dl
                  className="summary-grid"
                  style={{ marginTop: 14 }}
                  aria-label="Selected patient summary"
                >
                  <div className="summary-tile">
                    <dt>Dataset</dt>
                    <dd>
                      {selectedPatient.is_synthetic
                        ? "Synthetic demo"
                        : selectedPatient.data_source}
                    </dd>
                  </div>
                  <div className="summary-tile">
                    <dt>Recorded hours</dt>
                    <dd className="numeric">{selectedPatient.recorded_hours}</dd>
                  </div>
                  <div className="summary-tile">
                    <dt>Missing hours</dt>
                    <dd className="numeric">{selectedPatient.missing_hours}</dd>
                  </div>
                  <div className="summary-tile">
                    <dt>Alert episodes</dt>
                    <dd className="numeric">
                      {patientAlerts.error ? "—" : alertCount}
                    </dd>
                  </div>
                </dl>
              ) : (
                <p className="muted small" style={{ marginTop: 14 }}>
                  Select a patient to begin replay.
                </p>
              )}
            </div>
          </section>

          {replay.timeline ? (
            <ReplayControls
              hour={replay.hour}
              minHour={minHour}
              maxHour={maxHour}
              totalHours={replay.timeline.patient.total_hours}
              playing={playing}
              busy={replay.loading || replay.navigating}
              navigating={replay.navigating}
              evaluateAlerts={replay.evaluateAlerts}
              missingHour={
                replay.state !== null && !replay.state.timeline.has_observations
              }
              onPrevious={replay.previousHour}
              onNext={replay.nextHour}
              onPlayToggle={handlePlayToggle}
              onRestart={() => {
                setPlaying(false);
                replay.reset();
              }}
              onSeek={(hour) => {
                setPlaying(false);
                replay.goToHour(hour);
              }}
              onEvaluateToggle={replay.setEvaluateAlerts}
              onEvaluateNow={() => {
                if (replay.hour !== null) replay.goToHour(replay.hour);
              }}
            />
          ) : null}
        </>
      ) : null}

      {replay.loading ? (
        <LoadingState label="Building the causal prefix and scoring this hour…" />
      ) : null}
      {replay.error ? <ErrorState error={replay.error} onRetry={replay.reload} /> : null}

      {replay.state ? (
        <>
          {!replay.state.timeline.has_observations ? (
            <div className="missing-hour" role="status">
              <Info size={16} aria-hidden="true" />
              <span>
                Hour {replay.state.hour} has no recorded observation row. Missing
                measurements are shown as missing and are never treated as normal
                values.
              </span>
            </div>
          ) : null}

          <div className="replay-layout">
            <div className="replay-main">
              <SectionCard
                title="Clinical observations"
                description={`Measurements returned for hour ${replay.state.hour}. Availability is reported exactly as returned by the API.`}
                actions={
                  <span className="badge badge--soft-green">
                    {replay.state.observations.filter((item) => item.available).length}{" "}
                    recorded
                  </span>
                }
              >
                <ClinicalObservationGrid observations={replay.state.observations} />
              </SectionCard>

              <section className="card timeline-card" aria-labelledby="timeline-title">
                <div className="card__header">
                  <div className="card__header-copy">
                    <h2 id="timeline-title">Patient Timeline</h2>
                    <p>
                      Click any recorded hour to move the replay. Coral markers
                      indicate stored alert episodes.
                    </p>
                  </div>
                  <span className="badge badge--soft-green">
                    Hour {replay.hour ?? "—"}
                  </span>
                </div>
                {replay.timeline ? (
                  <PatientTimeline
                    timeline={replay.timeline}
                    hour={replay.hour}
                    alerts={patientAlerts.data?.alerts ?? []}
                    onSelect={(hour) => {
                      setPlaying(false);
                      replay.goToHour(hour);
                    }}
                    busy={replay.navigating}
                  />
                ) : null}
              </section>

              <SectionCard
                title="Temporal causality & model integrity"
                description="These values come from the served prediction. Automated leakage tests cover this guarantee."
              >
                <CausalityPanel causality={replay.state.causality} />
                <p className="chart-footnote">
                  <SourceBadge source={replay.state.prediction.source} />
                  <span style={{ marginLeft: 4 }}>
                    Model input hash:{" "}
                    <code className="hash">
                      {replay.state.causality.model_input_hash ?? "—"}
                    </code>
                  </span>
                </p>
              </SectionCard>
            </div>

            <div className="replay-side">
              <RiskScoreCard state={replay.state} threshold={threshold.data ?? null} />

              <SectionCard
                title="Linked views"
                description="Continue this patient-hour into explainability or the alert workflow."
              >
                <div className="alert-detail__links">
                  <Link className="detail-link" to={explanationHref}>
                    <Brain size={16} aria-hidden="true" />
                    <span>Explain this hour with SHAP</span>
                    <ArrowUpRight size={14} aria-hidden="true" />
                  </Link>
                  <Link className="detail-link" to="/alerts">
                    <ExternalLink size={16} aria-hidden="true" />
                    <span>Open the alert centre</span>
                    <ArrowUpRight size={14} aria-hidden="true" />
                  </Link>
                </div>
                {selectedPatient?.is_synthetic ? (
                  <p className="provenance" style={{ marginTop: 12 }}>
                    Data source: synthetic demonstration cohort. Nothing on this
                    page is real patient data.
                  </p>
                ) : null}
              </SectionCard>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}

function ActivityDot() {
  return (
    <span
      className="status-dot status-dot--online"
      aria-hidden="true"
      style={{ display: "inline-block" }}
    />
  );
}
