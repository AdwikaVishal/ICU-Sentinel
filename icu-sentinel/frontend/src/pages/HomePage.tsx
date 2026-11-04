import {
  BellRing,
  Brain,
  Layers,
  Play,
  ShieldAlert,
  SlidersHorizontal,
  Users,
} from "lucide-react";
import { Link } from "react-router-dom";

import { alertsApi } from "../api/client";
import type { AlertListResponse } from "../api/types";
import { ErrorState } from "../components/common/StateViews";
import { HeroSection } from "../components/dashboard/HeroSection";
import { KeyModuleCard } from "../components/dashboard/KeyModuleCard";
import { MetricCard } from "../components/dashboard/MetricCard";
import { RiskOverviewChart } from "../components/dashboard/RiskOverviewChart";
import type { ServiceState } from "../components/dashboard/SystemStatusCard";
import { useAsync } from "../hooks/useAsync";
import { useActiveAlerts } from "../hooks/useActiveAlerts";
import { useModelCatalog } from "../hooks/useExplanation";
import { useRiskSeries } from "../hooks/useRiskSeries";
import {
  usePatients,
  useProviderStatus,
  useThreshold,
  useTimeline,
} from "../hooks/useReplayResources";

function formatCount(value: number | null | undefined): string {
  return value === null || value === undefined
    ? "—"
    : value.toLocaleString("en-US");
}

function serviceState(
  loading: boolean,
  error: unknown,
  available: boolean,
): ServiceState {
  if (loading) return "checking";
  if (error || !available) return "offline";
  return "online";
}

export function HomePage() {
  const patients = usePatients();
  const provider = useProviderStatus();
  const threshold = useThreshold();
  const catalog = useModelCatalog();
  const activeAlerts = useActiveAlerts();

  const previewPatient =
    patients.data?.patients.find((patient) => patient.patient_id === "DEMO-0004") ??
    patients.data?.patients[0] ??
    null;

  const timeline = useTimeline(previewPatient?.patient_id ?? null);
  const riskSeries = useRiskSeries(previewPatient?.patient_id ?? null, timeline.data);
  const patientAlerts = useAsync<AlertListResponse>(
    (signal) =>
      alertsApi.list(
        { patientId: previewPatient?.patient_id, limit: 500 },
        signal,
      ),
    [previewPatient?.patient_id],
    Boolean(previewPatient),
  );

  const totalHours = patients.data?.patients.reduce(
    (sum, patient) => sum + patient.total_hours,
    0,
  );
  const recordedHours = patients.data?.patients.reduce(
    (sum, patient) => sum + patient.recorded_hours,
    0,
  );
  const coverage =
    totalHours && totalHours > 0 && recordedHours !== undefined
      ? `${((recordedHours / totalHours) * 100).toFixed(1)}%`
      : "—";

  const status = {
    replay: serviceState(
      patients.loading || provider.loading,
      patients.error ?? provider.error,
      Boolean(patients.data) && Boolean(provider.data),
    ),
    explainability: serviceState(
      catalog.loading,
      catalog.error,
      Boolean(catalog.data),
    ),
    alerts: serviceState(
      activeAlerts.loading || threshold.loading,
      activeAlerts.error ?? threshold.error,
      Boolean(activeAlerts.data) && Boolean(threshold.data),
    ),
    dataSource: patients.data?.data_source ?? null,
    isSynthetic: patients.data?.is_synthetic ?? null,
  };

  const activeAlertCount = activeAlerts.error
    ? null
    : (activeAlerts.data?.count ?? null);
  const modelCount = catalog.error ? null : (catalog.data?.count ?? null);
  const featureCount = provider.error
    ? null
    : (provider.data?.required_features?.length ?? null);

  return (
    <div className="page">
      <HeroSection status={status} />

      {patients.error ? (
        <ErrorState
          title="The patient catalog could not be loaded"
          error={patients.error}
          onRetry={patients.reload}
        />
      ) : null}

      <section aria-labelledby="kpi-heading">
        <h2 id="kpi-heading" className="sr-only">
          Cohort overview
        </h2>
        <div className="kpi-grid">
          <MetricCard
            icon={<Users size={19} />}
            value={formatCount(patients.data?.count)}
            label="Demo Patients"
            hint={
              patients.data?.is_synthetic
                ? "Synthetic demo cohort"
                : "Configured patient cohort"
            }
          />
          <MetricCard
            icon={<SlidersHorizontal size={19} />}
            value={formatCount(featureCount)}
            label="Clinical Features"
            hint="Active model input schema"
            tone="info"
          />
          <MetricCard
            icon={<Layers size={19} />}
            value={formatCount(modelCount)}
            label="Model Types"
            hint="Registered prediction providers"
            tone="cream"
          />
          <MetricCard
            icon={<BellRing size={19} />}
            value={formatCount(activeAlertCount)}
            label="Active Alerts"
            hint="Open alert episodes"
            tone="lilac"
          />
          <MetricCard
            icon={<ShieldAlert size={19} />}
            value={coverage}
            label="Data Coverage"
            hint="Recorded hours across cohort"
          />
        </div>
      </section>

      <section aria-labelledby="modules-heading">
        <div className="section-heading">
          <div>
            <h2 id="modules-heading">Key Modules</h2>
            <p>Three connected views for retrospective sepsis research.</p>
          </div>
          <Link className="section-heading__link" to="/help">
            How it works
          </Link>
        </div>
        <div className="module-grid" style={{ marginTop: 16 }}>
          <KeyModuleCard
            icon={<Play size={22} />}
            title="Patient Replay"
            description="Replay a patient's ICU stay hour by hour, view risk predictions and clinical observations."
            cta="Open Replay"
            to="/replay"
            tag="Timeline"
          />
          <KeyModuleCard
            icon={<Brain size={22} />}
            title="Explainability (SHAP)"
            description="Understand why the model raised a prediction using feature contributions and explainable AI."
            cta="Explore Explanations"
            to="/explanations"
            tone="cream"
            tag="Explainable AI"
          />
          <KeyModuleCard
            icon={<BellRing size={22} />}
            title="Alert Engine"
            description="Generate, track and manage sepsis risk alerts with thresholds and deduplication."
            cta="Manage Alerts"
            to="/alerts"
            tone="lilac"
            tag="Workflow"
          />
        </div>
      </section>

      <RiskOverviewChart
        points={riskSeries.data ?? []}
        threshold={threshold.data?.threshold ?? null}
        alerts={patientAlerts.data?.alerts ?? []}
        patientId={previewPatient?.patient_id ?? null}
        patientLabel={previewPatient?.patient_id}
        loading={timeline.loading || riskSeries.loading}
        error={timeline.error ?? riskSeries.error}
        onRetry={() => {
          timeline.reload();
          riskSeries.reload();
        }}
      />
    </div>
  );
}
