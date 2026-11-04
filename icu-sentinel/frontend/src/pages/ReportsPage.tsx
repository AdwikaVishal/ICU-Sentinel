import { BarChart3, Database, FileText, Info } from "lucide-react";

import { alertsApi } from "../api/client";
import type { AlertListResponse } from "../api/types";
import { SourceBadge } from "../components/Badges";
import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";
import { EmptyState, ErrorState, LoadingState } from "../components/common/StateViews";
import { RiskOverviewChart } from "../components/dashboard/RiskOverviewChart";
import { useAsync } from "../hooks/useAsync";
import { useAlertCentre, useGlobalExplanation, useModelCatalog } from "../hooks/useExplanation";
import { useRiskSeries } from "../hooks/useRiskSeries";
import {
  usePatients,
  useProviderStatus,
  useThreshold,
  useTimeline,
} from "../hooks/useReplayResources";

function formatPercent(value: number, total: number): string {
  if (total <= 0) return "—";
  return `${((value / total) * 100).toFixed(1)}%`;
}

export function ReportsPage() {
  const patients = usePatients();
  const provider = useProviderStatus();
  const threshold = useThreshold();
  const catalog = useModelCatalog();
  const globalExplanation = useGlobalExplanation();
  const alerts = useAlertCentre();

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
  const missingHours = patients.data?.patients.reduce(
    (sum, patient) => sum + patient.missing_hours,
    0,
  );
  const observationCount = patients.data?.patients.reduce(
    (sum, patient) => sum + patient.observation_count,
    0,
  );

  const alertList = alerts.all?.alerts ?? [];
  const activeAlerts = alertList.filter((alert) => alert.status === "ACTIVE").length;
  const acknowledgedAlerts = alertList.filter(
    (alert) => alert.status === "ACKNOWLEDGED",
  ).length;
  const clearedAlerts = alertList.filter(
    (alert) => alert.status === "CLEARED",
  ).length;

  const explainableModels = catalog.data?.models.filter((item) => item.explainable) ?? [];

  return (
    <div className="page">
      <PageHeader
        eyebrow={
          <>
            <BarChart3 size={12} aria-hidden="true" />
            Research analytics
          </>
        }
        title="Reports & Analytics"
        description="Cohort provenance, alert statistics and explanation availability for the current research configuration. Clinical performance metrics are shown only when the backend provides them."
      />

      <div className="report-grid">
        <SectionCard
          className="report-card"
          title="Dataset overview"
          description="Cohort-level figures returned by the patient API."
        >
          {patients.loading ? <LoadingState label="Loading cohort…" /> : null}
          {patients.error ? (
            <ErrorState error={patients.error} onRetry={patients.reload} />
          ) : null}
          {patients.data ? (
            <>
              <div className="report-metric">
                <span className="report-metric__label">Patients</span>
                <span className="report-metric__value">
                  {patients.data.count.toLocaleString("en-US")}
                </span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Total cohort hours</span>
                <span className="report-metric__value">
                  {totalHours?.toLocaleString("en-US") ?? "—"}
                </span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Recorded hours</span>
                <span className="report-metric__value">
                  {recordedHours?.toLocaleString("en-US") ?? "—"}
                </span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Missing hours</span>
                <span className="report-metric__value">
                  {missingHours?.toLocaleString("en-US") ?? "—"}
                </span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Recorded observations</span>
                <span className="report-metric__value">
                  {observationCount?.toLocaleString("en-US") ?? "—"}
                </span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Data coverage</span>
                <span className="report-metric__value">
                  {totalHours && recordedHours !== undefined
                    ? formatPercent(recordedHours, totalHours)
                    : "—"}
                </span>
              </div>
              <div className="row" style={{ gap: 8 }}>
                <span className="badge badge--soft-green">
                  <Database size={12} aria-hidden="true" />
                  {patients.data.data_source}
                </span>
                {patients.data.is_synthetic ? (
                  <span className="badge badge--synthetic">Synthetic demo</span>
                ) : null}
              </div>
            </>
          ) : null}
        </SectionCard>

        <SectionCard
          className="report-card"
          title="Model evaluation summary"
          description="Only reported backend results are displayed. No clinical performance metric is estimated in the frontend."
        >
          {provider.loading ? <LoadingState label="Loading provider…" /> : null}
          {provider.error ? (
            <ErrorState error={provider.error} onRetry={provider.reload} />
          ) : null}
          {provider.data ? (
            <div className="summary-grid">
              <div className="summary-tile">
                <dt>Provider</dt>
                <dd>{provider.data.provider_name}</dd>
              </div>
              <div className="summary-tile">
                <dt>Model</dt>
                <dd>
                  {provider.data.model?.display_name ??
                    provider.data.model?.name ??
                    "—"}
                </dd>
              </div>
              <div className="summary-tile">
                <dt>Version</dt>
                <dd>{provider.data.model?.version ?? "—"}</dd>
              </div>
              <div className="summary-tile">
                <dt>Feature schema</dt>
                <dd>{provider.data.model?.feature_schema_version ?? "—"}</dd>
              </div>
              <div className="summary-tile">
                <dt>Availability</dt>
                <dd>
                  <span
                    className={`badge ${
                      provider.data.available ? "badge--ok" : "badge--warn"
                    }`}
                  >
                    {provider.data.available ? "Available" : "Unavailable"}
                  </span>
                </dd>
              </div>
              <div className="summary-tile">
                <dt>Source</dt>
                <dd>
                  {provider.data.model ? (
                    <SourceBadge source={provider.data.model.source} />
                  ) : (
                    "—"
                  )}
                </dd>
              </div>
            </div>
          ) : null}
          <div className="evaluation-empty">
            <strong>Evaluation results are not available for the current configuration.</strong>
            <p>
              The P3 API exposes replay, explanation and alert outputs, but no
              AUROC, AUPRC, sensitivity, specificity or calibration endpoint.
              These metrics are intentionally not fabricated in the interface.
            </p>
          </div>
        </SectionCard>

        <SectionCard
          className="report-card"
          title="Alert statistics"
          description="Counts from persisted alert episodes for the configured threshold."
        >
          {alerts.loading ? <LoadingState label="Loading alert statistics…" /> : null}
          {alerts.error ? (
            <ErrorState error={alerts.error} onRetry={alerts.reload} />
          ) : null}
          {!alerts.loading && !alerts.error ? (
            <>
              <div className="report-metric">
                <span className="report-metric__label">Active episodes</span>
                <span className="report-metric__value">{activeAlerts}</span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Acknowledged</span>
                <span className="report-metric__value">{acknowledgedAlerts}</span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Cleared</span>
                <span className="report-metric__value">{clearedAlerts}</span>
              </div>
              <div className="report-metric">
                <span className="report-metric__label">Total episodes</span>
                <span className="report-metric__value">{alertList.length}</span>
              </div>
              {threshold.data ? (
                <p className="muted tiny">
                  Threshold {threshold.data.threshold.toFixed(2)} ·{" "}
                  {threshold.data.clinically_validated
                    ? "clinically validated"
                    : "not clinically validated"}
                </p>
              ) : null}
            </>
          ) : null}
        </SectionCard>

        <SectionCard
          className="report-card"
          title="Explanation availability"
          description="Registered explainer models and whether a global explanation is currently available."
        >
          {catalog.loading || globalExplanation.loading ? (
            <LoadingState label="Loading explanation metadata…" />
          ) : null}
          {catalog.error ? (
            <ErrorState error={catalog.error} onRetry={catalog.reload} />
          ) : null}
          {globalExplanation.error ? (
            <ErrorState
              error={globalExplanation.error}
              onRetry={globalExplanation.reload}
            />
          ) : null}
          {catalog.data ? (
            <div className="summary-grid">
              <div className="summary-tile">
                <dt>Registered models</dt>
                <dd className="numeric">{catalog.data.count}</dd>
              </div>
              <div className="summary-tile">
                <dt>Explainable models</dt>
                <dd className="numeric">{explainableModels.length}</dd>
              </div>
              <div className="summary-tile">
                <dt>Global explanation</dt>
                <dd>
                  <span
                    className={`badge ${
                      globalExplanation.data?.status === "available"
                        ? "badge--ok"
                        : "badge--muted"
                    }`}
                  >
                    {globalExplanation.data?.status ?? "Unavailable"}
                  </span>
                </dd>
              </div>
              <div className="summary-tile">
                <dt>Background rows</dt>
                <dd className="numeric">
                  {globalExplanation.data?.background_rows ?? "—"}
                </dd>
              </div>
            </div>
          ) : null}
          {globalExplanation.data &&
          globalExplanation.data.status !== "available" ? (
            <p className="muted tiny">
              {globalExplanation.data.reason ?? "No global explanation is available."}
            </p>
          ) : null}
        </SectionCard>

        <SectionCard
          className="report-card report-card--wide"
          title="Data quality summary"
          description="Recorded and missing hours per patient, exactly as returned by the patient API."
        >
          {patients.data && patients.data.patients.length > 0 ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Patient</th>
                    <th scope="col">Total hours</th>
                    <th scope="col">Recorded</th>
                    <th scope="col">Missing</th>
                    <th scope="col">Observations</th>
                    <th scope="col">Coverage</th>
                  </tr>
                </thead>
                <tbody>
                  {patients.data.patients.map((patient) => (
                    <tr key={patient.patient_id}>
                      <th scope="row">
                        {patient.patient_id}
                        {patient.is_synthetic ? (
                          <span className="badge badge--synthetic">Synthetic</span>
                        ) : null}
                      </th>
                      <td className="numeric">{patient.total_hours}</td>
                      <td className="numeric">{patient.recorded_hours}</td>
                      <td className="numeric">{patient.missing_hours}</td>
                      <td className="numeric">{patient.observation_count}</td>
                      <td className="numeric">
                        {formatPercent(patient.recorded_hours, patient.total_hours)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <EmptyState title="No cohort data is available">
              Configure processed data or enable the synthetic demonstration
              cohort to populate this report.
            </EmptyState>
          )}
        </SectionCard>
      </div>

      <RiskOverviewChart
        title="Risk prediction trend"
        subtitle="Actual hourly replay scores for one representative demo patient. Evaluation metrics are not inferred from this chart."
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

      <div className="disclaimer" role="note">
        <Info size={15} aria-hidden="true" />
        <span>
          <strong>Report scope.</strong> This page reports only data exposed by the
          current backend. It does not claim clinical validation or model
          performance beyond the configured research prototype.
        </span>
      </div>

      <div className="row" style={{ gap: 8 }}>
        <FileText size={14} className="muted" aria-hidden="true" />
        <span className="muted tiny">
          Evaluation modules can be added here when P2/P3 metric endpoints become
          available without redesigning the page.
        </span>
      </div>
    </div>
  );
}
