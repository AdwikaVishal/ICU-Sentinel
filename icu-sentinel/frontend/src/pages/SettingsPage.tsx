import {
  Activity,
  Brain,
  Database,
  Gauge,
  Settings,
  ShieldCheck,
  SlidersHorizontal,
} from "lucide-react";
import type { ReactNode } from "react";

import { API_BASE } from "../api/client";
import { SourceBadge } from "../components/Badges";
import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";
import { ErrorState, LoadingState } from "../components/common/StateViews";
import { useModelCatalog } from "../hooks/useExplanation";
import {
  usePatients,
  useProviderStatus,
  useThreshold,
} from "../hooks/useReplayResources";

function ConfigRow({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="settings-row">
      <div>
        <div className="settings-row__label">{label}</div>
        {hint ? <div className="settings-row__hint">{hint}</div> : null}
      </div>
      <div className="settings-row__value">{children}</div>
    </div>
  );
}

function AvailabilityTag({
  state,
}: {
  state: "online" | "offline" | "checking";
}) {
  const label =
    state === "online" ? "Available" : state === "offline" ? "Unavailable" : "Checking…";
  return (
    <span
      className={`badge ${
        state === "online"
          ? "badge--ok"
          : state === "offline"
            ? "badge--warn"
            : "badge--muted"
      }`}
    >
      {label}
    </span>
  );
}

function stateFor(loading: boolean, error: unknown, available: boolean) {
  if (loading) return "checking" as const;
  if (error || !available) return "offline" as const;
  return "online" as const;
}

export function SettingsPage() {
  const patients = usePatients();
  const provider = useProviderStatus();
  const threshold = useThreshold();
  const catalog = useModelCatalog();

  const replayState = stateFor(
    patients.loading || provider.loading,
    patients.error ?? provider.error,
    Boolean(patients.data) && Boolean(provider.data),
  );
  const explanationState = stateFor(
    provider.loading || catalog.loading,
    provider.error ?? catalog.error,
    Boolean(provider.data) && Boolean(catalog.data),
  );
  const alertState = stateFor(
    threshold.loading,
    threshold.error,
    Boolean(threshold.data),
  );

  return (
    <div className="page">
      <PageHeader
        eyebrow={
          <>
            <Settings size={12} aria-hidden="true" />
            Workspace configuration
          </>
        }
        title="Settings"
        description="Read-only view of the active research configuration, data provenance and service availability. Values are reported by the P3 backend and cannot be changed from this interface."
      />

      <div className="settings-grid">
        <SectionCard
          title="Data provenance"
          description="Source and synthetic status of the loaded cohort."
          actions={<Database size={18} aria-hidden="true" />}
        >
          {patients.loading ? <LoadingState label="Loading cohort…" /> : null}
          {patients.error ? (
            <ErrorState error={patients.error} onRetry={patients.reload} />
          ) : null}
          {patients.data ? (
            <>
              <ConfigRow label="Data source">
                <span className="badge badge--soft-green">
                  {patients.data.data_source}
                </span>
              </ConfigRow>
              <ConfigRow label="Patient records">
                {patients.data.count.toLocaleString("en-US")}
              </ConfigRow>
              <ConfigRow
                label="Synthetic data"
                hint="Synthetic records are demonstration data, not real patients."
              >
                <span
                  className={`badge ${
                    patients.data.is_synthetic
                      ? "badge--synthetic"
                      : "badge--soft-green"
                  }`}
                >
                  {patients.data.is_synthetic ? "Synthetic demo" : "Configured dataset"}
                </span>
              </ConfigRow>
              <ConfigRow label="Prediction source">
                {patients.data.prediction_source}
              </ConfigRow>
              <ConfigRow label="Provider">
                {patients.data.provider}
              </ConfigRow>
              <p className="notes">{patients.data.disclaimer}</p>
            </>
          ) : null}
        </SectionCard>

        <SectionCard
          title="Prediction model"
          description="Active provider and model reported by the replay service."
          actions={<Brain size={18} aria-hidden="true" />}
        >
          {provider.loading ? <LoadingState label="Loading provider…" /> : null}
          {provider.error ? (
            <ErrorState error={provider.error} onRetry={provider.reload} />
          ) : null}
          {provider.data ? (
            <>
              <ConfigRow label="Provider">
                {provider.data.provider_name}
              </ConfigRow>
              <ConfigRow label="Availability">
                <AvailabilityTag state={replayState} />
              </ConfigRow>
              <ConfigRow label="Model">
                {provider.data.model?.display_name ??
                  provider.data.model?.name ??
                  "—"}
              </ConfigRow>
              <ConfigRow label="Model version">
                {provider.data.model?.version ?? "—"}
              </ConfigRow>
              <ConfigRow label="Model source">
                {provider.data.model ? (
                  <SourceBadge source={provider.data.model.source} />
                ) : (
                  "—"
                )}
              </ConfigRow>
              <ConfigRow label="Feature schema">
                {provider.data.model?.feature_schema_version ?? "—"}
              </ConfigRow>
              <ConfigRow
                label="Required input features"
                hint="Fields expected by the active model input schema."
              >
                {provider.data.required_features?.length ?? "—"}
              </ConfigRow>
              {provider.data.reason ? (
                <p className="notes">{provider.data.reason}</p>
              ) : null}
            </>
          ) : null}
        </SectionCard>

        <SectionCard
          title="Alert threshold"
          description="Demonstration rule used to raise alert episodes. Thresholds are not clinical cut-offs."
          actions={<SlidersHorizontal size={18} aria-hidden="true" />}
        >
          {threshold.loading ? <LoadingState label="Loading threshold…" /> : null}
          {threshold.error ? (
            <ErrorState error={threshold.error} onRetry={threshold.reload} />
          ) : null}
          {!threshold.loading && !threshold.error && !threshold.data ? (
            <p className="notes">
              No alert threshold is configured for the current provider.
            </p>
          ) : null}
          {threshold.data ? (
            <>
              <ConfigRow label="Available">
                <AvailabilityTag state={alertState} />
              </ConfigRow>
              <ConfigRow label="Rule ID">{threshold.data.rule_id}</ConfigRow>
              <ConfigRow label="Risk threshold">
                {threshold.data.threshold.toFixed(2)}
              </ConfigRow>
              <ConfigRow label="Threshold source">
                {threshold.data.source}
              </ConfigRow>
              <ConfigRow label="Threshold version">
                {threshold.data.version}
              </ConfigRow>
              <ConfigRow
                label="Deduplication window"
                hint="Hours in which later scores update the same episode."
              >
                {threshold.data.dedup_window_hours} h
              </ConfigRow>
              <ConfigRow label="Clear below threshold">
                {threshold.data.clear_on_below_threshold ? "Enabled" : "Disabled"}
              </ConfigRow>
              <ConfigRow label="Clinically validated">
                <span
                  className={`badge ${
                    threshold.data.clinically_validated
                      ? "badge--ok"
                      : "badge--synthetic"
                  }`}
                >
                  {threshold.data.clinically_validated ? "Yes" : "No"}
                </span>
              </ConfigRow>
              <p className="notes">{threshold.data.disclosure}</p>
            </>
          ) : null}
        </SectionCard>

        <SectionCard
          title="Explainability"
          description="Model catalog and explanation method reported by the explanation service."
          actions={<Gauge size={18} aria-hidden="true" />}
        >
          {catalog.loading ? <LoadingState label="Loading model catalog…" /> : null}
          {catalog.error ? (
            <ErrorState error={catalog.error} onRetry={catalog.reload} />
          ) : null}
          {catalog.data ? (
            <>
              <ConfigRow label="Available">
                <AvailabilityTag state={explanationState} />
              </ConfigRow>
              {catalog.data.models.length === 0 ? (
                <p className="notes">
                  No models are exposed by the explanation catalog.
                </p>
              ) : null}
              {catalog.data.models.map((entry) => (
                <ConfigRow
                  key={`${entry.model.name}-${entry.model.version}`}
                  label={entry.model.display_name || entry.model.name}
                  hint={`${entry.model.version} · ${
                    entry.explanation_method ?? "no explanation method"
                  }`}
                >
                  <span
                    className={`badge ${
                      entry.explainable ? "badge--ok" : "badge--muted"
                    }`}
                  >
                    {entry.explainable ? "Explainable" : "Unavailable"}
                  </span>
                </ConfigRow>
              ))}
            </>
          ) : null}
        </SectionCard>

        <SectionCard
          className="report-card--wide"
          title="API connection"
          description="Frontend connection details and the P3 endpoints backing this workspace."
          actions={<Activity size={18} aria-hidden="true" />}
        >
          <ConfigRow label="API base URL">
            <code>{API_BASE || "Same origin (Vite dev proxy)"}</code>
          </ConfigRow>
          <ConfigRow label="Replay service">
            <AvailabilityTag state={replayState} />
          </ConfigRow>
          <ConfigRow label="Explanation service">
            <AvailabilityTag state={explanationState} />
          </ConfigRow>
          <ConfigRow label="Alert service">
            <AvailabilityTag state={alertState} />
          </ConfigRow>
          <div className="endpoint-list">
            <code>GET /api/v3/patients</code>
            <code>GET /api/v3/patients/{"{patient_id}"}/timeline</code>
            <code>GET /api/v3/replay/{"{patient_id}"}?hour={"{"}hour{"}"}</code>
            <code>GET /api/v3/explanations/{"{patient_id}"}/{"{hour}"}</code>
            <code>GET /api/v3/alerts?status=</code>
            <code>POST /api/v3/alerts/{"{alert_id}"}/acknowledge</code>
            <code>POST /api/v3/alerts/{"{alert_id}"}/clear</code>
          </div>
          <p className="notes">
            <ShieldCheck
              size={13}
              aria-hidden="true"
              style={{ display: "inline", marginRight: 5, verticalAlign: "-2px" }}
            />
            The frontend does not persist patient data, mutate model configuration or
            write clinical records. Alert acknowledgement and clearing are the only
            write actions and are limited to the research alert workflow.
          </p>
        </SectionCard>
      </div>
    </div>
  );
}
