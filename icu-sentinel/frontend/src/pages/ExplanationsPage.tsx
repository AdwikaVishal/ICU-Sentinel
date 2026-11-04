import { Brain, Info, RefreshCw, Sparkles } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";

import { PatientSelector } from "../components/PatientSelector";
import { SourceBadge } from "../components/Badges";
import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";
import {
  EmptyState,
  ErrorState,
  LoadingState,
} from "../components/common/StateViews";
import { ExplanationSummary } from "../components/explainability/ExplanationSummary";
import { GlobalImportanceChart } from "../components/explainability/GlobalImportanceChart";
import { ShapFeatureChart } from "../components/explainability/ShapFeatureChart";
import {
  useGlobalExplanation,
  useHourComparison,
  useModelCatalog,
  usePatientExplanation,
} from "../hooks/useExplanation";
import { usePatients, useTimeline } from "../hooks/useReplayResources";

export function ExplanationsPage() {
  const patients = usePatients();
  const [searchParams, setSearchParams] = useSearchParams();
  const requestedPatient = searchParams.get("patient");
  const requestedHourRaw = searchParams.get("hour");
  const requestedHour =
    requestedHourRaw !== null && Number.isFinite(Number(requestedHourRaw))
      ? Number(requestedHourRaw)
      : null;

  const [selected, setSelected] = useState<string | null>(null);
  const [hour, setHour] = useState<number | null>(null);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [hourA, setHourA] = useState<number | null>(null);
  const [hourB, setHourB] = useState<number | null>(null);
  const appliedRequestedHour = useRef(false);

  const timeline = useTimeline(selected);
  const explanation = usePatientExplanation(selected, hour);
  const comparison = useHourComparison(selected, hourA, hourB, compareEnabled);
  const catalog = useModelCatalog();
  const globalExplanation = useGlobalExplanation();

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
    setHour(null);
    setHourA(null);
    setHourB(null);
    appliedRequestedHour.current = false;
  }, [selected]);

  useEffect(() => {
    const data = timeline.data;
    if (!data || data.patient.patient_id !== selected) return;
    if (data.hours.length === 0) return;
    const firstHour = data.hours[0].hour;
    const lastHour = data.hours[data.hours.length - 1].hour;
    setHour((current) => {
      if (!appliedRequestedHour.current && requestedHour !== null) {
        appliedRequestedHour.current = true;
        if (requestedHour >= firstHour && requestedHour <= lastHour) {
          return requestedHour;
        }
      }
      return current === null || current < firstHour || current > lastHour
        ? firstHour
        : current;
    });
    setHourA((current) => current ?? firstHour);
    setHourB((current) => current ?? lastHour);
  }, [requestedHour, selected, timeline.data]);

  const hours = timeline.data?.hours ?? [];
  const available = explanation.data?.status === "available";

  const handleSelect = (patientId: string) => {
    setSelected(patientId);
    const next = new URLSearchParams(searchParams);
    next.set("patient", patientId);
    next.delete("hour");
    setSearchParams(next, { replace: true });
  };

  const handleHourChange = (nextHour: number) => {
    setHour(nextHour);
    const next = new URLSearchParams(searchParams);
    next.set("hour", String(nextHour));
    setSearchParams(next, { replace: true });
  };

  return (
    <div className="page">
      <PageHeader
        eyebrow={
          <>
            <Sparkles size={12} aria-hidden="true" />
            Explainable AI
          </>
        }
        title="Model Explainability"
        description="Understand which clinical features influenced a model prediction."
        actions={
          catalog.data?.models[0] ? (
            <span className="badge badge--soft-green">
              <Brain size={12} aria-hidden="true" />
              {catalog.data.models[0].explanation_method?.replaceAll("_", " ") ??
                "Explainer unavailable"}
            </span>
          ) : null
        }
      />

      {patients.loading ? <LoadingState label="Loading patients…" /> : null}
      {patients.error ? (
        <ErrorState error={patients.error} onRetry={patients.reload} />
      ) : null}
      {patients.data && patients.data.patients.length === 0 ? (
        <EmptyState title="No patients are available">
          Configure processed data or enable the synthetic demonstration cohort
          before requesting an explanation.
        </EmptyState>
      ) : null}

      {patients.data && patients.data.patients.length > 0 ? (
        <>
          <section className="selector-card" aria-label="Explanation controls">
            <PatientSelector
              patients={patients.data.patients}
              value={selected}
              onChange={handleSelect}
              disabled={patients.loading}
            />
            <div>
              <div className="selector-card__title">
                <Brain size={17} aria-hidden="true" />
                <span>Patient-hour explanation</span>
              </div>
              <p className="selector-card__hint">
                Explanations are computed from the same causal input used for the
                prediction. The API does not expose a model-selection parameter,
                so the active provider model is shown below.
              </p>

              <div className="toolbar" style={{ marginTop: 14 }}>
                <label className="field" style={{ minWidth: 190 }}>
                  <span className="field__label">Recorded hour</span>
                  <select
                    className="select"
                    value={hour ?? ""}
                    onChange={(event) => handleHourChange(Number(event.target.value))}
                    disabled={hours.length === 0}
                    aria-label="Explanation hour"
                  >
                    {hours.map((item) => (
                      <option key={item.hour} value={item.hour}>
                        Hour {item.hour}
                        {item.has_observations ? "" : " · no observations"}
                      </option>
                    ))}
                  </select>
                </label>
                <button
                  type="button"
                  className="btn btn--primary"
                  onClick={explanation.reload}
                  disabled={!selected || hour === null || explanation.loading}
                >
                  <RefreshCw
                    size={15}
                    aria-hidden="true"
                    className={explanation.loading ? "spin-icon" : undefined}
                  />
                  Generate explanation
                </button>
              </div>

              <div className="row" style={{ marginTop: 12, gap: 8 }}>
                {catalog.data?.models[0] ? (
                  <>
                    <span className="tiny muted">
                      {catalog.data.models[0].model.display_name ??
                        catalog.data.models[0].model.name}
                    </span>
                    <SourceBadge source={catalog.data.models[0].model.source} />
                  </>
                ) : null}
                <label className="checkbox-pill">
                  <input
                    type="checkbox"
                    checked={compareEnabled}
                    onChange={(event) => setCompareEnabled(event.target.checked)}
                  />
                  <span>Compare two hours</span>
                </label>
              </div>
            </div>
          </section>

          {timeline.loading ? <LoadingState label="Loading patient timeline…" /> : null}
          {timeline.error ? (
            <ErrorState error={timeline.error} onRetry={timeline.reload} />
          ) : null}
          {selected && !timeline.loading && hours.length === 0 ? (
            <EmptyState title="This patient has no recorded hours">
              A causal prefix cannot be built without recorded observations.
            </EmptyState>
          ) : null}

          {explanation.loading ? (
            <LoadingState label="Computing SHAP explanation…" />
          ) : null}
          {explanation.error ? (
            <ErrorState error={explanation.error} onRetry={explanation.reload} />
          ) : null}

          {explanation.data && explanation.data.status !== "available" ? (
            <SectionCard
              title="Explanation summary"
              description="The backend returned an explicit non-available status."
            >
              <EmptyState title={`Explanation ${explanation.data.status.replaceAll("_", " ")}`}>
                {explanation.data.reason ??
                  "No explanation is available for this patient-hour."}{" "}
                No SHAP value is fabricated when the model cannot be explained or
                the prediction is unavailable.
              </EmptyState>
            </SectionCard>
          ) : null}

          {explanation.data && available ? (
            <>
              <SectionCard
                title="Explanation summary"
                description="Availability, provenance and additivity validation returned by the explanation service."
              >
                <ExplanationSummary explanation={explanation.data} />
              </SectionCard>

              <SectionCard
                title="SHAP feature contributions"
                description="Signed feature contributions sorted by magnitude. Positive values push the model output up; negative values push it down."
              >
                <ShapFeatureChart contributions={explanation.data.contributions} />
              </SectionCard>

              <p className="provenance">{explanation.data.disclaimer}</p>
            </>
          ) : null}

          {compareEnabled ? (
            <SectionCard
              title="Compare two hours"
              description="Differences between two available explanations for the same patient."
            >
              <div className="toolbar">
                <label className="field">
                  <span className="field__label">Earlier hour</span>
                  <select
                    className="select"
                    value={hourA ?? ""}
                    onChange={(event) => setHourA(Number(event.target.value))}
                    aria-label="Earlier comparison hour"
                  >
                    {hours.map((item) => (
                      <option key={item.hour} value={item.hour}>
                        Hour {item.hour}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="field">
                  <span className="field__label">Later hour</span>
                  <select
                    className="select"
                    value={hourB ?? ""}
                    onChange={(event) => setHourB(Number(event.target.value))}
                    aria-label="Later comparison hour"
                  >
                    {hours.map((item) => (
                      <option key={item.hour} value={item.hour}>
                        Hour {item.hour}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              {comparison.loading ? (
                <LoadingState label="Comparing explanations…" />
              ) : null}
              {comparison.error ? (
                <ErrorState error={comparison.error} onRetry={comparison.reload} />
              ) : null}
              {comparison.data && comparison.data.status !== "available" ? (
                <EmptyState title={`Comparison ${comparison.data.status}`}>
                  {comparison.data.reason ??
                    "At least one of the selected hours cannot be explained."}
                </EmptyState>
              ) : null}
              {comparison.data && comparison.data.status === "available" ? (
                <div className="table-wrap" style={{ marginTop: 14 }}>
                  <table>
                    <thead>
                      <tr>
                        <th scope="col">Feature</th>
                        <th scope="col">SHAP at {comparison.data.hour_a}</th>
                        <th scope="col">SHAP at {comparison.data.hour_b}</th>
                        <th scope="col">Change</th>
                      </tr>
                    </thead>
                    <tbody>
                      {comparison.data.largest_changes.map((change) => (
                        <tr key={change.feature_name}>
                          <th scope="row">{change.display_name}</th>
                          <td className="numeric">{change.shap_a.toFixed(4)}</td>
                          <td className="numeric">{change.shap_b.toFixed(4)}</td>
                          <td className="numeric">{change.delta.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : null}
            </SectionCard>
          ) : null}
        </>
      ) : null}

      <SectionCard
        title="Global feature importance"
        description="Mean absolute SHAP over the configured background sample for the active model."
      >
        {globalExplanation.loading ? (
          <LoadingState label="Computing global importance…" />
        ) : null}
        {globalExplanation.error ? (
          <ErrorState
            error={globalExplanation.error}
            onRetry={globalExplanation.reload}
          />
        ) : null}
        {globalExplanation.data && globalExplanation.data.status !== "available" ? (
          <EmptyState
            title={`Global explanation ${globalExplanation.data.status}`}
          >
            {globalExplanation.data.reason ?? "No global explanation is available."}
          </EmptyState>
        ) : null}
        {globalExplanation.data && globalExplanation.data.status === "available" ? (
          <div className="stack">
            <div className="row" style={{ gap: 8 }}>
              <span className="tiny muted">
                Model:{" "}
                {globalExplanation.data.model?.display_name ??
                  globalExplanation.data.model?.name ??
                  "unknown"}
              </span>
              {globalExplanation.data.model ? (
                <SourceBadge source={globalExplanation.data.model.source} />
              ) : null}
              <span className="badge badge--muted">
                {globalExplanation.data.background_rows} background rows
              </span>
            </div>
            <GlobalImportanceChart features={globalExplanation.data.features} />
            {globalExplanation.data.feature_groups.length > 0 ? (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th scope="col">Feature group</th>
                      <th scope="col">Mean |SHAP|</th>
                      <th scope="col">Features</th>
                    </tr>
                  </thead>
                  <tbody>
                    {globalExplanation.data.feature_groups.map((group) => (
                      <tr key={group.category}>
                        <th scope="row">{group.category}</th>
                        <td className="numeric">{group.mean_abs_shap.toFixed(4)}</td>
                        <td className="numeric">{group.n_features}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
            <p className="muted tiny">{globalExplanation.data.disclaimer}</p>
          </div>
        ) : null}
      </SectionCard>

      <SectionCard
        title="Explanation model catalogue"
        description="Which models can be explained, and with which method. Unsupported models are reported explicitly."
      >
        {catalog.loading ? <LoadingState label="Loading catalogue…" /> : null}
        {catalog.error ? (
          <ErrorState error={catalog.error} onRetry={catalog.reload} />
        ) : null}
        {catalog.data && catalog.data.models.length === 0 ? (
          <EmptyState title="No models are registered">
            The explanation service has no model bundles to inspect.
          </EmptyState>
        ) : null}
        {catalog.data && catalog.data.models.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Version</th>
                  <th scope="col">Source</th>
                  <th scope="col">Explainable</th>
                  <th scope="col">Method</th>
                  <th scope="col">Background rows</th>
                </tr>
              </thead>
              <tbody>
                {catalog.data.models.map((entry) => (
                  <tr key={`${entry.model.name}:${entry.model.version}`}>
                    <th scope="row">
                      {entry.model.display_name ?? entry.model.name}
                    </th>
                    <td>{entry.model.version}</td>
                    <td>
                      <SourceBadge source={entry.model.source} />
                    </td>
                    <td>
                      {entry.explainable
                        ? "Yes"
                        : `No — ${entry.reason ?? "unsupported"}`}
                    </td>
                    <td>{entry.explanation_method?.replaceAll("_", " ") ?? "—"}</td>
                    <td className="numeric">{entry.background_rows}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </SectionCard>

      <div className="disclaimer" role="note">
        <Info size={15} aria-hidden="true" />
        <span>
          <strong>Research output.</strong> SHAP values explain the fitted model,
          not clinical causation. Synthetic demonstration data is used unless a
          processed dataset and P1 artifact are configured. Do not use these
          explanations as clinical evidence.
        </span>
      </div>

    </div>
  );
}
