import { useEffect, useState } from "react";

import { EmptyPanel, ErrorPanel, LoadingPanel, Section } from "../components/AsyncState";
import { GlobalImportanceChart } from "../components/GlobalImportanceChart";
import { ContributionChart } from "../components/ContributionChart";
import { SyntheticNote } from "../components/Disclaimer";
import { PatientSelector } from "../components/PatientSelector";
import { SourceBadge } from "../components/Badges";
import {
  useGlobalExplanation,
  useHourComparison,
  useModelCatalog,
  usePatientExplanation,
} from "../hooks/useExplanation";
import { usePatients, useTimeline } from "../hooks/useReplayResources";

export function ExplanationCentrePage() {
  const patients = usePatients();
  const [selected, setSelected] = useState<string | null>(null);
  const timeline = useTimeline(selected);
  const [hour, setHour] = useState<number | null>(null);
  const [compareEnabled, setCompareEnabled] = useState(false);
  const [hourA, setHourA] = useState<number | null>(null);
  const [hourB, setHourB] = useState<number | null>(null);

  const explanation = usePatientExplanation(selected, hour);
  const comparison = useHourComparison(selected, hourA, hourB, compareEnabled);
  const catalog = useModelCatalog();
  const globalExplanation = useGlobalExplanation();

  useEffect(() => {
    if (!selected && patients.data && patients.data.patients.length > 0) {
      setSelected(patients.data.patients[0].patient_id);
    }
  }, [patients.data, selected]);

  useEffect(() => {
    setHour(null);
    setHourA(null);
    setHourB(null);
  }, [selected]);

  useEffect(() => {
    const data = timeline.data;
    if (!data || data.patient.patient_id !== selected) return;
    if (data.hours.length === 0) return;
    const firstHour = data.hours[0].hour;
    const lastHour = data.hours[data.hours.length - 1].hour;
    setHour((current) =>
      current === null || current < firstHour || current > lastHour ? firstHour : current,
    );
    setHourA((current) => current ?? firstHour);
    setHourB((current) => current ?? lastHour);
  }, [timeline.data, selected]);

  const hours = timeline.data?.hours ?? [];

  return (
    <div className="page">
      <div className="page__heading">
        <div>
          <h2>Explanation Centre</h2>
          <p className="muted">
            SHAP values for the fitted demonstration model, or an explicit
            unavailable/unsupported state when a model cannot be explained.
          </p>
        </div>
      </div>

      {patients.loading ? <LoadingPanel label="Loading patients…" /> : null}
      {patients.error ? (
        <ErrorPanel error={patients.error} onRetry={patients.reload} />
      ) : null}
      {patients.data && patients.data.patients.length === 0 ? (
        <EmptyPanel title="No patients are available">
          Configure processed data or enable the synthetic demonstration cohort.
        </EmptyPanel>
      ) : null}

      <Section
        title="Model explanations for a patient-hour"
        description="Explanations are computed from the same causal input as the prediction. No SHAP value is fabricated for unsupported models."
      >
        <div className="toolbar">
          <PatientSelector
            patients={patients.data?.patients ?? []}
            value={selected}
            onChange={setSelected}
            disabled={patients.loading}
          />
          <label className="field">
            <span className="field__label">Recorded hour</span>
            <select
              value={hour ?? ""}
              onChange={(event) => setHour(Number(event.target.value))}
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
        </div>

        {timeline.loading ? <p className="muted">Loading timeline…</p> : null}
        {timeline.error ? <ErrorPanel error={timeline.error} onRetry={timeline.reload} /> : null}

        {!selected ? (
          <EmptyPanel title="Select a patient">
            Choose a patient to inspect the explanation at a recorded hour.
          </EmptyPanel>
        ) : null}
        {selected && !timeline.loading && hours.length === 0 ? (
          <EmptyPanel title="This patient has no recorded hours">
            A causal prefix cannot be built without recorded observations.
          </EmptyPanel>
        ) : null}

        {explanation.loading ? <LoadingPanel label="Computing SHAP explanation…" /> : null}
        {explanation.error ? (
          <ErrorPanel error={explanation.error} onRetry={explanation.reload} />
        ) : null}

        {explanation.data && explanation.data.status !== "available" ? (
          <div className="state-panel state-panel--empty" role="status">
            <div>
              <strong>
                Explanation {explanation.data.status.replaceAll("_", " ")}
              </strong>
              <p>{explanation.data.reason ?? "No explanation is available."}</p>
              <p className="muted small">
                No SHAP values are shown when the model cannot be explained or the
                prediction is unavailable.
              </p>
            </div>
          </div>
        ) : null}

        {explanation.data && explanation.data.status === "available" ? (
          <div className="stack">
            <dl className="facts facts--wide">
              <div>
                <dt>Model</dt>
                <dd>
                  {explanation.data.model?.display_name ?? explanation.data.model?.name}{" "}
                  {explanation.data.model ? (
                    <SourceBadge source={explanation.data.model.source} />
                  ) : null}
                </dd>
              </div>
              <div>
                <dt>Method</dt>
                <dd>{explanation.data.explanation_method.replaceAll("_", " ")}</dd>
              </div>
              <div>
                <dt>Output space</dt>
                <dd>{explanation.data.output_space ?? "—"}</dd>
              </div>
              <div>
                <dt>Base value</dt>
                <dd className="numeric">
                  {explanation.data.base_value?.toFixed(4) ?? "—"}
                </dd>
              </div>
              <div>
                <dt>Model output</dt>
                <dd className="numeric">
                  {explanation.data.model_output?.toFixed(4) ?? "—"}
                </dd>
              </div>
              <div>
                <dt>Additivity check</dt>
                <dd>
                  {explanation.data.validation ? (
                    <span
                      className={`badge ${
                        explanation.data.validation.passed ? "badge--ok" : "badge--warn"
                      }`}
                    >
                      {explanation.data.validation.passed ? "Passed" : "Failed"}
                      {explanation.data.validation.additive_error !== null
                        ? ` · error ${explanation.data.validation.additive_error.toExponential(2)}`
                        : ""}
                    </span>
                  ) : (
                    "Not reported"
                  )}
                </dd>
              </div>
            </dl>

            {explanation.data.validation && !explanation.data.validation.passed ? (
              <p className="inline-error" role="alert">
                The additivity check did not pass for this explanation. Treat the
                values as diagnostic output only.
              </p>
            ) : null}

            <ContributionChart
              contributions={explanation.data.contributions}
              outputSpace={explanation.data.output_space}
            />
            <p className="muted small">{explanation.data.disclaimer}</p>
          </div>
        ) : null}
      </Section>

      <Section
        title="Compare two hours"
        description="Differences between two available explanations for the same patient."
        actions={
          <label className="field field--inline">
            <input
              type="checkbox"
              checked={compareEnabled}
              onChange={(event) => setCompareEnabled(event.target.checked)}
            />
            <span>Show comparison</span>
          </label>
        }
      >
        {!compareEnabled ? (
          <p className="muted">
            Enable the comparison to fetch explanations for two recorded hours.
          </p>
        ) : (
          <>
            <div className="toolbar">
              <label className="field">
                <span className="field__label">Earlier hour</span>
                <select
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

            {comparison.loading ? <LoadingPanel label="Comparing explanations…" /> : null}
            {comparison.error ? (
              <ErrorPanel error={comparison.error} onRetry={comparison.reload} />
            ) : null}
            {comparison.data && comparison.data.status !== "available" ? (
              <EmptyPanel title={`Comparison ${comparison.data.status}`}>
                {comparison.data.reason ??
                  "At least one of the selected hours cannot be explained."}
              </EmptyPanel>
            ) : null}
            {comparison.data && comparison.data.status === "available" ? (
              <div className="table-wrap">
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
          </>
        )}
      </Section>

      <Section
        title="Global feature importance"
        description="Mean absolute SHAP over a synthetic background sample for the active model."
      >
        {globalExplanation.loading ? <LoadingPanel label="Computing global importance…" /> : null}
        {globalExplanation.error ? (
          <ErrorPanel error={globalExplanation.error} onRetry={globalExplanation.reload} />
        ) : null}
        {globalExplanation.data && globalExplanation.data.status !== "available" ? (
          <EmptyPanel title={`Global explanation ${globalExplanation.data.status}`}>
            {globalExplanation.data.reason ?? "No global explanation is available."}
          </EmptyPanel>
        ) : null}
        {globalExplanation.data && globalExplanation.data.status === "available" ? (
          <div className="stack">
            <p className="muted small">
              Model:{" "}
              {globalExplanation.data.model?.display_name ??
                globalExplanation.data.model?.name ??
                "unknown"}{" "}
              {globalExplanation.data.model ? (
                <SourceBadge source={globalExplanation.data.model.source} />
              ) : null}
            </p>
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
            <p className="muted small">{globalExplanation.data.disclaimer}</p>
          </div>
        ) : null}
      </Section>

      <Section
        title="Explanation model catalogue"
        description="Which models can be explained, and with which method. Unsupported models are reported explicitly."
      >
        {catalog.loading ? <LoadingPanel label="Loading catalogue…" /> : null}
        {catalog.error ? <ErrorPanel error={catalog.error} onRetry={catalog.reload} /> : null}
        {catalog.data && catalog.data.models.length === 0 ? (
          <EmptyPanel title="No models are registered">
            The explanation service has no model bundles to inspect.
          </EmptyPanel>
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
                    <th scope="row">{entry.model.display_name ?? entry.model.name}</th>
                    <td>{entry.model.version}</td>
                    <td>
                      <SourceBadge source={entry.model.source} />
                    </td>
                    <td>{entry.explainable ? "Yes" : `No — ${entry.reason ?? "unsupported"}`}</td>
                    <td>{entry.explanation_method?.replaceAll("_", " ") ?? "—"}</td>
                    <td className="numeric">{entry.background_rows}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Section>

      {selected ? <SyntheticNote isSynthetic={patients.data?.is_synthetic ?? true} /> : null}
    </div>
  );
}
