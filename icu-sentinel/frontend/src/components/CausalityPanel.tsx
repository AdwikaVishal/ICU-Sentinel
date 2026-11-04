import type { CausalityMetadata } from "../api/types";

export function CausalityPanel({ causality }: { causality: CausalityMetadata }) {
  return (
    <div className="causality">
      <dl className="facts">
        <div>
          <dt>Latest hour used</dt>
          <dd>{causality.max_input_hour}</dd>
        </div>
        <div>
          <dt>Rows in causal prefix</dt>
          <dd>{causality.prefix_row_count}</dd>
        </div>
        <div>
          <dt>Observations in prefix</dt>
          <dd>{causality.input_observation_count}</dd>
        </div>
        <div>
          <dt>Future data used</dt>
          <dd>
            <span
              className={`badge ${
                causality.future_data_used ? "badge--warn" : "badge--ok"
              }`}
            >
              {causality.future_data_used ? "Yes" : "No"}
            </span>
          </dd>
        </div>
        <div>
          <dt>Retrospective label as input</dt>
          <dd>
            <span
              className={`badge ${
                causality.retrospective_label_used_as_input
                  ? "badge--warn"
                  : "badge--ok"
              }`}
            >
              {causality.retrospective_label_used_as_input ? "Yes" : "No"}
            </span>
          </dd>
        </div>
        <div>
          <dt>Feature schema</dt>
          <dd>{causality.feature_schema_version ?? "—"}</dd>
        </div>
      </dl>
      <p className="muted small">
        Model input hash:{" "}
        <code className="hash">{causality.model_input_hash ?? "—"}</code>
      </p>
      <p className="muted small">
        The input is built from observations up to and including the selected hour
        only. Changing later hours cannot change this prediction.
      </p>
    </div>
  );
}
