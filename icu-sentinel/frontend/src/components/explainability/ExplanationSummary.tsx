import { AlertTriangle, CheckCircle2, Info } from "lucide-react";

import type { PatientExplanation } from "../../api/types";
import { SourceBadge } from "../Badges";

function outputSpaceLabel(space: string | null): string {
  if (!space) return "Not reported";
  if (space === "probability") return "Probability space";
  if (space.includes("log_odds") || space === "decision")
    return "Log-odds / decision space";
  return space.replaceAll("_", " ");
}

export function ExplanationSummary({
  explanation,
}: {
  explanation: PatientExplanation;
}) {
  const isLogOdds =
    explanation.output_space !== null &&
    (explanation.output_space.includes("log_odds") ||
      explanation.output_space === "decision");

  return (
    <div className="stack">
      <dl className="summary-grid">
        <div className="summary-tile">
          <dt>Prediction probability</dt>
          <dd className="numeric">
            {explanation.prediction_probability === null
              ? "—"
              : explanation.prediction_probability.toFixed(4)}
          </dd>
        </div>
        <div className="summary-tile">
          <dt>Model output</dt>
          <dd className="numeric">
            {explanation.model_output === null
              ? "—"
              : explanation.model_output.toFixed(4)}
          </dd>
        </div>
        <div className="summary-tile">
          <dt>Base value</dt>
          <dd className="numeric">
            {explanation.base_value === null
              ? "—"
              : explanation.base_value.toFixed(4)}
          </dd>
        </div>
        <div className="summary-tile">
          <dt>Output space</dt>
          <dd>{outputSpaceLabel(explanation.output_space)}</dd>
        </div>
        <div className="summary-tile">
          <dt>Method</dt>
          <dd>{explanation.explanation_method.replaceAll("_", " ")}</dd>
        </div>
        <div className="summary-tile">
          <dt>Additivity check</dt>
          <dd>
            {explanation.validation ? (
              <span
                className={`badge ${
                  explanation.validation.passed ? "badge--ok" : "badge--warn"
                }`}
              >
                {explanation.validation.passed ? (
                  <CheckCircle2 size={11} aria-hidden="true" />
                ) : (
                  <AlertTriangle size={11} aria-hidden="true" />
                )}
                {explanation.validation.passed ? "Passed" : "Failed"}
              </span>
            ) : (
              "Not reported"
            )}
          </dd>
        </div>
      </dl>

      <div className="row" style={{ gap: 8 }}>
        {explanation.model ? (
          <>
            <span className="tiny muted">
              {explanation.model.display_name ?? explanation.model.name} · v
              {explanation.model.version}
            </span>
            <SourceBadge source={explanation.model.source} />
          </>
        ) : null}
        <span className="tiny muted">
          Generated {new Date(explanation.generated_at).toLocaleString()}
        </span>
      </div>

      {isLogOdds ? (
        <div className="output-space-note">
          <Info size={15} aria-hidden="true" style={{ flex: "0 0 auto" }} />
          <span>
            These SHAP values are in <strong>log-odds space</strong>. They explain
            how features move the model's decision score; they do not sum directly
            to the displayed probability.
          </span>
        </div>
      ) : null}

      {explanation.validation && !explanation.validation.passed ? (
        <p className="inline-error" role="alert">
          The additivity check did not pass for this explanation. Treat the values
          as diagnostic output only.
          {explanation.validation.note ? ` ${explanation.validation.note}` : ""}
        </p>
      ) : null}
    </div>
  );
}
