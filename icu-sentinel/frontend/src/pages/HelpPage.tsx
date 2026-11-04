import {
  Activity,
  BellRing,
  Brain,
  Database,
  HelpCircle,
  ShieldCheck,
} from "lucide-react";
import { Link } from "react-router-dom";

import { PageHeader } from "../components/common/PageHeader";
import { SectionCard } from "../components/common/SectionCard";

const FAQS = [
  {
    question: "Is this clinical decision support?",
    answer:
      "No. ICU Sentinel is a retrospective research prototype. Risk scores, feature attributions and alert episodes are not diagnoses, treatment recommendations or validated clinical alarms. Do not use this interface for patient care.",
  },
  {
    question: "What data is shown?",
    answer:
      "The default workspace loads a synthetic demonstration cohort. When a processed dataset is configured by the deployment, the interface labels the active data source. Patient identifiers are limited to the configured research records.",
  },
  {
    question: "Why is the risk score missing at some hours?",
    answer:
      "The replay service reports explicit availability for each hour. A missing or unavailable score is shown as unavailable rather than replaced with a zero or inferred value. Missing observations are also disclosed.",
  },
  {
    question: "What does a SHAP contribution mean?",
    answer:
      "A SHAP value describes how a feature moved the model output for one prediction relative to the model baseline. It explains the model, not the patient's physiology or the cause of illness. Linear SHAP values are reported in the model's decision-function space.",
  },
  {
    question: "Why does an alert exist when the score is below the threshold?",
    answer:
      "Alert episodes are workflow records created by a configured rule. They retain the threshold and model identity that produced them. A later score can update or clear an episode according to the configured deduplication rules. Alert history is preserved for review.",
  },
  {
    question: "Can I change the model or threshold here?",
    answer:
      "No. Model, feature-schema and threshold configuration are deployment concerns. The Settings page is a read-only view of the active backend configuration. The only write actions in this interface are acknowledging and clearing research alert episodes.",
  },
];

export function HelpPage() {
  return (
    <div className="page">
      <div className="help-hero">
        <PageHeader
          eyebrow={
            <>
              <HelpCircle size={12} aria-hidden="true" />
              Documentation
            </>
          }
          title="Help & Research Guide"
          description="How the ICU Sentinel research workspace works, what its outputs mean and where its limits are. The same provenance and availability rules apply across every page."
        />
      </div>

      <div className="help-grid">
        <article className="help-card">
          <span className="help-card__icon" aria-hidden="true">
            <Activity size={20} />
          </span>
          <h2>Patient Replay</h2>
          <p>
            Replay a recorded ICU stay hour by hour. Each prediction uses only the
            selected hour and earlier observations; retrospective labels are never
            used as model inputs.
          </p>
          <ul>
            <li>Choose a patient and step or jump between recorded hours.</li>
            <li>Review observations, prediction availability and alert state.</li>
            <li>Missing hours remain visible and are never silently filled.</li>
          </ul>
          <Link className="section-heading__link" to="/replay">
            Open Patient Replay
          </Link>
        </article>

        <article className="help-card">
          <span className="help-card__icon" aria-hidden="true">
            <Brain size={20} />
          </span>
          <h2>Explanations (SHAP)</h2>
          <p>
            Feature contributions explain the configured model's output at a chosen
            hour. Positive contributions push the model output upward; negative
            contributions push it downward.
          </p>
          <ul>
            <li>Per-patient explanations include additive validation when reported.</li>
            <li>Global importance summarises the model's background cohort.</li>
            <li>Attribution is not causation and does not establish clinical validity.</li>
          </ul>
          <Link className="section-heading__link" to="/explanations">
            Open Explanations
          </Link>
        </article>

        <article className="help-card">
          <span className="help-card__icon" aria-hidden="true">
            <BellRing size={20} />
          </span>
          <h2>Alerts</h2>
          <p>
            Alerts are deduplicated research episodes generated when a risk score
            crosses the configured demonstration threshold. Acknowledging or clearing
            an alert records workflow state only.
          </p>
          <ul>
            <li>Active, acknowledged and cleared episodes are retained for review.</li>
            <li>Threshold source, version and validation status are always disclosed.</li>
            <li>An empty queue does not prove that a patient is stable.</li>
          </ul>
          <Link className="section-heading__link" to="/alerts">
            Open Alerts
          </Link>
        </article>

        <article className="help-card">
          <span className="help-card__icon" aria-hidden="true">
            <Database size={20} />
          </span>
          <h2>Data & provenance</h2>
          <p>
            The interface displays the data source, synthetic status, model identity
            and feature-schema version reported by the backend. Demo data is labelled
            wherever it appears.
          </p>
          <ul>
            <li>Synthetic records are for demonstration and software testing.</li>
            <li>Deployments using processed data must follow data-governance policy.</li>
            <li>Evaluation metrics are shown only when the backend reports them.</li>
          </ul>
          <Link className="section-heading__link" to="/settings">
            View configuration
          </Link>
        </article>
      </div>

      <SectionCard
        title="Research boundaries"
        description="Non-negotiable limits of this prototype."
        actions={<ShieldCheck size={18} aria-hidden="true" />}
      >
        <div className="evaluation-empty">
          <strong>Research only — not for clinical use</strong>
          <p>
            ICU Sentinel is a retrospective model-behaviour research tool. It has not
            been prospectively validated, does not replace clinical judgement and must
            not be used to diagnose sepsis, triage patients or direct treatment.
            Predictions, explanations and alerts may be incomplete, miscalibrated or
            unavailable, and must be interpreted together with the disclosed model and
            data provenance.
          </p>
        </div>
      </SectionCard>

      <SectionCard
        title="Frequently asked questions"
        description="Common questions about scores, explanations and alert workflow."
      >
        <div className="faq-list">
          {FAQS.map((item) => (
            <details className="faq-item" key={item.question}>
              <summary>{item.question}</summary>
              <p>{item.answer}</p>
            </details>
          ))}
        </div>
      </SectionCard>
    </div>
  );
}
