import type { ReactNode } from "react";

/**
 * Persistent provenance banner. The wording deliberately avoids alarmist
 * language: this is a research prototype, not a diagnosis.
 */
export function DisclaimerBar({ children }: { children?: ReactNode }) {
  return (
    <aside className="disclaimer" role="note">
      <strong>Research prototype.</strong>{" "}
      {children ??
        "Synthetic demonstration data, model scores and alerts are for retrospective evaluation only. They are not a diagnosis, do not confirm sepsis, and must not be used for clinical decisions."}
    </aside>
  );
}

export function SyntheticNote({ isSynthetic }: { isSynthetic: boolean }) {
  return (
    <p className="provenance">
      {isSynthetic
        ? "Data source: synthetic demonstration cohort. Nothing on this page is real patient data."
        : "Data source: configured processed dataset. Follow the deployment data-governance policy."}
    </p>
  );
}
