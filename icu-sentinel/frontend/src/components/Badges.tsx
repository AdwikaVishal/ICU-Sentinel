import type { AlertStatus, PredictionSource } from "../api/types";

const SOURCE_LABELS: Record<PredictionSource, string> = {
  synthetic_heuristic: "Synthetic heuristic (demo)",
  synthetic_demo_model: "Synthetic demo model",
  p1_classical_model: "P1 classical model",
  p2_boosting_model: "P2 boosting model",
  p3_temporal_model: "P3 temporal model",
  unavailable: "Unavailable",
};

export function SourceBadge({ source }: { source: PredictionSource | string }) {
  const synthetic = source.startsWith("synthetic");
  return (
    <span className={`badge ${synthetic ? "badge--synthetic" : "badge--model"}`}>
      {SOURCE_LABELS[source as PredictionSource] ?? source}
    </span>
  );
}

export function StatusBadge({ status }: { status: AlertStatus }) {
  return (
    <span className={`badge badge--status badge--${status.toLowerCase()}`}>
      {status}
    </span>
  );
}

export function AvailabilityBadge({
  available,
  label,
}: {
  available: boolean;
  label: string;
}) {
  return (
    <span className={`badge ${available ? "badge--ok" : "badge--muted"}`}>
      {label}
    </span>
  );
}
