import { replayApi } from "../api/client";
import type { TimelineResponse } from "../api/types";
import { useAsync, type AsyncResource } from "./useAsync";

export interface RiskPoint {
  hour: number;
  probability: number | null;
  status: string;
}

const MAX_PREVIEW_HOURS = 36;

/**
 * Fetch the actual replay score for each recorded hour of one patient.
 *
 * The P3 API does not expose a batch prediction endpoint, so the preview is
 * assembled from the existing replay endpoint. The number of preview requests
 * is capped to keep the dashboard responsive; the chart labels the truncation.
 */
export function useRiskSeries(
  patientId: string | null,
  timeline: TimelineResponse | null,
): AsyncResource<RiskPoint[]> {
  const hours = (timeline?.hours ?? [])
    .slice(0, MAX_PREVIEW_HOURS)
    .map((item) => item.hour);
  const enabled = Boolean(patientId) && hours.length > 0;

  return useAsync(
    async (signal) => {
      const points = await Promise.all(
        hours.map(async (hour) => {
          const state = await replayApi.getState(
            patientId as string,
            hour,
            false,
            signal,
          );
          return {
            hour,
            probability: state.prediction.available
              ? state.prediction.probability
              : null,
            status: state.prediction_status,
          } satisfies RiskPoint;
        }),
      );
      return points;
    },
    [patientId, timeline],
    enabled,
  );
}

export const RISK_PREVIEW_LIMIT = MAX_PREVIEW_HOURS;
