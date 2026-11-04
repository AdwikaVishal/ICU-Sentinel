import { replayApi } from "../api/client";
import type {
  AlertThresholdInfo,
  PatientListResponse,
  ProviderStatus,
  TimelineResponse,
} from "../api/types";
import { useAsync, type AsyncResource } from "./useAsync";

export function usePatients(): AsyncResource<PatientListResponse> {
  return useAsync((signal) => replayApi.listPatients(signal), []);
}

export function useProviderStatus(): AsyncResource<ProviderStatus> {
  return useAsync((signal) => replayApi.getProviderStatus(signal), []);
}

export function useTimeline(patientId: string | null): AsyncResource<TimelineResponse> {
  return useAsync(
    (signal) => replayApi.getTimeline(patientId as string, signal),
    [patientId],
    Boolean(patientId),
  );
}

export function useThreshold(): AsyncResource<AlertThresholdInfo | null> {
  return useAsync((signal) => replayApi.getThreshold(signal), []);
}
