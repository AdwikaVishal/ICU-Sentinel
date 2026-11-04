import { alertsApi } from "../api/client";
import type { AlertListResponse } from "../api/types";
import { useAsync, type AsyncResource } from "./useAsync";

/** Shell-level notification source for the active alert queue. */
export function useActiveAlerts(): AsyncResource<AlertListResponse> {
  return useAsync((signal) => alertsApi.active(200, signal), []);
}
