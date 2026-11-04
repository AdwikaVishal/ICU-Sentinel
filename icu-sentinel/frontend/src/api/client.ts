import type {
  AlertClearRequest,
  AlertListResponse,
  AlertRecord,
  AlertThresholdInfo,
  CausalInputResponse,
  FeatureMetadata,
  GlobalExplanation,
  HourlyExplanationComparison,
  ModelCatalogResponse,
  PatientExplanation,
  PatientListResponse,
  PatientSummary,
  ProviderStatus,
  ReplayState,
  TimelineResponse,
} from "./types";

/** Base URL for the P3 API. Empty string means same-origin / Vite proxy. */
export const API_BASE: string = (import.meta.env.VITE_P3_API_BASE ?? "").replace(
  /\/$/,
  "",
);

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;

  constructor(status: number, code: string, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

type QueryValue = string | number | boolean | null | undefined;

function buildQuery(query?: Record<string, QueryValue>): string {
  if (!query) return "";
  const params = new URLSearchParams();
  for (const [key, value] of Object.entries(query)) {
    if (value === null || value === undefined) continue;
    params.set(key, String(value));
  }
  const encoded = params.toString();
  return encoded ? `?${encoded}` : "";
}

async function request<T>(
  path: string,
  init: RequestInit = {},
  signal?: AbortSignal,
): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body !== undefined && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set("Accept", "application/json");

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw error;
    }
    throw new ApiError(
      0,
      "network_error",
      "Could not reach the P3 API. Is the backend running on port 8000?",
    );
  }

  const text = await response.text();
  let payload: unknown = null;
  if (text) {
    try {
      payload = JSON.parse(text) as unknown;
    } catch {
      payload = text;
    }
  }

  if (!response.ok) {
    const body = (payload ?? {}) as {
      error?: string;
      detail?: string | Array<{ msg?: string; loc?: unknown[] }>;
    };
    const detail =
      typeof body.detail === "string"
        ? body.detail
        : Array.isArray(body.detail)
          ? body.detail
              .map((item) =>
                item.msg
                  ? `${item.loc?.slice(1).join(".") ?? "request"}: ${item.msg}`
                  : "Invalid request",
              )
              .join("; ")
          : response.statusText || "Request failed";
    throw new ApiError(
      response.status,
      body.error ?? "http_error",
      detail,
    );
  }

  return payload as T;
}

export const replayApi = {
  listPatients(signal?: AbortSignal): Promise<PatientListResponse> {
    return request<PatientListResponse>("/api/v3/patients", {}, signal);
  },

  getPatient(patientId: string, signal?: AbortSignal): Promise<PatientSummary> {
    return request<PatientSummary>(
      `/api/v3/patients/${encodeURIComponent(patientId)}`,
      {},
      signal,
    );
  },

  getTimeline(patientId: string, signal?: AbortSignal): Promise<TimelineResponse> {
    return request<TimelineResponse>(
      `/api/v3/patients/${encodeURIComponent(patientId)}/timeline`,
      {},
      signal,
    );
  },

  getState(
    patientId: string,
    hour?: number,
    evaluateAlerts = false,
    signal?: AbortSignal,
  ): Promise<ReplayState> {
    return request<ReplayState>(
      `/api/v3/replay/${encodeURIComponent(patientId)}${buildQuery({
        hour,
        evaluate_alerts: evaluateAlerts,
      })}`,
      {},
      signal,
    );
  },

  step(
    patientId: string,
    hour: number,
    direction: "forward" | "backward",
    evaluateAlerts = true,
    signal?: AbortSignal,
  ): Promise<ReplayState> {
    return request<ReplayState>(
      `/api/v3/replay/${encodeURIComponent(patientId)}/step`,
      {
        method: "POST",
        body: JSON.stringify({ hour, direction, evaluate_alerts: evaluateAlerts }),
      },
      signal,
    );
  },

  jump(
    patientId: string,
    hour: number,
    evaluateAlerts = true,
    signal?: AbortSignal,
  ): Promise<ReplayState> {
    return request<ReplayState>(
      `/api/v3/replay/${encodeURIComponent(patientId)}/jump`,
      {
        method: "POST",
        body: JSON.stringify({ hour, evaluate_alerts: evaluateAlerts }),
      },
      signal,
    );
  },

  getModelInput(
    patientId: string,
    hour: number,
    signal?: AbortSignal,
  ): Promise<CausalInputResponse> {
    return request<CausalInputResponse>(
      `/api/v3/replay/${encodeURIComponent(patientId)}/model-input${buildQuery({ hour })}`,
      {},
      signal,
    );
  },

  getProviderStatus(signal?: AbortSignal): Promise<ProviderStatus> {
    return request<ProviderStatus>("/api/v3/provider/status", {}, signal);
  },

  getThreshold(signal?: AbortSignal): Promise<AlertThresholdInfo | null> {
    return request<AlertThresholdInfo | null>(
      "/api/v3/provider/threshold",
      {},
      signal,
    );
  },
};

export const explanationsApi = {
  listModels(signal?: AbortSignal): Promise<ModelCatalogResponse> {
    return request<ModelCatalogResponse>("/api/v3/explanations/models", {}, signal);
  },

  global(
    modelName?: string,
    modelVersion?: string,
    signal?: AbortSignal,
  ): Promise<GlobalExplanation> {
    return request<GlobalExplanation>(
      `/api/v3/explanations/global${buildQuery({
        model_name: modelName,
        model_version: modelVersion,
      })}`,
      {},
      signal,
    );
  },

  feature(
    featureName: string,
    signal?: AbortSignal,
  ): Promise<FeatureMetadata> {
    return request<FeatureMetadata>(
      `/api/v3/explanations/features/${encodeURIComponent(featureName)}`,
      {},
      signal,
    );
  },

  explain(
    patientId: string,
    hour: number,
    signal?: AbortSignal,
  ): Promise<PatientExplanation> {
    return request<PatientExplanation>(
      `/api/v3/explanations/${encodeURIComponent(patientId)}/${hour}`,
      {},
      signal,
    );
  },

  compare(
    patientId: string,
    hourA: number,
    hourB: number,
    signal?: AbortSignal,
  ): Promise<HourlyExplanationComparison> {
    return request<HourlyExplanationComparison>(
      `/api/v3/explanations/${encodeURIComponent(patientId)}/compare${buildQuery({
        hour_a: hourA,
        hour_b: hourB,
      })}`,
      {},
      signal,
    );
  },
};

export const alertsApi = {
  list(
    filters: { status?: string; patientId?: string; limit?: number } = {},
    signal?: AbortSignal,
  ): Promise<AlertListResponse> {
    return request<AlertListResponse>(
      `/api/v3/alerts${buildQuery({
        status: filters.status,
        patient_id: filters.patientId,
        limit: filters.limit,
      })}`,
      {},
      signal,
    );
  },

  active(limit = 200, signal?: AbortSignal): Promise<AlertListResponse> {
    return request<AlertListResponse>(
      `/api/v3/alerts/active${buildQuery({ limit })}`,
      {},
      signal,
    );
  },

  history(limit = 200, signal?: AbortSignal): Promise<AlertListResponse> {
    return request<AlertListResponse>(
      `/api/v3/alerts/history${buildQuery({ limit })}`,
      {},
      signal,
    );
  },

  threshold(signal?: AbortSignal): Promise<AlertThresholdInfo> {
    return request<AlertThresholdInfo>("/api/v3/alerts/threshold", {}, signal);
  },

  get(alertId: string, signal?: AbortSignal): Promise<AlertRecord> {
    return request<AlertRecord>(
      `/api/v3/alerts/${encodeURIComponent(alertId)}`,
      {},
      signal,
    );
  },

  acknowledge(alertId: string, signal?: AbortSignal): Promise<AlertRecord> {
    return request<AlertRecord>(
      `/api/v3/alerts/${encodeURIComponent(alertId)}/acknowledge`,
      { method: "POST" },
      signal,
    );
  },

  clear(
    alertId: string,
    body: AlertClearRequest = { reason: "manual_clear" },
    signal?: AbortSignal,
  ): Promise<AlertRecord> {
    return request<AlertRecord>(
      `/api/v3/alerts/${encodeURIComponent(alertId)}/clear`,
      { method: "POST", body: JSON.stringify(body) },
      signal,
    );
  },
};
