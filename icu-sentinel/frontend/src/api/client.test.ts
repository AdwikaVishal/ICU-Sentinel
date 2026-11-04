import { describe, expect, it, vi, beforeEach, afterEach } from "vitest";

import { ApiError, alertsApi, explanationsApi, replayApi } from "./client";
import type { PatientListResponse } from "./types";

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

describe("API client", () => {
  const fetchMock = vi.fn<typeof fetch>();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("returns parsed patient data", async () => {
    const payload: PatientListResponse = {
      patients: [],
      count: 0,
      data_source: "synthetic_demo",
      is_synthetic: true,
      prediction_source: "synthetic_demo_model",
      provider: "demo",
      disclaimer: "Not a diagnosis.",
    };
    fetchMock.mockResolvedValue(jsonResponse(200, payload));

    const result = await replayApi.listPatients();

    expect(result).toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/v3/patients",
      expect.objectContaining({ headers: expect.any(Headers) }),
    );
  });

  it("encodes query parameters and omits empty values", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { alerts: [], count: 0, filters: {} }));

    await alertsApi.list({ patientId: "DEMO 0004", status: undefined, limit: 25 });

    const url = String(fetchMock.mock.calls[0]?.[0]);
    expect(url).toContain("patient_id=DEMO+0004");
    expect(url).toContain("limit=25");
    expect(url).not.toContain("status=");
  });

  it("maps structured API errors to ApiError", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(404, {
        error: "patient_not_found",
        detail: "Patient not found: NOPE",
      }),
    );

    await expect(replayApi.getPatient("NOPE")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      code: "patient_not_found",
      message: "Patient not found: NOPE",
    });
  });

  it("reports validation errors from FastAPI detail arrays", async () => {
    fetchMock.mockResolvedValue(
      jsonResponse(422, {
        detail: [{ loc: ["body", "probability"], msg: "Input should be <= 1" }],
      }),
    );

    try {
      await alertsApi.clear("alert_1", { reason: "" });
      expect.fail("expected the request to reject");
    } catch (error) {
      expect(error).toBeInstanceOf(ApiError);
      expect((error as ApiError).status).toBe(422);
      expect((error as ApiError).message).toContain("probability");
    }
  });

  it("uses a readable network error when fetch fails", async () => {
    fetchMock.mockRejectedValue(new TypeError("Failed to fetch"));

    await expect(replayApi.listPatients()).rejects.toMatchObject({
      name: "ApiError",
      code: "network_error",
      status: 0,
    });
  });

  it("sends JSON bodies for mutations", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { action: "none", alert: null }));

    await alertsApi.acknowledge("alert_1");

    const [, init] = fetchMock.mock.calls[0] ?? [];
    expect(init?.method).toBe("POST");
  });

  it("builds patient explanation paths with encoded identifiers", async () => {
    fetchMock.mockResolvedValue(jsonResponse(200, { status: "unavailable" }));

    await explanationsApi.explain("patient/with slash", 12);

    expect(fetchMock.mock.calls[0]?.[0]).toBe(
      "/api/v3/explanations/patient%2Fwith%20slash/12",
    );
  });
});
