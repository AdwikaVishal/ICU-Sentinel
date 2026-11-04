import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";

import { App } from "./App";

function response(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function stubFetch(routes: Array<[string, unknown]>) {
  const fetchMock = vi.fn((input: RequestInfo | URL) => {
    const url = String(input);
    const match = routes.find(([fragment]) => url.includes(fragment));
    if (match === undefined) {
      return Promise.resolve(response({ error: "not_found", detail: url }, 404));
    }
    return Promise.resolve(response(match[1]));
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const EMPTY_PATIENTS = {
  patients: [],
  count: 0,
  data_source: "synthetic_demo",
  is_synthetic: true,
  prediction_source: "synthetic_demo_model",
  provider: "demo",
  disclaimer: "Not a diagnosis.",
};

const EMPTY_ALERTS = { alerts: [], count: 0, filters: {} };

const THRESHOLD = {
  rule_id: "demo_threshold",
  threshold: 0.6,
  source: "demo_configuration",
  version: "demo-threshold-v1",
  dedup_window_hours: 6,
  clear_on_below_threshold: true,
  clinically_validated: false,
  disclosure:
    "Configured demonstration threshold. Not a clinically validated decision threshold and not a diagnosis.",
};

const PROVIDER = {
  available: true,
  provider_name: "demo",
  model: null,
  reason: null,
  required_features: null,
};

const CATALOG = { models: [], count: 0 };

const GLOBAL_EXPLANATION = {
  status: "unavailable",
  model: null,
  explanation_method: "not_applicable",
  output_space: null,
  features: [],
  feature_groups: [],
  background_rows: 0,
  reason: "No model bundle",
  metadata: null,
  generated_at: "2026-01-01T00:00:00Z",
  disclaimer: "Research only.",
};

function defaultRoutes(): Array<[string, unknown]> {
  return [
    ["/api/v3/alerts/threshold", THRESHOLD],
    ["/api/v3/alerts/active", EMPTY_ALERTS],
    ["/api/v3/alerts/history", EMPTY_ALERTS],
    ["/api/v3/alerts", EMPTY_ALERTS],
    ["/api/v3/explanations/models", CATALOG],
    ["/api/v3/explanations/global", GLOBAL_EXPLANATION],
    ["/api/v3/provider/status", PROVIDER],
    ["/api/v3/provider/threshold", THRESHOLD],
    ["/api/v3/patients", EMPTY_PATIENTS],
  ];
}

function stubApi(overrides: Array<[string, unknown]> = []) {
  return stubFetch([...overrides, ...defaultRoutes()]);
}

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("App", () => {
  it("renders the research dashboard with live system status", async () => {
    stubApi();

    renderAt("/");

    expect(
      await screen.findByRole("heading", { name: /Sepsis in the ICU/i }),
    ).toBeInTheDocument();
    expect(screen.getByText("Key Modules")).toBeInTheDocument();
    expect(screen.getByText("Synthetic demonstration data")).toBeInTheDocument();
    expect(await screen.findByText("Synthetic Demo")).toBeInTheDocument();
    expect(screen.getByText("Demo Patients")).toBeInTheDocument();
  });

  it("renders the alert workflow with threshold disclosure and empty states", async () => {
    stubApi();

    renderAt("/alerts");

    expect(
      await screen.findByRole("heading", { name: "Sepsis Risk Alerts" }),
    ).toBeInTheDocument();

    expect(await screen.findByText("Not clinically validated")).toBeInTheDocument();
    expect(screen.getByText("Deduplication")).toBeInTheDocument();
    expect(await screen.findByText("No alert episodes match this view")).toBeInTheDocument();
    expect(
      screen.getByText(/Research workflow only/i, { exact: false }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/not clinical advice/i, { selector: "footer p" }),
    ).toBeInTheDocument();
  });

  it("shows an explicit empty state when no patients are configured", async () => {
    stubApi();

    renderAt("/replay");

    expect(await screen.findByText("No patients are available")).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: "Patient Replay" })).toBeInTheDocument();
  });

  it("renders the explanation page without fabricating SHAP values", async () => {
    stubApi();

    renderAt("/explanations");

    expect(await screen.findByText("No patients are available")).toBeInTheDocument();
    expect(
      screen.getByText("No models are registered"),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Global explanation unavailable"),
    ).toBeInTheDocument();
  });

  it("renders read-only configuration on the settings page", async () => {
    stubApi();

    renderAt("/settings");

    expect(
      await screen.findByRole("heading", { name: "Settings" }),
    ).toBeInTheDocument();
    expect(screen.getAllByText("Data provenance").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Alert threshold").length).toBeGreaterThan(0);
    expect(
      (await screen.findAllByText("Clinically validated")).length,
    ).toBeGreaterThan(0);
    expect(
      screen.getByText(/cannot be changed from this interface/i),
    ).toBeInTheDocument();
  });

  it("renders the research-only guidance on the help page", async () => {
    stubApi();

    renderAt("/help");

    expect(
      await screen.findByRole("heading", { name: "Help & Research Guide" }),
    ).toBeInTheDocument();
    expect(screen.getByText("Research only — not for clinical use")).toBeInTheDocument();
    expect(screen.getAllByText("Patient Replay").length).toBeGreaterThan(0);
    expect(screen.getByText(/SHAP value describes/i)).toBeInTheDocument();
  });

  it("renders an error panel when the API is unreachable", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))),
    );

    renderAt("/replay");

    expect(
      (await screen.findAllByText(/Could not reach the P3 API/i)).length,
    ).toBeGreaterThan(0);
  });
});
