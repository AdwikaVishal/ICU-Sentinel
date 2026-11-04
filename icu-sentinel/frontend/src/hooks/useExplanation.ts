import { useCallback, useEffect, useState } from "react";

import { alertsApi, explanationsApi } from "../api/client";
import type {
  AlertListResponse,
  AlertRecord,
  AlertThresholdInfo,
  GlobalExplanation,
  HourlyExplanationComparison,
  ModelCatalogResponse,
  PatientExplanation,
} from "../api/types";
import { useAsync, type AsyncResource } from "./useAsync";

export function useModelCatalog(): AsyncResource<ModelCatalogResponse> {
  return useAsync((signal) => explanationsApi.listModels(signal), []);
}

export function useGlobalExplanation(
  modelName?: string,
): AsyncResource<GlobalExplanation> {
  return useAsync(
    (signal) => explanationsApi.global(modelName, undefined, signal),
    [modelName],
  );
}

export function usePatientExplanation(
  patientId: string | null,
  hour: number | null,
): AsyncResource<PatientExplanation> {
  return useAsync(
    (signal) => explanationsApi.explain(patientId as string, hour as number, signal),
    [patientId, hour],
    Boolean(patientId) && hour !== null,
  );
}

export function useHourComparison(
  patientId: string | null,
  hourA: number | null,
  hourB: number | null,
  enabled: boolean,
): AsyncResource<HourlyExplanationComparison> {
  return useAsync(
    (signal) =>
      explanationsApi.compare(
        patientId as string,
        hourA as number,
        hourB as number,
        signal,
      ),
    [patientId, hourA, hourB],
    enabled && Boolean(patientId) && hourA !== null && hourB !== null,
  );
}

export interface AlertCentreController {
  active: AlertListResponse | null;
  history: AlertListResponse | null;
  all: AlertListResponse | null;
  threshold: AlertThresholdInfo | null;
  loading: boolean;
  error: Error | null;
  actionError: string | null;
  actionId: string | null;
  reload: () => void;
  acknowledge: (alertId: string) => Promise<boolean>;
  clear: (alertId: string, reason: string) => Promise<boolean>;
}

export function useAlertCentre(): AlertCentreController {
  const [active, setActive] = useState<AlertListResponse | null>(null);
  const [history, setHistory] = useState<AlertListResponse | null>(null);
  const [all, setAll] = useState<AlertListResponse | null>(null);
  const [threshold, setThreshold] = useState<AlertThresholdInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [actionId, setActionId] = useState<string | null>(null);
  const [version, setVersion] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    Promise.all([
      alertsApi.active(200, controller.signal),
      alertsApi.history(200, controller.signal),
      alertsApi.threshold(controller.signal),
      alertsApi.list({ limit: 500 }, controller.signal),
    ])
      .then(([activeResponse, historyResponse, thresholdResponse, allResponse]) => {
        if (controller.signal.aborted) return;
        setActive(activeResponse);
        setHistory(historyResponse);
        setThreshold(thresholdResponse);
        setAll(allResponse);
        setLoading(false);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause : new Error(String(cause)));
        setLoading(false);
      });
    return () => controller.abort();
  }, [version]);

  const reload = useCallback(() => setVersion((value) => value + 1), []);

  const runAction = useCallback(
    async (
      alertId: string,
      action: () => Promise<AlertRecord>,
    ): Promise<boolean> => {
      setActionId(alertId);
      setActionError(null);
      try {
        await action();
        setVersion((value) => value + 1);
        return true;
      } catch (cause) {
        setActionError(cause instanceof Error ? cause.message : String(cause));
        return false;
      } finally {
        setActionId(null);
      }
    },
    [],
  );

  const acknowledge = useCallback(
    (alertId: string) =>
      runAction(alertId, () => alertsApi.acknowledge(alertId)),
    [runAction],
  );

  const clear = useCallback(
    (alertId: string, reason: string) =>
      runAction(alertId, () => alertsApi.clear(alertId, { reason })),
    [runAction],
  );

  return {
    active,
    history,
    all,
    threshold,
    loading,
    error,
    actionError,
    actionId,
    reload,
    acknowledge,
    clear,
  };
}
