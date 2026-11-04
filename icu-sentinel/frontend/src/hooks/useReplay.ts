import { useCallback, useEffect, useRef, useState } from "react";

import { replayApi } from "../api/client";
import type { ReplayState, TimelineResponse } from "../api/types";

export interface ReplayController {
  state: ReplayState | null;
  timeline: TimelineResponse | null;
  loading: boolean;
  navigating: boolean;
  error: Error | null;
  alertError: string | null;
  hour: number | null;
  evaluateAlerts: boolean;
  setEvaluateAlerts: (value: boolean) => void;
  goToHour: (hour: number) => void;
  nextHour: () => void;
  previousHour: () => void;
  reset: () => void;
  goToEnd: () => void;
  reload: () => void;
}

/**
 * Replay navigation for one patient.
 *
 * Initial load never evaluates alerts. Navigation evaluates only when the
 * caller explicitly enables the alert rule, so opening the page cannot
 * create alert episodes as a side effect.
 */
export function useReplay(patientId: string | null): ReplayController {
  const [state, setState] = useState<ReplayState | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [navigating, setNavigating] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [alertError, setAlertError] = useState<string | null>(null);
  const [hour, setHour] = useState<number | null>(null);
  const [evaluateAlerts, setEvaluateAlerts] = useState(false);
  const [version, setVersion] = useState(0);
  const requestId = useRef(0);

  const reload = useCallback(() => setVersion((value) => value + 1), []);

  useEffect(() => {
    if (!patientId) {
      setState(null);
      setTimeline(null);
      setHour(null);
      setError(null);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    const current = ++requestId.current;
    setLoading(true);
    setError(null);
    setAlertError(null);
    setState(null);
    setHour(null);

    Promise.all([
      replayApi.getTimeline(patientId, controller.signal),
      replayApi.getState(patientId, undefined, false, controller.signal),
    ])
      .then(([timelineResponse, replayState]) => {
        if (controller.signal.aborted || current !== requestId.current) return;
        setTimeline(timelineResponse);
        setState(replayState);
        setHour(replayState.hour);
        setLoading(false);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted || current !== requestId.current) return;
        setError(cause instanceof Error ? cause : new Error(String(cause)));
        setLoading(false);
      });

    return () => controller.abort();
  }, [patientId, version]);

  const navigate = useCallback(
    (targetHour: number) => {
      if (!patientId) return;
      const current = ++requestId.current;
      setNavigating(true);
      setError(null);
      setAlertError(null);
      replayApi
        .jump(patientId, targetHour, evaluateAlerts)
        .then((replayState) => {
          if (current !== requestId.current) return;
          setState(replayState);
          setHour(replayState.hour);
          setAlertError(replayState.alert_error);
          setNavigating(false);
        })
        .catch((cause: unknown) => {
          if (current !== requestId.current) return;
          setError(cause instanceof Error ? cause : new Error(String(cause)));
          setNavigating(false);
        });
    },
    [patientId, evaluateAlerts],
  );

  const goToHour = useCallback(
    (target: number) => {
      if (!timeline) return;
      const firstHour = timeline.hours[0]?.hour ?? 0;
      const lastHour = timeline.hours[timeline.hours.length - 1]?.hour ?? firstHour;
      navigate(Math.min(lastHour, Math.max(firstHour, target)));
    },
    [navigate, timeline],
  );

  const minHour = timeline?.hours[0]?.hour ?? null;
  const maxHour = timeline?.hours[timeline.hours.length - 1]?.hour ?? null;

  const previousHour = useCallback(() => {
    if (hour === null || minHour === null) return;
    goToHour(Math.max(minHour, hour - 1));
  }, [goToHour, hour, minHour]);

  const nextHour = useCallback(() => {
    if (hour === null || maxHour === null) return;
    goToHour(Math.min(maxHour, hour + 1));
  }, [goToHour, hour, maxHour]);

  const reset = useCallback(() => {
    if (minHour !== null) goToHour(minHour);
  }, [goToHour, minHour]);

  const goToEnd = useCallback(() => {
    if (maxHour !== null) goToHour(maxHour);
  }, [goToHour, maxHour]);

  return {
    state,
    timeline,
    loading,
    navigating,
    error,
    alertError,
    hour,
    evaluateAlerts,
    setEvaluateAlerts,
    goToHour,
    nextHour,
    previousHour,
    reset,
    goToEnd,
    reload,
  };
}
