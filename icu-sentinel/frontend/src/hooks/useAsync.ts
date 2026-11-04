import { useCallback, useEffect, useState } from "react";

export interface AsyncResource<T> {
  data: T | null;
  loading: boolean;
  error: Error | null;
  reload: () => void;
}

/**
 * Small read-only resource hook with an abort signal.
 *
 * The loader is expected to pass the signal to the API client so that a
 * patient/hour change cannot commit a stale response.
 */
export function useAsync<T>(
  loader: (signal: AbortSignal) => Promise<T>,
  deps: readonly unknown[],
  enabled = true,
): AsyncResource<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(enabled);
  const [error, setError] = useState<Error | null>(null);
  const [version, setVersion] = useState(0);

  const reload = useCallback(() => setVersion((value) => value + 1), []);

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    loader(controller.signal)
      .then((value) => {
        if (controller.signal.aborted) return;
        setData(value);
        setLoading(false);
      })
      .catch((cause: unknown) => {
        if (controller.signal.aborted) return;
        setError(cause instanceof Error ? cause : new Error(String(cause)));
        setLoading(false);
      });
    return () => controller.abort();
    // The caller owns the dependency list; each value is part of the request.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, version]);

  return { data, loading, error, reload };
}
