import { act, renderHook, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { useAsync } from "./useAsync";

describe("useAsync", () => {
  it("exposes loading, data and an abortable loader", async () => {
    const loader = vi.fn((_signal: AbortSignal) => Promise.resolve(["a", "b"]));
    const { result } = renderHook(() => useAsync(loader, []));

    expect(result.current.loading).toBe(true);
    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.data).toEqual(["a", "b"]);
    expect(result.current.error).toBeNull();
    expect(loader).toHaveBeenCalledTimes(1);
    expect(loader.mock.calls[0]?.[0]).toBeInstanceOf(AbortSignal);
  });

  it("surfaces loader errors", async () => {
    const loader = vi.fn(() => Promise.reject(new Error("backend offline")));
    const { result } = renderHook(() => useAsync(loader, []));

    await waitFor(() => expect(result.current.loading).toBe(false));
    expect(result.current.error?.message).toBe("backend offline");
    expect(result.current.data).toBeNull();
  });

  it("does not load while disabled", async () => {
    const loader = vi.fn(() => Promise.resolve(1));
    const { result } = renderHook(() => useAsync(loader, [], false));

    expect(result.current.loading).toBe(false);
    expect(loader).not.toHaveBeenCalled();

    act(() => result.current.reload());
    await new Promise((resolve) => setTimeout(resolve, 0));
    expect(loader).not.toHaveBeenCalled();
    expect(result.current.data).toBeNull();
  });

  it("reload refetches the resource", async () => {
    const loader = vi
      .fn<() => Promise<number>>()
      .mockResolvedValueOnce(1)
      .mockResolvedValueOnce(2);
    const { result } = renderHook(() => useAsync(loader, []));

    await waitFor(() => expect(result.current.data).toBe(1));
    act(() => result.current.reload());
    await waitFor(() => expect(result.current.data).toBe(2));
    expect(loader).toHaveBeenCalledTimes(2);
  });
});
