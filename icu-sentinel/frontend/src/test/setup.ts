import "@testing-library/jest-dom/vitest";
import { cleanup } from "@testing-library/react";
import { afterEach } from "vitest";

afterEach(() => {
  cleanup();
});

// Recharts asks for ResizeObserver; jsdom does not implement it.
class ResizeObserverStub {
  observe() {}
  unobserve() {}
  disconnect() {}
}

globalThis.ResizeObserver =
  globalThis.ResizeObserver ?? (ResizeObserverStub as unknown as typeof ResizeObserver);

// AppShell resets scroll on route changes; jsdom does not implement it.
Object.defineProperty(window, "scrollTo", {
  value: () => {},
  writable: true,
});
