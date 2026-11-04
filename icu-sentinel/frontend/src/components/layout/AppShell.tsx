import { Menu, ShieldCheck } from "lucide-react";
import type { ReactNode } from "react";
import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";

import { MobileNavigation } from "./MobileNavigation";
import { Sidebar } from "./Sidebar";

export function AppShell({ children }: { children: ReactNode }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const location = useLocation();

  useEffect(() => {
    setMobileOpen(false);
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [location.pathname]);

  return (
    <div className="app-shell">
      <div className="app-shell__sidebar">
        <Sidebar />
      </div>

      <div className="app-shell__main">
        <button
          type="button"
          className="mobile-navigation-trigger"
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation menu"
        >
          <Menu size={18} aria-hidden="true" />
          <span>Menu</span>
        </button>

        <main className="app-shell__content" id="main-content">
          {children}
        </main>

        <footer className="app-footer">
          <p>
            <ShieldCheck
              size={12}
              aria-hidden="true"
              style={{ display: "inline", marginRight: 5, verticalAlign: "-2px" }}
            />
            ICU Sentinel research prototype · Predictions, explanations and alerts
            are not clinical advice.
          </p>
          <nav aria-label="Footer">
            <Link to="/help">Research use</Link>
            <Link to="/help">Data provenance</Link>
            <Link to="/settings">Configuration</Link>
          </nav>
        </footer>
      </div>

      <MobileNavigation open={mobileOpen} onClose={() => setMobileOpen(false)} />
    </div>
  );
}
