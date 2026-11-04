import {
  Activity,
  BarChart3,
  BellRing,
  Brain,
  HeartPulse,
  HelpCircle,
  Home,
  Settings,
} from "lucide-react";
import { NavLink } from "react-router-dom";

const NAV_ITEMS = [
  { to: "/", label: "Home", icon: Home, end: true },
  { to: "/replay", label: "Patient Replay", icon: Activity },
  { to: "/explanations", label: "Explanations", icon: Brain },
  { to: "/alerts", label: "Alerts", icon: BellRing },
  { to: "/reports", label: "Reports", icon: BarChart3 },
];

const FOOTER_ITEMS = [
  { to: "/settings", label: "Settings", icon: Settings },
  { to: "/help", label: "Help", icon: HelpCircle },
];

export function SidebarContent({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <div className="sidebar__inner">
      <NavLink to="/" className="sidebar__brand" onClick={onNavigate}>
        <span className="sidebar__brand-mark" aria-hidden="true">
          <HeartPulse size={24} strokeWidth={2.2} />
        </span>
        <span className="sidebar__brand-copy">
          <strong>ICU Sentinel</strong>
          <span>ICU Sepsis Early Warning</span>
        </span>
      </NavLink>

      <nav className="sidebar__nav" aria-label="Sidebar navigation">
        <span className="sidebar__nav-label">Workspace</span>
        {NAV_ITEMS.map(({ to, label, icon: Icon, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) =>
              isActive ? "sidebar__link is-active" : "sidebar__link"
            }
          >
            <Icon size={17} strokeWidth={1.9} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}

        <span className="sidebar__nav-label" style={{ marginTop: 14 }}>
          Support
        </span>
        {FOOTER_ITEMS.map(({ to, label, icon: Icon }) => (
          <NavLink
            key={to}
            to={to}
            onClick={onNavigate}
            className={({ isActive }) =>
              isActive ? "sidebar__link is-active" : "sidebar__link"
            }
          >
            <Icon size={17} strokeWidth={1.9} aria-hidden="true" />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      <div className="sidebar__footer">
        <div className="sidebar__mode">
          <span className="sidebar__mode-dot" aria-hidden="true" />
          <span>Research mode</span>
        </div>
        <p className="sidebar__fineprint">
          Retrospective decision support research. Not a clinical alarm or
          treatment recommendation.
        </p>
      </div>
    </div>
  );
}

export function Sidebar() {
  return (
    <aside className="sidebar" aria-label="ICU Sentinel">
      <SidebarContent />
    </aside>
  );
}
