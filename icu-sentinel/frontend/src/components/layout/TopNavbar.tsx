import {
  Bell,
  CalendarDays,
  ChevronDown,
  HelpCircle,
  HeartPulse,
  Menu,
  Settings,
  ShieldCheck,
  User,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { NavLink, useNavigate } from "react-router-dom";

import { replayApi } from "../../api/client";
import type { ProviderStatus } from "../../api/types";
import { useAsync, type AsyncResource } from "../../hooks/useAsync";

const LINKS = [
  { to: "/", label: "Home", end: true },
  { to: "/replay", label: "Patient Replay" },
  { to: "/explanations", label: "Explanations" },
  { to: "/alerts", label: "Alerts" },
  { to: "/reports", label: "Reports" },
];

function useDismiss(
  ref: React.RefObject<HTMLElement | null>,
  onDismiss: () => void,
  active: boolean,
) {
  useEffect(() => {
    if (!active) return;
    const onPointer = (event: PointerEvent) => {
      if (ref.current && !ref.current.contains(event.target as Node)) onDismiss();
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onDismiss();
    };
    document.addEventListener("pointerdown", onPointer);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("pointerdown", onPointer);
      document.removeEventListener("keydown", onKey);
    };
  }, [active, onDismiss, ref]);
}

export function TopNavbar({
  onOpenMenu,
  activeAlertCount,
}: {
  onOpenMenu: () => void;
  activeAlertCount: number | null;
}) {
  const navigate = useNavigate();
  const [notificationsOpen, setNotificationsOpen] = useState(false);
  const [profileOpen, setProfileOpen] = useState(false);
  const notificationRef = useRef<HTMLDivElement>(null);
  const profileRef = useRef<HTMLDivElement>(null);

  const provider: AsyncResource<ProviderStatus> = useAsync(
    (signal) => replayApi.getProviderStatus(signal),
    [],
  );

  useDismiss(
    notificationRef,
    () => setNotificationsOpen(false),
    notificationsOpen,
  );
  useDismiss(profileRef, () => setProfileOpen(false), profileOpen);

  const dataSourceLabel = provider.data
    ? provider.data.model?.source === "p2_boosting_model"
      ? "Real PhysioNet data · P2 boosting models"
      : "Synthetic demonstration cohort"
    : "—";

  return (
    <header className="topbar">
      <button
        type="button"
        className="icon-button topbar__menu"
        onClick={onOpenMenu}
        aria-label="Open navigation menu"
      >
        <Menu size={19} />
      </button>

      <NavLink to="/" className="topbar__brand" aria-label="ICU Sentinel home">
        <span className="topbar__brand-mark" aria-hidden="true">
          <HeartPulse size={19} />
        </span>
        <span>ICU Sentinel</span>
      </NavLink>

      <div className="topbar__context">
        <span className={`data-source-badge ${provider.data?.model?.source === "p2_boosting_model" ? "badge--model" : "badge--synthetic"}`}>
          {dataSourceLabel}
        </span>
        <CalendarDays size={15} aria-hidden="true" />
        <span>ICU clinical workspace</span>
      </div>

      <nav className="topbar__nav" aria-label="Primary navigation">
        {LINKS.map(({ to, label, end }) => (
          <NavLink
            key={to}
            to={to}
            end={end}
            className={({ isActive }) =>
              isActive ? "topbar__link is-active" : "topbar__link"
            }
          >
            {label}
          </NavLink>
        ))}
      </nav>

      <div className="topbar__actions">
        <div className="popover-anchor" ref={notificationRef}>
          <button
            type="button"
            className="icon-button"
            onClick={() => {
              setNotificationsOpen((value) => !value);
              setProfileOpen(false);
            }}
            aria-label="Notifications"
            aria-expanded={notificationsOpen}
            aria-haspopup="true"
          >
            <Bell size={18} />
            {activeAlertCount !== null && activeAlertCount > 0 ? (
              <span className="icon-button__count">
                {activeAlertCount > 9 ? "9+" : activeAlertCount}
              </span>
            ) : null}
          </button>
          {notificationsOpen ? (
            <div className="popover" role="dialog" aria-label="Notifications">
              <div className="popover__header">
                <strong>Notifications</strong>
                <span className="badge badge--muted">Research</span>
              </div>
              {activeAlertCount !== null && activeAlertCount > 0 ? (
                <button
                  type="button"
                  className="popover__item"
                  onClick={() => {
                    setNotificationsOpen(false);
                    navigate("/alerts");
                  }}
                >
                  <Bell size={16} aria-hidden="true" />
                  <span>
                    <strong>{activeAlertCount}</strong> active alert episode
                    {activeAlertCount === 1 ? "" : "s"} awaiting review
                  </span>
                </button>
              ) : (
                <div className="popover__empty">
                  No active alert episodes. An empty queue is not evidence that a
                  patient is stable.
                </div>
              )}
              <button
                type="button"
                className="popover__item"
                onClick={() => {
                  setNotificationsOpen(false);
                  navigate("/alerts");
                }}
              >
                <ShieldCheck size={16} aria-hidden="true" />
                <span>Open the alert centre</span>
              </button>
            </div>
          ) : null}
        </div>

        <div className="popover-anchor" ref={profileRef}>
          <button
            type="button"
            className="avatar-button"
            onClick={() => {
              setProfileOpen((value) => !value);
              setNotificationsOpen(false);
            }}
            aria-label="Open profile menu"
            aria-expanded={profileOpen}
            aria-haspopup="true"
          >
            <span className="avatar" aria-hidden="true">
              RS
            </span>
            <span className="avatar-button__copy">
              <strong>Research Staff</strong>
              <span>Clinical review</span>
            </span>
            <ChevronDown size={14} aria-hidden="true" />
          </button>
          {profileOpen ? (
            <div className="popover" role="dialog" aria-label="Profile menu">
              <div className="popover__header">
                <div>
                  <strong>Research Staff</strong>
                  <p className="tiny muted">Demo profile · local session</p>
                </div>
                <span className="avatar" aria-hidden="true">
                  RS
                </span>
              </div>
              <button
                type="button"
                className="popover__item"
                onClick={() => {
                  setProfileOpen(false);
                  navigate("/settings");
                }}
              >
                <Settings size={16} aria-hidden="true" />
                <span>Settings</span>
              </button>
              <button
                type="button"
                className="popover__item"
                onClick={() => {
                  setProfileOpen(false);
                  navigate("/help");
                }}
              >
                <HelpCircle size={16} aria-hidden="true" />
                <span>Help and documentation</span>
              </button>
              <button
                type="button"
                className="popover__item"
                onClick={() => {
                  setProfileOpen(false);
                  navigate("/help");
                }}
              >
                <User size={16} aria-hidden="true" />
                <span>About ICU Sentinel</span>
              </button>
            </div>
          ) : null}
        </div>
      </div>
    </header>
  );
}
