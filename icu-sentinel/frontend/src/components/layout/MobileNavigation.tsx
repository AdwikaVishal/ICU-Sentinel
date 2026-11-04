import { X } from "lucide-react";

import { SidebarContent } from "./Sidebar";

export function MobileNavigation({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  if (!open) return null;

  return (
    <div className="mobile-drawer is-open">
      <button
        type="button"
        className="mobile-drawer__scrim"
        onClick={onClose}
        aria-label="Close navigation menu"
      />
      <div
        className="mobile-drawer__panel"
        role="dialog"
        aria-modal="true"
        aria-label="Navigation menu"
      >
        <button
          type="button"
          className="mobile-drawer__close"
          onClick={onClose}
          aria-label="Close navigation menu"
        >
          <X size={17} />
        </button>
        <div className="sidebar">
          <SidebarContent onNavigate={onClose} />
        </div>
      </div>
    </div>
  );
}
