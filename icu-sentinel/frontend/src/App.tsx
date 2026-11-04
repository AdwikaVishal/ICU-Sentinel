import { lazy, Suspense } from "react";
import { Link, Route, Routes } from "react-router-dom";

import { PageHeader } from "./components/common/PageHeader";
import { AppShell } from "./components/layout/AppShell";

const HomePage = lazy(() =>
  import("./pages/HomePage").then((module) => ({ default: module.HomePage })),
);
const ReplayPage = lazy(() =>
  import("./pages/ReplayPage").then((module) => ({ default: module.ReplayPage })),
);
const ExplanationsPage = lazy(() =>
  import("./pages/ExplanationsPage").then((module) => ({
    default: module.ExplanationsPage,
  })),
);
const AlertsPage = lazy(() =>
  import("./pages/AlertsPage").then((module) => ({ default: module.AlertsPage })),
);
const ReportsPage = lazy(() =>
  import("./pages/ReportsPage").then((module) => ({ default: module.ReportsPage })),
);
const SettingsPage = lazy(() =>
  import("./pages/SettingsPage").then((module) => ({ default: module.SettingsPage })),
);
const HelpPage = lazy(() =>
  import("./pages/HelpPage").then((module) => ({ default: module.HelpPage })),
);
const PatientReplayPage = lazy(() =>
  import("./pages/PatientReplayPage").then((module) => ({
    default: module.PatientReplayPage,
  })),
);
const ExplanationCentrePage = lazy(() =>
  import("./pages/ExplanationCentrePage").then((module) => ({
    default: module.ExplanationCentrePage,
  })),
);
const AlertCentrePage = lazy(() =>
  import("./pages/AlertCentrePage").then((module) => ({
    default: module.AlertCentrePage,
  })),
);

function NotFoundPage() {
  return (
    <div className="page">
      <PageHeader
        title="Page not found"
        description="The page you requested does not exist in this research workspace."
      />
      <div className="card">
        <p className="muted">
          Return to the <Link to="/">dashboard</Link> or open{" "}
          <Link to="/replay">Patient Replay</Link>,{" "}
          <Link to="/explanations">Explanations</Link> or{" "}
          <Link to="/alerts">Alerts</Link>.
        </p>
      </div>
    </div>
  );
}

export function App() {
  return (
    <AppShell>
      <Suspense
        fallback={
          <div className="route-loading" role="status">
            Loading workspace…
          </div>
        }
      >
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/replay" element={<ReplayPage />} />
          <Route path="/explanations" element={<ExplanationsPage />} />
          <Route path="/alerts" element={<AlertsPage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="/help" element={<HelpPage />} />

          {/* Legacy views remain available for comparison and regression review. */}
          <Route path="/legacy/replay" element={<PatientReplayPage />} />
          <Route path="/legacy/explanations" element={<ExplanationCentrePage />} />
          <Route path="/legacy/alerts" element={<AlertCentrePage />} />

          <Route path="*" element={<NotFoundPage />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
