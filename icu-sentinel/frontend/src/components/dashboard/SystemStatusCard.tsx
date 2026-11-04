import { Activity, BellRing, Brain, Database } from "lucide-react";

export type ServiceState = "online" | "offline" | "checking";

export interface SystemStatus {
  replay: ServiceState;
  explainability: ServiceState;
  alerts: ServiceState;
  dataSource: string | null;
  isSynthetic: boolean | null;
}

function stateLabel(state: ServiceState): string {
  if (state === "online") return "Online";
  if (state === "offline") return "Unavailable";
  return "Checking…";
}

function StateIndicator({ state }: { state: ServiceState }) {
  const dot =
    state === "online"
      ? "status-dot--online"
      : state === "offline"
        ? "status-dot--offline"
        : "status-dot--idle";
  const tone =
    state === "online"
      ? ""
      : state === "offline"
        ? " system-status__state--offline"
        : " system-status__state--warn";
  return (
    <span className={`system-status__state${tone}`}>
      <span className={`status-dot ${dot}`} aria-hidden="true" />
      {stateLabel(state)}
    </span>
  );
}

export function SystemStatusCard({ status }: { status: SystemStatus }) {
  const services = [
    { name: "Replay Engine", icon: Activity, state: status.replay },
    { name: "Explainability (SHAP)", icon: Brain, state: status.explainability },
    { name: "Alert Engine", icon: BellRing, state: status.alerts },
  ];

  const allOnline = services.every((service) => service.state === "online");
  const anyOffline = services.some((service) => service.state === "offline");
  const overall = allOnline
    ? "All systems operational"
    : anyOffline
      ? "Partial availability"
      : "Checking services";

  const dataSourceLabel =
    status.dataSource === null
      ? "Unavailable"
      : status.dataSource === "synthetic_demo"
        ? "Synthetic Demo"
        : status.dataSource;

  return (
    <section className="system-status" aria-labelledby="system-status-title">
      <div className="system-status__header">
        <div>
          <h2 id="system-status-title">System Status</h2>
          <p>Live service availability</p>
        </div>
        <span className="system-status__overall">
          <span
            className={`status-dot ${
              allOnline
                ? "status-dot--online"
                : anyOffline
                  ? "status-dot--offline"
                  : "status-dot--warning"
            }`}
            aria-hidden="true"
          />
          {overall}
        </span>
      </div>

      <div className="system-status__rows">
        {services.map(({ name, icon: Icon, state }) => (
          <div className="system-status__row" key={name}>
            <span className="system-status__name">
              <Icon size={15} aria-hidden="true" />
              {name}
            </span>
            <StateIndicator state={state} />
          </div>
        ))}
        <div className="system-status__row">
          <span className="system-status__name">
            <Database size={15} aria-hidden="true" />
            Data Source
          </span>
          <span className="badge badge--synthetic">{dataSourceLabel}</span>
        </div>
      </div>

      <p className="system-status__footer">
        {status.isSynthetic === false
          ? "A configured processed dataset is in use. Follow the deployment data-governance policy."
          : "Synthetic demonstration cohort. Nothing on this dashboard is real patient data."}
      </p>
    </section>
  );
}
