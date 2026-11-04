import { Activity, ArrowUpRight, Info } from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useNavigate } from "react-router-dom";

import type { AlertRecord } from "../../api/types";
import type { RiskPoint } from "../../hooks/useRiskSeries";
import { ErrorState } from "../common/StateViews";

const GREEN = "#477c50";
const GREEN_SOFT = "#6e9874";
const WARNING = "#e5a844";
const CRITICAL = "#dc655b";
const MUTED = "#93a096";

interface ChartTooltipProps {
  active?: boolean;
  label?: string | number;
  payload?: Array<{ payload: RiskPoint }>;
}

function ChartTooltip({ active, label, payload }: ChartTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const point = payload[0]?.payload;
  if (!point) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__label">Recorded hour {label}</div>
      <div className="chart-tooltip__row">
        <span>Risk probability</span>
        <strong>
          {point.probability === null ? "Unavailable" : point.probability.toFixed(3)}
        </strong>
      </div>
      <div className="chart-tooltip__row">
        <span>Prediction status</span>
        <span>{point.status.replaceAll("_", " ")}</span>
      </div>
    </div>
  );
}

export function RiskOverviewChart({
  points,
  threshold,
  alerts,
  patientId,
  patientLabel,
  loading,
  error,
  onRetry,
  title = "Patient Risk Overview",
  subtitle = "Hourly model score from the replay engine. Synthetic demonstration cohort.",
}: {
  points: RiskPoint[];
  threshold: number | null;
  alerts: AlertRecord[];
  patientId: string | null;
  patientLabel?: string;
  loading: boolean;
  error?: Error | null;
  onRetry?: () => void;
  title?: string;
  subtitle?: string;
}) {
  const navigate = useNavigate();
  const openReplay = () => {
    if (!patientId) return;
    const firstHour = points[0]?.hour;
    navigate(
      `/replay?patient=${encodeURIComponent(patientId)}${
        firstHour !== undefined ? `&hour=${firstHour}` : ""
      }`,
    );
  };

  const alertHours = new Set(alerts.map((alert) => alert.last_hour));
  const data = points.map((point) => ({
    ...point,
    alert: alertHours.has(point.hour),
  }));

  return (
    <section className="card chart-card" aria-labelledby="risk-overview-title">
      <div className="chart-card__header">
        <div>
          <div className="chart-card__title">
            <h2 id="risk-overview-title">{title}</h2>
            <span className="badge badge--synthetic">Demo</span>
          </div>
          <p className="chart-card__subtitle">{subtitle}</p>
        </div>
        <div className="card__actions">
          {patientId ? (
            <span className="badge badge--soft-green">
              <Activity size={12} aria-hidden="true" />
              {patientLabel ?? patientId}
            </span>
          ) : null}
          {patientId ? (
            <button type="button" className="btn btn--quiet" onClick={openReplay}>
              Open replay
              <ArrowUpRight size={14} aria-hidden="true" />
            </button>
          ) : null}
        </div>
      </div>

      <div className="chart-legend" aria-label="Chart legend">
        <span>
          <span className="legend-swatch" aria-hidden="true" />
          Risk Probability
        </span>
        <span>
          <span className="legend-swatch legend-swatch--dashed" aria-hidden="true" />
          Alert Threshold {threshold !== null ? `(${threshold.toFixed(2)})` : ""}
        </span>
        <span>
          <span className="legend-dot" aria-hidden="true" />
          Alert Events
        </span>
      </div>

      {loading ? (
        <div className="chart-empty" role="status" aria-live="polite">
          <div>
            <span className="spinner" aria-hidden="true" />
            <p style={{ marginTop: 10 }}>Loading replay predictions…</p>
          </div>
        </div>
      ) : error ? (
        <ErrorState error={error} onRetry={onRetry} />
      ) : data.length === 0 ? (
        <div className="chart-empty">
          No recorded hours are available for a risk preview.
        </div>
      ) : (
        <>
          <div
            className="chart-wrap"
            role={patientId ? "button" : undefined}
            tabIndex={patientId ? 0 : undefined}
            aria-label={
              patientId
                ? `Open patient replay for ${patientLabel ?? patientId}`
                : undefined
            }
            onClick={openReplay}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                openReplay();
              }
            }}
            style={{ cursor: patientId ? "pointer" : "default" }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart
                data={data}
                margin={{ top: 12, right: 18, left: -10, bottom: 0 }}
              >
                <defs>
                  <linearGradient id="risk-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor={GREEN_SOFT} stopOpacity={0.34} />
                    <stop offset="100%" stopColor={GREEN_SOFT} stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid
                  strokeDasharray="4 5"
                  vertical={false}
                  stroke="#e8eee4"
                />
                <XAxis
                  dataKey="hour"
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value: number) => `${value}h`}
                  minTickGap={18}
                />
                <YAxis
                  domain={[0, 1]}
                  ticks={[0, 0.25, 0.5, 0.75, 1]}
                  tick={{ fontSize: 11, fill: MUTED }}
                  tickLine={false}
                  axisLine={false}
                  tickFormatter={(value: number) => value.toFixed(2)}
                  width={44}
                />
                <Tooltip
                  cursor={{ stroke: "#a9c9a3", strokeDasharray: "3 3" }}
                  content={(props) => (
                    <ChartTooltip
                      {...(props as unknown as ChartTooltipProps)}
                    />
                  )}
                />
                {threshold !== null ? (
                  <ReferenceLine
                    y={threshold}
                    stroke={WARNING}
                    strokeDasharray="6 5"
                    strokeWidth={1.6}
                  />
                ) : null}
                <Area
                  type="monotone"
                  dataKey="probability"
                  stroke={GREEN}
                  strokeWidth={2.6}
                  fill="url(#risk-fill)"
                  connectNulls={false}
                  dot={false}
                  activeDot={{ r: 5, stroke: "#ffffff", strokeWidth: 2 }}
                  isAnimationActive={false}
                />
                {alerts.map((alert) => {
                  const point = data.find(
                    (item) =>
                      item.hour === alert.last_hour && item.probability !== null,
                  );
                  if (!point || point.probability === null) return null;
                  return (
                    <ReferenceDot
                      key={alert.alert_id}
                      x={point.hour}
                      y={point.probability}
                      r={5.5}
                      fill={alert.status === "ACTIVE" ? CRITICAL : WARNING}
                      stroke="#ffffff"
                      strokeWidth={2}
                    />
                  );
                })}
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <p className="chart-footnote">
            <Info size={13} aria-hidden="true" />
            Scores are actual predictions from the configured demo model for{" "}
            {patientLabel ?? patientId}. They are not validated clinical risk and
            do not confirm sepsis.
          </p>
        </>
      )}
    </section>
  );
}
