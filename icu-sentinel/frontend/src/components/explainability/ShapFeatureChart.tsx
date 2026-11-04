import { Info } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FeatureContribution } from "../../api/types";

const POSITIVE = "#477c50";
const NEGATIVE = "#dc655b";
const MUTED = "#93a096";

interface ContributionTooltipProps {
  active?: boolean;
  payload?: Array<{ payload: { name: string; value: number; feature: string } }>;
}

function ContributionTooltip({ active, payload }: ContributionTooltipProps) {
  if (!active || !payload || payload.length === 0) return null;
  const item = payload[0]?.payload;
  if (!item) return null;
  return (
    <div className="chart-tooltip">
      <div className="chart-tooltip__label">{item.name}</div>
      <div className="chart-tooltip__row">
        <span>SHAP contribution</span>
        <strong style={{ color: item.value >= 0 ? POSITIVE : NEGATIVE }}>
          {item.value.toFixed(4)}
        </strong>
      </div>
      <div className="chart-tooltip__row tiny muted">
        <span>{item.feature}</span>
      </div>
    </div>
  );
}

export function ShapFeatureChart({
  contributions,
}: {
  contributions: FeatureContribution[];
}) {
  const data = [...contributions]
    .sort((a, b) => b.magnitude - a.magnitude)
    .slice(0, 12)
    .map((item) => ({
      name: item.display_name,
      feature: item.feature_name,
      value: item.shap_value,
      magnitude: item.magnitude,
      direction: item.direction,
    }));

  if (data.length === 0) {
    return (
      <p className="muted small">
        The backend returned an available explanation without feature
        contributions.
      </p>
    );
  }

  return (
    <div className="shap-chart">
      <div className="shap-chart__legend">
        <span>
          <span className="legend-dot" style={{ background: POSITIVE }} aria-hidden="true" />
          Positive contribution — pushes output up
        </span>
        <span>
          <span className="legend-dot" style={{ background: NEGATIVE }} aria-hidden="true" />
          Negative contribution — pushes output down
        </span>
      </div>
      <div style={{ width: "100%", height: Math.max(290, data.length * 36) }}>
        <ResponsiveContainer>
          <BarChart
            data={data}
            layout="vertical"
            margin={{ top: 4, right: 26, left: 8, bottom: 4 }}
          >
            <CartesianGrid
              strokeDasharray="4 5"
              horizontal={false}
              stroke="#e8eee4"
            />
            <XAxis
              type="number"
              tick={{ fontSize: 11, fill: MUTED }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value: number) => value.toFixed(2)}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={168}
              tick={{ fontSize: 11, fill: "#5f6d64" }}
              tickLine={false}
              axisLine={false}
            />
            <ReferenceLine x={0} stroke="#b9c9b4" strokeWidth={1.4} />
            <Tooltip
              cursor={{ fill: "rgba(229, 240, 222, 0.55)" }}
              content={(props) => (
                <ContributionTooltip
                  {...(props as unknown as ContributionTooltipProps)}
                />
              )}
            />
            <Bar dataKey="value" name="SHAP value" isAnimationActive={false} radius={4}>
              {data.map((item) => (
                <Cell
                  key={item.feature}
                  fill={item.value >= 0 ? POSITIVE : NEGATIVE}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <details className="chart__table">
        <summary>View contribution values</summary>
        <div className="table-wrap" style={{ marginTop: 8 }}>
          <table>
            <thead>
              <tr>
                <th scope="col">Feature</th>
                <th scope="col">Value</th>
                <th scope="col">SHAP</th>
                <th scope="col">Direction</th>
              </tr>
            </thead>
            <tbody>
              {contributions.map((item) => (
                <tr key={item.feature_name}>
                  <th scope="row">{item.display_name}</th>
                  <td className="numeric">
                    {item.value === null
                      ? "—"
                      : `${item.value.toFixed(2)}${item.unit ? ` ${item.unit}` : ""}`}
                  </td>
                  <td className="numeric">{item.shap_value.toFixed(4)}</td>
                  <td>{item.direction.replaceAll("_", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>

      <p className="chart-footnote">
        <Info size={13} aria-hidden="true" />
        Positive and negative values describe this fitted model, not a causal
        medical effect. No SHAP value is fabricated when the backend returns an
        unavailable explanation.
      </p>
    </div>
  );
}
