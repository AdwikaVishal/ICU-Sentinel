import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { FeatureContribution } from "../api/types";

const POSITIVE = "#b45309";
const NEGATIVE = "#1d4ed8";

function outputSpaceLabel(space: string | null): string {
  if (space === "probability") return "probability space";
  if (space === "log_odds" || space === "decision") return "log-odds space";
  return space ?? "model output space";
}

export function ContributionChart({
  contributions,
  outputSpace,
}: {
  contributions: FeatureContribution[];
  outputSpace: string | null;
}) {
  const data = contributions.slice(0, 12).map((item) => ({
    name: item.display_name,
    feature: item.feature_name,
    value: item.shap_value,
    magnitude: item.magnitude,
    direction: item.direction,
  }));

  if (data.length === 0) {
    return <p className="muted">No feature contributions were returned.</p>;
  }

  return (
    <div className="chart">
      <p className="muted small">
        SHAP contributions in {outputSpaceLabel(outputSpace)}. Positive values push
        the model output up; negative values push it down. This describes the model,
        not a clinical cause.
      </p>
      <div style={{ width: "100%", height: Math.max(240, data.length * 32) }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(value: number) => value.toFixed(2)} />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              formatter={(value: unknown) => [
                typeof value === "number" ? value.toFixed(4) : String(value ?? "—"),
                "SHAP value",
              ] as [string, string]}
            />
            <Bar dataKey="value" name="SHAP value" isAnimationActive={false}>
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
        <summary>Contribution values</summary>
        <div className="table-wrap">
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
                      : `${item.value}${item.unit ? ` ${item.unit}` : ""}`}
                  </td>
                  <td className="numeric">{item.shap_value.toFixed(4)}</td>
                  <td>{item.direction.replaceAll("_", " ")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </details>
    </div>
  );
}
