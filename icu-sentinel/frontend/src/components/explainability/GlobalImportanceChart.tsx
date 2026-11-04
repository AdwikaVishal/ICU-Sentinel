import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { GlobalFeatureImportance } from "../../api/types";

const GREEN = "#6e9874";
const MUTED = "#93a096";

export function GlobalImportanceChart({
  features,
}: {
  features: GlobalFeatureImportance[];
}) {
  const data = features.slice(0, 12).map((item) => ({
    name: item.display_name,
    feature: item.feature_name,
    value: item.mean_abs_shap,
  }));

  if (data.length === 0) {
    return (
      <p className="muted small">No global importance values were returned.</p>
    );
  }

  return (
    <div className="chart">
      <p className="muted small">
        Mean absolute SHAP across the configured background sample. Higher values
        mean the fitted model relies on a feature more — not that the feature
        causes sepsis.
      </p>
      <div style={{ width: "100%", height: Math.max(260, data.length * 31) }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
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
              tickFormatter={(value: number) => value.toFixed(3)}
            />
            <YAxis
              type="category"
              dataKey="name"
              width={170}
              tick={{ fontSize: 11, fill: "#5f6d64" }}
              tickLine={false}
              axisLine={false}
            />
            <Tooltip
              cursor={{ fill: "rgba(229, 240, 222, 0.55)" }}
              formatter={(value: unknown) => [
                typeof value === "number" ? value.toFixed(4) : String(value ?? "—"),
                "Mean |SHAP|",
              ]}
            />
            <Bar
              dataKey="value"
              name="Mean |SHAP|"
              fill={GREEN}
              radius={4}
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
