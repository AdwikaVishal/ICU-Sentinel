import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { GlobalFeatureImportance } from "../api/types";

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
    return <p className="muted">No global importance values were returned.</p>;
  }

  return (
    <div className="chart">
      <p className="muted small">
        Mean absolute SHAP value across the synthetic background sample. Higher
        values mean the fitted model relies on that feature more, not that the
        feature causes sepsis.
      </p>
      <div style={{ width: "100%", height: Math.max(240, data.length * 30) }}>
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ left: 8, right: 24 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} />
            <XAxis type="number" tickFormatter={(value: number) => value.toFixed(3)} />
            <YAxis
              type="category"
              dataKey="name"
              width={150}
              tick={{ fontSize: 12 }}
            />
            <Tooltip
              formatter={(value: unknown) => [
                typeof value === "number" ? value.toFixed(4) : String(value ?? "—"),
                "Mean |SHAP|",
              ] as [string, string]}
            />
            <Bar
              dataKey="value"
              name="Mean |SHAP|"
              fill="#475569"
              isAnimationActive={false}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
