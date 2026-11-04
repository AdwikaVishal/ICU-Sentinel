import type { ObservationValue } from "../api/types";
import { EmptyPanel } from "./AsyncState";

export function ObservationTable({ observations }: { observations: ObservationValue[] }) {
  if (observations.length === 0) {
    return (
      <EmptyPanel title="No observation row for this hour">
        The replay is still valid; this hour has no recorded measurements. Missing
        measurements are never treated as normal values.
      </EmptyPanel>
    );
  }

  return (
    <div className="table-wrap">
      <table>
        <caption className="sr-only">
          Observations at the selected hour. Unavailable values are shown as not
          recorded.
        </caption>
        <thead>
          <tr>
            <th scope="col">Observation</th>
            <th scope="col">Category</th>
            <th scope="col">Value</th>
            <th scope="col">Status</th>
          </tr>
        </thead>
        <tbody>
          {observations.map((observation) => (
            <tr key={observation.parameter}>
              <th scope="row">{observation.display_name}</th>
              <td className="muted">{observation.category}</td>
              <td className="numeric">
                {observation.available && observation.value !== null
                  ? `${observation.value}${observation.unit ? ` ${observation.unit}` : ""}`
                  : "—"}
              </td>
              <td>
                <span
                  className={`badge ${
                    observation.available ? "badge--ok" : "badge--muted"
                  }`}
                >
                  {observation.available ? "Recorded" : "Not recorded"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
