import { Search, UserRound } from "lucide-react";
import { useDeferredValue, useMemo, useState } from "react";

import type { PatientSummary } from "../api/types";

export function PatientSelector({
  patients,
  value,
  onChange,
  disabled,
}: {
  patients: PatientSummary[];
  value: string | null;
  onChange: (patientId: string) => void;
  disabled?: boolean;
}) {
  const [query, setQuery] = useState("");
  const deferredQuery = useDeferredValue(query);

  const filtered = useMemo(() => {
    const term = deferredQuery.trim().toLowerCase();
    if (!term) return patients;
    return patients.filter((patient) =>
      patient.patient_id.toLowerCase().includes(term),
    );
  }, [deferredQuery, patients]);

  return (
    <div>
      <div className="selector-card__title">
        <UserRound size={17} aria-hidden="true" />
        <span>Select a patient</span>
      </div>
      <p className="selector-card__hint">
        Search the configured cohort. Synthetic demo patients are labelled
        explicitly.
      </p>

      <div className="patient-search">
        <label className="field" style={{ minWidth: 0 }}>
          <span className="field__label">Patient search</span>
          <input
            type="search"
            className="input input--search"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search by patient ID"
            disabled={disabled || patients.length === 0}
            aria-label="Search patients by ID"
          />
        </label>
      </div>

      <div
        className="patient-list"
        role="listbox"
        aria-label="Patients"
        style={{ marginTop: 12 }}
      >
        {filtered.length === 0 ? (
          <div className="patient-list__empty">
            <Search size={16} aria-hidden="true" style={{ margin: "0 auto 6px" }} />
            No patients match “{query}”.
          </div>
        ) : (
          filtered.map((patient) => {
            const selected = patient.patient_id === value;
            return (
              <button
                key={patient.patient_id}
                type="button"
                role="option"
                aria-selected={selected}
                className={`patient-option${selected ? " is-selected" : ""}`}
                onClick={() => onChange(patient.patient_id)}
                disabled={disabled}
              >
                <span>
                  <span className="patient-option__id">
                    {patient.patient_id}
                    {patient.is_synthetic ? (
                      <span className="badge badge--synthetic">Synthetic</span>
                    ) : null}
                  </span>
                  <span className="patient-option__meta">
                    {patient.recorded_hours} recorded hours
                    {patient.missing_hours > 0
                      ? ` · ${patient.missing_hours} missing`
                      : ""}
                    {patient.has_retrospective_label
                      ? " · retrospective label available"
                      : ""}
                  </span>
                </span>
                <span className="patient-option__hours">
                  {patient.n_hours}h
                </span>
              </button>
            );
          })
        )}
      </div>
    </div>
  );
}
