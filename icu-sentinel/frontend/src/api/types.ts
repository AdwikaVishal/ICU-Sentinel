/**
 * Typed mirrors of the P3 API schemas.
 *
 * These interfaces are intentionally explicit about availability. A missing
 * prediction or explanation is represented by an explicit status/flag, not
 * by a zero value.
 */

export type PredictionSource =
  | "synthetic_heuristic"
  | "synthetic_demo_model"
  | "p1_classical_model"
  | "p2_boosting_model"
  | "p3_temporal_model"
  | "unavailable";

export type ExplanationStatus =
  | "available"
  | "unavailable"
  | "unsupported"
  | "error";

export type ExplanationMethod =
  | "linear_shap"
  | "tree_shap"
  | "kernel_shap"
  | "not_applicable";

export type AlertStatus = "ACTIVE" | "ACKNOWLEDGED" | "CLEARED";

export interface ModelIdentity {
  name: string;
  version: string;
  source: PredictionSource;
  display_name: string | null;
  is_synthetic: boolean;
  feature_schema_version: string | null;
}

export interface PredictionResult {
  patient_id: string;
  hour: number;
  probability: number | null;
  predicted_class: number | null;
  threshold: number | null;
  model: ModelIdentity;
  source: PredictionSource;
  available: boolean;
  unavailable_reason: string | null;
  prediction_id: string;
  feature_schema_version: string | null;
  model_input_hash: string | null;
  created_at: string;
  is_synthetic: boolean;
}

export interface ObservationValue {
  parameter: string;
  display_name: string;
  category: string;
  value: number | null;
  unit: string | null;
  available: boolean;
}

export interface PatientSummary {
  patient_id: string;
  n_hours: number;
  total_hours: number;
  recorded_hours: number;
  missing_hours: number;
  observation_count: number;
  data_source: string;
  source_dataset: string | null;
  is_synthetic: boolean;
  has_retrospective_label: boolean;
}

export interface PatientListResponse {
  patients: PatientSummary[];
  count: number;
  data_source: string;
  is_synthetic: boolean;
  prediction_source: string;
  provider: string;
  disclaimer: string;
}

export interface TimelineHourSummary {
  hour: number;
  available_observation_count: number;
  unavailable_observation_count: number;
  has_observations: boolean;
  retrospective_label: number | null;
  label_available: boolean;
  prediction_status: string;
}

export interface TimelineResponse {
  patient: PatientSummary;
  hours: TimelineHourSummary[];
  count: number;
  missing_hours: number;
  data_source: string;
  is_synthetic: boolean;
  disclaimer: string;
}

export interface ReplaySession {
  patient_id: string;
  current_hour: number;
  min_hour: number;
  max_hour: number;
  total_hours: number;
  at_start: boolean;
  at_end: boolean;
}

export interface CausalityMetadata {
  max_input_hour: number;
  prefix_row_count: number;
  input_observation_count: number;
  future_data_used: boolean;
  retrospective_label_used_as_input: boolean;
  model_input_hash: string | null;
  feature_schema_version: string | null;
}

export interface AlertRecord {
  alert_id: string;
  patient_id: string;
  rule_id: string;
  hour: number;
  first_hour: number;
  last_hour: number;
  probability: number;
  peak_probability: number;
  threshold: number;
  threshold_source: string;
  threshold_version: string;
  model_name: string;
  model_version: string;
  prediction_source: string;
  status: AlertStatus;
  created_at: string;
  updated_at: string;
  acknowledged_at: string | null;
  cleared_at: string | null;
  cleared_reason: string | null;
  occurrence_count: number;
  is_synthetic: boolean;
}

export interface ReplayState {
  session: ReplaySession;
  patient: PatientSummary;
  hour: number;
  timeline: TimelineHourSummary;
  observations: ObservationValue[];
  prediction: PredictionResult;
  predictions: PredictionResult[];
  alert: AlertRecord | null;
  alert_status: string;
  alert_error: string | null;
  retrospective_label: number | null;
  retrospective_label_available: boolean;
  label_disclosure: string;
  data_source: string;
  is_synthetic: boolean;
  prediction_status: string;
  explanation_available: boolean;
  causality: CausalityMetadata;
  disclaimer: string;
}

export interface ProviderStatus {
  available: boolean;
  provider_name: string;
  model: ModelIdentity | null;
  reason: string | null;
  required_features: string[] | null;
}

export interface CausalInputResponse {
  patient_id: string;
  hour: number;
  feature_schema_version: string;
  input_hash: string;
  feature_names: string[];
  feature_values: Record<string, number | null>;
  data_source: string;
  is_synthetic_data: boolean;
  prefix_row_count: number;
  observed_hours_used: number[];
  retrospective_label: number | null;
  label_disclosure: string;
}

export interface FeatureContribution {
  feature_name: string;
  display_name: string;
  category: string;
  description: string | null;
  unit: string | null;
  is_missing_indicator: boolean;
  value: number | null;
  shap_value: number;
  magnitude: number;
  direction: string;
  rank: number;
}

export interface ExplanationValidation {
  additive_error: number | null;
  tolerance: number | null;
  passed: boolean;
  checked_output_space: string;
  explanation_output: number | null;
  prediction_probability: number | null;
  note: string | null;
}

export interface ExplanationModelMetadata {
  model: ModelIdentity;
  explanation_method: ExplanationMethod;
  output_space: string;
  feature_schema_version: string;
  background_rows: number;
  generated_at: string;
}

export interface PatientExplanation {
  status: ExplanationStatus;
  patient_id: string;
  hour: number;
  prediction: PredictionResult | null;
  model: ModelIdentity | null;
  explanation_method: ExplanationMethod;
  output_space: string | null;
  base_value: number | null;
  model_output: number | null;
  prediction_probability: number | null;
  contributions: FeatureContribution[];
  positive_contributions: FeatureContribution[];
  negative_contributions: FeatureContribution[];
  validation: ExplanationValidation | null;
  input_hash: string | null;
  explanation_id: string | null;
  reason: string | null;
  metadata: ExplanationModelMetadata | null;
  generated_at: string;
  disclaimer: string;
}

export interface GlobalFeatureImportance {
  rank: number;
  feature_name: string;
  display_name: string;
  category: string;
  description: string | null;
  unit: string | null;
  is_missing_indicator: boolean;
  mean_abs_shap: number;
  mean_shap: number;
}

export interface GlobalFeatureGroup {
  category: string;
  mean_abs_shap: number;
  n_features: number;
}

export interface GlobalExplanation {
  status: ExplanationStatus;
  model: ModelIdentity | null;
  explanation_method: ExplanationMethod;
  output_space: string | null;
  features: GlobalFeatureImportance[];
  feature_groups: GlobalFeatureGroup[];
  background_rows: number;
  reason: string | null;
  metadata: ExplanationModelMetadata | null;
  generated_at: string;
  disclaimer: string;
}

export interface FeatureChange {
  feature_name: string;
  display_name: string;
  category: string;
  shap_a: number;
  shap_b: number;
  delta: number;
  magnitude: number;
  direction: string;
}

export interface HourlyExplanationComparison {
  status: ExplanationStatus;
  patient_id: string;
  hour_a: number;
  hour_b: number;
  explanation_a: PatientExplanation | null;
  explanation_b: PatientExplanation | null;
  changes: FeatureChange[];
  largest_changes: FeatureChange[];
  reason: string | null;
  generated_at: string;
  disclaimer: string;
}

export interface ModelCatalogEntry {
  model: ModelIdentity;
  explainable: boolean;
  explanation_method: ExplanationMethod | null;
  reason: string | null;
  background_rows: number;
  feature_schema_version: string;
  notes: string | null;
}

export interface ModelCatalogResponse {
  models: ModelCatalogEntry[];
  count: number;
}

export interface FeatureMetadata {
  feature_name: string;
  display_name: string;
  category: string;
  description: string | null;
  unit: string | null;
  feature_type: string | null;
  is_missing_indicator: boolean;
  is_engineered: boolean;
  metadata_available: boolean;
}

export interface AlertThresholdInfo {
  rule_id: string;
  threshold: number;
  source: string;
  version: string;
  dedup_window_hours: number;
  clear_on_below_threshold: boolean;
  clinically_validated: boolean;
  disclosure: string;
}

export interface AlertListResponse {
  alerts: AlertRecord[];
  count: number;
  filters: Record<string, string | number | null>;
}

export interface AlertEvaluationResult {
  action: string;
  alert: AlertRecord | null;
  reason: string | null;
  deduplicated: boolean;
}

export interface AlertClearRequest {
  reason: string;
}
