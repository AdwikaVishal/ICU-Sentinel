"""Check actual hyperparameters of currently saved models"""
import json
import joblib

print("=" * 70)
print("CURRENT SAVED MODEL HYPERPARAMETERS")
print("=" * 70)

# Extra Trees
print("\n[EXTRA TREES]")
with open("icu-sentinel/artifacts/p2_fast/extra_trees/extra_trees_1.0.0_metadata.json") as f:
    et_meta = json.load(f)

print(f"Timestamp: {et_meta['training_timestamp']}")
print(f"Features: {len(et_meta['selected_feature_names'])}")

et_artifact = joblib.load("icu-sentinel/artifacts/p2_fast/extra_trees/extra_trees_1.0.0.joblib")
if isinstance(et_artifact, dict) and "model" in et_artifact:
    et_model = et_artifact["model"]
else:
    et_model = et_artifact

# Handle Pipeline
if hasattr(et_model, "named_steps"):
    et_model = et_model.named_steps["classifier"]

print(f"n_estimators: {et_model.n_estimators}")
print(f"max_depth: {et_model.max_depth}")
print(f"max_features: {et_model.max_features}")
print(f"class_weight: {et_model.class_weight}")

# XGBoost
print("\n[XGBOOST]")
with open("icu-sentinel/artifacts/p2_fast/xgboost/xgboost_1.0.0_metadata.json") as f:
    xgb_meta = json.load(f)

print(f"Timestamp: {xgb_meta['training_timestamp']}")
print(f"Features: {len(xgb_meta['selected_feature_names'])}")

xgb_artifact = joblib.load("icu-sentinel/artifacts/p2_fast/xgboost/xgboost_1.0.0.joblib")
if isinstance(xgb_artifact, dict) and "model" in xgb_artifact:
    xgb_model = xgb_artifact["model"]
else:
    xgb_model = xgb_artifact

# Handle Pipeline
if hasattr(xgb_model, "named_steps"):
    xgb_model = xgb_model.named_steps["classifier"]

print(f"n_estimators: {xgb_model.n_estimators}")
print(f"max_depth: {xgb_model.max_depth}")
print(f"learning_rate: {xgb_model.learning_rate}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("These hyperparameters will be compared with p2_training_summary.json")
print("to determine which experiment generated the current artifacts.")
