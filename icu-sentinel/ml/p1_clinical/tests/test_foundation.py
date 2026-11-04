"""P1 foundation sanity tests.

Verifies shared schemas and synthetic fixtures work without real patient data.
"""

from ml.common.fixtures import make_synthetic_metadata, make_synthetic_patients
from ml.common.schemas import FeatureSet, PatientRecord, SplitMetadata


def test_patient_record_schema():
    rec = PatientRecord(
        patient_id="P001",
        hospital_id="H1",
        icu_id="ICU1",
        age=65,
        sepsis_label=1,
    )
    assert rec.patient_id == "P001"
    assert rec.sepsis_label == 1


def test_split_metadata_schema():
    md = SplitMetadata(
        split_name="train",
        patient_ids=["P001", "P002"],
        n_patients=2,
        n_positive=1,
        n_negative=1,
        random_state=42,
    )
    assert md.n_patients == 2
    assert md.split_name == "train"


def test_feature_set_schema():
    fs = FeatureSet(
        patient_id="P001",
        hour=3,
        features={"HR": 80.0, "HR_mean_6h": 81.0},
        sepsis_label=0,
    )
    assert fs.hour == 3
    assert fs.features["HR"] == 80.0


def test_synthetic_patients_shape():
    df = make_synthetic_patients(n_patients=3, hours_per_patient=10, seed=7)
    assert df.shape[0] == 3 * 10 * 8  # 3 patients x 10 hours x 8 parameters
    assert set(df.columns) >= {"patient_id", "hour", "parameter", "value"}
    assert df["patient_id"].nunique() == 3


def test_synthetic_metadata_shape():
    df = make_synthetic_metadata(n_patients=4, seed=7)
    assert len(df) == 4
    assert set(df.columns) >= {
        "patient_id", "hospital_id", "icu_id", "gender", "age",
        "died_in_hospital", "sepsis_label",
    }
