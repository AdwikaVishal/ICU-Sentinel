# Dataset Documentation — PhysioNet/CinC Challenge 2019

**Project:** ICU Sentinel  
**Owner:** Person 1 (P1) — Data engineering  
**Part:** PART 2 — Dataset acquisition and exploration  
**Status:** Complete  
**Date:** 2026-11-03  

---

## 1. Dataset identification

| Field | Value |
|-------|-------|
| Title | Early Prediction of Sepsis from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019 |
| Version | 1.0.0 |
| Published | Aug. 5, 2019 |
| Authors | Reyna M., Josef C., Jeter R., Shashikumar S., Moody B., Westover M. B., Sharma A., Nemati S., Clifford G. D. |
| DOI | https://doi.org/10.13026/v64v-d857 |
| Official page | https://physionet.org/content/challenge-2019/1.0.0/ |
| Related publication | Reyna MA, et al. Early Prediction of Sepsis From Clinical Data: The PhysioNet/Computing in Cardiology Challenge. *Critical Care Medicine* 48(2):210-217 (2019). https://doi.org/10.1097/CCM.0000000000004145 |

---

## 2. Access requirements and licensing

- **Access type:** Open Access, but requires a **PhysioNet credentialed account**.
- **Credentialing:** Users must complete PhysioNet training (human-subjects research / data use agreement) and obtain credentials before downloading.
- **Download location:** https://archive.physionet.org/pnw/challenge-2019-request-access
- **Do not bypass access restrictions.** This repository does not contain, and must not contain, any downloaded patient data.
- **Database size:** ~42 MB (complete training database).
- **Contents:** Training set A (20,336 subjects) + Training set B (20,000 subjects) = 40,336 subjects total.

> **Access status (this environment):** No PhysioNet credentials are available, so the real dataset could not be downloaded. The loader is implemented to work against the actual dataset when access becomes available. Unit tests use synthetic fixtures only. All generated reports derived from local data are clearly marked as synthetic.

---

## 3. File organization and format

- One pipe-delimited text file (`.psv`) per patient.
- Example path: `training/p00101.psv`
- File naming convention: `p` followed by 5–6 digits, extension `.psv` (documented example: `p00101.psv`).
- Separator: `|` (pipe). No spaces in actual files (spaces in documentation examples are for readability only).
- Header: Each file has the same header (41 column names).
- Rows: Each row represents a single hour of ICU data.
- Missing values: Encoded as `NaN` (no recorded measurement at that time interval).
- The source data must remain unchanged; this project only reads it.

### Column layout (Table 1 of official documentation)

**Vital signs (columns 1–8)**

| # | Column | Description | Units |
|---|--------|-------------|-------|
| 1 | HR | Heart rate | beats per minute |
| 2 | O2Sat | Pulse oximetry | % |
| 3 | Temp | Temperature | Deg C |
| 4 | SBP | Systolic BP | mm Hg |
| 5 | MAP | Mean arterial pressure | mm Hg |
| 6 | DBP | Diastolic BP | mm Hg |
| 7 | Resp | Respiration rate | breaths per minute |
| 8 | EtCO2 | End tidal carbon dioxide | mm Hg |

**Laboratory values (columns 9–34)**

| # | Column | Description | Units |
|---|--------|-------------|-------|
| 9 | BaseExcess | Measure of excess bicarbonate | mmol/L |
| 10 | HCO3 | Bicarbonate | mmol/L |
| 11 | FiO2 | Fraction of inspired oxygen | % |
| 12 | pH | N/A | unitless |
| 13 | PaCO2 | Partial pressure of CO2 from arterial blood | mm Hg |
| 14 | SaO2 | Oxygen saturation from arterial blood | % |
| 15 | AST | Aspartate transaminase | IU/L |
| 16 | BUN | Blood urea nitrogen | mg/dL |
| 17 | Alkalinephos | Alkaline phosphatase | IU/L |
| 18 | Calcium | — | mg/dL |
| 19 | Chloride | — | mmol/L |
| 20 | Creatinine | — | mg/dL |
| 21 | Bilirubin_direct | Bilirubin direct | mg/dL |
| 22 | Glucose | Serum glucose | mg/dL |
| 23 | Lactate | Lactic acid | mg/dL |
| 24 | Magnesium | — | mmol/dL |
| 25 | Phosphate | — | mg/dL |
| 26 | Potassium | — | mmol/L |
| 27 | Bilirubin_total | Total bilirubin | mg/dL |
| 28 | TroponinI | Troponin I | ng/mL |
| 29 | Hct | Hematocrit | % |
| 30 | Hgb | Hemoglobin | g/dL |
| 31 | PTT | Partial thromboplastin time | seconds |
| 32 | WBC | Leukocyte count | count*10^3/uL |
| 33 | Fibrinogen | — | mg/dL |
| 34 | Platelets | — | count*10^3/uL |

**Demographics (columns 35–40)**

| # | Column | Description | Units |
|---|--------|-------------|-------|
| 35 | Age | Years (100 for patients 90 or above) | years |
| 36 | Gender | Female (0) or Male (1) | code |
| 37 | Unit1 | Administrative identifier for ICU unit (MICU) | code |
| 38 | Unit2 | Administrative identifier for ICU unit (SICU) | code |
| 39 | HospAdmTime | Hours between hospital admit and ICU admit | hours |
| 40 | ICULOS | ICU length-of-stay (hours since ICU admit) | hours |

**Outcome (column 41)**

| # | Column | Description |
|---|--------|-------------|
| 41 | SepsisLabel | Sepsis outcome label (0/1) |

Full machine-readable data dictionary: `data/reports/data_dictionary.csv`.

---

## 4. Target label convention (SepsisLabel)

Sepsis is defined per the **Sepsis-3 guidelines** (Singer et al., 2016): a two-point change in SOFA score plus clinical suspicion of infection (ordering of blood cultures or IV antibiotics).

Defined time points per patient:

- **t_suspicion**: Clinical suspicion of infection = earlier of IV antibiotics and blood cultures, within specified duration constraints (antibiotics first → cultures within 24 h; cultures first → antibiotics within 72 h; antibiotics administered ≥ 72 consecutive hours).
- **t_SOFA**: Occurrence of end-organ damage = two-point SOFA deterioration within a 24-hour period.
- **t_sepsis**: Onset time of sepsis = min(t_suspicion, t_SOFA), provided t_SOFA occurs no more than 24 hours before or 12 hours after t_suspicion (i.e., t_suspicion − 24 ≤ t_SOFA ≤ t_suspicion + 12). Otherwise the patient is not marked as septic.

**SepsisLabel encoding (shifted 6 hours ahead):**

- For sepsis patients: `SepsisLabel = 1` if `t ≥ t_sepsis − 6`, else `0`.
- For non-sepsis patients: `SepsisLabel = 0` for all hours.

This means the label at hour `t` indicates whether sepsis onset occurs within the next 6 hours (inclusive), i.e., the task is to predict sepsis **6 hours early**.

> The labels in the training data are **shifted ahead by six hours** relative to the recorded sepsis onset, so models trained to predict `SepsisLabel` at time `t` are effectively predicting sepsis six hours before its onset.

**Important (scope constraint):** This document records the official label convention only. No independent six-hour prediction label is invented here; that is handled in the feature/label engineering stage (PART 3–4) using the official convention.

---

## 5. Data access — manual download procedure

Because credentials are required, the dataset must be obtained manually:

1. Create a PhysioNet account and complete the required credentialing/training at https://physionet.org/.
2. Request access to the Challenge 2019 dataset at https://archive.physionet.org/pnw/challenge-2019-request-access.
3. Download the complete training database (~42 MB).
4. Place the extracted files under `data/raw/` (e.g., `data/raw/training/p00101.psv`).
5. **Do not commit `data/raw/` to Git** — it is gitignored.

The loader (`ml/p1_clinical/ingestion/dataset_loader.py`) accepts a `root` directory and discovers all `*.psv` files recursively.

---

## 6. How to run ingestion and exploration

```bash
# Activate environment
source .venv/bin/activate

# Discover and validate patient files (requires data/raw/ populated)
python - <<'EOF'
from ml.p1_clinical.ingestion.dataset_loader import PhysioNetChallenge2019Loader
from ml.p1_clinical.ingestion.exploration import DatasetExplorer

loader = PhysioNetChallenge2019Loader("data/raw")
catalog = loader.catalog()
print("Patients discovered:", catalog.n_patients)

explorer = DatasetExplorer("data/raw")
overview = explorer.compute_overview()
print(overview.to_dict())
EOF

# Generate exploration reports (synthetic template if no real data)
python - <<'EOF'
from ml.p1_clinical.ingestion.exploration import generate_synthetic_exploration_outputs
generate_synthetic_exploration_outputs("data/reports", n_patients=10, n_hours=24, seed=42)
EOF
```

---

## 7. Scope boundaries (P1 only)

- **Implemented by P1:** dataset loader, validation, exploration, data dictionary, target-label documentation.
- **Not implemented by P1 (P2/P3):** boosting models, temporal models, SHAP, alerts, React application.
- **Not implemented in PART 2:** preprocessing, feature engineering, six-hour label engineering, clinical baselines, ML models. These are deferred to Parts 3–6.

---

## 8. Citations

When using this resource, please cite:

```
Reyna, M., Josef, C., Jeter, R., Shashikumar, S., Moody, B., Westover, M. B.,
Sharma, A., Nemati, S., & Clifford, G. D. (2019). Early Prediction of Sepsis
from Clinical Data: The PhysioNet/Computing in Cardiology Challenge 2019
(version 1.0.0). PhysioNet. https://doi.org/10.13026/v64v-d857
```

And the original publication:

```
Reyna MA, Josef CS, Jeter R, Shashikumar SP, Westover MB, Nemati S, Clifford GD,
Sharma A. Early Prediction of Sepsis From Clinical Data: The PhysioNet/
Computing in Cardiology Challenge. Critical Care Medicine 48(2):210-217 (2019).
https://doi.org/10.1097/CCM.0000000000004145
```

Plus the standard PhysioNet citation:

```
Pollard, T., et al. (2026). PhysioNet as a global platform for biomedical
research. Nature Health. https://doi.org/10.1038/s44360-026-00096-z
```
