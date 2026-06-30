# CHANGELOG_REVIEW

## 2026-06-30 — Real Run, SQL Layer, README Update

### What was done

#### 1. Illustrative metrics replaced with real run results

The `README.md` previously contained a results table labeled as
"illustrative targets based on published BioBERT readmission literature"
with a disclaimer directing users to run `evaluate.py` before citing any number.

That table has been **removed and replaced** with metrics from an actual end-to-end
pipeline run completed on 2026-06-30. No numbers in the README are fabricated.

**Real run configuration:**
- Dataset: 20,000 synthetic records (`scripts/synthetic_data_generator.py --n-records 20000 --positive-rate 0.25`)
- Splits: 12,000 train / 3,000 val / 5,000 test (stratified, `scripts/prepare_data.py`)
- Model: `dmis-lab/biobert-base-cased-v1.1` fine-tuned 3 epochs, batch 32, lr 2e-5, max_length 128
- Device: Apple Silicon MPS (GPU)
- Runtime: ~47 minutes (train) + ~5 min (eval) + ~5 min (predictions export)

**Real metrics (test set, `outputs/evaluation_results_20260630_212404.json`):**

| Metric | Value |
|--------|-------|
| AUROC  | 1.0000 |
| Precision (readmitted) | 1.00 |
| Recall (readmitted) | 1.00 |
| F1 (readmitted) | 1.00 |
| Accuracy | 1.00 |

**Confusion matrix:**

|  | Predicted Not Readmitted | Predicted Readmitted |
|--|--|--|
| Actual Not Readmitted | 3,750 | 0 |
| Actual Readmitted | 0 | 1,250 |

**Why perfect scores:** The synthetic generator (`scripts/synthetic_data_generator.py`)
uses completely non-overlapping template language per label class — e.g., "lives alone
with no identified support system" appears exclusively in readmitted=1 records while
"lives with spouse and has strong family support" appears exclusively in readmitted=0
records. BioBERT learns these discriminating phrases within the first epoch (train
loss drops from 0.65 → 0.02 by end of epoch 1). Perfect separation on synthetic data
is expected and does **not** imply clinical predictive performance. Real-world AUC
on MIMIC-IV discharge notes in the published BioBERT readmission literature ranges
from approximately 0.72–0.85.

**Training metrics file:** `outputs/training_metrics_20260630_203522.json`

---

#### 2. SQL extraction layer added

New files:
- `sql/build_db.py` — Loads `mimic_iv_sample.csv` into a normalized SQLite database
  with three tables: `patients`, `admissions`, `discharge_notes`
- `sql/extract_cohort.sql` — JOIN query that reconstructs the model-ready flat format
  from the normalized schema

**Verification:** The SQL-extracted dataset is a byte-exact match to the source CSV
(20,000 rows, confirmed on 2026-06-30). The SQL layer demonstrates the full
CSV → SQL normalization → JOIN extraction → BioBERT pipeline architecture that would
apply in a production MIMIC-IV/PostgreSQL environment.

---

#### 3. Predictions export script added

New file: `scripts/export_predictions.py`

Runs inference on any CSV split and writes a Tableau-ready CSV with columns:
`subject_id`, `hadm_id`, `readmitted_30d` (true label), `risk_score`,
`predicted_label`, `risk_level`, `comorbidities_count`, `social_factors_count`,
`clinical_factors_count`, `risk_factors_json`.

---

#### 4. README updated

- Removed: illustrative targets table and its disclaimer
- Added: real training metrics table, real test-set evaluation table, confusion matrix,
  SQL Architecture section, Predictions Export section, updated repo structure tree
- Updated: "Demonstrated skills" line to include SQL data extraction layer
- Retained: all disclaimers about synthetic data vs. MIMIC-IV clinical applicability

---

### Pending (MIMIC-IV credentialing)

- CITI Program certification (in progress)
- PhysioNet credentialed access application
- Re-run `evaluate.py` on real MIMIC-IV discharge notes once access is granted;
  replace synthetic-run results with real-data results at that time
