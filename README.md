# Hospital Readmission NLP Summarizer

> Fine-tuned BioBERT pipeline that extracts readmission risk factors from clinical discharge summaries and generates structured risk assessments.

## Overview

Hospital readmissions within 30 days cost the US healthcare system approximately $17.5B annually. This project fine-tunes [BioBERT](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) on MIMIC-IV discharge notes to predict whether a patient will be readmitted within 30 days, and generates a clinician-facing risk report identifying the key contributing factors.

**Demonstrated skills:** Clinical NLP, transformer fine-tuning, class-imbalanced classification, rule-based NER, SQL data extraction layer, modular Python pipeline design, reproducible ML workflows.

---

## Model Architecture

```
Discharge Summary (free text)
        │
        ▼
ClinicalTextPreprocessor
  - PHI redaction
  - Abbreviation expansion
  - Whitespace normalization
        │
        ▼
BioBERT Tokenizer (max 512 tokens, truncate + pad)
        │
        ▼
ReadmissionBioBERT
  - BioBERT encoder (dmis-lab/biobert-base-cased-v1.1)
  - [CLS] token pooling
  - Dropout(0.3) → Dense(768→256) → ReLU → Dropout → Linear(256→2)
        │
        ▼
Softmax → Readmission probability (0–1)
        │
        ▼
RiskFactorExtractor (regex over preprocessed text)
        │
        ▼
RiskSummaryGenerator → Structured risk report
```

---

## Results

### Training (20,000 synthetic records · 12,000 train / 3,000 val / 5,000 test · MPS device · batch 32 · max_length 128)

| Epoch | Train Loss | Val Accuracy | Val F1 | Val Recall |
|-------|-----------|--------------|--------|------------|
| 1     | 0.0239    | 1.0000       | 1.0000 | 1.0000     |
| 2     | 0.0003    | 1.0000       | 1.0000 | 1.0000     |
| 3     | 0.0001    | 1.0000       | 1.0000 | 1.0000     |

### Test-Set Evaluation (`evaluate.py` · 5,000 held-out records)

| Metric | Not Readmitted | Readmitted | Macro Avg |
|--------|---------------|------------|-----------|
| Precision | 1.00 | 1.00 | 1.00 |
| Recall    | 1.00 | 1.00 | 1.00 |
| F1        | 1.00 | 1.00 | 1.00 |
| **AUROC** | — | — | **1.0000** |

**Confusion matrix** (5,000 test records, 25% positive rate):

|  | Predicted: Not Readmitted | Predicted: Readmitted |
|--|--|--|
| Actual: Not Readmitted | 3,750 | 0 |
| Actual: Readmitted | 0 | 1,250 |

> **Important:** These metrics are from a run on **synthetic data only** (not MIMIC-IV). The synthetic generator uses non-overlapping template language per label class, so perfect separation is expected and *does not reflect real clinical performance*. MIMIC-IV credentialing (CITI + PhysioNet) is pending; replace these results with `evaluate.py` output on real data once access is granted.
>
> *Run date: 2026-06-30. Metrics file: `outputs/evaluation_results_20260630_212404.json`. Training metrics: `outputs/training_metrics_20260630_203522.json`.*

---

## Setup

```bash
# Clone the repository
git clone https://github.com/SaeMind/hospital-readmission-nlp.git
cd hospital-readmission-nlp

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
```

---

## Data Requirements

This project uses **MIMIC-IV** discharge notes. Access requires:

1. Complete the CITI Program certification: https://about.citiprogram.org/
2. Create a PhysioNet account and apply for credentialed access: https://physionet.org/
3. Accept the MIMIC-IV Data Use Agreement
4. Download the `discharge` table from the `note` module

See `data/README.md` for the expected CSV schema and split preparation instructions.

### Running without MIMIC-IV access (development / synthetic mode)

A synthetic data generator is included so the full pipeline can be exercised before credentialing is complete:

```bash
python scripts/synthetic_data_generator.py --n-records 2000 --output-dir data
python scripts/prepare_data.py --input data/mimic_iv_sample.csv
```

This produces `train.csv`, `val.csv`, and `test.csv` with the same schema MIMIC-IV would produce, using template-generated discharge summaries rather than real patient data. Metrics from synthetic data are for pipeline validation only and should not be cited as clinical performance.

---

## How to Train

```bash
# Prepare train/val/test splits from raw MIMIC-IV data
python scripts/prepare_data.py --input data/mimic_iv_discharge_summaries.csv

# Fine-tune BioBERT (approx. 30–45 min on a single GPU)
python train.py --epochs 3 --batch-size 16 --learning-rate 2e-5

# Outputs:
#   outputs/readmission_biobert_best.pt      (best checkpoint by val F1)
#   outputs/readmission_biobert_finetuned.pt (final epoch)
#   outputs/training_metrics_<timestamp>.json
```

---

## How to Run Inference

```bash
# From a text file
python predict.py --input path/to/discharge_summary.txt

# From a string
python predict.py --text "76-year-old male admitted with heart failure, lives alone, no follow-up arranged."

# Output: printed risk report + outputs/readmission_prediction_<timestamp>.json
```

**Sample output:**

```
READMISSION RISK ASSESSMENT
============================================
Risk Level  : HIGH
Risk Score  : 78.0%

KEY RISK FACTORS IDENTIFIED:

  Comorbidities:
    - Heart Failure

  Social Factors:
    - Limited Social Support

  Clinical Factors:
    - No Follow Up Arranged

RECOMMENDED ACTIONS:
  - Initiate intensive discharge planning
  - Schedule follow-up within 7 days
  - Refer to home health or transitional care program
  - Reconcile all medications before discharge
```

---

## SQL Extraction Layer

The pipeline includes a normalized SQLite database to demonstrate an end-to-end SQL → Python → ML architecture:

```
CSV (mimic_iv_sample.csv)
        │
        ▼
sql/build_db.py
  → patients        (subject_id, age_proxy)
  → admissions      (hadm_id, subject_id, readmitted_30d)
  → discharge_notes (hadm_id, discharge_summary)
        │
        ▼
sql/extract_cohort.sql  — JOIN query reconstructs model-ready flat format
        │
        ▼
MIMICDischargeDataset → BioBERT pipeline
```

Build the database and verify extraction:

```bash
python sql/build_db.py --input data/mimic_iv_sample.csv --db data/readmission.db
sqlite3 data/readmission.db < sql/extract_cohort.sql | head
```

The SQL-extracted cohort is verified to be a byte-exact match to the source CSV (confirmed on 2026-06-30 run). In a production MIMIC-IV deployment, `extract_cohort.sql` would run against a PostgreSQL/BigQuery instance rather than SQLite.

---

## How to Export Predictions for Tableau

```bash
python scripts/export_predictions.py \
    --test-path data/test.csv \
    --model-path outputs/readmission_biobert_best.pt

# Outputs: outputs/predictions_for_tableau_<timestamp>.csv
# Columns: subject_id, hadm_id, readmitted_30d, risk_score, predicted_label,
#          risk_level, comorbidities_count, social_factors_count,
#          clinical_factors_count, risk_factors_json
```

---

## How to Evaluate

```bash
python evaluate.py \
    --test-path data/test.csv \
    --model-path outputs/readmission_biobert_best.pt

# Outputs:
#   outputs/evaluation_results_<timestamp>.json
#   outputs/model_evaluation_report_<timestamp>.md
```

---

## Risk Factors Extracted

| Category | Factors |
|---|---|
| Comorbidities | Diabetes, Heart failure, Coronary artery disease, CKD, Pneumonia, Sepsis, COPD, Atrial fibrillation |
| Social factors | Homelessness, Substance use, Poor medication compliance, Limited social support |
| Clinical factors | Acute illness, Polypharmacy, Cognitive impairment, Recent surgery, No follow-up arranged, Age > 75 |

---

## Technologies Used

| Component | Technology |
|---|---|
| Pretrained model | BioBERT (dmis-lab/biobert-base-cased-v1.1) |
| Deep learning | PyTorch 2.1, HuggingFace Transformers 4.35 |
| Data processing | Pandas, scikit-learn |
| Clinical NLP | Regex-based PHI redaction, rule-based NER |
| Configuration | python-dotenv, config.py |
| Data validation | Pydantic 2.x |
| Language | Python 3.10+ with full type hints |

---

## Repository Structure

```
hospital-readmission-nlp/
├── src/
│   ├── data_loader.py          # MIMIC-IV Dataset + DataLoader factory
│   ├── text_preprocessor.py    # PHI redaction, abbreviation expansion
│   ├── biobert_model.py        # Model architecture + Trainer
│   ├── readmission_classifier.py  # Inference wrapper
│   ├── summary_generator.py    # Risk factor extraction + report generation
│   ├── config.py               # Centralized configuration
│   └── utils.py                # Seeding, I/O helpers, metric formatting
├── scripts/
│   ├── synthetic_data_generator.py  # Dev-mode dataset (no MIMIC-IV access needed)
│   ├── prepare_data.py         # Stratified train/val/test splits
│   └── export_predictions.py   # Inference + risk factor export → Tableau CSV
├── sql/
│   ├── build_db.py             # Load CSV into normalized SQLite (patients/admissions/notes)
│   └── extract_cohort.sql      # JOIN query → model-ready flat format
├── tests/
│   └── test_preprocessing.py   # Unit tests (no GPU/model download required)
├── data/
│   ├── raw/                    # MIMIC-IV downloads land here (gitignored)
│   ├── processed/              # Tokenized/cleaned intermediates (gitignored)
│   └── README.md               # Data sourcing and CITI requirement
├── models/                     # Model checkpoints (gitignored)
├── notebooks/                  # Exploratory analysis (gitignored)
├── outputs/                    # Predictions and reports (gitignored)
├── train.py                    # Training entry point
├── predict.py                  # Inference entry point
├── evaluate.py                 # Evaluation entry point
├── requirements.txt
├── .env.example
├── LICENSE
└── README.md
```

---

## Citation

If you use this project, please cite the BioBERT paper:

Lee, J., et al. (2020). BioBERT: a pre-trained biomedical language representation model for biomedical text mining. *Bioinformatics*, 36(4), 1234–1240.

---

## License

MIT License. See LICENSE for details. MIMIC-IV data is subject to its own PhysioNet Data Use Agreement and is not redistributed here.
