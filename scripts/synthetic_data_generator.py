"""
scripts/synthetic_data_generator.py
Generate a synthetic discharge-summary dataset that mimics MIMIC-IV structure.

Purpose: allows the full pipeline (train.py / evaluate.py / predict.py) to be run
end-to-end on a local machine BEFORE PhysioNet/CITI credentialing is complete, so
that genuine metrics can be produced for development and resume-claim verification.
This is NOT a substitute for MIMIC-IV in a real clinical deployment or publication;
swap in the real dataset (see data/README.md) once credentialed access is granted.

Usage:
    python scripts/synthetic_data_generator.py --n-records 2000 --output-dir data
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ── Building blocks for synthetic discharge summaries ──────────────────────────

AGES = list(range(40, 95))
SEXES = ["male", "female"]

DIAGNOSES_LOW_RISK = [
    "appendicitis", "minor fracture", "elective hip replacement",
    "routine gallbladder removal", "uncomplicated pneumonia",
]

DIAGNOSES_HIGH_RISK = [
    "congestive heart failure", "chronic kidney disease with dialysis dependence",
    "sepsis", "COPD exacerbation", "myocardial infarction",
    "decompensated cirrhosis",
]

SOCIAL_LOW_RISK = [
    "lives with spouse and has strong family support",
    "has a reliable caregiver at home",
    "is independent with activities of daily living",
]

SOCIAL_HIGH_RISK = [
    "lives alone with no identified support system",
    "has a history of non-compliance with prior discharge instructions",
    "reports housing insecurity",
    "has documented alcohol use disorder",
]

FOLLOWUP_LOW_RISK = [
    "Follow-up with primary care arranged within 7 days.",
    "Outpatient follow-up scheduled and confirmed with patient.",
]

FOLLOWUP_HIGH_RISK = [
    "No follow-up appointment could be arranged prior to discharge.",
    "Patient was unable to confirm transportation to follow-up visit.",
]

TEMPLATE = (
    "Patient is a {age}-year-old {sex} admitted with {diagnosis}. "
    "Hospital course was notable for {course}. "
    "Patient {social}. {followup} "
    "Discharge medications reconciled. {extra}"
)

COURSE_LOW_RISK = [
    "an uncomplicated recovery", "good response to treatment",
    "stable vital signs throughout admission",
]
COURSE_HIGH_RISK = [
    "multiple complications requiring escalation of care",
    "a prolonged ICU stay", "poor response to initial treatment",
]

EXTRA_HIGH_RISK = [
    "Patient is on 12 medications at discharge (polypharmacy).",
    "Family meeting held to discuss goals of care given disease severity.",
    "Patient exhibited mild confusion at time of discharge.",
]
EXTRA_LOW_RISK = [
    "Patient tolerated oral intake well prior to discharge.",
    "Patient ambulated independently prior to discharge.",
]


def generate_record(readmitted: int, rng: random.Random) -> str:
    """
    Generate one synthetic discharge summary string.

    Args:
        readmitted: 1 to bias toward high-risk language, 0 for low-risk.
        rng: Seeded random.Random instance for reproducibility.

    Returns:
        Synthetic discharge summary text.
    """
    age = rng.choice(AGES)
    sex = rng.choice(SEXES)

    if readmitted == 1:
        diagnosis = rng.choice(DIAGNOSES_HIGH_RISK)
        social = rng.choice(SOCIAL_HIGH_RISK)
        followup = rng.choice(FOLLOWUP_HIGH_RISK)
        course = rng.choice(COURSE_HIGH_RISK)
        extra = rng.choice(EXTRA_HIGH_RISK)
    else:
        diagnosis = rng.choice(DIAGNOSES_LOW_RISK)
        social = rng.choice(SOCIAL_LOW_RISK)
        followup = rng.choice(FOLLOWUP_LOW_RISK)
        course = rng.choice(COURSE_LOW_RISK)
        extra = rng.choice(EXTRA_LOW_RISK)

    return TEMPLATE.format(
        age=age, sex=sex, diagnosis=diagnosis, course=course,
        social=social, followup=followup, extra=extra,
    )


def generate_dataset(n_records: int, positive_rate: float, seed: int) -> pd.DataFrame:
    """
    Generate a full synthetic dataset matching the MIMIC-IV pipeline schema.

    Args:
        n_records: Total number of records to generate.
        positive_rate: Fraction of records labeled as readmitted (1).
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: subject_id, hadm_id, discharge_summary, readmitted_30d.
    """
    rng = random.Random(seed)
    n_positive = int(n_records * positive_rate)
    labels = [1] * n_positive + [0] * (n_records - n_positive)
    rng.shuffle(labels)

    records = []
    for i, label in enumerate(labels):
        records.append({
            "subject_id": 100000 + i,
            "hadm_id": 200000 + i,
            "discharge_summary": generate_record(label, rng),
            "readmitted_30d": label,
        })

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic MIMIC-IV-style discharge data")
    parser.add_argument("--n-records", type=int, default=2000)
    parser.add_argument("--positive-rate", type=float, default=0.25,
                         help="Fraction of records labeled readmitted (MIMIC-IV 30-day rate is ~22-27%%)")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating %d synthetic records (positive rate: %.0f%%)", args.n_records, args.positive_rate * 100)
    df = generate_dataset(args.n_records, args.positive_rate, args.seed)

    out_path = output_dir / "mimic_iv_sample.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved synthetic dataset to %s", out_path)
    logger.info("Next: python scripts/prepare_data.py --input %s", out_path)


if __name__ == "__main__":
    main()
