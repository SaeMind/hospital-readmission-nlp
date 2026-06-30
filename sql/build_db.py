"""
sql/build_db.py
Load mimic_iv_sample.csv into a normalized SQLite database with 3 tables:
  patients     — one row per subject_id
  admissions   — one row per hadm_id (FK → patients)
  discharge_notes — one row per hadm_id (FK → admissions)

Usage:
    python sql/build_db.py --input data/mimic_iv_sample.csv --db data/readmission.db
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def build_database(csv_path: str, db_path: str) -> None:
    df = pd.read_csv(csv_path)
    logger.info("Loaded %d rows from %s", len(df), csv_path)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS discharge_notes;
        DROP TABLE IF EXISTS admissions;
        DROP TABLE IF EXISTS patients;

        CREATE TABLE patients (
            subject_id   INTEGER PRIMARY KEY,
            age_proxy    INTEGER
        );

        CREATE TABLE admissions (
            hadm_id        INTEGER PRIMARY KEY,
            subject_id     INTEGER NOT NULL REFERENCES patients(subject_id),
            readmitted_30d INTEGER NOT NULL CHECK(readmitted_30d IN (0, 1))
        );

        CREATE TABLE discharge_notes (
            hadm_id           INTEGER PRIMARY KEY REFERENCES admissions(hadm_id),
            discharge_summary TEXT NOT NULL
        );
    """)

    # patients — derive a synthetic age proxy from subject_id offset
    patients = df[["subject_id"]].drop_duplicates().copy()
    patients["age_proxy"] = (patients["subject_id"] % 55) + 40
    patients.to_sql("patients", conn, if_exists="append", index=False)

    admissions = df[["hadm_id", "subject_id", "readmitted_30d"]].drop_duplicates()
    admissions.to_sql("admissions", conn, if_exists="append", index=False)

    notes = df[["hadm_id", "discharge_summary"]].drop_duplicates()
    notes.to_sql("discharge_notes", conn, if_exists="append", index=False)

    conn.commit()
    conn.close()

    logger.info("Database written to %s", db_path)
    logger.info("Tables: patients, admissions, discharge_notes")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/mimic_iv_sample.csv")
    parser.add_argument("--db", default="data/readmission.db")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_database(args.input, args.db)
