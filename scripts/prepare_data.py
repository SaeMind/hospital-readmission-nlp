"""
scripts/prepare_data.py
Split MIMIC-IV discharge summary CSV into stratified train/val/test splits.

Usage:
    python scripts/prepare_data.py --input data/mimic_iv_discharge_summaries.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from config import DATA_DIR, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare train/val/test splits from MIMIC-IV CSV")
    parser.add_argument("--input", required=True, help="Path to raw MIMIC-IV discharge CSV")
    parser.add_argument("--output-dir", default=str(DATA_DIR))
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading %s", args.input)
    df = pd.read_csv(args.input)
    logger.info("Total records: %d", len(df))

    # Train+val vs. test
    train_val, test = train_test_split(
        df, test_size=args.test_size, stratify=df["readmitted_30d"], random_state=args.seed
    )

    # Train vs. val (from the remaining)
    val_fraction = args.val_size / (1 - args.test_size)
    train, val = train_test_split(
        train_val, test_size=val_fraction, stratify=train_val["readmitted_30d"], random_state=args.seed
    )

    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        out_path = output_dir / f"{split_name}.csv"
        split_df.to_csv(out_path, index=False)
        pos = int(split_df["readmitted_30d"].sum())
        logger.info("%s: %d records | positive rate: %.1f%%", split_name, len(split_df), 100 * pos / len(split_df))

    logger.info("Splits saved to %s", output_dir)


if __name__ == "__main__":
    main()
