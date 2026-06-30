"""
scripts/export_predictions.py
Run inference on test.csv using the trained model, then export a CSV with:
  subject_id, hadm_id, readmitted_30d (true label), risk_score,
  predicted_label, risk_level, + one column per risk factor category
  (comorbidities_count, social_factors_count, clinical_factors_count)
  and a serialized JSON column of the full factor dict.

Output: outputs/predictions_for_tableau_<timestamp>.csv

Usage:
    python scripts/export_predictions.py \
        --test-path data/test.csv \
        --model-path outputs/readmission_biobert_best.pt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from biobert_model import ReadmissionBioBERT
from data_loader import MIMICDischargeDataset
from summary_generator import RiskFactorExtractor
from utils import select_device, timestamped_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def risk_level(score: float) -> str:
    if score > 0.7:
        return "HIGH"
    if score > 0.4:
        return "MODERATE"
    return "LOW"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path", default="data/test.csv")
    parser.add_argument("--model-path", default="outputs/readmission_biobert_best.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()

    model = ReadmissionBioBERT()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s on %s", args.model_path, device)

    df = pd.read_csv(args.test_path)
    dataset = MIMICDischargeDataset(args.test_path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    all_probs: list[float] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference", unit="batch"):
            logits = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            all_probs.extend(probs)

    extractor = RiskFactorExtractor()
    risk_factors_list = [extractor.extract_risk_factors(t) for t in tqdm(df["discharge_summary"], desc="Extracting risk factors")]

    df["risk_score"] = all_probs
    df["predicted_label"] = [1 if p > 0.5 else 0 for p in all_probs]
    df["risk_level"] = [risk_level(p) for p in all_probs]
    df["comorbidities_count"] = [len(rf["comorbidities"]) for rf in risk_factors_list]
    df["social_factors_count"] = [len(rf["social_factors"]) for rf in risk_factors_list]
    df["clinical_factors_count"] = [len(rf["clinical_factors"]) for rf in risk_factors_list]
    df["risk_factors_json"] = [json.dumps(rf) for rf in risk_factors_list]

    out_path = timestamped_path(Path(args.output_dir), "predictions_for_tableau", ".csv")
    df.to_csv(out_path, index=False)
    logger.info("Predictions saved to %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
