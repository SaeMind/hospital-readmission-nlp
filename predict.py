"""
predict.py
Inference entry point: score a single discharge summary and print a risk report.

Usage:
    python predict.py --input discharge_summary.txt
    python predict.py --text "76-year-old male with CHF, lives alone..."
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import FINETUNED_WEIGHTS, OUTPUT_DIR
from readmission_classifier import ReadmissionPredictor
from summary_generator import RiskFactorExtractor, RiskSummaryGenerator
from text_preprocessor import ClinicalTextPreprocessor
from utils import save_json, timestamped_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict 30-day readmission risk")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", help="Path to a .txt file containing a discharge summary")
    group.add_argument("--text", help="Discharge summary text passed directly as a string")
    parser.add_argument("--model-path", default=str(FINETUNED_WEIGHTS))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Load text ─────────────────────────────────────────────────────────────
    if args.input:
        summary_text = Path(args.input).read_text(encoding="utf-8")
    else:
        summary_text = args.text

    # ── Preprocess ────────────────────────────────────────────────────────────
    preprocessor = ClinicalTextPreprocessor()
    clean_text = preprocessor.preprocess(summary_text)

    # ── Predict ───────────────────────────────────────────────────────────────
    predictor = ReadmissionPredictor(model_path=args.model_path)
    prediction = predictor.predict(clean_text)

    # ── Risk factor extraction ────────────────────────────────────────────────
    extractor = RiskFactorExtractor()
    risk_factors = extractor.extract_risk_factors(clean_text)

    # ── Report generation ─────────────────────────────────────────────────────
    generator = RiskSummaryGenerator()
    report = generator.generate(
        readmission_risk=prediction["readmission_risk"],
        risk_factors=risk_factors,
    )

    print("\n" + report)

    # ── Save output ───────────────────────────────────────────────────────────
    output = {
        "input_summary": summary_text[:500] + "..." if len(summary_text) > 500 else summary_text,
        **prediction,
        "risk_factors": risk_factors,
        "report": report,
    }

    out_path = timestamped_path(Path(args.output_dir), "readmission_prediction", ".json")
    save_json(output, out_path)
    logger.info("Prediction saved to %s", out_path)


if __name__ == "__main__":
    main()
