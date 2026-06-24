"""
evaluate.py
Compute full evaluation metrics on the held-out test set and write a report.

Usage:
    python evaluate.py --test-path data/test.csv --model-path outputs/readmission_biobert_best.pt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import FINETUNED_WEIGHTS, OUTPUT_DIR, TEST_CSV
from data_loader import MIMICDischargeDataset
from biobert_model import ReadmissionBioBERT
from utils import save_json, select_device, timestamped_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BioBERT on test set")
    parser.add_argument("--test-path", default=str(TEST_CSV))
    parser.add_argument("--model-path", default=str(FINETUNED_WEIGHTS))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device()
    output_dir = Path(args.output_dir)

    # ── Load model ────────────────────────────────────────────────────────────
    model = ReadmissionBioBERT()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    logger.info("Model loaded from %s", args.model_path)

    # ── Load test data ────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    test_dataset = MIMICDischargeDataset(args.test_path)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # ── Inference ─────────────────────────────────────────────────────────────
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Scoring test set", unit="batch"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].tolist()

            logits = model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()
            preds = [1 if p > 0.5 else 0 for p in probs]

            all_labels.extend(labels)
            all_preds.extend(preds)
            all_probs.extend(probs)

    # ── Metrics ───────────────────────────────────────────────────────────────
    auroc = roc_auc_score(all_labels, all_probs)
    report = classification_report(all_labels, all_preds, target_names=["Not Readmitted", "Readmitted"])
    cm = confusion_matrix(all_labels, all_preds).tolist()

    logger.info("AUROC: %.4f", auroc)
    logger.info("\n%s", report)

    # ── Save report ───────────────────────────────────────────────────────────
    results = {
        "auroc": round(auroc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    json_path = timestamped_path(output_dir, "evaluation_results", ".json")
    save_json(results, json_path)

    md_path = timestamped_path(output_dir, "model_evaluation_report", ".md")
    md_content = f"""# Model Evaluation Report

## Test Set Performance

| Metric | Value |
|--------|-------|
| AUROC  | {auroc:.4f} |

## Classification Report

```
{report}
```

## Confusion Matrix

|  | Predicted: Not Readmitted | Predicted: Readmitted |
|--|--|--|
| Actual: Not Readmitted | {cm[0][0]} | {cm[0][1]} |
| Actual: Readmitted | {cm[1][0]} | {cm[1][1]} |
"""
    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Report saved to %s", md_path)


if __name__ == "__main__":
    main()
