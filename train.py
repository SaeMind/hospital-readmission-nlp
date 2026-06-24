"""
train.py
Entry point: fine-tune BioBERT on MIMIC-IV discharge summaries for 30-day readmission prediction.

Usage:
    python train.py --epochs 3 --batch-size 16 --learning-rate 2e-5
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import BATCH_SIZE, EPOCHS, LEARNING_RATE, OUTPUT_DIR, SEED, TRAIN_CSV, VAL_CSV
from data_loader import create_data_loaders
from biobert_model import ReadmissionBioBERT, ReadmissionTrainer
from utils import format_metrics, save_json, select_device, set_seed, timestamped_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune BioBERT for readmission prediction")
    parser.add_argument("--train-path", default=str(TRAIN_CSV))
    parser.add_argument("--val-path", default=str(VAL_CSV))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device()
    logger.info("Device: %s", device)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data...")
    train_loader, val_loader = create_data_loaders(
        args.train_path,
        args.val_path,
        batch_size=args.batch_size,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    logger.info("Initializing BioBERT model...")
    model = ReadmissionBioBERT()
    trainer = ReadmissionTrainer(model, device=device, learning_rate=args.learning_rate)

    # ── Training loop ─────────────────────────────────────────────────────────
    metrics_log = []
    best_f1 = 0.0

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d / %d", epoch, args.epochs)
        train_loss = trainer.train_epoch(train_loader)
        accuracy, f1, recall = trainer.evaluate(val_loader)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_accuracy": round(accuracy, 4),
            "val_f1": round(f1, 4),
            "val_recall": round(recall, 4),
        }
        metrics_log.append(row)
        logger.info("Metrics: %s", format_metrics(accuracy, f1, recall))

        if f1 > best_f1:
            best_f1 = f1
            best_path = output_dir / "readmission_biobert_best.pt"
            torch.save(model.state_dict(), best_path)
            logger.info("New best model saved to %s (F1=%.4f)", best_path, best_f1)

    # ── Save final model + metrics ────────────────────────────────────────────
    final_model_path = output_dir / "readmission_biobert_finetuned.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info("Final model saved to %s", final_model_path)

    metrics_path = timestamped_path(output_dir, "training_metrics", ".json")
    save_json(metrics_log, metrics_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Best F1       : %.4f", best_f1)
    logger.info("Final accuracy: %.4f", metrics_log[-1]["val_accuracy"])
    logger.info("Metrics saved : %s", metrics_path)


if __name__ == "__main__":
    main()
