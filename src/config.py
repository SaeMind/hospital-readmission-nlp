"""
config.py
Central configuration: paths, model hyperparameters, training settings.
All values can be overridden via environment variables or a .env file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if present (no-op if missing — falls back to defaults below)
load_dotenv()

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
TRAIN_CSV = Path(os.getenv("TRAIN_CSV", DATA_DIR / "train.csv"))
VAL_CSV = Path(os.getenv("VAL_CSV", DATA_DIR / "val.csv"))
TEST_CSV = Path(os.getenv("TEST_CSV", DATA_DIR / "test.csv"))
MIMIC_SAMPLE_CSV = Path(os.getenv("MIMIC_SAMPLE_CSV", DATA_DIR / "mimic_iv_sample.csv"))

# ── Model paths ───────────────────────────────────────────────────────────────
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT_DIR / "models"))
PRETRAINED_DIR = MODEL_DIR / "biobert_pretrained"
FINETUNED_DIR = MODEL_DIR / "readmission_finetuned"
FINETUNED_WEIGHTS = FINETUNED_DIR / "model.pt"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", ROOT_DIR / "outputs"))

# ── Model hyperparameters ─────────────────────────────────────────────────────
MODEL_NAME: str = os.getenv("MODEL_NAME", "dmis-lab/biobert-base-cased-v1.1")
MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", 512))
HIDDEN_SIZE: int = int(os.getenv("HIDDEN_SIZE", 768))
DROPOUT_RATE: float = float(os.getenv("DROPOUT_RATE", 0.3))
NUM_CLASSES: int = 2

# ── Training settings ─────────────────────────────────────────────────────────
EPOCHS: int = int(os.getenv("EPOCHS", 3))
BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", 16))
LEARNING_RATE: float = float(os.getenv("LEARNING_RATE", 2e-5))
SEED: int = int(os.getenv("SEED", 42))
NUM_WORKERS: int = int(os.getenv("NUM_WORKERS", 0))

# ── Inference thresholds ──────────────────────────────────────────────────────
HIGH_RISK_THRESHOLD: float = float(os.getenv("HIGH_RISK_THRESHOLD", 0.7))
MODERATE_RISK_THRESHOLD: float = float(os.getenv("MODERATE_RISK_THRESHOLD", 0.4))
