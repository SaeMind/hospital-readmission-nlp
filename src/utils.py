"""
utils.py
Shared helper functions: seeding, output paths, metric formatting, JSON I/O.
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def select_device() -> str:
    """
    Select the best available compute device.

    Priority: CUDA (NVIDIA GPU) > MPS (Apple Silicon GPU) > CPU.
    PyTorch's `torch.cuda.is_available()` alone returns False on Apple
    Silicon Macs even when a GPU is present — MPS must be checked separately.

    Returns:
        Device string: 'cuda', 'mps', or 'cpu'.
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int = 42) -> None:
    """
    Fix random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Random seed set to %d", seed)


def timestamped_path(output_dir: Path, stem: str, suffix: str) -> Path:
    """
    Build an output file path with a UTC timestamp appended to the stem.

    Args:
        output_dir: Directory for the output file.
        stem: Base filename without extension (e.g. 'predictions').
        suffix: File extension including dot (e.g. '.json').

    Returns:
        Path object like output_dir/predictions_20250610_143022.json

    Example:
        >>> p = timestamped_path(Path("outputs"), "metrics", ".json")
        >>> p.parent.name
        'outputs'
    """
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{stem}_{ts}{suffix}"


def save_json(data: Any, path: Path) -> None:
    """
    Serialize data to a JSON file with 2-space indentation.

    Args:
        data: JSON-serializable object.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("Saved JSON to %s", path)


def load_json(path: Path) -> Any:
    """
    Load a JSON file.

    Args:
        path: Source file path.

    Returns:
        Parsed Python object.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_metrics(accuracy: float, f1: float, recall: float) -> Dict[str, str]:
    """
    Format evaluation metrics as percentage strings for display.

    Args:
        accuracy: Accuracy value (0–1).
        f1: F1 score (0–1).
        recall: Recall value (0–1).

    Returns:
        Dict with string-formatted percentages.
    """
    return {
        "accuracy": f"{accuracy:.2%}",
        "f1": f"{f1:.2%}",
        "recall": f"{recall:.2%}",
    }
