"""
data_loader.py
Load and tokenize MIMIC-IV discharge summaries for readmission prediction.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class MIMICDischargeDataset(Dataset):
    """
    PyTorch Dataset for MIMIC-IV discharge summaries.

    Expects a CSV with columns:
        - discharge_summary (str): Free-text discharge note
        - readmitted_30d (int): 1 if readmitted within 30 days, else 0
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer_name: str = "dmis-lab/biobert-base-cased-v1.1",
        max_length: int = 512,
        sample_size: Optional[int] = None,
    ) -> None:
        """
        Initialize dataset.

        Args:
            csv_path: Path to CSV file.
            tokenizer_name: HuggingFace model identifier for tokenizer.
            max_length: Maximum token length (BioBERT cap: 512).
            sample_size: If provided, randomly sample this many records (for testing).
        """
        self.df = pd.read_csv(csv_path)

        if sample_size:
            self.df = self.df.sample(n=min(sample_size, len(self.df)), random_state=42)

        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length

        n_pos = int(self.df["readmitted_30d"].sum())
        n_neg = len(self.df) - n_pos
        logger.info("Loaded %d records | Readmitted: %d | Not readmitted: %d", len(self.df), n_pos, n_neg)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return a single tokenized sample.

        Args:
            idx: Row index.

        Returns:
            Dict with keys: input_ids, attention_mask, token_type_ids, label.
        """
        row = self.df.iloc[idx]
        text: str = row["discharge_summary"]
        label: int = int(row["readmitted_30d"])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "token_type_ids": encoding["token_type_ids"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def create_data_loaders(
    train_path: str,
    val_path: str,
    batch_size: int = 16,
    max_length: int = 512,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders.

    Args:
        train_path: Path to training CSV.
        val_path: Path to validation CSV.
        batch_size: Samples per batch.
        max_length: Maximum token length.
        num_workers: Parallel workers for data loading (0 = main process).

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_dataset = MIMICDischargeDataset(train_path, max_length=max_length)
    val_dataset = MIMICDischargeDataset(val_path, max_length=max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))
    return train_loader, val_loader
