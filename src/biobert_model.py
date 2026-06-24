"""
biobert_model.py
BioBERT-based classifier and trainer for 30-day readmission prediction.
"""

import logging
from typing import Tuple

import torch
import torch.nn as nn
from transformers import BertModel
from tqdm import tqdm

from utils import select_device

logger = logging.getLogger(__name__)


class ReadmissionBioBERT(nn.Module):
    """
    BioBERT encoder with a two-layer classification head.

    Architecture:
        BioBERT → [CLS] pooling → Dropout → Dense(768→256) → ReLU
        → Dropout → Linear(256→2) → logits
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        hidden_size: int = 768,
        dropout_rate: float = 0.3,
        num_classes: int = 2,
    ) -> None:
        """
        Initialize model.

        Args:
            model_name: HuggingFace model identifier.
            hidden_size: BERT hidden dimension (768 for base models).
            dropout_rate: Dropout probability applied before each linear layer.
            num_classes: Number of output classes (2: readmitted / not).
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.dense = nn.Linear(hidden_size, 256)
        self.activation = nn.ReLU()
        self.output = nn.Linear(256, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token IDs tensor of shape (batch_size, seq_length).
            attention_mask: Attention mask of shape (batch_size, seq_length).

        Returns:
            Logits tensor of shape (batch_size, num_classes).
        """
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = bert_out.last_hidden_state[:, 0, :]  # [CLS] token
        x = self.dropout(cls)
        x = self.activation(self.dense(x))
        x = self.dropout(x)
        return self.output(x)


class ReadmissionTrainer:
    """
    Manages training and evaluation loops for ReadmissionBioBERT.
    """

    def __init__(
        self,
        model: ReadmissionBioBERT,
        device: str = select_device(),
        learning_rate: float = 2e-5,
    ) -> None:
        """
        Initialize trainer.

        Args:
            model: Instantiated ReadmissionBioBERT model.
            device: Compute device ('cuda' or 'cpu').
            learning_rate: AdamW learning rate (2e-5 is standard for BERT fine-tuning).
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.loss_fn = nn.CrossEntropyLoss()
        logger.info("Trainer initialized on device: %s", device)

    def train_epoch(self, train_loader) -> float:
        """
        Run one full training epoch.

        Args:
            train_loader: DataLoader yielding batches with input_ids, attention_mask, label.

        Returns:
            Mean cross-entropy loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0

        progress = tqdm(train_loader, desc="Training", unit="batch")
        for batch in progress:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["label"].to(self.device)

            logits = self.model(input_ids, attention_mask)
            loss = self.loss_fn(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info("Epoch training loss: %.4f", avg_loss)
        return avg_loss

    def evaluate(self, val_loader) -> Tuple[float, float, float]:
        """
        Evaluate model on a validation set.

        Args:
            val_loader: DataLoader for validation data.

        Returns:
            Tuple of (accuracy, F1, recall).
        """
        self.model.eval()
        correct = total = tp = fp = fn = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", unit="batch"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        accuracy = correct / total
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)

        logger.info("Val accuracy: %.4f | F1: %.4f | Recall: %.4f", accuracy, f1, recall)
        return accuracy, f1, recall
