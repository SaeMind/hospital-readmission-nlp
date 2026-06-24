"""
readmission_classifier.py
Inference wrapper: load fine-tuned model and predict readmission risk.
"""

import logging
from typing import Dict, List

import torch
from transformers import BertTokenizer

from biobert_model import ReadmissionBioBERT

logger = logging.getLogger(__name__)


class ReadmissionPredictor:
    """
    Load a fine-tuned ReadmissionBioBERT checkpoint and run inference.

    Outputs per note:
        - readmission_risk (float 0–1)
        - high_risk (bool, threshold 0.5)
        - confidence (float, max softmax probability)
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        """
        Load model and tokenizer.

        Args:
            model_path: Path to saved state_dict (.pt file).
            device: Compute device ('cuda' or 'cpu').
        """
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
        self.model = ReadmissionBioBERT()
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        logger.info("Model loaded from %s on %s", model_path, device)

    def predict(self, discharge_summary: str) -> Dict:
        """
        Predict readmission risk for a single discharge note.

        Args:
            discharge_summary: Free-text discharge summary string.

        Returns:
            Dict with keys:
                - readmission_risk: float (0–1)
                - high_risk: bool
                - confidence: float

        Example:
            >>> predictor = ReadmissionPredictor("models/readmission_finetuned/model.pt")
            >>> predictor.predict("76-year-old male with CHF, discharged home alone...")
            {'readmission_risk': 0.78, 'high_risk': True, 'confidence': 0.89}
        """
        encoding = self.tokenizer(
            discharge_summary,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=1)[0]

        readmitted_prob = probs[1].item()
        confidence = probs.max().item()

        return {
            "readmission_risk": round(readmitted_prob, 4),
            "high_risk": readmitted_prob > 0.5,
            "confidence": round(confidence, 4),
        }

    def batch_predict(self, summaries: List[str]) -> List[Dict]:
        """
        Run prediction over a list of discharge summaries.

        Args:
            summaries: List of discharge summary strings.

        Returns:
            List of prediction dicts (same schema as predict()).
        """
        return [self.predict(s) for s in summaries]
