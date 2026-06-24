"""
text_preprocessor.py
Clean and normalize clinical discharge summaries before model ingestion.
"""

import logging
import re
from typing import Dict

logger = logging.getLogger(__name__)


class ClinicalTextPreprocessor:
    """
    Preprocess discharge summaries for NLP.

    Steps applied in order:
        1. PHI redaction (dates, MRN, phone, email)
        2. Abbreviation expansion
        3. Whitespace normalization and ASCII coercion
    """

    PHI_PATTERNS: Dict[str, str] = {
        "date": r"\d{1,2}/\d{1,2}/\d{2,4}",
        "mrn": r"MRN[:\s]+\d{7,9}",
        "age": r"Age[:\s]+\d{1,3}",
        "email": r"\S+@\S+",
        "phone": r"\d{3}-\d{3}-\d{4}",
        "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    }

    ABBREVIATIONS: Dict[str, str] = {
        "HTN": "hypertension",
        "DM": "diabetes mellitus",
        "CAD": "coronary artery disease",
        "CHF": "congestive heart failure",
        "AFib": "atrial fibrillation",
        "COPD": "chronic obstructive pulmonary disease",
        "PNA": "pneumonia",
        "MI": "myocardial infarction",
        "CKD": "chronic kidney disease",
        "ESRD": "end-stage renal disease",
        "HF": "heart failure",
        "AKI": "acute kidney injury",
        "CVA": "cerebrovascular accident",
        "TIA": "transient ischemic attack",
    }

    def remove_phi(self, text: str) -> str:
        """
        Redact Protected Health Information.

        Args:
            text: Raw clinical text.

        Returns:
            Text with PHI replaced by placeholder tokens (e.g. [DATE]).
        """
        for label, pattern in self.PHI_PATTERNS.items():
            text = re.sub(pattern, f"[{label.upper()}]", text, flags=re.IGNORECASE)
        return text

    def normalize_abbreviations(self, text: str) -> str:
        """
        Expand common clinical abbreviations to full terms.

        Args:
            text: Text with abbreviations.

        Returns:
            Text with expansions applied.
        """
        for abbrev, expansion in self.ABBREVIATIONS.items():
            text = re.sub(r"\b" + re.escape(abbrev) + r"\b", expansion, text)
        return text

    def clean_text(self, text: str) -> str:
        """
        Normalize whitespace, remove non-ASCII characters and control characters.

        Args:
            text: Text to clean.

        Returns:
            Cleaned text string.
        """
        # Collapse multiple whitespace
        text = re.sub(r"\s+", " ", text)
        # Strip separator lines (e.g. __________)
        text = re.sub(r"_{5,}", "", text)
        # Drop non-ASCII bytes
        text = text.encode("ascii", "ignore").decode("ascii")
        # Remove control characters except newline and tab
        text = "".join(ch for ch in text if ord(ch) >= 32 or ch in "\n\t")
        return text.strip()

    def preprocess(self, text: str) -> str:
        """
        Run the full preprocessing pipeline.

        Args:
            text: Raw discharge summary string.

        Returns:
            Cleaned, normalized text ready for tokenization.

        Example:
            >>> pp = ClinicalTextPreprocessor()
            >>> pp.preprocess("Pt has HTN and DM. DOB: 01/01/1950.")
            'Pt has hypertension and diabetes mellitus. DOB: [DATE].'
        """
        text = self.remove_phi(text)
        text = self.normalize_abbreviations(text)
        text = self.clean_text(text)
        return text
