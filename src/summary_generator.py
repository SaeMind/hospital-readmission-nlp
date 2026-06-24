"""
summary_generator.py
Rule-based risk factor extraction and human-readable report generation.
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger(__name__)

RISK_PATTERNS: Dict[str, Dict[str, str]] = {
    "comorbidities": {
        "diabetes": r"diabetes|dm\b|hba1c",
        "heart_failure": r"chf|congestive heart failure|heart failure",
        "coronary_artery_disease": r"\bcad\b|coronary artery disease|ischemic heart",
        "chronic_kidney_disease": r"\bckd\b|chronic kidney|esrd|dialysis",
        "pneumonia": r"pneumonia|\bpna\b|respiratory infection",
        "sepsis": r"sepsis|septic|\bsirs\b",
        "copd": r"copd|chronic obstructive pulmonary",
        "atrial_fibrillation": r"atrial fibrillation|afib|a-fib",
    },
    "social_factors": {
        "homelessness": r"homeless|housing insecurity",
        "substance_use": r"alcohol use|drug use|substance use|etoh",
        "poor_medication_compliance": r"non-complian|noncomplian|unreliable|does not take.*med",
        "limited_social_support": r"lives alone|no family|no support|no caregiver",
    },
    "clinical_factors": {
        "acute_illness": r"\bacute\b|critical|severe",
        "polypharmacy": r"on \d+ medications|polypharmacy",
        "cognitive_impairment": r"dementia|alzheimer|confusion|delirium",
        "recent_surgery": r"post-op|post-operative|post operative|s/p surgery",
        "no_follow_up_arranged": r"no follow.?up|follow-up not|unable to schedule",
        "age_over_75": r"(?:7[5-9]|8\d|9\d)[- ]?(?:year|yo|y\.o\.)",
    },
}


class RiskFactorExtractor:
    """
    Extract structured risk factors from discharge summary text using regex patterns.

    Returns a dict mapping category to list of matched factor names.
    """

    def extract_risk_factors(self, discharge_summary: str) -> Dict[str, List[str]]:
        """
        Scan discharge summary for known risk factor patterns.

        Args:
            discharge_summary: Preprocessed clinical note text.

        Returns:
            Dict mapping category (str) to list of matched factor names (List[str]).

        Example:
            >>> extractor = RiskFactorExtractor()
            >>> extractor.extract_risk_factors("Patient has CHF and lives alone.")
            {'comorbidities': ['heart_failure'], 'social_factors': ['limited_social_support'], ...}
        """
        result: Dict[str, List[str]] = {}
        for category, patterns in RISK_PATTERNS.items():
            hits = [
                factor
                for factor, pattern in patterns.items()
                if re.search(pattern, discharge_summary, re.IGNORECASE)
            ]
            result[category] = hits
        return result

    def total_risk_count(self, risk_factors: Dict[str, List[str]]) -> int:
        """
        Sum all detected risk factors across categories.

        Args:
            risk_factors: Output of extract_risk_factors().

        Returns:
            Total count of detected factors (int).
        """
        return sum(len(v) for v in risk_factors.values())


class RiskSummaryGenerator:
    """
    Generate structured text reports from model predictions and extracted risk factors.
    """

    def generate(
        self,
        readmission_risk: float,
        risk_factors: Dict[str, List[str]],
    ) -> str:
        """
        Build a human-readable readmission risk assessment report.

        Args:
            readmission_risk: Predicted probability (0–1) from ReadmissionPredictor.
            risk_factors: Structured risk factors from RiskFactorExtractor.

        Returns:
            Formatted multi-line string suitable for clinical review or logging.
        """
        if readmission_risk > 0.7:
            risk_level = "HIGH"
        elif readmission_risk > 0.4:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"

        lines = [
            "READMISSION RISK ASSESSMENT",
            "=" * 44,
            f"Risk Level  : {risk_level}",
            f"Risk Score  : {readmission_risk:.1%}",
            "",
            "KEY RISK FACTORS IDENTIFIED:",
        ]

        for category, factors in risk_factors.items():
            if factors:
                lines.append(f"\n  {category.replace('_', ' ').title()}:")
                for factor in factors:
                    lines.append(f"    - {factor.replace('_', ' ').title()}")

        lines += ["", "RECOMMENDED ACTIONS:"]
        if readmission_risk > 0.7:
            lines += [
                "  - Initiate intensive discharge planning",
                "  - Schedule follow-up within 7 days",
                "  - Refer to home health or transitional care program",
                "  - Reconcile all medications before discharge",
            ]
        elif readmission_risk > 0.4:
            lines += [
                "  - Standard discharge planning with follow-up within 14 days",
                "  - Provide patient education on warning signs",
            ]
        else:
            lines += [
                "  - Standard discharge process appropriate",
                "  - Routine follow-up per clinical judgment",
            ]

        return "\n".join(lines)
