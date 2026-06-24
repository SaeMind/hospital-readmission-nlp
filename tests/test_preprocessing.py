"""
tests/test_preprocessing.py
Unit tests for text_preprocessor.py and summary_generator.py.

These modules have no torch/transformers dependency, so this suite runs
fast and is suitable for CI without a GPU or model download.

Usage:
    pytest tests/test_preprocessing.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from text_preprocessor import ClinicalTextPreprocessor
from summary_generator import RiskFactorExtractor, RiskSummaryGenerator


# ── ClinicalTextPreprocessor ────────────────────────────────────────────────────

class TestClinicalTextPreprocessor:
    def setup_method(self) -> None:
        self.pp = ClinicalTextPreprocessor()

    def test_remove_phi_redacts_date(self) -> None:
        text = "DOB: 01/15/1950."
        result = self.pp.remove_phi(text)
        assert "01/15/1950" not in result
        assert "[DATE]" in result

    def test_remove_phi_redacts_mrn(self) -> None:
        text = "MRN: 1234567"
        result = self.pp.remove_phi(text)
        assert "1234567" not in result

    def test_normalize_abbreviations_expands_htn(self) -> None:
        result = self.pp.normalize_abbreviations("Patient has HTN.")
        assert "hypertension" in result
        assert "HTN" not in result

    def test_normalize_abbreviations_word_boundary(self) -> None:
        # "MI" should not match inside "MIDDLE"
        result = self.pp.normalize_abbreviations("Patient lives on MIDDLE street.")
        assert "myocardial infarction" not in result

    def test_clean_text_collapses_whitespace(self) -> None:
        result = self.pp.clean_text("Too    many     spaces")
        assert "  " not in result

    def test_preprocess_full_pipeline(self) -> None:
        text = "Pt has HTN and DM. DOB: 01/01/1950."
        result = self.pp.preprocess(text)
        assert "hypertension" in result
        assert "diabetes mellitus" in result
        assert "01/01/1950" not in result


# ── RiskFactorExtractor ──────────────────────────────────────────────────────────

class TestRiskFactorExtractor:
    def setup_method(self) -> None:
        self.extractor = RiskFactorExtractor()

    def test_detects_heart_failure(self) -> None:
        result = self.extractor.extract_risk_factors("Patient has congestive heart failure.")
        assert "heart_failure" in result["comorbidities"]

    def test_detects_social_isolation(self) -> None:
        result = self.extractor.extract_risk_factors("Patient lives alone.")
        assert "limited_social_support" in result["social_factors"]

    def test_no_false_positive_on_clean_note(self) -> None:
        result = self.extractor.extract_risk_factors(
            "Patient is healthy and was discharged in good condition."
        )
        total = self.extractor.total_risk_count(result)
        assert total == 0

    def test_total_risk_count(self) -> None:
        result = self.extractor.extract_risk_factors(
            "Patient has heart failure and lives alone with no follow-up arranged."
        )
        assert self.extractor.total_risk_count(result) >= 3


# ── RiskSummaryGenerator ─────────────────────────────────────────────────────────

class TestRiskSummaryGenerator:
    def setup_method(self) -> None:
        self.generator = RiskSummaryGenerator()

    def test_high_risk_label(self) -> None:
        report = self.generator.generate(0.85, {"comorbidities": ["heart_failure"]})
        assert "HIGH" in report

    def test_low_risk_label(self) -> None:
        report = self.generator.generate(0.1, {"comorbidities": []})
        assert "LOW" in report

    def test_moderate_risk_label(self) -> None:
        report = self.generator.generate(0.5, {"comorbidities": []})
        assert "MODERATE" in report

    def test_high_risk_includes_recommended_actions(self) -> None:
        report = self.generator.generate(0.9, {})
        assert "follow-up within 7 days" in report.lower()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
