import unittest

from utils.analysis_engine import build_candidate_insights, compute_grade


class AnalysisEngineTests(unittest.TestCase):
    def test_compute_grade_returns_expected_band(self):
        self.assertEqual(compute_grade(9.1), ("A+", "#34d399"))
        self.assertEqual(compute_grade(6.0), ("B", "#818cf8"))
        self.assertEqual(compute_grade(2.9), ("F", "#f87171"))

    def test_build_candidate_insights_highlights_strengths(self):
        strengths, concerns = build_candidate_insights(
            confidence_score=8.1,
            technical_score=7.8,
            eye_contact_pct=74.0,
            fluency_score=8.0,
            voice_score=7.2,
            posture_score=0.82,
        )

        self.assertTrue(strengths)
        self.assertTrue(concerns)
        self.assertTrue(any("Confidence" in item or "Answer relevance" in item for item in strengths))

    def test_build_candidate_insights_flags_weak_signals(self):
        strengths, concerns = build_candidate_insights(
            confidence_score=3.2,
            technical_score=4.1,
            eye_contact_pct=24.0,
            fluency_score=3.8,
            voice_score=4.0,
            posture_score=0.3,
        )

        self.assertTrue(strengths)
        self.assertTrue(concerns)
        self.assertTrue(any("weak" in item.lower() or "limited" in item.lower() for item in concerns))


if __name__ == "__main__":
    unittest.main()
