"""
Tests for Form Assessment Module
=================================

Tests form checking functions, feedback generation, and safety thresholds.

Run with:
    pytest tests/test_form_assessment.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.form_assessment import (
    FormAnalyzer,
    FormStatus,
    FeedbackPriority,
    FormFeedback,
    FormCheckResult,
    OverallFormAssessment,
    get_form_feedback_text,
    should_stop_exercise,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def form_analyzer():
    """Create a FormAnalyzer instance for testing."""
    return FormAnalyzer(exercise_type="knee_extension")


@pytest.fixture
def good_form_keypoints():
    """Keypoints representing good form for knee extension."""
    keypoints = np.zeros((17, 3))

    # Upper body stable
    keypoints[5] = [120, 120, 0.95]  # Left shoulder
    keypoints[6] = [180, 120, 0.95]  # Right shoulder

    # Hips level and stable (seated)
    keypoints[11] = [130, 280, 0.95]  # Left hip
    keypoints[12] = [170, 280, 0.95]  # Right hip

    # Knees
    keypoints[13] = [130, 400, 0.9]  # Left knee
    keypoints[14] = [170, 400, 0.9]  # Right knee

    # Ankles (leg extended)
    keypoints[15] = [130, 520, 0.85]  # Left ankle
    keypoints[16] = [170, 520, 0.85]  # Right ankle

    # Head/face (for shrug detection)
    keypoints[3] = [130, 100, 0.9]   # Left ear

    return keypoints


@pytest.fixture
def bad_form_keypoints():
    """Keypoints representing poor form - hip lifting and trunk lean."""
    keypoints = np.zeros((17, 3))

    # Upper body leaning (shoulders shifted forward)
    keypoints[5] = [160, 100, 0.95]  # Left shoulder (shifted forward)
    keypoints[6] = [220, 100, 0.95]  # Right shoulder

    # Hips uneven (hip hiking)
    keypoints[11] = [130, 260, 0.95]  # Left hip (raised)
    keypoints[12] = [170, 290, 0.95]  # Right hip (lower)

    # Knees
    keypoints[13] = [130, 400, 0.9]
    keypoints[14] = [170, 400, 0.9]

    # Ankles
    keypoints[15] = [130, 520, 0.85]
    keypoints[16] = [170, 520, 0.85]

    # Head (shoulder shrugging - ear close to shoulder)
    keypoints[3] = [130, 110, 0.9]  # Left ear (close to shoulder)

    return keypoints


@pytest.fixture
def asymmetric_keypoints():
    """Keypoints showing left/right asymmetry."""
    keypoints = np.zeros((17, 3))

    keypoints[5] = [120, 120, 0.95]
    keypoints[6] = [180, 120, 0.95]

    keypoints[11] = [130, 280, 0.95]
    keypoints[12] = [170, 280, 0.95]

    # Left knee more bent than right
    keypoints[13] = [150, 380, 0.9]  # Left knee (bent)
    keypoints[14] = [170, 400, 0.9]  # Right knee (straighter)

    keypoints[15] = [200, 400, 0.85]  # Left ankle (forming angle)
    keypoints[16] = [170, 520, 0.85]  # Right ankle (straighter)

    return keypoints


@pytest.fixture
def squat_knee_valgus_keypoints():
    """Keypoints showing knee valgus (knees caving inward) during squat."""
    keypoints = np.zeros((17, 3))

    # Upper body
    keypoints[5] = [130, 120, 0.95]
    keypoints[6] = [170, 120, 0.95]

    # Hips
    keypoints[11] = [120, 280, 0.95]  # Left hip
    keypoints[12] = [180, 280, 0.95]  # Right hip

    # Knees collapsed inward (valgus)
    keypoints[13] = [145, 400, 0.9]  # Left knee (inside left hip/ankle)
    keypoints[14] = [155, 400, 0.9]  # Right knee (inside right hip/ankle)

    # Ankles wider than knees
    keypoints[15] = [110, 520, 0.85]  # Left ankle
    keypoints[16] = [190, 520, 0.85]  # Right ankle

    return keypoints


# =============================================================================
# FORM ANALYZER TESTS
# =============================================================================

class TestFormAnalyzer:
    """Tests for the FormAnalyzer class."""

    def test_analyzer_initialization(self, form_analyzer):
        """Test that analyzer initializes correctly."""
        assert form_analyzer.exercise_type == "knee_extension"
        assert form_analyzer.frame_count == 0
        assert len(form_analyzer.rep_form_scores) == 0

    def test_analyze_good_form(self, form_analyzer, good_form_keypoints):
        """Good form should result in high score."""
        assessment = form_analyzer.analyze_frame(good_form_keypoints, frame_number=1)

        assert assessment is not None
        assert isinstance(assessment, OverallFormAssessment)
        assert assessment.overall_score >= 70, f"Good form should score >=70, got {assessment.overall_score}"
        assert assessment.status in [FormStatus.EXCELLENT, FormStatus.GOOD]

    def test_analyze_bad_form(self, form_analyzer, bad_form_keypoints):
        """Bad form should result in lower score and feedback."""
        # Need to set baseline first
        form_analyzer.baseline_hip_height = 280

        assessment = form_analyzer.analyze_frame(bad_form_keypoints, frame_number=1)

        assert assessment is not None
        assert assessment.overall_score < 90  # Should be lower than perfect
        assert len(assessment.feedback_messages) > 0  # Should have feedback

    def test_analyze_none_keypoints(self, form_analyzer):
        """Should handle None keypoints gracefully."""
        assessment = form_analyzer.analyze_frame(None, frame_number=1)

        assert assessment is not None
        # Should still return an assessment, just with default/low scores

    def test_reset_analyzer(self, form_analyzer, good_form_keypoints):
        """Reset should clear all history."""
        # Process some frames
        for i in range(5):
            form_analyzer.analyze_frame(good_form_keypoints, frame_number=i)

        assert form_analyzer.frame_count == 4  # Last frame number
        assert len(form_analyzer.form_score_history) > 0

        # Reset
        form_analyzer.reset()

        assert form_analyzer.frame_count == 0
        assert len(form_analyzer.form_score_history) == 0
        assert form_analyzer.baseline_trunk_angle is None


class TestAlignmentAssessment:
    """Tests for the assess_alignment method."""

    def test_good_alignment(self, form_analyzer, good_form_keypoints):
        """Proper alignment should pass."""
        result = form_analyzer.assess_alignment(good_form_keypoints)

        assert result.check_name == "alignment"
        assert result.score >= 70
        assert result.passed

    def test_uneven_shoulders(self, form_analyzer):
        """Uneven shoulders should be detected."""
        keypoints = np.zeros((17, 3))
        keypoints[5] = [120, 100, 0.95]  # Left shoulder higher
        keypoints[6] = [180, 140, 0.95]  # Right shoulder lower
        keypoints[11] = [130, 280, 0.95]
        keypoints[12] = [170, 280, 0.95]

        result = form_analyzer.assess_alignment(keypoints)

        assert result.score < 100
        # Check that shoulder issue was detected
        assert "shoulder" in str(result.details.get("issues", [])).lower() or \
               result.details.get("shoulder_level") is not None

    def test_uneven_hips(self, form_analyzer):
        """Uneven hips should be detected."""
        keypoints = np.zeros((17, 3))
        keypoints[5] = [120, 120, 0.95]
        keypoints[6] = [180, 120, 0.95]
        keypoints[11] = [130, 260, 0.95]  # Left hip higher
        keypoints[12] = [170, 300, 0.95]  # Right hip lower

        result = form_analyzer.assess_alignment(keypoints)

        # Should detect hip issue
        assert result.details.get("hip_level") is not None


class TestCompensationDetection:
    """Tests for the detect_compensation method."""

    def test_no_compensation_detected(self, form_analyzer, good_form_keypoints):
        """Good form should not show compensation."""
        # Need history for compensation detection
        for i in range(10):
            form_analyzer.analyze_frame(good_form_keypoints, frame_number=i)

        result = form_analyzer.detect_compensation(good_form_keypoints)

        assert "compensations" in result.details
        # May have empty list or no critical compensations

    def test_compensation_needs_history(self, form_analyzer, bad_form_keypoints):
        """Compensation detection needs frame history."""
        # Without history, should return default result
        result = form_analyzer.detect_compensation(bad_form_keypoints)

        assert result.check_name == "compensation"
        # Should note insufficient data
        assert "Insufficient data" in result.details.get("message", "") or result.passed


class TestSymmetryCalculation:
    """Tests for the calculate_symmetry method."""

    def test_symmetric_pose(self, form_analyzer, good_form_keypoints):
        """Symmetric pose should score high."""
        result = form_analyzer.calculate_symmetry(good_form_keypoints)

        assert result.check_name == "symmetry"
        assert result.score >= 70

    def test_asymmetric_pose(self, form_analyzer, asymmetric_keypoints):
        """Asymmetric pose should be detected."""
        result = form_analyzer.calculate_symmetry(asymmetric_keypoints)

        # Should detect asymmetry
        assert "asymmetries" in result.details
        # Score should reflect asymmetry

    def test_symmetry_none_keypoints(self, form_analyzer):
        """Should handle None keypoints."""
        result = form_analyzer.calculate_symmetry(None)

        assert result.passed  # Defaults to passing when can't assess


class TestFatigueTracking:
    """Tests for the track_fatigue method."""

    def test_no_fatigue_few_reps(self, form_analyzer):
        """Insufficient reps should not indicate fatigue."""
        form_analyzer.rep_form_scores = [85, 82]  # Only 2 reps

        result = form_analyzer.track_fatigue()

        assert result.check_name == "fatigue"
        assert "Need more reps" in result.details.get("message", "")

    def test_fatigue_declining_scores(self, form_analyzer):
        """Declining scores should indicate fatigue."""
        # Simulate declining form over reps
        form_analyzer.rep_form_scores = [90, 88, 85, 80, 75, 70, 65]

        result = form_analyzer.track_fatigue()

        assert result.check_name == "fatigue"
        assert result.details.get("decline", 0) > 0

    def test_no_fatigue_stable_scores(self, form_analyzer):
        """Stable scores should not indicate fatigue."""
        form_analyzer.rep_form_scores = [85, 84, 86, 85, 84, 85, 86]

        result = form_analyzer.track_fatigue()

        assert result.score >= 70  # Should be good


# =============================================================================
# EXERCISE-SPECIFIC FORM CHECKS
# =============================================================================

class TestKneeExtensionFormChecks:
    """Tests for knee extension specific checks."""

    def test_hip_stability_check(self):
        """Test hip stability during knee extension."""
        analyzer = FormAnalyzer(exercise_type="knee_extension")
        analyzer.baseline_hip_height = 280

        # Keypoints with hip lifted
        keypoints = np.zeros((17, 3))
        keypoints[5] = [120, 120, 0.95]
        keypoints[6] = [180, 120, 0.95]
        keypoints[11] = [130, 240, 0.95]  # Hip lifted (y decreased)
        keypoints[12] = [170, 280, 0.95]
        keypoints[13] = [130, 400, 0.9]
        keypoints[14] = [170, 400, 0.9]
        keypoints[15] = [130, 520, 0.85]
        keypoints[16] = [170, 520, 0.85]

        results = analyzer._check_knee_extension_form(keypoints)

        # Should have hip stability check
        hip_checks = [r for r in results if r.check_name == "hip_stability"]
        assert len(hip_checks) > 0

        # Hip lifting should fail the check
        if hip_checks:
            assert not hip_checks[0].passed


class TestSquatFormChecks:
    """Tests for squat specific checks."""

    def test_knee_valgus_detection(self, squat_knee_valgus_keypoints):
        """Should detect knee valgus (knees caving in)."""
        analyzer = FormAnalyzer(exercise_type="squat")

        results = analyzer._check_squat_form(squat_knee_valgus_keypoints)

        # Should have valgus check
        valgus_checks = [r for r in results if r.check_name == "knee_valgus"]
        assert len(valgus_checks) > 0

        # Should fail due to valgus
        if valgus_checks:
            assert not valgus_checks[0].passed


class TestPushupFormChecks:
    """Tests for push-up specific checks."""

    def test_hip_sag_detection(self):
        """Should detect hip sagging."""
        analyzer = FormAnalyzer(exercise_type="pushup")

        # Keypoints with hip sagging (body not straight)
        keypoints = np.zeros((17, 3))
        keypoints[5] = [100, 100, 0.95]   # Shoulder
        keypoints[11] = [200, 200, 0.95]  # Hip lower/forward
        keypoints[15] = [300, 100, 0.85]  # Ankle

        results = analyzer._check_pushup_form(keypoints)

        # Should have body alignment check
        alignment_checks = [r for r in results if r.check_name == "body_alignment"]
        assert len(alignment_checks) > 0


# =============================================================================
# FEEDBACK AND STATUS TESTS
# =============================================================================

class TestFormFeedback:
    """Tests for FormFeedback data class."""

    def test_feedback_creation(self):
        """Test creating feedback objects."""
        feedback = FormFeedback(
            message="Keep your back straight",
            priority=FeedbackPriority.WARNING,
            body_part="trunk",
            frame_number=42
        )

        assert feedback.message == "Keep your back straight"
        assert feedback.priority == FeedbackPriority.WARNING
        assert feedback.body_part == "trunk"
        assert feedback.frame_number == 42

    def test_feedback_priority_ordering(self):
        """Test that priorities can be compared."""
        assert FeedbackPriority.CRITICAL.value > FeedbackPriority.WARNING.value
        assert FeedbackPriority.WARNING.value > FeedbackPriority.SUGGESTION.value
        assert FeedbackPriority.SUGGESTION.value > FeedbackPriority.INFO.value


class TestFormStatus:
    """Tests for FormStatus enum."""

    def test_status_values(self):
        """Test that all status values exist."""
        assert FormStatus.EXCELLENT.value == "excellent"
        assert FormStatus.GOOD.value == "good"
        assert FormStatus.WARNING.value == "warning"
        assert FormStatus.POOR.value == "poor"
        assert FormStatus.STOP.value == "stop"


class TestHelperFunctions:
    """Tests for helper functions."""

    def test_get_form_feedback_text(self, form_analyzer, good_form_keypoints):
        """Test feedback text generation."""
        assessment = form_analyzer.analyze_frame(good_form_keypoints)

        text = get_form_feedback_text(assessment)

        assert "Form Score:" in text
        assert str(assessment.status.value) in text.lower()

    def test_should_stop_exercise_true(self):
        """Should return True for STOP status."""
        assessment = OverallFormAssessment(
            overall_score=30,
            status=FormStatus.STOP
        )

        assert should_stop_exercise(assessment) is True

    def test_should_stop_exercise_false(self):
        """Should return False for non-STOP status."""
        assessment = OverallFormAssessment(
            overall_score=85,
            status=FormStatus.GOOD
        )

        assert should_stop_exercise(assessment) is False


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_partial_keypoints(self, form_analyzer):
        """Should handle partially visible body."""
        # Only upper body visible
        keypoints = np.zeros((17, 3))
        keypoints[5] = [120, 120, 0.95]  # Left shoulder
        keypoints[6] = [180, 120, 0.95]  # Right shoulder
        keypoints[11] = [130, 280, 0.95]  # Left hip
        keypoints[12] = [170, 280, 0.95]  # Right hip
        # No legs

        assessment = form_analyzer.analyze_frame(keypoints)

        # Should still return an assessment
        assert assessment is not None
        assert isinstance(assessment.overall_score, float)

    def test_all_low_confidence(self, form_analyzer):
        """Should handle all low confidence keypoints."""
        keypoints = np.zeros((17, 3))
        for i in range(17):
            keypoints[i] = [100, 100 + i * 20, 0.1]  # All low confidence

        assessment = form_analyzer.analyze_frame(keypoints)

        # Should handle gracefully
        assert assessment is not None

    def test_rapid_frame_updates(self, form_analyzer, good_form_keypoints):
        """Should handle rapid consecutive frames."""
        for i in range(100):
            assessment = form_analyzer.analyze_frame(good_form_keypoints, frame_number=i)
            assert assessment is not None

        assert form_analyzer.frame_count == 99


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
