"""
Form Assessment Module for AI Physical Therapy Assistant
=========================================================

This module provides intelligent feedback on exercise form by analyzing
body positions and movements detected through pose estimation.

Key Features:
- Real-time form quality assessment
- Compensation pattern detection
- Symmetry analysis between left and right sides
- Fatigue tracking through form degradation
- Exercise-specific biomechanical checks

Biomechanical Principles:
-------------------------
1. Joint Stacking: Proper alignment reduces injury risk (e.g., knee over ankle)
2. Compensation: Using wrong muscles indicates weakness or pain avoidance
3. Symmetry: Left/right differences may indicate injury or imbalance
4. Fatigue: Form degradation over reps suggests muscle fatigue

Author: AI PT Assistant
"""

import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum
import time

from .utils import (
    extract_keypoint, calculate_angle, get_angle_from_keypoints,
    get_trunk_angle, get_shoulder_level_angle, get_hip_level_angle,
    get_hip_center, get_shoulder_center, get_keypoint_distance,
    get_form_color, KEYPOINT_NAMES
)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class FormStatus(Enum):
    """Overall form status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    POOR = "poor"
    STOP = "stop"  # Safety concern - should stop exercise


class FeedbackPriority(Enum):
    """Priority levels for feedback messages."""
    INFO = 1
    SUGGESTION = 2
    WARNING = 3
    CRITICAL = 4


@dataclass
class FormFeedback:
    """A single piece of form feedback."""
    message: str
    priority: FeedbackPriority
    body_part: str  # e.g., "knee", "shoulder", "trunk"
    timestamp: float = field(default_factory=time.time)
    frame_number: int = 0


@dataclass
class FormCheckResult:
    """Result of a single form check."""
    check_name: str
    passed: bool
    score: float  # 0-100
    feedback: Optional[FormFeedback] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OverallFormAssessment:
    """Complete form assessment for a frame or rep."""
    overall_score: float  # 0-100
    status: FormStatus
    checks: List[FormCheckResult] = field(default_factory=list)
    feedback_messages: List[FormFeedback] = field(default_factory=list)
    skeleton_colors: Dict[str, Tuple[int, int, int]] = field(default_factory=dict)


# =============================================================================
# SAFETY THRESHOLDS
# =============================================================================
# These are based on PT literature for safe exercise ranges

SAFETY_THRESHOLDS = {
    "knee_hyperextension": 185,  # Degrees - stop if knee goes past this
    "knee_valgus_angle": 15,     # Degrees of inward collapse
    "trunk_lean_max": 30,        # Degrees from vertical during standing exercises
    "shoulder_elevation": 175,   # Max safe shoulder flexion
    "hip_flexion_seated_max": 130,  # Max hip flexion when seated
    "lumbar_flexion_warning": 45,   # Degrees of forward bend
}


# =============================================================================
# FORM ANALYZER CLASS
# =============================================================================

class FormAnalyzer:
    """
    Analyzes exercise form using pose estimation data.

    This class provides methods for:
    - Real-time form assessment during exercise
    - Detecting compensation patterns
    - Tracking form quality over time
    - Generating appropriate feedback

    Usage:
        analyzer = FormAnalyzer(exercise_type="knee_extension")
        for frame_keypoints in video_frames:
            assessment = analyzer.analyze_frame(frame_keypoints)
            print(assessment.feedback_messages)
    """

    def __init__(self, exercise_type: str = "general",
                 history_length: int = 30):
        """
        Initialize the FormAnalyzer.

        Args:
            exercise_type: Type of exercise for specific checks
            history_length: Number of frames to keep for temporal analysis
        """
        self.exercise_type = exercise_type
        self.history_length = history_length

        # Historical data for temporal analysis
        self.angle_history: deque = deque(maxlen=history_length)
        self.form_score_history: deque = deque(maxlen=history_length)
        self.keypoint_history: deque = deque(maxlen=history_length)

        # Rep-level tracking
        self.rep_form_scores: List[float] = []
        self.current_rep_scores: List[float] = []

        # Frame counter
        self.frame_count = 0

        # Baseline measurements (set during first few frames)
        self.baseline_trunk_angle: Optional[float] = None
        self.baseline_hip_height: Optional[float] = None

        # Map exercise types to their specific check functions
        self.exercise_checks = {
            "knee_extension": self._check_knee_extension_form,
            "shoulder_flexion": self._check_shoulder_flexion_form,
            "squat": self._check_squat_form,
            "pushup": self._check_pushup_form,
            "hip_abduction": self._check_hip_abduction_form,
            "general": self._check_general_form,
        }

    def analyze_frame(self, keypoints: np.ndarray,
                      frame_number: int = 0) -> OverallFormAssessment:
        """
        Perform complete form analysis on a single frame.

        This is the main entry point for form assessment. It runs all
        relevant checks and aggregates the results.

        Args:
            keypoints: YOLO pose keypoints for the frame
            frame_number: Current frame number (for timestamps)

        Returns:
            OverallFormAssessment with scores, status, and feedback
        """
        self.frame_count = frame_number
        checks = []
        feedback_messages = []
        skeleton_colors = {}

        # Store keypoints in history
        if keypoints is not None:
            self.keypoint_history.append(keypoints.copy())

        # Set baseline measurements if not set
        if self.frame_count < 10 and keypoints is not None:
            self._update_baseline(keypoints)

        # Run general alignment checks
        alignment_result = self.assess_alignment(keypoints)
        checks.append(alignment_result)
        if alignment_result.feedback:
            feedback_messages.append(alignment_result.feedback)

        # Run exercise-specific checks
        exercise_check_func = self.exercise_checks.get(
            self.exercise_type, self._check_general_form
        )
        exercise_results = exercise_check_func(keypoints)
        checks.extend(exercise_results)

        for result in exercise_results:
            if result.feedback:
                feedback_messages.append(result.feedback)

        # Run compensation detection
        compensation_result = self.detect_compensation(keypoints)
        checks.append(compensation_result)
        if compensation_result.feedback:
            feedback_messages.append(compensation_result.feedback)

        # Run symmetry analysis
        symmetry_result = self.calculate_symmetry(keypoints)
        checks.append(symmetry_result)
        if symmetry_result.feedback:
            feedback_messages.append(symmetry_result.feedback)

        # Calculate overall score
        valid_scores = [c.score for c in checks if c.score >= 0]
        overall_score = np.mean(valid_scores) if valid_scores else 50.0

        # Update history
        self.form_score_history.append(overall_score)
        self.current_rep_scores.append(overall_score)

        # Determine status
        status = self._score_to_status(overall_score, checks)

        # Determine skeleton colors for visualization
        skeleton_colors = self._get_skeleton_colors(checks, keypoints)

        # Sort feedback by priority
        feedback_messages.sort(key=lambda f: f.priority.value, reverse=True)

        return OverallFormAssessment(
            overall_score=round(overall_score, 1),
            status=status,
            checks=checks,
            feedback_messages=feedback_messages[:3],  # Top 3 most important
            skeleton_colors=skeleton_colors
        )

    def assess_alignment(self, keypoints: np.ndarray) -> FormCheckResult:
        """
        Check if joints are properly stacked/aligned.

        Proper Alignment Principles:
        ----------------------------
        - Head over shoulders over hips (standing posture)
        - Knees tracking over toes (squats, lunges)
        - Shoulders level (not hiking one side)
        - Hips level (not dropping one side)

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            FormCheckResult with alignment assessment
        """
        if keypoints is None:
            return FormCheckResult(
                check_name="alignment",
                passed=False,
                score=0,
                details={"error": "No keypoints detected"}
            )

        issues = []
        score = 100.0

        # Check trunk alignment (should be relatively vertical for most exercises)
        trunk_angle = get_trunk_angle(keypoints)
        if trunk_angle is not None:
            if trunk_angle > 25:
                issues.append(f"Trunk leaning {trunk_angle:.0f}° from vertical")
                score -= min(trunk_angle - 15, 30)

        # Check shoulder level
        shoulder_level = get_shoulder_level_angle(keypoints)
        if shoulder_level is not None:
            if abs(shoulder_level) > 10:  # More than 10° from horizontal
                deviation = abs(shoulder_level)
                issues.append(f"Shoulders uneven by {deviation:.0f}°")
                score -= min(deviation, 20)

        # Check hip level
        hip_level = get_hip_level_angle(keypoints)
        if hip_level is not None:
            if abs(hip_level) > 8:  # More than 8° from horizontal
                deviation = abs(hip_level)
                issues.append(f"Hips uneven by {deviation:.0f}°")
                score -= min(deviation, 20)

        score = max(score, 0)
        passed = score >= 70

        feedback = None
        if issues:
            feedback = FormFeedback(
                message=issues[0],  # Most significant issue
                priority=FeedbackPriority.WARNING if score < 60 else FeedbackPriority.SUGGESTION,
                body_part="trunk",
                frame_number=self.frame_count
            )

        return FormCheckResult(
            check_name="alignment",
            passed=passed,
            score=score,
            feedback=feedback,
            details={
                "trunk_angle": trunk_angle,
                "shoulder_level": shoulder_level,
                "hip_level": hip_level,
                "issues": issues
            }
        )

    def detect_compensation(self, keypoints: np.ndarray) -> FormCheckResult:
        """
        Identify when patient uses compensatory movements.

        Compensation Patterns to Detect:
        --------------------------------
        1. Hip hiking during leg exercises
        2. Trunk lean to assist weak muscles
        3. Shoulder shrugging during arm movements
        4. Momentum/swinging instead of controlled movement

        These patterns indicate the target muscle isn't strong enough
        to complete the movement, so other muscles take over.

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            FormCheckResult with compensation detection
        """
        if keypoints is None or len(self.keypoint_history) < 5:
            return FormCheckResult(
                check_name="compensation",
                passed=True,
                score=100,
                details={"message": "Insufficient data for compensation detection"}
            )

        compensations = []
        score = 100.0

        # Get current and baseline/previous measurements
        current_trunk = get_trunk_angle(keypoints)
        current_hip_level = get_hip_level_angle(keypoints)

        # Check for sudden trunk lean changes (compensatory movement)
        if len(self.keypoint_history) >= 5 and current_trunk is not None:
            prev_keypoints = self.keypoint_history[-5]
            prev_trunk = get_trunk_angle(prev_keypoints)

            if prev_trunk is not None:
                trunk_change = abs(current_trunk - prev_trunk)
                if trunk_change > 15:
                    compensations.append("Using trunk momentum")
                    score -= 25

        # Check for hip hiking (one hip rising to assist movement)
        if current_hip_level is not None and self.baseline_hip_height is not None:
            left_hip = extract_keypoint(keypoints, 11)
            right_hip = extract_keypoint(keypoints, 12)

            if left_hip is not None and right_hip is not None:
                hip_diff = abs(left_hip[1] - right_hip[1])
                if hip_diff > 30:  # Significant height difference
                    compensations.append("Hip hiking detected")
                    score -= 20

        # Check for shoulder shrugging
        left_shoulder = extract_keypoint(keypoints, 5)
        left_ear = extract_keypoint(keypoints, 3)

        if left_shoulder is not None and left_ear is not None:
            shoulder_ear_dist = np.linalg.norm(left_shoulder - left_ear)
            # If shoulder is too close to ear, likely shrugging
            if shoulder_ear_dist < 50:  # Threshold depends on image scale
                compensations.append("Shoulder shrugging")
                score -= 15

        score = max(score, 0)
        passed = len(compensations) == 0

        feedback = None
        if compensations:
            feedback = FormFeedback(
                message=compensations[0],
                priority=FeedbackPriority.WARNING,
                body_part="compensation",
                frame_number=self.frame_count
            )

        return FormCheckResult(
            check_name="compensation",
            passed=passed,
            score=score,
            feedback=feedback,
            details={"compensations": compensations}
        )

    def calculate_symmetry(self, keypoints: np.ndarray) -> FormCheckResult:
        """
        Compare left vs right side metrics.

        Symmetry Analysis:
        ------------------
        - Compare joint angles between sides
        - Compare limb lengths (may indicate positioning issues)
        - Flag significant asymmetries that may indicate:
          * Prior injury compensation
          * Muscle imbalance
          * Poor positioning

        Note: Some asymmetry is normal. We flag >15% differences.

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            FormCheckResult with symmetry assessment
        """
        if keypoints is None:
            return FormCheckResult(
                check_name="symmetry",
                passed=True,
                score=100,
                details={"message": "Cannot assess symmetry without keypoints"}
            )

        asymmetries = []
        score = 100.0

        # Compare knee angles
        left_knee_angle = get_angle_from_keypoints(keypoints, [11, 13, 15])
        right_knee_angle = get_angle_from_keypoints(keypoints, [12, 14, 16])

        if left_knee_angle is not None and right_knee_angle is not None:
            knee_diff = abs(left_knee_angle - right_knee_angle)
            if knee_diff > 15:
                asymmetries.append(f"Knee angle asymmetry: {knee_diff:.0f}°")
                score -= min(knee_diff, 25)

        # Compare elbow angles
        left_elbow_angle = get_angle_from_keypoints(keypoints, [5, 7, 9])
        right_elbow_angle = get_angle_from_keypoints(keypoints, [6, 8, 10])

        if left_elbow_angle is not None and right_elbow_angle is not None:
            elbow_diff = abs(left_elbow_angle - right_elbow_angle)
            if elbow_diff > 20:
                asymmetries.append(f"Elbow angle asymmetry: {elbow_diff:.0f}°")
                score -= min(elbow_diff, 20)

        # Compare hip angles (for trunk flexion)
        left_hip_angle = get_angle_from_keypoints(keypoints, [5, 11, 13])
        right_hip_angle = get_angle_from_keypoints(keypoints, [6, 12, 14])

        if left_hip_angle is not None and right_hip_angle is not None:
            hip_diff = abs(left_hip_angle - right_hip_angle)
            if hip_diff > 15:
                asymmetries.append(f"Hip angle asymmetry: {hip_diff:.0f}°")
                score -= min(hip_diff, 20)

        score = max(score, 0)
        passed = len(asymmetries) == 0

        feedback = None
        if asymmetries:
            feedback = FormFeedback(
                message=asymmetries[0],
                priority=FeedbackPriority.INFO,
                body_part="bilateral",
                frame_number=self.frame_count
            )

        return FormCheckResult(
            check_name="symmetry",
            passed=passed,
            score=score,
            feedback=feedback,
            details={
                "left_knee": left_knee_angle,
                "right_knee": right_knee_angle,
                "left_elbow": left_elbow_angle,
                "right_elbow": right_elbow_angle,
                "asymmetries": asymmetries
            }
        )

    def track_fatigue(self) -> FormCheckResult:
        """
        Detect form degradation over time indicating fatigue.

        Fatigue Indicators:
        -------------------
        1. Declining form scores over reps
        2. Increasing compensation patterns
        3. Decreasing range of motion
        4. Slower rep tempo

        This helps identify when the patient should rest to avoid injury.

        Returns:
            FormCheckResult with fatigue assessment
        """
        if len(self.rep_form_scores) < 3:
            return FormCheckResult(
                check_name="fatigue",
                passed=True,
                score=100,
                details={"message": "Need more reps to assess fatigue"}
            )

        # Calculate trend in form scores
        scores = np.array(self.rep_form_scores)
        x = np.arange(len(scores))

        # Linear regression for trend
        slope = np.polyfit(x, scores, 1)[0]

        # Calculate recent average vs initial average
        initial_avg = np.mean(scores[:3])
        recent_avg = np.mean(scores[-3:])
        decline = initial_avg - recent_avg

        fatigue_score = 100.0
        issues = []

        if slope < -2:  # Score dropping more than 2 points per rep
            issues.append(f"Form declining {abs(slope):.1f} points/rep")
            fatigue_score -= min(abs(slope) * 5, 30)

        if decline > 15:
            issues.append(f"Form dropped {decline:.0f}% from start")
            fatigue_score -= min(decline, 30)

        fatigue_score = max(fatigue_score, 0)
        passed = fatigue_score >= 70

        feedback = None
        if issues:
            message = "Consider taking a rest" if fatigue_score < 60 else issues[0]
            feedback = FormFeedback(
                message=message,
                priority=FeedbackPriority.WARNING if fatigue_score < 60 else FeedbackPriority.SUGGESTION,
                body_part="fatigue",
                frame_number=self.frame_count
            )

        return FormCheckResult(
            check_name="fatigue",
            passed=passed,
            score=fatigue_score,
            feedback=feedback,
            details={
                "slope": slope,
                "initial_avg": initial_avg,
                "recent_avg": recent_avg,
                "decline": decline,
                "issues": issues
            }
        )

    def on_rep_complete(self) -> float:
        """
        Called when a rep is completed. Finalizes rep-level metrics.

        Returns:
            Average form score for the completed rep
        """
        if self.current_rep_scores:
            rep_avg = np.mean(self.current_rep_scores)
            self.rep_form_scores.append(rep_avg)
            self.current_rep_scores = []
            return rep_avg
        return 0.0

    # =========================================================================
    # EXERCISE-SPECIFIC FORM CHECKS
    # =========================================================================

    def _check_knee_extension_form(self, keypoints: np.ndarray) -> List[FormCheckResult]:
        """
        Exercise-specific checks for seated knee extension.

        Key Form Points:
        ----------------
        1. Hip should stay in contact with seat (no lifting)
        2. Movement should be controlled (no swinging)
        3. Full extension without hyperextension
        4. Upper body remains stable

        Common Compensations:
        - Leaning back to assist lift
        - Hip hiking/lifting off seat
        - Using momentum rather than quad strength

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            List of FormCheckResults for knee extension
        """
        results = []

        if keypoints is None:
            return results

        # Check 1: Hip stability (shouldn't lift off seat)
        left_hip = extract_keypoint(keypoints, 11)
        if left_hip is not None and self.baseline_hip_height is not None:
            hip_rise = self.baseline_hip_height - left_hip[1]  # Y decreases upward
            if hip_rise > 20:  # Hip lifted more than 20 pixels
                results.append(FormCheckResult(
                    check_name="hip_stability",
                    passed=False,
                    score=max(70 - hip_rise, 0),
                    feedback=FormFeedback(
                        message="Keep your hip on the seat",
                        priority=FeedbackPriority.WARNING,
                        body_part="hip",
                        frame_number=self.frame_count
                    ),
                    details={"hip_rise": hip_rise}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="hip_stability",
                    passed=True,
                    score=100,
                    details={"hip_rise": hip_rise}
                ))

        # Check 2: Knee angle safety (no hyperextension)
        knee_angle = get_angle_from_keypoints(keypoints, [11, 13, 15])
        if knee_angle is not None:
            if knee_angle > SAFETY_THRESHOLDS["knee_hyperextension"]:
                results.append(FormCheckResult(
                    check_name="knee_safety",
                    passed=False,
                    score=0,
                    feedback=FormFeedback(
                        message="STOP: Knee hyperextension risk!",
                        priority=FeedbackPriority.CRITICAL,
                        body_part="knee",
                        frame_number=self.frame_count
                    ),
                    details={"knee_angle": knee_angle}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="knee_safety",
                    passed=True,
                    score=100,
                    details={"knee_angle": knee_angle}
                ))

        # Check 3: Trunk stability
        trunk_angle = get_trunk_angle(keypoints)
        if trunk_angle is not None:
            if trunk_angle > 20:
                results.append(FormCheckResult(
                    check_name="trunk_stability",
                    passed=False,
                    score=max(80 - trunk_angle, 0),
                    feedback=FormFeedback(
                        message="Keep your back straight against the chair",
                        priority=FeedbackPriority.SUGGESTION,
                        body_part="trunk",
                        frame_number=self.frame_count
                    ),
                    details={"trunk_angle": trunk_angle}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="trunk_stability",
                    passed=True,
                    score=100,
                    details={"trunk_angle": trunk_angle}
                ))

        return results

    def _check_shoulder_flexion_form(self, keypoints: np.ndarray) -> List[FormCheckResult]:
        """
        Exercise-specific checks for shoulder flexion (arm raise).

        Key Form Points:
        ----------------
        1. No trunk lean backward to assist lift
        2. No shoulder shrugging (using traps instead of deltoids)
        3. Controlled movement without momentum
        4. Elbow can be slightly bent but consistent

        Common Compensations:
        - Arching back as arm goes up
        - Shrugging shoulders toward ears
        - Side bending to increase range

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            List of FormCheckResults for shoulder flexion
        """
        results = []

        if keypoints is None:
            return results

        # Check 1: Trunk lean (shouldn't arch back)
        trunk_angle = get_trunk_angle(keypoints)
        if trunk_angle is not None:
            # Get direction of lean
            shoulder_center = get_shoulder_center(keypoints)
            hip_center = get_hip_center(keypoints)

            if shoulder_center is not None and hip_center is not None:
                # Positive means leaning back (shoulder behind hips in x)
                lean_back = hip_center[0] - shoulder_center[0]

                if trunk_angle > 15 and lean_back > 0:
                    results.append(FormCheckResult(
                        check_name="trunk_lean",
                        passed=False,
                        score=max(80 - trunk_angle, 0),
                        feedback=FormFeedback(
                            message="Don't arch your back - keep core tight",
                            priority=FeedbackPriority.WARNING,
                            body_part="trunk",
                            frame_number=self.frame_count
                        ),
                        details={"trunk_angle": trunk_angle, "lean_back": lean_back}
                    ))
                else:
                    results.append(FormCheckResult(
                        check_name="trunk_lean",
                        passed=True,
                        score=100,
                        details={"trunk_angle": trunk_angle}
                    ))

        # Check 2: Shoulder shrugging
        left_shoulder = extract_keypoint(keypoints, 5)
        left_ear = extract_keypoint(keypoints, 3)

        if left_shoulder is not None and left_ear is not None:
            shoulder_ear_dist = left_ear[1] - left_shoulder[1]  # Vertical distance

            if shoulder_ear_dist < 40:  # Shoulder too close to ear
                results.append(FormCheckResult(
                    check_name="shoulder_shrug",
                    passed=False,
                    score=60,
                    feedback=FormFeedback(
                        message="Relax your shoulders - don't shrug",
                        priority=FeedbackPriority.SUGGESTION,
                        body_part="shoulder",
                        frame_number=self.frame_count
                    ),
                    details={"shoulder_ear_distance": shoulder_ear_dist}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="shoulder_shrug",
                    passed=True,
                    score=100,
                    details={"shoulder_ear_distance": shoulder_ear_dist}
                ))

        # Check 3: Safe shoulder angle (not exceeding safe range)
        # Using hip-shoulder-wrist angle as proxy for arm elevation
        arm_angle = get_angle_from_keypoints(keypoints, [11, 5, 9])
        if arm_angle is not None:
            if arm_angle > SAFETY_THRESHOLDS["shoulder_elevation"]:
                results.append(FormCheckResult(
                    check_name="shoulder_safety",
                    passed=False,
                    score=50,
                    feedback=FormFeedback(
                        message="Arm raised too high - stop at shoulder level",
                        priority=FeedbackPriority.WARNING,
                        body_part="shoulder",
                        frame_number=self.frame_count
                    ),
                    details={"arm_angle": arm_angle}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="shoulder_safety",
                    passed=True,
                    score=100,
                    details={"arm_angle": arm_angle}
                ))

        return results

    def _check_squat_form(self, keypoints: np.ndarray) -> List[FormCheckResult]:
        """
        Exercise-specific checks for squats.

        Key Form Points:
        ----------------
        1. Knees tracking over toes (no valgus/caving in)
        2. Weight in heels (not coming onto toes)
        3. Chest up, back straight
        4. Appropriate depth for ability

        Common Compensations:
        - Knee valgus (knees caving inward) - ACL injury risk
        - Forward lean (back rounding)
        - Heel rise (shifts load to knees)
        - Asymmetric depth

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            List of FormCheckResults for squats
        """
        results = []

        if keypoints is None:
            return results

        # Check 1: Knee valgus (knees caving inward)
        # Compare knee position to ankle position in X axis
        left_knee = extract_keypoint(keypoints, 13)
        left_ankle = extract_keypoint(keypoints, 15)
        right_knee = extract_keypoint(keypoints, 14)
        right_ankle = extract_keypoint(keypoints, 16)

        if all(p is not None for p in [left_knee, left_ankle, right_knee, right_ankle]):
            # Left knee should be at or outside left ankle in X
            left_valgus = left_knee[0] - left_ankle[0]  # Positive = knee inside ankle
            # Right knee should be at or outside right ankle
            right_valgus = right_ankle[0] - right_knee[0]  # Positive = knee inside ankle

            max_valgus = max(left_valgus, right_valgus)

            if max_valgus > 30:  # Significant valgus
                results.append(FormCheckResult(
                    check_name="knee_valgus",
                    passed=False,
                    score=max(50, 100 - max_valgus),
                    feedback=FormFeedback(
                        message="Push your knees out - don't let them cave in",
                        priority=FeedbackPriority.CRITICAL,
                        body_part="knee",
                        frame_number=self.frame_count
                    ),
                    details={"left_valgus": left_valgus, "right_valgus": right_valgus}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="knee_valgus",
                    passed=True,
                    score=100,
                    details={"left_valgus": left_valgus, "right_valgus": right_valgus}
                ))

        # Check 2: Forward trunk lean
        trunk_angle = get_trunk_angle(keypoints)
        if trunk_angle is not None:
            if trunk_angle > SAFETY_THRESHOLDS["trunk_lean_max"]:
                results.append(FormCheckResult(
                    check_name="trunk_position",
                    passed=False,
                    score=max(60, 100 - trunk_angle),
                    feedback=FormFeedback(
                        message="Keep your chest up - don't lean forward",
                        priority=FeedbackPriority.WARNING,
                        body_part="trunk",
                        frame_number=self.frame_count
                    ),
                    details={"trunk_angle": trunk_angle}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="trunk_position",
                    passed=True,
                    score=100,
                    details={"trunk_angle": trunk_angle}
                ))

        # Check 3: Depth symmetry
        if left_knee is not None and right_knee is not None:
            knee_height_diff = abs(left_knee[1] - right_knee[1])
            if knee_height_diff > 25:
                results.append(FormCheckResult(
                    check_name="squat_symmetry",
                    passed=False,
                    score=80,
                    feedback=FormFeedback(
                        message="Keep your weight even on both legs",
                        priority=FeedbackPriority.SUGGESTION,
                        body_part="bilateral",
                        frame_number=self.frame_count
                    ),
                    details={"knee_height_diff": knee_height_diff}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="squat_symmetry",
                    passed=True,
                    score=100,
                    details={"knee_height_diff": knee_height_diff}
                ))

        return results

    def _check_pushup_form(self, keypoints: np.ndarray) -> List[FormCheckResult]:
        """
        Exercise-specific checks for push-ups.

        Key Form Points:
        ----------------
        1. Body in straight line (no hip sag or pike)
        2. Elbows at proper angle (not flared out)
        3. Full range of motion
        4. Head in neutral position

        Common Compensations:
        - Hip sagging (weak core)
        - Hip piking (avoiding full push-up)
        - Partial reps (not going deep enough)
        - Head dropping or lifting

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            List of FormCheckResults for push-ups
        """
        results = []

        if keypoints is None:
            return results

        # Check 1: Body alignment (shoulder-hip-ankle should be straight)
        shoulder = extract_keypoint(keypoints, 5)
        hip = extract_keypoint(keypoints, 11)
        ankle = extract_keypoint(keypoints, 15)

        if all(p is not None for p in [shoulder, hip, ankle]):
            # Calculate angle - should be close to 180 for straight body
            body_line_angle = calculate_angle(shoulder, hip, ankle)

            if body_line_angle < 160:  # Hip piking
                results.append(FormCheckResult(
                    check_name="body_alignment",
                    passed=False,
                    score=max(50, body_line_angle - 90),
                    feedback=FormFeedback(
                        message="Keep your body straight - hips are too high",
                        priority=FeedbackPriority.WARNING,
                        body_part="hip",
                        frame_number=self.frame_count
                    ),
                    details={"body_line_angle": body_line_angle, "issue": "piking"}
                ))
            elif body_line_angle > 190:  # Hip sagging
                results.append(FormCheckResult(
                    check_name="body_alignment",
                    passed=False,
                    score=max(50, 270 - body_line_angle),
                    feedback=FormFeedback(
                        message="Engage your core - don't let your hips sag",
                        priority=FeedbackPriority.WARNING,
                        body_part="hip",
                        frame_number=self.frame_count
                    ),
                    details={"body_line_angle": body_line_angle, "issue": "sagging"}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="body_alignment",
                    passed=True,
                    score=100,
                    details={"body_line_angle": body_line_angle}
                ))

        # Check 2: Head position
        nose = extract_keypoint(keypoints, 0)
        if shoulder is not None and nose is not None:
            # Head should be roughly in line with body
            head_forward = nose[0] - shoulder[0]
            if abs(head_forward) > 50:
                results.append(FormCheckResult(
                    check_name="head_position",
                    passed=False,
                    score=70,
                    feedback=FormFeedback(
                        message="Keep your head neutral - look at the floor",
                        priority=FeedbackPriority.SUGGESTION,
                        body_part="head",
                        frame_number=self.frame_count
                    ),
                    details={"head_forward": head_forward}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="head_position",
                    passed=True,
                    score=100,
                    details={"head_forward": head_forward}
                ))

        return results

    def _check_hip_abduction_form(self, keypoints: np.ndarray) -> List[FormCheckResult]:
        """
        Exercise-specific checks for hip abduction (side leg raise).

        Key Form Points:
        ----------------
        1. Trunk stays stable (no leaning away)
        2. Leg moves sideways, not forward
        3. Toes point forward, not up
        4. Controlled movement

        Common Compensations:
        - Leaning trunk away from moving leg
        - Rotating hip externally
        - Using momentum/swinging

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            List of FormCheckResults for hip abduction
        """
        results = []

        if keypoints is None:
            return results

        # Check 1: Trunk stability
        trunk_angle = get_trunk_angle(keypoints)
        if trunk_angle is not None:
            if trunk_angle > 20:
                results.append(FormCheckResult(
                    check_name="trunk_stability",
                    passed=False,
                    score=max(70, 100 - trunk_angle * 2),
                    feedback=FormFeedback(
                        message="Keep your trunk still - don't lean away",
                        priority=FeedbackPriority.WARNING,
                        body_part="trunk",
                        frame_number=self.frame_count
                    ),
                    details={"trunk_angle": trunk_angle}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="trunk_stability",
                    passed=True,
                    score=100,
                    details={"trunk_angle": trunk_angle}
                ))

        # Check 2: Hip level (shouldn't rotate pelvis)
        hip_level = get_hip_level_angle(keypoints)
        if hip_level is not None:
            deviation = abs(hip_level - 90)
            if deviation > 15:
                results.append(FormCheckResult(
                    check_name="pelvis_rotation",
                    passed=False,
                    score=max(70, 100 - deviation * 2),
                    feedback=FormFeedback(
                        message="Keep your hips level - don't rotate",
                        priority=FeedbackPriority.SUGGESTION,
                        body_part="hip",
                        frame_number=self.frame_count
                    ),
                    details={"hip_level_deviation": deviation}
                ))
            else:
                results.append(FormCheckResult(
                    check_name="pelvis_rotation",
                    passed=True,
                    score=100,
                    details={"hip_level_deviation": deviation}
                ))

        return results

    def _check_general_form(self, keypoints: np.ndarray) -> List[FormCheckResult]:
        """
        General form checks applicable to most exercises.

        Args:
            keypoints: YOLO pose keypoints

        Returns:
            List of FormCheckResults for general form
        """
        results = []

        if keypoints is None:
            return results

        # Basic posture check
        trunk_angle = get_trunk_angle(keypoints)
        if trunk_angle is not None:
            score = max(0, 100 - trunk_angle * 2)
            results.append(FormCheckResult(
                check_name="posture",
                passed=trunk_angle < 25,
                score=score,
                details={"trunk_angle": trunk_angle}
            ))

        return results

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _update_baseline(self, keypoints: np.ndarray):
        """Update baseline measurements from initial frames."""
        if self.baseline_trunk_angle is None:
            self.baseline_trunk_angle = get_trunk_angle(keypoints)

        if self.baseline_hip_height is None:
            left_hip = extract_keypoint(keypoints, 11)
            if left_hip is not None:
                self.baseline_hip_height = left_hip[1]

    def _score_to_status(self, score: float, checks: List[FormCheckResult]) -> FormStatus:
        """Convert score and check results to overall status."""
        # Check for critical issues first
        for check in checks:
            if check.feedback and check.feedback.priority == FeedbackPriority.CRITICAL:
                return FormStatus.STOP

        if score >= 90:
            return FormStatus.EXCELLENT
        elif score >= 75:
            return FormStatus.GOOD
        elif score >= 60:
            return FormStatus.WARNING
        else:
            return FormStatus.POOR

    def _get_skeleton_colors(self, checks: List[FormCheckResult],
                            keypoints: np.ndarray) -> Dict[str, Tuple[int, int, int]]:
        """
        Determine colors for skeleton segments based on form checks.

        Returns a dictionary mapping body part names to BGR colors.
        """
        colors = {}

        # Default all to green
        for name in KEYPOINT_NAMES.values():
            colors[name] = (0, 255, 0)  # Green

        # Color code based on check results
        for check in checks:
            if check.feedback:
                body_part = check.feedback.body_part
                color = get_form_color(check.score)

                # Map body parts to keypoint names
                part_mapping = {
                    "knee": ["left_knee", "right_knee"],
                    "hip": ["left_hip", "right_hip"],
                    "shoulder": ["left_shoulder", "right_shoulder"],
                    "trunk": ["left_hip", "right_hip", "left_shoulder", "right_shoulder"],
                    "bilateral": ["left_knee", "right_knee", "left_hip", "right_hip"],
                }

                if body_part in part_mapping:
                    for kp_name in part_mapping[body_part]:
                        colors[kp_name] = color

        return colors

    def reset(self):
        """Reset analyzer state for a new exercise set."""
        self.angle_history.clear()
        self.form_score_history.clear()
        self.keypoint_history.clear()
        self.rep_form_scores = []
        self.current_rep_scores = []
        self.frame_count = 0
        self.baseline_trunk_angle = None
        self.baseline_hip_height = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_form_feedback_text(assessment: OverallFormAssessment) -> str:
    """
    Get human-readable feedback text from assessment.

    Args:
        assessment: OverallFormAssessment object

    Returns:
        Formatted feedback string
    """
    lines = [f"Form Score: {assessment.overall_score:.0f}% ({assessment.status.value})"]

    for feedback in assessment.feedback_messages:
        prefix = "!" if feedback.priority.value >= 3 else "-"
        lines.append(f"{prefix} {feedback.message}")

    return "\n".join(lines)


def should_stop_exercise(assessment: OverallFormAssessment) -> bool:
    """
    Check if exercise should be stopped for safety.

    Args:
        assessment: OverallFormAssessment object

    Returns:
        True if exercise should stop
    """
    return assessment.status == FormStatus.STOP
