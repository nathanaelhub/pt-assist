"""
Utility Functions for AI Physical Therapy Assistant
====================================================

This module contains shared helper functions used across the application,
including angle calculations, keypoint extraction, and visualization utilities.

YOLO11 Pose Keypoint Reference (17 keypoints):
    0: Nose           1: Left Eye       2: Right Eye
    3: Left Ear       4: Right Ear      5: Left Shoulder
    6: Right Shoulder 7: Left Elbow     8: Right Elbow
    9: Left Wrist    10: Right Wrist   11: Left Hip
   12: Right Hip     13: Left Knee     14: Right Knee
   15: Left Ankle    16: Right Ankle
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
import cv2


# =============================================================================
# KEYPOINT CONSTANTS
# =============================================================================

KEYPOINT_NAMES = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

# Keypoint pairs for skeleton drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2),      # Nose to eyes
    (1, 3), (2, 4),      # Eyes to ears
    (5, 6),              # Shoulders
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (5, 11), (6, 12),    # Torso sides
    (11, 12),            # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]

# Confidence threshold for keypoint detection
DEFAULT_CONFIDENCE_THRESHOLD = 0.3


# =============================================================================
# ANGLE CALCULATION FUNCTIONS
# =============================================================================

def calculate_angle(point1: np.ndarray, vertex: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the angle formed by three points, with vertex as the middle point.

    Mathematical Concept:
    ---------------------
    Uses the dot product formula to find the angle between two vectors:

        cos(θ) = (A · B) / (|A| * |B|)

    Where:
    - A is the vector from vertex to point1
    - B is the vector from vertex to point2
    - θ is the angle between them

    Args:
        point1: First point coordinates [x, y]
        vertex: Middle point (vertex of angle) [x, y]
        point2: Third point coordinates [x, y]

    Returns:
        Angle in degrees (0-180)

    Example:
        >>> hip = np.array([100, 200])
        >>> knee = np.array([100, 300])
        >>> ankle = np.array([100, 400])
        >>> angle = calculate_angle(hip, knee, ankle)  # ~180° (straight leg)
    """
    # Create vectors from vertex to each point
    vector1 = point1 - vertex
    vector2 = point2 - vertex

    # Calculate dot product: A · B = |A||B|cos(θ)
    dot_product = np.dot(vector1, vector2)

    # Calculate magnitudes (lengths) of vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0

    # Calculate cosine of angle
    cos_angle = dot_product / (magnitude1 * magnitude2)

    # Clamp to [-1, 1] to handle floating point errors
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # Convert to degrees
    angle = np.degrees(np.arccos(cos_angle))

    return angle


def calculate_angle_with_vertical(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the angle of a line segment with respect to vertical.

    Useful for checking trunk lean, arm elevation from vertical, etc.

    Args:
        point1: First point (typically upper) [x, y]
        point2: Second point (typically lower) [x, y]

    Returns:
        Angle in degrees from vertical (0° = perfectly vertical)
    """
    # Vector from point1 to point2
    vector = point2 - point1

    # Vertical reference (pointing down in image coordinates)
    vertical = np.array([0, 1])

    # Calculate angle
    dot_product = np.dot(vector, vertical)
    magnitude = np.linalg.norm(vector)

    if magnitude == 0:
        return 0.0

    cos_angle = dot_product / magnitude
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def calculate_angle_with_horizontal(point1: np.ndarray, point2: np.ndarray) -> float:
    """
    Calculate the angle of a line segment with respect to horizontal.

    Useful for checking hip level, shoulder alignment, etc.

    Args:
        point1: First point [x, y]
        point2: Second point [x, y]

    Returns:
        Angle in degrees from horizontal (0° = perfectly horizontal)
    """
    # Vector from point1 to point2
    vector = point2 - point1

    # Horizontal reference (pointing right)
    horizontal = np.array([1, 0])

    # Calculate angle
    dot_product = np.dot(vector, horizontal)
    magnitude = np.linalg.norm(vector)

    if magnitude == 0:
        return 0.0

    cos_angle = dot_product / magnitude
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


# =============================================================================
# KEYPOINT EXTRACTION FUNCTIONS
# =============================================================================

def extract_keypoint(keypoints: np.ndarray, index: int,
                     confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Optional[np.ndarray]:
    """
    Safely extract a keypoint from the YOLO pose output.

    YOLO Pose Output Format:
    ------------------------
    keypoints shape: (17, 3) for a single person where:
    - 17 is the number of body keypoints
    - 3 is [x, y, confidence]

    Args:
        keypoints: Array of keypoint coordinates and confidences
        index: Keypoint index (0-16)
        confidence_threshold: Minimum confidence to consider valid

    Returns:
        [x, y] coordinates if confidence > threshold, else None
    """
    if keypoints is None or len(keypoints) == 0:
        return None

    if index < 0 or index >= len(keypoints):
        return None

    # Get keypoint data [x, y, confidence]
    kp = keypoints[index]

    # Check confidence threshold
    if len(kp) >= 3 and kp[2] < confidence_threshold:
        return None

    return np.array([kp[0], kp[1]])


def get_angle_from_keypoints(keypoints: np.ndarray, indices: List[int],
                             confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Optional[float]:
    """
    Calculate angle from three keypoint indices.

    Args:
        keypoints: Full keypoint array from YOLO
        indices: [point1_idx, vertex_idx, point2_idx]
        confidence_threshold: Minimum keypoint confidence

    Returns:
        Calculated angle or None if keypoints not detected
    """
    if len(indices) != 3:
        return None

    point1 = extract_keypoint(keypoints, indices[0], confidence_threshold)
    vertex = extract_keypoint(keypoints, indices[1], confidence_threshold)
    point2 = extract_keypoint(keypoints, indices[2], confidence_threshold)

    if point1 is None or vertex is None or point2 is None:
        return None

    return calculate_angle(point1, vertex, point2)


def get_keypoint_distance(keypoints: np.ndarray, idx1: int, idx2: int,
                          confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Optional[float]:
    """
    Calculate Euclidean distance between two keypoints.

    Useful for tracking limb lengths or detecting reach distance.

    Args:
        keypoints: Full keypoint array
        idx1: First keypoint index
        idx2: Second keypoint index
        confidence_threshold: Minimum confidence

    Returns:
        Distance in pixels or None if keypoints not detected
    """
    p1 = extract_keypoint(keypoints, idx1, confidence_threshold)
    p2 = extract_keypoint(keypoints, idx2, confidence_threshold)

    if p1 is None or p2 is None:
        return None

    return np.linalg.norm(p2 - p1)


def get_midpoint(keypoints: np.ndarray, idx1: int, idx2: int,
                 confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Optional[np.ndarray]:
    """
    Calculate midpoint between two keypoints.

    Useful for finding body center, hip center, etc.

    Args:
        keypoints: Full keypoint array
        idx1: First keypoint index
        idx2: Second keypoint index
        confidence_threshold: Minimum confidence

    Returns:
        Midpoint [x, y] or None if keypoints not detected
    """
    p1 = extract_keypoint(keypoints, idx1, confidence_threshold)
    p2 = extract_keypoint(keypoints, idx2, confidence_threshold)

    if p1 is None or p2 is None:
        return None

    return (p1 + p2) / 2


# =============================================================================
# BODY SEGMENT FUNCTIONS
# =============================================================================

def get_hip_center(keypoints: np.ndarray) -> Optional[np.ndarray]:
    """Get the center point between left and right hips."""
    return get_midpoint(keypoints, 11, 12)


def get_shoulder_center(keypoints: np.ndarray) -> Optional[np.ndarray]:
    """Get the center point between left and right shoulders."""
    return get_midpoint(keypoints, 5, 6)


def get_trunk_angle(keypoints: np.ndarray) -> Optional[float]:
    """
    Calculate trunk lean angle from vertical.

    Uses the line from hip center to shoulder center.

    Returns:
        Angle in degrees from vertical (0° = upright)
    """
    hip_center = get_hip_center(keypoints)
    shoulder_center = get_shoulder_center(keypoints)

    if hip_center is None or shoulder_center is None:
        return None

    return calculate_angle_with_vertical(shoulder_center, hip_center)


def get_shoulder_level_angle(keypoints: np.ndarray) -> Optional[float]:
    """
    Calculate how level the shoulders are.

    Returns:
        Angle from horizontal (0° = perfectly level)
    """
    left_shoulder = extract_keypoint(keypoints, 5)
    right_shoulder = extract_keypoint(keypoints, 6)

    if left_shoulder is None or right_shoulder is None:
        return None

    return calculate_angle_with_horizontal(left_shoulder, right_shoulder)


def get_hip_level_angle(keypoints: np.ndarray) -> Optional[float]:
    """
    Calculate how level the hips are.

    Returns:
        Angle from horizontal (0° = perfectly level)
    """
    left_hip = extract_keypoint(keypoints, 11)
    right_hip = extract_keypoint(keypoints, 12)

    if left_hip is None or right_hip is None:
        return None

    return calculate_angle_with_horizontal(left_hip, right_hip)


# =============================================================================
# VISUALIZATION UTILITIES
# =============================================================================

def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  thickness: int = 2,
                  confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> np.ndarray:
    """
    Draw skeleton overlay on frame.

    Args:
        frame: Image to draw on
        keypoints: YOLO pose keypoints
        color: Line color (BGR)
        thickness: Line thickness
        confidence_threshold: Minimum confidence to draw

    Returns:
        Frame with skeleton drawn
    """
    if keypoints is None:
        return frame

    frame = frame.copy()

    # Draw connections
    for idx1, idx2 in SKELETON_CONNECTIONS:
        p1 = extract_keypoint(keypoints, idx1, confidence_threshold)
        p2 = extract_keypoint(keypoints, idx2, confidence_threshold)

        if p1 is not None and p2 is not None:
            pt1 = tuple(p1.astype(int))
            pt2 = tuple(p2.astype(int))
            cv2.line(frame, pt1, pt2, color, thickness)

    # Draw keypoints
    for i in range(17):
        point = extract_keypoint(keypoints, i, confidence_threshold)
        if point is not None:
            pt = tuple(point.astype(int))
            cv2.circle(frame, pt, 5, color, -1)

    return frame


def draw_angle_arc(frame: np.ndarray, keypoints: np.ndarray,
                   indices: List[int], angle: float,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   radius: int = 40) -> np.ndarray:
    """
    Draw angle visualization with arc and value.

    Args:
        frame: Image to draw on
        keypoints: YOLO pose keypoints
        indices: [point1_idx, vertex_idx, point2_idx]
        angle: Calculated angle value
        color: Drawing color (BGR)
        radius: Arc radius in pixels

    Returns:
        Frame with angle visualization
    """
    if keypoints is None or len(indices) != 3:
        return frame

    point1 = extract_keypoint(keypoints, indices[0])
    vertex = extract_keypoint(keypoints, indices[1])
    point2 = extract_keypoint(keypoints, indices[2])

    if point1 is None or vertex is None or point2 is None:
        return frame

    frame = frame.copy()

    # Convert to integer coordinates
    p1 = tuple(point1.astype(int))
    v = tuple(vertex.astype(int))
    p2 = tuple(point2.astype(int))

    # Draw lines
    cv2.line(frame, v, p1, color, 3)
    cv2.line(frame, v, p2, color, 3)

    # Draw points
    cv2.circle(frame, p1, 8, (255, 0, 0), -1)
    cv2.circle(frame, v, 10, color, -1)
    cv2.circle(frame, p2, 8, (255, 0, 0), -1)

    # Draw angle arc
    angle1 = np.arctan2(point1[1] - vertex[1], point1[0] - vertex[0])
    angle2 = np.arctan2(point2[1] - vertex[1], point2[0] - vertex[0])

    start_angle = np.degrees(min(angle1, angle2))
    end_angle = np.degrees(max(angle1, angle2))

    cv2.ellipse(frame, v, (radius, radius), 0, start_angle, end_angle, color, 2)

    # Draw angle text
    text_pos = (v[0] + 15, v[1] - 15)
    cv2.putText(frame, f"{angle:.1f}", text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


def get_form_color(score: float) -> Tuple[int, int, int]:
    """
    Get color based on form score.

    Args:
        score: Form score (0-100)

    Returns:
        BGR color tuple
    """
    if score >= 80:
        return (0, 255, 0)    # Green - good form
    elif score >= 60:
        return (0, 255, 255)  # Yellow - warning
    else:
        return (0, 0, 255)    # Red - poor form


# =============================================================================
# DATA CONVERSION UTILITIES
# =============================================================================

def keypoints_to_dict(keypoints: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Convert keypoints array to named dictionary.

    Useful for JSON serialization and debugging.

    Args:
        keypoints: YOLO pose keypoints array

    Returns:
        Dictionary with keypoint names and coordinates
    """
    result = {}
    for idx, name in KEYPOINT_NAMES.items():
        if idx < len(keypoints):
            kp = keypoints[idx]
            result[name] = {
                "x": float(kp[0]),
                "y": float(kp[1]),
                "confidence": float(kp[2]) if len(kp) > 2 else 1.0
            }
    return result


def normalize_keypoints(keypoints: np.ndarray,
                        frame_width: int, frame_height: int) -> np.ndarray:
    """
    Normalize keypoint coordinates to 0-1 range.

    Useful for comparing poses across different video resolutions.

    Args:
        keypoints: YOLO pose keypoints
        frame_width: Video frame width
        frame_height: Video frame height

    Returns:
        Normalized keypoints array
    """
    if keypoints is None:
        return None

    normalized = keypoints.copy()
    normalized[:, 0] /= frame_width   # x coordinates
    normalized[:, 1] /= frame_height  # y coordinates
    # confidence remains unchanged

    return normalized
