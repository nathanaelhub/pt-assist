"""
Observation Logger for AI Physical Therapy Assistant
====================================================

Creates structured, timestamped observation log files for each PT session.

Each session generates a new JSON file in the observation_logs/ folder,
capturing a full event stream (reps, exercise switches, session start/end)
plus a final summary.

File naming: session_YYYY-MM-DD_HHMMSS.json

Usage:
    from src.observation_logger import ObservationLogger

    logger = ObservationLogger(source="video")
    logger.start_session("Squat", target_reps=10)

    # After each rep completes:
    logger.log_rep(
        rep_number=1, exercise="Squat",
        rom=80.5, form_score=85.0,
        duration_seconds=2.1,
        min_angle=92.3, max_angle=172.8
    )

    # When the user switches exercises:
    logger.log_exercise_change("Squat", "Knee Extension")

    # At session end:
    logger.end_session(total_reps=10, avg_form_score=82.5, avg_rom=78.3)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any


class ObservationLogger:
    """
    Structured observation logger for PT sessions.

    Creates one JSON file per session in the observation_logs/ directory.
    The file is written incrementally after every event so partial sessions
    are preserved if the process is interrupted.

    Log file schema:
    {
        "session_id": "session_2026-03-23_113200",
        "source": "video" | "webcam",
        "start_time": "<ISO 8601>",
        "end_time": "<ISO 8601>" | null,
        "duration_seconds": float | null,
        "initial_exercise": "Squat",
        "target_reps": 10,
        "events": [
            {"type": "session_start", "timestamp": "...", "exercise": "...", "target_reps": 10},
            {"type": "rep", "timestamp": "...", "rep_number": 1, "exercise": "...",
             "rom": 80.5, "form_score": 85.0, "duration_seconds": 2.1,
             "min_angle": 92.3, "max_angle": 172.8},
            {"type": "exercise_change", "timestamp": "...",
             "from_exercise": "...", "to_exercise": "..."},
            {"type": "partial_rep", "timestamp": "...", "exercise": "...",
             "rom": 40.2, "min_angle": 95.0, "max_angle": 135.2,
             "duration_seconds": 1.3, "reason": "insufficient_rom"},
            {"type": "reset", "timestamp": "...", "exercise": "...", "rep_count_at_reset": 5},
            {"type": "session_end", "timestamp": "...", "total_reps": 10,
             "avg_form_score": 82.5, "avg_rom": 78.3, "duration_seconds": 480.0}
        ],
        "summary": {
            "total_reps": 10,
            "exercises_performed": [
                {"exercise": "Squat", "reps": 10, "avg_form_score": 82.5, "avg_rom": 78.3}
            ],
            "avg_form_score": 82.5,
            "avg_rom": 78.3,
            "warnings": []
        }
    }
    """

    def __init__(self, log_dir: Optional[str] = None, source: str = "video"):
        """
        Initialize the ObservationLogger.

        Args:
            log_dir: Directory to store logs.  Defaults to observation_logs/
                     inside the project root (two levels up from this file).
            source: "video" or "webcam"
        """
        if log_dir is None:
            project_root = Path(__file__).parent.parent
            self.log_dir = project_root / "observation_logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.session_data: Dict[str, Any] = {}
        self.file_path: Optional[Path] = None
        self._started = False
        self._start_dt: Optional[datetime] = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_session(self, exercise_name: str, target_reps: int = 10) -> str:
        """
        Start a new observation session and create the log file.

        Args:
            exercise_name: Name of the exercise being performed.
            target_reps: Target number of repetitions for the session.

        Returns:
            The session_id string (used as the filename stem).
        """
        now = datetime.now()
        self._start_dt = now
        session_id = f"session_{now.strftime('%Y-%m-%d_%H%M%S')}"

        self.session_data = {
            "session_id": session_id,
            "source": self.source,
            "start_time": now.isoformat(),
            "end_time": None,
            "duration_seconds": None,
            "initial_exercise": exercise_name,
            "target_reps": target_reps,
            "events": [
                {
                    "type": "session_start",
                    "timestamp": now.isoformat(),
                    "exercise": exercise_name,
                    "target_reps": target_reps,
                }
            ],
            "summary": {
                "total_reps": 0,
                "exercises_performed": [],
                "avg_form_score": 0.0,
                "avg_rom": 0.0,
                "warnings": [],
            },
        }

        self.file_path = self.log_dir / f"{session_id}.json"
        self._started = True
        self._save()

        return session_id

    def log_rep(
        self,
        rep_number: int,
        exercise: str,
        rom: float,
        form_score: float,
        duration_seconds: float,
        min_angle: float = 0.0,
        max_angle: float = 0.0,
    ) -> None:
        """
        Log a completed repetition.

        Args:
            rep_number: The rep count for this exercise block.
            exercise: Exercise name.
            rom: Range of motion achieved (degrees).
            form_score: Form quality score 0-100.
            duration_seconds: Time taken for this rep.
            min_angle: Minimum joint angle recorded during the rep.
            max_angle: Maximum joint angle recorded during the rep.
        """
        if not self._started:
            return

        event = {
            "type": "rep",
            "timestamp": datetime.now().isoformat(),
            "rep_number": rep_number,
            "exercise": exercise,
            "rom": round(float(rom), 1),
            "form_score": round(float(form_score), 1),
            "duration_seconds": round(float(duration_seconds), 2),
            "min_angle": round(float(min_angle), 1),
            "max_angle": round(float(max_angle), 1),
        }

        self.session_data["events"].append(event)
        self._save()

    def log_partial_rep(
        self,
        exercise: str,
        rom: float,
        min_angle: float,
        max_angle: float,
        duration_seconds: float,
        reason: str = "insufficient_rom",
    ) -> None:
        """
        Log a partial/incomplete repetition attempt.

        A partial rep is recorded when motion is detected but the user
        did not reach the required threshold to count as a completed rep
        (e.g., didn't go deep enough, stopped mid-movement, or reversed
        direction before reaching full range of motion).

        Args:
            exercise: Exercise name.
            rom: Range of motion actually achieved (degrees).
            min_angle: Minimum joint angle recorded during the attempt.
            max_angle: Maximum joint angle recorded during the attempt.
            duration_seconds: Time spent in the attempted movement.
            reason: Why the rep was not counted. One of:
                     "insufficient_rom" - didn't reach target angle
                     "reversed_early"   - changed direction before threshold
                     "interrupted"      - session ended / exercise switched mid-rep
        """
        if not self._started:
            return

        event = {
            "type": "partial_rep",
            "timestamp": datetime.now().isoformat(),
            "exercise": exercise,
            "rom": round(float(rom), 1),
            "min_angle": round(float(min_angle), 1),
            "max_angle": round(float(max_angle), 1),
            "duration_seconds": round(float(duration_seconds), 2),
            "reason": reason,
        }

        self.session_data["events"].append(event)
        self._save()

    def log_exercise_change(self, from_exercise: str, to_exercise: str) -> None:
        """
        Log when the user switches from one exercise to another.

        Args:
            from_exercise: Exercise being switched away from.
            to_exercise: Exercise being switched to.
        """
        if not self._started:
            return

        event = {
            "type": "exercise_change",
            "timestamp": datetime.now().isoformat(),
            "from_exercise": from_exercise,
            "to_exercise": to_exercise,
        }

        self.session_data["events"].append(event)
        self._save()

    def log_reset(self, exercise: str, rep_count_at_reset: int) -> None:
        """
        Log when the rep counter is reset mid-session.

        Args:
            exercise: Current exercise at the time of reset.
            rep_count_at_reset: Rep count at the time of reset.
        """
        if not self._started:
            return

        event = {
            "type": "reset",
            "timestamp": datetime.now().isoformat(),
            "exercise": exercise,
            "rep_count_at_reset": rep_count_at_reset,
        }

        self.session_data["events"].append(event)
        self._save()

    def end_session(
        self,
        total_reps: int,
        avg_form_score: float,
        avg_rom: float,
        warnings: Optional[List[str]] = None,
    ) -> str:
        """
        Mark the session as ended and write the final summary.

        Args:
            total_reps: Total reps completed across all exercises.
            avg_form_score: Average form score (0-100) for the session.
            avg_rom: Average range of motion in degrees.
            warnings: Any warnings generated during the session.

        Returns:
            Path to the saved log file.
        """
        if not self._started:
            return str(self.file_path) if self.file_path else ""

        now = datetime.now()
        duration = (now - self._start_dt).total_seconds() if self._start_dt else 0.0

        self.session_data["end_time"] = now.isoformat()
        self.session_data["duration_seconds"] = round(duration, 1)

        # Build per-exercise summary from the rep events collected so far
        exercise_stats: Dict[str, Dict[str, Any]] = {}
        for event in self.session_data["events"]:
            if event["type"] == "rep":
                ex = event["exercise"]
                if ex not in exercise_stats:
                    exercise_stats[ex] = {"reps": 0, "form_scores": [], "roms": []}
                exercise_stats[ex]["reps"] += 1
                exercise_stats[ex]["form_scores"].append(event["form_score"])
                exercise_stats[ex]["roms"].append(event["rom"])

        exercises_summary = []
        for ex_name, stats in exercise_stats.items():
            n = len(stats["form_scores"])
            exercises_summary.append(
                {
                    "exercise": ex_name,
                    "reps": stats["reps"],
                    "avg_form_score": round(
                        sum(stats["form_scores"]) / n if n else 0.0, 1
                    ),
                    "avg_rom": round(
                        sum(stats["roms"]) / n if n else 0.0, 1
                    ),
                }
            )

        self.session_data["summary"] = {
            "total_reps": int(total_reps),
            "exercises_performed": exercises_summary,
            "avg_form_score": round(float(avg_form_score), 1),
            "avg_rom": round(float(avg_rom), 1),
            "warnings": warnings or [],
        }

        # Append the session_end event
        self.session_data["events"].append(
            {
                "type": "session_end",
                "timestamp": now.isoformat(),
                "total_reps": int(total_reps),
                "avg_form_score": round(float(avg_form_score), 1),
                "avg_rom": round(float(avg_rom), 1),
                "duration_seconds": round(duration, 1),
            }
        )

        self._started = False
        self._save()

        print(f"[ObservationLogger] Session log saved: {self.file_path}")
        return str(self.file_path)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def session_id(self) -> Optional[str]:
        """Return the current session ID, or None if no session is active."""
        return self.session_data.get("session_id")

    @property
    def log_file_path(self) -> Optional[str]:
        """Return the absolute path to the current log file."""
        return str(self.file_path) if self.file_path else None

    @property
    def is_active(self) -> bool:
        """True if a session is currently in progress."""
        return self._started

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _save(self) -> None:
        """Write the current session_data dict to the JSON log file."""
        if self.file_path is None:
            return
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.session_data, f, indent=2)


# =============================================================================
# Convenience helpers for the rest of the app
# =============================================================================

def list_observation_logs(log_dir: Optional[str] = None) -> List[Path]:
    """
    Return all observation log files sorted newest-first.

    Args:
        log_dir: Override the default observation_logs/ directory.

    Returns:
        List of Path objects for each .json log file found.
    """
    if log_dir is None:
        project_root = Path(__file__).parent.parent
        log_dir = project_root / "observation_logs"
    else:
        log_dir = Path(log_dir)

    if not log_dir.exists():
        return []

    return sorted(log_dir.glob("session_*.json"), reverse=True)


def load_observation_log(file_path: str) -> Dict[str, Any]:
    """
    Load and return a single observation log as a dictionary.

    Args:
        file_path: Path to the JSON log file.

    Returns:
        Parsed log data dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
