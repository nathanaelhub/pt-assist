"""
Session Tracking Module for AI Physical Therapy Assistant
==========================================================

This module provides persistent tracking of PT sessions across multiple visits,
enabling progress monitoring and trend analysis.

Features:
- Session data persistence (JSON/CSV)
- Progress comparison across sessions
- ROM improvement tracking
- Form score progression
- Visualization helpers for reports

HIPAA Considerations:
--------------------
In a production healthcare environment, this data would be considered PHI
(Protected Health Information) and would require:
- Encryption at rest and in transit
- Access controls and audit logging
- Secure backup procedures
- Patient consent and data retention policies

The current implementation stores data in plaintext for educational purposes.
Production deployment would need encryption layers added.

Author: AI PT Assistant
"""

import json
import csv
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Optional: matplotlib for visualizations
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ExerciseRecord:
    """Record of a single exercise within a session."""
    exercise_name: str
    reps_completed: int
    target_reps: int
    avg_form_score: float
    min_rom: float  # Minimum angle achieved
    max_rom: float  # Maximum angle achieved
    rom_range: float  # max_rom - min_rom
    avg_rep_duration: float  # seconds
    form_warnings: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class SessionData:
    """Complete data for a PT session."""
    session_id: str
    patient_id: str  # Would be encrypted in production
    date: str
    start_time: str
    end_time: str
    duration_minutes: float
    exercises: List[ExerciseRecord] = field(default_factory=list)
    overall_form_score: float = 0.0
    therapist_notes: str = ""
    flags_for_review: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressMetrics:
    """Progress metrics comparing multiple sessions."""
    exercise_name: str
    sessions_count: int
    rom_trend: str  # "improving", "stable", "declining"
    rom_change_percent: float
    form_score_trend: str
    form_score_change: float
    reps_trend: str
    avg_reps_change: float
    dates: List[str] = field(default_factory=list)
    rom_values: List[float] = field(default_factory=list)
    form_scores: List[float] = field(default_factory=list)


# =============================================================================
# SESSION TRACKER CLASS
# =============================================================================

class SessionTracker:
    """
    Manages PT session data persistence and progress tracking.

    This class handles:
    - Creating and saving session records
    - Loading historical session data
    - Calculating progress metrics
    - Generating reports and visualizations

    Usage:
        tracker = SessionTracker(data_dir="./sessions", patient_id="PT001")
        tracker.start_session()
        # ... during exercise ...
        tracker.add_exercise_record(exercise_record)
        # ... end of session ...
        tracker.end_session(therapist_notes="Good progress today")
        tracker.save_session()
    """

    def __init__(self, data_dir: str = "./sessions", patient_id: str = "default"):
        """
        Initialize the SessionTracker.

        Args:
            data_dir: Directory for storing session data
            patient_id: Identifier for the patient (would be encrypted in production)
        """
        self.data_dir = Path(data_dir)
        self.patient_id = patient_id
        self.current_session: Optional[SessionData] = None
        self.session_history: List[SessionData] = []

        # Create data directory if it doesn't exist
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Load existing sessions for this patient
        self._load_patient_history()

    def _load_patient_history(self):
        """Load all historical sessions for the current patient."""
        self.session_history = []

        # Look for session files matching patient ID
        pattern = f"{self.patient_id}_*.json"
        session_files = sorted(self.data_dir.glob(pattern))

        for filepath in session_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    session = self._dict_to_session(data)
                    self.session_history.append(session)
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Could not load session file {filepath}: {e}")

    def _dict_to_session(self, data: Dict) -> SessionData:
        """Convert dictionary to SessionData object."""
        exercises = [
            ExerciseRecord(**ex) for ex in data.get('exercises', [])
        ]

        return SessionData(
            session_id=data['session_id'],
            patient_id=data['patient_id'],
            date=data['date'],
            start_time=data['start_time'],
            end_time=data.get('end_time', ''),
            duration_minutes=data.get('duration_minutes', 0),
            exercises=exercises,
            overall_form_score=data.get('overall_form_score', 0),
            therapist_notes=data.get('therapist_notes', ''),
            flags_for_review=data.get('flags_for_review', []),
            metadata=data.get('metadata', {})
        )

    def start_session(self, metadata: Dict[str, Any] = None) -> str:
        """
        Start a new PT session.

        Args:
            metadata: Optional metadata (e.g., equipment used, location)

        Returns:
            Session ID for the new session
        """
        now = datetime.now()
        session_id = f"{self.patient_id}_{now.strftime('%Y%m%d_%H%M%S')}"

        self.current_session = SessionData(
            session_id=session_id,
            patient_id=self.patient_id,
            date=now.strftime('%Y-%m-%d'),
            start_time=now.strftime('%H:%M:%S'),
            end_time='',
            duration_minutes=0,
            metadata=metadata or {}
        )

        return session_id

    def add_exercise_record(self, record: ExerciseRecord):
        """
        Add an exercise record to the current session.

        Args:
            record: ExerciseRecord with exercise data
        """
        if self.current_session is None:
            raise ValueError("No active session. Call start_session() first.")

        self.current_session.exercises.append(record)

        # Check for issues that should be flagged
        if record.avg_form_score < 60:
            self.current_session.flags_for_review.append(
                f"{record.exercise_name}: Low form score ({record.avg_form_score:.0f}%)"
            )

        if record.reps_completed < record.target_reps * 0.5:
            self.current_session.flags_for_review.append(
                f"{record.exercise_name}: Completed only {record.reps_completed}/{record.target_reps} reps"
            )

    def end_session(self, therapist_notes: str = ""):
        """
        End the current session and calculate summary metrics.

        Args:
            therapist_notes: Notes from the therapist about the session
        """
        if self.current_session is None:
            raise ValueError("No active session to end.")

        now = datetime.now()
        self.current_session.end_time = now.strftime('%H:%M:%S')

        # Calculate duration
        start = datetime.strptime(
            f"{self.current_session.date} {self.current_session.start_time}",
            '%Y-%m-%d %H:%M:%S'
        )
        end = datetime.strptime(
            f"{self.current_session.date} {self.current_session.end_time}",
            '%Y-%m-%d %H:%M:%S'
        )
        self.current_session.duration_minutes = (end - start).seconds / 60

        # Calculate overall form score
        if self.current_session.exercises:
            form_scores = [ex.avg_form_score for ex in self.current_session.exercises]
            self.current_session.overall_form_score = np.mean(form_scores)

        self.current_session.therapist_notes = therapist_notes

    def save_session(self, format: str = "json") -> str:
        """
        Save the current session to disk.

        Args:
            format: Output format ("json" or "csv")

        Returns:
            Path to the saved file

        HIPAA Note:
        -----------
        In production, this data should be encrypted before writing to disk.
        Consider using libraries like 'cryptography' for AES encryption.
        """
        if self.current_session is None:
            raise ValueError("No session to save.")

        # Add to history
        self.session_history.append(self.current_session)

        if format == "json":
            filepath = self._save_json()
        elif format == "csv":
            filepath = self._save_csv()
        else:
            raise ValueError(f"Unknown format: {format}")

        # Clear current session
        saved_session = self.current_session
        self.current_session = None

        return str(filepath)

    def _save_json(self) -> Path:
        """Save session as JSON file."""
        filepath = self.data_dir / f"{self.current_session.session_id}.json"

        # Convert to dictionary
        data = {
            'session_id': self.current_session.session_id,
            'patient_id': self.current_session.patient_id,
            'date': self.current_session.date,
            'start_time': self.current_session.start_time,
            'end_time': self.current_session.end_time,
            'duration_minutes': self.current_session.duration_minutes,
            'exercises': [asdict(ex) for ex in self.current_session.exercises],
            'overall_form_score': self.current_session.overall_form_score,
            'therapist_notes': self.current_session.therapist_notes,
            'flags_for_review': self.current_session.flags_for_review,
            'metadata': self.current_session.metadata
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)

        return filepath

    def _save_csv(self) -> Path:
        """Save session as CSV file."""
        filepath = self.data_dir / f"{self.current_session.session_id}.csv"

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header section
            writer.writerow(['PT Session Report'])
            writer.writerow(['Session ID', self.current_session.session_id])
            writer.writerow(['Patient ID', self.current_session.patient_id])
            writer.writerow(['Date', self.current_session.date])
            writer.writerow(['Duration (min)', f"{self.current_session.duration_minutes:.1f}"])
            writer.writerow(['Overall Form Score', f"{self.current_session.overall_form_score:.1f}%"])
            writer.writerow([])

            # Exercise details
            writer.writerow([
                'Exercise', 'Reps', 'Target', 'Form Score',
                'ROM Min', 'ROM Max', 'ROM Range', 'Avg Rep Time'
            ])

            for ex in self.current_session.exercises:
                writer.writerow([
                    ex.exercise_name,
                    ex.reps_completed,
                    ex.target_reps,
                    f"{ex.avg_form_score:.1f}%",
                    f"{ex.min_rom:.1f}°",
                    f"{ex.max_rom:.1f}°",
                    f"{ex.rom_range:.1f}°",
                    f"{ex.avg_rep_duration:.2f}s"
                ])

            writer.writerow([])

            # Flags and notes
            if self.current_session.flags_for_review:
                writer.writerow(['Flags for Review'])
                for flag in self.current_session.flags_for_review:
                    writer.writerow([flag])
                writer.writerow([])

            if self.current_session.therapist_notes:
                writer.writerow(['Therapist Notes'])
                writer.writerow([self.current_session.therapist_notes])

        return filepath

    def get_session_by_id(self, session_id: str) -> Optional[SessionData]:
        """Retrieve a specific session by ID."""
        for session in self.session_history:
            if session.session_id == session_id:
                return session
        return None

    def get_recent_sessions(self, count: int = 5) -> List[SessionData]:
        """Get the most recent sessions."""
        return self.session_history[-count:]

    # =========================================================================
    # PROGRESS TRACKING METHODS
    # =========================================================================

    def calculate_progress(self, exercise_name: str,
                          days_back: int = 30) -> Optional[ProgressMetrics]:
        """
        Calculate progress metrics for a specific exercise.

        Args:
            exercise_name: Name of exercise to analyze
            days_back: How many days of history to include

        Returns:
            ProgressMetrics object or None if insufficient data
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)

        # Collect data points
        dates = []
        rom_values = []
        form_scores = []
        reps_values = []

        for session in self.session_history:
            session_date = datetime.strptime(session.date, '%Y-%m-%d')
            if session_date < cutoff_date:
                continue

            for exercise in session.exercises:
                if exercise.exercise_name.lower() == exercise_name.lower():
                    dates.append(session.date)
                    rom_values.append(exercise.rom_range)
                    form_scores.append(exercise.avg_form_score)
                    reps_values.append(exercise.reps_completed)

        if len(dates) < 2:
            return None

        # Calculate trends
        rom_trend, rom_change = self._calculate_trend(rom_values)
        form_trend, form_change = self._calculate_trend(form_scores)
        reps_trend, reps_change = self._calculate_trend(reps_values)

        return ProgressMetrics(
            exercise_name=exercise_name,
            sessions_count=len(dates),
            rom_trend=rom_trend,
            rom_change_percent=rom_change,
            form_score_trend=form_trend,
            form_score_change=form_change,
            reps_trend=reps_trend,
            avg_reps_change=reps_change,
            dates=dates,
            rom_values=rom_values,
            form_scores=form_scores
        )

    def _calculate_trend(self, values: List[float]) -> Tuple[str, float]:
        """
        Calculate trend direction and magnitude.

        Args:
            values: List of values over time

        Returns:
            Tuple of (trend_direction, change_percent)
        """
        if len(values) < 2:
            return "stable", 0.0

        # Calculate linear regression slope
        x = np.arange(len(values))
        y = np.array(values)

        # Simple linear regression
        slope = np.polyfit(x, y, 1)[0]

        # Calculate percent change from first to last
        if values[0] != 0:
            percent_change = ((values[-1] - values[0]) / values[0]) * 100
        else:
            percent_change = 0

        # Determine trend direction
        if slope > 0.5:
            trend = "improving"
        elif slope < -0.5:
            trend = "declining"
        else:
            trend = "stable"

        return trend, round(percent_change, 1)

    def compare_sessions(self, session_id1: str, session_id2: str) -> Dict[str, Any]:
        """
        Compare two sessions side by side.

        Args:
            session_id1: First session ID (typically older)
            session_id2: Second session ID (typically newer)

        Returns:
            Dictionary with comparison data
        """
        s1 = self.get_session_by_id(session_id1)
        s2 = self.get_session_by_id(session_id2)

        if s1 is None or s2 is None:
            return {"error": "One or both sessions not found"}

        comparison = {
            "session1": {
                "id": s1.session_id,
                "date": s1.date,
                "overall_score": s1.overall_form_score
            },
            "session2": {
                "id": s2.session_id,
                "date": s2.date,
                "overall_score": s2.overall_form_score
            },
            "score_change": s2.overall_form_score - s1.overall_form_score,
            "exercises": {}
        }

        # Compare exercises
        s1_exercises = {ex.exercise_name: ex for ex in s1.exercises}
        s2_exercises = {ex.exercise_name: ex for ex in s2.exercises}

        all_exercises = set(s1_exercises.keys()) | set(s2_exercises.keys())

        for ex_name in all_exercises:
            ex1 = s1_exercises.get(ex_name)
            ex2 = s2_exercises.get(ex_name)

            comparison["exercises"][ex_name] = {
                "session1_rom": ex1.rom_range if ex1 else None,
                "session2_rom": ex2.rom_range if ex2 else None,
                "rom_change": (ex2.rom_range - ex1.rom_range) if (ex1 and ex2) else None,
                "session1_form": ex1.avg_form_score if ex1 else None,
                "session2_form": ex2.avg_form_score if ex2 else None,
                "form_change": (ex2.avg_form_score - ex1.avg_form_score) if (ex1 and ex2) else None
            }

        return comparison

    def get_exercises_needing_attention(self) -> List[Dict[str, Any]]:
        """
        Identify exercises that need extra focus based on recent performance.

        Criteria for attention:
        - Form score declining
        - ROM not improving
        - Frequently flagged in sessions

        Returns:
            List of exercises with concern details
        """
        concerns = []

        # Analyze each exercise
        exercise_names = set()
        for session in self.session_history:
            for ex in session.exercises:
                exercise_names.add(ex.exercise_name)

        for ex_name in exercise_names:
            progress = self.calculate_progress(ex_name, days_back=30)

            if progress is None:
                continue

            attention_reasons = []

            if progress.form_score_trend == "declining":
                attention_reasons.append(
                    f"Form score declining ({progress.form_score_change:+.1f}%)"
                )

            if progress.rom_trend == "declining":
                attention_reasons.append(
                    f"ROM declining ({progress.rom_change_percent:+.1f}%)"
                )

            # Check for frequent form warnings
            warning_count = 0
            for session in self.session_history[-5:]:
                for ex in session.exercises:
                    if ex.exercise_name == ex_name and ex.form_warnings:
                        warning_count += len(ex.form_warnings)

            if warning_count >= 3:
                attention_reasons.append(f"{warning_count} form warnings in recent sessions")

            if attention_reasons:
                concerns.append({
                    "exercise": ex_name,
                    "reasons": attention_reasons,
                    "current_form_score": progress.form_scores[-1] if progress.form_scores else 0,
                    "current_rom": progress.rom_values[-1] if progress.rom_values else 0
                })

        # Sort by number of concerns
        concerns.sort(key=lambda x: len(x["reasons"]), reverse=True)

        return concerns

    # =========================================================================
    # VISUALIZATION METHODS
    # =========================================================================

    def plot_progress(self, exercise_name: str, save_path: Optional[str] = None):
        """
        Generate a progress chart for an exercise.

        Creates a dual-axis plot showing ROM and form score trends.

        Args:
            exercise_name: Exercise to plot
            save_path: Optional path to save the figure

        Returns:
            matplotlib figure or None if matplotlib unavailable
        """
        if not MATPLOTLIB_AVAILABLE:
            print("matplotlib not available. Install with: pip install matplotlib")
            return None

        progress = self.calculate_progress(exercise_name, days_back=90)
        if progress is None:
            print(f"Insufficient data for {exercise_name}")
            return None

        # Convert dates
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in progress.dates]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # ROM plot (left axis)
        color1 = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Range of Motion (degrees)', color=color1)
        ax1.plot(dates, progress.rom_values, 'o-', color=color1, label='ROM')
        ax1.tick_params(axis='y', labelcolor=color1)

        # Add trend line for ROM
        z = np.polyfit(range(len(dates)), progress.rom_values, 1)
        p = np.poly1d(z)
        ax1.plot(dates, p(range(len(dates))), '--', color=color1, alpha=0.5)

        # Form score plot (right axis)
        ax2 = ax1.twinx()
        color2 = 'tab:green'
        ax2.set_ylabel('Form Score (%)', color=color2)
        ax2.plot(dates, progress.form_scores, 's-', color=color2, label='Form Score')
        ax2.tick_params(axis='y', labelcolor=color2)
        ax2.set_ylim(0, 100)

        # Title and formatting
        plt.title(f'{exercise_name} Progress\n'
                  f'ROM: {progress.rom_trend} ({progress.rom_change_percent:+.1f}%) | '
                  f'Form: {progress.form_score_trend} ({progress.form_score_change:+.1f}%)')

        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        plt.xticks(rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Saved progress chart to {save_path}")

        return fig

    def generate_session_summary_card(self, session: Optional[SessionData] = None) -> str:
        """
        Generate a text-based summary card for a session.

        Args:
            session: Session to summarize (defaults to current/most recent)

        Returns:
            Formatted text summary
        """
        if session is None:
            session = self.current_session or (
                self.session_history[-1] if self.session_history else None
            )

        if session is None:
            return "No session data available"

        lines = [
            "=" * 50,
            "PT SESSION SUMMARY",
            "=" * 50,
            f"Date: {session.date}",
            f"Duration: {session.duration_minutes:.0f} minutes",
            f"Overall Form Score: {session.overall_form_score:.0f}%",
            "",
            "EXERCISES:",
            "-" * 30
        ]

        for ex in session.exercises:
            completion = (ex.reps_completed / ex.target_reps * 100) if ex.target_reps > 0 else 0
            lines.append(
                f"  {ex.exercise_name}:\n"
                f"    Reps: {ex.reps_completed}/{ex.target_reps} ({completion:.0f}%)\n"
                f"    Form: {ex.avg_form_score:.0f}% | ROM: {ex.rom_range:.0f}°"
            )

        if session.flags_for_review:
            lines.extend([
                "",
                "FLAGS FOR REVIEW:",
                "-" * 30
            ])
            for flag in session.flags_for_review:
                lines.append(f"  ⚠ {flag}")

        if session.therapist_notes:
            lines.extend([
                "",
                "NOTES:",
                "-" * 30,
                f"  {session.therapist_notes}"
            ])

        lines.append("=" * 50)

        return "\n".join(lines)

    # =========================================================================
    # EXPORT METHODS
    # =========================================================================

    def export_patient_history(self, format: str = "csv") -> str:
        """
        Export all patient session history to a single file.

        Args:
            format: "csv" or "json"

        Returns:
            Path to exported file
        """
        filename = f"{self.patient_id}_history_{datetime.now().strftime('%Y%m%d')}"

        if format == "csv":
            filepath = self.data_dir / f"{filename}.csv"
            self._export_history_csv(filepath)
        else:
            filepath = self.data_dir / f"{filename}.json"
            self._export_history_json(filepath)

        return str(filepath)

    def _export_history_csv(self, filepath: Path):
        """Export all sessions to a single CSV."""
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'Session ID', 'Date', 'Duration (min)', 'Exercise',
                'Reps', 'Target Reps', 'Form Score', 'ROM Min',
                'ROM Max', 'ROM Range', 'Avg Rep Time'
            ])

            for session in self.session_history:
                for ex in session.exercises:
                    writer.writerow([
                        session.session_id,
                        session.date,
                        f"{session.duration_minutes:.1f}",
                        ex.exercise_name,
                        ex.reps_completed,
                        ex.target_reps,
                        f"{ex.avg_form_score:.1f}",
                        f"{ex.min_rom:.1f}",
                        f"{ex.max_rom:.1f}",
                        f"{ex.rom_range:.1f}",
                        f"{ex.avg_rep_duration:.2f}"
                    ])

    def _export_history_json(self, filepath: Path):
        """Export all sessions to a single JSON."""
        data = {
            "patient_id": self.patient_id,
            "export_date": datetime.now().isoformat(),
            "total_sessions": len(self.session_history),
            "sessions": []
        }

        for session in self.session_history:
            session_data = {
                'session_id': session.session_id,
                'date': session.date,
                'duration_minutes': session.duration_minutes,
                'overall_form_score': session.overall_form_score,
                'exercises': [asdict(ex) for ex in session.exercises],
                'flags_for_review': session.flags_for_review,
                'therapist_notes': session.therapist_notes
            }
            data["sessions"].append(session_data)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_exercise_record_from_tracker(exercise_name: str,
                                        tracker_session: Any) -> ExerciseRecord:
    """
    Create an ExerciseRecord from an ExerciseTracker session.

    This bridges the analyze.py ExerciseSession with the session tracker.

    Args:
        exercise_name: Name of the exercise
        tracker_session: ExerciseSession object from analyze.py

    Returns:
        ExerciseRecord for session tracking
    """
    # Extract metrics from tracker session
    rep_metrics = tracker_session.rep_metrics if hasattr(tracker_session, 'rep_metrics') else []

    if rep_metrics:
        min_roms = [r.min_angle for r in rep_metrics]
        max_roms = [r.max_angle for r in rep_metrics]
        form_scores = [r.form_score for r in rep_metrics]
        durations = [r.duration_seconds for r in rep_metrics]

        return ExerciseRecord(
            exercise_name=exercise_name,
            reps_completed=len(rep_metrics),
            target_reps=tracker_session.target_reps if hasattr(tracker_session, 'target_reps') else 10,
            avg_form_score=np.mean(form_scores) if form_scores else 0,
            min_rom=min(min_roms) if min_roms else 0,
            max_rom=max(max_roms) if max_roms else 0,
            rom_range=max(max_roms) - min(min_roms) if max_roms and min_roms else 0,
            avg_rep_duration=np.mean(durations) if durations else 0,
            form_warnings=tracker_session.warnings if hasattr(tracker_session, 'warnings') else []
        )
    else:
        return ExerciseRecord(
            exercise_name=exercise_name,
            reps_completed=0,
            target_reps=10,
            avg_form_score=0,
            min_rom=0,
            max_rom=0,
            rom_range=0,
            avg_rep_duration=0
        )


def quick_session_report(session_tracker: SessionTracker) -> str:
    """
    Generate a quick text report of the current/last session.

    Args:
        session_tracker: SessionTracker instance

    Returns:
        Formatted report string
    """
    return session_tracker.generate_session_summary_card()
