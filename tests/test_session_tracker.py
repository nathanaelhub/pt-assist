"""
Tests for Session Tracking Module
==================================

Tests data persistence, progress calculations, and export formats.

Run with:
    pytest tests/test_session_tracker.py -v
"""

import pytest
import json
import csv
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.session_tracker import (
    SessionTracker,
    SessionData,
    ExerciseRecord,
    ProgressMetrics,
    create_exercise_record_from_tracker,
    quick_session_report,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def session_tracker(temp_data_dir):
    """Create a SessionTracker with temporary storage."""
    return SessionTracker(data_dir=temp_data_dir, patient_id="test_patient")


@pytest.fixture
def sample_exercise_record():
    """Create a sample exercise record."""
    return ExerciseRecord(
        exercise_name="Knee Extension",
        reps_completed=10,
        target_reps=10,
        avg_form_score=85.0,
        min_rom=90.0,
        max_rom=165.0,
        rom_range=75.0,
        avg_rep_duration=2.5,
        form_warnings=["Minor hip movement"],
        notes="Good session"
    )


@pytest.fixture
def multiple_sessions(temp_data_dir):
    """Create a tracker with multiple sessions for progress testing."""
    tracker = SessionTracker(data_dir=temp_data_dir, patient_id="progress_test")

    # Create several sessions over time
    dates = [
        datetime.now() - timedelta(days=20),
        datetime.now() - timedelta(days=15),
        datetime.now() - timedelta(days=10),
        datetime.now() - timedelta(days=5),
        datetime.now(),
    ]

    # ROM improves over time: 70 -> 75 -> 78 -> 82 -> 85
    roms = [70, 75, 78, 82, 85]
    # Form scores: 75 -> 78 -> 80 -> 82 -> 85
    forms = [75, 78, 80, 82, 85]

    for i, (date, rom, form) in enumerate(zip(dates, roms, forms)):
        session = SessionData(
            session_id=f"progress_test_{date.strftime('%Y%m%d_%H%M%S')}",
            patient_id="progress_test",
            date=date.strftime('%Y-%m-%d'),
            start_time="10:00:00",
            end_time="10:30:00",
            duration_minutes=30,
            exercises=[
                ExerciseRecord(
                    exercise_name="Knee Extension",
                    reps_completed=10,
                    target_reps=10,
                    avg_form_score=form,
                    min_rom=90.0,
                    max_rom=90.0 + rom,
                    rom_range=rom,
                    avg_rep_duration=2.5
                )
            ],
            overall_form_score=form
        )
        tracker.session_history.append(session)

    return tracker


# =============================================================================
# SESSION TRACKER INITIALIZATION TESTS
# =============================================================================

class TestSessionTrackerInit:
    """Tests for SessionTracker initialization."""

    def test_tracker_creation(self, temp_data_dir):
        """Test basic tracker creation."""
        tracker = SessionTracker(data_dir=temp_data_dir, patient_id="test123")

        assert tracker.patient_id == "test123"
        assert tracker.data_dir == Path(temp_data_dir)
        assert tracker.current_session is None
        assert len(tracker.session_history) == 0

    def test_data_dir_creation(self, temp_data_dir):
        """Test that data directory is created if it doesn't exist."""
        new_dir = Path(temp_data_dir) / "new_subdir"
        tracker = SessionTracker(data_dir=str(new_dir), patient_id="test")

        assert new_dir.exists()


# =============================================================================
# SESSION LIFECYCLE TESTS
# =============================================================================

class TestSessionLifecycle:
    """Tests for session start, update, and end."""

    def test_start_session(self, session_tracker):
        """Test starting a new session."""
        session_id = session_tracker.start_session()

        assert session_tracker.current_session is not None
        assert "test_patient" in session_id
        assert session_tracker.current_session.patient_id == "test_patient"
        assert session_tracker.current_session.date == datetime.now().strftime('%Y-%m-%d')

    def test_start_session_with_metadata(self, session_tracker):
        """Test starting session with metadata."""
        metadata = {"location": "home", "equipment": "resistance_band"}
        session_id = session_tracker.start_session(metadata=metadata)

        assert session_tracker.current_session.metadata == metadata

    def test_add_exercise_record(self, session_tracker, sample_exercise_record):
        """Test adding exercise record to session."""
        session_tracker.start_session()
        session_tracker.add_exercise_record(sample_exercise_record)

        assert len(session_tracker.current_session.exercises) == 1
        assert session_tracker.current_session.exercises[0].exercise_name == "Knee Extension"

    def test_add_exercise_without_session_raises(self, session_tracker, sample_exercise_record):
        """Should raise error if no active session."""
        with pytest.raises(ValueError, match="No active session"):
            session_tracker.add_exercise_record(sample_exercise_record)

    def test_end_session(self, session_tracker, sample_exercise_record):
        """Test ending a session."""
        session_tracker.start_session()
        session_tracker.add_exercise_record(sample_exercise_record)
        session_tracker.end_session(therapist_notes="Good progress today")

        assert session_tracker.current_session.end_time != ""
        assert session_tracker.current_session.duration_minutes > 0
        assert session_tracker.current_session.therapist_notes == "Good progress today"
        assert session_tracker.current_session.overall_form_score > 0

    def test_end_session_without_start_raises(self, session_tracker):
        """Should raise error if no session to end."""
        with pytest.raises(ValueError, match="No active session"):
            session_tracker.end_session()


# =============================================================================
# DATA PERSISTENCE TESTS
# =============================================================================

class TestDataPersistence:
    """Tests for saving and loading session data."""

    def test_save_session_json(self, session_tracker, sample_exercise_record):
        """Test saving session as JSON."""
        session_tracker.start_session()
        session_tracker.add_exercise_record(sample_exercise_record)
        session_tracker.end_session()

        filepath = session_tracker.save_session(format="json")

        assert Path(filepath).exists()
        assert filepath.endswith(".json")

        # Verify content
        with open(filepath, 'r') as f:
            data = json.load(f)
            assert data['patient_id'] == "test_patient"
            assert len(data['exercises']) == 1

    def test_save_session_csv(self, session_tracker, sample_exercise_record):
        """Test saving session as CSV."""
        session_tracker.start_session()
        session_tracker.add_exercise_record(sample_exercise_record)
        session_tracker.end_session()

        filepath = session_tracker.save_session(format="csv")

        assert Path(filepath).exists()
        assert filepath.endswith(".csv")

        # Verify content
        with open(filepath, 'r') as f:
            content = f.read()
            assert "Knee Extension" in content
            assert "test_patient" in content

    def test_save_without_session_raises(self, session_tracker):
        """Should raise error if no session to save."""
        with pytest.raises(ValueError, match="No session to save"):
            session_tracker.save_session()

    def test_load_saved_session(self, temp_data_dir, sample_exercise_record):
        """Test that saved sessions are loaded on init."""
        # Create and save a session
        tracker1 = SessionTracker(data_dir=temp_data_dir, patient_id="load_test")
        tracker1.start_session()
        tracker1.add_exercise_record(sample_exercise_record)
        tracker1.end_session()
        tracker1.save_session()

        # Create new tracker - should load saved session
        tracker2 = SessionTracker(data_dir=temp_data_dir, patient_id="load_test")

        assert len(tracker2.session_history) == 1
        assert tracker2.session_history[0].patient_id == "load_test"


class TestSessionRetrieval:
    """Tests for retrieving session data."""

    def test_get_session_by_id(self, session_tracker, sample_exercise_record):
        """Test retrieving session by ID."""
        session_tracker.start_session()
        session_id = session_tracker.current_session.session_id
        session_tracker.add_exercise_record(sample_exercise_record)
        session_tracker.end_session()
        session_tracker.save_session()

        session = session_tracker.get_session_by_id(session_id)

        assert session is not None
        assert session.session_id == session_id

    def test_get_nonexistent_session(self, session_tracker):
        """Should return None for nonexistent session."""
        session = session_tracker.get_session_by_id("nonexistent_id")
        assert session is None

    def test_get_recent_sessions(self, multiple_sessions):
        """Test getting recent sessions."""
        recent = multiple_sessions.get_recent_sessions(count=3)

        assert len(recent) == 3
        # Should be most recent (last in history)
        assert recent[-1] == multiple_sessions.session_history[-1]


# =============================================================================
# PROGRESS TRACKING TESTS
# =============================================================================

class TestProgressTracking:
    """Tests for progress calculation."""

    def test_calculate_progress(self, multiple_sessions):
        """Test progress calculation over time."""
        progress = multiple_sessions.calculate_progress("Knee Extension", days_back=30)

        assert progress is not None
        assert progress.exercise_name == "Knee Extension"
        assert progress.sessions_count == 5

    def test_progress_rom_trend(self, multiple_sessions):
        """ROM should show improving trend."""
        progress = multiple_sessions.calculate_progress("Knee Extension", days_back=30)

        assert progress.rom_trend == "improving"
        assert progress.rom_change_percent > 0

    def test_progress_form_score_trend(self, multiple_sessions):
        """Form score should show improving trend."""
        progress = multiple_sessions.calculate_progress("Knee Extension", days_back=30)

        assert progress.form_score_trend == "improving"
        assert progress.form_score_change > 0

    def test_progress_insufficient_data(self, session_tracker):
        """Should return None with insufficient data."""
        progress = session_tracker.calculate_progress("Knee Extension", days_back=30)
        assert progress is None

    def test_compare_sessions(self, multiple_sessions):
        """Test comparing two sessions."""
        sessions = multiple_sessions.session_history
        comparison = multiple_sessions.compare_sessions(
            sessions[0].session_id,
            sessions[-1].session_id
        )

        assert "session1" in comparison
        assert "session2" in comparison
        assert "score_change" in comparison
        assert "exercises" in comparison

    def test_exercises_needing_attention(self, temp_data_dir):
        """Test identifying exercises that need attention."""
        tracker = SessionTracker(data_dir=temp_data_dir, patient_id="attention_test")

        # Create sessions with declining form
        for i in range(5):
            date = datetime.now() - timedelta(days=i)
            session = SessionData(
                session_id=f"attention_{i}",
                patient_id="attention_test",
                date=date.strftime('%Y-%m-%d'),
                start_time="10:00:00",
                end_time="10:30:00",
                duration_minutes=30,
                exercises=[
                    ExerciseRecord(
                        exercise_name="Problem Exercise",
                        reps_completed=8,
                        target_reps=10,
                        avg_form_score=80 - i * 5,  # Declining: 80, 75, 70, 65, 60
                        min_rom=90,
                        max_rom=150,
                        rom_range=60,
                        avg_rep_duration=2.5,
                        form_warnings=["Issue " + str(i)]
                    )
                ],
                overall_form_score=80 - i * 5
            )
            tracker.session_history.append(session)

        concerns = tracker.get_exercises_needing_attention()

        # Should identify the declining exercise
        assert len(concerns) > 0
        assert any(c["exercise"] == "Problem Exercise" for c in concerns)


# =============================================================================
# EXPORT TESTS
# =============================================================================

class TestExport:
    """Tests for data export functionality."""

    def test_export_history_csv(self, multiple_sessions):
        """Test exporting full history as CSV."""
        filepath = multiple_sessions.export_patient_history(format="csv")

        assert Path(filepath).exists()
        assert filepath.endswith(".csv")

        with open(filepath, 'r') as f:
            content = f.read()
            assert "Knee Extension" in content

    def test_export_history_json(self, multiple_sessions):
        """Test exporting full history as JSON."""
        filepath = multiple_sessions.export_patient_history(format="json")

        assert Path(filepath).exists()

        with open(filepath, 'r') as f:
            data = json.load(f)
            assert data['patient_id'] == "progress_test"
            assert data['total_sessions'] == 5


# =============================================================================
# VISUALIZATION TESTS
# =============================================================================

class TestVisualization:
    """Tests for visualization methods."""

    def test_generate_summary_card(self, session_tracker, sample_exercise_record):
        """Test generating session summary card."""
        session_tracker.start_session()
        session_tracker.add_exercise_record(sample_exercise_record)
        session_tracker.end_session()

        summary = session_tracker.generate_session_summary_card()

        assert "PT SESSION SUMMARY" in summary
        assert "Knee Extension" in summary
        assert "85" in summary  # form score

    def test_summary_card_no_session(self, session_tracker):
        """Should handle no session gracefully."""
        summary = session_tracker.generate_session_summary_card()
        assert "No session data available" in summary


# =============================================================================
# FLAG GENERATION TESTS
# =============================================================================

class TestFlagGeneration:
    """Tests for automatic flag generation."""

    def test_low_form_score_flag(self, session_tracker):
        """Low form score should generate flag."""
        session_tracker.start_session()

        low_form_record = ExerciseRecord(
            exercise_name="Test Exercise",
            reps_completed=10,
            target_reps=10,
            avg_form_score=50.0,  # Low form score
            min_rom=90,
            max_rom=150,
            rom_range=60,
            avg_rep_duration=2.5
        )
        session_tracker.add_exercise_record(low_form_record)

        # Should have generated a flag
        assert len(session_tracker.current_session.flags_for_review) > 0
        assert any("Low form score" in flag for flag in session_tracker.current_session.flags_for_review)

    def test_low_rep_completion_flag(self, session_tracker):
        """Low rep completion should generate flag."""
        session_tracker.start_session()

        low_reps_record = ExerciseRecord(
            exercise_name="Test Exercise",
            reps_completed=3,  # Only 3 of 10
            target_reps=10,
            avg_form_score=85.0,
            min_rom=90,
            max_rom=150,
            rom_range=60,
            avg_rep_duration=2.5
        )
        session_tracker.add_exercise_record(low_reps_record)

        assert len(session_tracker.current_session.flags_for_review) > 0
        assert any("3/10" in flag for flag in session_tracker.current_session.flags_for_review)


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================

class TestHelperFunctions:
    """Tests for helper functions."""

    def test_create_exercise_record_from_tracker(self):
        """Test converting tracker session to exercise record."""
        # Create a mock tracker session
        class MockRepMetrics:
            def __init__(self, min_a, max_a, form, dur):
                self.min_angle = min_a
                self.max_angle = max_a
                self.form_score = form
                self.duration_seconds = dur

        class MockTrackerSession:
            def __init__(self):
                self.rep_metrics = [
                    MockRepMetrics(90, 160, 85, 2.1),
                    MockRepMetrics(92, 165, 82, 2.3),
                    MockRepMetrics(88, 162, 88, 2.0),
                ]
                self.target_reps = 10
                self.warnings = ["Test warning"]

        mock_session = MockTrackerSession()
        record = create_exercise_record_from_tracker("Knee Extension", mock_session)

        assert record.exercise_name == "Knee Extension"
        assert record.reps_completed == 3
        assert record.avg_form_score == 85.0  # Average of 85, 82, 88
        assert record.target_reps == 10

    def test_quick_session_report(self, session_tracker, sample_exercise_record):
        """Test quick report generation."""
        session_tracker.start_session()
        session_tracker.add_exercise_record(sample_exercise_record)
        session_tracker.end_session()

        report = quick_session_report(session_tracker)

        assert "PT SESSION SUMMARY" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
