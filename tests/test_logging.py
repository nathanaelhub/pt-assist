"""Tests for ObservationLogger (partial reps) and PTARawLogger."""

import importlib.util
import json
import tempfile
from pathlib import Path

# Import logger modules directly to avoid src/__init__.py pulling in
# heavy dependencies (numpy, cv2) that may not be installed in test envs.
_project = Path(__file__).parent.parent

def _import_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_obs_mod = _import_module("observation_logger", _project / "src" / "observation_logger.py")
_pta_mod = _import_module("pta_raw_logger", _project / "src" / "pta_raw_logger.py")

ObservationLogger = _obs_mod.ObservationLogger
PTARawLogger = _pta_mod.PTARawLogger


class TestObservationLoggerPartialReps:
    """Test the partial_rep event type in ObservationLogger."""

    def test_log_partial_rep_creates_event(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ObservationLogger(log_dir=tmpdir, source="video")
            logger.start_session("Squat", target_reps=10)

            logger.log_partial_rep(
                exercise="Squat",
                rom=40.2,
                min_angle=95.0,
                max_angle=135.2,
                duration_seconds=1.3,
                reason="reversed_early",
            )

            # Read back the JSON
            log_file = list(Path(tmpdir).glob("session_*.json"))[0]
            data = json.loads(log_file.read_text())

            partial_events = [e for e in data["events"] if e["type"] == "partial_rep"]
            assert len(partial_events) == 1

            p = partial_events[0]
            assert p["exercise"] == "Squat"
            assert p["rom"] == 40.2
            assert p["min_angle"] == 95.0
            assert p["max_angle"] == 135.2
            assert p["duration_seconds"] == 1.3
            assert p["reason"] == "reversed_early"
            assert "timestamp" in p

    def test_partial_rep_not_logged_when_session_inactive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ObservationLogger(log_dir=tmpdir, source="video")
            # Don't start a session
            logger.log_partial_rep(
                exercise="Squat", rom=40.0,
                min_angle=90.0, max_angle=130.0,
                duration_seconds=1.0,
            )
            assert logger.file_path is None

    def test_multiple_partial_and_complete_reps(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = ObservationLogger(log_dir=tmpdir, source="webcam")
            logger.start_session("Knee Extension", target_reps=5)

            # Log a partial rep
            logger.log_partial_rep(
                exercise="Knee Extension", rom=20.0,
                min_angle=80.0, max_angle=100.0,
                duration_seconds=0.8, reason="insufficient_rom",
            )

            # Log a completed rep
            logger.log_rep(
                rep_number=1, exercise="Knee Extension",
                rom=75.0, form_score=85.0, duration_seconds=2.1,
                min_angle=92.3, max_angle=172.8,
            )

            # Log another partial rep
            logger.log_partial_rep(
                exercise="Knee Extension", rom=30.0,
                min_angle=85.0, max_angle=115.0,
                duration_seconds=1.1, reason="interrupted",
            )

            logger.end_session(total_reps=1, avg_form_score=85.0, avg_rom=75.0)

            log_file = list(Path(tmpdir).glob("session_*.json"))[0]
            data = json.loads(log_file.read_text())

            event_types = [e["type"] for e in data["events"]]
            assert event_types.count("partial_rep") == 2
            assert event_types.count("rep") == 1
            # Verify order: session_start, partial, rep, partial, session_end
            assert event_types == [
                "session_start", "partial_rep", "rep",
                "partial_rep", "session_end",
            ]


class TestPTARawLogger:
    """Test the pipe-delimited raw text PTA logger."""

    def test_creates_log_file_with_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="webcam")
            filename = logger.start_session("Elbow Flexion")

            assert filename == "PTA-Log-0001.txt"
            log_file = Path(tmpdir) / filename
            assert log_file.exists()

            content = log_file.read_text()
            assert "PTA Observation Log" in content
            assert "PTA-Log-0001.txt" in content
            assert "Version: 1.00" in content
            assert "************" in content
            assert "SYSTEM" in content
            assert "Camera Detected." in content

    def test_auto_increments_log_number(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create first log
            logger1 = PTARawLogger(log_dir=tmpdir, source="video")
            name1 = logger1.start_session("Squat")
            logger1.end_session()

            # Create second log
            logger2 = PTARawLogger(log_dir=tmpdir, source="video")
            name2 = logger2.start_session("Squat")
            logger2.end_session()

            assert name1 == "PTA-Log-0001.txt"
            assert name2 == "PTA-Log-0002.txt"

    def test_log_observation_writes_pipe_delimited(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="webcam")
            logger.start_session("Elbow Flexion")

            logger.log_observation(
                exercise="Elbow Flexion",
                min_angle=0.0, max_angle=180.0,
                angle=120.5, state="DOWN", count=3,
            )

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            lines = content.strip().split("\n")
            obs_line = [l for l in lines if "Observing" in l][0]

            assert " | " in obs_line
            assert "LOG" in obs_line
            assert "Observing Elbow Flexion" in obs_line
            assert "angle=120.5" in obs_line
            assert "state=DOWN" in obs_line
            assert "count=3" in obs_line

    def test_log_rep_completed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="video")
            logger.start_session("Squat")

            logger.log_rep_completed("Squat", rep_count=1)

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            assert "Rep Completed" in content
            assert 'exercise="Squat"' in content
            assert "repCount=1" in content

    def test_log_partial_rep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="video")
            logger.start_session("Knee Extension")

            logger.log_partial_rep("Knee Extension", rom=30.5, reason="reversed_early")

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            assert "Partial Rep" in content
            assert "rom=30.5" in content
            assert 'reason="reversed_early"' in content

    def test_log_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="webcam")
            logger.start_session("Squat")

            logger.log_error("OpenCV: Camera[0] no frame buffer readable.")

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            assert "ERROR" in content
            assert "Camera[0] no frame buffer readable" in content

    def test_log_exercise_change(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="webcam")
            logger.start_session("Squat")

            logger.log_exercise_change("Squat", "Knee Extension")

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            assert "Exercise Changed" in content
            assert 'from="Squat"' in content
            assert 'to="Knee Extension"' in content

    def test_end_session_writes_summary_and_halt(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="video")
            logger.start_session("Squat")

            result = logger.end_session(total_reps=5, avg_form_score=82.5, avg_rom=78.3)

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            assert "Session Summary" in content
            assert "totalReps=5" in content
            assert "System halted." in content
            assert "PTA-Log-0001.txt" in result

    def test_full_session_flow(self):
        """Test a complete session matching the reference PTA-Log-0001 format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = PTARawLogger(log_dir=tmpdir, source="webcam")
            logger.start_session("Elbow Flexion")

            # Simulate a few observations
            logger.log_observation("Elbow Flexion", 0.0, 180.0, 3.5, "DOWN", 0)
            logger.log_observation("Elbow Flexion", 0.0, 180.0, 120.0, "DOWN", 0)
            logger.log_observation("Elbow Flexion", 0.0, 180.0, 185.3, "UP", 0)
            logger.log_observation("Elbow Flexion", 0.0, 180.0, 62.3, "UP", 0)
            logger.log_observation("Elbow Flexion", 0.0, 180.0, -0.1, "DOWN", 1)
            logger.log_rep_completed("Elbow Flexion", 1)

            # Partial rep
            logger.log_partial_rep("Elbow Flexion", rom=25.0, reason="reversed_early")

            # Error
            logger.log_error("OpenCV: Camera[0] no frame buffer readable.")

            logger.end_session(total_reps=1, avg_form_score=80.0, avg_rom=75.0)

            content = Path(tmpdir, "PTA-Log-0001.txt").read_text()
            lines = [l for l in content.strip().split("\n") if l.strip()]

            # Header check
            assert lines[0] == "PTA Observation Log"
            assert "Version: 1.00" in content

            # Count pipe-delimited data lines (after header)
            data_lines = [l for l in lines if " | " in l]
            # Should have: 2 SYSTEM startup + exercise selected + 6 observations
            # + rep completed + partial rep + error + session summary + system halted
            assert len(data_lines) >= 10
