"""
PTA Raw Text Logger for AI Physical Therapy Assistant
======================================================

Creates pipe-delimited raw text observation log files for each PT session,
following the PTA-Log format specification.

Each session generates a new text file in the pta_logs/ folder with
auto-incrementing filenames: PTA-Log-0001.txt, PTA-Log-0002.txt, etc.

Format:
    Timestamp | Message Type | Message Text | [Data CSV of key=value pairs]

    Timestamp is in YYYY:MM:DD:HH:MM:SS:sss format
    Message Types are: SYSTEM, LOG, ERROR
    Fields are separated by '|' character
    Fields marked with [] are optional

Usage:
    from src.pta_raw_logger import PTARawLogger

    pta = PTARawLogger(source="webcam")
    pta.start_session("Elbow Flexion")

    # Log every frame observation:
    pta.log_observation("Elbow Flexion", min_angle=0.0, max_angle=180.0,
                        angle=120.5, state="DOWN", count=3)

    # Log a completed rep:
    pta.log_rep_completed("Elbow Flexion", rep_count=4)

    # Log a partial rep:
    pta.log_partial_rep("Elbow Flexion", rom=40.2, reason="reversed_early")

    # Log errors:
    pta.log_error("OpenCV: Camera[0] no frame buffer readable.")

    pta.end_session()
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


def _pta_timestamp() -> str:
    """Return current time in YYYY:MM:DD:HH:MM:SS:sss format."""
    now = datetime.now()
    return now.strftime("%Y:%m:%d:%H:%M:%S:") + f"{now.microsecond // 1000:03d}"


class PTARawLogger:
    """
    Pipe-delimited raw text logger for PT sessions.

    Creates one text file per session in the pta_logs/ directory,
    with auto-incrementing filenames (PTA-Log-0001.txt, PTA-Log-0002.txt, ...).

    Each line is written immediately (append mode) so partial sessions
    are preserved if the process crashes.
    """

    VERSION = "1.00"

    def __init__(self, log_dir: Optional[str] = None, source: str = "video"):
        """
        Initialize the PTA raw logger.

        Args:
            log_dir: Directory to store logs. Defaults to pta_logs/
                     inside the project root.
            source: "video" or "webcam"
        """
        if log_dir is None:
            project_root = Path(__file__).parent.parent
            self.log_dir = project_root / "pta_logs"
        else:
            self.log_dir = Path(log_dir)

        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.source = source
        self.file_path: Optional[Path] = None
        self._started = False
        self._log_number = 0

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def start_session(self, exercise_name: str) -> str:
        """
        Start a new session and create the log file with header.

        Args:
            exercise_name: Name of the initial exercise.

        Returns:
            The log filename (e.g. "PTA-Log-0001.txt").
        """
        self._log_number = self._next_log_number()
        filename = f"PTA-Log-{self._log_number:04d}.txt"
        self.file_path = self.log_dir / filename

        # Write header block
        start_ts = _pta_timestamp()
        header_lines = [
            "PTA Observation Log",
            filename,
            f"Version: {self.VERSION}",
            f"Start time: {start_ts}",
            "",
            "Timestamp | Message Type | Message Text | [Data CSV of key=value pairs]",
            "",
            "Timestamp is in YYYY:MM:DD:HH:MM:SS:sss format",
            "Message Types are: SYSTEM, LOG, ERROR",
            "Fields are separated by '|' character",
            "Fields marked with [] are optional",
            "************",
        ]

        with open(self.file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(header_lines) + "\n")

        self._started = True

        # Log system startup events
        self._write_line("SYSTEM", "Camera Detected.")
        self._write_line("SYSTEM", f"System started successfully. Source: {self.source}")
        self._write_line(
            "LOG",
            f"{exercise_name} exercise selected",
            f"min=0.0, max=0.0, angle=0.0, state=INIT, count=0",
        )

        return filename

    def log_observation(
        self,
        exercise: str,
        min_angle: float,
        max_angle: float,
        angle: float,
        state: str,
        count: int,
    ) -> None:
        """
        Log a single frame observation with current angle and state.

        Args:
            exercise: Current exercise name.
            min_angle: Minimum angle threshold for the exercise.
            max_angle: Maximum angle threshold for the exercise.
            angle: Current measured joint angle.
            state: Current state ("UP", "DOWN", "IDLE", "NEUTRAL").
            count: Current rep count.
        """
        if not self._started:
            return

        data = (
            f"min={min_angle:.1f}, max={max_angle:.1f}, "
            f"angle={angle:.1f}, state={state.upper()}, count={count}"
        )
        self._write_line("LOG", f"Observing {exercise}", data)

    def log_rep_completed(self, exercise: str, rep_count: int) -> None:
        """
        Log a completed repetition.

        Args:
            exercise: Exercise name.
            rep_count: Total rep count after this completion.
        """
        if not self._started:
            return

        data = f'exercise="{exercise}", repCount={rep_count}'
        self._write_line("LOG", "Rep Completed", data)

    def log_partial_rep(
        self,
        exercise: str,
        rom: float,
        reason: str = "reversed_early",
    ) -> None:
        """
        Log a partial/incomplete repetition attempt.

        Args:
            exercise: Exercise name.
            rom: Range of motion achieved.
            reason: Why the rep was not counted.
        """
        if not self._started:
            return

        data = f'exercise="{exercise}", rom={rom:.1f}, reason="{reason}"'
        self._write_line("LOG", "Partial Rep", data)

    def log_exercise_change(self, from_exercise: str, to_exercise: str) -> None:
        """Log when the user switches exercises."""
        if not self._started:
            return

        data = f'from="{from_exercise}", to="{to_exercise}"'
        self._write_line("LOG", "Exercise Changed", data)

    def log_reset(self, exercise: str, rep_count_at_reset: int) -> None:
        """Log when the rep counter is reset."""
        if not self._started:
            return

        data = f'exercise="{exercise}", repCountAtReset={rep_count_at_reset}'
        self._write_line("LOG", "Counter Reset", data)

    def log_error(self, message: str) -> None:
        """
        Log an error event.

        Args:
            message: Error description.
        """
        if not self._started:
            return

        self._write_line("ERROR", message)

    def end_session(
        self,
        total_reps: int = 0,
        avg_form_score: float = 0.0,
        avg_rom: float = 0.0,
    ) -> str:
        """
        End the session and write the closing system message.

        Args:
            total_reps: Total reps completed.
            avg_form_score: Average form score.
            avg_rom: Average ROM.

        Returns:
            Path to the log file.
        """
        if not self._started:
            return str(self.file_path) if self.file_path else ""

        data = (
            f"totalReps={total_reps}, "
            f"avgFormScore={avg_form_score:.1f}, "
            f"avgRom={avg_rom:.1f}"
        )
        self._write_line("LOG", "Session Summary", data)
        self._write_line("SYSTEM", "System halted.")
        self._started = False

        print(f"[PTARawLogger] Raw log saved: {self.file_path}")
        return str(self.file_path)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

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

    def _next_log_number(self) -> int:
        """Find the next available log number."""
        existing = list(self.log_dir.glob("PTA-Log-*.txt"))
        if not existing:
            return 1

        max_num = 0
        for path in existing:
            try:
                # Extract number from PTA-Log-NNNN.txt
                num = int(path.stem.split("-")[-1])
                max_num = max(max_num, num)
            except (ValueError, IndexError):
                continue
        return max_num + 1

    def _write_line(
        self, msg_type: str, message: str, data: Optional[str] = None
    ) -> None:
        """Write a single pipe-delimited log line."""
        if self.file_path is None:
            return

        ts = _pta_timestamp()
        parts = [ts, f" {msg_type} ", f' "{message}"']
        if data:
            parts.append(f" {data}")

        line = " |".join(parts)
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
