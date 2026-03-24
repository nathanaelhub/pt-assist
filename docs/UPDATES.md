# PT Assist — Project Updates

---

## Update 2 — 2026-02-21

### 1. Timestamped Session Log Files (`analyze.py`)

Added a `SessionLogger` class that writes structured, timestamped text logs to a `logs/` directory.

- **Auto-numbered files**: `logfile-01.log`, `logfile-02.log`, etc. Each new session always gets the next available number.
- **50-line cap per file**: When a file reaches 50 lines, the logger automatically rolls over to the next numbered file and writes a continuation header.
- **Timestamps on every entry**: Each line is prefixed with `[HH:MM:SS]`. The file header records the full `YYYY-MM-DD HH:MM:SS` creation time.
- **Session lifecycle events logged**:
  - `SESSION START` — exercise name, camera ID, imgsz, frame_skip
  - `REP N` — ROM, form score, duration, angle range per completed rep
  - `FORM WARNING/POOR/STOP` — form analyzer feedback (throttled to ~every 2 s)
  - `SWITCH` — exercise transitions via `n` key
  - `RESET` — rep counter reset via `r` key
  - `SESSION END` — written automatically on quit (`q`) or app exit, with total reps, avg form, avg ROM

**Usage:**
```
logs/
  logfile-01.log
  logfile-02.log
  ...
```

**Log entry examples:**
```
# logfile created: 2026-02-21 12:34:56
[12:34:56] SESSION START | Exercise: Elbow Flexion | Camera: 0 | imgsz: 320 | skip: 1
[12:35:02] REP 1 | ROM: 79.3° | Form: 88% | Duration: 2.1s | Angle: 91°-170°
[12:35:15] FORM WARNING | Keep your back straight against the chair
[12:35:30] SESSION END | Total Reps: 5 | Avg Form: 85% | Avg ROM: 76.4°
```

---

### 2. Camera FPS Improvement (`analyze.py`)

- **`imgsz` parameter** (default `320`): YOLO inference image size. Reduced from 640 → 320 gives ~4× throughput improvement with minimal accuracy loss for PT tracking distances.
- **`frame_skip` parameter** (default `1`): Run YOLO every Nth frame. On skipped frames the raw camera image is shown with the last known pose/angle overlaid. Motion stays smooth; YOLO skeleton updates at the reduced rate.
- **CLI flags added:**
  ```
  python analyze.py webcam --exercise elbow_flexion --camera 0 --imgsz 320 --skip-frames 2
  ```

---

### 3. Annotated Video Download — Web App (`interfaces/app.py`)

The **🎬 Video Analysis** page now runs real YOLO pose analysis instead of a fake progress simulation:

- Calls `analyze_video()` (the `main()` function from `analyze.py`) with `show_video=False`
- Annotated output video is saved to a temp file, displayed inline with `st.video()`, and available for download
- Results (reps, form score, ROM, per-rep table) come from the real `ExerciseSession` object
- Report JSON download reflects actual analysis data

---

### 4. Session Log Analysis Page — Web App (`interfaces/app.py`)

New **📋 Session Logs** page added to the sidebar:

- Dropdown to select any saved session JSON from `sessions/`
- Summary metrics: reps completed, avg form score, avg ROM, tempo consistency σ
- Charts: ROM per rep (bar), form score per rep (line), angle range per rep (area)
- Rep details table
- Recommendations from session analysis
- **Raw Logs viewer**: browse `logfile-XX.log` files from the `logs/` directory, view content inline, download individual log files

---

### 5. Idle/Mid-Movement Phase Detection (`analyze.py`)

Added a fourth exercise phase state to the `ExerciseTracker` state machine:

| State | Condition | Panel Color |
|-------|-----------|-------------|
| `UP` | Angle reached the up threshold ± tolerance | Green |
| `DOWN` | Angle reached the down threshold ± tolerance | Yellow |
| `IDLE` | Angle is between both thresholds (mid-movement) | Orange |
| `NEUTRAL` | Initial state, no movement detected yet | White |

- Rep counting uses a `last_peak_state` tracker so idle frames do not interfere with rep detection — a rep still counts when transitioning `UP → (IDLE) → DOWN` or `DOWN → (IDLE) → UP`
- The panel "Phase:" indicator displays `IDLE` in orange for easy identification

---

### 6. FormAnalyzer Wired into Webcam + Improved Panel (`analyze.py`)

- `FormAnalyzer` (from `src/form_assessment.py`) now runs on every inference frame in `analyze_webcam()`
- Right panel additions:
  - **Status** line: `EXCELLENT` / `GOOD` / `WARNING` / `POOR` / `STOP` with matching color
  - **Feedback messages**: up to 2 exercise-specific cues (trunk lean, hip hike, shoulder shrug, knee valgus, etc.)
- `r` key resets the FormAnalyzer alongside the tracker
- `n` key reinitializes FormAnalyzer for the new exercise type

---

## Update 1 — Initial Build

- YOLO11 pose estimation with `yolo11m-pose.pt`
- Rep counting state machine (UP / DOWN)
- Per-rep form scoring: ROM achievement (40%), smoothness (30%), phase symmetry (30%)
- Exercise configs in `config.yaml` (knee extension, shoulder flexion, squat, hip abduction, elbow flexion, push-up, trunk flexion, hip abduction, seated hip flexion)
- CLI: `python analyze.py webcam / video / image / aigym`
- Streamlit web app: Dashboard, Live Session (WebRTC), Video Analysis, Progress Report, How It Works
- Session export: JSON + CSV to `sessions/`
- `FormAnalyzer` in `src/form_assessment.py`: alignment, compensation detection, symmetry, fatigue tracking, exercise-specific checks
