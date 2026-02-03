# AI Physical Therapy Assistant

An intelligent exercise monitoring system using YOLO11 pose estimation to provide real-time feedback on physical therapy exercises.

## Features

- **Real-time pose detection** using YOLO11 pose estimation
- **Rep counting** with state machine logic
- **Form assessment** with biomechanical checks
- **Progress tracking** across multiple sessions
- **Multiple interfaces**: CLI, Web (Streamlit), and programmatic API

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Analyze a video
python analyze.py video --input your_video.mp4 --exercise knee_extension

# Start webcam session
python analyze.py webcam --exercise shoulder_flexion

# Launch web interface
streamlit run interfaces/app.py
```

## Supported Exercises

| Exercise | Description | Keypoints |
|----------|-------------|-----------|
| Knee Extension | Seated leg extension | Hip-Knee-Ankle |
| Shoulder Flexion | Standing arm raise | Hip-Shoulder-Elbow |
| Squat | Bodyweight squat | Hip-Knee-Ankle |
| Hip Abduction | Side leg raise | Hip-Hip-Knee |
| Push-up | Upper body exercise | Shoulder-Elbow-Wrist |

## Project Structure

```
ai-pt-assistant/
├── analyze.py              # Core analysis engine
├── config.yaml             # Exercise configurations
├── requirements.txt        # Dependencies
├── src/
│   ├── utils.py            # Helper functions
│   ├── form_assessment.py  # Form checking
│   └── session_tracker.py  # Progress tracking
├── interfaces/
│   ├── cli.py              # Command-line interface
│   └── app.py              # Streamlit web app
└── tests/                  # Test suite
```

## Documentation

- [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) - Detailed project documentation
- [presentation/slides.md](presentation/slides.md) - Classroom presentation

## Requirements

- Python 3.8+
- Webcam (for live sessions)
- See `requirements.txt` for Python packages

## License

Educational project - see LICENSE for details.
