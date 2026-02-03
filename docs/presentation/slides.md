# AI Physical Therapy Assistant
## Classroom Presentation

---

# Slide 1: Title

## AI Physical Therapy Assistant
### Intelligent Exercise Monitoring with Computer Vision

**Key Technologies:**
- YOLO11 Pose Estimation
- Python & OpenCV
- Real-time Video Analysis

*Demonstrating practical AI applications in healthcare*

---

# Slide 2: Problem Statement

## The Challenge in Physical Therapy

### Patient Challenges:
- 🏥 Limited access to in-person PT sessions
- 😕 Uncertainty about correct exercise form
- 📉 Difficulty tracking progress at home
- ⏰ Inconsistent exercise compliance

### Therapist Challenges:
- 📝 Manual documentation is time-consuming
- 👀 Can't observe patients between visits
- 📊 Subjective assessments vary
- 🔄 Difficult to track progress objectively

### The Cost:
- Poor outcomes from incorrect form
- Slower recovery times
- Increased re-injury risk
- Healthcare system inefficiency

---

# Slide 3: Solution Overview

## AI-Powered Exercise Monitoring

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Camera    │ ──> │  YOLO11     │ ──> │  Analysis   │
│   Input     │     │  Pose       │     │  Engine     │
└─────────────┘     └─────────────┘     └─────────────┘
                           │                    │
                           ▼                    ▼
                    ┌─────────────┐     ┌─────────────┐
                    │  17 Body    │     │  Real-time  │
                    │  Keypoints  │     │  Feedback   │
                    └─────────────┘     └─────────────┘
```

### What It Does:
1. **Detects** body position in real-time
2. **Calculates** joint angles mathematically
3. **Compares** to target exercise ranges
4. **Provides** immediate, actionable feedback
5. **Tracks** progress over multiple sessions

---

# Slide 4: Key AI/ML Concepts

## Technologies Demonstrated

### 1. Object Detection vs. Pose Estimation
| Object Detection | Pose Estimation |
|------------------|-----------------|
| "Where is the person?" | "How is the person positioned?" |
| Bounding box output | Keypoint coordinates output |
| Single point per object | 17 points per person |

### 2. Transfer Learning
- Using **pre-trained** YOLO11-pose model
- Trained on COCO dataset (330K images)
- Fine-tunable for specific needs

### 3. Real-time Inference
- ~30ms per frame (CPU)
- ~10ms per frame (GPU)
- Enables live video feedback

### 4. State Machine Logic
- Threshold-based rep counting
- No additional ML needed for counting
- Reliable, interpretable results

---

# Slide 5: YOLO Pose Keypoints

## The 17 Body Landmarks

```
           0 (Nose)
          /   \
     1,2 (Eyes)
        / | \
   3,4 (Ears)
        |
    5 ─ + ─ 6  (Shoulders)
       /|\
      / | \
     7  |  8  (Elbows)
    /   |   \
   9    |   10 (Wrists)
        |
   11 ─ + ─ 12 (Hips)
       / \
      /   \
    13    14 (Knees)
     |     |
    15    16 (Ankles)
```

### Key Combinations for PT:
- **Knee angle:** Hip(11) → Knee(13) → Ankle(15)
- **Shoulder flexion:** Hip(11) → Shoulder(5) → Elbow(7)
- **Squat depth:** Hip(11) → Knee(13) → Ankle(15)

---

# Slide 6: Angle Calculation

## The Math Behind Joint Angles

### Vector Mathematics

Given three points (P1, Vertex, P2):

```
        P1 (hip)
         \
          \  Vector A
           \
            V (knee) ← angle measured here
           /
          /  Vector B
         /
        P2 (ankle)
```

### Formula:
```
cos(θ) = (A · B) / (|A| × |B|)
```

### Code Implementation:
```python
def calculate_angle(hip, knee, ankle):
    vector_a = hip - knee
    vector_b = ankle - knee

    cos_angle = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) *
        np.linalg.norm(vector_b)
    )

    return np.degrees(np.arccos(cos_angle))
```

---

# Slide 7: Rep Counting State Machine

## How Reps Are Counted

### State Diagram:
```
              angle ≥ UP_THRESHOLD
    ┌─────────────────────────────────┐
    │                                 │
    ▼                                 │
┌───────┐                        ┌────┴───┐
│ DOWN  │                        │   UP   │
│ State │                        │  State │
└───┬───┘                        └────────┘
    │                                 ▲
    │                                 │
    └─────────────────────────────────┘
         angle ≤ DOWN_THRESHOLD
              (REP COUNTED!)
```

### Example: Knee Extension
- **DOWN state:** Knee bent (angle ~90°)
- **UP state:** Leg extended (angle ~170°)
- **Rep complete:** When returning from UP to DOWN

### Why This Works:
- Simple, deterministic logic
- No ML needed for counting
- Easy to debug and adjust
- Clinically meaningful thresholds

---

# Slide 8: Form Assessment

## Evaluating Exercise Quality

### Multi-Factor Scoring System:

| Factor | Weight | What It Measures |
|--------|--------|------------------|
| ROM Achievement | 40% | Did you reach target angles? |
| Smoothness | 30% | Was movement controlled? |
| Symmetry | 30% | Were both sides equal? |

### Compensation Detection:
- **Hip hiking** during leg exercises
- **Trunk lean** to assist movement
- **Shoulder shrugging** during arm raises
- **Momentum/swinging** instead of control

### Feedback Priority:
1. 🔴 **CRITICAL** - Stop exercise (safety concern)
2. 🟡 **WARNING** - Form issue to correct
3. 🟢 **SUGGESTION** - Minor improvement
4. ⚪ **INFO** - Observation

---

# Slide 9: Important AI Studio Project Features

## What Makes a Good AI Project?

### ✅ Clean Code Organization
```
ai-pt-assistant/
├── analyze.py           # Core engine
├── config.yaml          # Configuration
├── src/                 # Modules
│   ├── form_assessment.py
│   └── session_tracker.py
├── interfaces/          # User interfaces
│   ├── cli.py
│   └── app.py
└── tests/               # Test suite
```

### ✅ Configuration-Driven Behavior
- All parameters in YAML
- Change exercises without code changes
- Easy to customize for different patients

### ✅ Multiple Interfaces
- CLI for power users
- Web UI for general users
- Programmatic API for integration

### ✅ Educational Documentation
- Inline comments explaining concepts
- PROJECT_OVERVIEW.md
- "How It Works" page in app

### ✅ Reproducible Setup
- requirements.txt
- Clear installation steps
- Sample data included

---

# Slide 10: Technical Deep-Dive

## Under the Hood

### Processing Pipeline:
```python
# 1. Load video frame
frame = cap.read()

# 2. Run YOLO inference
results = model(frame)

# 3. Extract keypoints
keypoints = results[0].keypoints.data[0]

# 4. Calculate angle
angle = calculate_angle(
    keypoints[11],  # hip
    keypoints[13],  # knee
    keypoints[15]   # ankle
)

# 5. Update state machine
metrics = tracker.update(angle)

# 6. Generate feedback
feedback = form_analyzer.analyze_frame(keypoints)

# 7. Annotate frame
annotated = draw_metrics(frame, metrics, feedback)
```

### Performance Optimizations:
- **Angle smoothing:** 5-frame moving average
- **Confidence filtering:** Ignore low-confidence keypoints
- **Temporal consistency:** Prevent flickering states

---

# Slide 11: Demo Screenshots

## The Application in Action

### Live Session View:
```
┌────────────────────────────────────────────────────────┐
│  ┌─────────────────────────┐  ┌─────────────────────┐ │
│  │                         │  │ Exercise: Knee Ext  │ │
│  │     [Video Feed]        │  │ Reps: 7/10          │ │
│  │     with skeleton       │  │ Angle: 142.3°       │ │
│  │     overlay             │  │ Phase: UP           │ │
│  │                         │  │ Form: 85% ●         │ │
│  │        /\               │  │                     │ │
│  │       /  \              │  │ Feedback:           │ │
│  │      /    \             │  │ ✓ Good extension!   │ │
│  │     |      |            │  │                     │ │
│  └─────────────────────────┘  └─────────────────────┘ │
│                                                        │
│  [Start] [Pause] [Reset] [Save]                        │
└────────────────────────────────────────────────────────┘
```

### Progress Dashboard:
```
┌─────────────────────────────────────────────────────────┐
│  Progress Report - Last 30 Days                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Form Score Trend          ROM Trend                    │
│  ▲                         ▲                            │
│  │    .-·-·-·              │       .-·-·                │
│  │  ·-                     │    .-·                     │
│  │·-                       │ .-·                        │
│  └────────────>            └────────────>               │
│                                                         │
│  Exercises Needing Attention:                           │
│  ⚠️ Squat - Form declining, focus on knee alignment    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

# Slide 12: Lessons Learned

## Best Practices from This Project

### 1. Start with the Data Pipeline
- Understand input format (video, keypoints)
- Plan output format (reports, visualizations)
- Build end-to-end before adding features

### 2. Configuration Over Hard-coding
- Exercise parameters in YAML, not code
- Makes customization easy
- Enables non-developers to adjust

### 3. Modular Architecture
- Separate concerns (analysis, form checking, storage)
- Easier testing and debugging
- Supports future extensions

### 4. Real-time Constraints
- Smoothing prevents noisy outputs
- Confidence thresholds avoid bad data
- State machines provide stability

### 5. User Experience Matters
- Multiple interfaces for different users
- Clear, actionable feedback
- Progress tracking motivates patients

---

# Slide 13: Extension Ideas

## Future Possibilities

### Short-term:
- 🔊 **Voice feedback** - Text-to-speech cues
- 👥 **Multi-person** - Group exercise sessions
- 📱 **Mobile app** - iOS/Android wrapper

### Medium-term:
- 🎯 **Custom models** - Train for PT equipment
- ⌚ **Wearables** - Integrate with smartwatches
- 📞 **Telehealth** - Video call integration
- 🎮 **Gamification** - Points, achievements, challenges

### Long-term:
- 🏥 **EHR Integration** - Automatic documentation
- 📋 **Insurance** - Automated compliance reporting
- 🌍 **Multi-language** - Global accessibility
- 🔬 **Research** - Clinical trial data collection

---

# Slide 14: Code Quality Checklist

## Project Evaluation Criteria

### ✅ Documentation
- [ ] README with quick start
- [ ] PROJECT_OVERVIEW.md
- [ ] Inline code comments
- [ ] Docstrings on functions

### ✅ Code Organization
- [ ] Modular file structure
- [ ] Separation of concerns
- [ ] Consistent naming
- [ ] No code duplication

### ✅ Configuration
- [ ] External config files
- [ ] Environment variables support
- [ ] Sensible defaults

### ✅ Testing
- [ ] Unit tests
- [ ] Test fixtures
- [ ] Edge case coverage

### ✅ User Experience
- [ ] Multiple interfaces
- [ ] Clear error messages
- [ ] Progress indicators
- [ ] Help documentation

---

# Slide 15: Questions & Discussion

## Let's Explore Further

### Discussion Topics:
1. How could this be adapted for other healthcare applications?
2. What are the ethical considerations of AI in healthcare?
3. How would you handle patient data privacy (HIPAA)?
4. What additional sensors could improve accuracy?

### Hands-on Exercises:
1. Add a new exercise to config.yaml
2. Modify the form scoring weights
3. Create a custom feedback message
4. Run the web interface locally

### Resources:
- [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
- [OpenCV Python Tutorials](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

---

# Appendix A: Installation Commands

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run video analysis
python analyze.py video -i sample.mp4 -e knee_extension

# Start webcam session
python analyze.py webcam -e shoulder_flexion

# Launch web interface
streamlit run interfaces/app.py

# Run tests
pytest tests/
```

---

# Appendix B: Key Code Snippets

### Angle Calculation:
```python
def calculate_angle(p1, vertex, p2):
    v1 = p1 - vertex
    v2 = p2 - vertex
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
```

### Rep Counting:
```python
if angle >= up_threshold:
    state = "up"
elif angle <= down_threshold:
    if previous_state == "up":
        rep_count += 1  # Completed a rep!
    state = "down"
```

### Form Scoring:
```python
score = (rom_ratio * 40) + (smoothness * 30) + (symmetry * 30)
```

---

# Appendix C: YOLO Keypoint Reference

| Index | Keypoint | Index | Keypoint |
|-------|----------|-------|----------|
| 0 | Nose | 9 | Left Wrist |
| 1 | Left Eye | 10 | Right Wrist |
| 2 | Right Eye | 11 | Left Hip |
| 3 | Left Ear | 12 | Right Hip |
| 4 | Right Ear | 13 | Left Knee |
| 5 | Left Shoulder | 14 | Right Knee |
| 6 | Right Shoulder | 15 | Left Ankle |
| 7 | Left Elbow | 16 | Right Ankle |
| 8 | Right Elbow | | |

### Common Exercise Keypoint Combinations:
- **Knee Extension:** [11, 13, 15] or [12, 14, 16]
- **Shoulder Flexion:** [11, 5, 7] or [12, 6, 8]
- **Squat:** [11, 13, 15] (same as knee extension)
- **Push-up:** [5, 7, 9] or [6, 8, 10]
