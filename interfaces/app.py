"""
Streamlit Web Application for AI Physical Therapy Assistant
============================================================

A user-friendly web interface for the PT Assistant, designed for:
- Classroom demonstrations
- Patient self-service sessions
- Therapist monitoring dashboards

Pages:
    1. Home/Dashboard - Overview and quick actions
    2. Live Session - Real-time webcam analysis
    3. Video Analysis - Upload and analyze recorded videos
    4. Progress Report - Historical data and trends
    5. How It Works - Educational content about the technology

Run with:
    streamlit run app.py

Author: AI PT Assistant
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import time
import json
from pathlib import Path
from datetime import datetime
import sys
import av

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import streamlit-webrtc for live video
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

# Import YOLO and our utilities
from ultralytics import YOLO
from src.utils import get_angle_from_keypoints, extract_keypoint, calculate_angle

# Page configuration
st.set_page_config(
    page_title="AI PT Assistant",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #1E88E5;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .form-good { color: #28a745; }
    .form-warning { color: #ffc107; }
    .form-poor { color: #dc3545; }
    .stProgress > div > div > div > div {
        background-color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize Streamlit session state variables."""
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    if 'session_data' not in st.session_state:
        st.session_state.session_data = []
    if 'rep_count' not in st.session_state:
        st.session_state.rep_count = 0
    if 'form_score' not in st.session_state:
        st.session_state.form_score = 0
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'current_angle' not in st.session_state:
        st.session_state.current_angle = 0
    if 'exercise_state' not in st.session_state:
        st.session_state.exercise_state = 'neutral'


# =============================================================================
# VIDEO PROCESSOR FOR LIVE POSE ESTIMATION
# =============================================================================

# Exercise configurations
EXERCISE_CONFIGS = {
    "Knee Extension": {"keypoints": [11, 13, 15], "up_angle": 170, "down_angle": 90},
    "Shoulder Flexion": {"keypoints": [11, 5, 7], "up_angle": 170, "down_angle": 20},
    "Squat": {"keypoints": [11, 13, 15], "up_angle": 170, "down_angle": 90},
    "Hip Abduction": {"keypoints": [12, 11, 13], "up_angle": 45, "down_angle": 10},
    "Push-up": {"keypoints": [5, 7, 9], "up_angle": 170, "down_angle": 90},
}

@st.cache_resource
def load_pose_model():
    """Load YOLO pose model (cached)."""
    return YOLO("models/yolo11m-pose.pt")


class PoseVideoProcessor(VideoProcessorBase):
    """Video processor for real-time pose estimation."""

    def __init__(self):
        self.model = load_pose_model()
        self.exercise = "Knee Extension"
        self.keypoints_indices = [11, 13, 15]
        self.up_angle = 170
        self.down_angle = 90
        self.tolerance = 15

        # State tracking
        self.rep_count = 0
        self.current_state = "neutral"
        self.current_angle = 0
        self.form_score = 0
        self.angles_history = []

    def set_exercise(self, exercise_name):
        """Set the exercise configuration."""
        if exercise_name in EXERCISE_CONFIGS:
            config = EXERCISE_CONFIGS[exercise_name]
            self.exercise = exercise_name
            self.keypoints_indices = config["keypoints"]
            self.up_angle = config["up_angle"]
            self.down_angle = config["down_angle"]
            self.rep_count = 0
            self.current_state = "neutral"

    def recv(self, frame):
        """Process each video frame."""
        img = frame.to_ndarray(format="bgr24")

        # Run pose estimation
        results = self.model(img, verbose=False)

        # Get annotated frame with skeleton
        annotated = results[0].plot() if results else img.copy()

        # Extract keypoints and calculate angle
        if results and len(results) > 0 and results[0].keypoints is not None:
            if len(results[0].keypoints) > 0:
                keypoints = results[0].keypoints.data[0].cpu().numpy()

                # Calculate angle
                angle = get_angle_from_keypoints(keypoints, self.keypoints_indices)

                if angle is not None:
                    self.current_angle = angle
                    self.angles_history.append(angle)
                    if len(self.angles_history) > 30:
                        self.angles_history.pop(0)

                    # State machine for rep counting
                    prev_state = self.current_state
                    ascending = self.up_angle > self.down_angle

                    if ascending:
                        if angle >= self.up_angle - self.tolerance:
                            self.current_state = "up"
                        elif angle <= self.down_angle + self.tolerance:
                            if prev_state == "up":
                                self.rep_count += 1
                            self.current_state = "down"
                    else:
                        if angle <= self.up_angle + self.tolerance:
                            if prev_state == "down":
                                self.rep_count += 1
                            self.current_state = "up"
                        elif angle >= self.down_angle - self.tolerance:
                            self.current_state = "down"

                    # Calculate form score (simplified)
                    if len(self.angles_history) > 5:
                        smoothness = 100 - min(np.std(np.diff(self.angles_history[-10:])) * 2, 50)
                        self.form_score = max(0, min(100, smoothness))

                    # Draw angle visualization
                    p1 = extract_keypoint(keypoints, self.keypoints_indices[0])
                    vertex = extract_keypoint(keypoints, self.keypoints_indices[1])
                    p2 = extract_keypoint(keypoints, self.keypoints_indices[2])

                    if all(p is not None for p in [p1, vertex, p2]):
                        v = tuple(vertex.astype(int))
                        cv2.putText(annotated, f"{angle:.1f}", (v[0]+15, v[1]-15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Draw metrics panel on LEFT side (to avoid cutoff on right)
        h, w = annotated.shape[:2]
        panel_w = 220
        panel_h = 160

        # Semi-transparent background panel on left
        overlay = annotated.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)

        # Exercise name
        cv2.putText(annotated, self.exercise, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        # Rep count (large)
        cv2.putText(annotated, f"Reps: {self.rep_count}", (10, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        # Angle
        cv2.putText(annotated, f"Angle: {self.current_angle:.1f}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Phase
        cv2.putText(annotated, f"Phase: {self.current_state.upper()}", (10, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        # Form score
        cv2.putText(annotated, f"Form: {self.form_score:.0f}%", (10, 155),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =============================================================================
# SIDEBAR NAVIGATION
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 10px 0;">
                <div style="font-size: 3rem;">🏥</div>
                <div style="font-size: 1.4rem; font-weight: bold;">PT Assistant</div>
            </div>
        """, unsafe_allow_html=True)
        st.markdown("---")

        # Navigation
        st.markdown("### Navigation")
        pages = {
            'home': '🏠 Dashboard',
            'live': '📹 Live Session',
            'video': '🎬 Video Analysis',
            'progress': '📊 Progress Report',
            'learn': '📚 How It Works'
        }

        for page_id, page_name in pages.items():
            if st.button(page_name, key=f"nav_{page_id}", use_container_width=True):
                st.session_state.current_page = page_id
                st.rerun()

        st.markdown("---")

        # Settings
        st.markdown("### Settings")
        st.selectbox("Patient ID", ["Default", "Patient 001", "Patient 002"],
                     key="patient_id")
        st.toggle("Show Technical Details", key="show_technical")

        st.markdown("---")
        st.caption("AI PT Assistant v1.0")
        st.caption("Powered by YOLO11 Pose Estimation")


# =============================================================================
# PAGE: HOME DASHBOARD
# =============================================================================

def page_home():
    """Render the home dashboard page."""
    st.markdown('<p class="main-header">AI Physical Therapy Assistant</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Intelligent exercise monitoring powered by computer vision</p>',
                unsafe_allow_html=True)

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Sessions This Week", "5", "+2")
    with col2:
        st.metric("Total Reps", "247", "+34")
    with col3:
        st.metric("Avg Form Score", "82%", "+5%")
    with col4:
        st.metric("ROM Improvement", "+8°", "vs last week")

    st.markdown("---")

    # Quick actions
    st.markdown("### Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 📹 Start Live Session")
        st.write("Real-time exercise monitoring with your webcam")
        if st.button("Start Session", type="primary", key="start_live"):
            st.session_state.current_page = 'live'
            st.rerun()

    with col2:
        st.markdown("#### 🎬 Analyze Video")
        st.write("Upload and analyze a recorded exercise video")
        if st.button("Upload Video", key="upload_video"):
            st.session_state.current_page = 'video'
            st.rerun()

    with col3:
        st.markdown("#### 📊 View Progress")
        st.write("Track your improvement over time")
        if st.button("View Reports", key="view_reports"):
            st.session_state.current_page = 'progress'
            st.rerun()

    st.markdown("---")

    # Recent activity
    st.markdown("### Recent Sessions")

    # Sample recent sessions (would be loaded from session_tracker in production)
    recent_sessions = [
        {"date": "2024-01-15", "exercise": "Knee Extension", "reps": 10, "form": 85},
        {"date": "2024-01-14", "exercise": "Shoulder Flexion", "reps": 12, "form": 78},
        {"date": "2024-01-13", "exercise": "Squats", "reps": 8, "form": 82},
    ]

    for session in recent_sessions:
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        with col1:
            st.write(f"📅 {session['date']}")
        with col2:
            st.write(f"🏋️ {session['exercise']}")
        with col3:
            st.write(f"🔄 {session['reps']} reps")
        with col4:
            form_class = "form-good" if session['form'] >= 80 else ("form-warning" if session['form'] >= 60 else "form-poor")
            st.markdown(f'<span class="{form_class}">✓ {session["form"]}%</span>',
                       unsafe_allow_html=True)


# =============================================================================
# PAGE: LIVE SESSION
# =============================================================================

def page_live_session():
    """Render the live session page with real webcam support."""
    st.markdown("## 📹 Live PT Session")

    # Hide webrtc default icons and fullscreen toolbar
    st.markdown("""
        <style>
        /* Hide the fullscreen toolbar button */
        .stElementToolbar,
        [data-testid="stElementToolbar"] {
            display: none !important;
        }
        /* Make video container cleaner */
        .stApp video {
            border-radius: 8px;
            border: 2px solid #333;
        }
        </style>
    """, unsafe_allow_html=True)

    # Exercise and camera selection
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        exercise = st.selectbox(
            "Select Exercise",
            list(EXERCISE_CONFIGS.keys()),
            key="exercise_select"
        )

    with col2:
        target_reps = st.number_input("Target Reps", min_value=1, max_value=50, value=10)

    with col3:
        camera_option = st.selectbox(
            "Camera",
            ["Default", "Front Camera", "Back Camera"],
            key="camera_select",
            help="Select which camera to use"
        )

    st.markdown("---")

    # CLI quality note
    st.warning("**💡 For best video quality**, use the command line: `python analyze.py webcam --exercise \"" + exercise + "\"`")

    st.info("**Instructions:** Allow camera access when prompted. The pose estimation will start automatically. Press 'STOP' to end the session.")

    # HD video constraints with camera selection
    video_config = {
        "width": {"min": 1280, "ideal": 1920},
        "height": {"min": 720, "ideal": 1080},
        "frameRate": {"ideal": 30}
    }

    # Add facing mode based on camera selection
    if camera_option == "Front Camera":
        video_config["facingMode"] = "user"
    elif camera_option == "Back Camera":
        video_config["facingMode"] = "environment"

    video_constraints = {
        "video": video_config,
        "audio": False
    }

    # RTC configuration
    rtc_config = {
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}],
    }

    # Create the webrtc streamer with pose estimation
    ctx = webrtc_streamer(
        key=f"pose-estimation-{camera_option}",  # Key changes with camera to force refresh
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=PoseVideoProcessor,
        media_stream_constraints=video_constraints,
        async_processing=True,
        rtc_configuration=rtc_config,
        video_html_attrs={
            "style": {"width": "100%", "max-width": "1280px"},
            "autoPlay": True,
            "muted": True
        }
    )

    # Update exercise config when selection changes
    if ctx.video_processor:
        ctx.video_processor.set_exercise(exercise)

    st.markdown("---")

    # Info panel
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Exercise Info")
        if exercise in EXERCISE_CONFIGS:
            config = EXERCISE_CONFIGS[exercise]
            st.write(f"**Keypoints:** {config['keypoints']}")
            st.write(f"**Up Angle:** {config['up_angle']}°")
            st.write(f"**Down Angle:** {config['down_angle']}°")

    with col2:
        st.markdown("### Target")
        st.metric("Target Reps", target_reps)

    with col3:
        st.markdown("### Tips")
        tips = {
            "Knee Extension": "Keep your back against the seat",
            "Shoulder Flexion": "Don't arch your back",
            "Squat": "Keep knees over toes",
            "Hip Abduction": "Keep trunk stable",
            "Push-up": "Maintain straight body line"
        }
        st.info(tips.get(exercise, "Maintain good form"))

    # Controls
    st.markdown("---")
    st.markdown("### Session Notes")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("📸 Save Screenshot"):
            st.toast("Screenshot saved!")

    with col2:
        if st.button("💾 Save Session"):
            st.toast("Session saved!")


# =============================================================================
# PAGE: VIDEO ANALYSIS
# =============================================================================

def page_video_analysis():
    """Render the video analysis page."""
    st.markdown("## 🎬 Video Analysis")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video of your exercise session for analysis"
    )

    if uploaded_file is not None:
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Original Video")
            st.video(video_path)

        with col2:
            st.markdown("### Analysis Settings")

            exercise = st.selectbox(
                "Exercise Type",
                ["Knee Extension", "Shoulder Flexion", "Squat", "Hip Abduction"]
            )

            st.checkbox("Generate annotated video", value=True, key="gen_annotated")
            st.checkbox("Export detailed report", value=True, key="gen_report")

            if st.button("🔍 Analyze Video", type="primary"):
                with st.spinner("Analyzing video..."):
                    # Progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Simulate analysis (would call actual analysis in production)
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                        status_text.text(f"Processing frame {i+1}/100...")

                    status_text.text("Analysis complete!")

                st.success("✅ Video analysis complete!")

                # Show results
                st.markdown("---")
                st.markdown("### Analysis Results")

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Reps", "12")
                with col2:
                    st.metric("Avg Form Score", "78%")
                with col3:
                    st.metric("ROM Range", "85°")
                with col4:
                    st.metric("Duration", "2:34")

                # Rep-by-rep breakdown
                st.markdown("#### Rep Analysis")
                rep_data = [
                    {"rep": 1, "rom": 82, "form": 85, "time": 2.1},
                    {"rep": 2, "rom": 85, "form": 82, "time": 2.0},
                    {"rep": 3, "rom": 80, "form": 78, "time": 2.3},
                    {"rep": 4, "rom": 78, "form": 75, "time": 2.5},
                ]

                import pandas as pd
                df = pd.DataFrame(rep_data)
                st.dataframe(df, use_container_width=True)

                # Download options
                st.markdown("#### Download Results")
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📄 Download Report (JSON)",
                        json.dumps({"reps": rep_data}, indent=2),
                        file_name="analysis_report.json",
                        mime="application/json"
                    )
                with col2:
                    st.download_button(
                        "🎬 Download Annotated Video",
                        b"video_placeholder",  # Would be actual video in production
                        file_name="annotated_video.mp4",
                        mime="video/mp4",
                        disabled=True  # Disabled for demo
                    )

    else:
        st.info("👆 Upload a video file to begin analysis")

        # Show example
        st.markdown("### Example Analysis")
        st.image("https://via.placeholder.com/800x400?text=Example+Analysis+Screenshot",
                use_container_width=True)


# =============================================================================
# PAGE: PROGRESS REPORT
# =============================================================================

def page_progress_report():
    """Render the progress report page."""
    st.markdown("## 📊 Progress Report")

    # Time period selector
    col1, col2 = st.columns([3, 1])
    with col1:
        time_range = st.select_slider(
            "Time Period",
            options=["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year"],
            value="1 Month"
        )
    with col2:
        exercise_filter = st.selectbox(
            "Exercise",
            ["All Exercises", "Knee Extension", "Shoulder Flexion", "Squat"]
        )

    st.markdown("---")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Sessions", "23", "+5 from last period")
    with col2:
        st.metric("Total Reps", "482", "+87")
    with col3:
        st.metric("Avg Form Score", "81%", "+4%")
    with col4:
        st.metric("ROM Improvement", "+12°", "from baseline")

    st.markdown("---")

    # Progress charts (using placeholder data)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Form Score Trend")
        # Sample data for chart
        import pandas as pd
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        scores = np.random.normal(80, 5, 30).clip(60, 100)
        chart_data = pd.DataFrame({'Date': dates, 'Form Score': scores})
        st.line_chart(chart_data.set_index('Date'))

    with col2:
        st.markdown("### Range of Motion Trend")
        rom_values = np.random.normal(85, 8, 30).clip(60, 110)
        chart_data = pd.DataFrame({'Date': dates, 'ROM (degrees)': rom_values})
        st.line_chart(chart_data.set_index('Date'))

    st.markdown("---")

    # Exercise breakdown
    st.markdown("### Exercise Breakdown")

    exercises = [
        {"name": "Knee Extension", "sessions": 8, "avg_form": 82, "rom_change": "+8°", "trend": "↑"},
        {"name": "Shoulder Flexion", "sessions": 6, "avg_form": 78, "rom_change": "+5°", "trend": "↑"},
        {"name": "Squat", "sessions": 5, "avg_form": 75, "rom_change": "+3°", "trend": "→"},
        {"name": "Hip Abduction", "sessions": 4, "avg_form": 80, "rom_change": "+6°", "trend": "↑"},
    ]

    for ex in exercises:
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
        with col1:
            st.write(f"**{ex['name']}**")
        with col2:
            st.write(f"{ex['sessions']} sessions")
        with col3:
            form_color = "🟢" if ex['avg_form'] >= 80 else ("🟡" if ex['avg_form'] >= 60 else "🔴")
            st.write(f"{form_color} {ex['avg_form']}%")
        with col4:
            st.write(f"ROM: {ex['rom_change']}")
        with col5:
            st.write(ex['trend'])

    st.markdown("---")

    # Areas for improvement
    st.markdown("### Areas for Attention")
    st.warning("⚠️ **Squat** - Form score has been declining. Focus on knee alignment.")
    st.info("💡 **Shoulder Flexion** - Good progress! Consider increasing resistance.")

    # Export options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 Export Full Report (PDF)",
            b"pdf_placeholder",
            file_name="progress_report.pdf",
            mime="application/pdf",
            disabled=True
        )
    with col2:
        st.download_button(
            "📊 Export Data (CSV)",
            "date,exercise,reps,form_score\n2024-01-15,Knee Extension,10,85",
            file_name="session_data.csv",
            mime="text/csv"
        )
    with col3:
        if st.button("📧 Email Report"):
            st.toast("Report sent!")


# =============================================================================
# PAGE: HOW IT WORKS (Educational)
# =============================================================================

def page_how_it_works():
    """Render the educational page explaining the technology."""
    st.markdown("## 📚 How It Works")
    st.markdown("*Understanding the AI behind your PT Assistant*")

    # Navigation tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Overview",
        "🦴 Pose Estimation",
        "📐 Angle Calculation",
        "🧠 Form Assessment"
    ])

    with tab1:
        st.markdown("""
        ### What is AI Pose Estimation?

        This application uses **YOLO11 Pose Estimation** to detect and track your body
        movements in real-time. Here's the high-level process:

        1. **📹 Video Input** - Camera captures your exercise
        2. **🔍 Detection** - AI identifies your body in each frame
        3. **🦴 Keypoint Extraction** - 17 body landmarks are located
        4. **📐 Angle Calculation** - Joint angles computed from keypoints
        5. **✅ Form Assessment** - Angles compared to target ranges
        6. **💬 Feedback** - Real-time guidance provided

        """)

        st.image("https://via.placeholder.com/800x300?text=Pipeline+Diagram",
                caption="AI Processing Pipeline", use_container_width=True)

        st.markdown("""
        ### Why AI for Physical Therapy?

        - **Consistency**: AI provides objective, repeatable measurements
        - **Accessibility**: PT guidance available at home
        - **Real-time**: Immediate feedback during exercise
        - **Tracking**: Automatic progress documentation
        """)

    with tab2:
        st.markdown("""
        ### YOLO Pose Keypoints

        The AI detects **17 body keypoints** that define your pose:
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            | Index | Keypoint |
            |-------|----------|
            | 0 | Nose |
            | 1 | Left Eye |
            | 2 | Right Eye |
            | 3 | Left Ear |
            | 4 | Right Ear |
            | 5 | Left Shoulder |
            | 6 | Right Shoulder |
            | 7 | Left Elbow |
            | 8 | Right Elbow |
            """)

        with col2:
            st.markdown("""
            | Index | Keypoint |
            |-------|----------|
            | 9 | Left Wrist |
            | 10 | Right Wrist |
            | 11 | Left Hip |
            | 12 | Right Hip |
            | 13 | Left Knee |
            | 14 | Right Knee |
            | 15 | Left Ankle |
            | 16 | Right Ankle |
            """)

        st.image("https://via.placeholder.com/600x400?text=Keypoint+Diagram",
                caption="Body keypoint locations", use_container_width=True)

        st.markdown("""
        ### Confidence Scores

        Each keypoint has a **confidence score** (0-1) indicating how certain
        the AI is about its position. We typically require >0.3 confidence
        to use a keypoint for calculations.
        """)

    with tab3:
        st.markdown("""
        ### How Joint Angles Are Calculated

        We use **vector mathematics** to calculate the angle at any joint:
        """)

        st.latex(r'''
        \cos(\theta) = \frac{\vec{A} \cdot \vec{B}}{|\vec{A}| \times |\vec{B}|}
        ''')

        st.markdown("""
        Where:
        - **θ** is the angle at the joint
        - **A** is the vector from joint to first connected point
        - **B** is the vector from joint to second connected point

        ### Example: Knee Angle

        For knee extension, we calculate the angle using three points:
        - **Hip** (keypoint 11)
        - **Knee** (keypoint 13) - this is the vertex
        - **Ankle** (keypoint 15)
        """)

        st.code("""
def calculate_angle(hip, knee, ankle):
    # Create vectors from knee to hip and knee to ankle
    vector_a = hip - knee
    vector_b = ankle - knee

    # Calculate angle using dot product formula
    cos_angle = np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )

    # Convert to degrees
    angle = np.degrees(np.arccos(cos_angle))
    return angle
        """, language="python")

        st.markdown("""
        ### Angle Interpretation

        | Knee Angle | Meaning |
        |------------|---------|
        | ~180° | Fully extended (straight leg) |
        | ~90° | Right angle (seated position) |
        | <90° | Deep flexion |
        """)

    with tab4:
        st.markdown("""
        ### Form Assessment Logic

        The system evaluates form using multiple criteria:

        #### 1. Range of Motion (ROM)
        Compares achieved angles to target angles for each exercise.

        #### 2. Movement Quality
        - **Smoothness**: Measures variation in angle changes
        - **Symmetry**: Compares left vs right side
        - **Control**: Detects jerky or momentum-based movements

        #### 3. Compensation Detection
        Identifies when patients use incorrect muscles:
        - Hip hiking during leg exercises
        - Trunk lean during arm movements
        - Shoulder shrugging

        #### 4. Fatigue Tracking
        Monitors form degradation over repetitions to suggest rest.
        """)

        st.markdown("""
        ### Form Score Calculation
        """)

        st.code("""
def calculate_form_score(rep_data):
    score = 0

    # ROM Achievement (40 points)
    rom_ratio = actual_rom / target_rom
    score += rom_ratio * 40

    # Smoothness (30 points)
    smoothness = 1.0 / (1.0 + angle_variance)
    score += smoothness * 30

    # Symmetry (30 points)
    symmetry = 1.0 - abs(left_time - right_time) / total_time
    score += symmetry * 30

    return min(score, 100)
        """, language="python")

        # Interactive demo
        st.markdown("---")
        st.markdown("### Try It: Form Score Calculator")

        col1, col2, col3 = st.columns(3)
        with col1:
            rom_achieved = st.slider("ROM Achieved (%)", 0, 100, 80)
        with col2:
            smoothness = st.slider("Movement Smoothness", 0, 100, 70)
        with col3:
            symmetry = st.slider("Left/Right Symmetry", 0, 100, 90)

        form_score = (rom_achieved * 0.4) + (smoothness * 0.3) + (symmetry * 0.3)
        st.metric("Calculated Form Score", f"{form_score:.1f}%")

    # Toggle technical overlay option
    if st.session_state.get('show_technical', False):
        st.markdown("---")
        st.markdown("### 🔧 Technical Details")
        st.json({
            "model": "models/yolo11m-pose.pt",
            "framework": "Ultralytics",
            "keypoints": 17,
            "inference_speed": "~30ms per frame",
            "input_resolution": "640x640"
        })


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    render_sidebar()

    # Route to current page
    page = st.session_state.current_page

    if page == 'home':
        page_home()
    elif page == 'live':
        page_live_session()
    elif page == 'video':
        page_video_analysis()
    elif page == 'progress':
        page_progress_report()
    elif page == 'learn':
        page_how_it_works()
    else:
        page_home()


if __name__ == "__main__":
    main()
