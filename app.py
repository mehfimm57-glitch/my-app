"""
MoodTune AI - Main Streamlit Dashboard Application.
Real-Time Emotion-Based Music Recommendation System using CNN and OpenCV.
"""

from datetime import datetime
from pathlib import Path
import time
from typing import Dict, Any, List, Optional
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import streamlit as st

# Import project modules
from src.emotion_detector import EmotionDetector
from src.emotion_smoother import EmotionSmoother
from src.face_detector import FaceDetector
from src.preprocessing import preprocess_face
from src.recommender import MusicRecommender

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="MoodTune AI - Emotion Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Dark Music Dashboard CSS Styling
CUSTOM_CSS = """
<style>
    /* Global Background & Typography */
    .main {
        background-color: #0E1117;
        color: #F0F2F6;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    .stAppHeader {
        background-color: rgba(14, 17, 23, 0.8);
    }
    
    /* Header Container */
    .title-header {
        background: linear-gradient(135deg, #1E1E2F 0%, #0F172A 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
    }
    .title-header h1 {
        color: #FFFFFF;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin: 0;
        font-size: 2.3rem;
    }
    .title-header p {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-top: 0.3rem;
        margin-bottom: 0;
    }

    /* System Status Bar */
    .status-bar {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-bottom: 1.5rem;
    }
    .status-pill {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 0.85rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 6px;
    }

    /* Cards */
    .dashboard-card {
        background: #1E222D;
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.4rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
        margin-bottom: 1.2rem;
    }

    /* Emotion Showcase Card */
    .emotion-badge-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #3b82f6;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }
    .emotion-emoji {
        font-size: 4rem;
        margin-bottom: 0.2rem;
        line-height: 1;
    }
    .emotion-title {
        font-size: 2rem;
        font-weight: 800;
        color: #F8FAFC;
        letter-spacing: 1px;
        margin-bottom: 0.4rem;
    }
    .confidence-tag {
        display: inline-block;
        background: rgba(59, 130, 246, 0.2);
        color: #60A5FA;
        border: 1px solid rgba(96, 165, 250, 0.4);
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }

    /* Song Recommendation Card */
    .song-card {
        background: #181C26;
        border-left: 4px solid #10B981;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.8rem;
        transition: transform 0.2s ease;
    }
    .song-card:hover {
        border-left-color: #34D399;
        transform: translateY(-2px);
    }
    .song-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F3F4F6;
        margin: 0;
    }
    .song-artist {
        font-size: 0.9rem;
        color: #9CA3AF;
        margin-bottom: 0.4rem;
    }
    .song-meta {
        display: flex;
        gap: 8px;
        align-items: center;
        flex-wrap: wrap;
        font-size: 0.8rem;
    }
    .tag-genre {
        background: rgba(99, 102, 241, 0.2);
        color: #818CF8;
        padding: 2px 8px;
        border-radius: 6px;
    }
    .tag-score {
        background: rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 700;
    }

    /* Metrics highlight */
    .metric-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #38BDF8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Emotion Emoji Mapping
EMOTION_EMOJIS = {
    "Happy": "😊",
    "Sad": "😢",
    "Angry": "😠",
    "Fear": "😨",
    "Surprise": "😲",
    "Neutral": "😐",
    "Disgust": "🤢",
    "Uncertain expression": "🤔",
}

# Emotion Color Palette for Plotly Charts
EMOTION_COLORS = {
    "Happy": "#F59E0B",
    "Sad": "#3B82F6",
    "Angry": "#EF4444",
    "Fear": "#8B5CF6",
    "Surprise": "#EC4899",
    "Neutral": "#10B981",
    "Disgust": "#84CC16",
    "Uncertain expression": "#6B7280",
}


@st.cache_resource
def load_system_modules():
    """Cache and initialize core computer vision, model, and recommender modules."""
    face_detector = FaceDetector()
    emotion_detector = EmotionDetector()
    recommender = MusicRecommender()
    return face_detector, emotion_detector, recommender


def main():
    # Load backend components
    face_detector, emotion_detector, recommender = load_system_modules()

    # Initialize Session State Variables
    if "camera_active" not in st.session_state:
        st.session_state.camera_active = False
    if "smoother" not in st.session_state:
        st.session_state.smoother = EmotionSmoother()
    if "simulation_emotion" not in st.session_state:
        st.session_state.simulation_emotion = "Happy"

    # ==================== SIDEBAR CONTROLS ====================
    with st.sidebar:
        st.title("🎛️ MoodTune Controls")
        st.markdown("---")

        # Camera Input Mode Selector
        st.subheader("📷 Input Mode")
        input_mode = st.radio(
            "Select Video Source",
            ["Webcam Live Stream", "Upload Image Frame", "Demo Simulation"],
            index=0,
        )

        if input_mode == "Webcam Live Stream":
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("▶️ Start", use_container_width=True):
                    st.session_state.camera_active = True
            with col_btn2:
                if st.button("⏹️ Stop", use_container_width=True):
                    st.session_state.camera_active = False
        elif input_mode == "Demo Simulation":
            st.session_state.simulation_emotion = st.selectbox(
                "Simulated Facial Expression",
                [
                    "Happy",
                    "Sad",
                    "Angry",
                    "Fear",
                    "Surprise",
                    "Neutral",
                    "Disgust",
                ],
            )

        st.markdown("---")

        # AI Hyperparameters
        st.subheader("⚙️ AI Model Settings")
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.90,
            value=0.50,
            step=0.05,
            help="Predictions below this confidence display 'Uncertain expression'",
        )
        smoothing_window = st.slider(
            "Smoothing Window Size",
            min_value=3,
            max_value=15,
            value=7,
            step=2,
            help="Number of recent frames buffered to prevent recommendation flickering",
        )

        # Sync smoother settings
        st.session_state.smoother.confidence_threshold = conf_threshold
        st.session_state.smoother.window_size = smoothing_window

        st.markdown("---")

        # Recommendation Preferences
        st.subheader("🎧 Recommendation Filter")
        num_songs = st.slider("Max Songs", min_value=3, max_value=10, value=5)
        pref_genre = st.selectbox(
            "Preferred Genre",
            [
                "All",
                "Pop",
                "Dance",
                "Upbeat",
                "Acoustic",
                "Calm",
                "Lo-fi",
                "Ambient",
                "Rock",
                "Classical",
            ],
        )
        pref_lang = st.selectbox(
            "Preferred Language",
            ["All", "English", "Instrumental", "Spanish", "Hindi"],
        )

        st.markdown("---")
        st.subheader("📊 Dashboard Display")
        show_probs = st.checkbox("Show Probability Chart", value=True)
        show_history = st.checkbox("Show Emotion Analytics", value=True)
        show_scores = st.checkbox("Show Match Score Breakdown", value=True)

    # ==================== HEADER SECTION ====================
    st.markdown(
        """
        <div class="title-header">
            <h1>🎵 MoodTune AI</h1>
            <p>Real-Time Emotion-Based Music Recommendation System • <i>Computer Vision & Deep CNN Pipeline</i></p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==================== SYSTEM STATUS BAR ====================
    cam_status = (
        "🟢 Connected"
        if (input_mode == "Webcam Live Stream" and st.session_state.camera_active)
        else ("🟢 Ready (Demo)" if input_mode != "Webcam Live Stream" else "🔴 Stopped")
    )
    model_status = (
        "🟢 Trained CNN Loaded"
        if emotion_detector.is_model_loaded()
        else "🟡 Baseline/Pending Training"
    )
    detector_status = "🟢 OpenCV Cascade Active"
    rec_status = (
        f"🟢 Active ({len(recommender.df) if recommender.df is not None else 0} Songs)"
    )

    st.markdown(
        f"""
        <div class="status-bar">
            <div class="status-pill">📷 Camera: {cam_status}</div>
            <div class="status-pill">🧠 CNN Model: {model_status}</div>
            <div class="status-pill">👁️ Detector: {detector_status}</div>
            <div class="status-pill">🎧 Recommender: {rec_status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ==================== MAIN DASHBOARD LAYOUT ====================
    col_left, col_right = st.columns([1.1, 0.9])

    # Variables to store current frame prediction results
    current_frame_bgr: Optional[np.ndarray] = None
    detected_faces: List = []
    primary_bbox: Optional[tuple] = None
    raw_emotion: str = "Neutral"
    raw_confidence: float = 0.0
    probabilities: Dict[str, float] = {
        cls: 1.0 / 7.0 for cls in EmotionDetector.DEFAULT_CLASSES
    }

    # 1. Acquire Frame based on Input Mode
    if input_mode == "Webcam Live Stream":
        with col_left:
            st.subheader("📷 Live Camera Stream")
            if st.session_state.camera_active:
                img_input = st.camera_input(
                    "Capture webcam frame", key="webcam_stream"
                )
                if img_input:
                    bytes_data = img_input.getvalue()
                    pil_img = Image.open(img_input)
                    current_frame_bgr = cv2.cvtColor(
                        np.array(pil_img), cv2.COLOR_RGB2BGR
                    )
            else:
                st.info(
                    "Click **▶️ Start Camera** in the sidebar to activate webcam."
                )

    elif input_mode == "Upload Image Frame":
        with col_left:
            st.subheader("🖼️ Frame Image Upload")
            uploaded_file = st.file_uploader(
                "Upload face image (JPG / PNG)", type=["jpg", "jpeg", "png"]
            )
            if uploaded_file:
                pil_img = Image.open(uploaded_file)
                current_frame_bgr = cv2.cvtColor(
                    np.array(pil_img), cv2.COLOR_RGB2BGR
                )

    elif input_mode == "Demo Simulation":
        with col_left:
            st.subheader("🎭 Demo Simulation Mode")
            sim_emotion = st.session_state.simulation_emotion
            st.success(
                f"Simulating facial expression: **{sim_emotion}**"
            )
            # Create synthetic frame representation
            synthetic_frame = np.zeros((300, 400, 3), dtype=np.uint8)
            cv2.putText(
                synthetic_frame,
                f"Demo: {sim_emotion}",
                (60, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 191, 0),
                2,
            )
            col_left.image(
                cv2.cvtColor(synthetic_frame, cv2.COLOR_BGR2RGB),
                caption=f"Demo Simulation Frame ({sim_emotion})",
                use_container_width=True,
            )
            current_frame_bgr = synthetic_frame
            # Override probabilities for simulation
            probabilities = {
                cls: (0.90 if cls == sim_emotion else 0.016)
                for cls in EmotionDetector.DEFAULT_CLASSES
            }
            raw_emotion = sim_emotion
            raw_confidence = 0.90

    # 2. Process Computer Vision & CNN Pipeline if frame present
    if current_frame_bgr is not None and input_mode != "Demo Simulation":
        # Detect faces
        detected_faces = face_detector.detect_faces(current_frame_bgr)
        face_result = face_detector.get_primary_face(
            current_frame_bgr, detected_faces
        )

        if face_result:
            face_crop, primary_bbox = face_result
            # Preprocess face crop
            processed_tensor = preprocess_face(face_crop)
            # Predict emotion probabilities with CNN
            prediction_res = emotion_detector.predict_emotion(processed_tensor)
            raw_emotion = prediction_res["emotion"]
            raw_confidence = prediction_res["confidence"]
            probabilities = prediction_res["probabilities"]

            # Draw bounding box overlay
            label_text = f"{raw_emotion} ({raw_confidence*100:.1f}%)"
            annotated_frame = face_detector.draw_bounding_boxes(
                current_frame_bgr,
                detected_faces,
                primary_bbox,
                label=label_text,
            )
            # Render annotated camera frame
            col_left.image(
                cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                caption=f"Detected {len(detected_faces)} face(s)",
                use_container_width=True,
            )
        else:
            col_left.warning(
                "⚠️ No face detected in frame. Adjust position or lighting."
            )
            col_left.image(
                cv2.cvtColor(current_frame_bgr, cv2.COLOR_BGR2RGB),
                use_container_width=True,
            )
            raw_emotion = "Uncertain expression"
            raw_confidence = 0.0

    # 3. Update Temporal Smoother
    now_str = datetime.now().strftime("%H:%M:%S")
    (
        stable_emotion,
        smoothed_conf,
        smoothed_probs,
        state_changed,
    ) = st.session_state.smoother.update(
        raw_emotion, raw_confidence, probabilities, timestamp_str=now_str
    )

    # ==================== RIGHT COLUMN: EXPRESSION CARD ====================
    with col_right:
        st.subheader("😊 Expression Analysis")
        emoji = EMOTION_EMOJIS.get(stable_emotion, "😐")
        target_genres = MusicRecommender.EMOTION_GENRE_MAP.get(
            stable_emotion, ["Pop", "Ambient"]
        )
        genre_str = " / ".join(target_genres)

        st.markdown(
            f"""
            <div class="dashboard-card emotion-badge-card">
                <div class="emotion-emoji">{emoji}</div>
                <div class="emotion-title">{stable_emotion.upper()}</div>
                <div style="margin-bottom: 0.8rem;">
                    <span class="confidence-tag">Confidence: {smoothed_conf * 100:.1f}%</span>
                </div>
                <div style="font-size: 0.9rem; color: #94A3B8;">
                    Recommended Mood Genres: <b style="color: #60A5FA;">{genre_str}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Status badge under card
        if smoothed_conf >= 0.70:
            st.success("High Confidence Expression Detection")
        elif smoothed_conf >= conf_threshold:
            st.info("Moderate Confidence Expression Detection")
        else:
            st.warning("Low Confidence / Uncertain Facial Expression")

    # ==================== PROBABILITY DISTRIBUTION CHART ====================
    if show_probs:
        st.markdown("---")
        st.subheader("📊 Emotion Probability Vector (Softmax Distribution)")

        chart_df = pd.DataFrame(
            {
                "Emotion": list(smoothed_probs.keys()),
                "Probability": [v * 100 for v in smoothed_probs.values()],
            }
        ).sort_values(by="Probability", ascending=True)

        fig_bar = px.bar(
            chart_df,
            x="Probability",
            y="Emotion",
            orientation="h",
            text="Probability",
            color="Emotion",
            color_discrete_map=EMOTION_COLORS,
        )
        fig_bar.update_traces(
            texttemplate="%{text:.1f}%", textposition="outside"
        )
        fig_bar.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F3F4F6"),
            xaxis=dict(range=[0, 110], title="Probability (%)"),
            yaxis=dict(title=""),
            height=300,
            margin=dict(l=20, r=20, t=10, b=10),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ==================== RECOMMENDED PLAYLIST SECTION ====================
    st.markdown("---")
    st.subheader(f"🎧 Recommended Playlist for '{stable_emotion}'")

    recommendations = recommender.recommend(
        emotion=stable_emotion,
        num_songs=num_songs,
        preferred_genre=pref_genre if pref_genre != "All" else None,
        preferred_language=pref_lang if pref_lang != "All" else None,
    )

    if not recommendations:
        st.info("No matching tracks found for the selected genre/language filter.")
    else:
        rec_cols = st.columns(min(len(recommendations), 3))
        for idx, track in enumerate(recommendations):
            col_target = rec_cols[idx % 3]
            with col_target:
                rec_score = track.get("recommendation_score", 90.0)
                details = track.get("score_details", {})
                title = track.get("title", "Unknown Track")
                artist = track.get("artist", "Artist")
                genre = track.get("genre", "Pop")
                track_emotion = track.get("emotion", "Happy")
                duration = track.get("duration", "3:30")
                url = track.get("url", "#")

                card_html = f"""
                <div class="song-card">
                    <div class="song-title">🎵 {title}</div>
                    <div class="song-artist">by {artist} • {duration}</div>
                    <div class="song-meta">
                        <span class="tag-genre">{genre}</span>
                        <span class="tag-genre" style="background: rgba(236, 72, 153, 0.2); color: #F472B6;">{track_emotion}</span>
                        <span class="tag-score">Score: {rec_score}%</span>
                    </div>
                """

                if show_scores and details:
                    card_html += f"""
                    <div style="font-size: 0.75rem; color: #9CA3AF; margin-top: 0.5rem; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 0.4rem;">
                        Match: {details.get('emotion_match',0)}% | Genre: {details.get('genre_score',0)}% | Energy: {details.get('energy_score',0)}%
                    </div>
                    """
                card_html += "</div>"
                st.markdown(card_html, unsafe_allow_html=True)
                st.link_button(
                    f"▶️ Play '{title}'", url, use_container_width=True
                )

    # ==================== EMOTION HISTORY & ANALYTICS ====================
    if show_history:
        st.markdown("---")
        st.subheader("📈 Session Emotion Analytics")

        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(
                f"""
                <div class="dashboard-card" style="text-align: center;">
                    <div class="metric-number">{st.session_state.smoother.total_predictions}</div>
                    <div class="metric-label">Total Frame Predictions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col2:
            st.markdown(
                f"""
                <div class="dashboard-card" style="text-align: center;">
                    <div class="metric-number">{st.session_state.smoother.stable_change_count}</div>
                    <div class="metric-label">Stable Emotion Changes</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with m_col3:
            timeline = st.session_state.smoother.history_timeline
            most_common = (
                pd.Series([entry["emotion"] for entry in timeline]).mode()[0]
                if timeline
                else stable_emotion
            )
            st.markdown(
                f"""
                <div class="dashboard-card" style="text-align: center;">
                    <div class="metric-number" style="color: #F59E0B;">{most_common}</div>
                    <div class="metric-label">Most Detected Expression</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Emotion distribution pie chart
        timeline = st.session_state.smoother.history_timeline
        if timeline:
            df_timeline = pd.DataFrame(timeline)
            counts = df_timeline["emotion"].value_counts().reset_index()
            counts.columns = ["Emotion", "Count"]

            fig_pie = px.pie(
                counts,
                values="Count",
                names="Emotion",
                title="Session Expression Distribution",
                color="Emotion",
                color_discrete_map=EMOTION_COLORS,
                hole=0.4,
            )
            fig_pie.update_layout(
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F3F4F6"),
                height=320,
            )
            st.plotly_chart(fig_pie, use_container_width=True)


if __name__ == "__main__":
    main()
