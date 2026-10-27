"""
Music Recommendation Engine for MoodTune AI.
Filters track database by facial expression, computes multi-dimensional match scores,
and ranks personalized playlists for Streamlit rendering.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd


class MusicRecommender:
    """Emotion-aware music recommender system using weighted multi-criteria scoring."""

    # Explicit mapping between facial expressions and target musical genres/characteristics
    EMOTION_GENRE_MAP = {
        "Happy": ["Pop", "Dance", "Upbeat"],
        "Sad": ["Acoustic", "Calm", "Lo-fi"],
        "Angry": ["Chill", "Ambient", "Instrumental", "Rock"],
        "Fear": ["Ambient", "Calm", "Classical"],
        "Surprise": ["Pop", "Dance", "Energetic"],
        "Neutral": ["Lo-fi", "Ambient", "Focus"],
        "Disgust": ["Calm", "Instrumental", "Ambient"],
        "Uncertain expression": ["Lo-fi", "Ambient", "Calm"],
    }

    # Desired ideal energy range [0.0, 1.0] for each emotion
    EMOTION_ENERGY_TARGETS = {
        "Happy": 0.85,
        "Sad": 0.25,
        "Angry": 0.80,
        "Fear": 0.25,
        "Surprise": 0.90,
        "Neutral": 0.45,
        "Disgust": 0.30,
        "Uncertain expression": 0.40,
    }

    def __init__(self, csv_path: str = "data/music.csv"):
        """Initialize MusicRecommender.

        Args:
            csv_path (str): Path to CSV music database.
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            repo_root = Path(__file__).resolve().parent.parent
            fallback = repo_root / csv_path
            if fallback.exists():
                self.csv_path = fallback
        self.df: Optional[pd.DataFrame] = None
        self._load_database()

    def _load_database(self) -> None:
        """Load and validate music database from CSV."""
        if not self.csv_path.exists():
            print(f"Warning: Music database CSV not found at {self.csv_path}")
            return

        try:
            self.df = pd.read_csv(self.csv_path)
            # Ensure required numeric columns are correctly typed
            if "mood_score" in self.df.columns:
                self.df["mood_score"] = pd.to_numeric(
                    self.df["mood_score"], errors="coerce"
                ).fillna(50)
            if "energy" in self.df.columns:
                self.df["energy"] = pd.to_numeric(
                    self.df["energy"], errors="coerce"
                ).fillna(0.5)
        except Exception as e:
            print(f"Error loading music CSV from {self.csv_path}: {e}")
            self.df = None

    def calculate_score(
        self,
        row: pd.Series,
        target_emotion: str,
        preferred_genre: Optional[str] = None,
    ) -> Dict[str, float]:
        """Compute recommendation score based on emotion, genre, energy, and mood score.

        Formula:
        recommendation_score = emotion_match_score (40%)
                             + genre_score (30%)
                             + energy_score (15%)
                             + mood_score (15%)

        Args:
            row (pd.Series): Single track record from dataframe.
            target_emotion (str): Detected user emotion.
            preferred_genre (str, optional): User-selected genre override.

        Returns:
            Dict[str, float]: Score breakdown dictionary.
        """
        # 1. Emotion Match Score (40 pts)
        track_emotion = str(row.get("emotion", "")).strip()
        if track_emotion.lower() == target_emotion.lower():
            emotion_match = 100.0
        elif target_emotion in self.EMOTION_GENRE_MAP and str(
            row.get("genre", "")
        ) in self.EMOTION_GENRE_MAP[target_emotion]:
            emotion_match = 85.0
        else:
            emotion_match = 50.0

        # 2. Genre Compatibility Score (30 pts)
        track_genre = str(row.get("genre", "")).strip()
        target_genres = self.EMOTION_GENRE_MAP.get(
            target_emotion, ["Pop", "Ambient"]
        )

        if preferred_genre and preferred_genre.lower() != "all":
            if track_genre.lower() == preferred_genre.lower():
                genre_score = 100.0
            else:
                genre_score = 40.0
        elif track_genre in target_genres:
            genre_score = 95.0
        else:
            genre_score = 60.0

        # 3. Energy Score (15 pts) - based on proximity to target emotion energy
        track_energy = float(row.get("energy", 0.5))
        target_energy = self.EMOTION_ENERGY_TARGETS.get(target_emotion, 0.5)
        energy_diff = abs(track_energy - target_energy)
        energy_score = max(0.0, 100.0 * (1.0 - energy_diff))

        # 4. Mood Score (15 pts) - baseline mood score from dataset
        mood_score = float(row.get("mood_score", 50))

        # Weighted final score (0 - 100)
        final_score = (
            (0.40 * emotion_match)
            + (0.30 * genre_score)
            + (0.15 * energy_score)
            + (0.15 * mood_score)
        )

        return {
            "final_score": round(final_score, 1),
            "emotion_match": round(emotion_match, 1),
            "genre_score": round(genre_score, 1),
            "energy_score": round(energy_score, 1),
            "mood_score": round(mood_score, 1),
        }

    def recommend(
        self,
        emotion: str,
        num_songs: int = 5,
        preferred_genre: Optional[str] = None,
        preferred_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate ranked list of music recommendations tailored to emotion and preferences.

        Args:
            emotion (str): Primary user facial expression.
            num_songs (int): Number of top song recommendations to return.
            preferred_genre (str, optional): Optional genre filter.
            preferred_language (str, optional): Optional language filter.

        Returns:
            List[Dict[str, Any]]: Ranked list of song recommendation objects.
        """
        if self.df is None or self.df.empty:
            return []

        filtered_df = self.df.copy()

        # Filter by language if specified
        if preferred_language and preferred_language.lower() != "all":
            lang_match = filtered_df["language"].astype(str).str.lower() == preferred_language.lower()
            if lang_match.any():
                filtered_df = filtered_df[lang_match]

        # Filter by preferred genre if specified
        if preferred_genre and preferred_genre.lower() != "all":
            genre_match = filtered_df["genre"].astype(str).str.lower() == preferred_genre.lower()
            if genre_match.any():
                filtered_df = filtered_df[genre_match]

        # Compute recommendation scores for all candidate songs
        scored_tracks = []
        for _, row in filtered_df.iterrows():
            scores = self.calculate_score(row, emotion, preferred_genre)
            track_dict = row.to_dict()
            track_dict["recommendation_score"] = scores["final_score"]
            track_dict["score_details"] = scores
            scored_tracks.append(track_dict)

        # Sort descending by recommendation score
        scored_tracks.sort(
            key=lambda x: x["recommendation_score"], reverse=True
        )

        return scored_tracks[:num_songs]
