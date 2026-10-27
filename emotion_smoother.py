"""
Temporal Emotion Smoother for MoodTune AI.
Implements windowed majority voting, moving probability average, confidence thresholding,
and cooldown logic to stabilize live webcam emotion predictions.
"""

from collections import deque, Counter
from typing import Dict, Any, List, Optional, Tuple


class EmotionSmoother:
    """Stabilizes frame-by-frame facial expression predictions across video frames."""

    def __init__(
        self, window_size: int = 7, confidence_threshold: float = 0.50
    ):
        """Initialize EmotionSmoother.

        Args:
            window_size (int): Number of recent frames to buffer for temporal smoothing.
            confidence_threshold (float): Minimum confidence required to declare a stable emotion.
        """
        self.window_size = window_size
        self.confidence_threshold = confidence_threshold

        self.history_buffer: deque = deque(maxlen=window_size)
        self.probability_buffer: deque = deque(maxlen=window_size)

        self.current_stable_emotion: str = "Neutral"
        self.stable_change_count: int = 0
        self.total_predictions: int = 0
        self.history_timeline: List[Dict[str, Any]] = []

    def update(
        self,
        emotion: str,
        confidence: float,
        probabilities: Dict[str, float],
        timestamp_str: Optional[str] = None,
    ) -> Tuple[str, float, Dict[str, float], bool]:
        """Update window buffer with a new frame prediction and compute smoothed stable emotion.

        Args:
            emotion (str): Raw frame predicted emotion label.
            confidence (float): Raw frame prediction confidence (0.0 - 1.0).
            probabilities (Dict[str, float]): Raw 7-class probability map.
            timestamp_str (str, optional): Formatted time string for log history.

        Returns:
            Tuple[str, float, Dict[str, float], bool]:
                (stable_emotion, smoothed_confidence, smoothed_probabilities, state_changed_flag)
        """
        self.total_predictions += 1
        self.history_buffer.append(emotion)
        self.probability_buffer.append(probabilities)

        # Compute moving average of probabilities across the window
        smoothed_probabilities: Dict[str, float] = {}
        all_keys = set().union(*[p.keys() for p in self.probability_buffer if isinstance(p, dict)])
        if not all_keys and probabilities:
            all_keys = set(probabilities.keys())

        buf_len = max(len(self.probability_buffer), 1)
        for class_name in all_keys:
            class_avg = sum(
                p.get(class_name, 0.0) for p in self.probability_buffer if isinstance(p, dict)
            ) / buf_len
            smoothed_probabilities[class_name] = round(class_avg, 4)

        if not smoothed_probabilities:
            smoothed_probabilities = {emotion: 1.0} if emotion else {"Neutral": 1.0}

        # Find top smoothed class and its average probability score
        top_smoothed_emotion = max(
            smoothed_probabilities, key=lambda k: smoothed_probabilities[k]
        )
        smoothed_confidence = smoothed_probabilities[top_smoothed_emotion]

        # Majority vote check
        vote_counts = Counter(self.history_buffer)
        if vote_counts:
            majority_emotion, majority_votes = vote_counts.most_common(1)[0]
        else:
            majority_emotion, majority_votes = (top_smoothed_emotion, 1)

        # Apply confidence thresholding filter
        if smoothed_confidence < self.confidence_threshold:
            candidate_emotion = "Uncertain expression"
        else:
            # Combine majority vote with smoothed max emotion
            if majority_votes >= (len(self.history_buffer) // 2 + 1):
                candidate_emotion = majority_emotion
            else:
                candidate_emotion = top_smoothed_emotion

        # Track state transition changes
        state_changed = False
        if (
            candidate_emotion != "Uncertain expression"
            and candidate_emotion != self.current_stable_emotion
        ):
            self.current_stable_emotion = candidate_emotion
            self.stable_change_count += 1
            state_changed = True

        # Log timeline history entry
        if timestamp_str:
            self.history_timeline.append(
                {
                    "time": timestamp_str,
                    "emotion": candidate_emotion,
                    "confidence": smoothed_confidence,
                }
            )
            # Retain last 100 timeline records
            if len(self.history_timeline) > 100:
                self.history_timeline.pop(0)

        display_emotion = (
            self.current_stable_emotion
            if candidate_emotion != "Uncertain expression"
            else "Uncertain expression"
        )

        return (
            display_emotion,
            smoothed_confidence,
            smoothed_probabilities,
            state_changed,
        )

    def reset(self) -> None:
        """Clear buffer state and session history."""
        self.history_buffer.clear()
        self.probability_buffer.clear()
        self.current_stable_emotion = "Neutral"
        self.stable_change_count = 0
        self.total_predictions = 0
        self.history_timeline.clear()
