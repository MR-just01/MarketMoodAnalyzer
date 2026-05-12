import polars as pl
import numpy as np
import xgboost as xgb
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

@dataclass
class TradingSignal:
    """Trading signal output from the Decision Engine."""
    ticker: str
    signal: str         # "STRONG BUY", "BUY THE DIP", "PANIC SELL", "HOLD"
    strength: float     # 0.0 to 1.0
    reasoning: str      # Human-readable explanation
    mood_score: float   # The underlying mood score
    price: float        # Current price
    timestamp: str      # When this signal was generated
    ai_prediction: str = "" # XGBoost prediction: "BULLISH" or "BEARISH"

class DecisionEngine:
    """
    Upgraded Trading Decision Engine.
    Fuses FinBERT Sentiment logic with a trained XGBoost Machine Learning model.
    """
    
    # Sentiment direction mapping (FinBERT: 0=Positive, 1=Negative, 2=Neutral)
    DIRECTION = {0: 1.0, 1: -1.0, 2: 0.0}
    AI_LABELS = {0: "BEARISH", 1: "BULLISH"}

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        # 🧠 Load the trained AI brain from the JSON file
        self.model = self._load_model()

    def _load_model(self):
        """Helper to load the XGBoost brain saved from the Jupyter Notebook."""
        # Path logic: Go up from /app/services/ to /ai_service/
        model_path = os.path.join(os.path.dirname(__file__), "..", "..", "nvda_xgboost_model.json")
        
        if os.path.exists(model_path):
            try:
                model = xgb.XGBClassifier()
                model.load_model(model_path)
                logger.info("✅ Decision Engine: AI Brain Loaded Successfully.")
                return model
            except Exception as e:
                logger.error(f"❌ Failed to load AI model: {e}")
                return None
        else:
            logger.warning(f"⚠️ AI model not found at {model_path}. Running on basic logic.")
            return None

    def _compute_weighted_mood(self, sentiment_df: pl.DataFrame) -> float:
        """Calculate confidence-weighted mood score (-1.0 to +1.0)."""
        if len(sentiment_df) == 0:
            return 0.0
        
        # Ensure we have required columns
        cols = sentiment_df.columns
        weighted_scores = []
        for row in sentiment_df.iter_rows(named=True):
            direction = self.DIRECTION.get(row["sentiment"], 0.0)
            confidence = row.get("confidence", 0.5)
            weighted_scores.append(direction * confidence)
        
        return sum(weighted_scores) / len(weighted_scores) if weighted_scores else 0.0

    def calculate_signal(self, ticker: str, price: float, sentiment_df: pl.DataFrame, rsi: float, atr: float) -> TradingSignal:
        """
        Calculates signal by fusing AI predictions with sentiment rules.
        """
        mood = self._compute_weighted_mood(sentiment_df)
        
        # 1. Ask the AI Brain for its prediction
        ai_vote = "NEUTRAL"
        if self.model:
            # Prepare features exactly as the model saw them in training: [RSI, ATR, Mood]
            features = np.array([[rsi, atr, mood]])
            pred_index = self.model.predict(features)[0]
            ai_vote = self.AI_LABELS.get(pred_index, "NEUTRAL")

        # 2. Basic Logic Thresholds (Your existing rules)
        pos_count = len(sentiment_df.filter(pl.col("sentiment") == 0))
        total = len(sentiment_df)

        # 3. Final Signal Selection (Fusing Manual Rules + AI Vote)
        if mood > 0.6 and ai_vote == "BULLISH":
            signal = "STRONG BUY"
            strength = 0.9
        elif mood > 0.4 and ai_vote == "BULLISH":
            signal = "BUY"
            strength = 0.7
        elif mood < -0.5 and ai_vote == "BEARISH":
            signal = "PANIC SELL"
            strength = 0.9
        else:
            signal = "HOLD"
            strength = 0.5

        # 4. Generate reasoning string
        reasoning = (
            f"Sentiment: {mood:+.2f} ({pos_count}/{total} positive). "
            f"Technical: RSI={rsi:.1f}, ATR={atr:.2f}. "
            f"AI Forecast: {ai_vote}."
        )

        return TradingSignal(
            ticker=ticker,
            signal=signal,
            strength=strength,
            reasoning=reasoning,
            mood_score=round(mood, 4),
            price=price,
            timestamp=datetime.now().isoformat(),
            ai_prediction=ai_vote
        )