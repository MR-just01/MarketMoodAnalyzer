import polars as pl
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TickerMood:
    """Mood analysis result for a single ticker."""
    ticker: str
    mood: str                    # "PANIC", "HYPE", or "NEUTRAL"
    mood_score: float            # -100 to +100
    positive_count: int = 0
    negative_count: int = 0
    neutral_count: int = 0
    total_headlines: int = 0
    avg_confidence: float = 0.0
    top_bullish: List[str] = field(default_factory=list)
    top_bearish: List[str] = field(default_factory=list)


@dataclass
class MarketMood:
    """Overall market mood analysis result."""
    mood: str                    # "PANIC", "HYPE", or "NEUTRAL"
    mood_score: float            # -100 to +100
    confidence: float            # 0 to 100 (how confident we are in this reading)
    reasoning: str               # Human-readable explanation
    ticker_moods: Dict[str, TickerMood] = field(default_factory=dict)
    total_headlines: int = 0


class MoodEngine:
    """
    Market Mood Intelligence Engine.
    
    Takes analyzed sentiment data from FinBERT and produces a single
    clear signal: PANIC, HYPE, or NEUTRAL.
    
    Scoring Logic:
        For each headline:
            weighted_score = direction * confidence
            where: Positive = +1, Negative = -1, Neutral = 0
        
        mood_score = mean(weighted_scores) * 100
        
        If mood_score > +15  -> HYPE    (market is optimistic)
        If mood_score < -15  -> PANIC   (market is fearful)
        Otherwise            -> NEUTRAL (mixed signals)
    """
    
    # Mood classification thresholds on -100 to +100 scale
    HYPE_THRESHOLD = 15
    PANIC_THRESHOLD = -15
    
    # Sentiment direction mapping (FinBERT labels)
    DIRECTION = {0: +1.0, 1: -1.0, 2: 0.0}  # Positive, Negative, Neutral
    
    def classify_mood(self, score: float) -> str:
        """Classify a mood score into PANIC, HYPE, or NEUTRAL."""
        if score >= self.HYPE_THRESHOLD:
            return "HYPE"
        elif score <= self.PANIC_THRESHOLD:
            return "PANIC"
        else:
            return "NEUTRAL"
    
    def compute_ticker_mood(self, ticker: str, analyzed_df: pl.DataFrame) -> TickerMood:
        """
        Compute mood for a single ticker from its analyzed headlines.
        
        Args:
            ticker: Ticker symbol (e.g. "NVDA")
            analyzed_df: DataFrame with 'headline', 'sentiment', 'confidence' columns
        
        Returns:
            TickerMood with score, classification, and top headlines
        """
        if len(analyzed_df) == 0:
            return TickerMood(ticker=ticker, mood="NEUTRAL", mood_score=0.0)
        
        # Count sentiments
        pos_count = len(analyzed_df.filter(pl.col("sentiment") == 0))
        neg_count = len(analyzed_df.filter(pl.col("sentiment") == 1))
        neu_count = len(analyzed_df.filter(pl.col("sentiment") == 2))
        total = len(analyzed_df)
        
        # Compute confidence-weighted mood score
        weighted_scores = []
        for row in analyzed_df.iter_rows(named=True):
            direction = self.DIRECTION.get(row["sentiment"], 0.0)
            confidence = row.get("confidence", 0.5)
            weighted_scores.append(direction * confidence)
        
        mood_score = (sum(weighted_scores) / len(weighted_scores)) * 100 if weighted_scores else 0.0
        
        # Average confidence across all headlines
        avg_conf = analyzed_df["confidence"].mean() if "confidence" in analyzed_df.columns else 0.0
        
        # Extract top bullish headlines (positive, sorted by confidence)
        positive_df = analyzed_df.filter(pl.col("sentiment") == 0).sort("confidence", descending=True)
        top_bullish = positive_df["headline"].head(3).to_list() if len(positive_df) > 0 else []
        
        # Extract top bearish headlines (negative, sorted by confidence)
        negative_df = analyzed_df.filter(pl.col("sentiment") == 1).sort("confidence", descending=True)
        top_bearish = negative_df["headline"].head(3).to_list() if len(negative_df) > 0 else []
        
        return TickerMood(
            ticker=ticker,
            mood=self.classify_mood(mood_score),
            mood_score=round(mood_score, 1),
            positive_count=pos_count,
            negative_count=neg_count,
            neutral_count=neu_count,
            total_headlines=total,
            avg_confidence=round(avg_conf, 3) if avg_conf else 0.0,
            top_bullish=top_bullish,
            top_bearish=top_bearish,
        )
    
    def compute_market_mood(self, all_analyzed: Dict[str, pl.DataFrame]) -> MarketMood:
        """
        Compute overall market mood across all tickers.
        
        Args:
            all_analyzed: Dict mapping ticker -> analyzed DataFrame
        
        Returns:
            MarketMood with overall score, classification, and per-ticker breakdowns
        """
        if not all_analyzed:
            return MarketMood(
                mood="NEUTRAL",
                mood_score=0.0,
                confidence=0.0,
                reasoning="No data available for analysis.",
            )
        
        # Compute mood for each ticker
        ticker_moods = {}
        all_weighted_scores = []
        total_headlines = 0
        
        for ticker, adf in all_analyzed.items():
            tm = self.compute_ticker_mood(ticker, adf)
            ticker_moods[ticker] = tm
            total_headlines += tm.total_headlines
            
            # Collect all weighted scores for overall calculation
            for row in adf.iter_rows(named=True):
                direction = self.DIRECTION.get(row["sentiment"], 0.0)
                confidence = row.get("confidence", 0.5)
                all_weighted_scores.append(direction * confidence)
        
        # Overall mood score (weighted average across ALL headlines)
        overall_score = (sum(all_weighted_scores) / len(all_weighted_scores)) * 100 if all_weighted_scores else 0.0
        overall_mood = self.classify_mood(overall_score)
        
        # Confidence: how strongly the data points in one direction
        # High confidence = most headlines agree, low = mixed signals
        if all_weighted_scores:
            abs_scores = [abs(s) for s in all_weighted_scores]
            confidence = (sum(abs_scores) / len(abs_scores)) * 100
        else:
            confidence = 0.0
        
        # Generate human-readable reasoning
        reasoning = self._generate_reasoning(overall_mood, overall_score, ticker_moods)
        
        return MarketMood(
            mood=overall_mood,
            mood_score=round(overall_score, 1),
            confidence=round(confidence, 1),
            reasoning=reasoning,
            ticker_moods=ticker_moods,
            total_headlines=total_headlines,
        )
    
    def _generate_reasoning(self, mood: str, score: float, ticker_moods: Dict[str, TickerMood]) -> str:
        """Generate a plain-English explanation of the market mood."""
        
        # Opening statement
        mood_words = {
            "PANIC": "fearful",
            "HYPE": "optimistic", 
            "NEUTRAL": "showing mixed signals"
        }
        
        parts = [f"Market mood is **{mood}** (score: {score:+.1f}). The overall sentiment is {mood_words[mood]}."]
        
        # Per-ticker insights
        for ticker, tm in ticker_moods.items():
            total = tm.total_headlines
            if total == 0:
                continue
            
            if tm.mood == "PANIC":
                neg_pct = (tm.negative_count / total) * 100
                parts.append(
                    f"{ticker} is **bearish** — {neg_pct:.0f}% of {total} headlines are negative (score: {tm.mood_score:+.1f})."
                )
            elif tm.mood == "HYPE":
                pos_pct = (tm.positive_count / total) * 100
                parts.append(
                    f"{ticker} is **bullish** — {pos_pct:.0f}% of {total} headlines are positive (score: {tm.mood_score:+.1f})."
                )
            else:
                parts.append(
                    f"{ticker} is **neutral** with mixed signals across {total} headlines (score: {tm.mood_score:+.1f})."
                )
        
        return " ".join(parts)
