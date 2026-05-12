import sys
import os
from datetime import datetime, timedelta

# 1. PATH FIX: Ensure Python can see 'ai_service' and 'data_pipelines'
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)

from fastapi import FastAPI, BackgroundTasks
import polars as pl

# Internal Imports
from ai_service.app.services.decision_engine import DecisionEngine
from data_pipelines.builddataset import fetch_and_calculate_technicals, fetch_historical_news, score_sentiment
from ai_service.app.database import supabase  # Make sure this import exists!

app = FastAPI(title="Nvidia AI Trader")

# Initialize our specialized tools
engine = DecisionEngine()

@app.get("/health")
def health_check():
    """Check if the API is up and the AI Model is loaded."""
    return {
        "status": "online", 
        "model_loaded": engine.model is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/run-analysis")
async def run_market_analysis(background_tasks: BackgroundTasks):
    """Triggers the full AI pipeline in the background."""
    background_tasks.add_task(execute_trading_workflow)
    return {"message": "Analysis started. Check your terminal/database for results."}

def execute_trading_workflow():
    try:
        ticker = "NVDA"
        print(f"🚀 [SYSTEM] Starting live analysis for {ticker}...")

        # 1. GET LIVE DATA (Technical Indicators)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        price_df = fetch_and_calculate_technicals(ticker, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        latest_row = price_df.tail(1)
        
        current_price = float(latest_row['Close'][0])
        rsi = float(latest_row['RSI_14'][0])
        atr = float(latest_row['ATR_14'][0])

        # 2. GET LIVE NEWS (Sentiment)
        # Fetching news from the last 24 hours
        news_df = fetch_historical_news(ticker, (end_date - timedelta(days=1)).strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        sentiment_df = score_sentiment(news_df) 

        # 3. ASK THE BRAIN (XGBoost + Logic)
        polars_sentiment = pl.from_pandas(sentiment_df)
        signal = engine.calculate_signal(ticker, current_price, polars_sentiment, rsi, atr)

        # 4. SAVE TO DATABASE (Supabase)
        print(f"🎯 [RESULT] Signal: {signal.signal} | AI Prediction: {signal.ai_prediction}")
        
        data = {
            "ticker": ticker,
            "price": current_price,
            "signal": signal.signal,
            "strength": signal.strength,
            "mood_score": signal.mood_score,
            "ai_prediction": signal.ai_prediction,
            "reasoning": signal.reasoning,
            "created_at": datetime.now().isoformat()
        }

        # Insert into your Supabase table named 'signals'
        result = supabase.table("signals").insert(data).execute()
        print(f"✅ [DATABASE] Signal saved to Supabase successfully.")

    except Exception as e:
        print(f"❌ [ERROR] Workflow failed: {e}")

