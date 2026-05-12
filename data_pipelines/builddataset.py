import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from transformers import pipeline
import requests
import time
from datetime import datetime, timedelta

# ==========================================
# --- CONFIGURATION (Dynamic 5-Year Window) ---
# ==========================================
TICKER = "NVDA"

# Automatically calculate today and exactly 5 years ago
today = datetime.today()
five_years_ago = today - timedelta(days=5*365)

START_DATE = five_years_ago.strftime('%Y-%m-%d')
END_DATE = today.strftime('%Y-%m-%d')

print(f"🚀 Starting Data Pipeline for {TICKER}...")
print(f"📅 Date Range: {START_DATE} to {END_DATE}")

# ==========================================
# PHASE 1: THE MATH (Price Action & Indicators)
# ==========================================
def fetch_and_calculate_technicals(ticker, start, end):
    print("📈 Fetching historical price data...")
    df = yf.download(ticker, start=start, end=end)
    
    if isinstance(df.columns, pd.MultiIndex):
         df.columns = df.columns.get_level_values(0)
    
    # 1. Calculate RSI (14-day)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI_14'] = rsi_indicator.rsi()
    
    # 2. Calculate ATR (14-day Volatility)
    atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR_14'] = atr_indicator.average_true_range()
    
    # 3. Create Target Label (1 if next day goes up, 0 if down)
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Day_Close'] > df['Close']).astype(int)
    
    df = df.dropna().reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    return df[['Date', 'Close', 'RSI_14', 'ATR_14', 'Target']]

# ==========================================
# PHASE 2: THE NEWS (Alpaca API Fetcher)
# ==========================================
def fetch_historical_news(ticker, start_date, end_date):
    print(f"📰 Fetching real historical news for {ticker} via Alpaca API...")
    
    # 🛑 PASTE YOUR KEYS HERE 🛑
    API_KEY = "PKYUIIZI5N27KZQDEDLQJJCH7B"
    API_SECRET = "4SCgPwrGy4F2JLmZPEmyUV3EWV73VEYmwoetGwRHu61G"
    
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {
        "Apca-Api-Key-Id": "PKYUIIZI5N27KZQDEDLQJJCH7B",
        "Apca-Api-Secret-Key": "4SCgPwrGy4F2JLmZPEmyUV3EWV73VEYmwoetGwRHu61G",
        "accept": "application/json"
    }
    
    start_rfc = f"{start_date}T00:00:00Z"
    end_rfc = f"{end_date}T23:59:59Z"
    
    params = {
        "symbols": ticker,
        "start": start_rfc,
        "end": end_rfc,
        "limit": 50,
        "sort": "ASC" 
    }
    
    all_articles = []
    page_token = None
    
    while True:
        if page_token:
            params["page_token"] = page_token
            
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200:
                print(f"❌ API Error: {response.status_code} - {response.text}")
                break
                
            data = response.json()
            articles = data.get("news", [])
            
            if not articles:
                break
                
            for article in articles:
                raw_date = article.get("created_at")
                clean_date = datetime.strptime(raw_date[:10], "%Y-%m-%d").date()
                all_articles.append({
                    "Date": clean_date,
                    "Headline": article.get("headline")
                })
                
            print(f"✅ Fetched {len(articles)} articles. Total so far: {len(all_articles)}")
            time.sleep(1.0) # Rate limiting pause
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
        except Exception as e:
            print(f"⚠️ Network exception occurred: {e}")
            break
            
    return pd.DataFrame(all_articles)

# ==========================================
# PHASE 3: THE AI (FinBERT Scoring)
# ==========================================
def score_sentiment(news_df):
    if news_df.empty:
        print("⚠️ No news data found to score.")
        return pd.DataFrame(columns=['Date', 'FinBERT_Score'])
        
    print("🧠 Loading FinBERT AI Model (This might take a minute...)")
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    
    print("🔍 Analyzing headlines...")
    scores = []
    for headline in news_df['Headline']:
        # Truncate headline to avoid token limits
        result = sentiment_pipeline(headline[:500])[0]
        label = result['label']
        score = result['score']
        
        if label == "positive":
            math_score = score 
        elif label == "negative":
            math_score = -score
        else:
            math_score = 0
            
        scores.append(math_score)
        
    news_df['FinBERT_Score'] = scores
    daily_sentiment = news_df.groupby('Date')['FinBERT_Score'].mean().reset_index()
    return daily_sentiment

# ==========================================
# PHASE 4: THE FUSION (Merging the Dataset)
# ==========================================
def build_dataset():
    # 1. Get Math
    price_df = fetch_and_calculate_technicals(TICKER, START_DATE, END_DATE)
    
    # 2. Get News
    news_df = fetch_historical_news(TICKER, START_DATE, END_DATE)
    
    # 3. Score News
    sentiment_df = score_sentiment(news_df)
    
    # 4. Merge them together matching the exact dates
    print("🔗 Fusing Math and AI Sentiment...")
    final_dataset = pd.merge(price_df, sentiment_df, on='Date', how='inner')
    
    # Save to CSV
    output_filename = f"training_data_{TICKER}_5YR.csv"
    final_dataset.to_csv(output_filename, index=False)
    print(f"✅ SUCCESS! Dataset saved as {output_filename}")
    print(final_dataset.head())

if __name__ == "__main__":
    build_dataset()