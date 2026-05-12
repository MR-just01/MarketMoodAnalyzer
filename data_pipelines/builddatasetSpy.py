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
TICKER = "SPY"

today = datetime.today()
five_years_ago = today - timedelta(days=5*365)
START_DATE = five_years_ago.strftime('%Y-%m-%d')
END_DATE = today.strftime('%Y-%m-%d')

print(f"🚀 Starting Data Pipeline for {TICKER}...")
print(f"📅 Date Range: {START_DATE} to {END_DATE}")

def fetch_and_calculate_technicals(ticker, start, end):
    print("📈 Fetching historical price data...")
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex):
         df.columns = df.columns.get_level_values(0)
    rsi_indicator = RSIIndicator(close=df['Close'], window=14)
    df['RSI_14'] = rsi_indicator.rsi()
    atr_indicator = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14)
    df['ATR_14'] = atr_indicator.average_true_range()
    df['Next_Day_Close'] = df['Close'].shift(-1)
    df['Target'] = (df['Next_Day_Close'] > df['Close']).astype(int)
    df = df.dropna().reset_index()
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    return df[['Date', 'Close', 'RSI_14', 'ATR_14', 'Target']]

def fetch_historical_news(ticker, start_date, end_date):
    print(f"📰 Fetching real historical news for {ticker} via Alpaca API...")
    API_KEY = "PKYUIIZI5N27KZQDEDLQJJCH7B"
    API_SECRET = "4SCgPwrGy4F2JLmZPEmyUV3EWV73VEYmwoetGwRHu61G"
    url = "https://data.alpaca.markets/v1beta1/news"
    headers = {"Apca-Api-Key-Id": API_KEY, "Apca-Api-Secret-Key": API_SECRET, "accept": "application/json"}
    start_rfc = f"{start_date}T00:00:00Z"
    end_rfc = f"{end_date}T23:59:59Z"
    params = {"symbols": ticker, "start": start_rfc, "end": end_rfc, "limit": 50, "sort": "ASC"}
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
                all_articles.append({"Date": clean_date, "Headline": article.get("headline")})
            print(f"✅ Fetched {len(articles)} articles. Total so far: {len(all_articles)}")
            time.sleep(1.0)
            page_token = data.get("next_page_token")
            if not page_token:
                break
        except Exception as e:
            print(f"⚠️ Network exception occurred: {e}")
            break
    return pd.DataFrame(all_articles)

def score_sentiment(news_df):
    if news_df.empty:
        print("⚠️ No news data found to score.")
        return pd.DataFrame(columns=['Date', 'FinBERT_Score'])
    print("🧠 Loading FinBERT AI Model (This might take a minute...)")
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    print("🔍 Analyzing headlines...")
    scores = []
    for headline in news_df['Headline']:
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

def build_dataset():
    price_df = fetch_and_calculate_technicals(TICKER, START_DATE, END_DATE)
    news_df = fetch_historical_news(TICKER, START_DATE, END_DATE)
    sentiment_df = score_sentiment(news_df)
    print("🔗 Fusing Math and AI Sentiment...")
    final_dataset = pd.merge(price_df, sentiment_df, on='Date', how='inner')
    output_filename = f"training_data_{TICKER}_5YR.csv"
    final_dataset.to_csv(output_filename, index=False)
    print(f"✅ SUCCESS! Dataset saved as {output_filename}")

if __name__ == "__main__":
    build_dataset()