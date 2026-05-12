import polars as pl
import pandas as pd
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
from transformers import pipeline
import requests
import time
from datetime import datetime, timedelta

TICKER = "LLY"
today = datetime.today()
five_years_ago = today - timedelta(days=5*365)
START_DATE = five_years_ago.strftime('%Y-%m-%d')
END_DATE = today.strftime('%Y-%m-%d')

print(f"🚀 Starting Polars Data Pipeline for {TICKER}...")

def fetch_and_calculate_technicals(ticker, start, end):
    df = yf.download(ticker, start=start, end=end)
    if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
    df['RSI_14'] = RSIIndicator(close=df['Close'], window=14).rsi()
    df['ATR_14'] = AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    df = df.dropna().reset_index()
    pl_df = pl.from_pandas(df)
    pl_df = pl_df.with_columns([pl.col("Date").cast(pl.Date), pl.col("Close").shift(-1).alias("Next_Day_Close")])
    pl_df = pl_df.with_columns([(pl.col("Next_Day_Close") > pl.col("Close")).cast(pl.Int32).alias("Target")]).drop_nulls()
    return pl_df.select(['Date', 'Close', 'RSI_14', 'ATR_14', 'Target'])

def fetch_historical_news(ticker, start_date, end_date):
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
        if page_token: params["page_token"] = page_token
        try:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code != 200: break
            data = response.json()
            articles = data.get("news", [])
            if not articles: break
            for article in articles:
                clean_date = datetime.strptime(article.get("created_at")[:10], "%Y-%m-%d").date()
                all_articles.append({"Date": clean_date, "Headline": article.get("headline")})
            time.sleep(1.0)
            page_token = data.get("next_page_token")
            if not page_token: break
        except Exception: break
    if not all_articles: return pl.DataFrame(schema={"Date": pl.Date, "Headline": pl.Utf8})
    return pl.DataFrame(all_articles)

def score_sentiment(news_df):
    if news_df.is_empty(): return pl.DataFrame(schema={"Date": pl.Date, "FinBERT_Score": pl.Float64})
    sentiment_pipeline = pipeline("text-classification", model="ProsusAI/finbert")
    scores = []
    for headline in news_df.get_column('Headline'):
        result = sentiment_pipeline(headline[:500])[0]
        if result['label'] == "positive": scores.append(result['score'])
        elif result['label'] == "negative": scores.append(-result['score'])
        else: scores.append(0.0)
    news_df = news_df.with_columns(pl.Series(name="FinBERT_Score", values=scores))
    return news_df.group_by('Date').agg(pl.col('FinBERT_Score').mean())

def build_dataset():
    price_df = fetch_and_calculate_technicals(TICKER, START_DATE, END_DATE)
    news_df = fetch_historical_news(TICKER, START_DATE, END_DATE)
    sentiment_df = score_sentiment(news_df)
    final_dataset = price_df.join(sentiment_df, on='Date', how='inner')
    final_dataset.write_csv(f"training_data_{TICKER}_5YR.csv")
    print(f"✅ SUCCESS! Dataset saved for {TICKER}")

if __name__ == "__main__":
    build_dataset()