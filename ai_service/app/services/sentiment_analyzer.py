from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import polars as pl


class SentimentAnalyzer:
    """
    Financial sentiment analyzer powered by FinBERT.
    
    Classifies text as:
      0 -> Positive (Bullish)
      1 -> Negative (Bearish)
      2 -> Neutral
    
    Each prediction includes a confidence score (0.0 to 1.0).
    """
    
    LABELS = {0: "Positive", 1: "Negative", 2: "Neutral"}
    
    def __init__(self):
        # Load the official FinBERT model from Hugging Face
        self.model_name = "ProsusAI/finbert"
        print(f"Loading FinBERT model ({self.model_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()  # Set to evaluation mode (disables dropout)
        print("FinBERT model loaded successfully.")

    def analyze_dataframe(self, df):
        """
        Analyze sentiment of headlines directly from a Polars DataFrame.
        
        Args:
            df: Polars DataFrame with a 'headline' column
            
        Returns:
            Polars DataFrame with 'sentiment' and 'confidence' columns added
        """
        headlines = df["headline"].to_list()
        
        if not headlines:
            return df.with_columns([
                pl.lit(None).alias("sentiment"),
                pl.lit(None).alias("confidence")
            ])
        
        # Tokenize all headlines at once
        inputs = self.tokenizer(
            headlines, 
            padding=True, 
            truncation=True, 
            max_length=512,
            return_tensors='pt'
        )
        
        # Run inference WITHOUT computing gradients (saves memory + speed)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Convert raw logits to probabilities via Softmax
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        # Extract sentiment label and confidence for each headline
        sentiments = torch.argmax(predictions, dim=-1).tolist()
        confidences = torch.max(predictions, dim=-1).values.tolist()
        
        # Add results as new columns
        result_df = df.with_columns([
            pl.Series("sentiment", sentiments).cast(pl.Int32),
            pl.Series("confidence", confidences).cast(pl.Float64)
        ])
        
        return result_df

    def analyze_headlines(self, file_path):
        """
        Analyze sentiment from a CSV file (backward-compatible method).
        
        Args:
            file_path: Path to CSV with a 'headline' column
            
        Returns:
            Polars DataFrame with sentiment and confidence columns
        """
        print(f"Loading data from {file_path}...")
        df = pl.read_csv(file_path)
        
        result_df = self.analyze_dataframe(df)
        
        # Save the analyzed data
        output_path = file_path.replace(".csv", "_analyzed.csv")
        result_df.write_csv(output_path)
        print(f"Analysis complete! Saved to {output_path}")
        return result_df


if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.analyze_headlines("data/NVDA_news.csv")