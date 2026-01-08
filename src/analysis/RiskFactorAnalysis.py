import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import re

# Config
# Note: Run from Project Root (ChiviThesis/)
CSV_PATH = Path("Data/sec_10q/announcements_with_prices.csv")
TEXT_DIR = Path("Data/sec_10q/risk_factors")
OUTPUT_CSV = Path("Data/sec_10q/word_impact_analysis.csv")

def load_data():
    print("Loading price data...")
    df = pd.read_csv(CSV_PATH)
    
    # Calculate Returns (Target)
    # Return = (Price_T+5 - Price_T-1) / Price_T-1
    # We use T-1 as base to capture pre-announcement sentiment or immediate reaction
    
    df['Return_5D'] = (df['price_T+5'] - df['price_T-1']) / df['price_T-1']
    
    # Filter valid returns
    df = df.dropna(subset=['Return_5D'])
    
    # Load Texts
    print("Loading text data...")
    texts = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        acc = row['accessionNumber']
        txt_path = TEXT_DIR / f"{acc}.txt"
        
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                
            # Filter out placeholders or empty files
            if content and content != "SECTION_NOT_FOUND_OR_REFERENCE_ONLY" and len(content) > 100:
                texts.append(content)
                valid_indices.append(idx)
    
    # Filter DataFrame to match loaded texts
    df_clean = df.loc[valid_indices].copy()
    df_clean['text'] = texts
    
    print(f"Data loaded: {len(df_clean)} samples (from {len(df)} original rows)")
    return df_clean

def analyze_impact(df):
    print("Vectorizing text (TF-IDF)...")
    # TF-IDF Configuration
    # min_df=0.01: Ignore words appearing in less than 1% of docs (typos, unique names)
    # max_df=0.95: Ignore words appearing in more than 95% of docs (boilerplate)
    # stop_words='english': Remove common words
    tfidf = TfidfVectorizer(max_features=5000, 
                            stop_words='english', 
                            min_df=0.01, 
                            max_df=0.95,
                            ngram_range=(1, 2)) # Unigrams and Bigrams
    
    X = tfidf.fit_transform(df['text'])
    y = df['Return_5D'].values
    
    feature_names = tfidf.get_feature_names_out()
    
    print("Training Ridge Regression Model...")
    # Ridge Regression (Linear Regression with L2 regularization)
    # Good for handling multicollinearity in text data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance:")
    print(f"  MSE: {mse:.6f}")
    print(f"  R2 Score: {r2:.6f} (Note: Low R2 is expected for financial time series)")
    
    # Extract Coefficients
    coefs = model.coef_
    
    # Create DataFrame of words and their impact
    impact_df = pd.DataFrame({
        'word': feature_names,
        'coefficient': coefs,
        'abs_coefficient': np.abs(coefs)
    })
    
    impact_df = impact_df.sort_values(by='coefficient', ascending=False)
    
    return impact_df

def main():
    try:
        df = load_data()
        if df.empty:
            print("No valid data found to analyze.")
            return

        impact_df = analyze_impact(df)
        
        print(f"\nSaving results to {OUTPUT_CSV}...")
        impact_df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n--- Top 20 Positive Impact Words (Tend to increase price) ---")
        print(impact_df.head(20)[['word', 'coefficient']])
        
        print("\n--- Top 20 Negative Impact Words (Tend to decrease price) ---")
        print(impact_df.tail(20)[['word', 'coefficient']])
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
