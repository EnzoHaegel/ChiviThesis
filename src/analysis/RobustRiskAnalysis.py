import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import datetime

# Config
# Note: Run from Project Root (ChiviThesis/)
CSV_PATH = Path("Data/sec_10q/announcements_with_prices.csv")
TEXT_DIR = Path("Data/sec_10q/risk_factors")
OUTPUT_DIR = Path("Data/sec_10q/robust_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prep_data():
    print("Loading data...")
    df = pd.read_csv(CSV_PATH)
    
    # Parse Dates
    date_cols = [c for c in df.columns if 'date' in c or 'filing' in c]
    for c in date_cols:
        df[c] = pd.to_datetime(df[c], errors='coerce')
        
    # Get Date Range for SPY
    min_date = df['date_T-1'].min()
    max_date = df['date_T+10'].max()
    
    # Buffer dates slightly
    start_sp = min_date - datetime.timedelta(days=5)
    end_sp = max_date + datetime.timedelta(days=5)
    
    print(f"Fetching SPY (Market) data from {start_sp.date()} to {end_sp.date()}...")
    spy = yf.download("SPY", start=start_sp, end=end_sp, progress=False, auto_adjust=True)
    
    # Use 'Close' column (auto_adjust=True makes Close = Adj Close)
    # Ensure index is datetime normalized (no time component)
    spy.index = spy.index.normalize()
    
    # Flatten SPY columns if MultiIndex (common in new yfinance)
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = spy.columns.get_level_values(0)

    # Helper to lookup SPY price
    def get_spy_price(target_date):
        try:
            # If exact date missing, look back up to 3 days (weekend handling already done in main script, but SPY might differ slightly)
            # Use 'asof' for cleaner logic on sorted index
            idx = spy.index.asof(target_date)
            if pd.isna(idx): return np.nan
            val = spy.at[idx, 'Close']
            # Handle if val is series (duplicates)
            if isinstance(val, pd.Series): val = val.iloc[0]
            return float(val)
        except:
            return np.nan

    print("Calculating Market-Adjusted Returns (Alpha)...")
    
    horizons = [1, 5, 10]
    
    for h in horizons:
        # Stock Return
        col_price_base = 'price_T-1'
        col_price_target = f'price_T+{h}'
        col_date_base = 'date_T-1'
        col_date_target = f'date_T+{h}'
        
        # Calculate Vectors
        stock_ret = (df[col_price_target] - df[col_price_base]) / df[col_price_base]
        df[f'Stock_Ret_T{h}'] = stock_ret
        
        # Calculate Market Return for same periods
        spy_base = df[col_date_base].apply(get_spy_price)
        spy_target = df[col_date_target].apply(get_spy_price)
        
        # Ensure they are Series
        spy_base = pd.Series(spy_base, index=df.index)
        spy_target = pd.Series(spy_target, index=df.index)
        
        market_ret = (spy_target - spy_base) / spy_base
        
        # Alpha
        df[f'Alpha_T{h}'] = df[f'Stock_Ret_T{h}'] - market_ret

    # Drop NaNs created by missing prices or dates
    df = df.dropna(subset=[f'Alpha_T{h}' for h in horizons])
    
    # Load Texts
    print("Loading texts...")
    texts = []
    text_lengths = []
    valid_indices = []
    
    for idx, row in df.iterrows():
        acc = row['accessionNumber']
        txt_path = TEXT_DIR / f"{acc}.txt"
        
        content = ""
        if txt_path.exists():
            with open(txt_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        
        if content and content != "SECTION_NOT_FOUND_OR_REFERENCE_ONLY" and len(content) > 100:
            texts.append(content)
            text_lengths.append(len(content))
            valid_indices.append(idx)
            
    df_clean = df.loc[valid_indices].copy()
    df_clean['text'] = texts
    df_clean['log_len'] = np.log(text_lengths)
    
    print(f"Final dataset size: {len(df_clean)}")
    return df_clean

def train_models(df, tfidf_matrix, feature_names):
    results = {}
    
    horizons = [1, 5, 10]
    
    for h in horizons:
        target_col = f'Alpha_T{h}'
        print(f"\n--- Analyzing Horizon T+{h} (Alpha) ---")
        
        y = df[target_col].values
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(tfidf_matrix, y, test_size=0.2, random_state=42)
        
        # 1. Ridge Regression (Linear, Directional)
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train, y_train)
        ridge_pred = ridge.predict(X_test)
        ridge_r2 = r2_score(y_test, ridge_pred)
        
        # 2. Random Forest (Non-linear, Importance)
        # Limit depth to prevent overfitting on noisy data
        rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=42)
        rf.fit(X_train, y_train)
        rf_pred = rf.predict(X_test)
        rf_r2 = r2_score(y_test, rf_pred)
        
        print(f"  Ridge R2: {ridge_r2:.4f}")
        print(f"  RandomForest R2: {rf_r2:.4f}")
        
        # Save Top Coefficients (Ridge is best for 'impact words')
        coefs = ridge.coef_
        impact_df = pd.DataFrame({
            'word': feature_names,
            'coefficient': coefs,
            'abs_coefficient': np.abs(coefs)
        }).sort_values('coefficient', ascending=False)
        
        impact_df.to_csv(OUTPUT_DIR / f"impact_words_T{h}.csv", index=False)
        results[f'T{h}'] = impact_df
        
    return results

def main():
    try:
        df = load_and_prep_data()
        
        print("\nVectorizing (TF-IDF)...")
        # Slightly stricter min_df to remove noise for robust analysis
        tfidf = TfidfVectorizer(max_features=3000, 
                                stop_words='english', 
                                min_df=0.02, 
                                max_df=0.90,
                                ngram_range=(1, 2))
        
        X = tfidf.fit_transform(df['text'])
        feature_names = tfidf.get_feature_names_out()
        
        # Combine with control variables? 
        # For now, let's keep it pure text for the 'word list' request to stay clean.
        # Adding log_len essentially just adds 1 feature, hard to interpret in Word List.
        # We will check correlation of length separately.
        
        corr_len = df[[f'Alpha_T{h}' for h in [1,5,10]]].corrwith(df['log_len'])
        print(f"\nCorrelation between Text Length and Alpha:\n{corr_len}")
        
        train_models(df, X, feature_names)
        
        print(f"\nAnalysis Complete. Results saved to {OUTPUT_DIR}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
