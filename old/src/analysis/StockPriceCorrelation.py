import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
import os

# -----------------------------
# CONFIG
# -----------------------------
# Note: Run from Project Root (ChiviThesis/)
INPUT_CSV = Path("Data/sec_10q/downloaded_10q_reports.csv")
OUTPUT_CSV = Path("Data/sec_10q/announcements_with_prices.csv")
DAYS_RELATIVE = [-1, 1, 5, 10]

def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    # Load filing data
    print(f"Loading filings from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    # Pre-parse dates
    df['filingDate'] = pd.to_datetime(df['filingDate'])

    tickers = df['ticker'].unique()
    print(f"Found {len(tickers)} tickers. Fetching price data...")

    results = []

    # Process each ticker
    for ticker in tqdm(tickers):
        try:
            # Download a broad range of price data for this ticker
            ticker_data = yf.Ticker(ticker)
            hist = ticker_data.history(period="5y", interval="1d")

            if hist.empty:
                # Fallback for some tickers that might need a different handling or are delisted
                continue

            # Keep only 'Close'
            prices = hist['Close']
            trading_days = prices.index

            # Sub-dataframe for this ticker
            ticker_filings = df[df['ticker'] == ticker]

            for _, row in ticker_filings.iterrows():
                filing_date = row['filingDate']

                # Align timezones if necessary
                if prices.index.tz is not None:
                    if filing_date.tzinfo is None:
                        # yfinance usually returns data in the exchange's timezone
                        # We just need to make filing_date aware of the same TZ to compare
                        filing_date = filing_date.replace(tzinfo=prices.index.tz)

                # Find the nearest trading day for T=0 (>= filing_date)
                if filing_date not in trading_days:
                    future_days = trading_days[trading_days >= filing_date]
                    if len(future_days) == 0:
                        continue
                    actual_t0 = future_days[0]
                else:
                    actual_t0 = filing_date

                # Find position of actual_t0 in prices series
                pos = prices.index.get_loc(actual_t0)

                record = row.to_dict()
                record['actual_trading_date_T0'] = actual_t0.strftime('%Y-%m-%d')
                record['price_T0'] = prices.iloc[pos]

                for days in DAYS_RELATIVE:
                    target_pos = pos + days
                    col_prefix = f'T{"+" if days > 0 else ""}{days}'

                    if 0 <= target_pos < len(prices):
                        record[f'price_{col_prefix}'] = prices.iloc[target_pos]
                        record[f'date_{col_prefix}'] = prices.index[target_pos].strftime('%Y-%m-%d')
                    else:
                        record[f'price_{col_prefix}'] = None
                        record[f'date_{col_prefix}'] = None

                results.append(record)

        except Exception as e:
            print(f"Error processing {ticker}: {e}")

    # Create final dataframe
    if results:
        final_df = pd.DataFrame(results)
        # Ensure output directory exists
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nDone! saved to {OUTPUT_CSV}")
    else:
        print("\nNo results to save.")

if __name__ == "__main__":
    main()
