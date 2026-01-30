import pandas as pd
import glob
import os
from sec_edgar_downloader import Downloader
import sys


def main():
    # --- Configuration ---
    # IMPORTANT: SEC requires a valid User-Agent in the format "Name email@domain.com"
    # Please update this with your actual details to avoid being blocked.
    USER_AGENT_NAME = "EnzoHaegel"
    USER_AGENT_EMAIL = "enzo.haegel@epitech.eu"
    DOWNLOAD_FOLDER = "sec_10q"
    CSV_PATH = "./csv_raw/*.csv"
    START_DATE = "2018-01-01"
    END_DATE = "2024-12-31"

    print(
        f"🚀 Starting SEC 10-Q Downloader for period {START_DATE} to {END_DATE}...")

    # 1. Load Tickers from CSVs
    print("📂 Loading CSV files to find unique tickers...")
    all_files = glob.glob(CSV_PATH)
    if not all_files:
        print("❌ No CSV files found in ./csv_raw/")
        return

    tickers = set()
    for filename in all_files:
        try:
            # We only need the 'tic' column
            df = pd.read_csv(filename, usecols=['tic'])
            # Drop clean tickers
            unique_ts = df['tic'].dropna().unique()
            tickers.update(unique_ts)
            print(
                f"   Found {len(unique_ts)} tickers in {os.path.basename(filename)}")
        except Exception as e:
            print(f"   ⚠️ Warning: Could not read {filename}: {e}")

    sorted_tickers = sorted(list(tickers))
    print(f"✅ Found {len(sorted_tickers)} unique tickers total.")

    if not sorted_tickers:
        print("❌ No tickers found. Exiting.")
        return

    # 2. Initialize Downloader
    # The library downloads to: <download_folder>/sec-edgar-filings/<Ticker>/...
    try:
        dl = Downloader(USER_AGENT_NAME, USER_AGENT_EMAIL, DOWNLOAD_FOLDER)
    except TypeError:
        # Handle potential API changes or older versions if inputs differ
        dl = Downloader(DOWNLOAD_FOLDER)
        print("⚠️ Warning: Initialized Downloader without User-Agent (Older version?). Ensure you are compliant.")

    # 3. Download Filings
    print(f"⬇️ Downloading 10-Q filings to '{DOWNLOAD_FOLDER}'...")

    for i, ticker in enumerate(sorted_tickers):
        ticker_str = str(ticker).strip().upper()
        if not ticker_str:
            continue

        print(f"[{i+1}/{len(sorted_tickers)}] Processing {ticker_str}...")

        # Check for existing filings to skip
        # Structure: <DOWNLOAD_FOLDER>/sec-edgar-filings/<TICKER>/10-Q/<ACCESSION_NUMBER>
        ticker_path = os.path.join(
            DOWNLOAD_FOLDER, "sec-edgar-filings", ticker_str, "10-Q")
        existing_accessions = []
        if os.path.exists(ticker_path):
            for d in os.listdir(ticker_path):
                accession_dir = os.path.join(ticker_path, d)
                if os.path.isdir(accession_dir):
                    # Check if either the original or the _rf version exists
                    if os.path.exists(os.path.join(accession_dir, "full-submission.txt")) or \
                       os.path.exists(os.path.join(accession_dir, "full-submission_rf.txt")):
                        existing_accessions.append(d)

        if existing_accessions:
            print(
                f"   ℹ️  Found {len(existing_accessions)} valid existing filings (original or _rf), checking for new ones...")

        try:
            # Download 10-Q filings in the date range, skipping what we already have
            count = dl.get("10-Q", ticker_str, after=START_DATE,
                           before=END_DATE, accession_numbers_to_skip=existing_accessions)

            if count == 0:
                if not existing_accessions:
                    print(f"   ⚠️ No filings found.")
                    with open("errors-download.txt", "a") as err_f:
                        err_f.write(f"{ticker_str} | No results found\n")
                else:
                    print(f"   -> No new filings (all existing skipped or no updates).")
            else:
                print(f"   -> Downloaded {count} new filings.")

        except TypeError:
            # Fallback for older versions of sec-edgar-downloader that might not support the skip arg
            print(
                "   ⚠️ Warning: 'accession_numbers_to_skip' not supported, falling back to standard download.")
            try:
                count = dl.get("10-Q", ticker_str,
                               after=START_DATE, before=END_DATE)
                print(f"   -> Downloaded {count} filings.")
            except Exception as e:
                error_msg = str(e).replace('\n', ' ')
                print(f"   ❌ Error downloading: {error_msg}")
                with open("errors-download.txt", "a") as err_f:
                    err_f.write(f"{ticker_str} | Error: {error_msg}\n")

        except Exception as e:
            error_msg = str(e).replace('\n', ' ')
            print(f"   ❌ Error downloading: {error_msg}")
            with open("errors-download.txt", "a") as err_f:
                err_f.write(f"{ticker_str} | Error: {error_msg}\n")

    print("\n✨ All downloads complete!")


if __name__ == "__main__":
    main()
