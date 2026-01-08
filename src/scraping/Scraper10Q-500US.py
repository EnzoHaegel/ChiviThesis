#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download 10-K annual reports (last N years) for a list of US companies (tickers),
saving into clean, organized files.

Data source: SEC EDGAR (official)
- Ticker -> CIK map: https://www.sec.gov/files/company_tickers.json
- Company submissions: https://data.sec.gov/submissions/CIK##########.json
- Filing documents: https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{primaryDocument}

IMPORTANT:
- SEC requires a real User-Agent identifying you (name + contact).
- Be polite: rate limit your requests.
"""

from __future__ import annotations

import re
import json
import time
import csv
import sys
import io
from pathlib import Path
from datetime import datetime, date, timezone
from typing import Dict, List, Optional, Tuple

import requests

try:
    import pandas as pd  # only needed for the fallback S&P500 ticker fetch
except Exception:
    pd = None


# -------------------------
# CONFIG
# -------------------------

YEARS_BACK = 4
OUT_DIR = Path("sec_10q")
TICKERS_CSV = Path("tickers.csv")  # optional, one ticker per line or a column named 'ticker'
USE_SP500_FALLBACK_IF_NO_FILE = True

# Put your real identity here (SEC policy)
SEC_HEADERS = {
    "User-Agent": "Enzo HAEGEL (academic research) contact: your_email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}

SEC_HEADERS_DATA = {
    "User-Agent": SEC_HEADERS["User-Agent"],
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

REQUEST_TIMEOUT = 60
SLEEP_BETWEEN_REQUESTS_SEC = 0.25  # be nice to SEC; increase if you get 429
MAX_RETRIES = 4


# -------------------------
# HELPERS
# -------------------------

def has_public_bonds(submissions: dict) -> bool:
    """
    Check if a company has public bonds by looking for 424B prospectus filings,
    which are typically used for debt offerings.
    """
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    # 424B2, 424B3, 424B5 are common for bond offerings
    bond_forms = {f for f in forms if f.startswith("424B")}
    return len(bond_forms) > 0

def sanitize_filename(s: str, max_len: int = 160) -> str:
    s = s.strip()
    s = re.sub(r"[\/\\\:\*\?\"\<\>\|]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_len:
        s = s[:max_len].rstrip()
    return s

def http_get(url: str, headers: dict) -> requests.Response:
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            if r.status_code == 429:
                # Too many requests -> exponential backoff
                sleep_for = min(8.0, 0.5 * (2 ** (attempt - 1)))
                time.sleep(sleep_for)
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(min(8.0, 0.5 * (2 ** (attempt - 1))))
    raise RuntimeError(f"GET failed after retries: {url}\nLast error: {last_err}")

def load_tickers_from_csv(path: Path) -> List[str]:
    tickers: List[str] = []
    if not path.exists():
        return tickers

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(2048)
        f.seek(0)

        # If it's a simple one-ticker-per-line file (no commas)
        if "," not in sample and ";" not in sample and "\t" not in sample:
            for line in f:
                t = line.strip().upper()
                if t and not t.startswith("#"):
                    tickers.append(t)
            return sorted(set(tickers))

        # Otherwise try CSV with header
        reader = csv.DictReader(f)
        cols = [c.lower() for c in (reader.fieldnames or [])]
        # pick a likely column
        col = None
        for candidate in ["ticker", "symbol", "tic"]:
            if candidate in cols:
                col = reader.fieldnames[cols.index(candidate)]
                break
        if col is None:
            raise ValueError(
                f"CSV found but no ticker column. Columns={reader.fieldnames}. "
                "Use a column named 'ticker' (or symbol/tic), or use one ticker per line."
            )
        for row in reader:
            t = str(row.get(col, "")).strip().upper()
            if t and t != "NAN":
                tickers.append(t)
    return sorted(set(tickers))

def load_sp500_tickers() -> List[str]:
    if pd is None:
        raise RuntimeError("pandas is required for SP500 fallback. Install: pip install pandas lxml html5lib")
    # Wikipedia S&P 500 table
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia blocks requests without a browser-like User-Agent
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
    try:
        r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(io.StringIO(r.text))
    except Exception as e:
        raise RuntimeError(f"Failed to fetch S&P 500 list from Wikipedia: {e}")
    # The first table is typically the constituents table
    df = tables[0]
    col = None
    for c in df.columns:
        if str(c).lower() in ["symbol", "ticker symbol", "ticker"]:
            col = c
            break
    if col is None:
        # common name is 'Symbol'
        col = "Symbol" if "Symbol" in df.columns else df.columns[0]
    tickers = [str(x).strip().upper().replace(".", "-") for x in df[col].tolist()]
    return sorted(set([t for t in tickers if t and t != "NAN"]))

def fetch_ticker_to_cik_map() -> Dict[str, str]:
    url = "https://www.sec.gov/files/company_tickers.json"
    r = http_get(url, headers=SEC_HEADERS)
    data = r.json()

    mapping: Dict[str, str] = {}
    for _, v in data.items():
        ticker = str(v.get("ticker", "")).strip().upper()
        cik = v.get("cik_str", None)
        if not ticker or cik is None:
            continue
        cik10 = str(int(cik)).zfill(10)
        mapping[ticker] = cik10
    return mapping

def fetch_company_submissions(cik10: str) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    r = http_get(url, headers=SEC_HEADERS_DATA)
    return r.json()

def extract_recent_10q_filings(submissions: dict, years_back: int) -> List[dict]:
    cutoff = date.today().replace(year=date.today().year - years_back)
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    out: List[dict] = []
    n = min(len(forms), len(filing_dates), len(accession_numbers), len(primary_docs))
    for i in range(n):
        if forms[i] != "10-Q":
            continue
        try:
            fd = datetime.strptime(filing_dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        if fd < cutoff:
            continue
        out.append(
            {
                "form": forms[i],
                "filingDate": filing_dates[i],
                "accessionNumber": accession_numbers[i],
                "primaryDocument": primary_docs[i],
            }
        )


    # Keep newest first
    out.sort(key=lambda x: x["filingDate"], reverse=True)
    return out

def build_filing_url(cik10: str, accession_number: str, primary_document: str) -> str:
    cik_int = str(int(cik10))  # SEC path uses non-zero-padded integer
    acc_nodash = accession_number.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_nodash}/{primary_document}"

def save_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", errors="replace")

def save_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)

def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tickers = load_tickers_from_csv(TICKERS_CSV)
    if not tickers and USE_SP500_FALLBACK_IF_NO_FILE:
        print("No tickers.csv found or empty -> using S&P 500 constituents fallback.")
        tickers = load_sp500_tickers()

    if not tickers:
        print("No tickers provided. Create tickers.csv with 500 tickers (one per line) and rerun.")
        return 2

    print(f"Universe size: {len(tickers)} tickers")

    print("Fetching SEC ticker->CIK map...")
    t2cik = fetch_ticker_to_cik_map()

    report_rows = []
    errors = []

    for idx, ticker in enumerate(tickers, start=1):
        ticker_clean = ticker.strip().upper()
        cik10 = t2cik.get(ticker_clean)

        if cik10 is None:
            errors.append({"ticker": ticker_clean, "error": "CIK not found in SEC company_tickers.json"})
            continue

        try:
            time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

            submissions = fetch_company_submissions(cik10)

            # --- BOND FILTERING ---
            if not has_public_bonds(submissions):
                print(f"[{idx}/{len(tickers)}] {ticker_clean}: SKIP (No public bonds detected)")
                continue

            company_name = sanitize_filename(str(submissions.get("name", ticker_clean)))
            filings = extract_recent_10q_filings(submissions, YEARS_BACK)

            base_dir = OUT_DIR / sanitize_filename(ticker_clean) / company_name
            base_dir.mkdir(parents=True, exist_ok=True)

            # Save a small metadata json per company
            meta_path = base_dir / "company_submissions_meta.json"
            save_text(meta_path, json.dumps({"ticker": ticker_clean, "cik10": cik10, "name": submissions.get("name", ""),
                                            "fetched_at": datetime.now(timezone.utc).isoformat() + "Z"}, indent=2))

            if not filings:
                errors.append({"ticker": ticker_clean, "cik10": cik10, "error": "No 10-Q found in last years"})
                continue

            for f in filings:
                filing_date = f["filingDate"]
                acc = f["accessionNumber"]
                doc = f["primaryDocument"]
                url = build_filing_url(cik10, acc, doc)

                # Guess extension; keep it, but normalize name
                ext = Path(doc).suffix.lower() or ".html"
                year = filing_date[:4]
                fname = sanitize_filename(f"{year}_10-Q_{filing_date}_{acc}{ext}")
                out_path = base_dir / fname

                # Download
                time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
                r = http_get(url, headers=SEC_HEADERS)

                # Save raw bytes (HTML/XML/TXT/PDF depending on doc)
                save_bytes(out_path, r.content)

                report_rows.append({
                    "ticker": ticker_clean,
                    "cik10": cik10,
                    "company_name": submissions.get("name", ""),
                    "filingDate": filing_date,
                    "accessionNumber": acc,
                    "primaryDocument": doc,
                    "download_url": url,
                    "saved_path": str(out_path),
                })

            print(f"[{idx}/{len(tickers)}] {ticker_clean}: OK ({len(filings)} filings)")

        except Exception as e:
            errors.append({"ticker": ticker_clean, "cik10": cik10, "error": repr(e)})
            print(f"[{idx}/{len(tickers)}] {ticker_clean}: ERROR -> {e}", file=sys.stderr)

    # Save summary CSVs
    reports_csv = OUT_DIR / "downloaded_10q_reports.csv"
    errors_csv = OUT_DIR / "errors.csv"

    def write_csv(path: Path, rows: List[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            save_text(path, "")
            return
        cols = sorted({k for r in rows for k in r.keys()})
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)

    write_csv(reports_csv, report_rows)
    write_csv(errors_csv, errors)

    print("\nDone.")
    print(f"- Saved reports index: {reports_csv}")
    print(f"- Errors: {errors_csv}")
    print(f"- Output folder: {OUT_DIR.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
