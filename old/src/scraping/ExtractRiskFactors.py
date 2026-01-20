import csv
import re
import bs4
from pathlib import Path
from tqdm import tqdm
import unicodedata

# Config
INPUT_CSV = Path("sec_10q/announcements_with_prices.csv")
OUTPUT_DIR = Path("sec_10q/risk_factors")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    """Normalize whitespace and clean text."""
    # Normalize unicode (e.g. non-breaking spaces)
    text = unicodedata.normalize("NFKC", text)
    # Replace newlines and multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_content_between_anchors(soup, start_id, end_id=None):
    """Extracts text between two HTML elements identified by IDs."""
    start_elem = soup.find(id=start_id)
    if not start_elem:
        return None
    
    content = []
    curr = start_elem.find_next() # Start after the anchor element
    
    # Safety counter to prevent infinite loops in malformed HTML
    max_elements = 5000 
    count = 0
    
    while curr and count < max_elements:
        # Check if we hit the end anchor
        if end_id and curr.get('id') == end_id:
            break
            
        # Also check if we hit a substantial new header if end_id is vague
        # (This is harder to do safely without false positives, so relying on end_id is better)

        if curr.name: # Skip NavigableStrings at top level if desired, but text is what we want
             text = curr.get_text(strip=True)
             if text:
                 content.append(text)
        elif isinstance(curr, bs4.NavigableString):
             text = str(curr).strip()
             if text:
                 content.append(text)
                 
        curr = curr.find_next()
        count += 1
        
    return " ".join(content)

def extract_via_toc(soup):
    """Attempts to extract Risk Factors using the Table of Contents."""
    # 1. Find the Link
    link = soup.find('a', string=lambda t: t and "Risk Factors" in t)
    if not link:
        # Case insensitive regex match
        link = soup.find('a', string=re.compile(r"Item\s*1A\.?\s*Risk\s*Factors", re.IGNORECASE))
    
    if not link:
        return None, "No TOC link found"

    href = link.get('href')
    if not href or not href.startswith('#'):
        return None, "TOC link has no internal href"
    
    start_id = href[1:]
    
    # 2. Find the End Link (Next Item)
    # We look for the "Item 1A" link in the DOM tree and try to find the *next* link in the TOC
    # This assumes the TOC is structured as a list or sequence of links.
    
    toc_links = soup.find_all('a', href=True)
    end_id = None
    
    # Iterate to find our link, then pick the next valid internal link
    for i, l in enumerate(toc_links):
        if l == link:
            # Look ahead for the next link that points to an internal ID
            for j in range(i + 1, len(toc_links)):
                next_l = toc_links[j]
                next_href = next_l.get('href')
                
                if next_href and next_href.startswith('#'):
                     candidate_id = next_href[1:]
                     # CRITICAL FIX: Ensure we don't pick the same ID (e.g. page number links often point to same anchor)
                     if candidate_id == start_id:
                         continue
                         
                     # Heuristic: The next item usually is 'Item 2' or 'Unregistered Sales'
                     # We accept it if it's a different ID. 
                     # Refinement: We could check if next_text looks like "Item X", but specific text varies.
                     # Distinct ID is the most important baseline.
                     
                     end_id = candidate_id
                     break
            break
            
    # If we couldn't find a distinct end ID from TOC, extraction might be unbounded (risky),
    # but we can try extracting until a heuristic limit or just return found start.
    
    extracted_text = extract_content_between_anchors(soup, start_id, end_id)
    return extracted_text, f"TOC extraction (End anchor: {end_id})"


def extract_via_regex(soup):
    """Fallback: Search for the header directly in text."""
    # Find all text nodes matching the header pattern
    # We want a node that is LIKELY a header: short, standalone.
    
    pattern = re.compile(r"Item\s*1A\.?\s*Risk\s*Factors", re.IGNORECASE)
    matches = soup.find_all(string=pattern)
    
    candidate = None
    for m in matches:
        parent = m.parent
        # unwanted: TOC links
        if parent.name == 'a' or parent.find_parent('a'):
            continue
        
        # Check text length to ensure it's a header, not a sentence reference
        t = clean_text(parent.get_text())
        if len(t) < 100: 
            candidate = parent
            break
            
    if not candidate:
        return None, "Regex: Header not found"
        
    # Extract content after candidate
    # Stop at next major header "Item 2" or "Item 5" etc.
    content = []
    curr = candidate.find_next()
    count = 0
    stop_pattern = re.compile(r"^Item\s*[2-6]\.?", re.IGNORECASE)
    
    while curr and count < 3000: # Limit items
        text = curr.get_text(strip=True)
        # Check if we hit next item header
        if stop_pattern.match(text) and len(text) < 100:
            break
            
        if text:
            content.append(text)
        
        curr = curr.find_next()
        count += 1
        
    return " ".join(content), "Regex extraction"

def process_filing(row):
    accession = row['accessionNumber']
    ticker = row['ticker']
    
    # 1. Determine file path
    # The CSV has 'saved_path', strictly use that if correct.
    # Note: saved_path in CSV is relative to CWD usually, typically 'sec_10q/...'
    
    rel_path = row.get('saved_path')
    if not rel_path:
        return "Skipped (No path)"
        
    full_path = Path(rel_path)
    if not full_path.exists():
        return f"Skipped (File not found: {full_path})"
        
    # 2. Read HTML
    try:
        with open(full_path, 'r', encoding='utf-8') as f:
            soup = bs4.BeautifulSoup(f, 'html.parser')
    except Exception as e:
        return f"Error reading HTML: {e}"

    # 3. Strategy 1: TOC
    text, method = extract_via_toc(soup)
    
    # 4. Strategy 2: Regex Fallback
    if not text:
        text, method = extract_via_regex(soup)
    
    # 5. Save Result
    out_file = OUTPUT_DIR / f"{accession}.txt"
    
    status = "Extracted"
    if not text:
        # Save empty file or placeholder to indicate processed
        text = "SECTION_NOT_FOUND_OR_REFERENCE_ONLY"
        status = "Not Found"
        
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(clean_text(text))
        
    return f"{status} ({method})"

def main():
    if not INPUT_CSV.exists():
        print(f"Error: {INPUT_CSV} not found.")
        return

    print(f"Reading {INPUT_CSV}...")
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
    print(f"Processing {len(rows)} filings...")
    
    stats = {"Extracted": 0, "Not Found": 0, "Skipped": 0, "Error": 0}
    
    for row in tqdm(rows):
        res = process_filing(row)
        
        if "Extracted" in res: stats["Extracted"] += 1
        elif "Not Found" in res: stats["Not Found"] += 1
        elif "Skipped" in res: stats["Skipped"] += 1
        elif "Error" in res: stats["Error"] += 1
            
    print("\nProcessing Complete.")
    print(f"Summary: {stats}")

if __name__ == "__main__":
    main()
