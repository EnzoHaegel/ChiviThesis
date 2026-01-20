import bs4
import re
from pathlib import Path

path = Path(r"c:\Users\Enzo\Projects\ChiviThesis\Data\sec_10q\AAPL\Apple Inc\2022_10-Q_2022-01-28_0000320193-22-000007.htm")

def extract_via_toc_debug(soup):
    print("--- Debugging TOC Extraction ---")
    link = soup.find('a', string=lambda t: t and "Risk Factors" in t)
    if not link:
         link = soup.find('a', string=re.compile(r"Item\s*1A\.?\s*Risk\s*Factors", re.IGNORECASE))
    
    if not link:
        print("No TOC start link found")
        return

    print(f"Found Start Link: {link}")
    href = link.get('href')
    print(f"Start HREF: {href}")
    
    start_id = href[1:] if href else None
    
    toc_links = soup.find_all('a', href=True)
    end_id = None
    
    for i, l in enumerate(toc_links):
        if l == link:
            print(f"Located link at index {i}")
            # Look ahead
            for j in range(i + 1, len(toc_links)):
                next_l = toc_links[j]
                next_href = next_l.get('href')
                next_text = next_l.get_text(strip=True)
                print(f"  Checking candidate {j}: '{next_text}' -> {next_href}")
                
                if next_href and next_href.startswith('#'):
                     if len(next_text) < 100:
                         end_id = next_href[1:]
                         print(f"  Selected End ID: {end_id}")
                         break
            break
            
    if start_id and end_id:
        print(f"Start ID: {start_id}")
        start_elem = soup.find(id=start_id)
        if start_elem:
            print(f"Start Element found: <{start_elem.name}>")
        else:
            print("Start Element NOT found in DOM")

        print(f"End ID: {end_id}")
        end_elem = soup.find(id=end_id)
        if end_elem:
            print(f"End Element found: <{end_elem.name}>")
        else:
            print("End Element NOT found in DOM")

def main():
    with open(path, 'r', encoding='utf-8') as f:
        soup = bs4.BeautifulSoup(f, 'html.parser')
    extract_via_toc_debug(soup)

if __name__ == "__main__":
    main()
