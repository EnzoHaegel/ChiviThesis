import bs4
import re
from pathlib import Path

path = Path(r"c:\Users\Enzo\Projects\ChiviThesis\Data\sec_10q\ABBV\AbbVie Inc\2022_10-Q_2022-05-06_0001551152-22-000017.htm")

def investigate_abbv(path):
    print(f"Reading {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        soup = bs4.BeautifulSoup(f, 'html.parser')
        
    print("\n--- Searching for 'Risk Factors' anywhere ---")
    matches = soup.find_all(string=re.compile(r"risk\s+factors", re.IGNORECASE))
    for i, m in enumerate(matches[:10]):
        print(f"Match {i}: {repr(m)}")
        print(f"  Parent: {m.parent.name} {m.parent.attrs}")

    print("\n--- Searching for 'Item 1A' anywhere ---")
    matches = soup.find_all(string=re.compile(r"Item\s+1A", re.IGNORECASE))
    for i, m in enumerate(matches[:10]):
        print(f"Match {i}: {repr(m)}")
        print(f"  Parent: {m.parent.name} {m.parent.attrs}")

if __name__ == "__main__":
    investigate_abbv(path)
