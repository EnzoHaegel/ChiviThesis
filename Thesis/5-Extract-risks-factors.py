import os
import glob
import re
from typing import Optional
import time

SEARCH_DIR = "sec_10q"


def openTxt(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read()
    return content


def getEveryFiles(path: str):
    files = glob.glob(f"{path}/**/full-submission.txt", recursive=True)
    print(f"🔍 Found {len(files)} files to process in '{path}'...")
    print(f" First files: {"\n".join(files[:5])}")
    return files


def extractRiskFactors(file_content: str, min_chars: int = 70) -> Optional[str]:
    if not file_content:
        return None

    sep = r"[\s\.\u00A0]*"
    punct = r"[\s\.\u00A0:\-–—]*"
    start_re = re.compile(
        rf"\bitem{sep}1{sep}a{punct}risk{sep}factors\b", re.IGNORECASE,)
    end_1b_re = re.compile(rf"\bitem{sep}1{sep}b\b", re.IGNORECASE,)
    end_2_re = re.compile(
        rf"\bitem{sep}2{punct}unregistered{sep}sales{sep}of{sep}equity{sep}securities{sep}and{sep}use{sep}of{sep}proceeds\b", re.IGNORECASE,)

    for m in start_re.finditer(file_content):
        start_idx = m.end()
        m1 = end_1b_re.search(file_content, start_idx)
        m2 = end_2_re.search(file_content, start_idx)
        candidates = [mm for mm in (m1, m2) if mm is not None]
        if not candidates:
            section = file_content[start_idx:].strip()
            if len(section) > min_chars:
                return section
            continue
        end_match = min(candidates, key=lambda x: x.start())
        section = file_content[start_idx:end_match.start()].strip()
        if len(section) > min_chars:
            return section
    return None


def extractDateFromContent(file_content: str) -> Optional[str]:
    if not file_content:
        return None

    first_line = file_content.splitlines()[0]
    for token in first_line.split():
        if len(token) == 8 and token.isdigit() and token.startswith("20"):
            return token

    return None


def rewriteFile(pathfile: str, content: str):
    directory = os.path.dirname(pathfile)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(pathfile, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    files = getEveryFiles(SEARCH_DIR)
    total_files = len(files)

    start_time = time.time()
    processed = 0

    for idx, file in enumerate(files, start=1):
        content = openTxt(file)
        riskFactors = extractRiskFactors(content)
        date = extractDateFromContent(content)

        if not riskFactors or not date:
            processed += 1
        else:
            new_content = date + "\n" + riskFactors
            base, ext = os.path.splitext(file)
            new_file = base + "_rf" + ext
            rewriteFile(new_file, new_content)
            processed += 1

        # Calculate progress for every file iteration
        elapsed = time.time() - start_time
        avg_time_per_file = elapsed / processed
        remaining_files = total_files - processed
        eta = remaining_files * avg_time_per_file
        percent = (processed / total_files) * 100

        print(
            f"[{processed}/{total_files}] "
            f"{percent:.2f}% | "
            f"Elapsed: {elapsed:.1f}s | "
            f"ETA: {eta:.1f}s"
        )

    total_time = time.time() - start_time
    print(f"\nTerminé. {processed} fichiers traités en {total_time:.1f}s.")


if __name__ == "__main__":
    main()
