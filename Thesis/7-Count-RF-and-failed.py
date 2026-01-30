import os
import glob
from collections import defaultdict


def main():
    """
    Counts directories with _rf.txt (converted successfully) vs 
    directories with only full-submission.txt (still need conversion).
    Goal: Every subdirectory should have a _rf.txt file.
    """
    SEARCH_DIR = "sec_10q"

    rf_files = glob.glob(
        f"{SEARCH_DIR}/**/full-submission_rf.txt", recursive=True)
    original_files = glob.glob(
        f"{SEARCH_DIR}/**/full-submission.txt", recursive=True)

    rf_dirs = set(os.path.dirname(f) for f in rf_files)
    original_dirs = set(os.path.dirname(f) for f in original_files)

    converted_complete = rf_dirs - original_dirs
    pending_deletion = rf_dirs & original_dirs
    needs_conversion = original_dirs - rf_dirs

    all_10q_dirs = set()
    for root, dirs, files in os.walk(SEARCH_DIR):
        if root.count(os.sep) >= 4:
            all_10q_dirs.add(root)

    stats_by_ticker = defaultdict(
        lambda: {'complete': 0, 'pending': 0, 'failed': 0})

    for d in converted_complete:
        parts = d.replace('\\', '/').split('/')
        ticker = parts[2] if len(parts) > 2 else 'unknown'
        stats_by_ticker[ticker]['complete'] += 1

    for d in pending_deletion:
        parts = d.replace('\\', '/').split('/')
        ticker = parts[2] if len(parts) > 2 else 'unknown'
        stats_by_ticker[ticker]['pending'] += 1

    for d in needs_conversion:
        parts = d.replace('\\', '/').split('/')
        ticker = parts[2] if len(parts) > 2 else 'unknown'
        stats_by_ticker[ticker]['failed'] += 1

    print("=" * 60)
    print("📊 RF CONVERSION STATUS")
    print("=" * 60)
    print(
        f"\n✅ Complete (_rf.txt only, original deleted): {len(converted_complete)}")
    print(
        f"🔄 Pending deletion (_rf.txt + original exist): {len(pending_deletion)}")
    print(
        f"❌ Needs conversion (only original, no _rf.txt): {len(needs_conversion)}")
    print(f"\n📁 Total directories with _rf.txt: {len(rf_dirs)}")
    print(f"📁 Total directories with original: {len(original_dirs)}")

    total_work = len(converted_complete) + \
        len(pending_deletion) + len(needs_conversion)
    progress = (len(converted_complete) + len(pending_deletion)) / \
        total_work * 100 if total_work else 100
    print(f"\n📈 Conversion progress: {progress:.2f}%")
    complete_progress = len(converted_complete) / \
        total_work * 100 if total_work else 100
    print(f"📈 Fully complete: {complete_progress:.2f}%")

    if needs_conversion:
        print(f"\n--- Top 10 tickers needing conversion ---")
        failed_by_ticker = [(t, s['failed'])
                            for t, s in stats_by_ticker.items() if s['failed'] > 0]
        failed_by_ticker.sort(key=lambda x: x[1], reverse=True)
        for ticker, count in failed_by_ticker[:10]:
            print(f"  {ticker}: {count} need conversion")

        print(f"\n--- Sample directories needing conversion ---")
        for d in list(needs_conversion)[:5]:
            print(f"  {d}")

    with open("rf_conversion_report.txt", 'w', encoding='utf-8') as f:
        f.write("RF CONVERSION STATUS REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Complete (_rf only): {len(converted_complete)}\n")
        f.write(f"Pending deletion: {len(pending_deletion)}\n")
        f.write(f"Needs conversion: {len(needs_conversion)}\n")
        f.write(f"Conversion progress: {progress:.2f}%\n\n")

        if needs_conversion:
            f.write("DIRECTORIES NEEDING CONVERSION:\n")
            for d in sorted(needs_conversion):
                f.write(f"  {d}\n")

        if pending_deletion:
            f.write("\nDIRECTORIES PENDING ORIGINAL DELETION:\n")
            for d in sorted(pending_deletion):
                f.write(f"  {d}\n")

    print(f"\n📄 Full report saved to: rf_conversion_report.txt")


if __name__ == "__main__":
    main()
