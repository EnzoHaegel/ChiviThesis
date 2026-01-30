import os
import glob


def main():
    SEARCH_DIR = "sec_10q"
    # Search for all original submission files
    TARGET_PATTERN = os.path.join(SEARCH_DIR, "**", "full-submission.txt")

    print(
        f"🔍 Searching for 'full-submission.txt' files that have a corresponding '_rf' version in '{SEARCH_DIR}'...")

    all_full_files = glob.glob(TARGET_PATTERN, recursive=True)

    if not all_full_files:
        print("✅ No 'full-submission.txt' files found.")
        return

    # Filter to only those that have a corresponding _rf file
    files_to_delete = []
    for f in all_full_files:
        rf_file = f.replace(".txt", "_rf.txt")
        if os.path.exists(rf_file):
            files_to_delete.append(f)

    if not files_to_delete:
        print("ℹ️ Found files, but none have a corresponding '_rf' version yet. Nothing to delete.")
        return

    print(
        f"⚠️ Found {len(files_to_delete)} files to delete (processed files).")

    deleted_count = 0
    total_freed_bytes = 0

    for filepath in files_to_delete:
        try:
            file_size = os.path.getsize(filepath)
            os.remove(filepath)
            deleted_count += 1
            total_freed_bytes += file_size
            if deleted_count % 100 == 0:
                print(
                    f"🗑️ Deleted {deleted_count}/{len(files_to_delete)} files...")
        except Exception as e:
            print(f"❌ Error deleting {filepath}: {e}")

    # Convert bytes to MB for readability
    freed_mb = total_freed_bytes / (1024 * 1024)

    print("\n--- Summary ---")
    print(f"✅ Successfully deleted: {deleted_count} processed original files")
    print(f"💾 Total space freed: {freed_mb:.2f} MB")
    print(f"ℹ️ {len(all_full_files) - deleted_count} original files were KEPT (no '_rf' version found).")
    print("Done!")


if __name__ == "__main__":
    main()
