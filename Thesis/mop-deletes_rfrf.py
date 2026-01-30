import os
import glob


def main():
    SEARCH_DIR = "sec_10q"
    # TARGET_PATTERN matches anything ending in _rf_rf.txt
    TARGET_PATTERN = os.path.join(SEARCH_DIR, "**", "*_rf_rf.txt")

    print(f"🔍 Searching for all '*_rf_rf.txt' files in '{SEARCH_DIR}'...")

    files_to_delete = glob.glob(TARGET_PATTERN, recursive=True)

    if not files_to_delete:
        print("✅ No '*_rf_rf.txt' files found.")
        return

    print(f"⚠️ Found {len(files_to_delete)} duplicate files to delete.")

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

    freed_mb = total_freed_bytes / (1024 * 1024)

    print("\n--- Summary ---")
    print(f"✅ Successfully deleted: {deleted_count} files")
    print(f"💾 Total space freed: {freed_mb:.2f} MB")
    print("Done!")


if __name__ == "__main__":
    main()
