import os
import gc
import glob
import textwrap
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


MAX_LINE_LENGTH = 150


def wrap_text_smart(text):
    """
    Wraps text to MAX_LINE_LENGTH, breaking only after spaces or periods.
    """
    lines = text.split('\n')
    wrapped_lines = []

    for line in lines:
        if len(line) <= MAX_LINE_LENGTH:
            wrapped_lines.append(line)
        else:
            wrapped = textwrap.fill(
                line,
                width=MAX_LINE_LENGTH,
                break_long_words=False,
                break_on_hyphens=False
            )
            wrapped_lines.append(wrapped)

    return '\n'.join(wrapped_lines)


def process_file(filepath):
    """
    Reads a file, wraps long lines, and overwrites it.
    """
    try:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()

        original_size = len(content)
        wrapped_content = wrap_text_smart(content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(wrapped_content)

        new_size = len(wrapped_content)
        size_change = ((new_size - original_size) /
                       original_size) * 100 if original_size > 0 else 0

        del content
        del wrapped_content
        gc.collect()

        return ('success', filepath, size_change)

    except Exception as e:
        return ('error', filepath, str(e))


def main():
    SEARCH_DIR = "sec_10q"
    BATCH_SIZE = 100
    MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

    all_files = glob.glob(f"{SEARCH_DIR}/**/*.txt", recursive=True)

    print("=" * 60)
    print("📄 LINE WRAPPER - Making files readable")
    print("=" * 60)
    print(f"\n🔍 Found {len(all_files)} .txt files in '{SEARCH_DIR}'")
    print(f"📏 Max line length: {MAX_LINE_LENGTH} characters")
    print(f"🖥️  Using {MAX_WORKERS} CPU workers")
    print(f"📦 Processing in batches of {BATCH_SIZE}")

    if not all_files:
        print("✅ No files found.")
        return

    print("\n🚀 Starting line wrapping...\n")

    total_success = 0
    total_errors = 0

    for batch_start in range(0, len(all_files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(all_files))
        batch_files = all_files[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(all_files) + BATCH_SIZE - 1) // BATCH_SIZE

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(
                process_file, f): f for f in batch_files}

            with tqdm(total=len(batch_files), desc=f"Batch {batch_num}/{total_batches}", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    status, filepath, data = result

                    if status == 'success':
                        total_success += 1
                    else:
                        total_errors += 1
                        tqdm.write(f"❌ Error: {filepath}: {data}")

                    pbar.update(1)

        gc.collect()

    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"✅ Successfully wrapped: {total_success}")
    print(f"❌ Errors: {total_errors}")
    print("Done!")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
