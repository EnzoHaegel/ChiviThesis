import os
import re
import gc
import glob
import psutil
import multiprocessing
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


def get_memory_usage_mb():
    """Returns current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def pre_clean_html(content):
    """
    Fast regex-based pre-cleaning to remove heavy tags before BeautifulSoup parsing.
    This significantly reduces memory usage and parsing time.
    """
    content = re.sub(r'<script[^>]*>[\s\S]*?</script>',
                     '', content, flags=re.IGNORECASE)
    content = re.sub(r'<style[^>]*>[\s\S]*?</style>',
                     '', content, flags=re.IGNORECASE)
    content = re.sub(r'<svg[^>]*>[\s\S]*?</svg>', '',
                     content, flags=re.IGNORECASE)
    content = re.sub(
        r'<noscript[^>]*>[\s\S]*?</noscript>', '', content, flags=re.IGNORECASE)
    content = re.sub(r'<!--[\s\S]*?-->', '', content)
    return content


def process_file(filepath):
    """
    Reads a file, strips HTML tags (including script/style), and overwrites it with pure text.
    Optimized for large files with pre-cleaning and explicit memory management.
    """
    try:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()

        original_size = len(content)

        content = pre_clean_html(content)

        soup = BeautifulSoup(content, 'lxml')
        del content

        for tag in soup(['head', 'meta', 'link', 'path']):
            tag.decompose()

        text_content = soup.get_text(separator=' ', strip=True)
        soup.decompose()
        del soup

        text_content = re.sub(r'\s{3,}', '  ', text_content)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text_content)

        new_size = len(text_content)
        reduction = (1 - new_size / original_size) * \
            100 if original_size > 0 else 0

        del text_content
        gc.collect()

        return ('success', filepath, reduction)

    except Exception as e:
        return ('error', filepath, str(e))


def main():
    SEARCH_DIR = "sec_10q"
    PROCESSED_LOG = "processed_files.txt"
    BATCH_SIZE = 50
    MAX_WORKERS = max(1, multiprocessing.cpu_count() - 1)

    processed_files = set()
    if os.path.exists(PROCESSED_LOG):
        with open(PROCESSED_LOG, 'r', encoding='utf-8') as f:
            processed_files = set(line.strip() for line in f if line.strip())

    all_files = glob.glob(
        f"{SEARCH_DIR}/**/full-submission.txt", recursive=True)
    files = [f for f in all_files if f not in processed_files]

    print(f"🔍 Found {len(all_files)} total files in '{SEARCH_DIR}'.")
    print(f"⏭️  {len(all_files) - len(files)} files already processed. {len(files)} new files to process.")
    print(f"🖥️  Using {MAX_WORKERS} CPU workers (ProcessPoolExecutor)")
    print(
        f"📦 Processing in batches of {BATCH_SIZE} files with GC between batches")
    print(f"💾 Current memory usage: {get_memory_usage_mb():.1f} MB")

    if not files:
        print("✅ All files are already processed or no new files found.")
        return

    print("\n🚀 Starting optimized HTML stripping...\n")

    total_success = 0
    total_errors = 0
    batch_log_buffer = []

    for batch_start in range(0, len(files), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(files))
        batch_files = files[batch_start:batch_end]
        batch_num = (batch_start // BATCH_SIZE) + 1
        total_batches = (len(files) + BATCH_SIZE - 1) // BATCH_SIZE

        print(
            f"\n📦 Batch {batch_num}/{total_batches} ({len(batch_files)} files)")
        print(f"💾 Memory before batch: {get_memory_usage_mb():.1f} MB")

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(
                process_file, f): f for f in batch_files}

            with tqdm(total=len(batch_files), desc=f"Batch {batch_num}", unit="file") as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    status, filepath, data = result

                    if status == 'success':
                        total_success += 1
                        batch_log_buffer.append(filepath)
                        pbar.set_postfix_str(
                            f"✓ {os.path.basename(os.path.dirname(filepath))} (-{data:.0f}%)")
                    else:
                        total_errors += 1
                        tqdm.write(f"❌ Error: {filepath}: {data}")

                    pbar.update(1)

        if batch_log_buffer:
            with open(PROCESSED_LOG, 'a', encoding='utf-8') as log:
                log.write('\n'.join(batch_log_buffer) + '\n')
            batch_log_buffer.clear()

        gc.collect()
        print(f"💾 Memory after GC: {get_memory_usage_mb():.1f} MB")

    print("\n" + "=" * 50)
    print("📊 FINAL SUMMARY")
    print("=" * 50)
    print(f"✅ Successfully cleaned: {total_success}")
    print(f"❌ Errors: {total_errors}")
    print(f"💾 Final memory usage: {get_memory_usage_mb():.1f} MB")
    print("Done! Files have been overwritten with pure text content.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
