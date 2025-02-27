#!/usr/bin/env python3
import os
import re
import csv
import json
import argparse
import requests
import subprocess
import threading
import concurrent.futures
import copy
import signal
import time
from bs4 import BeautifulSoup

# ----------------------------
# Debug Function (toggle via --debug)
# ----------------------------
DEBUG = False
def debug(msg):
    if DEBUG:
        print("[DEBUG]", msg)

# ----------------------------
# Region Normalization
# ----------------------------
def normalize_region(region):
    if not region:
        return None
    reg = region.strip().lower()
    if reg in ["general american", "american", "american english"]:
        return "US"
    return region.strip()

# ----------------------------
# Utility Functions
# ----------------------------
def normalize_word(word):
    parts = word.strip().split()
    return "_".join(parts)

def get_wiktionary_url(normalized_word):
    return f"https://en.wiktionary.org/wiki/{normalized_word}"

def is_fully_processed(rows):
    for row in rows:
        path = row["file_path"].strip()
        if path and not os.path.exists(path):
            return False
    return True

def load_existing_csv(csv_path):
    processed = {}
    if not os.path.exists(csv_path):
        return processed
    with open(csv_path, mode="r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            norm = normalize_word(row["word"])
            processed.setdefault(norm, []).append(row)
    return processed

def load_skipped_terms(skipped_path):
    skipped = set()
    if os.path.exists(skipped_path):
        with open(skipped_path, mode="r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if term:
                    skipped.add(term)
    return skipped

def save_skipped_terms(skipped_path, skipped_set, order=None):
    temp_path = skipped_path + ".tmp"
    with open(temp_path, mode="w", encoding="utf-8") as f:
        if order:
            for norm in order:
                if norm in skipped_set:
                    f.write(norm + "\n")
            for norm in sorted(skipped_set - set(order)):
                f.write(norm + "\n")
        else:
            for term in sorted(skipped_set):
                f.write(term + "\n")
    os.replace(temp_path, skipped_path)
    print(f"Saved {len(skipped_set)} skipped terms to {skipped_path}")

def save_all_results(csv_path, all_results, fieldnames, order=None):
    temp_path = csv_path + ".tmp"
    with open(temp_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if order:
            for norm in order:
                if norm in all_results:
                    for row in all_results[norm]:
                        writer.writerow(row)
            for norm in sorted(set(all_results.keys()) - set(order)):
                for row in all_results[norm]:
                    writer.writerow(row)
        else:
            for rows in all_results.values():
                for row in rows:
                    writer.writerow(row)
    os.replace(temp_path, csv_path)
    total = sum(len(v) for v in all_results.values())
    print(f"Saved CSV with {total} rows to {csv_path}")

def merge_results(existing, new):
    for new_row in new:
        duplicate = False
        for exist_row in existing:
            if (exist_row["region"] == new_row["region"] and 
                exist_row["file_path"] == new_row["file_path"] and 
                exist_row["IPAs"] == new_row["IPAs"]):
                duplicate = True
                break
        if not duplicate:
            existing.append(new_row)
    return existing

def unique_preserve_order(seq):
    seen = set()
    result = []
    for item in seq:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# ----------------------------
# Snapshot Save Helpers
# ----------------------------
def snapshot_save(csv_file, skipped_file, all_results, skipped_terms, fieldnames, order, lock):
    with lock:
        snapshot_results = copy.deepcopy(all_results)
        snapshot_skipped = copy.deepcopy(skipped_terms)
    save_all_results(csv_file, snapshot_results, fieldnames, order)
    save_skipped_terms(skipped_file, snapshot_skipped, order)

def periodic_snapshot(csv_file, skipped_file, all_results, skipped_terms, fieldnames, order, lock, interval):
    while not shutdown_event.is_set():
        time.sleep(interval)
        print("Periodic snapshot saving...")
        snapshot_save(csv_file, skipped_file, all_results, skipped_terms, fieldnames, order, lock)

# ----------------------------
# Audio Conversion Function
# ----------------------------
def convert_audio_to_wav(input_path):
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    output_path = os.path.splitext(input_path)[0] + ".wav"
    debug(f"Converting {input_path} to WAV at {output_path}")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(input_path):
            os.remove(input_path)
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_path} to WAV: {e}")
        return input_path

# ----------------------------
# Scraping Functions
# ----------------------------
def get_pronunciation_section_html(word):
    api_url = "https://en.wiktionary.org/w/api.php"
    params = {"action": "parse", "page": word, "prop": "sections", "format": "json"}
    resp = requests.get(api_url, params=params)
    data = resp.json()
    sections = data.get("parse", {}).get("sections", [])
    pron_index = None
    for sec in sections:
        if sec["line"].lower().startswith("pronunciation"):
            pron_index = sec["index"]
            break
    if not pron_index:
        debug(f"Pronunciation section not found for {word}.")
        return None
    params = {"action": "parse", "page": word, "prop": "text", "section": pron_index, "format": "json"}
    resp = requests.get(api_url, params=params)
    return resp.json()["parse"]["text"]["*"]

def parse_plain_ipa_li(li):
    region_span = li.find('span', class_='usage-label-accent')
    if region_span:
        region = region_span.get_text(" ", strip=True)
    else:
        text = li.get_text(" ", strip=True)
        m = re.match(r'\(\s*([^)]*?)\s*\)', text)
        region = m.group(1) if m else "default"
    region = normalize_region(region)
    ipa_spans = li.find_all('span', class_='IPA')
    ipas = [span.get_text(strip=True) for span in ipa_spans]
    debug(f"Plain li: region={region}, IPAs={ipas}")
    return {'region': region, 'ipas': ipas} if ipas else None

def parse_audio_block(li):
    table = li.find('table', class_='audiotable')
    if not table:
        return None
    td = table.find('td')
    if td:
        text = td.get_text(" ", strip=True)
        m = re.search(r'Audio\s*\(([^)]+)\)', text)
        region = m.group(1).strip() if m else None
    else:
        region = None
    region = normalize_region(region)
    ipa_span = table.find('span', class_='IPA')
    narrow_ipa = ipa_span.get_text(strip=True) if ipa_span else None
    audio_tag = li.find('audio')
    if audio_tag and audio_tag.has_attr('data-mwtitle'):
        audio_file = audio_tag['data-mwtitle']
    else:
        a_tag = li.find('a', href=re.compile(r'/wiki/File:'))
        if a_tag:
            m = re.search(r'/wiki/File:(.+)', a_tag['href'])
            audio_file = m.group(1) if m else None
        else:
            audio_file = None
    debug(f"Audio block: region={region}, narrow_ipa={narrow_ipa}, file={audio_file}")
    return {'region': region, 'audio_file': audio_file, 'narrow_ipa': narrow_ipa} if region and audio_file else None

def is_broad(ipa):
    return ipa.startswith("/") and ipa.endswith("/")

def parse_pronunciation_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    top_ul = soup.find('ul')
    if not top_ul:
        return []
    
    plain_dict = {}
    audio_list = []
    
    for li in top_ul.find_all('li', recursive=False):
        if not li.find('table', class_='audiotable'):
            plain = parse_plain_ipa_li(li)
            if plain:
                region = plain['region']
                plain_dict.setdefault(region, [])
                for ipa in plain['ipas']:
                    if ipa not in plain_dict[region]:
                        plain_dict[region].append(ipa)
    
    union_plain = []
    for ipalist in plain_dict.values():
        for ipa in ipalist:
            if ipa not in union_plain:
                union_plain.append(ipa)
    debug(f"Union of plain IPA: {union_plain}")
    
    for li in top_ul.find_all('li', recursive=False):
        if li.find('table', class_='audiotable'):
            nested_ul = li.find('ul')
            if nested_ul:
                li_copy = BeautifulSoup(str(li), 'html.parser')
                for ul in li_copy.find_all('ul'):
                    ul.decompose()
                parent = parse_plain_ipa_li(li_copy)
                parent_ipas = parent['ipas'] if parent else []
                for audio_li in nested_ul.find_all('li', recursive=False):
                    audio = parse_audio_block(audio_li)
                    if audio:
                        mapping_ipas = parent_ipas.copy()
                        if audio.get('narrow_ipa'):
                            mapping_ipas.append(audio['narrow_ipa'])
                        audio_list.append({
                            'region': audio['region'],
                            'ipas': mapping_ipas,
                            'audio_file': audio['audio_file']
                        })
            else:
                audio = parse_audio_block(li)
                plain = parse_plain_ipa_li(li)
                mapping_ipas = plain['ipas'] if plain else []
                if audio and audio.get('narrow_ipa'):
                    mapping_ipas.append(audio['narrow_ipa'])
                audio_list.append({
                    'region': audio['region'] if audio else None,
                    'ipas': mapping_ipas,
                    'audio_file': audio['audio_file'] if audio else None
                })
    
    debug(f"Plain dict: {plain_dict}")
    debug(f"Audio list: {audio_list}")
    
    final_mappings = []
    for mapping in audio_list:
        region = mapping.get('region')
        ipas = mapping.get('ipas', [])
        if region in plain_dict:
            for ipa in plain_dict[region]:
                if ipa not in ipas:
                    ipas.append(ipa)
        elif (not ipas or len(ipas) == 0) and len(union_plain) == 1:
            debug(f"Merging union_plain into mapping for region {region} because IPA list is empty.")
            ipas = union_plain.copy()
        mapping['ipas'] = ipas
        broad_count = sum(1 for ipa in ipas if is_broad(ipa))
        if broad_count >= 2:
            debug(f"Dropping mapping for region {region} due to {broad_count} broad transcriptions.")
            continue
        if ipas and mapping.get('audio_file'):
            final_mappings.append(mapping)
    
    debug(f"Final mappings: {final_mappings}")
    return final_mappings

def get_audio_file_url(file_name):
    api_url = "https://en.wiktionary.org/w/api.php"
    params = {"action": "query", "titles": f"File:{file_name}", "prop": "imageinfo", "iiprop": "url", "format": "json"}
    resp = requests.get(api_url, params=params)
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "imageinfo" in page:
            return page["imageinfo"][0]["url"]
    return None

def download_audio_file(url, save_path):
    headers = {
        'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                       'AppleWebKit/537.36 (KHTML, like Gecko) '
                       'Chrome/115.0 Safari/537.36')
    }
    response = requests.get(url, stream=True, headers=headers)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {save_path}")
    else:
        print(f"Failed to download {url} (Status code: {response.status_code})")

def scrape_word(original_word, output_dir):
    normalized = normalize_word(original_word)
    html = get_pronunciation_section_html(normalized)
    if not html:
        print(f"Skipping {original_word} due to missing Pronunciation section.")
        return []
    mappings = parse_pronunciation_page(html)
    if not mappings:
        print(f"Skipping {original_word}: no valid pronunciation mapping found.")
        return []
    rows = []
    wiktionary_url = get_wiktionary_url(normalized)
    for mapping in mappings:
        file_path = ""
        full_url = get_audio_file_url(mapping['audio_file'])
        if full_url:
            file_path = os.path.join(output_dir, mapping['audio_file'])
            if not os.path.exists(file_path):
                download_audio_file(full_url, file_path)
            file_path = convert_audio_to_wav(file_path)
        else:
            print(f"Could not get full URL for {mapping['audio_file']}")
        rows.append({
            "word": original_word,
            "region": mapping['region'],
            "IPAs": json.dumps(mapping['ipas']),
            "file_path": file_path,
            "wiktionary_url": wiktionary_url
        })
    return rows

# ----------------------------
# Multithreaded Worker Function
# ----------------------------
def worker(word, output_dir, processed_words, skipped_terms, lock):
    normalized = normalize_word(word)
    with lock:
        if normalized in processed_words or normalized in skipped_terms:
            print(f"Skipping {word} (normalized: {normalized}): already processed or previously skipped.")
            return (normalized, None)
    print(f"Processing {word} (normalized: {normalized})...")
    new_rows = scrape_word(word, output_dir)
    with lock:
        if new_rows:
            processed_words.add(normalized)
        else:
            skipped_terms.add(normalized)
    return (normalized, new_rows)

# ----------------------------
# Signal Handler for Ctrl+C
# ----------------------------
def signal_handler(signum, frame):
    print("\nCtrl+C pressed! Saving final snapshot...")
    shutdown_event.set()  # Signal periodic snapshot thread to stop.
    snapshot_save(global_csv_file, global_skipped_file, global_all_results, global_skipped_terms, global_fieldnames, global_order, global_lock)
    print("Progress saved. Exiting.")
    os._exit(0)

# Global variables for state saving (set in main)
global_csv_file = ""
global_skipped_file = ""
global_all_results = {}
global_skipped_terms = set()
global_fieldnames = []
global_order = []
global_lock = threading.Lock()
shutdown_event = threading.Event()

# ----------------------------
# Main Function with Argparse, Multithreading, and State Resumption
# ----------------------------
def main():
    global global_csv_file, global_skipped_file, global_all_results, global_skipped_terms, global_fieldnames, global_order, global_lock

    parser = argparse.ArgumentParser(
        description="Scrape Wiktionary pronunciation info, download audio files, convert to WAV, using multithreading."
    )
    parser.add_argument("--word-file", type=str, default="target_words.txt",
                        help="Path to a file containing words to scrape (one word per line).")
    parser.add_argument("words", nargs="*",
                        help='Words to scrape (e.g., "7 Up cake", "$100 hamburger", "tomato", "example", etc.)')
    parser.add_argument("--output-dir", default="./audio_files",
                        help="Directory to save downloaded audio files (default: ./audio_files)")
    parser.add_argument("--csv-file", default="output.csv",
                        help="CSV file to store output (default: output.csv)")
    parser.add_argument("--skipped-file", default="skipped_terms.txt",
                        help="Text file to store words skipped due to missing data (default: skipped_terms.txt)")
    parser.add_argument("--save-interval", type=int, default=60,
                        help="Interval in seconds to perform periodic snapshot saving (default: 60 seconds)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of worker threads to use (default: 8)")
    parser.add_argument("--debug", action="store_true", help="Enable debug output.")
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug
    
    # Gather words from the word file (if it exists) and positional arguments.
    words_to_process = []
    if args.word_file and os.path.exists(args.word_file):
        with open(args.word_file, "r", encoding="utf-8") as f:
            words_to_process.extend([line.strip() for line in f if line.strip()])
    words_to_process.extend(args.words)
    words_to_process = unique_preserve_order(words_to_process)
    
    if not words_to_process:
        print("No words to process.")
        return
    
    order_list = [normalize_word(w) for w in words_to_process]
    
    os.makedirs(args.output_dir, exist_ok=True)
    global_fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]
    global_csv_file = args.csv_file
    global_skipped_file = args.skipped_file
    
    global_all_results = load_existing_csv(args.csv_file)
    global_skipped_terms = load_skipped_terms(args.skipped_file)
    processed_words = {norm for norm, rows in global_all_results.items() if is_fully_processed(rows)}
    
    # Register the signal handler for Ctrl+C.
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start the periodic snapshot thread.
    snapshot_thread = threading.Thread(
        target=periodic_snapshot,
        args=(args.csv_file, args.skipped_file, global_all_results, global_skipped_terms,
              global_fieldnames, order_list, global_lock, args.save_interval),
        daemon=True
    )
    snapshot_thread.start()
    
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, word, args.output_dir, processed_words, global_skipped_terms, global_lock): word
                   for word in words_to_process}
        for future in concurrent.futures.as_completed(futures):
            norm, new_rows = future.result()
            with global_lock:
                if new_rows:
                    if norm in global_all_results:
                        global_all_results[norm] = merge_results(global_all_results[norm], new_rows)
                    else:
                        global_all_results[norm] = new_rows
            count += 1
    
    # Final snapshot save.
    shutdown_event.set()  # Signal snapshot thread to stop.
    snapshot_save(args.csv_file, args.skipped_file, global_all_results, global_skipped_terms, global_fieldnames, order_list, global_lock)

if __name__ == "__main__":
    main()