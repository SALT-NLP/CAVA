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
    """
    Normalize region names so that similar ones are merged.
    For example, treat "General American", "American", and "American English" as equivalent to "US".
    """
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
    """
    Convert the given word into a valid Wiktionary page title.
    E.g., "7 Up Cake" becomes "7_Up_Cake" (preserving case).
    """
    parts = word.strip().split()
    normalized = "_".join(parts)
    return normalized

def get_wiktionary_url(normalized_word):
    """Construct the Wiktionary URL for the given normalized word."""
    return f"https://en.wiktionary.org/wiki/{normalized_word}"

def is_fully_processed(rows):
    """
    Returns True if every row with a non-empty file_path corresponds to an existing file.
    """
    for row in rows:
        path = row["file_path"].strip()
        if path and not os.path.exists(path):
            return False
    return True

def load_existing_csv(csv_path):
    """
    Load existing CSV rows into a dictionary mapping normalized word -> list of rows.
    Each row has keys: word, region, IPAs, file_path, wiktionary_url.
    """
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
    """Load skipped normalized words from a text file into a set."""
    skipped = set()
    if os.path.exists(skipped_path):
        with open(skipped_path, mode="r", encoding="utf-8") as f:
            for line in f:
                term = line.strip()
                if term:
                    skipped.add(term)
    return skipped

def save_skipped_terms(skipped_path, skipped_set):
    """Write the skipped normalized words to the text file (one per line)."""
    with open(skipped_path, mode="w", encoding="utf-8") as f:
        for term in sorted(skipped_set):
            f.write(term + "\n")
    print(f"Saved {len(skipped_set)} skipped terms to {skipped_path}")

def save_all_results(csv_path, all_results, fieldnames):
    """Overwrite the CSV file with all the results from the all_results dictionary."""
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in all_results.values():
            for row in rows:
                writer.writerow(row)
    print(f"Saved CSV with {sum(len(v) for v in all_results.values())} rows to {csv_path}")

# ----------------------------
# Audio Conversion Function
# ----------------------------
def convert_audio_to_wav(input_path):
    """
    If the input audio file is not in WAV format, convert it using ffmpeg.
    The converted file will have the same basename with a .wav extension.
    The original file is removed.
    Returns the path of the .wav file.
    """
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".wav":
        return input_path
    output_path = os.path.splitext(input_path)[0] + ".wav"
    debug(f"Converting {input_path} to WAV at {output_path}")
    try:
        subprocess.run(["ffmpeg", "-y", "-i", input_path, output_path],
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    params = {
        "action": "parse",
        "page": word,
        "prop": "sections",
        "format": "json"
    }
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
    params = {
        "action": "parse",
        "page": word,
        "prop": "text",
        "section": pron_index,
        "format": "json"
    }
    resp = requests.get(api_url, params=params)
    html = resp.json()["parse"]["text"]["*"]
    return html

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
    if ipas:
        return {'region': region, 'ipas': ipas}
    else:
        return None

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
    if region and audio_file:
        return {'region': region, 'audio_file': audio_file, 'narrow_ipa': narrow_ipa}
    else:
        return None

def is_broad(ipa):
    """
    Return True if the IPA string is a "broad" transcription (enclosed in slashes).
    """
    return ipa.startswith("/") and ipa.endswith("/")

def parse_pronunciation_page(html):
    """
    Two-pass approach:
      1. Build a plain_dict from all li's without an audio table.
      2. Process li's with an audio table.
    Then build a union of all plain IPA values (ignoring region).
    For each audio mapping, merge in any plain IPA where regions match.
    If an audio mapping's IPA list is empty and union has exactly one element, merge that union.
    Finally, if a mapping contains two or more broad transcriptions, drop it.
    Return only mappings that have both a non-empty IPA list and an audio_file.
    """
    soup = BeautifulSoup(html, 'html.parser')
    top_ul = soup.find('ul')
    if not top_ul:
        return []
    
    plain_dict = {}  # region -> IPA list from li's without audio
    audio_list = []  # list of mappings from li's with audio
    
    # First pass: plain mappings.
    for li in top_ul.find_all('li', recursive=False):
        if not li.find('table', class_='audiotable'):
            plain = parse_plain_ipa_li(li)
            if plain:
                region = plain['region']
                plain_dict.setdefault(region, [])
                for ipa in plain['ipas']:
                    if ipa not in plain_dict[region]:
                        plain_dict[region].append(ipa)
    
    # Build union of all plain IPA values (ignoring regions)
    union_plain = []
    for ipalist in plain_dict.values():
        for ipa in ipalist:
            if ipa not in union_plain:
                union_plain.append(ipa)
    debug(f"Union of plain IPA: {union_plain}")
    
    # Second pass: audio mappings.
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
        # Merge in plain IPA if region matches.
        if region in plain_dict:
            for ipa in plain_dict[region]:
                if ipa not in ipas:
                    ipas.append(ipa)
        # If IPA list is empty and union_plain has exactly one element, merge that.
        elif (not ipas or len(ipas) == 0) and len(union_plain) == 1:
            debug(f"Merging union_plain into mapping for region {region} because IPA list is empty.")
            ipas = union_plain.copy()
        mapping['ipas'] = ipas
        
        # NEW: Drop mapping if there are two or more broad transcriptions.
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
    params = {
        "action": "query",
        "titles": f"File:{file_name}",
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }
    resp = requests.get(api_url, params=params)
    data = resp.json()
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        if "imageinfo" in page:
            return page["imageinfo"][0]["url"]
    return None

def download_audio_file(url, save_path):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' +
                      'AppleWebKit/537.36 (KHTML, like Gecko) ' +
                      'Chrome/115.0 Safari/537.36'
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
            # Convert file to WAV if needed.
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
# Main Function with Argparse, Multithreading, and Skip List
# ----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Scrape Wiktionary pronunciation info, download audio files, convert to WAV, using multithreading."
    )
    parser.add_argument(
        "words",
        nargs="+",
        help='Words to scrape (e.g., "7 Up cake", "$100 hamburger", "tomato", "example", "a bad penny always turns up", "abbess").'
    )
    parser.add_argument(
        "--output-dir",
        default="./audio_files",
        help="Directory to save downloaded audio files (default: ./audio_files)"
    )
    parser.add_argument(
        "--csv-file",
        default="output.csv",
        help="CSV file to store output (default: output.csv)"
    )
    parser.add_argument(
        "--skipped-file",
        default="skipped_terms.txt",
        help="Text file to store words skipped due to missing data (default: skipped_terms.txt)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Save CSV and skipped file after processing every N words (default: 5)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker threads to use (default: 8)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output."
    )
    args = parser.parse_args()
    
    global DEBUG
    DEBUG = args.debug
    
    os.makedirs(args.output_dir, exist_ok=True)
    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]
    all_results = load_existing_csv(args.csv_file)
    skipped_terms = load_skipped_terms(args.skipped_file)
    processed_words = {norm for norm, rows in all_results.items() if is_fully_processed(rows)}
    
    # Lock to protect shared data.
    lock = threading.Lock()
    count = 0
    words_to_process = args.words
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, word, args.output_dir, processed_words, skipped_terms, lock): word for word in words_to_process}
        for future in concurrent.futures.as_completed(futures):
            norm, new_rows = future.result()
            with lock:
                if new_rows:
                    all_results[norm] = new_rows
            count += 1
            if count % args.save_interval == 0:
                with lock:
                    save_all_results(args.csv_file, all_results, fieldnames)
                    save_skipped_terms(args.skipped_file, skipped_terms)
    # Final save after all threads complete.
    save_all_results(args.csv_file, all_results, fieldnames)
    save_skipped_terms(args.skipped_file, skipped_terms)

if __name__ == "__main__":
    main()