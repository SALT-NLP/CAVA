import argparse
import requests
import sys
from datasets import load_dataset
from diskcache import Cache  # <-- 1) Diskcache

# ------------------------------------------------------------
# PART 1: POS Extraction Helpers
# ------------------------------------------------------------

# Common parts of speech we look for
KNOWN_POS = {
    "noun", "verb", "adjective", "adverb", "pronoun", "preposition",
    "conjunction", "interjection", "determiner", "article", "auxiliary",
    "proper noun", "abbreviation", "participle", "numeral", "infinitive", "gerund"
}

def get_wiktionary_sections(term):
    """
    Call the Wiktionary API to fetch the sections of a given term.
    Returns a list of section objects (each has a "line" field).
    """
    url = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "parse",
        "page": term,
        "prop": "sections",
        "format": "json"
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        # Non-200 => some network or server error
        return []
    data = response.json()
    if "error" in data:
        # The API might complain if the page doesn't exist
        return []
    return data.get("parse", {}).get("sections", [])

def extract_pos(term):
    """
    Given a term, query the Wiktionary API for sections
    and return a sorted, deduplicated list of recognized parts-of-speech.
    """
    sections = get_wiktionary_sections(term)
    pos_set = set()
    for section in sections:
        heading = section.get("line", "").strip()
        heading_lower = heading.lower()
        # Direct exact match?
        if heading_lower in KNOWN_POS:
            pos_set.add(heading_lower)
        else:
            # If the heading includes extra words, check the first word
            first_word = heading_lower.split()[0]
            if first_word in KNOWN_POS:
                pos_set.add(first_word)
    return sorted(pos_set)

# ------------------------------------------------------------
# PART 2: Main Script (Argparse, Processing, Reupload)
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Add or retrieve parts-of-speech data from Wiktionary."
    )
    parser.add_argument(
        "--words",
        nargs="*",
        help="List of words for which to fetch POS. Example usage: --words cat dog apple"
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        help="Hugging Face dataset repo ID, e.g. 'username/dataset_name'"
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="If set, push the modified dataset back to the same Hugging Face repo."
    )
    # 2) Argument to control parallel processing
    parser.add_argument(
        "--num_proc",
        type=int,
        default=1,
        help="Number of processes/threads for parallel processing in dataset.map (default: 1)."
    )
    # 1) Diskcache settings
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="pos_cache",
        help="Directory to store the diskcache (default: 'pos_cache')."
    )

    args = parser.parse_args()

    # Create the diskcache
    cache = Cache(directory=args.cache_dir, size_limit=0)
    EXPIRE_SECONDS = 7 * 24 * 60 * 60  # 1 week

    def get_cached_pos(term):
        """
        Check diskcache first. If 'term' not found, call extract_pos and store in cache.
        """
        cached_result = cache.get(term)
        if cached_result is not None:
            return cached_result
        # Not in cache => compute, then store with 1-week expiration
        pos_list = extract_pos(term)
        cache.set(term, pos_list, expire=EXPIRE_SECONDS)
        return pos_list

    # Case 1: If user passes a list of words
    if args.words and not args.hf_repo:
        print("Processing list of words:")
        for w in args.words:
            pos_list = get_cached_pos(w)
            if pos_list:
                print(f"{w}: {pos_list}")
            else:
                print(f"{w}: [No POS found]")
        return

    # Case 2: If user passes a huggingface repo
    if args.hf_repo:
        print(f"Loading dataset from {args.hf_repo} ...")
        dataset_dict = load_dataset(args.hf_repo)

        # We'll process all splits in the dataset
        for split_name, ds in dataset_dict.items():
            if "word" not in ds.column_names:
                print(f"Warning: no 'word' column found in split '{split_name}'. Skipping.")
                continue

            print(f"Extracting POS for split '{split_name}' with {len(ds)} rows...")

            # 2 & 3) Use dataset.map with num_proc for parallelism, desc for progress bar
            def pos_mapper(example):
                w = example["word"]
                return {"pos": get_cached_pos(w)}

            ds = ds.map(
                pos_mapper,
                num_proc=args.num_proc,
                desc=f"Processing split '{split_name}'"
            )
            dataset_dict[split_name] = ds

        # Optionally push to hub
        if args.push_to_hub:
            print("Pushing updated dataset to Hugging Face...")
            dataset_dict.push_to_hub(args.hf_repo)
            print("Push completed.")
        else:
            print("push_to_hub not specified; changes are local only.")
        return

    # If neither words nor hf_repo were passed, show help
    parser.print_help()

if __name__ == "__main__":
    main()

# ------------------------------------------------------------
# Example Usage:
# ------------------------------------------------------------
# python get_pos.py --words cat dog apple
# python get_pos.py --hf_repo MichaelR207/wiktionary_pronunciations --push_to_hub --num_proc 8
# ------------------------------------------------------------