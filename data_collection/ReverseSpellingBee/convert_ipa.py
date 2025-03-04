import argparse
import json
import re
from datasets import load_dataset
# (Assume you have your necessary imports for push_to_hub, etc.)

# --- Conversion function (using our regex-based replacement) ---
def ipa_to_oed_with_stress(ipa_str, mapping):
    """
    Convert an IPA transcription into an OED respelling string,
    handling syllable segmentation and stress markers using the provided mapping.
    
    Processing steps:
      1. Remove surrounding whitespace, slashes, and square brackets.
      2. Remove combining diacritics (e.g. U+0329) to clear syllabic markers.
      3. Insert a hyphen before any stress marker (ˈ for primary, ˌ for secondary)
         that is not at the start.
      4. Replace IPA syllable separators (dots) and spaces with hyphens.
      5. Split the transcription into syllables.
      6. For each syllable:
         - If it begins with a primary stress marker (ˈ), remove it and flag the syllable as primary stressed.
         - If it begins with a secondary stress marker (ˌ), remove it and flag the syllable as secondary stressed.
      7. Build a regex from the mapping keys (sorted descending by length) and replace each IPA symbol.
      8. For primary-stressed syllables, uppercase the entire syllable;
         for secondary-stressed syllables, append a "·" at the end.
      9. Rejoin syllables with hyphens and clean duplicate hyphens.
     10. Remove any remaining length markers (ː) and combining diacritics.
     11. Remove any empty parentheses.
    
    Returns:
      str: The final OED respelling.
    """
    import re
    # Step 1: Remove surrounding whitespace, slashes, and square brackets.
    text = ipa_str.strip().strip("/").strip("[]")
    
    # Step 2: Remove combining diacritics (e.g., U+0329 for syllabicity)
    text = re.sub(r'[\u0329]', '', text)
    
    # Step 3: Remove any parenthetical (ɹ) substrings.
    text = text.replace("(ɹ)", "")
    
    # Step 4: Insert a hyphen before any stress marker (ˈ or ˌ) not at the start.
    text = re.sub(r"(?<!^)([ˈˌ])", r"-\1", text)
    # Step 5: Replace dots and spaces with hyphens.
    text = text.replace(".", "-").replace(" ", "-")
    
    # Step 6: Split into syllables.
    syllables = text.split("-")
    
    processed_syllables = []
    stress_flags = []  # 1: primary, 2: secondary, 0: none.
    for syl in syllables:
        if syl.startswith("ˈ"):
            stress_flags.append(1)
            processed_syllables.append(syl[1:])
        elif syl.startswith("ˌ"):
            stress_flags.append(2)
            processed_syllables.append(syl[1:])
        else:
            stress_flags.append(0)
            processed_syllables.append(syl)
    
    # Step 7: Build a regex from mapping keys (sorted descending by length) and perform replacements.
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)
    pattern = re.compile("|".join(re.escape(k) for k in sorted_keys))
    def replace_func(match):
        return mapping[match.group(0)]
    mapped_syllables = [pattern.sub(replace_func, syl) for syl in processed_syllables]
    
    # Step 8: Apply stress formatting.
    final_syllables = []
    for flag, syl in zip(stress_flags, mapped_syllables):
        if flag == 1:
            final_syllables.append(syl.upper())
        elif flag == 2:
            final_syllables.append(syl + "·")
        else:
            final_syllables.append(syl)
    
    # Step 9: Rejoin syllables and clean up duplicate hyphens.
    result = "-".join(final_syllables)
    result = re.sub(r"-{2,}", "-", result)
    
    # Step 10: Remove any remaining length markers and combining diacritics.
    result = result.replace("ː", "")
    result = re.sub(r'[\u0300-\u036f]', '', result)
    
    # Step 11: Remove empty parentheses.
    result = re.sub(r'\(\)', '', result)
    
    return result

# Updated mapping dictionary – note that we keep the previous keys.
ipa_oed_mapping = {
    'aɪ': 'igh',
    'aʊ': 'ow',
    'aᴜ': 'ow',
    'dʒ': 'j',
    'd͡ʒ': 'j',      # tie bar dropped
    'e': 'ay',
    'eɪ': 'ay',
    'g': 'g',
    'ɡ': 'g',       # both variants
    'h': 'h',
    'hw': 'hw',
    'i': 'ee',
    'iː': 'ee',
    'ɪ': 'i',       # ensure ɪ maps to i
    'j': 'y',
    'ju': 'yoo',
    'juː': 'yoo',
    'k': 'k',
    'kʰ': 'kh',     # aspirated k becomes kh
    'o': 'oh',
    'oʊ': 'oh',
    's': 's',
    'tʃ': 'ch',
    't͡ʃ': 'ch',     # tie bar dropped
    'u': 'oo',
    'uː': 'oo',
    'x': 'kh',
    'æ': 'a',
    'ð': 'dh',
    'ŋ': 'ng',
    'ɑ': 'ah',
    'ɑr': 'ar',
    'ɑː': 'ah',
    'ɑːr': 'ar',
    'ɒ': 'o',
    'ɔ': 'aw',
    'ɔr': 'or',
    'ɔɪ': 'oy',
    'ɔː': 'aw',
    'ɔːr': 'or',
    'ə': 'uh',
    'ər': 'ur',
    'ɚ': 'ur',
    'ɛ': 'e',
    'ɛr': 'air',
    'ɛər': 'air',
    'ɜ': 'ur',      # map ɜ to ur
    'ɜː': 'ur',     # map ɜː to ur
    'ɜr': 'ur',
    'ɜːr': 'ur',
    'ɹ': 'r',
    'ɝ': 'ur',      # r-colored vowel maps to "ur"
    'ʃ': 'sh',
    'ʊ': 'uu',
    'ʊər': 'oor',
    'ʌ': 'u',
    'ʒ': 'zh',
    'θ': 'th',
    'ᴜ': 'uu',
    'ᴜr': 'oor',
    'ɾ': 'r',       # tapped r becomes r
    'ʔ': '',        # drop glottal stop
    'ɨ̞': 'i',      # normalized to plain i
    'ɫ': 'l',       # dark L normalized to l
    'ː': '',         # remove length markers
    'təʊ': 'toh',
}

# --- Dataset processing wrapper ---
def process_dataset(dataset, mapping):
    """
    Given a HuggingFace dataset with an "IPAs" column,
    parse the first element of the IPAs column (assumed to be JSON-encoded)
    and convert it using ipa_to_oed_with_stress.
    The resulting respelling is added to a new "OED" column.
    """
    def convert_example(example):
        ipas = example["IPAs"]
        ipa_first = ipas[0] if ipas and isinstance(ipas, list) and len(ipas) > 0 else ""
        example["OED"] = ipa_to_oed_with_stress(ipa_first, mapping)
        return example
    return dataset.map(convert_example)

def main():
    parser = argparse.ArgumentParser(
        description="Load a dataset from HuggingFace, convert the first element of the IPAs column to OED respelling, add an 'OED' column, and push the dataset to the hub."
    )
    parser.add_argument("--repo", type=str, default="MichaelR207/wiktionary_pronunciations",
                        help="HuggingFace repository ID for the dataset.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional dataset configuration name.")
    parser.add_argument("--split", type=str, default="train",
                        help="Dataset split to process (e.g. 'train').")
    args = parser.parse_args()
    
    # Load dataset from HuggingFace.
    if args.config:
        dataset = load_dataset(args.repo, args.config, split=args.split)
    else:
        dataset = load_dataset(args.repo, split=args.split)
    
    # Process the dataset.
    processed_dataset = process_dataset(dataset, ipa_oed_mapping)
    
    # Push the processed dataset back to the Hub.
    processed_dataset.push_to_hub(args.repo, private=False)
    
if __name__ == "__main__":
    main()