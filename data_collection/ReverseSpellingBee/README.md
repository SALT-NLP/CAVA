# ReverseSpellingBee Data Collection Pipeline

The pipeline runs in 3 steps

## Get the terms to scrape from wiktionary

`python get_terms.py`

This gets a list of terms to scrape from wiktionary and saves them to `target_words.txt`

## Scrape the terms and pronunciations

`python scrape.py`

This downloads all the IPAs and the pronunciation audio files for all the words in `target_words.txt`, skipping any word that is missing relevant information.  It will save to `/audio_files`, `output.csv`, and `skipped_terms.txt`.

## Upload to huggingface (Optional)

`python upload_dataset.py --target_repo {repo}`

This script bundles the files into an accessible huggingface dataset for ease of use.  Ensure that your HF_TOKEN environment variable is set and that you have permission to write to the specified repo.