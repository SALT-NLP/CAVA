import os
import json
import threading
import concurrent.futures
import tempfile
import pytest
from scrape import (
    normalize_word,
    scrape_word,
    load_existing_csv,
    load_skipped_terms,
    save_all_results,
    save_skipped_terms,
    worker
)

# Dummy functions to simulate network calls.
class DummyScrape:
    @staticmethod
    def get_pronunciation_section_html(word):
        # Return a simple HTML snippet for testing.
        return """
        <ul>
          <li>
            <span class="ib-brac">(</span>
            <span class="ib-content">
              <span class="usage-label-accent">
                <a href="https://en.wikipedia.org/wiki/Received_Pronunciation" title="w:Received Pronunciation">US</a>
              </span>
            </span>
            <span class="ib-brac">)</span> 
            <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
            <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>: 
            <span class="IPA">/dummy/</span>
          </li>
          <li>
            <style>.mw-parser-output .k-player .k-attribution{visibility:hidden}</style>
            <table class="audiotable">
              <tbody>
                <tr>
                  <td>Audio <span class="ib-brac">(</span>
                    <span class="usage-label-accent">
                      <span class="ib-content">
                        <a href="https://en.wikipedia.org/wiki/American_English" title="w:American English">US</a>
                      </span>
                    </span>
                    <span class="ib-brac">)</span>:
                  </td>
                  <td class="audiofile">
                    <span typeof="mw:File">
                      <span>
                        <span class="mw-tmh-player audio mw-file-element" data-mwtitle="dummy.ogg">
                          <audio></audio>
                          <a href="/wiki/File:dummy.ogg" title="Play audio" role="button"></a>
                        </span>
                      </span>
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </li>
        </ul>
        """
    @staticmethod
    def get_audio_file_url(file_name):
        return "http://dummy.url/" + file_name

@pytest.fixture(autouse=True)
def patch_network(monkeypatch):
    monkeypatch.setattr("scrape.get_pronunciation_section_html", DummyScrape.get_pronunciation_section_html)
    monkeypatch.setattr("scrape.get_audio_file_url", DummyScrape.get_audio_file_url)
    monkeypatch.setattr("scrape.download_audio_file", lambda url, path: print(f"Fake download {url} -> {path}"))
    monkeypatch.setattr("scrape.convert_audio_to_wav", lambda path: path.replace(".ogg", ".wav"))

def test_multithreading_integration(tmp_path):
    # Prepare temporary CSV and skipped file paths.
    csv_file = tmp_path / "output.csv"
    skipped_file = tmp_path / "skipped_terms.txt"
    output_dir = tmp_path / "audio_files"
    output_dir.mkdir()
    
    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]
    all_results = {}
    skipped_terms = set()
    processed_words = set()
    lock = threading.Lock()
    
    words = ["word1", "word2", "word3", "word4", "word5", "word6"]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(worker, word, str(output_dir), processed_words, skipped_terms, lock): word for word in words}
        for future in concurrent.futures.as_completed(futures):
            norm, new_rows = future.result()
            with lock:
                if new_rows:
                    all_results[norm] = new_rows
    
    # Save results (simulate incremental save)
    save_all_results(str(csv_file), all_results, fieldnames)
    save_skipped_terms(str(skipped_file), skipped_terms)
    
    # Verify that the dummy worker produced results.
    for word in words:
        norm = normalize_word(word)
        # In our dummy HTML every word returns one mapping.
        assert norm in all_results

# This test simulates multiple threads concurrently updating shared data
# and then calls the file-saving functions under a lock.

def test_concurrent_file_io(tmp_path):
    # Setup temporary CSV and skipped terms file paths.
    csv_file = tmp_path / "output.csv"
    skipped_file = tmp_path / "skipped_terms.txt"
    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]
    
    # Shared data structures
    all_results = {}
    skipped_terms = set()
    lock = threading.Lock()

    # Dummy worker function that updates the shared dictionary.
    def dummy_worker(word, region, ipas, file_path, wiktionary_url):
        norm = normalize_word(word)
        with lock:
            # Simulate updating results.
            all_results.setdefault(norm, []).append({
                "word": word,
                "region": region,
                "IPAs": json.dumps(ipas),
                "file_path": file_path,
                "wiktionary_url": wiktionary_url
            })
            # Also add to skipped if region is "skip" (simulate a failure case)
            if region == "skip":
                skipped_terms.add(norm)
    
    # Create a list of dummy data that multiple threads will process.
    dummy_data = [
        ("word1", "US", ["/ipa1/"], "./audio_files/word1.wav", "https://en.wiktionary.org/wiki/word1"),
        ("word2", "UK", ["/ipa2/"], "./audio_files/word2.wav", "https://en.wiktionary.org/wiki/word2"),
        ("word3", "skip", ["/ipa3/"], "./audio_files/word3.wav", "https://en.wiktionary.org/wiki/word3"),
        ("word4", "US", ["/ipa4/"], "./audio_files/word4.wav", "https://en.wiktionary.org/wiki/word4"),
        ("word5", "skip", ["/ipa5/"], "./audio_files/word5.wav", "https://en.wiktionary.org/wiki/word5"),
    ]
    
    # Use ThreadPoolExecutor to simulate concurrent updates.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(dummy_worker, *data) for data in dummy_data]
        # Wait for all futures to complete.
        for future in concurrent.futures.as_completed(futures):
            future.result()
    
    # Now, under the lock, save the results concurrently.
    with lock:
        save_all_results(str(csv_file), all_results, fieldnames)
        save_skipped_terms(str(skipped_file), skipped_terms)
    
    # Read back the CSV and verify that all non-skipped words are present.
    loaded_results = load_existing_csv(str(csv_file))
    # We expect word1, word2, word4 to be in all_results
    for word in ["word1", "word2", "word4"]:
        norm = normalize_word(word)
        assert norm in loaded_results, f"{word} should be in CSV"
    # And word3 and word5 should be in the skipped list.
    loaded_skipped = load_skipped_terms(str(skipped_file))
    for word in ["word3", "word5"]:
        norm = normalize_word(word)
        assert norm in loaded_skipped, f"{word} should be in skipped terms"

if __name__ == "__main__":
    pytest.main()