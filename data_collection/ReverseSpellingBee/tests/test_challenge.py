import os
import csv
import json
import threading
import concurrent.futures
import tempfile
import pytest
from scrape import (
    normalize_word,
    load_existing_csv,
    load_skipped_terms,
    save_all_results,
    save_skipped_terms,
    is_fully_processed,
    worker,
    scrape_word
)

# --- Dummy class to simulate network calls (if needed) ---
class DummyScrape:
    @staticmethod
    def get_pronunciation_section_html(word):
        # Return a simple HTML snippet with one plain IPA and one audio block.
        # This dummy HTML is used by some tests.
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

# --- Pytest fixtures to patch network calls and audio conversion ---
@pytest.fixture(autouse=True)
def patch_network(monkeypatch):
    monkeypatch.setattr("scrape.get_pronunciation_section_html", DummyScrape.get_pronunciation_section_html)
    monkeypatch.setattr("scrape.get_audio_file_url", DummyScrape.get_audio_file_url)
    monkeypatch.setattr("scrape.download_audio_file", lambda url, path: print(f"Fake download {url} -> {path}"))
    monkeypatch.setattr("scrape.convert_audio_to_wav", lambda path: path.replace(".ogg", ".wav"))

# Test 1: State Resumption - Simulate an existing CSV and skipped file, then add new words concurrently.
def test_state_resumption(tmp_path, monkeypatch):
    csv_file = tmp_path / "output.csv"
    skipped_file = tmp_path / "skipped_terms.txt"
    output_dir = tmp_path / "audio_files"
    output_dir.mkdir()

    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]

    # Prepopulate CSV with one entry.
    initial_data = {
        "word1": [
            {"word": "word1", "region": "US", "IPAs": json.dumps(["/ipa1/"]), "file_path": "./audio_files/word1.wav", "wiktionary_url": "https://en.wiktionary.org/wiki/word1"}
        ]
    }
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in initial_data.values():
            for row in rows:
                writer.writerow(row)
    # Prepopulate skipped_terms with one word.
    with open(skipped_file, "w", encoding="utf-8") as f:
        f.write("word2\n")

    # Monkey-patch scrape_word to return dummy mapping for new words.
    def dummy_scrape_word(word, output_dir):
        norm = normalize_word(word)
        # Return dictionary using "file_path" key!
        return [{
            "word": word,
            "region": "US",
            "IPAs": ["/ipa_dummy/"],
            "file_path": f"dummy.ogg",
        }]
    monkeypatch.setattr("scrape.scrape_word", dummy_scrape_word)

    # Load existing state.
    all_results = load_existing_csv(str(csv_file))
    skipped_terms = load_skipped_terms(str(skipped_file))
    processed_words = {normalize_word(word) for word in all_results}
    lock = threading.Lock()

    new_words = ["word3", "word4"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = {executor.submit(worker, word, str(output_dir), processed_words, skipped_terms, lock): word for word in new_words}
        for future in concurrent.futures.as_completed(futures):
            norm, new_rows = future.result()
            with lock:
                if new_rows:
                    all_results[norm] = new_rows

    # Save final state.
    save_all_results(str(csv_file), all_results, fieldnames)
    save_skipped_terms(str(skipped_file), skipped_terms)

    # Reload and verify.
    final_results = load_existing_csv(str(csv_file))
    assert "word1" in final_results
    assert "word3" in final_results
    assert "word4" in final_results
    final_skipped = load_skipped_terms(str(skipped_file))
    assert "word2" in final_skipped

# Test 2: Concurrent File I/O Race Conditions.
def test_concurrent_file_save(tmp_path):
    csv_file = tmp_path / "output.csv"
    skipped_file = tmp_path / "skipped_terms.txt"
    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]

    all_results = {}
    skipped_terms = set()
    lock = threading.Lock()

    def dummy_update(word):
        norm = normalize_word(word)
        with lock:
            all_results.setdefault(norm, []).append({
                "word": word,
                "region": "US",
                "IPAs": json.dumps(["/ipa_concurrent/"]),
                "file_path": f"./audio_files/{norm}.wav",
                "wiktionary_url": f"https://en.wiktionary.org/wiki/{norm}"
            })

    words = [f"word{i}" for i in range(10)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(dummy_update, words))

    def concurrent_save():
        with lock:
            save_all_results(str(csv_file), all_results, fieldnames)
            save_skipped_terms(str(skipped_file), skipped_terms)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(concurrent_save) for _ in range(5)]
        for future in concurrent.futures.as_completed(futures):
            future.result()

    loaded = load_existing_csv(str(csv_file))
    for word in words:
        norm = normalize_word(word)
        assert norm in loaded, f"{word} should be present in the saved CSV"

# Test 3: Missing Audio File Scenario.
def test_missing_audio_file(tmp_path):
    csv_file = tmp_path / "output.csv"
    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]
    data = [{
        "word": "missing_audio",
        "region": "US",
        "IPAs": json.dumps(["/ipa_missing/"]),
        "file_path": "./audio_files/missing_audio.wav",  # Simulate missing file.
        "wiktionary_url": "https://en.wiktionary.org/wiki/missing_audio"
    }]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(row)
    
    results = load_existing_csv(str(csv_file))
    for rows in results.values():
        assert not is_fully_processed(rows), "File should be reported as not fully processed if missing."

# Test 4: Test scraping of a tricky page (e.g., abbess) by simulating its HTML.
def test_scrape_tricky_abbess(tmp_path, monkeypatch):
    # We'll simulate the abbess page HTML (only English part) using a small snippet.
    abbess_html = """
    <ul>
      <li>
        <span class="ib-brac">(</span>
        <span class="ib-content">
          <span class="usage-label-accent">
            <a href="https://en.wikipedia.org/wiki/Received_Pronunciation" title="w:Received Pronunciation">Received Pronunciation</a>
          </span>
        </span>
        <span class="ib-brac">)</span>
        <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
        <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>:
        <span class="IPA">/ˈæb.ɪs/</span>, <span class="IPA">/ˈæb.ɛs/</span>
      </li>
      <li>
        <style>.mw-parser-output .k-player .k-attribution{visibility:hidden}</style>
        <table class="audiotable">
          <tbody>
            <tr>
              <td>Audio <span class="ib-brac">(</span>
                <span class="usage-label-accent">
                  <a href="https://en.wikipedia.org/wiki/American_English" title="w:American English">US</a>
                </span>
                <span class="ib-brac">)</span>:
              </td>
              <td class="audiofile">
                <span typeof="mw:File">
                  <span>
                    <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-us-abbess.ogg">
                      <audio></audio>
                      <a href="/wiki/File:en-us-abbess.ogg" title="Play audio" role="button"></a>
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
    # Monkey-patch get_pronunciation_section_html to return abbess_html.
    monkeypatch.setattr("scrape.get_pronunciation_section_html", lambda word: abbess_html)
    # Monkey-patch download and conversion functions so they don't actually try to download.
    monkeypatch.setattr("scrape.download_audio_file", lambda url, path: print(f"Fake download {url} -> {path}"))
    monkeypatch.setattr("scrape.convert_audio_to_wav", lambda path: path.replace(".ogg", ".wav"))
    
    output_dir = tmp_path / "audio_files"
    output_dir.mkdir()
    rows = scrape_word("abbess", str(output_dir))
    # According to our current logic, abbess should be dropped because it has two broad transcriptions.
    assert len(rows) == 0

# Test 5: Test scraping of a tricky page (e.g., tomato) by simulating its HTML.
def test_scrape_tricky_tomato(tmp_path, monkeypatch):
    tomato_html = """
    <ul>
      <li>
        <span class="ib-brac">(</span>
        <span class="ib-content">
          <span class="usage-label-accent">
            <a href="https://en.wikipedia.org/wiki/British_English" title="w:British English">UK</a><span class="ib-comma">,</span>
            <a href="https://en.wikipedia.org/wiki/Australian_English_phonology" title="w:Australian English phonology">General Australian</a>
          </span>
        </span>
        <span class="ib-brac">)</span>
        <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
        <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>:
        <span class="IPA">/təˈmɑː.təʊ/</span>
        <ul>
          <li>
            <style>.mw-parser-output .k-player .k-attribution{visibility:hidden}</style>
            <table class="audiotable">
              <tbody>
                <tr>
                  <td>Audio <span class="ib-brac">(</span>
                    <span class="usage-label-accent">
                      <span class="ib-content">
                        <a href="https://en.wikipedia.org/wiki/British_English" title="w:British English">UK</a>
                      </span>
                    </span>
                    <span class="ib-brac">)</span><span class="ib-semicolon">;</span>
                    <span class="IPA">[tʰə̥ˈmɑːtʰəʉ̯]</span>:
                  </td>
                  <td class="audiofile">
                    <span typeof="mw:File">
                      <span>
                        <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-uk-tomato.ogg">
                          <audio></audio>
                          <a href="/wiki/File:en-uk-tomato.ogg" title="Play audio" role="button"></a>
                        </span>
                      </span>
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </li>
          <li>
            <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r50165410">
            <table class="audiotable">
              <tbody>
                <tr>
                  <td>Audio <span class="ib-brac">(</span>
                    <span class="usage-label-accent">
                      <span class="ib-content">
                        <a href="https://en.wikipedia.org/wiki/Australian_English_phonology" title="w:Australian English phonology">General Australian</a>
                      </span>
                    </span>
                    <span class="ib-brac">)</span><span class="ib-semicolon">;</span>
                    <span class="IPA">[tʰə̥ˈmɐːtʰɐʉ̯]</span>:
                  </td>
                  <td class="audiofile">
                    <span typeof="mw:File">
                      <span>
                        <span class="mw-tmh-player audio mw-file-element" data-mwtitle="En-au-tomato.ogg">
                          <audio></audio>
                          <a href="/wiki/File:En-au-tomato.ogg" title="Play audio" role="button"></a>
                        </span>
                      </span>
                    </span>
                  </td>
                </tr>
              </tbody>
            </table>
          </li>
          <li>
            <span class="ib-brac">(</span>
            <span class="ib-content">
              <span class="usage-label-accent">
                <a href="https://en.wikipedia.org/wiki/American_English" title="w:American English">US</a>
              </span>
            </span>
            <span class="ib-brac">)</span>
            <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
            <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>:
            <span class="IPA">/təˈmeɪ.toʊ/</span>
            <ul>
              <li>
                <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r50165410">
                <table class="audiotable">
                  <tbody>
                    <tr>
                      <td>Audio <span class="ib-brac">(</span>
                        <span class="usage-label-accent">
                          <span class="ib-content">
                            <a href="https://en.wikipedia.org/wiki/American_English" title="w:American English">US</a>
                          </span>
                        </span>
                        <span class="ib-brac">)</span><span class="ib-semicolon">;</span>
                        <span class="IPA">[tʰə̥ˈmeɪɾoʊ]</span>:
                      </td>
                      <td class="audiofile">
                        <span typeof="mw:File">
                          <span>
                            <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-us-tomato.ogg">
                              <audio></audio>
                              <a href="/wiki/File:en-us-tomato.ogg" title="Play audio" role="button"></a>
                            </span>
                          </span>
                        </span>
                      </td>
                    </tr>
                  </tbody>
                </table>
              </li>
            </ul>
          </li>
        </ul>
    """
    monkeypatch.setattr("scrape.get_pronunciation_section_html", lambda word: tomato_html)
    monkeypatch.setattr("scrape.download_audio_file", lambda url, path: print(f"Fake download {url} -> {path}"))
    monkeypatch.setattr("scrape.convert_audio_to_wav", lambda path: path.replace(".ogg", ".wav"))
    
    output_dir = tmp_path / "audio_files"
    output_dir.mkdir()
    rows = scrape_word("tomato", str(output_dir))
    # Expect three mappings: UK, General Australian, and US.
    assert len(rows) == 3
    for row in rows:
        # Check that each file_path ends with .wav
        assert row["file_path"].endswith(".wav")
        # Check that the wiktionary URL is correct.
        assert row["wiktionary_url"] == "https://en.wiktionary.org/wiki/tomato"

if __name__ == "__main__":
    pytest.main()