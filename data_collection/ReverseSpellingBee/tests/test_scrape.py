import os
import csv
import json
import tempfile
import subprocess
import pytest
from io import StringIO
from bs4 import BeautifulSoup

# Import functions from your scrape.py file.
from scrape import (
    normalize_region,
    normalize_word,
    is_broad,
    convert_audio_to_wav,
    parse_plain_ipa_li,
    parse_audio_block,
    parse_pronunciation_page,
    get_wiktionary_url,
    scrape_word,
    load_existing_csv,
    load_skipped_terms,
    save_all_results,
    save_skipped_terms,
)

# ----------------------------
# Tests for Utility Functions
# ----------------------------
def test_normalize_region():
    assert normalize_region("General American") == "US"
    assert normalize_region("american english") == "US"
    assert normalize_region("Southern England") == "Southern England"
    assert normalize_region(None) is None

def test_normalize_word():
    # preserving case and replacing spaces with underscores
    assert normalize_word("7 Up Cake") == "7_Up_Cake"
    assert normalize_word("$100 hamburger") == "$100_hamburger"

def test_is_broad():
    assert is_broad("/test/") is True
    assert is_broad("[test]") is False
    assert is_broad("test") is False

def test_get_wiktionary_url():
    norm = normalize_word("tomato")
    url = get_wiktionary_url(norm)
    assert url == "https://en.wiktionary.org/wiki/tomato"

# ----------------------------
# Tests for File Saving Functions
# ----------------------------
def test_save_all_results(tmp_path):
    csv_file = tmp_path / "output.csv"
    fieldnames = ["word", "region", "IPAs", "file_path", "wiktionary_url"]
    # Create dummy data
    all_results = {
        "tomato": [
            {
                "word": "tomato",
                "region": "US",
                "IPAs": json.dumps(["/təˈmeɪ.toʊ/", "[tʰə̥ˈmeɪɾoʊ]"]),
                "file_path": "./audio_files/en-us-tomato.wav",
                "wiktionary_url": "https://en.wiktionary.org/wiki/tomato"
            }
        ]
    }
    save_all_results(str(csv_file), all_results, fieldnames)
    # Read the CSV back in and verify contents.
    with open(csv_file, "r", encoding="utf-8") as f:
        content = f.read()
    assert "tomato" in content
    assert "./audio_files/en-us-tomato.wav" in content

def test_save_skipped_terms(tmp_path):
    skipped_file = tmp_path / "skipped_terms.txt"
    skipped_set = {"$100_hamburger", "7_Up_cake"}
    save_skipped_terms(str(skipped_file), skipped_set)
    with open(skipped_file, "r", encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert "$100_hamburger" in lines
    assert "7_Up_cake" in lines

# ----------------------------
# Tests for Audio Conversion
# ----------------------------
def test_convert_audio_to_wav(tmp_path, monkeypatch):
    # Create a dummy .ogg file in tmp_path
    ogg_file = tmp_path / "dummy.ogg"
    ogg_file.write_bytes(b"dummy data")
    
    # Monkey-patch subprocess.run to simulate conversion.
    def fake_run(args, check, stdout, stderr):
        output = args[-1]
        with open(output, "wb") as f:
            f.write(b"converted wav data")
    monkeypatch.setattr(subprocess, "run", fake_run)
    
    result = convert_audio_to_wav(str(ogg_file))
    assert result.endswith(".wav")
    assert not ogg_file.exists()
    with open(result, "rb") as f:
        data = f.read()
    assert data == b"converted wav data"
    os.remove(result)

# ----------------------------
# Tests for Parsing Functions
# ----------------------------
EXAMPLE_PLAIN_HTML = """
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
  <span class="IPA">/ɪɡˈzɑːm.pəl/</span>
</li>
"""
def test_parse_plain_ipa_li():
    soup = BeautifulSoup(EXAMPLE_PLAIN_HTML, 'html.parser')
    li = soup.find('li')
    result = parse_plain_ipa_li(li)
    assert result is not None
    assert result['region'] == "Received Pronunciation"
    assert "/ɪɡˈzɑːm.pəl/" in result['ipas']

TOMATO_AUDIO_HTML = """
<li>
  <style>.mw-parser-output .k-player .k-attribution{visibility:hidden}</style>
  <table class="audiotable" style="vertical-align: middle; display: inline-block; list-style: none; line-height: 1em; border-collapse: collapse; margin: 0;">
    <tbody>
      <tr>
        <td>Audio <span class="ib-brac">(</span>
          <span class="usage-label-accent">
            <span class="ib-content">
              <a href="https://en.wikipedia.org/wiki/American_English" title="w:American English">US</a>
            </span>
          </span>
          <span class="ib-brac">)</span><span class="ib-colon">:</span>
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
"""
def test_parse_audio_block():
    soup = BeautifulSoup(TOMATO_AUDIO_HTML, 'html.parser')
    li = soup.find('li')
    result = parse_audio_block(li)
    assert result is not None
    assert result['region'] == "US"
    assert result['audio_file'] == "en-us-tomato.ogg"

# ----------------------------
# Integration Test for Parsing "example" Page
# ----------------------------
EXAMPLE_FULL_HTML = """
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
    <span class="IPA">/ɪɡˈzɑːm.pəl/</span>
  </li>
  <li>
    <span class="ib-brac">(</span>
    <span class="ib-content">
      <span class="usage-label-accent">
        <a href="https://en.wikipedia.org/wiki/English_language_in_Northern_England" title="w:English language in Northern England">Northern England</a>,
        <a href="https://en.wikipedia.org/wiki/Scottish_English" title="w:Scottish English">Scotland</a>
      </span>
    </span>
    <span class="ib-brac">)</span> 
    <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
    <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>: 
    <span class="IPA">/ɪɡˈzam.pəl/</span>
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
    <span class="IPA">/ɪɡˈzæm.pəl/</span>, <span class="IPA">[ɪɡˈzɛəmpəɫ]</span>
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
                <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-us-example.ogg">
                  <audio></audio>
                  <a href="/wiki/File:en-us-example.ogg" title="Play audio" role="button"></a>
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
def test_parse_pronunciation_page_example():
    mappings = parse_pronunciation_page(EXAMPLE_FULL_HTML)
    us_mappings = [m for m in mappings if m['region'] == "US"]
    assert len(us_mappings) == 1
    mapping = us_mappings[0]
    assert mapping['ipas']
    assert mapping['audio_file'] == "en-us-example.ogg"
    broad = [ipa for ipa in mapping['ipas'] if is_broad(ipa)]
    assert len(broad) < 2

# ----------------------------
# Test for the tricky case "abbess"
# ----------------------------
ABBESS_HTML = """
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
    <span class="ib-brac">(</span>
    <span class="ib-content">
      <span class="usage-label-accent">
        <a href="https://en.wikipedia.org/wiki/General_American_English" title="w:General American English">General American</a>
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
              <span class="ib-content">
                <a href="https://en.wikipedia.org/wiki/American_English" title="w:American English">US</a>
              </span>
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
  <li>
    <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r50165410">
    <table class="audiotable">
      <tbody>
        <tr>
          <td>Audio <span class="ib-brac">(</span>
            <span class="usage-label-accent">
              <span class="ib-content">
                <a href="https://en.wikipedia.org/wiki/Canadian_English" title="w:Canadian English">Canada</a>
              </span>
            </span>
            <span class="ib-brac">)</span>:
          </td>
          <td class="audiofile">
            <span typeof="mw:File">
              <span>
                <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-ca-abbess.ogg">
                  <audio></audio>
                  <a href="/wiki/File:en-ca-abbess.ogg" title="Play audio" role="button"></a>
                </span>
              </span>
            </span>
          </td>
        </tr>
      </tbody>
    </table>
  </li>
  <li>
    Rhymes: <a href="/w/index.php?title=Rhymes:English/%C3%A6b%C9%AAs&amp;action=edit&amp;redlink=1" class="new" title="Rhymes:English/æbɪs (page does not exist)"><span class="IPA">-æbɪs</span></a>, 
    <a href="/w/index.php?title=Rhymes:English/%C3%A6b%C9%9Bs&amp;action=edit&amp;redlink=1" class="new" title="Rhymes:English/æbɛs (page does not exist)"><span class="IPA">-æbɛs</span></a>
  </li>
</ul>
"""
def test_parse_pronunciation_page_abbess():
    mappings = parse_pronunciation_page(ABBESS_HTML)
    # According to our rules, if there are two or more broad transcriptions in one mapping, it is dropped.
    # Therefore, abbess should result in an empty list.
    assert mappings == []

# ----------------------------
# Integration Test for Skip Logic
# ----------------------------
def test_skip_logic(tmp_path):
    csv_file = tmp_path / "output.csv"
    skipped_file = tmp_path / "skipped_terms.txt"
    
    # Write a dummy CSV.
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "region", "IPAs", "file_path", "wiktionary_url"])
        writer.writeheader()
        writer.writerow({
            "word": "tomato",
            "region": "US",
            "IPAs": json.dumps(["/təˈmeɪ.toʊ/", "[tʰə̥ˈmeɪɾoʊ]"]),
            "file_path": "./audio_files/en-us-tomato.wav",
            "wiktionary_url": "https://en.wiktionary.org/wiki/tomato"
        })
    with open(skipped_file, "w", encoding="utf-8") as f:
        f.write("$100_hamburger\n")
    
    results = load_existing_csv(str(csv_file))
    skipped = load_skipped_terms(str(skipped_file))
    assert "tomato" in results
    assert "$100_hamburger" in skipped

# ----------------------------
# Integration Test for scrape_word using dummy network functions.
# ----------------------------
EXAMPLE_HTML = """
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
    <span class="IPA">/ɪɡˈzɑːm.pəl/</span>
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
                <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-us-example.ogg">
                  <audio></audio>
                  <a href="/wiki/File:en-us-example.ogg" title="Play audio" role="button"></a>
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
class DummyScrape:
    @staticmethod
    def get_pronunciation_section_html(word):
        return EXAMPLE_HTML
    @staticmethod
    def get_audio_file_url(file_name):
        return "http://dummy.url/" + file_name

@pytest.fixture(autouse=True)
def patch_network(monkeypatch):
    monkeypatch.setattr("scrape.get_pronunciation_section_html", DummyScrape.get_pronunciation_section_html)
    monkeypatch.setattr("scrape.get_audio_file_url", DummyScrape.get_audio_file_url)
    monkeypatch.setattr("scrape.download_audio_file", lambda url, path: print(f"Fake download {url} -> {path}"))
    monkeypatch.setattr("scrape.convert_audio_to_wav", lambda path: path.replace(".ogg", ".wav"))

def test_scrape_word_integration(tmp_path):
    output_dir = tmp_path / "audio_files"
    output_dir.mkdir()
    rows = scrape_word("example", str(output_dir))
    # Expect one mapping for US.
    assert len(rows) == 1
    row = rows[0]
    assert row["word"] == "example"
    data_ipas = json.loads(row["IPAs"])
    assert "/ɪɡˈzɑːm.pəl/" in data_ipas
    assert row["file_path"].endswith(".wav")
    assert row["wiktionary_url"] == "https://en.wiktionary.org/wiki/example"

# ----------------------------
# Integration Test for the tricky "tomato" page.
# ----------------------------
TOMATO_FULL_HTML = """
<ul>
  <li>
    <span class="ib-brac qualifier-brac">(</span>
    <span class="ib-content qualifier-content">
      <span class="usage-label-accent">
        <a href="https://en.wikipedia.org/wiki/British_English" class="extiw" title="w:British English">UK</a><span class="ib-comma label-comma">,</span> 
        <a href="https://en.wikipedia.org/wiki/Australian_English_phonology" class="extiw" title="w:Australian English phonology">General Australian</a>
      </span>
    </span>
    <span class="ib-brac qualifier-brac">)</span> 
    <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
    <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>: 
    <span class="IPA">/təˈmɑː.təʊ/</span>
    <ul>
      <li>
        <style data-mw-deduplicate="TemplateStyles:r50165410">.mw-parser-output .k-player .k-attribution{visibility:hidden}</style>
        <table class="audiotable" style="vertical-align: middle; display: inline-block; list-style: none; line-height: 1em; border-collapse: collapse; margin: 0;">
          <tbody>
            <tr>
              <td>Audio <span class="ib-brac qualifier-brac">(</span>
                <span class="usage-label-accent">
                  <span class="ib-content label-content">
                    <a href="https://en.wikipedia.org/wiki/British_English" class="extiw" title="w:British English">UK</a>
                  </span>
                </span>
                <span class="ib-brac qualifier-brac">)</span><span class="ib-semicolon qualifier-semicolon">;</span> 
                <span class="IPA">[tʰə̥ˈmɑːtʰəʉ̯]</span><span class="ib-colon qualifier-colon">:</span>
              </td>
              <td class="audiofile">
                <span typeof="mw:File">
                  <span>
                    <span class="mw-tmh-player audio mw-file-element" style="width:175px;" data-mwtitle="en-uk-tomato.ogg">
                      <audio></audio>
                      <a href="/wiki/File:en-uk-tomato.ogg" title="Play audio" role="button"></a>
                      <span class="mw-tmh-duration mw-tmh-label">
                        <span class="sr-only">Duration: 2 seconds.</span>
                        <span aria-hidden="true">0:02</span>
                      </span>
                    </span>
                  </span>
                </span>
              </td>
              <td class="audiometa" style="font-size: 80%;">(<a href="/wiki/File:en-uk-tomato.ogg" title="File:en-uk-tomato.ogg">file</a>)</td>
            </tr>
          </tbody>
        </table>
      </li>
      <li>
        <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r50165410">
        <table class="audiotable" style="vertical-align: middle; display: inline-block; list-style: none; line-height: 1em; border-collapse: collapse; margin: 0;">
          <tbody>
            <tr>
              <td>Audio <span class="ib-brac qualifier-brac">(</span>
                <span class="usage-label-accent">
                  <span class="ib-content label-content">
                    <a href="https://en.wikipedia.org/wiki/Australian_English_phonology" class="extiw" title="w:Australian English phonology">General Australian</a>
                  </span>
                </span>
                <span class="ib-brac qualifier-brac">)</span><span class="ib-semicolon qualifier-semicolon">;</span> 
                <span class="IPA">[tʰə̥ˈmɐːtʰɐʉ̯]</span><span class="ib-colon qualifier-colon">:</span>
              </td>
              <td class="audiofile">
                <span typeof="mw:File">
                  <span>
                    <span class="mw-tmh-player audio mw-file-element" style="width:175px;" data-mwtitle="En-au-tomato.ogg">
                      <audio></audio>
                      <a href="/wiki/File:En-au-tomato.ogg" title="Play audio" role="button"></a>
                      <span class="mw-tmh-duration mw-tmh-label">
                        <span class="sr-only">Duration: 2 seconds.</span>
                        <span aria-hidden="true">0:02</span>
                      </span>
                    </span>
                  </span>
                </span>
              </td>
              <td class="audiometa" style="font-size: 80%;">(<a href="/wiki/File:En-au-tomato.ogg" title="File:En-au-tomato.ogg">file</a>)</td>
            </tr>
          </tbody>
        </table>
      </li>
    </ul>
  </li>
  <li>
    <span class="ib-brac qualifier-brac">(</span>
    <span class="ib-content qualifier-content">
      <span class="usage-label-accent">
        <a href="https://en.wikipedia.org/wiki/American_English" class="extiw" title="w:American English">US</a>
      </span>
    </span>
    <span class="ib-brac qualifier-brac">)</span> 
    <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
    <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>: 
    <span class="IPA">/təˈmeɪ.toʊ/</span>
    <ul>
      <li>
        <link rel="mw-deduplicated-inline-style" href="mw-data:TemplateStyles:r50165410">
        <table class="audiotable" style="vertical-align: middle; display: inline-block; list-style: none; line-height: 1em; border-collapse: collapse; margin: 0;">
          <tbody>
            <tr>
              <td>Audio <span class="ib-brac qualifier-brac">(</span>
                <span class="usage-label-accent">
                  <span class="ib-content label-content">
                    <a href="https://en.wikipedia.org/wiki/American_English" class="extiw" title="w:American English">US</a>
                  </span>
                </span>
                <span class="ib-brac qualifier-brac">)</span><span class="ib-semicolon qualifier-semicolon">;</span> 
                <span class="IPA">[tʰə̥ˈmeɪɾoʊ]</span><span class="ib-colon qualifier-colon">:</span>
              </td>
              <td class="audiofile">
                <span typeof="mw:File">
                  <span>
                    <span class="mw-tmh-player audio mw-file-element" style="width:175px;" data-mwtitle="en-us-tomato.ogg">
                      <audio></audio>
                      <a href="/wiki/File:en-us-tomato.ogg" title="Play audio" role="button"></a>
                      <span class="mw-tmh-duration mw-tmh-label">
                        <span class="sr-only">Duration: 1 second.</span>
                        <span aria-hidden="true">0:01</span>
                      </span>
                    </span>
                  </span>
                </span>
              </td>
              <td class="audiometa" style="font-size: 80%;">(<a href="/wiki/File:en-us-tomato.ogg" title="File:en-us-tomato.ogg">file</a>)</td>
            </tr>
          </tbody>
        </table>
        </link>
      </li>
    </ul>
  </li>
  <li>
    <span class="ib-brac qualifier-brac">(</span>
    <span class="ib-content qualifier-content">
      <span class="usage-label-accent">
        <a href="https://en.wikipedia.org/wiki/Canadian_English" class="extiw" title="w:Canadian English">Canada</a>
      </span>
    </span>
    <span class="ib-brac qualifier-brac">)</span> 
    <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
    <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>: 
    <span class="IPA">[tʰəˈmeɪɾo]</span>, <span class="IPA">[tʰəˈme(ː)to(ː)]</span>, <span class="IPA">[tʰɵ-]</span>, <span class="IPA">[-ma-]</span>
  </li>
  <li>
    <span class="ib-brac qualifier-brac">(</span>
    <span class="ib-content qualifier-content">
      <span class="usage-label-accent">
        <a href="https://en.wikipedia.org/wiki/South_Asian_English" class="extiw" title="w:South Asian English">Indic</a>
      </span>
    </span>
    <span class="ib-brac qualifier-brac">)</span> 
    <a href="/wiki/Wiktionary:International_Phonetic_Alphabet" title="Wiktionary:International Phonetic Alphabet">IPA</a>
    <sup>(<a href="/wiki/Appendix:English_pronunciation" title="Appendix:English pronunciation">key</a>)</sup>: 
    <span class="IPA">/ʈoˈmæʈo/</span>
  </li>
  <li>
    Rhymes: <a href="/wiki/Rhymes:English/%C9%91%CB%90t%C9%99%CA%8A" title="Rhymes:English/ɑːtəʊ"><span class="IPA">-ɑːtəʊ</span></a>, 
    <a href="/wiki/Rhymes:English/e%C9%AAt%C9%99%CA%8A" title="Rhymes:English/eɪtəʊ"><span class="IPA">-eɪtəʊ</span></a>, 
    <a href="/wiki/Rhymes:English/%C3%A6t%C9%99%CA%8A" title="Rhymes:English/ætəʊ"><span class="IPA">-ætəʊ</span></a>
  </li>
</ul>
"""
def test_parse_pronunciation_page_tomato():
    mappings = parse_pronunciation_page(TOMATO_FULL_HTML)
    # For tomato, we expect three mappings:
    # 1. UK mapping with audio_file "en-uk-tomato.ogg" and IPA ["/təˈmɑː.təʊ/", "[tʰə̥ˈmɑːtʰəʉ̯]"]
    # 2. General Australian mapping with audio_file "En-au-tomato.ogg" and IPA ["/təˈmɑː.təʊ/", "[tʰə̥ˈmɐːtʰɐʉ̯]"]
    # 3. US mapping with audio_file "en-us-tomato.ogg" and IPA ["/təˈmeɪ.toʊ/", "[tʰə̥ˈmeɪɾoʊ]"]
    uk_mapping = next((m for m in mappings if m['region'] == "UK"), None)
    ga_mapping = next((m for m in mappings if m['region'] == "General Australian"), None)
    us_mapping = next((m for m in mappings if m['region'] == "US"), None)
    assert uk_mapping is not None
    assert ga_mapping is not None
    assert us_mapping is not None
    assert uk_mapping['audio_file'] == "en-uk-tomato.ogg"
    assert ga_mapping['audio_file'] == "En-au-tomato.ogg"
    assert us_mapping['audio_file'] == "en-us-tomato.ogg"
    uk_ipas = uk_mapping['ipas']
    ga_ipas = ga_mapping['ipas']
    us_ipas = us_mapping['ipas']
    assert "/təˈmɑː.təʊ/" in uk_ipas
    assert "[tʰə̥ˈmɑːtʰəʉ̯]" in uk_ipas
    assert "/təˈmɑː.təʊ/" in ga_ipas
    assert "[tʰə̥ˈmɐːtʰɐʉ̯]" in ga_ipas
    assert "/təˈmeɪ.toʊ/" in us_ipas
    assert "[tʰə̥ˈmeɪɾoʊ]" in us_ipas

# ----------------------------
# Integration Test for Skip Logic
# ----------------------------
def test_skip_logic(tmp_path):
    csv_file = tmp_path / "output.csv"
    skipped_file = tmp_path / "skipped_terms.txt"
    
    # Write a dummy CSV.
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["word", "region", "IPAs", "file_path", "wiktionary_url"])
        writer.writeheader()
        writer.writerow({
            "word": "tomato",
            "region": "US",
            "IPAs": json.dumps(["/təˈmeɪ.toʊ/", "[tʰə̥ˈmeɪɾoʊ]"]),
            "file_path": "./audio_files/en-us-tomato.wav",
            "wiktionary_url": "https://en.wiktionary.org/wiki/tomato"
        })
    with open(skipped_file, "w", encoding="utf-8") as f:
        f.write("$100_hamburger\n")
    
    results = load_existing_csv(str(csv_file))
    skipped = load_skipped_terms(str(skipped_file))
    assert "tomato" in results
    assert "$100_hamburger" in skipped

# ----------------------------
# Integration Test for scrape_word using dummy network functions.
# ----------------------------
EXAMPLE_HTML = """
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
    <span class="IPA">/ɪɡˈzɑːm.pəl/</span>
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
                <span class="mw-tmh-player audio mw-file-element" data-mwtitle="en-us-example.ogg">
                  <audio></audio>
                  <a href="/wiki/File:en-us-example.ogg" title="Play audio" role="button"></a>
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
class DummyScrape:
    @staticmethod
    def get_pronunciation_section_html(word):
        return EXAMPLE_HTML
    @staticmethod
    def get_audio_file_url(file_name):
        return "http://dummy.url/" + file_name

@pytest.fixture(autouse=True)
def patch_network(monkeypatch):
    monkeypatch.setattr("scrape.get_pronunciation_section_html", DummyScrape.get_pronunciation_section_html)
    monkeypatch.setattr("scrape.get_audio_file_url", DummyScrape.get_audio_file_url)
    monkeypatch.setattr("scrape.download_audio_file", lambda url, path: print(f"Fake download {url} -> {path}"))
    monkeypatch.setattr("scrape.convert_audio_to_wav", lambda path: path.replace(".ogg", ".wav"))

def test_scrape_word_integration(tmp_path):
    output_dir = tmp_path / "audio_files"
    output_dir.mkdir()
    rows = scrape_word("example", str(output_dir))
    # Expect one mapping for US.
    assert len(rows) == 1
    row = rows[0]
    assert row["word"] == "example"
    data_ipas = json.loads(row["IPAs"])
    assert "/ɪɡˈzɑːm.pəl/" in data_ipas
    assert row["file_path"].endswith(".wav")
    assert row["wiktionary_url"] == "https://en.wiktionary.org/wiki/example"

if __name__ == "__main__":
    pytest.main()