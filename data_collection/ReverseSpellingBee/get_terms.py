import requests
import time
import logging

# Set up logging to both the console and a file if desired.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define the API endpoint and initial parameters
url = "https://en.wiktionary.org/w/api.php"
params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": "Category:English_terms_with_audio_pronunciation",
    "cmlimit": "max",
    "format": "json"
}

titles = []
total_requests = 0
total_time = 0

while True:
    start_time = time.time()
    response = requests.get(url, params=params)
    elapsed = time.time() - start_time
    total_requests += 1
    total_time += elapsed

    data = response.json()

    # Extract the category members
    members = data.get("query", {}).get("categorymembers", [])
    for member in members:
        titles.append(member["title"])

    # Log current progress
    if titles:
        last_title = titles[-1]
        first_letter = last_title[0] if last_title else "?"
    else:
        last_title = "None"
        first_letter = "None"
    avg_time = total_time / total_requests
    logging.info(
        f"Request {total_requests}: Took {elapsed:.2f}s (avg {avg_time:.2f}s). "
        f"Total terms processed: {len(titles)}. "
        f"Latest term: '{last_title}' (starts with '{first_letter}')."
    )

    # Check for pagination
    if "continue" in data:
        params.update(data["continue"])
    else:
        break

# Write the titles to target_words.txt
with open("target_words.txt", "w", encoding="utf-8") as f:
    for title in titles:
        f.write(title + "\n")

logging.info(f"Completed scraping: Collected {len(titles)} terms in target_words.txt")