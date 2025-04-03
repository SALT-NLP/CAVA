import json
import os
import glob

# Path to your results directory containing all the individual JSON files
RESULTS_DIR = "results/"
OUTPUT_JSON = RESULTS_DIR + "combined_results.json"

# Fallback for missing/empty results
MISSING_RESULT = {
    "transcript": None,
    "audio_path": None,
    "processing_time": "9999999",
    "success": False,
    "correct": False
}


merged_data = {}

# Get all JSON files in the directory
all_files = glob.glob(os.path.join(RESULTS_DIR, "*.json"))

# Go through each JSON file
for file_path in all_files:
    with open(file_path, "r", encoding="utf-8") as f: #ignoring combined_results.json
        if "combined_results" in file_path:
            continue
        print(file_path)
        data = json.load(f)

    any_question = next(iter(data.values()))
    # For safety, gather all model names found in the entire file:
    all_model_names = set()
    for qinfo in data.values():
        if "results" in qinfo and len(qinfo["results"]) > 0:
            all_model_names.update(qinfo["results"].keys())
    
    # For each question ID in the file
    for qid, qinfo in data.items():
        
        # If this question_id isn't in merged_data yet, initialize it with all the common fields.
        if qid not in merged_data:
            merged_data[qid] = {
                "question": qinfo["question"],
                "correct_answer": qinfo["correct_answer"],
                "results": {},
                "category": qinfo["category"],
            }

        file_results = qinfo.get("results", {})
        
        for model_name, model_data in file_results.items():
            merged_data[qid]["results"][model_name] = model_data
        

# second pass: fill missing models with fallback values.
all_models = set()
for qid, qinfo in merged_data.items():
    all_models.update(qinfo["results"].keys())

for qid, qinfo in merged_data.items():
    for model_name in all_models:
        if model_name not in qinfo["results"]:
            # This means for question qid, 'model_name' didn't appear. Fill fallback.
            qinfo["results"][model_name] = MISSING_RESULT.copy()

# Write out the final combined results to JSON
with open(OUTPUT_JSON, "w", encoding="utf-8") as out_f:
    json.dump(merged_data, out_f, indent=2, ensure_ascii=False)

print(f"Combined results written to {OUTPUT_JSON}")
