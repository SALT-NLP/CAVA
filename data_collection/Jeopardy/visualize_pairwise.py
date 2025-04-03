import json
import matplotlib.pyplot as plt
import itertools
import numpy as np
from collections import defaultdict

ROOT_DIR = "/nlp/scr/askhan1/jeopardy/CATS"
RESULTS_DIR = f"{ROOT_DIR}/data_collection/Jeopardy/results/"
COMBINED_JSON_PATH = f'{RESULTS_DIR}/combined_results.json'

with open(COMBINED_JSON_PATH, 'r', encoding='utf-8') as f:
    combined_data = json.load(f)


models = [
    "GPT-4o_realtime",
    "Gemini-2.0_multimodal",
]

def parse_time(t):
    """
    Convert 'Processing Time' strings to float. If invalid, use float('inf').
    Example: "0.35297", "0.19769", or "0.35s".
    """
    if isinstance(t, str):
        t = t.strip()
        if t.endswith('s'):
            t = t[:-1]  # remove trailing 's'
    try:
        return float(t)
    except:
        return float('inf')


def compute_pairwise_results(q_ids, data_dict, models_list):
    """
    For each pair of models in models_list, compute:
      - wins1: times model1 "wins"
      - wins2: times model2 "wins"
      - ties:  times neither or both equally "win"
    
    Criteria:
      - If both correct, faster (lower Processing Time) wins.
      - If one correct, that model wins.
      - If neither correct, tie.
      - If both correct & same time, tie.
    Returns dict: { (m1, m2): { "wins1","wins2","ties","win_diff","total_rounds" } }
    """
    pairwise_results = {}
    for m1, m2 in itertools.combinations(models_list, 2):
        wins1 = 0
        wins2 = 0
        ties = 0
        for qid in q_ids:
            qinfo = data_dict[qid]
            results = qinfo.get("results", {})
            
            r1 = results.get(m1, {})
            r2 = results.get(m2, {})
            
            correct1 = bool(r1.get("correct", False))
            correct2 = bool(r2.get("correct", False))
            time1 = parse_time(r1.get("processing_time", "inf"))
            time2 = parse_time(r2.get("processing_time", "inf"))

            # Decide outcome
            if not correct1 and not correct2:
                # Both incorrect
                ties += 1
            elif correct1 and not correct2:
                # Only model1 correct
                wins1 += 1
            elif not correct1 and correct2:
                # Only model2 correct
                wins2 += 1
            else:
                # both correct -> compare times
                if time1 < time2:
                    wins1 += 1
                elif time2 < time1:
                    wins2 += 1
                else:
                    ties += 1
        
        total = wins1 + wins2 + ties
        if total == 0:
            pairwise_results[(m1, m2)] = {
                "wins1": 0,
                "wins2": 0,
                "ties": 0,
                "win_diff": 0,
                "total_rounds": 0
            }
        else:
            pairwise_results[(m1, m2)] = {
                "wins1": 100.0 * wins1 / total,
                "wins2": 100.0 * wins2 / total,
                "ties": 100.0 * ties / total,
                "win_diff": abs(wins1 - wins2) / total * 100.0,
                "total_rounds": total
            }
    return pairwise_results


def create_big_bar_chart(overall_results, cat_results, categories, title, output_file=None):
    """
    overall_results: dict from compute_pairwise_results() for ALL questions
    cat_results: dict of category -> pairwise_result dict
    categories: list of categories
      We'll produce a single chart that has:
        For each pair (sorted by overall difference):
          1. A row for "Overall"
          2. Then a row for each category
    We keep Model1 always on the left, Ties in the middle, Model2 on the right.
    """
    # Sort pairs by overall "win_diff" (descending)
    sorted_pairs = sorted(overall_results.items(), key=lambda x: x[1]["win_diff"], reverse=True)
    
    # Build a row-data list in the display order with "Overall" first
    row_data = []
    for (m1, m2), over_res in sorted_pairs:
        # Overall row
        row_data.append({
            "m1": m1,
            "m2": m2,
            "label": "Overall",
            "wins1": over_res["wins1"],
            "ties": over_res["ties"],
            "wins2": over_res["wins2"],
            "total_rounds": over_res["total_rounds"]
        })
        
        # Category-specific rows
        for cat in categories:
            pair_res = cat_results[cat].get((m1, m2), None)
            if not pair_res:
                pair_res = {"wins1": 0, "wins2": 0, "ties": 0, "win_diff":0, "total_rounds":0}
            
            row_data.append({
                "m1": m1,
                "m2": m2,
                "label": cat,
                "wins1": pair_res["wins1"],
                "ties": pair_res["ties"],
                "wins2": pair_res["wins2"],
                "total_rounds": pair_res["total_rounds"]
            })
    
    # Now draw them all in one figure
    fig, ax = plt.subplots(figsize=(16, max(8, len(row_data)*0.5)))
    ax.set_xlim(-10, 110)  # space on left & right for labels
    
    y_positions = np.arange(len(row_data))
    bar_height = 0.6
    
    # Colors
    color_model1 = 'blue'
    color_ties   = 'gray'
    color_model2 = 'red'
    
    for i, row in enumerate(row_data):
        wins1 = row["wins1"]
        ties_val = row["ties"]
        wins2 = row["wins2"]
        category = row["label"]
        
        # Plot stacked bars for this row
        ax.barh(y_positions[i], wins1, bar_height, left=0, color=color_model1)
        ax.barh(y_positions[i], ties_val, bar_height, left=wins1, color=color_ties)
        ax.barh(y_positions[i], wins2, bar_height, left=wins1 + ties_val, color=color_model2)
        
        # Add optional text inside each segment if wide enough
        if wins1 > 5:
            ax.text(wins1/2, y_positions[i], f"{wins1:.1f}%", 
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if ties_val > 5:
            ax.text(wins1 + ties_val/2, y_positions[i], f"{ties_val:.1f}%", 
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if wins2 > 5:
            ax.text(wins1 + ties_val + wins2/2, y_positions[i], f"{wins2:.1f}%", 
                    ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        
        # Display n=value and category name below the bar
        ax.text(50, y_positions[i] - bar_height/2 - 0.1,
                f"n = {row['total_rounds']}",
                ha='center', va='top', fontsize=8, color='black')
        
        ax.text(50, y_positions[i] - bar_height/2 - 0.25,
                f"{category}",
                ha='center', va='top', fontsize=8, color='black', fontweight='bold')
        
        # If this is the "Overall" row, highlight it
        if category == "Overall":
            ax.text(50, y_positions[i] - bar_height/2 - 0.25,
                   f"{category}",
                   ha='center', va='top', fontsize=8, color='black', 
                   fontweight='bold', bbox=dict(facecolor='lightyellow', alpha=0.8, pad=2))
        
        # Label on the left: model1
        ax.text(-5, y_positions[i], f"{row['m1']}",
                ha='right', va='center', fontsize=9, fontweight='bold', color=color_model1)
        
        # Label on the right: model2
        ax.text(105, y_positions[i], f"{row['m2']}",
                ha='left', va='center', fontsize=9, fontweight='bold', color=color_model2)
    
    # Remove y tick labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([""]*len(row_data))
    
    # Vertical grid lines
    for x_val in [0, 25, 50, 75, 100]:
        ax.axvline(x=x_val, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    
    # Title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Percentage (%)', fontsize=12)
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    else:
        plt.savefig('pairwise_all_in_one.png', dpi=300, bbox_inches='tight')
    
    plt.show()


all_qids = list(combined_data.keys())

category_to_qids = defaultdict(list)
for qid_str, qinfo in combined_data.items():
    cat = qinfo.get("category", "Uncategorized")
    category_to_qids[cat].append(qid_str)

# 6B) Compute overall pairwise stats
overall_stats = compute_pairwise_results(all_qids, combined_data, models)

# 6C) Compute category-wise pairwise stats
cat_results = {}
categories = sorted(category_to_qids.keys())
for cat in categories:
    qids_cat = category_to_qids[cat]
    cat_results[cat] = compute_pairwise_results(qids_cat, combined_data, models)

# Tracking correct counts for accuracy
model_correct_count = defaultdict(int)      # overall
model_correct_count_by_cat = defaultdict(lambda: defaultdict(int))

# Tracking total questions for each category is category_to_qids
num_total = len(all_qids)  # total questions overall

# Tracking sums of processing times (and how many times we got a valid time)
model_time_sum = defaultdict(float)      # overall
model_time_count = defaultdict(int)      # overall

model_time_sum_by_cat = defaultdict(lambda: defaultdict(float))   # by cat
model_time_count_by_cat = defaultdict(lambda: defaultdict(int))   # by cat

# Go through all questions, accumulate correctness & processing times
for qid_str, qinfo in combined_data.items():
    cat = qinfo.get("category", "Uncategorized")
    for model_name, mres in qinfo.get("results", {}).items():
        # 1) Correctness
        if mres.get("correct", False) is True:
            model_correct_count[model_name] += 1
            model_correct_count_by_cat[model_name][cat] += 1
        

        # 2) Processing Time
        raw_time = mres.get("processing_time", None)
        tval = float(raw_time)
        if tval != 9999999.0:
            model_time_sum[model_name] += tval
            model_time_count[model_name] += 1
            model_time_sum_by_cat[model_name][cat] += tval
            model_time_count_by_cat[model_name][cat] += 1


# -----------------------
# Print Accuracy (Overall)
# -----------------------
print("\n===========================")
print("Accuracy (Overall):")
print("===========================")
for m in models:
    correct = model_correct_count[m]
    accuracy_overall = correct / num_total * 100.0 if num_total else 0.0
    print(f"{m:25s} => {correct}/{num_total} = {accuracy_overall:.1f}%")

# -----------------------
# Print Accuracy (by Cat)
# -----------------------
print("\n===========================")
print("Accuracy (By Category):")
print("===========================")
for m in models:
    print(f"\n----- {m} -----")
    for cat_name in sorted(category_to_qids.keys()):
        cat_total = len(category_to_qids[cat_name])
        cat_correct = model_correct_count_by_cat[m][cat_name]
        accuracy_cat = cat_correct / cat_total * 100.0 if cat_total else 0.0
        print(f"  {cat_name:25s} => {cat_correct}/{cat_total} = {accuracy_cat:.1f}%")

# -----------------------
# Print Mean Processing Time (Overall)
# -----------------------
print("\n===========================")
print("Mean Processing Time (Overall) [seconds]:")
print("===========================")
for m in models:
    s = model_time_sum[m]
    c = model_time_count[m]
    # print(s, c)
    mean_t = s / c if c > 0 else float('inf')
    print(f"{m:25s} => {mean_t:.4f}s (across {c} answers)")

# -----------------------
# Print Mean Processing Time (By Category)
# -----------------------
print("\n===========================")
print("Mean Processing Time (By Category) [seconds]:")
print("===========================")
for m in models:
    print(f"\n----- {m} -----")
    for cat_name in sorted(category_to_qids.keys()):
        s_cat = model_time_sum_by_cat[m][cat_name]
        c_cat = model_time_count_by_cat[m][cat_name]
        mean_cat = s_cat / c_cat if c_cat > 0 else float('inf')
        print(f"  {cat_name:25s} => {mean_cat:.4f}s (across {c_cat} answers)")

# 6D) Plot everything in one large chart
chart_title = (
    f"All-In-One Pairwise Model Performance for {len(all_qids)} total questions\n"
    "Followed by Category-Specific Results"
)

create_big_bar_chart(
    overall_results=overall_stats,
    cat_results=cat_results,
    categories=categories,
    title=chart_title,
    output_file=f'{RESULTS_DIR}/pairwise_all_in_one.png'
)

