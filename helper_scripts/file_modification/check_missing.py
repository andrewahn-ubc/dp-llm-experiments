import os
import glob

pattern = "/Users/andrewahn/Desktop/DP-LLM-Safety Research/dp-llm-experiments/official/pair_results/test_pair_output_*.csv"

files = glob.glob(pattern)

existing_indices = set()

for f in files:
    basename = os.path.basename(f)
    num_str = basename.replace("test_pair_output_", "").replace(".csv", "")
    
    try:
        existing_indices.add(int(num_str))
    except ValueError:
        print(f"Skipping unexpected file: {basename}")

expected_indices = set(range(108))

# Find missing ones
missing = sorted(expected_indices - existing_indices)

print(f"Missing {len(missing)} test files:")

for i in missing:
    print(f"{i:02d}")  # zero-padded for consistency