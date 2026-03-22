import csv

# Input and output file paths
input_file = "MaliciousInstruct.txt"
output_file = "MaliciousInstruct.csv"

# Read lines from txt file
with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

# Write to CSV
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["behaviours"])  # column name
    for line in lines:
        writer.writerow([line])

print(f"CSV saved to {output_file}")