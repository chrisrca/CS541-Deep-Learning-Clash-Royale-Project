import pyarrow.parquet as pq

table = pq.read_table("./training.parquet")
total_rows = table.num_rows

print(f"Analyzing {total_rows} samples...")

# Initialize counters and trackers
valid_placements = 0
invalid_placements = 0

min_x = float('inf')
max_x = float('-inf')
min_y = float('inf')
max_y = float('-inf')

for i in range(total_rows):
    row_table = table.slice(i, 1)
    data = row_table.to_pydict()

    x = data["x"][0]
    y = data["y"][0]

    # Check if placement is valid (not -1, -1)
    if x != -1 or y != -1:
        valid_placements += 1
        # Only track min/max for valid placements
        min_x = min(min_x, x)
        max_x = max(max_x, x)
        min_y = min(min_y, y)
        max_y = max(max_y, y)
    else:
        invalid_placements += 1

# Print statistics
print("\n=== PLACEMENT VALIDITY ===")
valid_percentage = (valid_placements / total_rows) * 100 if total_rows > 0 else 0
invalid_percentage = (invalid_placements / total_rows) * 100 if total_rows > 0 else 0
print(f"Valid placements (not -1, -1): {valid_placements} ({valid_percentage:.2f}%)")
print(f"Invalid placements (-1, -1): {invalid_placements} ({invalid_percentage:.2f}%)")

print("\n=== X/Y RANGE (for valid placements) ===")
if valid_placements > 0:
    print(f"X range: min={min_x}, max={max_x}")
    print(f"Y range: min={min_y}, max={max_y}")
else:
    print("No valid placements found")
