import pandas as pd
csv_files = [
    "arena21_placements.csv",
    "arena22_placements.csv",
    "arena23_placements.csv",
    "arena24_placements.csv"
]

dfs = [pd.read_csv(f) for f in csv_files]
combined = pd.concat(dfs, ignore_index=True)
combined.to_csv("new_placements.csv", index=False)

print("Combined CSV saved as all_placements.csv")