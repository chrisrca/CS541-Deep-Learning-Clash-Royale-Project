import pandas as pd

# Replace with your parquet file name
parquet_file = "training_offset_1.parquet"
csv_file = "sample_100.csv"

# Read the parquet file
df = pd.read_parquet(parquet_file)
df_new = df.drop('png_bytes', axis =1)
# Get the first 100 rows
sample = df_new.head(100)

# Print the first 100 rows
print(sample)

# Save to CSV
sample.to_csv(csv_file, index=False)
print(f"Saved first 100 rows to {csv_file}")