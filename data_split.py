import pandas as pd
import os

# -----------------------------
# 1. Load cleaned dataset
# -----------------------------
input_path = "yellow_tripdata_2025-09_cleaned.parquet"
output_dir = "splits"
os.makedirs(output_dir, exist_ok=True)

df = pd.read_parquet(input_path)

# -----------------------------
# 2. Ensure sorted by pickup datetime
# -----------------------------
df.sort_values("tpep_pickup_datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------
# 3. Compute split indices
# -----------------------------
n = len(df)
train_end = int(n * 0.35)
val_end = int(n * 0.70)

train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:]

# -----------------------------
# 4. Save splits
# -----------------------------
train_path = os.path.join(output_dir, "train.parquet")
val_path = os.path.join(output_dir, "validate.parquet")
test_path = os.path.join(output_dir, "test.parquet")

train_df.to_parquet(train_path, index=False)
val_df.to_parquet(val_path, index=False)
test_df.to_parquet(test_path, index=False)

print(f"Training set: {len(train_df)} rows → {train_path}")
print(f"Validation set: {len(val_df)} rows → {val_path}")
print(f"Test set: {len(test_df)} rows → {test_path}")
