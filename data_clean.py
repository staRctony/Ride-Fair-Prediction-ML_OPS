import pandas as pd
import numpy as np

# -----------------------------
# 1. Load dataset
# -----------------------------
input_path = "yellow_tripdata_2025-09.parquet"
output_path = "yellow_tripdata_2025-09_cleaned.parquet"

df = pd.read_parquet(input_path)

# -----------------------------
# 2. Handle missing values
# -----------------------------
# Columns with missing values: passenger_count, RatecodeID
df.dropna(subset=["passenger_count", "RatecodeID"], inplace=True)

# Fill missing store_and_fwd_flag, congestion_surcharge, Airport_fee with 0 if any
df["store_and_fwd_flag"].fillna("N", inplace=True)
df["congestion_surcharge"].fillna(0, inplace=True)
df["Airport_fee"].fillna(0, inplace=True)

# -----------------------------
# 3. Fix negative and impossible values
# -----------------------------
num_cols = ["fare_amount", "total_amount", "extra", "mta_tax", "tip_amount",
            "tolls_amount", "improvement_surcharge", "congestion_surcharge", 
            "Airport_fee", "cbd_congestion_fee"]

# Remove rows with negative values
for col in num_cols:
    df = df[df[col] >= 0]

# Zero / impossible values
df = df[df["trip_distance"] > 0]
df.loc[df["passenger_count"] == 0, "passenger_count"] = 1
df = df[df["fare_amount"] > 0]
df = df[df["total_amount"] > 0]

# Remove rows with negative trip duration
df = df[df["tpep_dropoff_datetime"] >= df["tpep_pickup_datetime"]]

# -----------------------------
# 4. Remove outliers (realistic thresholds)
# -----------------------------
df = df[df["trip_distance"] < 100]      # miles
df = df[df["fare_amount"] < 500]       # USD
df = df[df["total_amount"] < 500]

# -----------------------------
# 5. Normalize text columns
# -----------------------------
# Convert store_and_fwd_flag to numeric
df["store_and_fwd_flag"] = df["store_and_fwd_flag"].str.upper().map({"Y":1, "N":0})

# -----------------------------
# 6. Encode categorical columns
# -----------------------------
cat_cols = ["VendorID", "RatecodeID", "payment_type", "PULocationID", "DOLocationID"]
for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# -----------------------------
# 7. Create time-based features
# -----------------------------
df["trip_duration"] = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60  # minutes
df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
df["is_weekend"] = df["pickup_dayofweek"].isin([5,6]).astype(int)

# -----------------------------
# 8. Sort by pickup datetime
# -----------------------------
df.sort_values("tpep_pickup_datetime", inplace=True)
df.reset_index(drop=True, inplace=True)

# -----------------------------
# 9. Save cleaned dataset as Parquet
# -----------------------------
df.to_parquet(output_path, index=False)
print(f"Cleaned dataset saved to {output_path}")
print(f"Rows remaining after cleaning: {len(df)}")
