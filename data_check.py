import pandas as pd

# ---- LOAD DATA ----
df = pd.read_parquet("yellow_tripdata_2025-09.parquet")

# Show all columns fully
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

print("========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== BASIC INFO ==========")
print(df.info())

print("\n========== SUMMARY STATISTICS ==========")
print(df.describe(include='all', datetime_is_numeric=True))

# ---- MISSING VALUES ----
print("\n========== MISSING VALUES ==========")
print(df.isna().sum())

# ---- NEGATIVE NUMBERS ----
numeric_columns = df.select_dtypes(include=['float', 'int']).columns
neg_counts = (df[numeric_columns] < 0).sum()
print("\n========== NEGATIVE VALUE COUNTS ==========")
print(neg_counts[neg_counts > 0])

# ---- ZERO / IMPOSSIBLE VALUES ----
print("\n========== ZERO / IMPOSSIBLE VALUES ==========")
print("Trip distance == 0:", (df['trip_distance'] == 0).sum())
print("Fare amount == 0:", (df['fare_amount'] == 0).sum())
print("Total amount == 0:", (df['total_amount'] == 0).sum())
print("Passenger count == 0:", (df['passenger_count'] == 0).sum())

# ---- DATETIME CHECK ----
print("\n========== DATETIME PARSING CHECK ==========")
invalid_pickup = df['tpep_pickup_datetime'].isna().sum()
invalid_dropoff = df['tpep_dropoff_datetime'].isna().sum()
print("tpep_pickup_datetime: invalid datetime values =", invalid_pickup)
print("tpep_dropoff_datetime: invalid datetime values =", invalid_dropoff)

# dropoff before pickup
bad_times = (df['tpep_dropoff_datetime'] < df['tpep_pickup_datetime']).sum()
print("Dropoff before pickup:", bad_times)

# ---- DUPLICATES ----
print("\n========== DUPLICATES ==========")
print("Duplicate rows:", df.duplicated().sum())

# ---- TEXT NORMALIZATION ----
print("\n========== TEXT NORMALIZATION PROBLEMS ==========")
flag_col = df['store_and_fwd_flag'].astype(str)
leading = flag_col.str.startswith(' ').sum()
trailing = flag_col.str.endswith(' ').sum()
uppercase = (flag_col != flag_col.str.upper()).sum()

print(f"store_and_fwd_flag: leading={leading}, trailing={trailing}, uppercase={uppercase}")

# ---- CATEGORY CHECKS ----
print("\n========== CATEGORY VALUE CHECKS ==========")
valid_flags = {'N', 'Y'}
wrong_flags = (~df['store_and_fwd_flag'].isin(valid_flags)).sum()

valid_payments = {0,1,2,3,4}
wrong_payment = (~df['payment_type'].isin(valid_payments)).sum()

print("store_and_fwd_flag: unexpected values count =", wrong_flags)
print("payment_type: unexpected values count =", wrong_payment)

# ---- OUTLIER CHECKS ----
print("\n========== OUTLIER FLAGGING (IQR METHOD) ==========")
outlier_report = {}
for col in ['trip_distance', 'fare_amount', 'total_amount']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    outliers = df[(df[col] < low) | (df[col] > high)]
    outlier_report[col] = len(outliers)

for col, count in outlier_report.items():
    print(f"{col}: possible outliers = {count}")

# ---- TIME ORDER CHECK ----
print("\n========== TIME ORDER CHECK ==========")
is_sorted = df['tpep_pickup_datetime'].is_monotonic_increasing
print("Dataset sorted by pickup time?:", is_sorted)

if not is_sorted:
    print("First few rows out of time order:")
    temp = df[['tpep_pickup_datetime']].copy()
    temp['prev'] = temp['tpep_pickup_datetime'].shift(1)
    bad_order_rows = df[temp['tpep_pickup_datetime'] < temp['prev']]
    print(bad_order_rows.head())
