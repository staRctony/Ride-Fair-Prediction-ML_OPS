import h2o
from h2o.automl import H2OAutoML
import pandas as pd
import os

shrink_frac = 0.10
print(f"Shrink fraction fixed at: {shrink_frac}")

train_path = "splits/train.parquet"
test_path = "splits/test.parquet"
valid_path = "splits/validate.parquet"

def shrink_parquet(input_path, output_path, frac):
    print(f"\nLoading {input_path} ...")
    df = pd.read_parquet(input_path)
    print(f"Original rows = {len(df):,}")

    df_small = df.sample(frac=frac, random_state=42)
    df_small.to_parquet(output_path, index=False)

    print(f"Saved smaller file: {output_path} (rows = {len(df_small):,})")
    return df_small

train_small = "splits/train_small.parquet"
test_small = "splits/test_small.parquet"
valid_small = "splits/validate_small.parquet"

train_df = shrink_parquet(train_path, train_small, shrink_frac)
shrink_parquet(test_path, test_small, shrink_frac)
shrink_parquet(valid_path, valid_small, shrink_frac)

h2o.init(max_mem_size="16G")

models_dir = "h2o_models"
os.makedirs(models_dir, exist_ok=True)
leaderboard_csv = os.path.join(models_dir, "h2o_automl_leaderboard.csv")
top_models_file = os.path.join(models_dir, "top3_model_paths.txt")

if os.path.exists(top_models_file):

    with open(top_models_file, "r") as f:
        model_paths = [line.strip() for line in f.readlines()]

    top_models = [h2o.load_model(p) for p in model_paths]
    print("Top 3 models loaded from saved files.")

    lb_df = pd.read_csv(leaderboard_csv)
    print("Leaderboard loaded from CSV")

else:
    print("\nLoading smaller train dataset into H2OFrame...")
    train_hf = h2o.H2OFrame(train_df)

    target = "total_amount"
    features = [col for col in train_df.columns if col != target]

    aml = H2OAutoML(
        max_models=20,
        seed=42,
        nfolds=5,
        sort_metric="RMSE"
    )

    aml.train(x=features, y=target, training_frame=train_hf)

    lb = aml.leaderboard
    lb_df = h2o.as_list(lb)
    lb_df.to_csv(leaderboard_csv, index=False)
    print(f"Leaderboard saved to {leaderboard_csv}")

    top_models = []
    model_paths = []

    for model_id in lb_df["model_id"][:3]:
        model = h2o.get_model(model_id)
        path = h2o.save_model(model, path=models_dir, force=True)
        top_models.append(model)
        model_paths.append(path)

    with open(top_models_file, "w") as f:
        f.write("\n".join(model_paths))

    print(f"Top 3 models saved in {models_dir}")
