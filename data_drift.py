import pandas as pd
import h2o
from h2o.estimators import H2ODeepLearningEstimator, H2OStackedEnsembleEstimator
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, RegressionPreset
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
TARGET = "total_amount"

TRAIN_PATH = r"C:\Users\Jaival Singh\Downloads\Mlops Project\splits\train_small.parquet"
TEST_PATH  = r"C:\Users\Jaival Singh\Downloads\Mlops Project\splits\test_small.parquet"

MODELS = {
    "DeepLearning": r"C:\Users\Jaival Singh\Downloads\Mlops Project\manual_models\downloaded_artifacts\DeepLearning\DeepLearning_model_python_1765241736141_1",
    "StackedEnsemble_All": r"C:\Users\Jaival Singh\Downloads\Mlops Project\manual_models\downloaded_artifacts\Stack_all\StackedEnsemble_model_python_1765241736141_9",
    "StackedEnsemble_Best": r"C:\Users\Jaival Singh\Downloads\Mlops Project\manual_models\downloaded_artifacts\Stack_best_of_family\StackedEnsemble_model_python_1765241736141_13"
}

OUTPUT_DIR = Path("drift_reports")
OUTPUT_DIR.mkdir(exist_ok=True)

# -----------------------------
# START H2O
# -----------------------------
h2o.init(max_mem_size="4G")  

# -----------------------------
# LOAD DATA
# -----------------------------
reference_df = pd.read_parquet(TRAIN_PATH)
production_df = pd.read_parquet(TEST_PATH)

X_ref = reference_df.drop(columns=[TARGET])
y_ref = reference_df[TARGET]

X_prod = production_df.drop(columns=[TARGET])
y_prod = production_df[TARGET]

# -----------------------------
# RUN DRIFT PER MODEL
# -----------------------------
for model_name, model_path in MODELS.items():
    print(f"\nRunning drift analysis for: {model_name}")

    # Load H2O model
    model = h2o.load_model(model_path)

    # Convert Pandas to H2OFrame
    X_ref_h2o = h2o.H2OFrame(X_ref)
    X_prod_h2o = h2o.H2OFrame(X_prod)

    # -------------------------
    # Predictions
    # -------------------------
    ref_preds = model.predict(X_ref_h2o).as_data_frame().values.flatten()
    prod_preds = model.predict(X_prod_h2o).as_data_frame().values.flatten()

    # Make copies of DataFrames for Evidently
    ref_df_copy = reference_df.copy()
    prod_df_copy = production_df.copy()

    ref_df_copy["prediction"] = ref_preds
    prod_df_copy["prediction"] = prod_preds

    # Rename target column for Evidently
    ref_df_copy.rename(columns={TARGET: "target"}, inplace=True)
    prod_df_copy.rename(columns={TARGET: "target"}, inplace=True)

    # -------------------------
    # DATA DRIFT REPORT
    # -------------------------
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=ref_df_copy, current_data=prod_df_copy)
    data_drift_path = OUTPUT_DIR / f"{model_name}_data_drift.html"
    data_drift_report.save_html(data_drift_path)

    # -------------------------
    # PERFORMANCE DRIFT REPORT
    # -------------------------
    perf_drift_report = Report(metrics=[RegressionPreset()])
    perf_drift_report.run(reference_data=ref_df_copy, current_data=prod_df_copy)
    perf_drift_path = OUTPUT_DIR / f"{model_name}_performance_drift.html"
    perf_drift_report.save_html(perf_drift_path)

    # -------------------------
    # PRINT BASIC METRICS
    # -------------------------
    rmse = np.sqrt(mean_squared_error(y_prod, prod_preds))
    mae = mean_absolute_error(y_prod, prod_preds)
    r2 = r2_score(y_prod, prod_preds)

    print(f"{model_name} metrics on simulated production data:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  R²:   {r2:.4f}")
    print(f"Saved HTML reports for {model_name} in {OUTPUT_DIR}")

print("\nDrift analysis completed.")

# -----------------------------
# SHUTDOWN H2O
# -----------------------------
h2o.shutdown(prompt=False)
