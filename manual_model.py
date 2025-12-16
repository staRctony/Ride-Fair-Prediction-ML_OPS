import h2o
from h2o.automl import H2OAutoML
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
import pandas as pd
import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os

# --------------------------------------
# Initialize H2O
# --------------------------------------
h2o.init(max_mem_size="16G")

# --------------------------------------
# Load parquet datasets
# --------------------------------------
train_df = pd.read_parquet("splits/train_small.parquet")
valid_df = pd.read_parquet("splits/validate_small.parquet")
test_df = pd.read_parquet("splits/train_small.parquet")

target = "total_amount"
features = [c for c in train_df.columns if c != target]

train_hf = h2o.H2OFrame(train_df)
valid_hf = h2o.H2OFrame(valid_df)
test_hf = h2o.H2OFrame(test_df)

# --------------------------------------
# MLflow experiment
# --------------------------------------
mlflow.set_experiment("H2O_Top3_Models")

# --------------------------------------
# Evaluation function
# --------------------------------------
def evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# Convert H2O target to numpy array
y_train = train_hf[target].as_data_frame().values.ravel()
y_valid = valid_hf[target].as_data_frame().values.ravel()
y_test = test_hf[target].as_data_frame().values.ravel()

# --------------------------------------
# 1) Deep Learning (with CV for stacking)
# --------------------------------------
with mlflow.start_run(run_name="DeepLearning_1"):
    dl = H2ODeepLearningEstimator(
        hidden=[50, 50],
        epochs=50,
        seed=42,
        nfolds=5,  # enable cross-validation
        keep_cross_validation_predictions=True  # required for stacked ensembles
    )
    dl.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)

    pred_train = dl.predict(train_hf).as_data_frame().values.ravel()
    pred_valid = dl.predict(valid_hf).as_data_frame().values.ravel()
    pred_test = dl.predict(test_hf).as_data_frame().values.ravel()

    metrics = {}
    for name, y_true, y_pred in [("Train", y_train, pred_train),
                                 ("Validation", y_valid, pred_valid),
                                 ("Test", y_test, pred_test)]:
        rmse, mae, r2 = evaluate(y_true, y_pred)
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_R2"] = r2

    mlflow.log_metrics(metrics)
    mlflow.h2o.log_model(dl, "DeepLearning_1_model")
    print("Deep Learning metrics:", metrics)

# --------------------------------------
# 2) Stacked Ensemble: BestOfFamily
# --------------------------------------
with mlflow.start_run(run_name="StackedEnsemble_BestOfFamily"):
    se_best = H2OStackedEnsembleEstimator(base_models=[dl.model_id])
    se_best.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)

    pred_train = se_best.predict(train_hf).as_data_frame().values.ravel()
    pred_valid = se_best.predict(valid_hf).as_data_frame().values.ravel()
    pred_test = se_best.predict(test_hf).as_data_frame().values.ravel()

    metrics = {}
    for name, y_true, y_pred in [("Train", y_train, pred_train),
                                 ("Validation", y_valid, pred_valid),
                                 ("Test", y_test, pred_test)]:
        rmse, mae, r2 = evaluate(y_true, y_pred)
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_R2"] = r2

    mlflow.log_metrics(metrics)
    mlflow.h2o.log_model(se_best, "StackedEnsemble_BestOfFamily_model")
    print("Stacked Ensemble BestOfFamily metrics:", metrics)

# --------------------------------------
# 3) Stacked Ensemble: AllModels (retrain top 3 in-session)
# --------------------------------------
# Create multiple deep learning models with CV
dl1 = H2ODeepLearningEstimator(hidden=[50, 50], epochs=50, seed=42,
                               nfolds=5, keep_cross_validation_predictions=True)
dl2 = H2ODeepLearningEstimator(hidden=[100, 50], epochs=50, seed=123,
                               nfolds=5, keep_cross_validation_predictions=True)
dl3 = H2ODeepLearningEstimator(hidden=[50, 100], epochs=50, seed=999,
                               nfolds=5, keep_cross_validation_predictions=True)

# Train each
for dl_model in [dl1, dl2, dl3]:
    dl_model.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)

# Build final stacked ensemble using CV-enabled models
with mlflow.start_run(run_name="StackedEnsemble_AllModels"):
    se_all = H2OStackedEnsembleEstimator(base_models=[dl1.model_id, dl2.model_id, dl3.model_id])
    se_all.train(x=features, y=target, training_frame=train_hf, validation_frame=valid_hf)

    pred_train = se_all.predict(train_hf).as_data_frame().values.ravel()
    pred_valid = se_all.predict(valid_hf).as_data_frame().values.ravel()
    pred_test = se_all.predict(test_hf).as_data_frame().values.ravel()

    metrics = {}
    for name, y_true, y_pred in [("Train", y_train, pred_train),
                                 ("Validation", y_valid, pred_valid),
                                 ("Test", y_test, pred_test)]:
        rmse, mae, r2 = evaluate(y_true, y_pred)
        metrics[f"{name}_RMSE"] = rmse
        metrics[f"{name}_MAE"] = mae
        metrics[f"{name}_R2"] = r2

    mlflow.log_metrics(metrics)
    mlflow.h2o.log_model(se_all, "StackedEnsemble_AllModels_model")
    print("Stacked Ensemble AllModels metrics:", metrics)
