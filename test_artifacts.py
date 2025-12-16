import mlflow
import mlflow.h2o
from dotenv import load_dotenv
import os

# ==== 1. Load AWS credentials from .env ====
load_dotenv()
os.environ["AWS_ACCESS_KEY_ID"]     = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"]    = "us-east-2"
mlflow.set_tracking_uri("http://localhost:5000")

RUN_IDS = [
    "331a22a3753b40858608429fdd298873"   # Model 3
]

MODEL_NAMES = [
    "StackedEnsemble_BestOfFamily",
    "StackedEnsemble_AllModels",
    "DeepLearning"
]

client = mlflow.tracking.MlflowClient()
loaded_models = []

print("CHECKING ALL 3 RUNS...\n" + "="*60)

for i, run_id in enumerate(RUN_IDS, 1):
    print(f"\n[{i}/3] RUN ID: {run_id}")
    print(f"     Expected: {MODEL_NAMES[i-1]}")
    
    try:
        # List artifacts
        artifacts = client.list_artifacts(run_id)
        artifact_paths = [a.path for a in artifacts]
        print(f"     Artifacts found → {artifact_paths}")
        
        # Check if model folder exists
        if "model" not in artifact_paths:
            print("     'model' folder MISSING → was not logged with mlflow.h2o.log_model()")
            continue
            
        # Load the H2O model
        model_uri = f"runs:/{run_id}/model"
        print(f"     Loading model from {model_uri} ...")
        model = mlflow.h2o.load_model(model_uri)
        
        print(f"     MODEL LOADED SUCCESSFULLY!")
        print(f"       → Type : {type(model).__name__}")
        print(f"       → ID   : {model.model_id}")
        loaded_models.append(model)
        
    except Exception as e:
        print(f"     FAILED: {e}")

print("\n" + "="*60)
if len(loaded_models) == 3:
    print("ALL 3 MODELS LOADED FROM S3 SUCCESSFULLY!")
    print("Your FastAPI app will work perfectly now.")
else:
    print(f"Only {len(loaded_models)}/3 models loaded. Fix the failed ones.")