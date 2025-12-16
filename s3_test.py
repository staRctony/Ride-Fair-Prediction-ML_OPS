import mlflow
from dotenv import load_dotenv
import os

# Load your .env with AWS keys
load_dotenv()

# Make sure boto3/mlflow can see the credentials
os.environ["AWS_ACCESS_KEY_ID"]     = os.getenv("AWS_ACCESS_KEY_ID")
os.environ["AWS_SECRET_ACCESS_KEY"] = os.getenv("AWS_SECRET_ACCESS_KEY")
os.environ["AWS_DEFAULT_REGION"]    = os.getenv("AWS_DEFAULT_REGION", "us-east-2")

# Point to your remote MLflow server (via SSH tunnel)
mlflow.set_tracking_uri("http://localhost:5000")

print("Uploading proof to S3...")

with mlflow.start_run(run_name="S3_PROOF_2025"):
    mlflow.log_metric("test", 999)
    
    
    with open("proof.txt", "w", encoding="utf-8") as f:
        f.write("S3 ARTIFACTS ARE WORKING - JAIVAL - DEC 2025")
    
    mlflow.log_artifact("proof.txt")
    print("DONE! Check your S3 bucket now -> proof.txt must appear in 5-10 seconds")

print("Open: https://s3.console.aws.amazon.com/s3/buckets/mlflow-artifacts-jaival")