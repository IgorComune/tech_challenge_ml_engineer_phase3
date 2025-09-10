import subprocess
import sys
from pathlib import Path
from typing import Optional

def run_mlflow_server(host: str = "127.0.0.1",
                      port: int = 8080,
                      backend_store_uri: str = "sqlite:///../mlruns/mlflow.db",
                      artifact_root: Optional[str] = "../mlruns") -> None:
    """
    Start an MLflow Tracking Server.

    Parameters
    ----------
    host : str
        Host to bind the MLflow server to.
    port : int
        Port to serve the MLflow server.
    backend_store_uri : str
        URI for the backend store (e.g., sqlite:///mlflow.db).
    artifact_root : str, optional
        Directory where MLflow artifacts (models, plots, etc.) will be stored.
    """
    artifact_root_path = Path(artifact_root).resolve()
    artifact_root_path.mkdir(parents=True, exist_ok=True)

    # Build the MLflow server command
    cmd = [
        sys.executable, "-m", "mlflow", "server",
        "--host", host,
        "--port", str(port),
        "--backend-store-uri", backend_store_uri,
        "--default-artifact-root", str(artifact_root_path)
    ]

    print(f"Starting MLflow Tracking Server at http://{host}:{port} ...")
    print("Command:", " ".join(cmd))

    # Run the server
    subprocess.run(cmd)

if __name__ == "__main__":
    run_mlflow_server()
