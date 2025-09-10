import subprocess
import sys
from pathlib import Path

def serve_mlflow_model(model_path: str = "../mlruns/models/rfr_model_v1",
                       port: int = 5000,
                       no_conda: bool = True) -> None:
    """
    Serve an MLflow model as a REST API endpoint.

    Parameters
    ----------
    model_path : str
        Path to the MLflow model folder (artifact path).
    port : int
        Port to serve the model on.
    no_conda : bool
        If True, MLflow will not attempt to create a Conda environment.
    """
    model_path = Path(model_path).resolve()
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")
    
    # Build the command
    cmd = [
        sys.executable, "-m", "mlflow", "models", "serve",
        "-m", str(model_path),
        "-p", str(port)
    ]
    
    if no_conda:
        cmd.append("--no-conda")
    
    print(f"Serving MLflow model from {model_path} on port {port} ...")
    print("Command:", " ".join(cmd))
    
    # Run the server
    subprocess.Popen(cmd)

if __name__ == "__main__":
    serve_mlflow_model()
