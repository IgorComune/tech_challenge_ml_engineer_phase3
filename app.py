import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import requests
import json
import time
import subprocess
import os
from typing import Optional, Dict, Any
import sys
import socket

# Page configuration
st.set_page_config(
    page_title="Order Prediction",
    page_icon="🚚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom title styling */
    .main-title {
        font-family: 'Inter', sans-serif;
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.2rem;
        font-weight: 300;
        text-align: center;
        color: #6B7280;
        margin-bottom: 3rem;
    }
    
    /* Button container */
    .button-container {
        display: flex;
        justify-content: center;
        gap: 2rem;
        margin: 3rem 0;
    }
    
    /* Custom button styling */
    .stButton > button {
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1.1rem;
        padding: 0.75rem 2.5rem;
        border-radius: 12px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Message box styling */
    .message-box {
        min-height: 200px;
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid #E5E7EB;
        font-family: 'Inter', sans-serif;
    }
    
    .message-box.empty {
        display: flex;
        align-items: center;
        justify-content: center;
        color: #9CA3AF;
        font-style: italic;
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        border: 2px dashed #D1D5DB;
    }
    
    /* Card title styling */
    .card-title {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Prediction result styling */
    .prediction-result {
        font-family: 'Inter', sans-serif;
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        color: #059669;
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #10b981;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
    }
    
    /* Server status indicators */
    .server-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 1rem 0;
    }
    
    .status-dot {
        width: 12px;
        height: 12px;
        border-radius: 50%;
    }
    
    .status-online {
        background-color: #10b981;
        animation: pulse 2s infinite;
    }
    
    .status-offline {
        background-color: #ef4444;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Server initialization section */
    .server-init-card {
        background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border-left: 4px solid #3b82f6;
    }
    
    .warning-card {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border-left: 4px solid #f59e0b;
    }
    
    .step-number {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 600;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Constants
CSV_PATH = "data/processed/dataframe_final.csv"
MODEL_PATH = "../mlruns/models/rfr_model_v1"
MLFLOW_URL = "http://127.0.0.1:5000"
API_URL = "http://127.0.0.1:5000/invocations"

def generate_random_input(
    df: pd.DataFrame,
    n_samples: int = 1,
    random_state: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate random values within the [min, max] range of each numeric column in the DataFrame.
    
    Useful for creating artificial data to test a model with .predict().

    Args:
        df: DataFrame with only numeric columns.
        n_samples: Number of rows to generate. Defaults to 1.
        random_state: Seed for reproducibility. Defaults to None.

    Returns:
        DataFrame with n_samples rows and the same columns as df,
        excluding 'Delivery_Time' if present.
    """
    rng = np.random.default_rng(seed=random_state)
    random_data = {}

    for col in df.select_dtypes(include=np.number).columns:
        col_min = df[col].min()
        col_max = df[col].max()
        random_data[col] = rng.uniform(low=col_min, high=col_max, size=n_samples)

    random_df = pd.DataFrame(random_data)

    # Remove 'Delivery_Time' if it exists
    if 'Delivery_Time' in random_df.columns:
        random_df = random_df.drop(columns='Delivery_Time')

    return random_df


def load_model(model_path: str):
    """
    Load a saved sklearn model from a local path or MLflow Tracking Server.

    Args:
        model_path: Path to the saved model.

    Returns:
        Loaded sklearn model.
    """
    model = mlflow.sklearn.load_model(model_path)
    return model


def predict_model(model, input_df: pd.DataFrame) -> pd.Series:
    """
    Make predictions using a loaded sklearn model.

    Args:
        model: Loaded sklearn model.
        input_df: DataFrame containing input features for prediction.

    Returns:
        Predicted values for each row in input_df.
    """
    predictions = model.predict(input_df)
    return pd.Series(predictions, index=input_df.index)


def send_random_input_to_api(
    df: pd.DataFrame,
    url: str = API_URL,
    n_samples: int = 1,
    random_state: Optional[int] = None,
    wait_server: int = 30
) -> Dict[str, Any]:
    """
    Generate random input data based on the numeric ranges of a DataFrame,
    convert it to JSON in MLflow 2.x format, and send it to a REST API endpoint.

    Args:
        df: DataFrame with numeric columns to base the random data on.
        url: API endpoint to send the request. Defaults to local host.
        n_samples: Number of random rows to generate. Defaults to 1.
        random_state: Seed for reproducibility. Defaults to None.
        wait_server: Max seconds to wait for server to be ready.

    Returns:
        JSON response from the API.
    """
    # Generate random input
    x_fake = generate_random_input(df, n_samples=n_samples, random_state=random_state)

    # Convert to JSON payload (MLflow 2.x expects {"dataframe_split": ...})
    payload = {"dataframe_split": json.loads(x_fake.to_json(orient="split"))}

    # Wait until server is ready
    start = time.time()
    while time.time() - start < wait_server:
        try:
            r = requests.get(url.replace("/invocations", "/ping"))
            if r.status_code in (200, 404, 405):  # Server is ready
                break
        except requests.exceptions.ConnectionError:
            time.sleep(1)
    else:
        print("❌ Server did not start within the wait time.")
        return {"error": "Server not ready"}

    # Send POST request
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
        return {"error": str(e)}


def check_server_status(url: str, timeout: int = 5) -> bool:
    """
    Check if a server is running and responding.
    
    Args:
        url: Server URL to check.
        timeout: Request timeout in seconds.
        
    Returns:
        True if server is responding, False otherwise.
    """
    try:
        response = requests.get(url, timeout=timeout)
        return response.status_code in [200, 404, 405]  # Server is responding
    except requests.exceptions.RequestException:
        return False


def wait_for_port(host: str, port: int, timeout: int = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError:
            time.sleep(1)
    return False

def start_server(script_name: str, host: str, port: int) -> bool:
    try:
        process = subprocess.Popen(
            [sys.executable, script_name],
            cwd="src",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        # st.info(f"Iniciando {script_name} (PID={process.pid})...")
        
        if wait_for_port(host, port):
            # st.success(f"{script_name} está pronto em {host}:{port}")
            return True
        # else:
        #     st.error(f"{script_name} não respondeu em {host}:{port} dentro do timeout")
        #     return False
    except Exception as e:
        st.error(f"Erro ao iniciar {script_name}: {e}")
        return False

# def start_mlflow_server() -> bool:
#     """
#     Start MLflow server using the mlflow_server.py script.
    
#     Returns:
#         True if server start command was executed successfully, False otherwise.
#     """
#     try:
#         # Change to src directory and run the script
#         subprocess.Popen(
#             ["python", "mlflow_server.py"],
#             cwd="src",
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#         return True
#     except Exception as e:
#         st.error(f"Error starting MLflow server: {str(e)}")
#         return False


# def start_api_server() -> bool:
#     """
#     Start API server using the api_server.py script.
    
#     Returns:
#         True if server start command was executed successfully, False otherwise.
#     """
#     try:
#         # Change to src directory and run the script
#         subprocess.Popen(
#             ["python", "api_server.py"],
#             cwd="src",
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE
#         )
#         return True
#     except Exception as e:
#         st.error(f"Error starting API server: {str(e)}")
#         return False


def load_data() -> Optional[pd.DataFrame]:
    """
    Load the processed dataframe from the specified CSV path.
    
    Returns:
        Loaded DataFrame or None if error occurred.
    """
    try:
        df = pd.read_csv(CSV_PATH)
        # Remove 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
    except FileNotFoundError:
        st.error(f"❌ CSV file not found at: {CSV_PATH}")
        return None
    except Exception as e:
        st.error(f"❌ Error loading CSV: {str(e)}")
        return None


def main():
    """Main application function."""
    # Title and subtitle
    st.markdown('<h1 class="main-title">🚚 Order Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Get instant delivery time predictions for your orders</p>', unsafe_allow_html=True)
    start_server("mlflow_server.py", host="127.0.0.1", port=5000)
    start_server("api_server.py", host="127.0.0.1", port=8000)


    # Initialize session state
    if 'random_data' not in st.session_state:
        st.session_state.random_data = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'df_final' not in st.session_state:
        st.session_state.df_final = load_data()
    if 'message_content' not in st.session_state:
        st.session_state.message_content = None

    # Check server status
    mlflow_online = check_server_status(f"{MLFLOW_URL}/ping")
    api_online = check_server_status(API_URL)
    
    # Action buttons
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if st.button("📋 Create Order", use_container_width=True, type="primary"):
            if st.session_state.df_final is not None:
                with st.spinner("Generating random order data..."):
                    st.session_state.random_data = generate_random_input(
                        st.session_state.df_final, 
                        n_samples=1, 
                        random_state=42
                    )
                    st.session_state.message_content = "order_created"
                st.success("✅ Order created successfully!")
            else:
                st.error("❌ Could not load data. Please check if the CSV file exists.")
    
    with col2:
        if st.button("🔮 Get Prediction", use_container_width=True, type="secondary"):
            if not (mlflow_online and api_online):
                st.error("❌ Servers are not online. Please start MLflow and API servers first.")
            elif st.session_state.random_data is not None:
                with st.spinner("Getting prediction from API..."):
                    response = send_random_input_to_api(
                        st.session_state.df_final,
                        n_samples=1
                    )
                    
                    if 'error' not in response:
                        # Extract prediction from response
                        if isinstance(response, list) and len(response) > 0:
                            prediction = response[0]
                        elif isinstance(response, dict) and 'predictions' in response:
                            prediction = response['predictions'][0]
                        else:
                            prediction = response
                        
                        st.session_state.prediction_result = prediction
                        st.session_state.message_content = "prediction_ready"
                        st.success("🎯 Prediction completed!")
                    else:
                        st.error(f"❌ API Error: {response['error']}")
            else:
                st.warning("⚠️ Please create an order first!")
    
    # Message box
    with st.container():
        if st.session_state.message_content is None:
            # Empty message box
            st.markdown('<div class="message-box empty">Click "Create Order" to generate order data or check server status below</div>', unsafe_allow_html=True)
        
        elif st.session_state.message_content == "order_created":
            # Show order data
            st.markdown('<div class="message-box">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Generated Order Data</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show dataframe right after the message box
            st.dataframe(st.session_state.random_data, use_container_width=True, hide_index=True)
        
        elif st.session_state.message_content == "prediction_ready":
            # Show prediction result
            # Format prediction value
            try:
                pred_value = float(st.session_state.prediction_result)
                formatted_pred = f"{pred_value:.2f}"
            except:
                formatted_pred = str(st.session_state.prediction_result)
            
            st.markdown(f'''
            <div class="message-box">
                <div class="card-title">🎯 Prediction Result</div>
                <div class="prediction-result">Your order will arrive in: {formatted_pred} minutes ⏱️</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Server Status Section (moved to bottom)
    st.markdown("---")
    st.markdown("### 🖥️ Server Status")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="server-status">', unsafe_allow_html=True)
        status_class = "status-online" if mlflow_online else "status-offline"
        status_text = "🟢 Online" if mlflow_online else "🔴 Offline"
        st.markdown(f'<div class="status-dot {status_class}"></div>', unsafe_allow_html=True)
        st.write(f"**MLflow Server:** {status_text}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="server-status">', unsafe_allow_html=True)
        status_class = "status-online" if api_online else "status-offline"
        status_text = "🟢 Online" if api_online else "🔴 Offline"
        st.markdown(f'<div class="status-dot {status_class}"></div>', unsafe_allow_html=True)
        st.write(f"**API Server:** {status_text}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show server initialization help only if servers are offline
    if not (mlflow_online and api_online):
        with st.expander("🚀 Need help starting servers?", expanded=False):
            st.markdown("**Start servers manually in your terminal:**")
            st.code("""
# Terminal 1: Start MLflow Server
cd your-project-directory
python src/mlflow_server.py

# Terminal 2: Start API Server (wait for MLflow to be ready first)
cd your-project-directory  
python src/api_server.py
            """, language="bash")
            
            if st.button("🔄 Refresh Server Status"):
                st.rerun()


if __name__ == "__main__":
    main()