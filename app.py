"""Aplicação Streamlit"""
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
from datetime import timedelta, datetime
import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import dagshub
import warnings

# warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Variáveis de ambiente
load_dotenv()
DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DAGSHUB_REPO = os.getenv("REPO_NAME")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
MODEL_URI = os.getenv("MLFLOW_MODEL_URI")


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
    
    /* Model status indicators */
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
CSV_PATH = "data/processed/amazon_delivery_processed.csv"

@st.cache_resource
def setup_mlflow():
    """Configura a conexão com o MLflow Tracking Server (DAGsHub)."""
    if 'MLFLOW_HOME' in os.environ:
        del os.environ['MLFLOW_HOME']
    if 'MLFLOW_TRACKING_URI' in os.environ:
        del os.environ['MLFLOW_TRACKING_URI']


    if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO and MLFLOW_TRACKING_URI:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            dagshub.init(
                repo_owner=DAGSHUB_USERNAME, 
                repo_name=DAGSHUB_REPO,
                mlflow=True
            )
            
            print(f"MLflow Tracking URI configurado para: {MLFLOW_TRACKING_URI}")
            return True
        except Exception as e:
            print(f"Aviso: Erro ao inicializar DAGsHub/MLflow: {e}")
            return False
    else:
        print("Aviso: Variáveis de ambiente DAGsHub/MLflow não configuradas. Pulando inicialização.")
        return False

@st.cache_resource
def load_model():
    """
    Carrega o modelo em produção a partir das credenciais de conexão com o Dagshub/Mlflow.
    """
    try:
        model_uri_clean = MODEL_URI.strip().strip('"').strip("'")
        loaded_model = mlflow.pyfunc.load_model(model_uri_clean)
        print(f"✅ Modelo '{model_uri_clean}' carregado com sucesso.")
        return loaded_model
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None



# funções de operação

def generate_random_input(df: pd.DataFrame, n_samples: int = 1, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Gera valores aleatórios para as colunas de features, 
    usando as estatísticas e distribuições do DataFrame de entrada.
    """
    rng = np.random.default_rng(seed=random_state)

    removed_columns = ["month", "holiday", "pickup_duration", "order_cicle_time", "Traffic",
                   "weekend", "is_sunny_weather", "Delivery_Time", "Unnamed: 0", 'day_sin', 'day_cos', 'month_sin', 'month_cos']

    features_df = df.drop(columns=[c for c in removed_columns if c in df.columns], errors='ignore')
    random_data: Dict[str, Any] = {}

    # --- Contínuas ---
    distance_raw = rng.normal(9.72, 5.59, size=n_samples)
    distance_clipped = np.round(np.abs(distance_raw),2)
    random_data['delivery_distance'] = distance_clipped.item()
    random_data['Agent_Rating'] = np.round(rng.normal(4.63, 0.32, size=n_samples).item(),2)
    age_clipped = np.clip(rng.normal(27.0, 5.76, size=n_samples), 18, 60).item()
    random_data['Agent_Age'] = int(age_clipped)
    

    # --- Categóricas ---
    random_data['Traffic']= rng.choice(['High', 'Jam', 'Low', 'Medium'], size=n_samples).item()
    random_data['Vehicle']= rng.choice(['motorcycle', 'van', 'scooter'], size=n_samples).item()
    random_data['Weather']= rng.choice(['Sunny', 'Stormy', 'Sandstorms', 'Cloudy', 'Fog', 'Windy'], size=n_samples).item()
    random_data['Area'] = rng.choice(['Urban', 'Metropolitian', 'Other', 'Semi-Urban'], size=n_samples).item()
    

    # --- Binária ---
    random_data['is_grocery'] = rng.choice([0, 1], size=n_samples).item()
    random_data['jam_or_high_traffic'] =  1 if random_data['Traffic'] in ['High', 'Jam'] else 0
    random_data['dist_gte_10'] = 1 if random_data['delivery_distance']>=10.0 else 0
    

    # --- Final ---
    random_df = pd.DataFrame(random_data, index=range(n_samples))
    for c in features_df.columns:
        if c not in random_df: random_df[c] = 0
    return random_df[features_df.columns]



def predict_model(model, input_df: pd.DataFrame) -> pd.Series:
    """
    Faz predições usando o modelo MLflow carregado localmente.

    Args:
        model: Modelo MLflow (pyfunc) carregado.
        input_df: DataFrame contendo as features de entrada para predição.

    Returns:
        Transformação exponencial dos valores preditos para cada linha em input_df (em minutos).
    """

    predictions = np.exp(model.predict(input_df))
    return pd.Series(predictions, index=input_df.index)



def load_data() -> Optional[pd.DataFrame]:
    """
    Carrega o DataFrame processado do caminho CSV especificado.
    """
    try:
        df = pd.read_csv(CSV_PATH)
        # Remove 'Unnamed: 0' column if it exists
        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)
        return df
    except FileNotFoundError:
        st.error(f"❌ Arquivo CSV não encontrado em: {CSV_PATH}")
        return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar CSV: {str(e)}")
        return None


def main():
    """Função principal da aplicação."""
    # Title and subtitle
    st.markdown('<h1 class="main-title">🚚 Order Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Obtenha predições instantâneas do tempo de entrega para seus pedidos</p>', unsafe_allow_html=True)

    # Initialize session state
    if 'random_data' not in st.session_state:
        st.session_state.random_data = None
    if 'prediction_result' not in st.session_state:
        st.session_state.prediction_result = None
    if 'df_final' not in st.session_state:
        st.session_state.df_final = load_data()
    if 'message_content' not in st.session_state:
        st.session_state.message_content = None
    if 'model' not in st.session_state:
        st.session_state.model = load_model()

    model_loaded = st.session_state.model is not None
    
    # Action buttons
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        if st.button("📋 Criar Pedido", width='stretch', type="primary"):
            if st.session_state.df_final is not None:
                with st.spinner("Gerando dados de pedido aleatórios..."):
              
                    st.session_state.random_data = generate_random_input(
                        st.session_state.df_final, 
                        n_samples=1, 
                        random_state=None # Use None para dados diferentes
                    )
                    st.session_state.prediction_result = None # Limpa a predição anterior
                    st.session_state.message_content = "order_created"
                st.success("✅ Pedido criado com sucesso!")
            else:
                st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo CSV existe.")
    
    with col2:
        if st.button("🔮 Obter Predição", width='stretch', type="secondary"):
            if not model_loaded:
                st.error("❌ O modelo MLflow não foi carregado. Verifique as variáveis de ambiente e o status do servidor MLflow Tracking.")
            elif st.session_state.random_data is None:
                st.warning("⚠️ Por favor, crie um pedido primeiro!")
            else:
                with st.spinner("Obtendo predição do modelo em produção..."):
                    try:
                        # Chamada direta à função de predição local
                        predictions = predict_model(st.session_state.model, st.session_state.random_data)
                        
                        if not predictions.empty:
                            prediction = predictions.iloc[0]
                            st.session_state.prediction_result = prediction
                            st.session_state.message_content = "prediction_ready"
                            st.success("🎯 Predição concluída!")
                        else:
                             st.error("❌ A predição retornou um resultado vazio.")
                             st.session_state.prediction_result = None
                             
                    except Exception as e:
                        st.error(f"❌ Erro ao fazer a predição: {str(e)}")
                        st.session_state.prediction_result = None


    # Message box
    with st.container():
        if st.session_state.message_content is None:
            # Empty message box
            st.markdown('<div class="message-box empty">Clique "Criar Pedido" para gerar dados e depois "Obter Predição"</div>', unsafe_allow_html=True)
        
        elif st.session_state.message_content == "order_created":
            # Show order data
            st.markdown('<div class="message-box">', unsafe_allow_html=True)
            st.markdown('<div class="card-title">📊 Dados do Pedido Gerado</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show dataframe right after the message box
            st.dataframe(st.session_state.random_data, width='stretch', hide_index=True)
        
        elif st.session_state.message_content == "prediction_ready":
            # Show prediction result
            try:
                pred_value = int(np.array(st.session_state.prediction_result).flat[0])
                minutos = timedelta(minutes=pred_value)
                hora_atual = datetime.now()
                hora_chegada = hora_atual + minutos
                formatted_pred = hora_chegada.strftime('%H:%M')
                formatted_min = str(pred_value)

            except:
                print(f"Erro no cálculo da hora de chegada: {e}")
                formatted_pred = "N/A"
                pred_value = 0.0 # 
            
            st.markdown(f'''
            <div class="message-box">
                <div class="card-title">🎯 Resultado da Predição</div>
                <div class="prediction-result">O pedido chegará às {formatted_pred}h.<br>
                Tempo total: {formatted_min} minutos. ⏱️</div>
            </div>
            ''', unsafe_allow_html=True)
    
    # Model Status Section (simplificado para refletir o carregamento do modelo)
    st.markdown("---")
    st.markdown("### ⚙️ Status do Modelo MLflow")
    
    status_class = "status-online" if model_loaded else "status-offline"
    status_text = "🟢 Carregado" if model_loaded else "🔴 Offline"
    
    st.markdown('<div class="server-status">', unsafe_allow_html=True)
    st.markdown(f'<div class="status-dot {status_class}"></div>', unsafe_allow_html=True)
    st.write(f"**Status do Modelo (MLflow):** {status_text}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if not model_loaded:
        st.warning(f"""
        **Aviso:** Não foi possível carregar o modelo. 
                   Verifique as credenciais do modelo em produção.
  
        """)
        if st.button("🔄 Tentar Recarregar o Modelo"):
            st.cache_resource.clear()
            st.session_state.model = load_model()
            st.rerun()


if __name__ == "__main__":
    main()
