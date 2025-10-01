"""
Script de execução de predição.
Carrega o modelo do MLflow/DagsHub e gera predições a partir de uma entrada de dicionário.
"""

import logging
import os
import sys
import warnings

import dagshub
import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.exceptions import InconsistentVersionWarning

# --- BLOCO DE CONTROLE UNIVERSAL DE WARNINGS  ---
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn.pipeline")
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.preprocessing"
)
warnings.filterwarnings("ignore", category=UserWarning)

warnings.filterwarnings(
    "ignore",
    message="The current default behavior, quantile_method='linear', will be changed",
    category=FutureWarning,
)
# Adiciona o diretório raiz ao path para importações internas (necessário para 'make')
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.config.logging_config import setup_logging
from src.modeling.predict import predict

# Configura o sistema de logging
setup_logging()
logger = logging.getLogger(__name__)

# Carrega variáveis do arquivo .env
load_dotenv()


def setup_mlops_tracking():
    """Configura o ambiente para rastreamento (MLflow/DagsHub), garantindo autenticação completa."""

    if "MLFLOW_HOME" in os.environ:
        del os.environ["MLFLOW_HOME"]
    if "MLFLOW_TRACKING_URI" in os.environ:
        del os.environ["MLFLOW_TRACKING_URI"]

    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
    DAGSHUB_REPO = os.getenv("REPO_NAME")
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

    # Verifica se as credenciais do Dagshub estão disponíveis antes de iniciar
    if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO:
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    else:
        print(
            "Aviso: Variáveis de ambiente DAGsHub não configuradas. Pulando inicialização do MLflow/DAGsHub."
        )


def define_data_input():
    """
    Define e formata o dicionário de entrada para predição.

    NOTA: As chaves devem corresponder às features esperadas pelo modelo.
    Este dicionário é convertido em um DataFrame.
    """

    # --- EXEMPLO DE ENTRADA VIA DICIONÁRIO (2 LINHAS) ---
    input_data_dict = {
        "Agent_Age": [39, 23],
        "Agent_Rating": [5.0, 4.1],
        "Weather": ["Sunny", "Fog"],
        "Vehicle": ["motorcycle", "motorcycle"],
        "Area": ["Metropolitian", "Semi-Urban"],
        "delivery_distance": [8.3, 12.4],
        "is_grocery": [0, 1],
        "jam_or_high_traffic": [1, 0],
        "dist_gte_10": [0, 1],
    }

    try:
        data_to_predict = pd.DataFrame(input_data_dict)
        logger.info(
            f"Dados de predição definidos a partir do dicionário. Shape: {data_to_predict.shape}"
        )
        return data_to_predict
    except Exception as e:
        logger.critical(
            f"ERRO CRÍTICO em define_data_input: Falha ao criar DataFrame a partir do dicionário: {e}"
        )
        return None


def run_prediction_pipeline():
    """Função principal para carregar o modelo e gerar predições."""

    setup_mlops_tracking()

    model = None

    MODEL_URI = os.getenv("MLFLOW_MODEL_URI")

    if not MODEL_URI:
        logger.critical(
            "ERRO CRÍTICO: Variável MLFLOW_MODEL_URI não encontrada. Abortando."
        )
        return

    # tratativa de carregamento do modelo
    clean_uri = MODEL_URI.strip().strip('"').strip("'")
    try:
        logger.info(f"Tentando carregar modelo com URI limpa: {clean_uri}")
        model = mlflow.pyfunc.load_model(clean_uri)
        logger.info(f"Modelo '{clean_uri}' carregado com sucesso.")

    except Exception as e:
        logger.critical(f"ERRO CRÍTICO ao carregar o modelo do MLflow: {e}")
        return

    # --- 3. ENTRADA DE DADOS E CONVERSÃO ---

    data_to_predict = define_data_input()

    if data_to_predict is None:
        logger.critical("Abordando a predição devido a falha na definição dos dados.")
        return

    # --- 4. PREDIÇÃO ---
    try:
        # np.exp é aplicado para reverter a transformação logarítmica (se o modelo foi treinado em log(y))
        predictions = np.exp(predict(data_to_predict, model=model))

        logger.info("-" * 45)
        logger.info(f"PREDIÇÃO DE ENTREGA CONCLUÍDA ({len(predictions)} amostras):")

        for i, pred in enumerate(predictions):
            logger.info(
                f" -> Amostra {i+1}: Tempo de Entrega Predito: {pred:.2f} minutos"
            )

        logger.info("-" * 45)

    except Exception as e:
        logger.error(f"ERRO durante a predição: {e}")


if __name__ == "__main__":
    # Ignora avisos de versão do scikit-learn
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    run_prediction_pipeline()
