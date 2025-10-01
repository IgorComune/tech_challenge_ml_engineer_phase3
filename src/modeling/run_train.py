"""
Módulo principal para treino, ajuste de hiperparâmetros e rastreamento
do modelo de Regressão por Árvore de Decisão com MLflow e DagsHub.
"""

# Importações de Bibliotecas
import os
import pickle
import sys
import traceback
import warnings

import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.tree import DecisionTreeRegressor

from src.modeling.train import (
    criar_pipeline,
    gerar_halving_random_search_cv,
    gerar_metricas,
    separar_dados_treino_teste,
)

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
# ----------------------------------------------------------------------------

# --- CONFIGURAÇÃO DE AMBIENTE ---

# raiz do projeto
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

# configuração do logging
from src.config.logging_config import setup_logging

logger = setup_logging()

# variáveis de ambiente
load_dotenv()


# --- FUNÇÕES DE CONFIGURAÇÃO ---


def setup_mlflow_tracking():
    """Configura a conexão com o MLflow/DagsHub."""
    DAGSHUB_USERNAME = os.getenv("DAGSHUB_USERNAME")
    DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
    DAGSHUB_REPO = os.getenv("REPO_NAME")

    MLFLOW_FALLBACK_URI = "file:./mlruns"

    if DAGSHUB_USERNAME and DAGSHUB_TOKEN and DAGSHUB_REPO:
        try:
            # conexão com dagshub
            dagshub.init(
                repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True
            )

            logger.info(f"MLflow Tracking URI configurada: {mlflow.get_tracking_uri()}")
        except Exception as e:
            logger.warning(f"Erro ao conectar com DAGsHub: {e}. Usando MLflow local.")
            mlflow.set_tracking_uri(MLFLOW_FALLBACK_URI)
    else:
        logger.warning("Variáveis DAGsHub não configuradas. Usando MLflow local.")
        mlflow.set_tracking_uri(MLFLOW_FALLBACK_URI)

    return True


# --- FUNÇÃO PRINCIPAL ---


def run_training_pipeline(
    dados_path: str,
    model_name: str = "DecisionTree_Baseline",
    run_name: str = "DecisionTree_Tuning_Baseline",
):
    """
    Executa o pipeline completo de carregamento, treino, ajuste de hiperparâmetros
    e log do modelo com MLflow.
    """
    try:
        # carregamento dos dados
        logger.info(f"Carregando dados de: {dados_path}")
        data = pd.read_csv(dados_path, sep=",")

        colunas_drop = [
            "month",
            "holiday",
            "pickup_duration",
            "order_cicle_time",
            "Traffic",
            "weekend",
            "is_sunny_weather",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]

        cols_to_drop = [c for c in colunas_drop if c in data.columns]

        X = data.drop(columns=cols_to_drop + ["order_cicle_time"], errors="ignore")
        y = data["order_cicle_time"]

        colunas_categoricas = ["Vehicle", "Area", "Weather"]
        colunas_numericas = [
            "delivery_distance",
        ]

        colunas_discretizacao = [
            "Agent_Rating",
            "Agent_Age",
        ]

        # experimento mlflow
        mlflow.set_experiment(experiment_name=model_name)
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            # separação dos dados
            X_treino, X_teste, y_treino, y_teste = separar_dados_treino_teste(
                X=X, y=y, teste_size=0.2, random_state=42
            )
            logger.info("Dados separados em treino e teste.")

            # construção do pipeline
            pipe = criar_pipeline(
                colunas_categoricas=colunas_categoricas,
                colunas_numericas=colunas_numericas,
                colunas_discretizacao=colunas_discretizacao,
                modelo=DecisionTreeRegressor(random_state=42),
            )

            params = {
                "model__criterion": ["squared_error", "friedman_mse", "absolute_error"],
                "model__max_depth": [3, 5, 8, 12, 15, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            }

            mlflow.log_params(params)
            logger.info("Parâmetros do HalvingRandomSearch logados.")

            # ajuste do modelo com halving random
            halving = gerar_halving_random_search_cv(
                pipeline=pipe, param_grid=params, n_jobs=-1
            )

            logger.info("Iniciando ajuste de hiperparâmetros (HalvingRandomSearch)...")
            halving.fit(X_treino, y_treino)
            best_model = halving.best_estimator_
            logger.info("Ajuste concluído.")

            try:

                best_params = halving.best_params_

                for key, value in best_params.items():

                    if key.startswith("model__") or key == "n_iter":

                        clean_key = key.replace("__", "_")
                        str_value = str(value)

                        if "class" in str_value and "object at" in str_value:
                            logger.warning(
                                f"Ignorando parâmetro '{key}' com valor não serializável: {str_value[:50]}..."
                            )
                            continue

                        mlflow.log_param(f"param_{clean_key}", str_value)

                logger.info(
                    "Parâmetros do HalvingRandomSearch logados com sucesso (Melhores)."
                )

            except Exception as e:
                logger.error(
                    f"Falha ao logar parâmetros do HalvingRandomSearch no MLflow (Melhores): {e}"
                )

            # ------------------------------------

            # persistência de artefatos
            y_pred = best_model.predict(X_teste)

            metricas_tuple = gerar_metricas(y_true=y_teste, y_pred=y_pred)

            metricas_dict = {
                "mae": metricas_tuple[0],
                "mse": metricas_tuple[1],
                "rmse": metricas_tuple[2],
                "r2_score": metricas_tuple[3],
            }
            # ------------------------------------------------------------------

            # artefato de métricas
            mlflow.log_metrics(metricas_dict)
            logger.info(f"Métricas de teste: {metricas_dict}")

            # persistência local
            path = os.path.join(
                os.path.dirname(__file__), "..", "models", "pred_model.pkl"
            )
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(best_model, f)
            logger.info(f"Modelo salvo localmente em: {path}")

            # artefato do mlflow
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="model",
                registered_model_name=model_name,
                input_example=X_treino.head(1),
            )
            logger.info("Modelo logado como artefato e registrado no MLflow.")

    except Exception as e:
        logger.error(f"ERRO CRÍTICO no pipeline de treino: {e}")
        logger.error(traceback.format_exc())


# --- EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    # Inicializa MLflow/DagsHub
    setup_mlflow_tracking()

    # Define o caminho do arquivo de dados processados
    CAMINHO_RAIZ = ROOT_DIR
    CAMINHO_DADOS = os.path.join(
        CAMINHO_RAIZ, "data", "processed", "amazon_delivery_processed.csv"
    )

    # Executa o pipeline
    run_training_pipeline(dados_path=CAMINHO_DADOS)
