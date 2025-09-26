"""Arquivo de treino do modelo"""
#%% carregamento das bibliotecas
import os
import sys
import mlflow
import pandas as pd
import pickle
from dotenv import load_dotenv
# diretório raiz
sys.path.append(os.path.abspath(os.path.join("..")))
from src.config.logging_config import setup_logging
from config.logging_config import setup_logging
from modeling.train import criar_pipeline, gerar_halving_random_search_cv, \
gerar_metricas, separar_dados_treino_teste
from sklearn.tree import DecisionTreeRegressor

##%% configuração do loggging
logger = setup_logging()

## variáveis de ambiente
load_dotenv()

#%% instancia do experimento no servidor mlflow
mlflow.set_experiment()

uri = os.getenv("MLFLOW_TRACKING_URI")
uri = mlflow.get_tracking_uri()
logger.info(f"MLflow Tracking URI: {uri}")

# início do experimento no mlflow
with mlflow.start_run(run_name="DecisionTree_Tuning_Baseline") as run:

    #%% carregar base de dados
    caminho_raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    caminho_dados_processados = os.path.join(
        caminho_raiz, "data", "processed", "amazon_delivery_processed.csv"
    )
    data = pd.read_csv(caminho_dados_processados, sep=",")

    X = data.drop(['month', 'holiday', 'pickup_duration', 'Agent_Age','order_cicle_time',
            'day_sin', 'day_cos', 'month_sin', 'month_cos',],axis=1)

    y = data['order_cicle_time']

    # definição das colunas categóricas e numéricas
    colunas_categoricas = ['Traffic','Vehicle','Area', 'Weather']
    colunas_numericas = [ 
        'delivery_distance','Agent_Rating',]

    #%% separação dos dados em treino e teste

    # separação dos dados em treino e teste
    X_treino, X_teste, y_treino, y_teste = separar_dados_treino_teste(X=X, y=y, teste_size=0.2, random_state=42)
    #%% treino do modelo

    pipe = criar_pipeline(colunas_categoricas=colunas_categoricas, 
                        colunas_numericas=colunas_numericas, 
                        modelo = DecisionTreeRegressor(random_state=42))


    # lista de parâmetros para ajuste
    params = {

        'model__criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
        'model__max_depth': [3, 5, 8, 12, 15, None],
        'model__min_samples_split': [2, 5, 10],
        'model__min_samples_leaf': [1, 2, 4]
    }


    # log dos parâmetros
    mlflow.log_params(params)

    halving = gerar_halving_random_search_cv(
        pipeline=pipe, param_grid=params, n_jobs=-1)

    halving.fit(X_treino, y_treino)
    best_model = halving.best_estimator_

    mlflow.log_params(halving.best_params_) 
    mlflow.log_metric("best_cv_score", halving.best_score_)
    logger.info(f"Métricas de treino logadas no MLflow: {halving.best_score_}")
    y_pred  = best_model.predict(X_teste)
    #%% métricas de desempenho

    metricas = gerar_metricas(y=y_teste, y_pred= y_pred)
    mlflow.log_metrics(metricas)
    logger.info(f"Métricas de teste logadas no MLflow: {metricas}")


    #%% persistência do modelo

    path = '../models/pred_model.pkl'
    with open(path,'wb') as f:

        pickle.dump(best_model, f)
    logger.info(f"Modelo salvo. path: {path}")

    #%% log do artefato mlflow
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",
        registered_model_name="DecisionTree_Baseline" 
    )
    logger.info("Modelo logado como artefato no MLflow.")