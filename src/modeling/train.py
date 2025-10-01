"""Módulo de funções e classes de treinamento"""

import logging
import time
from typing import Any, Dict

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_halving_search_cv
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    KFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

from src.features.transformers import IQRDetectorClip

# instância do objeto logger
logger = logging.getLogger(__name__)


def separar_dados_treino_teste(
    X: DataFrame, y: Series, teste_size: float = 0.2, random_state: int = 42
) -> tuple[DataFrame, DataFrame, Series, Series]:
    """
    Separa os dados em conjuntos de treino e teste com o objeto train_test_split.

    params:
        df (DataFrame): DataFrame contendo todas as ações e seus dados.
        target (str): Nome da coluna da variável target (y).
        teste_size (float): Tamanho da amostragem disponível para teste dos dados.
        random_state (int): Valor da semente de aleatoriedade. Garante a repretição do embaralhamento dos dados para atingir os mesmos resultados em várias rodagens.

    return:
        tuple: (X_train, X_test, y_train, y_test) combinando os dados de todas as ações.
    """
    logger.info("Iniciando a separação dos dados em treino e teste por ação.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=teste_size, random_state=random_state
        )

        logger.info(
            f"Separação concluída. X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}"
        )
        return X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Erro ao gerar a divisão dos dados : {e}")
        raise


def criar_pipeline(
    colunas_categoricas: list[str],
    colunas_numericas: list[str],
    colunas_discretizacao: list[str],
    modelo=None,
    usar_onehot: bool = True,
) -> Pipeline:
    """
    Cria um pipeline de pré-processamento + modelo.

    params:

        colunas_categoricas : list[str] Colunas categóricas.
        colunas_numericas : list[str] Colunas numéricas.
        modelo : estimator, default=None Modelo a ser usado no final do pipeline.
        usar_onehot : bool, default=True Se True aplica OneHotEncoder; se False mantém as categorias apenas imputadas.

    return:
        pipeline com modelo.

    """
    logger.info("Iniciando construção do pipeline de pré-processamento.")

    try:

        if usar_onehot:
            cat_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    ),
                ]
            )
        else:
            cat_pipeline = Pipeline(
                [("imputer", SimpleImputer(strategy="most_frequent"))]
            )

        discretizer_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("iqr", IQRDetectorClip()),
                (
                    "discretizer",
                    KBinsDiscretizer(
                        n_bins=10,
                        encode="ordinal",
                        strategy="quantile",
                        random_state=42,
                    ),
                ),
            ]
        )

        numeric_passthrough = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="mean")),
                ("iqr", IQRDetectorClip()),
            ]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("cat_features", cat_pipeline, colunas_categoricas),
                ("discretized_features", discretizer_pipeline, colunas_discretizacao),
                ("num_passthrough", numeric_passthrough, colunas_numericas),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )

        pipeline = Pipeline(steps=[("preprocessing", preprocessor), ("model", modelo)])

        logger.info(f"Pipeline de pré-processamento construído com sucesso.{pipeline}")
        return pipeline

    except Exception as e:
        logger.error(f"Erro ao construir o pipeline: {e}")
        raise e


def gerar_grid_search_cv(
    pipeline=Pipeline,
    param_grid=dict,
    cv=KFold(n_splits=5),
    scoring: str = "r2",
    n_jobs: int = None,
    verbose: int = 0,
) -> GridSearchCV:
    """
    Gera um grid de validação e ajuste de hiperparâmetros pelo método de teste exaustivo com cross-validação (Kfold) para os hiperpâmetros do modelo.

    params:
        pipeline (Pipeline): Modelo acoplado ao pipeline de pré-processamento dos dados.
        param_grid (dict): Grid com hiperparâmetros de ajuste.
        cv (Kfold=5): Objeto de cross-validação dos dados.
        scoring (str='r2'): Métrica de avaliação.
        n_jobs (int=None): Capacidade de processamento definido.
        verbose (int=0): Descrição textual dos processos de validação.

    return:
        GridSearchCV: Objeto com modelo ajustado e com pipeline de pré-processamento definido.

    """

    try:
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=verbose,
            n_jobs=n_jobs,
        )
        return grid

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise


def gerar_halving_random_search_cv(
    pipeline=Pipeline,
    param_grid=dict,
    cv=KFold(n_splits=5),
    scoring: str = "r2",
    n_jobs: int = None,
    verbose: int = 0,
) -> HalvingRandomSearchCV:
    """
    Gera um grid de validação e ajuste de hiperparâmetros com seleção randômica para cross-validação (Kfold) para os hiperpâmetros do modelo.

    params:
        pipeline (Pipeline): Modelo acoplado ao pipeline de pré-processamento dos dados.
        param_grid (dict): Grid com hiperparâmetros de ajuste.
        cv (Kfold=5): Objeto de cross-validação dos dados.
        scoring (str='r2'): Métrica de avaliação.
        n_jobs (int=None): Capacidade de processamento definido.
        verbose (int=0): Descrição textual dos processos de validação.

    return:
        HalvingSearchCV: Objeto com modelo ajustado e com pipeline de pré-processamento definido.
    """
    try:
        halv = HalvingRandomSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            cv=cv,
            scoring=scoring,
            n_candidates=100,
            factor=3,
            verbose=verbose,
            min_resources="exhaust",
            n_jobs=n_jobs,
        )
        return halv

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise


def gerar_metricas(y_true, y_pred):
    """
    Gera e imprime métricas de avaliação para modelos de regressão.

    params:

        y_true: valor real.
        y_pred: valor predição.

    return:
        mae: erro médio absoluto.
        mse: erro médio quadrático.
        rmse: raiz do erro médio quadrático.
        r2: coeficiente de de determinação.
    """
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"MSE: {mse:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")

    return mae, mse, rmse, r2


def coletar_metricas(
    model_name: str,
    metricas_treino=dict,
    metricas_teste=dict,
    tmp_execucao_treino=time,
    tmp_execucao_predicao=time,
    params=Any,
) -> Dict[str, Any]:
    """
    Coleta métricas de performance, tempo de execução e hiperparâmetros de um modelo.

    params:
        model (Any): O modelo treinado (ex: grid_search.best_estimator_).
        model_name (str): Nome do modelo para identificação.
        metricas_treino (dict): Dicionário com as métricas de treino.
        metricas_teste (dict): Dicionário com as métricas de teste.
        tmp_execucao_treino (time): tempo de execucao de treino.
        tmp_execucao_predicao (time): tempo de execucao de treino.
        params: (Any): Parâmetros do modelo

    return:
        Dict[str, Any]: Um dicionário com todas as métricas e parâmetros coletados.
    """

    registro = {
        "Modelo": model_name,
        "R2_Teste": metricas_teste["R2_Score"],
        "MAE_Teste": metricas_teste["MAE"],
        "MSE_Teste": metricas_teste["MSE"],
        "RMSE_Teste": metricas_teste["RMSE"],
        "R2_Treino": metricas_treino["R2_Score"],
        "MAE_Treino": metricas_treino["MAE"],
        "MSE_Treino": metricas_treino["MSE"],
        "RMSE_Treino": metricas_treino["RMSE"],
        "Tempo_Treino_Segundos": tmp_execucao_treino,
        "Tempo_Predicao_Segundos": tmp_execucao_predicao,
        # PARÂMETROS
        "Hiperparametros": params,
    }

    return registro
