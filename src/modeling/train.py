"""Módulo de funções e classes de treinamento"""
import logging
import numpy as np
from pandas import DataFrame, Series
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, GridSearchCV, HalvingRandomSearchCV,KFold
from src.features.transformers import IQRDetectorClip

# instância do objeto logger
logger = logging.getLogger(__name__)

    

def separar_dados_treino_teste(X: DataFrame,y: Series, teste_size: float=0.2,random_state: int = 42) -> tuple[DataFrame, DataFrame, Series, Series]:
    """
    Separa os dados em conjuntos de treino e teste com o objeto train_test_split.
    
    params:
        df (DataFrame): DataFrame contendo todas as ações e seus dados.
        target (str): Nome da coluna da variável target (y).
        teste_size (float): Tamanho da amostragem disponível para teste dos dados.
        random_state (int): Valor da semente de aleatoriedade. Garante a repretição do embaralhamento dos dados para atingir os mesmos resultados em várias rodagens.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test) combinando os dados de todas as ações.
    """
    logger.info("Iniciando a separação dos dados em treino e teste por ação.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=teste_size, random_state=random_state)
        
        logger.info(f"Separação concluída. X_train: {X_train.shape}, X_test: {X_test.shape}, y_train: {y_train.shape}, y_test: {y_test.shape}")
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"Erro ao gerar a divisão dos dados : {e}")
        raise

def criar_pipeline(colunas_categoricas: list[str],colunas_numericas: list[str],modelo=None,usar_onehot: bool = True) -> Pipeline:
    """
    Cria um pipeline de pré-processamento + modelo.
    
    params:
    ----------
    colunas_categoricas : list[str] Colunas categóricas.
    colunas_numericas : list[str] Colunas numéricas.
    modelo : estimator, default=None Modelo a ser usado no final do pipeline.
    usar_onehot : bool, default=True Se True aplica OneHotEncoder; se False mantém as categorias apenas imputadas.
    """
    logger.info("Iniciando construção do pipeline de pré-processamento.")

    try:
        if usar_onehot:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
        else:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])

        # pipeline numérico
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('iqr', IQRDetectorClip()),
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('cat_features', cat_pipeline, colunas_categoricas),
                ('num_features', numeric_pipeline, colunas_numericas)
            ],
            remainder='passthrough',
            verbose_feature_names_out=False
        )

        pipeline = Pipeline(steps=[
            ('preprocessing', preprocessor),
            ('model', modelo)
        ])

        logger.info(f"Pipeline de pré-processamento construído com sucesso.{pipeline}")
        return pipeline

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise

def gerar_grid_search_cv(pipeline=Pipeline, param_grid=dict, cv=KFold(n_splits=5), scoring:str='r2', n_jobs:int=None, verbose:int=0) -> GridSearchCV:

    try:
        grid = GridSearchCV(estimator=pipeline,param_grid=param_grid,cv=cv,scoring=scoring,verbose=verbose,n_jobs=n_jobs)
        return grid

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise

def gerar_halving_random_search_cv(pipeline=Pipeline, param_grid=dict, cv=KFold(n_splits=5), scoring:str='r2', n_jobs:int=None, verbose:int=0) -> HalvingRandomSearchCV:

    try:
        halv= HalvingRandomSearchCV(estimator=pipeline,param_distributions=param_grid,cv=cv, scoring=scoring, n_candidates=100, factor=3, 
                                     verbose=verbose,min_resources='exhaust',n_jobs=n_jobs)
        return halv

    except Exception as e:
        logger.error(f"Erro ao criar pipeline de pré-processamento: {e}")
        raise


def gerar_metricas(y_true, y_pred):
    """
    Gera e imprime métricas de avaliação para modelos de regressão.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # Adicione outras métricas que você precisar
    logger.info(f"MAE: {mae:.4f}")
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"R2 Score: {r2:.4f}")