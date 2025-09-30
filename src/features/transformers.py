"""Arquivo de funções e classes de pré-processamento dos dados"""

import logging
from typing import Optional

import numpy as np
from pandas import DataFrame, Series
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer, OneHotEncoder
import holidays

# instância do objeto logger
logger = logging.getLogger(__name__)

def dados_temporais(df: DataFrame, data:str) -> DataFrame:
    """Função que insere colunas com dados temporais a partir do index do Dataframe"""
    df['dayofweek'] = df[data].dt.dayofweek
    df['month'] = df[data].dt.month
    df['weekend'] = (df['dayofweek'] >= 5).astype(int)

    # criação do objeto com os feriados brasileiros
    br_holidays = holidays.BR()
    df['Feriado'] = df[data].apply(lambda x: x in br_holidays)

    return df

def transformacao_ciclica(df: DataFrame, dias_uteis:bool=False) -> DataFrame:
    """Transformação cíclica"""
    
    try:
        if not dias_uteis:
            logger.info(f'Transformação cíclica para as colunas de dados temporais.')
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)


        else:
            logger.info(f'Transformação cíclica com dias úteis para as colunas de dados temporais.')
            df['day_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
            df['day_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    except Exception as e:
        logger.error(e)
    return df

class IQRDetectorClip(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.5):
        self.threshold = threshold
        self.lower_bound = None
        self.upper_bound = None

    def fit(self, X, y=None):
        X = X.astype(float)
        q1 = np.percentile(X, 25, axis=0)
        q3 = np.percentile(X, 75, axis=0)
        
        iqr = q3 - q1

        self.lower_bound = q1 - self.threshold * iqr
        self.upper_bound = q3 + self.threshold * iqr
        return self

    def transform(self, X):
        X = X.astype(float)
        X_outliers_clipped = np.clip(X, a_min=self.lower_bound, a_max=self.upper_bound)
        return X_outliers_clipped
    
    def get_feature_names_out(self, input_features=None):
        return input_features



def power_transform(df: DataFrame,cat_col: str=None,metodo: str = 'yeo-johnson', cols: Optional[list] = None) -> DataFrame:
    """
    Aplica PowerTransformer (Box-Cox ou Yeo-Johnson) às colunas numéricas,
    agrupando os dados por uma coluna categórica.

    params:
        df (DataFrame): DataFrame de entrada com colunas numéricas e categóricas.
        cat_col (str): Nome da coluna categórica usada para agrupar.
        metodo (str, optional): Método do PowerTransformer ('yeo-johnson' ou 'box-cox').
        cols (list, optional): Lista de colunas numéricas a transformar. 
                               Se None, aplica em todas as numéricas.

    returns:
        pd.DataFrame: DataFrame com as colunas numéricas transformadas por grupo.
    """

    df = df.copy()
    
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()


    def _transform(group: DataFrame) -> DataFrame:
        try:

            cols_existentes = [col for col in cols if col in group.columns]

            if not cols_existentes or len(group) == 0:
                return group

            transformer = PowerTransformer(method=metodo, standardize=True)

            group[cols_existentes] = transformer.fit_transform(group[cols_existentes])
            return group
        except Exception as e:
            logger.error(f"Erro ao transformar grupo: {e}")
            return group

    try:
        if cat_col and cat_col in df.columns:
            return df.groupby(cat_col, group_keys=False).apply(_transform)
        else:

            return _transform(df)
        
    except Exception as e:
        logger.error(e)


def one_hot_encoding(dados_ajuste:DataFrame, target:Series):
    """Função que realiza o encoding de feature categóricas transformando os valores categoricos em colunas booleanas.
    
    """
    encoder = OneHotEncoder()
    encoder.fit(dados_ajuste, target)
    dados_encoded = encoder.transform(dados_ajuste)
    df_tratado = DataFrame(dados_encoded.toarray(), columns=encoder.get_feature_names_out())
    return df_tratado
