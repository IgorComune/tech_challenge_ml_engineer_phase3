"""Funções de tratamento dos dados"""
import logging
from pandas import DataFrame, Series
import holidays
import pandas as pd
import numpy as np
import sidetable as stb
from ipywidgets import interact
from IPython.display import display, HTML
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import OneHotEncoder
from typing import Optional
from geopy.distance import great_circle
from category_encoders import TargetEncoder

# instância do objeto logger
logger = logging.getLogger(__name__)

def amostra_dados(df: DataFrame) -> DataFrame:
    """Função para retornar a amostragem dos dados"""
    return df.sample(3)

def contagem_valores(coluna:Series) -> None: 
    """Função que realiza a contagem de valores por coluna"""
    return coluna.value_counts()

def dados_temporais(df: DataFrame, data:Series) -> DataFrame:
    """Função que insere colunas com dados temporais a partir do index do Dataframe"""
    df['dayofweek'] = data.dt.day_of_week
    df['month'] = data.dt.month

    # criação do objeto com os feriados brasileiros
    india_holidays = holidays.India()
    df['Feriado'] = df.index.to_series().apply(lambda x: x in india_holidays)

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

def verificacao_nulos(df:DataFrame) -> Series:
    """Função que realiza a contagem de valores nulos por feature do dataset"""
    output = df.isna().sum()
    return output

def filtrar_linhas_valores_nulos(df:DataFrame) -> DataFrame:
    """Função que aplica o filtro de valores nulos no dataframe e retorna um dataframe filtrado com a correspondência."""
    output = df[df.isna().any(axis=1)]
    logger.info(f"Contagem de linhas nulas para o dataframe:{output.shape[0]}")
    return output

def frequencia_valores_nulos(df:DataFrame) -> DataFrame:
    """Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna"""
    return df.stb.missing()

def verificar_linhas_duplicadas(df:DataFrame) -> DataFrame:
    """Função que retorna um dataframe contendo as linhas duplicadas do dataset inputado."""
    output = \
    (df.groupby(df.columns.tolist(), dropna=False)
    .size()
    .to_frame('n_duplicates')
    .query('n_duplicates>1')
    .sort_values('n_duplicates', ascending=False)
    .head(5)
    )
    return output

def remover_duplicados(df: DataFrame, coluna: str) -> DataFrame:
    """Função para remoção de valores duplicados."""
    df.drop_duplicates(subset=[coluna], keep='first', inplace=True)
    return df

def filtragem_interativa_valores_categoricos(df: DataFrame, coluna: str) -> DataFrame:
    """Função que aplica um filtro iterativo para selecionar os dados do dataset a partir dos valores da coluna selecionada."""
    
    lista = sorted(df[coluna].unique())
    @interact(valor = lista)
    def gerar_dataframe(valor):
        filtro = df.query(f"{coluna}=='{valor}'")

        return filtro

def filtrar_dataset(df: DataFrame, query:str) -> DataFrame:
    """Função que aplica um filtro em uma variável categorica ou em um conjunto delas através do método df.query"""
    output = None 
    try:
        output = df.query(query)
    except Exception as e:
        logger.error(e)
    return output

def substituir_valores(df: DataFrame, filtro_linhas:list, filtro_colunas:list, valor) -> DataFrame:
    """Função que subsitui os valores a partir dos filtros de linha ou coluna informados para o valor determinado."""
    df.loc[filtro_linhas, filtro_colunas] = valor
    return df

def selecao_colunas(df: DataFrame, colunas: list) -> DataFrame:
    """Função que seleciona as colunas para montagem do dataset"""
    return df[colunas]


def agrupar_dados(df: DataFrame, cols_agrup: list, cols_filter: list=None, agr=None) -> DataFrame:
    """Função que agrupa as colunas para montagem do dataset."""
    try:
        if not cols_filter:
            logger.info(f'Agrupamento selecionado: {cols_agrup}, método: {agr}')
            df = df.groupby(by=cols_agrup).agg(agr)
        else:
            logger.info(f'Agrupamento selecionado: {cols_agrup}, filtragem dataset:{cols_filter}, método: {agr}')
            df = df.groupby(by=cols_agrup)[cols_filter].agg(agr)

    except Exception as e:
        logger.error(e)

    return df

def distancia_dados_geolocalizacao(p1_lat, p1_lgt, p2_lat, p2_lgt) -> float:
    
    ponto1 = (p1_lat, p1_lgt)
    ponto2 = (p2_lat, p2_lgt)
    distancia = great_circle(ponto1, ponto2).km
    
    return distancia

def power_transform(df: pd.DataFrame,cat_col: str=None,metodo: str = 'yeo-johnson', cols: Optional[list] = None) -> DataFrame:
    """
    Aplica PowerTransformer (Box-Cox ou Yeo-Johnson) às colunas numéricas,
    agrupando os dados por uma coluna categórica.

    params:
        df (pd.DataFrame): DataFrame de entrada com colunas numéricas e categóricas.
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

    def _transform(group: pd.DataFrame) -> DataFrame:
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


def target_encoding(feature:Series, dados_ajuste:DataFrame, target:Series):
    """Função que realiza o encoding de feature categórica com alta dimensaionalidade 
    de categorias a partir da representação dessa feature no target.
    
    """
    encoder = TargetEncoder(cols=[feature])
    encoder.fit(dados_ajuste, target)
    dados_encoded = encoder.transform(dados_ajuste)
    return dados_encoded

def one_hot_encoding(dados_ajuste:DataFrame, target:Series):
    """Função que realiza o encoding de feature categóricas transformando os valores categoricos em colunas booleanas.
    
    """
    encoder = OneHotEncoder()
    encoder.fit(dados_ajuste, target)
    dados_encoded = encoder.transform(dados_ajuste)
    df_tratado = pd.DataFrame(dados_encoded.toarray(), columns=encoder.get_feature_names_out())
    return df_tratado