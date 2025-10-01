"""Funções utilitárias de manipulação dos dados"""

import logging
from typing import Any, Optional

import pandas as pd
import sidetable as stb
from geopy.distance import great_circle
from IPython.display import HTML, display
from ipywidgets import interact
from pandas import DataFrame
from sklearn.preprocessing import PowerTransformer

# instância do objeto logger
logger = logging.getLogger(__name__)


def filtragem_interativa_valores_categoricos(df: DataFrame, coluna: str) -> DataFrame:
    """
    Função que aplica um filtro iterativo para selecionar os dados do dataset a partir dos valores da coluna selecionada.

    params:

        df (DataFrame): DataFrame de entrada.
        coluna (str): coluna com categorias para filtragem interativa.

    returns:

        DataFrame: DataFrame com filtragem interativa.

    """

    lista = sorted(df[coluna].unique())

    @interact(valor=lista)
    def gerar_dataframe(valor):
        filtro = df.query(f"{coluna}=='{valor}'")
        return filtro


def filtrar_dataset(df: DataFrame, query: str) -> DataFrame:
    """
    Função que aplica um filtro em uma variável categorica ou em um conjunto delas através do método df.query

    params:

        df (DataFrame): DataFrame de entrada.
        query (str): query de filtragem dos dados do dataset.

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """
    output = None
    try:
        output = df.query(query)
        return output

    except Exception as e:
        logger.error(e)
        raise


def substituir_valores(
    df: DataFrame, filtro_linhas: list, filtro_colunas: list, valor: Any
) -> DataFrame:
    """
    Função que substitui os valores a partir dos filtros de linha ou coluna informados para o valor determinado.

    params:

        df (DataFrame): DataFrame de entrada.
        filtro_linhas (list): lista de valores de index para filtragem.
        filtro_colunas (list): lista de valores de index para filtragem.
        valor

    returns:

        DataFrame: DataFrame filtrado pela condição.

    """
    df.loc[filtro_linhas, filtro_colunas] = valor
    return df


def selecao_colunas(df: DataFrame, colunas: list) -> DataFrame:
    """
    Função que seleciona as colunas para montagem do dataset.

    params:

        df (DataFrame): DataFrame de entrada.
        colunas (list): lista de colunas para filtrar o dataset.

    returns:

        DataFrame: DataFrame filtrado com as colunas selecionadas.

    """
    return df[colunas]


def agrupar_dados(
    df: DataFrame, cols_agrup: list, cols_filter: list = None, agr=None
) -> DataFrame:
    """
    Função que agrupa as colunas e resume os valores por algum critério de agregação definido.

    params:

        df (DataFrame): DataFrame de entrada.
        cols_agrup (list): lista de colunas para agrupar o dataset.
        cols_filter (list): lista de colunas para filtrar o dataset.
        arg (list): função ou critério de agregação. Ex: np.sum(), 'sum', 'count'.

    returns:

        DataFrame: DataFrame filtrado com as colunas selecionadas.

    """
    try:
        if not cols_filter:
            logger.info(f"Agrupamento selecionado: {cols_agrup}, método: {agr}")
            df = df.groupby(by=cols_agrup).agg(agr)
        else:
            logger.info(
                f"Agrupamento selecionado: {cols_agrup}, filtragem dataset:{cols_filter}, método: {agr}"
            )
            df = df.groupby(by=cols_agrup)[cols_filter].agg(agr)

    except Exception as e:
        logger.error(e)

    return df


def distancia_dados_geolocalizacao(p1_lat, p1_lgt, p2_lat, p2_lgt) -> float:

    ponto1 = (p1_lat, p1_lgt)
    ponto2 = (p2_lat, p2_lgt)
    distancia = great_circle(ponto1, ponto2).km

    return distancia
