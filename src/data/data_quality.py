"""Funções para inspeção e avaliação da qualidade dos dados."""

import logging

from pandas import DataFrame, Series
import sidetable as stb

# Configuração do logger para este módulo
logger = logging.getLogger(__name__)


def amostra_dados(df: DataFrame) -> DataFrame:
    """
    Função para retornar uma amostragem aleatória dos dados.

    params:
        df (DataFrame): DataFrame de entrada.

    returns:
        DataFrame (DataFrame): DataFrame com a amostra de dados.

    """
    return df.sample(3)


def verificacao_nulos(df: DataFrame) -> Series:
    """
    Função que realiza a contagem de valores nulos por feature do dataset.

    params:
        df (DataFrame): DataFrame de entrada.

    returns:
        Series: contagem de valores nulos nas colunas do DataFrame.

    """
    return df.isna().sum()


def filtrar_linhas_valores_nulos(df: DataFrame) -> DataFrame:
    """
    Função que aplica o filtro de valores nulos no dataframe e retorna um dataframe filtrado.

    params:
        df (DataFrame): DataFrame de entrada.

    returns:
        DataFrame: tabela filtrada com as linhas contendo valores nulos.

    """
    output = df[df.isna().any(axis=1)]
    logger.info(f"Contagem de linhas com valores nulos: {output.shape[0]}")
    return output


def frequencia_valores_nulos(df: DataFrame) -> DataFrame:
    """
    Função que era uma tabela com a contagem e frequência de valores nulos por coluna.

    params:
        df (DataFrame): DataFrame de entrada.

    returns:
        DataFrame: tabela com estatísticas de valores nulos por coluna da tabela.

    """
    return df.stb.missing()


def verificar_linhas_duplicadas(df: DataFrame) -> DataFrame:
    """
    Função que retorna um dataframe contendo as linhas duplicadas do dataset inputado.

    params:
        df (DataFrame): DataFrame de entrada.

    returns:
        DataFrame: tabela com as linhas duplicadas.

    """
    output = (
        df.groupby(df.columns.tolist(), dropna=False)
        .size()
        .to_frame("n_duplicates")
        .query("n_duplicates>1")
        .sort_values("n_duplicates", ascending=False)
        .head(5)
    )
    return output


def remover_duplicados(df: DataFrame, coluna: str) -> DataFrame:
    """

    Função para remoção de valores duplicados em uma coluna.

    params:
        df (DataFrame): DataFrame de entrada.

    returns:
        DataFrame: tabela sem registros duplicados.

    """
    df.drop_duplicates(subset=[coluna], keep="first", inplace=True)
    return df
