"""Função para manipulação de datasets"""

import logging

import pandas as pd

# instância do objeto logger
logger = logging.getLogger(__name__)


def salvar_dataset(df: pd.DataFrame, path: str, sep: str = ",") -> pd.DataFrame:
    """Função para salvar o dataset na pasta destino no formato csv.

    params:

        df (pd.DataFrame): DataFrame de entrada.
        path (str): Caminho de destino de persistência do dataset.
        sep (str): Separador do arquivo csv.

    returns:
        pd.DataFrame: DataFrame com as colunas numéricas transformadas por grupo.

    """
    logger.info(f"Dados salvos no path:{path}")
    return df.to_csv(path, sep=sep, index=False)


def carregar_dataset(path: str, sep: str = ";", index: bool = False) -> pd.DataFrame:
    """
    Função para carregar o dataset da pasta destino no formato csv.

    params:

        path (str): Caminho de origem do arquivo.
        sep (str): Separador do arquivo csv.
        index(bool): Salvar o index do dataframe.

    returns:
        pd.DataFrame: DataFrame com os dados carregados.

    """
    try:

        logger.info(f"Captura do arquivo csv no path:{path}")
        df = pd.read_csv(path, sep=sep)

    except Exception as e:
        logger.error(f"Erro no carregamento dos dados do path:{path}", e)
    return df
