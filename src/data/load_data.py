"""Função para manipulação de datasets"""
import logging
import pandas as pd
# instância do objeto logger
logger = logging.getLogger(__name__)


def salvar_dataset(df:pd.DataFrame, path:str, sep:str=';') -> pd.DataFrame:
    """Função para salvar o dataset na pasta destino no formato csv."""
    logger.info(f'Dados salvos no path:{path}')
    return df.to_csv(path, sep=sep)


def carregar_dataset(path:str, sep:str=';') -> pd.DataFrame:
    """Função para salvar o dataset na pasta destino no formato csv."""    
    try: 
        logger.info(f'Captura do arquivo no path:{path}')
        df = pd.read_csv(path,sep=sep)
    except Exception as e:
        logger.error(f"Erro no carregamento dos dados do path:{path}",e)
    return df