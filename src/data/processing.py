"""Funções de tratamento dos dados"""
import logging
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from scipy.stats import chisquare, shapiro, kstest, norm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import pandas as pd
from ipywidgets import interact, HTML, Output, Dropdown, VBox, interactive
from IPython.display import display, HTML
from sklearn.preprocessing import PowerTransformer
from typing import Optional

# instância do objeto logger
logger = logging.getLogger(__name__)

def amostra_dados(df: DataFrame) -> DataFrame:
    """Função para retornar a amostragem dos dados"""
    return df.sample(3)

def contagem_valores(coluna:Series) -> None: 
    """Função que realiza a contagem de valores por coluna"""
    return coluna.value_counts()

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

def filtrar_feature_valor_categorico(df: DataFrame, query:str) -> DataFrame:
    """Função que aplica um filtro em uma variável categorica ou em um conjunto delas através do método df.query"""
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

def teste_qui_quadrado_normalidade(df:DataFrame, cat_col:str, num_cols:list, bins=10, alpha=0.05) -> DataFrame:
    "Função que gera um dataset com a avaliação do teste qui quadrado das categorias da feature indicada."
    results = []

    for category in df[cat_col].unique():
        df_category = df[df[cat_col] == category]
        
        for num_col in num_cols:
            data = df_category[num_col].dropna()

            if len(data) < bins:
                continue

            # padronizar
            zscores = (data - data.mean()) / data.std()

            # observado
            obs, bin_edges = np.histogram(zscores, bins=bins)
            
            # esperado (usando normal padrão)
            cdf_vals = norm.cdf(bin_edges)
            expected_probs = np.diff(cdf_vals)
            expected = expected_probs * len(zscores)

            # teste qui-quadrado
            chi2, p = chisquare(f_obs=obs, f_exp=expected)

            results.append({
                "Categoria": category,
                "Coluna": num_col,
                "Chi2": chi2,
                "p-value": p,
                "Normal?": "Sim" if p > alpha else "Não"
            })

    return pd.DataFrame(results)

def teste_normalidade_por_categoria_auto(df:DataFrame, cat_col:str, num_cols:list, alpha=0.05) -> DataFrame:
    """Função que aplica teste de normalidade para as categorias de uma coluna a partir da quantidade de amostra disponível e retorna um dataset com as análises."""
    results = []

    for category in df[cat_col].unique():
        subset = df[df[cat_col] == category]

        for num_col in num_cols:
            data = subset[num_col].dropna().values
            n = len(data)

            if n < 3:  # amostra muito pequena
                results.append({
                    "Categoria": category,
                    "Coluna": num_col,
                    "N": n,
                    "Teste": None,
                    "Estatística": None,
                    "p-value": None,
                    "Normal?": "Amostra insuficiente"
                })
                continue

            # escolha do teste
            if n < 500:
                test_name = "Shapiro-Wilk"
                stat, p = shapiro(data)
            else:
                test_name = "Kolmogorov-Smirnov"
                # padronizar antes de aplicar KS contra normal padrão
                zscores = (data - np.mean(data)) / np.std(data, ddof=1)
                stat, p = kstest(zscores, 'norm')

            results.append({
                "Categoria": category,
                "Coluna": num_col,
                "N": n,
                "Teste": test_name,
                "Estatística": stat,
                "p-value": p,
                "Normal?": "Sim" if p > alpha else "Não"
            })

    return pd.DataFrame(results)

def verificacao_outlier(array, extreme = False):

    "Função para verificar outliers em um array."
    q1,q3 = np.quantile(array, [0.25, 0.75])
    iqr = q3-q1

    factor = 3 if extreme else 1.5
    upper_outlier = q3+factor*iqr
    lower_outlier = q1-factor*iqr

    return (array < lower_outlier) | (array > upper_outlier)

def power_transform_coluna_categorica(df: pd.DataFrame,cat_col: str,metodo: str = 'yeo-johnson', cols: Optional[list] = None) -> pd.DataFrame:
    """
    Aplica PowerTransformer (Box-Cox ou Yeo-Johnson) às colunas numéricas,
    agrupando os dados por uma coluna categórica.

    Args:
        df (pd.DataFrame): DataFrame de entrada com colunas numéricas e categóricas.
        cat_col (str): Nome da coluna categórica usada para agrupar.
        metodo (str, optional): Método do PowerTransformer ('yeo-johnson' ou 'box-cox').
        cols (list, optional): Lista de colunas numéricas a transformar. 
                               Se None, aplica em todas as numéricas.

    Returns:
        pd.DataFrame: DataFrame com as colunas numéricas transformadas por grupo.
    """
    df = df.copy()
    
    # Seleção de colunas numéricas (caso o usuário não especifique)
    if cols is None:
        cols = df.select_dtypes(include='number').columns.tolist()

    def _transform(group: pd.DataFrame) -> DataFrame:
        transformer = PowerTransformer(method=metodo, standardize=True)
        group = group.copy()
        group[cols] = transformer.fit_transform(group[cols])
        return group

    return df.groupby(cat_col, group_keys=False).apply(_transform)


def analise_vif_interativo(df: pd.DataFrame, coluna: str):
    """Função que realiza o teste VIF para as features da tabela a partir da coluna de filtro."""

    lista = sorted(df[coluna].dropna().unique())

    @interact(valor_selecionado=lista)
    def executar_analise_vif(valor_selecionado):
        # Filtra pelo valor selecionado
        df_filtrado = df[df[coluna] == valor_selecionado].copy()

        # Seleciona apenas features numéricas
        features_num = df_filtrado.select_dtypes(include='number').columns
        df_features = df_filtrado[features_num].dropna()

        # Remove colunas constantes (sem variação)
        df_features = df_features.loc[:, df_features.nunique() > 1]

        if df_features.shape[1] < 2:
            display(HTML(f"<h3>Poucas features numéricas válidas para {coluna}: {valor_selecionado}</h3>"))
            return

        # Função para calcular o VIF
        def vif_calculator(df_to_vif):
            vif_data = pd.DataFrame()
            vif_data['Feature'] = df_to_vif.columns
            vif_data['VIF'] = [
                variance_inflation_factor(df_to_vif.values, i) 
                for i in range(df_to_vif.shape[1])
            ]
            # Substitui inf por NaN e remove linhas inválidas
            vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna()
            return vif_data.sort_values(by="VIF", ascending=False)

        vif_resultado = vif_calculator(df_features)

        # Exibe o resultado
        display(HTML(f"<h3>Análise VIF para {coluna}: {valor_selecionado}</h3>"))
        display(vif_resultado)

