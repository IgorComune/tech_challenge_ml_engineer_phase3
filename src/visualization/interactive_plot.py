"""Arquivo de funções geradoras de gráficos interativos"""

import logging
from typing import Any, Dict

from ipywidgets import interact
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotly.express as px
from scipy import stats
import seaborn as sns

sns.set_style("darkgrid")


# instância do objeto logger
logger = logging.getLogger(__name__)


def grafico_qq_plot(
    df: pd.DataFrame, interativo: bool = False, feature: str = None, path: str = None
) -> None:
    """
    Gera gráficos Q-Q Plot para as colunas numéricas.
    Se interativo=True e col_cat for informado, permite filtrar por valores dessa coluna.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str): Nome da coluna categórica para agrupar (modo interativo).
        interativo (bool): Se True, a análise é por categoria com um widget. Se False, a análise é geral.
        path (str): caminho para salvamento da imagem.

    return:
        None: plot com uma matriz espersa de valores nulos.

    """
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        def plot(dataframe):
            nrows = 3
            ncols = int(np.ceil(len(numeric_cols) / nrows))
            _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
            axs = axs.ravel()

            for i, col in enumerate(numeric_cols):
                data = dataframe[col].dropna()
                stats.probplot(data, dist="norm", plot=axs[i])
                axs[i].set_title(f"Q-Q Plot: {col}")
                axs[i].set_xlabel("Quantis Teóricos")
                axs[i].set_ylabel("Quantis da Amostra")
            for j in range(len(numeric_cols), len(axs)):
                axs[j].axis("off")

            plt.suptitle("Q-Q Plots Colunas Numéricas", fontsize=20, y=1.02)
            plt.tight_layout()
            if path:
                plt.savefig(path)
            plt.show()

        if interativo and feature and feature in df.columns:
            opcoes = sorted(df[feature].dropna().unique())

            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df[df[feature] == filtro])

        else:
            plot(df)

    except Exception as e:
        logger.error(f"Erro: {e}")




def grafico_heatmap(
    df: pd.DataFrame, interativo: bool = False, feature: str = None, path: str = None
) -> None:
    """
    Cria heatmap de correlação dos dados numéricos.
    Se interativo=True e feature informado, cria heatmap filtrado por valores dessa coluna.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str): Nome da coluna categórica para agrupar (modo interativo).
        interativo (bool): Se True, a análise é por categoria com um widget. Se False, a análise é geral.
        path (str): caminho para salvamento da imagem.

    return:
        None: plot com uma matriz espersa de valores nulos.
    """
    try:

        def plot(dataframe, titulo_extra=""):
            df_corr = dataframe.select_dtypes(include="number").corr()
            plt.figure(figsize=(10, 7))
            mask = np.triu(df_corr)
            sns.heatmap(df_corr, linewidths=0.5, cmap="vlag", mask=mask, annot=True)
            plt.title(f"Heatmap Correlação {titulo_extra}")
            if path:
                plt.savefig(path)
            plt.show()

        if interativo and feature and feature in df.columns:
            opcoes = sorted(df[feature].dropna().unique())

            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df[df[feature] == filtro], f"- {feature}: {filtro}")

        else:
            plot(df)

    except Exception as e:
        logger.error(f"Erro: {e}")


def gerar_mapa_scatter_plot(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: str = None,
    size_col: str = None,
    hover_name_col: str = None,
    hover_data_dict: Dict[str, Any] = None,
    center: Dict[str, float] = None,
    zoom: int = 1,
    height: int = None,
    title: str = "Mapa de Dispersão",
    jitter_amount: float = 0.005,
    path: str = None,
) -> None:
    """
    params:
        df (pd.DataFrame): O DataFrame a ser usado.
        lat_col (str): Nome da coluna para a latitude.
        lon_col (str): Nome da coluna para a longitude.
        color_col (str): Nome da coluna para a cor dos pontos.
        size_col (str): Nome da coluna para o tamanho dos pontos.
        hover_name_col (str): Nome da coluna para o nome ao passar o mouse.
        hover_data_dict (Dict[str, Any]): Dicionário com dados adicionais para o tooltip.
        zoom (int): Nível de zoom do mapa.
        center_lat (float): Latitude do centro do mapa.
        center_lon (float): Longitude do centro do mapa.
        title (str): Título do mapa.
        jitter_amount (float): Quantidade de jitter a ser adicionada para evitar sobreposição.
        path (str): caminho para salvamento da imagem no.

    return:
        None: plot com gráfico de dispersão em mapa.


    """
    # Adiciona jitter às coordenadas para evitar sobreposição
    df_temp = df.copy()
    df_temp[f"{lat_col}_jittered"] = df_temp[lat_col] + np.random.uniform(
        -jitter_amount, jitter_amount, size=len(df_temp)
    )
    df_temp[f"{lon_col}_jittered"] = df_temp[lon_col] + np.random.uniform(
        -jitter_amount, jitter_amount, size=len(df_temp)
    )

    fig = px.scatter_map(
        df_temp,
        lat=f"{lat_col}_jittered",
        lon=f"{lon_col}_jittered",
        color=color_col,
        size=size_col,
        hover_name=hover_name_col,
        hover_data=hover_data_dict,
        zoom=zoom,
        center=center,
        height=height,
        title=title,
    )

    # Define o estilo de mapa padrão
    fig.update_layout(mapbox_style="carto-positron")
    if path:
        fig.write_image(path)
    fig.show()
