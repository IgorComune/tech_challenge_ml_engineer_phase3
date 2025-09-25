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


def grafico_histograma(
    df: pd.DataFrame, interativo: bool = False, feature: str = None, path: str = None
) -> None:
    """
    Exibe histogramas das colunas numéricas.
    Se interativo=True, filtra por uma coluna categórica.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str): Nome da coluna categórica para agrupar (modo interativo).
        interativo (bool): Se True, a análise é por categoria com um widget. Se False, a análise é geral.
        path (str): caminho para salvamento da imagem.

    return:
        None: plot com uma matriz espersa de valores nulos.

    """

    def plot(dataframe, titulo_extra=""):
        dados_numericos = dataframe.select_dtypes(include="number")
        num_colunas = len(dados_numericos.columns)
        cols = 3
        rows = int(np.ceil(num_colunas / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        axs = axs.flatten()
        for i, col in enumerate(dados_numericos.columns):
            sns.histplot(dados_numericos[col], color="steelblue", alpha=0.7, ax=axs[i])
            axs[i].set_title(col)
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])
        fig.suptitle(f"Distribuição de Features Numéricas {titulo_extra}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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


def grafico_dispersao(
    df: DataFrame,
    y: Series,
    x: Series,
    titulo: str,
    xlabel: str,
    ylabel: str,
    interativo: bool = None,
    feature: str = None,
    res: bool = None,
    hue: Series = None,
    size: Series = None,
    path: str = None,
) -> None:
    """
    Gera um gráfico de dispersão para comparar.

    params:

        df (DataFrame): Dataframe de entrada.
        y (Series): coluna de valores do eixo y.
        x (Series): coluna de valores do eixo x.
        title (str): O título do gráfico.
        xlabel (str): O título do do eixo x.
        ylabel (str): O título do eixo y.
        interativo (bool)=True: adiciona um seletor interativo para filtrar pela coluna `feature`.
        res (bool)=True: adiciona linha horizontal y=0.
        feature (str): colunas para selecionar a plotagem por valores categóricos.
        hue (Series): coluna para gerar categorização da dispersão dos dados.
        size (Series): coluna para definir o tamanho da dispersão dos dados.
        path (str): caminho para salvamento da imagem.

    return:
        None: Exibe o gráfico.

    """
    try:

        def plot(valor_feature=None):
            plot_df = df.copy()
            if feature is not None and valor_feature is not None:
                plot_df = plot_df[plot_df[feature] == valor_feature]

            plt.figure(figsize=(10, 5))
            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, size=size, legend="full")
            if res:
                plt.axhline(y=0, color="red", linestyle="--", linewidth=2)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(titulo)
            plt.tight_layout()

            if path:
                plt.savefig(path)
            plt.show()

        if interativo and feature is not None:
            if feature not in df.columns:
                raise logger.warning(f"Coluna '{feature}' não existe no DataFrame.")
            valores = df[feature].dropna().unique()
            interact(plot, valor_feature=valores)
        else:
            plot()

    except Exception as e:
        return logger.error(e)


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
