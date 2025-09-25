"""Arquivo de funções geradoras de gráficos estáticos"""

import logging
from typing import Any, Dict

from ipywidgets import interact
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import plotly.express as px
from scipy import stats
import seaborn as sns
import sidetable as stb
from sklearn.preprocessing import RobustScaler, StandardScaler
import statsmodels.stats.multicomp as mc

sns.set_style("darkgrid")


# instância do objeto logger
logger = logging.getLogger(__name__)


def matriz_valores_nulos(df: DataFrame, path: str = None) -> None:
    """
    Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna

    params:
        df (DataFrame): DataFrame de entrada.
        path (str): caminho para salvamento da imagem.

    return:
        None: plot com uma matriz espersa de valores nulos.

    """
    try:
        msno.matrix(df, figsize=(10, 4))
        plt.title("Matriz esparsa de valores nulos.", fontdict={"fontsize": 12})
        if path:
            plt.savefig(path)
        return plt.show()

    except Exception as e:
        logger.error(f"Erro: {e}")


def grafico_boxplot(
    df: pd.DataFrame, interativo: bool = None, cat_col: str = None, path: str = None
) -> plt.plot:
    """
    Cria um gráfico com múltiplos subplots, onde cada subplot exibe um boxplot
    de uma coluna numérica, agrupado pelas categorias da feature indicada.

    params:
        df (pd.DataFrame): DataFrame com os dados.
        cat_col (str): O nome da coluna categórica para agrupar os dados (ex: 'room_type').
        path (str): caminho para salvamento da imagem.

    return:
        plt.plot: Gráfico box-plot geral ou interativo com filtro a partir de uma feature categórica
    """
    try:

        if not interativo:

            plt.figure(figsize=(14, 10))
            sns.boxplot(df)
            plt.xticks(rotation=60)
            plt.title(f"Análise Descritiva features numéricas")
            plt.tight_layout()
            if path:
                plt.savefig(path)

            return plt.show()
        else:

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if "latitude" in numeric_cols:
                numeric_cols.remove("latitude")
            if "longitude" in numeric_cols:
                numeric_cols.remove("longitude")

            if cat_col not in df.columns:
                print(f"Erro: A coluna categórica '{cat_col}' não foi encontrada no DataFrame.")
                return

            num_plots = len(numeric_cols)
            num_cols_grid = 3
            num_rows_grid = int(np.ceil(num_plots / num_cols_grid))

            fig, axs = plt.subplots(
                num_rows_grid, num_cols_grid, figsize=(5 * num_cols_grid, 4 * num_rows_grid)
            )

            # Achata a matriz de eixos para facilitar a iteração
            axs = axs.flatten() if num_plots > 1 else [axs]

            for i, col in enumerate(numeric_cols):
                sns.boxplot(data=df, x=cat_col, y=col, ax=axs[i])
                axs[i].set_title(f"Boxplot de {col} por {cat_col}", fontsize=12)
                axs[i].set_xlabel(cat_col)
                axs[i].set_ylabel(col)
                axs[i].tick_params(axis="x", rotation=60)

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(
                f"Análise de Distribuição por Categoria: '{cat_col}'", fontsize=16, y=1.02
            )
            if path:
                plt.savefig(path)
            plt.tight_layout()
            return plt.show()

    except Exception as e:
        logger.error(f"Erro: {e}")


def boxplot_comparativo_escalonamento_dados(
    df: pd.DataFrame, scale: str = "StandardScaler", path: str = None
) -> None:
    """
    Função que retorna um gráfico comparativo entre os dados com e sem escalonamento.

    params:
        df (pd.DataFrame): DataFrame com os dados.
        scale (str): Método de escalonamento dos dados: StandardScaler ou RobustScaler.
        path (str): caminho para salvamento da imagem.

    return:
        None: Gráfico box-plot geral ou interativo com filtro a partir de uma feature categórica.

    """

    try:
        if scale == "StandardScaler":
            scaler = StandardScaler()
        elif scale == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError(
                "Método de escalonamento errado. Use 'StandardScaler' ou 'MinMaxScaler'"
            )
        logger.info(f"Método de escalonamento: {scale}")

        # escalonamento dos dados
        features_numericas = df.select_dtypes(include="number")
        df_scaled = pd.DataFrame(
            scaler.fit_transform(features_numericas), columns=features_numericas.columns
        )

        # construção da figura
        fig, axs = plt.subplots(ncols=2, figsize=(20, 8))

        df.plot.box(ax=axs[0], title="Boxplot sem escalonamento")
        df_scaled.plot.box(ax=axs[1], title="Boxplot com escalonamento")
        fig.autofmt_xdate(rotation=60, ha="right")

        if path:
            plt.savefig(path)

        return plt.show()

    except Exception as e:
        return logger.error(e)


def boxplot_comparativo_escalonamento_entre_dfs(
    df1: DataFrame,
    cols_df1: list,
    df2: DataFrame,
    cols_df2: list,
    title1: str,
    title2: str,
    scale: str = "StandardScaler",
    path: str = None,
) -> None:
    """
    Escalona e compara dois DataFrames usando box plots.

    params:
        df1 (DataFrame): O primeiro DataFrame para escalonar e plotar.
        cols_df1 (list): Uma lista de colunas do df1 a serem escalonadas e plotadas.
        df2 (DataFrame): O segundo DataFrame para escalonar e plotar.
        cols_df2 (list): Uma lista de colunas do df2 a serem escalonadas e plotadas.
        title1 (str): O título para o primeiro gráfico.
        title2 (str): O título para o segundo gráfico.
        scale (str): O método de escalonamento a ser usado ('StandardScaler' ou 'RobustScaler').
        path (str): caminho para salvamento da imagem.

    return:
        None: Exibe o gráfico.
    """
    try:
        if scale == "StandardScaler":
            scaler = StandardScaler()
        elif scale == "RobustScaler":
            scaler = RobustScaler()
        else:
            raise ValueError(
                "Método de escalonamento inválido. Use 'StandardScaler' ou 'RobustScaler'."
            )

        logger.info(f"Método de escalonamento: {scale}")

        df1_features = df1[cols_df1]
        df1_scaled = pd.DataFrame(scaler.fit_transform(df1_features), columns=df1_features.columns)

        df2_features = df2[cols_df2]
        df2_scaled = pd.DataFrame(scaler.fit_transform(df2_features), columns=df2_features.columns)

        fig, axs = plt.subplots(ncols=2, figsize=(20, 8))

        df1_scaled.plot.box(ax=axs[0], title=title1)
        df2_scaled.plot.box(ax=axs[1], title=title2)

        fig.autofmt_xdate(rotation=60, ha="right")

        if path:
            plt.savefig(path)

        plt.show()

    except Exception as e:
        logger.error(e)


def grafico_pairplot_target(
    df: DataFrame, target: str, lista_features: list[str], path: str = None, tipo='str',
) -> None:
    """Função que retorna um gráfico pairplot das variáveis numéricas correlacionadas com o target indicado.

    params:
    df (DataFrame): Dataframe de entrada.
    target (str): feature alvo da previsão.
    lista_features (List[str]): lista de features para avaliar a correlação dos dados com o target.
     path (str): caminho para salvamento da imagem.

    return:
    None: gráfico pairplot com o target.

    """
    ax = sns.pairplot(data=df, y_vars=target, x_vars=lista_features, kind=tipo)
    ax.figure.suptitle("Gráfico de dispersão das variáveis", y=1.05)
    if path:
        plt.savefig(path)
    return plt.show()


def grafico_coluna(df, x_col, y_col, hue_col=None, title=None, path: str = None) -> None:
    """
    Cria um gráfico de colunas com a opção de um hue categórico.

    params:
        df (pd.DataFrame): O DataFrame com os dados.
        x_col (str): O nome da coluna para o eixo X (variável categórica).
        y_col (str): O nome da coluna para o eixo Y (variável numérica).
        hue_col (str, opcional): O nome da coluna para a cor (hue). Padrão é None.
        title (str, opcional): O título do gráfico.
        path (str): caminho para salvamento da imagem.

    return:
        None: plot com gráfico de colunas.

    """
    # Define o tamanho da figura
    plt.figure(figsize=(10, 6))

    # Cria o gráfico de colunas
    ax = sns.barplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        errorbar=None,  # Remove a barra de erro para simplificar o exemplo
        palette="pastel",
    )

    # Adiciona o título, se fornecido
    if title:
        plt.title(title, fontsize=16)

    # Melhora a visualização
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(f"Média de {y_col}", fontsize=12)
    plt.xticks(rotation=45, ha="right")  # Rotaciona os rótulos do eixo X para melhor visualização
    plt.tight_layout()  # Ajusta o layout para evitar sobreposições
    if path:
        plt.savefig(path)
    plt.show()


def grafico_replot(
    df: DataFrame,
    x: str,
    y: str,
    col_div: str,
    linha_div: str,
    hue: str,
    tipo: str = "scatter",
    titulo: str = None,
    path: str = None,
) -> None:
    """
    Cria um gráfico relacional (relplot) usando Seaborn, com divisões em linhas e colunas
    para visualização de subgrupos de dados.

    params:
        df (DataFrame): O DataFrame a ser usado para plotagem.
        x (str): A coluna para o eixo x.
        y (str): A coluna para o eixo y.
        col_div (str): A coluna para dividir o gráfico em subplots por coluna.
        linha_div (str): A coluna para dividir o gráfico em subplots por linha.
        hue (str): A coluna para diferenciar as cores dos pontos/linhas.
        tipo (str, opcional): O tipo de gráfico a ser gerado ('scatter' ou 'line').
                               Padrão é 'scatter'.
        titulo (str, opcional): Título principal para o gráfico. Padrão é None.
        path (str, opcional): Caminho completo para salvar a imagem do gráfico. Padrão é None.

    return:
        None: A função plota o gráfico diretamente e não retorna nenhum valor.
    """
    g = sns.relplot(data=df, x=x, y=y, col=col_div, row=linha_div, hue=hue, kind=tipo)

    if titulo:
        g.fig.suptitle(titulo, fontsize=16, fontweight="bold")
        g.fig.subplots_adjust(top=0.9)

    if path:
        plt.savefig(path)

    plt.show()


def grafico_catplot(
    df: DataFrame,
    x: str,
    y: str,
    col_div: str = None,
    linha_div: str = None,
    hue: str = None,
    tipo: str = "box",
    titulo: str = None,
    path: str = None,
) -> None:
    """
    Cria um gráfico categórico (catplot) usando Seaborn, ideal para visualizações
    que envolvem variáveis categóricas.

    Args:
        df (DataFrame): O DataFrame a ser usado para plotagem.
        x (str): A coluna para o eixo x. Pode ser categórica ou numérica.
        y (str): A coluna para o eixo y. Pode ser categórica ou numérica.
        col_div (str, opcional): A coluna para dividir o gráfico em subplots por coluna.
                                 Padrão é None.
        linha_div (str, opcional): A coluna para dividir o gráfico em subplots por linha.
                                   Padrão é None.
        hue (str, opcional): A coluna para diferenciar as cores das categorias. Padrão é None.
        tipo (str, opcional): O tipo de gráfico categórico a ser gerado
                              (ex: 'box', 'violin', 'swarm', 'bar'). Padrão é 'box'.
        titulo (str, opcional): Título principal para o gráfico. Padrão é None.
        path (str, opcional): Caminho completo para salvar a imagem do gráfico. Padrão é None.

    Returns:
        None: A função plota o gráfico diretamente e não retorna nenhum valor.
    """
    g = sns.catplot(data=df, x=x, y=y, col=col_div, row=linha_div, hue=hue, kind=tipo)

    if titulo:
        g.fig.suptitle(titulo, fontsize=16, fontweight="bold")
        g.fig.subplots_adjust(top=0.9)

    if path:
        plt.savefig(path)

    plt.show()
