"""Arquivo de funções geradoras de gráficos estáticos"""

import logging
import math
from typing import Any, Dict, Union

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import sidetable as stb
import statsmodels.stats.multicomp as mc
from ipywidgets import interact
from pandas import DataFrame, Series
from scipy import stats
from sklearn.preprocessing import RobustScaler, StandardScaler

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
) -> None:
    """
    Cria um gráfico com múltiplos subplots, onde cada subplot exibe um boxplot
    de uma coluna numérica, agrupado pelas categorias da feature indicada.

    params:
        df (pd.DataFrame): DataFrame com os dados.
        cat_col (str): O nome da coluna categórica para agrupar os dados (ex: 'room_type').
        path (str): caminho para salvamento da imagem.

    return:
        None: Gráfico box-plot geral ou interativo com filtro a partir de uma feature categórica
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
                print(
                    f"Erro: A coluna categórica '{cat_col}' não foi encontrada no DataFrame."
                )
                return

            num_plots = len(numeric_cols)
            num_cols_grid = 3
            num_rows_grid = int(np.ceil(num_plots / num_cols_grid))

            fig, axs = plt.subplots(
                num_rows_grid,
                num_cols_grid,
                figsize=(5 * num_cols_grid, 4 * num_rows_grid),
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
                f"Análise de Distribuição por Categoria: '{cat_col}'",
                fontsize=16,
                y=1.02,
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
    Escalona e compara dois DataFrames usando box-plots.

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
        None: Gráfico box-plot em subplots.
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
        df1_scaled = pd.DataFrame(
            scaler.fit_transform(df1_features), columns=df1_features.columns
        )

        df2_features = df2[cols_df2]
        df2_scaled = pd.DataFrame(
            scaler.fit_transform(df2_features), columns=df2_features.columns
        )

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
    df: DataFrame,
    target: str,
    lista_features: list[str],
    path: str = None,
    tipo="str",
) -> None:
    """Função que retorna um gráfico pairplot das variáveis numéricas correlacionadas com o target indicado.

    params:
    df (DataFrame): Dataframe de entrada.
    target (str): feature alvo da previsão.
    lista_features (List[str]): lista de features para avaliar a correlação dos dados com o target.
    path (str): caminho para salvamento da imagem.
    tipo (str): tipo de gráfico pairplot.


    return:
    None: gráfico pairplot com o target.

    """
    ax = sns.pairplot(data=df, y_vars=target, x_vars=lista_features, kind=tipo)
    ax.figure.suptitle("Gráfico de dispersão das variáveis", y=1.05)
    if path:
        plt.savefig(path)
    return plt.show()


def grafico_coluna(
    df,
    x_col,
    y_col,
    hue_col=None,
    title=None,
    path: str = None,
    palette: str = None,
) -> None:
    """
    Cria um gráfico de colunas com a opção de um hue categórico.

    params:
        df (pd.DataFrame): O DataFrame com os dados.
        x_col (str): O nome da coluna para o eixo X (variável categórica).
        y_col (str): O nome da coluna para o eixo Y (variável numérica).
        hue_col (str, opcional): O nome da coluna para a cor (hue). Padrão é None.
        title (str, opcional): O título do gráfico.
        path (str): caminho para salvamento da imagem.
        palette (str): paleta de cores do gráfico.

    return:
        None: gráfico de colunas.

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
        palette=palette,
    )

    # Adiciona o título, se fornecido
    if title:
        plt.title(title, fontsize=16)

    # Melhora a visualização
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(f"Média de {y_col}", fontsize=12)
    plt.xticks(
        rotation=45, ha="right"
    )  # Rotaciona os rótulos do eixo X para melhor visualização
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
        None: gráfico relacional.
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
        None: gráfico categórico relacional.
    """
    g = sns.catplot(data=df, x=x, y=y, col=col_div, row=linha_div, hue=hue, kind=tipo)

    if titulo:
        g.fig.suptitle(titulo, fontsize=16, fontweight="bold")
        g.fig.subplots_adjust(top=0.9)

    if path:
        plt.savefig(path)

    plt.show()


def grafico_dispersao(
    df: DataFrame,
    y: Series,
    x: Series,
    titulo: str,
    xlabel: str,
    ylabel: str,
    res: bool = None,
    regr: bool = None,
    hue: str = None,
    size: str = None,
    path: str = None,
    ax=None,
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
        res (bool)=True: adiciona linha horizontal y=0.
        regr (bool) = True: adiciona a linha de regressão do
        hue (str): coluna para gerar categorização da dispersão dos dados.
        size (str): coluna para definir o tamanho da dispersão dos dados.
        path (str): caminho para salvamento da imagem.
        ax (matplotlib axis): Eixo para plotar. Se None, cria novo.

    return:
        None: gráfico dispersão.

    """

    was_called_alone = ax is None

    try:
        if was_called_alone:
            fig, ax = plt.subplots(figsize=(10, 5))

        if not regr:
            sns.scatterplot(data=df, x=x, y=y, hue=hue, size=size, legend="full", ax=ax)
        else:
            sns.regplot(data=df, x=x, y=y, line_kws={"color": "red"}, ax=ax)

        if res and not regr:
            ax.axhline(y=0, color="red", linestyle="--", linewidth=2)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(titulo)

        if was_called_alone:
            fig.tight_layout()

            if path:
                fig.savefig(path)

            plt.show()
            plt.close(fig)

    except Exception as e:
        logger.error(f"Erro ao gerar o gráfico de dispersão: {e}")
        return


def grafico_histograma(
    df: pd.DataFrame,
    feature: Union[str, list[str]],
    path: str = None,
    titulo: str = None,
    ax=None,
) -> None:
    """
    Exibe histogramas de uma ou mais colunas numéricas em grid de 2 colunas.

    params:
        df (DataFrame): DataFrame de entrada.
        feature (str | list[str]): Nome da coluna ou lista de colunas a serem plotadas.
        path (str): Caminho para salvar a imagem (opcional).
        titulo (str): Título geral (usado apenas quando múltiplos plots).
        ax (matplotlib axis): Eixo para plotar. Se None, cria novo.

    return:
        None: gráfico histograma.
    """
    try:
        if isinstance(feature, str):
            feature = [feature]

        n_features = len(feature)
        n_cols = 2
        n_rows = math.ceil(n_features / n_cols)

        if ax is None:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
            axes = axes.flatten()
        else:
            axes = [ax]

        for i, col in enumerate(feature):
            sns.histplot(df[col], color="steelblue", alpha=0.7, ax=axes[i], kde=True)
            axes[i].set_title(f"{col}", fontsize=12)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        if titulo and ax is None:
            fig.suptitle(titulo, fontsize=16, y=1.02)

        plt.tight_layout()

        if path:
            plt.savefig(path, bbox_inches="tight")

        if ax is None:
            plt.show()

    except Exception as e:
        print(f"Erro ao gerar o gráfico: {e}")
        return


def grafico_lineplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    hue_col: str,  # Nova coluna para diferenciar as linhas (ex: 'Cenário')
    path: str = None,
    titulo: str = None,
    ax=None,
) -> None:
    """
    Exibe um único line plot com múltiplas linhas (hue) em relação a x_col.

    NOTA: O DataFrame de entrada (df) deve estar no formato 'long' (longitudinal),
    contendo as colunas x_col, y_col e hue_col.

    params:
        df (DataFrame): DataFrame de entrada (já no formato long).
        x_col (str): Nome da coluna para o eixo X (ex: 'dia').
        y_col (str): Nome da coluna para o eixo Y (ex: 'OTD (%)').
        hue_col (str): Nome da coluna que define as diferentes linhas (ex: 'Cenário').
        path (str): Caminho para salvar a imagem (opcional).
        titulo (str): Título principal do gráfico.
        ax (matplotlib axis): Eixo para plotar. Se None, cria novo.

    return:
        None: gráfico de linha.
    """
    try:
        if ax is None:
            # Cria a figura e o eixo, já que será um gráfico único
            fig, ax = plt.subplots(figsize=(12, 6))

        # 1. Gera o lineplot no formato longitudinal
        sns.lineplot(
            data=df, x=x_col, y=y_col, hue=hue_col, marker="o", dashes=False, ax=ax
        )

        # 2. Configuração de Títulos e Rótulos
        ax.set_title(titulo, fontsize=16, pad=20)
        ax.set_xlabel(x_col.replace("_", " ").title(), fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.grid(True, linestyle="--", alpha=0.7)

        # Ajusta a legenda
        ax.legend(
            title=hue_col,
        )

        plt.tight_layout()

        # Salva ou exibe o gráfico
        if path:
            plt.savefig(path, bbox_inches="tight")

        if "fig" in locals():
            plt.show()

    except Exception as e:
        print(f"Erro ao gerar o gráfico: {e}")
        return


def plot_performance(
    df_stacked: pd.DataFrame, df_plot: pd.DataFrame, coluna_linha: str
) -> None:
    """
    Gera gráfico de barras empilhadas (corrigidos/não corrigidos)
    com linha da economia média de tempo por dia.

    params:
        df_stacked (pd.DataFrame): colunas ['dia','Corrigido','Não Corrigido']
        df_plot (pd.DataFrame): colunas ['dia', 'media_economia', ...]
        coluna_linha (str): O nome REAL da coluna de média no df_plot (ex: 'media_economia')

    return:
        None: gráfico personalizado comparativo barra e linha.
    """
    fig, ax1 = plt.subplots(figsize=(14, 4))
    ax2 = ax1.twinx()

    # Paleta
    cor_nao = "#A9A9A9"
    cor_corr = "#E7875A"
    cor_linha = "#1E90FF"

    ax1.bar(
        df_stacked["dia"],
        df_stacked["Nao Corrigido"],
        0.85,
        label="Não Corrigido",
        color=cor_nao,
    )
    ax1.bar(
        df_stacked["dia"],
        df_stacked["Corrigido"],
        0.85,
        bottom=df_stacked["Nao Corrigido"],
        label="Corrigido",
        color=cor_corr,
    )

    ax2.plot(
        df_plot["dia"],
        df_plot[coluna_linha],
        marker="o",
        color=cor_linha,
        linewidth=2.5,
        label="Média Economia (min)",
    )

    ax1.set_xlabel("Dia da Simulação")
    ax1.set_ylabel("Volume de Pedidos", fontweight="bold")
    ax2.set_ylabel("Média Economia (min)", color=cor_linha, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=cor_linha)

    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    plt.title(
        "Performance Diária: Qtde de Correções x Economia Média (Linha)",
        fontsize=14,
        fontweight="bold",
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=True)

    plt.tight_layout()
    plt.show()
