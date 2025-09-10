"""Funções de visualização dos dados na etapa de EDA"""
from IPython.display import display
from ipywidgets import interact
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import missingno as msno
import sidetable as stb
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns
from scipy import stats

# instância do objeto logger
logger = logging.getLogger(__name__)

def matriz_valores_nulos(df:DataFrame)-> plt.plot:
    """Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna"""
    try:
        msno.matrix(df, figsize=(10,4))
        plt.title("Matriz esparsa de valores nulos.", fontdict={'fontsize':12})
        return plt.show()
    
    except Exception as e:
        logger.error(f"Erro: {e}")


def grafico_qq_plot(df: pd.DataFrame, interativo: bool = False, col_cat: str = None):
    """
    Gera gráficos Q-Q Plot para as colunas numéricas.
    Se interativo=True e col_cat for informado, permite filtrar por valores dessa coluna.
    """
    try:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        def plot(dataframe):
            nrows = 3
            ncols = int(np.ceil(len(numeric_cols)/nrows))
            _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 5*nrows))
            axs = axs.ravel()

            for i, col in enumerate(numeric_cols):
                data = dataframe[col].dropna()
                stats.probplot(data, dist="norm", plot=axs[i])
                axs[i].set_title(f'Q-Q Plot: {col}')
                axs[i].set_xlabel("Quantis Teóricos")
                axs[i].set_ylabel("Quantis da Amostra")
            for j in range(len(numeric_cols), len(axs)):
                axs[j].axis('off')

            plt.suptitle("Q-Q Plots Colunas Numéricas", fontsize=20, y=1.02)
            plt.tight_layout()
            plt.show()

        if interativo and col_cat and col_cat in df.columns:
            opcoes = sorted(df[col_cat].dropna().unique())
            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df[df[col_cat] == filtro])
        else:
            plot(df)

    except Exception as e:
        logger.error(f"Erro: {e}")


def grafico_boxplot(df: pd.DataFrame, interativo:bool=None, cat_col: str=None) -> plt.plot:
    """
    Cria um gráfico com múltiplos subplots, onde cada subplot exibe um boxplot
    de uma coluna numérica, agrupado pelas categorias da feature indicada.

    params:
        df (pd.DataFrame): DataFrame com os dados.
        cat_col (str): O nome da coluna categórica para agrupar os dados (ex: 'room_type').

    retunr:
        plt.plot: Gráfico box-plot geral ou interativo com filtro a partir de uma feature categórica
    """
    try:
        
        if not interativo:

            plt.figure(figsize=(14, 10))
            sns.boxplot(df)
            plt.xticks(rotation=60)
            plt.title(f"Análise Descritiva features numéricas")
            plt.tight_layout()
            return plt.show()
        else:

            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
            if 'latitude' in numeric_cols:
                numeric_cols.remove('latitude')
            if 'longitude' in numeric_cols:
                numeric_cols.remove('longitude')

            if cat_col not in df.columns:
                print(f"Erro: A coluna categórica '{cat_col}' não foi encontrada no DataFrame.")
                return
            
            num_plots = len(numeric_cols)
            num_cols_grid = 3 
            num_rows_grid = int(np.ceil(num_plots / num_cols_grid))
            
            fig, axs = plt.subplots(num_rows_grid, num_cols_grid, figsize=(5 * num_cols_grid, 4 * num_rows_grid))
            
            # Achata a matriz de eixos para facilitar a iteração
            axs = axs.flatten() if num_plots > 1 else [axs]

            for i, col in enumerate(numeric_cols):
                sns.boxplot(data=df, x=cat_col, y=col, ax=axs[i])
                axs[i].set_title(f'Boxplot de {col} por {cat_col}', fontsize=12)
                axs[i].set_xlabel(cat_col)
                axs[i].set_ylabel(col)
                axs[i].tick_params(axis='x', rotation=60)

            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Análise de Distribuição por Categoria: '{cat_col}'", fontsize=16, y=1.02)
            plt.tight_layout()
            return plt.show()

    except Exception as e:
        logger.error(f"Erro: {e}")


def boxplot_comparativo_escalonamento_dados(df: pd.DataFrame, scale:str='StandardScaler') -> plt.plot:
    "Função que retorna um gráfico comparativo entre os dados com e sem escalonamento."
    
    try:
        if scale == 'StandardScaler':
            scaler = StandardScaler()
        elif scale == 'RobustScaler':
            scaler = RobustScaler()
        else:
            raise ValueError("Método de escalonamento errado. Use 'StandardScaler' ou 'MinMaxScaler'")
        logger.info(f"Método de escalonamento: {scale}")

        # escalonamento dos dados
        features_numericas = df.select_dtypes(include='number')
        df_scaled = pd.DataFrame(scaler.fit_transform(features_numericas), columns=features_numericas.columns)

        # construção da figura
        fig, axs = plt.subplots(ncols = 2, figsize=(20,8))

        df.plot.box(ax=axs[0], title='Boxplot sem escalonamento')
        df_scaled.plot.box(ax=axs[1], title='Boxplot com escalonamento')
        fig.autofmt_xdate(rotation=60, ha='right')

        return plt.show()

    except Exception as e:
        return logger.error(e)


def grafico_histograma(df: pd.DataFrame, interativo: bool = False, feature: str = None):
    """
    Exibe histogramas das colunas numéricas.
    Se interativo=True, filtra por uma coluna categórica.
    """
    def plot(dataframe, titulo_extra=''):
        dados_numericos = dataframe.select_dtypes(include='number')
        num_colunas = len(dados_numericos.columns)
        cols = 3
        rows = int(np.ceil(num_colunas / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        axs = axs.flatten()
        for i, col in enumerate(dados_numericos.columns):
            sns.histplot(dados_numericos[col], color='steelblue', alpha=0.7, ax=axs[i])
            axs[i].set_title(col)
        for j in range(i+1, len(axs)):
            fig.delaxes(axs[j])
        fig.suptitle(f"Distribuição de Features Numéricas {titulo_extra}", fontsize=16)
        plt.tight_layout(rect=[0,0.03,1,0.95])
        plt.show()

    if interativo and feature and feature in df.columns:
        opcoes = sorted(df[feature].dropna().unique())
        @interact(filtro=opcoes)
        def _plot(filtro):
            plot(df[df[feature] == filtro], f"- {feature}: {filtro}")
    else:
        plot(df)


def grafico_heatmap(df: pd.DataFrame, interativo: bool = False, col_cat: str = None):
    """
    Cria heatmap de correlação dos dados numéricos.
    Se interativo=True e col_cat informado, cria heatmap filtrado por valores dessa coluna.
    """
    try:
        def plot(dataframe, titulo_extra=''):
            df_corr = dataframe.select_dtypes(include='number').corr()
            plt.figure(figsize=(10,7))
            mask = np.triu(df_corr)
            sns.heatmap(df_corr, linewidths=0.5, cmap='vlag', mask=mask, annot=True)
            plt.title(f"Heatmap Correlação {titulo_extra}")
            plt.show()

        if interativo and col_cat and col_cat in df.columns:
            opcoes = sorted(df[col_cat].dropna().unique())
            @interact(filtro=opcoes)
            def _plot(filtro):
                plot(df[df[col_cat] == filtro], f"- {col_cat}: {filtro}")
        else:
            plot(df)

    except Exception as e:
        logger.error(f"Erro: {e}")
