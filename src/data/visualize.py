"""Funções de visualização dos dados na etapa de EDA"""
from IPython.display import display
from ipywidgets import interact, HTML, Output, Dropdown, VBox, widgets
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
import missingno as msno
import pandas as pd
from sklearn.preprocessing import StandardScaler, PowerTransformer, RobustScaler
import seaborn as sns
from scipy import stats

# instância do objeto logger
logger = logging.getLogger(__name__)

def matriz_valores_nulos(df:DataFrame)-> plt.plot:
    """Função que gera uma matriz esparsa com a visualização dos valores nulos intercalado com valores preenchidos por coluna"""
    msno.matrix(df, figsize=(10,4))
    plt.title("Matriz esparsa de valores nulos.", fontdict={'fontsize':12})
    return plt.show()

def grafico_qq_plot(df:DataFrame) -> plt.plot:
    """Função que gera um gráfico qq-plot para análise visual de normalidade dos dados."""

    # seleção das colunas numéricas
    data = df.select_dtypes(include=np.number).columns.tolist()

    # definição das linhas e colunas da figura
    nrows = 3
    ncols = int(np.ceil(len(data)/nrows))

    # construção da figura
    _, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 5 * nrows))
    
    # achata os eixos em um vetor para imputação dos gráficos
    axs = axs.ravel()
    # Itera sobre as colunas numéricas e seus respectivos eixos.
    for i, col in enumerate(data):
        data = df[col].dropna() 
        
        stats.probplot(data, dist="norm", plot=axs[i])
        axs[i].set_title(f'Q-Q Plot: {col}')
        axs[i].set_xlabel("Quantis Teóricos")
        axs[i].set_ylabel("Quantis da Amostra")

    # Oculta os subplots vazios, se houver
    for j in range(len(data), len(axs)):
        axs[j].axis('off')

    plt.suptitle("Q-Q Plots Colunas Numéricas", fontsize=20, y=1.02)
    plt.tight_layout()

    return plt.show()

def grafico_boxplot(df: pd.DataFrame) -> plt.plot:
    """
    Exibe um boxplot das features numéricas para cada valor selecionado de uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com dados.
        distribuicao (list[float]): Lista de percentis para a análise (não usada diretamente aqui, mas mantida).
    """

    # construção da figura
    plt.figure(figsize=(14, 10))
    sns.boxplot(df)
    plt.xticks(rotation=60)
    plt.title(f"Análise Descritiva features numéricas")
    plt.tight_layout()
    return plt.show()

def grafico_boxplot_feature_interativo(df: pd.DataFrame, feature: str):
    """
    Exibe um boxplot interativo para todas as colunas numéricas dos registros 
    filtrados por uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        feature (str): Coluna categórica usada para filtrar os dados (ex: 'room_type').
    """
    opcoes = sorted(df[feature].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try:
            # 1. Filtra os dados
            dados_filtrados = df[df[feature] == coluna]
            dados_numericos = dados_filtrados.select_dtypes(include='number')

            if dados_numericos.empty:
                raise ValueError("Nenhum dado numérico disponível para esta categoria.")
      
            # 3. Gera um único boxplot com todas as colunas
            plt.figure(figsize=(14, 10))
            sns.boxplot(data=dados_numericos)

            # 4. Ajusta o layout do gráfico
            plt.xticks(rotation=60)
            plt.title(f"Distribuição de Features Numéricas para '{coluna}'")
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"Erro ao gerar gráfico: {e}")

def grafico_multi_boxplot(df: pd.DataFrame, cat_col: str) -> None:
    """
    Cria um gráfico com múltiplos subplots, onde cada subplot exibe um boxplot
    de uma coluna numérica, agrupado pelas categorias da feature indicada.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        cat_col (str): O nome da coluna categórica para agrupar os dados (ex: 'room_type').
    """
    try:
        # Seleciona apenas as colunas numéricas, excluindo as de localização
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'latitude' in numeric_cols:
            numeric_cols.remove('latitude')
        if 'longitude' in numeric_cols:
            numeric_cols.remove('longitude')

        if cat_col not in df.columns:
            print(f"Erro: A coluna categórica '{cat_col}' não foi encontrada no DataFrame.")
            return
        
        # Define o layout dos subplots
        num_plots = len(numeric_cols)
        num_cols_grid = 3 
        num_rows_grid = int(np.ceil(num_plots / num_cols_grid))
        
        fig, axs = plt.subplots(num_rows_grid, num_cols_grid, figsize=(5 * num_cols_grid, 4 * num_rows_grid))
        
        # Achata a matriz de eixos para facilitar a iteração
        axs = axs.flatten() if num_plots > 1 else [axs]

        # Itera sobre as colunas numéricas e cria um boxplot para cada uma
        for i, col in enumerate(numeric_cols):
            sns.boxplot(data=df, x=cat_col, y=col, ax=axs[i])
            axs[i].set_title(f'Boxplot de {col} por {cat_col}', fontsize=12)
            axs[i].set_xlabel(cat_col)
            axs[i].set_ylabel(col)
            axs[i].tick_params(axis='x', rotation=45)

        # Remove subplots não utilizados
        for j in range(i + 1, len(axs)):
            fig.delaxes(axs[j])

        fig.suptitle(f"Análise de Distribuição por Categoria: '{cat_col}'", fontsize=16, y=1.02)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Ocorreu um erro ao gerar o gráfico: {e}")

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

def histograma_feature_categorica(df: pd.DataFrame, feature: str):
    """
    Exibe histogramas das colunas numéricas para os registros filtrados por uma feature categórica.

    Args:
        df (pd.DataFrame): DataFrame com os dados.
        feature (str): Coluna categórica usada para filtrar os dados (ex: 'ticker').
    """
    opcoes = sorted(df[feature].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try:
            logger.info(f"Gerando histograma para {feature} = {coluna}")

            # Filtra os dados
            dados_filtrados = df[df[feature] == coluna]
            dados_numericos = dados_filtrados.select_dtypes(include='number')

            # Define layout dos subplots
            num_colunas = len(dados_numericos.columns)
            cols = 3  # Número de colunas de plots por linha
            rows = int(np.ceil(num_colunas / cols))

            fig, axs = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axs = axs.flatten()  # Para indexar linearmente

            for i, col in enumerate(dados_numericos.columns):
                sns.histplot(dados_numericos[col], color='steelblue', alpha=0.7, ax=axs[i])
                axs[i].set_title(col)

            # Remove eixos não utilizados
            for j in range(i + 1, len(axs)):
                fig.delaxes(axs[j])

            fig.suptitle(f"Distribuição de Features Numéricas - {feature}: {coluna}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            return plt.show()

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")
            print("Erro:", e)

def grafico_heatmap_interativo(df:DataFrame, col_cat:str) -> plt.plot:
    """Função que cria um gráfico heatmap para categorias da feature selecionada."""
        # verificação dos dados de entrada e análise da ausência da multicolinearidade
    opcoes = sorted(df[col_cat].dropna().unique())

    @interact(coluna=opcoes)
    def plot(coluna):
        try: 
            df_corr = df.query(f"{col_cat}=='{coluna}'").select_dtypes(include='number').corr()
            plt.figure(figsize=(7,4))
            mask = np.triu(df_corr)
            sns.heatmap(df_corr,linewidths=0.5, cmap='vlag', mask = mask)
            return plt.show()
        
        except Exception as e:
            logger.error(e)