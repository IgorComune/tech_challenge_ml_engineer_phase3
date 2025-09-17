"""Funções de visualização dos dados na etapa de EDA"""
from IPython.display import display
from ipywidgets import interact
import logging
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import missingno as msno
import sidetable as stb
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
import seaborn as sns
from scipy import stats
import plotly.express as px
from typing import Dict, Any
sns.set_style("darkgrid")

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

def boxplot_comparativo_escalonamento_entre_dfs(df1: DataFrame, cols_df1: list, df2: DataFrame, cols_df2: list, title1: str, title2: str, scale: str = 'StandardScaler') -> plt.plot:
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

    return:
        plt.plot: Exibe o gráfico.
    """
    try:
        if scale == 'StandardScaler':
            scaler = StandardScaler()
        elif scale == 'RobustScaler':
            scaler = RobustScaler()
        else:
            raise ValueError("Método de escalonamento inválido. Use 'StandardScaler' ou 'RobustScaler'.")
        
        logger.info(f"Método de escalonamento: {scale}")

        # Escalonar os dados do primeiro DataFrame
        df1_features = df1[cols_df1]
        df1_scaled = pd.DataFrame(scaler.fit_transform(df1_features), columns=df1_features.columns)

        # Escalonar os dados do segundo DataFrame
        df2_features = df2[cols_df2]
        df2_scaled = pd.DataFrame(scaler.fit_transform(df2_features), columns=df2_features.columns)

        # Criar a figura com dois subplots lado a lado
        fig, axs = plt.subplots(ncols=2, figsize=(20, 8))

        # Plotar os box plots nos respectivos subplots
        df1_scaled.plot.box(ax=axs[0], title=title1)
        df2_scaled.plot.box(ax=axs[1], title=title2)

        # Ajustar automaticamente os rótulos do eixo x
        fig.autofmt_xdate(rotation=60, ha='right')
        
        # Exibir os gráficos
        plt.show()
    
    except Exception as e:
        logger.error(e)



def grafico_dispersao(df: DataFrame, y:Series, x:Series, titulo:str, xlabel:str, ylabel:str, interativo:bool=None, feature:str = None, res:bool=None, 
                      hue:Series=None, size:Series=None) -> plt.plot:
    """
    Gera um gráfico de dispersão.
    - interativo=True: adiciona um seletor interativo para filtrar pela coluna `feature`.
    - res=True: adiciona linha horizontal y=0.
    """
    try:
        def plot(valor_feature=None):
            plot_df = df.copy()
            if feature is not None and valor_feature is not None:
                plot_df = plot_df[plot_df[feature] == valor_feature]
            
            plt.figure(figsize=(10,5))
            sns.scatterplot(data=plot_df, x=x, y=y, hue=hue, size=size, legend="full")
            if res:
                plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
                plt.xlim(-5,5)
                plt.ylim(-10,10)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title(titulo)
            plt.tight_layout()
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
    
def grafico_pairplot_target(df:DataFrame, target:str, lista_features:list[str]) -> plt.plot:
       """Função que retorna um gráfico pairplot das variáveis numéricas correlacionadas com o target indicado.
       
       :params df: Dataframe
       :params target: feature alvo da previsão.
       :params lista_features: lista de features para avaliar a correlação dos dados com o target.

       """
       ax = sns.pairplot(data=df, y_vars=target, x_vars=lista_features)
       ax.figure.suptitle('Gráfico de dispersão das variáveis', y=1.05)
       return plt.show()

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

def grafico_pairplot_target(df:DataFrame, target:str, lista_features:list[str], tipo:str='reg') -> plt.plot:
       """Função que retorna um gráfico pairplot das variáveis numéricas correlacionadas com o target indicado.
       :params df: Dataframe
       :params target: feature alvo da previsão.
       :params lista_features: lista de features para avaliar a correlação dos dados com o target.
       """
       try:
            if tipo=='reg':
                dict_line ={'line_kws':{'color':'red'}}   
            else:
                dict_line=None 

            ax = sns.pairplot(data=df, y_vars=target, x_vars=lista_features, kind=tipo, plot_kws=dict_line)
            ax.figure.suptitle('Gráfico de dispersão das variáveis', y=1.05)
            return plt.show()
       except Exception as e:
           logger.error(e)

def grafico_coluna(df, x_col, y_col, hue_col=None, title=None):
    """
    Cria um gráfico de colunas com a opção de um hue categórico.

    params:
        df (pd.DataFrame): O DataFrame com os dados.
        x_col (str): O nome da coluna para o eixo X (variável categórica).
        y_col (str): O nome da coluna para o eixo Y (variável numérica).
        hue_col (str, opcional): O nome da coluna para a cor (hue). Padrão é None.
        title (str, opcional): O título do gráfico.
    """
    # Define o tamanho da figura
    plt.figure(figsize=(10, 6))

    # Cria o gráfico de colunas
    ax = sns.barplot(
        data=df,
        x=x_col,
        y=y_col,
        hue=hue_col,
        errorbar=None, # Remove a barra de erro para simplificar o exemplo
        palette='pastel'
    )

    # Adiciona o título, se fornecido
    if title:
        plt.title(title, fontsize=16)

    # Melhora a visualização
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(f'Média de {y_col}', fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotaciona os rótulos do eixo X para melhor visualização
    plt.tight_layout() # Ajusta o layout para evitar sobreposições
    plt.show()

def gerar_mapa_scatter_plot(
    df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        color_col: str=None,
        size_col: str=None,
        hover_name_col: str=None,
        hover_data_dict: Dict[str, Any]=None,
        center: Dict[str, float] = None,
        zoom: int = 1,
        height:int=None,
        title: str = "Mapa de Dispersão",
        jitter_amount: float = 0.005
    ):
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
        """
        # Adiciona jitter às coordenadas para evitar sobreposição
        df_temp = df.copy()
        df_temp[f'{lat_col}_jittered'] = df_temp[lat_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df_temp))
        df_temp[f'{lon_col}_jittered'] = df_temp[lon_col] + np.random.uniform(-jitter_amount, jitter_amount, size=len(df_temp))

        fig = px.scatter_map(
            df_temp,
            lat=f'{lat_col}_jittered',
            lon=f'{lon_col}_jittered',
            color=color_col,
            size=size_col,
            hover_name=hover_name_col,
            hover_data=hover_data_dict,
            zoom=zoom,
            center=center,
            height=height,
            title=title
        )
        
        # Define o estilo de mapa padrão
        fig.update_layout(mapbox_style="carto-positron")
        
        fig.show()

def grafico_replot(df:DataFrame, x:str, y:str, col_div:str, linha_div:str,  hue:str, tipo:str='scatter', figsize:tuple=(12,8), titulo:str=None) ->plt.plot:
    # Define o tamanho do gráfico
    plt.figure(figsize=figsize)

    sns.relplot(
        data=df,
        x=x,
        y=y,
        col=col_div,  
        row=linha_div,
        hue=hue,
        kind=tipo
    )

    plt.suptitle(titulo, y=1.02)
    return plt.show()
