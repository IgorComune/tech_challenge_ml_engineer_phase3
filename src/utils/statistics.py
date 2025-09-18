"""Funções de validação de testes estatísticos"""
import logging
from IPython.display import display, HTML
from ipywidgets import interact
import numpy as np
from pandas import DataFrame
import warnings
import contextlib
from io import StringIO
from scipy.stats import chisquare, shapiro, kstest, norm, ttest_ind
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import statsmodels.stats.multicomp as mc
import re


# instância do objeto logger
logger = logging.getLogger(__name__)


def verificacao_outlier(array, extreme = False):

    "Função para verificar outliers em um array."
    q1,q3 = np.quantile(array, [0.25, 0.75])
    iqr = q3-q1

    factor = 3 if extreme else 1.5
    upper_outlier = q3+factor*iqr
    lower_outlier = q1-factor*iqr

    return (array < lower_outlier) | (array > upper_outlier)


def teste_normalidade(df:DataFrame, feature:str, num_cols:list, alpha=0.05, interativo:bool=False) -> DataFrame:
    """
    Aplica teste de normalidade para as categorias de uma coluna (feature) 
    ou geral se interativo=False.
    """
    def processar(dados, categoria):
        results = []
        for num_col in num_cols:
            data = dados[num_col].dropna().values
            n = len(data)
            if n < 3:
                results.append({
                    "Categoria": categoria,
                    "Coluna": num_col,
                    "N": n,
                    "Teste": None,
                    "Estatística": None,
                    "p-value": None,
                    "Normal?": "Amostra insuficiente"
                })
                continue
            if n < 500:
                test_name = "Shapiro-Wilk"
                stat, p = shapiro(data)
            else:
                test_name = "Kolmogorov-Smirnov"
                zscores = (data - np.mean(data)) / np.std(data, ddof=1)
                stat, p = kstest(zscores, 'norm')
            results.append({
                "Categoria": categoria,
                "Coluna": num_col,
                "N": n,
                "Teste": test_name,
                "Estatística": stat,
                "p-value": p,
                "Normal?": "Sim" if p > alpha else "Não"
            })
        return results

    if interativo:
        categorias = df[feature].dropna().unique()
        @interact(valor=categorias)
        def _interact(valor):
            dados = df[df[feature] == valor]
            res = processar(dados, valor)
            display(DataFrame(res))
    else:
        results=processar(df,'Geral')
        return DataFrame(results)


def teste_qui_quadrado(df:DataFrame, feature:str, num_cols:list, bins=10, alpha=0.05, interativo:bool=False) -> DataFrame:
    """
    Teste qui-quadrado das categorias da feature indicada 
    ou geral se interativo=False.
    """
    def processar(dados, categoria):
        results=[]
        for num_col in num_cols:
            data = dados[num_col].dropna()
            if len(data)<bins: 
                continue
            zscores = (data - data.mean())/data.std()
            obs, bin_edges = np.histogram(zscores,bins=bins)
            cdf_vals = norm.cdf(bin_edges)
            expected_probs = np.diff(cdf_vals)
            expected = expected_probs*len(zscores)
            chi2,p=chisquare(f_obs=obs,f_exp=expected)
            results.append({
                "Categoria": categoria,
                "Coluna": num_col,
                "Chi2": chi2,
                "p-value": p,
                "Normal?": "Sim" if p>alpha else "Não"
            })
        return results

    if interativo:
        categorias=df[feature].dropna().unique()
        @interact(valor=categorias)
        def _interact(valor):
            dados=df[df[feature]==valor]
            res=processar(dados,valor)
            display(DataFrame(res))
    else:
        results=[]
        for categoria in df[feature].dropna().unique():
            dados=df[df[feature]==categoria]
            results+=processar(dados,categoria)
        return DataFrame(results)


def analise_vif(df: DataFrame, feature: str, interativo: bool = False):
    """
    Realiza o teste VIF para as features numéricas.
    Se interativo=True, exibe dropdown para filtrar pela feature.
    Se não houver pelo menos 2 colunas numéricas válidas, retorna DataFrame vazio.
    """

    def vif_calculator(df_to_vif):
        """Calcula VIF silenciando prints e warnings."""
        vif_data = DataFrame()
        vif_data['Feature'] = df_to_vif.columns
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with contextlib.redirect_stdout(StringIO()):
                vif_data['VIF'] = [
                    variance_inflation_factor(df_to_vif.values, i)
                    for i in range(df_to_vif.shape[1])
                ]
        vif_data = vif_data.replace([np.inf, -np.inf], np.nan).dropna()
        return vif_data.sort_values(by="VIF", ascending=False)

    if interativo:
        categorias = sorted(df[feature].dropna().unique())

        @interact(valor=categorias)
        def executar(valor):
            df_filtrado = df[df[feature] == valor].copy()
            features_num = df_filtrado.select_dtypes(include='number').columns
            df_features = df_filtrado[features_num].dropna()
            df_features = df_features.loc[:, df_features.nunique() > 1]
            if df_features.shape[1] < 2:
                # retorna DataFrame vazio
                display(DataFrame(columns=['Feature', 'VIF']))
                return
            vif_resultado = vif_calculator(df_features)
            display(vif_resultado)

    else:
        # modo geral sem filtro
        features_num = df.select_dtypes(include='number').columns
        df_features = df[features_num].dropna()
        df_features = df_features.loc[:, df_features.nunique() > 1]
        if df_features.shape[1] < 2:
            return DataFrame(columns=['Feature', 'VIF'])
        return vif_calculator(df_features)


def teste_t_duas_amostras(df, coluna_flag, coluna_valor, alpha=0.05):
    """
    Realiza um teste t de Student para comparar a média de uma variável
    entre dois grupos definidos por uma coluna flag.

    params:
        alpha (float, opcional): O nível de significância para o teste. Padrão é 0.05.
    """
    try:
        grupo_1 = df.loc[df[coluna_flag] == 1, coluna_valor]
        grupo_0 = df.loc[df[coluna_flag] == 0, coluna_valor]
        
        logger.info(f"Grupo com '{coluna_flag}' = 1: {len(grupo_1)} observações.")
        logger.info(f"Grupo com '{coluna_flag}' = 0: {len(grupo_0)} observações.")
        
        if len(grupo_1) < 2 or len(grupo_0) < 2:
            logger.warning("Um dos grupos tem menos de 2 observações. O teste t não pode ser realizado.")
            return

        t_stat, p_value = ttest_ind(grupo_1, grupo_0, equal_var=False, nan_policy='omit')

        logger.info(f"Estatística t: {t_stat:.4f}")
        logger.info(f"P-valor: {p_value:.4f}")
        logger.info("-" * 30)

        if p_value < alpha:
            logger.info(f"O p-valor ({p_value:.4f}) é menor que {alpha}.")
            logger.info("Conclusão: Rejeitamos a hipótese nula.")
            logger.info("Existe uma diferença estatisticamente significativa no preço médio entre os grupos.")
        else:
            logger.info(f"O p-valor ({p_value:.4f}) é maior que {alpha}.")
            logger.info("Conclusão: Não rejeitamos a hipótese nula.")
            logger.info("Não há evidências suficientes para afirmar que o preço médio é diferente.")

    except Exception as e:
        logger.error(f"Ocorreu um erro inesperado: {e}", exc_info=True)


def teste_anova(df, formula, alpha=0.05):
    """
    Realiza uma análise de variância (ANOVA) com base em uma fórmula OLS.

    params:
        df (pd.DataFrame): O DataFrame a ser analisado.
        formula (str): A fórmula do modelo OLS (ex: 'price ~ C(room_type)').
        alpha (float, opcional): O nível de significância para o teste. Padrão é 0.05.
    """
    logger.info(f"Fórmula do modelo: {formula}")

    try:
        target = formula.split('~')[0].strip()
        predictors = re.findall(r'C\((.*?)\)', formula)
        
        # Verifica se pelo menos uma variável categórica foi encontrada na fórmula
        if not predictors:
            logger.error("Erro: Nenhuma variável categórica (C(nome)) foi encontrada na fórmula.")
            return

        modelo = ols(formula, data=df).fit()
        tabela_anova = anova_lm(modelo)
        
        logger.info("\n" + tabela_anova.to_string())
        
        # Conclusão do teste para cada preditor categórico
        logger.info("-" * 30)
        for predictor in predictors:
            p_valor = tabela_anova.loc[f'C({predictor})', 'PR(>F)']
            logger.info(f"P-valor para o preditor '{predictor}': {p_valor:.4f}")

            if p_valor < alpha:
                logger.info(f"O p-valor é menor que {alpha}. Rejeitamos a hipótese nula.")
                logger.info(f"Conclusão: Existe uma diferença estatisticamente significativa na média de '{target}' entre as categorias de '{predictor}'.")
            else:
                logger.info(f"O p-valor é maior que {alpha}. Não rejeitamos a hipótese nula.")
                logger.info(f"Conclusão: Não há evidências suficientes para afirmar que a média de '{target}' é diferente entre as categorias de '{predictor}'.")

    except Exception as e:
        logger.error(f"Ocorreu um erro durante a análise de ANOVA: {e}", exc_info=True)


def teste_tukey(df: DataFrame, target: str, coluna_grupo: str) -> None:
    """
    Realiza e imprime o Teste de Tukey HSD para a comparação de múltiplas médias.

    params:
        dataframe (DataFrame): O DataFrame que contém os dados.
        coluna_alvo (str): O nome da coluna numérica que será comparada (ex: 'tempo_total_pedido').
        coluna_grupo (str): O nome da coluna categórica que define os grupos.

    return:
        None: A função imprime a tabela de resultados do teste no console.
    """
    if target not in df.columns or coluna_grupo not in df.columns:
        print("Erro: Uma ou ambas as colunas não foram encontradas no df.")
        return
    
    try:
        comp = mc.MultiComparison(df[target], df[coluna_grupo])
        tabela_tukey = comp.tukeyhsd()
        
        # Imprime a tabela de resultados
        print(tabela_tukey)

    except Exception as e:
        print(f"Ocorreu um erro ao rodar o Teste de Tukey: {e}")