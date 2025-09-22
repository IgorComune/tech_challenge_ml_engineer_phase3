"""Modulo de funções geradoras de gráfico para avaliação de modelo"""
import logging
import numpy as np
from pandas import DataFrame, Series
from typing import Any,List
from sklearn.model_selection import KFold, learning_curve
import matplotlib.pyplot as plt
import shap
import graphviz

# inicializa o javascript para renderizar a imagem
shap.initjs()


# instância do objeto logger
logger = logging.getLogger(__name__)

def grafico_curva_aprendizagem(estimator: Any,X: DataFrame, y: Series, train_size: List[float], scoring: str = 'neg_mean_absolute_error',cv: Any=KFold) -> None:
    """
    Gera um gráfico da curva de aprendizado,comparando a pontuação de treino e validação. 
    
    params:
        estimator (Any): O estimador (modelo) a ser usado.
        X (DataFrame): DataFrame completo com todas as features, incluindo 'acao'.
        y (Series): Series completa com o valor alvo ('y_real').
        cv (Any): Estratégia de validação cruzada (ex: KFold).
        train_size (list[float]): Lista de frações de dados para treinamento.
        scoring (str): Métrica de pontuação.

    return:
        gráfico (None)
    """
    try:
        train_sizes, train_scores, test_scores = learning_curve(
                            estimator=estimator,X=X,y=y,cv=cv,scoring=scoring,train_sizes=train_size)

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Pontuação de Treino")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Pontuação de Validação Cruzada")
        plt.xlabel("Número de exemplos de treinamento")
        plt.ylabel(f"Pontuação ({scoring})")
        plt.title(f"Curva de Aprendizagem")
        plt.legend(loc="best")
        return plt.show()
    
    except Exception as e:
        return logger.error(e)
    
def plot_shap(model, X_df: DataFrame, n_features: int = 20):
    """
    Gera um gráfico de barras com a importância global das features.
    
    Args:
        model: O modelo LightGBM treinado.
        X_df (pd.DataFrame): O DataFrame de features (ex: X_test).
        n_features (int): O número de features a serem plotadas.
    """
    # Cria o objeto TreeExplainer
    explainer = shap.TreeExplainer(model)
    
    # Calcula os valores SHAP
    shap_values = explainer.shap_values(X_df)
    
    # Usa o summary_plot com plot_type="bar" para o gráfico de barras
    shap.summary_plot(shap_values, X_df, plot_type="bar", max_display=n_features, show=False)
    
    plt.title("Importância Global das Features (Média Absoluta dos SHAP Values)", fontsize=16)
    plt.show()
