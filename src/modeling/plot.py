"""Modulo de funções geradoras de gráfico para avaliação de modelo"""
import logging
import numpy as np
from pandas import DataFrame, Series
from typing import Any,List
from sklearn.model_selection import KFold, learning_curve
import matplotlib.pyplot as plt
import shap
import graphviz
from sklearn.tree import export_graphviz
import lightgbm as lgb

# inicializa o javascript para renderizar a imagem
shap.initjs()

# instância do objeto logger
logger = logging.getLogger(__name__)


def grafico_curva_aprendizagem(estimator: Any,X: DataFrame, y: Series, train_size: List[float], titulo:str='Curva de Aprendizagem',
                               scoring: str = 'neg_mean_absolute_error',cv: Any=KFold, path:str=None, ax=None) -> None:
    """
    Gera um gráfico da curva de aprendizado,comparando a pontuação de treino e validação. 
    
    params:
        estimator (Any): O estimador (modelo) a ser usado.
        X (DataFrame): DataFrame completo com todas as features, incluindo 'acao'.
        y (Series): Series completa com o valor alvo ('y_real').
        cv (Any): Estratégia de validação cruzada (ex: KFold).
        train_size (list[float]): Lista de frações de dados para treinamento.
        scoring (str): Métrica de pontuação.
        path (str): caminho para salvamento da imagem.

    return:
        gráfico (None)
    """
    
    was_called_alone = ax is None
    
    try:
        if was_called_alone:
            fig, ax = plt.subplots(figsize=(10, 6))
        
        train_sizes, train_scores, test_scores = learning_curve(
                                estimator=estimator,X=X,y=y,cv=cv,scoring=scoring,train_sizes=train_size)

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        ax.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Pontuação de Treino")
        ax.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Pontuação de Validação Cruzada")
        
        ax.set_xlabel("Número de exemplos de treinamento")
        ax.set_ylabel(f"Pontuação ({scoring})")
        ax.set_title(f"{titulo}")
        
        ax.legend(loc="best")
        
        if was_called_alone:
            fig.tight_layout()
            
            if path:
                fig.savefig(path)
            
            plt.show()
            plt.close(fig)
        
    except Exception as e:
        logger.error(f"Erro ao gerar a curva de aprendizado: {e}")
        return
    
def gerar_grafico_shap(model, X_df: DataFrame, n_features: int = 20, kind:str='bar', path:str=None) -> None:
    """
    Gera um gráfico de barras com a importância global das features.
    
    params:
        model: O modelo LightGBM treinado.
        X_df (pd.DataFrame): O DataFrame de features (ex: X_test).
        n_features (int): O número de features a serem plotadas.
    """
    try:
        if kind =='bar':
            explainer = shap.Explainer(model)
            shap_values = explainer.shap_values(X_df)
            shap.summary_plot(shap_values, X_df, plot_type=kind, max_display=n_features, show=False)

        elif kind =='beeswarm':
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_df)
            shap.summary_plot(shap_values, X_df, max_display=n_features, plot_type="dot", show=False)

    except Exception as e:
        raise logger.error(e)
    
    if path:
        try:
            plt.gcf().tight_layout()
            plt.savefig(path, dpi=300, bbox_inches='tight')
        except Exception as e:
            raise logger.error(e)

    plt.title("Importância Global das Features (Média Absoluta dos SHAP Values)", fontsize=16)
    plt.show()



def gerar_arvore(modelo: Any, feature_list: list, model_type: str = 'DecisionTree', filename: str = None) -> graphviz.Source:
    """
    Função que gera um gráfico de árvore de decisão para modelos Decision Tree, 
    Random Forest ou LightGBM, usando create_tree_digraph() para LightGBM.
    """
    
    model_type_lower = model_type.lower()
    
    # --- 1. Lógica para LightGBM (Usando create_tree_digraph) ---
    if model_type_lower == 'lightgbm':
        logger.info("Plotando a primeira árvore (índice 0) do modelo LightGBM.")

        # Obtém o booster (o modelo LightGBM puro)
        booster = modelo.booster_ 
        
        # PASSO CRUCIAL: Força a atualização do nome das features no booster
        # Isso corrige a ausência de nomes ('Column_X')
        # E permite que a próxima chamada funcione sem o argumento 'feature_names'
        booster.feature_name = feature_list 

        # Chama a função create_tree_digraph (preferida pela documentação)
        # NENHUM 'feature_names' AQUI para evitar o TypeError de argumento duplicado
        graph = lgb.create_tree_digraph(
            booster=booster, 
            tree_index=0,
            orientation='vertical'
        )
        
        # Renderiza e salva
        if filename:
            graph.render(filename=filename, format='png', cleanup=True)
            
        return graph 
        
    
    # --- 2. Lógica para Modelos Scikit-learn (Decision Tree e Random Forest) ---
    # ... (Seu código original para Random Forest e Decision Tree, que usa export_graphviz) ...
    
    arvore_para_plotar = None
    
    if model_type_lower == 'randomforest':
        # Seleciona a primeira árvore do ensemble
        arvore_para_plotar = modelo.estimators_[0]
        logger.info("Plotando a primeira árvore (índice 0) do modelo Random Forest.")
        
    elif model_type_lower == 'decisiontree':
        # Usa o modelo DT diretamente
        arvore_para_plotar = modelo
        
    else:
        logger.warning(f"model_type '{model_type}' não reconhecido. Falha na plotagem.")
        return graphviz.Source("") 

    # --- 3. Executa a Exportação do Scikit-learn ---
    
    data = export_graphviz(
        arvore_para_plotar, 
        out_file=None, 
        filled=True, 
        rounded=True, 
        class_names=None, # Mantemos None para Regressão
        feature_names=feature_list
    )
    
    graph = graphviz.Source(data)

    if filename:
        graph.render(filename=filename, format='png', cleanup=True) 
    
    return graph